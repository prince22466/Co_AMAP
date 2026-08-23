"""Pinned-public-engine round-robin evaluator for Kaggriculture agents.

The arena extracts the exact ``main.py`` source embedded in submission
notebooks, gives every agent a unique file-loader path, and runs each pairing
from both seats for every seed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
import tarfile
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
SUBMISSION_DIR = WORKSPACE / "submission_nb"
EXPECTED_ENGINE_VERSION = "1.32.7"
EXPECTED_DEFAULT_CONFIGURATION = {
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
    "marketParams": {},
}
DEFAULT_AGENTS = {
    path.stem.removeprefix("kaggriculture-sub_"): path
    for path in sorted(SUBMISSION_DIR.glob("kaggriculture-sub_v*.ipynb"))
}


@dataclass(frozen=True)
class AgentSource:
    name: str
    path: str
    sha256: str
    bytes: int
    source: str


def _read_notebook_main(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        lines = source.splitlines()
        if not lines:
            continue
        magic = re.match(r"^\s*%%writefile\s+(.+?)\s*$", lines[0])
        if magic and Path(magic.group(1).strip("'\"")).name == "main.py":
            matches.append((index, "\n".join(lines[1:]) + "\n"))
    if len(matches) != 1:
        indexes = [index for index, _ in matches]
        raise ValueError(
            f"{path}: expected exactly one %%writefile main.py cell; found {indexes}"
        )
    return matches[0][1]


def _read_archive_main(path: Path) -> str:
    with tarfile.open(path, "r:*") as archive:
        regular = [member for member in archive.getmembers() if member.isfile()]
        roots = [member for member in regular if member.name.replace("\\", "/") == "main.py"]
        if len(roots) != 1:
            raise ValueError(f"{path}: archive must contain one root-level main.py")
        extras = [member.name for member in regular if member is not roots[0]]
        if extras:
            raise ValueError(
                f"{path}: local arena supports single-file submissions only; "
                f"extra archive files: {extras[:5]}"
            )
        handle = archive.extractfile(roots[0])
        if handle is None:
            raise ValueError(f"{path}: unable to read main.py")
        return handle.read().decode("utf-8")


def read_agent_source(path: Path) -> str:
    path = path.resolve()
    lowered = path.name.lower()
    if lowered.endswith(".ipynb"):
        return _read_notebook_main(path)
    if lowered.endswith(".py"):
        return path.read_text(encoding="utf-8")
    if lowered.endswith((".tar.gz", ".tgz", ".tar")):
        return _read_archive_main(path)
    if path.is_dir():
        main_path = path / "main.py"
        if main_path.is_file():
            extras = [
                str(child.relative_to(path))
                for child in path.rglob("*")
                if child.is_file() and child != main_path
            ]
            if extras:
                raise ValueError(
                    f"{path}: local arena supports single-file submissions only; "
                    f"extra directory files: {extras[:5]}"
                )
            return main_path.read_text(encoding="utf-8")
    raise ValueError(f"{path}: use a .ipynb, .py, archive, or directory containing main.py")


def validate_source(name: str, source: str, path: Path) -> AgentSource:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValueError(f"{name}: invalid Python in {path}: {exc}") from exc
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "agent" not in functions:
        raise ValueError(f"{name}: {path} does not define top-level agent(obs)")
    payload = source.encode("utf-8")
    return AgentSource(
        name=name,
        path=str(path.resolve()),
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        source=source,
    )


def load_agent(name: str, path: Path) -> AgentSource:
    if not path.exists():
        raise ValueError(f"{name}: source not found: {path}")
    return validate_source(name, read_agent_source(path), path)


def _safe_filename(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return value or "agent"


def materialize_agents(agents: Iterable[AgentSource], directory: Path) -> dict[str, str]:
    paths = {}
    for agent in agents:
        # Kaggle loads a root-level main.py. A unique parent keeps modules isolated
        # while preserving that basename and relative-path anchor for each agent.
        agent_dir = directory / f"{_safe_filename(agent.name)}_{agent.sha256[:10]}"
        agent_dir.mkdir()
        path = agent_dir / "main.py"
        path.write_text(agent.source, encoding="utf-8")
        paths[agent.name] = str(path.resolve())
    return paths


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _run_game(job: dict) -> dict:
    """Worker entrypoint. Importing Kaggle here keeps --validate-only lightweight."""
    from kaggle_environments import make

    started = time.perf_counter()
    record = {
        "pair_a": job["pair_a"],
        "pair_b": job["pair_b"],
        "seed": job["seed"],
        "resolved_seed": None,
        "orientation": job["orientation"],
        "seat0": job["seat0"],
        "seat1": job["seat1"],
        "status0": "RUNNER_ERROR",
        "status1": "RUNNER_ERROR",
        "reward0": None,
        "reward1": None,
        "winner": "ERROR",
        "margin0": None,
        "runtime_seconds": None,
        "error": "",
    }
    try:
        env = make(
            "kaggriculture",
            configuration={
                "episodeSteps": job["episode_steps"],
                "seed": job["seed"],
            },
            debug=False,
        )
        record["resolved_seed"] = _get(env.info, "seed", None)
        if record["resolved_seed"] != job["seed"]:
            raise RuntimeError(
                "engine did not preserve requested seed: "
                f"requested={job['seed']} resolved={record['resolved_seed']}"
            )
        env.run([job["path0"], job["path1"]])
        final = env.steps[-1]
        statuses = [str(_get(state, "status", "")) for state in final]
        rewards = [_number(_get(state, "reward", None)) for state in final]
        record.update(
            status0=statuses[0],
            status1=statuses[1],
            reward0=rewards[0],
            reward1=rewards[1],
        )
        if rewards[0] is None or rewards[1] is None:
            record["winner"] = "ERROR"
        elif rewards[0] > rewards[1]:
            record["winner"] = job["seat0"]
            record["margin0"] = rewards[0] - rewards[1]
        elif rewards[1] > rewards[0]:
            record["winner"] = job["seat1"]
            record["margin0"] = rewards[0] - rewards[1]
        else:
            record["winner"] = "TIE"
            record["margin0"] = 0.0
    except Exception as exc:  # Preserve the rest of a long arena run.
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
    record["runtime_seconds"] = round(time.perf_counter() - started, 4)
    return record


def build_jobs(agent_paths, seeds, episode_steps, self_play=False):
    names = sorted(agent_paths)
    pairs = list(itertools.combinations(names, 2))
    if self_play:
        pairs.extend((name, name) for name in names)
    jobs = []
    for pair_a, pair_b in pairs:
        for seed in seeds:
            orientations = [(pair_a, pair_b, "A0_B1")]
            if pair_a != pair_b:
                orientations.append((pair_b, pair_a, "B0_A1"))
            for seat0, seat1, orientation in orientations:
                jobs.append(
                    {
                        "pair_a": pair_a,
                        "pair_b": pair_b,
                        "seed": seed,
                        "orientation": orientation,
                        "seat0": seat0,
                        "seat1": seat1,
                        "path0": agent_paths[seat0],
                        "path1": agent_paths[seat1],
                        "episode_steps": episode_steps,
                    }
                )
    return jobs


def _done(record):
    return (
        record["status0"] == "DONE"
        and record["status1"] == "DONE"
        and record["reward0"] is not None
        and record["reward1"] is not None
    )


def _agent_view(record, name):
    if record["seat0"] == name:
        reward, opponent_reward, seat = record["reward0"], record["reward1"], 0
        status = record["status0"]
    elif record["seat1"] == name:
        reward, opponent_reward, seat = record["reward1"], record["reward0"], 1
        status = record["status1"]
    else:
        raise ValueError(f"{name} is not in record")
    margin = None if reward is None or opponent_reward is None else reward - opponent_reward
    return reward, opponent_reward, margin, seat, status


def _agent_completed(record, name):
    reward, _, _, _, status = _agent_view(record, name)
    return status == "DONE" and reward is not None


def _agent_outcome(record, name):
    """Return win/tie/loss, treating an agent-side failure as a forfeit."""
    reward, opponent_reward, _, seat, _ = _agent_view(record, name)
    opponent_name = record["seat1"] if seat == 0 else record["seat0"]
    own_ok = _agent_completed(record, name)
    opponent_ok = _agent_completed(record, opponent_name)
    if own_ok and not opponent_ok:
        return "win"
    if not own_ok:
        return "loss"
    if reward > opponent_reward:
        return "win"
    if reward < opponent_reward:
        return "loss"
    return "tie"


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _percentile(values, fraction):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    index = max(0, min(len(values) - 1, math.floor((len(values) - 1) * fraction)))
    return values[index]


def make_leaderboard(records, agent_names):
    rows = []
    for name in sorted(agent_names):
        # Self-play is a loader validation probe, not evidence for ranking.
        relevant = [
            record
            for record in records
            if record["pair_a"] != record["pair_b"]
            and (record["seat0"] == name or record["seat1"] == name)
        ]
        views = [_agent_view(record, name) for record in relevant]
        completed = [view for record, view in zip(relevant, views) if _done(record)]
        outcomes = [_agent_outcome(record, name) for record in relevant]
        margins = [view[2] for view in completed]
        rewards = [view[0] for view in completed]
        opponent_rewards = [view[1] for view in completed]
        wins = outcomes.count("win")
        ties = outcomes.count("tie")
        losses = outcomes.count("loss")
        points = wins + 0.5 * ties
        agent_errors = sum(not _agent_completed(record, name) for record in relevant)
        opponent_errors = sum(
            not _agent_completed(
                record,
                record["seat1"] if record["seat0"] == name else record["seat0"],
            )
            for record in relevant
        )
        seat0_rewards = [view[0] for view in completed if view[3] == 0]
        seat1_rewards = [view[0] for view in completed if view[3] == 1]
        rows.append(
            {
                "agent": name,
                "games": len(relevant),
                "completed": len(completed),
                "errors": agent_errors,
                "opponent_errors": opponent_errors,
                "incomplete_games": len(relevant) - len(completed),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "points": points,
                "points_rate": points / len(relevant) if relevant else None,
                "mean_reward": _mean(rewards),
                "mean_opponent_reward": _mean(opponent_rewards),
                "mean_margin": _mean(margins),
                "median_margin": _median(margins),
                "p10_margin": _percentile(margins, 0.10),
                "worst_margin": min(margins) if margins else None,
                "mean_reward_seat0": _mean(seat0_rewards),
                "mean_reward_seat1": _mean(seat1_rewards),
            }
        )
    rows.sort(
        key=lambda row: (
            row["errors"],
            -(row["points_rate"] if row["points_rate"] is not None else -1),
            row["agent"],
        )
    )
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def make_pairings(records):
    grouped = {}
    for record in records:
        if record["pair_a"] == record["pair_b"]:
            continue
        grouped.setdefault((record["pair_a"], record["pair_b"]), []).append(record)
    rows = []
    for (agent_a, agent_b), games in sorted(grouped.items()):
        completed = [game for game in games if _done(game)]
        outcomes = [_agent_outcome(game, agent_a) for game in games]
        a_margins = [_agent_view(game, agent_a)[2] for game in completed]
        a_rewards = [_agent_view(game, agent_a)[0] for game in completed]
        b_rewards = [_agent_view(game, agent_b)[0] for game in completed]
        paired_seed_margins = []
        for seed in sorted({game["seed"] for game in completed}):
            seed_margins = [
                _agent_view(game, agent_a)[2]
                for game in completed
                if game["seed"] == seed
            ]
            if len(seed_margins) == 2:
                paired_seed_margins.append(statistics.mean(seed_margins))
        rows.append(
            {
                "agent_a": agent_a,
                "agent_b": agent_b,
                "games": len(games),
                "completed": len(completed),
                "errors": len(games) - len(completed),
                "a_errors": sum(not _agent_completed(game, agent_a) for game in games),
                "b_errors": sum(not _agent_completed(game, agent_b) for game in games),
                "a_wins": outcomes.count("win"),
                "ties": outcomes.count("tie"),
                "b_wins": outcomes.count("loss"),
                "mean_a_reward": _mean(a_rewards),
                "mean_b_reward": _mean(b_rewards),
                "mean_a_margin": _mean(a_margins),
                "mean_paired_seed_margin": _mean(paired_seed_margins),
                "worst_paired_seed_margin": min(paired_seed_margins) if paired_seed_margins else None,
            }
        )
    return rows


def _write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_seeds(value):
    try:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _parse_named_path(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("custom agents use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("agent name cannot be empty")
    return name, Path(raw_path.strip())


def _collect_agents(args):
    requested = {} if args.no_defaults else dict(DEFAULT_AGENTS)
    if args.candidate:
        if args.candidate_name in requested:
            raise ValueError(f"duplicate agent name: {args.candidate_name}")
        requested[args.candidate_name] = args.candidate
    for name, path in args.agent:
        if name in requested:
            raise ValueError(f"duplicate agent name: {name}")
        requested[name] = path
    if len(requested) < 2:
        raise ValueError("round robin requires at least two agents")
    return [load_agent(name, path) for name, path in sorted(requested.items())]


def _dependency_version():
    try:
        import kaggle_environments
    except ImportError as exc:
        raise RuntimeError(
            "kaggle-environments is not installed. Run: "
            "python -m pip install -r local_arena/requirements.txt"
        ) from exc
    return getattr(kaggle_environments, "__version__", "unknown")


def _audit_engine_configuration(episode_steps):
    from kaggle_environments import make

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": episode_steps, "seed": 0},
        debug=False,
    )
    configuration = dict(env.configuration)
    resolved_seed = _get(env.info, "seed", None)
    mismatches = {
        key: {"expected": expected, "actual": configuration.get(key)}
        for key, expected in EXPECTED_DEFAULT_CONFIGURATION.items()
        if configuration.get(key) != expected
    }
    if resolved_seed != 0:
        mismatches["seed_resolution"] = {"expected": 0, "actual": resolved_seed}
    return configuration, mismatches, resolved_seed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, help="candidate .ipynb, .py, archive, or directory")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--agent", action="append", type=_parse_named_path, default=[], help="add NAME=PATH")
    parser.add_argument("--no-defaults", action="store_true", help="omit automatically discovered versions")
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("2601,2602"))
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--self-play", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="extract and statically validate; do not run")
    parser.add_argument(
        "--allow-unverified-engine",
        action="store_true",
        help="run despite a package-version or default-configuration mismatch",
    )
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args(argv)

    if args.episode_steps <= 0:
        parser.error("--episode-steps must be positive")
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    try:
        agents = _collect_agents(args)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        parser.error(str(exc))

    print("Validated agent sources:")
    for agent in agents:
        print(f"  {agent.name:16s} {agent.bytes:7d} bytes  sha256={agent.sha256[:12]}  {agent.path}")
    if args.validate_only:
        print(f"Static validation OK ({len(agents)} agents); engine was not invoked.")
        return 0

    try:
        engine_version = _dependency_version()
    except RuntimeError as exc:
        parser.error(str(exc))
    try:
        (
            engine_configuration,
            configuration_mismatches,
            seed_probe_resolved,
        ) = _audit_engine_configuration(args.episode_steps)
    except Exception as exc:
        parser.error(f"unable to audit kaggriculture configuration: {exc}")
    verification_issues = {}
    if engine_version != EXPECTED_ENGINE_VERSION:
        verification_issues["engine_version"] = {
            "expected": EXPECTED_ENGINE_VERSION,
            "actual": engine_version,
        }
    if configuration_mismatches:
        verification_issues["configuration"] = configuration_mismatches
    if verification_issues and not args.allow_unverified_engine:
        parser.error(
            "local engine does not match the audited public baseline: "
            f"{json.dumps(verification_issues, sort_keys=True)}; use "
            "--allow-unverified-engine only for deliberate diagnostics"
        )
    if args.episode_steps != 720:
        print(
            "WARNING: episodeSteps is not 720; this is a smoke test, not an "
            "official-length comparison."
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kaggriculture_arena_") as temporary:
        paths = materialize_agents(agents, Path(temporary))
        jobs = build_jobs(paths, args.seeds, args.episode_steps, args.self_play)
        print(
            f"Running {len(jobs)} games with kaggle-environments {engine_version} "
            f"using {args.jobs} worker(s)..."
        )
        if args.jobs == 1:
            records = []
            for index, job in enumerate(jobs, 1):
                record = _run_game(job)
                records.append(record)
                print(
                    f"[{index:4d}/{len(jobs)}] {record['seat0']} vs {record['seat1']} "
                    f"seed={record['seed']} rewards={record['reward0']},{record['reward1']}"
                )
                if args.fail_fast and not _done(record):
                    raise RuntimeError(record["error"] or f"non-DONE status: {record}")
        else:
            records = []
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {executor.submit(_run_game, job): job for job in jobs}
                for index, future in enumerate(as_completed(futures), 1):
                    record = future.result()
                    records.append(record)
                    print(
                        f"[{index:4d}/{len(jobs)}] {record['seat0']} vs {record['seat1']} "
                        f"seed={record['seed']} rewards={record['reward0']},{record['reward1']}"
                    )
                    if args.fail_fast and not _done(record):
                        for pending in futures:
                            pending.cancel()
                        raise RuntimeError(record["error"] or f"non-DONE status: {record}")

    records.sort(key=lambda row: (row["pair_a"], row["pair_b"], row["seed"], row["orientation"]))
    leaderboard = make_leaderboard(records, [agent.name for agent in agents])
    pairings = make_pairings(records)
    _write_csv(output / "matches.csv", records)
    _write_csv(output / "pairings.csv", pairings)
    _write_csv(output / "leaderboard.csv", leaderboard)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "kaggle-environments",
        "engine_version": engine_version,
        "expected_engine_version": EXPECTED_ENGINE_VERSION,
        "engine_verified": not verification_issues,
        "engine_verification_scope": "pinned public package and audited default configuration",
        "production_engine_identity_confirmed": False,
        "engine_verification_issues": verification_issues,
        "environment": "kaggriculture",
        "episode_steps": args.episode_steps,
        "official_episode_length": args.episode_steps == 720,
        "configuration": engine_configuration,
        "seed_probe": {"requested": 0, "resolved": seed_probe_resolved},
        "seeds": args.seeds,
        "both_seats": True,
        "self_play": args.self_play,
        "failure_policy": "agent-side failures count as forfeits; zero errors rank first",
        "agents": [
            {key: value for key, value in asdict(agent).items() if key != "source"}
            for agent in agents
        ],
    }
    (output / "results.json").write_text(
        json.dumps(
            {"metadata": metadata, "leaderboard": leaderboard, "pairings": pairings, "matches": records},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nLeaderboard:")
    for row in leaderboard:
        points_rate = "n/a" if row["points_rate"] is None else f"{row['points_rate']:.1%}"
        mean_margin = "n/a" if row["mean_margin"] is None else f"{row['mean_margin']:.1f}"
        print(
            f"{row['rank']:2d}. {row['agent']:16s} "
            f"points={points_rate} "
            f"mean_margin={mean_margin} "
            f"errors={row['errors']}"
        )
    print(f"\nWrote matches.csv, pairings.csv, leaderboard.csv, and results.json to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
