"""Deterministic local analysis for v23 histories and experiment evidence."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

INTEREST_TERMS = (
    "money", "cash", "bank", "balance", "price", "inventory", "capacity",
    "wheat", "fertil", "egg", "milk", "wool", "strawberry", "melon",
    "worker", "hand", "animal", "crop", "land", "shed", "reward",
)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _numeric_scalars(value: Any, prefix: str = "", depth: int = 0, max_depth: int = 6) -> dict[str, float]:
    if depth > max_depth:
        return {}
    out: dict[str, float] = {}
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out[prefix or "$"] = float(value)
        return out
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(_numeric_scalars(child, path, depth + 1, max_depth))
    elif isinstance(value, list):
        # Lists of entities can be huge. Aggregate numeric leaf values by path
        # instead of emitting one path per entity index.
        bucket: dict[str, list[float]] = defaultdict(list)
        for child in value[:100]:
            child_map = _numeric_scalars(child, "", depth + 1, max_depth)
            for child_path, num in child_map.items():
                bucket[child_path].append(num)
        for child_path, nums in bucket.items():
            path = f"{prefix}[*].{child_path}" if prefix else f"[*].{child_path}"
            out[path] = float(sum(nums))
    return out


def _interesting_paths(scalars: dict[str, float], limit: int = 80) -> dict[str, float]:
    preferred = {
        path: value
        for path, value in scalars.items()
        if any(term in path.lower() for term in INTEREST_TERMS)
    }
    if len(preferred) >= limit:
        return dict(sorted(preferred.items())[:limit])
    for path, value in scalars.items():
        if path not in preferred:
            preferred[path] = value
        if len(preferred) >= limit:
            break
    return dict(sorted(preferred.items()))


def _action_labels(action: Any) -> list[str]:
    labels: list[str] = []
    if action is None:
        return labels
    if isinstance(action, str):
        return [action]
    if isinstance(action, dict):
        for key, value in action.items():
            if isinstance(value, list):
                if not value:
                    labels.append(f"{key}:EMPTY")
                for item in value[:50]:
                    if isinstance(item, str):
                        labels.append(f"{key}:{item}")
                    elif isinstance(item, dict):
                        if item:
                            first = next(iter(item))
                            labels.append(f"{key}:{first}")
                        else:
                            labels.append(f"{key}:{{}}")
                    elif isinstance(item, (list, tuple)) and item:
                        labels.append(f"{key}:{item[0]}")
                    else:
                        labels.append(f"{key}:{type(item).__name__}")
            elif isinstance(value, str):
                labels.append(f"{key}:{value}")
            elif value is not None:
                labels.append(str(key))
        return labels
    if isinstance(action, list):
        for item in action[:50]:
            labels.extend(_action_labels(item))
        return labels
    return [type(action).__name__]


def _state_snapshot(state: Any) -> dict[str, Any]:
    obs = _plain(_field(state, "observation", {}) or {})
    scalars = _interesting_paths(_numeric_scalars(obs))
    reward = _field(state, "reward", None)
    return {
        "reward": float(reward) if isinstance(reward, (int, float)) else None,
        "status": str(_field(state, "status", "")),
        "action": _plain(_field(state, "action", None)),
        "action_labels": _action_labels(_field(state, "action", None)),
        "scalars": scalars,
    }


def _deltas(previous: dict[str, float], current: dict[str, float], limit: int = 24) -> list[dict[str, float | str]]:
    rows = []
    for path in set(previous) | set(current):
        before = previous.get(path)
        after = current.get(path)
        if before is None or after is None:
            continue
        delta = after - before
        if delta:
            rows.append({
                "path": path,
                "before": before,
                "after": after,
                "delta": delta,
            })
    rows.sort(key=lambda row: abs(float(row["delta"])), reverse=True)
    return rows[:limit]


def load_loss_history(root: Path, episode: str) -> tuple[Path, dict[str, Any]]:
    episode = str(episode).strip()
    if not episode or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in episode):
        raise ValueError("episode must be a simple history stem")
    path = root / "working_files" / "loss_games_v20" / f"{episode}.json"
    path = path.resolve()
    expected = (root / "working_files" / "loss_games_v20").resolve()
    path.relative_to(expected)
    if not path.is_file():
        raise FileNotFoundError(f"history not found: {episode}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def analyze_loss_history(root: Path, episode: str, window_size: int = 24, top_windows: int = 8) -> dict[str, Any]:
    path, history = load_loss_history(root, episode)
    steps = history.get("steps") or []
    if not steps:
        raise ValueError("history has no steps")
    if not isinstance(steps[-1], list) or len(steps[-1]) != 2:
        raise ValueError("expected two players")

    final_rewards = []
    for state in steps[-1]:
        reward = _field(state, "reward", None)
        if not isinstance(reward, (int, float)):
            raise ValueError("terminal reward missing")
        final_rewards.append(float(reward))
    if final_rewards[0] == final_rewards[1]:
        loser_seat = 0
    else:
        loser_seat = 0 if final_rewards[0] < final_rewards[1] else 1
    opponent_seat = 1 - loser_seat

    action_counts = [Counter(), Counter()]
    turns: list[dict[str, Any]] = []
    prev = [None, None]
    all_paths = [Counter(), Counter()]

    for turn, pair in enumerate(steps):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        snaps = [_state_snapshot(pair[0]), _state_snapshot(pair[1])]
        for seat in (0, 1):
            action_counts[seat].update(snaps[seat]["action_labels"])
            all_paths[seat].update(snaps[seat]["scalars"].keys())

        if turn > 0:
            loser_delta = _deltas(
                prev[loser_seat]["scalars"], snaps[loser_seat]["scalars"]
            )
            opp_delta = _deltas(
                prev[opponent_seat]["scalars"], snaps[opponent_seat]["scalars"]
            )
            activity_score = sum(abs(float(x["delta"])) for x in loser_delta[:8])
            activity_score += sum(abs(float(x["delta"])) for x in opp_delta[:8])
        else:
            loser_delta, opp_delta, activity_score = [], [], 0.0

        turns.append({
            "turn": turn,
            "v20_action_labels": snaps[loser_seat]["action_labels"],
            "opponent_action_labels": snaps[opponent_seat]["action_labels"],
            "v20_top_deltas": loser_delta,
            "opponent_top_deltas": opp_delta,
            "activity_score": activity_score,
        })
        prev = snaps

    window_size = max(1, min(int(window_size), max(1, len(turns))))
    top_windows = max(1, min(int(top_windows), 20))
    windows = []
    for start in range(0, len(turns), window_size):
        chunk = turns[start:start + window_size]
        score = sum(float(row["activity_score"]) for row in chunk)
        windows.append({
            "start_turn": start,
            "end_turn": start + len(chunk) - 1,
            "activity_score": score,
            "v20_actions": Counter(
                label for row in chunk for label in row["v20_action_labels"]
            ).most_common(12),
            "opponent_actions": Counter(
                label for row in chunk for label in row["opponent_action_labels"]
            ).most_common(12),
        })
    critical = sorted(windows, key=lambda row: row["activity_score"], reverse=True)[:top_windows]

    return {
        "episode": episode,
        "source": str(path.relative_to(root)),
        "turns": len(steps),
        "v20_seat": loser_seat,
        "opponent_seat": opponent_seat,
        "final_rewards": final_rewards,
        "v20_final_margin": final_rewards[loser_seat] - final_rewards[opponent_seat],
        "v20_action_counts": action_counts[loser_seat].most_common(40),
        "opponent_action_counts": action_counts[opponent_seat].most_common(40),
        "frequent_v20_numeric_paths": all_paths[loser_seat].most_common(50),
        "critical_windows": critical,
        "interpretation_note": (
            "Critical windows rank state-change activity, not causal contribution. "
            "Use analyze_loss_window for detail and static replay for causal claims."
        ),
    }


def analyze_loss_window(root: Path, episode: str, start_turn: int, end_turn: int) -> dict[str, Any]:
    _, history = load_loss_history(root, episode)
    steps = history.get("steps") or []
    if not steps:
        raise ValueError("history has no steps")
    final_rewards = [float(_field(state, "reward", 0.0) or 0.0) for state in steps[-1]]
    loser = 0 if final_rewards[0] <= final_rewards[1] else 1
    opponent = 1 - loser
    start = max(0, int(start_turn))
    end = min(len(steps) - 1, int(end_turn))
    if end < start:
        raise ValueError("end_turn must be >= start_turn")

    rows = []
    prev = None
    for turn in range(start, end + 1):
        pair = steps[turn]
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        snaps = [_state_snapshot(pair[0]), _state_snapshot(pair[1])]
        item = {
            "turn": turn,
            "v20_action": snaps[loser]["action"],
            "opponent_action": snaps[opponent]["action"],
            "v20_action_labels": snaps[loser]["action_labels"],
            "opponent_action_labels": snaps[opponent]["action_labels"],
            "v20_scalars": snaps[loser]["scalars"],
            "opponent_scalars": snaps[opponent]["scalars"],
        }
        if prev is not None:
            item["v20_deltas"] = _deltas(prev[loser]["scalars"], snaps[loser]["scalars"], 40)
            item["opponent_deltas"] = _deltas(prev[opponent]["scalars"], snaps[opponent]["scalars"], 40)
        rows.append(item)
        prev = snaps

    return {
        "episode": episode,
        "start_turn": start,
        "end_turn": end,
        "v20_seat": loser,
        "final_margin": final_rewards[loser] - final_rewards[opponent],
        "rows": rows,
    }


def analyze_experiment_records(db: Any, review_id: str | None = None) -> dict[str, Any]:
    dossiers = db.batch_dossiers(review_id)
    if not dossiers:
        recent = [dict(row) for row in db.db.execute(
            """SELECT experiment_id,idea_id,hypothesis,status,candidate,
                      candidate_sha256,replay_cases,wins,losses,regressions,
                      mean_margin_improvement,best_margin_improvement,conclusion
               FROM experiments ORDER BY started_at DESC LIMIT 30"""
        ).fetchall()]
        return {"scope":"recent_experiments","experiments":recent}

    by_layer: dict[str, dict[str, float]] = {}
    by_components: dict[str, dict[str, float]] = {}
    ideas = []
    for dossier in dossiers:
        idea = dossier["idea"]
        experiments = dossier["experiments"]
        exp = experiments[-1] if experiments else {}
        games = _latest_games_by_episode(dossier, valid_only=True)
        improvements = [
            float(game["margin_improvement"])
            for game in games
            if isinstance(game.get("margin_improvement"), (int, float))
        ]
        wins = sum(1 for game in games if game.get("result") == "WIN")
        losses = sum(1 for game in games if game.get("result") == "LOSS")
        cases = len(games)
        mean = sum(improvements) / len(improvements) if improvements else 0.0
        layer = str(idea.get("causal_layer") or "other")
        components = sorted(idea.get("components") or [])
        comp_key = " + ".join(components) if components else "(unspecified)"
        for key, table in ((layer, by_layer), (comp_key, by_components)):
            row = table.setdefault(key, {
                "ideas":0, "replay_cases":0, "wins":0, "losses":0,
                "mean_margin_improvement_sum":0.0,
            })
            row["ideas"] += 1
            row["replay_cases"] += cases
            row["wins"] += wins
            row["losses"] += losses
            row["mean_margin_improvement_sum"] += mean
        ideas.append({
            "idea_id":idea["idea_id"],
            "batch_index":idea["batch_index"],
            "title":idea["title"],
            "causal_layer":layer,
            "components":components,
            "interaction_hypothesis":idea.get("interaction_hypothesis"),
            "system_prediction":idea.get("system_prediction"),
            "status":idea.get("status"),
            "experiment_id":exp.get("experiment_id"),
            "candidate":exp.get("candidate"),
            "candidate_sha256":exp.get("candidate_sha256"),
            "wins":wins,
            "losses":losses,
            "replay_cases":cases,
            "mean_margin_improvement":mean,
            "best_margin_improvement":max(improvements) if improvements else None,
            "regressions":sum(1 for x in improvements if x < 0),
            "game_record_paths":[
                game["game_record_path"]
                for call in dossier["replay_calls"]
                for game in call["games"]
                if game.get("game_record_path")
            ],
        })

    def finish(table: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
        out = []
        for key, row in table.items():
            n = max(1, int(row["ideas"]))
            out.append({
                "group":key,
                "ideas":int(row["ideas"]),
                "replay_cases":int(row["replay_cases"]),
                "wins":int(row["wins"]),
                "losses":int(row["losses"]),
                "mean_of_experiment_mean_margin_improvement":(
                    float(row["mean_margin_improvement_sum"]) / n
                ),
            })
        return sorted(
            out,
            key=lambda x: (
                -x["wins"],
                -x["mean_of_experiment_mean_margin_improvement"],
                x["group"],
            ),
        )

    return {
        "scope":"idea_batch",
        "review_id":dossiers[0]["idea"].get("review_id"),
        "ideas":ideas,
        "by_causal_layer":finish(by_layer),
        "by_component_combination":finish(by_components),
        "note":(
            "Aggregates are descriptive evidence only. Correlation between a component "
            "and outcome does not establish causality; use controlled replay experiments."
        ),
    }


def _find_scalar_paths_by_terms(scalars: dict[str, float], terms: tuple[str, ...]) -> dict[str, float]:
    return {
        path: value for path, value in scalars.items()
        if any(term in path.lower() for term in terms)
    }


def _load_game_record(root: Path, rel_path: str) -> dict[str, Any]:
    path = (root / rel_path).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"game record not found: {rel_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_games_by_episode(dossier: dict[str, Any], valid_only: bool = False) -> list[dict[str, Any]]:
    """Return one canonical replay row per episode, preferring the latest call."""
    latest: dict[str, dict[str, Any]] = {}
    for call in dossier.get("replay_calls", []):
        for game in call.get("games", []):
            episode = str(game.get("episode") or "")
            if not episode:
                continue
            row = dict(game)
            row["replay_call_id"] = call.get("replay_call_id")
            row["candidate"] = call.get("candidate")
            latest[episode] = row
    rows = list(latest.values())
    if valid_only:
        rows = [row for row in rows if bool(row.get("valid"))]
    return rows


def compare_candidate_to_v20(root: Path, db: Any, idea_id: str, episode: str) -> dict[str, Any]:
    dossier = db.idea_dossier(idea_id)
    if dossier is None:
        raise ValueError("unknown idea_id: " + idea_id)
    matches = [
        game for game in _latest_games_by_episode(dossier)
        if str(game.get("episode")) == str(episode)
    ]
    if not matches:
        raise ValueError(f"idea {idea_id} has no replay for episode {episode}")
    game = matches[-1]
    record_path = game.get("game_record_path")
    trace = _load_game_record(root, record_path) if record_path else None
    divergences = []
    if trace:
        for step in trace.get("steps", []):
            if step.get("diverged_from_v20"):
                divergences.append({
                    "replay_step": step.get("replay_step"),
                    "day": step.get("day"),
                    "hour": step.get("hour"),
                    "candidate_action": step.get("candidate_action"),
                    "recorded_v20_action": step.get("recorded_v20_action"),
                    "recorded_opponent_action": step.get("recorded_opponent_action"),
                })
    experiment = dossier["experiments"][-1] if dossier["experiments"] else {}
    return {
        "idea_id": idea_id,
        "episode": episode,
        "experiment_id": experiment.get("experiment_id"),
        "candidate": experiment.get("candidate"),
        "candidate_sha256": experiment.get("candidate_sha256"),
        "replay_call_id": game.get("replay_call_id"),
        "game_record_path": record_path,
        "original_v20_margin": game.get("original_v20_margin"),
        "candidate_margin": game.get("candidate_margin"),
        "margin_improvement": game.get("margin_improvement"),
        "result": game.get("result"),
        "action_divergences": game.get("action_divergences"),
        "first_divergence": divergences[0] if divergences else None,
        "divergence_examples": divergences[:20],
        "trace_steps": len(trace.get("steps", [])) if trace else 0,
        "interpretation_note": (
            "This comparison establishes measured action/outcome differences. "
            "Replay traces currently do not persist the full counterfactual state each turn, "
            "so state-level causal claims require additional replay instrumentation."
        ),
    }


def analyze_cash_flow(root: Path, episode: str) -> dict[str, Any]:
    _, history = load_loss_history(root, episode)
    steps = history.get("steps") or []
    if not steps:
        raise ValueError("history has no steps")
    rewards = [float(_field(state, "reward", 0.0) or 0.0) for state in steps[-1]]
    loser = 0 if rewards[0] <= rewards[1] else 1
    opponent = 1 - loser
    terms = ("money", "cash", "bank", "balance", "reward", "price")
    rows = []
    previous = None
    for turn, pair in enumerate(steps):
        snap = _state_snapshot(pair[loser])
        scalar = _find_scalar_paths_by_terms(snap["scalars"], terms)
        if previous is not None:
            changes = _deltas(previous, scalar, 20)
            if changes:
                rows.append({
                    "turn": turn,
                    "action_labels": snap["action_labels"],
                    "cash_like_changes": changes,
                })
        previous = scalar
    by_path: dict[str, float] = defaultdict(float)
    for row in rows:
        for change in row["cash_like_changes"]:
            by_path[str(change["path"])] += float(change["delta"])
    return {
        "episode": episode,
        "v20_seat": loser,
        "opponent_seat": opponent,
        "final_margin": rewards[loser] - rewards[opponent],
        "net_change_by_cash_like_path": sorted(
            [{"path": k, "net_delta": v} for k, v in by_path.items()],
            key=lambda x: abs(x["net_delta"]),
            reverse=True,
        )[:30],
        "largest_cash_events": sorted(
            rows,
            key=lambda row: sum(abs(float(x["delta"])) for x in row["cash_like_changes"]),
            reverse=True,
        )[:30],
        "note": (
            "This is an accounting-style decomposition of observable cash-like fields. "
            "It does not assign causality to policy components."
        ),
    }


def analyze_inventory_flow(root: Path, episode: str) -> dict[str, Any]:
    _, history = load_loss_history(root, episode)
    steps = history.get("steps") or []
    rewards = [float(_field(state, "reward", 0.0) or 0.0) for state in steps[-1]]
    loser = 0 if rewards[0] <= rewards[1] else 1
    terms = (
        "inventory", "capacity", "shed", "wheat", "fertil", "egg",
        "milk", "wool", "strawberry", "melon",
    )
    path_series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for turn, pair in enumerate(steps):
        snap = _state_snapshot(pair[loser])
        for path, value in _find_scalar_paths_by_terms(snap["scalars"], terms).items():
            path_series[path].append((turn, value))
    summaries = []
    for path, series in path_series.items():
        if not series:
            continue
        values = [v for _, v in series]
        peak_turn, peak_value = max(series, key=lambda x: x[1])
        summaries.append({
            "path": path,
            "start": series[0][1],
            "end": series[-1][1],
            "net_change": series[-1][1] - series[0][1],
            "peak": peak_value,
            "peak_turn": peak_turn,
        })
    summaries.sort(
        key=lambda x: (abs(float(x["net_change"])), abs(float(x["peak"]))),
        reverse=True,
    )
    return {
        "episode": episode,
        "v20_seat": loser,
        "final_margin": rewards[loser] - rewards[1 - loser],
        "inventory_capacity_paths": summaries[:60],
        "note": (
            "Generic schema-driven inventory analysis. Paths should be interpreted with "
            "competition semantics before inferring overflow or bottleneck causality."
        ),
    }


def analyze_worker_utilization(root: Path, episode: str) -> dict[str, Any]:
    _, history = load_loss_history(root, episode)
    steps = history.get("steps") or []
    rewards = [float(_field(state, "reward", 0.0) or 0.0) for state in steps[-1]]
    loser = 0 if rewards[0] <= rewards[1] else 1
    counts = Counter()
    per_turn = []
    categories = {
        "idle_pass": ("pass", "idle"),
        "transport": ("move", "pickup", "drop"),
        "crop_work": ("plant", "water", "fertil", "harvest"),
        "animal_work": ("feed", "care", "collect", "place"),
        "market_or_admin": ("hire", "buy", "sell", "land"),
    }
    for turn, pair in enumerate(steps):
        labels = _state_snapshot(pair[loser])["action_labels"]
        turn_categories = Counter()
        for label in labels:
            low = label.lower()
            matched = False
            for category, terms in categories.items():
                if any(term in low for term in terms):
                    counts[category] += 1
                    turn_categories[category] += 1
                    matched = True
            if not matched:
                counts["other"] += 1
                turn_categories["other"] += 1
        if turn_categories:
            per_turn.append({"turn": turn, "categories": dict(turn_categories)})
    total = sum(counts.values())
    return {
        "episode": episode,
        "v20_seat": loser,
        "final_margin": rewards[loser] - rewards[1 - loser],
        "action_category_counts": dict(counts),
        "action_category_share": {
            key: (value / total if total else 0.0)
            for key, value in counts.items()
        },
        "active_turn_examples": per_turn[:80],
        "note": (
            "Utilization is inferred from recorded action labels, not wall-clock worker "
            "occupancy. It is useful for spotting transport/idle-heavy policies."
        ),
    }


def component_effect_matrix(db: Any, review_id: str | None = None) -> dict[str, Any]:
    analysis = analyze_experiment_records(db, review_id)
    if analysis.get("scope") != "idea_batch":
        return analysis
    rows = []
    for item in analysis["ideas"]:
        components = item.get("components") or ["(unspecified)"]
        for component in components:
            rows.append({
                "component": component,
                "idea_id": item["idea_id"],
                "wins": int(item.get("wins") or 0),
                "losses": int(item.get("losses") or 0),
                "replay_cases": int(item.get("replay_cases") or 0),
                "mean_margin_improvement": float(item.get("mean_margin_improvement") or 0.0),
            })
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        g = grouped.setdefault(row["component"], {
            "component": row["component"],
            "ideas": 0,
            "wins": 0,
            "losses": 0,
            "replay_cases": 0,
            "mean_margin_sum": 0.0,
            "idea_ids": [],
        })
        g["ideas"] += 1
        g["wins"] += row["wins"]
        g["losses"] += row["losses"]
        g["replay_cases"] += row["replay_cases"]
        g["mean_margin_sum"] += row["mean_margin_improvement"]
        g["idea_ids"].append(row["idea_id"])
    matrix = []
    for g in grouped.values():
        n = max(1, g["ideas"])
        matrix.append({
            "component": g["component"],
            "ideas": g["ideas"],
            "idea_ids": g["idea_ids"],
            "wins": g["wins"],
            "losses": g["losses"],
            "replay_cases": g["replay_cases"],
            "mean_of_experiment_mean_margin_improvement": g["mean_margin_sum"] / n,
        })
    matrix.sort(
        key=lambda x: (
            -x["wins"],
            -x["mean_of_experiment_mean_margin_improvement"],
            x["component"],
        )
    )
    return {
        "review_id": analysis.get("review_id"),
        "component_matrix": matrix,
        "component_combinations": analysis.get("by_component_combination", []),
        "note": (
            "Descriptive matrix only. Components co-occur across ideas, so rows are not "
            "independent treatment effects."
        ),
    }


def cluster_loss_games(root: Path) -> dict[str, Any]:
    history_dir = root / "working_files" / "loss_games_v20"
    rows = []
    for path in sorted(history_dir.glob("*.json")):
        summary = analyze_loss_history(root, path.stem, window_size=24, top_windows=3)
        actions = Counter(dict(summary["v20_action_counts"]))
        transport = sum(v for k, v in actions.items() if any(t in k.lower() for t in ("move", "pickup", "drop")))
        crop = sum(v for k, v in actions.items() if any(t in k.lower() for t in ("plant", "water", "fertil", "harvest")))
        animal = sum(v for k, v in actions.items() if any(t in k.lower() for t in ("feed", "care", "collect", "place")))
        market = sum(v for k, v in actions.items() if any(t in k.lower() for t in ("sell", "buy", "hire", "land")))
        idle = sum(v for k, v in actions.items() if any(t in k.lower() for t in ("pass", "idle")))
        features = {
            "transport_actions": transport,
            "crop_actions": crop,
            "animal_actions": animal,
            "market_actions": market,
            "idle_actions": idle,
            "final_margin": float(summary["v20_final_margin"]),
        }
        dominant = max(
            ("transport", "crop", "animal", "market", "idle"),
            key=lambda name: features[f"{name}_actions"],
        )
        rows.append({
            "episode": path.stem,
            "cluster": dominant,
            "features": features,
            "critical_windows": summary["critical_windows"],
        })
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        clusters[row["cluster"]].append(row["episode"])
    representatives = []
    for cluster, episodes in sorted(clusters.items()):
        candidates = [row for row in rows if row["cluster"] == cluster]
        representative = max(
            candidates,
            key=lambda x: abs(float(x["features"]["final_margin"])),
        )
        representatives.append({
            "cluster": cluster,
            "episodes": episodes,
            "representative_episode": representative["episode"],
        })
    return {
        "games": rows,
        "clusters": representatives,
        "note": (
            "These are deterministic behavioral-signature groups, not statistical ML "
            "clusters. They are intended to diversify replay screens."
        ),
    }


def hypothesis_evidence(root: Path, db: Any, idea_id: str) -> dict[str, Any]:
    dossier = db.idea_dossier(idea_id)
    if dossier is None:
        raise ValueError("unknown idea_id: " + idea_id)
    idea = dossier["idea"]
    experiments = dossier["experiments"]
    exp = experiments[-1] if experiments else {}
    all_latest_games = _latest_games_by_episode(dossier)
    games = [game for game in all_latest_games if bool(game.get("valid"))]
    invalid_games = [game for game in all_latest_games if not bool(game.get("valid"))]
    improvements = [
        float(game["margin_improvement"])
        for game in games
        if isinstance(game.get("margin_improvement"), (int, float))
    ]
    wins = sum(1 for game in games if game.get("result") == "WIN")
    losses = sum(1 for game in games if game.get("result") == "LOSS")
    positive = sum(1 for x in improvements if x > 0)
    negative = sum(1 for x in improvements if x < 0)
    unchanged = sum(1 for x in improvements if x == 0)
    if wins > 0 and positive > negative:
        verdict = "supported_by_current_replay"
    elif losses > 0 and negative >= positive:
        verdict = "contradicted_by_current_replay"
    else:
        verdict = "mixed_or_insufficient"
    return {
        "idea_id": idea_id,
        "title": idea.get("title"),
        "hypothesis": idea.get("hypothesis"),
        "components": idea.get("components"),
        "interaction_hypothesis": idea.get("interaction_hypothesis"),
        "system_prediction": idea.get("system_prediction"),
        "promotion_rule": idea.get("promotion_rule"),
        "experiment_id": exp.get("experiment_id"),
        "candidate": exp.get("candidate"),
        "candidate_sha256": exp.get("candidate_sha256"),
        "replay_cases": len(games),
        "invalid_latest_episodes": len(invalid_games),
        "wins": wins,
        "losses": losses,
        "margin_improved_cases": positive,
        "margin_worsened_cases": negative,
        "margin_unchanged_cases": unchanged,
        "mean_margin_improvement": (
            sum(improvements) / len(improvements) if improvements else None
        ),
        "verdict": verdict,
        "per_game": [
            {
                "episode": game.get("episode"),
                "result": game.get("result"),
                "original_v20_margin": game.get("original_v20_margin"),
                "candidate_margin": game.get("candidate_margin"),
                "margin_improvement": game.get("margin_improvement"),
                "game_record_path": game.get("game_record_path"),
            }
            for game in games
        ],
        "note": (
            "Only the latest replay result per episode is counted, preventing staged "
            "1->5->25 screens from double-weighting repeated episodes. Verdict is a "
            "mechanical summary of current replay evidence, not a proof "
            "that the proposed causal mechanism is correct."
        ),
    }
