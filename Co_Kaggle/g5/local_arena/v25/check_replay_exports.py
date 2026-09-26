"""Integration checks for generated static-replay histories."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runner_path = ROOT / "local_arena/static_reply/reply_template.py"
spec = importlib.util.spec_from_file_location("replay_runner_check", runner_path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def check_export(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["errors"] == 0 and summary["games_completed"] == 1
    row = summary["games"][0]
    original = runner.load_history(ROOT / "game_history/v20" / (row["episode"] + ".json"))
    result = runner.load_history(Path(row["history"]))
    assert set(result) == set(original), "official top-level schema changed"
    assert len(result["steps"]) == len(original["steps"]) == 720
    assert result["rewards"] == row["rewards"] == runner.final_rewards(result["steps"][-1])
    assert result["statuses"] == ["DONE", "DONE"]
    assert result["info"]["seed"] == original["info"]["seed"]
    assert result["configuration"] == original["configuration"]
    opponent = 1 - row["candidate_seat"]
    for before, after in zip(original["steps"][1:], result["steps"][1:]):
        assert before[opponent]["action"] == after[opponent]["action"]
        for seat in (0, 1):
            assert set(before[seat]) == set(after[seat])
    print(f"PASS {directory.name}: official schema, 720 steps, fixed opponent, consistent rewards", flush=True)
    return result


if __name__ == "__main__":
    base = ROOT / "local_arena/v24"
    for name in ("replay_runner_check", "replay_runner_seat0_check"):
        check_export(base / name)
    # Reconsume an exported history through the official engine, verifying every
    # saved observation/reward/status, including the new provenance metadata.
    engine = runner.load_engine()
    result = runner.load_history(base / "replay_runner_seat0_check/histories/111553265.json")
    runner.verify_recorded_replay(engine, result)
    print("PASS exported history is exactly reproducible by the official engine", flush=True)
    result["steps"][0][0]["reward"] = -999
    try:
        runner.verify_recorded_replay(engine, result)
    except ValueError as exc:
        assert "initial state" in str(exc)
    else:
        raise AssertionError("corrupted history was not rejected")
    print("PASS corrupted input is rejected", flush=True)
