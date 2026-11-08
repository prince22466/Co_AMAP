"""S37 research A/B: identical S36 package, optional host-side edit gate.

control: unchanged official harness + S36 primary archive.
guard: same official harness and archive, but injects a deterministic
       ADK before/after tool plugin in this Python process.

Only the guard arm modifies the research harness. The controller cannot be
serialized by the restricted competition YAML compiler. This is NOT a
competition-ready S37 ZIP or a claimed Kaggle scoring improvement.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
from pathlib import Path
from unittest.mock import patch

from build_submission36 import MODEL, archive_path, build, payload
from submission37_repetition_guard import (
    new_adk_plugin, patch_app_initializer,
)
import run_submission24_eval as official


def source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_s36_archive() -> Path:
    """Rebuild exact S36 primary, verify all member bytes before evaluation."""
    import zipfile

    archive = archive_path("primary")
    build("primary")
    expected = payload("primary")
    with zipfile.ZipFile(archive) as zipped:
        if set(zipped.namelist()) != set(expected):
            raise RuntimeError("S36 archive members differ from known baseline")
        for path, data in expected.items():
            if zipped.read(path) != data:
                raise RuntimeError("S36 archive member drift: " + path)
    return archive


async def evaluate(args: argparse.Namespace) -> dict:
    if args.variant not in ("control", "guard"):
        raise ValueError("Expected control or guard")
    if official.MODEL != MODEL:
        raise RuntimeError("S36 model differs from official evaluation runner")
    args.archive = canonical_s36_archive()
    injected_plugins = []
    if args.variant == "control":
        report = await official.evaluate(args)
    else:
        # ADK App is instantiated inside swegemma.run_agent_sandbox. Patching
        # the CLASS object, rather than a module alias, catches existing
        # imports of App; all changes are scoped to this with-context.
        from google.adk.apps import App
        from google.adk.plugins.base_plugin import BasePlugin
        if (not hasattr(App, "__init__")
                or not hasattr(BasePlugin, "before_tool_callback")
                or "tool_args" not in inspect.signature(
                    BasePlugin.before_tool_callback).parameters):
            raise RuntimeError("Unsupported ADK tool callback version")
        original_init = App.__init__
        init = patch_app_initializer(
            original_init, new_adk_plugin, injected_plugins)
        with patch.object(App, "__init__", init):
            report = await official.evaluate(args)
        if (report.get("stage") == "evaluation_complete"
                and not injected_plugins):
            raise RuntimeError(
                "S37 guard did not attach to an ADK App. "
                "No valid guarded comparison can be claimed.")
    counts = {}
    events = []
    for plugin in injected_plugins:
        snapshot = plugin.guard.report()
        for name, value in snapshot["counts"].items():
            counts[name] = counts.get(name, 0) + value
        events.extend(snapshot["events"])
    telemetry = {
        "variant": args.variant,
        "archive": "submission_36 PRIMARY (identical in control and guard)",
        "archive_sha256": source_sha(args.archive),
        "guard_installed": bool(injected_plugins),
        "app_instances": len(injected_plugins),
        "guard_counts": counts,
        "guard_events": events[:1000],
        "research_only": True,
        "note": (
            "The S36 model, tools, sampling, prompt and official grading "
            "are unchanged. Host-side ADK callback only in the guard arm. "
            "Skipped executions may alter charged-tool budgets; record them."
        ),
    }
    if not args.output.is_dir():
        raise RuntimeError("Official runner did not create output directory")
    (args.output / "submission37_research_telemetry.json").write_text(
        json.dumps(telemetry, indent=2) + "\n", encoding="utf-8")
    report.update({
        "research_variant": args.variant,
        "comparison_base": "submission_36_primary",
        "guard_installed": telemetry["guard_installed"],
        "guard_counts": counts,
        "guarded_scoring_claim": False,
        "promotion_approved": False,
    })
    (args.output / "submission37_research_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=("control", "guard"))
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--model-url", default="http://localhost:8000/v1")
    parser.add_argument("--served-model", default=MODEL)
    parser.add_argument("--sandbox", choices=("docker", "subprocess"),
                        default="subprocess")
    parser.add_argument("--stage", choices=("pilot", "s1"), default="pilot")
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--time-limit", type=official.nonnegative_minutes,
                        default=4.0)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = asyncio.run(evaluate(args))
    print(json.dumps(summary, indent=2))
    if summary.get("stage") == "baseline_blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
