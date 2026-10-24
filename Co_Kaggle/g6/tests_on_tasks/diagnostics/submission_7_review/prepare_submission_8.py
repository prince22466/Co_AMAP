"""Validate the candidate and edit recipe, then build a checked Submission 8 ZIP.

Run the packaging unit suite separately. This does not run Linux shell or Gemma.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tests_on_tasks/tools"))
from submission_tools import MODEL, SubmissionValidator, build_submission


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_edit_recipe(prompt: str) -> list[str]:
    block = prompt.split("```sh\n", 1)[1].split("\n```", 1)[0].splitlines()
    assert block[0] == "python3 - <<'PY'" and block[-1] == "PY"
    body = "\n".join(block[1:-1])
    source = 'def greet():\n    return "old"\n'
    old = '    return "old"'
    new = '    label = "reader\'s change"\n    return label'
    cases = [
        ("multiline_quoted_replacement", source, old, new, None),
        ("missing_match_preserves_file", source, "absent", new, AssertionError),
        ("ambiguous_match_preserves_file", source + source, old, new, AssertionError),
        ("invalid_python_preserves_file", source, old, "    return (", SyntaxError),
    ]
    passed = []
    for name, initial, observed, replacement, expected_error in cases:
        with tempfile.NamedTemporaryFile(dir=HERE, suffix=".py", delete=False) as stream:
            fixture = Path(stream.name).resolve()
        assert fixture.parent == HERE and fixture.is_relative_to(ROOT)
        try:
            fixture.write_text(initial, encoding="utf-8")
            before = fixture.read_bytes()
            code = body.replace("Path('package/module.py')", f"Path({str(fixture)!r})")
            code = code.replace("'''exact observed source'''", repr(observed))
            code = code.replace("'''replacement source'''", repr(replacement))
            try:
                exec(compile(code, "<prompt-edit-recipe>", "exec"), {})
            except Exception as error:
                if expected_error is None or not isinstance(error, expected_error):
                    raise
                assert fixture.read_bytes() == before
            else:
                assert expected_error is None, f"Expected rejection: {name}"
                assert fixture.read_text(encoding="utf-8") == initial.replace(observed, replacement, 1)
            passed.append(name)
        finally:
            # Only the known single fixture is removed; no recursive cleanup.
            fixture.unlink()
    return passed


def main():
    baseline = ROOT / "submission_nb/submission_7"
    source = ROOT / "submission_nb/submission_8"
    artifacts = ROOT / "submission_nb/artifacts"
    old_manifest = json.loads((artifacts / "submission_7.manifest.json").read_text(encoding="utf-8"))
    assert digest(artifacts / "submission_7.zip") == old_manifest["archive_sha256"]
    for name, expected in old_manifest["packaged_file_sha256"].items():
        assert digest(baseline / name) == expected, f"Baseline changed: {name}"

    validator = SubmissionValidator(source)
    checked = validator.validate()
    assert checked["agents"] == 3 and checked["model"] == MODEL and checked["adapters"] == []
    assert len(checked["package_files"]) == 10 and checked["omitted_files"] == ["README.md"]
    root = validator.load_yaml(source / "agent.yaml")
    assert [t for t in root["tools"] if isinstance(t, str)] == ["run_command", "get_status", "submit_patch"]
    helpers = [t["agent_tool"] for t in root["tools"] if isinstance(t, dict)]
    assert len(helpers) == 2
    assert checked["agent_tool_names"] == ["contract_analyst_agent", "patch_reviewer_agent"]
    for helper in helpers:
        config = validator.load_yaml(source / helper["config_path"])
        assert helper["skip_summarization"] is True
        assert config["include_contents"] == "none" and config["tools"] == []
        assert config["model"] == MODEL

    unchanged = {
        "eval_config.yaml": "eval_config.yaml",
        "configs/sampling.yaml": "configs/sampling.yaml",
        "sub_agents/contract_resources/sampling.yaml": "sub_agents/contract_resources/sampling.yaml",
        "sub_agents/reviewer_resources/sampling.yaml": "sub_agents/counterexample_resources/sampling.yaml",
    }
    for new_name, old_name in unchanged.items():
        assert (source / new_name).read_bytes() == (baseline / old_name).read_bytes()
    for name in checked["package_files"]:
        content = (source / name).read_text(encoding="utf-8")
        assert not re.search(r"(?:fastapi|requests|rich)_\d+", content)
        if name.endswith(".md"):
            assert "{" not in content and "}" not in content
        assert all(line == line.rstrip() for line in content.splitlines())
    recipe_results = check_edit_recipe((source / "prompts/system.md").read_text(encoding="utf-8"))

    output = artifacts / "submission_8.zip"
    result = build_submission(source, output)
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        assert sorted(archive.namelist()) == checked["package_files"]
        for name in checked["package_files"]:
            assert archive.read(name) == (source / name).read_bytes()
    result.update({
        "archive": output.relative_to(ROOT).as_posix(),
        "archive_sha256": digest(output),
        "packaged_file_sha256": {name: digest(source / name) for name in checked["package_files"]},
        "prepared_date": "2026-10-18", "baseline": "submission_7",
        "public_score": None, "official_runtime_validation": "not run",
        "diagnostic_archive": "tests_on_tasks/diagnostics/submission-7-zip_First30Tasks.zip",
        "diagnostic_archive_sha256": digest(HERE.parent / "submission-7-zip_First30Tasks.zip"),
        "local_checks": [
            "dependency closure: ten active files; README omitted",
            "three 31B agents, isolated tool-free helpers, no adapters",
            "only run_command/get_status/submit_patch declared as direct coder tools",
            "all task budget and sampling files byte-identical to Submission 7",
            "no benchmark task IDs, session placeholders or trailing whitespace in active prompts",
            "ZIP integrity and exact source-byte equality passed",
            "Submission 7 source and archive hashes preserved",
        ],
        "edit_recipe_python_body_checks": recipe_results,
        "separately_observed_packaging_suite": {
            "command": "python -B -m unittest discover -s tests_on_tasks/tools -p test_submission_tools.py",
            "date": "2026-10-18", "passed": 22, "skipped": 1, "failed": 0,
            "note": "Recorded preparation result; this script does not rerun that suite",
        },
        "limitations": [
            "No official runtime compiler, Gemma inference or Linux shell execution in these checks",
            "Routing instructions do not enforce helper invocation counts or latency",
            "Task resolution and public score for Submission 8 remain unmeasured",
        ],
    })
    manifest = artifacts / "submission_8.manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "archive", "archive_bytes", "archive_sha256", "agents", "model", "edit_recipe_python_body_checks",
    )}, indent=2))


if __name__ == "__main__":
    main()
