"""Check that scored tasks recorded in Markdown have submission ONNX files.

The record directory and submission directory are matched by subfolder name.  A
non-zero exit status is returned when files are missing, duplicated, misplaced,
or incorrectly named, which makes this script useful both interactively and in
CI.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RECORDS_DIR = PROJECT_ROOT / "submission_records"
DEFAULT_ONNX_DIR = SCRIPT_DIR

TASK_RE = re.compile(r"(?i)\b(task\d{3})\b")
ONNX_RE = re.compile(r"(?i)\b(task\d{3})\.onnx\b")
VALID_FILENAME_RE = re.compile(r"(?i)^task\d{3}\.onnx$")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


@dataclass
class RecordResult:
    tasks: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)


def _cells(line: str) -> list[str]:
    """Return stripped cells from a simple Markdown table row."""
    return [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]


def parse_record(record_file: Path, artifact_rows_only: bool = False) -> RecordResult:
    """Extract scored task IDs from submission-summary Markdown tables.

    Normally the ``task`` column is authoritative.  Some historical rows have
    stale task cells but a correct ``taskNNN.onnx`` artifact name (and vice
    versa), so an existing archived ONNX beside the record resolves conflicts.
    """
    result = RecordResult()
    lines = record_file.read_text(encoding="utf-8-sig").splitlines()
    header: list[str] | None = None

    for line_no, line in enumerate(lines, 1):
        if not line.lstrip().startswith("|"):
            header = None
            continue
        cells = _cells(line)
        lowered = [cell.lower() for cell in cells]
        if "task" in lowered and "submission score" in lowered:
            header = lowered
            continue
        if header is None or all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if len(cells) < len(header):
            continue

        task_idx = header.index("task")
        score_idx = header.index("submission score")
        task_match = TASK_RE.search(cells[task_idx])
        score_text = cells[score_idx].replace(",", "").strip()
        if not task_match or not NUMBER_RE.fullmatch(score_text):
            continue
        if float(score_text) <= 0:
            continue

        task_id = task_match.group(1).lower()
        artifact_id = None
        if "submission file" in header:
            artifact_match = ONNX_RE.search(cells[header.index("submission file")])
            artifact_id = artifact_match.group(1).lower() if artifact_match else None

        if artifact_rows_only:
            if artifact_id:
                result.tasks.add(artifact_id)
            continue

        if artifact_id and artifact_id != task_id:
            artifact_archive = record_file.parent / f"{artifact_id}.onnx"
            if artifact_archive.is_file():
                chosen = artifact_id
            else:
                chosen = task_id
            result.warnings.append(
                f"{record_file.name}:{line_no}: task column says {task_id}, "
                f"artifact says {artifact_id}; using {chosen}"
            )
            task_id = chosen
        result.tasks.add(task_id)

    return result


def collect_expected(
    records_dir: Path, artifact_rows_only: bool = False
) -> tuple[dict[str, set[str]], list[str]]:
    expected: dict[str, set[str]] = {}
    warnings: list[str] = []
    for folder in sorted(
        path
        for path in records_dir.iterdir()
        if path.is_dir() and not path.name.startswith((".", "__"))
    ):
        markdown_files = sorted(folder.glob("*.md"))
        tasks: set[str] = set()
        if not markdown_files:
            warnings.append(f"{folder.name}: no Markdown record found")
        for record_file in markdown_files:
            parsed = parse_record(record_file, artifact_rows_only=artifact_rows_only)
            tasks.update(parsed.tasks)
            warnings.extend(parsed.warnings)
        if markdown_files and not tasks and not artifact_rows_only:
            warnings.append(
                f"{folder.name}: record contains no per-task scored rows; "
                "only filename and duplicate checks can be performed"
            )
        expected[folder.name] = tasks
    return expected, warnings


def check(
    records_dir: Path, onnx_dir: Path, artifact_rows_only: bool = False
) -> bool:
    """Print a complete audit and return True when no errors are found."""
    if not records_dir.is_dir():
        print(f"ERROR: records directory does not exist: {records_dir}")
        return False
    if not onnx_dir.is_dir():
        print(f"ERROR: ONNX directory does not exist: {onnx_dir}")
        return False

    expected, warnings = collect_expected(
        records_dir, artifact_rows_only=artifact_rows_only
    )
    actual: dict[str, set[str]] = {}
    locations: dict[str, list[str]] = defaultdict(list)
    invalid: list[str] = []

    for folder in sorted(
        path
        for path in onnx_dir.iterdir()
        if path.is_dir() and not path.name.startswith((".", "__"))
    ):
        names: set[str] = set()
        for file in sorted(folder.iterdir()):
            if not file.is_file() or file.suffix.lower() != ".onnx":
                continue
            if not VALID_FILENAME_RE.fullmatch(file.name):
                invalid.append(f"{folder.name}/{file.name}")
                continue
            filename = file.name.lower()
            names.add(filename)
            locations[filename].append(folder.name)
        actual[folder.name] = names

    missing: dict[str, list[str]] = {}
    unexpected: dict[str, list[str]] = {}
    all_folders = sorted(set(expected) | set(actual))
    for folder in all_folders:
        wanted = {f"{task}.onnx" for task in expected.get(folder, set())}
        present = actual.get(folder, set())
        if wanted - present:
            missing[folder] = sorted(wanted - present)
        # A batch-only record cannot identify which individual ONNX files are
        # expected, so do not manufacture "unexpected" errors for that folder.
        if (artifact_rows_only or expected.get(folder)) and present - wanted:
            unexpected[folder] = sorted(present - wanted)

    duplicates = {name: folders for name, folders in locations.items() if len(folders) > 1}
    error_count = sum(map(len, missing.values())) + sum(map(len, unexpected.values()))
    error_count += len(duplicates) + len(invalid)

    print("ONNX submission audit")
    print(f"Records: {records_dir}")
    print(f"ONNX files: {onnx_dir}")
    if artifact_rows_only:
        print("Mode: logged ONNX artifacts only")
    print(f"Expected scored tasks: {sum(map(len, expected.values()))}")
    print(f"Present valid ONNX files: {sum(map(len, actual.values()))}")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for folder, names in missing.items():
        print(f"MISSING [{folder}] ({len(names)}): {', '.join(names)}")
    for folder, names in unexpected.items():
        print(f"UNEXPECTED [{folder}] ({len(names)}): {', '.join(names)}")
    for name, folders in sorted(duplicates.items()):
        print(f"DUPLICATE {name}: {', '.join(folders)}")
    for path in invalid:
        print(f"INVALID FILENAME: {path} (expected taskNNN.onnx)")

    if error_count:
        print(f"FAILED: {error_count} issue(s) found.")
        return False
    print("OK: every scored task has exactly one correctly placed ONNX file.")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument(
        "--artifact-rows-only",
        action="store_true",
        help="expect only ONNX filenames explicitly logged in Submission file cells",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return (
        0
        if check(
            args.records_dir.resolve(),
            args.onnx_dir.resolve(),
            artifact_rows_only=args.artifact_rows_only,
        )
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
