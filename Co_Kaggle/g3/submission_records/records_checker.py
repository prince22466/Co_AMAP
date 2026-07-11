"""Check artifact filenames recorded in Markdown logs against folder contents."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path


RECORDS_DIR = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".onnx", ".ipynb"}
ARTIFACT_RE = re.compile(
    r"(?i)\b([a-z0-9][a-z0-9._-]*\.(?:onnx|ipynb))\b"
)


def markdown_cells(line: str) -> list[str]:
    return [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]


def logged_artifacts(record_file: Path) -> set[str]:
    """Read filenames from Submission file columns in Markdown tables."""
    artifacts: set[str] = set()
    submission_file_index: int | None = None

    for line in record_file.read_text(encoding="utf-8-sig").splitlines():
        if "|" not in line:
            if line.lstrip().startswith("#"):
                submission_file_index = None
            continue
        cells = markdown_cells(line)
        lowered = [cell.lower() for cell in cells]
        if "submission file" in lowered:
            submission_file_index = lowered.index("submission file")
            continue
        if submission_file_index is None or submission_file_index >= len(cells):
            continue
        match = ARTIFACT_RE.search(cells[submission_file_index])
        if match:
            artifacts.add(match.group(1).lower())

    return artifacts


def check(records_dir: Path) -> bool:
    if not records_dir.is_dir():
        print(f"ERROR: records directory does not exist: {records_dir}")
        return False

    expected: dict[str, set[str]] = {}
    actual: dict[str, set[str]] = {}
    locations: dict[str, list[str]] = defaultdict(list)
    warnings: list[str] = []

    folders = sorted(
        path
        for path in records_dir.iterdir()
        if path.is_dir() and not path.name.startswith((".", "__"))
    )
    for folder in folders:
        markdown_files = sorted(folder.glob("*.md"))
        logged: set[str] = set()
        for record_file in markdown_files:
            logged.update(logged_artifacts(record_file))
        if not markdown_files:
            warnings.append(f"{folder.name}: no Markdown log found")
        elif not logged:
            warnings.append(
                f"{folder.name}: log has no explicit .onnx or .ipynb filenames"
            )
        expected[folder.name] = logged

        present = {
            file.name.lower()
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES
        }
        actual[folder.name] = present
        for filename in present:
            locations[filename].append(folder.name)

    missing: dict[str, list[str]] = {}
    unexpected: dict[str, list[str]] = {}
    for folder in expected:
        if expected[folder] - actual[folder]:
            missing[folder] = sorted(expected[folder] - actual[folder])
        # Without explicit filenames, the log provides nothing to compare.
        if expected[folder] and actual[folder] - expected[folder]:
            unexpected[folder] = sorted(actual[folder] - expected[folder])

    duplicates = {
        filename: folder_names
        for filename, folder_names in locations.items()
        if len(folder_names) > 1
    }
    issue_count = (
        sum(map(len, missing.values()))
        + sum(map(len, unexpected.values()))
        + len(duplicates)
    )

    print("Submission records artifact audit")
    print(f"Records: {records_dir}")
    print(f"Logged artifacts: {sum(map(len, expected.values()))}")
    print(f"Present artifacts: {sum(map(len, actual.values()))}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for folder, filenames in missing.items():
        print(f"MISSING [{folder}] ({len(filenames)}): {', '.join(filenames)}")
    for folder, filenames in unexpected.items():
        print(f"UNEXPECTED [{folder}] ({len(filenames)}): {', '.join(filenames)}")
    for filename, folder_names in sorted(duplicates.items()):
        print(f"DUPLICATE {filename}: {', '.join(folder_names)}")

    if issue_count:
        print(f"FAILED: {issue_count} issue(s) found.")
        return False
    print("OK: all logged notebook and ONNX artifacts are present.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    args = parser.parse_args(argv)
    return 0 if check(args.records_dir.resolve()) else 1


if __name__ == "__main__":
    sys.exit(main())
