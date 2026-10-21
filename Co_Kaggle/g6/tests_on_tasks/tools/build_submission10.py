"""Validate and deterministically package submission 10; no model execution.

Requires PyYAML. Run verify_submission10_runtime.py on the built ZIP separately.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

import yaml


G6 = Path(__file__).resolve().parents[2]
BASE = G6 / "submission_nb/submission_9"
CANDIDATE = G6 / "submission_nb/submission_10"
MODEL = "gemma-4-31b-it-qat-w4a16-ct"
SOURCE_SKILL = "skills/repo_navigation"
RUNTIME_SKILL = "skills/repo-navigation"


def load_yaml(path, root, stack=()):
    path = path.resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or path in stack or len(stack) > 10:
        raise ValueError(f"invalid include: {path}")

    class Loader(yaml.SafeLoader):
        pass

    def include(loader, node):
        relative = Path(loader.construct_scalar(node))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("include traversal")
        target = path.parent / relative
        if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("include escapes submission")
        if target.suffix in {".yaml", ".yml"}:
            return load_yaml(target, root, stack + (path,))
        return target.read_text(encoding="utf-8")

    Loader.add_constructor("!include", include)
    return yaml.load(path.read_text(encoding="utf-8"), Loader=Loader)


def active_files(root):
    # Explicit release closure: documentation, tests, analysis and build code stay out.
    files = [root / "agent.yaml", root / "eval_config.yaml"]
    for name in ("configs", "prompts", "sub_agents", "skills"):
        files.extend(p for p in (root / name).rglob("*")
                     if "__pycache__" not in p.parts and (p.is_file() or p.is_symlink()))
    return sorted(files)


def validate(root=CANDIDATE, baseline=BASE):
    files = active_files(root)
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"unsafe file: {path}")
        if path.suffix not in {".yaml", ".yml", ".md", ".txt", ".py"}:
            raise ValueError(f"unexpected release file: {path}")
        if path.suffix == ".py":
            ast.parse(path.read_bytes(), filename=str(path))
    agent = load_yaml(root / "agent.yaml", root)
    base_agent = load_yaml(baseline / "agent.yaml", baseline)
    for key in ("name", "model", "tools", "generate_content_config"):
        if agent[key] != base_agent[key]:
            raise ValueError(f"baseline architecture changed: {key}")
    if agent["model"] != MODEL or agent.get("skills") != ["skills/repo_navigation"]:
        raise ValueError("unexpected model or skill registration")
    prompt = (root / "prompts/system.md").read_text()
    if "repair_gate.py" not in prompt:
        raise ValueError("missing repair gate instructions")
    preserved = []
    for path in active_files(baseline):
        relative = path.relative_to(baseline)
        if str(relative) in {"agent.yaml", "prompts/system.md", "skills/repo_navigation/SKILL.md"}:
            continue
        if path.read_bytes() != (root / relative).read_bytes():
            raise ValueError(f"baseline file changed: {relative}")
        preserved.append(str(relative))
    if not (root / SOURCE_SKILL / "scripts/repair_gate.py").is_file():
        raise ValueError("missing repair gate script")
    checker = (root / SOURCE_SKILL / "scripts/check.py").read_text()
    gate = (root / SOURCE_SKILL / "scripts/repair_gate.py").read_text()
    embedded = checker[checker.index("_CHILD_RUNNER ="):checker.index("\\ndef main():")]
    if embedded not in gate or "importlib.util" in gate:
        raise ValueError("repair gate must embed the unchanged checker for ADK runpy")

    baseline_paths = {p.relative_to(baseline).as_posix() for p in active_files(baseline)}
    candidate_paths = {p.relative_to(root).as_posix() for p in files}
    expected_new = {"skills/repo_navigation/scripts/repair_gate.py"}
    if candidate_paths - baseline_paths != expected_new:
        raise ValueError(f"unexpected added active files: {sorted(candidate_paths - baseline_paths)}")
    if baseline_paths - candidate_paths:
        raise ValueError(f"missing baseline active files: {sorted(baseline_paths - candidate_paths)}")

    for helper in agent["tools"]:
        if isinstance(helper, dict):
            config = load_yaml(root / helper["agent_tool"]["config_path"], root)
            if config["model"] != MODEL or config["tools"] != [] or config.get("skills"):
                raise ValueError("helper no longer tool-free or model changed")
    skill = root / agent["skills"][0]
    manifest = (skill / "SKILL.md").read_text()
    if not manifest.startswith("---\n"):
        raise ValueError("missing skill frontmatter")
    metadata = yaml.safe_load(manifest.split("---", 2)[1])
    if metadata["name"] != Path(RUNTIME_SKILL).name or not metadata.get("description"):
        raise ValueError("skill metadata does not match packaged directory")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", metadata["name"]):
        raise ValueError("invalid skill name")
    for relative in re.findall(r"(?:scripts|resources|references)/[a-z_]+\.(?:py|md)", manifest):
        relative = relative.replace("references/", "resources/", 1)
        if not (skill / relative).is_file():
            raise ValueError(f"missing skill resource: {relative}")
    package_text = "\n".join(p.read_text(encoding="utf-8") for p in files)
    if re.search(r"(?:fastapi|requests|rich|httpx)_\d{3,}", package_text):
        raise ValueError("task ID leaked into active package")
    return files, preserved


def release_entries(files):
    """Keep the requested source layout; project it to ADK 1.36.1's format."""
    entries, aliases = {}, {}
    for path in files:
        source_name = path.relative_to(CANDIDATE).as_posix()
        name = source_name.replace(SOURCE_SKILL + "/", RUNTIME_SKILL + "/", 1)
        data = path.read_bytes()
        if source_name == "agent.yaml":
            data = data.replace(SOURCE_SKILL.encode(), RUNTIME_SKILL.encode())
        entries[name] = data
        if name.startswith(RUNTIME_SKILL + "/resources/"):
            alias = name.replace("/resources/", "/references/", 1)
            if alias in entries:
                raise ValueError(f"duplicate generated resource: {alias}")
            entries[alias] = data
            aliases[alias] = source_name
    return dict(sorted(entries.items())), aliases


def build(output):
    files, preserved = validate()
    entries, aliases = release_entries(files)
    if len(entries) > 10000 or sum(map(len, entries.values())) >= 3 * 1024**3:
        raise ValueError("submission exceeds competition archive limits")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, data in entries.items():
            info = zipfile.ZipInfo(name, (2026, 10, 20, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, data)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError("corrupt ZIP")
        if set(z.namelist()) != set(entries):
            raise ValueError("archive entries mismatch")
        for name, data in entries.items():
            if z.read(name) != data:
                raise ValueError("archive/source mismatch")
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(G6), "rev-parse", "HEAD"], text=True, timeout=5).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        source_commit = None
    manifest = {
        "candidate": "submission_10", "baseline": "submission_9",
        "source_commit": source_commit,
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "file_count": len(entries), "unpacked_bytes": sum(map(len, entries.values())),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()},
        "runtime_skill_directory": RUNTIME_SKILL,
        "generated_reference_aliases": aliases,
        "byte_identical_baseline_files": preserved,
        "validation": {"static_structure": "passed", "zip_source_parity": "passed",
                       "official_compiler_smoke": "not run",
                       "gemma_evaluation": "not run", "public_score": None},
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=G6 / "submission_nb/artifacts/submission_10.zip")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        files, preserved = validate()
        print(json.dumps({"static_validation": "passed", "files": len(files), "preserved": len(preserved)}))
    else:
        result = build(args.output)
        print(json.dumps({"output": str(args.output), "files": result["file_count"],
                          "sha256": result["archive_sha256"]}))


if __name__ == "__main__":
    main()
