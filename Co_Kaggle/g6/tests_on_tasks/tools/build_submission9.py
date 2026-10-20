"""Validate and deterministically package submission 9; no model execution.

Requires PyYAML. Run verify_submission9_runtime.py on the built ZIP separately.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

import yaml


G6 = Path(__file__).resolve().parents[2]
BASE = G6 / "submission_nb/submission_7"
CANDIDATE = G6 / "submission_nb/submission_9"
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
        files.extend(p for p in (root / name).rglob("*") if p.is_file() or p.is_symlink())
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
    opening, remainder = (baseline / "prompts/system.md").read_text().split("\n\n", 1)
    prompt = (root / "prompts/system.md").read_text()
    if not prompt.startswith(opening + "\n\n") or not prompt.endswith(remainder):
        raise ValueError("original submission-7 instructions were modified")
    preserved = []
    for path in active_files(baseline):
        relative = path.relative_to(baseline)
        if str(relative) in {"agent.yaml", "prompts/system.md"}:
            continue
        if path.read_bytes() != (root / relative).read_bytes():
            raise ValueError(f"baseline file changed: {relative}")
        preserved.append(str(relative))
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
        for name, data in entries.items():
            if z.read(name) != data:
                raise ValueError("archive/source mismatch")
    manifest = {
        "candidate": "submission_9", "baseline": "submission_7",
        "source_commit": "cbe742951dcc089b56ba9331f64b3d64647ae595",
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "file_count": len(entries), "unpacked_bytes": sum(map(len, entries.values())),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()},
        "runtime_skill_directory": RUNTIME_SKILL,
        "generated_reference_aliases": aliases,
        "byte_identical_baseline_files": preserved,
        "validation": {"static_structure": "passed", "zip_source_parity": "passed",
                       "official_compiler_smoke": "separate hash-matched runtime_smoke.json report",
                       "gemma_evaluation": "not run", "public_score": None},
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=G6 / "submission_nb/artifacts/submission_9.zip")
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
