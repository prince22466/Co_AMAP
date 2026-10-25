"""Bounded source/definition inspection; no imports or workspace modifications."""
import argparse
import ast
import json
from pathlib import Path


def inspect(workspace, path, symbol=None, start=None, lines=60):
    root = Path(workspace).resolve(strict=True)
    if len(path) > 1024 or (symbol and len(symbol) > 512):
        raise ValueError("path/symbol exceeds the bounded inspection input limit")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("path must be relative to the assigned workspace")
    target = (root / relative).resolve(strict=True)
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("path escapes the workspace or is not a file")
    if target.stat().st_size > 1_000_000:
        raise ValueError("file exceeds the 1 MB inspection limit; use a bounded native read")
    source = target.read_text(encoding="utf-8")
    definitions = []

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                first = min([child.lineno] + [d.lineno for d in child.decorator_list])
                definitions.append({"name": name, "start": first, "end": child.end_lineno})
                visit(child, name + ".")
            else:
                visit(child, prefix)

    parse_error = None
    if target.suffix == ".py":
        try:
            visit(ast.parse(source, filename=path))
        except SyntaxError as exc:
            parse_error = {"line": exc.lineno, "message": exc.msg[:300]}
    matches = [d for d in definitions if d["name"] == symbol]
    if symbol and not matches:
        matches = [d for d in definitions if d["name"].rsplit(".", 1)[-1] == symbol]
    text_lines = source.splitlines()
    first = start if start is not None else matches[0]["start"] if len(matches) == 1 else 1
    if first < 1 or not 1 <= lines <= 100:
        raise ValueError("start must be positive and lines must be between 1 and 100")
    stop = min(len(text_lines), first + lines - 1)
    if len(matches) == 1:
        stop = min(stop, matches[0]["end"])
    result = {
        "path": path, "total_lines": len(text_lines), "parse_error": parse_error,
        "symbol": symbol, "matches": matches[:8], "definition_count": len(definitions),
        "match_count": len(matches), "matches_truncated": len(matches) > 8,
        "definitions": definitions[:16] if not symbol else [],
        "definitions_truncated": not symbol and len(definitions) > 16,
        "start_line": first, "end_line": first - 1, "next_line": None, "source": "",
    }
    if symbol and len(matches) != 1:
        result["note"] = "symbol missing or ambiguous; choose an exact qualified name or explicit range"
    # Definition names can themselves consume the entire stdout allowance.
    # Reserve space for source, retaining full names only while they fit.
    while len(json.dumps(result, ensure_ascii=False)) > 2400:
        key = max((k for k in ("definitions", "matches") if result[k]),
                  key=lambda k: len(json.dumps(result[k])), default=None)
        if key is None:
            break
        result[key].pop()
        result[key + "_truncated"] = True
    for number in range(first, stop + 1):
        row = f"{number}: {text_lines[number - 1]}\n"
        previous = result["source"]
        result["source"] += row
        if len(json.dumps(result, ensure_ascii=False)) > 4200:
            result["source"] = previous
            break
        result["end_line"] = number
    desired_end = matches[0]["end"] if len(matches) == 1 else len(text_lines)
    if result["end_line"] < desired_end:
        result["next_line"] = result["end_line"] + 1
    if stop >= first and result["end_line"] < first:
        result["note"] = "source line exceeds output budget; use a bounded native character read"
        result["next_line"] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--path", required=True)
    parser.add_argument("--symbol")
    parser.add_argument("--start", type=int)
    parser.add_argument("--lines", type=int, default=60)
    args = parser.parse_args()
    try:
        result = inspect(args.workspace, args.path, args.symbol, args.start, args.lines)
    except (OSError, ValueError, UnicodeError) as exc:
        result = {"outcome": "inspection_error", "error": str(exc)[:500]}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
