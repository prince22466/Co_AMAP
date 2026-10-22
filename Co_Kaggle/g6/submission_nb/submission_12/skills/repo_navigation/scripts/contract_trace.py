"""Read-only, bounded structural tracing for a named Python API parameter.

This reports evidence candidates, not a full call graph or behavioral proof.
Inspect only explicit workspace-relative paths; never crawl a repository.
"""
import argparse
import ast
import json
from pathlib import Path
import re

MAX_FILES = 6
MAX_BYTES_PER_FILE = 120_000
MAX_FUNCTIONS = 18
MAX_FORWARDERS_PER_FUNCTION = 6
MAX_OUTPUT = 4200


def _identifier(value):
    if not value or len(value) > 80 or not value.isidentifier():
        raise ValueError("parameter must be a Python identifier")
    return value


def _name(node):
    try:
        return ast.unparse(node)[:100]
    except (ValueError, TypeError, RecursionError):
        return "<dynamic>"


class _LocalCalls(ast.NodeVisitor):
    """Walk one function body without claiming calls in nested definitions."""

    def __init__(self, parameter, kwargs_name):
        self.parameter = parameter
        self.kwargs_name = kwargs_name
        self.forwarding = []

    def visit_FunctionDef(self, node):
        return

    def visit_AsyncFunctionDef(self, node):
        return

    def visit_ClassDef(self, node):
        return

    def visit_Call(self, node):
        for keyword in node.keywords:
            if keyword.arg == self.parameter:
                mode = ("parameter_value" if isinstance(keyword.value, ast.Name)
                        and keyword.value.id == self.parameter else "explicit_keyword")
                self.forwarding.append({"callee": _name(node.func), "line": node.lineno,
                                        "mode": mode})
            elif (keyword.arg is None and self.kwargs_name
                  and isinstance(keyword.value, ast.Name)
                  and keyword.value.id == self.kwargs_name):
                self.forwarding.append({"callee": _name(node.func), "line": node.lineno,
                                        "mode": "opaque_kwargs"})
        self.generic_visit(node)


class _Definitions(ast.NodeVisitor):
    def __init__(self, path, parameter, symbol):
        self.path = path
        self.parameter = parameter
        self.symbol = symbol
        self.stack = []
        self.items = []

    def visit_ClassDef(self, node):
        self.stack.append(node.name)
        for child in node.body:
            self.visit(child)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._function(node)

    def visit_AsyncFunctionDef(self, node):
        self._function(node)

    def _function(self, node):
        qualified = ".".join(self.stack + [node.name])
        positional_only = {arg.arg for arg in node.args.posonlyargs}
        keyword_capable = {arg.arg for arg in
                           (node.args.args + node.args.kwonlyargs)}
        variadic = node.args.kwarg.arg if node.args.kwarg else None
        accepts = ("explicit" if self.parameter in keyword_capable else
                   "positional_only" if self.parameter in positional_only else
                   "variadic_kwargs" if variadic else "not_declared")
        visitor = _LocalCalls(self.parameter, variadic)
        for statement in node.body:
            visitor.visit(statement)
        forward = visitor.forwarding[:MAX_FORWARDERS_PER_FUNCTION]
        wanted = (accepts != "not_declared" or bool(forward)
                  or (self.symbol is not None
                      and (qualified == self.symbol
                           or qualified.endswith("." + self.symbol))))
        if wanted:
            self.items.append({"symbol": qualified, "path": self.path,
                               "line": node.lineno, "accepts": accepts,
                               "forwards": forward,
                               "forwarding_candidates": len(visitor.forwarding),
                               "forwarding_truncated": len(visitor.forwarding) > len(forward)})
        self.stack.append(node.name)
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.visit(statement)
        self.stack.pop()


def trace(workspace, paths, parameter, symbol=None):
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir() or len(str(root)) > 1024:
        raise ValueError("workspace must be a short existing directory")
    parameter = _identifier(parameter)
    if symbol and (len(symbol) > 160 or not all(p.isidentifier() for p in symbol.split("."))):
        raise ValueError("symbol must be a dotted Python name")
    if not isinstance(paths, str) or len(paths) > 1800:
        raise ValueError("paths must be a short comma-separated list")
    names = [part.strip() for part in paths.split(",") if part.strip()]
    if not 1 <= len(names) <= MAX_FILES or len(set(names)) != len(names):
        raise ValueError("provide 1-6 distinct candidate paths")
    found, warnings = [], []
    for name in names:
        rel = Path(name)
        if (rel.is_absolute() or ".." in rel.parts or rel.suffix != ".py"
                or str(rel) in {".", ""}):
            raise ValueError("paths must be relative workspace Python files")
        candidate = root / rel
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
            raise ValueError("symlink or path outside workspace")
        if not candidate.is_file():
            warnings.append({"path": name, "issue": "missing_file"})
            continue
        if candidate.stat().st_size > MAX_BYTES_PER_FILE:
            warnings.append({"path": name, "issue": "file_too_large"})
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
            parsed = ast.parse(text, filename=name)
        except (SyntaxError, UnicodeError, OSError) as exc:
            warnings.append({"path": name, "issue": "parse_error",
                             "detail": str(exc)[:140]})
            continue
        v = _Definitions(name, parameter, symbol)
        try:
            v.visit(parsed)
        except RecursionError:
            warnings.append({"path": name, "issue": "traversal_too_deep"})
            continue
        found.extend(v.items)

    def requested(row):
        # A named public entry point must remain visible even in a large module.
        return (symbol is not None and
                (row["symbol"] == symbol or row["symbol"].endswith("." + symbol)))

    ordered = sorted(found, key=lambda row:
                     (not requested(row), row["accepts"] in
                      ("not_declared", "positional_only"), row["path"], row["line"]))
    seen = sum(row["forwarding_candidates"] for row in ordered)
    trimmed = ordered[:MAX_FUNCTIONS]
    result = {
        "outcome": "structural_candidates_only",
        "parameter": parameter,
        "paths_requested": names,
        "symbol": symbol,
        "functions": trimmed,
        "functions_found": len(ordered),
        "forwarding_candidates": seen,
        "warnings": warnings,
        "truncated": len(ordered) > len(trimmed),
        "limitations": ("Only static syntax in the named files. Keyword forwarding and "
                        "**kwargs do not prove runtime propagation; dynamic dispatch, "
                        "aliases, positional forwarding, decorators and consumers may 
                        "not be visible. Verify public behavior with executable tests."),
    }
    while len(json.dumps(result, ensure_ascii=False)) > MAX_OUTPUT and result["functions"]:
        result["functions"].pop()
        result["truncated"] = True
    while len(json.dumps(result, ensure_ascii=False)) > MAX_OUTPUT and result["warnings"]:
        result["warnings"].pop()
        result["truncated"] = True
    if len(json.dumps(result, ensure_ascii=False)) > MAX_OUTPUT:
        raise ValueError("trace output too large; provide shorter paths")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--paths", required=True)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--symbol")
    args = parser.parse_args()
    try:
        result = trace(args.workspace, args.paths, args.parameter, args.symbol)
    except (OSError, ValueError) as exc:
        result = {"outcome": "trace_error", "error": str(exc)[:400],
                  "limitations": "Use a targeted native source read instead of retrying blindly."}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
