"""Run a short assertion check against workspace imports in a fresh process."""
import argparse
import ast
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


_CHILD_RUNNER = r'''
import ast
import importlib
import json
from pathlib import Path
import sys
import uuid

request = json.loads(Path(sys.argv[1]).read_text())
status = Path(sys.argv[2])
result = {"completed": False, "module_origin": None,
          "assertions_passed": 0, "assertions_failed": 0}
status.write_text(json.dumps(result))

def record(passed):
    result["assertions_passed" if passed else "assertions_failed"] += 1

# Count executed assertions, including failures caught by the reproduction.
# Random globals avoid collisions and single underscores avoid class mangling.
prefix = "_skill_" + uuid.uuid4().hex
record_name, error_name = prefix + "_record", prefix + "_error"

class Instrument(ast.NodeTransformer):
    def visit_Assert(self, node):
        def mark(passed):
            return ast.Expr(ast.Call(ast.Name(record_name, ast.Load()),
                                     [ast.Constant(passed)], []))
        return ast.copy_location(ast.Try(
            body=[node],
            handlers=[ast.ExceptHandler(ast.Name(error_name, ast.Load()), None,
                                       [mark(False), ast.Raise()])],
            orelse=[mark(True)], finalbody=[]), node)

try:
    if request["module"]:
        module = importlib.import_module(request["module"])
        origin = getattr(module, "__file__", None)
        result["module_origin"] = origin
        if not origin or not Path(origin).resolve().is_relative_to(Path(request["root"])):
            result["origin_error"] = True
            raise RuntimeError("requested module did not load from the workspace")
    tree = ast.fix_missing_locations(Instrument().visit(ast.parse(request["code"])))
    namespace = {"__name__": "__main__", record_name: record, error_name: AssertionError}
    exec(compile(tree, "<skill-check>", "exec", optimize=0), namespace)
    result["completed"] = True
finally:
    status.write_text(json.dumps(result))
'''


def check(workspace, code, module=None, timeout=15):
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir() or len(str(root)) > 1024 or not 1 <= timeout <= 20:
        raise ValueError("workspace must exist and fit 1024 characters; timeout must be 1–20 seconds")
    if len(code) > 16000:
        raise ValueError("check exceeds 16000 characters; reduce it to the discriminating case")
    tree = ast.parse(code, filename="<skill-check>")
    assertions = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    if module and (len(module) > 300 or not all(part.isidentifier() for part in module.split("."))):
        raise ValueError("module must be a dotted Python module name")
    paths = [str(p) for p in (root / "src", root) if p.is_dir()]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths),
               PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0")
    # The child's marker file distinguishes normal completion from sys.exit(0).
    # Runtime results stay in /tmp, outside the captured repository patch.
    with tempfile.TemporaryDirectory(prefix="submission9-check-") as scratch:
        temp = Path(scratch)
        status = temp / "status.json"
        request = temp / "request.json"
        request.write_text(json.dumps({"root": str(root), "module": module, "code": code}))
        script = temp / "check.py"
        script.write_text(_CHILD_RUNNER, encoding="utf-8")
        timed_out = False
        with (temp / "stdout").open("wb") as stdout, (temp / "stderr").open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-B", str(script), str(request), str(status)], cwd=root, env=env,
                stdout=stdout, stderr=stderr, start_new_session=(os.name == "posix"),
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                process.wait()
        state = json.loads(status.read_text()) if status.exists() else {}

        def tail(path):
            with path.open("rb") as handle:
                size = path.stat().st_size
                handle.seek(max(0, size - 1400))
                return handle.read(1400).decode("utf-8", errors="replace"), size > 1400

        stdout, stdout_cut = tail(temp / "stdout")
        stderr, stderr_cut = tail(temp / "stderr")
        outcome = (
            "timeout" if timed_out else
            "import_origin_error" if state.get("origin_error") else
            "failed" if process.returncode or state.get("assertions_failed") else
            "incomplete" if not state.get("completed") else
            "passed" if state.get("assertions_passed") else
            "no_assertions_executed" if assertions else "exit_zero_without_assertions"
        )
        result = {
            "outcome": outcome, "returncode": process.returncode,
            "assertions_in_source": assertions, "completed": state.get("completed", False),
            "assertions_passed": state.get("assertions_passed", 0),
            "assertions_failed": state.get("assertions_failed", 0),
            "module_origin": state.get("module_origin"), "workspace": str(root),
            "stdout_tail": stdout, "stderr_tail": stderr,
            "output_truncated": stdout_cut or stderr_cut,
            "scope": "Only this check; assertion coverage and hidden-test success are not established.",
        }
        while len(json.dumps(result, ensure_ascii=False)) > 4400:
            key = max(("stdout_tail", "stderr_tail", "module_origin", "workspace"),
                      key=lambda k: len(result[k] or ""))
            result[key] = result[key][max(1, len(result[key]) // 2):]
            result["output_truncated"] = True
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--module")
    parser.add_argument("--code", required=True)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()
    try:
        result = check(args.workspace, args.code, args.module, args.timeout)
    except (OSError, ValueError, SyntaxError) as exc:
        result = {"outcome": "check_error", "error": str(exc)[:500]}
    # The tool itself can complete successfully while the check fails.
    # Always read outcome; this also preserves JSON under ADK's runpy wrapper.
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
