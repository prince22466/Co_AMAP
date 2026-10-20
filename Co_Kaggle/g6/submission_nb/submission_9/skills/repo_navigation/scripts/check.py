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


def check(workspace, code, module=None, timeout=15):
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir() or not 1 <= timeout <= 20:
        raise ValueError("workspace must exist; timeout must be between 1 and 20 seconds")
    if len(code) > 16000:
        raise ValueError("check exceeds 16000 characters; reduce it to the discriminating case")
    tree = ast.parse(code, filename="<skill-check>")
    assertions = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    if module and (not all(part.isidentifier() for part in module.split("."))):
        raise ValueError("module must be a dotted Python module name")
    paths = [str(p) for p in (root / "src", root) if p.is_dir()]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths), PYTHONDONTWRITEBYTECODE="1")
    # The child's marker file distinguishes normal completion from sys.exit(0).
    # Runtime results stay in /tmp, outside the captured repository patch.
    with tempfile.TemporaryDirectory(prefix="submission9-check-") as scratch:
        temp = Path(scratch)
        status = temp / "status.json"
        program = (
            "import importlib, json, sys\nfrom pathlib import Path\n"
            f"_root = Path({str(root)!r}).resolve()\n"
            f"_status = Path({str(status)!r})\n"
            "_result = {'completed': False, 'module_origin': None}\n"
            f"_module_name = {module!r}\n"
            "if _module_name:\n"
            "    _module = importlib.import_module(_module_name)\n"
            "    _origin = getattr(_module, '__file__', None)\n"
            "    _result['module_origin'] = _origin\n"
            "    if not _origin or not Path(_origin).resolve().is_relative_to(_root):\n"
            "        _result['origin_error'] = True\n"
            "        _status.write_text(json.dumps(_result))\n"
            "        raise RuntimeError('requested module did not load from the workspace')\n"
            "_status.write_text(json.dumps(_result))\n"
            f"exec(compile({code!r}, '<skill-check>', 'exec'), {{'__name__': '__main__'}})\n"
            "_result['completed'] = True\n"
            "_status.write_text(json.dumps(_result))\n"
        )
        script = temp / "check.py"
        script.write_text(program, encoding="utf-8")
        timed_out = False
        with (temp / "stdout").open("wb") as stdout, (temp / "stderr").open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-B", str(script)], cwd=root, env=env,
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
            "failed" if process.returncode else
            "incomplete" if not state.get("completed") else
            "passed" if assertions else "exit_zero_without_assertions"
        )
        result = {
            "outcome": outcome, "returncode": process.returncode,
            "assertions_in_source": assertions, "completed": state.get("completed", False),
            "module_origin": state.get("module_origin"), "workspace": str(root),
            "stdout_tail": stdout, "stderr_tail": stderr,
            "output_truncated": stdout_cut or stderr_cut,
            "scope": "Only this check; assertion coverage and hidden-test success are not established.",
        }
        while len(json.dumps(result, ensure_ascii=False)) > 4400:
            key = max(("stdout_tail", "stderr_tail"), key=lambda k: len(result[k]))
            result[key] = result[key][len(result[key]) // 2:]
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
