"""Graph-guided, budget-aware repair controller for a single coding task.

Executes the Submission 9 assertion checker and retains per-task evidence in
Git metadata, outside the captured implementation patch. Never edits code.
"""
import argparse
import hashlib
import ast
import os
import signal
import sys
import tempfile
import json
from pathlib import Path
import re
import subprocess

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



MAX_REPAIRS = 2
MIN_SECONDS_FOR_CHECK = 45
MIN_CALLS_FOR_CHECK = 6
MAX_REQUIREMENTS = 4
MAX_HASH_BYTES = 32 * 1024 * 1024


def identifiers(raw):
    names = [s.strip() for s in raw.split(",") if s.strip()]
    if len(names) > MAX_REQUIREMENTS or len(set(names)) != len(names):
        raise ValueError("use at most four distinct requirement IDs")
    if not all(re.fullmatch(r"[a-z][a-z0-9_]{0,39}", n) for n in names):
        raise ValueError("requirement IDs must be short lowercase identifiers")
    return names


def git(root, *args):
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, timeout=8, check=True)
    return done.stdout


def fingerprint(root):
    """Fingerprint working-tree and index changes, including renames and deletions.

    Parse porcelain -z as byte records; rename/copy records consume a second
    pathname. The Git index diff also detects staged-only content changes.
    """
    records = git(root, "status", "--porcelain=v1", "--untracked-files=all", "-z").split(b"\0")
    h = hashlib.sha256()
    count = 0
    hashed_bytes = 0
    i = 0
    while i < len(records):
        record = records[i]
        i += 1
        if not record:
            continue
        if len(record) < 4 or record[2:3] != b" ":
            raise ValueError("malformed porcelain status")
        status, raw_name = record[:2], record[3:]
        old_name = b""
        if b"R" in status or b"C" in status:
            if i >= len(records) or not records[i]:
                raise ValueError("missing rename/copy source")
            old_name = records[i]
            i += 1
        name = raw_name.decode("utf-8", "surrogateescape")
        if status == b"??" and re.fullmatch(r"\.adk_exec_[0-9a-f]{8}\.py", name):
            continue
        count += 1
        if count > 256:
            raise ValueError("too many changed files for a bounded gate")
        h.update(status + b"\0" + raw_name + b"\0" + old_name + b"\0")
        path = root / name
        if path.is_symlink():
            h.update(b"SYMLINK:" + os.fsencode(os.readlink(path)))
        elif path.is_file():
            with path.open("rb") as stream:
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    hashed_bytes += len(chunk)
                    if hashed_bytes > MAX_HASH_BYTES:
                        raise ValueError("changed files exceed bounded fingerprint size")
                    h.update(chunk)
        else:
            h.update(b"DELETED")
    staged = git(root, "diff", "--cached", "--binary")
    if len(staged) > MAX_HASH_BYTES:
        raise ValueError("staged diff exceeds bounded fingerprint size")
    h.update(hashlib.sha256(staged).digest())
    return h.hexdigest(), count


def state_file(root):
    directory = Path(git(root, "rev-parse", "--absolute-git-dir").decode().strip()).resolve()
    return directory / "submission10_repair_state.json"


def references_runtime_behavior(code):
    """Reject obviously vacuous assertions; not a semantic coverage proof."""
    tree = ast.parse(code)
    return any(
        isinstance(node, ast.Assert) and any(
            isinstance(child, (ast.Name, ast.Call, ast.Attribute, ast.Subscript))
            for child in ast.walk(node.test))
        for node in ast.walk(tree))


def decide(workspace, requirements, covers, code, module=None, timeout=12,
           time_left=None, calls_left=None):
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir() or len(str(root)) > 1024:
        raise ValueError("invalid workspace")
    required, covered_by_check = identifiers(requirements), identifiers(covers)
    if not required or not set(covered_by_check).issubset(required):
        raise ValueError("define 1-4 requirements and test only named requirements")
    if ((time_left is not None and time_left < 0) or
            (calls_left is not None and calls_left < 0)):
        raise ValueError("remaining budgets must be nonnegative")
    if not 1 <= timeout <= 20:
        raise ValueError("timeout must be 1-20 seconds")
    patch_fingerprint, changed_count = fingerprint(root)
    location = state_file(root)
    state = json.loads(location.read_text()) if location.exists() else {
        "requirements": required, "repairs_requested": 0, "pending_repair": False,
        "alternative_checks_requested": 0, "checked": {}, "last_failure": None}
    if state["requirements"] != required:
        raise ValueError("requirement IDs cannot change during the task")
    if state.get("patch_fingerprint") != patch_fingerprint:
        state["checked"] = {}  # An edit invalidates earlier successes.
        if state.get("pending_repair"):
            state["repairs_requested"] += 1
            state["pending_repair"] = False
        state["patch_fingerprint"] = patch_fingerprint
        state["last_failure"] = None
    enough_budget = (time_left is None or time_left >= MIN_SECONDS_FOR_CHECK) and (
        calls_left is None or calls_left >= MIN_CALLS_FOR_CHECK)
    result = {"next_action": None, "reason": "", "verification": None,
              "repairs_requested": state["repairs_requested"],
              "max_repairs": MAX_REPAIRS, "patch_changed": changed_count > 0,
              "budget_sufficient": enough_budget,
              "scope": "Requirement IDs are model-reported claims, not measured coverage. Passing inline assertions do not prove issue resolution or hidden-test success."}
    if not enough_budget:
        result.update(next_action="audit_then_submit",
                      reason="reserve remaining time and calls for patch capture")
    else:
        result["verification"] = check(root, code, module, timeout)
        test = result["verification"]
        outcome = test["outcome"]
        if outcome == "passed" and test["assertions_passed"] > 0 and test["completed"] and not references_runtime_behavior(code):
            if state["alternative_checks_requested"] == 0:
                state["alternative_checks_requested"] += 1
                result.update(next_action="alternative_check",
                              reason="vacuous assertion: no runtime value was inspected")
            else:
                result.update(next_action="audit_then_submit",
                              reason="vacuous check and alternative-check allowance exhausted")
        elif outcome == "passed" and test["assertions_passed"] > 0 and test["completed"]:
            state["pending_repair"] = False  # A passing recheck without an edit was not a repair.
            for name in covered_by_check:
                state["checked"][name] = hashlib.sha256(code.encode()).hexdigest()[:12]
            missing = [r for r in required if r not in state["checked"]]
            result.update(
                next_action="verify_uncovered" if missing else "audit_then_submit",
                reason=("some acceptance cases remain unverified" if missing
                        else "all model-declared requirement IDs have passing checks; semantic coverage is not proven"))
        elif outcome == "failed":
            signature = hashlib.sha256(
                (patch_fingerprint + code + str(module)).encode()).hexdigest()
            if state["last_failure"] == signature:
                result.update(next_action="audit_then_submit",
                              reason="identical failed check on unchanged patch")
            elif state["repairs_requested"] >= MAX_REPAIRS:
                result.update(next_action="audit_then_submit",
                              reason="repair attempt limit reached")
            else:
                state["pending_repair"] = True
                result.update(next_action="repair_if_code_defect",
                              reason="check failed; diagnose before editing")
            state["last_failure"] = signature
        elif state["alternative_checks_requested"] == 0:
            state["alternative_checks_requested"] += 1
            result.update(next_action="alternative_check",
                          reason=f"{outcome} is inconclusive, not a pass")
        else:
            result.update(next_action="audit_then_submit",
                          reason="alternative-check allowance exhausted")
    result["repairs_requested"] = state["repairs_requested"]
    result["pending_repair"] = state.get("pending_repair", False)
    result["covered"] = sorted(state["checked"])
    result["uncovered"] = [r for r in required if r not in state["checked"]]
    temporary = location.with_name(location.name + ".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
    temporary.replace(location)
    return result



def _load_graph(root):
    """Read task-local structural evidence; no imports from sibling skill scripts."""
    target = state_file(root).with_name("submission12_contract_graph.json")
    if not target.is_file() or target.stat().st_size > 20000:
        return None
    graph = json.loads(target.read_text())
    if graph.get("version") != 1 or not isinstance(graph.get("nodes"), list):
        return None
    if len(graph["nodes"]) > 18 or len(graph.get("edges", [])) > 48:
        return None
    return graph


def _public_invoked(code, symbol):
    """Conservative static check for an actual call to the named public API.

    This is a guard against falsely accepting an *internal-only* passing test,
    not evidence of complete semantic test coverage.
    """
    if not symbol or not all(part.isidentifier() for part in symbol.split(".")):
        return False
    final = symbol.split(".")[-1]
    for node in ast.walk(ast.parse(code, filename="<graph-public-check>")):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == final:
            return True
        if isinstance(func, ast.Attribute) and func.attr == final:
            return True
    return False


def _source_freshness(root, graph):
    hashes = graph.get("source_hashes", {})
    if not isinstance(hashes, dict) or len(hashes) > 6:
        return False
    for name, expected in hashes.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            return False
        path = root / name
        if (path.is_symlink() or not path.is_file()
                or not path.resolve().is_relative_to(root)):
            return False
        if path.stat().st_size > 120000:
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            return False
    return bool(hashes)


def _focus(graph, fresh):
    """Choose a bounded structural investigation target; never label as a defect."""
    if not fresh:
        return {"reason": "graph_stale_after_edit", "candidate": None,
                "confidence": "unknown"}
    publics = graph.get("public_nodes", [])[:4]
    nodes = {n["id"]: n for n in graph.get("nodes", [])
             if isinstance(n, dict) and isinstance(n.get("id"), str)}
    for item in publics:
        node = nodes.get(item)
        if node and node.get("accepts") in ("not_declared", "positional_only"):
            return {"reason": "public_signature_not_keyword_capable",
                    "candidate": item, "confidence": "static_candidate"}
    for edge in graph.get("edges", []):
        if edge.get("from") in publics and edge.get("mode") == "other_call":
            return {"reason": "public_call_may_not_forward_named_option",
                    "candidate": edge.get("callee"), "confidence": "static_candidate"}
    return {"reason": "verify_public_contract_in_runtime",
            "candidate": publics[0] if publics else None,
            "confidence": "static_candidate" if publics else "unknown"}


def graph_decide(workspace, requirements, covers, code, module=None, timeout=12,
                 time_left=None, calls_left=None):
    """Single invocation: structural graph state + actual check + repair decision.

    Simple/unmapped tasks fall back to the proven Submission 10 gate.
    """
    root = Path(workspace).resolve(strict=True)
    desired, declared = identifiers(requirements), identifiers(covers)
    if not desired or not set(declared).issubset(desired):
        raise ValueError("invalid requirement coverage")
    try:
        graph = _load_graph(root)
    except (ValueError, OSError, json.JSONDecodeError, KeyError, TypeError):
        graph = None

    if graph is None or set(graph.get("requirements", [])) != set(desired):
        result = decide(workspace, requirements, covers, code, module,
                        timeout, time_left, calls_left)
        result["controller"] = "submission10_fallback"
        result["graph"] = {"available": False, "reason": "no_matching_graph"}
        return result

    symbol = graph.get("public_symbol")
    # A passing Submission 10 internal-only check must not remain accepted
    # after graph guidance is activated for this task.
    evidence_path = state_file(root)
    if evidence_path.exists():
        previous = json.loads(evidence_path.read_text())
        marker = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()[:16]
        if previous.get("graph_marker") != marker:
            previous["checked"] = {}
            previous["graph_marker"] = marker
            scratch = evidence_path.with_name(evidence_path.name + ".tmp")
            scratch.write_text(json.dumps(previous, sort_keys=True) + "\n")
            scratch.replace(evidence_path)
    matched = _public_invoked(code, symbol)
    # Multiple claimed coverage IDs require independent assertions, not one
    # generic low-level check. The assertion runner still cannot prove semantics.
    assertions = sum(isinstance(node, ast.Assert)
                     for node in ast.walk(ast.parse(code)))
    public_claims = declared if matched and assertions >= len(declared) else []
    result = decide(workspace, requirements, ",".join(public_claims),
                    code, module, timeout, time_left, calls_left)
    fresh = _source_freshness(root, graph)
    focus = _focus(graph, fresh)
    result["controller"] = "graph_guided_repair"
    result["graph"] = {
        "available": True, "fresh": fresh, "public_symbol": symbol,
        "public_call_present": matched, "test_nodes_verified": public_claims
        if (result.get("verification") or {}).get("outcome") == "passed" else [],
        "nodes": len(graph.get("nodes", [])),
        "candidate_edges": len(graph.get("edges", [])),
        "requirements": [{"id": name, "status":
                          "public_checked" if name in result["covered"]
                          else "unverified"}
                         for name in desired],
        "focus": focus,
        "limitations": "Static edges and model-supplied requirement IDs are not proof of semantic coverage.",
    }
    if (result["next_action"] == "verify_uncovered"
            and (result.get("verification") or {}).get("outcome") == "passed"
            and declared and not public_claims):
        result["next_action"] = "verify_public_api"
        result["reason"] = "low-level pass cannot discharge public contract; execute public entry"
    elif result["next_action"] == "repair_if_code_defect":
        result["graph"]["focus"] = focus
        result["reason"] += "; graph focus is a hypothesis, not proof of a defect"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--covers", default="")
    parser.add_argument("--code", required=True)
    parser.add_argument("--module")
    parser.add_argument("--timeout", type=int, default=12)
    parser.add_argument("--time-left", "--time_left", dest="time_left", type=float)
    parser.add_argument("--calls-left", "--calls_left", dest="calls_left", type=int)
    args = parser.parse_args()
    try:
        result = graph_decide(args.workspace, args.requirements, args.covers,
                              args.code, args.module, args.timeout,
                              args.time_left, args.calls_left)
    except (OSError, ValueError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired, json.JSONDecodeError, SyntaxError,
            TypeError, KeyError) as exc:
        result = {"next_action": "audit_then_submit",
                  "error": str(exc)[:400],
                  "reason": "graph decision gate unavailable; protect patch capture budget"}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
