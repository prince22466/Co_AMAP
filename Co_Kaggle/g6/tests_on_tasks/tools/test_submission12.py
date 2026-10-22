"""CPU-only, non-benchmark tests for Submission 12's optional AST tracer and release."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from build_submission12 import BASE, CANDIDATE, build, validate

SOURCE = CANDIDATE / "skills/repo_navigation/scripts/contract_trace.py"
spec = importlib.util.spec_from_file_location("submission12_contract_trace", SOURCE)
tracer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracer)

GRAPH_GATE = CANDIDATE / "skills/repo_navigation/scripts/graph_repair_gate.py"
graph_spec = importlib.util.spec_from_file_location("submission12_graph_gate", GRAPH_GATE)
graph_gate = importlib.util.module_from_spec(graph_spec)
graph_spec.loader.exec_module(graph_gate)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="submission12-test-")
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "workspace"
        self.workspace.mkdir()

    def write(self, path, text):
        target = self.workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        return target


class TraceTests(Fixture):
    def test_explicit_signature_and_forwarding_to_constructor(self):
        self.write("src/api.py", """class Router:
    def include(self, option=False):
        return Route(option=option)
class Route:
    def __init__(self, option=False):
        self.option = option
""")
        result = tracer.trace(self.workspace, "src/api.py", "option")
        self.assertEqual(result["outcome"], "structural_candidates_only")
        index = {row["symbol"]: row for row in result["functions"]}
        self.assertEqual(index["Router.include"]["accepts"], "explicit")
        self.assertEqual(index["Router.include"]["forwards"][0]["callee"], "Route")
        self.assertEqual(index["Router.include"]["forwards"][0]["mode"], "parameter_value")
        self.assertEqual(index["Route.__init__"]["accepts"], "explicit")

    def test_missed_public_forwarding_is_visible_without_claiming_runtime(self):
        self.write("api.py", """def public(option=False):
    return private()
def private(option=False):
    return option
""")
        result = tracer.trace(self.workspace, "api.py", "option")
        public = next(row for row in result["functions"] if row["symbol"] == "public")
        self.assertEqual(public["forwards"], [])
        self.assertIn("do not prove runtime", result["limitations"])

    def test_positional_only_does_not_claim_keyword_compatibility(self):
        self.write("api.py", """def public(option, /):
    return option
def private(option=None):
    return option
""")
        result = tracer.trace(self.workspace, "api.py", "option")
        rows = {row["symbol"]: row for row in result["functions"]}
        self.assertEqual(rows["public"]["accepts"], "positional_only")
        self.assertEqual(rows["private"]["accepts"], "explicit")

    def test_requested_public_entry_survives_large_result_truncation(self):
        source = "\n".join("def f%d(option=None): return option" % i
                            for i in range(50))
        self.write("api.py", source + "\ndef public_api(): return 1\n")
        result = tracer.trace(self.workspace, "api.py", "option",
                              symbol="public_api")
        self.assertEqual(result["functions_found"], 51)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["functions"][0]["symbol"], "public_api")
        self.assertEqual(result["functions"][0]["accepts"], "not_declared")

    def test_deep_ast_traversal_reports_inconclusive_instead_of_crashing(self):
        self.write("api.py", "def f(option): return option\n")
        with patch.object(tracer._Definitions, "visit", side_effect=RecursionError):
            result = tracer.trace(self.workspace, "api.py", "option")
        self.assertEqual(result["functions"], [])
        self.assertEqual(result["warnings"][0]["issue"], "traversal_too_deep")

    def test_kwargs_forwarding_is_reported_as_opaque(self):
        self.write("api.py", """class Router:
    def wrapper(self, **kwargs):
        return self.target(**kwargs)
    def target(self, option=False):
        return option
""")
        result = tracer.trace(self.workspace, "api.py", "option")
        wrapper = next(row for row in result["functions"] if row["symbol"] == "Router.wrapper")
        self.assertEqual(wrapper["accepts"], "variadic_kwargs")
        self.assertEqual(wrapper["forwards"][0]["mode"], "opaque_kwargs")

    def test_forwarding_truncation_is_explicit_and_counts_all_candidates(self):
        source = "def router(option=None):\n" + "    sink(option=option)\n" * 9
        self.write("api.py", source)
        result = tracer.trace(self.workspace, "api.py", "option")
        router = next(row for row in result["functions"] if row["symbol"] == "router")
        self.assertEqual(len(router["forwards"]), tracer.MAX_FORWARDERS_PER_FUNCTION)
        self.assertTrue(router["forwarding_truncated"])
        self.assertEqual(router["forwarding_candidates"], 9)
        self.assertEqual(result["forwarding_candidates"], 9)

    def test_async_function_and_symbol_can_surface_missing_parameter(self):
        self.write("api.py", """async def serve():
    return 1
""")
        result = tracer.trace(self.workspace, "api.py", "option", symbol="serve")
        self.assertEqual(result["functions"][0]["accepts"], "not_declared")
        self.assertEqual(result["functions"][0]["symbol"], "serve")

    def test_untrusted_source_is_never_executed(self):
        self.write("api.py", """raise RuntimeError('must not run')
def entry(option=None):
    return option
""")
        result = tracer.trace(self.workspace, "api.py", "option")
        self.assertEqual(result["functions"][0]["symbol"], "entry")

    def test_invalid_or_outside_paths_rejected(self):
        self.write("api.py", "def f(option): pass\n")
        for paths in ("../outside.py", "/etc/passwd.py", "api.txt", "api.py,api.py"):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                tracer.trace(self.workspace, paths, "option")
        with self.assertRaises(ValueError):
            tracer.trace(self.workspace, "api.py", "not-valid")
        with self.assertRaises(ValueError):
            tracer.trace(self.workspace, ",".join("f%d.py" % i for i in range(7)), "option")

    def test_symlink_to_outside_workspace_is_rejected(self):
        outside = Path(self.temp.name) / "out.py"
        outside.write_text("def f(option): pass\n")
        (self.workspace / "link.py").symlink_to(outside)
        with self.assertRaises(ValueError):
            tracer.trace(self.workspace, "link.py", "option")

    def test_missing_bad_syntax_and_oversized_sources_do_not_crash(self):
        self.write("broken.py", "def broken(option:\n")
        self.write("huge.py", "x = 1\n" + " " * (tracer.MAX_BYTES_PER_FILE + 1))
        r = tracer.trace(self.workspace, "missing.py,broken.py,huge.py", "option")
        self.assertEqual([w["issue"] for w in r["warnings"]],
                         ["missing_file", "parse_error", "file_too_large"])

    def test_deep_parser_failure_is_reported_without_crashing(self):
        self.write("api.py", "def f(option=None): pass\n")
        with patch.object(tracer.ast, "parse", side_effect=RecursionError):
            result = tracer.trace(self.workspace, "api.py", "option")
        self.assertEqual(result["functions"], [])
        self.assertEqual(result["warnings"][0]["issue"], "parse_error")

    def test_bounded_output_for_many_functions(self):
        self.write("api.py", "\n".join("def f%d(option=None): return option" % i
                                          for i in range(80)))
        r = tracer.trace(self.workspace, "api.py", "option")
        self.assertTrue(r["truncated"])
        self.assertEqual(r["functions_found"], 80)
        self.assertLessEqual(len(json.dumps(r)), tracer.MAX_OUTPUT)

    def test_cli_invocation_produces_machine_readable_result(self):
        self.write("api.py", "def exposed(option=False): return option\n")
        result = subprocess.run(
            [sys.executable, str(SOURCE), "--workspace", str(self.workspace),
             "--paths", "api.py", "--parameter", "option"],
            capture_output=True, text=True, check=True, timeout=10)
        decoded = json.loads(result.stdout)
        self.assertEqual(decoded["functions"][0]["symbol"], "exposed")
        self.assertLess(len(result.stdout), 5000)


class GraphLoopTests(Fixture):
    def setUp(self):
        super().setUp()
        self.write("src/example_pkg/__init__.py",
                   "def private(option=False): return option\n"
                   "def public(option=False): return private()\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "baseline")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.workspace), *args])

    def build_graph(self, requirements="behavior,default"):
        return tracer.trace(self.workspace, "src/example_pkg/__init__.py",
                            "option", symbol="public",
                            requirements=requirements, store=True)

    def decide(self, code, covers="behavior", requirements="behavior,default",
               time_left=180, calls_left=30):
        return graph_gate.graph_decide(
            self.workspace, requirements, covers, code,
            module="example_pkg", time_left=time_left, calls_left=calls_left)

    def test_graph_persists_outside_patch_and_detects_missing_forward(self):
        traced = self.build_graph()
        self.assertTrue(traced["graph"]["stored"], traced)
        path = graph_gate.state_file(self.workspace).with_name(
            "submission12_contract_graph.json")
        self.assertTrue(path.exists())
        self.assertNotIn("submission12_contract_graph.json",
                         self.git("status", "--porcelain").decode())
        graph = json.loads(path.read_text())
        self.assertEqual(graph["requirements"], ["behavior", "default"])
        self.assertEqual(len(graph["public_nodes"]), 1)
        public_edges = [e for e in graph["edges"]
                        if e["from"] in graph["public_nodes"]]
        self.assertTrue(any(e["callee"] == "private"
                            and e["mode"] == "other_call" for e in public_edges))

    def test_internal_pass_does_not_discharge_public_requirement(self):
        self.build_graph()
        first = self.decide("from example_pkg import private\nassert private(True) is True")
        self.assertEqual(first["next_action"], "verify_public_api", first)
        self.assertEqual(first["covered"], [])
        self.assertEqual(first["controller"], "graph_guided_repair")
        self.assertEqual(first["graph"]["focus"]["reason"],
                         "public_call_may_not_forward_named_option")

    def test_graph_failure_then_edit_then_public_pass(self):
        self.build_graph()
        failed = self.decide("from example_pkg import public\nassert public(True) is True")
        self.assertEqual(failed["next_action"], "repair_if_code_defect", failed)
        self.assertEqual(failed["repairs_requested"], 0)
        self.write("src/example_pkg/__init__.py",
                   "def private(option=False): return option\n"
                   "def public(option=False): return private(option=option)\n")
        passed = self.decide(
            "from example_pkg import public\nassert public(True) is True",
            covers="behavior")
        self.assertEqual(passed["verification"]["outcome"], "passed", passed)
        self.assertEqual(passed["next_action"], "verify_uncovered", passed)
        self.assertEqual(passed["repairs_requested"], 1)
        self.assertFalse(passed["graph"]["fresh"])
        self.assertEqual(passed["graph"]["focus"]["reason"], "graph_stale_after_edit")
        self.assertEqual(passed["covered"], ["behavior"])

    def test_public_checks_accept_only_claimed_requirements(self):
        self.build_graph()
        self.write("src/example_pkg/__init__.py",
                   "def private(option=False): return option\n"
                   "def public(option=False): return private(option=option)\n")
        result = self.decide(
            "from example_pkg import public\n"
            "assert public(True) is True\n"
            "assert public() is False",
            covers="behavior,default")
        self.assertEqual(result["next_action"], "audit_then_submit", result)
        self.assertEqual(result["covered"], ["behavior", "default"])

    def test_one_assertion_cannot_claim_two_requirements(self):
        self.build_graph()
        self.write("src/example_pkg/__init__.py",
                   "def private(option=False): return option\n"
                   "def public(option=False): return private(option=option)\n")
        result = self.decide(
            "from example_pkg import public\nassert public(True) is True",
            covers="behavior,default")
        self.assertEqual(result["covered"], [], result)
        self.assertEqual(result["next_action"], "verify_public_api", result)

    def test_low_budget_never_dereferences_missing_verification(self):
        self.build_graph()
        result = self.decide("assert True", time_left=20, calls_left=2)
        self.assertEqual(result["next_action"], "audit_then_submit", result)
        self.assertIsNone(result["verification"])

    def test_missing_graph_uses_exact_submission10_fallback(self):
        result = self.decide(
            "from example_pkg import private\nassert private(True) is True",
            requirements="behavior", covers="behavior")
        self.assertEqual(result["controller"], "submission10_fallback")
        self.assertEqual(result["next_action"], "audit_then_submit")

    def test_activating_graph_revokes_previous_internal_only_pass(self):
        old = graph_gate.decide(
            self.workspace, "behavior", "behavior",
            "from example_pkg import private\nassert private(True) is True",
            module="example_pkg")
        self.assertEqual(old["covered"], ["behavior"])
        self.build_graph(requirements="behavior")
        current = self.decide(
            "from example_pkg import private\nassert private(True) is True",
            requirements="behavior")
        self.assertEqual(current["covered"], [], current)
        self.assertEqual(current["next_action"], "verify_public_api", current)

    def test_mismatched_requirement_graph_uses_fallback(self):
        self.build_graph(requirements="behavior")
        result = self.decide(
            "from example_pkg import private\nassert private(True) is True",
            requirements="behavior,default")
        self.assertEqual(result["controller"], "submission10_fallback")

    def test_coverage_graph_links_api_requirement_and_executed_check(self):
        self.build_graph(requirements="behavior")
        self.write("src/example_pkg/__init__.py",
                   "def private(option=False): return option\n"
                   "def public(option=False): return private(option=option)\n")
        result = self.decide(
            "from example_pkg import public\nassert public(True) is True",
            requirements="behavior")
        edges = result["graph"]["coverage_edges"]
        self.assertTrue(any(e["kind"] == "requires_public_api" for e in edges))
        self.assertTrue(any(e["kind"] == "passed_public_assertion" for e in edges))
        self.assertEqual(result["graph"]["test_node"]["outcome"], "passed")

    def test_missing_public_node_does_not_activate_graph_mode(self):
        self.build_graph(requirements="behavior")
        graph_path = graph_gate.state_file(self.workspace).with_name(
            "submission12_contract_graph.json")
        payload = json.loads(graph_path.read_text())
        payload["public_nodes"] = []
        graph_path.write_text(json.dumps(payload))
        out = self.decide(
            "from example_pkg import private\nassert private(True) is True",
            requirements="behavior")
        self.assertEqual(out["controller"], "submission10_fallback")

    def test_graph_controller_cli_invocation(self):
        self.build_graph(requirements="behavior")
        cmd = [sys.executable, str(GRAPH_GATE), "--workspace", str(self.workspace),
               "--requirements", "behavior", "--covers", "behavior",
               "--module", "example_pkg",
               "--code", "from example_pkg import private\nassert private(True) is True",
               "--time_left", "170", "--calls_left", "32"]
        done = subprocess.run(cmd, capture_output=True, text=True,
                              check=True, timeout=20)
        result = json.loads(done.stdout)
        self.assertEqual(result["next_action"], "verify_public_api", result)
        self.assertEqual(result["controller"], "graph_guided_repair")


class PackageTests(unittest.TestCase):
    def test_single_variable_release_closure(self):
        files, preserved = validate()
        self.assertGreater(len(files), 15)
        self.assertGreater(len(preserved), 12)

    def test_repeatable_archive_and_competition_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "one.zip"
            b = Path(tmp) / "two.zip"
            build(a)
            build(b)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with zipfile.ZipFile(a) as z:
                items = set(z.namelist())
                self.assertIn("agent.yaml", items)
                self.assertIn("skills/repo-navigation/scripts/contract_trace.py", items)
                self.assertIn("skills/repo-navigation/scripts/repair_gate.py", items)
                self.assertIn("skills/repo-navigation/scripts/graph_repair_gate.py", items)
                self.assertIn("skills/repo-navigation/references/contracts.md", items)
                self.assertNotIn("README.md", items)
                self.assertNotIn("skills/repo_navigation", z.read("agent.yaml").decode())
                self.assertIn("gemma-4-31b-it-qat-w4a16-ct", z.read("agent.yaml").decode())
                self.assertLess(sum(z.getinfo(n).file_size for n in items), 3 * 1024**3)

    def test_builder_rejects_extra_agent_config_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate"
            shutil.copytree(CANDIDATE, candidate)
            agent = candidate / "agent.yaml"
            agent.write_text(agent.read_text() + "\noutput_key: injected\n")
            with self.assertRaises(ValueError):
                validate(root=candidate, baseline=BASE)

    def test_builder_rejects_unexpected_active_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate"
            shutil.copytree(CANDIDATE, candidate)
            (candidate / "eval_config.yaml").write_text("evaluation: {}\n")
            with self.assertRaises(ValueError):
                validate(root=candidate, baseline=BASE)

    def test_builder_rejects_additional_scripts_and_deleted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate"
            shutil.copytree(CANDIDATE, candidate)
            (candidate / "skills/repo_navigation/scripts/surprise.py").write_text("pass\n")
            with self.assertRaises(ValueError):
                validate(root=candidate, baseline=BASE)


if __name__ == "__main__":
    unittest.main()
