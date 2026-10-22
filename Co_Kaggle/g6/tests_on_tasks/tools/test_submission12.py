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
                self.assertIn("skills/repo-navigation/references/contracts.md", items)
                self.assertNotIn("README.md", items)
                self.assertNotIn("skills/repo_navigation", z.read("agent.yaml").decode())
                self.assertIn("gemma-4-31b-it-qat-w4a16-ct", z.read("agent.yaml").decode())
                self.assertLess(sum(z.getinfo(n).file_size for n in items), 3 * 1024**3)

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
