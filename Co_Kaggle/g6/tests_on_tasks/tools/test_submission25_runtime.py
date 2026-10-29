"""Require the official compiler/ADK and exercise packaged skill execution."""
import asyncio
import unittest
from smoke_submission25 import run_smoke


class OfficialRuntimeTests(unittest.TestCase):
    def test_repair_dispatch_syntax_and_regression_rollback(self):
        result = asyncio.run(run_smoke())
        self.assertTrue(result['passed'], result)

    def test_unverified_partial_patch_can_still_submit(self):
        result = asyncio.run(run_smoke(partial=True))
        self.assertTrue(result['passed'], result)


if __name__ == '__main__':
    unittest.main()
