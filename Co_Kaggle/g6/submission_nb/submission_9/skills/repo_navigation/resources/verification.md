# Verify the requested behavior, not your own assumption

Write the expected result before the edit when feasible. Run the same discriminating check after the edit. Check the public entry point and a neighboring valid case; testing only the newly added private branch can miss forwarding and lifecycle errors. A fresh process is required when testing import changes or module-level caches.

Use scripts/check.py for short assertions. Supply the module under repair so its origin is checked. It sets PYTHONPATH to the observed workspace/src and workspace; dependencies still use the normal installed environment. A workspace import failure is evidence to investigate, not permission to fall back silently to an installed copy. Arbitrary subprocesses you write yourself must also use the workspace environment.

Read the script's JSON `outcome` and child output, not just the outer skill tool's success. `passed` means the bounded program completed with at least one assertion in its source, not that assertions covered the requirement or hidden tests. `exit_zero_without_assertions`, `incomplete`, `timeout`, `failed` and `import_origin_error` are not behavioral passes. Never swallow assertions or exit early to obtain success.

For existing pytest checks, select confirmed paths and use `python -m pytest` from the workspace with its source root first. Do not edit collected tests, conftest, pytest.ini, pyproject.toml or dependency versions to obtain a pass. The grader resets protected files. Local scratch assertions can live in stdin or /tmp; they are not submitted verification tests.

Classify failures: an assertion or implementation exception can identify a defect; a missing external dependency, recursive fixture, zero collected tests or timeout makes that check inconclusive. A missing module that the issue requires may instead be an omitted implementation. Look at the first relevant failure; do not automatically label every collection error as environmental.

After import deferral, execute the affected branch as well as importing the module. After a formatting fix, assert exact output including trailing separators. After a new option, exercise both its public default and explicit override. For HTTP behavior, construct requests or use local test clients; external network availability is not part of the check.
