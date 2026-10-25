# Recover without spending the remaining budget

Check the live native schema: read_file takes filepath/start_line/end_line; edit_file takes filepath/old_string/new_string; write_file takes filepath/content; run_command takes command. Use plain strings, not JSON encoded a second time. Keep source fragments small and unique. A missing-field rejection did not apply an edit. A match error requires rereading the current source.

If two corrected native attempts fail, issue one short shell command with a quoted heredoc. Substitute only evidence-backed strings:

```python
from pathlib import Path
p = Path("relative/source.py")
source = p.read_text()
old = "one unique source fragment"
new = "the replacement fragment"
assert source.count(old) == 1, "source changed or match is ambiguous"
p.write_text(source.replace(old, new, 1))
```

Pass this program as `python - <<'PY' ... PY` in run_command. Assert before writing; inspect its diff. Never repair escaping by globally replacing backslashes. Never overwrite a source file with truncated tool output. Do not put helper-script scratch files in the repository. Use stdin or create scratch under /tmp through run_command, never through workspace-only write_file.

A failed grep with exit 1 may simply mean no matches. Change the hypothesis or search term once; do not interpret it as a broken repository. A second test failure with the same import/fixture error calls for a smaller check, not a repeated whole suite.
