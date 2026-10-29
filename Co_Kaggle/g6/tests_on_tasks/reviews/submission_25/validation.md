# Submission 25 validation scope

This is a new agent-only E4B candidate. **No real E4B inference was available in
this workspace. There is no measured S25 task score or improvement claim.**

## Evidence driving the change

The [S24 archive](../../diagnostics/submission-24-zip_s1Tasks.zip) has SHA256
`ded8f8aa14c0ee95b07e9431f4ac04f8b4b2b7f9938a57de06e3832bc76a1e88`.
It solved only `rich_4077`: 1/15 versus S21/S22's 2/15 each. Its recorded evaluation
time was 49.6 minutes, with 394 charged tools and 550 observed model requests.
These include harness work; they are not a GPU billing measurement.

- `rich_3905`: a repeated second edit introduced invalid indentation. The audit
  found it, but the reviewer approved and the finalizer submitted it.
- `rich_3518`: failed ambiguous replacements and a shell edit left docstring-only
  changes; the reviewer claimed the intended field existed.
- `rich_3882`: invented API and zero collected tests; approved despite both.
- `rich_3454`: escaping/argument failures on replacement strings; empty result.
- `rich_3718`: 20 failed native edits out of 21 attempts; exhausted the turn budget.

These motivate mechanical protections and fewer stages. They do not establish
that the new prompting or tool interface improves model reasoning. No archived
task solution, ID, reference patch, or grading assertion is included in the ZIP.

## What local validation covers

Local result: **24 tests passed**, including both packaged official-runtime
scenarios. [Machine-readable evidence](validation.json) records the tested ZIP hash,
runtime versions and smoke conditions. The main smoke used 12 scripted model
responses and nine charged tools; these are fixture counts, not benchmark savings.

The unit and official-runtime suite exercises the packaged code with synthetic
repositories. It checks stale-range rejection, line-targeted editing despite
duplicate source text, malformed escaping, compile-time syntax validation,
missing edit arguments, repeat/attempt limits, passing-check rollback, interrupted
edit recovery, checks modifying source/tests, malformed/missing API probes,
real pytest fail-to-pass, zero cases, timeouts, protected paths, docstring-only
diffs, bounded read output, deterministic packaging, complete-run comparison,
pilot promotion, and explicit S25 archive selection in the shared evaluator.

Two tests use the actual official ADK compiler, skill executor, native submission
tool and subprocess sandbox. One exercises malformed wrapper recovery and a
baseline-fail -> repair -> pass -> rejected regression -> submit sequence; the
other verifies that an unavailable test does not discard a valid partial fix.
Responses are **scripted**: these tests measure runtime mechanics, not E4B's
ability to choose correct calls. The smoke reports include that scope explicitly.

Runtime: `adk-submission 0.2.13`, `adk-eval-core 0.1.0`, `swegemma 0.2.11`,
`google-adk 1.36.1`, `google-genai 2.11.0`, `litellm 1.82.4`. Local pytest fixtures
use `pytest 8.4.2`. Cache and compaction match the S24 runner. CI repeats the suite
and ensures rebuilding changes neither the committed ZIP nor its manifest.

## Review boundaries

The competition requires declarative agents and allows sandboxed skills. S25
does not register custom host callbacks, change harness tools, call other models,
train weights, or include adapters. The [organizer model update](../../../competition_material/Update_new_models.md)
permits the E4B QAT model used here; the older original 31B-only text is superseded.

The workbench's checks are cooperative safeguards, not an adversarial sandbox.
In particular, arbitrary probe Python can access other sandbox paths and Git
metadata. A successful probe may be too weak. Existing-file-only editing,
six-second checks, skill-dispatch dependence and journal recovery only on a
subsequent workbench call are explicit limits. Native submission remains available
without a passing test so incomplete attempts can still receive official grading.

## Real-model decision

Run the five-task pilot documented in [the candidate README](../../../submission_nb/submission_25/README.md).
Require >=3/5 with S24's `rich_4077` retained, <=1 empty patch and no agent errors
in every repeat before spending time on the full S1 set. Check skill dispatch and
final diffs, not only aggregate score.

The full target is >=6/15, >=5 net gains over S24, retaining all S24 wins and at
least three S21/S22 union wins, <=2 empty patches and no agent errors. Compare
matching model/runtime/budget manifests and every repeat. A single S1 development
run cannot establish significance or hidden-task generalization.
