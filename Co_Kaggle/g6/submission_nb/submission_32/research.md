# Submission 32 evidence review (2026-11-02)

This is a design investigation, **not** a claim of benchmark gains.
Use primary research and official project docs as evidence for *mechanisms*;
do not transfer published scores to E4B or this Kaggle environment.
Do not copy benchmark solutions or gold patches into the submission.

## External research and the concrete implication

| Source | Evidence | Transfer to S32 | Limitations |
|---|---|---|---|
| Yang et al., *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*, NeurIPS 2024, [paper](https://proceedings.nips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html), [ACI doc](https://github.com/SWE-agent/SWE-agent/blob/main/docs/background/aci.md) | Source viewer shows bounded context; syntax-aware edit rejection; concise search results. The interface itself materially affects performance. | Read 30–65 lines, search narrow symbols, validate Python syntax before guarded fallback. | Their bespoke ACI cannot be installed through the declarative competition compiler; prompts are not host-side enforcement. |
| Aider, [edit-format docs](https://aider.chat/docs/more/edit-formats.html) and [editing leaderboard](https://aider.chat/docs/leaderboards/edit) | Search/replace blocks economize on tokens; whole-file outputs are often easier but costly and can cause large edits; format reliability is a separate metric from correctness. | Use native short `old_string`/`new_string` first, never whole-file replacement; measure failed edit calls separately from task resolution. | Their reported model scores and edit formats were not tested with Gemma E4B here. |
| Xia et al., *Agentless: Demystifying LLM-based Software Engineering Agents*, [arXiv 2407.01489](https://arxiv.org/abs/2407.01489) | Simple localization → repair → patch validation performed strongly on SWE-bench Lite; complexity is not automatically beneficial. | One sequential reasoning/execution loop, not ADK SequentialAgent/LoopAgent or subagents. | Agentless uses its own programmatic framework; cannot import it into competition ZIP. |
| SWE-agent team, [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) | Minimal bash-based agent baseline demonstrates that a narrow interface and transparent trajectory can be useful. | Keep native 5-tool interface and one agent, not another nested wrapper/tool signature. | Pure bash agents use different models and sandbox constraints. |
| NVIDIA et al., [Nemotron-CORTEXA, ICML 2025](https://proceedings.mlr.press/v267/sohrabizadeh25a.html) | Precise localization and context selection influence repair success. | Prioritize symbol → actual source → caller, rather than graph similarity-only or broad searches. | Its embedding/retrieval model is not competition-permitted; S31 showed overuse of supplied graph tools. |
| OpenAI, [SWE-bench Verified methodology](https://openai.com/index/introducing-swe-bench-verified/) | Distinguishes fail-to-pass from pass-to-pass, and flags environment and underspecification problems. | Require an issue-specific behavior check and ordinary/control behavior; distinguish tests blocked from a correct solution. | Local checks are not the hidden scorer. |
| OpenAI, [2026 benchmark validity analysis](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | Finds contamination and tests that reject correct solutions in a subset of SWE-bench Verified. | Treat selected 15 public tasks as an engineering screen, not a guaranteed hidden-score predictor; retain held-out tasks. | Different benchmark and human-audit methodology from this competition. |

## Competition limits verified from this repository

- [Official harness guide](../../competition_material/competition_data/HARNESS_README.md): declarative `agent.yaml` + prompts + eval config; no arbitrary Python agent entrypoint; native `run_command`, `read_file`, `edit_file`, `write_file`, `get_status`, `submit_patch`. Offline sandbox; each tool call counts unless excluded. `submit_patch` expects **no arguments**.
- `run_command` executes in the sandbox and has a bounded stdout. Because it allows arbitrary shell commands, permissions on `edit_file` alone cannot enforce edit safety; no custom host-side gate is available in declarative ZIP.
- S32 uses the allowed **standard-library command through native `run_command`**, NOT a shipped custom Python entrypoint or `run_skill_script`.
- No graph tool: S30 exposed three but never used them; S31 made 91 visible calls on 13 detailed traces yet solved only 1/15.
- No `write_file`: S30 used it to overwrite `fastapi/routing.py`, producing a 187,143-character patch. This narrows the hypothesis to **existing-file repairs only**; new-file tasks may fail.

## Local empirical controls: same selected 15 S1 tasks

| Version | Resolved | Empty patches | Failed native edits | Largest patch |
|---|---:|---:|---:|---:|
| S29 | 3/15 | 3 | 86 | 2,893 chars |
| S30 | 4/15 | 1 | 46 | 187,143 chars |
| S31 | 1/15 | 4 | at least 56 from detailed trace | 2,456 chars |

Source: `tests_on_tasks/diagnostics/submission-{29,30,31}-zip_s1TasksPrintouts.txt`.
The S31 printed log has detailed tool traces for 13/15, so per-event aggregate
counts are *lower bounds*; the score summary covers all 15.

### S32 hypothesis and causal isolation

1. **Primary artifact:** one native E4B agent, short source slices, direct `edit_file`, optional
   *single-line-only* guarded fallback after TWO native failed anchors.
   This fallback includes original-line equality, file size, replacement
   length, `ast.parse`, same-directory temporary file and `os.replace`.
2. **Control artifact:** byte-identical agent configuration and budget, no
   guarded fallback; its instruction explicitly forbids shell mutation.
3. Compare both candidates with the S29/S30 wins on exactly the same pilot
   and 15 tasks, including **two repeats each** and **the same time-limit**.
4. Decision criterion: >=5/15 resolved on both repeats, retain all S29 wins
   and at least one new S30 success; reject unexpected whole-file diffs,
   edit loops, empty patches and any shell mutation lacking full ATIF review.
   A local syntax check is not a semantic correctness check.
5. Keep the reported candidate as experimental until real E4B tests; no
   model inference or compiler smoke test is claimed merely from ZIP generation.

### What is enforceable versus guidance

**Enforced by the submitted tool list:** no `write_file`, no graph tools,
no skill dispatch, only a single LlmAgent, fixed model/config limits.

**Enforced by fallback command *if correctly copied and executed*:** size
bound, line equality, single-line replacement, Python syntax and atomic
same-directory replace. The command operates only on an existing `.py`
file; this is a restricted fallback, not a general source editor.

**Not enforced by the competition host:** model's selection of the template,
two-failure threshold, path scope via `run_command`, per-step token/call
budgets, correctness of asserted expected behavior. A modified or malicious
shell command can bypass the guards. Thus manual ATIF and final diff review
remain mandatory. A command-printout alone cannot certify guard compliance.

## Why NOT to repeat prior failed approaches

- S25: replaced native tools with a `run_skill_script` interface that E4B
  regularly called with incorrect arguments and script paths; 0/15.
- S28: uncontrolled shell edits led to huge unexpected changes.
- S30: optional graph tools used 0 times; `write_file` destructive overwrite.
- S31: graph-only read restrictions triggered missing-tool errors,
  multi-agent pipeline repetition and premature verifier claims; 1/15.

**Verdict:** S32 tests a narrowly specified execution fallback and measures
its benefit *against an otherwise identical control*, not an unbounded
new agent system. Do not promote until real matched traces and correctness.
