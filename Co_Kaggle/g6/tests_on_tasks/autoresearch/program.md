# Weight-Space Programming (WSP) Autoresearch — Kaggle CLI Edition

## Mission & Paradigm: Weight-Space Programming Only
You are performing **Weight-Space Programming (WSP)**—NOT supervised fine-tuning, NOT
backpropagation/gradient descent, and NOT prompt engineering.

Instead of modifying the frozen 4B base weights of `gemma-4-e4b-it-qat-w4a16-ct` or
editing the frozen `submission_nb\artifacts\submission_36.zip` prompt, you directly program the low-rank weight
matrices (`lora_A`, `lora_B`) of a fresh PEFT LoRA adapter
(`adapters/wsp_r1/adapter_model.safetensors`) targeting selected `mlp.down_proj`
residual-stream write heads:

$$\Delta W_\ell = B_\ell A_\ell = \sum_{i=1}^{r} \lambda_{\ell,i} \, u_{\ell,i} \, v_{\ell,i}^\top \quad (\text{with } \alpha = r)$$

- **Rows of `lora_A` ($v_{\ell,i}$):** Top-$k$ sparsified MLP neuron detectors from
  contrastive failure-vs-healthy activations (conditional trigger gate).
- **Columns of `lora_B` ($\lambda_{\ell,i} u_{\ell,i}$):** Orthogonalized residual-stream
  steering directions pushing out of localized failure modes (`edit_file_loop`,
  `premature_submit`).
- **Rank discipline:** Start at `RANK = 1`. Expand to `RANK = 2` only after a validated
  rank-1 plateau.

## Benchmark, Artifact Retention & Stopping Conditions
- **Frozen Base Agent:** `submission_36` (`repair_agent`, `gemma-4-e4b-it-qat-w4a16-ct`,
  `temperature = 0.0`, `thinking_budget = 512`, `4 min / 40 tools / 48 turns`).
- **Baseline Score (no LoRA):** **3 / 15** deterministic on `--split 15`.
- **Target Score:** **>= 7 / 15** on `--split 15`.
- **Winning Submission Retention Rule:**
  - Whenever ANY iteration achieves **`solved >= 7` out of 15**, immediately package and
    permanently keep a clean competition-ready archive under `kept_submissions/` named
    `submission_wsp_<iter_id>_<solved>of15_<adapter_sha>.zip` (containing ONLY
    `agent.yaml`, `eval_config.yaml`, `instruction.md`, and `adapters/wsp_r1/`—NEVER
    `autoresearch_run.json` or `dataset-metadata.json`).
  - Never delete or overwrite a previously kept `>= 7/15` zip file.
- **STOP the autoresearch loop immediately when ANY of these occur:**
  1. **Target passed & archived:** `solved >= 7` on `--split 15`, its `submission_wsp_*.zip`
     is saved and verified in `kept_submissions/`, AND one `--split 30` holdout confirmation
     completes.
  2. **Iteration cap:** 12 weight-programming iterations completed.
  3. **GPU conservation:** Remaining weekly Kaggle GPU quota drops below 1.5 hours or
     Kaggle returns an explicit quota error.

## Phase A — One-Time Evaluator Setup (Reference: `gemma-4-developer-agent-testing_v11.ipynb`)
If `tests_on_tasks/gemma-4-developer-agent-testing_v12_wsp.ipynb` does not exist yet,
create it ONCE by copying `tests_on_tasks/gemma-4-developer-agent-testing_v11.ipynb` and
making **ONLY** these surgical modifications:

### Allowed Changes from `v11` -> `v12_wsp`
1. **Cell 2 (Rolling Dataset Input):** Point `Submission_DIR` to the single rolling Kaggle
   dataset `/kaggle/input/datasets/prince22466/g6-autoresearch-sub` and load optional
   `autoresearch_run.json` (`iter_id`, `split`, `expected_adapter_sha256`).
2. **Cell 4 (Clean `AGENT_DIR` + Early SHA Assertion):** Immediately after
   `shutil.copytree(Submission_DIR, AGENT_DIR)`, remove `autoresearch_run.json` and
   `dataset-metadata.json` from `AGENT_DIR` if present. Compute `adapter_sha` of
   `AGENT_DIR/adapters/wsp_r1/adapter_model.safetensors` and assert it equals
   `expected_adapter_sha256` (fail fast BEFORE starting vLLM if Kaggle served a stale version).
3. **Cell 6 (2-Second LoRA Preflight):** After `server_instance.start()`, verify
   `vllm_cfg.enable_lora` and `"wsp_r1" in adapter_map`, then send one 8-token
   `temperature=0.0` `/chat/completions` ping to `model="wsp_r1"` to confirm valid output.
4. **Cell 8 (4-Minute Cap + Split Toggle):** Change `'max_time_minutes': None` to
   `'max_time_minutes': INHERIT` (so `submission_36`'s `max_time_minutes: 4` is enforced),
   and select `QUICK_15` vs. `QUICK_30` based on `autoresearch_run.json["split"]`.
5. **Cell 11 (Compact Metrics Export):** Append one read-only post-evaluation cell that
   reads `TEST_RECORDS` and `RESULTS_DIR/report/*` and writes `/kaggle/working/neon_metrics.json`
   and `/kaggle/working/neon_failures.jsonl`.

### STRICTLY UNTOUCHABLE Evaluation Components in `v11` (DO NOT MODIFY)
- **DO NOT modify** `QUICK_15` (`[24, 14, 17, 19, 21, 22, 66, 67, 72, 75, 107, 113, 71, 106, 117]`)
  or `QUICK_30` (`QUICK_15 + [16, 13, 95, 68, 69, 74, 100, 108, 109, 114, 115, 118, 119, 98, 127]`).
- **DO NOT modify or delete** Cell 9 (`submission_metadata()` replacement + preflight check).
- **DO NOT modify** `CACHE_SETTINGS`, `COMPACTION_SETTINGS`, `build_submission_limits()`,
  `EvaluationBudget`, `HarnessLimits`, `EvalConfig`, `Evaluator`, `evaluate_with_diagnostics`,
  or how `resolved` is graded by `swegemma`.
- **DO NOT modify** vLLM parsers (`tool_call_parser='gemma4'`, `reasoning_parser='gemma4'`),
  `max_model_len=32768`, or `MODEL_PATH`.
- Once `v12_wsp.ipynb` runs cleanly on Iteration 1, **FREEZE `v12_wsp.ipynb` permanently**
  for all remaining iterations.

## Phase B — Autonomous Weight-Space Programming Loop (Kaggle CLI)
**Locked files:** `base_submission/*`, `basis_cache.pt`, `basis_diagnostics.json`,
`eval_bridge.py`, `kaggle_eval_kernel/*`.
**Mutable file:** `recipe.py` ONLY.

### Rules
1. **Zero Wasted GPU Runs:** Do NOT run a dummy/random smoke-test evaluation over 15 tasks.
   Start Iteration 1 directly with `MODE = "basis_r1"` (`RANK = 1`).
2. **Use L4x4 for Remote Runs:** Pass `--accelerator NvidiaL4x4` for `--split 15` and `--split 30` runs
   (matching `tests_on_tasks/gemma-4-developer-agent-testing_v11.ipynb`).
3. **Change 1–2 Geometric Knobs per Iteration:**
   - **Iter 1:** Read `basis_diagnostics.json`, set `KEY_PREFIX_FMT`/`D_IN`/`D_OUT`,
     `RANK = 1`, primary fault `"edit_file_loop"` (or highest `cohen_d`), top 4–5
     middle-to-late `down_proj` layers, `top_k = 128`, conservative gains `0.06–0.14`.
   - **Gains (`gains`):** Scale up `1.3x–1.5x` (cap `<= 0.50`) if `file_edit_errors_total`
     drops cleanly with `syntax_leaks == 0`; halve gains or drop the highest-index layer
     immediately if `syntax_leaks > 1` or `solved` drops below 3/15; test sign flip once
     if target errors strictly worsen.
   - **Trigger Sparsity (`top_k` on `lora_A`):** Tighten (`128 -> 64 -> 32`) if loops drop
     but previously solved tasks regress; widen (`128 -> 256`) only if steering has no effect.
   - **Layer Band (`ACTIVE_LAYERS`):** Prune weak layers or shift the 4–6 layer window by ±2.
   - **Rank-2 Expansion (`RANK = 2`):** Only after `solved >= 5/15` and 3 stalled runs;
     freeze Slot 0 and add Slot 1 (`"premature_submit"`, gains `0.05–0.10`, `ORTHOGONALIZE_B = True`).

### Execution Steps (Repeat Until Stopping Condition)
1. **Diagnose:** Read `results.tsv`, `basis_diagnostics.json`, and the best kept run's
   `runs/<best>/neon_failures.jsonl`.
2. **Hypothesize:** Write a 1-sentence mechanistic weight-space hypothesis.
3. **Program Weights:** Edit `recipe.py` (1–2 knobs only).
4. **Verify via Kaggle CLI:**
   `python eval_bridge.py --iter iter_XX --split 15 --accelerator NvidiaL4x4`
   Verify returned `adapter_sha256` matches compiled `adapter_model.safetensors` and
   `enable_lora == true`.
5. **Keep / Discard Gate:**
   - **KEEP** (`git add recipe.py && git commit -m "iter_XX: solved=X/15 ..."`) if:
     - `solved` strictly increases AND `syntax_leaks <= 1`, **OR**
     - `solved` ties best kept run AND (`file_edit_errors_total + premature_submits`)
       drops by `>= 20%` with `syntax_leaks <= 1`.
   - **DISCARD** otherwise (`git checkout -- recipe.py`).
6. **Log:** Append one row to `results.tsv`:
   `iter_id | git_sha | adapter_sha | rank | split | solved | edit_err | premature | leaks | avg_sec | decision | kept_zip | hypothesis`
7. **7/15 Retention & 30-Task Holdout Gate:**
   - **Milestone (`solved >= 5` first time):** Run `--split 30` once to check generalization
     before continuing `--split 15`.
   - **Pass Target (`solved >= 7` on `--split 15`):**
     1. Confirm `eval_bridge.py` archived `kept_submissions/submission_wsp_<iter_id>_<solved>of15_<sha>.zip`
        and verify its zip entries contain ONLY `agent.yaml`, `eval_config.yaml`,
        `instruction.md`, and `adapters/wsp_r1/{adapter_config.json,adapter_model.safetensors}`.
     2. Run one final holdout check:
        `python eval_bridge.py --iter iter_XX_holdout --split 30 --accelerator NvidiaL4x4`
     3. Log the 30-task result in `results.tsv`, **STOP the loop**, and report the kept zip
        path and scores to the user.