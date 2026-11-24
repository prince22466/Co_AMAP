# Adapter weight-space programming toward 7/15

Start from `iter_04` (best QUICK_15 score 5/15). Test whether a small FFN weight edit makes Gemma check the public API it just changed before submitting. On `rich_3052`, for example, the agent edited the constructor while the public `Prompt.ask(...)` call still failed. Keep the base model, agent and evaluator frozen. [Run ledger](results.tsv)

## Find the weights

1. **Pair decisions.** On development tasks outside QUICK_30, match unchecked completion with a useful check at the same repair phase. Include verified patches where submitting is correct. Split by repository snapshot. Reproduce the champion's action on the exact input. Use distinct aligned histories or continuation windows to measure a contrast. Identical prefixes give identical pre-branch states.
2. **Harvest once per prefix.** With the champion active, hook all 42 FFNs. Cache per-issue means of the gated input `h` to `down_proj` and its raw output over matched windows. Reuse these summaries to rank columns offline. Each edited replay still requires a forward pass.
3. **Rank columns.** Adapt Experiment 8's signed contribution score. Per layer, let `u` be the normalized mean unchecked-minus-useful difference in raw FFN output, and `dh` the corresponding mean input difference. Give issues equal weight. Score column `j` by `dh[j] * (u @ W_eff[:, j])`, where `W_eff` includes the dequantized base projection and champion update. Derive directions and masks on discovery folds. Shortlist layers whose selected columns also score positively on omitted snapshot groups. [Experiment 8 search code](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/tool_calls.jsonl)

## Edit and test

Start with two shortlisted layers. In each, test the strongest column alone and up to 16 positive-scoring columns together at gain 0.25. These four proposed candidates share a baseline. For each binary column mask, program one extra LoRA slot.

```text
A_new = mask * (u @ W_eff)
B_new = -gain * u
```

Copy the champion into the first slot. Set `r=2` and `lora_alpha=2` so scaling is one. Add an explicit-factor path to the compiler because its current normalization and orthogonalization can change the edit. Reload the saved factors and verify their write on recorded inputs. [Compiler](recipe.py) · [Evaluated rank-2 adapter](runs/iter_08/neon_metrics.json)

Compare with a zero-padded champion and a norm-matched random edit. Execute the generated check. If valid submissions regress, refine **A** for selectivity. If the tool is appropriate but its arguments are useless, test **B** at argument-generation positions. If only a token-local hook helps, test all-token application with a fresh KV cache before compiling. Drop directions that perform like the random control.

**Promotion.** Freeze the finalist and grade fresh candidate and control runs on held-out issues. Look for a relevant check that exposes a defect, a subsequent repair and a passing patch. On QUICK_15, record both new solves and lost solves. At 7/15 or better, archive the adapter and confirm the same SHA on QUICK_30, reporting the original and additional 15 separately. [Archive procedure](program.md)

For implementation, see [GeGLU hook code](colab_run_harvest.py) · [Colab setup](COLAB_WORKFLOW.md).

---

## Target tasks & forensic guardrails (`5/15 -> 7+/15`)

- **Locked-in champion floor (`5/15` in `iter_04`, zero regressions allowed):** `rich_4077`, `rich_3882`, `rich_3905`, `rich_3518`, `rich_3454`.
- **Primary conversion targets (need `+2` solves from these 4 near-misses):**
  1. **`rich_3052` (`rich/prompt.py`):** Champion patches `PromptBase.__init__`, `check_choice`, and `process_response`, but forgets to forward `case_sensitive: bool = True` in the public `@classmethod def ask(...)` wrapper (`line 115`) and calls `submit_patch` at 40s without testing. Running `pytest tests/test_prompt.py` (`0.08s`) immediately fails with `TypeError: PromptBase.ask() got an unexpected keyword argument 'case_sensitive'`.
  2. **`rich_3934` (`rich/live_render.py` / `rich/live.py`):** Champion edits `rich/live.py` (or `LiveRender.__rich_console__` in `rich/live_render.py`) and calls `submit_patch` at 38s without running `pytest tests/test_live.py`.
  3. **`rich_3718` (`rich/panel.py`):** Issue statement notes *"This really just reverts the change in 7a38204"*. Champion guesses `title_text.style = self.style` in 3 calls (`28s`) and calls `submit_patch` without running `git show 7a38204` or `pytest tests/test_panel.py`.
  4. **`fastapi_14258` (`fastapi/routing.py`):** Champion writes the exact required check (`if router is self: raise FastAPIError(...)`), but `APIRouter.include_router` has a 140-line signature (`lines 1258–1400`); after `read_file(1250, 1300)`, it pastes the check inside the parameter list at `line 1265` (`SyntaxError`) and submits without running `pytest` or reading `1300–1410`.
- **Benchmark harness caveat on `requests_7205` & `requests_7315` (`src/` layout shadowing):**
  - Unlike `rich/` and `fastapi/` (flat workspace root), `psf/requests` uses `./src/requests/` without `pythonpath = src` in `pytest.ini`. Under `sandbox='subprocess'`, bare `pytest` imports Kaggle's pre-installed `/usr/local/lib/python3.12/site-packages/requests` (`2.32.3`) instead of `./src/requests/`, causing even 100% ground-truth patches to `src/requests/utils.py` (`requests_7205`) and `src/requests/adapters.py` (`requests_7315`) to execute unpatched code.
  - **Rule:** Do not use bare `pytest` pass/fail on `requests_*` as the primary selector for weight edits (or use `PYTHONPATH=src pytest` when replaying `requests_*` checks); prioritize flipping `rich_3052`, `rich_3934`, `rich_3718`, `fastapi_14258`, `rich_4075`, and `rich_4076`.

---

## Mandatory harvesting & architecture settings

1. **Left-truncation with `max_length = 4096` (update `colab_run_harvest.py` from `1024 -> 4096`):**
   - Multi-turn repair trajectories reach `2,000–3,800` tokens before the post-edit decision turn. Always set:
     ```python
     tokenizer.truncation_side = "left"
     enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
     ```
   - `max_length=1024` truncates away earlier turns showing whether `pytest` was already run; right-truncation drops the latest `edit_file` diff and `<|thought>` trigger token.
2. **Gemma 4 E4B global vs. sliding-window layer selection:**
   - Only **7 of 42 layers** in Gemma 4 E4B use global full attention (`[5, 11, 17, 23, 29, 35, 41]`, `o_proj.d_in = 4096`). The remaining 35 layers use `sliding_window = 512` (`o_proj.d_in = 2048`) and cannot see across a long `edit_file` diff + thought block to know whether tests already ran.
   - While hooking all 42 FFNs (`down_proj`: `d_in = 10240, d_out = 2560`), prioritize shortlisting candidate FFN layers **at or immediately downstream of global attention blocks** (`[17..41]`, especially around global hubs `23, 29, 35` and champion layers `[29, 31, 34, 37, 38]`).
3. **Exact `W_eff` reconstruction on champion layers:**
   - Warm up the INT4 model once so `compressed_tensors` materializes the dequantized base weights `W_base` (`shape [2560, 10240]`).
   - On champion layers `[29, 31, 34, 37, 38]`, form the exact effective matrix before scoring columns:
     ```python
     W_eff = W_base.float() + (B_champ.float() @ A_champ.float())  # lora_alpha / r = 1.0
     ```
   - On non-champion layers, `W_eff = W_base.float()`.
4. **Two matched continuation windows to harvest per decision pair:**
   - **Window W1 (Thought-start commitment, `<|turn>model\n<|thought>\n` + first 8 thought tokens):** Where the model decides between planning a verification check vs. declaring completion.
   - **Window W2 (Tool-call dispatch & arguments, `<thought|><|tool_call>call:`):** Where the model selects `run_command` vs. `submit_patch` and writes `{"command": "pytest ..."}`.

---

## Additional necessary experiments & controls

### Experiment A — Signed response & asymmetric selectivity check (preventing verified-submit reversal)
With `A_new = mask * (u @ W_eff)` and `B_new = -gain * u`, the rank-1 update writes:
$$\Delta y(h) = B_{\text{new}}(A_{\text{new}} h) = -\text{gain} \cdot u \, (A_{\text{new}} h)$$
Because this projection is linear in $h$, if the scalar activation $s(h) = A_{\text{new}} h$ is positive on `unchecked_after_edit` states ($s(h) > 0$, pushing along $-u$ toward verification) but flips **negative** on `verified_ready_to_submit` or `mid_edit_code` states ($s(h) < 0$), the edit will reverse direction and push verified or mid-edit states *toward* premature completion or `FileEditError` loops (the failure mode seen in `iter_05..14` and `iter_25`).

- **Diagnostic gate before full evaluation:** Measure the distribution of $s(h) = A_{\text{new}} h$ across three held-out state sets:
  1. $H_{\text{unchecked}}$ (post-edit, no test run yet): require mean $s(h) > 0$ with >80% positive sign.
  2. $H_{\text{verified}}$ (post-edit, test just passed, `submit_patch` is correct): require $|s(h)| \ll s(H_{\text{unchecked}})$ (not strongly negative).
  3. $H_{\text{edit\_tokens}}$ (inside `edit_file` `old_string`/`new_string` generation): require near-zero activation so code strings stay unperturbed.
- **Null-space refinement for `A_new` if valid submissions or code edits regress:** Restricting to the selected column indices $S = \text{supp}(\text{mask})$ ($|S| \le 16$), damp response on preservation states $H_{\text{pres}} = [H_{\text{verified}}; H_{\text{edit\_tokens}}]$ via ridge projection on the $S \times S$ sub-Gram matrix:
  $$A_{\text{new}, S} = (u^T W_{\text{eff}, :, S})\left(I_{|S|} + \lambda_{\text{pres}} H_{\text{pres}, S}^T H_{\text{pres}, S}\right)^{-1}$$
  This preserves the exact sparse column support $S$ while suppressing interference on valid submissions and code edits.

### Experiment B — Scale-calibrated gain bracket (`K = 1` vs. `K = 16` columns)
Because `A_new = mask * (u @ W_eff)` is unnormalized, $\|A_{\text{new}}\|_2$ grows roughly as $\sqrt{K}$ when moving from the strongest single column ($K=1$) to $K \le 16$ positive-scoring columns.
- Alongside the baseline `gain = 0.25`, record the relative FFN perturbation ratio $\rho = \|\Delta y(h)\|_2 / \|W_{\text{eff}} h\|_2$ on discovery prefixes.
- If $K=16$ at `gain = 0.25` exceeds $\rho > 0.15$ (the threshold above which `iter_25` triggered `FileEditError` spikes), also test `gain / sqrt(K)` (or a 3-point bracket `gain in {0.10, 0.25, 0.40}`) matched in effective output norm to the $K=1$ candidate.

### Experiment C — GeGLU pre-gate comparison (`gate_proj` + `down_proj` vs. `down_proj` column mask)
If the linear `down_proj` column mask $A_{\text{new}} h$ still fires during `edit_file` argument generation under all-token application:
- Test pairing the selected column mask $S$ ($|S| \le 16$) in `down_proj` with a rank-1 `gate_proj` row edit targeting those same $S$ intermediate FFN columns (`target_modules: ["gate_proj", "down_proj"]`, `r = 2, lora_alpha = 2`).
- Driving the pre-GELU gate of columns $S$ negative ($\le -3.0$) on `verified_submit` and `edit_file` contexts exploits $\text{GELU}(+3) / \text{GELU}(-3) \approx 750\times$ to silence the new `down_proj` slot outside unverified post-edit turns.

### Experiment D — Second verification behavior if needed (`read_file` continuation on long signatures for `fastapi_14258`)
If the post-edit public-API check flips `rich_3052` and/or `rich_3934` (`5/15 -> 6/15`) while `fastapi_14258` still fails before reaching a valid edit because it pastes code inside a truncated 140-line function signature:
- Run the exact same paired-decision column-ranking pipeline on `truncated_signature_read -> read_file(next_lines)` vs. `blind_edit_inside_signature`, and compile it into Slot 2 (`r = 3, lora_alpha = 3`).

---

## Compiler (`recipe.py`) & verification checklist

1. **Add `MODE = "explicit_factors"` to [recipe.py](recipe.py):**
   - Bypass `sparsify_topk()` unit-normalization and `ORTHOGONALIZE_B` Gram-Schmidt rotation when writing explicit `(A_new, B_new)` factors.
   - Support **union of active layers** (`ACTIVE_LAYERS = sorted(set(CHAMP_LAYERS) | set(NEW_LAYERS))`):
     - Slot 0 holds the exact `iter_04` champion factors on `[29, 31, 34, 37, 38]` and exact zeros on any non-champion layer.
     - Slot 1 holds `(A_new, B_new)` on the shortlisted edit layer(s) and exact zeros on all other active layers.
     - Set `"r": 2, "lora_alpha": 2` in `adapter_config.json` so PEFT scaling is `lora_alpha / r = 1.0` everywhere.
2. **Control adapters required before benchmark promotion:**
   - **Zero-padded champion control (`r = 2, lora_alpha = 2`):** Slot 0 = `iter_04`, Slot 1 = `0`. Verify identical output logits and `adapter_model.safetensors` load behavior against `iter_04` (`r = 1`).
   - **Norm-matched random control (`r = 2, lora_alpha = 2`):** Slot 0 = `iter_04`, Slot 1 = random $K$-sparse `A_rand` and random `B_rand` matched to $\|A_{\text{new}}\|_2$ and $\|B_{\text{new}}\|_2$.
3. **Round-trip & all-token KV-cache verification gate (3-minute Colab L4 check before 40-minute `QUICK_15` run):**
   - Reload saved `adapter_model.safetensors`, verify `|(B_1 @ A_1) @ h - B_new (A_new @ h)| / |B_new (A_new @ h)| < 1e-3` in BF16 on cached `h`.
   - Run greedy autoregressive generation (`temperature=0.0`, fresh KV cache, all-token LoRA active) on held-out decision prefixes and verify:
     1. Unchecked post-edit prefixes emit a verification tool call (`run_command` with `pytest`) whose command actually executes and exposes the defect (e.g., `Prompt.ask` in `rich_3052`).
     2. Verified post-test prefixes still emit `submit_patch` (`0%` regression on valid submissions).
     3. Replay of `iter_04` `edit_file` turns on `rich_4077` and `rich_3882` produces valid JSON/Python with `0` syntax leaks.

