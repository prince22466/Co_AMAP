# Pure Weight-Space Programming (WSP) Progress Report — 30-Iteration Kaggle L4 x4 Campaign

**Status:** Paused due to Kaggle weekly GPU quota exhaustion (`Maximum weekly GPU quota of 45.00 hours reached` on both `NvidiaL4` and `NvidiaTeslaT4`).
**All Kaggle GPUs Confirmed OFF:**
- `prince22466/gemma-4-developer-agent-testing`: `KernelWorkerStatus.COMPLETE`
- `prince22466/g6-wsp-basis-harvest`: `KernelWorkerStatus.COMPLETE`

**Best Pure-WSP Checkpoint Restored & Packaged:**
- **Iteration:** `iter_04` (committed on branch `g6/wsp-autoresearch`, commit `cc3993b3`)
- **Score:** **`5 / 15`** (`rich_4077`, `rich_3882`, `rich_3905`, `rich_3518`, `rich_3454`) vs frozen baseline `submission_36` (`3 / 15`)
- **Metrics:** `file_edit_errors_total = 68` (down from `206`, `-67.0%`), `premature_submits = 6`, `syntax_leaks = 0`, `avg_duration_seconds = 130.86s`
- **Adapter SHA-256:** `8a018d2df7d8bc367925533ce0488ee1bbadeecd7ac813fdd3a97d5b9d3c9815`
- **Competition-Ready ZIP:** `submission_nb/artifacts/submission_40.zip` (`39,599 bytes`, verified 5 allowed files only)
- **30-Task Milestone Check (`iter_04_m30`):** `4 / 30` (`rich_3882`, `rich_3905`, `rich_3454`, `requests_6644`)

---

## Complete Summary of All 30 Iterations (`results.tsv`)

| Paradigm | Iterations | Best Solved | Best Edit Err | Key Scientific Finding |
|---|---|---|---|---|
| Frozen Baseline (`submission_36`, no LoRA) | `baseline_s36` | `3/15` | `206` | `leaks=4`, `prem=4`, `avg_sec=184.6s` |
| Static Unembedding Rank-1 `mlp.down_proj` | `iter_01`–`07`, `10`–`12`, `23` | **`5/15` (`iter_04`)** | `57`–`68` | `[29,31,34,37,38]`, `top_k=64`, gains `0.12–0.16` hit `5/15`, `leaks=0`. |
| Multi-Fault Rank-2 `mlp.down_proj` | `iter_08`–`09`, `15`–`17` | `3/15` | `34` (`iter_09`) | Rank-2 on same 5 layers collided (`cos(u_ef, u_vs)=0.765`). |
| Live Contrastive Activation Basis (`v2`) | `iter_13`–`16` | `3/15` | `94`–`118` | `u_live` orthogonal to `lm_head` logit axis. |
| One-Sided Orthogonal `submit` Interceptor | `iter_18`–`19` | `3/15` | `50` (`iter_19`) | Disjoint `[30,32,35,40]` cut `edit_err=50` but linear `down_proj` couldn't stop `prem=9`. |
| DLA + AlphaEdit Tool-Call `o_proj + down_proj` | `iter_20`–`21` | `2/15` | `46` (`iter_21`) | DLA proved `self_attn.o_proj` has **10×–44×** higher logit attribution than `down_proj`. |
| Positive / Negative LASER SVD (`r=32`, `r=8`) | `iter_22`, `iter_30` | `3/15` | `124`–`139` | Negative `r=8` held `prem=6` and solved `rich_3882`, but raised `edit_err=124`. |
| Rank-64 BF16 Quantization-Residual Recovery | `iter_24` | `2/15` | `63` | Recovered 21%–29% of INT4 `o_proj` error vs BF16 `gemma-4-E4B-it`. |
| Thought-Start (`<|thought>`) AlphaEdit `o_proj` | `iter_25`–`29` | **`4/15` (`iter_26`)** | **`26` (`iter_26`)** | `L16+L32 o_proj` achieved **all-time record low `edit_err=26` (-87%) & 0 timeouts**, and live GPU generation flipped `requests_7205` `submit_patch -> run_command(python3 test)`. |

---

## Key Discoveries & Ready-to-Run Artifacts for 3rd-Party Online GPUs

1. **Ready-to-Run Next Pure-WSP Experiment (`harvest_basis.py` v10):**
   - **Nonlinear `IF-THEN` GeGLU Transistor Circuit** (`tests_on_tasks/autoresearch/kaggle_basis_kernel/harvest_basis.py`): programs `mlp.gate_proj` + `mlp.up_proj` + `mlp.down_proj` on 16 naturally-OFF neurons using a signed ridge gate (`+3.0` on `prem_submit`, `-3.0` on `valid_submit` + healthy turns -> `GELU(+3.0)/GELU(-3.0) ≈ 750×` nonlinear ON/OFF switch ratio!).
   - Ready to run on any 1× L4 / A10G / A100 / 4090 GPU (~5 minutes) using `tests_on_tasks/autoresearch/dla_probes.json` (141 real test-trace probes).
2. **Saved Local Basis & Diagnostic Caches (`tests_on_tasks/autoresearch/`):**
   - `results.tsv`: Full 30-iteration log with git SHAs, adapter SHAs, metrics, and hypotheses.
   - `basis_cache.pt`: Static + live + hybrid + one-sided orthogonal `mlp.down_proj` bases.
   - `alphaedit_basis.pt` & `thought_basis.pt`: Live DLA + AlphaEdit `self_attn.o_proj` bases (`L16..L41`).
   - `dla_probes.json`: 141 exact Gemma-4-formatted decision probes extracted from real testing traces.
   - `laser_svd_cache.pt` & `bf16_residual_cache.pt`: Precomputed SVD caches for LASER and BF16 quantization-residual recovery.
   - `runs/iter_01..iter_30/`: Full `neon_metrics.json`, `neon_failures.jsonl`, and diagnostic ZIPs for every iteration.
3. **Architecture & Bottleneck Findings:**
   - Gemma 4 E4B uses `sliding_window=512` on 35/42 layers; only `[5, 11, 17, 23, 29, 35, 41]` are global full-attention layers.
   - Across all 30 zero-backprop pure-WSP iterations on `gemma-4-e4b-it`, the union of solved tasks remained the same 5 tasks (`rich_3905`, `rich_3454`, `rich_4077`, `rich_3882`, `rich_3518`), with `0/30` on the remaining 10 tasks.
   - On 3rd-party GPUs, the highest-leverage next steps are: (1) run the ready-to-go **Nonlinear `IF-THEN` GeGLU Transistor Circuit** (`harvest_basis.py` v10), (2) test pure WSP on **`gemma-4-12b-it-qat-w4a16-ct` / `31b`**, or (3) run short **LoRA SFT/DPO** on multi-step `edit -> pytest -> fix traceback -> submit` trajectories.
