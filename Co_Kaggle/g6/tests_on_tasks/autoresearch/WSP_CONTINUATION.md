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

Copy the champion into the first slot. Set `r=2` and `lora_alpha=2` so scaling is one. Add an explicit-factor path to the compiler because its current normalization and orthogonalization can change the edit. Reload the saved factors and verify their write on recorded inputs. [Compiler](recipe.py) ? [Evaluated rank-2 adapter](runs/iter_08/neon_metrics.json)

Compare with a zero-padded champion and a norm-matched random edit. Execute the generated check. If valid submissions regress, refine **A** for selectivity. If the tool is appropriate but its arguments are useless, test **B** at argument-generation positions. If only a token-local hook helps, test all-token application with a fresh KV cache before compiling. Drop directions that perform like the random control.

**Promotion.** Freeze the finalist and grade fresh candidate and control runs on held-out issues. Look for a relevant check that exposes a defect, a subsequent repair and a passing patch. On QUICK_15, record both new solves and lost solves. At 7/15 or better, archive the adapter and confirm the same SHA on QUICK_30, reporting the original and additional 15 separately. [Archive procedure](program.md)

For implementation, see [GeGLU hook code](colab_run_harvest.py) ? [Colab setup](COLAB_WORKFLOW.md).
