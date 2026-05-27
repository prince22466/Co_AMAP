---
name: fast-finetune-nemotron-moe
description: Use for this Kaggle/NVIDIA Nemotron-3-Nano-30B project when Codex needs to inspect, plan, debug, or speed up fine-tuning of the MoE/Mamba-style base model and produce a compatible LoRA adapter. Trigger on requests about Nemotron-3-Nano-30B, LoRA adapters, fast Kaggle training, validation scoring, target_modules, MoE layers, Mamba/Triton issues, early stopping, adapter_config.json, or aligning notebook validation with the competition metric.
---

# Fast Fine-Tune Nemotron MoE

## Workflow

1. Ground in the current project files first:
   - Find the active notebook, usually `nvidia-nemotron-submission-lora.ipynb`.
   - Inspect `train_LoraAdaptors/LoRA_001.py` for LoRA parameters.
   - Read `competition_material/Evaluation.md` and `competition_material/nvidia-nemotron-metric.ipynb` before changing scoring or submission behavior.

2. Keep the LoRA component small:
   - Put only LoRA parameters in component files such as `LoRA_001.py`.
   - Keep model loading, scoring, training loops, early stopping, and adapter saving in the notebook unless the user explicitly asks for module extraction.
   - Use `max_lora_rank <= 32`; the official metric runs with `max_lora_rank=32`.

3. Optimize for fast Kaggle feedback:
   - Use hard stop limits (`MAX_STEPS`) before relying on epochs.
   - Evaluate on a small validation subset during training.
   - Stop if validation accuracy gets worse than the best observed score.
   - Prefer short workflow checks before full runs; Kaggle can cancel long notebooks.

4. Match the evaluator:
   - Train prompts should teach final answers in `\boxed{...}`.
   - Validation scoring should extract boxed answers first, then fallback patterns/numbers.
   - Binary-string answers must compare exactly; numeric answers may use `rel_tol=1e-2`.

5. Inspect model layers before changing LoRA targets:
   - Print candidate `nn.Linear` modules containing `proj`, `attn`, `mlp`, `moe`, `expert`, or `gate`.
   - For Nemotron/Mamba blocks, expect names such as `in_proj`, `out_proj`, `up_proj`, and `down_proj`.
   - Change `TARGET_MODULES` only after confirming actual module names.

## Common Fixes

- If training runs too long, reduce `MAX_STEPS`, `MAX_LENGTH`, `EARLY_STOPPING_VAL_ROWS`, or `EVAL_EVERY_STEPS`; do not start by changing LoRA rank.
- If validation is slow, score fewer validation rows during early stopping and keep full validation for final reporting only.
- If generation fails with Triton `ptxas-blackwell` permission errors, copy the binary to `/kaggle/working`, `chmod +x`, and set `TRITON_PTXAS_BLACKWELL_PATH`.
- If LoRA attaches to no modules, print module names and adjust `TARGET_MODULES`; do not guess from another model family.
- If the score stays zero, verify that generated text contains extractable answers and that training examples include `\boxed{answer}`.

## Reference

Read `references/nemotron_lora_fast_finetune.md` when working on scoring, early stopping, target module selection, Kaggle runtime errors, or submission compatibility.
