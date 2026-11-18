# v16 — 4K throughput, preserve 16K

Original v16 control is the exact v14 production baseline (commit c0e5b3d37f7073ef04ec95c50207b5543ffdf626). This branch adds the provisional bounded native Triton top-p row-grid candidate from [PR #261](https://github.com/prince22466/Co_AMAP/pull/261). No v15 candidate is inherited. The scratch probe measured 3.631 → 1.007 ms at batch 64; the production integration, 4K/16K serving and MATH-500 are unverified.

The candidate is enabled by the existing Iluvatar backend hook only for the exact tested upstream vLLM 0.24.0 source. Existing backend selection, top-k-only and small-batch paths remain unchanged. The launch cap is 4 × detected compute units, not a fixed model or benchmark shape. Unlike the isolated probe's persistent workspace, the expanded-grid wrapper allocates only the rows needed by each launch via PyTorch's caching allocator. This avoids shared mutable scratch between streams or captured graphs, without adding a permanent plugin cache. At 16 compute units and vocabulary 130560, the expanded launch requires up to 31.88 MiB of temporary scratch; allocator reservations and graph pools may retain memory. The original upstream cache is unchanged. Allocation overhead, graph replay, peak/reserved memory, KV capacity and 16K non-regression must be measured on BI-V150.

CPU-only launch and fallback contract tests: `python3 -m pytest -q --noconftest vllm-plugin-FL/tests/functional_tests/test_top_p_row_grid_contract.py` (13 passed in a local CPU environment against the committed source; mocked kernel, not GPU validation).

Run focused GPU validation with the original v16 environment (from this directory):

```bash
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL"
export VLLM_PLUGINS=fl
python3 -m pytest -q -s --noconftest vllm-plugin-FL/tests/functional_tests/test_top_p_row_grid.py
```

Pin the original control SHA separately from this candidate's committed source; verify actual worker installation and execution, then follow the unchanged matched benchmark protocol.

Goal: >=60% 4K total tok/s improvement over the recorded v0 self-test, with no 16K regression relative to v14. See [AGENTS.md](AGENTS.md) for full-run and matched development targets, [program.md](program.md) for the Remote-SSH runbook, and [research queue](autoresearch/RESEARCH_QUEUE.md) for prior evidence and hypotheses.

Maintain [experiment ledger](autoresearch/EXPERIMENTS.md) and [handoff](autoresearch/HANDOFF.md).
