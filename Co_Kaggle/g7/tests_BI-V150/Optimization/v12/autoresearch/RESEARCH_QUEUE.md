# v12 attention hypotheses — re-rank using a fresh serving trace

The current kernel already implements tiled QK, online softmax and PV, with v10 BF16 GQA prefill and decode page lookups. A backend swap or an unchanged FlashAttention-like implementation **does not count**.

| Priority | Experiment | Exact? | Evidence/stop test |
|---|---|---|---|
| E1 | Optimize *existing* Triton attention dataflow: K/V staging and reuse, work partition, QK/PV and online-softmax reductions | Yes | Fewer actual instructions/bytes or spills; dense reference correctness; significant 4K/16K serving gain |
| E2 | Long-KV split-K / Flash-Decoding partials with stable online-softmax merge | Yes | Extra launch/scratch cost versus real decode benefit, 2D/3D/mixed preservation |
| E3 | Paged-KV page index reuse, double buffering, mixed prefill/decode load dependencies | Yes | Measured bytes/latency reduction in the critical path |
| S1 | Local window + retained prefix / global anchor KV tiles | No | Actual skipped loads/dots and **middle-context retrieval loss** versus dense |
| S2 | Query-aware selection of nonlocal KV pages using existing Q/K summaries | No | Selector overhead + recall versus dense attention over 16K |
| S3 | Hybrid dense prefill + selective decode | No | Decode gain, no model-ability regression, no hidden extra scheduling |
| R1 | Distributed Ring Attention | Yes, if multi-GPU | Only pursue with >1 authorized GPU and measured device communication bottleneck |

**First research cycle:** profile 4K and 16K prefill/mixed/decode, estimate Amdahl opportunity; test one measured **exact** idea and one **sparse** idea (or explicitly document why untestable). Work on one reversible patch at a time and do fast correctness + interleaved kernel A/B before serving. Drop approaches with no critical-path impact.

**Sparse quality gate:** validate against an explicit sparse PyTorch oracle, **then compare the same model with original dense attention** on 4K/8K/16K random-position factual retrieval (beginning, **middle**, end), distractors, multiple seeds and multi-hop questions. Track per-query allowed K/V blocks, skipped QK/PV work, selector time, attended-token ratio, runtime and error. Local+sink alone may discard decisive facts in the middle. Human-owned unchanged MATH-500 Level 3 >=95% plus organizer clearance for behavior-changing attention is required before a competition submission can rely on it.

**Context and references:** use original MiniCPM model/weights and existing FlagOS operators; no external retrieval models. FlashAttention-2: https://arxiv.org/abs/2307.08691 ; Quest: https://proceedings.mlr.press/v235/tang24l.html ; Ring Attention: https://proceedings.iclr.cc/paper_files/paper/2024/hash/1119587863e78451f080da2a768c4935-Abstract-Conference.html .
