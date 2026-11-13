# 技术报告（v13）

团队：prince22466 ｜ 模型：MiniCPM5-2B ｜ 框架：FlagOS / vLLM-FL ｜ 平台：天数 BI-V150

## 1. 测试结果与效果分析

官方 full-run：输入 4096 / 16384 tokens，输出 1024 tokens，并发 64，请求数分别为 256 / 128。每场景 4 次全部成功，官方脚本 SKIP_FIRST=1，以下采用后三次的汇总值。基线为历史自测第一组 full-run 汇总，并非同一时段重测。

| 场景 | 基线 (tok/s) | v13 (tok/s) | 绝对增益 | 相对增益 |
|---|---:|---:|---:|---:|
| 4K | 1881.95 | 2752.60 | +870.65 | +46.26% |
| 16K | 861.88 | 1890.18 | +1028.30 | +119.31% |

| 指标 | 4K 基线 → v13 | 16K 基线 → v13 |
|---|---:|---:|
| Mean TTFT (ms) | 12309.34 → 5651.44 | 636512.47 → 270913.54 |
| Mean TPOT (ms) | 156.12 → 110.08 | 518.00 → 241.94 |
| Output tok/s | 376.39 → 550.52 | 50.70 → 111.19 |

精度：v13 profiling/summary.txt 记录 math_500 eval >=0.95: True，记录达到 95% 门槛；赛事评测命令采用 MATH-500 Level 3，但 v13 未附具体准确率、子集配置、逐题结果或完整评测日志，不将 v10 的 97.1% 作为 v13 成绩。

Half-run（仅内部参考，不作为正式成绩）：v13 4K/16K 为 2418.05/1853.56 tok/s，v10 为 2372.38/1872.70 tok/s，分别变化 +1.93%/-1.02%。未做同机同批次配对复测，不能证明重构带来增益或统计等价。

16K 吞吐提升最大，TTFT 与 TPOT 同时改善；但未进行 v13 单项消融或 GPU 时间占比测量，不能将全部端到端增益归因于某一项优化。历史基线 CSV 另含第二组未标注汇总（1974.86/888.16 tok/s），未混用。

## 2. 方案设计与算法原理

瓶颈：长序列 Prefill 的 GQA 分块限制了每个 KV Tile 的 Query 复用；Decode 逐 Token Lane 生成页索引，存在重复查询。长上下文服务还受 KV Cache 容量、排队、抢占和混合批次干扰影响。

标准注意力 O = softmax(QK^T / sqrt(d) + M)V；GQA 16 个 Q Head、2 个 KV Head、Head Dim=128，每个 KV Head 对应 8 个 Q Head。逻辑位置 t 的物理页为 BlockTable[floor(t/B)]，页内偏移为 t mod B。对目标 MiniCPM 配置保持因果掩码、全量注意力和分页映射语义，不改变模型权重。

### Prefill：增加 Query 分块与 KV Tile 复用

| 参数 | v0 | v13（满足条件） |
|---|---:|---:|
| BLOCK_M | 16 | 64 |
| BLOCK_Q = BLOCK_M / 8 | 2 | 8 |
| num_stages | Triton 默认 | 2 |
| KV TILE_SIZE（2D） | 32 | 32 |
| num_warps | Triton 默认 | Triton 默认 |

![单个 KV Head 的 Query 分块与 KV Tile 复用](figures/prefill_gqa.svg)

每个 KV Head 的一个满 Query Block 由 2 个位置扩大至 8 个位置（每位置 8 个 Q Head），同一已加载 KV Tile 的最大复用位置数扩大 4 倍。KV Tile 大小保持 32；并非实测带宽或吞吐加速比。适用于 max_seqlen_q>1（含符合条件的混合批次）、BF16 Q/K/V、16Q/2KV、head_size=128、非量化 KV、因果注意力、无 Tensor Descriptor。排除逐序列因果、Sliding Window、Chunk Lookback、多模态前缀、ALiBi、QQ Bias、Sink、Soft Cap、输出量化。混合批次按最大 Query 长度走 2D 路径；单 Token 请求仍使用有效 Query 掩码，不会变成 8 个有效位置。此路径仍逐 Token 查询页表；不满足条件时保留原有 Triton 配置。

## 3. Decode：分页 KV 索引复用

DECODE_PAGE_LOOKUP 为编译期特化，仅在 max_seqlen_q=1、BF16 Q/K/V、16Q/2KV、head_size=128、非量化 KV、无 Tensor Descriptor、页大小为正的 2 次幂，且页大小能整除 Tile 大小或反之时启用。

当页大小 ≥ Tile 大小时，加载单个物理页索引并以 tl.broadcast_to 广播；否则加载 Tile 覆盖的多个页索引，以 tl.gather 分配给各 Lane。以 16-token KV 页为例，2D/3D Decode 的 Tile 分别为 32/16 tokens，对应最多 2/1 个页索引。多页尾 Tile 新增有效页掩码；K/V 读取保留原有 Token 掩码与页内偏移。

![Decode 分页 KV 索引复用（16-token 页示例）](figures/decode_page_lookup.svg)

优化的是地址生成和索引加载，而非减少历史 Token 或 K/V 数据；不改 Decode 的 Query/KV 分块。原有 2D/3D 选择及分段 Softmax 归约不变。Prefill 与 Pure Decode 特化互斥；混合批次仅可进入 Prefill 特化。DECODE_PAGE_LOOKUP 为 False 时保留原有地址路径（含已有 Tensor Descriptor 逻辑）。

## 4. 实现架构、验证与合规

v13 是 v10 的设备无关重构，统一算子文件与 v10 内容一致；对 v0 的源码树比较表明 FlagGems 与 benchmark 脚本均未改变，仅 vllm-plugin-FL 有生产代码改动。

vLLM → PlatformFL.get_attn_backend_cls → 原有 call_op("attention_backend") → 仅当选中 stock dense Triton 且非 MLA / sparse 时，映射到 OptimizedTritonAttentionBackend → OptimizedTritonAttentionImpl.forward → unified_attention。其他选中路径原样返回（含 FlagGems、自定义后端）。新增映射及两项优化条件均无 GPU 厂商或设备能力门控；原始 Triton 内核仍保留 head_size=256 的设备能力特化（与本方案 128 维条件互斥）。设备无关不代表其他 GPU 已验证。

核心文件（相对于 vllm-plugin-FL/）：
- vllm_fl/platform.py：保留原 dispatch 后特化 stock dense Triton。
- vllm_fl/dispatch/backends/triton_attention.py：继承原 Triton Backend/Impl，复用 metadata、KV Cache 更新和接口。
- vllm_fl/dispatch/backends/triton_unified_attention.py：统一 kernel 与两项算子特化，保留原计算和回退；不满足优化条件仍在该统一实现中执行原有通用路径，并非切换回 stock Backend。

测试文件（tests/functional_tests/）：test_biv150_attention.py、test_biv150_decode.py、test_biv150_mixed_paging.py。覆盖 dispatch、回退、BF16 GQA、2D/3D Decode、混合批次及分页正确性。文件名是历史命名，不构成设备门控；本报告不宣称已有 v13 GPU pytest 日志。

历史 BI-V150 环境记录：单卡 32 GiB；Ubuntu 24.04.2、Python 3.12.11、vLLM 0.24.0+empty、vllm-plugin-fl 0.3.0rc0+g13eb9be69、FlagGems 5.3.5、FlagTree 0.6.2a3+iluvatar3.6、PyTorch 2.10.0+corex.4.5.0、Transformers 5.14.1、EvalScope 1.11.1。部署与完整运行命令见 readme.md。沐曦 C500 未完成性能验证。

赛事合规：未通过超参数开启量化/投机采样，未调优官方 benchmark，未删除主要算子选择逻辑或切换至无优化算子，未直接合并新版 FlagOS。优化限于已有 Triton 注意力的分块与分页索引；不裁剪注意力范围。

证据（仓库 Co_Kaggle/g7/tests_BI-V150/）：Optimization/v13；profiling/v13/{4k16k_rawruns.csv,4k16k_summary.csv,summary.txt}；4k16K_summary_benchmark.csv；4k16K_rawrun_benchmark.csv；profiling/v0/profile_runs/attention_source_sections.txt（原算子配置）；HW_SW_info.txt；FlagOS_Env.txt；math_500_eval_benchmark.txt（历史环境与命令）。
