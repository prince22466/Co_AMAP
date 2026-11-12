# 技术报告（v8）

**优化对象：** MiniCPM5-2B  
**实验平台：** 天数 BI-V150  
**基础框架：** FlagOS / vLLM-FL  

### 方案设计思路

#### 1. 优化目标与约束

本方案面向 MiniCPM5-2B 在天数 BI-V150 上的高并发推理场景，以提升端到端总吞吐量（Total tokens/s）为首要目标，同时满足比赛规定的模型精度和时延要求。优化基于赛事指定的 FlagOS 推理框架，保持模型权重、推理参数、测试数据集和官方 benchmark 脚本不变。

比赛采用 4K 和 16K 两种性能测试场景，要求 MATH-500 Level 3 准确率不低于 0.95（官方基线为 0.962），并对首 Token 时延（TTFT）设置相对于基线的 1% 容差要求。基础代码版本按赛事要求使用 `vllm-plugin-FL` 的 `flagos-2026-s2` 分支和 FlagGems 的 `v5.3.5` 标签。

#### 2. 性能瓶颈与优化方向

大语言模型推理包括 Prefill（提示词预填充）和 Decode（逐 Token 解码）两个阶段。两阶段的计算形态和访存特征不同，需要分别针对实际瓶颈进行优化。

- **Prefill 阶段：** 长序列注意力涉及大量 Q/K/V 分块计算。原有 GQA（Grouped-Query Attention）分块方式对同一 KV 数据的复用不足，增加了重复访存及分块执行开销。
- **Decode 阶段：** 每次生成一个 Token 时，需要访问历史 KV Cache。原有分页 KV 地址计算会在同一物理页内重复查询页表，增加不必要的地址计算和页表加载。
- **长上下文服务：** 16K 高并发场景还受到 KV Cache 容量、请求排队、抢占及 Prefill/Decode 干扰的共同影响，因此不能仅依据单个算子的加速比判断最终服务吞吐量。

基于上述分析，本方案采用**面向推理阶段和张量形状的算子特化策略**，在同一个 Triton Unified Attention 算子中分别优化 Prefill 的 KV Tile 复用和 Decode 的物理页索引复用。

#### 3. 方案选择

开发过程中分别探索了注意力分块、RMSNorm、分组分页访问、KV Cache 写入分块及 GEMM 分组调度等方向，并通过算子测试及端到端实验筛选有效方案。

最终选择 **v1 + v5** 作为 v8：

| 贡献 | 优化方向 | v8 是否采用 |
|---|---|---|
| v1 | BF16 GQA Prefill 分块优化 | 是 |
| v5 | Pure Decode 分页 KV 索引复用 | 是 |
| v3 | RMSNorm 相关优化 | 否 |
| v4 | 分组分页访问优化 | 否 |
| v6 | KV Cache 分块优化 | 否 |
| v7 | GEMM 分组调度优化 | 否 |

未纳入的优化在当前组合中未表现出充分、一致的服务级收益。最终方案保留两项具有实际吞吐收益的注意力算子优化，以降低集成复杂度和回归风险。

### 算法原理及优化策略

#### 1. 基础算法：GQA 与分页 KV Cache

GQA 允许多个 Query Head 共享同一组 Key/Value Head。MiniCPM5-2B 在目标注意力配置下具有 16 个 Query Head、2 个 KV Head，Head Dimension 为 128，每个 KV Head 对应 8 个 Query Head。

对于一个 Query Head，注意力计算可表示为：

$$
O = \operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d}}+M\right)V
$$

其中 $d$ 为 Head Dimension，$M$ 为注意力掩码。不同 Query Head 可以复用相同 KV Head 的数据，但仍分别计算其注意力权重及输出。

vLLM 使用分页 KV Cache，通过逻辑页到物理页的映射定位历史 K/V 数据。对于逻辑 Token 位置 $t$ 和页大小 $B$，其物理页索引与页内偏移分别为：

$$
p(t)=\operatorname{BlockTable}\left[\left\lfloor t/B\right\rfloor\right],
\qquad r(t)=t\bmod B
$$

本方案不改变上述注意力公式、逻辑页映射、因果掩码及输出语义，而是优化 GPU 上的分块执行和物理页地址生成。

#### 2. Prefill：BF16 GQA 分块及 KV Tile 复用（v1）

原有 Triton Unified Attention 算子在目标 GQA 配置下使用 `BLOCK_M = 16`，每个 KV Head 对应 8 个 Query Head，因此一个 Query Block 包含的 Query 位置数为：

$$
BLOCK_Q = BLOCK_M / 8 = 2
$$

v8 在满足条件时将 `BLOCK_M` 增大至 32，使 `BLOCK_Q` 增大至 4。

| 分块参数 | 原始实现 | v8 |
|---|---:|---:|
| Query Head / KV Head 比例 | 8 | 8 |
| `BLOCK_M` | 16 | 32 |
| `BLOCK_Q` | 2 | 4 |

通过在同一个注意力计算块中处理更多 Query 位置，提高对已加载 KV Tile 的复用程度，减少重复加载同一 KV Tile 的机会，同时降低 Query Block 数量。实际收益取决于序列长度、分块边界及 GPU 资源占用，并不意味着全部算子耗时减半。

该优化通过 `optimize_prefill_gqa` 条件启用，适用范围包括：

- `max_seqlen_q > 1`，即包含 Prefill 的批次（也包括符合条件的混合批次）；
- Q/K/V 数据类型均为 BF16，16 Q Heads、2 KV Heads、Head Dimension 128；
- 非量化 KV Cache，因果注意力，且不使用 Tensor Descriptor；
- 不使用逐序列因果模式、Sliding Window、Chunk Lookback、多模态前缀、ALiBi、QQ Bias、Sink Tokens、Soft Cap 或输出量化等不兼容特性。

不符合条件的调用沿用原有分块选择逻辑。该优化仅调整算子内部的计算分块，不修改模型超参数，不裁剪注意力范围，也不引入稀疏近似。

#### 3. Decode：分页 KV 物理页索引复用（v5）

Decode 阶段需要反复读取历史 KV Cache。原有实现对一个 KV Tile 中的各 Token Lane 计算对应物理页索引，而同一页中的多个 Token 实际共享同一个物理页索引。

v8 在现有 Unified Attention Kernel 内引入编译期特化参数 `DECODE_PAGE_LOOKUP`，按照 KV Tile 与 KV 页的大小关系复用页表查询结果：

| 场景 | 标准 16-Token KV 页示例 | 优化方式 |
|---|---|---|
| 2D Decode | 32-Token Tile 覆盖 2 个页 | 加载 2 个物理页索引，通过 `tl.gather` 分配给相应 Token Lane |
| 3D Segmented Decode | 16-Token Tile 位于 1 个页 | 加载 1 个物理页索引，并广播至整个 Tile |
| 不符合特化条件 | 任意未支持配置 | 保留原有逐 Token 页表查询方式 |

通过将相同物理页的索引加载合并，减少重复的页表访存和地址生成指令。各 Token 的页内偏移、有效性掩码及 K/V 读取语义保持不变；不减少参与注意力计算的历史 Token。

特化仅在以下条件全部成立时启用：

- `max_seqlen_q == 1`，即 Pure Decode；
- BF16 Q/K/V、16 Q Heads、2 KV Heads、Head Dimension 128；
- KV 页大小为正的 2 的幂，且页大小与 Tile 大小之间满足整数倍对齐关系；
- 非量化 KV Cache，不使用 Tensor Descriptor。

该特化同时适用于符合条件的 2D 和 3D Decode 路径，但不改变框架原有的 2D/3D 选择逻辑，也不改变分段 Softmax 的归约过程。

#### 4. 两项优化的协同关系与创新点

Prefill 分块优化以 `max_seqlen_q > 1` 为条件，Decode 页表复用以 `max_seqlen_q == 1` 为条件，两者在同一次算子调用中互斥。在混合批次中，仅在符合条件时启用 Prefill 分块优化，不强行启用 Pure Decode 特化。

本方案的主要技术贡献在于：在同一个生产注意力算子中，根据推理阶段和合法输入形状分别优化 KV 数据复用和页表索引复用，并保留不支持配置的原有执行路径。两项优化均不改变注意力的数学语义，可分别进行正确性与性能验证。其思路可供其他 GQA 和分页注意力工作负载参考，但本报告不宣称未经测试的模型或硬件具有同等收益。

### 实现架构

#### 1. 整体架构

v8 沿用 vLLM 的 Attention Metadata、KV Cache 更新机制和 Backend 接口，在 Iluvatar 平台的现有 Triton 注意力路径内集成优化。

```text
vllm serve（官方规定的启动参数）
    |
    v
vLLM / vllm-plugin-FL Dispatch
    |
    |-- 显式请求 FlagGems Attention --> 原有 FlagGems 路径
    |
    |-- Iluvatar 平台
    |       |
    |       v
    |   BIV150TritonAttentionBackend / Impl
    |       |  复用原有 Attention Metadata、KV Cache 更新及接口
    |       v
    |   Triton Unified Attention Launcher
    |       |
    |       |-- 符合条件的 Prefill：GQA 分块优化
    |       |-- 符合条件的 Pure Decode：分页 KV 索引复用
    |       |-- 其他配置：保留原有算子执行逻辑
    |
    |-- 其他厂商 --> 原有 Triton Attention 路径
```

图中 Prefill 与 Decode 特化是同一个算子中的不同执行配置，不是两个独立的注意力算子。

#### 2. 核心代码及职责

以下路径均相对于 v8 源码目录中的 `vllm-plugin-FL/`：

| 源码文件 | 实现职责 |
|---|---|
| `vllm_fl/dispatch/backends/flaggems/flaggems.py` | 在 Iluvatar 平台选择包含实际优化的 BI-V150 Attention Backend；保留显式 FlagGems Attention 优先级和其他厂商的 Triton 回退路径。 |
| `vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py` | 实现 `BIV150TritonAttentionBackend` 和 `BIV150TritonAttentionImpl`，复用 vLLM 元数据、KV Cache 更新机制和注意力接口，调用统一算子入口。 |
| `vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py` | 集成实际算子优化：`optimize_prefill_gqa` 控制 Prefill 分块，`DECODE_PAGE_LOOKUP` 控制 Decode 页表复用。 |

最终 v8 仅保留一个生产 Attention Backend 和一个 Unified Attention Launcher，不包含独立的 v4、v6 或 v7 优化算子，也不包含为不同实验版本重复构建的注意力接口。

#### 3. 实验环境与依赖

历史开发环境快照（2026-10-25）记录如下：

| 项目 | 配置 |
|---|---|
| GPU | 天数 BI-V150 × 1，32 GiB 显存 |
| 操作系统 | Ubuntu 24.04.2 LTS |
| Python | 3.12.11 |
| vLLM | 0.24.0+empty |
| vllm-plugin-fl | 0.3.0rc0+g13eb9be69 |
| FlagGems | 5.3.5 |
| PyTorch | 2.10.0+corex.4.5.0 |
| flagtree | 0.6.2a3+iluvatar3.6 |
| 驱动 | 4.5.0 |

上述版本来自开发环境记录，最终提交时需从实际复现环境确认软件版本、基础框架版本及提交源码的 Git SHA。

模型路径为 `/workspace/MiniCPM5-2B`，精度测试数据集路径为 `/workspace/evalscope-datasets/math_500`。

#### 4. 功能测试与运行流程

在部署 v8 源码后，针对实际采用的两项优化运行以下功能测试：

- `vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py`：验证 Prefill、Backend 集成及回退条件。
- `vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py`：验证 Decode 页表复用、特化条件及回退行为。

随后按照提交的 [复现说明](readme.md)，将提交的 FlagGems 和 vllm-plugin-FL 源码部署至官方环境，使用原有编译执行流程及比赛指定启动参数启动 vLLM，运行官方性能和精度测试。开发历史记录中上述功能测试已成功执行；正式提交仍需归档最终版本的测试日志并确认实际 Engine 进程加载了预期实现。

### **其他说明**

#### 1. 实验设置

性能测试采用未修改的官方 `benchmark_throughput_serve.py`，使用以下固定测试场景：

| 场景 | 输入 Token 数 | 输出 Token 数 | 并发数 | 请求数 |
|---|---:|---:|---:|---:|
| 4K | 4,096 | 1,024 | 64 | 256 |
| 16K | 16,384 | 1,024 | 64 | 128 |

官方测试参数为 `--test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'`。服务使用规定的 `cudagraph_mode=FULL_DECODE_ONLY`、`--gpu-memory-utilization 0.85`、`--served-model-name minicpm` 和 `--max-model-len 131072` 等参数。

精度使用官方 EvalScope 命令在 MATH-500 Level 3（105 个样本）上评估，采样参数为 temperature 1.0、top_p 0.95、max_tokens 32768。

为确保对比有效，最终基线和优化版本应在相同镜像、驱动、模型权重、GPU 和服务参数下完成完整测试，并记录后台负载、KV Cache 压力、Prefix Cache 状态及请求失败情况。

#### 2. 性能测试结果与效果分析

以下为项目归档的 v8 完整测试（full-run）汇总结果，与历史自测基线第一组汇总均值对比。基线采用各工作负载对应的最近三次成功运行汇总均值。

| 测试场景 | 自测基线（tokens/s） | v8（tokens/s） | 绝对增益（tokens/s） | 相对增益 |
|---|---:|---:|---:|---:|
| 4K | 1,881.95 | 2,331.29 | +449.34 | **+23.88%** |
| 16K | 861.88 | 1,251.22 | +389.34 | **+45.17%** |

相对增益计算方式：`(v8 吞吐量 / 基线吞吐量 - 1) × 100%`，保留两位小数。

| 测试场景 | 自测基线平均 TTFT（ms） | v8 平均 TTFT（ms） |
|---|---:|---:|
| 4K | 12,309.34 | 8,738.76 |
| 16K | 636,512.47 | 428,970.85 |

在上述历史自测对比中，两个场景的总吞吐量均有提升，平均 TTFT 均有所下降。16K 的绝对时延较高，与高并发长上下文服务中的 KV Cache 容量压力和请求调度有关。

**证据范围：** 仓库目前保存了历史基线原始数据、汇总 CSV 和 v8 完整测试的文字汇总，但尚未归档最终 v8 完整测试的原始 CSV 和服务端 Trace。上述数据为内部自测的初步结果，不代表已通过官方复现，也不足以证明统计显著性或满足同条件下的官方 TTFT 容差要求。

#### 3. 精度测试结果

| 项目 | 准确率 | 说明 |
|---|---:|---|
| 官方基线 | 96.2% | 原报告记录的基线 |
| 比赛最低要求 | 95.0% | MATH-500 Level 3 |
| v8 内部汇总 | 98.1% | 最终 v8 的完整 EvalScope 原始报告尚未归档 |
| v1 开发记录 | 98.1% | 已归档的 105 样本 Level 3 评估汇总 |

v8 内部汇总报告的准确率高于比赛最低要求。由于最终 v8 的完整 EvalScope 报告尚未附于仓库，不能将已有 v1 的精度日志直接作为 v8 的正式验证证据。另有早期自测基线记录为 97.1%，与官方基线应分别看待。

最终提交需在 v8 实际运行环境中完成 105 个样本的评估，确认准确率不低于 0.95，并保存完整结果 JSON、日志和退出状态。功能测试通过不能替代模型精度测试。

#### 4. 长上下文瓶颈及探索实验

历史 v5 诊断表明，模型 KV Cache 约占用 42 KiB/Token，服务分配的 KV Cache 容量约为 522,960 Token（20.95 GiB）。若 64 个请求的 16K 前缀完全独立且同时驻留，则仅前缀就需要 1,048,576 Token（42 GiB），超过当时观察到的缓存容量。实际调度可通过 Chunked Prefill、前缀复用等机制改变驻留需求。

开发期间观察到 KV Cache 接近饱和、请求等待及抢占等现象，表明长上下文的端到端表现不仅取决于算子计算效率。Decode 页表复用能够减少算子内部开销，但不能消除 KV Cache 容量限制。CPU/GPU 启动间隙、抢占、KV 数据移动及 4K/16K 不同的缓存行为也属于后续值得分析的方向，并非 v8 新增的调度优化。

以下为开发阶段用于筛选组合的 half-run 汇总（单位：Total tokens/s）：

| 优化组合 | 4K | 16K | 选择结果 |
|---|---:|---:|---|
| v1 | 1,958.15 | 1,205.11 | 基础方案 |
| v1 + v3 | 1,997.06 | 1,197.71 | 未采用 |
| v1 + v4 | 1,990.66 | 1,193.87 | 未采用 |
| **v1 + v5** | **2,128.76** | **1,238.72** | **采用为 v8** |
| v1 + v5 + v6 | 2,082.80 | 1,246.19 | 未采用 |
| v1 + v5 + v7 | 2,094.59 | 1,247.47 | 未采用 |

这些实验仅用于开发筛选，与前文的官方场景完整测试汇总不等价。加入 v6 或 v7 后，16K 有小幅提升但 4K 回落，未体现足够明确的整体收益，因此未纳入最终 v8。

#### 5. 赛事合规性

本方案的优化来自现有 Triton Unified Attention 算子的实际分块和页表访存优化，不依赖测试请求操控或比赛评估流程调整。

- 未通过修改超参数开启量化或投机采样。
- 未修改官方 benchmark 脚本、测试场景、模型权重或规定的服务启动参数。
- 未通过切换至未经优化的算子或删除框架主要算子选择逻辑获得性能提升；保留原有优先级及回退机制。
- 未通过直接合并 FlagOS 推理框架最新分支获得性能提升。

本报告的合规说明针对所描述的 v8 设计与实现；最终仍需核对打包的实际代码版本及官方复现结果。

#### 6. 提交、复现与适用范围

按比赛要求，最终提交 ZIP 的目录结构为：

```text
<registered-team-name>/
  FlagGems/
  vllm-plugin-FL/
  readme.md
  report.pdf
```

正式提交前需完成以下复现与证据归档工作：

1. 确认基础框架版本、提交源码 Git SHA、软件依赖及实际 Engine 加载的算子路径。
2. 归档最终 v8 完整测试原始 CSV、服务日志、命令记录、GPU 指标、成功请求数及失败/超时事件；尽可能补充同条件基线复测。
3. 归档最终 v8 的 MATH-500 Level 3 完整评估报告及错误分析，核实精度和 TTFT 要求。
4. 核对参赛队伍信息，基于原始数据补充经审核的图表（如需要），校对导出的技术报告 PDF。
5. 在干净的官方环境中复现完整提交包，并按赛事要求提交。

**其他硬件平台：** 由于测试过程中遇到错误，本优化方案未能在沐曦-曦云 C500（显存 64 GB）上完成性能验证。本报告仅针对天数 BI-V150 与 MiniCPM5-2B，不宣称在其他模型或硬件上的性能收益。

#### 7. 参考资料与实验记录

- [赛事规则](https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn)
- [v8 源码与实现说明](../../tests_BI-V150/Optimization/v8/readme.md)
- [v8 内部测试汇总](../../tests_BI-V150/profiling/v8/summary)
- [历史基线原始数据](../../tests_BI-V150/4k16K_rawrun_benchmark.csv)
- [历史基线汇总](../../tests_BI-V150/4k16K_summary_benchmark.csv)
- [v1 性能与 Level 3 精度记录](../../tests_BI-V150/profiling/v1/summary.txt)
- [v5 Decode 与 16K 瓶颈分析](../../tests_BI-V150/profiling/v5/v5_decode_16k_bottleneck_findings.md)
- [硬件及软件环境记录](../../tests_BI-V150/HW_SW_info.txt)
- [FlagOS 环境记录](../../tests_BI-V150/FlagOS_Env.txt)
