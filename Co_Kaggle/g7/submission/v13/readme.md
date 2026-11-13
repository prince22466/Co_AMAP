# 复现说明（v13）

参赛团队：prince22466  
模型：MiniCPM5-2B  
优化位置：vllm-plugin-FL（FlagGems 与 v0 一致）

## 1. 环境

使用比赛提供的 BI-V150 镜像与模型，不升级框架或改变官方参数。历史环境记录如下（实际评审环境以官方镜像为准）：

| 项目 | 记录版本 |
|---|---|
| GPU | Iluvatar BI-V150 × 1，32 GiB |
| OS | Ubuntu 24.04.2 LTS |
| Python | 3.12.11 |
| vLLM | 0.24.0+empty |
| vllm-plugin-fl | 0.3.0rc0+g13eb9be69（基础版本） |
| FlagGems | 5.3.5 |
| FlagTree | 0.6.2a3+iluvatar3.6 |
| PyTorch | 2.10.0+corex.4.5.0 |
| Transformers | 5.14.1 |
| EvalScope | 1.11.1 |

历史环境记录见仓库 `tests_BI-V150/HW_SW_info.txt`、`FlagOS_Env.txt`。依赖由官方镜像提供，无需自行替换 Triton/FlagTree 或安装其他版本。沐曦 C500 尚未完成性能验证。

## 2. 部署与启动

解压提交 ZIP，进入包含 `FlagGems/`、`vllm-plugin-FL/` 的团队目录。以下以 `/workspace/prince22466` 为例，按实际解压位置调整。保留官方模型与 benchmark 目录；通过 PYTHONPATH 加载提交源码，无需覆盖官方 benchmark 或修改框架选算子配置。

```bash
cd /workspace/prince22466
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl

python3 - <<'PY'
import inspect
import flag_gems
import vllm_fl
from vllm_fl.dispatch.backends import triton_attention, triton_unified_attention
for module in (flag_gems, vllm_fl, triton_attention, triton_unified_attention):
    print(module.__name__, inspect.getfile(module))
PY
```

检查上述模块均来自提交目录。使用官方环境的默认选算子策略；不要额外强制 Attention Backend 或开启量化/投机采样。算子由 Triton 在运行时 JIT 编译，无新增手工编译步骤。

```bash
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

在默认 dense Triton 路径下，启动日志应显示 dispatch 最终选中 `vllm_fl.dispatch.backends.triton_attention.OptimizedTritonAttentionBackend`。其他后端的原有选择逻辑不变。

## 3. 功能测试（可选，非正式评测）

在相同源码环境下、独占 GPU 时运行：

```bash
cd /workspace/prince22466/vllm-plugin-FL
python3 -m pytest -q -s --noconftest \
  tests/functional_tests/test_biv150_attention.py \
  tests/functional_tests/test_biv150_decode.py \
  tests/functional_tests/test_biv150_mixed_paging.py
```

测试覆盖原有 dispatch 优先级、形状/特性回退、BF16 GQA、2D/3D Decode 及混合分页；测试脚本不是性能 benchmark。需在评审环境实际运行并保存结果。

## 4. 正式吞吐测试

服务就绪后，在另一个终端执行官方原始脚本（不修改脚本、不使用 half-run）：

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

核对每次请求成功数（4K 为 256、16K 为 128）、原始 CSV 与 summary CSV。脚本自带四次运行及 SKIP_FIRST=1 汇总规则，不作改动。

| 场景 | v13 历史 full-run Total tok/s |
|---|---:|
| 4K | 2752.60 |
| 16K | 1890.18 |

实际复测允许硬件状态与运行波动。结果来源：仓库 `tests_BI-V150/profiling/v13/4k16k_rawruns.csv`、`4k16k_summary.csv`。

## 5. 正式精度测试

使用官方预置数据 `/workspace/evalscope-datasets/math_500/test.jsonl`，按比赛 MATH-500 Level 3 口径执行（与吞吐 benchmark 分开运行）：

```bash
evalscope eval \
  --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 \
  --ignore-errors
```

保留完整评测日志和报告，核对 Level 3 样本覆盖率、异常请求和最终准确率（门槛 ≥95%）。v13 仓库的 `profiling/v13/summary.txt` 仅记录门槛判断为 True，未提供精确分数或完整日志，不以其他版本精度替代。
