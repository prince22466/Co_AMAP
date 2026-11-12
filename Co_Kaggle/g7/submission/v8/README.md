# submission version - v8
# code location - Co_Kaggle/g7/tests_BI-V150/Optimization/v8


# submission .zip file components(according to the competition rules) and corresponding locations
## Official ZIP layout

```text
<team-name>.zip
    ├── FlagGems(code location)
    ├── vllm-plugin-FL(code location
    ├── readme.md(this folder)
    └── report.pdf(this folder)
```

代码仓库：
vllm-plugin-FL：包含所有源代码和编译脚本，代码规范，确保在指定环境中能够顺利运行。代码需具有良好的注释和清晰的结构，便于评审人员理解和复现。
【可选】FlagGems：涉及到算子相关改动，提交到FlagGems

技术报告（report.pdf）：完整阐述方案设计思路、算法原理、实现架构、优化策略、实验环境、测试结果与效果分析；图文结合，逻辑清晰，完整支撑作品方案，便于评审人员理解整体技术路线与创新点。
复现说明（readme.md）：清晰列明软硬件环境、依赖版本、环境部署步骤、编译执行流程、数据集准备方式、运行命令与结果校验方法，步骤完整可落地，保障评审人员依据文档能够顺利复现实验结果。

## Compliance

No quantization-by-hyperparameter, speculative decoding, official benchmark edits, framework operator-selection removal, operator substitution without optimization, or wholesale update/merge of newer FlagOS inference code is proposed. The submission report describes only the tuned attention kernel mechanics present in **v8**. The work focuses on throughput subject to accuracy and latency constraints.

Official rules: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn
