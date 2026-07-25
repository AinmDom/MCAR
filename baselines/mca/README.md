# MCA 基线

这里保存 MCA/SUpDEq 的复现入口。它们用于生成 SH only、SUpDEq + SH 和
MCA 三组传统基线，不是 MCAR 残差网络的主项目。

本地依赖位于 `external/SUpDEq/`，HUTUBS 数据位于
`data/HRTF/hutubs/`。

主要入口：

- `matlab/run_mca_demo_export.m`：运行 SUpDEq 自带 KU100 MCA demo；
- `matlab/analyze_contralateral_high_frequency.m`：分析对侧高频误差；
- `matlab/run_hutubs_mca_p91_n3.m`：单被试严格复现；
- `matlab/run_hutubs_mca_batch.m`：96 被试、多个稀疏阶数批处理；
- `matlab/aggregate_hutubs_mca_results.m`：汇总批处理结果。

运行示例：

```powershell
matlab -batch "addpath('baselines/mca/matlab'); run_hutubs_mca_p91_n3"
```

批处理检查点写入 `artifacts/mca_reproduction/`，精选汇总结果写入
`results/baselines/mca/`。
