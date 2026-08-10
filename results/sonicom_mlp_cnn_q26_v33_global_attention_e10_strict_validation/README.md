# SONICOM Q26 v3.3 全局频谱上下文消融

## 结论

本次验证的“降采样轻量 attention + 门控全局 delta”在工程上稳定、确实产生了非零
修正，但没有改善 v3.2 epoch 39。两项 ERB 指标出现极小但配对统计明确的退化；高频
和 ILD 的微小均值改善均不显著。因此 v3.3 作为负结果架构消融保留，不替代 v3.2。

实验只使用 262 名 train 和 44 名 validation 被试；test 未读取，
`test_subject_count_read=0`。

## 架构与训练

- 起点：固定的 v3.2 seed `20260809` epoch 39，SHA-256
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`。
- 局部分支：原 kernel 7、dilation `1/2/4/8` CNN，连同 MLP 全部冻结。
- 全局分支：463 频点以 stride 4 降为 116 token，宽度 32、4 头、2 个 pre-norm
  self-attention block，再线性上采样至 463 频点。
- 融合：由局部/全局隐藏特征预测双耳逐频率 sigmoid gate，并以
  `local delta + gate × global delta` 叠加。
- 安全起点：全局输出层零初始化，因此 epoch 0 与源 v3.2 输出逐值相同。
- 可训练参数：22,356；总参数：196,983。
- 训练：10 epoch × 500 step，96 个固定 validation block，学习率 `1e-4`，cosine，
  `ERB/HF/strict ILD=0.75/0.25/0.75`。
- 最佳训练 checkpoint：epoch 9；W&B offline run ID `q401byar`。

初始 validation total 为 `0.6611154843`，epoch 9 为 `0.6611415790`，即 proxy
也没有超过未训练起点。训练用时 `1717.3 s`，optimizer skip 为 0，峰值 CUDA
allocated memory 为 `76.50 MiB`。

## 44 人严格重建

| 指标 | v3.2 epoch 39 | v3.3 epoch 9 | 相对改善 | v3.3 更优人数 |
|---|---:|---:|---:|---:|
| 全空间 ERB | 0.872882 dB | 0.873159 dB | -0.0317% | 12/44 |
| 对侧 25° ERB | 1.364204 dB | 1.364686 dB | -0.0353% | 14/44 |
| 对侧高频 | 3.608567 dB | 3.608384 dB | +0.0051% | 25/44 |
| 水平面严格 ILD MAE | 0.596141 dB | 0.595707 dB | +0.0727% | 22/44 |

全空间 ERB 的 candidate-baseline 均值差为 `+0.0002765 dB`，配对 t-test
`p=2.74e-5`；对侧 25° ERB 为 `+0.0004818 dB`，`p=0.00228`。二者虽然绝对值
很小，但方向明确为退化。高频与 ILD 的均值差分别为 `-0.0001836 dB` 和
`-0.0004334 dB`，配对 t-test `p=0.139 / 0.476`，不能视为可靠改善。

## 全局分支诊断

冻结的 MLP 与局部 CNN 在候选 checkpoint 中最大参数差为严格的 `0.0`。44 人全部
预测上，v3.3 相对 v3.2 的全局门控修正统计为：

- 平均绝对 delta：`0.012006 dB`；
- RMS delta：`0.015279 dB`；
- 绝对 delta 的 95% 分位：`0.029029 dB`；
- 最大绝对 delta：`0.093135 dB`；
- 10–20 kHz 平均绝对 delta：`0.013215 dB`。

这说明新增路径不是“没有参与前向”，而是学到的全局修正很小，且主要没有对准最终
ERB 目标。若继续研究这条路线，下一项有解释力的消融应是保持同一架构、联合微调
局部 CNN 与全局分支，而不是继续盲目增大 attention 宽度或 channel 数。

## 文件

- `summary.json`：严格评估配置与聚合结果。
- `comparison_vs_v32_epoch39.csv`：四项指标的直接对比。
- `paired_statistics.csv`：44 人配对统计。
- `global_context_diagnostics.json`：预测 delta、门控和 checkpoint 完整性诊断。
- `global_delta_by_frequency.csv`：逐频率全局 delta。
- `training_history.csv`、`training_report.json`：训练曲线与运行摘要。
- `figures/validation44_metric_overview.png`：44 人指标总览。
- `figures/validation44_contralateral_hrtf_overview.png`：44 人对侧 HRTF 总览。
