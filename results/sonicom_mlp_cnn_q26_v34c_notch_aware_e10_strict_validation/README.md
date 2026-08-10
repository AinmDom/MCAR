# SONICOM Q26 v3.4c 多尺度 notch-aware 损失消融

## 结论

v3.4c 证明可微 notch-depth map 能稳定改善局部谱凹陷。相对固定 v3.2 epoch 39，
44 名 validation 被试全部 767 个纯插值方向上的 notch-depth MAE 改善 `0.1139%`，
44/44 人改善；三个尺度也全部 44/44 人改善。严格重建的四项均值均未回退，其中对侧
25° ERB 和水平面 strict ILD 的配对统计一致显著。

但 notch 目标的绝对收益很小，且对侧 10–20 kHz 幅度 MAE 的改善仍不显著。v3.4c
适合作为正向目标函数消融，不足以全面替代 v3.2 epoch 39。

本实验只使用 262 名 train 和 44 名 validation 被试；没有读取已消费的 test，
`test_subject_count_read=0`。

## 损失定义

对于半径 $r$，局部 notch 原始深度定义为
$d_f^{(r)}=(H_{f-r}+H_{f+r})/2-H_f$。使用
$\tilde d_f^{(r)}=\tau\operatorname{softplus}((d_f^{(r)}-d_0)/\tau)$ 代替硬阈值，
其中 $d_0=1\,\mathrm{dB}$、$\tau=0.5\,\mathrm{dB}$。预测与 reference 的
notch-depth map MAE 会同时惩罚 notch 变浅、变深、位置偏移和伪 notch。

频率范围为 4–18 kHz，尺度半径为 4/8/16 个频点，即约 172/345/689 Hz。损失使用
全部纯插值方向并按 solid-angle direction weight 汇总。新增权重为 `0.30`；原
`ERB/HF/strict ILD=0.75/0.25/0.75` 不变，v3.4a D1/D2 与 v3.4b band-ILD 权重均为 0。

## 训练设置

- 起点：v3.2 seed `20260809` epoch 39，源 SHA-256 为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`。
- 架构：保持 v3.2 MLP 与 kernel 7、dilation `1/2/4/8` 局部 CNN。
- 冻结边界：MLP 冻结，只训练 74,402 个局部 CNN 参数。
- 预算：10 epoch × 500 step；96 个固定 validation block。
- 优化：AdamW，学习率 `1e-5`，weight decay `1e-5`，cosine，gradient clip `5`，AMP。
- 最佳 checkpoint：按预注册 total 选择 epoch 7；W&B offline ID `iohnfzkr`。

训练用时 `886.528 s`，optimizer skip 为 0，峰值 CUDA allocated memory 为
`399.790 MiB`。初始 total 为 `0.6913994942`，epoch 7 为 `0.6909071778`。
MLP 参数逐值未变，CNN 相对 L2 改变量为 `0.2802%`。候选 checkpoint SHA-256 为
`5A2A29B38D49717FBA2E2C459F7A987F18E799CA0EB58AD4F8EDAE293717C123`。

## 44 人严格重建

| 指标 | v3.2 epoch 39 | v3.4c epoch 7 | 相对改善 | v3.4c 更优人数 |
|---|---:|---:|---:|---:|
| 全空间 ERB | 0.872882 dB | 0.872650 dB | +0.0267% | 24/44 |
| 对侧 25° ERB | 1.364204 dB | 1.362914 dB | +0.0946% | 29/44 |
| 对侧高频 | 3.608567 dB | 3.608108 dB | +0.0127% | 28/44 |
| 水平面严格 ILD MAE | 0.596141 dB | 0.592944 dB | +0.5362% | 30/44 |

差值定义为 candidate minus baseline，负值表示 v3.4c 更好。

| 指标 | 均值差 | 95% CI | paired t-test | Wilcoxon | Cohen $d_z$ |
|---|---:|---:|---:|---:|---:|
| 全空间 ERB | -0.0002327 dB | [-0.0004418, -0.0000236] | 0.0300 | 0.0623 | -0.338 |
| 对侧 25° ERB | -0.0012899 dB | [-0.0021394, -0.0004403] | 0.00378 | 0.00378 | -0.462 |
| 对侧高频 | -0.0004595 dB | [-0.0014552, 0.0005361] | 0.357 | 0.138 | -0.140 |
| 水平面严格 ILD | -0.0031964 dB | [-0.0053911, -0.0010017] | 0.00531 | 0.00426 | -0.443 |

这是 validation 开发集消融。对侧 ERB 和 strict ILD 的方向最可靠；全空间 ERB 的
Wilcoxon 略高于 `0.05`，而 HF 的置信区间跨 0，不能宣称 HF 显著改善。

## 全方向 notch-depth 诊断

| 尺度 | v3.2 | v3.4c | 相对改善 | 改善人数 |
|---|---:|---:|---:|---:|
| 4 bins（172 Hz） | 0.230226 dB | 0.229861 dB | +0.1588% | 44/44 |
| 8 bins（345 Hz） | 0.458085 dB | 0.457578 dB | +0.1105% | 44/44 |
| 16 bins（689 Hz） | 0.826155 dB | 0.825303 dB | +0.1032% | 44/44 |
| 三尺度平均 | 0.504822 dB | 0.504247 dB | +0.1139% | 44/44 |

三尺度平均的 paired t-test 为 `4.14e-29`，Wilcoxon 为 `1.14e-13`，Cohen
$d_z=-4.19$。这说明目标确实生效且方向高度一致，但绝对改变量只有 `0.000575 dB`，
应避免把统计稳定性误写成较大的感知收益。

## 与独立 loss 消融比较

| 版本 | 主目标改善 | 全空间 ERB | 对侧 ERB | 对侧 HF | strict ILD |
|---|---:|---:|---:|---:|---:|
| v3.4a 谱差分 | D1/D2 +1.164%/+3.821% | +0.0303% | +0.0628% | +0.0026% | +0.4852% |
| v3.4b band ILD | SmoothL1 +0.7587% | +0.0425% | +0.2821% | +0.0055% | +0.5141% |
| v3.4c notch | notch-depth +0.1139% | +0.0267% | +0.0946% | +0.0127% | +0.5362% |

v3.4b 是对侧 ERB 最强的单项损失；v3.4c 的严格 HF 与 ILD 数值改善在三个 loss
消融中最大，但 HF 仍不显著。下一步可以做一次等比例降权组合：
`D1/D2/band-ILD/notch = 0.125/0.075/0.05/0.15`，把新增目标总贡献控制在约 `0.045`，
避免直接叠加原权重导致辅助损失占比过大。

## 文件

- `summary.json`：严格重建配置与聚合指标。
- `comparison_vs_v32_epoch39.csv`：四项严格指标直接对比。
- `paired_statistics.csv`：四项严格指标的 44 人配对统计。
- `notch_depth_per_subject.csv`：逐被试完整 notch-depth 结果。
- `notch_depth_paired_statistics.csv`：notch-depth 配对统计。
- `notch_depth_by_radius.csv`：逐尺度结果。
- `notch_depth_diagnostics.json`：notch、checkpoint 与训练诊断。
- `training_history.csv`、`training_report.json`：训练曲线与运行摘要。
- `figures/validation44_metric_overview.png`：44 人四指标总览。
- `figures/validation44_contralateral_hrtf_overview.png`：44 人对侧 HRTF 总览。
