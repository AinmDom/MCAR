# SONICOM Q26 v3.4b 分频带 ILD 损失消融

## 结论

v3.4b 验证了分频带 ILD 是有效且高度一致的定向目标。相对固定的 v3.2 epoch 39，
完整水平面方向上的 35-band ILD SmoothL1/MAE 分别降低 `0.7587% / 0.7082%`，均有
43/44 名 validation 被试改善，且 35/35 个频带的聚合误差下降。严格重建的对侧
25° ERB 改善 `0.2821%`、40/44 人改善，是目前几个低风险目标函数消融中最明确的
对侧 ERB 收益。

四项严格指标的均值都没有回退，但对侧 10–20 kHz 幅度 MAE 只改善 `0.0055%`，没有
统计显著性。因此 v3.4b 是当前最强的定向 loss 消融，不应被表述为对 v3.2 epoch 39
的全面替代。

本实验只使用 262 名 train 和 44 名 validation 被试；没有读取已消费的 test，
`test_subject_count_read=0`。

## 损失定义

对每个耳朵、方向和 ERB proxy band，将幅度 dB 转为线性能量并按频带权重汇总，再转回
band energy dB。定义左右耳 band ILD 为
$I_b=E_{L,b}-E_{R,b}$，新增目标为预测与 reference 的
$\operatorname{SmoothL1}_{\beta=0.5\,\mathrm{dB}}(\hat I_b-I_b)$。

41 个 ERB proxy filters 中保留中心频率位于 200 Hz–18 kHz 的 35 个频带；方向仅使用
72 个水平面纯插值方向，并在每名被试内按 solid-angle direction weight 汇总。新增
权重为 `0.10`，原 `ERB/HF/strict broadband ILD=0.75/0.25/0.75` 不变；v3.4a
D1/D2 权重均为 0，以保持独立归因。

## 训练设置

- 起点：v3.2 seed `20260809` epoch 39，源 checkpoint SHA-256 为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`。
- 架构：保持 v3.2 MLP + kernel 7、dilation `1/2/4/8` 局部 CNN。
- 冻结边界：MLP 冻结，只训练 74,402 个局部 CNN 参数。
- 预算：10 epoch × 500 step；96 个固定 validation block。
- 采样：每步 32 个全局插值方向，另采 32 个水平面方向，同时计算严格 HRIR ILD 与
  分频带 ILD。
- 优化：AdamW，学习率 `1e-5`，weight decay `1e-5`，cosine，gradient clip `5`，AMP。
- seed：训练 `20260809`，固定 validation sampler `20260805`。
- 最佳 checkpoint：按预注册 v3.4b total 选择 epoch 7；W&B offline ID `u8cv7js1`。

训练用时 `820.906 s`，optimizer skip 为 0，峰值 CUDA allocated memory 为
`410.870 MiB`。初始 total 为 `0.6896631916`，epoch 7 为 `0.6890342993`。
源到候选的 MLP 参数逐值未变；CNN 相对 L2 改变量为 `0.2989%`。候选 checkpoint
SHA-256 为 `EABFAA1B303F6016F9F8F7507D510813C44C948C81D9C074EECA0A8E84AA4DAF`。

## 44 人严格重建

| 指标 | v3.2 epoch 39 | v3.4b epoch 7 | 相对改善 | v3.4b 更优人数 |
|---|---:|---:|---:|---:|
| 全空间 ERB | 0.872882 dB | 0.872511 dB | +0.0425% | 33/44 |
| 对侧 25° ERB | 1.364204 dB | 1.360356 dB | +0.2821% | 40/44 |
| 对侧高频 | 3.608567 dB | 3.608369 dB | +0.0055% | 27/44 |
| 水平面严格 ILD MAE | 0.596141 dB | 0.593076 dB | +0.5141% | 32/44 |

差值定义为 candidate minus baseline，负值表示 v3.4b 更好。

| 指标 | 均值差 | 95% CI | paired t-test | Wilcoxon | Cohen $d_z$ |
|---|---:|---:|---:|---:|---:|
| 全空间 ERB | -0.0003710 dB | [-0.0005853, -0.0001568] | 0.00112 | 0.00113 | -0.527 |
| 对侧 25° ERB | -0.0038480 dB | [-0.0049010, -0.0027951] | 3.76e-9 | 3.14e-10 | -1.111 |
| 对侧高频 | -0.0001978 dB | [-0.0012683, 0.0008728] | 0.711 | 0.435 | -0.056 |
| 水平面严格 ILD | -0.0030646 dB | [-0.0051269, -0.0010022] | 0.00452 | 0.00334 | -0.452 |

这是 validation 开发集消融且同时比较多个指标。可靠结论是对侧 ERB、全空间 ERB 与
strict ILD 的改善方向一致；HF 的置信区间跨 0，不能宣称改善。

## 全水平面分频带 ILD 诊断

诊断使用 44 名 validation 被试的全部 72 个水平面纯插值方向，每名被试先按方向面积
加权，再对被试等权平均。

| band-ILD 指标 | v3.2 epoch 39 | v3.4b epoch 7 | 相对改善 | 改善人数 |
|---|---:|---:|---:|---:|
| 35-band SmoothL1 | 1.407891 dB | 1.397209 dB | +0.7587% | 43/44 |
| 35-band MAE | 1.629438 dB | 1.617899 dB | +0.7082% | 43/44 |

两项 paired t-test 分别为 `6.45e-19 / 2.33e-19`，Wilcoxon 均为 `2.27e-13`，
Cohen $d_z$ 为 `-2.294 / -2.358`。35/35 个频带的聚合误差均降低，说明实现和目标
均明确生效。相对收益最大的是约 250–500 Hz，6 kHz 附近与高频段收益较弱；因此
band ILD 的整体改善没有转化为明显的 10–20 kHz 对侧逐点幅度改善。

## 版本比较与下一步

| 版本 | 全空间 ERB | 对侧 25° ERB | 对侧 HF | strict ILD |
|---|---:|---:|---:|---:|
| v3.2.1 解冻 MLP | +0.0540% | -0.0034% | +0.0332% | +0.6658% |
| v3.4a 谱差分 | +0.0303% | +0.0628% | +0.0026% | +0.4852% |
| v3.4b band ILD | +0.0425% | +0.2821% | +0.0055% | +0.5141% |

正值为相对 v3.2 epoch 39 的改善，负值为回退。v3.4b 的对侧 ERB 收益最大，且与其
目标指标高度一致；v3.2.1 在全空间 ERB、HF 与 strict ILD 上数值略好，但对侧 ERB
轻微回退。现阶段没有一个开发版本在所有指标上绝对占优，仍以 v3.2 epoch 39 为固定
开发基准。

下一步应从同一基准独立测试 notch-aware loss；如果 notch 也能通过完整 validation
严格重建，再组合 v3.4a 的谱差分、v3.4b 的 band ILD 与 notch 目标。hard-direction
sampling 放在目标函数验证之后，以免训练分布变化干扰归因。

## 文件

- `summary.json`：严格重建配置与聚合指标。
- `comparison_vs_v32_epoch39.csv`：四项严格指标直接对比。
- `paired_statistics.csv`：四项严格指标的 44 人配对统计。
- `spectral_band_ild_per_subject.csv`：逐被试完整 band-ILD 结果。
- `spectral_band_ild_paired_statistics.csv`：band-ILD 配对统计。
- `spectral_band_ild_by_center_frequency.csv`：逐中心频率聚合结果。
- `spectral_band_ild_diagnostics.json`：band-ILD、checkpoint 与训练诊断。
- `training_history.csv`、`training_report.json`：训练曲线与运行摘要。
- `figures/validation44_metric_overview.png`：44 人四指标总览。
- `figures/validation44_contralateral_hrtf_overview.png`：44 人对侧 HRTF 总览。
