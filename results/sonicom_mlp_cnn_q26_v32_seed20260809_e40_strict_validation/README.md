# SONICOM Q26 v3.2 seed 20260809：40 epoch validation 严格重建

本实验从三 seed、各 10 epoch 的稳定性诊断中，按固定 validation total loss 最小值
预先选择 seed `20260809`，随后用完全相同的 v3.2 配方重新训练 40 epoch。模型选择
只依据固定 validation 复合损失；已消费的 test 没有参与训练、选模、推理或重建。

## 训练与 checkpoint

- 预算：`40 epoch × 500 step`，每轮 96 个固定 validation block；
- training seed：`20260809`；validation sampler seed：`20260805`；
- 损失权重：`ERB/HF/strict HRIR ILD = 0.75/0.25/0.75`；
- 最佳 checkpoint：epoch 39，validation total loss `0.661115`；
- 最佳 validation proxy：residual `2.542327 dB`、ERB `1.021642 dB`、
  对侧高频 `3.578575 dB`、strict ILD `0.589368 dB`；
- 用时 `5574.459 s`，峰值 CUDA allocated memory `399.222 MiB`；
- AMP optimizer skip 为 `3/20000`，最终 scale 为 `65536`；
- W&B 本地 offline run ID：`odwrgrln`；test 读取数：0。

epoch 40 的 strict ILD proxy 为 `0.589362 dB`，仅比 epoch 39 低约
`0.000006 dB`，但其 validation total loss `0.661117` 略高。因此严格遵循
预注册主规则，使用 epoch 39 的 `best.pt` 重建，而不事后按单项 ILD 换 checkpoint。

## 44 人严格 validation 结果

所有数值均由预测幅度加回 MCA magnitude、保留 MCA phase 后重建 HRIR 得到；ERB
使用 `AKerbError`，高频范围为 `10–20 kHz`，ILD 为水平面纯插值方向上的完整
HRIR 能量比 MAE。数值越低越好。

| 指标 | v3.2 locked epoch 6 (dB) | seed 20260809 epoch 39 (dB) | 相对改善 | 改善被试数 |
|---|---:|---:|---:|---:|
| 全空间 ERB | 0.884652 | **0.872882** | **1.330%** | 42/44 |
| 对侧 25° ERB | 1.383174 | **1.364204** | **1.372%** | 41/44 |
| 对侧高频 | 3.637638 | **3.608567** | **0.799%** | 40/44 |
| 水平面严格 ILD | 0.625015 | **0.596141** | **4.620%** | 32/44 |

相对 MCA，epoch 39 的四项改善分别为 `20.338% / 22.643% / 24.017% /
28.177%`。因此延长预算在当前 validation 上带来了四项一致改善，最大收益集中在
严格 ILD。

## 文件说明与结论边界

- `summary.json`：完整配置、聚合值和读取安全信息；
- `aggregate_metrics.csv`：MCA、MLP v1/v2、v3 与本候选的聚合指标；
- `per_subject_metrics.csv`：44 人逐被试指标；
- `metric_long.csv`：方法 × 指标的长表；
- `quality_checks.csv`：逐被试重建质量检查；
- `comparison_vs_v32_locked.csv`：本候选与 v3.2 locked epoch 6 的直接比较；
- `training_history.csv` 与 `training_report.json`：40 epoch 训练轨迹与最佳点摘要；
- `figures/`：44 人指标图与对侧 HRTF 总览。

该结果证明 v3.2 仍有训练预算优化空间，可作为后续新数据协议的优先候选；但 SONICOM
test 已在此前使用，因此不能用本 validation 结果替换既有论文主模型或重新报告最终
test 性能。若要形成新的论文主张，需要新的未见拆分或外部数据做一次性确认。
