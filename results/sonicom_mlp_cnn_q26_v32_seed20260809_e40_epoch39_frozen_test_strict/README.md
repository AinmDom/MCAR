# SONICOM Q26 v3.2 seed 20260809 epoch 39：冻结 test 重建

本目录记录 validation 已选定的 v3.2 seed `20260809`、epoch 39 checkpoint 在 44 名
SONICOM test 被试上的一次冻结评价。运行前已锁定 checkpoint、四项指标、对照模型和
结果保留规则；test 没有参与 seed、epoch 或 checkpoint 选择，评价后禁止据此继续调参。

## Checkpoint 完整性

- checkpoint：`artifacts/training/sonicom_mlp_cnn_q26_v32_seed20260809_e40/best.pt`；
- 内部 epoch：39；参数量：174,627；文件大小：1,367,029 bytes；
- test 前 SHA-256：`1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`；
- Python 推理后与 MATLAB 重建后 SHA-256 均与上述值完全一致；
- 没有调用训练器、优化器或权重保存逻辑，模型参数字节级未改动。

## 44 人严格 test 结果

误差越低越好。原 locked v3.2 epoch 6 与本冻结 epoch 39 使用相同 test 被试、方向、
MCA phase、ERB、高频和 HRIR-energy ILD 口径。

| 指标 | locked epoch 6 (dB) | frozen epoch 39 (dB) | 相对改善 | 改善被试数 |
|---|---:|---:|---:|---:|
| 全空间 ERB | 0.867805 | **0.855676** | **1.398%** | 42/44 |
| 对侧 25° ERB | 1.365327 | **1.349975** | **1.124%** | 36/44 |
| 对侧高频 | 3.611854 | **3.590454** | **0.592%** | 40/44 |
| 水平面严格 ILD | 0.686999 | **0.660297** | **3.887%** | 29/44 |

相对 MCA，epoch 39 四项分别改善 `20.932% / 22.671% / 23.585% / 20.387%`。
四项总体均值均优于 locked epoch 6，最大收益仍集中在严格 ILD。

## 执行记录与文件

- GPU test 推理：44/44 被试，输出均为 `[2,793,463]`，耗时 `26.573 s`；
- MATLAB 严格重建：44/44 被试，耗时约 `1032.1 s`；
- `summary.json`：完整聚合指标、配置和 test 读取记录；
- `aggregate_metrics.csv`：各方法的聚合均值与标准差；
- `per_subject_metrics.csv`：44 人逐被试结果；
- `metric_long.csv`：方法 × 指标长表；
- `quality_checks.csv`：逐被试重建质量检查；
- `comparison_vs_locked_epoch6.csv`：冻结 epoch 39 与 locked epoch 6 的直接比较；
- `figures/`：44 人指标曲线、聚合图和对侧 HRTF 总览。

这是已冻结候选在既有 held-out test 上的追加、一次性比较，不是重新用 test 选模。
结果无论好坏均已保留；后续不得根据本次 test 结果修改模型后再次选择版本。
