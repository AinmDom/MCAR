# SIREN Stage A2 — 五人 backbone matrix 联合粗筛结果

## 概述

按 [`STAGE_A2_PROTOCOL.md`](../experiments/film_siren/STAGE_A2_PROTOCOL.md)
执行 plain SIREN backbone 的 `frequency × first-omega` 联合粗筛：
`{linear, erb, dual} × {20, 30, 50}` = 9 配置 × 5 名固定 train 被试 = 45 次
独立训练，以 64 方向固定 direction-holdout 的 residual RMSE 作为主指标。

结果：**第一名 `dual × first-omega 20`（D-o20）**，aggregate holdout RMSE
`2.9865 dB`；按冻结 Top-2 规则，推进 **D-o20 为主配置、D-o30 为对照**
进入 Stage A3。

## 固定设置

- 被试：`P0289 / P0346 / P0085 / P0076 / P0010`
  （`Avg RMS dB (Free Field)` 五层分层，seed `20260819`，
  `configs/data/siren_a2_subjects_v1.csv`）；
- holdout：64 个纯插值方向（solid-angle 分层，seed `20260819`，
  `configs/data/siren_a2_holdout_v1.csv`，SHA-256
  `C1EEE29A6B33217D9267E76851D7F0E0C5D98FCBE245AABD9E4FA7C68C83A057`，
  加载时强制校验）；
- 训练方向：793 − 64 = 729；预算 100 epochs × 100 steps、16 dirs/batch、
  Adam `1e-4`、weight decay 0、FP32、width 256、6 sine layers、hidden omega 30；
- 主指标：holdout residual RMSE（等权，与 A1 同口径）；`test_subjects_read` 恒为 0。

## 结果：9 配置排名（holdout RMSE，越低越好）

| rank | config | aggregate holdout RMSE | aggregate holdout MAE | aggregate full RMSE |
|---:|---|---:|---:|---:|
| 1 | **dual-o20** | **2.9865** | 1.8304 | 2.0163 |
| 2 | dual-o30 | 3.0732 | 1.9268 | 1.8680 |
| 3 | linear-o20 | 3.0863 | 1.9324 | 2.0237 |
| 4 | linear-o30 | 3.2427 | 2.0833 | 2.1135 |
| 5 | dual-o50 | 3.3364 | 2.1541 | 2.1366 |
| 6 | erb-o20 | 3.4246 | 2.1844 | 2.7132 |
| 7 | linear-o50 | 3.5607 | 2.3368 | 2.5160 |
| 8 | erb-o30 | 3.6512 | 2.3775 | 2.5903 |
| 9 | erb-o50 | 3.9772 | 2.6323 | 3.0835 |

每配置 5 名被试的 holdout RMSE（dB）见
`results/sonicom_siren_a2_matrix/summary.csv`；逐 run 完整记录
（history、checkpoint SHA-256、decision）位于
`artifacts/training/sonicom_siren_a2_*_w256_d6_o*/<PXXXX>/`。
9 个配置全部 5/5 被试完成、全部 `KEEP`（best epoch 均早于预算末端），
无 NaN/Inf，`test_subjects_read` 均为 0。

## Top-2 决策（STAGE_A2_PROTOCOL.md 第 6 节）

- 第一名 D-o20 领先第二名 D-o30 `2.90%`（≥ 0.5%），不满足"紧咬"条件；
  第一名领先第三名 `3.34%`（> 2%）；
- 结论：**推进 D-o20 作为主配置进入 Stage A3，D-o30 作为对照**；
- 决策 JSON：`results/sonicom_siren_a2_matrix/top2.json`。

## 观察

- **dual 频率映射整体最优**：dual 三档全部进入前五（D-o20/30/50），
  linear 居中，ERB 最差（E-o50 垫底、前三档全部后三位）；
- **first-omega 20 在 dual 和 linear 下都一致优于 30、50**，提示当前
  `[-1,1]` 坐标下更低的 first-omega 更利于方向外推；
- full-field RMSE 排序与 holdout 不完全一致（如 D-o30 的 full RMSE
  `1.8680` 优于 D-o20 的 `2.0163`），但按协议以 holdout（训练未见方向）
  为选择依据；
- A1 的 P0002 全方向拟合 full-field RMSE 平台约 `1.50 dB`；A2 的 holdout
  是训练未见方向，误差高于拟合误差属预期，A2 用于配置相对排序。

## 下一步

- Top-2（D-o20 主、D-o30 对照）进入 Stage A3：依次搜索
  depth / width / hidden-omega（草案见
  [`STAGE_A3_PROTOCOL_DRAFT.md`](../experiments/film_siren/STAGE_A3_PROTOCOL_DRAFT.md)）；
- 随后固定 16–32 名 confirmation 被试冻结 `SIREN Backbone v1`。
