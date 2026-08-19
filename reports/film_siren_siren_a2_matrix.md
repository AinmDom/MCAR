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

## 审查记录（2026-08-19）

对 A2 全流程的独立核对（审查脚本 `scripts/_audit_a2_*.py`，审查后已删除）。

**通过项**

- 锁定文件可复现：重跑 `prepare_siren_a2_matrix` 生成的被试、64 个 holdout
  indices 与 SHA-256 与提交文件逐项一致；
- 9 个配置的 `mode / first_omega / seed / holdout_sha256 / test_subjects_read
  / subjects_count` 与冻结协议全部一致；
- `summary.csv` 与 45 个 `training_report.json` 的 per-subject holdout RMSE
  及 aggregate 均值完全一致（误差 < 1e-6 dB）；报告表格数字与 summary 一致；
- Top-2 决策算术复核：gap12 = 2.9025%、gap13 = 3.3387%，与 `top2.json`
  一致，规则分支正确（非"紧咬"分支，D-o20 为主、D-o30 为对照）；
- artifacts 完整：45 ×（best.pt / last.pt / history.csv / training_report.json）
  + 9 ×（matrix_summary.json / matrix_configuration.json），history 全部
  100 epochs；45 run 全部 KEEP、无 NaN/Inf、`test_subjects_read` 均为 0；
  参数数与 dual（331010）/ 单维（330754）预期一致；
- 排名稳健性：每配置 5 被试 std 0.12–0.18 dB，median 与 mean 排序一致，
  排名不被单被试主导（同被试跨配置的系统性差距大于被试内噪声）。

**发现的问题（均低风险，不影响 A2 结论）**

1. per-subject 目录没有 `configuration.json`：matrix 模式的信息由根目录
   `matrix_configuration.json` + per-subject `training_report.json` 完整承载，
   但与单 subject 模式的目录布局不一致（文档提示，非功能缺陷）；
2. `GradScaler` 在 subject 循环外创建、5 名被试复用：当前 FP32
   （`use_amp=False`）下为 no-op，无影响；若未来启用 AMP，应移入循环内
   以避免跨被试状态残留；
3. matrix checkpoint 的 `field_metrics` 字段实际保存 holdout 指标，
   命名不精确（建议改为 `holdout_metrics`）；
4. 每 epoch 全量重写 `history.csv`（性能微优化空间）。

**实验设计提示（A3 前请知悉）**

- best-epoch 选择与配置排名共用同一 64 方向 holdout（选择集语义，符合
  Stage A 协议），且 best epoch 普遍接近预算末端（73–98/100），说明 100
  epoch 预算下 holdout 误差未完全平台：A3 深化 D-o20 时建议评估更长预算，
  并依赖 A3-4 confirmation 被试做更独立的确认；
- 全 45 run 为单 seed（20260819）粗筛：配置间公平，但尚无 seed 稳健性
  证据；Stage C 的 3-seed 规则需覆盖该结论；
- ERB 整体最差、dual 最优的结论基于 5 被试 × 单 seed × 64 方向 holdout，
  样本有限，confirmation 阶段需复核频率映射结论。

**结论**：A2 结果可信、代码与协议一致，D-o20 作为主配置进入 Stage A3。
