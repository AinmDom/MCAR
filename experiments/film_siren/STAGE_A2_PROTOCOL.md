# Stage A2 预注册协议：五人 backbone matrix 联合粗筛

本文是 Stage A2 的预注册规范，按 `EXPERIMENT_CHECKLIST.md` 第 1、6 节要求，
在**第一次运行前**冻结搜索空间、最大实验数、预算、数据访问与 Top-K 规则。
本文件与生成的锁定 CSV 一起提交；运行期间不允许修改选择规则。

## 1. 目的与范围

- 在 5 名固定 SONICOM train 被试上，对 plain SIREN backbone 做
  `frequency × first-omega` 联合粗筛，为后续 depth/width/hidden-omega 搜索
  和 `SIREN Backbone v1` 冻结提供选择依据。
- Stage A2 是 backbone 选择实验，只读取 train 被试 HDF5；
  validation/test HDF5 一律不读取。
- A2 的结果不代表跨被试或方向泛化结论，只用于配置相对排序。

## 2. 搜索空间（冻结）

| 维度 | 取值 | 说明 |
|---|---|---|
| frequency mapping | `linear` / `erb` / `dual` | `train_siren.frequency_coordinates` 三种模式，公式已冻结并有单测 |
| first omega | `20` / `30` / `50` | 第一层 `sin(omega*(Wx+b))` 的 omega |

共 **9 个配置**。固定项（不搜索）：

- hidden width `256`、sine layer count `6`、hidden omega `30`
  （沿用 A1 v2 的 backbone，避免 A2 一次改变过多维度）；
- 输入坐标 `[x,y,z,f]`，dual 时 `f` 为 2 维（linear + erb）；
- 输出双耳 normalized residual；`r_db = r_norm * target_std + target_mean`；
- optimizer Adam、LR `1e-4`、weight decay `0`、gradient clip `5.0`、FP32
  （与 A1 v2 完全一致）。

## 3. 训练预算（冻结，9 配置相同）

- `100 epochs × 100 steps`、`16 directions/batch`、`32 directions/eval block`；
- 每 epoch 评估一次 holdout 方向；训练结束评估全部 793 方向用于审计；
- 最大实验数：`9 配置 × 5 被试 = 45 run`（不含 smoke）。

## 4. 被试锁定（冻结）

- 分层因子：metadata `Avg RMS dB (Free Field)`（259/262 train 被试有值）；
- 按该值分成 5 个 quintile 层，每层内用固定 seed `20260819` 抽 1 人；
- 输出 `configs/data/siren_a2_subjects_v1.csv`，含层号、分层值、hash；
- 禁止依据已有模型误差临时换人。

## 5. direction holdout（冻结）

- 从 767 个纯插值方向（`is_interpolation_evaluation=1`）中，
  按 `solid_angle_weight` 分层抽 **64 个 holdout 方向**（固定 seed `20260819`）；
- 每被试训练方向 = 全部 793 − holdout 64 = 729；
- holdout 方向对全部 9 配置、5 被试相同，保证可比；
- 输出 `configs/data/siren_a2_holdout_v1.csv`（indices + sha256）。

## 6. 指标与选择规则（冻结）

- 主指标：holdout 方向 residual **RMSE（dB，等权，与 A1 同口径）**；
- 辅助指标：holdout 方向 MAE、>8 kHz MAE、>10 kHz MAE、
  一阶/二阶谱差分 MAE、notch-depth MAE（复用 A1 的 `FieldMetrics`）；
- 每配置聚合：5 被试 holdout RMSE 的等权均值（同时报告 5 个单项与
  中位数、极差，检查是否有被试主导）；
- **Top-K 规则**：按聚合 holdout RMSE 升序选 **Top 2** 配置；
- 若第一名与第二名差距 < 0.5% 且与第三名差距 > 2%，则 Top 2 为
  (一, 二)；否则只推进第一名进入后续 depth/width/hidden-omega 搜索时
  再附带第二名作对照，最终由后续搜索决定。

## 7. 运行与保存

- 每 run 输出 `artifacts/training/<run_name>/`，含
  `configuration.json`（含 git state、config/config hash、normalization hash）、
  `history.csv`、`best.pt`、`last.pt`、`training_report.json`（含
  `test_subjects_read: 0`、checkpoint SHA-256、decision）；
- run 名规范：`sonicom_siren_a2_<freq>_w256_d6_o<omega>_<PXXXX>`；
- 聚合脚本输出 `artifacts/siren_a2_matrix/summary.csv` 与 Top-2 决策 JSON；
- 完成后更新 `docs/EXPERIMENT_LOG.md` 并提交本协议、锁定 CSV 与聚合结果。

## 8. 数据访问约束

- 只允许读取 `data/processed/sonicom_residual_q26_v1/subjects/<train>/q26.h5`；
- split 校验沿用 `verify_train_subject`（非 train 直接拒绝）；
- `test_subjects_read` 恒为 0。
