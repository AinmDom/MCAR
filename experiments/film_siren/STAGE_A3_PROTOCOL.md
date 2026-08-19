# Stage A3 预注册协议：backbone 深化搜索（depth / width / hidden-omega）

本文是 Stage A3 的预注册规范，在**第一次运行前**冻结搜索空间、预算决策规则、
逐维搜索顺序与 Top 决策规则。A2 的 5 名被试与 64 方向 holdout 锁定沿用不变。

## 1. 目的与范围

- 在 Stage A2 选出的 **D-o20（dual 频率映射 × first-omega 20）**主配置上，
  依次搜索 `depth（sine layer count）`、`width（hidden width）`、
  `hidden omega`，得到深化后的 backbone 候选；
- **D-o30 作为对照**：若某维 D-o30 上下文显著反超，按第 8 节规则处理；
- A3 仍只读取 train 被试；validation/test HDF5 一律不读取；
- 结果用于后续 confirmation（16–32 被试）与冻结 `SIREN Backbone v1`。

## 2. 固定上下文（冻结）

- frequency mapping：`dual`（linear + erb 双通道坐标）；
- first omega：`20`；
- 被试：A2 锁定 `P0289 / P0346 / P0085 / P0076 / P0010`
  （`configs/data/siren_a2_subjects_v1.csv`）；
- holdout：A2 锁定 64 方向（`configs/data/siren_a2_holdout_v1.csv`，
  SHA-256 强制校验）；
- 训练方向：793 − 64 = 729；优化器 Adam `1e-4`、weight decay 0、
  gradient clip 5.0、FP32、16 dirs/batch、32 dirs/eval block；
- 主指标：holdout residual RMSE（等权，与 A2 同口径）。

## 3. A3-0：预算平台验证（冻结决策规则）

- 在 D-o20 基线（width 256、depth 6、hidden omega 30）上运行
  **200 epochs × 100 steps × 5 被试**；
- 记录 5 名被试 best-epoch（holdout RMSE 最优）的中位数 `M`；
- **A3 训练预算**：`E_A3 = max(100, ceil(M × 1.25 / 50) × 50)`；
  - 若 `M > 175`（200 epoch 仍未见平台），`E_A3 = 200`，并在报告中注明
    "预算边缘，confirmation 阶段需更长预算复核"；
  - 若 200-epoch aggregate holdout RMSE 相对 A2（100 epoch）改善 < 0.5%，
    允许 `E_A3 = 100`（平台已到，长预算无收益）；
- 预算验证 run 也纳入 A3 报告（作为 D-o20 基线在更长预算下的表现）。

**冻结结果（2026-08-19）**：验证 run `sonicom_siren_a3_budget_e200` 的
5 名被试 best epoch 为 `200/189/150/128/133`，中位数 `M = 150`；
200-epoch aggregate holdout RMSE `2.9350 dB` 相对 A2 100-epoch
`2.9865 dB` 改善 `1.726%`（≥ 0.5%，不触发 100-epoch 豁免）。
按规则 **`E_A3 = 200`**。P0289 的 best epoch 为 200（预算末端，
单被试 RETEST），说明该被试 200 epoch 仍未平台，A3 报告需注明
"预算边缘，confirmation 阶段建议更长预算复核"。

## 4. 搜索空间（冻结）

| 阶段 | 维度 | 候选值 | 上下文 |
|---|---|---|---|
| A3-1 | depth（sine layer count） | `{4, 6, 8}` | width 256、hidden omega 30 |
| A3-2 | width（hidden width） | `{128, 256, 512}` | depth 用 A3-1 最优 |
| A3-3 | hidden omega | `{20, 30, 50}` | depth/width 用前两阶段最优 |

- 每阶段只改变一个维度，其余沿用前阶段最优值（D-o20 上下文固定）；
- 每候选 = 5 名被试各独立训练一个 plain SIREN（同一预算 `E_A3`），
  再按第 7 节聚合；
- 阶段间串行：A3-2 依赖 A3-1 结果，A3-3 依赖 A3-1/2 结果。

## 5. 配置与产物命名

- 预算验证：run 名 `sonicom_siren_a3_budget_e200`；
- A3-1：`sonicom_siren_a3_d{depth}`（如 `..._d4`）；
- A3-2：`sonicom_siren_a3_d{bestd}_w{width}`；
- A3-3：`sonicom_siren_a3_d{bestd}_w{bestw}_ho{ho}`；
- 每 run 产物结构与 A2 相同
  （`artifacts/training/<run_name>/<PXXXX>/` + `matrix_summary.json`）；
- 配置 JSON 由 `scripts/generate_siren_a3_configs.py` 确定性生成并提交。

## 6. 运行顺序

1. A3-0 预算验证（200 epoch × 5 被试）；
2. 按第 3 节规则冻结 `E_A3`，更新本协议；
3. A3-1 depth 搜索（3 候选 × 5 被试，预算 `E_A3`）→ 聚合 → Top 决策；
4. A3-2 width 搜索（3 候选 × 5 被试）→ 聚合 → Top 决策；
5. A3-3 hidden-omega 搜索（3 候选 × 5 被试）→ 聚合 → Top 决策；
6. 汇总：最终 backbone 候选 + 对照记录。

## 7. 聚合与 Top 决策规则（与 A2 一致，冻结）

- 每阶段每候选：5 被试 holdout RMSE 的等权均值（同时报告中位数与极差）；
- 排名按聚合 holdout RMSE 升序；
- **紧咬判定**：第一名与第二名差距 < 0.5% 且与第三名差距 > 2% →
  推进前两名进入下一阶段；否则推进第一名为主、第二名作对照；
- 每阶段聚合写入 `artifacts/siren_a3_matrix/<stage>/`，
  精选结果复制到 `results/sonicom_siren_a3_matrix/`。

**执行说明（2026-08-19，紧咬分支的成本控制）**：当某阶段触发"推进前两名"
时，后续维度深化以第一名为主线做完整候选搜索，第二名保留为候选对照、
不展开全候选搜索；若第一名在某维度结果明显恶化（聚合 holdout RMSE 劣于
第二名在上一阶段的记录），回到第二名复核。该说明是对本节"推进前两名进入
下一阶段"的执行解释，用于避免 2× 实验成本。

## 8. 数据访问约束

- 只允许读取 `data/processed/sonicom_residual_q26_v1/subjects/<train>/q26.h5`；
- split 校验沿用 `verify_train_subject`；`test_subjects_read` 恒为 0。

## 9. 产出与保存

- 每 run 完整记录（configuration、history、checkpoint SHA-256、decision）；
- 阶段聚合、最终报告 `reports/film_siren_siren_a3_matrix.md`、
  实验日志更新，全部提交；
- A3 结论供 confirmation（新被试锁定清单 + `run_matrix` 5 人硬检查放宽）
  与 `SIREN Backbone v1` 冻结使用。
