# SIREN Stage A3 — backbone 深化搜索结果（depth / width / hidden-omega）

## 概述

按 [`STAGE_A3_PROTOCOL.md`](../experiments/film_siren/STAGE_A3_PROTOCOL.md) 在
A2 选出的 **D-o20**（dual × first-omega 20）上逐维深化搜索
`depth → width → hidden-omega`，复用 A2 的 5 名固定 train 被试与 64 方向
holdout。最终 backbone 候选：

**`D-o20-d6-w256-ho20`**（dual、first-omega 20、6 层 sine、width 256、
hidden-omega 20），预算 200 epochs，aggregate holdout RMSE **`2.8956 dB`**。

## 预算平台验证（A3-0）

- run `sonicom_siren_a3_budget_e200`（D-o20 基线 200 epochs × 5 被试）；
- best epochs `200/189/150/128/133`，中位数 `M = 150`；
- 200-epoch aggregate holdout RMSE `2.9350` 相对 A2（100 epoch）`2.9865`
  改善 **1.726%**（≥ 0.5% 阈值）→ 按协议 **`E_A3 = 200`**；
- ⚠️ P0289 的 best epoch 为 200（预算末端，单被试 RETEST）——该被试
  200 epoch 仍未平台，confirmation 阶段建议更长预算复核。

## 三阶段搜索结果（holdout RMSE，越低越好）

| 阶段 | 候选 | aggregate holdout RMSE | 相对最优 | 决策 |
|---|---|---:|---:|---|
| A3-1 depth | **d6** | **2.9350** | — | d6 主线（紧咬：d4 +0.42% 保留对照；d8 +2.60% 淘汰） |
| | d4 | 2.9472 | +0.42% | |
| | d8 | 3.0112 | +2.60% | |
| A3-2 width | **w256** | **2.9350** | — | w256 主线（w128 +4.27% 对照；w512 +15.04% 淘汰） |
| | w128 | 3.0603 | +4.27% | |
| | w512 | 3.3763 | +15.04% | |
| A3-3 hidden-omega | **ho20** | **2.8956** | — | ho20 主线（ho30 +1.36% 对照；ho50 +16.22% 淘汰） |
| | ho30 | 2.9350 | +1.36% | |
| | ho50 | 3.3653 | +16.22% | |

每阶段完整 per-subject 明细见
`results/sonicom_siren_a3_matrix/<stage>/summary.csv` 与 `top.json`；
逐 run 完整记录位于 `artifacts/training/sonicom_siren_a3_*/<PXXXX>/`
（45 个 run 全部 KEEP 或预算边缘 RETEST、无 NaN/Inf、`test_subjects_read` 均为 0）。

## 对照链

| 配置 | 预算 | aggregate holdout RMSE |
|---|---:|---:|
| A2 粗筛 D-o20 | 100 ep | 2.9865 |
| A3 预算验证 D-o20 基线（d6-w256-ho30） | 200 ep | 2.9350 |
| **A3 最终 D-o20-d6-w256-ho20** | 200 ep | **2.8956** |

- 200 epoch 相对 100 epoch 改善 `1.726%`；最终 ho20 相对 A2 粗筛改善
  **`3.045%`**、相对 200-epoch 基线再改善 `1.343%`；
- A2 的对照配置 D-o30 未在 A3 复跑（按协议仅保留为 A2 对照记录）。

## 观察

- **更低 omega 一致更优**：A2 中 first-omega 20 优于 30/50；A3 中
  hidden-omega 20 同样优于 30/50——当前 `[-1,1]` 坐标下低频率系数
  （更平滑的激活）更利于方向外推；
- **width 256 是最优点**：w128 欠容量（+4.3%）、w512 明显过拟合/难优化
  （+15%，full RMSE 也恶化到 2.63）；
- depth 4 与 6 几乎等价（+0.42%），d8 略差——6 层保留为安全基线；
- 最终候选的 full-field RMSE `1.7160 dB` 也优于基线 `1.9004`（低 omega
  同时改善拟合与外推）。

## 审查提示（同 A2 口径，A3 前已声明）

- best-epoch 选择与排名共用同一 64 方向 holdout（选择集语义）；
- 全 45 run 单 seed（20260819）；结论需 confirmation（16–32 被试）与
  Stage C 3-seed 复核；
- P0289 在 200 epoch 仍未平台（预算边缘），confirmation 阶段建议
  200+ epoch 或更长预算复核。

## 下一步

1. **confirmation**：新增 16–32 名 train 被试锁定清单（沿用
   `Avg RMS dB` 分层），用 `D-o20-d6-w256-ho20` 复核；
2. 通过后冻结 **`SIREN Backbone v1`**；
3. 之后进入 Stage B（Q26 condition / FiLM conditioning）。

## 审查记录（2026-08-20）

对 A3 全流程的独立核对（审查脚本 `scripts/_audit_a3_*.py`，审查后已删除）。

**通过项**

- 10 个配置（budget 1 + depth 3 + width 3 + ho 3）的
  `mode(dual) / first_omega(20) / depth / width / hidden_omega / seed /
  holdout_sha256 / epochs(200) / steps(100) / optimizer / AMP(off) /
  test_subjects_read / subjects_count` 与冻结协议全部一致；
- 预算规则算术复核：best epochs `128/133/150/189/200` → 中位数 M=150 →
  `E_A3 = max(100, ⌈150×1.25/50⌉×50) = 200`；200-epoch 改善 `1.7258%`
  ≥ 0.5%，不触发 100-epoch 豁免——规则执行正确；
- 三阶段 45 个 per-subject report 与 `summary.csv` 完全一致
  （per-subject RMSE、aggregate 均值、hidden_width/depth/omega 维度）；
- 报告数字与 summary 一致；三阶段 Top 决策的 gap12/gap13 算术、
  紧咬/清晰分支标志与标签全部正确（depth 紧咬、width/ho 非紧咬）；
- artifacts 完整：50 ×（best.pt / last.pt / history.csv / training_report.json）
  + 10 ×（matrix_summary.json / matrix_configuration.json），history 全部
  200 epochs；
- 排名稳健性：每候选 5 被试 std 0.16–0.18 dB、range 0.41–0.52 dB，
  与 A2 同量级，排名不被单被试主导；
- 50 run：46 KEEP + 4 RETEST（4 个均为 P0289 在 200 epoch 预算末端，
  best=200），无 NaN/Inf，`test_subjects_read` 均为 0。

**发现的问题（均低风险，不影响 A3 结论）**

1. 报告"对照链"中的改善百分比（3.045% / 1.343%）用 4 位小数中间值计算，
   精确复核为 3.044% / 1.341%（≤0.002 个百分点四舍五入差）；
   数值结论不变，建议后续报告注明计算精度；
2. 紧咬分支的"成本控制执行说明"（`d0b4f2f`）是在 depth 阶段完成后、
   width 阶段开始前补充进协议的——虽透明记录且未影响已跑结果，
   但严格预注册应在首次运行前写入；confirmation 阶段协议如有类似
   补充应提前冻结；
3. A2 审查发现的代码问题（`GradScaler` 跨 subject 复用、checkpoint
   `field_metrics` 命名、per-subject 无 `configuration.json`）对 A3
   同样适用（A3 复用同一 `multi_subject_siren_matrix` 入口，无新增
   训练代码），建议在 confirmation 前一并修正。

**结论**：A3 结果可信、代码与协议一致，`D-o20-d6-w256-ho20`（holdout RMSE
2.8956 dB）作为 backbone 候选进入 confirmation。
