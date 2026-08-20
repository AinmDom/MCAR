# SIREN Stage A4 — confirmation 结果（32 名未见 train 被试）

> 状态：PENDING RESULTS。128 run（4 套 × 32 被试）训练完成后由
> `scripts/analyze_siren_a4.py` 生成 M/L/D/A 聚合与预注册标准判定并回填。

## 概述

按 [`STAGE_A4_CONFIRMATION_PROTOCOL.md`](../experiments/film_siren/STAGE_A4_CONFIRMATION_PROTOCOL.md)
用 32 名未参与 A2/A3 配置选择的 train 被试，验证 backbone 配置结论
（频率映射、first-omega、depth）在新被试上是否成立。

## 固定设置

- 被试：32 名（`configs/data/siren_a4_confirmation_subjects_v1.csv`，
  排除 A2 的 5 人，`Avg RMS dB` 8 层 × 4 人，seed 20260820）；
- holdout：A2 的 64 方向锁定（SHA-256 强制校验）；
- 预算：250 epochs × 100 steps；单 seed 20260820；Adam 1e-4、FP32；
- 配置集（4 套）：

| 编号 | 配置 | 验证的结论 |
|---|---|---|
| C1 (M) | D-o20-d6-w256-ho20（主候选） | 绝对水平可复现 |
| C2 (L) | linear-o30-d6-w256-ho30（历史基线） | dual 优于 linear |
| C3 (D) | D-o30-d6-w256-ho30（A2 对照） | first-omega 20 优于 30 |
| C4 (A) | D-o20-d4-w256-ho20（紧咬对照） | depth d6 ≈ d4 |

## 预注册判定标准（STAGE_A4 第 5 节）

- 主标准：`M <= L`；次标准：`M <= D`；绝对合理性：`2.4 <= M <= 3.4`；
- 信息性：`|M − A|` 相对差距 + 32 人配对 t-test。

## 结果

**32 名未见被试的 aggregate holdout RMSE（dB，越低越好）**：

| 编号 | 配置 | holdout RMSE | holdout MAE | full RMSE |
|---|---|---:|---:|---:|
| M | D-o20-d6-w256-ho20（主候选） | **2.8141** | 1.7273 | 1.7467 |
| L | linear-o30-d6-w256-ho30（历史基线） | 3.1733 | 2.0618 | 2.0313 |
| D | D-o30-d6-w256-ho30（A2 对照） | 2.9627 | 1.8510 | 1.8372 |
| A | D-o20-d4-w256-ho20（紧咬对照） | 2.8545 | 1.7571 | 1.7584 |

（holdout MAE / full RMSE 数值取自 `artifacts/siren_a4_confirmation/summary.csv`；
完整 32 人逐被试明细见 `results/sonicom_siren_a4_confirmation/summary.csv`。）

### 预注册标准判定（全部通过）

| 标准 | 条件 | 结果 |
|---|---|---|
| 主标准 | M ≤ L | `2.8141 ≤ 3.1733` ✅（M 优于 L 0.359 dB，32 人配对 t-test p≈0） |
| 次标准 | M ≤ D | `2.8141 ≤ 2.9627` ✅（M 优于 D 0.149 dB，p≈0） |
| 绝对合理性 | 2.4 ≤ M ≤ 3.4 | `2.8141` ✅ |
| 信息性 | \|M − A\| 相对差距 | `1.436%`（d6 仍略优于 d4，紧咬结论保持） |

判定 JSON：`results/sonicom_siren_a4_confirmation/decision.json`。

### 结论

**CONFIRMATION 通过。** A2/A3 的配置结论在 32 名从未参与配置选择的 train
被试上全部保持：

- **dual 优于 linear**：确认集优势 `+11.33%`（M 2.8141 vs L 3.1733），
  与 A2 的 `+8.6%` 方向一致且更明显；
- **first-omega 20 优于 30**：确认集优势 `+5.29%`（M vs D），A2 为 `+2.9%`；
- **d6 ≈ d4 紧咬保持**：确认集差距 `1.44%`，与 A3 的 `0.42%` 同量级；
- 选择偏差未导致任何结论反转；M 的绝对水平（2.81 dB）优于 A3 五人值
  （2.90 dB），落在合理性区间内。

**冻结 `SIREN Backbone v1 = D-o20-d6-w256-ho20`**
（dual、first-omega 20、depth 6、width 256、hidden-omega 20，250 epochs）。

完整性：128/128 run 完成、全部 `KEEP`、无 NaN/Inf、`test_subjects_read`
均为 0；代码修正（scaler / checkpoint 命名 / 人数放宽 / per-subject
configuration.json）随本阶段生效并经 smoke 验证。

## 代码修正记录（随 A4 生效）

- `GradScaler` 移入 subject 循环（避免 AMP 状态跨被试泄漏）；
- matrix checkpoint 字段 `field_metrics` → `holdout_metrics`；
- `run_matrix` 5 人硬检查放宽（subject 数由锁定 csv 决定）；
- 新增 per-subject `configuration.json`（含完整上下文与 hash）。

## 下一步

- 通过 → 冻结 **`SIREN Backbone v1 = D-o20-d6-w256-ho20`**；
- 之后进入 Stage B（Q26 condition / FiLM conditioning）。
