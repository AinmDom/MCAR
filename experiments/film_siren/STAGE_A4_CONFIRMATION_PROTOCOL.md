# Stage A4 预注册协议：backbone confirmation（32 名新 train 被试）

本文在**第一次运行前**冻结 confirmation 的配置集、被试锁定、预算、seed
与通过/不通过标准。confirmation 只做确认、不重新搜索（checklist 第 6 节）。

## 1. 目的与范围

- 用 32 名**未参与 A2/A3 任何配置选择**的 train 被试，验证
  A2/A3 选出的 backbone 配置结论在新被试上是否成立；
- 检验对象：频率映射（dual vs linear）、first-omega（20 vs 30）、
  depth（d6 vs d4 紧咬）三个 A2/A3 核心结论；
- confirmation 不是 test、不是论文级泛化证据；
  只读 train 被试 HDF5，`test_subjects_read` 恒为 0。

## 2. 配置集（冻结，4 套）

| 编号 | 配置 | 验证的结论 |
|---|---|---|
| C1（主候选） | `D-o20-d6-w256-ho20` | 主候选绝对水平可复现 |
| C2（历史基线） | `linear-o30-d6-w256-ho30`（= A1/A2 的 L-o30） | dual 优于 linear |
| C3（A2 对照） | `D-o30-d6-w256-ho30` | first-omega 20 优于 30 |
| C4（紧咬对照） | `D-o20-d4-w256-ho20` | depth d6 ≈ d4 |

- 每套 = 32 名被试各独立训练一个 plain SIREN；
- 同一训练协议：Adam `1e-4`、weight decay 0、gradient clip 5.0、FP32、
  16 dirs/batch、32 dirs/eval block、同一 64 方向 holdout；
- 同一随机 seed `20260820`（单 seed 确认，与 A2/A3 同口径）。

## 3. 被试锁定（冻结）

- 候选池：262 train 被试中**排除** A2/A3 的 5 名
  （P0289/P0346/P0085/P0076/P0010）；
- 分层：按 `Avg RMS dB (Free Field)` 排序分 **8 层**，每层固定 seed
  `20260820` 抽 4 人，共 32 人；
- 输出 `configs/data/siren_a4_confirmation_subjects_v1.csv`（含层号、
  分层值、hash）；禁止依据误差临时换人；
- holdout 沿用 A2 的 64 方向锁定
  （`configs/data/siren_a2_holdout_v1.csv`，SHA-256 强制校验）。

## 4. 预算与 seed（冻结）

- **250 epochs × 100 steps**（A3 预算验证显示 P0289 在 200 epoch 仍未
  平台；250 为折中，确认集上若 best-epoch 中位数接近 250 则在报告中
  注明需更长预算）；
- 单 seed `20260820`；若主标准处于边界（见第 5 节），追加 3-seed
  复核（协议另行冻结 seed 清单）。

## 5. 通过/不通过标准（冻结，运行前生效）

记确认集 32 人 aggregate holdout RMSE：
`M`（主候选）、`L`（C2 历史基线）、`D`（C3）、`A`（C4 d4 紧咬）。

- **主标准**：`M <= L`（主候选不劣于历史基线，A2 中 D-o20 2.9865 <
  L-o30 3.2427，优势 +8.6%，确认集上要求方向保持）；
- **次标准**：`M <= D`（first-omega 20 优势保持，A2 中 +2.9%）；
- **绝对合理性**：`2.4 <= M <= 3.4`（与 A2/A3 的 2.9–3.0 dB 量级一致，
  排除确认集恰好异常的极端情形）；
- **信息性报告**（不作为通过标准）：`|M - A|` 相对差距，复核 d6≈d4
  紧咬结论；`M/L/D/A` 的 32 人配对统计（差值均值、p 值）仅作报告。

**判定**：主标准 + 次标准 + 绝对合理性全部满足 → **通过**，
冻结 `SIREN Backbone v1 = D-o20-d6-w256-ho20`；
否则 → **不通过**，按第 7 节诊断并报告，不得静默改用确认集调参。

## 6. 运行与保存

- run 名：`sonicom_siren_a4_c{1,2,3,4}`；
- 每 run 产物结构与 A3 相同（`artifacts/training/<run>/<PXXXX>/` +
  `matrix_summary.json`）；
- 聚合写入 `artifacts/siren_a4_confirmation/`，精选复制到
  `results/sonicom_siren_a4_confirmation/`；
- 配置 JSON 由 `scripts/generate_siren_a4_configs.py` 确定性生成并提交。

## 7. 数据访问与失败处置

- 只读 `data/processed/sonicom_residual_q26_v1/subjects/<train>/q26.h5`；
- 任一 run 训练失败（非有限值、文件错误）→ 记录并仅重跑该 run，
  不改变任何协议参数；
- 主标准未满足时，报告必须说明可能原因（选择偏差、预算不足、
  单 seed 偶然）并给出后续方案，禁止直接调参重测。

## 8. 产出

- 协议、被试锁定、配置、聚合结果、判定报告
  `reports/film_siren_siren_a4_confirmation.md`、实验日志，全部提交。
