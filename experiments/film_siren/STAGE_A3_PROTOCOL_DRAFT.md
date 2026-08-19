# Stage A3 预注册协议草案：backbone 深化搜索（depth / width / hidden-omega）

> **状态：DRAFT**。本文件在 Stage A2 的 Top-2 结果确定后正式化；
> 当前记录的是预注册的搜索空间、预算与决策规则框架。
> 正式化时补充：Top-2 配置名、候选值最终清单、各阶段最大实验数。

## 1. 目的与范围

- 在 Stage A2 选出的 Top-2 配置（frequency × first-omega）上，依次搜索
  `depth（sine layer count）`、`width（hidden width）`、`hidden omega`，
  得到候选 backbone；随后用额外固定的 16–32 名 train 被试做 confirmation，
  通过后冻结 `SIREN Backbone v1`。
- 与 A2 相同：只读取 train 被试；固定 5 名 A2 被试 + 同一 64 方向 holdout；
  相同预算（100 epochs × 100 steps）、相同 optimizer/初始化/频率映射公式。

## 2. 搜索空间（候选，待 Top-2 后定稿）

| 阶段 | 维度 | 候选值 | 说明 |
|---|---|---|---|
| A3-1 | depth（sine layer count） | `{4, 6, 8}` | 在 Top-2 的频率/omega 上，每配置 × 每 depth 一个 run（5 被试聚合） |
| A3-2 | width（hidden width） | `{128, 256, 512}` | 使用 A3-1 最优 depth |
| A3-3 | hidden omega | `{20, 30, 50}`（若 A2 显示 omega 敏感，改为 `{25, 30, 35}`） | 使用 A3-1/2 最优 depth/width |

- 每阶段在固定 5 名 A2 被试上独立训练并聚合 holdout RMSE（与 A2 同口径）；
- 每阶段只改变一个维度，其余沿用上一阶段最优值；
- 若某阶段最优与次优差距 < 0.5% 且与第三差距 > 2%，同时推进两名进入下一阶段，
  否则只推进第一名（与 A2 的 Top-K 规则一致）。

## 3. Confirmation（A3-4）

- 额外固定 16–32 名 train 被试（分层规则沿用 A2：`Avg RMS dB (Free Field)`
  quintile 扩层，seed 固定；被试清单提前锁定并提交）；
- 用最终候选 backbone 在 confirmation 被试上各训练一个 plain SIREN
  （同一 holdout 协议，方向集合与 A2 相同）；
- 通过标准：confirmation 平均 holdout RMSE 不劣于 A2 粗筛最优配置在
  confirmation 被试上的对应值（或预注册的绝对阈值，正式化时冻结）；
- 通过后冻结 `SIREN Backbone v1`（记录 config/checkpoint hashes 与 git state）。

## 4. 数据访问

- 与 A2 相同：只读 train HDF5，`test_subjects_read` 恒为 0；
- 被试/方向锁定文件版本化（`siren_a2_*_v1` 不变，新增 confirmation 清单）。

## 5. 产出

- 各阶段 run 产物与 A2 相同结构（`artifacts/training/<run_name>/<PXXXX>/`）；
- 阶段聚合写入 `artifacts/siren_a3_matrix/`；
- 结果与决策写入实验日志并提交。
