# Stage B global + local MCA bounded gate 协议

> 状态：PRE-REGISTERED ARCHITECTURE GATE。本协议在任何 local-MCA 正式
> training run 前冻结，只判断 query-level MCA 是否值得重新打开 Stage B。

## 1. 目的与暂停边界

- 当前冻结对照为 `latent128 + full modulation + all placement`；
- `global` 指完整 Q26 condition 经 encoder 得到的 subject latent；
- `local MCA` 指仅由该被试 Q26 测量生成、在当前 query 方向和频点可获得的
  双耳 `mca_logmag_db`；
- 本 gate 只比较 `global` 与 `global+local MCA`。`local-only` 仅在 gate 通过后
  的完整三模型复核中加入；
- Stage C C4 在 gate 决策前暂停。既有 C1--C3 结果保留为 global 架构结果，
  不重解释为 global+local 结果。

## 2. 两个候选

两个候选使用完全相同的七维 query 接口：

`[x, y, z, dual_frequency_1, dual_frequency_2, local_mca_left, local_mca_right]`。

1. `global_zero_local`：最后两个通道恒为零，使用真实 Q26 global latent；不得
   读取 query 的 `mca_logmag_db`；
2. `global_plus_local_mca`：最后两个通道为当前 query 的双耳
   `mca_logmag_db`，使用 train-only `training_statistics.json` 中的统一
   `mca_mean/mca_std` 归一化，同时使用真实 Q26 global latent。

两者均为 latent128、full modulation、all placement。相同七维接口保证第一层
参数量和随机初始化可配对比较。

## 3. 数据安全

- local MCA 必须来自现有 `mca_logmag_db`，其生产仅依赖同被试 Q26 测量；
- 训练 query 和 validation 仅限 767 个 interpolation directions；
- `reference_logmag_db` 的 non-Q26 部分、`target_residual_db` 或由 target 推导的
  量不得进入 query feature；target 只进入监督和共享 metric；
- `global_zero_local` 必须记录 `local_mca_inputs_read=0`；
- SONICOM test 不解析、不构造路径、不打开，`test_subjects_read=0`。

## 4. 训练与评价

- 两个候选均使用 seed `20260821`，全部从 scratch；
- 150 cycles、262 steps/cycle、16 interpolation directions/step；
- normalized residual MSE、Adam、LR `1e-4`、weight decay 0、gradient clip 5、
  FP32；每 5 cycles 完整评价 44 validation subjects 的 767 个方向；
- checkpoint 仍按 aggregate solid-angle-weighted residual MAE 的单点最小值保存；
- 若任一 best cycle 为 150，则 gate 为 `RETEST`，不作架构判断；不得续训、
  临时调参或只延长其中一个候选。

## 5. 一次性门槛

记两个 run 的 best validation MAE 为 `M_global` 与 `M_global_local`：

`relative_improvement = (M_global - M_global_local) / M_global * 100%`。

- `relative_improvement >= 0.5%`：`PASS_EXPAND`，暂停 Stage C，补做
  global/local/global+local 三候选三 seed 的完整复核；
- 否则：`KEEP_GLOBAL`，Stage B 架构不变，恢复 Stage C C4；local-only 与
  global+local 只保留为后续解释性消融；
- 0.5% 是是否值得推翻既有架构和重跑 Stage C 的工程阈值，不解释为统计显著性；
- 同 seed、逐被试差值只作诊断，不改变上述判据。

## 6. 完整性要求

正式决策要求两个 run 均完整、finite、配置/协议/hash 匹配、Git clean，且
`test_subjects_read=0`。任一条件失败则输出 `winner=null`，不得以 smoke 或部分
训练结果代替正式 gate。
