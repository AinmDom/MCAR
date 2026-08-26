# Stage B global / local MCA / global+local 三模型扩展协议

> 状态：PRE-REGISTERED EXPANSION。冻结时点为 bounded gate 两个正式 E150
> runs 完成并输出 `PASS_EXPAND` 后、任何新增 expansion run 前。

## 1. 触发证据与复用

- `global_zero_local` seed20260821：`3.0888695415 dB`，best cycle 60，KEEP；
- `global_plus_local_mca` seed20260821：`2.7144948679 dB`，best cycle 70，KEEP；
- 相对改善 `12.120119%`，超过预注册 `0.5%` 工程门槛；
- 两个合规 gate runs 直接计入三-seed复核，不重跑。

## 2. 三个候选

所有候选使用相同七维 query 接口和 latent128/full/all FiLM-SIREN：

1. `global_zero_local`：真实 Q26 global latent，local 两通道为零；
2. `local_mca`：真实双耳 query MCA，global latent 固定为零；不构造或读取 Q26
   condition encoder 输入；
3. `global_plus_local_mca`：真实 Q26 global latent与真实双耳 query MCA。

local MCA 只允许使用 Q26-derived `mca_logmag_db`，按 train-only
`training_statistics.json` 的 `mca_mean/mca_std` 归一化。local-only 模型保留相同
encoder/FiLM参数结构以对齐容量，但 encoder 被旁路且不参与梯度；零 latent 只允许
学习被试无关的共享调制偏置。

## 3. 新增 runs

- seeds 固定为 `20260821/20260822/20260823`；
- 复用两个 gate seed20260821 runs；新增 local seed20260821，以及三个候选各自
  seed20260822/20260823，共新增7个 E150 runs；
- 每个 run 均从 scratch，150 cycles、262 steps/cycle、16 directions/step、
  normalized residual MSE、Adam LR `1e-4`、weight decay 0、gradient clip 5、FP32；
- 每5 cycles完整评价44个 validation subjects 的767个 interpolation directions；
- 任一新增 run best cycle=150 即暂停剩余运行并标记整体 `RETEST`。

## 4. 最终判据

每个候选分数为三个 seeds 的 best validation solid-angle-weighted residual MAE
等权均值，最低者为架构 winner。同步报告 seed std、matched-seed差值、逐被试差值
和每个seed胜出情况；这些诊断不改变均值判据。

- winner 为 `global_plus_local_mca`：重新冻结 Stage B 架构，并重新验证 Stage C；
- winner 为 `global_zero_local`：恢复原 global 架构与 Stage C C4；
- winner 为 `local_mca`：global encoder不提供增量价值，冻结前需做 local MCA
  subject-shuffle 因果检查；
- 任一 run 缺失、非有限、metadata/hash不匹配、Git dirty或读取test时，
  `winner=null`。

## 5. 数据访问与预算

- global-only 不读取 query MCA；local-only 不读取 Q26 encoder condition；
- global+local 只读取上述两类推理时可用输入；三者均禁止 non-Q26 reference 或
  target 进入输入；
- SONICOM test 不解析、不打开，所有产物记录 `test_subjects_read=0`；
- 本扩展新增7个正式 FiLM runs，独立登记，不改写既有 Stage B run 记录。
