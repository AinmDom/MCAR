# Stage B 预注册协议：Q26-conditioned FiLM-SIREN

> 状态：PRE-REGISTERED IMPLEMENTATION PROTOCOL。本文在任何 Stage B 正式
> training/search run 之前冻结。单元测试和不读取 validation target 的工程
> smoke 不计入搜索；任何协议修改必须先提交并记录原因。

## 1. 目标与边界

- 在冻结的 `SIREN Backbone v1 = D-o20-d6-w256-ho20` 架构上加入 Q26
  subject condition，使一个共享模型能够预测不同被试的 MCA residual；
- 冻结的是 Stage A 的 architecture/hyperparameters，不加载任何 Stage A
  单被试 checkpoint；SIREN、condition encoder 和 modulator 一起从 scratch
  训练；
- Stage B 只搜索 conditioning architecture；loss/optimizer/scheduler 留到
  Stage C；
- SONICOM test 不扫描、不打开、不用于任何选择，`test_subjects_read=0`。

## 2. 固定 split 与数据访问

- 严格沿用官方冻结 split：262 train / 44 validation / 44 test；**不再拆分
  train**，不创建 train 内部 validation；
- 每个正式候选使用全部262个 train subjects 联合训练，以全部44个 validation
  subjects 进行工程模型选择；validation 每次访问记入 ledger；
- condition 只能由 `Q26Condition` 提供：
  `reference_logmag_db[:, q26_indices, :]`、Q26 xyz、mask、frequency；
- 监督 target 走独立训练接口，encoder API 不暴露 target、MCA 或 non-Q26
  reference；
- Q26的26个测量方向只作 condition，不进入训练 query 或主指标；query 与
  主指标均使用767个 `interpolation_evaluation_mask=1` 方向。

## 3. Q26 输入归一化

- 统计来源：完整262 train subjects 的 Q26 reference magnitude；
- 分别对每只耳、每个频点，在 `subject × 26 directions` 轴计算 population
  mean/std，shape 均为 `[2,463]`；
- 固定文件：`configs/data/siren_b_q26_normalization_v1.json`；禁止按被试单独
  缩放；target residual 继续使用现有完整262-train `training_statistics.json`；
- statistics 文件保存 train subject IDs、split/Q26 hashes 与
  `test_subjects_read=0`。

## 4. Condition encoder v1 候选结构

输入为 normalized magnitude `[B,2,Q,463]`、xyz `[B,Q,3]`、mask `[B,Q]`：

1. 每个方向共享频谱编码：
   `Conv1d(2,32,k=9,s=2,p=4) → SiLU →`
   `Conv1d(32,32,k=9,s=2,p=4) → SiLU`，频率长度 `463→232→116`；
2. flatten 的 `32×116` 频谱特征与该方向 xyz 拼接；
3. 共享方向投影 `Linear(32×116+3,128) → SiLU`；
4. masked mean：`sum(mask*h)/sum(mask)`，all-masked 明确报错；
5. `Linear(128, latent_dim)` 得到 subject latent。

必须通过 encoder 级 direction-permutation invariance、masked-padding
invariance、finite forward/backward、save/load 与 CPU/CUDA smoke。

## 5. FiLM 定义

- Backbone：dual frequency、6个 sine layers、width256、first/hidden omega20、
  双耳 normalized residual output；
- modulator：共享 trunk `Linear(latent,128)→SiLU`，每个被调制层使用独立
  channel-wise head；head 输出每个 hidden channel 的 raw 参数；
- 只把最终 heads 的 weight/bias zero-init，trunk 正常初始化；
- phase：`sin(omega * gamma * u + beta)`，
  `gamma=1+0.5*tanh(raw_gamma)`、`beta=pi*tanh(raw_beta)`；
- amplitude：`alpha*sin(omega*u)`，
  `alpha=1+0.5*tanh(raw_alpha)`；
- full：`alpha*sin(omega*gamma*u+beta)`；
- saturation：`abs(tanh(raw))>0.95`，逐层、逐参数种类记录比例和
  mean/std/min/max；
- placement：`late=layers 4–6`、`hidden=layers 2–6`、`all=layers 1–6`。

`concat` 是独立分支：把 subject latent 拼接到 coordinate 后送入第一层，
不定义 placement；若 concat 在 modulation 阶段胜出，跳过 placement 搜索。

## 6. 训练与 validation 语义

- 一个 training cycle = 262 optimizer steps；每个 train subject 在该 cycle
  恰好出现一次，顺序由 run seed 的确定性 permutation 决定；
- 每 step 一个 subject、16个从767 interpolation directions 中等概率无放回
  采样的 query directions、完整463频点和双耳；
- loss：normalized residual MSE；Adam、LR `1e-4`、weight decay0、gradient
  clip5、FP32；不做 direction weighting；
- 固定预算：100 cycles（26200 steps）。每5 cycles 对44 validation subjects
  的全部767 directions 评价一次；
- best checkpoint 依据 validation aggregate solid-angle-weighted residual
  MAE，tie 时选更早 cycle；若 best 在最后一次评价点，记 `RETEST`，不得静默
  延长单个候选；
- validation target 只进入共享 metric，不进入 condition encoder；
- 模型内部输出 normalized residual，prediction adapter 统一转换为 residual dB。

## 7. 主指标

对每个 subject：

`MAE_s = sum_d(w_d * mean_{ear,f}|prediction-target|) / sum_d(w_d)`，

其中 `d` 仅为767个 interpolation directions。方向权重来自
`direction_features[:,5]`，在 mask 后重新归一化；44个 subject 等权平均作为
选择指标。同步报告 subject median/std/range、weighted RMSE 和 MCA
zero-residual baseline。

## 8. 顺序搜索与 seed

Screening seed 固定为 `20260821`；每阶段 Top-2 补齐
`20260822/20260823`，阶段赢家按三个 seed 的44-subject aggregate weighted
MAE 均值确定，并报告 seed std。

1. latent：`{64,128,256}`，固定 phase × hidden placement；若256第一且相对
   第二名差距 `<1%`，追加 `{384,512}` 后再确定 Top-2；
2. modulation：`{concat, amplitude, phase, full}`，使用 latent winner；非
   concat 固定 hidden placement；
3. placement：仅在非 concat winner 时搜索 `{late,hidden,all}`。

每阶段按三-seed均值升序；第一与第二相对差距 `<0.5%` 时标记 tight，但仍以
均值第一进入下一阶段，第二保留为对照。若 seed 排名发生反转，报告并将阶段
标记 `RETEST`，不临时扩展搜索。

预算上限分别冻结：

- 常规分支：最多10个 unique FiLM configurations、22个 FiLM training runs；
- latent扩展分支：最多12个 unique configurations、24个 FiLM runs；
- concat胜出并跳过 placement 时相应减少；
- unconditional baseline 另计3个 runs，总训练 run 上限27；smoke 不计。

## 9. 对照与因果检查

- MCA / zero-residual：无需训练，使用相同44-subject weighted metric；
- unconditional shared SIREN：相同 Backbone、262 train、训练 query、预算、
  三 seeds 与 checkpoint规则，但不读取 condition；
- winner 在 validation 上做 deterministic condition shuffle 和 train-mean
  latent 替换，仅作 evaluation-time ablation，不重新选模；
- Stage A4 per-subject plain-SIREN 结果只称为不同 cohort 的 representation
  reference，不解释为 Stage B validation 的同被试 oracle。

最终 ConditionEncoder v1 必须在三-seed均值上优于 unconditional baseline；
否则 Stage B 判定 `DO NOT FREEZE`，先诊断 conditioning 失效，不进入 Stage C。

## 10. 保存与失败处置

每个 run 保存 config/path/hash、git state、dataset/split/Q26/normalization
hash、262/44 subject IDs、seed、cycle/step定义、模型配置、参数量、best cycle、
完整 validation ledger、weighted metrics、modulation统计、NaN/Inf、显存/耗时、
checkpoint hashes 与 `test_subjects_read=0`。

任何非有限值或文件/schema错误只允许同配置同seed重跑；不得改参数后覆盖。
正式 search 前必须完成：数据泄漏测试、metric 单测、encoder/model 单测、共享
predictor adapter 测试和一个不读取 validation target 的 CUDA smoke。
