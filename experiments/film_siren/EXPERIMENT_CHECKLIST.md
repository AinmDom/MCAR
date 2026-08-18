# MCAR FiLM-SIREN 主模型实验清单

## 适用范围

- 数据：SONICOM Q26 residual dataset。
- 分支：`codex/project-structure-refactor`。
- 目标：构建 `MCA + FiLM-conditioned SIREN` residual reconstruction 模型，并判断其是否适合作为 MCAR v3.5.1 的后继工程主模型。
- 本文件是仓库内的规范版本，取代工作区外的草案。

## 0. 固定定义

- [ ] 网络预测双耳 log-magnitude residual：

  \[
  r=L_{\mathrm{ref}}-L_{\mathrm{MCA}},\qquad
  \widehat L=L_{\mathrm{MCA}}+\widehat r.
  \]

- [ ] 第一阶段保持 MCA 相位不变。
- [ ] train/validation/test 固定为 262/44/44，不再细分 train。
- [ ] normalization 只使用完整 262-subject train 统计。
- [ ] 所有严格指标由共享 MCAR evaluator 计算；模型代码只产生 `[ear,direction,frequency]` residual dB。
- [ ] Full-sphere 指标统一注明是否为 `interpolation_only`；正式 comparison 使用与 v3.5.1 完全相同的 mask 和 solid-angle weights。

## 1. 数据与 split 使用

### Train

- [ ] 所有正式训练使用 262 train subjects。
- [ ] Stage A 的单人实验只允许选择 train subject。
- [ ] 不从 test HDF5 扫描或读取训练所需信息。

### Validation

- [ ] validation 是工程模型选择 split，不声称为无偏泛化估计。
- [ ] 第一次运行前冻结搜索空间、最大实验数和 Top-K 规则。
- [ ] Stage A 优先用 train-subject 固定方向 holdout；完整 strict validation 只给每阶段 Top-K 使用。
- [ ] 每次查看 validation 都写入 experiment ledger；完整记录不等同于消除选择偏差。

### Test

- [ ] SONICOM test 已被项目历史实验使用，只称为 `historically consumed engineering benchmark`。
- [ ] FiLM-SIREN 候选进入 test 前必须冻结架构、三个 seed、ensemble、epoch、配置及 checkpoint hashes。
- [ ] test 不用于 architecture、hyperparameter、checkpoint、seed 或 ensemble-weight 选择。
- [ ] 同一次授权读取中可以输出三个已冻结 seed 和等权 ensemble 的结果，但不得据此修改模型。
- [ ] 新版本号不能恢复 test 的统计独立性；任何 test 后产生的模型都不能把同一 SONICOM test 当作新的确认集。
- [ ] 论文级泛化结论必须使用新的 untouched split、外部数据集或跨数据集确认；该项不是可选扩展。

## 2. 共享预测与严格评价

- [ ] 定义共享 `ResidualPredictorProtocol`。
- [ ] predictor 统一输出 residual dB，shape 为 `[2,D,F]`。
- [ ] MLP、MLP-CNN、SIREN、FiLM-SIREN 只实现 prediction adapter。
- [ ] 统一进入现有 MCA magnitude/phase reconstruction 和 strict evaluator。
- [ ] 用一个 validation subject 验证旧 predictor 与新 adapter 数值一致。
- [ ] 禁止为 FiLM-SIREN 复制 ERB、HF、strict ILD 等指标实现。

## 3. Q26 condition 数据接口

- [ ] 新增 `Q26Condition`：indices `[Q]`、双耳 magnitude `[2,Q,F]`、xyz `[Q,3]`、mask `[Q]`、frequency `[F]`。
- [ ] condition magnitude 只能来自 `reference_logmag_db[:, q26_indices, :]`。
- [ ] encoder 禁止读取 non-Q26 reference、target residual 或 dense-ground-truth-derived features。
- [ ] condition 路径和监督 target 路径在 API 层隔离。

必须通过：

- [ ] Q26 index correctness。
- [ ] sparse magnitude equality。
- [ ] non-Q26 mutation invariance。
- [ ] target-residual mutation invariance。
- [ ] non-Q26 NaN isolation。
- [ ] API leakage test。
- [ ] direction permutation invariance。
- [ ] padding/mask invariance。
- [ ] all-masked input 明确报错。

## 4. 标准 SIREN 规范

### 结构

- [x] `sine_layer_count` 表示 sine 层总数，包含第一层；baseline 固定为 6。
- [x] 第一层输入：`[x,y,z,f_coordinate]`，dual frequency 时为 5 维。
- [x] 每个 sine layer：`sin(omega * (W x + b))`。
- [x] width 256，first omega 30，hidden omega 30，双耳 linear output。
- [x] 无 normalization layer、无 dropout。

### 初始化

- [x] 第一层 weight：`U(-1/d_in, 1/d_in)`。
- [x] 后续 sine layer weight：`U(-sqrt(6/d_in)/omega, +sqrt(6/d_in)/omega)`。
- [x] output weight 使用最后一个 hidden layer 的 SIREN hidden bound。
- [x] bias 明确定义为 `U(-1/sqrt(d_in), +1/sqrt(d_in))`，以复现 seeded PyTorch/SIREN reference 行为。
- [x] 初始化定义和实际范围有单元测试。

### 输出单位

- [x] 模型内部输出 normalized residual。
- [x] `r_db = r_normalized * target_std + target_mean`。
- [ ] 所有 auditory loss、prediction adapter 和重建在 residual dB 上工作。
- [x] checkpoint 保存 normalization path 和 SHA-256。

### 数值测试

- [x] forward shape、finite backward gradients、NaN/Inf。
- [x] save/load round trip。
- [x] CPU FP32。
- [x] CUDA FP32 与 AMP smoke test。
- [ ] AMP 与 FP32 差异明显时允许 sine backbone 强制 FP32。

## 5. Stage A1：单 subject representation baseline

- [x] 固定使用 train subject P0002（numeric id 2）。
- [x] 只用于代码正确性、representation ceiling 和粗略失败排除，不用于最终 omega/depth/width 选择。
- [x] baseline 使用全部 793 directions 做 all-direction fit，明确标注为 training-field representation，不解释为空间泛化。
- [ ] 另建固定 direction-holdout 实验后才能报告方向插值能力。
- [x] baseline 使用 linear frequency 映射到 `[-1,1]`、Adam、LR `1e-4`、weight decay 0、FP32。
- [x] 训练目标为 normalized residual MSE；评价报告 raw residual dB MAE/RMSE、>8 kHz、>10 kHz、谱差分和 notch metric。

## 6. Stage A2-A7：正式 backbone 搜索

- [ ] 以固定 seed/分层规则选 5 个 train subjects，不根据已有模型误差临时换人。
- [ ] 每个配置为 5 个 subjects 分别训练 5 个独立 plain SIREN，再聚合；plain SIREN 不跨 subject 共用网络。
- [ ] 每个 subject 固定 all-direction fit 与 direction-holdout indices，并保存 indices/hash。
- [ ] 联合粗筛 `frequency in {linear,ERB,dual}` × `first omega in {20,30,50}`。
- [ ] 使用相同预算，选 Top 2 后依次搜索 depth、width、hidden omega。
- [ ] 用额外固定 16-32 个 train subjects 只做 confirmation，不重新搜索。
- [ ] 通过后冻结 `SIREN Backbone v1`。

Frequency mapping 必须保存公式、实现版本、输入范围和输出范围。当前数据范围来自冻结 training statistics，不允许每个 subject 单独缩放。

## 7. Stage B：FiLM conditioning

- [ ] Q26 双耳频谱经 shared 1D spectral encoder、per-direction MLP、masked-mean DeepSets pooling 得到 subject latent。
- [ ] 比较 latent dimension 64/128/256，必要时 384/512。
- [ ] 比较 latent concat、post-sine amplitude、pre-sine phase FiLM、full modulation。
- [ ] pre-sine 使用 `sin(omega * gamma * u + beta_phase)`。
- [ ] `gamma=1+g_max*tanh(raw_gamma)`，默认 `g_max=0.5`、raw zero-init。
- [ ] `beta_phase=beta_max*tanh(raw_beta)`，默认 `beta_max=pi`、raw zero-init。
- [ ] full modulation 使用 `alpha=1+a_max*tanh(raw_alpha)`，默认 `a_max=0.5`、raw zero-init。
- [ ] 比较后半层、除第一层外全部、全部 hidden layers 的 placement。
- [ ] 记录 modulation saturation/mean/std/min/max。
- [ ] global latent 稳定后比较 global、local MCA、global+local；local 信息必须为推理时可得信息。

## 8. Stage C：跨 subject 正式训练

- [ ] 训练 262 subjects，评价 44 validation subjects。
- [ ] 默认启用 interpolation-only、direction-weighted residual、dual-sampling strict HRIR ILD。
- [ ] 初始 objective：residual + 0.75 ERB + 0.25 HF + 0.75 strict ILD + 0.05 band ILD。
- [ ] 架构稳定前不同时搜索 loss。
- [ ] 之后再做 spectral-difference/notch ablation。
- [ ] 搜索 LR、Adam/AdamW、weight decay 和 scheduler；最大实验数量预注册。
- [ ] Top 3 configurations 各运行 3 seeds。

## 9. Epoch、scheduler 与 ensemble 冻结

- [ ] 由最终 configuration 三个开发 seed 的 validation best epochs 得到 `E_final=round(median(E1,E2,E3))`。
- [ ] best epoch 的 metric、平滑规则和最早/最晚 tie-break 预先固定。
- [ ] 正式训练从 scratch 运行到固定 `E_final`，不 early stop、不选择 validation checkpoint。
- [ ] `scheduler_horizon_epochs` 与 `stop_epoch` 分开保存；正式重训保持开发时完全相同的 LR trajectory，仅在 `E_final` 截断。
- [ ] 正式模型为三个 seed 的 1/3 等权 residual ensemble，禁止搜索权重或删除 seed。
- [ ] 同时报告三个组成模型和 ensemble；报告 single-network 与 effective-ensemble 参数量、显存和端到端延迟。

## 10. Frozen manifest 与 test registry

- [ ] frozen manifest 包含 model version、git commit/dirty state、config/checkpoint hashes、seed、ensemble、E_final、scheduler horizon。
- [ ] 同时保存 dataset schema、split CSV、Q26 grid、normalization、frequency mapping、interpolation mask policy 和 evaluator commit/hashes。
- [ ] frozen test 命令必须同时要求 `--allow-test` 与 manifest。
- [ ] registry 以 config/checkpoint hash 集为身份依据，不能靠重命名 model version 绕过。
- [ ] registry 原子写入 `started/completed/failed`；只允许对可证明未完成且模型未变的失败任务恢复。
- [ ] checkpoint/config/hash 不匹配时在打开 test HDF5 前失败。

## 11. 工程晋级统计规则

- [ ] 所有 gates 明确应用于冻结 SONICOM engineering test，不把它解释为独立论文确认。
- [ ] paired difference 定义为 FiLM-SIREN 减 MCAR v3.5.1，负值更好。
- [ ] primary endpoint：interpolation-only solid-angle-weighted Full-sphere ERB。
- [ ] bootstrap 以 44 个 subject paired rows 为重采样单位，10000 次，seed 20260818。
- [ ] 明确采用双侧 percentile 95% CI；primary superiority 要求 upper bound `<0`。
- [ ] secondary engineering non-inferiority margins：Contra ERB 0.02 dB、HF 0.05 dB、strict ILD 0.02 dB；margin 必须在看到正式 FiLM 结果前用历史稳定性/工程容忍度说明。
- [ ] 所有 secondary upper bounds 小于各自 margin。
- [ ] Full ERB wins 至少 26/44，解释为工程多数分布 gate，不声称 sign-test significance；若要 sign-test `p<0.05`，阈值改为至少 28/44 并在运行前冻结。

## 12. Ablation、稀疏度和 continuous query

- [ ] ablation 只用 train+validation，不进入 test。
- [ ] 比较 plain、concat、amplitude、phase FiLM、full modulation；global/local/global+local；frequency 和 omega。
- [ ] 可变方向数需要训练协议，而不仅是 set-shaped API：预先选择逐 Q 重训、Q6/Q14/Q26 mixed-cardinality training 或受控 direction dropout。
- [ ] 所有方法使用现有 nested sparse grids 和同一公平协议。
- [ ] 若论文声称 continuous implicit field，必须测试 unseen coordinates、azimuth/elevation sweep、连续性和 notch trajectory。
- [ ] 论文级结论必须完成独立/外部确认。

## 13. 每次实验保存

- [ ] Experiment ID、model version、branch、commit/dirty state。
- [ ] config/path/hash、dataset/split/grid/normalization hashes。
- [ ] subjects、direction protocol、test subjects read、seed。
- [ ] architecture、parameter count、frequency mapping、omega/depth/width。
- [ ] conditioning、latent、modulation bounds/placement。
- [ ] loss、optimizer、LR trajectory、epochs、scheduler horizon、best epoch。
- [ ] elapsed time、peak VRAM、inference time。
- [ ] residual MAE/RMSE、Full ERB、Contra ERB、HF、strict ILD、spectral difference、notch。
- [ ] NaN/Inf、leakage tests、test guard tests、checkpoint/result paths and hashes。
- [ ] Decision：KEEP、REJECT、RETEST、FREEZE、PROMOTE 或 DO NOT PROMOTE，并写明依据。

## 14. 执行顺序

1. 单 subject SIREN baseline 与数值测试。
2. Q26 loader、leakage tests、shared prediction adapter 和 test guard。
3. 五人 backbone matrix 与 direction holdout。
4. 多人 backbone confirmation 并冻结。
5. DeepSets condition encoder 与 FiLM variants。
6. 正式跨 subject objective/optimizer 搜索、Top-3、3 seeds。
7. 固定 epoch、LR trajectory、ensemble、manifest。
8. 一次冻结 SONICOM engineering comparison。
9. ablation、sparsity、continuous query。
10. 外部或独立论文级确认。
