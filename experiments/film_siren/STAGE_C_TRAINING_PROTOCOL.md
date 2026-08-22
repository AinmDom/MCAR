# Stage C 预注册协议：FiLM-SIREN 跨被试听觉目标训练

> 状态：PRE-REGISTERED。本文必须在首个 Stage C 正式训练 run 前提交。
> 单元测试、合成数据测试，以及不读取 validation target 的工程 smoke 不计入
> 搜索。任何搜索空间或判据变更必须先提交 amendment，不能覆盖既有 run。

## 1. 固定边界

- 架构冻结为 Stage B winner：`latent128 + full modulation + all placement`；
- 所有候选从 scratch 训练，不加载 Stage B checkpoint；
- 固定 split 为 262 train / 44 validation，不再划分 train；
- Q26 的26个方向只用于 condition；global query、训练监督与主 validation
  均限于767个 `interpolation_evaluation_mask=1` 方向；
- SONICOM test 在 Stage C 搜索、epoch 选择和消融中禁止解析或打开，所有正式
  产物必须记录 `test_subjects_read=0`。

## 2. 训练单位与双采样

- 一个 cycle 为262个 optimizer steps，每名 train subject 在每个 cycle 恰好
  出现一次，顺序由 run seed 的确定性 permutation 决定；
- 每个 step 对同一 subject 编码一次 Q26 condition；
- global batch 从767个插值方向中等概率、无放回抽16个方向，用于 residual、
  ERB、对侧高频及后续谱差分/notch；
- horizontal batch 独立地从72个水平面插值方向中等概率、无放回抽16个方向，
  用于 strict HRIR ILD 与 spectral-band ILD；两个 batch 不共享方向抽样；
- 双耳和全部463个频点始终成块进入 loss；FP32、gradient clip 5，不启用 AMP。

## 3. 初始 objective C0

模型输出 normalized residual。记预测 residual 为 $\hat r$，target 为 $r$，
修正幅度为 MCA + $\hat r$。初始总损失固定为：

`direction-weighted residual SmoothL1(normalized, beta=1)`
`+ 0.75 * ERB_MAE_dB / target_std`
`+ 0.25 * contralateral_HF_MAE_dB / target_std`
`+ 0.75 * strict_HRIR_ILD_MAE_dB / target_std`
`+ 0.05 * spectral_band_ILD_SmoothL1_dB / target_std`。

- ERB 使用50 Hz–20 kHz的41个 ERB-rate 三角带；
- HF 使用对侧半球的 `f > 10 kHz` residual absolute error；
- strict ILD 使用原 MCA selected-bin phase、outside-bin complex spectrum 和
  reference ILD 重建，与最终 HRIR 全带能量 ILD 定义一致；
- band ILD 使用中心频率200–18000 Hz的 ERB bands，SmoothL1 `beta=0.5 dB`；
- 所有方向聚合使用 `direction_features[:,5]` 并在当前 batch 内重新归一化。

## 4. Validation、best cycle 与排序

- 每5 cycles 对全部44名 validation subjects 评价；每名 subject 的 global
  objective 使用全部767个插值方向，horizontal objective 使用全部72个水平面
  插值方向，不使用随机 validation batch；
- checkpoint selection metric 为44名 subject 等权平均的 Stage C objective
  total；不平滑；严格小于历史最佳才更新，因此完全相等时保留更早 cycle；
- 同步报告44-subject等权的 residual MAE、ERB MAE、对侧 HF MAE、strict ILD
  MAE、band ILD MAE、谱差分和 notch诊断，以及全部767方向的面积加权 residual
  MAE/RMSE；
- 候选排序使用三个 seed 的 best validation total 等权均值，随后依次以
  strict ILD、ERB、HF、residual MAE 的三-seed均值作为 tie-break；数值完全相同
  时选较低复杂度（constant scheduler、Adam、较小 weight decay、较小 LR）；
- best 位于最后评价点时标记 `RETEST`，不得把该 run 当作收敛证据或单独延长。

## 5. 顺序搜索与最大预算

所有 screening 首先使用 seed `20260821`、150 cycles。后续阶段只继承上一阶段
winner，不做全笛卡尔积。

1. **C1 learning rate**：Adam、weight decay 0、constant scheduler，比较
   `{3e-5, 1e-4, 3e-4}`；
2. **C2 optimizer/decay**：固定 C1 winner LR，比较 C1 winner、
   `AdamW wd=1e-5`、`AdamW wd=1e-4`；
3. **C3 scheduler**：固定 C2 winner，比较 constant、cosine，以及
   5-cycle linear warmup + cosine；`scheduler_horizon_cycles=150`；
4. **C4 objective ablation**：固定 C3 winner，比较：
   - C0；
   - C0 + D1/D2，权重 `0.25/0.15`，频率 `>=4 kHz`；
   - C0 + multi-scale notch，权重 `0.30`，4–18 kHz、半径
     `{4,8,16}` bins、depth threshold `1 dB`、softplus temperature
     `0.5 dB`；
   - C0 + D1/D2 + notch，附加项权重不变。

重复的 inherited winner 不重跑。因此最多10个 unique configurations：C1三项、
C2新增两项、C3新增两项、C4新增三项。筛选完成后，按上述同一排序规则选全局
Top 3，每项补跑 seeds `20260822/20260823`；包括已有 screening runs 在内，
Stage C 搜索最多16个正式 training runs。失败仅允许同 config、同 seed 重跑。

## 6. 最终 epoch 与 ensemble（Stage 9）

- 最终 configuration 的三个开发 seed 得到 best cycles `E1/E2/E3`，固定
  `E_final = round(median(E1,E2,E3))`；
- 三个正式模型从 scratch 训练到固定 `stop_cycle=E_final`，不 early stop、不选
  validation checkpoint；
- scheduler 仍按开发时 `scheduler_horizon_cycles=150` 计算，只在 E_final 截断；
- 正式输出为三个 seed 的1/3等权 residual ensemble，不搜索权重、不删除 seed；
- test manifest/registry 单独预注册并提交后才可请求 test 访问。

## 7. 首个正式 run

首个 run 为 C1 的 `Adam / lr=1e-4 / wd=0 / constant / C0`，seed
`20260821`。它与 Stage B winner 的架构、采样数和150-cycle预算对齐，仅将
normalized residual MSE 替换为本协议的 Stage C 初始 objective。
