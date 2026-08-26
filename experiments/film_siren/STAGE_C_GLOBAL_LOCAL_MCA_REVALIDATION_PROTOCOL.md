# Stage C global+local MCA 独立重验证协议

> 状态：PRE-REGISTERED。本文必须在首个七维 global+local Stage C 正式 run
> 前提交。它落实 `STAGE_C_GLOBAL_LOCAL_MCA_ARCHITECTURE_AMENDMENT.md`，不覆盖、
> 合并或重新解释旧五维 global-only C1--C3 结果。

## 1. 新的固定架构边界

- query 固定为七维：`[x,y,z,dual_frequency_1,dual_frequency_2,
  normalized_mca_left,normalized_mca_right]`；
- global condition 固定为 Q26 encoder latent128，full modulation，all placement；
- local MCA 固定读取同一被试、同一 query 方向和频点的 Q26-derived
  `mca_logmag_db`，用 `training_statistics.json` 中 train-only MCA mean/std
  归一化；
- 所有候选从 scratch 训练，不加载 Stage B 或旧 Stage C checkpoint；
- 旧五维 C1--C3 仅作为历史记录，不进入新架构候选排序，也不作为 inherited
  winner 复用。

## 2. 数据、采样、objective 和验证

除 query 增加两路 local MCA 外，全部沿用 `STAGE_C_TRAINING_PROTOCOL.md`：

- 262 train / 44 validation；test 路径不得构造或打开；
- 每 cycle 262 steps，每名 train subject 恰好一次；
- global 767 中无放回抽16，horizontal 72 中独立无放回抽16；
- 双耳、463频点、FP32、无 AMP、gradient clip 5；
- C0 为 direction-weighted normalized residual SmoothL1，加权
  `0.75 ERB + 0.25 HF + 0.75 strict HRIR ILD + 0.05 band ILD`；
- 每5 cycles 对44名 validation subject 的完整767/72方向评价，以 subject
  等权 mean Stage C total 严格下降保存 best；末次评价为 best 时标记 RETEST。

## 3. 从头顺序重验证

screening seed 固定为 `20260821`，每个 unique configuration 为150 cycles。
阶段间只继承本协议上一阶段 winner：

1. **GL-C1 learning rate**：Adam、wd=0、constant、C0，比较
   `{3e-5,1e-4,3e-4}`；
2. **GL-C2 optimizer/decay**：固定 GL-C1 winner LR，比较 Adam wd=0、
   AdamW wd=`1e-5`、AdamW wd=`1e-4`；
3. **GL-C3 scheduler**：固定 GL-C2 winner，比较 constant、cosine、
   5-cycle linear warmup + cosine；horizon 固定150；
4. **GL-C4 objective**：固定 GL-C3 winner，比较 C0、C0+D1/D2、
   C0+notch、C0+D1/D2+notch；权重和频率边界完全沿用原 Stage C 协议。

重复的 inherited winner 不重跑，故 screening 最多10个 unique runs。候选使用
原协议的 total、strict ILD、ERB、HF、residual MAE 顺序排序；完全相同才使用
复杂度 tie-break。

## 4. Top-3、最终 cycle 与 ensemble

- screening 全局 Top 3 各补跑 seeds `20260822/20260823`，搜索最多16 runs；
- 最终 configuration 的三 seed best cycles 为 `E1/E2/E3`，固定
  `E_final=round(median(E1,E2,E3))`；
- 三个正式模型从 scratch 训练到固定 E_final，不 early stop、不选 validation
  checkpoint；scheduler horizon 仍为150；
- 最终 residual 为三个 seed 的1/3等权平均，不搜索权重、不删除 seed；
- 独立 frozen manifest 和 test registry 提交前不得访问 SONICOM test。

## 5. 独立身份与首个 run

- 新配置、run、结果目录统一使用 `sonicom_film_siren_gl_*` 前缀；
- provenance 必须记录 `conditioning_scope=global_plus_local_mca`、
  `coordinate_dimension=7`、local MCA policy/hashes、读取计数和
  `test_subjects_read=0`；
- 首个 run 为 GL-C1 的 `Adam / lr=1e-4 / wd=0 / constant / C0`，seed
  `20260821`。其余两个 GL-C1 候选在任何 GL-C1 决策前完成。
