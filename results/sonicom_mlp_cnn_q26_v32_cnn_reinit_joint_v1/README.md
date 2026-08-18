# SONICOM Q26 v3.2 CNN/MLP 初始化、联合解冻与输出融合实验

## 结论

本实验回答两个独立问题。第一，已有 v3 CNN 初始化具有明确价值：在相同 seed、
40 epoch 预算和 v3.2 损失下，从零输出 CNN 重新训练相对使用 v3 CNN 初始化的
epoch 39，在全空间 ERB、对侧 25° ERB、对侧高频和水平面严格 ILD 上分别回退
`1.347% / 1.472% / 1.312% / 0.996%`。前三项回退的 44 人配对统计一致显著；
ILD 差异不显著。因此不应以 CNN 重初始化替代当前预训练路径。

第二，在相同 CNN 重初始化 checkpoint 和相同 10 epoch 预算下，低学习率解冻 MLP
具有小而可辨别的收益。相对 CNN-only continuation，联合微调的全空间 ERB、对侧
高频和水平面严格 ILD 分别改善 `0.0528% / 0.0314% / 0.1797%`，三项 paired
t-test 的 `p` 分别为 `1.80e-5 / 0.00107 / 0.0148`。对侧 25° ERB 回退
`0.0570%`，但 t-test `p=0.0803` 且 95% CI 跨 0。MLP 相对 L2 改变量只有
`0.1179%`，说明解冻过程稳定，但它无法弥补 CNN 重初始化造成的整体差距。

第三，完全不加载 v2 或 CNN checkpoint、MLP/CNN 隐藏层随机初始化且两个输出头置零，
以 `MLP/CNN=1e-3/3e-4`、2 epoch warmup 从零联合训练 60 epoch 后，四项严格
validation 达到 `0.837668 / 1.302105 / 3.542947 / 0.598536 dB`。相对上一轮
重初始化联合模型，前三项分别改善 `5.419% / 6.226% / 3.007%` 且配对统计显著；
ILD 改善 `0.259%`，但不显著。说明足够训练预算下，完整 scratch 联合优化优于保留
v2 MLP 的本次分阶段结果；由于预算和学习率也不同，不能把差异单独归因于随机初始化。

第四，该输出融合模型正式记为 **MCAR v3.5**。两套独立初始化模型不做参数平均，而在 residual 输出层进行凸组合。固定
validation sampler 搜索得到上一轮联合模型 `30%` + scratch `70%`；proxy total
进一步降至 `0.641223`。严格四项为
`0.831843 / 1.291992 / 3.540258 / 0.581337 dB`，均优于 scratch。相对 scratch
分别改善 `0.695% / 0.777% / 0.076% / 2.874%`；除对侧高频的小幅差异外，其他
三项配对 t-test 均显著。

本实验只读取 262 名 train 和 44 名 validation 被试；`test_subject_count_read=0`。
论文主模型及既有 test 结论不变。

## 协议

- 阶段 1：保留正式 v2 MLP 并冻结；不加载 CNN checkpoint。CNN 内部层使用原生
  初始化，输出投影严格置零。训练 `40 epoch × 500 step`，CNN 学习率 `1e-4`。
- 阶段 2 对照：从阶段 1 最佳 checkpoint 出发，只训练 CNN，学习率 `1e-5`，
  共 `10 epoch × 500 step`。
- 阶段 2 联合：从同一 checkpoint 出发，CNN/MLP 学习率分别为
  `1e-5 / 1e-6`，共 `10 epoch × 500 step`。
- 三组均使用 training seed `20260809`、固定 validation sampler seed `20260805`、
  双采样 `32` 个全空间方向加 `32` 个水平面方向，以及
  `ERB/HF/strict-ILD = 0.75/0.25/0.75`。
- checkpoint 均按固定 validation total loss 最小值选择。
- scratch 分支不读取任何预训练 checkpoint，MLP/CNN 隐藏层使用原生随机初始化，
  两个输出投影严格置零；训练 `60 epoch × 500 step`，MLP/CNN 目标学习率为
  `1e-3/3e-4`，前 2 epoch 线性 warmup 后使用 cosine decay。
- 融合仅在模型输出上执行：`0.3 × previous_joint + 0.7 × scratch`。候选权重只用
  固定 validation sampler 的 96 batches 选择，未读取 test。

## 严格 validation 结果

| 方法 | total proxy | 全空间 ERB | 对侧 25° ERB | 对侧高频 | strict ILD |
|---|---:|---:|---:|---:|---:|
| v3 CNN 初始化，epoch 39 | 0.661115 | 0.872882 | 1.364204 | 3.608567 | 0.596141 |
| CNN 重初始化，epoch 36 | 0.673068 | 0.884641 | 1.384283 | 3.655912 | 0.602080 |
| CNN-only continuation，epoch 8 | 0.672634 | 0.886132 | 1.387762 | 3.653945 | 0.601172 |
| MLP+CNN 联合，epoch 8 | 0.672247 | 0.885664 | 1.388553 | 3.652797 | 0.600091 |
| scratch MLP+CNN，epoch 60 | 0.646540 | 0.837668 | 1.302105 | 3.542947 | 0.598536 |
| MCAR v3.5：previous 30% + scratch 70% | **0.641223** | **0.831843** | **1.291992** | **3.540258** | **0.581337** |

四项严格指标单位均为 dB，越低越好。短程低学习率联合微调仍只应作为初始化/解冻
消融；60 epoch scratch 与 30/70 输出融合则是更强的 validation-only 候选。由于权重
和融合比例均在 validation 上确定，且既有 SONICOM test 已经消费，本实验不追加读取
test，也不改写论文主模型结论。

## 文件

- `comparison.csv`：六种单模型/融合方案的训练代理与严格重建指标。
- `paired_statistics.csv`：原两组比较以及 scratch/融合新增比较的配对统计。
- `scratch_fusion_paired_statistics.csv`：只包含 scratch 与融合相关的 12 行统计。
- `parameter_diagnostics.json`：MLP/CNN 参数改变量与源 checkpoint 完整性。
- 三组完整逐被试结果分别位于相邻的
  `sonicom_mlp_cnn_q26_v32_cnn_reinit_e40_strict_validation/`、
  `sonicom_mlp_cnn_q26_v32_cnn_reinit_continuation_e10_strict_validation/` 和
  `sonicom_mlp_cnn_q26_v321_cnn_reinit_joint_unfreeze_e10_strict_validation/`。
- scratch 与融合的严格结果分别位于相邻的
  `sonicom_mlp_cnn_q26_v32_scratch_joint_e60_strict_validation/` 和
  `sonicom_mlp_cnn_q26_v32_previous_scratch_fusion_strict_validation/`；权重搜索报告位于
  `sonicom_mlp_cnn_q26_v32_previous_scratch_fusion/`。
