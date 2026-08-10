# Magnitude-Corrected and Time-Aligned Interpolation Residual（MCAR）：轻量头相关传输函数（Head-Related Transfer Function，HRTF）残差学习

本项目研究如何在幅度校正与时间对齐插值（Magnitude-Corrected and
Time-Aligned Interpolation，MCA）的 HRTF 插值结果上，用轻量神经网络降低剩余幅度误差。
MCA 是传统基线，项目主体是残差数据构造、模型训练和严格 HRTF 重建评估。

首版学习目标为：

```text
residual = log|H_ref| - log|H_MCA|
```

## 当前结果

SONICOM Q26 的一次性最终 test 已在模型完全锁定并获得用户明确授权后完成。论文主模型为
MLP+CNN v3.2（严格 ILD 权重 `0.75`、epoch 6）：44 名 test 被试上的全空间 ERB、
对侧 25° ERB、对侧高频和水平面严格 ILD MAE 分别为
`0.867805 / 1.365327 / 3.611854 / 0.686999 dB`；相对 MCA 分别改善
`19.81% / 21.79% / 23.13% / 17.17%`，相对正式 v3 分别改善
`0.493% / 0.400% / 0.227% / 2.830%`。v3.2 与 v3.1 的 test ILD 均值近似相同，
但 v3.2 的三项幅度明显更好，因此保持为预声明的单模型主方法。完整最终结果位于
`results/sonicom_mlp_cnn_q26_v32_final_test/` 和
`results/sonicom_mlp_cnn_q26_v32_final_test_strict/`；可直接用于论文的主表、
消融表、配对统计和矢量图位于 `results/sonicom_mlp_cnn_q26_v32_paper/`，完整
结果分析见 `reports/SONICOM_MLP_CNN_V32_FINAL_RESULTS.md`。test 已经使用，任何后续方法改进
不得再用该拆分选模；需要新的未见拆分或外部数据。以下段落保留此前各阶段的历史结果。

v3.2 的三 seed 稳定性诊断也已完成：固定同一 validation sampler，训练 seed
`20260809/20260810/20260811` 各运行 `10 epoch × 500 step`，不保存任何模型权重。
最佳 epoch 为 `10/9/10`，validation total loss 为
`0.671027 ± 0.000346`，ERB、高频和 strict ILD 的 seed 间样本标准差分别为
`0.000601 / 0.001387 / 0.001607 dB`。说明训练指标具有较高 seed 稳定性，且
10 epoch 相对原 6 epoch 仍有小幅 proxy 收益；由于权重未保留且 test 已消费，
该诊断不替换论文模型。结果位于 `results/sonicom_mlp_cnn_q26_v32_seed_stability/`。

按稳定性诊断中表现最好的 seed `20260809`，v3.2 又完成了一次预注册的 40 epoch
validation-only 预算扩展。固定 validation 主目标在 epoch 39 达到最佳；44 人严格
重建的全空间 ERB、对侧 25° ERB、对侧高频和水平面 ILD 为
`0.872882 / 1.364204 / 3.608567 / 0.596141 dB`，相对锁定 v3.2 epoch 6 分别改善
`1.330% / 1.372% / 0.799% / 4.620%`。test 读取数仍为 0；该结果只证明后续预算
优化空间，不改写已完成的一次性 test 论文结论。完整结果位于
`results/sonicom_mlp_cnn_q26_v32_seed20260809_e40_strict_validation/`。

在用户明确授权且 epoch 39 checkpoint、SHA-256、四项指标和对照规则全部预先冻结后，
该候选又执行了一次 44 人 test 追加评价；没有训练或修改参数，评价前后 checkpoint
SHA-256 均为 `1076EBA7…FE11F10C`。全空间 ERB、对侧 25° ERB、对侧高频和水平面
严格 ILD 为 `0.855676 / 1.349975 / 3.590454 / 0.660297 dB`，相对原 locked
epoch 6 分别改善 `1.398% / 1.124% / 0.592% / 3.887%`。该结果作为冻结候选的
透明追加比较保留，后续不得据此继续调参。完整结果位于
`results/sonicom_mlp_cnn_q26_v32_seed20260809_e40_epoch39_frozen_test_strict/`。

在 epoch 39 之后又完成了 validation-only 的低风险 v3.2.1 联合微调实验：解冻 MLP，
使用 `CNN/MLP=1e-5/1e-6` 的分组学习率训练 10 epoch，最佳点为 epoch 7。相对
v3.2 epoch 39，44 人严格 validation 的全空间 ERB、高频和水平面 ILD 分别改善
`0.054% / 0.033% / 0.666%`，对侧 25° ERB 回退仅 `0.0034%`。MLP 参数相对 L2
改变量仅 `0.0997%`，说明解冻稳定但总体收益很小，当前更适合作为消融而非替代模型。
test 读取数为 0；结果位于
`results/sonicom_mlp_cnn_q26_v321_joint_unfreeze_e10_strict_validation/`。

全局频谱上下文 v3.3 消融也已完成：保留并冻结 v3.2 epoch 39 的 MLP 与局部
dilated CNN，将 463 个频点按步长 4 降采样为 116 个 token，经两层、4 头、宽度 32
的轻量 self-attention 后上采样，并用频率自适应门控叠加全局 delta。仅新增的
22,356 个参数训练 10 epoch，最佳训练点为 epoch 9。相对 v3.2 epoch 39，44 人严格
validation 的全空间 ERB、对侧 25° ERB 分别回退 `0.0317% / 0.0353%`；高频和水平面
ILD 仅改善 `0.0051% / 0.0727%`，后两项均无统计显著性。全局分支平均只改变预测
`0.0120 dB`，说明实现确实生效但当前冻结式训练没有带来有效增益；该版本只保留为
负结果消融，不替代 v3.2。test 读取数为 0；结果位于
`results/sonicom_mlp_cnn_q26_v33_global_attention_e10_strict_validation/`。

高频谱差分目标 v3.4a 消融已完成：保持 v3.2 epoch 39 架构、冻结 MLP，以 4 kHz
以上预测误差的一阶/二阶频谱差分 MAE 约束局部 CNN，训练 10 epoch，最佳点为 epoch 7。
相对 v3.2 epoch 39，44 人严格 validation 的全空间 ERB、对侧 25° ERB、对侧高频和
水平面 ILD 分别改善 `0.0303% / 0.0628% / 0.0026% / 0.4852%`。完整 767 个插值
方向上的一阶/二阶差分误差降低 `1.164% / 3.821%`，且两项均为 44/44 人改善，证明
目标函数确实改善了谱形状；但高频幅度均值几乎不变，因此该版本作为正向 loss 消融
保留，尚不全面替代 v3.2。test 读取数为 0；结果位于
`results/sonicom_mlp_cnn_q26_v34a_spectral_diff_e10_strict_validation/`。

分频带 ILD 目标 v3.4b 消融也已完成：从同一 v3.2 epoch 39 出发，保持 MLP 冻结，
在独立水平面采样上新增 200 Hz–18 kHz 的 ERB-band ILD SmoothL1，训练 10 epoch，
最佳点为 epoch 7。相对 v3.2 epoch 39，44 人严格 validation 的全空间 ERB、对侧
25° ERB、对侧高频和水平面严格 ILD 分别改善
`0.0425% / 0.2821% / 0.0055% / 0.5141%`；其中对侧 ERB 为 40/44 人改善。
完整水平面方向上的分频带 ILD SmoothL1/MAE 分别改善 `0.7587% / 0.7082%`，均为
43/44 人改善，35/35 个频带的聚合误差下降。原对侧高频幅度指标仍不显著，因此 v3.4b
是当前最强的定向 loss 消融，但尚不作为全面替代模型。test 读取数为 0；结果位于
`results/sonicom_mlp_cnn_q26_v34b_band_ild_e10_strict_validation/`。

多尺度 notch-aware 目标 v3.4c 消融已完成：在 4–18 kHz 以 4/8/16 个频点半径构造
可微 notch-depth map，同时约束局部凹陷的位置与深度。保持 v3.2 epoch 39 架构、
冻结 MLP，独立训练 10 epoch，最佳点为 epoch 7。相对 v3.2 epoch 39，44 人严格
validation 的全空间 ERB、对侧 25° ERB、对侧高频和水平面严格 ILD 分别改善
`0.0267% / 0.0946% / 0.0127% / 0.5362%`。完整 767 个插值方向的 notch-depth MAE
改善 `0.1139%`，44/44 人以及三个尺度均改善。目标有效但收益弱于 v3.4a/v3.4b 的
主目标改善，且高频幅度仍不显著；因此保留为正向消融，不全面替代 v3.2。test 读取数
为 0；结果位于
`results/sonicom_mlp_cnn_q26_v34c_notch_aware_e10_strict_validation/`。

新增的 FSP-AE-Q26 横向基线已完成官方 checkpoint 逐值等价、40 epoch 正式训练、
44 人 validation 推理和 MATLAB 严格评价。锁定 epoch 40 的全空间 ERB、对侧 25°
ERB、对侧高频与水平面严格 ILD 为 `1.160 / 1.848 / 3.104 / 0.598 dB`；相对
MCAR v3.2，前两项回退 `31.16% / 33.60%`，后两项改善 `14.68% / 4.37%`。
该结果说明 FSP-AE 是高频与双耳线索互补基线，而不是 MCAR 的全面替代。test 未读取；
完整结果见 `reports/FSP_AE_Q26_VALIDATION_REPORT.md` 和
`results/sonicom_fsp_ae_q26_formal_validation/`。

项目已经完成多层感知机（Multi-Layer Perceptron，MLP）v1、听觉感知损失
MLP v2、MLP + 卷积神经网络（Convolutional Neural Network，CNN）v3，以及用严格
头相关脉冲响应（Head-Related Impulse Response，HRIR）能量耳间电平差
（Interaural Level Difference，ILD）损失微调的 v3.1。锁定的 12 名 test
被试上，v3.1 相对 MCA 的全空间等效矩形带宽（Equivalent Rectangular
Bandwidth，ERB）、对侧 25° ERB、对侧高频和水平面 ILD 平均绝对误差
（Mean Absolute Error，MAE）分别改善约
27.08%、29.56%、15.01% 和 26.31%。相对 v3，v3.1 将严格 ILD MAE 从
`0.661566` 降到 `0.652495 dB`，同时高频误差略降；全空间 ERB 仅回退
约 0.14%。这说明与最终重建口径对齐的可微 ILD 损失有效，但 v2 的
`0.646722 dB` 仍是更低的 ILD 单项结果。

SONICOM 测量扩展已完成 262 train / 44 validation 的 Q26 MCA residual
导出和正式轻量 MLP v1 训练。12/20 epoch 预算比较后锁定 20-epoch run 的
epoch 19；完整 validation 的 767 个纯插值方向面积加权 MAE 由 MCA 的
`3.322289` 降至 `2.825515 dB`（改善 `14.95%`），RMSE 由 `5.029170` 降至
`4.218639 dB`。44 名 test 被试仍未导出或读取。

以锁定的 v1 epoch 19 初始化的 SONICOM MLP v2 已完成权重预实验和正式训练。
按预先声明的 maximin 规则锁定 `ERB / 对侧高频 / ILD = 0.75 / 0.25 / 0.25`；
正式模型在固定 validation 上将三项感知代理分别改善
`4.74% / 0.85% / 11.80%`。完整 44 人、767 个纯插值方向的 raw residual
MAE/RMSE 为 `2.807803 / 4.210356 dB`，相对 v1 再改善
`0.63% / 0.20%`，相对 MCA 的 MAE 改善为 `15.49%`。44 名 test 被试仍未
导出或读取。

SONICOM v1/v2 的严格 validation 重建评估也已完成。相对 MCA，v2 的全空间
ERB、对侧 25° ERB、对侧高频与水平面 HRIR ILD MAE 分别改善
`14.69% / 17.83% / 16.49% / 21.27%`；相对 v1 再改善
`5.30% / 4.72% / 0.59% / 11.90%`。完整结果和 44 人 HRTF 总览位于
`results/sonicom_mlp_q26_v2_strict_validation/`，test 仍保持零读取。

SONICOM MLP+CNN v3 已从锁定的 v2 epoch 10 初始化并完成 CNN-only 正式训练。
完整 44 人、767 个纯插值方向的面积加权 raw residual MAE/RMSE 为
`2.581873 / 3.935166 dB`；MAE 相对 v2 改善 `8.05%`，相对 MCA 改善
`22.29%`，并在 44/44 名 validation 被试上优于 v2。全空间严格 HRIR ILD
MAE 由 v2 的 `0.641157` 降至 `0.621980 dB`，改善 `2.99%`。正式训练摘要位于
`results/sonicom_mlp_cnn_q26_v3_formal/`。最终 MATLAB 重建评估中，v3 相对 v2
将全空间 ERB、对侧 25° ERB、对侧高频和水平面严格 ILD MAE 分别改善
`5.07% / 4.20% / 7.98% / 1.32%`；相对 MCA 分别改善
`19.01% / 21.28% / 23.16% / 22.31%`。完整表格和 44 人 HRTF 总览位于
`results/sonicom_mlp_cnn_q26_v3_strict_validation/`，test 仍保持零读取。

SONICOM MLP+CNN v3.1 已完成面向 72 个纯插值水平面方向的严格 HRIR ILD
微调与 44 人最终重建评估。相对 v3，水平面 ILD MAE 从 `0.644804` 降至
`0.622905 dB`，改善 `3.40%`，并在 29/44 名 validation 被试上改善；代价是
全空间 ERB、对侧 25° ERB 和对侧高频误差分别回退
`3.42% / 0.77% / 1.74%`。尽管如此，v3.1 三项幅度误差仍分别优于 v2
`1.82% / 3.46% / 6.38%`。因此当前同时保留 v3 作为幅度均衡基线、v3.1
作为 ILD 优化基线，而不以 v3.1 全面替代 v3。完整结果和五方法 44 人 HRTF
总览位于 `results/sonicom_mlp_cnn_q26_v31_strict_validation/`；test 仍保持零读取。

详细报告位于：

- `reports/MLP_V1_REPORT.md`
- `reports/MLP_V2_REPORT.md`
- `reports/MLP_CNN_V3_REPORT.md`
- `reports/MLP_CNN_V31_REPORT.md`
- `reports/SONICOM_MLP_CNN_V32_FINAL_RESULTS.md`
- `reports/FSP_AE_Q26_VALIDATION_REPORT.md`

## 项目结构

```text
MCAR/
├─ src/mcar/                 Python 公共实现
├─ matlab/+mcar/             MATLAB 数据导出与严格评估
├─ baselines/mca/            MCA/SUpDEq 基线与复现入口
├─ experiments/              各实验的运行说明
├─ configs/                  数据划分与锁定实验配置
├─ data/                     原始及处理后数据
├─ artifacts/                checkpoint、Hierarchical Data Format version 5（HDF5）、MATLAB MAT-file（MAT）、日志（不提交）
├─ results/                  精选 Comma-Separated Values（CSV）、JavaScript Object Notation（JSON）和论文图
├─ reports/                  完整技术报告
├─ external/SUpDEq/          本地第三方工具包（不提交）
└─ docs/                     实验日志与结构说明
```

各目录的职责和旧路径迁移表见 `docs/PROJECT_STRUCTURE.md`。

## 环境

Python 包采用 `src` 布局：

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[tracking]"
```

MATLAB 实验依赖本地 `external/SUpDEq/` 和
`data/HRTF/hutubs/`。这些大型第三方文件和数据不会提交到 Git。

## 主要入口

检查并下载冻结的 SONICOM 干净测量队列：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.download_sonicom `
  --dry-run `
  --workers 8

D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.download_sonicom `
  --workers 8
```

下载器默认获取 350 名被试的
`FreeFieldCompMinPhase_44kHz` SOFA；选择规则、pilot 命令和输出清单见
`data/HRTF/README.md`。

生成 SONICOM-Q26-v1 稀疏方向、793 点面积权重和固定被试划分：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.prepare_sonicom_configs
```

准备并验证 FSP-AE-Q26 横向基线（默认仅 train/validation）：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.prepare_sonicom_fsp_ae `
  --splits train val
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_fsp_ae `
  configs/experiments/sonicom_fsp_ae_q26_budget_pilot.json --device cuda
```

完整协议、推理与严格 MATLAB 评价命令见 `experiments/fsp_ae/README.md`。

该步骤输出 26 点测量原生稀疏网格以及
`262 train / 44 validation / 44 test` 的自由场 EQ 分层划分，配置位于
`configs/data/`。

使用已锁定的 Q26 和 Tikhonov 参数导出 SONICOM train/validation residual：

```powershell
matlab -batch "addpath('matlab'); ids=readtable('configs/data/sonicom_subject_split_v1.csv','TextType','string'); ids=ids.subject_id(ids.split~=\"test\"); mcar.export_sonicom_residual_dataset(ids,6,'sonicom_residual_q26_v1',1e-2,true,false)"
```

详细 pilot、参数选择和验证命令见 `experiments/sonicom_data/README.md`。

对 SONICOM train/validation 数据执行面积加权、纯插值方向的 MLP smoke
training：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_mlp_v1 `
  data/processed/sonicom_residual_q26_v1 `
  --run-name sonicom_mlp_q26_smoke_v1 `
  --epochs 3 --steps-per-epoch 120 --validation-steps 64 `
  --direction-sampling solid_angle --interpolation-only `
  --wandb-mode online --wandb-project mcar-sonicom
```

导出 residual 数据：

```powershell
matlab -batch "addpath('matlab'); mcar.export_hutubs_residual_dataset([], 3, 6, 'hutubs_residual_v1_n03')"
```

训练 v1：

```powershell
.venv/Scripts/python -m mcar.training.train_mlp_v1 `
  data/processed/hutubs_residual_v1_n03 `
  --run-name mlp_n03_v1
```

训练 v2：

```powershell
.venv/Scripts/python -m mcar.training.train_mlp_v2 `
  data/processed/hutubs_residual_v1_n03 `
  artifacts/training/mlp_n03_v1/best.pt `
  --run-name mlp_n03_v2
```

训练 v3：

```powershell
.venv/Scripts/python -m mcar.training.train_mlp_cnn_v3 `
  data/processed/hutubs_residual_v1_n03 `
  artifacts/training/mlp_n03_v2/best.pt `
  --run-name mlp_cnn_n03_v3
```

用 v3 checkpoint 进行严格 ILD 微调（v3.1）：

```powershell
.venv/Scripts/python -m mcar.training.train_mlp_cnn_v3 `
  data/processed/hutubs_residual_v1_n03 `
  artifacts/training/mlp_n03_v2/best.pt `
  --initial-cnn-checkpoint artifacts/training/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt `
  --config configs/experiments/mlp_cnn_v31_finetune_v3.json `
  --run-name mlp_cnn_n03_v31_finetune_v3_strict_ild `
  --wandb-mode online
```

更完整的训练、验证和重建命令位于 `experiments/`。

## 版本与回溯

重构前版本固定为提交 `a02e337`，标签为
`pre-project-restructure-20260726`。大型本地数据不受 Git 管理，但本次
重构只移动了它们，没有删除。
