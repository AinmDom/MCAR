# Magnitude-Corrected and Time-Aligned Interpolation Residual（MCAR）：轻量头相关传输函数（Head-Related Transfer Function，HRTF）残差学习

本项目研究如何在幅度校正与时间对齐插值（Magnitude-Corrected and
Time-Aligned Interpolation，MCA）的 HRTF 插值结果上，用轻量神经网络降低剩余幅度误差。
MCA 是传统基线，项目主体是残差数据构造、模型训练和严格 HRTF 重建评估。

首版学习目标为：

```text
residual = log|H_ref| - log|H_MCA|
```

## 当前结果

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

详细报告位于：

- `reports/MLP_V1_REPORT.md`
- `reports/MLP_V2_REPORT.md`
- `reports/MLP_CNN_V3_REPORT.md`
- `reports/MLP_CNN_V31_REPORT.md`

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
