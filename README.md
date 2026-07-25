# MCAR：MCA 后的轻量 HRTF 残差学习

本项目研究如何在 MCA（Magnitude-Corrected and Time-Aligned
Interpolation）HRTF 插值结果上，用轻量神经网络降低剩余幅度误差。
MCA 是传统基线，项目主体是残差数据构造、模型训练和严格 HRTF 重建评估。

首版学习目标为：

```text
residual = log|H_ref| - log|H_MCA|
```

## 当前结果

项目已经完成 MLP v1、听觉感知损失 MLP v2 和 MLP + CNN v3。锁定的 12
名 test 被试上，v3 相对 MCA 的全空间 ERB、对侧 25° ERB、对侧高频和
水平面 ILD MAE 分别改善约 27.18%、29.88%、14.99% 和 25.28%。
v3 相对 v2 的前三项幅度指标继续改善，但严格 ILD MAE 回升约 2.30%，
这是后续 v3.1 需要解决的主要问题。

详细报告位于：

- `reports/MLP_V1_REPORT.md`
- `reports/MLP_V2_REPORT.md`
- `reports/MLP_CNN_V3_REPORT.md`

## 项目结构

```text
MCAR/
├─ src/mcar/                 Python 公共实现
├─ matlab/+mcar/             MATLAB 数据导出与严格评估
├─ baselines/mca/            MCA/SUpDEq 基线与复现入口
├─ experiments/              各实验的运行说明
├─ configs/                  数据划分与锁定实验配置
├─ data/                     原始及处理后数据
├─ artifacts/                checkpoint、HDF5、MAT、日志（不提交）
├─ results/                  精选 CSV、JSON 和论文图
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

更完整的训练、验证和重建命令位于 `experiments/`。

## 版本与回溯

重构前版本固定为提交 `a02e337`，标签为
`pre-project-restructure-20260726`。大型本地数据不受 Git 管理，但本次
重构只移动了它们，没有删除。
