# 处理后数据

这里保存由 MATLAB 导出的 residual-learning Hierarchical Data Format
version 5（HDF5）文件。大型文件不会提交 Git。

当前正式数据集：

```text
data/processed/hutubs_residual_v1_n03/
├─ subjects/                 96 被试 HDF5（忽略）
└─ training_statistics.json train-only 归一化统计（提交）
```

导出入口：

```powershell
matlab -batch "addpath('matlab'); mcar.export_hutubs_residual_dataset([], 3, 6, 'hutubs_residual_v1_n03')"
```
