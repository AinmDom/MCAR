# 处理后数据

这里保存由 MATLAB 导出的 residual-learning Hierarchical Data Format
version 5（HDF5）文件。大型文件不会提交 Git。

当前正式数据集：

```text
data/processed/hutubs_residual_v1_n03/
├─ subjects/                 96 被试 HDF5（忽略）
└─ training_statistics.json train-only 归一化统计（提交）

data/processed/sonicom_fsp_ae_q26_v1/
├─ subjects/train/           262 被试 FSP-AE 输入缓存（忽略）
├─ subjects/val/             44 被试 FSP-AE 输入缓存（忽略）
└─ normalization.json        train-only 流式归一化统计（忽略）
```

导出入口：

```powershell
matlab -batch "addpath('matlab'); mcar.export_hutubs_residual_dataset([], 3, 6, 'hutubs_residual_v1_n03')"
```

FSP-AE SONICOM 缓存入口默认只允许 train/validation：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.prepare_sonicom_fsp_ae `
  --splits train val
```
