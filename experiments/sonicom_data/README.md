# SONICOM 测量数据处理与 MCA residual 导出

本实验把冻结的 350 人 SONICOM 测量队列转换为可供现有 MLP/CNN 管线读取的
逐被试 HDF5。它是测量数据扩展实验，不替代 HUTUBS 上的严格 Lebedev 论文复现。

## 1. 生成几何与划分配置

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.prepare_sonicom_configs
```

该命令生成 SONICOM-Q26-v1、793 点面积权重、767 点纯插值 mask，以及按自由场
EQ 文件分层的 `262 train / 44 validation / 44 test` 固定划分。

## 2. 非 test pilot

已使用 EQ001、EQ006 和 EQ008 各一名 train、一名 validation 被试，对
Tikhonov epsilon 进行如下扫描：

```text
0, 1e-8, 1e-6, 1e-4, 1e-2, 3e-2, 1e-1, 3e-1, 1
```

导出示例：

```powershell
matlab -batch "addpath('matlab'); mcar.export_sonicom_residual_dataset([2 1 243 277 238 242], 6, 'sonicom_pilot_q26_tikh1e2', 1e-2, true, false)"
```

评估命令：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.evaluation.evaluate_sonicom_tikhonov_pilot
```

参数只根据 validation 的 767 点纯插值 ERB proxy MAE 选择，test 未读取。最终
锁定 `tikhonov_epsilon = 0.01`。逐被试和汇总结果位于
`results/sonicom_data_preparation/tikhonov_pilot_v1/`。

## 3. 正式导出

正式参数见 `configs/experiments/sonicom_q26_residual_v1.json`。在首次全量导出
阶段只处理 train 和 validation，共 306 人：

```powershell
matlab -batch "addpath('matlab'); ids=readtable('configs/data/sonicom_subject_split_v1.csv','TextType','string'); ids=ids.subject_id(ids.split~=\"test\"); mcar.export_sonicom_residual_dataset(ids, 6, 'sonicom_residual_q26_v1', 1e-2, true, false)"
```

验证全部 HDF5 后，仅使用 train 文件计算归一化统计：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.compute_training_statistics `
  data/processed/sonicom_residual_q26_v1
```

test 在模型和训练参数锁定前保持不导出。最终评估时必须显式传入
`allowTest=true`，防止开发阶段误读。
