# MCA residual learning

本目录用于在已经复现的 MCA 基线上构建和训练轻量残差模型。首版任务固定为：

```text
target_residual_db = reference_logmag_db - mca_logmag_db
```

## 首版实验范围

- 数据集：96 个 HUTUBS simulated HRTF。
- 稀疏输入：Lebedev `N=3`（26 个方向）。
- 目标网格：Fliege `N=29`（900 个方向）。
- 频率范围：FFT 正频率栅格中 `50 Hz <= f <= 20 kHz`。
- 双耳作为独立通道，划分严格按 subject 进行。
- 初始模型输入：MCA log-magnitude、MCA correction-filter log-magnitude、
  方向单位向量、归一化频率和耳别。
- 初始模型输出：单个方向、频率和耳别处的 log-magnitude residual（dB）。

## Subject-wise 划分

划分文件为 `config/subject_split_v1.csv`：

- train：72 人
- validation：12 人
- test：12 人
- 随机种子：`20260723`
- 生成方式：MATLAB `rng(20260723, 'twister'); randperm(96)`

同一被试不会跨集合出现。公开人体测量缺失的 pp79、pp92、pp18 分别位于
train、validation、test。

## 数据文件

每个被试和阶数保存为一个 HDF5 文件，便于断点续跑和按被试加载。Python
读取后的主要数组布局为 `[ear, direction, frequency]`：

| HDF5 dataset | 含义 | 单位 |
|---|---|---|
| `/mca_logmag_db` | MCA 插值后的 log-magnitude | dB |
| `/reference_logmag_db` | dense reference log-magnitude | dB |
| `/correction_logmag_db` | MCA correction filter magnitude | dB |
| `/target_residual_db` | reference 减 MCA | dB |
| `/direction_features` | azimuth、elevation、x、y、z、Fliege weight | degree / unitless |
| `/frequency_hz` | FFT 正频率 | Hz |

大型 HDF5、状态表和中间文件写入 `data/`，不提交 Git。

## 运行

先运行 pp91、N=3 的小规模检查：

```matlab
addpath(fullfile(pwd, 'residual_learning', 'matlab'));
export_hutubs_residual_dataset(91, 3, 1, 'pilot_pp91_n03');
```

验证：

```powershell
python -m venv residual_learning/.venv
residual_learning/.venv/Scripts/python -m pip install `
  -r residual_learning/requirements.txt
residual_learning/.venv/Scripts/python `
  residual_learning/python/validate_residual_hdf5.py `
  residual_learning/data/pilot_pp91_n03/subjects/pp91/n03.h5
```

验证通过后，全量导出 N=3：

```matlab
addpath(fullfile(pwd, 'residual_learning', 'matlab'));
export_hutubs_residual_dataset(1:96, 3, 6, 'hutubs_residual_v1_n03');
```

全量文件验证及训练集统计：

```powershell
$files = Get-ChildItem `
  residual_learning/data/hutubs_residual_v1_n03/subjects `
  -Recurse -Filter *.h5
residual_learning/.venv/Scripts/python `
  residual_learning/python/validate_residual_hdf5.py `
  --summary-only $files.FullName
residual_learning/.venv/Scripts/python `
  residual_learning/python/compute_training_statistics.py `
  residual_learning/data/hutubs_residual_v1_n03
```

归一化统计量严格只从 72 个 train 被试计算；validation 和 test 不参与均值、
标准差或超参数估计。

## 训练环境和首版 MLP

训练使用本机 Conda 环境 `D:\miniconda3\envs\ml`。已验证其中的 PyTorch
`2.8.0+cu128` 可使用 RTX 5060，CUDA build 为 12.8。该环境还需要：

```powershell
D:\miniconda3\envs\ml\python.exe -m pip install -r residual_learning/requirements.txt
```

首版模型是约 10 万参数的残差 MLP。它使用归一化后的 MCA magnitude、MCA
correction-filter magnitude、方向单位向量、log-frequency 和耳别，共 7 个输入
特征，输出归一化的 dB residual。

先执行单被试过拟合检查：

```powershell
D:\miniconda3\envs\ml\python.exe residual_learning/python/train_residual_mlp.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  --run-name overfit_pp91 --overfit-subject 91 --epochs 8 `
  --steps-per-epoch 300 --validation-steps 64
```

训练日志和检查点位于 `runs/`，不提交 Git。过拟合检查通过后，再用完整
subject-wise train/validation 划分训练。

一轮首版跨被试训练和独立测试：

```powershell
D:\miniconda3\envs\ml\python.exe residual_learning/python/train_residual_mlp.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  --run-name mlp_n03_v1 --epochs 12 --steps-per-epoch 600 --validation-steps 96
D:\miniconda3\envs\ml\python.exe residual_learning/python/evaluate_residual_mlp.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/runs/mlp_n03_v1/best.pt --split test
```

这里的 MAE/RMSE 是逐 FFT 频点的 residual 指标；独立 test 的论文 ERB、ILD 和
对侧高频指标将在模型基本泛化通过后再计算。

当前已完成的 `mlp_n03_v1`（100,225 参数）在完整未见 test 集上将逐频点 residual
MAE 从 `2.6007 dB` 降至 `2.2172 dB`（`14.75%`）。这证明残差可学习，但尚不等价
于论文的 ERB/ILD 指标；下一阶段需要将预测 residual 回填到 MCA 幅度后重新计算。
