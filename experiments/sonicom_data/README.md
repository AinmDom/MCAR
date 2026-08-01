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

2026-07-31 的正式导出已经完成。306/306 个 HDF5 与固定划分一一对应，其中
262 个 train、44 个 validation、0 个 test，总大小约 `3.96 GiB`。全量
strict-ILD 和 residual 恒等式校验通过，共包含 `224,701,308` 个 residual
样本。只使用 train 的 `192,391,316` 个样本计算得到 target mean/std 为
`-0.062128/4.954315 dB`。可提交的运行摘要位于
`results/sonicom_data_preparation/full_export_v1/summary.json`，大型 HDF5 和
完整训练统计继续保存在 Git 忽略的数据目录中。

test 在模型和训练参数锁定前保持不导出。最终评估时必须显式传入
`allowTest=true`，防止开发阶段误读。

## 4. residual MLP smoke training

smoke 配置位于
`configs/experiments/sonicom_mlp_q26_smoke_v1.json`。它只用于确认训练链路，
不作为正式模型结论：3 epoch × 120 step，使用 262 名 train、44 名
validation，方向按球面面积权重抽样，并通过
`interpolation_evaluation_mask` 排除 26 个稀疏输入方向。

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_mlp_v1 `
  data/processed/sonicom_residual_q26_v1 `
  --run-name sonicom_mlp_q26_smoke_v1 `
  --epochs 3 `
  --steps-per-epoch 120 `
  --validation-steps 64 `
  --directions-per-batch 64 `
  --frequencies-per-batch 128 `
  --direction-sampling solid_angle `
  --interpolation-only `
  --seed 20260731 `
  --wandb-mode online `
  --wandb-project mcar-sonicom `
  --wandb-group q26-smoke `
  --wandb-job-type mlp-v1-smoke `
  --wandb-tags mlp-v1 sonicom q26 smoke
```

训练完成后，在全部 44 名 validation 被试和 767 个纯插值方向上做面积加权
逐频点评估：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.evaluation.evaluate_residual_mlp `
  data/processed/sonicom_residual_q26_v1 `
  artifacts/training/sonicom_mlp_q26_smoke_v1/best.pt `
  --split val `
  --direction-weighting solid_angle `
  --interpolation-only
```

该 smoke 已于 2026-07-31 完成。RTX 5060 上 3 epoch × 120 step 用时
`24.10 s`，峰值 CUDA allocated memory 为 `54.09 MiB`。抽样 validation
MAE 相对 MCA 改善 `6.97%`；最佳 epoch 3 在完整 44 人、767 点面积加权
validation 上将 MAE 从 `3.322289` 降到 `3.100075 dB`，改善 `6.69%`，
RMSE 从 `5.029170` 降到 `4.580354 dB`。W&B run 为
`https://wandb.ai/luyoung/mcar-sonicom/runs/hmekqxio`，精选摘要位于
`results/sonicom_mlp_q26_smoke_v1/`。该结果只证明链路可训练，不用于替代
正式预算下的模型结论。

## 5. residual MLP v1 正式预算比较

正式配置位于
`configs/experiments/sonicom_mlp_q26_v1_budget_comparison.json`。在相同数据
口径、seed、模型和每 epoch 600 个训练 step 下，分别训练 12 和 20 epoch；
每组均在完整 44 人、767 个纯插值方向上做面积加权评估。

12-epoch run 用时 `346.35 s`，最佳为 epoch 12，完整 validation MAE/RMSE 为
`2.872643 / 4.265143 dB`；20-epoch run 用时 `602.75 s`，最佳为 epoch 19，
完整 validation MAE/RMSE 为 `2.825515 / 4.218639 dB`。后者相对 MCA MAE
改善 `14.95%`，相对 12-epoch 再改善 `1.64%`，因此被锁定为 SONICOM MLP v1。

- 12 epoch W&B：`https://wandb.ai/luyoung/mcar-sonicom/runs/6wi22jix`
- 20 epoch W&B：`https://wandb.ai/luyoung/mcar-sonicom/runs/ek1nvkhh`
- 锁定配置：`configs/experiments/sonicom_mlp_q26_v1_locked.json`
- 精选结果：`results/sonicom_mlp_q26_v1_budget_comparison/`

锁定 checkpoint 位于已忽略的
`artifacts/training/sonicom_mlp_q26_v1_e20/best.pt`。模型选型期间 test 仍未
导出或读取；下一步以该 checkpoint 初始化 SONICOM 感知损失 MLP v2。

## 6. residual MLP v2 smoke

使用锁定的 v1 epoch 19 初始化，采用原 HUTUBS v2 的复合权重：ERB `0.50`、
对侧高频 `0.25`、ILD spectral proxy `0.25`。双耳 sampler 在 767 个插值
方向中均匀抽样，损失内部再使用面积权重；residual SmoothL1/MAE 也使用同一
方向权重，避免 SONICOM 纬圈密度偏置和重复加权。

3 epoch × 120 step 用时 `33.15 s`，峰值显存 `144.29 MiB`。相对 epoch 0
的锁定 v1，固定 validation 的 ERB/高频/ILD 代理分别改善
`1.50% / 0.53% / 7.23%`，四项指标均未退化。完整 44 人 raw residual MAE
为 `2.824472 dB`，略优于 v1 的 `2.825515 dB`。W&B run：
`https://wandb.ai/luyoung/mcar-sonicom/runs/buhhtjwi`；精选结果位于
`results/sonicom_mlp_q26_v2_smoke/`。

该 smoke 只确认默认损失方向有效；其后的权重预实验与正式训练见下一节。

## 7. residual MLP v2 权重预实验与正式训练

权重预实验保持 smoke 的 `3 epoch × 120 step`、固定 validation、seed 和数据
口径不变，比较以下四组 `ERB / 对侧高频 / ILD` 权重：

- balanced：`0.50 / 0.25 / 0.25`
- erb_heavy：`0.75 / 0.25 / 0.25`
- ild_heavy：`0.50 / 0.25 / 0.50`
- erb_ild_heavy：`0.75 / 0.25 / 0.50`

选择规则在运行前锁定：raw residual 相对初始 v1 的退化不得超过 `0.5%`；在
合格候选中，首先最大化 ERB、对侧高频和 ILD 三项相对改善率的最小值，然后
依次比较三项平均改善率、raw residual MAE 和总权重。四组均满足 residual
约束；`erb_heavy` 的最小感知改善率最高，为 `0.5305%`，因此锁定
`0.75 / 0.25 / 0.25`。完整比较表位于
`results/sonicom_mlp_q26_v2_weight_pilot/comparison.csv`。

正式训练使用 10 epoch × 500 step、96 个固定 validation block、AdamW
`3e-4`、cosine schedule、AMP 和 seed `20260731`。运行用时 `296.34 s`，峰值
CUDA allocated memory 为 `144.29 MiB`，最佳点为 epoch 10。固定 validation
从初始 v1 到 v2 的 residual / ERB / 对侧高频 / ILD 为
`2.847674→2.825701 / 1.175123→1.119363 / 4.027407→3.993006 /
0.738389→0.651290 dB`，分别改善 `0.77% / 4.74% / 0.85% / 11.80%`。

完整 44 人、767 个纯插值方向、`31,250,648` 个逐频点样本上的面积加权 raw
residual MAE/RMSE 为 `2.807803 / 4.210356 dB`；锁定 v1 为
`2.825515 / 4.218639 dB`，因此 v2 分别再改善 `0.63% / 0.20%`。相对 MCA
`3.322289 dB`，v2 的 MAE 改善为 `15.49%`。锁定内容如下：

- 配置：`configs/experiments/sonicom_mlp_q26_v2_locked.json`
- checkpoint：`artifacts/training/sonicom_mlp_q26_v2_formal_erb075_ild025/best.pt`
- 精选曲线与摘要：`results/sonicom_mlp_q26_v2_formal/`
- W&B run ID：`h7kmgoxc`，当前保存在本地 offline run 中

新增的三个权重预实验和正式 run 尚未上传 W&B；本地数据完整保留，等待用户
明确授权上传。已在线的 balanced smoke run `buhhtjwi` 不受影响。模型选择期间
test 读取数保持为 0。下一步先在 validation 上执行预测 residual 回填，按最终
HRTF 重建口径计算严格 ERB magnitude error、对侧高频误差和 HRIR ILD；完成并
锁定方法后再显式决定是否导出 test。

## 8. residual MLP v1/v2 严格 validation 重建评估

严格评估配置位于
`configs/experiments/sonicom_mlp_q26_v2_strict_validation.json`。GPU 推理分别
读取锁定的 v1 epoch 19 与 v2 epoch 10 checkpoint，只处理固定划分中的 44 名
validation 被试：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.evaluation.predict_sonicom_residuals `
  data/processed/sonicom_residual_q26_v1 `
  artifacts/training/sonicom_mlp_q26_v1_e20/best.pt `
  sonicom_q26_validation_v1

D:\miniconda3\envs\ml\python.exe -m mcar.evaluation.predict_sonicom_residuals `
  data/processed/sonicom_residual_q26_v1 `
  artifacts/training/sonicom_mlp_q26_v2_formal_erb075_ild025/best.pt `
  sonicom_q26_validation_v2
```

两次推理分别用时 `8.66 s` 和 `8.43 s`。每名被试输出完整的
`2 ears × 793 directions × 463 frequency bins` residual；预测位于 Git 忽略的
`artifacts/reconstruction/`。随后运行 MATLAB 严格评估：

```powershell
matlab -batch "addpath('matlab'); mcar.evaluate_sonicom_validation_reconstruction([], 'sonicom_mlp_q26_v2_strict_validation', true)"
```

回填时把预测 residual 加到 `50–20000 Hz` 的 MCA log magnitude，保留原 MCA
相位；频段外复数谱也保持 MCA 原值，再转换为双边频谱、IFFT 并裁剪为原始
256-sample HRIR。ERB magnitude error 使用与原 HUTUBS 评估一致的
`AKerbError`；空间统计只包含 767 个纯插值方向并使用归一化 solid-angle
weight。对侧高频统计 `10–20 kHz` 的耳特异对侧开放半球；ILD 使用水平面纯
插值方向上的 `10*log10(left HRIR energy/right HRIR energy)`。

44 人均值如下：

| 指标 | MCA (dB) | MLP v1 (dB) | MLP v2 (dB) | v2 vs MCA | v2 vs v1 |
|---|---:|---:|---:|---:|---:|
| 全空间 ERB | 1.095738 | 0.987171 | 0.934826 | 14.69% | 5.30% |
| 对侧 25° ERB | 1.763508 | 1.520947 | 1.449139 | 17.83% | 4.72% |
| 对侧高频 | 4.749208 | 3.989510 | 3.965865 | 16.49% | 0.59% |
| 水平面严格 ILD MAE | 0.830014 | 0.741741 | 0.653461 | 21.27% | 11.90% |

v2 相对 v1 在 44/44 名被试上改善全空间 ERB 和对侧 25° ERB，在 33/44 名上
改善对侧高频，在 37/44 名上改善严格 ILD。HDF5 中预存的 reference ILD 与由
原始 SOFA HRIR 重新计算的最大误差仅 `9.54e-7 dB`。数值表、summary 和三张
总览图位于 `results/sonicom_mlp_q26_v2_strict_validation/`。本阶段 test 读取数
仍为 0。

## 9. MLP+CNN v3 权重预实验、正式训练与全量 validation

v3 从锁定的 SONICOM MLP v2 epoch 10 初始化。MLP 的 `100,225` 个参数完全
冻结，只训练沿频率轴建模局部谱形的 1D CNN；模型总参数量为 `174,627`，其中
CNN 可训练参数为 `74,402`。CNN 输出层零初始化，因此 epoch 0 与 v2 严格等价。
为复用 HUTUBS 训练器而不改变其默认行为，v3 CLI 新增
`--interpolation-only` 和 `--direction-weighted-residual`；SONICOM 正式训练同时
启用两项，只采样 767 个纯插值方向并对 residual 损失使用 solid-angle weight。

先以 `3 epoch × 120 step` 固定预算比较 ILD spectral proxy 权重
`0.25 / 0.50 / 0.75`，ERB 和对侧高频权重固定为 `0.75 / 0.25`。预先声明的
选择规则要求 residual、ERB 和对侧高频均不得退化，再最大化 ERB、对侧高频和
ILD 三项相对改善率的最小值。三个短 run 的 ILD proxy 均短暂退化；`0.75` 的
退化最小（`-2.08%`）且满足其他约束，因此锁定为正式训练权重。完整比较位于
`results/sonicom_mlp_cnn_q26_v3_weight_pilot/`。

正式配置为 `10 epoch × 500 step`、96 个固定 validation block、每 block 32 个
方向、AdamW、cosine schedule、AMP 和 seed `20260731`，损失权重为
`ERB / 对侧高频 / ILD = 0.75 / 0.25 / 0.75`：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_mlp_cnn_v3 `
  data/processed/sonicom_residual_q26_v1 `
  artifacts/training/sonicom_mlp_q26_v2_formal_erb075_ild025/best.pt `
  --config configs/experiments/sonicom_mlp_cnn_q26_v3_formal.json `
  --run-name sonicom_mlp_cnn_q26_v3_formal_ild075 `
  --wandb-mode offline
```

正式训练用时 `376.47 s`，峰值 CUDA allocated memory 为 `220.46 MiB`，没有
跳过 optimizer step，最佳点为 epoch 10。相对固定 epoch 0 v2，residual、ERB、
对侧高频与 ILD proxy 分别改善 `8.00% / 5.94% / 8.36% / 4.53%`；短预算中的
ILD 退化已在更长训练后反转。正式摘要和曲线位于
`results/sonicom_mlp_cnn_q26_v3_formal/`。

随后对 44 名 validation、767 个纯插值方向执行完整面积加权逐频点评估：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.evaluation.evaluate_mlp_cnn_v3 `
  data/processed/sonicom_residual_q26_v1 `
  artifacts/training/sonicom_mlp_cnn_q26_v3_formal_ild075/best.pt `
  --split val `
  --directions-per-block 64 `
  --direction-weighting solid_angle `
  --interpolation-only `
  --strict-ild `
  --output-dir artifacts/evaluation/sonicom_mlp_cnn_q26_v3_formal_ild075
```

评估覆盖 `31,250,648` 个逐频点样本。v3 raw residual MAE/RMSE 为
`2.581873 / 3.935166 dB`，v2 为 `2.807803 / 4.210356 dB`，MCA MAE 为
`3.322289 dB`；v3 MAE 相对 v2 改善 `8.05%`，相对 MCA 改善 `22.29%`，并在
44/44 名被试上优于 v2。由 HDF5 reference HRIR 与预测幅度计算的全空间严格
HRIR ILD MAE 从 v2 的 `0.641157` 降至 `0.621980 dB`，改善 `2.99%`，35/44 名
被试改善。该 ILD 是全空间诊断指标，不替代下一阶段的水平面最终重建指标。

锁定配置为 `configs/experiments/sonicom_mlp_cnn_q26_v3_locked.json`，checkpoint
位于已忽略的
`artifacts/training/sonicom_mlp_cnn_q26_v3_formal_ild075/best.pt`。正式 W&B run
ID 为 `5mjq0y51`，当前仅保存在本地 offline run，未经用户明确授权不会上传。
首次全量评估在计算完 44 人后触发了只适用于 HUTUBS 的历史参考值断言；现已把
该断言限定到 HUTUBS 数据集并成功重跑评估，训练无需重跑。本阶段 test 读取数
仍为 0。

## 10. MLP+CNN v3 严格 validation 重建评估

新增 `predict_sonicom_mlp_cnn_residuals.py`，按完整双耳频谱调用 v3 CNN，并把
每名 validation 被试的 `2 ears × 793 directions × 463 frequency bins`
residual 原子写入 Git 忽略的 `artifacts/reconstruction/`。44 人 GPU 推理用时
`14.34 s`，checkpoint epoch 为 10，test 读取数为 0：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.evaluation.predict_sonicom_mlp_cnn_residuals `
  data/processed/sonicom_residual_q26_v1 `
  artifacts/training/sonicom_mlp_cnn_q26_v3_formal_ild075/best.pt `
  sonicom_q26_validation_mlp_cnn_v3
```

`evaluate_sonicom_validation_reconstruction.m` 已扩展为向后兼容的可选 v3 模式；
不传第四个参数时原 v1/v2 入口保持不变。正式命令为：

```powershell
matlab -batch "addpath('matlab'); mcar.evaluate_sonicom_validation_reconstruction([], 'sonicom_mlp_cnn_q26_v3_strict_validation', true, 'sonicom_q26_validation_mlp_cnn_v3')"
```

44 人最终重建均值如下：

| 指标 | MCA (dB) | MLP v2 (dB) | MLP+CNN v3 (dB) | v3 vs MCA | v3 vs v2 |
|---|---:|---:|---:|---:|---:|
| 全空间 ERB | 1.095738 | 0.934826 | 0.887467 | 19.01% | 5.07% |
| 对侧 25° ERB | 1.763508 | 1.449139 | 1.388265 | 21.28% | 4.20% |
| 对侧高频 | 4.749208 | 3.965865 | 3.649390 | 23.16% | 7.98% |
| 水平面严格 ILD MAE | 0.830014 | 0.653461 | 0.644804 | 22.31% | 1.32% |

v3 相对 v2 分别在 `44/44`、`41/44`、`44/44` 和 `27/44` 名被试上改善上述
四项指标。全空间 ERB 与对侧高频取得全员改善；水平面 ILD 的总体均值继续下降，
但被试级改善仅略过半，说明下一轮优化若继续强调 ILD，应采用与最终水平面 HRIR
能量完全一致的损失或执行 v3.1 式微调，而不能只依据 spectral proxy。

配置、完整 CSV、JSON、聚合图、逐被试指标图和 44 人对侧 HRTF 总览位于
`configs/experiments/sonicom_mlp_cnn_q26_v3_strict_validation.json` 与
`results/sonicom_mlp_cnn_q26_v3_strict_validation/`。最终评估正常退出，test
读取数仍为 0。

## 11. MLP+CNN v3.1 水平面严格 ILD 微调与重建评估

v3.1 从锁定的 v3 epoch 10 初始化，保持 100,225 个 MLP 参数冻结，只训练
74,402 个 CNN 参数。训练采样限制在 767 个纯插值方向与零仰角方向的交集，即
72 个水平面方向；严格 ILD 损失由预测幅度与 MCA 相位重建 HRIR 后直接计算，
与最终 MATLAB 水平面 HRIR 能量 ILD 指标对齐。

固定 `ERB / 对侧高频 = 0.75 / 0.25`，以 `3 epoch × 120 step` 比较严格 ILD
权重 `0.5 / 1.0 / 2.0`。三组均未触发 residual、ERB 或高频退化约束；按预先
声明的排序规则选择 `2.0`。正式训练使用 `6 epoch × 500 step`、96 个固定
validation block、每 batch 32 个水平面方向、AdamW、cosine schedule、AMP 和
seed `20260731`：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_mlp_cnn_v3 `
  data/processed/sonicom_residual_q26_v1 `
  artifacts/training/sonicom_mlp_q26_v2_formal_erb075_ild025/best.pt `
  --initial-cnn-checkpoint artifacts/training/sonicom_mlp_cnn_q26_v3_formal_ild075/best.pt `
  --config configs/experiments/sonicom_mlp_cnn_q26_v31_formal.json `
  --run-name sonicom_mlp_cnn_q26_v31_horizontal_formal_ild200 `
  --wandb-mode offline
```

训练用时 `296.56 s`，峰值 CUDA allocated memory 为 `221.96 MiB`，0 个跳步，
最佳点为 epoch 6。固定水平面 validation 上，v3 到 v3.1 的 residual、ERB、
对侧高频和严格 ILD 分别改善 `1.43% / 2.76% / 1.46% / 3.61%`。正式 W&B run
ID 为 `ta1d0lzk`，目前只保存在本地 offline run，未经明确授权不上传。

随后生成 44 名 validation 被试的完整 residual：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.evaluation.predict_sonicom_mlp_cnn_residuals `
  data/processed/sonicom_residual_q26_v1 `
  artifacts/training/sonicom_mlp_cnn_q26_v31_horizontal_formal_ild200/best.pt `
  sonicom_q26_validation_mlp_cnn_v31
```

推理用时 `9.23 s`，输出位于已忽略的 `artifacts/reconstruction/`。经 1 人 smoke
通过后，执行五方法 44 人严格重建：

```powershell
matlab -batch "addpath('matlab'); mcar.evaluate_sonicom_validation_reconstruction([], 'sonicom_mlp_cnn_q26_v31_strict_validation', true, 'sonicom_q26_validation_mlp_cnn_v3', 'sonicom_q26_validation_mlp_cnn_v31')"
```

最终重建均值如下：

| 指标 | MCA (dB) | MLP v2 (dB) | MLP+CNN v3 (dB) | MLP+CNN v3.1 (dB) | v3.1 vs v3 | v3.1 vs v2 |
|---|---:|---:|---:|---:|---:|---:|
| 全空间 ERB | 1.095738 | 0.934826 | 0.887467 | 0.917835 | -3.42% | 1.82% |
| 对侧 25° ERB | 1.763508 | 1.449139 | 1.388265 | 1.398997 | -0.77% | 3.46% |
| 对侧高频 | 4.749208 | 3.965865 | 3.649390 | 3.712955 | -1.74% | 6.38% |
| 水平面严格 ILD MAE | 0.830014 | 0.653461 | 0.644804 | 0.622905 | 3.40% | 4.68% |

表中正值表示误差下降，负值表示误差上升。v3.1 相对 v3 在 29/44 名被试上
改善水平面 ILD，但三项幅度指标分别只有 `0/44`、`18/44` 和 `3/44` 名被试
改善。因此 v3.1 验证了严格水平面 ILD 微调有效，却不是 v3 的全面替代：当前
锁定 v3 为幅度均衡基线，锁定 v3.1 为 ILD 优化基线。完整 CSV、JSON、聚合图、
逐被试图和五方法 44 人 HRTF 总览位于
`results/sonicom_mlp_cnn_q26_v31_strict_validation/`；test 读取数仍为 0。
