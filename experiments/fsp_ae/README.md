# FSP-AE 横向基线

本目录说明 FSP-AE-Q26 的复现与 SONICOM 适配流程。模型来自 Ito 等人的
[官方 FSP-AE 仓库](https://github.com/ikets/FSP-AE)，上游代码采用
CC BY 4.0；论文 DOI 为
[10.1109/OJSP.2025.3613132](https://doi.org/10.1109/OJSP.2025.3613132)。
本项目没有复制上游仓库或 checkpoint，只在公共源码树中保留带署名的兼容实现。

## 方法边界

- 网络结构、参数名和信号重建遵循官方实现，共 `235,065` 个可训练参数。
- 损失保持为幅度 LSD 加 `2500 × ITD L1`。
- SONICOM 适配使用固定 Q26 输入、`44.1 kHz`、`1024` 点 FFT 和 512 个非直流正频率点。
- 幅度与 ITD 归一化统计只由 262 名 train 被试流式计算；validation 不参与统计，test 默认拒绝访问。
- 为控制显存，训练从每名被试的完整参考方向中抽取目标方向并分块解码。它与论文的多数据集、随机稀疏度训练协议不同，因此结果名称固定为 **FSP-AE-Q26 adaptation**，不能表述为官方论文 checkpoint 在 SONICOM 上的原始结果。

## 数据准备

安装与本机 PyTorch 匹配的 torchaudio 后，默认只准备 train 和 validation：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.prepare_sonicom_fsp_ae `
  --splits train val
```

输出位于忽略目录 `data/processed/sonicom_fsp_ae_q26_v1/`。测试拆分需要同时显式传入
`--splits test` 与 `--allow-test`；开发、调参和 validation 阶段禁止这样做。

## 训练与推理

单步端到端 smoke：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_fsp_ae `
  configs/experiments/sonicom_fsp_ae_q26_smoke.json --device cuda
```

两轮预算与收敛 pilot：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_fsp_ae `
  configs/experiments/sonicom_fsp_ae_q26_budget_pilot.json --device cuda
```

预声明 10/20/40 epoch 快照的正式 train/validation 轨迹：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_fsp_ae `
  configs/experiments/sonicom_fsp_ae_q26_formal_budget.json --device cuda
```

长训练中断后可从最近的原子快照恢复：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.training.train_fsp_ae `
  configs/experiments/sonicom_fsp_ae_q26_formal_budget.json --device cuda `
  --resume artifacts/training/sonicom_fsp_ae_q26_formal_budget_40/checkpoint_epoch_0020.pt
```

validation 推理示例：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.evaluation.predict_sonicom_fsp_ae `
  artifacts/training/sonicom_fsp_ae_q26_smoke/best.pt `
  sonicom_fsp_ae_q26_smoke_validation `
  --split val --subject-limit 1 `
  --target-directions-per-chunk 16 --device cuda
```

将预测接入已有 MATLAB 严格横向评价：

```powershell
matlab -batch "addpath('matlab'); mcar.evaluate_sonicom_interpolation_baselines(1,'sonicom_fsp_ae_q26_smoke_strict',false,'val','sonicom_q26_validation_mlp_cnn_v32_locked_ild075',false,'sonicom_fsp_ae_q26_smoke_validation')"
```

## 当前验证状态

- 官方 HUTUBS checkpoint 的幅度、ITD 和 HRIR 重建逐值比较最大绝对误差均为 0。
- 完整缓存为 262 train / 44 validation / 0 test；train-only 幅度均值/总体标准差为
  `-26.081498 / 18.320320 dB`，ITD 为 `1.715133e-6 / 3.793023e-4 s`。
- 两轮 pilot 的 validation LSD 为 `5.965827 → 5.177868 dB`，复合损失为
  `6.452062 → 5.474776`；RTX 5060 用时 `14.236 s`，峰值 CUDA allocated memory
  为 `526.671 MiB`。
- 正式 40 epoch 轨迹在 epoch 40 取得最低 validation 复合损失 `3.026746`；
  44 人严格指标为 `1.160 / 1.848 / 3.104 / 0.598 dB`。完整解释和精选表位于
  `reports/FSP_AE_Q26_VALIDATION_REPORT.md` 与
  `results/sonicom_fsp_ae_q26_formal_validation/`。
- 单步 checkpoint 的单人严格指标只用于验证接口，不能用于模型比较或论文结论。
  正式 checkpoint 已在 validation 上锁定；不得重用已经完成一次性评价的 SONICOM
  test 进行调参或选模。
