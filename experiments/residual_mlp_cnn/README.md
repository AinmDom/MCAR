# Residual 多层感知机（Multi-Layer Perceptron，MLP）+ 卷积神经网络（Convolutional Neural Network，CNN）v3 / v3.1

v3 冻结 v2 ResidualMLP，在完整双耳频谱上训练轻量一维 CNN。公共模型位于
`src/mcar/models/residual_mlp_cnn.py`，锁定参数位于
`configs/experiments/mlp_cnn_v3.json`。

```powershell
.venv/Scripts/python -m mcar.training.train_mlp_cnn_v3 `
  data/processed/hutubs_residual_v1_n03 `
  artifacts/training/mlp_n03_v2/best.pt `
  --run-name mlp_cnn_n03_v3 `
  --wandb-mode online
```

完整实验结论见 `reports/MLP_CNN_V3_REPORT.md`。

v3.1 不改变网络结构，而是从锁定的 v3 checkpoint 继续微调 CNN，并把
训练中的耳间电平差（Interaural Level Difference，ILD）项改为与最终 MATLAB
重建一致的可微头相关脉冲响应（Head-Related Impulse Response，HRIR）能量 ILD：

```powershell
.venv/Scripts/python -m mcar.training.train_mlp_cnn_v3 `
  data/processed/hutubs_residual_v1_n03 `
  artifacts/training/mlp_n03_v2/best.pt `
  --initial-cnn-checkpoint artifacts/training/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt `
  --config configs/experiments/mlp_cnn_v31_finetune_v3.json `
  --run-name mlp_cnn_n03_v31_finetune_v3_strict_ild `
  --wandb-mode online
```

训练器同时保存按 validation 总损失选择的 `best.pt`，以及按采样 validation
严格 ILD 选择的 `best_strict_ild.pt`。正式 v3.1 在完整 validation 上预先
锁定 `best.pt`（epoch 3），随后只解封一次 test。完整结论见
`reports/MLP_CNN_V31_REPORT.md`。
