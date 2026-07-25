# Residual MLP + CNN v3

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
