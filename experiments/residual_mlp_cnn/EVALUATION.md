# MLP + CNN v3 严格重建

```powershell
.venv/Scripts/python -m mcar.evaluation.predict_mlp_cnn_reconstruction `
  artifacts/reconstruction/mlp_cnn_n03_v3_cnn_only `
  artifacts/reconstruction/mlp_n03_v1 `
  artifacts/training/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt `
  data/processed/hutubs_residual_v1_n03/training_statistics.json

matlab -batch "addpath('matlab'); mcar.evaluate_mlp_cnn_v3_reconstruction; mcar.plot_v1_v2_v3_reconstruction_comparison"
```

精选结果位于 `results/residual_mlp_cnn/mlp_cnn_n03_v3/evaluation/`。
