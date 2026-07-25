# Residual MLP v1/v2

v1 学习逐方向、逐频点的 MCA log-magnitude residual；v2 在同一模型上加入
ERB、对侧高频和双耳 ILD proxy 损失。两者共享 `src/mcar/data.py` 和
`src/mcar/models/residual_mlp.py`。

数据划分固定在 `configs/data/subject_split_v1.csv`，锁定参数位于
`configs/experiments/mlp_v1.json` 和 `mlp_v2.json`。

训练：

```powershell
.venv/Scripts/python -m mcar.training.train_mlp_v1 `
  data/processed/hutubs_residual_v1_n03 `
  --run-name mlp_n03_v1

.venv/Scripts/python -m mcar.training.train_mlp_v2 `
  data/processed/hutubs_residual_v1_n03 `
  artifacts/training/mlp_n03_v1/best.pt `
  --run-name mlp_n03_v2
```

checkpoint 和日志位于 `artifacts/training/`；精选训练指标与严格评估结果位于
`results/residual_mlp/`。完整结论见 `reports/MLP_V1_REPORT.md` 和
`reports/MLP_V2_REPORT.md`。
