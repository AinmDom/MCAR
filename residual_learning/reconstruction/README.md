# Residual magnitude reconstruction

该目录保存 test 被试的复数 MCA/reference 缓存、GPU residual 预测、回填后的
论文指标与可视化结果。

完整流程分为三步：

```powershell
matlab -batch "addpath('residual_learning/matlab'); prepare_test_reconstruction_inputs(4, 'mlp_n03_v1')"

D:\miniconda3\envs\ml\python.exe residual_learning/python/predict_reconstructed_residuals.py `
  residual_learning/reconstruction/mlp_n03_v1 `
  residual_learning/runs/mlp_n03_v1/best.pt `
  residual_learning/data/hutubs_residual_v1_n03/training_statistics.json

matlab -batch "addpath('residual_learning/matlab'); evaluate_test_reconstruction('mlp_n03_v1')"
```

v2 复用 v1 已生成的复数 cache 和模型输入：

```powershell
D:\miniconda3\envs\ml\python.exe residual_learning/python/predict_reconstructed_residuals.py `
  residual_learning/reconstruction/mlp_n03_v2 `
  residual_learning/runs/mlp_n03_v2/best.pt `
  residual_learning/data/hutubs_residual_v1_n03/training_statistics.json `
  --input-root residual_learning/reconstruction/mlp_n03_v1

matlab -batch "addpath('residual_learning/matlab'); evaluate_test_reconstruction('mlp_n03_v2', 'mlp_n03_v1')"
matlab -batch "addpath('residual_learning/matlab'); plot_v1_v2_reconstruction_comparison('mlp_n03_v1', 'mlp_n03_v2')"
```

复数缓存、模型输入、预测 HDF5 和 MAT 汇总体积较大，不提交 Git。仓库仅保留
v1/v2 的最终 CSV、逐被试对比图、12 人 HRTF 总览和指标总览。
