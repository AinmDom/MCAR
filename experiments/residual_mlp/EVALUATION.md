# 多层感知机（Multi-Layer Perceptron，MLP）严格头相关传输函数（Head-Related Transfer Function，HRTF）重建

```powershell
matlab -batch "addpath('matlab'); mcar.prepare_test_reconstruction_inputs(4, 'mlp_n03_v1')"

.venv/Scripts/python -m mcar.evaluation.predict_reconstructed_residuals `
  artifacts/reconstruction/mlp_n03_v1 `
  artifacts/training/mlp_n03_v1/best.pt `
  data/processed/hutubs_residual_v1_n03/training_statistics.json

matlab -batch "addpath('matlab'); mcar.evaluate_test_reconstruction('mlp_n03_v1')"
```

中间 Hierarchical Data Format version 5（HDF5）、MATLAB MAT-file（MAT）和
预测 residual 位于 `artifacts/reconstruction/`；审核后的逗号分隔值
（Comma-Separated Values，CSV）和便携式网络图形（Portable Network
Graphics，PNG）位于 `results/residual_mlp/`。
