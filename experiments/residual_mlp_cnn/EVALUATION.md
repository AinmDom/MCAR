# 多层感知机（Multi-Layer Perceptron，MLP）+ 卷积神经网络（Convolutional Neural Network，CNN）v3 / v3.1 严格重建

```powershell
.venv/Scripts/python -m mcar.evaluation.predict_mlp_cnn_reconstruction `
  artifacts/reconstruction/mlp_cnn_n03_v3_cnn_only `
  artifacts/reconstruction/mlp_n03_v1 `
  artifacts/training/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt `
  data/processed/hutubs_residual_v1_n03/training_statistics.json

matlab -batch "addpath('matlab'); mcar.evaluate_mlp_cnn_v3_reconstruction; mcar.plot_v1_v2_v3_reconstruction_comparison"
```

精选结果位于 `results/residual_mlp_cnn/mlp_cnn_n03_v3/evaluation/`。

v3.1 在 Python 完整 residual 评估时增加 `--strict-ild`，要求数据集含
Hierarchical Data Format version 5（HDF5）schema 1.1 的原幅度校正与时间对齐
插值（Magnitude-Corrected and Time-Aligned Interpolation，MCA）相位、
频带外复频谱和参考头相关脉冲响应（Head-Related Impulse Response，HRIR）
耳间电平差（Interaural Level Difference，ILD）：

```powershell
.venv/Scripts/python -m mcar.evaluation.evaluate_mlp_cnn_v3 `
  data/processed/hutubs_residual_v1_n03 `
  artifacts/training/mlp_cnn_n03_v31_finetune_v3_strict_ild_wandb_online/best.pt `
  --split val `
  --strict-ild `
  --directions-per-block 32

.venv/Scripts/python -m mcar.evaluation.predict_mlp_cnn_reconstruction `
  artifacts/reconstruction/mlp_cnn_n03_v31_finetune_v3 `
  artifacts/reconstruction/mlp_n03_v1 `
  artifacts/training/mlp_cnn_n03_v31_finetune_v3_strict_ild_wandb_online/best.pt `
  data/processed/hutubs_residual_v1_n03/training_statistics.json

matlab -batch "addpath('matlab'); mcar.evaluate_test_reconstruction('mlp_cnn_n03_v31_finetune_v3','mlp_n03_v1','artifacts/reconstruction/mlp_cnn_n03_v31_finetune_v3','artifacts/reconstruction/mlp_n03_v1','MCA+ResidualMLPCNNv31','MCA + residual MLP-CNN v3.1'); mcar.plot_v1_v2_v3_v31_reconstruction_comparison"
```

精选的 v3.1 逗号分隔值（Comma-Separated Values，CSV）、JavaScript 对象
表示法（JavaScript Object Notation，JSON）和 16 张测试结果图位于
`results/residual_mlp_cnn/mlp_cnn_n03_v31/evaluation/`。
