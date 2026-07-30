# 多层感知机-卷积神经网络（Multi-Layer Perceptron–Convolutional Neural Network，MLP-CNN）v3.1 selected results

这是从原 v3 epoch 9 继续进行严格头相关脉冲响应-耳间电平差（Head-Related
Impulse Response–Interaural Level Difference，HRIR-ILD）微调后、在 validation
上锁定的 epoch 3 结果。这里仅保存适合 Git 跟踪的逗号分隔值
（Comma-Separated Values，CSV）、JavaScript 对象表示法（JavaScript Object
Notation，JSON）和便携式网络图形（Portable Network Graphics，PNG）；
checkpoint、预测 Hierarchical Data Format version 5（HDF5）、MATLAB MAT-file
（MAT）与 Weights & Biases（W&B）本地缓存位于被忽略的 `artifacts/`。

- `training/`：配置、训练历史、训练报告以及完整 validation/test residual 指标。
- `evaluation/`：12 名 test 被试的严格等效矩形带宽（Equivalent Rectangular
  Bandwidth，ERB）、高频、ILD 与重建质量结果。
- `evaluation/figures/`：逐被试头相关传输函数（Head-Related Transfer
  Function，HRTF）图、12 被试总览和版本对比图。

完整方法、模型选择过程和结论见
`reports/MLP_CNN_V31_REPORT.md`。正式 W&B run：
<https://wandb.ai/luyoung/mcar-mlp-cnn-v31/runs/yhl3z70n>。
