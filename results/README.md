# 精选结果

这里仅保存经过口径审核、需要用于报告、论文或项目交接的轻量结果。

- `baselines/mca/`：球谐函数（Spherical Harmonics，SH）、方向均衡空间上采样
  （Spatial Upsampling by Directional Equalization，SUpDEq）+ SH 和幅度校正与
  时间对齐插值（Magnitude-Corrected and Time-Aligned Interpolation，MCA）基线汇总；
- `residual_mlp/`：多层感知机（Multi-Layer Perceptron，MLP）v1/v2 的训练指标与严格重建结果；
- `residual_mlp_cnn/`：v3 的训练指标、严格重建表格和精选图片。

checkpoint、Hierarchical Data Format version 5（HDF5）、MATLAB MAT-file
（MAT）、日志及未筛选图片应留在 `artifacts/`。
