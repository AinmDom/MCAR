# 本地运行产物

`artifacts/` 保存可由源码和配置重新生成、但不应提交到 Git 的文件：

- `training/`：checkpoint、Weights & Biases（W&B）缓存和训练日志；
- `reconstruction/`：复数缓存、模型输入、预测 Hierarchical Data Format
  version 5（HDF5）和 MATLAB 中间结果；
- `mca_reproduction/`：幅度校正与时间对齐插值（Magnitude-Corrected and
  Time-Aligned Interpolation，MCA）批处理检查点和逐被试结果；
- `figures/`：尚未筛选为正式结果的生成图；
- `logs/`：本机运行日志。

需要长期保留和提交的逗号分隔值（Comma-Separated Values，CSV）、
JavaScript 对象表示法（JavaScript Object Notation，JSON）与论文图片应整理到
`results/`。
