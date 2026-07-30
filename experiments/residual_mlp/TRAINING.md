# 多层感知机（Multi-Layer Perceptron，MLP）训练产物

训练程序将 checkpoint、配置快照和日志写入
`artifacts/training/<run-name>/`。该目录不提交 Git。

完成 validation/test 审核后，需要长期保留的 `history.csv`、指标 JavaScript
对象表示法（JavaScript Object Notation，JSON）文件和
报告整理到 `results/residual_mlp/<run-name>/training/`。
