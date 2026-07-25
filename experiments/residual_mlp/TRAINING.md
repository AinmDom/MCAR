# MLP 训练产物

训练程序将 checkpoint、配置快照和日志写入
`artifacts/training/<run-name>/`。该目录不提交 Git。

完成 validation/test 审核后，需要长期保留的 `history.csv`、指标 JSON 和
报告整理到 `results/residual_mlp/<run-name>/training/`。
