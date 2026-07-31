# 配置

- `data/subject_split_v1.csv`：按被试固定的 train/validation/test 划分；
- `data/sonicom_sparse_grid_q26_v1.csv`：从 SONICOM 实测方向中选择的 26 点
  左右对称稀疏输入网格；
- `data/sonicom_reference_grid_v1.csv`：793 个实测方向、球面面积权重、稀疏
  输入 mask 和纯插值评估 mask；
- `data/sonicom_subject_split_v1.csv`：按自由场 EQ 文件分层的
  `262 train / 44 validation / 44 test` 固定划分；
- `data/sonicom_preparation_report_v1.json`：网格兼容性、Q26 覆盖与球谐条件数、
  分层配额和人口统计缺失情况；
- `experiments/sonicom_q26_residual_v1.json`：根据非 test pilot 锁定的
  SONICOM MCA residual 正式导出参数；
- `experiments/*.json`：从正式运行记录中提取、去除本机绝对路径后的锁定参数。

当前训练命令行界面（Command-Line Interface，CLI）仍以命令行参数为准；
JavaScript 对象表示法（JavaScript Object Notation，JSON）用于审计和复现实验参数，
不直接驱动训练程序。
