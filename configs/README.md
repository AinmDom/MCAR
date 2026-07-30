# 配置

- `data/subject_split_v1.csv`：按被试固定的 train/validation/test 划分；
- `experiments/*.json`：从正式运行记录中提取、去除本机绝对路径后的锁定参数。

当前训练命令行界面（Command-Line Interface，CLI）仍以命令行参数为准；
JavaScript 对象表示法（JavaScript Object Notation，JSON）用于审计和复现实验参数，
不直接驱动训练程序。
