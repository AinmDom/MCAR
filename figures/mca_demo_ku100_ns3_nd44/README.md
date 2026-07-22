# KU100 MCA demo 图表

此目录用于存放 KU100、Lebedev `Ns = 3`、`Nd = 44` 的 MCA baseline 导出结果：8
个 PNG 图、对应的 MATLAB FIG 文件，以及 `mca_demo_metrics.mat`。这些均为可再
生实验产物，不提交到 Git。

## 生成方式

1. 按 [`../../SUpDEq-master/README.project.md`](../../SUpDEq-master/README.project.md)
   准备本地 SUpDEq 工具包和其 KU100 示例数据。
2. 从项目根目录运行：

   ```powershell
   matlab -wait -batch "run('scripts/run_mca_demo_export.m')"
   ```

脚本会覆盖同名导出文件；没有单独下载包。

该 README 是此目录中唯一应提交到本项目 Git 仓库的文件。