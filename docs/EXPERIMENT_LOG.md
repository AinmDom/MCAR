# 项目实验日志

本文件是项目唯一的进展与实验记录。以日期与时间作为二级标题，按时间倒序记录每个阶段性改动、实验、结论、阻塞项和下一步。跨电脑继续项目时，先阅读本文件。

## 记录模板

```md
## YYYY-MM-DD HH:mm（UTC±HH:mm）：工作或实验名称

- 改动：
- 目的：
- 代码 / 命令：
- 数据集与版本：
- 稀疏网格与采样点数：
- 关键参数：
- 指标结果：
- 输出文件位置：
- 阻塞项：
- 结论与下一步：
```

## 2026-07-22 21:41（UTC+08:00）：MCA demo 可复现产物与 Git 管理策略

- 改动：在 `.gitignore` 中忽略 `figures/mca_demo_ku100_ns3_nd44/`；将该目录中已跟踪的 17 个生成产物从 Git 索引移除，本地文件保留。
- 目的：只提交可复现实验代码和实验记录，避免将 MATLAB 生成的图片、FIG 与大型指标 MAT 文件加入 Git。
- 代码 / 命令：复现实验只需运行项目脚本 `scripts/run_mca_demo_export.m`；该脚本会自动调用本地第三方工具包中的 `SUpDEq-master/supdeq_start.m` 和 `SUpDEq-master/supdeq_demo_MCA.m`。Windows 命令为：`matlab -wait -batch "run('D:\\course\\CUC_2\\CSMT\\MCA_residual_project\\scripts\\run_mca_demo_export.m')"`。
- 数据集与版本：KU100；依赖本机 `SUpDEq-master/` 中的 `HRIR_L2702.sofa` 及配套工具箱。
- 稀疏网格与采样点数：Lebedev `Ns = 3`；目标 Lebedev `Nd = 44`（2702 个方向）。
- 关键参数：`mc = inf`；头半径 `0.0875 m`。
- 指标结果：脚本预期生成 8 个 `.png`、8 个 `.fig` 和 `mca_demo_metrics.mat`，共 17 个产物。
- 输出文件位置：`../figures/mca_demo_ku100_ns3_nd44/`。该目录不提交 Git；下方 21:20 记录中的相对图表链接仅在本机运行导出脚本后有效。
- 阻塞项：跨电脑复现前需自行准备被忽略的 `SUpDEq-master/` 本地第三方依赖和数据。
- 结论与下一步：后续不对该目录执行 `git add` 或 `git push`；提交 `scripts/run_mca_demo_export.m`、`.gitignore` 与本实验日志即可。需要查看或重新计算图表时，在本机运行上述导出脚本。

## 2026-07-22 21:20（UTC+08:00）：MCA demo 图表导出与归档

- 改动：运行 `SUpDEq-master/supdeq_demo_MCA.m`，通过 `scripts/run_mca_demo_export.m` 保存全部 8 张图；为每张图同时保留 PNG 与可编辑 MATLAB FIG 版本。
- 目的：归档 KU100 基线实验的 HRIR、LSD 与 ERB-band magnitude-error 可视化结果，供后续 residual MLP 对比使用。
- 代码 / 命令：`matlab -wait -batch "run('D:\\course\\CUC_2\\CSMT\\MCA_residual_project\\scripts\\run_mca_demo_export.m')"`。
- 数据集与版本：KU100，由 SUpDEq 本地数据 `HRIR_L2702.sofa` 构建。
- 稀疏网格与采样点数：Lebedev `Ns = 3`；目标 Lebedev `Nd = 44`（2702 个方向）。
- 关键参数：`mc = inf`；头半径 `0.0875 m`；MCA 在空间混叠频率 `fA = 1871.66 Hz` 以上执行最小相位幅度校正。
- 指标结果：成功生成 SH、SUpDEq + SH、MCA 的左耳全方向平均 LSD 与 ERB-band magnitude-error 曲线，以及正前方和对侧耳 HRIR 对比图。对侧耳约 10 kHz 以上的独立定量改善仍待计算。
- 输出文件位置（均为相对于本日志的路径）：
  - [正前方 conventional vs MCA HRIR](../figures/mca_demo_ku100_ns3_nd44/01_frontal_conventional_vs_mca_hrir.png)
  - [对侧耳 SH vs MCA HRIR](../figures/mca_demo_ku100_ns3_nd44/02_contralateral_sh_vs_mca_hrir.png)
  - [对侧耳 conventional vs MCA HRIR](../figures/mca_demo_ku100_ns3_nd44/03_contralateral_conventional_vs_mca_hrir.png)
  - [对侧耳 SH vs reference HRIR](../figures/mca_demo_ku100_ns3_nd44/04_contralateral_sh_vs_reference_hrir.png)
  - [对侧耳 conventional vs reference HRIR](../figures/mca_demo_ku100_ns3_nd44/05_contralateral_conventional_vs_reference_hrir.png)
  - [对侧耳 MCA vs reference HRIR](../figures/mca_demo_ku100_ns3_nd44/06_contralateral_mca_vs_reference_hrir.png)
  - [SH、SUpDEq + SH、MCA 的 LSD 曲线](../figures/mca_demo_ku100_ns3_nd44/07_lsd_left_ear.png)
  - [SH、SUpDEq + SH、MCA 的 ERB-band magnitude-error 曲线](../figures/mca_demo_ku100_ns3_nd44/08_erb_magnitude_error_left_ear.png)
  - 对应 `.fig` 文件和 `mca_demo_metrics.mat` 位于 `../figures/mca_demo_ku100_ns3_nd44/`；大型 MAT 文件不提交 Git。
- 阻塞项：无运行阻塞；尚未完成对侧耳 `>10 kHz` 区域的独立数值统计。
- 结论与下一步：MCA demo 已复现并完成图表归档；下一步从指标数据中计算 conventional 与 MCA 在对侧耳高频区域的误差差值和相对改善百分比。

## 2026-07-22 21:01（UTC+08:00）：KU100 的 SH、SUpDEq + SH 与 MCA 基线评估记录

- 改动：登记三组基线（SH、SUpDEq + SH、MCA）的 LSD 曲线与 ERB-band magnitude-error 曲线；登记 MCA 相对于 conventional（SUpDEq + SH）在对侧耳、约 10 kHz 以上区域的改善分析。
- 目的：建立残差 MLP 之前的可复现 MCA baseline，并突出其预期最有价值的高频对侧耳改善区域。
- 代码 / 命令：本次未重新执行 MATLAB；已发现现有导出特征文件。
- 数据集与版本：KU100；具体数据源版本尚待补充。
- 稀疏网格与采样点数：源网格为 Lebedev `Ns = 3`；目标网格为 Lebedev `Nd = 44`。
- 关键参数：SH only：`ppMethod = 'None'`、`ipMethod = 'SH'`、`mc = nan`；conventional：`ppMethod = 'SUpDEq'`、`ipMethod = 'SH'`、`mc = nan`；MCA：`ppMethod = 'SUpDEq'`、`ipMethod = 'SH'`、`mc = inf`；头半径 `0.0875 m`。
- 指标结果：需保存/报告三组 LSD 曲线与 ERB-band magnitude-error 曲线；需按对侧耳和频率 `>~10 kHz` 切片，量化 MCA 相对 conventional 的误差改善。当前未提供曲线数值、汇总统计或图像，因此不在此记录未核验的数值性结论。
- 输出文件位置：已有特征导出 `outputs/mca_demo_ku100_N3/mca_demo_features.mat`（约 281 MB，未跟踪）；LSD、ERB 曲线及对侧高频统计的图片/数据文件路径待补充。大型 MAT 与结果图不直接提交 Git，除非另行确认确有必要保留。
- 阻塞项：缺少实际绘图文件、指标数值、频率定义和对侧耳判定规则；尚无法核验“10 kHz 以上改善”的幅度与统计方式。
- 结论与下一步：从导出特征计算并保存三组基线的 LSD 与 ERB 曲线；另外输出 conventional 与 MCA 在对侧、`>10 kHz` 区域的均值/中位数误差和差值（或相对改善百分比），在日志中补齐图表路径和数值结果。

## 2026-07-22：项目初始化与协作规范

- 改动：整理 README；建立 `AGENTS.md` 与本日志，统一由本文件记录项目进展。
- 目的：明确 MCA + 轻量残差网络研究路线，并支持两台电脑之间无缝接续。
- 代码 / 命令：未运行实验。
- 数据集与版本：尚未确定；初步计划使用 SUpDEq 工具包示例数据，正式实验后续考虑 HUTUBS simulated HRTFs。
- 稀疏网格与采样点数：尚未运行；优先使用 Lebedev 网格。
- 关键参数：不适用。
- 指标结果：不适用。
- 输出文件位置：不适用。
- 环境说明：`SUpDEq-master/` 是本地忽略的第三方 MATLAB 工具包，每台电脑需自行准备，不提交到 Git。
- 阻塞项：尚未记录 MATLAB 与 SUpDEq demo 在当前电脑上的实际运行情况；尚未确定正式训练数据集和存储位置。
- 结论与下一步：确认 MATLAB 可用后，在 `SUpDEq-master` 目录运行 `supdeq_start` 与 `supdeq_demo_MCA`；在本日志记录数据集、Lebedev 网格、关键输出，并导出 `interpHRTF_sh`、`interpHRTF_con`、`interpHRTF_mca` 与 `refHRTF`。
