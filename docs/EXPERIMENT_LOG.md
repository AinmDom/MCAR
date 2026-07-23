# 项目实验日志

本文件是项目唯一的进展与实验记录。以日期作为二级标题，按时间倒序记录实验过程、结果、结论、阻塞项和下一步。跨电脑继续项目时，先阅读本文件。

## 记录模板

```md
## YYYY-MM-DD：工作或实验名称

- 实验目标：
- 数据集与版本：
- 稀疏网格与采样点数：
- 方法与关键参数：
- 实验过程：
- 指标结果：
- 输出文件位置：
- 阻塞项：
- 结论与下一步：
```

## 2026-07-23：HUTUBS simulated 单被试 MCA 复现检查点

- 实验目标：验证 HUTUBS simulated SOFA 与 MCA 论文的 Lebedev 网格、个体头半径和 MCA 基线参数可在本机 SUpDEq 环境中一致运行，再进入 96 位受试者的批处理。
- 数据集与版本：HUTUBS simulated `pp91_HRIRs_simulated.sofa`，本机 `data/HRTF/hutubs`；SOFA 为 `SimpleFreeFieldHRIR`，数据形状为 `1730 x 2 x 256`，采样率 `44.1 kHz`。
- 稀疏网格与采样点数：源参考为 Lebedev `N=35`（1730 点）；稀疏输入为 Lebedev `N=3`（26 点）；误差评估目标为 Fliege `N=29`（900 点）。已逐行验证 SOFA 方向与 SUpDEq Lebedev `N=35` 一致，最大方位/余纬差分别为 `5.68e-14` / `2.84e-14` 度。
- 方法与关键参数：SH only（`'None'`, `'SH'`, `mc=nan`）、conventional（`'SUpDEq'`, `'SH'`, `mc=nan`）与 MCA（`'SUpDEq'`, `'SH'`, `mc=inf`）。MCA 使用默认最小相位校正、空间混叠频率以下限制以及向下 1/3 octave fade。根据 HUTUBS `x1,x2,x3` 人体测量值和 Algazi 公式，pp91 的头半径为 `0.088883 m`（`8.8883 cm`），与论文示例的约 `8.89 cm` 一致。
- 实验过程：新增本地忽略的 `reproduce/run_hutubs_mca_p91_n3.m`，在 MATLAB R2025b Update 2 中实际执行 `matlab -batch "addpath(fullfile(pwd,'reproduce')); run_hutubs_mca_p91_n3"`；命令成功结束，退出码为 0，耗时约 50.5 秒。
- 指标结果：ERB 绝对幅度误差（方向加权、ERB 频带等权）中，左耳全空间由 conventional `1.0890 dB` 降至 MCA `0.7611 dB`，左耳对侧 25 度区域由 `2.7124 dB` 降至 `1.8014 dB`；右耳全空间由 `0.9645 dB` 降至 `0.7340 dB`，右耳对侧 25 度区域由 `2.5212 dB` 降至 `1.6867 dB`。输出曲线显示 MCA 在空间混叠频率约 `1842.54 Hz` 以上相较 conventional 进一步降低误差。
- 输出文件位置：`reproduce/pp91_n3/pp91_n3_erb_summary.csv`、`pp91_n3_results.mat`、`pp91_n3_left_erb_error.png`、`pp91_n3_left_erb_error.fig`；整个 `reproduce/` 当前已由 `.gitignore` 忽略，不提交。
- 阻塞项：初始受限环境中 MATLAB 批处理启动出现 `File system inconsistency`；以本机正常权限启动后已成功运行，当前无实验阻塞。
- 结论与下一步：单被试 `pp91, N=3` 已通过网格、头半径、运行链路与误差趋势验证。下一步将脚本泛化至 `pp1`–`pp96` 与稀疏阶数 `N=1`–`10`，并补全论文的跨受试者均值/标准差、空间误差、ILD 与 ITD 汇总。

## 2026-07-23：下载 HUTUBS HRTF 数据集

- 实验目标：获取完整 HUTUBS 数据库，作为后续 MCA 基线、跨个体 residual 学习与评估的主要数据集。
- 数据集与版本：SOFA Acoustics HUTUBS 数据库目录（访问日期 `2026-07-23`）；源地址：`https://sofacoustics.org/data/database/hutubs/`。目录文件的网页标注修改日期为 `2017-06-23` 至 `2020-01-31`。
- 数据文件：共 `253` 个文件，包括 96 位受试者的 `192` 个 measured/simulated HRIR SOFA 文件、`58` 个 3D 头部网格 PLY 文件、`2` 个 PDF 文档和 `1` 个 CSV 人体测量文件。
- 实验过程：从目录索引自动提取全部文件链接，再使用 `curl 8.21.0` 下载；启用失败重试、断点续传和最多 `8` 个并行传输。由于本机 Windows Schannel 无法访问证书吊销检查服务，下载时使用 `--ssl-no-revoke` 跳过吊销状态检查，TLS 加密连接仍保留。
- 完整性检查：实得 `253/253` 个文件，无零字节文件；`pp1`–`pp96` 的 measured/simulated SOFA 文件全部成对齐全，无缺失或意外 SOFA 文件；全部 SOFA、PDF 和 PLY 文件的格式签名检查通过。文件总大小为 `1,442,534,622` 字节（`1,375.71 MiB`，`1.343 GiB`）。
- 输出文件位置：本机 `D:\cuc\CSMT\MCAR\data\HRTF\hutubs`。数据文件已移动到 Git 仓库内的忽略目录，不提交到 Git；目录说明见 `data/HRTF/README.md`。
- 阻塞项：无。
- 结论与下一步：完整 HUTUBS 数据集已可用。后续优先读取 simulated SOFA 复现 MCA 设置，并解析 SOFA 元数据与人体测量表，固定跨个体训练/验证/测试划分。

## 2026-07-23：下载 AXD HRTF 数据集

- 实验目标：获取 AXD 个体化 HRTF SOFA 文件，作为后续跨个体 MCA residual 学习与评估的数据集。
- 数据集与版本：SOFA Acoustics AXD 数据库目录（访问日期 `2026-07-23`）；源地址：`https://sofacoustics.org/data/database/axd/`。目录文件的网页标注修改日期为 `2022-06-28` 至 `2023-03-08`。
- 数据文件：共 `140` 个 `.sofa` 文件，包括 `p0001.sofa`–`p0040.sofa` 和 `p0101.sofa`–`p0200.sofa`。
- 实验过程：使用 `curl 8.21.0` 从源目录下载；启用失败重试、断点续传和最多 `8` 个并行传输。由于本机 Windows Schannel 无法访问证书吊销检查服务，下载时使用 `--ssl-no-revoke` 跳过吊销状态检查，TLS 加密连接仍保留。
- 完整性检查：实得 `140/140` 个预期文件，无缺失、无意外文件、无零字节文件；所有文件均具有 NetCDF4/HDF5 文件签名。SOFA 文件总大小为 `413,764,409` 字节（`394.60 MiB`）。
- 输出文件位置：本机 `D:\cuc\CSMT\MCAR\data\HRTF\axd`。数据文件已移动到 Git 仓库内的忽略目录，不提交到 Git；目录说明见 `data/HRTF/README.md`。
- 阻塞项：无。
- 结论与下一步：AXD 数据集已可用。后续在数据预处理阶段读取 SOFA 元数据，核对各个体采样方向、采样率、IR 长度及坐标约定，再确定跨个体训练/验证/测试划分。

## 2026-07-23：KU100 对侧耳高频误差定量统计

- 实验目标：将 MCA demo 中“对侧耳 10 kHz 以上误差降低”的可视化趋势转化为可复现、可用于论文的双耳定量结果。
- 数据集与版本：KU100，SUpDEq 本地 `HRIR_L2702.sofa`；MATLAB R2025b。
- 稀疏网格与采样点数：源网格为 Lebedev `Ns = 3`；目标网格为 Lebedev `Nd = 44`，共 2702 个方向。
- 方法与关键参数：
  - 基线参数与 2026-07-22 实验一致：SH、SUpDEq + SH（conventional）和 MCA（`mc = inf`），头半径 `0.0875 m`。
  - SUpDEq 坐标为 `0°=前、90°=左、180°=后、270°=右`。根据球面坐标的笛卡尔横向坐标判定半球：左耳对侧为右半球，右耳对侧为左半球；排除横向坐标为零的正中面方向。每耳纳入 1321 个方向。
  - 高频范围严格定义为 `f > 10 kHz` 且 `f <= fs/2`，本次包含 150 个频点。
  - 单样本误差为估计与 reference 的绝对 log-magnitude 差，单位为 dB；幅度下限为 `-200 dB`。均值在方向维使用 Lebedev 权重、频率维等权；中位数为全部方向-频率样本的非加权中位数。
- 实验过程：
  1. 新增 `scripts/analyze_contralateral_high_frequency.m`，独立复现三种插值结果并读取 dense reference。
  2. 分别计算左耳对侧、右耳对侧和双耳汇总的 SH、conventional、MCA 高频绝对 log-magnitude error。
  3. 导出完整统计 MAT、两张 CSV 表以及双耳频率误差曲线的 PNG/FIG。
  4. 实际执行命令：`matlab -wait -batch "addpath('D:\cuc\CSMT\MCAR\scripts'); analyze_contralateral_high_frequency"`。命令成功结束，退出码为 0。
- 指标结果：
  - 左耳对侧：conventional 加权平均误差为 `5.5850 dB`，MCA 为 `4.6349 dB`，降低 `0.9500 dB`，相对改善 `17.01%`；中位数由 `4.0917 dB` 降至 `3.2449 dB`，相对改善 `20.69%`。
  - 右耳对侧：conventional 加权平均误差为 `5.1573 dB`，MCA 为 `4.6283 dB`，降低 `0.5291 dB`，相对改善 `10.26%`；中位数由 `3.7299 dB` 降至 `3.2801 dB`，相对改善 `12.06%`。
  - 双耳汇总：SH、conventional、MCA 的加权平均误差分别为 `7.5781 dB`、`5.3711 dB`、`4.6316 dB`。MCA 相对 conventional 降低 `0.7395 dB`，相对改善 `13.77%`；中位数由 `3.9044 dB` 降至 `3.2626 dB`，降低 `0.6417 dB`，相对改善 `16.44%`。
- 输出文件位置：`../figures/mca_demo_ku100_ns3_nd44/contralateral_high_frequency/`。该目录是脚本生成且被 Git 忽略的本地实验产物，包含：
  - `contralateral_high_frequency_summary.csv`
  - `contralateral_high_frequency_improvement.csv`
  - `contralateral_high_frequency_statistics.mat`
  - `contralateral_high_frequency_error.png`
  - `contralateral_high_frequency_error.fig`
- 阻塞项：本项无运行阻塞。当前结果仅来自 KU100 单一人头模型与 `Ns = 3` 稀疏网格，尚不能说明跨个体或跨稀疏度的稳定性。
- 结论与下一步：MCA 在 KU100 双耳对侧 `>10 kHz` 区域相对 conventional 确有稳定的定量改善，解决了上一阶段“只有可视化趋势”的阻塞项。下一步固定 residual 数据集字段与训练/验证/测试划分，并实现训练特征导出脚本。

## 2026-07-22：KU100 的 MCA 基线复现与图表导出

- 实验目标：复现并比较 SH、SUpDEq + SH 和 MCA 三种 HRTF 空间插值基线，为后续 MCA residual MLP 提供输入、目标和评价基准。
- 数据集与版本：KU100，使用 SUpDEq 本地数据 `HRIR_L2702.sofa` 构建稀疏集与参考集。
- 稀疏网格与采样点数：源网格为 Lebedev `Ns = 3`；目标网格为 Lebedev `Nd = 44`，共 2702 个方向。
- 方法与关键参数：
  - SH：`ppMethod = 'None'`、`ipMethod = 'SH'`、`mc = nan`。
  - SUpDEq + SH（conventional）：`ppMethod = 'SUpDEq'`、`ipMethod = 'SH'`、`mc = nan`。
  - MCA：`ppMethod = 'SUpDEq'`、`ipMethod = 'SH'`、`mc = inf`。
  - 头半径为 `0.0875 m`；本次 MCA 使用最小相位幅度校正，并在空间混叠频率 `fA = 1871.66 Hz` 以上执行校正。
- 实验过程：
  1. 使用 `supdeq_getSparseDataset` 从 KU100 参考数据生成 `Ns = 3` 的稀疏 HRTF 集，并创建 `Nd = 44` 的目标 Lebedev 网格。
  2. 分别执行 SH、SUpDEq + SH 和 MCA 插值，得到 `interpHRTF_sh`、`interpHRTF_con` 和 `interpHRTF_mca`。
  3. 在目标网格上取得 KU100 reference HRTF，并转换得到 reference HRIR。
  4. 比较正前方与对侧左耳的 HRIR，重点检查 MCA 对对侧高频空间混叠误差的修正。
  5. 使用 `supdeq_calcLSD_HRIR` 计算三种方法的左耳 LSD，并使用 `AKerbError` 计算 ERB-band magnitude error；两组曲线均对 2702 个方向取平均。
  6. 通过 `scripts/run_mca_demo_export.m` 自动运行 `SUpDEq-master/supdeq_demo_MCA.m`，并导出全部实验图和指标数据。复现命令为 `matlab -wait -batch "run('D:\\course\\CUC_2\\CSMT\\MCAR\\scripts\\run_mca_demo_export.m')"`。
- 指标结果：
  - 已成功生成 SH、SUpDEq + SH、MCA 的左耳全方向平均 LSD 曲线和 ERB-band magnitude-error 曲线。
  - 已生成 6 张正前方或对侧耳 HRIR 对比图。demo 的对侧示例表明，MCA 能修正约 10 kHz 以上的高频凸起，使结果更接近 reference；该改善尚未形成独立的数值统计。
  - 导出脚本共生成 8 个 `.png`、8 个 `.fig` 和 `mca_demo_metrics.mat`，共 17 个可复现实验产物。
- 输出文件位置：`../figures/mca_demo_ku100_ns3_nd44/`。该目录以及 `../outputs/` 中的大型 MAT 文件均为本地可复现产物，不提交 Git；下列相对链接需先运行导出脚本：
  - [正前方 conventional vs MCA HRIR](../figures/mca_demo_ku100_ns3_nd44/01_frontal_conventional_vs_mca_hrir.png)
  - [对侧耳 SH vs MCA HRIR](../figures/mca_demo_ku100_ns3_nd44/02_contralateral_sh_vs_mca_hrir.png)
  - [对侧耳 conventional vs MCA HRIR](../figures/mca_demo_ku100_ns3_nd44/03_contralateral_conventional_vs_mca_hrir.png)
  - [对侧耳 SH vs reference HRIR](../figures/mca_demo_ku100_ns3_nd44/04_contralateral_sh_vs_reference_hrir.png)
  - [对侧耳 conventional vs reference HRIR](../figures/mca_demo_ku100_ns3_nd44/05_contralateral_conventional_vs_reference_hrir.png)
  - [对侧耳 MCA vs reference HRIR](../figures/mca_demo_ku100_ns3_nd44/06_contralateral_mca_vs_reference_hrir.png)
  - [SH、SUpDEq + SH、MCA 的 LSD 曲线](../figures/mca_demo_ku100_ns3_nd44/07_lsd_left_ear.png)
  - [SH、SUpDEq + SH、MCA 的 ERB-band magnitude-error 曲线](../figures/mca_demo_ku100_ns3_nd44/08_erb_magnitude_error_left_ear.png)
- 阻塞项：
  1. 尚未定义并实现对侧耳区域的统一判定规则，也未对 `>10 kHz` 区域计算 conventional 与 MCA 的均值、中位数、误差差值和相对改善百分比，因此目前只有可视化趋势，缺少论文可用的定量结论。
  2. residual MLP 的训练数据生成流程尚未固定：仍需明确输入特征、log-magnitude residual 目标、训练/验证/测试划分以及跨方向或跨个体的划分方式。
  3. residual MLP 尚未实现和训练，因此当前只能完成 SH、SUpDEq + SH、MCA 三种传统基线比较，尚不能评估 MCA + residual MLP 的增益。
- 结论与下一步：MCA demo 已成功复现，传统基线和图表导出流程已经跑通。下一步先完成对侧耳高频区域的定量统计，再固定 residual 数据集格式与划分方式，最后训练轻量 MLP，并以 LSD、ERB-band magnitude error、ILD error 和对侧高频误差与 MCA baseline 比较。
