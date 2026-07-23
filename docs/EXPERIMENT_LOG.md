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

## 2026-07-23：Residual MLP v1（HUTUBS N=3，跨被试）

- 实验目标：在已经导出的 HUTUBS N=3 residual 数据集上，实现并验证首版轻量 MLP，确认模型能在严格的 subject-wise 未见被试上降低 `reference_logmag_db - mca_logmag_db`。
- 训练环境：Conda 环境 `D:\miniconda3\envs\ml`，Python `3.9.23`，PyTorch `2.8.0+cu128`，CUDA build `12.8`，h5py `3.14.0`。GPU 为 RTX 5060（8151 MiB，compute capability 12.0）；已实际验证 HDF5 读取、CUDA 张量计算和混合精度反向传播。
- 模型与输入：`ResidualMLP` 有 `100,225` 个可训练参数；7 个输入特征依次为归一化 MCA log-magnitude、归一化 MCA correction-filter log-magnitude、方向单位向量 `x/y/z`、归一化 `log10(frequency)` 和耳别（左=-1，右=1）。网络为宽度 128 的输入层、3 个 SiLU 残差块和单输出层，预测归一化的 dB residual。
- 采样与优化：为避免将 1.10 GiB 数据整体载入内存，每个 batch 从随机被试和随机耳朵读取 `64` 个方向与 `128` 个频点的笛卡尔块，共 `8192` 样本。归一化参数严格来自 72 个 train 被试；优化器 AdamW（学习率 `1e-3`、weight decay `1e-5`）、SmoothL1 损失、cosine learning-rate decay、CUDA FP16 AMP。训练共 12 epoch × 600 steps（随机有放回采样），每 epoch 以 96 个 validation block 监控。
- 过拟合检查：pp91 单被试运行 8 epoch × 300 steps 后，随机留出 block 的 MAE 从 MCA zero-residual 基线约 `2.34 dB` 降至约 `1.60 dB`，数据读取、梯度和 checkpoint 链路均通过。
- 训练过程：跨被试训练命令为 `train_residual_mlp.py ... --run-name mlp_n03_v1 --epochs 12 --steps-per-epoch 600 --validation-steps 96`；总耗时 `380.2 s`（约 6 分 20 秒）。根据 validation MAE 选择 epoch 10 的 best checkpoint。
- 独立指标结果：在完整 validation（12 被试、`10,000,800` 样本）上，逐频点 residual MAE 从 MCA 基线 `2.5719 dB` 降至 `2.1886 dB`（`14.90%`），RMSE 从 `4.1438` 降至 `3.5346 dB`。在严格未见的 test（12 被试、`10,000,800` 样本）上，MAE 从 `2.6007` 降至 `2.2172 dB`（`14.75%`），RMSE 从 `4.1861` 降至 `3.5792 dB`。
- 输出文件位置：训练代码为 `residual_learning/python/residual_data.py`、`residual_model.py`、`train_residual_mlp.py`、`evaluate_residual_mlp.py`；本地 checkpoint、CSV 和 JSON 位于 `residual_learning/runs/mlp_n03_v1`，由嵌套 `.gitignore` 忽略。
- 结论与限制：首版轻量 MLP 已在未见被试上稳定降低原始频谱 residual，证明 MCA 后残差包含可跨被试学习的规律。当前结果不是论文 ERB magnitude error、ILD 或 ITD；模型只预测幅度 residual，尚未将预测值回填为校正 HRTF 并重算这些最终听觉指标。
- 下一步：实现“预测 residual → 修正 MCA 幅度 → 结合原 MCA 相位”的重建与评估器，首先在完整 test 集报告 auditory-band magnitude error、对侧高频误差和 ILD；ITD 预计保持 MCA 水平，因为本阶段不修改相位。

## 2026-07-23：Residual learning 数据集 v1（HUTUBS N=3）

- 实验目标：建立独立的 `residual_learning/` 阶段目录，固定无被试泄漏的数据划分，并为轻量网络导出 `target = reference_logmag_db - mca_logmag_db` 的首版训练数据。
- 数据集与划分：96 个 HUTUBS simulated HRTF；使用 MATLAB `rng(20260723, 'twister'); randperm(96)` 固定 subject-wise 划分为 train/validation/test = `72/12/12`。同一被试只属于一个集合。公开人体测量缺失的 pp79、pp92、pp18 分别位于 train、validation、test。
- 稀疏网格与目标网格：首版固定 Lebedev `N=3`（26 个稀疏方向），目标为 Fliege `N=29`（900 个方向），双耳独立保存。选择条件为 `50 Hz <= f <= 20 kHz`；由于 44.1 kHz、1024 点 FFT 的离散频率栅格，实际得到 463 个频点，范围 `86.1328`–`19982.8125 Hz`。
- 数据格式：每个被试/阶数独立保存一个 HDF5；Python 侧频谱布局统一为 `[ear, direction, frequency]`。主要字段为 `/mca_logmag_db`、`/reference_logmag_db`、`/correction_logmag_db`、`/target_residual_db`、`/direction_features` 和 `/frequency_hz`。方向特征包含 azimuth、elevation、单位向量 `x/y/z` 和 Fliege weight。文件先写入 `.partial`，完整校验和元数据写入后再原子改名，可断点续跑。
- 实验过程：先执行 `export_hutubs_residual_dataset(91,3,1,'pilot_pp91_n03')` 完成 pp91 试导出，再执行 `export_hutubs_residual_dataset(1:96,3,6,'hutubs_residual_v1_n03')`，使用 6 个 MATLAB process workers 完成全量导出；全量运行耗时 `920.8 s`（约 15 分 21 秒）。项目 Python 虚拟环境安装 `h5py 3.16.0`，使用 `validate_residual_hdf5.py` 和 `compute_training_statistics.py` 做读取验证及训练集统计。
- 完整性检查：96/96 HDF5 成功，0 failure、0 partial；每个文件包含 `833,400` 个样本，总计 `80,006,400` 个样本。遍历读取全部 96 个文件后，train/validation/test 属性计数为 `72/12/12`，全部数组尺寸一致且数值有限，`reference - MCA = residual` 的最大恒等误差为 `0 dB`。输出总大小 `1,179,616,094` 字节（`1124.97 MiB`，约 `1.10 GiB`）。
- 训练集统计：仅使用 72 个 train 被试的 `60,004,800` 个样本计算归一化参数。MCA log-magnitude 均值/标准差为 `0.4367/9.3164 dB`；correction-filter log-magnitude 为 `0.8184/2.6988 dB`；目标 residual 均值/标准差为 `-0.0905/4.0625 dB`，平均绝对值为 `2.5156 dB`，范围约 `-67.71`–`69.57 dB`。
- 指标解释：这里的 residual 是逐 FFT 频点的细粒度谱差，不是论文 41 个 auditory band 上的能量误差，因此其平均绝对值不能直接与前一阶段约 0.8 dB 的 ERB 指标比较。后续训练仍应以原始 residual 为监督，并用论文 ERB、ILD、对侧高频误差作为最终评价。
- 输出文件位置：源码、划分和说明位于 `residual_learning/`；大型数据位于 `residual_learning/data/hutubs_residual_v1_n03`，由嵌套 `.gitignore` 忽略。训练集统计在该目录的 `training_statistics.json`。
- 硬件准备：本机 NVIDIA GeForce RTX 5060，显存 `8151 MiB`，驱动 `595.79`，CUDA compute capability `12.0`；适合使用混合精度训练轻量 MLP，但 batch size 需按 8 GB 显存控制。
- 阻塞项：当前虚拟环境尚未安装 PyTorch；在安装前应核对支持 RTX 5060 / compute capability 12.0 的官方 CUDA wheel 版本。
- 结论与下一步：Residual 数据集 v1 已可直接供 Python 训练。下一步实现按 HDF5 随机采样的 PyTorch Dataset、仅由 train 统计量归一化的轻量 MLP，以及 zero-residual（即原 MCA）对照；先跑小规模过拟合检查，再进行完整训练。

## 2026-07-23：HUTUBS 96 被试 MCA 全量复现与跨被试汇总

- 实验目标：按论文技术评估设置，在全部 96 个 HUTUBS simulated SOFA 上完成 Lebedev `N=1`–`10` 的 SH only、SUpDEq + SH 和 MCA 基线，并汇总跨被试幅度、ILD、ITD 指标与论文 Fig. 5 风格曲线。
- 数据集与版本：本机 `data/HRTF/hutubs` 中 `pp1`–`pp96` 的 96 个 simulated SOFA；每个文件包含 Lebedev `N=35` 的 1730 个方向、双耳 256 点 HRIR，采样率 44.1 kHz。
- 稀疏网格与采样点数：稀疏输入为 Lebedev `N=1`–`10`；幅度误差目标为 Fliege `N=29`（900 方向）；双耳线索目标为水平面 0–359 度（360 方向）。ERB 指标实际得到 41 个中心频率，范围 `50`–`19792.31 Hz`，与论文的 50 Hz–20 kHz、41 个听觉滤波器一致。
- 方法与关键参数：SH only（`'None'`, `'SH'`, `mc=nan`）、conventional（`'SUpDEq'`, `'SH'`, `mc=nan`）与 MCA（`'SUpDEq'`, `'SH'`, `mc=inf`）；FFT oversize 为 4。MCA 使用 SUpDEq 默认最小相位幅度校正、空间混叠频率以下限制和 `fadeDown`。头半径由 HUTUBS `x1,x2,x3` 与 Algazi 公式计算；公开人体测量表中 pp18、pp79、pp92 缺值，三者透明地使用其余 93 人的平均半径 `0.091021 m`，这是相对论文私有完整元数据的已知偏差。
- 实验过程：使用 6 个 MATLAB process workers 按被试并行，每个被试内部按 `N=1`–`10` 顺序执行并逐阶保存独立检查点。实际命令为 `matlab -batch "addpath(fullfile(pwd,'reproduce')); run_hutubs_mca_batch(1:96,1:10,6,'hutubs_mca_batch')"`；总耗时 `11471.8 s`（约 3 小时 11 分 12 秒）。随后执行 `matlab -batch "addpath(fullfile(pwd,'reproduce')); aggregate_hutubs_mca_results"`，约 57 秒完成跨被试汇总和绘图。
- 完整性检查：96/96 被试成功，0 个失败；共 960 个逐阶 MAT、960 个 ERB CSV、960 个双耳指标 CSV、96 个完成标记；每位被试均恰有 10 个阶数。结果总大小 `203,774,606` 字节（约 194.3 MiB）。
- 统计口径修正：批处理 CSV 最初使用 Fliege 求积权重做方向平均；论文 Fig. 5 是对选中方向做普通算术平均。逐阶 MAT 已保存每个方向的跨频率误差，因此汇总阶段直接由检查点重建论文严格口径，不需要重新插值。两种口径均保留，并以文件名明确区分；论文对照图使用未加权方向平均。
- 幅度指标结果：左耳对侧 25 度区域中，N=2 conventional/MCA 为 `2.7789/1.8558 dB`，改善 `0.9231 dB`，96/96 被试改善；N=3 为 `2.6819/1.8965 dB`，改善 `0.7854 dB`，96/96 改善。左耳前方 25 度的最大改善位于 N=4，由 `0.7460 dB` 降至 `0.4319 dB`，改善 `0.3141 dB`，95/96 改善。左耳全空间在 N=3 由 `1.0980 dB` 降至 `0.8035 dB`，改善 `0.2945 dB`，96/96 改善。N=1 全空间 MCA 略差（`1.5664` 升至 `1.6370 dB`），与论文所述低阶 N=1 难以进一步改善的现象一致。
- 双耳指标结果：N=3 水平面平均绝对 ILD 误差由 conventional `1.5623 dB` 降至 MCA `0.8611 dB`；平均绝对 ITD 误差为 `11.7113/11.6454 µs`，几乎不变，符合幅度校正主要改善 ILD 而不改变低频到达时间的预期。
- 论文对照结论：复现曲线重现原文 Fig. 5 的核心模式：对侧误差显著高于前方；N=1 对侧改善很小；最大幅度改善集中在 N=2–5；前方改善峰值在 N=3–4；跨被试标准差在 MCA 后总体减小。当前结论可作为后续 residual 网络必须超过的 MCA 基线。
- 输出文件位置：根目录 `reproduce/hutubs_mca_batch`；汇总在 `reproduce/hutubs_mca_batch/aggregate`，包括严格口径 ERB 表、加权口径 ERB 表、双耳指标表、改进表、MAT 汇总以及 `fig05_region_magnitude_error.png`、`full_sphere_magnitude_error.png`、`binaural_error_by_order.png`、`frequency_error_n3_n6.png`。`reproduce/` 根目录的 MATLAB 复现脚本纳入 Git，子目录中的大型检查点和生成结果继续由 `.gitignore` 忽略。
- 阻塞项：无计算阻塞。正式写作时需注明 pp18、pp79、pp92 的头半径替代策略；若获得三人的完整人体测量值，可仅重跑这三人做敏感性检查。
- 结论与下一步：96 被试 MCA 传统基线已完成，可进入 residual 学习数据导出。下一步应先固定 subject-wise 训练/验证/测试划分，再导出 `log|H_ref| - log|H_MCA|` 目标及方向、频率、耳别、MCA 幅度与 correction-filter 特征，避免同一被试跨集合造成数据泄漏。

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
