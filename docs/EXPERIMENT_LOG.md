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

## 2026-07-25：MLP + CNN v3 的 pp91 CNN-only 过拟合检查

- 实验目标：验证新增双耳频谱 CNN 在冻结 v2 MLP 时，能否从真实 HUTUBS 完整频谱中学习有效的 delta residual，并同时降低 residual、ERB、高频和 ILD 训练代理指标；本实验只作为单被试管线检查，不作为跨被试结论。
- 训练入口：新增 `residual_learning/mlp_cnn_v3/python/train_mlp_cnn_v3.py`。训练器加载 v2 epoch 9 checkpoint，冻结全部 `100,225` 个 MLP 参数，只优化 `74,402` 个 CNN + FiLM 参数。validation sampler 每次以相同 seed 重新创建，因此初始 v2 与每个 epoch 使用完全相同的固定 validation batch 序列。
- 数据与采样：HUTUBS simulated pp91、Lebedev N=3 residual HDF5；每个 batch 采样 16 个 Fliege 方向、双耳和全部 463 个频点，共 `14,816` 个逐频点样本。train 和 validation 均使用 pp91，但采样序列分别使用 seed `20260725` 和 `20260726`。
- 损失与优化：保持 v2 复合损失不变，即 residual SmoothL1 + `0.50 × ERB proxy/target_std + 0.25 × 对侧高频/target_std + 0.25 × ILD proxy/target_std`；AdamW 学习率 `3e-4`、weight decay `1e-5`、cosine decay、gradient clip `5.0`、FP16 AMP。训练为 4 epoch × 120 steps，每 epoch 使用 48 个固定 validation batch。
- AMP 修正：第一次运行使用 PyTorch 默认初始 loss scale `65536`，零初始化 CNN 输出头的首步放大梯度出现非有限值，训练器按预期中止。烟雾测试的未缩放梯度有限，说明问题来自初始缩放而非数据或损失。训练器现将 `amp_initial_scale` 默认设为 `1024`，并在 unscale 后检测梯度；AMP 溢出步会由 GradScaler 安全跳过并计数。本次完整检查 480 个优化步均未跳过，最终 scale 为 `1024`。
- 固定验证结果：初始 v2 的复合损失/residual/ERB/高频/ILD 为 `0.53335 / 1.9304 / 0.6441 / 3.0559 / 0.5875 dB`；最佳 epoch 4 为 `0.40730 / 1.6244 / 0.5436 / 2.3722 / 0.3170 dB`。相对初始 v2 分别降低 `23.63% / 15.85% / 15.60% / 22.37% / 46.05%`。epoch 4 的 CNN delta 平均绝对值为 `0.8512 dB`。
- 收敛过程：validation total loss 从 epoch 1 到 4 依次为 `0.49027、0.44813、0.41958、0.40730`，连续下降；对应高频误差为 `2.817、2.593、2.429、2.372 dB`，ILD 为 `0.457、0.391、0.359、0.317 dB`。最佳点仍位于最后一个 epoch，说明检查没有出现提前退化。
- 资源与耗时：4 epoch 总训练耗时 `35.74 s`，单 epoch 约 `8.7–9.1 s`；峰值 CUDA allocated memory 为 `119.82 MiB`。本机 RTX 5060 8 GB 有充分余量，完整训练可优先尝试 32 方向 batch。
- 实际命令：`D:\miniconda3\envs\ml\python.exe residual_learning/mlp_cnn_v3/python/train_mlp_cnn_v3.py residual_learning/data/hutubs_residual_v1_n03 residual_learning/runs/mlp_n03_v2/best.pt --run-name overfit_pp91_cnn_only --overfit-subject 91 --epochs 4 --steps-per-epoch 120 --validation-steps 48 --directions-per-batch 16`，退出码为 0。
- 输出与 Git：本地结果位于 `residual_learning/mlp_cnn_v3/runs/overfit_pp91_cnn_only`，包含 configuration、initial validation、history、best/last checkpoint 和 training report；该单被试生成目录由 v3 嵌套 `.gitignore` 忽略。训练源码和 README 保留为待提交文件。
- 结论与下一步：CNN-only 分支已经通过单被试可学习性、指标方向、checkpoint、固定验证和 AMP 稳定性检查。下一步在原 72 train / 12 validation 被试上进行完整 CNN-only 训练，保持 v2 MLP 和损失权重不变；根据固定 validation 复合损失选最佳 epoch，再做完整 validation residual 评估。test 和 MATLAB 严格指标在模型确定前继续锁定。

## 2026-07-25：MLP + CNN v3 独立工作流与真实数据梯度检查

- 实验目标：在 v2 指标感知 MLP 的基础上加入频率上下文建模，同时保持已有 subject-wise 划分、HDF5 数据、复合损失和最终 MATLAB 指标口径不变。第一步只实现推荐的网络结构并验证真实数据前向、v2 无损初始化、复合损失和 CNN 反向传播，不开始完整训练。
- 工作流目录：新增 `residual_learning/mlp_cnn_v3`，内部独立保存 v3 的 Python 模型与检查脚本，以及后续 `runs/` 和 `reconstruction/`。源码和说明可提交；checkpoint、训练缓存、预测 HDF5 和 MAT 等生成产物由嵌套 `.gitignore` 默认忽略。
- 模型结构：保留 v2 的 7 输入、宽度 128、3 个 residual block 的 `ResidualMLP` 作为逐频点基础预测器；新增双耳 `BinauralSpectralCNN`，输入为左右耳归一化 MCA magnitude、correction magnitude、v2 residual 以及归一化 log-frequency，共 7 个频谱通道。CNN 使用 48 通道 stem、4 个 kernel 7 的深度可分离 residual block，dilation 为 `1/2/4/8`，理论感受野为 91 个频点；方向 `x/y/z` 经 64 维 MLP 生成逐块 FiLM 调制。CNN 输出左右耳两个 delta residual 通道，并与 v2 MLP 输出相加。
- 初始化与复杂度：CNN 输出层和 FiLM 线性层零初始化，确保加载 v2 epoch 9 checkpoint 后，训练起点严格满足 `v3 = v2 + 0`。v2 MLP 为 `100,225` 参数，CNN + FiLM 为 `74,402` 参数，总计 `174,627` 参数；冻结 MLP 的第一阶段仅训练 `74,402` 参数。
- 环境检查：`D:\miniconda3\envs\ml` 中 PyTorch `2.8.0+cu128`、CUDA build `12.8` 和 h5py `3.14.0` 可用，GPU 为 NVIDIA GeForce RTX 5060。v2 `best.pt`、训练统计量和 pp91 N=3 HDF5 均存在并可读取。
- 烟雾测试：`smoke_test_mlp_cnn.py` 使用真实 pp91 双耳完整 463 点频谱和 v2 复合损失运行两个优化步。4 方向与 16 方向测试均通过；输入/输出分别为 `[direction, 2, 463, 7]` 和 `[direction, 2, 463]`。零初始化时 v3 与 v2 最大逐点差值为 `0`；第一个优化步只有零初始化输出头的 2 个参数张量获得非零梯度，完成一次更新后第二步已有 62 个 CNN 参数张量获得非零梯度，所有已计算梯度和损失均为有限值，冻结 MLP 未产生梯度。
- 资源检查：16 方向、FP16 AMP、完整 v2 复合损失的峰值 CUDA allocated memory 为 `197.12 MiB`，远低于本机约 8 GB 显存；正式 CNN-only 训练可以从 16 或 32 方向起步。
- 实际命令：`D:\miniconda3\envs\ml\python.exe residual_learning/mlp_cnn_v3/python/smoke_test_mlp_cnn.py --dataset-root residual_learning/data/hutubs_residual_v1_n03 --checkpoint residual_learning/runs/mlp_n03_v2/best.pt --subject 91 --directions 16`，退出码为 0。
- 输出文件位置：模型为 `residual_learning/mlp_cnn_v3/python/mlp_cnn_model.py`，真实数据检查为 `residual_learning/mlp_cnn_v3/python/smoke_test_mlp_cnn.py`，结构、运行命令和目录约定见 `residual_learning/mlp_cnn_v3/README.md`。
- 结论与下一步：v3 第一阶段模型链路已通过，当前不需要补充数据或环境。下一步实现 CNN-only 训练入口，冻结 v2 MLP，保持 v2 的 `0.50/0.25/0.25` ERB/对侧高频/ILD 权重不变；先在 pp91 做短程过拟合检查，再决定正式训练的方向 batch size 与 epoch 数。

## 2026-07-24：Residual MLP v2 双耳指标感知训练与回填评估

- 实验目标：在 v1 已证明逐频点 residual 可学习的基础上，加入双耳完整频谱与听觉指标感知损失，重点扩大 ERB 改善并消除 pp33、pp81 的 ILD 退化；最终仍在原 12 个严格未见 test 被试上，用 MATLAB `AKerbError`、对侧高频和水平面 ILD 的相同口径评估。
- 模型与初始化：网络仍为 7 输入、宽度 128、3 个 SiLU residual block、单输出的 `ResidualMLP`，可训练参数保持 `100,225`；从 v1 epoch 10 的 `best.pt` 初始化，随后对全部参数继续训练。这样可将 v1 作为稳定起点，并把优化重点转向听觉指标，而不是从零重新学习。
- 双耳采样：新增 `BinauralSpectrumSampler`。每个 batch 随机选择一个 train 被试和 32 个 Fliege 方向，同时读取左右耳与全部 463 个频点，张量布局为 `[ear=2, direction=32, frequency=463]`，共 `29,632` 个逐频点样本。该布局保证同方向双耳能量和完整导出频谱可同时参与损失。
- 复合损失：总损失为归一化 residual SmoothL1，加 `0.50 × ERB代理MAE/target_std`、`0.25 × 对侧高频MAE/target_std`、`0.25 × ILD代理MAE/target_std`。ERB 代理使用 50 Hz–20 kHz、41 个 ERB-rate 三角带，对修正后与 reference 的频带能量 dB 做误差；对侧高频项按左右耳相反半球选择 `>10 kHz` 频点；ILD 代理由双耳频谱能量比计算。代理项只用于可微训练，最终结果仍由 MATLAB 原始 `AKerbError` 和 HRIR 能量 ILD 计算。
- 数值与过拟合检查：真实 pp91 batch 的四项损失均有限，反向梯度全部有限；16 方向检查的 CUDA 峰值分配约 `134.4 MiB`。pp91 短程检查为 4 epoch × 120 steps，validation 代理 ERB 从 epoch 1 的 `0.603` 降至 `0.561 dB`，对侧高频从 `2.538` 降至 `2.331 dB`，ILD 从 `0.411` 降至 `0.358 dB`，证明双耳采样与复合梯度链路有效。
- 完整训练：72 个 train、12 个 validation 被试；AdamW 学习率 `3e-4`、weight decay `1e-5`、cosine decay、FP16 AMP；10 epoch × 500 steps，每 epoch 96 个 validation batch，总耗时 `331.3 s`（约 5 分 31 秒）。固定 validation batch 上的 v1 初始代理指标为 residual/ERB/高频/ILD `2.164/0.773/3.301/0.739 dB`；根据复合 validation loss 选择 epoch 9，得到 `2.117/0.693/3.267/0.609 dB`。
- 完整逐频点指标：v2 在 validation 的 MAE 为 `2.1398 dB`，相对 MCA `2.5719 dB` 改善 `16.80%`；在 test 的 MAE 为 `2.1684 dB`，相对 MCA `2.6007 dB` 改善 `16.62%`，RMSE 为 `3.5526 dB`。v1 的 test MAE 为 `2.2172 dB`、改善 `14.75%`，因此 v2 没有以牺牲原始 residual 精度换取听觉指标。
- 严格论文指标：全空间 ERB error 从 MCA `0.8027 ± 0.0255 dB` 降至 v2 `0.6115 ± 0.0356 dB`，改善 `23.82%`；对侧 25° ERB 从 `1.8603 ± 0.1159 dB` 降至 `1.3401 ± 0.0999 dB`，改善 `27.96%`；对侧半球 `>10 kHz` error 从 `4.2583 ± 0.2641 dB` 降至 `3.7204 ± 0.2436 dB`，改善 `12.63%`；ILD MAE 从 `0.8854 ± 0.2092 dB` 降至 `0.6467 ± 0.2662 dB`，改善 `26.96%`。四项均为 12/12 test 被试改善。
- v1→v2 增益：相对 v1，v2 进一步降低全空间 ERB `10.57%`、对侧 25° ERB `14.00%`、对侧高频 `0.94%`、ILD `16.31%`。pp33 的 ILD 从 v1 的退化 `-5.57%` 变为相对 MCA 改善 `14.60%`；pp81 从退化 `-18.97%` 变为改善 `0.52%`，说明双耳 ILD 损失解决了主要失败案例，但 pp81 仍是最弱改善被试。
- 回填与可视化：复用 v1 的 12 份 complex MCA/reference cache，使用 v2 epoch 9 checkpoint 预测 1260 方向 residual，并保持 MCA 相位回填。最大相位误差 `6.29e-16 rad`、幅度恒等误差 `7.11e-15 dB`。输出 12 张逐被试四面板图、12 人对侧 HRTF 总览、逐被试指标总览和 v1/v2 直接对比图。
- 输出与 Git：训练源码为 `train_residual_mlp_v2.py` 与扩展后的 `residual_data.py`；评估复用 `predict_reconstructed_residuals.py` 和 `evaluate_test_reconstruction.m`，新增 `plot_v1_v2_reconstruction_comparison.m`。训练结果位于 `residual_learning/runs/mlp_n03_v2`，回填结果位于 `residual_learning/reconstruction/mlp_n03_v2`。Git 仅保留训练 history/report、val/test JSON、最终 CSV 和 PNG；checkpoint、本机配置、预测 HDF5、MAT、日志及 `overfit_pp91_v2` 继续忽略。
- 结论与下一步：v2 的指标感知训练在不增加模型参数的情况下显著扩大 ERB 与 ILD 改善，并让所有 test 被试的四项严格指标都优于 MCA。下一步应进行损失消融（residual+ERB、+高频、+ILD）以量化每个损失项贡献，并将 v2 从固定 N=3 扩展到其他稀疏阶数，优先 N=1、2、4、6。

## 2026-07-24：Residual MLP v1 幅度回填与论文指标评估

- 实验目标：在严格未见的 12 个 test 被试上，将 MLP 预测的 `reference_logmag_db - mca_logmag_db` 回填到 N=3 MCA 幅度，同时逐频点保留原 MCA 相位；计算与传统 MCA 复现一致的 ERB magnitude error、对侧高频误差和水平面 ILD error，并生成 12 个被试的直观 HRTF 对比图。
- 数据集与被试：HUTUBS simulated HRTF；test 被试为 pp8、pp18、pp22、pp26、pp31、pp33、pp45、pp47、pp59、pp70、pp73、pp81。稀疏输入为 Lebedev N=3（26 点）；评估方向为 Fliege N=29（900 点）和水平面 0–359°（360 点）。pp18 缺少公开人体测量值，沿用此前规则，以 93 名有效被试的平均 Algazi 半径 `0.091021 m` 代替。
- 重建方法：MATLAB 重新生成每个被试的复数 MCA/reference HRTF 和 correction filter；Python/PyTorch 使用 epoch 10 的 `best.pt`、训练集归一化参数、RTX 5060 FP16 AMP，对 1260 个方向和 463 个频点（实际 `86.13–19982.81 Hz`）预测 residual；回到 MATLAB 后执行 `corrected_logmag = mca_logmag + predicted_residual`，并使用 `exp(j*angle(H_MCA))` 恢复复数 HRTF。20 kHz 以上保持原 MCA 不变。
- 指标口径：ERB 指标调用与传统基线相同的 `AKerbError`，范围 50 Hz–Nyquist，方向按 Fliege 权重、ERB band 等权；同时报告全空间与每耳对侧 25° 区域。对侧高频指标为每耳相反开放半球、`f > 10 kHz` 至 Nyquist 的绝对 log-magnitude error，方向按 Fliege 权重、频率等权。ILD 为 360 个水平面方向上左右 HRIR 全带能量比 `10*log10(E_L/E_R)` 相对 reference 的平均绝对误差。
- 汇总结果：全空间 ERB magnitude error 从 MCA `0.8027 ± 0.0255 dB` 降至 `0.6838 ± 0.0326 dB`，平均降低 `0.1189 dB / 14.82%`，12/12 被试改善；对侧 25° ERB error 从 `1.8603 ± 0.1159 dB` 降至 `1.5582 ± 0.0971 dB`，降低 `0.3020 dB / 16.24%`，12/12 改善；对侧半球高频误差从 `4.2583 ± 0.2641 dB` 降至 `3.7557 ± 0.2420 dB`，降低 `0.5026 dB / 11.80%`，12/12 改善；ILD MAE 从 `0.8854 ± 0.2092 dB` 降至 `0.7727 ± 0.3001 dB`，降低 `0.1127 dB / 12.73%`，10/12 改善。
- 重建质量检查：12 个被试的最大相位保持误差为 `6.22e-16 rad`，最大幅度回填恒等误差为 `7.11e-15 dB`；证明输出确实只改变指定频点的幅度，没有改变 MCA 相位。MATLAB Code Analyzer 对两个新增脚本均报告 0 问题，Python 文件可成功编译；准备阶段 12/12、GPU 推理 12/12、最终评估 12/12 均完成。
- 可视化：每个 test 被试各有一张四面板图，包括左右耳对侧 HRTF（Reference/MCA/MCA+Residual MLP）、全空间 ERB error 频率曲线和水平面 ILD；另有一张 4×3 的 12 人左耳对侧 HRTF 总览与一张逐被试指标总览。
- 输出与 Git：源码为 `prepare_test_reconstruction_inputs.m`、`predict_reconstructed_residuals.py`、`evaluate_test_reconstruction.m`；结果位于 `residual_learning/reconstruction/mlp_n03_v1`。Git 仅保留最终 CSV 与 PNG；复数 cache、模型输入、预测 HDF5、MAT、运行日志和含本机路径的推理报告继续忽略。
- 结论与下一步：首版 100,225 参数 residual MLP 在严格未见被试上不仅降低逐频点 residual MAE，也稳定降低论文口径的 ERB 与对侧高频误差，并整体改善 ILD。下一步应分析 pp33 与 pp81 的 ILD 退化原因，并考虑在训练中加入 ERB/ILD 感知损失或双耳联合特征，而不修改相位。

## 2026-07-23：Residual MLP v1（HUTUBS N=3，跨被试）

- 实验目标：在已经导出的 HUTUBS N=3 residual 数据集上，实现并验证首版轻量 MLP，确认模型能在严格的 subject-wise 未见被试上降低 `reference_logmag_db - mca_logmag_db`。
- 训练环境：Conda 环境 `D:\miniconda3\envs\ml`，Python `3.9.23`，PyTorch `2.8.0+cu128`，CUDA build `12.8`，h5py `3.14.0`。GPU 为 RTX 5060（8151 MiB，compute capability 12.0）；已实际验证 HDF5 读取、CUDA 张量计算和混合精度反向传播。
- 模型与输入：`ResidualMLP` 有 `100,225` 个可训练参数；7 个输入特征依次为归一化 MCA log-magnitude、归一化 MCA correction-filter log-magnitude、方向单位向量 `x/y/z`、归一化 `log10(frequency)` 和耳别（左=-1，右=1）。网络为宽度 128 的输入层、3 个 SiLU 残差块和单输出层，预测归一化的 dB residual。
- 采样与优化：为避免将 1.10 GiB 数据整体载入内存，每个 batch 从随机被试和随机耳朵读取 `64` 个方向与 `128` 个频点的笛卡尔块，共 `8192` 样本。归一化参数严格来自 72 个 train 被试；优化器 AdamW（学习率 `1e-3`、weight decay `1e-5`）、SmoothL1 损失、cosine learning-rate decay、CUDA FP16 AMP。训练共 12 epoch × 600 steps（随机有放回采样），每 epoch 以 96 个 validation block 监控。
- 过拟合检查：pp91 单被试运行 8 epoch × 300 steps 后，随机留出 block 的 MAE 从 MCA zero-residual 基线约 `2.34 dB` 降至约 `1.60 dB`，数据读取、梯度和 checkpoint 链路均通过。
- 训练过程：跨被试训练命令为 `train_residual_mlp.py ... --run-name mlp_n03_v1 --epochs 12 --steps-per-epoch 600 --validation-steps 96`；总耗时 `380.2 s`（约 6 分 20 秒）。根据 validation MAE 选择 epoch 10 的 best checkpoint。
- 独立指标结果：在完整 validation（12 被试、`10,000,800` 样本）上，逐频点 residual MAE 从 MCA 基线 `2.5719 dB` 降至 `2.1886 dB`（`14.90%`），RMSE 从 `4.1438` 降至 `3.5346 dB`。在严格未见的 test（12 被试、`10,000,800` 样本）上，MAE 从 `2.6007` 降至 `2.2172 dB`（`14.75%`），RMSE 从 `4.1861` 降至 `3.5792 dB`。
- 输出文件位置：训练代码为 `residual_learning/python/residual_data.py`、`residual_model.py`、`train_residual_mlp.py`、`evaluate_residual_mlp.py`；`residual_learning/runs/mlp_n03_v1` 中仅保留 `history.csv`、`val_metrics.json` 和 `test_metrics.json`，checkpoint、含本机绝对路径的配置与其他运行产物由嵌套 `.gitignore` 忽略。
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
- 输出文件位置：源码、划分和说明位于 `residual_learning/`；大型数据位于 `residual_learning/data/hutubs_residual_v1_n03`，由嵌套 `.gitignore` 忽略；仅保留该目录的 `training_statistics.json` 以复现训练归一化。
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
