# 项目实验日志

缩写说明：多层感知机（Multi-Layer Perceptron，MLP）、卷积神经网络
（Convolutional Neural Network，CNN）、头相关传输函数（Head-Related
Transfer Function，HRTF）、头相关脉冲响应（Head-Related Impulse Response，
HRIR）、幅度校正与时间对齐插值（Magnitude-Corrected and Time-Aligned
Interpolation，MCA）、耳间电平差（Interaural Level Difference，ILD）、
等效矩形带宽（Equivalent Rectangular Bandwidth，ERB）、平均绝对误差
（Mean Absolute Error，MAE）、均方根误差（Root Mean Squared Error，RMSE）、
Hierarchical Data Format version 5（HDF5）、快速傅里叶逆变换（Inverse Fast
Fourier Transform，IFFT）、快速傅里叶变换（Fast Fourier Transform，FFT）、
方向均衡空间上采样（Spatial Upsampling by Directional Equalization，
SUpDEq）、球谐函数（Spherical Harmonics，SH）、耳间时间差（Interaural Time
Difference，ITD）、空间声学数据格式（Spatially Oriented Format for
Acoustics，SOFA）、对数谱失真（Log-Spectral Distortion，LSD）、图形处理器
（Graphics Processing Unit，GPU）、统一计算设备架构（Compute Unified Device
Architecture，CUDA）、自动混合精度（Automatic Mixed Precision，AMP）、
16 位浮点数（16-bit Floating Point，FP16）、特征级线性调制（Feature-wise
Linear Modulation，FiLM）、Sigmoid 线性单元（Sigmoid Linear Unit，SiLU）、
命令行界面（Command-Line Interface，CLI）、应用程序编程接口（Application
Programming Interface，API）、统一资源定位符（Uniform Resource Locator，
URL）、传输层安全协议（Transport Layer Security，TLS）、Weights & Biases
（W&B）、逗号分隔值（Comma-Separated Values，CSV）、JavaScript 对象表示法
（JavaScript Object Notation，JSON）、便携式网络图形（Portable Network
Graphics，PNG）、多边形文件格式（Polygon File Format，PLY）和便携式文档格式
（Portable Document Format，PDF）、图号（Figure，Fig.）、MATLAB Figure
（FIG）、冲激响应（Impulse Response，IR）、Network Common Data Form version
4（NetCDF4）、标识符（Identifier，ID）、吉字节（gigabyte，GB）、吉比字节
（gibibyte，GiB）和兆二进制字节（mebibyte，MiB）。MATLAB MAT-file 缩写为
MAT。HUTUBS、AXD 和 KU100 是数据集或设备专名，不作首字母展开。

## 2026-08-01：SONICOM Q26 MLP+CNN v3.1 水平面严格 ILD 微调

- 工作目标：从锁定 v3 epoch 10 继续微调冻结 MLP 的 CNN，使训练损失与最终
  水平面 HRIR 能量 ILD 完全对齐，同时保持 residual、ERB 和对侧高频收益；
  44 名 test 继续不导出、不读取。
- 对齐改动：`BinauralSpectrumSampler` 和 `train_mlp_cnn_v3.py` 新增默认关闭的
  `horizontal_only`/`--horizontal-only`。SONICOM v3.1 将 767 个纯插值方向与
  零仰角条件取交集，得到 72 个水平面方向；HUTUBS 与既有 v3/v3.1 默认行为不变。
  回归测试覆盖插值 mask 与水平面 mask 的交集。
- 代码 smoke：从 v3 checkpoint 初始化，1 epoch、2 step、32 方向，严格 HRIR
  IFFT-ILD 可正常反传，0 个跳步，峰值 CUDA allocated memory `221.96 MiB`。
- 权重预实验：固定 ERB/高频权重 `0.75/0.25`、学习率 `1e-4`，比较严格水平面
  ILD 权重 `0.5/1.0/2.0`，每组 `3 epoch × 120 step`。三组 residual、ERB、
  高频和严格 ILD 均相对初始 v3 改善；严格 ILD 改善依次为
  `0.40% / 0.68% / 0.93%`。按预声明规则选择 `2.0`，其 residual/ERB/高频仍
  改善 `0.87% / 1.22% / 1.01%`。比较位于
  `results/sonicom_mlp_cnn_q26_v31_weight_pilot/`。
- 正式训练：从 v3 epoch 10 初始化，冻结 100,225 参数 MLP，只训练 74,402 参数
  CNN；`6 epoch × 500 step`、96 个固定 validation block、每 batch 32 个水平面
  方向、AdamW、cosine schedule、AMP、seed `20260731`。用时 `296.56 s`，峰值
  显存 `221.96 MiB`，0 个跳步；总目标与严格 ILD 最优点均为 epoch 6。
- 固定水平面 validation：v3 到 v3.1 的 residual 为
  `2.721718 → 2.682673 dB`，ERB 为 `1.130511 → 1.099314 dB`，对侧高频为
  `3.638068 → 3.585039 dB`，严格 HRIR ILD 为 `0.659525 → 0.635705 dB`，分别
  改善 `1.43% / 2.76% / 1.46% / 3.61%`。
- 完整诊断：44 人、767 个纯插值方向的 raw residual MAE/RMSE 为
  `2.637748 / 3.979081 dB`，相对 v2 MAE 改善 `6.06%`；全空间严格 ILD 为
  `0.629976 dB`。由于本模型只针对水平面微调，全空间 ILD 仅作诊断，不用于
  checkpoint 选择。
- 产物：正式 checkpoint 位于已忽略的
  `artifacts/training/sonicom_mlp_cnn_q26_v31_horizontal_formal_ild200/best.pt`；
  精选曲线与摘要位于 `results/sonicom_mlp_cnn_q26_v31_formal/`。44 人完整 residual
  已生成到 `artifacts/reconstruction/sonicom_q26_validation_mlp_cnn_v31/`，用时
  `9.23 s`。W&B run `ta1d0lzk` 当前为本地 offline，未经授权不上传。
- MATLAB smoke：经用户明确授权后，以 P0001 运行 MCA/v1/v2/v3/v3.1 五方法
  端到端严格重建，进程正常退出。v3.1 相对 v3 的全空间 ERB、对侧 25° ERB、
  对侧高频分别回退 `4.32% / 0.55% / 1.88%`，水平面 ILD 改善 `2.24%`；
  smoke 只用于确认链路，不用于模型结论。
- 44 人严格重建：正式 MATLAB 批处理正常退出。MCA/v2/v3/v3.1 的全空间 ERB
  分别为 `1.095738 / 0.934826 / 0.887467 / 0.917835 dB`；对侧 25° ERB 为
  `1.763508 / 1.449139 / 1.388265 / 1.398997 dB`；对侧高频为
  `4.749208 / 3.965865 / 3.649390 / 3.712955 dB`；水平面严格 ILD MAE 为
  `0.830014 / 0.653461 / 0.644804 / 0.622905 dB`。
- 最终结论：v3.1 相对 v3 将目标水平面 ILD 改善 `3.40%`，29/44 人改善；
  全空间 ERB、对侧 25° ERB、对侧高频分别回退
  `3.42% / 0.77% / 1.74%`。但 v3.1 三项幅度指标仍相对 v2 改善
  `1.82% / 3.46% / 6.38%`，ILD 相对 v2 改善 `4.68%`。因此保留 v3 为
  幅度均衡基线，保留 v3.1 为 ILD 优化基线；下一轮若要单模型兼顾，应在
  validation 上做多目标/Pareto 折中，而不能宣称 v3.1 全面优于 v3。
- 质量与产物：44 行逐被试表、880 行 method-metric 长表、44 行质量检查和三张
  总览图位于 `results/sonicom_mlp_cnn_q26_v31_strict_validation/`；配置位于
  `configs/experiments/sonicom_mlp_cnn_q26_v31_strict_validation.json`。
  五方法 44 人 HRTF 总览已人工检查，44 个子图与六条曲线完整；test 读取数仍为 0。

## 2026-08-01：SONICOM Q26 MLP+CNN v3 严格 validation 重建评估

- 工作目标：把锁定 v3 epoch 10 的完整 residual 回填到 MCA 幅度，在固定 44 名
  validation 被试上按最终 HRTF/HRIR 口径计算 `AKerbError`、对侧高频和水平面
  严格 ILD，并生成 44 人 HRTF 总览；test 继续不导出、不读取。
- 推理实现：新增 `predict_sonicom_mlp_cnn_residuals.py`，以完整双耳频谱调用
  `ResidualMLPCNN`，按方向分块并原子写出 `2×793×463` residual。1 人 smoke
  通过后正式处理 44 人，用时 `14.34 s`，RTX 5060、AMP、174,627 参数，输出
  位于已忽略的 `artifacts/reconstruction/sonicom_q26_validation_mlp_cnn_v3/`。
- MATLAB 实现：将 `evaluate_sonicom_validation_reconstruction.m` 扩展为可选
  v3 模式，第四个参数指定 v3 prediction run；省略时仍严格复现原 MCA/v1/v2
  行为。代码检查只有既存的动态扩展性能提示，1 人端到端 smoke 和 44 人正式
  运行均成功，正式 MATLAB 进程退出码为 0。
- 最终指标：MCA/v2/v3 的全空间 ERB 为
  `1.095738 / 0.934826 / 0.887467 dB`，v3 相对 v2 改善 `5.07%`、相对 MCA
  改善 `19.01%`，44/44 人优于 v2。
- 对侧 25° ERB：MCA/v2/v3 为
  `1.763508 / 1.449139 / 1.388265 dB`，v3 相对 v2 改善 `4.20%`、相对 MCA
  改善 `21.28%`，41/44 人优于 v2。
- 对侧高频：MCA/v2/v3 为
  `4.749208 / 3.965865 / 3.649390 dB`，v3 相对 v2 改善 `7.98%`、相对 MCA
  改善 `23.16%`，44/44 人优于 v2。
- 水平面严格 ILD：MCA/v2/v3 为
  `0.830014 / 0.653461 / 0.644804 dB`，v3 相对 v2 改善 `1.32%`、相对 MCA
  改善 `22.31%`，27/44 人优于 v2。总体均值改善与全空间严格 ILD 诊断方向一致，
  但被试级稳定性弱于 ERB 和高频。
- 质量与产物：44 个 per-subject 行、704 个 method-metric 长表行、44 个质量检查
  行均完整且全部数值有限；配置位于
  `configs/experiments/sonicom_mlp_cnn_q26_v3_strict_validation.json`，完整 CSV、
  JSON、聚合图、逐被试指标图和 44 人对侧 HRTF 总览位于
  `results/sonicom_mlp_cnn_q26_v3_strict_validation/`。总览图已人工检查，包含
  Reference、MCA、MLP v1、MLP v2 和 MLP+CNN v3 五条曲线且 44 个子图完整。
- 结论与下一步：v3 在最终重建口径上四项总体指标均优于 v2，尤其对侧高频新增
  `7.98%` 收益，证明频率 CNN 的主要价值成立。下一步可锁定 v3 为 SONICOM
  MLP+CNN 基线，并在 validation 上开展与水平面 HRIR 能量完全对齐的 v3.1 ILD
  微调预实验；在损失与训练预算再次锁定前继续保持 test 零读取。

## 2026-08-01：SONICOM Q26 MLP+CNN v3 正式训练与全量 validation

- 工作目标：以已锁定的 SONICOM MLP v2 epoch 10 为基线，在不读取 44 名 test
  的前提下加入沿频率轴建模局部谱形的 1D CNN，完成权重预实验、CNN-only
  正式训练和 44 名 validation 的全量评估。
- 兼容性改动：`train_mlp_cnn_v3.py` 新增 `--interpolation-only` 与
  `--direction-weighted-residual`，并将两项写入 checkpoint/config；SONICOM
  只采样 767 个纯插值方向，residual 损失按 solid-angle weight 加权。默认值
  保持关闭，因此不改变既有 HUTUBS v3/v3.1 入口。`evaluate_mlp_cnn_v3.py`
  同步支持纯插值筛选、solid-angle 加权与加权样本累计。
- 模型：锁定 v2 MLP 的 `100,225` 个参数，只训练 `74,402` 参数的频率 1D CNN；
  总参数为 `174,627`，CNN channel 为 48。CNN 输出层零初始化，训练前模型与
  v2 输出严格一致。最小代码 smoke 为 1 epoch、2 step，初始 CNN delta 为 0，
  峰值 CUDA allocated memory `70.32 MiB`，无跳步。
- 权重预实验：固定 `ERB / 对侧高频 = 0.75 / 0.25`，比较 ILD proxy 权重
  `0.25 / 0.50 / 0.75`，每组 `3 epoch × 120 step`。预先声明的选择规则要求
  residual、ERB 和高频不退化，再最大化 ERB、高频和 ILD 三项改善率最小值。
  三组短 run 的 ILD proxy 分别退化 `3.50% / 2.78% / 2.08%`，因此选择退化
  最小且满足其他约束的 `0.75`，计划依靠正式预算与最终 HRIR 口径复核。
- 正式训练：损失权重 `0.75 / 0.25 / 0.75`，`10 epoch × 500 step`、96 个固定
  validation block、每 block 32 个方向、AdamW、cosine schedule、AMP、seed
  `20260731`。运行用时 `376.47 s`，峰值 CUDA allocated memory
  `220.46 MiB`，跳过 optimizer step 数为 0，最佳点为 epoch 10。
- 固定 validation proxy：v2 到 v3 的 residual MAE 为
  `2.825701 → 2.599614 dB`，ERB 为 `1.119363 → 1.052854 dB`，对侧高频为
  `3.993006 → 3.659349 dB`，ILD proxy 为 `0.651290 → 0.621770 dB`；分别改善
  `8.00% / 5.94% / 8.36% / 4.53%`。短预算 ILD 退化在正式训练中已反转。
- 完整 validation raw 指标：44 人、767 个纯插值方向、`31,250,648` 个逐频点
  样本上，MCA/v2/v3 MAE 为 `3.322289 / 2.807803 / 2.581873 dB`，v3 RMSE 为
  `3.935166 dB`。v3 MAE 相对 v2 改善 `8.05%`、相对 MCA 改善 `22.29%`，并在
  44/44 人上优于 v2。
- 全空间严格 HRIR ILD 诊断：v2/v3 MAE 为
  `0.641157 / 0.621980 dB`，v3 改善 `2.99%`，35/44 人改善。该指标直接使用
  reference HRIR 能量，但采用全空间口径；后续仍需 MATLAB 重建评估水平面 ILD。
- 产物：权重比较位于 `results/sonicom_mlp_cnn_q26_v3_weight_pilot/`，正式曲线
  与摘要位于 `results/sonicom_mlp_cnn_q26_v3_formal/`，锁定配置为
  `configs/experiments/sonicom_mlp_cnn_q26_v3_locked.json`。checkpoint 和完整
  per-subject 评估继续保存在 Git 忽略的 `artifacts/`。W&B run ID 为
  `5mjq0y51`，当前为本地 offline，未经用户明确授权不上传。
- 运行问题：第一次完整评估在 44 人计算结束后触发旧 HUTUBS 固定参考值断言；
  已把断言限定到 `hutubs_residual_v1_n03` 并成功重跑，训练和 checkpoint 无需
  重跑。此次恢复也先检查了后台进程与现有输出，未启动重复训练。
- 结论与下一步：冻结 MLP、只训练频率 CNN 已同时改善 raw residual、四项固定
  proxy 和全空间严格 ILD，v3 可进入最终 reconstruction 复核。下一步输出 44 人
  v3 residual，按 MATLAB 最终口径计算 `AKerbError`、对侧高频、水平面严格
  HRIR ILD，并生成 44 人 HRTF 总览；test 读取数继续保持为 0。

## 2026-08-01：SONICOM Q26 MLP v1/v2 严格 validation 重建评估

- 工作目标：只在锁定的 44 名 validation 被试上，将正式 v1/v2 residual 回填
  到 MCA 幅度，按最终 HRTF/HRIR 重建口径比较 `AKerbError`、对侧高频误差和
  水平面严格 ILD；test 继续不导出、不读取。
- 推理实现：新增 `predict_sonicom_residuals.py`，强制只接受 split=`val` 的
  HDF5，原子写出完整 `2×793×463` residual。v1 epoch 19 与 v2 epoch 10 的
  44 人 GPU 推理分别用时 `8.661 s` 和 `8.432 s`，RTX 5060、AMP、100,225
  参数；两套预测位于已忽略的 `artifacts/reconstruction/`。
- 严格重建：新增 `evaluate_sonicom_validation_reconstruction.m`。在
  `50–20000 Hz` 把 residual 加到 MCA log magnitude，保留 MCA 相位和频段外
  复数谱，转换双边频谱后 IFFT 并裁剪到原始 256-sample HRIR。原始 validation
  SOFA 只用于 reference HRIR，未重新运行 MCA。
- 指标口径：ERB 使用与 HUTUBS 论文复现一致的 `AKerbError`，范围
  `50–22050 Hz`；全空间和耳特异对侧 25° 区域均只纳入 767 个纯插值方向并按
  SONICOM solid-angle weight 加权。对侧高频为 `10–20 kHz` 对侧开放半球的
  面积加权幅度 MAE。严格 ILD 为水平面纯插值方向上完整 HRIR 能量比的 MAE。
- 全空间 ERB：MCA/v1/v2 均值为
  `1.095738 / 0.987171 / 0.934826 dB`；v2 相对 MCA 改善 `14.69%`，相对 v1
  再改善 `5.30%`。v2 在 44/44 人上优于 v1。
- 对侧 25° ERB：MCA/v1/v2 为
  `1.763508 / 1.520947 / 1.449139 dB`；v2 相对 MCA 改善 `17.83%`，相对 v1
  再改善 `4.72%`。v2 在 44/44 人上优于 v1。
- 对侧高频：MCA/v1/v2 为
  `4.749208 / 3.989510 / 3.965865 dB`；v2 相对 MCA 改善 `16.49%`，相对 v1
  再改善 `0.59%`。v2 在 33/44 人上优于 v1，说明正式 v2 保持了 v1 的主要
  高频收益，但增量较小。
- 水平面严格 ILD：MCA/v1/v2 为
  `0.830014 / 0.741741 / 0.653461 dB`；v2 相对 MCA 改善 `21.27%`，相对 v1
  再改善 `11.90%`，37/44 人改善。该严格结果与训练期 ILD proxy 的方向一致。
- 质量检查：44 个 per-subject 行、528 个 method-metric 长表行和全部数值有限；
  HDF5 reference ILD 与原始 SOFA HRIR 重算值的最大差异为 `9.54e-7 dB`。
  单人 smoke 先发现并修正 HDF5 行/列向量隐式扩展问题，正式运行无异常。
- 产物：配置位于
  `configs/experiments/sonicom_mlp_q26_v2_strict_validation.json`；完整 CSV、
  JSON、44 人指标曲线、聚合柱状图和对侧 HRTF 总览位于
  `results/sonicom_mlp_q26_v2_strict_validation/`。响应流在正式 MATLAB 运行
  中途断开，但后台进程正常完成并写出 `status=completed`，没有重跑或并发写入。
- 结论与下一步：严格 reconstruction 指标确认 v2 的主要增益集中在 ERB 与
  ILD，高频相对 v1 仅小幅改善但没有总体退化。SONICOM v2 可视为 validation
  阶段锁定；在解封 44 名 test 前，应先决定是否把当前 v2 作为最终 MLP 基线，
  或继续在同一开发划分上训练 SONICOM MLP+CNN。

## 2026-07-31：SONICOM Q26 residual MLP v2 权重选择与正式锁定

- 工作目标：在不读取 44 名 test 的前提下，用少量固定预算选择 SONICOM v2
  的 ERB、对侧高频和 ILD spectral proxy 权重，并完成正式训练与完整
  validation raw residual 评估。
- 选择规则：所有候选必须满足 residual MAE 相对初始 v1 退化不超过 `0.5%`；
  首先最大化 ERB、对侧高频、ILD 三项相对改善率的最小值，再依次比较三项平均
  改善、residual MAE 和较小总权重。该规则在候选训练前锁定，避免根据结果临时
  改口径。
- 候选设置：四组均使用 3 epoch × 120 step、32 个固定 validation block、
  32 个插值方向、完整 463 频点、方向面积加权损失、AdamW `3e-4`、AMP 和 seed
  `20260731`。权重分别为 balanced `0.50/0.25/0.25`、erb_heavy
  `0.75/0.25/0.25`、ild_heavy `0.50/0.25/0.50`、erb_ild_heavy
  `0.75/0.25/0.50`。
- 候选结果：四组的 ERB/高频/ILD 最小相对改善率依次为
  `0.5260% / 0.5305% / 0.4986% / 0.5050%`；三项平均改善率依次为
  `3.0861% / 3.1430% / 3.5051% / 3.5107%`。根据预定的首要 maximin 规则，
  选择 `erb_heavy = 0.75/0.25/0.25`，而不是用平均值事后偏向更高 ILD 权重。
- 正式训练：10 epoch × 500 step、96 个固定 validation block、cosine schedule，
  其余采样和优化设置与 pilot 相同。RTX 5060 上用时 `296.339 s`，峰值 CUDA
  allocated memory `144.292 MiB`，最佳 epoch 为 10。
- 固定 validation：初始 v1 的 residual/ERB/高频/ILD 为
  `2.847674 / 1.175123 / 4.027407 / 0.738389 dB`；epoch 10 为
  `2.825701 / 1.119363 / 3.993006 / 0.651290 dB`，分别改善
  `0.77% / 4.74% / 0.85% / 11.80%`，四项同时改善。
- 完整 raw validation：遍历 44 人、767 个纯插值方向和 `31,250,648` 个样本。
  v2 MAE/RMSE 为 `2.807803 / 4.210356 dB`，锁定 v1 为
  `2.825515 / 4.218639 dB`，分别改善 `0.63% / 0.20%`；相对 MCA
  `3.322289 dB`，v2 MAE 改善 `15.49%`。
- 锁定产物：正式 checkpoint 为
  `artifacts/training/sonicom_mlp_q26_v2_formal_erb075_ild025/best.pt`；锁定配置为
  `configs/experiments/sonicom_mlp_q26_v2_locked.json`；权重比较和正式精选结果
  分别位于 `results/sonicom_mlp_q26_v2_weight_pilot/` 与
  `results/sonicom_mlp_q26_v2_formal/`。checkpoint 和完整运行状态继续由
  `artifacts/` 忽略。
- W&B 状态：balanced run `buhhtjwi` 已在线。三个新增 pilot run
  `n2jlw45k / vpwlyqc8 / q5g8hg2h` 和正式 run `h7kmgoxc` 已完整保存在本地
  offline 目录。由于上传可能包含配置与本机路径信息，当前未获得明确上传授权，
  因此没有绕过限制进行同步。
- 防泄漏与结论：处理目录仍只有 262 train + 44 validation，test 读取数为 0。
  v2 已按 validation 锁定；下一步是在 validation 上回填 residual，执行严格
  重建 ERB magnitude error、对侧高频误差与 HRIR ILD 评估。严格指标完成前
  不导出 test。

## 2026-07-31：SONICOM Q26 residual MLP v2 感知损失 smoke

- 工作目标：以锁定的 v1 epoch 19 checkpoint 初始化 MLP v2，验证 SONICOM
  双耳完整频谱、ERB、对侧高频、ILD spectral proxy、面积权重和 W&B 链路；
  开发阶段继续不读取 44 名 test。
- 加权策略：`BinauralSpectrumSampler` 新增 interpolation-only 模式，在 767
  个合格方向中均匀无放回抽样；损失内部使用第六列 solid-angle weight。
  residual SmoothL1 和 MAE 同步改为可选方向加权，SONICOM 启用该选项，避免
  方向按面积抽样后又在损失内加权造成双重面积权重。HUTUBS 默认行为不变。
- 损失与配置：模型仍为 100,225 参数；每 batch 包含双耳、32 方向和全部 463
  频点，共 29,632 个样本；总损失为面积加权 residual SmoothL1，加
  `0.50 × ERB + 0.25 × 对侧高频 + 0.25 × ILD proxy`，各感知项除以
  target std。smoke 为 3 epoch × 120 step、32 个固定 validation block、
  AdamW `3e-4`、AMP、seed `20260731`。
- 实现与验证：v2 训练器新增 W&B、固定 validation sampler、epoch 0 的 v1
  基准、CUDA 峰值和完整报告。合成可变方向测试、Python compileall 和真实
  `2×8×463` GPU 前向/反向均通过；真实梯度 smoke 四项损失有限，峰值显存
  `51.89 MiB`。
- smoke 结果：正式短程耗时 `33.146 s`，峰值显存 `144.292 MiB`，最佳 epoch
  3。固定 validation 从 v1 到 v2 的 residual/ERB/高频/ILD 为
  `2.853286→2.849955 / 1.170491→1.152948 / 4.053157→4.031838 /
  0.758558→0.703688 dB`，分别改善 `0.12% / 1.50% / 0.53% / 7.23%`；
  复合 loss 改善 `0.85%`。
- 完整 raw validation：遍历 44 人、767 插值方向和 `31,250,648` 个样本，
  v2 MAE/RMSE 为 `2.824472 / 4.229951 dB`；v1 为
  `2.825515 / 4.218639 dB`。MAE 略改善 `0.037%`，RMSE 小幅波动，说明短程
  感知微调没有以明显 raw residual 退化换取代理指标。
- W&B 与产物：run ID `buhhtjwi`，地址
  `https://wandb.ai/luyoung/mcar-sonicom/runs/buhhtjwi`；配置位于
  `configs/experiments/sonicom_mlp_q26_v2_smoke.json`，精选 history/summary
  位于 `results/sonicom_mlp_q26_v2_smoke/`，checkpoint 继续由 artifacts 忽略。
- 结论与下一步：默认 HUTUBS v2 权重在 SONICOM 上方向正确，四项 validation
  代理均改善。下一步只在 train/validation 上比较少量权重候选与正式预算，
  优先保留 ILD 权重、检查更高 ERB 权重是否扩大听觉收益且不损害 raw MAE；
  test 继续不导出。

## 2026-07-31：SONICOM Q26 residual MLP v1 正式训练与预算锁定

- 工作目标：比较 12/20 epoch 正式预算，以完整 validation 的 767 点面积加权
  residual MAE 锁定 SONICOM MLP v1；44 名 test 继续未导出、未读取。
- 公共设置：262 train、44 validation；ResidualMLP 宽度 128、3 个 residual
  block、100,225 参数；每 epoch 600 个训练 block、96 个固定 validation
  block，每 block 8192 样本；solid-angle、interpolation-only、AdamW、初始
  学习率 `1e-3`、cosine schedule、AMP、seed `20260731`。
- 12-epoch 候选：耗时 `346.345 s`，抽样 validation 最佳为 epoch 12，MAE
  `2.912809 dB`。完整 validation MAE/RMSE 为
  `2.872643 / 4.265143 dB`，相对 MCA MAE 改善 `13.53%`；W&B run ID
  `6wi22jix`。
- 20-epoch 候选：耗时 `602.747 s`，抽样 validation 最佳为 epoch 19，MAE
  `2.866083 dB`，epoch 20 为 `2.866198 dB`。完整 validation MAE/RMSE 为
  `2.825515 / 4.218639 dB`，相对 MCA `3.322289 / 5.029170 dB` 的 MAE
  改善 `14.95%`；W&B run ID `ek1nvkhh`。
- 选择结论：20-epoch 候选的完整 validation MAE 相对 12-epoch 再降低
  `1.64%`，因此锁定 `sonicom_mlp_q26_v1_e20/best.pt` 的 epoch 19。后期曲线
  已基本平台化，暂不继续增加 epoch。
- 中断恢复：首次启动 20-epoch 命令时界面中断；检查确认无 Python/W&B 进程、
  输出目录或半成品 checkpoint，随后从头安全重跑，不存在 checkpoint 混合。
- 产物：预算配置和锁定配置位于 `configs/experiments/`；比较表、两条 history
  和摘要位于 `results/sonicom_mlp_q26_v1_budget_comparison/`。checkpoint 与
  W&B 本地状态继续由 `artifacts/` 忽略。
- 下一步：以 epoch 19 checkpoint 初始化 SONICOM MLP v2，在 train/validation
  上调节 ERB、对侧高频和 ILD 感知损失；v2 锁定前继续不导出 test。

## 2026-07-31：SONICOM Q26 residual MLP smoke training

- 工作目标：在不读取锁定 test 的前提下，验证 SONICOM 全量开发数据能否进入
  现有 100,225 参数轻量 ResidualMLP，并打通 RTX 5060、自动混合精度、球面
  面积采样、纯插值 mask、固定 validation、完整 validation 和 W&B 在线记录。
- 兼容改动：`ResidualBlockSampler` 新增可选 `solid_angle` 方向抽样和
  `interpolation_only` 模式，默认仍为 HUTUBS 原有的 uniform + 全方向。
  SONICOM 训练按 `direction_features[:, 5]` 的正面积权重，从 767 个
  interpolation directions 无放回抽样，不让 26 个稀疏输入点稀释训练指标。
- 训练器：`train_mlp_v1` 新增 W&B 参数、固定 validation seed、epoch 耗时、
  CUDA 峰值、训练报告和 W&B run 元数据。每个 epoch 都从同一 validation seed
  重建 sampler，保证候选 epoch 使用相同随机 validation blocks。
- smoke 配置：262 train、44 validation、0 test；3 epoch × 120 train step，
  每步 `64 directions × 128 frequencies = 8192` 样本；每 epoch 64 个固定
  validation block；宽度 128、3 个 residual block、AdamW、初始学习率
  `1e-3`、AMP，seed `20260731`。
- 运行结果：RTX 5060 上耗时 `24.095 s`，峰值 CUDA allocated memory
  `54.090 MiB`，无异常退出。抽样 validation 的 MCA MAE 固定为
  `3.374041 dB`；模型在 epoch 1/2/3 为
  `3.235233 / 3.172290 / 3.138882 dB`，最佳 epoch 3 改善 `6.97%`。
- 完整 validation：扩展 `evaluate_residual_mlp` 支持 interpolation-only 和
  solid-angle weighting。遍历 44 名 validation、767 方向、双耳和 463
  频点，共 `31,250,648` 个样本；MCA 的 MAE/RMSE 为
  `3.322289 / 5.029170 dB`，模型为 `3.100075 / 4.580354 dB`，MAE 改善
  `6.69%`。
- W&B：在线同步完成，run ID `hmekqxio`，地址为
  `https://wandb.ai/luyoung/mcar-sonicom/runs/hmekqxio`。第一次在受限沙箱
  内初始化时网络访问被拒，尚未开始训练；随后使用受控网络权限重跑成功，不存在
  checkpoint 混用。
- 防泄漏与产物：复查 processed 目录仍为 306 个 train/validation HDF5，
  锁定 test 文件为 0。checkpoint、W&B 本地状态和完整配置位于已忽略的
  `artifacts/training/sonicom_mlp_q26_smoke_v1/`；精选 history 与摘要位于
  `results/sonicom_mlp_q26_smoke_v1/`。
- 结论与下一步：面积加权 SONICOM-only MLP 链路已经可训练，且极短训练已取得
  明确 validation 改善，但 smoke 预算不足以作为最终结论。下一步应保持同一
  数据口径，先比较 12/20 epoch 的正式 MLP v1 学习曲线并以完整 validation
  锁定 checkpoint；再在不读取 test 的情况下进入感知损失 v2。

## 2026-07-31：SONICOM Q26 正式 MCA residual 数据导出

- 工作目标：使用 pilot 锁定的 `SONICOM-Q26-v1`、三阶球谐和
  Tikhonov epsilon `0.01`，生成 SONICOM-only 网络训练所需的完整
  train/validation MCA residual 数据；继续禁止读取 44 名锁定 test 被试。
- 实际执行：从固定 split 读取全部非 test ID，以 6 个 MATLAB process worker
  调用 `mcar.export_sonicom_residual_dataset`。导出器使用原子写入和完成文件
  跳过机制；从首个到最后一个文件完成历时约 `733 s`。
- 导出结果：306/306 个逐被试 HDF5 完成，其中 `262 train / 44 validation /
  0 test`；无缺失、无意外 ID、无 test 泄漏。总大小 `4,254,380,898 bytes`
  （约 `3.96 GiB`），每名被试为 2 耳 × 793 方向 × 463 频点，共
  `734,318` 个 residual 样本。
- 全量验证：使用 `validate_residual_hdf5 --summary-only
  --require-strict-ild` 遍历 306 个文件和 `224,701,308` 个样本。所有必需
  dataset、有限值、动态方向尺寸、26 个稀疏索引、767 点纯插值 mask、方向
  权重、完整频谱 strict-ILD 元数据和完成标记均通过；`reference - MCA =
  target_residual` 的最大恒等误差为 `0 dB`。
- train-only 统计：`compute_training_statistics` 仅纳入 262 名 train 的
  `192,391,316` 个样本；target residual 的 mean/std/mean absolute 为
  `-0.062128 / 4.954315 / 3.250640 dB`，范围为
  `-87.844574～82.283562 dB`。完整统计保存于已忽略数据目录的
  `training_statistics.json`，实验配置已记录其相对路径。
- 产物策略：大型 HDF5 及完整 train ID 统计继续位于
  `data/processed/sonicom_residual_q26_v1/` 并由 Git 忽略；可提交的运行摘要
  位于 `results/sonicom_data_preparation/full_export_v1/summary.json`。
- 结论与下一步：SONICOM-only residual 网络所需的开发数据已经就绪且无
  test 泄漏。下一步先对现有轻量 residual MLP 做最小改动的 SONICOM smoke
  training，确认 793 点面积权重、动态方向 sampler、validation 全量评估和
  W&B 日志，再锁定正式训练预算；test 继续保持未导出。

## 2026-07-31：SONICOM Q26 MCA pilot 与 Tikhonov 参数锁定

- 工作目标：在不读取 44 名锁定 test 被试的前提下，验证 SONICOM-Q26-v1
  能否完整运行 SUpDEq + SH + MCA，建立 793/767 双口径 HDF5，并根据
  validation 选择无权三阶球谐最小二乘的 Tikhonov epsilon。
- 导出实现：新增
  `matlab/+mcar/export_sonicom_residual_dataset.m`。每名被试直接对
  `(793, 2, 256)` HRIR 做 1024 点 FFT，从固定 26 个真实测量索引取得稀疏
  HRTF，以实际 Q26 方位角/余纬度做三阶球谐变换，再在原始 793 点上运行 MCA。
  头半径使用所有 SOFA 共有的 nominal `0.09 m` ReceiverPosition，频率范围
  为 `50 Hz～20 kHz` 共 463 点。
- 防泄漏：导出器默认 `allowTest=false`，在初始化 SUpDEq 或读取 SOFA 前检查
  固定 split；用 P0003 实测保护逻辑，得到预期的 locked-test 错误并通过断言。
  正式开发命令只能处理 train/validation，最终评估必须显式启用 test。
- HDF5 schema 2.0：每名被试保存
  `mca/reference/correction/target_residual`，Python 布局均为
  `[2, 793, 463]`；同时保存 793 点面积权重、26 个稀疏索引、767 点纯插值
  mask、MCA 选中频点相位、50 个频带外复频点、参考严格 HRIR-ILD 和 256 点
  HRIR 裁剪长度。每文件包含 `734,318` 个 residual 样本。
- Python 兼容：`src/mcar/data.py`、训练统计和 HDF5 验证器从固定 900 方向改为
  动态方向数，文件发现从 HUTUBS 专用 `pp*/n*.h5` 泛化为 `*/*.h5`；已用旧
  HUTUBS `pp1/n03.h5` 回归验证 900 点路径不变。SONICOM 的方向特征第六列为
  `normalized_solid_angle_weight`，训练统计从 HDF5 属性读取名称，不再硬编码
  `fliege_weight`。
- pilot 被试：P0002/P0243/P0238 为 train，P0001/P0277/P0242 为 validation，
  分别覆盖 `reference_eq_001/006/008`，test 为 0。首先用 P0002/P0001
  完成真实双被试 smoke test，约 10 秒/人，平均绝对 residual 分别为
  `3.311601/3.290179 dB`；纯插值 767 点为 `3.331109/3.310949 dB`，
  证明全 793 点会被输入方向轻微乐观化。
- 参数扫描：固定 Q26 和其余 MCA 参数，扫描 epsilon
  `0 / 1e-8 / 1e-6 / 1e-4 / 1e-2 / 3e-2 / 1e-1 / 3e-1 / 1`。选择规则在
  汇总前固定为“validation 767 点 ERB proxy MAE 最低；再依次以 residual MAE、
  严格 HRIR-ILD MAE 和较小 epsilon 打破并列”。
- validation 结果：epsilon `0` 的 767 点 residual/ERB/对侧高频/严格 ILD
  为 `3.289591 / 1.275841 / 4.779132 / 0.977043 dB`；`0.01` 为
  `3.276051 / 1.264245 / 4.772924 / 0.986837 dB`；`0.03` 为
  `3.272364 / 1.265599 / 4.786129 / 1.035681 dB`；`0.1` 已全面退化至
  `3.380890 / 1.404554 / 5.006937 / 1.340158 dB`。因此 `0.01` 是 ERB
  内部最优点并被锁定；相对无正则 ERB 改善约 `0.91%`，代价是严格 ILD
  增加约 `1.00%`。
- 全量 pilot 验证：9 档 × 6 人共 54 个 HDF5、`39,653,172` 个样本全部通过
  strict-ILD、有限值、频率覆盖、方向权重、26/767 mask 互补、残差恒等和
  complete 标记检查；最大 MCA correction 恒等误差
  `7.6294e-06 dB`，所有 residual 恒等误差为 0。全量 pilot HDF5 约
  `713.28 MiB`，由 `data/processed/*` 忽略。锁定 epsilon 的 3 名 train
  被试可正常生成训练统计；逐点 sampler 输出 `(512, 7)`，严格 ILD 双耳 sampler
  输出 `(2, 4, 463, 7)`，证明 793 点数据已可直接进入现有 MLP/CNN 管线。
- 结果与配置：评估脚本为
  `src/mcar/evaluation/evaluate_sonicom_tikhonov_pilot.py`；逐被试 CSV、
  聚合 CSV 和 JSON 位于
  `results/sonicom_data_preparation/tikhonov_pilot_v1/`；正式锁定参数位于
  `configs/experiments/sonicom_q26_residual_v1.json`；完整命令见
  `experiments/sonicom_data/README.md`。
- 结论与下一步：SONICOM Q26 MCA residual 链路已经打通，epsilon 锁定为
  `0.01`。下一步用 6 个 MATLAB worker 导出 262 train + 44 validation
  共 306 人，继续保持 test 未导出；全量 HDF5 通过后只用 262 名 train 计算
  归一化统计，再开始 SONICOM-only residual MLP 基线。

## 2026-07-31：SONICOM-Q26-v1 几何配置与固定被试划分

- 工作目标：在正式 350 人 SONICOM 测量队列上建立不会伪造缺失方向、可供
  MCA residual 导出的稀疏输入网格、参考方向权重和全新锁定被试划分。
- 网格审计：350 个 SOFA 共用同一个 793 点 `SourcePosition`；网格包含
  `-45/-30/-20/-10/0/10/20/30/45/60/75°` 共 11 个水平圆环，每环
  72 个、方位角间隔 5°，另有一个 90° 北极点。实测区域为仰角
  `-45°～90°`，覆盖球面立体角 `10.726068 sr`，不存在南极及
  `-45°` 以下方向。
- Lebedev 兼容性：将 SUpDEq Lebedev `N=3` 的 26 个目标方向直接映射到
  SONICOM，只有 17 个精确重合，最近邻只有 25 个唯一点；平均最近角误差
  `3.350581°`，最大误差来自南极点并达到 `45°`。因此不采用直接最近邻，
  也不通过高阶球谐外推伪造缺失球冠。
- Q26 方法：新增
  `src/mcar/data_tools/prepare_sonicom_configs.py`。算法固定包含北极点，
  遍历另一个中垂面种子，并贪心选择 12 对真实左右镜像方向；主目标为最大化
  最小大圆距离，并以正则化三阶实球谐 Gram 矩阵的 log determinant 打破并列。
  选点只使用几何坐标，不读取 HRIR/HRTF 数值。
- Q26 结果：`SONICOM-Q26-v1` 含 26 个唯一实测方向且严格左右对称；最小
  点间角 `31.915839°`，对全部 793 点的最近覆盖距离平均 `15.080257°`、
  95 百分位 `24.814217°`、最大 `33.108876°`。三阶实球谐设计矩阵为
  `26 × 16`、秩 16、条件数 `2.476614`，通过预设的秩和条件数检查。
- 方向权重：依据各仰角圆环相邻中点形成的球面单元，在实测
  `-45°～90°` 球冠内计算面积权重并归一化至 1。配置同时标记 26 个稀疏
  输入方向和 767 个纯插值评估方向；正式指标以 767 点为主、793 点为辅助。
- 被试划分：使用固定 seed `20260731`，首先按 11 个
  `Free Field EQ File` 分层并做整数配额，再在每个 EQ 组内用由 seed 和
  EQ 名称共同派生的稳定 SHA-256 seed 洗牌。结果严格为
  `262 train / 44 validation / 44 test`，无交叉、遗漏或重复。仅含 2 人的
  `reference_eq_005` 无法覆盖三个 split，按比例均保留在 train；其余 10 个
  EQ 批次均覆盖 validation 和 test。
- 元数据审计：train/validation/test 已知年龄人数为 `148/21/21`；出生性别
  缺失人数为 `114/24/24`。年龄、出生性别、族裔、EQ 文件和 subject ID
  只用于分层或审计，不作为首版网络输入；后续归一化统计只能读取 train。
- 输出配置：`configs/data/sonicom_sparse_grid_q26_v1.csv` 保存 Q26 索引、
  坐标、镜像索引和选点角色；`sonicom_reference_grid_v1.csv` 保存 793 点
  坐标、面积权重和 mask；`sonicom_subject_split_v1.csv` 保存固定划分；
  `sonicom_preparation_report_v1.json` 保存源网格 SHA-256、全部几何指标和
  分层分布。
- 验证：350 个 SOFA 网格逐文件一致性检查、离线合成网格测试、Q26 唯一性与
  镜像闭包、球谐满秩与条件数、面积权重正值与和为 1、split 行列总数检查、
  Python `compileall` 和配置实际生成均通过。
- 结论与下一步：SONICOM 不能作为严格 Lebedev 论文复现的直接替代，但
  `SONICOM-Q26-v1` 可作为测量数据上的受控扩展实验。下一步实现独立 SONICOM
  MCA residual 导出入口，使用实际 Q26 坐标、三阶无权最小二乘球谐变换、
  nominal `0.09 m` 头半径和待由 validation pilot 选择的 Tikhonov 参数；
  先运行少量 train/validation 被试，不读取 44 名锁定 test。

## 2026-07-30：SONICOM 正式训练队列下载器与全队列可用性检查

- 工作目标：把 SONICOM 官网完整数据集整理为可复现、可断点续传且不会因官网
  元数据变化而静默漂移的训练数据下载流程；本阶段只实现与验证下载，不改动既有
  HUTUBS 锁定实验。
- 数据集与版本：数据根目录为
  `https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET`；使用官网当前
  `metadata.csv` 和日期固定的
  `important_information/Outliers_2026-05-06.csv`。当前快照包含 372 条
  被试元数据，其中 364 条 `HRTF == TRUE`。
- 选择口径：仅保留 `HRTF == TRUE`、不在官方 HRTF 异常清单且自由场均衡
  文件未标记为 `EQ File Corrupted` 的测量被试。12 名官方异常被试、8 名
  无 HRTF 被试和 2 名自由场均衡损坏被试合计排除 22 名，冻结后的干净队列为
  350 名。脚本在官网元数据行数或队列人数变化时默认停止，只有人工复核后才允许
  使用 `--allow-metadata-drift`。
- 文件版本：每名被试只下载
  `PXXXX/HRTF/HRTF/44kHz/PXXXX_FreeFieldCompMinPhase_44kHz.sofa`。该版本
  采用 44.1 kHz 采样率，已进行 5 ms 截窗和最小相位自由场补偿，同时保留个体
  耳间时间差（ITD）；不下载去 ITD 版本、原始 50 ms 版本、合成 HRTF、扫描网格
  和其他重复采样率版本。
- 实现：新增 `src/mcar/data_tools/download_sonicom.py`，仅依赖 Python
  标准库，支持并发请求、网络重试、HTTP Range 断点续传、`.part` 临时文件、
  完成后的原子替换、远端 `Content-Length` 校验、重复运行跳过、pilot 被试
  子集、只下载元数据和全队列只读检查。
- 可追溯输出：正式运行会把官方元数据和异常说明原样保存，并生成
  `manifests/clean_subjects.csv`、`manifests/excluded_subjects.csv` 和
  `manifests/selection_report.json`；报告中记录选择规则及所有元数据文件的
  SHA-256 摘要。SOFA 默认写入
  `data/HRTF/sonicom_measured_ffcmp_minphase_44k1/subjects/`，整个大型数据
  目录继续由 Git 忽略。
- 全队列网络检查：实际执行
  `D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.download_sonicom
  --dry-run --workers 8`；350/350 个目标 SOFA 的 HEAD 请求全部成功，远端
  总量为 `872.82 MiB`，耗时约 `162.8 s`，且 dry run 未写入数据文件。
- 真实下载：先在已忽略的 `artifacts/download_smoke/sonicom_p0001/` 完成
  P0001 下载及重复运行跳过检查，再向正式目录下载全队列。首轮完成 347/350 个，
  并保留 P0310 的零字节 `.part`；续传命令只指定 P0310、P0312、P0313，三者
  全部下载成功，最终得到 350/350 个 SOFA、无残留 `.part`，总量
  `872.82 MiB`。
- 全量内容审计：manifest 中 350 名被试与 `subjects/` 中 350 个文件一一对应，
  无缺失或多余文件。所有文件的 `Data.IR` 形状均为 `(793, 2, 256)`、采样率
  均为 `44100 Hz`、SOFA 约定均为 `SimpleFreeFieldHRIR`，所有脉冲响应数值
  有限，且 350 名被试的 793 点 `SourcePosition` 方向网格完全一致。
- 验证：Python `compileall`、命令行 `--help`、离线选择规则测试、全队列
  网络检查、真实单文件下载及重复运行、全量下载和 350 文件内容审计均通过。
  正式数据位于
  `data/HRTF/sonicom_measured_ffcmp_minphase_44k1/`，并由 Git 忽略。
- 结论与下一步：下载链路、冻结队列和正式训练数据已经就绪，无数据侧阻塞。
  下一步先检查 MCA 输入所需的稀疏方向是否能从 SONICOM 统一网格直接抽取，
  再按自由场均衡文件或其他可用属性进行 subject-wise 分层，建立
  `262 train / 44 validation / 44 test` 的新数据划分，原 HUTUBS test 结论
  保持锁定。

## 2026-07-30：项目文档英文缩写首次出现规范化

- 工作目标：修复项目文档中英文缩写首次出现时缺少全称的问题，降低说明页、
  实验日志和技术报告的阅读门槛。
- 修改范围：检查全部受 Git 跟踪的 Markdown 文档及结果目录中的文本报告；
  不修改命令、文件名、类名、数据和实验结论。
- 统一规则：同一文档内首次自然语言出现采用“中文名称（English Full Name，
  缩写）”或“English Full Name（缩写）”；HUTUBS、AXD、KU100 等无正式首字母
  展开方式的数据集或设备名称按专名保留。
- 核对来源：MCA 使用论文题名中的 `Magnitude-Corrected and Time-Aligned
  Interpolation`；SUpDEq 使用官方文档中的 `Spatial Upsampling by Directional
  Equalization`；SOFA 使用规范名称 `Spatially Oriented Format for Acoustics`。
- 公式兼容性：将 v3.1 报告中两处 `\[...\]` 显示公式同步改为项目约定的
  `$$\begin{aligned}...\end{aligned}$$` 形式，公式内容未变。
- 验证结果：缩写首次出现自动审计、`git diff --check` 和报告公式分隔符检查均
  通过；修改仅涉及文档。

## 2026-07-27：MLP + CNN v3.1 严格 HRIR-ILD 对齐微调与锁定测试

- 实验目标：修复 v3 训练 ILD proxy 与最终严格 HRIR 能量 ILD 口径不一致的问题；不扩大网络，只在 train/validation 上实现可微严格 ILD、完成受控选型，再对锁定的 12 名 test 被试进行一次最终评估。
- 数据集与划分：HUTUBS simulated，Lebedev `N=3` MCA residual；固定 72 train / 12 validation / 12 test subject-wise split。为 train/validation 84 个 HDF5 导出 schema 1.1，新增 MCA 选中频点相位、50 个频带外复频点、频点索引、参考 HRIR ILD、513 点单边谱长度和 256 点 HRIR 裁剪长度。开发和选型阶段没有读取 test。
- 数据完整性：`python -m mcar.data_tools.validate_residual_hdf5 ... --require-strict-ild` 验证 84 个文件、`70,005,600` 个样本、72/12 split、完整 513 点频率覆盖、有限值和零 residual 恒等误差 0，全部通过。
- 实现：`src/mcar/losses.py` 用预测幅度、原 MCA 相位和未修改的频带外复谱重建 513 点单边谱，镜像为 1024 点双边谱，IFFT 后裁剪 256 点 HRIR，再计算双耳能量 ILD MAE；全链路支持 autograd。训练器新增 `--ild-loss-mode strict_hrir`、`--initial-cnn-checkpoint`、严格/旧 proxy 双日志，以及 `best.pt` 和 `best_strict_ild.pt` 双 checkpoint。评估器新增 `--strict-ild` 完整方向穷举评估。
- 测试：合成严格 ILD 单元测试得到零误差 `0`、扰动误差 `0.285678 dB`、最大梯度 `0.245291`；真实 pp91 CUDA smoke test 的 zero-init identity error 为 0，严格 ILD 与旧 proxy 不同，第二步共有 62 个 CNN 参数梯度张量非零。
- 候选实验：从 v2 零初始化 CNN 的严格 ILD 权重 0.25 在完整 validation 上得到 `0.660663 dB`，未超过 v2；权重 1.0 的严格-ILD checkpoint 得到 `0.655195 dB`，但 raw residual MAE 升至 `2.110205 dB`。因此最终改用原 v3 epoch 9 为初始化，以 `1e-4` 学习率和严格 ILD 权重 1.0 微调冻结 MLP 后的 CNN。
- 正式训练：6 epoch × 500 step，32 directions/block，训练参数 74,402、总参数 174,627，耗时 `286.389 s`，0 个跳过 step，峰值 CUDA allocated memory `221.963 MiB`。W&B run：`https://wandb.ai/luyoung/mcar-mlp-cnn-v31/runs/yhl3z70n`。
- validation 锁定：查看 test 前以完整 validation 锁定总损失最优 epoch 3。v2 / 原 v3 / v3.1 的 raw residual MAE 为 `2.139782 / 2.072657 / 2.078053 dB`，严格 ILD MAE 为 `0.658844 / 0.656999 / 0.654040 dB`。v3.1 相对 v2 的 validation 严格 ILD 改善 `0.729%`。
- 锁定 raw test：共 `10,000,800` 样本。MCA / v2 / v3.1 MAE 为 `2.600671 / 2.168373 / 2.092767 dB`，RMSE 为 `4.186115 / 3.552562 / 3.448699 dB`。v3.1 相对 v2 MAE 改善 `3.487%`、相对 MCA 改善 `19.530%`，12/12 被试优于 v2；相对原 v3 `2.091041 dB` 仅回退约 `0.083%`。
- 严格重建 test：MCA 的全空间 ERB / 对侧 25° ERB / 对侧 `>10 kHz` / 水平面 ILD MAE 为 `0.802741 / 1.860286 / 4.258336 / 0.885425 dB`；v3.1 为 `0.585398 / 1.310334 / 3.618947 / 0.652495 dB`，相对 MCA 改善 `27.08% / 29.56% / 15.01% / 26.31%`，四项均为 12/12 被试改善。
- v3.1 与 v3：严格 ILD 由 `0.661566` 降到 `0.652495 dB`，改善 `1.371%`，8/12 被试改善；对侧高频改善 `0.033%`；全空间 ERB 和对侧 25° ERB 分别小幅回退 `0.138%` 和 `0.454%`。与 v2 相比，前三项仍改善 `4.269% / 2.219% / 2.726%` 且均为 12/12，但 ILD 仍高 `0.893%`，只有 4/12 优于 v2。
- 重建质量：12 被试最大左右相位误差 `6.12e-16 / 5.99e-16 rad`，最大幅度恒等误差均为 `7.11e-15 dB`，全部通过断言。已生成 12 张逐被试图、12 被试 HRTF 总览、指标总览、v1/v2/v3/v3.1 总对比与逐被试 ILD 对比，并完成视觉检查。
- 输出位置：本机 checkpoint 与全量产物位于 `artifacts/training/mlp_cnn_n03_v31_finetune_v3_strict_ild_wandb_online/`、`artifacts/evaluation/` 和 `artifacts/reconstruction/mlp_cnn_n03_v31_finetune_v3/`；精选结果位于 `results/residual_mlp_cnn/mlp_cnn_n03_v31/`；完整报告位于 `reports/MLP_CNN_V31_REPORT.md`。
- 结论：严格 HRIR-ILD 损失有效收回了 v3 的大部分 ILD 退化，同时几乎保持 v3 幅度性能。v3.1 是当前推荐的 MLP-CNN 多指标折中模型，但 v2 仍保有最低 ILD 单项结果。当前 test 已解封，后续不得据此继续调权重；下一步应在新 validation 设计或交叉验证上研究 Pareto 选型、双耳结构约束及跨稀疏阶数泛化。

> 2026-07-26 项目已按职责重构。本文此前记录的命令保留当时的历史路径；
> 当前路径与入口以 `README.md`、`docs/PROJECT_STRUCTURE.md` 和
> `experiments/` 为准。

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

## 2026-07-26：项目结构职责化重构

- 工作目标：将 MCA/SUpDEq demo 从项目主体降为传统基线，并把公共实现、实验定义、本地运行产物、精选结果和报告彻底分离。
- 回滚锚点：重构前提交为 `a02e337`；创建标签 `pre-project-restructure-20260726`，在分支 `codex/project-structure-refactor` 上实施。
- 结构调整：MCA 入口迁入 `baselines/mca/`；Python 公共代码抽取为 `src/mcar/` package；MATLAB 公共函数迁入 `matlab/+mcar/`；数据划分和锁定参数归入 `configs/`；实验说明归入 `experiments/`；技术报告迁入 `reports/`。
- 数据与产物：约 2.2 GB 本地 SUpDEq、HDF5、checkpoint、MAT、W&B 缓存和批处理结果均在同一磁盘内无损移动到 `external/`、`data/processed/` 和 `artifacts/`，未删除实验数据。v1/v2/v3 精选 CSV、JSON 和 PNG 迁入 `results/`；v3 首次把正式训练及严格重建轻量结果纳入 Git 候选范围。
- 代码调整：移除 Python 脚本的 `sys.path` 拼接，改用 `mcar.*` 包导入；新增 `pyproject.toml` 和项目根路径 helper；训练输出统一写入 `artifacts/training/`。MATLAB 入口改用 `mcar.*` package，SUpDEq 路径统一为 `external/SUpDEq/`。
- 文档调整：重写根 README，使项目主体明确为 MCA 后残差学习；新增 `docs/PROJECT_STRUCTURE.md`、baseline/experiment/config/result 说明，并同步更新 AGENTS 路径约定。
- 验证结果：Python 3.9 `compileall` 通过；所有核心 package 导入和 v1/v3 CLI `--help` 通过；11 个 JSON 文件解析通过；MATLAB 在沙箱外成功解析 5 个新 package/baseline 入口；真实 pp91、2 方向、463 频点的 v3 前后向 smoke test 通过，输出形状为 `2 × 2 × 463`，zero-init identity error 为 0，CNN 第二步得到 62 个非零梯度张量。
- 输出位置：结构和迁移表见 `docs/PROJECT_STRUCTURE.md`；当前可执行命令见根 README 与 `experiments/`。
- 结论与下一步：重构未改变模型参数、数据划分或已锁定实验结论。下一步应在新结构上实现 v3.1 可微 ILD 损失，并保持源码复用和 artifacts/results 发布边界。

## 2026-07-26：MLP + CNN v3 首版完整技术报告

- 工作目标：在 v3 正式训练、完整 validation、锁定 test 和严格 HRTF 重建全部完成后，形成与 v1/v2 报告体系一致的独立技术报告。
- 报告文件：新增 `docs/MLP_CNN_V3_REPORT.md`，共 25 章，覆盖摘要、版本演进、需求与验收、数据协议、模型选型、MLP-CNN 融合、参数与感受野、初始化、双耳完整频谱采样、复合损失、W&B、训练过程、validation/test、防泄漏协议、MCA 幅度回填、严格指标、逐被试结果、工程结构、完整复现命令、有效性威胁、v3.1 建议和项目结论。
- 数值审校：报告中的训练、完整 residual、严格 ERB/高频/ILD、资源与重建质量数字均与本地 JSON/CSV/MATLAB 输出交叉核对。补充确认 v3 相对 v2 的全空间 ERB、对侧 25° ERB和对侧高频均为 12/12 test 被试改善；严格 ILD 只有 3/12 优于 v2，9/12 回升，说明 ILD 退化是系统性弱点而非单个异常值。
- 口径修正：旧 v3 README 所写“91 点感受野”只对应膨胀卷积主干；报告按实际网络计算，计入 kernel-7 stem 后完整理论感受野为 97 点，约 `4.18 kHz`，并已同步修正 README。
- Git 状态：报告与源码保持可提交，checkpoint、HDF5、W&B 缓存、MAT 和批量生成图片继续由嵌套 `.gitignore` 忽略；本次未提交或推送，也未改动用户已有的根 `.gitignore` 变更。
- 结论与下一步：首版报告已经可以用于项目交接、阶段汇报和论文方法章节素材。后续若进入 v3.1，优先实现与最终 HRIR 能量定义对齐的可微 ILD 损失，并在不访问 test 的情况下完成 validation 选型。

## 2026-07-25：MLP + CNN v3 锁定 test 与严格 HRTF 重建评估

- 实验目标：在完整 validation 已确认且模型、超参数和 epoch 9 checkpoint 全部锁定后，解封固定的 12 个 test 被试；先穷举 raw residual，再把 v3 residual 回填到 MCA 幅度，并使用与 v1/v2 完全相同的 MATLAB 口径计算论文对应的 ERB magnitude error、对侧高频误差和 ILD，同时生成 12 被试 HRTF 总览。
- test 范围与防泄漏：test 被试为 pp8、pp18、pp22、pp26、pp31、pp33、pp45、pp47、pp59、pp70、pp73、pp81。仅在 validation 阶段完成并明确锁定 epoch 9 后执行 `--split test --allow-test`；test 结果未用于重新训练、模型选择或超参数调整。
- 完整 raw residual：12 个 HDF5 × 2 耳 × 900 方向 × 463 频点，共 `10,000,800` 个样本。MCA zero-residual 的 MAE/RMSE 为 `2.600671/4.186115 dB`，v2 为 `2.168373/3.552562 dB`，v3 为 `2.091041/3.452171 dB`。v3 相对 MCA 的 MAE 改善 `19.60%`，相对 v2 的 MAE 改善 `3.57%`；12/12 test 被试的 v3 MAE 均低于 v2，逐被试改善范围为 `1.42%–4.79%`。内置 v2 与历史 test 结果的 MAE/RMSE 差值仅 `-7.68e-9/-9.18e-8 dB`，一致性检查通过。
- 完整频谱 GPU 推理：新增 `residual_learning/mlp_cnn_v3/python/predict_mlp_cnn_reconstruction.py`。复用 v1 已缓存且与 v2 共用的 900 个 Fliege dense 方向 + 360 个水平面方向，按 32 个方向分块，但双耳和完整 463 点频率轴始终联合输入 CNN。RTX 5060 上 12 被试总耗时 `6.01 s`，峰值 CUDA allocated memory `34.81 MiB`，全部预测 HDF5 完整写入。
- 严格重建实现：扩展 `residual_learning/matlab/evaluate_test_reconstruction.m`，在不改变 v1/v2 默认调用方式的前提下支持显式输出根目录、缓存根目录、方法键和图例名称；新增 v3 入口 `residual_learning/mlp_cnn_v3/matlab/evaluate_mlp_cnn_v3_reconstruction.m` 与 v1/v2/v3 对比图脚本。预测 residual 逐频点加到 MCA log-magnitude，复数相位完全沿用 MCA。
- 严格指标结果：MCA 的全球 ERB、对侧 25° ERB、对侧 `>10 kHz` magnitude error、水平面 ILD MAE 分别为 `0.802741/1.860286/4.258336/0.885425 dB`；v3 分别为 `0.584593/1.304415/3.620148/0.661566 dB`，相对 MCA 改善 `27.18%/29.88%/14.99%/25.28%`，四项均为 12/12 被试改善。
- v3 对 v2：v2 四项严格指标为 `0.611503/1.340068/3.720378/0.646722 dB`。v3 在全球 ERB、对侧 25° ERB和对侧高频上进一步降低 `4.40%/2.66%/2.69%`；ILD MAE 则由 `0.646722` 升至 `0.661566 dB`，相对退化 `2.30%`。因此 v3 的 CNN 频率上下文对三项幅度指标形成稳定增益，但当前训练中的 ILD proxy 未保证严格 HRIR 能量 ILD 相对 v2 单调改善，此结论必须保留，不能只汇报相对 MCA 的正向数字。
- 重建质量检查：12 被试最大左/右相位误差分别为 `6.00e-16/6.22e-16 rad`，最大幅度回填恒等误差均为 `7.11e-15 dB`，远低于 `1e-5 rad/1e-4 dB` 断言阈值。12 张单被试图、`test12_contralateral_hrtf_overview.png`、`test12_metric_overview.png` 与 `v1_v2_v3_aggregate_comparison.png` 均已生成并完成视觉检查。
- 实际命令：raw test 使用 `D:\miniconda3\envs\ml\python.exe residual_learning/mlp_cnn_v3/python/evaluate_mlp_cnn_v3.py residual_learning/data/hutubs_residual_v1_n03 residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt --split test --allow-test --directions-per-block 32`；预测使用 `D:\miniconda3\envs\ml\python.exe residual_learning/mlp_cnn_v3/python/predict_mlp_cnn_reconstruction.py residual_learning/mlp_cnn_v3/reconstruction/mlp_cnn_n03_v3_cnn_only residual_learning/reconstruction/mlp_n03_v1 residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt residual_learning/data/hutubs_residual_v1_n03/training_statistics.json --directions-per-block 32`；MATLAB 使用 `matlab.exe -batch "addpath('D:/cuc/CSMT/MCAR/residual_learning/mlp_cnn_v3/matlab'); evaluate_mlp_cnn_v3_reconstruction; plot_v1_v2_v3_reconstruction_comparison"`。
- 输出与 Git：本地结果位于 `residual_learning/mlp_cnn_v3/reconstruction/mlp_cnn_n03_v3_cnn_only/`，包括预测 HDF5、CSV、MAT、JSON 和 15 张 PNG；raw test JSON/CSV 位于正式 run 目录。两类生成物均由 v3 的 `.gitignore` 忽略。适合提交的是 Python/MATLAB 源码、README、依赖说明、占位 `.gitignore` 和本实验日志；本次未提交或推送。
- 运行异常：MATLAB 在受限沙箱内首次启动报 `File system inconsistency`，在已授权的正常本机权限下启动成功。命令接口 60 秒超时后 MATLAB 子进程继续完成全部评估；没有重复启动、没有丢失或覆盖结果。Codex 服务层中途出现一次 503，不影响本地 Python/MATLAB 进程及产物。
- 结论与下一步：v3 的核心假设得到部分验证——频谱 CNN 在未见 test 上可稳定降低 raw residual 和三项严格幅度误差，但 ILD 相对 v2 小幅退化。下一步不应直接增加模型规模；优先做一个受控 v3.1 实验，在保持当前架构和 test 锁定的前提下，仅在 train/validation 上校准严格 ILD 对齐的可微损失或提高双耳能量约束，随后以 validation 选择 checkpoint，再进行一次最终 test。也可先整理 v3 详细报告和可提交的轻量结果清单。

## 2026-07-25：MLP + CNN v3 完整 validation residual 评估

- 实验目标：在不访问 test 的前提下，对 v3 epoch 9 best checkpoint 执行无随机采样的完整 validation 遍历，计算全部 raw residual MAE/RMSE；同时从同一个 checkpoint 得到 MCA zero-residual、冻结 v2 MLP 和最终 v3 三组结果，验证 CNN 的增益及逐被试稳定性。
- 评估器：新增 `residual_learning/mlp_cnn_v3/python/evaluate_mlp_cnn_v3.py`。由于 CNN 必须观察完整频谱，评估按 32 个方向分块，但每块保留双耳和全部 463 个频点；每个 HDF5 的 900 个方向全部遍历。`--split test` 必须额外显式提供 `--allow-test`，避免模型选型阶段意外读取 test。
- 数据范围：固定 validation 被试 pp13、pp35、pp39、pp41、pp51、pp53、pp55、pp56、pp58、pp74、pp84、pp92；12 个文件 × 2 耳 × 900 方向 × 463 频点，共 `10,000,800` 个样本，实际计数完全一致。
- 汇总结果：MCA zero-residual 的 MAE/RMSE 为 `2.571911/4.143771 dB`；v2 MLP 为 `2.139782/3.506980 dB`；v3 MLP + CNN 为 `2.072657/3.428915 dB`。v3 相对 MCA 的 MAE 改善为 `19.41%`，相对 v2 的 MAE/RMSE 分别降低 `3.14%/2.23%`；全样本 CNN delta 平均绝对值为 `0.5687 dB`。
- v2 口径交叉检查：评估器内置 v2 MLP 的 MAE/RMSE 与原 `residual_learning/runs/mlp_n03_v2/val_metrics.json` 的差值分别为 `+1.10e-8/-1.83e-8 dB`，远低于 `1e-4 dB` 阈值，证明新评估器的数据遍历、归一化、AMP 和反归一化口径与原 v2 完整评估一致。
- 逐被试结果：12/12 validation 被试的 v3 MAE 均低于 v2。相对改善依次为 pp13 `1.80%`、pp35 `3.51%`、pp39 `1.58%`、pp41 `3.19%`、pp51 `3.57%`、pp53 `4.44%`、pp55 `2.65%`、pp56 `4.12%`、pp58 `1.74%`、pp74 `2.36%`、pp84 `4.24%`、pp92 `4.28%`；范围 `1.58%–4.44%`，逐被试改善百分比的非加权平均为 `3.12%`，没有退化案例。
- 资源与完整性：完整评估耗时 `5.15 s`，峰值 CUDA allocated memory `35.20 MiB`；checkpoint epoch 为 9、训练阶段标记 `cnn_only_frozen_mlp`、模型参数 `174,627`。Python 编译、样本计数、逐被试/总计合并和 v2 历史交叉检查均通过。
- 实际命令：`D:\miniconda3\envs\ml\python.exe residual_learning/mlp_cnn_v3/python/evaluate_mlp_cnn_v3.py residual_learning/data/hutubs_residual_v1_n03 residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt --split val --directions-per-block 32`，退出码为 0。
- 输出与 Git：本地生成 `val_full_metrics.json` 和 `val_per_subject_metrics.csv`，位于正式 v3 run 目录并由 `runs/.gitignore` 忽略；关键结果已记录在本日志与 v3 README，未读取 test，未提交或推送。
- 结论与下一步：v3 在完整未见 validation 的 raw residual MAE、RMSE 和全部 12 个被试上稳定优于 v2，满足进入固定 test 严格评估的前置条件。下一步锁定 epoch 9 checkpoint，在 12 个 test 被试上先做完整 raw residual 评估，再生成 v3 双耳完整频谱预测、回填 MCA 幅度并使用 MATLAB 计算严格 ERB、对侧高频与 ILD；模型和超参数不再根据 test 调整。

## 2026-07-25：MLP + CNN v3 CNN-only 跨被试在线完整训练

- 实验目标：从原始 v2 epoch 9 checkpoint 重新开始，在 72 train / 12 validation 被试上完整训练双耳频谱 CNN + FiLM；v2 MLP 全程冻结，test 集不参与训练和选型。通过 W&B 在线监控曲线，并以固定 validation 复合损失选择最佳 checkpoint。
- W&B 运行：project 为 `mcar-mlp-cnn-v3`，run name 为 `mlp_cnn_n03_v3_cnn_only`，run id 为 `qtkrxras`，URL 为 `https://wandb.ai/luyoung/mcar-mlp-cnn-v3/runs/qtkrxras`。第一次从受限进程启动时被系统禁止访问 `api.wandb.ai:443`，进程只停留在 W&B 握手阶段，未写 configuration/history/checkpoint；停止后在新的本地 run 目录以允许网络访问的进程重新启动，在线 run 正常创建并最终以 exit code 0 完成同步。
- 数据与采样：HUTUBS simulated N=3 residual HDF5；72 个 train 和 12 个 validation 被试；每个 batch 随机选择一个被试、32 个 Fliege 方向、双耳和全部 463 个频点，共 `29,632` 个逐频点样本。validation 每个 epoch 均以 seed `20260726` 重建相同的 96 个 batch，确保 epoch 0 的 v2 基线与所有 v3 epoch 可直接比较。
- 模型与训练参数：总参数 `174,627`，其中冻结 v2 MLP `100,225`，可训练 CNN + FiLM `74,402`；10 epoch × 500 steps；AdamW 学习率 `3e-4`、weight decay `1e-5`、cosine decay、gradient clip `5.0`、FP16 AMP 初始 scale `1024`。损失保持 residual SmoothL1 + `0.50 × ERB proxy + 0.25 × 对侧高频 + 0.25 × ILD proxy` 的 v2 权重。
- 固定 validation 基线：训练开始前的 v2 total/residual/ERB/高频/ILD 为 `0.594113 / 2.139116 / 0.695902 / 3.297190 / 0.646540 dB`，CNN delta 为 0。
- 收敛过程：validation total 从 epoch 1–10 依次为 `0.588567、0.585525、0.581558、0.579968、0.579476、0.575839、0.574931、0.573803、0.573431、0.573586`。前 9 个 epoch 总体持续下降，epoch 10 略回升，因此根据预定规则选择 epoch 9。
- 最佳 epoch 9：validation total/residual/ERB/高频/ILD 为 `0.573431 / 2.073348 / 0.664612 / 3.216917 / 0.641232 dB`，相对初始 v2 分别降低 `3.48% / 3.07% / 4.50% / 2.43% / 0.82%`，五项均改善；validation CNN delta 平均绝对值为 `0.5657 dB`。训练侧相应 total/residual/ERB/高频/ILD 为 `0.537791 / 1.977483 / 0.645931 / 2.986437 / 0.617659 dB`。
- epoch 10 取舍：epoch 10 的 residual/ERB 进一步变为 `2.073211/0.663860 dB`，但高频和 ILD 回升到 `3.218165/0.641742 dB`，使 total 为 `0.573586`，未超过 epoch 9；因此不按单一 residual 或 ERB 指标改选 checkpoint。
- 稳定性与资源：训练总耗时 `364.19 s`（约 6 分 4 秒），峰值 CUDA allocated memory `220.46 MiB`；5000 个优化步均无 AMP 跳步，最终 AMP scale 增长到 `4096`。在线曲线与本地 `history.csv` 一致；正常生成 `training_report.json`。
- checkpoint 验证：`best.pt` 为 epoch 9，`last.pt` 为 epoch 10；两者均可由 PyTorch 成功重新读取。best checkpoint 的训练阶段标记为 `cnn_only_frozen_mlp`，包含 82 个 state tensors。
- 输出与 Git：本地结果位于 `residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online`；configuration、initial validation、history、best/last checkpoint、training report 和 W&B 本地缓存均由 v3 `runs/.gitignore` 忽略。W&B 仅同步标量曲线与 summary，没有上传 checkpoint 或数据集。
- 结论与下一步：冻结 MLP 的频谱 CNN 在严格未见 validation 被试上同时改善五项训练代理指标，证明收益不是 pp91 单被试记忆。增益较 pp91 过拟合明显收缩，尤其 ILD 仅改善 `0.82%`，说明跨被试泛化仍是限制。下一步先为 v3 实现完整 validation 遍历评估，报告全部 `10,000,800` 样本的 raw residual MAE/RMSE；通过后再锁定 epoch 9，在 test 上执行 residual 回填、MATLAB `AKerbError`、对侧高频与 ILD 严格评估。

## 2026-07-25：MLP + CNN v3 训练器接入 Weights & Biases

- 实验目标：为完整 CNN-only 跨被试训练加入在线曲线监控，同时保持本地 CSV、checkpoint 和 JSON 报告为复现主记录；W&B 不自动上传模型权重或数据集。
- 中断运行状态：原无 W&B 完整训练由用户主动停止。进程停止前已完整保存 epoch 1–6，`best.pt` 与 `last.pt` 均为可读取的 epoch 6，固定 validation total loss 为 `0.57584`，6 个 epoch 均无 AMP 跳步；因未完成全部 10 epoch，不存在只在正常结束时生成的 `training_report.json`。该运行保留在本地忽略目录，不需要清理，也不与新 W&B run 拼接。
- 训练器修改：`train_mlp_cnn_v3.py` 新增 `--wandb-mode disabled|online|offline`、project、entity、run name、group、job type、tags、notes 和可选 `--wandb-watch`。默认 mode 为 disabled，保证原有本地命令不产生联网行为；正式监控时显式传入 `--wandb-mode online`。默认 project 为 `mcar-mlp-cnn-v3`。
- 记录指标：W&B 以 epoch 为公共横轴，记录 train/validation 的复合损失、residual MAE、ERB proxy MAE、对侧高频 MAE、ILD proxy MAE 和 CNN delta 平均绝对值；同时记录 validation 相对初始 v2 的五项改善百分比、learning rate、AMP scale、当轮/累计跳步数、epoch 耗时、峰值 CUDA allocated memory、当前 epoch 是否为最佳以及当前最佳 epoch。
- 运行摘要：正常结束后将最佳 epoch、最佳 validation 五项指标、总耗时、峰值显存和 AMP 跳步总数写入 W&B summary；相同信息仍保存在本地 `training_report.json`。`best.pt`、`last.pt`、HDF5 和其他大型产物不上传 W&B。
- 本地目录：为避免受限环境下 W&B core 写入用户 AppData 失败，训练器在 import W&B 前将 `WANDB_DIR`、`WANDB_DATA_DIR`、`WANDB_CACHE_DIR`、`WANDB_CONFIG_DIR` 和 `WANDB_ARTIFACT_DIR` 全部指向当前 `runs/<run-name>` 下的忽略目录。首次 offline 测试虽成功但出现 AppData debug-log 权限错误；重定向后的第二次测试不再出现该错误。
- 依赖：新增 `residual_learning/mlp_cnn_v3/requirements.txt`，复用公共 residual-learning 依赖并要求 `wandb>=0.21,<1`。本机 `ml` 环境已安装并验证 W&B `0.21.1`。
- 离线集成测试：使用 pp91、4 方向、1 epoch × 2 steps、2 个 validation batch 和 `--wandb-mode offline` 实际运行，退出码为 0。W&B run id 为 `2b1v9fvt`；初始 validation、epoch 1 的全部 train/validation 曲线、改善百分比、optimizer/system/checkpoint 指标以及最终 summary 均成功写入。checkpoint、本地 JSON 和 W&B offline run 同时生成，AMP 跳步为 0。
- Git 约定：W&B 的 `wandb/`、`wandb_state/`、offline run、checkpoint 和测试输出均位于 v3 `runs/` 下，由嵌套 `.gitignore` 忽略；只保留训练器、v3 README、requirements 和本实验日志。
- 正式运行建议：从原 v2 epoch 9 checkpoint 重新开始，不复用被中断 run name；本地 run 使用 `mlp_cnn_n03_v3_cnn_only_wandb`，W&B 显示名使用 `mlp_cnn_n03_v3_cnn_only`，保持 72/12 划分、10 epoch × 500 steps、96 个固定 validation batch、32 方向和现有复合损失权重。在线开始前只需确保本机已执行 `python -m wandb login`，API key 不写入项目或命令。

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
- 输出文件位置：`../artifacts/figures/mca_demo_ku100_ns3_nd44/contralateral_high_frequency/`。该目录是脚本生成且被 Git 忽略的本地实验产物，包含：
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
- 输出文件位置：`../artifacts/figures/mca_demo_ku100_ns3_nd44/`。该目录中的大型 MAT 文件均为本地可复现产物，不提交 Git；下列相对链接需先运行导出脚本：
  - [正前方 conventional vs MCA HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/01_frontal_conventional_vs_mca_hrir.png)
  - [对侧耳 SH vs MCA HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/02_contralateral_sh_vs_mca_hrir.png)
  - [对侧耳 conventional vs MCA HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/03_contralateral_conventional_vs_mca_hrir.png)
  - [对侧耳 SH vs reference HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/04_contralateral_sh_vs_reference_hrir.png)
  - [对侧耳 conventional vs reference HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/05_contralateral_conventional_vs_reference_hrir.png)
  - [对侧耳 MCA vs reference HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/06_contralateral_mca_vs_reference_hrir.png)
  - [SH、SUpDEq + SH、MCA 的 LSD 曲线](../artifacts/figures/mca_demo_ku100_ns3_nd44/07_lsd_left_ear.png)
  - [SH、SUpDEq + SH、MCA 的 ERB-band magnitude-error 曲线](../artifacts/figures/mca_demo_ku100_ns3_nd44/08_erb_magnitude_error_left_ear.png)
- 阻塞项：
  1. 尚未定义并实现对侧耳区域的统一判定规则，也未对 `>10 kHz` 区域计算 conventional 与 MCA 的均值、中位数、误差差值和相对改善百分比，因此目前只有可视化趋势，缺少论文可用的定量结论。
  2. residual MLP 的训练数据生成流程尚未固定：仍需明确输入特征、log-magnitude residual 目标、训练/验证/测试划分以及跨方向或跨个体的划分方式。
  3. residual MLP 尚未实现和训练，因此当前只能完成 SH、SUpDEq + SH、MCA 三种传统基线比较，尚不能评估 MCA + residual MLP 的增益。
- 结论与下一步：MCA demo 已成功复现，传统基线和图表导出流程已经跑通。下一步先完成对侧耳高频区域的定量统计，再固定 residual 数据集格式与划分方式，最后训练轻量 MLP，并以 LSD、ERB-band magnitude error、ILD error 和对侧高频误差与 MCA baseline 比较。
