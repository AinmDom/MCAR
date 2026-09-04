# Hybrid LSD / ERB 机制审计与开发建议

日期：2026-09-04。状态：代码与既有 validation 证据审计完成；以下消融为建议，未预注册、未实现、未训练。起点 HEAD：`ae71552da02f0b4e0292e72696c2aba554fe123a`，working tree clean。

当前证据支持把**训练目标与 LSD 的不一致**列为优先检验假设，尚不能证明 ERB 是性能差距的主因。

## 23:13 补充：MCA 内部 ERB 分支（纠正此前解释范围）

用户进一步明确关注 MCA 数据生成，而非仅关注训练损失。此前审计没有追踪 `external/SUpDEq/supdeq_interpHRTF.m` 内部，关于 ERB 坐标不压缩输出的结论仍成立，但不足以描述整个流程；不能理解为 MCA 内部没有 ERB 处理。

实际调用见 `matlab/+mcar/export_sonicom_residual_dataset.m:287`：SUpDEq + SH，mc=inf（开启无限幅度增益校正，不是关闭校正）、minimum phase=true、fadeDown。上游 `supdeq_interpHRTF.m:669` 的条件为 `~isnan(mc)`。

实际 MCA 包含两条相连分支：普通复数 HRTF 做 SUpDEq/SH 插值和去均衡；另以 ERB 分析得到稀疏输入的带能量，空间插值到目标方向，并减去普通插值结果的 ERB 带能量，得到 dB 校正量（677–766行）。768–769行用 `AKinterpolateSpectrum(..., ferb, NFFT, {'nearest' 'spline' 'nearest'}, fs)` 将校正量插值到普通 FFT 网格。再作低频限制、dB到线性增益转换、最小相位构造，869–870行乘回原普通复数 HRTF。输出一直是完整 FFT 网格，而不是 ERB band 向量。

因此有“从 ERB 带中心到普通 FFT 频点的插值”，没有“ERB 反卷积”。ERB 带能量积分是有损摘要，样条插值不会恢复带内未知细节；但主支路的普通 HRTF 本身并未被这份摘要替代。除以已知非零 `corrFilt_lim` 至多撤销 MCA 校正，得到保存的 `HRTF_*_ip_noMC`，不能还原真实目标 HRTF。

导出器把普通频谱的 `20log10(abs(H))` 写入 `/mca_logmag_db`，标签为 reference−MCA；Hybrid 重建在普通频点上做 `mcaDb + residualDb`（`matlab/+mcar/evaluate_film_siren_spectral_cnn_d1_validation.m:120`）。不存在漏做逆ERB的证据。ERB驱动的平滑增益是否限制LSD需要单独检验，上轮四组只检验训练loss，不直接消融MCA内部ERB校正。

本轮只读审计，没有新数据访问或实验。用户已选四组各3 seeds、共12 run；随后要求先暂停准备并澄清本机制，尚未生成新配置、修改训练器或启动训练。支持性原始来源：[MCA论文](https://arxiv.org/abs/2303.09966)、[官方SUpDEq实现](https://github.com/AudioGroupCologne/SUpDEq)。

## 已核实的实现

- `src/mcar/models/film_siren_spectral_cnn.py` 的输出为完整频率网格上的 `base + delta`。CNN 接收双耳 MCA、correction、FiLM residual 和 log-frequency，共七通道；没有将最终输出压缩为 ERB band。
- `src/mcar/training/train_siren.py:frequency_coordinates` 的 dual mapping 是每个频点同时提供 linear 与 ERB 坐标，不做频谱平滑、重采样或删点。它可能影响归纳偏置，但不存在由这个映射直接造成的频点丢失。
- `src/mcar/training/train_mlp_v2.py:make_erb_weights` 用 41 个 ERB-rate 三角权重构造 AKerbError 的训练代理。`band_energy_db` 对线性功率积分再转 dB；它不是最终 MATLAB HRIR/AKerbError 指标本身。
- `configs/experiments/sonicom_film_siren_spectral_cnn_final_seed20260821_e190.json` 与 `src/mcar/training/film_siren_stage_c.py` 表明正式 Hybrid 使用方向加权 normalized residual Smooth L1，加 ERB/HF/strict ILD/band ILD/D1/D2/notch，系数依次为 `0.75/0.25/0.75/0.05/0.25/0.15/0.30`，附加项除以 train-only target std。没有直接的全频 LSD 项。不能仅由系数大小推断各项梯度主导程度。
- `src/mcar/evaluation/secondary_metrics.py:full_sphere_lsd` 先对每耳每方向的逐频点 dB 误差平方求均值并开方，再平均双耳、按方向立体角加权，最后在被试间汇总。463 bins：86.1328125–19982.8125 Hz；793 方向排除 Q26 后评价 767 个方向。

## 机制解释及证据限度

设频谱 dB 值为 x，某一带能量为 `B = 10 log10(sum_f w_f 10^(x_f/10))`。其对 x_f 的导数为该频点的带内功率占比。因此低能量深凹口通常获得较小的带能量梯度；带内不同细节也可能形成近似相同的积分能量。逐频点 LSD 则直接惩罚这些 dB 偏差，且线性频率网格下高频占有更多 bins。

这解释了 ERB 误差小与 LSD 偏大可以同时发生，但不是对当前模型误差来源的实测分解。Smooth L1 大误差区的线性惩罚、冻结 FiLM 父模型、CNN 的表达能力、稀疏空间信息和各损失之间的梯度冲突都仍是可能因素。现有 D1/D2 与 notch 项已经约束细节，不能断言 Hybrid 完全忽略凹口。

## 既有 validation 数值

同一 44-subject / 767-direction / 463-bin 口径，均值单位 dB，越低越好：

| 方法 | FullSphereLSD |
|---|---:|
| FSP-AE | 3.0700346219267796 |
| RANF | 3.3614502402651145 |
| Hybrid E190 | 3.5489256190074472 |
| Bounded E25 | 3.5874447316845934 |
| MCAR v3.5.1 | 3.6097610496792503 |
| FiLM E130 ensemble | 3.7857646944239676 |

来源：`results/sonicom_film_secondary_metrics_v1_validation/aggregate_metrics.csv` 与 `results/sonicom_film_secondary_baseline_extension_ranf_fsp_v1_validation/aggregate_metrics.csv`。两套 `quality_checks.json` 均为 passed、44 subjects、all_finite=true、test read=0；数值与 `docs/EXPERIMENT_LOG.md` 一致。这里重述既有结果，没有新推理、统计检验或逐频点分析。Hybrid 较 FiLM 父路线和 MCAR 的均值更低，故不能将其概括为 ERB 造成的整体失败。

## 建议的下一轮开发

先注册独立的 LSD 开发阶段，保持已冻结论文主模型和旧 test 结果身份。按统一规则在开发数据上比较以下 2×2 消融：

| 组别 | 原 ERB magnitude 项 | 新增 LSD 项 | 用途 |
|---|---|---|---|
| A | 保留 | 无 | 同预算复现对照 |
| B | 保留 | 有 | 检验直接监督 LSD 的收益 |
| C | 去除 | 无 | 检验现有 ERB 项的条件效应 |
| D | 去除 | 有 | 检验 ERB 与 LSD 的相互作用 |

本消融中 band ILD 固定保留，所以“去除 ERB”仅指 ERB magnitude loss，不表示清除所有 ERB 相关运算。其他损失、父模型、CNN 初始化、batch 顺序、seeds、优化器与预算须匹配；从相同冻结父模型和 zero-init CNN 重新训练，不能让 warm-start 历史混入因果比较。

新增项建议为 `lambda_lsd * mean_ear,direction(sqrt(mean_frequency(error_db^2)+epsilon^2)) / target_std`，epsilon 仅用于零误差附近的数值稳定。必须沿用评价器的聚合顺序，不能先对全部方向/被试平方池化再开方；报告端仍使用原始无 epsilon 指标。残差 dB 误差等于 MCA 加回后的全谱 dB 误差。lambda 与 epsilon 的数值在正式实验前冻结，lambda 的尺度可先用 train-only 梯度量级诊断，不能根据 test 决定。

各组以同一个 validation LSD 规则选择开发 checkpoint，附带检查 Full/Contra ERB、HF、strict ILD 和被试尾部；不再比较各自不同定义的 raw total。预算末点最优需标记为预算不足，正式周期在开发后另行冻结。第一轮无需改 ERB 坐标、网络容量或解冻 backbone；若四组均受限，再做独立结构消融。

训练前可预注册一个开发集误差分解：固定低/中/高频区间，统计平方误差贡献，以及每方向频谱均值偏差与去均值误差；`MSE = bias^2 + centered_MSE`，但均值 LSD 不能照此直接相加。只有这样才能区分宽带偏移、局部细节和高频问题。当前尚未运行此诊断。

已消费 SONICOM test，且本议题发生在已知结果之后；新增结果须标为后续探索性开发。沿用 train/validation 作开发并不恢复独立性，最终确认需要未消费的外部/新被试 holdout。当前不创建 test registry、不覆盖既有预测。

## 文献核验

后续决策（2026-09-04）：用户仅授权B组×3 seeds，复用旧Hybrid对照；本报告此前2×2建议
不再作为执行计划。训练协议以 `experiments/film_siren/STAGE_D_HYBRID_LSD_B_E190_PROTOCOL.md`
为准，固定E190 last.pt，不按validation LSD另选checkpoint。

- [Chen et al., Exploring Frequency-Domain Feature Modeling for HRTF Magnitude Upsampling, 2026 preprint](https://arxiv.org/html/2602.11670v1)：第 IV.5 节直接使用 LSD + spectral gradient loss，第 V.2 节按 validation LSD 选 checkpoint。支持优先检验直接 LSD 监督；其 48 kHz、106 bins、稀疏网格及数据划分不同，论文绝对数值不可与本表直接排名。
- [Yao et al., JASA 2024, Perceptually enhanced spectral distance metric for head-related transfer function quality prediction](https://pubmed.ncbi.nlm.nih.gov/39699206/)：听觉实验表明 LSD 仍广泛使用，但对空间与音色感知的预测存在局限。因此优化 LSD 时应同时保留听觉/双耳指标。

完整性：本次未打开 subject SOFA/HDF5 或 checkpoint；新增 `test_subjects_read=0`；训练、best/末点预算与新实验 finite=N/A。仅新增审计报告与本地交接记录，未修改模型、配置、指标或正式实验结果。
