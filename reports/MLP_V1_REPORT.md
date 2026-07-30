# 幅度校正与时间对齐插值（Magnitude-Corrected and Time-Aligned Interpolation，MCA）Residual 多层感知机（Multi-Layer Perceptron，MLP）v1 完整设计与实验报告

缩写说明：头相关传输函数（Head-Related Transfer Function，HRTF）、头相关脉冲
响应（Head-Related Impulse Response，HRIR）、Sigmoid 线性单元（Sigmoid
Linear Unit，SiLU）、平均绝对误差（Mean Absolute Error，MAE）、均方根误差
（Root Mean Squared Error，RMSE）、等效矩形带宽（Equivalent Rectangular
Bandwidth，ERB）、耳间电平差（Interaural Level Difference，ILD）、方向均衡
空间上采样（Spatial Upsampling by Directional Equalization，SUpDEq）、球谐
函数（Spherical Harmonics，SH）、耳间时间差（Interaural Time Difference，
ITD）、空间声学数据格式（Spatially Oriented Format for Acoustics，SOFA）、
Hierarchical Data Format version 5（HDF5）、快速傅里叶变换（Fast Fourier
Transform，FFT）、快速傅里叶逆变换（Inverse Fast Fourier Transform，IFFT）、
图形处理器（Graphics Processing Unit，GPU）、统一计算设备架构（Compute
Unified Device Architecture，CUDA）、16 位浮点数（16-bit Floating Point，
FP16）、自动混合精度（Automatic Mixed Precision，AMP）、卷积神经网络
（Convolutional Neural Network，CNN）、修正线性单元（Rectified Linear Unit，
ReLU）、逗号分隔值（Comma-Separated Values，CSV）、JavaScript 对象表示法
（JavaScript Object Notation，JSON）、便携式网络图形（Portable Network
Graphics，PNG）、标识符（Identifier，ID）、吉比字节（gibibyte，GiB）和
兆二进制字节（mebibyte，MiB）。MATLAB MAT-file 缩写为 MAT；HUTUBS、AXD 和
KU100 是数据集或设备专名，不作首字母展开。

> 路径说明：本报告记录的是重构前实际执行环境；当前源码、数据和结果路径
> 见 `../docs/PROJECT_STRUCTURE.md` 与 `../experiments/residual_mlp/`。

> 数据集：HUTUBS simulated HRTF，96 个被试
>
> 稀疏输入：Lebedev `N=3`，26 个方向
>
> 模型：Residual MLP v1，`100,225` 个可训练参数
>
> 学习目标：MCA 与 dense reference 之间的 log-magnitude residual
>
> 实验日期：2026-07-23 至 2026-07-24

## 摘要

本项目研究如何在 Magnitude-Corrected and Time-Aligned Interpolation（MCA）已经完成 HRTF 空间插值的基础上，使用轻量神经网络进一步降低剩余幅度误差。与直接从稀疏 HRTF 生成完整复数 HRTF 的端到端方法不同，MLP v1 保留 MCA 的时间对齐、球谐插值、幅度校正和相位结果，只学习 dense reference 与 MCA 之间的 log-magnitude residual：

$$\begin{aligned}
r_{\mathrm{target}}(s,e,\Omega,f)
=
20\log_{10}\left|H_{\mathrm{ref}}(s,e,\Omega,f)\right|
-
20\log_{10}\left|H_{\mathrm{MCA}}(s,e,\Omega,f)\right|.
\end{aligned}$$

其中 $s$ 为被试，$e$ 为耳朵，$\Omega$ 为空间方向，$f$ 为频率。模型输出预测 residual $\hat r$，并通过

$$\begin{aligned}
\widehat L_{\mathrm{corrected}}
=
L_{\mathrm{MCA}}+\hat r
\end{aligned}$$

修正 MCA 幅度。复数重建时保留原 MCA 相位：

$$\begin{aligned}
\widehat H_{\mathrm{corrected}}
=
10^{\widehat L_{\mathrm{corrected}}/20}
\exp\left(j\angle H_{\mathrm{MCA}}\right).
\end{aligned}$$

实验使用 96 个 HUTUBS simulated HRTF，被试级固定划分为 72 人训练、12 人验证和 12 人测试。同一被试的左右耳、全部方向和全部频率只属于一个 split。首版实验固定 Lebedev `N=3` 稀疏输入，即使用 26 个方向，通过 MCA 插值至 dense 目标网格。

MLP v1 使用 7 维输入：MCA log-magnitude、MCA correction-filter log-magnitude、方向单位向量 `x/y/z`、log-frequency 和耳别。网络由宽度 128 的输入层、3 个 SiLU residual block 和单输出层组成，共 `100,225` 个可训练参数。每个训练 batch 从一个随机训练被试和一个随机耳朵中抽取 64 个方向与 128 个频点的笛卡尔积，共 8192 个逐频点样本。优化目标为归一化 residual 的 SmoothL1 loss。

在完整 validation 上，MLP v1 将逐频点 residual MAE 从 MCA zero-residual baseline 的 `2.5719 dB` 降至 `2.1886 dB`，改善 `14.90%`；在严格未见 test 上，从 `2.6007 dB` 降至 `2.2172 dB`，改善 `14.75%`。

将预测 residual 回填到 MCA 幅度并使用 MATLAB 严格评价后：

| 指标，越低越好 | MCA | MCA + MLP v1 | 相对改善 | 改善人数 |
|---|---:|---:|---:|---:|
| 全空间 ERB magnitude error | `0.8027 dB` | `0.6838 dB` | `14.82%` | 12/12 |
| 对侧 25° ERB magnitude error | `1.8603 dB` | `1.5582 dB` | `16.24%` | 12/12 |
| 对侧 `>10 kHz` 幅度误差 | `4.2583 dB` | `3.7557 dB` | `11.80%` | 12/12 |
| 水平面 ILD MAE | `0.8854 dB` | `0.7727 dB` | `12.73%` | 10/12 |

结果证明 MCA 后 residual 包含能够跨被试学习的稳定结构，也证明“物理插值基线 + 轻量幅度残差网络”的路线可行。v1 的主要不足是单耳、逐频点训练没有显式约束左右耳能量关系，导致 pp33 和 pp81 的 ILD 分别退化 `5.57%` 与 `18.97%`。这一失败直接说明后续版本应加入双耳联合频谱和 ILD 感知约束，而不是简单扩大网络规模。

## 1. 研究背景与问题定义

### 1.1 HRTF 空间插值

头相关传输函数（Head-Related Transfer Function，HRTF）描述声源从不同空间方向传播至左右耳时的频率响应。其方向相关结构由头部遮挡、耳廓、躯干和个体形态共同形成，是双耳渲染和空间音频个性化的核心数据。

高分辨率 HRTF 测量需要在大量方向重复采集，耗时且对被试稳定性要求高。实际系统希望只测量少量方向，再通过空间插值得到完整方向网格上的 HRTF。稀疏方向采样会产生空间混叠和高频细节损失，尤其是头部遮挡较强的对侧耳区域。

### 1.2 MCA 强基线

当前项目使用 SUpDEq 工具链完成 MCA：

1. 对稀疏 HRTF 做 SUpDEq 时间对齐或方向均衡；
2. 在对齐表示上进行球谐插值；
3. 通过 MCA correction filter 修正插值幅度；
4. 输出 dense 复数 HRTF 和 HRIR。

此前 96 被试传统基线复现已经证明，MCA 在低阶稀疏输入下明显优于 conventional SUpDEq + SH，特别是在对侧区域与 ILD 指标上。因此 residual MLP 面对的是一个融合声学先验和空间插值规则的强基线。

### 1.3 为什么不直接生成完整 HRTF

直接预测完整 HRTF 需要同时学习：

- 被试差异；
- 左右耳关系；
- 空间方向；
- 高频谱峰谷；
- 复数相位和相位环绕；
- 群时延、ITD 与因果性。

在当前 96 个被试规模下，这会增加模型容量、训练难度和物理不一致风险，还会重复学习 MCA 已经能解释的低频和平滑空间结构。

残差学习只预测：

$$\begin{aligned}
r_{\mathrm{target}}=L_{\mathrm{ref}}-L_{\mathrm{MCA}}.
\end{aligned}$$

如果模型输出零，系统自然退化为 MCA。因此 MCA 也是 residual 模型的 zero-residual baseline，不需要额外构造对照模型。

### 1.4 v1 核心研究问题

v1 需要回答：

> 只使用方向、频率、耳别、MCA 幅度和 MCA correction filter，一个约 10 万参数的 MLP 能否在严格未见被试上预测 MCA residual，并使回填后的听觉指标优于 MCA？

这个问题分为三层：

1. **可优化性**：数据读取和梯度链路是否正确，模型能否在单被试上过拟合；
2. **跨被试泛化**：完整 validation/test 的逐频点 residual 是否下降；
3. **实际有效性**：回填后的严格 ERB、对侧高频和 ILD 是否改善。

## 2. 需求说明与验收标准

### 2.1 功能需求

v1 应实现：

- 从 HUTUBS simulated SOFA 生成 MCA、reference、correction filter 和 residual；
- 按被试划分 train/validation/test；
- 从大型 HDF5 按 block 随机读取训练样本；
- 使用统一的训练集统计量归一化；
- 训练轻量 MLP 预测一个 dB residual；
- 完整遍历 validation/test 计算 residual MAE 和 RMSE；
- 对测试方向预测 residual 并回填 MCA 幅度；
- 保持 MCA 相位不变；
- 用 MATLAB 计算 ERB、对侧高频和 ILD；
- 为 12 个测试被试分别输出可视化和指标。

### 2.2 非功能需求

- 所有归一化统计量只能来自 train；
- test 不参与训练、checkpoint 选择或超参数估计；
- 模型控制在约 10 万参数；
- 数据不应一次性全部载入内存或显存；
- 训练支持 CUDA FP16 AMP；
- 导出数据需可断点续跑并验证完整性；
- 大型数据、checkpoint、MAT cache 和预测中间文件不得提交 Git；
- 训练历史、最终指标 CSV/JSON 和必要结果图应保留；
- 逐频点 residual 与最终听觉指标必须分别报告。

### 2.3 验收标准

| 编号 | 验收项 | 判定要求 | 结果 |
|---|---|---|---|
| A1 | 数据完整性 | 96/96 HDF5 有效，无 partial 和非有限值 | 通过 |
| A2 | 监督恒等式 | `reference - MCA = residual` | 通过，最大误差 `0 dB` |
| A3 | 单被试检查 | pp91 residual MAE 明显下降 | 通过 |
| A4 | Validation 泛化 | 完整 validation MAE 优于 MCA | 通过 |
| A5 | Test 泛化 | 完整 test MAE 优于 MCA | 通过 |
| A6 | ERB | 全空间和对侧测试均值优于 MCA | 通过 |
| A7 | 对侧高频 | 测试均值优于 MCA | 通过 |
| A8 | ILD | 测试均值优于 MCA | 通过 |
| A9 | 相位保持 | 误差仅为浮点量级 | 通过 |
| A10 | 逐被试稳定性 | 四项指标均 12/12 改善 | 未完全通过，ILD 为 10/12 |

## 3. 总体技术路线

```text
HUTUBS simulated SOFA
        ↓
Lebedev N=3 稀疏抽取
        ↓
SUpDEq 时间对齐 + SH + MCA
        ↓
MCA / reference / correction filter
        ↓
HDF5 residual 数据集
        ↓
训练集归一化 + 随机 block sampler
        ↓
Residual MLP v1
        ↓
完整 validation/test residual 评价
        ↓
测试方向 residual 推理
        ↓
回填 MCA 幅度 + 保留 MCA 相位
        ↓
AKerbError / 对侧高频 / ILD / 可视化
```

MATLAB 负责 MCA 数据生成、复数重建和严格声学评价；Python/PyTorch 负责 HDF5 采样、MLP 训练和 GPU 推理。

## 4. 数据集与实验协议

### 4.1 HUTUBS simulated HRTF

使用 96 个 simulated SOFA，每个文件包含：

- Lebedev `N=35` 的 1730 个方向；
- 左右耳各 256 点 HRIR；
- 采样率 `44.1 kHz`；
- SOFA convention：`SimpleFreeFieldHRIR`。

SOFA 方向已与 SUpDEq Lebedev `N=35` 网格逐行核对，方位和余纬差仅为数值误差。

### 4.2 稀疏输入与目标网格

| 项目 | 设置 |
|---|---|
| 原始 reference | Lebedev `N=35`，1730 方向 |
| 稀疏输入 | Lebedev `N=3`，26 方向 |
| 幅度训练/评价网格 | Fliege `N=29`，900 方向 |
| ILD 评价网格 | 水平面 0°–359°，360 方向 |
| HRIR 长度 | 256 |
| 采样率 | 44.1 kHz |
| FFT oversize | 4 |
| 频谱表示 | 1024 点 FFT |

训练名义频率范围为 `50 Hz <= f <= 20 kHz`。由于 FFT 离散栅格，实际包含 463 个频点，范围为：

```text
86.1328125 Hz – 19982.8125 Hz
```

### 4.3 MCA 参数

当前 MCA 生成设置：

```matlab
supdeq_interpHRTF(
    sparseHRTF,
    targetGrid,
    'SUpDEq',
    'SH',
    inf,
    headRadius
)
```

其中：

- `ppMethod='SUpDEq'`；
- `ipMethod='SH'`；
- `mc=inf`；
- 使用默认最小相位 correction；
- correction 限制在空间混叠频率以下的稳定范围；
- 使用 `fadeDown`。

### 4.4 头半径

个体头半径根据 HUTUBS 的 `x1/x2/x3` 与 Algazi 公式计算。公开人体测量表中 pp18、pp79、pp92 缺值，分别位于 test、train、validation。三者使用其余 93 名有效被试的平均头半径：

```text
0.091021 m
```

这是相对论文私有完整元数据的已知偏差。

### 4.5 被试级划分

使用：

```matlab
rng(20260723, 'twister');
randperm(96);
```

得到固定 subject-wise split：

| Split | 被试数 | 样本数 |
|---|---:|---:|
| Train | 72 | `60,004,800` |
| Validation | 12 | `10,000,800` |
| Test | 12 | `10,000,800` |

validation 被试：

```text
pp13, pp35, pp39, pp41, pp51, pp53,
pp55, pp56, pp58, pp74, pp84, pp92
```

test 被试：

```text
pp8, pp18, pp22, pp26, pp31, pp33,
pp45, pp47, pp59, pp70, pp73, pp81
```

同一被试的全部数据只属于一个 split，不存在耳朵、方向或频率级泄漏。

## 5. Residual 数据生成

### 5.1 监督目标

对于 magnitude floor `-200 dB` 后的频谱：

$$\begin{aligned}
L_{\mathrm{MCA}}=20\log_{10}|H_{\mathrm{MCA}}|,
\end{aligned}$$

$$\begin{aligned}
L_{\mathrm{ref}}=20\log_{10}|H_{\mathrm{ref}}|,
\end{aligned}$$

$$\begin{aligned}
r_{\mathrm{target}}=L_{\mathrm{ref}}-L_{\mathrm{MCA}}.
\end{aligned}$$

该 residual 是逐 FFT 频点的细粒度 dB 差，不是 41 个 ERB band 的能量误差。

### 5.2 HDF5 布局

每个被试和稀疏阶数保存一个文件。Python 读取布局为：

```text
[ear=2, direction=900, frequency=463]
```

主要字段：

| HDF5 dataset | 形状 | 含义 |
|---|---|---|
| `/mca_logmag_db` | `[2,900,463]` | MCA log-magnitude |
| `/reference_logmag_db` | `[2,900,463]` | dense reference log-magnitude |
| `/correction_logmag_db` | `[2,900,463]` | MCA correction filter 幅度 |
| `/target_residual_db` | `[2,900,463]` | reference 减 MCA |
| `/direction_features` | `[900,6]` | azimuth、elevation、x、y、z、weight |
| `/frequency_hz` | `[463]` | 频率坐标 |

每个被试包含：

$$\begin{aligned}
2\times900\times463=833{,}400
\end{aligned}$$

个逐频点样本。

### 5.3 数据导出过程

先运行 pp91、N=3 试导出，再使用 6 个 MATLAB process workers 导出全部 96 人：

```matlab
export_hutubs_residual_dataset(91, 3, 1, 'pilot_pp91_n03');
export_hutubs_residual_dataset(1:96, 3, 6, 'hutubs_residual_v1_n03');
```

全量导出耗时 `920.8 s`，约 15 分 21 秒。

### 5.4 完整性检查

导出结果：

- 96/96 HDF5 成功；
- 0 failure；
- 0 partial；
- 80,006,400 个样本；
- 总大小 `1,179,616,094` 字节，约 `1.10 GiB`；
- 全部数组形状一致；
- 全部数值有限；
- split 属性计数为 72/12/12；
- `reference - MCA = residual` 最大恒等误差为 `0 dB`。

写入时先生成 `.partial` 文件，完成数据、属性和校验信息后再原子改名，避免中断文件被误认为完整数据。

### 5.5 训练集统计

归一化只使用 train：

| 变量 | Count | Mean | Std | Mean absolute | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| MCA log-magnitude | 60,004,800 | `0.4367` | `9.3164` | `7.4731` | `-83.1018` | `26.1371` |
| correction log-magnitude | 60,004,800 | `0.8184` | `2.6988` | `1.9160` | `-15.0186` | `21.4613` |
| target residual | 60,004,800 | `-0.0905` | `4.0625` | `2.5156` | `-67.7077` | `69.5672` |

单位均为 dB。频率归一化使用：

| 变量 | Mean | Std |
|---|---:|---:|
| `log10(frequency)` | `3.874925` | `0.410294` |

## 6. 输入特征与归一化

### 6.1 7 维输入

每个耳朵—方向—频率点的输入顺序：

| 索引 | 特征 | 处理 |
|---:|---|---|
| 1 | MCA log-magnitude | train mean/std 标准化 |
| 2 | correction log-magnitude | train mean/std 标准化 |
| 3 | 方向单位向量 `x` | 原值 |
| 4 | 方向单位向量 `y` | 原值 |
| 5 | 方向单位向量 `z` | 原值 |
| 6 | `log10(frequency)` | train mean/std 标准化 |
| 7 | 耳别 | 左 `-1`，右 `+1` |

方向使用单位向量而不是直接使用方位角和仰角，可避免角度在 0°/360° 处的不连续。

### 6.2 输入标准化

以 MCA 为例：

$$\begin{aligned}
x_{\mathrm{MCA}}
=
\frac{L_{\mathrm{MCA}}-\mu_{\mathrm{MCA}}}
{\sigma_{\mathrm{MCA}}}.
\end{aligned}$$

correction 和 log-frequency 同理。

### 6.3 目标标准化

$$\begin{aligned}
y
=
\frac{r_{\mathrm{target}}-\mu_r}{\sigma_r}.
\end{aligned}$$

模型输出归一化 residual $\hat y$，推理时恢复：

$$\begin{aligned}
\hat r=\hat y\sigma_r+\mu_r.
\end{aligned}$$

### 6.4 特征选择说明

- **MCA magnitude**：提供当前基线的局部频谱状态。
- **Correction magnitude**：暴露 MCA 已施加的校正强度。
- **方向 `x/y/z`**：表示空间位置和对侧/同侧关系。
- **log-frequency**：使低频与高频尺度更平滑。
- **耳别**：允许同一网络学习左右耳条件差异。

当前没有加入人体测量、被试 ID 或耳廓参数，因此模型学习的是跨被试共享规律，而不是记忆特定被试。

## 7. 模型选型

### 7.1 为什么选择 MLP

当前单个训练样本只有 7 个标量特征，输出一个标量 residual，适合低维回归。MLP 的优势：

- 结构简单；
- 参数少；
- 对 HDF5 随机 block 友好；
- 不要求固定方向邻域；
- 推理可以按任意方向和频率分块；
- 便于验证 residual 学习路线本身。

首版没有选择 CNN、Transformer 或图网络，因为：

- CNN 需要规则化的方向—频率网格和邻域定义；
- Transformer 需要更大显存和更多数据；
- 图网络需额外定义球面邻接；
- v1 的首要任务是建立可信的最小基线，而不是比较复杂架构。

这不代表 MLP 一定是最终最佳结构。

### 7.2 网络结构

```text
Input: 7 features
    ↓
Linear(7,128) + SiLU
    ↓
Residual Block × 3
    ↓
Linear(128,1)
    ↓
Normalized residual
```

每个 residual block：

```text
x → Linear(128,128) → SiLU → Linear(128,128)
 \__________________________________________+
                         ↓
                       SiLU
```

### 7.3 参数量

| 部分 | 计算 | 参数量 |
|---|---:|---:|
| 输入层 | `7×128+128` | 1,024 |
| 单个 residual block | `2×(128×128+128)` | 33,024 |
| 3 个 residual block | `3×33,024` | 99,072 |
| 输出层 | `128×1+1` | 129 |
| 总计 | — | `100,225` |

### 7.4 激活和残差连接

使用 SiLU：

$$\begin{aligned}
\operatorname{SiLU}(x)=x\sigma(x).
\end{aligned}$$

相对 ReLU，SiLU 在零点附近连续可导，适合回归。残差连接为深层 MLP 提供更直接的梯度路径，并允许 block 学习对已有表示的增量变换。

## 8. 训练采样与优化

### 8.1 ResidualBlockSampler

为避免将 1.10 GiB 数据一次性载入内存，每个 batch：

1. 随机选择一个 train HDF5；
2. 随机选择左耳或右耳；
3. 无放回选择 64 个方向；
4. 无放回选择 128 个频点；
5. 构造方向—频率笛卡尔积。

batch 大小：

$$\begin{aligned}
64\times128=8192.
\end{aligned}$$

这种方式兼顾随机性和 HDF5 连续读取效率。被试、耳朵和 block 在 step 间有放回，因此一个 epoch 不是完整遍历数据集。

### 8.2 损失函数

训练 loss 为归一化目标上的 SmoothL1：

$$\begin{aligned}
\mathcal L
=
\operatorname{SmoothL1}(\hat y,y;\beta=1).
\end{aligned}$$

当误差较小时为二次项，较大时近似 L1，能够降低极端 residual 对梯度的支配。

### 8.3 优化参数

| 参数 | 设置 |
|---|---|
| Optimizer | AdamW |
| 初始学习率 | `1e-3` |
| Weight decay | `1e-5` |
| Scheduler | CosineAnnealingLR |
| Epoch | 12 |
| 每 epoch | 600 steps |
| Validation | 每轮 96 个随机 block |
| Batch size | 8,192 |
| AMP | CUDA FP16 |
| 梯度裁剪 | `max_norm=5` |
| 随机种子 | `20260723` |
| Checkpoint 选择 | 最低 validation MAE |

12 个 epoch 共进行 7200 个训练 step，相当于约 `58,982,400` 次样本呈现。由于随机有放回，这不是独立样本数量。

### 8.4 计算环境

```text
Conda: D:\miniconda3\envs\ml
Python 3.9.23
PyTorch 2.8.0+cu128
CUDA build 12.8
h5py 3.14.0
NVIDIA GeForce RTX 5060
显存 8151 MiB
Compute capability 12.0
```

完整训练耗时 `380.2 s`，约 6 分 20 秒。

## 9. 训练前验证与正式训练

### 9.1 pp91 单被试过拟合检查

在正式跨被试训练前，使用 pp91 同时作为 train 和 validation：

```text
8 epoch × 300 steps
validation steps = 64
```

随机留出 block 的 MCA zero-residual MAE 约为 `2.34 dB`，训练后模型 MAE 约为 `1.60 dB`。该实验不能证明泛化，但验证了：

- HDF5 索引与形状正确；
- 输入和目标归一化正确；
- 模型能从 residual 中学习结构；
- CUDA AMP 前向、反向和梯度缩放有效；
- checkpoint 写入和读取有效。

### 9.2 正式训练历史

| Epoch | Train loss | Train MAE | Sampled val MAE | Sampled MCA val MAE | Learning rate |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.30810` | `2.3996` | `2.3920` | `2.5585` | `1.000e-3` |
| 2 | `0.29232` | `2.3263` | `2.3687` | `2.5745` | `9.830e-4` |
| 3 | `0.28681` | `2.2951` | `2.3819` | `2.6224` | `9.330e-4` |
| 4 | `0.27788` | `2.2408` | `2.2804` | `2.5626` | `8.536e-4` |
| 5 | `0.27502` | `2.2257` | `2.2604` | `2.5602` | `7.500e-4` |
| 6 | `0.26929` | `2.1942` | `2.2776` | `2.5720` | `6.294e-4` |
| 7 | `0.27080` | `2.1965` | `2.2602` | `2.5740` | `5.000e-4` |
| 8 | `0.26378` | `2.1563` | `2.2461` | `2.6032` | `3.706e-4` |
| 9 | `0.26163` | `2.1406` | `2.1749` | `2.5431` | `2.500e-4` |
| **10** | **`0.26226`** | **`2.1416`** | **`2.1618`** | **`2.5420`** | **`1.464e-4`** |
| 11 | `0.26105` | `2.1339` | `2.2038` | `2.5941` | `6.699e-5` |
| 12 | `0.25756` | `2.1142` | `2.1869` | `2.5796` | `1.704e-5` |

单位为 dB，train loss 为归一化 SmoothL1。epoch 10 的 sampled validation MAE 最低，因此选为 `best.pt`。

### 9.3 对训练曲线的解释

训练 MAE 总体下降，但 sampled validation MAE 有波动，原因包括：

- 每轮 validation 只抽取 96 个随机 block；
- block 来自不同被试、耳朵、方向和频率；
- sampled MCA baseline 本身也在 `2.54–2.62 dB` 间波动；
- 不同被试和对侧高频区域的 residual 难度不同。

因此 sampled validation 只用于 checkpoint 选择，最终性能必须通过完整 split 遍历确认。

epoch 12 的 train MAE 低于 epoch 10，但 validation 没有继续改善，说明继续降低训练误差不一定提高跨被试泛化。

## 10. 完整 residual 评价

### 10.1 评价方法

`evaluate_residual_mlp.py` 按 block 遍历 validation 或 test 的每个样本，不使用随机子集：

- 每个 split 12 个被试；
- 每个被试 833,400 个样本；
- 每个 split 共 `10,000,800` 个样本。

预测值反归一化至 dB 后计算：

$$\begin{aligned}
\mathrm{MAE}
=
\frac{1}{N}\sum_i|\hat r_i-r_i|,
\end{aligned}$$

$$\begin{aligned}
\mathrm{RMSE}
=
\sqrt{\frac{1}{N}\sum_i(\hat r_i-r_i)^2}.
\end{aligned}$$

MCA zero-residual baseline 对应 $\hat r_i=0$。

### 10.2 Validation 结果

| 指标 | MCA | MLP v1 | 变化 |
|---|---:|---:|---:|
| MAE | `2.571911 dB` | `2.188647 dB` | 改善 `14.9019%` |
| RMSE | `4.143771 dB` | `3.534563 dB` | 降低 `0.609208 dB` |

### 10.3 Test 结果

| 指标 | MCA | MLP v1 | 变化 |
|---|---:|---:|---:|
| MAE | `2.600671 dB` | `2.217170 dB` | 改善 `14.7462%` |
| RMSE | `4.186115 dB` | `3.579208 dB` | 降低 `0.606908 dB` |

validation 与 test 的 MAE 改善分别为 `14.90%` 和 `14.75%`，非常接近，说明模型在当前固定划分下具有稳定的跨被试泛化。

### 10.4 逐频点指标的边界

该结果证明模型能预测原始 residual，但不能直接等同于听觉效果：

- 所有 FFT 频点等权；
- 不考虑 ERB 频带能量聚合；
- 不显式强调对侧区域；
- 左右耳分别评价；
- 不直接评价 ILD。

因此必须继续执行复数重建和严格指标评价。

## 11. 测试推理与幅度回填

### 11.1 测试输入准备

MATLAB 为 12 个测试被试重新生成：

- complex MCA HRTF；
- complex dense reference HRTF；
- MCA correction filter；
- 900 个 Fliege 方向；
- 360 个水平面方向；
- 463 个网络频率；
- 方向和频率元数据。

每个被试共 1260 个推理方向。

### 11.2 GPU residual 推理

Python 使用：

- v1 epoch 10 `best.pt`；
- train-only normalization；
- RTX 5060；
- CUDA FP16 AMP。

对每个方向、耳朵和频率预测 dB residual。12 个测试被试全部成功完成。测试预测 residual 的总体范围约为 `-9.69 dB` 至 `51.07 dB`，比训练目标的极端范围更集中，表现出一定的回归收缩。

### 11.3 幅度回填

网络覆盖频点：

$$\begin{aligned}
L_{\mathrm{corrected}}
=
L_{\mathrm{MCA}}+\hat r.
\end{aligned}$$

复数重建：

$$\begin{aligned}
H_{\mathrm{corrected}}
=
10^{L_{\mathrm{corrected}}/20}
\exp(j\angle H_{\mathrm{MCA}}).
\end{aligned}$$

20 kHz 以上频点保持原 MCA 不变。重建后通过 IFFT 得到 HRIR。

### 11.4 质量断言

代码要求：

```text
max phase error <= 1e-5 rad
max magnitude identity error <= 1e-4 dB
```

实际 12 人最大值：

| 检查项 | 实际最大值 |
|---|---:|
| 相位保持误差 | `6.22e-16 rad` |
| 幅度回填恒等误差 | `7.11e-15 dB` |

两者均处于浮点误差量级，证明回填没有改变 MCA 相位。

## 12. 最终声学指标

### 12.1 ERB magnitude error

将 MCA、corrected 和 reference HRTF 转换为 HRIR，调用 MATLAB `AKerbError`：

- 频率范围：50 Hz–Nyquist；
- 实际 41 个 ERB band；
- 中心范围约 `50–19792 Hz`；
- 单个 band 计算估计与 reference 的能量差；
- 方向维使用 Fliege 求积权重；
- ERB band 等权平均；
- 左右耳在被试汇总时取平均。

该指标与训练 residual MAE 的量纲相同但计算域不同。

### 12.2 全空间 ERB

使用全部 900 个 Fliege 方向，衡量整体幅度插值质量。

### 12.3 对侧 25° ERB

- 左耳对侧中心：方位角 270°；
- 右耳对侧中心：方位角 90°；
- 使用 25° great-circle 半径。

该区域头部遮挡强，是稀疏球谐插值高频误差较大的区域。

### 12.4 对侧高频幅度误差

定义：

- 左耳使用右侧开放半球；
- 右耳使用左侧开放半球；
- 正中面排除；
- 频率为 `f>10 kHz` 至 Nyquist；
- 单样本为绝对 log-magnitude error；
- 方向按 Fliege 权重；
- 频率等权；
- 左右耳在被试内取平均。

### 12.5 水平面 ILD MAE

对 360 个水平面方向计算：

$$\begin{aligned}
E_L(\Omega)=\sum_n|h_L(n,\Omega)|^2,
\end{aligned}$$

$$\begin{aligned}
E_R(\Omega)=\sum_n|h_R(n,\Omega)|^2,
\end{aligned}$$

$$\begin{aligned}
\mathrm{ILD}(\Omega)
=
10\log_{10}
\frac{E_L(\Omega)}{E_R(\Omega)}.
\end{aligned}$$

最终指标为估计 ILD 与 reference ILD 的平均绝对误差。

### 12.6 为什么暂不以 ITD 作为网络增益指标

v1 只修改幅度并保留 MCA 相位，因此不会主动学习到达时间或 ITD residual。ITD 应主要保持 MCA 水平。当前阶段的重点是验证幅度与双耳能量差的改善。

## 13. 严格测试结果

### 13.1 汇总结果

以下均为 12 个测试被试的均值 ± 被试间标准差：

| 指标 | MCA | MCA + MLP v1 | 绝对改善 | 相对改善 | 改善人数 |
|---|---:|---:|---:|---:|---:|
| 全空间 ERB | `0.8027 ± 0.0255 dB` | `0.6838 ± 0.0326 dB` | `0.1189 dB` | `14.82%` | 12/12 |
| 对侧 25° ERB | `1.8603 ± 0.1159 dB` | `1.5582 ± 0.0971 dB` | `0.3020 dB` | `16.24%` | 12/12 |
| 对侧 `>10 kHz` | `4.2583 ± 0.2641 dB` | `3.7557 ± 0.2420 dB` | `0.5026 dB` | `11.80%` | 12/12 |
| 水平面 ILD MAE | `0.8854 ± 0.2092 dB` | `0.7727 ± 0.3001 dB` | `0.1127 dB` | `12.73%` | 10/12 |

### 13.2 结果解释

1. **逐频点学习转化为严格指标改善**：v1 不只是降低训练目标，回填后的 ERB、高频和平均 ILD 也改善。
2. **对侧区域收益更明显**：对侧 25° ERB 的绝对改善 `0.3020 dB`，大于全空间的 `0.1189 dB`。
3. **高频改善稳定**：12/12 被试的对侧 `>10 kHz` 误差下降，说明特征中包含可用于修正空间混叠残差的信息。
4. **ILD 平均改善但不够稳定**：平均下降 `12.73%`，但只有 10/12 被试改善。
5. **改善幅度存在个体差异**：全空间 ERB 和 ILD 的 corrected 被试间标准差高于 MCA，说明平均误差下降并不代表所有被试得到相同幅度的收益。

### 13.3 标准差变化

| 指标 | MCA std | v1 std | 变化 |
|---|---:|---:|---|
| 全空间 ERB | `0.0255` | `0.0326` | 增加 |
| 对侧 25° ERB | `0.1159` | `0.0971` | 降低 |
| 对侧高频 | `0.2641` | `0.2420` | 降低 |
| ILD MAE | `0.2092` | `0.3001` | 增加 |

对侧 ERB 和高频的误差离散程度下降，是较积极的稳定性信号；ILD 离散度增加则与 pp33、pp81 的退化一致。

## 14. 逐被试结果

| 被试 | 全空间 ERB | 改善 | 对侧 25° ERB | 改善 | 对侧高频 | 改善 | ILD MAE | 改善 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pp8 | `0.6866` | `14.50%` | `1.5286` | `13.62%` | `3.5687` | `12.27%` | `0.7323` | `14.76%` |
| pp18 | `0.6806` | `14.83%` | `1.6556` | `15.60%` | `4.1983` | `12.34%` | `0.5345` | `36.31%` |
| pp22 | `0.6610` | `18.49%` | `1.4605` | `20.27%` | `3.6750` | `14.08%` | `0.4603` | `29.70%` |
| pp26 | `0.6267` | `18.44%` | `1.5085` | `20.47%` | `3.5141` | `11.20%` | `0.7035` | `12.72%` |
| pp31 | `0.6676` | `11.49%` | `1.4831` | `8.76%` | `3.4887` | `12.76%` | `0.6555` | `9.28%` |
| pp33 | `0.7315` | `12.07%` | `1.4626` | `16.95%` | `3.9437` | `9.81%` | `0.8209` | `-5.57%` |
| pp45 | `0.6484` | `18.94%` | `1.5058` | `16.95%` | `3.8366` | `14.47%` | `0.6540` | `19.28%` |
| pp47 | `0.6808` | `15.52%` | `1.6294` | `18.63%` | `3.9679` | `10.26%` | `0.6887` | `43.08%` |
| pp59 | `0.6947` | `18.54%` | `1.4992` | `19.21%` | `3.7719` | `13.78%` | `0.7923` | `5.55%` |
| pp70 | `0.6761` | `15.36%` | `1.5302` | `16.20%` | `3.4041` | `11.56%` | `0.7602` | `15.47%` |
| pp73 | `0.7117` | `10.54%` | `1.7598` | `9.33%` | `4.0059` | `7.86%` | `0.8084` | `0.24%` |
| pp81 | `0.7398` | `8.92%` | `1.6757` | `17.82%` | `3.6937` | `11.17%` | `1.6622` | `-18.97%` |

单位均为 dB，改善率相对每个被试自己的 MCA baseline。

### 14.1 最佳与最弱案例

- 全空间 ERB 最大改善：pp45，`18.94%`。
- 对侧 25° ERB 最大改善：pp26，`20.47%`。
- 对侧高频最大改善：pp45，`14.47%`。
- ILD 最大改善：pp47，`43.08%`。
- 全空间 ERB 最小改善：pp81，`8.92%`。
- 对侧 25° ERB 最小改善：pp31，`8.76%`。
- 对侧高频最小改善：pp73，`7.86%`。

所有最小改善仍为正值，说明 v1 对单耳幅度相关的三项指标具有较强稳定性。

### 14.2 ILD 失败案例

| 被试 | MCA ILD MAE | v1 ILD MAE | 变化 |
|---|---:|---:|---:|
| pp33 | `0.7777 dB` | `0.8209 dB` | 退化 `5.57%` |
| pp81 | `1.3971 dB` | `1.6622 dB` | 退化 `18.97%` |

pp73 只改善 `0.24%`，接近无变化。

### 14.3 ILD 退化原因

v1 的结构允许左右耳共享参数，但训练 batch 中只出现一个耳朵，loss 也只约束单耳逐频点 residual。因此：

- 左右耳不会在同一个 loss 中共同出现；
- 模型不知道同方向两耳的宽带能量差；
- 单耳 MAE 下降不保证两耳误差相互抵消；
- 少量系统性耳间偏差就可能放大 ILD error。

因此 pp33 和 pp81 的失败更像训练目标缺失，而不是单纯模型容量不足。

## 15. 可视化结果

v1 输出：

- 12 张逐被试四面板图；
- 1 张 12 人左耳对侧 HRTF 总览；
- 1 张逐被试指标总览。

主要入口：

- [12 人对侧 HRTF 总览](../results/residual_mlp/mlp_n03_v1/evaluation/figures/test12_contralateral_hrtf_overview.png)
- [12 人指标总览](../results/residual_mlp/mlp_n03_v1/evaluation/figures/test12_metric_overview.png)
- [pp33 重建对比](../results/residual_mlp/mlp_n03_v1/evaluation/figures/pp33_reconstruction_comparison.png)
- [pp81 重建对比](../results/residual_mlp/mlp_n03_v1/evaluation/figures/pp81_reconstruction_comparison.png)

每张逐被试图包括：

1. 左耳对侧 270° HRTF；
2. 右耳对侧 90° HRTF；
3. 全空间 ERB error 曲线；
4. 水平面 360° ILD 曲线。

可视化用于检查平均指标之外的频谱形状和方向性异常，不能代替定量评价。

## 16. 工程实现与结果文件

### 16.1 主要源码

| 文件 | 作用 |
|---|---|
| `matlab/+mcar/export_hutubs_residual_dataset.m` | residual HDF5 导出 |
| `src/mcar/data_tools/validate_residual_hdf5.py` | HDF5 完整性检查 |
| `src/mcar/data_tools/compute_training_statistics.py` | train-only 统计 |
| `src/mcar/data.py` | block sampler 和特征构造 |
| `src/mcar/models/residual_mlp.py` | ResidualMLP |
| `src/mcar/training/train_mlp_v1.py` | v1 训练 |
| `src/mcar/evaluation/evaluate_residual_mlp.py` | 完整 residual 评价 |
| `matlab/+mcar/prepare_test_reconstruction_inputs.m` | 测试 complex cache |
| `src/mcar/evaluation/predict_reconstructed_residuals.py` | GPU residual 推理 |
| `matlab/+mcar/evaluate_test_reconstruction.m` | 回填、指标与绘图 |

### 16.2 轻量结果

- [训练历史](../results/residual_mlp/mlp_n03_v1/training/history.csv)
- [Validation residual 指标](../results/residual_mlp/mlp_n03_v1/training/val_metrics.json)
- [Test residual 指标](../results/residual_mlp/mlp_n03_v1/training/test_metrics.json)
- [严格指标汇总](../results/residual_mlp/mlp_n03_v1/evaluation/aggregate_metrics.csv)
- [逐被试指标](../results/residual_mlp/mlp_n03_v1/evaluation/per_subject_metrics.csv)
- [ERB 明细](../results/residual_mlp/mlp_n03_v1/evaluation/erb_summary.csv)
- [对侧高频明细](../results/residual_mlp/mlp_n03_v1/evaluation/contralateral_high_frequency_summary.csv)
- [ILD 明细](../results/residual_mlp/mlp_n03_v1/evaluation/ild_summary.csv)
- [重建质量](../results/residual_mlp/mlp_n03_v1/evaluation/reconstruction_quality.csv)

大型 HDF5、SOFA、checkpoint、本机配置、complex cache、预测 HDF5 和 MAT 中间结果由 `.gitignore` 忽略。

## 17. 复现流程

### 17.1 数据导出

```matlab
addpath(fullfile(pwd, 'residual_learning', 'matlab'));
export_hutubs_residual_dataset(1:96, 3, 6, ...
    'hutubs_residual_v1_n03');
```

### 17.2 数据验证与统计

```powershell
$files = Get-ChildItem `
  residual_learning/data/hutubs_residual_v1_n03/subjects `
  -Recurse -Filter *.h5

residual_learning/.venv/Scripts/python `
  residual_learning/python/validate_residual_hdf5.py `
  --summary-only $files.FullName

residual_learning/.venv/Scripts/python `
  residual_learning/python/compute_training_statistics.py `
  residual_learning/data/hutubs_residual_v1_n03
```

### 17.3 单被试检查

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/python/train_residual_mlp.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  --run-name overfit_pp91 `
  --overfit-subject 91 `
  --epochs 8 `
  --steps-per-epoch 300 `
  --validation-steps 64
```

### 17.4 正式训练

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/python/train_residual_mlp.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  --run-name mlp_n03_v1 `
  --epochs 12 `
  --steps-per-epoch 600 `
  --validation-steps 96 `
  --directions-per-batch 64 `
  --frequencies-per-batch 128 `
  --width 128 `
  --block-count 3 `
  --learning-rate 1e-3 `
  --weight-decay 1e-5 `
  --seed 20260723
```

### 17.5 完整 validation/test residual 评价

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/python/evaluate_residual_mlp.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/runs/mlp_n03_v1/best.pt `
  --split val

D:\miniconda3\envs\ml\python.exe `
  residual_learning/python/evaluate_residual_mlp.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/runs/mlp_n03_v1/best.pt `
  --split test
```

### 17.6 测试回填和严格评价

```powershell
matlab -batch "addpath('residual_learning/matlab'); prepare_test_reconstruction_inputs(4, 'mlp_n03_v1')"

D:\miniconda3\envs\ml\python.exe `
  residual_learning/python/predict_reconstructed_residuals.py `
  residual_learning/reconstruction/mlp_n03_v1 `
  residual_learning/runs/mlp_n03_v1/best.pt `
  residual_learning/data/hutubs_residual_v1_n03/training_statistics.json

matlab -batch "addpath('residual_learning/matlab'); evaluate_test_reconstruction('mlp_n03_v1')"
```

## 18. 有效性威胁与局限

### 18.1 单次随机种子

当前 v1 只有一次正式训练，尚未报告：

- 多随机种子均值和标准差；
- bootstrap confidence interval；
- 配对显著性检验；
- 效应量。

随机 block 采样和 GPU 计算均可能带来运行方差。

### 18.2 固定数据域

当前结果只覆盖：

- HUTUBS simulated；
- Lebedev `N=3`；
- 固定 MCA 参数；
- 当前 subject split；
- 当前频率与目标网格。

不能直接推出对 HUTUBS measured、AXD、其他数据库或其他稀疏阶数同样有效。

### 18.3 逐频点单耳训练

v1 没有：

- 完整频谱 loss；
- ERB 感知 loss；
- 对侧区域专门 loss；
- 双耳联合 batch；
- ILD loss。

因此训练目标与最终声学指标只有间接关系。

### 18.4 相位和时间线索

模型只修正 magnitude，不能改善：

- 相位 residual；
- 群时延；
- HRIR onset；
- ITD；
- 非最小相位结构。

这是当前方法有意设置的边界。

### 18.5 缺失人体测量

pp18、pp79、pp92 使用平均头半径。尽管三者分散在三个 split，但替代值可能影响 MCA baseline 和 residual 分布。获得完整值后应进行敏感性重跑。

### 18.6 测试集规模

test 只有 12 个被试。虽然每人包含大量方向和频率，但真正独立的统计单位仍是被试，不能把 1000 万逐频点样本当作 1000 万个独立个体证据。

### 18.7 未进行架构和特征消融

当前没有独立量化：

- correction filter 特征的贡献；
- ear flag 的贡献；
- residual block 相对普通 MLP 的贡献；
- width 和 block count 的影响；
- MCA magnitude 与方向特征的交互。

因此可以证明整个 v1 系统有效，但不能将增益归因于某个单独组件。

## 19. 后续实验建议

### 19.1 双耳指标感知训练

最直接的改进是：

- 一个 batch 同时包含左右耳；
- 保留完整频谱；
- 加入 ERB、对侧高频和 ILD 代理损失；
- 保持 v1 网络参数量不变，以隔离训练策略贡献。

### 19.2 v1 特征消融

建议比较：

| 实验 | 输入 |
|---|---|
| F0 | MCA magnitude + direction + frequency + ear |
| F1 | F0 + correction filter |
| F2 | direction + frequency + ear，不含 MCA magnitude |
| F3 | MCA magnitude + correction，不含方向 |
| F4 | 完整 7 特征 |

### 19.3 模型消融

保持训练预算相同，比较：

- 普通 3–4 层 MLP；
- residual MLP；
- width 64/128/256；
- block count 1/3/5；
- 参数量匹配的一维频谱 CNN。

### 19.4 多随机种子

建议运行 3–5 个种子，报告：

- validation/test mean ± std；
- 12 人配对差值；
- 95% bootstrap interval；
- 改善人数；
- pp33、pp81 是否稳定退化。

### 19.5 不同稀疏阶数

优先扩展：

```text
N = 1, 2, 4, 6
```

以判断网络收益随稀疏度的变化，并与 96 被试 MCA 基线曲线对应。

### 19.6 外部泛化

模型与超参数冻结后，建议依次测试：

1. HUTUBS measured；
2. AXD；
3. 新 subject holdout；
4. 不同方向网格或采样率。

## 20. 项目结论

MLP v1 使用 7 维输入和 `100,225` 个参数，在严格的 subject-wise 划分下成功学习 MCA 后的 log-magnitude residual。

核心结果：

- 完整 validation residual MAE 改善 `14.90%`；
- 完整 test residual MAE 改善 `14.75%`；
- 全空间 ERB 改善 `14.82%`，12/12 被试有效；
- 对侧 25° ERB 改善 `16.24%`，12/12 被试有效；
- 对侧高频误差改善 `11.80%`，12/12 被试有效；
- 平均 ILD MAE 改善 `12.73%`，10/12 被试有效；
- 相位保持误差仅为 `6.22e-16 rad`。

v1 证明了以下技术路线成立：

> 使用 MCA 保留物理先验和主要插值能力，再用轻量 MLP 学习剩余幅度误差，可以在不修改相位和不过度增加复杂度的情况下，进一步改善未见被试的 HRTF 幅度插值质量。

同时，pp33 和 pp81 的 ILD 退化明确揭示了 v1 的边界：单耳逐频点 loss 不能保证双耳能量关系。后续优化应优先修正训练目标和 batch 组织，而不是直接扩大模型。
