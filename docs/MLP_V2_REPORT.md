# MCA Residual MLP v2 完整设计与实验报告

> 数据集：HUTUBS simulated HRTF，96 个被试
>
> 稀疏输入：Lebedev `N=3`，26 个方向
>
> 模型：Residual MLP v2，`100,225` 个可训练参数
>
> 训练方式：从 v1 最优模型初始化的双耳指标感知微调
>
> 实验日期：2026-07-24

## 摘要

MLP v1 已经证明，MCA（Magnitude-Corrected and Time-Aligned Interpolation）完成 HRTF 插值后，dense reference 与 MCA 之间仍存在能够跨被试学习的 log-magnitude residual。然而，v1 使用单耳、局部方向—频率 block 和逐频点 SmoothL1 loss，没有显式建模完整频谱的听觉频带能量，也没有约束左右耳之间的能量关系。虽然 v1 在测试集平均意义上改善了 ERB magnitude error、对侧高频误差和 ILD，但 pp33 与 pp81 的 ILD 分别退化 `5.57%` 和 `18.97%`。

MLP v2 针对这一问题，在不增加网络参数量和输入特征的前提下，重新组织训练 batch，并引入听觉指标感知的复合损失。每个 v2 batch 同时读取一个被试、32 个相同空间方向、左右双耳以及全部 463 个训练频点，张量形状为 `[2,32,463]`。完整双耳频谱使模型能够同时优化：

1. 逐频点 residual SmoothL1；
2. 41 个 ERB-rate 三角频带上的能量误差代理；
3. 每耳对侧开放半球 `>10 kHz` 的幅度误差；
4. 同方向左右耳宽带能量比形成的 ILD 误差代理。

v2 保持 v1 的 7 维输入、宽度 128、3 个 SiLU residual block 和单输出结构，共 `100,225` 个参数。从 v1 epoch 10 的最优 checkpoint 初始化全部网络权重，但新建 AdamW 优化器，以 `3e-4` 学习率进行 10 epoch 微调。根据复合 validation loss 选择 epoch 9。

在严格未见训练过程的 12 个测试被试上，v2 将逐频点 residual MAE 从 MCA 的 `2.6007 dB` 降至 `2.1684 dB`，相对改善 `16.62%`，同时优于 v1 的 `2.2172 dB`。将预测 residual 回填到 MCA 幅度并保留 MCA 原相位后，v2 得到：

| 指标，越低越好 | MCA | v1 | v2 | v2 相对 MCA | v2 相对 v1 |
|---|---:|---:|---:|---:|---:|
| 全空间 ERB | `0.8027 dB` | `0.6838 dB` | `0.6115 dB` | `23.82%` | `10.57%` |
| 对侧 25° ERB | `1.8603 dB` | `1.5582 dB` | `1.3401 dB` | `27.96%` | `14.00%` |
| 对侧 `>10 kHz` 幅度误差 | `4.2583 dB` | `3.7557 dB` | `3.7204 dB` | `12.63%` | `0.94%` |
| 水平面 ILD MAE | `0.8854 dB` | `0.7727 dB` | `0.6467 dB` | `26.96%` | `16.31%` |

四项严格指标均为 12/12 测试被试优于 MCA。pp33 的 ILD 从 v1 相对 MCA 退化 `5.57%` 变为改善 `14.60%`；pp81 从退化 `18.97%` 变为改善 `0.52%`。结果表明，双耳完整频谱采样与指标感知损失在不增加模型容量的情况下，显著增强了 ERB 和 ILD 表现，并解决了 v1 的主要逐被试失败案例。

需要强调：训练中的 ERB 和 ILD 项是为反向传播设计的代理指标，不等同于最终 MATLAB `AKerbError` 和完整 HRIR 能量 ILD。此外，v2 的设计受到 v1 在同一测试集上的失败案例启发。测试被试没有参与梯度或归一化，但该测试集已对研究决策产生反馈，因此 v2 结果应视为内部开发结果；正式论文还应在模型与超参数冻结后，增加新的外部数据或重新保留的最终测试集。

## 1. 研究背景

### 1.1 MCA 后残差学习

MCA 结合时间对齐、球谐空间插值和幅度校正，是本项目的强传统基线。残差网络不从零生成 HRTF，而是预测：

$$
r_{\mathrm{target}}(s,e,\Omega,f)
=
L_{\mathrm{ref}}(s,e,\Omega,f)
-
L_{\mathrm{MCA}}(s,e,\Omega,f),
$$

其中

$$
L=20\log_{10}|H|.
$$

模型输出 $\hat r$ 后，修正幅度为：

$$
\widehat L_{\mathrm{corrected}}
=
L_{\mathrm{MCA}}+\hat r.
$$

如果模型输出零，系统自然退化为原 MCA。因此 MCA 也是 residual 模型的 zero-residual baseline。

### 1.2 v1 已解决的问题

v1 使用逐方向、逐频率、逐耳朵的 7 维特征，证明了以下结论：

- MCA 后 residual 不是完全随机噪声；
- 约 10 万参数的 MLP 可以跨被试学习 residual；
- validation 与 test 的逐频点 MAE 均比 MCA 低约 15%；
- 回填后的全空间 ERB、对侧 ERB 和对侧高频误差均为 12/12 测试被试改善；
- 平均 ILD 得到改善，但 pp33 和 pp81 出现退化。

### 1.3 v1 的结构性不足

v1 每个 batch 只包含一个耳朵，并只随机抽取 128 个频点。其训练目标为单个频点的 residual SmoothL1：

$$
\mathcal L_{\mathrm{v1}}
=
\operatorname{SmoothL1}(\hat r_{\mathrm{norm}},r_{\mathrm{norm}}).
$$

这种训练方式存在三个问题：

1. **没有完整频谱上下文**：无法直接计算听觉频带能量误差。
2. **没有左右耳配对**：无法直接形成 ILD 损失。
3. **训练目标与最终指标不完全一致**：逐频点误差下降不保证 ERB 或 ILD 同比例下降。

v2 的设计目标不是换用更大的模型，而是让同一模型在训练时看到计算最终听觉代理指标所需的数据结构。

## 2. 需求说明

### 2.1 功能需求

v2 应满足：

- 复用 v1 的 HDF5 数据、被试划分和归一化统计量；
- 保持 v1 的输入特征、网络结构和参数量；
- 从 v1 最优 checkpoint 初始化全部模型参数；
- 一个 batch 内包含配对双耳和完整训练频谱；
- 同时优化 residual、ERB、高频和 ILD；
- 训练过程中四项损失及梯度均保持有限；
- 模型选择只使用 validation split；
- 最终测试仍使用 MATLAB 严格指标；
- 回填时只修改 MCA 幅度，不修改相位；
- 对 12 个测试被试分别检查，不只报告平均值。

### 2.2 性能目标

相对 v1，v2 的目标为：

1. test 逐频点 residual MAE 不退化；
2. 全空间 ERB 和对侧 25° ERB 进一步降低；
3. 对侧高频误差至少不劣于 v1；
4. 平均 ILD MAE 进一步降低；
5. pp33 与 pp81 的 ILD 不再劣于 MCA；
6. 四项严格指标尽可能达到 12/12 被试改善。

### 2.3 非功能需求

- 模型保持轻量，参数量不得高于 v1；
- batch 适合 8 GB 消费级 GPU；
- 大型 HDF5、checkpoint、MAT cache 和预测中间文件不提交 Git；
- 保留训练 history、validation/test JSON、最终 CSV 和结果图；
- 明确区分训练代理指标和最终严格指标；
- 记录随机种子、环境、耗时和 checkpoint 选择依据。

## 3. 数据集与实验协议

### 3.1 数据来源

使用 96 个 HUTUBS simulated HRTF。每个原始 SOFA 包含：

- Lebedev `N=35` 的 1730 个方向；
- 左右耳各 256 点 HRIR；
- 采样率 `44.1 kHz`。

### 3.2 MCA 和采样网格

| 项目 | 设置 |
|---|---|
| 稀疏输入 | Lebedev `N=3`，26 方向 |
| 幅度目标 | Fliege `N=29`，900 方向 |
| ILD 目标 | 水平面 0°–359°，360 方向 |
| 频谱表示 | 1024 点 FFT |
| 网络频率 | 463 点，`86.1328–19982.8125 Hz` |
| MCA | SUpDEq + SH + `mc=inf` |

MCA 使用最小相位幅度校正、空间混叠频率限制和 `fadeDown`。测试重建共处理 1260 个方向，即 900 个 Fliege 方向和 360 个水平面方向。

### 3.3 被试划分

固定划分由 MATLAB `rng(20260723,'twister'); randperm(96)` 生成：

| Split | 被试数 | 用途 |
|---|---:|---|
| Train | 72 | 梯度训练与归一化统计 |
| Validation | 12 | checkpoint 选择 |
| Test | 12 | 最终重建评价 |

测试被试为：

```text
pp8, pp18, pp22, pp26, pp31, pp33,
pp45, pp47, pp59, pp70, pp73, pp81
```

同一被试的左右耳、方向和频率全部位于同一个 split。pp18 缺少公开人体测量值，沿用 MCA 基线规则，使用 93 名有效被试的平均 Algazi 头半径 `0.091021 m`。

### 3.4 HDF5 数据布局

每个被试文件的主要频谱张量为：

```text
[ear=2, direction=900, frequency=463]
```

主要字段：

| 字段 | 含义 | 单位 |
|---|---|---|
| `mca_logmag_db` | MCA log-magnitude | dB |
| `reference_logmag_db` | dense reference log-magnitude | dB |
| `correction_logmag_db` | MCA correction filter 幅度 | dB |
| `target_residual_db` | reference 减 MCA | dB |
| `direction_features` | azimuth、elevation、x、y、z、Fliege weight | degree / unitless |
| `frequency_hz` | FFT 正频率 | Hz |

每个被试有 `833,400` 个逐频点样本，96 个被试共 `80,006,400` 个样本，约 `1.10 GiB`。

### 3.5 归一化

所有统计量只由 72 个训练被试、`60,004,800` 个样本计算：

| 变量 | 均值 | 标准差 |
|---|---:|---:|
| MCA log-magnitude | `0.436740 dB` | `9.316422 dB` |
| correction log-magnitude | `0.818364 dB` | `2.698846 dB` |
| target residual | `-0.090459 dB` | `4.062468 dB` |
| `log10(frequency)` | `3.874925` | `0.410294` |

目标归一化为：

$$
r_{\mathrm{norm}}
=
\frac{r-\mu_r}{\sigma_r}.
$$

validation 和 test 不参与任何均值或标准差估计。

## 4. 模型选型与网络结构

### 4.1 为什么保持 v1 模型不变

v2 的核心假设是：v1 的容量已经足以学习 residual，主要瓶颈在于数据组织和损失目标，而不是网络过小。保持结构不变有三点价值：

- v1 与 v2 的差异更容易归因于训练策略；
- 参数量和推理成本不增加；
- 可以直接从 v1 checkpoint 微调，减少重新学习基础 residual 的成本。

### 4.2 输入与输出

每个耳朵—方向—频率点的 7 维输入依次为：

1. 归一化 MCA log-magnitude；
2. 归一化 MCA correction-filter log-magnitude；
3. 方向单位向量 $x$；
4. 方向单位向量 $y$；
5. 方向单位向量 $z$；
6. 归一化 $\log_{10}(f)$；
7. 耳别，左耳为 `-1`，右耳为 `+1`。

模型输出一个归一化 dB residual。

### 4.3 网络结构

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

每个 residual block 为：

```text
x → Linear(128,128) → SiLU → Linear(128,128)
 \__________________________________________+
                         ↓
                       SiLU
```

参数量：

| 部分 | 参数量 |
|---|---:|
| 输入层 `Linear(7,128)` | 1,024 |
| 3 个 residual block | 99,072 |
| 输出层 `Linear(128,1)` | 129 |
| 总计 | `100,225` |

### 4.4 初始化方式

v2 从 v1 epoch 10 的 `best.pt` 加载模型权重：

```text
v1 best model weights
    ↓
load_state_dict
    ↓
all v2 parameters remain trainable
```

v2 不继承 v1 的 optimizer state，而是新建 AdamW 优化器。这使网络保留 v1 已学到的 residual 映射，同时以新的较低学习率适应复合指标目标。

## 5. 双耳完整频谱采样

### 5.1 v1 与 v2 batch 对比

| 属性 | v1 | v2 |
|---|---|---|
| 每 batch 被试 | 1 | 1 |
| 耳朵 | 随机单耳 | 左右双耳同时 |
| 方向 | 64 | 32 |
| 频率 | 随机 128 | 全部 463 |
| 样本数 | 8,192 | 29,632 |
| 可计算 ERB 代理 | 不完整 | 是 |
| 可计算 ILD 代理 | 否 | 是 |

### 5.2 v2 张量布局

`BinauralSpectrumSampler` 每次随机选择：

- 一个训练被试；
- 32 个不重复的 Fliege 方向；
- 左右两只耳朵；
- 全部 463 个频率。

得到：

```text
features:          [2, 32, 463, 7]
target residual:   [2, 32, 463]
MCA magnitude:     [2, 32, 463]
direction feature: [32, 6]
frequency:         [463]
```

总逐频点样本数为：

$$
2\times32\times463=29{,}632.
$$

这一布局的关键不是增加样本量，而是保持同一方向的双耳与完整频谱关系，使频带能量和双耳能量差保持可微。

## 6. 复合损失函数

### 6.1 总损失

v2 的总损失为：

$$
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{res}}
+0.50\frac{\mathcal L_{\mathrm{ERB}}}{\sigma_r}
+0.25\frac{\mathcal L_{\mathrm{HF}}}{\sigma_r}
+0.25\frac{\mathcal L_{\mathrm{ILD}}}{\sigma_r},
$$

其中 $\sigma_r=4.062468\ \mathrm{dB}$ 是训练集 target residual 标准差。除 residual loss 已在归一化空间计算外，其余 dB 指标除以 $\sigma_r$，使各项进入相近的数值尺度。

这些权重是首版经验设置，不代表各听觉指标的重要性比例已经通过系统搜索确定。

### 6.2 逐频点 residual loss

$$
\mathcal L_{\mathrm{res}}
=
\operatorname{SmoothL1}
(\hat r_{\mathrm{norm}},r_{\mathrm{norm}};\beta=1).
$$

该项保留 v1 的基本学习目标，防止模型只追求聚合指标而牺牲局部频谱准确度。

### 6.3 ERB 代理损失

首先定义 ERB-rate：

$$
E(f)=21.4\log_{10}(1+0.004367f).
$$

在 $E(50)$ 至 $E(20000)$ 之间均匀放置 41 个中心。对第 $b$ 个频带构造三角权重：

$$
w_b(f)
=
\max\left(
0,\,
1-\frac{|E(f)-c_b|}{1.5\Delta c}
\right),
$$

并在频率维归一化。对于 log-magnitude $L(f)$，频带能量 dB 为：

$$
B_b
=
10\log_{10}
\sum_f w_b(f)10^{L(f)/10}.
$$

修正后频谱与 reference 的 ERB 代理误差为：

$$
\mathcal L_{\mathrm{ERB}}
=
\operatorname{Mean}_{e,\Omega,b}
\left|
\widehat B_{e,\Omega,b}
-
B^{\mathrm{ref}}_{e,\Omega,b}
\right|.
$$

方向维使用 Fliege 权重。该项鼓励模型在听觉频带能量层面接近 reference。

### 6.4 对侧高频损失

对侧方向由方向单位向量的横向分量 $y$ 判断：

- 左耳对侧：$y<0$；
- 右耳对侧：$y>0$；
- 正中面不属于任何开放半球。

高频掩码为 $f>10\ \mathrm{kHz}$。由于

$$
L_{\mathrm{corrected}}-L_{\mathrm{ref}}
=
\hat r-r_{\mathrm{target}},
$$

损失可直接写为对侧高频 residual 误差的加权 MAE：

$$
\mathcal L_{\mathrm{HF}}
=
\operatorname{Mean}_{e,\Omega\in\mathrm{contra},f>10\,\mathrm{kHz}}
|\hat r-r_{\mathrm{target}}|.
$$

方向按 Fliege 权重归一化，频率等权。

### 6.5 ILD 代理损失

对每个耳朵和方向，从完整训练频谱计算宽带能量 dB：

$$
P_e(\Omega)
=
10\log_{10}\sum_f10^{L_e(\Omega,f)/10}.
$$

代理 ILD 为：

$$
\mathrm{ILD}_{\mathrm{proxy}}(\Omega)
=
P_L(\Omega)-P_R(\Omega).
$$

损失为：

$$
\mathcal L_{\mathrm{ILD}}
=
\sum_\Omega \widetilde w_\Omega
\left|
\widehat{\mathrm{ILD}}_{\mathrm{proxy}}(\Omega)
-
\mathrm{ILD}^{\mathrm{ref}}_{\mathrm{proxy}}(\Omega)
\right|.
$$

该项使同方向左右耳的总体校正保持协调，是 v2 修复 v1 ILD 失败的核心约束。

### 6.6 代理指标与严格指标的区别

训练代理与最终指标不可混为一谈：

| 项目 | 训练代理 | 最终严格评价 |
|---|---|---|
| ERB | 41 个 ERB-rate 三角带，频域能量 | MATLAB `AKerbError`，由重建 HRIR 计算 |
| ILD | 选定 463 个频点的频域能量比 | 完整 HRIR 能量比 |
| 方向 | 每 batch 32 个 Fliege 方向 | 900 Fliege + 360 水平面方向 |
| 用途 | 可微优化和 validation 选择 | 最终报告 |

代理指标只要求与最终目标方向相关、梯度稳定，而不是数值完全相等。

## 7. 训练设置与验证

### 7.1 优化配置

| 参数 | 设置 |
|---|---|
| 初始化 | v1 epoch 10 `best.pt` |
| Optimizer | AdamW |
| 初始学习率 | `3e-4` |
| Weight decay | `1e-5` |
| Scheduler | CosineAnnealingLR |
| Epoch | 10 |
| 每 epoch | 500 steps |
| Validation | 每轮 96 个由固定种子采样的 batch，轮间样本不同 |
| AMP | CUDA FP16 |
| 梯度裁剪 | `max_norm=5` |
| 随机种子 | `20260724` |
| Checkpoint 选择 | 最低 validation total loss |

### 7.2 计算环境

```text
Python 3.9.23
PyTorch 2.8.0+cu128
CUDA build 12.8
h5py 3.14.0
NVIDIA GeForce RTX 5060
显存 8151 MiB
```

完整训练耗时 `331.265 s`，约 5 分 31 秒，平均每个 epoch 约 33.1 秒。

### 7.3 数值与梯度检查

正式训练前完成：

- 真实 HDF5 batch 前向计算；
- residual、ERB、高频、ILD 四项数值有限；
- 总损失反向传播成功；
- 所有模型梯度有限；
- 16 方向检查的 CUDA 峰值分配约 `134.4 MiB`。

这些检查验证了 logsumexp 能量计算、双耳索引、区域掩码和 AMP 训练链路。

### 7.4 pp91 短程过拟合检查

pp91 运行 4 epoch × 120 steps。validation 代理指标：

| Epoch | Residual MAE | ERB proxy | 对侧高频 | ILD proxy |
|---:|---:|---:|---:|---:|
| 1 | `1.694 dB` | `0.603 dB` | `2.538 dB` | `0.411 dB` |
| 4 | `1.618 dB` | `0.561 dB` | `2.331 dB` | `0.358 dB` |

四项均下降，说明复合损失确实能驱动模型在期望方向上更新。

## 8. 正式训练过程

### 8.1 v1 初始化点

在单独记录的确定性 validation 采样序列上，尚未微调的 v1 checkpoint 得到：

| 模型 | Total loss | Residual MAE | ERB proxy | 对侧高频 | ILD proxy |
|---|---:|---:|---:|---:|---:|
| v1 初始化 | `0.608981` | `2.1637 dB` | `0.7731 dB` | `3.3008 dB` | `0.7390 dB` |

这组数值是 v2 的直接起点，用于判断指标感知微调是否真正产生增益。

### 8.2 每轮 validation 结果

| Epoch | Total loss | Residual MAE | ERB proxy | 对侧高频 | ILD proxy |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.60461` | `2.1553` | `0.7446` | `3.3092` | `0.7175` |
| 2 | `0.60219` | `2.1539` | `0.7224` | `3.3085` | `0.6783` |
| 3 | `0.60600` | `2.1653` | `0.7305` | `3.3293` | `0.6857` |
| 4 | `0.61293` | `2.1898` | `0.7333` | `3.3441` | `0.6943` |
| 5 | `0.60082` | `2.1588` | `0.7148` | `3.3117` | `0.6522` |
| 6 | `0.60960` | `2.1702` | `0.7225` | `3.3705` | `0.6825` |
| 7 | `0.58952` | `2.1230` | `0.6929` | `3.2708` | `0.6558` |
| 8 | `0.59122` | `2.1245` | `0.7008` | `3.2849` | `0.6590` |
| **9** | **`0.58462`** | **`2.1168`** | **`0.6933`** | **`3.2668`** | **`0.6086`** |
| 10 | `0.60015` | `2.1606` | `0.7061` | `3.3172` | `0.6465` |

单位为 dB，total loss 为无量纲复合值。epoch 9 的 validation total loss 最低，因此被选为最终 checkpoint。

validation 曲线并非单调下降，原因包括：

- 每轮只评估 96 个随机方向 batch；
- 一个 batch 只来自一个被试；
- 不同被试和方向的 residual 难度差异较大；
- 复合 loss 的各项可能存在局部权衡。

因此 checkpoint 选择依赖固定 validation 采样器和整体复合目标，而不是最后一轮或单一训练 loss。

### 8.3 v1 起点到 v2 最优点

将 v1 起点记录与 v2 最优轮的采样式 validation 均值作描述性对比：

| 指标 | v1 起点 | v2 epoch 9 | 变化 |
|---|---:|---:|---:|
| Total loss | `0.608981` | `0.584618` | 降低 `4.00%` |
| Residual MAE | `2.1637` | `2.1168` | 降低 `2.17%` |
| ERB proxy | `0.7731` | `0.6933` | 降低 `10.32%` |
| 对侧高频 | `3.3008` | `3.2668` | 降低 `1.03%` |
| ILD proxy | `0.7390` | `0.6086` | 降低 `17.64%` |

ERB 和 ILD 的代理改善明显大于对侧高频，这一趋势与最终严格指标基本一致。但每轮 validation 继续从同一固定种子的随机数序列向后采样，并非在完全相同的 96 个 batch 上重复评价，因此表中的变化只能用于观察趋势，不能视为严格配对改善率。完整 val/test 遍历和 MATLAB 严格评价才是最终依据。

## 9. 推理、幅度回填与复数重建

### 9.1 推理范围

对 12 个测试被试分别预测：

- 900 个 Fliege 方向；
- 360 个水平面方向；
- 左右双耳；
- 463 个训练频率。

即每个被试共 1260 个方向。v2 复用 v1 已生成的 complex MCA/reference cache 和模型输入，只替换 checkpoint 与预测 residual。

### 9.2 幅度回填

对于网络覆盖的频率：

$$
\widehat L_{\mathrm{corrected}}
=
L_{\mathrm{MCA}}+\hat r.
$$

复数 HRTF 使用 MCA 相位：

$$
\widehat H_{\mathrm{corrected}}
=
10^{\widehat L_{\mathrm{corrected}}/20}
\exp\left(j\angle H_{\mathrm{MCA}}\right).
$$

20 kHz 以上没有参与网络预测的频点保持原 MCA 不变。之后通过 IFFT 得到 HRIR，供 `AKerbError` 和完整 HRIR 能量 ILD 使用。

### 9.3 重建质量

12 个测试被试的最大误差：

| 检查项 | 最大值 |
|---|---:|
| 相位保持误差 | `6.29e-16 rad` |
| 幅度回填恒等误差 | `7.11e-15 dB` |

两者均为浮点数误差量级，证明网络只改变了指定频点的幅度，没有意外修改 MCA 相位。

## 10. 最终评价指标

### 10.1 完整逐频点 residual

对 validation/test 的所有 HDF5 样本进行完整遍历：

- 每个 split 12 个被试；
- 每个 split `10,000,800` 个样本；
- 不使用随机 block 近似。

报告 residual MAE 和 RMSE。MCA baseline 等价于 $\hat r=0$。

### 10.2 全空间 ERB magnitude error

严格 ERB 指标调用 MATLAB `AKerbError`：

- 输入为重建后的 HRIR；
- 频率范围 50 Hz–Nyquist；
- 实际得到 41 个频带，约 `50–19792 Hz`；
- 方向维按 Fliege 求积权重平均；
- 频带维等权平均；
- 左右耳结果在被试内取平均。

### 10.3 对侧 25° ERB magnitude error

左耳对侧区域以方位角 270° 为中心，右耳以 90° 为中心，使用 25° great-circle 半径。该区域头部遮挡强、高频变化快，是低阶空间插值较困难的区域。

### 10.4 对侧高频幅度误差

定义为：

- 左耳：右侧开放半球；
- 右耳：左侧开放半球；
- 正中面排除；
- 频率：`f>10 kHz` 至 22.05 kHz；
- 误差：估计与 reference 的绝对 log-magnitude 差；
- 方向按 Fliege 权重，频率等权。

### 10.5 水平面 ILD MAE

对水平面 360 个方向计算完整 HRIR 能量：

$$
E_L=\sum_n|h_L[n]|^2,\qquad
E_R=\sum_n|h_R[n]|^2,
$$

$$
\mathrm{ILD}=10\log_{10}\frac{E_L}{E_R}.
$$

最终报告估计 ILD 与 reference ILD 的平均绝对误差。

## 11. 实验结果

### 11.1 完整逐频点结果

| Split | 方法 | MAE | RMSE | MAE 相对 MCA 改善 |
|---|---|---:|---:|---:|
| Validation | MCA | `2.5719 dB` | `4.1438 dB` | — |
| Validation | v1 | `2.1886 dB` | `3.5346 dB` | `14.90%` |
| Validation | v2 | `2.1398 dB` | `3.5070 dB` | `16.80%` |
| Test | MCA | `2.6007 dB` | `4.1861 dB` | — |
| Test | v1 | `2.2172 dB` | `3.5792 dB` | `14.75%` |
| Test | v2 | `2.1684 dB` | `3.5526 dB` | `16.62%` |

v2 在加入聚合听觉损失后，逐频点 MAE 和 RMSE 仍优于 v1。这说明 v2 没有通过牺牲基础 residual 准确度换取 ERB 或 ILD 改善。

### 11.2 严格论文相关指标

以下均为 12 个测试被试的均值 ± 被试间标准差：

| 指标 | MCA | v1 | v2 |
|---|---:|---:|---:|
| 全空间 ERB | `0.8027 ± 0.0255 dB` | `0.6838 ± 0.0326 dB` | `0.6115 ± 0.0356 dB` |
| 对侧 25° ERB | `1.8603 ± 0.1159 dB` | `1.5582 ± 0.0971 dB` | `1.3401 ± 0.0999 dB` |
| 对侧 `>10 kHz` | `4.2583 ± 0.2641 dB` | `3.7557 ± 0.2420 dB` | `3.7204 ± 0.2436 dB` |
| 水平面 ILD MAE | `0.8854 ± 0.2092 dB` | `0.7727 ± 0.3001 dB` | `0.6467 ± 0.2662 dB` |

相对改善：

| 指标 | v1 相对 MCA | v2 相对 MCA | v2 相对 v1 | v2 改善人数 |
|---|---:|---:|---:|---:|
| 全空间 ERB | `14.82%` | `23.82%` | `10.57%` | 12/12 |
| 对侧 25° ERB | `16.24%` | `27.96%` | `14.00%` | 12/12 |
| 对侧 `>10 kHz` | `11.80%` | `12.63%` | `0.94%` | 12/12 |
| 水平面 ILD MAE | `12.73%` | `26.96%` | `16.31%` | 12/12 |

### 11.3 结果解释

主要观察为：

1. **ERB 提升最大**：v2 相对 v1 又降低全空间 ERB `10.57%`、对侧 ERB `14.00%`，证明 ERB 代理损失与严格 `AKerbError` 方向一致。
2. **ILD 修复显著**：v2 相对 v1 将 ILD MAE 降低 `16.31%`，且从 v1 的 10/12 改善提高到 12/12。
3. **高频提升有限**：v2 相对 v1 的对侧高频只进一步改善 `0.94%`，说明 v1 已经取得较多高频收益，或当前高频权重和采样策略仍不足。
4. **模型容量不是主要增益来源**：v1 与 v2 参数量完全相同，变化来自训练组织、损失和继续微调。
5. **标准差不都下降**：v2 的全空间 ERB 和 ILD 被试间标准差高于 MCA，说明所有人虽都改善，但改善幅度并不均匀；pp81 的 ILD 是主要弱点。

相对 MCA，v2 的四项指标都是 12/12 被试改善；相对 v1，则并非每个被试的每项指标都更好：

| 指标 | v2 优于 v1 的被试数 | 例外 |
|---|---:|---|
| 全空间 ERB | 12/12 | 无 |
| 对侧 25° ERB | 12/12 | 无 |
| 对侧高频 | 11/12 | pp59：`3.7719 → 3.7743 dB` |
| ILD MAE | 11/12 | pp47：`0.6887 → 0.8872 dB` |

pp59 的高频变化仅约 `0.0024 dB`，接近无变化；pp47 的 ILD 相对 v1 明显退化，但仍优于其 MCA 基线 `1.2099 dB`。这说明复合 loss 的总体收益可能伴随个别被试的指标再分配，后续应通过消融和多种子实验确认稳定性。

## 12. 逐被试结果

下表列出 v2 的绝对指标及相对 MCA 改善：

| 被试 | 全空间 ERB | 改善 | 对侧 25° ERB | 改善 | 对侧高频 | 改善 | ILD MAE | 改善 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pp8 | `0.6151` | `23.40%` | `1.3556` | `23.39%` | `3.5440` | `12.88%` | `0.5789` | `32.62%` |
| pp18 | `0.6203` | `22.38%` | `1.4664` | `25.24%` | `4.1724` | `12.88%` | `0.4590` | `45.31%` |
| pp22 | `0.5867` | `27.64%` | `1.2873` | `29.72%` | `3.6490` | `14.69%` | `0.3971` | `39.36%` |
| pp26 | `0.5492` | `28.52%` | `1.2712` | `32.98%` | `3.4542` | `12.71%` | `0.4687` | `41.85%` |
| pp31 | `0.5985` | `20.66%` | `1.2517` | `22.99%` | `3.4441` | `13.87%` | `0.6340` | `12.26%` |
| pp33 | `0.6710` | `19.35%` | `1.2674` | `28.03%` | `3.9207` | `10.34%` | `0.6641` | `14.60%` |
| pp45 | `0.5572` | `30.33%` | `1.2283` | `32.26%` | `3.7922` | `15.46%` | `0.4917` | `39.32%` |
| pp47 | `0.6354` | `21.15%` | `1.4944` | `25.37%` | `3.9369` | `10.96%` | `0.8872` | `26.67%` |
| pp59 | `0.6247` | `26.75%` | `1.3011` | `29.88%` | `3.7743` | `13.72%` | `0.6254` | `25.44%` |
| pp70 | `0.5985` | `25.08%` | `1.2670` | `30.61%` | `3.3853` | `12.05%` | `0.5491` | `38.95%` |
| pp73 | `0.6341` | `20.29%` | `1.5073` | `22.35%` | `3.9425` | `9.32%` | `0.6158` | `24.01%` |
| pp81 | `0.6473` | `20.32%` | `1.3831` | `32.17%` | `3.6291` | `12.72%` | `1.3898` | `0.52%` |

单位均为 dB。

### 12.1 最佳与最弱案例

- pp45 的全空间 ERB 改善最大，为 `30.33%`。
- pp26 的对侧 25° ERB 改善最大，为 `32.98%`。
- pp45 的对侧高频改善最大，为 `15.46%`。
- pp18 的 ILD 改善最大，为 `45.31%`。
- pp73 的对侧高频改善最小，为 `9.32%`，但仍优于 MCA。
- pp81 的 ILD 只改善 `0.52%`，是 v2 最主要的剩余困难案例。

### 12.2 pp33 与 pp81 的修复

| 被试 | MCA ILD | v1 ILD | v1 相对 MCA | v2 ILD | v2 相对 MCA |
|---|---:|---:|---:|---:|---:|
| pp33 | `0.7777` | `0.8209` | `-5.57%` | `0.6641` | `14.60%` |
| pp81 | `1.3971` | `1.6622` | `-18.97%` | `1.3898` | `0.52%` |

pp33 已得到明确修复。pp81 虽不再退化，但改善很小，说明固定的全局 ILD proxy 对某些个体仍不足。后续可分析 pp81 的方位误差分布、左右耳残差相关性和特定频带能量偏差。

## 13. 可视化结果

v2 输出 15 张 PNG：

- 12 张逐被试四面板图；
- 1 张 12 人对侧 HRTF 总览；
- 1 张逐被试指标总览；
- 1 张 v1/v2 聚合指标对比。

主要入口：

- [12 人对侧 HRTF 总览](../residual_learning/reconstruction/mlp_n03_v2/figures/test12_contralateral_hrtf_overview.png)
- [12 人指标总览](../residual_learning/reconstruction/mlp_n03_v2/figures/test12_metric_overview.png)
- [v1/v2 聚合指标对比](../residual_learning/reconstruction/mlp_n03_v2/figures/v1_v2_aggregate_comparison.png)
- [pp33 重建对比](../residual_learning/reconstruction/mlp_n03_v2/figures/pp33_reconstruction_comparison.png)
- [pp81 重建对比](../residual_learning/reconstruction/mlp_n03_v2/figures/pp81_reconstruction_comparison.png)

逐被试四面板图包含：

1. 左耳对侧 270° HRTF；
2. 右耳对侧 90° HRTF；
3. 全空间 ERB error 频率曲线；
4. 水平面 ILD 曲线。

## 14. 工程实现与文件结构

### 14.1 主要源码

| 文件 | 作用 |
|---|---|
| `residual_learning/python/residual_model.py` | v1/v2 共用网络 |
| `residual_learning/python/residual_data.py` | `BinauralSpectrumSampler` 与特征构造 |
| `residual_learning/python/train_residual_mlp_v2.py` | v2 复合损失训练 |
| `residual_learning/python/evaluate_residual_mlp.py` | 完整 residual MAE/RMSE |
| `residual_learning/python/predict_reconstructed_residuals.py` | 测试方向 residual 推理 |
| `residual_learning/matlab/evaluate_test_reconstruction.m` | 回填、严格指标和绘图 |
| `residual_learning/matlab/plot_v1_v2_reconstruction_comparison.m` | v1/v2 聚合对比图 |

### 14.2 保留结果

- [训练历史](../residual_learning/runs/mlp_n03_v2/history.csv)
- [训练报告](../residual_learning/runs/mlp_n03_v2/training_report.json)
- [Validation 完整 residual 指标](../residual_learning/runs/mlp_n03_v2/val_metrics.json)
- [Test 完整 residual 指标](../residual_learning/runs/mlp_n03_v2/test_metrics.json)
- [严格指标汇总](../residual_learning/reconstruction/mlp_n03_v2/aggregate_metrics.csv)
- [逐被试严格指标](../residual_learning/reconstruction/mlp_n03_v2/per_subject_metrics.csv)
- [v1/v2 对比表](../residual_learning/reconstruction/mlp_n03_v2/v1_v2_comparison.csv)

checkpoint、含本机绝对路径的配置、预测 HDF5、MAT cache、日志和过拟合运行由 `.gitignore` 忽略。

## 15. 复现流程

### 15.1 v2 训练

在仓库根目录执行：

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/python/train_residual_mlp_v2.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/runs/mlp_n03_v1/best.pt `
  --run-name mlp_n03_v2 `
  --epochs 10 `
  --steps-per-epoch 500 `
  --validation-steps 96 `
  --directions-per-batch 32 `
  --learning-rate 3e-4 `
  --weight-decay 1e-5 `
  --erb-weight 0.50 `
  --high-frequency-weight 0.25 `
  --ild-weight 0.25 `
  --seed 20260724
```

### 15.2 完整 residual 评价

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/python/evaluate_residual_mlp.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/runs/mlp_n03_v2/best.pt `
  --split val

D:\miniconda3\envs\ml\python.exe `
  residual_learning/python/evaluate_residual_mlp.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/runs/mlp_n03_v2/best.pt `
  --split test
```

### 15.3 residual 推理与严格重建

v2 复用 v1 的 complex cache：

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/python/predict_reconstructed_residuals.py `
  residual_learning/reconstruction/mlp_n03_v2 `
  residual_learning/runs/mlp_n03_v2/best.pt `
  residual_learning/data/hutubs_residual_v1_n03/training_statistics.json `
  --input-root residual_learning/reconstruction/mlp_n03_v1

matlab -batch "addpath('residual_learning/matlab'); evaluate_test_reconstruction('mlp_n03_v2', 'mlp_n03_v1')"

matlab -batch "addpath('residual_learning/matlab'); plot_v1_v2_reconstruction_comparison('mlp_n03_v1', 'mlp_n03_v2')"
```

## 16. 有效性威胁与局限

### 16.1 测试集重复反馈

v2 的损失设计明确受到 v1 在同一测试集上 pp33、pp81 ILD 退化的启发。测试样本没有参与训练、归一化或 checkpoint 选择，因此不存在直接的数据或梯度泄漏；但测试结果影响了后续方法设计，这属于 adaptive test reuse。

因此：

- v2 当前结果适合作为内部开发和方法可行性证据；
- 不宜把同一 12 人测试集继续用于大量超参数搜索；
- 正式论文应冻结 v2 后，在新保留被试、重新划分或外部数据集上做最终评价。

### 16.2 单次随机种子

当前只有一次 v2 正式训练。深度学习训练和随机 block 采样存在方差，尚未报告：

- 多随机种子均值和标准差；
- 逐被试 bootstrap 置信区间；
- 配对显著性检验；
- 效应量。

### 16.3 固定数据域

当前只验证：

- HUTUBS simulated HRTF；
- Lebedev `N=3`；
- 固定 MCA 参数；
- 当前训练频率和网格。

尚未验证 HUTUBS measured、AXD、其他数据库或真实稀疏测量条件。

### 16.4 训练代理与最终指标不完全一致

ERB proxy 不是 `AKerbError`，ILD proxy 也不使用完整 HRIR。代理虽然带来正确方向的改善，但不能保证每次代理下降都对应严格指标下降。对侧高频严格指标只比 v1 改善 `0.94%`，提示代理设计仍有优化空间。

### 16.5 只修正幅度

v2 保持 MCA 相位，因此没有学习：

- 相位 residual；
- 群时延；
- ITD；
- HRIR 到达时间；
- 因果或最小相位以外的结构。

这保证了系统边界清晰，但也限制了可解决的问题范围。

### 16.6 缺失人体测量值

pp18、pp79、pp92 缺少公开 `x1/x2/x3`，分别位于 test、train、validation。当前使用其余 93 人平均头半径。该替代可能带来小幅 MCA 和 residual 偏差，应在获得完整元数据后做敏感性检查。

## 17. 推荐的后续实验

### 17.1 损失消融

保持模型、初始化、数据和训练预算一致，至少比较：

| 实验 | Loss |
|---|---|
| A0 | residual only |
| A1 | residual + ERB |
| A2 | residual + ERB + high-frequency |
| A3 | residual + ERB + ILD |
| A4 | residual + ERB + high-frequency + ILD |

消融可回答：

- ERB 增益主要来自 ERB 项还是双耳完整频谱采样；
- 高频项对 `0.94%` 的额外提升贡献多大；
- ILD 项是否是 pp33/pp81 修复的主要原因；
- 多项损失之间是否存在冲突。

### 17.2 损失权重搜索

当前 `0.50/0.25/0.25` 是首版经验权重。应在 validation 上进行有限、预先定义的搜索，例如：

- ERB：`0.25, 0.50, 1.00`；
- 高频：`0.10, 0.25, 0.50`；
- ILD：`0.10, 0.25, 0.50`。

测试集不得参与权重选择。

### 17.3 多随机种子

建议至少运行 3–5 个种子，报告：

- mean ± standard deviation；
- 每个指标 12 人配对差值；
- 95% bootstrap confidence interval；
- 改善被试比例。

### 17.4 不同稀疏阶数

优先扩展到 Lebedev：

```text
N = 1, 2, 4, 6
```

其中 N=1 可检验极低密度下 residual 网络是否仍有效，N=2/4 接近 MCA 改善较明显区域，N=6 可判断稀疏度提高后网络收益是否衰减。

### 17.5 外部泛化

在模型和超参数冻结后，优先考虑：

1. HUTUBS measured HRTF；
2. AXD；
3. 新的 subject-wise holdout；
4. 不同采样网格或采样率。

外部数据的坐标系、方向覆盖、HRIR 长度和采样率必须先统一。

### 17.6 pp81 专项诊断

建议对 pp81 绘制：

- 360° ILD error 差值；
- 左右耳 residual 相关性；
- 分 ERB band 的左右耳能量误差；
- 不同方位的 ILD proxy 与严格 ILD 差异；
- MCA correction filter 与其他被试的分布对比。

目标是判断 pp81 属于人体结构离群、MCA 基线异常、代理指标偏差，还是模型欠拟合。

## 18. 项目结论

MLP v2 在保持 `100,225` 个参数和相同 7 维输入的条件下，通过双耳完整频谱采样和听觉指标感知复合损失，显著增强了 v1 的 ERB 与 ILD 表现。

在当前 12 个测试被试上：

- 逐频点 residual MAE 相对 MCA 改善 `16.62%`；
- 全空间 ERB 改善 `23.82%`；
- 对侧 25° ERB 改善 `27.96%`；
- 对侧高频误差改善 `12.63%`；
- ILD MAE 改善 `26.96%`；
- 四项严格指标均为 12/12 被试优于 MCA。

相对 v1，v2 的主要贡献不是扩大网络，而是使训练数据布局和损失函数与最终听觉目标更一致。结果支持以下技术判断：

> MCA 负责物理先验、时间对齐和主体空间插值；轻量 residual MLP 负责学习 MCA 后的系统性幅度误差；双耳指标感知训练负责把逐频点修正进一步约束到 ERB、对侧高频和 ILD 目标。

当前 v2 已构成一个完整、可复现的内部实验版本。下一阶段最重要的工作不是继续无约束增加模型复杂度，而是完成损失消融、多随机种子、不同稀疏阶数和冻结后的外部测试，使现有性能增益成为更可靠的论文结论。
