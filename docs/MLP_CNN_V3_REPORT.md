# MCA Residual MLP-CNN v3 完整设计与实验报告

> 数据集：HUTUBS simulated HRTF，96 个被试
>
> 稀疏输入：Lebedev `N=3`，26 个方向
>
> 模型：冻结 Residual MLP v2 + 双耳一维频谱 CNN + 方向 FiLM
>
> 参数量：总计 `174,627`，其中 `74,402` 个参数参与 v3 训练
>
> 学习目标：MCA 与 dense reference 之间的 log-magnitude residual
>
> 实验日期：2026-07-25

## 摘要

本报告研究在 MCA（Magnitude-Corrected and Time-Aligned Interpolation）和
Residual MLP v2 的基础上，加入双耳一维频谱卷积网络，利用相邻频率之间的
结构继续降低 HRTF 插值后的剩余幅度误差。

v1 证明了 MCA 后的 log-magnitude residual 可以跨被试学习；v2 在不增加模型
容量的情况下，通过双耳完整频谱采样和 ERB、对侧高频、ILD 感知的复合损失，
显著改善了听觉相关指标。然而，v1 和 v2 的 MLP 对每个
“被试—耳朵—方向—频点”仍然独立预测。完整频谱只在损失层面联合，模型本身
并不能直接观察相邻频点、谱峰、谱谷、notch 宽度或左右耳频谱之间的对应关系。

v3 保留 v2 MLP 作为稳定的逐频点基线，并增加一个轻量双耳频谱 CNN。模型先由
冻结 MLP 得到基础预测，再由 CNN 在完整 463 点双耳频谱上预测增量：

$$
\hat r_{\mathrm{v3}}
=
\hat r_{\mathrm{v2}}
+
\Delta r_{\mathrm{CNN}}.
$$

CNN 输入由左右耳 MCA、MCA correction、v2 residual 以及共享 log-frequency
组成，共 7 个频谱通道。网络使用 48 个隐藏通道、4 个深度可分离膨胀卷积
residual block，dilation 为 `1,2,4,8`。方向单位向量
$(x,y,z)$ 通过方向编码器生成 FiLM 参数，对每个频谱 block 进行条件调制。
CNN 输出层和 FiLM 层采用零初始化，使模型在训练开始时严格满足
$\hat r_{\mathrm{v3}}=\hat r_{\mathrm{v2}}$。

正式训练冻结 v2 的 `100,225` 个参数，只优化 `74,402` 个 CNN 与 FiLM 参数。
训练使用 72 个被试，固定 12 个 validation 被试用于 checkpoint 选择，另外
12 个 test 被试保持锁定。每个 batch 包含一个被试、32 个共同方向、左右耳和
全部 463 个频点，共 29,632 个逐频点样本。训练 10 epoch，每个 epoch 500
steps，使用 AdamW、cosine learning-rate decay、FP16 AMP，并通过 W&B 与本地
CSV 同步记录曲线。依据预先规定的 validation 复合损失选择 epoch 9。

完整 test 集共包含 `10,000,800` 个逐频点样本。v3 将 raw residual MAE 从
MCA 的 `2.6007 dB` 和 v2 的 `2.1684 dB` 降至 `2.0910 dB`，相对 v2 改善
`3.57%`；12/12 个 test 被试均优于 v2。将预测 residual 回填到 MCA 幅度并
保留原 MCA 相位后，严格 MATLAB 指标如下：

| 指标，越低越好 | MCA | v1 MLP | v2 MLP | v3 MLP-CNN | v3 相对 v2 |
|---|---:|---:|---:|---:|---:|
| 全空间 ERB magnitude error | `0.8027` | `0.6838` | `0.6115` | **`0.5846 dB`** | 改善 `4.40%` |
| 对侧 25° ERB magnitude error | `1.8603` | `1.5582` | `1.3401` | **`1.3044 dB`** | 改善 `2.66%` |
| 对侧 `>10 kHz` 幅度误差 | `4.2583` | `3.7557` | `3.7204` | **`3.6201 dB`** | 改善 `2.69%` |
| 水平面 ILD MAE | `0.8854` | `0.7727` | **`0.6467`** | `0.6616 dB` | 退化 `2.30%` |

结果表明，频率上下文确实能在冻结强 MLP 基线的条件下进一步改善 raw residual
以及三项严格幅度指标。但严格 ILD 相对 v2 小幅退化，说明当前训练使用的
频域 ILD proxy 与最终由重建 HRIR 全时域能量计算的 ILD 仍不完全一致。
因此，v3 验证了 MLP-CNN 融合方向的有效性，同时也明确给出了 v3.1 的首要任务：
在不盲目扩大网络的前提下，改进严格 ILD 对齐的双耳能量约束。

## 1. 研究背景与版本演进

### 1.1 MCA 后残差学习

HRTF 空间上采样需要从少量测量方向恢复 dense 方向上的双耳响应。MCA 已经
结合时间对齐、球谐插值和幅度校正，是一个较强的物理与信号处理基线。本项目
不让神经网络从零生成 HRTF，而是只学习 MCA 与 reference 之间仍然存在的幅度
残差：

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

预测 residual 为零时，系统自然退化为原 MCA。这种设计保留 MCA 已有的时间
对齐、相位和空间插值结构，把学习任务限制在更容易优化的幅度修正范围内。

### 1.2 v1：验证 residual 可学习

v1 使用 7 维逐频点输入和 `100,225` 参数的 Residual MLP，通过 SmoothL1
学习 normalized residual。v1 在 test 上将 raw residual MAE 从
`2.6007 dB` 降至 `2.2172 dB`，并改善全空间 ERB、对侧 ERB、对侧高频与
平均 ILD。

v1 的主要问题是训练和模型都是单耳、逐频点的。左右耳之间没有联合约束，
pp33 和 pp81 的严格 ILD 出现退化。

### 1.3 v2：指标感知训练

v2 保持 MLP 架构和参数量不变，但把 batch 改为双耳完整频谱，并加入：

- ERB 频带能量误差；
- 对侧开放半球 `>10 kHz` 幅度误差；
- 左右耳宽带能量比形成的 ILD proxy。

v2 解决了 v1 的主要 ILD 失败，并把严格全空间 ERB 从 `0.6838 dB` 降到
`0.6115 dB`。但是，MLP 的前向过程仍对每个频点独立；完整频谱只在损失计算
时被联合使用。

### 1.4 v3：让模型直接观察频率上下文

v3 的核心问题是：

> 在保持 v2 强基线稳定、总参数量轻量的前提下，让网络直接观察双耳完整频谱，
> 是否可以继续降低 MCA residual？

因此 v3 没有替换 MLP，也没有直接训练一个大型端到端 CNN，而是采用
“冻结 MLP 基线 + CNN 增量修正”的保守融合方式。

## 2. 需求说明与验收标准

### 2.1 功能需求

v3 需要满足：

1. 无损加载 v2 最优 checkpoint；
2. 保持原 7 维逐频点特征和归一化口径；
3. 在同一方向联合读取左右耳完整 463 点频谱；
4. 使用 CNN 建模相邻频率结构；
5. 使用方向条件调节相同频谱模式在不同空间区域的解释；
6. 输出双耳 residual 增量，并与 v2 输出相加；
7. 训练开始时严格复现 v2；
8. 保持 train、validation、test 的被试级隔离；
9. 支持 CUDA、AMP、固定 validation 和 W&B 监控；
10. 支持完整 test 推理、MCA 幅度回填和 MATLAB 严格评价。

### 2.2 非功能需求

- 模型应保持轻量，新增参数控制在十万以内；
- 不复制或提交大型 HDF5、SOFA、checkpoint 和 W&B 缓存；
- 推理必须保留完整频率轴，不能因为显存分块破坏 CNN 上下文；
- validation 指标和 test 指标必须由穷举遍历计算，而不是随机 batch 估计；
- test 访问必须显式解锁；
- 所有关键命令、随机种子、checkpoint 规则和结果位置必须记录；
- v1/v2 默认 MATLAB 评价入口不能因 v3 扩展而失效。

### 2.3 验收标准

| 验收项 | 标准 | 实际结果 |
|---|---|---|
| v2 权重加载 | strict load 成功 | 通过 |
| 零初始化等价性 | 初始 v3 与 v2 逐点一致 | 通过 |
| 梯度路径 | CNN 有有限非零梯度，冻结 MLP 无梯度 | 通过 |
| 单被试学习能力 | pp91 多项 proxy 指标下降 | 通过 |
| 跨被试 validation | v3 优于 v2，且无退化被试 | 12/12 通过 |
| test raw residual | v3 优于 v2 | MAE 改善 `3.57%` |
| 严格幅度指标 | 至少主要幅度指标优于 v2 | 三项全部通过 |
| 严格 ILD | 不劣于 v2 | 未通过，退化 `2.30%` |
| 重建相位保持 | 误差低于 `1e-5 rad` | 最大 `6.22e-16 rad` |
| 幅度回填恒等性 | 误差低于 `1e-4 dB` | 最大 `7.11e-15 dB` |

v3 因此不是“所有目标全部完成”的终版，而是一个获得稳定幅度增益、同时暴露
严格 ILD 对齐问题的有效研究版本。

## 3. 数据集与实验协议

### 3.1 数据来源

实验使用 HUTUBS simulated HRTF，共 96 个被试。原始数据位于本机：

```text
D:\cuc\CSMT\MCAR\data\HRTF\hutubs
```

SOFA 与训练 HDF5 都是本地大型数据，不进入 Git。数据来源、获取方式和本机
路径记录在项目实验日志中。

### 3.2 稀疏输入与 dense 目标

- 稀疏输入：Lebedev `N=3`，26 个方向；
- MCA dense residual 数据：Fliege 900 个方向；
- 训练频率：463 点；
- 频率范围：`86.13–19982.81 Hz`；
- 频率间隔：约 `43.0664 Hz`；
- 采样率：`44.1 kHz`；
- 左右耳共同参与 v2/v3 batch。

严格重建阶段额外使用 360 个水平面方向计算 ILD。因此每个 test 被试预测：

```text
2 ears × (900 dense + 360 horizontal) directions × 463 frequencies
```

### 3.3 被试级固定划分

| Split | 数量 | 被试 |
|---|---:|---|
| Train | 72 | 除 validation/test 之外的被试 |
| Validation | 12 | pp13、pp35、pp39、pp41、pp51、pp53、pp55、pp56、pp58、pp74、pp84、pp92 |
| Test | 12 | pp8、pp18、pp22、pp26、pp31、pp33、pp45、pp47、pp59、pp70、pp73、pp81 |

划分单位是被试，而不是逐方向或逐频点。同一人的左右耳、全部方向和全部频率
只能属于一个 split，避免个体频谱泄漏。

### 3.4 HDF5 核心字段

每个 `n03.h5` 包含：

| 字段 | 典型形状 | 含义 |
|---|---|---|
| `mca_logmag_db` | `[2,900,463]` | MCA log-magnitude |
| `correction_logmag_db` | `[2,900,463]` | MCA correction-filter magnitude |
| `target_residual_db` | `[2,900,463]` | reference 与 MCA 的 dB residual |
| `direction_features` | `[900,6]` | 方向角、单位向量和权重 |
| `frequency_hz` | `[463]` | 训练频率 |

v3 复用 v1/v2 已验证的数据，不重新生成 target，也不重新计算训练集归一化统计。

### 3.5 数据归一化

逐频点 MLP 输入仍为：

$$
\mathbf{x}
=
[\tilde L_{\mathrm{MCA}},
\tilde L_{\mathrm{corr}},
x,y,z,
\widetilde{\log_{10}f},
e],
$$

其中耳别 $e$ 为左耳 `-1`、右耳 `+1`。MCA、correction、log-frequency 和
target 使用仅由 72 个 train 被试计算的统计量标准化。validation 和 test
不得参与统计量估计。

## 4. 模型选型

### 4.1 为什么选择一维频谱 CNN

HRTF 幅度在频率上具有强局部结构。耳廓 notch、头影衰减、谱峰和谱谷通常不是
独立频点事件，而是在一段连续频率范围内形成形状。逐频点 MLP 可以根据当前
频率和幅度估计平均 residual，但不能直接区分：

- 单点异常与连续 notch；
- 相同幅值位于谱峰上升沿还是下降沿；
- 左右耳相同频段的相对能量变化；
- correction-filter 在邻近频率上的趋势。

一维卷积适合沿频率轴提取局部模式，同时比二维方向—频率 CNN 更容易控制参数
量。第一版只卷积频率，不卷积方向，有利于隔离“加入频率上下文”这一变量。

### 4.2 为什么不直接替换 MLP

v2 已经是稳定且经过严格验证的强基线。直接用 CNN 从头预测 residual 会同时
改变：

- 基础函数逼近器；
- 优化起点；
- 参数量；
- 频率上下文；
- 双耳融合方式。

这样即使结果改善，也难以判断来源。v3 将 MLP 冻结，只训练 CNN delta，使
实验问题更清楚：

> 在完全保留 v2 的情况下，仅增加频率上下文能带来多少净增益？

### 4.3 为什么使用双耳联合输入

CNN 同时接收左右耳 MCA、correction 和 v2 prediction。这样右耳输出可以
参考左耳谱形，左耳输出也可以参考右耳谱形，为双耳能量一致性提供结构基础。

但“联合输入”本身不等于“严格 ILD 保证”。最终实验也证明，若损失与最终
HRIR ILD 不完全对齐，双耳 CNN 仍可能在幅度误差下降时让严格 ILD 略微回升。

### 4.4 为什么加入方向 FiLM

相同的频谱形状在正前方、同侧和对侧的物理含义不同。尤其对侧区域受头影影响
更强，高频修正模式具有明显方向依赖。

v3 不在方向维做卷积，而是用方向向量生成每个 block 的缩放与平移：

$$
\mathrm{FiLM}(\mathbf{h}\mid\Omega)
=
(1+\boldsymbol{\gamma}_{\Omega})\odot\mathbf{h}
+
\boldsymbol{\beta}_{\Omega}.
$$

它以较少参数为 CNN 提供连续方向条件。

## 5. MLP-CNN 融合结构

### 5.1 总体数据流

```text
point features [D,2,F,7]
          │
          ├── frozen v2 MLP ──> base residual [D,2,F]
          │
          └── assemble binaural spectrum channels
                    │
                    v
          CNN input [D,7,F]
                    │
       direction xyz ──> encoder ──> FiLM
                    │
                    v
          spectral CNN delta [D,2,F]
                    │
                    v
final residual = base residual + CNN delta
```

其中 $D$ 为方向数，$F=463$。

### 5.2 冻结 v2 MLP

MLP 结构保持：

```text
Linear(7,128) + SiLU
3 × [Linear(128,128) + SiLU + Linear(128,128) + residual + SiLU]
Linear(128,1)
```

参数量为 `100,225`。v3 训练时：

- `requires_grad=False`；
- 始终保持 evaluation mode；
- optimizer 不包含 MLP 参数；
- checkpoint 中仍保存 MLP 权重，便于单文件推理。

### 5.3 CNN 的 7 个输入通道

对每个方向，将左右耳的逐频点特征和 v2 预测重新组织为：

| 通道 | 内容 |
|---:|---|
| 1 | 左耳 normalized MCA magnitude |
| 2 | 左耳 normalized correction magnitude |
| 3 | 左耳 v2 normalized residual prediction |
| 4 | 右耳 normalized MCA magnitude |
| 5 | 右耳 normalized correction magnitude |
| 6 | 右耳 v2 normalized residual prediction |
| 7 | normalized log-frequency |

方向 `x/y/z` 不作为随频率重复的普通卷积通道，而是进入独立方向编码器。

### 5.4 CNN stem

```text
ReflectionPad1d(3)
Conv1d(7,48,kernel_size=7)
GroupNorm(8 groups)
SiLU
```

反射填充避免零填充在频谱两端制造突变。GroupNorm 不依赖 batch 统计，适合方向
batch 较小且频率长度固定的训练。

### 5.5 深度可分离膨胀 residual block

共有 4 个 block，dilation 依次为 `1,2,4,8`。每个 block：

```text
Conv1d(48,96,kernel_size=1)
GroupNorm + SiLU
ReflectionPad1d(3 × dilation)
Depthwise Conv1d(96,96,kernel_size=7,dilation=dilation,groups=96)
GroupNorm + SiLU
Conv1d(96,48,kernel_size=1)
GroupNorm
FiLM(direction)
Residual add + SiLU
```

深度卷积负责频率邻域建模，两个 `1×1` 卷积负责通道混合。相较普通
`96×96×7` 卷积，depthwise 形式显著减少参数。

### 5.6 方向编码器与 FiLM

方向编码器为：

```text
Linear(3,64) + SiLU
Linear(64,64) + SiLU
```

每个频谱 block 都有独立 `Linear(64,96)`，输出 48 个 scale 和 48 个 shift。

### 5.7 输出与残差融合

CNN 输出层：

```text
Conv1d(48,2,kernel_size=1)
```

两个输出通道分别为左右耳 normalized delta。最终：

$$
\hat r_{\mathrm{norm,v3}}
=
\hat r_{\mathrm{norm,v2}}
+
\Delta\hat r_{\mathrm{norm,CNN}}.
$$

反归一化后：

$$
\hat r_{\mathrm{dB}}
=
\hat r_{\mathrm{norm}}\sigma_r+\mu_r.
$$

需要注意，delta 位于 normalized target 空间。报告中的 CNN delta dB 通过乘以
target 标准差换算得到。

### 5.8 感受野

4 个 dilation block 的频率感受野增量为：

$$
6(1+2+4+8)=90.
$$

若只计算膨胀主干，其感受野为 91 个位置；加上 kernel-7 stem 后，相对原始
输入的完整理论感受野为 97 个频点。频率间隔约 `43.07 Hz`，对应约
`4.18 kHz` 的局部频率跨度。边界处使用 reflection padding。

### 5.9 参数量

| 模块 | 参数量 | v3 训练状态 |
|---|---:|---|
| v2 Residual MLP | `100,225` | 冻结 |
| CNN stem、频谱 blocks、方向编码器、FiLM、输出层 | `74,402` | 可训练 |
| 总计 | `174,627` | — |

新增参数相对 v2 为 `74.24%`，但总模型仍小于 0.18M 参数。

## 6. 初始化与可恢复性设计

### 6.1 从 v2 checkpoint 初始化

v3 从 `residual_learning/runs/mlp_n03_v2/best.pt` 加载 MLP。checkpoint epoch
为 9。加载使用 strict state-dict 校验，避免漏层或键名错误。

### 6.2 零增量初始化

CNN output convolution 的 weight 和 bias 全部初始化为零。FiLM 的 weight
和 bias 也初始化为零。因此训练前：

$$
\Delta r_{\mathrm{CNN}}=0
$$

且

$$
\hat r_{\mathrm{v3}}=\hat r_{\mathrm{v2}}.
$$

这种初始化具有三点价值：

1. 新网络的起点不是随机劣化模型；
2. 初始 validation 可作为完全相同采样口径下的 v2 基线；
3. 任意 v3 checkpoint 的收益可解释为 CNN 的净贡献。

### 6.3 中断恢复与 checkpoint

每个完整 epoch 结束后写入：

- `last.pt`：最新完整 epoch；
- `best.pt`：validation total 最优 epoch；
- `history.csv`：逐 epoch 指标；
- `configuration.json`：参数、文件列表和环境；
- 正常结束时写 `training_report.json`。

此前一次无 W&B 训练被用户主动停止在完整 epoch 6。其 checkpoint 有效，不需要
清理，但正式 W&B 实验从原 v2 重新开始，未拼接两段优化历史。

## 7. 双耳完整频谱采样

### 7.1 Batch 形状

正式训练每个 batch 随机选择：

- 1 个 train 被试；
- 32 个 Fliege 方向；
- 左右 2 耳；
- 全部 463 个频点。

原始张量形状：

```text
[ear=2, direction=32, frequency=463, feature=7]
```

模型输入转置为：

```text
[direction=32, ear=2, frequency=463, feature=7]
```

每个 batch 包含：

$$
2\times32\times463=29,632
$$

个逐频点监督样本。

### 7.2 为什么不能切分频率

CNN 的意义在于利用完整频谱上下文。如果像 v1 那样随机截取 128 个频点：

- 卷积看到的上下文不完整；
- 频谱边界随 batch 随机移动；
- 宽带能量和 ILD proxy 改变定义；
- train 与完整频谱推理不一致。

因此 v3 只在方向维控制显存，不切分 463 点频率轴。完整 validation、test 和
重建推理也遵守相同规则。

### 7.3 固定 validation

训练采样 seed 为 `20260725`，validation sampler seed 为 `20260726`。
每个 epoch 的 validation 都重新使用同一 seed 创建 sampler，因此 96 个
validation batch 完全相同。初始 v2 和所有 v3 epoch 可直接比较。

## 8. 损失函数

### 8.1 总损失

v3 沿用 v2 的复合损失，只改变模型：

$$
\mathcal L
=
\mathcal L_{\mathrm{res}}
+
0.50\frac{\mathcal L_{\mathrm{ERB}}}{\sigma_r}
+
0.25\frac{\mathcal L_{\mathrm{HF}}}{\sigma_r}
+
0.25\frac{\mathcal L_{\mathrm{ILD}}}{\sigma_r}.
$$

$\sigma_r$ 是 train target residual 的标准差，使各 dB 指标与 normalized
residual loss 大致处于可组合尺度。

### 8.2 Residual SmoothL1

$$
\mathcal L_{\mathrm{res}}
=
\operatorname{SmoothL1}
(\hat r_{\mathrm{norm}},r_{\mathrm{norm}};\beta=1).
$$

该项提供逐频点监督。训练日志另外记录反归一化后的 residual MAE，单位为 dB。

### 8.3 ERB 代理损失

463 个频点通过 41 个 ERB-rate 三角权重聚合为频带能量。对 corrected 与
reference 的 band-energy dB 求绝对差，并用 Fliege 方向权重加权。

它是可微频域代理，不等同于最终 MATLAB `AKerbError`。后者从完整重建 HRTF
得到 HRIR 后按严格听觉流程评价。

### 8.4 对侧高频损失

频率条件为：

$$
f>10\,\mathrm{kHz}.
$$

左耳对侧使用横向坐标 $y<0$，右耳对侧使用 $y>0$，排除正中面。误差为预测
residual 与 target residual 的绝对差，方向维使用归一化权重。

### 8.5 ILD 代理损失

由 log-magnitude 恢复各频点功率并在 463 个频点求和，得到每个方向的左右耳
宽带能量：

$$
E_e(\Omega)
=
10\log_{10}\sum_f10^{L_e(\Omega,f)/10}.
$$

代理 ILD：

$$
\mathrm{ILD}(\Omega)=E_L(\Omega)-E_R(\Omega).
$$

训练最小化 predicted 与 reference ILD 的方向加权 MAE。

### 8.6 Proxy 与严格指标不一致

训练 proxy 使用：

- 仅 463 个 `86.13–19982.81 Hz` 频点；
- 频域功率和；
- 随机选取的 32 个 dense 方向；
- 训练数据中的 log-magnitude。

最终严格 ILD 使用：

- 回填后的完整 HRTF；
- 原 MCA 相位；
- 双边频谱 IFFT 得到的 HRIR；
- 截取真实 HRIR 长度；
- 360 个水平面方向；
- 全 HRIR 时域能量。

二者相关但不完全等价。这一差异是解释 v3 严格 ILD 退化的关键。

## 9. 训练配置与计算环境

| 配置 | 数值 |
|---|---|
| 初始 checkpoint | v2 best epoch 9 |
| 训练阶段 | `cnn_only_frozen_mlp` |
| Epoch | 10 |
| Steps/epoch | 500 |
| Validation steps/epoch | 96，固定 |
| Directions/batch | 32 |
| Optimizer | AdamW |
| 初始学习率 | `3e-4` |
| Weight decay | `1e-5` |
| Scheduler | CosineAnnealingLR |
| Gradient clip | `5.0` |
| Seed | `20260725` |
| AMP | FP16 autocast + GradScaler |
| 初始 AMP scale | `1024` |
| GPU | NVIDIA GeForce RTX 5060 |
| PyTorch | `2.8.0+cu128` |
| CUDA | `12.8` |

首版曾使用 PyTorch 默认初始 loss scale 65536。零初始化输出层第一步的放大梯度
发生溢出，因此训练器把初始 scale 固定为 1024，并显式检测有限梯度和跳步数。
正式 5000 个优化 step 没有任何 AMP 跳步。

## 10. W&B 训练监控

正式运行：

- Project：`mcar-mlp-cnn-v3`
- Run name：`mlp_cnn_n03_v3_cnn_only`
- Run ID：`qtkrxras`
- URL：<https://wandb.ai/luyoung/mcar-mlp-cnn-v3/runs/qtkrxras>

W&B 逐 epoch 记录：

- train/validation total loss；
- residual、ERB proxy、对侧高频、ILD proxy；
- CNN delta 平均绝对值；
- 相对初始 v2 的改善百分比；
- learning rate、AMP scale、跳步数；
- epoch 时间与峰值 CUDA allocated memory；
- 当前 epoch 是否为 best。

模型 checkpoint 和 HDF5 没有上传。W&B 只作为曲线监控，本地 CSV/JSON 和
checkpoint 才是复现主记录。

## 11. 训练前验证

### 11.1 Smoke test

使用真实 pp91 数据和 v2 checkpoint 验证：

1. MLP strict load；
2. 输入、base、delta、final 张量形状；
3. 初始 v3 与 v2 逐点相等；
4. 复合损失可前向计算；
5. CNN 梯度有限且存在非零张量；
6. 冻结 MLP 不产生梯度。

全部通过后才进入训练。

### 11.2 pp91 短程过拟合

4 epoch × 120 steps，16 directions/batch。固定 validation 上：

| 指标 | 初始 v2 | CNN-only epoch 4 | 相对降低 |
|---|---:|---:|---:|
| 复合损失 | `0.53335` | `0.40730` | `23.63%` |
| Residual MAE | `1.9304` | `1.6244 dB` | `15.85%` |
| ERB proxy | `0.6441` | `0.5436 dB` | `15.60%` |
| 对侧高频 | `3.0559` | `2.3722 dB` | `22.37%` |
| ILD proxy | `0.5875` | `0.3170 dB` | `46.05%` |

该实验只证明梯度和模型容量足以学习一个被试，不用于泛化结论。

## 12. 正式训练结果

### 12.1 初始 v2 基线

在固定的 96 个 validation batch 上，训练前：

| 指标 | 初始 v2 |
|---|---:|
| Total | `0.594113` |
| Residual MAE | `2.139116 dB` |
| ERB proxy | `0.695902 dB` |
| 对侧高频 | `3.297190 dB` |
| ILD proxy | `0.646540 dB` |
| CNN delta | `0 dB` |

### 12.2 Validation 收敛历史

| Epoch | Total | Residual | ERB | 高频 | ILD | CNN delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.588567` | `2.1235` | `0.7001` | `3.2521` | `0.6610` | `0.2993` |
| 2 | `0.585525` | `2.1130` | `0.6949` | `3.2480` | `0.6550` | `0.3614` |
| 3 | `0.581558` | `2.0997` | `0.6857` | `3.2363` | `0.6520` | `0.4100` |
| 4 | `0.579968` | `2.0920` | `0.6751` | `3.2381` | `0.6455` | `0.4620` |
| 5 | `0.579476` | `2.0901` | `0.6744` | `3.2402` | `0.6519` | `0.5007` |
| 6 | `0.575839` | `2.0783` | `0.6658` | `3.2251` | `0.6424` | `0.5270` |
| 7 | `0.574931` | `2.0765` | `0.6660` | `3.2217` | `0.6472` | `0.5502` |
| 8 | `0.573803` | `2.0745` | `0.6649` | `3.2184` | **`0.6408`** | `0.5556` |
| **9** | **`0.573431`** | `2.0733` | `0.6646` | **`3.2169`** | `0.6412` | `0.5657` |
| 10 | `0.573586` | **`2.0732`** | **`0.6639`** | `3.2182` | `0.6417` | `0.5666` |

单位为 dB 的列只展示到四位小数。

### 12.3 Checkpoint 选择

预定规则是最小 validation total loss，因此选择 epoch 9。epoch 10 的 residual
和 ERB 更低，但高频与 ILD 略回升，使 total 由 `0.573431` 升到 `0.573586`。
没有因为观察 test 或单一指标而改选 epoch。

epoch 9 相对初始 v2：

| 指标 | 相对改善 |
|---|---:|
| Total | `3.48%` |
| Residual | `3.07%` |
| ERB proxy | `4.50%` |
| 对侧高频 | `2.43%` |
| ILD proxy | `0.82%` |

### 12.4 训练资源

- 总耗时：`364.19 s`，约 6 分 4 秒；
- 峰值 CUDA allocated memory：`220.46 MiB`；
- 优化 steps：5000；
- AMP 跳步：0；
- 最终 AMP scale：4096；
- `best.pt`：epoch 9；
- `last.pt`：epoch 10；
- 两个 checkpoint 均成功重新加载。

## 13. 完整 Validation 评价

训练中的 96 个 validation batch 是固定抽样，仍不足以代表全部样本。模型选定
后，穷举 12 个 validation 被试的：

```text
12 subjects × 2 ears × 900 directions × 463 frequencies
= 10,000,800 samples
```

结果：

| 方法 | Raw residual MAE | Raw residual RMSE | MAE 相对 MCA |
|---|---:|---:|---:|
| MCA zero residual | `2.57191` | `4.14377 dB` | 0% |
| v2 MLP | `2.13978` | `3.50698 dB` | `16.80%` |
| v3 MLP-CNN | **`2.07266`** | **`3.42891 dB`** | `19.41%` |

v3 相对 v2 的 MAE/RMSE 分别改善 `3.14%/2.23%`。12/12 validation 被试
全部改善，范围 `1.58%–4.44%`。

评估器内置的 v2 与历史 v2 validation 结果只相差
`+1.10e-8/-1.83e-8 dB`，证明归一化、反归一化、方向遍历和 AMP 口径一致。

## 14. Test 解锁协议

### 14.1 防止提前访问

评估器默认不能读取 test。必须同时提供：

```text
--split test --allow-test
```

只有以下条件满足后才解锁：

1. 正式训练结束；
2. 根据 validation 选择 epoch 9；
3. 完整 validation 穷举结果通过；
4. 模型架构、损失和超参数冻结；
5. test 结果不用于本轮重新训练。

### 14.2 Test raw residual

| 方法 | MAE | RMSE | MAE 相对 MCA |
|---|---:|---:|---:|
| MCA zero residual | `2.600671` | `4.186115 dB` | 0% |
| v2 MLP | `2.168373` | `3.552562 dB` | `16.62%` |
| v3 MLP-CNN | **`2.091041`** | **`3.452171 dB`** | `19.60%` |

v3 相对 v2：

- MAE 改善 `3.57%`；
- RMSE 改善约 `2.83%`；
- 12/12 被试 MAE 均改善；
- CNN delta 全样本平均绝对值 `0.5736 dB`。

重新计算的 v2 MAE/RMSE 与历史结果仅相差
`-7.68e-9/-9.18e-8 dB`。

### 14.3 Test 逐被试 raw residual

| 被试 | v2 MAE | v3 MAE | v3 相对 v2 |
|---|---:|---:|---:|
| pp8 | `2.2082` | `2.1253 dB` | `3.75%` |
| pp18 | `2.3166` | `2.2130 dB` | `4.48%` |
| pp22 | `2.1326` | `2.0610 dB` | `3.36%` |
| pp26 | `2.0715` | `2.0028 dB` | `3.31%` |
| pp31 | `2.0712` | `1.9831 dB` | `4.25%` |
| pp33 | `2.3268` | `2.2710 dB` | `2.40%` |
| pp45 | `2.1764` | `2.0797 dB` | `4.44%` |
| pp47 | `2.2054` | `2.1325 dB` | `3.31%` |
| pp59 | `2.1917` | `2.0868 dB` | `4.79%` |
| pp70 | `1.9964` | `1.9681 dB` | `1.42%` |
| pp73 | `2.2447` | `2.1373 dB` | `4.79%` |
| pp81 | `2.0790` | `2.0320 dB` | `2.26%` |

raw residual 的逐被试稳定性是 v3 最强的结果之一：没有任何 test 被试退化。

## 15. 完整频谱推理与 MCA 回填

### 15.1 推理输入

严格评价复用 v1 已缓存、v2 已验证的 reconstruction 输入：

```text
residual_learning/reconstruction/mlp_n03_v1
```

每个 test 被试包含 900 dense 方向和 360 水平面方向。v3 输出保存在独立目录，
不覆盖 v1/v2。

### 15.2 GPU 推理

推理按 32 个方向分块，每块保留：

```text
[32 directions, 2 ears, 463 frequencies, 7 point features]
```

12 被试总耗时 `6.01 s`，峰值 CUDA allocated memory `34.81 MiB`。

### 15.3 幅度回填

预测 dB residual 加到 MCA magnitude：

$$
\widehat L_{\mathrm{v3}}
=
L_{\mathrm{MCA}}+\hat r_{\mathrm{v3}}.
$$

复数 HRTF：

$$
\widehat H_{\mathrm{v3}}
=
10^{\widehat L_{\mathrm{v3}}/20}
\exp(j\angle H_{\mathrm{MCA}}).
$$

训练频率范围外的 bin 不修改。相位完全取自 MCA。

### 15.4 重建质量

| 检查 | 最大误差 |
|---|---:|
| 左耳相位 | `6.00e-16 rad` |
| 右耳相位 | `6.22e-16 rad` |
| 左耳幅度回填恒等性 | `7.11e-15 dB` |
| 右耳幅度回填恒等性 | `7.11e-15 dB` |

远低于代码断言的 `1e-5 rad` 与 `1e-4 dB`。

## 16. 最终严格评价指标

### 16.1 全空间 ERB magnitude error

使用 MATLAB `AKerbError`，在 900 个 dense 方向计算 reference 与 estimate 的
听觉频带幅度误差。方向维使用 Fliege 权重，左右耳和 ERB band 取平均。

### 16.2 对侧 25° ERB

左耳中心为方位 270°，右耳中心为方位 90°，使用大圆距离不超过 25° 的区域。
该指标聚焦头影强、空间混叠明显的困难区域。

### 16.3 对侧高频幅度误差

频率范围：

$$
10\,\mathrm{kHz}<f\leq22.05\,\mathrm{kHz}.
$$

方向为每耳对侧开放半球，排除正中面；方向维加权，频率维等权。

### 16.4 水平面 ILD MAE

从重建 HRTF 生成双边频谱，经 IFFT 得到 HRIR并截取原长度。对 360 个水平面
方向计算：

$$
\mathrm{ILD}
=
10\log_{10}
\frac{\sum_n h_L^2[n]}{\sum_n h_R^2[n]}.
$$

最终指标为 estimate 与 reference 的平均绝对差。

## 17. 严格 Test 结果

### 17.1 汇总

| 指标 | MCA | v1 | v2 | v3 | v3 对 MCA | v3 对 v2 |
|---|---:|---:|---:|---:|---:|---:|
| 全空间 ERB | `0.80274` | `0.68380` | `0.61150` | **`0.58459`** | `27.18%` | `+4.40%` |
| 对侧 25° ERB | `1.86029` | `1.55824` | `1.34007` | **`1.30441`** | `29.88%` | `+2.66%` |
| 对侧 >10 kHz | `4.25834` | `3.75571` | `3.72038` | **`3.62015`** | `14.99%` | `+2.69%` |
| ILD MAE | `0.88542` | `0.77274` | **`0.64672`** | `0.66157` | `25.28%` | `-2.30%` |

### 17.2 幅度结果解释

v3 相对 v2 的提升不大，但在三个互补的严格幅度指标上方向一致：

- 全空间 ERB：说明总体听觉频带幅度更接近 reference；
- 对侧 25° ERB：说明困难对侧区域继续改善；
- 对侧高频：说明增益并非只来自低频或前方方向。

三项严格幅度指标相对 v2 也都是 12/12 test 被试改善。逐被试相对改善范围
分别为：

- 全空间 ERB：`2.45%–7.09%`，逐被试非加权平均 `4.44%`；
- 对侧 25° ERB：`1.51%–4.71%`，逐被试非加权平均 `2.68%`；
- 对侧高频：`0.02%–4.23%`，逐被试非加权平均 `2.68%`。

结合 raw residual 的 12/12 改善，可以确认 CNN 的幅度收益不是由少数个体
主导的平均现象。需要注意，对侧高频最弱被试的 v3 相对 v2 增益只有 `0.02%`，
因此应表述为稳定但有限的提升，而不是数量级变化。

### 17.3 ILD 结果解释

v3 的严格 ILD 仍比 MCA 好 `25.28%`，且 12/12 被试都比 MCA 好，但相对 v2
退化 `2.30%`。这并不与 validation ILD proxy 改善 `0.82%` 矛盾，因为二者
评价链路不同。

逐被试比较中，v3 严格 ILD 只有 3/12 优于 v2，9/12 出现不同程度回升；
相对 v2 的逐被试变化范围为改善 `8.99%` 到退化 `9.77%`，逐被试非加权平均
为退化 `3.59%`。因此 ILD 问题不是单个异常被试造成，而是当前 v3 的系统性
弱点。

可能原因包括：

1. proxy 只覆盖 463 个选中频点；
2. 严格 ILD 使用完整重建 HRIR 时域能量；
3. CNN 优化局部谱形时可能改变左右耳能量的细小平衡；
4. ILD 权重 `0.25` 相对幅度项可能不足；
5. checkpoint 按复合 proxy 选择，而非严格 ILD 选择。

## 18. 严格指标逐被试结果

| 被试 | 全空间 ERB 改善 | 对侧 25° ERB 改善 | 对侧高频改善 | ILD 对 MCA 改善 |
|---|---:|---:|---:|---:|
| pp8 | `27.29%` | `25.32%` | `15.08%` | `27.13%` |
| pp18 | `26.72%` | `27.09%` | `15.53%` | `39.96%` |
| pp22 | `29.80%` | `31.31%` | `16.84%` | `34.64%` |
| pp26 | `32.76%` | `34.18%` | `14.92%` | `39.29%` |
| pp31 | `23.68%` | `24.61%` | `17.19%` | `20.15%` |
| pp33 | `22.68%` | `31.42%` | `11.82%` | `9.31%` |
| pp45 | `35.28%` | `34.20%` | `18.31%` | `34.14%` |
| pp47 | `23.18%` | `26.50%` | `12.49%` | `28.92%` |
| pp59 | `29.92%` | `32.61%` | `17.37%` | `21.70%` |
| pp70 | `28.15%` | `32.39%` | `12.07%` | `35.05%` |
| pp73 | `24.47%` | `24.32%` | `12.99%` | `22.41%` |
| pp81 | `22.27%` | `34.16%` | `14.99%` | `3.65%` |

上表的参照均为 MCA。所有 48 个“被试 × 指标”组合都为正改善。

pp45 的全空间 ERB 改善最大，为 `35.28%`。pp81 的 ILD 仍是最弱案例，只比
MCA 改善 `3.65%`，说明该个体的双耳能量关系仍应专项分析。

## 19. 可视化结果

本地生成：

- 12 张 `pp*_reconstruction_comparison.png`；
- `test12_contralateral_hrtf_overview.png`；
- `test12_metric_overview.png`；
- `v1_v2_v3_aggregate_comparison.png`。

位置：

```text
residual_learning/mlp_cnn_v3/reconstruction/
  mlp_cnn_n03_v3_cnn_only/figures/
```

12 被试总览展示左耳方位 270° 对侧 HRTF。v3 曲线在多数频段从 MCA 向
reference 移动，尤其在高频整体包络上更稳定；极深、个体化 notch 的位置和
深度仍不能完全恢复，这是仅使用通用跨被试幅度 residual 的合理边界。

生成图目前属于本地可复现实验产物，由 `.gitignore` 忽略。若用于论文，应挑选
必要图复制到项目 `figures/`，并保留生成命令和数据来源。

## 20. 工程实现

### 20.1 v3 独立工作流

```text
residual_learning/mlp_cnn_v3/
├── README.md
├── requirements.txt
├── python/
│   ├── mlp_cnn_model.py
│   ├── smoke_test_mlp_cnn.py
│   ├── train_mlp_cnn_v3.py
│   ├── evaluate_mlp_cnn_v3.py
│   └── predict_mlp_cnn_reconstruction.py
├── matlab/
│   ├── evaluate_mlp_cnn_v3_reconstruction.m
│   └── plot_v1_v2_v3_reconstruction_comparison.m
├── runs/
│   ├── .gitignore
│   └── README.md
└── reconstruction/
    ├── .gitignore
    └── README.md
```

### 20.2 公共代码扩展

`residual_learning/matlab/evaluate_test_reconstruction.m` 增加可选：

- output root；
- cache root；
- corrected method key；
- corrected display name。

默认参数仍保持 v1/v2 行为，v3 通过独立 wrapper 指定自己的输出目录和名称。

### 20.3 Git 管理

应提交：

- Python/MATLAB 源码；
- v3 README 与本报告；
- requirements；
- 嵌套 `.gitignore` 与目录说明；
- `docs/EXPERIMENT_LOG.md`。

不应提交：

- HUTUBS SOFA/HDF5；
- `best.pt`、`last.pt`；
- W&B 本地缓存；
- 预测 residual HDF5；
- 大型 MAT；
- 批量生成 PNG，除非明确选为论文图。

## 21. 完整复现流程

以下命令均从项目根目录执行。

### 21.1 环境检查

```powershell
D:\miniconda3\envs\ml\python.exe -c `
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

安装 v3 依赖：

```powershell
D:\miniconda3\envs\ml\python.exe -m pip install `
  -r residual_learning/mlp_cnn_v3/requirements.txt
```

### 21.2 Smoke test

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/smoke_test_mlp_cnn.py `
  --dataset-root residual_learning/data/hutubs_residual_v1_n03 `
  --checkpoint residual_learning/runs/mlp_n03_v2/best.pt `
  --subject 91 --directions 4
```

### 21.3 正式训练

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/train_mlp_cnn_v3.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/runs/mlp_n03_v2/best.pt `
  --run-name mlp_cnn_n03_v3_cnn_only_wandb_online `
  --epochs 10 --steps-per-epoch 500 --validation-steps 96 `
  --directions-per-batch 32 `
  --wandb-mode online `
  --wandb-project mcar-mlp-cnn-v3 `
  --wandb-run-name mlp_cnn_n03_v3_cnn_only
```

### 21.4 完整 Validation

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/evaluate_mlp_cnn_v3.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt `
  --split val --directions-per-block 32
```

### 21.5 锁定 Test

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/evaluate_mlp_cnn_v3.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt `
  --split test --allow-test --directions-per-block 32
```

### 21.6 完整频谱重建预测

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/predict_mlp_cnn_reconstruction.py `
  residual_learning/mlp_cnn_v3/reconstruction/mlp_cnn_n03_v3_cnn_only `
  residual_learning/reconstruction/mlp_n03_v1 `
  residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt `
  residual_learning/data/hutubs_residual_v1_n03/training_statistics.json `
  --directions-per-block 32
```

### 21.7 MATLAB 严格评价与绘图

```powershell
matlab.exe -batch `
  "addpath('D:/cuc/CSMT/MCAR/residual_learning/mlp_cnn_v3/matlab'); evaluate_mlp_cnn_v3_reconstruction; plot_v1_v2_v3_reconstruction_comparison"
```

## 22. 有效性威胁与局限

### 22.1 v3 仍属于内部开发链

v3 的架构动机来自 v1/v2 在同一项目 test 被试上的结果。虽然本轮 epoch 和
超参数只由 validation 选择，历史 test 已对研究方向产生过反馈。因此论文中的
最终无偏结论仍应增加外部数据集或新的最终保留集。

### 22.2 单次随机种子

当前正式 v3 只有 seed `20260725`。改进幅度为 2%–4%，需要多随机种子报告
均值、标准差和置信区间，确认收益大于训练方差。

### 22.3 只验证 Lebedev N=3

当前结论只适用于 26 个稀疏方向。更稀疏时 residual 更大，CNN 可能更有价值；
更密集时 MCA 已更准确，CNN 增益可能缩小。

### 22.4 只在 HUTUBS simulated 域内验证

simulated HRTF 具有统一的建模流程。真实测量数据包含噪声、设备差异、方向误差
和被试移动。应在 HUTUBS measured 或 AXD 上进行外部泛化。

### 22.5 不使用人体测量特征

模型没有身高、头宽、耳廓尺寸等个体特征，只能从 MCA 与 correction 频谱推断
个体差异。这限制了极深个体化 notch 的恢复。

### 22.6 方向上下文尚未显式建模

CNN 只沿频率卷积；FiLM 提供单方向条件，但相邻空间方向之间没有卷积或图传播。
因此 v3 不能显式利用空间连续性。

### 22.7 只修正幅度

v3 完全保留 MCA 相位，不能改善残余 ITD、群时延或相位误差。当前研究目标明确
限制在幅度 residual，但不能把结果扩展为完整复数 HRTF 全面改善。

### 22.8 Proxy 与严格 ILD 不一致

这是当前最重要的局限。validation ILD proxy 改善不保证最终 HRIR ILD 改善。
如果继续直接增大 CNN，可能进一步降低幅度误差却放大双耳能量偏移。

### 22.9 缺少结构消融

尚未分别验证：

- 无 FiLM；
- 单耳 CNN；
- 普通卷积对比 depthwise；
- dilation schedule；
- 不输入 v2 prediction；
- 只训练 CNN 输出头；
- MLP 与 CNN 联合微调。

因此目前只能证明完整 v3 方案有效，不能量化每个组件的独立贡献。

## 23. v3.1 推荐实验

### 23.1 首选：严格 ILD 对齐

保持架构、数据划分和 test 锁定不变，只修改 train/validation 损失。优先尝试：

1. 把当前 463 点频域 ILD proxy 扩展为与重建频谱一致的能量定义；
2. 加入未修正频段的 MCA 能量；
3. 使用可微双边频谱与 IFFT，按 HRIR 时域能量计算 ILD；
4. 或直接约束 CNN 左右耳 delta 的能量差，防止无必要漂移。

这一实验最容易解释，因为它只改变损失与严格指标的对齐方式。

### 23.2 ILD 权重小范围搜索

在 validation 上尝试：

```text
0.25, 0.50, 1.00
```

同时报告 residual、ERB、高频与 ILD 的 Pareto 关系。不能只按 ILD 选模型，也
不能访问 test 选择权重。

### 23.3 频谱平滑正则

约束 CNN delta 的一阶或二阶频率差分，避免相邻频点产生不必要的高频抖动：

$$
\mathcal L_{\mathrm{smooth}}
=
\|\Delta_f\Delta r_{\mathrm{CNN}}\|_1.
$$

需要注意，正则过强可能抹平真实 notch。

### 23.4 FiLM 消融

比较：

- 完整方向 FiLM；
- 无 FiLM；
- 只在 stem 后使用一次 FiLM；
- 将 `x/y/z` 作为普通常量频谱通道。

该消融可判断方向条件是否真正贡献，还是 CNN 主要依赖频谱输入。

### 23.5 多随机种子

至少运行 3 个 seed。报告：

- validation best epoch；
- test raw residual；
- 四项严格指标；
- 每个被试改善人数；
- 训练耗时和方差。

### 23.6 外部泛化

模型冻结后在 HUTUBS measured 或 AXD 上：

1. 重新生成 MCA 输入；
2. 仅用 train 统计归一化；
3. 不微调直接推理；
4. 报告域外指标；
5. 再讨论少量域适配。

## 24. 对论文写作的建议

v3 适合被描述为：

> 基于冻结逐频点残差基线的方向条件双耳频谱细化网络。

论文中应明确区分：

- MCA：传统强基线；
- v1：逐频点 residual 可学习性；
- v2：听觉指标感知训练；
- v3：模型层面的频率上下文与双耳联合细化。

结果不能只写“v3 四项均优于 MCA”，还必须报告：

> v3 相对 v2 改善三项严格幅度指标，但 ILD MAE 相对 v2 退化 2.30%。

这一负结果不是实验失败，而是揭示训练 proxy 与最终指标不一致，为下一版方法
提供明确、可验证的研究问题。

## 25. 项目结论

MLP-CNN v3 在冻结 v2 MLP 的前提下，仅通过 `74,402` 个新增可训练参数引入
双耳频率上下文和方向 FiLM。正式实验完成了 smoke test、单被试过拟合、
72/12 跨被试训练与 validation、W&B 监控、完整 validation 穷举、锁定 test、
GPU 完整频谱预测、MCA 幅度回填和 MATLAB 严格声学评价。

核心结论为：

1. v3 在 validation 和 test 的全部 12 个被试上稳定降低 raw residual MAE；
2. test raw residual MAE 相对 v2 再降低 `3.57%`；
3. 严格全空间 ERB、对侧 25° ERB 和对侧高频分别相对 v2改善
   `4.40%`、`2.66%` 和 `2.69%`；
4. v3 相对 MCA 的四项严格指标均在 12/12 被试上改善；
5. 严格 ILD 相对 v2 退化 `2.30%`，暴露 proxy 对齐问题；
6. 幅度回填严格保持 MCA 相位，工程链路和数值断言全部通过；
7. 下一步应优先改进 ILD 损失，而不是直接扩大模型规模。

因此，v3 已经证明“逐频点 MLP 基线 + 双耳频谱 CNN 增量细化”是有效且具有
解释性的方向。它不是最终模型，但已经形成一条完整、可复现、能够继续做严格
消融和外部验证的技术路线。
