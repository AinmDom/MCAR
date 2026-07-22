# MCA + 轻量残差网络 HRTF 插值项目说明

本文件用于记录当前 CSMT 论文项目中，基于 **MCA（Magnitude-Corrected and Time-Aligned Interpolation）** 方法继续做轻量残差网络优化时，SUpDEq 工具包中哪些文件最有用，以及推荐的复现实验流程。

## 1. 项目目标

当前选题方向：

> 基于幅度校正与时间对齐插值的轻量残差学习 HRTF 空间上采样方法

核心思路：

1. 使用 SUpDEq 工具包复现 MCA 插值结果；
2. 将 MCA 作为强传统 baseline；
3. 分析 MCA 后仍然存在的残余误差；
4. 使用轻量神经网络学习残差；
5. 验证是否能在 MCA 基础上进一步降低 HRTF 插值误差。

建议学习目标不是从零预测 HRTF，而是学习：

```text
Residual = Reference HRTF - MCA HRTF
```

更具体地，建议先从 log-magnitude residual 开始：

```text
Residual = log|H_ref| - log|H_MCA|
```

这样可以避免一开始处理完整复数 HRTF 相位问题，降低实现难度。

## 2. SUpDEq 工具包位置

当前工具包路径：

```text
..\CSMT\MCAR\SUpDEq-master
```

该工具包是 MCA 论文作者公开的 MATLAB 工具包。README 中说明，`supdeq_interpHRTF` 已经集成了 MCA 后处理步骤。

## 3. 最重要的文件

### 3.1 初始化文件

```text
SUpDEq-master\supdeq_start.m
```

作用：

- 添加 SUpDEq 主目录；
- 添加 evaluation、materials、thirdParty 等依赖路径；
- 启动 SOFA、AKtools、SOFiA 等第三方工具。

建议每次 MATLAB 开始实验时先运行：

```matlab
cd('D:\course\CUC_2\CSMT\2026\SUpDEq-master')
supdeq_start
```

### 3.2 MCA 示例脚本

```text
SUpDEq-master\supdeq_demo_MCA.m
```

这是复现 MCA 的第一入口。

该脚本比较了三种结果：

```matlab
interpHRTF_sh  = supdeq_interpHRTF(sparseHRTF, sgD, 'None',   'SH', nan, headRadius);
interpHRTF_con = supdeq_interpHRTF(sparseHRTF, sgD, 'SUpDEq', 'SH', nan, headRadius);
interpHRTF_mca = supdeq_interpHRTF(sparseHRTF, sgD, 'SUpDEq', 'SH', inf, headRadius);
```

对应含义：

| 变量 | 含义 |
|---|---|
| `interpHRTF_sh` | 仅 SH 插值，无时间对齐，无 MCA |
| `interpHRTF_con` | SUpDEq 时间对齐 + SH 插值，无 MCA |
| `interpHRTF_mca` | SUpDEq 时间对齐 + SH 插值 + MCA |

这个脚本可以作为后续实验代码的模板。

### 3.3 核心函数

```text
SUpDEq-master\supdeq_interpHRTF.m
```

这是最关键的函数。它完成：

- HRTF 预处理 / 时间对齐；
- 空间插值；
- MCA 幅度校正；
- 输出 MCA 前后的 HRTF/HRIR；
- 保存部分中间变量。

函数接口：

```matlab
[interpHRTFset, HRTF_L_ip, HRTF_R_ip] = supdeq_interpHRTF( ...
    HRTFset, ipSamplingGrid, ppMethod, ipMethod, mc, ...
    headRadius, tikhEps, limitMC, mcKnee, mcMinPhase, limFade)
```

常用参数：

| 参数 | 说明 |
|---|---|
| `HRTFset` | 稀疏 HRTF 数据集 |
| `ipSamplingGrid` | 目标插值网格 |
| `ppMethod` | 预处理 / 时间对齐方法 |
| `ipMethod` | 插值方法 |
| `mc` | MCA 最大 boost 设置 |
| `headRadius` | 头半径参数 |

常用 `ppMethod`：

| 值 | 含义 |
|---|---|
| `'None'` | 不做时间对齐 |
| `'SUpDEq'` | 使用 SUpDEq 时间对齐 / directional equalization |
| `'OBTA'` | onset-based time alignment |
| `'MagPhase'` | 幅度相位分离式处理 |

常用 `ipMethod`：

| 值 | 含义 |
|---|---|
| `'SH'` | 球谐插值 |
| `'NN'` | Natural Neighbor 插值 |
| `'Bary'` | Barycentric 插值 |
| `'SARITA'` | SARITA 时域插值 |

常用 `mc`：

| 值 | 含义 |
|---|---|
| `nan` | 不做 MCA |
| `inf` | 做 MCA，不限制 boost，论文 demo 中使用 |
| 有限数值 | 做 MCA，并限制最大 boost |

## 4. 对残差网络最有价值的中间变量

`supdeq_interpHRTF.m` 会在输出结构体中保存 MCA 相关中间结果。

重点变量：

```matlab
interpHRTF_mca.p.HRTF_L_ip_noMC
interpHRTF_mca.p.HRTF_R_ip_noMC
interpHRTF_mca.p.corrFilt_lim
interpHRTF_mca.HRTF_L
interpHRTF_mca.HRTF_R
interpHRTF_mca.HRIR_L
interpHRTF_mca.HRIR_R
```

含义：

| 变量 | 含义 | 用途 |
|---|---|---|
| `p.HRTF_L_ip_noMC` | MCA 前左耳插值 HRTF | 可作为网络输入 |
| `p.HRTF_R_ip_noMC` | MCA 前右耳插值 HRTF | 可作为网络输入 |
| `p.corrFilt_lim` | MCA correction filter | 可作为网络输入特征 |
| `HRTF_L` | MCA 后左耳 HRTF | baseline 结果 |
| `HRTF_R` | MCA 后右耳 HRTF | baseline 结果 |
| `HRIR_L` | MCA 后左耳 HRIR | 时域分析 |
| `HRIR_R` | MCA 后右耳 HRIR | 时域分析 |

建议残差网络第一版使用：

```text
输入：
1. MCA 后 log-magnitude HRTF
2. MCA correction filter
3. 方位角 / 仰角
4. 频率或 ERB band index
5. 左右耳标记

输出：
Reference 与 MCA 之间的 log-magnitude residual
```

## 5. 评价指标相关文件

路径：

```text
SUpDEq-master\evaluation
```

重点文件：

| 文件 | 用途 |
|---|---|
| `supdeq_calcLSD_HRIR.m` | 计算 HRIR 对应频谱的 LSD |
| `supdeq_calcSpectralDifference.m` | 计算频谱差异 |
| `supdeq_calcSpectralDifference_HRIR.m` | 从 HRIR 计算频谱差异 |
| `supdeq_calcILD.m` | 计算 ILD |
| `supdeq_calcITD.m` | 计算 ITD |
| `supdeq_plotIR.m` | 绘制 HRIR |
| `supdeq_plotTF.m` | 绘制频率响应 |
| `supdeq_plotGrid.m` | 绘制空间采样网格 |

论文实验建议至少使用：

- LSD / SD；
- auditory-band magnitude error；
- ILD error；
- ITD error；
- contralateral region 的单独误差统计。

## 6. 采样网格与稀疏数据生成

重点文件：

| 文件 | 用途 |
|---|---|
| `supdeq_lebedev.m` | 生成 Lebedev 球面采样网格 |
| `supdeq_fliege.m` | 生成 Fliege 网格 |
| `supdeq_gauss.m` | 生成 Gauss 网格 |
| `supdeq_equiangular.m` | 生成等角网格 |
| `supdeq_getSparseDataset.m` | 从 dense 数据中抽取 sparse 数据 |

`materials` 中也有生成稀疏集合的脚本：

```text
materials\get_sparse_HRTF_set_Lebedev.m
materials\get_sparse_HRTF_set_Fliege.m
materials\get_sparse_HRTF_set_Gauss.m
materials\get_sparse_HRTF_set_Equiangular.m
```

建议第一阶段只使用 Lebedev 网格，因为 MCA 论文主实验就是基于 Lebedev sparse grid。

## 7. 数据文件

### 7.1 自带 reference 数据

```text
materials\HRIRs_ref
```

包含 dense reference HRIR/HRTF，例如：

- `HRIR_L2702.sofa`
- `HRIRs_FABIAN_sfd_N35.mat`
- `HRIRs_HMSII_sfd_N35.mat`

### 7.2 自带 sparse 数据

```text
materials\HRIRs_sparse
```

包含 sparse 示例数据，例如：

- `sparseHRTFdataset_L14.mat`
- `sparseHRTFdataset_L38.mat`
- `sparseHRTFdataset_L86.mat`

这些数据适合先跑通工具链。正式论文实验建议后续使用 HUTUBS simulated HRTF，尽量贴近 MCA 原论文设置。

## 8. 当前阶段不优先看的文件

以下内容暂时不是 MCA + 轻量残差网络主线：

| 文件 / 模块 | 原因 |
|---|---|
| `supdeq_dvf.m` | 近场 HRTF 合成，当前不做距离变化 |
| `supdeq_rangeExt.m` | 距离外推，暂不需要 |
| `supdeq_shiftDistance.m` | 距离移动，暂不需要 |
| `supdeq_demo_ALFE.m` | 低频扩展相关，当前不是主线 |
| `supdeq_alfe.m` | 同上 |
| thirdParty 中的大量 demo | 依赖库，不建议先读源码 |

## 9. 推荐实验路线

### Step 1：跑通 MCA demo

```matlab
cd('D:\course\CUC_2\CSMT\2026\SUpDEq-master')
supdeq_start
supdeq_demo_MCA
```

目标：

- 确认工具包能正常运行；
- 得到 `interpHRTF_sh`、`interpHRTF_con`、`interpHRTF_mca`、`refHRTF`；
- 复现 MCA 优于 conventional time-aligned interpolation 的趋势。

### Step 2：导出训练数据

建议导出：

```matlab
save('mca_features.mat', ...
    'interpHRTF_mca', 'interpHRTF_con', 'refHRTF', '-v7.3')
```

后续在 Python / PyTorch 中读取：

```text
MCA result:
interpHRTF_mca.HRTF_L / HRTF_R

Reference:
refHRTF.HRTF_L / HRTF_R

MCA correction:
interpHRTF_mca.p.corrFilt_lim
```

### Step 3：构造残差学习任务

推荐先使用 log-magnitude：

```text
X = features(direction, frequency, ear, H_MCA, correction_filter)
y = log|H_ref| - log|H_MCA|
```

### Step 4：训练轻量模型

第一版模型建议：

- MLP；
- 输入为方向、频率、左右耳、MCA 幅度、correction filter；
- 输出为 residual；
- loss 使用 MSE 或 L1；
- 指标使用 LSD / auditory-band magnitude error。

### Step 5：与 MCA 比较

比较方法：

1. SH only；
2. SUpDEq + SH；
3. MCA；
4. MCA + residual MLP。

重点分析：

- 高频误差是否下降；
- contralateral region 是否改善；
- ILD error 是否改善；
- 模型参数量和推理时间是否足够小。

## 10. 推荐论文贡献表述

可以初步写成：

1. 本文在 MCA 插值框架基础上提出一种轻量残差学习方法，不从零预测 HRTF，而是只学习 MCA 后的剩余误差；
2. 本文分析 MCA 插值残差在频率、空间区域和双耳侧别上的分布特征；
3. 实验表明，轻量残差网络可以在保持较低复杂度的同时，进一步降低 MCA 在高频和对侧区域的插值误差。

## 11. 当前最小可行版本

如果时间紧，CSMT 最小可行版本如下：

```text
Dataset:
HUTUBS simulated HRTFs 或工具包自带 HRTF 先行验证

Sparse grid:
Lebedev N = 1 到 10

Baseline:
SH only
SUpDEq + SH
MCA

Proposed:
MCA + MLP residual

Metrics:
LSD
Auditory-band magnitude error
ILD error
Contralateral high-frequency error
```

该版本已经可以形成一篇完整的 CSMT 初稿。

- 0722：添加了git

