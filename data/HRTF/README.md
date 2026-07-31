# 头相关传输函数（Head-Related Transfer Function，HRTF）数据集

本目录保存幅度校正与时间对齐插值（Magnitude-Corrected and Time-Aligned
Interpolation，MCA）基线、残差学习和跨个体评估使用的本地 HRTF 数据。数据文件体积较大，已通过仓库根目录的 `.gitignore` 排除；Git 仅跟踪本说明文件，不跟踪 `axd/` 和 `hutubs/` 中的实际数据。

## 目录与来源

数据文件采用空间声学数据格式（Spatially Oriented Format for Acoustics，
SOFA）。表中的兆二进制字节（mebibyte，MiB）按 $2^{20}$ 字节计算。

| 本地目录 | 数据来源 | 本地内容 | 大小 |
|---|---|---|---:|
| `axd/` | [SOFA Acoustics AXD](https://sofacoustics.org/data/database/axd/) | 140 个 SOFA 文件：`p0001`–`p0040`、`p0101`–`p0200` | 394.60 MiB |
| `hutubs/` | [SOFA Acoustics HUTUBS](https://sofacoustics.org/data/database/hutubs/) | 192 个 SOFA、58 个多边形文件格式（Polygon File Format，PLY）、2 个便携式文档格式（Portable Document Format，PDF）、1 个逗号分隔值（Comma-Separated Values，CSV）文件，共 253 个文件 | 1,375.71 MiB |
| `sonicom_measured_ffcmp_minphase_44k1/` | [SONICOM 官方完整数据集](https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/) | 由下载器根据官方元数据冻结的 350 名干净测量被试；每人一个保留 ITD 的 44.1 kHz 自由场补偿 SOFA | 872.82 MiB |

AXD 和 HUTUBS 于 `2026-07-23` 从 SOFA Acoustics 的公开目录下载；
SONICOM 正式队列于 `2026-07-30` 从其官网完整数据集下载。详细下载参数、
文件数量和完整性检查结果见
[`docs/EXPERIMENT_LOG.md`](../../docs/EXPERIMENT_LOG.md)。

## 本地路径

```text
D:\cuc\CSMT\MCAR\data\HRTF\axd
D:\cuc\CSMT\MCAR\data\HRTF\hutubs
```

HUTUBS 后续实验优先使用头相关脉冲响应（Head-Related Impulse Response，
HRIR）文件 `pp1_HRIRs_simulated.sofa`–`pp96_HRIRs_simulated.sofa`；同目录的
measured SOFA、头部网格和人体测量文件保留用于对照与扩展实验。

## 下载 SONICOM 正式训练队列

下载器位于 `src/mcar/data_tools/download_sonicom.py`，只使用 Python 标准库。
它先读取官网 `metadata.csv` 和日期固定的异常清单，再选择满足以下条件的
被试：

- `HRTF == TRUE`；
- 不在 `Outliers_2026-05-06.csv`；
- `Free Field EQ File != "EQ File Corrupted"`。

当前冻结快照为 372 条元数据、350 名干净测量被试。下载器只获取
`PXXXX_FreeFieldCompMinPhase_44kHz.sofa`：该版本已截窗和自由场补偿，
保留个体 ITD，并与现有 HUTUBS 的 44.1 kHz 采样率一致。

先做全队列只读检查：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.download_sonicom `
  --dry-run `
  --workers 8
```

确认后正式下载：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.download_sonicom `
  --workers 8
```

仅下载少量被试做 pilot：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.download_sonicom `
  --subjects P0001 P0002 P0003
```

脚本支持断点续传、临时 `.part` 文件、下载完成后的原子替换、远端
`Content-Length` 校验和重复运行跳过。输出还包含：

```text
metadata_and_readme/                 官方元数据及异常说明
manifests/clean_subjects.csv         350 名正式队列及来源 URL
manifests/excluded_subjects.csv      被排除被试与原因
manifests/selection_report.json      元数据 SHA-256 与选择口径
subjects/                            下载的测量 SOFA
```

如果官网元数据条数或干净被试数改变，脚本默认停止，防止实验队列无意漂移。
只有人工审核新版元数据后，才应使用 `--allow-metadata-drift`。

## SONICOM 处理约定

SONICOM 的 793 点网格由 11 个 5° 方位角间隔的水平圆环和一个北极点组成，
仰角只覆盖 `-45°` 至 `90°`。它不能直接提供完整 Lebedev `N=3` 网格：
26 个目标中只有 17 个精确匹配，最近邻映射只有 25 个唯一实测点，南极点的
最近误差为 45°。因此 SONICOM 扩展实验不伪造南极方向，也不直接复用 HUTUBS
的 Lebedev/Fliege 网格。

处理配置由以下命令生成：

```powershell
D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.prepare_sonicom_configs
```

推荐主线使用 `SONICOM-Q26-v1`：从真实测量位置中选择北极点、一个中垂面种子
和 12 对左右镜像方向，以最大化最小球面间隔，并用三阶实球谐设计矩阵打破并列。
全部 793 点使用覆盖实测球冠的归一化球面面积权重；主评估在排除 26 个输入点后
的 767 个纯插值方向上进行，同时保留全 793 点辅助指标。

## 数据管理

- 不要将 SOFA、PLY、PDF、CSV 或其他大型数据文件提交到 Git。
- 跨电脑恢复项目时，根据上表来源重新下载数据，并保持相同目录结构。
- 新增或更换数据版本时，在 `docs/EXPERIMENT_LOG.md` 记录来源、访问日期、文件数量、完整性检查和本机路径。
- 数据使用、引用和再分发应遵循来源网站及数据集文档中的要求。
