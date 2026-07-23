# HRTF 数据集

本目录保存 MCA 基线、残差学习和跨个体评估使用的本地 HRTF 数据。数据文件体积较大，已通过仓库根目录的 `.gitignore` 排除；Git 仅跟踪本说明文件，不跟踪 `axd/` 和 `hutubs/` 中的实际数据。

## 目录与来源

| 本地目录 | 数据来源 | 本地内容 | 大小 |
|---|---|---|---:|
| `axd/` | [SOFA Acoustics AXD](https://sofacoustics.org/data/database/axd/) | 140 个 SOFA 文件：`p0001`–`p0040`、`p0101`–`p0200` | 394.60 MiB |
| `hutubs/` | [SOFA Acoustics HUTUBS](https://sofacoustics.org/data/database/hutubs/) | 192 个 SOFA、58 个 PLY、2 个 PDF、1 个 CSV，共 253 个文件 | 1,375.71 MiB |

两个数据集均于 `2026-07-23` 从 SOFA Acoustics 的公开目录下载。详细下载参数、文件数量和完整性检查结果见 [`docs/EXPERIMENT_LOG.md`](../../docs/EXPERIMENT_LOG.md)。

## 本地路径

```text
D:\cuc\CSMT\MCAR\data\HRTF\axd
D:\cuc\CSMT\MCAR\data\HRTF\hutubs
```

HUTUBS 后续实验优先使用 `pp1_HRIRs_simulated.sofa`–`pp96_HRIRs_simulated.sofa`；同目录的 measured SOFA、头部网格和人体测量文件保留用于对照与扩展实验。

## 数据管理

- 不要将 SOFA、PLY、PDF、CSV 或其他大型数据文件提交到 Git。
- 跨电脑恢复项目时，根据上表来源重新下载数据，并保持相同目录结构。
- 新增或更换数据版本时，在 `docs/EXPERIMENT_LOG.md` 记录来源、访问日期、文件数量、完整性检查和本机路径。
- 数据使用、引用和再分发应遵循来源网站及数据集文档中的要求。
