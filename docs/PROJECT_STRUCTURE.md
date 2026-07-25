# 项目结构约定

## 设计原则

项目按职责分为五层：

1. `baselines/`：传统方法与论文复现，不作为主项目源码；
2. `src/` 和 `matlab/`：可复用的 Python/MATLAB 实现；
3. `experiments/` 和 `configs/`：实验入口、说明和锁定参数；
4. `artifacts/`：可再生成的大文件和运行状态；
5. `results/` 和 `reports/`：需要提交、引用和交接的最终结果。

新模型版本应优先复用 `src/mcar/`，版本差异写入配置和实验说明，避免再
复制新的 `python/`、`matlab/`、`runs/` 和 `reconstruction/` 子树。

## 目录职责

| 目录 | 内容 | Git |
|---|---|---|
| `src/mcar/` | 数据采样、模型、训练与推理代码 | 提交 |
| `matlab/+mcar/` | 数据导出、HRTF 回填和严格指标 | 提交 |
| `baselines/mca/` | MCA demo、HUTUBS 复现和汇总 | 提交源码 |
| `configs/` | 数据划分和无本机路径的锁定参数 | 提交 |
| `data/processed/` | residual HDF5 和 train-only 统计 | 仅提交统计 |
| `artifacts/` | checkpoint、缓存、MAT、HDF5、日志 | 不提交 |
| `results/` | 精选 CSV、JSON、PNG | 提交 |
| `reports/` | 完整实验报告 | 提交 |
| `external/SUpDEq/` | 第三方 MATLAB 工具包 | 不提交；说明在 `external/README.md` |

## 主要旧路径迁移

| 旧路径 | 新路径 |
|---|---|
| `SUpDEq-master/` | `external/SUpDEq/` |
| `scripts/run_mca_demo_export.m` | `baselines/mca/matlab/run_mca_demo_export.m` |
| `reproduce/*.m` | `baselines/mca/matlab/*.m` |
| `reproduce/hutubs_mca_batch/` | `artifacts/mca_reproduction/hutubs_mca_batch/` |
| `residual_learning/data/` | `data/processed/` |
| `residual_learning/python/` | `src/mcar/` |
| `residual_learning/matlab/` | `matlab/+mcar/` |
| `residual_learning/runs/` | `artifacts/training/` |
| `residual_learning/reconstruction/` | `artifacts/reconstruction/` |
| 精选训练与重建结果 | `results/` |
| `docs/MLP_*_REPORT.md` | `reports/MLP_*_REPORT.md` |

## 发布结果

训练和评估程序首先写入 `artifacts/`。确认数据口径、test 锁定和重建质量后，
再把需要长期保存的 CSV、JSON 和 PNG 整理到相应 `results/` 目录，并在
`docs/EXPERIMENT_LOG.md` 记录生成命令和结论。
