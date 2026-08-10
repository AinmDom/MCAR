# 项目协作指引

## 项目目标

本项目研究在 MCA（Magnitude-Corrected and Time-Aligned Interpolation）HRTF 插值结果上，使用轻量残差网络降低剩余插值误差。

首版学习任务为：

```text
residual = log|H_ref| - log|H_MCA|
```

## 每次开始工作

1. 阅读根目录 `README.md`，了解完整研究路线。
2. 阅读 `docs/EXPERIMENT_LOG.md`，确认当前工作、阻塞项、实验结论和下一步。
3. 检查 `git status` 和最近提交；除非用户明确要求，不覆盖或丢弃已有改动。

## 目录与数据约定

- `docs/`：项目状态、实验记录与结构说明，必须提交到 Git。
- `src/mcar/` 与 `matlab/+mcar/`：项目公共实现；新版本不得复制独立源码树。
- `baselines/mca/`：MCA/SUpDEq 基线入口，不作为主项目。
- `artifacts/`：checkpoint、缓存和生成物，已忽略；精选结果放入 `results/`。
- `external/SUpDEq/`：本地第三方 MATLAB 工具包，已由 `.gitignore` 忽略；不要把其源码或数据提交进本仓库。
- 大型 HRTF 数据集、MAT 文件、中间缓存和模型权重不得直接提交；需在 `docs/EXPERIMENT_LOG.md` 记录其获取方式、版本和本机路径。

## 实验约定

- 首先通过 `baselines/mca/matlab/run_mca_demo_export.m` 复现 MCA demo。
- 基线顺序：SH only、SUpDEq + SH、MCA、MCA + residual MLP。
- 优先使用 Lebedev 稀疏网格；重点指标为 LSD、auditory-band magnitude error、ILD error，以及对侧高频误差。
- 每次实验结束都更新 `docs/EXPERIMENT_LOG.md`：数据集、网格、参数、指标、结论和结果文件位置。
- 每完成一个可恢复的阶段，按日期更新 `docs/EXPERIMENT_LOG.md` 并创建清晰的中文 Git 提交信息。

## Markdown 报告公式兼容性

- 所有实验报告的行内公式使用 `$...$`；显示公式必须使用 `$$...$$`。
- 多行显示公式必须使用 `$$\begin{aligned} ... \\ ... \end{aligned}$$`，两个 `$$` 不得单独占一行。
- 新增或修改报告后，至少在 GitHub Markdown 预览与本地 IDE 预览中检查公式渲染。

## MATLAB

Use the configured MATLAB MCP whenever MATLAB execution is useful for verifying the task.

In particular, use MATLAB MCP when:

- running or debugging `.m` files;
- checking numerical or algorithmic results;
- running MATLAB tests;
- validating MATLAB code changes;
- working with MATLAB toolboxes;
- executing or validating Simulink-related work.

Do not claim that MATLAB code has been executed unless it was actually executed through MATLAB MCP.

For simple source-code reading, documentation edits, or refactoring that does not require execution, MATLAB MCP is optional.

## 验证与交接

- 修改 MATLAB 或 Python 实验代码时，记录实际执行的命令和结果；没有执行时应明确说明原因。
- 跨电脑继续工作时，先同步 Git，再按“每次开始工作”流程恢复上下文。
- 交接前确保状态文档与 Git 提交对应，且下一步是可执行、可验证的具体任务。
