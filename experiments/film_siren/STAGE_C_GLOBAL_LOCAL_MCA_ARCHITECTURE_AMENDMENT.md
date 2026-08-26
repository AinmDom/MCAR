# Stage C global+local MCA 架构边界修订

> 状态：PRE-REGISTERED REVALIDATION REQUIREMENT。本文在新架构任何 Stage C
> 正式 run 前冻结，不改写既有 global-only C1--C3 结果。

Stage B 三模型三seed复核已将架构从五维 global-only FiLM-SIREN 重新冻结为：

- query：`[x,y,z,dual_frequency_1,dual_frequency_2,normalized_mca_left,normalized_mca_right]`；
- global condition：Q26 encoder latent128；
- modulation：full、all placement；
- local MCA：Q26-derived `mca_logmag_db`，使用train-only mean/std归一化。

既有 Stage C C1--C3 run均在旧五维global-only架构上完成，只能保留为该架构的
历史结果。它们不能证明学习率、optimizer/weight decay或scheduler在新七维
global+local架构上仍是winner；C4在完成新架构重验证前保持暂停。

后续 Stage C 必须先提交独立、有限预算的重验证协议，至少比较既有各阶段winner
与必要对照；所有候选从scratch、只用262 train/44 validation、禁止读取test。
