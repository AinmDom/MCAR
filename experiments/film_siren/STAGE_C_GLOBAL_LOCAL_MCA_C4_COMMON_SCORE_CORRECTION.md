# Stage C C4 共同评分修订与 D1/D2+notch 前瞻扩展

> 状态：PRE-REGISTERED BEFORE NEW RUNS。本文不改写已完成的 C4 原始记录，也不
> 改写已冻结的 cosine+C0 engineering-test 结论。它透明记录原 C4 跨 objective
> 排序口径的问题，并冻结后续只使用 train+validation 的修正实验。

## 1. 修订原因

原 C4 以各候选自身的 `mean_stage_c_objective_total` 直接排序。C0 的 total 不含
D1/D2/notch，而其他候选的 total 含额外非负项，因此这些 raw total 不在同一量尺上，
不能用来判断哪一个训练 objective 产生了更好的共同基础终点。

本修订定义统一 C0 诊断分数：

`residual SmoothL1 + (0.75*ERB + 0.25*HF + 0.75*strict ILD +`
`0.05*band-ILD SmoothL1) / train-only target_std`。

只读重算 seed `20260821` 的全部 validation ledger 后，各候选最佳统一 C0 为：

1. D1/D2+notch：`0.7075494388355034`（cycle 135）；
2. D1/D2：`0.7093399539540178`（cycle 140）；
3. notch：`0.7144781864630014`（cycle 135）；
4. C0：`0.7170006462537132`（cycle 140）。

该发现发生在 C4 完成之后，因此是开发集上的显式 protocol correction，不伪装成
原协议预注册结果。用户随后明确授权将 D1/D2+notch 推进到最终 validation 横向比较。

## 2. 三 seed 开发扩展

- 候选固定为原 C4 D1/D2+notch：D1/D2=`0.25/0.15`，notch=`0.30`；
- optimizer/scheduler 固定为 AdamW `lr=1e-4`, `wd=1e-4`、warmup-cosine、
  warmup 5、horizon 150；
- 架构固定为七维 global+local MCA FiLM-SIREN；
- 已有 seed `20260821` 保留原 run 和 provenance；新增 seeds
  `20260822/20260823`，各150 cycles，从 scratch、独立输出目录；
- 三个 run 使用同一个增强 objective，因此 run 内 best cycle 和三 seed 聚合均按
  增强 objective raw total；同时报告统一 C0、ERB、HF、strict ILD 和 residual，
  但不再与 C0 的 raw total 混排；
- 任一新增 run 的 best cycle 为150时标记 RETEST，不进入最终固定周期阶段。

## 3. 正式固定周期与 ensemble

- `E_final = round(median(E1,E2,E3))`，其中 Ei 是三个开发 seed 的增强 objective
  best cycle；
- seeds固定为 `20260821/20260822/20260823`，全部从 scratch 训练到 E_final；
- scheduler horizon仍为150；不 early stop，不用 best.pt 组成 ensemble；
- 每个正式成员唯一权威 checkpoint 为 E_final 的 `last.pt`；
- 三成员 residual-dB 以1/3等权平均，不搜索权重。

## 4. Validation 横向比较与 test 边界

正式 ensemble 只生成44名 validation 被试预测。最终横向表固定为四种方法：
D1/D2+notch ensemble、MCAR v3.5.1、RANF 和 FSP-AE。不得把旧 v1/v2 加入该表；
原 C0 FiLM-SIREN只保留为历史开发证据，不进入最终横向表。评价应使用专用四方法
入口或在相同冻结指标下合并逐被试结果，不调用会强制纳入 v1/v2 的旧入口。所有配置、
报告和汇总必须记录 `test_subjects_read=0`。

已消费的 SONICOM test 不得用于本轮训练、checkpoint、E_final、ensemble 或横向排序。
若 validation 结果值得进一步确认，必须先生成新的 hash-locked manifest/registry，
再请求用户单独明确授权；本次授权不包含任何新 test 访问。
