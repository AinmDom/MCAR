# Stage C D1/D2+notch 正式训练冻结（E_final=130）

> 状态：FROZEN BEFORE FORMAL TRAINING。本文在三个正式固定周期 run 启动前提交；
> 它执行既有 C4 共同评分修订的正式训练规则，不新增搜索维度，也不改写已冻结的
> cosine+C0 engineering-test 结论。

## 1. 最终配置与 E_final

C4 共同评分修订（`STAGE_C_GLOBAL_LOCAL_MCA_C4_COMMON_SCORE_CORRECTION.md`）三 seed
开发扩展的 best cycles 为 `140/130/120`，按预注册的 `round(median)` 冻结：

- 最终配置 = AdamW `lr=1e-4`, `wd=1e-4` + warmup-cosine（warmup 5，horizon 150）
  + D1/D2/notch objective 权重 `0.25/0.15/0.30` + 七维 global+local MCA FiLM-SIREN；
- `E_final = 130`；
- 三 seed standardized C0 均值 `0.7067727497385984` 仅作开发诊断，不参与正式成员
  的 checkpoint 选择。

## 2. 三个正式 run

- seeds 固定为 `20260821/20260822/20260823`；
- 三个模型均从 scratch 训练，互不恢复或共享随机状态；
- `stop_cycle=130`，每 cycle 262 steps；
- warmup-cosine scheduler horizon 仍为150，warmup 5，仅在 cycle 130 截断；
- 数据、采样、数值设置与 objective 权重完全继承 C4 修正候选；
- 三个 run 可并行执行，必须使用独立输出目录和 provenance。

## 3. checkpoint 与 ensemble

- validation 每5 cycles照常执行，只用于完整性和诊断；
- 不 early stop，不用 validation 选择 checkpoint；`best.pt` 仅为诊断产物；
- 每个正式模型唯一权威 checkpoint 为 cycle 130 的 `last.pt`；
- 任一 run 训练失败时只允许相同配置、相同 seed 从 scratch 重跑；不得回退到
  `best.pt`，也不得替换或删除 seed；
- 最终 residual 为三个权威 checkpoint 输出的1/3等权平均，不搜索权重。

## 4. Validation 横向比较与 test 边界

本阶段仅使用262 train和44 validation subjects，test 路径不得构造或打开，所有正式
报告必须为 `test_subjects_read=0`。正式 ensemble 在44名 validation 被试上生成预测，
最终横向表固定为四种方法：D1/D2+notch ensemble、MCAR v3.5.1、RANF 和 FSP-AE；
不得把旧 v1/v2 加入该表，原 C0 FiLM-SIREN 只保留为历史开发证据。评价使用专用四
方法入口或在相同冻结指标下合并逐被试结果，不调用会强制纳入 v1/v2 的旧入口。

已消费的 SONICOM test 不得用于本轮训练、checkpoint、E_final、ensemble 或横向排序。
若 validation 结果值得进一步确认，必须先生成新的 hash-locked manifest/registry，
再请求用户单独明确授权；本次授权不包含任何新 test 访问。
