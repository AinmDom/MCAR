# Stage C global+local MCA 最终训练冻结

> 状态：FROZEN BEFORE FINAL TRAINING。本文在三个正式固定周期 run 启动前提交；
> 它执行既有 Stage C 协议的最终训练规则，不新增搜索维度。

## 1. 最终配置与 E_final

Top-3 三-seed best validation Stage C total 的等权均值为：

1. cosine+C0：`0.7147615646774118`；
2. warmup_cosine+C0：`0.7148944180120121`；
3. constant+C0：`0.7172623832117427`。

因此最终配置冻结为 AdamW `lr=1e-4`, `wd=1e-4` + cosine scheduler + C0。
其三个开发 seed best cycles 为 `140/130/140`，故
`E_final=round(median)=140`。主指标不相等，不进入 tie-break。

## 2. 三个正式 run

- seeds 固定为 `20260821/20260822/20260823`；
- 三个模型均从 scratch 训练，互不恢复或共享随机状态；
- `stop_cycle=140`，每 cycle 262 steps；
- cosine scheduler horizon 仍为150，无 warmup，仅在 cycle 140截断；
- 七维 global+local MCA FiLM-SIREN、C0 objective、数据、采样和数值设置完全继承
  最终候选；
- 三个 run 可并行执行，必须使用独立输出目录和 provenance。

## 3. checkpoint 与 ensemble

- validation 每5 cycles照常执行，只用于完整性和诊断；
- 不 early stop，不用 validation 选择 checkpoint；`best.pt` 仅为诊断产物；
- 每个正式模型唯一权威 checkpoint 为 cycle 140 的 `last.pt`；
- 任一 run 训练失败时只允许相同配置、相同 seed 从 scratch 重跑；不得回退到
  `best.pt`，也不得替换或删除 seed；
- 最终 residual 为三个权威 checkpoint 输出的1/3等权平均，不搜索权重。

## 4. test 边界

本阶段仅使用262 train和44 validation subjects，test 路径不得构造或打开，所有正式
报告必须为 `test_subjects_read=0`。三个 run 完成并核验后，另行生成包含 config/
checkpoint hashes 的 frozen manifest 和 test registry；两者提交前不得请求 test
访问。
