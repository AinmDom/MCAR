# GL-C3 三候选独立并行运行修订

> 状态：PRE-REGISTERED BEFORE GL-C3。GL-C2 三个 run 正在进行；本文不改变
> GL-C2 winner（尚未产生），只修订 GL-C3 的执行方式。配置基底（optimizer、
> learning rate、weight decay）待 GL-C2 winner 确定后按本文执行。

GL-C2 winner 尚未产生。根据用户在 GL-C2 运行期间、任何 GL-C3 run 启动前
确认的三点决策，GL-C3 按以下方式执行：

1. **三配置全部从 scratch 重跑**：GL-C3 不复用 GL-C2 winner 的 scheduler 数值，
   constant 候选也作为独立新 run 重跑，即新增三个 run：
   1. constant（重跑）；
   2. cosine；
   3. 5-cycle linear warmup + cosine。
2. **并行执行**：三个 run 可同时启动，各自保持独立 seed 状态、scheduler、
   输出目录和 provenance。
3. **配置时机**：三个 GL-C3 配置以 GL-C2 winner 的 optimizer / learning rate /
   weight decay 为基底生成，仅在 scheduler 与 identity 字段上不同；配置在
   GL-C2 winner 确定后生成并提交，不预先假定 winner。

三者使用相同七维 global+local 架构、C0 objective、seed `20260821`、150 cycles、
`scheduler_horizon_cycles=150` 和数据顺序规则。GL-C3 排名只使用这三个新 run；
GL-C2 winner 只保留为 C2 决策证据，不混入 C3 排名，也不作为 constant 候选的
继承数值。

该修订使 screening 物理训练上限从11增加为12（GL-C1 3 + GL-C2 3 + GL-C3 3 +
GL-C4 3），Top-3 补种子后的正式搜索上限从17增加为18；除此以外，GL-C4、Top-3、
E_final 和 ensemble 规则不变。
