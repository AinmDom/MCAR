# GL-C2 三候选独立并行运行修订

> 状态：PRE-REGISTERED BEFORE GL-C2。GL-C1 三个 run 已完成并冻结；本文不改变
> GL-C1 winner，只修订 GL-C2 的执行方式。

GL-C1 winner 为 `Adam / lr=1e-4 / wd=0`。根据用户在查看 GL-C1 完成状态后、
任何 GL-C2 run 启动前的明确要求，GL-C2 不复用 GL-C1 artifact，而是以相同
seed `20260821` 同时启动三个从 scratch 的独立 run：

1. Adam、lr=`1e-4`、wd=`0`；
2. AdamW、lr=`1e-4`、wd=`1e-5`；
3. AdamW、lr=`1e-4`、wd=`1e-4`。

三者使用相同七维 global+local 架构、C0 objective、constant scheduler、150
cycles 和数据顺序规则。GL-C2 排名只使用这三个新 run；旧 GL-C1 winner 只保留为
C1 决策证据，不混入 C2 排名。三个进程可同时执行，但各自保持独立 seed 状态、
optimizer、输出目录和 provenance。

该修订使 screening 物理训练上限从10增加为11，Top-3 补种子后的正式搜索上限从
16增加为17；除此以外，GL-C3、GL-C4、Top-3、E_final 和 ensemble 规则不变。
