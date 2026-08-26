# GL-C4 四候选独立并行运行修订

> 状态：PRE-REGISTERED BEFORE GL-C4。GL-C3 三个 run 已完成并冻结；本文不改变
> GL-C3 winner，只修订 GL-C4 的执行方式。配置基底（optimizer、learning rate、
> weight decay、scheduler）为 GL-C3 winner，已确定。

GL-C3 winner 为 `AdamW / lr=1e-4 / wd=1e-4 / warmup_cosine (warmup 5)`。根据
用户在 GL-C3 完成后的明确要求，GL-C4 不复用 GL-C3 winner 的 C0 数值，而是以
相同 seed `20260821` 同时启动四个从 scratch 的独立 run：

1. C0（重跑）；
2. C0 + D1/D2，权重 `0.25/0.15`，频率 `>=4 kHz`；
3. C0 + multi-scale notch，权重 `0.30`，4–18 kHz、半径 `{4,8,16}` bins、
   depth threshold `1 dB`、softplus temperature `0.5 dB`；
4. C0 + D1/D2 + notch，附加项权重不变。

四者使用相同七维 global+local 架构、GL-C3 winner 的 optimizer/scheduler
（AdamW `lr=1e-4` `wd=1e-4`、warmup_cosine warmup 5）、150 cycles 和数据顺序
规则。GL-C4 排名只使用这四个新 run；GL-C3 winner 只保留为 C3 决策证据，不混入
C4 排名，也不作为 C0 候选的继承数值。四个进程可同时执行，但各自保持独立 seed
状态、objective、输出目录和 provenance。

该修订使 screening 物理训练上限从12增加为13（GL-C1 3 + GL-C2 3 + GL-C3 3 +
GL-C4 4），Top-3 补种子后的正式搜索上限从18增加为19；除此以外，Top-3、E_final
和 ensemble 规则不变。
