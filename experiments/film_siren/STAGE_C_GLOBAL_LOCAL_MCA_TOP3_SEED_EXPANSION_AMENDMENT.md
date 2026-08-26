# GL Top-3 种子扩展并行运行修订

> 状态：PRE-REGISTERED BEFORE TOP-3 SEED EXPANSION。GL-C4 四个 run 已完成并
> 冻结；本文不改变任何 screening winner，只修订 Top-3 种子扩展的执行方式。

按重验证协议第 4 节，全局 Top-3 各补跑 seeds `20260822/20260823`。Top-3 由
`scripts/analyze_siren_gl_top3.py` 在全部 13 个 screening run 上按 unique
配置（optimizer + scheduler + objective 权重）分组选出（GL-C4 c0 与 GL-C3
winner 为同一配置，共享一个 slot，组内取最优 run）。`results/
sonicom_film_siren_gl_top3/selection.json` 给出最终 Top-3：

1. `sonicom_film_siren_gl_c3_warmup_cosine_seed20260821_e150`
   （AdamW `lr=1e-4` `wd=1e-4`、warmup_cosine warmup 5、C0）；
2. `sonicom_film_siren_gl_c2_adamw_wd1e4_seed20260821_e150`
   （AdamW `lr=1e-4` `wd=1e-4`、constant、C0）；
3. `sonicom_film_siren_gl_c3_cosine_seed20260821_e150`
   （AdamW `lr=1e-4` `wd=1e-4`、cosine、C0）。

根据用户确认，六个扩展 run 全部从 scratch、以相同配置基底同时并行启动；命名
`sonicom_film_siren_gl_top3_rank{n}_{variant}_seed{seed}_e150`，seed 分别为
`20260822`/`20260823`，150 cycles，scheduler horizon 150。每个 run 保持独立
seed 状态、输出目录和 provenance。

该修订使正式搜索总上限为19个正式训练 runs（screening 13：GL-C1 3 + GL-C2 3 +
GL-C3 3 + GL-C4 4；Top-3 补 seeds 6：3 配置 × seeds `20260822/20260823`）。
失败仅允许同 config、同 seed 重跑。Top-3、E_final 和 ensemble 规则不变。
