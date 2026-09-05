# 论文写作工作区

作者于2026-09-04指定：后续论文写作相关工作统一在此目录及子目录进行。

## 入口

- [SUpDEq原始期刊论文补录（R2缺项已解除）](literature/R2_INTAKE_20260904.md)
- [最新MCA主线审阅PDF（14页，非投稿版）](CSMT_2026/build_mca_reframe/main.pdf)
- [作者问题清单](AUTHOR_ISSUES.md)
- [MCA主线与比较角色修订记录](REVISION_MCA_SCOPE_20260904.md)
- [本轮数值及编译核对](CSMT_2026/MCA_REFRAME_CHECK_20260904.json)
- [上一版六章PDF（13页，历史预览）](CSMT_2026/build_intro_related/main.pdf)
- [引言与相关工作写作记录](CSMT_2026/INTRO_RELATED_WRITING_NOTES_ARS.md)
- [17份文献身份、核读范围与用途映射](literature/LITERATURE_MAP_20260904.md)
- [本轮编译与保留性检查](CSMT_2026/INTRO_RELATED_CHECK_20260904.json)
- [SONICOM来源快照核验](data_provenance/SONICOM_PROVENANCE_CHECK.md)

- [最新补充材料与作者决定](MATERIAL_INTAKE_20260904.md)
- [文献接收清单与哈希](MATERIAL_INTAKE_20260904.json)
- [论文架构与证据映射](PAPER_OUTLINE_ARS.md)
- [LaTeX主文件](CSMT_2026/main.tex)
- [讨论章](CSMT_2026/sections/discussion.tex)
- [讨论写作记录](CSMT_2026/DISCUSSION_WRITING_NOTES_ARS.md)
- [源码检查记录](CSMT_2026/SOURCE_CHECK.json)
- [迁移核验与旧路径映射](MIGRATION_CHECK.json)

CSMT_2026/内保留sections/、tables/、references.bib、参考文献样式、材料记录和分章写作记录。现有引言、相关工作、方法、设置、结果和讨论六章；下一步补结论、双语摘要，再进行全文统一审阅。全部写作继续放在该目录树中，避免出现平行稿件。

2026-09-04作者最新决定已落实：题目突出MCA残差补偿；横向主表只保留7个外部方法与Hybrid；MCAR、Bounded只在5.4“消融与内部变体分析”中出现。其原数值和不利结果保留，明确不是严格单因素消融。外部八方法的加粗不代表原十配置归档排名或统计显著。

## 路径与证据约定

### 版本控制范围

- 纳入Git：论文框架、LaTeX正文/图表/书目、写作记录、小型来源快照、文献身份索引与结构预检记录。
- 仅本地保留：ref/中的原始文献（沿用仓库既有忽略规则）、全文提取缓存、页面检查截图、LaTeX编译目录及PDF。忽略不等于删除，请保留本地材料备份；新检出仓库需自行准备对应文献，审阅PDF可按下方命令重新生成。
- 本轮整理不搬动或删除已有文件，不改变实验代码、配置、冻结结果及源PDF。提交仅在本地完成，不等于公开发布或推送；公开材料仍须按作者另行确认的范围筛选。
- docs/AI_HANDOFF.md沿用仓库忽略规则，仅作本机动态交接；稳定写作状态由本目录记录承载。

- 原reports/PAPER_OUTLINE_ARS.md现为本目录PAPER_OUTLINE_ARS.md。
- 原reports/CSMT_2026/现为本目录CSMT_2026/，内部层级不变。
- 文档中的可点击相对链接按文档所在目录解析；记录中的results/、src/、configs/、experiments/、docs/等证据路径以MCAR仓库根目录为基准。
- 实验结果、代码、数据和作者原始模板不迁入、不复制为新实验产物。旧Bounded历史稿保留在reports/，不是当前Hybrid写作入口。
- 历史交接与实验日志保留原文，其旧写作路径按迁移映射查找。仓库级交接仍维护于docs/AI_HANDOFF.md。

## 编译入口

2026-09-04已核实TeX Live 2026核心工具与LaTeX Workshop 10.18.0可用。后续完整编译可在PowerShell执行：

```powershell
Set-Location 'D:\course\CUC_2\CSMT\MCAR\paper write\CSMT_2026'
latexmk -xelatex -interaction=nonstopmode -file-line-error -halt-on-error -outdir=build_mca_reframe main.tex
```

2026-09-04本轮已用上述命令成功编译，build_mca_reframe/main.pdf为14页、292942 bytes，正文仍引用13条文献。各页已渲染并目视检查，未见遮挡、裁切或空白页；最终日志无未解析引用/交叉引用、缺字、Overfull或Underfull。仍有中文粗体替代（含新增三级标题的楷体）和caption默认配置提示；尚未完成正式格式验收。132个移表/主表数值匹配原稿及冻结CSV，组件表和方向敏感性表未改。结论、双语摘要未写，14页仍是部分稿。

根目录原main.pdf（11页）及build_intro_related/main.pdf（13页）均保持原样，为历史预览。SOURCE_CHECK.json、INTRO_RELATED_CHECK_20260904.json、迁移及材料接收记录保留历史含义；当前修订检查见MCA_REFRAME_CHECK_20260904.json，问题状态见AUTHOR_ISSUES.md。

当前配置：采用作者指定中文模板，参考文献与附录计入不少于12页目标；实验设置先保留详细版。文献放在ref/；暂仅计划公开最终Hybrid主模型的训练配置和权重，尚未发布。
