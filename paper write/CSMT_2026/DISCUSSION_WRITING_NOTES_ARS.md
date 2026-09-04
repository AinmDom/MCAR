# 讨论章写作记录（ARS分章初稿）

日期：2026-09-04，Asia/Hong_Kong。范围仅为第6章内部初稿及章节接入，不运行完整ARS管线、独立同行评审或引文最终认证。criteria_binding_unavailable。延续作者确认的简体中文、Hybrid E190+Q26、现有证据及至少12页完整稿计划。

## 写前提纲与论点意图

原讨论预算1000个汉字；先保存本提纲及material_passport.json中的M-CSMT-20260904-discussion-v1，再撰写正文。论证由已批准框架和作者答复归纳（argument inferred），不将推断当作作者新增决定。

1. MCA残差补偿的价值：区分相对传统基线的谱误差收益与相对强学习基线的多指标取舍。
2. 局部频谱与双耳线索：解释评价定义的区别，以及训练项和补充指标的对应限制。
3. 结构机制与容量：说明条件基场和频谱CNN的设计分工，保持机制解释、组件观察及因果证据的区别。
4. 泛化边界：限定Q敏感性的模型、split和方向域；不把Q50结果外推为更多测量有害。
5. 统计与个体限制：保留历史test使用、结果已知后主叙事选择、多重比较和P0339案例。
6. 未来工作：记录未完成验证方向，不安排新实验。作者随后确认优先改善水平面ILD和个体失败案例，正文已按该方向组织；不扩展为新实验授权。

## 证据入口

- E1：results/sonicom_complete_ten_method_test_v1/paper_complete_test_wide.csv；补充端点描述性排序。
- E2：results/sonicom_film_siren_spectral_cnn_final_e190_frozen_test_ten_method/paired_bootstrap.csv；Hybrid−各对照的主要配对结果。
- E4：results/sonicom_film_siren_spectral_cnn_final_e190_validation/paired_bootstrap_comparisons.csv；FILMENS比较为validation组件证据。
- E8：results/sonicom_ten_method_direction_sensitivity_v1/paired_direction_effects.csv；HYBRID的Q14/Q50−Q26。
- S1：src/mcar/models/film_siren_spectral_cnn.py、src/mcar/models/residual_mlp_cnn.py；冻结骨干、七通道整谱与零输出初始化。
- S2：configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_ensemble_manifest.json；冻结操作点。
- S3：docs/EXPERIMENT_LOG.md的2026-09-03写作决策与2026-08-31冻结test记录；仅作已有历史证据。
- 定义和重建公式：sections/method.tex、sections/experimental_setup.tex；结果及保留个案见sections/results.tex。

外部解释文献尚未全文核对，本章仅解释本项目方法、定义与已有结果，不新增未经核验引用，不据文献标题推断机制。已有R6定位缺口保留。

## 交付检查

讨论章含三个小节、七个正文段落，共1069个源码汉字（不含注释，含标题），相对1000字预算+6.9%。第3至6章正文累计6851个汉字，另有结果表文字328个；尚不包含引言、相关工作、结论与摘要，不以此证明页数合规。详细实验设置仍超过原预算，整稿装配时需统筹，暂不擅自删改。

源码的章节输入、交叉引用、引用键、括号与环境检查通过。原方法、设置、结果、四张结果表及书目SHA256未变，实验日志和八份既有源表SHA256亦未变。本轮没有重新核算164个表格数值；其2026-09-03转录检查在源文件和表格字节未变条件下沿用，已在SOURCE_CHECK.json标明日期。未产生新统计或外部引用。

作者报告TeX Live正在下载，本轮不操作安装器、不监控下载、不调用尚未确认可用的编译器，未生成PDF或验收版式。

## 反向提纲与论点—证据对应

| 段落／论点 | 证据 | 状态与边界 |
|---|---|---|
| P1：MCA补偿价值与学习方法间取舍 | E2；表primary-results | 支持既有比较；不是总体优越或感知收益 |
| P2：局部谱形、绝对误差及两种ILD须分别解释 | 方法与设置定义；E1；表spectral-results | 定义与描述性结果支持；不归因于某损失项 |
| P3：串联结构有设计依据，组件观察不能隔离因果 | S1/S2；E4；表component-results | 完整开发方案有证据；独立组件效应缺证据 |
| P4：初始恒等与完整推断成本是不同问题 | S1/S2；作者确认无Hybrid完整效率实验 | 初始化由代码支持；效率优势不作结论 |
| P5：Q敏感性限制在固定模型和共同验证域 | E8；表q-results | 观察有证据；输入改变与适配能力只是可能解释 |
| P6：历史test、多重比较及单库限制 | S3；实验设置与作者写作选择历史 | 保留证据范围，不声称新独立确认或统计等效 |
| P7：个体ILD风险与未来优先级 | 结果章P0339保留案例；作者本轮答复 | 个体观察与研究方向；失败原因、感知及效率仍待验证 |

## 五维写作自检

- 贡献：是否只重复排名？否，说明MCA剩余误差可被补偿，同时限定强学习对照的取舍；不声称文献首创。
- 清晰度：是否一段一个任务？七段分别负责价值、指标、结构、成本、输入边界、统计范围与未来方向；术语沿用前文。
- 实验证据：是否把验证组件比较写成CNN因果证明？否，显式保留结构、目标及训练预算的混杂；独立效应仍缺证据。
- 评价完整性：是否回避ILD、失败个体或多重性？否，三者均在正文；不把客观误差改写为听感提升。
- 设计合理性：是否把零初始化写成全程不退化或把新增参数少写成部署轻量？否，区分函数起点、冻结参数及完整推断成本。

上述为同一写作者的有限自检与论点—证据索引，不是独立评审、完整Claim Registry审计、引文终审或投稿认证。无来源机制解释保留为假设，未添加虚构文献。

## 下一步

补写引言／相关工作，继续核对作者所供文献的全文与最近邻关系，再补结论和双语摘要。待TeX Live安装完成后确认编译工具可用，再单独进行PDF生成、图表宽度、页数、匿名性和格式检查。资料侧仍需补R6定位、数据下载manifest与元数据哈希链、RANF终版字段及最终开放范围。
