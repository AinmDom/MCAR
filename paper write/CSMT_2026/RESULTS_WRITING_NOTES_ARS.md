# 结果章起草记录（ARS）

范围：新增第5节结果，不新增实验，不改第3、4节的技术正文。仅为分章初稿及写作自检，非独立同行评审、完整ARS认证或投稿定稿。

## 起草前小纲要

1. 相对MCA的幅度残差补偿：三项频谱指标的均值、相对改善及直接配对区间；单独说明ILD未检出差异。
2. 十方法主要比较：完整列出四个端点，保留各学习方法的优势和代价。
3. 局部频谱细节：四个次要端点完整十方法表，区分最低均值和统计结论。
4. 同验证集FiLM E130与Hybrid E190的组件比较。
5. 验证集Q14/Q26/Q50共同743方向上的输入敏感性。
6. 个体差异：既有胜出人数及保留的P0339失败案例。

论证蓝图状态：argument inferred。以上结构来自已确认总框架及作者“MCA残差补偿优先”的选择。新的结果章论点意向已经追加到material_passport.json；前一轮的意向记录不改写。

## 数据来源（路径相对仓库根）

- E1w：results/sonicom_complete_ten_method_test_v1/paper_complete_test_wide.csv
- E1a：results/sonicom_complete_ten_method_test_v1/aggregate_metrics.csv
- E1s：results/sonicom_complete_ten_method_test_v1/per_subject_metrics.csv（仅查询已报告的P0339 ILD值）
- E2：results/sonicom_film_siren_spectral_cnn_final_e190_frozen_test_ten_method/paired_bootstrap.csv
- E4a：results/sonicom_film_siren_spectral_cnn_final_e190_validation/aggregate_metrics.csv
- E4p：results/sonicom_film_siren_spectral_cnn_final_e190_validation/paired_bootstrap_comparisons.csv
- E8a：results/sonicom_ten_method_direction_sensitivity_v1/aggregate_metrics.csv
- E8p：results/sonicom_ten_method_direction_sensitivity_v1/paired_direction_effects.csv

相关D/E协议和docs/EXPERIMENT_LOG.md已查。仅读取已生成CSV、JSON、协议和日志，未打开原始HRTF／HRIR／checkpoint，未重算bootstrap。

## 转录约定

四项primary主表报告均值±样本标准差，四位小数；仅加粗该列最低均值，标准差不参与“最佳”判定。次要指标表报告均值，不虚构未提供的Hybrid中心配对检验。正文配对区间与均值差通常保留六位，避免近零区间四舍五入成零。

所有表格数据由已读取的源行格式化写入.tex，原始精度及源路径随转录记录保存；该记录不是新的实验产物。相对MCA改善率定义为(MCA均值−Hybrid均值)/MCA均值×100%，不是逐被试百分比的平均，也不是直接从dB差换算线性声压下降。所有置信区间使用现有CSV，不在本轮生成。

主要配对：Hybrid−baseline，10000次、seed20260831。验证组件配对使用既有seed20260828结果。方向敏感性使用既有seed20260902结果。补充端点描述性区间与直接配对区间不是同一种量，不互换。

## 论点—证据对应

| 论点 | 证据 | 支持范围 |
|---|---|---|
| 相对MCA三项频谱误差改善，ILD区间跨零 | E1w、E2 | 同44名历史test被试、Q26、767方向；不是感知收益 |
| 相对学习方法有收益与代价 | E1a、E2 | 均值和未校正配对区间；不同对照不合并成全局结论 |
| 四项最低均值及两项指定次优重点 | E1w | 描述性排序；次要指标无新增直接检验，不声明“六项最佳” |
| 追加细化器后的开发集变化 | E4a、E4p | 44人validation、767方向、三成员对三成员；不隔离结构／预算／损失因果作用 |
| Q14退化、Q50两个谱指标退化 | E8a、E8p | 44人validation、共同743方向及各方法既定路径；不称增加测量一般有害 |
| 个体差异与P0339水平ILD失败案例 | E2、E1s | 保留原44人统计；描述性、事后观察，不剔除、不修补 |

## 写作自检问题

- 贡献：是否回答MCA之后为何还值得学习残差？用三项频谱变化直接回答，不把数据优势等同新颖性。
- 清晰度：是否先总体比较再解释细节？按MCA→学习方法→局部频谱→开发证据→敏感性→个体差异展开，段落角色和来源在源码注释中标明。
- 实验证据：是否只写均值领先？同时报告直接配对区间、ILD代价和个体胜出人数；无区间的次要比较明确仅作描述。
- 评价完整性：是否把重点指标当成全部结果？保留十方法，不把二十端点转换成总分；完整二十端点表仍见冻结E1。
- 设计解释：是否用结果推断CNN、损失或集成各自必然有效？不作独立因果归因，留给讨论说明有限证据。
- 范围：是否将历史test当作未见确认集，或把val数字混入test表？各表题均给出split、方向数、样本量，正文显式区分。

## 构建状态与下一步

2026-09-03结果章写作时，作者确认尚未准备编译环境。不安装软件、不上传到Overleaf；只做源码与转录检查，不报告PDF或页数验收。实验设置先保留上一轮详细版，这不是作者已经回答了设置详略问题。

结果章原工作预算为1800个汉字，实际源码计数1686（正文1358＋四表文字328，不含注释，含标题与表题），偏差−6.33%，不是出版字数。四张表共164个数值已逐项匹配源表的指定精度；章节连接、环境、括号、标签和引用键通过基础源码检查，记录见[SOURCE_CHECK.json](SOURCE_CHECK.json)。全文字数、表格宽度、浮动位置、匿名性及不少于12页均待整篇编译检查。

2026-09-04进度：讨论章已完成内部初稿，见[讨论写作记录](DISCUSSION_WRITING_NOTES_ARS.md)。作者报告TeX Live正在下载，尚未确认可用；下一步写引言／相关工作，再补结论与双语摘要。缺少的感知／效率／隔离消融证据仍列为局限，不作为本轮新增实验任务。
