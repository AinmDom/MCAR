# CSMT 中文论文：方法与实验设置写作记录

状态：分章节初稿，不是完整稿、同行评审结论或投稿认证。日期：2026-09-03（Asia/Hong_Kong）。

## 本轮作者已确认

- 方法名称“FiLM-SIREN双耳频谱细化模型”，简称Hybrid；E190仅为实验预算标识。
- 研究起点是对MCA插值结果学习残差，补充修正HRTF幅度细节；先讲相对MCA的价值，再比较学习方法。
- FiLM与SIREN参考已有论文；作者确认CNN、组合方式及损失训练为项目内自行设计。代码表明Hybrid复用项目已有的BinauralSpectralCNN，不将其描述为在Hybrid阶段从零发明的全新卷积算子。
- 计划公开代码／配置／模型等材料，具体范围以后确定。本轮不发布、不上传，也不写“代码已公开”。
- 中文LaTeX、至少12页完整稿、Young内部占位、9月7日前完成完整初稿、仅投CSMT、暂不增加实验，沿用已确认选择。

## 起草前的小纲要

1. 用MCA幅度残差定义学习任务，并界定观测输入与监督目标。
2. 说明个体条件编码、双频率坐标及FiLM-SIREN基场。
3. 说明七通道双耳CNN、零输出初始化、集成与严格重建。
4. 给出损失、采样、分阶段搜索与正式预算。
5. 说明数据快照、十方法对照、指标与统计。
6. 限定组件比较和输入方向敏感性的解释范围。

论证状态：argument inferred。由已确认框架与作者答复推导CER链，不冒充已有正式Argument Blueprint或完整ARS认证。

## 模块与流程草图

| 模块 | 做什么／如何做 | 设计动机及可支持的范围 |
|---|---|---|
| MCA基线及残差构造 | 稀疏观测产生MCA幅度、correction特征及相位；参考幅度减去MCA得到监督残差 | 保留传统重建链路，仅学习剩余幅度误差；不是学习全部复数HRTF |
| 条件基场 | 观测集编码得到128维条件；七维查询输入六层FiLM-SIREN | 同时提供个体全局条件和查询位置MCA信息；选型依据是历史有限搜索 |
| 双耳频谱细化 | 七通道整谱输入、方向调制、四个扩张卷积块、双耳增量输出 | 显式提供跨频率与双耳信息；模块与损失的独立因果贡献尚未全部隔离 |
| 零输出初始化 | 新CNN末层权重和偏置置零 | 初始函数等于FiLM父模型；不保证训练后所有误差都下降 |
| 固定集成及重建 | 三成员dB残差均值加到MCA；恢复MCA相位与未预测频点后截窗 | 使预测可经统一严格链路评价；不是学习相位或证明感知改善 |

~~~text
Q26观测 ──→ MCA ──→ MCA幅度、correction、相位/未预测频点
    │                       │
    └→ 集合编码 → 条件z     ├→ 查询坐标 + 双耳MCA
                      └─────┴→ 冻结FiLM-SIREN → 基场残差
MCA幅度 + correction + 基场残差 + log频率 → 双耳CNN ← 方向xyz
                                    基场 + 增量
                                         ↓
                       三成员dB残差均值 + MCA幅度
                                         ↓
                恢复相位/未预测频点 → IFFT → 截取256点HRIR
~~~

## 构建与阅读

入口为[main.tex](main.tex)，正文为[method.tex](sections/method.tex)与[experimental_setup.tex](sections/experimental_setup.tex)。从本目录执行：

~~~powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
~~~

这只是为后续可用的XeLaTeX环境准备的命令。本轮PATH未发现xelatex、latexmk、tectonic或pdflatex，没有安装编译器，没有生成或验收PDF。源码检查不能替代编译和视觉排版检查。也尚未验证12页；这是全文目标，不用空白内容把两章强行凑足。

工作副本依据作者提供的中文模板保留A4、页边距、标题层级、无页码、双语题注与顺序编码参考文献。主文件明确标为内部两章草稿，暂不伪造中英文摘要、收稿日期、基金或单位。Young不能直接带入匿名投稿版。原模板未修改。正式匿名版及整篇元信息在全文阶段处理。

## 证据索引

以下路径相对仓库根目录。段落角色与证据ID也保留在.tex注释中。

| ID | 仓库证据 |
|---|---|
| S1 | src/mcar/models/film_siren.py；src/mcar/training/train_film_siren.py |
| S2 | src/mcar/models/film_siren_spectral_cnn.py；src/mcar/models/residual_mlp_cnn.py |
| S3 | src/mcar/losses.py；src/mcar/training/film_siren_stage_c.py；src/mcar/training/train_mlp_v2.py；src/mcar/training/train_film_siren_stage_c.py |
| S4 | configs/experiments/sonicom_film_siren_spectral_cnn_final_seed20260821_e190.json；sonicom_film_siren_spectral_cnn_final_e190_ensemble_manifest.json（同目录） |
| S5 | experiments/film_siren/STAGE_D_FILM_SIREN_SPECTRAL_CNN_FORMAL_E190_FREEZE.md |
| S6 | configs/data/sonicom_preparation_report_v1.json；sonicom_subject_split_v1.csv（同目录）；src/mcar/data_tools/download_sonicom.py；docs/EXPERIMENT_LOG.md的SONICOM下载条目 |
| S7 | matlab/+mcar/evaluate_sonicom_interpolation_baselines.m；src/mcar/evaluation/secondary_metrics.py |
| S8 | experiments/film_siren/STAGE_D_HYBRID_E190_FROZEN_TEST_PROTOCOL.md；STAGE_E_COMPLETE_TEN_METHOD_SECONDARY_DEFERRED_TEST_PROTOCOL.md（同目录） |
| S9 | experiments/film_siren/STAGE_E_TEN_METHOD_DIRECTION_SENSITIVITY_PROTOCOL.md |
| S10 | docs/EXPERIMENT_LOG.md的A2–A4、B1–B9、C、D阶段；results/sonicom_siren_a4_confirmation/decision.json；results/sonicom_film_siren_b_placement/retest_decision.json |
| S11 | results/sonicom_complete_ten_method_test_v1/verification.json；框架E1–E8 |

## 文献读取边界

保留框架原有R1–R12编号，本轮.bib只纳入两章直接引用的R1、R6、R7、R11，其余8条仍在框架中，不代表已删除。未进行完整文献综述或相邻工作排除检索。

- R1：[作者预印本摘要及期刊引用记录](https://arxiv.org/abs/2303.09966)支持时间对齐与插值后幅度校正的概括。预印本只作阅读渠道；书目保留作者提供的2023期刊DOI。未将预印本特有实验结果移植成终版结论。
- R7：[AAAI出版摘要](https://ojs.aaai.org/index.php/AAAI/article/view/11671)支持条件驱动的逐特征仿射调制。正文带幅度／相位边界的正弦公式来自S1，不归给FiLM原文。
- R11：[NeurIPS出版摘要](https://proceedings.neurips.cc/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html)支持周期激活的隐式表示；本项目HRTF效果不能由该论文直接推出。
- R6：书目身份已在前轮核验，本轮AES全文入口抓取失败，正文仅作SONICOM数据集归属引用，精确定位未补齐。保留anchor:none并显式标为待核，不声称引用审计通过。
- [SONICOM官方数据说明](https://www.sonicom.eu/tools-and-resources/hrtf-dataset/)说明最小相位的是自由场补偿滤波器，不代表HRTF自身为最小相位。网页人数为另一时期描述，不用于证明本项目350人队列。
- 本轮只查公开文献，没有上传私有稿件。引用来源标记以LaTeX注释保存，不参与排版；这不是ARS最终引用认证。

## 论点—证据对应与写作自检

| 论点 | 证据 | 当前支持状态 |
|---|---|---|
| MCA残差任务、条件基场及七通道CNN | 作者答复；S1–S4；R1/R7/R11仅支持各自已有概念 | 实现描述有支持；独立新颖性仍需相邻文献对照 |
| 初始Hybrid等于FiLM、三成员dB集成 | S2、S4、S5 | 支持；不可换成“初始等于MCAR” |
| 正式E190来自190/190/170的中位数规则 | S5；S10 | 支持；不是使用test选预算 |
| 架构及调制经有限候选搜索选定 | S10及冻结配置 | 支持；不是全局最优或最终模型全组件消融 |
| 350人、262/44/44、Q26及测量域 | S6 | 本地配置／日志支持；原始下载manifest在记录的相对位置未找到，需恢复元数据hash链 |
| 严格ERB与训练代理不同、ILD由HRIR能量计算 | S3、S7 | 支持；不能把band ILD与strict ILD合并 |
| 历史test及补充端点统计边界 | S8、S11 | 支持；未新增统计、未假称未见确认集 |
| Q敏感性中RANF进行适配、其他指定网络冻结 | S9 | 支持；不可写成十方法全部不训练 |
| 轻量、感知定位更好、各组件独立贡献 | 当前缺乏本稿对应的完整证据 | 不写为结论；不启动新实验补齐 |

反向提纲：方法从任务定义→基场→CNN→重建→训练目标，实验设置从数据与访问边界→对照→指标／统计→有限选型证据与敏感性。段落角色在源码注释中明确，没有用结果排名倒推结构必然有效。

五方面自检（作者可复核的写作自检，不是独立同行评审）：

- 贡献：归属已由作者确认，但“自行设计”与“文献中首次”分开；先讲MCA残差补偿。
- 清晰度：区分MCA correction与CNN增量、FiLM双频率坐标与CNN log频率、归一化残差与dB残差。
- 实验证据：有限搜索历史、冻结预算和既有组件比较分层说明；不把开发搜索变成新消融。
- 评价完整性：保留20端点分类、ILD代价、44被试配对及多重比较限制；后续结果章已起草，过程与核对见[结果写作记录](RESULTS_WRITING_NOTES_ARS.md)。
- 设计合理性：可解释输入、残差与初始化；没有独立证据的效率、感知和因果机制不作保证。

## 下一步

截至2026-09-04，方法、实验设置、结果与讨论四章及结果四表已形成内部源码初稿。讨论见[讨论写作记录](DISCUSSION_WRITING_NOTES_ARS.md)，未来优先改善ILD及个体案例已由作者确认。下一步写引言／相关工作，再补结论与双语摘要。作者报告TeX Live正在下载，本轮未操作安装器或编译；实验设置暂保留详细版，详略偏好仍待答复。

资料侧继续核对：R6全文定位、SONICOM下载元数据hash链、RANF终版年字段、最近邻工作、最终代码开放范围。排版侧需可用XeLaTeX环境和整篇编译／视觉检查。上述工作不授予新实验、数据下载、外部发布或test访问权限。
