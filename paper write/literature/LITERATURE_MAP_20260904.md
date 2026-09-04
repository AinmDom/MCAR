# 文献身份、核读范围与正文用途

## 最新补录：R2原始期刊PDF已收到

2026-09-04后续更新：已核对新补入的R2原始期刊论文，共12页、结构预检PASS；实际核读及目视检查第1–4页（摘要、引言、II方法/图1、评价开头）。作者、卷期、页码与DOI均匹配现有R2题录。见[R2补录与引用定位](R2_INTAKE_20260904.md)及[R2索引](R2.index.json)。

当前本地有18份不同内容的文献PDF（目录实际19个文件，含R2S同哈希副本），结构状态16 PASS、2 UNAVAILABLE；原17份PDF_INDEX.json和以下17份核读记录是较早阶段快照，保留不改。“R2原始PDF待补”已解除，正式勘误全文仍待核。下一轮RW-P1机制句应引R2，采样扩展句保留R2S；本次未改正文或重编PDF。

日期：2026-09-04（Asia/Hong_Kong）。对象：作者提供的17份PDF及原DOI清单。用途：引言与相关工作的限定来源核读，不是系统综述、完整文献审计、独立评审或新颖性认证。

## 核读与检索范围

- 输入：17份PDF，路径、SHA256、页数和结构判定见[PDF_INDEX.json](PDF_INDEX.json)。逐页提取文件`R*.text.json`仅供本地检索，提取全篇不等于已逐页核读。未登记任何作者已读状态。
- 实际阅读：所有17份均核读题名、出版标识及摘要；核心方法另核读下表所列相关页。`PDF页`从文件第一页起计，包含存储库封面，不等于印刷页码。正文优先采用可见章节名定位。
- 外部补核：以精确题名、DOI检索出版方、作者/机构页与arXiv；另检索`"HRTF" "residual" "neural" "MCA"`了解相邻线索。主要核对日期为2026-09-04，覆盖已提供的2017–2026年来源。不宣称完成数据库系统检索、前后向引文穷尽或达到检索饱和。
- 纳入原则：直接支持任务、传统插值、条件表示、学习对照、数据格式或评价边界。泛空间音频综述保留为背景储备，不强制堆入正文；原研究支撑技术机制，综述用于分类与边界，不替代本项目实验。
- 原passport的`literature_corpus`是四条旧摘要快照，缺少完整标准corpus字段；不把它当作合规完整语料或覆盖证书，本表记录本轮真实核读。原条目与旧论点意图保留。

## 版本与身份的重要处理

1. **RANF（R5）按终版卷年2026引用**：本地终版首页记载在线发表2025-12-04、current version 2026-01-01，页脚`VOLUME 7, 2026`，页码32–41，DOI仍为10.1109/OJSP.2025.3640517。[MERL](https://www.merl.com/publications/TR2026-007)的题录与BibTeX使用在线年2025，差异保留，不混成另一个版本。IEEE在线页面本轮返回418；终版页脚为本轮年字段依据。
2. **R2与R2S不是同一论文**：原DOI 10.1109/TASLP.2019.2908057对应Pörschmann、Arend、Brinkmann的期刊论文 *Directional Equalization of Sparse Head-Related Transfer Function Sets for Spatial Upsampling*，2019，27(6):1060–1071；[机构题录](https://research.aalto.fi/en/publications/directional-equalization-of-sparse-head-related-transfer-function/)已核。提供的Arend2019 PDF是Arend、Pörschmann的ICA2019采样方案论文（R2S），2643–2650页，[作者公开原文](https://audiogroup.web.th-koeln.de/PUBLIKATIONEN/Arend_ICA2019.pdf)亦确认。正文现用R2S已读内容，R2只保留题录，不伪称已读原期刊全文。另发现[作者机构出版清单](https://www.th-koeln.de/mam/downloads/deutsch/hochschule/aktuell/presse/projekte_publikationen_2021.pdf)列R2勘误DOI 10.1109/TASLP.2020.3010608，内容尚未核读，后续深入复述R2公式前应一并核对。
3. **HUTUBS三种标识分开**：R9是期刊论文10.17743/jaes.2019.0024；本地封面给出的10.14279/depositonce-15233是同文接受稿；R10的10.14279/depositonce-8487是数据资源。后两者不作为两篇新增论文，也不暗示本文使用HUTUBS做了额外实验。
4. **FiLM（R7）文件名2017，正文为AAAI-18**，按2018论文引用。R3 DOI含2020但卷期为2021。R6 DOI含2022但发表于2023。均不按文件名/DOI数字猜年份。
5. **SOFA（R8）封面漏作者**：HAL封面列五人，正文题名页列六人，包括Franz Zotter。本稿采用正文六作者，不照抄封面不完整列表。
6. **两篇综述是预印本**：R17明确arXiv:2509.00400v1；`JOURNAL OF LATEX CLASS FILES`是模板占位，不能当刊名。[arXiv记录](https://arxiv.org/abs/2509.00400v1)支持该身份。R18采用已读arXiv:2506.19404v1；[版本页](https://arxiv.org/abs/2506.19404v1)另给关联会议DOI 10.1109/I3DA65421.2025.11202063，本轮不把未核全文的会议版元数据与v1定位混写。
7. **SIREN与AKtools**：SIREN引用NeurIPS2020正式作品；AKtools引用AES142 Engineering Brief309，2017。AKtools存储库10.14279/depositonce-15094是接受稿标识，不伪称常规会议出版DOI。

## 来源 × 用途矩阵

以下“支持/背景”针对具体用途，不是支持Hybrid优于该文献；“质量”限定为用途匹配及已读内容可靠性，不作期刊声望评分。

| ID / 来源 | 类型、年份 | 实际核读范围（PDF页） | 能支持的内容 / 本稿用途 | 限制与当前处理 |
|---|---|---|---|---|
| R1 MCA | 原始方法，2023 | 1–3：摘要、引言、II方法、图1及校正流程 | 支持时间对齐、频率平滑幅度目标与后校正；引言、相关工作、方法 | 核心用途匹配；不移植其听音结果作为Hybrid效果；未全篇精读 |
| R2S Spatial upsampling…Influence of the spherical sampling scheme | 原始方法/采样比较，2019 | 1：摘要与引言；另核作者公开题名与摘要 | 支持SUpDEq方向均衡、对不同球面网格的比较这一概述；相关工作 | 不是R2期刊原文；结构预检UNAVAILABLE，仅使用独立可见章节定位，不复述数值和公式 |
| R3 Assessing Spherical Harmonics Interpolation… | 原始评价，2021 | 1：摘要、0引言 | 支持稀疏SH的截断/混叠问题及时间对齐作用；引言、相关工作 | 不把其听音阈值推广到本项目Q26；未完整评价方法/统计 |
| R4 Spatial Upsampling…Source Position and Frequency（FSP-AE） | 原始方法，2025 | 1–5：摘要、I–III、IV.A–C前段及图1 | 支持位置/频率条件化编码解码、位置独立潜变量、幅度/ITD类型标识；相关工作 | 不把跨数据集能力或原报数值写成项目已验证结论；未全篇精读 |
| R5 RANF | 原始方法期刊扩展，卷年2026 | 1–5：摘要、I、II、III.A–D前段、图1–2 | 支持检索增强、部分参数适配、卷积/BLSTM频率序列、可选panning辅助；相关工作 | 与ICASSP短文分开；已经存在跨频率建模，禁止“首次谱建模”主张；未全篇精读 |
| R6 The SONICOM HRTF Dataset | 数据集原始说明，2023 | 1、5、9、10：摘要/0引言、3.1–3.3、5发布、6结论等 | 支持数据背景、最小相位补偿滤波器、793发布方向、变体和采样率；引言、设置 | 论文当时120人，不能支撑本项目350人。后者由归档快照支持；未找到明确数据许可证 |
| R7 FiLM | 原始方法，2018 | 1–2：摘要、1、2.1、2.2 | 支持条件化逐特征仿射调制；引言、相关工作、方法 | 原实验为视觉推理，不证明HRTF效果；项目调制公式另据代码 |
| R8 Spatially Oriented Format for Acoustics 2.1 | 格式综述/规范说明，2022 | 1–2：HAL封面、正文作者/摘要、0引言 | 支持SOFA存储空间声学数据；设置 | 正文六作者优先；不声称本项目文件通过AES标准全套符合性测试 |
| R9 A Cross-Evaluated Database…（HUTUBS） | 数据集原始论文接受稿，2019 | 1–2：封面身份、摘要、0引言 | 支持测量/模拟、网格/人体参数等数据库背景；储备 | 结构预检UNAVAILABLE，不用本地页码锚；本文未做HUTUBS实验，不强制引用 |
| R11 Implicit Neural Representations with Periodic Activation Functions | 原始方法，2020 | 1–3：摘要、引言、相关工作及3定义前段 | 支持周期激活隐式信号表示；引言、相关工作、方法 | 不把其信号细节拟合结论直接当作Hybrid的因果机制证明 |
| R12 AKtools | 软件工程简报接受稿，2017 | 1–2：封面、摘要、1、2.1 | 支持工具身份、功能与可复现处理背景；储备 | 具体项目调用版本尚需代码逐项绑定；不自动将EUPL推广到本项目全部代码或数据 |
| R13 Measurement of Head-Related Transfer Functions: A Review | 综述，2020 | 1：摘要、1引言 | 背景：高密度个体测量成本与耗时；引言 | 摘要/引言级核读，未核其全部测量系统比较 |
| R14 Spatial audio signal processing…review and challenges | 综述，2022 | 1：摘要、1引言 | 背景：录制声场到双耳重放的处理链；储备 | 重点为录制声场/阵列，不直接证明稀疏个体HRTF重建贡献，当前不引入 |
| R15 A Review on Head-Related Transfer Function Generation for Spatial Audio | 综述，2024 | 1：摘要、1引言 | 背景：物理、形态、学习与插值路线分类；相关工作 | 不以综述替代最近方法原文，不声明覆盖所有相邻研究 |
| R16 A Survey on Machine Learning Techniques for HRTF Individualization | 综述，2025 | 1摘要/引言，7输入分类和人体测量段 | 背景：按数据输入/模型/评价组织个体化研究；相关工作 | 阅读范围不足以核定所有收录研究的质量与数值 |
| R17 Deep Learning for Personalized Binaural Audio Reproduction | 综述预印本v1，2025 | 1：题名/版本、摘要、I前段 | 背景：显式HRTF滤波与端到端重放分类；储备 | 不伪装成已刊期刊文，不为凑引用强行纳入 |
| R18 Loss functions incorporating auditory spatial perception…a review | 综述预印本v1，2025 | 1–2：摘要、I、II、表I及III前段 | 背景：空间线索度量与感知质量并非直接对应；相关工作 | 范围主要为双耳信号损失，不是本文精确目标/系数来源；关联会议版未核全文 |

R2保留已核题录、等待期刊全文（未伪装为第18份已读PDF）；R10仅为已识别数据资源，不列作已核读论文。`.bib`共18条工作记录，未引用项不通过`nocite`强制输出。

已知来源年份、研究类型和来源机构以文件为据；没有人口地域数据，不推断地域分布。17份PDF中没有某个单一年份或单一出版来源达到70%；将2010年代与2020年代作粗粒度划分时，2020年代13/17（76.47%），记为DISTRIBUTIONAL_SKEW_ADVISORY，符合本轮以当代方法为主、保留FiLM/AKtools/SUpDEq/HUTUBS基础来源的用途，不解释为缺陷或检索充分性证明。

## PDF结构预检边界

17份中15份PASS，2份UNAVAILABLE，0份FAIL。R9三种页数均15，但有xref覆盖警告；R2S带加密标记，预检不认证，即使阅读器可提取文本也不改成PASS。正文R2S采用可见`Abstract`定位，R9本轮未在正文引用。所有PASS仅为STRUCTURE_ONLY，不能证明文字完整、OCR准确、图表正确或引用语义成立。

Windows版预检器的`--output`要求POSIX dirfd，直接调用返回参数错误；后改为读取检查器标准输出并原样保存JSON侧车，未修改检查器、未降低判定。提取脚本与产物均留本目录，原PDF未改。

## 主张—证据与段落定位

| 稿件主张/段落 | 直接证据 | 处理 |
|---|---|---|
| 引言I-P1，稀疏测量到未观测方向 | R6 0引言、R13 1引言；本文任务配置 | 只说明输入/输出及测量动机，不声称已部署 |
| I-P2，MCA基础与后续补偿问题 | R1 I–II、R3摘要 | 采用“继续研究”的问题框架，不捏造文献空白 |
| I-P3/P5，冻结基场+双耳整谱增量 | R7/R11概念；`src/mcar/models/film_siren_spectral_cnn.py`、正式E190 manifest/协议 | 精确结构贡献与复用归属分开；不称全球最优或轻量已验证 |
| I-P4/P6，三问题、收益和ILD代价 | 已确认框架RQ1–RQ3、结果章及其冻结源表 | 不新增统计；本轮仅读历史协议/配置/日志与既有写作数值，不访问原始test |
| RW-P1，经典谱/空间处理 | R3、R2S、R1 | 仅使用核读范围支持的机制概述 |
| RW-P2，学习输入与最近方法 | R15/R16分类、R4 IV、R5 II–III | 强对照不省略；区分原模型能力与项目统一重评价 |
| RW-P3，条件表示与客观评价边界 | R7 2.1、R11摘要、R18 II | 不由误差下降推导听感收益，不引用他文证明自设计系数 |
| 设置来源段 | R6 3.2/5、R8摘要、`paper write/data_provenance/SONICOM_PROVENANCE_CHECK.md` | 清除已失效manifest缺失标记，仍保留许可证未核 |

## 下一步文献工作

- 补读R2期刊原文及其勘误；未补齐前不以R2支撑详细公式或结果。
- R18如改用正式会议版，应取得该版全文并重新核对定位；现阶段明确引用v1。
- 完整稿阶段检查所有可见引用的语义、版本与GB/T样式；17份材料接收/核读不等于完整引用认证。
- 当前没有独立确认集、完整隔离消融、Hybrid端到端效率或听音证据；这些限制保留，不启动补实验。
