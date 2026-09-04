# 2026-09-04 作者补充材料接收记录

## 本次范围

后续更新：作者已补入原电脑7份来源文件，4份元数据哈希、纳入／排除规则及冻结split一致性均已核对。此前“快照待提供”的缺项已关闭；数据许可仍待核。见[SONICOM来源核验](data_provenance/SONICOM_PROVENANCE_CHECK.md)。以下保留首次接收阶段的记录，不回写为当时已核验。

使用academic-research-suite的材料接收规范，区分作者确认、文件已收到、历史记录支持及待核读状态。这里只接收材料并更新写作配置，不生成新正文，不修改实验，不上传或发布。
机器可读清单与文件哈希见[MATERIAL_INTAKE_20260904.json](MATERIAL_INTAKE_20260904.json)。其中路径以MCAR仓库根目录为基准。

## 已确认的写作配置

| 项目 | 当前决定 |
|---|---|
| 模板 | 使用作者指定的 `D:/course/CUC_2/CSMT/论文模板/Latex Templates_Chinese` |
| 页数 | 不少于12页，参考文献与附录计入；这是作者声明的配置，不代表组委会另行认证 |
| 补充材料 | 是否接受未知；关键证据不依赖另行上传补充材料 |
| 实验设置 | 先保留详细版，不再询问同一偏好 |
| 编辑环境 | VS Code + LaTeX Workshop |
| 公开意向 | 暂仅公开最终Hybrid主模型的训练配置和权重；不承诺公开完整源码、数据或全部实验材料 |
| 公开状态 | 尚未发布；不因本记录授予上传、权重检查或对外公开权限 |

其余已确认决定保持：中文CSMT会议稿、Hybrid E190+Q26为写作主模型、Young内部占位、9月7日前完整初稿、先讲MCA残差补偿再比较学习方法、不新增实验、未来工作优先水平面ILD与个体失败案例。

## 文献接收

[ref目录](ref)中共有17份PDF，合计66753429 bytes：database目录3份，method目录8份，review目录6份。全部文件非空，已记录SHA-256；哈希只用于固定接收时文件，不是PDF解析或内容完整性认证。

文件名可初步对应此前的11项论文／工具文献；HUTUBS数据资源不要求再提供一篇论文PDF。review目录中的6份补充材料按作者分类接收，尚不据文件夹名称判定每份都是综述文章。没有解析PDF、核对全文、确认正式出版年或标记作者已读。

下一阶段先核对标题、作者、DOI、正式版／接受稿／预印本关系，再建立原始方法文献与综述的用途映射。FiLM文件名中的2017、RANF文件名中的2025不直接写入正式书目年字段。HUTUBS论文、接受稿与数据集按不同引用对象处理，不重复计算论文数量。

## SONICOM来源记录：原快照在另一台电脑，历史证据已核对

### 本轮可直接核查的证据

- [数据目录说明](../data/HRTF/README.md)和[实验日志](../docs/EXPERIMENT_LOG.md)的2026-07-30条目。
- [下载器](../src/mcar/data_tools/download_sonicom.py)中的根URL、文件路径模板、异常快照名、队列规则、逐文件下载与manifest字段。
- Git提交`ef93a5fcaef47e9929a45936dd2bd3e3591f1d29`存在，提交时间2026-07-30 23:43:06 +08:00。提交存在不等于大型数据本身被Git归档。
- [数据准备报告](../configs/data/sonicom_preparation_report_v1.json)保留350人队列的262/44/44配置；本轮不重划分、不读取原始测试数据。

历史日志记录：372条元数据，排除22人，清洁队列350人；350/350 HEAD检查成功，首轮347/350，随后对P0310、P0312、P0313续传，最终350/350、0残留.part、872.82 MiB。日志还记录全部Data.IR为(793,2,256)、44100 Hz且数值有限。这些是历史内容审计结果，**本轮没有扫描350份SOFA重新验证**。

### 跨电脑路径与未核项目

作者提供的`D:/cuc/CSMT/MCAR/data/HRTF/sonicom_measured_ffcmp_minphase_44k1`和本机仓库对应的`D:/course/CUC_2/CSMT/MCAR/data/HRTF/sonicom_measured_ffcmp_minphase_44k1`均不存在。本机`data/HRTF/`仅列出README.md；按目标文件名搜索仓库没有定位到selection_report、clean_subjects、excluded_subjects及官方元数据快照。

作者随后明确确认：HRTF数据及上述来源核验在另一台电脑上，路径差异不代表数据丢失。后续只需将那台电脑的原有manifests和metadata_and_readme两目录复制至paper write/data_provenance/sonicom/下，保留结构。不要重新生成manifest、修改时间字段或重新下载当前元数据替代历史快照；不需要复制SOFA。

需取得的7份小文件：manifests/{selection_report.json,clean_subjects.csv,excluded_subjects.csv}；metadata_and_readme/{metadata.csv,README.txt}；metadata_and_readme/important_information/{Outliers_2026-05-06.csv,README_SONICOM_OUTLIERS.txt}。若目录还有原始许可证或下载日志，可以一并提供；没有则如实保留缺项。目标资料目录仅为推荐接收位置，本轮未创建或复制内容。

以下暂按作者提供的信息保留，不能标为快照核验完成：
- 官方README标注：v0.1, 16/03/2022；
- manifest创建时间：2026-07-31T01:52:14.740315Z，即香港时间2026-07-31 09:52:14.740315；
- metadata_sha256记录及对应原文件的哈希匹配。

### 原始压缩包与版本措辞

下载器直接逐个下载`PXXXX/HRTF/HRTF/44kHz/PXXXX_FreeFieldCompMinPhase_44kHz.sofa`，没有先下载压缩包再解压的步骤。因此该下载路径不要求补原始压缩包名称；不伪造压缩包、独立stdout/stderr日志或缺失的哈希。

版本说明使用“44.1 kHz、截窗、最小相位滤波器自由场补偿、保留ITD”。[SONICOM官方页面](https://www.sonicom.eu/tools-and-resources/hrtf-dataset/)明确最小相位的是补偿滤波器，不是HRTF本身。官网页面所列200人属于另一时期描述，不替代本项目日志中的350人冻结队列。数据集目录的2022和README日期也不作为完整冻结版本号。

明确数据许可证文本或编号仍待查，不能由公开下载入口推断具体许可。官方介绍页已读取，其传输站未能通过本轮浏览工具读取；不据此判断网站对作者不可用。没有替换、下载新元数据或授予再分发权利。

## 编译工具与既有PDF

本轮只读版本查询确认：
- XeTeX 0.999998，TeX Live 2026；
- BibTeX 0.99e；
- latexmk 4.87；
- LaTeX Workshop 10.18.0已经安装；
- xelatex、bibtex、latexmk与VS Code均可由系统路径找到。

latexmk版本查询出现Perl locale回退警告，但完成输出；未改系统区域设置。无需为使用LaTeX Workshop另装TeXworks。

检测到本轮开始前已有`CSMT_2026/main.pdf`，264238 bytes，修改时间2026-09-04 13:46:36 +08:00。其[main.log](CSMT_2026/main.log)记录11页，但有`No file main.bbl`、引用／交叉引用未解析以及字体／caption警告。没有打开PDF做视觉验收，没有宣称完整稿已达12页，也没有重新编译或覆盖该产物。

下一次编译应采用XeLaTeX+BibTeX的完整依赖流程（可由latexmk管理），再检查参考文献与交叉引用；这里只记录下一步，不创建编辑器配置或安装软件。

## 剩余工作与责任

- 作者侧：从另一台电脑提供原有manifests与metadata_and_readme小目录；若能取得明确许可证说明一并补充。真实署名及声明继续暂缓。
- 助手侧：核读已收到的文献，定位SONICOM正文引用和RANF版本；写引言／相关工作，再补结论与双语摘要，处理完整编译和图表。
- 投稿前：确认补充材料规则、数据许可、最终公开包内容与可用方式。只公开配置和权重不等于完整开源；若缺模型定义或依赖说明，复用可能受限，应如实说明，不擅自扩大公开范围。
- 不阻塞继续写作：原始压缩包不存在、独立下载stdout/stderr未归档、补充材料规则未知及暂未公开源码。
- 保留边界：原始test_subjects_read=0；训练／推理／指标重算／bootstrap／checkpoint读取均无；finite与best/末点预算本轮N/A。Baton保持UNKNOWN。

本记录是最新接收状态；此前写作记录、SOURCE_CHECK.json、MIGRATION_CHECK.json和material_passport.json中的阶段性缺项保留历史含义，不据材料到位自动改为引用通过或全文完成。
