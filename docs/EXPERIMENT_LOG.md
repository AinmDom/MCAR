# 项目实验日志

## 2026-09-09：Hybrid E190 单成员五区结构图完成（Nature figure / Python）

- 时间/agent：2026-09-09T21:35:54+08:00，CODEX；状态：图稿已完成并核验。
- 动作：按作者五区要求绘制横向单成员结构图，核对远程项目分支 `codex/project-structure-refactor` 的提交 `99f8052078d1713a028f857b338fc5a092c8eb5c`。相关模型和正式配置与本地 HEAD `7371f792316b6dd50d4161fe7e4b51a9c01c7cd3` 一致；远程 main 为旧目录布局，不作为本图依据。
- 证据：`paper write/figures/hybrid_e190_single_member/` 中的 `draw_hybrid_e190.py`、`provenance.json`（5个源文件的blob/hash）、`FIGURE_CONTRACT.md`、`QA_NOTES.md`、`exports.json` 及最终 PNG/SVG/PDF/TIFF。复现：`python "paper write/figures/hybrid_e190_single_member/draw_hybrid_e190.py" --skill-scripts <nature-figure scripts目录>`。
- 关键结果：保留7通道精确顺序、冻结Q26 FiLM-SIREN、可训练方向条件CNN、4块膨胀率1/2/4/8以及base+delta加法。PNG/TIFF为9600×5106、标称600 dpi；SVG/PDF文字可编辑。图中明确单成员，并按冻结manifest写3成员等权发布脚注；补充每方向N=F与实际grid输出[2,D,F]的形状说明。
- 完整性：最终PDF碰撞检查0 FAIL/0 WARN；最小字形8.1 pt，112个text runs均≥5 pt；源码检查20 PASS/1 WARN/0 FAIL（WARN仅为有意保留全部细节的406.4 mm大幅母图）；5区域与整图均已视觉核验，绘图区对齐N/A（单axes），4个区域间距均70绘图单位。`test_subjects_read=0`，finite/训练/best/末点预算=N/A；未运行训练、评价或数据汇总，未更改.tex。Git：开工干净，图稿目录受既有 `/paper write` 规则忽略，保持本地未暂存；本条日志待提交，无commit/push。
- 下一步：作者可直接使用高分辨率PNG或在SVG/PDF上继续排版；若缩为期刊双栏宽度，先精简文字再检查实际字号。
- 阻塞项：无。沙箱初始化故障以已授权的沙箱外执行完成绘图；原日志所列实验恢复任务未接管、未覆盖。

## 2026-09-06：Q14/Q26/Q50次要指标首次运行在缓存表示转换处中止

- 时间/agent：2026-09-06T12:15:00+08:00，CODEX。660/660传统基线文件和MATLAB退出已核验后，
  启动冻结Python evaluator；它在P0001、首个残差方法加载时因v7.3 cache的复数MCA频谱直接与
  residual dB相加而中止（`UFuncNoLoopError`）。
- 动作：在加载cache时将reference/MCA复数频谱按既有MATLAB定义转换为
  `20*log10(max(abs(spectrum),1e-10))`；不改变预测、端点、掩码、统计或数据边界。
- 证据：`scripts/evaluate_ten_method_direction_sensitivity_secondary.py`；失败堆栈定位
  `residual()`的MCA+residual操作；源缓存为
  `artifacts/sparsity/sonicom_bounded_e25_input_direction_sensitivity_v1/.../cache.mat`。
- 完整性：失败发生在首个listener、首个方法，未写入任何endpoint CSV或最终结果目录；
  `test_subject_count_read=0`，训练/best/末点=N/A，660个输入HDF5未改；脚本更正待提交。
- 下一步：提交该结果盲表示修正，重新执行完整validation evaluator；要求21,120/480/320/288且全finite。
- 阻塞项：无；首次运行未生成可解读的数值结果。

## 2026-09-06：Q26 test八方法ITD补充表完成（结果级派生）

- 时间/agent：2026-09-06T12:07:24+08:00，CODEX。按用户要求整理Q26/test的ITD指标；为满足“八个方法”，方法固定为SH only、SUpDEq SH、SUpDEq NN、SUpDEq Barycentric、MCA、FSP-AE、RANF和Hybrid E190，明确排除MCAR v3.5.1与Bounded E25。
- 动作：从已完成的十方法补充结果`results/sonicom_complete_ten_method_secondary_deferred_test_v1/`按键提取两个已注册ITD端点`ITDWeightedMAE_us`和`ITDMaximumAbsoluteError_us`，生成`results/sonicom_eight_method_hybrid_itd_test_v1/`；新增派生脚本`scripts/build_eight_method_itd_test_supplement.py`。未重新读取原始test subject/prediction，未训练、推理、调参或选择模型。
- 证据：`results/sonicom_eight_method_hybrid_itd_test_v1/summary.json`、`metric_long.csv`、`aggregate_metrics.csv`、`paper_itd_wide.csv`、`README.md`；源manifest identity=`E9403269543B6C135871BD8008809B488F2E29977601C95E3FF2BD491FDB229E`。
- 完整性：44名test、8方法、2端点，逐被试`704`行、汇总`16`行，16个method×endpoint单元各44人；全部numeric finite；派生表相对源逐键最大绝对误差`0.0 us`；`source_test_subject_count_read=44`（源结果历史评价），`new_test_subject_count_read_during_derivation=0`；训练/best/末点=N/A。
- 关键结果（均值，us；lower is better）：ITD weighted MAE为SH`50.3169912016`、SUpDEq SH`15.9970410049`、NN`15.1370557204`、Bary`15.4320967292`、MCA`18.3996845274`、FSP-AE`18.3582875747`、RANF`23.5839908543`、HYBRID`18.0456512678`；ITD maximum absolute error为SH`280.2438383779`、SUpDEq SH`143.8210241428`、NN`144.7088090407`、Bary`145.1822909902`、MCA`141.3944136485`、FSP-AE`123.6387278451`、RANF`195.4308668659`、HYBRID`151.2784113980`。
- 下一步：可直接引用该八方法ITD补充包；如需论文表格，使用`paper_itd_wide.csv`。当前validation恢复任务仍按既有日志处理，本步未访问或覆盖其目录。
- 阻塞项：无。

## 2026-09-06：Q14/Q26/Q50传统基线恢复正式运行中

- 时间/agent：2026-09-06T11:14:00+08:00，CODEX。目录级恢复更正提交`e2eeb8f`后，已启动
  MATLAB恢复进程PID22392（launcher27280），从validation split第41名P0341至第44名P0369运行。
- 动作：逐文件保留610个原有HDF5，已存在prediction直接跳过；仅重建/写入原先缺失的Q单元。
  当前文件数630/660，P0341/Q50已补齐，后续listener继续运行。
- 证据：`matlab/+mcar/evaluate_ten_method_direction_sensitivity.m`；
  `artifacts/reconstruction/sonicom_ten_method_direction_sensitivity_secondary_classical_v1/`；恢复命令：
  `mcar.evaluate_ten_method_direction_sensitivity(4, ..., ..., 41)`。
- 完整性：运行中，不将中间文件或指标视为完成；只读44人validation、固定Q50排除mask，
  `test_subject_count_read=0`；训练/best/末点=N/A；次要指标finite/汇总待运行后核验。
- 下一步：等待PID退出并核验660个预测，随后运行冻结的Python evaluator，检查21,120/480/320/288。
- 阻塞项：无；不得并行覆盖同一导出或结果目录。

## 2026-09-06：Q14/Q26/Q50传统基线恢复首次启动在目录保护处中止

- 时间/agent：2026-09-06T11:12:00+08:00，CODEX。恢复MATLAB命令在启动时因导出根目录已存在
  而触发旧的目录级`assert`后退出（exit 1）；未打开任何listener缓存、未写入HDF5、未计算或查看
  次要指标。
- 动作：将恢复逻辑收紧为“目录可复用、prediction.h5逐文件不可覆盖且已存在即跳过”；该改动只
  允许补齐已记录的50个缺失文件，不能重写610个既有文件。
- 证据：`matlab/+mcar/evaluate_ten_method_direction_sensitivity.m`；首次恢复命令的MATLAB
  报错定位于该文件第43行。
- 完整性：传统基线仍为610/660；test_subject_count_read=0；训练/best/末点/次要指标finite=N/A；
  无运行中MATLAB进程；Git恢复更正待提交。
- 下一步：提交目录级恢复更正，重启第41--44个validation listener，仅补缺失HDF5，然后核验660。
- 阻塞项：无；该失败发生在读数据之前，未改变冻结边界或任何已有产物。

## 2026-09-06：Q14/Q26/Q50十方法次要指标传统基线恢复准备

- 时间/agent：2026-09-06T11:05:00+08:00，CODEX。用户报告传统基线重建已结束后，核验实际
  artifact，发现只存在`610/660`个标准化传统基线HDF5；缺失单元为P0341/Q50及
  P0354、P0360、P0369的Q14/Q26/Q50，共10个listener-Q单元（50个方法文件）。
- 动作：在任何次要/deferred端点计算前，对已冻结导出函数增加仅恢复控制：可从validation
  split的第41个listener开始、跳过已有且不覆盖的prediction HDF5。恢复将仅补上述缺失文件，
  使用原冻结SUpDEq/MCA定义；不训练、不改checkpoint、不访问test。
- 证据：`matlab/+mcar/evaluate_ten_method_direction_sensitivity.m`；
  `artifacts/reconstruction/sonicom_ten_method_direction_sensitivity_secondary_classical_v1/`；
  `configs/experiments/sonicom_ten_method_direction_sensitivity_secondary_v1.json`。
- 完整性：原MATLAB PID6604及launcher23348均已退出；已有610个文件不覆盖，补齐目标660。
  次要指标行、finite、汇总=N/A（尚未运行）；`test_subject_count_read=0`；训练/best/末点=N/A。
  Git基准`bdcb52a`，恢复修正待提交。
- 下一步：提交恢复修正，启动P0341--P0369的4 listener补齐；核验660文件后运行冻结Python
  evaluator并检查21,120/480/320/288的行数。
- 阻塞项：无；唯一异常为已记录的不完整传统基线导出，未产生或查看任何补充指标结果。

## 2026-09-06：作者取消 AI_HANDOFF 交接机制，开工改为读实验日志前三条

- 时间/agent：2026-09-06T10:01:50+08:00，COPILOT。作者认为 `docs/AI_HANDOFF.md` 过长、
  交接冗余，要求在 `AGENTS.md` 中取消交接规则，改为开工前只读 `docs/EXPERIMENT_LOG.md`
  顶部最近三条记录确认进度、最新结论、运行中任务与阻塞项。
- 动作：改写 `AGENTS.md`——移除 `Baton`/当前接力状态/交接页维护/“哪些时点必须交接”，
  把“开始工作前”改为查看日志前三条 + `git status --short` + 分支/HEAD；合并出“记录规则”
  一节，所有可验证步骤统一收敛到 `docs/EXPERIMENT_LOG.md` 顶部（时间倒序，无需另设
  交接页）；保留“目标/任务模式训练启动”“数据与实验边界”“论文写作工作区”及作者
  2026-09-06 的 LaTeX 编译约定。未改动其他文件，既有未提交 AGENTS.md 修改（训练启动
  精简、LaTeX 约定）原样保留。
- 证据：`AGENTS.md`。原 `docs/AI_HANDOFF.md` 已归档至 `docs/archive/AI_HANDOFF.md`
  （作者要求归档，保留历史；文件原被 `.gitignore` 忽略，归档路径同样不入版本库），
  不再按协作规则维护，其 Baton 状态已失效。
- 完整性：本条为协作约定变更，不涉及训练/评价、checkpoint 或 test 访问；
  `test_subject_count_read=0`，训练/末点预算判据=N/A。Git：AGENTS.md、
  docs/EXPERIMENT_LOG.md、.gitignore 均未提交。
- 下一步：无（归档已完成）。
- 阻塞项：无。

## 2026-09-03：作者指定Hybrid E190为当前论文主模型（写作决策，无新实验）

- 时间/agent：2026-09-03T17:11:40+08:00，CODEX。作者明确要求将Hybrid E190作为主模型并修改
  `reports/PAPER_OUTLINE_ARS.md`。本次更新论文呈现角色：Hybrid E190+Q26为主方法，
  Bounded E25为对照，MCAR v3.5.1仍为工程主模型；既有结果已知后作出的作者选择不得回写成
  test前预注册选型，原训练、评价和历史模型决策不重写。
- 冻结模型保持identity `A3CFAC9C206E0A53FD2FA130817673AAFE07B66855322B5D34824E9173BDCCFE`：
  三个seed20260821/22/23的cycle190 last.pt、residual-dB权重1/3；冻结FiLM E130，仅训练新增
  双耳频谱CNN。开发best cycles190/190/170、中位数E190、调度horizon200均按既有协议引用，
  不修改配置或checkpoint。证据：`experiments/film_siren/STAGE_D_FILM_SIREN_SPECTRAL_CNN_FORMAL_E190_FREEZE.md`
  及 `configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_ensemble_manifest.json`。
- 已核验并保留的44人engineering-test、Q26/767方向四primary均值为
  `0.8175440242629026/1.2003965313140803/3.454911258906816/0.7794497051040513 dB`。
  直接Hybrid−MCAR统计来自
  `results/sonicom_film_siren_spectral_cnn_final_e190_frozen_test_ten_method/paired_bootstrap.csv`
  （10000次被试配对bootstrap，seed20260831）：Contra25差−0.07409270078644113 dB，
  HF差−0.053569703946212764 dB，二者CI均小于0；Full差−0.00039302453397463915 dB，
  CI[−0.030995271684337845,+0.055464770777911954]；ILD差+0.13474061344812602 dB，
  CI[+0.012705474934938392,+0.34484925204924877]。框架保留ILD代价及P0339失败案例，不声称全指标支配。
- 方向敏感性使用现有44人validation/743方向结果：Hybrid的Q14四指标退化；Q50的Full/Contra25
  退化、HF/ILD的CI含零。Bounded门控及效率材料未移植为Hybrid证据；CNN可训练参数按现有代码
  维度计数为74402/成员，完整Hybrid效率证据仍待补。
- 完整性与范围：原始 `test_subjects_read=0`；仅读取已生成test CSV、配置、代码和日志，
  未训练、未加载checkpoint、未推理、未运行评价器或新bootstrap。既有primary summary为
  completed/all_finite=true、1760行，20端点verification为8800行且finite、源值误差0；
  不把这些历史验证说成本轮重新验证checkpoint。训练/best/末点预算判据=N/A。
- Git：基准 `codex/project-structure-refactor` / `ae71552da02f0b4e0292e72696c2aba554fe123a`；
  接续上一轮未跟踪框架并修改，增量更新交接页及本条，无提交。下一步为Hybrid正文写作与
  原始文献核验；旧Baton仍UNKNOWN，不启动有重叠风险的实验。
  
  ## 2026-09-05：Hybrid LSD B validation比较完成

- 使用冻结ensemble identity `D1611128A86118ACE0EFD471A4F99456F1267B10D1C39C68AD769239FB589CE6`
  从三个E190 `last.pt`以1/3 residual-dB平均，仅推理44名validation；44/44预测均shape
  `[2,793,463]`且finite，`test_subjects_read=0`。输出：
  `artifacts/reconstruction/sonicom_hybrid_lsd_b_e190_ensemble_validation/`。
- 新旧Hybrid的FullSphereLSD为`3.522658134565499`与`3.5489256190074472 dB`；
  candidate-old=`-0.026267484441947323 dB`，相对降低`0.7401531410313007%`。
  44人配对bootstrap（10,000次、seed20260904）95%区间
  `[-0.029413178596792797,-0.023110671650463797] dB`，win/tie/loss=`43/0/1`；
  最大退化`+0.0030848768436446683 dB`（P0113）。
- 空间上，距Q26的0--10/10--20/20--30度分箱均值差为
  `-0.03671992540414542/-0.02634195261223161/-0.019643460354326514 dB`，区间均小于0；
  30--180度为`-0.003498631694016369 dB`，区间
  `[-0.01535449968429978,0.009002081559052919]`，无法确认改善，且P0166最大退化
  `+0.10160025649015836 dB`。说明整体收益主要靠近输入方向，远区有尾部风险。
- 相对旧Hybrid的权衡：FullSphereERB `+0.006262305233118432 dB`
  （95%CI `[0.005347437741850103,0.0071898296881079335]`，43/44退化）；
  Contra25ERB `+0.01635809484387329 dB`（`[0.01244823377522675,0.02023573630197221]`，40/44退化）；
  ContraHF `+0.0004634001250298074 dB`（区间跨0）；严格Horizontal ILD
  `-0.0017787096177044213 dB`（`[-0.0036125545997007795,-0.000016538471795259764]`）。
  HF一/二阶差分分别`+0.0020616847005757418 dB/bin`与`+0.006357118148695339 dB/bin²`，
  均44/44退化且区间大于0；notch深度`+0.00040992146188562566 dB`；ERB-band ILD
  `+0.0023401650515469637 dB`且区间跨0。
- 完整报告：`reports/HYBRID_LSD_B_VALIDATION_COMPARISON_20260905.md/.json`；严格指标目录
  `results/sonicom_hybrid_lsd_b_vs_old_hybrid_validation/`；secondary目录
  `results/sonicom_hybrid_lsd_b_secondary_validation/`，quality passed/all_finite。
  secondary manifest identity=`041FB8EFEAF15BE56D9DF41CDB07E11956031FA8BCD19435D79805C34FCD2999`。
  大型279136行逐方向LSD表仅本地保留并有manifest hash/复现脚本，不入Git。
- 运行记录：MATLAB首次在沙箱内因filesystem startup错误退出，沙箱外同一命令44/44完成；
  secondary首次产物计算口径正确但summary硬编码旧bootstrap seed，发现后修复为读取manifest并
  删除/重建整个结果目录，最终summary明确seed20260904。汇总脚本首次因primary均值未合并而
  中止，产生的自有JSON已删除，修复后完整重建。上述故障未改数据、模型、预注册比较或最终数值。
- 决策：B作为validation候选得到小而一致的LSD收益，同时出现小幅ERB和频谱平滑代价；
  不提升论文/工程主模型，不访问已消费test，也不把本实验解释成MCA内部ERB校正的因果消融。
  若继续，优先研究LSD权重/频段或远方向加权，但必须另建validation协议；独立确认需要新holdout。

## 2026-09-05：Hybrid LSD B三seed训练完成并通过完整性核验

- attempt2三个进程均exit0，结束于`2026-09-05T01:44:31+08:00`。
  `reports/HYBRID_LSD_B_TRAINING_VERIFICATION_20260905.json`独立核验3/3 run均为
  completed、E190、49,780步（合计149,340步）、38次五cycle间隔validation、
  `FIXED_CYCLE_COMPLETE`，权威checkpoint为E190 `last.pt`。
- 三份history和ledger的全部数值有限，三个`last.pt`全部model_state张量有限，实际SHA256
  与training_report完全一致；训练Git均clean `bd00151`、FiLM frozen=true、train/val=262/44、
  `test_subjects_read=0`。stderr为空。best cycle=185/190/170，但只作总objective诊断，
  不改变预注册E190末点选择。
- E190单模型validation FullSphereLSD按ledger命名字段为seed20260821/22/23：
  `3.5702218142422764 / 3.5682887760075657 / 3.56688669052991 dB`。
  早先交接中把history末行约`2.48`的另一列误读为LSD，已在后续交接明确更正；正式数字以
  verification JSON和ledger字段为准。
- 已冻结三份`last.pt`的1/3 residual-dB ensemble manifest：
  `configs/experiments/sonicom_hybrid_lsd_b_e190_ensemble_manifest.json`，identity
  `D1611128A86118ACE0EFD471A4F99456F1267B10D1C39C68AD769239FB589CE6`。
  下一步只在44 validation被试生成新预测并与旧Hybrid配对比较，不读取test。

## 2026-09-04：Hybrid LSD B三seed并发训练已启动（未完成）

- attempt2于`2026-09-04T23:38:47+08:00`启动；controller PID47352，
  seed20260821/20260822/20260823分别PID2856/22080/43544。
  启动证据：`reports/HYBRID_LSD_B_START_20260904.json`；实时进程退出记录：
  `outputs/hybrid_lsd_b_e190/attempt2/launch.json`，该目录内分别保存stdout/stderr。
- 三run均从干净提交`bd00151f7fc717fc748454e5419fc16d3f65d06e`启动，configuration.json
  确认每run train=262、val=44、FiLM frozen=true、LSD weight=1.0；与冻结配置hash一致。
  输出目录为`artifacts/training/sonicom_film_siren_spectral_cnn_lsd_b_seed{seed}_e190/`。
- 23:39:17+08:00单次启动核验：四进程存活、三configuration.json齐全、stderr均空；
  GPU used=2469 MiB。尚未观察首个完整cycle，不宣称训练完成或全程finite。
  preflight finite已通过；后续须核验190 cycles/49,780 steps/last.pt、ledger/history与权重finite、
  Git/hash及test_subjects_read=0。best仍为总loss诊断，权威末点190不变。
- 初次启动失败已保留证据并修复日志目录忽略规则；未改loss/配置/预算或关闭clean-Git检查。
  不重训旧对照，不访问test。启动后Codex停止主动监控，下一次接手按协议核验并比较validation。

## 2026-09-04：Hybrid LSD B三seed协议冻结、启动前验证通过

- 启动尝试1（23:36:35+08:00）：controller PID43156、训练PID29812/40732/41124，
  三者均exit1，原因是新增运行日志目录未被Git忽略，trainer的clean-Git guard拒绝。
  失败发生在dataset解析前，optimizer_steps=0、test_subjects_read=0，没有正式训练产物。
  保留`outputs/hybrid_lsd_b_e190/launch.json`及stderr；新增仅该运行日志目录的ignore规则，
  attempt2使用独立日志子目录并沿用原三份配置。未关闭clean-Git检查。

- 用户将此前四组设想缩减为B组，明确要求seeds `20260821/20260822/20260823`
  同时运行，旧Hybrid结果作为对照。预注册协议：
  `experiments/film_siren/STAGE_D_HYBRID_LSD_B_E190_PROTOCOL.md`。
- 原MCA/ERB处理、模型、损失和训练设置不变；从各自冻结FiLM E130开始，新建零输出CNN；
  仅新增`1.0 * LSD_loss / target_std`，训练sqrt内加入`(1e-6 dB)^2`，报告LSD不加epsilon。
  LSD先逐耳/方向跨463频点算RMS，再双耳和立体角方向平均；validation最后44被试等权平均。
  三份同seed配置自动比较通过，仅身份字段及两个LSD参数不同。
- 固定E190、49,780步、horizon200/warmup2；权威checkpoint为`last.pt`，
  `best.pt`仅原总loss诊断。不得按LSD改选checkpoint、调权重或追加预算。
  完成后构造三成员等权residual-dB ensemble，复用旧44名validation预测/结果，报告
  LSD新减旧配对bootstrap（10,000次、seed20260904）及ERB/HF/ILD/尾部权衡。
- `configs/experiments/sonicom_hybrid_lsd_b_e190_preparation_manifest.json`冻结3个父模型、
  3个旧对照配置/checkpoint/完成报告及44个旧validation预测的SHA256；旧报告均E190完成、
  clean training Git、test_subjects_read=0。旧Hybrid LSD均值3.5489256190074472 dB。
- `reports/HYBRID_LSD_B_PREFLIGHT_20260904.json`记录41/41测试通过；CUDA单train被试P0002、
  16 global+16 horizontal前后向passed，total=1.108809232711792，采样LSD=3.5230913162231445 dB，
  output梯度norm=3.8066632747650146，骨干无梯度，峰值allocated=268505600 bytes。
  该smoke非validation结论，亦不代表完整run峰值。pytest缓存目录写权限警告不影响测试结果。
- 当前状态为已准备、尚未训练；启动前将本轮代码/配置/证据提交，require_clean_git=true。
  正式run启动和PID以后续交接及`outputs/hybrid_lsd_b_e190/launch.json`为准。
  新test_subjects_read=0；正式训练finite/best/完成性待运行后核验。禁止复用已消费test做独立确认。
  
  ## 2026-09-03：十方法完整20指标test评价完成

- 在用户明确授权后，按结果盲预注册协议
  `experiments/film_siren/STAGE_E_COMPLETE_TEN_METHOD_SECONDARY_DEFERRED_TEST_PROTOCOL.md`
  和manifest identity `E9403269543B6C135871BD8008809B488F2E29977601C95E3FF2BD491FDB229E`
  完成44名test被试的十方法补充评价。方法固定为SH only、SUpDEq SH、SUpDEq NN、
  SUpDEq Barycentric、MCA、MCAR v3.5.1、FSP-AE、RANF、Hybrid E190和Bounded E25；
  不训练、不更新参数、不改变预测，也不根据test调参或选型。
- 8种外部/经典方法的统一表示为352/352 HDF5，频谱shape均`[2,793,463]`、HRIR均
  `[793,2,256]`、split=`test`且finite；其4项既有primary逐被试复现最大误差`0 dB`。
  统一评价session `95875`自然结束exit 0，新9个secondary和7个deferred端点的
  per-subject/aggregate/paired行数为`7040/160/144`，band profile=`350`、spatial map=`7670`，
  全部numeric finite，`test_subject_count_read=44`。
- 随后只读合并既有4项primary，完整包的per-subject/aggregate/paired/wide行数为
  `8800/200/180/20`；每个method-endpoint恰有44名被试，primary与supplementary源值逐键最大误差
  均为`0`，聚合均值复核误差`0`。合并不重新读取test，`new_test_subject_count_read=0`。
- 19个误差/率端点中，18个有唯一最低均值：FSP-AE最低10项（Contra HF、Full LSD、四个空间LSD、
  两个dominant-notch位置误差、notch miss rate、ITD最大误差），Hybrid E190最低4项（Contra25、
  HF一/二阶差分、multi-scale notch depth），Bounded E25最低2项（Full ERB、ERB-band ILD），
  MCAR v3.5.1最低Horizontal ILD，SUpDEq NN最低ITD weighted MAE。notch spurious rate十方法均为0，
  并列第1；ReferenceNotchFraction十方法均为1，作为方法无关描述量不排名。因此不支持任何方法
  “全指标支配”的声明。
- 代表性均值：Bounded的Full ERB=`0.7941314445 dB`、ERB-band ILD=`1.5733966177 dB`；
  Hybrid的Contra25=`1.2003965313 dB`、HF一/二阶差分=`0.3997093568/0.2145803510 dB/bin[/bin^2]`、
  notch-depth=`0.4876985604 dB`；FSP-AE的Contra HF=`3.1643546865 dB`、Full LSD=
  `3.1202205122 dB`、notch miss=`0.3766895693`、ITD max=`123.6387278451 us`；MCAR v3.5.1的
  Horizontal ILD=`0.6447090917 dB`；SUpDEq NN的ITD weighted MAE=`15.1370557204 us`。
- 20张独立dot-and-whisker图均含10方法均值和10,000次subject-bootstrap 95% CI；20 PNG和20
  vector PDF非空，PNG约`4151--4245 × 2814--2815`，所有均值位于对应区间内，机器验证和四类
  代表图目视抽查均通过。证据：`results/sonicom_complete_ten_method_secondary_deferred_test_v1/`、
  `results/sonicom_complete_ten_method_test_v1/`、`outputs/ten_method_test_metric_figures_v1/`。
- 本批结果属于**post-lock supplementary test characterization**：test在项目历史中已经消费，
  所以它补齐同一test上的横向描述与配对统计，但不是新的未见独立确认，也不能用于训练、调参、
  模型晋升或改变Bounded E25/MCAR v3.5.1的既有角色。训练/best/末点预算判据=N/A。

## 2026-09-02：十方法Q14/Q26/Q50正式统一评价完成

- 正式MATLAB session `55629`自然结束，132/132个validation subject×direction-count单元完成，exit=`0`。
  统一口径为44名validation受试者、10方法、嵌套Q14/Q26/Q50、固定排除全部Q50输入方向后的743
  评价方向、四项dB误差、10,000次paired-listener bootstrap（seed `20260902`）；test不读。
- 表格完整性：`metric_long.csv`/`aggregate_metrics.csv`/`paired_direction_effects.csv`/
  `bounded_interactions.csv`/`quality_checks.csv`分别`5280/120/80/72/132`行；120个
  method×direction×metric组均恰有44名受试者。所有numeric列finite，summary为
  `status=completed`、`all_finite=true`、`test_subject_count_read=0`。NN/Bary输入点回代最大误差
  `4.2494e-15/4.8036e-15`，FSP频率误差0，RANF grid误差0 deg、观测HRIR误差`1.4857e-8`。
- Full-sphere ERB的Q14/Q26/Q50均值：SH-only `2.792926/2.658619/2.962315`；SUpDEq+SH
  `1.842031/1.815283/1.997039`；SUpDEq+NN `2.076842/1.836076/1.655922`；SUpDEq+Bary
  `2.010790/1.731476/1.508370`；MCA `1.396600/1.089779/1.007027`；MCAR v3.5.1
  `1.181744/0.826642/0.838240`；FSP-AE `2.108286/1.155485/2.036753`；RANF
  `1.102148/1.071196/1.028151`；Hybrid E190 `1.105695/0.800820/0.817103`；Bounded E25
  `1.111461/0.792813/0.810526 dB`。
- 相对Q26的四指标显著性模式（95% paired-bootstrap CI）：Q14时Bounded、MCAR、Hybrid、FSP-AE、
  MCA、SH-only均为4/4显著退化；RANF为2退化/2不显著；SUpDEq+SH为2退化/1改善/1不显著，
  NN与Bary均为3退化/1不显著。Q50时Bounded和FSP-AE为4/4显著退化，MCAR为3退化/1不显著，
  Hybrid为2退化/2不显著；RANF为2改善/2不显著，Bary为4改善，NN为3改善/1退化，MCA为
  3改善/1不显著。由此Q50小幅退化是冻结Q26学习模型的分布失配现象，不能概括为更多测量一般有害。
- Bounded E25仍是论文主模型和Q26正式操作点；工程主模型仍为MCAR v3.5.1。该补充实验不用于
  调参、模型选择或新增test访问。图`ten_method_direction_sensitivity.{png,pdf}`已目视通过；
  训练/末点预算判据=N/A。证据根：`results/sonicom_ten_method_direction_sensitivity_v1/`。

## 2026-09-02：十方法Q14/Q26/Q50正式统一评价已启动

- 在manifest identity `A317503131FDF3D1DAA73C13146C3BB9CCCBB39105B8D4717B97A0B118219539`
  已提交、tracked worktree clean的HEAD `d50b03a`上启动正式评价；MATLAB PID=`10048`（helper
  `39128`）、session=`55629`，输出根为`results/sonicom_ten_method_direction_sensitivity_v1/`。
- 固定范围为44名validation受试者、10方法、Q14/Q26/Q50，所有条件统一排除50个Q50输入方向并在
  743方向评价FullSphereERB、Contralateral25ERB、ContralateralHighFrequency、HorizontalILDMAE；
  统计为10,000次paired-listener bootstrap（seed `20260902`）。
- 启动已进入`[1/132] P0001 Q14`。本条仅表示正式run已启动，finite、完整行数、统计与图表均为
  PENDING；不得重复启动或把部分输出作为结论。`test_subject_count_read=0`，训练/末点预算判据=N/A。

## 2026-09-02：十方法统一评价器修复后manifest重新冻结

- 配置容器兼容修复已在1-subject smoke验证并提交为`3a1ec45`。随后在clean tracked worktree、
  RANF adapter clean commit `c50a1589c99654fb52d1a73ee17ac419b9a2654f`上运行原manifest生成脚本。
- 新manifest identity为`A317503131FDF3D1DAA73C13146C3BB9CCCBB39105B8D4717B97A0B118219539`；
  修复后MATLAB evaluator SHA-256为`D2B9DA8B2CB8FA24A975A0BA2EEDE95E5C88B73C26E98DD68037DB95F1A51E25`。
  其余方法、配置、grid、input manifest和checkpoint resource不变。
- manifest固定split=`val`、44 subjects、10 methods、Q14/Q26/Q50、统一743评价方向，
  `test_subject_count_read=0`。该manifest提交且工作树clean后方可启动44人正式评价。

## 2026-09-02：十方法统一评价1-subject smoke修复后通过

- 修复仅兼容MATLAB `jsondecode`对异构methods对象返回cell array的行为；方法、输入、方向网格、
  固定743方向mask、四项指标、配对效应和统计规则均未改变。`checkcode`为0条问题。
- 修复后的1-subject validation smoke自然完成，exit=`0`。`metric_long.csv`、
  `aggregate_metrics.csv`、`paired_direction_effects.csv`、`bounded_interactions.csv`、
  `quality_checks.csv`行数分别为`120/120/80/72/3`，summary为`status=completed`、
  `all_finite=true`，PNG与vector PDF均非空。
- 本步骤仅用于执行链路完整性，不接纳单被试数值作为科研结论；固定评价方向数=`743`、
  bootstrap=`10000`（seed `20260902`）、`test_subject_count_read=0`、训练/预算判据=N/A。
  正式44人评价前须先在clean HEAD重生成并提交hash-locked manifest。

## 2026-09-02：十方法统一评价首次1-subject smoke因配置解析失败，未接纳结果

- 命令在正式指标计算前于`matlab/+mcar/evaluate_ten_method_direction_sensitivity.m:49`退出，
  MATLAB exit=`1`。原因是配置中的十个方法对象字段异构，`jsondecode`返回cell array，而评价器
  误按struct array访问`config.methods.id`。
- `results/sonicom_ten_method_direction_sensitivity_smoke/`未创建，metric/aggregate/effect/
  interaction/quality均未产生，因此无数值被接纳、无任何模型或实验结论变化。
- 修复边界仅为cell/struct配置容器兼容，不改变已冻结方法、输入、方向网格、743方向mask、指标、
  bootstrap或选择规则；修复通过smoke后必须重新生成并提交hash-locked manifest，方可运行44人正式评价。
  本次未读取test，`test_subject_count_read=0`；训练/预算判据=N/A。

## 2026-09-02：十方法方向敏感度RANF Q14/Q50原生适配完成

- 顺序controller自然结束，exit=`0`、phase=`complete`，运行区间
  `2026-09-02T00:54:50+08:00`至`09:48:03+08:00`，总耗时`31,993 s`。Q14适配耗时
  `1:57:27`，Q50适配耗时`6:52:08`；两档均从同一冻结Q26-pretrained checkpoint
  `7b288f6f...3e367`重启，train-only retrieval=262人，1000 epochs、batch size 3，评价44名
  validation被试，`test_subject_count_read=0`。
- Q14/Q50各生成44个prediction SOFA，eval log均完整；best/adaptation/adaptation_loss三个checkpoint
  每档均含119个finite tensor。Q14/Q50 adaptation checkpoint SHA-256分别为
  `da92bd635217619416140016b35aa8012a8f8cc4251fc889535739e2b02d77e6`和
  `a30e124d9543917f56e8d09165dc10c55023cb952b25a83e9baaf29fd2d473af`。
- RANF原生汇总用于完整性诊断而非最终统一指标：Q14 ITD/ILD/LSD mean为
  `11.5074880883 us / 0.7750957795 dB / 3.4310708305 dB`，Q50为
  `9.6767661170 us / 0.6764719991 dB / 2.8810324954 dB`；两档所有被试均低于原生阈值。
- 下一步：把Q14/Q50 SOFA导出到根项目artifact布局，Q26复用既有validation RANF；完成44×3
  provenance检查后运行五种classical/MCA现算及十方法固定743方向统一评价。

## 2026-09-02：十方法方向敏感度RANF Q14→Q50原生适配已启动

- 在RANF adapter commit `c50a1589c99654fb52d1a73ee17ac419b9a2654f`和冻结预训练checkpoint
  SHA-256 `7b288f6f9198664e9ce75fee418e90ec3da2f5d0b7996a4edefe7d5f8da3e367`下，启动顺序
  controller：先Q14，再Q50。每档重新从同一checkpoint开始，retrieval bank仅用262名train被试，
  原生适配预算固定1000 epochs、batch size 3，随后评价44名validation被试；test不读。
- 启动确认：Windows后台WSL PID=`39500`，Linux controller PID=`305`，phase=`q14:adaptation`。
  controller根为`/home/ill3/ranf-work/exp/mcar_ten_method_direction_sensitivity/`；Q14/Q50实验根分别为
  `/home/ill3/ranf-work/exp/mcar_ten_method_direction_sensitivity_q14_validation/`和
  `..._q50_validation/`。控制器会在Q14自然完成后自动进入Q50。
- 本条仅表示已启动，不表示完成。训练/评价finite、adaptation checkpoint、1000-epoch末点和44/44输出
  均为PENDING；按Codex长任务约定不主动轮询。完成后需导出两档SOFA、验证Q26既有RANF复用，才可运行
  5280行十方法统一评价。`test_subject_count_read=0`。

## 2026-09-02：十方法方向敏感度三种冻结学习方法推理完成

- 在修复后manifest identity `B625A3D0E145FD906854BF87DEB2821D1C49552FC1DF04AE2CD10B364B3C098E`
  下完成MCAR v3.5.1、Hybrid E190、FSP-AE的44 validation×Q14/Q26/Q50产物。Q26中心档直接
  复用各方法既有formal validation artifact；Q14/Q50仅改变当前观测方向集合，不更新参数。
- 独立完整性扫描为396/396 HDF5：MCAR与Hybrid residual均为`[2,793,463]`；FSP-AE HRIR为
  `[793,2,256]`，其magnitude/ITD/frequency附属张量也全部finite。所有split=`val`、Q标签覆盖
  14/26/50，三种方法各132份。Q26最大绝对复现差均为`0`；两个completed report均记录
  `test_subject_count_read=0`。MCAR/Hybrid用时`84.2155316 s`，FSP-AE用时`332.0170047 s`。
- 本步骤只确认正式推理产物完整，不读取或汇总Q14/Q50指标；五种classical/MCA将在最终统一评价器中
  现算。RANF Q14/Q50仍需按原生1000-epoch、batch=3适配，Q26复用既有formal validation RANF。
- 证据：`artifacts/sparsity/sonicom_ten_method_direction_sensitivity_v1/`；
  `inference_hybrid_mcar_report.json`；`inference_fspae_report.json`。训练/best/末点预算判据=N/A。

## 2026-09-02：十方法方向敏感度首次冻结推理被MCAR Q26复现闸门拒绝

- 在manifest identity `41E1EE0752E563A68872F6469B2E591E832F383227A8CB66C2157F1BB5A55961`
  锁定后，首次运行MCAR v3.5.1与Hybrid E190的44 validation × Q14/Q26/Q50冻结推理。264个
  HDF5均完成且张量shape为`[2,793,463]`、finite；test读取为0。
- 正式接纳前的Q26逐元素复现闸门拒绝该run：从Bounded内部diagnostics导出的MCAR base相对既有
  formal MCAR v3.5.1最大绝对差为`0.10586357116699219 dB`，超过预注册`1e-5 dB`。Hybrid未触发
  Q26失败。根因范围已收敛为实现路径：Bounded内部base使用全精度forward，而既有formal MCAR
  两分量是固定64-direction block、CUDA AMP后再作0.3/0.7 output ensemble；这不是模型或超参数
  选择问题。
- 决策：该run状态为FAILED/REJECTED，不进入任何方法比较，不运行十方法评价；264个文件整体移入
  diagnostics保留。随后只允许作意图保持的复现修复：显式按正式MCAR两个冻结checkpoint、AMP=true、
  direction block=64、权重0.3/0.7运行，并重新生成依赖manifest。不得根据Q14/Q50结果改变方法。
- 完整性：prediction files=`264`；失败在报告写出前由gate抛出，因此无completed report；
  `test_subject_count_read=0`；训练/best/末点预算判据=N/A。

## 2026-09-01：Bounded E25 Q14/Q26/Q50 输入方向敏感性实验完成

- 在结果前冻结的validation-only协议下，使用论文主模型Bounded E25三成员等权ensemble
  （identity `72B7319F8334307664A02BC6369FBD9EECE1D22383C402EC9683D1F6F05F2705`）完成
  Q14/Q26/Q50三档实验。模型、gate和ensemble权重全部冻结，不重训；每档只基于当前Q观测重新计算
  MCA/correction并送入同一permutation-invariant condition encoder。三档严格嵌套，并统一排除完整
  Q50输入集合，只评价相同743方向，因此方向档之间可作44名被试内配对比较。
- 准备与推理完整性全部通过：132/132 current-Q cache、132/132 model input、132/132 prediction；
  预测shape均为`[2,793,463]`且全部finite。Q26 MCA、correction和冻结ensemble residual均对既有正式
  Q26入口逐元素复现，三者最大绝对差均为`0 dB`，通过预注册`1e-5 dB`门槛。hash-locked input
  inventory identity为`D55191DCD1859E80DDA77B8676324149463DAE9DE9CBCF31A79B84E13158D7BC`。
- 固定743方向口径下，Q14/Q26/Q50的FullSphere ERB均值分别为
  `1.1114611924 / 0.7928133567 / 0.8105263646 dB`；Contralateral25 ERB为
  `1.7266513069 / 1.2138574436 / 1.2720120439 dB`；Contralateral HF为
  `4.0523591552 / 3.4999635794 / 3.5339064948 dB`；Horizontal ILD为
  `0.7262354882 / 0.5591198738 / 0.6140608872 dB`。这里的Q26均值使用743共同方向，不能与主表
  767方向Q26均值直接混用。
- 固定10000次paired-listener percentile bootstrap（seed=`20260901`）显示，Q14相对Q26四项均
  显著变差：差值依次为`+0.3186478357`、`+0.5127938633`、`+0.5523955758`、
  `+0.1671156144 dB`，95% CI分别为`[+0.3013398095,+0.3362769453]`、
  `[+0.4760909555,+0.5508624901]`、`[+0.4846624611,+0.6226475607]`和
  `[+0.0995746024,+0.2350619852]`。Q50相对Q26也未改善，四项差值均为正且CI不跨0：
  `+0.0177130079 [+0.0107869169,+0.0239559429]`、
  `+0.0581546003 [+0.0437688555,+0.0724108812]`、
  `+0.0339429154 [+0.0169544766,+0.0519414385]`、
  `+0.0549410134 [+0.0203934510,+0.0900073993] dB`。
- 决策：论文主配置正式写为**Bounded E25 + Q26输入**。Q14证实方向进一步稀疏会造成稳定退化；
  Q50结果说明对一个在Q26条件分布上训练并冻结的模型，直接增加观测方向不产生单调收益，反而有小幅
  失配代价。这不证明“更多测量一般有害”，也不能外推到重新训练的Q50模型。正式结果位于
  `results/sonicom_bounded_e25_input_direction_sensitivity_v1/`，含528行metric long、12行aggregate、
  8行paired effect、132行quality、summary及PNG/vector PDF；全部finite，固定评价方向数743，
  `test_subject_count_read=0`，训练/末点预算判据=N/A。

## 2026-09-01：Bounded E25正式固定为论文主模型，输入方向敏感性实验预注册

- 用户明确将formal Bounded E25设为论文主模型。规范模型保持identity
  `72B7319F8334307664A02BC6369FBD9EECE1D22383C402EC9683D1F6F05F2705`：三个seed
  `20260821/22/23`的cycle25 `last.pt`，residual-dB权重固定为`1/3`。该决策不改变当前工程部署
  主模型MCAR v3.5.1，也不改写Bounded已冻结test的已知trade-off：宽带/对侧ERB占优，高频细节
  弱于FSP-AE，test ILD与MCAR v3.5.1统计持平，故不声明全指标支配。
- 为补充输入稀疏度证据，在任何新增预测或指标前冻结validation-only sensitivity协议。中心为原Q26；
  更稀疏档固定为历史嵌套Q14；更稠密档固定为Q50，即保留全部Q26并按与原Q26一致的“最小大圆距离
  优先、order-3 SH Gram logdet次级、索引最终打破并列”规则新增12对左右镜像实测方向。三档严格满足
  `Q14 ⊂ Q26 ⊂ Q50`，只使用44名validation被试，`test_subject_count_read=0`。
- Q14/Q26/Q50最小点间角为`39.36699615882736/31.915838841829665/20.53677810221674°`；Q50
  mean/p95/max覆盖距离为`11.0677284280/17.9638601298/27.3447980931°`，order-3 SH设计矩阵
  rank=`16`、condition=`2.4482635509`。网格只读取坐标，不读取HRTF值或结果。
- 每一档均不重训：只用当前Q观测重新计算order-3 MCA与correction，并把同一Q的观测幅度/坐标送入
  冻结的permutation-invariant condition encoder；FiLM、两个MCAR分量、bounded gate和ensemble权重
  均不更新。Q26新入口必须以最大绝对差`<=1e-5 dB`复现既有formal validation residual后，Q14/Q50
  才可接纳。
- 三档统一排除完整Q50输入集合，只评价相同743方向。端点仍为FullSphere ERB、Contra25 ERB、
  Contra HF和Horizontal ILD；预注册对比为`Q14-Q26`与`Q50-Q26`，固定10000次paired-listener
  percentile bootstrap、seed=`20260901`。协议、配置、网格和实现见
  `experiments/film_siren/STAGE_E_BOUNDED_E25_DIRECTION_SENSITIVITY_PROTOCOL.md`、
  `configs/experiments/sonicom_bounded_e25_input_direction_sensitivity_v1.json`及
  `configs/data/sonicom_nested_sparse_grid_q14_q26_q50_v1.json`。本条仅为模型决策与结果前冻结，
  训练/末点预算判据=N/A；44×3重建、推理、finite与严格指标均尚未运行。

## 2026-09-01：十方法 24 个指标逐指标论文级可视化完成

- 以已提交的十方法v2逐被试CSV为唯一数据源，为4个primary validation、4个frozen engineering
  test、9个secondary validation和7个deferred validation指标分别生成一张独立对比图，共24张。
  每张图完整包含10种登记方法和每方法44名被试；点为subject mean，横向whisker为固定10,000次
  subject-bootstrap 95% CI（base seed=`20260901`），除`ReferenceNotchFraction`外均按均值从低到高
  排序。比例指标转换为百分比显示，`ReferenceNotchFraction`明确标记为method-invariant描述量；
  全方法均值完全相同的spurious-rate图标记为tie，不虚构单一赢家。
- 绘图数据位于`results/sonicom_complete_ten_method_metric_figure_data_v1/`，包含240行
  method-metric summary、24行figure index和完整性摘要；全部数值finite，所有mean位于各自bootstrap
  区间内。该步骤只读取已提交结果CSV，不运行模型、不读取原始HRTF/test数据，新增
  `test_subject_count_read=0`；上游唯一冻结test仍为44人。
- 图件位于`outputs/ten_method_metric_figures_v1/`：`24/24`张300-dpi PNG和`24/24`张vector PDF，
  PNG分辨率均不低于`3573×2391`、全部文件非空、PDF头有效。首次渲染后抽检发现接近100%的
  `ReferenceNotchFraction`通用留白超出语义范围，以及全均值并列图不应标单一best；已在版式层修正
  并从头重导，未改任何数据或统计。最终批次代表性抽检覆盖primary、secondary、百分比、描述量和
  ITD端点，布局、标签、CI和排序均通过。
- 完整交付另打包为`outputs/ten_method_metric_figures_v1.zip`，53个archive entries，SHA-256=
  `2C4354F7B2C704C03195FD4A17B2B295E9685C07811CBB20E7F95BF4A73CA1EF`。训练、推理和预算判据
  均为N/A；四个证据层继续保持分离，不据这些图生成跨层总排名。

## 2026-08-31：五传统基线 secondary/deferred validation 补齐与十方法完整表 v2 完成

- 按用户要求补齐`SHOnly`、`SUpDEqSH`、`SUpDEqNN`、`SUpDEqBary`和`MCA`。新增结果前先以
  commit `bc871ea`冻结validation-only amendment、独立MATLAB导出器和Python评价器；导出器
  硬拒绝非`val` split及`allowTest=true`。44人×5方法共导出`220/220`个HDF5，Python可见
  频谱shape=`[2,793,463]`、HRIR shape=`[793,2,256]`，全部split=`val`且finite。用相同表示
  回算既有四个primary validation指标，`880`个subject-method-endpoint单元与历史表最大绝对差
  为`0 dB`，证明导出没有改变既有重建口径。
- prediction inventory和所有评价依赖随后冻结于commit `0cfaa4e`，manifest identity=
  `E088B3F0748B3FB2F4772D5ED25D7CC2B16B04E2E2EFCDB65B91B453E008ABC8`；只有在该提交之后才
  首次计算新增指标。正式结果位于
  `results/sonicom_film_secondary_classical_baseline_extension_v1_validation/`：5个scalar
  secondary得到`1100`行per-subject、7个deferred得到`1540`行per-subject，另有35-band ILD、
  4个空间距离bin及767方向map；quality=`passed`、全部finite、频率最大差`0 Hz`、
  `test_subject_count_read=0`。
- 五传统基线中，MCA在FullSphereLSD=`4.7977744230 dB`和ERBBandILDMean=`1.9954270829 dB`
  最优；SUpDEq NN在MultiScaleNotchDepthMAE=`0.7451926158 dB`、dominant-notch penalized MAE=
  `1018.3931737286 Hz`及ITD weighted MAE=`12.6764186527 us`最优；SUpDEq Bary在HF一阶差分=
  `0.5831033472 dB/bin`略优于NN，并在FullSphereLSD=`5.7320541973 dB`优于其它SUpDEq变体。
  `ReferenceNotchFraction`仍是描述量，不用于优劣声明。
- 十方法完整数据v2位于`results/sonicom_complete_ten_method_comparison_v2/`，并同时纳入用户已完成
  的Hybrid E190唯一冻结test。primary validation、frozen engineering test、secondary validation、
  deferred validation现均覆盖`10/10`方法；方法×指标表`240/240`格均有44-subject finite数值，
  Bounded-vs-all配对表由128行扩至`207`行，band profile=`350`行、spatial map=`7670`行。汇总
  步骤没有读取新test数据，`new_test_subject_count_read=0`；上游唯一冻结test读数仍为44。
- 更新工作簿为
  `outputs/stage_e_complete_ten_method_comparison_v2/stage_e_complete_ten_method_comparison_v2.xlsx`，
  SHA-256=`D6168669F01F594D2DB5B30BCEC41B333CDBF9B7B6B198E6868A6FF2FA8EC683`。11个sheet全部
  渲染核验；Complete Table=`25×16`、All Means=`241×15`、Paired=`208×18`、Band ILD=
  `351×7`、Spatial Map=`7671×8`，overview检查=`[10,240,240,0]`，公式错误0。机制、效率与
  localization的结构性不可比状态保持不变，未作数值插补。训练/末点预算判据=N/A。

## 2026-08-31：Hybrid E190 唯一一次冻结 test 推理与十方法严格评价完成

- 经仓库证据确认，FiLM-SIREN + zero-init MCAR spectral CNN 的三成员 Hybrid E190 此前从未读取
  SONICOM test。依据用户本次明确授权，先冻结协议、三个 cycle190 `last.pt`、等权 `1/3` ensemble、
  44 名 test 被试、四个主端点及 10000 次 paired bootstrap，再执行首次且唯一一次推理。manifest
  identity=`CE01EE0D9F6DE2A454251CC25474FB371C68FEA0D33F3ADA8FA41019DFFBA1FB`；registry
  已原子锁定为 `completed`，`test_subject_count_read=44`，禁止覆盖或再次推理。
- 预测产物位于`artifacts/reconstruction/sonicom_film_siren_spectral_cnn_final_e190_ensemble_test/`：
  `44/44` HDF5，逐文件 shape=`[2,793,463]`、split=`test`、manifest identity一致且全部finite。
  严格原始评价与十方法合并分别位于
  `results/sonicom_film_siren_spectral_cnn_final_e190_frozen_test_raw/`和
  `results/sonicom_film_siren_spectral_cnn_final_e190_frozen_test_ten_method/`。最终长表为
  `44 subjects × 10 methods × 4 metrics = 1760`行、全部finite；重复计算的共享基线与历史冻结
  test表最大绝对差为`0 dB`；合并步骤新增test读取为`0`。
- Hybrid E190 的 test 均值依次为：FullSphere ERB=`0.8175440243 dB`、Contralateral-25 ERB=
  `1.2003965313 dB`、Contralateral HF=`3.4549112589 dB`、Horizontal ILD=`0.7794497051 dB`。
  对应四项赢家为Bounded E25=`0.7941314445`、Hybrid E190=`1.2003965313`、FSP-AE=
  `3.1643546865`、MCAR v3.5.1=`0.6447090917 dB`，因此Hybrid不是四指标整体支配解。
- 固定10000次subject-paired bootstrap（seed=`20260831`）显示：相对MCAR v3.5.1，Hybrid在
  Contra25与HF显著更优（差`-0.0740927008`，95% CI `[-0.1119315495,-0.0458923772]`；
  差`-0.0535697039`，CI `[-0.0890215416,-0.0223871616]`），Full持平（差
  `-0.0003930245`，CI `[-0.0309952717,+0.0554647708]`），ILD显著更差（差
  `+0.1347406134`，CI `[+0.0127054749,+0.3448492520]`）。相对Bounded，Full和ILD显著更差、
  Contra25持平、HF小幅显著更优。
- Hybrid ILD均值受单一尾部被试P0339=`6.0496669795 dB`明显牵引；中位数为
  `0.6340056391 dB`，次高仅P0234=`1.1779 dB`。该点按冻结协议保留，不做test驱动剔除、
  修补或重测。训练完整性继承已冻结E190三成员证据；本步骤训练/末点预算判据=N/A。

## 2026-08-30：完整十方法横向对比集合登记与分层数据表完成

- 按用户指定顺序将完整论文横向集合冻结为：SH only、SUpDEq SH、SUpDEq NN、
  SUpDEq Barycentric、MCA、MCAR v3.5.1、FSP-AE、RANF、Hybrid E190、Bounded E25。
  机器可读注册表为`configs/experiments/sonicom_complete_horizontal_comparison_methods_v1.json`，
  汇总协议为`experiments/film_siren/STAGE_E_COMPLETE_TEN_METHOD_COMPARISON_PROTOCOL.md`。
  这是一项既有冻结结果的consolidation，不是新的result-blind模型选择：未启动训练/评价，未读取
  原始test HDF5/SOFA，新增`test_subject_count_read=0`；历史frozen engineering test上游读数仍为44。
- 数据产物位于`results/sonicom_complete_ten_method_comparison_v1/`。主validation覆盖`10/10`
  方法×4端点，historical frozen test覆盖`9/10`×4端点（Hybrid E190未进入冻结test方法集），
  secondary/deferred validation各覆盖有同协议冻结产物的`5/10`方法（MCAR v3.5.1、FSP-AE、
  RANF、Hybrid E190、Bounded E25）。完整method-endpoint状态表240格，其中156格有44-subject
  均值/样本标准差且全部finite；其余格显式使用`NOT RUN`，不作插补。Bounded-vs-all可用方法
  的paired table共128行，固定10000次PCG64 subject bootstrap；另含175行band ILD profile、
  3835行spatial map、615行Bounded机制汇总、44行correction-benefit Spearman和2项部署效率。
- 单张论文宽表为`paper_complete_comparison_wide.csv`。四个primary validation赢家依次为：
  FullSphere ERB Bounded E25=`0.7966490938 dB`；Contralateral-25 ERB Hybrid E190=
  `1.2145930843 dB`（Bounded=`1.2166261334`，差`+0.0020330491 dB`、95% bootstrap CI
  `[-0.0084009848,+0.0124296471]`）；Contralateral HF FSP-AE=`3.1038093697 dB`；
  Horizontal ILD Bounded E25=`0.5650044448 dB`。故Bounded在宽带Full与ILD最优、Contra25与
  Hybrid统计持平，但不支配HF细节。
- historical frozen test四项赢家依次为Bounded E25 Full=`0.7941314445 dB`、Bounded E25
  Contra25=`1.2015177313 dB`、FSP-AE HF=`3.1643546865 dB`、MCAR v3.5.1 ILD=
  `0.6447090917 dB`。Hybrid E190在test列严格标记`NOT RUN`，没有因本次横向汇总事后补测。
  validation supplementary的9项中，FSP-AE在FullSphereLSD与四个spatial-LSD bin最优，
  Hybrid E190在HF一/二阶差分与multi-scale notch-depth最优，Bounded E25在ERB-band ILD最优；
  deferred notch/ITD仅作exploratory解释，`ReferenceNotchFraction`为描述量且不排名。
- 工作簿`outputs/stage_e_complete_ten_method_comparison/stage_e_complete_ten_method_comparison.xlsx`
  SHA-256=`5229C695EA37C792C47ADD4EAB6A614EC7D7E1F851A50C244FED77EF28961EB1`。含11个sheet：
  Overview、Method Registry、Complete Table、All Means、Paired vs Bounded、Availability、
  Band ILD、Spatial Map、Efficiency、Mechanism、Correction Correlation。逐sheet渲染核验，
  结构检查为registry 10方法、Complete Table `25×16`、All Means `241×15`、paired
  `129×18`、Spatial Map `3836×8`等，公式错误扫描0。
- 可比性边界保持冻结：经典五基线缺少同协议secondary/deferred重建产物，标记`NOT RUN`；
  除Bounded外没有共同gate/correction内部量，标记`NOT APPLICABLE`；其余方法没有冻结同硬件
  benchmark，效率标记`NOT AVAILABLE`；model-based localization因官方immutable依赖缺失，
  全部标记`NOT RUN`。不同split与证据等级不得合并排名或平均。

## 2026-08-30：Stage E 对 MCAR/RANF/FSP-AE 分层横向对比完成

- 在RANF/FSP-AE validation-only新增指标扩展完成后，将Stage E正式Bounded E25 ensemble与
  MCAR v3.5.1、RANF、FSP-AE做分层横向汇总。主证据固定使用已提交的44-subject frozen
  engineering test四指标；secondary和deferred端点仅使用44-subject validation。汇总过程只读
  已冻结CSV，不读取数据HDF5，不重跑模型或评价，新增`test_subject_count_read=0`；上游候选唯一
  test inference读数仍为44。
- 产物：`results/sonicom_stage_e_external_comparison/`，包含45行candidate-vs-baseline长表、
  64行四方法均值表、分层scorecard、可比性状态、论文主表和英文Markdown报告；全部numeric
  finite。另生成可筛选工作簿`outputs/stage_e_external_comparison/
  stage_e_external_comparison.xlsx`，SHA-256=`68A85681DA6047AE488EAC6FA46B5076BFA25CD0A27A9803BF652BC2A93DC380`；
  5个sheet逐一渲染核验、公式错误扫描0，Summary含12项primary配对比较与论文结论。
- frozen test主结论：Stage E相对MCAR的Full/Contra25/HF均显著更好，差值
  `-0.0238056/-0.0729715/-0.0319009 dB`，horizontal ILD差`+0.0296724 dB`但CI跨0；
  相对RANF，Full/Contra25显著更好，HF/ILD持平；相对FSP-AE，Full/Contra25显著更好，
  HF显著更差`+0.312225 dB`，ILD持平。Stage E在四方法Full与Contra25均排第1、ILD第2、
  HF第3。
- validation supplementary显示：相对MCAR，Stage E在LSD、HF一/二阶差分和ERB-band ILD
  显著更好，仅multi-scale notch-depth显著更差；相对RANF/FSP-AE，Stage E在plain LSD与HF
  谱差显著更差，但ERB-band ILD显著更好。ITD weighted MAE与MCAR/FSP-AE持平、显著优于
  RANF；exploratory dominant-notch location则显著弱于三者。故论文必须把“宽带听觉加权
  ERB优势”和“细粒度HF谱形/notch定位代价”并列陈述，不能概括为全指标支配。
- 可比性边界：RANF/FSP-AE无共同gate/correction内部量，机制标记`NOT APPLICABLE`；三基线
  未按冻结RTX 5060协议重做效率，标记`NOT AVAILABLE`；model-based localization仍因官方
  immutable依赖缺失标记`NOT RUN`。报告生成曾有两次empty prewrite和三次partial格式兼容失败，
  均未接纳；完整失败目录移至ignored `artifacts/report_generation_failures/`保留审计，最终包从头
  生成并通过完整性核验。

## 2026-08-30：RANF/FSP-AE新增指标横向扩展完成（validation only）

- 用户要求补齐RANF、FSP-AE横向对比后，先提交comparator-specific result-blind amendment与
  实现`b155556`，再在clean Git上冻结manifest并提交为`19017aa`。manifest identity为
  `63B5C5D61C32949A70A767B9CAA9A85BDDF869DD2817756A23FF5A83979B0A35`。端点公式及既有四方法
  secondary结果在本扩展前已知，且RANF/FSP-AE四个primary结果历史上已知；但本条新增的LSD、
  HF谱差、notch-depth、band ILD、空间bins、ITD与dominant-notch comparator outputs在freeze前
  未计算、汇总或查看。本扩展不能改变已冻结候选或授权test。
- 输入映射固定为RANF `Data.IR [793,2,256]`做1024点FFT后按processed strict selected bins取值；
  FSP-AE直接使用`predicted_magnitude_db [793,2,512]`且processed bin `i`映射到FSP bin `i-1`；
  ITD均直接使用各自`[793,2,256]` HRIR。44 RANF/FSP-AE/refererence inventories及代码/runtime均
  哈希守卫。RANF方向最大误差`0 deg`、FSP频点最大误差`0 Hz`；相关测试`12 passed, 1 skipped`。
- 产物：`results/sonicom_film_secondary_baseline_extension_ranf_fsp_v1_validation/`。RANF/FSP-AE
  各44 subjects；per-subject/aggregate secondary rows=`440/10`，band rows=`3150`，spatial
  bins/map=`360/1534`，per-subject/aggregate deferred rows=`616/14`，paired secondary/primary
  与deferred rows=`26/12`；全部finite、无partial目录、`test_subject_count_read=0`。
- 五个secondary scalar均值（顺序RANF/FSP-AE）：FullSphereLSD=`3.3614502403/3.0700346219 dB`，
  HF first=`0.4085859819/0.4105746265 dB/bin`，HF second=`0.2194569700/0.2188125815 dB/bin²`，
  multi-scale notch depth=`0.4913962991/0.5033549626 dB`，ERB-band ILD mean=
  `1.7228081524/1.5991126461 dB`。BOUNDED减RANF/FSP-AE的均值差分别为：LSD
  `+0.2259944914/+0.5174101098`、HF first `+0.0186709017/+0.0166822570`、HF second
  `+0.0200060701/+0.0206504586`、notch depth `+0.0156038058/+0.0036451424`（后者95% CI
  `[-0.0002420884,+0.0077829808]`跨0）、band ILD `-0.2309854085/-0.1072899022`，其余上述CI均
  不跨0。故RANF/FSP-AE在全频LSD和HF谱形上明显优于BOUNDED，而BOUNDED的band ILD更好；
  notch-depth对RANF更差、对FSP-AE统计持平。
- 作为同一tail-risk表的已提交primary rows，BOUNDED减RANF/FSP-AE为：FullSphereERB
  `-0.2779629989/-0.3636615811 dB`，Contra25ERB `-0.4108726125/-0.6312911442 dB`，
  HF `+0.0418051579/+0.4060050018 dB`，Horizontal ILD `-0.2251626784/-0.0327038128 dB`；
  除BOUNDED-vs-FSP-AE ILD的CI`[-0.0734252727,+0.0082334221]`跨0外，其余CI不跨0。因此原四个
  auditory primary端点与新增plain-LSD/谱导数端点给出的优劣方向不同，论文必须分开解释。
- deferred均值（RANF/FSP-AE）：ITD weighted MAE=`17.589353/15.543916 us`，maximum=
  `155.598963/104.403413 us`；BOUNDED差为weighted `-2.068515/-0.023078 us`（对FSP-AE CI
  `[-1.077635,+0.949463]`跨0），maximum `-25.390629/+25.804921 us`。Exploratory dominant-notch
  penalized MAE=`859.001/829.641 Hz`，均优于BOUNDED `962.414 Hz`；BOUNDED差
  `+103.413/+132.772 Hz`且CI不跨0。机制内部量对RANF/FSP-AE定义性不可比，标记
  `NOT APPLICABLE`；未做同RTX 5060协议效率benchmark，标记`NOT AVAILABLE`；localization继续
  因官方AMT/SAM依赖缺失标记`NOT RUN`，这些状态不是零值。

## 2026-08-30：FiLM deferred secondary endpoints完成（定位模型除外）

- 在结果前提交核心/入口/amendment链`53b7ed2`、`99298cf`、`58462a9`，首次manifest
  `1cff90c`运行时于P0001机制诊断前因predictor方法归属错误安全中止；两个partial目录为空，
  没有写出或查看端点值。最小修正`e6f33b4`后从clean Git重新冻结manifest `ff6782c`，最终
  identity为`63328B94B930D6A18269FF0D17C33647BD08488292832A7D0B80175989B05066`。
  15项synthetic tests通过；全流程仅访问44名validation被试，`test_subject_count_read=0`。
- 产物：小型表格位于`results/sonicom_film_deferred_secondary_metrics_v1_validation/`；44份
  compressed gate/correction全点诊断位于`artifacts/reconstruction/
  sonicom_bounded_mcar_film_gate_diagnostics_v1_validation/`（约743.6 MB，不进入Git）。
  endpoint/mechanism-subject/mechanism-aggregate/Spearman rows=`1232/27060/615/44`，全部
  finite；每份diagnostic的member gate/member correction/ensemble correction shape分别为
  `[3,2,793,463]/[3,2,793,463]/[2,793,463]`。重新推导的Bounded ensemble residual与既有
  prediction最大绝对差`1.9073486328125e-6 dB`，低于冻结容差`5e-5 dB`。
- ITD使用官方FSP-AE兼容流程：1.6 kHz low-pass、44.1→384 kHz resampling、±1 ms raw
  cross-correlation argmax且不做额外插值；reference为measured SOFA Data.IR，候选为继承MCA
  phase/outside bins的严格HRIR。BOUNDED/Hybrid/FiLMENS/MCAR的weighted MAE为
  `15.520837/15.522398/15.543201/15.520266 us`；maximum absolute error均值为
  `130.208334/130.208335/130.326708/129.971590 us`。BOUNDED相对MCAR的weighted MAE差
  `+0.000571 us`，95% CI`[-0.025863,+0.027849]`；maximum差`+0.236743 us`，CI
  `[-0.355113,+0.887785]`。两项均统计持平，符合magnitude-only、继承相位的sanity定位。
- exploratory dominant-notch按endpoint-specific结果盲规则冻结为：interpolation DTF、
  Savitzky–Golay 11 bins/degree3、4--18 kHz、prominence>=1 dB、separation>=500 Hz、只取
  dominant notch、1500 Hz匹配/缺失惩罚。BOUNDED/Hybrid/FiLMENS/MCAR penalized MAE为
  `962.414/935.677/1024.122/949.249 Hz`；BOUNDED相对MCAR差`+13.164 Hz`，CI
  `[+6.877,+19.613]`，miss-rate差`+0.009435`，CI`[+0.004957,+0.013902]`。因此该探索性
  口径下Bounded略差于MCAR，不能用来宣称notch-location改善；因相关notch-depth结果在定义前
  已知，该端点明确不升级为confirmatory证据。
- 机制结果显示gate大面积饱和：三个成员subject-level mean `|g|`均值为
  `0.496210/0.496950/0.497559`，`|g|>=0.45`比例为`0.982841/0.986554/0.988253`，而
  `|g|<0.01`仅`0.000236/0.000105/0.000076`。deployed ensemble mean/median/P95/P99
  `|C|`为`0.626335/0.435213/1.866539/2.750039 dB`；每被试`|C|`与pointwise benefit的
  Spearman rho中位数`0.055844`，IQR`[0.012436,0.087927]`。解释为bounded gate多数点贴近
  最大mix边界，但correction幅度仍由FiLM-minus-base差控制；弱正相关仅是机制描述。
- 冻结效率环境为RTX 5060、driver 595.79、torch 2.8.0+cu128、float32、TF32关闭、block32、
  P0001完整793×463重建。5 warmup+30 synchronized repeats：单成员trainable/total/unique
  parameters=`514/1,794,250/1,794,250`，checkpoint `6.927 MiB`，FLOPs/MACs=
  `8.38301924032e11/4.19150962016e11`，median/P95=`322.466/331.902 ms`；三成员顺序
  ensemble=`1,542/5,382,750/4,684,242` parameters，checkpoint总计`20.782 MiB`，
  FLOPs/MACs=`2.514905772096e12/1.257452886048e12`，median/P95=`986.562/1017.960 ms`，
  end-to-end含HDF5 I/O=`1094.793 ms`。峰值CUDA allocated/reserved均为`88.841/108 MiB`。
  训练provenance：单成员elapsed `1166.247 s`、`0.323958 GPU-h`；三成员并行wall-clock按
  最慢成员`1166.247 s`，累计`0.959975 GPU-h`。
- model-based localization没有运行：本地仅有SUpDEq，没有可哈希冻结的AMT/SAM官方模型。
  不以自造proxy替代；后续若补充，必须另行提交官方包/commit、坐标测试、Monte Carlo设置、
  seed和reference-self阈值后才能生成输出。该阻塞不影响本条已完成的gate、ITD、notch与效率。

## 2026-08-30：FiLM secondary metrics validation tranche完成

- 按预注册协议与结果前 implementation amendment（manifest identity
  `8F61017E...3D520E`）运行一次 validation-only secondary evaluator。四种方法固定为
  BOUNDED E25、Hybrid E190、FiLM E130 和 MCAR v3.5.1；没有新增训练、test读取或模型
  选择。实现提交链为`90ab7d0`、`afdf10b`、`6d1b47b`、`6abe3a7`，冻结/重冻结文档与
  manifest提交`5adc21e`、`70238c0`。
- 产物：`results/sonicom_film_secondary_metrics_v1_validation/`。44/44 subjects、
  4 methods，`per_subject_metrics.csv`=`880` rows，`aggregate_metrics.csv`=`20` rows，
  `paired_tail_risk.csv`=`39` rows，band profile=`6300` rows，spatial bins=`720` rows，
  direction map=`3068` rows，per-subject ear/direction LSD=`279136` rows；全部 finite。
  quality checks为793 directions、26 Q26、767 interpolation、72 horizontal、463 frequency
  bins、35个200--18000 Hz ERB bands，`test_subject_count_read=0`。
- secondary均值（dB，越低越好；HF一阶单位dB/bin，二阶单位dB/bin²；顺序
  BOUNDED/Hybrid/FiLM E130/MCAR）：Full-sphere LSD=`3.5874447317/3.5489256190/
  3.7857646944/3.6097610497`；HF first=`0.4272568835/0.4027026539/0.4653138654/
  0.4295869212`；HF second=`0.2394630401/0.2167975699/0.2704213827/0.2538098598`；
  multi-scale notch depth=`0.5070001049/0.4884118526/0.5490026433/0.5001038916`；
  ERB-band ILD mean=`1.4918227440/1.4992052533/1.5738732923/1.5625394014`。
- BOUNDED相对MCAR的paired均值差（负值较好；10000次PCG64 bootstrap，seed
  `20260829`）为：LSD=`-0.0223163180`，95% CI`[-0.0372003017,-0.0072102922]`，
  wins/losses=`31/13`；HF first=`-0.0023300377`，CI`[-0.0037384654,-0.0006473509]`，
  `34/10`；HF second=`-0.0143468197`，CI`[-0.0162102093,-0.0125566868]`，`43/1`；
  notch depth=`+0.0068962134`，CI`[+0.0054273908,+0.0086348736]`，`1/43`；band ILD
  `-0.0707166574`，CI`[-0.0850487964,-0.0563517803]`，`40/4`。因此补充证据显示
  BOUNDED改善LSD、两项HF谱形和band ILD，但notch-depth略差；不能宣称secondary全面
  支配MCAR。
- 相对Hybrid，BOUNDED的LSD差`+0.0385191127 dB`（CI`[+0.0243161743,+0.05228911299]`）、
  HF first/second分别`+0.0245542296/+0.0226654702`且CI均不跨0、notch depth
  `+0.0185882524`且CI不跨0；band ILD差`-0.0073825094 dB`且CI跨0。空间LSD随Q26距离
  分层的BOUNDED均值为`3.4486082680/3.5643975513/3.7588286006/3.7685922991 dB`
  （0--10/10--20/20--30/>=30 deg），对MCAR差为`-0.0013011441/-0.0174319943/
  -0.0436498371/-0.1434810683 dB`，其中后三区CI不跨0、最近邻bin CI跨0。
- 图形仅读取已完成CSV生成：`figures/secondary_relative_to_mcar.{png,pdf}` 与
  `figures/spatial_lsd_by_q26_distance.{png,pdf}`，MATLAB R2025b静态检查通过并视觉核验。
  首次沙箱启动遇已知`File system inconsistency`，升级本机环境后成功导出；无不完整图形
  被采用。定位模型、ITD、gate/correction、效率和离散notch-location仍按协议deferred，
  不得从本条结果外推这些端点。

## 2026-08-29：Stage E bounded E25 frozen test九方法评价完成

- 用户授权后，先提交结果前协议/工具`ace2bd0`、Python兼容修正`1c04627`，再于
  clean Git生成并提交test manifest/registry `8c0ea47`。test wrapper identity为
  `9F9A82D48D3D35BF63CEFCA3CB0998C11473F298A407233B46E6555D29CBDF61`，
  三个成员仍严格对应validation冻结identity `72B7319F...F2705`，只使用三个cycle25
  `last.pt`的1/3 residual-dB均值。
- 唯一候选test推理44/44完成；每份prediction为`[2,793,463]`、finite、split=test、
  identity一致。registry=`completed`，`test_subject_count_read=44`。没有生成Hybrid或
  FiLM ensemble的新test预测，也没有test后调权、选成员、重训或重测。
- 严格test均值（Bounded / MCAR v3.5.1 / RANF / FSP-AE，dB，越低越好）：Full ERB=
  `0.7941314445/0.8179370488/1.0632355612/1.1844928283`；Contra25 ERB=
  `1.2015177313/1.2744892321/1.5573225324/1.9088722833`；Contra HF=
  `3.4765800998/3.5084809629/3.4696827665/3.1643546865`；horizontal ILD=
  `0.6743815182/0.6447090917/0.7754627714/0.7586678875`。
- 相对MCAR，Bounded的Full/Contra25/HF差分别为`-0.0238056043/-0.0729715008/
  -0.0319008630 dB`，95% CI分别为`[-0.0362774928,-0.0016266649]`、
  `[-0.0905437052,-0.0594867716]`、`[-0.0470893796,-0.0170815437]`，胜场
  `42/44,44/44,34/44`，三项均显著更好。ILD差`+0.0296724266 dB`，CI
  `[-0.0248789107,+0.1210173598]`，Bounded胜`25/44`，统计上未发现差异，但不能宣称
  均值更优。相对RANF，Full/Contra25显著更好，HF与ILD统计持平；相对FSP-AE，
  Full/Contra25显著更好、HF显著更差、ILD持平；相对MCA前三项显著更好、ILD持平。
- 完整性：最终九方法表`1584=44×9×4`行、全部metric finite；raw评价44份quality
  rows，767 interpolation directions；本次重算的七个共享冻结基线与既有八方法表逐项
  最大差`0 dB`，既有表SHA-256仍为`3D74AF13...D63`；paired bootstrap 10000次、
  seed `20260829`。这是项目已历史消费test上的锁定工程评价，不包装为全项目从未看过的
  独立确认，但本候选在本次访问前已由validation冻结，结果无论方向均最终报告。

## 2026-08-29：Stage E bounded E25唯一候选test评价已授权并预注册

- 用户明确授权`可以先做test评价`；授权记录时间为
  `2026-08-29T23:31:38+08:00`。本次只允许冻结identity
  `72B7319F...F2705`的三成员cycle25等权ensemble读取44名test被试一次，不允许
  调权、成员/周期选择、重训或根据test结果追加候选。
- 最终九方法表固定为Bounded候选加已冻结八方法表；paired bootstrap固定比较
  MCAR v3.5.1、RANF、FSP-AE和MCA，四指标、10000次、seed `20260829`。结果无论
  优劣均为最终可报告结果。
- 本条登记时仅完成协议与工具准备，`test_subject_count_read=0`；须在协议/工具提交且
  manifest/registry于clean Git生成并再次提交后，才可开始唯一test推理。

## 2026-08-29：Stage E formal E25 bounded-correction ensemble严格validation完成

- manifest identity `72B7319F8334307664A02BC6369FBD9EECE1D22383C402EC9683D1F6F05F2705`
  锁定三个cycle25 `last.pt`和`1/3,1/3,1/3` residual-dB平均。44/44 validation
  predictions均为`[2,793,463]`、finite、split=`val`且identity一致；
  `test_subject_count_read=0`。
- 严格指标均值（Bounded E25 ensemble / Hybrid E190 / FiLM E130 ensemble /
  MCAR v3.5.1，dB，越低越好）：Full ERB=`0.7966490938/0.8044203111/
  0.8275426681/0.8308081769`；Contra25 ERB=`1.2166261334/1.2145930843/
  1.2307760839/1.2889532413`；Contra HF=`3.5098143716/3.5019939596/
  3.7027516824/3.5380866071`；horizontal ILD=`0.5650044448/0.6298114075/
  0.6305984391/0.5811462483`。
- 相对Hybrid E190，Full差`-0.0077712173 dB`，95% CI
  `[-0.0141895164,-0.0020268012]`；Contra25差`+0.0020330491`，CI
  `[-0.0085863638,+0.0125082741]`；HF差`+0.0078204120`，CI
  `[-0.0122404210,+0.0272680784]`；ILD差`-0.0648069627`，CI
  `[-0.0882033298,-0.0434434925]`。即Full与ILD显著更好，Contra25/HF统计持平。
- 相对MCAR，Full/Contra25/HF差=`-0.0341590831/-0.0723271080/
  -0.0282722355 dB`且三个95% CI均低于0；ILD差=`-0.0161418035 dB`，CI
  `[-0.0338212935,+0.0012352252]`，均值更好但统计持平。相对FiLM E130 ensemble
  四项均显著改善。故该bounded ensemble成为当前论文主候选：相对Hybrid消除了ILD
  trade-off，同时保持Contra25/HF，且Full进一步改善；不宣称四项均值全部支配Hybrid。
- 完整性：metric/quality/aggregate/pairwise分别`704/44/16/12`行，44 subjects×4
  methods×4 metrics全部finite；三个既有基线逐被试复算最大差=`0 dB`；每被试767
  interpolation directions、72 horizontal directions，reference ILD metadata最大误差
  `9.5357133e-7 dB`；bootstrap 10000次、seed `20260828`；禁止test。MATLAB沙箱内
  首次启动因已知`File system inconsistency`在评价器前退出，升级到本机环境后同一冻结
  命令正常完成，没有不完整结果被采用。

## 2026-08-29：Stage E formal E25三成员训练完成，严格validation入口冻结

- 三个from-scratch formal run全部`FIXED_CYCLE_COMPLETE`：seeds
  `20260821/22/23`均完成25 cycles/6550 optimizer steps，best cycles=`25/25/20`，
  best validation objective=`0.7088460515845906/0.7071173556826331/
  0.7063871716911142`。权威checkpoint按结果前协议固定为各自cycle25 `last.pt`，
  不以seed23的cycle20 `best.pt`替换，也不进行成员筛选或权重调节。
- 完整性：每run history 25行且numeric finite、ledger 5项cycles=`5,10,15,20,25`；
  best/last/authoritative SHA-256全部与report匹配；三个`last.pt`各202 tensors且全部
  finite；stderr均0 bytes；运行commit `5b7e0f0` clean；每run
  `test_subjects_read=0`。E200保持未启动。
- 已实现结果前冻结的manifest生成器和hash-guarded三成员推理入口。正式ensemble固定为
  三个cycle25 residual-dB预测的`1/3,1/3,1/3`平均；后续只允许在44 validation
  subjects上与Hybrid E190、FiLM E130 ensemble、MCAR v3.5.1按既有四指标和10000次
  paired bootstrap比较。相关测试`28 passed`；禁止test。

## 2026-08-29：Stage E matched-seed E40完成，冻结formal E25

- seeds `20260821/22/23`的best cycles=`25/25/20`，best validation objective=
  `0.7088460516/0.7071173557/0.7063871717`，均值±sample std=
  `0.7074501930±0.0012627779`。0/3在cycle40达到best，因此不是endpoint-limited；
  按预注册中位数规则冻结`E_final=25`。
- 三run均40 cycles/10480 steps、history各40行、ledger各8项`5:5:40`；best/last
  checkpoint实算哈希与report匹配，每份202 tensors全部finite，stderr均0；
  `test_subjects_read=0`。E200保持未启动。
- 已冻结三份from-scratch fixed-E25 formal配置：gate重新零初始化，same-seed FiLM
  E130与MCAR v3.5.1组件不变；scheduler horizon保留40以复现搜索run前25 cycles的
  学习率序列；权威checkpoint固定为cycle25 `last.pt`。完成后仅允许1/3等权ensemble。
  证据：`results/sonicom_bounded_mcar_film_correction_matched_e40/selection.json`；
  `experiments/film_siren/STAGE_E_BOUNDED_CORRECTION_FORMAL_E25_FREEZE.md`。

## 2026-08-29：Stage E D1严格validation通过，推进matched seeds E40

- best cycle25在44 validation subjects完成全`[2,793,463]` residual预测并通过
  strict HRIR重建评价。Candidate / Hybrid E190 / FiLM E130 ensemble / MCAR v3.5.1
  的Full ERB=`0.8046509057/0.8044203111/0.8275426681/0.8308081769`，Contra25=
  `1.2230433965/1.2145930843/1.2307760839/1.2889532413`，HF=
  `3.5261775230/3.5019939596/3.7027516824/3.5380866071`，ILD=
  `0.5752025319/0.6298114075/0.6305984391/0.5811462483 dB`。
- Candidate相对MCAR四项均值差=`-0.0261572711/-0.0659098448/-0.0119090841/
  -0.0059437163 dB`；Full/Contra25 bootstrap 95% CI完全低于0，HF/ILD CI跨0。
  相对Hybrid为`+0.0002305947/+0.0084503123/+0.0241835634/-0.0546088756 dB`：
  Full持平、Contra25持平、HF显著较差、ILD显著更好。因此它是更均衡的Pareto候选，
  不是对Hybrid的全指标支配。
- 预注册结构门槛通过：相对MCAR改善4/4且无均值退化。E200仍因best cycle25而拒绝；
  下一步冻结seeds `20260822/23`同构E40并行稳定性扩展。完整性：prediction44/44
  finite；metric/quality分别704/44行、4方法×4指标；10000 bootstrap；
  `test_subject_count_read=0`。证据：`results/
  sonicom_bounded_mcar_film_correction_d1_e40_validation/`。

## 2026-08-29：Stage E bounded correction E40完成，E200无预算依据

- seed `20260821` 完成40 cycles / 10480 optimizer steps，best cycle=`25`，
  best validation objective total=`0.7088460515845906`，预注册决策=`KEEP`。
  cycle `20/25/30/35/40`分别为`0.7089483006434008 / 0.7088460515845906 /
  0.7088552428917452 / 0.7088877585801211 / 0.7088913091204383`；25之后没有
  再改善，因此E40足够，禁止无依据扩展到E200。
- 完整性：history 40行、ledger 8项且cycles=`5:5:40`、best/last checkpoint
  SHA-256分别`0AE57906...8D1290 / DD0D9F1F...BDCA3`并与report一致；两份checkpoint
  各202 tensors全部finite，活动stderr 0 bytes，`test_subjects_read=0`。
- 结果前冻结下一步：用best cycle25生成44 validation完整预测，并比较bounded
  candidate、Hybrid E190、corrected FiLM E130 ensemble、MCAR v3.5.1的既有四项严格
  指标与10000次paired bootstrap。只有通过结构推进门槛才增加matched seeds。
  证据：`results/sonicom_bounded_mcar_film_correction_d1_e40/selection.json`。

## 2026-08-29：Stage E 冻结 MCAR + 有界 FiLM-SIREN correction 预注册

- 预注册第二条融合路线：MCAR v3.5.1的两个组成模型与corrected FiLM-SIREN E130
  全部冻结，仅训练一个由FiLM最终隐特征驱动的逐query双耳门控头。公式为
  `r = r_mcar + 0.5*tanh(g_theta)*(r_film-r_mcar)`，其中
  `r_mcar=0.3*previous_joint+0.7*v351b`。门控输出层权重/偏置严格零初始化，
  因而优化前逐点精确等于MCAR；signed mixing fraction严格限制在`[-0.5,0.5]`。
- 首轮固定为seed `20260821`、E40 structure screen；AdamW `lr=1e-3, wd=0`、
  2-cycle warmup+cosine，沿用D1/D2+notch完整objective、262 train/44 validation、
  每5 cycles完整validation。若best cycle=40则标记`RETEST`，不得直接晋升。
- 结果前冻结的结构推进门槛：严格validation相对MCAR至少2/4指标改善，且任何均值
  退化不超过`0.02 dB`；是否同时支配MCAR与Hybrid仅在四项差值全部为负时报告。
  禁止test，任何推理/训练记录必须为`test_subjects_read=0`。
- 真实checkpoint CUDA smoke已通过：载入FiLM E130与MCAR两个成员后，cycle-zero
  identity error=`0.0`、maximum gate=`0.0`、仅`514`个参数可训练、反向梯度finite；
  相关pytest最终为`22 passed`（含完整Stage-E prediction block回归）。首次启动因
  Stage-E类型未加入入口白名单而在数据加载前退出；修复后第二次启动完成runtime
  provenance、但因适配层漏取grid维度在首个loss前退出。两次均为0 optimizer step、
  未产生checkpoint，并保留stderr与失败配置目录；对应缺口已各加回归覆盖。证据：
  `experiments/film_siren/
  STAGE_E_BOUNDED_MCAR_FILM_CORRECTION_PROTOCOL.md`；`configs/experiments/
  sonicom_bounded_mcar_film_correction_d1_seed20260821_e40.json`。

## 2026-08-28：Hybrid E190 vs MCAR/RANF/FSP-AE 严格 validation 完成

- 在结果前冻结commit `79fefd8`后，对44个validation subjects重新运行同一
  严格评价器，比较Hybrid E190 1/3 ensemble、MCAR v3.5.1、RANF和FSP-AE。
  RANF使用已修正的完整HRIR提取` squeeze(ranfHrir(:, ear, :)).'`；FSP-AE
  使用MATLAB `h5read`后的`[time,ear,direction]`维度。无test访问。
- 聚合均值（HYBRID / MCAR / RANF / FSP-AE，dB，越低越好）：Full ERB
  **`0.8044203111 / 0.8308081769 / 1.0746120927 / 1.1603106749`**；
  Contra25 ERB **`1.2145930843 / 1.2889532413 / 1.6274987459 / 1.8479172775`**；
  Contra HF **`3.5019939596 / 3.5380866071 / 3.4680092137 / 3.1038093697`**；
  horizontal ILD **`0.6298114075 / 0.5811462483 / 0.7901671232 / 0.5977082576`**。
- 10000次subject-paired percentile bootstrap（seed `20260819`，差值HYBRID−baseline）：
  相对MCAR，Full/Contra25/HF为`-0.0263878658/-0.0743601571/-0.0360926475 dB`，
  95% CI均低于0；ILD为`+0.0486651592`，CI `[0.0120743595,0.0866040936]`。
  相对RANF，Full/Contra25/ILD为`-0.2701917816/-0.4129056616/-0.1603557157`，
  CI均低于0；HF为`+0.0339847459`，CI `[-0.0094837640,0.0763655861]`，持平。
  相对FSP-AE，Full/Contra25为`-0.3558903638/-0.6333241933`，CI均低于0；
  HF为`+0.3981845898`，CI `[0.3524902093,0.4452281050]`；ILD为`+0.0321031499`，
  CI `[-0.0111095174,0.0752905671]`，持平。
- 论文定位：Hybrid是Full ERB和Contra25 ERB的明确第一，其HF与RANF统计持平、
  但显著弱于FSP-AE；ILD显著弱于MCAR、与FSP-AE持平、显著优于RANF。
  因此Hybrid仍是最强的广域/对侧ERB主候选，但应将FSP-AE高频与MCAR ILD
  作为明确trade-off，不宣称全指标支配。
- 完整性：`per_subject_metrics.csv` 44行、`metric_long.csv` 704行、
  `quality_checks.csv` 44行、aggregate 4行、paired comparisons 12项；44 unique
  subjects、4 methods、4 metrics，全numeric finite。Hybrid/MCAR逐被试重复评价与前一轮
  最大绝对差`0 dB`；summary/decision均`test_subject_count_read=0`。证据：
  `results/sonicom_film_siren_spectral_cnn_final_e190_vs_ranf_fsp_v351_validation/`。

## 2026-08-28：Stage D E190 ensemble 严格 validation 完成，升为论文主候选

- 固定identity `A3CFAC9C...BDCCFE`的三成员cycle190 ensemble完成44/44
  validation预测，每被试shape `[2,793,463]`且finite，
  `test_subject_count_read=0`。随后按结果前冻结的四方法、四指标口径和
  10000次paired bootstrap（seed `20260828`）完成严格评价。
- 指标均值（HYBRID / FILMENS / MCAR，dB，越低越好）：Full ERB
  **`0.8044203111 / 0.8275426681 / 0.8308081769`**；Contra25 ERB
  **`1.2145930843 / 1.2307760839 / 1.2889532413`**；Contra HF
  **`3.5019939596 / 3.7027516824 / 3.5380866071`**；horizontal ILD
  **`0.6298114075 / 0.6305984391 / 0.5811462483`**。
- HYBRID相对FILMENS：Full/Contra25/HF差值为`-0.0231223570 /
  -0.0161829996 / -0.2007577228 dB`，95% CI均完全低于0，wins=`44/43/44`；
  ILD差`-0.0007870316 dB`，CI `[-0.0029768697, 0.0014431890]`，统计持平。
- HYBRID相对MCAR：Full/Contra25/HF差值为`-0.0263878658 /
  -0.0743601571 / -0.0360926475 dB`，95% CI分别`[-0.0363633634,-0.0155767976] /
  [-0.0946435229,-0.0542189783] / [-0.0683660626,-0.0041390024]`，三项均显著
  改善；ILD差`+0.0486651592 dB`，CI `[0.0115071882,0.0868538356]`，MCAR显著更好。
- 决策：`PROMOTE_PAPER_PRIMARY_CANDIDATE`。该ensemble在三个主要spectral指标上同时
  显著超过FILMENS与MCAR，并与FILMENS的ILD持平；但不宣称全指标支配MCAR，
  论文应明确报告ILD trade-off。截稿前建议以此为主结果，FILMENS和MCAR
  分别作为消融/强基线，不再用本次validation结果调整权重。
- 完整性：`metric_long.csv` 704行，`quality_checks.csv` 44行，aggregate
  16行，bootstrap 12行；44 unique subjects、4 methods、4 metrics，全部numeric
  finite。证据：`results/sonicom_film_siren_spectral_cnn_final_e190_validation/`；
  `artifacts/reconstruction/sonicom_film_siren_spectral_cnn_final_e190_ensemble_validation/`。

## 2026-08-28：Stage D 正式 E190 三成员完成，冻结 validation ensemble

- 三个正式 run（seeds `20260821/20260822/20260823`）均从 scratch 完成固定
  `190 cycles / 49780 optimizer steps`，report 均为 `FIXED_CYCLE_COMPLETE`。唯一
  权威 checkpoint 为 cycle190 `last.pt`，SHA-256 分别为
  `0596FD3B...F4A6DC / C89CCF90...FEC789 / 465B24AD...B34BDE`。
- 训练完整性：history 各190行、ledger 各38项且 cycles精确为`5:5:190`；
  best cycles=`190/190/170`，best objective=`0.7246045510877263 /
  0.7207740897482092 / 0.7215893431143328`；best/last实算哈希与report匹配，
  每份checkpoint的102个tensors全部finite，stderr均0 bytes。运行commit
  `5548fbbf79e99872c0e5ae5c63dd1e65a89a141d` clean，每run
  `test_subjects_read=0`。
- 已冻结三成员`1/3`等权residual-dB ensemble manifest，identity
  `A3CFAC9C206E0A53FD2FA130817673AAFE07B66855322B5D34824E9173BDCCFE`；生成器
  重跑后文件SHA-256仍为`49109D146962D9049A03E5C0C98900FC0DF5AFE7F228D758C5E78D9CE4D50D03`。
  比较口径为44 validation subjects、HYBRID/PARENT/FILMENS/MCAR、四个严格
  指标与10000次paired bootstrap（seed `20260828`）；不授权test。
- 工具核验：Python syntax通过，相关pytest `4 passed`；MATLAB `checkcode`无问题；
  `git diff --check`仅有Windows换行提示。下一步在工具与manifest提交后生成
  44/44 validation预测并执行冻结评价。
- 证据：`artifacts/training/sonicom_film_siren_spectral_cnn_final_seed*_e190/`；
  `configs/experiments/sonicom_film_siren_spectral_cnn_final_e190_ensemble_manifest.json`；
  `experiments/film_siren/STAGE_D_FILM_SIREN_SPECTRAL_CNN_FORMAL_E190_FREEZE.md`。

## 2026-08-28：Stage D E200 三seed完成，冻结正式周期 E190

- E200 best-cycle搜索：seeds `20260821/20260822/20260823` 均从scratch完成200
  cycles/52400 steps，best cycles=`190/190/170`，best增强objective=
  `0.7246045510877263/0.7207740897482092/0.7215893431143328`；无seed在cycle200
  取best，满足预注册的非末点条件，故冻结`E_final=median=190`。
- 共同cycle190口径：三seed objective均值`0.722358788956295±0.0019989651472323`；
  residual MAE=`2.47966014255177±0.00318725296403402 dB`、ERB MAE=
  `0.962147936224937±0.00223736177458845 dB`、Contra HF=
  `3.54623398275086±0.00416603582010714 dB`、strict ILD=
  `0.648820926971508±0.00825336034189354 dB`。这些仅用于冻结预算，不是最终横向结论。
- 完整性：三run均`completed/KEEP`；history各200行且所有numeric finite，ledger各40项
  `5:5:200`；best/last checkpoint实算SHA-256均与report匹配，checkpoint cycle字段
  分别为best `190/190/170`与last `200/200/200`，每份102 tensors全部finite；stderr
  均0 bytes；运行commit `ce14764` clean；每run `test_subjects_read=0`。
- 正式冻结：三成员仍以same-seed corrected E130 FiLM parent为冻结base，zero-init CNN
  从scratch训练；固定190 cycles，warmup-cosine horizon保持200、warmup2；
  `formal_fixed_cycle=true`、唯一权威checkpoint为cycle190 `last.pt`。完成后以1/3等权
  residual ensemble在44 validation上与corrected FiLM ensemble和MCAR v3.5.1比较；
  不授权test。
- 证据：`results/sonicom_film_siren_spectral_cnn_e200_selection/selection.json`；
  `experiments/film_siren/STAGE_D_FILM_SIREN_SPECTRAL_CNN_FORMAL_E190_FREEZE.md`；
  E200训练目录`artifacts/training/sonicom_film_siren_spectral_cnn_d2_seed*_e200/`。

## 2026-08-28：Stage D 三 seed E40 完成，按用户决定直接执行 E200 best-cycle 搜索

- 三个matched seeds `20260821/20260822/20260823` 均完成40 cycles/10480 steps，
  best cycle均为`40`，共同增强objective分别为`0.7409078153696927 /
  0.736345036463304 / 0.7387022132223303`，均值`0.7386516883517757`、样本标准差
  `0.00228180902116173`。三者cycle35→40仅继续下降`0.0000614619 /
  0.0000224737 / 0.0000581132`，接近平台但按既有末点规则全部为`RETEST`。
- 相对各自same-seed corrected E130 FiLM parent，三seed平均objective、residual、ERB、
  Contra HF、strict ILD差值（hybrid−parent）分别为`-0.0386363096 /
  -0.1314350294 / -0.0198127713 / -0.1594190074 / -0.0011006029 dB`。结构改善
  在三个seed上方向一致；但E40只能作为结构证据，不能冻结正式周期。
- 完整性：三run均`completed/RETEST`、history各40行、ledger各8项且cycles固定
  `5:5:40`；best/last实算SHA-256与report匹配，三run的best/last model state均逐tensor
  exact equal，102个tensor全部finite；新增两run stderr均0 bytes；运行来源commit
  `c31171d` clean；每run `test_subjects_read=0`。
- 决策：`EXTEND_E200`。用户明确要求跳过增量E80，直接寻找E200范围内best点。D2从
  scratch重跑相同三seed与same-seed parent，只将训练预算和warmup-cosine horizon从
  40改为200；其余架构、输入、objective、优化器和数据边界全部不变。若任何seed仍在
  cycle200取best，则标记`ENDPOINT_LIMITED`并停止为本论文递归
  延长；否则以三个best cycle中位数冻结共同正式周期。
- 证据：`results/sonicom_film_siren_spectral_cnn_matched_seed_e40/selection.json`；
  `experiments/film_siren/
  STAGE_D_FILM_SIREN_SPECTRAL_CNN_D2_E200_BEST_CYCLE_SEARCH.md`；三个E40训练目录位于
  `artifacts/training/sonicom_film_siren_spectral_cnn_d1_seed*_e40/`。全过程未访问test。

## 2026-08-28：Stage D D1 E40 严格 validation 完成，结构通过但候选未冻结

- 训练完整性：seed `20260821` 的 frozen-FiLM + zero-init MCAR spectral CNN 完成
  40 cycles/10480 steps，best cycle=`40`、共同增强 objective=
  `0.7409078153696927`，history 40行、validation ledger 8项、best/last hashes匹配且
  model state逐tensor相同、全部finite、stderr 0。因best在预算末点，预算判据仍为
  `RETEST`，不能把E40称为已收敛或正式候选。
- 看结果前冻结的比较：D1 `best.pt`、其seed20260821 E130 parent、corrected E130
  三成员FiLM ensemble、MCAR v3.5.1；44 validation subjects；严格Full ERB、Contra25
  ERB、Contra HF、horizontal ILD；差值为HYBRID−baseline，10000次paired bootstrap，
  seed `20260828`。两套新增预测均44/44、shape `[2,793,463]`、全部finite。
- 四指标均值（dB，越低越好；HYBRID / PARENT / FILMENS / MCAR）：Full ERB
  `0.8418443025 / 0.8553604685 / 0.8275426681 / 0.8308081769`；Contra25 ERB
  `1.2546339378 / 1.2599857587 / 1.2307760839 / 1.2889532413`；Contra HF
  `3.5987376194 / 3.7635604766 / 3.7027516824 / 3.5380866071`；horizontal ILD
  `0.6589462748 / 0.6579809652 / 0.6305984391 / 0.5811462483`。
- 关键配对证据：相对PARENT，HYBRID的Full ERB差`-0.0135162`（95% CI
  `[-0.0154699,-0.0115756]`，44/44胜）、Contra25差`-0.0053518`（CI
  `[-0.0077130,-0.0029622]`，32/44胜）、HF差`-0.1648229`（CI
  `[-0.1779204,-0.1515857]`，44/44胜），ILD差`+0.0009653`（CI跨0）。相对
  FILMENS，HF改善`-0.1040141`（CI `[-0.1174916,-0.0905233]`，44/44胜），但
  Full/Contra25/ILD分别退化`+0.0143016/+0.0238579/+0.0283478 dB`且CI均不跨0。
  相对MCAR，Contra25改善`-0.0343193`（CI `[-0.0552841,-0.0140495]`），但HF与ILD
  分别退化`+0.0606510/+0.0778000 dB`且CI均为正；Full ERB差`+0.0110361`、CI跨0。
- 决策：`ADVANCE_MATCHED_SEED_SCREEN`，不是`PROMOTE_FINAL_MODEL`。该CNN对单成员
  FiLM产生跨被试一致的ERB/HF修正，证明结构互补性；但尚未超过FiLM ensemble或MCAR
  的整体Pareto前沿。下一步只补matched seeds `20260822/20260823` 的同构E40开发run，
  再依据三seed best-cycle轨迹冻结共同预算；不为D1调整架构、权重或访问test。
- 完整性与证据：严格评价44 unique subjects、704 metric rows、quality 44行，全部
  finite，`test_subject_count_read=0`。结果位于 `results/
  sonicom_film_siren_spectral_cnn_d1_e40_validation/`；预测位于 `artifacts/
  reconstruction/sonicom_film_siren_spectral_cnn_d1_seed20260821_e40_best_validation/`
  与 `artifacts/reconstruction/
  sonicom_film_siren_gl_final_d1d2_notch_seed20260821_e130_validation/`；评价规则冻结
  commit `7185645`。

## 2026-08-28：Stage D FiLM-SIREN + zero-init MCAR spectral CNN 预注册并完成冒烟

- 研究问题：沿用 corrected C4 的 train+validation 口径，以正式 E130 FiLM-SIREN
  为连续个体化基线，增加 MCAR 的双耳局部频谱 CNN，尝试在保留 Full/Contra ERB
  优势的同时修复 Contra HF 与 horizontal ILD。该路线是新开发候选，不改写已冻结
  的 FiLM/MCAR 结论，也不授予任何新 test 访问。
- D1 架构：载入 seed `20260821` 的权威 cycle130 `last.pt`（SHA-256
  `E37676D843D608B4DDD7B311EBA8B28A933183A03E7BAA49E1CF20E35F8B4129`）并冻结
  全部 FiLM-SIREN/Q26 encoder 参数；接入原 MCAR `BinauralSpectralCNN`（width48、
  kernel7、dilations 1/2/4/8、direction width64）。七通道固定为左右耳各自的
  normalized MCA/correction/FiLM base residual 加 normalized log-frequency；双耳
  输出层零初始化，最终为 `base + delta`。
- D1 预算：单 seed `20260821`、40 cycles、每cycle 262 train subjects；AdamW
  `lr=1e-4/wd=1e-4`，2-cycle warmup + cosine horizon40；每5 cycles在44 validation
  被试记录共同 D1/D2+notch objective并按最小值选择best。FiLM backbone禁止解冻；
  D1只作one-seed开发筛选，若通过再补seeds 20260822/23并冻结共同正式周期。
- 实现与验证：新增混合模型、Stage-C训练入口兼容分支、validation-only预测适配器和
  冒烟脚本。零初始化逐点等价性以`rtol=0, atol=0`通过；真实train被试P0002、2个
  global+2个horizontal方向、完整463频点的一步前反向冒烟通过，loss
  `0.5122836232185364`、spectral output gradient norm `3.0965585708618164`，
  FiLM主体无梯度。相关pytest `57 passed`，compileall和`git diff --check`通过；仅有
  pytest cache目录权限warning。
- 边界与证据：`test_subjects_read=0`；协议见 `experiments/film_siren/
  STAGE_D_FILM_SIREN_SPECTRAL_CNN_PROTOCOL.md`，配置见 `configs/experiments/
  sonicom_film_siren_spectral_cnn_d1_seed20260821_e40.json`（SHA-256
  `E77B5A34A08D07DEF4B3B436FB5A05282DBFC54DEACE256692A2E711C55F4155`）。

## 2026-08-28：FiLM-SIREN D1/D2+notch 正式 E130 三成员完成，四方法 validation 严格比较

- 正式训练：三个 seed `20260821/20260822/20260823` 从 scratch 固定130 cycles
  全部自然结束，均 `status=completed`、`decision=FIXED_CYCLE_COMPLETE`、130
  cycles/34060 steps；history 各130行且0坏值、ledger各26项且末项cycle130；
  `best.pt`/`last.pt` 实算 SHA-256 均与 report 一致（authoritative 均为
  cycle130 `last.pt`）；stderr 无错误；运行时 git `5356f59` clean；
  `test_subjects_read=0`、`local_mca_inputs_read=35204`。三成员 best cycle
  （仅诊断）为 `120/125/115`，best total `0.7795758/0.7748292/0.7757417`。
- 冻结 manifest：`configs/experiments/
  sonicom_film_siren_gl_final_d1d2_notch_e130_ensemble_manifest.json`，identity
  `1CE74CF0AEC21D29FDE22B952FBB038EDBA8600938037855028ECE84B06936BE`，三成员
  config/checkpoint SHA-256 逐项锁定，权重严格1/3。validation 44被试 ensemble
  推理44/44完成，全部 shape `[2,793,463]` 且 finite，elapsed `25.2042783 s`，
  `test_subject_count_read=0`。
- 四方法严格评价（专用入口 `matlab/+mcar/
  evaluate_film_siren_d1d2_notch_four_method_validation.m`，不含v1/v2、禁止test）：
  44/44 unique subjects、704 metric rows、quality checks一致（每人767
  interpolation + 72 horizontal directions、41 ERB bands、reference ILD metadata
  max error ~1e-6 dB）。聚合均值（dB，越低越好）：
  - Full ERB：FILM `0.8275426680633419` / MCAR v3.5.1 `0.8308081768778546` /
    RANF `1.0746120926761717` / FSP-AE `1.1603106749036973`；
  - Contra25 ERB：`1.2307760839028756` / `1.2889532413126572` / `1.627498745890078` /
    `1.8479172775430348`；
  - Contra HF：`3.7027516824075195` / `3.538086607059492` / `3.4680092137126852` /
    `3.103809369743327`；
  - Horizontal ILD：`0.63059843905319` / `0.5811462482733765` /
    `0.7901671231596392` / `0.5977082576411121`。
  - 更正记录：初版评价器对 RANF 的 HRIR 提取存在 bug（`ranfHrir(:, :, 1)` 只取
    单时间样本而非整段 HRIR），导致 RANF 的 ERB/ILD 虚高（6.54/9.12/11.25）；
    修复为 `squeeze(ranfHrir(:, 1, :)).'` 后重跑，RANF 恢复合理量级并与历史
    learned-sparsity 研究（ERB 1.06、ILD 0.78）一致。FILM/MCAR/FSP-AE 数值不受
    影响，本条目全部数值为修复后重跑的正式值。
- paired bootstrap（44 subject rows、10000次、seed20260819、双侧percentile 95%
  CI，diff=FILM−baseline，负为优）：Full ERB vs MCAR `-0.0032655`（CI
  `[-0.0140183,+0.0079552]`、wins29/44）、vs FSP-AE `-0.3327680`（CI
  `[-0.3628319,-0.3038376]`、wins44/44）、vs RANF `-0.2470694`（CI
  `[-0.2654305,-0.2280328]`、wins44/44）；
  Contra25 ERB vs MCAR `-0.0581772`（CI `[-0.0789625,-0.0379363]`、wins36/44）、
  vs FSP-AE `-0.6171412`（wins44/44）、vs RANF `-0.3967227`（CI
  `[-0.4536546,-0.3382808]`、wins43/44）；
  Contra HF vs MCAR `+0.1646651`（CI 正、wins3/44）、vs FSP-AE `+0.5989423`
  （wins0/44）、vs RANF `+0.2347425`（CI 正、wins4/44）；Horizontal ILD vs MCAR
  `+0.0494522`（CI `[+0.0123300,+0.0878907]`、wins13/44）、vs FSP-AE `+0.0328902`
  （CI 跨0、wins18/44）、vs RANF `-0.1595687`（CI
  `[-0.2133469,-0.1046279]`、wins36/44）。
- 结论口径：这是44 validation 上的开发/工程比较，不是独立论文确认；FILM 在
  Full ERB 上略优/持平 MCAR 且显著优于 FSP-AE/RANF，在 Contra25 ERB 上显著优于
  三者，但在 Contra HF 和 Horizontal ILD 上仍劣于 MCAR v3.5.1 与 FSP-AE（RANF
  除外）。`test_subjects_read=0`；已消费的 test 未用于任何选择或调参。
- 证据：`results/sonicom_film_siren_gl_final_d1d2_notch_vs_ranf_fsp_v351_validation/`
  （per_subject_metrics/metric_long/aggregate_metrics/quality_checks/summary/
  four_method_decision/figures）；manifest identity 见上；训练目录 `artifacts/
  training/sonicom_film_siren_gl_final_d1d2_notch_seed{20260821,20260822,
  20260823}_e130/`；validation预测 `artifacts/reconstruction/
  sonicom_film_siren_gl_final_d1d2_notch_e130_ensemble_validation/`。

## 2026-08-27：FiLM-SIREN D1/D2+notch 正式 E130 三成员训练启动

- 配置冻结：按 C4 共同评分修订与三 seed 开发扩展，`E_final=130`（best cycles
  `140/130/120`，`round(median)`）。正式配置 = AdamW `lr=1e-4/wd=1e-4` +
  warmup-cosine（warmup5、horizon150）+ D1/D2/notch objective 权重
  `0.25/0.15/0.30` + 七维 global+local MCA FiLM-SIREN，`formal_fixed_cycle=true`、
  `checkpoint_policy=fixed_stop_cycle_last`。
- 三个正式成员：seeds `20260821/20260822/20260823`，均从 scratch、固定130 cycles、
  每cycle 262 steps、scheduler horizon150截断、独立输出目录/provenance；唯一权威
  checkpoint为各自cycle130 `last.pt`；三成员将以1/3 residual-dB等权组成ensemble。
- 边界：只解析262 train+44 validation，`test_subjects_read=0`；最终横向表固定为
  新D1/D2+notch ensemble、MCAR v3.5.1、RANF、FSP-AE四方法，不含v1/v2或原C0。
- 证据：`experiments/film_siren/
  STAGE_C_GLOBAL_LOCAL_MCA_C4_CORRECTED_FORMAL_TRAINING_FREEZE.md`；
  `scripts/generate_siren_gl_c4_formal_configs.py`；三份 `configs/experiments/
  sonicom_film_siren_gl_final_d1d2_notch_seed{20260821,20260822,20260823}_e130.json`；
  来源选择 `results/sonicom_film_siren_gl_c4_corrected_expansion/selection.json`。
- 启动核验：三份配置字段/hash 独立核验通过；相关测试49项通过（仅pytest cache
  权限warning）；训练 PID/输出目录见交接页；finite与预算完成性待训练结束后核验。
- 评价工具（结果前冻结）：`matlab/+mcar/
  evaluate_film_siren_d1d2_notch_four_method_validation.m` 为专用四方法严格评价
  入口，只比较新ensemble、MCAR v3.5.1、RANF、FSP-AE，不构造v1/v2、不允许test
  split；`scripts/generate_siren_gl_c4_validation_manifest.py` 生成无test授权
  的 validation manifest；`scripts/analyze_siren_gl_c4_four_method_validation.py`
  做预注册 paired bootstrap（10000次、seed20260819）。三者已提交 commit `750088e`。
  三套基线（v351/RANF/FSP-AE）validation 预测均为44/44且与冻结val split完全一致。

## 2026-08-27：FiLM-SIREN C4 修正候选三种子完成，冻结 E_final=130

- 运行结果：D1/D2+notch 的三个开发种子 `20260821/20260822/20260823` 均完成
  150 cycles，增强 objective 各自选择的 best cycle 为 `140/130/120`，对应 raw
  objective total 为 `0.7760627865791321 / 0.7751666510646994 /
  0.7733549123460596`。没有 best 落在 cycle 150，因此不触发 RETEST；按预注册的
  `round(median(best_cycles))` 冻结正式训练周期 **`E_final=130`**。
- 共同量尺结果：三个 best checkpoint 的 standardized C0 为
  `0.7075952639451233 / 0.7072001765231013 / 0.7055228087475707`，三种子均值
  **`0.7067727497385984`**。同点均值 residual MAE=`2.6629008361787507 dB`、
  ERB MAE=`0.9987630993127823 dB`、contralateral HF MAE=
  `3.754646817843119 dB`、strict ILD MAE=`0.63998863688021 dB`。
- 历史开发对照：相对原 FiLM-SIREN cosine+C0 三种子均值，standardized C0、
  residual、HF、strict ILD 分别降低约 `1.12% / 1.24% / 0.85% / 1.44%`；ERB
  增加约 `0.04%`。这只支持推进正式 validation 比较，不是对 MCAR v3.5.1、RANF
  或 FSP-AE 的最终结论。
- 完整性：三 run 均 `status=completed`、`decision=KEEP`；每个 history 150 行、
  validation ledger 30 个条目且末项 cycle 150；`best.pt`/`last.pt` 的实算 SHA-256
  均与 report 一致；新增两 run 的 stderr 均为0 bytes；所有聚合数值 finite，
  `test_subjects_read=0`。训练来源为 clean commit
  `9acc82c21895ffec6b87e8832d7f288f4fc8e20b`。
- 冻结后续：下一步从 scratch 训练三个 seed `20260821/20260822/20260823` 的固定
  130-cycle 正式成员，scheduler horizon 保持150，权威 checkpoint 只能是 cycle130
  `last.pt`；随后在44名 validation 被试上做1/3 residual-dB等权 ensemble，并只与
  MCAR v3.5.1、RANF、FSP-AE进行同口径横向比较，不纳入v1/v2或原C0。不得读取test；
  任何新test步骤仍需单独预注册和用户明确授权。
- 证据：`results/sonicom_film_siren_gl_c4_corrected_expansion/selection.json`；
  `experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_C4_COMMON_SCORE_CORRECTION.md`；
  三个训练目录位于`artifacts/training/
  sonicom_film_siren_gl_c4_{d1d2_notch_seed20260821_e150,
  corrected_d1d2_notch_seed20260822_e150,
  corrected_d1d2_notch_seed20260823_e150}/`。

## 2026-08-27：FiLM-SIREN C4 共同评分修订与 D1/D2+notch 扩展冻结

- 修订原因：原 C4 将 C0 与含 D1/D2/notch 附加非负项的候选按各自 raw total
  直接混排，量尺不一致。该问题不改写原始 decision，也不推翻已完成的 frozen test；
  现将其透明登记为开发集 protocol correction。
- 统一复核：以 train-only target std `4.954314859581426` 重算共同 C0，seed
  `20260821` 的 D1/D2+notch、D1/D2、notch、C0 最佳值分别为
  `0.7075494388355034 / 0.7093399539540178 / 0.7144781864630014 /
  0.7170006462537132`。D1/D2+notch 同时降低基础 residual、ERB、HF 和 strict ILD，
  因此用户授权把该候选推进到最终 validation 横向比较。
- 前瞻协议：保留已有 seed `20260821`，新增 seeds `20260822/20260823` 的同配置
  150-cycle 从 scratch run；三 seed 用相同增强 objective 选 best cycle并冻结
  `E_final=round(median)`。随后从 scratch 训练三个固定周期成员，只以末点 `last.pt`
  做1/3等权 residual ensemble。
- 边界：本轮只使用262 train和44 validation；最终横向表固定为新D1/D2+notch
  ensemble、MCAR v3.5.1、RANF和FSP-AE，不纳入旧v1/v2，原C0只作历史开发证据。
  `test_subjects_read=0`；任何新test访问都需要另行冻结manifest/registry并重新取得
  明确授权。
- 证据：`experiments/film_siren/
  STAGE_C_GLOBAL_LOCAL_MCA_C4_COMMON_SCORE_CORRECTION.md`；
  `scripts/generate_siren_gl_c4_corrected_expansion_configs.py`；
  `scripts/analyze_siren_gl_c4_corrected_expansion.py`。

## 2026-08-27：FiLM-SIREN 验证集 44 被试 HRTF 重建对比图

- 任务与边界：按用户要求，在冻结 SONICOM validation split 的 44 名被试上绘制逐被试
  HRTF 重建曲线；比较 FiLM-SIREN GL cosine+C0 E140 三成员 1/3 ensemble、RANF、
  FSP-AE 和 MCAR v3.5.1，并以 reference 为锚点。全过程只读 validation，
  `test_subjects_read=0`，不改变已冻结的 engineering test 决策。
- 预测来源：RANF、FSP-AE、MCAR v3.5.1 三套既有 validation 预测均为 44/44，subject
  集合与冻结 val split 差异为 0。FiLM-SIREN 以冻结 manifest
  `5B5A4444D342500BFC3C35F0701A26132C26886AEA073D4BA30FD8E2C6F5A407`重新生成
  validation 预测，44/44 输出均为 `[2,793,463]` 且 finite，用时 `23.9362304 s`。
- 绘图口径：每名被试选择左耳最接近 `(270°,0°)` 的纯插值方向，使用相同方向顺序与
  strict-ILD 频率 bin，在 `100 Hz--20 kHz` 绘制 reference 与四种重建的 dB 频响；
  44 名被试采用固定 `8×6` tiled layout、统一 `[-60,25] dB` 纵轴并导出 PNG/PDF。
- 可视方向摘要：该单方向、未加权频点 MAE 仅用于辅助阅读图形，不能代替全空间严格
  指标或用于模型晋升。44 人均值依次为 FSP-AE `3.66570582309139 dB`、FiLM-SIREN
  `3.97510342990629 dB`、RANF `4.06820359685896 dB`、MCAR v3.5.1
  `4.11332155808857 dB`；对应被试标准差为 `0.638492343815334 / 0.798592747169056 /
  0.87755140456774 / 0.927696757383611 dB`。
- 完整性与产物：逐被试表 44 行且 SubjectLabel 唯一，所有数值 finite；图像已人工检查，
  图例、颜色、标题和 44 面板布局可读。沙箱内 MATLAB 首次因环境级
  `File system inconsistency`未启动；正常本机环境以同一命令完成。实现位于
  `matlab/+mcar/plot_film_siren_ranf_fsp_v351_validation_subject_hrtf.m`；正式图、
  quality checks、逐被试与聚合 CSV 位于
  `results/sonicom_validation_film_siren_ranf_fsp_v351_subject_hrtf/`。

## 2026-08-27：FiLM-SIREN frozen SONICOM engineering test 完成，DO NOT PROMOTE

- 授权与身份：用户于`2026-08-27T12:53:26+08:00`明确授权按冻结manifest执行一次
  SONICOM engineering test；授权记录commit `de2e3bd`。候选身份为manifest
  `5B5A4444D342500BFC3C35F0701A26132C26886AEA073D4BA30FD8E2C6F5A407`，三个
  cosine+C0 E140成员只使用cycle140 `last.pt`，1/3 residual-dB等权ensemble。
- 预测完整性：唯一一次test预测44/44完成，所有输出shape `[2,793,463]`且finite，
  elapsed `31.0566976 s`；registry原子状态为`completed`、
  `test_subject_count_read=44`并以commit `988ec1a`锁定。无失败、重试、删seed或调权。
- 严格评价完整性：MATLAB正式评价44/44 unique subjects、所有数值finite；每人固定767
  interpolation-only directions和72 horizontal interpolation directions；reference ILD
  metadata max error=`9.53052371244212e-07 dB`。首次沙箱内MATLAB启动因环境级
  `File system inconsistency`立即失败且未生成结果；随后同一冻结命令在正常环境完成，
  不构成数据/模型重试。
- test均值（FiLM-SIREN vs MCAR v3.5.1，dB，越低越好）：Full ERB
  `0.8355821306114706` vs `0.8179370487968771`；Contra25 ERB
  `1.2135525652687749` vs `1.2744892321005215`；HF
  `3.683294867107304` vs `3.5084809628530294`；strict horizontal ILD
  `0.7417994296287659` vs `0.6447090916559254`。
- 预注册paired bootstrap（44 subject rows、10000次、seed`20260818`、双侧percentile
  95% CI；difference=FiLM-SIREN−v3.5.1）：Full ERB difference
  `+0.017645081814593403`，CI `[-0.01147642414780341, +0.07060930613949087]`，
  wins `30/44`；**primary superiority失败**（upper不小于0），但26/44工程多数gate通过。
  Contra ERB difference `-0.06093666683174658`，CI
  `[-0.09254066080423129, -0.03636937636157839]`，wins `39/44`，0.02 dB NI gate通过。
  HF difference `+0.17481390425427343`，CI
  `[+0.1423438822598124, +0.20824836432767746]`，wins `3/44`，0.05 dB NI gate失败。
  strict ILD difference `+0.0970903379728404`，CI
  `[+0.014165527351709391, +0.23405523689892418]`，wins `18/44`，0.02 dB NI gate失败。
- 决策：冻结为 **`DO_NOT_PROMOTE`**。候选只在Contra ERB通过预注册gate；primary
  superiority、HF NI与strict ILD NI均失败。因此继续保留**MCAR v3.5.1为当前工程主模型**。
  本结果是已被项目历史消费的SONICOM engineering test，不是独立论文确认；不得以此
  结果重训、重新选择seed或修改ensemble权重。FiLM-SIREN仍作为已冻结研究候选和
  ablation/continuous-query研究对象保留。
- 证据：`results/sonicom_film_siren_gl_final_vs_v351_frozen_test/`中的
  `per_subject_metrics.csv`、`quality_checks.csv`、`aggregate_metrics.csv`、
  `summary.json`与`statistical_decision.json`；预测产物位于`artifacts/reconstruction/
  sonicom_film_siren_gl_final_cosine_c0_e140_ensemble_test/`。

## 2026-08-27：FiLM-SIREN 最终 ensemble manifest 与测试注册表冻结

- 冻结对象：三个正式 cosine+C0、cycle140 `last.pt` 以固定 `1/3` 权重在 residual-dB
  空间求均值；manifest 身份 SHA-256 为
  `5B5A4444D342500BFC3C35F0701A26132C26886AEA073D4BA30FD8E2C6F5A407`。三个成员的
  config/checkpoint hashes、seed、`E_final=140`、scheduler horizon=150，以及
  split/Q26/normalization/frequency mapping/interpolation policy/evaluator资源均已逐项锁定。
- 测试边界：独立 registry 状态为 `not_started`、authorization=`null`、
  `test_subject_count_read=0`。测试预测入口要求 manifest 校验成功，并且同时提供
  `--split test --allow-test --registry ...`；hash不匹配会在构造任何test HDF5路径前
  失败，registry以manifest身份和三组config/checkpoint hashes识别模型，并以原子替换
  写入 `started/completed/failed`。
- 统计判据：冻结 paired difference=`FiLM-SIREN - MCAR v3.5.1`；主终点为
  interpolation-only、solid-angle-weighted Full-sphere ERB；44名被试配对bootstrap
  10000次、seed `20260818`、双侧percentile 95% CI，主终点要求upper `<0`；三个次要
  NI margin依次为Contra ERB `0.02 dB`、HF `0.05 dB`、strict ILD `0.02 dB`且各自
  upper小于margin；另要求Full ERB至少`26/44`被试胜出，仅解释为工程多数gate。
- 实现验证：七维global+local预测器、manifest/resource guard、requested-split-only路径
  构造及registry原子状态测试共25项通过；最终manifest逐项hash复核通过。以1名validation
  被试P0001做三成员真实推理冒烟测试，输出shape `[2,793,463]`且全部finite，用时
  `0.7117471 s`，`test_subject_count_read=0`。统计脚本另在既有44行历史冻结比较CSV上
  完成10000次bootstrap冒烟测试。pytest仅有无法创建cache的非功能性权限warning。
- 证据：`configs/experiments/
  sonicom_film_siren_gl_final_cosine_c0_e140_ensemble_manifest.json`；
  `configs/experiments/sonicom_film_siren_gl_final_test_registry.json`；
  `src/mcar/evaluation/predict_film_siren_ensemble.py`；
  `scripts/analyze_siren_gl_frozen_test.py`；evaluator/tooling commit
  `4d17da4f7ecff283c51d525852e05d1583bf3a50`。
- 下一步：提交manifest/registry与本条日志，确认Git clean；随后只向用户请求一次明确的
  frozen SONICOM engineering test授权。未获授权前不得运行test预测、MATLAB test评价
  或正式统计分析，也不得根据test结果重训、重选seed或调ensemble权重。

## 2026-08-27：FiLM-SIREN Stage C 三个正式 E140 模型训练完成

- 实验目标：按已冻结的 cosine+C0 最终配置，以 seeds
  `20260821/20260822/20260823` 从 scratch 并行训练三个固定周期 ensemble member；
  `stop_cycle=140`，cosine scheduler horizon=150，不 early stop、不用 validation
  选择 checkpoint，唯一权威 checkpoint 为各自 cycle140 `last.pt`。
- 结果：3/3 run 全部 `status=completed`、`decision=FIXED_CYCLE_COMPLETE`，每个
  `36680` optimizer steps（140×262）。cycle140 validation Stage C total 分别为
  seed21 `0.71618937226859`、seed22 `0.715140467340296`、seed23
  `0.7132705422964963`；这些数值只作固定末点诊断，不用于删 seed、调权或重新选周期。
  三个成员全部按1/3权重进入最终 residual-dB ensemble。
- 权威 checkpoint SHA-256：seed21
  `52C1F432F8CFA24DE9AA4BB86941F29E15122FB24072C359CC84D3AD937876B3`；seed22
  `063B894A3608C3D9E2384C9151E8A3A1AAE3E15E77569E4E942AE20418BCF7CC`；seed23
  `06C7C4CCF62B9AB77D08324036CFD12888FC5EC4E4E28864D348A301B7C815DB`。
  三个报告的 authoritative hash 均与磁盘 `last.pt` 独立重算结果一致；`best.pt`
  仅为诊断产物。
- 完整性：三个原进程均已自然退出；history.csv 各140行、末行cycle140、0个
  NaN/Inf；validation ledger各28个评价点、末点cycle140；stderr均0 bytes；全部
  `test_subjects_read=0`、`local_mca_inputs_read=37912`、
  `condition_inputs_read=306`；运行时 commit
  `00100edb14fcd72ef7c046c8ffd6e2dac29f1af4`且Git clean。单模型峰值CUDA
  allocated均为 `411.9921875 MiB`。
- 产物：三个 run 位于 `artifacts/training/
  sonicom_film_siren_gl_final_cosine_c0_seed*_e140/`；训练冻结协议为
  `experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_FINAL_TRAINING_FREEZE.md`。
- 下一步：冻结包含三个 config/checkpoint hash及数据/evaluator身份的 ensemble
  manifest和独立test registry并提交；两者完成前不得访问SONICOM test。

## 2026-08-27：FiLM-SIREN Stage C Top-3 三-seed 汇总与最终配置冻结

- 实验目标：完成全局 Top-3 的 seeds `20260822/20260823` 扩展后，按预注册规则将
  每个候选的 screening seed `20260821` 与两个扩展 seed 的 best validation Stage C
  total 作三-seed等权平均，选出最终配置；再以最终配置三 seed 的 best cycles 中位数
  冻结 `E_final`。
- 三-seed结果（mean Stage C total，越低越好）：**cosine+C0**
  **`0.7147615646774118`**（单 seed `0.7162436748092825`@140、
  `0.7145319106903943`@130、`0.7135091085325588`@140）< warmup_cosine+C0
  `0.7148944180120121`（`0.7148047604344108`@140、`0.7157314690676603`@120、
  `0.7141470245339654`@135）< constant+C0 `0.7172623832117427`
  （`0.7162069299004294`@140、`0.7173739319497888`@145、
  `0.71820628778501`@120）。主指标不相等，因此不进入 tie-break。
- 决策：冻结最终配置为 **AdamW `lr=1e-4`, `wd=1e-4` + cosine scheduler + C0
  objective**。最终配置 best cycles 为 `140/130/140`，故按
  `round(median(E1,E2,E3))` 冻结 **`E_final=140`**。此前实时交接中建议的
  warmup_cosine / 135 是沿用 screening rank 1、未按三-seed规则重排，现已作废；
  本次修正严格执行既有预注册规则，不需要 amendment。
- 正式重训边界：seeds `20260821/20260822/20260823` 三模型全部从 scratch；固定训练
  140 cycles，每 cycle 262 steps；不 early stop、不选择 validation best checkpoint；
  cosine scheduler horizon 保持150并在 cycle 140截断；最终只使用每个 run 的末点
  checkpoint，三模型 residual 作1/3等权平均，不搜索权重、不删除 seed。
- 完整性：用于汇总的9个 run 全部 `status=completed`、KEEP；history.csv 各150行且
  无 NaN/Inf；全部 `test_subjects_read=0`；运行时 Git clean。证据为
  `results/sonicom_film_siren_gl_top3/selection.json`、三个 screening run 与六个
  `artifacts/training/sonicom_film_siren_gl_top3_*/{training_report.json,
  validation_ledger.json,history.csv}`。
- 下一步：生成三份正式 E140 配置并在 clean Git commit 上启动；完成后核验末点
  checkpoint hashes，冻结1/3 ensemble manifest 与独立 test registry，之后才可请求
  SONICOM engineering test 访问。

## 2026-08-27：FiLM-SIREN Stage C GL-C4：objective ablation 完成 + 全局 Top-3 选定

- 实验目标：固定 GL-C3 winner（AdamW `lr=1e-4` `wd=1e-4`、warmup_cosine warmup 5），
  按 `STAGE_C_GLOBAL_LOCAL_MCA_C4_PARALLEL_AMENDMENT.md` 从 scratch 同时启动四个
  独立 run（C0 也重跑）：C0、C0+D1/D2（权重 `0.25/0.15`、≥4kHz）、C0+multi-scale
  notch（权重 `0.30`、4–18kHz、半径 `{4,8,16}`、thr `1dB`、softplus `0.5dB`）、
  C0+D1/D2+notch；seed `20260821`、150 cycles。
- GL-C4 结果（best validation Stage C total，越低越好）：**C0（重跑）**
  **`0.7170006659897891`**（best cycle `140`，KEEP）< C0+D1/D2
  `0.7442312213507566`（cycle `140`，KEEP）< C0+notch `0.7498216263272546`
  （cycle `135`，KEEP）< C0+D1/D2+notch `0.7760627865791321`（cycle `140`，
  KEEP）。四个候选 best 均早于 cycle 150，全部 `KEEP`、无 RETEST。
- GL-C4 决策：冻结 **C0（纯 Stage C total，不加 D1/D2/notch）** 为 GL-C4 winner；
  附加项在 best-cycle 上均未改善 objective total，按协议不调整权重或频率边界。
- 全局 Top-3（`results/sonicom_film_siren_gl_top3/selection.json`）：13 个
  screening run 按 unique 配置（optimizer+scheduler+objective 权重）分组排名
  （10 unique；GL-C4 c0 与 GL-C3 winner 共享 slot、GL-C3 constant 与 GL-C2
  winner 共享 slot、GL-C2 adam_wd0 与 GL-C1 lr1e4 共享 slot），组内取最优 run：
  1. warmup_cosine+C0（GL-C3 winner）`0.7148047604344108`
  2. constant+C0（GL-C2 winner）`0.7162069299004294`
  3. cosine+C0 `0.7162436748092825`
  均为 KEEP。Top-3 各补跑 seeds `20260822/20260823`（6 个从 scratch run，命名
  `sonicom_film_siren_gl_top3_rank{n}_{variant}_seed{seed}_e150`），并行启动。
- 完整性：4/4 GL-C4 run + 13/13 screening run 均 `status=completed`、history.csv
  各 150 行、无 NaN/Inf；全部 `test_subjects_read=0`、`local_mca_inputs_read=40620`、
  git `09d5083`（GL-C4 运行时）clean；汇总脚本 `scripts/analyze_siren_gl_c4.py`
  与 `scripts/analyze_siren_gl_top3.py` 已提交（commit `49a4435`）。
- 产物：`results/sonicom_film_siren_gl_c4_objective/decision.json`；
  `results/sonicom_film_siren_gl_top3/selection.json`；六配置
  `configs/experiments/sonicom_film_siren_gl_top3_*.json`；修订
  `experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_TOP3_SEED_EXPANSION_AMENDMENT.md`。
- 下一步：六个 Top-3 补 seed run 结束后，以最终配置的三 seed best cycles 确定
  `E_final=round(median(E1,E2,E3))`，再进入固定 E_final 正式重训 + 1/3 等权
  ensemble。

## 2026-08-27：FiLM-SIREN Stage C GL-C3：scheduler 重验证完成

- 实验目标：在七维 global+local MCA 架构上，按
  `STAGE_C_GLOBAL_LOCAL_MCA_REVALIDATION_PROTOCOL.md` 与
  `STAGE_C_GLOBAL_LOCAL_MCA_C3_PARALLEL_AMENDMENT.md`，固定 GL-C2 winner
  `AdamW / lr=1e-4 / wd=1e-4`、C0 objective，从 scratch 同时启动三个独立 run：
  constant（重跑）、cosine、5-cycle linear warmup + cosine（warmup 5）；seed
  `20260821`、150 cycles、scheduler horizon 150。
- 结果（best validation Stage C total，越低越好）：warmup+cosine
  **`0.7148047604344108`**（best cycle `140`，KEEP）< cosine
  `0.7162436748092825`（cycle `140`，KEEP）< constant（重跑）
  `0.7215440015901219`（cycle `140`，KEEP）。三个候选 best 均早于 cycle 150，
  无末点 best，全部 `KEEP`、不触发 `RETEST`。与 GL-C2 winner 同配置的 constant
  重跑（`0.721544`）比 GL-C2 AdamW/wd`1e-4` run（`0.716207`）略差，但与 GL-C1
  winner 同配置的 Adam/wd0 类似存在 run 间波动；三新 run 同 seed 同架构下排名稳定。
- 决策：冻结 **warmup+cosine（warmup 5 cycles）+ AdamW / lr `1e-4` / wd `1e-4`**
  为 GL-C3 winner 进入 GL-C4 objective 搜索；不做 scheduler × objective 全笛卡尔积。
- 完整性：3/3 run 完成且 `status=completed`；history.csv 各 150 行、无 NaN/Inf；
  全部 `test_subjects_read=0`、`local_mca_inputs_read=40620`、
  `condition_inputs_read=306`、`conditioning_scope=global_plus_local_mca`；
  运行时 git `5c8a025` clean；FP32、无 AMP、gradient clip 5。
- 产物：决策位于
  `results/sonicom_film_siren_gl_c3_scheduler/decision.json`；配置位于
  `configs/experiments/sonicom_film_siren_gl_c3_*.json`；逐 run 产物位于
  `artifacts/training/sonicom_film_siren_gl_c3_*/`。
- 下一步：固定该 winner 进入 GL-C4 objective ablation（C0 继承、新增
  D1/D2、notch、D1/D2+notch，权重与频率边界见
  `STAGE_C_TRAINING_PROTOCOL.md` 第 5.4 条）。

## 2026-08-26：FiLM-SIREN Stage C GL-C1：七维 global+local learning-rate 重验证完成

- 实验目标：在七维 global+local MCA 架构上（query
  `[x,y,z,dual-f1,dual-f2,normalized MCA-L,normalized MCA-R]`、Q26 global
  latent128、full modulation、all placement），按
  `experiments/film_siren/STAGE_C_GLOBAL_LOCAL_MCA_REVALIDATION_PROTOCOL.md`
  从头重新验证 Stage C 搜索第一步：固定 Adam、wd=0、constant scheduler、C0
  objective，比较 learning rate `{3e-5,1e-4,3e-4}`；screening seed `20260821`、
  150 cycles。旧五维 global-only C1 只作历史对照，不进入新架构候选排序。
- 结果（best validation Stage C total，越低越好）：`1e-4`
  **`0.7209096320650794`**（best cycle `125`，KEEP）< `3e-5`
  `0.7245849723165686`（cycle `145`，KEEP）< `3e-4`
  `0.7827154099941254`（cycle `65`，KEEP）。三个候选 best 均早于 cycle 150，
  无末点 best，全部 `KEEP`、不触发 `RETEST`。
- 决策：冻结 **Adam / lr `1e-4` / wd `0`** 为 GL-C1 winner 进入 GL-C2
  optimizer/weight-decay 搜索；较大的 `3e-4` 明显退化，较小的 `3e-5` 与 winner
  接近但未胜出。
- 完整性：3/3 run 完成且 `status=completed`，history.csv 各 150 行、无 NaN/Inf；
  全部 `test_subjects_read=0`、`local_mca_inputs_read=40620`、
  `condition_inputs_read=306`、`conditioning_scope=global_plus_local_mca`；
  运行时 git `c20764e` clean（预注册提交）；FP32、无 AMP、gradient clip 5。
- 产物：决策位于
  `results/sonicom_film_siren_gl_c1_learning_rate/decision.json`；配置位于
  `configs/experiments/sonicom_film_siren_gl_c1_*.json`；逐 run 产物位于
  `artifacts/training/sonicom_film_siren_gl_c1_*/`。
- 下一步：固定该 winner 进入 GL-C2（三配置并行修订见
  `STAGE_C_GLOBAL_LOCAL_MCA_C2_PARALLEL_AMENDMENT.md`，结果见下一条目）。

## 2026-08-26：FiLM-SIREN Stage C GL-C2：optimizer / weight-decay 重验证完成

- 实验目标：在七维 global+local MCA 架构上，按
  `STAGE_C_GLOBAL_LOCAL_MCA_REVALIDATION_PROTOCOL.md` 与
  `STAGE_C_GLOBAL_LOCAL_MCA_C2_PARALLEL_AMENDMENT.md`，固定 GL-C1 winner
  `lr=1e-4`、constant scheduler、C0 objective，从 scratch 同时启动三个独立 run
  （不复用 GL-C1 artifact）：Adam/wd0、AdamW/wd`1e-5`、AdamW/wd`1e-4`，seed
  `20260821`、150 cycles。
- 结果（best validation Stage C total，越低越好）：AdamW/wd`1e-4`
  **`0.7162069299004294`**（best cycle `140`，KEEP）< AdamW/wd`1e-5`
  `0.71834020587531`（cycle `110`，KEEP）< Adam/wd0 `0.7222889431498267`
  （best cycle **`150`**，**RETEST**）。Adam/wd0 与 GL-C1 winner 为同配置同 seed
  重跑，但 best 不同（GL-C1 为 `0.7209096320650794` @ cycle 125），且末点 best
  触发 RETEST，按协议不得作为收敛证据；其数值仍参与排名且不影响 winner 判定。
- 决策：冻结 **AdamW / lr `1e-4` / weight decay `1e-4`** 为 GL-C2 winner 进入
  GL-C3 scheduler 搜索；不做 optimizer × scheduler 全笛卡尔积。
- 完整性：3/3 run 完成且 `status=completed`；history.csv 各 150 行、无 NaN/Inf；
  全部 `test_subjects_read=0`、`local_mca_inputs_read=40620`、git
  `f9e1250` clean；配置/协议/hash 匹配（三个 `training_report.json` 的
  `config_path`/`config_sha256` 与 `configs/experiments/*.json` 一致）。
- 产物：决策位于
  `results/sonicom_film_siren_gl_c2_optimizer_weight_decay/decision.json`；逐 run
  产物位于 `artifacts/training/sonicom_film_siren_gl_c2_*/`。
- 下一步：按已预注册的 `STAGE_C_GLOBAL_LOCAL_MCA_C3_PARALLEL_AMENDMENT.md`
  生成 GL-C3 三配置（constant/cosine/warmup_cosine 全部从 scratch 重跑，基底
  = GL-C2 winner AdamW/wd`1e-4`），并行启动并登记 PID。

## 2026-08-26：FiLM-SIREN Stage B9：global / local MCA / global+local 三模型扩展完成

- 触发原因：Stage B8 bounded gate 的 `global+local MCA` 单 seed 相对
  `global_zero_local` 改善 `12.1201%`，远超预注册的 `0.5%` 扩展门槛；因此按
  `experiments/film_siren/STAGE_B_GLOBAL_LOCAL_MCA_EXPANSION_PROTOCOL.md`
  暂停 Stage C C4，重新打开 Stage B 架构比较。
- 运行前协议：协议在新增 run 前由 commit `927ea80` 冻结；比较
  `global_zero_local`、`local_mca`、`global_plus_local_mca` 三个候选，统一使用
  7 维 query 接口、`latent128 + full + all` 容量、seeds
  `20260821/20260822/20260823`、150 cycles 和同一 validation MAE 判据。
  两个合规 gate seed-20260821 run 直接复用，新增 7 个 run；任一新增 run 的
  best cycle 等于 150 时整体转为 `RETEST`。
- 数据边界：local MCA 只能来自 Q26-derived `mca_logmag_db`；global-only 的
  local 通道恒为 0，local-only 不读取 condition encoder 输入；三者均禁止把
  non-Q26 reference 或 target 作为输入，且必须记录 `test_subjects_read=0`。
- 结果（三-seed weighted MAE）：`global_plus_local_mca`
  **`2.708064 ± 0.005515 dB`** < `local_mca`
  `2.808525 ± 0.000926 dB` < `global_zero_local`
  `3.088933 ± 0.002255 dB`。global+local 在 3/3 matched seeds 上均为第一，
  相对 global-only 改善 `12.3301%`，相对 local-only 改善 `3.5770%`；local-only
  本身相对 global-only 改善 `9.0779%`。
- 收敛与完整性：9/9 所需 run 全部完成并为 `KEEP`；三个候选的 best cycles
  分别为 global `60/75/75`、local `130/125/135`、global+local
  `70/75/60`，均早于 cycle 150，不触发 `RETEST`。全部 Git clean、finite、
  `test_subjects_read=0`；global-only 的 `local_mca_inputs_read=0`，local-only 的
  condition encoder 输入读取数为 0。
- 决策：按冻结的三-seed均值判据，Stage B 架构 winner 重新冻结为
  **`global_plus_local_mca + latent128 + full modulation + all placement`**。
  原 global-only Stage C C1--C3 结果只保留为历史对照；后续需在新架构上重新验证
  Stage C，不能直接恢复旧 C4 或沿用旧 C1--C3 winner。
- 产物：协议、7 个新增配置和聚合脚本分别位于
  `experiments/film_siren/STAGE_B_GLOBAL_LOCAL_MCA_EXPANSION_PROTOCOL.md`、
  `configs/experiments/sonicom_film_siren_b_expansion_*.json` 与
  `scripts/analyze_siren_b_global_local_mca_expansion.py`；9 个逐 run 报告位于
  `artifacts/training/sonicom_film_siren_b_{gate,expansion}_*/`。本条数字由这些
  `training_report.json` 按预注册分析器的 population mean/std 公式只读复核。

## 2026-08-26：FiLM-SIREN Stage B8：global + local MCA bounded gate 通过

- 阶段前协议规划：按总清单中“global latent 稳定后比较 global/local/
  global+local”的既定路线，在任何 local-MCA 正式 run 前提交
  `STAGE_B_GLOBAL_LOCAL_MCA_GATE_PROTOCOL.md`（commit `8b26df5`）。为控制重新
  打开 Stage B 的成本，先只比较 `global_zero_local` 与
  `global_plus_local_mca`；两者采用相同 7 维输入接口和同 seed 初始化，唯一差异
  是 query 的双耳 MCA 两通道为 0 还是真实推理时可用值。
- 冻结预算：seed `20260821`、150 cycles、normalized residual MSE、Adam
  `lr=1e-4`、weight decay 0、每 5 cycles 完整评价 44 validation 被试；若任一
  best cycle=150 则 `RETEST`，否则相对改善达到 `0.5%` 才允许扩展。
- 结果：`global_zero_local` 在 cycle 60 达到 best `3.088870 dB`；
  `global_plus_local_mca` 在 cycle 70 达到 best `2.714495 dB`。两者均为
  `KEEP`，相对改善 **`12.1201%`**，判定 `PASS_EXPAND`。
- 完整性：两个 run 均 finite、Git clean、配置/协议/hash 匹配；global-only
  记录 `local_mca_inputs_read=0`，两者均记录 `test_subjects_read=0`。
- 决策：暂停 Stage C C4，进入 Stage B9 三候选三-seed扩展；现有 Stage C
  C1--C3 结果继续保留为 global 架构结果，不改写、不外推为 global+local 结果。
- 产物：`results/sonicom_film_siren_b_global_local_mca_gate/{summary.csv,
  decision.json}`；逐 run 产物位于 `artifacts/training/sonicom_film_siren_b_gate_*/`。

## 2026-08-25：FiLM-SIREN Stage C4：objective ablation 配置冻结，因 Stage B8 gate 暂停

- 协议来源：C4 搜索空间已在 Stage C 首个正式 run 前随
  `STAGE_C_TRAINING_PROTOCOL.md` 冻结；固定继承 C3 winner，并比较 C0、
  C0+D1/D2、C0+multi-scale notch、C0+D1/D2+notch，新增项权重和频率范围均
  不随结果调整。
- 准备结果：已生成三个新增配置
  `sonicom_film_siren_c4_{d1d2,notch,d1d2_notch}_seed20260821_e150.json`；继承的
  C0 即 C3 warmup+cosine winner，不重复训练。
- 暂停原因：在任何 C4 正式 run 前，按 Stage B 原路线启动了 local MCA bounded
  gate；协议规定 gate 决策前暂停 C4。当前没有 C4 正式 training artifact，不能
  把“配置已生成”记为“实验已完成”。
- 后续条件：只有 Stage B9 重新冻结 global/local/global+local 架构并明确是否需
  重跑 Stage C 后，才能决定恢复原 C4 或在新架构上重新执行 C1--C4。

## 2026-08-25：FiLM-SIREN Stage C3：scheduler 筛选完成

- 实验目标：固定 C2 winner `AdamW / lr=1e-4 / wd=1e-4` 与 C0 objective，按
  预注册协议比较 constant、cosine、5-cycle linear warmup + cosine；统一 seed
  `20260821`、150 cycles，scheduler horizon 固定为 150。
- 结果（best validation Stage C total）：warmup+cosine **`0.824966`**
  （cycle 65）< cosine `0.825290`（cycle 65）< inherited constant
  `0.826732`（cycle 65）。三个候选均在末点评价前达到 best，均为 `KEEP`。
- 决策：冻结 **warmup+cosine（warmup 5 cycles）** 为 C3 winner，并按原协议
  生成 C4 objective ablation 配置。该结论只适用于当时的 global-only
  `latent128 + full + all` 架构。
- 完整性与产物：全部 run `test_subjects_read=0`；决策位于
  `results/sonicom_film_siren_c3_scheduler/decision.json`，新增 run 位于
  `artifacts/training/sonicom_film_siren_c3_*/`。

## 2026-08-25：FiLM-SIREN Stage C2：optimizer / weight-decay 筛选完成

- 实验目标：固定 C1 winner `lr=1e-4`、constant scheduler 和 C0 objective，
  比较 inherited Adam/wd0、AdamW/wd`1e-5`、AdamW/wd`1e-4`；统一 seed
  `20260821`、150 cycles。
- 结果（best validation Stage C total）：AdamW/wd`1e-4` **`0.826732`**
  （cycle 65）< inherited Adam/wd0 `0.827722`（cycle 90）< AdamW/wd`1e-5`
  `0.829612`（cycle 65）；三个候选均为 `KEEP`。
- 决策：冻结 **AdamW / lr `1e-4` / weight decay `1e-4`** 进入 C3 scheduler
  搜索；不做 optimizer × scheduler 全笛卡尔积。
- 完整性与产物：全部 run `test_subjects_read=0`；决策位于
  `results/sonicom_film_siren_c2_optimizer_weight_decay/decision.json`，新增 run
  位于 `artifacts/training/sonicom_film_siren_c2_*/`。

## 2026-08-25：FiLM-SIREN Stage C1：learning-rate 筛选完成

- 实验目标：在冻结的 Stage B global 架构和 C0 auditory objective 上，固定
  Adam/wd0/constant，比较 learning rate `{3e-5,1e-4,3e-4}`；三个 screening
  run 均使用 seed `20260821`、150 cycles，按未平滑的 validation objective
  total 单点最小值排序。
- 结果：`1e-4` **`0.827722`**（best cycle 90）< `3e-5` `0.829001`
  （cycle 65）< `3e-4` `0.873646`（cycle 40）；三者 best 均早于 cycle 150，
  均为 `KEEP`。
- 决策：冻结 **`lr=1e-4`** 进入 C2；较大的 `3e-4` 明显退化，较小的
  `3e-5` 与 winner 接近但未胜出。
- 完整性与产物：全部 run `test_subjects_read=0`；决策位于
  `results/sonicom_film_siren_c1_learning_rate/decision.json`，逐 run 产物位于
  `artifacts/training/sonicom_film_siren_c1_*/`。

## 2026-08-22：FiLM-SIREN Stage C0：跨被试听觉目标训练协议预注册与实现

- 阶段前协议规划：在首个 Stage C 正式 run 前提交
  `experiments/film_siren/STAGE_C_TRAINING_PROTOCOL.md` 与训练实现（commit
  `be80c91`）。架构固定为当时 Stage B winner `latent128 + full + all`，所有
  Stage C 候选从 scratch 训练，不加载 Stage B checkpoint；split 固定
  262 train / 44 validation，Q26 condition 方向不进入 767-direction query。
- C0 objective：direction-weighted normalized residual SmoothL1 + `0.75` ERB
  + `0.25` 对侧高频 + `0.75` strict HRIR ILD + `0.05` spectral-band ILD；global
  与 horizontal directions 双采样，validation 使用完整 767/72 方向，不使用
  随机 validation batch。
- 搜索顺序与上限：C1 learning rate → C2 optimizer/decay → C3 scheduler →
  C4 objective ablation；先用 seed `20260821` 做最多 10 个 unique configs，再给
  全局 Top-3 补 seeds `20260822/20260823`，Stage C 搜索最多 16 个正式 runs。
- checkpoint/最终模型规则：每 5 cycles 评价，Stage C objective total 严格变小
  才更新 best；最终三 seed 的 best cycles 中位数定义 `E_final`，三模型从 scratch
  固定训练至 `E_final` 并做 1/3 等权 residual ensemble，禁止搜索 ensemble 权重。
- 工程验证：新增 Stage C loss/训练器、配置、smoke 与单元测试；所有搜索阶段禁止
  解析或打开 SONICOM test，正式产物必须为 `test_subjects_read=0`。

## 2026-08-22：FiLM-SIREN Stage B7：condition 因果消融完成，首次冻结 Stage B

- 运行前协议：`STAGE_B_CONDITION_ABLATION_PROTOCOL.md` 在评价前由 commit
  `64df86d` 冻结；只对 B6 的三个 E150 best checkpoints 做 deterministic
  condition shuffle 与 train-mean latent 两项推理干预，不重新训练、不重新选择
  checkpoint。
- 结果（三-seed weighted MAE）：normal **`3.092959 dB`**；shuffle
  `3.157990 dB`（+`2.10%`）；train-mean latent `3.138841 dB`（+`1.48%`）。
  两项干预均在 3/3 matched seeds 上变差。
- 结论：模型确实使用了 subject-specific Q26 condition，而不只是共享 SIREN
  容量或被试无关偏置；Stage B 架构首次冻结为
  **`latent128 + full modulation + all placement`**，状态改为
  `STAGE B FROZEN / READY FOR STAGE C`。
- 完整性与产物：评价时 Git clean、`test_subjects_read=0`；结果位于
  `results/sonicom_film_siren_b_condition_ablation/{summary.csv,per_subject.csv,
  decision.json}`。该冻结后来因预注册路线中的 local-MCA gate 重新打开，原结果
  保留且不被覆盖。

## 2026-08-22：FiLM-SIREN Stage B6：conditioned / unconditional 对称 E150 复核通过

- 触发与协议修订：B5 的 unconditional seed-20260821 在 E100 末点达到 best，
  因而不得直接宣布 conditioning check 通过。先提交
  `STAGE_B_CONDITIONING_CHECK_E150_AMENDMENT.md`（commit `eeb7884`），冻结
  conditioned winner 与 unconditional baseline 各 3 seeds、全部从 scratch 的
  对称 150-cycle复核及平台判据。
- 结果：conditioned `latent128/full/all` 三-seed均值
  **`3.092959 ± 0.004507 dB`**，best cycles `45/75/65`；unconditional shared
  SIREN 为 `3.143885 ± 0.001311 dB`，best cycles `135/135/145`。6/6 runs
  全部在 cycle 150 前达到 best，均为 `KEEP`。
- 判定：conditioned 在 3/3 matched seeds 上胜出，相对改善 `1.6199%`；
  conditioning check 正式通过，解除 B5 的预算阻断，允许执行 condition 因果消融。
- 完整性与产物：全部 `test_subjects_read=0`，unconditional 另记录
  `condition_inputs_read=0`；结果位于
  `results/sonicom_siren_b_conditioning_check_e150/{summary.csv,decision.json}`。

## 2026-08-22：FiLM-SIREN Stage B5：unconditional shared-SIREN 对照完成，但 E100 预算阻断

- 阶段前协议规划：在 baseline 正式训练前提交
  `STAGE_B_UNCONDITIONAL_BASELINE_PROTOCOL.md`（commit `8a55699`）；对照使用
  相同冻结 backbone、262/44 split、训练 query、seeds 和 100-cycle预算，但完全
  不构造/读取 condition 输入。
- 结果：unconditional 三-seed均值 `3.147294 ± 0.003072 dB`；matched
  conditioned all-placement 为 `3.092207 ± 0.006925 dB`，三 seed 全胜，表面
  相对改善 `1.7503%`。
- 阻断：unconditional seed-20260821 的 best 位于 cycle 100，按预注册规则标为
  `RETEST`；另外两个 best cycles 为 95/90。故此时
  `conditioning_check_passed=null`、Stage B 保持 `PENDING`，不得先做 condition
  ablation 或进入 Stage C。
- 产物：`results/sonicom_shared_siren_b_unconditional/{summary.csv,
  decision.json}`；全部 run `test_subjects_read=0`，baseline 的
  `condition_inputs_read=0`。

## 2026-08-22：FiLM-SIREN Stage B4：placement 五-seed RETEST 完成并冻结 all

- 运行前协议：针对 B3 的 seed 排名反转，先提交
  `STAGE_B_PLACEMENT_RETEST_PROTOCOL.md`（commit `4da1eff`）；hidden/all 各新增
  seeds `20260824/20260825`，不改变 full modulation 边界、正则或 100-cycle
  预算，最终只按五-seed等权均值一次性决胜。
- 结果：all `3.091073 ± 0.005545 dB`，5 seeds 中胜 4；hidden
  `3.093648 ± 0.002327 dB`，胜 1。配对均值 `all-hidden=-0.002575 dB`
  （all 约优 `0.083%`），exact bootstrap 95% 区间
  `[-0.005852,0.001622] dB` 跨 0，仅作稳定性诊断。
- 决策：按冻结的均值判据选择 **all placement**；优势很小，日志保留“统计稳定性
  证据有限”的解释边界，不把工程决胜规则表述成显著性结论。
- 完整性：10/10 所需 run（含复用的 6 个 B3 run）均为 `KEEP`，无末点 best，
  `test_subjects_read=0`；结果位于
  `results/sonicom_film_siren_b_placement/{retest_summary.csv,retest_decision.json}`。

## 2026-08-20：FiLM-SIREN Stage B3：placement 搜索完成，因 seed 排名反转进入 RETEST

- 阶段前规划：B2 选出非 concat 的 full modulation 后，按 Stage B 预注册顺序
  比较 late（4--6 层）、hidden（2--6 层）、all（1--6 层）；screening seed
  `20260821`，Top-2 再补 seeds `20260822/20260823`。
- 单 seed：hidden `3.096799 dB` < all `3.101953 dB` < late
  `3.121709 dB`；hidden/all 差距小于 `0.5%`，Top-2 为 hidden/all。
- 三 seed：all `3.092207 ± 0.006925 dB` < hidden
  `3.094049 ± 0.002161 dB`，但 screening seed 是 hidden 胜、另两个 seeds 是
  all 胜，且均值差仅约 `0.060%`。
- 决策：严格执行协议“seed 排名反转则 RETEST”，输出 `winner=null`；没有按单
  seed 冻结 hidden，也没有跳过阻断按三-seed均值直接冻结 all。六个 run 的 best
  均早于 cycle 100，故问题是配置排序稳定性而非预算不足。
- 诊断与产物：记录 all/hidden 多层 amplitude/gamma saturation，结果位于
  `results/sonicom_film_siren_b_placement/{summary.csv,decision.json}`，全部
  `test_subjects_read=0`。

## 2026-08-20：FiLM-SIREN Stage B2：modulation 搜索完成并冻结 full

- 阶段前规划：固定 B1 winner latent=128 和 hidden placement，比较
  concat、amplitude、phase、full；screening seed `20260821` 选 Top-2，再补
  seeds `20260822/20260823`。配置与分析器在正式运行前由 commit `a50c212`
  提交。
- 单 seed 排名：full `3.092576 dB` < phase `3.108488 dB` < amplitude
  `3.121326 dB` < concat `3.180237 dB`；Top-2 为 full/phase，差距不 tight。
- 三 seed：full **`3.091884 ± 0.002857 dB`**，phase
  `3.108313 ± 0.000850 dB`；三 seed 排名无反转，所有 best cycles 早于 100。
- 决策：冻结 **full modulation**，不触发 concat 胜出时的 placement 跳过分支，
  继续 B3 placement 搜索。
- 完整性与产物：8 个正式 run 全部 `KEEP`、finite、Git clean、
  `test_subjects_read=0`；结果位于
  `results/sonicom_film_siren_b_modulation/{summary.csv,decision.json}`。

## 2026-08-20：FiLM-SIREN Stage B1：latent dimension 搜索、预算修订与 E150 冻结

- 阶段前规划：按 Stage B 协议固定 phase × hidden placement，先以 seed
  `20260821` 比较 latent `{64,128,256}`，Top-2 补 seeds
  `20260822/20260823`；只有 256 第一且领先第二不足 1% 才扩展到 384/512。
- E100 screening：128 `3.111088 dB` < 256 `3.112155 dB` < 64
  `3.112587 dB`，三者差距很小；Top-2 为 128/256，未触发 384/512 分支。
  三-seed E100 均值为 128 `3.107508 dB`、256 `3.108659 dB`，但存在末点 best，
  因此 provisional winner 不得直接冻结。
- 预算修订：先后提交 `STAGE_B_LATENT_BUDGET_AMENDMENT.md` 与
  `STAGE_B_LATENT_PLATEAU_AMENDMENT.md`（commits `522f445`、`0521e2f`），
  冻结 128/256 × 3 seeds 全部从 scratch 的 E150 对称复核，以及末点 best 的
  前后窗口平台判据；修订发生在对应 E150 决策前，不覆盖 E100 记录。
- E150 结果：128 **`3.103014 ± 0.003435 dB`** < 256
  `3.105485 ± 0.001499 dB`；6/6 runs 完整。两个 cycle-150 best 分别按冻结平台
  规则判为 `KEEP_PLATEAU`，其余为 `KEEP`，不再要求延长预算。
- 决策：冻结 **latent dimension=128** 进入 B2；全部 run
  `test_subjects_read=0`。结果位于
  `results/sonicom_film_siren_b_latent/{summary.csv,decision.json,
  e150_summary.csv,e150_decision.json}`。

## 2026-08-20：FiLM-SIREN Stage B0：conditioning 搜索协议预注册与基础设施完成

- 编号说明：Stage B 协议原文按 latent / modulation / placement / 对照与因果检查
  描述顺序，没有给所有后续修订单独编号；为满足实验账本逐步检索，本日志按实际
  执行和阻断顺序记为 B0--B9。编号只用于日志索引，不追溯改名协议文件或 run。
- 阶段前协议规划：在任何 Stage B 正式 search run 前提交
  `experiments/film_siren/STAGE_B_FILM_PROTOCOL.md`（commit `9ef88db`），冻结
  Stage A4 backbone `D-o20-d6-w256-ho20`、262 train / 44 validation split、
  767 个 interpolation query、Q26-only condition 边界、100-cycle基础预算、
  主指标、seed顺序、Top-2规则、最大 run 数和失败处置。
- 搜索边界：Stage B 只搜索 conditioning architecture，loss/optimizer/scheduler
  留给 Stage C；顺序固定为 B1 latent → B2 modulation → B3 placement →
  unconditional/因果检查，不做全笛卡尔积。Stage A checkpoint 不复用，SIREN、
  encoder 与 modulator 全部从 scratch 联合训练。
- 数据接口：实现 `Q26Condition`、train-only Q26 normalization、DeepSets
  condition encoder、FiLM-SIREN、共享 residual predictor/metric 与 Stage B
  trainer；condition 只能读取 26 个测量方向的双耳 reference magnitude、xyz、
  mask 和 frequency，监督 target 走隔离接口。
- 审计：direction permutation、padding/mask、all-masked、non-Q26 mutation、
  target mutation、NaN isolation、API leakage、predictor adapter、metric 与
  CPU/CUDA smoke 均在正式搜索前覆盖；SONICOM test 禁止解析/打开，正式 run
  统一记录 `test_subjects_read=0`。

## 2026-08-20：FiLM-SIREN Stage A4：confirmation 通过，冻结 SIREN Backbone v1

- 实验目标：按 `experiments/film_siren/STAGE_A4_CONFIRMATION_PROTOCOL.md`
  用 32 名未参与 A2/A3 配置选择的 train 被试，验证 backbone 配置结论；
  4 套配置（M 主候选 D-o20-d6-w256-ho20、L 历史基线 linear-o30、D A2 对照
  D-o30、A d4 紧咬对照）× 32 被试 × 250 epochs = 128 run。
- 预注册：协议（4 套配置/32 被试锁定/250 epochs/通过标准）在运行前冻结
  （commit `124c267`）；32 名被试按 `Avg RMS dB` 8 层 × 4 人锁定
  （`siren_a4_confirmation_subjects_v1.csv`，排除 A2 的 5 人、零重叠）；
  代码修正随本阶段生效（GradScaler 移入循环、checkpoint 字段命名、
  run_matrix 人数放宽、per-subject configuration.json）。
- 结果（32 人 aggregate holdout RMSE）：M `2.8141`、L `3.1733`、D `2.9627`、
  A `2.8545` dB。**预注册三项标准全部通过**：主标准 M≤L（0.359 dB，配对
  t-test p≈0）、次标准 M≤D（0.149 dB，p≈0）、绝对合理性
  2.4≤2.8141≤3.4；信息性 M vs A 差距 1.44%（d6≈d4 紧咬保持）。
- 结论：A2/A3 结论在新被试上全部保持（dual 优于 linear +11.3%、first-omega
  20 优于 30 +5.3%、紧咬保持），选择偏差未导致反转，M 绝对水平 2.81 dB
  优于 A3 五人值 2.90。**冻结 `SIREN Backbone v1 = D-o20-d6-w256-ho20`**
  （250 epochs）。
- 完整性：128/128 run 全部 KEEP、无 NaN/Inf、`test_subjects_read` 恒为 0。
- 产物：聚合与判定 `results/sonicom_siren_a4_confirmation/{summary.csv,
  decision.json}`；报告 `reports/film_siren_siren_a4_confirmation.md`；
  逐 run 记录 `artifacts/training/sonicom_siren_a4_c{1..4}/`。
- 下一步：Stage B（Q26 condition 数据接口 + FiLM conditioning），
  基于冻结的 `SIREN Backbone v1`。

## 2026-08-20：FiLM-SIREN Stage A3：backbone 深化搜索完成，候选 D-o20-d6-w256-ho20

- 实验目标：按 `experiments/film_siren/STAGE_A3_PROTOCOL.md` 在 A2 选出的
  D-o20 上逐维搜索 depth → width → hidden-omega（3 候选 × 5 名 A2 被试，
  预算 200 epochs），得到深化 backbone 候选供 confirmation。
- 预算验证（A3-0）：`sonicom_siren_a3_budget_e200` 的 best epochs
  `200/189/150/128/133`，中位数 M=150；200-epoch aggregate holdout RMSE
  `2.9350` 相对 A2 100-epoch `2.9865` 改善 `1.726%`（≥0.5% 阈值）→ 按协议
  冻结 **E_A3 = 200**。P0289 在 200 epoch 仍未平台（best=200），报告注明
  "预算边缘，confirmation 建议更长预算复核"。
- A3-1 depth：d6 `2.9350` < d4 `2.9472`(+0.42%) < d8 `3.0112`(+2.60%)；
  触发紧咬分支（两名 <0.5%、第三名 >2%）→ d6 主线、d4 保留对照；协议补充
  执行说明（紧咬时后续深化取第一名主线、第二名对照，控制 2× 成本）。
- A3-2 width：w256 `2.9350` < w128 `3.0603`(+4.27%) < w512 `3.3763`(+15.04%)；
  基线 256 最优（w128 欠容量、w512 过拟合），非紧咬 → w256 主线。
- A3-3 hidden-omega：ho20 `2.8956` < ho30 `2.9350`(+1.36%) < ho50
  `3.3653`(+16.22%)；非紧咬 → ho20 主线。与 first-omega 20 最优趋势一致
  （更低 omega 更利于方向外推）。
- 最终候选：**`D-o20-d6-w256-ho20`**（dual、first-omega 20、6 层、width
  256、hidden-omega 20，200 epochs），aggregate holdout RMSE **`2.8956 dB`**，
  相对 A2 粗筛（100 ep）改善 `3.045%`，相对 200-epoch 基线再改善 `1.343%`；
  full-field RMSE `1.7160`。
- 完整性：50 个 subject run（budget 5 + depth 15 + width 15 + ho 15）全部
  完成，46 KEEP + 4 RETEST（4 个均为 P0289 在 200 epoch 预算末端，best=200，
  收敛慢现象），无 NaN/Inf，`test_subjects_read` 恒为 0。
- 产物：三阶段排名/决策
  `results/sonicom_siren_a3_matrix/{depth,width,ho}/`；汇总报告
  `reports/film_siren_siren_a3_matrix.md`；逐 run 记录
  `artifacts/training/sonicom_siren_a3_*/`。
- 下一步：confirmation（16–32 名新 train 被试复核）→ 冻结 `SIREN Backbone v1`
  → Stage B（Q26 FiLM conditioning）。

## 2026-08-19：FiLM-SIREN Stage A2：五人 backbone matrix 联合粗筛完成，dual-o20 锁定为主配置

- 实验目标：按 `experiments/film_siren/STAGE_A2_PROTOCOL.md` 预注册协议，在 5 名
  固定 train 被试上对 plain SIREN backbone 做 `frequency {linear,erb,dual} ×
  first-omega {20,30,50}` 联合粗筛（9 配置 × 5 被试 = 45 run），以固定 64 方向
  direction-holdout 的 residual RMSE 为主指标选 Top-2。
- 预注册与锁定：协议、搜索空间、预算（100 epochs × 100 steps）、Top-2 规则在
  首次运行前冻结并提交（commit `8028932`）；5 名被试
  `P0289/P0346/P0085/P0076/P0010` 按 `Avg RMS dB (Free Field)` 五层分层（seed
  `20260819`）锁定于 `configs/data/siren_a2_subjects_v1.csv`；64 个纯插值 holdout
  方向（solid-angle 分层、SHA-256 强制校验）锁定于
  `configs/data/siren_a2_holdout_v1.csv`。
- 训练入口：`mcar.training.train_siren` 新增 `multi_subject_siren_matrix` 类型
  （每被试独立 plain SIREN，训练 729 方向、每 epoch 评估 64 方向 holdout，输出
  history/checkpoint/training_report，含 SHA-256 与 git state）；45 run 全部
  exit 0、全部 `KEEP`（best epoch 73–98，均早于预算末端）、无 NaN/Inf、
  `test_subjects_read` 恒为 0。
- 排名（aggregate holdout RMSE，dB）：**D-o20 2.9865** < D-o30 3.0732 <
  L-o20 3.0863 < L-o30 3.2427 < D-o50 3.3364 < E-o20 3.4246 < L-o50 3.5607 <
  E-o30 3.6512 < E-o50 3.9772。dual 映射整体最优，first-omega 20 在 dual 与
  linear 下均一致优于 30/50，ERB 最差。
- Top-2 决策（协议第 6 节规则）：D-o20 领先 D-o30 2.90%（≥0.5%），领先第三名
  3.34%（>2%），因此推进 **D-o20 为主配置、D-o30 为对照**进入 Stage A3
  （depth/width/hidden-omega 搜索草案：
  `experiments/film_siren/STAGE_A3_PROTOCOL_DRAFT.md`）。
- 产物：逐 run 记录在 `artifacts/training/sonicom_siren_a2_*/`；排名表
  `results/sonicom_siren_a2_matrix/summary.csv`、决策
  `results/sonicom_siren_a2_matrix/top2.json`、报告
  `reports/film_siren_siren_a2_matrix.md`。
- 说明：A2 是配置相对排序实验，holdout 为训练未见方向，误差高于 A1 的
  P0002 全方向拟合（~1.50 dB）属预期；不解释为跨被试或方向泛化结论。

## 2026-08-18：FiLM-SIREN Stage A1：P0002 单被试纯 SIREN 表示上限测试完成

- 实验目标：启动 FiLM-SIREN 新主模型路线的 Stage A1，首先验证不带任何 subject
  condition 的纯坐标 SIREN 是否具有足够能力表示单个 SONICOM 被试的完整 MCA residual
  field。本阶段只验证表示能力和数值稳定性，不用于选择最终 frequency coordinate、
  omega、depth 或 width，也不代表跨被试或未见方向泛化性能。
- 协议状态说明：A1 是 representation smoke/ceiling，不是正式架构搜索；初始设置由
  `sonicom_siren_single_subject_baseline_v1.json` 固定，但项目总清单
  `EXPERIMENT_CHECKLIST.md` 是随 A1-v1 结果在 commit `3c31740` 中落库，故不将其
  倒称为 A1-v1 的“运行前预注册”。A1-v2 只按 v1 的末点 best 证据把预算从 20
  epochs 提高到 100，结构、seed、采样和 optimizer 均不变；正式搜索从 A2 的独立
  运行前预注册协议开始。
- 基础结构：固定 SONICOM train 被试 `P0002`，查询坐标为
  `[x,y,z,f_linear]`，频率 `86.1328125–19982.8125 Hz` 线性映射到 `[-1,1]`；
  SIREN hidden width `256`、`6` 个 sine layers、first/hidden omega 均为 `30`，
  双耳联合输出两个 normalized residual channels。模型共 `330754` 个参数；
  optimizer 为 Adam，学习率 `1e-4`，weight decay `0`，FP32，seed `20260818`。
  单 epoch 为 `100 steps`，每 step 随机采样 `16` 个方向并使用完整 `463` 个频点。
- A1-v1 smoke run：首先运行 `20 epoch × 100 step`。完整 residual field 的
  MAE/RMSE 从初始化的 `3.3205 / 5.0429 dB` 降至
  `1.1645 / 1.8520 dB`；`>8 kHz`、`>10 kHz` MAE 分别降至
  `1.3100 / 1.3366 dB`，first spectral-difference 降至
  `0.4203 dB/bin`，notch-depth MAE 降至 `0.3894 dB`。训练无 NaN/Inf，
  但最佳 checkpoint 位于最后一个 epoch，说明 20 epoch 尚不足以作为表示上限，
  因此登记为 `RETEST`。
- A1-v2 长预算确认：保持结构、随机种子、采样协议和优化器完全不变，从头重新训练
  `100 epoch × 100 step`。最佳 checkpoint 位于 **epoch 98**，完整 residual
  field MAE/RMSE 达到 **`0.936514 / 1.498697 dB`**；`>8 kHz` 和
  `>10 kHz` MAE 分别为 **`1.041481 / 1.063229 dB`**，
  first spectral-difference 为 **`0.382707 dB/bin`**，
  notch-depth MAE 为 **`0.318079 dB`**。相对 20-epoch v1，
  residual MAE/RMSE 分别进一步改善约 `19.6% / 19.1%`，高频 MAE 进一步改善
  约 `20%`，说明延长训练预算确有必要。
- 收敛判断：epoch 91 的 full-field RMSE 为 `1.525809 dB`，epoch 100 为
  `1.511698 dB`，最后 10 epoch 总下降仅 `0.9248%`；同时曲线在约
  `1.50–1.53 dB` 范围内波动，最佳点为 epoch 98 而非预算末端。因此认为当前
  linear-frequency / omega-30 SIREN 已达到足以完成 Stage A1 的**实际优化平台**，
  不再继续针对 P0002 单人追加 200 epoch。
- 谱形状诊断：second spectral-difference 从初始化的 `0.255108 dB/bin²`
  上升至 `0.270947 dB/bin²`，且相对 20-epoch v1 的 `0.2647 dB/bin²`
  仍略有回退。说明 residual MSE 能显著改善整体幅度、高频误差和 notch depth，
  但不会自动保证二阶局部谱形状指标同步改善。当前不因此修改 loss，而将该指标保留为
  后续 frequency/omega 搜索和 spectral-loss 消融的 guard metric。
- 运行与完整性：A1-v2 总训练/评价用时 `48.04 s`，CUDA 峰值显存
  `121.95 MiB`，完整 `793×463` query 推理约 `31.28 ms`；全程无 NaN/Inf，
  test subject 读取数为 `0`。最佳 checkpoint SHA-256 为
  `FCFF8B909D3040E44720F59C98EEC8AD5EEE6094F92899E2EE41C00317EBEA20`。
  运行时 Git 分支为 `codex/project-structure-refactor`，HEAD 为
  `3c3174022fa313dda7c8058f71d6cf39d085f93f`；运行时工作区记录为 `dirty=true`，
  因此进入下一阶段前需提交本轮 config、报告和文档，使正式 backbone 搜索从 clean
  Git state 开始。
- 阶段结论：**`KEEP — A1 COMPLETE`**。纯 SIREN 已证明能够稳定、高精度地拟合
  单个 SONICOM 被试的完整 MCA residual field；继续针对 P0002 优化的科研价值有限。
  下一阶段转入固定多名 train subjects 的 backbone 搜索，并联合粗筛
  `frequency coordinate × first omega`，避免把单个被试的特殊频谱结构固化为最终
  SIREN 超参数。
- 产物：A1-v1 配置与报告位于
  `configs/experiments/sonicom_siren_single_subject_baseline_v1.json` 和
  `reports/film_siren_siren_a1_p0002_baseline_v1.md`；A1-v2 配置与详细报告位于
  `configs/experiments/sonicom_siren_single_subject_baseline_v2.json` 和
  `reports/film_siren_siren_a1_p0002_baseline_v2.md`。训练产物位于
  `artifacts/training/sonicom_siren_a1_p0002_linear_w256_d6_o30_v1/` 与
  `artifacts/training/sonicom_siren_a1_p0002_linear_w256_d6_o30_v2/`。

## 2026-08-17：MCAR v3.5.1 晋升为当前工程主模型及八方法横向对比

- 主模型决策：用户明确决定将冻结的 **MCAR v3.5.1** 设为新的当前工程主模型，
  替代 v3.2 seed-20260809 epoch 39。模型仍固定为 previous-joint `30%` + B 支
  epoch 23 `70%` 的 residual 输出融合，不重新训练、不调整融合权重。
- 横向对比：从已冻结的八方法逐被试结果中原样保留七个基线，仅以 v3.5.1 的冻结逐被试
  结果替换旧 MCAR v3.2 e39 行；合并阶段不再读取 test、不运行模型推理。合并后为
  `44 被试 × 8 方法 × 4 指标 = 1408` 行，缺失/非有限值均为 0，七个基线逐值不变。
- 四指标结果与排名：全空间 ERB `0.817937 ± 0.175483 dB`（第 1），对侧 25° ERB
  `1.274489 ± 0.147744 dB`（第 1），对侧高频 `3.508481 ± 0.287913 dB`（第 3），
  水平面 ILD MAE `0.644709 ± 0.247907 dB`（第 1）。高频项低于 v3.5.1 的是
  FSP-AE 和 RANF；其余三项 v3.5.1 均为八方法最优。
- 产物：规范清单位于
  `configs/experiments/sonicom_eight_method_q26_v351_main_test.json`，完整数据表、质量检查、
  排名表与汇总位于 `results/sonicom_eight_method_q26_v351_main_test/`，2×2 横向条形图
  同时导出 PNG 和矢量 PDF。历史预声明的 v3.2 epoch 6 论文模型记录不被静默改写；
  validation/test 均已被项目历史使用，独立确认仍需新拆分或外部数据。

## 2026-08-17：SONICOM MCAR v3.5.1 冻结测试与 v3.2 epoch 39 对比

- 冻结协议：用户明确授权后，测试前锁定 v3.5.1 的 previous-joint/B-epoch23
  checkpoint SHA-256 与 `30/70` 输出融合比例，禁止参数、checkpoint 或权重更新；
  对照为此前定义的当前主模型 v3.2 seed-20260809 epoch 39。SONICOM test 已被历史
  冻结评估消费，本次是追加冻结比较，不宣称为全新未见测试集确认。
- 推理与完整性：两个 v3.5.1 分量分别完成 44/44 test residual 推理，固定融合后也是
  44/44；每个预测均为 `2×793×463`。严格重建完成 44 人、每人 767 个纯插值方向、
  72 个水平面方向和 41 个 ERB 频带。逐被试结果表为 `44×22`、SubjectID 唯一、
  缺失值 0、数值全有限，三个冻结 checkpoint 的 SHA-256 前后均不变。
- 测试结果：v3.2 e39 的四项为
  `0.855676 / 1.349975 / 3.590454 / 0.660297 dB`；v3.5.1 为
  `0.817937 / 1.274489 / 3.508481 / 0.644709 dB`，分别改善
  `4.410% / 5.592% / 2.283% / 2.361%`，逐被试改善人数为 `43/42/44/26`。
- 配对统计：前三项 paired t-test 为 `p=1.05e-16 / 2.89e-11 / 2.63e-21`，
  Holm 校正后仍显著；对应 Wilcoxon 也显著。水平面 ILD 的 paired t-test
  `p=0.0883`、Wilcoxon `p=0.0874`，均值更低但不显著，应解释为测试集上方向一致的
  小幅改善而非确定性提升。
- 结论与产物：v3.5.1 在冻结测试的三个频谱指标上稳定、显著优于当时的 v3.2 e39，
  ILD 不回退且均值改善。本次运行结束时先登记为更强的工程主模型候选；用户随后在
  同日明确将其晋升为当前工程主模型。论文中仍需注明该 test 已被项目历史实验消费。协议位于
  `configs/experiments/sonicom_mlp_cnn_q26_v351_frozen_test.json`，完整重建、质量检查、
  MATLAB/SciPy 交叉核验和逐被试统计位于
  `results/sonicom_mlp_cnn_q26_v351_vs_v32e39_frozen_test_strict/`。

## 2026-08-17：SONICOM MCAR v3.5.1：三分支低学习率续训与重新融合

- 实验目标：从 v3.5 的 scratch 分量 epoch 60 分叉，比较原损失续训、弱频带 ILD
  与更强 strict+band ILD 三种协议。三支均联合解冻全部 `174627` 个 MLP/CNN 参数，
  使用 MLP/CNN 学习率 `3e-5/1e-5`、固定 train/validation seed 和
  `40 epoch × 500 step`；全程只读 262 train / 44 validation，test 读取数为 0。
- 训练结果：A/B/C 的训练目标最佳 epoch 分别为 `23/23/1`。统一换回 v3.5 原始
  objective 后，三支综合最佳 checkpoint 的 total 分别为
  `0.646327 / 0.646199 / 0.646672`，说明额外 band-ILD=0.05 的 B 支略优，而
  strict-ILD 提高到 0.9 的 C 支没有带来持续收益。三支源 checkpoint 前后 SHA-256
  均保持 `B7DE9642...2F6` 不变。
- 融合重选：对六个 checkpoint（每支 total 最优和 strict-ILD 最优）统一筛选，
  入围者再与固定 previous-joint 分量搜索凸组合。最佳仍为 previous `30%` + B 支
  epoch 23 `70%`，proxy total 为 `0.640584`，相对 v3.5 的 `0.641223` 改善
  `0.0997%`。B checkpoint SHA-256 为 `3D657783...391EB`。
- 严格 validation：v3.5.1 的全空间 ERB、对侧 25° ERB、对侧高频和水平面 ILD 为
  `0.830808 / 1.288953 / 3.538087 / 0.581146 dB`，相对 v3.5 分别改善
  `0.124% / 0.235% / 0.061% / 0.033%`。前三项 paired t-test 为
  `p=1.63e-9 / 0.0265 / 2.17e-4`，改善人数为 `39/25/29`；ILD 为 21/44 人改善且
  不显著（`p=0.854`），应解释为基本持平。
- 结论：该 30/70 输出融合登记为 **MCAR v3.5.1**，替代 v3.5 成为当前
  validation-best 工作候选，但改善绝对值很小且验证集已被重复用于选择，不据此改写
  冻结论文主模型。规范 manifest 为
  `configs/experiments/sonicom_mlp_cnn_q26_v351.json`；proxy 和严格结果分别位于
  `results/sonicom_mlp_cnn_q26_v351_proxy_ensembles/` 与
  `results/sonicom_mlp_cnn_q26_v351_candidate_strict_validation/`。

## 2026-08-16：SONICOM MCAR v3.5：MLP+CNN 完整 scratch 训练与 30/70 输出融合

- 实验目标：检验完全不保留 v2 checkpoint 时，MLP 与 CNN 同时从零学习是否可行，
  并将其与上一轮“保留 v2 MLP、CNN 重初始化后联合解冻”的最佳结果做输出级融合。
  两模型由不同随机初始化得到，隐藏单元不存在参数级一一对应，因此只融合 predicted
  residual，不平均 checkpoint 权重。全流程只用 train/validation，test 读取数为 0。
- scratch 协议：MLP/CNN 隐藏层使用原生随机初始化，两个输出投影严格置零；全部
  `174627` 个参数共同训练。MLP/CNN 目标学习率为 `1e-3/3e-4`，前 2 epoch 线性
  warmup 后 cosine decay；固定 `seed=20260809`、validation sampler `20260805`，
  使用 v3.2 双采样和 `ERB/HF/strict-ILD=0.75/0.25/0.75`，训练
  `60 epoch × 500 step`。
- scratch 训练：最佳 total 位于 epoch 60，为 `0.646540`；strict-ILD 单项最佳位于
  epoch 35，为 `0.578411 dB`。训练共用时 `6745.61 s`，8/30000 个 optimizer step
  因 AMP 非有限梯度跳过，最终 scale 为 `32768`，显存峰值 `599.47 MiB`。零输出起点
  total 为 `0.918035`，确认训练未读取任何预训练权重。
- scratch 严格 validation：全空间 ERB、对侧 25° ERB、对侧高频和水平面 ILD 为
  `0.837668 / 1.302105 / 3.542947 / 0.598536 dB`。相对上一轮联合模型分别改善
  `5.419% / 6.226% / 3.007% / 0.259%`；前三项为 42--44/44 人改善且配对统计
  显著，ILD 差异不显著。协议同时改变了初始化、总预算和学习率，因此该差异不能单独
  归因于 scratch 初始化。
- 融合选择：在固定 validation sampler 的 96 batches 上，以 0.05 局部步长搜索
  scratch 权重；最佳为上一轮联合模型 `30%` + scratch `70%`，proxy total
  `0.641223`，优于上一轮的 `0.672247` 与 scratch 的 `0.646540`。44 人完整 residual
  由同一权重线性组合后再进行原相位严格重建。
- 融合严格 validation：四项为
  `0.831843 / 1.291992 / 3.540258 / 0.581337 dB`。相对 scratch 分别改善
  `0.695% / 0.777% / 0.076% / 2.874%`；全空间 ERB、对侧 ERB、ILD 的 paired
  t-test 为 `p=1.52e-11 / 3.65e-4 / 1.20e-4`，高频差异不显著（`p=0.255`）。
  相对上一轮联合模型四项分别改善 `6.077% / 6.954% / 3.081% / 3.125%`。
- 结论与产物：30/70 输出融合正式记为 **MCAR v3.5**，是本组 validation-only 实验的最佳结果，但不追加消费
  已使用过的 SONICOM test，也不改写论文主模型。scratch 配置与运行器为
  `configs/experiments/sonicom_mlp_cnn_q26_v32_scratch_joint_e60.json`、
  `scripts/run_v32_scratch_joint_e60.py`；融合配置、脚本、权重 sweep、严格结果和合并表
  v3.5 的规范 manifest 为 `configs/experiments/sonicom_mlp_cnn_q26_v35.json`；源实验配置和脚本
  分别位于 `configs/experiments/sonicom_mlp_cnn_q26_v32_previous_scratch_fusion_v1.json`、
  `scripts/fuse_v32_previous_scratch_validation.py`、
  `results/sonicom_mlp_cnn_q26_v32_previous_scratch_fusion/`、
  `results/sonicom_mlp_cnn_q26_v32_previous_scratch_fusion_strict_validation/` 和
  `results/sonicom_mlp_cnn_q26_v32_cnn_reinit_joint_v1/`。

## 2026-08-16：SONICOM v3.2 CNN 重初始化与 MLP 低学习率联合微调

- 实验目标：检验当前 v3.2 是否依赖正式 v3 CNN 初始化，并在同一重初始化起点上用
  matched control 判断完整解冻 MLP 的独立价值。全流程仅使用 262 train / 44
  validation，被消费的 test 未读取。
- 阶段 1：正式 v2 MLP 保持冻结，不加载任何 CNN checkpoint；CNN 内部层采用原生
  初始化且输出投影严格置零。固定 `seed=20260809`、validation sampler
  `20260805`，按当前 v3.2 双采样与 `ERB/HF/strict-ILD=0.75/0.25/0.75`
  训练 `40 epoch × 500 step`。最佳点为 epoch 36，validation total 为
  `0.673068`，用时 `4551.36 s`，3/20000 个 AMP optimizer step 跳过。
- 阶段 2：从阶段 1 的同一 SHA-256 `A684F414…82DAF68B9` checkpoint 分叉。
  CNN-only 对照和 MLP+CNN 联合分支均训练 `10 epoch × 500 step`，CNN 学习率均为
  `1e-5`；联合分支额外以 `1e-6` 解冻全部 MLP。两支最佳点均为 epoch 8，total
  分别为 `0.672634 / 0.672247`，无 optimizer skip，源 checkpoint 前后哈希一致。
- 严格 validation：已有 v3 CNN 初始化的 epoch 39 四项为
  `0.872882 / 1.364204 / 3.608567 / 0.596141 dB`；CNN 重初始化为
  `0.884641 / 1.384283 / 3.655912 / 0.602080 dB`，分别回退
  `1.347% / 1.472% / 1.312% / 0.996%`。前三项的配对统计一致显著，说明 v3
  CNN 初始化具有明确价值。
- 解冻判断：联合分支四项为
  `0.885664 / 1.388553 / 3.652797 / 0.600091 dB`。相对匹配的 CNN-only
  `0.886132 / 1.387762 / 3.653945 / 0.601172 dB`，全空间 ERB、高频和 ILD
  改善 `0.0528% / 0.0314% / 0.1797%`；对侧 ERB 回退 `0.0570%`，95% CI
  跨 0。MLP 相对 L2 仅变化 `0.1179%`，表明低学习率解冻稳定且略有价值，但不能
  弥补 CNN 重初始化差距。
- 结论与产物：保持已有 v3 CNN 初始化路径，不以本实验替换主模型；联合微调保留为
  正向但小幅的消融。锁定配置与运行器分别为
  `configs/experiments/sonicom_mlp_cnn_q26_v32_cnn_reinit_joint_v1.json` 和
  `scripts/run_v32_cnn_reinit_joint_v1.py`；精选比较、配对统计和参数诊断位于
  `results/sonicom_mlp_cnn_q26_v32_cnn_reinit_joint_v1/`。

## 2026-08-14：SONICOM 八方法稀疏度横向比较

> 状态：有效。该实验在此前 MCAR v3.2、FSP-AE、RANF 三种学习方法稀疏度实验的基础上，
> 补齐五种横向基线，形成统一的八方法比较；此前三种学习方法的逐受试者结果原样复用。

- 冻结设计：固定 SONICOM test 的 44 名受试者、嵌套 `Q=6/14/26` 输入方向，以及排除
  完整 Q26 后的 767 个共同评价方向。新增方法为 SH only、SUpDEq + SH、SUpDEq +
  Natural Neighbor、SUpDEq + Barycentric 和 MCA；四项指标及统计口径与三学习方法实验一致。
- 完整性：五基线正式计算产生 `44 x 3 x 5 x 4 = 2640` 条有限指标，132 个受试者-稀疏度
  质量记录的评价方向数均为 767，Natural Neighbor/Barycentric 对观测 HRTF 的最大改写误差
  分别为 `4.58e-16/1.13e-15`。Q26 与既有正式横向结果逐项复核，最大差异为
  `9.85e-8 dB`。合并后八方法长表为 `44 x 3 x 8 x 4 = 4224` 行；原有 1584 条学习方法
  结果键完全一致、最大数值差为 `0 dB`。
- 极稀疏端点：RANF 在 Q6 的四项均值均排名第一，依次为
  `1.210/1.575/3.997/0.998 dB`。MCAR 相对各指标最强的 Q6 横向基线分别降低
  `0.330/0.450/0.576/0.281 dB`，10,000 次配对受试者 bootstrap 均为双侧
  `p=0.0002`。FSP-AE 仅在对侧半球高频项优于最强横向基线 `0.683 dB`，其余三项分别
  高 `1.252/0.733/0.571 dB`。
- 稀疏退化：RANF 的 Q6-Q26 退化依次为 `0.147/0.017/0.527/0.222 dB`，四项均小于
  MCAR；MCAR 减 RANF 的退化差为 `0.890/0.325/1.371/0.650 dB`，95% bootstrap
  区间均不跨 0。SH only 的若干曲线虽为负斜率，但其绝对端点误差较高，不能据此解释为
  有效的稀疏鲁棒性。
- 产物：冻结配置与协议分别为 `configs/experiments/sonicom_eight_method_sparsity_v1.json`
  和 `docs/SONICOM_EIGHT_METHOD_SPARSITY_PROTOCOL.md`；五基线原始结果位于
  `results/sonicom_five_baseline_sparsity_v1/`，八方法主表、排名、全部 224 个配对对比、
  120 个学习方法对基线对比和主图位于 `results/sonicom_eight_method_sparsity_v1/`。

## 2026-08-13：RIEC Q26 外部验证（已废弃）

- 数据与协议：下载并校验 103 名 RIEC 人类受试者；排除 046/080 dummy heads。冻结
  RIEC-Q26-v1（26 输入、839 插值方向、865 参考方向），按种子 20260813 划分
  77 train / 13 validation / 13 test。频段保持 50--20000 Hz，高频保持
  10000--20000 Hz；归一化只使用 77 名 train。SOFA 北极方向的已知 1.0 m
  元数据问题按官方 1.5 m 处理常量修正，不改原文件。
- 预处理：103/103 residual HDF5 完成，48 kHz、512 点 HRIR、2048 点 FFT、851 个
  频率 bin；每个文件包含 strict-HRIR ILD 重建元数据。train-only 统计覆盖
  113,361,710 个 residual 样本，target mean/std 为 0.282808/5.242230 dB。
- 固定训练链：完全复用锁定 SONICOM 预算和超参数，未做 RIEC 大规模搜索。v1 最佳
  epoch 20（validation MAE 2.976304 dB）；v2 最佳 epoch 5；关键基线 v3 最佳
  epoch 10；主方法 v3.2 总损失最佳 epoch 5、strict-ILD 单项最佳 epoch 1。
  所有训练 0 个 optimizer step 跳过。
- 一次性 test：模型与评估器锁定后才显式解封 13 名 test，并只读一次。固体角加权、
  仅 839 个插值方向的 magnitude MAE：MCA 3.569504 dB，v3 2.726095 dB，
  v3.2 2.727559 dB；相对 MCA 分别改善 23.628% 和 23.587%。strict-HRIR ILD
  MAE：v3 0.888739 dB，v3.2 0.903760 dB。
- 结论：冻结的 MCAR 框架在 RIEC 上相对 MCA 有明确外部泛化收益，但 v3.2 没有超过
  v3 关键基线（magnitude MAE 高约 0.054%，strict ILD 也略差）。test 后不改变模型、
  checkpoint 或超参数。正式结果位于 `results/riec_q26_v3_final_test/` 和
  `results/riec_q26_v32_final_test/`。

## 2026-08-13：补充 FSP-AE 的八方法 SONICOM-Q26 epoch 39 最终横向比较

- 工作目标：补齐此前 MCAR v3.2 epoch 39 七方法最终横向比较遗漏的 FSP-AE，形成
  SH only、SUpDEq + SH、SUpDEq + Natural Neighbor、SUpDEq + Barycentric、
  MCA、RANF、MCAR v3.2 epoch 39 和 FSP-AE 的完整八方法论文结果。
- 复用与口径：FSP-AE 已在同一 SONICOM 固定 test 划分上完成冻结推理和严格评价，故本次
  未重新训练、推理或读取 HRTF/test 数据，而是以
  `results/sonicom_seven_method_q26_epoch39_final_test/metric_long.csv` 为主表，仅从
  `results/sonicom_fsp_ae_q26_locked_final_test/metric_long.csv` 提取 44 名被试的
  FSP-AE 四项逐被试指标。两份源结果的 44 名被试及顺序一致；SH、三种 SUpDEq、MCA
  和 RANF 共 24 个共有逐被试指标变量最大绝对差严格为 `0 dB`。旧 FSP-AE 结果目录中的
  MCAR 使用较早 checkpoint，本次没有复用其 MCAR 行，所有相对量均针对论文主模型
  epoch 39 重新计算。
- 冻结模型：MCAR v3.2 epoch 39 checkpoint SHA-256 为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`；
  FSP-AE-Q26 adaptation epoch 40 checkpoint SHA-256 为
  `25DB1EB83A1B647B2E0E12C6BE1FF9EE31BF2B216DC50B57C19ADC84D1E12F5F`。
  输入为固定 Q26 的 26 个实测方向，评价排除输入，仅统计 767 个纯插值方向；44 名 test
  被试、四项指标、solid-angle/频段/HRIR-ILD 口径均与七方法表完全一致。
- 八方法结果（测量域 ERB / 对侧 25 度 ERB / 对侧半球 10--20 kHz / 水平面 ILD，
  `mean +/- subject standard deviation`，单位 dB、越低越好）：SH only 为
  `2.685 +/- 0.107 / 3.835 +/- 0.448 / 9.067 +/- 0.824 / 3.551 +/- 0.664`；
  SUpDEq + SH 为 `1.882 +/- 0.506 / 2.228 +/- 0.195 / 5.994 +/- 0.344 /
  2.018 +/- 1.564`；SUpDEq + Natural Neighbor 为 `1.852 +/- 0.290 /
  2.241 +/- 0.212 / 5.595 +/- 0.258 / 1.637 +/- 0.469`；SUpDEq + Barycentric 为
  `1.752 +/- 0.257 / 2.182 +/- 0.193 / 5.476 +/- 0.217 / 1.615 +/- 0.442`；
  MCA 为 `1.082 +/- 0.096 / 1.746 +/- 0.155 / 4.699 +/- 0.266 /
  0.829 +/- 0.158`；RANF 为 `1.063 +/- 0.126 / 1.557 +/- 0.164 /
  3.470 +/- 0.254 / 0.775 +/- 0.189`；MCAR v3.2 epoch 39 为
  `0.856 +/- 0.160 / 1.350 +/- 0.179 / 3.590 +/- 0.275 / 0.660 +/- 0.250`；
  FSP-AE 为 `1.184 +/- 0.349 / 1.909 +/- 0.701 / 3.164 +/- 0.685 /
  0.759 +/- 1.040`。
- 排名与配对统计：按被试均值，MCAR 在测量域 ERB、对侧 25 度 ERB和水平面 ILD 三项
  排名第一；FSP-AE 在对侧半球高频排名第一。FSP-AE 减 MCAR 的四项均值差及 100,000
  次固定 seed `20260813` 配对被试 bootstrap 95% 区间依次为
  `0.3288 [0.2843,0.3979] / 0.5589 [0.4574,0.7350] /
  -0.4261 [-0.5396,-0.2412] / 0.0984 [-0.0579,0.3704] dB`。MCAR 在前两项分别
  44/44 人更低；FSP-AE 在高频项 43/44 人更低。水平面 ILD 的均值由 MCAR 更低，但区间
  跨 0，且 FSP-AE 在 27/44 人更低，因此不能宣称 MCAR 在该项显著优于 FSP-AE。
- 完整性：新长表为精确的 `44 x 8 x 4 = 1408` 行，聚合表 32 行，逐被试表和质量检查
  各 44 行；被试-方法-指标键无重复，所有指标均为有限值，FSP-AE 频率网格最大偏差为
  `0 Hz`。合并过程读取 test 被试或模型 prediction 的数量为 `0`。
- 产物：冻结配置为
  `configs/experiments/sonicom_eight_method_q26_epoch39_final_test.json`；可复现合并器为
  `matlab/+mcar/merge_sonicom_eight_method_epoch39_results.m`；正式结果位于
  `results/sonicom_eight_method_q26_epoch39_final_test/`，包括论文标签主表、逐被试宽/长
  表、四指标排名、MCA/RANF/FSP-AE 相对 epoch 39 的配对 bootstrap、质量检查、JSON
  摘要和八方法聚合图。旧七方法目录保留为审计来源，不覆盖或删除。

## 2026-08-13：SONICOM 三种学习方法稀疏度实验

> 状态：有效。该实验是当前用于比较学习型方法稀疏鲁棒性的正式实验，方法为
> MCAR v3.2、FSP-AE 和 RANF。

- 冻结协议：使用固定 SONICOM test 划分中的 44 名受试者，在嵌套的
  `Q=6/14/26` 观测方向上比较三种方法。所有方法和稀疏度共用同一个评价掩码，
  排除完整 Q26 输入集合，仅评价剩余 767 个方向。SONICOM test 此前已用于 Q26
  冻结评测，因此本实验是预先固定的新稀疏度分析，不表述为未见外部验证。
- 方法：MCAR 使用冻结的 v3.2 epoch 39 residual checkpoint，并在每个 Q 下重新计算
  MCA 物理先验；FSP-AE 使用冻结的 epoch 40 checkpoint，仅编码当前 Q 的观测；
  RANF 从同一个冻结的 Q26 预训练 checkpoint 出发，在每个 Q 下重新计算训练受试者
  检索距离，并分别执行 1000 epoch、batch size 3 的逐受试者适配。
- 完整性：完成 `44 x 3 = 132` 个受试者-稀疏度案例，三方法、四指标共
  `44 x 3 x 3 x 4 = 1584` 行逐受试者结果。指标为测量域 ERB、对侧 25 度 ERB、
  对侧半球 10--20 kHz 幅度误差和水平面 ILD MAE；全部案例使用 767 个评价方向。
- 主要结果：RANF 在 Q6 的四项平均误差均最低，且四项 `Q6-Q26` 退化均最小。
  测量域 ERB 的 `Q6-Q26` 退化分别为 RANF `0.147 dB`、MCAR `1.037 dB`、
  FSP-AE `2.290 dB`。RANF 相对 MCAR 和 FSP-AE 的所有 Q6 端点及退化对比均由
  10,000 次配对受试者 bootstrap 支持（seed `20260812`，双侧 `p=0.0002`）。
  结论是 RANF 在极稀疏输入下最稳定；MCAR 在 Q26 的多数绝对指标仍更低，FSP-AE
  对方向数减少最敏感。
- 产物：冻结配置为
  `configs/experiments/sonicom_learned_methods_sparsity_v1.json`，完整协议为
  `docs/SONICOM_LEARNED_SPARSITY_PROTOCOL.md`，逐受试者表、聚合表、配对统计、
  质量检查、RANF 适配成本和主图位于
  `results/sonicom_learned_methods_sparsity_v1/`。有效实验代码与结果提交为
  `4879ca2`（`完成SONICOM三种学习方法稀疏度实验`）。

## 2026-08-13：HUTUBS 外部数据集稀疏度实验（已废弃）

> 状态：已于 2026-08-13 停止作为有效实验使用。后续稀疏度结论以
> SONICOM 上 MCAR、FSP-AE 和 RANF 三种学习方法的统一实验为准。HUTUBS
> 实验的协议和正式结果仅作为历史审计记录，分别归档在
> `废弃实验/HUTUBS稀疏度实验/experiment_config.json` 和
> `废弃实验/HUTUBS稀疏度实验/results/`；可再生成的中间产物与专用代码已清理。

- 目标与冻结协议：执行前固定的协议现归档于
  `废弃实验/HUTUBS稀疏度实验/experiment_config.json`，
  在从未参与 SONICOM 模型训练或选择的 HUTUBS simulated HRIR test 子集上做零样本外部
  验证。固定 12 名被试 `8/18/22/26/31/33/45/47/59/70/73/81`，固定 Lebedev
  order `1/2/3`，即 `Q=6/14/26` 个输入方向；三组方向均为 HUTUBS order-35 原始
  采样网格的精确子集。所有稀疏度共用一个评价掩码，排除 Q26 输入点，避免不同 Q 使用
  不同目标点。实验结束前未按结果改变稀疏度、被试、方法、checkpoint、指标或掩码。
- 方法：传统 SH 插值、MCA、冻结的 SONICOM MCAR v3.2 seed `20260809` epoch `39`
  以及冻结的 SONICOM FSP-AE epoch `40` 直接生成基线。选择 FSP-AE 是为了保持全部学习
  方法均不在 HUTUBS 上更新参数；需要逐被试适配的 RANF 不适合作为本实验的“直接生成且
  冻结”对照。MCAR checkpoint SHA-256 仍为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`。
- 完整性：完成 `12 x 3 = 36` 个被试-稀疏度案例，四方法、四指标共
  `12 x 3 x 4 x 4 = 576` 行，全部为有限值；每个案例使用 899 个固定稠密评价方向。
  三层 FSP-AE 输出频率与评价频率的最大偏差均为 `0 Hz`。正式结果位于
  `废弃实验/HUTUBS稀疏度实验/results/`。
- 主要稳健性结果（同一被试 `Q6-Q26` 误差，越小越稳健）：MCAR 的全空间 ERB、对侧
  25 度 ERB、对侧 10--20 kHz 误差退化分别为 `0.325/-0.096/0.592 dB`，MCA 为
  `0.820/0.522/1.634 dB`。两者配对退化差为
  `-0.495 [-0.536,-0.453] / -0.617 [-0.678,-0.552] / -1.042
  [-1.239,-0.850] dB`（10,000 次配对被试 bootstrap，seed `20260812`），三个谱指标
  均明确支持 MCAR 在进一步稀疏时比 MCA 更稳健。水平 ILD 的退化差为
  `0.031 [-0.123,0.181] dB`，不能证明 MCAR 比 MCA 更稳健。
- 极稀疏端点 `Q=6`：MCAR 相对 SH 在四项指标均更低；相对 MCA，MCAR 在对侧 25 度
  ERB 与对侧高频上更低，分别相差 `-0.064 [-0.112,-0.004]` 与
  `-0.483 [-0.655,-0.292] dB`，但全空间 ERB 与水平 ILD 更高，分别相差
  `+0.483 [0.444,0.523]` 与 `+0.233 [0.057,0.400] dB`。因此结果支持“谱细节的稀疏
  稳健性”，不支持 MCAR 在所有指标和绝对误差上全面优于 MCA。
- 直接生成基线：FSP-AE 在 Q6 四项绝对误差均明显高于 MCAR；其水平 ILD 曲线在输入
  变少时反而下降，属于整体误差很高时的反常斜率，不能单独解释为有用的稀疏稳健性。
  结论必须联合读取极稀疏端点误差与退化差，而不能只比较曲线平坦度。
- 产物：`metric_long.csv` 为逐被试长表，`aggregate_metrics.csv` 为均值与被试标准差，
  `robustness_statistics.csv` 为每方法退化与每减半方向数斜率，
  `paired_robustness_contrasts.csv` 为 MCAR 对三条基线的配对端点/退化差，
  `figures/performance_vs_sparsity.png/.pdf` 为主图，`summary.json` 与
  `paired_robustness_summary.json` 为机器可读汇总。

## 2026-08-12：MCAR v3.2 epoch 39 七方法 SONICOM-Q26 最终 test 横向比较

- 工作目标：将已经确定为论文主模型的 MCAR v3.2 seed `20260809` epoch `39`
  接入与 SH only、SUpDEq + SH、SUpDEq + Natural Neighbor、SUpDEq +
  Barycentric、MCA 和 RANF 完全相同的冻结严格评价器，在 SONICOM-Q26 test 上重新
  生成七方法完整横向结果。此次只执行推理结果读取、传统方法重建和统一评价，不训练、
  不调参、不修改模型，也不根据结果改变 epoch、seed、损失、指标或基线设置。
- 冻结协议：执行前写入
  `configs/experiments/sonicom_seven_method_q26_epoch39_final_test.json`。数据为固定
  `262/44/44` 划分中的 44 名 test 被试；输入严格采用 `SONICOM-Q26-v1` 的 26 个实测
  方向，评价排除全部输入点，只统计剩余 767 个纯插值方向。主模型 checkpoint 为
  `artifacts/training/sonicom_mlp_cnn_q26_v32_seed20260809_e40/best.pt`，内部 epoch
  为 `39`，评价前后 SHA-256 均为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`。
- 评价口径：七方法全部由
  `mcar.evaluate_sonicom_interpolation_baselines` 计算同一四项 subject-level 指标并以
  `mean +/- subject standard deviation` 汇总，单位均为 dB、越低越好。历史字段
  `FullSphereERB` 实际只覆盖 SONICOM 实测域（仰角 `-45 deg` 至 `90 deg`），论文表改称
  `MeasuredDomainERB`；历史字段 `ContralateralHighFrequency` 实际为对侧半球
  `10--20 kHz` 幅度误差，论文表改称 `ContralateralHemisphereHF`。其余两项为对侧
  中心 25 度区域 ERB 误差和 72 个水平面纯插值方向的严格 HRIR 能量 ILD MAE。
- 执行过程：冻结 checkpoint、epoch 39 的 44 份 test prediction 和 RANF 的 44 份
  prediction 预检完整。MATLAB MCP 同步连接在 600 s 工具上限处超时，但 MATLAB
  计算进程没有终止，持续稳定运行并最终完整写出 CSV、JSON 和图；未使用旧表手工替换
  MCAR 行。正式结果根目录为
  `results/sonicom_seven_method_q26_epoch39_final_test/`。
- 七方法完整结果（依次为测量域 ERB / 对侧 25 度 ERB / 对侧半球 HF / 水平面 ILD，
  `mean +/- subject standard deviation`）：SH only 为
  `2.685 +/- 0.107 / 3.835 +/- 0.448 / 9.067 +/- 0.824 / 3.551 +/- 0.664`；
  SUpDEq + SH 为
  `1.882 +/- 0.506 / 2.228 +/- 0.195 / 5.994 +/- 0.344 / 2.018 +/- 1.564`；
  SUpDEq + Natural Neighbor 为
  `1.852 +/- 0.290 / 2.241 +/- 0.212 / 5.595 +/- 0.258 / 1.637 +/- 0.469`；
  SUpDEq + Barycentric 为
  `1.752 +/- 0.257 / 2.182 +/- 0.193 / 5.476 +/- 0.217 / 1.615 +/- 0.442`；
  MCA 为 `1.082 +/- 0.096 / 1.746 +/- 0.155 / 4.699 +/- 0.266 / 0.829 +/- 0.158`；
  RANF 为 `1.063 +/- 0.126 / 1.557 +/- 0.164 / 3.470 +/- 0.254 / 0.775 +/- 0.189`；
  MCAR v3.2 epoch 39 为
  `0.856 +/- 0.160 / 1.350 +/- 0.179 / 3.590 +/- 0.275 / 0.660 +/- 0.250`。
- 相对 MCA：MCAR epoch 39 的四项均值分别改善
  `20.932% / 22.671% / 23.585% / 20.387%`，逐被试改善数为
  `43/44 / 43/44 / 44/44 / 37/44`。100,000 次固定 seed `20260812` 的配对
  subject bootstrap 中，MCA 减 MCAR 的均值差及 95% 区间分别为
  `0.2265 [0.2003, 0.2476] / 0.3958 [0.3048, 0.4591] /
  1.1082 [1.0393, 1.1641] / 0.1691 [0.0818, 0.2362] dB`。
- 相对 RANF：MCAR epoch 39 在测量域 ERB、对侧 25 度 ERB 和水平面 ILD 上分别改善
  `19.522% / 13.314% / 14.851%`，并分别在 `43/44 / 42/44 / 35/44` 名被试上
  更好；三项 RANF 减 MCAR 的均值差及 95% bootstrap 区间为
  `0.2076 [0.1768, 0.2320] / 0.2073 [0.1252, 0.2711] /
  0.1152 [0.0467, 0.1786] dB`。对侧半球 HF 上 RANF 更好：MCAR 相对 RANF
  回退 `3.481%`，只有 `13/44` 名被试优于 RANF，RANF 减 MCAR 的差为
  `-0.1208 [-0.1807, -0.0661] dB`。因此论文不能宣称 MCAR 四项全面最优，应写成
  MCAR 在总体幅度、对侧局部幅度和水平面 ILD 上占优，RANF 在对侧半球高频幅度上占优。
- 完整性检查：`metric_long.csv` 为精确的 `44 x 7 x 4 = 1232` 行，44 名被试、
  7 个方法和 4 个指标全部存在且无非有限值；六个非 MCAR 方法相对 2026-08-11 七方法
  结果的最大绝对差为 `0 dB`；MCAR epoch 39 相对此前单模型冻结 test 结果的最大绝对
  差为 `0 dB`；RANF 对 26 个 Q26 观测方向的最大 HRIR 改写误差仍为 `0`。
- 正式产物：论文准确标签表为 `paper_comparison_paper_labels.csv`，MCA/RANF 配对
  bootstrap 表为 `paired_comparison_mcar_epoch39.csv`，完整七方法表为
  `paper_comparison.csv`，逐被试宽表为 `per_subject_metrics.csv`，长表为
  `metric_long.csv`，质量检查为 `quality_checks.csv`，聚合图为
  `figures/test44_aggregate_baselines.png`。
- 论文规范边界：epoch 39 最初按固定 validation total-loss 规则选择，但 SONICOM test
  已在 2026-08-09 对该 checkpoint 做过一次冻结评价；本次是同一冻结模型的七方法完整
  横向重算，不能描述为新的、此前完全未见的 confirmatory test。此后不得继续根据
  SONICOM test 调整或筛选模型；独立泛化主张需由冻结 epoch 39 在外部数据集上的零样本
  评价承担。

## 2026-08-11：RANF Q26 横向基线正式训练、适配与原生评价

- 工作目标：将上游 RANF v2.0.0（commit
  `92956c74b0ac975169066a539fff47129fd7b6d1`）作为独立横向基线接入已经冻结的
  MCAR SONICOM-Q26 协议。实现位于独立 worktree
  `baselines/ranf/worktree_q26/`，训练环境为 WSL2 Ubuntu 22.04、Python 3.10、
  PyTorch `2.7.0+cu128`，GPU 为 NVIDIA GeForce RTX 5060 8 GB。
- 冻结数据协议：使用 350 名清洁 SONICOM FreeFieldCompMinPhase 44.1 kHz 被试以及既有
  `262/44/44` train/validation/test 划分；稀疏输入严格采用
  `configs/data/sonicom_sparse_grid_q26_v1.csv` 的 26 个方向。检索候选仅允许来自 train，
  validation/test 不得成为候选。RANF 为适配上游连续编号约束建立内部编号，原始编号由
  `~/ranf-work/mcar_q26_formal/subject_mapping.csv` 保留。
- 正式预训练：seed `20260731`，RAdam，学习率 `1e-3`，batch size `256`，每 epoch
  `816` 个 batch，最大 `200` epoch，early-stopping patience `20`。batch size 256 是
  RTX 5060 上实测吞吐最优的基线实现参数，不宣称严格复现原论文的 batch 64。运行时间为
  `2026-08-11 00:27:10` 至 `02:37:39`（`2:10:29`），在 epoch `118`
  提前停止，共完成 epoch `0--118`，退出码为 `0`。最佳 validation loss 为
  `15.59127`，最佳 combined metric 为 `16.30561`。W&B run ID 为 `qanlzdld`，
  地址为 `https://wandb.ai/luyoung/MCAR-RANF/runs/qanlzdld`。
- 预训练 checkpoint 完整性：`best.ckpt` 的 SHA-256 为
  `7b288f6f9198664e9ce75fee418e90ec3da2f5d0b7996a4edefe7d5f8da3e367`。
- test seal 与适配：模型和超参数冻结后，于 `2026-08-11 10:04:58` 显式释放 test
  seal。只使用每名 test 被试的 Q26 稀疏观测执行上游 subject-specific adaptation：
  `1000` epoch、batch size `3`，于 `13:30:33` 正常结束，用时 `3:25:35`，退出码
  为 `0`。`adaptation.ckpt` SHA-256 为
  `e1b17d5dc0dc550caedba5ac5f8630acc96cff8c9bf1daaeb09a6d3ad64371a3`；
  `adaptation_loss.ckpt` SHA-256 为
  `f8d86ce6d708ec14d4c77d9a10673ae86d770e987167a596a4aca215460b8641`。
- 原生最终评价：第一次评价在完成 6 名被试后因 CUDA OOM 退出。根因是上游
  `3_evaluating_neural_field.py` 在循环中保留每名被试的 autograd graph 和 GPU tensor，
  导致显存随被试数累计；与适配 checkpoint 无关。评价入口改为
  `torch.inference_mode()`、移除无用 GPU tensor 列表，并令重试覆盖不完整日志。
  `14:11:46` 开始仅重跑评价，`14:13:39` 完成全部 44 名 test 被试，最终 pipeline
  phase 为 `complete`、退出码为 `0`；ITD/ILD/LSD 三类记录均为 `44/44`。
- RANF 原生 LAP 指标：ITD mean/max 为 `17.599497/238.338295 us`，ILD mean/max
  为 `0.716003/0.998331 dB`，LSD mean/max 为 `3.222125/3.969062 dB`。
  唯一超过 100 us ITD 阈值的是内部编号 `P0320`，经冻结映射后对应原始 SONICOM
  test 被试 `P0339`，论文与逐被试表必须使用原始编号。
- 原生产物：WSL 中实验目录为 `~/ranf-work/exp/mcar_q26_formal/`，包含
  `final_summary.txt`、`final_pipeline_console.log`、`log/eval/eval.log`、44 份
  `pred_*.sofa` 和对应 target SOFA。上述 ITD/ILD/LSD 是 RANF/LAP 原生口径，不能与
  MCAR 论文表中的 FullSphere ERB、Contralateral 25-degree ERB、Contralateral HF 和
  strict Horizontal ILD 直接混排；横向结论必须把 RANF SOFA 送入同一冻结严格评价器。
- 统一严格评价：将 44 份 RANF SOFA 按 `subject_mapping.csv` 恢复为原始 SONICOM
  test 编号，导出到
  `artifacts/reconstruction/sonicom_ranf_q26_final_test/`，并接入与 SH only、
  SUpDEq、MCA 和 MCAR 完全相同的 MATLAB 严格评价入口。1 人 smoke 通过后完成 44 人
  正式评价，MATLAB 正常退出，用时 `584.8 s`。评价方向为排除 Q26 输入后的 767 个
  方向；四项 RANF 指标均为 `44/44` 有限值，RANF 对 26 个观测方向的 HRIR 最大改写
  误差为 `0`。重新计算所得既有六方法结果与原冻结横向表逐值一致，最大绝对差为 `0`。
- 统一横向结果（mean +/- subject standard deviation，单位 dB，越低越好）：RANF 的
  FullSphere ERB 为 `1.063 +/- 0.126`、Contralateral 25-degree ERB 为
  `1.557 +/- 0.164`、Contralateral HF 为 `3.470 +/- 0.254`、strict Horizontal
  ILD MAE 为 `0.775 +/- 0.189`。相对 MCA 分别改善
  `1.752% / 10.794% / 26.156% / 6.501%`。
- 与 MCAR v3.2 的关系：MCAR 在 FullSphere ERB、Contralateral 25-degree ERB 和
  Horizontal ILD 上分别降低 `18.381% / 12.329% / 11.408%`，并分别在
  `43/44`、`42/44`、`33/44` 名被试上优于 RANF；但 RANF 的 Contralateral HF
  均值比 MCAR 低 `4.098%`，RANF 在该项上优于 MCAR 的被试数为 `33/44`。因此论文
  不应宣称 MCAR 对 RANF 四项全面领先，应表述为 MCAR 在总体谱误差、对侧局部谱误差和
  水平面 ILD 上占优，而 RANF 在对侧高频幅度误差上更强。
- 正式表格位于 `results/sonicom_ranf_q26_final_test/`：
  `per_subject_core_comparison.csv` 为 SH only、SUpDEq + SH、MCA、RANF、MCAR v3.2
  五方法逐被试宽表，`paper_core_comparison.csv` 为五方法论文聚合表；
  `per_subject_metrics.csv` 与 `paper_comparison.csv` 另保留 Natural Neighbor 和
  Barycentric，构成七方法完整表。`metric_long.csv` 用于后续配对统计。

## 2026-08-10：SONICOM Q26 v3.4c 多尺度 notch-aware 损失消融

- 工作目标：从固定 v3.2 seed `20260809` epoch 39 独立检验 notch-aware 目标，直接
  约束耳廓相关谱凹陷的位置与深度。只使用 262 train / 44 validation；禁止读取已消费
  的 test，全部产物均记录 `test_subject_count_read=0`。
- 目标定义：在 4–18 kHz，对半径 $r\in\{4,8,16\}$ 个频点定义局部凹陷原始深度
  $d_f^{(r)}=(H_{f-r}+H_{f+r})/2-H_f$，再以
  $\tilde d_f^{(r)}=\tau\operatorname{softplus}((d_f^{(r)}-d_0)/\tau)$ 构造可微
  notch-depth map，其中 $d_0=1\,\mathrm{dB}$、$\tau=0.5\,\mathrm{dB}$。预测与
  reference map 的方向面积加权 MAE 同时惩罚缺失 notch、伪 notch、位置偏移和深度偏差。
- 校准：Q26 频率间隔 `43.0664 Hz`，三个半径约为 `172/345/689 Hz`。v3.2 epoch 39
  的 44 人完整 validation notch-depth MAE 为 `0.504822 dB`；权重预先锁定为 `0.30`，
  预计新增 total 贡献 `0.03057`、占新初始 total 约 `4.42%`。原
  `ERB/HF/strict ILD=0.75/0.25/0.75` 不变，v3.4a D1/D2 与 v3.4b band-ILD 权重均为 0。
- 实现与测试：公共 loss 新增多尺度 Softplus notch-depth map，接入 v2 公共 loss 与 v3
  CLI、history、W&B、console、report。CPU 测试覆盖相同谱为零、缺失/伪 notch、方向
  权重、有限非零梯度、目标增量恒等式和 v3.4c stage；v3.4a、v3.4b、dual-sampling、
  joint optimizer、global attention 回归均通过。
- smoke：RTX 5060 上真实 SONICOM 1-step CUDA+AMP 通过；总参数 174,627，局部 CNN
  的 74,402 个参数可训练，optimizer skip 为 0，smoke 不保存 checkpoint。
- 正式训练：执行
  `D:\miniconda3\envs\ml\python.exe -u scripts\run_v34c_notch_aware_e10.py`。
  10 epoch × 500 step、96 个固定 validation block，用时 `886.528 s`，峰值 CUDA
  allocated memory `399.790 MiB`，optimizer skip 为 0，W&B offline ID `iohnfzkr`。
  按预注册 v3.4c total 选择 epoch 7：total 从 `0.6913994942` 降至 `0.6909071778`，
  抽样 validation notch-depth MAE 从 `0.500122` 降至 `0.499545 dB`。
- checkpoint 完整性：源 SHA-256 为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`，候选为
  `5A2A29B38D49717FBA2E2C459F7A987F18E799CA0EB58AD4F8EDAE293717C123`。MLP 的
  100,225 个参数逐值未变；CNN 相对 L2 改变量为 `0.2802%`。
- 推理与重建：epoch 7 对 44/44 人生成 `[2,793,463]` residual，用时 `11.750 s`。
  MATLAB 1 人 smoke 四项均改善；44 人严格重建用时 `435.1 s`，正常退出。summary
  为 `status=completed`、`split=val`、`subject_count=44`、`test_subject_count_read=0`。
- 严格结果：相对 v3.2 epoch 39，全空间 ERB 从 `0.872882` 降至 `0.872650 dB`
  （改善 `0.0267%`，24/44 人），对侧 25° ERB 从 `1.364204` 降至
  `1.362914 dB`（改善 `0.0946%`，29/44 人），对侧高频从 `3.608567` 降至
  `3.608108 dB`（改善 `0.0127%`，28/44 人），水平面 strict ILD 从 `0.596141`
  降至 `0.592944 dB`（改善 `0.5362%`，30/44 人）。
- 配对统计：四项 candidate-baseline 均值差为
  `-0.0002327 / -0.0012899 / -0.0004595 / -0.0031964 dB`；paired t-test 为
  `0.0300 / 0.00378 / 0.357 / 0.00531`，Wilcoxon 为
  `0.0623 / 0.00378 / 0.138 / 0.00426`。对侧 ERB 与 strict ILD 方向可信，高频仍不显著。
- 全方向 notch 诊断：44 人全部 767 个纯插值方向上的 notch-depth MAE 从 `0.504822`
  降至 `0.504247 dB`（改善 `0.1139%`，44/44 人），paired t-test `4.14e-29`，
  Wilcoxon `1.14e-13`，Cohen $d_z=-4.19$。4/8/16-bin 三个尺度分别改善
  `0.1588% / 0.1105% / 0.1032%`，每个尺度均为 44/44 人改善。
- 版本判断：v3.4c 是正向且高度一致的 notch 消融，但目标改善幅度小于 v3.4a 的谱差分
  和 v3.4b 的 band-ILD；其严格 HF 数值改善是三个 loss 消融中最大，但仍不显著。
  v3.4b 仍是对侧 ERB 最强单项，v3.4c 的 strict ILD 略强。
- 下一步：三个独立目标都已验证为正，可测试一次等比例降权的组合消融，建议将原权重
  同时减半为 `D1/D2/band-ILD/notch = 0.125/0.075/0.05/0.15`，把新增目标总贡献控制
  在约 `0.045`，避免简单叠加后辅助 loss 占比过大。完整结果位于
  `results/sonicom_mlp_cnn_q26_v34c_notch_aware_e10_strict_validation/`。

## 2026-08-10：SONICOM Q26 v3.4b 分频带 ILD 损失消融

- 工作目标：在固定 v3.2 seed `20260809` epoch 39 上独立检验分频带 ILD 目标，直接
  约束左右耳在不同听觉频带内的能量平衡。只使用 262 train / 44 validation；禁止读取
  已消费的 test，所有产物均记录 `test_subject_count_read=0`。
- 目标设计：沿用现有 41 个 ERB proxy filters，保留中心频率位于 200 Hz–18 kHz 的
  35 个频带。在每个水平面纯插值方向上，由 corrected/reference 双耳幅度谱分别计算
  左减右 band-energy ILD，以 $\beta=0.5\,\mathrm{dB}$ 的 SmoothL1 作为训练项，并按
  SONICOM solid-angle direction weight 汇总。新增权重固定为 `0.10`；原
  `ERB/HF/strict broadband ILD=0.75/0.25/0.75` 不变，v3.4a 的 D1/D2 权重均置 0。
- 权重校准：在 v3.2 epoch 39 的 44 人、72 个水平面纯插值方向上，分频带 ILD
  SmoothL1/MAE 为 `1.407891 / 1.629438 dB`。新项预计贡献 `0.02842`，约占新初始
  total 的 `4.12%`，与 v3.4a 的新增目标尺度接近。
- 实现与测试：公共 loss 新增双耳 ERB-band ILD SmoothL1/MAE；v3 训练器在同一个
  horizontal forward 中联合计算严格 HRIR broadband ILD 与 band ILD，并将指标接入
  history、W&B、console 与 report。新测试覆盖零误差、解析方向加权值、有限非零梯度、
  目标增量恒等式与零权重旧行为；未改 HDF5 或模型结构。
- smoke：RTX 5060 上真实 SONICOM 1-step CUDA+AMP 通过；总参数 174,627，其中局部
  CNN 的 74,402 个参数可训练，MLP 冻结；optimizer skip 为 0，不保存 smoke 权重。
- 正式训练：执行
  `D:\miniconda3\envs\ml\python.exe -u scripts\run_v34b_band_ild_e10.py`。
  10 epoch × 500 step、96 个固定 validation block，用时 `820.906 s`，峰值 CUDA
  allocated memory `410.870 MiB`，optimizer skip 为 0，W&B offline ID `u8cv7js1`。
  按预注册 v3.4b total 选择 epoch 7：total 从 `0.6896631916` 降至 `0.6890342993`；
  抽样 validation band SmoothL1/MAE 从 `1.414343 / 1.636103` 降至
  `1.403707 / 1.624716 dB`。
- checkpoint 完整性：源 SHA-256 为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`，候选为
  `EABFAA1B303F6016F9F8F7507D510813C44C948C81D9C074EECA0A8E84AA4DAF`。MLP 的
  100,225 个参数逐值未变；CNN 相对 L2 改变量为 `0.2989%`。
- 推理与重建：epoch 7 对 44/44 人生成 `[2,793,463]` residual，用时 `11.223 s`。
  MATLAB 1 人 smoke 四项均改善；44 人严格重建用时约 `726.7 s`，正常退出。summary
  为 `status=completed`、`split=val`、`subject_count=44`、`test_subject_count_read=0`。
- 严格结果：相对 v3.2 epoch 39，全空间 ERB 从 `0.872882` 降至 `0.872511 dB`
  （改善 `0.0425%`，33/44 人），对侧 25° ERB 从 `1.364204` 降至
  `1.360356 dB`（改善 `0.2821%`，40/44 人），对侧高频从 `3.608567` 降至
  `3.608369 dB`（改善 `0.0055%`，27/44 人），水平面 strict ILD 从 `0.596141`
  降至 `0.593076 dB`（改善 `0.5141%`，32/44 人）。
- 配对统计：全空间 ERB / 对侧 ERB / HF / strict ILD 的 candidate-baseline 均值差为
  `-0.0003710 / -0.0038480 / -0.0001978 / -0.0030646 dB`；paired t-test 为
  `0.00112 / 3.76e-9 / 0.711 / 0.00452`，Wilcoxon 为
  `0.00113 / 3.14e-10 / 0.435 / 0.00334`。前三项中的 HF 仍无显著改善。
- 全水平面 band-ILD 诊断：35-band SmoothL1 从 `1.407891` 降至 `1.397209 dB`
  （改善 `0.7587%`，43/44 人），MAE 从 `1.629438` 降至 `1.617899 dB`
  （改善 `0.7082%`，43/44 人）；paired t-test 分别为 `6.45e-19 / 2.33e-19`，
  35/35 个频带的聚合误差均下降。最大相对收益集中在约 250–500 Hz；6 kHz 附近与
  高频频带收益较弱，这解释了原对侧 10–20 kHz 幅度指标几乎不变。
- 版本判断：v3.4b 在目标对齐、对侧 ERB 改善幅度和被试一致性上优于 v3.4a，是当前
  最强的定向 loss 消融；但 v3.2.1 在全空间 ERB、高频和 strict ILD 的数值改善略大，
  且 v3.4b 的 HF 仍不显著，所以仍不宣称其全面替代 v3.2 epoch 39。
- 下一步：从同一固定基准独立测试 notch-aware loss，避免立即把多个新目标叠加导致
  无法归因；若 notch 实验也为正，再做 D1/D2 + band ILD + notch 的组合验证。
  完整结果位于
  `results/sonicom_mlp_cnn_q26_v34b_band_ild_e10_strict_validation/`。

## 2026-08-10：SONICOM Q26 v3.4a 高频谱一阶/二阶差分损失消融

- 工作目标：不再提高原 10–20 kHz 对侧幅度标量权重，而是直接约束谱峰谷形状。实验从
  固定 v3.2 seed `20260809` epoch 39 出发，只使用 262 train / 44 validation；禁止
  读取已消费的 test，所有产物均记录 `test_subject_count_read=0`。
- 目标设计：令 $e_f=\hat r_f-r_f$，新增 4 kHz 以上的
  $L_{D1}=\operatorname{MAE}(e_{f+1}-e_f)$ 与
  $L_{D2}=\operatorname{MAE}(e_{f+2}-2e_{f+1}+e_f)$。Q26 为 463 点、间隔
  `43.0664 Hz` 的等距网格，单位为 `dB/bin` 和 `dB/bin²`。选择 4 kHz 是为覆盖
  4–10 kHz 耳廓 notch 区域；原 10–20 kHz HF loss 保持不变。
- 权重校准：v3.2 epoch 39 的完整 validation D1/D2 为 `0.441563 / 0.277322`，train
  代表性抽样为 `0.442725 / 0.283576`，分布吻合。预先锁定 D1/D2 权重
  `0.25/0.15`，预计只占初始 total 的约 `4.6%`；原
  `ERB/HF/strict ILD=0.75/0.25/0.75` 不变，避免同时修改多个因素。
- 实现：在公共 `losses.py` 增加方向面积加权 D1/D2，在 v2 公共 loss 入口接入并保持
  新权重为 0 时旧行为不变；v3 训练器新增 CLI、history、W&B、console 和 report 字段。
  新增 CPU 单元测试覆盖零误差、常数/线性/二次误差、方向加权、目标增量恒等式和有限
  backward；dual-sampling、joint optimizer、global attention 回归测试均通过。
- smoke：RTX 5060 上真实 SONICOM 1-step CUDA+AMP 与 strict HRIR-ILD 反向传播通过，
  174,627 个总参数中 74,402 个局部 CNN 参数可训练；optimizer skip 为 0。smoke 不
  参与选模。
- 正式训练：执行
  `D:\miniconda3\envs\ml\python.exe -u scripts\run_v34a_spectral_diff_e10.py`。
  MLP 冻结，局部 CNN 学习率 `1e-5`；10 epoch × 500 step、96 个固定 validation
  block，用时 `812.893 s`，峰值 CUDA allocated memory `399.405 MiB`，optimizer
  skip 为 0，W&B offline ID `e80qlsxd`。按预注册 v3.4a total 选择 epoch 7：total
  从 `0.6915137445` 降至 `0.6904440752`，validation D1/D2 从
  `0.437596 / 0.274691` 降至 `0.432445 / 0.264063`。
- checkpoint 完整性：源 SHA-256 为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`，候选为
  `0FFF110F11C5149AD34D65549F2808975F666862A8C9462381125A617A4DD8BC`。MLP 的
  100,225 个参数逐值未变，最大绝对差为 0；CNN 相对 L2 改变量为 `0.3124%`。
- 推理与重建：epoch 7 对 44/44 人生成 `[2,793,463]` residual，用时 `8.829 s`。
  MATLAB 1 人 smoke 通过；44 人严格重建用时 `428.1 s`，正常退出。summary 为
  `status=completed`、`split=val`、`subject_count=44`、`test_subject_count_read=0`。
  推理实际执行
  `D:\miniconda3\envs\ml\python.exe -u -m mcar.evaluation.predict_sonicom_mlp_cnn_residuals data\processed\sonicom_residual_q26_v1 artifacts\training\sonicom_mlp_cnn_q26_v34a_spectral_diff_e10\best.pt sonicom_q26_validation_mlp_cnn_v34a_spectral_diff_e10 --directions-per-block 32`；
  严格重建实际执行 `matlab.exe -batch "addpath('D:/cuc/CSMT/MCAR/matlab'); mcar.evaluate_sonicom_validation_reconstruction(inf, 'sonicom_mlp_cnn_q26_v34a_spectral_diff_e10_strict_validation', true, 'sonicom_q26_validation_mlp_cnn_v32_seed20260809_e40', 'sonicom_q26_validation_mlp_cnn_v34a_spectral_diff_e10', 'MLP+CNN v3.4a spectral diff e10 epoch 7')"`。
- 严格结果：相对 v3.2 epoch 39，全空间 ERB 从 `0.872882` 降到 `0.872618 dB`
  （改善 `0.0303%`，25/44 人），对侧 25° ERB 从 `1.364204` 降到
  `1.363347 dB`（改善 `0.0628%`，29/44 人），高频从 `3.608567` 降到
  `3.608472 dB`（改善 `0.0026%`，25/44 人），水平面 strict ILD 从 `0.596141`
  降到 `0.593248 dB`（改善 `0.4852%`，30/44 人）。
- 配对统计：全空间 ERB / 对侧 ERB / HF / ILD 的 candidate-baseline 均值差为
  `-0.0002649 / -0.0008567 / -0.0000954 / -0.0028925 dB`；paired t-test 为
  `0.0298 / 0.0401 / 0.8979 / 0.0106`，Wilcoxon 为
  `0.0852 / 0.0171 / 0.3756 / 0.0068`。因此对侧 ERB 和 ILD 的方向最可信，HF
  没有可辨别改善；所有绝对效果仍很小。
- 全方向谱形状诊断：在 44 人全部 767 个纯插值方向上，D1 从 `0.441563` 降至
  `0.436421 dB/bin`（改善 `1.164%`），D2 从 `0.277322` 降至
  `0.266727 dB/bin²`（改善 `3.821%`），两项均为 44/44 人改善；paired t-test
  为 `1.85e-30 / 2.07e-29`。说明新增目标明确生效，但谱斜率/曲率改善不等于逐点
  高频幅度 MAE 改善。
- 结论与下一步：v3.4a 是正向目标函数消融，但综合收益小，暂不替代 v3.2 epoch 39。
  下一项建议独立测试分频带 ILD，以直接约束双耳频谱平衡；随后再单独测试 notch-aware
  loss。hard-direction sampling 放在最后，因为它会改变训练方向分布、归因风险最高。
  完整结果位于
  `results/sonicom_mlp_cnn_q26_v34a_spectral_diff_e10_strict_validation/`。

## 2026-08-10：SONICOM Q26 v3.3 全局频谱 attention 与门控融合消融

- 工作目标：检验现有局部 CNN 的约 97 频点感受野是否限制性能。在固定 v3.2 seed
  `20260809` epoch 39 上新增轻量全局分支，只使用 262 train / 44 validation，禁止
  读取已消费的 test，所有报告均为 `test_subject_count_read=0`。
- 架构：保留原 kernel 7、dilation `1/2/4/8` CNN；新增 stride-4 卷积将 463 个频点
  降为 116 token，经宽度 32、4 头、2 个 pre-norm self-attention block 后线性上采样。
  局部与全局隐藏特征共同生成双耳逐频率 sigmoid gate，最终 delta 为
  `local delta + gate × global delta`。全局输出层零初始化，保证 epoch 0 与 v3.2
  逐值等价。新增 checkpoint 架构自动推断兼容 SONICOM 和通用重建入口。
- 可解释消融：MLP 与局部 CNN 全部冻结，只训练 `global_context/global_gate` 的 22,356
  个参数；总参数为 196,983。源 checkpoint SHA-256 为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`，训练前后未变。
  新增回归测试验证旧 checkpoint 兼容、初始输出逐值相同、冻结边界与 optimizer 参数组。
- smoke 与测试：`py_compile`、全局分支回归测试、v3.2.1 optimizer 回归和 dual-sampling
  strict-ILD 回归均通过。RTX 5060 上 1 train step + 1 validation block 正常，严格
  HRIR-ILD 反向传播有限，optimizer skip 为 0，smoke 不参与选模。
- 正式训练：执行
  `D:\miniconda3\envs\ml\python.exe -u scripts\run_v33_global_attention_e10.py`。
  10 epoch × 500 step 用时 `1717.274 s`，峰值 CUDA allocated memory `76.500 MiB`，
  optimizer skip 为 0，W&B offline ID `q401byar`。初始 fixed-validation total 为
  `0.661115484`；训练 epoch 中最佳为 epoch 9 的 `0.661141579`，仍回退 `0.00395%`。
- 推理与重建：epoch 9 对 44 人生成 `[2,793,463]` residual，用时 `28.863 s`。
  MATLAB 1 人 smoke 通过；44 人严格重建命令首次因权限审核超时未启动且无残留进程，
  同命令安全重试后完成，用时 `603.7 s`。summary 为 `status=completed`、`split=val`、
  `subject_count=44`、`test_subject_count_read=0`。
- 严格结果：相对 v3.2 epoch 39，全空间 ERB 从 `0.872882` 变为 `0.873159 dB`
  （回退 `0.0317%`，12/44 人改善），对侧 25° ERB 从 `1.364204` 变为
  `1.364686 dB`（回退 `0.0353%`，14/44 人改善），高频从 `3.608567` 变为
  `3.608384 dB`（改善 `0.0051%`，25/44 人改善），水平面严格 ILD 从 `0.596141`
  变为 `0.595707 dB`（改善 `0.0727%`，22/44 人改善）。
- 配对统计：两项 ERB 的 candidate-baseline 均值差为 `+0.0002765 / +0.0004818 dB`，
  paired t-test `p=2.74e-5 / 0.00228`，即绝对退化虽小但方向明确。高频和 ILD 的
  均值差为 `-0.0001836 / -0.0004334 dB`，`p=0.139 / 0.476`，改善不显著。
- 生效诊断：冻结 MLP/局部 CNN 的最大参数差严格为 0；44 人预测相对 v3.2 的平均绝对
  全局 delta 为 `0.012006 dB`，95% 分位 `0.029029 dB`，最大 `0.093135 dB`，说明
  新增分支确实生效，但当前冻结式训练得到的全局修正没有对准最终 ERB 目标。
- 结论：v3.3 不替代 v3.2，作为负结果架构消融保留。若继续该路线，下一项应固定同一
  架构并联合微调局部 CNN 与全局分支，以检验“冻结局部表征”是否限制全局上下文收益；
  不建议在此证据下直接增大 attention 宽度或 channel 数。完整结果位于
  `results/sonicom_mlp_cnn_q26_v33_global_attention_e10_strict_validation/`。

## 2026-08-09：SONICOM Q26 v3.2.1 低风险解冻 MLP 10 epoch

- 工作目标：按用户要求，从固定的 v3.2 seed `20260809` epoch 39 checkpoint 出发，
  解冻 MLP 并训练 10 epoch，观察联合微调效果。实验只使用 262 train / 44 validation；
  test 不参与训练、选模、推理或重建，全部报告记录 `test_subject_count_read=0`。
- 实现：`train_mlp_cnn_v3.py` 新增默认关闭的 `--unfreeze-mlp` 与必须显式提供的
  `--mlp-learning-rate`。默认 v3/v3.1/v3.2 冻结行为不变；联合模式建立 CNN/MLP 两个
  AdamW 参数组，有限性检查和梯度裁剪覆盖全部可训练参数，cosine scheduler 同比例更新
  两组学习率，history/W&B 分别记录两组 LR。新增回归测试验证冻结兼容、两组参数、LR
  和 v3.2.1 stage 名称。
- 低风险控制：正式配置为
  `configs/experiments/sonicom_mlp_cnn_q26_v321_joint_unfreeze_e10.json`，runner 为
  `scripts/run_v321_joint_unfreeze_e10.py`。源 checkpoint SHA-256 固定为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`，训练前后未变；
  新输出使用独立目录。CNN LR 为 `1e-5`，MLP LR 为 `1e-6`，其余双采样、
  `ERB/HF/strict ILD=0.75/0.25/0.75`、seed `20260809`、validation seed `20260805`、
  每轮 500 step 和 96 个 validation block 均与起点一致。
- smoke：1 train step + 1 validation block 在 RTX 5060 上完成，174,627 个参数全部
  可训练，严格 HRIR ILD 联合反向传播有限，optimizer skip 为 0；smoke 不参与选模。
- 正式训练：实际执行
  `D:\miniconda3\envs\ml\python.exe -u scripts\run_v321_joint_unfreeze_e10.py`。10 epoch
  用时 `1021.001 s`，峰值 CUDA allocated memory `601.926 MiB`，无 optimizer skip；
  W&B offline run ID `qdlzionj`。最佳 epoch 7 的固定 validation total 为
  `0.660417`，相对初始 `0.661115` 改善 `0.106%`；residual/ERB/HF/strict ILD 为
  `2.541352 / 1.020789 / 3.578026 / 0.586724 dB`。
- 参数变化：epoch 7 相对源 epoch 39，MLP 的 16 个浮点 tensor 全部改变，但参数整体
  相对 L2 变化仅 `0.0997%`、最大绝对变化 `5.32e-4`；CNN 的 66 个 tensor 相对 L2
  变化 `0.2774%`、最大绝对变化 `3.77e-3`，符合低风险微调设定。
- validation 推理与严格重建：GPU 对 44/44 被试生成 `[2,793,463]` residual，用时
  `12.449 s`；随后 1 人 smoke 与 44 人 MATLAB 严格重建均正常退出，完整重建用时
  `448.2 s`。summary 为 `status=completed`、`split=val`、`subject_count=44`、
  `test_subject_count_read=0`。
- 严格结果：相对 v3.2 epoch 39，v3.2.1 的全空间 ERB 从 `0.872882` 降到
  `0.872411 dB`（改善 `0.0540%`，28/44 人），对侧 25° ERB 从 `1.364204` 变为
  `1.364251 dB`（回退 `0.0034%`，22/44 人），高频从 `3.608567` 降到
  `3.607370 dB`（改善 `0.0332%`，31/44 人），水平面 strict ILD 从 `0.596141`
  降到 `0.592172 dB`（改善 `0.6658%`，28/44 人）。
- 配对统计与结论：全空间 ERB 的 paired t-test `p=0.0093`，strict ILD `p=0.0020`；
  对侧 25°无差异（`p=0.959`），高频变化接近零且 t-test 为 `p=0.058`。因此低学习率
  解冻 MLP 数值稳定，并为全空间 ERB/ILD 带来小幅 validation 收益，但不足以支持全面
  替换 epoch 39；更适合作为联合微调消融。完整结果位于
  `results/sonicom_mlp_cnn_q26_v321_joint_unfreeze_e10_strict_validation/`。

## 2026-08-09：SONICOM Q26 v3.2 epoch 39 冻结 checkpoint 一次性 test 重建

- 授权与边界：用户明确要求使用当前固定的 epoch 39 checkpoint 运行一次 44 人 test
  重建，并禁止修改模型参数。在读取 test 前新增独立冻结协议
  `configs/experiments/sonicom_mlp_cnn_q26_v32_seed20260809_e40_frozen_test.json`，锁定
  checkpoint、四项指标、原 v3.2 epoch 6 对照和“无论结果好坏均保留”的规则；本次
  结果不得用于继续调整 seed、epoch、损失或结构。
- 完整性：冻结 checkpoint 为
  `artifacts/training/sonicom_mlp_cnn_q26_v32_seed20260809_e40/best.pt`，内部 epoch 39、
  174,627 参数、1,367,029 bytes。test 前、Python 推理后和 MATLAB 重建后的 SHA-256
  均为 `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`；没有调用
  训练器、优化器或权重保存逻辑，参数字节级未改动。
- test 推理：实际执行
  `D:\miniconda3\envs\ml\python.exe -u -m mcar.evaluation.predict_sonicom_mlp_cnn_residuals data\processed\sonicom_residual_q26_v1 artifacts\training\sonicom_mlp_cnn_q26_v32_seed20260809_e40\best.pt sonicom_q26_test_mlp_cnn_v32_seed20260809_e40_epoch39_frozen --split test --allow-test --directions-per-block 32`。
  44/44 被试全部生成 `[2,793,463]` residual，耗时 `26.573 s`；报告为
  `status=completed`、`split=test`、`checkpoint_epoch=39`、`test_subject_count_read=44`。
- 严格重建：实际执行
  `matlab.exe -batch "addpath('D:/cuc/CSMT/MCAR/matlab'); mcar.evaluate_sonicom_validation_reconstruction(inf, 'sonicom_mlp_cnn_q26_v32_seed20260809_e40_epoch39_frozen_test_strict', true, 'sonicom_q26_test_mlp_cnn_v32_locked_ild075', 'sonicom_q26_test_mlp_cnn_v32_seed20260809_e40_epoch39_frozen', 'MLP+CNN v3.2 seed 20260809 e40 epoch 39 frozen', 'test', 'sonicom_q26_test_v1', 'sonicom_q26_test_v2', true)"`。
  44 人全部完成，MATLAB 正常退出；summary 为 `status=completed`、`split=test`、
  `subject_count=44`、`test_subject_count_read=44`。
- test 结果：冻结 epoch 39 的全空间 ERB、对侧 25° ERB、对侧高频和水平面严格 ILD
  为 `0.855676 / 1.349975 / 3.590454 / 0.660297 dB`。相对原 locked epoch 6 的
  `0.867805 / 1.365327 / 3.611854 / 0.686999 dB` 分别改善
  `1.398% / 1.124% / 0.592% / 3.887%`；逐被试改善数为
  `42/44 / 36/44 / 40/44 / 29/44`。相对 MCA 分别改善
  `20.932% / 22.671% / 23.585% / 20.387%`。
- 产物与结论：完整 CSV/JSON、直接比较表和三张 44 人图位于
  `results/sonicom_mlp_cnn_q26_v32_seed20260809_e40_epoch39_frozen_test_strict/`。冻结
  epoch 39 在四项 test 总体均值上均优于 locked epoch 6，且 checkpoint 确认未改动；
  本次作为一次性追加比较完整保留，后续不再根据该 test 结果调参或选择新版本。图形
  QA 发现原指标图横轴硬编码为 validation；已把指标绘图提取为公共
  `plot_sonicom_metric_overview.m`，仅从现有 CSV 重画为 test 标签，没有重新读取 test
  数据或 checkpoint，也没有改变任何指标。

## 2026-08-09：SONICOM Q26 v3.2 最佳 seed 40 epoch 与严格 validation 重建

- 用户要求：从三 seed、各 10 epoch 的诊断中选择表现最好的一个，重新训练 40 epoch，
  按最佳 checkpoint 在 validation 上重建。按预声明的固定 validation total loss 最小
  规则，seed `20260809` 的 10-epoch 最佳值 `0.670662` 低于另外两个 seed，因此在正式
  40-epoch 运行前锁定该 seed；test 已消费，本实验全程禁止读取 test。
- 配置与命令：配置为
  `configs/experiments/sonicom_mlp_cnn_q26_v32_seed20260809_e40.json`，runner 为
  `scripts/run_v32_seed20260809_e40.py`。训练实际执行
  `D:\miniconda3\envs\ml\python.exe -u scripts\run_v32_seed20260809_e40.py`；保持
  v3.2 双采样、冻结 MLP、`ERB/HF/strict ILD=0.75/0.25/0.75`、每轮 500 step、
  96 个固定 validation block、training seed `20260809`、validation sampler seed
  `20260805`、AdamW `1e-4` 和 cosine schedule。
- 正式训练：40 epoch 全部完成，用时 `5574.459 s`，RTX 5060 峰值 CUDA allocated
  memory `399.222 MiB`。最佳主 checkpoint 为 epoch 39，固定 validation total 为
  `0.661115`；residual/ERB/对侧高频/strict ILD 分别为
  `2.542327 / 1.021642 / 3.578575 / 0.589368 dB`。epoch 40 的 strict ILD 单项仅低
  `0.000006 dB`，但 total 略高，因此仍按预注册主规则选择 epoch 39。累计 AMP
  optimizer skip 为 `3/20000`，最终 scale `65536`，没有数值失败；W&B offline run ID
  为 `odwrgrln`。
- validation 预测：实际命令为
  `D:\miniconda3\envs\ml\python.exe -u -m mcar.evaluation.predict_sonicom_mlp_cnn_residuals data\processed\sonicom_residual_q26_v1 artifacts\training\sonicom_mlp_cnn_q26_v32_seed20260809_e40\best.pt sonicom_q26_validation_mlp_cnn_v32_seed20260809_e40 --directions-per-block 32`。
  44/44 被试全部生成 `[2,793,463]` residual，用时 `23.485 s`；报告明确记录
  `split=val`、`checkpoint_epoch=39`、`test_subject_count_read=0`。
- 严格重建：先运行 1 人 smoke，确认 P0001 的五方法 ERB/HF/HRIR-ILD 全部有限；随后
  实际执行 `matlab.exe -batch "addpath('D:/cuc/CSMT/MCAR/matlab'); mcar.evaluate_sonicom_validation_reconstruction(inf, 'sonicom_mlp_cnn_q26_v32_seed20260809_e40_strict_validation', true, 'sonicom_q26_validation_mlp_cnn_v3', 'sonicom_q26_validation_mlp_cnn_v32_seed20260809_e40', 'MLP+CNN v3.2 seed 20260809 e40')"`。
  44 人完整 MATLAB 运行用时 `476.1 s`，正常退出；summary 为 `status=completed`、
  `subject_count=44`、`test_subject_count_read=0`。
- 严格 validation 均值：全空间 ERB、对侧 25° ERB、对侧高频和水平面严格 ILD 为
  `0.872882 / 1.364204 / 3.608567 / 0.596141 dB`。相对锁定 v3.2 epoch 6 分别改善
  `1.330% / 1.372% / 0.799% / 4.620%`，逐被试改善数为
  `42/44 / 41/44 / 40/44 / 32/44`；相对 MCA 分别改善
  `20.338% / 22.643% / 24.017% / 28.177%`。
- 产物与结论：完整 CSV/JSON、比较表和三张 44 人结果图位于
  `results/sonicom_mlp_cnn_q26_v32_seed20260809_e40_strict_validation/`。40 epoch 候选在
  validation 上四项一致优于 locked epoch 6，说明模型仍有预算优化空间；但 test 已经
  消费，因此本实验不能替换既有论文主模型或重新评价 test，正式新主张需要新的未见
  拆分或外部数据。

## 2026-08-09：SONICOM Q26 v3.2 三 seed、10 epoch 稳定性诊断

- 用户要求：先评估训练随机性，运行三个 seed、每个 10 epoch，并且不保存每次
  生成的模型。该实验只使用 262 train / 44 validation；已消费 test 不参与，
  runner 状态和最终摘要均记录 `test_subject_count_read=0`。
- 科学控制：训练 seed 固定为 `20260809 / 20260810 / 20260811`，三次共享与正式
  v3.2 相同的 validation sampler seed `20260805`，避免把 validation batch 差异
  误计为训练方差。其余配方保持 v3.2：从正式 v3 初始化、冻结 MLP、双采样、
  `ERB/HF/strict ILD=0.75/0.25/0.75`、每轮 500 step、96 个固定 validation block。
- 存储实现：`train_mlp_cnn_v3.py` 新增默认关闭的 `--no-save-checkpoints` 和可选
  `--validation-seed`。默认行为不变；诊断模式仍写配置、初始评估、history 和报告，
  但不写 `last.pt/best.pt/best_strict_ild.pt`。1 step GPU smoke 实测完成、
  `checkpoint_files_saved=false`、`.pt` 数量为 0。
- 可恢复队列：新增 `scripts/run_v32_seed_stability.py` 与预注册配置
  `configs/experiments/sonicom_mlp_cnn_q26_v32_seed_stability.json`。runner 顺序使用
  单张 RTX 5060；对已完成 seed 可跳过，并在全部完成后生成聚合 CSV/JSON。实际
  主命令为 `D:\miniconda3\envs\ml\python.exe -u scripts\run_v32_seed_stability.py`。
- 完成情况：三个 seed 最佳 epoch 为 `10 / 9 / 10`，耗时分别为
  `854.864 / 773.962 / 775.703 s`，总计 `2404.529 s`；显存峰值均为
  `399.222 MiB`，30 个 epoch 总 optimizer skip 为 0。三个正式输出目录逐一检查，
  `.pt` 文件数全部为 0。
- 最佳固定 validation 均值 ± seed 间样本标准差：total loss
  `0.671027 ± 0.000346`；residual `2.561170 ± 0.000436 dB`；ERB
  `1.033725 ± 0.000601 dB`；对侧高频 `3.607326 ± 0.001387 dB`；strict ILD
  `0.615038 ± 0.001607 dB`。相对标准差依次为
  `0.052% / 0.017% / 0.058% / 0.038% / 0.261%`，训练指标整体高度稳定，ILD
  仍是最敏感的一项。
- 预算诊断：相对正式 v3.2 第 6 epoch 的同口径 validation，三 seed 最佳均值在
  total/residual/ERB/HF/ILD 上分别低 `0.356% / 0.191% / 0.222% / 0.209% /
  1.040%`。由于比较同时包含 seed 和预算变化，且没有保留权重进行完整 MATLAB
  重建，这只能支持“10 epoch 值得在新协议中继续验证”，不能替换锁定论文模型。
- 产物：`results/sonicom_mlp_cnn_q26_v32_seed_stability/` 中包含 3 行逐 seed
  汇总、30 行 history、JSON 统计和 README。论文 v3.2 epoch 6 checkpoint 与既有
  一次性 test 结论保持不变。

## 2026-08-09：FSP-AE-Q26 横向结果可视化

- 工作目标：把已经锁定的 44 人 validation 横向结果整理为无需依赖表格即可理解的图形；不重新训练、不重新评价、不读取 test，也不改变任何统计口径。
- 数据来源：只读取 `results/sonicom_fsp_ae_q26_formal_validation/aggregate_metrics.csv` 与 `per_subject_metrics.csv`。脚本校验七方法 × 四指标的 28 个聚合单元和 44 个逐被试行；生成清单明确记录 `test_subject_count_read=0`。
- 图 1：七方法四指标均值 ± 被试标准差的 2×2 横向条形图。统一保留零起点，MCAR v3.2 与 FSP-AE-Q26 分别用蓝色和橙色突出，避免只看相对百分比掩盖绝对误差。
- 图 2：FSP-AE 与 MCAR v3.2 的逐被试配对散点，使用等比例坐标和恒等线；四项胜出数为 `0/44 / 0/44 / 44/44 / 24/44`，可直观看出两项 ERB 的一致回退、高频的全员改善和 ILD 的个体差异。
- 图 3：FSP-AE 相对 MCA 与 MCAR 的发散条形图，正值表示误差降低、负值表示回退；相对 MCAR 的四项变化为 `-31.16% / -33.60% / +14.68% / +4.37%`。
- 实现与产物：新增 `scripts/plot_fsp_ae_horizontal_comparison.py`，使用可选 `plotting` 依赖 Pillow，从 CSV 确定性生成三张 `1800×1180` PNG。图片和 `manifest.json` 位于 `results/sonicom_fsp_ae_q26_formal_validation/figures/`，并已逐张检查标签、坐标尺度、误差棒和数据一致性。

## 2026-08-06：FSP-AE-Q26 正式预算锁定与严格 validation

- 工作目标：在端到端 smoke 通过后，只使用 262 train / 44 validation 比较预声明的 10/20/40 epoch 预算，按固定 validation 复合损失锁定 checkpoint，再执行 44 人、767 个纯插值方向的七方法严格横向评价。test 不参与训练、选模、推理或评价，全部摘要中的读取数为 0。
- 可恢复训练：`train_fsp_ae.py` 新增原子 checkpoint/JSON 写入、epoch 快照、最佳 epoch、训练随机发生器状态与 `--resume`。实际用 1 train / 1 validation 的两轮恢复 smoke 验证从 epoch 1 快照继续到 epoch 2；优化器、调度器、历史与最佳点均恢复正确。正式配置为 `configs/experiments/sonicom_fsp_ae_q26_formal_budget.json`，快照固定在 epoch 10/20/40，学习率在 epoch 24/36 后从 `1e-3` 降至 `1e-4 / 1e-5`。
- 正式训练：实际命令为 `D:\miniconda3\envs\ml\python.exe -u -m mcar.training.train_fsp_ae configs\experiments\sonicom_fsp_ae_q26_formal_budget.json --device cuda`。每轮遍历全部 262 名 train 被试、每人随机 32 个目标方向；validation 固定为全部 44 人、每人 128 个目标方向。40 epoch 用时 `972.423 s`，峰值 CUDA allocated memory `526.671 MiB`，test 读取为 0。
- 预算结果：epoch 10/20/40 的 validation LSD 为 `3.367172 / 3.195450 / 2.994123 dB`，ITD L1 为 `3.095580e-5 / 2.167831e-5 / 1.304956e-5 s`，复合损失为 `3.444562 / 3.249646 / 3.026746`。两次降学习率后都继续改善，最佳点为 epoch 40，因此按预声明主规则锁定 `artifacts/training/sonicom_fsp_ae_q26_formal_budget_40/best.pt`；没有使用严格指标或 test 事后改变预算。
- 完整推理：实际命令为 `D:\miniconda3\envs\ml\python.exe -u -m mcar.evaluation.predict_sonicom_fsp_ae artifacts\training\sonicom_fsp_ae_q26_formal_budget_40\best.pt sonicom_fsp_ae_q26_formal_validation --split val --target-directions-per-chunk 16 --device cuda`。44/44 被试、793 方向、512 频点与 256 点 HRIR 全部写出，用时 `595.218 s`，test 读取为 0。
- 严格评价：实际命令为 `matlab.exe -batch "addpath('matlab'); mcar.evaluate_sonicom_interpolation_baselines(inf,'sonicom_fsp_ae_q26_formal_strict_validation',false,'val','sonicom_q26_validation_mlp_cnn_v32_locked_ild075',false,'sonicom_fsp_ae_q26_formal_validation')"`，用时 `781.4 s`，正常退出。七方法全部重算，FSP-AE 频率网格最大误差为 0，reference ILD 元数据最大误差为 `9.536e-7 dB`，44 个被试和全部指标均完整有限。
- 正式均值 ± 被试标准差（dB）：FSP-AE-Q26 的全空间 ERB、对侧 25° ERB、对侧高频与水平面严格 ILD 为 `1.160±0.162 / 1.848±0.175 / 3.104±0.254 / 0.598±0.153`；MCA 为 `1.096±0.145 / 1.764±0.177 / 4.749±0.234 / 0.830±0.181`；MCAR v3.2 为 `0.885±0.149 / 1.383±0.165 / 3.638±0.257 / 0.625±0.175`。
- 横向解释：FSP-AE 相对 MCA 的四项变化为 `-5.89% / -4.79% / +34.65% / +27.99%`，逐被试胜出为 `21/44 / 13/44 / 44/44 / 38/44`；相对 MCAR v3.2 为 `-31.16% / -33.60% / +14.68% / +4.37%`，胜出为 `0/44 / 0/44 / 44/44 / 24/44`。它在对侧高频和水平面 ILD 上具有明确互补优势，但全局与对侧 ERB 不及以 MCA 为物理先验的 MCAR，不能视为全面替代。
- 产物与下一步：精选 CSV/JSON、训练轨迹和质量检查位于 `results/sonicom_fsp_ae_q26_formal_validation/`，完整报告位于 `reports/FSP_AE_Q26_VALIDATION_REPORT.md`。正式 checkpoint、完整预测与原始 MATLAB 输出继续位于 Git 忽略的 `artifacts/`。若研究 FSP-AE 与 MCAR 融合，必须在新的未见拆分或外部数据上预声明规则；不得使用已经完成一次性评价的 SONICOM test 继续选模。

## 2026-08-06：FSP-AE-Q26 横向基线端到端验证

- 工作目标：新增频率与声源位置条件自编码器（Frequency and Source Position-conditioned Autoencoder，FSP-AE）横向学习基线，先依次验证官方实现等价、SONICOM 单被试数据口径、短训练预算、validation 推理以及既有 MATLAB 严格评价接口。当前阶段只使用 262 train / 44 validation，test 读取数保持为 0；任何 smoke 或 pilot 结果均不参与最终模型主张。
- 上游与署名：方法来自 Ito 等人的 IEEE OJSP 2025 论文，DOI `10.1109/OJSP.2025.3613132`；参考实现为 `https://github.com/ikets/FSP-AE`，许可证 CC BY 4.0。项目没有提交上游仓库与 checkpoint，只新增带来源说明的兼容实现、测试和 SONICOM 适配入口。
- 官方等价性：使用单独获取的官方 `exp/v1/checkpoint_best.pt`，在相同随机输入下并排载入官方类与本项目类。模型参数数为 `235,065`，预测幅度、ITD 和最小相位加 ITD 的 HRIR 重建最大绝对误差均为 0。实际命令为 `D:\miniconda3\envs\ml\python.exe tests\test_fsp_ae_official_compatibility.py C:\Users\27334\AppData\Local\Temp\codex-fsp-ae-review C:\Users\27334\AppData\Local\Temp\codex-fsp-ae-review\exp\v1\checkpoint_best.pt`。
- SONICOM 适配：固定使用 SONICOM-Q26-v1 的 26 个测量方向、793 个参考方向、`44.1 kHz`、`1024` 点 FFT 和 512 个非直流正频率点。损失保持官方形式，即幅度 LSD 加 `2500 × ITD L1`。为适配 512 个频点并控制显存，每名被试抽取目标方向且分块解码；因此方法名称固定为 FSP-AE-Q26 adaptation，不等同于上游多数据集、随机稀疏度训练协议。
- 数据准备：实际命令为 `D:\miniconda3\envs\ml\python.exe -u -m mcar.data_tools.prepare_sonicom_fsp_ae --splits train val`，用时 `259.5 s`，输出 262 train / 44 validation 缓存。train-only 幅度均值/总体标准差为 `-26.0814984268 / 18.3203195278 dB`，样本数 `212,752,384`；ITD 均值/总体标准差为 `1.7151331712e-6 / 3.7930226798e-4 s`，样本数 `207,766`。归一化记录 `complete=true`，test 被试数与读取数均为 0；安全回归测试确认默认 test 访问会被拒绝。
- 单步 GPU smoke：实际命令为 `D:\miniconda3\envs\ml\python.exe -u -m mcar.training.train_fsp_ae configs\experiments\sonicom_fsp_ae_q26_smoke.json --device cuda`。1 train / 1 validation 被试、1 epoch / 1 step 正常反向传播；validation LSD、ITD L1 和复合损失分别为 `19.384510 dB / 5.683e-4 s / 20.805220`，RTX 5060 用时 `1.733 s`，test 读取数为 0。该 checkpoint 只验证链路。
- 预算 pilot：实际命令为 `D:\miniconda3\envs\ml\python.exe -u -m mcar.training.train_fsp_ae configs\experiments\sonicom_fsp_ae_q26_budget_pilot.json --device cuda`。两轮各随机读取 64 名 train 被试，validation 固定为前 8 名；epoch 1/2 的 train LSD 为 `8.532744 / 5.353823 dB`，validation LSD 为 `5.965827 / 5.177868 dB`，validation 复合损失为 `6.452062 / 5.474776`。总用时 `14.236 s`，峰值 CUDA allocated memory `526.671 MiB`，读取计数为 262 train / 8 validation / 0 test。曲线明确下降，支持扩大预算，但两轮不足以锁定正式 checkpoint。
- 推理与严格评价：用单步 checkpoint 对 P0001 的全部 793 方向分块预测幅度、ITD 和 256 点 HRIR，用时 `14.5 s`；随后运行 `matlab -batch "addpath('matlab'); mcar.evaluate_sonicom_interpolation_baselines(1,'sonicom_fsp_ae_q26_smoke_strict',false,'val','sonicom_q26_validation_mlp_cnn_v32_locked_ild075',false,'sonicom_fsp_ae_q26_smoke_validation')"`。MATLAB `checkcode` 零告警，频率网格、数组形状、严格 HRIR ILD 与既有六基线均完成。FSP-AE smoke 的四项指标为 `13.797 / 11.690 / 12.673 / 18.508 dB`，仅证明接口能完整运行；单步随机模型数值没有科学比较意义，不得进入论文表或用于否定该方法。
- 验证与产物：`test_fsp_ae_model.py`、`test_fsp_ae_data_safety.py`、官方兼容测试和 Python 编译均通过；分块解码与完整解码的逐参数梯度最大绝对误差为 `3.815e-6`。checkpoint、缓存、预测和 MATLAB smoke 位于 Git 忽略的 `artifacts/` 与 `data/processed/`；公共实现位于 `src/mcar/`，配置位于 `configs/experiments/`，完整运行说明位于 `experiments/fsp_ae/README.md`。
- 结论与下一步：官方结构与信号链已经无误差复现，SONICOM-Q26 端到端训练和严格评价已打通，短预算下损失稳定下降。下一步应只在 train/validation 上比较预声明的训练预算和收敛曲线，完成 44 人 validation 严格评价后冻结 FSP-AE checkpoint；由于 SONICOM test 已被既有主实验一次性使用，FSP-AE 不得据此调参或选模，正式横向结论需要新的未见拆分或外部数据。

## 2026-08-05：SONICOM Q26 SUpDEq + NN/Barycentric 横向基线

- 工作目标：在已经冻结且完成一次性 test 评价的 MCAR v3.2 之外，补齐非学习型横向方法。未重新训练、选择 checkpoint 或修改 MCAR；所有方法统一使用 `SONICOM-Q26-v1`、三阶 SH、头半径 `0.09 m`、44 名锁定 test 被试、767 个纯插值方向和既有四项严格指标。
- 方法范围：同一评价脚本同时计算 SH only、SUpDEq + SH、SUpDEq + Natural Neighbor（NN）、SUpDEq + Barycentric、MCA 和冻结的 MCAR v3.2。SH 两条链路固定使用与 MCA 数据导出一致的 Tikhonov epsilon `0.01`；NN/Bary 不使用 magnitude correction，直接调用上游 SUpDEq 的方向均衡、自然邻域和重心插值定义。
- 实现：新增 `matlab/+mcar/evaluate_sonicom_interpolation_baselines.m`。NN/Bary 的空间权重只由固定 Q26 和 793 点参考网格决定，因此用上游 `findvoronoi` 和 `TriangleRayIntersection` 一次构建后供 44 人复用。首名被试同时通过原生 `supdeq_interpHRTF(..., 'SUpDEq', 'NN'/'Bary', nan, ...)` 重算；批量算子与原生入口的最大复频谱差分别为 `4.97e-16` 和 `4.58e-16`。
- 兼容性：SUpDEq 所带 SFS Toolbox 2.5.0 的 `findvoronoi.m` 有两处旧式 `1:size(idx)`，当前 MATLAB 会拒绝向量冒号操作。脚本从未修改或提交 `external/SUpDEq/`，而是在 `artifacts/matlab_compat/` 生成仅把两处循环上界改为 `size(idx,1)` 的临时优先路径副本；其余数值运算保持上游实现不变，并由上述原生入口一致性检查验证。
- smoke：实际命令为 `matlab -batch "addpath('matlab'); mcar.evaluate_sonicom_interpolation_baselines(1,'sonicom_supdeq_nn_bary_q26_smoke',false,'val','sonicom_q26_validation_mlp_cnn_v32_locked_ild075',false)"`。P0001 完成六方法、四指标、原生一致性与 reference ILD 校验；MATLAB `checkcode` 零告警。
- 最终 test：实际命令为 `matlab -batch "addpath('matlab'); mcar.evaluate_sonicom_interpolation_baselines(inf,'sonicom_supdeq_nn_bary_q26_final_test',true,'test','sonicom_q26_test_mlp_cnn_v32_locked_ild075',true)"`，正常退出，44/44 被试完成。`metric_long.csv` 为 `44 × 6 × 4 = 1056` 行，逐被试表和质量检查均为 44 行，全部指标为有限数；reference ILD 元数据最大误差维持在 `1e-6 dB` 量级。
- 最终均值 ± 被试标准差（dB，越低越好）：SH only 为 `2.685±0.107 / 3.835±0.448 / 9.067±0.824 / 3.551±0.664`；SUpDEq + SH 为 `1.882±0.506 / 2.228±0.195 / 5.994±0.344 / 2.018±1.564`；SUpDEq + NN 为 `1.852±0.290 / 2.241±0.212 / 5.595±0.258 / 1.637±0.469`；SUpDEq + Barycentric 为 `1.752±0.257 / 2.182±0.193 / 5.476±0.217 / 1.615±0.442`；MCA 为 `1.082±0.096 / 1.746±0.155 / 4.699±0.266 / 0.829±0.158`；MCAR v3.2 为 `0.868±0.162 / 1.365±0.167 / 3.612±0.277 / 0.687±0.274`。四列依次为全空间 ERB、对侧 25° ERB、对侧高频和水平面严格 ILD。
- 横向结论：MCAR v3.2 相对 SUpDEq + NN 的四项均值降低 `53.14% / 39.06% / 35.45% / 58.03%`，相对 SUpDEq + Barycentric 降低 `50.47% / 37.43% / 34.04% / 57.46%`；两组比较的四项逐被试胜出数均为 `44/44`。MCA/MCAR 两个锚点与既有 `sonicom_mlp_cnn_q26_v32_final_test_strict` 聚合值逐项差为 0，证明新横向链路没有改变冻结评价口径。SUpDEq + SH 在 P0339 出现明显离群（全空间 ERB `4.898 dB`、水平 ILD `11.772 dB`），正文应保留均值 ± 标准差和逐被试/稳健性信息，不把 NN/Bary 相对 SH 的差异作为核心主张。
- 产物：论文表、逐被试表、长表、聚合表、质量检查、JSON、说明和聚合图位于 `results/sonicom_supdeq_nn_bary_q26_final_test/`。核心横向结论是 MCAR 相对两种新增非学习插值基线具有一致且逐被试稳定的优势；这些客观结果不替代后续主观感知实验。

缩写说明：多层感知机（Multi-Layer Perceptron，MLP）、卷积神经网络
（Convolutional Neural Network，CNN）、头相关传输函数（Head-Related
Transfer Function，HRTF）、头相关脉冲响应（Head-Related Impulse Response，
HRIR）、幅度校正与时间对齐插值（Magnitude-Corrected and Time-Aligned
Interpolation，MCA）、耳间电平差（Interaural Level Difference，ILD）、
等效矩形带宽（Equivalent Rectangular Bandwidth，ERB）、平均绝对误差
（Mean Absolute Error，MAE）、均方根误差（Root Mean Squared Error，RMSE）、
Hierarchical Data Format version 5（HDF5）、快速傅里叶逆变换（Inverse Fast
Fourier Transform，IFFT）、快速傅里叶变换（Fast Fourier Transform，FFT）、
方向均衡空间上采样（Spatial Upsampling by Directional Equalization，
SUpDEq）、球谐函数（Spherical Harmonics，SH）、耳间时间差（Interaural Time
Difference，ITD）、空间声学数据格式（Spatially Oriented Format for
Acoustics，SOFA）、对数谱失真（Log-Spectral Distortion，LSD）、图形处理器
（Graphics Processing Unit，GPU）、统一计算设备架构（Compute Unified Device
Architecture，CUDA）、自动混合精度（Automatic Mixed Precision，AMP）、
16 位浮点数（16-bit Floating Point，FP16）、特征级线性调制（Feature-wise
Linear Modulation，FiLM）、Sigmoid 线性单元（Sigmoid Linear Unit，SiLU）、
命令行界面（Command-Line Interface，CLI）、应用程序编程接口（Application
Programming Interface，API）、统一资源定位符（Uniform Resource Locator，
URL）、传输层安全协议（Transport Layer Security，TLS）、Weights & Biases
（W&B）、逗号分隔值（Comma-Separated Values，CSV）、JavaScript 对象表示法
（JavaScript Object Notation，JSON）、便携式网络图形（Portable Network
Graphics，PNG）、多边形文件格式（Polygon File Format，PLY）和便携式文档格式
（Portable Document Format，PDF）、图号（Figure，Fig.）、MATLAB Figure
（FIG）、冲激响应（Impulse Response，IR）、Network Common Data Form version
4（NetCDF4）、标识符（Identifier，ID）、吉字节（gigabyte，GB）、吉比字节
（gibibyte，GiB）和兆二进制字节（mebibyte，MiB）。MATLAB MAT-file 缩写为
MAT。HUTUBS、AXD 和 KU100 是数据集或设备专名，不作首字母展开。

## 2026-08-05：SONICOM Q26 v3.2 论文成果包

- 工作范围：在 test 已正式消费且主模型保持锁定的前提下，只对现有最终 CSV/JSON
  做论文再表达；未读取 SOFA/HDF5/prediction/checkpoint，未训练、推理、选模或调参。
- 可复现实现：新增 `src/mcar/reporting/generate_sonicom_paper_assets.py`。生成器校验
  final summary 必须为 `completed_primary_model_unchanged`、`split=test`、44 名被试；
  输出来源和防泄漏状态写入 `manifest.json`。无额外依赖时生成 CSV、LaTeX 和 SVG；
  环境含 Pillow 时同时生成 300-dpi PNG。
- 论文表格：生成六方法主结果表、v2/v3/v3.1/v3.2 架构与训练策略消融表、两组
  配对 bootstrap 统计表；同时提供 CSV 与 booktabs LaTeX 版本。主表报告 44 名被试
  的 `mean ± subject std`，不是把频点当作独立样本。
- 论文图：生成相对 MCA 的四指标误差降低图、v3.2 相对 v3/v3.1 的 paired-subject
  bootstrap 点区间图、validation-to-test cohort shift 图，均提供可编辑 SVG 和 PNG。
  三张 PNG 已人工视觉检查；配对图首次预览发现长标签裁切和左右面板重叠，扩大
  标签/人数栏间距后重生成并复检通过。
- 实际命令：`C:\Users\27334\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe src/mcar/reporting/generate_sonicom_paper_assets.py`。另以
  `D:\miniconda3\envs\ml\python.exe -m py_compile` 验证生成器语法；`ml` 环境不含
  Pillow，因此只负责语法检查，未修改训练环境。
- 报告与结论：新增 `reports/SONICOM_MLP_CNN_V32_FINAL_RESULTS.md`，覆盖数据协议、
  模型演进、严格重建、主结果、配对统计、泛化、可发表主张和局限。论文主模型仍为
  v3.2 epoch 6；明确保留两项审慎结论：v3.2/v3.1 的 test ILD 无明确差异，且
  v3.2 test ILD 相对 validation 高 `9.92%`。
- 产物：全部位于 `results/sonicom_mlp_cnn_q26_v32_paper/`；生成器和精选成果应
  提交到 Git，大型数据和 checkpoint 继续留在已忽略的 `data/` 与 `artifacts/`。

## 2026-08-04：SONICOM Q26 v3.2 一次性最终 test 解封

- 授权与锁定：用户已在 Codex 对话中明确回复“开始吧，读取 test”，授权读取固定的 44 名 SONICOM test 被试。首次读取前新增 `configs/experiments/sonicom_mlp_cnn_q26_v32_final_test.json`，锁定正式 v3.2 epoch 6 为论文主模型，同时锁定既有 v1、v2、v3、v3.1 checkpoint 仅作基线/消融对照。
- 防泄漏规则：本次 test 结果不得用于修改网络、损失权重、超参数、checkpoint 或论文主模型选择；v3.1 即使某个 test 单项更优，也只按预声明作为 ILD 专用消融报告。任何后续方法改进都必须视为新研究，并使用新的未见拆分或外部数据集。
- 预定流程：以 `allowTest=true` 和已经锁定的 Q26/Tikhonov `0.01` 参数一次性导出 44 名 test residual；随后用五个冻结 checkpoint 生成完整 residual，最后按与 validation 完全相同的 MATLAB 口径计算全空间 ERB、对侧 25° ERB、对侧高频和水平面 HRIR 能量 ILD，并生成逐被试表与 HRTF 总览。
- test 导出：首次命令因 PowerShell 对 MATLAB 字符串 `split=="test"` 的引号解析失败，在读取 SOFA 前退出且未生成 HDF5；改为 `strcmp(ids.split,'test')` 后成功。实际成功命令为 `matlab.exe -batch "addpath('D:/cuc/CSMT/MCAR/matlab'); ids=readtable('D:/cuc/CSMT/MCAR/configs/data/sonicom_subject_split_v1.csv','TextType','string'); ids=ids.subject_id(strcmp(ids.split,'test')); mcar.export_sonicom_residual_dataset(ids, 6, 'sonicom_residual_q26_v1', 1e-2, true, true)"`。6 个 Parallel Computing Toolbox worker 完成 44/44、失败 0；每人 `734,318` 个 residual 样本，所有 HDF5 的 split 属性均为 `test`。
- 防误读扩展：`predict_sonicom_residuals.py` 与 `predict_sonicom_mlp_cnn_residuals.py` 新增向后兼容的 `--split {val,test}` 与 `--allow-test`。默认仍为 validation；`--split test` 缺少 `--allow-test` 时已实测抛出拒绝异常。严格 MATLAB 评估器新增可选 split、v1/v2 prediction 名称与 `allowTest`，test 同样需要 `splitName='test'` 和 `allowTest=true`；原六参数 validation 调用及原文件名保持兼容。
- 五模型 GPU 推理：v1、v2、v3、v3.1、v3.2 分别使用运行名 `sonicom_q26_test_v1`、`sonicom_q26_test_v2`、`sonicom_q26_test_mlp_cnn_v3`、`sonicom_q26_test_mlp_cnn_v31`、`sonicom_q26_test_mlp_cnn_v32_locked_ild075`。命令统一在原 validation 推理入口后增加 `--split test --allow-test`；五组均为 44/44 文件、`2×793×463`、RTX 5060 + AMP，耗时依次为 `10.04 / 9.86 / 11.56 / 11.45 / 11.25 s`，报告均记录 `test_subject_count_read = 44`。
- v3.2 raw 诊断：实际命令为 `D:\miniconda3\envs\ml\python.exe -u -m mcar.evaluation.evaluate_mlp_cnn_v3 data/processed/sonicom_residual_q26_v1 artifacts/training/sonicom_mlp_cnn_q26_v32_locked_ild075/best.pt --split test --allow-test --directions-per-block 32 --direction-weighting solid_angle --interpolation-only --strict-ild --output-dir artifacts/evaluation/sonicom_mlp_cnn_q26_v32_final_test`。在 `31,250,648` 个加权样本上，MCA/v2/v3.2 raw MAE 为 `3.319215 / 2.795766 / 2.574928 dB`；v3.2 相对 MCA/v2 改善 `22.42% / 7.90%`，44/44 人优于 v2。全空间严格 HRIR ILD 从 v2 的 `0.702075` 降至 `0.678218 dB`，改善 `3.40%`。
- 严格重建 smoke：以 P0003 和 `publishResults=false` 完成一次 test smoke，双重授权、split 断言、幅度回填、相位保持、HRIR 与三张图均通过。主结果实际命令为 `matlab.exe -batch "addpath('D:/cuc/CSMT/MCAR/matlab'); mcar.evaluate_sonicom_validation_reconstruction(inf, 'sonicom_mlp_cnn_q26_v32_final_test_strict', true, 'sonicom_q26_test_mlp_cnn_v3', 'sonicom_q26_test_mlp_cnn_v32_locked_ild075', 'MLP+CNN v3.2 locked (ILD 0.75)', 'test', 'sonicom_q26_test_v1', 'sonicom_q26_test_v2', true)"`；v3.1 消融仅把第五个预测改为 `sonicom_q26_test_mlp_cnn_v31`，输出名改为 `sonicom_mlp_cnn_q26_v31_final_test_strict`。两次均为 44/44、正常退出，重复的 MCA/v1/v2/v3 四套聚合值逐位一致。
- 最终严格 test 指标：v3.2 的全空间 ERB / 对侧 25° ERB / 对侧 `>10 kHz` magnitude error / 水平面严格 ILD MAE 为 `0.867805 / 1.365327 / 3.611854 / 0.686999 dB`。相对 MCA 改善 `19.81% / 21.79% / 23.13% / 17.17%`，相对 v2 改善 `5.20% / 4.60% / 7.27% / 4.89%`，相对 v3 改善 `0.493% / 0.400% / 0.227% / 2.830%`；逐被试优于 v3 的人数为 `37/44、31/44、36/44、29/44`。
- validation 到 test：v3.2 的 test 三项幅度误差相对 validation 分别低 `1.90% / 1.29% / 0.71%`，但水平面严格 ILD 从 `0.625015` 上升到 `0.686999 dB`，高 `9.92%`。这是固定 cohort 间的泛化差异，必须如实报告；它不改变预声明模型，并且 v3.2 在 test 上仍将 v3 的 ILD 降低 `2.83%`。
- v3.1 消融：v3.1 四项为 `0.899932 / 1.383141 / 3.684134 / 0.687717 dB`。v3.2 三项幅度分别优于 v3.1 `3.57% / 1.29% / 1.96%`；ILD 均值只低 `0.000718 dB`。配对 100,000 次 subject bootstrap（seed `20260804`）中该 ILD 差的 95% 区间为 `[-0.00687, 0.00815] dB`，v3.2 仅 21/44 人优于 v3.1，故两者 ILD 应报告为无明确差异，不能宣称 v3.2 显著更优。相对 v3 的四项配对平均改善 bootstrap 区间均大于 0；sign-test 未做多重比较校正，原始统计量保存在结果 JSON 中。
- 产物与结论：最终综合摘要、六方法比较表和配对统计位于 `results/sonicom_mlp_cnn_q26_v32_final_test/`；主模型与 v3.1 消融的完整表格、质量检查和 44 人图分别位于 `results/sonicom_mlp_cnn_q26_v32_final_test_strict/` 与 `results/sonicom_mlp_cnn_q26_v31_final_test_strict/`。聚合图和主模型 44 人 HRTF 总览已人工检查，子图与六条曲线完整，P0339 离群个体未隐藏。论文主模型保持预声明的 v3.2 epoch 6，不依据 test 做任何回调或换模；test 至此已使用，后续改进必须使用新拆分或外部数据。

## 2026-08-04：SONICOM Q26 MLP+CNN v3.2 ILD=0.75 正式锁定训练与严格 validation

- 预注册与隔离：在正式训练前新增 `configs/experiments/sonicom_mlp_cnn_q26_v32_locked.json`，锁定 v3.2 双采样方案、`ILD=0.75`、训练预算、checkpoint 选择规则和严格重建门槛。全流程只读取 `262` 名 train 与 `44` 名 validation 被试；训练、完整 residual 推理和 MATLAB 重建均未导出或读取 test，最终报告记录 `test_subject_count_read = 0`。
- 正式训练设置：从正式 v3 checkpoint `artifacts/training/sonicom_mlp_cnn_q26_v3_formal_ild075/best.pt` 初始化，冻结 `100,225` 个 MLP 参数，仅训练 `74,402` 个 CNN 参数，总参数量 `174,627`。双采样每 step 使用 32 个全空间纯插值方向计算 residual/ERB/对侧高频损失，并独立使用 32 个水平面纯插值方向计算严格 HRIR 能量 ILD；权重为 `ERB/HF/ILD = 0.75/0.25/0.75`。预算为 `6 epoch × 500 step`、96 个固定 validation block、AdamW、学习率 `1e-4`、weight decay `1e-5`、cosine schedule、gradient clip `5.0`、AMP、seed `20260804`。
- 正式训练结果：实际命令为 `D:\miniconda3\envs\ml\python.exe -u -m mcar.training.train_mlp_cnn_v3 data/processed/sonicom_residual_q26_v1 artifacts/training/sonicom_mlp_q26_v2_formal_erb075_ild025/best.pt --initial-cnn-checkpoint artifacts/training/sonicom_mlp_cnn_q26_v3_formal_ild075/best.pt --run-name sonicom_mlp_cnn_q26_v32_locked_ild075 --epochs 6 --steps-per-epoch 500 --validation-steps 96 --directions-per-batch 32 --ild-directions-per-batch 32 --interpolation-only --direction-weighted-residual --dual-sampling-strict-ild --ild-loss-mode strict_hrir --erb-weight 0.75 --high-frequency-weight 0.25 --ild-weight 0.75 --learning-rate 1e-4 --weight-decay 1e-5 --gradient-clip 5.0 --seed 20260804 --wandb-mode offline ...`。训练耗时 `892.19 s`，峰值 CUDA allocated memory `399.22 MiB`，0 次 optimizer step 跳过；总损失与严格 ILD 最优点均为 epoch 6。固定 validation 相对初始化的 ERB、对侧高频和严格 ILD 分别改善 `0.329% / 0.325% / 3.161%`。正式 checkpoint 位于已忽略的 `artifacts/training/sonicom_mlp_cnn_q26_v32_locked_ild075/best.pt`，W&B offline run ID 为 `9zg62j1b`。
- 完整 validation 推理：以 epoch 6 `best.pt` 对 44 名 validation 被试生成完整 `2×793×463` residual，实际命令为 `D:\miniconda3\envs\ml\python.exe -u -m mcar.evaluation.predict_sonicom_mlp_cnn_residuals data/processed/sonicom_residual_q26_v1 artifacts/training/sonicom_mlp_cnn_q26_v32_locked_ild075/best.pt sonicom_q26_validation_mlp_cnn_v32_locked_ild075 --directions-per-block 32`。RTX 5060 + AMP 用时 `28.70 s`，44/44 文件完整，输出位于已忽略的 `artifacts/reconstruction/sonicom_q26_validation_mlp_cnn_v32_locked_ild075/`。
- 严格重建：经用户授权启动 MATLAB R2025b，实际命令为 `matlab.exe -batch "addpath('D:/cuc/CSMT/MCAR/matlab'); mcar.evaluate_sonicom_validation_reconstruction(inf, 'sonicom_mlp_cnn_q26_v32_locked_strict_validation', true, 'sonicom_q26_validation_mlp_cnn_v3', 'sonicom_q26_validation_mlp_cnn_v32_locked_ild075', 'MLP+CNN v3.2 locked (ILD 0.75)')"`。进程正常退出，44/44 被试、767 个纯插值方向、72 个水平面方向和 41 个 ERB 频带全部完成，质量检查无异常。
- 回归验证：直接运行 `D:\miniconda3\envs\ml\python.exe tests/test_dual_sampling_strict_ild.py`，合成 HDF5 的全空间 6 方向与水平面 3 方向采样均符合预期，总损失和 CNN 梯度有限，脚本输出 `status = passed`、退出码为 0。
- 正式严格指标：全空间 ERB / 对侧 25° ERB / 对侧 `>10 kHz` magnitude error / 水平面严格 ILD MAE 依次为 `0.884652 / 1.383174 / 3.637638 / 0.625015 dB`。相对正式 v3 的 `0.887467 / 1.388265 / 3.649390 / 0.644804 dB`，四项分别改善 `0.317% / 0.367% / 0.322% / 3.069%`，逐被试优于 v3 的人数分别为 `39/44、29/44、36/44、30/44`。三项幅度均未回退且严格 ILD 改善，因此通过预注册的正式锁定门槛。
- 相对 pilot 与模型定位：相对短预算 v3.2 `ILD=0.75` pilot，正式模型四项再改善 `0.112% / 0.094% / 0.390% / 1.883%`。其严格 ILD 比 v3.1 的 `0.622905 dB` 高约 `0.34%`，但三项幅度显著优于 v3.1；因此正式 v3.2 是论文主线的单模型均衡方案，v3.1 仅保留为 ILD 专用消融/上界，不替代主模型。
- 结果与下一步：精选训练 history 与摘要位于 `results/sonicom_mlp_cnn_q26_v32_formal/`；完整严格重建 CSV、JSON、聚合图、逐被试指标图与 44 人对侧 HRTF 总览位于 `results/sonicom_mlp_cnn_q26_v32_locked_strict_validation/`。v3.2 epoch 6 checkpoint 至此正式锁定，不再基于 validation 调参；下一步是在用户再次明确授权后对封存 test 做唯一一次最终评估，并同步整理论文主表、消融表、统计显著性与图注。W&B 仍为本地 offline，未上传。

## 2026-08-04：SONICOM Q26 MLP+CNN v3.3 对侧高频权重 continuation pilot

- 工作目标：在 v3.2 `ILD=0.75` 的严格重建中观察到对侧高频相对 v3 仅回退 `0.07%`，故不读取 test，以该 checkpoint 为初值，仅将对侧高频损失权重由 `0.25` 提高到 `0.35` 与 `0.50`；ERB 权重和严格 HRIR ILD 权重均固定为 `0.75`。
- GPU pilot：两个候选均为 `3 epoch x 120 step`、固定 32-block validation、双采样 32 个全空间纯插值方向加 32 个水平面 strict-ILD 方向、冻结 MLP、只训练 74,402 个 CNN 参数、AMP、seed `20260804`、W&B offline。`HF=0.35` 和 `HF=0.50` 均无跳步，最优 checkpoint 均为 epoch 3；44 人 validation residual GPU 推理分别耗时 `8.85 s` 与 `10.40 s`，均记录 `test_subject_count_read = 0`。
- 严格 44 人 validation 重建：两组均以 MATLAB 最终 HRTF/HRIR 口径完成 `44/44` 个被试、`767` 个纯插值方向、`72` 个水平面方向和 `41` 个 ERB 频带；quality-check 表完整，test 读取均为 `0`。`HF=0.35` 的全空间 ERB / 对侧 25 度 ERB / 对侧高频 / 严格 ILD 为 `0.890455 / 1.390734 / 3.648732 / 0.638315 dB`；`HF=0.50` 为 `0.890404 / 1.390762 / 3.648336 / 0.638254 dB`。
- 结论：两组 v3.3 都将对侧高频从 v3 的 `3.649390 dB` 略降至 `3.648732 / 3.648336 dB`，但绝对收益仅 `0.00066 / 0.00105 dB`（`0.018% / 0.029%`），不足以抵消相对 v3 的全空间 ERB `0.337% / 0.331%` 回退和对侧 25 度 ERB `0.178% / 0.180%` 回退；严格 ILD 虽改善约 `1.01%`，仍逊于 v3.2 `ILD=0.75`。因此两个 v3.3 候选均不进入正式模型 Pareto 前沿，不进行 test 评估，也不再扩大该简单高频加权路线。
- 产物：配置位于 `configs/experiments/sonicom_mlp_cnn_q26_v33_high_frequency_pilot.json`；完整严格结果位于 `results/sonicom_mlp_cnn_q26_v33_hf035_strict_validation/` 与 `results/sonicom_mlp_cnn_q26_v33_hf050_strict_validation/`；checkpoint 与 residual 继续位于 Git 忽略的 `artifacts/`。
- 下一步：保留 v3 作为高频基线、v3.2 `ILD=0.75` 作为均衡模型、v3.1 作为 ILD 专用模型。后续若继续改进高频，应采用频率选择性或方向条件化的机制，而非继续单独增大标量高频损失权重；test 仍只在用户明确选定单一模型后一次性运行。

## 2026-08-04：SONICOM Q26 MLP+CNN v3.2 四候选严格 validation 重建

- 执行范围：在已锁定的 44 名 SONICOM validation 被试上，对双采样 v3.2 的严格 ILD 权重 `0.75 / 1.0 / 1.25 / 1.5` 全部完成端到端 HRTF/HRIR 重建评估。每次均以 MCA、MLP v1、MLP v2、CNN v3 和对应 v3.2 候选作对照；预测 residual 只回填 MCA 选定频点幅度，保留 MCA 相位和选定频点外复频谱。
- MATLAB 命令形式：`matlab -batch "addpath('matlab'); mcar.evaluate_sonicom_validation_reconstruction(inf, '<output>', true, 'sonicom_q26_validation_mlp_cnn_v3', '<v32-prediction>', 'MLP+CNN v3.2 (ILD <weight>)')"`。由于受限执行环境会使 MATLAB 在启动前报 `File system inconsistency`，实际使用用户授权的正常桌面权限运行；最小 `disp(version)` 已确认 `R2025b Update 2` 正常启动。
- 严格质量控制：先以 `ILD=0.75`、P0001 完成 1 人 smoke，输出完整、`767` 个纯插值方向、`72` 个水平面方向、`41` 个 ERB 频带，reference ILD metadata 最大误差为 `9.25e-7 dB`。四个正式 run 均完成 `44/44` 被试、quality-check 表完整，均记录 `test_subject_count_read = 0`；未导出、未读取 test。
- 严格 44 人结果（单位均为 dB，越低越好；顺序为全空间 ERB / 对侧 25 度 ERB / 对侧高频 / 水平面严格 ILD）：v3 为 `0.887467 / 1.388265 / 3.649390 / 0.644804`；v3.2 `ILD=0.75` 为 `0.885648 / 1.384470 / 3.651891 / 0.637008`；`1.0` 为 `0.886108 / 1.384700 / 3.653414 / 0.637023`；`1.25` 为 `0.886420 / 1.385363 / 3.654458 / 0.636516`；`1.5` 为 `0.886701 / 1.386305 / 3.655021 / 0.635610`。
- 结论：相对 v3，四个候选均降低全空间 ERB、对侧 25 度 ERB 与严格 ILD；代价是对侧高频小幅回退 `0.07% / 0.11% / 0.14% / 0.15%`。`ILD=1.0` 被 `0.75` 在四项指标上同时支配，不保留为候选；v3.2 内部 Pareto 前沿为 `0.75`（最佳三项幅度）、`1.25` 和 `1.5`（逐步更低的严格 ILD）。推荐将 `0.75` 作为 v3.2 的均衡模型，将 `1.5` 作为 v3.2 的 ILD 优先模型；原 v3 仍保留为对侧高频最佳基线，原 v3.1 仍保留为 ILD 最低的专用基线。
- 结果与可视化：四组完整表、JSON、聚合图、逐被试指标图和 44 人对侧 HRTF 总览分别位于 `results/sonicom_mlp_cnn_q26_v32_dual_ild075_strict_validation/`、`results/sonicom_mlp_cnn_q26_v32_dual_ild100_strict_validation/`、`results/sonicom_mlp_cnn_q26_v32_dual_ild125_strict_validation/`、`results/sonicom_mlp_cnn_q26_v32_dual_ild150_strict_validation/`。已人工检查 `ILD=0.75` 的 `validation44_contralateral_hrtf_overview.png`：44 个子图和 MCA/v1/v2/v3/v3.2/reference 六条曲线均完整，方法标签为 v3.2 而非 v3.1。
- 下一步：若要从 v3.2 选定单一正式模型，建议先在不读 test 的约束下将 `ILD=0.75` 与 `1.5` 的取舍写入实验报告，再由用户决定是否对封存 test 进行一次性最终评估；也可先改进对侧高频损失以消除现有的微小回退。

## 2026-08-04：SONICOM Q26 MLP+CNN v3.2 MATLAB 严格重建启动阻塞

- 执行授权与目标：用户已明确授权启动本机 MATLAB；计划先对 `ILD=0.75` 候选运行 1 人严格重建 smoke，再依次完成四个 v3.2 Pareto 候选的 44 人重建评估。
- 实际命令：`matlab -batch "addpath('matlab'); mcar.evaluate_sonicom_validation_reconstruction(1, 'sonicom_mlp_cnn_q26_v32_dual_ild075_strict_smoke', false, 'sonicom_q26_validation_mlp_cnn_v3', 'sonicom_q26_validation_mlp_cnn_v32_dual_ild075', 'MLP+CNN v3.2 (ILD 0.75)')"`。
- 阻塞现象：MATLAB R2025b 在执行任何项目脚本前即两次退出，错误均为 `Fatal Startup Error: System Error: File system inconsistency`。第二次使用独立首选项目录 `artifacts/matlab_pref_v32/` 复试，结果相同；因此该问题不是本次评估器参数或 v3.2 prediction 文件导致。
- 安全处理：首次重试前已观察到两个正在运行且仍响应的 `MATLAB.exe` 进程，无法确认是否为用户会话，故未强制终止；未修改 MATLAB 安装、未清理系统缓存，亦未读取 test 集、未写出任何 smoke 或 44 人重建结果。
- 后续处置：用户关闭原 MATLAB 会话后，受限环境中的启动仍失败；改用用户明确授权的正常桌面权限执行最小 `matlab -batch "disp(version)"` 已成功，随后 1 人 smoke 与四个 44 人严格重建均已完成，详见本日志同日的 v3.2 严格 validation 条目。

## 2026-08-03：SONICOM Q26 MLP+CNN v3.2 双采样严格 ILD smoke

- 工作目标：修复 v3.1 将 residual、ERB、对侧高频与严格 ILD 全部限制在 72 个
  水平面方向上的监督范围缩窄问题。v3.2 每一步分别采样全空间 32 个纯插值方向，
  计算 residual、ERB 与对侧高频；再独立采样 32 个水平面纯插值方向，只计算严格
  HRIR ILD。两项损失在同一次 CNN 更新中相加；MLP 继续冻结，test 不导出、不读取。
- 实现：`train_mlp_cnn_v3.py` 新增默认关闭的
  `--dual-sampling-strict-ild` 与 `--ild-directions-per-batch`。该模式要求
  `--ild-loss-mode strict_hrir --interpolation-only`，并禁止与
  `--horizontal-only` 同时启用，防止全空间幅度批次被意外缩小。训练、固定
  validation、checkpoint 与 W&B 均记录全空间和水平面批次的独立方向统计。
- 回归：新增 `tests/test_dual_sampling_strict_ild.py`。合成 HDF5 测试确认全空间
  sampler 使用 6 个方向、水平面 strict-ILD sampler 使用 3 个方向；总损失和
  CNN 梯度均有限。既有可变方向采样与严格 ILD 梯度测试也通过。
- 真实 GPU smoke：从锁定 SONICOM v3 epoch 10 初始化，以严格 ILD 权重 `1.0`
  运行 `1 epoch × 2 step` 与 2 个固定 validation block。实际训练阶段为
  `cnn_only_frozen_mlp_global_magnitude_horizontal_hrir_ild_v32`，用时 `4.62 s`，
  峰值 CUDA allocated memory `399.22 MiB`，0 个跳步；checkpoint、history、
  配置与 W&B offline run `hh4ybnyc` 均写出。smoke 仅确认链路，不用于模型选择。
- 后续：在不读取 test 的前提下，按预注册配置比较严格 ILD 权重
  `0.75 / 1.0 / 1.25 / 1.5`。先以固定 validation 过滤幅度回退超过 `0.5%` 的
  候选，再仅对幸存者进行 44 人严格 MATLAB 重建并选择 Pareto 前沿。
- Pareto pilot：四个候选均以 epoch 2 为固定 validation 最优点，均无跳步，峰值
  显存均为 `399.22 MiB`。相对初始 v3，`ILD=0.75/1.0/1.25/1.5` 的严格 ILD
  分别改善 `1.26% / 1.47% / 1.56% / 1.67%`；对侧高频分别回退
  `0.25% / 0.33% / 0.39% / 0.42%`，ERB 变化为
  `+0.05% / -0.04% / -0.08% / -0.11%`。四者均通过“不超过 0.5% 幅度回退且
  ILD 改善”的预注册门槛，因此均进入 44 人严格重建阶段。固定 validation 上
  `ILD=1.5` 的 ILD 最低，而 `ILD=0.75` 的 ERB/高频最好；没有单一支配候选。
  比较 CSV/JSON 位于 `results/sonicom_mlp_cnn_q26_v32_dual_pareto_pilot/`，
  test 读取数仍为 0。
- 44 人 prediction 准备：四个 epoch 2 checkpoint 均已在 44 名 validation 被试上
  生成完整 `2×793×463` residual，`ILD=0.75/1.0/1.25/1.5` 的 GPU 推理时间为
  `9.67 / 8.89 / 8.83 / 9.01 s`，每个 run 均有 44 个 prediction 文件，test
  读取数均为 0。预测位于 Git 忽略的 `artifacts/reconstruction/`，待获得本机
  MATLAB 明确执行授权后进行四次最终严格重建。
- MATLAB 评估器准备：`evaluate_sonicom_validation_reconstruction` 新增向后兼容的
  可选第六参数 `comparisonLabel`。默认仍显示 `MLP+CNN v3.1`；v3.2 的四次独立
  重建将传入相应候选名称，使图形、长表方法名和 summary 配置不把 v3.2 误标为
  v3.1。尚未启动 MATLAB，因此本改动目前只完成静态检查。


## 2026-08-01：SONICOM Q26 MLP+CNN v3.1 水平面严格 ILD 微调

- 工作目标：从锁定 v3 epoch 10 继续微调冻结 MLP 的 CNN，使训练损失与最终
  水平面 HRIR 能量 ILD 完全对齐，同时保持 residual、ERB 和对侧高频收益；
  44 名 test 继续不导出、不读取。
- 对齐改动：`BinauralSpectrumSampler` 和 `train_mlp_cnn_v3.py` 新增默认关闭的
  `horizontal_only`/`--horizontal-only`。SONICOM v3.1 将 767 个纯插值方向与
  零仰角条件取交集，得到 72 个水平面方向；HUTUBS 与既有 v3/v3.1 默认行为不变。
  回归测试覆盖插值 mask 与水平面 mask 的交集。
- 代码 smoke：从 v3 checkpoint 初始化，1 epoch、2 step、32 方向，严格 HRIR
  IFFT-ILD 可正常反传，0 个跳步，峰值 CUDA allocated memory `221.96 MiB`。
- 权重预实验：固定 ERB/高频权重 `0.75/0.25`、学习率 `1e-4`，比较严格水平面
  ILD 权重 `0.5/1.0/2.0`，每组 `3 epoch × 120 step`。三组 residual、ERB、
  高频和严格 ILD 均相对初始 v3 改善；严格 ILD 改善依次为
  `0.40% / 0.68% / 0.93%`。按预声明规则选择 `2.0`，其 residual/ERB/高频仍
  改善 `0.87% / 1.22% / 1.01%`。比较位于
  `results/sonicom_mlp_cnn_q26_v31_weight_pilot/`。
- 正式训练：从 v3 epoch 10 初始化，冻结 100,225 参数 MLP，只训练 74,402 参数
  CNN；`6 epoch × 500 step`、96 个固定 validation block、每 batch 32 个水平面
  方向、AdamW、cosine schedule、AMP、seed `20260731`。用时 `296.56 s`，峰值
  显存 `221.96 MiB`，0 个跳步；总目标与严格 ILD 最优点均为 epoch 6。
- 固定水平面 validation：v3 到 v3.1 的 residual 为
  `2.721718 → 2.682673 dB`，ERB 为 `1.130511 → 1.099314 dB`，对侧高频为
  `3.638068 → 3.585039 dB`，严格 HRIR ILD 为 `0.659525 → 0.635705 dB`，分别
  改善 `1.43% / 2.76% / 1.46% / 3.61%`。
- 完整诊断：44 人、767 个纯插值方向的 raw residual MAE/RMSE 为
  `2.637748 / 3.979081 dB`，相对 v2 MAE 改善 `6.06%`；全空间严格 ILD 为
  `0.629976 dB`。由于本模型只针对水平面微调，全空间 ILD 仅作诊断，不用于
  checkpoint 选择。
- 产物：正式 checkpoint 位于已忽略的
  `artifacts/training/sonicom_mlp_cnn_q26_v31_horizontal_formal_ild200/best.pt`；
  精选曲线与摘要位于 `results/sonicom_mlp_cnn_q26_v31_formal/`。44 人完整 residual
  已生成到 `artifacts/reconstruction/sonicom_q26_validation_mlp_cnn_v31/`，用时
  `9.23 s`。W&B run `ta1d0lzk` 当前为本地 offline，未经授权不上传。
- MATLAB smoke：经用户明确授权后，以 P0001 运行 MCA/v1/v2/v3/v3.1 五方法
  端到端严格重建，进程正常退出。v3.1 相对 v3 的全空间 ERB、对侧 25° ERB、
  对侧高频分别回退 `4.32% / 0.55% / 1.88%`，水平面 ILD 改善 `2.24%`；
  smoke 只用于确认链路，不用于模型结论。
- 44 人严格重建：正式 MATLAB 批处理正常退出。MCA/v2/v3/v3.1 的全空间 ERB
  分别为 `1.095738 / 0.934826 / 0.887467 / 0.917835 dB`；对侧 25° ERB 为
  `1.763508 / 1.449139 / 1.388265 / 1.398997 dB`；对侧高频为
  `4.749208 / 3.965865 / 3.649390 / 3.712955 dB`；水平面严格 ILD MAE 为
  `0.830014 / 0.653461 / 0.644804 / 0.622905 dB`。
- 最终结论：v3.1 相对 v3 将目标水平面 ILD 改善 `3.40%`，29/44 人改善；
  全空间 ERB、对侧 25° ERB、对侧高频分别回退
  `3.42% / 0.77% / 1.74%`。但 v3.1 三项幅度指标仍相对 v2 改善
  `1.82% / 3.46% / 6.38%`，ILD 相对 v2 改善 `4.68%`。因此保留 v3 为
  幅度均衡基线，保留 v3.1 为 ILD 优化基线；下一轮若要单模型兼顾，应在
  validation 上做多目标/Pareto 折中，而不能宣称 v3.1 全面优于 v3。
- 质量与产物：44 行逐被试表、880 行 method-metric 长表、44 行质量检查和三张
  总览图位于 `results/sonicom_mlp_cnn_q26_v31_strict_validation/`；配置位于
  `configs/experiments/sonicom_mlp_cnn_q26_v31_strict_validation.json`。
  五方法 44 人 HRTF 总览已人工检查，44 个子图与六条曲线完整；test 读取数仍为 0。

## 2026-08-01：SONICOM Q26 MLP+CNN v3 严格 validation 重建评估

- 工作目标：把锁定 v3 epoch 10 的完整 residual 回填到 MCA 幅度，在固定 44 名
  validation 被试上按最终 HRTF/HRIR 口径计算 `AKerbError`、对侧高频和水平面
  严格 ILD，并生成 44 人 HRTF 总览；test 继续不导出、不读取。
- 推理实现：新增 `predict_sonicom_mlp_cnn_residuals.py`，以完整双耳频谱调用
  `ResidualMLPCNN`，按方向分块并原子写出 `2×793×463` residual。1 人 smoke
  通过后正式处理 44 人，用时 `14.34 s`，RTX 5060、AMP、174,627 参数，输出
  位于已忽略的 `artifacts/reconstruction/sonicom_q26_validation_mlp_cnn_v3/`。
- MATLAB 实现：将 `evaluate_sonicom_validation_reconstruction.m` 扩展为可选
  v3 模式，第四个参数指定 v3 prediction run；省略时仍严格复现原 MCA/v1/v2
  行为。代码检查只有既存的动态扩展性能提示，1 人端到端 smoke 和 44 人正式
  运行均成功，正式 MATLAB 进程退出码为 0。
- 最终指标：MCA/v2/v3 的全空间 ERB 为
  `1.095738 / 0.934826 / 0.887467 dB`，v3 相对 v2 改善 `5.07%`、相对 MCA
  改善 `19.01%`，44/44 人优于 v2。
- 对侧 25° ERB：MCA/v2/v3 为
  `1.763508 / 1.449139 / 1.388265 dB`，v3 相对 v2 改善 `4.20%`、相对 MCA
  改善 `21.28%`，41/44 人优于 v2。
- 对侧高频：MCA/v2/v3 为
  `4.749208 / 3.965865 / 3.649390 dB`，v3 相对 v2 改善 `7.98%`、相对 MCA
  改善 `23.16%`，44/44 人优于 v2。
- 水平面严格 ILD：MCA/v2/v3 为
  `0.830014 / 0.653461 / 0.644804 dB`，v3 相对 v2 改善 `1.32%`、相对 MCA
  改善 `22.31%`，27/44 人优于 v2。总体均值改善与全空间严格 ILD 诊断方向一致，
  但被试级稳定性弱于 ERB 和高频。
- 质量与产物：44 个 per-subject 行、704 个 method-metric 长表行、44 个质量检查
  行均完整且全部数值有限；配置位于
  `configs/experiments/sonicom_mlp_cnn_q26_v3_strict_validation.json`，完整 CSV、
  JSON、聚合图、逐被试指标图和 44 人对侧 HRTF 总览位于
  `results/sonicom_mlp_cnn_q26_v3_strict_validation/`。总览图已人工检查，包含
  Reference、MCA、MLP v1、MLP v2 和 MLP+CNN v3 五条曲线且 44 个子图完整。
- 结论与下一步：v3 在最终重建口径上四项总体指标均优于 v2，尤其对侧高频新增
  `7.98%` 收益，证明频率 CNN 的主要价值成立。下一步可锁定 v3 为 SONICOM
  MLP+CNN 基线，并在 validation 上开展与水平面 HRIR 能量完全对齐的 v3.1 ILD
  微调预实验；在损失与训练预算再次锁定前继续保持 test 零读取。

## 2026-08-01：SONICOM Q26 MLP+CNN v3 正式训练与全量 validation

- 工作目标：以已锁定的 SONICOM MLP v2 epoch 10 为基线，在不读取 44 名 test
  的前提下加入沿频率轴建模局部谱形的 1D CNN，完成权重预实验、CNN-only
  正式训练和 44 名 validation 的全量评估。
- 兼容性改动：`train_mlp_cnn_v3.py` 新增 `--interpolation-only` 与
  `--direction-weighted-residual`，并将两项写入 checkpoint/config；SONICOM
  只采样 767 个纯插值方向，residual 损失按 solid-angle weight 加权。默认值
  保持关闭，因此不改变既有 HUTUBS v3/v3.1 入口。`evaluate_mlp_cnn_v3.py`
  同步支持纯插值筛选、solid-angle 加权与加权样本累计。
- 模型：锁定 v2 MLP 的 `100,225` 个参数，只训练 `74,402` 参数的频率 1D CNN；
  总参数为 `174,627`，CNN channel 为 48。CNN 输出层零初始化，训练前模型与
  v2 输出严格一致。最小代码 smoke 为 1 epoch、2 step，初始 CNN delta 为 0，
  峰值 CUDA allocated memory `70.32 MiB`，无跳步。
- 权重预实验：固定 `ERB / 对侧高频 = 0.75 / 0.25`，比较 ILD proxy 权重
  `0.25 / 0.50 / 0.75`，每组 `3 epoch × 120 step`。预先声明的选择规则要求
  residual、ERB 和高频不退化，再最大化 ERB、高频和 ILD 三项改善率最小值。
  三组短 run 的 ILD proxy 分别退化 `3.50% / 2.78% / 2.08%`，因此选择退化
  最小且满足其他约束的 `0.75`，计划依靠正式预算与最终 HRIR 口径复核。
- 正式训练：损失权重 `0.75 / 0.25 / 0.75`，`10 epoch × 500 step`、96 个固定
  validation block、每 block 32 个方向、AdamW、cosine schedule、AMP、seed
  `20260731`。运行用时 `376.47 s`，峰值 CUDA allocated memory
  `220.46 MiB`，跳过 optimizer step 数为 0，最佳点为 epoch 10。
- 固定 validation proxy：v2 到 v3 的 residual MAE 为
  `2.825701 → 2.599614 dB`，ERB 为 `1.119363 → 1.052854 dB`，对侧高频为
  `3.993006 → 3.659349 dB`，ILD proxy 为 `0.651290 → 0.621770 dB`；分别改善
  `8.00% / 5.94% / 8.36% / 4.53%`。短预算 ILD 退化在正式训练中已反转。
- 完整 validation raw 指标：44 人、767 个纯插值方向、`31,250,648` 个逐频点
  样本上，MCA/v2/v3 MAE 为 `3.322289 / 2.807803 / 2.581873 dB`，v3 RMSE 为
  `3.935166 dB`。v3 MAE 相对 v2 改善 `8.05%`、相对 MCA 改善 `22.29%`，并在
  44/44 人上优于 v2。
- 全空间严格 HRIR ILD 诊断：v2/v3 MAE 为
  `0.641157 / 0.621980 dB`，v3 改善 `2.99%`，35/44 人改善。该指标直接使用
  reference HRIR 能量，但采用全空间口径；后续仍需 MATLAB 重建评估水平面 ILD。
- 产物：权重比较位于 `results/sonicom_mlp_cnn_q26_v3_weight_pilot/`，正式曲线
  与摘要位于 `results/sonicom_mlp_cnn_q26_v3_formal/`，锁定配置为
  `configs/experiments/sonicom_mlp_cnn_q26_v3_locked.json`。checkpoint 和完整
  per-subject 评估继续保存在 Git 忽略的 `artifacts/`。W&B run ID 为
  `5mjq0y51`，当前为本地 offline，未经用户明确授权不上传。
- 运行问题：第一次完整评估在 44 人计算结束后触发旧 HUTUBS 固定参考值断言；
  已把断言限定到 `hutubs_residual_v1_n03` 并成功重跑，训练和 checkpoint 无需
  重跑。此次恢复也先检查了后台进程与现有输出，未启动重复训练。
- 结论与下一步：冻结 MLP、只训练频率 CNN 已同时改善 raw residual、四项固定
  proxy 和全空间严格 ILD，v3 可进入最终 reconstruction 复核。下一步输出 44 人
  v3 residual，按 MATLAB 最终口径计算 `AKerbError`、对侧高频、水平面严格
  HRIR ILD，并生成 44 人 HRTF 总览；test 读取数继续保持为 0。

## 2026-08-01：SONICOM Q26 MLP v1/v2 严格 validation 重建评估

- 工作目标：只在锁定的 44 名 validation 被试上，将正式 v1/v2 residual 回填
  到 MCA 幅度，按最终 HRTF/HRIR 重建口径比较 `AKerbError`、对侧高频误差和
  水平面严格 ILD；test 继续不导出、不读取。
- 推理实现：新增 `predict_sonicom_residuals.py`，强制只接受 split=`val` 的
  HDF5，原子写出完整 `2×793×463` residual。v1 epoch 19 与 v2 epoch 10 的
  44 人 GPU 推理分别用时 `8.661 s` 和 `8.432 s`，RTX 5060、AMP、100,225
  参数；两套预测位于已忽略的 `artifacts/reconstruction/`。
- 严格重建：新增 `evaluate_sonicom_validation_reconstruction.m`。在
  `50–20000 Hz` 把 residual 加到 MCA log magnitude，保留 MCA 相位和频段外
  复数谱，转换双边频谱后 IFFT 并裁剪到原始 256-sample HRIR。原始 validation
  SOFA 只用于 reference HRIR，未重新运行 MCA。
- 指标口径：ERB 使用与 HUTUBS 论文复现一致的 `AKerbError`，范围
  `50–22050 Hz`；全空间和耳特异对侧 25° 区域均只纳入 767 个纯插值方向并按
  SONICOM solid-angle weight 加权。对侧高频为 `10–20 kHz` 对侧开放半球的
  面积加权幅度 MAE。严格 ILD 为水平面纯插值方向上完整 HRIR 能量比的 MAE。
- 全空间 ERB：MCA/v1/v2 均值为
  `1.095738 / 0.987171 / 0.934826 dB`；v2 相对 MCA 改善 `14.69%`，相对 v1
  再改善 `5.30%`。v2 在 44/44 人上优于 v1。
- 对侧 25° ERB：MCA/v1/v2 为
  `1.763508 / 1.520947 / 1.449139 dB`；v2 相对 MCA 改善 `17.83%`，相对 v1
  再改善 `4.72%`。v2 在 44/44 人上优于 v1。
- 对侧高频：MCA/v1/v2 为
  `4.749208 / 3.989510 / 3.965865 dB`；v2 相对 MCA 改善 `16.49%`，相对 v1
  再改善 `0.59%`。v2 在 33/44 人上优于 v1，说明正式 v2 保持了 v1 的主要
  高频收益，但增量较小。
- 水平面严格 ILD：MCA/v1/v2 为
  `0.830014 / 0.741741 / 0.653461 dB`；v2 相对 MCA 改善 `21.27%`，相对 v1
  再改善 `11.90%`，37/44 人改善。该严格结果与训练期 ILD proxy 的方向一致。
- 质量检查：44 个 per-subject 行、528 个 method-metric 长表行和全部数值有限；
  HDF5 reference ILD 与原始 SOFA HRIR 重算值的最大差异为 `9.54e-7 dB`。
  单人 smoke 先发现并修正 HDF5 行/列向量隐式扩展问题，正式运行无异常。
- 产物：配置位于
  `configs/experiments/sonicom_mlp_q26_v2_strict_validation.json`；完整 CSV、
  JSON、44 人指标曲线、聚合柱状图和对侧 HRTF 总览位于
  `results/sonicom_mlp_q26_v2_strict_validation/`。响应流在正式 MATLAB 运行
  中途断开，但后台进程正常完成并写出 `status=completed`，没有重跑或并发写入。
- 结论与下一步：严格 reconstruction 指标确认 v2 的主要增益集中在 ERB 与
  ILD，高频相对 v1 仅小幅改善但没有总体退化。SONICOM v2 可视为 validation
  阶段锁定；在解封 44 名 test 前，应先决定是否把当前 v2 作为最终 MLP 基线，
  或继续在同一开发划分上训练 SONICOM MLP+CNN。

## 2026-07-31：SONICOM Q26 residual MLP v2 权重选择与正式锁定

- 工作目标：在不读取 44 名 test 的前提下，用少量固定预算选择 SONICOM v2
  的 ERB、对侧高频和 ILD spectral proxy 权重，并完成正式训练与完整
  validation raw residual 评估。
- 选择规则：所有候选必须满足 residual MAE 相对初始 v1 退化不超过 `0.5%`；
  首先最大化 ERB、对侧高频、ILD 三项相对改善率的最小值，再依次比较三项平均
  改善、residual MAE 和较小总权重。该规则在候选训练前锁定，避免根据结果临时
  改口径。
- 候选设置：四组均使用 3 epoch × 120 step、32 个固定 validation block、
  32 个插值方向、完整 463 频点、方向面积加权损失、AdamW `3e-4`、AMP 和 seed
  `20260731`。权重分别为 balanced `0.50/0.25/0.25`、erb_heavy
  `0.75/0.25/0.25`、ild_heavy `0.50/0.25/0.50`、erb_ild_heavy
  `0.75/0.25/0.50`。
- 候选结果：四组的 ERB/高频/ILD 最小相对改善率依次为
  `0.5260% / 0.5305% / 0.4986% / 0.5050%`；三项平均改善率依次为
  `3.0861% / 3.1430% / 3.5051% / 3.5107%`。根据预定的首要 maximin 规则，
  选择 `erb_heavy = 0.75/0.25/0.25`，而不是用平均值事后偏向更高 ILD 权重。
- 正式训练：10 epoch × 500 step、96 个固定 validation block、cosine schedule，
  其余采样和优化设置与 pilot 相同。RTX 5060 上用时 `296.339 s`，峰值 CUDA
  allocated memory `144.292 MiB`，最佳 epoch 为 10。
- 固定 validation：初始 v1 的 residual/ERB/高频/ILD 为
  `2.847674 / 1.175123 / 4.027407 / 0.738389 dB`；epoch 10 为
  `2.825701 / 1.119363 / 3.993006 / 0.651290 dB`，分别改善
  `0.77% / 4.74% / 0.85% / 11.80%`，四项同时改善。
- 完整 raw validation：遍历 44 人、767 个纯插值方向和 `31,250,648` 个样本。
  v2 MAE/RMSE 为 `2.807803 / 4.210356 dB`，锁定 v1 为
  `2.825515 / 4.218639 dB`，分别改善 `0.63% / 0.20%`；相对 MCA
  `3.322289 dB`，v2 MAE 改善 `15.49%`。
- 锁定产物：正式 checkpoint 为
  `artifacts/training/sonicom_mlp_q26_v2_formal_erb075_ild025/best.pt`；锁定配置为
  `configs/experiments/sonicom_mlp_q26_v2_locked.json`；权重比较和正式精选结果
  分别位于 `results/sonicom_mlp_q26_v2_weight_pilot/` 与
  `results/sonicom_mlp_q26_v2_formal/`。checkpoint 和完整运行状态继续由
  `artifacts/` 忽略。
- W&B 状态：balanced run `buhhtjwi` 已在线。三个新增 pilot run
  `n2jlw45k / vpwlyqc8 / q5g8hg2h` 和正式 run `h7kmgoxc` 已完整保存在本地
  offline 目录。由于上传可能包含配置与本机路径信息，当前未获得明确上传授权，
  因此没有绕过限制进行同步。
- 防泄漏与结论：处理目录仍只有 262 train + 44 validation，test 读取数为 0。
  v2 已按 validation 锁定；下一步是在 validation 上回填 residual，执行严格
  重建 ERB magnitude error、对侧高频误差与 HRIR ILD 评估。严格指标完成前
  不导出 test。

## 2026-07-31：SONICOM Q26 residual MLP v2 感知损失 smoke

- 工作目标：以锁定的 v1 epoch 19 checkpoint 初始化 MLP v2，验证 SONICOM
  双耳完整频谱、ERB、对侧高频、ILD spectral proxy、面积权重和 W&B 链路；
  开发阶段继续不读取 44 名 test。
- 加权策略：`BinauralSpectrumSampler` 新增 interpolation-only 模式，在 767
  个合格方向中均匀无放回抽样；损失内部使用第六列 solid-angle weight。
  residual SmoothL1 和 MAE 同步改为可选方向加权，SONICOM 启用该选项，避免
  方向按面积抽样后又在损失内加权造成双重面积权重。HUTUBS 默认行为不变。
- 损失与配置：模型仍为 100,225 参数；每 batch 包含双耳、32 方向和全部 463
  频点，共 29,632 个样本；总损失为面积加权 residual SmoothL1，加
  `0.50 × ERB + 0.25 × 对侧高频 + 0.25 × ILD proxy`，各感知项除以
  target std。smoke 为 3 epoch × 120 step、32 个固定 validation block、
  AdamW `3e-4`、AMP、seed `20260731`。
- 实现与验证：v2 训练器新增 W&B、固定 validation sampler、epoch 0 的 v1
  基准、CUDA 峰值和完整报告。合成可变方向测试、Python compileall 和真实
  `2×8×463` GPU 前向/反向均通过；真实梯度 smoke 四项损失有限，峰值显存
  `51.89 MiB`。
- smoke 结果：正式短程耗时 `33.146 s`，峰值显存 `144.292 MiB`，最佳 epoch
  3。固定 validation 从 v1 到 v2 的 residual/ERB/高频/ILD 为
  `2.853286→2.849955 / 1.170491→1.152948 / 4.053157→4.031838 /
  0.758558→0.703688 dB`，分别改善 `0.12% / 1.50% / 0.53% / 7.23%`；
  复合 loss 改善 `0.85%`。
- 完整 raw validation：遍历 44 人、767 插值方向和 `31,250,648` 个样本，
  v2 MAE/RMSE 为 `2.824472 / 4.229951 dB`；v1 为
  `2.825515 / 4.218639 dB`。MAE 略改善 `0.037%`，RMSE 小幅波动，说明短程
  感知微调没有以明显 raw residual 退化换取代理指标。
- W&B 与产物：run ID `buhhtjwi`，地址
  `https://wandb.ai/luyoung/mcar-sonicom/runs/buhhtjwi`；配置位于
  `configs/experiments/sonicom_mlp_q26_v2_smoke.json`，精选 history/summary
  位于 `results/sonicom_mlp_q26_v2_smoke/`，checkpoint 继续由 artifacts 忽略。
- 结论与下一步：默认 HUTUBS v2 权重在 SONICOM 上方向正确，四项 validation
  代理均改善。下一步只在 train/validation 上比较少量权重候选与正式预算，
  优先保留 ILD 权重、检查更高 ERB 权重是否扩大听觉收益且不损害 raw MAE；
  test 继续不导出。

## 2026-07-31：SONICOM Q26 residual MLP v1 正式训练与预算锁定

- 工作目标：比较 12/20 epoch 正式预算，以完整 validation 的 767 点面积加权
  residual MAE 锁定 SONICOM MLP v1；44 名 test 继续未导出、未读取。
- 公共设置：262 train、44 validation；ResidualMLP 宽度 128、3 个 residual
  block、100,225 参数；每 epoch 600 个训练 block、96 个固定 validation
  block，每 block 8192 样本；solid-angle、interpolation-only、AdamW、初始
  学习率 `1e-3`、cosine schedule、AMP、seed `20260731`。
- 12-epoch 候选：耗时 `346.345 s`，抽样 validation 最佳为 epoch 12，MAE
  `2.912809 dB`。完整 validation MAE/RMSE 为
  `2.872643 / 4.265143 dB`，相对 MCA MAE 改善 `13.53%`；W&B run ID
  `6wi22jix`。
- 20-epoch 候选：耗时 `602.747 s`，抽样 validation 最佳为 epoch 19，MAE
  `2.866083 dB`，epoch 20 为 `2.866198 dB`。完整 validation MAE/RMSE 为
  `2.825515 / 4.218639 dB`，相对 MCA `3.322289 / 5.029170 dB` 的 MAE
  改善 `14.95%`；W&B run ID `ek1nvkhh`。
- 选择结论：20-epoch 候选的完整 validation MAE 相对 12-epoch 再降低
  `1.64%`，因此锁定 `sonicom_mlp_q26_v1_e20/best.pt` 的 epoch 19。后期曲线
  已基本平台化，暂不继续增加 epoch。
- 中断恢复：首次启动 20-epoch 命令时界面中断；检查确认无 Python/W&B 进程、
  输出目录或半成品 checkpoint，随后从头安全重跑，不存在 checkpoint 混合。
- 产物：预算配置和锁定配置位于 `configs/experiments/`；比较表、两条 history
  和摘要位于 `results/sonicom_mlp_q26_v1_budget_comparison/`。checkpoint 与
  W&B 本地状态继续由 `artifacts/` 忽略。
- 下一步：以 epoch 19 checkpoint 初始化 SONICOM MLP v2，在 train/validation
  上调节 ERB、对侧高频和 ILD 感知损失；v2 锁定前继续不导出 test。

## 2026-07-31：SONICOM Q26 residual MLP smoke training

- 工作目标：在不读取锁定 test 的前提下，验证 SONICOM 全量开发数据能否进入
  现有 100,225 参数轻量 ResidualMLP，并打通 RTX 5060、自动混合精度、球面
  面积采样、纯插值 mask、固定 validation、完整 validation 和 W&B 在线记录。
- 兼容改动：`ResidualBlockSampler` 新增可选 `solid_angle` 方向抽样和
  `interpolation_only` 模式，默认仍为 HUTUBS 原有的 uniform + 全方向。
  SONICOM 训练按 `direction_features[:, 5]` 的正面积权重，从 767 个
  interpolation directions 无放回抽样，不让 26 个稀疏输入点稀释训练指标。
- 训练器：`train_mlp_v1` 新增 W&B 参数、固定 validation seed、epoch 耗时、
  CUDA 峰值、训练报告和 W&B run 元数据。每个 epoch 都从同一 validation seed
  重建 sampler，保证候选 epoch 使用相同随机 validation blocks。
- smoke 配置：262 train、44 validation、0 test；3 epoch × 120 train step，
  每步 `64 directions × 128 frequencies = 8192` 样本；每 epoch 64 个固定
  validation block；宽度 128、3 个 residual block、AdamW、初始学习率
  `1e-3`、AMP，seed `20260731`。
- 运行结果：RTX 5060 上耗时 `24.095 s`，峰值 CUDA allocated memory
  `54.090 MiB`，无异常退出。抽样 validation 的 MCA MAE 固定为
  `3.374041 dB`；模型在 epoch 1/2/3 为
  `3.235233 / 3.172290 / 3.138882 dB`，最佳 epoch 3 改善 `6.97%`。
- 完整 validation：扩展 `evaluate_residual_mlp` 支持 interpolation-only 和
  solid-angle weighting。遍历 44 名 validation、767 方向、双耳和 463
  频点，共 `31,250,648` 个样本；MCA 的 MAE/RMSE 为
  `3.322289 / 5.029170 dB`，模型为 `3.100075 / 4.580354 dB`，MAE 改善
  `6.69%`。
- W&B：在线同步完成，run ID `hmekqxio`，地址为
  `https://wandb.ai/luyoung/mcar-sonicom/runs/hmekqxio`。第一次在受限沙箱
  内初始化时网络访问被拒，尚未开始训练；随后使用受控网络权限重跑成功，不存在
  checkpoint 混用。
- 防泄漏与产物：复查 processed 目录仍为 306 个 train/validation HDF5，
  锁定 test 文件为 0。checkpoint、W&B 本地状态和完整配置位于已忽略的
  `artifacts/training/sonicom_mlp_q26_smoke_v1/`；精选 history 与摘要位于
  `results/sonicom_mlp_q26_smoke_v1/`。
- 结论与下一步：面积加权 SONICOM-only MLP 链路已经可训练，且极短训练已取得
  明确 validation 改善，但 smoke 预算不足以作为最终结论。下一步应保持同一
  数据口径，先比较 12/20 epoch 的正式 MLP v1 学习曲线并以完整 validation
  锁定 checkpoint；再在不读取 test 的情况下进入感知损失 v2。

## 2026-07-31：SONICOM Q26 正式 MCA residual 数据导出

- 工作目标：使用 pilot 锁定的 `SONICOM-Q26-v1`、三阶球谐和
  Tikhonov epsilon `0.01`，生成 SONICOM-only 网络训练所需的完整
  train/validation MCA residual 数据；继续禁止读取 44 名锁定 test 被试。
- 实际执行：从固定 split 读取全部非 test ID，以 6 个 MATLAB process worker
  调用 `mcar.export_sonicom_residual_dataset`。导出器使用原子写入和完成文件
  跳过机制；从首个到最后一个文件完成历时约 `733 s`。
- 导出结果：306/306 个逐被试 HDF5 完成，其中 `262 train / 44 validation /
  0 test`；无缺失、无意外 ID、无 test 泄漏。总大小 `4,254,380,898 bytes`
  （约 `3.96 GiB`），每名被试为 2 耳 × 793 方向 × 463 频点，共
  `734,318` 个 residual 样本。
- 全量验证：使用 `validate_residual_hdf5 --summary-only
  --require-strict-ild` 遍历 306 个文件和 `224,701,308` 个样本。所有必需
  dataset、有限值、动态方向尺寸、26 个稀疏索引、767 点纯插值 mask、方向
  权重、完整频谱 strict-ILD 元数据和完成标记均通过；`reference - MCA =
  target_residual` 的最大恒等误差为 `0 dB`。
- train-only 统计：`compute_training_statistics` 仅纳入 262 名 train 的
  `192,391,316` 个样本；target residual 的 mean/std/mean absolute 为
  `-0.062128 / 4.954315 / 3.250640 dB`，范围为
  `-87.844574～82.283562 dB`。完整统计保存于已忽略数据目录的
  `training_statistics.json`，实验配置已记录其相对路径。
- 产物策略：大型 HDF5 及完整 train ID 统计继续位于
  `data/processed/sonicom_residual_q26_v1/` 并由 Git 忽略；可提交的运行摘要
  位于 `results/sonicom_data_preparation/full_export_v1/summary.json`。
- 结论与下一步：SONICOM-only residual 网络所需的开发数据已经就绪且无
  test 泄漏。下一步先对现有轻量 residual MLP 做最小改动的 SONICOM smoke
  training，确认 793 点面积权重、动态方向 sampler、validation 全量评估和
  W&B 日志，再锁定正式训练预算；test 继续保持未导出。

## 2026-07-31：SONICOM Q26 MCA pilot 与 Tikhonov 参数锁定

- 工作目标：在不读取 44 名锁定 test 被试的前提下，验证 SONICOM-Q26-v1
  能否完整运行 SUpDEq + SH + MCA，建立 793/767 双口径 HDF5，并根据
  validation 选择无权三阶球谐最小二乘的 Tikhonov epsilon。
- 导出实现：新增
  `matlab/+mcar/export_sonicom_residual_dataset.m`。每名被试直接对
  `(793, 2, 256)` HRIR 做 1024 点 FFT，从固定 26 个真实测量索引取得稀疏
  HRTF，以实际 Q26 方位角/余纬度做三阶球谐变换，再在原始 793 点上运行 MCA。
  头半径使用所有 SOFA 共有的 nominal `0.09 m` ReceiverPosition，频率范围
  为 `50 Hz～20 kHz` 共 463 点。
- 防泄漏：导出器默认 `allowTest=false`，在初始化 SUpDEq 或读取 SOFA 前检查
  固定 split；用 P0003 实测保护逻辑，得到预期的 locked-test 错误并通过断言。
  正式开发命令只能处理 train/validation，最终评估必须显式启用 test。
- HDF5 schema 2.0：每名被试保存
  `mca/reference/correction/target_residual`，Python 布局均为
  `[2, 793, 463]`；同时保存 793 点面积权重、26 个稀疏索引、767 点纯插值
  mask、MCA 选中频点相位、50 个频带外复频点、参考严格 HRIR-ILD 和 256 点
  HRIR 裁剪长度。每文件包含 `734,318` 个 residual 样本。
- Python 兼容：`src/mcar/data.py`、训练统计和 HDF5 验证器从固定 900 方向改为
  动态方向数，文件发现从 HUTUBS 专用 `pp*/n*.h5` 泛化为 `*/*.h5`；已用旧
  HUTUBS `pp1/n03.h5` 回归验证 900 点路径不变。SONICOM 的方向特征第六列为
  `normalized_solid_angle_weight`，训练统计从 HDF5 属性读取名称，不再硬编码
  `fliege_weight`。
- pilot 被试：P0002/P0243/P0238 为 train，P0001/P0277/P0242 为 validation，
  分别覆盖 `reference_eq_001/006/008`，test 为 0。首先用 P0002/P0001
  完成真实双被试 smoke test，约 10 秒/人，平均绝对 residual 分别为
  `3.311601/3.290179 dB`；纯插值 767 点为 `3.331109/3.310949 dB`，
  证明全 793 点会被输入方向轻微乐观化。
- 参数扫描：固定 Q26 和其余 MCA 参数，扫描 epsilon
  `0 / 1e-8 / 1e-6 / 1e-4 / 1e-2 / 3e-2 / 1e-1 / 3e-1 / 1`。选择规则在
  汇总前固定为“validation 767 点 ERB proxy MAE 最低；再依次以 residual MAE、
  严格 HRIR-ILD MAE 和较小 epsilon 打破并列”。
- validation 结果：epsilon `0` 的 767 点 residual/ERB/对侧高频/严格 ILD
  为 `3.289591 / 1.275841 / 4.779132 / 0.977043 dB`；`0.01` 为
  `3.276051 / 1.264245 / 4.772924 / 0.986837 dB`；`0.03` 为
  `3.272364 / 1.265599 / 4.786129 / 1.035681 dB`；`0.1` 已全面退化至
  `3.380890 / 1.404554 / 5.006937 / 1.340158 dB`。因此 `0.01` 是 ERB
  内部最优点并被锁定；相对无正则 ERB 改善约 `0.91%`，代价是严格 ILD
  增加约 `1.00%`。
- 全量 pilot 验证：9 档 × 6 人共 54 个 HDF5、`39,653,172` 个样本全部通过
  strict-ILD、有限值、频率覆盖、方向权重、26/767 mask 互补、残差恒等和
  complete 标记检查；最大 MCA correction 恒等误差
  `7.6294e-06 dB`，所有 residual 恒等误差为 0。全量 pilot HDF5 约
  `713.28 MiB`，由 `data/processed/*` 忽略。锁定 epsilon 的 3 名 train
  被试可正常生成训练统计；逐点 sampler 输出 `(512, 7)`，严格 ILD 双耳 sampler
  输出 `(2, 4, 463, 7)`，证明 793 点数据已可直接进入现有 MLP/CNN 管线。
- 结果与配置：评估脚本为
  `src/mcar/evaluation/evaluate_sonicom_tikhonov_pilot.py`；逐被试 CSV、
  聚合 CSV 和 JSON 位于
  `results/sonicom_data_preparation/tikhonov_pilot_v1/`；正式锁定参数位于
  `configs/experiments/sonicom_q26_residual_v1.json`；完整命令见
  `experiments/sonicom_data/README.md`。
- 结论与下一步：SONICOM Q26 MCA residual 链路已经打通，epsilon 锁定为
  `0.01`。下一步用 6 个 MATLAB worker 导出 262 train + 44 validation
  共 306 人，继续保持 test 未导出；全量 HDF5 通过后只用 262 名 train 计算
  归一化统计，再开始 SONICOM-only residual MLP 基线。

## 2026-07-31：SONICOM-Q26-v1 几何配置与固定被试划分

- 工作目标：在正式 350 人 SONICOM 测量队列上建立不会伪造缺失方向、可供
  MCA residual 导出的稀疏输入网格、参考方向权重和全新锁定被试划分。
- 网格审计：350 个 SOFA 共用同一个 793 点 `SourcePosition`；网格包含
  `-45/-30/-20/-10/0/10/20/30/45/60/75°` 共 11 个水平圆环，每环
  72 个、方位角间隔 5°，另有一个 90° 北极点。实测区域为仰角
  `-45°～90°`，覆盖球面立体角 `10.726068 sr`，不存在南极及
  `-45°` 以下方向。
- Lebedev 兼容性：将 SUpDEq Lebedev `N=3` 的 26 个目标方向直接映射到
  SONICOM，只有 17 个精确重合，最近邻只有 25 个唯一点；平均最近角误差
  `3.350581°`，最大误差来自南极点并达到 `45°`。因此不采用直接最近邻，
  也不通过高阶球谐外推伪造缺失球冠。
- Q26 方法：新增
  `src/mcar/data_tools/prepare_sonicom_configs.py`。算法固定包含北极点，
  遍历另一个中垂面种子，并贪心选择 12 对真实左右镜像方向；主目标为最大化
  最小大圆距离，并以正则化三阶实球谐 Gram 矩阵的 log determinant 打破并列。
  选点只使用几何坐标，不读取 HRIR/HRTF 数值。
- Q26 结果：`SONICOM-Q26-v1` 含 26 个唯一实测方向且严格左右对称；最小
  点间角 `31.915839°`，对全部 793 点的最近覆盖距离平均 `15.080257°`、
  95 百分位 `24.814217°`、最大 `33.108876°`。三阶实球谐设计矩阵为
  `26 × 16`、秩 16、条件数 `2.476614`，通过预设的秩和条件数检查。
- 方向权重：依据各仰角圆环相邻中点形成的球面单元，在实测
  `-45°～90°` 球冠内计算面积权重并归一化至 1。配置同时标记 26 个稀疏
  输入方向和 767 个纯插值评估方向；正式指标以 767 点为主、793 点为辅助。
- 被试划分：使用固定 seed `20260731`，首先按 11 个
  `Free Field EQ File` 分层并做整数配额，再在每个 EQ 组内用由 seed 和
  EQ 名称共同派生的稳定 SHA-256 seed 洗牌。结果严格为
  `262 train / 44 validation / 44 test`，无交叉、遗漏或重复。仅含 2 人的
  `reference_eq_005` 无法覆盖三个 split，按比例均保留在 train；其余 10 个
  EQ 批次均覆盖 validation 和 test。
- 元数据审计：train/validation/test 已知年龄人数为 `148/21/21`；出生性别
  缺失人数为 `114/24/24`。年龄、出生性别、族裔、EQ 文件和 subject ID
  只用于分层或审计，不作为首版网络输入；后续归一化统计只能读取 train。
- 输出配置：`configs/data/sonicom_sparse_grid_q26_v1.csv` 保存 Q26 索引、
  坐标、镜像索引和选点角色；`sonicom_reference_grid_v1.csv` 保存 793 点
  坐标、面积权重和 mask；`sonicom_subject_split_v1.csv` 保存固定划分；
  `sonicom_preparation_report_v1.json` 保存源网格 SHA-256、全部几何指标和
  分层分布。
- 验证：350 个 SOFA 网格逐文件一致性检查、离线合成网格测试、Q26 唯一性与
  镜像闭包、球谐满秩与条件数、面积权重正值与和为 1、split 行列总数检查、
  Python `compileall` 和配置实际生成均通过。
- 结论与下一步：SONICOM 不能作为严格 Lebedev 论文复现的直接替代，但
  `SONICOM-Q26-v1` 可作为测量数据上的受控扩展实验。下一步实现独立 SONICOM
  MCA residual 导出入口，使用实际 Q26 坐标、三阶无权最小二乘球谐变换、
  nominal `0.09 m` 头半径和待由 validation pilot 选择的 Tikhonov 参数；
  先运行少量 train/validation 被试，不读取 44 名锁定 test。

## 2026-07-30：SONICOM 正式训练队列下载器与全队列可用性检查

- 工作目标：把 SONICOM 官网完整数据集整理为可复现、可断点续传且不会因官网
  元数据变化而静默漂移的训练数据下载流程；本阶段只实现与验证下载，不改动既有
  HUTUBS 锁定实验。
- 数据集与版本：数据根目录为
  `https://transfer.ic.ac.uk:9090/2022_SONICOM-HRTF-DATASET`；使用官网当前
  `metadata.csv` 和日期固定的
  `important_information/Outliers_2026-05-06.csv`。当前快照包含 372 条
  被试元数据，其中 364 条 `HRTF == TRUE`。
- 选择口径：仅保留 `HRTF == TRUE`、不在官方 HRTF 异常清单且自由场均衡
  文件未标记为 `EQ File Corrupted` 的测量被试。12 名官方异常被试、8 名
  无 HRTF 被试和 2 名自由场均衡损坏被试合计排除 22 名，冻结后的干净队列为
  350 名。脚本在官网元数据行数或队列人数变化时默认停止，只有人工复核后才允许
  使用 `--allow-metadata-drift`。
- 文件版本：每名被试只下载
  `PXXXX/HRTF/HRTF/44kHz/PXXXX_FreeFieldCompMinPhase_44kHz.sofa`。该版本
  采用 44.1 kHz 采样率，已进行 5 ms 截窗和最小相位自由场补偿，同时保留个体
  耳间时间差（ITD）；不下载去 ITD 版本、原始 50 ms 版本、合成 HRTF、扫描网格
  和其他重复采样率版本。
- 实现：新增 `src/mcar/data_tools/download_sonicom.py`，仅依赖 Python
  标准库，支持并发请求、网络重试、HTTP Range 断点续传、`.part` 临时文件、
  完成后的原子替换、远端 `Content-Length` 校验、重复运行跳过、pilot 被试
  子集、只下载元数据和全队列只读检查。
- 可追溯输出：正式运行会把官方元数据和异常说明原样保存，并生成
  `manifests/clean_subjects.csv`、`manifests/excluded_subjects.csv` 和
  `manifests/selection_report.json`；报告中记录选择规则及所有元数据文件的
  SHA-256 摘要。SOFA 默认写入
  `data/HRTF/sonicom_measured_ffcmp_minphase_44k1/subjects/`，整个大型数据
  目录继续由 Git 忽略。
- 全队列网络检查：实际执行
  `D:\miniconda3\envs\ml\python.exe -m mcar.data_tools.download_sonicom
  --dry-run --workers 8`；350/350 个目标 SOFA 的 HEAD 请求全部成功，远端
  总量为 `872.82 MiB`，耗时约 `162.8 s`，且 dry run 未写入数据文件。
- 真实下载：先在已忽略的 `artifacts/download_smoke/sonicom_p0001/` 完成
  P0001 下载及重复运行跳过检查，再向正式目录下载全队列。首轮完成 347/350 个，
  并保留 P0310 的零字节 `.part`；续传命令只指定 P0310、P0312、P0313，三者
  全部下载成功，最终得到 350/350 个 SOFA、无残留 `.part`，总量
  `872.82 MiB`。
- 全量内容审计：manifest 中 350 名被试与 `subjects/` 中 350 个文件一一对应，
  无缺失或多余文件。所有文件的 `Data.IR` 形状均为 `(793, 2, 256)`、采样率
  均为 `44100 Hz`、SOFA 约定均为 `SimpleFreeFieldHRIR`，所有脉冲响应数值
  有限，且 350 名被试的 793 点 `SourcePosition` 方向网格完全一致。
- 验证：Python `compileall`、命令行 `--help`、离线选择规则测试、全队列
  网络检查、真实单文件下载及重复运行、全量下载和 350 文件内容审计均通过。
  正式数据位于
  `data/HRTF/sonicom_measured_ffcmp_minphase_44k1/`，并由 Git 忽略。
- 结论与下一步：下载链路、冻结队列和正式训练数据已经就绪，无数据侧阻塞。
  下一步先检查 MCA 输入所需的稀疏方向是否能从 SONICOM 统一网格直接抽取，
  再按自由场均衡文件或其他可用属性进行 subject-wise 分层，建立
  `262 train / 44 validation / 44 test` 的新数据划分，原 HUTUBS test 结论
  保持锁定。

## 2026-07-30：项目文档英文缩写首次出现规范化

- 工作目标：修复项目文档中英文缩写首次出现时缺少全称的问题，降低说明页、
  实验日志和技术报告的阅读门槛。
- 修改范围：检查全部受 Git 跟踪的 Markdown 文档及结果目录中的文本报告；
  不修改命令、文件名、类名、数据和实验结论。
- 统一规则：同一文档内首次自然语言出现采用“中文名称（English Full Name，
  缩写）”或“English Full Name（缩写）”；HUTUBS、AXD、KU100 等无正式首字母
  展开方式的数据集或设备名称按专名保留。
- 核对来源：MCA 使用论文题名中的 `Magnitude-Corrected and Time-Aligned
  Interpolation`；SUpDEq 使用官方文档中的 `Spatial Upsampling by Directional
  Equalization`；SOFA 使用规范名称 `Spatially Oriented Format for Acoustics`。
- 公式兼容性：将 v3.1 报告中两处 `\[...\]` 显示公式同步改为项目约定的
  `$$\begin{aligned}...\end{aligned}$$` 形式，公式内容未变。
- 验证结果：缩写首次出现自动审计、`git diff --check` 和报告公式分隔符检查均
  通过；修改仅涉及文档。

## 2026-07-27：MLP + CNN v3.1 严格 HRIR-ILD 对齐微调与锁定测试

- 实验目标：修复 v3 训练 ILD proxy 与最终严格 HRIR 能量 ILD 口径不一致的问题；不扩大网络，只在 train/validation 上实现可微严格 ILD、完成受控选型，再对锁定的 12 名 test 被试进行一次最终评估。
- 数据集与划分：HUTUBS simulated，Lebedev `N=3` MCA residual；固定 72 train / 12 validation / 12 test subject-wise split。为 train/validation 84 个 HDF5 导出 schema 1.1，新增 MCA 选中频点相位、50 个频带外复频点、频点索引、参考 HRIR ILD、513 点单边谱长度和 256 点 HRIR 裁剪长度。开发和选型阶段没有读取 test。
- 数据完整性：`python -m mcar.data_tools.validate_residual_hdf5 ... --require-strict-ild` 验证 84 个文件、`70,005,600` 个样本、72/12 split、完整 513 点频率覆盖、有限值和零 residual 恒等误差 0，全部通过。
- 实现：`src/mcar/losses.py` 用预测幅度、原 MCA 相位和未修改的频带外复谱重建 513 点单边谱，镜像为 1024 点双边谱，IFFT 后裁剪 256 点 HRIR，再计算双耳能量 ILD MAE；全链路支持 autograd。训练器新增 `--ild-loss-mode strict_hrir`、`--initial-cnn-checkpoint`、严格/旧 proxy 双日志，以及 `best.pt` 和 `best_strict_ild.pt` 双 checkpoint。评估器新增 `--strict-ild` 完整方向穷举评估。
- 测试：合成严格 ILD 单元测试得到零误差 `0`、扰动误差 `0.285678 dB`、最大梯度 `0.245291`；真实 pp91 CUDA smoke test 的 zero-init identity error 为 0，严格 ILD 与旧 proxy 不同，第二步共有 62 个 CNN 参数梯度张量非零。
- 候选实验：从 v2 零初始化 CNN 的严格 ILD 权重 0.25 在完整 validation 上得到 `0.660663 dB`，未超过 v2；权重 1.0 的严格-ILD checkpoint 得到 `0.655195 dB`，但 raw residual MAE 升至 `2.110205 dB`。因此最终改用原 v3 epoch 9 为初始化，以 `1e-4` 学习率和严格 ILD 权重 1.0 微调冻结 MLP 后的 CNN。
- 正式训练：6 epoch × 500 step，32 directions/block，训练参数 74,402、总参数 174,627，耗时 `286.389 s`，0 个跳过 step，峰值 CUDA allocated memory `221.963 MiB`。W&B run：`https://wandb.ai/luyoung/mcar-mlp-cnn-v31/runs/yhl3z70n`。
- validation 锁定：查看 test 前以完整 validation 锁定总损失最优 epoch 3。v2 / 原 v3 / v3.1 的 raw residual MAE 为 `2.139782 / 2.072657 / 2.078053 dB`，严格 ILD MAE 为 `0.658844 / 0.656999 / 0.654040 dB`。v3.1 相对 v2 的 validation 严格 ILD 改善 `0.729%`。
- 锁定 raw test：共 `10,000,800` 样本。MCA / v2 / v3.1 MAE 为 `2.600671 / 2.168373 / 2.092767 dB`，RMSE 为 `4.186115 / 3.552562 / 3.448699 dB`。v3.1 相对 v2 MAE 改善 `3.487%`、相对 MCA 改善 `19.530%`，12/12 被试优于 v2；相对原 v3 `2.091041 dB` 仅回退约 `0.083%`。
- 严格重建 test：MCA 的全空间 ERB / 对侧 25° ERB / 对侧 `>10 kHz` / 水平面 ILD MAE 为 `0.802741 / 1.860286 / 4.258336 / 0.885425 dB`；v3.1 为 `0.585398 / 1.310334 / 3.618947 / 0.652495 dB`，相对 MCA 改善 `27.08% / 29.56% / 15.01% / 26.31%`，四项均为 12/12 被试改善。
- v3.1 与 v3：严格 ILD 由 `0.661566` 降到 `0.652495 dB`，改善 `1.371%`，8/12 被试改善；对侧高频改善 `0.033%`；全空间 ERB 和对侧 25° ERB 分别小幅回退 `0.138%` 和 `0.454%`。与 v2 相比，前三项仍改善 `4.269% / 2.219% / 2.726%` 且均为 12/12，但 ILD 仍高 `0.893%`，只有 4/12 优于 v2。
- 重建质量：12 被试最大左右相位误差 `6.12e-16 / 5.99e-16 rad`，最大幅度恒等误差均为 `7.11e-15 dB`，全部通过断言。已生成 12 张逐被试图、12 被试 HRTF 总览、指标总览、v1/v2/v3/v3.1 总对比与逐被试 ILD 对比，并完成视觉检查。
- 输出位置：本机 checkpoint 与全量产物位于 `artifacts/training/mlp_cnn_n03_v31_finetune_v3_strict_ild_wandb_online/`、`artifacts/evaluation/` 和 `artifacts/reconstruction/mlp_cnn_n03_v31_finetune_v3/`；精选结果位于 `results/residual_mlp_cnn/mlp_cnn_n03_v31/`；完整报告位于 `reports/MLP_CNN_V31_REPORT.md`。
- 结论：严格 HRIR-ILD 损失有效收回了 v3 的大部分 ILD 退化，同时几乎保持 v3 幅度性能。v3.1 是当前推荐的 MLP-CNN 多指标折中模型，但 v2 仍保有最低 ILD 单项结果。当前 test 已解封，后续不得据此继续调权重；下一步应在新 validation 设计或交叉验证上研究 Pareto 选型、双耳结构约束及跨稀疏阶数泛化。

> 2026-07-26 项目已按职责重构。本文此前记录的命令保留当时的历史路径；
> 当前路径与入口以 `README.md`、`docs/PROJECT_STRUCTURE.md` 和
> `experiments/` 为准。

本文件是项目唯一的进展与实验记录。以日期作为二级标题，按时间倒序记录实验过程、结果、结论、阻塞项和下一步。跨电脑继续项目时，先阅读本文件。

## 记录模板

```md
## YYYY-MM-DD：工作或实验名称

- 实验目标：
- 数据集与版本：
- 稀疏网格与采样点数：
- 方法与关键参数：
- 实验过程：
- 指标结果：
- 输出文件位置：
- 阻塞项：
- 结论与下一步：
```

## 2026-07-26：项目结构职责化重构

- 工作目标：将 MCA/SUpDEq demo 从项目主体降为传统基线，并把公共实现、实验定义、本地运行产物、精选结果和报告彻底分离。
- 回滚锚点：重构前提交为 `a02e337`；创建标签 `pre-project-restructure-20260726`，在分支 `codex/project-structure-refactor` 上实施。
- 结构调整：MCA 入口迁入 `baselines/mca/`；Python 公共代码抽取为 `src/mcar/` package；MATLAB 公共函数迁入 `matlab/+mcar/`；数据划分和锁定参数归入 `configs/`；实验说明归入 `experiments/`；技术报告迁入 `reports/`。
- 数据与产物：约 2.2 GB 本地 SUpDEq、HDF5、checkpoint、MAT、W&B 缓存和批处理结果均在同一磁盘内无损移动到 `external/`、`data/processed/` 和 `artifacts/`，未删除实验数据。v1/v2/v3 精选 CSV、JSON 和 PNG 迁入 `results/`；v3 首次把正式训练及严格重建轻量结果纳入 Git 候选范围。
- 代码调整：移除 Python 脚本的 `sys.path` 拼接，改用 `mcar.*` 包导入；新增 `pyproject.toml` 和项目根路径 helper；训练输出统一写入 `artifacts/training/`。MATLAB 入口改用 `mcar.*` package，SUpDEq 路径统一为 `external/SUpDEq/`。
- 文档调整：重写根 README，使项目主体明确为 MCA 后残差学习；新增 `docs/PROJECT_STRUCTURE.md`、baseline/experiment/config/result 说明，并同步更新 AGENTS 路径约定。
- 验证结果：Python 3.9 `compileall` 通过；所有核心 package 导入和 v1/v3 CLI `--help` 通过；11 个 JSON 文件解析通过；MATLAB 在沙箱外成功解析 5 个新 package/baseline 入口；真实 pp91、2 方向、463 频点的 v3 前后向 smoke test 通过，输出形状为 `2 × 2 × 463`，zero-init identity error 为 0，CNN 第二步得到 62 个非零梯度张量。
- 输出位置：结构和迁移表见 `docs/PROJECT_STRUCTURE.md`；当前可执行命令见根 README 与 `experiments/`。
- 结论与下一步：重构未改变模型参数、数据划分或已锁定实验结论。下一步应在新结构上实现 v3.1 可微 ILD 损失，并保持源码复用和 artifacts/results 发布边界。

## 2026-07-26：MLP + CNN v3 首版完整技术报告

- 工作目标：在 v3 正式训练、完整 validation、锁定 test 和严格 HRTF 重建全部完成后，形成与 v1/v2 报告体系一致的独立技术报告。
- 报告文件：新增 `docs/MLP_CNN_V3_REPORT.md`，共 25 章，覆盖摘要、版本演进、需求与验收、数据协议、模型选型、MLP-CNN 融合、参数与感受野、初始化、双耳完整频谱采样、复合损失、W&B、训练过程、validation/test、防泄漏协议、MCA 幅度回填、严格指标、逐被试结果、工程结构、完整复现命令、有效性威胁、v3.1 建议和项目结论。
- 数值审校：报告中的训练、完整 residual、严格 ERB/高频/ILD、资源与重建质量数字均与本地 JSON/CSV/MATLAB 输出交叉核对。补充确认 v3 相对 v2 的全空间 ERB、对侧 25° ERB和对侧高频均为 12/12 test 被试改善；严格 ILD 只有 3/12 优于 v2，9/12 回升，说明 ILD 退化是系统性弱点而非单个异常值。
- 口径修正：旧 v3 README 所写“91 点感受野”只对应膨胀卷积主干；报告按实际网络计算，计入 kernel-7 stem 后完整理论感受野为 97 点，约 `4.18 kHz`，并已同步修正 README。
- Git 状态：报告与源码保持可提交，checkpoint、HDF5、W&B 缓存、MAT 和批量生成图片继续由嵌套 `.gitignore` 忽略；本次未提交或推送，也未改动用户已有的根 `.gitignore` 变更。
- 结论与下一步：首版报告已经可以用于项目交接、阶段汇报和论文方法章节素材。后续若进入 v3.1，优先实现与最终 HRIR 能量定义对齐的可微 ILD 损失，并在不访问 test 的情况下完成 validation 选型。

## 2026-07-25：MLP + CNN v3 锁定 test 与严格 HRTF 重建评估

- 实验目标：在完整 validation 已确认且模型、超参数和 epoch 9 checkpoint 全部锁定后，解封固定的 12 个 test 被试；先穷举 raw residual，再把 v3 residual 回填到 MCA 幅度，并使用与 v1/v2 完全相同的 MATLAB 口径计算论文对应的 ERB magnitude error、对侧高频误差和 ILD，同时生成 12 被试 HRTF 总览。
- test 范围与防泄漏：test 被试为 pp8、pp18、pp22、pp26、pp31、pp33、pp45、pp47、pp59、pp70、pp73、pp81。仅在 validation 阶段完成并明确锁定 epoch 9 后执行 `--split test --allow-test`；test 结果未用于重新训练、模型选择或超参数调整。
- 完整 raw residual：12 个 HDF5 × 2 耳 × 900 方向 × 463 频点，共 `10,000,800` 个样本。MCA zero-residual 的 MAE/RMSE 为 `2.600671/4.186115 dB`，v2 为 `2.168373/3.552562 dB`，v3 为 `2.091041/3.452171 dB`。v3 相对 MCA 的 MAE 改善 `19.60%`，相对 v2 的 MAE 改善 `3.57%`；12/12 test 被试的 v3 MAE 均低于 v2，逐被试改善范围为 `1.42%–4.79%`。内置 v2 与历史 test 结果的 MAE/RMSE 差值仅 `-7.68e-9/-9.18e-8 dB`，一致性检查通过。
- 完整频谱 GPU 推理：新增 `residual_learning/mlp_cnn_v3/python/predict_mlp_cnn_reconstruction.py`。复用 v1 已缓存且与 v2 共用的 900 个 Fliege dense 方向 + 360 个水平面方向，按 32 个方向分块，但双耳和完整 463 点频率轴始终联合输入 CNN。RTX 5060 上 12 被试总耗时 `6.01 s`，峰值 CUDA allocated memory `34.81 MiB`，全部预测 HDF5 完整写入。
- 严格重建实现：扩展 `residual_learning/matlab/evaluate_test_reconstruction.m`，在不改变 v1/v2 默认调用方式的前提下支持显式输出根目录、缓存根目录、方法键和图例名称；新增 v3 入口 `residual_learning/mlp_cnn_v3/matlab/evaluate_mlp_cnn_v3_reconstruction.m` 与 v1/v2/v3 对比图脚本。预测 residual 逐频点加到 MCA log-magnitude，复数相位完全沿用 MCA。
- 严格指标结果：MCA 的全球 ERB、对侧 25° ERB、对侧 `>10 kHz` magnitude error、水平面 ILD MAE 分别为 `0.802741/1.860286/4.258336/0.885425 dB`；v3 分别为 `0.584593/1.304415/3.620148/0.661566 dB`，相对 MCA 改善 `27.18%/29.88%/14.99%/25.28%`，四项均为 12/12 被试改善。
- v3 对 v2：v2 四项严格指标为 `0.611503/1.340068/3.720378/0.646722 dB`。v3 在全球 ERB、对侧 25° ERB和对侧高频上进一步降低 `4.40%/2.66%/2.69%`；ILD MAE 则由 `0.646722` 升至 `0.661566 dB`，相对退化 `2.30%`。因此 v3 的 CNN 频率上下文对三项幅度指标形成稳定增益，但当前训练中的 ILD proxy 未保证严格 HRIR 能量 ILD 相对 v2 单调改善，此结论必须保留，不能只汇报相对 MCA 的正向数字。
- 重建质量检查：12 被试最大左/右相位误差分别为 `6.00e-16/6.22e-16 rad`，最大幅度回填恒等误差均为 `7.11e-15 dB`，远低于 `1e-5 rad/1e-4 dB` 断言阈值。12 张单被试图、`test12_contralateral_hrtf_overview.png`、`test12_metric_overview.png` 与 `v1_v2_v3_aggregate_comparison.png` 均已生成并完成视觉检查。
- 实际命令：raw test 使用 `D:\miniconda3\envs\ml\python.exe residual_learning/mlp_cnn_v3/python/evaluate_mlp_cnn_v3.py residual_learning/data/hutubs_residual_v1_n03 residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt --split test --allow-test --directions-per-block 32`；预测使用 `D:\miniconda3\envs\ml\python.exe residual_learning/mlp_cnn_v3/python/predict_mlp_cnn_reconstruction.py residual_learning/mlp_cnn_v3/reconstruction/mlp_cnn_n03_v3_cnn_only residual_learning/reconstruction/mlp_n03_v1 residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt residual_learning/data/hutubs_residual_v1_n03/training_statistics.json --directions-per-block 32`；MATLAB 使用 `matlab.exe -batch "addpath('D:/cuc/CSMT/MCAR/residual_learning/mlp_cnn_v3/matlab'); evaluate_mlp_cnn_v3_reconstruction; plot_v1_v2_v3_reconstruction_comparison"`。
- 输出与 Git：本地结果位于 `residual_learning/mlp_cnn_v3/reconstruction/mlp_cnn_n03_v3_cnn_only/`，包括预测 HDF5、CSV、MAT、JSON 和 15 张 PNG；raw test JSON/CSV 位于正式 run 目录。两类生成物均由 v3 的 `.gitignore` 忽略。适合提交的是 Python/MATLAB 源码、README、依赖说明、占位 `.gitignore` 和本实验日志；本次未提交或推送。
- 运行异常：MATLAB 在受限沙箱内首次启动报 `File system inconsistency`，在已授权的正常本机权限下启动成功。命令接口 60 秒超时后 MATLAB 子进程继续完成全部评估；没有重复启动、没有丢失或覆盖结果。Codex 服务层中途出现一次 503，不影响本地 Python/MATLAB 进程及产物。
- 结论与下一步：v3 的核心假设得到部分验证——频谱 CNN 在未见 test 上可稳定降低 raw residual 和三项严格幅度误差，但 ILD 相对 v2 小幅退化。下一步不应直接增加模型规模；优先做一个受控 v3.1 实验，在保持当前架构和 test 锁定的前提下，仅在 train/validation 上校准严格 ILD 对齐的可微损失或提高双耳能量约束，随后以 validation 选择 checkpoint，再进行一次最终 test。也可先整理 v3 详细报告和可提交的轻量结果清单。

## 2026-07-25：MLP + CNN v3 完整 validation residual 评估

- 实验目标：在不访问 test 的前提下，对 v3 epoch 9 best checkpoint 执行无随机采样的完整 validation 遍历，计算全部 raw residual MAE/RMSE；同时从同一个 checkpoint 得到 MCA zero-residual、冻结 v2 MLP 和最终 v3 三组结果，验证 CNN 的增益及逐被试稳定性。
- 评估器：新增 `residual_learning/mlp_cnn_v3/python/evaluate_mlp_cnn_v3.py`。由于 CNN 必须观察完整频谱，评估按 32 个方向分块，但每块保留双耳和全部 463 个频点；每个 HDF5 的 900 个方向全部遍历。`--split test` 必须额外显式提供 `--allow-test`，避免模型选型阶段意外读取 test。
- 数据范围：固定 validation 被试 pp13、pp35、pp39、pp41、pp51、pp53、pp55、pp56、pp58、pp74、pp84、pp92；12 个文件 × 2 耳 × 900 方向 × 463 频点，共 `10,000,800` 个样本，实际计数完全一致。
- 汇总结果：MCA zero-residual 的 MAE/RMSE 为 `2.571911/4.143771 dB`；v2 MLP 为 `2.139782/3.506980 dB`；v3 MLP + CNN 为 `2.072657/3.428915 dB`。v3 相对 MCA 的 MAE 改善为 `19.41%`，相对 v2 的 MAE/RMSE 分别降低 `3.14%/2.23%`；全样本 CNN delta 平均绝对值为 `0.5687 dB`。
- v2 口径交叉检查：评估器内置 v2 MLP 的 MAE/RMSE 与原 `residual_learning/runs/mlp_n03_v2/val_metrics.json` 的差值分别为 `+1.10e-8/-1.83e-8 dB`，远低于 `1e-4 dB` 阈值，证明新评估器的数据遍历、归一化、AMP 和反归一化口径与原 v2 完整评估一致。
- 逐被试结果：12/12 validation 被试的 v3 MAE 均低于 v2。相对改善依次为 pp13 `1.80%`、pp35 `3.51%`、pp39 `1.58%`、pp41 `3.19%`、pp51 `3.57%`、pp53 `4.44%`、pp55 `2.65%`、pp56 `4.12%`、pp58 `1.74%`、pp74 `2.36%`、pp84 `4.24%`、pp92 `4.28%`；范围 `1.58%–4.44%`，逐被试改善百分比的非加权平均为 `3.12%`，没有退化案例。
- 资源与完整性：完整评估耗时 `5.15 s`，峰值 CUDA allocated memory `35.20 MiB`；checkpoint epoch 为 9、训练阶段标记 `cnn_only_frozen_mlp`、模型参数 `174,627`。Python 编译、样本计数、逐被试/总计合并和 v2 历史交叉检查均通过。
- 实际命令：`D:\miniconda3\envs\ml\python.exe residual_learning/mlp_cnn_v3/python/evaluate_mlp_cnn_v3.py residual_learning/data/hutubs_residual_v1_n03 residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt --split val --directions-per-block 32`，退出码为 0。
- 输出与 Git：本地生成 `val_full_metrics.json` 和 `val_per_subject_metrics.csv`，位于正式 v3 run 目录并由 `runs/.gitignore` 忽略；关键结果已记录在本日志与 v3 README，未读取 test，未提交或推送。
- 结论与下一步：v3 在完整未见 validation 的 raw residual MAE、RMSE 和全部 12 个被试上稳定优于 v2，满足进入固定 test 严格评估的前置条件。下一步锁定 epoch 9 checkpoint，在 12 个 test 被试上先做完整 raw residual 评估，再生成 v3 双耳完整频谱预测、回填 MCA 幅度并使用 MATLAB 计算严格 ERB、对侧高频与 ILD；模型和超参数不再根据 test 调整。

## 2026-07-25：MLP + CNN v3 CNN-only 跨被试在线完整训练

- 实验目标：从原始 v2 epoch 9 checkpoint 重新开始，在 72 train / 12 validation 被试上完整训练双耳频谱 CNN + FiLM；v2 MLP 全程冻结，test 集不参与训练和选型。通过 W&B 在线监控曲线，并以固定 validation 复合损失选择最佳 checkpoint。
- W&B 运行：project 为 `mcar-mlp-cnn-v3`，run name 为 `mlp_cnn_n03_v3_cnn_only`，run id 为 `qtkrxras`，URL 为 `https://wandb.ai/luyoung/mcar-mlp-cnn-v3/runs/qtkrxras`。第一次从受限进程启动时被系统禁止访问 `api.wandb.ai:443`，进程只停留在 W&B 握手阶段，未写 configuration/history/checkpoint；停止后在新的本地 run 目录以允许网络访问的进程重新启动，在线 run 正常创建并最终以 exit code 0 完成同步。
- 数据与采样：HUTUBS simulated N=3 residual HDF5；72 个 train 和 12 个 validation 被试；每个 batch 随机选择一个被试、32 个 Fliege 方向、双耳和全部 463 个频点，共 `29,632` 个逐频点样本。validation 每个 epoch 均以 seed `20260726` 重建相同的 96 个 batch，确保 epoch 0 的 v2 基线与所有 v3 epoch 可直接比较。
- 模型与训练参数：总参数 `174,627`，其中冻结 v2 MLP `100,225`，可训练 CNN + FiLM `74,402`；10 epoch × 500 steps；AdamW 学习率 `3e-4`、weight decay `1e-5`、cosine decay、gradient clip `5.0`、FP16 AMP 初始 scale `1024`。损失保持 residual SmoothL1 + `0.50 × ERB proxy + 0.25 × 对侧高频 + 0.25 × ILD proxy` 的 v2 权重。
- 固定 validation 基线：训练开始前的 v2 total/residual/ERB/高频/ILD 为 `0.594113 / 2.139116 / 0.695902 / 3.297190 / 0.646540 dB`，CNN delta 为 0。
- 收敛过程：validation total 从 epoch 1–10 依次为 `0.588567、0.585525、0.581558、0.579968、0.579476、0.575839、0.574931、0.573803、0.573431、0.573586`。前 9 个 epoch 总体持续下降，epoch 10 略回升，因此根据预定规则选择 epoch 9。
- 最佳 epoch 9：validation total/residual/ERB/高频/ILD 为 `0.573431 / 2.073348 / 0.664612 / 3.216917 / 0.641232 dB`，相对初始 v2 分别降低 `3.48% / 3.07% / 4.50% / 2.43% / 0.82%`，五项均改善；validation CNN delta 平均绝对值为 `0.5657 dB`。训练侧相应 total/residual/ERB/高频/ILD 为 `0.537791 / 1.977483 / 0.645931 / 2.986437 / 0.617659 dB`。
- epoch 10 取舍：epoch 10 的 residual/ERB 进一步变为 `2.073211/0.663860 dB`，但高频和 ILD 回升到 `3.218165/0.641742 dB`，使 total 为 `0.573586`，未超过 epoch 9；因此不按单一 residual 或 ERB 指标改选 checkpoint。
- 稳定性与资源：训练总耗时 `364.19 s`（约 6 分 4 秒），峰值 CUDA allocated memory `220.46 MiB`；5000 个优化步均无 AMP 跳步，最终 AMP scale 增长到 `4096`。在线曲线与本地 `history.csv` 一致；正常生成 `training_report.json`。
- checkpoint 验证：`best.pt` 为 epoch 9，`last.pt` 为 epoch 10；两者均可由 PyTorch 成功重新读取。best checkpoint 的训练阶段标记为 `cnn_only_frozen_mlp`，包含 82 个 state tensors。
- 输出与 Git：本地结果位于 `residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online`；configuration、initial validation、history、best/last checkpoint、training report 和 W&B 本地缓存均由 v3 `runs/.gitignore` 忽略。W&B 仅同步标量曲线与 summary，没有上传 checkpoint 或数据集。
- 结论与下一步：冻结 MLP 的频谱 CNN 在严格未见 validation 被试上同时改善五项训练代理指标，证明收益不是 pp91 单被试记忆。增益较 pp91 过拟合明显收缩，尤其 ILD 仅改善 `0.82%`，说明跨被试泛化仍是限制。下一步先为 v3 实现完整 validation 遍历评估，报告全部 `10,000,800` 样本的 raw residual MAE/RMSE；通过后再锁定 epoch 9，在 test 上执行 residual 回填、MATLAB `AKerbError`、对侧高频与 ILD 严格评估。

## 2026-07-25：MLP + CNN v3 训练器接入 Weights & Biases

- 实验目标：为完整 CNN-only 跨被试训练加入在线曲线监控，同时保持本地 CSV、checkpoint 和 JSON 报告为复现主记录；W&B 不自动上传模型权重或数据集。
- 中断运行状态：原无 W&B 完整训练由用户主动停止。进程停止前已完整保存 epoch 1–6，`best.pt` 与 `last.pt` 均为可读取的 epoch 6，固定 validation total loss 为 `0.57584`，6 个 epoch 均无 AMP 跳步；因未完成全部 10 epoch，不存在只在正常结束时生成的 `training_report.json`。该运行保留在本地忽略目录，不需要清理，也不与新 W&B run 拼接。
- 训练器修改：`train_mlp_cnn_v3.py` 新增 `--wandb-mode disabled|online|offline`、project、entity、run name、group、job type、tags、notes 和可选 `--wandb-watch`。默认 mode 为 disabled，保证原有本地命令不产生联网行为；正式监控时显式传入 `--wandb-mode online`。默认 project 为 `mcar-mlp-cnn-v3`。
- 记录指标：W&B 以 epoch 为公共横轴，记录 train/validation 的复合损失、residual MAE、ERB proxy MAE、对侧高频 MAE、ILD proxy MAE 和 CNN delta 平均绝对值；同时记录 validation 相对初始 v2 的五项改善百分比、learning rate、AMP scale、当轮/累计跳步数、epoch 耗时、峰值 CUDA allocated memory、当前 epoch 是否为最佳以及当前最佳 epoch。
- 运行摘要：正常结束后将最佳 epoch、最佳 validation 五项指标、总耗时、峰值显存和 AMP 跳步总数写入 W&B summary；相同信息仍保存在本地 `training_report.json`。`best.pt`、`last.pt`、HDF5 和其他大型产物不上传 W&B。
- 本地目录：为避免受限环境下 W&B core 写入用户 AppData 失败，训练器在 import W&B 前将 `WANDB_DIR`、`WANDB_DATA_DIR`、`WANDB_CACHE_DIR`、`WANDB_CONFIG_DIR` 和 `WANDB_ARTIFACT_DIR` 全部指向当前 `runs/<run-name>` 下的忽略目录。首次 offline 测试虽成功但出现 AppData debug-log 权限错误；重定向后的第二次测试不再出现该错误。
- 依赖：新增 `residual_learning/mlp_cnn_v3/requirements.txt`，复用公共 residual-learning 依赖并要求 `wandb>=0.21,<1`。本机 `ml` 环境已安装并验证 W&B `0.21.1`。
- 离线集成测试：使用 pp91、4 方向、1 epoch × 2 steps、2 个 validation batch 和 `--wandb-mode offline` 实际运行，退出码为 0。W&B run id 为 `2b1v9fvt`；初始 validation、epoch 1 的全部 train/validation 曲线、改善百分比、optimizer/system/checkpoint 指标以及最终 summary 均成功写入。checkpoint、本地 JSON 和 W&B offline run 同时生成，AMP 跳步为 0。
- Git 约定：W&B 的 `wandb/`、`wandb_state/`、offline run、checkpoint 和测试输出均位于 v3 `runs/` 下，由嵌套 `.gitignore` 忽略；只保留训练器、v3 README、requirements 和本实验日志。
- 正式运行建议：从原 v2 epoch 9 checkpoint 重新开始，不复用被中断 run name；本地 run 使用 `mlp_cnn_n03_v3_cnn_only_wandb`，W&B 显示名使用 `mlp_cnn_n03_v3_cnn_only`，保持 72/12 划分、10 epoch × 500 steps、96 个固定 validation batch、32 方向和现有复合损失权重。在线开始前只需确保本机已执行 `python -m wandb login`，API key 不写入项目或命令。

## 2026-07-25：MLP + CNN v3 的 pp91 CNN-only 过拟合检查

- 实验目标：验证新增双耳频谱 CNN 在冻结 v2 MLP 时，能否从真实 HUTUBS 完整频谱中学习有效的 delta residual，并同时降低 residual、ERB、高频和 ILD 训练代理指标；本实验只作为单被试管线检查，不作为跨被试结论。
- 训练入口：新增 `residual_learning/mlp_cnn_v3/python/train_mlp_cnn_v3.py`。训练器加载 v2 epoch 9 checkpoint，冻结全部 `100,225` 个 MLP 参数，只优化 `74,402` 个 CNN + FiLM 参数。validation sampler 每次以相同 seed 重新创建，因此初始 v2 与每个 epoch 使用完全相同的固定 validation batch 序列。
- 数据与采样：HUTUBS simulated pp91、Lebedev N=3 residual HDF5；每个 batch 采样 16 个 Fliege 方向、双耳和全部 463 个频点，共 `14,816` 个逐频点样本。train 和 validation 均使用 pp91，但采样序列分别使用 seed `20260725` 和 `20260726`。
- 损失与优化：保持 v2 复合损失不变，即 residual SmoothL1 + `0.50 × ERB proxy/target_std + 0.25 × 对侧高频/target_std + 0.25 × ILD proxy/target_std`；AdamW 学习率 `3e-4`、weight decay `1e-5`、cosine decay、gradient clip `5.0`、FP16 AMP。训练为 4 epoch × 120 steps，每 epoch 使用 48 个固定 validation batch。
- AMP 修正：第一次运行使用 PyTorch 默认初始 loss scale `65536`，零初始化 CNN 输出头的首步放大梯度出现非有限值，训练器按预期中止。烟雾测试的未缩放梯度有限，说明问题来自初始缩放而非数据或损失。训练器现将 `amp_initial_scale` 默认设为 `1024`，并在 unscale 后检测梯度；AMP 溢出步会由 GradScaler 安全跳过并计数。本次完整检查 480 个优化步均未跳过，最终 scale 为 `1024`。
- 固定验证结果：初始 v2 的复合损失/residual/ERB/高频/ILD 为 `0.53335 / 1.9304 / 0.6441 / 3.0559 / 0.5875 dB`；最佳 epoch 4 为 `0.40730 / 1.6244 / 0.5436 / 2.3722 / 0.3170 dB`。相对初始 v2 分别降低 `23.63% / 15.85% / 15.60% / 22.37% / 46.05%`。epoch 4 的 CNN delta 平均绝对值为 `0.8512 dB`。
- 收敛过程：validation total loss 从 epoch 1 到 4 依次为 `0.49027、0.44813、0.41958、0.40730`，连续下降；对应高频误差为 `2.817、2.593、2.429、2.372 dB`，ILD 为 `0.457、0.391、0.359、0.317 dB`。最佳点仍位于最后一个 epoch，说明检查没有出现提前退化。
- 资源与耗时：4 epoch 总训练耗时 `35.74 s`，单 epoch 约 `8.7–9.1 s`；峰值 CUDA allocated memory 为 `119.82 MiB`。本机 RTX 5060 8 GB 有充分余量，完整训练可优先尝试 32 方向 batch。
- 实际命令：`D:\miniconda3\envs\ml\python.exe residual_learning/mlp_cnn_v3/python/train_mlp_cnn_v3.py residual_learning/data/hutubs_residual_v1_n03 residual_learning/runs/mlp_n03_v2/best.pt --run-name overfit_pp91_cnn_only --overfit-subject 91 --epochs 4 --steps-per-epoch 120 --validation-steps 48 --directions-per-batch 16`，退出码为 0。
- 输出与 Git：本地结果位于 `residual_learning/mlp_cnn_v3/runs/overfit_pp91_cnn_only`，包含 configuration、initial validation、history、best/last checkpoint 和 training report；该单被试生成目录由 v3 嵌套 `.gitignore` 忽略。训练源码和 README 保留为待提交文件。
- 结论与下一步：CNN-only 分支已经通过单被试可学习性、指标方向、checkpoint、固定验证和 AMP 稳定性检查。下一步在原 72 train / 12 validation 被试上进行完整 CNN-only 训练，保持 v2 MLP 和损失权重不变；根据固定 validation 复合损失选最佳 epoch，再做完整 validation residual 评估。test 和 MATLAB 严格指标在模型确定前继续锁定。

## 2026-07-25：MLP + CNN v3 独立工作流与真实数据梯度检查

- 实验目标：在 v2 指标感知 MLP 的基础上加入频率上下文建模，同时保持已有 subject-wise 划分、HDF5 数据、复合损失和最终 MATLAB 指标口径不变。第一步只实现推荐的网络结构并验证真实数据前向、v2 无损初始化、复合损失和 CNN 反向传播，不开始完整训练。
- 工作流目录：新增 `residual_learning/mlp_cnn_v3`，内部独立保存 v3 的 Python 模型与检查脚本，以及后续 `runs/` 和 `reconstruction/`。源码和说明可提交；checkpoint、训练缓存、预测 HDF5 和 MAT 等生成产物由嵌套 `.gitignore` 默认忽略。
- 模型结构：保留 v2 的 7 输入、宽度 128、3 个 residual block 的 `ResidualMLP` 作为逐频点基础预测器；新增双耳 `BinauralSpectralCNN`，输入为左右耳归一化 MCA magnitude、correction magnitude、v2 residual 以及归一化 log-frequency，共 7 个频谱通道。CNN 使用 48 通道 stem、4 个 kernel 7 的深度可分离 residual block，dilation 为 `1/2/4/8`，理论感受野为 91 个频点；方向 `x/y/z` 经 64 维 MLP 生成逐块 FiLM 调制。CNN 输出左右耳两个 delta residual 通道，并与 v2 MLP 输出相加。
- 初始化与复杂度：CNN 输出层和 FiLM 线性层零初始化，确保加载 v2 epoch 9 checkpoint 后，训练起点严格满足 `v3 = v2 + 0`。v2 MLP 为 `100,225` 参数，CNN + FiLM 为 `74,402` 参数，总计 `174,627` 参数；冻结 MLP 的第一阶段仅训练 `74,402` 参数。
- 环境检查：`D:\miniconda3\envs\ml` 中 PyTorch `2.8.0+cu128`、CUDA build `12.8` 和 h5py `3.14.0` 可用，GPU 为 NVIDIA GeForce RTX 5060。v2 `best.pt`、训练统计量和 pp91 N=3 HDF5 均存在并可读取。
- 烟雾测试：`smoke_test_mlp_cnn.py` 使用真实 pp91 双耳完整 463 点频谱和 v2 复合损失运行两个优化步。4 方向与 16 方向测试均通过；输入/输出分别为 `[direction, 2, 463, 7]` 和 `[direction, 2, 463]`。零初始化时 v3 与 v2 最大逐点差值为 `0`；第一个优化步只有零初始化输出头的 2 个参数张量获得非零梯度，完成一次更新后第二步已有 62 个 CNN 参数张量获得非零梯度，所有已计算梯度和损失均为有限值，冻结 MLP 未产生梯度。
- 资源检查：16 方向、FP16 AMP、完整 v2 复合损失的峰值 CUDA allocated memory 为 `197.12 MiB`，远低于本机约 8 GB 显存；正式 CNN-only 训练可以从 16 或 32 方向起步。
- 实际命令：`D:\miniconda3\envs\ml\python.exe residual_learning/mlp_cnn_v3/python/smoke_test_mlp_cnn.py --dataset-root residual_learning/data/hutubs_residual_v1_n03 --checkpoint residual_learning/runs/mlp_n03_v2/best.pt --subject 91 --directions 16`，退出码为 0。
- 输出文件位置：模型为 `residual_learning/mlp_cnn_v3/python/mlp_cnn_model.py`，真实数据检查为 `residual_learning/mlp_cnn_v3/python/smoke_test_mlp_cnn.py`，结构、运行命令和目录约定见 `residual_learning/mlp_cnn_v3/README.md`。
- 结论与下一步：v3 第一阶段模型链路已通过，当前不需要补充数据或环境。下一步实现 CNN-only 训练入口，冻结 v2 MLP，保持 v2 的 `0.50/0.25/0.25` ERB/对侧高频/ILD 权重不变；先在 pp91 做短程过拟合检查，再决定正式训练的方向 batch size 与 epoch 数。

## 2026-07-24：Residual MLP v2 双耳指标感知训练与回填评估

- 实验目标：在 v1 已证明逐频点 residual 可学习的基础上，加入双耳完整频谱与听觉指标感知损失，重点扩大 ERB 改善并消除 pp33、pp81 的 ILD 退化；最终仍在原 12 个严格未见 test 被试上，用 MATLAB `AKerbError`、对侧高频和水平面 ILD 的相同口径评估。
- 模型与初始化：网络仍为 7 输入、宽度 128、3 个 SiLU residual block、单输出的 `ResidualMLP`，可训练参数保持 `100,225`；从 v1 epoch 10 的 `best.pt` 初始化，随后对全部参数继续训练。这样可将 v1 作为稳定起点，并把优化重点转向听觉指标，而不是从零重新学习。
- 双耳采样：新增 `BinauralSpectrumSampler`。每个 batch 随机选择一个 train 被试和 32 个 Fliege 方向，同时读取左右耳与全部 463 个频点，张量布局为 `[ear=2, direction=32, frequency=463]`，共 `29,632` 个逐频点样本。该布局保证同方向双耳能量和完整导出频谱可同时参与损失。
- 复合损失：总损失为归一化 residual SmoothL1，加 `0.50 × ERB代理MAE/target_std`、`0.25 × 对侧高频MAE/target_std`、`0.25 × ILD代理MAE/target_std`。ERB 代理使用 50 Hz–20 kHz、41 个 ERB-rate 三角带，对修正后与 reference 的频带能量 dB 做误差；对侧高频项按左右耳相反半球选择 `>10 kHz` 频点；ILD 代理由双耳频谱能量比计算。代理项只用于可微训练，最终结果仍由 MATLAB 原始 `AKerbError` 和 HRIR 能量 ILD 计算。
- 数值与过拟合检查：真实 pp91 batch 的四项损失均有限，反向梯度全部有限；16 方向检查的 CUDA 峰值分配约 `134.4 MiB`。pp91 短程检查为 4 epoch × 120 steps，validation 代理 ERB 从 epoch 1 的 `0.603` 降至 `0.561 dB`，对侧高频从 `2.538` 降至 `2.331 dB`，ILD 从 `0.411` 降至 `0.358 dB`，证明双耳采样与复合梯度链路有效。
- 完整训练：72 个 train、12 个 validation 被试；AdamW 学习率 `3e-4`、weight decay `1e-5`、cosine decay、FP16 AMP；10 epoch × 500 steps，每 epoch 96 个 validation batch，总耗时 `331.3 s`（约 5 分 31 秒）。固定 validation batch 上的 v1 初始代理指标为 residual/ERB/高频/ILD `2.164/0.773/3.301/0.739 dB`；根据复合 validation loss 选择 epoch 9，得到 `2.117/0.693/3.267/0.609 dB`。
- 完整逐频点指标：v2 在 validation 的 MAE 为 `2.1398 dB`，相对 MCA `2.5719 dB` 改善 `16.80%`；在 test 的 MAE 为 `2.1684 dB`，相对 MCA `2.6007 dB` 改善 `16.62%`，RMSE 为 `3.5526 dB`。v1 的 test MAE 为 `2.2172 dB`、改善 `14.75%`，因此 v2 没有以牺牲原始 residual 精度换取听觉指标。
- 严格论文指标：全空间 ERB error 从 MCA `0.8027 ± 0.0255 dB` 降至 v2 `0.6115 ± 0.0356 dB`，改善 `23.82%`；对侧 25° ERB 从 `1.8603 ± 0.1159 dB` 降至 `1.3401 ± 0.0999 dB`，改善 `27.96%`；对侧半球 `>10 kHz` error 从 `4.2583 ± 0.2641 dB` 降至 `3.7204 ± 0.2436 dB`，改善 `12.63%`；ILD MAE 从 `0.8854 ± 0.2092 dB` 降至 `0.6467 ± 0.2662 dB`，改善 `26.96%`。四项均为 12/12 test 被试改善。
- v1→v2 增益：相对 v1，v2 进一步降低全空间 ERB `10.57%`、对侧 25° ERB `14.00%`、对侧高频 `0.94%`、ILD `16.31%`。pp33 的 ILD 从 v1 的退化 `-5.57%` 变为相对 MCA 改善 `14.60%`；pp81 从退化 `-18.97%` 变为改善 `0.52%`，说明双耳 ILD 损失解决了主要失败案例，但 pp81 仍是最弱改善被试。
- 回填与可视化：复用 v1 的 12 份 complex MCA/reference cache，使用 v2 epoch 9 checkpoint 预测 1260 方向 residual，并保持 MCA 相位回填。最大相位误差 `6.29e-16 rad`、幅度恒等误差 `7.11e-15 dB`。输出 12 张逐被试四面板图、12 人对侧 HRTF 总览、逐被试指标总览和 v1/v2 直接对比图。
- 输出与 Git：训练源码为 `train_residual_mlp_v2.py` 与扩展后的 `residual_data.py`；评估复用 `predict_reconstructed_residuals.py` 和 `evaluate_test_reconstruction.m`，新增 `plot_v1_v2_reconstruction_comparison.m`。训练结果位于 `residual_learning/runs/mlp_n03_v2`，回填结果位于 `residual_learning/reconstruction/mlp_n03_v2`。Git 仅保留训练 history/report、val/test JSON、最终 CSV 和 PNG；checkpoint、本机配置、预测 HDF5、MAT、日志及 `overfit_pp91_v2` 继续忽略。
- 结论与下一步：v2 的指标感知训练在不增加模型参数的情况下显著扩大 ERB 与 ILD 改善，并让所有 test 被试的四项严格指标都优于 MCA。下一步应进行损失消融（residual+ERB、+高频、+ILD）以量化每个损失项贡献，并将 v2 从固定 N=3 扩展到其他稀疏阶数，优先 N=1、2、4、6。

## 2026-07-24：Residual MLP v1 幅度回填与论文指标评估

- 实验目标：在严格未见的 12 个 test 被试上，将 MLP 预测的 `reference_logmag_db - mca_logmag_db` 回填到 N=3 MCA 幅度，同时逐频点保留原 MCA 相位；计算与传统 MCA 复现一致的 ERB magnitude error、对侧高频误差和水平面 ILD error，并生成 12 个被试的直观 HRTF 对比图。
- 数据集与被试：HUTUBS simulated HRTF；test 被试为 pp8、pp18、pp22、pp26、pp31、pp33、pp45、pp47、pp59、pp70、pp73、pp81。稀疏输入为 Lebedev N=3（26 点）；评估方向为 Fliege N=29（900 点）和水平面 0–359°（360 点）。pp18 缺少公开人体测量值，沿用此前规则，以 93 名有效被试的平均 Algazi 半径 `0.091021 m` 代替。
- 重建方法：MATLAB 重新生成每个被试的复数 MCA/reference HRTF 和 correction filter；Python/PyTorch 使用 epoch 10 的 `best.pt`、训练集归一化参数、RTX 5060 FP16 AMP，对 1260 个方向和 463 个频点（实际 `86.13–19982.81 Hz`）预测 residual；回到 MATLAB 后执行 `corrected_logmag = mca_logmag + predicted_residual`，并使用 `exp(j*angle(H_MCA))` 恢复复数 HRTF。20 kHz 以上保持原 MCA 不变。
- 指标口径：ERB 指标调用与传统基线相同的 `AKerbError`，范围 50 Hz–Nyquist，方向按 Fliege 权重、ERB band 等权；同时报告全空间与每耳对侧 25° 区域。对侧高频指标为每耳相反开放半球、`f > 10 kHz` 至 Nyquist 的绝对 log-magnitude error，方向按 Fliege 权重、频率等权。ILD 为 360 个水平面方向上左右 HRIR 全带能量比 `10*log10(E_L/E_R)` 相对 reference 的平均绝对误差。
- 汇总结果：全空间 ERB magnitude error 从 MCA `0.8027 ± 0.0255 dB` 降至 `0.6838 ± 0.0326 dB`，平均降低 `0.1189 dB / 14.82%`，12/12 被试改善；对侧 25° ERB error 从 `1.8603 ± 0.1159 dB` 降至 `1.5582 ± 0.0971 dB`，降低 `0.3020 dB / 16.24%`，12/12 改善；对侧半球高频误差从 `4.2583 ± 0.2641 dB` 降至 `3.7557 ± 0.2420 dB`，降低 `0.5026 dB / 11.80%`，12/12 改善；ILD MAE 从 `0.8854 ± 0.2092 dB` 降至 `0.7727 ± 0.3001 dB`，降低 `0.1127 dB / 12.73%`，10/12 改善。
- 重建质量检查：12 个被试的最大相位保持误差为 `6.22e-16 rad`，最大幅度回填恒等误差为 `7.11e-15 dB`；证明输出确实只改变指定频点的幅度，没有改变 MCA 相位。MATLAB Code Analyzer 对两个新增脚本均报告 0 问题，Python 文件可成功编译；准备阶段 12/12、GPU 推理 12/12、最终评估 12/12 均完成。
- 可视化：每个 test 被试各有一张四面板图，包括左右耳对侧 HRTF（Reference/MCA/MCA+Residual MLP）、全空间 ERB error 频率曲线和水平面 ILD；另有一张 4×3 的 12 人左耳对侧 HRTF 总览与一张逐被试指标总览。
- 输出与 Git：源码为 `prepare_test_reconstruction_inputs.m`、`predict_reconstructed_residuals.py`、`evaluate_test_reconstruction.m`；结果位于 `residual_learning/reconstruction/mlp_n03_v1`。Git 仅保留最终 CSV 与 PNG；复数 cache、模型输入、预测 HDF5、MAT、运行日志和含本机路径的推理报告继续忽略。
- 结论与下一步：首版 100,225 参数 residual MLP 在严格未见被试上不仅降低逐频点 residual MAE，也稳定降低论文口径的 ERB 与对侧高频误差，并整体改善 ILD。下一步应分析 pp33 与 pp81 的 ILD 退化原因，并考虑在训练中加入 ERB/ILD 感知损失或双耳联合特征，而不修改相位。

## 2026-07-23：Residual MLP v1（HUTUBS N=3，跨被试）

- 实验目标：在已经导出的 HUTUBS N=3 residual 数据集上，实现并验证首版轻量 MLP，确认模型能在严格的 subject-wise 未见被试上降低 `reference_logmag_db - mca_logmag_db`。
- 训练环境：Conda 环境 `D:\miniconda3\envs\ml`，Python `3.9.23`，PyTorch `2.8.0+cu128`，CUDA build `12.8`，h5py `3.14.0`。GPU 为 RTX 5060（8151 MiB，compute capability 12.0）；已实际验证 HDF5 读取、CUDA 张量计算和混合精度反向传播。
- 模型与输入：`ResidualMLP` 有 `100,225` 个可训练参数；7 个输入特征依次为归一化 MCA log-magnitude、归一化 MCA correction-filter log-magnitude、方向单位向量 `x/y/z`、归一化 `log10(frequency)` 和耳别（左=-1，右=1）。网络为宽度 128 的输入层、3 个 SiLU 残差块和单输出层，预测归一化的 dB residual。
- 采样与优化：为避免将 1.10 GiB 数据整体载入内存，每个 batch 从随机被试和随机耳朵读取 `64` 个方向与 `128` 个频点的笛卡尔块，共 `8192` 样本。归一化参数严格来自 72 个 train 被试；优化器 AdamW（学习率 `1e-3`、weight decay `1e-5`）、SmoothL1 损失、cosine learning-rate decay、CUDA FP16 AMP。训练共 12 epoch × 600 steps（随机有放回采样），每 epoch 以 96 个 validation block 监控。
- 过拟合检查：pp91 单被试运行 8 epoch × 300 steps 后，随机留出 block 的 MAE 从 MCA zero-residual 基线约 `2.34 dB` 降至约 `1.60 dB`，数据读取、梯度和 checkpoint 链路均通过。
- 训练过程：跨被试训练命令为 `train_residual_mlp.py ... --run-name mlp_n03_v1 --epochs 12 --steps-per-epoch 600 --validation-steps 96`；总耗时 `380.2 s`（约 6 分 20 秒）。根据 validation MAE 选择 epoch 10 的 best checkpoint。
- 独立指标结果：在完整 validation（12 被试、`10,000,800` 样本）上，逐频点 residual MAE 从 MCA 基线 `2.5719 dB` 降至 `2.1886 dB`（`14.90%`），RMSE 从 `4.1438` 降至 `3.5346 dB`。在严格未见的 test（12 被试、`10,000,800` 样本）上，MAE 从 `2.6007` 降至 `2.2172 dB`（`14.75%`），RMSE 从 `4.1861` 降至 `3.5792 dB`。
- 输出文件位置：训练代码为 `residual_learning/python/residual_data.py`、`residual_model.py`、`train_residual_mlp.py`、`evaluate_residual_mlp.py`；`residual_learning/runs/mlp_n03_v1` 中仅保留 `history.csv`、`val_metrics.json` 和 `test_metrics.json`，checkpoint、含本机绝对路径的配置与其他运行产物由嵌套 `.gitignore` 忽略。
- 结论与限制：首版轻量 MLP 已在未见被试上稳定降低原始频谱 residual，证明 MCA 后残差包含可跨被试学习的规律。当前结果不是论文 ERB magnitude error、ILD 或 ITD；模型只预测幅度 residual，尚未将预测值回填为校正 HRTF 并重算这些最终听觉指标。
- 下一步：实现“预测 residual → 修正 MCA 幅度 → 结合原 MCA 相位”的重建与评估器，首先在完整 test 集报告 auditory-band magnitude error、对侧高频误差和 ILD；ITD 预计保持 MCA 水平，因为本阶段不修改相位。

## 2026-07-23：Residual learning 数据集 v1（HUTUBS N=3）

- 实验目标：建立独立的 `residual_learning/` 阶段目录，固定无被试泄漏的数据划分，并为轻量网络导出 `target = reference_logmag_db - mca_logmag_db` 的首版训练数据。
- 数据集与划分：96 个 HUTUBS simulated HRTF；使用 MATLAB `rng(20260723, 'twister'); randperm(96)` 固定 subject-wise 划分为 train/validation/test = `72/12/12`。同一被试只属于一个集合。公开人体测量缺失的 pp79、pp92、pp18 分别位于 train、validation、test。
- 稀疏网格与目标网格：首版固定 Lebedev `N=3`（26 个稀疏方向），目标为 Fliege `N=29`（900 个方向），双耳独立保存。选择条件为 `50 Hz <= f <= 20 kHz`；由于 44.1 kHz、1024 点 FFT 的离散频率栅格，实际得到 463 个频点，范围 `86.1328`–`19982.8125 Hz`。
- 数据格式：每个被试/阶数独立保存一个 HDF5；Python 侧频谱布局统一为 `[ear, direction, frequency]`。主要字段为 `/mca_logmag_db`、`/reference_logmag_db`、`/correction_logmag_db`、`/target_residual_db`、`/direction_features` 和 `/frequency_hz`。方向特征包含 azimuth、elevation、单位向量 `x/y/z` 和 Fliege weight。文件先写入 `.partial`，完整校验和元数据写入后再原子改名，可断点续跑。
- 实验过程：先执行 `export_hutubs_residual_dataset(91,3,1,'pilot_pp91_n03')` 完成 pp91 试导出，再执行 `export_hutubs_residual_dataset(1:96,3,6,'hutubs_residual_v1_n03')`，使用 6 个 MATLAB process workers 完成全量导出；全量运行耗时 `920.8 s`（约 15 分 21 秒）。项目 Python 虚拟环境安装 `h5py 3.16.0`，使用 `validate_residual_hdf5.py` 和 `compute_training_statistics.py` 做读取验证及训练集统计。
- 完整性检查：96/96 HDF5 成功，0 failure、0 partial；每个文件包含 `833,400` 个样本，总计 `80,006,400` 个样本。遍历读取全部 96 个文件后，train/validation/test 属性计数为 `72/12/12`，全部数组尺寸一致且数值有限，`reference - MCA = residual` 的最大恒等误差为 `0 dB`。输出总大小 `1,179,616,094` 字节（`1124.97 MiB`，约 `1.10 GiB`）。
- 训练集统计：仅使用 72 个 train 被试的 `60,004,800` 个样本计算归一化参数。MCA log-magnitude 均值/标准差为 `0.4367/9.3164 dB`；correction-filter log-magnitude 为 `0.8184/2.6988 dB`；目标 residual 均值/标准差为 `-0.0905/4.0625 dB`，平均绝对值为 `2.5156 dB`，范围约 `-67.71`–`69.57 dB`。
- 指标解释：这里的 residual 是逐 FFT 频点的细粒度谱差，不是论文 41 个 auditory band 上的能量误差，因此其平均绝对值不能直接与前一阶段约 0.8 dB 的 ERB 指标比较。后续训练仍应以原始 residual 为监督，并用论文 ERB、ILD、对侧高频误差作为最终评价。
- 输出文件位置：源码、划分和说明位于 `residual_learning/`；大型数据位于 `residual_learning/data/hutubs_residual_v1_n03`，由嵌套 `.gitignore` 忽略；仅保留该目录的 `training_statistics.json` 以复现训练归一化。
- 硬件准备：本机 NVIDIA GeForce RTX 5060，显存 `8151 MiB`，驱动 `595.79`，CUDA compute capability `12.0`；适合使用混合精度训练轻量 MLP，但 batch size 需按 8 GB 显存控制。
- 阻塞项：当前虚拟环境尚未安装 PyTorch；在安装前应核对支持 RTX 5060 / compute capability 12.0 的官方 CUDA wheel 版本。
- 结论与下一步：Residual 数据集 v1 已可直接供 Python 训练。下一步实现按 HDF5 随机采样的 PyTorch Dataset、仅由 train 统计量归一化的轻量 MLP，以及 zero-residual（即原 MCA）对照；先跑小规模过拟合检查，再进行完整训练。

## 2026-07-23：HUTUBS 96 被试 MCA 全量复现与跨被试汇总

- 实验目标：按论文技术评估设置，在全部 96 个 HUTUBS simulated SOFA 上完成 Lebedev `N=1`–`10` 的 SH only、SUpDEq + SH 和 MCA 基线，并汇总跨被试幅度、ILD、ITD 指标与论文 Fig. 5 风格曲线。
- 数据集与版本：本机 `data/HRTF/hutubs` 中 `pp1`–`pp96` 的 96 个 simulated SOFA；每个文件包含 Lebedev `N=35` 的 1730 个方向、双耳 256 点 HRIR，采样率 44.1 kHz。
- 稀疏网格与采样点数：稀疏输入为 Lebedev `N=1`–`10`；幅度误差目标为 Fliege `N=29`（900 方向）；双耳线索目标为水平面 0–359 度（360 方向）。ERB 指标实际得到 41 个中心频率，范围 `50`–`19792.31 Hz`，与论文的 50 Hz–20 kHz、41 个听觉滤波器一致。
- 方法与关键参数：SH only（`'None'`, `'SH'`, `mc=nan`）、conventional（`'SUpDEq'`, `'SH'`, `mc=nan`）与 MCA（`'SUpDEq'`, `'SH'`, `mc=inf`）；FFT oversize 为 4。MCA 使用 SUpDEq 默认最小相位幅度校正、空间混叠频率以下限制和 `fadeDown`。头半径由 HUTUBS `x1,x2,x3` 与 Algazi 公式计算；公开人体测量表中 pp18、pp79、pp92 缺值，三者透明地使用其余 93 人的平均半径 `0.091021 m`，这是相对论文私有完整元数据的已知偏差。
- 实验过程：使用 6 个 MATLAB process workers 按被试并行，每个被试内部按 `N=1`–`10` 顺序执行并逐阶保存独立检查点。实际命令为 `matlab -batch "addpath(fullfile(pwd,'reproduce')); run_hutubs_mca_batch(1:96,1:10,6,'hutubs_mca_batch')"`；总耗时 `11471.8 s`（约 3 小时 11 分 12 秒）。随后执行 `matlab -batch "addpath(fullfile(pwd,'reproduce')); aggregate_hutubs_mca_results"`，约 57 秒完成跨被试汇总和绘图。
- 完整性检查：96/96 被试成功，0 个失败；共 960 个逐阶 MAT、960 个 ERB CSV、960 个双耳指标 CSV、96 个完成标记；每位被试均恰有 10 个阶数。结果总大小 `203,774,606` 字节（约 194.3 MiB）。
- 统计口径修正：批处理 CSV 最初使用 Fliege 求积权重做方向平均；论文 Fig. 5 是对选中方向做普通算术平均。逐阶 MAT 已保存每个方向的跨频率误差，因此汇总阶段直接由检查点重建论文严格口径，不需要重新插值。两种口径均保留，并以文件名明确区分；论文对照图使用未加权方向平均。
- 幅度指标结果：左耳对侧 25 度区域中，N=2 conventional/MCA 为 `2.7789/1.8558 dB`，改善 `0.9231 dB`，96/96 被试改善；N=3 为 `2.6819/1.8965 dB`，改善 `0.7854 dB`，96/96 改善。左耳前方 25 度的最大改善位于 N=4，由 `0.7460 dB` 降至 `0.4319 dB`，改善 `0.3141 dB`，95/96 改善。左耳全空间在 N=3 由 `1.0980 dB` 降至 `0.8035 dB`，改善 `0.2945 dB`，96/96 改善。N=1 全空间 MCA 略差（`1.5664` 升至 `1.6370 dB`），与论文所述低阶 N=1 难以进一步改善的现象一致。
- 双耳指标结果：N=3 水平面平均绝对 ILD 误差由 conventional `1.5623 dB` 降至 MCA `0.8611 dB`；平均绝对 ITD 误差为 `11.7113/11.6454 µs`，几乎不变，符合幅度校正主要改善 ILD 而不改变低频到达时间的预期。
- 论文对照结论：复现曲线重现原文 Fig. 5 的核心模式：对侧误差显著高于前方；N=1 对侧改善很小；最大幅度改善集中在 N=2–5；前方改善峰值在 N=3–4；跨被试标准差在 MCA 后总体减小。当前结论可作为后续 residual 网络必须超过的 MCA 基线。
- 输出文件位置：根目录 `reproduce/hutubs_mca_batch`；汇总在 `reproduce/hutubs_mca_batch/aggregate`，包括严格口径 ERB 表、加权口径 ERB 表、双耳指标表、改进表、MAT 汇总以及 `fig05_region_magnitude_error.png`、`full_sphere_magnitude_error.png`、`binaural_error_by_order.png`、`frequency_error_n3_n6.png`。`reproduce/` 根目录的 MATLAB 复现脚本纳入 Git，子目录中的大型检查点和生成结果继续由 `.gitignore` 忽略。
- 阻塞项：无计算阻塞。正式写作时需注明 pp18、pp79、pp92 的头半径替代策略；若获得三人的完整人体测量值，可仅重跑这三人做敏感性检查。
- 结论与下一步：96 被试 MCA 传统基线已完成，可进入 residual 学习数据导出。下一步应先固定 subject-wise 训练/验证/测试划分，再导出 `log|H_ref| - log|H_MCA|` 目标及方向、频率、耳别、MCA 幅度与 correction-filter 特征，避免同一被试跨集合造成数据泄漏。

## 2026-07-23：HUTUBS simulated 单被试 MCA 复现检查点

- 实验目标：验证 HUTUBS simulated SOFA 与 MCA 论文的 Lebedev 网格、个体头半径和 MCA 基线参数可在本机 SUpDEq 环境中一致运行，再进入 96 位受试者的批处理。
- 数据集与版本：HUTUBS simulated `pp91_HRIRs_simulated.sofa`，本机 `data/HRTF/hutubs`；SOFA 为 `SimpleFreeFieldHRIR`，数据形状为 `1730 x 2 x 256`，采样率 `44.1 kHz`。
- 稀疏网格与采样点数：源参考为 Lebedev `N=35`（1730 点）；稀疏输入为 Lebedev `N=3`（26 点）；误差评估目标为 Fliege `N=29`（900 点）。已逐行验证 SOFA 方向与 SUpDEq Lebedev `N=35` 一致，最大方位/余纬差分别为 `5.68e-14` / `2.84e-14` 度。
- 方法与关键参数：SH only（`'None'`, `'SH'`, `mc=nan`）、conventional（`'SUpDEq'`, `'SH'`, `mc=nan`）与 MCA（`'SUpDEq'`, `'SH'`, `mc=inf`）。MCA 使用默认最小相位校正、空间混叠频率以下限制以及向下 1/3 octave fade。根据 HUTUBS `x1,x2,x3` 人体测量值和 Algazi 公式，pp91 的头半径为 `0.088883 m`（`8.8883 cm`），与论文示例的约 `8.89 cm` 一致。
- 实验过程：新增本地忽略的 `reproduce/run_hutubs_mca_p91_n3.m`，在 MATLAB R2025b Update 2 中实际执行 `matlab -batch "addpath(fullfile(pwd,'reproduce')); run_hutubs_mca_p91_n3"`；命令成功结束，退出码为 0，耗时约 50.5 秒。
- 指标结果：ERB 绝对幅度误差（方向加权、ERB 频带等权）中，左耳全空间由 conventional `1.0890 dB` 降至 MCA `0.7611 dB`，左耳对侧 25 度区域由 `2.7124 dB` 降至 `1.8014 dB`；右耳全空间由 `0.9645 dB` 降至 `0.7340 dB`，右耳对侧 25 度区域由 `2.5212 dB` 降至 `1.6867 dB`。输出曲线显示 MCA 在空间混叠频率约 `1842.54 Hz` 以上相较 conventional 进一步降低误差。
- 输出文件位置：`reproduce/pp91_n3/pp91_n3_erb_summary.csv`、`pp91_n3_results.mat`、`pp91_n3_left_erb_error.png`、`pp91_n3_left_erb_error.fig`；整个 `reproduce/` 当前已由 `.gitignore` 忽略，不提交。
- 阻塞项：初始受限环境中 MATLAB 批处理启动出现 `File system inconsistency`；以本机正常权限启动后已成功运行，当前无实验阻塞。
- 结论与下一步：单被试 `pp91, N=3` 已通过网格、头半径、运行链路与误差趋势验证。下一步将脚本泛化至 `pp1`–`pp96` 与稀疏阶数 `N=1`–`10`，并补全论文的跨受试者均值/标准差、空间误差、ILD 与 ITD 汇总。

## 2026-07-23：下载 HUTUBS HRTF 数据集

- 实验目标：获取完整 HUTUBS 数据库，作为后续 MCA 基线、跨个体 residual 学习与评估的主要数据集。
- 数据集与版本：SOFA Acoustics HUTUBS 数据库目录（访问日期 `2026-07-23`）；源地址：`https://sofacoustics.org/data/database/hutubs/`。目录文件的网页标注修改日期为 `2017-06-23` 至 `2020-01-31`。
- 数据文件：共 `253` 个文件，包括 96 位受试者的 `192` 个 measured/simulated HRIR SOFA 文件、`58` 个 3D 头部网格 PLY 文件、`2` 个 PDF 文档和 `1` 个 CSV 人体测量文件。
- 实验过程：从目录索引自动提取全部文件链接，再使用 `curl 8.21.0` 下载；启用失败重试、断点续传和最多 `8` 个并行传输。由于本机 Windows Schannel 无法访问证书吊销检查服务，下载时使用 `--ssl-no-revoke` 跳过吊销状态检查，TLS 加密连接仍保留。
- 完整性检查：实得 `253/253` 个文件，无零字节文件；`pp1`–`pp96` 的 measured/simulated SOFA 文件全部成对齐全，无缺失或意外 SOFA 文件；全部 SOFA、PDF 和 PLY 文件的格式签名检查通过。文件总大小为 `1,442,534,622` 字节（`1,375.71 MiB`，`1.343 GiB`）。
- 输出文件位置：本机 `D:\cuc\CSMT\MCAR\data\HRTF\hutubs`。数据文件已移动到 Git 仓库内的忽略目录，不提交到 Git；目录说明见 `data/HRTF/README.md`。
- 阻塞项：无。
- 结论与下一步：完整 HUTUBS 数据集已可用。后续优先读取 simulated SOFA 复现 MCA 设置，并解析 SOFA 元数据与人体测量表，固定跨个体训练/验证/测试划分。

## 2026-07-23：下载 AXD HRTF 数据集

- 实验目标：获取 AXD 个体化 HRTF SOFA 文件，作为后续跨个体 MCA residual 学习与评估的数据集。
- 数据集与版本：SOFA Acoustics AXD 数据库目录（访问日期 `2026-07-23`）；源地址：`https://sofacoustics.org/data/database/axd/`。目录文件的网页标注修改日期为 `2022-06-28` 至 `2023-03-08`。
- 数据文件：共 `140` 个 `.sofa` 文件，包括 `p0001.sofa`–`p0040.sofa` 和 `p0101.sofa`–`p0200.sofa`。
- 实验过程：使用 `curl 8.21.0` 从源目录下载；启用失败重试、断点续传和最多 `8` 个并行传输。由于本机 Windows Schannel 无法访问证书吊销检查服务，下载时使用 `--ssl-no-revoke` 跳过吊销状态检查，TLS 加密连接仍保留。
- 完整性检查：实得 `140/140` 个预期文件，无缺失、无意外文件、无零字节文件；所有文件均具有 NetCDF4/HDF5 文件签名。SOFA 文件总大小为 `413,764,409` 字节（`394.60 MiB`）。
- 输出文件位置：本机 `D:\cuc\CSMT\MCAR\data\HRTF\axd`。数据文件已移动到 Git 仓库内的忽略目录，不提交到 Git；目录说明见 `data/HRTF/README.md`。
- 阻塞项：无。
- 结论与下一步：AXD 数据集已可用。后续在数据预处理阶段读取 SOFA 元数据，核对各个体采样方向、采样率、IR 长度及坐标约定，再确定跨个体训练/验证/测试划分。

## 2026-07-23：KU100 对侧耳高频误差定量统计

- 实验目标：将 MCA demo 中“对侧耳 10 kHz 以上误差降低”的可视化趋势转化为可复现、可用于论文的双耳定量结果。
- 数据集与版本：KU100，SUpDEq 本地 `HRIR_L2702.sofa`；MATLAB R2025b。
- 稀疏网格与采样点数：源网格为 Lebedev `Ns = 3`；目标网格为 Lebedev `Nd = 44`，共 2702 个方向。
- 方法与关键参数：
  - 基线参数与 2026-07-22 实验一致：SH、SUpDEq + SH（conventional）和 MCA（`mc = inf`），头半径 `0.0875 m`。
  - SUpDEq 坐标为 `0°=前、90°=左、180°=后、270°=右`。根据球面坐标的笛卡尔横向坐标判定半球：左耳对侧为右半球，右耳对侧为左半球；排除横向坐标为零的正中面方向。每耳纳入 1321 个方向。
  - 高频范围严格定义为 `f > 10 kHz` 且 `f <= fs/2`，本次包含 150 个频点。
  - 单样本误差为估计与 reference 的绝对 log-magnitude 差，单位为 dB；幅度下限为 `-200 dB`。均值在方向维使用 Lebedev 权重、频率维等权；中位数为全部方向-频率样本的非加权中位数。
- 实验过程：
  1. 新增 `scripts/analyze_contralateral_high_frequency.m`，独立复现三种插值结果并读取 dense reference。
  2. 分别计算左耳对侧、右耳对侧和双耳汇总的 SH、conventional、MCA 高频绝对 log-magnitude error。
  3. 导出完整统计 MAT、两张 CSV 表以及双耳频率误差曲线的 PNG/FIG。
  4. 实际执行命令：`matlab -wait -batch "addpath('D:\cuc\CSMT\MCAR\scripts'); analyze_contralateral_high_frequency"`。命令成功结束，退出码为 0。
- 指标结果：
  - 左耳对侧：conventional 加权平均误差为 `5.5850 dB`，MCA 为 `4.6349 dB`，降低 `0.9500 dB`，相对改善 `17.01%`；中位数由 `4.0917 dB` 降至 `3.2449 dB`，相对改善 `20.69%`。
  - 右耳对侧：conventional 加权平均误差为 `5.1573 dB`，MCA 为 `4.6283 dB`，降低 `0.5291 dB`，相对改善 `10.26%`；中位数由 `3.7299 dB` 降至 `3.2801 dB`，相对改善 `12.06%`。
  - 双耳汇总：SH、conventional、MCA 的加权平均误差分别为 `7.5781 dB`、`5.3711 dB`、`4.6316 dB`。MCA 相对 conventional 降低 `0.7395 dB`，相对改善 `13.77%`；中位数由 `3.9044 dB` 降至 `3.2626 dB`，降低 `0.6417 dB`，相对改善 `16.44%`。
- 输出文件位置：`../artifacts/figures/mca_demo_ku100_ns3_nd44/contralateral_high_frequency/`。该目录是脚本生成且被 Git 忽略的本地实验产物，包含：
  - `contralateral_high_frequency_summary.csv`
  - `contralateral_high_frequency_improvement.csv`
  - `contralateral_high_frequency_statistics.mat`
  - `contralateral_high_frequency_error.png`
  - `contralateral_high_frequency_error.fig`
- 阻塞项：本项无运行阻塞。当前结果仅来自 KU100 单一人头模型与 `Ns = 3` 稀疏网格，尚不能说明跨个体或跨稀疏度的稳定性。
- 结论与下一步：MCA 在 KU100 双耳对侧 `>10 kHz` 区域相对 conventional 确有稳定的定量改善，解决了上一阶段“只有可视化趋势”的阻塞项。下一步固定 residual 数据集字段与训练/验证/测试划分，并实现训练特征导出脚本。

## 2026-07-22：KU100 的 MCA 基线复现与图表导出

- 实验目标：复现并比较 SH、SUpDEq + SH 和 MCA 三种 HRTF 空间插值基线，为后续 MCA residual MLP 提供输入、目标和评价基准。
- 数据集与版本：KU100，使用 SUpDEq 本地数据 `HRIR_L2702.sofa` 构建稀疏集与参考集。
- 稀疏网格与采样点数：源网格为 Lebedev `Ns = 3`；目标网格为 Lebedev `Nd = 44`，共 2702 个方向。
- 方法与关键参数：
  - SH：`ppMethod = 'None'`、`ipMethod = 'SH'`、`mc = nan`。
  - SUpDEq + SH（conventional）：`ppMethod = 'SUpDEq'`、`ipMethod = 'SH'`、`mc = nan`。
  - MCA：`ppMethod = 'SUpDEq'`、`ipMethod = 'SH'`、`mc = inf`。
  - 头半径为 `0.0875 m`；本次 MCA 使用最小相位幅度校正，并在空间混叠频率 `fA = 1871.66 Hz` 以上执行校正。
- 实验过程：
  1. 使用 `supdeq_getSparseDataset` 从 KU100 参考数据生成 `Ns = 3` 的稀疏 HRTF 集，并创建 `Nd = 44` 的目标 Lebedev 网格。
  2. 分别执行 SH、SUpDEq + SH 和 MCA 插值，得到 `interpHRTF_sh`、`interpHRTF_con` 和 `interpHRTF_mca`。
  3. 在目标网格上取得 KU100 reference HRTF，并转换得到 reference HRIR。
  4. 比较正前方与对侧左耳的 HRIR，重点检查 MCA 对对侧高频空间混叠误差的修正。
  5. 使用 `supdeq_calcLSD_HRIR` 计算三种方法的左耳 LSD，并使用 `AKerbError` 计算 ERB-band magnitude error；两组曲线均对 2702 个方向取平均。
  6. 通过 `scripts/run_mca_demo_export.m` 自动运行 `SUpDEq-master/supdeq_demo_MCA.m`，并导出全部实验图和指标数据。复现命令为 `matlab -wait -batch "run('D:\\course\\CUC_2\\CSMT\\MCAR\\scripts\\run_mca_demo_export.m')"`。
- 指标结果：
  - 已成功生成 SH、SUpDEq + SH、MCA 的左耳全方向平均 LSD 曲线和 ERB-band magnitude-error 曲线。
  - 已生成 6 张正前方或对侧耳 HRIR 对比图。demo 的对侧示例表明，MCA 能修正约 10 kHz 以上的高频凸起，使结果更接近 reference；该改善尚未形成独立的数值统计。
  - 导出脚本共生成 8 个 `.png`、8 个 `.fig` 和 `mca_demo_metrics.mat`，共 17 个可复现实验产物。
- 输出文件位置：`../artifacts/figures/mca_demo_ku100_ns3_nd44/`。该目录中的大型 MAT 文件均为本地可复现产物，不提交 Git；下列相对链接需先运行导出脚本：
  - [正前方 conventional vs MCA HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/01_frontal_conventional_vs_mca_hrir.png)
  - [对侧耳 SH vs MCA HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/02_contralateral_sh_vs_mca_hrir.png)
  - [对侧耳 conventional vs MCA HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/03_contralateral_conventional_vs_mca_hrir.png)
  - [对侧耳 SH vs reference HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/04_contralateral_sh_vs_reference_hrir.png)
  - [对侧耳 conventional vs reference HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/05_contralateral_conventional_vs_reference_hrir.png)
  - [对侧耳 MCA vs reference HRIR](../artifacts/figures/mca_demo_ku100_ns3_nd44/06_contralateral_mca_vs_reference_hrir.png)
  - [SH、SUpDEq + SH、MCA 的 LSD 曲线](../artifacts/figures/mca_demo_ku100_ns3_nd44/07_lsd_left_ear.png)
  - [SH、SUpDEq + SH、MCA 的 ERB-band magnitude-error 曲线](../artifacts/figures/mca_demo_ku100_ns3_nd44/08_erb_magnitude_error_left_ear.png)
- 阻塞项：
  1. 尚未定义并实现对侧耳区域的统一判定规则，也未对 `>10 kHz` 区域计算 conventional 与 MCA 的均值、中位数、误差差值和相对改善百分比，因此目前只有可视化趋势，缺少论文可用的定量结论。
  2. residual MLP 的训练数据生成流程尚未固定：仍需明确输入特征、log-magnitude residual 目标、训练/验证/测试划分以及跨方向或跨个体的划分方式。
  3. residual MLP 尚未实现和训练，因此当前只能完成 SH、SUpDEq + SH、MCA 三种传统基线比较，尚不能评估 MCA + residual MLP 的增益。
- 结论与下一步：MCA demo 已成功复现，传统基线和图表导出流程已经跑通。下一步先完成对侧耳高频区域的定量统计，再固定 residual 数据集格式与划分方式，最后训练轻量 MLP，并以 LSD、ERB-band magnitude error、ILD error 和对侧高频误差与 MCA baseline 比较。
