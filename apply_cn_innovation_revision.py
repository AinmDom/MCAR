from pathlib import Path
import argparse
import shutil
import sys


INTRO_OLD = r'''时间对齐可以在插值前去除随方向变化的传播时延，从而降低相位快速变化带来的建模难度\cite{R3}。幅度校正与时间对齐插值（Magnitude-Corrected and Time-Aligned Interpolation，MCA）进一步利用频率平滑的幅度表示构造校正滤波器，在传统插值链路中同时处理时延和幅度误差\cite{R1}。然而，MCA仍会保留方向相关的剩余幅度误差，尤其集中于对侧高频和局部频谱结构。与直接从稀疏输入重建完整复数HRTF相比，在MCA结果上学习残差可以保留已有插值链路的相位与低频结构，并将学习目标集中于传统方法尚未消除的误差。

基于这一思路，本文提出FiLM条件化SIREN与频谱CNN模型（FiLM-conditioned SIREN with Spectral CNN，FSC）。模型由三个协同模块组成：首先，稀疏双耳空间观测经条件编码器压缩为被试级embedding特征，特征级线性调制（Feature-wise Linear Modulation，FiLM）\cite{R7}据此生成各层调制参数；随后，正弦表示网络（Sinusoidal Representation Network，SIREN）\cite{R11}以方向、频率和MCA幅度为坐标建立连续残差场；最后，双耳频谱卷积网络联合处理左右耳整谱，对跨频率局部结构和双耳关系进行进一步细化。模型仅预测MCA幅度残差，重建阶段沿用MCA相位及未预测频点，从而保持信号处理基线与学习模块之间的明确接口。

本文的主要工作如下：（1）提出一种基于MCA残差学习与条件隐式表示的稀疏HRTF插值框架，将学习目标由“直接重建完整HRTF”转化为“MCA后的幅度残差补偿”，在保留传统插值与重建链路的同时聚焦其剩余误差；（2）将每个被试的稀疏空间双耳观测压缩为固定维度的被试级embedding，并利用FiLM以该嵌入调制SIREN连续残差场，再结合双耳频谱CNN完成跨频率细化；（3）以ERB听觉频带误差作为正文核心评价维度，在训练中引入ERB代理目标，并以严格HRIR域ERB评价检验最终重建；（4）在统一Q26测试协议下与传统及学习式方法比较，并通过Q14/Q26/Q50独立训练的稀疏度实验考察模型随观测方向数量变化的行为；（5）在ARI外部数据库上保持模型结构、目标和E130/E190轮次预算不变地重新训练与评价，检验MCA残差补偿能否在另一数据库复现。实验结果显示，FSC的主要收益集中于ERB和频带ILD等幅度/声级指标，而ITD及全域LSD相对不同外部方法仍存在取舍。'''

INTRO_NEW = r'''时间对齐可以在插值前去除随方向变化的传播时延，从而降低相位快速变化带来的建模难度\cite{R3}。幅度校正与时间对齐插值（Magnitude-Corrected and Time-Aligned Interpolation，MCA）进一步利用频率平滑的幅度表示构造校正滤波器，使时延补偿、较平滑的频谱重建以及相位恢复均由显式信号处理链路完成\cite{R1}。然而，MCA仍可能留下具有结构性的方向相关幅度误差，尤其集中于对侧高频以及局部谱峰和谱凹口附近。

这一现象促使本文采用残差学习而非直接重建完整HRTF。与让神经网络重新学习MCA已经能够表示的全部成分相比，本文将MCA保留为显式重建先验，仅学习其剩余幅度误差。因此，目标HRTF幅度被分解为MCA基线与学习残差两部分。该分解在保留既有相位与重建链路的同时，使学习模块集中处理传统插值器仍未充分重建的频谱结构。

剩余MCA误差同时表现出两类互补结构。其一，误差随声源方向和频率连续变化，并具有明显个体差异，因此适合采用由个体信息调制的共享连续残差表示；其二，HRTF的精细结构具有显著的局部频率相关性，相邻频点共同构成谱峰与谱凹口，左右耳频谱之间也存在耦合关系，而纯粹的逐坐标隐式表示并不会显式编码这些局部跨频率相互作用。

基于上述考虑，本文提出FiLM条件化SIREN与频谱CNN模型（FiLM-conditioned SIREN with Spectral CNN，FSC）。条件编码器首先概括被试相关的声学特征，特征级线性调制（Feature-wise Linear Modulation，FiLM）\cite{R7}据此调制正弦表示网络（Sinusoidal Representation Network，SIREN）\cite{R11}，在方向--频率域表示连续MCA残差场；随后，双耳频谱CNN联合处理左右耳整谱，对局部跨频率结构和耳间关系进行进一步细化。两部分承担互补角色：条件隐式场建模连续、个体相关的残差变化，频谱CNN则显式处理独立坐标查询本身不能直接表示的局部频谱邻域。模型仅预测MCA幅度残差，重建阶段沿用MCA相位及未预测频点。

另一个设计考虑来自训练目标本身。在线性频率网格上逐频点计算幅度误差会等价对待不同FFT频点，而人耳的频率分辨率并非在线性频率轴上均匀分布，更适合由听觉频带组织。因此，本文在优化阶段引入ERB导向的代理项，使残差细化同时受到听觉频带尺度的约束；最终评价则使用严格HRIR域ERB指标，而不是直接以训练代理作为最终评价。这里ERB用于描述听觉频带尺度上的误差，并不替代主观听音评价，因此仍保留ILD、ITD和LSD作为互补指标。

本文的主要贡献如下：（1）将稀疏HRTF插值表述为显式MCA重建之上的残差细化，而非直接预测完整HRTF。传统的时间对齐、幅度校正、相位与重建链路均被保留，学习模块专门处理MCA之后仍存在的结构化幅度误差；（2）提出连续表示与局部细化互补的残差架构。由被试条件调制的FiLM-SIREN在方向--频率域表示连续MCA残差场，双耳频谱CNN则显式细化局部跨频率和耳间频谱结构，使连续坐标表示与局部频谱邻域建模分别承担不同功能；（3）构建听觉频带导向的残差学习目标，在训练中引入可微ERB代理项，并以严格HRIR域ERB误差作为核心评价维度，同时保留ILD、ITD和LSD等互补指标；（4）在统一协议下通过传统与学习式方法比较、CNN配对消融、Q14/Q26/Q50独立训练的稀疏度实验以及ARI数据库独立重训练，对上述设计进行验证。'''


LEARNING_OLD = r'''HRTF个性化既可以依赖人体形态信息，也可以依赖稀疏声学测量，两类任务所提供的个体信息并不相同\cite{R15,R16}。本文关注后者，即目标个体已具有少量实测HRTF的稀疏插值问题。HRTF Field以神经场将声源方位映射为连续HRTF幅度，从而在不同空间采样方案之间获得统一表示\cite{R22}；Thuillier等提出的球面神经过程元学习器则以同一被试的稀疏HRTF观测集合作为条件，在单位球面上查询未观测方向，同时输出预测不确定度\cite{R26}。FSP-AE将正则化线性回归解释为自编码器，并利用声源方向、频率和输入类型共同处理幅度与ITD\cite{R4}；RANF结合检索HRTF、卷积和双向序列建模，并可在目标个体上进行参数适配\cite{R5}。与上述学习式方法相比，FSC以前置MCA重建为参照，仅学习附加幅度残差，ITD则由复用MCA相位后的完整HRIR重建间接决定。由于不同方法在个体适配、相位恢复和输出表示上存在差异，本文统一评价其最终重建结果，而不将原论文分数与本项目重评价结果直接混排。'''

LEARNING_NEW = r'''HRTF个性化既可以依赖人体形态信息，也可以依赖稀疏声学测量，两类任务所提供的个体信息并不相同\cite{R15,R16}。本文关注后者，即目标个体已具有少量实测HRTF的稀疏插值问题。HRTF Field以神经场将声源方位映射为连续HRTF幅度，从而在不同空间采样方案之间获得统一表示\cite{R22}；Thuillier等提出的球面神经过程元学习器则以同一被试的稀疏HRTF观测集合作为条件，在单位球面上查询未观测方向，同时输出预测不确定度\cite{R26}。FSP-AE将正则化线性回归解释为自编码器，并利用声源方向、频率和输入类型共同处理幅度与ITD\cite{R4}；RANF结合检索HRTF、卷积和双向序列建模，并可在目标个体上进行参数适配\cite{R5}。这些研究表明，非线性模型与连续表示能够有效支持HRTF插值。本文的区别不在于使用神经插值器本身，而在于所学习的函数：FSC在学习模型之前保留一个显式传统重建，并表示相对于该重建仍未消除的幅度误差。MCA由此提供明确的信号处理参照，学习模型负责对结构化剩余误差进行细化，同时保留MCA相位与原有重建路径。由于不同方法在个体适配、相位恢复和输出表示上存在差异，本文统一评价其最终重建结果，而不将原论文分数与本项目重评价结果直接混排。'''


IMPLICIT_OLD = r'''FiLM通过条件生成的逐特征缩放和平移对中间特征进行调制\cite{R7}，SIREN则利用周期激活表示连续信号及其高频细节\cite{R11}。卷积网络可通过局部连接和权重共享建模邻域结构\cite{R21}；本文据此在连续残差场后引入双耳频谱CNN，以显式利用邻近频点和左右耳联合信息。本文的贡献不在于首次提出FiLM、SIREN或卷积频谱建模，而在于它们与MCA残差接口、双耳频谱细化及分阶段优化的组合。与此同时，空间音频中的客观误差与主观感知并非一一对应\cite{R18}，因此本文分别报告频谱、ILD和ITD结果，不将单一指标下降直接解释为定位或外化感改善。'''

IMPLICIT_NEW = r'''FiLM通过条件生成的逐特征缩放和平移对中间特征进行调制\cite{R7}，SIREN则利用周期激活表示连续信号及其高频细节\cite{R11}。这些性质使二者的组合适合表示以连续声源方向和频率为自变量、同时具有个体差异的残差函数。然而，基于坐标的隐式预测本身不会显式约束相邻频点之间的局部关系，而这种局部关系正是HRTF谱峰、谱凹口等结构的重要组成部分，双耳重建还进一步涉及左右耳频谱之间的耦合。卷积网络通过局部连接和共享卷积核提供显式的局部邻域归纳偏置\cite{R21}。因此，FSC将连续残差表示交由FiLM-SIREN完成，将局部双耳频谱细化交由后续频谱CNN完成，而不是把CNN作为缺乏明确动机的附加模块。

频谱目标的设计同样基于听觉频率分辨率。等效矩形带宽（Equivalent Rectangular Bandwidth，ERB）提供一种基于听觉滤波器的频率尺度，其带宽随频率升高而增大\cite{R23}。这意味着频谱偏差不仅可以在单个FFT频点上描述，也可以在听觉频带积分后进行刻画。因此，FSC将ERB同时作为优化和评价尺度：训练阶段使用可微代理，最终报告的严格ERB误差则由完整重建HRIR计算。该区分避免把训练代理本身直接当作最终评价准则。与此同时，空间音频中的客观误差与主观感知并非一一对应\cite{R18}，因此本文分别报告频谱、ILD和ITD结果，不将单一指标下降直接解释为定位或外化感改善。'''


METHOD_OLD = r'''本研究以幅度校正与时间对齐插值（Magnitude-Corrected and Time-Aligned Interpolation，MCA）为基础，学习其插值结果中尚未消除的HRTF幅度残差。MCA在时间对齐插值之外引入基于频率平滑表示的幅度校正\cite{R1}。
本文将这一传统重建结果保留为基线，利用稀疏观测提供的个体信息预测附加残差，而非直接回归完整复数HRTF。FSC的核心是以被试级条件嵌入调制连续MCA残差场，并由FiLM条件调制、SIREN隐式残差表示和CNN双耳频谱细化三个模块构成一体化网络。三者在同一前向链路中依次完成个体条件生成、逐方向逐频率残差表示和跨频率双耳校正；分阶段优化仅是训练策略，不改变完整模型由三个模块共同组成的定义。'''

METHOD_NEW = r'''本研究以幅度校正与时间对齐插值（Magnitude-Corrected and Time-Aligned Interpolation，MCA）为基础，学习其插值结果中尚未消除的HRTF幅度残差。MCA在时间对齐插值之外引入基于频率平滑表示的幅度校正\cite{R1}。
本文将这一传统重建结果保留为基线，利用稀疏观测提供的个体信息预测附加残差，而非直接回归完整复数HRTF。这种分解并非单纯的初始化策略，而是在训练和推断过程中始终将MCA固定为显式信号处理先验，并将学习函数定义为相对于该先验所需的校正。FSC的核心是以被试级条件嵌入调制连续MCA残差场，并由FiLM条件调制、SIREN隐式残差表示和CNN双耳频谱细化三个模块构成一体化网络。三者在同一前向链路中依次完成个体条件生成、逐方向逐频率残差表示和跨频率双耳校正；分阶段优化仅是训练策略，不改变完整模型由三个模块共同组成的定义。'''


def replace_once_or_skip(text, old, new, marker, name):
    if marker in text:
        print(f"Already updated: {name}")
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{name}: expected exactly one copy of the old text, found {count}. "
            "Your local main.tex may differ from the current remote version."
        )
    print(f"Updating: {name}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize the Chinese innovation narrative with the latest English manuscript."
    )
    parser.add_argument(
        "--paper-dir",
        default=r"paper write/grand_paper",
        help="Path to the Chinese grand_paper directory",
    )
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir).resolve()
    main_tex = paper_dir / "main.tex"

    if not main_tex.exists():
        raise FileNotFoundError(f"main.tex not found: {main_tex}")

    original = main_tex.read_text(encoding="utf-8")
    updated = original

    updated = replace_once_or_skip(
        updated, INTRO_OLD, INTRO_NEW,
        "本文的主要贡献如下：（1）将稀疏HRTF插值表述为显式MCA重建之上的残差细化",
        "Introduction motivation and contributions",
    )
    updated = replace_once_or_skip(
        updated, LEARNING_OLD, LEARNING_NEW,
        "本文的区别不在于使用神经插值器本身，而在于所学习的函数",
        "Learning-based related-work distinction",
    )
    updated = replace_once_or_skip(
        updated, IMPLICIT_OLD, IMPLICIT_NEW,
        "FSC将连续残差表示交由FiLM-SIREN完成",
        "Implicit representation and ERB rationale",
    )
    updated = replace_once_or_skip(
        updated, METHOD_OLD, METHOD_NEW,
        "这种分解并非单纯的初始化策略",
        "Method-level MCA-prior clarification",
    )

    if updated == original:
        print("No changes needed; the Chinese manuscript already contains this revision.")
        return

    backup = paper_dir / "main.before_innovation_revision_cn.tex"
    if not backup.exists():
        shutil.copyfile(main_tex, backup)

    main_tex.write_text(updated, encoding="utf-8")

    print()
    print(f"Updated: {main_tex}")
    print(f"Backup : {backup}")
    print("Only main.tex was changed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
