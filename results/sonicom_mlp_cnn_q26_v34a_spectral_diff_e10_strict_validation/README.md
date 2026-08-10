# SONICOM Q26 v3.4a 高频谱差分损失消融

## 结论

v3.4a 验证了“直接约束谱形状”比继续提高单一高频幅度权重更有针对性。相对固定的
v3.2 epoch 39，4 kHz 以上的一阶谱差分误差降低 `1.164%`，二阶谱差分误差降低
`3.821%`，并且 44/44 名 validation 被试两项都改善。最终严格重建的全空间 ERB、
对侧 25° ERB 和水平面 ILD 也小幅改善；其中对侧 ERB 与 ILD 的配对检验一致显著。

但是，对侧 10–20 kHz 幅度 MAE 只改善 `0.0026%`，没有统计显著性。说明新损失确实
改善了局部谱形状，却还没有把收益充分转化为原有的高频幅度均值指标。v3.4a 适合作为
正向的目标函数消融保留，但当前证据不足以让它全面替换 v3.2 epoch 39。

本实验只使用 262 名 train 和 44 名 validation 被试；没有读取已消费的 test，
`test_subject_count_read=0`。

## 损失定义

设逐频率残差预测误差为 $e_f=\hat r_f-r_f$。新增两项分别为
$L_{D1}=\operatorname{MAE}(e_{f+1}-e_f)$ 和
$L_{D2}=\operatorname{MAE}(e_{f+2}-2e_{f+1}+e_f)$。仅保留完整差分模板均位于
4 kHz 以上的频点，并在每名被试内按 SONICOM solid-angle direction weight 汇总。
当前 Q26 频率网格等间隔，因而单位分别记为 `dB/bin` 和 `dB/bin²`。

选择 4 kHz 而不是 10 kHz 作为起点，是为了覆盖 4–10 kHz 的耳廓 notch 区域；原有
10–20 kHz 对侧高频幅度项仍保持不变。总目标只新增
`0.25 × D1/target_std + 0.15 × D2/target_std`，原
`ERB/HF/strict ILD=0.75/0.25/0.75` 不变。

## 训练设置

- 起点：固定 v3.2 seed `20260809` epoch 39，源 checkpoint SHA-256 为
  `1076EBA7EC24C25914E5569E1C14EDDB584A6649E7551194FBA38984FE11F10C`。
- 架构：保持 v3.2 MLP + kernel 7、dilation `1/2/4/8` 局部 CNN；不使用 v3.3
  attention 分支。
- 冻结边界：MLP 冻结，只训练 74,402 个局部 CNN 参数。
- 预算：10 epoch × 500 step；96 个固定 validation block。
- 采样：每步 32 个全局插值方向，另采 32 个水平面方向计算严格 HRIR ILD。
- 优化：AdamW，学习率 `1e-5`，weight decay `1e-5`，cosine，gradient clip `5`，AMP。
- seed：训练 `20260809`，固定 validation sampler `20260805`。
- 最佳 checkpoint：按 v3.4a 固定 validation total 选择 epoch 7；W&B offline ID
  `e80qlsxd`。

训练用时 `812.893 s`，optimizer skip 为 0，峰值 CUDA allocated memory 为
`399.405 MiB`。初始 total 为 `0.6915137445`，epoch 7 为 `0.6904440752`。
源到候选的 MLP 参数最大绝对差严格为 `0.0`；CNN 相对 L2 改变量为 `0.3124%`。

## 44 人严格重建

| 指标 | v3.2 epoch 39 | v3.4a epoch 7 | 相对改善 | v3.4a 更优人数 |
|---|---:|---:|---:|---:|
| 全空间 ERB | 0.872882 dB | 0.872618 dB | +0.0303% | 25/44 |
| 对侧 25° ERB | 1.364204 dB | 1.363347 dB | +0.0628% | 29/44 |
| 对侧高频 | 3.608567 dB | 3.608472 dB | +0.0026% | 25/44 |
| 水平面严格 ILD MAE | 0.596141 dB | 0.593248 dB | +0.4852% | 30/44 |

44 人配对统计如下。差值定义为 candidate minus baseline，负值表示 v3.4a 更好。

| 指标 | 均值差 | 95% CI | paired t-test | Wilcoxon | Cohen $d_z$ |
|---|---:|---:|---:|---:|---:|
| 全空间 ERB | -0.0002649 dB | [-0.0005025, -0.0000273] | 0.0298 | 0.0852 | -0.339 |
| 对侧 25° ERB | -0.0008567 dB | [-0.0016729, -0.0000405] | 0.0401 | 0.0171 | -0.319 |
| 对侧高频 | -0.0000954 dB | [-0.0015853, 0.0013945] | 0.8979 | 0.3756 | -0.019 |
| 水平面严格 ILD | -0.0028925 dB | [-0.0050757, -0.0007093] | 0.0106 | 0.0068 | -0.403 |

这些检验属于开发集消融且同时比较多个指标，不应把接近 `0.05` 的单个 p 值解释成
大效果。可靠结论是：对侧 ERB 与 ILD 的方向较一致，但绝对改善仍然很小；高频幅度
指标没有可辨别改善。

## 全方向谱差分诊断

谱差分诊断使用 44 名 validation 被试的全部 767 个纯插值方向，而不是训练时抽样
block；每名被试先按方向面积加权，再对被试等权平均。

| 谱形状指标 | v3.2 epoch 39 | v3.4a epoch 7 | 相对改善 | 改善人数 |
|---|---:|---:|---:|---:|
| 4 kHz+ 一阶差分 MAE | 0.441563 dB/bin | 0.436421 dB/bin | +1.164% | 44/44 |
| 4 kHz+ 二阶差分 MAE | 0.277322 dB/bin² | 0.266727 dB/bin² | +3.821% | 44/44 |

D1/D2 的 paired t-test 分别为 `1.85e-30 / 2.07e-29`，Wilcoxon 均为
`1.14e-13`，Cohen $d_z$ 为 `-4.52 / -4.26`。这证明实现与优化目标均有效；严格
高频幅度 MAE 几乎不变则说明“谱斜率/曲率更准确”与“逐点幅度整体更准确”并不等价。

## 版本判断与下一步

v3.4a 是一个有价值的正向 loss 消融，但它的综合严格指标不如 v3.2.1 联合微调：
v3.2.1 在全空间 ERB、高频和 ILD 上分别改善 `0.054% / 0.033% / 0.666%`，而 v3.4a
的优势主要集中在对侧 ERB 与谱形状。当前仍应保留 v3.2 epoch 39 作为开发基准。

下一项最有解释力的独立消融是分频带 ILD，而不是立刻叠加 notch-aware loss 或
hard-direction sampling。它可以直接检验双耳频谱平衡，并延续 v3.4a 在严格 ILD 上的
正向信号。notch-aware loss 随后单独做；hard sampling 最后做，因为它会改变训练方向
分布，归因风险最高。

## 文件

- `summary.json`：严格重建配置与聚合指标。
- `comparison_vs_v32_epoch39.csv`：四项最终指标直接对比。
- `paired_statistics.csv`：四项严格指标的 44 人配对统计。
- `spectral_difference_per_subject.csv`：逐被试 D1/D2 结果。
- `spectral_difference_paired_statistics.csv`：D1/D2 配对统计。
- `spectral_difference_diagnostics.json`：谱差分、checkpoint 和训练诊断。
- `training_history.csv`、`training_report.json`：训练曲线与运行摘要。
- `figures/validation44_metric_overview.png`：44 人四指标总览。
- `figures/validation44_contralateral_hrtf_overview.png`：44 人对侧 HRTF 总览。

