# FSP-AE-Q26 SONICOM validation 报告

## 结论

FSP-AE-Q26 adaptation 已完成正式 train/validation 预算锁定和 44 人严格重建评价。
锁定模型在对侧高频与水平面严格 ILD 上分别优于 MCAR v3.2 `14.68%` 和
`4.37%`，其中高频逐被试 `44/44` 胜出；但全空间与对侧 25° ERB 分别回退
`31.16%` 和 `33.60%`。因此 FSP-AE 是具有明显高频/双耳线索优势的互补横向
基线，而不是当前 MCAR 主模型的全面替代。

## 实验协议

- 输入为固定 SONICOM-Q26-v1 的 26 个测量方向，目标网格为 793 个方向。
- 数据为 262 train / 44 validation；幅度与 ITD 统计只由 train 计算。
- 网络、参数名、官方 checkpoint 加载和 HRIR 重建已与上游 FSP-AE 逐值验证，
  幅度、ITD、HRIR 最大绝对误差均为 0；参数量为 `235,065`。
- SONICOM 适配采用 `44.1 kHz`、`1024` 点 FFT、512 个非直流正频率点，以及
  官方的 LSD 加 `2500 × ITD L1` 损失。
- 预声明同一训练轨迹的 10/20/40 epoch 快照；主选择标准为固定 validation
  复合损失最小值，平局时依次选择较低 LSD 和较早 epoch。
- 严格评价只纳入 767 个纯插值方向，使用与既有横向表完全相同的全空间 ERB、
  对侧 25° ERB、对侧高频与水平面 HRIR ILD 口径。
- test 未用于训练、选模、推理或严格评价，读取数为 0。

## 预算与锁定

40 epoch 正式训练用时 `972.423 s`，峰值 CUDA allocated memory 为
`526.671 MiB`。预算节点的 validation 复合损失为：

| Epoch | Validation LSD | Validation ITD L1 | 复合损失 |
|---:|---:|---:|---:|
| 10 | 3.367172 dB | 3.095580e-5 s | 3.444562 |
| 20 | 3.195450 dB | 2.167831e-5 s | 3.249646 |
| 40 | 2.994123 dB | 1.304956e-5 s | 3.026746 |

epoch 24 和 36 后的两次学习率下降都带来进一步改善，最佳点为 epoch 40，
没有依据提前锁定 10 或 20 epoch。锁定 checkpoint 位于 Git 忽略的
`artifacts/training/sonicom_fsp_ae_q26_formal_budget_40/best.pt`。

## 严格 validation 结果

均值 ± 被试标准差，单位为 dB，越低越好：

| 方法 | 全空间 ERB | 对侧 25° ERB | 对侧高频 | 水平面严格 ILD |
|---|---:|---:|---:|---:|
| SH only | 2.666 ± 0.155 | 3.830 ± 0.361 | 9.016 ± 0.621 | 3.629 ± 0.636 |
| SUpDEq + SH | 1.819 ± 0.237 | 2.246 ± 0.152 | 6.006 ± 0.366 | 1.739 ± 0.446 |
| SUpDEq + NN | 1.849 ± 0.250 | 2.219 ± 0.166 | 5.662 ± 0.286 | 1.563 ± 0.474 |
| SUpDEq + Barycentric | 1.742 ± 0.220 | 2.162 ± 0.159 | 5.527 ± 0.271 | 1.533 ± 0.440 |
| MCA | 1.096 ± 0.145 | 1.764 ± 0.177 | 4.749 ± 0.234 | 0.830 ± 0.181 |
| MCAR v3.2 | 0.885 ± 0.149 | 1.383 ± 0.165 | 3.638 ± 0.257 | 0.625 ± 0.175 |
| FSP-AE-Q26 | 1.160 ± 0.162 | 1.848 ± 0.175 | 3.104 ± 0.254 | 0.598 ± 0.153 |

FSP-AE 相对 MCA 在高频和 ILD 上改善 `34.65% / 27.99%`，逐被试胜出
`44/44 / 38/44`；全空间和对侧 ERB 则回退 `5.89% / 4.79%`。相对 MCAR，
高频和 ILD 改善 `14.68% / 4.37%`，逐被试胜出 `44/44 / 24/44`；两项 ERB
均为 `0/44` 胜出。

## 质量检查与解释

严格评价包含 44 个完整被试行，所有指标均为有限数；每名被试固定为 767 个纯
插值方向。FSP-AE 频率网格与严格评价网格最大误差为 0，reference ILD 元数据
最大复算误差为 `9.536e-7 dB`。

结果表明，直接联合学习幅度与 ITD 能显著增强对侧高频和双耳时间/能量线索，
但其全局频谱精度仍弱于以 MCA 为物理先验的 MCAR。后续若研究组合方法，应在新的
未见拆分或外部数据上预声明融合规则；不得使用已经完成一次性评价的 SONICOM test
继续调参。

精选结果位于 `results/sonicom_fsp_ae_q26_formal_validation/`。

## 横向结果图

![七方法严格指标总览](../results/sonicom_fsp_ae_q26_formal_validation/figures/figure_1_method_overview.png)

![FSP-AE 与 MCAR 逐被试对比](../results/sonicom_fsp_ae_q26_formal_validation/figures/figure_2_fsp_vs_mcar_subjects.png)

![FSP-AE 横向权衡](../results/sonicom_fsp_ae_q26_formal_validation/figures/figure_3_fsp_tradeoff.png)
