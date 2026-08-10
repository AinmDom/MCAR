# SONICOM Q26 v3.2.1：低学习率解冻 MLP 的 10 epoch 实验

本实验从固定的 v3.2 seed `20260809` epoch 39 checkpoint 开始，解冻 MLP 并与 CNN
联合微调 10 epoch。为控制灾难性遗忘，CNN 学习率固定为 `1e-5`，MLP 使用其 1/10，
即 `1e-6`；其余双采样、ERB/HF/strict-ILD 权重、训练 seed 和固定 validation sampler
均保持不变。test 没有用于训练、选模、预测或重建。

## 训练结果

- 10 epoch × 500 step，用时 `1021.001 s`；
- 174,627 个参数全部可训练，最佳 epoch 为 7；
- validation total：`0.661115 → 0.660417`，改善约 `0.106%`；
- residual/ERB/HF/strict ILD：
  `2.542327/1.021642/3.578575/0.589368 →
  2.541352/1.020789/3.578026/0.586724 dB`；
- 10 轮 optimizer skip 为 0，峰值 CUDA allocated memory `601.926 MiB`；
- W&B offline run ID：`qdlzionj`；
- 源 epoch 39 checkpoint 前后 SHA-256 不变。

MLP 的 16 个浮点 state tensor 均发生更新，但相对参数 L2 改变量只有 `0.0997%`；CNN
的 66 个 tensor 相对改变量为 `0.2774%`。因此这是一次真实但幅度很小的联合微调。

## 44 人严格 validation 重建

| 指标 | v3.2 epoch 39 (dB) | v3.2.1 epoch 7 (dB) | 相对变化 | 改善被试数 |
|---|---:|---:|---:|---:|
| 全空间 ERB | 0.872882 | **0.872411** | **+0.0540%** | 28/44 |
| 对侧 25° ERB | **1.364204** | 1.364251 | **-0.0034%** | 22/44 |
| 对侧高频 | 3.608567 | **3.607370** | **+0.0332%** | 31/44 |
| 水平面严格 ILD | 0.596141 | **0.592172** | **+0.6658%** | 28/44 |

配对统计显示：全空间 ERB 的均值差为 `-0.000472 dB`，95% CI
`[-0.000821,-0.000123]`，paired t-test `p=0.0093`；严格 ILD 差为
`-0.003969 dB`，95% CI `[-0.006395,-0.001543]`，`p=0.0020`。对侧 25° ERB
没有差异（`p=0.959`）；高频变化非常小，t-test `p=0.058`、Wilcoxon `p=0.042`。

## 结论

低学习率解冻 MLP 没有破坏模型，并对全空间 ERB 和 strict ILD 产生了可重复方向的
小幅 validation 收益；但绝对量级很小，对侧 25° ERB没有改善，高频收益也接近零。
因此 v3.2.1 可以保留为“联合微调有效”的消融证据，但当前不足以证明需要替换 v3.2
epoch 39。后续若继续，应只在 validation 上预声明更高的 MLP 学习率或更长预算；不得
读取 test 来决定参数。

完整数值见 `summary.json`、`comparison_vs_v32_epoch39.csv`、
`paired_statistics.csv`、`training_history.csv` 和 `training_report.json`；三张 44 人
结果图位于 `figures/`。
