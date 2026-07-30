# MLP + CNN v3.1：严格 HRIR-ILD 对齐微调报告

## 摘要

v3.1 的目标不是扩大网络，而是修复 v3 的训练目标与最终 ILD 评估口径不一致
的问题。v3 使用完整双耳频谱 CNN 后，测试集三项幅度指标均优于 v2，但严格
HRIR 能量 ILD MAE 从 v2 的 `0.646722 dB` 回升到 `0.661566 dB`。原训练器
使用选定频点幅度能量和作为 ILD proxy；最终评估则保留 MCA 相位、补回频带外
频谱、构造共轭对称双边谱、IFFT、裁剪 256 点 HRIR 后再计算双耳能量差。

本版将后一条完整链路实现为 PyTorch 可微损失，并从锁定 v3 checkpoint
微调冻结 MLP 后的 CNN。模型在 validation 上完成选择后才进行一次 test
评估。最终 v3.1 在 12 个未见被试上的严格 ILD MAE 为 `0.652495 dB`，
相对 v3 改善 `1.371%`；全空间 ERB 仅回退 `0.138%`，对侧高频误差改善
`0.033%`。v3.1 部分修复了 v3 的 ILD 退化，但仍未超过 v2 的 ILD 单项成绩。

## 需求与验收标准

本实验遵守以下约束：

1. 不改变 v3 的 MLP-CNN 架构，不以增加容量掩盖目标函数问题。
2. MLP 继续冻结，仅更新 CNN residual correction。
3. 严格 ILD 损失必须与 MATLAB 最终重建的相位、双边谱、IFFT 和 HRIR
   裁剪口径对齐。
4. 只使用 72 名 train 和 12 名 validation 被试开发与选型。
5. validation 锁定 checkpoint 后，只对 12 名 test 被试评估一次。
6. 保留全空间 ERB、对侧 25° ERB、对侧 `>10 kHz` 与 ILD 四项严格指标，
   并输出 12 名 test 被试的 HRTF 图。
7. checkpoint、HDF5 与 W&B 缓存保留在 `artifacts/`；精选 CSV、JSON 与
   PNG 发布到 `results/`。

## v3 问题诊断

原 ILD proxy 直接对训练选定的 463 个单边频点计算左右能量和。它忽略了：

- 50 个训练频带外单边频点；
- MCA 复频谱相位；
- 单边谱到双边谱的共轭对称构造；
- IFFT 后有限长度 HRIR 的裁剪；
- 相位与裁剪共同引起的时域能量变化。

因此 proxy 与最终严格 ILD 有相关性，但并非同一个目标。v3 的严格测试结果
说明，优化 proxy 不保证 HRIR 能量 ILD 单调改善。

## 数据协议升级

`matlab/+mcar/export_hutubs_residual_dataset.m` 新增可选的严格 ILD 元数据
导出。开启后，HDF5 schema 从 1.0 升级到 1.1，并增加：

- 选中频点的 MCA 相位，形状 `[2, 900, 463]`；
- 频带外 MCA 复频谱实部和虚部，形状 `[2, 900, 50]`；
- 选中与频带外的零基频点索引；
- 每个方向的参考严格 HRIR ILD；
- 单边频谱长度 `513` 与 HRIR 裁剪长度 `256`。

训练与 validation 共 84 个 HDF5 已重新导出并校验。验证器确认
`70,005,600` 个 residual 样本、72/12 subject split、完整 513 点频率覆盖，
以及零残差恒等误差为 0。test HDF5 没有为调参而提前升级或访问。

## 可微严格 ILD

设网络校正后的选中频点幅度为 \(\hat A[k]\)，MCA 相位为 \(\phi[k]\)。
选中频点复谱为：

\[
\hat H[k] = \hat A[k] e^{j\phi[k]}.
\]

频带外频点沿用原 MCA 复谱。拼成 513 点单边谱后，按实信号约束镜像得到
1024 点双边谱，计算 IFFT 并取前 256 个样本。左右耳能量与 ILD 为：

\[
E_e = \sum_{n=0}^{255} h_e[n]^2,\qquad
\mathrm{ILD}=10\log_{10}\frac{E_L+\epsilon}{E_R+\epsilon}.
\]

损失对预测 ILD 与参考 HRIR ILD 取绝对误差，并支持按方向积分权重加权。
实现位于 `src/mcar/losses.py`，整个过程保留 autograd。

单元测试覆盖：

- 预测等于参考时严格 ILD loss 为 0；
- 单耳幅度扰动后 loss 为正；
- 反向传播梯度有限且非零。

真实 pp91 CUDA smoke test 还确认 zero-init 恒等误差为 0、严格 ILD 与旧 proxy
数值不同、第二个优化步骤有 62 个 CNN 梯度张量非零。

## 训练策略与模型选择

### 受控候选

首先从 v2 零初始化 CNN，分别测试严格 ILD 权重 `0.25` 和 `1.0`：

- 权重 0.25 的完整 validation 严格 ILD 为 `0.660663 dB`，比 v2 退化；
- 权重 1.0 的严格-ILD checkpoint 为 `0.655195 dB`，但 raw residual
  MAE 增至 `2.110205 dB`。

这说明从零重新学习时，短训练中的严格 ILD 约束与幅度目标存在明显竞争。

### 锁定方案

最终采用从原 v3 checkpoint 继续微调：

- 初始化：原 v3 epoch 9；
- 冻结 v2 MLP，仅训练 74,402 个 CNN 参数；
- 总参数：174,627；
- epoch：6；
- 每 epoch 500 step；
- 每 step 32 个方向；
- learning rate：`1e-4`，余弦调度；
- 严格 ILD 权重：`1.0`；
- 自动混合精度：开启；
- 随机种子与原实验保持一致。

训练器同时保存总 validation loss 最优和采样严格 ILD 最优 checkpoint。
完整 validation 穷举评估后，在查看 test 前锁定总损失最优的 epoch 3：

| validation 模型 | raw residual MAE (dB) | 严格 ILD MAE (dB) |
|---|---:|---:|
| v2 | 2.139782 | 0.658844 |
| 原 v3 | 2.072657 | 0.656999 |
| v3.1 epoch 3 | 2.078053 | 0.654040 |

v3.1 epoch 3 的 validation 严格 ILD 相对 v2 改善 `0.729%`，并保留原 v3
的大部分 raw residual 收益。训练 6 epoch 耗时 `286.39 s`，无跳过的
optimizer step，峰值 CUDA allocated memory 为 `221.96 MiB`。

W&B：
<https://wandb.ai/luyoung/mcar-mlp-cnn-v31/runs/yhl3z70n>

## 锁定测试结果

### 完整 raw residual

test 含 12 名未见被试、`10,000,800` 个 residual 样本：

| 方法 | MAE (dB) | RMSE (dB) |
|---|---:|---:|
| MCA（zero residual） | 2.600671 | 4.186115 |
| v2 MLP | 2.168373 | 3.552562 |
| v3.1 | 2.092767 | 3.448699 |

v3.1 相对 v2 的 raw MAE 降低 `3.487%`，相对 MCA 降低 `19.530%`，
12/12 被试均优于 v2。与原 v3 的 `2.091041 dB` 相比仅回退约 `0.083%`。

### 严格 HRTF 重建

| 指标 | MCA | v2 | v3 | v3.1 | v3.1 相对 MCA |
|---|---:|---:|---:|---:|---:|
| 全空间 ERB (dB) | 0.802741 | 0.611503 | 0.584593 | 0.585398 | 27.08% |
| 对侧 25° ERB (dB) | 1.860286 | 1.340068 | 1.304415 | 1.310334 | 29.56% |
| 对侧 >10 kHz (dB) | 4.258336 | 3.720378 | 3.620148 | 3.618947 | 15.01% |
| 水平面 ILD MAE (dB) | 0.885425 | 0.646722 | 0.661566 | 0.652495 | 26.31% |

相对原 v3：

- 严格 ILD 改善 `1.371%`，8/12 被试改善；
- 对侧高频改善 `0.033%`，6/12 被试改善；
- 全空间 ERB 回退 `0.138%`；
- 对侧 25° ERB 回退 `0.454%`。

相对 v2，v3.1 的前三项幅度指标仍分别改善约 `4.269%`、`2.219%` 和
`2.726%`，且每项均为 12/12 被试改善；ILD 仍比 v2 高约 `0.893%`，
只有 4/12 被试优于 v2。

## 重建正确性与图形

预测 residual 只回填 MCA log-magnitude，复相位保持不变。12 名 test 被试
的最大左右相位误差为 `6.12e-16/5.99e-16 rad`，最大幅度恒等误差均为
`7.11e-15 dB`，远低于断言阈值。

精选输出位于：

- `results/residual_mlp_cnn/mlp_cnn_n03_v31/training/`
- `results/residual_mlp_cnn/mlp_cnn_n03_v31/evaluation/`
- `results/residual_mlp_cnn/mlp_cnn_n03_v31/evaluation/figures/`

图形包含 12 张逐被试 HRTF 重建图、12 被试对侧 HRTF 总览、指标总览、
v1/v2/v3/v3.1 总对比和逐被试 ILD 对比。

## 结论与后续建议

v3.1 验证了核心假设：训练损失与最终 HRIR 能量 ILD 对齐后，可以在几乎
不牺牲 v3 幅度性能的前提下，稳定收回一部分 ILD 退化。它应作为当前
“幅度与 ILD 折中”的推荐 MLP-CNN checkpoint；若只追求 ILD 单项，v2
仍然更优。

下一步不建议继续查看 test 调权重。更可靠的路线是：

1. 在现有 validation 上做 subject-balanced 的多目标 checkpoint 选择；
2. 报告 v2、v3、v3.1 的 Pareto 前沿，而不是强行用单一模型统治全部指标；
3. 扩展到多个 Lebedev 稀疏阶数，验证严格 ILD 损失是否跨稀疏度稳定；
4. 若继续改进网络，引入很小的双耳共享/反对称分支，并保持严格 ILD
   链路不变；
5. 新一轮结构或超参数研究应使用新的 validation 设计或交叉验证，不再以
   当前 test 结果反向调参。
