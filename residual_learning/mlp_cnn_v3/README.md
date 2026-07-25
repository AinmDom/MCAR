# MLP + CNN residual learning v3

本目录用于在 `mlp_n03_v2` 的基础上，引入双耳多尺度频谱 CNN，构建
MLP + CNN 的 v3 工作流。v3 首版保持数据划分、目标、训练代理损失和最终
MATLAB 评价口径不变，只考察频率上下文与 CNN/MLP 融合带来的增益。

## 设计目标

v2 MLP 对每个方向、耳朵和频点独立预测：

```text
target_residual_db = reference_logmag_db - mca_logmag_db
```

v3 将 v2 的预测作为稳定基线，再由双耳 1D CNN 预测增量：

```text
v3_residual = v2_mlp_residual + cnn_delta_residual
```

CNN 在完整 463 点频谱上工作，联合读取左右耳，使其能够学习相邻频点的谱峰、
谱谷、notch 和双耳能量关系。方向单位向量通过 FiLM 调制 CNN 中间特征。

## 首版结构

- 基础网络：原 `ResidualMLP`，7 输入、宽度 128、3 个 residual block。
- CNN 输入通道：
  - 左耳归一化 MCA magnitude；
  - 左耳归一化 correction-filter magnitude；
  - 左耳 v2 MLP residual；
  - 右耳归一化 MCA magnitude；
  - 右耳归一化 correction-filter magnitude；
  - 右耳 v2 MLP residual；
  - 归一化 log-frequency。
- CNN stem：`Conv1d(7, 48, kernel_size=7)`、GroupNorm、SiLU。
- CNN 主干：4 个深度可分离 residual block，dilation 为 `1, 2, 4, 8`。
- 方向条件：`x/y/z -> Direction MLP -> per-block FiLM`。
- 输出：`Conv1d(48, 2, kernel_size=1)`，零初始化，分别预测左右耳增量。
- 膨胀卷积主干感受野为 91 个位置；计入 kernel-7 stem 后，相对原始输入的
  完整理论感受野为 97 个频点，约覆盖 4.18 kHz。

输出层零初始化保证新模型加载 v2 checkpoint 后，在训练开始时严格满足：

```text
v3 prediction == v2 prediction
```

## 目录

```text
mlp_cnn_v3/
├── python/
│   ├── mlp_cnn_model.py
│   └── smoke_test_mlp_cnn.py
├── runs/
└── reconstruction/
```

`runs/` 和 `reconstruction/` 中的 checkpoint、缓存及临时产物默认忽略。
后续确认需要保留的轻量 CSV、JSON 和 PNG 再单独加入版本控制。

## 环境与 W&B

v3 在 residual-learning 公共依赖之外增加 W&B：

```powershell
D:\miniconda3\envs\ml\python.exe -m pip install `
  -r residual_learning/mlp_cnn_v3/requirements.txt
```

在线训练前只需在本机完成一次登录：

```powershell
D:\miniconda3\envs\ml\python.exe -m wandb login
```

API key 只应保存在 W&B 的本机凭据中，不写入命令、源码、JSON、Git 或实验日志。
训练器默认 `--wandb-mode disabled`，以保持原有本地运行行为；需要监控时显式使用
`--wandb-mode online`。默认 project 为 `mcar-mlp-cnn-v3`。

W&B 每个 epoch 记录：

- train/validation 的 total loss、residual、ERB proxy、对侧高频和 ILD；
- CNN delta 平均绝对值；
- validation 相对初始 v2 的五项改善百分比；
- learning rate、AMP scale、当轮及累计跳步数；
- epoch 耗时、峰值 CUDA allocated memory；
- 当前 epoch 是否成为最佳 checkpoint，以及当前最佳 epoch。

默认不会上传 `best.pt`、`last.pt` 或数据集。若确实需要观察梯度，可额外增加
`--wandb-watch`；它会带来额外运行开销，因此正式首轮训练不启用。

## 第一步烟雾测试

从项目根目录执行：

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/smoke_test_mlp_cnn.py `
  --dataset-root residual_learning/data/hutubs_residual_v1_n03 `
  --checkpoint residual_learning/runs/mlp_n03_v2/best.pt `
  --subject 91 --directions 4
```

测试使用真实 pp91 HDF5 和 v2 checkpoint，检查：

1. v2 MLP 权重能无损载入；
2. v3 输入、基础预测、CNN 增量和最终输出形状正确；
3. 零初始化时 v3 与 v2 输出逐点相同；
4. v2 的 residual、ERB、高频和 ILD 复合损失可以前向计算；
5. CNN 参数可以进行有限值反向传播；
6. MLP 冻结时不会意外产生梯度。

## 下一步

烟雾测试通过后，使用独立训练入口进行 CNN-only 训练。训练器会加载 v2
checkpoint、冻结全部 MLP 参数、只优化 CNN + FiLM，并对每个 epoch 使用完全
相同的固定 validation batch 序列。

pp91 短程过拟合命令：

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/train_mlp_cnn_v3.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/runs/mlp_n03_v2/best.pt `
  --run-name overfit_pp91_cnn_only `
  --overfit-subject 91 `
  --epochs 4 --steps-per-epoch 120 --validation-steps 48 `
  --directions-per-batch 16
```

输出包括初始 v2 固定验证指标、逐 epoch history、`best.pt`、`last.pt` 和训练
报告。为隔离架构贡献，第一轮保持 v2 的 ERB/对侧高频/ILD 权重
`0.50/0.25/0.25` 不变。

### pp91 检查结果

上述命令已在 RTX 5060 上实际完成。4 epoch × 120 steps 总耗时 `35.74 s`，
最佳结果为 epoch 4。固定的 48 个 validation batch 上：

| 指标 | 初始 v2 | CNN-only epoch 4 | 相对降低 |
|---|---:|---:|---:|
| 复合损失 | 0.53335 | 0.40730 | 23.63% |
| residual MAE | 1.9304 dB | 1.6244 dB | 15.85% |
| ERB proxy MAE | 0.6441 dB | 0.5436 dB | 15.60% |
| 对侧高频 MAE | 3.0559 dB | 2.3722 dB | 22.37% |
| ILD proxy MAE | 0.5875 dB | 0.3170 dB | 46.05% |

epoch 4 的 CNN 增量平均绝对值为 `0.8512 dB`。训练只更新 74,402 个 CNN +
FiLM 参数，MLP 全程冻结；峰值 CUDA allocated memory 为 `119.82 MiB`。

首次使用 PyTorch 默认 AMP loss scale 65536 时，零初始化输出头的首步放大梯度
发生溢出。训练器现将初始 scale 固定为 1024，并显式检测、记录和安全跳过溢出
步。本次正式检查 480 个优化步的跳步数为 0，最终 scale 保持 1024。

该结果仅证明 CNN 分支对单被试存在可学习能力，不代表跨被试泛化。下一阶段应
使用原 72/12 train/validation 划分进行 CNN-only 训练，并继续保持 test 集锁定。

## W&B 完整跨被试训练

此前未接入 W&B 的运行被主动停止在完整 epoch 6；其中 checkpoint 和 history
有效，但没有正常结束时才生成的 `training_report.json`。该目录保留为本地中断
实验，不复用其 run name，也不与新的 W&B 曲线拼接。

从原始 v2 epoch 9 重新开始正式 W&B 运行：

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/train_mlp_cnn_v3.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/runs/mlp_n03_v2/best.pt `
  --run-name mlp_cnn_n03_v3_cnn_only_wandb_online `
  --epochs 10 --steps-per-epoch 500 --validation-steps 96 `
  --directions-per-batch 32 `
  --wandb-mode online `
  --wandb-project mcar-mlp-cnn-v3 `
  --wandb-run-name mlp_cnn_n03_v3_cnn_only
```

如账户需要显式指定团队或用户名，可补充：

```text
--wandb-entity <entity>
```

网络不可用时，可以先使用 `--wandb-mode offline` 训练；之后在生成的
`runs/<run-name>/wandb/` 下执行 `wandb sync`。该目录和 W&B 本地缓存均由
`runs/.gitignore` 忽略。

### 正式 CNN-only 训练结果

上述在线训练已完成，W&B run 为
[mlp_cnn_n03_v3_cnn_only](https://wandb.ai/luyoung/mcar-mlp-cnn-v3/runs/qtkrxras)，
run id `qtkrxras`。根据固定 validation 复合损失选择 epoch 9：

| validation 指标 | 初始 v2 | v3 epoch 9 | 相对降低 |
|---|---:|---:|---:|
| 复合损失 | 0.59411 | 0.57343 | 3.48% |
| residual MAE | 2.1391 dB | 2.0733 dB | 3.07% |
| ERB proxy MAE | 0.6959 dB | 0.6646 dB | 4.50% |
| 对侧高频 MAE | 3.2972 dB | 3.2169 dB | 2.43% |
| ILD proxy MAE | 0.6465 dB | 0.6412 dB | 0.82% |

训练总耗时 `364.19 s`（约 6 分 4 秒），峰值 CUDA allocated memory
`220.46 MiB`，5000 个优化步的 AMP 跳步数为 0，最终 AMP scale 为 4096。
epoch 9 的 validation CNN delta 平均绝对值为 `0.5657 dB`。`best.pt` 为
epoch 9，`last.pt` 为 epoch 10，两个 checkpoint 均已成功重新载入。

epoch 10 的 residual 和 ERB 分别继续降至 `2.0732/0.6639 dB`，但高频和 ILD
小幅回升，使复合损失从 epoch 9 的 `0.57343` 变为 `0.57359`；因此严格按照
预先约定保留 epoch 9。下一步需要在完整 validation 文件上遍历全部样本计算
raw residual MAE/RMSE，再决定是否解锁固定 test 并执行 MATLAB 严格重建指标。

## 完整 validation residual 评估

训练时的 validation 由固定随机方向 batch 构成。正式进入 test 前，使用穷举
评估器遍历 12 个 validation 被试的全部 900 方向、双耳和 463 个频点：

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/evaluate_mlp_cnn_v3.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt `
  --split val --directions-per-block 32
```

评估器从同一个 checkpoint 同时计算：

- zero residual，即原 MCA；
- 冻结在 v3 内部的 v2 MLP；
- v2 MLP + CNN delta 的最终 v3；
- CNN delta 平均绝对值；
- 每个 validation 被试的独立 MAE/RMSE 和 v3 相对 v2 改善。

方向分块但不切分 463 点频率轴，确保 CNN 始终看到完整频谱。评估器会将内置
v2 结果与已有 `mlp_n03_v2/val_metrics.json` 交叉核对，偏差超过 `1e-4 dB`
立即报错。为避免意外提前访问 test，`--split test` 还必须显式增加
`--allow-test`。

### 完整 validation 结果

epoch 9 best checkpoint 已完成穷举评估：

| 方法 | Raw residual MAE | Raw residual RMSE | MAE 相对 MCA 改善 |
|---|---:|---:|---:|
| MCA（zero residual） | 2.57191 dB | 4.14377 dB | 0% |
| v2 MLP | 2.13978 dB | 3.50698 dB | 16.80% |
| v3 MLP + CNN | 2.07266 dB | 3.42891 dB | 19.41% |

v3 相对 v2 的 MAE 降低 `3.14%`，RMSE 降低 `2.23%`；CNN delta 的全样本
平均绝对值为 `0.5687 dB`。12/12 个 validation 被试的 MAE 均低于 v2，
逐被试改善范围为 `1.58%–4.44%`，未发现退化被试。

评估器重算得到的 v2 MAE/RMSE 与原 `mlp_n03_v2/val_metrics.json` 的差值分别
只有 `1.10e-8/-1.83e-8 dB`，远小于 `1e-4 dB` 一致性阈值。完整评估耗时
`5.15 s`，峰值 CUDA allocated memory 为 `35.20 MiB`。

该结果满足进入固定 test 严格评估的前置条件，但本步骤尚未读取 test。

## 锁定 test 与严格 HRTF 重建结果

在 validation 结论固定后，不再调整网络、损失或超参数，解锁 epoch 9
checkpoint 的 12 个 test 被试。完整 raw residual 评估命令为：

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/evaluate_mlp_cnn_v3.py `
  residual_learning/data/hutubs_residual_v1_n03 `
  residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt `
  --split test --allow-test --directions-per-block 32
```

12 个 test 被试共 `10,000,800` 个样本；结果如下：

| 方法 | Raw residual MAE | Raw residual RMSE | MAE 相对 MCA 改善 |
|---|---:|---:|---:|
| MCA（zero residual） | 2.60067 dB | 4.18612 dB | 0% |
| v2 MLP | 2.16837 dB | 3.55256 dB | 16.62% |
| v3 MLP + CNN | 2.09104 dB | 3.45217 dB | 19.60% |

v3 相对 v2 的 MAE 降低 `3.57%`，12/12 被试均改善。重算的 v2 MAE/RMSE
与历史 test 结果仅相差 `-7.68e-9/-9.18e-8 dB`。

严格重建分两步执行。Python 保留完整 463 点频率轴，按方向分块进行双耳
CNN 推理：

```powershell
D:\miniconda3\envs\ml\python.exe `
  residual_learning/mlp_cnn_v3/python/predict_mlp_cnn_reconstruction.py `
  residual_learning/mlp_cnn_v3/reconstruction/mlp_cnn_n03_v3_cnn_only `
  residual_learning/reconstruction/mlp_n03_v1 `
  residual_learning/mlp_cnn_v3/runs/mlp_cnn_n03_v3_cnn_only_wandb_online/best.pt `
  residual_learning/data/hutubs_residual_v1_n03/training_statistics.json `
  --directions-per-block 32
```

MATLAB 随后把 residual 加到 MCA log-magnitude，并严格保留原 MCA 相位：

```powershell
matlab.exe -batch `
  "addpath('residual_learning/mlp_cnn_v3/matlab'); evaluate_mlp_cnn_v3_reconstruction; plot_v1_v2_v3_reconstruction_comparison"
```

| 严格重建指标 | MCA | v2 MLP | v3 MLP-CNN | v3 相对 MCA改善 | v3 相对 v2 |
|---|---:|---:|---:|---:|---:|
| 全球 ERB | 0.80274 | 0.61150 | 0.58459 dB | 27.18% | 改善 4.40% |
| 对侧 25° ERB | 1.86029 | 1.34007 | 1.30441 dB | 29.88% | 改善 2.66% |
| 对侧 >10 kHz | 4.25834 | 3.72038 | 3.62015 dB | 14.99% | 改善 2.69% |
| ILD MAE | 0.88542 | 0.64672 | 0.66157 dB | 25.28% | 退化 2.30% |

v3 的三项幅度指标均进一步优于 v2，但 ILD 相对 v2 小幅退化，说明当前
CNN-only 损失中的 ILD proxy 尚不足以保证最终 HRIR 能量 ILD 单调改善。
相对 MCA，v3 在全部 12 个被试的四项指标上仍全部改善。重建恒等检查的最大
相位误差为 `6.22e-16 rad`，最大幅度回填误差为 `7.11e-15 dB`。

本地完整结果位于
`mlp_cnn_v3/reconstruction/mlp_cnn_n03_v3_cnn_only/`，其中包括 12 张被试
重建图、12 被试 HRTF 总览、指标总览和 v1/v2/v3 对比图。该目录由
`reconstruction/.gitignore` 忽略，不提交预测 HDF5、MAT 或批量生成图。
