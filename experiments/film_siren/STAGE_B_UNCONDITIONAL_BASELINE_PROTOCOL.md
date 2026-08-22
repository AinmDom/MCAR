# Stage B unconditional shared-SIREN baseline 协议

> 状态：PRE-REGISTERED BASELINE PROTOCOL。本文在任何 unconditional baseline
> 正式 run 之前冻结。

## 1. 目的与模型

检验已冻结的 conditioned architecture（latent128 + full modulation + all
placement）是否确实利用了 subject condition，而不只是共享 SIREN 的训练收益。
baseline 是一个跨262个 train subjects共享的 plain SIREN：

- dual frequency coordinate，input dimension 5；
- 6个 sine layers、width256、first/hidden omega20、双耳输出；
- 不包含 condition encoder、latent、FiLM head或其他 subject-specific input；
- 不读取 Q26 magnitude、Q26 direction indices或Q26 normalization。

## 2. 对齐的训练与验证

- seeds：`20260821/20260822/20260823`，三个 run 均从 scratch；
- 100 cycles × 262 steps，每cycle每个train subject恰好一次；
- 每step从767 interpolation directions等概率无放回采样16方向；
- normalized residual MSE，Adam LR `1e-4`、weight decay0、gradient clip5、FP32；
- 每5 cycles在44 validation subjects的全部767 interpolation directions上评价；
- best checkpoint、weighted MAE/RMSE、MCA zero-residual baseline、末点评价
  `RETEST`规则与 conditioned runs相同；
- SONICOM test不扫描、不打开，`test_subjects_read=0`；正式报告另记录
  `condition_inputs_read=0`。

## 3. 冻结判据

baseline分数为三个 seeds 的44-subject aggregate weighted MAE等权均值。公平的
conditioned对照使用 frozen `all` placement 在相同三个 seeds
`20260821/22/23`上的均值；placement RETEST新增的两个 seeds只用于架构选择，
不混入这项三-seed配对比较。

- conditioned均值严格低于 unconditional均值：conditioning check通过；
- conditioned均值大于或等于 unconditional均值：Stage B标记`DO NOT FREEZE`，
  先诊断 conditioning失效，不进入Stage C；
- 任一 baseline run 缺失、非有限、读取condition/test、dirty Git或报告
  `decision=RETEST`：不作通过判定，不静默延长或调参。

同seed的 `conditioned - unconditional` 配对差值与胜出数只作诊断，不改变上述
三-seed均值判据。baseline另计3个正式training runs，符合主协议预算。
