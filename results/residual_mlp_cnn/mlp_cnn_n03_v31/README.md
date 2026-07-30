# MLP-CNN v3.1 selected results

这是从原 v3 epoch 9 继续进行严格 HRIR-ILD 微调后、在 validation 上锁定的
epoch 3 结果。这里仅保存适合 Git 跟踪的 CSV、JSON 和 PNG；checkpoint、
预测 HDF5、MAT 与 W&B 本地缓存位于被忽略的 `artifacts/`。

- `training/`：配置、训练历史、训练报告以及完整 validation/test residual 指标。
- `evaluation/`：12 名 test 被试的严格 ERB、高频、ILD 与重建质量结果。
- `evaluation/figures/`：逐被试 HRTF 图、12 被试总览和版本对比图。

完整方法、模型选择过程和结论见
`reports/MLP_CNN_V31_REPORT.md`。正式 W&B run：
<https://wandb.ai/luyoung/mcar-mlp-cnn-v31/runs/yhl3z70n>。
