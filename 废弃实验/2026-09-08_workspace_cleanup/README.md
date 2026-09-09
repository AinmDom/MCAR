# 2026-09-08 工作区清理归档

本目录保存从工作区移出的、不可作为正式实验依据的恢复阶段产物，以便需要时人工追溯。

## 已归档内容

- `sonicom_ten_method_direction_sensitivity_secondary_recovery_staging_v1/`
  - 原路径：`results/sonicom_ten_method_direction_sensitivity_secondary_recovery_staging_v1/`
  - 仅包含恢复阶段最后 4 名 validation listener（P0341、P0354、P0360、P0369）的汇总。
  - `summary.json` 虽标记 `completed`，但 `subject_count=4`，不满足正式协议要求的 44 名 listener；不得作为论文或正式实验结果引用。
- `sonicom_ten_method_direction_sensitivity_secondary_v1.partial/`
  - 原路径：`results/sonicom_ten_method_direction_sensitivity_secondary_v1.partial/`
  - 空的中间输出目录，没有可验证结果。

对应 evaluator 的结构化复数 HDF5 读取修正保留在
`scripts/evaluate_ten_method_direction_sensitivity_secondary.py`，不属于本废弃归档。

如需恢复调查，可将上述目录移回各自原路径；恢复不等于将其升级为正式证据。
