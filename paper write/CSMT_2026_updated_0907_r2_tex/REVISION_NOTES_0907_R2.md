# 2026-09-07 R2 修改说明

- 正文五项指标：全域 ERB、对侧 25° ERB、频带 ILD、ITD 加权 MAE、全域 LSD。
- LAP 2024 LSD 移至 `supplementary_paired_statistics.tex`，并保留既有 FSC-centered paired bootstrap 结果。
- “双耳次要指标”与“局部频谱次要指标”合并为表“其余双耳指标与局部频谱指标”。
- matched-density 稀疏度表补入频带 ILD 与 ITD 加权 MAE；补充材料记录 LAP 2024 LSD 的 Q14/Q26/Q50 结果。
- 新增 FSC-Q26 参数量与推断开销表：单成员 1.519M 参数；三成员 4.557M 参数；RTX 5060 FP32 下三成员模型推断中位数 331.49 ms，端到端中位数 513.24 ms。
- 删除损失权重、稀疏度指标缺失、LAP 主表评价域及参数量/推断时间相关红色待修改提醒；保留结构图工程命名的红色提醒。
