# SONICOM Q26 MCAR v3.5：previous/scratch 输出融合

## 结论

本模型正式记为 **MCAR v3.5**。上一轮 CNN 重初始化后低学习率联合解冻的模型，与本次不加载任何 checkpoint、MLP 和
CNN 同时从零训练的模型，采用 residual 输出级凸组合。两模型由不同随机初始化得到，
因此不平均参数。固定 44 人 validation sampler 上搜索后，最佳组合为上一轮模型
`30%`、scratch 模型 `70%`，validation total loss 为 `0.641223`，低于上一轮
单模型的 `0.672247` 和 scratch 单模型的 `0.646540`。

严格 validation 重建的四项指标为：

| 方法 | 全空间 ERB | 对侧 25° ERB | 对侧高频 | strict ILD |
|---|---:|---:|---:|---:|
| 上一轮联合模型 | 0.885664 | 1.388553 | 3.652797 | 0.600091 |
| scratch joint e60 | 0.837668 | 1.302105 | 3.542947 | 0.598536 |
| MCAR v3.5（30/70 输出融合） | **0.831843** | **1.291992** | **3.540258** | **0.581337** |

相对 scratch，融合分别改善 `0.695% / 0.777% / 0.076% / 2.874%`。全空间
ERB、对侧 ERB 和 ILD 的 44 人 paired t-test 分别为
`p=1.52e-11 / 3.65e-4 / 1.20e-4`；对侧高频改善很小且不显著
（`p=0.255`）。相对上一轮联合模型，四项分别改善
`6.077% / 6.954% / 3.081% / 3.125%`。

全流程只读取 validation，`test_subject_count_read=0`。该融合是新的
validation-only 候选，不改写已经消费过的一次性 test 主结论。

## 文件

- `fusion_report.json`：数据源、选择协议、最佳权重和完整 proxy sweep。
- `validation_proxy_sweep.csv`：所有候选权重的固定 validation 指标。
- 严格重建逐被试结果位于相邻的
  `sonicom_mlp_cnn_q26_v32_previous_scratch_fusion_strict_validation/`。
- 融合后的 44 人 residual 位于
  `artifacts/reconstruction/sonicom_q26_validation_fusion_previous_scratch/`。
- 合并后的六模型表和配对统计位于
  `results/sonicom_mlp_cnn_q26_v32_cnn_reinit_joint_v1/`。
