# Stage B latent E150 平台判据修订

> 冻结时点：首个 E150 run（latent128/seed20260821）触发“cycle150单点最佳”
> 暂停后、其余5个E150 run开始前。本文不改变训练、checkpoint或排名指标，
> 只修正预算平台判据。

## 1. 暂停审计证据

首个E150 run的 interpolation-weighted validation MAE：

- cycles100–125（6个评价点）均值：`3.1156504553 dB`；
- cycles130–150（5个评价点）均值：`3.1157310894 dB`；
- 后窗相对前窗变化：约 `-0.0026%`（后窗略差，不是持续改善）；
- cycles100–150评价点std：`0.0047711 dB`；
- cycle150单点相对先前best改善：`0.0025274 dB`，小于一个窗口std。

结论：cycle150恰好成为单点best，但窗口均值没有下降趋势。仅以“最后一个评价
点是否为最小值”判断预算不足，对当前validation波动过敏，会导致无上限延长。

## 2. 冻结平台规则

对每个150-cycle run：

1. 仍按单点最小weighted MAE保存best checkpoint；
2. 若best cycle `<150`，预算判定为 `KEEP`；
3. 若best cycle `=150`，计算：
   - early-window = cycles100/105/110/115/120/125的MAE均值；
   - late-window = cycles130/135/140/145/150的MAE均值；
   - window improvement = `(early-late)/early × 100%`；
4. window improvement `<0.1%` → `KEEP_PLATEAU`：末点best视为平台噪声中的
   最低点，不再延长；
5. window improvement `>=0.1%` → `RETEST`：Stage B latent整体暂停，不冻结
   winner，也不自动追加预算。

0.1%阈值在其余5个E150 run开始前冻结，统一应用于latent128/256的全部三个
seeds。最终架构排名仍使用每run best checkpoint的44-subject weighted MAE，
三seed等权平均；不删除或替换任何seed。
