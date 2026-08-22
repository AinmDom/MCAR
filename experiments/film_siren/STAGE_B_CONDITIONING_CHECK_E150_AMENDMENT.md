# Stage B conditioning check 统一 E150 预算修订

> 冻结时点：unconditional baseline 的三个 E100 runs 全部完成并触发预算
> `RETEST`之后、任何 E150 conditioning-check run之前。本文不改变架构、优化器、
> 数据或主指标。

## 1. E100 审计证据

unconditional E100的三个seeds最佳cycles为`100/95/90`，weighted MAE分别为
`3.145802/3.144507/3.151574 dB`。seed20260821触发主协议的末点评价
`RETEST`。

使用预先存在的每5-cycle validation ledger作趋势诊断：early window为
cycles70/75/80，late window为cycles85/90/95/100；三个seeds的窗口均值改善
分别为`0.1991%/0.1364%/0.1926%`，均不低于既有latent平台规则采用的
`0.1%`阈值。因此100 cycles对unconditional baseline处于预算边缘，不能直接
通过conditioning check。

## 2. 对称 E150 复核（冻结）

为保持训练预算公平，不只延长baseline。以下六个runs全部从scratch训练150
cycles：

- frozen conditioned winner：latent128 + full + all placement，
  seeds `20260821/20260822/20260823`；
- unconditional shared plain-SIREN，完全相同三个seeds；
- 除`cycles=150`和run identity/protocol metadata外，各自与对应E100配置相同；
- 不续训E100 checkpoint，不调整LR、正则、调制边界、采样或validation频率；
- 最终conditioning check只使用六个E150 runs，E100保留为预算诊断。

## 3. E150 平台判据

best checkpoint仍按单点最低validation weighted MAE保存。预算判据对六个runs
完全相同：

1. best cycle `<150`：`KEEP`；
2. best cycle `=150`：early window为cycles100/105/110/115/120/125，late
   window为cycles130/135/140/145/150；
3. window improvement = `(early_mean-late_mean)/early_mean × 100%`；
4. improvement `<0.1%`：`KEEP_PLATEAU`；
5. improvement `>=0.1%`：`RETEST`，不通过conditioning check且不自动再延长。

该窗口规则在六个E150 runs开始前冻结，避免以单个validation噪声无限延长。

## 4. 最终判据与上限

分别计算conditioned和unconditional的E150三-seed等权均值：conditioned严格
更低则conditioning check通过，否则`DO NOT FREEZE`。同seed配对差值与seed
胜出数只作诊断。

本修订新增3个FiLM runs与3个unconditional runs：placement修订后的FiLM run
上限由34提高到37；unconditional baseline run上限由3提高到6；unique
architecture不变。所有run必须记录condition/test访问计数，SONICOM test始终
不读取。
