# Stage B conditioning architecture 搜索状态

> 状态：`RETEST / DO NOT FREEZE`  
> 冻结时点：placement Top-2 的六个100-cycle runs全部完成并由预注册分析器汇总后。

## 已完成决策

- latent dimension：`128`；E150三seed均值 `3.103014 dB`，优于256的
  `3.105485 dB`；
- modulation：`full`；三seed均值 `3.091884 ± 0.002857 dB`，优于phase的
  `3.108313 ± 0.000850 dB`；
- placement screening：hidden `3.096799 dB`、all `3.101953 dB`、late
  `3.121709 dB`；hidden/all差距小于0.5%，因此补齐三seeds。

以上全部正式 runs 均记录 `test_subjects_read=0`，没有读取SONICOM test。

## Placement 阻断结果

| placement | seed20260821 | seed20260822 | seed20260823 | 三seed均值 ± std (dB) |
|---|---:|---:|---:|---:|
| all | 3.101953 | 3.086500 | 3.088167 | 3.092207 ± 0.006925 |
| hidden | 3.096799 | 3.091519 | 3.093828 | 3.094049 ± 0.002161 |

- screening seed20260821为hidden胜；seed20260822和20260823均为all胜；
- 三seed均值排序因此反转为all第一，但相对差距仅约`0.060%`；
- 按 `STAGE_B_FILM_PROTOCOL.md` 第8节，“seed排名发生反转”必须标记
  `RETEST`，不得临时扩展搜索；分析器据此输出 `winner=null`；
- 这不是训练预算问题：六个run的best cycles均早于cycle100，全部为`KEEP`。

## 调制饱和诊断（best checkpoint）

- all的seed方差较大（std `0.006925 dB`，约为hidden的3.2倍）；
- all seed20260821在cycle40达到best，此时绝大多数调制参数尚未饱和；
- all seed20260822/23在cycle75/65达到best，第5层amplitude saturation分别约
  `92.1% / 87.8%`，第6层gamma约`37.6% / 20.2%`；
- hidden三seeds的第5层amplitude saturation约`97.8%–99.4%`，第6层gamma约
  `67.9%–77.7%`；hidden更稳定，但同样存在明显边界挤压。

因此当前结果既不能按单seed冻结hidden，也不能忽略预注册规则按三seed均值直接
冻结all。Stage B在此暂停；在正式修订RETEST规则前，不运行unconditional
baseline、condition ablation或Stage C。

## 后续需要正式化的选择

若继续，应先提交新的RETEST协议，明确额外seed数量、是否调整full调制边界/正则、
winner判据和新增run上限，再运行任何训练。不能依据当前结果临时挑选有利方案。
