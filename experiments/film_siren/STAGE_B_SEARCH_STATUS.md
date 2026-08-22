# Stage B conditioning architecture 搜索状态

> 状态：`PLACEMENT FROZEN / BASELINE PENDING`
> 冻结时点：placement RETEST 的新增四个100-cycle runs全部完成并由预注册
> 五-seed分析器汇总后。

## 已完成决策

- latent dimension：`128`；E150三seed均值 `3.103014 dB`，优于256的
  `3.105485 dB`；
- modulation：`full`；三seed均值 `3.091884 ± 0.002857 dB`，优于phase的
  `3.108313 ± 0.000850 dB`；
- placement screening：hidden `3.096799 dB`、all `3.101953 dB`、late
  `3.121709 dB`；hidden/all差距小于0.5%，因此补齐三seeds。

以上全部正式 runs 均记录 `test_subjects_read=0`，没有读取SONICOM test。

## Placement 三-seed阻断结果

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

该结果不能按单seed冻结hidden，也不能忽略预注册规则按三seed均值直接冻结all，
因此按下节的预注册RETEST处理。

## RETEST 最终结果

`STAGE_B_PLACEMENT_RETEST_PROTOCOL.md` 在新增训练前冻结：hidden/all各追加
seed20260824与20260825，不改变full调制边界、正则或100-cycle预算；最终使用
五seed等权均值一次性决胜。

| placement | 五seed均值 ± std (dB) | seed胜出数 |
|---|---:|---:|
| all | 3.091073 ± 0.005545 | 4/5 |
| hidden | 3.093648 ± 0.002327 | 1/5 |

- 10/10个所需run完整，均为`KEEP`，`test_subjects_read=0`；
- 配对均值 `all - hidden = -0.002575 dB`，即all约优`0.083%`；
- 配对均值的exact bootstrap 95%区间为`[-0.005852, 0.001622] dB`，跨0，
  因而优势很小且稳定性证据有限；该区间按冻结协议仅作诊断；
- 按冻结的五seed均值判据，placement winner正式确定为`all`。

Stage B conditioning architecture现冻结为：latent128 + full modulation + all
placement。下一步是三seed unconditional shared-SIREN baseline；完成并确认
conditioned winner优于baseline后，才做condition shuffle/train-mean latent消融并
考虑进入Stage C。
