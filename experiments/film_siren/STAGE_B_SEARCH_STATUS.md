# Stage B conditioning architecture 搜索状态

> 状态：`STAGE B FROZEN / READY FOR STAGE C`
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
placement。

## Unconditional baseline E100 预算阻断

| seed | best cycle | weighted MAE (dB) | decision |
|---|---:|---:|---|
| 20260821 | 100 | 3.145802 | RETEST |
| 20260822 | 95 | 3.144507 | KEEP |
| 20260823 | 90 | 3.151574 | KEEP |

- unconditional E100三seed均值为`3.147294 ± 0.003072 dB`；
- matched conditioned all-placement E100均值为`3.092207 ± 0.006925 dB`，
  三个seed均胜，表面相对改善`1.750%`；
- 但seed20260821的best在cycle100，按预注册规则 conditioning check保持
  `PENDING`；
- 三个baseline seeds从cycles70–80到85–100的窗口均值仍约改善
  `0.14%–0.20%`，不能把末点best直接视为平台噪声。

在正式预算修订并完成公平的长预算复核前，不运行condition shuffle/train-mean
latent消融，也不进入Stage C。

## Conditioning check E150 最终结果

`STAGE_B_CONDITIONING_CHECK_E150_AMENDMENT.md` 在六个长预算runs前冻结，
conditioned winner与unconditional baseline均以相同三个seeds从scratch训练150
cycles：

| model | 三seed均值 ± std (dB) | best cycles |
|---|---:|---|
| conditioned latent128/full/all | 3.092959 ± 0.004507 | 45 / 75 / 65 |
| unconditional shared SIREN | 3.143885 ± 0.001311 | 135 / 135 / 145 |

- 6/6 runs完整，均在cycle150前达到best，预算判据全部`KEEP`；
- conditioned在三个matched seeds上全部胜出，相对unconditional改善`1.620%`；
- 所有runs均记录`test_subjects_read=0`，unconditional另记录
  `condition_inputs_read=0`；
- conditioning check正式通过，Stage B不再因baseline预算阻断。

下一步仅对三个conditioned E150 best checkpoints做deterministic condition shuffle
和train-mean latent推理消融，不重新训练或选模；消融完成后再决定是否冻结整个
Stage B并进入Stage C。

## Condition 消融结果

三个E150 winner checkpoints按预注册协议完成了纯推理干预：

| evaluation | 三seed均值 MAE (dB) | 相对normal变化 |
|---|---:|---:|
| normal condition | 3.092959 | — |
| deterministic condition shuffle | 3.157990 | +2.10% |
| train-mean latent | 3.138841 | +1.48% |

- shuffle在3/3 seeds上均变差，各seed相对劣化约`1.25%–2.80%`；
- train-mean latent也在3/3 seeds上均变差，各seed相对劣化约
  `0.88%–1.81%`；
- 消融只替换冻结模型的latent，不训练、不重新选择checkpoint；
- 评价运行记录干净Git和`test_subjects_read=0`。

结论：conditioned模型不仅优于无condition的共享SIREN，而且在错误condition与
去个体化latent下稳定退化，支持其确实使用了subject-specific Q26信息。Stage B
架构正式冻结为`latent128 + full modulation + all placement`，可进入Stage C；
SONICOM test仍完全未读取。
