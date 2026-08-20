# Stage B latent 搜索预算修订：统一 150 cycles 复核

> 冻结时点：latent 64/128/256 的 seed20260821 screening 与 Top-2
> 128/256 的三个100-cycle seeds 全部完成后；任何150-cycle rerun开始前。

## 触发证据

- latent128 best cycles：`85 / 95 / 100`；
- latent256 best cycles：`85 / 75 / 100`；
- 两个配置的 seed20260823 均在cycle100达到本run最佳，按主协议标记
  `RETEST`；
- 该现象跨两个latent同时发生，判断为100-cycle预算对部分seed不足，而非
  单个配置异常；
- 原三seed均值（仅作诊断）：latent128 `3.107508 dB`，latent256
  `3.108659 dB`，但在预算复核前不冻结winner。

## 冻结修订

- latent Top-2（128/256）× 三个原 seeds（20260821/22/23）全部从 scratch
  统一重跑 **150 cycles**；
- 不从100-cycle checkpoint续训，不改变初始化、subject order、optimizer、
  validation interval或其他参数；run name追加 `_e150`；
- latent最终决策只使用6个150-cycle run，100-cycle结果保留为预算诊断；
- 若任一150-cycle run仍在cycle150达到最佳，则 Stage B latent阶段整体标记
  `RETEST` 并暂停，不再自动延长；
- 本修订新增6个FiLM runs，因此 Stage B FiLM training run 上限由24提高到30；
  unique architecture上限不变；
- test不读取，`test_subjects_read=0`。

选择150而不是对单个seed追加任意epoch，是为了给全部Top-2配置与seed相同的
50%预算扩展，同时保持计算上限明确、比较对称和从scratch可复现。
