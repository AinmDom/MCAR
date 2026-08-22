# Stage B condition 因果消融协议

> 状态：PRE-REGISTERED EVALUATION PROTOCOL。本文在读取三个 E150 winner
> checkpoints进行任何消融评价前冻结。消融不训练、不选模、不访问test。

## 1. 冻结对象与主参考

对象为conditioning check已通过的三个E150 best checkpoints：latent128 + full
modulation + all placement，seeds `20260821/20260822/20260823`。每个seed的正常
condition性能直接读取该checkpoint已保存的44-subject best validation结果。

## 2. 两项推理时干预

1. **deterministic condition shuffle**：按validation subject ID升序排列，将每个
   target subject配给下一个subject的Q26 condition，最后一个循环配给第一个；
   44个subjects均无self-match。query坐标与target仍属于原target subject；
2. **train-mean latent**：用同一checkpoint的condition encoder分别编码全部262
   train subjects的完整Q26 condition，对262个latent作逐维算术平均；对全部44
   validation targets使用这个固定mean latent。

两项干预都只替换送入冻结FiLM-SIREN的latent，不改backbone、checkpoint、query、
target、metric或normalization。

## 3. 指标与解释边界

使用与主实验完全相同的767 interpolation directions、solid-angle-weighted
residual MAE/RMSE和44 subjects等权均值。逐seed报告：normal、shuffle、
train-mean-latent以及相对normal的MAE变化。

- 消融变差支持模型使用了subject-specific Q26信息；
- 消融持平或改善提示condition可能被忽略或编码退化，需要诊断；
- 结果只作因果解释，不改变已经冻结的architecture winner，也不重新选择seed或
  checkpoint；
- 不设置事后显著性门槛，不据消融结果调参重测。

输入只允许262 train与44 validation subjects。SONICOM test路径不构造、不打开，
输出记录`test_subjects_read=0`；正式运行要求干净Git并记录checkpoint hashes。
