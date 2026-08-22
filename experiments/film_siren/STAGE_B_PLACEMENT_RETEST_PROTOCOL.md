# Stage B placement 排名反转复核协议

> 状态：PRE-REGISTERED RETEST AMENDMENT。本文在新增 placement RETEST
> training run 之前冻结。它只解决已观察到的 seed 排名反转，不修改模型、训练
> 预算或 validation 指标。

## 1. 触发原因与冻结输入

Stage B placement screening 与 Top-2 三 seed 复核已经完成。固定候选为
`hidden` 和 `all`，二者均使用 latent dimension 128、`full` modulation：

| placement | seed20260821 | seed20260822 | seed20260823 | 三 seed 均值 (dB) |
|---|---:|---:|---:|---:|
| all | 3.101953 | 3.086500 | 3.088167 | 3.092207 |
| hidden | 3.096799 | 3.091519 | 3.093828 | 3.094049 |

screening seed 上 hidden 胜出，后两个 seed 上 all 胜出，因此触发主协议第8节的
`RETEST`。六个 run 均完整训练100 cycles、best cycle早于100且
`test_subjects_read=0`；本次复核不是预算扩展。

## 2. 新增 runs（冻结）

- 候选：`hidden`、`all`；
- 新 seeds：`20260824`、`20260825`；
- 共新增4个从 scratch 的独立 run；
- 每个 run 与对应 placement 的 seed20260821 配置完全相同，只改变
  experiment/model/run identity、`search_stage=placement_retest`、seed，并记录
  本修订路径；
- 仍为100 cycles、262 steps/cycle、Adam LR `1e-4`、FP32、相同 Q26
  normalization、完整262 train subjects与44 validation subjects；
- 不改变 `gamma_max_delta=0.5`、`beta_max=pi`、
  `amplitude_max_delta=0.5`，不添加正则。饱和统计仅作诊断。

运行顺序固定为 hidden seed20260824、hidden seed20260825、all seed20260824、
all seed20260825。顺序不影响决策；每个 run 使用自己的冻结 seed。

## 3. 一次性最终判据

将原有三个 seeds 与新增两个 seeds 合并：
`{20260821,20260822,20260823,20260824,20260825}`。每个 placement 的最终分数
为五个 seed 的44-subject aggregate solid-angle-weighted residual MAE 的等权
均值；均值较低者为唯一 winner。

- 原 screening 排名是否再次反转不再是阻断条件，也不会递归追加 seed；
- 同 seed 的 `all - hidden` 配对差值、胜出次数、seed std和确定性 bootstrap
  区间只作稳定性诊断，不改变 winner；
- 若五 seed 均值恰好逐位相等，则 Stage B placement 保持 `RETEST`、
  `winner=null`，不再加 run；
- 若任一所需 run 缺失、非有限、元数据不匹配、读取test、dirty Git，分析失败，
  不输出 winner；
- 若任一新增 run 按冻结 checkpoint 规则得到 `decision=RETEST`（包括 best 在
  cycle100），立即暂停剩余训练；不得静默延长、续训或改变参数。

## 4. 上限与数据边界

截至本修订前，Stage B conditioning architecture/search 实际使用28个 FiLM
training runs；本修订新增4个，预期累计32个。为保留失败后仅按同配置同 seed
重跑的审计余量，主协议经 latent 修订后的 FiLM run 上限由30提高到34；unique
architecture上限不变。unconditional baseline 仍单独计3个 runs。

SONICOM test 不扫描、不构造路径、不打开且不参与任何选择；所有正式产物必须
记录 `test_subjects_read=0`。
