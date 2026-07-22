# 本项目的本地 SUpDEq 依赖说明

此目录存放未纳入本仓库版本控制的第三方 MATLAB 工具包 **SUpDEq**，包括其
`materials/` 中的示例 HRTF/HRIR 数据及随附第三方依赖。不要在此目录提交源
码、数据或 MATLAB 生成文件。

## 获取方式

从官方仓库克隆到本目录：

```powershell
git clone https://github.com/AudioGroupCologne/SUpDEq.git SUpDEq-master
```

随后在 MATLAB 中运行 `SUpDEq-master/supdeq_start.m`。本项目的 MCA 图表导出脚
本还依赖该工具包提供的 KU100 数据 `materials/HRIRs_ref/HRIR_L2702.sofa`。