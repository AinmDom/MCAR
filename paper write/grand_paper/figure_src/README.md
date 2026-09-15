# 中文稿图3--图8重建说明

以下命令均从仓库根目录运行。它们只生成带 `_zh` 后缀的中文版资源，
不会覆盖英文稿图片；图1和图2不在这些命令的处理范围内。

图3和图4由冻结的SONICOM Q26输入及既定FSC预测生成：

```powershell
matlab -batch "addpath('matlab'); mcar.generate_fsc_individual_reconstruction_figures('zh');"
```

图5和图6使用英文图源目录内的冻结CSV：

```powershell
& '.venv/Scripts/python.exe' 'paper write/grand_paper_en/figure_src/plot_learning_methods.py' `
  --language zh --output-dir 'paper write/grand_paper/figure'

& '.venv/Scripts/python.exe' 'paper write/grand_paper_en/figure_src/plot_cnn_ablation.py' `
  --language zh --output-dir 'paper write/grand_paper/figure'
```

图7和图8复用英文稿的几何及稀疏度绘图函数。当前机器没有历史记录中使用的
外部版面对齐辅助脚本，因此本地重建时跳过该项附加检查，生成后需进行视觉检查：

```powershell
& '.venv/Scripts/python.exe' 'paper write/grand_paper_en/figure_src/make_submission_figures.py' `
  --only quantitative --language zh `
  --output-dir 'paper write/grand_paper/figure' --skip-alignment-qa
```

Python依赖声明位于 `paper write/grand_paper_en/figure_src/requirements.txt`。
所有中文版图片沿用英文图的数值、误差线、坐标范围、颜色和符号编码，只替换
图内文字及输出路径。
