# Normalization_RW_DDM

面向三臂老虎机任务的在线实验与建模原型，用于检验除法归一化
（divisive normalization, DN）如何改变强化学习价值到漂移扩散/竞赛过程的映射。

## 项目结构

```text
Project_Target/
  experiment/
    index.html                    # 只反馈所选选项奖励的单文件实验
    all_outcomes_feedback.html    # 反馈三个选项金额的单文件实验
  analysis/            # DN-RL-DDM 拟合代码
  docs/                # 研究设计与模型说明
Project_Target.md      # 原始项目目标记录
```

## 运行在线实验

```bash
cd Project_Target/experiment
python3 -m http.server 8000
```

然后在浏览器打开其中一个版本：

- `http://localhost:8000/index.html`: 只显示被试所选选项本次获得的点数；
- `http://localhost:8000/all_outcomes_feedback.html`: 同时显示当前三个选项本次对应的点数。

实验包含：

- 开始全屏呈现；
- 摄像头眼动检测和九点蓝点校准；
- 0-100 点结果范围和 `1 点 = 0.02 pence` 换算说明；
- 3 题理解测验；
- 12 试次训练，低于 60% 正确率时重复训练，达标后最多可自愿重复 2 次；
- 学习阶段：4 个固定三元组，每个呈现 45 次，共 180 试次；
- 迁移选择阶段：10 个线索的 45 个二元组合，每个呈现 4 次，共 180 试次；
- 显式评分阶段：10 个线索各评分 4 次，共 40 试次；
- 迁移和评分阶段不提供反馈。

实验会在选择试次记录 webgazer 原始 gaze 样本、选项 AOI 边界、AOI 停留时间、停留比例和首次注视选项。蓝点校准按钮使用最高层级显示，避免摄像头预览或 webgazer 画布打开后遮挡蓝点。

> 实验的 CSS 和 JavaScript 已内联在各自 HTML 中，便于单文件分发和审阅。
> 实验开始会请求全屏；三臂刺激用相对位置呈等腰三角形排列，以适配不同屏幕尺寸。

## 拟合 DN-RL-DDM

安装分析依赖：

```bash
cd Project_Target/analysis
python3 -m pip install -r requirements.txt
```

对 jsPsych 导出的 JSON 或 CSV 做模型比较：

```bash
python3 fit_dn_rlddm.py ../data/example_participant.json --output fit_results.csv
```

拟合单一模型：

```bash
python3 fit_dn_rlddm.py ../data/example_participant.json --variant gaze_divisive
```

默认比较五个变体：

- `none`: RW 学习价值直接进入漂移；
- `range`: `(Q_i - min(Q)) / (range(Q) + sigma)`；
- `divisive`: `Q_i / (sigma + sum_j Q_j)`；
- `gaze_range`: range normalization 前加入注视权重；
- `gaze_divisive`: all-options divisive normalization 前加入注视权重。

更完整的设计与公式见 `Project_Target/docs/model_spec.md`。
