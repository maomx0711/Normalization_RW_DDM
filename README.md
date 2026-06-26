# Normalization_RW_DDM

面向三臂老虎机任务的在线实验与建模原型，用于检验除法归一化
（divisive normalization, DN）如何改变强化学习价值到漂移扩散/竞赛过程的映射。

## 项目结构

```text
Project_Target/
  experiment/
    index.html         # jsPsych + webgazer 单文件在线实验
  analysis/            # DN-RL-DDM 拟合代码
  docs/                # 研究设计与模型说明
Project_Target.md      # 原始项目目标记录
```

## 运行在线实验

```bash
cd Project_Target/experiment
python3 -m http.server 8000
```

然后在浏览器打开 `http://localhost:8000`。实验会记录：

- 三选项选择、反应时、反馈奖励；
- 区块条件（range normalization probe 或 all-options divisive-sum probe）；
- 浏览器眼动估计的原始 gaze 样本；
- 每个选项 AOI 的 dwell time、dwell proportion、first AOI 等指标。

> 浏览器眼动使用 webgazer.js，需要 HTTPS 或 localhost，并依赖被试授权摄像头。
> 实验的 CSS 和 JavaScript 已内联在 `index.html` 中，便于单文件分发和审阅。
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
