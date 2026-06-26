# 三臂老虎机中的除法归一化 RL-DDM

## 研究目标

本项目把三个线索放在同一个在线任务中：

1. 决策神经科学中的价值除法归一化：选项价值不仅由自身决定，也受其它备选项价值抑制。
2. 强化学习中的在线价值更新：被试需要通过反馈学习三臂老虎机的隐藏奖励概率。
3. 漂移扩散/序贯采样模型：选择和反应时共同约束价值到决策过程的映射。

实验记录选择、反应时和浏览器眼动指标，用模型比较回答：

- range normalization 是否解释不同价值范围下的选择/RT 变化；
- all-options divisive normalization 是否解释全部选项价值和变化带来的上下文效应；
- 注视是否调节价值进入归一化和漂移过程的权重。

## 任务设计

实验包含练习区块和正式区块。正式区块包括四类环境：

| 条件 | 操纵 | 奖励概率示例 |
| --- | --- | --- |
| `range_low` | 价值范围小 | `[0.45, 0.55, 0.65]` |
| `range_high` | 价值范围大 | `[0.25, 0.55, 0.85]` |
| `low_total_value` | 三个选项总价值较低 | `[0.20, 0.35, 0.50]` |
| `high_total_value` | 三个选项总价值较高 | `[0.55, 0.70, 0.85]` |

每个区块中，三种外观和奖励概率会随机配对。每一试次三选项的位置也会随机化，
因此模型应根据 `display_order_json` 读取当次屏幕上三选项的顺序。

## 数据字段

选择试次的关键字段：

- `selected_arm`: 被试选择的选项 ID；
- `rt`: jsPsych 记录的反应时，单位毫秒；
- `reward_value`: 当次获得的奖励值；
- `reward_magnitude`: 可获得奖励大小，用于归一化反馈；
- `display_order_json`: 当次屏幕上从左到右的选项 ID；
- `arm_reward_probabilities_json`: 当前区块真实奖励概率，仅用于诊断与模拟；
- `gaze_samples_json`: 原始 gaze 样本；
- `gaze_dwell_prop_json`: 三个选项 AOI 的注视比例；
- `gaze_first_aoi_arm`: 第一个落入选项 AOI 的 gaze 样本对应选项。

## 学习层：Rescorla-Wagner 更新

对每名被试、每个区块单独初始化三选项价值：

```text
Q_i(0) = 0.5
```

选择选项 `c` 并获得单位化反馈 `r_t` 后：

```text
Q_c(t + 1) = Q_c(t) + alpha * (r_t - Q_c(t))
Q_i(t + 1) = Q_i(t), i != c
```

其中 `alpha` 是学习率。当前实现把奖励值除以 `reward_magnitude` 得到 `r_t`，
因此反馈在 `[0, 1]` 内。

## 归一化层

设当前屏幕上的三选项价值为 `Q = [Q_1, Q_2, Q_3]`。

### 无归一化

```text
N_i = Q_i
```

### Range normalization

```text
N_i = (Q_i - min(Q)) / (max(Q) - min(Q) + sigma)
```

该形式对应价值范围改变时的动态尺度调整。若区块价值范围变大，给定绝对差异会被更强压缩；
若范围变小，相同绝对差异会被放大。

### All-options divisive normalization

```text
N_i = Q_i / (sigma + sum_j Q_j)
```

该形式直接体现其它备选项对目标选项编码的除性抑制。即使两个目标选项的差异不变，
第三个选项价值升高也会增加分母，压缩全部选项的有效漂移输入。

### 眼动加权变体

如果使用 `gaze_range` 或 `gaze_divisive`，先用当前试次的 AOI 注视比例 `G_i` 调整价值：

```text
Q'_i = Q_i * exp(gaze_weight * G_i)
```

再把 `Q'` 输入 range 或 divisive-sum 归一化。`gaze_weight > 0` 表示注视增强价值输入；
`gaze_weight < 0` 表示注视与价值输入呈负向关系。

## 决策层：多选项 race-DDM 近似

经典 DDM 常用于二选一任务。三臂老虎机可用多累积器 race-DDM 近似：

```text
drift_i = drift_scale * N_i
```

选择概率使用 drift 的 softmax 近似：

```text
P(choice = i) = exp(drift_i) / sum_j exp(drift_j)
```

反应时用序贯采样的直觉约束：被选项相对其它选项的证据优势越大，越快达到边界：

```text
evidence_i = abs(N_i - mean(N_not_i))
mean_rt_i = non_decision_time + boundary / max(drift_scale * evidence_i, epsilon)
```

观测 RT 以 `mean_rt_i` 为中心的 lognormal 分布建模。该实现是可快速拟合的
race-DDM surrogate；若需要更严格的多选项扩散似然，可在同一学习和归一化层之上替换
为 Monte Carlo first-passage 或 LBA/MLBA likelihood。

## 模型比较

默认 CLI 比较：

```text
none
range
divisive
gaze_range
gaze_divisive
```

输出包含负对数似然、AIC、BIC、优化状态和参数估计。研究解释上建议优先关注：

- `range` 是否优于 `none`，且在 `range_low`/`range_high` 中改进明显；
- `divisive` 是否优于 `none`，且在 `low_total_value`/`high_total_value` 中改进明显；
- gaze 变体是否进一步降低 AIC/BIC，并且 `gaze_weight` 的方向符合注视增强或注视竞争假设。

## 与仓库内文献的对应

- Louie、Khaw 和 Glimcher 关于价值编码除法归一化的工作支持 all-options 分母形式。
- Khaw、Glimcher 和 Louie 关于动态价值适应的工作支持历史/范围相关的归一化思路。
- Fontanesi 等 RLDDM 综述和模型说明了如何把学习价值映射到漂移扩散过程以同时解释选择和 RT。
