/* global initJsPsych, jsPsychHtmlButtonResponse, jsPsychHtmlKeyboardResponse, jsPsychSurveyHtmlForm, jsPsychCallFunction, webgazer */

const EXPERIMENT_VERSION = "0.1.0";
const PRACTICE_TRIALS = 6;
const TRIALS_PER_BLOCK = 30;
const REWARD_MAGNITUDE = 10;

const ARM_APPEARANCE = [
  { id: "A", label: "蓝色星球", symbol: "A", color: "#2563eb" },
  { id: "B", label: "绿色星球", symbol: "B", color: "#16a34a" },
  { id: "C", label: "紫色星球", symbol: "C", color: "#7c3aed" },
];

const BLOCK_TEMPLATES = [
  {
    condition: "range_low",
    normalization_probe: "range",
    description: "小价值范围：三个选项较接近",
    probabilities: [0.45, 0.55, 0.65],
  },
  {
    condition: "range_high",
    normalization_probe: "range",
    description: "大价值范围：三个选项差异较大",
    probabilities: [0.25, 0.55, 0.85],
  },
  {
    condition: "low_total_value",
    normalization_probe: "divisive_sum",
    description: "总价值较低：全部选项价值和较小",
    probabilities: [0.20, 0.35, 0.50],
  },
  {
    condition: "high_total_value",
    normalization_probe: "divisive_sum",
    description: "总价值较高：全部选项价值和较大",
    probabilities: [0.55, 0.70, 0.85],
  },
];

const state = {
  participantId: null,
  eyeTracking: {
    enabled: false,
    started: false,
    currentTrial: null,
    currentCheck: null,
    error: null,
  },
};

const jsPsych = initJsPsych({
  show_progress_bar: true,
  auto_update_progress_bar: true,
  on_finish: () => {
    stopWebGazer();
  },
});

function safeJson(value) {
  return JSON.stringify(value);
}

function roundNumber(value, digits = 2) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function makeParticipantId() {
  return `P${jsPsych.randomization.randomID(8)}`;
}

function makeArmMap(arms) {
  return Object.fromEntries(arms.map((arm) => [arm.id, arm]));
}

function assignArms(template) {
  const shuffledAppearances = jsPsych.randomization.shuffle(ARM_APPEARANCE);
  const shuffledProbabilities = jsPsych.randomization.shuffle(template.probabilities);
  return shuffledAppearances.map((appearance, index) => ({
    ...appearance,
    reward_probability: shuffledProbabilities[index],
    reward_magnitude: REWARD_MAGNITUDE,
    expected_value: shuffledProbabilities[index] * REWARD_MAGNITUDE,
  }));
}

function renderArmChoice(arm, displayIndex) {
  return `
    <div class="bandit-card">
      <div class="bandit-symbol" style="background:${arm.color}">${arm.symbol}</div>
      <div class="bandit-label">${arm.label}</div>
      <div class="bandit-key">选项 ${displayIndex + 1}</div>
    </div>
  `;
}

function makeButtonHtml(displayOrder) {
  return displayOrder.map(
    (arm, index) =>
      `<button class="jspsych-btn bandit-choice" data-choice-index="${index}" data-arm-id="${arm.id}">%choice%</button>`,
  );
}

function makeChoiceStimulus(block, trialInBlock, isPractice) {
  const phase = isPractice ? "练习" : "正式";
  return `
    <div class="bandit-stimulus">
      <div class="bandit-status">
        <span class="pill">${phase}试次</span>
        <span class="pill">区块 ${block.block_index + 1}</span>
        <span class="pill">试次 ${trialInBlock + 1}</span>
      </div>
      <p class="bandit-prompt">请选择一个选项。每个选项都有稳定但未知的奖励概率。</p>
    </div>
  `;
}

function summarizeGaze(samples, rects, selectedArm) {
  const dwellByArm = Object.fromEntries(rects.map((rect) => [rect.arm_id, 0]));
  let firstAoiArm = null;
  let validSamples = 0;

  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index];
    if (!Number.isFinite(sample.x) || !Number.isFinite(sample.y)) {
      continue;
    }

    validSamples += 1;
    const hit = rects.find(
      (rect) =>
        sample.x >= rect.left &&
        sample.x <= rect.right &&
        sample.y >= rect.top &&
        sample.y <= rect.bottom,
    );

    if (!hit) {
      continue;
    }

    if (firstAoiArm === null) {
      firstAoiArm = hit.arm_id;
    }

    const next = samples[index + 1];
    const rawDuration = next ? next.t - sample.t : 16.7;
    const duration = Math.max(0, Math.min(rawDuration, 100));
    dwellByArm[hit.arm_id] += duration;
  }

  const totalDwell = Object.values(dwellByArm).reduce((sum, duration) => sum + duration, 0);
  const dwellPropByArm = Object.fromEntries(
    Object.entries(dwellByArm).map(([arm, duration]) => [
      arm,
      totalDwell > 0 ? duration / totalDwell : 0,
    ]),
  );

  return {
    gaze_valid_samples: validSamples,
    gaze_total_aoi_ms: roundNumber(totalDwell),
    gaze_first_aoi_arm: firstAoiArm,
    gaze_dwell_ms_json: safeJson(
      Object.fromEntries(
        Object.entries(dwellByArm).map(([arm, duration]) => [arm, roundNumber(duration)]),
      ),
    ),
    gaze_dwell_prop_json: safeJson(
      Object.fromEntries(
        Object.entries(dwellPropByArm).map(([arm, proportion]) => [arm, roundNumber(proportion, 4)]),
      ),
    ),
    gaze_selected_dwell_prop: selectedArm ? roundNumber(dwellPropByArm[selectedArm] || 0, 4) : null,
  };
}

function startGazeTrial(meta) {
  const buttons = Array.from(document.querySelectorAll(".bandit-choice"));
  const rects = buttons.map((button) => {
    const rect = button.getBoundingClientRect();
    return {
      arm_id: button.dataset.armId,
      choice_index: Number(button.dataset.choiceIndex),
      left: rect.left,
      right: rect.right,
      top: rect.top,
      bottom: rect.bottom,
      width: rect.width,
      height: rect.height,
    };
  });

  state.eyeTracking.currentTrial = {
    meta,
    rects,
    samples: [],
    started_at: performance.now(),
  };
}

function stopGazeTrial(data) {
  const current = state.eyeTracking.currentTrial;
  state.eyeTracking.currentTrial = null;

  if (!current) {
    data.gaze_enabled = state.eyeTracking.enabled;
    data.gaze_n_samples = 0;
    return;
  }

  const compactSamples = current.samples.map((sample) => ({
    x: roundNumber(sample.x),
    y: roundNumber(sample.y),
    t: roundNumber(sample.t - current.started_at),
  }));
  const summary = summarizeGaze(current.samples, current.rects, data.selected_arm);

  data.gaze_enabled = state.eyeTracking.enabled;
  data.gaze_error = state.eyeTracking.error;
  data.gaze_n_samples = compactSamples.length;
  data.gaze_aoi_rects_json = safeJson(current.rects.map((rect) => ({ ...rect })));
  data.gaze_samples_json = safeJson(compactSamples);
  Object.assign(data, summary);
}

function startValidationCheck(targetName, targetX, targetY) {
  state.eyeTracking.currentCheck = {
    targetName,
    targetX,
    targetY,
    samples: [],
    started_at: performance.now(),
  };
}

function stopValidationCheck(data) {
  const check = state.eyeTracking.currentCheck;
  state.eyeTracking.currentCheck = null;

  if (!check || check.samples.length === 0) {
    data.gaze_check_samples = 0;
    data.gaze_check_mean_error_px = null;
    return;
  }

  const errors = check.samples
    .filter((sample) => Number.isFinite(sample.x) && Number.isFinite(sample.y))
    .map((sample) => Math.hypot(sample.x - check.targetX, sample.y - check.targetY));
  const meanError = errors.reduce((sum, error) => sum + error, 0) / Math.max(errors.length, 1);

  data.gaze_check_target = check.targetName;
  data.gaze_check_samples = errors.length;
  data.gaze_check_mean_error_px = roundNumber(meanError);
}

async function startWebGazer() {
  if (!window.webgazer) {
    state.eyeTracking.error = "webgazer_not_loaded";
    return;
  }

  try {
    webgazer
      .setGazeListener((gazeData) => {
        if (!gazeData) {
          return;
        }

        const sample = {
          x: gazeData.x,
          y: gazeData.y,
          t: performance.now(),
        };

        if (state.eyeTracking.currentTrial) {
          state.eyeTracking.currentTrial.samples.push(sample);
        }
        if (state.eyeTracking.currentCheck) {
          state.eyeTracking.currentCheck.samples.push(sample);
        }
      })
      .showPredictionPoints(true)
      .showVideoPreview(true)
      .showFaceOverlay(false)
      .showFaceFeedbackBox(false);

    if (typeof webgazer.setRegression === "function") {
      webgazer.setRegression("ridge");
    }

    await webgazer.begin();
    state.eyeTracking.enabled = true;
    state.eyeTracking.started = true;
    state.eyeTracking.error = null;
  } catch (error) {
    state.eyeTracking.enabled = false;
    state.eyeTracking.error = error && error.message ? error.message : "webgazer_start_failed";
  }
}

function stopWebGazer() {
  if (window.webgazer && state.eyeTracking.started) {
    try {
      webgazer.end();
    } catch (error) {
      state.eyeTracking.error = error && error.message ? error.message : "webgazer_stop_failed";
    }
  }
}

function makeCalibrationTrial(point, repeatIndex) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="calibration-wrap">
        <p class="small center">眼动校准：请看着蓝点并点击它（${repeatIndex + 1}/3）。</p>
      </div>
    `,
    choices: ["+"],
    button_html: [
      `<button class="jspsych-btn calibration-dot" style="left:${point.x}vw; top:${point.y}vh">%choice%</button>`,
    ],
    data: {
      task: "gaze_calibration",
      calibration_x_vw: point.x,
      calibration_y_vh: point.y,
      calibration_repeat: repeatIndex + 1,
    },
    on_load: () => {
      const dot = document.querySelector(".calibration-dot");
      if (!dot || !window.webgazer || typeof webgazer.recordScreenPosition !== "function") {
        return;
      }
      dot.addEventListener("mousedown", (event) => {
        webgazer.recordScreenPosition(event.clientX, event.clientY, "click");
      });
    },
  };
}

function makeChoiceTrial(block, trialInBlock, isPractice = false) {
  const displayOrder = jsPsych.randomization.shuffle(block.arms);
  const armMap = makeArmMap(block.arms);

  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: makeChoiceStimulus(block, trialInBlock, isPractice),
    choices: displayOrder.map((arm, index) => renderArmChoice(arm, index)),
    button_html: makeButtonHtml(displayOrder),
    margin_horizontal: "10px",
    data: {
      task: "choice",
      phase: isPractice ? "practice" : "formal",
      experiment_version: EXPERIMENT_VERSION,
      block_index: block.block_index,
      condition: block.condition,
      normalization_probe: block.normalization_probe,
      block_description: block.description,
      trial_in_block: trialInBlock + 1,
      display_order_json: safeJson(displayOrder.map((arm) => arm.id)),
      arm_reward_probabilities_json: safeJson(
        Object.fromEntries(block.arms.map((arm) => [arm.id, arm.reward_probability])),
      ),
      arm_expected_values_json: safeJson(
        Object.fromEntries(block.arms.map((arm) => [arm.id, arm.expected_value])),
      ),
      reward_magnitude: REWARD_MAGNITUDE,
    },
    on_load: () => {
      startGazeTrial({
        block_index: block.block_index,
        trial_in_block: trialInBlock + 1,
        condition: block.condition,
        display_order: displayOrder.map((arm) => arm.id),
      });
    },
    on_finish: (data) => {
      const selectedArm = displayOrder[data.response];
      const selected = armMap[selectedArm.id];
      const reward = Math.random() < selected.reward_probability ? REWARD_MAGNITUDE : 0;

      data.selected_arm = selectedArm.id;
      data.selected_label = selectedArm.label;
      data.selected_display_index = data.response;
      data.reward = reward > 0 ? 1 : 0;
      data.reward_value = reward;
      data.reward_probability = selected.reward_probability;
      data.is_optimal_true_ev =
        selected.expected_value === Math.max(...block.arms.map((arm) => arm.expected_value));
      stopGazeTrial(data);
    },
  };
}

function makeFeedbackTrial() {
  return {
    type: jsPsychHtmlKeyboardResponse,
    choices: "NO_KEYS",
    trial_duration: 900,
    stimulus: () => {
      const lastChoice = jsPsych.data.get().filter({ task: "choice" }).last(1).values()[0];
      const rewardClass = lastChoice.reward > 0 ? "reward" : "no-reward";
      const rewardText = lastChoice.reward > 0 ? `+${lastChoice.reward_value}` : "+0";
      return `
        <div class="feedback ${rewardClass}">
          <p>你选择了：${lastChoice.selected_label}</p>
          <div class="amount">${rewardText}</div>
          <p class="small">下一试次即将开始</p>
        </div>
      `;
    },
    data: {
      task: "feedback",
    },
  };
}

function makeDownloadHtml() {
  const data = jsPsych.data.get();
  const filenameStem = `${state.participantId || "participant"}_${new Date()
    .toISOString()
    .replace(/[:.]/g, "-")}`;
  const jsonBlob = new Blob([data.json()], { type: "application/json" });
  const csvBlob = new Blob([data.csv()], { type: "text/csv" });
  const jsonUrl = URL.createObjectURL(jsonBlob);
  const csvUrl = URL.createObjectURL(csvBlob);

  return `
    <div class="screen center">
      <h2>实验结束，感谢参与！</h2>
      <p>请下载数据文件并按研究流程上传或保存。JSON 包含逐试次眼动样本，CSV 便于快速检查。</p>
      <div class="download-links">
        <a class="download-link" href="${jsonUrl}" download="${filenameStem}.json">下载 JSON</a>
        <a class="download-link" href="${csvUrl}" download="${filenameStem}.csv">下载 CSV</a>
      </div>
      <p class="small">如果浏览器阻止下载，请右键链接另存为。</p>
    </div>
  `;
}

function buildTimeline() {
  const timeline = [];

  timeline.push({
    type: jsPsychSurveyHtmlForm,
    preamble: `
      <div class="screen">
        <h1>三臂老虎机与除法归一化决策实验</h1>
        <p>
          本任务会记录你的选择、反应时以及浏览器摄像头估计的眼动数据。
          眼动数据仅用于计算选项注视指标，例如首次注视和各选项停留时间。
        </p>
      </div>
    `,
    html: `
      <p><label>被试编号（可留空自动生成）：
        <input name="participant_id" type="text" autocomplete="off" />
      </label></p>
      <p><label>
        <input name="consent" type="checkbox" required />
        我已阅读说明，并同意在本浏览器中采集任务数据和摄像头眼动估计。
      </label></p>
    `,
    button_label: "继续",
    data: {
      task: "metadata",
      experiment_version: EXPERIMENT_VERSION,
    },
    on_finish: (data) => {
      const response = data.response || {};
      state.participantId = response.participant_id
        ? String(response.participant_id).trim()
        : makeParticipantId();
      jsPsych.data.addProperties({
        participant_id: state.participantId,
        experiment_version: EXPERIMENT_VERSION,
        user_agent: navigator.userAgent,
        screen_width: window.screen.width,
        screen_height: window.screen.height,
      });
    },
  });

  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="screen">
        <h2>任务说明</h2>
        <ul>
          <li>每一试次会出现三个选项。请选择你认为最可能获得奖励的选项。</li>
          <li>选项的奖励概率需要通过反馈学习，且不同区块的价值范围或总价值会变化。</li>
          <li>请尽量自然注视和选择，不需要刻意盯住某个位置。</li>
        </ul>
        <p>点击继续后，浏览器会请求摄像头权限用于眼动估计。</p>
      </div>
    `,
    choices: ["启动眼动并继续"],
    data: { task: "instruction" },
  });

  timeline.push({
    type: jsPsychCallFunction,
    async: true,
    func: (done) => {
      startWebGazer().finally(done);
    },
    data: { task: "webgazer_start" },
  });

  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="screen center">
        <h2>眼动校准</h2>
        <p>接下来请依次看着屏幕上的蓝点并点击它。若摄像头不可用，实验仍会记录选择和反应时。</p>
      </div>
    `,
    choices: ["开始校准"],
    data: { task: "gaze_calibration_instruction" },
  });

  const calibrationPoints = [
    { x: 12, y: 16 },
    { x: 50, y: 16 },
    { x: 88, y: 16 },
    { x: 12, y: 50 },
    { x: 50, y: 50 },
    { x: 88, y: 50 },
    { x: 12, y: 84 },
    { x: 50, y: 84 },
    { x: 88, y: 84 },
  ];
  calibrationPoints.forEach((point) => {
    for (let repeatIndex = 0; repeatIndex < 3; repeatIndex += 1) {
      timeline.push(makeCalibrationTrial(point, repeatIndex));
    }
  });

  timeline.push({
    type: jsPsychHtmlKeyboardResponse,
    choices: "NO_KEYS",
    trial_duration: 1800,
    stimulus: `
      <div class="screen center">
        <h2>眼动检查</h2>
        <p>请注视屏幕中央的十字。</p>
        <div style="font-size:54px; font-weight:800;">+</div>
      </div>
    `,
    data: { task: "gaze_validation" },
    on_load: () => {
      startValidationCheck("center", window.innerWidth / 2, window.innerHeight / 2);
    },
    on_finish: (data) => {
      stopValidationCheck(data);
    },
  });

  const practiceBlock = {
    block_index: 0,
    condition: "practice",
    normalization_probe: "practice",
    description: "练习区块",
    arms: assignArms({
      probabilities: [0.30, 0.50, 0.70],
    }),
  };

  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="screen center">
        <h2>练习开始</h2>
        <p>先完成 ${PRACTICE_TRIALS} 个练习试次，熟悉选择和反馈。</p>
      </div>
    `,
    choices: ["开始练习"],
    data: { task: "practice_instruction" },
  });

  for (let trial = 0; trial < PRACTICE_TRIALS; trial += 1) {
    timeline.push(makeChoiceTrial(practiceBlock, trial, true));
    timeline.push(makeFeedbackTrial());
  }

  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div class="screen center">
        <h2>正式任务</h2>
        <p>正式任务包含 ${BLOCK_TEMPLATES.length} 个区块，每个区块 ${TRIALS_PER_BLOCK} 个试次。</p>
        <p>区块之间选项的奖励环境会变化，请继续根据反馈学习。</p>
      </div>
    `,
    choices: ["开始正式任务"],
    data: { task: "formal_instruction" },
  });

  const blocks = jsPsych.randomization.shuffle(BLOCK_TEMPLATES).map((template, index) => ({
    ...template,
    block_index: index + 1,
    arms: assignArms(template),
  }));

  blocks.forEach((block) => {
    timeline.push({
      type: jsPsychHtmlButtonResponse,
      stimulus: `
        <div class="screen center">
          <h2>新区块</h2>
          <p>${block.description}</p>
          <p class="small">选项外观会保留，但奖励概率已经重新分配。</p>
        </div>
      `,
      choices: ["继续"],
      data: {
        task: "block_instruction",
        block_index: block.block_index,
        condition: block.condition,
        normalization_probe: block.normalization_probe,
      },
    });

    for (let trial = 0; trial < TRIALS_PER_BLOCK; trial += 1) {
      timeline.push(makeChoiceTrial(block, trial, false));
      timeline.push(makeFeedbackTrial());
    }
  });

  timeline.push({
    type: jsPsychHtmlButtonResponse,
    stimulus: makeDownloadHtml,
    choices: ["完成"],
    data: { task: "finish" },
  });

  return timeline;
}

jsPsych.run(buildTimeline());
