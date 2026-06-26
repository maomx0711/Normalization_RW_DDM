"""Divisive-normalization reinforcement-learning diffusion model.

The experiment has three available actions.  This module treats the decision
stage as a multi-alternative race-DDM approximation: normalized learned values
set option-specific drift evidence, choice probabilities are computed from the
drift scale, and response times are modeled as faster when the winning option
has stronger evidence over its competitors.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


EPSILON = 1e-9


@dataclass(frozen=True)
class ModelParameters:
    alpha: float
    drift_scale: float
    norm_sigma: float
    boundary: float
    non_decision_time: float
    rt_sigma: float
    gaze_weight: float = 0.0


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(p, EPSILON), 1.0 - EPSILON)
    return math.log(p / (1.0 - p))


def stable_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def lognormal_logpdf(rt: float, mean_rt: float, sigma: float) -> float:
    if rt <= 0 or mean_rt <= 0 or sigma <= 0:
        return -np.inf

    mu = math.log(mean_rt) - 0.5 * sigma**2
    z = (math.log(rt) - mu) / sigma
    return -math.log(rt * sigma * math.sqrt(2.0 * math.pi)) - 0.5 * z**2


def parse_json_cell(value, default):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def load_jspsych_data(path: str | Path, formal_only: bool = True) -> pd.DataFrame:
    """Load exported jsPsych JSON or CSV and return choice trials only."""

    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        data = pd.DataFrame(raw)
    else:
        data = pd.read_csv(path)

    if "task" not in data:
        raise ValueError("Input file does not look like this experiment's jsPsych export.")

    choices = data.loc[data["task"] == "choice"].copy()
    if formal_only and "phase" in choices:
        choices = choices.loc[choices["phase"] == "formal"].copy()

    if choices.empty:
        raise ValueError("No choice trials found in the exported data.")

    if "participant_id" not in choices:
        choices["participant_id"] = "participant"

    choices["display_order"] = choices["display_order_json"].apply(parse_json_cell, default=[])
    choices["reward_probabilities"] = choices["arm_reward_probabilities_json"].apply(
        parse_json_cell,
        default={},
    )
    choices["gaze_dwell_prop"] = choices.get("gaze_dwell_prop_json", pd.Series(index=choices.index)).apply(
        parse_json_cell,
        default={},
    )
    choices["rt_s"] = pd.to_numeric(choices["rt"], errors="coerce") / 1000.0
    choices["reward_value"] = pd.to_numeric(choices["reward_value"], errors="coerce").fillna(0.0)
    choices["reward_magnitude"] = pd.to_numeric(
        choices["reward_magnitude"],
        errors="coerce",
    ).replace(0, np.nan)
    choices["reward_unit"] = (choices["reward_value"] / choices["reward_magnitude"]).fillna(
        pd.to_numeric(choices.get("reward", 0), errors="coerce").fillna(0.0),
    )

    choices = choices.dropna(subset=["selected_arm", "rt_s"])
    choices = choices.loc[choices["rt_s"] > 0].copy()
    sort_columns = [column for column in ["participant_id", "phase", "block_index", "trial_in_block"] if column in choices]
    return choices.sort_values(sort_columns).reset_index(drop=True)


def unpack_theta(theta: np.ndarray, variant: str) -> ModelParameters:
    alpha = sigmoid(theta[0])
    drift_scale = math.exp(theta[1])
    norm_sigma = math.exp(theta[2])
    boundary = math.exp(theta[3])
    non_decision_time = 0.05 + 0.95 * sigmoid(theta[4])
    rt_sigma = 0.05 + math.exp(theta[5])
    gaze_weight = theta[6] if "gaze" in variant else 0.0
    return ModelParameters(
        alpha=alpha,
        drift_scale=drift_scale,
        norm_sigma=norm_sigma,
        boundary=boundary,
        non_decision_time=non_decision_time,
        rt_sigma=rt_sigma,
        gaze_weight=gaze_weight,
    )


def default_theta(variant: str) -> np.ndarray:
    theta = np.array(
        [
            logit(0.25),
            math.log(7.0),
            math.log(0.20),
            math.log(0.45),
            logit((0.30 - 0.05) / 0.95),
            math.log(0.25),
        ],
        dtype=float,
    )
    if "gaze" in variant:
        theta = np.concatenate([theta, np.array([0.50])])
    return theta


def theta_bounds(variant: str) -> list[tuple[float, float]]:
    bounds = [
        (-6.0, 3.0),  # alpha
        (-4.0, 5.0),  # drift scale
        (-8.0, 3.0),  # normalization constant
        (-5.0, 3.0),  # boundary
        (-6.0, 6.0),  # non-decision time
        (-5.0, 2.0),  # RT sigma
    ]
    if "gaze" in variant:
        bounds.append((-5.0, 5.0))
    return bounds


def normalize_values(
    q_values: np.ndarray,
    params: ModelParameters,
    variant: str,
    gaze_props: np.ndarray | None = None,
) -> np.ndarray:
    values = np.clip(q_values.astype(float), 0.0, None)

    if "gaze" in variant and gaze_props is not None:
        attention = np.exp(params.gaze_weight * np.nan_to_num(gaze_props, nan=0.0))
        values = values * attention

    if "range" in variant:
        value_range = np.max(values) - np.min(values)
        return (values - np.min(values)) / (value_range + params.norm_sigma + EPSILON)

    if "divisive" in variant:
        return values / (params.norm_sigma + np.sum(values) + EPSILON)

    return values


def row_log_likelihood(
    row: pd.Series,
    q_by_arm: dict[str, float],
    params: ModelParameters,
    variant: str,
    rt_weight: float,
) -> float:
    display_order = list(row["display_order"])
    if not display_order or row["selected_arm"] not in display_order:
        return 0.0

    q_values = np.array([q_by_arm.get(arm, 0.5) for arm in display_order], dtype=float)
    gaze_dict = row.get("gaze_dwell_prop", {}) or {}
    gaze_props = np.array([float(gaze_dict.get(arm, 0.0)) for arm in display_order], dtype=float)
    normalized = normalize_values(q_values, params, variant, gaze_props)

    choice_index = display_order.index(row["selected_arm"])
    choice_logits = params.drift_scale * normalized
    choice_probs = stable_softmax(choice_logits)
    choice_ll = math.log(max(choice_probs[choice_index], EPSILON))

    chosen_value = normalized[choice_index]
    other_values = np.delete(normalized, choice_index)
    evidence_advantage = abs(chosen_value - float(np.mean(other_values)))
    decision_drive = max(params.drift_scale * evidence_advantage, 0.03)
    predicted_rt = params.non_decision_time + params.boundary / decision_drive
    predicted_rt = min(max(predicted_rt, 0.08), 8.0)
    rt_ll = lognormal_logpdf(float(row["rt_s"]), predicted_rt, params.rt_sigma)

    reward = float(row["reward_unit"])
    selected_arm = row["selected_arm"]
    q_by_arm[selected_arm] = q_by_arm.get(selected_arm, 0.5) + params.alpha * (
        reward - q_by_arm.get(selected_arm, 0.5)
    )

    if not np.isfinite(rt_ll):
        rt_ll = -50.0

    return choice_ll + rt_weight * rt_ll


def negative_log_likelihood(
    theta: np.ndarray,
    choices: pd.DataFrame,
    variant: str,
    rt_weight: float = 1.0,
) -> float:
    params = unpack_theta(theta, variant)
    total_ll = 0.0

    group_columns = [column for column in ["participant_id", "phase", "block_index"] if column in choices]
    grouped: Iterable[tuple[object, pd.DataFrame]]
    grouped = choices.groupby(group_columns, sort=False) if group_columns else [(None, choices)]

    for _, group in grouped:
        arms = sorted({arm for order in group["display_order"] for arm in order})
        q_by_arm = {arm: 0.5 for arm in arms}
        for _, row in group.iterrows():
            total_ll += row_log_likelihood(row, q_by_arm, params, variant, rt_weight)

    nll = -total_ll
    if not np.isfinite(nll):
        return 1e12
    return float(nll)


def fit_model(
    choices: pd.DataFrame,
    variant: str = "divisive",
    rt_weight: float = 1.0,
    n_starts: int = 12,
    seed: int = 13,
) -> dict:
    """Fit one model variant with transformed parameters."""

    rng = np.random.default_rng(seed)
    base_theta = default_theta(variant)
    bounds = theta_bounds(variant)
    best = None

    for start in range(n_starts):
        if start == 0:
            theta0 = base_theta.copy()
        else:
            theta0 = base_theta + rng.normal(0.0, 0.75, size=base_theta.shape)
            theta0 = np.array(
                [min(max(value, low + 1e-4), high - 1e-4) for value, (low, high) in zip(theta0, bounds)],
            )

        result = minimize(
            negative_log_likelihood,
            theta0,
            args=(choices, variant, rt_weight),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )
        if best is None or result.fun < best.fun:
            best = result

    if best is None:
        raise RuntimeError("No optimization result was produced.")

    params = unpack_theta(best.x, variant)
    n_trials = int(len(choices))
    n_params = len(best.x)
    nll = float(best.fun)

    return {
        "variant": variant,
        "n_trials": n_trials,
        "n_parameters": n_params,
        "negative_log_likelihood": nll,
        "aic": 2 * n_params + 2 * nll,
        "bic": n_params * math.log(max(n_trials, 1)) + 2 * nll,
        "success": bool(best.success),
        "optimizer_message": str(best.message),
        "parameters": asdict(params),
        "theta": best.x.tolist(),
    }


def compare_models(
    choices: pd.DataFrame,
    variants: Iterable[str],
    rt_weight: float = 1.0,
    n_starts: int = 12,
    seed: int = 13,
) -> pd.DataFrame:
    results = [
        fit_model(choices, variant=variant, rt_weight=rt_weight, n_starts=n_starts, seed=seed)
        for variant in variants
    ]
    table = pd.DataFrame(results)
    parameter_table = pd.json_normalize(table.pop("parameters")).add_prefix("param_")
    return pd.concat([table, parameter_table], axis=1).sort_values("aic").reset_index(drop=True)
