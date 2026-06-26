#!/usr/bin/env python3
"""Fit DN-RL-DDM variants to exported jsPsych data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dn_rlddm import compare_models, fit_model, load_jspsych_data


DEFAULT_VARIANTS = ["none", "range", "divisive", "gaze_range", "gaze_divisive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit reinforcement-learning divisive-normalization DDM variants.",
    )
    parser.add_argument("data", type=Path, help="jsPsych JSON or CSV export from the experiment")
    parser.add_argument(
        "--variant",
        choices=DEFAULT_VARIANTS,
        help="Fit a single variant instead of comparing all default variants.",
    )
    parser.add_argument(
        "--include-practice",
        action="store_true",
        help="Include practice trials. By default only formal trials are fitted.",
    )
    parser.add_argument(
        "--rt-weight",
        type=float,
        default=1.0,
        help="Weight of the RT log likelihood relative to the choice likelihood.",
    )
    parser.add_argument(
        "--n-starts",
        type=int,
        default=12,
        help="Number of optimizer restarts per model.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for optimizer starts.")
    parser.add_argument("--output", type=Path, help="Optional path for JSON or CSV fit results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    choices = load_jspsych_data(args.data, formal_only=not args.include_practice)

    if args.variant:
      result = fit_model(
          choices,
          variant=args.variant,
          rt_weight=args.rt_weight,
          n_starts=args.n_starts,
          seed=args.seed,
      )
      print(json.dumps(result, ensure_ascii=False, indent=2))
      if args.output:
          args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
      return

    table = compare_models(
        choices,
        variants=DEFAULT_VARIANTS,
        rt_weight=args.rt_weight,
        n_starts=args.n_starts,
        seed=args.seed,
    )
    print(table.to_string(index=False))

    if args.output:
        if args.output.suffix.lower() == ".json":
            args.output.write_text(table.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        else:
            table.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
