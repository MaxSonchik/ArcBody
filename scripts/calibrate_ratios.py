#!/usr/bin/env python
"""Regenerate the reference ratio distribution used for prompt wording.

Descriptors like "broad shoulders" only mean something relative to a reference
population, and the reference has to be the distribution of *this estimator's
own outputs* — not of textbook anatomy. The two differ: the estimator measures
shoulder breadth a little below the acromion, and leg length from the hip joint
rather than the crotch, so anatomical percentiles would label an ordinary body
as narrow-shouldered and short-legged.

Run this after changing the estimator, or against a real cohort, and paste the
result into ``arcbody/measure/calibration.py``.

    python scripts/calibrate_ratios.py --samples 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcbody.config import PerceptionSettings, QualitySettings  # noqa: E402
from arcbody.measure import anthropometry as anth  # noqa: E402
from arcbody.measure.quality import assess  # noqa: E402
from arcbody.measure.schema import RATIO_NAMES  # noqa: E402
from arcbody.perception.classic import ClassicPerception  # noqa: E402
from arcbody.training.synthetic import (  # noqa: E402
    render,
    sample_appearance,
    sample_params,
    sample_pose,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    backend = ClassicPerception(PerceptionSettings())
    collected: dict[str, list[float]] = {name: [] for name in RATIO_NAMES}

    for index in range(args.samples):
        params = sample_params(rng)
        capture = render(
            params, sample_pose(rng, easy=True), sample_appearance(rng), rng=rng
        )
        observation = backend.analyse(capture.image).subject
        if observation is None:
            continue
        report = assess(observation, QualitySettings(), PerceptionSettings())
        if not report.passed:
            continue
        measurements = anth.estimate(
            [observation], report, stature_cm=params.stature_cm
        )
        for name, value in measurements.ratios.items():
            collected[name].append(value)
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{args.samples}", file=sys.stderr)

    table = {
        name: [
            round(float(np.percentile(values, 10)), 4),
            round(float(np.percentile(values, 50)), 4),
            round(float(np.percentile(values, 90)), 4),
        ]
        for name, values in collected.items()
        if len(values) >= 20
    }
    counts = {name: len(values) for name, values in collected.items()}
    payload = {"percentiles_10_50_90": table, "sample_counts": counts}
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
