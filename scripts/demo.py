#!/usr/bin/env python
"""End-to-end demonstration on a synthetic subject.

Renders one body, runs the full pipeline over a front and a side view, prints
the measurements against the ground truth the body was built from, and writes
the control maps to ``outputs/demo/``.

    python scripts/demo.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcbody import imaging  # noqa: E402
from arcbody.config import Settings  # noqa: E402
from arcbody.pipeline import ArcBodyPipeline, ImageInput  # noqa: E402
from arcbody.training.synthetic import (  # noqa: E402
    render,
    sample_appearance,
    sample_params,
    sample_pose,
)
from arcbody.types import ViewLabel  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--backend", default="classic", choices=["auto", "yolo", "classic"])
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    params = sample_params(rng)
    captures = {
        view: render(
            params, sample_pose(rng, easy=True), sample_appearance(rng), view=view, rng=rng
        )
        for view in (ViewLabel.FRONT, ViewLabel.SIDE)
    }

    settings = Settings(
        perception={"backend": args.backend},
        gallery={"database_path": args.out / "demo.sqlite3"},
    )
    pipeline = ArcBodyPipeline(settings)
    result = pipeline.analyse(
        [ImageInput(image=capture.image, view=view) for view, capture in captures.items()],
        stature_cm=params.stature_cm,
        weight_kg=74.0,
    )

    truth = params.measurements_cm()
    print(f"\nsubject: {params.stature_cm:.0f} cm, quality {result.profile.quality.score:.2f}\n")
    print(f"{'measurement':<20}{'estimate':>12}{'truth':>10}{'error':>9}   source")
    print("-" * 68)
    for name, value in sorted(result.profile.measurements.values.items()):
        expected = truth.get(name)
        error = (
            f"{abs(value.value_cm - expected) / expected * 100:>7.1f}%"
            if expected
            else "      -"
        )
        interval = f"{value.value_cm:>7.1f} cm"
        print(f"{name:<20}{interval:>12}{expected or 0:>9.1f}{error:>9}   {value.source.value}")

    print(f"\nprompt:\n  {result.prompt.prompt}")
    print(f"\nnegative:\n  {result.negative_prompt}")
    if result.warnings:
        print("\nwarnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    args.out.mkdir(parents=True, exist_ok=True)
    if result.control_maps is not None:
        for name, array in (
            ("pose", result.control_maps.pose),
            ("silhouette", result.control_maps.silhouette),
            ("normalised_crop", result.control_maps.normalised_crop),
        ):
            (args.out / f"{name}.png").write_bytes(imaging.encode_png(array))
    for view, capture in captures.items():
        (args.out / f"source_{view.value}.png").write_bytes(imaging.encode_png(capture.image))
    print(f"\nwrote control maps and source frames to {args.out}/")

    pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
