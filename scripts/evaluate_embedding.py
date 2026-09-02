#!/usr/bin/env python
"""Score a trained checkpoint through the *service* path, not the training path.

The training loop's own validation feeds the network perfect masks straight from
the renderer. The service does not: it segments the photo first, and every
segmentation error lands on the silhouette channel the encoder leans on. So a
checkpoint's real quality is what it does after perception, on subjects it has
never seen — which is what this measures.

    python scripts/evaluate_embedding.py --weights weights/arcbody.pt

Reports verification (AUC, EER, TAR at fixed FAR) and identification (rank-1,
mAP), plus the same figures for an untrained trunk as a floor. Without that
baseline a mediocre number is impossible to interpret.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcbody.config import EmbeddingSettings, PerceptionSettings  # noqa: E402
from arcbody.embed.encoder import BodyEncoder  # noqa: E402
from arcbody.perception.classic import ClassicPerception  # noqa: E402
from arcbody.training.metrics import (  # noqa: E402
    identification_metrics,
    verification_metrics,
)
from arcbody.training.synthetic import (  # noqa: E402
    render,
    sample_appearance,
    sample_params,
    sample_pose,
)
from arcbody.types import ViewLabel


def collect(
    encoder: BodyEncoder,
    perception: ClassicPerception,
    *,
    identities: int,
    per_identity: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed a fresh cohort through segmentation, as the API would."""
    rng = np.random.default_rng(seed)
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    for identity in range(identities):
        params = sample_params(rng)
        for index in range(per_identity):
            view = ViewLabel.SIDE if index % 5 == 4 else ViewLabel.FRONT
            capture = render(
                params,
                sample_pose(rng, easy=True),
                sample_appearance(rng),
                size=(384, 512),
                view=view,
                rng=rng,
            )
            observation = perception.analyse(capture.image).subject
            if observation is None:
                continue
            embeddings.append(encoder.encode(capture.image, observation).embedding)
            labels.append(identity)
        if (identity + 1) % 20 == 0:
            print(f"  {identity + 1}/{identities} subjects", file=sys.stderr)
    return np.stack(embeddings), np.array(labels)


def score(embeddings: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    first_seen: dict[int, int] = {}
    for index, label in enumerate(labels):
        first_seen.setdefault(int(label), index)
    gallery = np.array(sorted(first_seen.values()))
    queries = np.setdiff1d(np.arange(len(labels)), gallery)
    return {
        "verification": verification_metrics(embeddings, labels).as_dict(),
        "identification": identification_metrics(
            embeddings[gallery], labels[gallery], embeddings[queries], labels[queries]
        ).as_dict(),
        "samples": int(len(labels)),
        "identities": int(len(set(labels.tolist()))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("weights/arcbody.pt"))
    parser.add_argument("--identities", type=int, default=60)
    parser.add_argument("--per-identity", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    perception = ClassicPerception(PerceptionSettings())
    report: dict[str, object] = {
        "cohort": {"identities": args.identities, "per_identity": args.per_identity},
    }

    print("embedding the cohort with an untrained trunk (the floor)...", file=sys.stderr)
    untrained = BodyEncoder(EmbeddingSettings(weights_path=None))
    report["untrained_baseline"] = score(
        *collect(
            untrained,
            perception,
            identities=args.identities,
            per_identity=args.per_identity,
            seed=args.seed,
        )
    )

    if args.weights.exists():
        print(f"embedding the cohort with {args.weights}...", file=sys.stderr)
        trained = BodyEncoder(EmbeddingSettings(weights_path=args.weights))
        report["trained"] = score(
            *collect(
                trained,
                perception,
                identities=args.identities,
                per_identity=args.per_identity,
                seed=args.seed,
            )
        )
        report["checkpoint"] = str(args.weights)
    else:
        report["trained"] = None
        report["note"] = f"no checkpoint at {args.weights}; run 'make train' first"

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
