#!/usr/bin/env python
"""Measure gallery enrolment and search at scale.

The architecture notes claimed brute-force cosine search would hold to ~100k
people "by construction". That is an argument, not a measurement, and the two
disagree often enough that the claim needed a number behind it.

    python scripts/benchmark_gallery.py --people 100000

Reports enrolment throughput, index build time, and query latency percentiles.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcbody.config import GallerySettings  # noqa: E402
from arcbody.gallery.store import Gallery  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=int, default=100_000)
    parser.add_argument("--profiles-each", type=int, default=1)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as directory:
        gallery = Gallery(GallerySettings(database_path=Path(directory) / "bench.sqlite3"))

        vectors = rng.normal(size=(args.people, args.dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        started = time.perf_counter()
        for index in range(args.people):
            for profile in range(args.profiles_each):
                noise = rng.normal(0, 0.05, args.dim).astype(np.float32)
                vector = vectors[index] + (noise if profile else 0.0)
                gallery.enrol(f"p{index:07d}", vector / np.linalg.norm(vector))
            if (index + 1) % 10_000 == 0:
                print(f"  enrolled {index + 1}/{args.people}", file=sys.stderr)
        enrol_seconds = time.perf_counter() - started

        # First query pays for the index build; the rest do not.
        started = time.perf_counter()
        gallery.identify(vectors[0], top_k=5)
        first_query_seconds = time.perf_counter() - started

        latencies = []
        for index in rng.choice(args.people, size=args.queries, replace=False):
            probe = vectors[int(index)] + rng.normal(0, 0.05, args.dim).astype(np.float32)
            probe /= np.linalg.norm(probe)
            started = time.perf_counter()
            matches = gallery.identify(probe, top_k=5)
            latencies.append((time.perf_counter() - started) * 1000.0)
            assert matches and matches[0].person_id == f"p{int(index):07d}"

        # The pattern that actually hurts: a write between every read, which
        # invalidates the index and makes each query pay a full rebuild.
        interleaved = []
        for step in range(min(20, args.queries)):
            gallery.enrol(f"late{step:04d}", vectors[step])
            started = time.perf_counter()
            gallery.identify(vectors[step], top_k=5)
            interleaved.append((time.perf_counter() - started) * 1000.0)

        people, profiles = gallery.count()
        report = {
            "people": people,
            "profiles": profiles,
            "dim": args.dim,
            "enrol_seconds": round(enrol_seconds, 2),
            "enrol_per_second": round(profiles / max(enrol_seconds, 1e-9), 1),
            "first_query_seconds": round(first_query_seconds, 3),
            "query_ms": {
                "median": round(statistics.median(latencies), 3),
                "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
                "max": round(max(latencies), 3),
            },
            "query_after_write_ms": {
                "median": round(statistics.median(interleaved), 3),
                "max": round(max(interleaved), 3),
            },
            "index_bytes": people * args.dim * 4,
        }
        gallery.close()

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
