"""Evaluation metrics for a body-identity embedding.

The two numbers that matter to the service map directly onto its two endpoints.
Verification — "is this generated image still the same body?" — is a threshold
question, so it is scored by ROC AUC and equal error rate, plus the true-accept
rate at a fixed low false-accept rate, which is the operating point anyone
actually deploys at. Identification — "which enrolled person is this?" — is a
ranking question, so rank-1 and mAP.

All of it is computed on *held-out identities*. An embedding evaluated on
identities it was trained to classify measures memorisation, not generalisation,
and would flatter the model exactly where the service needs the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VerificationMetrics:
    auc: float
    eer: float
    eer_threshold: float
    tar_at_far: dict[str, float] = field(default_factory=dict)
    positive_pairs: int = 0
    negative_pairs: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "auc": round(self.auc, 4),
            "eer": round(self.eer, 4),
            "eer_threshold": round(self.eer_threshold, 4),
            "tar_at_far": {k: round(v, 4) for k, v in self.tar_at_far.items()},
            "positive_pairs": self.positive_pairs,
            "negative_pairs": self.negative_pairs,
        }


@dataclass
class IdentificationMetrics:
    rank1: float
    rank5: float
    mean_average_precision: float
    queries: int

    def as_dict(self) -> dict[str, object]:
        return {
            "rank1": round(self.rank1, 4),
            "rank5": round(self.rank5, 4),
            "mAP": round(self.mean_average_precision, 4),
            "queries": self.queries,
        }


def _pair_scores(
    embeddings: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of every distinct pair, and whether it is a match."""
    normalised = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8
    )
    similarity = normalised @ normalised.T
    upper = np.triu_indices(len(labels), k=1)
    scores = similarity[upper]
    same = labels[upper[0]] == labels[upper[1]]
    return scores.astype(np.float64), same


def verification_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    far_targets: tuple[float, ...] = (0.01, 0.001),
) -> VerificationMetrics:
    """Score every pair and summarise the separation between same and different."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels)
    if len(labels) < 3:
        raise ValueError("verification needs at least three samples")

    scores, same = _pair_scores(embeddings, labels)
    positives = scores[same]
    negatives = scores[~same]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("need both matching and non-matching pairs")

    # AUC as the probability a random positive outscores a random negative,
    # computed by rank sum so it costs one sort rather than a sweep.
    order = np.argsort(np.concatenate([positives, negatives]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    positive_rank_sum = ranks[: positives.size].sum()
    auc = (positive_rank_sum - positives.size * (positives.size + 1) / 2.0) / (
        positives.size * negatives.size
    )

    thresholds = np.unique(np.concatenate([positives, negatives]))
    # Sweep from the strictest threshold down; TAR and FAR are both monotone.
    tar = np.array([(positives >= t).mean() for t in thresholds])
    far = np.array([(negatives >= t).mean() for t in thresholds])
    frr = 1.0 - tar
    crossing = int(np.argmin(np.abs(far - frr)))
    eer = float((far[crossing] + frr[crossing]) / 2.0)

    tar_at_far: dict[str, float] = {}
    for target in far_targets:
        allowed = np.flatnonzero(far <= target)
        tar_at_far[f"far_{target:g}"] = float(tar[allowed].max()) if allowed.size else 0.0

    return VerificationMetrics(
        auc=float(auc),
        eer=eer,
        eer_threshold=float(thresholds[crossing]),
        tar_at_far=tar_at_far,
        positive_pairs=int(positives.size),
        negative_pairs=int(negatives.size),
    )


def identification_metrics(
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    queries: np.ndarray,
    query_labels: np.ndarray,
) -> IdentificationMetrics:
    """Rank each query against the gallery and summarise the ranking."""
    gallery = np.asarray(gallery, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    gallery_labels = np.asarray(gallery_labels)
    query_labels = np.asarray(query_labels)
    if gallery.size == 0 or queries.size == 0:
        raise ValueError("identification needs a non-empty gallery and query set")

    gallery_n = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-8)
    query_n = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)
    similarity = query_n @ gallery_n.T
    order = np.argsort(-similarity, axis=1)
    ranked_labels = gallery_labels[order]
    hits = ranked_labels == query_labels[:, None]

    rank1 = float(hits[:, 0].mean())
    rank5 = float(hits[:, : min(5, hits.shape[1])].any(axis=1).mean())

    # Average precision per query, averaged. Queries whose identity is absent
    # from the gallery contribute 0, which is the honest score for them.
    average_precisions = []
    for row in hits:
        relevant = int(row.sum())
        if relevant == 0:
            average_precisions.append(0.0)
            continue
        positions = np.flatnonzero(row) + 1
        precision = np.arange(1, relevant + 1) / positions
        average_precisions.append(float(precision.mean()))

    return IdentificationMetrics(
        rank1=rank1,
        rank5=rank5,
        mean_average_precision=float(np.mean(average_precisions)),
        queries=int(len(query_labels)),
    )


def ratio_error(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute percentage error of the auxiliary ratio head."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & (np.abs(target) > 1e-6)
    if not mask.any():
        return float("nan")
    return float(np.abs((predicted[mask] - target[mask]) / target[mask]).mean() * 100.0)
