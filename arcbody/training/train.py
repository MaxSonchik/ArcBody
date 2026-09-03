"""Fine-tuning the ArcBody encoder.

The loop is deliberately ordinary — AdamW, cosine schedule, warmup — because the
interesting decisions are elsewhere:

* **Identities are split, not samples.** The validation set contains people the
  model has never seen. Splitting by sample would let it memorise a person from
  one photo and be scored on another, which is the failure mode this metric
  exists to catch.
* **The margin warms up.** Starting at the full angular margin on a randomly
  initialised trunk makes the loss nearly flat, because no sample is near its
  class centre yet. Ramping it in over the first epochs is the difference
  between converging and not.
* **The auxiliary ratio loss is masked.** Real corpora label some subjects and
  not others; unlabelled ones must contribute to the identity loss and nothing
  else.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from arcbody.embed.arcface import ArcMarginProduct
from arcbody.embed.model import ArcBodyNet
from arcbody.measure.schema import RATIO_NAMES
from arcbody.training.datasets import SyntheticBodyDataset, collate
from arcbody.training.losses import PKSampler, batch_hard_triplet_loss
from arcbody.training.metrics import (
    identification_metrics,
    ratio_error,
    verification_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Everything the run needs, in one serialisable place."""

    # data
    #
    # Identity *count* dominates, not captures per identity. A run with 160
    # people and 8 captures each memorises its training set within a couple of
    # epochs and scores chance AUC on held-out people: an angular margin can
    # only learn what separates bodies in general if it has seen enough bodies
    # to generalise over. Trading captures for identities at a constant sample
    # budget is the single highest-leverage knob here, which is why the default
    # is many people photographed a few times rather than the reverse.
    identities: int = 900
    per_identity: int = 4
    val_identities: int = 120
    val_per_identity: int = 4
    seed: int = 0
    image_size: tuple[int, int] = (256, 384)
    input_width: int = 128
    input_height: int = 256
    easy_poses: bool = False

    # model
    embedding_dim: int = 256
    sub_centers: int = 1
    arc_scale: float = 24.0
    arc_margin: float = 0.25
    dropout: float = 0.1

    # optimisation
    epochs: int = 12
    batch_size: int = 48
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    warmup_epochs: int = 1
    margin_warmup_epochs: int = 3
    ratio_loss_weight: float = 2.0
    # Identity loss and triplet loss are complementary, not alternatives: the
    # classifier organises the embedding space globally while the triplet term
    # sharpens local neighbourhoods, and with only a few captures per person the
    # triplet term is what actually makes the run converge in a sane number of
    # epochs. See arcbody.training.losses for why.
    triplet_loss_weight: float = 1.0
    triplet_margin: float = 0.0  # 0 selects the soft margin
    identities_per_batch: int = 12
    instances_per_identity: int = 4
    label_smoothing: float = 0.1
    grad_clip: float = 5.0
    workers: int = 2
    device: str = "cpu"

    # output
    output: Path = Path("weights/arcbody.pt")
    log_every: int = 20
    metrics_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output"] = str(self.output)
        payload["metrics_path"] = str(self.metrics_path) if self.metrics_path else None
        payload["image_size"] = list(self.image_size)
        return payload


@dataclass
class EpochReport:
    epoch: int
    loss: float
    identity_loss: float
    triplet_loss: float
    ratio_loss: float
    accuracy: float
    seconds: float
    validation: dict[str, Any] = field(default_factory=dict)


def build_datasets(
    config: TrainConfig,
) -> tuple[SyntheticBodyDataset, SyntheticBodyDataset]:
    """Train and validation sets over *disjoint* identity pools."""

    def build(identities: int, per_identity: int, seed: int) -> SyntheticBodyDataset:
        return SyntheticBodyDataset(
            identities=identities,
            per_identity=per_identity,
            seed=seed,
            easy=config.easy_poses,
            size=config.image_size,
            input_width=config.input_width,
            input_height=config.input_height,
        )

    train = build(config.identities, config.per_identity, config.seed)
    # A different seed draws a different pool of people, so no validation
    # subject is a training subject.
    validation = build(config.val_identities, config.val_per_identity, config.seed + 9973)
    return train, validation


def _margin_for_epoch(config: TrainConfig, epoch: int) -> float:
    """Linear ramp of the angular margin over the first epochs."""
    if config.margin_warmup_epochs <= 0:
        return config.arc_margin
    progress = min(1.0, (epoch + 1) / config.margin_warmup_epochs)
    return config.arc_margin * progress


def _learning_rate(config: TrainConfig, epoch: int, step: int, steps_per_epoch: int) -> float:
    """Linear warmup into a cosine decay."""
    total = max(1, config.epochs * steps_per_epoch)
    current = epoch * steps_per_epoch + step
    warmup = config.warmup_epochs * steps_per_epoch
    if warmup > 0 and current < warmup:
        return config.learning_rate * (current + 1) / warmup
    progress = (current - warmup) / max(1, total - warmup)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def masked_ratio_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Smooth L1 over the labelled ratios only.

    Smooth L1 rather than MSE because a mis-segmented sample produces a wildly
    wrong ratio, and squaring it would let one bad silhouette dominate a batch.
    """
    if mask.sum() < 1:
        return predicted.sum() * 0.0
    per_element = F.smooth_l1_loss(predicted, target, beta=0.02, reduction="none")
    return (per_element * mask).sum() / mask.sum().clamp(min=1.0)


@torch.no_grad()
def evaluate(model: ArcBodyNet, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Embed the validation set and score verification and identification."""
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    predicted_ratios: list[np.ndarray] = []
    target_ratios: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for batch in loader:
        output = model.encode(batch["tensor"].to(device))
        embeddings.append(output.embedding.cpu().numpy())
        predicted_ratios.append(output.ratios.cpu().numpy())
        labels.append(batch["identity"].numpy())
        target_ratios.append(batch["ratios"].numpy())
        masks.append(batch["ratio_mask"].numpy())

    all_embeddings = np.concatenate(embeddings)
    all_labels = np.concatenate(labels)

    # One capture per identity forms the gallery, the rest are queries — the
    # same shape as enrolment followed by lookups in production.
    first_seen: dict[int, int] = {}
    for index, label in enumerate(all_labels):
        first_seen.setdefault(int(label), index)
    gallery_index = np.array(sorted(first_seen.values()))
    query_index = np.setdiff1d(np.arange(len(all_labels)), gallery_index)

    report: dict[str, Any] = {
        "verification": verification_metrics(all_embeddings, all_labels).as_dict(),
        "ratio_mape": round(
            ratio_error(
                np.concatenate(predicted_ratios),
                np.concatenate(target_ratios),
                np.concatenate(masks),
            ),
            3,
        ),
    }
    if query_index.size:
        report["identification"] = identification_metrics(
            all_embeddings[gallery_index],
            all_labels[gallery_index],
            all_embeddings[query_index],
            all_labels[query_index],
        ).as_dict()
    return report


def train(config: TrainConfig) -> dict[str, Any]:
    """Run the whole fine-tune and write a checkpoint. Returns the final report."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    train_set, val_set = build_datasets(config)
    # Batches are built P identities x K instances so the triplet term always
    # has positive pairs to mine; plain shuffling over 900 identities almost
    # never puts two captures of one person in the same batch.
    sampler = PKSampler(
        [index // train_set.per_identity for index in range(len(train_set))],
        identities_per_batch=config.identities_per_batch,
        instances=config.instances_per_identity,
        seed=config.seed,
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=config.workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        collate_fn=collate,
    )

    model = ArcBodyNet(config.embedding_dim, dropout=config.dropout).to(device)
    head = ArcMarginProduct(
        config.embedding_dim,
        config.identities,
        scale=config.arc_scale,
        margin=config.arc_margin,
        sub_centers=config.sub_centers,
    ).to(device)

    optimiser = torch.optim.AdamW(
        [{"params": model.parameters()}, {"params": head.parameters()}],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    steps_per_epoch = max(1, len(train_loader))
    history: list[EpochReport] = []
    logger.info(
        "training on %d identities x %d captures (%d samples), %d parameters",
        config.identities,
        config.per_identity,
        len(train_set),
        model.num_parameters(),
    )

    for epoch in range(config.epochs):
        model.train()
        sampler.set_epoch(epoch)
        head.set_margin(_margin_for_epoch(config, epoch))

        started = time.perf_counter()
        totals = {
            "loss": 0.0,
            "identity": 0.0,
            "triplet": 0.0,
            "ratio": 0.0,
            "correct": 0.0,
            "count": 0.0,
        }

        for step, batch in enumerate(train_loader):
            learning_rate = _learning_rate(config, epoch, step, steps_per_epoch)
            for group in optimiser.param_groups:
                group["lr"] = learning_rate

            tensors = batch["tensor"].to(device)
            identities = batch["identity"].to(device)
            output = model(tensors)
            logits = head(model.classifier_input(output.embedding), identities)

            identity_loss = criterion(logits, identities)
            triplet_loss = batch_hard_triplet_loss(
                output.embedding, identities, margin=config.triplet_margin
            )
            ratio_loss = masked_ratio_loss(
                output.ratios, batch["ratios"].to(device), batch["ratio_mask"].to(device)
            )
            loss = (
                identity_loss
                + config.triplet_loss_weight * triplet_loss
                + config.ratio_loss_weight * ratio_loss
            )

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), config.grad_clip
            )
            optimiser.step()

            batch_size = identities.numel()
            totals["loss"] += loss.item() * batch_size
            totals["identity"] += identity_loss.item() * batch_size
            totals["triplet"] += triplet_loss.item() * batch_size
            totals["ratio"] += ratio_loss.item() * batch_size
            # Accuracy is read off the *plain* cosine, not the margin-penalised
            # logits. The margin lowers the target class by construction, so
            # early in training argmax over the logits never selects it and the
            # metric reads a flat 0.000 while the model is in fact learning.
            with torch.no_grad():
                cosine = head.cosine(model.classifier_input(output.embedding.detach()))
                totals["correct"] += float((cosine.argmax(1) == identities).sum().item())
            totals["count"] += batch_size

            if config.log_every and step % config.log_every == 0:
                logger.info(
                    "epoch %d step %d/%d loss %.4f (id %.4f trip %.4f ratio %.4f) "
                    "lr %.2e margin %.3f",
                    epoch,
                    step,
                    steps_per_epoch,
                    loss.item(),
                    identity_loss.item(),
                    triplet_loss.item(),
                    ratio_loss.item(),
                    learning_rate,
                    head.margin,
                )

        count = max(1.0, totals["count"])
        report = EpochReport(
            epoch=epoch,
            loss=totals["loss"] / count,
            identity_loss=totals["identity"] / count,
            triplet_loss=totals["triplet"] / count,
            ratio_loss=totals["ratio"] / count,
            accuracy=totals["correct"] / count,
            seconds=time.perf_counter() - started,
            validation=evaluate(model, val_loader, device),
        )
        history.append(report)
        verification = report.validation.get("verification", {})
        logger.info(
            "epoch %d done in %.1fs loss %.4f train-acc %.3f | held-out AUC %.4f EER %.4f "
            "rank1 %.3f ratio-MAPE %.2f%%",
            epoch,
            report.seconds,
            report.loss,
            report.accuracy,
            verification.get("auc", float("nan")),
            verification.get("eer", float("nan")),
            report.validation.get("identification", {}).get("rank1", float("nan")),
            report.validation.get("ratio_mape", float("nan")),
        )

    config.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "embedding_dim": config.embedding_dim,
            "input_width": config.input_width,
            "input_height": config.input_height,
            "ratio_names": list(RATIO_NAMES),
            "config": config.as_dict(),
            "validation": history[-1].validation if history else {},
        },
        config.output,
    )
    logger.info("wrote checkpoint to %s", config.output)

    final: dict[str, Any] = {
        "checkpoint": str(config.output),
        "epochs": [
            {
                "epoch": item.epoch,
                "loss": round(item.loss, 5),
                "accuracy": round(item.accuracy, 4),
                "seconds": round(item.seconds, 1),
                "validation": item.validation,
            }
            for item in history
        ],
        "final_validation": history[-1].validation if history else {},
        "config": config.as_dict(),
    }
    if config.metrics_path:
        config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        config.metrics_path.write_text(json.dumps(final, indent=2))
    return final


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the ArcBody encoder.")
    parser.add_argument("--identities", type=int, default=TrainConfig.identities)
    parser.add_argument("--per-identity", type=int, default=TrainConfig.per_identity)
    parser.add_argument("--val-identities", type=int, default=TrainConfig.val_identities)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--embedding-dim", type=int, default=TrainConfig.embedding_dim)
    parser.add_argument("--sub-centers", type=int, default=TrainConfig.sub_centers)
    parser.add_argument("--arc-scale", type=float, default=TrainConfig.arc_scale)
    parser.add_argument("--arc-margin", type=float, default=TrainConfig.arc_margin)
    parser.add_argument("--triplet-weight", type=float, default=TrainConfig.triplet_loss_weight)
    parser.add_argument(
        "--identities-per-batch", type=int, default=TrainConfig.identities_per_batch
    )
    parser.add_argument("--val-per-identity", type=int, default=TrainConfig.val_per_identity)
    parser.add_argument("--workers", type=int, default=TrainConfig.workers)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--easy-poses", action="store_true")
    parser.add_argument("--output", type=Path, default=TrainConfig.output)
    parser.add_argument("--metrics", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = TrainConfig(
        identities=args.identities,
        per_identity=args.per_identity,
        val_identities=args.val_identities,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        sub_centers=args.sub_centers,
        arc_scale=args.arc_scale,
        arc_margin=args.arc_margin,
        triplet_loss_weight=args.triplet_weight,
        identities_per_batch=args.identities_per_batch,
        val_per_identity=args.val_per_identity,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        easy_poses=args.easy_poses,
        output=args.output,
        metrics_path=args.metrics,
    )
    report = train(config)
    print(json.dumps(report["final_validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
