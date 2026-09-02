"""Loading the encoder and turning observations into embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from arcbody.config import EmbeddingSettings
from arcbody.embed.crop import build_tensor_input
from arcbody.embed.model import ArcBodyNet
from arcbody.measure.schema import RATIO_NAMES
from arcbody.types import PersonObservation

logger = logging.getLogger(__name__)

#: Seed used when no checkpoint is available, so an untrained service is at
#: least reproducible from run to run instead of returning a different vector
#: for the same photo after every restart.
UNTRAINED_SEED = 20240917


@dataclass
class EncodedBody:
    """One subject's embedding and the ratios the network read off the crop."""

    embedding: np.ndarray
    ratios: dict[str, float]
    trained: bool

    def as_metadata(self) -> dict[str, object]:
        return {"encoder_trained": self.trained, "embedding_dim": int(self.embedding.size)}


class BodyEncoder:
    """Wraps :class:`ArcBodyNet` with checkpoint loading and batching."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self.device = torch.device(settings.device)
        self.trained = False
        self.model = self._build()

    def _build(self) -> ArcBodyNet:
        path = self.settings.weights_path
        if path is None or not Path(path).exists():
            # An untrained trunk is not useless — random convolutional features
            # are a real, if weak, baseline — but it must never be mistaken for
            # a trained one, so the fact travels in every response.
            logger.warning(
                "no encoder checkpoint at %s; running an untrained trunk. Embeddings will be "
                "reproducible but far weaker than a fine-tuned model. Train one with "
                "'arcbody-train'.",
                path,
            )
            torch.manual_seed(UNTRAINED_SEED)
            model = ArcBodyNet(self.settings.dim)
        else:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            state = payload.get("model", payload)
            dim = int(payload.get("embedding_dim", self.settings.dim))
            model = ArcBodyNet(dim)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                logger.warning(
                    "checkpoint %s did not match the model exactly (missing=%d unexpected=%d)",
                    path,
                    len(missing),
                    len(unexpected),
                )
            self.trained = True
            logger.info("loaded encoder checkpoint from %s", path)

        model.eval()
        return model.to(self.device)

    # -- inference --------------------------------------------------------

    def encode(self, image: np.ndarray, observation: PersonObservation) -> EncodedBody:
        return self.encode_batch([(image, observation)])[0]

    def encode_batch(
        self, items: list[tuple[np.ndarray, PersonObservation]]
    ) -> list[EncodedBody]:
        """Encode several crops in one forward pass."""
        if not items:
            return []
        tensors = np.stack(
            [
                build_tensor_input(
                    image,
                    observation,
                    input_width=self.settings.input_width,
                    input_height=self.settings.input_height,
                )
                for image, observation in items
            ]
        )
        batch = torch.from_numpy(tensors).to(self.device)
        output = self.model.encode(batch)
        embeddings = output.embedding.cpu().numpy().astype(np.float32)
        ratios = output.ratios.cpu().numpy().astype(np.float32)
        return [
            EncodedBody(
                embedding=embeddings[index],
                ratios={name: float(ratios[index, i]) for i, name in enumerate(RATIO_NAMES)},
                trained=self.trained,
            )
            for index in range(len(items))
        ]


def fuse(embeddings: list[np.ndarray]) -> np.ndarray:
    """Combine several views of one person into a single unit vector.

    The mean of unit vectors, renormalised — the maximum-likelihood direction
    for observations scattered around a common one. Views that disagree pull the
    result towards the middle *and* shorten it before renormalisation, so
    :func:`view_agreement` can report that disagreement rather than hiding it.
    """
    if not embeddings:
        raise ValueError("cannot fuse an empty list of embeddings")
    stacked = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-8:
        return stacked[0]
    return (mean / norm).astype(np.float32)


def view_agreement(embeddings: list[np.ndarray]) -> float:
    """Mean resultant length of the views — 1.0 when they agree perfectly.

    A low value on photos the client says are the same person is a warning that
    one of them is a different subject, or badly segmented.
    """
    if len(embeddings) < 2:
        return 1.0
    stacked = np.stack([np.asarray(e, dtype=np.float32) for e in embeddings])
    return float(np.linalg.norm(stacked.mean(axis=0)))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine between two embeddings, safe for unnormalised input."""
    a = np.asarray(left, dtype=np.float32).ravel()
    b = np.asarray(right, dtype=np.float32).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-8:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
