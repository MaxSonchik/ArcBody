"""Datasets for the encoder.

Two implementations sit behind one shape, and the split is the whole point:
:class:`SyntheticBodyDataset` makes the pipeline runnable and testable anywhere,
:class:`FolderBodyDataset` is where a real corpus attaches. Training code never
learns which one it got.

A sample is ``(tensor, identity, ratios, ratio_mask)``. The mask matters: real
corpora rarely carry tape-measure ground truth for every subject, so the
auxiliary ratio loss must be able to skip the ones they lack instead of
regressing towards a zero that means "unknown".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from arcbody import imaging
from arcbody.embed.crop import build_tensor_input
from arcbody.measure.schema import NUM_RATIOS, RATIO_NAMES
from arcbody.training.synthetic import (
    BodyParams,
    render,
    sample_appearance,
    sample_params,
    sample_pose,
)
from arcbody.types import BoundingBox, PersonObservation, ViewLabel


@dataclass
class BodySample:
    """One training example."""

    tensor: torch.Tensor
    identity: int
    ratios: torch.Tensor
    ratio_mask: torch.Tensor


def collate(samples: Sequence[BodySample]) -> dict[str, torch.Tensor]:
    return {
        "tensor": torch.stack([s.tensor for s in samples]),
        "identity": torch.tensor([s.identity for s in samples], dtype=torch.long),
        "ratios": torch.stack([s.ratios for s in samples]),
        "ratio_mask": torch.stack([s.ratio_mask for s in samples]),
    }


def observation_from_mask(
    mask: np.ndarray, keypoints: np.ndarray, view: ViewLabel
) -> PersonObservation:
    """Wrap a known-good mask as an observation, skipping perception."""
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    bbox = BoundingBox(
        x1=float(columns[0]),
        y1=float(rows[0]),
        x2=float(columns[-1] + 1),
        y2=float(rows[-1] + 1),
    )
    return PersonObservation(
        bbox=bbox,
        keypoints=keypoints,
        image_size=(mask.shape[1], mask.shape[0]),
        detection_score=1.0,
        mask=mask,
        view=view,
        backend="ground_truth",
    )


def perturb_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Erode or dilate slightly, imitating a segmenter's edge error.

    Training on perfect masks and serving on segmented ones is a train/serve
    skew that shows up exactly where it hurts — the silhouette edge, which is
    where every breadth is read. A pixel or two of jitter costs nothing and
    removes the gap.
    """
    amount = int(rng.integers(-2, 3))
    if amount == 0:
        return mask
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=bool)
    if amount > 0:
        return ndimage.binary_dilation(mask, structure=structure, iterations=amount)
    return ndimage.binary_erosion(mask, structure=structure, iterations=-amount)


class SyntheticBodyDataset(Dataset[BodySample]):
    """Rendered subjects with exact identity and ratio labels.

    Identities are drawn once and fixed; captures are rendered on demand from a
    per-index seed, so the dataset is reproducible without holding thousands of
    images in memory, and two runs with the same seed see the same data.
    """

    def __init__(
        self,
        *,
        identities: int,
        per_identity: int,
        seed: int = 0,
        size: tuple[int, int] = (256, 384),
        input_width: int = 128,
        input_height: int = 256,
        easy: bool = False,
        side_view_fraction: float = 0.2,
        identity_offset: int = 0,
    ) -> None:
        if identities < 1 or per_identity < 1:
            raise ValueError("identities and per_identity must both be positive")
        self.identities = identities
        self.per_identity = per_identity
        self.seed = seed
        self.size = size
        self.input_width = input_width
        self.input_height = input_height
        self.easy = easy
        self.side_view_fraction = side_view_fraction
        self.identity_offset = identity_offset

        generator = np.random.default_rng(seed)
        self.params: list[BodyParams] = [sample_params(generator) for _ in range(identities)]

    def __len__(self) -> int:
        return self.identities * self.per_identity

    def params_for(self, identity: int) -> BodyParams:
        return self.params[identity]

    def __getitem__(self, index: int) -> BodySample:
        identity = index // self.per_identity
        capture = index % self.per_identity
        rng = np.random.default_rng((self.seed, identity, capture))

        params = self.params[identity]
        view = (
            ViewLabel.SIDE
            if rng.random() < self.side_view_fraction
            else ViewLabel.FRONT
        )
        rendered = render(
            params,
            sample_pose(rng, easy=self.easy),
            sample_appearance(rng),
            size=self.size,
            view=view,
            rng=rng,
        )
        mask = perturb_mask(rendered.mask, rng)
        observation = observation_from_mask(mask, rendered.keypoints, view)
        tensor = build_tensor_input(
            rendered.image,
            observation,
            input_width=self.input_width,
            input_height=self.input_height,
        )

        ratios = params.ratios()
        return BodySample(
            tensor=torch.from_numpy(tensor),
            identity=identity + self.identity_offset,
            ratios=torch.tensor(
                [ratios[name] for name in RATIO_NAMES], dtype=torch.float32
            ),
            ratio_mask=torch.ones(NUM_RATIOS, dtype=torch.float32),
        )


class FolderBodyDataset(Dataset[BodySample]):
    """A real corpus laid out one directory per identity.

    ::

        root/
          person_0001/
            front_a.jpg
            side_a.jpg
            ratios.json      # optional
          person_0002/
            ...

    ``ratios.json`` may hold any subset of :data:`RATIO_NAMES`; whatever is
    present supervises the auxiliary head and the rest is masked out. This is
    the seam for BodyM, SURREAL, an internal capture set, or anything else —
    plug the directory in and the training loop does not change.

    Masks are not required. Where a directory holds ``<image>_mask.png`` it is
    used; otherwise the mask channel is zeroed, which the encoder is trained to
    tolerate because the synthetic set contains segmentation failures too.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        input_width: int = 128,
        input_height: int = 256,
        max_pixels: int = 40_000_000,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root {self.root} does not exist")
        self.input_width = input_width
        self.input_height = input_height
        self.max_pixels = max_pixels

        self.items: list[tuple[Path, int]] = []
        self.ratios: dict[int, dict[str, float]] = {}
        directories = sorted(p for p in self.root.iterdir() if p.is_dir())
        if not directories:
            raise ValueError(f"no identity directories under {self.root}")
        for identity, directory in enumerate(directories):
            self.ratios[identity] = self._read_ratios(directory)
            for image_path in sorted(directory.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                if image_path.stem.endswith("_mask"):
                    continue
                self.items.append((image_path, identity))
        if not self.items:
            raise ValueError(f"no images found under {self.root}")
        self.identities = len(directories)

    @staticmethod
    def _read_ratios(directory: Path) -> dict[str, float]:
        import json

        path = directory / "ratios.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        return {
            name: float(payload[name])
            for name in RATIO_NAMES
            if isinstance(payload.get(name), int | float)
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> BodySample:
        image_path, identity = self.items[index]
        image = imaging.load(image_path, max_pixels=self.max_pixels)

        mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
        if mask_path.exists():
            mask = imaging.load(mask_path, max_pixels=self.max_pixels)[:, :, 0] > 127
        else:
            mask = np.zeros(image.shape[:2], dtype=bool)

        if mask.any():
            observation = observation_from_mask(
                mask, np.zeros((17, 3), dtype=np.float32), ViewLabel.UNKNOWN
            )
        else:
            height, width = image.shape[:2]
            observation = PersonObservation(
                bbox=BoundingBox(0, 0, float(width), float(height)),
                keypoints=np.zeros((17, 3), dtype=np.float32),
                image_size=(width, height),
                backend="folder",
            )

        tensor = build_tensor_input(
            image,
            observation,
            input_width=self.input_width,
            input_height=self.input_height,
        )
        known = self.ratios.get(identity, {})
        values = torch.tensor(
            [known.get(name, 0.0) for name in RATIO_NAMES], dtype=torch.float32
        )
        mask_vector = torch.tensor(
            [1.0 if name in known else 0.0 for name in RATIO_NAMES], dtype=torch.float32
        )
        return BodySample(
            tensor=torch.from_numpy(tensor),
            identity=identity,
            ratios=values,
            ratio_mask=mask_vector,
        )
