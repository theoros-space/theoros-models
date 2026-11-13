# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Dataset adapters (MOD-REQ-103 to MOD-REQ-106).

Every adapter is described by a card in ``datasets/`` of the repository. Two
adapters exist:

* :class:`SyntheticDataset` renders scenes on the fly (card ``synthetic``);
* :class:`TileStoreDataset` reads the tile store, the on-disk format into
  which external datasets are converted once they have a card. The format is
  documented in ``datasets/README.md``.

Test splits are reachable only with ``evaluation=True`` so that training and
model selection cannot read them by accident (MOD-REQ-105).
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from theoros_models.data.sample import DENSE_TASKS, Sample
from theoros_models.data.synthetic import render_profile_scene
from theoros_models.data.tiles import to_reflectance
from theoros_models.layers.heads import HEAD_NAMES
from theoros_models.sensors.loader import load_profile
from theoros_models.sensors.schema import SensorProfile

SPLITS = ("train", "validation", "test")
INDEX_FIELDS = ("sample_id", "split", "sensor", "tasks", "sha256")
STORE_FORMAT_VERSION = 1
_UINT16_MAX = 65535

Group = tuple[str, tuple[str, ...]]
"""Sampling group of a sample: sensor name and labelled tasks."""


class SampleDataset(Protocol):
    """What the sampler and loader need from a dataset."""

    def __len__(self) -> int:
        """Number of samples."""

    def __getitem__(self, index: int) -> Sample:
        """Load one sample."""

    def group(self, index: int) -> Group:
        """Return the sampling group of a sample without loading it."""


class SyntheticDataset:
    """Synthetic scenes cycling over a list of sensor profiles (card ``synthetic``).

    Sample ``i`` renders profile ``i mod P`` with seed ``seed * 1_000_003 + i``.

    Args:
        profiles: Sensor profiles to render.
        length: Number of samples.
        size: Tile edge in pixels.
        seed: Dataset seed.
        tasks: Tasks whose labels are emitted.
    """

    card = "synthetic"

    def __init__(
        self,
        profiles: Sequence[SensorProfile],
        length: int,
        size: int = 256,
        seed: int = 0,
        tasks: Sequence[str] = HEAD_NAMES,
    ) -> None:
        if not profiles:
            raise ValueError("at least one profile is required")
        self.profiles = tuple(profiles)
        self.length, self.size, self.seed = length, size, seed
        self.tasks = tuple(task for task in HEAD_NAMES if task in tasks)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Sample:
        if not 0 <= index < self.length:
            raise IndexError(index)
        profile = self.profiles[index % len(self.profiles)]
        return render_profile_scene(profile, self.size, self.seed * 1_000_003 + index, self.tasks)

    def group(self, index: int) -> Group:
        """Return the sensor name and labelled tasks of sample ``index``."""
        return self.profiles[index % len(self.profiles)].name, self.tasks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tile_store(
    root: str | Path,
    name: str,
    card: str,
    samples: Iterable[tuple[Sample, str]],
    profiles: Mapping[str, SensorProfile],
) -> Path:
    """Write samples into a tile store (MOD-REQ-104).

    Reflectance is stored as uint16 through each profile's encoding, so the
    store holds the same kind of values as the source products.

    Args:
        root: Directory to create; must not exist.
        name: Dataset name.
        card: Name of the dataset card, ``datasets/<card>.md``.
        samples: Pairs of sample and split name.
        profiles: Profile for every sensor named by a sample.

    Returns:
        The store root.
    """
    root = Path(root)
    root.mkdir(parents=True)
    (root / "samples").mkdir()
    (root / "profiles").mkdir()
    for sensor, profile in profiles.items():
        (root / "profiles" / f"{sensor}.yaml").write_text(
            yaml.safe_dump(profile.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
    meta = {"format": STORE_FORMAT_VERSION, "name": name, "card": card}
    (root / "dataset.yaml").write_text(yaml.safe_dump(meta), encoding="utf-8")
    rows = []
    for sample, split in samples:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        encoding = profiles[sample.sensor].encoding
        stored = np.rint((sample.image.astype(np.float64) - encoding.offset) / encoding.scale)
        arrays: dict[str, np.ndarray] = {
            "values": np.clip(stored, 0, _UINT16_MAX).astype(np.uint16)
        }
        for task in DENSE_TASKS:
            label = getattr(sample, task)
            if label is not None:
                arrays[task] = label
        if sample.vessels is not None:
            arrays["vessels"] = sample.vessels
        path = root / "samples" / f"{sample.sample_id}.npz"
        np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "split": split,
                "sensor": sample.sensor,
                "tasks": "|".join(sample.labelled_tasks()),
                "sha256": _sha256(path),
            }
        )
    with (root / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return root


class TileStoreDataset:
    """Read one split of a tile store.

    Args:
        root: Store root.
        split: ``train``, ``validation`` or ``test``.
        evaluation: Must be True to read the test split.
        verify: Check the SHA-256 of every sample file when it is loaded.
    """

    def __init__(
        self, root: str | Path, split: str, *, evaluation: bool = False, verify: bool = True
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; use one of {', '.join(SPLITS)}")
        if split == "test" and not evaluation:
            raise PermissionError(
                "the test split is for final evaluation only; pass evaluation=True"
            )
        self.root, self.split, self.verify = Path(root), split, verify
        meta = yaml.safe_load((self.root / "dataset.yaml").read_text(encoding="utf-8"))
        if meta.get("format") != STORE_FORMAT_VERSION or not meta.get("card"):
            raise ValueError(f"{self.root}: unsupported store or missing card")
        self.name: str = meta["name"]
        self.card: str = meta["card"]
        with (self.root / "index.csv").open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        self.profiles = {
            sensor: load_profile(self.root / "profiles" / f"{sensor}.yaml")
            for sensor in sorted({row["sensor"] for row in self.rows})
        }

    def __len__(self) -> int:
        return len(self.rows)

    def group(self, index: int) -> Group:
        """Return the sensor name and labelled tasks of sample ``index``."""
        row = self.rows[index]
        return row["sensor"], tuple(task for task in row["tasks"].split("|") if task)

    def __getitem__(self, index: int) -> Sample:
        row = self.rows[index]
        path = self.root / "samples" / f"{row['sample_id']}.npz"
        if self.verify and _sha256(path) != row["sha256"]:
            raise OSError(f"{path}: checksum mismatch")
        profile = self.profiles[row["sensor"]]
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files}
        return Sample(
            image=to_reflectance(arrays["values"], profile.encoding),
            bands=profile.bands,
            sensor=row["sensor"],
            sample_id=row["sample_id"],
            cloud=arrays.get("cloud"),
            land_water=arrays.get("land_water"),
            fire=arrays.get("fire"),
            vessels=arrays.get("vessels"),
        )
