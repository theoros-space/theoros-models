# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Pack samples of different sensors into one batch (MOD-REQ-109).

Images are normalised and padded with zero channels to the largest band
count in the batch; the band mask tells the stem which channels are real.
Missing dense labels are filled with 255 and marked unlabelled, and vessel
annotations are padded to the largest vessel count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from theoros_models.data.sample import DENSE_TASKS, IGNORE, VESSEL_COLUMNS, Sample
from theoros_models.data.tiles import normalise
from theoros_models.model import BandBatch


@dataclass(frozen=True)
class Batch:
    """A packed batch.

    Attributes:
        x: Normalised reflectance (B, N_max, H, W) float32, zero in padded channels.
        bands: Band metadata and mask.
        cloud: (B, H, W) int64, 255 where unlabelled.
        land_water: (B, H, W) int64, 255 where unlabelled.
        fire: (B, H, W) int64, 255 where unlabelled.
        vessels: (B, K_max, 4) float32, rows beyond ``vessel_count`` are zero.
        vessel_count: (B,) int64.
        labelled: (B, 4) bool, in head order.
        sensors: Sensor name per sample.
        sample_ids: Identifier per sample.
    """

    x: Tensor
    bands: BandBatch
    cloud: Tensor
    land_water: Tensor
    fire: Tensor
    vessels: Tensor
    vessel_count: Tensor
    labelled: Tensor
    sensors: tuple[str, ...]
    sample_ids: tuple[str, ...]

    def to(self, device: torch.device) -> Batch:
        """Return the batch with every tensor on ``device``."""
        return Batch(
            self.x.to(device),
            BandBatch(self.bands.meta.to(device), self.bands.mask.to(device)),
            self.cloud.to(device),
            self.land_water.to(device),
            self.fire.to(device),
            self.vessels.to(device),
            self.vessel_count.to(device),
            self.labelled.to(device),
            self.sensors,
            self.sample_ids,
        )


def collate(samples: Sequence[Sample]) -> Batch:
    """Pack samples of equal tile size into a batch.

    Args:
        samples: One or more samples; band counts may differ.

    Returns:
        The packed batch.
    """
    if not samples:
        raise ValueError("cannot collate an empty list")
    size = samples[0].size
    if any(sample.size != size for sample in samples):
        raise ValueError("all samples in a batch need the same tile size")
    batch = len(samples)
    width = max(len(sample.bands) for sample in samples)
    x = np.zeros((batch, width, size, size), dtype=np.float32)
    meta = np.tile(np.array([1000.0, 10.0, 10.0], dtype=np.float32), (batch, width, 1))
    mask = np.zeros((batch, width), dtype=bool)
    dense = {task: np.full((batch, size, size), IGNORE, dtype=np.int64) for task in DENSE_TASKS}
    most_vessels = max(
        (len(sample.vessels) for sample in samples if sample.vessels is not None), default=0
    )
    vessels = np.zeros((batch, most_vessels, VESSEL_COLUMNS), dtype=np.float32)
    vessel_count = np.zeros(batch, dtype=np.int64)
    for row, sample in enumerate(samples):
        count = len(sample.bands)
        x[row, :count] = normalise(sample.image)
        meta[row, :count] = [(b.centre_nm, b.fwhm_nm, b.gsd_m) for b in sample.bands]
        mask[row, :count] = True
        for task in DENSE_TASKS:
            label = getattr(sample, task)
            if label is not None:
                dense[task][row] = label
        if sample.vessels is not None:
            vessels[row, : len(sample.vessels)] = sample.vessels
            vessel_count[row] = len(sample.vessels)
    return Batch(
        x=torch.from_numpy(x),
        bands=BandBatch(torch.from_numpy(meta), torch.from_numpy(mask)),
        cloud=torch.from_numpy(dense["cloud"]),
        land_water=torch.from_numpy(dense["land_water"]),
        fire=torch.from_numpy(dense["fire"]),
        vessels=torch.from_numpy(vessels),
        vessel_count=torch.from_numpy(vessel_count),
        labelled=torch.tensor([sample.labelled() for sample in samples], dtype=torch.bool),
        sensors=tuple(sample.sensor for sample in samples),
        sample_ids=tuple(sample.sample_id for sample in samples),
    )
