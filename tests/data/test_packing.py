# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for sample validation and batch packing (MOD-REQ-109)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from theoros_models.data.packing import collate
from theoros_models.data.sample import IGNORE, Sample
from theoros_models.data.synthetic import render_profile_scene
from theoros_models.data.tiles import normalise
from theoros_models.sensors import load_catalogue_profile


def _samples() -> list[Sample]:
    full = render_profile_scene(load_catalogue_profile("sentinel2_msi"), 32, 1)
    partial = render_profile_scene(load_catalogue_profile("test_5band"), 32, 2, tasks=("cloud",))
    vessels = np.array([[4.0, 5.0, 60.0, 9.0], [10.0, 11.0, 30.0, 5.0]], dtype=np.float32)
    return [dataclasses.replace(full, vessels=vessels), partial]


def test_collate_pads_bands_and_labels() -> None:
    samples = _samples()
    batch = collate(samples)
    assert batch.x.shape == (2, 13, 32, 32)
    assert batch.bands.mask.sum(dim=1).tolist() == [13, 5]
    torch.testing.assert_close(batch.x[1, :5], torch.from_numpy(normalise(samples[1].image)))
    assert torch.equal(batch.x[1, 5:], torch.zeros(8, 32, 32))
    assert batch.labelled.tolist() == [[True, True, True, True], [True, False, False, False]]
    assert bool((batch.land_water[1] == IGNORE).all())
    assert batch.cloud.dtype == torch.int64
    assert batch.vessels.shape == (2, 2, 4)
    assert batch.vessel_count.tolist() == [2, 0]
    assert batch.sensors == ("sentinel2_msi", "test_5band")
    assert batch.sample_ids == ("sentinel2_msi-1", "test_5band-2")
    moved = batch.to(torch.device("cpu"))
    assert torch.equal(moved.x, batch.x)


def test_collate_rejects() -> None:
    samples = _samples()
    small = render_profile_scene(load_catalogue_profile("test_5band"), 16, 2)
    with pytest.raises(ValueError, match="same tile size"):
        collate([samples[0], small])
    with pytest.raises(ValueError, match="empty"):
        collate([])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"image": np.zeros((5, 8, 8), dtype=np.float64)}, "float32"),
        ({"image": np.zeros((5, 8, 9), dtype=np.float32)}, "square"),
        ({"image": np.zeros((4, 8, 8), dtype=np.float32)}, "4 channels for 5 bands"),
        ({"cloud": np.zeros((8, 8), dtype=np.int64)}, "cloud label must be uint8"),
        ({"cloud": np.full((8, 8), 4, dtype=np.uint8)}, "values above 3"),
        ({"fire": np.full((8, 8), 2, dtype=np.uint8)}, "values above 1"),
        ({"vessels": np.zeros((2, 3), dtype=np.float32)}, "vessels must be"),
    ],
)
def test_sample_validation(change: dict[str, object], message: str) -> None:
    base = render_profile_scene(load_catalogue_profile("test_5band"), 8, 1)
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(base, **change)  # type: ignore[arg-type]


def test_sample_ignore_values_allowed() -> None:
    base = render_profile_scene(load_catalogue_profile("test_5band"), 8, 1)
    sample = dataclasses.replace(base, fire=np.full((8, 8), IGNORE, dtype=np.uint8))
    assert sample.labelled()[3]
