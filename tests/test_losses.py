# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for the masked multi-task loss (MOD-REQ-120, MOD-REQ-121)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from theoros_models.arch import TIERS
from theoros_models.data.augment import transform_points
from theoros_models.data.packing import Batch, collate
from theoros_models.data.synthetic import render_profile_scene
from theoros_models.layers import d4
from theoros_models.layers.heads import HEAD_NAMES
from theoros_models.losses import LossConfig, multitask_loss, vessel_targets
from theoros_models.model import TheorosNet, TheorosOutput
from theoros_models.sensors import load_catalogue_profile


def _batch() -> Batch:
    first = render_profile_scene(load_catalogue_profile("sentinel2_msi"), 32, 1)
    vessels = np.array([[10.0, 12.0, 80.0, 12.0]], dtype=np.float32)
    first = dataclasses.replace(first, vessels=vessels)
    second = render_profile_scene(
        load_catalogue_profile("planetscope_superdove"), 32, 2, tasks=("cloud",)
    )
    return collate([first, second])


def test_vessel_targets() -> None:
    vessels = torch.tensor([[[10.0, 12.0, 80.0, 12.0], [0.0, 0.0, 1.0, 1.0]]])
    heatmap, sizes, centres = vessel_targets(vessels, torch.tensor([1]), 32, 1.0)
    assert heatmap.shape == (1, 8, 8)
    assert centres.sum() == 1
    assert bool(centres[0, 2, 3])
    assert float(heatmap[0, 2, 3]) == 1.0
    assert float(heatmap.max()) == 1.0
    torch.testing.assert_close(sizes[0, :, 2, 3], torch.log(torch.tensor([80.0, 12.0])))
    assert float(heatmap[0, 7, 7]) < 1e-6


@pytest.mark.parametrize("g", range(8))
def test_vessel_targets_equivariant(g: int) -> None:
    points = np.array([[9.3, 21.7, 80.0, 12.0], [25.0, 4.2, 40.0, 7.0]], dtype=np.float32)
    moved = transform_points(points, g, 32)
    heatmap, _, _ = vessel_targets(torch.from_numpy(points)[None], torch.tensor([2]), 32, 1.5)
    moved_map, _, _ = vessel_targets(torch.from_numpy(moved)[None], torch.tensor([2]), 32, 1.5)
    torch.testing.assert_close(moved_map, d4.transform(heatmap, g), rtol=0, atol=1e-6)


@torch.no_grad()
def test_loss_finite_and_per_head() -> None:
    torch.manual_seed(0)
    batch = _batch()
    model = TheorosNet(TIERS["nano"])
    output = model(batch.x, batch.bands)
    total, parts = multitask_loss(output, batch)
    assert set(parts) == set(HEAD_NAMES)
    assert all(bool(torch.isfinite(value)) and float(value) > 0 for value in parts.values())
    config = LossConfig(head_weights={"cloud": 2.0, "land_water": 0.0, "vessel": 1.0, "fire": 1.0})
    weighted, _ = multitask_loss(output, batch, config)
    expected = 2 * parts["cloud"] + parts["vessel"] + parts["fire"]
    torch.testing.assert_close(weighted, expected)
    torch.testing.assert_close(total, sum(parts.values()))


def test_masking() -> None:
    torch.manual_seed(0)
    batch = _batch()
    model = TheorosNet(TIERS["nano"])
    output = model(batch.x, batch.bands)
    for name in HEAD_NAMES:
        output.heads[name].retain_grad()
    total, _ = multitask_loss(output, batch)
    torch.autograd.backward(total)
    # Sample 1 labels only cloud and its sensor has no fire bands.
    for name in ("land_water", "vessel", "fire"):
        grad = output.heads[name].grad
        assert grad is not None
        assert torch.equal(grad[1], torch.zeros_like(grad[1])), name
        assert float(grad[0].abs().sum()) > 0, name
    assert float(output.heads["cloud"].grad[1].abs().sum()) > 0


@torch.no_grad()
def test_ignored_pixels_and_missing_heads() -> None:
    batch = _batch()
    torch.manual_seed(0)
    model = TheorosNet(TIERS["nano"])
    output = model(batch.x, batch.bands)
    batch = dataclasses.replace(batch, cloud=torch.full_like(batch.cloud, 255))
    _, parts = multitask_loss(output, batch)
    assert float(parts["cloud"]) == 0.0
    reduced = TheorosOutput({"cloud": output.heads["cloud"]}, output.enabled)
    _, parts = multitask_loss(reduced, batch)
    assert float(parts["fire"]) == 0.0


@torch.no_grad()
def test_class_weights() -> None:
    batch = _batch()
    torch.manual_seed(0)
    output = TheorosNet(TIERS["nano"])(batch.x, batch.bands)
    _, plain = multitask_loss(output, batch)
    _, weighted = multitask_loss(
        output, batch, LossConfig(cloud_class_weights=(1.0, 3.0, 3.0, 3.0))
    )
    assert float(weighted["cloud"]) != pytest.approx(float(plain["cloud"]))
