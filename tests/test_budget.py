# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for the analytic budget and tier definitions (MOD-REQ-001, 002, 010, 066, 067)."""

from __future__ import annotations

import itertools

import pytest
import torch

from theoros_models.arch import (
    FLIGHT_INT8_BYTES_LIMIT,
    FLIGHT_PARAMETER_LIMIT,
    TIER_TOLERANCE,
    TIERS,
    TierSpec,
    get_tier,
)
from theoros_models.budget import Budget, Op, compute_budget
from theoros_models.layers.d4 import (
    D4BatchNorm2d,
    D4Conv2d,
    D4DepthwiseConv2d,
    D4InvariantConv1x1,
    D4LiftConv2d,
)
from theoros_models.layers.heads import HEAD_NAMES
from theoros_models.model import TheorosNet

SMALL = TierSpec(
    name="small",
    nominal_parameters=1,
    stem_channels=3,
    stage_fields=(1, 2, 2),
    stage_blocks=(1, 2, 1),
    expansion=2,
    neck_fields=2,
    head_fields=1,
    hyper_embed=4,
    hyper_hidden=5,
    hyper_frequencies=2,
)


@pytest.mark.parametrize(("name", "bands"), list(itertools.product(TIERS, range(3, 14))))
def test_flight_limits(name: str, bands: int) -> None:
    budget = compute_budget(TIERS[name], bands=bands)
    assert budget.flight_parameters < FLIGHT_PARAMETER_LIMIT
    assert budget.int8_bytes <= FLIGHT_INT8_BYTES_LIMIT


@pytest.mark.parametrize("name", TIERS)
def test_tier_sizes(name: str) -> None:
    spec = TIERS[name]
    budget = compute_budget(spec, bands=13)
    deviation = abs(budget.flight_parameters / spec.nominal_parameters - 1.0)
    assert deviation <= TIER_TOLERANCE


def _flight_counts(model: TheorosNet, bands: int) -> tuple[int, int]:
    weights = bands * model.spec.stem_channels
    biases = model.spec.stem_channels
    for module in model.modules():
        if isinstance(module, D4LiftConv2d | D4Conv2d | D4DepthwiseConv2d | D4InvariantConv1x1):
            expanded = module.expanded_weight()
            weights += expanded.numel()
            biases += expanded.shape[0]
    return weights, biases


@pytest.mark.parametrize("spec", [SMALL, *TIERS.values()], ids=lambda spec: spec.name)
def test_analytic_matches_module(spec: TierSpec) -> None:
    model = TheorosNet(spec)
    budget = compute_budget(spec, bands=7)
    assert budget.training_parameters == sum(p.numel() for p in model.parameters())
    assert (budget.flight_weights, budget.flight_biases) == _flight_counts(model, 7)
    norms = sum(isinstance(m, D4BatchNorm2d) for m in model.modules())
    convs = sum(isinstance(m, D4LiftConv2d | D4Conv2d | D4DepthwiseConv2d) for m in model.modules())
    assert norms == convs


def test_hand_computed_small_tier() -> None:
    budget = compute_budget(SMALL, bands=4, heads=("fire",), tile_size=16)
    ops = {op.name: op for op in budget.ops}
    assert ops["stem"].weights == 12
    assert ops["stem"].macs == 12 * 16 * 16
    assert ops["lift"].shape == (8, 4, 4)
    assert ops["lift"].weights == 8 * 3 * 16
    expand = ops["stage1.block0.expand"]
    assert (expand.weights, expand.biases, expand.shape) == (8 * 16, 16, (16, 4, 4))
    assert ops["stage2.block0.depthwise"].shape == (16, 2, 2)
    assert ops["stage2.block0.depthwise"].weights == 16 * 4
    assert "stage2.block0.add" not in ops
    assert "stage2.block1.add" in ops
    assert ops["head.fire.output"].weights == 8
    assert [op.name for op in budget.ops if op.name.startswith("head.")] == [
        "head.fire.depthwise",
        "head.fire.pointwise",
        "head.fire.output",
    ]
    assert budget.int8_bytes == budget.flight_weights + 4 * budget.flight_biases
    assert budget.macs == sum(op.macs for op in budget.ops)


def test_peak_activation_liveness() -> None:
    ops = (
        Op("input", "input", (), (1, 10, 10)),
        Op("a", "conv", ("input",), (2, 10, 10)),
        Op("b", "conv", ("a",), (1, 10, 10)),
        Op("c", "add", ("a", "b"), (1, 10, 10)),
    )
    budget = Budget("t", 1, (), 0, ops)
    # While c is produced, a, b and c are live: 200 + 100 + 100 bytes.
    assert budget.peak_activation_bytes == 400


def test_head_subset_shrinks_budget() -> None:
    full = compute_budget(TIERS["base"])
    partial = compute_budget(TIERS["base"], heads=("cloud", "land_water"))
    assert partial.heads == ("cloud", "land_water")
    assert full.heads == HEAD_NAMES
    assert partial.flight_parameters < full.flight_parameters
    with pytest.raises(ValueError, match="unknown heads: smoke"):
        compute_budget(TIERS["base"], heads=("smoke",))


def test_peak_activation_reported() -> None:
    budget = compute_budget(TIERS["nano"])
    stem = next(op for op in budget.ops if op.name == "stem")
    assert budget.peak_activation_bytes >= 13 * 256 * 256 + stem.elements


def test_get_tier() -> None:
    assert get_tier("base") is TIERS["base"]
    with pytest.raises(KeyError, match="known: nano, base, max"):
        get_tier("huge")


def test_model_parameter_count_independent_of_bands() -> None:
    model = TheorosNet(TIERS["nano"])
    assert compute_budget(TIERS["nano"], bands=3).training_parameters == sum(
        p.numel() for p in model.parameters()
    )
    assert isinstance(model.stem.kernel[0], torch.nn.Linear)
