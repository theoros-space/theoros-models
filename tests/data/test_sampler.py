# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Tests for balanced sampling (MOD-REQ-110)."""

from __future__ import annotations

from collections import Counter

import pytest

from theoros_models.data.sampler import BalancedSampler

GROUPS = ["a"] * 900 + ["b"] * 90 + ["c"] * 10


def _shares(sampler: BalancedSampler) -> dict[str, float]:
    counts = Counter(GROUPS[i] for i in sampler)
    return {key: counts[key] / len(sampler) for key in "abc"}


def test_groups_equally_likely() -> None:
    shares = _shares(BalancedSampler(GROUPS, 30_000, seed=1))
    for share in shares.values():
        assert share == pytest.approx(1 / 3, abs=0.02)


def test_weights() -> None:
    shares = _shares(BalancedSampler(GROUPS, 30_000, seed=1, weights={"a": 2.0, "c": 0.0}))
    assert shares == pytest.approx({"a": 2 / 3, "b": 1 / 3, "c": 0.0}, abs=0.02)


def test_deterministic_by_seed_and_epoch() -> None:
    sampler = BalancedSampler(GROUPS, 100, seed=3)
    first = list(sampler)
    assert list(sampler) == first
    sampler.set_epoch(1)
    assert list(sampler) != first
    assert list(BalancedSampler(GROUPS, 100, seed=4)) != first
    assert len(sampler) == 100


def test_ranks_share_one_sequence() -> None:
    whole = list(BalancedSampler(GROUPS, 120, seed=5))
    parts = [list(BalancedSampler(GROUPS, 40, seed=5, rank=r, world_size=3)) for r in range(3)]
    assert all(len(part) == 40 for part in parts)
    interleaved = [parts[i % 3][i // 3] for i in range(120)]
    assert interleaved == whole


def test_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        BalancedSampler([], 1)
    with pytest.raises(ValueError, match="rank 2 outside"):
        BalancedSampler(GROUPS, 1, rank=2, world_size=2)
    with pytest.raises(ValueError, match="non-negative"):
        BalancedSampler(GROUPS, 1, weights={"a": -1.0})
