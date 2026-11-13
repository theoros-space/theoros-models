# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""One epoch of synthetic multi-sensor data for every tier on CPU (MOD-REQ-111)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from theoros_models.arch import TIERS
from theoros_models.data.augment import Augmenter
from theoros_models.data.datasets import SyntheticDataset
from theoros_models.data.packing import Batch, collate
from theoros_models.data.sample import Sample
from theoros_models.data.sampler import BalancedSampler
from theoros_models.losses import multitask_loss
from theoros_models.model import TheorosNet
from theoros_models.sensors import load_catalogue_profile

PROFILES = ("sentinel2_msi", "landsat8_oli", "planetscope_superdove", "generic_cubesat_4band")


class _Augmented(torch.utils.data.Dataset[Sample]):
    def __init__(self, dataset: SyntheticDataset, seed: int) -> None:
        self.dataset, self.seed = dataset, seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        rng = np.random.default_rng([self.seed, index])
        return Augmenter()(self.dataset[index], rng)


@pytest.mark.parametrize("tier", TIERS)
def test_one_epoch(tier: str) -> None:
    torch.manual_seed(0)
    dataset = SyntheticDataset([load_catalogue_profile(p) for p in PROFILES], 8, size=64, seed=9)
    sampler = BalancedSampler([dataset.group(i) for i in range(len(dataset))], 8, seed=1)
    loader: DataLoader[Sample] = DataLoader(
        _Augmented(dataset, 3), batch_size=4, sampler=sampler, collate_fn=collate
    )
    model = TheorosNet(TIERS[tier]).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    steps = 0
    batch: Batch
    for batch in loader:
        output = model(batch.x, batch.bands)
        total, parts = multitask_loss(output, batch)
        assert bool(torch.isfinite(total)), parts
        optimiser.zero_grad()
        torch.autograd.backward(total)
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
        optimiser.step()
        steps += 1
    assert steps == 2
    changed = [
        not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters(), strict=True)
    ]
    assert all(changed)
