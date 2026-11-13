# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Balanced sampling across sensors and labelled tasks (MOD-REQ-110).

Samples are grouped by sensor and by the set of tasks they label. Each draw
first picks a group, uniformly or by the given weights, then a sample
uniformly within the group, so a large dataset of one sensor does not crowd
out a small one. Draws are a deterministic function of seed and epoch, and
each distributed rank receives a disjoint share of the same sequence.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Sampler


class BalancedSampler(Sampler[int]):
    """Group-balanced sampling with replacement.

    Args:
        groups: Group key of every dataset index.
        samples_per_rank: Indices yielded per epoch on each rank.
        seed: Base seed.
        rank: This process's rank.
        world_size: Number of processes.
        weights: Optional relative weight per group key; missing keys weigh 1.
    """

    def __init__(
        self,
        groups: Sequence[Hashable],
        samples_per_rank: int,
        *,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        weights: Mapping[Hashable, float] | None = None,
    ) -> None:
        if not groups:
            raise ValueError("cannot sample from an empty dataset")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} outside world size {world_size}")
        self.keys = sorted(set(groups), key=repr)
        self.members = [np.flatnonzero([group == key for group in groups]) for key in self.keys]
        raw = np.array([1.0 if weights is None else weights.get(key, 1.0) for key in self.keys])
        if np.any(raw < 0) or raw.sum() <= 0:
            raise ValueError("group weights must be non-negative with a positive sum")
        self.probabilities = raw / raw.sum()
        self.samples_per_rank, self.seed = samples_per_rank, seed
        self.rank, self.world_size = rank, world_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch whose draws :meth:`__iter__` yields."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_rank

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng([self.seed, self.epoch])
        total = self.samples_per_rank * self.world_size
        chosen = rng.choice(len(self.keys), size=total, p=self.probabilities)
        picks = [int(rng.choice(self.members[group])) for group in chosen]
        return iter(picks[self.rank :: self.world_size])
