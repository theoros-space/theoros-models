# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Helpers for tests of torch modules."""

from __future__ import annotations

import torch
from torch import nn

from theoros_models.layers.d4 import D4BatchNorm2d


def randomise_norms(module: nn.Module, generator: torch.Generator) -> None:
    """Give every D4 batch norm non-trivial statistics and affine parameters."""
    with torch.no_grad():
        for norm in module.modules():
            if isinstance(norm, D4BatchNorm2d):
                shape = norm.running_mean.shape  # type: ignore[union-attr]
                norm.running_mean.copy_(torch.randn(shape, generator=generator))  # type: ignore[union-attr]
                norm.running_var.copy_(torch.rand(shape, generator=generator) + 0.5)  # type: ignore[union-attr]
                norm.weight.copy_(torch.randn(shape, generator=generator))
                norm.bias.copy_(torch.randn(shape, generator=generator))
