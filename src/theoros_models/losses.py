# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Per-head losses with masking for missing labels and capabilities (MOD-REQ-120, MOD-REQ-121).

A head's loss is averaged over the samples that both carry its label and
whose band set supports the head; every other sample contributes nothing,
so its gradient through that head is exactly zero. Dense outputs at stride 4
are upsampled bilinearly to label resolution, which commutes with the
symmetries of the square.

* cloud: cross-entropy over 4 classes with optional class weights;
* land and water, fire: binary cross-entropy with a positive-class weight;
* vessel: penalty-reduced focal loss on a Gaussian centre heatmap plus an L1
  loss on log length and log width at the centre pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from theoros_models.arch import STRIDES
from theoros_models.data.packing import Batch
from theoros_models.data.sample import IGNORE
from theoros_models.layers.heads import HEAD_NAMES
from theoros_models.model import TheorosOutput

OUTPUT_STRIDE = STRIDES[0]
_FOCAL_ALPHA = 2.0
_FOCAL_BETA = 4.0
_PROBABILITY_EPS = 1e-6


@dataclass(frozen=True)
class LossConfig:
    """Weights of the multi-task loss.

    Attributes:
        head_weights: Multiplier of each head's loss.
        cloud_class_weights: Cross-entropy weight per cloud class, or None.
        water_pos_weight: Positive-class weight of the water loss.
        fire_pos_weight: Positive-class weight of the fire loss; fire pixels are rare.
        vessel_sigma_px: Heatmap Gaussian standard deviation in output pixels.
        vessel_size_weight: Weight of the size regression against the heatmap loss.
    """

    head_weights: dict[str, float] = field(default_factory=lambda: dict.fromkeys(HEAD_NAMES, 1.0))
    cloud_class_weights: tuple[float, float, float, float] | None = None
    water_pos_weight: float = 1.0
    fire_pos_weight: float = 10.0
    vessel_sigma_px: float = 1.0
    vessel_size_weight: float = 0.1


def vessel_targets(
    vessels: Tensor, count: Tensor, size: int, sigma_px: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Render vessel annotations as stride 4 training targets.

    The heatmap is the pixelwise maximum of unit Gaussians at the vessel
    centres, evaluated at output pixel centres, so it transforms exactly with
    the tile. The size target holds log length and log width at the output
    pixel containing each centre.

    Args:
        vessels: (B, K, 4) rows of centre row, centre column (input pixels),
            length in m and width in m.
        count: (B,) valid rows per sample.
        size: Input tile edge in pixels.
        sigma_px: Gaussian standard deviation in output pixels.

    Returns:
        Heatmap (B, S, S), size target (B, 2, S, S) and a bool centre mask
        (B, S, S), with S = size / 4.
    """
    batch, rows = vessels.shape[:2]
    out = size // OUTPUT_STRIDE
    heatmap = torch.zeros(batch, out, out, dtype=vessels.dtype, device=vessels.device)
    sizes = torch.zeros(batch, 2, out, out, dtype=vessels.dtype, device=vessels.device)
    centres = torch.zeros(batch, out, out, dtype=torch.bool, device=vessels.device)
    grid = torch.arange(out, dtype=vessels.dtype, device=vessels.device) + 0.5
    for b in range(batch):
        for k in range(min(rows, int(count[b]))):
            row, col = vessels[b, k, 0] / OUTPUT_STRIDE, vessels[b, k, 1] / OUTPUT_STRIDE
            gauss = torch.exp(
                -((grid[:, None] - row) ** 2 + (grid[None, :] - col) ** 2) / (2 * sigma_px**2)
            )
            heatmap[b] = torch.maximum(heatmap[b], gauss)
            i = int(torch.clamp(row.floor(), 0, out - 1))
            j = int(torch.clamp(col.floor(), 0, out - 1))
            heatmap[b, i, j] = 1.0
            centres[b, i, j] = True
            sizes[b, :, i, j] = torch.log(vessels[b, k, 2:4])
    return heatmap, sizes, centres


def _upsample(logits: Tensor) -> Tensor:
    return F.interpolate(
        logits, scale_factor=float(OUTPUT_STRIDE), mode="bilinear", align_corners=False
    )


def _masked_mean(per_sample: Tensor, active: Tensor) -> Tensor:
    weights = active.to(per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1.0)


def _dense_binary(logits: Tensor, target: Tensor, pos_weight: float) -> Tensor:
    valid = target != IGNORE
    weight = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    per_pixel = F.binary_cross_entropy_with_logits(
        _upsample(logits)[:, 0], (target == 1).to(logits.dtype), pos_weight=weight, reduction="none"
    )
    return (per_pixel * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)


def _cloud(logits: Tensor, target: Tensor, class_weights: Tensor | None) -> Tensor:
    per_pixel = F.cross_entropy(
        _upsample(logits), target, weight=class_weights, ignore_index=IGNORE, reduction="none"
    )
    valid = target != IGNORE
    return per_pixel.sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp(min=1)


def _vessel(output: Tensor, batch: Batch, config: LossConfig) -> Tensor:
    heatmap, sizes, centres = vessel_targets(
        batch.vessels.to(output.dtype),
        batch.vessel_count,
        batch.x.shape[-1],
        config.vessel_sigma_px,
    )
    probability = torch.sigmoid(output[:, 0]).clamp(_PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
    positive = heatmap == 1.0
    focal = torch.where(
        positive,
        -((1 - probability) ** _FOCAL_ALPHA) * torch.log(probability),
        -((1 - heatmap) ** _FOCAL_BETA) * probability**_FOCAL_ALPHA * torch.log(1 - probability),
    )
    count = centres.sum(dim=(1, 2)).to(output.dtype)
    focal_loss = focal.sum(dim=(1, 2)) / count.clamp(min=1.0)
    size_error = (output[:, 1:3] - sizes).abs().sum(dim=1) * centres
    size_loss = size_error.sum(dim=(1, 2)) / count.clamp(min=1.0)
    return focal_loss + config.vessel_size_weight * size_loss


def multitask_loss(
    output: TheorosOutput, batch: Batch, config: LossConfig | None = None
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the weighted sum of head losses.

    Args:
        output: Model output for the batch.
        batch: The packed batch with labels.
        config: Loss weights; defaults to :class:`LossConfig`.

    Returns:
        The total loss and the unweighted loss of each head; a head without
        any active sample has loss 0.
    """
    config = config or LossConfig()
    active = batch.labelled & output.enabled
    reference = next(iter(output.heads.values()))
    zero = reference.sum() * 0.0
    losses: dict[str, Tensor] = {}
    for column, name in enumerate(HEAD_NAMES):
        logits = output.heads.get(name)
        if logits is None or not bool(active[:, column].any()):
            losses[name] = zero
            continue
        if name == "cloud":
            weights = (
                None
                if config.cloud_class_weights is None
                else torch.tensor(
                    config.cloud_class_weights, dtype=logits.dtype, device=logits.device
                )
            )
            per_sample = _cloud(logits, batch.cloud, weights)
        elif name == "land_water":
            per_sample = _dense_binary(logits, batch.land_water, config.water_pos_weight)
        elif name == "fire":
            per_sample = _dense_binary(logits, batch.fire, config.fire_pos_weight)
        else:
            per_sample = _vessel(logits, batch, config)
        losses[name] = _masked_mean(per_sample, active[:, column])
    total = sum(
        (config.head_weights.get(name, 1.0) * value for name, value in losses.items()), zero
    )
    return total, losses
