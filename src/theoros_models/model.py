# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""TheorosNet in training form (MOD-REQ-011, MOD-REQ-004).

Data flow for a batch of tiles with N bands each:

1. spectral stem: (B, N, H, W) normalised reflectance to (B, C_s, H, W);
2. D4 lift, 4 x 4 at stride 4: to regular fields at stride 4;
3. backbone: inverted residual stages at strides 4, 8 and 16;
4. feature pyramid back to stride 4;
5. heads: cloud, land and water, vessel and fire, each (B, K, H / 4, W / 4).

Heads that a band set cannot support (see :mod:`theoros_models.capabilities`)
are reported in the output and skipped when no sample in the batch supports
them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from theoros_models.arch import LIFT_KERNEL, STRIDES, TierSpec
from theoros_models.capabilities import assess_bands
from theoros_models.layers.blocks import InvertedResidual
from theoros_models.layers.d4 import D4BatchNorm2d, D4LiftConv2d
from theoros_models.layers.heads import HEAD_NAMES, HEAD_SPECS, DenseHead
from theoros_models.layers.neck import FeaturePyramid
from theoros_models.layers.stem import CENTRE, GSD, SpectralStem
from theoros_models.sensors.schema import SensorProfile

# Initial probability of the sparse logits (vessel centres, active fire), so
# training starts near the background rate instead of at one half.
SPARSE_PRIOR = 0.01

# Metadata written into padded band slots; any finite positive values work
# because padded bands are masked out of the stem.
_PADDING_META = (1000.0, 10.0, 10.0)


@dataclass(frozen=True)
class _BandView:
    name: str
    centre_nm: float
    gsd_m: float


@dataclass(frozen=True)
class BandBatch:
    """Band metadata for a batch, padded to a common band count.

    Attributes:
        meta: (B, N, 3) float: centre wavelength in nm, FWHM in nm, GSD in m.
        mask: (B, N) bool: True for real bands, False for padding.
    """

    meta: Tensor
    mask: Tensor

    def __post_init__(self) -> None:
        if self.meta.ndim != 3 or self.meta.shape[-1] != 3:  # noqa: PLR2004
            raise ValueError(f"meta must have shape (B, N, 3), got {tuple(self.meta.shape)}")
        if self.mask.shape != self.meta.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("mask must be a bool tensor of shape (B, N)")
        if not bool(self.mask.any(dim=1).all()):
            raise ValueError("every sample needs at least one real band")

    @classmethod
    def from_profiles(
        cls,
        profiles: Sequence[SensorProfile],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> BandBatch:
        """Build the metadata of one sample per profile, padding to the largest band count."""
        width = max(len(profile.bands) for profile in profiles)
        meta = torch.tensor(_PADDING_META, dtype=dtype).repeat(len(profiles), width, 1)
        mask = torch.zeros(len(profiles), width, dtype=torch.bool)
        for row, profile in enumerate(profiles):
            for column, band in enumerate(profile.bands):
                meta[row, column] = torch.tensor(
                    (band.centre_nm, band.fwhm_nm, band.gsd_m), dtype=dtype
                )
                mask[row, column] = True
        return cls(meta.to(device), mask.to(device))

    @classmethod
    def from_profile(
        cls,
        profile: SensorProfile,
        batch_size: int = 1,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> BandBatch:
        """Build the metadata of ``batch_size`` samples from the same profile."""
        return cls.from_profiles([profile] * batch_size, dtype=dtype, device=device)

    def capabilities(self) -> Tensor:
        """Return which heads each sample's band set supports, (B, 4) bool in head order."""
        rows = []
        meta = self.meta.detach().cpu().tolist()
        mask = self.mask.detach().cpu().tolist()
        for sample_meta, sample_mask in zip(meta, mask, strict=True):
            bands = [
                _BandView(f"band{i}", values[CENTRE], values[GSD])
                for i, (values, real) in enumerate(zip(sample_meta, sample_mask, strict=True))
                if real
            ]
            enabled = set(assess_bands(bands).enabled)
            rows.append([name in enabled for name in HEAD_NAMES])
        return torch.tensor(rows, dtype=torch.bool, device=self.mask.device)


@dataclass(frozen=True)
class TheorosOutput:
    """Model outputs.

    Attributes:
        heads: Head name to output (B, K, H / 4, W / 4) for every head that at
            least one sample supports; channel meanings are in ``HEAD_SPECS``.
        enabled: (B, 4) bool in head order: whether each sample's band set
            supports each head. Outputs of unsupported heads carry no meaning
            and must be ignored.
    """

    heads: dict[str, Tensor]
    enabled: Tensor


class TheorosNet(nn.Module):
    """The band-flexible, D4-equivariant multi-task network.

    Args:
        spec: Tier architecture.
    """

    def __init__(self, spec: TierSpec) -> None:
        super().__init__()
        self.spec = spec
        self.stem = SpectralStem(
            spec.stem_channels, spec.hyper_embed, spec.hyper_hidden, spec.hyper_frequencies
        )
        first = spec.stage_fields[0]
        self.lift = D4LiftConv2d(spec.stem_channels, first, LIFT_KERNEL, stride=LIFT_KERNEL)
        self.lift_norm = D4BatchNorm2d(first)
        self.lift_act = nn.ReLU6()
        stages = []
        in_fields = first
        for index, (fields, blocks) in enumerate(
            zip(spec.stage_fields, spec.stage_blocks, strict=True)
        ):
            layers = []
            for block in range(blocks):
                stride = 2 if index > 0 and block == 0 else 1
                layers.append(InvertedResidual(in_fields, fields, spec.expansion, stride))
                in_fields = fields
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.neck = FeaturePyramid(spec.stage_fields, spec.neck_fields)
        self.heads = nn.ModuleDict(
            {
                head.name: DenseHead(spec.neck_fields, spec.head_fields, len(head.channels))
                for head in HEAD_SPECS
            }
        )
        with torch.no_grad():
            prior_logit = math.log(SPARSE_PRIOR / (1.0 - SPARSE_PRIOR))
            for name in ("vessel", "fire"):
                head = cast(DenseHead, self.heads[name])
                head.output.bias[0] = prior_logit

    def features(self, x: Tensor, bands: BandBatch) -> list[Tensor]:
        """Return the backbone outputs at strides 4, 8 and 16.

        Args:
            x: Normalised reflectance (B, N, H, W), H equal to W and a multiple of 16.
            bands: Matching band metadata.

        Returns:
            Regular feature maps (B, 8 * F_i, H / s_i, W / s_i).
        """
        batch, count, height, width = x.shape
        if (batch, count) != tuple(bands.mask.shape):
            raise ValueError(
                f"input has {batch} samples of {count} bands but metadata describes "
                f"{tuple(bands.mask.shape)}"
            )
        if height != width or height % STRIDES[-1]:
            raise ValueError(
                f"tiles must be square with a side divisible by 16, got {height}x{width}"
            )
        y = self.stem(x, bands.meta, bands.mask)
        y = self.lift_act(self.lift_norm(self.lift(y)))
        outputs = []
        for stage in self.stages:
            y = stage(y)
            outputs.append(y)
        return outputs

    def forward(self, x: Tensor, bands: BandBatch) -> TheorosOutput:
        """Run every supported head.

        Args:
            x: Normalised reflectance (B, N, H, W).
            bands: Matching band metadata.

        Returns:
            Head outputs at stride 4 and the per-sample capability mask.
        """
        enabled = bands.capabilities()
        pyramid = self.neck(self.features(x, bands))
        heads = {
            name: head(pyramid)
            for column, (name, head) in enumerate(self.heads.items())
            if bool(enabled[:, column].any())
        }
        return TheorosOutput(heads, enabled)
