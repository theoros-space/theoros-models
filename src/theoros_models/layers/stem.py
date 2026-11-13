# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Spectral hypernetwork stem (MOD-REQ-060).

The stem reads any set of 3 to 13 bands. For each band a small network maps
the band's centre wavelength, FWHM and GSD, together with a summary of the
whole band set, to a 1 x 1 kernel; the stem output is the sum of every band
filtered by its kernel plus a generated bias. For a fixed sensor the kernels
are constants, so the compiler replaces the hypernetwork by one plain 1 x 1
convolution. A 1 x 1 kernel commutes with every spatial symmetry, which keeps
the network D4-equivariant from its first layer.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from theoros_models.sensors.schema import MAX_BANDS, MAX_WAVELENGTH_NM, MIN_WAVELENGTH_NM

# Metadata columns of the (B, N, 3) band tensor.
CENTRE, FWHM, GSD = 0, 1, 2
_FWHM_REFERENCE_NM = 100.0
_GSD_REFERENCE_M = 10.0


def band_features(meta: Tensor, frequencies: int) -> Tensor:
    """Encode band metadata as features for the hypernetwork.

    The centre wavelength, scaled to [0, 1] over 400 to 2400 nm, is encoded by
    sines and cosines at ``frequencies`` octaves; FWHM and GSD enter as
    logarithms relative to 100 nm and 10 m.

    Args:
        meta: Band metadata (B, N, 3): centre in nm, FWHM in nm, GSD in m.
        frequencies: Number of octaves.

    Returns:
        Features (B, N, 2 * frequencies + 2).
    """
    position = (meta[..., CENTRE] - MIN_WAVELENGTH_NM) / (MAX_WAVELENGTH_NM - MIN_WAVELENGTH_NM)
    scales = math.pi * 2.0 ** torch.arange(frequencies, device=meta.device, dtype=meta.dtype)
    angles = position[..., None] * scales  # (B, N, F)
    return torch.cat(
        [
            torch.sin(angles),
            torch.cos(angles),
            torch.log(meta[..., FWHM : FWHM + 1] / _FWHM_REFERENCE_NM),
            torch.log(meta[..., GSD : GSD + 1] / _GSD_REFERENCE_M),
        ],
        dim=-1,
    )


class SpectralStem(nn.Module):
    """Generate per-band 1 x 1 kernels from band metadata and apply them.

    Args:
        out_channels: Trivial output channels.
        embed: Width of the per-band embedding.
        hidden: Width of the kernel generator's hidden layer.
        frequencies: Octaves of the wavelength encoding.
    """

    def __init__(self, out_channels: int, embed: int, hidden: int, frequencies: int) -> None:
        super().__init__()
        self.out_channels, self.frequencies = out_channels, frequencies
        self.embed = nn.Sequential(
            nn.Linear(2 * frequencies + 2, embed), nn.GELU(), nn.Linear(embed, embed)
        )
        self.kernel = nn.Sequential(
            nn.Linear(2 * embed + 1, hidden), nn.GELU(), nn.Linear(hidden, out_channels)
        )
        self.bias = nn.Linear(embed + 1, out_channels)

    def generate(self, meta: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Generate the stem kernels for a batch of band sets.

        Args:
            meta: Band metadata (B, N, 3); padded entries may hold any finite
                positive values.
            mask: True for real bands, False for padding (B, N).

        Returns:
            Kernels (B, N, out_channels), zero for padded bands, and biases
            (B, out_channels).
        """
        weights = mask.to(meta.dtype)[..., None]  # (B, N, 1)
        count = weights.sum(dim=1)  # (B, 1)
        embedded = self.embed(band_features(meta, self.frequencies))  # (B, N, E)
        context = (embedded * weights).sum(dim=1) / count  # (B, E)
        size = torch.log(count) / math.log(MAX_BANDS)  # (B, 1)
        summary = torch.cat([context, size], dim=-1)  # (B, E + 1)
        per_band = torch.cat([embedded, summary[:, None].expand(-1, meta.shape[1], -1)], dim=-1)
        kernels = self.kernel(per_band) * weights
        return kernels, self.bias(summary)

    def forward(self, x: Tensor, meta: Tensor, mask: Tensor) -> Tensor:
        """Apply the generated kernels.

        Args:
            x: Normalised reflectance (B, N, H, W); padded bands may hold any
                finite value, they are ignored.
            meta: Band metadata (B, N, 3).
            mask: Band validity (B, N).

        Returns:
            Trivial features (B, out_channels, H, W).
        """
        kernels, bias = self.generate(meta, mask)
        return torch.einsum("bnhw,bnc->bchw", x, kernels) + bias[:, :, None, None]
