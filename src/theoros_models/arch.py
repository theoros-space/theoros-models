# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""TheorosNet tier definitions and flight limits (MOD-REQ-001, MOD-REQ-002, MOD-REQ-010).

Widths are counted in regular fields; a field is 8 channels once the D4
group convolutions are expanded for flight.
"""

from __future__ import annotations

from dataclasses import dataclass

TILE_SIZE = 256
STRIDES = (4, 8, 16)
LIFT_KERNEL = 4
FLIGHT_PARAMETER_LIMIT = 1_000_000
FLIGHT_INT8_BYTES_LIMIT = 1024 * 1024
TIER_TOLERANCE = 0.05


@dataclass(frozen=True)
class TierSpec:
    """Architecture of one tier.

    Attributes:
        name: Tier name.
        nominal_parameters: Nominal flight parameter count (MOD-REQ-010).
        stem_channels: Trivial channels produced by the spectral stem.
        stage_fields: Fields of the backbone stages at strides 4, 8 and 16.
        stage_blocks: Inverted residual blocks per stage; the first block of
            the stride 8 and 16 stages downsamples.
        expansion: Hidden width multiplier of the inverted residual blocks.
        neck_fields: Fields of the feature pyramid.
        head_fields: Hidden fields of every head.
        hyper_embed: Width of the stem's per-band embedding.
        hyper_hidden: Width of the stem's kernel generator.
        hyper_frequencies: Octaves of the stem's wavelength encoding.
    """

    name: str
    nominal_parameters: int
    stem_channels: int
    stage_fields: tuple[int, int, int]
    stage_blocks: tuple[int, int, int]
    expansion: int
    neck_fields: int
    head_fields: int
    hyper_embed: int = 32
    hyper_hidden: int = 64
    hyper_frequencies: int = 8


TIERS: dict[str, TierSpec] = {
    "nano": TierSpec(
        name="nano",
        nominal_parameters=63_000,
        stem_channels=12,
        stage_fields=(2, 4, 6),
        stage_blocks=(2, 3, 4),
        expansion=2,
        neck_fields=4,
        head_fields=3,
        hyper_embed=16,
        hyper_hidden=32,
    ),
    "base": TierSpec(
        name="base",
        nominal_parameters=450_000,
        stem_channels=16,
        stage_fields=(8, 12, 16),
        stage_blocks=(2, 3, 4),
        expansion=2,
        neck_fields=12,
        head_fields=3,
    ),
    "max": TierSpec(
        name="max",
        nominal_parameters=890_000,
        stem_channels=16,
        stage_fields=(12, 16, 24),
        stage_blocks=(2, 3, 4),
        expansion=2,
        neck_fields=12,
        head_fields=4,
    ),
}


def get_tier(name: str) -> TierSpec:
    """Return a tier by name.

    Raises:
        KeyError: With the known tier names, if ``name`` is not one.
    """
    try:
        return TIERS[name]
    except KeyError:
        raise KeyError(f"unknown tier {name!r}; known: {', '.join(TIERS)}") from None
