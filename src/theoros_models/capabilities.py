# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Which heads a band set can support, and why (MOD-REQ-004, MOD-REQ-065).

A head is enabled only when the sensor has the bands its task physically
needs. Each need is a set of spectral windows, any one of which satisfies it;
a band satisfies a window when its centre wavelength lies inside the window
and, where the window states one, its GSD is fine enough. The physical
justification of every window is recorded here and in docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from theoros_models.sensors.schema import SensorProfile


class BandLike(Protocol):
    """Anything with a band name, centre wavelength and GSD."""

    @property
    def name(self) -> str:
        """Band name."""

    @property
    def centre_nm(self) -> float:
        """Centre wavelength in nm."""

    @property
    def gsd_m(self) -> float:
        """Ground sample distance in m."""


@dataclass(frozen=True)
class Window:
    """A spectral window a band centre must fall into.

    Attributes:
        name: Short label, for example ``SWIR 1.6 um``.
        low_nm: Lowest accepted centre wavelength in nm, inclusive.
        high_nm: Highest accepted centre wavelength in nm, inclusive.
        max_gsd_m: Coarsest accepted GSD in m, or None for any.
    """

    name: str
    low_nm: float
    high_nm: float
    max_gsd_m: float | None = None

    def accepts(self, centre_nm: float, gsd_m: float) -> bool:
        """Return whether a band with this centre and GSD satisfies the window."""
        fine_enough = self.max_gsd_m is None or gsd_m <= self.max_gsd_m
        return self.low_nm <= centre_nm <= self.high_nm and fine_enough


@dataclass(frozen=True)
class Need:
    """One physical requirement of a head, met by a band in any of its windows.

    Attributes:
        windows: Alternative windows.
        reason: Why the task needs such a band.
    """

    windows: tuple[Window, ...]
    reason: str

    def describe(self) -> str:
        """Return the windows as text, for example ``NIR 760 to 900 nm``."""
        parts = []
        for window in self.windows:
            text = f"{window.name} ({window.low_nm:g} to {window.high_nm:g} nm"
            if window.max_gsd_m is not None:
                text += f", GSD at most {window.max_gsd_m:g} m"
            parts.append(text + ")")
        return " or ".join(parts)


BLUE = Window("blue", 420.0, 520.0)
GREEN = Window("green", 520.0, 600.0)
RED_OR_NIR = Window("red or NIR", 620.0, 900.0)
NIR = Window("NIR", 760.0, 900.0)
NIR_FINE = Window("NIR", 760.0, 900.0, max_gsd_m=10.0)
SWIR_16 = Window("SWIR 1.6 um", 1550.0, 1750.0)
SWIR_22 = Window("SWIR 2.2 um", 2080.0, 2350.0)

# Keys are in head order, matching theoros_models.layers.heads.HEAD_NAMES.
NEEDS: dict[str, tuple[Need, ...]] = {
    "cloud": (
        Need(
            (BLUE,),
            "cloud and haze scatter strongly in the blue, where clear land and water are dark",
        ),
        Need(
            (RED_OR_NIR,),
            "clouds stay bright and spectrally flat from the visible to the near infrared, "
            "unlike vegetation, water and soil",
        ),
    ),
    "land_water": (
        Need((GREEN,), "water reflects most in the green, the reference of water indices"),
        Need(
            (NIR, SWIR_16),
            "water absorbs near-infrared and shortwave infrared light almost completely, "
            "while land reflects it",
        ),
    ),
    "vessel": (
        Need(
            (NIR_FINE,),
            "vessels are brightest against water in the near infrared, and a 20 m vessel "
            "covers at least two pixels only at 10 m GSD or finer",
        ),
    ),
    "fire": (
        Need(
            (SWIR_22,),
            "thermal emission of flaming fire raises 2.2 um radiance far above reflected sunlight",
        ),
        Need(
            (SWIR_16,),
            "1.6 um rises less with fire and provides the reference that separates fire from "
            "bright surfaces",
        ),
    ),
}


@dataclass(frozen=True)
class HeadCapability:
    """Assessment of one head for one band set.

    Attributes:
        head: Head name.
        enabled: True when every need is met.
        satisfied_by: For each need in order, the first band name meeting it, or None.
        missing: Descriptions of the needs that are not met, with their reasons.
    """

    head: str
    enabled: bool
    satisfied_by: tuple[str | None, ...]
    missing: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityReport:
    """Assessment of every head for one band set.

    Attributes:
        heads: One entry per head, in head order.
    """

    heads: tuple[HeadCapability, ...]

    @property
    def enabled(self) -> tuple[str, ...]:
        """Names of the enabled heads, in head order."""
        return tuple(head.head for head in self.heads if head.enabled)

    @property
    def disabled(self) -> dict[str, tuple[str, ...]]:
        """Disabled heads mapped to the needs they miss."""
        return {head.head: head.missing for head in self.heads if not head.enabled}

    def format(self) -> str:
        """Return a human-readable table of the assessment."""
        lines = []
        for head in self.heads:
            if head.enabled:
                bands = ", ".join(name for name in head.satisfied_by if name is not None)
                lines.append(f"{head.head:<12}enabled   bands {bands}")
            else:
                lines.append(f"{head.head:<12}disabled")
                lines.extend(f"{'':<12}  needs {missing}" for missing in head.missing)
        return "\n".join(lines)


def assess_bands(bands: Sequence[BandLike]) -> CapabilityReport:
    """Assess every head for a band set.

    Args:
        bands: Bands with name, centre wavelength in nm and GSD in m.

    Returns:
        The capability report.
    """
    heads = []
    for head, needs in NEEDS.items():
        satisfied: list[str | None] = []
        missing = []
        for need in needs:
            match = next(
                (
                    band.name
                    for band in bands
                    if any(w.accepts(band.centre_nm, band.gsd_m) for w in need.windows)
                ),
                None,
            )
            satisfied.append(match)
            if match is None:
                missing.append(f"{need.describe()}: {need.reason}")
        heads.append(HeadCapability(head, not missing, tuple(satisfied), tuple(missing)))
    return CapabilityReport(tuple(heads))


def assess_profile(profile: SensorProfile) -> CapabilityReport:
    """Assess every head for a sensor profile."""
    return assess_bands(profile.bands)
