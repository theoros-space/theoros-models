# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Sensor profile schema.

A sensor profile describes the spectral bands an instrument delivers and how
its stored pixel values map to top-of-atmosphere reflectance. The compiler
turns one trained TheorosNet plus one profile into a flight graph, so every
rule here guards flight-relevant behaviour (MOD-REQ-020 to MOD-REQ-030).

All wavelengths are in nm, ground sample distances in m.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from theoros_models.sensors import srf as _srf

MIN_BANDS = 3
MAX_BANDS = 13
MIN_WAVELENGTH_NM = 400.0
MAX_WAVELENGTH_NM = 2400.0
MIN_GSD_M = 0.1
MAX_GSD_M = 1000.0
MAX_BAND_IOU = 0.5
SRF_MIN_SAMPLES = 3
SRF_MIN_WAVELENGTH_NM = 300.0
SRF_MAX_WAVELENGTH_NM = 2600.0
SRF_CENTROID_TOLERANCE_FWHM = 0.25
SRF_FWHM_RELATIVE_TOLERANCE = 0.25
SCHEMA_VERSION = 1

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=64)]
BandName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_]*$", max_length=32)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SpectralResponse(_Frozen):
    """Tabulated spectral response function of one band (MOD-REQ-026).

    Attributes:
        wavelengths_nm: Strictly increasing sample wavelengths in nm, shape (S,).
        response: Relative response at each wavelength, non-negative, shape (S,).
            The scale is arbitrary; consumers normalise to a peak of 1.
    """

    wavelengths_nm: tuple[FiniteFloat, ...]
    response: tuple[FiniteFloat, ...]

    @model_validator(mode="after")
    def _check_samples(self) -> SpectralResponse:
        x, y = self.wavelengths_nm, self.response
        if len(x) != len(y):
            raise ValueError(f"wavelengths_nm has {len(x)} samples but response has {len(y)}")
        if len(x) < SRF_MIN_SAMPLES:
            raise ValueError(f"an SRF needs at least {SRF_MIN_SAMPLES} samples, got {len(x)}")
        if any(b <= a for a, b in itertools.pairwise(x)):
            raise ValueError("SRF wavelengths must be strictly increasing")
        if x[0] < SRF_MIN_WAVELENGTH_NM or x[-1] > SRF_MAX_WAVELENGTH_NM:
            raise ValueError(
                f"SRF wavelengths must lie within {SRF_MIN_WAVELENGTH_NM:g} to "
                f"{SRF_MAX_WAVELENGTH_NM:g} nm"
            )
        if min(y) < 0.0:
            raise ValueError("SRF responses must be non-negative")
        if max(y) <= 0.0:
            raise ValueError("SRF responses must have a positive peak")
        return self


class Band(_Frozen):
    """One spectral band of a sensor.

    Attributes:
        name: Band identifier as used by the instrument, for example ``B8A``.
        centre_nm: Centre wavelength in nm.
        fwhm_nm: Full width at half maximum in nm. Where a mission publishes a
            bandwidth rather than a measured FWHM, the published bandwidth is used.
        gsd_m: Ground sample distance in m.
        srf: Optional tabulated spectral response; a Gaussian with ``centre_nm``
            and ``fwhm_nm`` is assumed when absent (MOD-REQ-027).
    """

    name: BandName
    centre_nm: FiniteFloat
    fwhm_nm: FiniteFloat = Field(gt=0.0)
    gsd_m: FiniteFloat = Field(ge=MIN_GSD_M, le=MAX_GSD_M)
    srf: SpectralResponse | None = None

    @property
    def lower_nm(self) -> float:
        """Lower edge of the half-power interval in nm."""
        return self.centre_nm - self.fwhm_nm / 2.0

    @property
    def upper_nm(self) -> float:
        """Upper edge of the half-power interval in nm."""
        return self.centre_nm + self.fwhm_nm / 2.0

    @model_validator(mode="after")
    def _check_band(self) -> Band:
        if self.lower_nm < MIN_WAVELENGTH_NM or self.upper_nm > MAX_WAVELENGTH_NM:
            raise ValueError(
                f"band {self.name}: half-power interval {self.lower_nm:g} to "
                f"{self.upper_nm:g} nm leaves {MIN_WAVELENGTH_NM:g} to {MAX_WAVELENGTH_NM:g} nm"
            )
        if self.srf is not None:
            centroid = _srf.centroid_nm(self.srf.wavelengths_nm, self.srf.response)
            if abs(centroid - self.centre_nm) > SRF_CENTROID_TOLERANCE_FWHM * self.fwhm_nm:
                raise ValueError(
                    f"band {self.name}: SRF centroid {centroid:.1f} nm is more than "
                    f"{SRF_CENTROID_TOLERANCE_FWHM:g} FWHM from the declared centre "
                    f"{self.centre_nm:g} nm"
                )
            width = _srf.measured_fwhm_nm(self.srf.wavelengths_nm, self.srf.response)
            if abs(width - self.fwhm_nm) > SRF_FWHM_RELATIVE_TOLERANCE * self.fwhm_nm:
                raise ValueError(
                    f"band {self.name}: SRF FWHM {width:.1f} nm differs from the declared "
                    f"{self.fwhm_nm:g} nm by more than {SRF_FWHM_RELATIVE_TOLERANCE:.0%}"
                )
        return self


class ReflectanceEncoding(_Frozen):
    """Affine map from stored pixel values to TOA reflectance (MOD-REQ-030).

    Reflectance equals ``scale * value + offset`` for every band.

    Attributes:
        scale: Reflectance per stored unit, positive.
        offset: Reflectance at stored value 0.
    """

    scale: FiniteFloat = Field(gt=0.0)
    offset: FiniteFloat = 0.0


def interval_iou(a: Band, b: Band) -> float:
    """Return the intersection over union of two half-power intervals.

    Args:
        a: First band.
        b: Second band.

    Returns:
        A value in [0, 1]; 0 when the intervals do not overlap.
    """
    intersection = min(a.upper_nm, b.upper_nm) - max(a.lower_nm, b.lower_nm)
    if intersection <= 0.0:
        return 0.0
    union = max(a.upper_nm, b.upper_nm) - min(a.lower_nm, b.lower_nm)
    return intersection / union


class SensorProfile(_Frozen):
    """Spectral description of one instrument as delivered to the model.

    Attributes:
        schema_version: Version of this schema; currently 1.
        name: Profile identifier in lower snake case.
        platform: Spacecraft or family, for example ``Sentinel-2``.
        instrument: Instrument name, for example ``MSI``.
        description: Free text, excluded from the fingerprint.
        reference: Public source of the band values, excluded from the fingerprint.
        encoding: Stored value to reflectance map.
        bands: 3 to 13 bands in strictly increasing centre wavelength.
    """

    schema_version: Literal[1]
    name: Identifier
    platform: Annotated[str, Field(min_length=1, max_length=64)]
    instrument: Annotated[str, Field(min_length=1, max_length=64)]
    description: str = ""
    reference: str = ""
    encoding: ReflectanceEncoding
    bands: tuple[Band, ...]

    @field_validator("bands")
    @classmethod
    def _check_bands(cls, bands: tuple[Band, ...]) -> tuple[Band, ...]:
        if not MIN_BANDS <= len(bands) <= MAX_BANDS:
            raise ValueError(f"a profile needs {MIN_BANDS} to {MAX_BANDS} bands, got {len(bands)}")
        names = [band.name for band in bands]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate band names: {', '.join(duplicates)}")
        for previous, current in itertools.pairwise(bands):
            if current.centre_nm <= previous.centre_nm:
                raise ValueError(
                    "bands must be in strictly increasing centre wavelength: "
                    f"{current.name} ({current.centre_nm:g} nm) follows "
                    f"{previous.name} ({previous.centre_nm:g} nm)"
                )
        for i, a in enumerate(bands):
            for b in bands[i + 1 :]:
                iou = interval_iou(a, b)
                if iou > MAX_BAND_IOU:
                    raise ValueError(
                        f"bands {a.name} and {b.name} are near-duplicates: half-power "
                        f"interval IoU {iou:.2f} exceeds {MAX_BAND_IOU:g}"
                    )
        return bands

    @property
    def band_names(self) -> tuple[str, ...]:
        """Band names in profile order."""
        return tuple(band.name for band in self.bands)

    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the flight-relevant content (MOD-REQ-029).

        The digest covers every field except ``description`` and ``reference``,
        serialised as canonical JSON with sorted keys and no whitespace. Floats
        are written in their shortest round-trip form.

        Returns:
            ``"sha256:"`` followed by 64 lowercase hexadecimal digits.
        """
        content = self.model_dump(mode="json", exclude={"description", "reference"})
        canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
