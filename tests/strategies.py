# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Hypothesis strategies for sensor profiles."""

from __future__ import annotations

from typing import Any

from hypothesis import strategies as st

from theoros_models.sensors.schema import MAX_GSD_M, MIN_GSD_M

# Band edges are drawn on a 0.1 nm lattice strictly inside 400 to 2400 nm so
# that recomputed half-power edges never touch the range limits or each other.
_EDGE = st.integers(min_value=4010, max_value=23990)
GSD = st.floats(min_value=MIN_GSD_M, max_value=MAX_GSD_M, allow_nan=False)


@st.composite
def disjoint_bands(draw: st.DrawFn, count: int) -> list[dict[str, Any]]:
    """Draw count bands with disjoint half-power intervals in increasing order."""
    edges = sorted(draw(st.lists(_EDGE, min_size=2 * count, max_size=2 * count, unique=True)))
    bands = []
    for i in range(count):
        lower, upper = edges[2 * i] / 10.0, edges[2 * i + 1] / 10.0
        bands.append(
            {
                "name": f"B{i + 1}",
                "centre_nm": (lower + upper) / 2.0,
                "fwhm_nm": upper - lower,
                "gsd_m": draw(GSD),
            }
        )
    return bands


def profile_dict(bands: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    """Wrap bands in a complete profile mapping."""
    data: dict[str, Any] = {
        "schema_version": 1,
        "name": "generated",
        "platform": "Test",
        "instrument": "Generated",
        "encoding": {"scale": 1.0e-4, "offset": 0.0},
        "bands": bands,
    }
    data.update(overrides)
    return data


@st.composite
def valid_profiles(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a valid profile mapping with 3 to 13 bands."""
    count = draw(st.integers(min_value=3, max_value=13))
    return profile_dict(draw(disjoint_bands(count)))
