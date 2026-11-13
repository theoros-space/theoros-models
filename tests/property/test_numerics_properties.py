# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Property tests for fixed-point numerics (MOD-REQ-040 to MOD-REQ-044)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tests import numerics_reference as ref
from theoros_models import numerics as nm
from theoros_models.numerics import RoundingMode

INT32 = st.integers(min_value=nm.INT32_MIN, max_value=nm.INT32_MAX)
MULTIPLIERS = st.floats(min_value=2.0**-31, max_value=2.0**29)


@given(MULTIPLIERS)
def test_quantize_multiplier_error(real: float) -> None:
    mantissa, shift = nm.quantize_multiplier(real)
    assert 2**30 <= mantissa <= 2**31 - 1
    assert nm.MULTIPLIER_MIN_SHIFT <= shift <= nm.MULTIPLIER_MAX_SHIFT
    represented = Fraction(mantissa) * Fraction(2) ** (shift - 31)
    assert abs(represented - Fraction(real)) <= Fraction(real) * Fraction(1, 2**31)


@given(INT32, INT32)
def test_srdhm_random(a: int, b: int) -> None:
    assert int(nm.saturating_rounding_doubling_high_mul(a, b)) == ref.srdhm(a, b)


@given(
    hnp.arrays(np.int64, st.integers(1, 64), elements=INT32),
    st.floats(min_value=2.0**-24, max_value=8.0),
    st.integers(min_value=-128, max_value=127),
)
def test_requantize_within_one(acc: np.ndarray, real: float, zero_point: int) -> None:
    mantissa, shift = nm.quantize_multiplier(real)
    out = nm.requantize(acc, mantissa, shift, zero_point)
    exact = Fraction(real)
    for value, got in zip(acc.tolist(), out.tolist(), strict=True):
        correct = ref.round_fraction(Fraction(value) * exact, RoundingMode.HALF_EVEN)
        expected = ref.clamp(correct + zero_point, -128, 127)
        assert abs(got - expected) <= 1


@given(
    st.floats(min_value=1e-4, max_value=10.0),
    st.integers(min_value=-128, max_value=127),
    st.data(),
)
def test_quantize_roundtrip(scale: float, zero_point: int, data: st.DataObject) -> None:
    low = scale * (-128 - zero_point)
    high = scale * (127 - zero_point)
    x = data.draw(hnp.arrays(np.float64, 32, elements=st.floats(min_value=low, max_value=high)))
    codes = nm.quantize(x, scale, zero_point)
    back = nm.dequantize(codes, scale, zero_point)
    assert np.all(np.abs(back - x) <= scale / 2.0 * (1.0 + 1e-9))


@given(
    st.floats(min_value=1e-3, max_value=1.0),
    st.integers(min_value=-128, max_value=127),
    st.floats(min_value=1.0, max_value=1e6),
)
def test_quantize_saturates(scale: float, zero_point: int, excess: float) -> None:
    beyond = np.array([scale * (127 - zero_point) + excess, scale * (-128 - zero_point) - excess])
    np.testing.assert_array_equal(nm.quantize(beyond, scale, zero_point), [127, -128])


@given(
    st.lists(st.integers(min_value=-(2**52), max_value=2**52), min_size=1, max_size=32),
    st.integers(min_value=0, max_value=52),
    st.sampled_from(list(RoundingMode)),
)
def test_shift_and_float_rounding_agree(values: list[int], shift: int, mode: RoundingMode) -> None:
    x = np.array(values, dtype=np.int64)
    np.testing.assert_array_equal(
        nm.rounding_shift_right(x, shift, mode), nm.round_to_int(x / 2.0**shift, mode)
    )


@given(
    st.lists(st.floats(min_value=-(2.0**62), max_value=2.0**62), min_size=1, max_size=32),
    st.sampled_from(list(RoundingMode)),
)
def test_round_to_int_matches_reference(values: list[float], mode: RoundingMode) -> None:
    expected = [ref.round_fraction(Fraction(v), mode) for v in values]
    np.testing.assert_array_equal(nm.round_to_int(np.array(values), mode), expected)
