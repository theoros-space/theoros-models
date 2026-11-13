# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Exhaustive and edge-case tests for fixed-point numerics (MOD-REQ-040 to MOD-REQ-044)."""

from __future__ import annotations

import itertools
from fractions import Fraction

import numpy as np
import pytest

from tests import numerics_reference as ref
from theoros_models import numerics as nm
from theoros_models.numerics import RoundingMode

MODES = list(RoundingMode)
INT32_EDGES = sorted(
    {
        value
        for k in range(32)
        for value in (2**k, 2**k - 1, 2**k + 1, -(2**k), -(2**k) + 1, -(2**k) - 1)
        if nm.INT32_MIN <= value <= nm.INT32_MAX
    }
    | {0, nm.INT32_MIN, nm.INT32_MAX, 46341, -46341, 1 << 30, 3 << 29}
)


@pytest.mark.parametrize(
    ("bits", "expected"),
    [(8, (-128, 127)), (16, (-32768, 32767)), (32, (-(2**31), 2**31 - 1))],
)
def test_int_range(bits: int, expected: tuple[int, int]) -> None:
    assert nm.int_range(bits) == expected


def test_int_range_rejects_other_widths() -> None:
    with pytest.raises(ValueError, match="unsupported integer width 12"):
        nm.int_range(12)


@pytest.mark.parametrize("bits", [8, 16])
def test_saturate_exhaustive(bits: int) -> None:
    low, high = nm.int_range(bits)
    x = np.arange(-(2**17), 2**17, dtype=np.int64)
    expected = np.array([ref.clamp(int(v), low, high) for v in x])
    np.testing.assert_array_equal(nm.saturate(x, bits), expected)


def test_saturate_int32_edges() -> None:
    x = np.array([-(2**40), nm.INT32_MIN - 1, nm.INT32_MIN, 0, nm.INT32_MAX, 2**40])
    np.testing.assert_array_equal(
        nm.saturate(x, 32),
        [nm.INT32_MIN, nm.INT32_MIN, nm.INT32_MIN, 0, nm.INT32_MAX, nm.INT32_MAX],
    )


def test_saturate_rejects_floats() -> None:
    with pytest.raises(TypeError, match="integer array"):
        nm.saturate(np.array([1.5]))


def test_saturating_add_sub_int8_exhaustive() -> None:
    a, b = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128), indexing="ij")
    add = np.array(
        [
            [ref.clamp(int(x) + int(y), -128, 127) for y in range(-128, 128)]
            for x in range(-128, 128)
        ]
    )
    sub = np.array(
        [
            [ref.clamp(int(x) - int(y), -128, 127) for y in range(-128, 128)]
            for x in range(-128, 128)
        ]
    )
    np.testing.assert_array_equal(nm.saturating_add(a, b), add)
    np.testing.assert_array_equal(nm.saturating_sub(a, b), sub)
    np.testing.assert_array_equal(nm.saturating_add(a.astype(np.int8), b.astype(np.int8)), add)


@pytest.mark.parametrize("bits", [16, 32])
def test_saturating_add_sub_wide_edges(bits: int) -> None:
    low, high = nm.int_range(bits)
    edges = [v for v in INT32_EDGES if low <= v <= high]
    a, b = (np.array(v) for v in zip(*itertools.product(edges, repeat=2), strict=True))
    add = [ref.clamp(int(x) + int(y), low, high) for x, y in zip(a, b, strict=True)]
    sub = [ref.clamp(int(x) - int(y), low, high) for x, y in zip(a, b, strict=True)]
    np.testing.assert_array_equal(nm.saturating_add(a, b, bits), add)
    np.testing.assert_array_equal(nm.saturating_sub(a, b, bits), sub)


def test_saturating_ops_reject_out_of_range_operands() -> None:
    with pytest.raises(ValueError, match="a does not fit in int8"):
        nm.saturating_add([128], [0])
    with pytest.raises(ValueError, match="b does not fit in int16"):
        nm.saturating_sub([0], [-(2**15) - 1], bits=16)
    with pytest.raises(TypeError, match="integer array"):
        nm.saturating_add([0.0], [0])


@pytest.mark.parametrize("mode", MODES)
def test_round_half_integers_exhaustive(mode: RoundingMode) -> None:
    halves = np.arange(-4000, 4001, dtype=np.float64) / 2.0
    expected = [ref.round_fraction(Fraction(float(v)), mode) for v in halves]
    np.testing.assert_array_equal(nm.round_to_int(halves, mode), expected)


@pytest.mark.parametrize("mode", MODES)
def test_round_hard_values(mode: RoundingMode) -> None:
    below_half = np.nextafter(0.5, 0.0)
    values = np.array(
        [
            below_half,
            -below_half,
            np.nextafter(0.5, 1.0),
            2.0**52 - 0.5,
            -(2.0**52) + 0.5,
            2.0**53,
            -(2.0**62),
            np.nextafter(2.0**63, 0.0),
            1e-300,
            -1e-300,
            0.0,
            -0.0,
        ]
    )
    expected = [ref.round_fraction(Fraction(float(v)), mode) for v in values]
    np.testing.assert_array_equal(nm.round_to_int(values, mode), expected)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 2.0**63, -(2.0**63)])
def test_round_rejects_unrepresentable(value: float) -> None:
    with pytest.raises(ValueError, match=r"non-finite|int64 range"):
        nm.round_to_int(np.array([0.0, value]))


@pytest.mark.parametrize("mode", MODES)
def test_rounding_shift_exhaustive(mode: RoundingMode) -> None:
    x = np.arange(-2048, 2048, dtype=np.int64)
    for shift in range(14):
        expected = [ref.divide_pow2(int(v), shift, mode) for v in x]
        np.testing.assert_array_equal(nm.rounding_shift_right(x, shift, mode), expected)


@pytest.mark.parametrize("mode", MODES)
def test_rounding_shift_matches_float_rounding_int16(mode: RoundingMode) -> None:
    # Every int16 value divided by 2^s is exact in float64, so the integer
    # and floating-point definitions must agree everywhere.
    x = np.arange(-(2**15), 2**15, dtype=np.int64)
    for shift in range(21):
        np.testing.assert_array_equal(
            nm.rounding_shift_right(x, shift, mode),
            nm.round_to_int(x / 2.0**shift, mode),
        )


@pytest.mark.parametrize("mode", MODES)
def test_rounding_shift_wide_values(mode: RoundingMode) -> None:
    values = [2**62 - 1, -(2**62), 2**61 + 2**60, -(2**61) - 1, 3 << 40, -(5 << 33)]
    for shift in (1, 17, 31, 40, 61, 62):
        expected = [ref.divide_pow2(v, shift, mode) for v in values]
        np.testing.assert_array_equal(nm.rounding_shift_right(values, shift, mode), expected)


def test_rounding_shift_per_element_shifts() -> None:
    x = np.array([[5, 5, 5], [-5, -5, -5]])
    out = nm.rounding_shift_right(x, np.array([0, 1, 2]), RoundingMode.HALF_AWAY_FROM_ZERO)
    np.testing.assert_array_equal(out, [[5, 3, 1], [-5, -3, -1]])


def test_rounding_shift_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="shift must be between 0 and 62"):
        nm.rounding_shift_right([1], -1)
    with pytest.raises(ValueError, match="shift must be between 0 and 62"):
        nm.rounding_shift_right([1], 63)
    with pytest.raises(TypeError, match="integer array"):
        nm.rounding_shift_right([1.0], 1)


def test_srdhm_matches_reference() -> None:
    a, b = (np.array(v) for v in zip(*itertools.product(INT32_EDGES, repeat=2), strict=True))
    expected = [ref.srdhm(int(x), int(y)) for x, y in zip(a, b, strict=True)]
    np.testing.assert_array_equal(nm.saturating_rounding_doubling_high_mul(a, b), expected)


def test_srdhm_overflow_and_ties() -> None:
    srdhm = nm.saturating_rounding_doubling_high_mul
    assert srdhm(nm.INT32_MIN, nm.INT32_MIN) == nm.INT32_MAX
    assert srdhm(nm.INT32_MIN, nm.INT32_MAX) == -nm.INT32_MAX
    # a * b / 2^31 = +-0.5 and +-1.5: ties go toward positive infinity.
    assert srdhm(1 << 15, 1 << 15) == 1
    assert srdhm(-(1 << 15), 1 << 15) == 0
    assert srdhm(3 << 15, 1 << 15) == 2
    assert srdhm(-(3 << 15), 1 << 15) == -1


def test_srdhm_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="a does not fit in int32"):
        nm.saturating_rounding_doubling_high_mul([2**31], [1])


@pytest.mark.parametrize(
    ("real", "expected"),
    [
        (0.0, (0, 0)),
        (0.5, (2**30, 0)),
        (1.0, (2**30, 1)),
        (0.75, (3 << 29, 0)),
        (2.0**-31, (2**30, -30)),
        (2.0**-32, (2**30, -31)),
        (2.0**-33, (0, 0)),
        (1.0 - 2.0**-40, (2**30, 1)),
        (2.0**30 - 1.0, (2**31 - 2, 30)),
    ],
)
def test_quantize_multiplier_values(real: float, expected: tuple[int, int]) -> None:
    assert nm.quantize_multiplier(real) == expected


# 2^30 - 2^-22 rounds up to a mantissa of 2^31 and so to shift 31.
@pytest.mark.parametrize("real", [-1e-9, np.inf, np.nan, 2.0**30, 2.0**30 - 2.0**-22, 1e12])
def test_quantize_multiplier_rejects(real: float) -> None:
    with pytest.raises(ValueError, match="multiplier"):
        nm.quantize_multiplier(real)


def test_multiply_by_quantized_multiplier_examples() -> None:
    mantissa, shift = nm.quantize_multiplier(0.25)
    np.testing.assert_array_equal(
        nm.multiply_by_quantized_multiplier([-6, -2, 2, 6, 1000], mantissa, shift),
        [-2, -1, 1, 2, 250],
    )
    # 4.0 is mantissa 2^30 with shift 3: the left shift saturates at int32
    # before the high multiplication halves the value, as NUMERICS.md defines.
    mantissa, shift = nm.quantize_multiplier(4.0)
    assert (mantissa, shift) == (2**30, 3)
    np.testing.assert_array_equal(
        nm.multiply_by_quantized_multiplier([3, 2**29, -(2**29)], mantissa, shift),
        [12, 2**30, -(2**30)],
    )


def test_multiply_by_quantized_multiplier_per_channel() -> None:
    params = [nm.quantize_multiplier(r) for r in (0.5, 0.125, 2.0)]
    mantissas = np.array([m for m, _ in params])[:, None]
    shifts = np.array([s for _, s in params])[:, None]
    x = np.array([[16, -16], [16, -16], [16, -16]])
    out = nm.multiply_by_quantized_multiplier(x, mantissas, shifts)
    np.testing.assert_array_equal(out, [[8, -8], [2, -2], [32, -32]])


def test_multiply_by_quantized_multiplier_rejects_shift() -> None:
    with pytest.raises(ValueError, match="shift must be between -31 and 30"):
        nm.multiply_by_quantized_multiplier([1], 2**30, 31)


def test_requantize_clamps_and_offsets() -> None:
    mantissa, shift = nm.quantize_multiplier(0.01)
    acc = np.array([-100_000, -150, -50, 0, 50, 149, 100_000])
    out = nm.requantize(acc, mantissa, shift, zero_point=-3)
    assert out.dtype == np.int8
    np.testing.assert_array_equal(out, [-128, -5, -4, -3, -2, -2, 127])
    relu6 = nm.requantize(acc, mantissa, shift, zero_point=-3, qmin=-3, qmax=60)
    np.testing.assert_array_equal(relu6, [-3, -3, -3, -3, -2, -2, 60])


@pytest.mark.parametrize(
    ("zero_point", "qmin", "qmax", "message"),
    [
        (0, -129, 127, "qmin <= qmax"),
        (0, 10, 5, "qmin <= qmax"),
        (0, -128, 128, "qmin <= qmax"),
        (-128, -127, 127, "zero point -128 outside"),
    ],
)
def test_requantize_rejects_bounds(zero_point: int, qmin: int, qmax: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        nm.requantize([0], 2**30, 0, zero_point, qmin=qmin, qmax=qmax)


def test_quantize_per_tensor_examples() -> None:
    x = np.array([-1.0, -0.125, 0.125, 0.375, 0.625, 1000.0, -1000.0])
    np.testing.assert_array_equal(nm.quantize(x, 0.25, 0), [-4, 0, 0, 2, 2, 127, -128])
    np.testing.assert_array_equal(
        nm.quantize(x, 0.25, 0, mode=RoundingMode.HALF_AWAY_FROM_ZERO),
        [-4, -1, 1, 2, 3, 127, -128],
    )
    np.testing.assert_array_equal(nm.quantize(x, 0.25, 5, qmin=-127)[-1], -127)


def test_quantize_per_channel() -> None:
    weights = np.array([[[0.5, -0.5]], [[0.5, -0.5]]])  # (2, 1, 2)
    codes = nm.quantize(weights, np.array([0.5, 0.25]), np.array([0, 0]), axis=0)
    np.testing.assert_array_equal(codes, [[[1, -1]], [[2, -2]]])
    back = nm.dequantize(codes, np.array([0.5, 0.25]), np.array([0, 0]), axis=0)
    np.testing.assert_array_equal(back, weights)


@pytest.mark.parametrize(
    ("scale", "zero_point", "axis", "message"),
    [
        (0.0, 0, None, "finite and positive"),
        (np.nan, 0, None, "finite and positive"),
        (0.1, 200, None, "zero points outside"),
        (np.array([0.1, 0.1]), 0, None, "must be scalars"),
        (np.array([[0.1, 0.1]]), np.array([0]), 0, "one-dimensional"),
    ],
)
def test_quantize_rejects_parameters(
    scale: object, zero_point: object, axis: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        nm.quantize(np.zeros((2, 2)), scale, zero_point, axis=axis)  # type: ignore[arg-type]


def test_quantize_rejects_bounds() -> None:
    with pytest.raises(ValueError, match="qmin <= qmax"):
        nm.quantize([0.0], 1.0, 0, qmin=5, qmax=4)


def test_dequantize_rejects_wide_codes() -> None:
    with pytest.raises(ValueError, match="q does not fit in int8"):
        nm.dequantize([300], 0.1, 0)
