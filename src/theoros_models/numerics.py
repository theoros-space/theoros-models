# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Fixed-point and int8 arithmetic shared bit-exactly with theoros-edge.

Every function here is the executable definition of an operation in
docs/NUMERICS.md (MOD-REQ-040 to MOD-REQ-044). Integer results are computed
in int64 so that no intermediate overflows; inputs that do not fit their
declared width are rejected rather than wrapped.
"""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

INT8_MIN, INT8_MAX = -(2**7), 2**7 - 1
INT16_MIN, INT16_MAX = -(2**15), 2**15 - 1
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
MAX_SHIFT = 62
MULTIPLIER_MIN_SHIFT = -31
MULTIPLIER_MAX_SHIFT = 30

_RANGES = {8: (INT8_MIN, INT8_MAX), 16: (INT16_MIN, INT16_MAX), 32: (INT32_MIN, INT32_MAX)}
_INT64_LIMIT = 2.0**63
_HALF = 0.5


class RoundingMode(StrEnum):
    """How a value exactly halfway between two integers, or any fraction, is rounded."""

    HALF_EVEN = "half_even"
    """Nearest integer, ties to the even neighbour."""
    HALF_AWAY_FROM_ZERO = "half_away_from_zero"
    """Nearest integer, ties away from zero."""
    HALF_UP = "half_up"
    """Nearest integer, ties toward positive infinity."""
    TOWARD_ZERO = "toward_zero"
    """Truncation: discard the fraction."""
    FLOOR = "floor"
    """Largest integer not above the value."""


def int_range(bits: int) -> tuple[int, int]:
    """Return the inclusive range of a signed two's complement integer.

    Args:
        bits: Width, one of 8, 16 or 32.

    Returns:
        ``(minimum, maximum)``.

    Raises:
        ValueError: For any other width.
    """
    try:
        return _RANGES[bits]
    except KeyError:
        raise ValueError(f"unsupported integer width {bits}; use 8, 16 or 32") from None


def _as_int(x: ArrayLike, bits: int, name: str) -> IntArray:
    values = np.asarray(x)
    if values.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array, got dtype {values.dtype}")
    out = values.astype(np.int64)
    low, high = int_range(bits)
    if out.size and (out.min() < low or out.max() > high):
        raise ValueError(f"{name} does not fit in int{bits}")
    return out


def saturate(x: ArrayLike, bits: int = 8) -> IntArray:
    """Clamp integers to a signed range (MOD-REQ-041).

    Args:
        x: Integers of any shape and integer dtype up to int64.
        bits: Target width, 8, 16 or 32.

    Returns:
        ``x`` clamped to the range of the target width, int64, same shape.
    """
    values = np.asarray(x)
    if values.dtype.kind not in "iu":
        raise TypeError(f"x must be an integer array, got dtype {values.dtype}")
    low, high = int_range(bits)
    return np.clip(values.astype(np.int64), low, high)


def saturating_add(a: ArrayLike, b: ArrayLike, bits: int = 8) -> IntArray:
    """Add two signed integers of a width and saturate (MOD-REQ-041).

    Args:
        a: Operands that fit the width, broadcastable with ``b``.
        b: Operands that fit the width.
        bits: Width, 8, 16 or 32.

    Returns:
        ``a + b`` clamped to the width, int64.
    """
    return saturate(_as_int(a, bits, "a") + _as_int(b, bits, "b"), bits)


def saturating_sub(a: ArrayLike, b: ArrayLike, bits: int = 8) -> IntArray:
    """Subtract two signed integers of a width and saturate (MOD-REQ-041).

    Args:
        a: Minuends that fit the width, broadcastable with ``b``.
        b: Subtrahends that fit the width.
        bits: Width, 8, 16 or 32.

    Returns:
        ``a - b`` clamped to the width, int64.
    """
    return saturate(_as_int(a, bits, "a") - _as_int(b, bits, "b"), bits)


def round_to_int(x: ArrayLike, mode: RoundingMode = RoundingMode.HALF_EVEN) -> IntArray:
    """Round floating-point values to integers exactly (MOD-REQ-040).

    Rounding works on the magnitude: for a non-negative value ``a`` the
    fraction ``a - floor(a)`` is exact in binary floating point, so ties are
    detected without the error that ``x + 0.5`` or ``x - floor(x)`` for
    negative ``x`` introduce near half-integers.

    Args:
        x: Finite values with magnitude below 2^63, any shape.
        mode: Rounding mode.

    Returns:
        Rounded values, int64, same shape.

    Raises:
        ValueError: If any value is not finite or is out of the int64 range.
    """
    values = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("cannot round non-finite values")
    if values.size and np.max(np.abs(values)) >= _INT64_LIMIT:
        raise ValueError("values out of the int64 range")
    if mode is RoundingMode.FLOOR:
        return np.floor(values).astype(np.int64)
    if mode is RoundingMode.TOWARD_ZERO:
        return np.trunc(values).astype(np.int64)
    negative = values < 0.0
    magnitude = np.abs(values)
    whole = np.floor(magnitude)
    fraction = magnitude - whole
    base = whole.astype(np.int64)
    tie = fraction == _HALF
    if mode is RoundingMode.HALF_EVEN:
        tie_up = (base % 2) == 1
    elif mode is RoundingMode.HALF_AWAY_FROM_ZERO:
        tie_up = np.ones_like(tie)
    else:
        tie_up = ~negative
    rounded = base + ((fraction > _HALF) | (tie & tie_up))
    return np.asarray(np.where(negative, -rounded, rounded), dtype=np.int64)


def rounding_shift_right(
    x: ArrayLike, shift: ArrayLike, mode: RoundingMode = RoundingMode.HALF_AWAY_FROM_ZERO
) -> IntArray:
    """Divide integers by a power of two with exact rounding (MOD-REQ-040).

    Args:
        x: Integers of any shape whose magnitude is below 2^62.
        shift: Exponents from 0 to 62, broadcastable with ``x``.
        mode: Rounding mode applied to the exact quotient ``x / 2^shift``.

    Returns:
        The rounded quotient, int64, broadcast shape.

    Raises:
        ValueError: If a shift is out of range.
    """
    values = np.asarray(x)
    if values.dtype.kind not in "iu":
        raise TypeError(f"x must be an integer array, got dtype {values.dtype}")
    values = values.astype(np.int64)
    shifts = np.asarray(shift, dtype=np.int64)
    if shifts.size and (shifts.min() < 0 or shifts.max() > MAX_SHIFT):
        raise ValueError(f"shift must be between 0 and {MAX_SHIFT}")
    quotient = np.right_shift(values, shifts)
    remainder = values - np.left_shift(quotient, shifts)
    if mode is RoundingMode.FLOOR:
        return np.asarray(quotient, dtype=np.int64)
    if mode is RoundingMode.TOWARD_ZERO:
        return np.asarray(quotient + ((remainder != 0) & (values < 0)), dtype=np.int64)
    shifted = shifts > 0
    half = np.left_shift(np.int64(1), np.maximum(shifts - 1, 0))
    above = shifted & (remainder > half)
    tie = shifted & (remainder == half)
    if mode is RoundingMode.HALF_EVEN:
        tie_up = (quotient % 2) == 1
    elif mode is RoundingMode.HALF_AWAY_FROM_ZERO:
        tie_up = values >= 0
    else:
        tie_up = np.ones_like(tie)
    return np.asarray(quotient + (above | (tie & tie_up)), dtype=np.int64)


def quantize_multiplier(real: float) -> tuple[int, int]:
    """Represent a positive real multiplier in Q31 fixed point (MOD-REQ-042).

    The result satisfies ``real ~= mantissa * 2^(shift - 31)`` with the
    mantissa from 2^30 to 2^31 - 1, so the relative error is at most 2^-31.
    Multipliers below 2^-32 flush to ``(0, 0)``, which requantises every
    accumulator to the output zero point.

    Multipliers from 2^30 - 2^-2 upward round to a shift of 31 and are rejected.

    Args:
        real: Multiplier, finite, 0 or from 2^-32 to below 2^30 - 2^-2.

    Returns:
        ``(mantissa, shift)``.

    Raises:
        ValueError: If the multiplier is negative, not finite or too large.
    """
    if not math.isfinite(real) or real < 0.0:
        raise ValueError(f"multiplier must be finite and non-negative, got {real!r}")
    if real == 0.0:
        return 0, 0
    fraction, exponent = math.frexp(real)
    mantissa = int(round_to_int(fraction * 2.0**31, RoundingMode.HALF_AWAY_FROM_ZERO))
    if mantissa == 2**31:
        mantissa //= 2
        exponent += 1
    if exponent > MULTIPLIER_MAX_SHIFT:
        raise ValueError(f"multiplier {real!r} is not below 2^{MULTIPLIER_MAX_SHIFT}")
    if exponent < MULTIPLIER_MIN_SHIFT:
        return 0, 0
    return mantissa, exponent


def saturating_rounding_doubling_high_mul(a: ArrayLike, b: ArrayLike) -> IntArray:
    """Return ``a * b / 2^31`` rounded to nearest, the gemmlowp high multiply (MOD-REQ-043).

    The nudge of ``1 - 2^30`` for negative products combined with division
    truncating toward zero rounds ties toward positive infinity for both
    signs. The only overflowing input, ``a = b = -2^31``, saturates to
    2^31 - 1.

    Args:
        a: int32 values, broadcastable with ``b``.
        b: int32 values.

    Returns:
        int32 results stored as int64, broadcast shape.
    """
    a64 = _as_int(a, 32, "a")
    b64 = _as_int(b, 32, "b")
    overflow = (a64 == INT32_MIN) & (b64 == INT32_MIN)
    product = a64 * b64
    nudge = np.where(product >= 0, np.int64(1 << 30), np.int64(1 - (1 << 30)))
    numerator = product + nudge
    high = np.where(numerator >= 0, numerator >> 31, -((-numerator) >> 31))
    return np.where(overflow, np.int64(INT32_MAX), high)


def multiply_by_quantized_multiplier(
    x: ArrayLike, mantissa: ArrayLike, shift: ArrayLike
) -> IntArray:
    """Scale int32 values by a Q31 multiplier (MOD-REQ-043).

    A positive shift first multiplies ``x`` by ``2^shift`` with int32
    saturation; the high multiplication follows; a negative shift then
    divides by ``2^-shift`` rounding half away from zero.

    Args:
        x: int32 values.
        mantissa: Q31 mantissas from :func:`quantize_multiplier`, broadcastable.
        shift: Shifts from :func:`quantize_multiplier`, broadcastable.

    Returns:
        Scaled int32 values stored as int64, broadcast shape.
    """
    values = _as_int(x, 32, "x")
    shifts = np.asarray(shift, dtype=np.int64)
    if shifts.size and (shifts.min() < MULTIPLIER_MIN_SHIFT or shifts.max() > MULTIPLIER_MAX_SHIFT):
        raise ValueError(f"shift must be between {MULTIPLIER_MIN_SHIFT} and {MULTIPLIER_MAX_SHIFT}")
    left = np.maximum(shifts, 0)
    right = np.maximum(-shifts, 0)
    scaled = saturate(np.left_shift(values, left), 32)
    high = saturating_rounding_doubling_high_mul(scaled, mantissa)
    return rounding_shift_right(high, right, RoundingMode.HALF_AWAY_FROM_ZERO)


def requantize(
    accumulator: ArrayLike,
    mantissa: ArrayLike,
    shift: ArrayLike,
    zero_point: int,
    *,
    qmin: int = INT8_MIN,
    qmax: int = INT8_MAX,
) -> NDArray[np.int8]:
    """Requantise int32 accumulators to int8 (MOD-REQ-043).

    Computes ``clamp(multiply_by_quantized_multiplier(acc) + zero_point,
    qmin, qmax)``. Activation functions that clip, such as ReLU6, are folded
    into ``qmin`` and ``qmax``.

    Args:
        accumulator: int32 accumulators, for example (C, H, W).
        mantissa: Q31 mantissas, per tensor or broadcastable per channel.
        shift: Shifts matching ``mantissa``.
        zero_point: Output zero point within ``[qmin, qmax]``.
        qmin: Lowest output value, at least -128.
        qmax: Highest output value, at most 127.

    Returns:
        int8 values, broadcast shape.
    """
    _check_code_range(qmin, qmax)
    if not qmin <= zero_point <= qmax:
        raise ValueError(f"zero point {zero_point} outside [{qmin}, {qmax}]")
    scaled = multiply_by_quantized_multiplier(accumulator, mantissa, shift)
    return np.clip(scaled + zero_point, qmin, qmax).astype(np.int8)


def _check_code_range(qmin: int, qmax: int) -> None:
    if not INT8_MIN <= qmin <= qmax <= INT8_MAX:
        raise ValueError(f"need -128 <= qmin <= qmax <= 127, got {qmin} and {qmax}")


def _per_axis(values: ArrayLike, ndim: int, axis: int | None) -> NDArray[np.generic]:
    array = np.asarray(values)
    if axis is None:
        if array.ndim != 0:
            raise ValueError("per-tensor parameters must be scalars; pass axis for per-channel")
        return array
    if array.ndim != 1:
        raise ValueError("per-channel parameters must be one-dimensional")
    shape = [1] * ndim
    shape[axis] = array.shape[0]
    return array.reshape(shape)


def quantize(
    x: ArrayLike,
    scale: ArrayLike,
    zero_point: ArrayLike,
    axis: int | None = None,
    *,
    qmin: int = INT8_MIN,
    qmax: int = INT8_MAX,
    mode: RoundingMode = RoundingMode.HALF_EVEN,
) -> NDArray[np.int8]:
    """Quantise real values to affine int8 (MOD-REQ-044).

    Computes ``clamp(round(x / scale) + zero_point, qmin, qmax)``. Weights
    use ``qmin = -127`` and zero point 0 for a symmetric range.

    Args:
        x: Real values, any shape.
        scale: Positive step size, a scalar or, with ``axis``, shape (C,).
        zero_point: Integer zero point, a scalar or shape (C,).
        axis: Channel axis of ``x`` for per-channel parameters, or None.
        qmin: Lowest code, at least -128.
        qmax: Highest code, at most 127.
        mode: Rounding mode for ``x / scale``.

    Returns:
        int8 codes, same shape as ``x``.
    """
    values = np.asarray(x, dtype=np.float64)
    scales = _per_axis(scale, values.ndim, axis).astype(np.float64)
    zeros = _per_axis(zero_point, values.ndim, axis).astype(np.int64)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("scales must be finite and positive")
    _check_code_range(qmin, qmax)
    if np.any(zeros < qmin) or np.any(zeros > qmax):
        raise ValueError(f"zero points outside [{qmin}, {qmax}]")
    limit = float(INT8_MAX - INT8_MIN + 1)
    ratio = np.clip(values / scales, -limit - 1.0, limit + 1.0)
    codes = np.clip(round_to_int(ratio, mode) + zeros, qmin, qmax)
    return np.asarray(codes, dtype=np.int8)


def dequantize(
    q: ArrayLike, scale: ArrayLike, zero_point: ArrayLike, axis: int | None = None
) -> FloatArray:
    """Map affine int8 codes back to real values (MOD-REQ-044).

    Args:
        q: Integer codes, any shape.
        scale: Step size, a scalar or, with ``axis``, shape (C,).
        zero_point: Zero point, a scalar or shape (C,).
        axis: Channel axis for per-channel parameters, or None.

    Returns:
        ``scale * (q - zero_point)`` as float64, same shape as ``q``.
    """
    codes = _as_int(q, 8, "q")
    scales = _per_axis(scale, codes.ndim, axis).astype(np.float64)
    zeros = _per_axis(zero_point, codes.ndim, axis).astype(np.int64)
    return np.asarray(scales * (codes - zeros), dtype=np.float64)
