# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Scalar reference implementations of the numerics contract.

These use Python integers and fractions, which never overflow or round, and
are written from the definitions in docs/NUMERICS.md independently of the
vectorised implementation under test.
"""

from __future__ import annotations

import math
from fractions import Fraction

from theoros_models.numerics import RoundingMode


def round_fraction(value: Fraction, mode: RoundingMode) -> int:
    """Round an exact rational value to an integer."""
    floor = math.floor(value)
    fraction = value - floor
    if mode is RoundingMode.FLOOR:
        return floor
    if mode is RoundingMode.TOWARD_ZERO:
        return floor + 1 if value < 0 and fraction else floor
    if fraction != Fraction(1, 2):
        return floor + 1 if fraction > Fraction(1, 2) else floor
    if mode is RoundingMode.HALF_EVEN:
        return floor + (floor % 2)
    if mode is RoundingMode.HALF_AWAY_FROM_ZERO:
        return floor + 1 if value > 0 else floor
    return floor + 1


def divide_pow2(x: int, shift: int, mode: RoundingMode) -> int:
    """Round x / 2^shift with integer arithmetic only."""
    divisor = 1 << shift
    quotient, remainder = divmod(x, divisor)
    if mode is RoundingMode.FLOOR or remainder == 0:
        return quotient
    if mode is RoundingMode.TOWARD_ZERO:
        return quotient + 1 if x < 0 else quotient
    twice = 2 * remainder
    if twice != divisor:
        return quotient + 1 if twice > divisor else quotient
    if mode is RoundingMode.HALF_EVEN:
        return quotient + (quotient % 2)
    if mode is RoundingMode.HALF_AWAY_FROM_ZERO:
        return quotient + 1 if x > 0 else quotient
    return quotient + 1


def srdhm(a: int, b: int) -> int:
    """Return a * b / 2^31 rounded half toward positive infinity, saturated."""
    if a == b == -(2**31):
        return 2**31 - 1
    return divide_pow2(a * b, 31, RoundingMode.HALF_UP)


def clamp(value: int, low: int, high: int) -> int:
    """Clamp an integer to an inclusive range."""
    return max(low, min(high, value))
