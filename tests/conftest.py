# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Shared test configuration: deterministic Hypothesis profiles."""

import os

from hypothesis import HealthCheck, settings

settings.register_profile("default", derandomize=True, deadline=None, max_examples=100)
settings.register_profile(
    "thorough",
    derandomize=False,
    deadline=None,
    max_examples=2000,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
