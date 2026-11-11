# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JointFM as a scenario generator for NVIDIA's portfolio-optimization package."""

from .generator import generate
from .sampler import (
    generate_horizon_samples_jointfm,
    generate_samples_and_mean_jointfm,
    generate_samples_jointfm,
)
from .settings import JointFMSettings

__all__ = [
    "JointFMSettings",
    "generate",
    "generate_horizon_samples_jointfm",
    "generate_samples_and_mean_jointfm",
    "generate_samples_jointfm",
]
