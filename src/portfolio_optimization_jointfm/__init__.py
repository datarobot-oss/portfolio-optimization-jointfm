# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
