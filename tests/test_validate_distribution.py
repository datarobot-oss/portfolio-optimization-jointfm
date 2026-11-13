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

"""The build check refuses a distribution that users could not install or use."""

import zipfile

import pytest
import validate_distribution

PROJECT_METADATA = {
    "name": "portfolio-optimization-jointfm",
    "version": "0.1.0",
    "description": "JointFM scenario generator",
    "license": "Apache-2.0",
    "requires_python": ">=3.11",
}
ENTRY_POINTS = (
    "[portfolio_optimization.scenario_generators]\n"
    "jointfm = portfolio_optimization_jointfm.generator:generate\n"
)


def _metadata(*requirements):
    """Return wheel core metadata that matches ``PROJECT_METADATA``."""
    lines = [
        "Metadata-Version: 2.4",
        f"Name: {PROJECT_METADATA['name']}",
        f"Version: {PROJECT_METADATA['version']}",
        f"Summary: {PROJECT_METADATA['description']}",
        f"License-Expression: {PROJECT_METADATA['license']}",
        "Classifier: Typing :: Typed",
        f"Requires-Python: {PROJECT_METADATA['requires_python']}",
        *(f"Requires-Dist: {requirement}" for requirement in requirements),
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_wheel(path, metadata, entry_points=ENTRY_POINTS):
    """Write a wheel with every required member and the given metadata."""
    dist_info = "portfolio_optimization_jointfm-0.1.0.dist-info"
    with zipfile.ZipFile(path, "w") as wheel:
        for member in validate_distribution.REQUIRED_WHEEL_MEMBERS:
            wheel.writestr(member, "PEP 561 typed package marker.\n")
        wheel.writestr(f"{dist_info}/METADATA", metadata)
        wheel.writestr(f"{dist_info}/RECORD", "")
        wheel.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\n")
        wheel.writestr(f"{dist_info}/entry_points.txt", entry_points)
    return path


@pytest.mark.parametrize(
    "requirement",
    [
        "jointfm-client==0.16.*",
        "numpy>=2.0",
        "pandas[performance]>=2.0",
        'pydantic>=2.12; python_version >= "3.11"',
    ],
)
def test_pypi_requirements_pass(requirement):
    """Plain PyPI requirements, with extras or markers, pass the check."""
    validate_distribution._validate_requirements([requirement], "wheel")


@pytest.mark.parametrize(
    "requirement",
    [
        "portfolio-optimization",
        "portfolio_optimization>=26.6",
        "Portfolio.Optimization[cuda13]",
    ],
)
def test_nvidia_package_is_refused_in_any_spelling(requirement):
    """NVIDIA's package is not on PyPI, so no spelling of its name may be required."""
    with pytest.raises(SystemExit, match="not installed from PyPI"):
        validate_distribution._validate_requirements([requirement], "wheel")


@pytest.mark.parametrize(
    "requirement",
    [
        "portfolio-optimization @ git+https://github.com/example/repo.git@abc123",
        "numpy@https://example.com/numpy-2.0.tar.gz",
        'pandas[performance] @ file:///tmp/pandas ; python_version >= "3.11"',
    ],
)
def test_direct_reference_is_refused(requirement):
    """PyPI refuses an upload whose requirement points at a URL instead of an index."""
    with pytest.raises(SystemExit, match="direct-reference"):
        validate_distribution._validate_requirements([requirement], "wheel")


def test_wheel_with_forbidden_requirement_fails(tmp_path):
    """The wheel check reads Requires-Dist from the built wheel's METADATA."""
    wheel = _write_wheel(
        tmp_path / "package.whl", _metadata("numpy>=2.0", "portfolio-optimization")
    )
    with pytest.raises(SystemExit, match="not installed from PyPI"):
        validate_distribution._validate_wheel(
            wheel, "portfolio_optimization_jointfm", PROJECT_METADATA
        )


def test_wheel_with_pypi_requirements_passes(tmp_path):
    """A wheel with the entry point and only PyPI requirements passes."""
    wheel = _write_wheel(
        tmp_path / "package.whl", _metadata("jointfm-client==0.16.*", "numpy>=2.0")
    )
    validate_distribution._validate_wheel(
        wheel, "portfolio_optimization_jointfm", PROJECT_METADATA
    )


@pytest.mark.parametrize(
    "entry_points",
    [
        "[console_scripts]\n"
        "jointfm = portfolio_optimization_jointfm.generator:generate\n",
        "[portfolio_optimization.scenario_generators]\n"
        "jointfm = portfolio_optimization_jointfm.sampler:generate\n",
    ],
)
def test_wheel_without_the_generator_entry_point_fails(tmp_path, entry_points):
    """NVIDIA's registry finds the generator only through its group, name, target."""
    wheel = _write_wheel(tmp_path / "package.whl", _metadata(), entry_points)
    with pytest.raises(SystemExit, match="entry-point group"):
        validate_distribution._validate_wheel(
            wheel, "portfolio_optimization_jointfm", PROJECT_METADATA
        )
