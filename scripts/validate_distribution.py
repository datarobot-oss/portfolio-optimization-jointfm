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

"""Validate the source distribution and wheel before they are uploaded to PyPI.

The checks refuse what PyPI would accept but users could not install or use: a
wheel without the scenario-generator entry point that NVIDIA's package looks
up, and a dependency that pip cannot resolve from PyPI. PyPI refuses an upload
whose ``Requires-Dist`` carries a direct reference (``name @ url``), and NVIDIA's
``portfolio-optimization`` is not on PyPI, so naming it would install whatever
package claims that name there instead.
"""

from __future__ import annotations

import argparse
import configparser
import re
import tarfile
import tomllib
import zipfile
from collections.abc import Iterable
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path

ENTRY_POINT_GROUP = "portfolio_optimization.scenario_generators"
ENTRY_POINT_NAME = "jointfm"
ENTRY_POINT_TARGET = "portfolio_optimization_jointfm.generator:generate"
# Canonical names (PEP 503) of packages that users install from git, not PyPI.
FORBIDDEN_REQUIREMENTS = ("portfolio-optimization",)
FORBIDDEN_SDIST_MEMBERS = (
    ".env",
    "config.yaml",
)
FORBIDDEN_SDIST_PREFIXES = (
    "private/",
    "scripts/",
    "tests/",
)
REQUIRED_SDIST_MEMBERS = (
    "CHANGELOG.md",
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
    "src/portfolio_optimization_jointfm/__init__.py",
    "src/portfolio_optimization_jointfm/generator.py",
    "src/portfolio_optimization_jointfm/py.typed",
)
REQUIRED_WHEEL_MEMBERS = (
    "portfolio_optimization_jointfm/__init__.py",
    "portfolio_optimization_jointfm/generator.py",
    "portfolio_optimization_jointfm/py.typed",
)
# A PEP 508 requirement's name, its optional extras, and the `@` that starts a
# direct reference.
REQUIREMENT_HEAD = re.compile(
    r"\s*(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"\s*(?:\[[^\]]*\])?\s*(?P<direct_reference>@)?"
)


def main() -> None:
    """Validate the built artifacts in the configured distribution directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--dist-dir", default="dist", type=Path)
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve()
    dist_dir = _resolve_dist_dir(project_root, arguments.dist_dir)
    project_metadata = _load_project_metadata(project_root / "pyproject.toml")
    distribution_name = project_metadata["name"]
    package_version = project_metadata["version"]
    archive_name = _archive_distribution_name(distribution_name)

    wheel_path = _expect_one(
        dist_dir.glob(f"{archive_name}-{package_version}-*.whl"),
        f"wheel for {distribution_name} {package_version}",
    )
    sdist_path = _expect_one(
        dist_dir.glob(f"{archive_name}-{package_version}.tar.gz"),
        f"source distribution for {distribution_name} {package_version}",
    )

    _validate_wheel(wheel_path, archive_name, project_metadata)
    _validate_sdist(sdist_path, archive_name, package_version)
    print(f"Validated {sdist_path.name} and {wheel_path.name}")


def _resolve_dist_dir(project_root: Path, dist_dir: Path) -> Path:
    """Return ``dist_dir`` itself when absolute, else relative to the project."""
    if dist_dir.is_absolute():
        return dist_dir
    return project_root / dist_dir


def _load_project_metadata(pyproject_path: Path) -> dict[str, str]:
    """Read the ``[project]`` fields that the wheel metadata must repeat."""
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)
    project = pyproject["project"]
    license_value = project["license"]
    if isinstance(license_value, dict):
        license_value = license_value["text"]
    return {
        "name": project["name"],
        "version": project["version"],
        "description": project["description"],
        "license": license_value,
        "requires_python": project["requires-python"],
    }


def _archive_distribution_name(distribution_name: str) -> str:
    """Normalize a distribution name the way wheel and sdist file names spell it."""
    return re.sub(r"[-_.]+", "_", distribution_name).lower()


def _canonical_name(name: str) -> str:
    """Normalize a project name as PEP 503 compares names."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _expect_one(paths: Iterable[Path], description: str) -> Path:
    """Return the only path in ``paths``; exit when there is none or several."""
    matches = sorted(paths)
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one {description}, found {len(matches)}")
    return matches[0]


def _validate_wheel(
    wheel_path: Path,
    archive_name: str,
    project_metadata: dict[str, str],
) -> None:
    """Check the wheel's members, metadata, entry point, and typed marker."""
    package_version = project_metadata["version"]
    dist_info_dir = f"{archive_name}-{package_version}.dist-info"
    required_members = {
        *REQUIRED_WHEEL_MEMBERS,
        f"{dist_info_dir}/METADATA",
        f"{dist_info_dir}/RECORD",
        f"{dist_info_dir}/WHEEL",
        f"{dist_info_dir}/entry_points.txt",
    }
    with zipfile.ZipFile(wheel_path) as wheel_archive:
        wheel_members = set(wheel_archive.namelist())
        _require_members(wheel_members, required_members, wheel_path.name)
        _validate_wheel_metadata(
            wheel_archive.read(f"{dist_info_dir}/METADATA"),
            project_metadata,
        )
        _validate_wheel_entry_points(
            wheel_archive.read(f"{dist_info_dir}/entry_points.txt").decode("utf-8"),
        )
        typed_marker = wheel_archive.read(
            "portfolio_optimization_jointfm/py.typed"
        ).decode("utf-8")
        _require(typed_marker.strip() != "", "wheel py.typed marker must not be empty")


def _validate_wheel_metadata(
    metadata_bytes: bytes,
    project_metadata: dict[str, str],
) -> None:
    """Check that the wheel's core metadata repeats ``pyproject.toml``."""
    metadata = _parse_metadata(metadata_bytes)
    _require(
        metadata["Name"] == project_metadata["name"], "wheel metadata name mismatch"
    )
    _require(
        metadata["Version"] == project_metadata["version"],
        "wheel metadata version mismatch",
    )
    _require(
        metadata["Summary"] == project_metadata["description"],
        "wheel metadata summary mismatch",
    )
    _require(
        metadata["Requires-Python"] == project_metadata["requires_python"],
        "wheel metadata Python requirement mismatch",
    )
    license_value = metadata.get("License-Expression") or metadata.get("License")
    _require(
        license_value == project_metadata["license"], "wheel metadata license mismatch"
    )
    classifiers = metadata.get_all("Classifier", [])
    _require(
        "Typing :: Typed" in classifiers, "wheel metadata must advertise typed package"
    )
    _validate_requirements(metadata.get_all("Requires-Dist", []), "wheel")


def _parse_metadata(metadata_bytes: bytes) -> EmailMessage:
    """Parse core metadata (``METADATA`` or ``PKG-INFO``) into its header fields."""
    return BytesParser(policy=policy.default).parsebytes(metadata_bytes)


def _validate_requirements(requirements: Iterable[str], artifact: str) -> None:
    """Refuse every ``Requires-Dist`` entry that pip cannot install from PyPI.

    Args:
        requirements: The ``Requires-Dist`` values, as PEP 508 strings.
        artifact: ``wheel`` or ``sdist``, named in the error message.

    Raises:
        SystemExit: An entry is not a PEP 508 requirement, carries a direct
            reference (``name @ url``), or names a package of
            ``FORBIDDEN_REQUIREMENTS``.
    """
    for requirement in requirements:
        head = REQUIREMENT_HEAD.match(requirement)
        _require(
            head is not None,
            f"{artifact} metadata has an unparsable requirement: {requirement}",
        )
        _require(
            head["direct_reference"] is None,
            f"{artifact} metadata has a direct-reference requirement, which PyPI "
            f"refuses: {requirement}",
        )
        _require(
            _canonical_name(head["name"]) not in FORBIDDEN_REQUIREMENTS,
            f"{artifact} metadata requires a package that is not installed from "
            f"PyPI: {requirement}",
        )


def _validate_wheel_entry_points(entry_points_text: str) -> None:
    """Check that the wheel declares the scenario generator NVIDIA looks up."""
    entry_points = configparser.ConfigParser()
    entry_points.read_string(entry_points_text)
    _require(
        entry_points.get(ENTRY_POINT_GROUP, ENTRY_POINT_NAME, fallback=None)
        == ENTRY_POINT_TARGET,
        f"wheel must declare {ENTRY_POINT_NAME} = {ENTRY_POINT_TARGET} in the "
        f"entry-point group {ENTRY_POINT_GROUP}",
    )


def _validate_sdist(sdist_path: Path, archive_name: str, package_version: str) -> None:
    """Check the sdist's members, its forbidden files, and its dependencies."""
    root_prefix = f"{archive_name}-{package_version}/"
    required_members = {f"{root_prefix}{member}" for member in REQUIRED_SDIST_MEMBERS}
    with tarfile.open(sdist_path, "r:gz") as sdist_archive:
        sdist_members = set(sdist_archive.getnames())
        _require_members(sdist_members, required_members, sdist_path.name)
        pkg_info = sdist_archive.extractfile(f"{root_prefix}PKG-INFO")
        _require(pkg_info is not None, "sdist PKG-INFO is not a regular file")
        pkg_info_metadata = _parse_metadata(pkg_info.read())
    _validate_requirements(pkg_info_metadata.get_all("Requires-Dist", []), "sdist")
    for member in sdist_members:
        _require(
            member == root_prefix.rstrip("/") or member.startswith(root_prefix),
            f"sdist member outside archive root: {member}",
        )
        relative_member = member.removeprefix(root_prefix)
        _require(
            relative_member not in FORBIDDEN_SDIST_MEMBERS,
            f"sdist includes forbidden member: {relative_member}",
        )
        _require(
            not any(
                relative_member.startswith(prefix)
                for prefix in FORBIDDEN_SDIST_PREFIXES
            ),
            f"sdist includes forbidden member: {relative_member}",
        )


def _require_members(
    found_members: set[str],
    required_members: set[str],
    artifact_name: str,
) -> None:
    """Exit naming every required member that the artifact lacks."""
    missing_members = sorted(required_members - found_members)
    _require(
        not missing_members,
        f"{artifact_name} is missing required members: {', '.join(missing_members)}",
    )


def _require(condition: bool, message: str) -> None:
    """Exit with ``message`` when ``condition`` does not hold."""
    if not condition:
        raise SystemExit(message)


if __name__ == "__main__":
    main()
