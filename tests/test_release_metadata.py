# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-9 — the things that must be true before a release is publishable.

A published artifact cannot be edited: PyPI does not allow re-uploading a
version, so a wheel whose metadata disagrees with its code is permanent. These
checks are cheap and run on every PR, which is the only point at which they are
still free to fix.
"""

from __future__ import annotations

import re
from importlib.metadata import version as dist_version
from pathlib import Path

import pytest

import aethics_eval

# tomllib is stdlib from 3.11; this package supports 3.10, where it is not
# available and `tomli` provides the identical API. Importing tomllib
# unconditionally passes locally on a modern interpreter and fails collection
# on a third of the CI matrix.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 only
    import tomli as tomllib  # type: ignore[no-redef]

SDK_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = SDK_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_version_is_not_duplicated(pyproject: dict) -> None:
    """The version lives in exactly one place.

    It used to be written in both pyproject.toml and __init__.py, which is a
    release hazard rather than an untidiness: the two can drift, and then the
    wheel's metadata describes a different version from the code inside it.
    """
    project = pyproject["project"]

    assert "version" not in project, (
        "pyproject.toml declares a static version, which duplicates "
        'aethics_eval.__version__. Use `dynamic = ["version"]` instead.'
    )
    assert "version" in project.get("dynamic", []), (
        "pyproject.toml should declare the version as dynamic"
    )
    assert pyproject["tool"]["hatch"]["version"]["path"] == (
        "src/aethics_eval/__init__.py"
    ), "the dynamic version must be read from the module that defines it"


def test_installed_metadata_matches_module_version() -> None:
    """What the build produced matches what the code says.

    Catches a stale editable install and, more importantly, a build backend
    misconfiguration where the packaged version silently diverges.
    """
    assert dist_version("aethics-eval") == aethics_eval.__version__, (
        f"installed distribution reports {dist_version('aethics-eval')!r} but "
        f"aethics_eval.__version__ is {aethics_eval.__version__!r}. Reinstall, "
        f"or check [tool.hatch.version] in pyproject.toml."
    )


def test_version_is_pep440_and_semver_shaped() -> None:
    """A publishable version: MAJOR.MINOR.PATCH, optionally pre-release.

    PyPI accepts a lot of shapes PEP 440 permits; this project commits to
    SemVer, because the result schema's compatibility policy is expressed in
    those terms and a version that does not parse as SemVer makes that policy
    unenforceable.
    """
    assert re.fullmatch(
        r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+)?", aethics_eval.__version__
    ), (
        f"{aethics_eval.__version__!r} is not MAJOR.MINOR.PATCH with an "
        f"optional pre-release or dev suffix"
    )


def test_licence_declared_and_flagged_while_unassigned(pyproject: dict) -> None:
    """The licence field says what is actually true.

    ``NOASSERTION`` is the SPDX identifier for "no licence has been granted",
    which is the honest state today. This test does not require a licence to be
    chosen — it requires the field to not silently claim one that was never
    agreed.

    When AETHICS decides, update this test in the same PR as the LICENSE file,
    so the two cannot land apart.
    """
    declared = pyproject["project"]["license"]
    text = declared if isinstance(declared, str) else declared.get("text", "")

    assert text, "pyproject.toml must declare a licence field"
    if text == "NOASSERTION":
        pytest.skip(
            "licence still unassigned (NOASSERTION) — the release workflow "
            "refuses to publish in this state, see .github/workflows/release.yml"
        )
    assert (SDK_ROOT / "LICENSE").is_file(), (
        f"licence is declared as {text!r} but no LICENSE file ships with the "
        f"package. A declared licence with no text is worse than none."
    )


def test_readme_exists_for_the_project_description(pyproject: dict) -> None:
    """The README becomes the PyPI project page; a missing one fails the upload."""
    readme = pyproject["project"]["readme"]
    name = readme if isinstance(readme, str) else readme.get("file", "")
    assert (SDK_ROOT / name).is_file(), f"readme {name!r} is missing"


# ── The licence, once assigned, must stay assigned ──────────────────


def test_licence_is_apache_2_0_everywhere_it_is_declared(pyproject: dict) -> None:
    """One licence, four places, no drift.

    The declaration lives in pyproject metadata, in LICENSE, in the SPDX header
    machinery, and in a header on every source file. They are edited at
    different times by different people; a mismatch between them is the kind of
    thing nobody notices until a compliance scanner does.
    """
    root = Path(__file__).resolve().parent.parent

    lic = pyproject["project"]["license"]
    declared = lic if isinstance(lic, str) else lic.get("text", "")
    assert declared == "Apache-2.0", f"pyproject declares {declared!r}"

    licence_file = root / "LICENSE"
    assert licence_file.is_file(), "LICENSE is missing"
    body = licence_file.read_text(encoding="utf-8")
    assert "Apache License" in body and "Version 2.0" in body
    # The patent grant is the reason this licence was chosen over MIT.
    assert "Grant of Patent License" in body

    assert (root / "NOTICE").is_file(), "Apache-2.0 NOTICE file is missing"

    machinery = (root / "scripts" / "apply_spdx_headers.py").read_text(encoding="utf-8")
    assert 'LICENSE_ID = "Apache-2.0"' in machinery

    sample = (root / "src" / "aethics_eval" / "manifest.py").read_text(encoding="utf-8")
    assert "SPDX-License-Identifier: Apache-2.0" in sample


def test_the_osi_classifier_matches_the_licence(pyproject: dict) -> None:
    """Licence-compliance tooling reads the classifier, not the prose."""
    classifiers = pyproject["project"].get("classifiers", [])
    licence_classifiers = [c for c in classifiers if c.startswith("License ::")]
    assert licence_classifiers == ["License :: OSI Approved :: Apache Software License"]
