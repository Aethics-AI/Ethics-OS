# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Step 1 of a new user's day: install it, import it, find the command."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import aethics_eval

PUBLIC_PACKAGES = (
    "aethics_eval",
    "aethics_eval.tasks",
    "aethics_eval.results",
    "aethics_eval.aggregation",
    "aethics_eval.models",
    "aethics_eval.manifest",
    "aethics_eval.cli",
)


def test_every_documented_top_level_name_imports() -> None:
    for name in aethics_eval.__all__:
        assert hasattr(aethics_eval, name), (
            f"aethics_eval.{name} is in __all__ but missing"
        )


def test_public_subpackages_import_cleanly() -> None:
    for mod in PUBLIC_PACKAGES:
        importlib.import_module(mod)


def test_version_is_a_semver_string() -> None:
    parts = aethics_eval.__version__.split(".")
    assert len(parts) == 3 and all(p.isdigit() for p in parts)


def test_core_import_does_not_pull_torch() -> None:
    """README: 'The core install pulls no PyTorch.' Checked in a fresh process
    so another test's imports cannot mask it."""
    code = (
        "import sys, aethics_eval, aethics_eval.tasks, aethics_eval.results, "
        "aethics_eval.aggregation, aethics_eval.cli; "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False False"


def _aethics_exe() -> str | None:
    scripts = Path(sys.executable).parent
    for name in ("aethics", "aethics.exe"):
        if (scripts / name).exists():
            return str(scripts / name)
    return shutil.which("aethics")


def test_aethics_command_is_installed_and_answers_help() -> None:
    exe = _aethics_exe()
    assert exe, "the `aethics` console script was not installed"
    out = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    for cmd in ("eval", "list-tasks", "validate", "show", "verify", "cache"):
        assert cmd in out.stdout, f"`aethics --help` does not list {cmd}"


def test_python_dash_m_style_entry_point_works() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "aethics_eval.cli", "list-tasks", "--json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert '"crows_pairs"' in out.stdout
