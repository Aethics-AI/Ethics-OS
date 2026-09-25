# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-1 step 3 — the public/proprietary boundary, enforced mechanically.

aethics_eval is a standalone evaluation library. It must never import the
FastAPI product it was extracted from, nor the proprietary compliance layer
described in §8 of the work order. A careless `from core.x import y` in a
future PR is exactly how open-sourced packages quietly re-couple to their
parent, so the contract is a test rather than a convention.

Two layers of checking:

  * test_import_linter_contract runs the real import-linter contract declared
    in pyproject.toml. That is the authoritative check.
  * test_no_forbidden_imports_in_source is a source-level grep that needs no
    third-party tooling and reports the exact offending file and line. It
    catches lazy function-level imports that would otherwise only surface at
    runtime.

The grep is not redundant: import-linter analyses a module graph built from
importable modules, so a module that cannot be imported at all is invisible
to it. During the extraction that is precisely the state the coupled modules
are in.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

# The boundary, verbatim from POS-1 step 3.
FORBIDDEN_ROOTS = ("core", "app", "routes", "auth", "supabase", "fastapi")

SDK_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = SDK_ROOT / "src" / "aethics_eval"

# `from ..anything import x` escapes the package regardless of what it names,
# so relative imports above the package root are forbidden outright.
_RELATIVE_ESCAPE = re.compile(r"^\s*from\s+\.\.")
_ABSOLUTE_FORBIDDEN = re.compile(
    r"^\s*(?:from\s+(" + "|".join(FORBIDDEN_ROOTS) + r")(?:\.|\s)"
    r"|import\s+(" + "|".join(FORBIDDEN_ROOTS) + r")(?:\.|\s|$))"
)


def _source_files() -> list[Path]:
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert files, f"no sources found under {PACKAGE_ROOT}"
    return files


def test_package_root_exists() -> None:
    """Guards against the test silently passing if the layout moves."""
    assert PACKAGE_ROOT.is_dir(), f"expected package at {PACKAGE_ROOT}"
    assert (PACKAGE_ROOT / "__init__.py").is_file()


def test_no_forbidden_imports_in_source() -> None:
    """No module may import the product, at module level or inside a function."""
    violations: list[str] = []

    for path in _source_files():
        rel = path.relative_to(SDK_ROOT)
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if line.lstrip().startswith("#"):
                continue
            if _RELATIVE_ESCAPE.match(line):
                violations.append(
                    f"{rel}:{lineno}: relative import escapes the package"
                    f" — {line.strip()}"
                )
            elif _ABSOLUTE_FORBIDDEN.match(line):
                violations.append(f"{rel}:{lineno}: {line.strip()}")

    assert not violations, (
        "aethics_eval may not import "
        + ", ".join(FORBIDDEN_ROOTS)
        + ".\nViolations:\n  "
        + "\n  ".join(violations)
    )


def _console_script(name: str) -> Path | None:
    """Locate a console script belonging to *this* interpreter's environment.

    Not simply `sys.executable.parent / name`: that is the POSIX layout, where
    the interpreter and its scripts share `bin/`. On Windows the interpreter
    sits at the environment root and scripts live in a sibling `Scripts\\`
    directory with an `.exe` suffix, so the POSIX guess misses and the test
    fails on a platform where nothing is actually wrong. That is exactly how
    this first ran on the matrix.

    Resolved against the current interpreter rather than PATH, so a
    same-named script from some other environment cannot satisfy the contract
    check. `shutil.which` is a last resort for layouts neither path covers.
    """
    names = [name, f"{name}.exe"] if sys.platform == "win32" else [name]
    exe_dir = Path(sys.executable).parent
    candidate_dirs = [
        Path(sysconfig.get_path("scripts")),
        exe_dir,
        exe_dir / "Scripts",
        exe_dir / "bin",
    ]
    for directory in candidate_dirs:
        for candidate in (directory / n for n in names):
            if candidate.is_file():
                return candidate

    found = shutil.which(name)
    return Path(found) if found else None


def test_import_linter_contract() -> None:
    """Run the contract declared in pyproject.toml [tool.importlinter].

    Invoked through the lint-imports console script. `python -m
    importlinter.cli` exits 0 without checking anything, which makes it a
    silently vacuous test — do not "simplify" this back to a -m invocation.
    """
    pytest.importorskip(
        "importlinter",
        reason="import-linter missing; install the [dev] extra",
    )

    entry_point = _console_script("lint-imports")
    # Deliberately an assert, not a skip: import-linter is present (the
    # importorskip above passed), so a missing script is a broken install, and
    # skipping would silently retire the boundary check.
    assert entry_point is not None, (
        "import-linter is installed but its lint-imports console script was "
        f"not found for {sys.executable}; install the [dev] extra"
    )

    result = subprocess.run(
        [str(entry_point)],
        cwd=SDK_ROOT,
        capture_output=True,
        text=True,
    )

    # Guard against a vacuous pass: the run must actually analyse the package
    # and report on the contract, not no-op its way to a zero exit code.
    assert "Contracts:" in result.stdout, (
        "import-linter produced no contract report — it is not actually "
        f"checking anything:\n{result.stdout}\n{result.stderr}"
    )
    assert "broken" not in result.stdout.lower() or "0 broken" in result.stdout, (
        f"import-linter contract failed:\n{result.stdout}\n{result.stderr}"
    )
    assert result.returncode == 0, (
        f"import-linter contract failed:\n{result.stdout}\n{result.stderr}"
    )


def test_public_surface_imports_cleanly() -> None:
    """`import aethics_eval` must succeed with only the core install present."""
    result = subprocess.run(
        [sys.executable, "-c", "import aethics_eval; print(aethics_eval.__version__)"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"import aethics_eval failed:\n{result.stdout}\n{result.stderr}"
    )
    assert result.stdout.strip() == "0.1.0"
