# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Nothing credential-shaped may enter the tree.

Run on every PR: the tracked tree must contain no credential files and no
secret-shaped content. The full-history check is
`python -m scripts.scan_secrets`, which scans every commit reachable from HEAD.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SDK_ROOT = Path(__file__).resolve().parent.parent
_SCANNER = SDK_ROOT / "scripts" / "scan_secrets.py"


def _load():
    spec = importlib.util.spec_from_file_location("scan_secrets", _SCANNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(
    not _SCANNER.is_file(), reason="scanner not present in this checkout"
)


def _sdk_files() -> list[Path]:
    skip = {
        "__pycache__",
        ".git",
        ".venv",
        "dist",
        "build",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
    return [
        p
        for p in SDK_ROOT.rglob("*")
        if p.is_file() and not any(part in skip for part in p.parts)
    ]


def test_no_credential_files_in_the_sdk_tree() -> None:
    mod = _load()
    bad = [
        str(p.relative_to(SDK_ROOT))
        for p in _sdk_files()
        if mod.FORBIDDEN_NAME.search(p.as_posix())
    ]
    assert not bad, f"credential files in the publishable tree: {bad}"


def test_no_secret_shaped_content_in_the_sdk_tree() -> None:
    """Reports the pattern and the path, never the matched value.

    A scanner that echoes what it found copies the credential into the CI log,
    which is one of the places it should least be.
    """
    mod = _load()
    findings: list[str] = []
    for path in _sdk_files():
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for label, rx in mod.PATTERNS.items():
            m = rx.search(content)
            if m and not mod.PLACEHOLDER.search(m.group(0)):
                findings.append(f"{label} in {path.relative_to(SDK_ROOT)}")
    assert not findings, "secret-shaped content:\n  " + "\n  ".join(findings)


def test_example_placeholders_are_not_flagged() -> None:
    """The scanner must not cry wolf on a .env.example, or it gets ignored."""
    mod = _load()
    assert mod.FORBIDDEN_NAME.search(".env") is not None
    assert mod.FORBIDDEN_NAME.search("config/.env.production") is not None
    assert mod.FORBIDDEN_NAME.search(".env.example") is None
    assert mod.FORBIDDEN_NAME.search(".env.template") is None
