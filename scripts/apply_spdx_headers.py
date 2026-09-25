#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Apply or update SPDX headers across the package.

POS-2 asks for the header machinery to exist before the licence is chosen, so
that setting it later is a one-line change rather than a hand edit of every
file. The identifier lives in one constant below.

Until the project lead confirms the licence (§7 — Apache-2.0 is the standing
confirmed as Apache-2.0), LICENSE_ID was "NOASSERTION" — the SPDX-registered way
of saying "deliberately not stated" rather than implying a licence we have not
agreed. Change the constant, re-run, done.

    python scripts/apply_spdx_headers.py --check    # CI: fail if any missing
    python scripts/apply_spdx_headers.py            # write headers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Confirmed: Apache-2.0 (§7, for the patent grant) ────────────────
LICENSE_ID = "Apache-2.0"
COPYRIGHT = "Copyright (c) 2026 AETHICS"

SDK_ROOT = Path(__file__).resolve().parent.parent
# scripts/ is included deliberately. It was omitted originally, which meant
# `--check` reported "headers present in all 65 files" while three files in
# this very directory had none — a gate that passes by not looking. The
# scripts ship in the public repository even though they are not in the wheel,
# so they carry the licence header like everything else.
TARGETS = [
    SDK_ROOT / "src" / "aethics_eval",
    SDK_ROOT / "tests",
    SDK_ROOT / "scripts",
]

SPDX_COPYRIGHT_PREFIX = "# SPDX-FileCopyrightText:"
SPDX_LICENSE_PREFIX = "# SPDX-License-Identifier:"


def header_lines() -> list[str]:
    return [
        f"{SPDX_COPYRIGHT_PREFIX} {COPYRIGHT}",
        f"{SPDX_LICENSE_PREFIX} {LICENSE_ID}",
    ]


def _python_files() -> list[Path]:
    files: list[Path] = []
    for target in TARGETS:
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
    return files


def _split_shebang(lines: list[str]) -> tuple[list[str], list[str]]:
    """A shebang must stay on line 1, so headers go after it."""
    if lines and lines[0].startswith("#!"):
        return lines[:1], lines[1:]
    return [], lines


def _strip_existing(lines: list[str]) -> list[str]:
    """Drop any SPDX lines already present, so the script is idempotent."""
    return [
        ln
        for ln in lines
        if not ln.startswith((SPDX_COPYRIGHT_PREFIX, SPDX_LICENSE_PREFIX))
    ]


def process(path: Path, check_only: bool) -> bool:
    """Return True if the file already has the correct header."""
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()

    wanted = header_lines()
    if all(w in lines[:5] for w in wanted):
        return True
    if check_only:
        return False

    shebang, rest = _split_shebang(lines)
    rest = _strip_existing(rest)
    # Leave a blank line between the header and whatever follows.
    while rest and not rest[0].strip():
        rest.pop(0)

    new = shebang + wanted + [""] + rest
    path.write_text("\n".join(new) + "\n", encoding="utf-8")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files missing the header and exit non-zero; write nothing",
    )
    args = parser.parse_args()

    files = _python_files()
    if not files:
        print(f"no Python files found under {[str(t) for t in TARGETS]}")
        return 1

    missing = [p for p in files if not process(p, args.check)]

    if args.check:
        if missing:
            print(f"Missing or stale SPDX header ({len(missing)} files):")
            for p in missing:
                print(f"  {p.relative_to(SDK_ROOT)}")
            print(f"\nRun: python {Path(__file__).relative_to(SDK_ROOT)}")
            return 1
        print(f"SPDX headers present and current in all {len(files)} files.")
        return 0

    if missing:
        print(f"Applied SPDX header ({LICENSE_ID}) to {len(missing)} files.")
    else:
        print(f"All {len(files)} files already current — nothing to do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
