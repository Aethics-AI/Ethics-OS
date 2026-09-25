# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Scan a git ref's full history for credentials.

    python -m scripts.scan_secrets                    # everything reachable from HEAD
    python -m scripts.scan_secrets --ref <ref>        # any ref
    python -m scripts.scan_secrets --tree-only        # working tree, no history

Detection only. Matched values are never printed: a scanner that echoes the
credential it found copies it into your terminal scrollback, your CI log, and
whatever ships those logs onward. Paths and pattern names are enough to act on.

Exit codes: 0 clean, 1 findings, 2 usage.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Dict, List, Set, Tuple

# Files that must never reach a public repository, whatever they contain.
FORBIDDEN_NAME = re.compile(
    r"(^|/)(\.env(\.(?!example|sample|template)[^/]*)?"
    r"|.*\.pem|.*\.p12|.*\.pfx|id_rsa|id_dsa|id_ecdsa|id_ed25519"
    r"|.*\.keystore|credentials\.json|service[-_]account.*\.json)$",
    re.I,
)

# Credential-shaped content.
PATTERNS: Dict[str, re.Pattern] = {
    "AWS access key id": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "Slack token": re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    "Stripe live secret key": re.compile(r"sk_live_[A-Za-z0-9]{20,}"),
    "Stripe webhook secret": re.compile(r"whsec_[A-Za-z0-9]{20,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9]{32,}"),
    "HuggingFace token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "JWT": re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}"),
    "Private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "Assigned secret env var": re.compile(
        r"(SERVICE_KEY|SECRET_KEY|JWT_SECRET|ENCRYPTION_KEY|API_KEY|ACCESS_TOKEN)"
        r"\s*[:=]\s*['\"]?[A-Za-z0-9/+_.-]{24,}"
    ),
}

# Placeholders in example files are not secrets.
PLACEHOLDER = re.compile(
    r"your[-_]?|xxx|placeholder|changeme|<[^>]+>|example|dummy|redacted|\.\.\.",
    re.I,
)


def _git(*args: str) -> str:
    # Fixed argv, no shell, arguments are git refs and object ids produced by
    # git itself. Resolved from PATH deliberately: this must run wherever the
    # publisher runs it, including CI.
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout


def _objects(ref: str) -> List[Tuple[str, str]]:
    out = []
    for line in _git("rev-list", ref, "--objects").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            out.append((parts[0], parts[1]))
    return out


def scan(ref: str, tree_only: bool = False) -> Tuple[List[str], Dict[str, Set[str]]]:
    """Return (forbidden paths, {pattern name -> paths})."""
    if tree_only:
        paths = [p for p in _git("ls-files").splitlines() if p]
        pairs = [("", p) for p in paths]
    else:
        pairs = _objects(ref)

    forbidden = sorted({p for _, p in pairs if FORBIDDEN_NAME.search(p)})
    hits: Dict[str, Set[str]] = {}

    for sha, path in pairs:
        content = (
            open(path, encoding="utf-8", errors="replace").read()
            if tree_only
            else (
                _git("cat-file", "-p", sha)
                if _git("cat-file", "-t", sha).strip() == "blob"
                else ""
            )
        )
        if not content:
            continue
        for label, rx in PATTERNS.items():
            m = rx.search(content)
            if m and not PLACEHOLDER.search(m.group(0)):
                hits.setdefault(label, set()).add(path)

    return forbidden, hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--tree-only", action="store_true")
    args = ap.parse_args()

    forbidden, hits = scan(args.ref, args.tree_only)
    where = "working tree" if args.tree_only else f"full history of {args.ref}"
    print(f"scanned: {where}")

    if not forbidden and not hits:
        print("clean — no credential files, no secret-shaped content.")
        return 0

    if forbidden:
        print(f"\nforbidden files ({len(forbidden)}):")
        for p in forbidden:
            print(f"  {p}")
    if hits:
        print("\nsecret-shaped content:")
        for label, paths in sorted(hits.items()):
            print(f"  {label}: {', '.join(sorted(paths))}")
    print("\nDo not publish this ref. Values are withheld deliberately.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
