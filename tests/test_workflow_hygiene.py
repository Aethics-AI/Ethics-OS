# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-6 - the CI configuration is itself checked.

Two defects motivated this file, both of the same kind: a gate that looked
green because nothing examined it.

  * `validity-nightly.yml` was added by VOS-6 with unpinned `@v4` / `@v5`
    actions and no `permissions:` block, while POS-6's own workflows pinned
    every action by commit SHA. Nothing compared them, so the weaker file sat
    alongside the stronger one for weeks.
  * It also ran on the same 03:00 cron as `nightly.yml`, so the repository had
    two nightly workflows and no single place to look for last night's result.

A tag like `@v4` is a moving reference: the owner can repoint it at any commit,
which is the supply-chain hole SHA pinning closes. Asserting it here means the
next workflow cannot reintroduce the hole quietly.

Skipped when `.github/` is absent, e.g. in an unpacked sdist, since workflow
files do not travel with the package.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

pytestmark = pytest.mark.skipif(
    not WORKFLOWS.is_dir(), reason="no .github/workflows here (published subtree)"
)

# `uses: owner/repo@ref` - ref is a SHA only if it is 40 hex characters.
_USES = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)
_SHA = re.compile(r"^[0-9a-f]{40}$")


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def test_every_action_is_pinned_to_a_commit_sha() -> None:
    unpinned: list[str] = []
    for path in _workflow_files():
        for ref in _USES.findall(path.read_text(encoding="utf-8")):
            if "@" not in ref:
                unpinned.append(f"{path.name}: {ref} (no ref at all)")
                continue
            _, _, rev = ref.rpartition("@")
            if not _SHA.match(rev):
                unpinned.append(f"{path.name}: {ref}")
    assert not unpinned, (
        "Actions must be pinned to a full commit SHA, not a moving tag:\n  "
        + "\n  ".join(unpinned)
    )


def test_every_workflow_declares_permissions() -> None:
    """An omitted `permissions:` inherits the repository default, which may be
    write. Declaring it is how a workflow states what it actually needs."""
    missing = [
        p.name
        for p in _workflow_files()
        if not re.search(r"^permissions:", p.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert not missing, "Workflows with no permissions block: " + ", ".join(missing)


def test_no_two_workflows_share_a_cron_schedule() -> None:
    """Two workflows on one cron is the duplication this ticket removed."""
    crons: Counter[str] = Counter()
    where: dict[str, list[str]] = {}
    for path in _workflow_files():
        for cron in re.findall(
            r"^\s*-\s*cron:\s*[\"']([^\"']+)[\"']",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        ):
            crons[cron] += 1
            where.setdefault(cron, []).append(path.name)
    clashes = [f"{c!r}: {', '.join(where[c])}" for c, n in crons.items() if n > 1]
    assert not clashes, "Workflows sharing a cron schedule:\n  " + "\n  ".join(clashes)
