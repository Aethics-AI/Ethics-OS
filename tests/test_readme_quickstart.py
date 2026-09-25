# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-7 - the README quickstart must run on the install the README just described.

The Install section offers a light core install and an optional `[local]` extra
(~2 GB, torch). The Quickstart that followed opened with

    aethics eval --model local:gpt2 --tasks crows_pairs --limit 200 -o out.json

which needs that extra. So the first command a new reader ran failed unless they
had already installed something the page presented as optional - against §0's
bar of "install it and get a number in five minutes".

This test does not run the commands (that needs network for the dataset). It
pins the property that broke: the quickstart block must not depend on an
optional extra or on credentials. `docs/quickstart.md` uses the `fake:`
provider for the same reason.
"""

from __future__ import annotations

import re
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parent.parent
README = SDK_ROOT / "README.md"

# Providers that need something beyond `pip install aethics-eval`.
_NEEDS_EXTRA = "local:"  # torch/transformers, the [local] extra
_NEEDS_CREDENTIALS = ("openai:", "hf:")


def _quickstart_block() -> str:
    """The first fenced bash block under the '## Quickstart' heading."""
    text = README.read_text(encoding="utf-8")
    start = text.index("## Quickstart")
    # Stop at the next '### ' or '## ' heading so the "then a real model"
    # subsection - which is *allowed* to need the extra - is excluded.
    rest = text[start + len("## Quickstart") :]
    end = re.search(r"^#{2,3} ", rest, re.MULTILINE)
    section = rest[: end.start()] if end else rest

    block = re.search(r"```bash\n(.*?)```", section, re.DOTALL)
    assert block, "no bash block found under '## Quickstart'"
    return block.group(1)


def test_quickstart_runs_on_the_core_install() -> None:
    block = _quickstart_block()
    assert _NEEDS_EXTRA not in block, (
        "the quickstart uses a 'local:' model, which needs the [local] extra "
        "(~2 GB). The first command in the README must work on the core "
        "install the section above it describes - move the local example under "
        "'Then a real model'."
    )


def test_quickstart_needs_no_credentials() -> None:
    block = _quickstart_block()
    used = [p for p in _NEEDS_CREDENTIALS if p in block]
    assert not used, (
        f"the quickstart uses {used}, which needs an API key. A reader with no "
        "credentials must still reach a result."
    )


def test_quickstart_actually_produces_and_checks_a_result() -> None:
    """A quickstart that only lists tasks has not shown the tool working."""
    block = _quickstart_block()
    for expected in ("aethics eval", "aethics show", "aethics verify"):
        assert expected in block, f"quickstart never runs `{expected}`"
