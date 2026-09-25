# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Print the task registry's metadata: what ``aethics list-tasks`` renders from
``list_tasks()``.

    python -m scripts.list_tasks
"""

from __future__ import annotations

from aethics_eval.tasks import list_tasks


def render() -> str:
    metas = list_tasks()
    out = ["# aethics list-tasks\n"]
    out.append(f"{len(metas)} registered task(s).\n")
    out.append("| task | metric | logprobs | licence | citation |")
    out.append("|------|--------|----------|---------|----------|")
    for m in metas:
        out.append(
            f"| `{m.name}` | {m.metric} | {'yes' if m.requires_logprobs else 'no'} "
            f"| {m.licence or '—'} | {m.citation or '—'} |"
        )
    out.append("")
    for m in metas:
        out.append(f"- **{m.name}** — {m.description}")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    print(render())
