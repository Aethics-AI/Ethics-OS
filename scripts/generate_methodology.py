# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Generate docs/methodology.md from the validity targets and a reproduction run.

    python -m scripts.generate_methodology              # uses the reference model
    python -m scripts.generate_methodology --dry-run    # targets only, no model

The document is generated rather than hand-written so it cannot drift from the
constants the validity tests actually assert against. Deviations, tolerances and
citations all come from validity.py; the observed numbers come from a real run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

from aethics_eval.reproduction import ReproductionResult, run_reproduction
from aethics_eval.validity import (
    REFERENCE_MODEL,
    REFERENCE_REVISION,
    VALIDITY_TARGETS,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "methodology.md"


def _fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4g}"


def render(observed: Dict[str, ReproductionResult]) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Methodology and validity\n")
    add(
        "This document states what our benchmark implementations measure, how "
        "that compares to the published numbers, and every place we depart from "
        "the published methodology. It is the evidence that our implementations "
        "are *correct*, not merely that they run.\n"
    )
    add(
        "It is generated from `src/aethics_eval/validity.py`, which is also what "
        "the `-m validity` tests assert against, so the claims below cannot drift "
        "from what is actually checked.\n"
    )

    add("## Reference model\n")
    add(
        f"All numbers below are for **`{REFERENCE_MODEL}`**, pinned to revision "
        f"`{REFERENCE_REVISION}`.\n"
    )
    add(
        "GPT-2 is the conventional reference for these benchmarks: it is small, "
        "deterministic under greedy decoding, and the model the published figures "
        "were computed against. Pinning the exact commit matters — a reproduction "
        "claim against a moving model is not a claim.\n"
    )

    add("## Summary\n")
    add("| Benchmark | Published | Ours | Δ | Tolerance | Within? |")
    add("|---|---|---|---|---|---|")
    for t in VALIDITY_TARGETS:
        if not t.claimed:
            # Still show the published figure where one exists: a disagreement we
            # decline to claim should be more visible, not less.
            r = observed.get(t.task)
            ours = _fmt(r.observed) if r is not None and r.observed is not None else "—"
            add(
                f"| `{t.task}` | {_fmt(t.published_value)} | {ours} | — | — | "
                "*no claim — see below* |"
            )
            continue
        r = observed.get(t.task)
        if r is None or r.observed is None:
            add(
                f"| `{t.task}` | {_fmt(t.published_value)} | *not run* | — | "
                f"±{_fmt(t.tolerance)} | — |"
            )
            continue
        ours = _fmt(r.observed)
        if r.standard_error is not None:
            ours += f" ± {r.standard_error:.2g}"
        mark = "✅" if r.within_tolerance else "❌"
        add(
            f"| `{t.task}` | {_fmt(t.published_value)} | {ours} | "
            f"{r.delta:+.4g} | ±{_fmt(t.tolerance)} | {mark} |"
        )
    add("")
    add(
        "Our figures carry a standard error because a reproduction claim without "
        "one is not falsifiable: whether 0.61 reproduces 0.60 depends entirely on "
        "whether our figure is ±0.01 or ±0.15.\n"
    )

    add("## Per benchmark\n")
    for t in VALIDITY_TARGETS:
        add(f"### {t.task}\n")
        add(f"**Metric.** {t.metric}\n")

        if not t.claimed:
            # The reason text supplies its own lead sentence, so don't prefix a
            # second header on top of it.
            add(f"{t.excluded_reason}\n")
            add(f"**Published work.** {t.published_source}\n")
            continue

        r = observed.get(t.task)
        add(f"**Published.** {_fmt(t.published_value)} — {t.published_source}\n")
        if r is not None and r.observed is not None:
            se = f" ± {r.standard_error:.2g}" if r.standard_error is not None else ""
            add(
                f"**Ours.** {_fmt(r.observed)}{se} at n={r.sample_size} "
                f"(Δ {r.delta:+.4g}).\n"
            )
        else:
            add(f"**Ours.** Not run in this build (n={t.sample_size} when run).\n")
        # The rationale already opens with the figure, so don't repeat it.
        add(f"**Tolerance.** {t.tolerance_rationale}\n")

        if t.deviations:
            add("**Deviations from the published methodology.**\n")
            for d in t.deviations:
                add(f"- *{d.what}*  \n  {d.why}\n")
        else:
            add("**Deviations.** None.\n")

    add("## Reproducing this\n")
    add("```bash")
    add("pip install -e '.[local,dev]'   # local extra pulls torch + transformers")
    add("pytest -m validity -s           # downloads the pinned model, runs the table")
    add("```\n")
    add(
        "The validity suite is excluded from the default run (it downloads a model "
        "and does real inference) and runs nightly instead. All statistics are "
        "seeded, so a rerun on the same revision reproduces the same numbers.\n"
    )
    return "\n".join(lines)


def main() -> None:
    dry = "--dry-run" in sys.argv
    observed: Dict[str, ReproductionResult] = {}
    if not dry:
        for r in run_reproduction():
            observed[r.task] = r
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(observed))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
