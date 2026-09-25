# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-6 — the reproduction harness.

Runs a benchmark against the pinned reference model and reports our number
alongside the published one, with the VOS-5 uncertainty attached. A reproduction
claim without an error bar is not falsifiable: "0.61 reproduces 0.60" means
nothing until you know whether our figure is ±0.01 or ±0.15.

Runs through the VOS-3 task registry rather than the legacy suite, so what is
validated is the code path we actually ship.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .validity import VALIDITY_TARGETS, ValidityTarget

#: Where each task's reproducible metric lives in its result details. These are
#: the published quantities (a stereotype-preference rate, an accuracy gap), not
#: our derived 0-1 "score", which is a presentation transform of them.
_METRIC_KEYS: Dict[str, str] = {
    "crows_pairs": "stereotype_preference_rate",
    "stereoset": "stereotype_score",
    "winobias": "accuracy_gap",
}


@dataclass
class ReproductionResult:
    """One benchmark's reproduction attempt."""

    task: str
    published: Optional[float]
    observed: Optional[float]
    tolerance: Optional[float]
    standard_error: Optional[float]
    sample_size: int
    within_tolerance: bool
    note: str = ""

    @property
    def delta(self) -> Optional[float]:
        if self.published is None or self.observed is None:
            return None
        return round(self.observed - self.published, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "published": self.published,
            "observed": self.observed,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "standard_error": self.standard_error,
            "sample_size": self.sample_size,
            "within_tolerance": self.within_tolerance,
            "note": self.note,
        }


def _reference_model():
    """The pinned reference model, loaded locally for exact log-probabilities."""
    from .logprob_scorer import LogprobScorer
    from .models import LocalHFModel

    return LocalHFModel(scorer=LogprobScorer())  # pinned model + revision


async def reproduce_one(
    target: ValidityTarget, model: Any = None
) -> ReproductionResult:
    """Run one benchmark against the reference model and compare to published."""
    from .tasks import get_task, run_task

    # An excluded target with no measurable metric (BOLD) is reported as-is. One
    # that *is* measurable (StereoSet) still gets measured: we decline to claim a
    # reproduction, but the number we actually get should be visible, not hidden.
    if not target.claimed and target.task not in _METRIC_KEYS:
        return ReproductionResult(
            task=target.task,
            published=target.published_value,
            observed=None,
            tolerance=None,
            standard_error=None,
            sample_size=0,
            within_tolerance=False,
            note=target.excluded_reason or "no reproduction claim",
        )

    model = model if model is not None else _reference_model()
    result = await run_task(get_task(target.task), model, limit=target.sample_size)

    key = _METRIC_KEYS[target.task]
    observed = result.details.get(key)
    unc = result.uncertainty

    if observed is None:
        return ReproductionResult(
            task=target.task,
            published=target.published_value,
            observed=None,
            tolerance=target.tolerance,
            standard_error=None,
            sample_size=result.samples_tested,
            within_tolerance=False,
            note=f"not measured (no {key} in result)",
        )

    return ReproductionResult(
        task=target.task,
        published=target.published_value,
        observed=round(float(observed), 4),
        tolerance=target.tolerance,
        standard_error=unc.standard_error if unc else None,
        sample_size=result.samples_tested,
        within_tolerance=target.within_tolerance(float(observed)),
    )


async def reproduce_all(model: Any = None) -> List[ReproductionResult]:
    """Run every benchmark we can measure against the reference model.

    Includes targets we make no claim for but can still measure (StereoSet), so
    the methodology table can show the number we actually get alongside the
    published one. ``within_tolerance`` stays False for those — measuring is not
    claiming.
    """
    targets = [t for t in VALIDITY_TARGETS if t.claimed or t.task in _METRIC_KEYS]
    return [await reproduce_one(t, model=model) for t in targets]


def run_reproduction(model: Any = None) -> List[ReproductionResult]:
    """Synchronous entry point for scripts and CI."""
    return asyncio.run(reproduce_all(model=model))
