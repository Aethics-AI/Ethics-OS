# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-5 — build an ``Uncertainty`` from raw per-sample scores.

One place that turns a list of per-sample values (and, for matched tasks, the two
paired condition lists) into the statistics every metric ships with: standard
error, a seeded bootstrap CI, effect size, a significance test, and the
under-powered flag from ``EVAL_MANIFEST.min_samples``.

Tasks call this rather than each reimplementing the statistics, so every metric
reports uncertainty the same way and the seeding stays consistent.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from aethics_eval.scoring import (
    bootstrap_confidence_interval,
    is_underpowered,
    paired_cohens_d,
    paired_permutation_test,
    standard_error,
)

from .schema import Uncertainty

#: Bootstrap resample count. 1,000 is the conventional default and what the
#: method string reports, so a reader knows exactly what produced the interval.
DEFAULT_RESAMPLES = 1000


def build_uncertainty(
    scores: Sequence[float],
    *,
    task_name: Optional[str] = None,
    paired_a: Optional[Sequence[float]] = None,
    paired_b: Optional[Sequence[float]] = None,
    n_resamples: int = DEFAULT_RESAMPLES,
) -> Uncertainty:
    """Compute the uncertainty around the mean of ``scores``.

    Args:
        scores: per-sample values the metric aggregates.
        task_name: used to look up the ``min_samples`` power threshold.
        paired_a / paired_b: the two matched condition lists, when the task
            compares two conditions (stereotype pairs, pro/anti sets). Given
            these, significance and effect size use the **paired** tests — the
            correct null for matched data.
        n_resamples: bootstrap resamples (seeded, so reruns match).

    A statistic that cannot be computed is left as None rather than defaulted.

    >>> scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    >>> u = build_uncertainty(scores)
    >>> u.method
    'bootstrap_1000'
    >>> u.standard_error
    0.1633
    >>> u.sample_size
    10

    Pass ``task_name`` to have the run checked against that task's power
    threshold. Ten samples is far below CrowS-Pairs' floor:

    >>> build_uncertainty(scores, task_name="crows_pairs").underpowered
    True

    A single sample has no spread to estimate, so the statistics are absent
    rather than invented:

    >>> lonely = build_uncertainty([1.0])
    >>> lonely.method is None and lonely.confidence_interval is None
    True
    """
    values: List[float] = list(scores)
    n = len(values)

    sem = standard_error(values)
    ci = (
        bootstrap_confidence_interval(values, n_resamples=n_resamples)
        if n >= 2
        else None
    )

    effect_size = p_value = None
    method_parts = []
    if ci is not None:
        method_parts.append(f"bootstrap_{n_resamples}")

    if paired_a is not None and paired_b is not None:
        a, b = list(paired_a), list(paired_b)
        effect_size = paired_cohens_d(a, b)
        p_value = paired_permutation_test(a, b)
        method_parts.append("paired_permutation")

    return Uncertainty(
        method="+".join(method_parts) or None,
        standard_error=sem,
        confidence_interval=ci,
        effect_size=effect_size,
        p_value=p_value,
        sample_size=n,
        underpowered=is_underpowered(task_name, n) if task_name else False,
    )


def format_with_uncertainty(value: Optional[float], unc: Optional[Uncertainty]) -> str:
    """Render a metric as ``0.61 ± 0.04`` for human-readable output.

    Falls back to the bare value when no standard error is available, and to
    ``not measured`` when the value itself is None. Appends a ``(underpowered)``
    marker so a thin run is visible at a glance rather than only in the JSON.

    >>> scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    >>> u = build_uncertainty(scores, task_name="crows_pairs")
    >>> format_with_uncertainty(0.6, u)
    '0.6 ± 0.16 (underpowered)'

    No uncertainty to hand — the bare value, not a fabricated interval:

    >>> format_with_uncertainty(0.5773, None)
    '0.5773'

    And an unmeasured metric says so, rather than rendering as zero:

    >>> format_with_uncertainty(None, u)
    'not measured'
    """
    if value is None:
        return "not measured"
    text = f"{value:.4g}"
    if unc is not None and unc.standard_error is not None:
        text = f"{text} ± {unc.standard_error:.2g}"
    if unc is not None and unc.underpowered:
        text = f"{text} (underpowered)"
    return text
