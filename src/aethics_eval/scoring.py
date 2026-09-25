# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Canonical Scoring Module

Single source of truth for score conversion, normalization, and
statistical analysis across the entire AETHICS platform.

Convention:
    - All scores at the API / DB boundary are 0-100, **higher = better**.
    - Internal test modules may use their own native scales but MUST
      call the appropriate converter before returning results.
"""

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

# ── Scale Converters ────────────────────────────────────────────────


def bias_to_canonical(raw_bias: float) -> float:
    """Convert bias score (0-100, lower=less biased) → canonical (0-100, higher=better).

    >>> bias_to_canonical(20)   # 20% biased → 80 canonical
    80.0
    """
    return round(100.0 - raw_bias, 2)


def zero_one_to_canonical(score_01: float) -> float:
    """Convert 0-1 score → 0-100. Useful for benchmark scores.

    >>> zero_one_to_canonical(0.85)
    85.0
    """
    return round(score_01 * 100.0, 2)


def canonical_to_zero_one(score_100: float) -> float:
    """Convert 0-100 canonical → 0-1 for framework compliance checks.

    The inverse of :func:`zero_one_to_canonical`, to 4 decimal places.

    >>> canonical_to_zero_one(84.55)
    0.8455
    >>> canonical_to_zero_one(zero_one_to_canonical(0.72))
    0.72
    """
    return round(score_100 / 100.0, 4)


# ── Statistical Rigor ───────────────────────────────────────────────


@dataclass
class ScoredMetric:
    """A single metric with full statistical context.

    Every score returned by the evaluation engine should be wrapped
    in this structure so downstream consumers know how much to trust it.

    ``reliability`` is the field that carries the honesty guarantee: a proxy
    measurement is never presented as a direct one, and a metric that could not
    be taken says so rather than defaulting to zero.

    >>> m = ScoredMetric(value=84.55, sample_size=1508, reliability="direct")
    >>> m.value
    84.55
    >>> m.to_dict()["reliability"]
    'direct'

    **Always check ``reliability`` before reading ``value``.** ``value`` is
    typed ``float`` and is always populated, so an unmeasured metric still
    carries a number — at least one caller emits a ``50.0`` placeholder with
    ``reliability="not_measured"``:

    >>> nm = ScoredMetric(value=50.0, sample_size=0, reliability="not_measured")
    >>> nm.value                     # a placeholder, not a measurement
    50.0
    >>> nm.reliability
    'not_measured'

    Reading that ``50.0`` as a score would report a middling result for a test
    that never ran. The flag is the claim; the number is not.

    .. warning::
       ``confidence_interval`` is **not** guaranteed to be an interval on
       ``value``. The benchmark aggregations currently pass the interval of the
       raw metric (0-1) alongside a ``value`` on the canonical 0-100 scale, so
       the two can be on different scales and describe different quantities.
       Do not render "value ± interval" without checking. Known defect.
    """

    value: float  # canonical 0-100
    confidence_interval: Tuple[float, float] = (0.0, 0.0)  # 95% CI
    effect_size: Optional[float] = None  # Cohen's d where applicable
    p_value: Optional[float] = None  # significance test p-value
    sample_size: int = 0
    reliability: str = "direct"  # direct | derived | proxy | not_measured

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bootstrap_confidence_interval(
    scores: List[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap 95% confidence interval for the mean.

    Args:
        scores: List of individual measurement scores.
        n_resamples: Number of bootstrap resamples (1000 is standard).
        confidence: Confidence level (0.95 = 95%).

    Returns:
        Tuple (lower_bound, upper_bound) on the same scale as input scores.

    The interval is on the **same scale as the input**, so feeding 0-1 scores
    returns a 0-1 interval. It is not converted to canonical 0-100.

    Resampling uses a fixed internal seed, so the result is deterministic for a
    given input. Seeding the global RNG has no effect, and the CLI's ``--seed``
    does not change it — that flag governs which samples are drawn, not the
    resampling here.

    >>> bootstrap_confidence_interval([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    (0.3, 0.9)

    Ten coin-flip-ish outcomes cannot pin down much, and the width says so.
    A single sample has no spread to estimate, so the interval collapses to the
    point rather than pretending to a range:

    >>> bootstrap_confidence_interval([0.42])
    (0.42, 0.42)
    >>> bootstrap_confidence_interval([])
    (0.0, 0.0)
    """
    if len(scores) < 2:
        mean = scores[0] if scores else 0.0
        return (mean, mean)

    import random

    rng = random.Random(42)  # fixed seed for reproducibility

    means = []
    n = len(scores)
    for _ in range(n_resamples):
        sample = [rng.choice(scores) for _ in range(n)]
        means.append(statistics.mean(sample))

    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_idx = int(alpha * n_resamples)
    upper_idx = int((1.0 - alpha) * n_resamples) - 1
    return (round(means[lower_idx], 4), round(means[upper_idx], 4))


def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """Calculate Cohen's d effect size between two groups.

    Returns:
        Positive d means group_a > group_b.
        |d| < 0.2 → negligible, 0.2-0.5 → small, 0.5-0.8 → medium, >0.8 → large.

    >>> cohens_d([10, 12, 11, 13], [10, 12, 11, 13])   # identical groups
    0.0
    >>> cohens_d([20, 22, 21, 23], [10, 12, 11, 13])   # separated groups
    7.746

    Effect size answers "how big is the difference", which is a different
    question from "is it real" — pair it with :func:`permutation_test`. A large
    d on four samples is not a finding.

    Returns ``0.0`` when either group has fewer than two values, since the
    pooled standard deviation is undefined there:

    >>> cohens_d([1.0], [2.0])
    0.0
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0

    mean_a = statistics.mean(group_a)
    mean_b = statistics.mean(group_b)
    var_a = statistics.variance(group_a)
    var_b = statistics.variance(group_b)

    pooled_std = math.sqrt((var_a + var_b) / 2.0)
    if pooled_std == 0:
        return 0.0
    return round((mean_a - mean_b) / pooled_std, 4)


def permutation_test(
    group_a: List[float],
    group_b: List[float],
    n_permutations: int = 5000,
) -> float:
    """Two-sided permutation test for difference in means.

    Returns:
        p-value. Values < 0.05 → statistically significant difference.

    Makes no distributional assumption: it shuffles the pooled values and counts
    how often a split at least as extreme as the observed one appears by chance.

    >>> permutation_test([20, 22, 21, 23, 20], [10, 12, 11, 13, 10])
    0.008
    >>> permutation_test([1, 2, 3, 4], [1, 2, 3, 4])   # no difference at all
    1.0

    Like the bootstrap, this is seeded internally and so is deterministic.

    Returns ``1.0`` — the least significant answer available — when either group
    is empty, rather than implying a result from no data:

    >>> permutation_test([], [1.0, 2.0])
    1.0

    The groups are unpaired. Where samples are matched by construction, as
    stereotype sentence pairs are, an unpaired test discards the pairing and
    understates significance; use the paired helpers below instead.
    """
    if not group_a or not group_b:
        return 1.0

    import random

    rng = random.Random(42)

    observed_diff = abs(statistics.mean(group_a) - statistics.mean(group_b))
    combined = group_a + group_b
    n_a = len(group_a)
    count_extreme = 0

    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = abs(statistics.mean(perm_a) - statistics.mean(perm_b))
        if perm_diff >= observed_diff:
            count_extreme += 1

    return round(count_extreme / n_permutations, 4)


# ── Uncertainty (VOS-5) ─────────────────────────────────────────────
#
# A point estimate is not a claim. 0.61 from 30 samples and 0.61 from 3,000 are
# different statements, and every comparable harness (lm-eval) reports a standard
# error alongside the metric. These helpers supply that, plus the paired tests our
# tasks actually need: stereotype pairs are matched by construction, so comparing
# them with an unpaired test throws away the pairing and understates significance.


def standard_error(scores: List[float]) -> Optional[float]:
    """Standard error of the mean: s / sqrt(n).

    Returns None for fewer than two samples — the SEM is undefined there, and a
    fabricated 0.0 would read as "perfectly precise" (the honesty rule: absent,
    not invented).
    """
    n = len(scores)
    if n < 2:
        return None
    return round(statistics.stdev(scores) / math.sqrt(n), 4)


def paired_differences(group_a: List[float], group_b: List[float]) -> List[float]:
    """Element-wise a - b for matched pairs (truncated to the shorter list)."""
    return [a - b for a, b in zip(group_a, group_b, strict=False)]


def paired_permutation_test(
    group_a: List[float],
    group_b: List[float],
    n_permutations: int = 5000,
) -> float:
    """Two-sided **paired** permutation test on matched observations.

    Randomly flips the sign of each pair's difference — the correct null for
    matched data (does the difference within a pair have a consistent direction?).
    ``permutation_test`` shuffles *between* groups, which assumes independent
    samples and is the wrong null for stereotype pairs.

    Seeded, so a rerun on the same input returns the same p-value.
    """
    diffs = paired_differences(group_a, group_b)
    if len(diffs) < 2:
        return 1.0

    import random

    rng = random.Random(42)  # fixed seed for reproducibility

    observed = abs(statistics.mean(diffs))
    count_extreme = 0
    for _ in range(n_permutations):
        flipped = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(statistics.mean(flipped)) >= observed:
            count_extreme += 1
    return round(count_extreme / n_permutations, 4)


def paired_cohens_d(group_a: List[float], group_b: List[float]) -> Optional[float]:
    """Cohen's d for matched pairs: mean(diff) / stdev(diff).

    The paired counterpart to ``cohens_d``. Returns None when it is undefined
    (fewer than two pairs, or zero variance in the differences).
    """
    diffs = paired_differences(group_a, group_b)
    if len(diffs) < 2:
        return None
    sd = statistics.stdev(diffs)
    if sd == 0:
        return None
    return round(statistics.mean(diffs) / sd, 4)


def is_underpowered(task_name: str, sample_size: int) -> bool:
    """Whether ``sample_size`` falls below the task's ``min_samples`` threshold.

    Thresholds live in ``EVAL_MANIFEST.min_samples`` (eval_config.py) — the
    minimum needed for statistical power. An under-powered result is reported and
    flagged, never silently averaged in as if it carried the same weight.
    Unknown task names are not flagged (no threshold to judge against).
    """
    from .eval_config import EVAL_MANIFEST

    threshold = EVAL_MANIFEST.min_samples.get(task_name)
    if threshold is None:
        return False
    return sample_size < threshold


# ── Audit Result Hashing ────────────────────────────────────────────


def hash_audit_results(results: Dict[str, Any]) -> str:
    """Produce a deterministic SHA-256 hash of complete audit results.

    This hash is stored alongside the audit and can be verified later
    to confirm that results have not been tampered with.

    Keys are sorted before hashing, so dict ordering does not affect the result:

    >>> hash_audit_results({"score": 84.55, "model": "gpt2"})
    'e217b2a7f19a1403afe812334c5af2be1f0f4a6e62a121d9e59530652f41b32c'
    >>> a = hash_audit_results({"score": 84.55, "model": "gpt2"})
    >>> b = hash_audit_results({"model": "gpt2", "score": 84.55})
    >>> a == b
    True

    .. warning::
       This hashes **whatever dict it is given**. If that dict contains only
       final scores, then editing an underlying sample and adjusting the totals
       to match produces a document that still hashes correctly — which was the
       substance of the F5 finding.

       For tamper-evidence over the actual evidence, use the run manifest
       (:mod:`aethics_eval.manifest`) and ``aethics verify``, which hash
       per-sample scores into a Merkle tree and recompute the root. The totals
       are then not what is hashed, so they cannot be made to agree.
    """
    canonical = json.dumps(results, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
