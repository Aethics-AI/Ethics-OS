# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-5 Part A — uncertainty statistics.

Covers standard error, paired significance testing (the correct null for matched
stereotype pairs), paired effect size, and the under-powered check driven by
EVAL_MANIFEST.min_samples. Seeded throughout: the DoD requires bootstrap and
permutation results to be reproducible.
"""

import math
import statistics

import pytest

from aethics_eval.scoring import (
    bootstrap_confidence_interval,
    is_underpowered,
    paired_cohens_d,
    paired_differences,
    paired_permutation_test,
    standard_error,
)

# ── Standard error ──────────────────────────────────────────────────


def test_standard_error_matches_the_formula():
    scores = [0.2, 0.4, 0.6, 0.8, 1.0]
    expected = statistics.stdev(scores) / math.sqrt(len(scores))
    assert standard_error(scores) == pytest.approx(expected, abs=1e-4)


def test_standard_error_shrinks_as_samples_grow():
    """The whole point of VOS-5: 0.61 from 30 samples != 0.61 from 3,000."""
    small = [0.0, 1.0] * 15  # n=30
    large = [0.0, 1.0] * 1500  # n=3000, same mean
    assert statistics.mean(small) == statistics.mean(large)
    assert standard_error(large) < standard_error(small)


def test_standard_error_is_none_below_two_samples():
    # Undefined, so absent — never a fabricated 0.0 that reads as perfect precision.
    assert standard_error([]) is None
    assert standard_error([0.5]) is None


def test_standard_error_zero_for_identical_values():
    assert standard_error([0.5, 0.5, 0.5]) == 0.0


# ── Paired significance ─────────────────────────────────────────────


def test_paired_differences_are_elementwise():
    assert paired_differences([1.0, 2.0, 3.0], [0.5, 1.0, 1.5]) == [0.5, 1.0, 1.5]


def test_paired_test_detects_a_consistent_within_pair_difference():
    # Every pair differs in the same direction -> highly significant.
    a = [1.0] * 20
    b = [0.0] * 20
    assert paired_permutation_test(a, b) < 0.05


def test_paired_test_finds_no_effect_when_differences_cancel():
    a = [1.0, 0.0] * 10
    b = [0.0, 1.0] * 10  # differences alternate +1/-1, mean 0
    assert paired_permutation_test(a, b) > 0.05


def test_paired_test_is_seeded_and_reproducible():
    a = [0.9, 0.2, 0.7, 0.4, 0.8, 0.3]
    b = [0.1, 0.5, 0.2, 0.6, 0.3, 0.7]
    assert paired_permutation_test(a, b) == paired_permutation_test(a, b)


def test_paired_test_needs_at_least_two_pairs():
    assert paired_permutation_test([1.0], [0.0]) == 1.0
    assert paired_permutation_test([], []) == 1.0


def test_bootstrap_ci_is_seeded_and_reproducible():
    scores = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8]
    assert bootstrap_confidence_interval(scores) == bootstrap_confidence_interval(
        scores
    )


# ── Paired effect size ──────────────────────────────────────────────


def test_paired_cohens_d_sign_and_magnitude():
    a = [1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.5, 0.0]
    d = paired_cohens_d(a, b)
    assert d is not None and d > 0  # a consistently exceeds b


def test_paired_cohens_d_none_when_undefined():
    assert paired_cohens_d([1.0], [0.0]) is None  # too few pairs
    assert paired_cohens_d([1.0, 1.0], [0.0, 0.0]) is None  # zero variance in diffs


# ── Under-powered detection (min_samples) ───────────────────────────


def test_underpowered_uses_manifest_thresholds():
    from aethics_eval.eval_config import EVAL_MANIFEST

    threshold = EVAL_MANIFEST.min_samples["winobias"]  # 50
    assert is_underpowered("winobias", threshold - 1) is True
    assert is_underpowered("winobias", threshold) is False
    assert is_underpowered("winobias", threshold + 1) is False


def test_thirty_sample_winobias_run_is_underpowered():
    """DoD: a 30-sample run is flagged underpowered (winobias needs 50)."""
    assert is_underpowered("winobias", 30) is True


def test_unknown_task_is_not_flagged():
    # No threshold to judge against — don't invent one.
    assert is_underpowered("some_third_party_task", 1) is False
