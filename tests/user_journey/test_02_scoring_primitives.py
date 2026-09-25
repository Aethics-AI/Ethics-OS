# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""The statistics and scale converters, used directly from ``aethics_eval``.

The README's "Library use" section invites people to call these on their own
data, so they are tested here as independent functions with hand-checkable
answers, plus the properties the docs promise (determinism, scale, edge
cases).
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from aethics_eval import (
    EVAL_INFERENCE_PARAMS,
    EVAL_MANIFEST,
    EVAL_METHODOLOGY_VERSION,
    EvalMethodologyManifest,
    ScoredMetric,
    bias_to_canonical,
    bootstrap_confidence_interval,
    canonical_to_zero_one,
    cohens_d,
    hash_audit_results,
    permutation_test,
    zero_one_to_canonical,
)
from aethics_eval.scoring import (
    is_underpowered,
    paired_cohens_d,
    paired_permutation_test,
    standard_error,
)

# ── Scale converters ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"), [(0, 100.0), (20, 80.0), (100, 0.0), (33.333, 66.67)]
)
def test_bias_to_canonical_inverts_and_rounds(raw, expected) -> None:
    assert bias_to_canonical(raw) == expected


@pytest.mark.parametrize("x", [0.0, 0.25, 0.5, 0.72, 0.8455, 1.0])
def test_zero_one_and_canonical_round_trip(x) -> None:
    assert canonical_to_zero_one(zero_one_to_canonical(x)) == pytest.approx(x, abs=1e-4)


def test_zero_one_to_canonical_scale() -> None:
    assert zero_one_to_canonical(0.85) == 85.0
    assert canonical_to_zero_one(84.55) == 0.8455


# ── Bootstrap CI ────────────────────────────────────────────────────


def test_bootstrap_matches_the_readme_example() -> None:
    assert bootstrap_confidence_interval([1, 0, 1, 1, 0, 1, 0, 1, 1, 0]) == (0.3, 0.9)


def test_bootstrap_is_deterministic_and_ignores_global_seed() -> None:
    data = [1, 0, 1, 1, 0, 1]
    random.seed(1)
    a = bootstrap_confidence_interval(data)
    random.seed(999)
    b = bootstrap_confidence_interval(data)
    assert a == b


def test_bootstrap_interval_brackets_the_mean_on_input_scale() -> None:
    rng = random.Random(0)
    data = [rng.random() for _ in range(200)]
    lo, hi = bootstrap_confidence_interval(data)
    mean = sum(data) / len(data)
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_bootstrap_narrows_as_data_grows() -> None:
    """The README's central point: more samples, tighter interval."""
    small = [1, 0] * 10
    large = [1, 0] * 500
    lo_s, hi_s = bootstrap_confidence_interval(small)
    lo_l, hi_l = bootstrap_confidence_interval(large)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_edge_cases() -> None:
    assert bootstrap_confidence_interval([]) == (0.0, 0.0)
    assert bootstrap_confidence_interval([0.42]) == (0.42, 0.42)
    assert bootstrap_confidence_interval([0.5, 0.5, 0.5]) == (0.5, 0.5)


# ── Effect size and significance ────────────────────────────────────


def test_cohens_d_sign_and_magnitude() -> None:
    assert cohens_d([10, 12, 11, 13], [10, 12, 11, 13]) == 0.0
    assert cohens_d([20, 22, 21, 23], [10, 12, 11, 13]) == 7.746
    assert cohens_d([10, 12, 11, 13], [20, 22, 21, 23]) == -7.746


def test_cohens_d_undefined_cases_return_zero() -> None:
    assert cohens_d([1.0], [2.0]) == 0.0
    assert cohens_d([3, 3, 3], [3, 3, 3]) == 0.0


def test_permutation_test_detects_a_real_difference() -> None:
    assert permutation_test([20, 22, 21, 23, 20], [10, 12, 11, 13, 10]) < 0.05
    assert permutation_test([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert permutation_test([], [1.0]) == 1.0


def test_permutation_test_is_deterministic() -> None:
    a, b = [1, 3, 2, 5, 4], [2, 2, 3, 1, 2]
    assert permutation_test(a, b) == permutation_test(a, b)


def test_paired_statistics() -> None:
    a = [0.9, 0.8, 0.85, 0.95, 0.7, 0.88]
    b = [0.1, 0.2, 0.15, 0.05, 0.3, 0.12]
    assert paired_permutation_test(a, b) < 0.05
    assert paired_cohens_d(a, b) > 0
    assert paired_cohens_d([1.0], [0.0]) is None  # undefined, not invented
    assert paired_cohens_d([1, 2], [0, 1]) is None  # zero-variance differences
    assert paired_permutation_test([1.0], [0.0]) == 1.0


def test_standard_error() -> None:
    assert standard_error([1.0]) is None
    assert standard_error([]) is None
    assert standard_error([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]) == 0.1633


def test_underpowered_flag_uses_manifest_thresholds() -> None:
    assert is_underpowered("crows_pairs", 5) is True
    assert (
        is_underpowered("crows_pairs", EVAL_MANIFEST.min_samples["crows_pairs"])
        is False
    )
    assert is_underpowered("no_such_task", 1) is False


# ── ScoredMetric ────────────────────────────────────────────────────


def test_scored_metric_serialises_with_reliability() -> None:
    m = ScoredMetric(value=84.55, confidence_interval=(80.0, 89.0), sample_size=1508)
    d = m.to_dict()
    assert d["value"] == 84.55
    assert d["reliability"] == "direct"
    assert tuple(d["confidence_interval"]) == (80.0, 89.0)


# ── Methodology fingerprint & inference params ──────────────────────


def test_readme_fingerprint_prefix() -> None:
    assert EVAL_MANIFEST.fingerprint()[:16] == "266b57eaa394ea0b"
    assert EVAL_MANIFEST.methodology_version == EVAL_METHODOLOGY_VERSION


def test_fingerprint_changes_when_method_changes() -> None:
    assert EvalMethodologyManifest().fingerprint() == EVAL_MANIFEST.fingerprint()
    other = EvalMethodologyManifest(methodology_version="99.0.0")
    assert other.fingerprint() != EVAL_MANIFEST.fingerprint()


def test_inference_params_are_deterministic_and_frozen() -> None:
    assert EVAL_INFERENCE_PARAMS.temperature == 0.0
    assert EVAL_INFERENCE_PARAMS.do_sample is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        EVAL_INFERENCE_PARAMS.temperature = 0.7  # type: ignore[misc]


# ── Audit hash ──────────────────────────────────────────────────────


def test_hash_audit_results_is_order_independent_and_sensitive() -> None:
    a = hash_audit_results({"score": 84.55, "model": "gpt2"})
    assert a == hash_audit_results({"model": "gpt2", "score": 84.55})
    assert a == "e217b2a7f19a1403afe812334c5af2be1f0f4a6e62a121d9e59530652f41b32c"
    assert a != hash_audit_results({"score": 84.56, "model": "gpt2"})
