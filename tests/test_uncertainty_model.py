# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-5 Part B — the Uncertainty model and its builder.

Covers: standard error / CI / effect size / p-value / sample size land on the
schema object, paired tests are used when a task compares two conditions, the
under-powered flag comes from EVAL_MANIFEST.min_samples, statistics that cannot
be computed stay None, and the ± rendering.
"""

import json

from aethics_eval.results import (
    DatasetInfo,
    ModelInfo,
    RunResult,
    TaskResult,
    Uncertainty,
    build_uncertainty,
    format_with_uncertainty,
    validate_result,
)

# ── build_uncertainty ───────────────────────────────────────────────


def test_builds_the_core_statistics():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    u = build_uncertainty(scores)
    assert u.sample_size == 8
    assert u.standard_error is not None and u.standard_error > 0
    lo, hi = u.confidence_interval
    assert lo <= sum(scores) / len(scores) <= hi
    assert "bootstrap_1000" in u.method


def test_paired_inputs_use_the_paired_tests():
    # Matched pairs whose groups overlap: only the paired test sees the effect.
    a = [0.50, 0.60, 0.70, 0.80, 0.90, 0.55, 0.65, 0.75]
    b = [0.45, 0.55, 0.65, 0.75, 0.85, 0.50, 0.60, 0.70]
    u = build_uncertainty(
        [x - y for x, y in zip(a, b, strict=False)], paired_a=a, paired_b=b
    )
    assert "paired_permutation" in u.method
    assert u.p_value is not None and u.p_value < 0.05
    assert u.effect_size is not None


def test_unpaired_leaves_significance_absent_not_zero():
    u = build_uncertainty([0.3, 0.6, 0.9])
    assert u.p_value is None  # no comparison was made — absent, not a fake 1.0
    assert u.effect_size is None


def test_single_sample_leaves_statistics_absent():
    u = build_uncertainty([0.5])
    assert u.sample_size == 1
    assert u.standard_error is None  # undefined for n<2, never a fabricated 0.0
    assert u.confidence_interval is None


def test_underpowered_flag_from_min_samples():
    # winobias needs 50 samples.
    thin = build_uncertainty([1.0, 0.0] * 15, task_name="winobias")  # n=30
    assert thin.underpowered is True
    ok = build_uncertainty([1.0, 0.0] * 30, task_name="winobias")  # n=60
    assert ok.underpowered is False


def test_no_task_name_means_no_power_claim():
    u = build_uncertainty([1.0] * 3)  # no task_name -> nothing to judge against
    assert u.underpowered is False


def test_builder_is_reproducible():
    scores = [0.2, 0.9, 0.4, 0.7, 0.1, 0.8]
    assert build_uncertainty(scores) == build_uncertainty(scores)


# ── Rendering (the `0.61 ± 0.04` the DoD asks for) ──────────────────


def test_renders_value_plus_minus_stderr():
    u = Uncertainty(standard_error=0.04)
    assert format_with_uncertainty(0.61, u) == "0.61 ± 0.04"


def test_renders_underpowered_marker():
    u = Uncertainty(standard_error=0.04, underpowered=True)
    assert format_with_uncertainty(0.61, u) == "0.61 ± 0.04 (underpowered)"


def test_renders_not_measured_for_null_metric():
    assert format_with_uncertainty(None, Uncertainty()) == "not measured"


def test_renders_bare_value_without_stderr():
    assert format_with_uncertainty(0.61, Uncertainty()) == "0.61"
    assert format_with_uncertainty(0.61, None) == "0.61"


# ── Schema integration ──────────────────────────────────────────────


def test_uncertainty_serialises_and_validates_in_a_result():
    u = build_uncertainty([1.0, 0.0] * 15, task_name="winobias")
    run = RunResult(
        run_id="unc-1",
        model=ModelInfo(id="m", provider="hf"),
        manifest_fingerprint="0" * 64,
        tasks=[
            TaskResult(
                task_id="winobias",
                task_version="1.0.0",
                dataset=DatasetInfo(name="uclanlp/wino_bias", revision="main"),
                metric="coref_accuracy_gap",
                metric_value=0.61,
                uncertainty=u,
                sample_count=u.sample_size,
                method="classifier",
                reliability="direct",
            )
        ],
    )
    payload = json.loads(run.to_json())
    unc = payload["tasks"][0]["uncertainty"]
    assert unc["standard_error"] is not None
    assert unc["sample_size"] == 30
    assert unc["underpowered"] is True
    validate_result(payload)  # still conforms to the published schema
