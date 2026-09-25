# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Aggregation: requirement results → dimension scores → readiness.

The tests that matter most are the ones separating a score of zero from the
absence of one, and the ones holding the readiness score back when a dimension
is missing.
"""

from __future__ import annotations

import pytest

from aethics_eval.aggregation import (
    DIMENSION_ORDER,
    DIMENSION_WEIGHT,
    SCALE_MAX,
    Dimension,
    RequirementResult,
    dimension_score_report,
    readiness_for,
    score_dimension,
    score_dimensions,
)

SEC = Dimension.SECURITY


def _measured(dim: Dimension, score: float) -> RequirementResult:
    return RequirementResult(dim, "measured", score)


def _full_assessment(score: float = 0.8) -> list[RequirementResult]:
    return [_measured(d, score) for d in DIMENSION_ORDER]


# ── Dimension scores ────────────────────────────────────────────────


def test_score_is_mean_of_measured_on_a_ten_point_scale() -> None:
    s = score_dimension(SEC, [_measured(SEC, 0.9), _measured(SEC, 0.6)])
    assert s.score == 7.5
    assert s.status == "measured"
    assert s.coverage == 1.0


def test_unmeasured_statuses_are_excluded_not_averaged_as_zero() -> None:
    results = [
        _measured(SEC, 0.8),
        RequirementResult(SEC, "requires_attestation", 0.0),
        RequirementResult(SEC, "not_measured", 0.0),
        RequirementResult(SEC, "low_coverage", 0.0),
    ]
    s = score_dimension(SEC, results)
    assert s.score == 8.0, "a stored 0.0 on an unmeasured result leaked into the mean"
    assert (s.scored, s.attestation_only, s.not_measured, s.low_coverage) == (
        1,
        1,
        1,
        1,
    )
    assert s.status == "partially_measured"
    assert s.coverage == 0.25


def test_nothing_measured_is_none_not_zero() -> None:
    s = score_dimension(SEC, [RequirementResult(SEC, "requires_attestation")])
    assert s.score is None
    assert s.status == "not_measured"
    payload = s.to_dict()
    assert "score" in payload and payload["score"] is None
    assert "not a score of zero" in str(payload["note"])


def test_a_genuine_zero_is_reported_as_zero() -> None:
    s = score_dimension(SEC, [_measured(SEC, 0.0)])
    assert s.score == 0.0
    assert s.status == "measured"


def test_no_requirements_is_its_own_status() -> None:
    s = score_dimension(SEC, [])
    assert s.status == "no_requirements"
    assert s.coverage is None


@pytest.mark.parametrize("bad", [-0.1, 1.5, None])
def test_measured_score_must_be_present_and_in_range(bad) -> None:
    with pytest.raises(ValueError):
        score_dimension(SEC, [RequirementResult(SEC, "measured", bad)])


def test_results_are_routed_by_their_dimension() -> None:
    scores = score_dimensions(
        [_measured(SEC, 1.0), _measured(Dimension.TRANSPARENCY, 0.5)]
    )
    assert scores[SEC].score == SCALE_MAX
    assert scores[Dimension.TRANSPARENCY].score == 5.0
    assert scores[Dimension.HUMAN_OVERSIGHT].status == "no_requirements"


def test_report_lists_dimensions_in_order() -> None:
    report = dimension_score_report(_full_assessment())
    assert [d["dimension"] for d in report["dimensions"]] == [
        d.value for d in DIMENSION_ORDER
    ]
    assert report["not_measured"] == []


# ── Readiness ───────────────────────────────────────────────────────


def test_readiness_is_the_equal_weighted_mean_when_all_six_scored() -> None:
    results = _full_assessment(0.8)
    results.append(_measured(SEC, 0.2))  # security averages to 5.0
    r = readiness_for(results)
    assert r.available
    assert r.score == round((8.0 * 5 + 5.0) / 6, 1)
    assert set(DIMENSION_WEIGHT.values()) == {1.0}


def test_readiness_is_withheld_when_any_dimension_is_missing() -> None:
    results = [
        r for r in _full_assessment(1.0) if r.dimension is not Dimension.HUMAN_OVERSIGHT
    ]
    results.append(RequirementResult(Dimension.HUMAN_OVERSIGHT, "requires_attestation"))
    r = readiness_for(results)
    assert r.score is None, "a partial average was published as readiness"
    assert r.status == "incomplete_coverage"
    assert r.missing == (Dimension.HUMAN_OVERSIGHT,)
    payload = r.to_dict()
    assert payload["score"] is None
    assert payload["missing_dimensions"] == ["human oversight"]


def test_readiness_with_nothing_measured() -> None:
    r = readiness_for([])
    assert r.score is None
    assert r.status == "nothing_measured"
    assert r.dimensions_scored == 0
