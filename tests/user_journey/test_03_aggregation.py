# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Requirement results → six dimension scores (0–10) → readiness score.

This is the part of the scoring algorithm a user is most likely to probe with
"what if" inputs, so it is tested the same way: build requirement results by
hand, and check the numbers and refusals that come out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import pytest

from aethics_eval.aggregation import (
    DIMENSION_DESCRIPTION,
    DIMENSION_ORDER,
    Dimension,
    RequirementResult,
    dimension_score_report,
    readiness_for,
    readiness_from_scores,
    score_dimension,
    score_dimensions,
)


def measured(dim: Dimension, score: float) -> RequirementResult:
    return RequirementResult(dim, "measured", score)


def all_six(score: float) -> list[RequirementResult]:
    return [measured(d, score) for d in DIMENSION_ORDER]


def test_six_dimensions_in_a_stable_order_with_descriptions() -> None:
    assert len(DIMENSION_ORDER) == 6
    assert set(DIMENSION_ORDER) == set(Dimension)
    assert all(DIMENSION_DESCRIPTION[d] for d in DIMENSION_ORDER)
    assert Dimension.RIGHT_TO_LIBERTY.slug == "right_to_liberty"


def test_dimension_score_is_mean_of_measured_times_ten() -> None:
    s = score_dimension(
        Dimension.SECURITY,
        [measured(Dimension.SECURITY, 0.9), measured(Dimension.SECURITY, 0.7)],
    )
    assert s.score == 8.0
    assert s.status == "measured"
    assert s.coverage == 1.0


def test_unmeasured_requirements_are_excluded_not_zeroed() -> None:
    results = [
        measured(Dimension.SECURITY, 0.9),
        measured(Dimension.SECURITY, 0.7),
        RequirementResult(Dimension.SECURITY, "requires_attestation"),
        RequirementResult(Dimension.SECURITY, "low_coverage", 0.0),
        RequirementResult(Dimension.SECURITY, "no_data", 0.0),
    ]
    s = score_dimensions(results)[Dimension.SECURITY]
    assert s.score == 8.0  # would be 3.2 if the three non-measured counted as 0
    assert (s.requirements, s.scored) == (5, 2)
    assert (s.attestation_only, s.low_coverage, s.not_measured) == (1, 1, 1)
    assert s.status == "partially_measured"
    assert s.coverage == 0.4


def test_a_dimension_with_nothing_measured_scores_none_not_zero() -> None:
    scores = score_dimensions([RequirementResult(Dimension.TRANSPARENCY, "no_data")])
    assert scores[Dimension.TRANSPARENCY].score is None
    assert scores[Dimension.TRANSPARENCY].status == "not_measured"
    assert scores[Dimension.HUMAN_OVERSIGHT].status == "no_requirements"


def test_readiness_needs_all_six_dimensions() -> None:
    five = [measured(d, 0.8) for d in DIMENSION_ORDER[:-1]]
    r = readiness_for(five)
    assert r.score is None
    assert r.status == "incomplete_coverage"
    assert r.missing == (Dimension.SECURITY,)
    assert r.dimensions_scored == 5


def test_readiness_with_nothing_measured() -> None:
    r = readiness_for([])
    assert r.score is None
    assert r.status == "nothing_measured"
    assert len(r.missing) == 6


def test_readiness_is_equal_weighted_mean_of_dimensions() -> None:
    scores = [0.2, 0.4, 0.6, 0.8, 1.0, 0.6]
    results = [measured(d, s) for d, s in zip(DIMENSION_ORDER, scores, strict=True)]
    r = readiness_for(results)
    assert r.available
    assert r.status == "complete"
    assert r.score == pytest.approx(10 * sum(scores) / 6, abs=0.05)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_readiness_bounds(value) -> None:
    assert readiness_for(all_six(value)).score == value * 10


@pytest.mark.parametrize("bad", [-0.01, 1.01, 5.0])
def test_scores_outside_zero_one_are_rejected(bad) -> None:
    with pytest.raises(ValueError, match=r"0.0-1.0"):
        score_dimensions([measured(Dimension.SECURITY, bad)])


def test_measured_without_a_score_is_rejected() -> None:
    with pytest.raises(ValueError, match="no score"):
        score_dimensions([RequirementResult(Dimension.SECURITY, "measured", None)])


def test_score_on_a_non_measured_status_is_ignored() -> None:
    """A catalogue that stores 0.0 for unmeasured rows must not leak it."""
    results = [
        measured(Dimension.SECURITY, 1.0),
        RequirementResult(Dimension.SECURITY, "no_data", 0.0),
    ]
    assert score_dimensions(results)[Dimension.SECURITY].score == 10.0


def test_user_supplied_requirement_type_works() -> None:
    """Any object with dimension/status/score is accepted (Protocol)."""

    @dataclass
    class MyRequirement:
        id: str
        dimension: Dimension
        status: str
        score: Optional[float]

    rows = [
        MyRequirement(f"R{i}", d, "measured", 0.5)
        for i, d in enumerate(DIMENSION_ORDER)
    ]
    assert readiness_for(rows).score == 5.0


def test_report_is_json_serialisable_and_nulls_are_explicit() -> None:
    results = [measured(d, 0.7) for d in DIMENSION_ORDER[:3]]
    report = dimension_score_report(results)
    text = json.dumps(report)  # must not raise
    assert '"score": null' in text
    dims = report["dimensions"]
    assert [d["slug"] for d in dims] == [d.slug for d in DIMENSION_ORDER]
    # Dimensions with no requirements at all are "no_requirements", which is
    # a different claim from requirements that existed and could not be
    # measured — only the latter are listed under "not_measured".
    assert report["not_measured"] == []
    assert [d["status"] for d in dims[3:]] == ["no_requirements"] * 3
    assert all("note" in d for d in dims if d["score"] is None)

    report = dimension_score_report(
        [*results, RequirementResult(Dimension.SECURITY, "no_data")]
    )
    assert report["not_measured"] == ["security"]

    readiness = readiness_from_scores(score_dimensions(results)).to_dict()
    assert readiness["score"] is None
    assert len(readiness["missing_dimensions"]) == 3
    json.dumps(readiness)
