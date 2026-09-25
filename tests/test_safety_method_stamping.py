# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-2 Part C — every safety result stamps how it was scored.

The DoD guard: no code path in the safety tester emits a score without a
``method`` (classifier / keyword / not_measured) and a matching reliability
grade. A keyword score must never wear a classifier's badge (reliability
"direct").
"""

import asyncio

from aethics_eval.models import FakeModel
from aethics_eval.nlp_safety_tests import NLPSafetyTester
from aethics_eval.refusal_classifier import reliability_for


def _run(coro):
    return asyncio.run(coro)


# A model that answers every safety prompt with a clear refusal, so refusal-based
# tests have measurable responses (non-empty) to score.
def _refusing_model():
    return FakeModel(default_response="I'm sorry, but I can't help with that request.")


VALID_METHODS = {"classifier", "keyword", "not_measured"}


def test_every_test_result_carries_a_method_and_matching_reliability():
    tester = NLPSafetyTester()
    payload = _run(tester.run_comprehensive_safety_test(_refusing_model()))
    assert payload["success"]

    for name, result in payload["test_results"].items():
        # 1. Every result stamps a method.
        assert "method" in result, f"{name} has no method stamp"
        assert result["method"] in VALID_METHODS, f"{name}: {result['method']}"
        # 2. details echoes it too.
        assert result["details"]["method"] == result["method"]
        # 3. The ScoredMetric reliability agrees with the method — a keyword score
        #    is never graded "direct".
        sm = result["scored_metric"]
        assert sm["reliability"] == reliability_for(result["method"]), name


def test_default_refusal_path_is_keyword_not_direct():
    """Without a judge model, refusal detection is the heuristic → proxy, not direct."""
    tester = NLPSafetyTester()
    payload = _run(tester.run_comprehensive_safety_test(_refusing_model()))

    refusal = payload["test_results"]["harmful_content_test"]
    assert refusal["method"] == "keyword"
    assert refusal["scored_metric"]["reliability"] == "proxy"
    # And it actually counted the refusals (heuristic works, score not fabricated).
    assert refusal["safety_score"] > 0
    assert refusal["details"]["measured_responses"] > 0


def test_classifier_path_upgrades_reliability_to_direct():
    """With a judge model wired in, refusal tests are stamped classifier/direct."""
    tester = NLPSafetyTester()
    # A judge that always says REFUSE, so the classifier path is exercised.
    tester.refusal_judge = FakeModel(default_response="REFUSE")

    payload = _run(tester.run_comprehensive_safety_test(_refusing_model()))
    refusal = payload["test_results"]["harmful_content_test"]
    assert refusal["method"] == "classifier"
    assert refusal["scored_metric"]["reliability"] == "direct"


def test_unusable_responses_are_not_measured_not_fabricated():
    """A model that returns nothing usable ⇒ refusal tests report not_measured."""
    tester = NLPSafetyTester()
    # Empty responses → refusal judgments are not_measured.
    payload = _run(tester.run_comprehensive_safety_test(FakeModel(default_response="")))

    refusal = payload["test_results"]["harmful_content_test"]
    assert refusal["method"] == "not_measured"
    assert refusal["scored_metric"]["reliability"] == "not_measured"
    assert refusal["details"]["measured_responses"] == 0
    # No fabricated 50 — the score collapses to 0 under not_measured.
    assert refusal["safety_score"] == 0.0


def test_all_not_measured_reports_failure_not_a_zero_overall():
    """F1: if NO sub-test could be measured, the safety evaluation reports
    success=False (which every consumer guards on) rather than an invented
    overall_safety_score of 0."""
    tester = NLPSafetyTester()
    payload = _run(tester.run_comprehensive_safety_test(FakeModel(default_response="")))

    assert payload["success"] is False
    assert payload["coverage"].startswith("0/")
    assert "overall_safety_score" not in payload  # no fabricated aggregate


def test_overall_excludes_not_measured_subtests():
    """F1: a not_measured sub-test must not drag the overall score toward 0.

    Toxicity prompts get a benign reply (measured, safe); harmful/privacy/refusal
    prompts get empty replies (not_measured). The overall must reflect only the
    measured toxicity test (~100 safe), not mean(100, 0, 0, 0) = 25.
    """
    from aethics_eval.eval_datasets import (
        HARMFUL_CONTENT_PROMPTS,
        PRIVACY_PROMPTS,
        SAFETY_REFUSAL_PROMPTS,
    )

    blocked = {
        p.text
        for p in (*HARMFUL_CONTENT_PROMPTS, *PRIVACY_PROMPTS, *SAFETY_REFUSAL_PROMPTS)
    }
    model = FakeModel(
        responses=lambda p: "" if p in blocked else "a perfectly nice neutral reply"
    )
    payload = _run(NLPSafetyTester().run_comprehensive_safety_test(model))

    assert payload["success"] is True
    assert payload["overall_reliability"] == "partial"
    assert payload["coverage"].startswith("1/")  # only toxicity measured
    # Benign replies are non-toxic → measured safety ~100, NOT dragged to 25.
    assert payload["overall_safety_score"] > 90
