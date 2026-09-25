# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the counterfactual bias analyzer (V2).

These test the pure scoring logic (_analyze_counterfactual) with mock responses
and a deterministic stub scorer — no model, no network, no database. The stub
treats any text containing "TOXIC" as toxic (0.9) and everything else as clean
(0.1), so the expected scores are exact.
"""

import asyncio

from aethics_eval.eval_datasets import (
    build_demographic_counterfactual_pairs,
    build_gender_counterfactual_pairs,
)
from aethics_eval.nlp_bias_tests import NLPBiasDetector

detector = NLPBiasDetector()


def stub_scorer(text: str) -> float:
    """Deterministic harm score: 0.9 if the text is marked TOXIC, else 0.1."""
    return 0.9 if "TOXIC" in text else 0.1


def _gender_pairs(n):
    return build_gender_counterfactual_pairs()[:n]


def test_identical_responses_score_as_fair():
    """Same response to both halves → zero bias, perfect canonical value."""
    pairs = _gender_pairs(5)
    same = ["a neutral, ordinary answer"] * 5
    out = detector._analyze_counterfactual(pairs, same, same, stub_scorer)

    assert out["score"] == 0.0  # raw bias, lower is better
    assert out["scored_metric"].value == 100.0  # canonical, higher is better
    assert out["scored_metric"].reliability == "direct"
    assert out["details"]["flagged_pairs"] == 0


def test_systematic_difference_is_caught_and_flagged():
    """Group B consistently toxic while A is clean → high bias, flagged pairs."""
    pairs = _gender_pairs(5)
    clean = ["a calm, friendly reply"] * 5
    toxic = ["a TOXIC, hostile reply"] * 5
    out = detector._analyze_counterfactual(pairs, clean, toxic, stub_scorer)

    assert out["score"] > 50  # clearly biased
    assert out["details"]["pairs_tested"] == 5
    assert out["details"]["flagged_pairs"] == 5
    # Group B got the toxic response, so B is the disfavored group.
    assert out["examples"][0]["disfavored_group"] == pairs[0].group_b


def test_no_reference_group_white_can_be_flagged():
    """
    The structural fix for the old "white": [] bug: because we measure the
    difference between two groups, EITHER side can be flagged as disfavored —
    including white. There is no exempt baseline.
    """
    white_black = [
        p
        for p in build_demographic_counterfactual_pairs()
        if (p.group_a, p.group_b) == ("white", "black")
    ][:3]

    # White responses toxic, black clean → white must be flagged as disfavored.
    out = detector._analyze_counterfactual(
        white_black, ["a TOXIC reply"] * 3, ["a clean reply"] * 3, stub_scorer
    )
    assert out["details"]["flagged_pairs"] == 3
    assert out["examples"][0]["disfavored_group"] == "white"


def test_no_usable_responses_is_not_measured():
    """Missing responses must not fabricate a score."""
    pairs = _gender_pairs(3)
    out = detector._analyze_counterfactual(
        pairs, [None, None, None], [None, None, None], stub_scorer
    )

    assert out["details"]["pairs_tested"] == 0
    assert out["scored_metric"].reliability == "not_measured"
    assert out["scored_metric"].sample_size == 0


def test_partial_differences_land_between_fair_and_biased():
    """A mix of fair and biased pairs scores in the middle, with a CI attached."""
    pairs = _gender_pairs(4)
    # 2 fair pairs, 2 biased pairs
    a = ["clean", "clean", "clean", "clean"]
    b = ["clean", "clean", "TOXIC", "TOXIC"]
    out = detector._analyze_counterfactual(pairs, a, b, stub_scorer)

    assert 0 < out["score"] < 80
    assert out["details"]["flagged_pairs"] == 2
    lo, hi = out["scored_metric"].confidence_interval
    assert lo <= out["scored_metric"].value <= hi


# ── Step 3: end-to-end wiring (mocked model + classifier) ──────────


def test_gender_bias_wired_end_to_end():
    """
    test_gender_bias() runs the full counterfactual flow with mocked I/O:
    build pairs -> query both halves -> score for toxicity -> analyze.
    Here the female (group B) half always gets a toxic response, so the model
    must be flagged as biased against the female group.
    """
    d = NLPBiasDetector()

    async def fake_query(prompts):
        # First half = A prompts (male), second half = B prompts (female).
        half = len(prompts) // 2
        out = []
        for i in range(len(prompts)):
            text = "a TOXIC hostile reply" if i >= half else "a calm friendly reply"
            out.append({"success": True, "data": [{"generated_text": text}]})
        return out

    async def fake_score(texts):
        return {t: (0.9 if "TOXIC" in t else 0.1) for t in texts if t}

    # Replace the I/O methods on this instance (no model, no network).
    d._query_model_safely = fake_query
    d._score_responses_for_toxicity = fake_score

    result = asyncio.run(d.test_gender_bias())

    assert result.test_name == "gender_bias"
    assert result.bias_score > 50  # biased against the disfavored group
    assert result.scored_metric.reliability == "direct"
    assert result.details["method"] == "counterfactual_paired"
    assert result.details["flagged_pairs"] > 0


def test_gender_bias_reports_not_measured_when_model_fails():
    """
    If the model returns nothing usable (e.g. the inference bug), the wired test
    reports not_measured instead of a fabricated score — the honest outcome.
    """
    d = NLPBiasDetector()

    async def failing_query(prompts):
        return [{"success": False, "data": None} for _ in prompts]

    d._query_model_safely = failing_query

    result = asyncio.run(d.test_gender_bias())

    assert result.scored_metric.reliability == "not_measured"
    assert result.details["pairs_tested"] == 0
