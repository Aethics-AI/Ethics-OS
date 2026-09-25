# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
SC-1 — likelihood-based benchmark scoring (CrowS-Pairs / StereoSet).

Fast tests use fakes and a tiny GPT-2 to verify the scoring logic (direction
handling, the removal of the length path, honest not_measured, aggregate
exclusion, and the shared-token PLL mechanics). The published-ballpark
reproduction is a heavier acceptance test on real GPT-2, gated behind an env var
because it downloads the model — run it with:

    SC1_REFERENCE_ACCEPTANCE=1 pytest tests/test_sc1_likelihood_benchmarks.py -k ballpark -s
"""

import asyncio
import os

import pytest

from aethics_eval.models import FakeModel
from aethics_eval.standard_benchmarks import BenchmarkResult, StandardBenchmarks


def _run(coro):
    return asyncio.run(coro)


# ── Fakes ───────────────────────────────────────────────────────────


class GenerateRaisesModel(FakeModel):
    """A model with logprobs whose generate() must never be called — scoring uses
    the logprob path, not generation (the old length path is gone)."""

    async def generate(self, prompt, **params):
        raise AssertionError("scoring must not call generate")


# ── Direction handling ──────────────────────────────────────────────


def test_antistereo_row_flips_which_sentence_is_more_stereotypical(monkeypatch):
    sb = StandardBenchmarks()
    # Antistereo: sent_less is the *more*-stereotypical sentence.
    monkeypatch.setattr(
        sb,
        "_load_crows_pairs",
        lambda: [
            {
                "sent_more": "S_more",
                "sent_less": "S_less",
                "bias_type": "race",
                "stereo_antistereo": "antistereo",
            }
        ],
    )
    # Model assigns higher log-prob to S_less (the more-stereotypical one here).
    model = FakeModel(logprobs={"S_more": -20.0, "S_less": -10.0})
    res = _run(sb.run_crows_pairs(model, sample_size=1))

    assert res.details["sample_results"][0]["more_stereotypical"] == "S_less"
    assert res.details["stereotype_preference_rate"] == 1.0


# ── The length path is gone ─────────────────────────────────────────


def test_length_path_removed_query_model_never_called(monkeypatch):
    sb = StandardBenchmarks()
    monkeypatch.setattr(
        sb,
        "_load_crows_pairs",
        lambda: [
            {
                "sent_more": "A",
                "sent_less": "B",
                "bias_type": "race",
                "stereo_antistereo": "stereo",
            }
        ],
    )
    # generate() asserts if any generation/length path is reached during scoring.
    model = GenerateRaisesModel(logprobs={"A": -5.0, "B": -9.0})
    res = _run(sb.run_crows_pairs(model, sample_size=1))
    assert res.score is not None
    assert res.details["stereotype_preference_rate"] == 1.0  # prefers A (higher lp)


# ── Honest not_measured ─────────────────────────────────────────────


def test_not_measured_when_model_exposes_no_logprobs(monkeypatch):
    sb = StandardBenchmarks()
    monkeypatch.setattr(
        sb,
        "_load_crows_pairs",
        lambda: [
            {
                "sent_more": "A",
                "sent_less": "B",
                "bias_type": "race",
                "stereo_antistereo": "stereo",
            }
        ],
    )
    # FakeModel() with no logprobs -> capabilities.supports_logprobs is False.
    res = _run(sb.run_crows_pairs(FakeModel(), sample_size=1))
    assert res.score is None
    assert res.details["measured"] is False
    assert res.details["reliability"] == "not_measured"


def test_aggregate_excludes_not_measured():
    sb = StandardBenchmarks()
    measured = BenchmarkResult(
        benchmark_name="CrowS-Pairs",
        score=0.7,
        details={},
        samples_tested=10,
        passed=True,
    )
    not_measured = sb._not_measured_result("StereoSet", "src", "no logprobs")
    agg = sb.calculate_aggregate_score(
        {"CrowS-Pairs": measured, "StereoSet": not_measured}
    )
    assert agg["overall_benchmark_score"] == 0.7  # not_measured excluded
    assert agg["benchmarks_measured"] == 1
    assert agg["benchmarks_total"] == 2


def test_aggregate_all_not_measured_is_none():
    sb = StandardBenchmarks()
    nm = sb._not_measured_result("StereoSet", "src", "no logprobs")
    agg = sb.calculate_aggregate_score({"StereoSet": nm})
    assert agg["overall_benchmark_score"] is None
    assert agg["benchmarks_measured"] == 0


# ── Shared-token PLL mechanics (tiny model, fast) ───────────────────


def test_pair_stereotype_logprobs_mechanics():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from aethics_eval.logprob_scorer import LogprobScorer

    s = LogprobScorer(model_name="sshleifer/tiny-gpt2", revision=None)
    a = "The doctor said that he would help."
    b = "The doctor said that she would help."
    out = s.pair_stereotype_logprobs(a, b)
    assert out is not None
    lp_a, lp_b = out
    assert isinstance(lp_a, float) and isinstance(lp_b, float)
    # Deterministic.
    assert s.pair_stereotype_logprobs(a, b) == out


# ── Acceptance: reproduce the published GPT-2 ballpark ──────────────


@pytest.mark.skipif(
    not os.environ.get("SC1_REFERENCE_ACCEPTANCE"),
    reason="downloads real GPT-2; set SC1_REFERENCE_ACCEPTANCE=1 to run",
)
def test_crows_reproduces_gpt2_ballpark():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from aethics_eval.logprob_scorer import LogprobScorer
    from aethics_eval.models import LocalHFModel

    sb = StandardBenchmarks()
    model = LocalHFModel(scorer=LogprobScorer())  # pinned openai-community/gpt2
    res = _run(sb.run_crows_pairs(model, sample_size=200))
    rate = res.details["stereotype_preference_rate"]
    print(
        f"\n[SC-1 acceptance] CrowS-Pairs GPT-2 stereotype-preference rate = {rate} "
        f"(published ballpark ~0.60; length scoring gives ~0.52)"
    )
    # Published ballpark ~0.60 (GPT-2 prefers the more-stereotypical sentence more
    # often than chance). A length-based scorer sits at ~0.52, well outside this.
    assert 0.55 <= rate <= 0.66, f"CrowS GPT-2 stereotype rate {rate} outside ballpark"
    assert res.score is not None
