# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for two BOLD defects found while building the CLI (POS-3).

Both were invisible from the outside: the benchmark returned a plausible
number every time, so nothing looked wrong.

1. BOLD ships grouped by domain — rows 0-2362 are all ``gender``, and the five
   domains are contiguous blocks. ``run_bold`` sampled ``prompts[:n]``, so any
   n below 2364 produced a single-domain sample. BOLD's metric is *cross-domain*
   sentiment disparity, so a single-domain sample cannot yield a score at all.
   The default sample_size is 30. The comparison was therefore never made.

2. On that path ``run_bold`` returned a hardcoded ``score=0.5`` with
   ``samples_tested=0``, and ``calculate_aggregate_score`` averaged it in as a
   real measurement. That is the F6 failure mode — neutral defaults standing in
   for things that were never measured — which was fixed for the other three
   benchmarks and missed here.

Together they meant BOLD reported 0.5 for every model, every run, regardless
of the model's actual behaviour.
"""

from __future__ import annotations

from aethics_eval.standard_benchmarks import _stratify_by_domain

# Shaped like BOLD: contiguous blocks, wildly uneven sizes.
ROWS = (
    [{"domain": "gender", "name": f"g{i}"} for i in range(50)]
    + [{"domain": "race", "name": f"r{i}"} for i in range(30)]
    + [{"domain": "profession", "name": f"p{i}"} for i in range(5)]
)


def test_naive_slicing_would_yield_one_domain() -> None:
    """Documents the bug, so the fix below has a stated reason to exist."""
    assert len({r["domain"] for r in ROWS[:30]}) == 1


def test_stratified_sample_spans_every_domain() -> None:
    picked = _stratify_by_domain(ROWS, 30)
    assert len(picked) == 30
    assert {r["domain"] for r in picked} == {"gender", "race", "profession"}


def test_small_samples_still_span_domains() -> None:
    """n equal to the domain count must give one of each, not three of one."""
    picked = _stratify_by_domain(ROWS, 3)
    assert {r["domain"] for r in picked} == {"gender", "race", "profession"}


def test_stratification_is_deterministic() -> None:
    """No sampling, no seed to record — two runs must select the same rows.

    Reproducibility is the whole point of the run manifest (POS-5); a random
    sample here would silently undermine it.
    """
    assert _stratify_by_domain(ROWS, 17) == _stratify_by_domain(ROWS, 17)


def test_exhausted_domains_do_not_stall_or_duplicate() -> None:
    """`profession` has 5 rows; asking for 60 must not loop or repeat them."""
    picked = _stratify_by_domain(ROWS, 60)
    assert len(picked) == 60
    names = [r["name"] for r in picked]
    assert len(names) == len(set(names)), "a row was selected twice"


def test_requesting_everything_returns_everything() -> None:
    assert len(_stratify_by_domain(ROWS, len(ROWS) + 100)) == len(ROWS)


def test_unreachable_model_yields_none_not_a_neutral_score() -> None:
    """The F6 check, exercised rather than grepped.

    Every generation fails, so no domain accumulates data. BOLD used to answer
    0.5 here with samples_tested=0 — a fabricated number that
    calculate_aggregate_score then averaged in as a real measurement.
    """
    import asyncio

    from aethics_eval.models import Capabilities, Generation
    from aethics_eval.standard_benchmarks import create_benchmark_suite

    class DeadModel:
        """Every request fails, as an unreachable endpoint would."""

        @property
        def capabilities(self) -> Capabilities:
            return Capabilities(
                supports_logprobs=False, supports_echo=False, max_context=2048
            )

        async def generate(self, prompt: str, **_: object) -> Generation:
            return Generation(text="", success=False, error="unreachable")

        async def sequence_logprob(self, text: str) -> None:
            return None

    result = asyncio.run(create_benchmark_suite().run_bold(DeadModel(), sample_size=10))

    assert result.score is None, (
        f"BOLD returned {result.score!r} having measured nothing. A score with "
        f"no samples behind it is a fabricated measurement."
    )
    assert result.samples_tested == 0
    assert result.passed is False


def test_aggregate_excludes_unmeasured_benchmarks() -> None:
    """A not_measured benchmark must not drag the mean toward its placeholder."""
    from aethics_eval.standard_benchmarks import BenchmarkResult, create_benchmark_suite

    suite = create_benchmark_suite()
    agg = suite.calculate_aggregate_score(
        {
            "A": BenchmarkResult("A", 0.8, {}, samples_tested=10, passed=True),
            "B": BenchmarkResult("B", None, {}, samples_tested=0, passed=False),
        }
    )

    assert agg["overall_benchmark_score"] == 0.8, "unmeasured result was averaged in"
    assert agg["benchmarks_measured"] == 1
    assert agg["benchmarks_total"] == 2


def test_run_bold_actually_uses_stratified_sampling() -> None:
    """Guards the call site, not just the helper.

    The helper can be correct while `run_bold` still slices `prompts[:n]`,
    which is exactly the state this fix found. Exercised end to end: a small
    sample must still span more than one domain, because BOLD's metric is a
    comparison *between* domains.
    """
    import asyncio

    from aethics_eval.models import Capabilities, Generation
    from aethics_eval.standard_benchmarks import create_benchmark_suite

    class ConstantModel:
        @property
        def capabilities(self) -> Capabilities:
            return Capabilities(
                supports_logprobs=False, supports_echo=False, max_context=2048
            )

        async def generate(self, prompt: str, **_: object) -> Generation:
            return Generation(text="a helpful and kind person", success=True)

        async def sequence_logprob(self, text: str) -> None:
            return None

    result = asyncio.run(
        create_benchmark_suite().run_bold(ConstantModel(), sample_size=10)
    )

    domains = result.details.get("domains_tested", [])
    assert len(domains) > 1, (
        f"BOLD sampled {len(domains)} domain(s) from 10 prompts: {domains}. "
        f"The dataset is ordered by domain, so prompts[:n] yields one domain "
        f"for any n below ~2364 and the cross-domain comparison never happens."
    )
    assert result.score is not None, "a multi-domain sample should produce a score"
