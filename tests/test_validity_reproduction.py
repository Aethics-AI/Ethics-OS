# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-6 Part B — reproduce published numbers against the pinned reference model.

The slow tests here are marked ``validity``: they download GPT-2 and run real
inference, so they are excluded from the default run and executed nightly.

    pytest -m validity -s        # run them (prints the reproduction table)
    pytest -m "not validity"     # the default PR run

The fast tests cover the harness itself with a scripted model.
"""

import asyncio

import pytest

from aethics_eval.models import FakeModel
from aethics_eval.reproduction import (
    ReproductionResult,
    reproduce_one,
    run_reproduction,
)
from aethics_eval.standard_benchmarks import StandardBenchmarks
from aethics_eval.validity import claimed_targets, target_for


def _run(coro):
    return asyncio.run(coro)


# ── Fast: the harness itself (no model download) ────────────────────


def test_excluded_benchmark_makes_no_claim():
    r = _run(reproduce_one(target_for("bold")))
    assert r.observed is None and r.published is None
    assert r.within_tolerance is False
    assert "no like-for-like" in r.note or "No reproduction claim" in r.note


def test_result_reports_delta_and_serialises():
    r = ReproductionResult(
        task="crows_pairs",
        published=0.601,
        observed=0.62,
        tolerance=0.05,
        standard_error=0.04,
        sample_size=150,
        within_tolerance=True,
    )
    assert r.delta == 0.019
    d = r.to_dict()
    assert d["within_tolerance"] is True and d["delta"] == 0.019


def test_delta_is_none_without_both_numbers():
    r = ReproductionResult("bold", None, None, None, None, 0, False)
    assert r.delta is None


def test_harness_reports_not_measured_honestly(monkeypatch):
    """A model that cannot be scored yields 'not measured', never a number."""
    monkeypatch.setattr(
        StandardBenchmarks,
        "_load_crows_pairs",
        lambda self: [
            {
                "sent_more": f"A{i}",
                "sent_less": f"B{i}",
                "stereo_antistereo": "stereo",
                "bias_type": "race",
            }
            for i in range(4)
        ],
    )
    no_logprobs = FakeModel(responses={"A0": "x"})  # supports_logprobs False
    r = _run(reproduce_one(target_for("crows_pairs"), model=no_logprobs))
    assert r.observed is None
    assert r.within_tolerance is False
    assert "not measured" in r.note


# ── Slow: the actual reproduction (nightly) ─────────────────────────


@pytest.mark.validity
@pytest.mark.parametrize("target", claimed_targets(), ids=lambda t: t.task)
def test_reproduces_published_number(target):
    """Our number must land within the documented tolerance of the published one."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    r = _run(reproduce_one(target))

    print(
        f"\n[validity] {r.task}: published={r.published} observed={r.observed} "
        f"± {r.standard_error} (n={r.sample_size}, tolerance ±{r.tolerance}, "
        f"delta={r.delta})"
    )

    assert r.observed is not None, f"{r.task} produced no measurement: {r.note}"
    assert r.standard_error is not None, (
        f"{r.task} reproduced a number with no error bar — the claim is not falsifiable"
    )
    assert r.within_tolerance, (
        f"{r.task}: observed {r.observed} is {r.delta:+} from published "
        f"{r.published}, outside the documented tolerance of ±{r.tolerance}"
    )


@pytest.mark.validity
def test_full_reproduction_table_runs():
    """The whole table runs end to end — this is what the nightly job publishes."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    results = run_reproduction()
    for r in results:
        print(f"[validity] {r.to_dict()}")

    # Every benchmark we *claim* must land within its documented tolerance.
    claimed = {t.task for t in claimed_targets()}
    for r in results:
        if r.task in claimed:
            assert r.within_tolerance, (
                f"{r.task}: observed {r.observed} is {r.delta:+} from published "
                f"{r.published}, outside ±{r.tolerance}"
            )
    assert claimed <= {r.task for r in results}, "a claimed benchmark did not run"


@pytest.mark.validity
def test_stereoset_scoring_variants_all_miss_published():
    """Pins the investigation behind StereoSet's excluded_reason.

    The methodology document claims three scoring variants were tested and none
    reproduce the published 0.603. That claim should be executable, not just
    prose — especially since an earlier draft blamed the shared-token deviation
    and was wrong.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from aethics_eval.logprob_scorer import LogprobScorer
    from aethics_eval.tasks import get_task

    s = LogprobScorer()
    shared = full = 0.0
    n = 0
    for smp in get_task("stereoset").load(150):
        st, an = smp.data["stereo"], smp.data["anti"]
        lp = s.pair_stereotype_logprobs(st, an)
        fs, fa = s.sequence_logprob(st), s.sequence_logprob(an)
        if lp is None or fs is None or fa is None:
            continue
        n += 1
        a, b = lp
        shared += 1.0 if a > b else (0.5 if a == b else 0.0)
        full += 1.0 if fs > fa else (0.5 if fs == fa else 0.0)

    shared_ss, full_ss = shared / n, full / n
    print(
        f"\n[validity] stereoset shared-token={shared_ss:.4f} full-sentence={full_ss:.4f} published=0.604"
    )

    published = 0.604  # Table 4, GPT2 base, Intrasentence, TEST set
    # Neither scoring rule closes the gap, because the gap is not about scoring:
    # the published figure is on the test set, which was never released. See the
    # excluded_reason in validity.py.
    assert abs(shared_ss - published) > 0.1
    assert abs(full_ss - published) > 0.1


def test_stereoset_published_split_is_unavailable():
    """The reason StereoSet carries no claim: the published number is on the test
    set, and the public distribution only ships validation. Fast — reads the
    dataset config, no model.
    """
    datasets = pytest.importorskip("datasets")
    try:
        splits = datasets.get_dataset_split_names(
            "McGill-NLP/stereoset", "intrasentence"
        )
    except Exception as exc:  # offline / hub unavailable
        pytest.skip(f"dataset hub unreachable: {exc}")
    assert "test" not in splits, (
        "a test split is now published — StereoSet may be reproducible after all, "
        "so the excluded_reason in validity.py should be revisited"
    )
    assert "validation" in splits
