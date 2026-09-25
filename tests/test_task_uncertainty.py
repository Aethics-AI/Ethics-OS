# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-5 Part C — every task emits uncertainty.

The DoD guard: no metric ships without an uncertainty field or an explicit
documented reason. The only permitted reason is not_measured — there is no
metric value to put error bars around.

Also asserts the VOS-3 byte-identical parity is preserved: uncertainty rides on
the runtime TaskResult, deliberately NOT in to_dict(), which is the legacy
BenchmarkResult shape the parity tests compare.
"""

import asyncio

import pytest

from aethics_eval.models import FakeModel
from aethics_eval.standard_benchmarks import StandardBenchmarks
from aethics_eval.tasks import get_task, run_task


def _run(coro):
    return asyncio.run(coro)


# ── Fixtures: enough samples that the stats are computable ──────────

_N = 8

_CROWS_ROWS = [
    {
        "sent_more": f"A{i}",
        "sent_less": f"B{i}",
        "stereo_antistereo": "stereo",
        "bias_type": "race",
    }
    for i in range(_N)
]
_CROWS_LP = {(f"A{i}", f"B{i}"): (-1.0 - i * 0.1, -2.0) for i in range(_N)}


def _stereoset_examples():
    return [
        {
            "sentences": {
                "gold_label": [0, 1, 2],
                "sentence": [f"S{i}", f"A{i}", f"U{i}"],
            },
            "bias_type": "race",
        }
        for i in range(_N)
    ]


_STEREO_LP = {(f"S{i}", f"A{i}"): (-1.0 - i * 0.1, -2.0) for i in range(_N)}


def _wino_row(occ_gold, occ2, pronoun):
    return {
        "tokens": [
            "The",
            occ2,
            "called",
            "the",
            occ_gold,
            "because",
            pronoun,
            "was",
            "late",
            ".",
        ],
        "coreference_clusters": ["3", "4", "6", "6"],
    }


_WINO_PRO = [_wino_row(f"nurse{i}", f"doctor{i}", "she") for i in range(_N)]
_WINO_ANTI = [_wino_row(f"nurse{i}", f"doctor{i}", "he") for i in range(_N)]


def _bold_rows():
    rows = []
    for domain in ("gender", "race"):
        for i in range(_N // 2):
            rows.append(
                {
                    "domain": domain,
                    "category": f"{domain}_cat{i}",
                    "prompts": [f"{domain} prompt {i}"],
                    "name": f"{domain}{i}",
                }
            )
    return rows


def _content_logprob(text):
    return -float(sum(ord(c) for c in text))


@pytest.fixture
def all_tasks(monkeypatch):
    """Each registered task paired with a model that can actually score it."""
    monkeypatch.setattr(
        StandardBenchmarks, "_load_crows_pairs", lambda self: list(_CROWS_ROWS)
    )
    monkeypatch.setattr(
        StandardBenchmarks, "_load_stereoset", lambda self: _stereoset_examples()
    )
    monkeypatch.setattr(
        StandardBenchmarks,
        "_load_winobias",
        lambda self: (list(_WINO_PRO), list(_WINO_ANTI)),
    )
    monkeypatch.setattr(StandardBenchmarks, "_load_bold", lambda self: _bold_rows())
    return [
        ("crows_pairs", FakeModel(pair_logprobs=dict(_CROWS_LP))),
        ("stereoset", FakeModel(pair_logprobs=dict(_STEREO_LP))),
        ("winobias", FakeModel(logprobs=_content_logprob)),
        ("bold", FakeModel(default_response="a calm and pleasant remark")),
    ]


# ── The DoD guard ───────────────────────────────────────────────────


def test_no_metric_ships_without_uncertainty(all_tasks):
    for name, model in all_tasks:
        result = _run(run_task(get_task(name), model, limit=_N))
        if result.score is None:
            continue  # not_measured: no value to put error bars around
        assert result.uncertainty is not None, (
            f"{name} produced a score with no uncertainty"
        )
        u = result.uncertainty
        assert u.sample_size > 0, f"{name} uncertainty has no sample_size"
        assert u.standard_error is not None, f"{name} has no standard error"
        assert u.confidence_interval is not None, f"{name} has no confidence interval"
        assert u.method, f"{name} does not say how uncertainty was computed"


def test_matched_tasks_use_paired_significance(all_tasks):
    """Stereotype pairs and pro/anti sets are matched — they must use the paired
    test, not the unpaired one that discards the pairing."""
    for name, model in all_tasks:
        if name == "bold":
            continue  # cross-domain, genuinely unpaired
        result = _run(run_task(get_task(name), model, limit=_N))
        assert "paired_permutation" in result.uncertainty.method, name
        assert result.uncertainty.p_value is not None, name


def test_bold_is_unpaired_and_says_so(all_tasks):
    # BOLD compares domains of differing sizes — no matched pairs to exploit.
    result = _run(run_task(get_task("bold"), dict(all_tasks)["bold"], limit=_N))
    if result.score is not None:
        assert "paired_permutation" not in (result.uncertainty.method or "")


def test_underpowered_flag_is_set_on_thin_runs(all_tasks):
    """DoD: a run below the task's min_samples is flagged, not silently averaged."""
    for name, model in all_tasks:
        result = _run(run_task(get_task(name), model, limit=_N))  # 8 << every threshold
        if result.score is not None:
            assert result.uncertainty.underpowered is True, (
                f"{name} not flagged at n={_N}"
            )


def test_uncertainty_is_serialisable(all_tasks):
    name, model = all_tasks[0]
    result = _run(run_task(get_task(name), model, limit=_N))
    d = result.uncertainty_dict()
    assert isinstance(d, dict) and d["sample_size"] > 0


# ── Parity is preserved ─────────────────────────────────────────────


def test_uncertainty_stays_out_of_legacy_to_dict(all_tasks):
    """to_dict() is the byte-identical legacy shape the VOS-3/VOS-7 parity guard
    compares. Uncertainty must not leak into it."""
    for name, model in all_tasks:
        result = _run(run_task(get_task(name), model, limit=_N))
        assert "uncertainty" not in result.to_dict(), (
            f"{name} leaked uncertainty into to_dict()"
        )
