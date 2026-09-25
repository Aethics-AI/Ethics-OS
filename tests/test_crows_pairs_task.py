# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-3 Part B — CrowS-Pairs migrated onto the Task registry.

The parity guard: the registry task must produce a byte-identical result to the
pre-refactor engine (StandardBenchmarks.run_crows_pairs) on the same data and
model. Offline — the loader is monkeypatched to fixed rows and a FakeModel
supplies scripted pair log-probs.
"""

import asyncio

from aethics_eval.models import FakeModel
from aethics_eval.standard_benchmarks import StandardBenchmarks
from aethics_eval.tasks import get_task, list_tasks, run_task


def _run(coro):
    return asyncio.run(coro)


# Fixed rows: exercise stereo + antistereo directions, a tie, and repeated
# categories (so category_counts is non-trivial).
_ROWS = [
    {
        "sent_more": "A",
        "sent_less": "B",
        "stereo_antistereo": "stereo",
        "bias_type": "race",
    },
    {
        "sent_more": "C",
        "sent_less": "D",
        "stereo_antistereo": "antistereo",
        "bias_type": "gender",
    },
    {
        "sent_more": "E",
        "sent_less": "F",
        "stereo_antistereo": "stereo",
        "bias_type": "race",
    },
    {
        "sent_more": "G",
        "sent_less": "H",
        "stereo_antistereo": "stereo",
        "bias_type": "age",
    },
]

# pair_stereotype_logprobs is called with (more_stereotypical, less_stereotypical)
# AFTER direction handling: row1 (antistereo) flips to (D, C).
_PAIR_LOGPROBS = {
    ("A", "B"): (-1.0, -2.0),  # prefers stereo -> 1.0
    ("D", "C"): (-3.0, -1.0),  # prefers less  -> 0.0
    ("E", "F"): (-2.0, -2.0),  # tie           -> 0.5
    ("G", "H"): (-1.0, -5.0),  # prefers stereo -> 1.0
}


def test_crows_pairs_is_registered():
    assert "crows_pairs" in {m.name for m in list_tasks()}
    meta = get_task("crows_pairs").meta
    assert meta.requires_logprobs is True
    assert meta.licence == "CC-BY-SA-4.0"


def test_crows_pairs_registry_matches_legacy_byte_identical(monkeypatch):
    monkeypatch.setattr(
        StandardBenchmarks, "_load_crows_pairs", lambda self: list(_ROWS)
    )
    model = FakeModel(pair_logprobs=dict(_PAIR_LOGPROBS))

    legacy = _run(StandardBenchmarks().run_crows_pairs(model, sample_size=4)).to_dict()
    migrated = _run(run_task(get_task("crows_pairs"), model, limit=4)).to_dict()

    assert migrated == legacy, (
        "registry CrowS-Pairs diverged from the pre-refactor engine"
    )
    # sanity: it actually measured (not the not_measured path)
    assert legacy["score"] is not None and legacy["samples_tested"] == 4


def test_crows_pairs_not_measured_without_logprobs(monkeypatch):
    monkeypatch.setattr(
        StandardBenchmarks, "_load_crows_pairs", lambda self: list(_ROWS)
    )
    # A model with no logprob support -> both paths report not_measured.
    no_lp = FakeModel(
        responses={"A": "x"}
    )  # supports_logprobs False, no pair method result
    legacy = _run(StandardBenchmarks().run_crows_pairs(no_lp, sample_size=4)).to_dict()
    migrated = _run(run_task(get_task("crows_pairs"), no_lp, limit=4)).to_dict()

    assert legacy["score"] is None
    assert migrated["score"] is None
