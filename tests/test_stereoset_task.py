# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-3 Part B — StereoSet migrated onto the Task registry, byte-identical to
StandardBenchmarks.run_stereoset. Offline: loader monkeypatched to fixed
examples, FakeModel supplies scripted pair log-probs.
"""

import asyncio

from aethics_eval.models import FakeModel
from aethics_eval.standard_benchmarks import StandardBenchmarks
from aethics_eval.tasks import get_task, list_tasks, run_task


def _run(coro):
    return asyncio.run(coro)


def _ex(labels, sents, bias_type):
    return {
        "sentences": {"gold_label": labels, "sentence": sents},
        "bias_type": bias_type,
    }


# ex3 has no anti-stereotype (no label 1) → skipped, like the engine.
_EXAMPLES = [
    _ex([0, 1, 2], ["S0", "A0", "U0"], "race"),
    _ex([2, 1, 0], ["U1", "A1", "S1"], "gender"),
    _ex([0, 1, 2], ["S2", "A2", "U2"], "race"),
    _ex([0, 2, 2], ["S3", "U3a", "U3b"], "religion"),  # skipped
]

_PAIR_LOGPROBS = {
    ("S0", "A0"): (-1.0, -2.0),  # -> 1.0
    ("S1", "A1"): (-3.0, -1.0),  # -> 0.0
    ("S2", "A2"): (-2.0, -2.0),  # -> 0.5
}


def test_stereoset_is_registered():
    assert "stereoset" in {m.name for m in list_tasks()}
    assert get_task("stereoset").meta.requires_logprobs is True


def test_stereoset_registry_matches_legacy_byte_identical(monkeypatch):
    monkeypatch.setattr(
        StandardBenchmarks, "_load_stereoset", lambda self: list(_EXAMPLES)
    )
    model = FakeModel(pair_logprobs=dict(_PAIR_LOGPROBS))

    legacy = _run(StandardBenchmarks().run_stereoset(model, sample_size=4)).to_dict()
    migrated = _run(run_task(get_task("stereoset"), model, limit=4)).to_dict()

    assert migrated == legacy, (
        "registry StereoSet diverged from the pre-refactor engine"
    )
    assert legacy["score"] is not None and legacy["samples_tested"] == 3  # ex3 skipped
    assert legacy["details"]["categories"] == {"race": 2, "gender": 1}
