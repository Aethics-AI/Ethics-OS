# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-11 — the manifest leaf hash and SampleScore must not drift apart.

PR #64 dropped ``response_digest`` from ``SampleScore`` while ``manifest.py``
carried on hashing it into every leaf. ``aethics verify`` then failed for files
nobody had touched: the recorded root had been computed over a field the
persisted evidence no longer carried, so recomputation could not reproduce it.

That is the worst shape a bug can take here. ``verify`` exists to tell an
auditor whether a result was altered, so a false "this was tampered with" is
more damaging than no check at all — it teaches people to distrust the tool
rather than the data.

The fix landed. Nothing stopped it recurring, which is what this file is for.

**Deliberately not a hardcoded list of fields.** A list would have to be updated
by the same person making the change that breaks the invariant, which is the
one moment they are not thinking about it. Instead the hashed fields are
discovered by mutation: change a value, and if the hash moves, that field is
part of the evidence chain. That finding survives any future edit to
``leaf_hash``.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Dict, Set

import pytest

from aethics_eval.manifest import build_manifest, leaf_hash, merkle_root
from aethics_eval.tasks.base import SampleScore

# A value that is valid for every parameter type leaf_hash currently takes and
# is distinguishable from the baseline. Mutation only has to change the hash;
# it does not have to be meaningful.
_BASELINE: Dict[str, Any] = {
    "sample_id": "sample-0",
    "value": 1.0,
    "measured": True,
    "response_digest": "a" * 64,
}
_MUTATED: Dict[str, Any] = {
    "sample_id": "sample-1",
    "value": 0.5,
    "measured": False,
    "response_digest": "b" * 64,
}


def _leaf_parameters() -> Set[str]:
    """Parameter names `leaf_hash` accepts, read from the function itself."""
    return set(inspect.signature(leaf_hash).parameters)


def _hashed_fields() -> Set[str]:
    """Which parameters actually affect the leaf hash.

    Determined by mutation rather than by reading the body: a parameter is part
    of the evidence chain if and only if changing it changes the digest. A
    parameter that is accepted and ignored is not part of the chain and is not
    this test's business.
    """
    hashed: Set[str] = set()
    baseline = leaf_hash(**_BASELINE)

    for name in _leaf_parameters():
        if name not in _MUTATED:
            pytest.fail(
                f"leaf_hash gained a parameter this test has no sample value "
                f"for: {name!r}. Add one to _BASELINE and _MUTATED so the guard "
                f"can keep checking it."
            )
        altered = {**_BASELINE, name: _MUTATED[name]}
        if leaf_hash(**altered) != baseline:
            hashed.add(name)

    return hashed


def _sample_score_fields() -> Set[str]:
    return {f.name for f in dataclasses.fields(SampleScore)}


def test_every_hashed_field_exists_on_sample_score() -> None:
    """The invariant PR #64 broke.

    Anything hashed into a leaf has to be something a `SampleScore` can carry.
    If it is not, the value reaching the hash at write time cannot be recovered
    from the persisted evidence at verify time, and `verify` reports tampering
    that did not happen.
    """
    hashed = _hashed_fields()
    available = _sample_score_fields()

    missing = sorted(hashed - available)
    assert not missing, (
        "These fields change the manifest leaf hash but do not exist on "
        f"SampleScore: {', '.join(missing)}.\n\n"
        "That is the PR #64 regression: the evidence root gets computed over a "
        "field the persisted evidence cannot carry, so `aethics verify` fails "
        "for files nobody edited.\n\n"
        "Either add the field to SampleScore, or stop hashing it into the leaf."
    )


def test_the_guard_detects_a_field_that_is_not_on_sample_score() -> None:
    """Negative test: prove this file would actually catch the regression.

    A guard that cannot fail is not a guard. This reproduces the #64 shape —
    a field in the leaf payload with no home on SampleScore — and asserts the
    comparison above notices.
    """
    hashed_with_intruder = _hashed_fields() | {"a_field_nobody_declared"}
    missing = sorted(hashed_with_intruder - _sample_score_fields())

    assert missing == ["a_field_nobody_declared"], (
        "The comparison in this file no longer detects a hashed field that is "
        "absent from SampleScore, so the guard above is vacuous."
    )


def test_leaf_hash_actually_hashes_something() -> None:
    """Guard the guard, second way.

    If `leaf_hash` stopped depending on its inputs, `_hashed_fields()` would
    return the empty set and the invariant test would pass by describing
    nothing. Assert the chain is load-bearing.
    """
    hashed = _hashed_fields()
    assert hashed, (
        "No parameter of leaf_hash changes its output. Either the function no "
        "longer hashes its inputs, or the mutation values are indistinguishable "
        "from the baseline — both make this file meaningless."
    )
    # `value` and `measured` are the two the honesty rules depend on: a skipped
    # sample and a zero-scoring sample must not hash alike.
    for essential in ("value", "measured"):
        assert essential in hashed, (
            f"{essential!r} no longer affects the leaf hash. A sample that was "
            f"not measured and one that scored zero would become "
            f"indistinguishable in the evidence chain."
        )


def test_generated_run_round_trips_through_the_evidence_chain() -> None:
    """End to end: scores in, manifest out, recomputation matches.

    The field-level checks above compare shapes. This one runs the actual path
    a result takes, so a change that keeps the shapes aligned but breaks the
    arithmetic still fails.
    """
    scores = [
        SampleScore("s-0", 1.0, measured=True),
        SampleScore("s-1", 0.0, measured=False),
        SampleScore("s-2", 0.5, measured=True),
    ]
    rows = [
        {"sample_id": s.sample_id, "value": s.value, "measured": s.measured}
        for s in scores
    ]

    manifest = build_manifest(
        run_id="run-0",
        methodology_fingerprint="f" * 64,
        library_version="0.0.0-test",
        seeds={"seed": 42},
        model={"spec": "fake:test", "params": {}},
        datasets=[],
        task_samples={"demo": rows},
    )
    assert len(manifest.tasks) == 1
    recorded_root = manifest.tasks[0].evidence_root

    recomputed = merkle_root(
        [leaf_hash(r["sample_id"], r["value"], r["measured"]) for r in rows]
    )
    assert recomputed == recorded_root, (
        "Recomputing the evidence root from the same samples produced a "
        "different value. Verification of an unmodified result would fail."
    )

    # And altering one sample must move the root, or the chain proves nothing.
    tampered = [dict(r) for r in rows]
    tampered[0]["value"] = 0.9
    altered = merkle_root(
        [leaf_hash(r["sample_id"], r["value"], r["measured"]) for r in tampered]
    )
    assert altered != recorded_root, (
        "Editing a sample did not change the evidence root, so the manifest "
        "cannot detect tampering."
    )
