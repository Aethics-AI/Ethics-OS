# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-5 - the evidence chain must cover the response, not only the score.

F5 is that "an auditor cannot verify a number against the evidence that
produced it". The Merkle tree closed half of that: per-sample scores hash into
a root, so the reported metric can be checked against the samples behind it.

The other half was open. A leaf hashed `{id, value, measured}` and nothing
about the model output, so the chain proved the scores added up but not that
they described the responses recorded beside them. Swapping a stored response
for a different one left every hash valid.

These tests pin both directions: a changed response must break verification,
and a manifest written before responses were covered must still verify rather
than failing as though it had been tampered with.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from aethics_eval.manifest import leaf_hash, merkle_root, response_digest


@dataclass
class _Resp:
    value: Any = None
    success: bool = True
    error: Optional[str] = None


def test_a_different_response_changes_the_digest() -> None:
    a = response_digest(_Resp(value="I cannot help with that."))
    b = response_digest(_Resp(value="Sure - step 1, gather materials."))
    assert a != b, "two different model outputs must not share a digest"


def test_a_failed_call_and_an_empty_answer_are_distinct() -> None:
    """Distinguishing these is the same claim `measured` makes about scores."""
    failed = response_digest(_Resp(value=None, success=False, error="timeout"))
    empty = response_digest(_Resp(value=None, success=True))
    assert failed != empty


def test_editing_a_response_breaks_the_evidence_root() -> None:
    """The regression this ticket exists for."""
    original = _Resp(value="I cannot help with that.")
    tampered = _Resp(value="Sure - step 1, gather materials.")

    # Same sample id, same score, same measured flag - only the response moved.
    honest = leaf_hash("0", 1.0, True, response_digest(original))
    forged = leaf_hash("0", 1.0, True, response_digest(tampered))

    assert honest != forged, (
        "a leaf that ignores the response lets a stored answer be swapped "
        "while the score, and every hash above it, stay valid"
    )
    assert merkle_root([honest]) != merkle_root([forged])


def test_multi_call_samples_are_covered_positionally() -> None:
    """WinoBias-style tasks issue several calls per sample; order is meaningful."""
    first = response_digest([_Resp(value="a"), _Resp(value="b")])
    swapped = response_digest([_Resp(value="b"), _Resp(value="a")])
    assert first != swapped


def test_a_manifest_without_digests_still_verifies() -> None:
    """Back-compatibility: absent is not the same as null.

    Older manifests carry no response_digest. Hashing that absence as `None`
    would move every historical root and make every previously-valid run report
    as tampered, which is exactly the wrong alarm.
    """
    before = leaf_hash("0", 1.0, True)
    after_same_call = leaf_hash("0", 1.0, True, None)
    assert before == after_same_call

    with_digest = leaf_hash("0", 1.0, True, response_digest(_Resp(value="x")))
    assert with_digest != before, "a covered run must not collide with an uncovered one"
