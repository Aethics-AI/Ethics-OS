# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-5 — run manifest: determinism and tamper evidence.

F5 was that ``audit_hash`` covered only the final scores. You could confirm the
totals had not been edited and nothing else — the samples those totals came
from sat outside the hash. "The score is 0.61" was checkable; "the score is
0.61 because of *these* samples" was not.

Two properties carry the whole ticket, and both are tested by trying to break
them rather than by reading the code:

* same seed, same pinned inputs → same run hash
* edit any one sample → the run hash changes

The second is the one that matters. A tampered result must not be repairable
by also editing the totals to agree, because the totals are not what is hashed.
"""

from __future__ import annotations

import copy

import pytest

from aethics_eval.manifest import (
    MANIFEST_VERSION,
    DatasetRecord,
    build_manifest,
    inclusion_proof,
    leaf_hash,
    merkle_root,
    verify_inclusion,
    verify_manifest,
)

SAMPLES = {
    "bold": [
        {"sample_id": f"s{i}", "value": 0.5 + i / 100, "measured": True}
        for i in range(7)
    ],
    "winobias": [{"sample_id": "w1", "value": 0.9, "measured": True}],
}

INPUTS = dict(
    run_id="run-1",
    methodology_fingerprint="266b57eaa394ea0b",
    library_version="0.1.0",
    seeds={"global": 42},
    model={"id": "gpt2", "provider": "hf", "params": {"temperature": 0.0}},
    datasets=[DatasetRecord("AlexaAI/bold", "be5f5a99", "0ddee997", 7201)],
)


def _manifest(samples=None, **over):
    return build_manifest(task_samples=samples or SAMPLES, **{**INPUTS, **over})


# ── determinism ─────────────────────────────────────────────────────


def test_same_inputs_produce_the_same_run_hash() -> None:
    """The DoD property. If this drifts, reproducibility is a claim not a fact."""
    assert _manifest().run_hash == _manifest().run_hash


def test_run_id_does_not_affect_the_run_hash() -> None:
    """Two runs of the same thing must hash the same.

    A run id or timestamp in the hash would make every run unique and the hash
    would prove nothing about reproducibility.
    """
    assert _manifest(run_id="a").run_hash == _manifest(run_id="b").run_hash


def test_sample_order_does_not_affect_the_hash() -> None:
    """Tasks may run concurrently; completion order must not move the hash."""
    shuffled = {k: list(reversed(v)) for k, v in SAMPLES.items()}
    assert _manifest().run_hash == _manifest(shuffled).run_hash


def test_a_different_seed_changes_the_hash() -> None:
    assert _manifest().run_hash != _manifest(seeds={"global": 43}).run_hash


def test_a_different_dataset_revision_changes_the_hash() -> None:
    """Scoring different data must not be able to report the same hash."""
    other = [DatasetRecord("AlexaAI/bold", "deadbeef", "0ddee997", 7201)]
    assert _manifest().run_hash != _manifest(datasets=other).run_hash


def test_different_model_params_change_the_hash() -> None:
    hot = {"id": "gpt2", "provider": "hf", "params": {"temperature": 0.7}}
    assert _manifest().run_hash != _manifest(model=hot).run_hash


def test_manifest_carries_its_version() -> None:
    assert _manifest().manifest_version == MANIFEST_VERSION


# ── verification: the clean case ────────────────────────────────────


def test_an_untouched_result_verifies() -> None:
    report = verify_manifest(_manifest().to_dict(), SAMPLES)
    assert report.ok, report.problems
    assert report.samples_checked == 8
    assert report.tasks_checked == 2


def test_verification_recomputes_rather_than_trusting() -> None:
    """The recorded hash must be re-derived, not read back."""
    report = verify_manifest(_manifest().to_dict(), SAMPLES)
    assert report.run_hash_computed == report.run_hash_expected
    assert report.run_hash_computed  # not an empty-string comparison


# ── verification: tampering ─────────────────────────────────────────


def _tampered(mutate):
    manifest = _manifest().to_dict()
    samples = copy.deepcopy(SAMPLES)
    mutate(samples)
    return verify_manifest(manifest, samples)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        (
            "edit a value",
            lambda s: s["bold"].__setitem__(0, {**s["bold"][0], "value": 0.99}),
        ),
        (
            "flip measured to False",
            lambda s: s["bold"].__setitem__(1, {**s["bold"][1], "measured": False}),
        ),
        ("delete a sample", lambda s: s["bold"].pop()),
        (
            "append a sample",
            lambda s: s["bold"].append(
                {"sample_id": "x", "value": 1.0, "measured": True}
            ),
        ),
        (
            "rename a sample",
            lambda s: s["bold"].__setitem__(2, {**s["bold"][2], "sample_id": "zz"}),
        ),
        ("remove a whole task", lambda s: s.pop("winobias")),
        (
            "add an unrecorded task",
            lambda s: s.update({"ghost": [{"sample_id": "g", "value": 1.0}]}),
        ),
    ],
)
def test_tampering_is_detected(name: str, mutate) -> None:
    report = _tampered(mutate)
    assert not report.ok, f"{name} went undetected — the manifest proves nothing"
    assert report.problems


def test_a_skipped_sample_and_a_zero_score_hash_differently() -> None:
    """`measured=False` and `value=0.0` are different claims.

    Collapsing them would let "we could not measure this" be presented as "we
    measured it and got nothing" without changing the hash.
    """
    assert leaf_hash("s", 0.0, measured=True) != leaf_hash("s", 0.0, measured=False)


def test_editing_the_metric_cannot_repair_a_tampered_result() -> None:
    """The point of hashing evidence rather than totals.

    Someone who edits a sample cannot make verification pass by also editing
    the reported metric to match — the metric is not what is hashed.
    """
    manifest = _manifest().to_dict()
    for task in manifest["tasks"]:
        task["metric_value"] = 0.42  # "fix" the totals

    samples = copy.deepcopy(SAMPLES)
    samples["bold"][0] = {**samples["bold"][0], "value": 0.99}

    assert not verify_manifest(manifest, samples).ok


# ── Merkle tree ─────────────────────────────────────────────────────


def test_empty_tree_has_a_hash_of_its_own() -> None:
    """ "No samples" is a statable fact, not an absence."""
    assert merkle_root([]) not in ("", None)


def test_root_changes_with_any_leaf() -> None:
    a = [leaf_hash(f"s{i}", i) for i in range(5)]
    b = list(a)
    b[2] = leaf_hash("s2", 999)
    assert merkle_root(a) != merkle_root(b)


def test_odd_nodes_are_promoted_not_duplicated() -> None:
    """Duplicating an odd node lets two different leaf sets share a root.

    That is CVE-2012-2459 in Bitcoin. With promotion, appending a duplicate of
    the last leaf must change the root.
    """
    leaves = [leaf_hash(f"s{i}", i) for i in range(3)]
    assert merkle_root(leaves) != merkle_root([*leaves, leaves[-1]])


def test_leaf_and_node_hashes_live_in_different_spaces() -> None:
    """Domain separation: an internal node must not be presentable as a leaf."""
    from aethics_eval.manifest import _pair

    a, b = leaf_hash("a", 1), leaf_hash("b", 2)
    assert _pair(a, b) != leaf_hash(a, b)


@pytest.mark.parametrize("size", [1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32])
def test_inclusion_proof_verifies_for_every_index(size: int) -> None:
    """Proves one sample belongs to a run without disclosing the others.

    That matters here specifically: three of our four datasets are ShareAlike
    and cannot be redistributed, so evidencing a score must not require
    shipping the corpus.
    """
    leaves = [leaf_hash(f"s{i}", i / 10) for i in range(size)]
    root = merkle_root(leaves)
    for i in range(size):
        assert verify_inclusion(leaves[i], inclusion_proof(leaves, i), root), (
            f"index {i} of {size} failed"
        )


def test_inclusion_proof_rejects_a_substituted_leaf() -> None:
    """Guards against the proof being vacuously true."""
    leaves = [leaf_hash(f"s{i}", i / 10) for i in range(9)]
    root = merkle_root(leaves)
    proof = inclusion_proof(leaves, 3)

    assert not verify_inclusion(leaf_hash("s3", 99.0), proof, root)
    assert not verify_inclusion(leaves[3], proof, "0" * 64)
    assert not verify_inclusion(leaves[3], inclusion_proof(leaves, 7), root)
    assert not verify_inclusion(leaves[3], [], root)
    assert verify_inclusion(leaves[3], proof, root)  # positive control


def test_inclusion_proof_rejects_an_out_of_range_index() -> None:
    with pytest.raises(IndexError):
        inclusion_proof([leaf_hash("a", 1)], 5)
