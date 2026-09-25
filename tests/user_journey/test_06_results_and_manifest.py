# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Results you can load, validate and prove untouched — from Python."""

from __future__ import annotations

import copy
import json

import jsonschema
import pytest

from aethics_eval.manifest import (
    DatasetRecord,
    build_manifest,
    inclusion_proof,
    leaf_hash,
    merkle_root,
    verify_inclusion,
    verify_manifest,
)
from aethics_eval.results import (
    SCHEMA_VERSION,
    DatasetInfo,
    IncompatibleSchemaVersion,
    ModelInfo,
    RunResult,
    TaskResult,
    build_uncertainty,
    current_manifest_fingerprint,
    format_with_uncertainty,
    load_result,
    load_result_file,
    result_schema,
    validate_result,
)

# ── RunResult round trip ────────────────────────────────────────────


def _result() -> RunResult:
    return RunResult(
        run_id="r1",
        model=ModelInfo(id="gpt2", provider="local", params={"temperature": 0.0}),
        manifest_fingerprint=current_manifest_fingerprint(),
        tasks=[
            TaskResult(
                task_id="crows_pairs",
                task_version="3.0.0",
                dataset=DatasetInfo(name="nyu-mll/crows-pairs", revision="abc"),
                metric="stereotype_preference_distance",
                metric_value=0.5,
                uncertainty=build_uncertainty(
                    [1.0, 0.0, 1.0, 1.0], task_name="crows_pairs"
                ),
                sample_count=4,
                method="sequence_logprob",
                reliability="direct",
            ),
            TaskResult(
                task_id="winobias",
                task_version="3.0.0",
                dataset=DatasetInfo(name="uclanlp/wino_bias", revision="def"),
                metric="coref_accuracy_gap",
                metric_value=None,
                method="not_measured",
                reliability="not_measured",
            ),
        ],
    )


def test_run_result_round_trips_through_json(tmp_path) -> None:
    original = _result()
    text = original.to_json()
    validate_result(text)

    loaded = load_result(text)
    assert loaded == original
    assert loaded.schema_version == SCHEMA_VERSION

    p = tmp_path / "r.json"
    p.write_text(text)
    assert load_result_file(p) == original


def test_unmeasured_serialises_as_null() -> None:
    data = json.loads(_result().to_json())
    wino = next(t for t in data["tasks"] if t["task_id"] == "winobias")
    assert wino["metric_value"] is None


def test_loader_version_policy() -> None:
    data = json.loads(_result().to_json())
    load_result({**data, "schema_version": "1.99.0"})  # newer minor: fine
    with pytest.raises(IncompatibleSchemaVersion):
        load_result({**data, "schema_version": "2.0.0"})
    with pytest.raises(IncompatibleSchemaVersion):
        load_result({**data, "schema_version": "garbage"})


def test_loader_tolerates_unknown_fields() -> None:
    data = json.loads(_result().to_json())
    data["something_new"] = {"x": 1}
    assert load_result(data).run_id == "r1"


def test_validation_rejects_malformed_results() -> None:
    data = json.loads(_result().to_json())
    del data["run_id"]
    with pytest.raises(jsonschema.ValidationError):
        validate_result(data)
    with pytest.raises(jsonschema.ValidationError):
        load_result(data)


def test_published_schema_file_matches_the_code() -> None:
    from pathlib import Path

    committed = Path(__file__).parents[2] / "schemas" / "result-v1.json"
    if not committed.exists():
        pytest.skip("not running from a source checkout")
    assert json.loads(committed.read_text(encoding="utf-8")) == result_schema()


def test_format_with_uncertainty() -> None:
    u = build_uncertainty([1.0, 0.0] * 5, task_name="crows_pairs")
    assert format_with_uncertainty(0.6, u) == "0.6 ± 0.17 (underpowered)"
    assert format_with_uncertainty(None, u) == "not measured"
    assert format_with_uncertainty(0.5773, None) == "0.5773"


# ── Manifest: tamper evidence ───────────────────────────────────────


SAMPLES = {
    "crows_pairs": [
        {"sample_id": str(i), "value": v, "measured": True}
        for i, v in enumerate([1.0, 1.0, 0.0, 1.0])
    ],
    "winobias": [{"sample_id": "0", "value": 0.0, "measured": False}],
}
METRICS = {"crows_pairs": 0.5, "winobias": None}


def _manifest():
    return build_manifest(
        run_id="run-1",
        methodology_fingerprint=current_manifest_fingerprint(),
        library_version="0.1.0",
        seeds={"python": 42},
        model={"spec": "fake:x"},
        datasets=[DatasetRecord(name="nyu-mll/crows-pairs", revision="abc")],
        task_samples=SAMPLES,
        task_metrics=METRICS,
    ).to_dict()


def test_untouched_manifest_verifies() -> None:
    report = verify_manifest(_manifest(), SAMPLES, METRICS)
    assert report.ok, report.problems
    assert report.tasks_checked == 2
    assert report.samples_checked == 5
    assert report.run_hash_expected == report.run_hash_computed


def test_manifest_is_deterministic() -> None:
    assert _manifest()["run_hash"] == _manifest()["run_hash"]


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(
            lambda s, m: s["crows_pairs"][2].update(value=1.0), id="edit-sample"
        ),
        pytest.param(
            lambda s, m: s["winobias"][0].update(measured=True), id="flip-measured"
        ),
        pytest.param(lambda s, m: s["crows_pairs"].pop(), id="drop-sample"),
        pytest.param(
            lambda s, m: s["crows_pairs"].append(
                {"sample_id": "9", "value": 1.0, "measured": True}
            ),
            id="add-sample",
        ),
        pytest.param(lambda s, m: m.update(crows_pairs=0.99), id="edit-reported-score"),
        pytest.param(lambda s, m: m.update(winobias=0.0), id="null-to-zero"),
    ],
)
def test_tampering_is_detected(tamper) -> None:
    manifest = _manifest()
    samples, metrics = copy.deepcopy(SAMPLES), dict(METRICS)
    tamper(samples, metrics)
    report = verify_manifest(manifest, samples, metrics)
    assert not report.ok
    assert report.problems


def test_editing_the_manifest_itself_is_detected() -> None:
    manifest = _manifest()
    manifest["run_hash"] = "0" * 64
    assert not verify_manifest(manifest, SAMPLES, METRICS).ok


def test_inclusion_proof_for_a_single_sample() -> None:
    """Share one sample's evidence without revealing the rest."""
    leaves = [leaf_hash(str(i), float(i % 2)) for i in range(7)]
    root = merkle_root(leaves)
    for i in range(7):
        assert verify_inclusion(leaves[i], inclusion_proof(leaves, i), root)
    forged = leaf_hash("3", 0.5)
    assert not verify_inclusion(forged, inclusion_proof(leaves, 3), root)
