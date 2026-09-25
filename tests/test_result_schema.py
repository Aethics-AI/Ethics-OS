# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-4 Part A — the versioned result schema models.

Covers the contract: required fields, a top-level schema_version, honest
not_measured (metric_value None, never a default), UTC timestamps, forward-
compatible loading (unknown fields ignored), and a real manifest fingerprint.
"""

import json
import re

from aethics_eval.results import (
    SCHEMA_VERSION,
    DatasetInfo,
    ModelInfo,
    RunResult,
    SampleScore,
    TaskResult,
    Uncertainty,
    current_manifest_fingerprint,
)


def _task(**over):
    base = dict(
        task_id="crows_pairs",
        task_version="1.0.0",
        dataset=DatasetInfo(name="nyu-mll/crows-pairs", revision="master"),
        metric="stereotype_preference_distance",
        metric_value=0.83,
        uncertainty=Uncertainty(
            method="bootstrap_1000", confidence_interval=(0.7, 0.9)
        ),
        sample_count=30,
        method="classifier",
        reliability="direct",
        passed=True,
    )
    base.update(over)
    return TaskResult(**base)


def _run(**over):
    base = dict(
        run_id="run-123",
        model=ModelInfo(
            id="Qwen/Qwen2.5-7B-Instruct", provider="hf", params={"temperature": 0}
        ),
        manifest_fingerprint="a" * 64,
        tasks=[_task()],
    )
    base.update(over)
    return RunResult(**base)


def test_schema_version_is_present_and_current():
    r = _run()
    # Assert the shape, not a literal version: minor bumps are expected as
    # optional fields are added (see docs/result-schema-compatibility.md). The
    # major must stay 1 — that is the promise to existing readers.
    assert r.schema_version == SCHEMA_VERSION
    assert SCHEMA_VERSION.split(".")[0] == "1"
    payload = json.loads(r.to_json())
    assert payload["schema_version"] == SCHEMA_VERSION


def test_required_fields_serialise():
    payload = json.loads(_run().to_json())
    assert payload["run_id"] == "run-123"
    assert payload["model"] == {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "provider": "hf",
        "params": {"temperature": 0},
    }
    assert re.fullmatch(r"[0-9a-f]{64}", payload["manifest_fingerprint"])
    t = payload["tasks"][0]
    for key in (
        "task_id",
        "task_version",
        "dataset",
        "metric",
        "metric_value",
        "uncertainty",
        "sample_count",
        "method",
        "reliability",
    ):
        assert key in t
    assert t["dataset"] == {"name": "nyu-mll/crows-pairs", "revision": "master"}


def test_not_measured_is_null_never_a_default():
    r = _run(
        tasks=[
            _task(
                metric_value=None,
                reliability="not_measured",
                method="not_measured",
                sample_count=0,
            )
        ]
    )
    payload = json.loads(r.to_json())
    assert payload["tasks"][0]["metric_value"] is None


def test_omitted_metric_value_defaults_to_null_not_zero():
    """The default for an unset metric_value must be null, never a number.

    A numeric default (e.g. 0.0) would silently turn "we could not measure this"
    into "this model scored zero" — the exact F1/F6 dishonesty, reintroduced at
    the schema layer. Guards the field default itself, not just an explicit None.
    """
    t = TaskResult(
        task_id="bold",
        task_version="1.0.0",
        dataset=DatasetInfo(name="AlexaAI/bold", revision="main"),
        metric="cross_domain_sentiment_consistency",
        # metric_value deliberately omitted
        method="not_measured",
        reliability="not_measured",
    )
    assert t.metric_value is None, (
        "unset metric_value must default to null, not a number"
    )
    assert json.loads(_run(tasks=[t]).to_json())["tasks"][0]["metric_value"] is None


def test_timestamps_are_utc_iso():
    payload = json.loads(_run().to_json())
    # ISO-8601 with a UTC offset (+00:00 or Z)
    assert re.search(r"(\+00:00|Z)$", payload["created_at"])


def test_unknown_fields_ignored_for_forward_compat():
    # A v1.x result with an added field must still load under the v1 reader.
    raw = json.loads(_run().to_json())
    raw["some_future_field"] = 42
    raw["tasks"][0]["another_future_field"] = "x"
    r = RunResult.model_validate(raw)
    assert r.run_id == "run-123"
    assert not hasattr(r, "some_future_field")


def test_manifest_fingerprint_helper_is_real_hex():
    fp = current_manifest_fingerprint()
    assert re.fullmatch(r"[0-9a-f]{64}", fp)


def test_sample_scores_optional_and_typed():
    r = _run(tasks=[_task(sample_scores=[SampleScore(sample_id="0", value=1.0)])])
    payload = json.loads(r.to_json())
    assert payload["tasks"][0]["sample_scores"][0] == {
        "sample_id": "0",
        "value": 1.0,
        "measured": True,
        # POS-11: present because the manifest hashes it into the evidence
        # leaf. None when the task produced no model response to digest.
        "response_digest": None,
    }
