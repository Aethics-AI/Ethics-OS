# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-4 Part C — golden-file round trip.

Pins the serialisation: a fixed canonical RunResult must serialise to exactly the
committed golden JSON (regenerate with the command in _write_golden if the schema
intentionally changes). And the documented loader must read that golden back into
typed objects — the DoD's "reads a v1 result back into typed objects".
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aethics_eval.results import (
    SCHEMA_VERSION,
    DatasetInfo,
    IncompatibleSchemaVersion,
    ModelInfo,
    RunResult,
    SampleScore,
    TaskResult,
    Uncertainty,
    load_result,
    load_result_file,
)

_GOLDEN = Path(__file__).resolve().parent / "golden" / "run_result_v1.json"

_CREATED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_COMPLETED = datetime(2026, 1, 2, 3, 9, 10, tzinfo=timezone.utc)


def canonical_run() -> RunResult:
    """A fixed, deterministic run covering measured + not_measured tasks,
    populated + default uncertainty, and present + absent sample_scores."""
    return RunResult(
        run_id="golden-run-0001",
        model=ModelInfo(
            id="Qwen/Qwen2.5-7B-Instruct",
            provider="hf",
            params={"temperature": 0, "max_new_tokens": 100},
        ),
        manifest_fingerprint="0" * 64,
        created_at=_CREATED,
        completed_at=_COMPLETED,
        tasks=[
            TaskResult(
                task_id="crows_pairs",
                task_version="1.0.0",
                dataset=DatasetInfo(name="nyu-mll/crows-pairs", revision="master"),
                metric="stereotype_preference_distance",
                metric_value=0.83,
                uncertainty=Uncertainty(
                    method="bootstrap_1000",
                    confidence_interval=(0.71, 0.91),
                    effect_size=0.42,
                    p_value=0.03,
                ),
                sample_count=30,
                method="classifier",
                reliability="direct",
                passed=True,
                sample_scores=[
                    SampleScore(sample_id="0", value=1.0),
                    SampleScore(sample_id="1", value=0.0),
                ],
            ),
            TaskResult(
                task_id="bold",
                task_version="1.0.0",
                dataset=DatasetInfo(name="AlexaAI/bold", revision="main"),
                metric="cross_domain_sentiment_consistency",
                metric_value=None,  # not_measured — stays null
                sample_count=0,
                method="not_measured",
                reliability="not_measured",
                passed=False,
            ),
        ],
    )


def test_serialisation_matches_golden():
    assert _GOLDEN.exists(), "golden file missing — generate it (see module docstring)"
    assert canonical_run().to_json() == _GOLDEN.read_text(), (
        "serialisation drifted from the committed golden file"
    )


def test_loader_round_trips_golden_into_typed_objects():
    run = load_result_file(_GOLDEN)
    assert isinstance(run, RunResult)
    # Track the current version rather than a literal: additive (minor) bumps are
    # expected, and the golden is regenerated with them. A *major* bump is what
    # must break a reader, and test_loader_rejects_newer_major covers that.
    assert run.schema_version == SCHEMA_VERSION
    assert run.run_id == "golden-run-0001"
    assert run.model.provider == "hf"

    crows, bold = run.tasks
    assert crows.metric_value == 0.83
    assert crows.uncertainty.confidence_interval == (0.71, 0.91)
    assert crows.sample_scores[0].value == 1.0
    # not_measured survives the round trip as null, never a default
    assert bold.metric_value is None
    assert bold.reliability == "not_measured"


def test_loader_rejects_newer_major():
    bad = json.loads(_GOLDEN.read_text())
    bad["schema_version"] = "2.0.0"
    with pytest.raises(IncompatibleSchemaVersion):
        load_result(bad)


def test_loader_accepts_newer_minor_with_extra_fields():
    fwd = json.loads(_GOLDEN.read_text())
    fwd["schema_version"] = "1.7.0"
    fwd["a_future_field"] = True
    run = load_result(fwd)  # minor-newer + extra fields -> still readable
    assert run.run_id == "golden-run-0001"
