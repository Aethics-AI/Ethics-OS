# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-4 Part B — the published JSON Schema is validated in CI.

- drift guard: the committed schemas/result-v1.json matches the RunResult model
  (regenerate with `python -m scripts.export_schema` if this fails).
- a valid result validates against the schema; an invalid one is rejected;
  extra fields are tolerated (additive/minor forward compatibility).
"""

import json
from pathlib import Path

import jsonschema
import pytest

from aethics_eval.results import (
    DatasetInfo,
    ModelInfo,
    RunResult,
    TaskResult,
    schema_json,
    validate_result,
)

_PUBLISHED = Path(__file__).resolve().parent.parent / "schemas" / "result-v1.json"


def _valid_run():
    return RunResult(
        run_id="run-1",
        model=ModelInfo(id="Qwen/Qwen2.5-7B-Instruct", provider="hf"),
        manifest_fingerprint="a" * 64,
        tasks=[
            TaskResult(
                task_id="crows_pairs",
                task_version="1.0.0",
                dataset=DatasetInfo(name="nyu-mll/crows-pairs", revision="master"),
                metric="stereotype_preference_distance",
                metric_value=0.83,
                sample_count=30,
                method="classifier",
                reliability="direct",
            )
        ],
    )


def test_published_schema_matches_model_no_drift():
    assert _PUBLISHED.exists(), "schemas/result-v1.json is missing"
    committed = _PUBLISHED.read_text()
    assert committed == schema_json(), (
        "schemas/result-v1.json is out of sync with RunResult — "
        "run `python -m scripts.export_schema`"
    )


def test_valid_result_passes_validation():
    validate_result(_valid_run().to_json())  # raises if invalid


def test_missing_required_field_is_rejected():
    bad = json.loads(_valid_run().to_json())
    del bad["run_id"]  # required
    with pytest.raises(jsonschema.ValidationError):
        validate_result(bad)


def test_wrong_type_is_rejected():
    bad = json.loads(_valid_run().to_json())
    bad["tasks"][0]["sample_count"] = "thirty"  # must be integer
    with pytest.raises(jsonschema.ValidationError):
        validate_result(bad)


def test_extra_fields_tolerated():
    ok = json.loads(_valid_run().to_json())
    ok["future_field"] = 1
    ok["tasks"][0]["future_task_field"] = "x"
    validate_result(ok)  # additive/minor forward compat — must not raise
