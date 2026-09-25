# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""VOS-4 - what `aethics eval` writes must be the contract we published.

The CLI used to write its own envelope stamped `schema_version 0.1.0`. The
published contract, `schemas/result-v1.json`, was generated from `RunResult`
and had no producer a user could invoke:

    $ aethics eval ... -o r.json
    $ jsonschema -i r.json schemas/result-v1.json
    'run_id' is a required property

    $ aethics validate r.json
    r.json: valid (schema_version 0.1.0)

Two documents, one of them documented and unreachable, and a `validate` command
answering a question nobody asked in the voice of the one they did.

These tests are the standing guard: the file the CLI writes must satisfy the
file we publish.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aethics_eval.cli import app
from aethics_eval.results.schema import SCHEMA_VERSION

jsonschema = pytest.importorskip("jsonschema")

runner = CliRunner()
SDK_ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = SDK_ROOT / "schemas" / "result-v1.json"


@pytest.fixture(scope="module")
def result(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("cli") / "r.json"
    res = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "fake:x",
            "--tasks",
            "bold",
            "--limit",
            "4",
            "--output",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    return json.loads(out.read_text(encoding="utf-8"))


def test_cli_output_satisfies_the_published_schema(result) -> None:
    schema = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)


def test_cli_stamps_the_published_schema_version(result) -> None:
    assert result["schema_version"] == SCHEMA_VERSION, (
        "the CLI must stamp the version of the contract it emits, not a "
        "private envelope version"
    )


def test_uncertainty_is_serialised_not_merely_displayed(result) -> None:
    """VOS-5's DoD: no metric serialises without an uncertainty field.

    `aethics show` rendered "± " and "underpowered" by recomputing them at
    display time, so neither reached the file. A flag that exists only while
    the table is on screen cannot be read by anything downstream.
    """
    task = result["tasks"][0]
    unc = task.get("uncertainty")
    assert unc, "a measured task serialised with no uncertainty"
    assert "underpowered" in unc, "the power flag is not in the result"
    assert unc["sample_size"] == task["sample_count"]

    # Four samples is well under any threshold, so this run must say so.
    assert unc["underpowered"] is True
    assert result["summary"]["underpowered_tasks"] == ["bold"]


def test_per_sample_evidence_travels_with_the_task(result) -> None:
    scores = result["tasks"][0]["sample_scores"]
    assert scores and len(scores) == 4
    assert {"sample_id", "value", "measured"} <= set(scores[0])


def test_validate_accepts_what_eval_writes(tmp_path, result) -> None:
    """The pair has to agree. It is the pair that was broken."""
    p = tmp_path / "r.json"
    p.write_text(json.dumps(result), encoding="utf-8")
    out = runner.invoke(app, ["validate", str(p)])
    assert out.exit_code == 0, out.output


def test_a_task_without_a_scored_metric_is_not_called_not_measured(tmp_path) -> None:
    """A registered task that produced a metric measured something.

    Plugins need not emit the legacy ScoredMetric, so there is no reliability to
    copy. Defaulting to "not_measured" mislabelled a real measurement - and a
    consumer filtering on that flag would have dropped it.
    """
    from aethics_eval.cli import _default_reliability

    assert _default_reliability({"score": 0.0}) == "direct"
    assert _default_reliability({"score": 0.83}) == "direct"
    assert _default_reliability({"score": None}) == "not_measured"
