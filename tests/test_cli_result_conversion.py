# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""Regressions from the review of the CLI-to-RunResult conversion (PR #64).

Every test here corresponds to a defect found reviewing that change, not to a
hypothetical. Grouped by what breaks if they regress.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from aethics_eval.cli import (
    _normalise_evidence,
    _task_details,
    app,
    resolve_tasks,
)
from aethics_eval.manifest import leaf_hash

runner = CliRunner()


# ── verify must not cry tampering on an untouched file ──────────────


def test_evidence_is_normalised_before_it_is_hashed() -> None:
    """The worst defect in the review.

    The manifest hashes per-sample values; the result file stores them. The
    converter coerced with `float(value or 0.0)` *after* the manifest was
    built, so an int 1 became 1.0 and a not-measured None became 0.0 - both of
    which hash differently. `aethics verify` then reported tampering on a file
    nobody had touched, which is the worst thing this tool can do, because the
    alarm is the whole product.
    """
    assert leaf_hash("0", 1, True) != leaf_hash("0", 1.0, True)
    assert leaf_hash("0", None, False) != leaf_hash("0", 0.0, False)

    raw = {
        "t": [
            {"sample_id": "0", "value": 1, "measured": True},
            {"sample_id": "1", "value": None, "measured": False},
        ]
    }
    out = _normalise_evidence(raw)["t"]
    assert out[0]["value"] == 1.0 and isinstance(out[0]["value"], float)
    assert out[1]["value"] == 0.0
    # The not-measured claim survives the placeholder.
    assert out[1]["measured"] is False
    # Normalising twice changes nothing, so hashing and serialising agree.
    assert _normalise_evidence(out and raw) == _normalise_evidence(
        _normalise_evidence(raw)
    )


def test_normalisation_preserves_a_response_digest() -> None:
    """POS-5's digest is part of the hashed leaf and must survive."""
    raw = {
        "t": [
            {"sample_id": "0", "value": 1.0, "measured": True, "response_digest": "abc"}
        ]
    }
    assert _normalise_evidence(raw)["t"][0]["response_digest"] == "abc"


def test_end_to_end_run_verifies(tmp_path) -> None:
    out = tmp_path / "r.json"
    assert (
        runner.invoke(
            app,
            [
                "eval",
                "--model",
                "fake:x",
                "--tasks",
                "bold",
                "--limit",
                "4",
                "-o",
                str(out),
            ],
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["verify", str(out)]).exit_code == 0


# ── nothing silently drops the reason a score is missing ────────────


def test_a_failed_task_keeps_its_error() -> None:
    """_run puts the failure reason in a top-level "error"; the converter only
    copied "details", so the reason never reached the file and `show` printed
    an empty NOTE. The reason a measurement is missing is the most useful thing
    in the file when one is."""
    details = _task_details({"score": None, "error": "HTTPError: 503"})
    assert details and details["error"] == "HTTPError: 503"


def test_existing_details_are_not_clobbered() -> None:
    details = _task_details({"score": 1.0, "details": {"methodology": "x"}})
    assert details["methodology"] == "x"


def test_a_task_with_no_details_and_no_error_reports_none() -> None:
    assert _task_details({"score": 1.0}) is None


# ── the contract's fields are actually populated ────────────────────


def test_method_is_read_from_the_key_tasks_actually_use(tmp_path) -> None:
    """Tasks record their scoring method under "methodology"; the converter
    checked only "method", so it always fell through to reliability."""
    out = tmp_path / "r.json"
    runner.invoke(
        app,
        [
            "eval",
            "--model",
            "fake:x",
            "--tasks",
            "bold",
            "--limit",
            "3",
            "-o",
            str(out),
        ],
    )
    task = json.loads(out.read_text(encoding="utf-8"))["tasks"][0]
    assert task["method"] not in ("direct", "not_measured"), (
        "method fell back to reliability instead of the task's methodology"
    )


def test_created_at_precedes_completed_at(tmp_path) -> None:
    """created_at was stamped when the RunResult was constructed - after the
    run - so every result showed a zero-second duration."""
    out = tmp_path / "r.json"
    runner.invoke(
        app,
        [
            "eval",
            "--model",
            "fake:x",
            "--tasks",
            "bold",
            "--limit",
            "3",
            "-o",
            str(out),
        ],
    )
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["created_at"] < d["completed_at"]


# ── display must survive every shape the contract allows ────────────


def test_show_handles_a_null_summary() -> None:
    """`summary` is optional in the 1.2.0 contract, so a library-produced
    RunResult carries `"summary": null`. `data.get("summary", {})` returns None
    for that, and `aethics show` died with AttributeError."""
    from pathlib import Path

    golden = Path(__file__).resolve().parent / "golden" / "run_result_v1.json"
    result = runner.invoke(app, ["show", str(golden)])
    assert result.exit_code == 0, result.output
    assert "mean" in result.output


# ── an unrelated pip install must not change a default run ──────────


def test_the_default_task_set_is_the_built_ins() -> None:
    """`all` was the default, and `all` now includes third-party tasks, so
    installing an unrelated package silently changed which benchmarks ran and
    what summary.mean_score averaged over."""
    assert sorted(t.task_id for t in resolve_tasks("builtin")) == [
        "bold",
        "crows_pairs",
        "stereoset",
        "winobias",
    ]


def test_all_still_means_all() -> None:
    """The opt-in must remain available, or plugins are unreachable in bulk."""
    names = {t.task_id for t in resolve_tasks("all")}
    assert {"bold", "crows_pairs", "stereoset", "winobias"} <= names
