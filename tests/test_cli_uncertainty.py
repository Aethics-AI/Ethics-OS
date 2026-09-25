# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-5 Part D — `aethics show` renders uncertainty.

The DoD asks for `0.61 ± 0.04`. Also checks the honest edges: a not_measured
task is never dressed up with an error bar or called under-powered, and a score
with no uncertainty available renders bare rather than inventing one.
"""

import json

from typer.testing import CliRunner

from aethics_eval.cli import app

runner = CliRunner()


def _write(tmp_path, tasks, summary=None):
    payload = {
        "model": "hf:openai-community/gpt2",
        "provenance": {
            "methodology_version": "2.0.0",
            "methodology_fingerprint": "a" * 64,
        },
        "tasks": tasks,
        "summary": summary
        or {"mean_score": 0.5, "tasks_measured": len(tasks), "tasks_run": len(tasks)},
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(payload))
    return p


def test_show_renders_value_plus_minus_stderr(tmp_path):
    p = _write(
        tmp_path,
        {
            "crows_pairs": {
                "score": 0.6134,
                "samples_tested": 30,
                "reliability": "direct",
                "uncertainty": {"standard_error": 0.0412, "underpowered": False},
            }
        },
    )
    out = runner.invoke(app, ["show", str(p)]).stdout
    assert "0.6134 ± 0.041" in out


def test_show_falls_back_to_ci_half_width(tmp_path):
    # No explicit stderr, but the legacy result carries a CI.
    p = _write(
        tmp_path,
        {
            "winobias": {
                "score": 0.8750,
                "samples_tested": 50,
                "reliability": "direct",
                "scored_metric": {"confidence_interval": [0.83, 0.92]},
            }
        },
    )
    out = runner.invoke(app, ["show", str(p)]).stdout
    assert "0.8750 ± 0.045" in out


def test_show_renders_bare_score_when_no_uncertainty(tmp_path):
    p = _write(
        tmp_path,
        {"bold": {"score": 0.5000, "samples_tested": 30, "reliability": "direct"}},
    )
    out = runner.invoke(app, ["show", str(p)]).stdout
    assert "0.5000" in out
    assert "±" not in out  # never invent an error bar


def test_show_flags_underpowered(tmp_path):
    p = _write(
        tmp_path,
        {
            "stereoset": {
                "score": 0.7251,
                "samples_tested": 30,
                "reliability": "direct",
                "uncertainty": {"standard_error": 0.0631, "underpowered": True},
            }
        },
    )
    out = runner.invoke(app, ["show", str(p)]).stdout
    assert "underpowered" in out


def test_show_derives_underpowered_from_min_samples(tmp_path):
    # No explicit flag: winobias needs 50, so 30 samples is under-powered.
    p = _write(
        tmp_path,
        {"winobias": {"score": 0.8, "samples_tested": 30, "reliability": "direct"}},
    )
    out = runner.invoke(app, ["show", str(p)]).stdout
    assert "underpowered" in out


def test_not_measured_is_never_called_underpowered(tmp_path):
    """A task with no measurement has nothing to be under-powered about, and must
    not be shown with an error bar either."""
    p = _write(
        tmp_path,
        {"bold": {"score": None, "samples_tested": 0, "reliability": "not_measured"}},
        summary={"mean_score": None, "tasks_measured": 0, "tasks_run": 1},
    )
    out = runner.invoke(app, ["show", str(p)]).stdout
    assert "not_measured" in out
    assert "underpowered" not in out
    assert "±" not in out
    # and the existing honesty line survives
    assert "not a score of zero" in out
