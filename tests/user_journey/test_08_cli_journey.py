# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""The CLI, walked through the way the quickstart walks a new user through it:
eval → show → validate → verify, then the unhappy paths.

Offline: datasets are the hand-written rows from conftest.py. Where a test
needs a model with log-probabilities (which only ``local:`` provides, and that
needs a 2 GB download), ``build_model`` is swapped for a scripted one so the
CLI's plumbing for those tasks is still exercised end to end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aethics_eval import EVAL_MANIFEST
from aethics_eval import cli as cli_mod
from aethics_eval.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, app
from aethics_eval.results import SCHEMA_VERSION, load_result_file

from .conftest import biased_logprob_model

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, list(args))


def eval_to(path: Path, *args: str):
    return invoke("eval", *args, "--output", str(path))


@pytest.fixture
def logprob_cli(monkeypatch):
    """Make `--model local:scripted` resolve to a scripted log-prob model."""
    real = cli_mod.build_model

    def build(spec, base_url=None, max_context=4096):
        if spec == "local:scripted":
            return biased_logprob_model()
        return real(spec, base_url=base_url, max_context=max_context)

    monkeypatch.setattr(cli_mod, "build_model", build)


# ── Quickstart, step by step ────────────────────────────────────────


def test_list_tasks_table_and_json() -> None:
    r = invoke("list-tasks")
    assert r.exit_code == EXIT_OK
    for name in ("bold", "crows_pairs", "stereoset", "winobias"):
        assert name in r.stdout
    rows = {
        row["task"]: row for row in json.loads(invoke("list-tasks", "--json").stdout)
    }
    assert rows["bold"]["requires_logprobs"] is False
    assert rows["winobias"]["license"] == "MIT"
    assert all(
        rows[n]["pinned_revision"]
        for n in ("bold", "crows_pairs", "stereoset", "winobias")
    )


def test_first_run_show_validate_verify(offline_benchmarks, tmp_path) -> None:
    out = tmp_path / "first.json"
    r = eval_to(
        out,
        "--model",
        "fake:I cannot help with that.",
        "--tasks",
        "bold",
        "--limit",
        "6",
    )
    assert r.exit_code == EXIT_OK, r.output
    assert "run hash:" in r.stderr
    assert out.exists()

    show = invoke("show", str(out))
    assert show.exit_code == EXIT_OK
    assert "fake:I cannot help with that." in show.stdout
    assert EVAL_MANIFEST.fingerprint()[:16] in show.stdout
    assert "underpowered" in show.stdout  # 6 < 30
    assert "(1/1 measured)" in show.stdout

    val = invoke("validate", str(out))
    assert val.exit_code == EXIT_OK, val.output
    assert f"valid (schema_version {SCHEMA_VERSION})" in val.stdout

    ver = invoke("verify", str(out))
    assert ver.exit_code == EXIT_OK, ver.output
    assert "VERIFIED" in ver.stdout
    assert "samples checked   : 6" in ver.stdout


def test_result_file_contents(offline_benchmarks, tmp_path) -> None:
    out = tmp_path / "r.json"
    eval_to(
        out, "--model", "fake:hello", "--tasks", "bold", "--limit", "6", "--seed", "7"
    )
    result = load_result_file(out)  # also validates against the schema

    assert result.model.provider == "fake"
    assert result.model.id == "hello"
    assert result.model.params["temperature"] == 0.0
    assert result.manifest_fingerprint == EVAL_MANIFEST.fingerprint()
    assert result.created_at.tzinfo is not None
    assert result.completed_at >= result.created_at

    (task,) = result.tasks
    assert task.task_id == "bold"
    assert task.sample_count == 6
    assert len(task.sample_scores) == 6
    assert task.uncertainty.underpowered is True
    assert task.dataset.name == "AlexaAI/bold"
    assert task.dataset.revision != "unpinned"

    assert result.summary.tasks_run == 1
    assert result.summary.tasks_measured == 1
    assert result.summary.mean_score == task.metric_value
    assert result.summary.underpowered_tasks == ["bold"]
    assert result.provenance["seeds"]
    assert json.dumps(result.provenance).count("7") >= 1


def test_default_run_is_the_four_builtins(offline_benchmarks, tmp_path) -> None:
    """With fake: only bold can be measured; the other three are null, and the
    mean is taken over bold alone rather than averaging in zeros."""
    out = tmp_path / "all.json"
    r = eval_to(out, "--model", "fake:hello", "--limit", "4")
    assert r.exit_code == EXIT_OK, r.output

    data = json.loads(out.read_text())
    by_id = {t["task_id"]: t for t in data["tasks"]}
    assert set(by_id) == {"bold", "crows_pairs", "stereoset", "winobias"}
    for t in ("crows_pairs", "stereoset", "winobias"):
        assert by_id[t]["metric_value"] is None
        assert by_id[t]["reliability"] == "not_measured"
    assert data["summary"]["tasks_measured"] == 1
    assert data["summary"]["mean_score"] == by_id["bold"]["metric_value"]
    assert "(1/4 measured)" in invoke("show", str(out)).stdout
    assert invoke("verify", str(out)).exit_code == EXIT_OK


def test_logprob_benchmarks_through_the_cli(
    offline_benchmarks, logprob_cli, tmp_path
) -> None:
    out = tmp_path / "lp.json"
    r = eval_to(
        out, "--model", "local:scripted", "--tasks", "crows_pairs,stereoset,winobias"
    )
    assert r.exit_code == EXIT_OK, r.output

    data = json.loads(out.read_text())
    scores = {t["task_id"]: t["metric_value"] for t in data["tasks"]}
    assert scores == {"crows_pairs": 0.5, "stereoset": 0.0, "winobias": 0.0}
    assert data["summary"]["mean_score"] == pytest.approx(0.1667, abs=1e-4)
    for t in data["tasks"]:
        assert t["uncertainty"]["confidence_interval"] is not None
        assert t["uncertainty"]["p_value"] is not None

    assert invoke("validate", str(out)).exit_code == EXIT_OK
    assert invoke("verify", str(out)).exit_code == EXIT_OK
    show = invoke("show", str(out)).stdout
    assert "0.5000 ±" in show


def test_stdout_is_pure_json_when_no_output_file(offline_benchmarks) -> None:
    r = invoke("eval", "--model", "fake:x", "--tasks", "bold", "--limit", "4")
    assert r.exit_code == EXIT_OK
    json.loads(r.stdout)  # progress went to stderr, not here
    assert "running" in r.stderr


def test_evidence_roots_are_reproducible(offline_benchmarks, tmp_path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    eval_to(a, "--model", "fake:x", "--tasks", "bold", "--limit", "6")
    eval_to(b, "--model", "fake:x", "--tasks", "bold", "--limit", "6")
    ma, mb = (json.loads(p.read_text())["manifest"] for p in (a, b))
    assert ma["run_id"] != mb["run_id"]
    assert ma["tasks"] == mb["tasks"]  # same evidence → same per-task roots


# ── When nothing can be measured ────────────────────────────────────


def test_nothing_measured_exits_1_and_says_it_is_not_zero(
    offline_benchmarks, tmp_path
) -> None:
    out = tmp_path / "nm.json"
    r = eval_to(out, "--model", "fake:hello", "--tasks", "crows_pairs", "--limit", "4")
    assert r.exit_code == EXIT_FAILURE
    assert "no task produced a measurement" in r.stderr

    data = json.loads(out.read_text())
    assert data["tasks"][0]["metric_value"] is None
    assert data["summary"]["mean_score"] is None

    show = invoke("show", str(out)).stdout
    assert "not_measured" in show
    assert "This is not a score of zero." in show


# ── Tampering ───────────────────────────────────────────────────────


@pytest.fixture
def good_result(offline_benchmarks, tmp_path) -> Path:
    out = tmp_path / "good.json"
    assert (
        eval_to(out, "--model", "fake:x", "--tasks", "bold", "--limit", "6").exit_code
        == 0
    )
    return out


def _edit(path: Path, fn) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def _set_sample(d):
    d["tasks"][0]["sample_scores"][0]["value"] = -1.0


def _set_score(d):
    d["tasks"][0]["metric_value"] = 0.123


def _set_both(d):
    # Edit a sample AND fix up the totals to match — the F5 scenario.
    d["tasks"][0]["sample_scores"][0]["value"] = -1.0
    d["tasks"][0]["metric_value"] = 0.0
    d["summary"]["mean_score"] = 0.0


def _drop_sample(d):
    d["tasks"][0]["sample_scores"].pop()


@pytest.mark.parametrize("tamper", [_set_sample, _set_score, _set_both, _drop_sample])
def test_verify_catches_tampering(good_result, tamper) -> None:
    _edit(good_result, tamper)
    r = invoke("verify", str(good_result))
    assert r.exit_code == EXIT_FAILURE
    assert "FAILED" in r.stdout


def test_verify_refuses_a_result_without_a_manifest(good_result) -> None:
    _edit(good_result, lambda d: d.pop("manifest"))
    r = invoke("verify", str(good_result))
    assert r.exit_code == EXIT_FAILURE
    assert "no manifest" in r.stderr


def test_verify_refuses_a_result_without_evidence(good_result) -> None:
    _edit(good_result, lambda d: d["tasks"][0].update(sample_scores=None))
    assert invoke("verify", str(good_result)).exit_code == EXIT_FAILURE


def test_validate_rejects_a_score_with_no_samples(good_result) -> None:
    _edit(good_result, lambda d: d["tasks"][0].update(sample_count=0))
    r = invoke("validate", str(good_result))
    assert r.exit_code == EXIT_FAILURE
    assert "not a measurement" in r.stderr


def test_validate_rejects_schema_violations(good_result) -> None:
    _edit(good_result, lambda d: d.pop("run_id"))
    r = invoke("validate", str(good_result))
    assert r.exit_code == EXIT_FAILURE
    assert "result-v1.json" in r.stderr


@pytest.mark.parametrize("cmd", ["show", "validate", "verify"])
def test_commands_on_missing_and_broken_files(cmd, tmp_path) -> None:
    assert invoke(cmd, str(tmp_path / "nope.json")).exit_code == EXIT_USAGE
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert invoke(cmd, str(broken)).exit_code == EXIT_FAILURE


# ── Config files ────────────────────────────────────────────────────


def test_json_config(offline_benchmarks, tmp_path) -> None:
    out = tmp_path / "cfg-out.json"
    cfg = tmp_path / "run.json"
    cfg.write_text(
        json.dumps(
            {
                "model": "fake:x",
                "tasks": "bold",
                "limit": 4,
                "output": str(out),
                "seed": 5,
            }
        )
    )
    r = invoke("eval", "--config", str(cfg))
    assert r.exit_code == EXIT_OK, r.output
    assert json.loads(out.read_text())["tasks"][0]["sample_count"] == 4


def test_yaml_config_and_flag_override(offline_benchmarks, tmp_path) -> None:
    out = tmp_path / "y.json"
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model: fake:x\ntasks: bold\nlimit: 4\noutput: {out.as_posix()}\n")
    r = invoke("eval", "--config", str(cfg), "--limit", "6")
    assert r.exit_code == EXIT_OK, r.output
    assert json.loads(out.read_text())["tasks"][0]["sample_count"] == 6


def test_config_with_a_typo_is_refused(tmp_path) -> None:
    cfg = tmp_path / "run.json"
    cfg.write_text(json.dumps({"model": "fake:x", "limt": 4}))
    r = invoke("eval", "--config", str(cfg))
    assert r.exit_code == EXIT_USAGE
    assert "limt" in r.stderr


# ── Usage errors are exit 2, not a traceback ────────────────────────


@pytest.mark.parametrize(
    "args",
    [
        ["eval"],  # no model
        ["eval", "--model", "fake:x", "--tasks", "hellaswag"],
        ["eval", "--model", "nocolon", "--tasks", "bold"],
        ["eval", "--model", "gpt:4", "--tasks", "bold"],
        ["eval", "--model", "openai:gpt-4o", "--tasks", "bold"],  # no --base-url
        ["eval", "--model", "fake:x", "--limit", "0"],
        ["eval", "--model", "fake:x", "--api-key", "sk-123"],  # deliberately absent
    ],
)
def test_usage_errors(args) -> None:
    r = invoke(*args)
    assert r.exit_code == EXIT_USAGE, r.output
    assert r.exception is None or isinstance(r.exception, SystemExit)


# ── Dataset cache command ───────────────────────────────────────────


def test_cache_on_an_empty_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AETHICS_CACHE_DIR", str(tmp_path / "empty"))
    r = invoke("cache")
    assert r.exit_code == EXIT_OK
    assert "empty" in r.stdout
    assert "removed 0" in invoke("cache", "--clear").stdout
