# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-3 — CLI contract tests.

The CLI is the surface most people will judge this package by, and the
properties that make it usable are easy to break by accident:

* stdout must stay machine-readable, or `aethics eval ... | jq` breaks
* exit codes must distinguish bad usage from a failed run, or CI cannot
  tell "you typed it wrong" from "the model was unreachable"
* `validate` must reject a score with nothing behind it, which is the whole
  reason it exists rather than being a JSON-schema check

Everything here runs offline through the `fake:` provider and Typer's
CliRunner. No network, no model downloads.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from aethics_eval.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, app

runner = CliRunner()


# ── list-tasks ──────────────────────────────────────────────────────


def test_list_tasks_lists_every_benchmark() -> None:
    result = runner.invoke(app, ["list-tasks"])
    assert result.exit_code == EXIT_OK, result.output
    for task in ("winobias", "stereoset", "crows_pairs", "bold"):
        assert task in result.output


def test_list_tasks_json_carries_licence_and_citation() -> None:
    """Machine-readable output is what a task registry consumer would use."""
    result = runner.invoke(app, ["list-tasks", "--json"])
    assert result.exit_code == EXIT_OK, result.output

    rows = json.loads(result.output)

    # A subset check, not an exact count. The CLI now lists tasks from the VOS-3
    # registry, so an installed third-party package legitimately adds rows -
    # that is the feature. Asserting len(rows) == 4 would make `pip install
    # some-tasks` fail our own suite.
    by_name = {r["task"]: r for r in rows}
    for builtin in ("bold", "crows_pairs", "stereoset", "winobias"):
        assert builtin in by_name, f"{builtin} missing from list-tasks"

    for row in rows:
        assert "requires_logprobs" in row

    # Licence and citation are asserted on the datasets we ship, which are the
    # ones POS-2 is accountable for. An out-of-tree task has no entry in our
    # dataset registry and we cannot vouch for its attribution.
    for name in ("bold", "crows_pairs", "stereoset", "winobias"):
        row = by_name[name]
        assert row["license"], f"{name} has no licence"
        assert row["citation"], f"{name} has no citation"


def test_list_tasks_shows_every_dataset_as_pinned() -> None:
    """POS-4 closed the last unpinned load.

    This test previously asserted the opposite — that the CLI *warned* about
    CrowS-Pairs reading from `master`. POS-4 pinned it to a commit, so the
    warning is gone and the assertion is inverted. The warning path still
    exists in the CLI and fires if a dataset ever loses its revision.
    """
    result = runner.invoke(app, ["list-tasks"])
    assert result.exit_code == EXIT_OK, result.output
    assert "unpinned" not in result.output.lower(), (
        "a dataset lost its pinned revision — two runs can now read different "
        "data while reporting identical provenance"
    )
    # Checked against the JSON rather than by slicing the table, which broke as
    # soon as a plugin with a longer name widened the columns.
    #
    # Only rows that *have* a dataset can be pinned: a task with no third-party
    # corpus renders "-", which is a different claim from "NO" and must not be
    # read as an unpinned load.
    rows = json.loads(runner.invoke(app, ["list-tasks", "--json"]).output)
    with_dataset = [r for r in rows if r["dataset"] != "-"]
    assert with_dataset, "no dataset-backed tasks listed at all"
    unpinned = [r["task"] for r in with_dataset if not r["pinned_revision"]]
    assert not unpinned, f"unpinned dataset load(s): {unpinned}"


# ── argument handling: bad usage is exit 2, not a traceback ──────────


@pytest.mark.parametrize(
    "args",
    [
        ["eval", "--model", "no-colon-here", "--tasks", "bold"],
        ["eval", "--model", "hf:gpt2", "--tasks", "not_a_real_task"],
        ["eval", "--model", "nosuchprovider:x", "--tasks", "bold"],
        ["eval", "--model", "hf:", "--tasks", "bold"],
        ["validate", "/nonexistent/path.json"],
        ["show", "/nonexistent/path.json"],
    ],
)
def test_bad_usage_exits_two(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == EXIT_USAGE, (
        f"{args} → {result.exit_code}\n{result.output}"
    )


def test_openai_provider_fails_with_a_useful_message() -> None:
    """Not implemented yet — but it must say so, not fail obscurely."""
    result = runner.invoke(app, ["eval", "--model", "openai:gpt-4o", "--tasks", "bold"])
    assert result.exit_code == EXIT_USAGE
    assert "openai" in result.output.lower()


# ── eval ────────────────────────────────────────────────────────────


def test_eval_writes_valid_json_with_provenance(tmp_path) -> None:
    out = tmp_path / "r.json"
    result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "fake:I cannot help",
            "--tasks",
            "bold",
            "--limit",
            "2",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    assert out.is_file()

    data = json.loads(out.read_text())
    # model is a typed ModelInfo in the VOS-4 result, not a bare spec string.
    assert data["model"]["provider"] == "fake"
    assert data["model"]["id"] == "I cannot help"
    assert [t["task_id"] for t in data["tasks"]] == ["bold"]

    prov = data["provenance"]
    assert prov["methodology_fingerprint"], (
        "a result with no fingerprint is untraceable"
    )
    assert prov["inference_params"]["temperature"] == 0.0, "eval must be deterministic"
    assert "AlexaAI/bold" in prov["datasets"]


def test_eval_stdout_is_parseable_json(tmp_path) -> None:
    """Progress goes to stderr so stdout can be piped. Guard it."""
    result = runner.invoke(
        app, ["eval", "--model", "fake:x", "--tasks", "bold", "--limit", "2"]
    )
    assert result.exit_code == EXIT_OK, result.output
    # CliRunner merges streams by default; parse the JSON object out of the tail.
    start = result.output.index("{")
    json.loads(result.output[start:])


def test_eval_records_the_dataset_revision_it_used(tmp_path) -> None:
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
            "2",
            "--output",
            str(out),
        ],
    )
    ds = json.loads(out.read_text())["provenance"]["datasets"]["AlexaAI/bold"]
    assert ds["pinned"] is True
    assert ds["revision"] == "be5f5a99b386a7c4fa7ea905685ee2d2c98301eb"


# ── validate: the honesty checks ────────────────────────────────────


def _write(tmp_path, payload) -> str:
    p = tmp_path / "r.json"
    p.write_text(json.dumps(payload))
    return str(p)


# The VOS-4 RunResult shape, which `aethics eval` now writes. `validate`
# checks it against the published schemas/result-v1.json and then applies the
# honesty rules a JSON Schema cannot express.
GOOD = {
    "schema_version": "1.2.0",
    "run_id": "test-run-0001",
    "model": {"id": "x", "provider": "fake", "params": {}},
    "manifest_fingerprint": "abc123",
    "created_at": "2026-01-02T03:04:05Z",
    "tasks": [
        {
            "task_id": "bold",
            "task_version": "3.0.0",
            "dataset": {"name": "AlexaAI/bold", "revision": "deadbeef"},
            "metric": "cross_domain_sentiment_consistency",
            "metric_value": 0.5,
            "uncertainty": {"sample_size": 3, "underpowered": True},
            "sample_count": 3,
            "method": "direct",
            "reliability": "direct",
        }
    ],
    "summary": {"tasks_run": 1, "tasks_measured": 1, "mean_score": 0.5},
    "provenance": {
        "methodology_fingerprint": "abc123",
        "inference_params": {},
        "datasets": {},
    },
}


def test_validate_accepts_a_well_formed_result(tmp_path) -> None:
    result = runner.invoke(app, ["validate", _write(tmp_path, GOOD)])
    assert result.exit_code == EXIT_OK, result.output


def test_validate_rejects_a_score_with_no_samples(tmp_path) -> None:
    """The check this command exists for.

    A score of 0.91 from zero samples is the F3/F4 failure mode: a number
    presented as a measurement when nothing was measured. Structural schema
    validation would pass it happily.
    """
    bad = json.loads(json.dumps(GOOD))
    bad["tasks"][0]["metric_value"] = 0.91
    bad["tasks"][0]["sample_count"] = 0

    result = runner.invoke(app, ["validate", _write(tmp_path, bad)])
    assert result.exit_code == EXIT_FAILURE
    assert "not a measurement" in result.output


def test_validate_rejects_mean_score_with_nothing_measured(tmp_path) -> None:
    bad = json.loads(json.dumps(GOOD))
    bad["summary"] = {"tasks_run": 1, "tasks_measured": 0, "mean_score": 0.7}

    result = runner.invoke(app, ["validate", _write(tmp_path, bad)])
    assert result.exit_code == EXIT_FAILURE
    assert "mean_score" in result.output


@pytest.mark.parametrize("missing", ["provenance", "tasks", "model"])
def test_validate_requires_the_envelope(tmp_path, missing: str) -> None:
    bad = json.loads(json.dumps(GOOD))
    del bad[missing]

    result = runner.invoke(app, ["validate", _write(tmp_path, bad)])
    assert result.exit_code == EXIT_FAILURE
    assert missing in result.output


def test_validate_requires_provenance_fields(tmp_path) -> None:
    bad = json.loads(json.dumps(GOOD))
    bad["provenance"] = {}

    result = runner.invoke(app, ["validate", _write(tmp_path, bad)])
    assert result.exit_code == EXIT_FAILURE
    assert "methodology_fingerprint" in result.output


def test_validate_rejects_an_empty_task_set(tmp_path) -> None:
    bad = json.loads(json.dumps(GOOD))
    bad["tasks"] = {}

    result = runner.invoke(app, ["validate", _write(tmp_path, bad)])
    assert result.exit_code == EXIT_FAILURE
    # An empty task set is caught, either as a schema violation or by the
    # honesty layer's own message - both name `tasks` as the problem.
    assert "tasks" in result.output


def test_validate_rejects_malformed_json(tmp_path) -> None:
    p = tmp_path / "r.json"
    p.write_text("{not json")

    result = runner.invoke(app, ["validate", str(p)])
    assert result.exit_code == EXIT_FAILURE
    assert "invalid JSON" in result.output


# ── show ────────────────────────────────────────────────────────────


def test_show_renders_a_result(tmp_path) -> None:
    result = runner.invoke(app, ["show", _write(tmp_path, GOOD)])
    assert result.exit_code == EXIT_OK, result.output
    assert "bold" in result.output
    assert "0.5000" in result.output


def test_show_never_prints_not_measured_as_zero(tmp_path) -> None:
    """A missing measurement is not a score of zero, and must not look like one."""
    nm = json.loads(json.dumps(GOOD))
    nm["tasks"][0] = {
        "task_id": "bold",
        "task_version": "3.0.0",
        "dataset": {"name": "AlexaAI/bold", "revision": "deadbeef"},
        "metric": "cross_domain_sentiment_consistency",
        "metric_value": None,
        "sample_count": 0,
        "method": "not_measured",
        "reliability": "not_measured",
        "details": {"error": "unreachable"},
    }
    nm["summary"] = {"tasks_run": 1, "tasks_measured": 0, "mean_score": None}

    result = runner.invoke(app, ["show", _write(tmp_path, nm)])
    assert result.exit_code == EXIT_OK, result.output
    assert "not_measured" in result.output
    assert "0.0000" not in result.output
    assert "not a score of zero" in result.output


# ── config file ─────────────────────────────────────────────────────


def test_config_file_supplies_settings(tmp_path) -> None:
    cfg = tmp_path / "eval.yaml"
    cfg.write_text('model: "fake:x"\ntasks: bold\nlimit: 2\n')

    result = runner.invoke(app, ["eval", "--config", str(cfg)])
    assert result.exit_code == EXIT_OK, result.output
    assert "fake:x" in result.output


def test_command_line_overrides_config(tmp_path) -> None:
    """A committed config describes the run; a flag overrides it for one go."""
    cfg = tmp_path / "eval.yaml"
    cfg.write_text('model: "fake:x"\ntasks: bold\nlimit: 2\n')

    result = runner.invoke(app, ["eval", "--config", str(cfg), "--limit", "4"])
    assert result.exit_code == EXIT_OK, result.output
    assert "limit=4" in result.output


def test_config_rejects_unknown_keys(tmp_path) -> None:
    """A silently-ignored setting in a checked-in config is worse than an error.

    The run would look configured and not be.
    """
    cfg = tmp_path / "eval.yaml"
    cfg.write_text('model: "fake:x"\nlimt: 3\n')

    result = runner.invoke(app, ["eval", "--config", str(cfg)])
    assert result.exit_code == EXIT_USAGE
    assert "limt" in result.output


def test_config_json_is_accepted(tmp_path) -> None:
    cfg = tmp_path / "eval.json"
    cfg.write_text(json.dumps({"model": "fake:x", "tasks": "bold", "limit": 2}))

    result = runner.invoke(app, ["eval", "--config", str(cfg)])
    assert result.exit_code == EXIT_OK, result.output


def test_missing_config_is_a_usage_error(tmp_path) -> None:
    result = runner.invoke(app, ["eval", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == EXIT_USAGE


def test_model_is_required_from_somewhere(tmp_path) -> None:
    cfg = tmp_path / "eval.yaml"
    cfg.write_text("tasks: bold\n")

    result = runner.invoke(app, ["eval", "--config", str(cfg)])
    assert result.exit_code == EXIT_USAGE
    assert "--model is required" in result.output


# ── openai provider ─────────────────────────────────────────────────


def test_openai_requires_a_base_url() -> None:
    result = runner.invoke(app, ["eval", "--model", "openai:gpt-4o", "--tasks", "bold"])
    assert result.exit_code == EXIT_USAGE
    assert "base-url" in result.output


def test_openai_builds_with_a_base_url() -> None:
    """Constructing the provider must not require a key or a live endpoint."""
    from aethics_eval.cli import build_model

    m = build_model("openai:gpt-4o", base_url="https://example.invalid/v1")
    assert m.capabilities.supports_logprobs is False, (
        "the generic chat endpoint returns no logprobs — claiming otherwise "
        "would make the benchmarks substitute a proxy instead of not_measured"
    )


def test_api_key_is_never_a_command_line_flag() -> None:
    """argv is visible in shell history and to `ps`.

    If a --api-key flag is ever added, this test should fail and the person
    adding it should have to justify it.
    """
    result = runner.invoke(app, ["eval", "--help"])
    assert "--api-key" not in result.output
    assert "--token" not in result.output
