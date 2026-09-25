# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-5 — ``--seed`` and ``aethics verify`` end to end.

``validate`` asks whether a result is well-formed. ``verify`` asks whether it
is the result that was actually produced. The difference is F5: the old audit
hash covered the final scores, so an edited sample with the totals adjusted to
match would sail through.

The test that carries the ticket is
``test_a_consistent_forgery_still_fails`` — it edits a sample, then edits the
score and the summary so the file is internally consistent, and verification
must still reject it.

Runs offline through the ``fake:`` provider.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from aethics_eval.cli import EXIT_FAILURE, EXIT_OK, EXIT_USAGE, app

runner = CliRunner()


def _evidence(data, task="bold"):
    """Per-sample evidence for a task.

    It lives in tasks[].sample_scores now that `aethics eval` writes the VOS-4
    RunResult, rather than in a top-level "evidence" block.
    """
    for t in data["tasks"]:
        if t["task_id"] == task:
            return t["sample_scores"]
    raise AssertionError(f"task {task!r} not in result")


def _run(tmp_path, *extra, seed="42"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "r.json"
    result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "fake:x",
            "--tasks",
            "bold",
            "--limit",
            "6",
            "--seed",
            seed,
            "--output",
            str(out),
            *extra,
        ],
    )
    assert result.exit_code == EXIT_OK, result.output
    return out, json.loads(out.read_text())


def _unmeasurable_run(tmp_path):
    """A run where nothing could be measured.

    crows_pairs needs log-probabilities and the fake provider has none, so every
    task reports null. `eval` exits non-zero precisely because no measurement was
    produced — that is the honest outcome, not a failure to write a result — so
    this cannot go through `_run`, which asserts EXIT_OK.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "unmeasured.json"
    runner.invoke(
        app,
        [
            "eval",
            "--model",
            "fake:x",
            "--tasks",
            "crows_pairs",
            "--limit",
            "4",
            "--seed",
            "42",
            "--output",
            str(out),
        ],
    )
    assert out.is_file(), "eval must still write a result when nothing is measured"
    return out, json.loads(out.read_text())


# ── determinism ─────────────────────────────────────────────────────


def test_same_seed_gives_the_same_run_hash(tmp_path) -> None:
    """The DoD property, through the CLI rather than the library."""
    _, a = _run(tmp_path / "a")
    _, b = _run(tmp_path / "b")
    assert a["manifest"]["run_hash"] == b["manifest"]["run_hash"]


def test_a_different_seed_gives_a_different_run_hash(tmp_path) -> None:
    _, a = _run(tmp_path / "a", seed="42")
    _, b = _run(tmp_path / "b", seed="99")
    assert a["manifest"]["run_hash"] != b["manifest"]["run_hash"]


def test_the_seed_actually_applied_is_recorded(tmp_path) -> None:
    """The result records which RNGs were really seeded, not just the request."""
    _, data = _run(tmp_path, seed="7")
    applied = data["provenance"]["seeds"]
    assert applied["python_random"] == 7
    assert applied["numpy"] == 7, "numpy is a core dependency and must be seeded"


def test_the_hash_covers_the_requested_seed_not_the_applied_set(tmp_path) -> None:
    """Deliberate: the applied set varies with what is installed.

    An earlier version hashed the applied map, and because numpy was seeded
    only "if already imported", two identical runs in one process hashed
    differently — the first before numpy was imported, the second after. The
    hash now covers the requested seed, which cannot drift.
    """
    _, data = _run(tmp_path, seed="7")
    assert data["manifest"]["seeds"] == {"seed": 7}


def test_the_manifest_records_what_it_needs_to_be_reproducible(tmp_path) -> None:
    _, data = _run(tmp_path)
    m = data["manifest"]

    assert m["methodology_fingerprint"], "no fingerprint — untraceable"
    assert m["library_version"], "no library version — cannot reproduce"
    assert m["seeds"], "no seeds — cannot reproduce"
    assert m["datasets"], "no dataset record — cannot tell what was scored"
    assert m["tasks"], "no evidence roots"
    assert m["run_hash"]


def test_evidence_is_written_alongside_the_result(tmp_path) -> None:
    """A manifest with nothing to check against proves nothing."""
    _, data = _run(tmp_path)
    assert _evidence(data), "no per-sample evidence recorded"
    assert len(_evidence(data)) == data["manifest"]["tasks"][0]["sample_count"]


# ── verify ──────────────────────────────────────────────────────────


def test_verify_passes_on_an_untouched_result(tmp_path) -> None:
    path, _ = _run(tmp_path)
    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == EXIT_OK, result.output
    assert "VERIFIED" in result.output


def test_verify_recomputes_rather_than_echoing(tmp_path) -> None:
    """Both hashes must be printed, so a reader can see they were compared."""
    path, data = _run(tmp_path)
    result = runner.invoke(app, ["verify", str(path)])
    assert data["manifest"]["run_hash"] in result.output
    assert "run hash computed" in result.output


def test_verify_fails_on_an_edited_sample(tmp_path) -> None:
    path, data = _run(tmp_path)
    _evidence(data)[0]["value"] = 0.123
    path.write_text(json.dumps(data))

    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == EXIT_FAILURE
    assert "FAILED" in result.output


def test_a_consistent_forgery_still_fails(tmp_path) -> None:
    """The F5 fix, stated as a test.

    Edit a sample, then edit the score and the summary so the file agrees with
    itself. Under the old audit hash — which covered only the totals — this
    would have verified. It must not.
    """
    path, data = _run(tmp_path)

    _evidence(data)[0]["value"] = 0.0
    data["tasks"][0]["metric_value"] = 0.8333
    data["summary"]["mean_score"] = 0.8333
    path.write_text(json.dumps(data))

    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == EXIT_FAILURE
    assert "evidence root mismatch" in result.output


def test_verify_fails_when_only_the_reported_score_is_edited(tmp_path) -> None:
    """The inverse of the forgery above, and the easier edit of the two.

    `test_a_consistent_forgery_still_fails` edits a sample and adjusts the score
    to match. This edits *only* the score: every sample, the evidence root and
    the run hash stay untouched and still agree with each other.

    That verified. The manifest keeps its own copy of the score and the run hash
    covers that copy, so nothing compared it to the number the file actually
    reports — the one `aethics show` prints. A result claiming a score its own
    evidence contradicts is precisely what this command exists to reject.
    """
    path, data = _run(tmp_path)
    original = data["tasks"][0]["metric_value"]
    data["tasks"][0]["metric_value"] = 0.5
    assert original != 0.5, "pick a value the run does not already produce"
    path.write_text(json.dumps(data))

    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == EXIT_FAILURE, result.output
    assert "reported score does not match the evidence" in result.output
    # Named, so the reader knows which number to distrust.
    assert "0.5" in result.output


def test_editing_the_score_in_the_manifest_instead_also_fails(tmp_path) -> None:
    """Editing the hashed copy rather than the displayed one.

    This was already caught, by the run hash rather than by the new comparison.
    Asserted so that a future change to one path cannot silently open the other.
    """
    path, data = _run(tmp_path)
    data["manifest"]["tasks"][0]["metric_value"] = 0.5
    path.write_text(json.dumps(data))

    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == EXIT_FAILURE, result.output


def test_an_untouched_score_still_verifies(tmp_path) -> None:
    """The new check must not reject honest results.

    A float that has been through JSON is compared with a tolerance, so a run
    whose score round-trips still passes.
    """
    path, data = _run(tmp_path)
    path.write_text(json.dumps(data))  # re-serialise, changing nothing

    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == EXIT_OK, result.output
    assert "VERIFIED" in result.output


def test_a_not_measured_task_still_verifies(tmp_path) -> None:
    """``None`` is a score too.

    A task that could not be measured reports null. It must verify like any
    other result, and must not be quietly treated as equal to 0.0 — the whole
    library rests on those being different claims.
    """
    path, data = _unmeasurable_run(tmp_path)
    assert any(t["metric_value"] is None for t in data["tasks"]), (
        "expected an unmeasurable task; the fake provider has no logprobs"
    )

    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == EXIT_OK, result.output
    assert "VERIFIED" in result.output


def test_a_not_measured_task_edited_to_a_score_fails(tmp_path) -> None:
    """Turning "we could not measure this" into a number is the worst case.

    It is the one edit that manufactures a measurement out of nothing, so it
    must fail rather than being treated as a null-vs-zero equivalence.
    """
    path, data = _unmeasurable_run(tmp_path)
    for t in data["tasks"]:
        if t["metric_value"] is None:
            t["metric_value"] = 0.0
            break
    path.write_text(json.dumps(data))

    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == EXIT_FAILURE, result.output
    assert "reported score does not match the evidence" in result.output


def test_verify_fails_when_evidence_is_removed(tmp_path) -> None:
    """Deleting the evidence must not turn into a pass by default."""
    path, data = _run(tmp_path)
    for t in data["tasks"]:
        t.pop("sample_scores", None)
    path.write_text(json.dumps(data))

    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == EXIT_FAILURE
    assert "proves nothing" in result.output


def test_verify_refuses_a_result_with_no_manifest(tmp_path) -> None:
    """An older result is unverifiable, and must say so rather than pass."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": "0.1.0", "tasks": {}}))

    result = runner.invoke(app, ["verify", str(path)])
    assert result.exit_code == EXIT_FAILURE
    assert "no manifest" in result.output


def test_verify_missing_file_is_a_usage_error(tmp_path) -> None:
    result = runner.invoke(app, ["verify", str(tmp_path / "nope.json")])
    assert result.exit_code == EXIT_USAGE
