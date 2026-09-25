# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""The quickstart, typed into a real shell against the real pinned datasets.

Nothing is patched: this is the installed ``aethics`` executable, run in a
subprocess, reading WinoBias / StereoSet / CrowS-Pairs / BOLD at their pinned
revisions. The datasets come from the local cache when present, otherwise
they are downloaded once (hence ``network``).

Deselect with: ``pytest -m "not network"``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.network


def aethics(*args: str, env: dict | None = None, timeout: int = 600):
    scripts = Path(sys.executable).parent
    exe = next(
        (
            str(scripts / n)
            for n in ("aethics.exe", "aethics")
            if (scripts / n).exists()
        ),
        None,
    )
    cmd = [exe, *args] if exe else [sys.executable, "-m", "aethics_eval.cli", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})},
    )


@pytest.fixture(scope="module")
def first_run(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("qs") / "first.json"
    r = aethics(
        "eval",
        "--model",
        "fake:I cannot help with that.",
        "--tasks",
        "bold",
        "--limit",
        "5",
        "--output",
        str(out),
    )
    if r.returncode != 0 and "load failed" in r.stderr:
        pytest.skip(f"BOLD could not be fetched: {r.stderr[-300:]}")
    assert r.returncode == 0, r.stderr
    return out


def test_quickstart_list_tasks() -> None:
    r = aethics("list-tasks")
    assert r.returncode == 0, r.stderr
    assert "4 tasks" in r.stdout or "tasks." in r.stdout
    assert "unpinned" not in (r.stdout + r.stderr).lower()


def test_quickstart_first_run_matches_the_docs(first_run) -> None:
    data = json.loads(first_run.read_text(encoding="utf-8"))
    (task,) = data["tasks"]
    # Docs: a model that refuses everything scores 1.0 on BOLD — a correct
    # measurement of a useless model.
    assert task["metric_value"] == 1.0
    assert task["sample_count"] == 5
    assert task["uncertainty"]["underpowered"] is True
    assert data["provenance"]["datasets"]["AlexaAI/bold"]["pinned"] is True


def test_quickstart_show(first_run) -> None:
    r = aethics("show", str(first_run))
    assert r.returncode == 0, r.stderr
    assert "methodology  : 3.0.0" in r.stdout
    assert "fingerprint  : 266b57eaa394ea0b..." in r.stdout
    assert "underpowered" in r.stdout
    assert "(1/1 measured)" in r.stdout


def test_quickstart_validate_and_verify(first_run) -> None:
    v = aethics("validate", str(first_run))
    assert v.returncode == 0, v.stderr
    assert "valid" in v.stdout
    ver = aethics("verify", str(first_run))
    assert ver.returncode == 0, ver.stdout
    assert "VERIFIED" in ver.stdout
    assert "samples checked   : 5" in ver.stdout


def test_quickstart_not_measured_section(tmp_path) -> None:
    out = tmp_path / "nm.json"
    r = aethics(
        "eval",
        "--model",
        "fake:hello",
        "--tasks",
        "crows_pairs",
        "--limit",
        "5",
        "--output",
        str(out),
    )
    assert r.returncode == 1
    assert "not_measured" in r.stderr
    assert (
        json.loads(out.read_text(encoding="utf-8"))["tasks"][0]["metric_value"] is None
    )
    assert "This is not a score of zero." in aethics("show", str(out)).stdout


def test_every_builtin_loads_real_data_and_runs(tmp_path) -> None:
    """All four real datasets parse, and the three logprob tasks report
    not_measured (not 0) for a generation-only model."""
    out = tmp_path / "all.json"
    r = aethics("eval", "--model", "fake:hello", "--limit", "10", "--output", str(out))
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    by_id = {t["task_id"]: t for t in data["tasks"]}
    assert set(by_id) == {"bold", "crows_pairs", "stereoset", "winobias"}
    assert "FAILED" not in r.stderr, "a real dataset failed to load or parse"
    assert by_id["bold"]["metric_value"] is not None
    for t in ("crows_pairs", "stereoset", "winobias"):
        assert by_id[t]["metric_value"] is None
    for name, info in data["provenance"]["datasets"].items():
        assert info.get("revision"), f"{name} has no recorded revision"
    assert aethics("verify", str(out)).returncode == 0


def test_offline_with_an_empty_cache_fails_honestly(tmp_path) -> None:
    out = tmp_path / "off.json"
    r = aethics(
        "eval",
        "--model",
        "fake:x",
        "--tasks",
        "bold",
        "--limit",
        "3",
        "--offline",
        "--output",
        str(out),
        env={"AETHICS_CACHE_DIR": str(tmp_path / "empty-cache")},
    )
    assert r.returncode == 1
    assert "offline" in r.stderr.lower()
    assert (
        json.loads(out.read_text(encoding="utf-8"))["tasks"][0]["metric_value"] is None
    )


def test_manifest_records_the_exact_rows_scored(first_run) -> None:
    data = json.loads(first_run.read_text(encoding="utf-8"))
    (ds,) = data["manifest"]["datasets"]
    assert ds["content_hash"], "no content hash for the dataset rows scored"
    assert ds["row_count"], "no row count for the dataset scored"


def test_cache_lists_populated_datasets(first_run) -> None:
    r = aethics("cache")
    assert r.returncode == 0
    assert "bold" in r.stdout.lower()
