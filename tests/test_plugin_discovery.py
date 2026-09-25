# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-3 Part D — out-of-tree task discovery via entry points.

Simulates a third party's `pip install my-tasks`: an entry point in the
`aethics_eval.tasks` group is injected, and the real `_load_plugins` path loads
it, registers the task, and runs it — with no changes to aethics_eval's source.
The example package lives in ../examples/example-task.
"""

import asyncio
import importlib
import importlib.metadata
from pathlib import Path

import aethics_eval.tasks.registry as registry
from aethics_eval.models import FakeModel
from aethics_eval.tasks import get_task, list_tasks, run_task


def _run(coro):
    return asyncio.run(coro)


_EXAMPLE_SRC = Path(__file__).parent.parent / "examples" / "example-task" / "src"


class _ExampleEntryPoint:
    name = "example_yes_rate"
    group = "aethics_eval.tasks"

    def load(self):
        # importing the module runs its @register_task decorator
        return importlib.import_module("aethics_eval_example")


def test_out_of_tree_task_discovered_and_runs(monkeypatch):
    # Make the example importable (as if pip-installed) ...
    monkeypatch.syspath_prepend(str(_EXAMPLE_SRC))

    # ... and make it discoverable via the entry-point group.
    def fake_entry_points(group=None):
        return [_ExampleEntryPoint()] if group == "aethics_eval.tasks" else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    # Re-arm the one-shot plugin loader so discovery runs again this test.
    monkeypatch.setattr(registry, "_plugins_loaded", False)

    # Discovery: the external task shows up alongside the built-ins.
    names = {m.name for m in list_tasks()}
    assert "example_yes_rate" in names, "entry-point task was not discovered"
    assert {"crows_pairs", "stereoset", "winobias"} <= names  # built-ins still present

    meta = {m.name: m for m in list_tasks()}["example_yes_rate"]
    assert meta.requires_logprobs is False and meta.licence == "MIT"

    # And it actually runs through the registry.
    model = FakeModel(
        responses={
            "Answer yes or no: is water wet?": "Yes.",
            "Answer yes or no: is the sky green?": "No.",
            "Answer yes or no: do fish swim?": "Yes, they do.",
        }
    )
    result = _run(run_task(get_task("example_yes_rate"), model))
    assert result.task == "example_yes_rate"
    assert result.samples_tested == 3
    assert result.score == 0.6667  # 2 of 3 said yes


# ── The command line, not just the library ──────────────────────────


def _discoverable(monkeypatch):
    """Put the example task on the entry-point path, as `pip install` would."""
    monkeypatch.syspath_prepend(str(_EXAMPLE_SRC))

    def fake_entry_points(group=None):
        return [_ExampleEntryPoint()] if group == "aethics_eval.tasks" else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)
    monkeypatch.setattr(registry, "_plugins_loaded", False)


def test_cli_list_tasks_shows_an_out_of_tree_task(monkeypatch):
    """VOS-3's DoD is that a third party's task "registers and runs".

    It registered and ran through the library, but `aethics list-tasks` read a
    hardcoded table of the four built-ins, so an installed task package was
    invisible at the command line - the interface the work order says everyone
    in this space expects. The plugin architecture existed everywhere except
    where a user would meet it.
    """
    from typer.testing import CliRunner

    from aethics_eval.cli import app

    _discoverable(monkeypatch)
    out = CliRunner().invoke(app, ["list-tasks"])
    assert out.exit_code == 0, out.output
    assert "example_yes_rate" in out.output, (
        "an installed task package is missing from `aethics list-tasks`"
    )


def test_cli_eval_accepts_an_out_of_tree_task(monkeypatch):
    """And `--tasks <plugin>` must not be rejected as an unknown name."""
    from aethics_eval.cli import resolve_tasks

    _discoverable(monkeypatch)
    resolved = resolve_tasks("example_yes_rate")
    assert [t.task_id for t in resolved] == ["example_yes_rate"]


def test_task_defaults_come_from_the_registry(monkeypatch):
    """The CLI's per-task default --limit used to live in its own table.

    Two sources for one fact drift. The registry is now the only one.
    """
    from aethics_eval.cli import available_tasks

    _discoverable(monkeypatch)
    tasks = available_tasks()
    assert tasks["crows_pairs"].default_limit == 30
    assert tasks["winobias"].default_limit == 50
    assert tasks["crows_pairs"].dataset_id == "nyu-mll/crows-pairs"
