# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-3 Part C — the migration is complete: the registry can stand in for the
legacy StandardBenchmarks suite.

The byte-identical parity guard that VOS-7 depends on lives in the per-benchmark
tests (test_crows_pairs_task / test_stereoset_task / test_winobias_task /
test_bold_task) — each asserts the registry result == the pre-refactor engine.
This file guards *completeness*: no benchmark is left unmigrated, and if someone
adds a new run_* method to StandardBenchmarks it fails until a task exists.
"""

from aethics_eval.standard_benchmarks import StandardBenchmarks
from aethics_eval.tasks import get_task, list_tasks

# legacy benchmark display name -> registered task name
_MIGRATED = {
    "WinoBias": "winobias",
    "StereoSet": "stereoset",
    "CrowS-Pairs": "crows_pairs",
    "BOLD": "bold",
}


def test_every_legacy_benchmark_has_a_registered_task():
    names = {m.name for m in list_tasks()}
    for display, task_name in _MIGRATED.items():
        assert task_name in names, f"{display} not migrated to the registry"


def test_no_run_method_is_left_unmigrated():
    # Every run_<benchmark> on StandardBenchmarks must have a registry task, so a
    # newly added benchmark can't silently skip the registry (and its parity guard).
    run_benchmarks = {
        m[len("run_") :]
        for m in dir(StandardBenchmarks)
        if m.startswith("run_") and m != "run_all_benchmarks"
    }
    task_names = {m.name for m in list_tasks()}
    missing = run_benchmarks - task_names
    assert not missing, f"benchmarks without a registered task: {sorted(missing)}"


def test_each_migrated_task_exposes_discovery_metadata():
    for name in _MIGRATED.values():
        meta = get_task(name).meta
        assert meta.metric, f"{name} missing metric"
        assert meta.licence, f"{name} missing licence"
        assert meta.citation, f"{name} missing citation"
        assert isinstance(meta.requires_logprobs, bool)
