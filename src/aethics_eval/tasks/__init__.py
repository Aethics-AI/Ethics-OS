# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Task registry and plugin system (VOS-3).

Public API:
    Task, Sample, Request, Response, SampleScore, TaskResult, TaskMeta
    register_task, get_task, list_tasks, task_names, run_task

Built-in benchmark tasks self-register on import (added in Part B); external
packages register via the ``aethics_eval.tasks`` entry-point group.
"""

from .base import (
    Request,
    Response,
    Sample,
    SampleScore,
    Task,
    TaskMeta,
    TaskResult,
)
from .registry import get_task, list_tasks, register_task, task_names
from .runner import dispatch, run_task

__all__ = [
    "Request",
    "Response",
    "Sample",
    "SampleScore",
    "Task",
    "TaskMeta",
    "TaskResult",
    "dispatch",
    "get_task",
    "list_tasks",
    "register_task",
    "run_task",
    "task_names",
]

# Import last (after the public API is defined) so built-in benchmark tasks
# self-register on `import aethics_eval.tasks` without a circular import.
from importlib import import_module as _import_module

_import_module("aethics_eval.tasks.benchmarks")
