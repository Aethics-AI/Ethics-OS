# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Built-in benchmark tasks. Importing this package registers them (each module
runs its ``@register_task`` decorator on import). Migrated from the monolithic
``StandardBenchmarks`` in VOS-3.
"""

from importlib import import_module as _import_module

# Each module self-registers via @register_task on import.
for _mod in ("crows_pairs", "stereoset", "winobias", "bold"):
    _import_module(f"aethics_eval.tasks.benchmarks.{_mod}")
