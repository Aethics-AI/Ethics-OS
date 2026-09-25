# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-3 — task registry and plugin discovery.

``@register_task`` puts a Task class in the registry keyed by name. ``list_tasks``
and ``get_task`` read it back. External packages register automatically via
``importlib.metadata`` entry points in the ``aethics_eval.tasks`` group — so
``pip install my-tasks`` makes new benchmarks available with no changes here.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from .base import Task, TaskMeta

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[Task]] = {}
_ENTRY_POINT_GROUP = "aethics_eval.tasks"
_plugins_loaded = False


def register_task(
    name: str,
    *,
    metric: str,
    requires_logprobs: bool,
    citation: str = "",
    licence: str = "",
    description: str = "",
    dataset_id: str = "",
    default_limit: int = 50,
):
    """Class decorator: register a Task under ``name`` with discovery metadata.

    >>> from aethics_eval.tasks import Task, get_task, task_names
    >>> @register_task("demo_task", metric="yes_fraction", requires_logprobs=False,
    ...                licence="MIT", description="A demo.")
    ... class DemoTask(Task):
    ...     def load(self, limit=None): return []
    ...     def build_request(self, sample): ...
    ...     def score(self, sample, response): ...
    ...     def aggregate(self, scores): ...
    >>> DemoTask.meta.name
    'demo_task'
    >>> DemoTask.meta.licence
    'MIT'
    >>> "demo_task" in task_names()
    True

    Registering the same name twice for a different class is refused, so a
    plugin cannot silently shadow a built-in benchmark:

    >>> try:
    ...     @register_task("demo_task", metric="x", requires_logprobs=False)
    ...     class Impostor(Task):
    ...         def load(self, limit=None): return []
    ...         def build_request(self, sample): ...
    ...         def score(self, sample, response): ...
    ...         def aggregate(self, scores): ...
    ... except ValueError as exc:
    ...     print(exc)
    task 'demo_task' is already registered

    Decorating something that is not a Task is a TypeError rather than a
    mysterious failure later:

    >>> try:
    ...     register_task("nope", metric="x", requires_logprobs=False)(object)
    ... except TypeError as exc:
    ...     print("must decorate a Task subclass" in str(exc))
    True

    >>> _REGISTRY.pop("demo_task") is DemoTask   # tidy up after the example
    True
    """

    def decorate(cls: Type[Task]) -> Type[Task]:
        if not (isinstance(cls, type) and issubclass(cls, Task)):
            raise TypeError(
                f"@register_task must decorate a Task subclass, got {cls!r}"
            )
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"task {name!r} is already registered")
        cls.meta = TaskMeta(
            name=name,
            metric=metric,
            requires_logprobs=requires_logprobs,
            citation=citation,
            licence=licence,
            description=description,
            dataset_id=dataset_id,
            default_limit=default_limit,
        )
        _REGISTRY[name] = cls
        return cls

    return decorate


def _load_plugins() -> None:
    """Import external task packages registered via entry points (once)."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    from importlib.metadata import entry_points

    # The pre-3.10 fallback (`entry_points().get(...)`) has been removed: the
    # package declares requires-python >=3.10, where the keyword form always
    # works, and EntryPoints lost `.get` in 3.12 — so the fallback could only
    # ever have raised AttributeError rather than falling back. POS-6.
    eps = entry_points(group=_ENTRY_POINT_GROUP)
    for ep in eps:
        try:
            ep.load()  # importing the module runs its @register_task decorators
        except Exception as exc:
            logger.warning("failed to load task plugin %r: %s", ep.name, exc)


def get_task(name: str) -> Task:
    """Instantiate the task registered as ``name`` (loading plugins if needed).

    Returns a new instance, not the class:

    >>> from aethics_eval.tasks import Task
    >>> isinstance(get_task("crows_pairs"), Task)
    True

    An unknown name lists what is available rather than failing blankly:

    >>> try:
    ...     get_task("no_such_task")
    ... except KeyError as exc:
    ...     print("no task registered" in str(exc))
    True
    """
    if name not in _REGISTRY:
        _load_plugins()
    if name not in _REGISTRY:
        raise KeyError(
            f"no task registered as {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()


def list_tasks() -> List[TaskMeta]:
    """All registered tasks' metadata, name-sorted (loads plugins first).

    >>> metas = list_tasks()
    >>> names = [m.name for m in metas]
    >>> all(t in names for t in ("bold", "crows_pairs", "stereoset", "winobias"))
    True
    >>> next(m for m in metas if m.name == "crows_pairs").requires_logprobs
    True

    Deliberately a subset check, not an exact list: installed plugins add
    entries, so asserting the full list would make this example depend on which
    packages happen to be present.

    Includes tasks installed by third-party packages via the
    ``aethics_eval.tasks`` entry-point group.

    .. note::
       ``aethics list-tasks`` on the command line does **not** currently use
       this — it reads a hardcoded table of the four built-ins, so plugin tasks
       are invisible there. Known gap; the library API is the reliable one.
    """
    _load_plugins()
    return sorted((cls.meta for cls in _REGISTRY.values()), key=lambda m: m.name)


def task_names() -> List[str]:
    """Registered task names, sorted.

    >>> {"bold", "crows_pairs", "stereoset", "winobias"} <= set(task_names())
    True

    The cheapest way to confirm a plugin installed correctly: install the
    package, then check its task appears here.
    """
    _load_plugins()
    return sorted(_REGISTRY)
