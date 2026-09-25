# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-7 — the public API's docstring examples must actually run.

The work order asks for a worked example on every public symbol. An example
that is never executed is worse than no example at all: it carries the
authority of having been checked while drifting from the code the moment a
signature or a rounding rule changes.

Deliberately a test module rather than `--doctest-modules` in pyproject.toml.
A path argument overrides `testpaths`, and CI runs `pytest tests/`, so the
config-level approach collects doctests locally and silently skips them in CI
— the same failure mode this file exists to prevent, one level up.

Covers the three packages that declare a public surface via `__all__`:
`aethics_eval`, `aethics_eval.tasks` and `aethics_eval.results`. The legacy
engine's docstrings were not written as runnable examples and would fail en
masse; turning those into examples is real work for whoever next owns them.

Three layers, because the obvious single check is not enough:

  * the examples run and match their stated output;
  * each module actually contains examples (`doctest.testmod` reports zero
    failures for a module with none, so deleting every `>>>` would pass);
  * every name in each `__all__` has one, and no package declaring `__all__`
    escapes the list.
"""

from __future__ import annotations

import doctest
import importlib

import pytest

import aethics_eval
from aethics_eval import aggregation, eval_config, scoring
from aethics_eval import results as results_pkg
from aethics_eval.results import loader, schema, uncertainty, validation
from aethics_eval.tasks import base, registry, runner

# Modules whose docstring examples are executed. Add a module here when its
# symbols become public.
DOCTESTED_MODULES = [
    aggregation,
    eval_config,
    scoring,
    base,
    registry,
    runner,
    schema,
    loader,
    uncertainty,
    validation,
    results_pkg,
]

# Every package that declares a public surface via __all__. The first version of
# this file checked only `aethics_eval.__all__` and passed at 12/12 while 30
# symbols in `tasks` and `results` had no example at all — a gate that looked
# like it enforced the requirement and enforced a smaller one. Listing the
# packages here is what stops that recurring.
PUBLIC_PACKAGES = [
    "aethics_eval",
    "aethics_eval.tasks",
    "aethics_eval.results",
]


@pytest.mark.parametrize("module", DOCTESTED_MODULES, ids=lambda m: m.__name__)
def test_docstring_examples_run(module) -> None:
    """Every `>>>` example in the module produces exactly what it claims."""
    results = doctest.testmod(module, verbose=False, report=False)

    assert results.failed == 0, (
        f"{results.failed} of {results.attempted} docstring examples in "
        f"{module.__name__} do not match their stated output. Run "
        f"`python -m doctest -v {module.__file__}` for the diff."
    )


@pytest.mark.parametrize("module", DOCTESTED_MODULES, ids=lambda m: m.__name__)
def test_module_actually_has_examples(module) -> None:
    """Guard against the test passing vacuously.

    `doctest.testmod` reports zero failures for a module containing no examples
    at all, so deleting every `>>>` would leave this file green. Assert that
    examples exist as well as that they pass.
    """
    finder = doctest.DocTestFinder()
    example_count = sum(len(t.examples) for t in finder.find(module))

    assert example_count > 0, (
        f"{module.__name__} has no docstring examples — either they were "
        f"removed, or this module does not belong in DOCTESTED_MODULES."
    )


@pytest.mark.parametrize("package", PUBLIC_PACKAGES)
def test_every_public_symbol_has_an_example(package: str) -> None:
    """POS-7's actual requirement: an example on each public symbol.

    Reads each package's `__all__` rather than a hand-maintained list, so a
    newly exported symbol fails here until someone documents it.
    """
    module = importlib.import_module(package)
    exported = getattr(module, "__all__", None)
    assert exported, f"{package} declares no __all__ — public surface undefined"

    undocumented = []
    for name in exported:
        obj = getattr(module, name)
        # Module-level constants (`__version__`, SCHEMA_VERSION, the version
        # strings) carry the docstring of their type, not their own, so there
        # is nothing to check. Same for plain ints like DEFAULT_RESAMPLES.
        if isinstance(obj, (str, int, float)):
            continue
        doc = doctest.inspect.getdoc(obj) or ""
        if ">>>" not in doc:
            undocumented.append(name)

    assert not undocumented, (
        f"POS-7 requires a worked example in the docstring of every public "
        f"symbol. Missing from {package}: " + ", ".join(sorted(undocumented))
    )


def test_public_packages_covers_every_package_with_an_all() -> None:
    """Guard the guard: a new package declaring __all__ must be listed above.

    The failure this prevents is subtle — adding a public package and simply
    forgetting to add it to PUBLIC_PACKAGES leaves the requirement unenforced
    for that package while every test still passes. That is exactly how the
    first version of this file reported 12/12 with 30 symbols undocumented.
    """
    import pkgutil

    found = []
    for info in pkgutil.walk_packages(aethics_eval.__path__, prefix="aethics_eval."):
        if not info.ispkg:
            continue
        mod = importlib.import_module(info.name)
        if getattr(mod, "__all__", None):
            found.append(info.name)

    missing = sorted(set(found) - set(PUBLIC_PACKAGES))
    assert not missing, (
        "These packages declare __all__ but are not covered by the public-symbol "
        "check. Add them to PUBLIC_PACKAGES: " + ", ".join(missing)
    )
