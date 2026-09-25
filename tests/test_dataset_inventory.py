# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-2 — a dataset cannot enter the package undocumented.

The failure this guards against is someone adding a `load_dataset("foo/bar")`
call in six months and shipping it with no licence, no citation and no pinned
revision. Attribution mistakes in a public package are the hardest kind to
walk back, so the check is a test rather than a review convention.

Three properties:

  * every dataset the loaders actually fetch is in the licence registry
  * every registry entry is documented in docs/datasets.md
  * nothing marked non-redistributable is vendored into the source tree

The first is deliberately a source scan rather than an import: it has to see
loaders in modules that cannot currently be imported (the hf_client cord),
and it has to catch a new loader even if no test ever calls it.
"""

from __future__ import annotations

import re
from pathlib import Path

from aethics_eval.dataset_licenses import (
    DATASET_LICENSES,
    ORIGINAL_PROMPT_SETS,
    get_license,
    non_redistributable,
)

SDK_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = SDK_ROOT / "src" / "aethics_eval"
DATASETS_DOC = SDK_ROOT / "docs" / "datasets.md"

# `load_dataset("owner/name", ...)` — the HuggingFace path.
_LOAD_DATASET = re.compile(r"""load_dataset\(\s*["']([^"']+)["']""")
# Raw file pulls, e.g. the CrowS-Pairs CSV on GitHub.
_RAW_URL = re.compile(r"""["'](https://raw\.githubusercontent\.com/[^"']+)["']""")


def _source_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _loaded_dataset_ids() -> dict[str, str]:
    """Map dataset identifier -> where it is loaded, by scanning source."""
    found: dict[str, str] = {}
    for path in _source_files():
        rel = path.relative_to(SDK_ROOT)
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for hit in _LOAD_DATASET.findall(line):
                found.setdefault(hit, f"{rel}:{lineno}")
            for url in _RAW_URL.findall(line):
                # github.com/<owner>/<repo>/... -> "owner/repo"
                parts = url.split("/")
                if len(parts) > 4:
                    found.setdefault(f"{parts[3]}/{parts[4]}", f"{rel}:{lineno}")
    return found


def test_every_loaded_dataset_is_registered() -> None:
    """A loader with no licence entry fails the build."""
    loaded = _loaded_dataset_ids()
    assert loaded, "no dataset loaders found — has the scan pattern gone stale?"

    missing = {ds: where for ds, where in loaded.items() if ds not in DATASET_LICENSES}
    assert not missing, (
        "These datasets are loaded but have no entry in dataset_licenses.py.\n"
        "Add the licence, citation, source URL and pinned revision before "
        "shipping:\n  "
        + "\n  ".join(f"{ds}  (loaded at {where})" for ds, where in missing.items())
    )


def test_registry_entries_are_complete() -> None:
    """No entry may be half-filled — a blank licence is worse than none."""
    problems: list[str] = []
    for ds_id, rec in DATASET_LICENSES.items():
        for field in ("name", "source_url", "license_id", "license_url", "citation"):
            if not getattr(rec, field):
                problems.append(f"{ds_id}: {field} is empty")
        if rec.dataset_id != ds_id:
            problems.append(f"{ds_id}: dataset_id mismatch ({rec.dataset_id!r})")
    assert not problems, "Incomplete licence records:\n  " + "\n  ".join(problems)


def test_every_dataset_is_documented() -> None:
    """docs/datasets.md must mention every registered dataset and its licence."""
    assert DATASETS_DOC.is_file(), f"missing {DATASETS_DOC}"
    doc = DATASETS_DOC.read_text(encoding="utf-8")

    undocumented = [ds_id for ds_id in DATASET_LICENSES if ds_id not in doc]
    assert not undocumented, (
        "Registered but absent from docs/datasets.md: " + ", ".join(undocumented)
    )

    unlicensed = [
        ds_id for ds_id, rec in DATASET_LICENSES.items() if rec.license_id not in doc
    ]
    assert not unlicensed, "Documented without their licence identifier: " + ", ".join(
        unlicensed
    )

    uncited = [
        ds_id
        for ds_id, rec in DATASET_LICENSES.items()
        # first author surname is enough of a fingerprint for the citation
        if rec.citation.split(",")[0] not in doc
    ]
    assert not uncited, "Documented without their required citation: " + ", ".join(
        uncited
    )


def test_documented_revisions_match_the_registry() -> None:
    """The pinned revision in the docs must be the one the loader actually uses.

    This guards a drift we actually hit: POS-2 documented CrowS-Pairs as
    "not pinned - see below" and told POS-4 to fix it. POS-4 did, but the
    document was not updated, so for several weeks docs/datasets.md understated
    the package's own provenance guarantees. Coverage was already tested; the
    *values* were not, so nothing failed.

    A documented revision that disagrees with the registry is worse than an
    undocumented one, because it is the version a reader will cite.
    """
    doc = DATASETS_DOC.read_text(encoding="utf-8")

    wrong: list[str] = []
    for ds_id, rec in DATASET_LICENSES.items():
        if not rec.revision:
            continue
        if rec.revision not in doc:
            wrong.append(
                f"{ds_id}: registry pins {rec.revision!r}, "
                "which does not appear in docs/datasets.md"
            )
    problem_text = ("\n  ").join(wrong)
    assert not wrong, (
        "Documented revision disagrees with the registry:" + "\n  " + problem_text
    )


def test_docs_do_not_claim_an_unpinned_dataset_while_one_is_pinned() -> None:
    """No stale "not pinned" / "must replace" language for a dataset that is pinned.

    The prose is what a reader trusts, so a leftover TODO in it is a factual
    error about the package even when every structured field is correct.
    """
    doc = DATASETS_DOC.read_text(encoding="utf-8")
    pinned = {ds_id for ds_id, rec in DATASET_LICENSES.items() if rec.revision}

    stale_markers = ("**not pinned", "must replace", "still loading from a moving")
    hits = [m for m in stale_markers if m in doc.lower()]
    assert not hits or not pinned, (
        "docs/datasets.md still carries unresolved-pinning language "
        f"({', '.join(hits)}) while these datasets are pinned: "
        + ", ".join(sorted(pinned))
    )


def _module_level_constants() -> list[tuple[Path, int, str, int]]:
    """Find module-level CONSTANTS holding literal collections, with their size.

    Returns (path, lineno, name, entry_count). Entry count is measured by
    counting lines until the collection closes at column 0, which is crude but
    good enough to tell a handful of templates from a pasted corpus.
    """
    found: list[tuple[Path, int, str, int]] = []
    for path in _source_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^([A-Z][A-Z0-9_]+)\s*(?::[^=]+)?=\s*[\[\({]", line)
            if not m:
                continue
            size = 0
            for follow in lines[i + 1 :]:
                if re.match(r"^[\]\)}]", follow):
                    break
                size += 1
            found.append((path, i + 1, m.group(1), size))
    return found


def test_share_alike_data_is_not_vendored_by_name() -> None:
    """No constant may be named after a dataset we cannot redistribute.

    Three of the four datasets are CC-BY-SA-4.0. Shipping their rows would put
    a ShareAlike obligation on this package and collide with a permissive
    licence, so they are loaded at runtime and never copied.

    Constants imitating a benchmark's *format* are legitimate and must declare
    their provenance in ORIGINAL_PROMPT_SETS. Anything else named after a
    blocked benchmark needs a human to look at it.
    """
    blocked = non_redistributable()
    assert blocked, "expected at least one non-redistributable dataset"

    suspicious: list[str] = []
    for path, lineno, const, _ in _module_level_constants():
        if const in ORIGINAL_PROMPT_SETS:
            continue
        for rec in blocked.values():
            token = rec.name.split()[0].upper().replace("-", "_")
            if token in const:
                suspicious.append(f"{path.relative_to(SDK_ROOT)}:{lineno}: {const}")

    assert not suspicious, (
        "Module-level constants named after a ShareAlike benchmark. If these "
        "contain upstream rows they cannot ship; if they are original, add them "
        "to ORIGINAL_PROMPT_SETS with a note on where the content came from:\n  "
        + "\n  ".join(suspicious)
    )


# A corpus paste looks different from a hand-written prompt set: it is large.
# 60 is comfortably above every legitimate set in the package today (the
# largest is ~124 lines of generated occupation prompts, which is declared)
# and far below the size of any real benchmark split.
VENDORED_CORPUS_THRESHOLD = 150


def test_no_large_undeclared_literal_collections() -> None:
    """Catch a pasted corpus even when it is not named after its source.

    The name-based check above is trivially defeated by calling the constant
    something else. This one is content-shaped instead: any large module-level
    literal collection that has not declared its provenance is flagged, whatever
    it is called. It cannot prove data is not vendored — only a human reading
    the diff can — but it removes the easy path.
    """
    undeclared: list[str] = []
    for path, lineno, const, size in _module_level_constants():
        if const in ORIGINAL_PROMPT_SETS:
            continue
        if size >= VENDORED_CORPUS_THRESHOLD:
            undeclared.append(
                f"{path.relative_to(SDK_ROOT)}:{lineno}: {const} (~{size} lines)"
            )

    assert not undeclared, (
        "Large literal collections with no declared provenance. If this is "
        "third-party data, check its licence before it ships — three of our "
        "four datasets are ShareAlike and cannot be redistributed. If it is "
        "our own, add it to ORIGINAL_PROMPT_SETS with a note saying so:\n  "
        + "\n  ".join(undeclared)
    )


def test_original_prompt_sets_declare_provenance() -> None:
    """Every exemption must say where its content came from.

    ORIGINAL_PROMPT_SETS is the escape hatch for the two checks above, so an
    entry has to cost something: a name plus a real sentence about origin. An
    empty or placeholder note is not a declaration.
    """
    thin = [
        name
        for name, provenance in ORIGINAL_PROMPT_SETS.items()
        if len(provenance.strip()) < 15
    ]
    assert not thin, (
        "These exemptions have no meaningful provenance note. State where the "
        "content came from:\n  " + "\n  ".join(thin)
    )


def test_get_license_rejects_unknown_dataset() -> None:
    """The lookup must fail loudly, not return a permissive default."""
    try:
        get_license("someone/undocumented-dataset")
    except KeyError as exc:
        assert "licence registry" in str(exc)
    else:
        raise AssertionError("unknown dataset did not raise")
