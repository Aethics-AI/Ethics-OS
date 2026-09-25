# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-4 — dataset loading: pinned, cached, honest.

The failure this guards against is a benchmark scoring substitute data and
reporting the real dataset's name. That is not hypothetical: WinoBias used to
fall back to one hand-written sentence pair when the download failed, and the
result still said ``data_source: uclanlp/wino_bias``. A number produced that
way ends up in a compliance report.

So the rule the tests enforce is narrow and absolute: **a load either returns
the pinned data or raises.** There is no third option, and no code path turns
a raise into a score.

Everything here is offline — the fetch callback is a local function.
"""

from __future__ import annotations

import json

import pytest

from aethics_eval.dataset_loading import (
    DatasetLoadError,
    OfflineUnavailableError,
    cached_datasets,
    clear_cache,
    content_hash,
    load_dataset_pinned,
)

ROWS = [{"text": f"row {i}", "label": i % 2} for i in range(20)]


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never touch the developer's real cache."""
    monkeypatch.setenv("AETHICS_CACHE_DIR", str(tmp_path / "cache"))


# ── pinning ─────────────────────────────────────────────────────────


def test_the_pinned_revision_is_passed_to_the_fetcher() -> None:
    """The whole point of pinning: the loader must actually receive the SHA."""
    seen = {}

    def fetch(revision):
        seen["revision"] = revision
        return ROWS

    _, prov = load_dataset_pinned("AlexaAI/bold", fetch)

    assert seen["revision"] == "be5f5a99b386a7c4fa7ea905685ee2d2c98301eb"
    assert prov.pinned is True
    assert prov.revision == seen["revision"]


def test_crows_pairs_is_pinned() -> None:
    """POS-2 found this loading from `master`. It must not go back."""
    from aethics_eval.dataset_licenses import DATASET_LICENSES

    rec = DATASET_LICENSES["nyu-mll/crows-pairs"]
    assert rec.revision, (
        "CrowS-Pairs has no pinned revision. Loading from a branch means two "
        "runs can read different data and report identical provenance."
    )
    assert len(rec.revision) == 40, "expected a full commit SHA"


def test_every_registered_dataset_is_pinned() -> None:
    from aethics_eval.dataset_licenses import DATASET_LICENSES

    unpinned = [k for k, v in DATASET_LICENSES.items() if not v.revision]
    assert not unpinned, f"unpinned datasets: {unpinned}"


# ── honesty: raise, never substitute ────────────────────────────────


def test_a_failed_fetch_raises_rather_than_returning_anything() -> None:
    def fetch(revision):
        raise ConnectionError("network down")

    with pytest.raises(DatasetLoadError, match="load failed"):
        load_dataset_pinned("AlexaAI/bold", fetch)


def test_an_empty_result_is_treated_as_a_failure() -> None:
    """`return []` is a failed load wearing a success costume.

    The previous loaders returned [] on error and the benchmark scored zero
    rows, which reads downstream as "we measured, and found nothing".
    """
    with pytest.raises(DatasetLoadError, match="no rows"):
        load_dataset_pinned("AlexaAI/bold", lambda rev: [])


def test_none_is_treated_as_a_failure() -> None:
    with pytest.raises(DatasetLoadError):
        load_dataset_pinned("AlexaAI/bold", lambda rev: None)


# ── caching ─────────────────────────────────────────────────────────


def test_second_load_comes_from_cache_without_fetching() -> None:
    calls = []

    def fetch(revision):
        calls.append(1)
        return ROWS

    load_dataset_pinned("AlexaAI/bold", fetch)
    rows, prov = load_dataset_pinned("AlexaAI/bold", fetch)

    assert len(calls) == 1, "the second load hit the network"
    assert prov.from_cache is True
    assert rows == ROWS


def test_content_hash_is_stable_across_runs() -> None:
    """The DoD property: two runs on one revision produce identical hashes."""
    _, a = load_dataset_pinned("AlexaAI/bold", lambda rev: ROWS)
    clear_cache()
    _, b = load_dataset_pinned("AlexaAI/bold", lambda rev: ROWS)

    assert a.content_hash == b.content_hash
    assert a.row_count == b.row_count == len(ROWS)


def test_hash_changes_when_the_data_changes() -> None:
    """A hash that never changes is not evidence of anything."""
    assert content_hash(ROWS) != content_hash(ROWS[:-1])


def test_variants_are_cached_separately() -> None:
    """WinoBias loads type1_pro and type1_anti — they must not collide."""
    load_dataset_pinned("uclanlp/wino_bias", lambda rev: ROWS, variant="pro")
    rows, prov = load_dataset_pinned(
        "uclanlp/wino_bias", lambda rev: ROWS[:5], variant="anti"
    )
    assert prov.from_cache is False, "the anti variant read the pro cache"
    assert len(rows) == 5


def test_a_tampered_cache_is_refused() -> None:
    """Integrity check: scoring edited data silently would be worse than failing."""
    load_dataset_pinned("AlexaAI/bold", lambda rev: ROWS)

    entries = cached_datasets()
    assert entries, "nothing was cached"

    from aethics_eval.dataset_loading import cache_root

    path = next(cache_root().glob("*.json"))
    data = json.loads(path.read_text())
    data["rows"].append({"text": "smuggled in", "label": 0})
    path.write_text(json.dumps(data))

    with pytest.raises(DatasetLoadError, match="integrity"):
        load_dataset_pinned("AlexaAI/bold", lambda rev: ROWS)


def test_an_unreadable_cache_is_refused_not_ignored() -> None:
    load_dataset_pinned("AlexaAI/bold", lambda rev: ROWS)

    from aethics_eval.dataset_loading import cache_root

    next(cache_root().glob("*.json")).write_text("{ not json")

    with pytest.raises(DatasetLoadError, match="unreadable"):
        load_dataset_pinned("AlexaAI/bold", lambda rev: ROWS)


# ── offline ─────────────────────────────────────────────────────────


def test_offline_serves_from_cache() -> None:
    load_dataset_pinned("AlexaAI/bold", lambda rev: ROWS)

    rows, prov = load_dataset_pinned(
        "AlexaAI/bold",
        lambda rev: pytest.fail("offline mode reached the network"),
        offline=True,
    )
    assert prov.from_cache is True
    assert rows == ROWS


def test_offline_refuses_rather_than_fetching() -> None:
    """An 'offline' run that quietly went online is a lie about the run."""
    with pytest.raises(OfflineUnavailableError, match="not cached"):
        load_dataset_pinned(
            "AlexaAI/bold",
            lambda rev: pytest.fail("offline mode reached the network"),
            offline=True,
        )


def test_offline_error_says_how_to_fix_it() -> None:
    with pytest.raises(OfflineUnavailableError) as exc:
        load_dataset_pinned("AlexaAI/bold", lambda rev: ROWS, offline=True)
    assert "without --offline" in str(exc.value)


# ── cache management ────────────────────────────────────────────────


def test_clear_cache_removes_entries() -> None:
    load_dataset_pinned("AlexaAI/bold", lambda rev: ROWS)
    assert cached_datasets()

    removed = clear_cache()
    assert removed >= 1
    assert cached_datasets() == []


def test_cached_datasets_reports_rows_and_hash() -> None:
    load_dataset_pinned("AlexaAI/bold", lambda rev: ROWS)
    entry = cached_datasets()[0]

    assert entry["row_count"] == len(ROWS)
    assert entry["content_hash"]
    assert entry["size_bytes"] > 0


# ── benchmark level: a failed load must not become a score ──────────


def _working_model():
    """A model that would score fine — so any not_measured is the dataset's doing."""
    from aethics_eval.models import Capabilities, Generation

    class M:
        @property
        def capabilities(self) -> Capabilities:
            return Capabilities(
                supports_logprobs=True, supports_echo=True, max_context=2048
            )

        async def generate(self, prompt: str, **_: object) -> Generation:
            return Generation(text="a helpful reply", success=True)

        async def sequence_logprob(self, text: str) -> float:
            return -12.5

    return M()


@pytest.mark.parametrize(
    ("method", "loader"),
    [
        ("run_winobias", "_load_winobias"),
        ("run_stereoset", "_load_stereoset"),
        ("run_crows_pairs", "_load_crows_pairs"),
        ("run_bold", "_load_bold"),
    ],
)
def test_load_failure_yields_not_measured_not_a_score(
    monkeypatch, method: str, loader: str
) -> None:
    """The DoD check: a simulated load failure must not produce a number.

    The model is healthy, so the only thing wrong is the dataset. Every
    benchmark must answer score=None with reliability not_measured — never a
    fabricated value, and never a score computed from substitute data.
    """
    import asyncio

    from aethics_eval.standard_benchmarks import create_benchmark_suite

    suite = create_benchmark_suite()

    def boom(*_a, **_k):
        raise DatasetLoadError("simulated: upstream unreachable")

    monkeypatch.setattr(suite, loader, boom)

    result = asyncio.run(getattr(suite, method)(_working_model(), sample_size=5))

    assert result.score is None, (
        f"{method} returned {result.score!r} after the dataset failed to load. "
        f"A score computed from data we never got is a fabricated measurement."
    )
    assert result.samples_tested == 0
    assert result.passed is False
    assert result.details.get("reliability") == "not_measured"
    assert "unreachable" in str(result.details.get("reason", ""))


def test_load_failure_does_not_claim_upstream_provenance(monkeypatch) -> None:
    """WinoBias used to fall back to one hand-written pair and still report
    ``data_source: uclanlp/wino_bias``. The fallback is gone; if any equivalent
    comes back, the row count gives it away."""
    import asyncio

    from aethics_eval.standard_benchmarks import create_benchmark_suite

    suite = create_benchmark_suite()
    monkeypatch.setattr(
        suite,
        "_load_winobias",
        lambda *_a, **_k: (_ for _ in ()).throw(DatasetLoadError("offline")),
    )

    result = asyncio.run(suite.run_winobias(_working_model(), sample_size=5))

    assert result.samples_tested == 0, (
        "samples were scored despite the dataset failing to load — something is "
        "substituting data"
    )
    assert result.score is None
