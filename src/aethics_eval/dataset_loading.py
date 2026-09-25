# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-4 — dataset loading: pinned, cached, honest.

Three properties, in order of how much they matter.

**Honest.** A load that fails raises. It never substitutes a fallback and it
never reports provenance it did not use. That sounds obvious; the code this
replaces returned a single hand-written sentence pair when the WinoBias
download failed, and the result still said ``data_source:
uclanlp/wino_bias``. A benchmark scored against one invented pair, reported
as if it came from the published dataset, is worse than no benchmark — it is
a number someone will put in a compliance report.

**Pinned.** Every dataset loads at the revision recorded in
``dataset_licenses.py``. Without that, two runs a week apart can read
different data and report identical provenance, which quietly breaks the
reproducibility the run manifest is supposed to guarantee (POS-5).

**Cached.** Loaded rows are stored on disk with a content hash, keyed by
dataset and revision. ``--offline`` then works from that cache and refuses
rather than reaching for the network. The hash goes into the result, so two
runs can be compared without re-downloading anything.

The cache is content-addressed by revision, so it never needs invalidating:
a new revision is a new key.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .dataset_licenses import DATASET_LICENSES

CACHE_VERSION = "1"


class DatasetLoadError(RuntimeError):
    """A dataset could not be loaded.

    Raised rather than returning partial or substitute data. Callers turn this
    into a ``not_measured`` result; nothing in this package may turn it into a
    score.
    """


class OfflineUnavailableError(DatasetLoadError):
    """Offline mode was requested and the dataset is not in the cache."""


@dataclass(frozen=True)
class DatasetProvenance:
    """What was actually loaded — recorded alongside every result.

    ``content_hash`` is over the loaded rows, not the source file, so it
    covers any parsing this package does. Two runs producing the same hash
    scored the same bytes.
    """

    dataset_id: str
    revision: Optional[str]
    row_count: int
    content_hash: str
    from_cache: bool
    pinned: bool = field(default=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "row_count": self.row_count,
            "content_hash": self.content_hash,
            "from_cache": self.from_cache,
            "pinned": self.pinned,
        }


# ── Cache location ──────────────────────────────────────────────────


def cache_root() -> Path:
    """Where cached datasets live.

    ``AETHICS_CACHE_DIR`` overrides, which is what the tests use and what a CI
    job wants so a cache can be restored between runs.
    """
    override = os.environ.get("AETHICS_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "aethics-eval" / "datasets"


def _cache_path(dataset_id: str, revision: Optional[str], variant: str) -> Path:
    """One file per (dataset, revision, variant).

    The dataset id is hashed into the filename rather than used directly:
    ids contain ``/`` and arbitrary characters, and a cache path should not
    depend on them being filesystem-safe.
    """
    key = f"{CACHE_VERSION}|{dataset_id}|{revision or 'UNPINNED'}|{variant}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    safe = dataset_id.replace("/", "__")
    return cache_root() / f"{safe}.{variant}.{digest}.json"


# ── Hashing ─────────────────────────────────────────────────────────


def content_hash(rows: Any) -> str:
    """Deterministic hash of loaded rows.

    ``sort_keys`` so dict ordering cannot change the hash, and ``default=str``
    so a stray non-JSON type degrades to something stable instead of raising
    at the point we are trying to record provenance.
    """
    payload = json.dumps(rows, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ── The loader ──────────────────────────────────────────────────────


def load_dataset_pinned(
    dataset_id: str,
    fetch: Callable[[Optional[str]], Any],
    *,
    variant: str = "default",
    offline: bool = False,
    use_cache: bool = True,
) -> tuple[Any, DatasetProvenance]:
    """Load a dataset at its pinned revision, from cache when possible.

    Args:
        dataset_id: key into ``DATASET_LICENSES``; also the cache key.
        fetch: called with the pinned revision (or None) to do the actual
            download. Kept as a callback so this module needs to know nothing
            about ``datasets``, HTTP, or CSV parsing.
        variant: distinguishes several loads of one dataset — WinoBias pulls
            ``type1_pro`` and ``type1_anti`` separately.
        offline: never touch the network. A cache miss raises.
        use_cache: set False to force a fetch, e.g. when refreshing.

    Returns:
        ``(rows, provenance)``.

    Raises:
        OfflineUnavailableError: offline and not cached.
        DatasetLoadError: the fetch failed. Never returns substitute data.
    """
    record = DATASET_LICENSES.get(dataset_id)
    revision = record.revision if record else None
    pinned = revision is not None

    path = _cache_path(dataset_id, revision, variant)

    if use_cache and path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            rows = cached["rows"]
            stored = cached["content_hash"]
            actual = content_hash(rows)
            if stored != actual:
                # Corrupted or hand-edited. Refuse it rather than score data
                # that is not what we recorded.
                raise DatasetLoadError(
                    f"{dataset_id}: cache integrity check failed at {path} "
                    f"(recorded {stored[:12]}, computed {actual[:12]}). "
                    f"Delete the file to re-download."
                )
            return rows, DatasetProvenance(
                dataset_id=dataset_id,
                revision=revision,
                row_count=_row_count(rows),
                content_hash=actual,
                from_cache=True,
                pinned=pinned,
            )
        except DatasetLoadError:
            raise
        except Exception as exc:
            raise DatasetLoadError(
                f"{dataset_id}: cached copy at {path} is unreadable — {exc}. "
                f"Delete the file to re-download."
            ) from exc

    if offline:
        raise OfflineUnavailableError(
            f"{dataset_id}: --offline was requested but it is not cached "
            f"(looked in {path}). Run once without --offline to populate it."
        )

    try:
        rows = fetch(revision)
    except Exception as exc:
        raise DatasetLoadError(f"{dataset_id}: load failed — {exc}") from exc

    if rows is None or _row_count(rows) == 0:
        # An empty result is a failed load wearing a success costume. Earlier
        # code returned [] here and the benchmark scored zero rows.
        raise DatasetLoadError(
            f"{dataset_id}: loaded successfully but returned no rows"
        )

    digest = content_hash(rows)

    if use_cache:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"content_hash": digest, "rows": rows}, default=str),
                encoding="utf-8",
            )
        except OSError:
            # An unwritable cache must not fail a run that already has data.
            pass

    return rows, DatasetProvenance(
        dataset_id=dataset_id,
        revision=revision,
        row_count=_row_count(rows),
        content_hash=digest,
        from_cache=False,
        pinned=pinned,
    )


def _row_count(rows: Any) -> int:
    """Row count for a list, or for a tuple of lists (WinoBias's pro/anti)."""
    if isinstance(rows, tuple):
        return sum(len(r) for r in rows if hasattr(r, "__len__"))
    try:
        return len(rows)
    except TypeError:
        return 0


def clear_cache() -> int:
    """Delete every cached dataset. Returns how many files were removed."""
    root = cache_root()
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.glob("*.json"):
        path.unlink()
        removed += 1
    return removed


def cached_datasets() -> List[Dict[str, Any]]:
    """What is currently cached — backs ``aethics cache --list``."""
    root = cache_root()
    if not root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        entry: Dict[str, Any] = {"file": path.name, "size_bytes": path.stat().st_size}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entry["content_hash"] = data.get("content_hash", "")[:16]
            entry["row_count"] = _row_count(data.get("rows"))
        except Exception:
            entry["content_hash"] = "UNREADABLE"
            entry["row_count"] = 0
        out.append(entry)
    return out
