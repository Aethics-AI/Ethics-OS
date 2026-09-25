# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""POS-3 — the ``aethics`` command line interface.

``lm_eval --model hf --tasks hellaswag`` is the interface everyone in this
space expects. A library without one reads as unfinished, so this is the
surface most people will judge the package by.

Four commands::

    aethics eval --model hf:gpt2 --tasks crows_pairs --limit 50 -o results.json
    aethics list-tasks
    aethics validate results.json
    aethics show results.json

Conventions, because a CLI that does not compose is a CLI people script
around rather than with:

* results go to **stdout**, progress and diagnostics to **stderr**, so
  ``aethics eval ... | jq`` works without ``2>/dev/null``.
* exit codes: 0 success, 1 runtime failure, 2 bad usage.
* every run stamps the methodology fingerprint and the dataset revisions it
  actually used, so a result can be traced back to how it was produced.

The task table below is temporary. VOS-3 introduces a real registry with
plugin discovery via entry points; when it lands, ``_TASKS`` is replaced by
that lookup and everything else here is unchanged. It is deliberately built
from the two sources that already exist — the benchmark suite's methods and
the dataset licence registry — rather than being a third parallel list of
facts to keep in sync.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from . import __version__
from .dataset_licenses import DATASET_LICENSES
from .eval_config import EVAL_INFERENCE_PARAMS, EVAL_MANIFEST
from .manifest import (
    DEFAULT_SEED,
    DatasetRecord,
    build_manifest,
    canonical_seeds,
    set_global_seeds,
    verify_manifest,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Reproducible bias and safety evaluation for language models.",
)

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


# Result envelope version. VOS-4 owns the real schema; this is the minimum
# needed for `validate` and `show` to be meaningful in the meantime, and is
# deliberately a separate constant so bumping it is a visible decision.
def _registry_metas() -> List[Any]:
    from .tasks import list_tasks as _lt

    return list(_lt())


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


def err(msg: str) -> None:
    """Diagnostics go to stderr so stdout stays pipeable."""
    typer.echo(msg, err=True)


# ── Tasks ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Task:
    """One runnable benchmark.

    ``method`` is the coroutine on StandardBenchmarks; ``dataset_id`` keys into
    DATASET_LICENSES so citation and licence are never restated here.
    """

    task_id: str
    method: str
    dataset_id: str
    requires_logprobs: bool
    default_limit: int


# The legacy suite still owns four method names; the registry does not model
# them because a registered Task runs through its own four-stage API. Kept only
# as a fallback for a task that is registered but has no registry entry.
# The benchmarks this package ships, as distinct from whatever a third-party
# package has registered. Used to keep the default run stable across unrelated
# installs.
_BUILTIN_TASKS = frozenset({"bold", "crows_pairs", "stereoset", "winobias"})

_LEGACY_METHODS = {
    "winobias": "run_winobias",
    "stereoset": "run_stereoset",
    "crows_pairs": "run_crows_pairs",
    "bold": "run_bold",
}


def available_tasks() -> Dict[str, Task]:
    """Every registered task, including ones installed by third-party packages.

    This used to be a hardcoded table of the four built-ins. VOS-3 shipped a
    registry with entry-point discovery, and the CLI never moved onto it, so
    `pip install` of an out-of-tree task package registered correctly and was
    then invisible to `aethics list-tasks` and rejected by `aethics eval` -
    the plugin architecture existed in the library and not at the command line
    anyone actually uses.
    """
    from .tasks import list_tasks as _registry_list_tasks

    return {
        m.name: Task(
            task_id=m.name,
            method=_LEGACY_METHODS.get(m.name, ""),
            dataset_id=m.dataset_id,
            requires_logprobs=m.requires_logprobs,
            default_limit=m.default_limit,
        )
        for m in _registry_list_tasks()
    }


def resolve_tasks(spec: str) -> List[Task]:
    """Parse ``--tasks a,b,c``, or ``all``. Unknown names are a usage error."""
    known = available_tasks()
    choice = spec.strip().lower()

    # "builtin" is the default when --tasks is omitted: the four benchmarks
    # this package ships. It is deliberately NOT "all".
    #
    # `all` includes tasks registered by installed third-party packages, which
    # is what the registry is for - but as a *default* that meant `pip install
    # some-tasks` silently changed which benchmarks ran and what
    # summary.mean_score averaged over, in an unrelated project, with nothing in
    # the output saying so. Widening a run is a decision, so it is now asked for.
    if choice in ("builtin", "builtins", "default"):
        return [known[k] for k in sorted(known) if k in _BUILTIN_TASKS]

    if choice == "all":
        selected = [known[k] for k in sorted(known)]
        external = [t.task_id for t in selected if t.task_id not in _BUILTIN_TASKS]
        if external:
            err(
                f"note: --tasks all includes {len(external)} task(s) from installed "
                f"packages: {', '.join(external)}"
            )
        return selected

    wanted = [t.strip() for t in spec.split(",") if t.strip()]
    if not wanted:
        raise typer.BadParameter("no tasks given")

    unknown = [t for t in wanted if t not in known]
    if unknown:
        raise typer.BadParameter(
            f"unknown task(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(known))}. Run `aethics list-tasks`."
        )
    return [known[t] for t in wanted]


# ── Config file ─────────────────────────────────────────────────────

# Keys a config file may set. Anything else is a typo and is reported as one,
# because a silently-ignored setting in a checked-in config is worse than an
# error — the run looks configured and is not.
_CONFIG_KEYS = frozenset(
    {"model", "tasks", "limit", "output", "base_url", "max_context", "offline", "seed"}
)


def load_config(path: Path) -> Dict[str, Any]:
    """Read a YAML or JSON run configuration.

    The point of ``--config`` is that a run can be committed to version
    control and reproduced, so unknown keys are an error rather than a
    shrug: a config that claims to set something it does not is a worse
    failure than one that refuses to load.
    """
    if not path.is_file():
        raise typer.BadParameter(f"config not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except Exception as exc:
        raise typer.BadParameter(f"{path}: could not parse — {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{path}: top level must be a mapping")

    unknown = sorted(set(data) - _CONFIG_KEYS)
    if unknown:
        raise typer.BadParameter(
            f"{path}: unknown key(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(_CONFIG_KEYS))}"
        )
    return data


# ── Model providers ─────────────────────────────────────────────────

_PROVIDERS = ("hf", "local", "openai", "fake")

# Checked in order. A key is never accepted as a command-line flag: argv is
# visible in shell history and to `ps`, and a config file is committed to
# version control. An environment variable is the only one of the three that
# is not routinely recorded somewhere.
_API_KEY_ENV = ("AETHICS_API_KEY", "OPENAI_API_KEY")


class _OpenAICompatClient:
    """Minimal sync client for an OpenAI-shaped ``/chat/completions``.

    ``OpenAICompatModel`` expects an object with ``generate(prompt, **params)``
    returning ``{"success": bool, "text": str, "error": ...}``. This is that,
    and nothing more — enough to score a hosted model, with no vendor SDK.

    The key is read from the environment and never logged, echoed, or written
    into a result file.
    """

    def __init__(self, base_url: str, model_id: str, api_key: Optional[str]):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model_id = model_id
        self._key = api_key

    def generate(self, prompt: str, **params: Any) -> Dict[str, Any]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"

        body = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": params.get("max_tokens", 150),
            # Deterministic, matching EVAL_INFERENCE_PARAMS. An evaluation that
            # samples is not reproducible.
            "temperature": 0.0,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(self._url, headers=headers, json=body)
            if resp.status_code != 200:
                # Body may echo the key back in an auth error; report the
                # status only.
                return {
                    "success": False,
                    "error": f"HTTP {resp.status_code} from {self._url}",
                }
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return {"success": True, "text": text}
        except Exception as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


def build_model(
    spec: str,
    base_url: Optional[str] = None,
    max_context: int = 4096,
) -> Any:
    """Turn ``provider:model_id`` into a Model.

    Imported lazily: `local` pulls torch, and `aethics list-tasks` should not
    need a 2GB dependency resolved to print a table.
    """
    if ":" not in spec:
        raise typer.BadParameter(
            f"--model must be provider:model_id (e.g. hf:gpt2). "
            f"Providers: {', '.join(_PROVIDERS)}"
        )

    provider, _, model_id = spec.partition(":")
    provider = provider.strip().lower()

    if provider not in _PROVIDERS:
        raise typer.BadParameter(
            f"unknown provider {provider!r}. Providers: {', '.join(_PROVIDERS)}"
        )
    if provider != "fake" and not model_id.strip():
        raise typer.BadParameter(
            f"{provider}: requires a model id, e.g. {provider}:gpt2"
        )

    from . import models as _m

    if provider == "hf":
        return _m.HFInferenceModel(model_id)
    if provider == "local":
        return _m.LocalHFModel(model_id)
    if provider == "openai":
        if not base_url:
            raise typer.BadParameter(
                "openai: needs --base-url (or base_url in --config), e.g. "
                "--base-url https://api.openai.com/v1"
            )
        key = next((os.environ[v] for v in _API_KEY_ENV if os.environ.get(v)), None)
        if not key:
            err(
                f"note: no API key found in {' or '.join(_API_KEY_ENV)} — "
                f"sending unauthenticated"
            )
        return _m.OpenAICompatModel(
            _OpenAICompatClient(base_url, model_id, key),
            # The generic chat endpoint returns no log-probs, so the benchmarks
            # that need them will report not_measured rather than a proxy.
            supports_logprobs=False,
            max_context=max_context,
        )
    # fake: offline, scriptable — the whole point is running with no network.
    return _m.FakeModel(default_response=model_id or "I cannot help with that.")


# ── eval ────────────────────────────────────────────────────────────


def _dataset_provenance(tasks: List[Task]) -> Dict[str, Any]:
    """Record which dataset revisions a run actually used."""
    out: Dict[str, Any] = {}
    for t in tasks:
        if not t.dataset_id:
            # A task with no third-party corpus has no dataset provenance to
            # record. Keying it under "" put every such task in one bucket and
            # fed an empty name into the manifest's dataset list.
            continue
        rec = DATASET_LICENSES.get(t.dataset_id)
        if rec is None:
            out[t.dataset_id] = {"revision": None, "note": "not in licence registry"}
            continue
        out[t.dataset_id] = {
            "revision": rec.revision,
            "license": rec.license_id,
            "pinned": rec.revision is not None,
        }
    return out


async def _run(
    model: Any, tasks: List[Task], limit: Optional[int], offline: bool = False
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run the tasks.

    Returns (per-task results, dataset provenance observed, per-sample evidence).

    The evidence is what POS-5's manifest hashes — the samples behind each
    score, which the aggregate throws away. Collected here rather than
    reconstructed later, because a manifest built from anything other than the
    values actually scored would be evidence of nothing.
    """
    from .standard_benchmarks import active_suite, create_benchmark_suite

    suite = create_benchmark_suite(offline=offline)
    # Registry tasks load their data through the active suite, so this is what
    # carries --offline to them and collects the provenance read back below.
    with active_suite(suite):
        return await _run_in_suite(suite, model, tasks, limit)


async def _run_in_suite(
    suite: Any, model: Any, tasks: List[Task], limit: Optional[int]
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    from .manifest import run_task_collecting_evidence
    from .tasks import get_task

    results: Dict[str, Any] = {}
    evidence: Dict[str, Any] = {}
    # The result *objects*, not just their dicts. TaskResult.to_dict() omits
    # `uncertainty` on purpose - that dict is the byte-identical parity guard -
    # so the VOS-5 numbers are only reachable from the object, via
    # uncertainty_dict(). Serialising the dict alone is why standard_error and
    # underpowered never reached the result file.
    raw: Dict[str, Any] = {}

    for task in tasks:
        n = limit if limit is not None else task.default_limit
        err(f"  {task.task_id}: running (limit={n})...")
        try:
            # Prefer the VOS-3 registry: it is the real task implementation and
            # it exposes per-sample scores, which the legacy suite method does
            # not. Fall back to the suite for anything not yet migrated.
            try:
                registered = get_task(task.task_id)
            except Exception:
                registered = None

            if registered is not None:
                res, samples = await run_task_collecting_evidence(
                    registered, model, limit=n
                )
                evidence[task.task_id] = samples
            else:
                fn = getattr(suite, task.method)
                res = await fn(model, sample_size=n)

            results[task.task_id] = res.to_dict()
            raw[task.task_id] = res
            score = res.score
            err(
                f"  {task.task_id}: "
                + ("not_measured" if score is None else f"score={score}")
            )
        except Exception as exc:  # one bad task must not sink the run
            err(f"  {task.task_id}: FAILED — {type(exc).__name__}: {exc}")
            results[task.task_id] = {
                "benchmark_name": task.task_id,
                "score": None,
                "error": f"{type(exc).__name__}: {exc}",
                "samples_tested": 0,
                "reliability": "not_measured",
            }
    return results, suite.provenance, evidence, raw


def _default_reliability(res: Dict[str, Any]) -> str:
    """Reliability for a task that reports no ScoredMetric of its own.

    Registered tasks - an out-of-tree plugin especially - need not produce the
    legacy ScoredMetric, so there may be no reliability to copy. Defaulting to
    "not_measured" was wrong in the direction that matters: a plugin that scored
    3 samples and returned 0.0 was labelled not_measured, which is the exact
    mislabelling the rest of this work exists to remove, and it would have
    excluded a real measurement from any consumer filtering on the flag.

    A task that produced a metric measured something. A task that produced None
    did not.
    """
    return "not_measured" if res.get("score") is None else "direct"


def _normalise_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce evidence values once, before anything hashes or serialises them.

    The manifest hashes per-sample values and the result file stores them; if
    the two disagree by so much as int 1 versus float 1.0, `aethics verify`
    reports tampering on an untouched file. That is the worst failure this tool
    has, because the alarm is the whole product.

    So normalisation happens here, in one place, ahead of both consumers. A
    value that is not a number becomes 0.0 and keeps its `measured` flag - the
    flag is what carries the not-measured claim, not the placeholder.
    """
    out: Dict[str, Any] = {}
    for task_id, samples in evidence.items():
        rows = []
        for e in samples or []:
            v = e.get("value")
            rows.append(
                {
                    "sample_id": str(e.get("sample_id", "")),
                    "value": float(v) if isinstance(v, (int, float)) else 0.0,
                    "measured": bool(e.get("measured", True)),
                    **(
                        {"response_digest": e["response_digest"]}
                        if e.get("response_digest") is not None
                        else {}
                    ),
                }
            )
        out[task_id] = rows
    return out


def _task_details(res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Per-task extras, with a failed task's reason folded in.

    A task that raised gets a top-level "error" from _run and no details at
    all, so copying details alone dropped the reason entirely: the result file
    recorded a null score with no explanation and `aethics show` printed an
    empty NOTE. The reason a measurement is missing is the most useful thing in
    the file when one is.
    """
    details = dict(res.get("details") or {})
    if res.get("error") and "error" not in details:
        details["error"] = res["error"]
    return details or None


def _build_run_result(
    *,
    model_spec: str,
    selected: List[Task],
    per_task: Dict[str, Any],
    raw_results: Dict[str, Any],
    evidence: Dict[str, Any],
    provenance: Dict[str, Any],
    dataset_info: Dict[str, Any],
    run_manifest: Any,
    measured: List[float],
    started_at: Any,
) -> Any:
    """Assemble the VOS-4 ``RunResult`` that ``aethics eval`` writes.

    The CLI used to write its own envelope stamped ``schema_version 0.1.0``,
    which meant the published contract in ``schemas/result-v1.json`` had no
    producer a user could invoke: `aethics eval` output did not validate
    against it ("'run_id' is a required property"), while `aethics validate`
    reported "valid" because it was checking the other envelope. Two shapes,
    one of them documented and unreachable.

    Per-sample evidence now lives in ``tasks[].sample_scores`` rather than a
    separate top-level block, which is where the typed contract already had a
    place for it.
    """
    from .results.schema import (
        DatasetInfo,
        ModelInfo,
        RunResult,
        RunSummary,
        SampleScore,
        TaskResult,
        Uncertainty,
    )

    by_id = {t.task_id: t for t in selected}
    metas = {m.name: m for m in _registry_metas()}

    provider, _, model_id = model_spec.partition(":")

    task_results = []
    underpowered = []
    for task_id, res in per_task.items():
        meta = metas.get(task_id)
        sm = res.get("scored_metric") or {}
        obj = raw_results.get(task_id)

        # VOS-5 lives on the object, not in to_dict(). Prefer it; fall back to
        # the ScoredMetric so a task from the legacy suite still reports what
        # statistics it has.
        unc_src = obj.uncertainty_dict() if obj is not None else None
        if unc_src:
            unc = Uncertainty(**unc_src)
        else:
            ci = sm.get("confidence_interval")
            unc = Uncertainty(
                confidence_interval=tuple(ci) if ci else None,
                effect_size=sm.get("effect_size"),
                p_value=sm.get("p_value"),
                sample_size=sm.get("sample_size", 0) or 0,
                underpowered=_is_underpowered(task_id, res),
            )
        if unc.underpowered:
            underpowered.append(task_id)

        ds_id = by_id[task_id].dataset_id if task_id in by_id else ""
        ds = dataset_info.get(ds_id, {})

        # The manifest hashed these values already. Coercing them here - which
        # an earlier revision did, with float(e.get("value") or 0.0) - rewrites
        # the evidence after it was hashed: an int 1 becomes 1.0 and a
        # not-measured None becomes 0.0, both of which hash differently, so
        # `aethics verify` reported tampering on a file nobody had touched.
        # Normalisation now happens once, before the manifest is built
        # (_normalise_evidence), so there is nothing left to coerce here.
        raw_samples = evidence.get(task_id)
        samples = (
            [
                SampleScore(
                    sample_id=str(e.get("sample_id", "")),
                    value=e.get("value"),
                    measured=bool(e.get("measured", True)),
                    # Carried through deliberately. The manifest hashes this
                    # into every leaf, so dropping it here makes the recorded
                    # root uncomputable from the persisted evidence and
                    # `aethics verify` fails on a result nobody edited.
                    response_digest=e.get("response_digest"),
                )
                for e in raw_samples
            ]
            if raw_samples is not None
            else None
        )

        task_results.append(
            TaskResult(
                task_id=task_id,
                # The methodology version governs how every task scores, and is
                # what changes when scoring changes. Tasks carry no independent
                # version of their own yet.
                task_version=EVAL_MANIFEST.methodology_version,
                dataset=DatasetInfo(
                    name=ds_id or "-",
                    revision=str(ds.get("revision") or "unpinned"),
                ),
                metric=(meta.metric if meta else "score"),
                metric_value=res.get("score"),
                uncertainty=unc,
                sample_count=res.get("samples_tested", 0) or 0,
                # Tasks record their scoring method under "methodology";
                # "method" is the safety/refusal vocabulary. Checking only
                # "method" meant this always fell through to reliability, so
                # the contract's scoring-method field was never populated.
                method=(res.get("details") or {}).get("method")
                or (res.get("details") or {}).get("methodology")
                or sm.get("reliability")
                or _default_reliability(res),
                reliability=sm.get("reliability") or _default_reliability(res),
                passed=res.get("passed"),
                sample_scores=samples,
                details=_task_details(res),
            )
        )

    return RunResult(
        run_id=run_manifest.run_id,
        model=ModelInfo(
            id=model_id or model_spec,
            provider=provider or "unknown",
            params=EVAL_INFERENCE_PARAMS.to_dict(),
        ),
        manifest_fingerprint=EVAL_MANIFEST.fingerprint(),
        created_at=started_at,
        completed_at=_utc_now(),
        tasks=task_results,
        summary=RunSummary(
            tasks_run=len(per_task),
            tasks_measured=len(measured),
            # Absent rather than 0.0 when nothing was measured - a mean of no
            # measurements is not a score.
            mean_score=(round(sum(measured) / len(measured), 4) if measured else None),
            underpowered_tasks=sorted(underpowered),
        ),
        provenance=provenance,
        manifest=run_manifest.to_dict(),
    )


@app.command()
def eval(  # `eval` is the expected verb here; the shadowing is local
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="provider:model_id, e.g. hf:gpt2"
    ),
    tasks: Optional[str] = typer.Option(
        None,
        "--tasks",
        "-t",
        help=(
            "Comma-separated task ids, 'builtin' for the four shipped "
            "benchmarks (the default), or 'all' to include tasks from "
            "installed third-party packages."
        ),
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", min=1, help="Samples per task. Default: per-task."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write JSON here instead of stdout."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML/JSON run config. Flags override it."
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="Endpoint for openai: models."
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Use cached datasets only; never fetch."
    ),
    seed: int = typer.Option(
        DEFAULT_SEED, "--seed", help="Global RNG seed, recorded in the manifest."
    ),
) -> None:
    """Score a model against one or more benchmarks.

    Settings resolve command line first, then --config, then defaults, so a
    committed config describes the run and a flag can override it for one
    invocation without editing the file.
    """
    try:
        cfg = load_config(config) if config else {}

        model = model or cfg.get("model")
        tasks = tasks or cfg.get("tasks") or "builtin"
        limit = limit if limit is not None else cfg.get("limit")
        output = output or (Path(cfg["output"]) if cfg.get("output") else None)
        base_url = base_url or cfg.get("base_url")
        offline = offline or bool(cfg.get("offline", False))
        seed = seed if seed != DEFAULT_SEED else int(cfg.get("seed", DEFAULT_SEED))
        max_context = int(cfg.get("max_context", 4096))

        if not model:
            raise typer.BadParameter(
                "--model is required (or set `model:` in --config)"
            )

        selected = resolve_tasks(tasks)
        m = build_model(model, base_url=base_url, max_context=max_context)
    except typer.BadParameter as exc:
        err(f"error: {exc}")
        raise typer.Exit(EXIT_USAGE) from exc

    err(f"model : {model}")
    err(f"tasks : {', '.join(t.task_id for t in selected)}")
    if config:
        err(f"config: {config}")
    if offline:
        err("mode  : offline (cached datasets only)")
    # Seed before any task runs, and record what was actually applied rather
    # than what was asked for.
    seeds_applied = set_global_seeds(seed)
    err(f"seed  : {seed}")
    err("")

    started_at = _utc_now()
    try:
        per_task, observed, evidence, raw_results = asyncio.run(
            _run(m, selected, limit, offline=offline)
        )
        # Once, before the manifest hashes it and before the result stores it.
        evidence = _normalise_evidence(evidence)
    except KeyboardInterrupt:
        err("interrupted")
        raise typer.Exit(EXIT_FAILURE) from None

    measured = [r["score"] for r in per_task.values() if r.get("score") is not None]

    provenance = {
        "methodology_fingerprint": EVAL_MANIFEST.fingerprint(),
        "methodology_version": EVAL_MANIFEST.methodology_version,
        "inference_params": EVAL_INFERENCE_PARAMS.to_dict(),
        # What was actually loaded, falling back to the registry's declaration
        # for datasets a failed run never reached.
        "datasets": {**_dataset_provenance(selected), **observed},
        "offline": offline,
        "seeds": seeds_applied,
    }

    # POS-5: the manifest. Evidence lives beside the result rather than inside
    # it — the ticket asks for a manifest "committed alongside results", and a
    # sidecar keeps a run's per-sample evidence out of the published schema
    # until Victor decides whether it belongs there (that is a MINOR bump on
    # his contract, and his call to make).
    datasets = [
        DatasetRecord(
            name=name,
            revision=info.get("revision"),
            content_hash=info.get("content_hash"),
            row_count=info.get("row_count"),
        )
        for name, info in sorted({**_dataset_provenance(selected), **observed}.items())
    ]
    run_manifest = build_manifest(
        run_id=uuid.uuid4().hex,
        methodology_fingerprint=EVAL_MANIFEST.fingerprint(),
        library_version=__version__,
        seeds=canonical_seeds(seed),
        model={"spec": model, "params": EVAL_INFERENCE_PARAMS.to_dict()},
        datasets=datasets,
        task_samples=evidence,
        task_metrics={k: v.get("score") for k, v in per_task.items()},
    )
    run_result = _build_run_result(
        model_spec=model,
        selected=selected,
        per_task=per_task,
        raw_results=raw_results,
        evidence=evidence,
        provenance=provenance,
        dataset_info={**_dataset_provenance(selected), **observed},
        run_manifest=run_manifest,
        measured=measured,
        started_at=started_at,
    )

    text = run_result.to_json()
    if output:
        output.write_text(text + "\n", encoding="utf-8")
        err("")
        err(f"wrote {output}")
        err(f"run hash: {run_manifest.run_hash}")
    else:
        typer.echo(text)

    if not measured:
        err("")
        err("no task produced a measurement — see errors above")
        raise typer.Exit(EXIT_FAILURE)


# ── list-tasks ──────────────────────────────────────────────────────


@app.command("list-tasks")
def list_tasks(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """List available benchmarks with licence, citation and requirements."""
    from .tasks import list_tasks as _registry_list_tasks

    rows = []
    for meta in _registry_list_tasks():
        rec = DATASET_LICENSES.get(meta.dataset_id) if meta.dataset_id else None
        rows.append(
            {
                "task": meta.name,
                "dataset": meta.dataset_id or "-",
                # A task with no third-party corpus (an out-of-tree plugin, say)
                # has no dataset licence to report. Fall back to whatever the
                # task declared about itself rather than printing UNKNOWN, which
                # reads as a licensing gap when there is none.
                "license": (rec.license_id if rec else (meta.licence or "UNKNOWN")),
                "pinned_revision": (rec.revision if rec else None),
                "redistributable": (rec.redistributable if rec else None),
                "requires_logprobs": meta.requires_logprobs,
                "default_limit": meta.default_limit,
                "citation": rec.citation if rec else (meta.citation or None),
                "metric": meta.metric,
                "builtin": meta.dataset_id in DATASET_LICENSES,
            }
        )

    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return

    # Widths are computed, not fixed: an installed plugin can have a longer name
    # than any built-in, and a table that silently loses its alignment the first
    # time someone adds a task is a table that advertises plugins as an
    # afterthought.
    w_task = max(len("TASK"), *(len(r["task"]) for r in rows))
    w_lic = max(len("LICENCE"), *(len(str(r["license"])) for r in rows))

    typer.echo(
        f"{'TASK':{w_task}} {'LICENCE':{w_lic}} {'LOGPROBS':9} {'PINNED':7} DATASET"
    )
    typer.echo("-" * (w_task + w_lic + 9 + 7 + 12))
    for r in rows:
        # A task with no third-party dataset has nothing to pin, which is not
        # the same claim as a dataset that could be pinned and is not.
        pinned = (
            "-" if r["dataset"] == "-" else ("yes" if r["pinned_revision"] else "NO")
        )
        typer.echo(
            f"{r['task']:{w_task}} {r['license']!s:{w_lic}} "
            f"{'yes' if r['requires_logprobs'] else 'no':9} "
            f"{pinned:7} {r['dataset']}"
        )
    typer.echo("")
    typer.echo(f"{len(rows)} tasks. `aethics list-tasks --json` for citations.")

    unpinned = [
        r["task"] for r in rows if r["dataset"] != "-" and not r["pinned_revision"]
    ]
    if unpinned:
        typer.echo("")
        err(
            f"warning: {', '.join(unpinned)} load from an unpinned reference — "
            "two runs may read different data (POS-4)."
        )


# ── cache ───────────────────────────────────────────────────────────


@app.command()
def cache(
    clear: bool = typer.Option(False, "--clear", help="Delete every cached dataset."),
) -> None:
    """Show or clear the local dataset cache.

    ``--offline`` is only usable if you can see what is cached, so this exists
    alongside it rather than as a debugging afterthought.
    """
    from .dataset_loading import cache_root, cached_datasets, clear_cache

    if clear:
        n = clear_cache()
        typer.echo(f"removed {n} cached dataset file(s) from {cache_root()}")
        return

    entries = cached_datasets()
    typer.echo(f"cache: {cache_root()}")
    if not entries:
        typer.echo("empty — run an eval once without --offline to populate it.")
        return

    typer.echo("")
    typer.echo(f"{'ROWS':>8}  {'SIZE':>9}  {'HASH':16}  FILE")
    typer.echo("-" * 74)
    for e in entries:
        typer.echo(
            f"{e['row_count']:>8}  {e['size_bytes'] / 1024:>7.0f}KB  "
            f"{e['content_hash']:16}  {e['file']}"
        )


# ── validate ────────────────────────────────────────────────────────

_REQUIRED_PROVENANCE = ("methodology_fingerprint", "inference_params", "datasets")


@app.command()
def validate(path: Path = typer.Argument(..., help="Result JSON to check.")) -> None:
    """Check a result file is well-formed and honestly reported.

    Two layers. First the published VOS-4 JSON Schema in
    ``schemas/result-v1.json`` - the contract third-party tooling reads.
    Then the honesty checks a schema cannot express: that no score is asserted
    without samples behind it, and that a mean is not reported over nothing.

    This used to check neither. It validated the CLI's own ``0.1.0`` envelope,
    so a file that did not satisfy the published schema was reported "valid" -
    the worst kind of check, one that answers a question nobody asked in the
    voice of the question they did.
    """
    if not path.is_file():
        err(f"error: {path} not found")
        raise typer.Exit(EXIT_USAGE)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        err(f"invalid JSON: {exc}")
        raise typer.Exit(EXIT_FAILURE) from exc

    problems: List[str] = []

    # Layer 1: the published contract.
    from .results.validation import validate_result

    try:
        validate_result(data)
    except Exception as exc:
        first = str(exc).splitlines()[0]
        problems.append(f"does not satisfy schemas/result-v1.json: {first}")

    # Layer 2: honesty properties a JSON Schema cannot express.
    for key in _REQUIRED_PROVENANCE:
        if key not in (data.get("provenance") or {}):
            problems.append(f"missing provenance.{key}")

    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        problems.append("tasks must be a list of task results")
    elif not tasks:
        problems.append("tasks is empty — nothing was run")
    else:
        for res in tasks:
            if not isinstance(res, dict):
                problems.append("every entry in tasks must be an object")
                continue
            name = res.get("task_id", "?")
            if "metric_value" not in res:
                problems.append(f"tasks.{name}: no metric_value field")
            # The honesty check: a score must be backed by samples.
            if res.get("metric_value") is not None and not res.get("sample_count"):
                problems.append(
                    f"tasks.{name}: metric_value {res['metric_value']} with "
                    f"sample_count={res.get('sample_count')!r} — a score with no "
                    f"samples behind it is not a measurement"
                )
            # VOS-5: a measured metric must carry uncertainty.
            if res.get("metric_value") is not None and not res.get("uncertainty"):
                problems.append(
                    f"tasks.{name}: measured but carries no uncertainty — a point "
                    f"estimate with no error bar is not a reportable result"
                )

    summary = data.get("summary") or {}
    if summary.get("mean_score") is not None and not summary.get("tasks_measured"):
        problems.append(
            "summary: mean_score present but tasks_measured is 0 or missing"
        )

    if problems:
        err(f"{path}: {len(problems)} problem(s)")
        for p in problems:
            err(f"  - {p}")
        raise typer.Exit(EXIT_FAILURE)

    typer.echo(f"{path}: valid (schema_version {data.get('schema_version')})")


# ── show ────────────────────────────────────────────────────────────


def _format_score(score: float, res: Dict[str, Any]) -> str:
    """Render a score with its uncertainty: ``0.61 ± 0.04`` (VOS-5).

    Prefers an explicit ``uncertainty.standard_error``. Falls back to half the
    width of the 95% CI, which is the same quantity for a symmetric interval and
    is what the legacy result already carries. Bare score if neither is present —
    an absent error bar is never invented.
    """
    unc = res.get("uncertainty") or {}
    sem = unc.get("standard_error")
    if sem is None:
        ci = (res.get("scored_metric") or {}).get("confidence_interval")
        if isinstance(ci, (list, tuple)) and len(ci) == 2 and None not in ci:
            sem = abs(ci[1] - ci[0]) / 2.0
    if sem is None:
        return f"{score:.4f}"
    return f"{score:.4f} ± {sem:.2g}"


def _is_underpowered(task_name: str, res: Dict[str, Any]) -> bool:
    """Whether this task's sample count is below its min_samples threshold.

    A not_measured task is never "underpowered": there is no measurement to be
    under-powered about, and saying so would imply we scored it thinly rather
    than not at all.
    """
    if res.get("score") is None:
        return False
    unc = res.get("uncertainty") or {}
    if "underpowered" in unc:
        return bool(unc["underpowered"])
    from .scoring import is_underpowered

    return is_underpowered(task_name, res.get("samples_tested", 0) or 0)


def _evidence_from(data: Dict[str, Any]) -> Dict[str, Any]:
    """Per-sample evidence, keyed by task id.

    It lives in ``tasks[].sample_scores`` now that `aethics eval` writes the
    VOS-4 ``RunResult``. Results written by the older CLI envelope kept it in a
    top-level ``evidence`` block, and those files still verify - a verification
    tool that stops being able to check its own past output is worse than no
    tool, because the failure looks like tampering.
    """
    top = data.get("evidence")
    if top:
        return top

    out: Dict[str, Any] = {}
    tasks = data.get("tasks")
    if isinstance(tasks, list):
        for t in tasks:
            samples = t.get("sample_scores")
            if samples:
                out[t.get("task_id", "")] = samples
    return out


def _reported_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """The score each task *displays*, keyed by task id.

    The manifest carries its own copy of every score and the run hash covers
    that copy, not this one. They are written together and should never differ —
    but nothing was comparing them, so editing the number `aethics show` prints
    left every hash intact and the file verified. These are the values to hold
    the manifest against.

    Both envelopes again: `aethics eval` writes a list of tasks with
    ``metric_value``, the older CLI keyed tasks by id and called it ``score``.
    """
    tasks = data.get("tasks")

    if isinstance(tasks, dict):
        return {
            task_id: row.get("score")
            for task_id, row in tasks.items()
            if isinstance(row, dict) and "score" in row
        }

    out: Dict[str, Any] = {}
    for t in tasks or []:
        if isinstance(t, dict) and "metric_value" in t:
            out[t.get("task_id", "")] = t.get("metric_value")
    return out


@app.command()
def verify(path: Path = typer.Argument(..., help="Result JSON to verify.")) -> None:
    """Recompute a run's hashes from its evidence and confirm they match.

    ``validate`` asks whether a result is well-formed. ``verify`` asks whether
    it is the result that was actually produced — every per-sample score is
    re-hashed, every task's evidence root is rebuilt, and the run hash is
    recomputed from those roots plus the run's inputs.

    That distinction is the whole of F5. The old audit hash covered the final
    scores, so an edited sample with the totals adjusted to match would pass.
    Here it cannot: the totals are not what is hashed.

    The reported score is checked against the manifest's hashed copy as well.
    Hashing the evidence rather than the totals is what makes an edited sample
    detectable, but it also meant the number the file *displays* sat outside the
    hash — editing only that left every hash intact and the result verified.
    """
    if not path.is_file():
        err(f"error: {path} not found")
        raise typer.Exit(EXIT_USAGE)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        err(f"invalid JSON: {exc}")
        raise typer.Exit(EXIT_FAILURE) from exc

    manifest = data.get("manifest")
    if not manifest:
        err(
            f"{path}: no manifest — this result predates POS-5, or was written "
            f"by something that does not record evidence. Nothing to verify."
        )
        raise typer.Exit(EXIT_FAILURE)

    evidence = _evidence_from(data)
    if not evidence:
        err(
            f"{path}: manifest present but no evidence alongside it. The hashes "
            f"cannot be recomputed, so the manifest proves nothing."
        )
        raise typer.Exit(EXIT_FAILURE)

    report = verify_manifest(manifest, evidence, _reported_metrics(data))

    typer.echo(f"run hash recorded : {report.run_hash_expected}")
    typer.echo(f"run hash computed : {report.run_hash_computed}")
    typer.echo(f"tasks checked     : {report.tasks_checked}")
    typer.echo(f"samples checked   : {report.samples_checked}")
    typer.echo("")

    if report.ok:
        typer.echo("VERIFIED — the evidence matches the recorded hashes.")
        return

    typer.echo(f"FAILED — {len(report.problems)} problem(s):")
    for problem in report.problems:
        typer.echo(f"  - {problem}")
    raise typer.Exit(EXIT_FAILURE)


def _task_rows(data: Dict[str, Any]) -> List[Any]:
    """Task results as ``(name, legacy-shaped dict)``, from either envelope.

    `aethics eval` writes the VOS-4 ``RunResult``, where tasks are a list and
    the score is ``metric_value``. The older CLI envelope keyed tasks by id and
    called it ``score``. Normalised here so the renderer, and `_format_score` /
    `_is_underpowered` with it, stays a single implementation.
    """
    tasks = data.get("tasks")
    if isinstance(tasks, dict):
        return sorted(tasks.items())

    rows = []
    for t in tasks or []:
        rows.append(
            (
                t.get("task_id", "?"),
                {
                    "score": t.get("metric_value"),
                    "samples_tested": t.get("sample_count", 0),
                    "reliability": t.get("reliability", ""),
                    "uncertainty": t.get("uncertainty") or {},
                    "error": (t.get("details") or {}).get("error", ""),
                },
            )
        )
    return sorted(rows)


@app.command()
def show(path: Path = typer.Argument(..., help="Result JSON to display.")) -> None:
    """Print a result file as a human-readable table."""
    if not path.is_file():
        err(f"error: {path} not found")
        raise typer.Exit(EXIT_USAGE)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        err(f"invalid JSON: {exc}")
        raise typer.Exit(EXIT_FAILURE) from exc

    prov = data.get("provenance") or {}
    model = data.get("model")
    # `model` is a ModelInfo object in the VOS-4 result and was a bare string in
    # the older CLI envelope.
    if isinstance(model, dict):
        model = f"{model.get('provider', '?')}:{model.get('id', '?')}"
    typer.echo(f"model        : {model or '?'}")
    typer.echo(f"methodology  : {prov.get('methodology_version', '?')}")
    fp = prov.get("methodology_fingerprint") or data.get("manifest_fingerprint")
    typer.echo(f"fingerprint  : {fp[:16] + '...' if fp else '?'}")
    typer.echo("")

    rows = _task_rows(data)
    w = max([len("TASK"), *(len(n) for n, _ in rows)]) if rows else len("TASK")

    typer.echo(f"{'TASK':{w}} {'SCORE':>16}  {'SAMPLES':>7}  NOTE")
    typer.echo("-" * (w + 54))
    for name, res in rows:
        score = res.get("score")
        shown = "not_measured" if score is None else _format_score(score, res)
        note = res.get("error", "") or res.get("reliability", "")
        if _is_underpowered(name, res):
            note = f"underpowered; {note}" if note else "underpowered"
        typer.echo(
            f"{name:{w}} {shown:>16}  {res.get('samples_tested', 0):>7}  {note[:26]}"
        )

    s = data.get("summary") or {}
    mean = s.get("mean_score")
    typer.echo("-" * (w + 54))
    typer.echo(
        f"{'mean':{w}} {('n/a' if mean is None else f'{mean:.4f}'):>16}  "
        f"({s.get('tasks_measured', 0)}/{s.get('tasks_run', 0)} measured)"
    )

    if mean is None:
        typer.echo("")
        typer.echo("No task produced a measurement. This is not a score of zero.")


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except Exception as exc:  # last resort: a traceback is not a UI
        err(f"unexpected error: {type(exc).__name__}: {exc}")
        sys.exit(EXIT_FAILURE)


if __name__ == "__main__":
    main()
