# Writing a task

A benchmark in `aethics-eval` is a class with four methods. You can ship one
from your own package without touching this repository — install it and the
registry finds it.

This page builds a complete working task. The finished version lives in
[`examples/example-task/`](https://github.com/Aethics-AI/Ethics-OS/tree/main/examples/example-task)
and is installable as-is.

!!! warning "The CLI does not see plugin tasks yet"

    Out-of-tree tasks register correctly and run through the **library**, but
    `aethics list-tasks` and `aethics eval` still read a hardcoded table of the
    four built-in benchmarks, so your task will not appear there and
    `--tasks your_task` returns *unknown task*.

    Run yours via the Python API (shown at the end) until that is fixed. This is
    a known gap, not something you have done wrong.

## The shape of a task

Four methods, each with one job:

| Method | Answers |
|---|---|
| `load(limit)` | What are the samples? |
| `build_request(sample)` | What do we ask the model? |
| `score(sample, response)` | How did it do on this one? |
| `aggregate(scores)` | What is the headline number? |

The runner drives them. You never call the model yourself, which is what lets
the same task run against a local model, a hosted API or a stub.

## 1. Register the class

```python
from aethics_eval.tasks import (
    Request,
    Response,
    Sample,
    SampleScore,
    Task,
    TaskResult,
    register_task,
)


@register_task(
    "example_yes_rate",
    metric="yes_fraction",
    requires_logprobs=False,
    citation="AETHICS example package",
    licence="MIT",
    description="Fraction of yes/no prompts the model answers 'yes'.",
)
class YesRateTask(Task):
    """Toy generation task: how often does the model answer 'yes'?"""
```

The metadata is not decoration. `requires_logprobs` decides whether a provider
can run your task at all — say `True` and a model without log-probabilities
will report `not_measured` instead of quietly scoring badly. `licence` and
`citation` are printed by `list-tasks`, because a benchmark whose licence
nobody can find does not get used in an audit.

## 2. `load` — produce samples

```python
class YesRateTask(Task):  # continued
    _PROMPTS = [
        "Answer yes or no: is water wet?",
        "Answer yes or no: is the sky green?",
        "Answer yes or no: do fish swim?",
    ]

    def load(self, limit=None):
        prompts = self._PROMPTS if limit is None else self._PROMPTS[:limit]
        return [Sample(id=str(i), data={"prompt": p}) for i, p in enumerate(prompts)]
```

`id` must be stable across runs — it goes into the evidence hash, so a sample
that changes identity between runs breaks `aethics verify`. Use the dataset's
own identifier where there is one, not the enumeration index of a shuffled list.

Honour `limit`. Users pass `--limit` expecting it to bound the work.

If you load from a remote dataset, use
[`load_dataset_pinned`](https://github.com/Aethics-AI/Ethics-OS/blob/main/src/aethics_eval/dataset_loading.py)
so your data is fetched at a fixed revision and cached, rather than silently
changing when upstream does.

## 3. `build_request` — say what to ask

```python
class YesRateTask(Task):  # continued
    def build_request(self, sample):
        return Request(method="generate", args=(sample.data["prompt"],))
```

`method` is `"generate"` for text, or `"conditional_logprob"` /
`"pair_stereotype_logprobs"` for likelihood scoring. Declaring the call instead
of making it is what lets the runner batch, retry and record it.

## 4. `score` — judge one sample

```python
class YesRateTask(Task):  # continued
    def score(self, sample, response):
        text = getattr(response.value, "text", "") if response.success else ""
        measured = response.success and getattr(response.value, "success", True)
        said_yes = "yes" in (text or "").lower()
        return SampleScore(
            sample.id,
            1.0 if said_yes else 0.0,
            measured=bool(measured),
            detail={"prompt": sample.data["prompt"], "response": text},
        )
```

**`measured` is the important argument.** Set it `False` when the model did not
answer — a timeout, a refusal to respond, a transport error. A failed call is
not a score of zero, and this flag is what keeps that distinction alive all the
way to the result file.

The temptation is to return `0.0` and move on. Don't. That is how an evaluation
tool ends up reporting a confident number about a model it never reached.

## 5. `aggregate` — the headline number

```python
class YesRateTask(Task):  # continued
    def aggregate(self, scores):
        measured = [s for s in scores if s.measured]
        if not measured:
            return TaskResult(
                task="example_yes_rate",
                score=None,
                samples_tested=0,
                passed=False,
                details={"measured": False, "reliability": "not_measured"},
            )
        rate = sum(s.value for s in measured) / len(measured)
        return TaskResult(
            task="example_yes_rate",
            score=round(rate, 4),
            samples_tested=len(measured),
            passed=rate >= 0.5,
            threshold=0.5,
            details={"yes_rate": round(rate, 4), "n": len(measured)},
        )
```

Two rules that the built-in benchmarks all follow:

- **Aggregate only over `measured` scores**, and report `samples_tested` as
  that count — not the number requested. Otherwise a task where nine of ten
  calls failed reports a tenth of the evidence as if it were all of it.
- **If nothing was measured, `score` is `None`.** Never `0.0`. Downstream code
  excludes `None` from means; it cannot rescue a fabricated zero.

Report uncertainty in `details` if you can — a bare point estimate invites
over-reading, as the [quickstart](quickstart.md) demonstrates with a 50-sample
run that pointed the wrong way.

## 6. Ship it

Declare an entry point so installing the package registers the task:

```toml
# pyproject.toml
[project]
name = "aethics-eval-example-task"
dependencies = ["aethics-eval"]

[project.entry-points."aethics_eval.tasks"]
example_yes_rate = "aethics_eval_example"
```

The value points at a module whose import runs `@register_task` — usually your
`__init__.py`. No changes to `aethics_eval` are needed.

```bash
pip install .
```

## 7. Check it registered

```console
$ python -c "from aethics_eval.tasks import task_names; print(sorted(task_names()))"
['bold', 'crows_pairs', 'example_yes_rate', 'stereoset', 'winobias']
```

And run it:

```console
$ python -c "
import asyncio
from aethics_eval.tasks import get_task, run_task
from aethics_eval.models import FakeModel
res = asyncio.run(run_task(get_task('example_yes_rate'), FakeModel(default_response='yes'), limit=3))
print(res.task, res.score, res.samples_tested)
"
example_yes_rate 1.0 3
```

That is real output from the example package. As noted at the top, the CLI will
not list or run it yet.

## Testing your task

Use `FakeModel` — scriptable, no network:

```python
from aethics_eval.models import FakeModel

FakeModel(default_response="yes")  # same answer every time
FakeModel(responses={"prompt text": "no"})  # per-prompt answers
FakeModel(logprobs={"a sentence": -12.0})  # drives the likelihood path
FakeModel()  # no logprob support
```

Passing `logprobs` flips `capabilities.supports_logprobs` on, so it is how you
exercise a `requires_logprobs=True` task; a bare `FakeModel()` is how you
exercise the `not_measured` branch.

Worth covering explicitly:

- the happy path scores as expected;
- a model that fails every call yields `score is None`, not `0.0`;
- if `requires_logprobs=True`, a plain `FakeModel()` gives `not_measured`;
- `load(limit=n)` returns at most `n` samples.

That third one is easy to skip and is exactly the property that stops your task
reporting a number it has not earned.

## See also

- [Model providers](model-providers.md) — what a `Model` must implement
- [Result schema](result-schema.md) — where your `details` end up
- [Datasets](datasets.md) — licensing before you add a dataset
