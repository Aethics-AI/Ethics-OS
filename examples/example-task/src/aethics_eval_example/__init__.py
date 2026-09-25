"""
Demo out-of-tree task package (VOS-3 Part D).

This is what a third party ships: a pip-installable package with a Task that
registers itself via the ``aethics_eval.tasks`` entry point (declared in this
package's pyproject.toml). Once installed, ``aethics list-tasks`` and
``list_tasks()`` show it automatically — no changes to aethics_eval's source.

Importing this module registers the task (the @register_task decorator runs), so
the entry point simply points here.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

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
    description="Demo out-of-tree task: fraction of yes/no prompts the model answers 'yes'.",
)
class YesRateTask(Task):
    """Toy generation task: how often does the model answer 'yes'?"""

    _PROMPTS: ClassVar[list[str]] = [
        "Answer yes or no: is water wet?",
        "Answer yes or no: is the sky green?",
        "Answer yes or no: do fish swim?",
    ]

    def __init__(self, prompts: Optional[List[str]] = None):
        self._prompts = prompts if prompts is not None else self._PROMPTS

    def load(self, limit: Optional[int] = None) -> List[Sample]:
        prompts = self._prompts if limit is None else self._prompts[:limit]
        return [Sample(id=str(i), data={"prompt": p}) for i, p in enumerate(prompts)]

    def build_request(self, sample: Sample) -> Request:
        return Request(method="generate", args=(sample.data["prompt"],))

    def score(self, sample: Sample, response: Response) -> SampleScore:
        text = getattr(response.value, "text", "") if response.success else ""
        measured = response.success and getattr(response.value, "success", True)
        said_yes = "yes" in (text or "").lower()
        return SampleScore(
            sample.id,
            1.0 if said_yes else 0.0,
            measured=bool(measured),
            detail={"prompt": sample.data["prompt"], "response": text},
        )

    def aggregate(self, scores: List[SampleScore]) -> TaskResult:
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
