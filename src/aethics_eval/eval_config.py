# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation Configuration & Deterministic Inference Settings

Centralizes all evaluation parameters to ensure reproducibility.
Every audit stores the exact config used, so results can be replicated.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

EVAL_METHODOLOGY_VERSION = "3.0.0"
PROMPT_SET_VERSION = "2.0.0"
SCORING_ALGORITHM_VERSION = "2.0.0"


@dataclass(frozen=True)
class InferenceParams:
    """Immutable inference parameters for deterministic evaluation.

    Using temperature=0 and do_sample=False ensures that re-running
    the same audit on the same model produces identical outputs.

    >>> from aethics_eval import EVAL_INFERENCE_PARAMS
    >>> EVAL_INFERENCE_PARAMS.temperature
    0.0
    >>> EVAL_INFERENCE_PARAMS.do_sample
    False

    Frozen, so a caller cannot quietly change the terms of an evaluation:

    >>> EVAL_INFERENCE_PARAMS.temperature = 0.7
    Traceback (most recent call last):
        ...
    dataclasses.FrozenInstanceError: cannot assign to field 'temperature'

    That immutability is the point. These values are hashed into the
    methodology fingerprint, so a run that sampled at temperature 0.7 could
    otherwise claim the fingerprint of a deterministic one.

    >>> sorted(EVAL_INFERENCE_PARAMS.to_dict())[:3]
    ['do_sample', 'max_new_tokens', 'repetition_penalty']
    """

    temperature: float = 0.0
    do_sample: bool = False
    max_new_tokens: int = 100
    top_p: float = 1.0
    top_k: int = 0
    return_full_text: bool = False
    # 1.0 = no penalty, so we measure the model's natural output. Referenced by
    # hf_client (EVAL_INFERENCE_PARAMS.repetition_penalty); its absence here made
    # every evaluation inference call raise AttributeError, which is what left all
    # audits with zero responses and default scores (Finding 3).
    repetition_penalty: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# The single source of truth for evaluation inference.
EVAL_INFERENCE_PARAMS = InferenceParams()


@dataclass
class EvalMethodologyManifest:
    """Immutable manifest stamped onto every audit for traceability.

    Records *how* an evaluation was performed — methodology version, prompt set,
    scoring algorithm and inference parameters — and hashes the lot into a
    fingerprint. Two results carrying the same fingerprint were produced by the
    same method, which is what makes them comparable.

    >>> from aethics_eval import EVAL_MANIFEST
    >>> EVAL_MANIFEST.methodology_version
    '3.0.0'
    >>> EVAL_MANIFEST.fingerprint()[:16]
    '266b57eaa394ea0b'

    The fingerprint is deterministic, and changes if any component changes:

    >>> EVAL_MANIFEST.fingerprint() == EvalMethodologyManifest().fingerprint()
    True
    >>> changed = EvalMethodologyManifest(methodology_version="3.0.1")
    >>> changed.fingerprint() == EVAL_MANIFEST.fingerprint()
    False

    It says nothing about *what* was measured — same method, different model or
    dataset, same fingerprint. For that, use the per-run manifest in
    :mod:`aethics_eval.manifest`, whose ``run_hash`` covers the evidence.

    ``min_samples`` records the sample count each benchmark is expected to
    reach, and drives the ``underpowered`` flag on results:

    >>> EVAL_MANIFEST.min_samples["crows_pairs"]
    30

    .. note::
       Treat these as smoke-test floors rather than a guarantee of statistical
       power. Measured on GPT-2, CrowS-Pairs at 30 pairs gives a 95% interval of
       roughly [0.50, 0.83] on the preference rate — wide enough to be
       consistent with no preference at all. The full 1,508 pairs narrow it to
       about [0.554, 0.603]. Clearing this bar is not the same as having a
       publishable number.
    """

    methodology_version: str = EVAL_METHODOLOGY_VERSION
    prompt_set_version: str = PROMPT_SET_VERSION
    scoring_algorithm_version: str = SCORING_ALGORITHM_VERSION
    inference_params: Dict[str, Any] = field(
        default_factory=lambda: EVAL_INFERENCE_PARAMS.to_dict()
    )

    # Minimum sample sizes per benchmark to ensure statistical power
    min_samples: Dict[str, int] = field(
        default_factory=lambda: {
            "winobias": 50,
            "stereoset": 50,
            "crows_pairs": 30,
            "bold": 30,
            "gender_bias": 30,
            "occupational_bias": 18,
            "demographic_bias": 32,
            "completion_bias": 10,
            "toxicity": 10,
            "harmful_content": 8,
            "privacy": 8,
            "safety_refusal": 8,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Deterministic SHA-256 hash of the full manifest."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


# Singleton manifest — import this wherever you need it
EVAL_MANIFEST = EvalMethodologyManifest()
