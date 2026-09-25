# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""aethics-eval — reproducible bias and safety evaluation for language models.

The top-level surface is the configuration, provenance and statistics
primitives. Tasks, results and aggregation live in their own subpackages:
``aethics_eval.tasks``, ``aethics_eval.results`` and ``aethics_eval.aggregation``.

Everything below is import-safe: pure standard library at module level, with
datasets/torch/transformers imported lazily at the point of use.
"""

from .eval_config import (
    EVAL_INFERENCE_PARAMS,
    EVAL_MANIFEST,
    EVAL_METHODOLOGY_VERSION,
    EvalMethodologyManifest,
    InferenceParams,
)
from .scoring import (
    ScoredMetric,
    bias_to_canonical,
    bootstrap_confidence_interval,
    canonical_to_zero_one,
    cohens_d,
    hash_audit_results,
    permutation_test,
    zero_one_to_canonical,
)

__version__ = "0.1.0"

__all__ = [
    # configuration & provenance
    "EVAL_INFERENCE_PARAMS",
    "EVAL_MANIFEST",
    "EVAL_METHODOLOGY_VERSION",
    "EvalMethodologyManifest",
    "InferenceParams",
    # scoring primitives
    "ScoredMetric",
    "__version__",
    "bias_to_canonical",
    "bootstrap_confidence_interval",
    "canonical_to_zero_one",
    "cohens_d",
    "hash_audit_results",
    "permutation_test",
    "zero_one_to_canonical",
]
