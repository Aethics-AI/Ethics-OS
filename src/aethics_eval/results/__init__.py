# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Versioned result schema (VOS-4).

Public API:
    RunResult, TaskResult, SampleScore, Uncertainty, DatasetInfo, ModelInfo
    SCHEMA_VERSION, utc_now, current_manifest_fingerprint
"""

from .loader import IncompatibleSchemaVersion, load_result, load_result_file
from .schema import (
    SCHEMA_VERSION,
    DatasetInfo,
    ModelInfo,
    RunResult,
    RunSummary,
    SampleScore,
    TaskResult,
    Uncertainty,
    utc_now,
)
from .uncertainty import (
    DEFAULT_RESAMPLES,
    build_uncertainty,
    format_with_uncertainty,
)
from .validation import result_schema, schema_json, validate_result


def current_manifest_fingerprint() -> str:
    """The current methodology-manifest fingerprint (EVAL_MANIFEST.fingerprint()),
    stamped onto a RunResult for reproducibility/traceability.

    >>> current_manifest_fingerprint()[:16]
    '266b57eaa394ea0b'

    Two results sharing this value were produced by the same methodology, which
    is what makes them comparable. It says nothing about *what* was measured —
    same method on a different model or dataset gives the same fingerprint.

    >>> from aethics_eval import EVAL_MANIFEST
    >>> current_manifest_fingerprint() == EVAL_MANIFEST.fingerprint()
    True
    """
    from aethics_eval.eval_config import EVAL_MANIFEST

    return EVAL_MANIFEST.fingerprint()


__all__ = [
    "DEFAULT_RESAMPLES",
    "SCHEMA_VERSION",
    "DatasetInfo",
    "IncompatibleSchemaVersion",
    "ModelInfo",
    "RunResult",
    "RunSummary",
    "SampleScore",
    "TaskResult",
    "Uncertainty",
    "build_uncertainty",
    "current_manifest_fingerprint",
    "format_with_uncertainty",
    "load_result",
    "load_result_file",
    "result_schema",
    "schema_json",
    "utc_now",
    "validate_result",
]
