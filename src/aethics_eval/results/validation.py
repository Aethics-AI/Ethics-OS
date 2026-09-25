# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-4 — JSON Schema export and result validation.

The published contract lives at ``schemas/result-v1.json`` (generated from the
``RunResult`` model via ``scripts/export_schema.py``). ``validate_result``
checks a payload against it; a CI test keeps the committed file in sync with the
model (no drift).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Union

from .schema import RunResult


def _without_doctests(text: str) -> str:
    """Take the prose above a docstring's first `>>>` example.

    Pydantic copies a model's docstring into the schema's ``description``, so
    POS-7's requirement (a worked example on every public symbol) would
    otherwise paste Python doctests into a JSON Schema read by non-Python
    tooling — measured at +46% file size — and make the committed contract
    churn on every documentation edit.

    Truncation, deliberately, rather than excising examples from the middle:
    reassembling prose that surrounds an example cannot be done reliably and
    produced sentences cut in half when tried. **The models in schema.py
    therefore put all prose before all examples**, which makes this exact. A
    docstring that interleaves them will lose the trailing prose from the
    schema — visibly, in the committed file, where the drift test surfaces it.

    >>> _without_doctests("Prose.\\n\\nExample:\\n\\n>>> 1 + 1\\n2\\n")
    'Prose.'
    >>> _without_doctests("Just prose.")
    'Just prose.'
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith(">>>"):
            lines = lines[:i]
            break

    # Drop the lead-in that introduced the example ("Example:", "Usage:"),
    # which reads as a broken promise once the example is gone.
    while lines and (not lines[-1].strip() or lines[-1].rstrip().endswith(":")):
        lines.pop()
    return "\n".join(lines).strip()


def _strip_doctests(node: Any) -> Any:
    """Recursively rewrite every ``description`` in a JSON Schema fragment."""
    if isinstance(node, dict):
        return {
            k: (
                _without_doctests(v)
                if k == "description" and isinstance(v, str)
                else _strip_doctests(v)
            )
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_strip_doctests(item) for item in node]
    return node


def result_schema() -> Dict[str, Any]:
    """The JSON Schema for a RunResult (this is what schemas/result-v1.json holds).

    >>> schema = result_schema()
    >>> schema["title"]
    'RunResult'
    >>> sorted(schema["required"])
    ['manifest_fingerprint', 'model', 'run_id']

    Descriptions carry the models' prose but not their doctest examples — see
    :func:`_without_doctests`:

    >>> ">>>" in json.dumps(schema)
    False
    """
    return _strip_doctests(RunResult.model_json_schema())


def schema_json(indent: int = 2) -> str:
    """The schema serialised with stable key order — the exact bytes committed to
    schemas/result-v1.json, so the drift check is a plain string compare.

    >>> text = schema_json()
    >>> text.endswith(chr(10))         # trailing newline, as committed
    True
    >>> text == schema_json()          # stable across calls
    True
    """
    return json.dumps(result_schema(), indent=indent, sort_keys=True) + "\n"


def validate_result(data: Union[str, bytes, Dict[str, Any]]) -> None:
    """Validate a result payload against the published schema.

    Accepts a dict or a JSON string/bytes. Raises ``jsonschema.ValidationError``
    if the payload does not conform (extra fields are allowed — additive/minor
    forward compatibility).

    >>> payload = {
    ...     "run_id": "abc123",
    ...     "model": {"id": "gpt2", "provider": "local"},
    ...     "manifest_fingerprint": "266b57eaa394ea0b",
    ... }
    >>> validate_result(payload) is None
    True

    A missing required field is rejected:

    >>> import jsonschema
    >>> try:
    ...     validate_result({"model": {"id": "gpt2", "provider": "local"}})
    ... except jsonschema.ValidationError as exc:
    ...     print("run_id" in str(exc))
    True

    .. note::
       This validates against the published VOS-4 schema. The ``aethics
       validate`` CLI command checks the CLI's own envelope instead, which is a
       different and currently divergent shape.
    """
    import jsonschema

    if isinstance(data, (str, bytes)):
        data = json.loads(data)
    jsonschema.validate(instance=data, schema=result_schema())
