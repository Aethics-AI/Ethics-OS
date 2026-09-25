# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-4 — load a serialised result back into typed objects.

``load_result`` is the documented reader downstream tooling uses. It checks the
result's ``schema_version`` against what this reader supports (a newer MAJOR is
rejected; a newer MINOR is fine — unknown fields are ignored), optionally
validates against the published JSON Schema, and returns a typed ``RunResult``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

from .schema import SCHEMA_VERSION, RunResult
from .validation import validate_result

_SUPPORTED_MAJOR = int(SCHEMA_VERSION.split(".")[0])


class IncompatibleSchemaVersion(ValueError):
    """A result's schema_version major is newer than this reader supports.

    Raised rather than parsing a future result halfway — a v1 reader meeting a
    2.x result must fail loudly, never silently misinterpret it.

    >>> issubclass(IncompatibleSchemaVersion, ValueError)
    True
    """


def load_result(
    data: Union[str, bytes, Dict[str, Any]], *, validate: bool = True
) -> RunResult:
    """Parse a serialised result into a typed ``RunResult``.

    Args:
        data: a dict, or a JSON string/bytes.
        validate: also check the payload against the published JSON Schema.

    Raises:
        IncompatibleSchemaVersion: the result's major version is newer than
            this reader (e.g. a 2.x result read by a 1.x reader).
        jsonschema.ValidationError: (when ``validate``) the payload is malformed.

    >>> payload = {
    ...     "schema_version": "1.1.0",
    ...     "run_id": "abc123",
    ...     "model": {"id": "gpt2", "provider": "local"},
    ...     "manifest_fingerprint": "266b57eaa394ea0b",
    ... }
    >>> result = load_result(payload)
    >>> result.run_id
    'abc123'

    A newer MINOR is accepted — that is what makes additive changes safe:

    >>> load_result({**payload, "schema_version": "1.7.0"}).run_id
    'abc123'

    A newer MAJOR is refused outright:

    >>> try:
    ...     load_result({**payload, "schema_version": "2.0.0"})
    ... except IncompatibleSchemaVersion as exc:
    ...     print("not readable by this" in str(exc))
    True
    """
    if isinstance(data, (str, bytes)):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise TypeError(f"expected a dict or JSON string, got {type(data).__name__}")

    version = str(data.get("schema_version", ""))
    major = version.split(".")[0]
    if not major.isdigit() or int(major) != _SUPPORTED_MAJOR:
        raise IncompatibleSchemaVersion(
            f"result schema_version {version!r} is not readable by this "
            f"loader (supports {_SUPPORTED_MAJOR}.x)"
        )

    if validate:
        validate_result(data)
    return RunResult.model_validate(data)


def load_result_file(path: Union[str, Path], *, validate: bool = True) -> RunResult:
    """Load a result from a JSON file into a typed ``RunResult``.

    Same version policy as :func:`load_result`; this only adds reading the file.

    >>> import json, tempfile, pathlib
    >>> payload = {
    ...     "schema_version": "1.1.0",
    ...     "run_id": "abc123",
    ...     "model": {"id": "gpt2", "provider": "local"},
    ...     "manifest_fingerprint": "266b57eaa394ea0b",
    ... }
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     path = pathlib.Path(tmp) / "result.json"
    ...     _ = path.write_text(json.dumps(payload))
    ...     load_result_file(path).run_id
    'abc123'
    """
    return load_result(Path(path).read_text(), validate=validate)
