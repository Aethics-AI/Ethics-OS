# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
Write the published VOS-4 result JSON Schema to schemas/result-v1.json.

    python -m scripts.export_schema

Run this whenever the RunResult model changes. The CI drift test fails if the
committed file is out of sync with the model.
"""

from __future__ import annotations

from pathlib import Path

from aethics_eval.results.validation import schema_json

OUT = Path(__file__).resolve().parent.parent / "schemas" / "result-v1.json"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(schema_json())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
