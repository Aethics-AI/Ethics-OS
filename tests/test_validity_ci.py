# SPDX-FileCopyrightText: Copyright (c) 2026 AETHICS
# SPDX-License-Identifier: Apache-2.0

"""
VOS-6 Part D — the nightly job is wired correctly.

Cheap structural checks on the workflow. They exist because the failure mode is
silent: a validity job that never runs, or runs on PRs and gets disabled for
being slow, leaves us with a methodology document nothing verifies.
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# The validity job used to live in its own validity-nightly.yml. It was folded
# into nightly.yml, which is where the rest of the slow lane already ran - the
# two files shared a 03:00 cron and only one of them pinned its actions.
_WORKFLOW = (
    Path(__file__).resolve().parent.parent / ".github" / "workflows" / "nightly.yml"
)

# `.github/` does not travel with the published subtree, so absence there is
# legitimate. Absence in a full checkout is not, and must not quietly skip:
# these assertions exist precisely because a validity job that stops running
# fails silently.
_IN_FULL_CHECKOUT = (
    Path(__file__).resolve().parent.parent / ".github" / "workflows"
).is_dir()


@pytest.fixture(scope="module")
def workflow():
    if not _IN_FULL_CHECKOUT:
        pytest.skip("no .github/workflows here (published subtree)")
    assert _WORKFLOW.exists(), (
        f"{_WORKFLOW.name} is missing from a checkout that has .github/workflows. "
        "The validity job must live in a workflow this test can see - if it moved "
        "again, point _WORKFLOW at its new home rather than letting these skip."
    )
    return yaml.safe_load(_WORKFLOW.read_text())


def _triggers(wf):
    # PyYAML parses a bare `on:` key as the boolean True.
    return wf.get("on") or wf.get(True) or {}


def test_runs_nightly_and_not_on_pull_requests(workflow):
    trig = _triggers(workflow)
    assert "schedule" in trig, "validity must run on a schedule"
    assert "pull_request" not in trig, (
        "validity downloads a model and does real inference — it must not run on PRs"
    )


def test_can_be_triggered_manually(workflow):
    # Needed when methodology.md changes and you want the table refreshed now.
    assert "workflow_dispatch" in _triggers(workflow)


def test_job_runs_the_validity_marker_and_the_drift_check(workflow):
    steps = workflow["jobs"]["validity"]["steps"]
    body = "\n".join(s.get("run", "") for s in steps)
    assert "-m validity" in body, "the job must actually run the validity tests"
    assert "generate_methodology" in body and "git diff" in body, (
        "the job must fail if docs/methodology.md has drifted from a real run"
    )


def test_installs_the_local_extra(workflow):
    # torch/transformers live in the [local] extra; without it the reference
    # model cannot load and every validity test would skip rather than fail.
    body = "\n".join(s.get("run", "") for s in workflow["jobs"]["validity"]["steps"])
    assert "local" in body and "pip install" in body
