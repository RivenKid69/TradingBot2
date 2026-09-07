# -*- coding: utf-8 -*-
from __future__ import annotations

import zipfile
from pathlib import Path

from ccea.guardrails.build_artifact_check import verify_cloud_artifact
from tools.build_zone_distributions import build_wheel_like_artifact


def test_cloud_distribution_excludes_guardrails_and_deprecated_agent_shims(tmp_path):
    root = tmp_path / "repo"
    (root / "packages/cloud").mkdir(parents=True)
    (root / "packages/shared").mkdir(parents=True)
    (root / "packages/agent").mkdir(parents=True)
    (root / "ccea/artifact").mkdir(parents=True)
    (root / "ccea/crypto").mkdir(parents=True)
    (root / "ccea/models").mkdir(parents=True)
    (root / "ccea/contracts").mkdir(parents=True)
    (root / "ccea/protocol").mkdir(parents=True)
    (root / "ccea/telemetry").mkdir(parents=True)

    # These must NOT be present in cloud artifact (would trip artifact scan / boundary)
    (root / "ccea/guardrails").mkdir(parents=True)
    (root / "ccea/agent").mkdir(parents=True)

    (root / "packages/cloud/app.py").write_text("x = 1\n", encoding="utf-8")
    (root / "packages/shared/contracts.py").write_text("y = 2\n", encoding="utf-8")
    (root / "ccea/guardrails/build_artifact_check.py").write_text(
        'PROHIBITED = "LiveExecutionEngine"\n', encoding="utf-8"
    )
    (root / "ccea/agent/__init__.py").write_text(
        "from packages.agent.daemon.agentd import AgentDaemon\n", encoding="utf-8"
    )

    out = tmp_path / "dist" / "ccea_cloud-1.0.0-py3-none-any.whl"
    build_wheel_like_artifact(
        root=root,
        output_file=out,
        dist_name="ccea_cloud",
        version="1.0.0",
        include_paths=[
            "packages/cloud",
            "packages/shared",
            "ccea/artifact",
            "ccea/crypto",
            "ccea/models",
            "ccea/contracts",
            "ccea/protocol",
            "ccea/telemetry",
        ],
    )

    with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())
        assert not any(n.startswith("ccea/guardrails/") for n in names)
        assert not any(n.startswith("ccea/agent/") for n in names)

    scan = verify_cloud_artifact(out)
    assert scan.passed is True, [str(v) for v in scan.violations]
