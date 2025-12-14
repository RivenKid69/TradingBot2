# -*- coding: utf-8 -*-
from __future__ import annotations

import json

from ccea.crypto.keys import KeyAlgorithm, generate_keypair
from packages.agent.daemon.preflight import PreflightChecker, PreflightConfig, PreflightCheckType, PreflightCheckResult
from packages.cloud.builder.artifact_builder import ArtifactBuilder, BuildConfig
from packages.shared.contracts.manifest import ArtifactFormat


def test_preflight_requires_signature_when_artifact_present(tmp_path):
    artifact_path = tmp_path / "artifact.zip"
    artifact_path.write_bytes(b"artifact")

    checker = PreflightChecker(
        config=PreflightConfig(
            skip_broker_check=True,
            skip_time_sync=True,
            skip_network_check=True,
            require_vault_unlocked=False,
        )
    )

    result = checker.run_preflight(
        artifact_path=artifact_path,
        manifest={"schema_version": "1.0.0", "entrypoint": "x.py"},
        signature=None,
    )

    sig_check = next(c for c in result.checks if c.check_type == PreflightCheckType.SIGNATURE_VERIFICATION)
    assert sig_check.result == PreflightCheckResult.FAILED
    assert sig_check.required is True


def test_preflight_accepts_manifest_embedded_signature(tmp_path):
    signing_key = generate_keypair(KeyAlgorithm.ED25519, key_id="test-key")
    builder = ArtifactBuilder(signing_key=signing_key)

    src = tmp_path / "src"
    src.mkdir()
    (src / "strategy.py").write_text("class Strategy: pass\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    build = builder.build(
        BuildConfig(
            strategy_id="s1",
            strategy_name="Strategy 1",
            version="1.0.0",
            entrypoint="strategy:Strategy",
            source_path=src,
            output_format=ArtifactFormat.ZIP_BUNDLE,
            output_path=out_dir,
        )
    )
    assert build.success is True

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    checker = PreflightChecker(
        config=PreflightConfig(
            skip_broker_check=True,
            skip_time_sync=True,
            skip_network_check=True,
            require_vault_unlocked=False,
        )
    )
    result = checker.run_preflight(
        artifact_path=out_dir / "artifact.zip",
        artifact_digest=manifest["artifact_digest"],
        manifest=manifest,
        signature=None,
    )

    sig_check = next(c for c in result.checks if c.check_type == PreflightCheckType.SIGNATURE_VERIFICATION)
    assert sig_check.result == PreflightCheckResult.PASSED

    digest_check = next(c for c in result.checks if c.check_type == PreflightCheckType.DIGEST_VERIFICATION)
    assert digest_check.result == PreflightCheckResult.PASSED

