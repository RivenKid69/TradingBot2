# -*- coding: utf-8 -*-
"""Ed25519 model-signature enforcement on the agent daemon path (P0-E).

Locks in that the daemon's artifact-activation path verifies model checkpoints
through the SAME signature gate the RL inference loader uses — closing the
"agentd load paths bypass the gate" half of §4.7. Fail-closed for LIVE.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from service_experiment_tracking import ModelRegistry
from services.model_signature_gate import ModelSignatureError
from packages.agent.daemon.model_gate import find_model_files, verify_artifact_models


# ------------------------------------------------------------------ fixtures


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(root=str(tmp_path / "registry"))


def _artifact_with_model(tmp_path, name="artifact", content=b"dummy sb3 checkpoint"):
    art_dir = tmp_path / name
    art_dir.mkdir()
    model = art_dir / "policy.zip"
    model.write_bytes(content)
    return art_dir, model


# ------------------------------------------------------------------ find_model_files


def test_find_model_files_picks_checkpoints(tmp_path):
    (tmp_path / "policy.zip").write_bytes(b"x")
    (tmp_path / "weights.pt").write_bytes(b"x")
    (tmp_path / "readme.txt").write_text("not a model")
    (tmp_path / "config.yaml").write_text("k: v")
    found = {p.name for p in find_model_files(tmp_path)}
    assert found == {"policy.zip", "weights.pt"}


def test_find_model_files_empty_for_code_only(tmp_path):
    (tmp_path / "strategy.py").write_text("# pure python strategy")
    assert find_model_files(tmp_path) == []


# ------------------------------------------------------------------ verify (signed)


def test_signed_model_passes_enforce(tmp_path, registry):
    art_dir, model = _artifact_with_model(tmp_path)
    registry.register("prod-model", artifact_path=str(model))  # signs it
    verdicts = verify_artifact_models(art_dir, live=True, registry=registry, context="t")
    assert len(verdicts) == 1
    assert verdicts[0].ok and verdicts[0].signature_valid


# ------------------------------------------------------------------ verify (unsigned) — fail-closed


def test_unsigned_model_raises_in_live_enforce(tmp_path, registry):
    art_dir, _ = _artifact_with_model(tmp_path)  # NOT registered → no signature
    with pytest.raises(ModelSignatureError):
        verify_artifact_models(art_dir, live=True, registry=registry, context="t")


def test_tampered_model_raises(tmp_path, registry):
    art_dir, model = _artifact_with_model(tmp_path, content=b"original")
    registry.register("m", artifact_path=str(model))
    model.write_bytes(b"tampered-after-signing")  # sha256 no longer matches registry
    with pytest.raises(ModelSignatureError):
        verify_artifact_models(art_dir, live=True, registry=registry, context="t")


# ------------------------------------------------------------------ warn policy


def test_unsigned_model_warns_not_raises(tmp_path, registry):
    art_dir, _ = _artifact_with_model(tmp_path)
    # explicit warn policy → verdict returned, no exception (research/backtest)
    verdicts = verify_artifact_models(
        art_dir, live=False, policy="warn", registry=registry, context="t"
    )
    assert len(verdicts) == 1 and not verdicts[0].ok


def test_off_policy_skips(tmp_path, registry):
    art_dir, _ = _artifact_with_model(tmp_path)
    verdicts = verify_artifact_models(
        art_dir, live=True, policy="off", registry=registry, context="t"
    )
    assert len(verdicts) == 1 and verdicts[0].ok


# ------------------------------------------------------------------ code-only strategy


def test_code_only_artifact_no_checkpoint(tmp_path, registry):
    art_dir = tmp_path / "code_only"
    art_dir.mkdir()
    (art_dir / "strategy.py").write_text("# no model checkpoint")
    verdicts = verify_artifact_models(art_dir, live=True, registry=registry, context="t")
    assert verdicts == []  # nothing to verify; manifest/digest controls apply


# ------------------------------------------------------------------ RunController wiring (fail-closed)


def test_run_controller_fails_closed_on_unsigned_model(tmp_path, registry, monkeypatch):
    """RunController.initialize must abort (return False) for a LIVE run whose
    artifact carries an unsigned model checkpoint."""
    from packages.agent.daemon import agentd as agentd_mod
    from packages.agent.daemon.agentd import RunController, RunControllerConfig
    from packages.shared.contracts.config import ExecutionMode

    # Point the gate's default registry at our temp (empty) registry so the
    # unsigned checkpoint is unregistered → gate fails in enforce.
    monkeypatch.setattr("service_experiment_tracking.get_registry", lambda: registry)

    art_dir, _ = _artifact_with_model(tmp_path)  # unsigned

    class _Artifact:
        artifact_id = "art-1"
        extracted_path = art_dir
        manifest = None

    class _ArtifactManager:
        def get_active_artifact(self):
            return _Artifact()

        def get_artifact(self, _id):
            return _Artifact()

    errors = []
    ctrl = RunController(
        config=RunControllerConfig(execution_mode=ExecutionMode.LIVE, sandbox_enabled=False),
        on_error=errors.append,
    )
    ok = ctrl.initialize(_ArtifactManager())
    assert ok is False  # fail-closed
    assert not ctrl.is_initialized
    assert any("signature" in e.lower() or "подпис" in e.lower() for e in errors)


def test_run_controller_allows_signed_model(tmp_path, registry, monkeypatch):
    from packages.agent.daemon.agentd import RunController, RunControllerConfig
    from packages.shared.contracts.config import ExecutionMode

    monkeypatch.setattr("service_experiment_tracking.get_registry", lambda: registry)

    art_dir, model = _artifact_with_model(tmp_path)
    registry.register("prod", artifact_path=str(model))  # signed

    class _Artifact:
        artifact_id = "art-2"
        extracted_path = art_dir
        manifest = None

    class _ArtifactManager:
        def get_active_artifact(self):
            return _Artifact()

        def get_artifact(self, _id):
            return _Artifact()

    ctrl = RunController(
        config=RunControllerConfig(execution_mode=ExecutionMode.LIVE, sandbox_enabled=False),
    )
    ok = ctrl.initialize(_ArtifactManager())
    assert ok is True and ctrl.is_initialized
