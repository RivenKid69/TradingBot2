# -*- coding: utf-8 -*-
"""Тесты experiment tracking + model registry (P0: MLOps воспроизводимость)."""

from __future__ import annotations

import json
import os

import pytest

from core_experiment import ModelStage, RunStatus
from service_experiment_tracking import (
    ArtifactSigner, ExperimentTracker, ModelRegistry, hash_config, sha256_file,
)


@pytest.fixture()
def signer(tmp_path):
    return ArtifactSigner(key_dir=str(tmp_path / "keys"))


@pytest.fixture()
def tracker(tmp_path, signer):
    return ExperimentTracker(root=str(tmp_path / "experiments"), signer=signer)


@pytest.fixture()
def registry(tmp_path, signer):
    return ModelRegistry(root=str(tmp_path / "model_registry"), signer=signer)


def _write(p, text="hello-model"):
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return str(p)


# --- Signer ---------------------------------------------------------------
def test_signer_sign_and_verify(signer, tmp_path):
    f = _write(tmp_path / "a.json")
    s = signer.sign_file(f)
    assert s["sha256"] == sha256_file(f)
    assert s["algo"] in ("ed25519", "hmac-sha256")
    assert signer.verify(s["sha256"], s["signature"], s["algo"], s.get("public_key")) is True


def test_signer_detects_tamper(signer, tmp_path):
    f = _write(tmp_path / "a.json", "original")
    s = signer.sign_file(f)
    # подменяем содержимое
    _write(tmp_path / "a.json", "tampered")
    new_digest = sha256_file(f)
    assert new_digest != s["sha256"]
    assert signer.verify(new_digest, s["signature"], s["algo"], s.get("public_key")) is False


def test_signer_persists_keypair(tmp_path):
    kd = str(tmp_path / "keys")
    s1 = ArtifactSigner(key_dir=kd)
    f = _write(tmp_path / "a.json")
    sig = s1.sign_file(f)
    # новый инстанс с тем же каталогом ключей должен верифицировать
    s2 = ArtifactSigner(key_dir=kd)
    assert s2.verify(sig["sha256"], sig["signature"], sig["algo"], sig.get("public_key")) is True


# --- Tracker --------------------------------------------------------------
def test_run_lifecycle_and_metrics(tracker):
    with tracker.run("exp1", params={"lr": 0.001}) as run:
        run.set_tags({"asset": "crypto"})
        run.log_metric("sharpe", 1.0, step=0)
        run.log_metric("sharpe", 1.3, step=1)
        rid = run.run_id
    rec = tracker.get_run("exp1", rid)
    assert rec is not None
    assert rec.status == RunStatus.FINISHED.value
    assert rec.params["lr"] == 0.001
    assert rec.tags["asset"] == "crypto"
    assert rec.metrics["sharpe"] == 1.3            # последнее значение
    hist = tracker.read_metric_history("exp1", rid, "sharpe")
    assert [h["value"] for h in hist] == [1.0, 1.3]   # полная история


def test_run_failure_marks_failed(tracker):
    rid = None
    with pytest.raises(RuntimeError):
        with tracker.run("exp1") as run:
            rid = run.run_id
            raise RuntimeError("boom")
    rec = tracker.get_run("exp1", rid)
    assert rec.status == RunStatus.FAILED.value


def test_lineage_capture(tracker):
    with tracker.run("exp1") as run:
        run.set_lineage(dataset_uri="data/x.parquet", config_uri="configs/c.yaml",
                        data_hash="dh", config_hash="ch")
        rid = run.run_id
    rec = tracker.get_run("exp1", rid)
    assert rec.lineage.dataset_uri == "data/x.parquet"
    assert rec.lineage.data_hash == "dh"
    assert rec.lineage.config_hash == "ch"
    # git_commit может быть None вне git-репо — но поле присутствует
    assert hasattr(rec.lineage, "git_commit")


def test_log_artifact_signed(tracker, tmp_path):
    f = _write(tmp_path / "model.json", '{"w": [1,2,3]}')
    with tracker.run("exp1") as run:
        ref = run.log_artifact(f, name="model.json")
        rid = run.run_id
    assert os.path.exists(ref.path)
    assert os.path.exists(ref.path + ".sig")
    assert ref.sha256 == sha256_file(f)
    rec = tracker.get_run("exp1", rid)
    assert len(rec.artifacts) == 1
    assert tracker.signer.verify_file(rec.artifacts[0].path, rec.artifacts[0]) is True


def test_list_experiments_and_runs(tracker):
    with tracker.run("expA"):
        pass
    with tracker.run("expA"):
        pass
    with tracker.run("expB"):
        pass
    assert set(tracker.list_experiments()) == {"expA", "expB"}
    assert len(tracker.list_runs("expA")) == 2


# --- Registry -------------------------------------------------------------
def test_register_autoversion(registry, tmp_path):
    f = _write(tmp_path / "m.json")
    v1 = registry.register("alpha", artifact_path=f, metrics={"sharpe": 1.1})
    v2 = registry.register("alpha", artifact_path=f, metrics={"sharpe": 1.4})
    assert v1.version == 1 and v2.version == 2
    assert registry.verify("alpha", 1) is True
    assert registry.verify("alpha", 2) is True


def test_production_transition_archives_previous(registry, tmp_path):
    f = _write(tmp_path / "m.json")
    registry.register("alpha", artifact_path=f)
    registry.register("alpha", artifact_path=f)
    registry.transition("alpha", 1, "production")
    assert registry.get("alpha", stage="production").version == 1
    registry.transition("alpha", 2, "production")
    # v1 должна стать archived, единственный production — v2
    assert registry.get("alpha", stage="production").version == 2
    assert registry.get_version("alpha", 1).stage == ModelStage.ARCHIVED.value


def test_rollback_to_previous_production(registry, tmp_path):
    f = _write(tmp_path / "m.json")
    registry.register("alpha", artifact_path=f)
    registry.register("alpha", artifact_path=f)
    registry.transition("alpha", 1, "production")
    registry.transition("alpha", 2, "production")   # v1 -> archived
    rolled = registry.rollback("alpha")             # вернуть на v1
    assert rolled.version == 1
    assert registry.get("alpha", stage="production").version == 1


def test_rollback_explicit_version(registry, tmp_path):
    f = _write(tmp_path / "m.json")
    for _ in range(3):
        registry.register("alpha", artifact_path=f)
    registry.transition("alpha", 3, "production")
    rolled = registry.rollback("alpha", to_version=2)
    assert rolled.version == 2
    assert registry.get("alpha", stage="production").version == 2


def test_registry_detects_tampered_artifact(registry, tmp_path):
    f = _write(tmp_path / "m.json", "good")
    mv = registry.register("alpha", artifact_path=f)
    # подменяем сохранённый артефакт в реестре
    with open(mv.artifact.path, "w", encoding="utf-8") as fh:
        fh.write("EVIL")
    assert registry.verify("alpha", mv.version) is False


# --- helpers --------------------------------------------------------------
def test_hash_config_deterministic():
    a = {"x": 1, "y": [1, 2], "z": {"k": "v"}}
    b = {"z": {"k": "v"}, "y": [1, 2], "x": 1}   # другой порядок ключей
    assert hash_config(a) == hash_config(b)
    assert hash_config(a) != hash_config({"x": 2})
