"""Тесты Ed25519-гейта модельных артефактов (services/model_signature_gate).

Закрывает §4.7 из PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md: подпись реестра
теперь ПРОВЕРЯЕТСЯ на пути загрузки модели в live, fail-closed в enforce.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-signature-gate")

from service_experiment_tracking import ArtifactSigner, ModelRegistry
from services.model_signature_gate import (
    ModelSignatureError,
    find_registry_entry,
    resolve_policy,
    verify_model_artifact,
)


@pytest.fixture()
def registry(tmp_path: Path) -> ModelRegistry:
    signer = ArtifactSigner(key_dir=str(tmp_path / "keys"))
    return ModelRegistry(root=str(tmp_path / "registry"), signer=signer)


@pytest.fixture()
def signed_model(tmp_path: Path, registry: ModelRegistry):
    src = tmp_path / "ppo_agent.zip"
    src.write_bytes(b"fake-sb3-checkpoint-bytes-v1")
    mv = registry.register("test_alpha", artifact_path=str(src))
    return src, mv


# ------------------------------------------------------------- policy resolution


def test_policy_resolution_order(monkeypatch):
    monkeypatch.delenv("RIVEN_MODEL_SIGNATURE_POLICY", raising=False)
    assert resolve_policy(None, live=True) == "enforce"  # live default — fail-closed
    assert resolve_policy(None, live=False) == "warn"  # research default
    assert resolve_policy("off", live=True) == "off"  # явный аргумент сильнее
    monkeypatch.setenv("RIVEN_MODEL_SIGNATURE_POLICY", "enforce")
    assert resolve_policy(None, live=False) == "enforce"  # env сильнее default'а
    monkeypatch.setenv("RIVEN_MODEL_SIGNATURE_POLICY", "bogus")
    assert resolve_policy(None, live=False) == "warn"  # мусор в env игнорируется


# ------------------------------------------------------------------ happy path


def test_registered_signed_model_passes(registry, signed_model, monkeypatch):
    monkeypatch.delenv("RIVEN_REQUIRE_PRODUCTION_MODEL", raising=False)
    src, mv = signed_model
    verdict = verify_model_artifact(str(src), policy="enforce", registry=registry)
    assert verdict.ok is True
    assert verdict.registered is True
    assert verdict.signature_valid is True
    assert verdict.model_name == "test_alpha" and verdict.version == 1


def test_lookup_matches_by_digest_not_path(registry, signed_model, tmp_path):
    # Пользователь грузит копию файла из другого места — digest тот же.
    src, _mv = signed_model
    copy = tmp_path / "elsewhere" / "model_copy.zip"
    copy.parent.mkdir()
    copy.write_bytes(src.read_bytes())
    verdict = verify_model_artifact(str(copy), policy="enforce", registry=registry)
    assert verdict.ok is True and verdict.model_name == "test_alpha"


# ------------------------------------------------------------------ fail-closed


def test_unregistered_model_enforce_raises(registry, tmp_path):
    rogue = tmp_path / "rogue.zip"
    rogue.write_bytes(b"totally-unsigned-payload")
    with pytest.raises(ModelSignatureError, match="не зарегистрирован"):
        verify_model_artifact(str(rogue), policy="enforce", registry=registry)


def test_tampered_artifact_enforce_raises(registry, signed_model):
    src, mv = signed_model
    # Подмена байтов В КОПИИ РЕЕСТРА: digest реестровой записи больше не бьётся
    # ни с одним файлом; загрузка исходника всё ещё валидна, а загрузка
    # подменённой реестровой копии — нет.
    reg_copy = Path(mv.artifact.path)
    reg_copy.write_bytes(b"EVIL" + reg_copy.read_bytes())
    with pytest.raises(ModelSignatureError, match="не зарегистрирован"):
        verify_model_artifact(str(reg_copy), policy="enforce", registry=registry)


def test_forged_signature_enforce_raises(registry, signed_model, tmp_path):
    src, mv = signed_model
    # Ломаем подпись в реестре (артефакт тот же, подпись — мусор).
    import json

    meta_path = os.path.join(registry.root, "test_alpha", "registry.json")
    data = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    data["versions"][0]["artifact"]["signature"] = "00" * 64
    Path(meta_path).write_text(json.dumps(data), encoding="utf-8")
    registry2 = ModelRegistry(root=registry.root, signer=registry.signer)
    with pytest.raises(ModelSignatureError, match="НЕвалидна"):
        verify_model_artifact(str(src), policy="enforce", registry=registry2)


def test_missing_file_enforce_raises(registry):
    with pytest.raises(ModelSignatureError, match="не найден"):
        verify_model_artifact("no/such/model.zip", policy="enforce", registry=registry)


# -------------------------------------------------------------------- политики


def test_warn_returns_verdict_without_raising(registry, tmp_path):
    rogue = tmp_path / "rogue.zip"
    rogue.write_bytes(b"unsigned")
    verdict = verify_model_artifact(str(rogue), policy="warn", registry=registry)
    assert verdict.ok is False and verdict.registered is False
    assert verdict.checked is True


def test_off_skips_check(registry, tmp_path):
    verdict = verify_model_artifact("no/such/file.zip", policy="off", registry=registry)
    assert verdict.ok is True and verdict.checked is False


def test_production_stage_requirement(registry, signed_model, monkeypatch):
    src, mv = signed_model
    monkeypatch.setenv("RIVEN_REQUIRE_PRODUCTION_MODEL", "1")
    with pytest.raises(ModelSignatureError, match="production"):
        verify_model_artifact(str(src), policy="enforce", registry=registry)
    registry.transition("test_alpha", 1, "production", force=True)
    verdict = verify_model_artifact(str(src), policy="enforce", registry=registry)
    assert verdict.ok is True and verdict.stage == "production"


# --------------------------------------------------------- проводка в RL loader


def test_rl_loader_gate_blocks_before_deserialization(tmp_path, monkeypatch):
    """В enforce-политике загрузчик обязан упасть ДО pickle-десериализации."""
    from service_rl_inference import make_sb3_distributional_loader

    rogue = tmp_path / "rogue_ckpt.zip"
    rogue.write_bytes(b"malicious-pickle-bytes")

    class _Explosive:
        @staticmethod
        def load(*a, **k):  # pragma: no cover - не должен вызваться
            raise AssertionError("десериализация началась ДО проверки подписи!")

    loader = make_sb3_distributional_loader(ppo_cls=_Explosive, live=True)
    monkeypatch.delenv("RIVEN_MODEL_SIGNATURE_POLICY", raising=False)
    with pytest.raises(ModelSignatureError):
        loader(str(rogue), "cpu")


def test_rl_loader_warn_mode_proceeds_to_load(tmp_path, monkeypatch):
    from service_rl_inference import make_sb3_distributional_loader

    rogue = tmp_path / "rogue_ckpt.zip"
    rogue.write_bytes(b"whatever")
    sentinel = {}

    class _Recorder:
        @staticmethod
        def load(*a, **k):
            sentinel["loaded"] = True
            raise RuntimeError("stop here")  # дальше нам не нужно

    loader = make_sb3_distributional_loader(ppo_cls=_Recorder, live=False)  # warn
    monkeypatch.delenv("RIVEN_MODEL_SIGNATURE_POLICY", raising=False)
    assert loader(str(rogue), "cpu") is None  # load упал → нейтральный сигнал (как раньше)
    assert sentinel.get("loaded") is True  # но до load дошли: warn не блокирует


def test_find_registry_entry_none_for_unknown(registry, tmp_path):
    assert find_registry_entry("ff" * 32, registry=registry) is None
