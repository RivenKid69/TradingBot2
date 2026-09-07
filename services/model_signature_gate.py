"""Гейт Ed25519-подписи модельных артефактов на пути в live.

Закрывает пункт §4.7 из PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md: реестр моделей
подписывал артефакты (Ed25519, ``service_experiment_tracking.ArtifactSigner``),
но ни одна точка ЗАГРУЗКИ модели в живой контур подпись не проверяла — защита
была декоративной.

Почему это security-контрол, а не бюрократия: SB3-чекпоинт (``.zip``)
десериализуется через pickle, то есть **загрузка неподписанного/подменённого
файла = исполнение произвольного кода** в процессе с брокерскими ключами.
Проверка целостности и подписи ДО десериализации — стандарт supply-chain
безопасности (SLSA/TUF); CCEA design doc прямо требует
«Artifact Signature Verification: REQUIRED».

Политики (``RIVEN_MODEL_SIGNATURE_POLICY`` или явный аргумент):

* ``enforce`` — незарегистрированный/неподписанный/битый артефакт вызывает
  ``ModelSignatureError`` (fail-closed). Дефолт для live-контекстов.
* ``warn``    — вердикт логируется, загрузка продолжается. Дефолт для
  research/backtest, чтобы не ломать локальные эксперименты.
* ``off``     — проверка пропускается (только явным решением оператора).

Дополнительно (``RIVEN_REQUIRE_PRODUCTION_MODEL=1``): в enforce-режиме артефакт
обязан быть в stage=production реестра (дисциплина promotion-gate:
champion/challenger, см. P2 №16).
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_TRUTHY = ("1", "true", "yes", "on")
VALID_POLICIES = ("enforce", "warn", "off")


class ModelSignatureError(RuntimeError):
    """Артефакт модели не прошёл проверку подписи в enforce-режиме."""


@dataclass
class SignatureVerdict:
    """Честная раскладка проверки одного артефакта."""

    path: str
    policy: str
    checked: bool = False
    sha256: Optional[str] = None
    registered: bool = False
    model_name: Optional[str] = None
    version: Optional[int] = None
    stage: Optional[str] = None
    algo: Optional[str] = None
    signature_valid: bool = False
    production_required: bool = False
    ok: bool = False
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_policy(explicit: Optional[str] = None, *, live: bool = False) -> str:
    """explicit > env RIVEN_MODEL_SIGNATURE_POLICY > (enforce для live, иначе warn)."""
    for cand in (explicit, os.environ.get("RIVEN_MODEL_SIGNATURE_POLICY")):
        if cand:
            cand = str(cand).strip().lower()
            if cand in VALID_POLICIES:
                return cand
            logger.warning("model-signature: неизвестная политика %r — игнорирую", cand)
    return "enforce" if live else "warn"


def _production_required() -> bool:
    return os.environ.get("RIVEN_REQUIRE_PRODUCTION_MODEL", "").strip().lower() in _TRUTHY


def find_registry_entry(sha256: str, registry: Any = None):
    """Найти (model_name, ModelVersion) по sha256 артефакта во всём реестре.

    Поиск по содержимому (digest), а не по пути: пользователь может загружать
    исходный файл, а не копию из реестра — цифровой отпечаток одинаков.
    """
    if registry is None:
        from service_experiment_tracking import get_registry

        registry = get_registry()
    try:
        names = [
            d for d in os.listdir(registry.root) if os.path.isdir(os.path.join(registry.root, d))
        ]
    except OSError:
        return None
    for name in sorted(names):
        try:
            for mv in registry.list_versions(name):
                art = getattr(mv, "artifact", None)
                if art is not None and getattr(art, "sha256", None) == sha256:
                    return name, mv
        except Exception:
            continue
    return None


def verify_model_artifact(
    path: str,
    *,
    policy: Optional[str] = None,
    live: bool = False,
    registry: Any = None,
    context: str = "",
) -> SignatureVerdict:
    """Проверить артефакт модели перед загрузкой/десериализацией.

    Возвращает :class:`SignatureVerdict`; в политике ``enforce`` любой провал
    (нет файла, не зарегистрирован, подпись невалидна, при требовании —
    не production) поднимает :class:`ModelSignatureError` ДО того, как кто-либо
    успеет распаковать pickle.
    """
    eff_policy = resolve_policy(policy, live=live)
    verdict = SignatureVerdict(
        path=str(path), policy=eff_policy, production_required=_production_required()
    )

    if eff_policy == "off":
        verdict.ok = True
        verdict.reason = "проверка выключена политикой 'off'"
        return verdict

    verdict.checked = True

    if not path or not os.path.exists(path):
        verdict.reason = f"файл артефакта не найден: {path}"
        return _finalize(verdict, context)

    from service_experiment_tracking import get_registry, sha256_file

    if registry is None:
        registry = get_registry()

    verdict.sha256 = sha256_file(path)
    entry = find_registry_entry(verdict.sha256, registry=registry)
    if entry is None:
        verdict.reason = (
            "артефакт не зарегистрирован в model registry (подписи нет). "
            "Зарегистрируйте: from service_experiment_tracking import get_registry; "
            f"get_registry().register('<name>', artifact_path=r'{path}')"
        )
        return _finalize(verdict, context)

    name, mv = entry
    art = mv.artifact
    verdict.registered = True
    verdict.model_name = name
    verdict.version = int(mv.version)
    verdict.stage = getattr(mv, "stage", None)
    verdict.algo = getattr(art, "algo", None)

    try:
        verdict.signature_valid = bool(
            registry.signer.verify(
                verdict.sha256, art.signature, art.algo, getattr(art, "public_key", None)
            )
        )
    except Exception as exc:
        verdict.signature_valid = False
        verdict.reason = f"ошибка проверки подписи: {exc}"
        return _finalize(verdict, context)

    if not verdict.signature_valid:
        verdict.reason = (
            f"подпись НЕвалидна для '{name}' v{mv.version} (algo={verdict.algo}) — "
            "артефакт подменён или подписан другим ключом"
        )
        return _finalize(verdict, context)

    if verdict.production_required and str(verdict.stage) != "production":
        verdict.reason = (
            f"'{name}' v{mv.version} подписан, но stage={verdict.stage!r}, а "
            "RIVEN_REQUIRE_PRODUCTION_MODEL=1 требует production (promotion-gate)"
        )
        return _finalize(verdict, context)

    verdict.ok = True
    verdict.reason = (
        f"подпись валидна: '{name}' v{mv.version} (stage={verdict.stage}, {verdict.algo})"
    )
    logger.info("model-signature[%s]: OK %s", context or "load", verdict.reason)
    return verdict


def _finalize(verdict: SignatureVerdict, context: str) -> SignatureVerdict:
    """Единая точка исхода для провалов: enforce → исключение, warn → лог."""
    if verdict.ok:
        return verdict
    msg = f"model-signature[{context or 'load'}]: {verdict.reason} (path={verdict.path})"
    if verdict.policy == "enforce":
        logger.error(msg)
        raise ModelSignatureError(msg)
    logger.warning("%s — политика 'warn', загрузка продолжается", msg)
    return verdict


def assert_model_trusted(
    path: str, *, context: str = "live", policy: Optional[str] = None
) -> SignatureVerdict:
    """Строгий вариант для торговых путей: дефолтная политика enforce."""
    return verify_model_artifact(path, policy=policy, live=True, context=context)


__all__ = [
    "ModelSignatureError",
    "SignatureVerdict",
    "VALID_POLICIES",
    "assert_model_trusted",
    "find_registry_entry",
    "resolve_policy",
    "verify_model_artifact",
]
