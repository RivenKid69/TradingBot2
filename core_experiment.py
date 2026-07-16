# -*- coding: utf-8 -*-
"""
core_experiment.py
==================

Слой ``core_`` — контракты для experiment tracking и model registry.
Без тяжёлых зависимостей (только stdlib): dataclasses/enum/typing.

Назначение (P0-блокер «нельзя защитить выбор модели перед LP/регулятором»):
лёгкий, файловый, воспроизводимый аналог MLflow:
  * **Run** — прогон обучения/бэктеста: params + metrics-история + теги.
  * **Lineage** — связь модель → данные (хэш) → конфиг (хэш) → git-commit → родитель.
  * **ArtifactRef** — артефакт с SHA-256 и криптоподписью (provenance).
  * **ModelVersion** — версия модели в реестре со стадией (staging/production/...).

Сериализация — через ``to_dict``/``from_dict`` (чистый JSON, без pickle).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RunStatus(str, Enum):
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    KILLED = "KILLED"


class ModelStage(str, Enum):
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class Lineage:
    """Происхождение прогона: что → из чего получено (для воспроизводимости/аудита)."""
    data_hash: Optional[str] = None        # хэш датасета (контент/версия)
    config_hash: Optional[str] = None      # хэш конфига
    git_commit: Optional[str] = None       # коммит кода
    git_dirty: Optional[bool] = None       # были ли незакоммиченные изменения
    dataset_uri: Optional[str] = None      # путь/идентификатор датасета
    config_uri: Optional[str] = None       # путь к конфигу
    parent_run_id: Optional[str] = None    # родительский прогон (для PBT/ансамблей)
    # Reproducibility fingerprint (P2 #23): seed + environment lockfile so a run can
    # be deterministically reproduced and dirty-tree promotions can be blocked.
    seed: Optional[int] = None
    python_version: Optional[str] = None
    platform: Optional[str] = None
    package_versions: Dict[str, str] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Lineage":
        d = dict(d or {})
        known = {f.name for f in dataclasses.fields(cls)}
        extra = {k: d.pop(k) for k in list(d.keys()) if k not in known}
        obj = cls(**{k: v for k, v in d.items() if k in known})
        if extra:
            obj.extra.update(extra)
        return obj


@dataclass
class ArtifactRef:
    """Артефакт с интегритетом и подписью (provenance)."""
    path: str                              # путь к сохранённому артефакту
    sha256: str                            # хэш содержимого
    size_bytes: int = 0
    algo: str = "none"                     # подпись: ed25519 | hmac-sha256 | none
    signature: Optional[str] = None        # hex-подпись
    public_key: Optional[str] = None       # hex pubkey (для ed25519)
    name: Optional[str] = None             # логическое имя артефакта

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactRef":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RunRecord:
    """Прогон эксперимента."""
    run_id: str
    experiment: str
    status: str = RunStatus.RUNNING.value
    start_ms: int = 0
    end_ms: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)   # последнее значение по ключу
    lineage: Lineage = field(default_factory=Lineage)
    artifacts: List[ArtifactRef] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["lineage"] = self.lineage.to_dict()
        d["artifacts"] = [a.to_dict() for a in self.artifacts]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        d = dict(d)
        d["lineage"] = Lineage.from_dict(d.get("lineage"))
        d["artifacts"] = [ArtifactRef.from_dict(a) for a in d.get("artifacts", [])]
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ModelVersion:
    """Версия модели в реестре."""
    name: str
    version: int
    run_id: Optional[str] = None
    stage: str = ModelStage.NONE.value
    artifact: Optional[ArtifactRef] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    lineage: Lineage = field(default_factory=Lineage)
    description: str = ""
    created_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["artifact"] = self.artifact.to_dict() if self.artifact else None
        d["lineage"] = self.lineage.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelVersion":
        d = dict(d)
        d["artifact"] = ArtifactRef.from_dict(d["artifact"]) if d.get("artifact") else None
        d["lineage"] = Lineage.from_dict(d.get("lineage"))
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


__all__ = [
    "RunStatus", "ModelStage", "Lineage", "ArtifactRef", "RunRecord", "ModelVersion",
]
