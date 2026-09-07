# -*- coding: utf-8 -*-
"""
service_feature_store.py
========================

Feature Store (P2): версионирование на уровне ОТДЕЛЬНОЙ фичи — ключ ``(name, asof,
content_hash)`` — и **online-кэш для inference**, вместо per-run/file-level.

Зачем профи: одна и та же фича переиспользуется многими стратегиями; нужно знать
ТОЧНУЮ версию (какой контент/как посчитан) для воспроизводимости, и быстро отдавать
её на inference без пересчёта. Это закрывает разрыв «per-run parquet → нет переиспользования».

Возможности:
  * **Версии по контент-хэшу:** ``write`` бампит версию ТОЛЬКО при изменении содержимого
    (одинаковый df → та же версия, без дублей). Каждая версия несёт asof, hash, lineage.
  * **As-of чтение:** ``read(name, asof_ms=...)`` отдаёт версию, актуальную на дату.
  * **Online-кэш:** ``cache_put/cache_get`` (TTL) — горячая отдача на inference.
  * **Materialize:** ``materialize(names, asof)`` собирает несколько фич в один кадр (с кэшем).

Хранилище: ``feature_store/<name>/v<version>/{data.parquet, meta.json}`` +
``feature_store/<name>/index.json``. Stdlib + pandas. Слой service_.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _atomic_write(path: str, data: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def content_hash(df: pd.DataFrame) -> str:
    """Детерминированный хэш содержимого фичи (значения + индекс + колонки)."""
    h = hashlib.sha256()
    try:
        h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    except Exception:
        h.update(df.to_csv().encode("utf-8"))
    h.update("|".join(map(str, df.columns)).encode("utf-8"))
    return h.hexdigest()


@dataclass
class FeatureVersion:
    name: str
    version: int
    content_hash: str
    asof_ms: int
    created_ms: int
    rows: int
    columns: List[str]
    lineage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureVersion":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class FeatureStore:
    def __init__(self, root: Optional[str] = None) -> None:
        self.root = root or os.path.join(_ROOT, "feature_store")
        os.makedirs(self.root, exist_ok=True)
        self._lock = threading.RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}  # key -> {"df":..., "expire": ms|None}

    # ---- paths ----
    def _name_dir(self, name: str) -> str:
        return os.path.join(self.root, name)

    def _index_path(self, name: str) -> str:
        return os.path.join(self._name_dir(name), "index.json")

    def _vdir(self, name: str, version: int) -> str:
        return os.path.join(self._name_dir(name), f"v{version}")

    def _load_index(self, name: str) -> List[Dict[str, Any]]:
        p = self._index_path(name)
        if not os.path.exists(p):
            return []
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_index(self, name: str, versions: List[Dict[str, Any]]) -> None:
        _atomic_write(self._index_path(name), json.dumps(versions, indent=2, ensure_ascii=False))

    # ---- write (version by content hash) ----
    def write(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        asof_ms: Optional[int] = None,
        lineage: Optional[Dict[str, Any]] = None,
    ) -> FeatureVersion:
        with self._lock:
            asof = int(asof_ms if asof_ms is not None else _now_ms())
            chash = content_hash(df)
            index = self._load_index(name)
            if index and index[-1]["content_hash"] == chash:
                # контент не изменился → та же версия (без дубля), обновим asof
                latest = FeatureVersion.from_dict(index[-1])
                if asof > latest.asof_ms:
                    index[-1]["asof_ms"] = asof
                    self._save_index(name, index)
                    latest.asof_ms = asof
                return latest
            version = (index[-1]["version"] + 1) if index else 1
            vdir = self._vdir(name, version)
            os.makedirs(vdir, exist_ok=True)
            df.to_parquet(os.path.join(vdir, "data.parquet"))
            fv = FeatureVersion(
                name=name,
                version=version,
                content_hash=chash,
                asof_ms=asof,
                created_ms=_now_ms(),
                rows=int(len(df)),
                columns=[str(c) for c in df.columns],
                lineage=dict(lineage or {}),
            )
            _atomic_write(
                os.path.join(vdir, "meta.json"),
                json.dumps(fv.to_dict(), indent=2, ensure_ascii=False),
            )
            index.append(fv.to_dict())
            self._save_index(name, index)
            return fv

    # ---- read (as-of or specific version) ----
    def _resolve_version(self, name: str, *, asof_ms: Optional[int], version: Optional[int]):
        index = self._load_index(name)
        if not index:
            return None
        if version is not None:
            for v in index:
                if v["version"] == version:
                    return FeatureVersion.from_dict(v)
            return None
        if asof_ms is None:
            return FeatureVersion.from_dict(index[-1])  # latest
        # последняя версия с asof_ms <= запрошенного (PIT)
        eligible = [v for v in index if v["asof_ms"] <= int(asof_ms)]
        if not eligible:
            return None
        return FeatureVersion.from_dict(max(eligible, key=lambda v: v["version"]))

    def read(
        self, name: str, *, asof_ms: Optional[int] = None, version: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        fv = self._resolve_version(name, asof_ms=asof_ms, version=version)
        if fv is None:
            return None
        path = os.path.join(self._vdir(name, fv.version), "data.parquet")
        if not os.path.exists(path):
            return None
        return pd.read_parquet(path)

    def get_version(
        self, name: str, *, asof_ms: Optional[int] = None, version: Optional[int] = None
    ) -> Optional[FeatureVersion]:
        return self._resolve_version(name, asof_ms=asof_ms, version=version)

    def list_versions(self, name: str) -> List[FeatureVersion]:
        return [FeatureVersion.from_dict(v) for v in self._load_index(name)]

    def list_features(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted([d for d in os.listdir(self.root) if os.path.isfile(self._index_path(d))])

    # ---- online cache (inference) ----
    @staticmethod
    def _ckey(name: str, asof_ms: Optional[int], version: Optional[int]) -> str:
        return f"{name}|{asof_ms}|{version}"

    def cache_put(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        asof_ms: Optional[int] = None,
        version: Optional[int] = None,
        ttl_sec: Optional[float] = None,
    ) -> None:
        exp = (_now_ms() + int(ttl_sec * 1000)) if ttl_sec else None
        self._cache[self._ckey(name, asof_ms, version)] = {"df": df, "expire": exp}

    def cache_get(
        self, name: str, *, asof_ms: Optional[int] = None, version: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        rec = self._cache.get(self._ckey(name, asof_ms, version))
        if rec is None:
            return None
        if rec["expire"] is not None and _now_ms() > rec["expire"]:
            self._cache.pop(self._ckey(name, asof_ms, version), None)
            return None
        return rec["df"]

    def get(
        self,
        name: str,
        *,
        asof_ms: Optional[int] = None,
        version: Optional[int] = None,
        use_cache: bool = True,
        ttl_sec: Optional[float] = 300.0,
    ) -> Optional[pd.DataFrame]:
        """Inference-путь: сначала online-кэш, иначе диск (+положить в кэш)."""
        if use_cache:
            cached = self.cache_get(name, asof_ms=asof_ms, version=version)
            if cached is not None:
                return cached
        df = self.read(name, asof_ms=asof_ms, version=version)
        if df is not None and use_cache:
            self.cache_put(name, df, asof_ms=asof_ms, version=version, ttl_sec=ttl_sec)
        return df

    def materialize(
        self, names: Sequence[str], *, asof_ms: Optional[int] = None, use_cache: bool = True
    ) -> pd.DataFrame:
        """Собрать несколько фич в один кадр (join по индексу) для обучения/inference."""
        frames = []
        for n in names:
            df = self.get(n, asof_ms=asof_ms, use_cache=use_cache)
            if df is not None:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = frames[0]
        for f in frames[1:]:
            out = out.join(f, how="outer")
        return out


_GLOBAL_STORE: Optional[FeatureStore] = None


def get_feature_store() -> FeatureStore:
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = FeatureStore()
    return _GLOBAL_STORE


__all__ = ["FeatureStore", "FeatureVersion", "content_hash", "get_feature_store"]
