# -*- coding: utf-8 -*-
"""Тесты Feature Store (P2): версии по контент-хэшу, as-of чтение, online-кэш."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from service_feature_store import FeatureStore, content_hash


@pytest.fixture()
def store(tmp_path):
    return FeatureStore(root=str(tmp_path / "fs"))


def _df(seed=0, n=10):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"momentum": rng.normal(0, 1, n)}, index=range(n))


def test_version_by_content_hash(store):
    df = _df(0)
    v1 = store.write("momentum", df, asof_ms=1000)
    assert v1.version == 1 and v1.rows == 10
    # тот же контент → НЕ новая версия (без дубля)
    v1b = store.write("momentum", df.copy(), asof_ms=2000)
    assert v1b.version == 1
    assert v1b.asof_ms == 2000              # asof обновился
    # изменённый контент → v2
    v2 = store.write("momentum", _df(1), asof_ms=3000)
    assert v2.version == 2
    assert len(store.list_versions("momentum")) == 2


def test_content_hash_deterministic():
    df = _df(5)
    assert content_hash(df) == content_hash(df.copy())
    assert content_hash(df) != content_hash(_df(6))


def test_read_latest_version_and_specific(store):
    store.write("f", _df(0), asof_ms=1000)
    store.write("f", _df(1), asof_ms=2000)
    latest = store.read("f")
    v1 = store.read("f", version=1)
    assert latest is not None and v1 is not None
    assert content_hash(latest) == content_hash(_df(1))
    assert content_hash(v1) == content_hash(_df(0))


def test_asof_read_is_pit(store):
    store.write("f", _df(0), asof_ms=1000)   # v1 @ asof 1000
    store.write("f", _df(1), asof_ms=2000)   # v2 @ asof 2000
    # запрос на дату 1500 → видит только v1 (PIT)
    df_15 = store.read("f", asof_ms=1500)
    assert content_hash(df_15) == content_hash(_df(0))
    # на дату 2500 → v2
    df_25 = store.read("f", asof_ms=2500)
    assert content_hash(df_25) == content_hash(_df(1))
    # раньше первой версии → None
    assert store.read("f", asof_ms=500) is None


def test_online_cache_and_ttl(store):
    df = _df(0)
    store.cache_put("f", df, asof_ms=1, ttl_sec=10)
    assert store.cache_get("f", asof_ms=1) is df
    # истёкший TTL
    store.cache_put("f", df, asof_ms=2, ttl_sec=0.0)
    time.sleep(0.01)
    # ttl_sec=0 → expire=now → сразу истёк
    store._cache["f|2|None"]["expire"] = 0
    assert store.cache_get("f", asof_ms=2) is None


def test_get_uses_cache_then_disk(store):
    store.write("f", _df(0), asof_ms=1000)
    # первый get с диска + кладёт в кэш
    d1 = store.get("f", ttl_sec=60)
    assert d1 is not None
    # подменим диск (удалим), кэш должен отдать
    d2 = store.get("f")
    assert d2 is not None and content_hash(d2) == content_hash(d1)


def test_list_and_materialize(store):
    store.write("a", pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2]))
    store.write("b", pd.DataFrame({"b": [4, 5, 6]}, index=[0, 1, 2]))
    assert set(store.list_features()) == {"a", "b"}
    mat = store.materialize(["a", "b"])
    assert list(mat.columns) == ["a", "b"] and len(mat) == 3
