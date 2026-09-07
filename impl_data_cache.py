# -*- coding: utf-8 -*-
"""
impl_data_cache.py
==================

Parquet-кэш сырых баров для free-источников (Stage D0). Free-тиры (binance/yahoo/oanda)
лимитированы по rate — кэш защищает от повторных запросов и ускоряет повторные прогоны
в Lab. Атомарная запись (tmp + os.replace), TTL по mtime файла.

Путь: ``<root>/<vendor>/<symbol>_<timeframe>.parquet`` (символ санитизируется).
Кэш **best-effort**: любые ошибки чтения/записи мягко логируются и не валят сборку.

``now_ms`` инъецируется (детерминизм тестов); по умолчанию — системное время.
Слой ``impl_`` (без зависимостей от service).
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_SANITIZE = re.compile(r"[^A-Za-z0-9_.=-]+")


def _now_ms() -> int:
    return int(time.time() * 1000)


class ParquetCache:
    """Файловый parquet-кэш баров по (vendor, symbol, timeframe)."""

    def __init__(
        self, root: str = os.path.join("data", "cache", "xs"), *, enabled: bool = True
    ) -> None:
        self.root = str(root)
        self.enabled = bool(enabled)

    def _path(self, vendor: str, symbol: str, timeframe: str) -> str:
        sym = _SANITIZE.sub("_", str(symbol))
        tf = _SANITIZE.sub("_", str(timeframe) or "na")
        ven = _SANITIZE.sub("_", str(vendor) or "na")
        return os.path.join(self.root, ven, f"{sym}_{tf}.parquet")

    def get(
        self,
        vendor: str,
        symbol: str,
        timeframe: str,
        *,
        ttl_ms: Optional[int] = None,
        now_ms: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Вернуть кэш или None (если нет/устарел/ошибка)."""
        if not self.enabled:
            return None
        p = self._path(vendor, symbol, timeframe)
        if not os.path.exists(p):
            return None
        if ttl_ms is not None:
            now = now_ms if now_ms is not None else _now_ms()
            try:
                mtime_ms = int(os.path.getmtime(p) * 1000)
            except OSError:
                return None
            if now - mtime_ms > int(ttl_ms):
                return None  # устарел
        try:
            return pd.read_parquet(p)
        except Exception as exc:  # pragma: no cover - повреждённый кэш
            logger.warning("ParquetCache: read failed %s: %s", p, exc)
            return None

    def put(self, vendor: str, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """Атомарно записать кэш. True если успешно."""
        if not self.enabled or df is None or len(df) == 0:
            return False
        p = self._path(vendor, symbol, timeframe)
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + ".tmp"
            df.to_parquet(tmp, index=False)
            os.replace(tmp, p)  # атомарно на одной FS
            return True
        except Exception as exc:  # pragma: no cover - нет parquet-движка/прав
            logger.warning("ParquetCache: write failed %s: %s", p, exc)
            try:
                if os.path.exists(p + ".tmp"):
                    os.remove(p + ".tmp")
            except OSError:
                pass
            return False

    def clear(self, vendor: Optional[str] = None) -> None:
        """Удалить кэш (всего вендора или весь корень)."""
        import shutil

        target = os.path.join(self.root, _SANITIZE.sub("_", vendor)) if vendor else self.root
        try:
            if os.path.isdir(target):
                shutil.rmtree(target)
        except OSError as exc:  # pragma: no cover
            logger.warning("ParquetCache: clear failed %s: %s", target, exc)


__all__ = ["ParquetCache"]
