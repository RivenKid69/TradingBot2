"""Conservative, auditable repair for the Lite prices dataset."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def repair_prices_file(path: str | Path, *, forward_fill_limit: int = 5) -> Dict[str, Any]:
    """Repair ordering, duplicates, infinities and short internal gaps atomically.

    Long/leading gaps are intentionally left unresolved: inventing a long market
    history would be less correct than reporting that new data is required.
    A single ``.preheal.bak`` copy is retained for operator rollback.
    """

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(str(target))
    if target.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("Lite auto-heal supports Parquet price data only")

    frame = pd.read_parquet(target)
    if frame.empty:
        raise ValueError("prices dataset is empty")

    rows_before = len(frame)
    missing_before = int(frame.isna().sum().sum())
    numeric = list(frame.select_dtypes(include=[np.number]).columns)
    infinite_before = int(np.isinf(frame[numeric].to_numpy(dtype=float, copy=True)).sum()) if numeric else 0
    if numeric:
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)

    symbol_col = next((c for c in ("symbol", "ticker", "instrument") if c in frame.columns), None)
    time_col = next((c for c in ("ts", "ts_ms", "timestamp", "datetime", "date") if c in frame.columns), None)
    sort_cols = [c for c in (symbol_col, time_col) if c]
    if sort_cols:
        frame = frame.sort_values(sort_cols, kind="stable")
        before_dedup = len(frame)
        frame = frame.drop_duplicates(subset=sort_cols, keep="last")
        duplicates_removed = before_dedup - len(frame)
    else:
        duplicates_removed = 0

    fill_columns = [c for c in frame.columns if c not in {symbol_col, time_col}]
    fillable_missing_before = int(frame[fill_columns].isna().sum().sum()) if fill_columns else 0
    if fill_columns:
        if symbol_col:
            frame[fill_columns] = frame.groupby(symbol_col, sort=False, dropna=False)[fill_columns].ffill(
                limit=max(1, int(forward_fill_limit))
            )
        else:
            frame[fill_columns] = frame[fill_columns].ffill(limit=max(1, int(forward_fill_limit)))

    missing_after = int(frame.isna().sum().sum())
    fillable_missing_after = int(frame[fill_columns].isna().sum().sum()) if fill_columns else 0
    cells_filled = max(0, fillable_missing_before - fillable_missing_after)

    backup = target.with_suffix(target.suffix + ".preheal.bak")
    shutil.copy2(target, backup)
    fd, tmp_name = tempfile.mkstemp(prefix=target.stem + ".heal-", suffix=target.suffix, dir=target.parent)
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, target)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass

    return {
        "path": str(target), "backup": str(backup),
        "rows_before": rows_before, "rows_after": len(frame),
        "duplicates_removed": int(duplicates_removed),
        "infinite_values_replaced": infinite_before,
        "missing_before": missing_before, "missing_after": missing_after,
        "cells_filled": cells_filled,
        "complete": missing_after == 0,
        "forward_fill_limit": max(1, int(forward_fill_limit)),
    }


__all__ = ["repair_prices_file"]
