"""Dataset Versioning / Lineage (feature-store-lite).

Self-contained research utility backing a "Dataset Versioning / Lineage" panel.

Provides content-addressed snapshots of datasets (parquet/csv) with schema,
time-range, file/content hashing, a JSON registry with lineage (parent_id),
and a structured diff between two snapshots.

No project-internal imports: stdlib + numpy + pandas only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _file_sha256(path: str, chunk_size: int = 8192) -> str:
    """Chunked SHA-256 of the raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_dataframe(path: str) -> pd.DataFrame:
    """Load a dataframe based on file extension (parquet/csv)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    # Best effort fallback: try parquet then csv.
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def _content_hash(df: pd.DataFrame) -> str:
    """Stable content hash: sha256 over hash_pandas_object (columns sorted)."""
    df_sorted = df[sorted(df.columns)]
    row_hashes = pd.util.hash_pandas_object(df_sorted, index=True)
    return hashlib.sha256(row_hashes.values.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Schema / metadata extraction
# ---------------------------------------------------------------------------

_TIME_COL_CANDIDATES = ("ts", "ts_ms", "timestamp", "time")


def _extract_schema(df: pd.DataFrame) -> List[Dict[str, str]]:
    return [{"name": str(name), "dtype": str(df[name].dtype)} for name in sorted(df.columns)]


def _to_jsonable(value: Any) -> Any:
    """Coerce a scalar (possibly numpy/pandas) into a JSON-serializable form."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).isoformat()
    try:
        # numpy scalar generic
        return value.item()
    except Exception:
        return str(value)


def _extract_time_range(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    for col in _TIME_COL_CANDIDATES:
        if col in df.columns:
            series = df[col].dropna()
            if series.empty:
                return {"col": col, "start": None, "end": None}
            return {
                "col": col,
                "start": _to_jsonable(series.min()),
                "end": _to_jsonable(series.max()),
            }
    return None


# ---------------------------------------------------------------------------
# Registry I/O
# ---------------------------------------------------------------------------


def _load_registry(registry_path: str) -> Dict[str, Any]:
    if not os.path.exists(registry_path):
        return {"datasets": []}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "datasets" not in data:
            return {"datasets": []}
        if not isinstance(data["datasets"], list):
            data["datasets"] = []
        return data
    except Exception:
        return {"datasets": []}


def _atomic_write_registry(registry_path: str, data: Dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(registry_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{registry_path}.tmp.{os.getpid()}.{int(time.time() * 1e6)}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
    os.replace(tmp_path, registry_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def snapshot_dataset(
    path: str,
    config: Optional[Dict[str, Any]] = None,
    parent_id: Optional[str] = None,
    registry_path: str = "models/dataset_registry.json",
) -> Dict[str, Any]:
    """Create (or update) a content-addressed snapshot of a dataset file.

    Returns the snapshot entry dict.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(abs_path)

    file_sha256 = _file_sha256(abs_path)
    size_bytes = os.path.getsize(abs_path)

    content_hash: Optional[str] = None
    schema: List[Dict[str, str]] = []
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    time_range: Optional[Dict[str, Any]] = None
    load_error: Optional[str] = None

    try:
        df = _load_dataframe(abs_path)
        content_hash = _content_hash(df)
        schema = _extract_schema(df)
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])
        time_range = _extract_time_range(df)
    except Exception as exc:  # pragma: no cover - depends on bad input
        load_error = f"{type(exc).__name__}: {exc}"

    entry_id = content_hash[:12] if content_hash else file_sha256[:12]
    created_at = datetime.utcnow().isoformat() + "Z"

    entry: Dict[str, Any] = {
        "id": entry_id,
        "path": abs_path,
        "file_sha256": file_sha256,
        "content_hash": content_hash,
        "schema": schema,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "time_range": time_range,
        "size_bytes": size_bytes,
        "created_at": created_at,
        "last_seen": created_at,
        "parent_id": parent_id,
        "config": dict(config) if config is not None else None,
    }
    if load_error is not None:
        entry["load_error"] = load_error

    registry = _load_registry(registry_path)
    datasets = registry["datasets"]

    existing = next((d for d in datasets if d.get("id") == entry_id), None)
    if existing is not None:
        # Dedup: update last_seen (and parent_id/config if newly supplied).
        existing["last_seen"] = created_at
        if parent_id is not None:
            existing["parent_id"] = parent_id
        if config is not None:
            existing["config"] = dict(config)
        result = existing
    else:
        datasets.append(entry)
        result = entry

    _atomic_write_registry(registry_path, registry)
    return result


def list_datasets(registry_path: str = "models/dataset_registry.json") -> List[Dict[str, Any]]:
    """Return the list of dataset snapshot entries from the registry."""
    return _load_registry(registry_path)["datasets"]


def _find_entry(datasets: List[Dict[str, Any]], entry_id: str) -> Optional[Dict[str, Any]]:
    return next((d for d in datasets if d.get("id") == entry_id), None)


def diff_datasets(
    id1: str,
    id2: str,
    registry_path: str = "models/dataset_registry.json",
) -> Dict[str, Any]:
    """Structured diff between two registry snapshots (id1 -> id2)."""
    datasets = list_datasets(registry_path)
    e1 = _find_entry(datasets, id1)
    e2 = _find_entry(datasets, id2)
    if e1 is None:
        raise KeyError(f"dataset id not found: {id1}")
    if e2 is None:
        raise KeyError(f"dataset id not found: {id2}")

    schema1 = {s["name"]: s["dtype"] for s in (e1.get("schema") or [])}
    schema2 = {s["name"]: s["dtype"] for s in (e2.get("schema") or [])}

    names1 = set(schema1)
    names2 = set(schema2)

    schema_added = sorted(names2 - names1)
    schema_removed = sorted(names1 - names2)
    dtype_changed = [
        {"name": name, "from": schema1[name], "to": schema2[name]}
        for name in sorted(names1 & names2)
        if schema1[name] != schema2[name]
    ]

    n1 = e1.get("n_rows")
    n2 = e2.get("n_rows")
    n_rows_delta = (n2 - n1) if (n1 is not None and n2 is not None) else None

    hash_equal = e1.get("content_hash") is not None and e1.get("content_hash") == e2.get(
        "content_hash"
    )

    return {
        "schema_added": schema_added,
        "schema_removed": schema_removed,
        "dtype_changed": dtype_changed,
        "n_rows_delta": n_rows_delta,
        "hash_equal": bool(hash_equal),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_register(args: argparse.Namespace) -> int:
    config = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    result = snapshot_dataset(
        path=args.path,
        config=config,
        parent_id=args.parent,
        registry_path=args.registry,
    )
    print(json.dumps(result, indent=2))
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    print(json.dumps(list_datasets(args.registry), indent=2))
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    result = diff_datasets(args.id1, args.id2, registry_path=args.registry)
    print(json.dumps(result, indent=2))
    return 0


def _selftest() -> int:
    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="dataset_versioning_selftest_")
    try:
        registry_path = os.path.join(tmp_dir, "dataset_registry.json")
        path_a = os.path.join(tmp_dir, "df_a.parquet")
        path_b = os.path.join(tmp_dir, "df_b.parquet")

        n = 20
        df_a = pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
                "x": np.arange(n, dtype=np.float64),
                "y": np.arange(n, dtype=np.int64) * 2,
            }
        )
        # df_b = df_a + one extra column + 5 more rows.
        df_b = df_a.copy()
        df_b["z"] = np.arange(n, dtype=np.float64) + 0.5
        extra = pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-21", periods=5, freq="D"),
                "x": np.arange(n, n + 5, dtype=np.float64),
                "y": np.arange(n, n + 5, dtype=np.int64) * 2,
                "z": np.arange(n, n + 5, dtype=np.float64) + 0.5,
            }
        )
        df_b = pd.concat([df_b, extra], ignore_index=True)

        df_a.to_parquet(path_a)
        df_b.to_parquet(path_b)

        entry_a = snapshot_dataset(path_a, registry_path=registry_path)
        entry_b = snapshot_dataset(path_b, parent_id=entry_a["id"], registry_path=registry_path)

        id_a = entry_a["id"]
        id_b = entry_b["id"]

        assert id_a != id_b, "ids must differ"

        ds = list_datasets(registry_path)
        assert len(ds) == 2, f"expected 2 entries, got {len(ds)}"

        d = diff_datasets(id_a, id_b, registry_path=registry_path)
        assert "z" in d["schema_added"], f"schema_added missing 'z': {d['schema_added']}"
        assert d["n_rows_delta"] == 5, f"n_rows_delta != 5: {d['n_rows_delta']}"
        assert d["hash_equal"] is False, "hash_equal should be False"

        # Re-register a -> dedup, still 2 entries.
        snapshot_dataset(path_a, registry_path=registry_path)
        ds_after = list_datasets(registry_path)
        assert len(ds_after) == 2, f"dedup failed, got {len(ds_after)} entries"

        print("VERSIONING SELFTEST OK")
        print(json.dumps({"id_a": id_a, "id_b": id_b, "diff": d}, indent=2))
        return 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dataset Versioning / Lineage (feature-store-lite)."
    )
    parser.add_argument("--selftest", action="store_true", help="run self-test and exit")

    sub = parser.add_subparsers(dest="command")

    p_reg = sub.add_parser("register", help="snapshot a dataset into the registry")
    p_reg.add_argument("path")
    p_reg.add_argument("--parent", default=None, help="parent dataset id")
    p_reg.add_argument("--config", default=None, help="path to a JSON config file")
    p_reg.add_argument("--registry", default="models/dataset_registry.json")
    p_reg.set_defaults(func=_cmd_register)

    p_list = sub.add_parser("list", help="list registered datasets")
    p_list.add_argument("--registry", default="models/dataset_registry.json")
    p_list.set_defaults(func=_cmd_list)

    p_diff = sub.add_parser("diff", help="diff two datasets by id")
    p_diff.add_argument("id1")
    p_diff.add_argument("id2")
    p_diff.add_argument("--registry", default="models/dataset_registry.json")
    p_diff.set_defaults(func=_cmd_diff)

    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
