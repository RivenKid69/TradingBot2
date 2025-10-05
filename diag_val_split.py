import argparse, json, os, re, sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import yaml

def to_ts(x) -> Optional[int]:
    if x is None: return None
    try:
        ts = pd.Timestamp(x)
        return int(ts.timestamp())
    except Exception:
        try:
            # целое/строка секунд или миллисекунд
            s = int(str(x).strip())
            if s > 10_000_000_000:  # мс -> сек
                s = s // 1000
            return s
        except Exception:
            return None

def read_yaml(p: Path) -> Dict[str, Any]:
    try:
        with p.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}"}

def detect_splits(cfg: Dict[str, Any], repo: Path) -> Tuple[Optional[Tuple[int,int]], Optional[Tuple[int,int]], str]:
    """Возвращает (train_win, val_win, source_note)."""
    d = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    # Вариант 1: прямые таймстемпы в конфиге
    t = (to_ts(d.get("train_start_ts")), to_ts(d.get("train_end_ts")))
    v = (to_ts(d.get("val_start_ts")),   to_ts(d.get("val_end_ts")))
    if all(t) and all(v):
        return t, v, "config:data.{train_*,val_*}"

    # Вариант 2: отдельный файл сплитов
    split_path = d.get("split_path") or d.get("splits_path")
    if split_path:
        sp = (repo / split_path) if not os.path.isabs(split_path) else Path(split_path)
        spd = read_yaml(sp)
        for key in ("splits", "dataset_splits"):
            if key in spd:
                S = spd[key]
                def win_of(name):
                    arr = S.get(name)
                    if isinstance(arr, list) and arr:
                        a = arr[0]
                        return (to_ts(a.get("start")), to_ts(a.get("end")))
                    elif isinstance(arr, dict):
                        return (to_ts(arr.get("start")), to_ts(arr.get("end")))
                    return (None, None)
                t = win_of("train")
                v = win_of("val")
                if all(t) and all(v):
                    return t, v, f"{sp.name}:{key}"
    # Вариант 3: offline.yaml как запасной
    off = read_yaml(repo / "configs/offline.yaml")
    for key in ("splits","dataset_splits"):
        if key in off and isinstance(off[key], dict) and "time" in off[key]:
            timeS = off[key]["time"]
            t = (to_ts(timeS.get("train", {}).get("start")), to_ts(timeS.get("train", {}).get("end")))
            v = (to_ts(timeS.get("val",   {}).get("start")), to_ts(timeS.get("val",   {}).get("end")))
            if all(t) and all(v):
                return t, v, f"configs/offline.yaml:{key}.time"
    return (None,None), (None,None), "not_found"

def find_processed_dir(cfg: Dict[str, Any], repo: Path) -> Path:
    d = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    p = d.get("processed_dir") or d.get("processed_path") or "data/processed"
    P = (repo / p) if not os.path.isabs(p) else Path(p)
    return P

def scan_feathers(processed: Path, max_files: int = 400) -> List[Dict[str, Any]]:
    files = list(processed.rglob("*.feather"))[:max_files]
    out = []
    for f in files:
        try:
            df = pd.read_feather(f)
        except Exception as e:
            out.append({"file": str(f), "error": f"{type(e).__name__}: {e}"})
            continue
        col_ts = None
        for c in ("timestamp","open_time","time","ts"):
            if c in df.columns:
                col_ts = c
                break
        if col_ts is None:
            out.append({"file": str(f), "error": "no timestamp-like column"})
            continue
        ts = pd.to_numeric(df[col_ts], errors="coerce").dropna().astype("int64")
        # если миллисекунды
        if ts.max() > 10_000_000_000:
            ts = ts // 1000
        # иногда open_time = ms + часовой сдвиг; не трогаем — главное min/max
        sym = df.get("symbol")
        symbol = (str(sym.iloc[0]) if sym is not None and len(sym) else Path(f).stem)
        out.append({
            "file": str(f),
            "symbol": symbol,
            "rows": int(len(df)),
            "ts_min": int(ts.min()) if len(ts) else None,
            "ts_max": int(ts.max()) if len(ts) else None,
        })
    return out

def coverage(rows, win: Tuple[int,int]):
    a,b = win
    if not (a and b): return 0
    cnt=0
    for r in rows:
        mi, ma = r.get("ts_min"), r.get("ts_max")
        if mi is None or ma is None: continue
        if not (ma < a or mi > b):   # пересечение
            cnt += 1
    return cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_train_spot_bar.yaml")
    ap.add_argument("--logdir", default="wsl_diag_logs")
    args = ap.parse_args()

    repo = Path.cwd()
    logdir = Path(args.logdir); logdir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    cfg = read_yaml(cfg_path)

    processed = find_processed_dir(cfg, repo)
    train_win, val_win, src = detect_splits(cfg, repo)

    # Скан данных
    rows = []
    if processed.exists():
        rows = scan_feathers(processed, max_files=800)
    else:
        rows = []
    # Итоги
    import datetime as dt
    pretty = lambda ts: (dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else None)

    train_cov = coverage(rows, train_win)
    val_cov   = coverage(rows, val_win)

    result = {
        "repo": str(repo),
        "config": str(cfg_path),
        "processed_dir": str(processed),
        "splits_source": src,
        "train_window": {"start": train_win[0], "end": train_win[1], "start_utc": pretty(train_win[0]), "end_utc": pretty(train_win[1])},
        "val_window":   {"start": val_win[0],   "end": val_win[1],   "start_utc": pretty(val_win[0]),   "end_utc": pretty(val_win[1])},
        "files_scanned": len(rows),
        "train_symbols_covering": train_cov,
        "val_symbols_covering":   val_cov,
        "notes": []
    }

    if src == "not_found":
        result["notes"].append("❗ Сплиты не найдены ни в config.data.*, ни в data.split_path, ни в configs/offline.yaml.")
    if val_cov == 0:
        result["notes"].append("❗ Ни один символ не пересекается с валид. окном — как в твоей ошибке ValueError.")

    # Топ проблем/символов
    df = pd.DataFrame(rows)
    df.to_csv(logdir / "diag_val_split_raw.csv", index=False)
    # Сводная по символам
    sym = (df.groupby("symbol")[["ts_min","ts_max","rows"]]
             .agg({"ts_min":"min","ts_max":"max","rows":"sum"})
             .reset_index()
          if not df.empty else pd.DataFrame(columns=["symbol","ts_min","ts_max","rows"]))
    sym["ts_min_utc"] = sym["ts_min"].apply(lambda x: pretty(int(x)) if pd.notna(x) else None)
    sym["ts_max_utc"] = sym["ts_max"].apply(lambda x: pretty(int(x)) if pd.notna(x) else None)
    sym["covers_train"] = sym.apply(lambda r: int(not (r.ts_max < train_win[0] or r.ts_min > train_win[1])) if train_win[0] and train_win[1] and pd.notna(r.ts_min) and pd.notna(r.ts_max) else 0, axis=1)
    sym["covers_val"]   = sym.apply(lambda r: int(not (r.ts_max < val_win[0]   or r.ts_min > val_win[1]))   if val_win[0]   and val_win[1]   and pd.notna(r.ts_min) and pd.notna(r.ts_max) else 0, axis=1)
    sym.sort_values(["covers_val","covers_train","rows"], ascending=[False, False, False], inplace=True)
    sym.to_csv(logdir / "diag_val_split_by_symbol.csv", index=False)

    # JSON
    with (logdir / "diag_val_split.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Короткая текстовая сводка
    with (logdir / "diag_val_split.txt").open("w", encoding="utf-8") as f:
        f.write(f"[CONFIG] {cfg_path}\n")
        f.write(f"[PROCESSED] {processed} (exists={processed.exists()})\n")
        f.write(f"[SPLITS] source={src}\n")
        f.write(f"  TRAIN: {train_win}  {pretty(train_win[0])} .. {pretty(train_win[1])}\n")
        f.write(f"  VAL  : {val_win}    {pretty(val_win[0])} .. {pretty(val_win[1])}\n")
        f.write(f"[FILES SCANNED] {len(rows)}\n")
        f.write(f"[COVERAGE] train_symbols={train_cov}, val_symbols={val_cov}\n")
        for n in result["notes"]:
            f.write(f"[NOTE] {n}\n")

    print("--- DIAG SUMMARY ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))
