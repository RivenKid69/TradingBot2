# make_walkforward_splits.py
from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import yaml

from splits import make_walkforward_splits


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Неизвестный формат файла данных: {ext}")


def _write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
        return
    if ext in (".csv", ".txt"):
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Неизвестный формат файла вывода: {ext}")


def _write_manifest(manifest, json_path: str, yaml_path: str) -> None:
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    data = [m.to_dict() for m in manifest]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _write_phase_tables(df_out: pd.DataFrame, base: str, ext: str) -> tuple[list[str], list[str]]:
    train_df = df_out[df_out["wf_role"] == "train"].copy()
    val_df = df_out[df_out["wf_role"] == "val"].copy()
    if set(train_df.index) & set(val_df.index):
        raise ValueError("Train/validation rows overlap")
    train_path = f"{base}_train{ext}"
    val_path = f"{base}_val{ext}"
    _write_table(train_df, train_path)
    _write_table(val_df, val_path)
    return [train_path], [val_path]


def main():
    ap = argparse.ArgumentParser(description="Сгенерировать walk-forward сплиты с PURGE (горизонт h) и EMBARGO (буфер).")
    ap.add_argument("--config", help="Путь к YAML-файлу конфигурации (из него считывается путь к данным).")
    ap.add_argument("--data", help="Входной датасет (CSV/Parquet) с колонкой ts_ms (UTC миллисекунды).")
    ap.add_argument("--out", default="", help="Путь к выходному датасету с колонками wf_fold,wf_role. По умолчанию рядом с суффиксом _wf.")
    ap.add_argument("--ts_col", default="ts_ms", help="Имя колонки времени.")
    ap.add_argument("--symbol_col", default="symbol", help="Имя колонки символа (может отсутствовать).")
    ap.add_argument("--interval_ms", type=int, default=None, help="Интервал бара в мс (если не задан — оценим автоматически).")
    ap.add_argument(
        "--train_span_bars",
        type=int,
        default=7 * 6,
        help=(
            "Длина train-окна в барах. По умолчанию 42 бара (~7 дней при таймфрейме 4h)."
        ),
    )
    ap.add_argument(
        "--val_span_bars",
        type=int,
        default=6,
        help=(
            "Длина val-окна в барах. По умолчанию 6 баров (~1 день при таймфрейме 4h)."
        ),
    )
    ap.add_argument(
        "--step_bars",
        type=int,
        default=6,
        help=(
            "Шаг окна в барах. По умолчанию 6 баров (~1 день при таймфрейме 4h)."
        ),
    )
    ap.add_argument(
        "--horizon_bars",
        type=int,
        default=15,
        help=(
            "Горизонт таргета в барах (PURGE). По умолчанию 15 баров (~2.5 дня при таймфрейме 4h)."
        ),
    )
    ap.add_argument(
        "--embargo_bars",
        type=int,
        default=2,
        help=(
            "Буфер EMBARGO в барах. По умолчанию 2 бара (~8 часов при таймфрейме 4h)."
        ),
    )
    ap.add_argument("--manifest_dir", default="logs/walkforward", help="Куда записать манифесты (JSON/YAML).")
    ap.add_argument(
        "--n_splits",
        type=int,
        default=None,
        help=(
            "Простой режим (Lite UI): количество walk-forward сплитов. "
            "Совместно с --train_size_pct пересчитывается в train/val/step-окна по фактической длине данных."
        ),
    )
    ap.add_argument(
        "--train_size_pct",
        type=float,
        default=None,
        help="Простой режим (Lite UI): доля train-окна внутри одного сплита, в процентах (например 80).",
    )
    args = ap.parse_args()

    data_path = args.data
    ts_col = args.ts_col
    symbol_col = args.symbol_col

    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f) or {}
            cfg_data_section = cfg_data.get("data", {})
            if not data_path and "path" in cfg_data_section:
                data_path = cfg_data_section["path"]
            if ts_col == "ts_ms" and "ts_col" in cfg_data_section:
                ts_col = cfg_data_section["ts_col"]
            if symbol_col == "symbol" and "symbol_col" in cfg_data_section:
                symbol_col = cfg_data_section["symbol_col"]
        except Exception as e:
            print(f"Предупреждение: ошибка чтения --config: {e}")

    if not data_path:
        ap.error("Необходимо указать --data или --config с корректным путем к данным.")

    df = _read_table(data_path)

    train_span_bars = int(args.train_span_bars)
    val_span_bars = int(args.val_span_bars)
    step_bars = int(args.step_bars)

    if args.n_splits is not None:
        # Lite-режим: пересчитываем окна из фактической длины данных, чтобы
        # UI-контракт (n_splits/train_size_pct) отражался в реальных барах.
        #
        # Скользящее окно шагает на val_span, поэтому число фолдов равно
        # 1 + (usable - (train+val)) / val. Требуя train + n*val = usable при
        # train/(train+val) = train_frac, получаем ровно n_splits фолдов:
        #   span1 = usable / (train_frac + n*(1-train_frac))
        n_splits = max(1, int(args.n_splits))
        train_pct = float(args.train_size_pct) if args.train_size_pct is not None else 80.0
        train_frac = min(max(train_pct / 100.0, 0.05), 0.95)
        n_bars = int(df[ts_col].nunique()) if ts_col in df.columns else int(len(df))
        overhead = int(args.horizon_bars) + int(args.embargo_bars)
        usable = n_bars - overhead
        if usable < n_splits * 4:
            ap.error(
                f"Недостаточно данных для n_splits={n_splits}: после вычета purge/embargo "
                f"({overhead} баров) остаётся {usable} баров, нужно минимум {n_splits * 4}. "
                "Уменьшите число сплитов или загрузите больше истории."
            )
        denom = train_frac + n_splits * (1.0 - train_frac)
        span1 = usable / denom
        train_span_bars = max(1, int(span1 * train_frac))
        val_span_bars = max(1, int(round(span1)) - train_span_bars)
        step_bars = val_span_bars
        if train_span_bars <= int(args.horizon_bars):
            ap.error(
                f"train-окно ({train_span_bars} баров) не превышает purge-горизонт "
                f"({args.horizon_bars} баров) — все фолды были бы пустыми. "
                "Уменьшите n_splits/--horizon_bars или загрузите больше истории."
            )
        expected_folds = 1 + max(0, (usable - (train_span_bars + val_span_bars)) // step_bars)
        print(
            f"Lite-режим: n_splits={n_splits}, train_size_pct={train_pct:.0f} -> "
            f"train_span_bars={train_span_bars}, val_span_bars={val_span_bars}, step_bars={step_bars} "
            f"(баров в данных: {n_bars}, ожидается фолдов: ~{expected_folds})."
        )

    df_out, manifest = make_walkforward_splits(
        df,
        ts_col=ts_col,
        symbol_col=(symbol_col if symbol_col in df.columns else None),
        interval_ms=args.interval_ms,
        train_span_bars=train_span_bars,
        val_span_bars=val_span_bars,
        step_bars=step_bars,
        horizon_bars=int(args.horizon_bars),
        embargo_bars=int(args.embargo_bars),
    )

    base, ext = os.path.splitext(data_path)
    out_path = args.out.strip() or f"{base}_wf{ext if ext.lower() in ('.csv', '.parquet', '.pq', '.txt') else '.parquet'}"
    _write_table(df_out, out_path)
    train_paths, val_paths = _write_phase_tables(df_out, base, ext)

    json_path = os.path.join(args.manifest_dir, "walkforward_manifest.json")
    yaml_path = os.path.join(args.manifest_dir, "walkforward_manifest.yaml")
    _write_manifest(manifest, json_path=json_path, yaml_path=yaml_path)

    total = int(len(df_out))
    used = int((df_out["wf_role"] != "none").sum())
    n_train = int((df_out["wf_role"] == "train").sum())
    n_val = int((df_out["wf_role"] == "val").sum())
    print(f"Готово. Записан датасет со сплитами: {out_path}")
    print(f"Всего строк: {total}. В сплитах train: {n_train}, val: {n_val}, вне окон: {total - used}.")
    print(f"Train path: {train_paths[0]}, Val path: {val_paths[0]}")
    print(f"Манифесты: {json_path} и {yaml_path}")
    return train_paths, val_paths


if __name__ == "__main__":
    main()
