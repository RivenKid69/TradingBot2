# -*- coding: utf-8 -*-
"""
run_conformal_calibration.py
CLI tool to calibrate conformal prediction intervals (CQR, EnbPI, ACI, Naive)
using predictions and targets from validation datasets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Add startup dir to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from service_conformal import create_conformal_service
from core_conformal import ConformalMethod


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Неизвестный формат файла данных: {ext}")


def main():
    ap = argparse.ArgumentParser(description="Калибровка конформных интервалов (Conformal Prediction) на валидационных данных.")
    ap.add_argument("--config", default="configs/conformal.yaml", help="Пусть к YAML файлу конфигурации conformal.")
    ap.add_argument("--predictions_path", required=True, help="Путь к Parquet/CSV файлу с прогнозами модели.")
    ap.add_argument("--out_state", default="models/conformal_state.json", help="Путь для сохранения откалиброванного состояния.")
    ap.add_argument("--y_col", default="y", help="Имя колонки истинного значения (таргета).")
    ap.add_argument("--score_col", default="score", help="Имя колонки точечного предсказания (скора).")
    ap.add_argument("--score_lower_col", default="score_lower", help="Имя колонки нижней границы (для CQR).")
    ap.add_argument("--score_upper_col", default="score_upper", help="Имя колонки верхней границы (для CQR).")
    ap.add_argument("--filter_val", action="store_true", help="Оставить только строки wf_role == 'val'.")
    ap.add_argument("--wf_role_col", default="wf_role", help="Колонка роли выборки (для фильтрации).")
    args = ap.parse_args()

    print(f"Загрузка конфигурации конформного оценивания из: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}
    
    conformal_cfg = config_data.get("conformal", config_data)
    service = create_conformal_service(conformal_cfg)
    
    if not service.is_enabled():
        print("Ошибка: Конформное оценивание отключено в конфигурационном файле (enabled: false)")
        sys.exit(1)

    print(f"Загрузка данных прогнозов из: {args.predictions_path}")
    df = _read_table(args.predictions_path)

    if args.filter_val and args.wf_role_col in df.columns:
        df = df.loc[df[args.wf_role_col].astype(str) == "val"].reset_index(drop=True)
        print(f"Отфильтровано по роли 'val'. Строк для калибровки: {len(df)}")
    else:
        print(f"Всего строк в таблице для калибровки: {len(df)}")

    # Проверка наличия колонок
    if args.score_col not in df.columns or args.y_col not in df.columns:
        # Попытка фолбека к eff_ret
        y_col_real = args.y_col
        if y_col_real not in df.columns:
            for potential_y in ["y", "eff_ret_60", "eff_ret", "target"]:
                if potential_y in df.columns:
                    y_col_real = potential_y
                    break
        score_col_real = args.score_col
        if score_col_real not in df.columns:
            for potential_score in ["score", "prediction", "ref_price"]:
                if potential_score in df.columns:
                    score_col_real = potential_score
                    break
        
        if score_col_real not in df.columns or y_col_real not in df.columns:
            raise ValueError(f"Колонки {args.score_col} и {args.y_col} не найдены в датасете. Доступные колонки: {list(df.columns)}")
        
        print(f"Используются авто-колонки: предсказание = '{score_col_real}', истина = '{y_col_real}'")
        args.score_col = score_col_real
        args.y_col = y_col_real

    predictions = pd.to_numeric(df[args.score_col], errors="coerce").astype(float).to_numpy()
    true_values = pd.to_numeric(df[args.y_col], errors="coerce").astype(float).to_numpy()

    # Удаление NaN из выборок
    valid_mask = np.isfinite(predictions) & np.isfinite(true_values)
    predictions = predictions[valid_mask]
    true_values = true_values[valid_mask]

    predicted_lower = None
    predicted_upper = None

    if service.config.method == ConformalMethod.CQR:
        print("Используется метод CQR. Проверка наличия квантилей...")
        if args.score_lower_col in df.columns and args.score_upper_col in df.columns:
            predicted_lower = pd.to_numeric(df[args.score_lower_col], errors="coerce").astype(float).to_numpy()[valid_mask]
            predicted_upper = pd.to_numeric(df[args.score_upper_col], errors="coerce").astype(float).to_numpy()[valid_mask]
            print(f"Квантили загружены из колонок {args.score_lower_col} и {args.score_upper_col}")
        else:
            # Фолбек: сгенерировать квантили на основе разброса остатков
            print("Предупреждение: Колонки квантилей не найдены. Выполняется фолбек-оценка квантилей по остаткам.")
            residuals = true_values - predictions
            q_lo = float(np.percentile(residuals, 5))
            q_hi = float(np.percentile(residuals, 95))
            predicted_lower = predictions + q_lo
            predicted_upper = predictions + q_hi

    # Запуск калибровки
    print(f"Запуск калибровки ConformalPredictionService (метод: {service.config.method.name})...")
    result = service.calibrate(
        predictions=predictions,
        true_values=true_values,
        predicted_lower=predicted_lower,
        predicted_upper=predicted_upper
    )

    if result.success:
        print("Калибровка выполнена успешно!")
        print(f"Выборка калибровки: {result.samples_used} точек.")
        if result.empirical_coverage is not None:
            print(f"Эмпирическое покрытие (empirical coverage): {result.empirical_coverage:.2%}")
        if result.calibration_quantile is not None:
            print(f"Калибровочный квантиль (calibration quantile / offset): {result.calibration_quantile:.6f}")
        
        # Сохранение состояния
        out_state_path = Path(args.out_state)
        out_state_path.parent.mkdir(parents=True, exist_ok=True)
        service.save_state(out_state_path)
        print(f"Состояние сохранено в файл: {out_state_path}")
        
        # Также запишем текстовый отчет рядом
        report_txt_path = out_state_path.with_suffix(".txt")
        with open(report_txt_path, "w", encoding="utf-8") as rf:
            rf.write(f"Conformal Calibration Report\n")
            rf.write(f"============================\n")
            rf.write(f"Method: {service.config.method.name}\n")
            rf.write(f"Target Coverage: {service.config.coverage_target:.2%}\n")
            rf.write(f"Empirical Coverage: {result.empirical_coverage:.2% if result.empirical_coverage is not None else 'N/A'}\n")
            rf.write(f"Offset (Quantile): {result.calibration_quantile:.6f if result.calibration_quantile is not None else 'N/A'}\n")
            rf.write(f"Samples used: {result.samples_used}\n")
    else:
        print(f"Ошибка калибровки: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
