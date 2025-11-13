#!/usr/bin/env python3
"""
Интеграционный тест для проверки создания GARCH признаков через make_features.
"""

import pandas as pd
import numpy as np
import math
import os
import tempfile
from transformers import FeatureSpec, apply_offline_features


def test_make_features_with_garch():
    """Тест создания признаков включая GARCH через apply_offline_features."""
    print("=== Интеграционный тест: создание признаков с GARCH ===\n")

    # Генерируем тестовые данные
    np.random.seed(42)
    n_points = 800
    base_price = 100.0
    symbol = "BTCUSDT"

    data = []
    ts_start = 1600000000000  # начальная временная метка

    for i in range(n_points):
        # Генерируем волатильность с кластеризацией
        if i % 400 < 200:
            vol = 0.02
        else:
            vol = 0.04

        log_return = np.random.normal(0, vol)
        close_price = base_price * math.exp(log_return)

        # Генерируем OHLC
        open_price = base_price
        high_price = max(open_price, close_price) * 1.005
        low_price = min(open_price, close_price) * 0.995

        data.append({
            "ts_ms": ts_start + i * 60000,  # каждую минуту
            "symbol": symbol,
            "price": close_price,
            "open": open_price,
            "high": high_price,
            "low": low_price,
        })

        base_price = close_price

    df = pd.DataFrame(data)

    print(f"Создан тестовый датафрейм:")
    print(f"  - Строк: {len(df)}")
    print(f"  - Колонки: {list(df.columns)}")
    print(f"  - Период: {n_points} минут\n")

    # Создаем спецификацию признаков с GARCH
    spec = FeatureSpec(
        lookbacks_prices=[5, 15, 60],
        rsi_period=14,
        yang_zhang_windows=[1440],  # 24ч
        parkinson_windows=[1440],   # 24ч
        garch_windows=[500, 720],   # 500 мин, 12ч
    )

    print("Спецификация признаков:")
    print(f"  - lookbacks_prices: {spec.lookbacks_prices}")
    print(f"  - rsi_period: {spec.rsi_period}")
    print(f"  - yang_zhang_windows: {spec.yang_zhang_windows}")
    print(f"  - parkinson_windows: {spec.parkinson_windows}")
    print(f"  - garch_windows: {spec.garch_windows}\n")

    # Применяем оффлайн расчет признаков
    print("Вычисляем признаки...")
    result_df = apply_offline_features(
        df,
        spec=spec,
        ts_col="ts_ms",
        symbol_col="symbol",
        price_col="price",
        open_col="open",
        high_col="high",
        low_col="low",
    )

    print(f"\nРезультат:")
    print(f"  - Строк: {len(result_df)}")
    print(f"  - Колонки: {len(result_df.columns)}")
    print(f"  - Список признаков: {list(result_df.columns)}\n")

    # Проверяем что GARCH признаки созданы
    garch_features = [col for col in result_df.columns if 'garch' in col.lower()]
    print(f"GARCH признаки: {garch_features}")

    if len(garch_features) == 0:
        print("  ❌ ОШИБКА: GARCH признаки не созданы!")
        return False

    # Проверяем последние строки где должны быть валидные GARCH значения
    last_rows = result_df.tail(10)
    print(f"\nПоследние 10 строк:")
    print(last_rows[['ts_ms', 'symbol', 'ref_price'] + garch_features].to_string())

    # Проверяем что есть непустые значения GARCH
    for feat in garch_features:
        non_nan_count = result_df[feat].notna().sum()
        valid_count = result_df[feat].apply(lambda x: not pd.isna(x) and not (isinstance(x, float) and math.isnan(x))).sum()
        print(f"\n{feat}:")
        print(f"  - Непустых значений: {non_nan_count}/{len(result_df)}")
        print(f"  - Валидных числовых значений: {valid_count}/{len(result_df)}")

        if valid_count > 0:
            valid_values = result_df[feat].dropna()
            valid_values = valid_values[~valid_values.apply(lambda x: math.isnan(x) if isinstance(x, float) else False)]
            if len(valid_values) > 0:
                print(f"  - Мин: {valid_values.min():.6f}")
                print(f"  - Макс: {valid_values.max():.6f}")
                print(f"  - Среднее: {valid_values.mean():.6f}")

    # Проверяем что другие признаки не пострадали (updated for 4h timeframe migration)
    expected_features = ['ref_price', 'rsi', 'sma_5', 'sma_15', 'sma_60',
                        'ret_4h', 'ret_24h',  # Changed from ret_5m, ret_15m, ret_60m
                        'yang_zhang_24h', 'parkinson_24h']

    print(f"\n\nПроверка других признаков:")
    all_present = True
    for feat in expected_features:
        if feat in result_df.columns:
            print(f"  ✓ {feat}")
        else:
            print(f"  ❌ {feat} - ОТСУТСТВУЕТ!")
            all_present = False

    if all_present:
        print("\n✓ Все ожидаемые признаки присутствуют")
    else:
        print("\n❌ ОШИБКА: некоторые признаки отсутствуют")
        return False

    print("\n" + "=" * 60)
    print("✓ ИНТЕГРАЦИОННЫЙ ТЕСТ УСПЕШНО ПРОЙДЕН")
    print("=" * 60)

    return True


def test_make_features_script():
    """Тест скрипта make_features.py с параметрами GARCH."""
    print("\n\n=== Тест скрипта make_features.py ===\n")

    # Создаем временный файл с данными
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_input = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.parquet', delete=False) as f:
        temp_output = f.name

    try:
        # Генерируем данные
        np.random.seed(123)
        n_points = 800
        base_price = 50000.0

        data = []
        ts_start = 1600000000000

        for i in range(n_points):
            vol = 0.02 if i % 400 < 200 else 0.03
            log_return = np.random.normal(0, vol)
            close_price = base_price * math.exp(log_return)
            open_price = base_price
            high_price = max(open_price, close_price) * 1.003
            low_price = min(open_price, close_price) * 0.997

            data.append({
                "ts_ms": ts_start + i * 60000,
                "symbol": "BTCUSDT",
                "price": close_price,
                "open": open_price,
                "high": high_price,
                "low": low_price,
            })

            base_price = close_price

        df = pd.DataFrame(data)
        df.to_csv(temp_input, index=False)

        print(f"Создан временный входной файл: {temp_input}")
        print(f"Временный выходной файл: {temp_output}\n")

        # Запускаем make_features.py
        import subprocess
        cmd = [
            "python", "make_features.py",
            "--in", temp_input,
            "--out", temp_output,
            "--lookbacks", "5,15,60",
            "--rsi-period", "14",
            "--yang-zhang-windows", "1440",
            "--parkinson-windows", "1440",
            "--garch-windows", "500,720",
            "--open-col", "open",
            "--high-col", "high",
            "--low-col", "low",
        ]

        print(f"Команда: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode != 0:
            print(f"\n❌ ОШИБКА: make_features.py завершился с кодом {result.returncode}")
            return False

        # Проверяем что выходной файл создан
        if not os.path.exists(temp_output):
            print(f"\n❌ ОШИБКА: выходной файл не создан: {temp_output}")
            return False

        # Читаем результат
        result_df = pd.read_parquet(temp_output)

        print(f"\nВыходной файл создан успешно:")
        print(f"  - Строк: {len(result_df)}")
        print(f"  - Колонок: {len(result_df.columns)}")
        print(f"  - Признаки: {list(result_df.columns)}")

        # Проверяем GARCH признаки
        garch_features = [col for col in result_df.columns if 'garch' in col.lower()]
        if len(garch_features) > 0:
            print(f"\n✓ GARCH признаки созданы: {garch_features}")
            for feat in garch_features:
                valid_count = result_df[feat].notna().sum()
                print(f"  {feat}: {valid_count} валидных значений")
        else:
            print("\n❌ ОШИБКА: GARCH признаки не найдены!")
            return False

        print("\n" + "=" * 60)
        print("✓ ТЕСТ СКРИПТА make_features.py УСПЕШНО ПРОЙДЕН")
        print("=" * 60)

        return True

    finally:
        # Удаляем временные файлы
        if os.path.exists(temp_input):
            os.unlink(temp_input)
        if os.path.exists(temp_output):
            os.unlink(temp_output)


def main():
    """Запуск всех интеграционных тестов."""
    print("\n" + "=" * 60)
    print("ИНТЕГРАЦИОННЫЕ ТЕСТЫ GARCH С make_features")
    print("=" * 60 + "\n")

    tests = [
        test_make_features_with_garch,
        test_make_features_script,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ ИСКЛЮЧЕНИЕ в {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"ИТОГО: {passed} успешно, {failed} неудачно")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
