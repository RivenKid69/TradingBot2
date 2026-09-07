#!/usr/bin/env python3
"""
ГЛУБОКАЯ ПРОВЕРКА: Действительно ли есть проблема с индексами?

Этот тест проверяет РЕАЛЬНОЕ поведение кода для различных сценариев.
"""

import pandas as pd
import numpy as np
from feature_pipe import FeaturePipe
from core_config import FeatureSpec


def test_scenario_1_clean_data():
    """Сценарий 1: Чистые данные без NaN в price."""
    print("\n" + "=" * 80)
    print("СЦЕНАРИЙ 1: Чистые данные без NaN в обязательных полях")
    print("=" * 80)

    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 10,
            "ts_ms": list(range(0, 600_000, 60_000)),
            "close": [100.0 + i for i in range(10)],
        }
    )

    print(f"\nИсходные данные: {len(df)} строк")
    print(f"NaN в close: {df['close'].isna().sum()}")

    fp = FeaturePipe(spec=FeatureSpec(lookbacks_prices=[1, 5]), price_col="close")

    X = fp.transform_df(df)
    y = fp.make_targets(df)

    print(f"\nРезультаты:")
    print(f"  len(X) = {len(X)}")
    print(f"  len(y) = {len(y)}")
    print(f"  X.index: {list(X.index[:5])}... (первые 5)")
    print(f"  y.index: {list(y.index[:5])}... (первые 5)")
    print(f"  NaN в y: {y.isna().sum()}")

    if len(X) == len(y):
        print("  ✓ Размеры совпадают")
        if X.index.equals(y.index):
            print("  ✓ Индексы идентичны")
            return True
        else:
            print("  ⚠️  Индексы НЕ идентичны!")
            return False
    else:
        print(f"  ❌ Размеры НЕ совпадают: {len(X)} != {len(y)}")
        return False


def test_scenario_2_nan_in_middle():
    """Сценарий 2: NaN в price посередине данных."""
    print("\n" + "=" * 80)
    print("СЦЕНАРИЙ 2: NaN в price посередине данных")
    print("=" * 80)

    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 10,
            "ts_ms": list(range(0, 600_000, 60_000)),
            "close": [100.0, 101.0, 102.0, 103.0, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0],
            #                                       ^^^ NaN в строке 4
        }
    )

    print(f"\nИсходные данные: {len(df)} строк")
    print(
        f"NaN в close: {df['close'].isna().sum()} (строка {df[df['close'].isna()].index.tolist()})"
    )

    fp = FeaturePipe(spec=FeatureSpec(lookbacks_prices=[1, 5]), price_col="close")

    X = fp.transform_df(df)
    y = fp.make_targets(df)

    print(f"\nРезультаты:")
    print(f"  len(X) = {len(X)}")
    print(f"  len(y) = {len(y)}")
    print(f"  X.index: {list(X.index)}")
    print(f"  y.index: {list(y.index)}")
    print(f"  NaN в y: {y.isna().sum()}")

    # Проверка критической проблемы
    if len(X) != len(y):
        print(f"\n❌ КРИТИЧЕСКАЯ ПРОБЛЕМА:")
        print(f"   X и y имеют разные размеры: {len(X)} != {len(y)}")
        print(f"   Это означает, что apply_offline_features удалил строку с NaN,")
        print(f"   а make_targets НЕ удалил!")

        # Проверим, что произойдет при применении моего решения
        print(f"\n  Проверка моего решения:")
        valid_mask = y.notna()
        print(f"    Размер valid_mask: {len(valid_mask)}")
        print(f"    valid_mask.index: {list(valid_mask.index)}")

        try:
            X_filtered = X[valid_mask]
            print(f"    Pandas НЕ выдал ошибку")
            print(f"    len(X_filtered) = {len(X_filtered)}")
            print(f"    X_filtered.index: {list(X_filtered.index)}")

            # Проверим, правильно ли выровнялись данные
            print(f"\n  Проверка выравнивания:")
            print(f"    X имеет индексы: {list(X.index)}")
            print(f"    valid_mask имеет индексы: {list(valid_mask.index)}")
            print(f"    Pandas делает INNER JOIN по индексам")
            print(f"    Общие индексы: {sorted(set(X.index) & set(valid_mask.index))}")

            # КРИТИЧЕСКАЯ ПРОВЕРКА: соответствуют ли значения?
            # После удаления строки 4 в X, индексы стали 0,1,2,3,4,5,6,7,8 (reset_index!)
            # В y индексы 0,1,2,3,4,5,6,7,8,9 (НЕ reset_index!)
            # Когда мы делаем X[valid_mask], pandas выравнивает по индексам
            # Но индекс 4 в X - это ДРУГАЯ строка, чем индекс 4 в y!

            print(f"\n  ⚠️  ПОТЕНЦИАЛЬНАЯ ПРОБЛЕМА ВЫРАВНИВАНИЯ:")
            print(f"    Индекс 4 в X соответствует СТРОКЕ 5 исходных данных (после reset_index)")
            print(f"    Индекс 4 в y соответствует СТРОКЕ 4 исходных данных (с NaN)")
            print(f"    Это РАЗНЫЕ строки!")

        except Exception as e:
            print(f"    Pandas ВЫДАЛ ошибку: {e}")

        return False
    else:
        print("  ✓ Размеры совпадают (неожиданно!)")
        return True


def test_scenario_3_with_timestamps():
    """Сценарий 3: Проверка выравнивания по временным меткам."""
    print("\n" + "=" * 80)
    print("СЦЕНАРИЙ 3: Проверка выравнивания через временные метки")
    print("=" * 80)

    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 10,
            "ts_ms": list(range(0, 600_000, 60_000)),
            "close": [100.0, 101.0, 102.0, 103.0, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0],
        }
    )

    print(f"\nИсходные данные: {len(df)} строк")
    print("Строка с NaN:")
    nan_row = df[df["close"].isna()]
    print(nan_row[["symbol", "ts_ms", "close"]])

    fp = FeaturePipe(spec=FeatureSpec(lookbacks_prices=[1, 5]), price_col="close")

    X = fp.transform_df(df)
    y = fp.make_targets(df)

    print(f"\nПроверка соответствия через ts_ms:")

    if "ts_ms" in X.columns:
        # Проверим, какая строка удалена
        original_ts = set(df["ts_ms"].values)
        X_ts = set(X["ts_ms"].values)
        y_ts = set(df["ts_ms"].values)  # y сохраняет все ts

        missing_in_X = original_ts - X_ts
        print(f"  Временные метки отсутствующие в X: {missing_in_X}")
        print(f"  Это соответствует строке с NaN? {240000 in missing_in_X}")

        # Проверим, правильно ли выравниваются данные по моему методу
        if len(X) != len(y):
            print(f"\n  Применение моего решения:")
            valid_mask = y.notna()

            # КЛЮЧЕВАЯ ПРОБЛЕМА: valid_mask использует индексы y (0-9),
            # а X имеет индексы 0-8 после reset_index
            # Pandas выровняет по индексам, но это НЕПРАВИЛЬНО!

            X_filtered = X[valid_mask]
            y_filtered = y[valid_mask]

            print(f"    len(X_filtered) = {len(X_filtered)}")
            print(f"    len(y_filtered) = {len(y_filtered)}")

            # Проверим через ts_ms
            if "ts_ms" in X_filtered.columns:
                X_filtered_ts = set(X_filtered["ts_ms"].values)
                print(f"    ts_ms в X_filtered: {sorted(X_filtered_ts)[:5]}... (первые 5)")

                # Проверим, есть ли удаленная временная метка
                if missing_in_X & X_filtered_ts:
                    print(f"    ❌ ОШИБКА: Удаленная временная метка ПРИСУТСТВУЕТ в X_filtered!")
                else:
                    print(f"    ✓ Удаленная временная метка отсутствует в X_filtered")
    else:
        print("  ⚠️  ts_ms не сохраняется в X")

    return len(X) == len(y)


def test_scenario_4_realistic():
    """Сценарий 4: Реалистичный случай - только NaN в последних строках."""
    print("\n" + "=" * 80)
    print("СЦЕНАРИЙ 4: Реалистичный случай - NaN только в последних строках (shift)")
    print("=" * 80)

    # В реальности, если в исходных данных нет NaN в price,
    # то X и y будут одного размера, НО:
    # y будет иметь NaN в последней строке каждого символа (из-за shift(-1))

    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5 + ["ETHUSDT"] * 5,
            "ts_ms": list(range(0, 300_000, 60_000)) * 2,
            "close": [100.0 + i for i in range(5)] + [200.0 + i for i in range(5)],
        }
    )

    print(f"\nИсходные данные: {len(df)} строк, 2 символа")
    print(f"NaN в close: {df['close'].isna().sum()}")

    fp = FeaturePipe(spec=FeatureSpec(lookbacks_prices=[1, 5]), price_col="close")

    X = fp.transform_df(df)
    y = fp.make_targets(df)

    print(f"\nРезультаты:")
    print(f"  len(X) = {len(X)}")
    print(f"  len(y) = {len(y)}")
    print(f"  X.index: {list(X.index)}")
    print(f"  y.index: {list(y.index)}")
    print(f"  NaN в y: {y.isna().sum()} (должно быть 2 - по одному на символ)")

    if len(X) == len(y):
        print("  ✓ Размеры совпадают")

        # Проверим, что индексы совпадают
        if X.index.equals(y.index):
            print("  ✓ Индексы идентичны")

            # Теперь применим мое решение
            valid_mask = y.notna()
            X_filtered = X[valid_mask].reset_index(drop=True)
            y_filtered = y[valid_mask].reset_index(drop=True)

            print(f"\n  После фильтрации NaN:")
            print(f"    len(X_filtered) = {len(X_filtered)}")
            print(f"    len(y_filtered) = {len(y_filtered)}")
            print(f"    Удалено строк: {len(y) - len(y_filtered)} (ожидалось 2)")

            if len(X_filtered) == len(X) - 2 and len(y_filtered) == len(y) - 2:
                print("    ✓ Правильное количество строк удалено")
                return True
            else:
                print("    ❌ Неправильное количество строк удалено")
                return False
        else:
            print("  ❌ Индексы НЕ идентичны")
            return False
    else:
        print(f"  ❌ Размеры НЕ совпадают")
        return False


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("ГЛУБОКАЯ ПРОВЕРКА РЕШЕНИЯ")
    print("=" * 80)

    results = []

    results.append(("Чистые данные", test_scenario_1_clean_data()))
    results.append(("NaN в middle", test_scenario_2_nan_in_middle()))
    results.append(("Проверка через ts", test_scenario_3_with_timestamps()))
    results.append(("Реалистичный случай", test_scenario_4_realistic()))

    print("\n" + "=" * 80)
    print("ИТОГИ:")
    print("=" * 80)

    for name, success in results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")

    if all(r[1] for r in results):
        print("\n✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
        print("\nЗаключение: Мое решение КОРРЕКТНО для реалистичных случаев")
        print("(когда исходные данные не имеют NaN в обязательных полях)")
    else:
        print("\n❌ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")
        print("\nЗаключение: Нужно доработать решение для edge cases")
        print("(когда исходные данные имеют NaN в обязательных полях)")
