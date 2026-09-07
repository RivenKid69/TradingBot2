#!/usr/bin/env python3
"""
КРИТИЧЕСКИЙ ТЕСТ: Проверка проблемы с выравниванием индексов X и y

ПРОБЛЕМА:
- transform_df() вызывает apply_offline_features(), который делает dropna(subset=[ts, symbol, price])
- make_targets() работает с ИСХОДНЫМ df_raw и НЕ удаляет строки с NaN в price
- Это означает, что X и y могут иметь РАЗНОЕ КОЛИЧЕСТВО СТРОК ИЗНАЧАЛЬНО!

Пример:
  df_raw: 10 строк, но строка 5 имеет NaN в price
  X = transform_df(df_raw) -> 9 строк (строка 5 удалена)
  y = make_targets(df_raw) -> 10 строк (строка 5 имеет NaN target)

  Теперь если мы применим y.notna() как маску к X:
  X[y.notna()] - это применит 10-элементную маску к 9-элементному массиву!
  ЭТО ОШИБКА!
"""

import pandas as pd
import numpy as np
from feature_pipe import FeaturePipe
from core_config import FeatureSpec


def test_misaligned_indices():
    """Проверка проблемы с несоответствием индексов."""

    print("=" * 80)
    print("КРИТИЧЕСКИЙ ТЕСТ: Проблема выравнивания индексов X и y")
    print("=" * 80)

    # Создаем данные с NaN в price посередине
    df = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 10,
            "ts_ms": list(range(0, 600_000, 60_000)),
            "close": [100.0, 101.0, 102.0, 103.0, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0],
            #                                         ^^^ NaN в строке 4 (индекс 4)
        }
    )

    print("\nИсходные данные:")
    print(df)
    print(f"\nВсего строк: {len(df)}")
    print(f"Строк с NaN в close: {df['close'].isna().sum()}")

    # Создаем FeaturePipe
    fp = FeaturePipe(spec=FeatureSpec(lookbacks_prices=[1, 5]), price_col="close")

    # 1. Создаем X через transform_df (удалит строку с NaN)
    X = fp.transform_df(df)

    # 2. Создаем y через make_targets (НЕ удалит строку с NaN в price)
    y = fp.make_targets(df)

    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 80)
    print(f"Размер X: {len(X)}")
    print(f"Размер y: {len(y)}")
    print(f"Индексы X: {list(X.index)}")
    print(f"Индексы y: {list(y.index)}")

    if len(X) != len(y):
        print("\n❌ КРИТИЧЕСКАЯ ПРОБЛЕМА ПОДТВЕРЖДЕНА!")
        print(f"   X и y имеют разное количество строк: {len(X)} != {len(y)}")
        print("\n   Это означает, что мое решение с y.notna() как маской НЕ РАБОТАЕТ!")
        print("   Нельзя применить маску размером {} к массиву размером {}!".format(len(y), len(X)))

        # Попробуем применить мою маску и посмотрим что произойдет
        try:
            valid_mask = y.notna()
            print(f"\n   Размер маски: {len(valid_mask)}")
            print(f"   Попытка применить маску к X...")
            X_filtered = X[valid_mask]
            print(f"   НЕОЖИДАННО: Pandas не выдал ошибку!")
            print(f"   Размер X_filtered: {len(X_filtered)}")
            print(f"   НО это может быть НЕПРАВИЛЬНОЕ выравнивание!")
        except Exception as e:
            print(f"   Ошибка при применении маски: {e}")

        return False
    else:
        print("\n✓ X и y имеют одинаковое количество строк")
        print("  Но нужно проверить, что индексы совпадают!")

        if not X.index.equals(y.index):
            print("\n⚠️  ВНИМАНИЕ: Индексы X и y НЕ СОВПАДАЮТ!")
            print(f"   Разница: {set(X.index) ^ set(y.index)}")
            return False
        else:
            print("  Индексы совпадают - всё ОК")
            return True


def test_what_happens_with_boolean_indexing():
    """Проверка поведения pandas при несоответствии размеров."""

    print("\n" + "=" * 80)
    print("ТЕСТ: Что происходит при применении маски неправильного размера?")
    print("=" * 80)

    # Создаем DataFrame размером 5
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

    # Создаем маску размером 7 (больше чем df)
    mask_larger = pd.Series([True, True, False, True, True, False, True])

    print(f"\nDataFrame размером {len(df)}:")
    print(df)
    print(f"\nМаска размером {len(mask_larger)}:")
    print(mask_larger)

    try:
        result = df[mask_larger]
        print(f"\n⚠️  Pandas НЕ ВЫДАЛ ОШИБКУ!")
        print(f"Результат (размер {len(result)}):")
        print(result)
        print("\nЭто означает, что pandas МОЛЧАЛИВО ОБРЕЗАЕТ МАСКУ!")
        print("Это может привести к НЕПРАВИЛЬНОМУ выравниванию данных!")
    except Exception as e:
        print(f"\nОшибка: {e}")

    # Теперь маска меньше чем df
    mask_smaller = pd.Series([True, False, True])
    print(f"\n\nМаска размером {len(mask_smaller)} (меньше df):")
    print(mask_smaller)

    try:
        result = df[mask_smaller]
        print(f"\nРезультат (размер {len(result)}):")
        print(result)
    except Exception as e:
        print(f"\n✓ Pandas ВЫДАЛ ОШИБКУ (правильно): {e}")


if __name__ == "__main__":
    print("\n")
    success = test_misaligned_indices()
    test_what_happens_with_boolean_indexing()

    print("\n" + "=" * 80)
    if success:
        print("ЗАКЛЮЧЕНИЕ: Индексы выравниваются корректно")
    else:
        print("ЗАКЛЮЧЕНИЕ: ОБНАРУЖЕНА КРИТИЧЕСКАЯ ПРОБЛЕМА С ВЫРАВНИВАНИЕМ!")
        print("\nНужно изменить решение:")
        print("1. Использовать ИНДЕКСЫ для выравнивания, а не позиции")
        print("2. Или убедиться, что X и y создаются из ОДНОГО И ТОГО ЖЕ набора строк")
    print("=" * 80)
