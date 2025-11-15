#!/usr/bin/env python3
"""Простой тест для проверки логики tbr_momentum без зависимостей."""

def test_momentum_logic():
    """Проверка логики вычисления momentum."""
    print("Тест логики momentum вычисления:\n")

    # Симуляция ratio_list с 10 значениями
    ratio_list = [0.5, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68]

    print(f"ratio_list = {ratio_list}")
    print(f"len(ratio_list) = {len(ratio_list)}\n")

    # Проверим разные окна
    windows = [1, 2, 3, 6]
    window_names = ["4h", "8h", "12h", "24h"]

    for window, name in zip(windows, window_names):
        print(f"Window {name} (window={window} bars):")

        if len(ratio_list) >= window + 1:
            current = ratio_list[-1]
            past = ratio_list[-(window + 1)]
            momentum = current - past

            print(f"  current (ratio_list[-1]) = {current}")
            print(f"  past (ratio_list[-{window+1}]) = {past}")
            print(f"  momentum = {current} - {past} = {momentum}")
            print(f"  ✓ OK\n")
        else:
            print(f"  ✗ Not enough data (need {window+1}, have {len(ratio_list)})\n")

    # Проверим граничный случай с малым количеством данных
    print("\n" + "="*60)
    print("Граничный случай: только 2 значения")
    print("="*60 + "\n")

    small_list = [0.5, 0.52]
    print(f"ratio_list = {small_list}")

    for window, name in zip(windows, window_names):
        print(f"\nWindow {name} (window={window} bars):")

        if len(small_list) >= window + 1:
            current = small_list[-1]
            past = small_list[-(window + 1)]
            momentum = current - past
            print(f"  momentum = {momentum}")
            print(f"  ✓ OK")
        else:
            print(f"  ✗ NaN (need {window+1} values, have {len(small_list)})")

    # Проверим потенциальную проблему
    print("\n\n" + "="*60)
    print("ПОТЕНЦИАЛЬНАЯ ПРОБЛЕМА:")
    print("="*60 + "\n")

    print("Если tbr_momentum признаки заполняются NaN, это может быть из-за:")
    print("1. Недостаточно данных в начале (нужно window+1 значений)")
    print("2. Volume данные отсутствуют (taker_buy_ratio не вычисляется)")
    print("3. Индексация deque может быть неправильной\n")

    # Проверим индексацию deque
    from collections import deque

    print("Проверка индексации deque:")
    dq = deque([0.5, 0.52, 0.54, 0.56, 0.58, 0.60], maxlen=10)
    print(f"deque = {list(dq)}")

    # Конвертируем в список
    ratio_list_from_deque = list(dq)
    print(f"list(deque) = {ratio_list_from_deque}")

    window = 2
    print(f"\nДля window={window}:")
    print(f"  ratio_list_from_deque[-1] = {ratio_list_from_deque[-1]}")
    print(f"  ratio_list_from_deque[-(window+1)] = ratio_list_from_deque[-{window+1}] = {ratio_list_from_deque[-(window+1)]}")

    momentum = ratio_list_from_deque[-1] - ratio_list_from_deque[-(window+1)]
    print(f"  momentum = {ratio_list_from_deque[-1]} - {ratio_list_from_deque[-(window+1)]} = {momentum}")

    print("\n" + "="*60)
    print("✅ ВСЕ ПРОВЕРКИ ЗАВЕРШЕНЫ")
    print("="*60)

if __name__ == "__main__":
    test_momentum_logic()
