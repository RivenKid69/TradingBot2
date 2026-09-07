#!/usr/bin/env python3
"""
КРИТИЧЕСКАЯ ПРОВЕРКА: Атака на мое исправление

Проверяем:
1. Семантику lookback: правильно ли я понимаю, что такое lb?
2. Индексацию: seq[-(lb+1)] действительно дает цену lb баров назад?
3. Edge cases: не упадет ли код на граничных случаях?
4. SMA: не сломал ли я вычисление SMA?
5. Все lookbacks: работают ли ret_12h, ret_24h, ret_200h правильно?
"""

import math
from transformers import FeatureSpec, OnlineFeatureTransformer


def test_semantic_correctness():
    """
    Проверка семантики: ret_4h должен быть доходностью ЗА ПОСЛЕДНИЕ 4 часа,
    то есть log(цена_сейчас / цена_4h_назад)
    """
    print("=" * 80)
    print("ТЕСТ 1: Проверка семантики lookback")
    print("=" * 80)

    spec = FeatureSpec(
        lookbacks_prices=[240],  # 4h = 240 минут = 1 бар в 4h таймфрейме
        bar_duration_minutes=240,
        rsi_period=14,
    )

    print(f"lookbacks_prices (минуты): {spec._lookbacks_prices_minutes}")
    print(f"lookbacks_prices (бары):   {spec.lookbacks_prices}")
    print()

    transformer = OnlineFeatureTransformer(spec)

    # Бар 0: Цена 100
    feats0 = transformer.update(
        symbol="TEST", ts_ms=0, close=100.0, open_price=100.0, high=100.0, low=100.0
    )

    # Бар 1: Цена 110 (рост на 10%)
    feats1 = transformer.update(
        symbol="TEST", ts_ms=240 * 60 * 1000, close=110.0, open_price=110.0, high=110.0, low=110.0
    )

    print("Бар 0 (цена=100):")
    print(f"  ret_4h: {feats0.get('ret_4h', 'НЕТ')}")
    print()

    print("Бар 1 (цена=110, рост на 10%):")
    ret_4h = feats1.get("ret_4h", None)
    print(f"  ret_4h: {ret_4h}")

    if ret_4h is not None:
        expected = math.log(110.0 / 100.0)  # log(1.1) ≈ 0.0953
        print(f"  Ожидание: {expected:.6f}")
        print(f"  Разница: {abs(ret_4h - expected):.9f}")

        if abs(ret_4h - expected) < 1e-6:
            print("  ✅ СЕМАНТИКА ПРАВИЛЬНАЯ: ret_4h = log(цена_текущая / цена_4h_назад)")
            return True
        else:
            print("  ❌ ОШИБКА СЕМАНТИКИ!")
            return False
    else:
        print("  ❌ ret_4h отсутствует!")
        return False


def test_indexing_correctness():
    """
    Проверка индексации: seq[-(lb+1)] действительно дает правильную цену?
    """
    print("\n" + "=" * 80)
    print("ТЕСТ 2: Проверка индексации seq[-(lb+1)]")
    print("=" * 80)

    spec = FeatureSpec(
        lookbacks_prices=[240, 720, 1440], bar_duration_minutes=240, rsi_period=14  # 4h, 12h, 24h
    )

    print(f"lookbacks_prices (бары): {spec.lookbacks_prices}")
    print()

    transformer = OnlineFeatureTransformer(spec)

    # Создаем последовательность цен
    prices = [100, 102, 104, 106, 108, 110, 112, 114]

    for i, price in enumerate(prices):
        transformer.update(
            symbol="TEST",
            ts_ms=i * 240 * 60 * 1000,
            close=price,
            open_price=price,
            high=price,
            low=price,
        )

    # Проверяем последний бар (индекс 7, цена 114)
    feats = transformer.update(
        symbol="TEST", ts_ms=8 * 240 * 60 * 1000, close=116, open_price=116, high=116, low=116
    )

    # Теперь в деке: [100, 102, 104, 106, 108, 110, 112, 114, 116]
    # seq[-1] = 116 (текущая)
    # seq[-2] = 114 (1 бар назад) - для ret_4h (lb=1)
    # seq[-4] = 110 (3 бара назад) - для ret_12h (lb=3)
    # seq[-7] = 104 (6 баров назад) - для ret_24h (lb=6)

    all_passed = True

    # ret_4h: lb=1, должен быть log(116/114)
    if "ret_4h" in feats:
        expected_4h = math.log(116.0 / 114.0)
        actual_4h = feats["ret_4h"]
        passed = abs(actual_4h - expected_4h) < 1e-6
        status = "✅" if passed else "❌"
        print(f"ret_4h:  {actual_4h:.6f} (ожидание: {expected_4h:.6f}) {status}")
        print(f"         Должно быть: log(116/114) = log(current / seq[-2])")
        if not passed:
            all_passed = False
    else:
        print("ret_4h:  ❌ ОТСУТСТВУЕТ")
        all_passed = False

    print()

    # ret_12h: lb=3, должен быть log(116/110)
    if "ret_12h" in feats:
        expected_12h = math.log(116.0 / 110.0)
        actual_12h = feats["ret_12h"]
        passed = abs(actual_12h - expected_12h) < 1e-6
        status = "✅" if passed else "❌"
        print(f"ret_12h: {actual_12h:.6f} (ожидание: {expected_12h:.6f}) {status}")
        print(f"         Должно быть: log(116/110) = log(current / seq[-4])")
        if not passed:
            all_passed = False
    else:
        print("ret_12h: ❌ ОТСУТСТВУЕТ")
        all_passed = False

    print()

    # ret_24h: lb=6, должен быть log(116/104)
    if "ret_24h" in feats:
        expected_24h = math.log(116.0 / 104.0)
        actual_24h = feats["ret_24h"]
        passed = abs(actual_24h - expected_24h) < 1e-6
        status = "✅" if passed else "❌"
        print(f"ret_24h: {actual_24h:.6f} (ожидание: {expected_24h:.6f}) {status}")
        print(f"         Должно быть: log(116/104) = log(current / seq[-7])")
        if not passed:
            all_passed = False
    else:
        print("ret_24h: ❌ ОТСУТСТВУЕТ")
        all_passed = False

    return all_passed


def test_edge_cases():
    """
    Проверка граничных случаев
    """
    print("\n" + "=" * 80)
    print("ТЕСТ 3: Граничные случаи (edge cases)")
    print("=" * 80)

    spec = FeatureSpec(
        lookbacks_prices=[240, 720], bar_duration_minutes=240, rsi_period=14  # lb=1, lb=3
    )

    transformer = OnlineFeatureTransformer(spec)

    all_passed = True

    # Случай 1: Первый бар (len(seq)=1, lb=1)
    print("\nСлучай 1: Первый бар (len=1, lb=1)")
    feats1 = transformer.update(
        symbol="TEST", ts_ms=0, close=100.0, open_price=100.0, high=100.0, low=100.0
    )

    # len(seq)=1, условие: len(seq) > lb → 1 > 1 → False
    # ret_4h НЕ должен быть вычислен
    if "ret_4h" not in feats1:
        print("  ✅ ret_4h отсутствует (правильно, нужен минимум len=2)")
    else:
        print(f"  ❌ ret_4h присутствует: {feats1['ret_4h']} (ОШИБКА!)")
        all_passed = False

    # Случай 2: Второй бар (len(seq)=2, lb=1)
    print("\nСлучай 2: Второй бар (len=2, lb=1)")
    feats2 = transformer.update(
        symbol="TEST", ts_ms=240 * 60 * 1000, close=110.0, open_price=110.0, high=110.0, low=110.0
    )

    # len(seq)=2, условие: 2 > 1 → True
    # ret_4h должен быть вычислен: log(110/100)
    if "ret_4h" in feats2:
        expected = math.log(110.0 / 100.0)
        actual = feats2["ret_4h"]
        if abs(actual - expected) < 1e-6:
            print(f"  ✅ ret_4h = {actual:.6f} (правильно)")
        else:
            print(f"  ❌ ret_4h = {actual:.6f}, ожидание: {expected:.6f}")
            all_passed = False
    else:
        print("  ❌ ret_4h отсутствует (должен быть!)")
        all_passed = False

    # Случай 3: Третий бар (len=3, lb=3)
    print("\nСлучай 3: Третий бар (len=3, lb=3)")
    feats3 = transformer.update(
        symbol="TEST",
        ts_ms=2 * 240 * 60 * 1000,
        close=120.0,
        open_price=120.0,
        high=120.0,
        low=120.0,
    )

    # len(seq)=3, условие для ret_12h: 3 > 3 → False
    # ret_12h НЕ должен быть вычислен
    if "ret_12h" not in feats3:
        print("  ✅ ret_12h отсутствует (правильно, нужен минимум len=4)")
    else:
        print(f"  ❌ ret_12h присутствует: {feats3['ret_12h']} (ОШИБКА!)")
        all_passed = False

    # Случай 4: Четвертый бар (len=4, lb=3)
    print("\nСлучай 4: Четвертый бар (len=4, lb=3)")
    feats4 = transformer.update(
        symbol="TEST",
        ts_ms=3 * 240 * 60 * 1000,
        close=130.0,
        open_price=130.0,
        high=130.0,
        low=130.0,
    )

    # len(seq)=4, условие: 4 > 3 → True
    # ret_12h должен быть: log(130/100)
    if "ret_12h" in feats4:
        expected = math.log(130.0 / 100.0)
        actual = feats4["ret_12h"]
        if abs(actual - expected) < 1e-6:
            print(f"  ✅ ret_12h = {actual:.6f} (правильно)")
        else:
            print(f"  ❌ ret_12h = {actual:.6f}, ожидание: {expected:.6f}")
            all_passed = False
    else:
        print("  ❌ ret_12h отсутствует (должен быть!)")
        all_passed = False

    return all_passed


def test_sma_not_broken():
    """
    Проверка, что SMA не сломался
    """
    print("\n" + "=" * 80)
    print("ТЕСТ 4: SMA не сломался")
    print("=" * 80)

    spec = FeatureSpec(
        lookbacks_prices=[240, 720], bar_duration_minutes=240, rsi_period=14  # 4h, 12h
    )

    transformer = OnlineFeatureTransformer(spec)

    # Добавляем 5 баров
    prices = [100, 102, 104, 106, 108]
    for i, price in enumerate(prices):
        transformer.update(
            symbol="TEST",
            ts_ms=i * 240 * 60 * 1000,
            close=price,
            open_price=price,
            high=price,
            low=price,
        )

    feats = transformer.update(
        symbol="TEST", ts_ms=5 * 240 * 60 * 1000, close=110, open_price=110, high=110, low=110
    )

    all_passed = True

    # sma_240 (lb=1): среднее последнего 1 элемента = 110
    if "sma_240" in feats:
        expected_sma1 = 110.0
        actual_sma1 = feats["sma_240"]
        if abs(actual_sma1 - expected_sma1) < 1e-6:
            print(f"✅ sma_240 (lb=1): {actual_sma1:.2f} (правильно)")
        else:
            print(f"❌ sma_240: {actual_sma1:.2f}, ожидание: {expected_sma1:.2f}")
            all_passed = False
    else:
        print("❌ sma_240 отсутствует!")
        all_passed = False

    # sma_720 (lb=3): среднее последних 3 элементов = (106+108+110)/3 = 108
    if "sma_720" in feats:
        expected_sma3 = (106.0 + 108.0 + 110.0) / 3.0
        actual_sma3 = feats["sma_720"]
        if abs(actual_sma3 - expected_sma3) < 1e-6:
            print(f"✅ sma_720 (lb=3): {actual_sma3:.2f} (правильно)")
        else:
            print(f"❌ sma_720: {actual_sma3:.2f}, ожидание: {expected_sma3:.2f}")
            all_passed = False
    else:
        print("❌ sma_720 отсутствует!")
        all_passed = False

    return all_passed


def main():
    print("🔴 КРИТИЧЕСКАЯ ПРОВЕРКА ИСПРАВЛЕНИЯ")
    print("Атакуем свое решение, ищем ошибки...\n")

    results = {
        "Семантика lookback": test_semantic_correctness(),
        "Индексация": test_indexing_correctness(),
        "Граничные случаи": test_edge_cases(),
        "SMA не сломан": test_sma_not_broken(),
    }

    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ ПРОЙДЕН" if passed else "❌ НЕ ПРОЙДЕН"
        print(f"{test_name:30s}: {status}")

    print("\n" + "=" * 80)

    if all(results.values()):
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Исправление корректно!")
        return 0
    else:
        print("❌ ОБНАРУЖЕНЫ ОШИБКИ! Исправление требует доработки!")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
