#!/usr/bin/env python3
"""
Тест для проверки исправления КРИТИЧЕСКОГО БАГА #1: ret_4h = 0

Проверяем, что все returns (ret_4h, ret_12h, ret_24h, ret_200h)
вычисляются правильно после исправления.
"""

import math
from transformers import FeatureSpec, OnlineFeatureTransformer


def test_all_returns():
    """
    Проверяет, что все returns вычисляются корректно
    """
    # Создаем spec для 4h таймфрейма
    spec = FeatureSpec(
        lookbacks_prices=[240, 720, 1440, 12000],  # 4h, 12h, 24h, 200h в минутах
        bar_duration_minutes=240,
        rsi_period=14
    )

    print("=" * 80)
    print("ТЕСТ: Проверка всех returns после исправления бага")
    print("=" * 80)
    print(f"\nSpec после инициализации:")
    print(f"  lookbacks_prices (бары):  {spec.lookbacks_prices}")
    print(f"  _lookbacks_prices_minutes: {spec._lookbacks_prices_minutes}")
    print()

    # Создаем трансформер
    transformer = OnlineFeatureTransformer(spec)

    # Симулируем последовательность цен с ростом
    prices = []
    base_price = 29000.0
    growth_rate = 0.003  # ~0.3% рост каждые 4 часа
    num_bars = 60  # 60 баров = 10 дней

    for i in range(num_bars):
        price = base_price * (1 + growth_rate) ** i
        prices.append(price)

    # Обрабатываем все бары
    print("Обработка баров...")
    all_feats = []
    for i, price in enumerate(prices):
        feats = transformer.update(
            symbol="BTCUSDT",
            ts_ms=1000000000 + i * 240 * 60 * 1000,
            close=price,
            volume=1000.0,
            open_price=price - 50,
            high=price + 50,
            low=price - 50
        )
        all_feats.append(feats)

    # Проверяем последний бар (должны быть доступны все returns)
    last_feats = all_feats[-1]

    print("\nПоследний бар (индекс {}):".format(len(prices) - 1))
    print("-" * 80)

    # Проверяем каждый return
    expected_returns = {
        "ret_4h": (prices[-1], prices[-2]),    # 1 бар назад
        "ret_12h": (prices[-1], prices[-4]),   # 3 бара назад
        "ret_24h": (prices[-1], prices[-7]),   # 6 баров назад
        "ret_200h": (prices[-1], prices[-51]), # 50 баров назад
    }

    all_passed = True

    for ret_name, (current, old) in expected_returns.items():
        if ret_name in last_feats:
            actual_ret = last_feats[ret_name]
            expected_ret = math.log(current / old)
            diff = abs(actual_ret - expected_ret)

            status = "✓" if diff < 0.0001 else "✗"
            if status == "✗":
                all_passed = False

            print(f"{ret_name:12s}: {actual_ret:10.6f} (ожидание: {expected_ret:10.6f}) {status}")

            # Проверка, что ret НЕ равен нулю (критический баг)
            if actual_ret == 0.0:
                print(f"  ❌ ОШИБКА: {ret_name} равен нулю!")
                all_passed = False
        else:
            print(f"{ret_name:12s}: ОТСУТСТВУЕТ ❌")
            all_passed = False

    print("-" * 80)

    # Проверяем средние бары (где должны быть доступны некоторые returns)
    print("\nПроверка промежуточных баров:")
    print("-" * 80)

    test_indices = [1, 3, 6, 10, 20]
    for idx in test_indices:
        if idx >= len(all_feats):
            continue

        feats = all_feats[idx]
        print(f"\nБар {idx} (цена: {prices[idx]:.2f}):")

        # ret_4h должен быть доступен с бара 1
        if idx >= 1:
            if "ret_4h" in feats:
                actual_ret = feats["ret_4h"]
                expected_ret = math.log(prices[idx] / prices[idx - 1])
                diff = abs(actual_ret - expected_ret)
                status = "✓" if diff < 0.0001 else "✗"
                if status == "✗" or actual_ret == 0.0:
                    all_passed = False
                print(f"  ret_4h:  {actual_ret:10.6f} (ожидание: {expected_ret:10.6f}) {status}")
            else:
                print(f"  ret_4h:  ОТСУТСТВУЕТ ❌")
                all_passed = False

        # ret_12h должен быть доступен с бара 3
        if idx >= 3:
            if "ret_12h" in feats:
                actual_ret = feats["ret_12h"]
                expected_ret = math.log(prices[idx] / prices[idx - 3])
                diff = abs(actual_ret - expected_ret)
                status = "✓" if diff < 0.0001 else "✗"
                if status == "✗" or actual_ret == 0.0:
                    all_passed = False
                print(f"  ret_12h: {actual_ret:10.6f} (ожидание: {expected_ret:10.6f}) {status}")

        # ret_24h должен быть доступен с бара 6
        if idx >= 6:
            if "ret_24h" in feats:
                actual_ret = feats["ret_24h"]
                expected_ret = math.log(prices[idx] / prices[idx - 6])
                diff = abs(actual_ret - expected_ret)
                status = "✓" if diff < 0.0001 else "✗"
                if status == "✗" or actual_ret == 0.0:
                    all_passed = False
                print(f"  ret_24h: {actual_ret:10.6f} (ожидание: {expected_ret:10.6f}) {status}")

    print("\n" + "=" * 80)

    if all_passed:
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ: returns вычисляются корректно!")
        return True
    else:
        print("❌ ТЕСТЫ НЕ ПРОЙДЕНЫ: обнаружены ошибки в вычислении returns")
        return False


if __name__ == "__main__":
    import sys
    success = test_all_returns()
    sys.exit(0 if success else 1)
