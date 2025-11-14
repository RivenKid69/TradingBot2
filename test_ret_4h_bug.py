#!/usr/bin/env python3
"""
Тест для воспроизведения КРИТИЧЕСКОГО БАГА #1: ret_4h ВСЕГДА = 0

Проблема: При lb=1 (для ret_4h в 4h таймфрейме), window содержит только текущую
цену, поэтому first = current_price, и ret = log(price/price) = 0

Ожидаемое поведение: ret_4h должен быть log(current_price / price_4h_ago)
"""

import math
from transformers import FeatureSpec, OnlineFeatureTransformer


def test_ret_4h_not_zero():
    """
    Демонстрирует, что ret_4h ВСЕГДА равен 0 при текущей реализации
    """
    # Создаем spec для 4h таймфрейма
    spec = FeatureSpec(
        lookbacks_prices=[240, 720, 1440, 12000],  # 4h, 12h, 24h, 200h в минутах
        bar_duration_minutes=240,
        rsi_period=14,
        yang_zhang_windows=None,
        parkinson_windows=None
    )

    print("=" * 80)
    print("ТЕСТ: ret_4h ВСЕГДА РАВЕН НУЛЮ (КРИТИЧЕСКИЙ БАГ #1)")
    print("=" * 80)
    print(f"\nSpec после инициализации:")
    print(f"  lookbacks_prices (бары):  {spec.lookbacks_prices}")
    print(f"  _lookbacks_prices_minutes: {spec._lookbacks_prices_minutes}")
    print()

    # Создаем трансформер
    transformer = OnlineFeatureTransformer(spec)

    # Симулируем последовательность цен с ростом ~0.34% каждые 4 часа
    prices = [29000.0, 29100.0, 29200.0, 29300.0]
    expected_returns = [
        0.0,  # Первый бар: нет предыдущей цены
        math.log(29100.0 / 29000.0),  # ~0.00344
        math.log(29200.0 / 29100.0),  # ~0.00343
        math.log(29300.0 / 29200.0),  # ~0.00342
    ]

    print("Побарная обработка:")
    print("-" * 80)
    print(f"{'Бар':<5} {'Цена':<10} {'ret_4h (факт)':<20} {'ret_4h (ожидание)':<20} {'Статус':<10}")
    print("-" * 80)

    all_passed = True

    for i, price in enumerate(prices):
        # Обновляем трансформер
        feats = transformer.update(
            symbol="BTCUSDT",
            ts_ms=1000000000 + i * 240 * 60 * 1000,  # Каждые 4 часа
            close=price,
            volume=1000.0,
            taker_buy_base=500.0,
            open_price=price - 50,
            high=price + 50,
            low=price - 50
        )

        actual_ret = feats.get("ret_4h", float('nan'))
        expected_ret = expected_returns[i]

        # Проверка: для i=0 ret должен быть 0 (нет предыдущих данных)
        # Для i>0 ret должен быть != 0
        if i == 0:
            status = "✓" if math.isnan(actual_ret) or actual_ret == 0.0 else "✗"
            if status == "✗":
                all_passed = False
        else:
            # БАГ: ret_4h всегда 0 вместо ожидаемого значения
            status = "✓" if abs(actual_ret - expected_ret) < 0.0001 else "✗ БАГ"
            if status != "✓":
                all_passed = False

        print(f"{i:<5} {price:<10.1f} {actual_ret:<20.6f} {expected_ret:<20.6f} {status:<10}")

    print("-" * 80)
    print()

    # Детальная диагностика последнего бара
    print("ДЕТАЛЬНАЯ ДИАГНОСТИКА (Бар 3):")
    print("-" * 80)

    # Получаем внутреннее состояние
    state = transformer._state.get("BTCUSDT", {})
    prices_deque = list(state.get("prices", []))

    print(f"Deque prices: {prices_deque}")
    print(f"Длина deque:  {len(prices_deque)}")
    print()

    # Симулируем логику из OnlineFeatureTransformer.update() для ret_4h
    lb = spec.lookbacks_prices[0]  # Первый lookback (для ret_4h) = 1 бар
    print(f"lb (lookback в барах для ret_4h): {lb}")
    print()

    if len(prices_deque) >= lb:
        window = prices_deque[-lb:]
        print(f"window = prices_deque[-{lb}:] = {window}")
        print(f"Длина window: {len(window)}")
        print()

        first = float(window[0])
        current_price = prices[-1]

        print(f"first = window[0] = {first}  # ТЕКУЩАЯ ЦЕНА!")
        print(f"current_price     = {current_price}  # ТА ЖЕ ЦЕНА!")
        print()

        ret_actual = math.log(current_price / first) if first > 0 else 0.0
        print(f"ret_4h = log({current_price} / {first}) = log(1) = {ret_actual}  # ❌ ВСЕГДА НОЛЬ!")
        print()

        # Правильное вычисление
        if len(prices_deque) > lb:
            price_lb_ago = float(prices_deque[-(lb+1)])
            ret_correct = math.log(current_price / price_lb_ago)
            print(f"ПРАВИЛЬНОЕ ВЫЧИСЛЕНИЕ:")
            print(f"  price_{lb}_bars_ago = prices_deque[-{lb+1}] = {price_lb_ago}")
            print(f"  ret_4h = log({current_price} / {price_lb_ago}) = {ret_correct:.6f}  # ✓ ПРАВИЛЬНО!")

    print("=" * 80)

    if not all_passed:
        print("\n❌ ТЕСТ НЕ ПРОЙДЕН: БАГ ПОДТВЕРЖДЕН - ret_4h всегда равен 0")
        print("\nКОРНЕВАЯ ПРИЧИНА:")
        print("  Когда lb=1, window = seq[-1:] содержит только ТЕКУЩУЮ цену")
        print("  first = window[0] = current_price")
        print("  ret = log(current_price / current_price) = log(1) = 0")
        print("\nРЕШЕНИЕ:")
        print("  Нужно брать цену lb+1 элементов назад: seq[-(lb+1)]")
        print("  ret = log(current_price / seq[-(lb+1)])")
        return False
    else:
        print("\n✓ ТЕСТ ПРОЙДЕН: ret_4h вычисляется корректно")
        return True


if __name__ == "__main__":
    import sys
    success = test_ret_4h_not_zero()
    sys.exit(0 if success else 1)
