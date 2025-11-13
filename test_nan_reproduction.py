#!/usr/bin/env python3
"""
Тест для воспроизведения проблемы с NaN значениями в полосах Боллинджера.
"""

import math


def test_nan_propagation():
    """Проверяем, как NaN распространяются через вычисления."""

    print("=" * 80)
    print("ТЕСТ 1: Проверка распространения NaN через операции")
    print("=" * 80)

    # Эмуляция ранних баров, когда индикаторы еще не готовы
    bb_lower = float('nan')
    bb_upper = float('nan')
    price = 100.0

    # Вычисления из obs_builder.pyx строка 168
    bb_width = bb_upper - bb_lower
    print(f"bb_upper - bb_lower = {bb_upper} - {bb_lower} = {bb_width}")
    print(f"Is bb_width NaN? {math.isnan(bb_width)}")

    bb_squeeze = math.tanh((bb_upper - bb_lower) / (price + 1e-8))
    print(f"bb_squeeze = tanh(({bb_upper} - {bb_lower}) / {price}) = {bb_squeeze}")
    print(f"Is bb_squeeze NaN? {math.isnan(bb_squeeze)}")

    print()

    # Проверка clipf с NaN
    print("=" * 80)
    print("ТЕСТ 2: Поведение _clipf с NaN")
    print("=" * 80)

    # Эмуляция логики _clipf
    value = bb_squeeze
    lower = -1.0
    upper = 2.0

    print(f"value = {value}")
    print(f"value < lower? {value < lower}")
    print(f"value > upper? {value > upper}")

    # Реальная логика _clipf
    if value < lower:
        result = lower
    elif value > upper:
        result = upper
    else:
        result = value

    print(f"Результат _clipf: {result}")
    print(f"Is result NaN? {math.isnan(result)}")

    print()

    # Проверка других индикаторов
    print("=" * 80)
    print("ТЕСТ 3: Другие индикаторы с NaN")
    print("=" * 80)

    momentum = float('nan')  # Первые 10 баров
    macd = float('nan')      # Первые ~26 баров
    macd_signal = float('nan')

    price_momentum = math.tanh(momentum / (price * 0.01 + 1e-8))
    trend_strength = math.tanh((macd - macd_signal) / (price * 0.01 + 1e-8))

    print(f"momentum = {momentum}")
    print(f"price_momentum = tanh({momentum} / {price * 0.01}) = {price_momentum}")
    print(f"Is price_momentum NaN? {math.isnan(price_momentum)}")
    print()

    print(f"macd = {macd}, macd_signal = {macd_signal}")
    print(f"trend_strength = tanh(({macd} - {macd_signal}) / {price * 0.01}) = {trend_strength}")
    print(f"Is trend_strength NaN? {math.isnan(trend_strength)}")

    print()

    # Итоговая проверка: что попадает в observation
    print("=" * 80)
    print("ИТОГ: NaN в observation vector")
    print("=" * 80)

    observation = []

    # Прямое присваивание индикаторов (строки 99-112)
    observation.extend([
        42.0,  # rsi14 (может быть готов раньше)
        macd,  # NaN!
        macd_signal,  # NaN!
        momentum,  # NaN!
        15.0,  # atr (может быть готов раньше)
        20.0,  # cci (может быть готов раньше)
        1000.0,  # obv
    ])

    # Производные признаки (строки 160-177)
    observation.extend([
        price_momentum,  # NaN!
        bb_squeeze,      # NaN!
        trend_strength,  # NaN!
    ])

    nan_count = sum(1 for x in observation if isinstance(x, float) and math.isnan(x))
    print(f"Всего признаков: {len(observation)}")
    print(f"Из них NaN: {nan_count}")
    print(f"Observation vector: {observation}")

    if nan_count > 0:
        print("\n⚠️  ПРОБЛЕМА ПОДТВЕРЖДЕНА: NaN значения попадают в observation!")
        return False
    else:
        print("\n✓ Проблем не обнаружено")
        return True


def test_existing_solution():
    """Проверяем, работает ли существующее решение для некоторых случаев."""

    print("\n" + "=" * 80)
    print("ТЕСТ 4: Существующее решение для bb_lower/bb_upper (строки 180-195)")
    print("=" * 80)

    bb_lower = float('nan')
    bb_upper = float('nan')
    price = 100.0

    # Существующий код из obs_builder.pyx
    bb_width = bb_upper - bb_lower
    bb_valid = not math.isnan(bb_lower)
    min_bb_width = price * 0.0001

    if (not bb_valid) or bb_width <= min_bb_width:
        feature_val = 0.5
    else:
        feature_val = max(-1.0, min(2.0, (price - bb_lower) / (bb_width + 1e-9)))

    print(f"bb_valid = not isnan({bb_lower}) = {bb_valid}")
    print(f"feature_val (позиция в полосах) = {feature_val}")
    print(f"Is feature_val NaN? {math.isnan(feature_val)}")

    # Вторая часть
    if bb_valid:
        feature_val2 = max(0.0, min(10.0, bb_width / (price + 1e-8)))
    else:
        feature_val2 = 0.0

    print(f"feature_val2 (ширина полос) = {feature_val2}")
    print(f"Is feature_val2 NaN? {math.isnan(feature_val2)}")

    print("\n✓ Существующее решение РАБОТАЕТ для строк 180-195")
    print("✗ НО не работает для строк 168-170 (bb_squeeze)!")


if __name__ == "__main__":
    test_nan_propagation()
    test_existing_solution()

    print("\n" + "=" * 80)
    print("ВЫВОДЫ")
    print("=" * 80)
    print("""
1. ✓ Проблема РЕАЛЬНА и воспроизводима
2. ✓ NaN появляются в первые N баров пока индикаторы не накопят историю:
   - Bollinger Bands: первые 20 баров
   - Momentum: первые 10 баров
   - MACD: первые ~26 баров
   - RSI: первые ~14 баров
3. ✓ _clipf НЕ обрабатывает NaN (сравнения с NaN всегда False)
4. ✓ get_or_nan просто передает NaN дальше
5. ✓ Существующее решение в строках 180-195 РАБОТАЕТ
6. ✗ НО много других мест где NaN попадают в observation:
   - Строки 99-112: прямое присваивание индикаторов
   - Строка 168-170: bb_squeeze = tanh((bb_upper - bb_lower) / price)
   - Строка 160-162: price_momentum = tanh(momentum / price)
   - Строка 175-177: trend_strength = tanh((macd - macd_signal) / price)

НУЖНО: Комплексное решение для ВСЕХ индикаторов, не только bb_lower/bb_upper
    """)
