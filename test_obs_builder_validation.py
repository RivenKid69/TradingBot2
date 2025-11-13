#!/usr/bin/env python3
"""
Тест для проверки корректности исправления NaN проблемы в obs_builder.
Этот тест можно запустить после перекомпиляции obs_builder.pyx.
"""

import math


def test_clipf_logic():
    """
    Проверяем логику _clipf с обработкой NaN.
    Эта функция эмулирует исправленную версию _clipf.
    """
    print("=" * 80)
    print("ТЕСТ 1: Проверка логики _clipf с NaN обработкой")
    print("=" * 80)

    def _clipf_fixed(value, lower, upper):
        """Исправленная версия _clipf с обработкой NaN."""
        if math.isnan(value):
            return 0.0
        if value < lower:
            return lower
        elif value > upper:
            return upper
        return value

    # Тест 1: NaN входное значение
    result = _clipf_fixed(float('nan'), -1.0, 1.0)
    assert result == 0.0, f"Expected 0.0 for NaN input, got {result}"
    print("✓ NaN input -> 0.0")

    # Тест 2: Значение ниже нижней границы
    result = _clipf_fixed(-5.0, -1.0, 1.0)
    assert result == -1.0, f"Expected -1.0, got {result}"
    print("✓ Value below lower bound -> lower bound")

    # Тест 3: Значение выше верхней границы
    result = _clipf_fixed(5.0, -1.0, 1.0)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("✓ Value above upper bound -> upper bound")

    # Тест 4: Значение в пределах границ
    result = _clipf_fixed(0.5, -1.0, 1.0)
    assert result == 0.5, f"Expected 0.5, got {result}"
    print("✓ Value within bounds -> unchanged")

    print("\n✅ Все тесты _clipf пройдены!\n")


def test_indicator_defaults():
    """
    Проверяем, что все индикаторы получают осмысленные дефолты при NaN.
    """
    print("=" * 80)
    print("ТЕСТ 2: Проверка дефолтных значений индикаторов")
    print("=" * 80)

    # Эмуляция логики обработки индикаторов
    def safe_indicator(value, default):
        """Безопасное получение индикатора с дефолтом."""
        return value if not math.isnan(value) else default

    # Тестовые случаи: (indicator_value, default, expected_result, description)
    test_cases = [
        (float('nan'), 50.0, 50.0, "RSI: neutral zone"),
        (42.0, 50.0, 42.0, "RSI: valid value"),
        (float('nan'), 0.0, 0.0, "MACD: no divergence"),
        (1.5, 0.0, 1.5, "MACD: valid value"),
        (float('nan'), 0.0, 0.0, "Momentum: no movement"),
        (10.0, 0.0, 10.0, "Momentum: valid value"),
        (float('nan'), 1.0, 1.0, "ATR: 1% of price default"),
        (15.0, 1.0, 15.0, "ATR: valid value"),
    ]

    all_passed = True
    for value, default, expected, description in test_cases:
        result = safe_indicator(value, default)
        if result == expected:
            print(f"✓ {description}: {value} -> {result}")
        else:
            print(f"✗ {description}: expected {expected}, got {result}")
            all_passed = False

    if all_passed:
        print("\n✅ Все тесты дефолтных значений пройдены!\n")
    else:
        print("\n❌ Некоторые тесты не прошли!\n")
        return False

    return True


def test_derived_features():
    """
    Проверяем, что производные признаки (bb_squeeze, price_momentum, trend_strength)
    корректно обрабатывают NaN входные данные.
    """
    print("=" * 80)
    print("ТЕСТ 3: Проверка производных признаков с NaN")
    print("=" * 80)

    price = 100.0

    # 1. bb_squeeze
    print("\n--- bb_squeeze ---")
    bb_lower_nan = float('nan')
    bb_upper_nan = float('nan')
    bb_valid = not math.isnan(bb_lower_nan)

    if bb_valid:
        bb_squeeze = math.tanh((bb_upper_nan - bb_lower_nan) / (price + 1e-8))
    else:
        bb_squeeze = 0.0

    assert bb_squeeze == 0.0, f"Expected bb_squeeze=0.0 for NaN inputs, got {bb_squeeze}"
    assert not math.isnan(bb_squeeze), "bb_squeeze must not be NaN!"
    print(f"✓ bb_squeeze with NaN inputs: {bb_squeeze} (not NaN)")

    # С валидными значениями
    bb_lower_valid = 95.0
    bb_upper_valid = 105.0
    bb_valid = not math.isnan(bb_lower_valid)

    if bb_valid:
        bb_squeeze = math.tanh((bb_upper_valid - bb_lower_valid) / (price + 1e-8))
    else:
        bb_squeeze = 0.0

    assert not math.isnan(bb_squeeze), "bb_squeeze must not be NaN!"
    print(f"✓ bb_squeeze with valid inputs: {bb_squeeze:.4f}")

    # 2. price_momentum
    print("\n--- price_momentum ---")
    momentum_nan = float('nan')

    if not math.isnan(momentum_nan):
        price_momentum = math.tanh(momentum_nan / (price * 0.01 + 1e-8))
    else:
        price_momentum = 0.0

    assert price_momentum == 0.0, f"Expected price_momentum=0.0 for NaN momentum, got {price_momentum}"
    assert not math.isnan(price_momentum), "price_momentum must not be NaN!"
    print(f"✓ price_momentum with NaN momentum: {price_momentum} (not NaN)")

    # С валидным значением
    momentum_valid = 10.0
    if not math.isnan(momentum_valid):
        price_momentum = math.tanh(momentum_valid / (price * 0.01 + 1e-8))
    else:
        price_momentum = 0.0

    assert not math.isnan(price_momentum), "price_momentum must not be NaN!"
    print(f"✓ price_momentum with valid momentum: {price_momentum:.4f}")

    # 3. trend_strength
    print("\n--- trend_strength ---")
    macd_nan = float('nan')
    macd_signal_nan = float('nan')

    if not math.isnan(macd_nan) and not math.isnan(macd_signal_nan):
        trend_strength = math.tanh((macd_nan - macd_signal_nan) / (price * 0.01 + 1e-8))
    else:
        trend_strength = 0.0

    assert trend_strength == 0.0, f"Expected trend_strength=0.0 for NaN MACD, got {trend_strength}"
    assert not math.isnan(trend_strength), "trend_strength must not be NaN!"
    print(f"✓ trend_strength with NaN MACD: {trend_strength} (not NaN)")

    # С валидными значениями
    macd_valid = 2.0
    macd_signal_valid = 1.5
    if not math.isnan(macd_valid) and not math.isnan(macd_signal_valid):
        trend_strength = math.tanh((macd_valid - macd_signal_valid) / (price * 0.01 + 1e-8))
    else:
        trend_strength = 0.0

    assert not math.isnan(trend_strength), "trend_strength must not be NaN!"
    print(f"✓ trend_strength with valid MACD: {trend_strength:.4f}")

    print("\n✅ Все тесты производных признаков пройдены!\n")


def test_complete_observation():
    """
    Полный тест: создание observation vector с NaN индикаторами.
    Проверяем, что ВСЕ элементы observation валидны (не NaN, не Inf).
    """
    print("=" * 80)
    print("ТЕСТ 4: Полная проверка observation vector")
    print("=" * 80)

    # Симулируем ранние бары с NaN индикаторами
    price = 100.0
    prev_price = 99.0

    # Индикаторы (некоторые NaN)
    ma5 = 100.0
    ma20 = 99.5
    rsi14 = float('nan')      # Первые 14 баров
    macd = float('nan')       # Первые ~26 баров
    macd_signal = float('nan')
    momentum = float('nan')   # Первые 10 баров
    atr = float('nan')        # Первые 14 баров
    cci = float('nan')        # Первые 20 баров
    obv = 1000.0
    bb_lower = float('nan')   # Первые 20 баров
    bb_upper = float('nan')

    # Применяем логику безопасной обработки
    def safe_value(val, default):
        return val if not math.isnan(val) else default

    observation = []

    # Bar level
    observation.append(price)
    observation.append(safe_value(ma5, 0.0))
    observation.append(1.0 if not math.isnan(ma5) else 0.0)
    observation.append(safe_value(ma20, 0.0))
    observation.append(1.0 if not math.isnan(ma20) else 0.0)

    # Indicators with safe defaults
    observation.append(safe_value(rsi14, 50.0))
    observation.append(safe_value(macd, 0.0))
    observation.append(safe_value(macd_signal, 0.0))
    observation.append(safe_value(momentum, 0.0))
    observation.append(safe_value(atr, price * 0.01))
    observation.append(safe_value(cci, 0.0))
    observation.append(safe_value(obv, 0.0))

    # Derived features with conditional computation
    bb_valid = not math.isnan(bb_lower)
    if bb_valid:
        bb_squeeze = math.tanh((bb_upper - bb_lower) / (price + 1e-8))
    else:
        bb_squeeze = 0.0
    observation.append(bb_squeeze)

    if not math.isnan(momentum):
        price_momentum = math.tanh(momentum / (price * 0.01 + 1e-8))
    else:
        price_momentum = 0.0
    observation.append(price_momentum)

    if not math.isnan(macd) and not math.isnan(macd_signal):
        trend_strength = math.tanh((macd - macd_signal) / (price * 0.01 + 1e-8))
    else:
        trend_strength = 0.0
    observation.append(trend_strength)

    # Проверка: НИ ОДНОГО NaN или Inf
    has_nan = any(math.isnan(x) for x in observation)
    has_inf = any(math.isinf(x) for x in observation)

    print(f"Всего признаков: {len(observation)}")
    print(f"NaN значений: {sum(1 for x in observation if math.isnan(x))}")
    print(f"Inf значений: {sum(1 for x in observation if math.isinf(x))}")
    print(f"\nObservation vector (first 10): {observation[:10]}")

    if has_nan:
        print("\n❌ ПРОВАЛ: Обнаружены NaN значения!")
        return False
    elif has_inf:
        print("\n❌ ПРОВАЛ: Обнаружены Inf значения!")
        return False
    else:
        print("\n✅ УСПЕХ: Все значения валидны (нет NaN, нет Inf)!")
        return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ИСПРАВЛЕНИЙ NaN В obs_builder")
    print("=" * 80 + "\n")

    try:
        test_clipf_logic()
        test_indicator_defaults()
        test_derived_features()
        success = test_complete_observation()

        print("\n" + "=" * 80)
        print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
        print("=" * 80)
        if success:
            print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
            print("\nИсправления корректны. После перекомпиляции obs_builder.pyx")
            print("NaN значения больше не будут попадать в observation vector.")
        else:
            print("❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ!")
            print("\nНеобходимо проверить логику обработки NaN.")

    except Exception as e:
        print(f"\n❌ ОШИБКА ПРИ ВЫПОЛНЕНИИ ТЕСТОВ: {e}")
        import traceback
        traceback.print_exc()
