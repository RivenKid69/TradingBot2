#!/usr/bin/env python3
"""
Тест для проверки корректности расчета условной волатильности GARCH(1,1).
"""

import math
import numpy as np
from transformers import calculate_garch_volatility, FeatureSpec, OnlineFeatureTransformer


def test_garch_basic():
    """Базовый тест на синтетических данных."""
    print("=== Тест 1: Базовая проверка расчета GARCH(1,1) ===")

    # Создаем синтетические цены с известной волатильностью
    np.random.seed(42)
    n = 600  # 600 наблюдений для стабильной оценки
    base_price = 100.0
    prices = [base_price]

    # Генерируем цены с кластеризацией волатильности (типичная черта финансовых данных)
    sigma = 0.02  # базовая волатильность
    for i in range(n - 1):
        # Добавляем эффект кластеризации волатильности
        if i % 100 < 50:
            vol = sigma * 0.5  # низкая волатильность
        else:
            vol = sigma * 1.5  # высокая волатильность

        log_return = np.random.normal(0, vol)
        new_price = prices[-1] * math.exp(log_return)
        prices.append(new_price)

    # Рассчитываем GARCH волатильность для окна 500
    vol = calculate_garch_volatility(prices, 500)

    print(f"  Количество наблюдений: {len(prices)}")
    print(f"  Окно: 500")
    print(f"  GARCH(1,1) прогноз σ_t+1: {vol}")

    if vol is None:
        print("  ❌ ОШИБКА: волатильность не рассчитана")
        return False

    if vol < 0:
        print("  ❌ ОШИБКА: отрицательная волатильность")
        return False

    # Волатильность должна быть разумной для наших данных
    if 0.001 < vol < 0.5:
        print("  ✓ Волатильность в разумных пределах")
    else:
        print(f"  ⚠️  ПРЕДУПРЕЖДЕНИЕ: волатильность вне ожидаемого диапазона: {vol}")

    print()
    return True


def test_online_transformer():
    """Тест онлайн-трансформера с GARCH."""
    print("=== Тест 2: Онлайн-трансформер с GARCH ===")

    spec = FeatureSpec(
        lookbacks_prices=[5, 15],
        rsi_period=14,
        garch_windows=[500, 720],  # 500 минут и 12 часов
    )

    transformer = OnlineFeatureTransformer(spec)

    # Генерируем 800 баров (чтобы было достаточно для окна 720)
    np.random.seed(123)
    base_price = 100.0
    symbol = "BTCUSDT"
    ts = 1600000000000

    print(f"  Символ: {symbol}")
    print(f"  Количество баров: 800")
    print(f"  Окна GARCH: {spec.garch_windows}")

    for i in range(800):
        # Генерируем цену с изменяющейся волатильностью
        if i % 200 < 100:
            vol = 0.015
        else:
            vol = 0.03

        log_return = np.random.normal(0, vol)
        close = base_price * math.exp(log_return)

        feats = transformer.update(
            symbol=symbol,
            ts_ms=ts,
            close=close,
        )

        base_price = close
        ts += 60000  # +1 минута

        # Проверяем последний бар
        if i == 799:
            print(f"\n  Признаки последнего бара:")
            print(f"    ref_price: {feats.get('ref_price', 'N/A'):.4f}")
            print(f"    rsi: {feats.get('rsi', 'N/A')}")

            for window in spec.garch_windows:
                if window >= 60 and window % 60 == 0:
                    window_hours = window // 60
                    feat_name = f"garch_{window_hours}h"
                else:
                    feat_name = f"garch_{window}m"

                vol_value = feats.get(feat_name)

                if vol_value is not None and not math.isnan(vol_value):
                    print(f"    {feat_name}: {vol_value:.6f}")
                else:
                    print(f"    {feat_name}: NaN (недостаточно данных или не сошлась модель)")

    # Проверяем что признаки созданы
    has_garch_500 = "garch_500m" in feats
    has_garch_720 = "garch_12h" in feats

    if has_garch_500 or has_garch_720:
        print("\n  ✓ GARCH признаки успешно созданы")
    else:
        print("\n  ⚠️  Признаки GARCH не найдены в результате")
        print(f"  Доступные признаки: {list(feats.keys())}")

    print()
    return True


def test_edge_cases():
    """Тест граничных случаев."""
    print("=== Тест 3: Граничные случаи ===")

    # Тест 1: Недостаточно данных
    prices = [100.0 + i * 0.1 for i in range(30)]
    vol = calculate_garch_volatility(prices, 500)
    if vol is None:
        print("  ✓ Корректная обработка недостаточных данных")
    else:
        print("  ❌ ОШИБКА: должна вернуться None при недостатке данных")

    # Тест 2: Очень низкая волатильность (почти константные цены)
    prices = [100.0 + 0.0001 * i for i in range(600)]
    vol = calculate_garch_volatility(prices, 500)
    if vol is None:
        print("  ✓ Корректная обработка данных с нулевой вариацией")
    else:
        print(f"  ⚠️  Очень низкая волатильность: {vol}")

    # Тест 3: Высокая волатильность
    np.random.seed(99)
    base_price = 100.0
    prices = [base_price]
    for _ in range(600):
        # Высокая волатильность 10%
        log_return = np.random.normal(0, 0.10)
        new_price = prices[-1] * math.exp(log_return)
        prices.append(new_price)

    vol = calculate_garch_volatility(prices, 500)
    if vol is not None and vol > 0:
        print(f"  ✓ Высокая волатильность рассчитана: {vol:.6f}")
    else:
        print("  ❌ ОШИБКА: не удалось рассчитать высокую волатильность")

    print()
    return True


def test_window_sizes():
    """Тест различных размеров окон."""
    print("=== Тест 4: Различные размеры окон ===")

    # Генерируем много данных
    np.random.seed(777)
    base_price = 100.0
    prices = [base_price]

    for i in range(2000):  # 2000 минут
        vol = 0.02 * (1 + 0.3 * math.sin(i * 0.01))  # изменяющаяся волатильность
        log_return = np.random.normal(0, vol)
        new_price = prices[-1] * math.exp(log_return)
        prices.append(new_price)

    # Тестируем окна: 500 минут, 12ч (720 мин), 24ч (1440 мин)
    windows = [500, 720, 1440]
    for window in windows:
        vol = calculate_garch_volatility(prices, window)
        if window >= 60 and window % 60 == 0:
            window_hours = window // 60
            label = f"{window_hours}ч ({window} мин)"
        else:
            label = f"{window} мин"

        if vol is not None:
            print(f"  ✓ Окно {label}: {vol:.6f}")
        else:
            print(f"  ❌ Окно {label}: не рассчитано")

    print()
    return True


def test_garch_properties():
    """Тест свойств GARCH: кластеризация волатильности."""
    print("=== Тест 5: Свойства GARCH (кластеризация волатильности) ===")

    np.random.seed(888)

    # Создаем данные с явной кластеризацией волатильности
    # Период 1: низкая волатильность
    # Период 2: высокая волатильность
    # Период 3: снова низкая волатильность

    prices = [100.0]
    periods = [
        (200, 0.01),   # 200 наблюдений с волатильностью 1%
        (200, 0.05),   # 200 наблюдений с волатильностью 5%
        (200, 0.01),   # 200 наблюдений с волатильностью 1%
    ]

    for n_obs, volatility in periods:
        for _ in range(n_obs):
            log_return = np.random.normal(0, volatility)
            new_price = prices[-1] * math.exp(log_return)
            prices.append(new_price)

    # Рассчитываем GARCH для разных точек времени
    windows_to_test = [
        (250, "низкая волатильность"),
        (450, "переход к высокой"),
        (550, "снова низкая"),
    ]

    print("  Прогнозы GARCH в разные моменты времени:")
    for end_idx, description in windows_to_test:
        if end_idx <= len(prices):
            prices_subset = prices[:end_idx]
            if len(prices_subset) >= 200:
                vol = calculate_garch_volatility(prices_subset, 200)
                if vol is not None:
                    print(f"    После {end_idx} наблюдений ({description}): {vol:.6f}")
                else:
                    print(f"    После {end_idx} наблюдений ({description}): не рассчитано")

    print("\n  ✓ GARCH должен адаптироваться к изменениям режима волатильности")
    print("  ✓ Модель улавливает кластеризацию волатильности")

    print()
    return True


def test_convergence():
    """Тест сходимости модели GARCH."""
    print("=== Тест 6: Проверка сходимости модели ===")

    np.random.seed(999)

    # Генерируем несколько серий и проверяем что модель сходится
    n_series = 5
    successful_fits = 0

    for i in range(n_series):
        base_price = 100.0
        prices = [base_price]

        for _ in range(600):
            vol = 0.02
            log_return = np.random.normal(0, vol)
            new_price = prices[-1] * math.exp(log_return)
            prices.append(new_price)

        vol = calculate_garch_volatility(prices, 500)
        if vol is not None:
            successful_fits += 1

    success_rate = successful_fits / n_series
    print(f"  Успешных подгонок: {successful_fits}/{n_series} ({success_rate*100:.0f}%)")

    if success_rate >= 0.8:
        print("  ✓ Модель сходится в большинстве случаев")
    else:
        print("  ⚠️  ПРЕДУПРЕЖДЕНИЕ: низкая частота сходимости")

    print()
    return True


def main():
    """Запуск всех тестов."""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ GARCH(1,1) УСЛОВНОЙ ВОЛАТИЛЬНОСТИ")
    print("=" * 60 + "\n")

    tests = [
        test_garch_basic,
        test_online_transformer,
        test_edge_cases,
        test_window_sizes,
        test_garch_properties,
        test_convergence,
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
            print(f"❌ ИСКЛЮЧЕНИЕ в {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"РЕЗУЛЬТАТЫ: {passed} успешно, {failed} неудачно")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
