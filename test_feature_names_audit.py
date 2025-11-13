#!/usr/bin/env python3
"""
Тест для проверки согласованности имен признаков между трансформером и mediator.
"""

from transformers import FeatureSpec, OnlineFeatureTransformer, _format_window_name
import pandas as pd

def test_feature_name_consistency():
    """Проверяет согласованность имен признаков"""

    # Создаем spec для 4h интервала
    spec = FeatureSpec(
        lookbacks_prices=[240, 720, 1440, 12000],
        yang_zhang_windows=[2880, 10080, 43200],
        parkinson_windows=[2880, 10080],
        garch_windows=[10080, 20160, 43200],
        taker_buy_ratio_windows=[480, 960, 1440],
        taker_buy_ratio_momentum=[240, 480, 720],
        cvd_windows=[1440, 10080],
        bar_duration_minutes=240
    )

    print("=" * 80)
    print("ПРОВЕРКА СОГЛАСОВАННОСТИ ИМЕН ПРИЗНАКОВ")
    print("=" * 80)

    # Проверяем конвертацию из минут в бары
    print("\n1. Конвертация окон из минут в бары:")
    print(f"   lookbacks_prices (минуты): {spec._lookbacks_prices_minutes}")
    print(f"   lookbacks_prices (бары):   {spec.lookbacks_prices}")
    print(f"   yang_zhang (минуты): {spec._yang_zhang_windows_minutes}")
    print(f"   yang_zhang (бары):   {spec.yang_zhang_windows}")

    # Создаем трансформер
    transformer = OnlineFeatureTransformer(spec)

    # Проверяем имена признаков, которые будут сгенерированы
    print("\n2. Имена признаков SMA (должны быть в минутах для онлайн, в барах для оффлайн):")
    for i, lb in enumerate(spec.lookbacks_prices):
        lb_minutes = spec._lookbacks_prices_minutes[i]
        print(f"   Онлайн:  sma_{lb_minutes} (окно={lb} баров)")
        print(f"   Оффлайн: sma_{lb} (окно={lb} баров)")

    print("\n3. Имена признаков returns (должны использовать _format_window_name):")
    for i, lb in enumerate(spec.lookbacks_prices):
        lb_minutes = spec._lookbacks_prices_minutes[i]
        ret_name = f"ret_{_format_window_name(lb_minutes)}"
        print(f"   {ret_name} (окно={lb} баров, {lb_minutes} минут)")

    print("\n4. Имена признаков Yang-Zhang:")
    for i, window in enumerate(spec.yang_zhang_windows):
        window_minutes = spec._yang_zhang_windows_minutes[i]
        window_name = _format_window_name(window_minutes)
        feature_name = f"yang_zhang_{window_name}"
        print(f"   {feature_name} (окно={window} баров, {window_minutes} минут)")

    print("\n5. Имена признаков GARCH:")
    for i, window in enumerate(spec.garch_windows):
        window_minutes = spec._garch_windows_minutes[i]
        window_name = _format_window_name(window_minutes)
        feature_name = f"garch_{window_name}"
        print(f"   {feature_name} (окно={window} баров, {window_minutes} минут)")

    print("\n6. Имена признаков в mediator.py (ожидаемые):")
    expected_features = [
        "cvd_24h", "cvd_7d",
        "yang_zhang_48h", "yang_zhang_7d",
        "garch_200h", "garch_14d",
        "ret_12h", "ret_24h", "ret_4h",
        "sma_50",  # ПРОБЛЕМА: ожидается sma_50, но генерируется sma_12000!
        "yang_zhang_30d",
        "parkinson_48h", "parkinson_7d",
        "garch_30d",
        "taker_buy_ratio",
        "taker_buy_ratio_sma_24h",
        "taker_buy_ratio_sma_8h", "taker_buy_ratio_sma_16h",
        "taker_buy_ratio_momentum_4h", "taker_buy_ratio_momentum_8h", "taker_buy_ratio_momentum_12h"
    ]
    for feat in expected_features:
        print(f"   {feat}")

    print("\n7. КРИТИЧЕСКИЕ ПРОБЛЕМЫ:")

    # Проверка SMA имен
    print("\n   А) ПРОБЛЕМА С ИМЕНАМИ SMA:")
    print("      - mediator.py ожидает: sma_50 (в барах)")
    print(f"      - OnlineFeatureTransformer генерирует: sma_{spec._lookbacks_prices_minutes[3]} (в минутах)")
    print(f"      - apply_offline_features генерирует: sma_{spec.lookbacks_prices[3]} (в барах)")
    print("      НЕСООТВЕТСТВИЕ между онлайн и оффлайн!")

    # Проверка default значений в make_features.py vs FeatureSpec
    print("\n   Б) ПРОБЛЕМА С DEFAULT ЗНАЧЕНИЯМИ:")
    print("      - FeatureSpec.__post_init__: [240, 720, 1440, 12000]")
    print("      - make_features.py default:  [240, 720, 1440] (отсутствует 12000!)")
    print("      НЕСООТВЕТСТВИЕ!")

if __name__ == "__main__":
    test_feature_name_consistency()
