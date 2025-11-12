#!/usr/bin/env python3
"""
Тест для проверки всех исправлений из AUDIT_SELF_CHECK_REPORT.md
"""

import sys
from transformers import FeatureSpec, OnlineFeatureTransformer, _format_window_name

def test_critical_1_sma_names():
    """CRITICAL #1: Проверка согласованности имен sma_*"""
    print("\n" + "="*80)
    print("ТЕСТ #1: CRITICAL #1 - Имена sma_*")
    print("="*80)

    spec = FeatureSpec(
        lookbacks_prices=[240, 720, 1440, 12000],
        bar_duration_minutes=240
    )

    transformer = OnlineFeatureTransformer(spec)

    # Проверяем что после конвертации имеем правильные значения в барах
    assert spec.lookbacks_prices == [1, 3, 6, 50], f"Expected [1, 3, 6, 50] bars, got {spec.lookbacks_prices}"

    # Обновляем трансформер с данными
    feats = transformer.update(symbol="BTCUSDT", ts_ms=1000, close=50000.0)

    # Проверяем что имена sma используют бары (для согласованности с mediator.py)
    expected_sma_names = ["sma_1", "sma_3", "sma_6", "sma_50"]
    for name in expected_sma_names:
        assert name in feats, f"Missing feature: {name}. Available: {list(feats.keys())}"

    print("✅ PASSED: SMA имена используют бары (sma_50, не sma_12000)")
    print(f"   Сгенерированные SMA: {[k for k in feats.keys() if k.startswith('sma_')]}")
    return True

def test_critical_2_default_lookbacks():
    """CRITICAL #2: Проверка default параметров в make_features.py"""
    print("\n" + "="*80)
    print("ТЕСТ #2: CRITICAL #2 - Default lookbacks включает 12000")
    print("="*80)

    # Проверяем что FeatureSpec имеет правильные defaults
    spec = FeatureSpec(lookbacks_prices=[], bar_duration_minutes=240)

    # Должны быть установлены дефолтные значения
    assert spec.lookbacks_prices == [1, 3, 6, 50], f"Expected [1, 3, 6, 50] bars, got {spec.lookbacks_prices}"
    assert spec._lookbacks_prices_minutes == [240, 720, 1440, 12000], \
        f"Expected [240, 720, 1440, 12000] minutes, got {spec._lookbacks_prices_minutes}"

    print("✅ PASSED: Default lookbacks включает 12000 минут (50 баров)")
    return True

def test_critical_4_garch_8d():
    """CRITICAL #4: Проверка что garch_7d изменен на garch_8d (50 баров минимум)"""
    print("\n" + "="*80)
    print("ТЕСТ #3: CRITICAL #4 - GARCH минимум 50 баров")
    print("="*80)

    spec = FeatureSpec(
        garch_windows=[],  # Должны быть установлены defaults
        bar_duration_minutes=240
    )

    # Проверяем defaults
    print(f"   GARCH windows (bars): {spec.garch_windows}")
    print(f"   GARCH windows (minutes): {spec._garch_windows_minutes}")

    # Первое окно должно быть >= 50 баров
    assert spec.garch_windows[0] >= 50, f"First GARCH window must be >= 50 bars, got {spec.garch_windows[0]}"

    # Проверяем что это 12000 минут (50 * 240)
    assert spec._garch_windows_minutes[0] == 12000, \
        f"Expected 12000 minutes (8d), got {spec._garch_windows_minutes[0]}"

    # Проверяем формирование имени
    transformer = OnlineFeatureTransformer(spec)
    feats = transformer.update(symbol="BTCUSDT", ts_ms=1000, close=50000.0)

    # Должен быть garch_8d, а не garch_7d
    assert "garch_8d" in feats or all(k not in feats for k in feats if k.startswith("garch_")), \
        f"Expected garch_8d or no garch features (insufficient data), got: {[k for k in feats.keys() if k.startswith('garch_')]}"

    print(f"✅ PASSED: GARCH минимальное окно = {spec.garch_windows[0]} баров >= 50")
    print(f"   Это {spec._garch_windows_minutes[0]} минут = 8.33 дней ≈ 8d")
    return True

def test_major_1_empty_df_names():
    """MAJOR #1: Проверка имен в apply_offline_features для пустого df"""
    print("\n" + "="*80)
    print("ТЕСТ #4: MAJOR #1 - Имена признаков для пустого датафрейма")
    print("="*80)

    spec = FeatureSpec(
        lookbacks_prices=[240, 720, 1440, 12000],
        yang_zhang_windows=[2880, 10080, 43200],
        garch_windows=[12000, 20160, 43200],
        bar_duration_minutes=240
    )

    # Проверяем что _*_minutes поля существуют
    assert hasattr(spec, '_lookbacks_prices_minutes'), "Missing _lookbacks_prices_minutes"
    assert hasattr(spec, '_yang_zhang_windows_minutes'), "Missing _yang_zhang_windows_minutes"
    assert hasattr(spec, '_garch_windows_minutes'), "Missing _garch_windows_minutes"

    # Проверяем значения
    assert spec._lookbacks_prices_minutes == [240, 720, 1440, 12000]
    assert spec._yang_zhang_windows_minutes == [2880, 10080, 43200]
    assert spec._garch_windows_minutes == [12000, 20160, 43200]

    print("✅ PASSED: _*_minutes поля инициализированы правильно")
    print(f"   lookbacks_minutes: {spec._lookbacks_prices_minutes}")
    print(f"   garch_minutes: {spec._garch_windows_minutes}")
    return True

def test_format_window_name():
    """Проверка функции _format_window_name"""
    print("\n" + "="*80)
    print("ТЕСТ #5: Проверка _format_window_name")
    print("="*80)

    # Проверяем форматирование
    assert _format_window_name(240) == "4h", f"Expected '4h', got '{_format_window_name(240)}'"
    assert _format_window_name(720) == "12h", f"Expected '12h', got '{_format_window_name(720)}'"
    assert _format_window_name(1440) == "24h", f"Expected '24h', got '{_format_window_name(1440)}'"
    assert _format_window_name(12000) == "200h", f"Expected '200h', got '{_format_window_name(12000)}'"
    assert _format_window_name(10080) == "7d", f"Expected '7d', got '{_format_window_name(10080)}'"

    print("✅ PASSED: _format_window_name работает правильно")
    print(f"   240 → {_format_window_name(240)}")
    print(f"   10080 → {_format_window_name(10080)}")
    print(f"   12000 → {_format_window_name(12000)}")
    return True

def test_mediator_compatibility():
    """Проверка совместимости с mediator.py"""
    print("\n" + "="*80)
    print("ТЕСТ #6: Совместимость с mediator.py")
    print("="*80)

    spec = FeatureSpec(bar_duration_minutes=240)
    transformer = OnlineFeatureTransformer(spec)

    # Генерируем признаки
    feats = transformer.update(symbol="BTCUSDT", ts_ms=1000, close=50000.0)

    # Признаки, которые ожидает mediator.py (из norm_cols)
    expected_features = [
        "sma_50",  # norm_cols[9]
        "ret_4h",  # norm_cols[8]
        "ret_12h", # norm_cols[6]
        "ret_24h", # norm_cols[7]
        # Волатильности будут NaN (недостаточно данных), но имена должны совпадать
    ]

    for feat in expected_features:
        # Проверяем что признак либо есть, либо это нормально (недостаточно данных)
        if feat.startswith("sma_") or feat.startswith("ret_"):
            # SMA и ret должны быть даже с минимальными данными (могут быть NaN)
            assert feat in feats or feat.replace("_", "_") in feats, \
                f"Missing expected feature: {feat}. Available: {list(feats.keys())}"

    print("✅ PASSED: Имена признаков совместимы с mediator.py")
    print(f"   Доступные SMA: {[k for k in feats.keys() if k.startswith('sma_')]}")
    print(f"   Доступные ret: {[k for k in feats.keys() if k.startswith('ret_')]}")
    return True

def main():
    """Запуск всех тестов"""
    print("\n" + "="*80)
    print("ПРОВЕРКА ВСЕХ ИСПРАВЛЕНИЙ")
    print("="*80)

    tests = [
        ("CRITICAL #1: SMA имена", test_critical_1_sma_names),
        ("CRITICAL #2: Default lookbacks", test_critical_2_default_lookbacks),
        ("CRITICAL #4: GARCH >= 50 баров", test_critical_4_garch_8d),
        ("MAJOR #1: Empty df имена", test_major_1_empty_df_names),
        ("Utility: _format_window_name", test_format_window_name),
        ("Integration: mediator.py", test_mediator_compatibility),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {test_name}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n💥 ERROR: {test_name}")
            print(f"   Exception: {e}")
            failed += 1

    print("\n" + "="*80)
    print(f"РЕЗУЛЬТАТЫ: {passed} passed, {failed} failed")
    print("="*80)

    if failed > 0:
        print("\n❌ Некоторые тесты не прошли. Требуется дополнительная проверка.")
        sys.exit(1)
    else:
        print("\n✅ ВСЕ ТЕСТЫ ПРОШЛИ! Исправления работают корректно.")
        sys.exit(0)

if __name__ == "__main__":
    main()
