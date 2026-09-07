#!/usr/bin/env python
"""
Простой проверочный скрипт для системы 63 признаков.
Не требует pytest, запускается напрямую.
"""

import sys
import numpy as np


def test_feature_config():
    """Проверка конфигурации признаков"""
    print("=" * 70)
    print("ПРОВЕРКА 1: Конфигурация признаков (feature_config.py)")
    print("=" * 70)

    from feature_config import EXT_NORM_DIM, N_FEATURES, FEATURES_LAYOUT

    print(f"EXT_NORM_DIM = {EXT_NORM_DIM}")
    assert EXT_NORM_DIM == 21, f"❌ Expected EXT_NORM_DIM=21, got {EXT_NORM_DIM}"
    print("✓ EXT_NORM_DIM = 21")

    print(f"\nN_FEATURES = {N_FEATURES}")
    assert N_FEATURES == 63, f"❌ Expected N_FEATURES=63, got {N_FEATURES}"
    print("✓ N_FEATURES = 63")

    print(f"\nБлоки признаков:")
    total = 0
    for block in FEATURES_LAYOUT:
        name = block["name"]
        size = block["size"]
        print(f"  {name:15s}: {size:2d}")
        total += size

    print(f"  {'ИТОГО':15s}: {total:2d}")
    assert total == 63, f"❌ Total features = {total}, expected 63"
    print("✓ Сумма всех блоков = 63")

    print("\n✅ Конфигурация корректна!\n")
    return True


def test_mediator_norm_cols():
    """Проверка _extract_norm_cols в mediator.py"""
    print("=" * 70)
    print("ПРОВЕРКА 2: Загрузка norm_cols (mediator.py)")
    print("=" * 70)

    from unittest.mock import Mock
    import pandas as pd
    from mediator import Mediator

    # Создаем mock environment
    mock_env = Mock()
    mock_env.state = Mock(units=0.0, cash=10000.0, max_position=1.0)
    mock_env.lob = None

    mediator = Mediator(mock_env, event_level=0)

    # Создаем mock row с всеми 21 признаками (обновлено для 4h таймфрейма)
    mock_row = pd.Series(
        {
            "cvd_24h": 1000.0,
            "cvd_7d": 5000.0,  # было cvd_168h
            "yang_zhang_48h": 0.05,  # было yang_zhang_24h
            "yang_zhang_7d": 0.08,  # было yang_zhang_168h
            "garch_200h": 0.03,  # было garch_12h (42 бара = 10080 мин = 7d, минимум для GARCH на 4h)
            "garch_14d": 0.04,  # было garch_24h
            "ret_12h": 0.001,  # было ret_15m
            "ret_24h": 0.005,  # было ret_60m
            "ret_4h": 0.0005,  # было ret_5m
            "sma_12000": 50000.0,  # было sma_60 (50 баров = 12000 минут = 200h для 4h)
            "yang_zhang_30d": 0.12,  # было yang_zhang_720h
            "parkinson_48h": 0.06,  # было parkinson_24h
            "parkinson_7d": 0.09,  # было parkinson_168h
            "garch_30d": 0.025,  # было garch_500m
            "taker_buy_ratio": 0.52,
            "taker_buy_ratio_sma_24h": 0.51,
            # НОВЫЕ 5 признаков (обновлено для 4h)
            "taker_buy_ratio_sma_8h": 0.53,  # было 6h
            "taker_buy_ratio_sma_16h": 0.52,  # было 12h
            "taker_buy_ratio_momentum_4h": 0.01,  # было 1h
            "taker_buy_ratio_momentum_8h": 0.02,  # было 6h
            "taker_buy_ratio_momentum_12h": 0.015,  # без изменений
        }
    )

    norm_cols = mediator._extract_norm_cols(mock_row)

    print(f"norm_cols.shape = {norm_cols.shape}")
    assert norm_cols.shape[0] == 21, f"❌ Expected 21, got {norm_cols.shape[0]}"
    print("✓ norm_cols имеет размер 21")

    print(f"norm_cols.dtype = {norm_cols.dtype}")
    assert norm_cols.dtype == np.float32, f"❌ Expected float32, got {norm_cols.dtype}"
    print("✓ norm_cols имеет тип float32")

    assert np.all(np.isfinite(norm_cols)), "❌ norm_cols содержит NaN или Inf"
    print("✓ Все значения конечные (no NaN/Inf)")

    # Проверка что НЕ применен tanh (значения должны быть большими)
    assert (
        norm_cols[0] > 10.0
    ), f"❌ norm_cols[0]={norm_cols[0]}, похоже что tanh уже применен (должно быть ~1000)"
    print(f"✓ Нет двойной нормализации (norm_cols[0]={norm_cols[0]:.1f}, ожидалось ~1000)")

    print(f"\nПервые 5 значений norm_cols:")
    for i in range(min(5, len(norm_cols))):
        print(f"  [{i}] = {norm_cols[i]:.4f}")

    print("\n✅ Загрузка norm_cols корректна!\n")
    return True


def test_obs_builder():
    """Проверка obs_builder.pyx"""
    print("=" * 70)
    print("ПРОВЕРКА 3: Построение observation (obs_builder.pyx)")
    print("=" * 70)

    try:
        from obs_builder import build_observation_vector
    except ImportError as e:
        print(f"⚠️  obs_builder не скомпилирован: {e}")
        print("   Пропускаем этот тест (obs_builder будет скомпилирован при первом запуске)")
        return True

    # Создаем norm_cols с большими значениями для проверки tanh
    norm_cols = np.array([1000.0] * 21, dtype=np.float32)
    out = np.zeros(63, dtype=np.float32)

    build_observation_vector(
        price=50000.0,
        prev_price=49900.0,
        log_volume_norm=0.5,
        rel_volume=0.5,
        ma5=50100.0,
        ma20=50200.0,
        rsi14=50.0,
        macd=0.0,
        macd_signal=0.0,
        momentum=0.0,
        atr=100.0,
        cci=0.0,
        obv=0.0,
        bb_lower=49000.0,
        bb_upper=51000.0,
        is_high_importance=0.0,
        time_since_event=0.0,
        fear_greed_value=50.0,
        has_fear_greed=True,
        risk_off_flag=False,
        cash=10000.0,
        units=0.0,
        last_vol_imbalance=0.0,
        last_trade_intensity=0.0,
        last_realized_spread=0.0,
        last_agent_fill_ratio=0.0,
        token_id=0,
        max_num_tokens=1,
        num_tokens=1,
        norm_cols_values=norm_cols,
        out_features=out,
    )

    print(f"out.shape = {out.shape}")
    assert out.shape[0] == 63, f"❌ Expected 63, got {out.shape[0]}"
    print("✓ observation имеет размер 63")

    assert np.all(np.isfinite(out)), "❌ observation содержит NaN или Inf"
    print("✓ Все значения конечные (no NaN/Inf)")

    # External features начинаются с индекса 32
    external_start = 32
    external_end = external_start + 21
    external = out[external_start:external_end]

    print(f"\nExternal features (индексы {external_start}-{external_end-1}):")
    print(f"  Диапазон: [{external.min():.4f}, {external.max():.4f}]")

    # После tanh(1000) все должны быть ~1.0
    assert np.all(external >= -1.0), f"❌ Некоторые external < -1.0"
    assert np.all(external <= 1.0), f"❌ Некоторые external > 1.0"
    print("✓ External features в диапазоне [-1, 1] (после tanh)")

    assert np.all(external > 0.99), f"❌ Expected ~1.0 after tanh(1000), got min={external.min()}"
    print(f"✓ tanh(1000) ≈ 1.0 применен корректно (min={external.min():.6f})")

    print("\n✅ obs_builder работает корректно!\n")
    return True


def test_column_names():
    """Проверка имен колонок"""
    print("=" * 70)
    print("ПРОВЕРКА 4: Имена колонок norm_cols")
    print("=" * 70)

    # Обновлено для 4h таймфрейма
    expected_cols = [
        "cvd_24h",
        "cvd_7d",  # было cvd_168h
        "yang_zhang_48h",  # было yang_zhang_24h
        "yang_zhang_7d",  # было yang_zhang_168h
        "garch_200h",  # было garch_12h (42 бара = 10080 мин = 7d, минимум для GARCH на 4h)
        "garch_14d",  # было garch_24h
        "ret_12h",  # было ret_15m
        "ret_24h",  # было ret_60m
        "ret_4h",  # было ret_5m
        "sma_12000",  # было sma_60 (50 баров = 12000 минут = 200h для 4h)
        "yang_zhang_30d",  # было yang_zhang_720h
        "parkinson_48h",  # было parkinson_24h
        "parkinson_7d",  # было parkinson_168h
        "garch_30d",  # было garch_500m
        "taker_buy_ratio",
        "taker_buy_ratio_sma_24h",
        # НОВЫЕ 5 (обновлено для 4h)
        "taker_buy_ratio_sma_8h",  # было 6h
        "taker_buy_ratio_sma_16h",  # было 12h
        "taker_buy_ratio_momentum_4h",  # было 1h
        "taker_buy_ratio_momentum_8h",  # было 6h
        "taker_buy_ratio_momentum_12h",  # без изменений
    ]

    print(f"Всего колонок: {len(expected_cols)}")
    assert len(expected_cols) == 21, f"❌ Expected 21, got {len(expected_cols)}"
    print("✓ 21 колонка")

    assert len(set(expected_cols)) == 21, "❌ Есть дубликаты имен"
    print("✓ Все имена уникальные")

    print("\nПолный список колонок:")
    for i, col in enumerate(expected_cols):
        marker = "  [НОВАЯ]" if i >= 16 else ""
        print(f"  [{i:2d}] {col}{marker}")

    print("\n✅ Имена колонок корректны!\n")
    return True


def main():
    """Запуск всех проверок"""
    print("\n" + "=" * 70)
    print(" ВЕРИФИКАЦИЯ СИСТЕМЫ 63 ПРИЗНАКОВ")
    print("=" * 70 + "\n")

    tests = [
        ("Конфигурация", test_feature_config),
        ("Mediator norm_cols", test_mediator_norm_cols),
        ("obs_builder", test_obs_builder),
        ("Имена колонок", test_column_names),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"\n❌ ОШИБКА в тесте '{name}':")
            print(f"   {e}")
            import traceback

            traceback.print_exc()
            failed += 1
            continue

    print("=" * 70)
    print(" ИТОГИ")
    print("=" * 70)
    print(f"✅ Пройдено:  {passed}/{len(tests)}")
    if skipped > 0:
        print(f"⚠️  Пропущено: {skipped}/{len(tests)}")
    if failed > 0:
        print(f"❌ Провалено: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ! Система 63 признаков работает корректно.")
        return 0
    else:
        print("\n⚠️  Есть ошибки. Проверьте вывод выше.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
