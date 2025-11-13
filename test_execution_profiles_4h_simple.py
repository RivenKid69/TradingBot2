#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_execution_profiles_4h_simple.py

Простой тест для проверки execution profiles для 4h (без импорта core_config).
"""

import os
import re
import yaml


def test_core_config_has_4h_profiles():
    """Проверяем что core_config.py содержит 4H профили в enum."""
    print("Проверка core_config.py...")

    with open("core_config.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Ищем определение enum ExecutionProfile
    # и проверяем что там есть 4H профили
    assert 'MKT_OPEN_NEXT_4H = "MKT_OPEN_NEXT_4H"' in content, \
        "core_config.py должен содержать MKT_OPEN_NEXT_4H в ExecutionProfile enum!"
    assert 'VWAP_CURRENT_4H = "VWAP_CURRENT_4H"' in content, \
        "core_config.py должен содержать VWAP_CURRENT_4H в ExecutionProfile enum!"

    print("  ✓ ExecutionProfile enum содержит 4H профили")


def test_yaml_configs_use_4h_profiles():
    """Проверяем что YAML конфиги используют 4H профили."""
    print("Проверка YAML конфигов...")

    configs_to_check = [
        ("configs/config_sim.yaml", "MKT_OPEN_NEXT_4H"),
        ("configs/config_live.yaml", "MKT_OPEN_NEXT_4H"),
        ("configs/config_train.yaml", "MKT_OPEN_NEXT_4H"),
        ("configs/config_eval.yaml", "MKT_OPEN_NEXT_4H"),
        ("configs/config_template.yaml", "MKT_OPEN_NEXT_4H"),
    ]

    for config_path, expected_profile in configs_to_check:
        if not os.path.exists(config_path):
            print(f"  ⚠ {config_path} не найден, пропускаем")
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        execution_profile = data.get("execution_profile")

        assert execution_profile == expected_profile, \
            f"{config_path}: execution_profile должен быть {expected_profile}, а не {execution_profile}"
        print(f"  ✓ {config_path}: execution_profile = {execution_profile}")


def test_timing_yaml_has_4h_profiles():
    """Проверяем что timing.yaml содержит 4H профили."""
    print("Проверка timing.yaml...")

    timing_path = "configs/timing.yaml"
    if not os.path.exists(timing_path):
        print(f"  ⚠ {timing_path} не найден, пропускаем")
        return

    with open(timing_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    profiles = data.get("profiles", {})

    # Должны быть определены 4H профили
    assert "MKT_OPEN_NEXT_4H" in profiles, \
        "timing.yaml должен содержать профиль MKT_OPEN_NEXT_4H"

    # Проверяем значения для 4H профиля
    mkt_open_4h = profiles["MKT_OPEN_NEXT_4H"]
    assert mkt_open_4h.get("decision_mode") == "CLOSE_TO_OPEN", \
        "MKT_OPEN_NEXT_4H должен иметь decision_mode = CLOSE_TO_OPEN"
    assert mkt_open_4h.get("decision_delay_ms") == 8000, \
        f"MKT_OPEN_NEXT_4H должен иметь decision_delay_ms = 8000 для 4h, а не {mkt_open_4h.get('decision_delay_ms')}"

    print(f"  ✓ timing.yaml содержит MKT_OPEN_NEXT_4H профиль")

    # Проверяем дефолтные значения
    defaults = data.get("defaults", {})
    assert defaults.get("timeframe_ms") == 14400000, \
        f"defaults.timeframe_ms должен быть 14400000 (4h), а не {defaults.get('timeframe_ms')}"
    assert defaults.get("close_lag_ms") == 8000, \
        f"defaults.close_lag_ms должен быть 8000 для 4h, а не {defaults.get('close_lag_ms')}"

    print(f"  ✓ timing.yaml имеет правильные дефолтные значения для 4h")


if __name__ == "__main__":
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ EXECUTION PROFILES ДЛЯ 4H ТАЙМФРЕЙМА")
    print("=" * 70)
    print()

    test_core_config_has_4h_profiles()
    print()
    test_yaml_configs_use_4h_profiles()
    print()
    test_timing_yaml_has_4h_profiles()

    print()
    print("=" * 70)
    print("✓ ВСЕ ТЕСТЫ EXECUTION PROFILES ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 70)
