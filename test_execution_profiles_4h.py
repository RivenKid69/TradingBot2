#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_execution_profiles_4h.py

Тест для проверки execution profiles для 4h таймфрейма.
Проверяет:
1. ExecutionProfile enum содержит 4H профили
2. YAML конфиги используют правильные 4H профили
3. Timing profiles определены для 4H
"""

import os
import yaml
from core_config import ExecutionProfile


def test_execution_profile_enum_has_4h_profiles():
    """Проверяем что ExecutionProfile enum содержит 4H профили."""
    print("Проверка ExecutionProfile enum...")

    # Должны быть определены 4H профили
    assert hasattr(ExecutionProfile, "MKT_OPEN_NEXT_4H"), \
        "ExecutionProfile.MKT_OPEN_NEXT_4H не определен!"
    assert hasattr(ExecutionProfile, "VWAP_CURRENT_4H"), \
        "ExecutionProfile.VWAP_CURRENT_4H не определен!"

    # Проверяем значения
    assert ExecutionProfile.MKT_OPEN_NEXT_4H.value == "MKT_OPEN_NEXT_4H"
    assert ExecutionProfile.VWAP_CURRENT_4H.value == "VWAP_CURRENT_4H"

    print("✓ ExecutionProfile enum содержит 4H профили")


def test_yaml_configs_use_4h_profiles():
    """Проверяем что YAML конфиги используют 4H профили."""
    print("Проверка YAML конфигов...")

    configs_to_check = [
        "configs/config_sim.yaml",
        "configs/config_live.yaml",
        "configs/config_train.yaml",
        "configs/config_eval.yaml",
        "configs/config_template.yaml",
    ]

    for config_path in configs_to_check:
        if not os.path.exists(config_path):
            print(f"  ⚠ {config_path} не найден, пропускаем")
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        execution_profile = data.get("execution_profile")

        # Для основных конфигов должен быть 4H профиль
        if config_path in ["configs/config_sim.yaml", "configs/config_template.yaml"]:
            assert execution_profile == "MKT_OPEN_NEXT_4H", \
                f"{config_path}: execution_profile должен быть MKT_OPEN_NEXT_4H, а не {execution_profile}"
            print(f"  ✓ {config_path}: execution_profile = {execution_profile}")
        else:
            # Для live/train/eval проверяем что используется один из правильных профилей
            valid_4h_profiles = ["MKT_OPEN_NEXT_4H", "VWAP_CURRENT_4H", "LIMIT_MID_BPS"]
            assert execution_profile in valid_4h_profiles, \
                f"{config_path}: execution_profile должен быть один из {valid_4h_profiles}, а не {execution_profile}"
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

    test_execution_profile_enum_has_4h_profiles()
    print()
    test_yaml_configs_use_4h_profiles()
    print()
    test_timing_yaml_has_4h_profiles()

    print()
    print("=" * 70)
    print("✓ ВСЕ ТЕСТЫ EXECUTION PROFILES ПРОЙДЕНЫ УСПЕШНО!")
    print("=" * 70)
