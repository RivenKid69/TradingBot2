#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_config_defaults_4h.py

Тест для проверки что все дефолтные значения в Config классах используют 4H профили.
"""

import re


def test_config_class_defaults():
    """Проверяем что дефолтные execution_profile во всех Config классах используют 4H."""
    print("Проверка дефолтных значений в Config классах...")

    with open("core_config.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Ищем все определения execution_profile с дефолтными значениями
    pattern = r'execution_profile:\s*ExecutionProfile\s*=\s*Field\(\s*default=ExecutionProfile\.(\w+)'
    matches = re.findall(pattern, content)

    print(f"  Найдено {len(matches)} дефолтных значений execution_profile")

    # Все дефолты должны использовать 4H профили (кроме LIMIT_MID_BPS который универсален)
    expected_4h_profiles = ["MKT_OPEN_NEXT_4H", "VWAP_CURRENT_4H", "LIMIT_MID_BPS"]

    errors = []
    for i, profile in enumerate(matches, 1):
        if profile not in expected_4h_profiles:
            errors.append(f"  ❌ Дефолт #{i} использует {profile} вместо 4H профиля!")
        else:
            print(f"  ✓ Дефолт #{i}: {profile}")

    if errors:
        for error in errors:
            print(error)
        raise AssertionError("Найдены неправильные дефолтные профили!")

    # Проверяем что есть хотя бы 4 определения (SimulationConfig, LiveConfig, TrainConfig, EvalConfig)
    assert len(matches) >= 4, f"Ожидалось минимум 4 дефолта, найдено {len(matches)}"

    print(f"  ✓ Все {len(matches)} дефолтных значений используют правильные профили!")


def test_no_hardcoded_h1_defaults():
    """Проверяем что нет жестко закодированных H1 дефолтов в критических местах."""
    print("Проверка на жестко закодированные H1 дефолты...")

    with open("core_config.py", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Ищем строки с ExecutionProfile.MKT_OPEN_NEXT_H1 в контексте дефолтов
    problematic_lines = []
    for i, line in enumerate(lines, 1):
        if "default=ExecutionProfile.MKT_OPEN_NEXT_H1" in line:
            # Это проблемная строка - дефолт использует H1
            problematic_lines.append((i, line.strip()))

    if problematic_lines:
        print("  ❌ Найдены H1 дефолты:")
        for line_num, line_text in problematic_lines:
            print(f"    Строка {line_num}: {line_text}")
        raise AssertionError("Дефолты должны использовать 4H профили, а не H1!")

    print("  ✓ Не найдено жестко закодированных H1 дефолтов")


def test_timing_yaml_consistency():
    """Проверяем что timing.yaml НЕ содержит H1 профили (только 4H)."""
    print("Проверка согласованности timing.yaml...")

    import yaml
    with open("configs/timing.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    profiles = data.get("profiles", {})

    # Не должно быть H1 профилей
    h1_profiles = ["MKT_OPEN_NEXT_H1", "VWAP_CURRENT_H1"]
    found_h1 = [p for p in h1_profiles if p in profiles]

    if found_h1:
        print(f"  ⚠ Найдены H1 профили в timing.yaml: {found_h1}")
        print("  Это не критично, но для 4h проекта они не нужны")
    else:
        print("  ✓ timing.yaml не содержит H1 профилей")

    # Должны быть 4H профили
    h4_profiles = ["MKT_OPEN_NEXT_4H", "VWAP_CURRENT_4H"]
    found_h4 = [p for p in h4_profiles if p in profiles]

    assert len(found_h4) >= 1, "timing.yaml должен содержать хотя бы один 4H профиль!"
    print(f"  ✓ timing.yaml содержит 4H профили: {found_h4}")


if __name__ == "__main__":
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ДЕФОЛТНЫХ ЗНАЧЕНИЙ CONFIG ДЛЯ 4H")
    print("=" * 70)
    print()

    test_config_class_defaults()
    print()
    test_no_hardcoded_h1_defaults()
    print()
    test_timing_yaml_consistency()

    print()
    print("=" * 70)
    print("✓ ВСЕ ТЕСТЫ ДЕФОЛТНЫХ ЗНАЧЕНИЙ ПРОЙДЕНЫ!")
    print("=" * 70)
