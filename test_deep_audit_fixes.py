#!/usr/bin/env python3
"""
Тест для проверки исправлений из глубокого аудита интеграции 11 признаков для 4h интервала.
Проверяет что все найденные проблемы исправлены.
"""

import sys
from typing import Dict, List

def test_feature_name_consistency():
    """Проверка соответствия имен признаков между transformers.py и mediator.py"""
    print("=" * 80)
    print("ТЕСТ 1: Проверка соответствия имен признаков")
    print("=" * 80)

    # Импортируем функцию форматирования имен из transformers
    from transformers import _format_window_name

    # Определяем окна GARCH из transformers.py (строка 449)
    garch_windows_minutes = [50 * 240, 14 * 24 * 60, 30 * 24 * 60]  # 12000, 20160, 43200

    # Генерируем имена признаков
    garch_feature_names = [
        f"garch_{_format_window_name(w)}" for w in garch_windows_minutes
    ]

    print(f"\nГенерируемые имена GARCH признаков:")
    for i, (window, name) in enumerate(zip(garch_windows_minutes, garch_feature_names)):
        print(f"  {i+1}. {name} (окно={window} минут = {window/1440:.2f} дней)")

    # Читаем mediator.py и проверяем какие имена используются
    with open("mediator.py", "r") as f:
        mediator_content = f.read()

    expected_names = ["garch_200h", "garch_14d", "garch_30d"]  # 50 баров = 12000 мин = 200h, минимум для GARCH на 4h

    print(f"\nОжидаемые имена в mediator.py:")
    for i, name in enumerate(expected_names):
        print(f"  {i+1}. {name}")

    print(f"\nПроверка соответствия:")
    all_match = True
    for generated, expected in zip(garch_feature_names, expected_names):
        match = generated == expected
        status = "✅" if match else "❌"
        print(f"  {status} {generated} == {expected}: {match}")
        if not match:
            all_match = False

    # Проверяем что garch_200h присутствует в mediator.py
    if "garch_200h" in mediator_content:
        print(f"\n✅ Признак 'garch_200h' найден в mediator.py")
    else:
        print(f"\n❌ Признак 'garch_200h' НЕ найден в mediator.py")
        all_match = False

    # Проверяем что старое имя garch_7d больше НЕ используется
    if "garch_7d" not in mediator_content or mediator_content.count("garch_7d") == 0:
        print(f"✅ Старое имя 'garch_7d' не используется в mediator.py")
    else:
        # Проверяем что это не упоминание в комментариях или истории
        import re
        # Ищем использование в коде (не в комментариях)
        code_usage = re.findall(r'["\']garch_7d["\']', mediator_content)
        if code_usage:
            print(f"❌ Старое имя 'garch_7d' все еще используется в коде: {len(code_usage)} раз")
            all_match = False
        else:
            print(f"✅ 'garch_7d' упоминается только в комментариях (OK)")

    return all_match


def test_all_feature_names():
    """Проверка всех имен признаков из 11 новых"""
    print("\n" + "=" * 80)
    print("ТЕСТ 2: Проверка всех 11 новых признаков")
    print("=" * 80)

    from transformers import _format_window_name

    # Все окна и их ожидаемые имена
    feature_windows = {
        "returns": {
            "windows": [240, 720, 1440],
            "prefix": "ret_",
            "expected": ["ret_4h", "ret_12h", "ret_24h"]
        },
        "yang_zhang": {
            "windows": [48 * 60, 7 * 24 * 60, 30 * 24 * 60],
            "prefix": "yang_zhang_",
            "expected": ["yang_zhang_48h", "yang_zhang_7d", "yang_zhang_30d"]
        },
        "parkinson": {
            "windows": [48 * 60, 7 * 24 * 60],
            "prefix": "parkinson_",
            "expected": ["parkinson_48h", "parkinson_7d"]
        },
        "garch": {
            "windows": [50 * 240, 14 * 24 * 60, 30 * 24 * 60],
            "prefix": "garch_",
            "expected": ["garch_200h", "garch_14d", "garch_30d"]  # 50 баров = 12000 мин = 200h, минимум для GARCH на 4h
        },
        "cvd": {
            "windows": [24 * 60, 7 * 24 * 60],
            "prefix": "cvd_",
            "expected": ["cvd_24h", "cvd_7d"]
        },
        "taker_buy_ratio_sma": {
            "windows": [8 * 60, 16 * 60, 24 * 60],
            "prefix": "taker_buy_ratio_sma_",
            "expected": ["taker_buy_ratio_sma_8h", "taker_buy_ratio_sma_16h", "taker_buy_ratio_sma_24h"]
        },
        "taker_buy_ratio_momentum": {
            "windows": [4 * 60, 8 * 60, 12 * 60],
            "prefix": "taker_buy_ratio_momentum_",
            "expected": ["taker_buy_ratio_momentum_4h", "taker_buy_ratio_momentum_8h", "taker_buy_ratio_momentum_12h"]
        }
    }

    all_match = True

    for feature_type, config in feature_windows.items():
        print(f"\n{feature_type.upper()}:")
        for window, expected_name in zip(config["windows"], config["expected"]):
            generated_name = f"{config['prefix']}{_format_window_name(window)}"
            match = generated_name == expected_name
            status = "✅" if match else "❌"
            print(f"  {status} {generated_name} == {expected_name}")
            if not match:
                all_match = False

    return all_match


def test_mediator_norm_cols_indices():
    """Проверка индексов в _extract_norm_cols"""
    print("\n" + "=" * 80)
    print("ТЕСТ 3: Проверка индексов norm_cols в mediator.py")
    print("=" * 80)

    # Читаем код mediator.py
    with open("mediator.py", "r") as f:
        content = f.read()

    # Ожидаемые признаки для каждого индекса
    expected_mapping = {
        0: "cvd_24h",
        1: "cvd_7d",
        2: "yang_zhang_48h",
        3: "yang_zhang_7d",
        4: "garch_200h",  # 50 баров = 12000 мин = 200h, минимум для GARCH на 4h
        5: "garch_14d",
        6: "ret_12h",
        7: "ret_24h",
        8: "ret_4h",
        9: "sma_12000",  # было sma_50. 50 баров = 12000 минут = 200h
        10: "yang_zhang_30d",
        11: "parkinson_48h",
        12: "parkinson_7d",
        13: "garch_30d",
        14: "taker_buy_ratio",
        15: "taker_buy_ratio_sma_24h",
        16: "taker_buy_ratio_sma_8h",
        17: "taker_buy_ratio_sma_16h",
        18: "taker_buy_ratio_momentum_4h",
        19: "taker_buy_ratio_momentum_8h",
        20: "taker_buy_ratio_momentum_12h",
    }

    print(f"\nПроверка маппинга (всего {len(expected_mapping)} признаков):")

    all_correct = True
    for idx, feature_name in expected_mapping.items():
        # Ищем строку вида: norm_cols[idx] = self._get_safe_float(row, "feature_name", 0.0)
        import re
        pattern = rf'norm_cols\[{idx}\]\s*=\s*self\._get_safe_float\(row,\s*["\'](\w+)["\']'
        matches = re.findall(pattern, content)

        if matches:
            actual_name = matches[0]
            match = actual_name == feature_name
            status = "✅" if match else "❌"
            print(f"  {status} norm_cols[{idx:2d}] = {actual_name:30s} (ожидается: {feature_name})")
            if not match:
                all_correct = False
        else:
            print(f"  ❌ norm_cols[{idx:2d}] = НЕ НАЙДЕН (ожидается: {feature_name})")
            all_correct = False

    return all_correct


def test_comment_accuracy():
    """Проверка точности комментариев в transformers.py"""
    print("\n" + "=" * 80)
    print("ТЕСТ 4: Проверка точности комментариев")
    print("=" * 80)

    with open("transformers.py", "r") as f:
        content = f.read()

    # Проверяем что старый неточный комментарий исправлен
    if "8d = 50 баров = 12000 минут" in content:
        print("❌ Старый неточный комментарий '8d = 50 баров = 12000 минут' все еще присутствует")
        return False
    else:
        print("✅ Старый неточный комментарий удален")

    # Проверяем что новый точный комментарий присутствует
    if "50 баров (8.33d) = 12000 минут = 200h" in content:
        print("✅ Новый точный комментарий '50 баров (8.33d) = 12000 минут = 200h' присутствует")
        return True
    else:
        print("❌ Новый точный комментарий не найден")
        return False


def main():
    """Запуск всех тестов"""
    print("\n" + "=" * 80)
    print("ТЕСТ ИСПРАВЛЕНИЙ ГЛУБОКОГО АУДИТА 11 ПРИЗНАКОВ ДЛЯ 4H ИНТЕРВАЛА")
    print("=" * 80)

    results = []

    # Тест 1: Соответствие имен GARCH признаков
    try:
        result1 = test_feature_name_consistency()
        results.append(("Соответствие имен GARCH признаков", result1))
    except Exception as e:
        print(f"❌ ОШИБКА в тесте 1: {e}")
        results.append(("Соответствие имен GARCH признаков", False))

    # Тест 2: Все имена признаков
    try:
        result2 = test_all_feature_names()
        results.append(("Все имена 11 признаков", result2))
    except Exception as e:
        print(f"❌ ОШИБКА в тесте 2: {e}")
        results.append(("Все имена 11 признаков", False))

    # Тест 3: Индексы в mediator.py
    try:
        result3 = test_mediator_norm_cols_indices()
        results.append(("Индексы norm_cols в mediator.py", result3))
    except Exception as e:
        print(f"❌ ОШИБКА в тесте 3: {e}")
        results.append(("Индексы norm_cols в mediator.py", False))

    # Тест 4: Точность комментариев
    try:
        result4 = test_comment_accuracy()
        results.append(("Точность комментариев", result4))
    except Exception as e:
        print(f"❌ ОШИБКА в тесте 4: {e}")
        results.append(("Точность комментариев", False))

    # Итоговый отчет
    print("\n" + "=" * 80)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print()
    print(f"Пройдено тестов: {passed}/{total}")

    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Исправления применены корректно.")
        return 0
    else:
        print(f"\n❌ ОБНАРУЖЕНЫ ПРОБЛЕМЫ! {total - passed} тест(ов) не прошли.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
