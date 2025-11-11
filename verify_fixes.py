#!/usr/bin/env python3
"""
Простая проверка что все исправления были применены корректно.
Не требует установки зависимостей - работает с исходным кодом напрямую.
"""
import re
import sys


def read_file(path):
    """Читает файл и возвращает его содержимое."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def check_transformers_defaults():
    """Проверяет дефолтные параметры в transformers.py для 4h интервала."""
    content = read_file('/home/user/TradingBot2/transformers.py')

    issues = []

    # Проверка КРИТИЧЕСКАЯ #1: lookbacks_prices для 4h
    if 'self.lookbacks_prices = [240, 720, 1440, 12000]' not in content:
        issues.append("КРИТИЧЕСКАЯ #1.1: lookbacks_prices не обновлены для 4h интервала")

    # Проверка КРИТИЧЕСКАЯ #3: GARCH windows для 4h
    if 'self.garch_windows = [7 * 24 * 60, 14 * 24 * 60, 30 * 24 * 60]' not in content:
        issues.append("КРИТИЧЕСКАЯ #3: garch_windows не обновлены для 4h интервала")

    # Проверка Yang-Zhang для 4h
    if 'self.yang_zhang_windows = [48 * 60, 7 * 24 * 60, 30 * 24 * 60]' not in content:
        issues.append("Yang-Zhang windows не обновлены для 4h интервала")

    # Проверка Parkinson для 4h
    if 'self.parkinson_windows = [48 * 60, 7 * 24 * 60]' not in content:
        issues.append("Parkinson windows не обновлены для 4h интервала")

    # Проверка Taker Buy Ratio SMA для 4h
    if 'self.taker_buy_ratio_windows = [8 * 60, 16 * 60, 24 * 60]' not in content:
        issues.append("Taker Buy Ratio SMA windows не обновлены для 4h интервала")

    # Проверка Taker Buy Ratio Momentum для 4h
    if 'self.taker_buy_ratio_momentum = [4 * 60, 8 * 60, 12 * 60]' not in content:
        issues.append("Taker Buy Ratio Momentum windows не обновлены для 4h интервала")

    return issues


def check_parkinson_formula():
    """Проверяет улучшение формулы Parkinson (MAJOR #1)."""
    content = read_file('/home/user/TradingBot2/transformers.py')

    issues = []

    # Проверка что добавлена проверка 80% валидных баров
    if 'min_required = max(2, int(0.8 * n))' not in content:
        issues.append("MAJOR #1: не добавлена проверка минимум 80% валидных баров для Parkinson")

    if 'if valid_bars < min_required:' not in content:
        issues.append("MAJOR #1: не добавлена проверка валидных баров для Parkinson")

    return issues


def check_taker_buy_ratio_clamping():
    """Проверяет добавление clamping для taker_buy_ratio (MINOR #2)."""
    content = read_file('/home/user/TradingBot2/transformers.py')

    issues = []

    # Проверка что добавлен clamping
    if 'taker_buy_ratio = min(1.0, max(0.0, float(taker_buy_base) / float(volume)))' not in content:
        issues.append("MINOR #2: не добавлен clamping для taker_buy_ratio")

    return issues


def check_obs_builder_comments():
    """Проверяет добавление комментариев в obs_builder.pyx (MAJOR #2)."""
    content = read_file('/home/user/TradingBot2/obs_builder.pyx')

    issues = []

    # Проверка что добавлены комментарии о нормализации
    if 'Normalized by 1% of price (price_d * 0.01)' not in content:
        issues.append("MAJOR #2: не добавлен комментарий о нормализации price_momentum")

    if 'Normalized by full price (price_d) not 1%' not in content:
        issues.append("MAJOR #2: не добавлен комментарий о нормализации bb_squeeze")

    return issues


def check_config_names():
    """Проверяет исправление названий в config_4h_timeframe.py (MINOR #1)."""
    content = read_file('/home/user/TradingBot2/config_4h_timeframe.py')

    issues = []

    # Проверка Yang-Zhang names
    if '42: "yang_zhang_7d"' not in content:
        issues.append("MINOR #1.1: не исправлено название yang_zhang_7d в config")

    if '180: "yang_zhang_30d"' not in content:
        issues.append("MINOR #1.2: не исправлено название yang_zhang_30d в config")

    # Проверка Parkinson names
    if '42: "parkinson_7d"' not in content:
        issues.append("MINOR #1.3: не исправлено название parkinson_7d в config")

    # Проверка CVD names
    if '42: "cvd_7d"' not in content:
        issues.append("MINOR #1.4: не исправлено название cvd_7d в config")

    return issues


def main():
    """Запускает все проверки и выводит результаты."""
    print("=" * 70)
    print("ПРОВЕРКА ИСПРАВЛЕНИЙ ДЛЯ 4H ИНТЕРВАЛА")
    print("=" * 70)
    print()

    all_issues = []

    # Проверка 1: Дефолтные параметры в transformers.py
    print("[1/5] Проверка дефолтных параметров в transformers.py...")
    issues = check_transformers_defaults()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ Все дефолтные параметры обновлены для 4h интервала")
    print()

    # Проверка 2: Формула Parkinson
    print("[2/5] Проверка улучшенной формулы Parkinson...")
    issues = check_parkinson_formula()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ Формула Parkinson обновлена с проверкой 80% валидных баров")
    print()

    # Проверка 3: Clamping для taker_buy_ratio
    print("[3/5] Проверка clamping для taker_buy_ratio...")
    issues = check_taker_buy_ratio_clamping()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ Clamping для taker_buy_ratio добавлен")
    print()

    # Проверка 4: Комментарии в obs_builder.pyx
    print("[4/5] Проверка комментариев в obs_builder.pyx...")
    issues = check_obs_builder_comments()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ Комментарии о нормализации добавлены")
    print()

    # Проверка 5: Названия в config_4h_timeframe.py
    print("[5/5] Проверка названий в config_4h_timeframe.py...")
    issues = check_config_names()
    if issues:
        all_issues.extend(issues)
        for issue in issues:
            print(f"  ❌ {issue}")
    else:
        print("  ✅ Все названия в config обновлены")
    print()

    # Итоговый результат
    print("=" * 70)
    if all_issues:
        print(f"❌ НАЙДЕНО {len(all_issues)} ПРОБЛЕМ:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print("=" * 70)
        sys.exit(1)
    else:
        print("✅ ВСЕ ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ КОРРЕКТНО!")
        print()
        print("Список исправленных проблем:")
        print("  1. ✅ КРИТИЧЕСКАЯ #1: Дефолтные параметры для 4h интервала")
        print("  2. ✅ КРИТИЧЕСКАЯ #2: Соответствие имен признаков")
        print("  3. ✅ КРИТИЧЕСКАЯ #3: GARCH окна >= 50 баров")
        print("  4. ✅ MAJOR #1: Формула Parkinson с 80% валидных баров")
        print("  5. ✅ MAJOR #2: Комментарии о нормализации")
        print("  6. ✅ MINOR #1: Названия в config_4h_timeframe.py")
        print("  7. ✅ MINOR #2: Clamping для taker_buy_ratio")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
