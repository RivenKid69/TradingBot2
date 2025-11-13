"""
Тест для проверки всех исправлений миграции с 1h на 4h таймфрейм.

Этот тест проверяет:
1. sigma_window = 42 в feature_pipe.py, no_trade.py, dynamic_no_trade_guard.py
2. SMA имена в mediator.py (sma_1200, sma_5040)
3. timeframe_ms = 14400000 в core_config.py
4. Дефолтные параметры в transformers.py для 4h

Автор: Claude AI
Дата: 2025-11-13
"""

import sys
import os

def test_sigma_window_in_feature_pipe():
    """Проверяем что sigma_window = 42 в feature_pipe.py"""
    with open('feature_pipe.py', 'r') as f:
        content = f.read()
        assert 'sigma_window: int = 42' in content, \
            "feature_pipe.py должен иметь sigma_window: int = 42"
        assert '42 × 4h = 168h = 7 дней' in content, \
            "feature_pipe.py должен иметь комментарий о 7 днях для 4h"
    print("✓ feature_pipe.py: sigma_window = 42")

def test_sigma_window_in_no_trade():
    """Проверяем что sigma_window = 42 в no_trade.py"""
    with open('no_trade.py', 'r') as f:
        content = f.read()
        assert 'sigma_window = 42' in content, \
            "no_trade.py должен иметь sigma_window = 42"
        assert '42 × 4h = 168h = 7 дней' in content, \
            "no_trade.py должен иметь комментарий о 7 днях для 4h"
    print("✓ no_trade.py: sigma_window = 42")

def test_sigma_window_in_dynamic_no_trade_guard():
    """Проверяем что sigma_window = 42 в dynamic_no_trade_guard.py"""
    with open('dynamic_no_trade_guard.py', 'r') as f:
        content = f.read()
        # Должно быть 2 упоминания: в getattr(..., 42) и в if sigma_window = 42
        assert content.count('42') >= 2, \
            "dynamic_no_trade_guard.py должен иметь 2 упоминания 42"
        assert '42 × 4h = 168h = 7 дней' in content, \
            "dynamic_no_trade_guard.py должен иметь комментарий о 7 днях для 4h"
    print("✓ dynamic_no_trade_guard.py: sigma_window = 42")

def test_sma_names_in_mediator():
    """Проверяем правильные имена SMA в mediator.py"""
    with open('mediator.py', 'r') as f:
        content = f.read()
        assert '"sma_1200"' in content, \
            "mediator.py должен использовать sma_1200 (5 баров × 240 мин)"
        assert '"sma_5040"' in content, \
            "mediator.py должен использовать sma_5040 (21 бар × 240 мин)"
        # Проверяем что старые имена удалены
        assert '"sma_5"' not in content or 'sma_1200' in content.split('"sma_5"')[0][-100:], \
            "mediator.py не должен использовать sma_5 без обновления"
        assert '"sma_21"' not in content or 'sma_5040' in content.split('"sma_21"')[0][-100:], \
            "mediator.py не должен использовать sma_21 без обновления"
    print("✓ mediator.py: SMA имена обновлены (sma_1200, sma_5040)")

def test_timeframe_ms_in_core_config():
    """Проверяем что timeframe_ms = 14400000 в core_config.py"""
    with open('core_config.py', 'r') as f:
        content = f.read()
        assert 'timeframe_ms: int = Field(default=14_400_000)' in content, \
            "core_config.py должен иметь timeframe_ms = 14_400_000 (4h)"
        assert '4h timeframe' in content, \
            "core_config.py должен иметь комментарий о 4h timeframe"
    print("✓ core_config.py: timeframe_ms = 14_400_000 (4h)")

def test_timeframe_default_in_app():
    """Проверяем дефолтный timeframe в app.py"""
    with open('app.py', 'r') as f:
        content = f.read()
        assert 'timeframe", "4h")' in content, \
            'app.py должен иметь дефолт timeframe="4h"'
        # Проверяем что старый дефолт "1m" заменен
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'timeframe=getattr(sim_cfg.data, "timeframe"' in line:
                assert '"4h")' in line, \
                    f"app.py line {i+1}: timeframe должен быть '4h', не '1m'"
    print("✓ app.py: timeframe default = '4h'")

def test_bar_duration_in_transformers():
    """Проверяем bar_duration_minutes = 240 в transformers.py"""
    with open('transformers.py', 'r') as f:
        content = f.read()
        assert 'bar_duration_minutes: int = 240' in content, \
            "transformers.py должен иметь bar_duration_minutes = 240 (4h)"
        assert '4h timeframe' in content, \
            "transformers.py должен иметь комментарий о 4h timeframe"
    print("✓ transformers.py: bar_duration_minutes = 240 (4h)")

def test_lookbacks_default_in_transformers():
    """Проверяем дефолтные lookbacks для 4h в transformers.py"""
    with open('transformers.py', 'r') as f:
        content = f.read()
        # Проверяем что дефолтные lookbacks содержат правильные значения для 4h
        assert '[240, 720, 1200, 1440, 5040, 10080, 12000]' in content, \
            "transformers.py должен иметь правильные дефолтные lookbacks для 4h"
        # Проверяем окна Yang-Zhang для 4h
        assert '[48 * 60, 7 * 24 * 60, 30 * 24 * 60]' in content or \
               '[2880, 10080, 43200]' in content, \
            "transformers.py должен иметь правильные окна Yang-Zhang для 4h"
    print("✓ transformers.py: lookbacks и окна правильные для 4h")

def main():
    """Запускаем все тесты"""
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЙ МИГРАЦИИ 1H → 4H")
    print("="*70 + "\n")

    try:
        test_sigma_window_in_feature_pipe()
        test_sigma_window_in_no_trade()
        test_sigma_window_in_dynamic_no_trade_guard()
        test_sma_names_in_mediator()
        test_timeframe_ms_in_core_config()
        test_timeframe_default_in_app()
        test_bar_duration_in_transformers()
        test_lookbacks_default_in_transformers()

        print("\n" + "="*70)
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*70 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n✗ ТЕСТ ПРОВАЛЕН: {e}\n")
        return 1
    except Exception as e:
        print(f"\n✗ ОШИБКА ПРИ ВЫПОЛНЕНИИ ТЕСТА: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
