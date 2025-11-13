"""
Конфигурация признаков для 4-часового интервала (4h).

Этот файл содержит рекомендуемые параметры для всех признаков
при переходе с 1-минутного интервала на 4-часовой.

Основной принцип пересчета:
- 1 бар на 4h = 240 минут
- Временное окно в минутах / 240 = окно в барах
- Сохраняем те же временные периоды, но в барах

Автор: Анализ системы признаков
Дата: 2025-11-11
"""

from typing import List

# =============================================================================
# ПАРАМЕТРЫ ДЛЯ 4H ИНТЕРВАЛА
# =============================================================================

# Базовый интервал
TIMEFRAME = "4h"
BAR_DURATION_MINUTES = 240  # 4 часа = 240 минут


# =============================================================================
# SMA (Simple Moving Averages) — в барах
# =============================================================================

# На 1m было: [5, 15, 60] минут
# Пересчет для 4h:
# - 5 минут → 0.02 бара (не имеет смысла)
# - 15 минут → 0.06 бара (не имеет смысла)
# - 60 минут → 0.25 бара (не имеет смысла)

# Рекомендуемые значения для 4h (в барах):
SMA_LOOKBACKS = [5, 21, 50]
# 5 баров = 20 часов (краткосрочный)
# 21 баров = 84 часа = 3.5 дня (среднесрочный, близко к недельному)
# 50 баров = 200 часов = 8.3 дня (долгосрочный, классический SMA50)

# Альтернативный вариант с SMA200:
# SMA_LOOKBACKS = [5, 21, 200]
# 200 баров = 800 часов = 33 дня (классический SMA200 для долгосрочного тренда)


# =============================================================================
# RETURNS (Логарифмические доходности) — в барах
# =============================================================================

# На 1m было: [5, 15, 60] минут
# Эти периоды не имеют смысла на 4h (меньше одного бара)

# Рекомендуемые значения для 4h (в барах):
RETURN_LOOKBACKS = [1, 3, 6, 42]
# 1 бар = 4 часа (ret_4h)
# 3 бара = 12 часов (ret_12h)
# 6 баров = 24 часа = 1 день (ret_24h, ret_1d)
# 42 бара = 168 часов = 7 дней (ret_7d)

# Названия для маппинга:
RETURN_NAMES = {
    1: "ret_4h",
    3: "ret_12h",
    6: "ret_24h",  # или ret_1d
    42: "ret_7d",
}


# =============================================================================
# RSI (Relative Strength Index)
# =============================================================================

# Стандартный период RSI универсален для всех таймфреймов
RSI_PERIOD = 14  # 14 баров на 4h = 56 часов


# =============================================================================
# YANG-ZHANG VOLATILITY — в барах
# =============================================================================

# На 1m было: [1440, 10080, 43200] минут = [24h, 7d, 30d]
# Пересчет для 4h:
YANG_ZHANG_WINDOWS = [12, 42, 180]
# 12 баров = 48 часов = 2 дня (минимально рекомендуемое окно для Yang-Zhang)
# 42 бара = 168 часов = 7 дней
# 180 баров = 720 часов = 30 дней

# ВАЖНО: Yang-Zhang требует минимум 20+ баров для стабильных оценок
# Окно 6 баров (24h) на 4h слишком мало — используем 12 баров (48h)

YANG_ZHANG_NAMES = {
    12: "yang_zhang_48h",  # или yang_zhang_2d
    42: "yang_zhang_7d",   # или yang_zhang_168h
    180: "yang_zhang_30d",  # или yang_zhang_720h
}


# =============================================================================
# PARKINSON VOLATILITY — в барах
# =============================================================================

# На 1m было: [1440, 10080] минут = [24h, 7d]
# Пересчет для 4h:
PARKINSON_WINDOWS = [12, 42]
# 12 баров = 48 часов = 2 дня (вместо 24h для большей стабильности)
# 42 бара = 168 часов = 7 дней

PARKINSON_NAMES = {
    12: "parkinson_48h",  # или parkinson_2d
    42: "parkinson_7d",   # или parkinson_168h
}


# =============================================================================
# GARCH VOLATILITY — в барах
# =============================================================================

# КРИТИЧНО: GARCH требует минимум 50-100 наблюдений для стабильной оценки!
# На 1m было: [500, 720, 1440] минут (500-1440 баров данных)
# На 4h это превращается в 2-6 баров — НЕДОСТАТОЧНО

# ВАРИАНТ 1: Использовать длинные окна (рекомендуется)
GARCH_WINDOWS_OPTION_1 = [50, 84, 180]
# 50 баров = 200 часов = 8.33 дня (минимум для GARCH, требуется >= 50 наблюдений)
# 84 бара = 336 часов = 14 дней
# 180 баров = 720 часов = 30 дней

GARCH_NAMES_OPTION_1 = {
    50: "garch_200h",
    84: "garch_14d",
    180: "garch_30d",
}

# ВАРИАНТ 2: Не использовать GARCH на 4h (заменить на более простые метрики)
GARCH_WINDOWS_OPTION_2: List[int] = []

# Выберите один из вариантов:
USE_GARCH = True  # False = не использовать GARCH на 4h
GARCH_WINDOWS = GARCH_WINDOWS_OPTION_1 if USE_GARCH else GARCH_WINDOWS_OPTION_2
GARCH_NAMES = GARCH_NAMES_OPTION_1 if USE_GARCH else {}


# =============================================================================
# TAKER BUY RATIO — в барах
# =============================================================================

# На 1m было:
# - SMA windows: [360, 720, 1440] минут = [6h, 12h, 24h]
# - Momentum windows: [60, 360, 720] минут = [1h, 6h, 12h]

# Пересчет для 4h:
TAKER_BUY_RATIO_SMA_WINDOWS = [2, 4, 6]
# 2 бара = 8 часов
# 4 бара = 16 часов
# 6 баров = 24 часа

TAKER_BUY_RATIO_MOMENTUM_WINDOWS = [1, 2, 3, 6]
# 1 бар = 4 часа (вместо 1h)
# 2 бара = 8 часов
# 3 бара = 12 часов
# 6 баров = 24 часа (новое, для долгосрочного моментума)

TAKER_BUY_RATIO_SMA_NAMES = {
    2: "taker_buy_ratio_sma_8h",
    4: "taker_buy_ratio_sma_16h",
    6: "taker_buy_ratio_sma_24h",
}

TAKER_BUY_RATIO_MOMENTUM_NAMES = {
    1: "taker_buy_ratio_momentum_4h",
    2: "taker_buy_ratio_momentum_8h",
    3: "taker_buy_ratio_momentum_12h",
    6: "taker_buy_ratio_momentum_24h",
}


# =============================================================================
# CVD (Cumulative Volume Delta) — в барах
# =============================================================================

# На 1m было: [1440, 10080] минут = [24h, 7d]
# Пересчет для 4h:
CVD_WINDOWS = [6, 42]
# 6 баров = 24 часа = 1 день
# 42 бара = 168 часов = 7 дней

CVD_NAMES = {
    6: "cvd_24h",
    42: "cvd_7d",  # или cvd_168h
}


# =============================================================================
# MICROSTRUCTURE FEATURES — ЗАМЕНА
# =============================================================================

# На 1m использовались микроструктурные признаки:
# - ofi_proxy (Order Flow Imbalance)
# - qimb (Quote Imbalance)
# - micro_dev (Microstructure Deviation)

# На 4h эти признаки НЕ ПРИМЕНИМЫ (требуют высокочастотные данные)

# ВАРИАНТЫ ЗАМЕНЫ:

# Вариант A: Свечные паттерны (рекомендуется)
USE_CANDLESTICK_PATTERNS = True
CANDLESTICK_PATTERNS = [
    "doji",              # Доджи (неопределенность)
    "hammer",            # Молот (разворот вверх)
    "shooting_star",     # Падающая звезда (разворот вниз)
    "engulfing_bull",    # Бычье поглощение
    "engulfing_bear",    # Медвежье поглощение
    "inside_bar",        # Внутренний бар (консолидация)
]

# Вариант B: Макро-индикаторы
USE_MACRO_INDICATORS = False
MACRO_INDICATORS = [
    "price_to_sma_ratio",  # Отношение цены к SMA
    "trend_strength",       # Сила тренда (ADX-подобный)
    "regime_volatility",    # Режим волатильности (high/low)
]

# Вариант C: Удалить (заполнить нулями)
# Установите оба флага в False

# Вариант D: Support/Resistance индикаторы
USE_SUPPORT_RESISTANCE = False
SUPPORT_RESISTANCE = [
    "distance_to_support",    # Расстояние до ближайшей поддержки
    "distance_to_resistance", # Расстояние до ближайшего сопротивления
    "breakout_indicator",     # Индикатор пробоя
]


# =============================================================================
# NORMALIZATION PARAMETERS — для volume
# =============================================================================

# На 1m использовались:
# - log_volume_norm: tanh(log1p(quote_volume / 1e6))
# - rel_volume: tanh(log1p(volume / 100))

# На 4h объемы будут в ~240 раз больше, нужна перекалибровка

# Рекомендуемые значения для 4h:
VOLUME_NORM_DIVISOR = 240e6  # 240 * 1e6 (примерно)
REL_VOLUME_DIVISOR = 24000    # 240 * 100 (примерно)

# Эти значения нужно откалибровать на реальных 4h данных!
# Используйте percentile-based calibration:
# - Найдите медиану и 95-й перцентиль объема на 4h
# - Установите divisor так, чтобы медиана ≈ 0.5 после tanh(log1p(volume/divisor))


# =============================================================================
# METADATA — time_since_event нормализация
# =============================================================================

# На 1m использовалось: tanh(time_since_event / 24)
# На 4h можно использовать меньший делитель, так как один бар = 4 часа

TIME_SINCE_EVENT_DIVISOR = 6  # tanh(time_since_event / 6)
# Это нормализует события в пределах ~24 часов (6 баров * 4 часа)


# =============================================================================
# ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ИЗ MARKETSIMULATOR
# =============================================================================

# Эти индикаторы используют стандартные периоды, универсальные для всех таймфреймов:

# MACD: (12, 26, 9) — стандартные периоды
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bollinger Bands: (20, 2) — стандартные параметры
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

# ATR: 14 — стандартный период
ATR_PERIOD = 14

# CCI: 20 — стандартный период
CCI_PERIOD = 20

# Momentum: период нужно определить (например, 10)
MOMENTUM_PERIOD = 10


# =============================================================================
# СВОДКА ИЗМЕНЕНИЙ
# =============================================================================

SUMMARY = """
ИЗМЕНЕНИЯ ПРИ ПЕРЕХОДЕ С 1M НА 4H:

1. SMA: [5, 15, 60] минут → [5, 21, 50] баров (20h, 3.5d, 8d)
2. Returns: [5, 15, 60] минут → [1, 3, 6, 42] баров (4h, 12h, 1d, 7d)
3. RSI: 14 (без изменений)
4. Yang-Zhang: [24h, 7d, 30d] → [12, 42, 180] баров (48h, 7d, 30d)
5. Parkinson: [24h, 7d] → [12, 42] баров (48h, 7d)
6. GARCH:
   - ВАРИАНТ 1: [50, 84, 180] баров (200h, 14d, 30d) - минимум 50 наблюдений
   - ВАРИАНТ 2: УДАЛИТЬ (не использовать на 4h)
7. Taker Buy Ratio SMA: [6h, 12h, 24h] → [2, 4, 6] баров (8h, 16h, 24h)
8. Taker Buy Ratio Momentum: [1h, 6h, 12h] → [1, 2, 3, 6] баров (4h, 8h, 12h, 24h)
9. CVD: [24h, 7d] → [6, 42] баров (24h, 7d)
10. Microstructure: УДАЛИТЬ → ЗАМЕНИТЬ на свечные паттерны

КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ:
- Все временные окна пересчитаны с минут в бары (/ 240)
- Микроструктурные признаки заменены на свечные паттерны
- GARCH либо удален, либо использует длинные окна (200h-30 дней, минимум 50 наблюдений)
- Короткие ретёрны (5m, 15m, 60m) заменены на 4h, 12h, 24h, 7d
- Нормализация объемов требует перекалибровки

КОЛИЧЕСТВО ПРИЗНАКОВ:
- Было: 56 признаков
- Станет: 56-60 признаков (в зависимости от замены микроструктуры)
  - Если используем 3 свечных паттерна: 56
  - Если используем 6 свечных паттернов: 59
"""


# =============================================================================
# ЭКСПОРТ КОНФИГУРАЦИИ ДЛЯ ИСПОЛЬЗОВАНИЯ В КОДЕ
# =============================================================================

def get_feature_spec_4h():
    """
    Возвращает FeatureSpec для 4h интервала.

    Используется в transformers.py для создания OnlineFeatureTransformer.

    ВАЖНО: Все параметры (SMA_LOOKBACKS, YANG_ZHANG_WINDOWS, etc.) определены в БАРАХ,
    но FeatureSpec ожидает МИНУТЫ. Поэтому мы конвертируем: бары * 240 минут/бар.

    ОБНОВЛЕНИЕ: lookbacks_prices теперь содержит компромиссные окна для SMA и returns,
    так как transformers.py использует один параметр для обоих признаков.
    Окна: [240, 720, 1440, 5040, 10080, 12000] минут = [4h, 12h, 24h, 3.5d, 7d, 8.3d]
    Это охватывает основные returns (4h, 12h, 24h, 7d) и важные SMA (3.5d, 8.3d).
    """
    from transformers import FeatureSpec

    # Компромиссные окна для SMA и returns (в барах)
    # [1, 3, 6, 21, 42, 50] = [4h, 12h, 24h, 3.5d, 7d, 8.3d]
    COMBINED_LOOKBACKS = [1, 3, 6, 21, 42, 50]

    return FeatureSpec(
        lookbacks_prices=[x * BAR_DURATION_MINUTES for x in COMBINED_LOOKBACKS],
        rsi_period=RSI_PERIOD,
        yang_zhang_windows=[x * BAR_DURATION_MINUTES for x in YANG_ZHANG_WINDOWS],
        parkinson_windows=[x * BAR_DURATION_MINUTES for x in PARKINSON_WINDOWS],
        garch_windows=[x * BAR_DURATION_MINUTES for x in GARCH_WINDOWS],
        taker_buy_ratio_windows=[x * BAR_DURATION_MINUTES for x in TAKER_BUY_RATIO_SMA_WINDOWS],
        taker_buy_ratio_momentum=[x * BAR_DURATION_MINUTES for x in TAKER_BUY_RATIO_MOMENTUM_WINDOWS],
        cvd_windows=[x * BAR_DURATION_MINUTES for x in CVD_WINDOWS],
        bar_duration_minutes=BAR_DURATION_MINUTES,  # КРИТИЧНО!
    )


def get_config_summary():
    """Выводит сводку конфигурации."""
    print(SUMMARY)
    print("\nПАРАМЕТРЫ:")
    print(f"  Timeframe: {TIMEFRAME}")
    print(f"  Bar duration: {BAR_DURATION_MINUTES} minutes")
    print(f"\nSMA lookbacks (bars): {SMA_LOOKBACKS}")
    print(f"  → Times: {[f'{b*4}h' for b in SMA_LOOKBACKS]}")
    print(f"\nReturn lookbacks (bars): {RETURN_LOOKBACKS}")
    print(f"  → Times: {[RETURN_NAMES[b] for b in RETURN_LOOKBACKS]}")
    print(f"\nRSI period: {RSI_PERIOD} bars")
    print(f"\nYang-Zhang windows (bars): {YANG_ZHANG_WINDOWS}")
    print(f"  → Times: {[YANG_ZHANG_NAMES[w] for w in YANG_ZHANG_WINDOWS]}")
    print(f"\nParkinson windows (bars): {PARKINSON_WINDOWS}")
    print(f"  → Times: {[PARKINSON_NAMES[w] for w in PARKINSON_WINDOWS]}")

    if USE_GARCH:
        print(f"\nGARCH windows (bars): {GARCH_WINDOWS}")
        print(f"  → Times: {[GARCH_NAMES[w] for w in GARCH_WINDOWS]}")
    else:
        print("\nGARCH: DISABLED (not recommended for 4h)")

    print(f"\nTaker Buy Ratio SMA (bars): {TAKER_BUY_RATIO_SMA_WINDOWS}")
    print(f"  → Times: {[TAKER_BUY_RATIO_SMA_NAMES[w] for w in TAKER_BUY_RATIO_SMA_WINDOWS]}")
    print(f"\nTaker Buy Ratio Momentum (bars): {TAKER_BUY_RATIO_MOMENTUM_WINDOWS}")
    print(f"  → Times: {[TAKER_BUY_RATIO_MOMENTUM_NAMES[w] for w in TAKER_BUY_RATIO_MOMENTUM_WINDOWS]}")
    print(f"\nCVD windows (bars): {CVD_WINDOWS}")
    print(f"  → Times: {[CVD_NAMES[w] for w in CVD_WINDOWS]}")

    print("\nMICROSTRUCTURE REPLACEMENT:")
    if USE_CANDLESTICK_PATTERNS:
        print(f"  Using candlestick patterns: {CANDLESTICK_PATTERNS}")
    elif USE_MACRO_INDICATORS:
        print(f"  Using macro indicators: {MACRO_INDICATORS}")
    elif USE_SUPPORT_RESISTANCE:
        print(f"  Using support/resistance: {SUPPORT_RESISTANCE}")
    else:
        print("  Disabled (filled with zeros)")


if __name__ == "__main__":
    get_config_summary()
    print("\n" + "="*70)
    print("Для использования в коде:")
    print("  from config_4h_timeframe import get_feature_spec_4h")
    print("  spec = get_feature_spec_4h()")
    print("="*70)
