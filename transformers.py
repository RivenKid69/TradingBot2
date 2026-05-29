# transformers.py
"""
Feature transformation module for technical indicators and volatility models.

DEFENSIVE EXCEPTION PATTERNS
============================
This module uses intentional silent exception handling in specific contexts:

1. **GARCH Model Fallback** (lines ~531-533):
   - Pattern: `except (ValueError, RuntimeError, LinAlgError, Exception): pass`
   - Purpose: GARCH optimization may fail to converge; fallback to EWMA is expected behavior
   - Cascade: GARCH → EWMA → Historical Volatility (three-tier fallback)
   - Rationale: Model non-convergence is data-dependent, not a bug

2. **OHLCV Parsing** (lines ~1437-1459):
   - Pattern: `try: float(bar[key]) except Exception: pass`
   - Purpose: Parse available OHLCV fields; skip malformed values silently
   - Rationale: Continue with available data rather than fail entire update

3. **Data Validation Skip** (line ~486):
   - Pattern: `if invalid_data: pass` (not exception, control flow)
   - Purpose: Skip GARCH attempt if data is unsuitable

These patterns are INTENTIONAL per trading system requirements:
- Data continuity takes precedence over strict validation
- Fallback cascades ensure feature availability
- Silent handling prevents noisy logs during normal operation

Tech Debt Tracking: docs/reports/TECH_DEBT_REGISTRY.md#data-transformers-defensive-exceptions
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import warnings

import pandas as pd
import numpy as np
from arch import arch_model


def _format_window_name(window_minutes: int) -> str:
    """
    Форматирует имя окна в зависимости от величины для 4h интервала.

    Логика:
    - Для GARCH (длинные окна >= 7 дней): используем дни (7d, 14d, 30d)
    - Для остальных признаков: используем часы (4h, 12h, 24h, 48h, 168h, 720h)
    - Для очень коротких окон: используем минуты

    Args:
        window_minutes: Размер окна в минутах

    Returns:
        Строка формата "Xd" (дни для длинных окон), "Xh" (часы) или "Xm" (минуты)

    Examples:
        >>> _format_window_name(10080)  # 7 дней (GARCH)
        '7d'
        >>> _format_window_name(1440)   # 24 часа (не GARCH)
        '24h'
        >>> _format_window_name(240)    # 4 часа
        '4h'
        >>> _format_window_name(42)     # 42 минуты
        '42m'
    """
    # Для длинных окон >= 7 дней (10080 минут) используем дни (для GARCH)
    if window_minutes >= 10080 and window_minutes % 1440 == 0:  # >= 7 дней и кратно дню
        return f"{window_minutes // 1440}d"
    elif window_minutes >= 60 and window_minutes % 60 == 0:     # часы
        return f"{window_minutes // 60}h"
    else:                                                        # минуты
        return f"{window_minutes}m"


def calculate_close_to_close_volatility(close_prices: List[float], n: int) -> Optional[float]:
    """
    Рассчитывает стандартную close-to-close волатильность (стандартное отклонение log returns).

    Используется как fallback метод когда OHLC данные недоступны для Yang-Zhang волатильности.
    Согласно исследованиям (Quant StackExchange), это рекомендуемый подход когда
    полные OHLC данные недоступны.

    Формула:
    σ = sqrt((1/(n-1)) Σ(r_i - μ)²)
    где r_i = log(C_i/C_{i-1}) - логарифмический возврат

    Args:
        close_prices: список цен закрытия
        n: размер окна

    Returns:
        Close-to-close волатильность или None если недостаточно данных
    """
    if not close_prices or len(close_prices) < n or n < 2:
        return None

    # Берем последние n цен
    prices = list(close_prices)[-n:]

    try:
        # Рассчитываем log returns
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                log_returns.append(math.log(prices[i] / prices[i-1]))

        if len(log_returns) < 2:
            return None

        # Рассчитываем среднее и стандартное отклонение
        mean_return = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_return) ** 2 for r in log_returns) / (len(log_returns) - 1)

        if variance < 0:
            return None

        return math.sqrt(variance)

    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None


def calculate_yang_zhang_volatility(
    ohlc_bars: List[Dict[str, float]],
    n: int,
    close_prices: Optional[List[float]] = None
) -> Optional[float]:
    """
    Рассчитывает волатильность Yang-Zhang для последних n баров.

    КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Добавлен fallback к close-to-close volatility
    когда OHLC данные недостаточны или недоступны.

    Формула Yang-Zhang:
    σ²_YZ = σ²_o + k·σ²_c + (1-k)·σ²_rs
    где:
    - σ²_o = ночная волатильность = (1/(n-1)) Σ(log(O_i/C_{i-1}) - μ_o)²
    - σ²_c = волатильность open-close = (1/(n-1)) Σ(log(C_i/O_i) - μ_c)²
    - σ²_rs = Роджерс-Сатчелл = (1/n) Σ[log(H_i/C_i)·log(H_i/O_i) + log(L_i/C_i)·log(L_i/O_i)]
    - k = 0.34 (эмпирически оптимальный вес)

    Fallback к close-to-close:
    Если OHLC данные недоступны или недостаточны, используется стандартная
    close-to-close волатильность. Это рекомендованный подход согласно
    финансовым исследованиям (Quant StackExchange).

    Args:
        ohlc_bars: список словарей с ключами 'open', 'high', 'low', 'close'
        n: размер окна
        close_prices: опциональный список цен закрытия для fallback метода

    Returns:
        Волатильность Yang-Zhang или close-to-close если OHLC недоступны,
        или None если данных недостаточно
    """
    # Пытаемся рассчитать Yang-Zhang если есть достаточно OHLC данных
    if ohlc_bars and len(ohlc_bars) >= n and n >= 2:
        yz_result = _try_calculate_yang_zhang(ohlc_bars, n)
        if yz_result is not None:
            return yz_result

    # Fallback к close-to-close volatility
    if close_prices is not None:
        return calculate_close_to_close_volatility(close_prices, n)

    return None


def _try_calculate_yang_zhang(ohlc_bars: List[Dict[str, float]], n: int) -> Optional[float]:
    """
    Внутренняя функция для попытки расчета Yang-Zhang волатильности.
    Возвращает None если данные недостаточны или некорректны.
    """
    # Берем последние n баров
    bars = list(ohlc_bars)[-n:]

    try:
        # k - эмпирически оптимальный вес
        k = 0.34

        # Расчет ночной волатильности σ²_o
        overnight_returns = []
        for i in range(1, len(bars)):
            prev_close = bars[i - 1].get("close", 0.0)
            curr_open = bars[i].get("open", 0.0)
            if prev_close > 0 and curr_open > 0:
                overnight_returns.append(math.log(curr_open / prev_close))

        if len(overnight_returns) < 2:
            return None

        mean_overnight = sum(overnight_returns) / len(overnight_returns)
        sigma_o_sq = sum((r - mean_overnight) ** 2 for r in overnight_returns) / (len(overnight_returns) - 1)

        # Расчет open-close волатильности σ²_c
        oc_returns = []
        for bar in bars:
            open_price = bar.get("open", 0.0)
            close_price = bar.get("close", 0.0)
            if open_price > 0 and close_price > 0:
                oc_returns.append(math.log(close_price / open_price))

        if len(oc_returns) < 2:
            return None

        mean_oc = sum(oc_returns) / len(oc_returns)
        sigma_c_sq = sum((r - mean_oc) ** 2 for r in oc_returns) / (len(oc_returns) - 1)

        # Расчет Rogers-Satchell волатильности σ²_rs
        rs_sum = 0.0
        rs_count = 0
        for bar in bars:
            high = bar.get("high", 0.0)
            low = bar.get("low", 0.0)
            open_price = bar.get("open", 0.0)
            close_price = bar.get("close", 0.0)

            if high > 0 and low > 0 and open_price > 0 and close_price > 0:
                # log(H/C) * log(H/O) + log(L/C) * log(L/O)
                term1 = math.log(high / close_price) * math.log(high / open_price)
                term2 = math.log(low / close_price) * math.log(low / open_price)
                rs_sum += term1 + term2
                rs_count += 1

        if rs_count == 0:
            return None

        # FIX (2025-11-26): Rogers-Satchell uses n, NOT (n-1)
        # ═══════════════════════════════════════════════════════════════════════════
        # The original Yang-Zhang (2000) formula specifies:
        #   σ²_rs = (1/n) Σ[log(H/C)·log(H/O) + log(L/C)·log(L/O)]
        #
        # This is DIFFERENT from σ_o and σ_c which use (n-1) because:
        # 1. σ_o and σ_c are CENTERED estimators (subtract mean) → Bessel's correction
        # 2. σ_rs is an UNCENTERED sum → uses n, not (n-1)
        #
        # The RS estimator is NOT a sample variance — it's an average of products.
        # Bessel's correction applies ONLY to centered sample variance estimates.
        #
        # Previous "CRITICAL FIX #2" was incorrect — it inflated RS by n/(n-1):
        #   +11.1% for n=10
        #   +2.0% for n=50
        #
        # Reference: Yang, D. & Zhang, Q. (2000) "Drift-Independent Volatility
        #            Estimation Based on High, Low, Open, and Close Prices"
        # Reference: Rogers, L.C.G. & Satchell, S.E. (1991) "Estimating Variance
        #            from High, Low and Closing Prices"
        # ═══════════════════════════════════════════════════════════════════════════
        sigma_rs_sq = rs_sum / rs_count

        # Комбинированная Yang-Zhang волатильность
        sigma_yz_sq = sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq

        # Возвращаем стандартное отклонение (квадратный корень из дисперсии)
        if sigma_yz_sq < 0:
            return None

        return math.sqrt(sigma_yz_sq)

    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None


def calculate_parkinson_volatility(ohlc_bars: List[Dict[str, float]], n: int) -> Optional[float]:
    """
    Рассчитывает волатильность диапазона Паркинсона (Parkinson Range Volatility) для последних n баров.

    Академическая формула (Parkinson, 1980):
    σ_P = sqrt[(1/(4n·ln(2))) · Σ(ln(H_i/L_i))²]

    ДОКУМЕНТАЦИЯ (MEDIUM #2): Intentional deviation from academic formula
    ===========================================================================
    Текущая реализация использует valid_bars (effective sample size) вместо n
    в знаменателе. Это отклонение от оригинальной формулы, но статистически
    корректно для unbiased estimation при наличии missing data.

    Обоснование:
    - Академическая: знаменатель = 4·n·ln(2) (assumes complete data)
    - Текущая: знаменатель = 4·valid_bars·ln(2) (adapts to missing data)
    - Статистика: unbiased mean = sum / effective_sample_size (Casella & Berger, 2002)
    - Результат: при пропусках данных volatility adjusted upward (корректно)

    Эта реализация предпочтительна для production систем, где missing data
    неизбежны (сетевые сбои, биржевые проблемы). Автоматическая адаптация
    к missing data без потери statistical validity.

    Оценщик Паркинсона в 7,4 раза более эффективен, чем close-to-close estimator.

    Args:
        ohlc_bars: список словарей с ключами 'high' и 'low'
        n: размер окна (минимум 2, minimum 80% data completeness required)

    Returns:
        Волатильность Паркинсона или None если недостаточно данных

    References:
        - Parkinson, M. (1980). "The Extreme Value Method..."
        - Casella & Berger (2002). "Statistical Inference" (unbiased estimation)
    """
    if not ohlc_bars or len(ohlc_bars) < n or n < 2:
        return None

    # Берем последние n баров
    bars = list(ohlc_bars)[-n:]

    try:
        sum_sq = 0.0
        valid_bars = 0

        for bar in bars:
            high = bar.get("high", 0.0)
            low = bar.get("low", 0.0)

            # Проверяем валидность данных
            if high > 0 and low > 0 and high >= low:
                # log(H_i/L_i)²
                log_hl = math.log(high / low)
                sum_sq += log_hl ** 2
                valid_bars += 1

        # Требуем минимум 2 валидных бара и минимум 80% от запрошенного окна
        # Это обеспечивает статистическую надежность оценки
        min_required = max(2, int(0.8 * n))
        if valid_bars < min_required:
            return None

        # σ² = (1/(4n·ln(2))) · Σ(ln(H_i/L_i))²
        # Используем valid_bars (количество реально использованных данных) для корректной оценки
        # Это стандартная статистическая практика: среднее = сумма / количество_слагаемых
        parkinson_var = sum_sq / (4 * valid_bars * math.log(2))

        # Возвращаем стандартное отклонение
        return math.sqrt(parkinson_var)

    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None


def _calculate_ewma_volatility(prices: List[float], lambda_decay: float = 0.94) -> Optional[float]:
    """
    Рассчитывает волатильность с использованием EWMA (Exponentially Weighted Moving Average).

    EWMA - это robust альтернатива GARCH, которая:
    - Не требует большого объема данных (минимум 2 точки)
    - Не требует сложной оптимизации
    - Является частным случаем GARCH(1,1)
    - Дает хорошие прогнозы в условиях изменчивой волатильности

    Формула: σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}

    Args:
        prices: список цен (минимум 2)
        lambda_decay: decay factor (обычно 0.94 для дневных данных, 0.97 для месячных)
                     RiskMetrics рекомендует 0.94

    Returns:
        Прогноз волатильности или None если недостаточно данных
    """
    if not prices or len(prices) < 2:
        return None

    try:
        # Берем все доступные цены для EWMA
        price_array = np.array(prices, dtype=float)

        # Проверяем валидность данных
        if np.any(price_array <= 0) or np.any(~np.isfinite(price_array)):
            return None

        # Вычисляем логарифмические доходности
        log_returns = np.log(price_array[1:] / price_array[:-1])

        # Проверяем валидность доходностей
        if not np.all(np.isfinite(log_returns)):
            return None

        # CRITICAL FIX #4: Robust EWMA initialization to prevent cold start bias
        # Old approach: variance = first_return² (unreliable, 2-5x bias)
        # New approach: Use median of squared returns (robust to outliers)
        # Reference: RiskMetrics Technical Document (1996) - EWMA initialization
        if len(log_returns) >= 10:
            # Sufficient data: use sample variance
            variance = np.var(log_returns, ddof=1)
        elif len(log_returns) >= 3:
            # Limited data: use median of squared returns (robust estimator)
            # Median is less sensitive to first-return spikes than mean
            variance = float(np.median(log_returns ** 2))
        else:
            # Very limited data (<3 returns): fall back to mean of squared returns
            # This is more stable than using only first return
            variance = float(np.mean(log_returns ** 2))

        # Рекурсивно вычисляем EWMA
        for ret in log_returns:
            variance = lambda_decay * variance + (1 - lambda_decay) * (ret ** 2)

        # Возвращаем стандартное отклонение (волатильность)
        volatility = np.sqrt(variance)

        if not np.isfinite(volatility) or volatility <= 0:
            return None

        return float(volatility)

    except (ValueError, Exception):
        return None


def _calculate_historical_volatility(prices: List[float], min_periods: int = 2) -> Optional[float]:
    """
    Рассчитывает простую историческую волатильность (standard deviation).

    Это самый базовый и robust метод оценки волатильности:
    - Требует минимум данных (2+ точки)
    - Всегда сходится
    - Не требует оптимизации

    Args:
        prices: список цен
        min_periods: минимальное количество точек (по умолчанию 2)

    Returns:
        Историческая волатильность или None если недостаточно данных
    """
    if not prices or len(prices) < min_periods:
        return None

    try:
        # Берем все доступные цены
        price_array = np.array(prices, dtype=float)

        # Проверяем валидность данных
        if np.any(price_array <= 0) or np.any(~np.isfinite(price_array)):
            return None

        # Вычисляем логарифмические доходности
        log_returns = np.log(price_array[1:] / price_array[:-1])

        # Проверяем валидность доходностей
        if not np.all(np.isfinite(log_returns)):
            return None

        # Вычисляем стандартное отклонение
        # CRITICAL FIX: для 1 доходности (2 цены) используем ddof=0 вместо ddof=1
        # ddof=1 при len=1 дает деление на 0 (nan)
        if len(log_returns) == 1:
            # Для единственной доходности: volatility = abs(return)
            volatility = abs(log_returns[0])
        else:
            # Для >= 2 доходностей: используем стандартное отклонение с ddof=1
            volatility = np.std(log_returns, ddof=1)

        if not np.isfinite(volatility) or volatility < 0:
            return None

        return float(volatility)

    except (ValueError, Exception):
        return None


def calculate_garch_volatility(prices: List[float], n: int) -> Optional[float]:
    """
    Рассчитывает условную волатильность с robust fallback стратегией.

    Стратегия (cascading fallback):
    1. GARCH(1,1) - если достаточно данных (n >= 50) и модель сходится
    2. EWMA - если GARCH не сходится или данных 2-49 баров
    3. Historical Volatility - финальный fallback для минимальных данных (2+ бара)
    4. Minimum floor - для flat markets (волатильность < 1e-10)

    Модель GARCH(1,1):
    r_t = μ + ε_t
    ε_t = σ_t * z_t, где z_t ~ N(0,1)
    σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}

    EWMA (если GARCH не подходит):
    σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}, где λ=0.94

    Args:
        prices: список цен
        n: размер скользящего окна для GARCH (рекомендуется 50-500+ наблюдений)

    Returns:
        Прогноз условной волатильности или None только если < 2 точек данных

    References:
        - RiskMetrics Technical Document (1996) - EWMA параметры
        - Brownlees & Gallo (2010) - Comparison of volatility measures
        - Hansen & Lunde (2005) - Forecast comparison
    """
    MIN_GARCH_OBSERVATIONS = 50
    MIN_EWMA_OBSERVATIONS = 2
    VOLATILITY_FLOOR = 1e-10  # минимальная волатильность для flat markets

    # Валидация входных данных
    if not prices or len(prices) < MIN_EWMA_OBSERVATIONS:
        return None

    available_data = len(prices)

    # Попытка 1: GARCH(1,1) - только если достаточно данных
    if available_data >= MIN_GARCH_OBSERVATIONS and n >= MIN_GARCH_OBSERVATIONS:
        try:
            # Берем последние n цен
            window_size = min(n, available_data)
            price_window = np.array(prices[-window_size:], dtype=float)

            # Проверяем валидность данных
            if np.any(price_window <= 0) or np.any(~np.isfinite(price_window)):
                # Переходим к EWMA
                pass
            else:
                # Вычисляем логарифмические доходности
                log_returns = np.log(price_window[1:] / price_window[:-1])

                # Проверяем, что есть вариация в данных (используем ddof=1 для выборки)
                returns_std = np.std(log_returns, ddof=1)
                if returns_std >= VOLATILITY_FLOOR:
                    # Конвертируем в процентные доходности для лучшей численной стабильности
                    returns_pct = log_returns * 100

                    # Подавляем предупреждения от arch (они могут быть шумными)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        # Создаем и подгоняем модель GARCH(1,1)
                        model = arch_model(
                            returns_pct,
                            vol='Garch',
                            p=1,  # GARCH порядок
                            q=1,  # ARCH порядок
                            dist='normal',  # нормальное распределение
                            rescale=False
                        )

                        # Подгоняем модель с максимальной итерацией
                        result = model.fit(
                            update_freq=0,
                            disp='off',
                            show_warning=False,
                            options={'maxiter': 1000}
                        )

                        # Получаем прогноз условной волатильности на 1 шаг вперед
                        forecast = result.forecast(horizon=1, reindex=False)
                        forecast_variance = forecast.variance.values[-1, 0]

                        if np.isfinite(forecast_variance) and forecast_variance > 0:
                            # Конвертируем обратно из процентного масштаба
                            forecast_volatility = np.sqrt(forecast_variance) / 100

                            if np.isfinite(forecast_volatility):
                                # GARCH успешно!
                                return float(forecast_volatility)

        except (ValueError, RuntimeError, np.linalg.LinAlgError, Exception):
            # GARCH не сошелся, переходим к EWMA
            pass

    # Попытка 2: EWMA - robust fallback для недостаточных данных или несходимости GARCH
    ewma_result = _calculate_ewma_volatility(prices, lambda_decay=0.94)
    if ewma_result is not None:
        # Применяем minimum floor для flat markets
        return float(max(ewma_result, VOLATILITY_FLOOR))

    # Попытка 3: Historical Volatility - финальный fallback
    hist_vol = _calculate_historical_volatility(prices, min_periods=MIN_EWMA_OBSERVATIONS)
    if hist_vol is not None:
        # Применяем minimum floor для flat markets
        return float(max(hist_vol, VOLATILITY_FLOOR))

    # Если все методы не сработали (очень редкий случай)
    return None


def _convert_minutes_to_bars_with_validation(
    window_minutes: int,
    bar_duration_minutes: int,
    window_name: str = "window"
) -> int:
    """
    Конвертирует окно из минут в бары с валидацией кратности.

    Args:
        window_minutes: Размер окна в минутах
        bar_duration_minutes: Длительность одного бара в минутах
        window_name: Имя окна для сообщений об ошибках

    Returns:
        Размер окна в барах (минимум 1)

    Warnings:
        UserWarning если окно не кратно bar_duration_minutes
    """
    bars = max(1, window_minutes // bar_duration_minutes)
    actual_minutes = bars * bar_duration_minutes

    # Предупреждение если окно не кратно длительности бара
    if actual_minutes != window_minutes:
        discrepancy_minutes = window_minutes - actual_minutes
        discrepancy_pct = 100.0 * abs(discrepancy_minutes) / window_minutes
        warnings.warn(
            f"{window_name} {window_minutes} minutes is not divisible by "
            f"bar_duration_minutes={bar_duration_minutes}. "
            f"Actual window will be {actual_minutes} minutes ({bars} bars), "
            f"discrepancy: {discrepancy_minutes} minutes ({discrepancy_pct:.2f}%). "
            f"Consider using windows that are multiples of {bar_duration_minutes} "
            f"for exact alignment.",
            UserWarning,
            stacklevel=3
        )

    return bars


@dataclass
class FeatureSpec:
    """
    Единая спецификация фич:
      - lookbacks_prices: окна для SMA и лог-ретёрнов (в минутах для 1m входа)
      - rsi_period: период RSI по Вайльдеру (EMA-уподоблённое сглаживание)
      - yang_zhang_windows: окна для волатильности Yang-Zhang (в минутах)
      - parkinson_windows: окна для волатильности Паркинсона (в минутах)
      - garch_windows: окна для условной волатильности GARCH(1,1) (в минутах, рекомендуется 500+)
      - taker_buy_ratio_windows: окна для скользящего среднего taker_buy_ratio (в минутах)
      - taker_buy_ratio_momentum: окна для моментума taker_buy_ratio (в минутах)
      - cvd_windows: окна для кумулятивной дельты объема (в минутах)
      - bar_duration_minutes: длительность одного бара в минутах (1 для 1m, 240 для 4h)

    После __post_init__:
      - lookbacks_prices и другие окна конвертируются в БАРЫ для корректной работы
      - Исходные значения в минутах сохраняются в _*_minutes полях для именования признаков
    """

    lookbacks_prices: List[int] = field(default_factory=list)
    sma_windows: Optional[List[int]] = None  # legacy alias for lookbacks_prices (minutes)
    ret_windows: Optional[List[int]] = None  # legacy alias for return windows (minutes)
    bid_ask_spread_windows: Optional[List[int]] = None  # compatibility placeholder
    volume_windows: Optional[List[int]] = None  # compatibility placeholder
    vwap_windows: Optional[List[int]] = None  # compatibility placeholder
    intrabar_range_windows: Optional[List[int]] = None  # compatibility placeholder
    rsi_period: int = 14
    yang_zhang_windows: Optional[List[int]] = None
    parkinson_windows: Optional[List[int]] = None
    garch_windows: Optional[List[int]] = None
    taker_buy_ratio_windows: Optional[List[int]] = None
    taker_buy_ratio_momentum: Optional[List[int]] = None
    cvd_windows: Optional[List[int]] = None
    bar_duration_minutes: int = 240  # 4h timeframe (changed from 1 for 1m)

    # Исходные значения в минутах (для именования признаков после конвертации в бары)
    _lookbacks_prices_minutes: Optional[List[int]] = None
    _yang_zhang_windows_minutes: Optional[List[int]] = None
    _parkinson_windows_minutes: Optional[List[int]] = None
    _garch_windows_minutes: Optional[List[int]] = None
    _taker_buy_ratio_windows_minutes: Optional[List[int]] = None
    _taker_buy_ratio_momentum_minutes: Optional[List[int]] = None
    _cvd_windows_minutes: Optional[List[int]] = None

    
    
    def __post_init__(self) -> None:
        legacy_lookbacks: list[int] = []
        if self.sma_windows is not None:
            legacy_lookbacks.extend(
                [int(abs(x)) for x in self.sma_windows if int(abs(x)) > 0]
            )
        if self.ret_windows is not None:
            legacy_lookbacks.extend(
                [int(abs(x)) for x in self.ret_windows if int(abs(x)) > 0]
            )

        user_supplied_any = (
            bool(self.lookbacks_prices)
            or self.sma_windows is not None
            or self.ret_windows is not None
        )

        if not isinstance(self.lookbacks_prices, list):
            self.lookbacks_prices = []

        if not self.lookbacks_prices and legacy_lookbacks:
            # Legacy API: map sma_windows/ret_windows -> lookbacks_prices (minutes)
            self.lookbacks_prices = legacy_lookbacks

        if not self.lookbacks_prices and not user_supplied_any:
            # Default 4h windows for SMA/returns when nothing provided (minutes)
            self.lookbacks_prices = [240, 720, 1200, 1440, 5040, 10080, 12000]

        self.lookbacks_prices = [
            int(abs(x)) for x in self.lookbacks_prices if int(abs(x)) > 0
        ]

        # Preserve original minute windows for naming
        self._lookbacks_prices_minutes = list(self.lookbacks_prices)

        # Convert minute windows to bars with validation
        self.lookbacks_prices = [
            _convert_minutes_to_bars_with_validation(
                w, self.bar_duration_minutes, "lookbacks_prices"
            )
            for w in self.lookbacks_prices
        ]

        self.rsi_period = int(self.rsi_period)

        # Yang-Zhang для 4h интервала: 48ч, 7д, 30д в минутах
        # 48h = 12 баров = 2880 минут
        # 7d = 42 бара = 10080 минут
        # 30d = 180 баров = 43200 минут
        if self.yang_zhang_windows is None:
            self.yang_zhang_windows = [48 * 60, 7 * 24 * 60, 30 * 24 * 60]  # 2880, 10080, 43200 минут
        elif isinstance(self.yang_zhang_windows, list):
            self.yang_zhang_windows = [
                int(abs(x)) for x in self.yang_zhang_windows if int(abs(x)) > 0
            ]
        else:
            self.yang_zhang_windows = []

        # CRITICAL FIX #1: Сохраняем исходные значения в минутах для именования признаков
        self._yang_zhang_windows_minutes = list(self.yang_zhang_windows)

        # CRITICAL FIX #1 (ENHANCED): Конвертируем окна из минут в бары с валидацией
        self.yang_zhang_windows = [
            _convert_minutes_to_bars_with_validation(
                w, self.bar_duration_minutes, "yang_zhang_windows"
            )
            for w in self.yang_zhang_windows
        ]

        # Инициализация окон Parkinson для 4h интервала: 48ч, 7д в минутах
        # 48h = 12 баров = 2880 минут
        # 7d = 42 бара = 10080 минут
        if self.parkinson_windows is None:
            self.parkinson_windows = [48 * 60, 7 * 24 * 60]  # 2880, 10080 минут
        elif isinstance(self.parkinson_windows, list):
            self.parkinson_windows = [
                int(abs(x)) for x in self.parkinson_windows if int(abs(x)) > 0
            ]
        else:
            self.parkinson_windows = []

        # CRITICAL FIX #1: Сохраняем исходные значения в минутах для именования признаков
        self._parkinson_windows_minutes = list(self.parkinson_windows)

        # CRITICAL FIX #1 (ENHANCED): Конвертируем окна из минут в бары с валидацией
        self.parkinson_windows = [
            _convert_minutes_to_bars_with_validation(
                w, self.bar_duration_minutes, "parkinson_windows"
            )
            for w in self.parkinson_windows
        ]

        # Инициализация окон Taker Buy Ratio скользящего среднего для 4h интервала: 8ч, 16ч, 24ч в минутах
        # 8h = 2 бара = 480 минут
        # 16h = 4 бара = 960 минут
        # 24h = 6 баров = 1440 минут
        if self.taker_buy_ratio_windows is None:
            self.taker_buy_ratio_windows = [8 * 60, 16 * 60, 24 * 60]  # 480, 960, 1440 минут
        elif isinstance(self.taker_buy_ratio_windows, list):
            self.taker_buy_ratio_windows = [
                int(abs(x)) for x in self.taker_buy_ratio_windows if int(abs(x)) > 0
            ]
        else:
            self.taker_buy_ratio_windows = []

        # CRITICAL FIX #1: Сохраняем исходные значения в минутах для именования признаков
        self._taker_buy_ratio_windows_minutes = list(self.taker_buy_ratio_windows)

        # CRITICAL FIX #1 (ENHANCED): Конвертируем окна из минут в бары с валидацией
        self.taker_buy_ratio_windows = [
            _convert_minutes_to_bars_with_validation(
                w, self.bar_duration_minutes, "taker_buy_ratio_windows"
            )
            for w in self.taker_buy_ratio_windows
        ]

        # Инициализация окон моментума Taker Buy Ratio для 4h интервала: 4ч, 8ч, 12ч, 24ч в минутах
        # 4h = 1 бар = 240 минут
        # 8h = 2 бара = 480 минут
        # 12h = 3 бара = 720 минут
        # 24h = 6 баров = 1440 минут (для долгосрочного моментума)
        #
        # АРХИТЕКТУРНОЕ РЕШЕНИЕ: Генерируется 4 окна, используется 3 в observation
        # - Генерируем ВСЕ 4 окна для гибкости (бэктесты, анализ, будущие эксперименты)
        # - mediator.py использует только 3 первых в norm_cols[18,19,20] для компактности
        # - observation vector = 62 признаков (21 external из них) [updated in v62]
        # - Это стандартная практика (аналогично sklearn, XGBoost: fit на всех, predict на подмножестве)
        if self.taker_buy_ratio_momentum is None:
            self.taker_buy_ratio_momentum = [4 * 60, 8 * 60, 12 * 60, 24 * 60]  # 240, 480, 720, 1440 минут
        elif isinstance(self.taker_buy_ratio_momentum, list):
            self.taker_buy_ratio_momentum = [
                int(abs(x)) for x in self.taker_buy_ratio_momentum if int(abs(x)) > 0
            ]
        else:
            self.taker_buy_ratio_momentum = []

        # CRITICAL FIX #1: Сохраняем исходные значения в минутах для именования признаков
        self._taker_buy_ratio_momentum_minutes = list(self.taker_buy_ratio_momentum)

        # CRITICAL FIX #1 (ENHANCED): Конвертируем окна из минут в бары с валидацией
        self.taker_buy_ratio_momentum = [
            _convert_minutes_to_bars_with_validation(
                w, self.bar_duration_minutes, "taker_buy_ratio_momentum"
            )
            for w in self.taker_buy_ratio_momentum
        ]

        # Инициализация окон Cumulative Volume Delta: 24ч, 7д в минутах (без изменений)
        if self.cvd_windows is None:
            self.cvd_windows = [24 * 60, 7 * 24 * 60]  # 1440, 10080 минут
        elif isinstance(self.cvd_windows, list):
            self.cvd_windows = [
                int(abs(x)) for x in self.cvd_windows if int(abs(x)) > 0
            ]
        else:
            self.cvd_windows = []

        # CRITICAL FIX #1: Сохраняем исходные значения в минутах для именования признаков
        self._cvd_windows_minutes = list(self.cvd_windows)

        # CRITICAL FIX #1 (ENHANCED): Конвертируем окна из минут в бары с валидацией
        self.cvd_windows = [
            _convert_minutes_to_bars_with_validation(
                w, self.bar_duration_minutes, "cvd_windows"
            )
            for w in self.cvd_windows
        ]

        # Инициализация окон GARCH для 4h интервала: 200h, 14д, 30д в минутах
        # GARCH с fallback стратегией:
        # - GARCH(1,1): требует минимум 50 наблюдений (строка 379+)
        # - EWMA fallback: работает с 2+ баров (robust, не требует оптимизации)
        # - Historical vol: финальный fallback для 2+ баров
        # 50 баров = 12000 минут = 200h (оптимальное окно для полного GARCH на 4h)
        # 14d = 84 бара = 20160 минут
        # 30d = 180 баров = 43200 минут
        if self.garch_windows is None:
            self.garch_windows = [50 * 240, 14 * 24 * 60, 30 * 24 * 60]  # 12000, 20160, 43200 минут
        elif isinstance(self.garch_windows, list):
            self.garch_windows = [
                int(abs(x)) for x in self.garch_windows if int(abs(x)) > 0
            ]
        else:
            self.garch_windows = []

        # CRITICAL FIX #1: Сохраняем исходные значения в минутах для именования признаков
        self._garch_windows_minutes = list(self.garch_windows)

        # CRITICAL FIX #1 (ENHANCED): Конвертируем окна из минут в бары с валидацией
        self.garch_windows = [
            _convert_minutes_to_bars_with_validation(
                w, self.bar_duration_minutes, "garch_windows"
            )
            for w in self.garch_windows
        ]


class OnlineFeatureTransformer:
    """
    Онлайн-трансформер: состояние на символ, детерминистичное поведение.
    Полностью соответствует онлайновой логике (как раньше в FeaturePipe):
      - SMA и ретёрны из окна цен (1 точка в минуту)
      - RSI по Вайльдеру: скользящие avg_gain/avg_loss с периодом p
      - Yang-Zhang волатильность: комплексная OHLC-волатильность
      - Parkinson волатильность: волатильность диапазона High-Low

    ВАЖНО: Семантика вычисления индикаторов (NO LOOK-AHEAD BIAS):
    ======================================================================
    update() вызывается после ЗАКРЫТИЯ бара с данными закрытого бара.
    Все индикаторы вычисляются на основе данных, доступных ПОСЛЕ закрытия:

    - SMA_n = среднее последних n ЗАКРЫТЫХ баров (включая текущий)
      Пример: SMA_5 = (P_t + P_{t-1} + P_{t-2} + P_{t-3} + P_{t-4}) / 5
      где P_t - цена закрытия текущего (только что закрытого) бара

    - Returns = log(P_t / P_{t-n}), доходность от n баров назад к текущему

    - RSI обновляется с учетом изменения от предыдущего к текущему закрытию

    Защита от утечки данных:
    - Признаки доступны ПОСЛЕ ts_ms (времени закрытия бара)
    - Решения принимаются в момент decision_ts = ts_ms + decision_delay_ms
    - Target вычисляется от decision_ts (см. LeakGuard, LabelBuilder)
    - При правильной настройке decision_delay_ms > 0 нет look-ahead bias

    Lag индикаторов (природный запаздывающий характер):
    - SMA имеет lag ≈ window/2 (половина окна) - присуще индикатору
    - Для SMA_5 lag ≈ 2.5 бара (Murphy, 1999: "Technical Analysis Indicators")
    - Модель должна учитывать этот lag при интерпретации сигналов

    Ссылки:
    - Murphy, J.J. (1999). Technical Analysis of Financial Markets
    - de Prado, M.L. (2018). "Advances in Financial Machine Learning"
      Глава 7: The Dangers of Backtesting (look-ahead bias prevention)
    """

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec
        self._state: Dict[str, Dict[str, Any]] = {}

    def _ensure_state(self, symbol: str) -> Dict[str, Any]:
        st = self._state.get(symbol)
        if st is None:
            # Определяем максимальную длину окна для всех фич
            # CRITICAL FIX #2: Все окна уже конвертированы из минут в бары в FeatureSpec.__post_init__
            # Поэтому maxlen теперь корректно устанавливается в барах, а не в минутах
            all_windows = self.spec.lookbacks_prices + [self.spec.rsi_period + 1]
            if self.spec.yang_zhang_windows:
                all_windows.extend(self.spec.yang_zhang_windows)
            if self.spec.parkinson_windows:
                all_windows.extend(self.spec.parkinson_windows)
            if self.spec.garch_windows:
                all_windows.extend(self.spec.garch_windows)
            if self.spec.taker_buy_ratio_windows:
                all_windows.extend(self.spec.taker_buy_ratio_windows)
            if self.spec.taker_buy_ratio_momentum:
                all_windows.extend(self.spec.taker_buy_ratio_momentum)
            if self.spec.cvd_windows:
                all_windows.extend(self.spec.cvd_windows)
            maxlen = max(all_windows) if all_windows else 100

            # CRITICAL FIX: Momentum calculation requires window + 1 elements
            # (need both current value and value from 'window' bars ago)
            # Ensure maxlen is sufficient for all momentum windows
            if self.spec.taker_buy_ratio_momentum:
                max_momentum_window = max(self.spec.taker_buy_ratio_momentum)
                maxlen = max(maxlen, max_momentum_window + 1)
            # Add safety buffer for lagging indicators (like Stochastic %D which requires w + 2)
            # Awesome Oscillator requires at least 34 bars.
            maxlen = max(maxlen, 34) + 20

            st = {
                "prices": deque(maxlen=maxlen),
                "avg_gain": None,
                "avg_loss": None,
                "last_close": None,
                # RSI initialization fix: collect first rsi_period gains/losses for SMA init
                # Reference: Wilder (1978), "New Concepts in Technical Trading Systems"
                # Previous bug: Initialized with SINGLE value -> bias for first ~30 bars
                # Now: Collect first rsi_period values, then compute SMA (like MarketSimulator.cpp)
                "gain_history": deque(maxlen=self.spec.rsi_period),
                "loss_history": deque(maxlen=self.spec.rsi_period),
                "rsi_initialized": False,
                # Для Yang-Zhang волатильности нужны OHLC
                "ohlc_bars": deque(maxlen=maxlen),
                # Для Taker Buy Ratio нужны значения ratio
                "taker_buy_ratios": deque(maxlen=maxlen),
                # Для Cumulative Volume Delta нужны дельты объема
                "volume_deltas": deque(maxlen=maxlen),
                # Для Price Volume Trend
                "pvt": 0.0,
            }
            self._state[symbol] = st
        return st

    def update(
        self,
        *,
        symbol: str,
        ts_ms: int,
        close: float,
        open_price: Optional[float] = None,
        high: Optional[float] = None,
        low: Optional[float] = None,
        volume: Optional[float] = None,
        taker_buy_base: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Обновляет состояние трансформера новым баром и возвращает признаки.

        ВАЖНО: Семантика вызова (NO LOOK-AHEAD BIAS):
        ==============================================
        Эта функция вызывается ПОСЛЕ закрытия бара с данными ЗАКРЫТОГО бара.
        Возвращаемые признаки доступны ПОСЛЕ ts_ms и используются для решений
        в момент decision_ts = ts_ms + decision_delay_ms (см. LeakGuard).

        Временная последовательность:
        1. Бар закрывается в момент ts_ms
        2. update() вызывается с close, open, high, low закрытого бара
        3. Признаки вычисляются на основе истории + текущий закрытый бар
        4. Признаки доступны для решений в момент decision_ts >= ts_ms

        Формулы индикаторов (на основе закрытых баров):
        - SMA_n = (close_t + close_{t-1} + ... + close_{t-n+1}) / n
        - Returns = log(close_t / close_{t-n})
        - RSI использует изменение от close_{t-1} к close_t

        Args:
            symbol: символ торгового инструмента
            ts_ms: временная метка ЗАКРЫТИЯ бара в миллисекундах
            close: цена закрытия ЗАКРЫТОГО бара
            open_price: цена открытия (опционально для Yang-Zhang)
            high: максимальная цена (опционально для Yang-Zhang)
            low: минимальная цена (опционально для Yang-Zhang)
            volume: объем торгов (опционально для Taker Buy Ratio)
            taker_buy_base: объем покупок taker (опционально для Taker Buy Ratio)

        Returns:
            Dict с признаками:
            - ts_ms, symbol, ref_price (базовая информация)
            - sma_*, ret_* (цена и доходность)
            - rsi (momentum)
            - yang_zhang_*, parkinson_*, garch_* (волатильность)
            - taker_buy_ratio*, cvd_* (объемные индикаторы)

        Note:
            Lag индикаторов ~window/2 присущ SMA (Murphy, 1999).
            Модель должна учитывать этот lag при интерпретации.
        """
        sym = str(symbol).upper()
        price = float(close)
        st = self._ensure_state(sym)

        last = st["last_close"]
        if last is not None:
            delta = price - float(last)
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)

            p = self.spec.rsi_period

            # RSI initialization fix: use SMA of first rsi_period values
            # Reference: Wilder (1978), "New Concepts in Technical Trading Systems"
            if not st["rsi_initialized"]:
                # Accumulate gains/losses until we have rsi_period values
                st["gain_history"].append(gain)
                st["loss_history"].append(loss)

                if len(st["gain_history"]) == p:
                    # Initialize with SMA (not single value!)
                    st["avg_gain"] = sum(st["gain_history"]) / p
                    st["avg_loss"] = sum(st["loss_history"]) / p
                    st["rsi_initialized"] = True
            else:
                # Normal Wilder smoothing: (prev * (p-1) + new) / p
                st["avg_gain"] = ((float(st["avg_gain"]) * (p - 1)) + gain) / p
                st["avg_loss"] = ((float(st["avg_loss"]) * (p - 1)) + loss) / p
        st["last_close"] = price

        st["prices"].append(price)

        # Сохраняем OHLC данные для Yang-Zhang и других индикаторов
        if open_price is not None and high is not None and low is not None:
            ohlc_bar = {
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume) if volume is not None else 0.0,
            }
            st["ohlc_bars"].append(ohlc_bar)

        # Вычисляем и сохраняем Taker Buy Ratio
        if volume is not None and taker_buy_base is not None and volume > 0:
            # Добавляем clamping на случай аномальных данных (taker_buy_base > volume)
            # Нормальный диапазон: [0.0, 1.0]
            raw_ratio = float(taker_buy_base) / float(volume)
            taker_buy_ratio = min(1.0, max(0.0, raw_ratio))

            # CRITICAL FIX: Data quality check - warn on anomalous values
            if raw_ratio > 1.0:
                warnings.warn(
                    f"Data quality issue: taker_buy_base ({taker_buy_base}) > volume ({volume}) "
                    f"for {sym} at {ts_ms}. Ratio clamped from {raw_ratio:.4f} to 1.0",
                    UserWarning,
                    stacklevel=2
                )
            elif raw_ratio < 0.0:
                warnings.warn(
                    f"Data quality issue: negative taker_buy_base ({taker_buy_base}) "
                    f"for {sym} at {ts_ms}. Ratio clamped from {raw_ratio:.4f} to 0.0",
                    UserWarning,
                    stacklevel=2
                )

            st["taker_buy_ratios"].append(taker_buy_ratio)

        # Вычисляем и сохраняем Volume Delta для CVD
        # CVD формула: buy_volume - sell_volume
        # где buy_volume = taker_buy_base, sell_volume = volume - taker_buy_base
        if volume is not None and taker_buy_base is not None:
            buy_volume = float(taker_buy_base)
            sell_volume = float(volume) - buy_volume
            volume_delta = buy_volume - sell_volume
            st["volume_deltas"].append(volume_delta)

        feats: Dict[str, Any] = {
            "ts_ms": int(ts_ms),
            "symbol": sym,
            "ref_price": float(price),
        }

        seq = list(st["prices"])
        # CRITICAL FIX #1: Используем минуты для имен SMA (sma_240, sma_1440, sma_12000),
        # форматированные имена для returns (ret_4h, ret_24h, ret_200h)
        for i, lb in enumerate(self.spec.lookbacks_prices):
            lb_minutes = self.spec._lookbacks_prices_minutes[i]

            # Для SMA нужно минимум lb элементов
            if len(seq) >= lb:
                window = seq[-lb:]
                sma = sum(window) / float(lb)
                # Используем значение в минутах для имени SMA (сырые минуты: 240, 1440, 12000)
                feats[f"sma_{lb_minutes}"] = float(sma)

            # CRITICAL FIX: Для returns нужно минимум lb+1 элементов
            # (цена lb баров назад + текущая цена)
            # При lb=1: нужно сравнить текущую цену с ценой 1 бар назад
            if len(seq) > lb:
                # Берем цену lb баров назад (-(lb+1) элемент)
                old_price = float(seq[-(lb + 1)])
                # Используем форматированное значение для имени returns (4h, 24h, 200h)
                ret_name = f"ret_{_format_window_name(lb_minutes)}"

                # CRITICAL FIX #4: Validate both prices for safe log-return computation
                # Log returns require BOTH prices > 0 (division by old_price and log domain)
                # Invalid: old_price ≤ 0 or price ≤ 0 → would produce -inf/NaN
                # Safe fallback: NaN instead of 0.0 (more explicit about missing data)
                if old_price > 0 and price > 0:
                    feats[ret_name] = float(math.log(price / old_price))
                else:
                    # NaN is safer than 0.0 for invalid log returns
                    # Training loops should handle NaN (drop or mask)
                    feats[ret_name] = float("nan")

        # CRITICAL FIX: Handle edge cases for RSI calculation (Wilder's formula)
        if st["avg_gain"] is not None and st["avg_loss"] is not None:
            avg_gain = float(st["avg_gain"])
            avg_loss = float(st["avg_loss"])

            if avg_loss == 0.0 and avg_gain > 0.0:
                # Pure uptrend: RS = infinity → RSI = 100
                feats["rsi"] = float(100.0)
            elif avg_gain == 0.0 and avg_loss > 0.0:
                # Pure downtrend: RS = 0 → RSI = 0
                feats["rsi"] = float(0.0)
            elif avg_gain == 0.0 and avg_loss == 0.0:
                # No price movement: neutral RSI
                feats["rsi"] = float(50.0)
            else:
                # Normal case: both avg_gain and avg_loss > 0
                rs = avg_gain / avg_loss
                feats["rsi"] = float(100.0 - (100.0 / (1.0 + rs)))
        else:
            feats["rsi"] = float("nan")

        # ---------------------------------------------------------
        # NEW HIGH-QUALITY INDICATORS CALCULATIONS
        # ---------------------------------------------------------
        
        # 1. Bollinger Bands & Volatility Indicators
        for i, lb in enumerate(self.spec.lookbacks_prices):
            lb_minutes = self.spec._lookbacks_prices_minutes[i]
            if len(seq) >= lb:
                window = seq[-lb:]
                sma = sum(window) / float(lb)
                variance = sum((x - sma) ** 2 for x in window) / float(lb)
                std = math.sqrt(variance)
                
                feats[f"bollinger_upper_{lb_minutes}"] = float(sma + 2.0 * std)
                feats[f"bollinger_lower_{lb_minutes}"] = float(sma - 2.0 * std)
                feats[f"bollinger_bandwidth_{lb_minutes}"] = float((4.0 * std) / sma) if sma > 1e-8 else 0.0
                feats[f"bollinger_percent_b_{lb_minutes}"] = float((price - (sma - 2.0 * std)) / (4.0 * std)) if std > 1e-8 else 0.5
            else:
                feats[f"bollinger_upper_{lb_minutes}"] = float("nan")
                feats[f"bollinger_lower_{lb_minutes}"] = float("nan")
                feats[f"bollinger_bandwidth_{lb_minutes}"] = float("nan")
                feats[f"bollinger_percent_b_{lb_minutes}"] = float("nan")

        # Helper function for EMA
        def _calc_ema(prices_list, n):
            if len(prices_list) < n:
                return None
            ema = sum(prices_list[:n]) / float(n)
            multiplier = 2.0 / (n + 1)
            for p in prices_list[n:]:
                ema = (p - ema) * multiplier + ema
            return ema

        # 2. MACD
        macd_list = []
        for length in range(26, len(seq) + 1):
            sub_seq = seq[:length]
            e12 = _calc_ema(sub_seq, 12)
            e26 = _calc_ema(sub_seq, 26)
            if e12 is not None and e26 is not None:
                macd_list.append(e12 - e26)
        
        if len(macd_list) >= 9:
            macd_sig = _calc_ema(macd_list, 9)
            if macd_sig is not None:
                feats["macd"] = float(macd_list[-1])
                feats["macd_signal"] = float(macd_sig)
                feats["macd_histogram"] = float(macd_list[-1] - macd_sig)
            else:
                feats["macd"] = float("nan")
                feats["macd_signal"] = float("nan")
                feats["macd_histogram"] = float("nan")
        else:
            feats["macd"] = float("nan")
            feats["macd_signal"] = float("nan")
            feats["macd_histogram"] = float("nan")

        # 3. OHLC-dependent indicators (Stochastic, ATR, CCI)
        ohlc_list = list(st["ohlc_bars"]) if st["ohlc_bars"] else []
        
        # Calculate True Range (TR) list for ATR
        tr_list = []
        for i in range(len(ohlc_list)):
            high_val = ohlc_list[i]["high"]
            low_val = ohlc_list[i]["low"]
            if i == 0:
                tr = high_val - low_val
            else:
                close_prev = ohlc_list[i-1]["close"]
                tr = max(high_val - low_val, abs(high_val - close_prev), abs(low_val - close_prev))
            tr_list.append(tr)

        # Calculate Typical Price (TP) list for CCI
        tp_list = [(b["high"] + b["low"] + b["close"]) / 3.0 for b in ohlc_list]

        for i, lb in enumerate(self.spec.lookbacks_prices):
            lb_minutes = self.spec._lookbacks_prices_minutes[i]
            
            # Stochastic Oscillator (%K, %D)
            k_history = []
            for shift in [2, 1, 0]:
                idx_end = len(ohlc_list) - shift
                if idx_end >= lb:
                    window_bars = ohlc_list[idx_end - lb : idx_end]
                    lw = min(b["low"] for b in window_bars)
                    hw = max(b["high"] for b in window_bars)
                    cl = window_bars[-1]["close"]
                    denom = hw - lw
                    k_val = (cl - lw) / denom * 100.0 if denom > 1e-8 else 50.0
                    k_history.append(k_val)

            if len(k_history) == 3:
                feats[f"stoch_k_{lb_minutes}"] = float(k_history[-1])
                feats[f"stoch_d_{lb_minutes}"] = float(sum(k_history) / 3.0)
            else:
                feats[f"stoch_k_{lb_minutes}"] = float("nan")
                feats[f"stoch_d_{lb_minutes}"] = float("nan")

            # ATR (Wilder-like simple average)
            if len(tr_list) >= lb:
                feats[f"atr_{lb_minutes}"] = float(sum(tr_list[-lb:]) / float(lb))
            else:
                feats[f"atr_{lb_minutes}"] = float("nan")

            # CCI
            if len(tp_list) >= lb:
                window_tp = tp_list[-lb:]
                sma_tp = sum(window_tp) / float(lb)
                mean_dev = sum(abs(x - sma_tp) for x in window_tp) / float(lb)
                cci = (tp_list[-1] - sma_tp) / (0.015 * mean_dev) if mean_dev > 1e-8 else 0.0
                feats[f"cci_{lb_minutes}"] = float(cci)
            else:
                feats[f"cci_{lb_minutes}"] = float("nan")

        # 4. CMO (Chande Momentum Oscillator)
        gains = []
        losses = []
        for idx in range(1, len(seq)):
            diff = seq[idx] - seq[idx-1]
            gains.append(max(diff, 0.0))
            losses.append(max(-diff, 0.0))

        for i, lb in enumerate(self.spec.lookbacks_prices):
            lb_minutes = self.spec._lookbacks_prices_minutes[i]
            if len(gains) >= lb:
                sum_g = sum(gains[-lb:])
                sum_l = sum(losses[-lb:])
                total = sum_g + sum_l
                cmo = 100.0 * (sum_g - sum_l) / total if total > 1e-8 else 0.0
                feats[f"cmo_{lb_minutes}"] = float(cmo)
            else:
                feats[f"cmo_{lb_minutes}"] = float("nan")

        # 5. ROC (Rate of Change)
        for i, lb in enumerate(self.spec.lookbacks_prices):
            lb_minutes = self.spec._lookbacks_prices_minutes[i]
            if len(seq) > lb:
                old_px = float(seq[-(lb + 1)])
                roc = (price - old_px) / old_px * 100.0 if old_px > 1e-8 else 0.0
                feats[f"roc_{lb_minutes}"] = float(roc)
            else:
                feats[f"roc_{lb_minutes}"] = float("nan")

        # 6. EMA (Exponential Moving Average)
        for i, lb in enumerate(self.spec.lookbacks_prices):
            lb_minutes = self.spec._lookbacks_prices_minutes[i]
            if len(seq) >= lb:
                ema_val = _calc_ema(seq, lb)
                if ema_val is not None:
                    feats[f"ema_{lb_minutes}"] = float(ema_val)
                else:
                    feats[f"ema_{lb_minutes}"] = float("nan")
            else:
                feats[f"ema_{lb_minutes}"] = float("nan")

        # 7. Williams %R
        if ohlc_list:
            for i, lb in enumerate(self.spec.lookbacks_prices):
                lb_minutes = self.spec._lookbacks_prices_minutes[i]
                if len(ohlc_list) >= lb:
                    window_bars = ohlc_list[-lb:]
                    highest_high = max(b["high"] for b in window_bars)
                    lowest_low = min(b["low"] for b in window_bars)
                    denom = highest_high - lowest_low
                    williams_r = (highest_high - price) / denom * -100.0 if denom > 1e-8 else -50.0
                    feats[f"williams_r_{lb_minutes}"] = float(williams_r)
                else:
                    feats[f"williams_r_{lb_minutes}"] = float("nan")
        else:
            for i, lb in enumerate(self.spec.lookbacks_prices):
                lb_minutes = self.spec._lookbacks_prices_minutes[i]
                feats[f"williams_r_{lb_minutes}"] = float("nan")

        # 8. Keltner Channels
        for i, lb in enumerate(self.spec.lookbacks_prices):
            lb_minutes = self.spec._lookbacks_prices_minutes[i]
            sma_val = feats.get(f"sma_{lb_minutes}")
            atr_val = feats.get(f"atr_{lb_minutes}")
            if sma_val is not None and atr_val is not None and not math.isnan(sma_val) and not math.isnan(atr_val):
                feats[f"keltner_middle_{lb_minutes}"] = float(sma_val)
                feats[f"keltner_upper_{lb_minutes}"] = float(sma_val + 2.0 * atr_val)
                feats[f"keltner_lower_{lb_minutes}"] = float(sma_val - 2.0 * atr_val)
            else:
                feats[f"keltner_middle_{lb_minutes}"] = float("nan")
                feats[f"keltner_upper_{lb_minutes}"] = float("nan")
                feats[f"keltner_lower_{lb_minutes}"] = float("nan")

        # 9. On-Balance Volume (OBV)
        if volume is not None:
            prev_obv = st.get("obv", 0.0)
            if last is not None:
                if price > last:
                    current_obv = prev_obv + float(volume)
                elif price < last:
                    current_obv = prev_obv - float(volume)
                else:
                    current_obv = prev_obv
            else:
                current_obv = float(volume)
            st["obv"] = current_obv
            feats["obv"] = float(current_obv)
        else:
            feats["obv"] = float("nan")

        # 10. Money Flow Index (MFI)
        if ohlc_list and all("volume" in b for b in ohlc_list):
            tps = [(b["high"] + b["low"] + b["close"]) / 3.0 for b in ohlc_list]
            rmfs = [tps[idx] * ohlc_list[idx].get("volume", 0.0) for idx in range(len(ohlc_list))]
            for i, lb in enumerate(self.spec.lookbacks_prices):
                lb_minutes = self.spec._lookbacks_prices_minutes[i]
                if len(ohlc_list) > lb:
                    pmf_sum = 0.0
                    nmf_sum = 0.0
                    for idx in range(len(ohlc_list) - lb, len(ohlc_list)):
                        tp_curr = tps[idx]
                        tp_prev = tps[idx-1]
                        rmf = rmfs[idx]
                        if tp_curr > tp_prev:
                            pmf_sum += rmf
                        elif tp_curr < tp_prev:
                            nmf_sum += rmf
                    if nmf_sum > 1e-8:
                        mfr = pmf_sum / nmf_sum
                        mfi = 100.0 - (100.0 / (1.0 + mfr))
                    else:
                        mfi = 100.0 if pmf_sum > 1e-8 else 50.0
                    feats[f"mfi_{lb_minutes}"] = float(mfi)
                else:
                    feats[f"mfi_{lb_minutes}"] = float("nan")
        else:
            for i, lb in enumerate(self.spec.lookbacks_prices):
                lb_minutes = self.spec._lookbacks_prices_minutes[i]
                feats[f"mfi_{lb_minutes}"] = float("nan")

        # 11. Linear Regression Slope
        def _calc_slope(y):
            n = len(y)
            if n < 2:
                return 0.0
            x_mean = (n + 1) / 2.0
            y_mean = sum(y) / float(n)
            num = sum((idx + 1 - x_mean) * (y[idx] - y_mean) for idx in range(n))
            den = n * (n**2 - 1) / 12.0
            return num / den if den > 1e-8 else 0.0

        for i, lb in enumerate(self.spec.lookbacks_prices):
            lb_minutes = self.spec._lookbacks_prices_minutes[i]
            if len(seq) >= lb:
                feats[f"slope_{lb_minutes}"] = float(_calc_slope(seq[-lb:]))
            else:
                feats[f"slope_{lb_minutes}"] = float("nan")

        # 12. Average Directional Index (ADX)
        if len(ohlc_list) >= 2:
            for i, lb in enumerate(self.spec.lookbacks_prices):
                lb_minutes = self.spec._lookbacks_prices_minutes[i]
                if len(ohlc_list) >= lb + 1:
                    tr_w = []
                    plus_dm_w = []
                    minus_dm_w = []
                    window_bars = ohlc_list[-(lb + 1):]
                    for idx in range(1, len(window_bars)):
                        h_curr = window_bars[idx]["high"]
                        l_curr = window_bars[idx]["low"]
                        c_prev = window_bars[idx-1]["close"]
                        h_prev = window_bars[idx-1]["high"]
                        l_prev = window_bars[idx-1]["low"]
                        
                        tr = max(h_curr - l_curr, abs(h_curr - c_prev), abs(l_curr - c_prev))
                        tr_w.append(tr)
                        
                        up_move = h_curr - h_prev
                        down_move = l_prev - l_curr
                        
                        if up_move > down_move and up_move > 0:
                            plus_dm = up_move
                        else:
                            plus_dm = 0.0
                        plus_dm_w.append(plus_dm)
                        
                        if down_move > up_move and down_move > 0:
                            minus_dm = down_move
                        else:
                            minus_dm = 0.0
                        minus_dm_w.append(minus_dm)
                    
                    def _smooth_wilder(vals, period):
                        return sum(vals) / float(period) if period > 0 else 0.0
                    
                    smooth_tr = _smooth_wilder(tr_w, lb)
                    smooth_plus_dm = _smooth_wilder(plus_dm_w, lb)
                    smooth_minus_dm = _smooth_wilder(minus_dm_w, lb)
                    
                    if smooth_tr > 1e-8:
                        plus_di = 100.0 * (smooth_plus_dm / smooth_tr)
                        minus_di = 100.0 * (smooth_minus_dm / smooth_tr)
                        denom_di = plus_di + minus_di
                        dx = 100.0 * abs(plus_di - minus_di) / denom_di if denom_di > 1e-8 else 0.0
                    else:
                        plus_di = 0.0
                        minus_di = 0.0
                        dx = 0.0
                    
                    if len(ohlc_list) >= 2 * lb:
                        tr_all = []
                        plus_dm_all = []
                        minus_dm_all = []
                        for idx in range(len(ohlc_list) - 2 * lb + 1, len(ohlc_list)):
                            h_curr = ohlc_list[idx]["high"]
                            l_curr = ohlc_list[idx]["low"]
                            c_prev = ohlc_list[idx-1]["close"]
                            h_prev = ohlc_list[idx-1]["high"]
                            l_prev = ohlc_list[idx-1]["low"]
                            
                            tr = max(h_curr - l_curr, abs(h_curr - c_prev), abs(l_curr - c_prev))
                            tr_all.append(tr)
                            
                            up_move = h_curr - h_prev
                            down_move = l_prev - l_curr
                            
                            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
                            plus_dm_all.append(plus_dm)
                            
                            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
                            minus_dm_all.append(minus_dm)
                        
                        def _get_wilder_smoothed_list(vals, period):
                            smoothed = []
                            curr = sum(vals[:period]) / float(period)
                            smoothed.append(curr)
                            for v in vals[period:]:
                                curr = curr - (curr / float(period)) + v
                                smoothed.append(curr)
                            return smoothed
                        
                        tr_smooth = _get_wilder_smoothed_list(tr_all, lb)
                        plus_smooth = _get_wilder_smoothed_list(plus_dm_all, lb)
                        minus_smooth = _get_wilder_smoothed_list(minus_dm_all, lb)
                        
                        dx_history = []
                        for idx in range(len(tr_smooth)):
                            t_s = tr_smooth[idx]
                            p_s = plus_smooth[idx]
                            m_s = minus_smooth[idx]
                            if t_s > 1e-8:
                                p_di = 100.0 * (p_s / t_s)
                                m_di = 100.0 * (m_s / t_s)
                                den = p_di + m_di
                                dx_val = 100.0 * abs(p_di - m_di) / den if den > 1e-8 else 0.0
                            else:
                                dx_val = 0.0
                            dx_history.append(dx_val)
                        
                        adx_smooth = _get_wilder_smoothed_list(dx_history, lb)
                        feats[f"adx_{lb_minutes}"] = float(adx_smooth[-1])
                        feats[f"plus_di_{lb_minutes}"] = float(100.0 * (plus_smooth[-1] / tr_smooth[-1]) if tr_smooth[-1] > 1e-8 else 0.0)
                        feats[f"minus_di_{lb_minutes}"] = float(100.0 * (minus_smooth[-1] / tr_smooth[-1]) if tr_smooth[-1] > 1e-8 else 0.0)
                    else:
                        feats[f"adx_{lb_minutes}"] = float(dx)
                        feats[f"plus_di_{lb_minutes}"] = float(plus_di)
                        feats[f"minus_di_{lb_minutes}"] = float(minus_di)
                else:
                    feats[f"adx_{lb_minutes}"] = float("nan")
                    feats[f"plus_di_{lb_minutes}"] = float("nan")
                    feats[f"minus_di_{lb_minutes}"] = float("nan")
        else:
            for i, lb in enumerate(self.spec.lookbacks_prices):
                lb_minutes = self.spec._lookbacks_prices_minutes[i]
                feats[f"adx_{lb_minutes}"] = float("nan")
                feats[f"plus_di_{lb_minutes}"] = float("nan")
                feats[f"minus_di_{lb_minutes}"] = float("nan")

        # 13. Awesome Oscillator (AO)
        if ohlc_list and len(ohlc_list) >= 34:
            tp_ao = [(b["high"] + b["low"]) / 2.0 for b in ohlc_list]
            sma_5 = sum(tp_ao[-5:]) / 5.0
            sma_34 = sum(tp_ao[-34:]) / 34.0
            feats["ao"] = float(sma_5 - sma_34)
        else:
            feats["ao"] = float("nan")

        # 14. Price Volume Trend (PVT)
        if volume is not None:
            prev_pvt = st.get("pvt", 0.0)
            if last is not None and last > 1e-8:
                pvt_diff = float(volume) * (price - float(last)) / float(last)
                current_pvt = prev_pvt + pvt_diff
            else:
                current_pvt = 0.0
            st["pvt"] = current_pvt
            feats["pvt"] = float(current_pvt)
        else:
            feats["pvt"] = float("nan")

        # 15. Volatility of Log Returns, Price Momentum, Donchian Channels, Chaikin Money Flow
        log_rets_1bar = []
        for idx in range(1, len(seq)):
            p_curr = seq[idx]
            p_prev = seq[idx-1]
            if p_prev > 1e-8 and p_curr > 1e-8:
                log_rets_1bar.append(math.log(p_curr / p_prev))
            else:
                log_rets_1bar.append(0.0)

        cmf_mfvs = []
        cmf_vols = []
        if ohlc_list:
            for b in ohlc_list:
                h = b["high"]
                l = b["low"]
                c = b["close"]
                v = b.get("volume", 0.0)
                denom = h - l
                mfm = ((c - l) - (h - c)) / denom if denom > 1e-8 else 0.0
                cmf_mfvs.append(mfm * v)
                cmf_vols.append(v)

        for i, lb in enumerate(self.spec.lookbacks_prices):
            lb_minutes = self.spec._lookbacks_prices_minutes[i]

            # Std Returns
            if len(log_rets_1bar) >= lb:
                window_rets = log_rets_1bar[-lb:]
                mean_ret = sum(window_rets) / float(lb)
                var_ret = sum((r - mean_ret) ** 2 for r in window_rets) / float(lb)
                std_ret = math.sqrt(var_ret)
                feats[f"std_returns_{lb_minutes}"] = float(std_ret)
            else:
                feats[f"std_returns_{lb_minutes}"] = float("nan")

            # Momentum
            if len(seq) > lb:
                feats[f"mom_{lb_minutes}"] = float(price - seq[-(lb + 1)])
            else:
                feats[f"mom_{lb_minutes}"] = float("nan")

            # Donchian Channels
            if ohlc_list and len(ohlc_list) >= lb:
                window_bars = ohlc_list[-lb:]
                highest_high = max(b["high"] for b in window_bars)
                lowest_low = min(b["low"] for b in window_bars)
                feats[f"donchian_upper_{lb_minutes}"] = float(highest_high)
                feats[f"donchian_lower_{lb_minutes}"] = float(lowest_low)
                feats[f"donchian_middle_{lb_minutes}"] = float((highest_high + lowest_low) / 2.0)
            else:
                feats[f"donchian_upper_{lb_minutes}"] = float("nan")
                feats[f"donchian_lower_{lb_minutes}"] = float("nan")
                feats[f"donchian_middle_{lb_minutes}"] = float("nan")

            # Chaikin Money Flow (CMF)
            if ohlc_list and len(ohlc_list) >= lb:
                sum_mfv = sum(cmf_mfvs[-lb:])
                sum_vol = sum(cmf_vols[-lb:])
                cmf_val = sum_mfv / sum_vol if sum_vol > 1e-8 else 0.0
                feats[f"cmf_{lb_minutes}"] = float(cmf_val)
            else:
                feats[f"cmf_{lb_minutes}"] = float("nan")

        # Рассчитываем Yang-Zhang волатильность для каждого окна
        # CRITICAL FIX: Теперь всегда вычисляем, используя fallback к close-to-close если OHLC недоступны
        if self.spec.yang_zhang_windows:
            ohlc_list = list(st["ohlc_bars"]) if st["ohlc_bars"] else []
            close_list = list(st["prices"])  # Всегда доступны для fallback
            # CRITICAL FIX #1: Используем исходные значения в минутах для именования, бары для индексирования
            for i, window in enumerate(self.spec.yang_zhang_windows):
                # Создаем имя признака с поддержкой дней, часов и минут
                window_minutes = self.spec._yang_zhang_windows_minutes[i]
                window_name = _format_window_name(window_minutes)
                feature_name = f"yang_zhang_{window_name}"

                # Передаем как OHLC, так и close цены для hybrid подхода
                yz_vol = calculate_yang_zhang_volatility(
                    ohlc_list,
                    window,
                    close_prices=close_list
                )
                if yz_vol is not None:
                    feats[feature_name] = float(yz_vol)
                else:
                    feats[feature_name] = float("nan")

        # Рассчитываем Parkinson волатильность для каждого окна
        if self.spec.parkinson_windows and st["ohlc_bars"]:
            ohlc_list = list(st["ohlc_bars"])
            # CRITICAL FIX #1: Используем исходные значения в минутах для именования, бары для индексирования
            for i, window in enumerate(self.spec.parkinson_windows):
                # Создаем имя признака с поддержкой дней, часов и минут
                window_minutes = self.spec._parkinson_windows_minutes[i]
                window_name = _format_window_name(window_minutes)
                feature_name = f"parkinson_{window_name}"

                if len(ohlc_list) >= window:
                    pk_vol = calculate_parkinson_volatility(ohlc_list, window)
                    if pk_vol is not None:
                        feats[feature_name] = float(pk_vol)
                    else:
                        feats[feature_name] = float("nan")
                else:
                    feats[feature_name] = float("nan")

        # Рассчитываем условную волатильность с robust fallback стратегией
        # GARCH(1,1) -> EWMA -> Historical Volatility
        if self.spec.garch_windows:
            price_list = list(st["prices"])
            # CRITICAL FIX #1: Используем исходные значения в минутах для именования, бары для индексирования
            for i, window in enumerate(self.spec.garch_windows):
                # Создаем имя признака с поддержкой дней, часов и минут
                window_minutes = self.spec._garch_windows_minutes[i]
                window_name = _format_window_name(window_minutes)
                feature_name = f"garch_{window_name}"

                # calculate_garch_volatility использует cascading fallback:
                # 1. Пробует GARCH(1,1) если >= 50 баров
                # 2. Fallback на EWMA если недостаточно данных или GARCH не сходится
                # 3. Fallback на Historical Volatility для минимальных данных (2+ бара)
                # Возвращает None только если < 2 баров
                garch_vol = calculate_garch_volatility(price_list, window)
                if garch_vol is not None:
                    feats[feature_name] = float(garch_vol)
                else:
                    # Только если данных меньше 2 баров (очень редко)
                    feats[feature_name] = float("nan")

        # Рассчитываем Taker Buy Ratio и его производные
        if st["taker_buy_ratios"]:
            ratio_list = list(st["taker_buy_ratios"])

            # Добавляем текущее значение taker_buy_ratio
            if ratio_list:
                feats["taker_buy_ratio"] = float(ratio_list[-1])
            else:
                feats["taker_buy_ratio"] = float("nan")

            # Рассчитываем скользящее среднее для каждого окна
            if self.spec.taker_buy_ratio_windows:
                # CRITICAL FIX #1: Используем исходные значения в минутах для именования, бары для индексирования
                for i, window in enumerate(self.spec.taker_buy_ratio_windows):
                    # Создаем имя признака с поддержкой дней, часов и минут
                    window_minutes = self.spec._taker_buy_ratio_windows_minutes[i]
                    window_name = _format_window_name(window_minutes)
                    feature_name = f"taker_buy_ratio_sma_{window_name}"

                    if len(ratio_list) >= window:
                        window_data = ratio_list[-window:]
                        sma = sum(window_data) / float(len(window_data))
                        feats[feature_name] = float(sma)
                    else:
                        feats[feature_name] = float("nan")

            # Рассчитываем моментум (Rate of Change за последние N периодов)
            # CRITICAL FIX: Используем ROC (Rate of Change) вместо абсолютной разницы
            # ROC = (current - past) / past - это стандарт для momentum индикаторов
            # Преимущества: учитывает базовый уровень, сопоставимо между периодами
            if self.spec.taker_buy_ratio_momentum:
                # CRITICAL FIX #1: Используем исходные значения в минутах для именования, бары для индексирования
                for i, window in enumerate(self.spec.taker_buy_ratio_momentum):
                    # Создаем имя признака с поддержкой дней, часов и минут
                    window_minutes = self.spec._taker_buy_ratio_momentum_minutes[i]
                    window_name = _format_window_name(window_minutes)
                    feature_name = f"taker_buy_ratio_momentum_{window_name}"

                    if len(ratio_list) >= window + 1:
                        current = ratio_list[-1]
                        past = ratio_list[-(window + 1)]

                        # CRITICAL FIX: Используем ROC вместо абсолютной разницы
                        # Защита от деления на очень маленькие числа
                        # Threshold 0.01 (1%) prevents extreme ROC values
                        # For taker_buy_ratio in [0, 1], this is reasonable
                        if abs(past) > 0.01:
                            # ROC (Rate of Change): процентное изменение
                            momentum = (current - past) / past
                        else:
                            # Fallback для случая когда past очень маленькое (<1%)
                            # Используем знак разницы без деления
                            # +1.0 для роста, -1.0 для падения, 0 для неизменности
                            if current > past + 0.001:  # Выросло значительно
                                momentum = 1.0
                            elif current < past - 0.001:  # Упало значительно
                                momentum = -1.0
                            else:  # Практически не изменилось
                                momentum = 0.0

                        feats[feature_name] = float(momentum)
                    else:
                        feats[feature_name] = float("nan")

            # Z-score нормализация будет применена автоматически в FeaturePipeline

        # Рассчитываем Cumulative Volume Delta (CVD) для каждого окна
        if st["volume_deltas"] and self.spec.cvd_windows:
            delta_list = list(st["volume_deltas"])

            # CRITICAL FIX #1: Используем исходные значения в минутах для именования, бары для индексирования
            for i, window in enumerate(self.spec.cvd_windows):
                # Создаем имя признака с поддержкой дней, часов и минут
                window_minutes = self.spec._cvd_windows_minutes[i]
                window_name = _format_window_name(window_minutes)
                feature_name = f"cvd_{window_name}"

                if len(delta_list) >= window:
                    # CVD = кумулятивная сумма volume_delta за окно
                    window_data = delta_list[-window:]
                    cvd = sum(window_data)
                    feats[feature_name] = float(cvd)
                else:
                    feats[feature_name] = float("nan")

        return feats


def apply_offline_features(
    df: pd.DataFrame,
    *,
    spec: FeatureSpec,
    ts_col: str = "ts_ms",
    symbol_col: str = "symbol",
    price_col: str = "price",
    open_col: Optional[str] = None,
    high_col: Optional[str] = None,
    low_col: Optional[str] = None,
    volume_col: Optional[str] = None,
    taker_buy_base_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Оффлайн-расчёт фич с точным соответствием онлайновому трансформеру.
    На входе ожидается таблица 1m-просэмплированных цен (price) и опционально OHLC, volume, taker_buy_base.
    На выходе: ts_ms, symbol, ref_price, sma_*, ret_*m, rsi, yang_zhang_*h, parkinson_*h, taker_buy_ratio*.

    Args:
        df: DataFrame с данными
        spec: спецификация признаков
        ts_col: имя колонки временной метки
        symbol_col: имя колонки символа
        price_col: имя колонки цены (обычно close)
        open_col: имя колонки open (опционально)
        high_col: имя колонки high (опционально)
        low_col: имя колонки low (опционально)
        volume_col: имя колонки volume (опционально для Taker Buy Ratio)
        taker_buy_base_col: имя колонки taker_buy_base (опционально для Taker Buy Ratio)
    """
    if df is None or df.empty:
        base_cols = [ts_col, symbol_col, "ref_price", "rsi", "macd", "macd_signal", "macd_histogram", "obv", "ao", "pvt"]
        # КРИТИЧНО: используем МИНУТЫ для имен SMA (согласованность с mediator.py и update())
        # Онлайн (update) генерирует: sma_240, sma_720, sma_1200, sma_1440, sma_5040, sma_12000
        # Оффлайн (apply_offline_features) должен генерировать ТЕ ЖЕ имена
        base_cols += [f"sma_{x}" for x in spec._lookbacks_prices_minutes]
        # Используем минуты для имен returns (форматирование через _format_window_name)
        base_cols += [f"ret_{_format_window_name(x)}" for x in spec._lookbacks_prices_minutes]
        if spec.yang_zhang_windows:
            base_cols += [f"yang_zhang_{_format_window_name(w)}" for w in spec._yang_zhang_windows_minutes]
        if spec.parkinson_windows:
            base_cols += [f"parkinson_{_format_window_name(w)}" for w in spec._parkinson_windows_minutes]
        if spec.garch_windows:
            base_cols += [f"garch_{_format_window_name(w)}" for w in spec._garch_windows_minutes]
        if spec.taker_buy_ratio_windows or spec.taker_buy_ratio_momentum:
            base_cols.append("taker_buy_ratio")
        if spec.taker_buy_ratio_windows:
            base_cols += [f"taker_buy_ratio_sma_{_format_window_name(w)}" for w in spec._taker_buy_ratio_windows_minutes]
        if spec.taker_buy_ratio_momentum:
            base_cols += [f"taker_buy_ratio_momentum_{_format_window_name(w)}" for w in spec._taker_buy_ratio_momentum_minutes]
        if spec.cvd_windows:
            base_cols += [f"cvd_{_format_window_name(w)}" for w in spec._cvd_windows_minutes]
        
        # High quality indicators from previous/current updates
        for x in spec._lookbacks_prices_minutes:
            base_cols.extend([
                f"bollinger_upper_{x}", f"bollinger_lower_{x}",
                f"bollinger_bandwidth_{x}", f"bollinger_percent_b_{x}",
                f"stoch_k_{x}", f"stoch_d_{x}", f"atr_{x}", f"cci_{x}",
                f"cmo_{x}", f"roc_{x}", f"ema_{x}", f"williams_r_{x}",
                f"keltner_middle_{x}", f"keltner_upper_{x}", f"keltner_lower_{x}",
                f"mfi_{x}", f"slope_{x}", f"adx_{x}", f"plus_di_{x}", f"minus_di_{x}",
                f"std_returns_{x}", f"mom_{x}", f"donchian_upper_{x}",
                f"donchian_lower_{x}", f"donchian_middle_{x}", f"cmf_{x}"
            ])
        return pd.DataFrame(columns=base_cols)

    d = df.copy()
    if symbol_col not in d.columns or ts_col not in d.columns:
        raise ValueError(f"Вход должен содержать колонки '{symbol_col}' и '{ts_col}'")
    if price_col not in d.columns:
        raise ValueError(f"Вход должен содержать колонку цены '{price_col}'")

    # Определяем какие колонки нужно сохранить
    cols_to_keep = [ts_col, symbol_col, price_col]
    has_ohlc = False
    if open_col and high_col and low_col:
        if open_col in d.columns and high_col in d.columns and low_col in d.columns:
            cols_to_keep.extend([open_col, high_col, low_col])
            has_ohlc = True

    has_volume_data = False
    if volume_col and taker_buy_base_col:
        if volume_col in d.columns and taker_buy_base_col in d.columns:
            cols_to_keep.extend([volume_col, taker_buy_base_col])
            has_volume_data = True

    # CRITICAL FIX: Selective dropna to prevent temporal discontinuity
    # Only drop rows where REQUIRED fields (ts, symbol, price) are NaN
    # Keep rows where OPTIONAL fields (OHLC, volume, taker_buy_base) have NaN
    # This prevents data gaps and maintains temporal continuity
    d = d[cols_to_keep].copy()
    # Drop only if required fields are NaN
    required_cols = [ts_col, symbol_col, price_col]
    d = d.dropna(subset=required_cols).copy()

    d[ts_col] = d[ts_col].astype("int64")
    d[symbol_col] = d[symbol_col].astype(str)

    d = d.sort_values([symbol_col, ts_col]).reset_index(drop=True)

    out_rows: List[Dict[str, Any]] = []
    current_symbol: Optional[str] = None
    transformer: Optional[OnlineFeatureTransformer] = None

    for _, row in d.iterrows():
        sym = str(row[symbol_col]).upper()
        ts = int(row[ts_col])
        px = float(row[price_col])

        if transformer is None or current_symbol != sym:
            current_symbol = sym
            transformer = OnlineFeatureTransformer(spec)

        # Передаем OHLC и volume данные если доступны
        update_kwargs = {
            "symbol": sym,
            "ts_ms": ts,
            "close": px,
        }

        if has_ohlc:
            # Handle NaN in OHLC data gracefully - skip if any are NaN
            open_val = row[open_col]
            high_val = row[high_col]
            low_val = row[low_col]
            if not (pd.isna(open_val) or pd.isna(high_val) or pd.isna(low_val)):
                update_kwargs["open_price"] = float(open_val)
                update_kwargs["high"] = float(high_val)
                update_kwargs["low"] = float(low_val)

        if has_volume_data:
            # Handle NaN in volume data gracefully - skip if any are NaN
            vol_val = row[volume_col]
            tbb_val = row[taker_buy_base_col]
            if not (pd.isna(vol_val) or pd.isna(tbb_val)):
                update_kwargs["volume"] = float(vol_val)
                update_kwargs["taker_buy_base"] = float(tbb_val)

        feats = transformer.update(**update_kwargs)

        out_rows.append(feats)

    out = pd.DataFrame(out_rows)
    if symbol_col != "symbol" and "symbol" in out.columns:
        out = out.rename(columns={"symbol": symbol_col})
    out = out.sort_values([symbol_col, ts_col]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Backwards compatibility shims (legacy API)
# ---------------------------------------------------------------------------
# Older code paths/tests import TransformerSpec/FeatureTransformer/OnlineFeatureTransform.
# Map them to the current FeatureSpec/OnlineFeatureTransformer implementation to
# keep the legacy surface stable without duplicating logic.
TransformerSpec = FeatureSpec
OnlineFeatureTransform = OnlineFeatureTransformer


class FeatureTransformer:
    """Lightweight wrapper around OnlineFeatureTransformer for legacy callers."""

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec
        self._transformer = OnlineFeatureTransformer(spec)
        self._last_symbol: Optional[str] = None
        self._ts_counter: int = 0

    def _coerce_ts_ms(self, bar: Dict[str, Any]) -> int:
        """Extract or synthesize a timestamp in milliseconds for update()."""
        ts = bar.get("ts_ms") or bar.get("timestamp")
        if ts is None and "time_key" in bar:
            try:
                ts = int(pd.Timestamp(bar["time_key"]).value // 1_000_000)
            except Exception:
                ts = None
        if ts is None:
            # Fallback to a simple counter to preserve ordering
            self._ts_counter += 1
            return self._ts_counter
        try:
            return int(ts)
        except Exception:
            self._ts_counter += 1
            return self._ts_counter

    def transform(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single bar dict and return computed features."""
        symbol = str(bar.get("symbol", "")).upper()
        self._last_symbol = symbol or self._last_symbol

        update_kwargs: Dict[str, Any] = {
            "symbol": symbol or "UNKNOWN",
            "ts_ms": self._coerce_ts_ms(bar),
            "close": float(bar.get("close", 0.0)),
        }

        # Optional OHLCV fields if present
        if all(k in bar for k in ("open", "high", "low")):
            try:
                update_kwargs["open_price"] = float(bar["open"])
                update_kwargs["high"] = float(bar["high"])
                update_kwargs["low"] = float(bar["low"])
            except Exception:
                pass

        if "volume" in bar:
            try:
                update_kwargs["volume"] = float(bar["volume"])
            except Exception:
                pass

        # Taker buy volume naming varies across datasets
        tbb_value = None
        if "taker_buy_base_volume" in bar:
            tbb_value = bar["taker_buy_base_volume"]
        elif "taker_buy_base" in bar:
            tbb_value = bar["taker_buy_base"]
        if tbb_value is not None:
            try:
                update_kwargs["taker_buy_base"] = float(tbb_value)
            except Exception:
                pass

        return self._transformer.update(**update_kwargs)

    def get_state(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Expose internal state for inspection (e.g., avg_gain/avg_loss for RSI)."""
        sym = (symbol or self._last_symbol or "").upper()
        if not sym:
            return {}
        return self._transformer._state.get(sym, {})
