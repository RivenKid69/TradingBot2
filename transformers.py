# features/transformers.py
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
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


def calculate_yang_zhang_volatility(ohlc_bars: List[Dict[str, float]], n: int) -> Optional[float]:
    """
    Рассчитывает волатильность Yang-Zhang для последних n баров.

    Формула:
    σ²_YZ = σ²_o + k·σ²_c + (1-k)·σ²_rs
    где:
    - σ²_o = ночная волатильность = (1/(n-1)) Σ(log(O_i/C_{i-1}) - μ_o)²
    - σ²_c = волатильность open-close = (1/(n-1)) Σ(log(C_i/O_i) - μ_c)²
    - σ²_rs = Роджерс-Сатчелл = (1/n) Σ[log(H_i/C_i)·log(H_i/O_i) + log(L_i/C_i)·log(L_i/O_i)]
    - k = 0.34 (эмпирически оптимальный вес)

    Args:
        ohlc_bars: список словарей с ключами 'open', 'high', 'low', 'close'
        n: размер окна

    Returns:
        Волатильность Yang-Zhang или None если недостаточно данных
    """
    if not ohlc_bars or len(ohlc_bars) < n or n < 2:
        return None

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

    Формула:
    σ_Parkinson = sqrt[(1/(4n·log(2))) · Σ(log(H_i/L_i))²]

    Оценщик Паркинсона в 7,4 раза более эффективен, чем оценщик close-to-close,
    так как использует информацию о дневном диапазоне (High-Low).

    Args:
        ohlc_bars: список словарей с ключами 'high' и 'low'
        n: размер окна

    Returns:
        Волатильность Паркинсона или None если недостаточно данных
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
        parkinson_var = sum_sq / (4 * valid_bars * math.log(2))

        # Возвращаем стандартное отклонение
        return math.sqrt(parkinson_var)

    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None


def calculate_garch_volatility(prices: List[float], n: int) -> Optional[float]:
    """
    Рассчитывает условную волатильность GARCH(1,1) на основе логарифмических доходностей.

    Модель GARCH(1,1):
    r_t = μ + ε_t
    ε_t = σ_t * z_t, где z_t ~ N(0,1)
    σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}

    Функция подгоняет модель GARCH(1,1) на скользящем окне из последних n наблюдений
    и возвращает прогноз условной волатильности на один шаг вперед σ_{t+1}.

    Args:
        prices: список цен
        n: размер скользящего окна (рекомендуется 500+ наблюдений)

    Returns:
        Прогноз условной волатильности σ_{t+1} или None если недостаточно данных
        или подгонка модели не удалась
    """
    if not prices or len(prices) < n or n < 50:  # минимум 50 наблюдений для стабильной оценки
        return None

    try:
        # Берем последние n цен
        price_window = np.array(prices[-n:], dtype=float)

        # Проверяем валидность данных
        if np.any(price_window <= 0) or np.any(~np.isfinite(price_window)):
            return None

        # Вычисляем логарифмические доходности
        log_returns = np.log(price_window[1:] / price_window[:-1])

        # Проверяем, что есть вариация в данных
        if np.std(log_returns) < 1e-10:
            return None

        # Конвертируем в процентные доходности для лучшей численной стабильности
        returns_pct = log_returns * 100

        # Подавляем предупреждения от arch (они могут быть шумными)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Создаем и подгоняем модель GARCH(1,1)
            # rescale=False чтобы сохранить исходный масштаб данных
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
                update_freq=0,  # не выводим итерационный процесс
                disp='off',  # отключаем вывод
                show_warning=False,  # отключаем предупреждения
                options={'maxiter': 1000}
            )

            # Получаем прогноз условной волатильности на 1 шаг вперед
            # forecast возвращает дисперсию, нам нужно стандартное отклонение
            forecast = result.forecast(horizon=1, reindex=False)
            forecast_variance = forecast.variance.values[-1, 0]

            if not np.isfinite(forecast_variance) or forecast_variance <= 0:
                return None

            # Конвертируем обратно из процентного масштаба
            forecast_volatility = np.sqrt(forecast_variance) / 100

            return float(forecast_volatility)

    except (ValueError, RuntimeError, np.linalg.LinAlgError, Exception):
        # Модель GARCH может не сходиться в некоторых случаях
        return None


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
    """

    lookbacks_prices: List[int]
    rsi_period: int = 14
    yang_zhang_windows: Optional[List[int]] = None
    parkinson_windows: Optional[List[int]] = None
    garch_windows: Optional[List[int]] = None
    taker_buy_ratio_windows: Optional[List[int]] = None
    taker_buy_ratio_momentum: Optional[List[int]] = None
    cvd_windows: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.lookbacks_prices, list)
            or len(self.lookbacks_prices) == 0
        ):
            # Для 4h интервала: 4h, 12h, 24h, 50 баров (200h)
            # 1 бар 4h = 240 минут
            # 4h = 1 бар = 240 минут
            # 12h = 3 бара = 720 минут
            # 24h = 6 баров = 1440 минут
            # 50 баров = 200h = 12000 минут (для sma_50)
            self.lookbacks_prices = [240, 720, 1440, 12000]
        self.lookbacks_prices = [
            int(abs(x)) for x in self.lookbacks_prices if int(abs(x)) > 0
        ]
        self.rsi_period = int(self.rsi_period)

        # Инициализация окон Yang-Zhang для 4h интервала: 48ч, 7д, 30д в минутах
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

        # Инициализация окон моментума Taker Buy Ratio для 4h интервала: 4ч, 8ч, 12ч в минутах
        # 4h = 1 бар = 240 минут
        # 8h = 2 бара = 480 минут
        # 12h = 3 бара = 720 минут
        if self.taker_buy_ratio_momentum is None:
            self.taker_buy_ratio_momentum = [4 * 60, 8 * 60, 12 * 60]  # 240, 480, 720 минут
        elif isinstance(self.taker_buy_ratio_momentum, list):
            self.taker_buy_ratio_momentum = [
                int(abs(x)) for x in self.taker_buy_ratio_momentum if int(abs(x)) > 0
            ]
        else:
            self.taker_buy_ratio_momentum = []

        # Инициализация окон Cumulative Volume Delta: 24ч, 7д в минутах (без изменений)
        if self.cvd_windows is None:
            self.cvd_windows = [24 * 60, 7 * 24 * 60]  # 1440, 10080 минут
        elif isinstance(self.cvd_windows, list):
            self.cvd_windows = [
                int(abs(x)) for x in self.cvd_windows if int(abs(x)) > 0
            ]
        else:
            self.cvd_windows = []

        # Инициализация окон GARCH для 4h интервала: 7д, 14д, 30д в минутах
        # КРИТИЧНО: GARCH требует минимум 50 наблюдений (строка 212)!
        # 7d = 42 бара = 10080 минут
        # 14d = 84 бара = 20160 минут
        # 30d = 180 баров = 43200 минут
        if self.garch_windows is None:
            self.garch_windows = [7 * 24 * 60, 14 * 24 * 60, 30 * 24 * 60]  # 10080, 20160, 43200 минут
        elif isinstance(self.garch_windows, list):
            self.garch_windows = [
                int(abs(x)) for x in self.garch_windows if int(abs(x)) > 0
            ]
        else:
            self.garch_windows = []


class OnlineFeatureTransformer:
    """
    Онлайн-трансформер: состояние на символ, детерминистичное поведение.
    Полностью соответствует онлайновой логике (как раньше в FeaturePipe):
      - SMA и ретёрны из окна цен (1 точка в минуту)
      - RSI по Вайльдеру: скользящие avg_gain/avg_loss с периодом p
      - Yang-Zhang волатильность: комплексная OHLC-волатильность
      - Parkinson волатильность: волатильность диапазона High-Low
    """

    def __init__(self, spec: FeatureSpec) -> None:
        self.spec = spec
        self._state: Dict[str, Dict[str, Any]] = {}

    def _ensure_state(self, symbol: str) -> Dict[str, Any]:
        st = self._state.get(symbol)
        if st is None:
            # Определяем максимальную длину окна для всех фич
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

            st = {
                "prices": deque(maxlen=maxlen),  # type: deque[float]
                "avg_gain": None,  # type: Optional[float]
                "avg_loss": None,  # type: Optional[float]
                "last_close": None,  # type: Optional[float]
                # Для Yang-Zhang волатильности нужны OHLC
                "ohlc_bars": deque(maxlen=maxlen),  # type: deque[Dict[str, float]]
                # Для Taker Buy Ratio нужны значения ratio
                "taker_buy_ratios": deque(maxlen=maxlen),  # type: deque[float]
                # Для Cumulative Volume Delta нужны дельты объема
                "volume_deltas": deque(maxlen=maxlen),  # type: deque[float]
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

        Args:
            symbol: символ торгового инструмента
            ts_ms: временная метка в миллисекундах
            close: цена закрытия
            open_price: цена открытия (опционально для Yang-Zhang)
            high: максимальная цена (опционально для Yang-Zhang)
            low: минимальная цена (опционально для Yang-Zhang)
            volume: объем торгов (опционально для Taker Buy Ratio)
            taker_buy_base: объем покупок taker (опционально для Taker Buy Ratio)
        """
        sym = str(symbol).upper()
        price = float(close)
        st = self._ensure_state(sym)

        last = st["last_close"]
        if last is not None:
            delta = price - float(last)
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            if st["avg_gain"] is None or st["avg_loss"] is None:
                st["avg_gain"] = float(gain)
                st["avg_loss"] = float(loss)
            else:
                p = self.spec.rsi_period
                st["avg_gain"] = ((float(st["avg_gain"]) * (p - 1)) + gain) / p
                st["avg_loss"] = ((float(st["avg_loss"]) * (p - 1)) + loss) / p
        st["last_close"] = price

        st["prices"].append(price)

        # Сохраняем OHLC данные для Yang-Zhang
        if open_price is not None and high is not None and low is not None:
            ohlc_bar = {
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
            }
            st["ohlc_bars"].append(ohlc_bar)

        # Вычисляем и сохраняем Taker Buy Ratio
        if volume is not None and taker_buy_base is not None and volume > 0:
            # Добавляем clamping на случай аномальных данных (taker_buy_base > volume)
            # Нормальный диапазон: [0.0, 1.0]
            taker_buy_ratio = min(1.0, max(0.0, float(taker_buy_base) / float(volume)))
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
        for lb in self.spec.lookbacks_prices:
            if len(seq) >= lb:
                window = seq[-lb:]
                sma = sum(window) / float(lb)
                feats[f"sma_{lb}"] = float(sma)
                first = float(window[0])
                # Создаем имя для returns с поддержкой дней, часов и минут
                ret_name = f"ret_{_format_window_name(lb)}"
                feats[ret_name] = (
                    float(math.log(price / first)) if first > 0 else 0.0
                )

        if (
            st["avg_gain"] is not None
            and st["avg_loss"] is not None
            and float(st["avg_loss"]) > 0.0
        ):
            rs = float(st["avg_gain"]) / float(st["avg_loss"])
            feats["rsi"] = float(100.0 - (100.0 / (1.0 + rs)))
        else:
            feats["rsi"] = float("nan")

        # Рассчитываем Yang-Zhang волатильность для каждого окна
        if self.spec.yang_zhang_windows and st["ohlc_bars"]:
            ohlc_list = list(st["ohlc_bars"])
            for window in self.spec.yang_zhang_windows:
                # Создаем имя признака с поддержкой дней, часов и минут
                window_name = _format_window_name(window)
                feature_name = f"yang_zhang_{window_name}"

                if len(ohlc_list) >= window:
                    yz_vol = calculate_yang_zhang_volatility(ohlc_list, window)
                    if yz_vol is not None:
                        feats[feature_name] = float(yz_vol)
                    else:
                        feats[feature_name] = float("nan")
                else:
                    feats[feature_name] = float("nan")

        # Рассчитываем Parkinson волатильность для каждого окна
        if self.spec.parkinson_windows and st["ohlc_bars"]:
            ohlc_list = list(st["ohlc_bars"])
            for window in self.spec.parkinson_windows:
                # Создаем имя признака с поддержкой дней, часов и минут
                window_name = _format_window_name(window)
                feature_name = f"parkinson_{window_name}"

                if len(ohlc_list) >= window:
                    pk_vol = calculate_parkinson_volatility(ohlc_list, window)
                    if pk_vol is not None:
                        feats[feature_name] = float(pk_vol)
                    else:
                        feats[feature_name] = float("nan")
                else:
                    feats[feature_name] = float("nan")

        # Рассчитываем условную волатильность GARCH(1,1) для каждого окна
        if self.spec.garch_windows:
            price_list = list(st["prices"])
            for window in self.spec.garch_windows:
                # Создаем имя признака с поддержкой дней, часов и минут
                window_name = _format_window_name(window)
                feature_name = f"garch_{window_name}"

                if len(price_list) >= window:
                    garch_vol = calculate_garch_volatility(price_list, window)
                    if garch_vol is not None:
                        feats[feature_name] = float(garch_vol)
                    else:
                        feats[feature_name] = float("nan")
                else:
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
                for window in self.spec.taker_buy_ratio_windows:
                    # Создаем имя признака с поддержкой дней, часов и минут
                    window_name = _format_window_name(window)
                    feature_name = f"taker_buy_ratio_sma_{window_name}"

                    if len(ratio_list) >= window:
                        window_data = ratio_list[-window:]
                        sma = sum(window_data) / float(len(window_data))
                        feats[feature_name] = float(sma)
                    else:
                        feats[feature_name] = float("nan")

            # Рассчитываем моментум (изменение за последние N периодов)
            if self.spec.taker_buy_ratio_momentum:
                for window in self.spec.taker_buy_ratio_momentum:
                    # Создаем имя признака с поддержкой дней, часов и минут
                    window_name = _format_window_name(window)
                    feature_name = f"taker_buy_ratio_momentum_{window_name}"

                    if len(ratio_list) >= window + 1:
                        current = ratio_list[-1]
                        past = ratio_list[-(window + 1)]
                        momentum = current - past
                        feats[feature_name] = float(momentum)
                    else:
                        feats[feature_name] = float("nan")

            # Z-score нормализация будет применена автоматически в FeaturePipeline

        # Рассчитываем Cumulative Volume Delta (CVD) для каждого окна
        if st["volume_deltas"] and self.spec.cvd_windows:
            delta_list = list(st["volume_deltas"])

            for window in self.spec.cvd_windows:
                # Создаем имя признака с поддержкой дней, часов и минут
                window_name = _format_window_name(window)
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
        base_cols = [ts_col, symbol_col, "ref_price", "rsi"]
        base_cols += [f"sma_{x}" for x in spec.lookbacks_prices]
        # Используем _format_window_name для правильного форматирования имен
        base_cols += [f"ret_{_format_window_name(x)}" for x in spec.lookbacks_prices]
        if spec.yang_zhang_windows:
            base_cols += [f"yang_zhang_{_format_window_name(w)}" for w in spec.yang_zhang_windows]
        if spec.parkinson_windows:
            base_cols += [f"parkinson_{_format_window_name(w)}" for w in spec.parkinson_windows]
        if spec.garch_windows:
            base_cols += [f"garch_{_format_window_name(w)}" for w in spec.garch_windows]
        if spec.taker_buy_ratio_windows or spec.taker_buy_ratio_momentum:
            base_cols.append("taker_buy_ratio")
        if spec.taker_buy_ratio_windows:
            base_cols += [f"taker_buy_ratio_sma_{_format_window_name(w)}" for w in spec.taker_buy_ratio_windows]
        if spec.taker_buy_ratio_momentum:
            base_cols += [f"taker_buy_ratio_momentum_{_format_window_name(w)}" for w in spec.taker_buy_ratio_momentum]
        if spec.cvd_windows:
            base_cols += [f"cvd_{_format_window_name(w)}" for w in spec.cvd_windows]
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

    d = d[cols_to_keep].dropna().copy()
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
            update_kwargs["open_price"] = float(row[open_col])
            update_kwargs["high"] = float(row[high_col])
            update_kwargs["low"] = float(row[low_col])

        if has_volume_data:
            update_kwargs["volume"] = float(row[volume_col])
            update_kwargs["taker_buy_base"] = float(row[taker_buy_base_col])

        feats = transformer.update(**update_kwargs)

        out_rows.append(feats)

    out = pd.DataFrame(out_rows)
    out = out.sort_values([symbol_col, ts_col]).reset_index(drop=True)
    return out
