"""
Тесты для проверки исправлений интеграции 11 признаков для 4h интервала.

Проверяет:
1. КРИТИЧЕСКАЯ #1: Дефолтные параметры в transformers.py для 4h интервала
2. КРИТИЧЕСКАЯ #2: Соответствие имен признаков между mediator.py и transformers.py
3. КРИТИЧЕСКАЯ #3: Окна GARCH достаточно велики для стабильной оценки
4. MAJOR #1: Формула Parkinson с проверкой минимального процента валидных баров
5. MINOR #2: Clamping для taker_buy_ratio
"""
from __future__ import annotations

import math
import unittest

from transformers import FeatureSpec, OnlineFeatureTransformer, _format_window_name


class TestCritical1_DefaultParameters(unittest.TestCase):
    """Проверка КРИТИЧЕСКАЯ #1: Дефолтные параметры для 4h интервала."""

    def test_default_lookbacks_prices_for_4h(self):
        """Проверяет что дефолтные lookbacks_prices подходят для 4h интервала."""
        # CRITICAL FIX #1: Указываем bar_duration_minutes=240 для 4h интервала
        spec = FeatureSpec(lookbacks_prices=[], bar_duration_minutes=240)

        # Для 4h интервала (1 бар = 240 минут):
        # Исходные значения: 240, 720, 1440, 12000 минут
        # После конвертации: 1, 3, 6, 50 баров
        expected_bars = [1, 3, 6, 50]
        self.assertEqual(spec.lookbacks_prices, expected_bars,
            f"Дефолтные lookbacks_prices должны быть {expected_bars} баров для 4h, "
            f"но получили {spec.lookbacks_prices}"
        )

        # Проверяем что исходные значения в минутах сохранены
        expected_minutes = [240, 720, 1440, 12000]
        self.assertEqual(spec._lookbacks_prices_minutes, expected_minutes,
            f"Исходные lookbacks_prices должны быть {expected_minutes} минут для 4h, "
            f"но получили {spec._lookbacks_prices_minutes}"
        )

    def test_default_garch_windows_for_4h(self):
        """Проверяет что дефолтные GARCH окна достаточно велики для 4h интервала."""
        # CRITICAL FIX #1: Указываем bar_duration_minutes=240 для 4h интервала
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        # Для 4h интервала (1 бар = 240 минут):
        # Исходные: 10080, 20160, 43200 минут
        # После конвертации: 42, 84, 180 баров
        # КРИТИЧНО: все окна должны быть >= 50 баров (минимум для GARCH)
        expected_bars = [42, 84, 180]
        self.assertEqual(spec.garch_windows, expected_bars,
            f"Дефолтные garch_windows должны быть {expected_bars} баров для 4h, "
            f"но получили {spec.garch_windows}"
        )

        # Проверяем что все окна >= 50 баров
        for bars in spec.garch_windows:
            self.assertGreaterEqual(bars, 50,
                f"GARCH окно {bars} баров, но требуется минимум 50 баров!"
            )

    def test_default_yang_zhang_windows_for_4h(self):
        """Проверяет что дефолтные Yang-Zhang окна правильные для 4h интервала."""
        # CRITICAL FIX #1: Указываем bar_duration_minutes=240 для 4h интервала
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        # Для 4h интервала: 2880, 10080, 43200 минут = 12, 42, 180 баров
        expected = [12, 42, 180]
        self.assertEqual(spec.yang_zhang_windows, expected,
            f"Дефолтные yang_zhang_windows должны быть {expected} баров для 4h, "
            f"но получили {spec.yang_zhang_windows}"
        )

    def test_default_parkinson_windows_for_4h(self):
        """Проверяет что дефолтные Parkinson окна правильные для 4h интервала."""
        # CRITICAL FIX #1: Указываем bar_duration_minutes=240 для 4h интервала
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        # Для 4h интервала: 2880, 10080 минут = 12, 42 бара
        expected = [12, 42]
        self.assertEqual(spec.parkinson_windows, expected,
            f"Дефолтные parkinson_windows должны быть {expected} баров для 4h, "
            f"но получили {spec.parkinson_windows}"
        )

    def test_default_taker_buy_ratio_windows_for_4h(self):
        """Проверяет что дефолтные Taker Buy Ratio SMA окна правильные для 4h интервала."""
        # CRITICAL FIX #1: Указываем bar_duration_minutes=240 для 4h интервала
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        # Для 4h интервала: 480, 960, 1440 минут = 2, 4, 6 баров
        expected = [2, 4, 6]
        self.assertEqual(spec.taker_buy_ratio_windows, expected,
            f"Дефолтные taker_buy_ratio_windows должны быть {expected} баров для 4h, "
            f"но получили {spec.taker_buy_ratio_windows}"
        )

    def test_default_taker_buy_ratio_momentum_for_4h(self):
        """Проверяет что дефолтные Taker Buy Ratio Momentum окна правильные для 4h интервала."""
        # CRITICAL FIX #1: Указываем bar_duration_minutes=240 для 4h интервала
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        # Для 4h интервала: 240, 480, 720 минут = 1, 2, 3 бара
        expected = [1, 2, 3]
        self.assertEqual(spec.taker_buy_ratio_momentum, expected,
            f"Дефолтные taker_buy_ratio_momentum должны быть {expected} баров для 4h, "
            f"но получили {spec.taker_buy_ratio_momentum}"
        )


class TestCritical2_FeatureNamesMatch(unittest.TestCase):
    """Проверка КРИТИЧЕСКАЯ #2: Соответствие имен признаков."""

    def test_generated_feature_names_for_4h(self):
        """Проверяет что transformers.py генерирует правильные имена для 4h интервала."""
        # CRITICAL FIX #1: Указываем bar_duration_minutes=240 для 4h интервала
        spec = FeatureSpec(lookbacks_prices=[240, 720, 1440, 12000], bar_duration_minutes=240)
        transformer = OnlineFeatureTransformer(spec)

        # Генерируем тестовые признаки
        feats = transformer.update(
            symbol="BTC/USDT",
            ts_ms=1000000000,
            close=50000.0,
            open_price=49900.0,
            high=50100.0,
            low=49800.0,
            volume=100.0,
            taker_buy_base=60.0,
        )

        # Проверяем имена returns для 4h интервала
        # CRITICAL FIX #3: Исправлено имя ret_50 → ret_200h (12000 минут = 200 часов)
        expected_return_names = ["ret_4h", "ret_12h", "ret_24h", "ret_200h"]
        for name in expected_return_names:
            self.assertIn(name, feats, f"Ожидали признак {name}, но его нет в {list(feats.keys())}")

        # Проверяем имена GARCH для 4h интервала
        # КРИТИЧНО: GARCH требует минимум 50 баров (12000 минут = 200h)
        expected_garch_names = ["garch_200h", "garch_14d", "garch_30d"]
        for name in expected_garch_names:
            self.assertIn(name, feats, f"Ожидали признак {name}, но его нет в {list(feats.keys())}")

        # Проверяем имена Yang-Zhang для 4h интервала
        expected_yz_names = ["yang_zhang_48h", "yang_zhang_7d", "yang_zhang_30d"]
        for name in expected_yz_names:
            self.assertIn(name, feats, f"Ожидали признак {name}, но его нет в {list(feats.keys())}")

        # Проверяем имена Parkinson для 4h интервала
        expected_pk_names = ["parkinson_48h", "parkinson_7d"]
        for name in expected_pk_names:
            self.assertIn(name, feats, f"Ожидали признак {name}, но его нет в {list(feats.keys())}")

        # Проверяем имена Taker Buy Ratio SMA для 4h интервала
        expected_tbr_sma_names = ["taker_buy_ratio_sma_8h", "taker_buy_ratio_sma_16h", "taker_buy_ratio_sma_24h"]
        for name in expected_tbr_sma_names:
            self.assertIn(name, feats, f"Ожидали признак {name}, но его нет в {list(feats.keys())}")

        # Проверяем имена Taker Buy Ratio Momentum для 4h интервала
        expected_tbr_mom_names = ["taker_buy_ratio_momentum_4h", "taker_buy_ratio_momentum_8h", "taker_buy_ratio_momentum_12h"]
        for name in expected_tbr_mom_names:
            self.assertIn(name, feats, f"Ожидали признак {name}, но его нет в {list(feats.keys())}")

        # Проверяем наличие sma_50 (критично для mediator.py:1032)
        self.assertIn("sma_50", feats, "Ожидали признак sma_50, но его нет в features")

    def test_format_window_name_for_4h(self):
        """Проверяет функцию _format_window_name для 4h интервала."""
        # Для GARCH (длинные окна >= 7 дней)
        self.assertEqual(_format_window_name(10080), "7d", "10080 минут должно быть 7d")
        self.assertEqual(_format_window_name(20160), "14d", "20160 минут должно быть 14d")
        self.assertEqual(_format_window_name(43200), "30d", "43200 минут должно быть 30d")

        # Для обычных признаков (часы)
        self.assertEqual(_format_window_name(240), "4h", "240 минут должно быть 4h")
        self.assertEqual(_format_window_name(480), "8h", "480 минут должно быть 8h")
        self.assertEqual(_format_window_name(720), "12h", "720 минут должно быть 12h")
        self.assertEqual(_format_window_name(1440), "24h", "1440 минут должно быть 24h")
        self.assertEqual(_format_window_name(2880), "48h", "2880 минут должно быть 48h")
        self.assertEqual(_format_window_name(12000), "50")  # 12000 минут = 200 часов (не кратно дню)


class TestMajor1_ParkinsonFormula(unittest.TestCase):
    """Проверка MAJOR #1: Формула Parkinson с проверкой минимального процента валидных баров."""

    def test_parkinson_requires_80_percent_valid_bars(self):
        """Проверяет что Parkinson требует минимум 80% валидных баров."""
        from transformers import calculate_parkinson_volatility

        # Окно n=10, нужно минимум max(2, int(0.8*10)) = 8 валидных баров

        # Случай 1: Все бары валидны (10/10 = 100%) → должно работать
        valid_bars = [
            {"high": 101.0, "low": 99.0} for _ in range(10)
        ]
        result = calculate_parkinson_volatility(valid_bars, 10)
        self.assertIsNotNone(result, "Должно работать с 100% валидных баров")

        # Случай 2: 8 валидных баров (8/10 = 80%) → должно работать
        mixed_bars = [
            {"high": 101.0, "low": 99.0} for _ in range(8)
        ] + [
            {"high": 0.0, "low": 0.0},  # Невалидный
            {"high": 0.0, "low": 0.0},  # Невалидный
        ]
        result = calculate_parkinson_volatility(mixed_bars, 10)
        self.assertIsNotNone(result, "Должно работать с 80% валидных баров")

        # Случай 3: 7 валидных баров (7/10 = 70%) → должно вернуть None
        few_valid_bars = [
            {"high": 101.0, "low": 99.0} for _ in range(7)
        ] + [
            {"high": 0.0, "low": 0.0} for _ in range(3)
        ]
        result = calculate_parkinson_volatility(few_valid_bars, 10)
        self.assertIsNone(result, "Не должно работать с 70% валидных баров")


class TestMinor2_TakerBuyRatioClamping(unittest.TestCase):
    """Проверка MINOR #2: Clamping для taker_buy_ratio."""

    def test_taker_buy_ratio_clamped_to_0_1(self):
        """Проверяет что taker_buy_ratio ограничен диапазоном [0.0, 1.0]."""
        # CRITICAL FIX #1: Указываем bar_duration_minutes=240 для 4h интервала
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        transformer = OnlineFeatureTransformer(spec)

        # Случай 1: Нормальное значение (taker_buy_base = 60, volume = 100) → 0.6
        feats1 = transformer.update(
            symbol="BTC/USDT",
            ts_ms=1000000000,
            close=50000.0,
            volume=100.0,
            taker_buy_base=60.0,
        )
        self.assertGreaterEqual(feats1["taker_buy_ratio"], 0.0, "taker_buy_ratio должен быть >= 0")
        self.assertLessEqual(feats1["taker_buy_ratio"], 1.0, "taker_buy_ratio должен быть <= 1")
        self.assertAlmostEqual(feats1["taker_buy_ratio"], 0.6, places=6, msg="Ожидали 0.6")

        # Случай 2: Аномалия - taker_buy_base > volume (110 > 100) → должен быть clamped к 1.0
        feats2 = transformer.update(
            symbol="BTC/USDT",
            ts_ms=1000000001,
            close=50000.0,
            volume=100.0,
            taker_buy_base=110.0,  # Аномалия!
        )
        self.assertEqual(feats2["taker_buy_ratio"], 1.0,
            f"taker_buy_ratio должен быть clamped к 1.0, но получили {feats2['taker_buy_ratio']}"
        )


class TestIntegrationEnd2End(unittest.TestCase):
    """Интеграционный тест проверки всех исправлений вместе."""

    def test_full_pipeline_with_4h_defaults(self):
        """Проверяет полный пайплайн с дефолтными параметрами для 4h интервала."""
        # CRITICAL FIX #1: Указываем bar_duration_minutes=240 для 4h интервала
        spec = FeatureSpec(lookbacks_prices=[], bar_duration_minutes=240)  # Используем дефолты
        transformer = OnlineFeatureTransformer(spec)

        # Генерируем достаточно данных для GARCH (минимум 50 баров для 7d окна)
        # 7d = 42 бара для 4h, но мы генерируем 50+ для надежности
        base_price = 50000.0

        for i in range(200):  # Генерируем 200 баров
            feats = transformer.update(
                symbol="BTC/USDT",
                ts_ms=1000000000 + i * 1000,
                close=base_price + i * 10,
                open_price=base_price + i * 10 - 5,
                high=base_price + i * 10 + 10,
                low=base_price + i * 10 - 10,
                volume=100.0 + i,
                taker_buy_base=60.0 + i * 0.1,
            )

        # Проверяем что все ожидаемые признаки присутствуют
        # CRITICAL FIX #3: Исправлены имена согласно дефолтным значениям lookbacks_prices=[240,720,1440,12000]
        # 12000 минут = 200 часов, форматируется как "200h"
        expected_features = [
            "ret_4h", "ret_12h", "ret_24h", "ret_200h",
            "sma_240", "sma_720", "sma_1440", "sma_12000",
            "garch_200h", "garch_14d", "garch_30d",
            "yang_zhang_48h", "yang_zhang_7d", "yang_zhang_30d",
            "parkinson_48h", "parkinson_7d",
            "taker_buy_ratio",
            "taker_buy_ratio_sma_8h", "taker_buy_ratio_sma_16h", "taker_buy_ratio_sma_24h",
            "taker_buy_ratio_momentum_4h", "taker_buy_ratio_momentum_8h", "taker_buy_ratio_momentum_12h",
            "cvd_24h", "cvd_7d",
            "rsi",
        ]

        missing_features = [f for f in expected_features if f not in feats]
        self.assertEqual(missing_features, [], f"Отсутствуют признаки: {missing_features}")

        # Проверяем что GARCH признаки не NaN (достаточно данных)
        for garch_name in ["garch_200h", "garch_14d", "garch_30d"]:
            self.assertFalse(math.isnan(feats[garch_name]),
                f"GARCH признак {garch_name} не должен быть NaN при наличии 200 баров данных"
            )

        # Проверяем что Yang-Zhang признаки не NaN
        for yz_name in ["yang_zhang_48h", "yang_zhang_7d", "yang_zhang_30d"]:
            self.assertFalse(math.isnan(feats[yz_name]),
                f"Yang-Zhang признак {yz_name} не должен быть NaN при наличии 200 баров данных"
            )

        # Проверяем что Parkinson признаки не NaN
        for pk_name in ["parkinson_48h", "parkinson_7d"]:
            self.assertFalse(math.isnan(feats[pk_name]),
                f"Parkinson признак {pk_name} не должен быть NaN при наличии 200 баров данных"
            )

        # Проверяем что taker_buy_ratio в допустимых пределах
        self.assertGreaterEqual(feats["taker_buy_ratio"], 0.0, "taker_buy_ratio должен быть >= 0")
        self.assertLessEqual(feats["taker_buy_ratio"], 1.0, "taker_buy_ratio должен быть <= 1")


if __name__ == "__main__":
    # Запускаем тесты
    unittest.main(verbosity=2)
