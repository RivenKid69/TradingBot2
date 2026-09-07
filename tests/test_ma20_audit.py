"""
ТЕСТ АУДИТА ПРИЗНАКА MA20 (21-ПЕРИОДНОЕ СКОЛЬЗЯЩЕЕ СРЕДНЕЕ)

Цель: Проверка корректности вычисления ma20 и отсутствия look-ahead bias

Проверяемые аспекты:
1. Корректность формулы SMA_21 = (P_t + P_{t-1} + ... + P_{t-20}) / 21
2. Отсутствие look-ahead bias (используются только закрытые бары)
3. Обработка NaN при недостаточном количестве баров
4. Флаг валидности в obs_builder
5. Численная стабильность для 21 элемента
"""

import unittest
from transformers import FeatureSpec, OnlineFeatureTransform


class TestMA20Audit(unittest.TestCase):
    """Тестирование корректности признака ma20"""

    def test_ma20_correct_formula(self):
        """
        Проверка: SMA_21 = (P_t + P_{t-1} + ... + P_{t-20}) / 21

        Ожидание: ma20 вычисляется ВКЛЮЧАЯ текущий закрытый бар
        """
        spec = FeatureSpec(
            lookbacks_prices=[5040],  # 21 бар для 4h таймфрейма
            bar_duration_minutes=240
        )
        transform = OnlineFeatureTransform(spec)

        # Подаем 21 бар с известными ценами
        prices = [100.0 + i for i in range(21)]  # 100, 101, 102, ..., 120

        for i, price in enumerate(prices):
            feats = transform.update(
                symbol="BTCUSDT",
                ts_ms=1000000 + i * 240 * 60 * 1000,  # 4h интервалы
                close=price
            )

        # После 21 бара должен появиться sma_5040
        self.assertIn("sma_5040", feats, "sma_5040 должен появиться после 21 бара")

        # Проверяем формулу: среднее последних 21 цен
        expected_sma21 = sum(prices) / 21.0  # (100+101+...+120) / 21 = 110.0
        actual_sma21 = feats["sma_5040"]

        self.assertAlmostEqual(
            actual_sma21,
            expected_sma21,
            places=6,
            msg=f"SMA_21 должен быть {expected_sma21}, получили {actual_sma21}"
        )

        print(f"✅ SMA_21 вычислен правильно: {actual_sma21:.6f}")

    def test_ma20_includes_current_bar(self):
        """
        Проверка: Текущий закрытый бар ВКЛЮЧАЕТСЯ в расчет SMA_21

        Важно: Это НЕ look-ahead bias, а правильная семантика end-of-bar trading
        """
        spec = FeatureSpec(
            lookbacks_prices=[5040],  # 21 бар
            bar_duration_minutes=240
        )
        transform = OnlineFeatureTransform(spec)

        # Подаем 20 баров по 100.0
        for i in range(20):
            transform.update(
                symbol="BTCUSDT",
                ts_ms=1000000 + i * 240 * 60 * 1000,
                close=100.0
            )

        # 21-й бар = 200.0 (сильное изменение)
        feats = transform.update(
            symbol="BTCUSDT",
            ts_ms=1000000 + 20 * 240 * 60 * 1000,
            close=200.0
        )

        # SMA_21 = (20*100.0 + 1*200.0) / 21 = 2200/21 ≈ 104.76
        expected = (20 * 100.0 + 200.0) / 21.0
        actual = feats["sma_5040"]

        self.assertAlmostEqual(
            actual,
            expected,
            places=6,
            msg=f"SMA_21 должен ВКЛЮЧАТЬ текущую цену 200.0"
        )

        # Проверка: если бы НЕ включалась текущая цена, было бы 100.0
        self.assertNotAlmostEqual(
            actual,
            100.0,
            places=1,
            msg="SMA_21 НЕ должен игнорировать текущую цену"
        )

        print(f"✅ Текущий бар правильно включен в SMA_21: {actual:.6f} vs expected {expected:.6f}")

    def test_ma20_no_lookahead_bias(self):
        """
        Критическая проверка: Отсутствие look-ahead bias

        Проверяем что SMA_21 на баре t использует только P_t, P_{t-1}, ..., P_{t-20}
        """
        spec = FeatureSpec(
            lookbacks_prices=[5040],
            bar_duration_minutes=240
        )
        transform = OnlineFeatureTransform(spec)

        # Подаем 22 бара: [100, 101, 102, ..., 121]
        for i in range(22):
            feats = transform.update(
                symbol="BTCUSDT",
                ts_ms=1000000 + i * 240 * 60 * 1000,
                close=100.0 + i
            )

            # После 21-го бара проверяем SMA_21
            if i == 20:  # 21-й бар (индекс 20)
                # SMA_21 на баре 20: среднее [100, 101, ..., 120]
                expected = sum(range(100, 121)) / 21.0  # 110.0
                self.assertAlmostEqual(
                    feats["sma_5040"],
                    expected,
                    places=6,
                    msg="SMA_21 на баре 20 должен использовать только бары 0-20"
                )

            elif i == 21:  # 22-й бар (индекс 21)
                # SMA_21 на баре 21: среднее [101, 102, ..., 121]
                expected = sum(range(101, 122)) / 21.0  # 111.0
                self.assertAlmostEqual(
                    feats["sma_5040"],
                    expected,
                    places=6,
                    msg="SMA_21 на баре 21 должен сдвинуться (rolling window)"
                )

                # КРИТИЧЕСКАЯ ПРОВЕРКА: SMA_21 НЕ должен быть 110.0
                self.assertNotAlmostEqual(
                    feats["sma_5040"],
                    110.0,
                    places=1,
                    msg="SMA_21 должен обновиться, НЕ использовать старые данные"
                )

        print("✅ Look-ahead bias ОТСУТСТВУЕТ - SMA_21 использует только прошлые + текущий бар")

    def test_ma20_nan_handling(self):
        """
        Проверка: Обработка NaN при недостаточном количестве баров
        """
        spec = FeatureSpec(
            lookbacks_prices=[5040],  # Требуется 21 бар
            bar_duration_minutes=240
        )
        transform = OnlineFeatureTransform(spec)

        # Подаем только 10 баров (меньше чем 21)
        for i in range(10):
            feats = transform.update(
                symbol="BTCUSDT",
                ts_ms=1000000 + i * 240 * 60 * 1000,
                close=100.0
            )

        # sma_5040 НЕ должен появиться (недостаточно данных)
        self.assertNotIn(
            "sma_5040",
            feats,
            msg="sma_5040 НЕ должен появиться при < 21 баров"
        )

        # Подаем еще 11 баров (итого 21)
        for i in range(10, 21):
            feats = transform.update(
                symbol="BTCUSDT",
                ts_ms=1000000 + i * 240 * 60 * 1000,
                close=100.0
            )

        # Теперь sma_5040 должен появиться
        self.assertIn(
            "sma_5040",
            feats,
            msg="sma_5040 должен появиться после 21 бара"
        )

        self.assertAlmostEqual(
            feats["sma_5040"],
            100.0,
            places=6,
            msg="SMA_21 всех 100.0 должен быть 100.0"
        )

        print("✅ NaN handling корректен - sma_5040 появляется только после 21 бара")

    def test_ma20_numerical_stability(self):
        """
        Проверка: Численная стабильность для 21 элемента

        Примечание: Для 21 элемента простое суммирование обычно стабильно,
        но проверим на крайних значениях
        """
        spec = FeatureSpec(
            lookbacks_prices=[5040],
            bar_duration_minutes=240
        )
        transform = OnlineFeatureTransform(spec)

        # Тест 1: Очень большие числа
        large_price = 1e8  # 100 миллионов
        for i in range(21):
            feats = transform.update(
                symbol="BTCUSDT",
                ts_ms=1000000 + i * 240 * 60 * 1000,
                close=large_price + i
            )

        expected = (21 * large_price + sum(range(21))) / 21.0
        self.assertAlmostEqual(
            feats["sma_5040"],
            expected,
            places=3,  # Допускаем небольшую погрешность для больших чисел
            msg=f"SMA_21 должен работать с большими числами"
        )

        # Тест 2: Очень маленькие числа
        transform2 = OnlineFeatureTransform(spec)
        small_price = 1e-6  # микроскопические значения
        for i in range(21):
            feats2 = transform2.update(
                symbol="BTCUSDT",
                ts_ms=2000000 + i * 240 * 60 * 1000,
                close=small_price * (i + 1)
            )

        expected2 = small_price * sum(range(1, 22)) / 21.0
        self.assertAlmostEqual(
            feats2["sma_5040"],
            expected2,
            delta=1e-9,  # Очень малая погрешность
            msg="SMA_21 должен работать с малыми числами"
        )

        print("✅ Численная стабильность достаточна для 21 элемента")

    def test_ma20_lag_property(self):
        """
        Проверка: Свойство запаздывания (lag) для SMA_21

        Теоретический lag ≈ (N-1)/2 = (21-1)/2 = 10 баров
        """
        spec = FeatureSpec(
            lookbacks_prices=[5040],
            bar_duration_minutes=240
        )
        transform = OnlineFeatureTransform(spec)

        for i in range(60):
            price = 100.0 if i < 30 else 200.0
            feats = transform.update(
                symbol="BTCUSDT",
                ts_ms=1000000 + i * 240 * 60 * 1000,
                close=price
            )

            # Bar 30: SMA starts moving but still lags far behind the jump
            if i == 30:
                # SMA_21 = (20*100.0 + 1*200.0) / 21 ≈ 104.76
                self.assertLess(
                    feats["sma_5040"],
                    150.0,
                    msg="SMA_21 should lag behind the sharp move at bar 30"
                )

            # Bar 49: one old price remains in the window -> lag persists
            elif i == 49:
                # SMA_21 = (1*100.0 + 20*200.0) / 21 ≈ 195.24
                self.assertGreater(
                    feats["sma_5040"],
                    180.0,
                    msg="SMA_21 should have mostly caught up after 20 bars"
                )
                self.assertLess(
                    feats["sma_5040"],
                    200.0,
                    msg="SMA_21 should retain some lag with one old value remaining"
                )

            # Bar 50: window fully refreshed with new level -> SMA reaches 200
            elif i == 50:
                self.assertAlmostEqual(
                    feats["sma_5040"],
                    200.0,
                    places=6,
                    msg="SMA_21 should reach the new price once the full window is updated"
                )

        print("Lag property for SMA_21 behaves as expected (~10 bars)")

if __name__ == "__main__":
    unittest.main(verbosity=2)
