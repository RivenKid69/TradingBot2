"""
Тесты для проверки исправления forward-looking bias.

Этот модуль проверяет что:
1. Новый дефолт LeakConfig использует decision_delay_ms=8000
2. Низкие значения delay вызывают предупреждения
3. STRICT_LEAK_GUARD режим корректно блокирует низкие значения
4. Все конфигурации используют безопасные значения

Reference: de Prado (2018) "Advances in Financial Machine Learning", Chapter 7
"""

import os
import sys
import unittest
import warnings
import pandas as pd

sys.path.append(os.getcwd())

from leakguard import LeakGuard, LeakConfig


class TestForwardLookingBiasFix(unittest.TestCase):
    """Тесты для проверки исправления forward-looking bias."""

    def test_default_leakconfig_uses_safe_delay(self):
        """Проверка: дефолтный LeakConfig использует безопасное значение delay >= 8000."""
        cfg = LeakConfig()
        self.assertGreaterEqual(
            cfg.decision_delay_ms,
            8000,
            "Default LeakConfig должен использовать decision_delay_ms >= 8000",
        )

    def test_default_leakconfig_exact_value(self):
        """Проверка: дефолтное значение равно рекомендуемому минимуму 8000ms."""
        cfg = LeakConfig()
        self.assertEqual(
            cfg.decision_delay_ms, 8000, "Default decision_delay_ms должен быть 8000ms"
        )

    def test_zero_delay_triggers_warning(self):
        """Проверка: decision_delay_ms=0 вызывает критическое предупреждение."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lg = LeakGuard(LeakConfig(decision_delay_ms=0))

            # Должно быть хотя бы одно предупреждение
            self.assertGreater(len(w), 0, "decision_delay_ms=0 должен вызывать UserWarning")

            # Проверяем что предупреждение содержит нужный текст
            warning_messages = [str(warning.message) for warning in w]
            has_forward_bias_warning = any(
                "FORWARD-LOOKING BIAS" in msg or "decision_delay_ms=0" in msg
                for msg in warning_messages
            )
            self.assertTrue(
                has_forward_bias_warning,
                f"Предупреждение должно содержать информацию о forward-looking bias. Got: {warning_messages}",
            )

    def test_low_delay_triggers_warning(self):
        """Проверка: decision_delay_ms < 8000 вызывает предупреждение."""
        for low_delay in [100, 1000, 5000, 7999]:
            with self.subTest(delay=low_delay):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    lg = LeakGuard(LeakConfig(decision_delay_ms=low_delay))

                    # Должно быть предупреждение о низком delay
                    self.assertGreater(
                        len(w), 0, f"decision_delay_ms={low_delay} должен вызывать UserWarning"
                    )

    def test_safe_delay_no_warning(self):
        """Проверка: decision_delay_ms >= 8000 не вызывает предупреждения."""
        for safe_delay in [8000, 10000, 30000, 60000]:
            with self.subTest(delay=safe_delay):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    lg = LeakGuard(LeakConfig(decision_delay_ms=safe_delay))

                    # Не должно быть предупреждений о forward-looking bias
                    warning_messages = [str(warning.message) for warning in w]
                    has_bias_warning = any(
                        "FORWARD-LOOKING BIAS" in msg or "below recommended minimum" in msg
                        for msg in warning_messages
                    )
                    self.assertFalse(
                        has_bias_warning,
                        f"decision_delay_ms={safe_delay} не должен вызывать предупреждение о bias. Got: {warning_messages}",
                    )

    def test_strict_mode_blocks_zero_delay(self):
        """Проверка: STRICT_LEAK_GUARD=true блокирует decision_delay_ms=0."""
        old_env = os.getenv("STRICT_LEAK_GUARD")
        try:
            os.environ["STRICT_LEAK_GUARD"] = "true"

            with self.assertRaises(ValueError) as cm:
                lg = LeakGuard(LeakConfig(decision_delay_ms=0))

            self.assertIn(
                "decision_delay_ms=0",
                str(cm.exception),
                "ValueError должен упоминать decision_delay_ms=0",
            )
            self.assertIn(
                "STRICT mode", str(cm.exception), "ValueError должен упоминать STRICT mode"
            )
        finally:
            if old_env is None:
                os.environ.pop("STRICT_LEAK_GUARD", None)
            else:
                os.environ["STRICT_LEAK_GUARD"] = old_env

    def test_strict_mode_blocks_low_delay(self):
        """Проверка: STRICT_LEAK_GUARD=true блокирует любой delay < 8000."""
        old_env = os.getenv("STRICT_LEAK_GUARD")
        try:
            os.environ["STRICT_LEAK_GUARD"] = "true"

            for low_delay in [100, 1000, 5000, 7999]:
                with self.subTest(delay=low_delay):
                    with self.assertRaises(ValueError) as cm:
                        lg = LeakGuard(LeakConfig(decision_delay_ms=low_delay))

                    self.assertIn(
                        f"decision_delay_ms={low_delay}",
                        str(cm.exception),
                        f"ValueError должен упоминать decision_delay_ms={low_delay}",
                    )
        finally:
            if old_env is None:
                os.environ.pop("STRICT_LEAK_GUARD", None)
            else:
                os.environ["STRICT_LEAK_GUARD"] = old_env

    def test_strict_mode_allows_safe_delay(self):
        """Проверка: STRICT_LEAK_GUARD=true разрешает delay >= 8000."""
        old_env = os.getenv("STRICT_LEAK_GUARD")
        try:
            os.environ["STRICT_LEAK_GUARD"] = "true"

            for safe_delay in [8000, 10000, 30000]:
                with self.subTest(delay=safe_delay):
                    # Не должно быть исключения
                    lg = LeakGuard(LeakConfig(decision_delay_ms=safe_delay))
                    self.assertEqual(lg.cfg.decision_delay_ms, safe_delay)
        finally:
            if old_env is None:
                os.environ.pop("STRICT_LEAK_GUARD", None)
            else:
                os.environ["STRICT_LEAK_GUARD"] = old_env

    def test_negative_delay_raises_error(self):
        """Проверка: отрицательный decision_delay_ms вызывает ошибку."""
        with self.assertRaises(ValueError) as cm:
            lg = LeakGuard(LeakConfig(decision_delay_ms=-1000))

        self.assertIn(
            "must be >= 0",
            str(cm.exception),
            "Должна быть ошибка о том, что delay должен быть >= 0",
        )

    def test_attach_decision_time_with_default_config(self):
        """Проверка: attach_decision_time работает с дефолтным конфигом."""
        lg = LeakGuard()  # Использует дефолтный LeakConfig с delay=8000
        df = pd.DataFrame({"ts_ms": [1000, 2000, 3000]})
        result = lg.attach_decision_time(df, ts_col="ts_ms")

        self.assertIn("decision_ts", result.columns)
        # С дефолтным delay=8000
        self.assertEqual(result["decision_ts"].iloc[0], 9000)  # 1000 + 8000
        self.assertEqual(result["decision_ts"].iloc[1], 10000)  # 2000 + 8000
        self.assertEqual(result["decision_ts"].iloc[2], 11000)  # 3000 + 8000

    def test_backward_compatibility_explicit_zero(self):
        """Проверка: явное указание delay=0 все еще работает (с предупреждением)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Явно создаём конфиг с delay=0
            cfg = LeakConfig(decision_delay_ms=0)
            lg = LeakGuard(cfg)

            # Должно быть предупреждение
            self.assertGreater(len(w), 0)

            # Но конфиг должен работать
            df = pd.DataFrame({"ts_ms": [1000, 2000]})
            result = lg.attach_decision_time(df)
            self.assertEqual(result["decision_ts"].iloc[0], 1000)  # 1000 + 0


class TestTimingProfilesUseSafeDelays(unittest.TestCase):
    """Проверка что все timing profiles используют безопасные значения."""

    def test_timing_yaml_min_delays(self):
        """Проверка: все профили в timing.yaml используют delay >= 8000."""
        import yaml

        timing_path = "configs/timing.yaml"
        if not os.path.exists(timing_path):
            self.skipTest(f"{timing_path} not found")

        with open(timing_path, "r") as f:
            config = yaml.safe_load(f)

        profiles = config.get("profiles", {})
        for profile_name, profile_config in profiles.items():
            with self.subTest(profile=profile_name):
                delay = profile_config.get("decision_delay_ms", 0)
                self.assertGreaterEqual(
                    delay,
                    8000,
                    f"Profile '{profile_name}' должен использовать decision_delay_ms >= 8000, got {delay}",
                )

    def test_legacy_sim_yaml_uses_safe_delay(self):
        """Проверка: legacy_sim.yaml теперь использует безопасное значение."""
        import yaml

        legacy_path = "configs/legacy_sim.yaml"
        if not os.path.exists(legacy_path):
            self.skipTest(f"{legacy_path} not found")

        with open(legacy_path, "r") as f:
            config = yaml.safe_load(f)

        leakguard_config = config.get("leakguard", {})
        delay = leakguard_config.get("decision_delay_ms", 0)

        self.assertGreaterEqual(
            delay,
            8000,
            f"legacy_sim.yaml должен использовать decision_delay_ms >= 8000, got {delay}",
        )


if __name__ == "__main__":
    unittest.main()
