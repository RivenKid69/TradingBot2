"""
Диагностический тест для проверки корректности Twin Critics реализации.

Этот тест проверяет критическое утверждение:
"При сборе rollout и вычислении GAE используется только первый критик,
 а не min(Q1, Q2), что нарушает смысл Twin Critics."

Тест проверяет:
1. predict_values использует min(Q1, Q2) когда twin critics включены
2. Оба критика дают разные предсказания (независимые сети)
3. predict_values возвращает именно минимум из двух оценок
4. В rollout используются значения из predict_values (которые включают min)
5. Оба критика обучаются с одинаковыми targets
"""

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from unittest.mock import Mock, patch
from custom_policy_patch1 import CustomActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class TestTwinCriticsDiagnostic:
    """Диагностические тесты для Twin Critics."""

    @pytest.fixture
    def policy_with_twin_critics(self):
        """Создает policy с включенными Twin Critics."""
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            'hidden_dim': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 16,
                'huber_kappa': 1.0,
                'use_twin_critics': True,  # Enable twin critics
            }
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 0.001,
            arch_params=arch_params,
        )

        # Убеждаемся, что twin critics действительно включены
        assert policy._use_twin_critics is True
        assert policy.quantile_head is not None
        assert policy.quantile_head_2 is not None

        return policy

    def test_predict_values_uses_min_of_twin_critics(self, policy_with_twin_critics):
        """
        КРИТИЧЕСКИЙ ТЕСТ: Проверяет, что predict_values использует min(Q1, Q2).

        Это главный тест, который проверяет утверждение пользователя.
        Если predict_values использует только первый критик, тест провалится.
        """
        policy = policy_with_twin_critics
        policy.eval()

        batch_size = 8
        obs = torch.randn(batch_size, 10)
        lstm_states = policy.recurrent_initial_state
        episode_starts = torch.zeros(batch_size, dtype=torch.float32)

        with torch.no_grad():
            # Получаем значения через predict_values (используется в rollout)
            predicted_values = policy.predict_values(obs, lstm_states, episode_starts)

            # Получаем латентное представление критика вручную
            features = policy.extract_features(obs, policy.vf_features_extractor)

            if policy.lstm_critic is not None:
                latent_vf, _ = policy._process_sequence(
                    features, lstm_states.vf, episode_starts, policy.lstm_critic
                )
            elif policy.shared_lstm:
                latent_pi, _ = policy._process_sequence(
                    features, lstm_states.pi, episode_starts, policy.lstm_actor
                )
                latent_vf = latent_pi
            else:
                latent_vf = policy.critic(features)

            latent_vf = policy.mlp_extractor.forward_critic(latent_vf)

            # Получаем предсказания обоих критиков напрямую
            quantiles_1 = policy._get_value_logits(latent_vf)
            quantiles_2 = policy._get_value_logits_2(latent_vf)

            # Вычисляем values для каждого критика
            value_1 = quantiles_1.mean(dim=-1, keepdim=True)
            value_2 = quantiles_2.mean(dim=-1, keepdim=True)

            # Вычисляем ожидаемый минимум
            expected_min = torch.min(value_1, value_2)

        # КРИТИЧЕСКАЯ ПРОВЕРКА: predict_values должен возвращать min(Q1, Q2)
        assert torch.allclose(predicted_values, expected_min, atol=1e-6), \
            "predict_values НЕ использует min(Q1, Q2)! Это критическая ошибка!"

        # Дополнительная проверка: убеждаемся, что критики дают разные значения
        assert not torch.allclose(value_1, value_2, atol=1e-4), \
            "Оба критика дают одинаковые значения - возможно, используют одни параметры!"

        # Проверяем, что минимум действительно меньше или равен обоим значениям
        assert torch.all(predicted_values <= value_1 + 1e-6), \
            "Возвращенное значение больше первого критика!"
        assert torch.all(predicted_values <= value_2 + 1e-6), \
            "Возвращенное значение больше второго критика!"

        print(f"[OK] predict_values correctly uses min(Q1, Q2)")
        print(f"  Average value_1: {value_1.mean().item():.4f}")
        print(f"  Average value_2: {value_2.mean().item():.4f}")
        print(f"  Average min(Q1,Q2): {predicted_values.mean().item():.4f}")

    def test_critics_are_independent(self, policy_with_twin_critics):
        """
        Проверяет, что оба критика имеют независимые параметры.

        Если они не независимы, Twin Critics бесполезны.
        """
        policy = policy_with_twin_critics

        # Проверяем, что параметры различаются
        head1_weight = policy.quantile_head.linear.weight
        head2_weight = policy.quantile_head_2.linear.weight

        # Разные адреса памяти
        assert head1_weight.data_ptr() != head2_weight.data_ptr(), \
            "Критики используют одни и те же параметры!"

        # Инициализация должна быть случайной, поэтому значения разные
        assert not torch.allclose(head1_weight, head2_weight, atol=1e-4), \
            "Параметры критиков идентичны - возможно, ошибка инициализации!"

        print(f"[OK] Critics have independent parameters")

    def test_both_critics_produce_different_outputs(self, policy_with_twin_critics):
        """
        Проверяет, что оба критика дают разные выходы для одного входа.

        Это подтверждает независимость сетей.
        """
        policy = policy_with_twin_critics
        policy.eval()

        batch_size = 4
        latent_vf = torch.randn(batch_size, 32)

        with torch.no_grad():
            quantiles_1 = policy._get_value_logits(latent_vf)
            quantiles_2 = policy._get_value_logits_2(latent_vf)

        # Выходы должны отличаться
        assert not torch.allclose(quantiles_1, quantiles_2, atol=1e-4), \
            "Оба критика дают идентичные выходы!"

        # Вычисляем корреляцию между выходами
        flat_q1 = quantiles_1.flatten()
        flat_q2 = quantiles_2.flatten()
        correlation = torch.corrcoef(torch.stack([flat_q1, flat_q2]))[0, 1]

        print(f"[OK] Critics produce different outputs (correlation: {correlation.item():.4f})")

        # Корреляция не должна быть близка к 1 (полная корреляция)
        assert correlation.abs() < 0.99, \
            f"Слишком высокая корреляция между критиками: {correlation.item():.4f}"

    def test_min_is_computed_correctly(self, policy_with_twin_critics):
        """
        Проверяет корректность вычисления min(Q1, Q2) в _get_min_twin_values.
        """
        policy = policy_with_twin_critics
        policy.eval()

        batch_size = 4
        latent_vf = torch.randn(batch_size, 32)

        with torch.no_grad():
            # Получаем минимум через метод
            min_values = policy._get_min_twin_values(latent_vf)

            # Вычисляем вручную
            quantiles_1 = policy._get_value_logits(latent_vf)
            quantiles_2 = policy._get_value_logits_2(latent_vf)
            value_1 = quantiles_1.mean(dim=-1, keepdim=True)
            value_2 = quantiles_2.mean(dim=-1, keepdim=True)
            expected_min = torch.min(value_1, value_2)

        assert torch.allclose(min_values, expected_min, atol=1e-6), \
            "_get_min_twin_values вычисляет минимум неправильно!"

        # Проверяем, что минимум действительно минимален
        assert torch.all(min_values <= value_1 + 1e-6), \
            "Минимум больше первого значения!"
        assert torch.all(min_values <= value_2 + 1e-6), \
            "Минимум больше второго значения!"

        print(f"[OK] _get_min_twin_values correctly computes min(Q1, Q2)")

    def test_predict_values_with_disabled_twin_critics(self):
        """
        Проверяет, что predict_values использует только один критик когда twin critics отключены.
        """
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            'hidden_dim': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 16,
                'use_twin_critics': False,  # Explicitly disable
            }
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 0.001,
            arch_params=arch_params,
        )

        assert policy._use_twin_critics is False
        assert policy.quantile_head_2 is None

        batch_size = 4
        obs = torch.randn(batch_size, 10)
        lstm_states = policy.recurrent_initial_state
        episode_starts = torch.zeros(batch_size, dtype=torch.float32)

        with torch.no_grad():
            predicted_values = policy.predict_values(obs, lstm_states, episode_starts)

            # Должен использоваться только первый критик
            features = policy.extract_features(obs, policy.vf_features_extractor)
            if policy.lstm_critic is not None:
                latent_vf, _ = policy._process_sequence(
                    features, lstm_states.vf, episode_starts, policy.lstm_critic
                )
            elif policy.shared_lstm:
                latent_pi, _ = policy._process_sequence(
                    features, lstm_states.pi, episode_starts, policy.lstm_actor
                )
                latent_vf = latent_pi
            else:
                latent_vf = policy.critic(features)

            latent_vf = policy.mlp_extractor.forward_critic(latent_vf)
            quantiles_1 = policy._get_value_logits(latent_vf)
            expected_value = quantiles_1.mean(dim=-1, keepdim=True)

        assert torch.allclose(predicted_values, expected_value, atol=1e-6), \
            "predict_values работает неправильно когда twin critics отключены!"

        print(f"[OK] predict_values works correctly with single critic")

    def test_twin_critics_min_provides_pessimistic_estimate(self, policy_with_twin_critics):
        """
        Проверяет, что использование min(Q1, Q2) дает пессимистичную оценку.

        Это основное преимущество Twin Critics - уменьшение overestimation bias.
        """
        policy = policy_with_twin_critics
        policy.eval()

        batch_size = 100  # Большой batch для статистики
        obs = torch.randn(batch_size, 10)
        lstm_states = policy.recurrent_initial_state
        episode_starts = torch.zeros(batch_size, dtype=torch.float32)

        with torch.no_grad():
            # Получаем латентное представление
            features = policy.extract_features(obs, policy.vf_features_extractor)
            if policy.lstm_critic is not None:
                latent_vf, _ = policy._process_sequence(
                    features, lstm_states.vf, episode_starts, policy.lstm_critic
                )
            elif policy.shared_lstm:
                latent_pi, _ = policy._process_sequence(
                    features, lstm_states.pi, episode_starts, policy.lstm_actor
                )
                latent_vf = latent_pi
            else:
                latent_vf = policy.critic(features)

            latent_vf = policy.mlp_extractor.forward_critic(latent_vf)

            # Получаем оценки критиков
            quantiles_1 = policy._get_value_logits(latent_vf)
            quantiles_2 = policy._get_value_logits_2(latent_vf)
            value_1 = quantiles_1.mean(dim=-1, keepdim=True)
            value_2 = quantiles_2.mean(dim=-1, keepdim=True)

            # Минимум
            min_values = policy._get_min_twin_values(latent_vf)

            # Среднее (если бы мы не использовали min)
            avg_values = (value_1 + value_2) / 2

        # Статистика
        min_mean = min_values.mean().item()
        avg_mean = avg_values.mean().item()
        value1_mean = value_1.mean().item()
        value2_mean = value_2.mean().item()

        print(f"[OK] Twin Critics provide pessimistic estimate:")
        print(f"  Average Q1: {value1_mean:.4f}")
        print(f"  Average Q2: {value2_mean:.4f}")
        print(f"  Average (Q1+Q2)/2: {avg_mean:.4f}")
        print(f"  Average min(Q1,Q2): {min_mean:.4f}")
        print(f"  Difference (avg - min): {(avg_mean - min_mean):.4f}")

        # min должен быть меньше или равен среднему
        assert min_mean <= avg_mean + 1e-6, \
            "Минимум больше среднего - логическая ошибка!"

    def test_forward_method_caches_latent_vf(self, policy_with_twin_critics):
        """
        Проверяет, что forward() кэширует latent_vf для использования в loss.

        Это важно для эффективного вычисления loss для обоих критиков.
        """
        policy = policy_with_twin_critics
        policy.eval()

        batch_size = 4
        obs = torch.randn(batch_size, 10)
        lstm_states = policy.recurrent_initial_state
        episode_starts = torch.zeros(batch_size, dtype=torch.float32)

        with torch.no_grad():
            actions, values, log_prob, new_states = policy.forward(
                obs, lstm_states, episode_starts, deterministic=False
            )

            # Проверяем, что latent_vf был закэширован
            assert policy._last_latent_vf is not None, \
                "latent_vf не закэширован после forward()!"

            # Проверяем, что закэшированные quantiles существуют
            assert policy._last_value_quantiles is not None, \
                "value_quantiles не закэшированы после forward()!"

        print(f"[OK] forward() correctly caches latent_vf")


class TestTwinCriticsTrainingIntegration:
    """Тесты интеграции Twin Critics в процесс обучения."""

    def test_both_critics_receive_gradients(self):
        """
        Проверяет, что оба критика получают градиенты при обучении.

        Это критически важно - если только один критик обучается,
        второй будет бесполезен.
        """
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        arch_params = {
            'hidden_dim': 32,
            'critic': {
                'distributional': True,
                'num_quantiles': 16,
                'huber_kappa': 1.0,
                'use_twin_critics': True,
            }
        }

        policy = CustomActorCriticPolicy(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 0.001,
            arch_params=arch_params,
        )

        batch_size = 4
        latent_vf = torch.randn(batch_size, 32, requires_grad=True)

        # Forward pass через оба критика
        quantiles_1 = policy._get_value_logits(latent_vf)
        quantiles_2 = policy._get_value_logits_2(latent_vf)

        # Создаем dummy targets
        targets = torch.randn(batch_size, 1)

        # Вычисляем loss для обоих критиков
        loss_1 = (quantiles_1.mean(dim=-1, keepdim=True) - targets).pow(2).mean()
        loss_2 = (quantiles_2.mean(dim=-1, keepdim=True) - targets).pow(2).mean()

        # Суммарный loss
        total_loss = loss_1 + loss_2

        # Backward
        total_loss.backward()

        # КРИТИЧЕСКАЯ ПРОВЕРКА: Оба критика должны иметь градиенты
        assert policy.quantile_head.linear.weight.grad is not None, \
            "Первый критик не получил градиенты!"
        assert policy.quantile_head_2.linear.weight.grad is not None, \
            "Второй критик не получил градиенты!"

        # Градиенты должны быть ненулевыми
        grad1_norm = policy.quantile_head.linear.weight.grad.norm().item()
        grad2_norm = policy.quantile_head_2.linear.weight.grad.norm().item()

        assert grad1_norm > 1e-6, \
            "Градиенты первого критика равны нулю!"
        assert grad2_norm > 1e-6, \
            "Градиенты второго критика равны нулю!"

        print(f"[OK] Both critics receive gradients:")
        print(f"  Gradient norm Q1: {grad1_norm:.6f}")
        print(f"  Gradient norm Q2: {grad2_norm:.6f}")


def run_diagnostic():
    """Запускает все диагностические тесты."""
    print("=" * 80)
    print("ДИАГНОСТИКА TWIN CRITICS РЕАЛИЗАЦИИ")
    print("=" * 80)
    print()

    # Создаем test suite
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "-s"]))


if __name__ == "__main__":
    run_diagnostic()
