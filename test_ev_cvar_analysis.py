"""
Тестовый скрипт для анализа вычислений EV и CVaR.
Проверяет корректность математики в distributional_ppo.py
"""

import math
import torch
import numpy as np

# Имитация функции _cvar_from_quantiles из distributional_ppo.py
def _cvar_from_quantiles_original(predicted_quantiles: torch.Tensor, alpha: float) -> torch.Tensor:
    """Оригинальная реализация из distributional_ppo.py"""
    num_quantiles = predicted_quantiles.shape[1]
    if num_quantiles == 0:
        return predicted_quantiles.new_zeros(predicted_quantiles.shape[0])

    mass = 1.0 / float(num_quantiles)
    k_float = alpha * num_quantiles
    full_mass = int(min(num_quantiles, math.floor(k_float)))
    frac = float(k_float - full_mass)

    device = predicted_quantiles.device
    dtype = predicted_quantiles.dtype

    tail_sum = predicted_quantiles.new_zeros(predicted_quantiles.shape[0], dtype=dtype, device=device)
    if full_mass > 0:
        tail_sum = predicted_quantiles[:, :full_mass].sum(dim=1)

    partial = predicted_quantiles.new_zeros(predicted_quantiles.shape[0], dtype=dtype, device=device)
    if frac > 1e-8 and full_mass < num_quantiles:
        partial = predicted_quantiles[:, full_mass] * frac

    expectation = mass * (tail_sum + partial)
    tail_mass = max(alpha, mass * (full_mass + frac))
    return expectation / tail_mass


def test_ev_from_uniform_quantiles():
    """
    Тест: EV из равномерных квантилей через простое среднее

    Для равномерно распределенных квантилей (midpoints):
    τ_i = (i + 0.5) / N для i = 0, ..., N-1

    EV ≈ (1/N) * Σ Q(τ_i)  - это корректное приближение интеграла
    """
    print("\n=== TEST: EV from uniform quantiles ===")

    # Создаем синтетическое распределение: нормальное N(0, 1)
    # Квантили для N=51: τ = [0.01, 0.03, 0.05, ..., 0.97, 0.99]
    num_quantiles = 51
    taus = torch.linspace(0.0, 1.0, steps=num_quantiles + 1)
    midpoints = 0.5 * (taus[:-1] + taus[1:])

    # Для стандартного нормального распределения используем обратную функцию
    from scipy.stats import norm
    quantile_values = torch.tensor([norm.ppf(tau.item()) for tau in midpoints], dtype=torch.float32)
    quantile_values = quantile_values.unsqueeze(0)  # batch_size = 1

    # EV через простое среднее
    ev_mean = quantile_values.mean(dim=1).item()

    # Истинное EV для N(0,1) = 0
    true_ev = 0.0

    error = abs(ev_mean - true_ev)
    print(f"Num quantiles: {num_quantiles}")
    print(f"EV via mean: {ev_mean:.6f}")
    print(f"True EV: {true_ev:.6f}")
    print(f"Error: {error:.6f}")
    print(f"Status: {'✓ PASS' if error < 0.01 else '✗ FAIL'}")

    return error < 0.01


def test_cvar_from_quantiles_issue():
    """
    Тест: Проверка корректности CVaR из квантилей (midpoints)

    ПРОБЛЕМА: Квантили представляют midpoints интервалов, а не границы.
    Для α < 1/(2*N) интерполяция может быть некорректной.
    """
    print("\n=== TEST: CVaR from quantiles (potential issue) ===")

    # 5 квантилей: τ = [0.1, 0.3, 0.5, 0.7, 0.9]
    num_quantiles = 5
    alpha = 0.05  # CVaR для худших 5%

    # Синтетические квантили для монотонного распределения
    # Пусть Q(τ) = -2 + 4*τ (линейная функция от -2 до 2)
    taus = torch.linspace(0.0, 1.0, steps=num_quantiles + 1)
    midpoints = 0.5 * (taus[:-1] + taus[1:])
    quantile_values = torch.tensor([-2.0 + 4.0 * tau.item() for tau in midpoints], dtype=torch.float32)
    quantile_values = quantile_values.unsqueeze(0)

    print(f"Quantile levels (τ): {midpoints.tolist()}")
    print(f"Quantile values Q(τ): {quantile_values.tolist()}")

    # Вычисляем CVaR через оригинальную функцию
    cvar_computed = _cvar_from_quantiles_original(quantile_values, alpha).item()

    # Истинный CVaR для линейной функции Q(τ) = -2 + 4*τ:
    # CVaR_α = (1/α) * ∫₀^α Q(τ) dτ = (1/α) * ∫₀^α (-2 + 4τ) dτ
    # = (1/α) * [-2τ + 2τ²]₀^α = (1/α) * (-2α + 2α²)
    # = -2 + 2α
    true_cvar = -2.0 + 2.0 * alpha

    # Что возвращает функция?
    # α * num_quantiles = 0.05 * 5 = 0.25
    # full_mass = floor(0.25) = 0
    # frac = 0.25
    # expectation = (1/5) * (0 + Q[0] * 0.25) = 0.2 * (-1.6) * 0.25 = -0.08
    # tail_mass = max(0.05, 0.2 * 0.25) = max(0.05, 0.05) = 0.05
    # cvar = -0.08 / 0.05 = -1.6

    # Но Q[0] = -1.6 соответствует τ=0.1, а не τ=0.05!
    # Для α=0.05 нужно интерполировать между τ=0 и τ=0.1

    error = abs(cvar_computed - true_cvar)
    print(f"\nCVaR α={alpha}")
    print(f"CVaR computed: {cvar_computed:.6f}")
    print(f"CVaR true: {true_cvar:.6f}")
    print(f"Error: {error:.6f}")

    # Проверим, какое значение получилось
    # Функция вернет Q[0] = -1.6, а true_cvar = -1.9
    expected_computed = quantile_values[0, 0].item()  # -1.6
    print(f"Expected (Q[0]): {expected_computed:.6f}")
    print(f"Match: {abs(cvar_computed - expected_computed) < 1e-6}")

    print(f"\nStatus: {'✓ PASS' if error < 0.1 else '✗ FAIL (significant bias)'}")

    return error, true_cvar, cvar_computed


def test_cvar_with_larger_alpha():
    """
    Тест: CVaR с большим α (например, α=0.5)

    Для больших α ошибка должна быть меньше.
    """
    print("\n=== TEST: CVaR with α=0.5 ===")

    num_quantiles = 5
    alpha = 0.5

    taus = torch.linspace(0.0, 1.0, steps=num_quantiles + 1)
    midpoints = 0.5 * (taus[:-1] + taus[1:])
    quantile_values = torch.tensor([-2.0 + 4.0 * tau.item() for tau in midpoints], dtype=torch.float32)
    quantile_values = quantile_values.unsqueeze(0)

    cvar_computed = _cvar_from_quantiles_original(quantile_values, alpha).item()
    true_cvar = -2.0 + 2.0 * alpha

    error = abs(cvar_computed - true_cvar)
    print(f"CVaR α={alpha}")
    print(f"CVaR computed: {cvar_computed:.6f}")
    print(f"CVaR true: {true_cvar:.6f}")
    print(f"Error: {error:.6f}")
    print(f"Status: {'✓ PASS' if error < 0.1 else '✗ FAIL'}")

    return error < 0.1


def test_quantile_mean_as_ev():
    """
    Тест: Проверка, что quantiles.mean() корректно вычисляет EV
    """
    print("\n=== TEST: Quantiles mean as EV ===")

    # Для экспоненциального распределения с λ=1: Q(τ) = -ln(1-τ)
    num_quantiles = 51
    taus = torch.linspace(0.0, 1.0, steps=num_quantiles + 1)
    midpoints = 0.5 * (taus[:-1] + taus[1:])

    quantile_values = torch.tensor(
        [-math.log(1 - tau.item() + 1e-10) for tau in midpoints],
        dtype=torch.float32
    ).unsqueeze(0)

    ev_computed = quantile_values.mean(dim=1).item()
    true_ev = 1.0  # EV экспоненциального распределения с λ=1

    error = abs(ev_computed - true_ev)
    print(f"EV computed: {ev_computed:.6f}")
    print(f"EV true: {true_ev:.6f}")
    print(f"Error: {error:.6f}")
    print(f"Status: {'✓ PASS' if error < 0.05 else '✗ FAIL'}")

    return error < 0.05


if __name__ == "__main__":
    print("="*70)
    print("АНАЛИЗ ВЫЧИСЛЕНИЙ EV И CVaR")
    print("="*70)

    results = {}

    # Тест 1: EV из квантилей
    results['ev_uniform'] = test_ev_from_uniform_quantiles()

    # Тест 2: CVaR из квантилей (потенциальная проблема)
    error, true_val, computed_val = test_cvar_from_quantiles_issue()
    results['cvar_small_alpha'] = (error, true_val, computed_val)

    # Тест 3: CVaR с большим α
    results['cvar_large_alpha'] = test_cvar_with_larger_alpha()

    # Тест 4: EV через mean
    results['ev_mean'] = test_quantile_mean_as_ev()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n1. EV calculation via quantiles.mean():")
    print(f"   ✓ CORRECT - для равномерных квантилей (midpoints)")

    print("\n2. CVaR calculation from quantiles:")
    error, true_cvar, computed_cvar = results['cvar_small_alpha']
    bias_pct = (computed_cvar - true_cvar) / abs(true_cvar) * 100
    print(f"   ⚠ POTENTIAL ISSUE for small α (α < 1/N)")
    print(f"   - Bias: {bias_pct:.2f}% for α=0.05 with N=5")
    print(f"   - Reason: квантили - это midpoints, не boundaries")
    print(f"   - Impact: CVaR менее консервативен (ближе к нулю)")

    print("\n3. Рекомендации:")
    print("   - Для EV: текущая реализация корректна")
    print("   - Для CVaR: рассмотреть интерполяцию для α < 1/(2*N)")
    print("   - Или использовать больше квантилей (N > 20*1/α)")
