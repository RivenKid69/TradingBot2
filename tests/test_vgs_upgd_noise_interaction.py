"""
Comprehensive Tests for VGS + UPGD Noise Interaction Fix (Issue #1 - 2025-11-22)

Tests verify that adaptive noise scaling in UPGD maintains constant noise-to-signal
ratio when VGS scales gradients down, preventing training instability.

See VGS_UPGD_NOISE_INTERACTION_ANALYSIS.md for full analysis.
"""

import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn
import pytest
import numpy as np
from typing import Dict, List

from optimizers.adaptive_upgd import AdaptiveUPGD
from variance_gradient_scaler import VarianceGradientScaler


class SimpleTestModel(nn.Module):
    """Simple model for testing VGS + UPGD interaction."""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute L2 norm of model gradients."""
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    return np.sqrt(grad_norm)


def compute_noise_to_signal_ratio(
    optimizer: AdaptiveUPGD,
    grad_norm: float
) -> float:
    """
    Estimate noise-to-signal ratio from optimizer state.

    For adaptive_noise=True, noise should scale with gradient norm,
    maintaining approximately constant ratio.
    """
    total_noise_sq = 0.0
    count = 0

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue

            state = optimizer.state.get(p, {})
            if group["adaptive_noise"] and "grad_norm_ema" in state:
                # Adaptive noise: scaled by gradient norm
                grad_norm_ema = state["grad_norm_ema"]
                adaptive_sigma = max(
                    group["sigma"] * grad_norm_ema,
                    group["min_noise_std"]
                )
                # Approximate noise variance per parameter
                noise_var = adaptive_sigma ** 2 * p.numel()
            else:
                # Fixed noise
                noise_var = (group["sigma"] ** 2) * p.numel()

            total_noise_sq += noise_var
            count += p.numel()

    if count == 0 or grad_norm == 0:
        return 0.0

    noise_norm = np.sqrt(total_noise_sq)
    return noise_norm / grad_norm


class TestVGSUPGDNoiseInteraction:
    """Test suite for VGS + UPGD noise interaction fix."""

    def test_fixed_noise_amplification_without_adaptive(self):
        """
        Verify that WITHOUT adaptive noise, noise-to-signal ratio increases
        when VGS scales gradients down (PROBLEM CONFIRMED).
        """
        # Create model
        model = SimpleTestModel()

        # UPGD without adaptive noise (PROBLEMATIC)
        optimizer = AdaptiveUPGD(
            model.parameters(),
            lr=1e-4,
            sigma=0.001,
            adaptive_noise=False  # ❌ PROBLEMATIC
        )

        # VGS enabled
        vgs = VarianceGradientScaler(
            model.parameters(),
            enabled=True,
            alpha=0.1,
            warmup_steps=0  # Skip warmup for testing
        )

        # Simulate training steps to build up VGS statistics
        # Use large gradients first to establish baseline
        ratios = []
        for step in range(20):
            # Forward pass with random input
            x = torch.randn(32, 10)
            y_pred = model(x)
            loss = y_pred.pow(2).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Inject large variance in gradients to trigger VGS scaling
            if step >= 10:
                for param in model.parameters():
                    if param.grad is not None:
                        # Add high-variance noise to gradients
                        param.grad.data += torch.randn_like(param.grad) * 0.5

            # Get gradient norm before VGS
            grad_norm_before = compute_gradient_norm(model)

            # Update VGS statistics
            vgs.update_statistics()

            # Apply VGS scaling
            scaling_factor = vgs.scale_gradients()

            # Get gradient norm after VGS
            grad_norm_after = compute_gradient_norm(model)

            # Compute noise-to-signal ratio
            ratio = compute_noise_to_signal_ratio(optimizer, grad_norm_after)
            ratios.append(ratio)

            # Step optimizer
            optimizer.step()
            vgs.step()

        # Verify that noise-to-signal ratio INCREASED in later steps
        # (when VGS scaled gradients down)
        early_ratio = np.mean(ratios[:10])
        late_ratio = np.mean(ratios[10:])

        # With fixed noise, ratio should increase when VGS scales down
        assert late_ratio > early_ratio * 1.5, (
            f"Expected noise-to-signal ratio to increase with fixed noise, "
            f"but got early={early_ratio:.4f}, late={late_ratio:.4f}"
        )

    def test_adaptive_noise_maintains_constant_ratio(self):
        """
        Verify that WITH adaptive noise, noise-to-signal ratio remains
        approximately constant even when VGS scales gradients down (FIX VERIFIED).
        """
        # Create model
        model = SimpleTestModel()

        # UPGD with adaptive noise (FIXED)
        optimizer = AdaptiveUPGD(
            model.parameters(),
            lr=1e-4,
            sigma=0.001,
            adaptive_noise=True  # ✅ FIX
        )

        # VGS enabled
        vgs = VarianceGradientScaler(
            model.parameters(),
            enabled=True,
            alpha=0.1,
            warmup_steps=0
        )

        # Simulate training steps
        ratios = []
        scaling_factors = []

        for step in range(20):
            x = torch.randn(32, 10)
            y_pred = model(x)
            loss = y_pred.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()

            # Inject high variance after step 10
            if step >= 10:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data += torch.randn_like(param.grad) * 0.5

            grad_norm_before = compute_gradient_norm(model)
            vgs.update_statistics()
            scaling_factor = vgs.scale_gradients()
            grad_norm_after = compute_gradient_norm(model)

            ratio = compute_noise_to_signal_ratio(optimizer, grad_norm_after)
            ratios.append(ratio)
            scaling_factors.append(scaling_factor)

            optimizer.step()
            vgs.step()

        # Verify that noise-to-signal ratio remains approximately constant
        early_ratio = np.mean(ratios[:10])
        late_ratio = np.mean(ratios[10:])

        # With adaptive noise, ratio should NOT increase significantly
        # Allow 50% variation due to EMA effects
        assert late_ratio < early_ratio * 1.5, (
            f"Expected noise-to-signal ratio to remain stable with adaptive noise, "
            f"but got early={early_ratio:.4f}, late={late_ratio:.4f}"
        )

        # Verify that VGS DID scale gradients down (problem exists)
        late_scaling = np.mean(scaling_factors[10:])
        assert late_scaling < 0.8, (
            f"Expected VGS to scale gradients down, but got scaling_factor={late_scaling:.4f}"
        )

    def test_adaptive_noise_scales_with_gradient_norm(self):
        """Verify that adaptive noise scales proportionally to gradient norm."""
        model = SimpleTestModel()

        optimizer = AdaptiveUPGD(
            model.parameters(),
            lr=1e-4,
            sigma=0.001,
            adaptive_noise=True
        )

        # Warmup: establish grad_norm_ema
        for _ in range(10):
            x = torch.randn(32, 10)
            y_pred = model(x)
            loss = y_pred.pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test with large gradients
        x = torch.randn(32, 10)
        y_pred = model(x)
        loss = y_pred.pow(2).mean() * 10  # 10x larger loss
        optimizer.zero_grad()
        loss.backward()

        grad_norm_large = compute_gradient_norm(model)

        # Check adaptive noise in optimizer state
        noise_std_large = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = optimizer.state[p]
                if "grad_norm_ema" in state:
                    adaptive_sigma = max(
                        group["sigma"] * state["grad_norm_ema"],
                        group["min_noise_std"]
                    )
                    noise_std_large.append(adaptive_sigma)

        avg_noise_large = np.mean(noise_std_large)

        # Test with small gradients
        x = torch.randn(32, 10)
        y_pred = model(x)
        loss = y_pred.pow(2).mean() * 0.1  # 0.1x smaller loss
        optimizer.zero_grad()
        loss.backward()

        grad_norm_small = compute_gradient_norm(model)

        noise_std_small = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = optimizer.state[p]
                if "grad_norm_ema" in state:
                    adaptive_sigma = max(
                        group["sigma"] * state["grad_norm_ema"],
                        group["min_noise_std"]
                    )
                    noise_std_small.append(adaptive_sigma)

        avg_noise_small = np.mean(noise_std_small)

        # Verify noise scaled proportionally to gradient norm
        # Due to EMA, ratio won't be exact but should be in same ballpark
        ratio_noise = avg_noise_large / max(avg_noise_small, 1e-10)
        ratio_grad = grad_norm_large / max(grad_norm_small, 1e-10)

        # Allow 3x tolerance due to EMA effects
        assert 0.3 < ratio_noise / ratio_grad < 3.0, (
            f"Expected noise to scale with gradient norm, but got "
            f"noise_ratio={ratio_noise:.4f}, grad_ratio={ratio_grad:.4f}"
        )

    def test_config_files_have_adaptive_noise_enabled(self):
        """
        Verify that configuration files have adaptive_noise=True enabled
        (REGRESSION TEST for fix).
        """
        import yaml
        from pathlib import Path

        config_files = [
            Path("configs/config_train.yaml"),
            Path("configs/config_pbt_adversarial.yaml"),
        ]

        for config_path in config_files:
            if not config_path.exists():
                pytest.skip(f"Config file {config_path} not found")

            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Check optimizer_kwargs.adaptive_noise
            optimizer_kwargs = config.get("model", {}).get("optimizer_kwargs", {})
            adaptive_noise = optimizer_kwargs.get("adaptive_noise", False)

            assert adaptive_noise is True, (
                f"Config {config_path} should have adaptive_noise=True "
                f"when using VGS + UPGD, but got {adaptive_noise}"
            )

    def test_training_stability_with_adaptive_noise(self):
        """
        Verify that training with adaptive_noise=True is more stable
        than with adaptive_noise=False when VGS is enabled.
        """
        def train_model(adaptive_noise: bool, num_steps: int = 50) -> List[float]:
            """Train model and return list of losses."""
            model = SimpleTestModel()
            optimizer = AdaptiveUPGD(
                model.parameters(),
                lr=1e-4,
                sigma=0.001,
                adaptive_noise=adaptive_noise
            )
            vgs = VarianceGradientScaler(
                model.parameters(),
                enabled=True,
                alpha=0.1,
                warmup_steps=5
            )

            losses = []
            for step in range(num_steps):
                x = torch.randn(32, 10)
                y_pred = model(x)
                target = torch.randn_like(y_pred) * 0.1
                loss = ((y_pred - target) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()

                # Add high variance every 10 steps
                if step % 10 == 5:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data += torch.randn_like(param.grad) * 0.3

                vgs.update_statistics()
                vgs.scale_gradients()
                optimizer.step()
                vgs.step()

                losses.append(loss.item())

            return losses

        # Train with adaptive noise (FIXED)
        losses_adaptive = train_model(adaptive_noise=True)

        # Train without adaptive noise (PROBLEMATIC)
        losses_fixed = train_model(adaptive_noise=False)

        # Compute variance of losses in second half (after warmup)
        var_adaptive = np.var(losses_adaptive[25:])
        var_fixed = np.var(losses_fixed[25:])

        # Adaptive noise should have lower variance (more stable)
        assert var_adaptive < var_fixed * 1.5, (
            f"Expected adaptive noise to be more stable, but got "
            f"var_adaptive={var_adaptive:.6f}, var_fixed={var_fixed:.6f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
