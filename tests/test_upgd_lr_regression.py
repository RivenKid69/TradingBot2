"""
Regression test for UPGD learning rate multiplier bug.

This test ensures that the 2x learning rate multiplier bug does not reappear.

Bug History:
- UPGD and AdaptiveUPGD previously used alpha=-2.0*lr instead of alpha=-1.0*lr
- This resulted in 2x the effective learning rate compared to standard optimizers
- Fixed by changing alpha=-2.0*lr to alpha=-1.0*lr in both optimizers

This regression test verifies:
1. Source code uses correct alpha=-1.0*lr (not -2.0*lr)
2. Step sizes are consistent between UPGD variants and UPGDW
3. Learning rate behaves as documented
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from optimizers import UPGD, AdaptiveUPGD, UPGDW


class TestUPGDLearningRateRegression:
    """Regression tests for UPGD learning rate multiplier."""

    def test_no_2x_multiplier_in_source(self):
        """
        Verify that alpha=-2.0*lr is not present in UPGD/AdaptiveUPGD source code.

        This is the most direct test - it checks the actual code for the bug.
        """
        import inspect

        upgd_source = inspect.getsource(UPGD.step)
        adaptive_source = inspect.getsource(AdaptiveUPGD.step)

        # Should NOT contain alpha=-2.0
        assert "alpha=-2.0" not in upgd_source, "UPGD still contains alpha=-2.0 multiplier bug!"
        assert (
            "alpha=-2.0" not in adaptive_source
        ), "AdaptiveUPGD still contains alpha=-2.0 multiplier bug!"

        # SHOULD contain alpha=-1.0
        assert "alpha=-1.0" in upgd_source, "UPGD should use alpha=-1.0*lr"
        assert "alpha=-1.0" in adaptive_source, "AdaptiveUPGD should use alpha=-1.0*lr"

    def test_adaptive_upgd_matches_upgdw_step_size(self):
        """
        Verify that AdaptiveUPGD and UPGDW produce the same step sizes.

        Both optimizers use the same adaptive learning rate formula (Adam-style).
        After the fix, they should produce identical or very similar steps.
        """
        torch.manual_seed(123)
        lr = 0.1

        # Create two identical parameters
        param1 = nn.Parameter(torch.ones(5, 5))
        param2 = nn.Parameter(torch.ones(5, 5))

        # Same optimizer settings
        opt1 = AdaptiveUPGD(
            [param1], lr=lr, beta1=0.9, beta2=0.999, beta_utility=0.999, sigma=0.0, weight_decay=0.0
        )

        opt2 = UPGDW([param2], lr=lr, betas=(0.9, 0.999), weight_decay=0.0, sigma=0.0)

        # Run multiple steps with same gradients
        for _ in range(5):
            grad = torch.randn(5, 5) * 0.1

            param1.grad = grad.clone()
            param2.grad = grad.clone()

            init1 = param1.data.clone()
            init2 = param2.data.clone()

            opt1.step()
            opt2.step()

            change1 = param1.data - init1
            change2 = param2.data - init2

            # They should be very close (within numerical precision)
            torch.testing.assert_close(
                change1,
                change2,
                rtol=1e-4,
                atol=1e-6,
                msg="AdaptiveUPGD and UPGDW should produce the same steps",
            )

            opt1.zero_grad()
            opt2.zero_grad()

    def test_step_size_not_doubled(self):
        """
        Verify that the step size is NOT 2x what it should be.

        With alpha=-1.0*lr, the update should be: param -= 1.0*lr*grad*(1-utility)
        With alpha=-2.0*lr (bug), it would be: param -= 2.0*lr*grad*(1-utility)

        This test verifies the former, not the latter.
        """
        torch.manual_seed(42)
        lr = 0.01

        param = nn.Parameter(torch.ones(3))

        opt = UPGD(
            [param], lr=lr, sigma=0.0, weight_decay=0.0, beta_utility=0.0  # No EMA for clean test
        )

        # Set gradient
        grad = torch.tensor([0.5, -0.5, 1.0])
        param.grad = grad.clone()

        initial = param.data.clone()
        opt.step()

        change = param.data - initial

        # With the BUG (2x multiplier), the magnitude would be ~2x larger
        # We verify it's NOT that large

        # Expected magnitude with 1x (accounting for utility scaling)
        # The exact value depends on utility, but it should be O(lr * grad)
        # not O(2 * lr * grad)

        magnitude = change.abs().mean().item()

        # If the bug were present, magnitude would be ~2x larger
        # So magnitude should be less than 2*lr*grad_mag
        grad_mag = grad.abs().mean().item()
        max_expected = 2.0 * lr * grad_mag  # This would be the buggy version

        # With fix, magnitude should be < max_expected/2 (approximately)
        assert (
            magnitude < max_expected
        ), f"Step size too large: {magnitude} >= {max_expected}. Bug may be present!"

        # More precise check: magnitude should be closer to lr*grad_mag, not 2*lr*grad_mag
        expected_1x = lr * grad_mag
        expected_2x = 2.0 * lr * grad_mag

        error_from_1x = abs(magnitude - expected_1x)
        error_from_2x = abs(magnitude - expected_2x)

        assert (
            error_from_1x < error_from_2x
        ), f"Step size closer to 2x ({expected_2x}) than 1x ({expected_1x}). Bug may be present!"

    def test_upgd_upgdw_similar_magnitudes(self):
        """
        Verify that UPGD (non-adaptive) and UPGDW have reasonably similar step magnitudes.

        They use different formulas (UPGD: basic, UPGDW: Adam-style), but both
        should be in the same ballpark after the fix.
        """
        torch.manual_seed(99)
        lr = 0.01

        param1 = nn.Parameter(torch.ones(10))
        param2 = nn.Parameter(torch.ones(10))

        opt1 = UPGD([param1], lr=lr, sigma=0.0, weight_decay=0.0)
        opt2 = UPGDW([param2], lr=lr, betas=(0.0, 0.0), sigma=0.0, weight_decay=0.0)

        grad = torch.randn(10) * 0.5

        param1.grad = grad.clone()
        param2.grad = grad.clone()

        init1 = param1.data.clone()
        init2 = param2.data.clone()

        opt1.step()
        opt2.step()

        change1 = (param1.data - init1).abs().mean().item()
        change2 = (param2.data - init2).abs().mean().item()

        ratio = change1 / change2 if change2 > 0 else 0

        # Ratio should be close to 1.0, definitely NOT 2.0
        assert 0.1 < ratio < 10.0, f"Unexpected ratio {ratio} between UPGD and UPGDW step sizes"

        # More importantly, ratio should NOT be close to 2.0 (bug signature)
        assert not (
            1.8 < ratio < 2.2
        ), f"Ratio {ratio} is close to 2.0 - this suggests the bug is still present!"

    def test_learning_rate_scaling(self):
        """
        Verify that doubling the learning rate doubles the step size.

        This confirms that lr is applied correctly with 1x multiplier, not 2x.
        """
        torch.manual_seed(77)

        lr_small = 0.01
        lr_large = 0.02  # 2x

        param1 = nn.Parameter(torch.ones(5))
        param2 = nn.Parameter(torch.ones(5))

        opt1 = AdaptiveUPGD(
            [param1],
            lr=lr_small,
            sigma=0.0,
            weight_decay=0.0,
            beta1=0.0,
            beta2=0.0,
            beta_utility=0.0,
        )
        opt2 = AdaptiveUPGD(
            [param2],
            lr=lr_large,
            sigma=0.0,
            weight_decay=0.0,
            beta1=0.0,
            beta2=0.0,
            beta_utility=0.0,
        )

        grad = torch.ones(5) * 0.5

        param1.grad = grad.clone()
        param2.grad = grad.clone()

        init1 = param1.data.clone()
        init2 = param2.data.clone()

        opt1.step()
        opt2.step()

        change1 = (param1.data - init1).abs().mean().item()
        change2 = (param2.data - init2).abs().mean().item()

        ratio = change2 / change1 if change1 > 0 else 0

        # With correct implementation, doubling lr should double the step
        # Ratio should be close to 2.0
        assert (
            1.8 < ratio < 2.2
        ), f"Learning rate scaling incorrect: 2x lr gave {ratio}x step (expected ~2.0)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
