"""
Basic UPGD Optimizer

Implements the core Utility-based Perturbed Gradient Descent algorithm.

The algorithm protects useful weights from forgetting by applying smaller
modifications to high-utility parameters and larger perturbations to
low-utility parameters to maintain plasticity.

Algorithm:
    1. Compute utility for each parameter: u = -grad * param
    2. Track exponential moving average of utility with bias correction
    3. Find global MIN and MAX utility across all parameters
    4. Scale utility via min-max normalization then sigmoid (NOT sigmoid(u/global_max),
       which inverts protection under negative utilities — docs/PLATFORM_REFERENCE.md "НЕ БАГИ" #5/#19/#28):
       normalized = (u - global_min) / (global_max - global_min + eps)
       scaled_utility = sigmoid(2 * (normalized - 0.5))
    5. Apply gradient update with perturbation: param -= lr * (grad + noise) * (1 - scaled_utility)

Hyperparameters:
    - lr: Learning rate (default: 1e-5)
    - weight_decay: L2 penalty coefficient (default: 0.001)
    - beta_utility: EMA decay rate for utility (default: 0.999)
    - sigma: Gaussian noise standard deviation (default: 0.001)
"""

import torch
from typing import Any, Dict, Iterable, Optional, Union


class UPGD(torch.optim.Optimizer):
    """
    Utility-based Perturbed Gradient Descent optimizer.

    This optimizer protects useful weights from catastrophic forgetting while
    maintaining plasticity through controlled perturbations.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-5)
        weight_decay: L2 penalty coefficient (default: 0.001)
        beta_utility: Exponential moving average decay for utility (default: 0.999)
        sigma: Standard deviation of Gaussian noise for perturbation (default: 0.001)

    Example:
        >>> optimizer = UPGD(model.parameters(), lr=1e-4)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]],
        lr: float = 1e-5,
        weight_decay: float = 0.001,
        beta_utility: float = 0.999,
        sigma: float = 0.001,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta_utility < 1.0:
            raise ValueError(f"Invalid beta_utility parameter: {beta_utility}")
        if not 0.0 <= sigma:
            raise ValueError(f"Invalid sigma value: {sigma}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=beta_utility,
            sigma=sigma,
        )
        super(UPGD, self).__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super(UPGD, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # First pass: compute utilities and find global min/max for normalization
        # BUGFIX: Use min-max normalization instead of division by global_max
        # This fixes inverted scaling when all utilities are negative
        global_min_util = torch.tensor(torch.inf, device="cpu")
        global_max_util = torch.tensor(-torch.inf, device="cpu")

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)

                state["step"] += 1

                # Update exponential moving average of utility
                # Utility = -grad * param (measures importance of weight)
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )

                # Track global min/max utility for normalization
                # FIX (2026-05-27): Track min/max on BIAS-CORRECTED utility
                # to prevent early-training scaling collapse where everything is clamped to 1.0 or 0.0
                bias_correction = 1.0 - group["beta_utility"] ** state["step"]
                bias_corrected_utility = avg_utility / bias_correction

                current_util_min = bias_corrected_utility.min()
                current_util_max = bias_corrected_utility.max()

                if current_util_min < global_min_util:
                    global_min_util = current_util_min.cpu()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max.cpu()

        # Second pass: apply updates with scaled utility
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                device = p.device

                # Bias correction for EMA
                bias_correction_utility = 1 - group["beta_utility"] ** state["step"]

                # Generate Gaussian noise for perturbation
                noise = torch.randn_like(p.grad) * group["sigma"]

                # Min-max normalization: maps utility to [0, 1] regardless of sign
                # High utility (after normalization) → close to 1 → small update (protection)
                # Low utility (after normalization) → close to 0 → large update (exploration)
                global_min_on_device = global_min_util.to(device)
                global_max_on_device = global_max_util.to(device)

                # Handle edge case where all utilities are equal
                epsilon = 1e-8
                util_range = global_max_on_device - global_min_on_device + epsilon

                # Normalize to [0, 1]
                normalized_utility = (
                    (state["avg_utility"] / bias_correction_utility) - global_min_on_device
                ) / util_range

                # Clamp to [0, 1] to handle numerical issues
                normalized_utility = torch.clamp(normalized_utility, 0.0, 1.0)

                # ═══════════════════════════════════════════════════════════════════════════
                # НЕ БАГ #51: SIGMOID UTILITY SCALING IS BY DESIGN
                # ═══════════════════════════════════════════════════════════════════════════
                # Maps [0, 1] → [0.27, 0.73] with sigmoid, providing GENTLER scaling.
                #
                # WHY THIS IS INTENTIONAL (NOT a bug):
                # 1. Smooth gradients: Sigmoid provides smooth gradients at boundaries
                #    (vs hard clamp which has zero gradient at edges)
                # 2. No dead zones: Even high-utility weights get 27% update (1-0.73)
                #    → prevents complete weight freezing → maintains some plasticity
                # 3. No full exposure: Even low-utility weights have 27% protection (1-0.73)
                #    → prevents uncontrolled weight changes → improves stability
                #
                # Weight factor = (1 - scaled_utility):
                # - normalized=0 (min utility) → scaled=0.27 → factor=0.73 (73% update)
                # - normalized=1 (max utility) → scaled=0.73 → factor=0.27 (27% update)
                #
                # Alternative (NOT recommended): scaled_utility = normalized_utility
                # This gives full [0,1] range but has sharp boundaries and can cause
                # instability with fully frozen or fully exposed weights.
                #
                # Reference: docs/NOT_BUGS_AND_FAQ.md → "НЕ БАГИ" → #51
                # Reference: Dohare et al. (2023) "Maintaining Plasticity in Deep Continual Learning"
                # ═══════════════════════════════════════════════════════════════════════════
                scaled_utility = torch.sigmoid(2.0 * (normalized_utility - 0.5))

                # Apply weight decay (L2 regularization)
                # Update: param -= lr * (grad + noise) * (1 - scaled_utility) - lr * weight_decay * param
                # Rearranged: param *= (1 - lr * weight_decay); param -= lr * (grad + noise) * (1 - scaled_utility)
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (p.grad.data + noise) * (1 - scaled_utility),
                    alpha=-1.0 * group["lr"],
                )

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Resets the gradients of all optimized parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zero.
                        This can slightly improve performance.
        """
        super().zero_grad(set_to_none=set_to_none)
