"""
Adaptive UPGD Optimizer

Combines UPGD with Adam-style adaptive learning rates for improved performance
in deep neural networks with heterogeneous parameter scales.

This optimizer combines:
1. Utility-based weight protection from UPGD
2. Adaptive learning rates from Adam (first and second moments)
3. Bias correction for both utility and moment estimates

Algorithm:
    1. Compute utility: u = -grad * param
    2. Update first moment: m = beta1 * m + (1-beta1) * grad
    3. Update second moment: v = beta2 * v + (1-beta2) * grad^2
    4. Apply bias correction to all three
    5. Scale utility via MIN-MAX normalization then sigmoid (NOT sigmoid(u/global_max)):
       normalized = (u - global_min) / (global_max - global_min + eps)
       scaled_utility = sigmoid(2 * (normalized - 0.5))
       (Dividing by global_max INVERTS protection under negative utilities —
        see docs/PLATFORM_REFERENCE.md "НЕ БАГИ" #5/#19/#28. The code uses min-max; do not "simplify".)
    6. Update: param -= lr * (m / (sqrt(v) + eps) + noise) * (1 - scaled_utility)
       High utility → scaled_utility→1 → small update (protect); low → large update.

This is particularly effective for RL tasks with PPO where parameter scales vary.
"""

import torch
from typing import Any, Dict, Iterable, Optional, Union


class AdaptiveUPGD(torch.optim.Optimizer):
    """
    Adaptive UPGD optimizer combining utility-based protection with Adam.

    This optimizer provides adaptive learning rates while protecting important
    weights from catastrophic forgetting.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 1e-5)
        weight_decay: L2 penalty coefficient (default: 0.001)
        beta_utility: EMA decay for utility (default: 0.999)
        sigma: Standard deviation of noise (default: 0.001).
            When adaptive_noise=False: absolute noise std.
            When adaptive_noise=True: relative noise-to-signal ratio.
        beta1: Adam first moment decay (default: 0.9)
        beta2: Adam second moment decay (default: 0.999)
        eps: Numerical stability constant for Adam (default: 1e-8)
        adaptive_noise: Enable adaptive noise scaling (default: False).
            When True, noise scales proportionally to gradient magnitude,
            maintaining constant noise-to-signal ratio regardless of VGS scaling.
        noise_beta: EMA decay for gradient norm tracking (default: 0.999).
            Only used when adaptive_noise=True and instant_noise_scale=False.
        min_noise_std: Minimum noise std to prevent noise from becoming zero (default: 1e-6).
            Only used when adaptive_noise=True.
        instant_noise_scale: Use current gradient norm directly instead of EMA (default: True).
            When True, bypasses EMA smoothing for immediate VGS compatibility.
            Ensures noise-to-signal ratio remains constant when VGS scales gradients.
            When False, uses EMA smoothing (original behavior, slow VGS adaptation).

    Example:
        >>> optimizer = AdaptiveUPGD(model.parameters(), lr=3e-4)
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
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        adaptive_noise: bool = False,
        noise_beta: float = 0.999,
        min_noise_std: float = 1e-6,
        instant_noise_scale: bool = True,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= beta_utility < 1.0:
            raise ValueError(f"Invalid beta_utility parameter: {beta_utility}")
        if not 0.0 <= sigma:
            raise ValueError(f"Invalid sigma value: {sigma}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps value: {eps}")
        if not 0.0 <= noise_beta < 1.0:
            raise ValueError(f"Invalid noise_beta parameter: {noise_beta}")
        if not 0.0 <= min_noise_std:
            raise ValueError(f"Invalid min_noise_std value: {min_noise_std}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            beta_utility=beta_utility,
            sigma=sigma,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            adaptive_noise=adaptive_noise,
            noise_beta=noise_beta,
            min_noise_std=min_noise_std,
            instant_noise_scale=instant_noise_scale,
        )
        super(AdaptiveUPGD, self).__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super(AdaptiveUPGD, self).__setstate__(state)
        # Backward compatibility: add adaptive_noise parameters if missing
        for group in self.param_groups:
            group.setdefault("adaptive_noise", False)
            group.setdefault("noise_beta", 0.999)
            group.setdefault("min_noise_std", 1e-6)
            group.setdefault("instant_noise_scale", True)  # Default to VGS-compatible mode

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        """
        Perform a single optimization step with adaptive learning rates.

        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)

        Returns:
            Loss value if closure is provided, otherwise None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # First pass: compute utilities, moments, and find global min/max for normalization
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
                    state["first_moment"] = torch.zeros_like(p.data)
                    state["sec_moment"] = torch.zeros_like(p.data)
                    # Initialize gradient norm EMA for adaptive noise (if enabled)
                    if group["adaptive_noise"]:
                        state["grad_norm_ema"] = 1.0  # Neutral starting point

                state["step"] += 1

                # Update utility EMA: u = -grad * param
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )

                # Update Adam moments
                first_moment = state["first_moment"]
                sec_moment = state["sec_moment"]
                first_moment.mul_(group["beta1"]).add_(p.grad.data, alpha=1 - group["beta1"])
                sec_moment.mul_(group["beta2"]).add_(p.grad.data**2, alpha=1 - group["beta2"])

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

        # Second pass: apply updates with adaptive learning rates and scaled utility
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                device = p.device

                # Bias corrections
                bias_correction_utility = 1 - group["beta_utility"] ** state["step"]
                bias_correction_beta1 = 1 - group["beta1"] ** state["step"]
                bias_correction_beta2 = 1 - group["beta2"] ** state["step"]

                # Bias-corrected moments (Adam-style)
                exp_avg = state["first_moment"] / bias_correction_beta1
                exp_avg_sq = state["sec_moment"] / bias_correction_beta2

                # Generate perturbation noise with optional adaptive scaling
                if group["adaptive_noise"]:
                    # Compute current gradient norm (per parameter)
                    current_grad_norm = p.grad.data.norm().item()

                    # Update gradient norm EMA (for backwards compatibility and diagnostics)
                    grad_norm_ema = state["grad_norm_ema"]
                    grad_norm_ema = (
                        group["noise_beta"] * grad_norm_ema
                        + (1 - group["noise_beta"]) * current_grad_norm
                    )
                    state["grad_norm_ema"] = grad_norm_ema

                    if group["instant_noise_scale"]:
                        # VGS-COMPATIBLE: Use current gradient norm directly
                        # This ensures noise-to-signal ratio stays constant when VGS scales gradients
                        # FIX (2025-11-26): Resolves 212x amplification with VGS + adaptive_noise
                        grad_norm_for_noise = current_grad_norm
                    else:
                        # Original behavior: Use EMA with bias correction
                        # Note: This can cause noise amplification when VGS scales gradients
                        # because EMA adapts slowly (beta=0.999 means ~0.1% per step)
                        bias_correction_noise = 1 - group["noise_beta"] ** state["step"]
                        grad_norm_for_noise = grad_norm_ema / bias_correction_noise

                    # Scale sigma to maintain constant noise-to-signal ratio
                    # Ensure minimum noise floor to prevent zero noise
                    adaptive_sigma = max(
                        group["sigma"] * grad_norm_for_noise, group["min_noise_std"]
                    )
                    noise = torch.randn_like(p.grad) * adaptive_sigma
                else:
                    # Fixed noise (original behavior)
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

                # Adaptive update with utility-based protection:
                # update = (m / (sqrt(v) + eps) + noise) * (1 - scaled_utility)
                # High utility (important weights) → scaled_utility ≈ 1 → small update
                # Low utility (less important) → scaled_utility ≈ 0 → full update + noise
                adaptive_grad = exp_avg / (exp_avg_sq.sqrt() + group["eps"])
                perturbed_update = (adaptive_grad + noise) * (1 - scaled_utility)

                # Apply weight decay and update
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    perturbed_update,
                    alpha=-1.0 * group["lr"],  # BUGFIX: Changed from -2.0 to -1.0
                )

        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Resets the gradients of all optimized parameters.

        Args:
            set_to_none: If True, set gradients to None instead of zero.
        """
        super().zero_grad(set_to_none=set_to_none)
