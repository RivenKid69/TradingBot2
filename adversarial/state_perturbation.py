"""
State Perturbation Module for Adversarial Training

Implements PGD (Projected Gradient Descent) and FGSM (Fast Gradient Sign Method)
attacks for generating adversarial state perturbations in reinforcement learning.

Based on:
- "Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations" (NeurIPS 2020)
- "State-Adversarial PPO for robust deep reinforcement learning" (SA-PPO)

This module provides the inner maximization component for robust RL training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class PerturbationConfig:
    """Configuration for adversarial state perturbations.

    Attributes:
        epsilon: Maximum L-infinity norm of perturbation
        attack_steps: Number of PGD attack iterations
        attack_lr: Step size for PGD attack
        random_init: Whether to use random initialization for PGD
        norm_type: Type of norm constraint ('linf', 'l2')
        clip_min: Minimum value for state clipping (None = no clipping)
        clip_max: Maximum value for state clipping (None = no clipping)
        attack_method: Attack method to use ('pgd', 'fgsm')
    """
    epsilon: float = 0.075
    attack_steps: int = 3
    attack_lr: float = 0.03
    random_init: bool = True
    norm_type: str = "linf"
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    attack_method: str = "pgd"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {self.epsilon}")
        if self.attack_steps < 1:
            raise ValueError(f"attack_steps must be >= 1, got {self.attack_steps}")
        if self.attack_lr <= 0:
            raise ValueError(f"attack_lr must be > 0, got {self.attack_lr}")
        if self.norm_type not in ("linf", "l2"):
            raise ValueError(f"norm_type must be 'linf' or 'l2', got {self.norm_type}")
        if self.attack_method not in ("pgd", "fgsm"):
            raise ValueError(f"attack_method must be 'pgd' or 'fgsm', got {self.attack_method}")
        if self.clip_min is not None and self.clip_max is not None:
            if self.clip_min >= self.clip_max:
                raise ValueError(f"clip_min ({self.clip_min}) must be < clip_max ({self.clip_max})")


class StatePerturbation:
    """Generates adversarial perturbations on state observations.

    This class implements the inner maximization problem for robust RL:
        max_{||delta|| <= epsilon} L(s + delta, a)

    where L is the loss function (e.g., policy loss, value loss).
    """

    def __init__(self, config: PerturbationConfig):
        """Initialize state perturbation generator.

        Args:
            config: Configuration for perturbation generation
        """
        self.config = config
        self._attack_count = 0
        self._total_perturbation_norm = 0.0

    def reset_stats(self) -> None:
        """Reset attack statistics."""
        self._attack_count = 0
        self._total_perturbation_norm = 0.0

    def get_stats(self) -> Dict[str, float]:
        """Get attack statistics.

        Returns:
            Dictionary with attack statistics
        """
        if self._attack_count == 0:
            return {
                "attack_count": 0,
                "avg_perturbation_norm": 0.0,
            }

        return {
            "attack_count": self._attack_count,
            "avg_perturbation_norm": self._total_perturbation_norm / self._attack_count,
        }

    def fgsm_attack(
        self,
        state: Tensor,
        loss_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Fast Gradient Sign Method (FGSM) attack.

        Single-step attack using the sign of the gradient:
            delta = epsilon * sign(grad_s L(s, a))

        Args:
            state: Input state tensor [batch_size, ...]
            loss_fn: Loss function that takes perturbed state and returns scalar loss

        Returns:
            Adversarial perturbation delta
        """
        # Enable gradient tracking for state
        state_adv = state.detach().clone()
        state_adv.requires_grad = True

        # Compute loss and gradients
        loss = loss_fn(state_adv)
        grad = torch.autograd.grad(loss, state_adv, create_graph=False)[0]

        # Generate perturbation
        if self.config.norm_type == "linf":
            delta = self.config.epsilon * grad.sign()
        elif self.config.norm_type == "l2":
            # Normalize gradient to unit L2 norm, then scale by epsilon
            grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1, keepdim=True)
            grad_norm = grad_norm.view(grad.size(0), *([1] * (len(grad.shape) - 1)))
            delta = self.config.epsilon * grad / (grad_norm + 1e-8)
        else:
            raise ValueError(f"Unsupported norm_type: {self.config.norm_type}")

        # Clip perturbation if bounds specified
        if self.config.clip_min is not None or self.config.clip_max is not None:
            state_adv = state + delta
            state_adv = self._clip_state(state_adv, state)
            delta = state_adv - state

        # Update statistics
        self._update_stats(delta)

        return delta.detach()

    def pgd_attack(
        self,
        state: Tensor,
        loss_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Projected Gradient Descent (PGD) attack.

        Multi-step iterative attack with projection:
            delta_t+1 = Proj_{||delta|| <= epsilon}(delta_t + alpha * sign(grad_s L(s + delta_t, a)))

        Args:
            state: Input state tensor [batch_size, ...]
            loss_fn: Loss function that takes perturbed state and returns scalar loss

        Returns:
            Adversarial perturbation delta
        """
        batch_size = state.size(0)

        # Initialize perturbation
        if self.config.random_init:
            # Random initialization within epsilon ball
            if self.config.norm_type == "linf":
                delta = torch.empty_like(state).uniform_(-self.config.epsilon, self.config.epsilon)
            elif self.config.norm_type == "l2":
                delta = torch.randn_like(state)
                delta_norm = torch.norm(delta.view(batch_size, -1), p=2, dim=1, keepdim=True)
                delta_norm = delta_norm.view(batch_size, *([1] * (len(delta.shape) - 1)))
                # Random radius within epsilon
                radius = torch.rand(batch_size, 1, device=state.device) * self.config.epsilon
                radius = radius.view(batch_size, *([1] * (len(delta.shape) - 1)))
                delta = delta * radius / (delta_norm + 1e-8)
            else:
                raise ValueError(f"Unsupported norm_type: {self.config.norm_type}")
        else:
            delta = torch.zeros_like(state)

        # Iterative attack
        for step in range(self.config.attack_steps):
            delta.requires_grad = True

            # Compute loss on perturbed state
            state_adv = state + delta
            loss = loss_fn(state_adv)

            # Compute gradient
            grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

            # Update perturbation
            with torch.no_grad():
                if self.config.norm_type == "linf":
                    # L-infinity PGD step
                    delta = delta + self.config.attack_lr * grad.sign()
                    # Project to epsilon ball
                    delta = torch.clamp(delta, -self.config.epsilon, self.config.epsilon)
                elif self.config.norm_type == "l2":
                    # L2 PGD step
                    grad_norm = torch.norm(grad.view(batch_size, -1), p=2, dim=1, keepdim=True)
                    grad_norm = grad_norm.view(batch_size, *([1] * (len(grad.shape) - 1)))
                    delta = delta + self.config.attack_lr * grad / (grad_norm + 1e-8)
                    # Project to L2 ball
                    delta_norm = torch.norm(delta.view(batch_size, -1), p=2, dim=1, keepdim=True)
                    delta_norm = delta_norm.view(batch_size, *([1] * (len(delta.shape) - 1)))
                    delta = delta * torch.clamp(delta_norm, max=self.config.epsilon) / (delta_norm + 1e-8)

                # Clip to valid state range if bounds specified
                if self.config.clip_min is not None or self.config.clip_max is not None:
                    state_adv = state + delta
                    state_adv = self._clip_state(state_adv, state)
                    delta = state_adv - state

        # Update statistics
        self._update_stats(delta)

        return delta.detach()

    def generate_perturbation(
        self,
        state: Tensor,
        loss_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """Generate adversarial perturbation using configured attack method.

        Args:
            state: Input state tensor [batch_size, ...]
            loss_fn: Loss function that takes perturbed state and returns scalar loss

        Returns:
            Adversarial perturbation delta
        """
        if self.config.attack_method == "fgsm":
            return self.fgsm_attack(state, loss_fn)
        elif self.config.attack_method == "pgd":
            return self.pgd_attack(state, loss_fn)
        else:
            raise ValueError(f"Unknown attack method: {self.config.attack_method}")

    def _clip_state(self, state_adv: Tensor, state_orig: Tensor) -> Tensor:
        """Clip adversarial state to valid range.

        Args:
            state_adv: Adversarial state
            state_orig: Original state (for reference)

        Returns:
            Clipped adversarial state
        """
        if self.config.clip_min is not None and self.config.clip_max is not None:
            return torch.clamp(state_adv, self.config.clip_min, self.config.clip_max)
        elif self.config.clip_min is not None:
            return torch.clamp(state_adv, min=self.config.clip_min)
        elif self.config.clip_max is not None:
            return torch.clamp(state_adv, max=self.config.clip_max)
        return state_adv

    def _update_stats(self, delta: Tensor) -> None:
        """Update attack statistics.

        Args:
            delta: Perturbation tensor
        """
        with torch.no_grad():
            batch_size = delta.size(0)
            if self.config.norm_type == "linf":
                norm = torch.abs(delta.view(batch_size, -1)).max(dim=1)[0].mean().item()
            elif self.config.norm_type == "l2":
                norm = torch.norm(delta.view(batch_size, -1), p=2, dim=1).mean().item()
            else:
                norm = 0.0

            self._attack_count += 1
            self._total_perturbation_norm += norm


def create_policy_loss_fn(
    model: nn.Module,
    states: Tensor,
    actions: Tensor,
    advantages: Tensor,
    old_log_probs: Tensor,
    clip_range: float = 0.2,
) -> Callable[[Tensor], Tensor]:
    """Create a loss function for policy adversarial training.

    This loss function can be used with StatePerturbation to generate
    adversarial perturbations that maximize policy loss.

    Args:
        model: Policy model
        states: Original states (not perturbed)
        actions: Actions taken
        advantages: Advantage estimates
        old_log_probs: Log probabilities from old policy
        clip_range: PPO clip range

    Returns:
        Loss function that takes perturbed states and returns scalar loss
    """
    def loss_fn(states_perturbed: Tensor) -> Tensor:
        """Compute PPO policy loss on perturbed states."""
        # Get log probabilities from policy
        with torch.enable_grad():
            dist = model.get_distribution(states_perturbed)
            log_probs = dist.log_prob(actions)

            # PPO clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        return loss

    return loss_fn


def create_value_loss_fn(
    model: nn.Module,
    states: Tensor,
    returns: Tensor,
) -> Callable[[Tensor], Tensor]:
    """Create a loss function for value adversarial training.

    This loss function can be used with StatePerturbation to generate
    adversarial perturbations that maximize value loss.

    Args:
        model: Value model
        states: Original states (not perturbed)
        returns: Target returns

    Returns:
        Loss function that takes perturbed states and returns scalar loss
    """
    def loss_fn(states_perturbed: Tensor) -> Tensor:
        """Compute MSE value loss on perturbed states."""
        with torch.enable_grad():
            values = model.predict_values(states_perturbed)
            loss = F.mse_loss(values, returns)

        return loss

    return loss_fn
