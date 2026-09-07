"""
State-Adversarial PPO (SA-PPO) Implementation

Extends DistributionalPPO with adversarial training for robustness.

Based on "Robust Deep Reinforcement Learning against Adversarial Perturbations
on State Observations" (NeurIPS 2020 Spotlight).

Key features:
1. Adversarial state perturbations during training
2. Robust KL regularization based on SA-MDP
3. Mixed clean/adversarial training for balance
4. Compatible with existing DistributionalPPO architecture
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from adversarial.state_perturbation import PerturbationConfig, StatePerturbation

logger = logging.getLogger(__name__)


@dataclass
class SAPPOConfig:
    """Configuration for State-Adversarial PPO.

    Attributes:
        enabled: Whether to enable adversarial training
        perturbation: Configuration for state perturbations
        adversarial_ratio: Ratio of adversarial samples (0.0 = all clean, 1.0 = all adversarial)
        robust_kl_coef: Coefficient for robust KL regularization
        warmup_updates: Number of updates before enabling adversarial training
        attack_policy: Whether to attack policy loss
        attack_value: Whether to attack value loss
        adaptive_epsilon: Whether to adapt epsilon based on training progress
        epsilon_schedule: Schedule for epsilon adaptation ('constant', 'linear', 'cosine')
        epsilon_final: Final epsilon value for schedule
        max_updates: Optional override for epsilon schedule duration (computed from model if None)
    """
    enabled: bool = True
    perturbation: PerturbationConfig = None
    adversarial_ratio: float = 0.5
    robust_kl_coef: float = 0.1
    warmup_updates: int = 10
    attack_policy: bool = True
    attack_value: bool = True
    adaptive_epsilon: bool = False
    epsilon_schedule: str = "constant"
    epsilon_final: float = 0.05
    max_updates: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize default perturbation config if not provided."""
        if self.perturbation is None:
            self.perturbation = PerturbationConfig()

        # Validate parameters
        if not 0.0 <= self.adversarial_ratio <= 1.0:
            raise ValueError(f"adversarial_ratio must be in [0, 1], got {self.adversarial_ratio}")
        if self.robust_kl_coef < 0:
            raise ValueError(f"robust_kl_coef must be >= 0, got {self.robust_kl_coef}")
        if self.warmup_updates < 0:
            raise ValueError(f"warmup_updates must be >= 0, got {self.warmup_updates}")
        if self.epsilon_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(f"epsilon_schedule must be 'constant', 'linear', or 'cosine', got {self.epsilon_schedule}")


class StateAdversarialPPO:
    """State-Adversarial PPO wrapper for robust training.

    This class wraps a PPO model and adds adversarial training capabilities.
    It can be used with DistributionalPPO or any PPO-based algorithm.
    """

    def __init__(
        self,
        config: SAPPOConfig,
        model: Any,  # DistributionalPPO or similar
    ):
        """Initialize SA-PPO wrapper.

        Args:
            config: SA-PPO configuration
            model: PPO model to wrap (e.g., DistributionalPPO)
        """
        self.config = config
        self.model = model
        self.perturbation_gen = StatePerturbation(config.perturbation)

        self._update_count = 0
        self._adversarial_enabled = False

        # Compute max_updates for epsilon schedule
        self._max_updates = self._compute_max_updates()

        # Statistics
        self._total_adversarial_samples = 0
        self._total_clean_samples = 0
        self._total_robust_kl_penalty = 0.0

    def _get_distribution(
        self,
        obs: Tensor,
        lstm_states: Optional[Any] = None,
        episode_starts: Optional[Tensor] = None,
    ) -> Any:
        """Helper to get distribution from policy, supporting both feedforward and recurrent policies."""
        if lstm_states is not None and episode_starts is not None:
            dist_output = self.model.policy.get_distribution(obs, lstm_states, episode_starts)
        else:
            dist_output = self.model.policy.get_distribution(obs)
            
        if isinstance(dist_output, tuple):
            return dist_output[0]
        return dist_output

    def _predict_values(
        self,
        obs: Tensor,
        lstm_states: Optional[Any] = None,
        episode_starts: Optional[Tensor] = None,
    ) -> Tensor:
        """Helper to predict values from policy, supporting both feedforward and recurrent policies."""
        if lstm_states is not None and episode_starts is not None:
            return self.model.policy.predict_values(obs, lstm_states, episode_starts)
        return self.model.policy.predict_values(obs)

    @property
    def is_adversarial_enabled(self) -> bool:
        """Check if adversarial training is currently enabled."""
        return self._adversarial_enabled and self._update_count >= self.config.warmup_updates

    def reset_stats(self) -> None:
        """Reset training statistics."""
        self._total_adversarial_samples = 0
        self._total_clean_samples = 0
        self._total_robust_kl_penalty = 0.0
        self.perturbation_gen.reset_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics.

        Returns:
            Dictionary with SA-PPO statistics
        """
        total_samples = self._total_adversarial_samples + self._total_clean_samples
        perturbation_stats = self.perturbation_gen.get_stats()

        return {
            "sa_ppo/enabled": self.is_adversarial_enabled,
            "sa_ppo/update_count": self._update_count,
            "sa_ppo/adversarial_samples": self._total_adversarial_samples,
            "sa_ppo/clean_samples": self._total_clean_samples,
            "sa_ppo/adversarial_ratio": (
                self._total_adversarial_samples / total_samples if total_samples > 0 else 0.0
            ),
            "sa_ppo/robust_kl_penalty": (
                self._total_robust_kl_penalty / self._update_count if self._update_count > 0 else 0.0
            ),
            "sa_ppo/current_epsilon": self._get_current_epsilon(),
            "sa_ppo/attack_count": perturbation_stats.get("attack_count", 0),
            "sa_ppo/avg_perturbation_norm": perturbation_stats.get("avg_perturbation_norm", 0.0),
        }

    def on_training_start(self) -> None:
        """Called when training starts."""
        self._adversarial_enabled = self.config.enabled
        logger.info(
            f"SA-PPO initialized: enabled={self.config.enabled}, "
            f"warmup_updates={self.config.warmup_updates}, "
            f"adversarial_ratio={self.config.adversarial_ratio}"
        )

    def on_update_start(self) -> None:
        """Called at the start of each update."""
        self._update_count += 1

        # Update epsilon schedule
        if self.config.adaptive_epsilon:
            self._update_epsilon_schedule()

    def on_update_end(self) -> None:
        """Called at the end of each update."""
        pass

    def _compute_max_updates(self) -> int:
        """Compute maximum updates for epsilon schedule.

        Dynamically computes max_updates from model configuration.

        Priority:
        1. config.max_updates (explicit override)
        2. total_timesteps / n_steps from model
        3. Infer from current progress (num_timesteps)
        4. Conservative default (10000)

        Returns:
            Maximum number of updates for epsilon schedule
        """
        # Priority 1: Explicit override in config
        if self.config.max_updates is not None:
            logger.info(f"SA-PPO: Using configured max_updates={self.config.max_updates}")
            return self.config.max_updates

        # Helper to check if a value is a valid numeric type (and not a mock)
        is_num = lambda x: isinstance(x, (int, float)) and not hasattr(x, "mock_add_spec")

        # Priority 2: Compute from total_timesteps and n_steps
        total_timesteps = getattr(self.model, 'total_timesteps', None)
        n_steps = getattr(self.model, 'n_steps', None)

        if is_num(total_timesteps) and is_num(n_steps) and n_steps > 0:
            max_updates = int(total_timesteps // n_steps)
            logger.info(
                f"SA-PPO: Computed max_updates={max_updates} from "
                f"total_timesteps={total_timesteps}, n_steps={n_steps}"
            )
            return max_updates

        # Priority 3: Infer from current progress (assume halfway through training)
        num_timesteps = getattr(self.model, 'num_timesteps', 0)
        if is_num(num_timesteps) and num_timesteps > 0 and is_num(n_steps) and n_steps > 0:
            estimated_max = int((num_timesteps * 2) // n_steps)
            logger.warning(
                f"SA-PPO: Cannot determine total_timesteps. "
                f"Estimating max_updates={estimated_max} from current progress "
                f"(num_timesteps={num_timesteps}, n_steps={n_steps})"
            )
            return estimated_max

        # Priority 4: Conservative default (more conservative than old hardcoded 1000)
        logger.warning(
            "SA-PPO: Cannot determine max_updates from model config. "
            "Using conservative default value of 10000. "
            "Set config.max_updates explicitly for accurate epsilon schedule."
        )
        return 10000

    def compute_adversarial_loss(
        self,
        states: Tensor,
        actions: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_log_probs: Tensor,
        old_values: Optional[Tensor] = None,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        lstm_states: Optional[Any] = None,
        episode_starts: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute adversarial loss with robust training.

        This method:
        1. Splits batch into clean and adversarial samples
        2. Generates adversarial perturbations for adversarial samples
        3. Computes loss on mixed batch
        4. Adds robust KL regularization
        5. Includes entropy regularization for exploration

        Args:
            states: State observations [batch_size, ...]
            actions: Actions taken [batch_size, ...]
            advantages: Advantage estimates [batch_size]
            returns: Target returns [batch_size]
            old_log_probs: Log probs from old policy [batch_size]
            old_values: Values from old policy [batch_size] (optional)
            clip_range: PPO clip range
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            lstm_states: LSTM states from rollout buffer
            episode_starts: Episode starts tensor

        Returns:
            Tuple of (total_loss, info_dict)
        """
        info = {}

        # Check if adversarial training is enabled
        if not self.is_adversarial_enabled:
            # Fall back to standard training
            return self._compute_standard_loss(
                states, actions, advantages, returns, old_log_probs, old_values, clip_range, ent_coef, vf_coef, lstm_states, episode_starts
            )

        batch_size = states.size(0)
        num_adversarial = int(batch_size * self.config.adversarial_ratio)
        num_clean = batch_size - num_adversarial

        # Update sample counts
        self._total_adversarial_samples += num_adversarial
        self._total_clean_samples += num_clean

        # Split into clean and adversarial samples
        # For simplicity, take first num_clean as clean, rest as adversarial
        states_clean = states[:num_clean]
        states_adv_base = states[num_clean:]

        actions_clean = actions[:num_clean]
        actions_adv = actions[num_clean:]

        advantages_clean = advantages[:num_clean]
        advantages_adv = advantages[num_clean:]

        returns_clean = returns[:num_clean]
        returns_adv = returns[num_clean:]

        old_log_probs_clean = old_log_probs[:num_clean]
        old_log_probs_adv = old_log_probs[num_clean:]

        # Split recurrent states if present
        lstm_states_clean = None
        lstm_states_adv = None
        episode_starts_clean = None
        episode_starts_adv = None
        if lstm_states is not None:
            lstm_states_clean = tuple(s[:, :num_clean] for s in lstm_states)
            lstm_states_adv = tuple(s[:, num_clean:] for s in lstm_states)
        if episode_starts is not None:
            episode_starts_clean = episode_starts[:num_clean]
            episode_starts_adv = episode_starts[num_clean:]

        # Generate adversarial perturbations for policy
        states_adv_perturbed = states_adv_base
        if self.config.attack_policy and num_adversarial > 0:
            # Create policy loss function for attack
            def policy_loss_fn(s_perturbed: Tensor) -> Tensor:
                dist = self._get_distribution(s_perturbed, lstm_states_adv, episode_starts_adv)
                log_probs = dist.log_prob(actions_adv)
                ratio = torch.exp(log_probs - old_log_probs_adv)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                # Negative because we want to maximize loss (worst-case)
                return -torch.min(ratio * advantages_adv, clipped_ratio * advantages_adv).mean()

            # Generate perturbations
            delta = self.perturbation_gen.generate_perturbation(states_adv_base, policy_loss_fn)
            states_adv_perturbed = states_adv_base + delta

        # Combine clean and adversarial states
        states_combined = torch.cat([states_clean, states_adv_perturbed], dim=0)
        actions_combined = torch.cat([actions_clean, actions_adv], dim=0)
        advantages_combined = torch.cat([advantages_clean, advantages_adv], dim=0)
        returns_combined = torch.cat([returns_clean, returns_adv], dim=0)
        old_log_probs_combined = torch.cat([old_log_probs_clean, old_log_probs_adv], dim=0)

        # Compute policy loss
        dist = self._get_distribution(states_combined, lstm_states, episode_starts)
        log_probs = dist.log_prob(actions_combined)
        ratio = torch.exp(log_probs - old_log_probs_combined)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.min(ratio * advantages_combined, clipped_ratio * advantages_combined).mean()

        # Compute entropy for exploration
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)
        entropy_loss = -torch.mean(entropy)

        # Compute value loss (optionally with adversarial perturbations)
        if self.config.attack_value and num_adversarial > 0:
            # Generate perturbations for value function
            def value_loss_fn(s_perturbed: Tensor) -> Tensor:
                values = self._predict_values(s_perturbed, lstm_states_adv, episode_starts_adv)
                return nn.functional.mse_loss(values, returns_adv.view_as(values))

            delta_value = self.perturbation_gen.generate_perturbation(states_adv_base, value_loss_fn)
            states_adv_value = states_adv_base + delta_value

            # Combine for value computation
            states_value_combined = torch.cat([states_clean, states_adv_value], dim=0)
        else:
            states_value_combined = states_combined

        values = self._predict_values(states_value_combined, lstm_states, episode_starts)
        value_loss = nn.functional.mse_loss(values, returns_combined.view_as(values))

        # Compute robust KL regularization
        robust_kl_penalty = 0.0
        kl_method = "none"
        if self.config.robust_kl_coef > 0 and num_adversarial > 0:
            # Get distributions from clean and adversarial states
            with torch.no_grad():
                dist_clean = self._get_distribution(states_adv_base, lstm_states_adv, episode_starts_adv)

            dist_adv = self._get_distribution(states_adv_perturbed, lstm_states_adv, episode_starts_adv)

            # Compute KL divergence: KL(clean || adversarial)
            # Use analytical KL divergence when available (exact for Gaussian)
            try:
                # Analytical KL divergence (exact for Gaussian distributions)
                kl_div = torch.distributions.kl_divergence(dist_clean, dist_adv).mean()
                kl_method = "analytical"
            except NotImplementedError:
                # Fallback: Monte Carlo approximation
                # KL(π₁||π₂) ≈ E_π₁[log π₁(a) - log π₂(a)]
                with torch.no_grad():
                    log_probs_clean = dist_clean.log_prob(actions_adv)

                log_probs_adv = dist_adv.log_prob(actions_adv)
                kl_div = (log_probs_clean - log_probs_adv).mean()
                kl_method = "monte_carlo"

            robust_kl_penalty = self.config.robust_kl_coef * kl_div
            self._total_robust_kl_penalty += robust_kl_penalty.item()

        # Total loss with entropy regularization (CRITICAL FIX)
        # Standard PPO loss includes entropy to encourage exploration and prevent policy collapse
        # References: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
        total_loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss + robust_kl_penalty

        # Info dict
        info.update({
            "sa_ppo/policy_loss": policy_loss.item(),
            "sa_ppo/value_loss": value_loss.item(),
            "sa_ppo/entropy_loss": entropy_loss.item(),
            "sa_ppo/entropy": -entropy_loss.item(),  # Actual entropy value (positive)
            "sa_ppo/robust_kl_penalty": robust_kl_penalty if isinstance(robust_kl_penalty, float) else robust_kl_penalty.item(),
            "sa_ppo/kl_method": kl_method,  # Log KL computation method (analytical/monte_carlo)
            "sa_ppo/num_adversarial": num_adversarial,
            "sa_ppo/num_clean": num_clean,
        })

        return total_loss, info

    def _compute_standard_loss(
        self,
        states: Tensor,
        actions: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_log_probs: Tensor,
        old_values: Optional[Tensor],
        clip_range: float,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        lstm_states: Optional[Any] = None,
        episode_starts: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute standard PPO loss without adversarial training.

        Args:
            states: State observations [batch_size, ...]
            actions: Actions taken [batch_size, ...]
            advantages: Advantage estimates [batch_size]
            returns: Target returns [batch_size]
            old_log_probs: Log probs from old policy [batch_size]
            old_values: Values from old policy [batch_size] (optional)
            clip_range: PPO clip range
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            lstm_states: LSTM states from rollout buffer
            episode_starts: Episode starts tensor

        Returns:
            Tuple of (total_loss, info_dict)
        """
        # Standard PPO policy loss
        dist = self._get_distribution(states, lstm_states, episode_starts)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Compute entropy for exploration (CRITICAL FIX)
        entropy = dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)
        entropy_loss = -torch.mean(entropy)

        # Standard value loss
        values = self._predict_values(states, lstm_states, episode_starts)
        value_loss = nn.functional.mse_loss(values, returns.view_as(values))

        # Total loss with entropy regularization (CRITICAL FIX)
        # Standard PPO loss includes entropy to encourage exploration and prevent policy collapse
        # References: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
        total_loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss

        info = {
            "sa_ppo/policy_loss": policy_loss.item(),
            "sa_ppo/value_loss": value_loss.item(),
            "sa_ppo/entropy_loss": entropy_loss.item(),
            "sa_ppo/entropy": -entropy_loss.item(),  # Actual entropy value (positive)
            "sa_ppo/robust_kl_penalty": 0.0,
        }

        return total_loss, info

    def _get_current_epsilon(self) -> float:
        """Get current epsilon value based on schedule.

        Uses dynamically computed max_updates for proper epsilon annealing.
        """
        if not self.config.adaptive_epsilon:
            return self.config.perturbation.epsilon

        # Use computed max_updates (from __init__)
        progress = min(1.0, self._update_count / self._max_updates)

        epsilon_init = self.config.perturbation.epsilon
        epsilon_final = self.config.epsilon_final

        if self.config.epsilon_schedule == "linear":
            epsilon = epsilon_init + (epsilon_final - epsilon_init) * progress
        elif self.config.epsilon_schedule == "cosine":
            import math
            epsilon = epsilon_final + 0.5 * (epsilon_init - epsilon_final) * (1 + math.cos(math.pi * progress))
        else:  # constant
            epsilon = epsilon_init

        return epsilon

    def _update_epsilon_schedule(self) -> None:
        """Update epsilon value based on schedule."""
        current_epsilon = self._get_current_epsilon()
        self.config.perturbation.epsilon = current_epsilon

    def apply_adversarial_augmentation(
        self,
        states: Tensor,
        actions: Tensor,
        advantages: Tensor,
        old_log_probs: Tensor,
        clip_range: float,
        lstm_states: Optional[Any] = None,
        episode_starts: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, float]]:
        """Apply adversarial augmentation to a batch of states.

        This method generates adversarial perturbations and returns augmented states
        for use in downstream loss computation. It does NOT compute loss itself.

        Args:
            states: State observations [batch_size, ...]
            actions: Actions taken [batch_size, ...]
            advantages: Advantage estimates [batch_size]
            old_log_probs: Log probs from old policy [batch_size]
            clip_range: PPO clip range
            lstm_states: LSTM states from rollout buffer
            episode_starts: Episode starts tensor

        Returns:
            Tuple of (augmented_states, sample_mask, info_dict)
            - augmented_states: Combined clean + adversarial states [batch_size, ...]
            - sample_mask: Mask indicating which samples are adversarial [batch_size] (0=clean, 1=adv)
            - info_dict: Statistics dictionary
        """
        info = {}

        # Check if adversarial training is enabled
        if not self.is_adversarial_enabled:
            # Return original states with all-zero mask (all clean)
            sample_mask = torch.zeros(states.size(0), device=states.device, dtype=torch.float32)
            info.update({
                "sa_ppo/num_adversarial": 0,
                "sa_ppo/num_clean": states.size(0),
            })
            return states, sample_mask, info

        batch_size = states.size(0)
        num_adversarial = int(batch_size * self.config.adversarial_ratio)
        num_clean = batch_size - num_adversarial

        # Update sample counts
        self._total_adversarial_samples += num_adversarial
        self._total_clean_samples += num_clean

        # Split into clean and adversarial samples
        states_clean = states[:num_clean]
        states_adv_base = states[num_clean:]

        actions_adv = actions[num_clean:]
        advantages_adv = advantages[num_clean:]
        old_log_probs_adv = old_log_probs[num_clean:]

        # Split recurrent states if present
        lstm_states_adv = None
        episode_starts_adv = None
        if lstm_states is not None:
            lstm_states_adv = tuple(s[:, num_clean:] for s in lstm_states)
        if episode_starts is not None:
            episode_starts_adv = episode_starts[num_clean:]

        # Generate adversarial perturbations for policy
        states_adv_perturbed = states_adv_base
        if self.config.attack_policy and num_adversarial > 0:
            # Create policy loss function for attack
            def policy_loss_fn(s_perturbed: Tensor) -> Tensor:
                dist = self._get_distribution(s_perturbed, lstm_states_adv, episode_starts_adv)
                log_probs = dist.log_prob(actions_adv)
                ratio = torch.exp(log_probs - old_log_probs_adv)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                # Negative because we want to maximize loss (worst-case)
                return -torch.min(ratio * advantages_adv, clipped_ratio * advantages_adv).mean()

            # Generate perturbations
            delta = self.perturbation_gen.generate_perturbation(states_adv_base, policy_loss_fn)
            states_adv_perturbed = states_adv_base + delta

        # Combine clean and adversarial states
        states_combined = torch.cat([states_clean, states_adv_perturbed], dim=0)

        # Create sample mask (0 = clean, 1 = adversarial)
        sample_mask = torch.cat([
            torch.zeros(num_clean, device=states.device, dtype=torch.float32),
            torch.ones(num_adversarial, device=states.device, dtype=torch.float32),
        ], dim=0)

        # Info dict
        info.update({
            "sa_ppo/num_adversarial": num_adversarial,
            "sa_ppo/num_clean": num_clean,
        })

        return states_combined, sample_mask, info

    def compute_robust_kl_penalty(
        self,
        states_clean: Tensor,
        states_adv: Tensor,
        actions: Tensor,
        lstm_states_clean: Optional[Any] = None,
        lstm_states_adv: Optional[Any] = None,
        episode_starts_clean: Optional[Tensor] = None,
        episode_starts_adv: Optional[Tensor] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute robust KL regularization between clean and adversarial policies.

        Uses analytical KL divergence when available (exact for Gaussian).

        Args:
            states_clean: Clean state observations [batch_size, ...]
            states_adv: Adversarial (perturbed) state observations [batch_size, ...]
            actions: Actions to evaluate [batch_size, ...]
            lstm_states_clean: Clean LSTM states
            lstm_states_adv: Adversarial LSTM states
            episode_starts_clean: Clean episode starts
            episode_starts_adv: Adversarial episode starts

        Returns:
            Tuple of (robust_kl_penalty, info_dict)
        """
        info = {}

        if not self.is_adversarial_enabled or self.config.robust_kl_coef <= 0:
            return 0.0, info

        batch_size = states_clean.size(0)
        if batch_size == 0:
            return 0.0, info

        # Get distributions from clean and adversarial states
        with torch.no_grad():
            dist_clean = self._get_distribution(states_clean, lstm_states_clean, episode_starts_clean)

        dist_adv = self._get_distribution(states_adv, lstm_states_adv, episode_starts_adv)

        # Compute KL divergence: KL(clean || adversarial)
        # Prefer analytical KL divergence for better accuracy and efficiency
        kl_method = "unknown"
        try:
            # Analytical KL divergence (exact for Gaussian distributions)
            # References:
            # - PyTorch: torch.distributions.kl.kl_divergence
            # - For Normal distributions: KL(π₁||π₂) = log(σ₂/σ₁) + (σ₁²+(μ₁-μ₂)²)/(2σ₂²) - 1/2
            kl_div = torch.distributions.kl_divergence(dist_clean, dist_adv).mean()
            kl_method = "analytical"
        except NotImplementedError:
            # Fallback: Monte Carlo approximation
            # KL(π₁||π₂) ≈ E_π₁[log π₁(a) - log π₂(a)]
            # Use actions sampled from clean distribution (from rollout buffer)
            with torch.no_grad():
                log_probs_clean = dist_clean.log_prob(actions)

            log_probs_adv = dist_adv.log_prob(actions)
            kl_div = (log_probs_clean - log_probs_adv).mean()
            kl_method = "monte_carlo"

        robust_kl_penalty = float((self.config.robust_kl_coef * kl_div).item())
        self._total_robust_kl_penalty += robust_kl_penalty

        info.update({
            "sa_ppo/robust_kl_penalty": robust_kl_penalty,
            "sa_ppo/kl_method": kl_method,
            "sa_ppo/kl_divergence": float(kl_div.item()),
        })

        return robust_kl_penalty, info
