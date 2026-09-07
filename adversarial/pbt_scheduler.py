"""
Population-Based Training (PBT) Scheduler

Implements population-based hyperparameter optimization for RL training.

Based on "Population Based Training of Neural Networks" (DeepMind 2017):
- Maintains a population of parallel training runs
- Periodically evaluates and ranks population members
- Exploitation: Copy parameters from better performers
- Exploration: Perturb hyperparameters for diversity

Key features:
1. Asynchronous population management
2. Configurable exploitation and exploration strategies
3. Hyperparameter mutation (perturbation and resampling)
4. Model checkpoint management
5. Performance-based ranking
"""

from __future__ import annotations

import copy
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class HyperparamConfig:
    """Configuration for a single hyperparameter.

    Attributes:
        name: Hyperparameter name
        min_value: Minimum allowed value (for continuous parameters)
        max_value: Maximum allowed value (for continuous parameters)
        values: Discrete values to choose from (for categorical parameters)
        perturbation_factor: Multiplicative factor for perturbation (default: 1.2)
        resample_probability: Probability of resampling instead of perturbing (default: 0.25)
        is_log_scale: Whether to use log scale for perturbations (default: False)
    """

    name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    values: Optional[List[Any]] = None
    perturbation_factor: float = 1.2
    resample_probability: float = 0.25
    is_log_scale: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.values is None and (self.min_value is None or self.max_value is None):
            raise ValueError(
                f"Either 'values' or 'min_value'/'max_value' must be specified for {self.name}"
            )
        if self.values is not None and (self.min_value is not None or self.max_value is not None):
            raise ValueError(
                f"Cannot specify both 'values' and 'min_value'/'max_value' for {self.name}"
            )
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value >= self.max_value
        ):
            raise ValueError(f"min_value must be < max_value for {self.name}")
        if not 0.0 <= self.resample_probability <= 1.0:
            raise ValueError(f"resample_probability must be in [0, 1] for {self.name}")

    @property
    def is_continuous(self) -> bool:
        """Check if hyperparameter is continuous."""
        return self.values is None

    @property
    def is_categorical(self) -> bool:
        """Check if hyperparameter is categorical."""
        return self.values is not None


@dataclass
class PBTConfig:
    """Configuration for Population-Based Training.

    Attributes:
        population_size: Number of parallel training runs
        perturbation_interval: Training steps between perturbations
        hyperparams: List of hyperparameter configurations
        exploit_method: Exploitation strategy ('truncation', 'binary_tournament')
        explore_method: Exploration strategy ('perturb', 'resample', 'both')
        truncation_ratio: Ratio of population to truncate (for 'truncation' method)
        checkpoint_dir: Directory to save checkpoints
        metric_name: Name of metric to optimize
        metric_mode: Optimization mode ('max' or 'min')
        ready_percentage: Percentage of population that must be ready for exploitation
        min_ready_members: Minimum absolute number of ready members (fallback for deadlock prevention)
        ready_check_max_wait: Maximum number of consecutive failed ready checks before fallback
        optimizer_exploit_strategy: Strategy for optimizer state during exploit
                                    ('reset' or 'copy')
                                    - 'reset': Reset optimizer state after exploit (recommended)
                                    - 'copy': Copy optimizer state from source agent (advanced)
    """

    population_size: int = 10
    perturbation_interval: int = 5
    hyperparams: List[HyperparamConfig] = field(default_factory=list)
    exploit_method: str = "truncation"
    explore_method: str = "both"
    truncation_ratio: float = 0.2
    checkpoint_dir: str = "pbt_checkpoints"
    metric_name: str = "mean_reward"
    metric_mode: str = "max"
    ready_percentage: float = 0.8
    min_ready_members: int = 2  # Minimum viable population for PBT
    ready_check_max_wait: int = 10  # Max consecutive failed checks before fallback
    optimizer_exploit_strategy: str = "reset"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")
        if self.perturbation_interval < 1:
            raise ValueError(
                f"perturbation_interval must be >= 1, got {self.perturbation_interval}"
            )
        if self.exploit_method not in ("truncation", "binary_tournament"):
            raise ValueError(
                f"exploit_method must be 'truncation' or 'binary_tournament', got {self.exploit_method}"
            )
        if self.explore_method not in ("perturb", "resample", "both"):
            raise ValueError(
                f"explore_method must be 'perturb', 'resample', or 'both', got {self.explore_method}"
            )
        if not 0.0 < self.truncation_ratio < 1.0:
            raise ValueError(f"truncation_ratio must be in (0, 1), got {self.truncation_ratio}")
        if self.metric_mode not in ("max", "min"):
            raise ValueError(f"metric_mode must be 'max' or 'min', got {self.metric_mode}")
        if not 0.0 < self.ready_percentage <= 1.0:
            raise ValueError(f"ready_percentage must be in (0, 1], got {self.ready_percentage}")
        if self.min_ready_members < 2:
            raise ValueError(
                f"min_ready_members must be >= 2 (minimum viable population), got {self.min_ready_members}"
            )
        if self.ready_check_max_wait < 1:
            raise ValueError(f"ready_check_max_wait must be >= 1, got {self.ready_check_max_wait}")
        if self.optimizer_exploit_strategy not in ("reset", "copy"):
            raise ValueError(
                f"optimizer_exploit_strategy must be 'reset' or 'copy', got {self.optimizer_exploit_strategy}"
            )


@dataclass
class PopulationMember:
    """Represents a single member of the PBT population.

    Attributes:
        member_id: Unique identifier for this member
        hyperparams: Current hyperparameter values
        performance: Current performance metric
        step: Current training step
        checkpoint_path: Path to saved checkpoint
        history: History of hyperparameter changes and performance
    """

    member_id: int
    hyperparams: Dict[str, Any]
    performance: Optional[float] = None
    step: int = 0
    checkpoint_path: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record_step(self, step: int, performance: float, hyperparams: Dict[str, Any]) -> None:
        """Record a training step in history.

        Args:
            step: Training step number
            performance: Performance metric value
            hyperparams: Current hyperparameter values
        """
        self.step = step
        self.performance = performance
        self.history.append(
            {
                "step": step,
                "performance": performance,
                "hyperparams": copy.deepcopy(hyperparams),
            }
        )


class PBTScheduler:
    """Population-Based Training scheduler.

    Manages a population of training runs with periodic exploitation and exploration.
    """

    def __init__(self, config: PBTConfig, seed: Optional[int] = None):
        """Initialize PBT scheduler.

        Args:
            config: PBT configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.population: List[PopulationMember] = []
        self._exploitation_count = 0
        self._exploration_count = 0
        self._failed_ready_checks = (
            0  # Track consecutive failed ready checks for deadlock prevention
        )

        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        logger.info(
            f"PBT Scheduler initialized: population_size={config.population_size}, "
            f"perturbation_interval={config.perturbation_interval}, "
            f"metric={config.metric_name} ({config.metric_mode}), "
            f"ready_percentage={config.ready_percentage}, "
            f"min_ready_members={config.min_ready_members} (fallback)"
        )

    def initialize_population(
        self,
        initial_hyperparams: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PopulationMember]:
        """Initialize population with random or provided hyperparameters.

        Args:
            initial_hyperparams: Optional list of initial hyperparameter configurations

        Returns:
            List of initialized population members
        """
        if initial_hyperparams is not None:
            if len(initial_hyperparams) != self.config.population_size:
                raise ValueError(
                    f"initial_hyperparams length ({len(initial_hyperparams)}) must match "
                    f"population_size ({self.config.population_size})"
                )
            hyperparams_list = initial_hyperparams
        else:
            # Generate random initial hyperparameters
            hyperparams_list = [
                self._sample_hyperparams() for _ in range(self.config.population_size)
            ]

        # Create population members
        self.population = [
            PopulationMember(
                member_id=i,
                hyperparams=hyperparams,
            )
            for i, hyperparams in enumerate(hyperparams_list)
        ]

        logger.info(f"Population initialized with {len(self.population)} members")
        return self.population

    def should_exploit_and_explore(self, member: PopulationMember) -> bool:
        """Check if a member should undergo exploitation and exploration.

        Args:
            member: Population member to check

        Returns:
            True if member should be perturbed
        """
        return member.step > 0 and member.step % self.config.perturbation_interval == 0

    def exploit_and_explore(
        self,
        member: PopulationMember,
        model_state_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[str]]:
        """Perform exploitation and exploration for a population member.

        ═══════════════════════════════════════════════════════════════════════════════
        НЕ БАГ #52: VGS STATE IS PROPERLY HANDLED DURING PBT EXPLOIT
        ═══════════════════════════════════════════════════════════════════════════════
        When loading v2_full_parameters checkpoint, VGS state is INCLUDED:
        - Line 357: has_vgs = "vgs_state" in new_parameters
        - Line 359-361: Logs VGS state presence
        - Caller (DistributionalPPO.set_parameters) restores VGS via _restore_vgs_state()

        This ensures gradient variance statistics are transferred along with model weights,
        maintaining VGS consistency after PBT exploitation.

        Reference: docs/NOT_BUGS_AND_FAQ.md → "НЕ БАГИ" → #52
        ═══════════════════════════════════════════════════════════════════════════════

        Args:
            member: Population member to update
            model_state_dict: DEPRECATED. Current model state dict (unused, for backward compatibility)

        Returns:
            Tuple of (new_model_parameters, new_hyperparams, checkpoint_format)
            - new_model_parameters: Full model parameters including:
                * Policy/value network weights
                * VGS state (gradient variance statistics) - ALWAYS included in v2
                * Optimizer state (optional, depends on strategy)
            - new_hyperparams: Updated hyperparameters
            - checkpoint_format: "v2_full_parameters", "v1_policy_only", or None

        Note:
            Optimizer state handling depends on config.optimizer_exploit_strategy:
            - 'reset': Optimizer state is REMOVED from new_model_parameters
                      (caller should reset optimizer)
            - 'copy': Optimizer state is INCLUDED in new_model_parameters
                     (caller should load optimizer state from checkpoint)
        """
        # Ready check with fallback mechanism to prevent deadlock
        # Check if enough population members are ready
        ready_count = sum(1 for m in self.population if m.performance is not None)
        required_count = int(self.config.population_size * self.config.ready_percentage)
        min_count = self.config.min_ready_members

        # Primary check: ready_percentage threshold
        if ready_count < required_count:
            self._failed_ready_checks += 1

            # Fallback mechanism: if we've waited too long and have minimum viable population
            if (
                self._failed_ready_checks >= self.config.ready_check_max_wait
                and ready_count >= min_count
            ):
                logger.warning(
                    f"PBT deadlock prevention: {self._failed_ready_checks} consecutive failed ready checks. "
                    f"Proceeding with fallback: ready_count={ready_count} >= min_ready_members={min_count}. "
                    f"Some workers may have crashed. Consider reducing ready_percentage or checking worker health."
                )
                self._failed_ready_checks = 0  # Reset counter after fallback
            else:
                # Still waiting for more members
                if self._failed_ready_checks == 1:
                    # Log first failure at INFO level
                    logger.info(
                        f"Not enough ready members ({ready_count}/{required_count}), skipping PBT. "
                        f"Will use fallback after {self.config.ready_check_max_wait} consecutive failures."
                    )
                elif self._failed_ready_checks % 5 == 0:
                    # Log every 5th failure at WARNING level
                    logger.warning(
                        f"Still waiting for ready members ({ready_count}/{required_count}). "
                        f"Failed checks: {self._failed_ready_checks}/{self.config.ready_check_max_wait}. "
                        f"Minimum viable population: {min_count}."
                    )
                return None, member.hyperparams, None
        else:
            # Sufficient members ready - reset failure counter
            if self._failed_ready_checks > 0:
                logger.info(
                    f"Sufficient members now ready ({ready_count}/{required_count}). "
                    f"Resuming normal PBT operation after {self._failed_ready_checks} failed checks."
                )
                self._failed_ready_checks = 0

        # Exploitation: decide whether to copy from better performer
        new_parameters = None
        checkpoint_format = None

        if self._should_exploit(member):
            source_member = self._select_source_member(member)
            if source_member is not None and source_member.checkpoint_path is not None:
                # Load checkpoint from better performer
                logger.info(
                    f"Member {member.member_id} exploiting from member {source_member.member_id} "
                    f"(performance: {member.performance:.4f} -> {source_member.performance:.4f})"
                )

                # SECURITY: Fail-closed model loading per THREAT_MODEL_MODEL_LOADING.md
                # First try secure loading (weights_only=True)
                # Fall back to unsafe loading only with explicit opt-in via env var
                checkpoint = None
                try:
                    checkpoint = torch.load(
                        source_member.checkpoint_path, map_location="cpu", weights_only=True
                    )
                except Exception as secure_load_error:
                    # Secure loading failed - check if unsafe fallback is allowed
                    if os.environ.get("ALLOW_UNSAFE_MODEL_LOAD", "").lower() in (
                        "1",
                        "true",
                        "yes",
                    ):
                        logger.warning(
                            f"Member {member.member_id}: Secure loading failed ({type(secure_load_error).__name__}). "
                            f"Using unsafe fallback (ALLOW_UNSAFE_MODEL_LOAD=1). "
                            f"SECURITY METRIC: pbt_unsafe_load_count=1"
                        )
                        checkpoint = torch.load(
                            source_member.checkpoint_path, map_location="cpu", weights_only=False
                        )
                    else:
                        logger.error(
                            f"Member {member.member_id}: Checkpoint cannot be loaded securely. "
                            f"Error: {secure_load_error}. "
                            f"Set ALLOW_UNSAFE_MODEL_LOAD=1 to enable legacy loading, "
                            f"or convert checkpoint using tools/convert_legacy_models.py"
                        )
                        raise RuntimeError(
                            f"Checkpoint at {source_member.checkpoint_path} cannot be loaded securely. "
                            f"See docs/security/THREAT_MODEL_MODEL_LOADING.md for mitigation options."
                        ) from secure_load_error

                # Handle both old and new checkpoint formats
                if isinstance(checkpoint, dict) and "format_version" in checkpoint:
                    # New format (v2): Contains metadata and full parameters
                    checkpoint_format = checkpoint["format_version"]
                    new_parameters = checkpoint["data"]

                    if checkpoint_format == "v2_full_parameters":
                        has_vgs = (
                            "vgs_state" in new_parameters
                            if isinstance(new_parameters, dict)
                            else False
                        )
                        has_optimizer = (
                            "optimizer_state" in new_parameters
                            if isinstance(new_parameters, dict)
                            else False
                        )
                        logger.info(
                            f"Member {member.member_id}: Loaded v2 checkpoint "
                            f"(VGS: {has_vgs}, Optimizer: {has_optimizer})"
                        )
                    else:
                        logger.warning(
                            f"Member {member.member_id}: Loaded v1 checkpoint (policy_only). "
                            f"VGS state and optimizer state will NOT be transferred!"
                        )
                else:
                    # Legacy format (v1): Direct policy state_dict
                    checkpoint_format = "v1_policy_only"
                    new_parameters = checkpoint
                    logger.warning(
                        f"Member {member.member_id}: Loaded legacy checkpoint format. "
                        f"VGS state and optimizer state NOT included. Consider re-saving checkpoints."
                    )

                # Handle optimizer state based on strategy
                if isinstance(new_parameters, dict) and "optimizer_state" in new_parameters:
                    if self.config.optimizer_exploit_strategy == "reset":
                        # Remove optimizer state - caller should reset optimizer
                        new_parameters = dict(new_parameters)  # Make a copy
                        new_parameters.pop("optimizer_state", None)
                        logger.info(
                            f"Member {member.member_id}: Optimizer state REMOVED (strategy=reset). "
                            f"Optimizer will be reset after loading weights."
                        )
                    else:  # copy
                        # Keep optimizer state - caller should load it
                        logger.info(
                            f"Member {member.member_id}: Optimizer state INCLUDED (strategy=copy). "
                            f"Optimizer state will be copied from source agent."
                        )
                elif self.config.optimizer_exploit_strategy == "copy":
                    logger.warning(
                        f"Member {member.member_id}: Optimizer state NOT found in checkpoint "
                        f"but strategy='copy'. Optimizer will be reset. "
                        f"Consider using model.get_parameters(include_optimizer=True)."
                    )

                # Copy hyperparameters from source
                member.hyperparams = copy.deepcopy(source_member.hyperparams)
                self._exploitation_count += 1

        # Exploration: perturb hyperparameters
        new_hyperparams = self._explore_hyperparams(member.hyperparams)
        member.hyperparams = new_hyperparams
        self._exploration_count += 1

        return new_parameters, new_hyperparams, checkpoint_format

    def update_performance(
        self,
        member: PopulationMember,
        performance: float,
        step: int,
        model_state_dict: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update performance and save checkpoint for a population member.

        Args:
            member: Population member to update
            performance: Performance metric value
            step: Current training step
            model_state_dict: DEPRECATED. Model state dict to save (for backward compatibility)
            model_parameters: Full model parameters including VGS state and optimizer state (preferred).
                             If not provided, falls back to model_state_dict.
                             Should include optimizer_state if optimizer_exploit_strategy='copy'.
        """
        member.record_step(step, performance, member.hyperparams)

        # Save checkpoint if model state provided
        # Prefer model_parameters (includes VGS state and optimizer state) over model_state_dict
        checkpoint_data = model_parameters if model_parameters is not None else model_state_dict

        if checkpoint_data is not None:
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir, f"member_{member.member_id}_step_{step}.pt"
            )

            # Add metadata to distinguish checkpoint format
            has_vgs = "vgs_state" in checkpoint_data if isinstance(checkpoint_data, dict) else False
            has_optimizer = (
                "optimizer_state" in checkpoint_data if isinstance(checkpoint_data, dict) else False
            )

            checkpoint_to_save = {
                "format_version": (
                    "v2_full_parameters" if model_parameters is not None else "v1_policy_only"
                ),
                "data": checkpoint_data,
                "step": step,
                "performance": performance,
                "has_optimizer_state": has_optimizer,
            }

            torch.save(checkpoint_to_save, checkpoint_path)
            member.checkpoint_path = checkpoint_path

            if model_parameters is not None:
                logger.debug(
                    f"Member {member.member_id} step {step}: Saved full model parameters "
                    f"(VGS: {has_vgs}, Optimizer: {has_optimizer})"
                )
            else:
                logger.warning(
                    f"Member {member.member_id} step {step}: Saved legacy format (policy_only). "
                    f"VGS state and optimizer state will NOT be preserved during exploitation. "
                    f"Use model_parameters=model.get_parameters(include_optimizer=True) instead."
                )

        logger.debug(
            f"Member {member.member_id} step {step}: "
            f"performance={performance:.4f}, "
            f"hyperparams={member.hyperparams}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get PBT statistics.

        Returns:
            Dictionary with PBT statistics
        """
        performances = [m.performance for m in self.population if m.performance is not None]

        return {
            "pbt/population_size": len(self.population),
            "pbt/ready_members": len(performances),
            "pbt/mean_performance": np.mean(performances) if performances else 0.0,
            "pbt/std_performance": np.std(performances) if performances else 0.0,
            "pbt/min_performance": np.min(performances) if performances else 0.0,
            "pbt/max_performance": np.max(performances) if performances else 0.0,
            "pbt/exploitation_count": self._exploitation_count,
            "pbt/exploration_count": self._exploration_count,
            "pbt/failed_ready_checks": self._failed_ready_checks,  # Track deadlock risk
        }

    def _should_exploit(self, member: PopulationMember) -> bool:
        """Determine if a member should exploit from better performers.

        Args:
            member: Population member to check

        Returns:
            True if member should exploit
        """
        if member.performance is None:
            return False

        # Rank members by performance
        ranked_members = self._rank_population()

        if self.config.exploit_method == "truncation":
            # Check if member is in bottom truncation_ratio
            truncation_threshold = int(len(ranked_members) * self.config.truncation_ratio)
            member_rank = next(
                i for i, m in enumerate(ranked_members) if m.member_id == member.member_id
            )
            return member_rank >= len(ranked_members) - truncation_threshold
        elif self.config.exploit_method == "binary_tournament":
            # Randomly select another member and compare
            other_member = random.choice(
                [m for m in self.population if m.member_id != member.member_id]
            )
            if other_member.performance is None:
                return False
            if self.config.metric_mode == "max":
                return member.performance < other_member.performance
            else:
                return member.performance > other_member.performance
        else:
            return False

    def _select_source_member(self, member: PopulationMember) -> Optional[PopulationMember]:
        """Select a better-performing member to exploit from.

        Args:
            member: Population member that will exploit

        Returns:
            Source member to copy from, or None if no suitable source found
        """
        # Rank members by performance
        ranked_members = self._rank_population()

        if self.config.exploit_method == "truncation":
            # Select from top truncation_ratio
            truncation_threshold = int(len(ranked_members) * self.config.truncation_ratio)
            top_members = ranked_members[:truncation_threshold]
            # Randomly select from top performers
            return random.choice(top_members) if top_members else None
        elif self.config.exploit_method == "binary_tournament":
            # Select the better performer from binary tournament
            other_member = random.choice(
                [m for m in self.population if m.member_id != member.member_id]
            )
            if other_member.performance is None:
                return None
            if self.config.metric_mode == "max":
                return other_member if other_member.performance > member.performance else None
            else:
                return other_member if other_member.performance < member.performance else None
        else:
            return None

    def _rank_population(self) -> List[PopulationMember]:
        """Rank population members by performance.

        Returns:
            List of members sorted by performance (best first)
        """
        # Filter members with performance data
        valid_members = [m for m in self.population if m.performance is not None]

        # Sort by performance
        reverse = self.config.metric_mode == "max"
        return sorted(valid_members, key=lambda m: m.performance, reverse=reverse)

    def _explore_hyperparams(self, current_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """Explore (perturb) hyperparameters.

        Args:
            current_hyperparams: Current hyperparameter values

        Returns:
            New hyperparameter values after exploration
        """
        new_hyperparams = copy.deepcopy(current_hyperparams)

        for hyperparam_config in self.config.hyperparams:
            name = hyperparam_config.name
            if name not in new_hyperparams:
                continue

            current_value = new_hyperparams[name]

            # Decide whether to resample or perturb
            if self.config.explore_method == "resample":
                should_resample = True
            elif self.config.explore_method == "perturb":
                should_resample = False
            else:  # "both"
                should_resample = random.random() < hyperparam_config.resample_probability

            if should_resample:
                # Resample from distribution
                new_value = self._sample_hyperparam(hyperparam_config)
            else:
                # Perturb current value
                new_value = self._perturb_hyperparam(hyperparam_config, current_value)

            new_hyperparams[name] = new_value

        return new_hyperparams

    def _sample_hyperparams(self) -> Dict[str, Any]:
        """Sample initial hyperparameters for a new population member.

        Returns:
            Dictionary of sampled hyperparameter values
        """
        hyperparams = {}
        for hyperparam_config in self.config.hyperparams:
            hyperparams[hyperparam_config.name] = self._sample_hyperparam(hyperparam_config)
        return hyperparams

    def _sample_hyperparam(self, config: HyperparamConfig) -> Any:
        """Sample a single hyperparameter value.

        Args:
            config: Hyperparameter configuration

        Returns:
            Sampled value
        """
        if config.is_categorical:
            return random.choice(config.values)
        else:
            if config.is_log_scale:
                log_min = np.log(config.min_value)
                log_max = np.log(config.max_value)
                return np.exp(random.uniform(log_min, log_max))
            else:
                return random.uniform(config.min_value, config.max_value)

    def _perturb_hyperparam(self, config: HyperparamConfig, current_value: Any) -> Any:
        """Perturb a single hyperparameter value.

        Args:
            config: Hyperparameter configuration
            current_value: Current value to perturb

        Returns:
            Perturbed value
        """
        if config.is_categorical:
            # For categorical, select adjacent value or random value
            current_index = config.values.index(current_value)
            # Randomly move left or right
            if random.random() < 0.5:
                new_index = (current_index + 1) % len(config.values)
            else:
                new_index = (current_index - 1) % len(config.values)
            return config.values[new_index]
        else:
            # For continuous, multiply or divide by perturbation_factor
            if random.random() < 0.5:
                new_value = current_value * config.perturbation_factor
            else:
                new_value = current_value / config.perturbation_factor

            # Clip to valid range
            new_value = max(config.min_value, min(config.max_value, new_value))

            return new_value
