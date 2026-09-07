"""
Integration module for Population-Based Training + Adversarial Training.

This module provides integration between PBT, SA-PPO, and the main training pipeline.
It wraps DistributionalPPO with adversarial training capabilities and coordinates
population-based hyperparameter optimization.

Usage:
    from training_pbt_adversarial_integration import (
        create_pbt_sappo_model,
        PBTTrainingCoordinator,
        load_pbt_adversarial_config,
    )

    # Load config
    config = load_pbt_adversarial_config("configs/config_pbt_adversarial.yaml")

    # Create coordinator
    coordinator = PBTTrainingCoordinator(config)

    # Run training
    coordinator.train()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from adversarial import (
    HyperparamConfig,
    PBTConfig,
    PBTScheduler,
    PopulationMember,
    SAPPOConfig,
    StateAdversarialPPO,
    PerturbationConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class PBTAdversarialConfig:
    """Combined configuration for PBT + Adversarial Training.

    Attributes:
        pbt_enabled: Whether PBT is enabled
        adversarial_enabled: Whether adversarial training is enabled
        pbt: PBT configuration
        adversarial: SA-PPO configuration
        base_config_path: Path to base training config (e.g., config_train.yaml)
    """

    pbt_enabled: bool = True
    adversarial_enabled: bool = True
    pbt: Optional[PBTConfig] = None
    adversarial: Optional[SAPPOConfig] = None
    base_config_path: Optional[str] = None


def load_pbt_adversarial_config(config_path: str) -> PBTAdversarialConfig:
    """Load PBT + Adversarial configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        PBTAdversarialConfig instance
    """
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Parse PBT config
    pbt_enabled = data.get("pbt", {}).get("enabled", True)
    pbt_config = None
    if pbt_enabled:
        pbt_data = data["pbt"]

        # Parse hyperparameters
        hyperparams = []
        for hp_data in pbt_data.get("hyperparams", []):
            hp = HyperparamConfig(
                name=hp_data["name"],
                min_value=hp_data.get("min_value"),
                max_value=hp_data.get("max_value"),
                values=hp_data.get("values"),
                perturbation_factor=hp_data.get("perturbation_factor", 1.2),
                resample_probability=hp_data.get("resample_probability", 0.25),
                is_log_scale=hp_data.get("is_log_scale", False),
            )
            hyperparams.append(hp)

        pbt_config = PBTConfig(
            population_size=pbt_data.get("population_size", 10),
            perturbation_interval=pbt_data.get("perturbation_interval", 5),
            hyperparams=hyperparams,
            exploit_method=pbt_data.get("exploit_method", "truncation"),
            explore_method=pbt_data.get("explore_method", "both"),
            truncation_ratio=pbt_data.get("truncation_ratio", 0.2),
            checkpoint_dir=pbt_data.get("checkpoint_dir", "pbt_checkpoints"),
            metric_name=pbt_data.get("metric_name", "mean_reward"),
            metric_mode=pbt_data.get("metric_mode", "max"),
            ready_percentage=pbt_data.get("ready_percentage", 0.8),
        )

    # Parse Adversarial config
    # Only enable if adversarial key exists in YAML or use default
    adversarial_data = data.get("adversarial", {})
    adversarial_enabled = adversarial_data.get("enabled", True) if adversarial_data else False
    adversarial_config = None
    if adversarial_enabled and adversarial_data:
        adv_data = adversarial_data

        # Parse perturbation config
        pert_data = adv_data.get("perturbation", {})
        perturbation_config = PerturbationConfig(
            epsilon=pert_data.get("epsilon", 0.075),
            attack_steps=pert_data.get("attack_steps", 3),
            attack_lr=pert_data.get("attack_lr", 0.03),
            random_init=pert_data.get("random_init", True),
            norm_type=pert_data.get("norm_type", "linf"),
            clip_min=pert_data.get("clip_min"),
            clip_max=pert_data.get("clip_max"),
            attack_method=pert_data.get("attack_method", "pgd"),
        )

        adversarial_config = SAPPOConfig(
            enabled=True,
            perturbation=perturbation_config,
            adversarial_ratio=adv_data.get("adversarial_ratio", 0.5),
            robust_kl_coef=adv_data.get("robust_kl_coef", 0.1),
            warmup_updates=adv_data.get("warmup_updates", 10),
            attack_policy=adv_data.get("attack_policy", True),
            attack_value=adv_data.get("attack_value", True),
            adaptive_epsilon=adv_data.get("adaptive_epsilon", False),
            epsilon_schedule=adv_data.get("epsilon_schedule", "constant"),
            epsilon_final=adv_data.get("epsilon_final", 0.05),
        )

    return PBTAdversarialConfig(
        pbt_enabled=pbt_enabled,
        adversarial_enabled=adversarial_enabled,
        pbt=pbt_config,
        adversarial=adversarial_config,
        base_config_path=data.get("base_config_path"),
    )


def create_sappo_wrapper(
    model: Any,
    sa_ppo_config: SAPPOConfig,
) -> StateAdversarialPPO:
    """Create SA-PPO wrapper for a PPO model.

    Args:
        model: PPO model to wrap (e.g., DistributionalPPO)
        sa_ppo_config: SA-PPO configuration

    Returns:
        StateAdversarialPPO wrapper instance
    """
    sa_ppo = StateAdversarialPPO(sa_ppo_config, model)
    sa_ppo.on_training_start()

    logger.info(
        f"SA-PPO wrapper created: enabled={sa_ppo_config.enabled}, "
        f"adversarial_ratio={sa_ppo_config.adversarial_ratio}, "
        f"epsilon={sa_ppo_config.perturbation.epsilon}"
    )

    return sa_ppo


class PBTTrainingCoordinator:
    """Coordinator for PBT + Adversarial Training.

    This class manages the full training pipeline with PBT and adversarial training.
    It coordinates:
    - Population initialization
    - Parallel training of population members
    - Exploitation and exploration steps
    - Adversarial training for each member
    - Performance tracking and checkpointing
    """

    def __init__(
        self,
        config: PBTAdversarialConfig,
        seed: Optional[int] = None,
    ):
        """Initialize PBT training coordinator.

        Args:
            config: PBT + Adversarial configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed

        # Initialize PBT scheduler if enabled
        self.pbt_scheduler = None
        if config.pbt_enabled and config.pbt is not None:
            self.pbt_scheduler = PBTScheduler(config.pbt, seed=seed)
            logger.info("PBT scheduler initialized")

        # Storage for SA-PPO wrappers for each population member
        self.sa_ppo_wrappers: Dict[int, StateAdversarialPPO] = {}

        logger.info(
            f"PBT Training Coordinator initialized: "
            f"pbt_enabled={config.pbt_enabled}, "
            f"adversarial_enabled={config.adversarial_enabled}"
        )

    def initialize_population(
        self,
        initial_hyperparams: Optional[List[Dict[str, Any]]] = None,
    ) -> List[PopulationMember]:
        """Initialize PBT population.

        Args:
            initial_hyperparams: Optional initial hyperparameter configurations

        Returns:
            List of initialized population members
        """
        if self.pbt_scheduler is None:
            raise RuntimeError("PBT scheduler not initialized")

        population = self.pbt_scheduler.initialize_population(initial_hyperparams)
        logger.info(f"Population initialized with {len(population)} members")

        return population

    def create_member_model(
        self,
        member: PopulationMember,
        base_model_factory: Any,
        **model_kwargs: Any,
    ) -> Tuple[Any, Optional[StateAdversarialPPO]]:
        """Create model for a population member with optional SA-PPO wrapper.

        Args:
            member: Population member
            base_model_factory: Factory function to create base model
            **model_kwargs: Additional keyword arguments for model creation

        Returns:
            Tuple of (model, sa_ppo_wrapper)
            sa_ppo_wrapper is None if adversarial training is disabled
        """
        # Update model kwargs with member's hyperparameters
        model_kwargs.update(member.hyperparams)

        # Create base model
        model = base_model_factory(**model_kwargs)

        # Wrap with SA-PPO if enabled
        sa_ppo_wrapper = None
        if self.config.adversarial_enabled and self.config.adversarial is not None:
            sa_ppo_wrapper = create_sappo_wrapper(model, self.config.adversarial)
            self.sa_ppo_wrappers[member.member_id] = sa_ppo_wrapper

            # CRITICAL FIX: Set wrapper on model to enable adversarial training
            # Without this, compute_adversarial_loss is NEVER called!
            if hasattr(model, "set_sa_ppo_wrapper"):
                model.set_sa_ppo_wrapper(sa_ppo_wrapper)
                logger.info(f"SA-PPO wrapper attached to model for member {member.member_id}")
            else:
                logger.warning(
                    f"Model for member {member.member_id} does not support SA-PPO wrapper "
                    "(missing set_sa_ppo_wrapper method). Adversarial training DISABLED."
                )

        logger.info(
            f"Model created for member {member.member_id} with hyperparams: "
            f"{member.hyperparams}"
        )

        return model, sa_ppo_wrapper

    def on_member_update_start(self, member: PopulationMember) -> None:
        """Called at the start of each training update for a member.

        Args:
            member: Population member
        """
        # Notify SA-PPO wrapper
        if member.member_id in self.sa_ppo_wrappers:
            self.sa_ppo_wrappers[member.member_id].on_update_start()

    def on_member_update_end(
        self,
        member: PopulationMember,
        performance: float,
        step: int,
        model_state_dict: Optional[Dict[str, Any]] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[str]]:
        """Called at the end of each training update for a member.

        This method:
        1. Updates performance in PBT scheduler
        2. Checks if exploitation and exploration should occur
        3. Returns new model state and hyperparameters if PBT step occurred

        Args:
            member: Population member
            performance: Performance metric value
            step: Current training step
            model_state_dict: DEPRECATED. Model state dict to save (for backward compatibility)
            model_parameters: Full model parameters including VGS state and optimizer state (preferred).
                             Should include optimizer_state if optimizer_exploit_strategy='copy'.

        Returns:
            Tuple of (new_model_parameters, new_hyperparams, checkpoint_format)
            - new_model_parameters: Full model parameters if exploitation occurred, None otherwise
            - new_hyperparams: Updated hyperparameters
            - checkpoint_format: "v2_full_parameters", "v1_policy_only", or None

        Note:
            After receiving new_model_parameters from exploit operation:
            - If optimizer_exploit_strategy='reset': new_model_parameters will NOT contain 'optimizer_state'
              → Caller should reset optimizer after loading model weights
            - If optimizer_exploit_strategy='copy': new_model_parameters will contain 'optimizer_state'
              → Caller should load optimizer state from new_model_parameters

        IMPORTANT:
            After applying new_model_parameters, caller MUST call apply_exploited_parameters()
            to properly handle LSTM state reset and optimizer reset.
        """
        # Update performance in PBT scheduler
        if self.pbt_scheduler is not None:
            self.pbt_scheduler.update_performance(
                member,
                performance,
                step,
                model_state_dict=model_state_dict,
                model_parameters=model_parameters,
            )

        # Check if should exploit and explore
        new_parameters = None
        new_hyperparams = member.hyperparams
        checkpoint_format = None

        if self.pbt_scheduler is not None and self.pbt_scheduler.should_exploit_and_explore(member):
            new_parameters, new_hyperparams, checkpoint_format = (
                self.pbt_scheduler.exploit_and_explore(
                    member, model_state_dict=None  # Deprecated parameter, not used
                )
            )

        # Notify SA-PPO wrapper
        if member.member_id in self.sa_ppo_wrappers:
            self.sa_ppo_wrappers[member.member_id].on_update_end()

        return new_parameters, new_hyperparams, checkpoint_format

    def apply_exploited_parameters(
        self,
        model: Any,
        new_parameters: Dict[str, Any],
        member: PopulationMember,
    ) -> None:
        """Apply exploited parameters from PBT to model.

        This method handles:
        1. Loading policy weights from new_parameters
        2. Resetting LSTM states (FIX 2025-11-22 - prevents temporal mismatch)
        3. Resetting optimizer state if strategy='reset'
        4. Loading optimizer state if strategy='copy' and available

        Args:
            model: Model to update (e.g., DistributionalPPO)
            new_parameters: New parameters from PBT exploit
            member: Population member being updated

        CRITICAL FIX (2025-11-22):
            This method MUST be called after PBT exploit to prevent LSTM state
            temporal mismatch. Without LSTM reset, the model will use old LSTM
            states with new policy weights, causing 5-15% value loss spike for
            1-2 episodes until states converge.

        See: LSTM_STATE_RESET_AFTER_PBT_ANALYSIS.md for full analysis
        """
        if new_parameters is None:
            return

        # Extract policy state dict
        if isinstance(new_parameters, dict) and "policy_state" in new_parameters:
            policy_state = new_parameters["policy_state"]
        else:
            # Legacy format: direct policy state_dict
            policy_state = new_parameters

        # Load policy weights
        if hasattr(model, "policy"):
            model.policy.load_state_dict(policy_state)
            logger.info(f"Member {member.member_id}: Loaded exploited policy weights")
        elif hasattr(model, "load_state_dict"):
            model.load_state_dict(policy_state)
            logger.info(f"Member {member.member_id}: Loaded exploited model weights")
        else:
            raise ValueError(
                f"Model for member {member.member_id} does not support load_state_dict. "
                "Cannot apply exploited parameters."
            )

        # CRITICAL FIX (2025-11-22): Reset LSTM states after loading new weights
        # This prevents temporal mismatch between old LSTM states and new policy
        if hasattr(model, "reset_lstm_states_to_initial"):
            model.reset_lstm_states_to_initial()
            logger.info(
                f"Member {member.member_id}: LSTM states reset to initial after PBT exploit "
                "(prevents temporal mismatch)"
            )
        else:
            logger.debug(
                f"Member {member.member_id}: Model does not have LSTM states "
                "(reset_lstm_states_to_initial not found)"
            )

        # Handle optimizer state based on strategy
        optimizer_strategy = (
            self.config.pbt.optimizer_exploit_strategy if self.config.pbt else "reset"
        )

        if optimizer_strategy == "reset":
            # Reset optimizer to fresh state
            if hasattr(model, "optimizer") and model.optimizer is not None:
                # CRITICAL FIX (ISSUE #4): Use NEW learning rate from PBT hyperparams
                # Previously, used current_lr from old optimizer, ignoring PBT optimization
                # This caused PBT hyperparameter optimization to be ineffective for learning_rate
                if hasattr(member, "hyperparams") and "learning_rate" in member.hyperparams:
                    new_lr = float(member.hyperparams["learning_rate"])
                    logger.info(
                        f"Member {member.member_id}: Using NEW learning rate from PBT: {new_lr:.2e}"
                    )
                else:
                    # Fallback to current LR if hyperparams not available
                    new_lr = model.optimizer.param_groups[0]["lr"]
                    logger.warning(
                        f"Member {member.member_id}: learning_rate NOT found in hyperparams, "
                        f"using current optimizer LR: {new_lr:.2e}"
                    )

                optimizer_class = type(model.optimizer)
                optimizer_kwargs = {
                    "lr": new_lr,
                    **{k: v for k, v in model.optimizer.defaults.items() if k != "lr"},
                }
                model.optimizer = optimizer_class(model.policy.parameters(), **optimizer_kwargs)
                logger.info(
                    f"Member {member.member_id}: Optimizer reset to fresh state "
                    f"(strategy=reset, lr={new_lr:.2e})"
                )
        else:  # copy
            # Load optimizer state from new_parameters if available
            if isinstance(new_parameters, dict) and "optimizer_state" in new_parameters:
                if hasattr(model, "optimizer") and model.optimizer is not None:
                    model.optimizer.load_state_dict(new_parameters["optimizer_state"])
                    logger.info(
                        f"Member {member.member_id}: Optimizer state loaded from checkpoint "
                        "(strategy=copy)"
                    )

                    # CRITICAL FIX (ISSUE #4): Apply NEW learning rate from PBT hyperparams
                    # Even with copy strategy, we must update LR to match PBT optimization
                    if hasattr(member, "hyperparams") and "learning_rate" in member.hyperparams:
                        new_lr = float(member.hyperparams["learning_rate"])
                        for group in model.optimizer.param_groups:
                            group["lr"] = new_lr
                        logger.info(
                            f"Member {member.member_id}: Applied NEW learning rate from PBT: {new_lr:.2e}"
                        )
                    else:
                        logger.warning(
                            f"Member {member.member_id}: learning_rate NOT found in hyperparams, "
                            "keeping LR from copied optimizer state"
                        )
                else:
                    logger.warning(
                        f"Member {member.member_id}: optimizer_state found in checkpoint but "
                        "model.optimizer is None. Cannot load optimizer state."
                    )
            else:
                logger.warning(
                    f"Member {member.member_id}: optimizer_exploit_strategy='copy' but "
                    "optimizer_state NOT found in new_parameters. Optimizer will remain unchanged."
                )

        # Load VGS state if available
        if isinstance(new_parameters, dict) and "vgs_state" in new_parameters:
            if (
                hasattr(model, "_variance_gradient_scaler")
                and model._variance_gradient_scaler is not None
            ):
                model._variance_gradient_scaler.load_state_dict(new_parameters["vgs_state"])
                logger.info(f"Member {member.member_id}: VGS state loaded from checkpoint")
            else:
                logger.warning(
                    f"Member {member.member_id}: vgs_state found in checkpoint but "
                    "VGS not initialized. VGS state will NOT be loaded."
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from PBT and SA-PPO.

        Returns:
            Dictionary with training statistics
        """
        stats = {}

        # PBT stats
        if self.pbt_scheduler is not None:
            stats.update(self.pbt_scheduler.get_stats())

        # SA-PPO stats (aggregate from all members)
        if self.sa_ppo_wrappers:
            sa_ppo_stats = {}
            for member_id, sa_ppo in self.sa_ppo_wrappers.items():
                member_stats = sa_ppo.get_stats()
                for key, value in member_stats.items():
                    if key not in sa_ppo_stats:
                        sa_ppo_stats[key] = []
                    sa_ppo_stats[key].append(value)

            # Average stats across members
            for key, values in sa_ppo_stats.items():
                if isinstance(values[0], (int, float)):
                    stats[f"avg_{key}"] = sum(values) / len(values)

        return stats


def is_pbt_adversarial_enabled_by_default() -> bool:
    """Check if PBT + Adversarial Training is enabled by default.

    This function checks the default configuration to determine if
    PBT and Adversarial Training are enabled by default in the system.

    Returns:
        True if both PBT and Adversarial are enabled by default
    """
    # Check default config file
    default_config_path = "configs/config_pbt_adversarial.yaml"
    if os.path.exists(default_config_path):
        try:
            config = load_pbt_adversarial_config(default_config_path)
            return config.pbt_enabled and config.adversarial_enabled
        except Exception as e:
            logger.warning(f"Failed to load default config: {e}")
            return False

    # Check class defaults
    default_config = PBTAdversarialConfig()
    return default_config.pbt_enabled and default_config.adversarial_enabled


# Default instance for convenience
DEFAULT_PBT_ADVERSARIAL_CONFIG = PBTAdversarialConfig(
    pbt_enabled=True,
    adversarial_enabled=True,
)


if __name__ == "__main__":
    import argparse
    import numpy as np

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser(description="PBT + Adversarial Training Coordinator CLI")
    parser.add_argument(
        "--config",
        default="configs/config_pbt_adversarial.yaml",
        help="Path to PBT + Adversarial YAML configuration",
    )
    parser.add_argument(
        "--steps", type=int, default=15, help="Number of simulated training steps to run"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    args = parser.parse_args()

    print("=" * 80)
    print("RUNNING PBT + ADVERSARIAL TRAINING COORDINATOR DEMO")
    print("=" * 80)
    print(f"Loading configuration from: {args.config}")

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at: {args.config}")
        print("Creating a temporary config file with reasonable defaults...")
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        default_yaml = """
pbt:
  enabled: true
  population_size: 4
  perturbation_interval: 5
  exploit_method: truncation
  explore_method: both
  truncation_ratio: 0.25
  checkpoint_dir: artifacts/pbt_checkpoints
  metric_name: mean_reward
  metric_mode: max
  ready_percentage: 0.5
  hyperparams:
    - name: learning_rate
      min_value: 1.0e-5
      max_value: 5.0e-4
      perturbation_factor: 1.2
      resample_probability: 0.25
      is_log_scale: true
adversarial:
  enabled: true
  perturbation:
    epsilon: 0.05
    attack_steps: 3
    attack_lr: 0.02
    random_init: true
    norm_type: linf
    attack_method: pgd
  adversarial_ratio: 0.5
  robust_kl_coef: 0.1
  warmup_updates: 2
  attack_policy: true
  attack_value: true
"""
        with open(args.config, "w", encoding="utf-8") as f:
            f.write(default_yaml.strip())

    config = load_pbt_adversarial_config(args.config)
    print(f"PBT Enabled: {config.pbt_enabled}")
    print(f"Adversarial Enabled: {config.adversarial_enabled}")

    coordinator = PBTTrainingCoordinator(config, seed=args.seed)
    population = coordinator.initialize_population()

    # Define a simple PPO model to mock the training update
    class DemoPPOModel(torch.nn.Module):
        def __init__(self, learning_rate=3e-4):
            super().__init__()
            self.policy = torch.nn.Sequential(
                torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
            )
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        def state_dict(self):
            return self.policy.state_dict()

        def load_state_dict(self, state_dict):
            self.policy.load_state_dict(state_dict)

        def train_step(self, states, actions, advantages, returns, old_log_probs):
            self.optimizer.zero_grad()
            outputs = self.policy(states)
            loss = torch.mean((outputs - 0.5) ** 2)
            loss.backward()
            self.optimizer.step()
            return loss.item()

    # Instantiate models for the population
    models = {}
    for member in population:

        def model_factory(**kwargs):
            lr = kwargs.get("learning_rate", 3e-4)
            return DemoPPOModel(learning_rate=lr)

        model, sa_ppo = coordinator.create_member_model(member, model_factory)
        models[member.member_id] = (model, sa_ppo)

    print(f"\nCreated population models for {len(population)} members.")
    print("Running PBT training loop...")

    # Run the coordination loop
    for step in range(1, args.steps + 1):
        print(f"\n--- Step {step}/{args.steps} ---")
        for member in population:
            model, sa_ppo = models[member.member_id]

            # Start update
            coordinator.on_member_update_start(member)

            # Simulated training inputs
            states = torch.randn(4, 10)
            actions = torch.randn(4, 2)
            advantages = torch.randn(4)
            returns = torch.randn(4)
            old_log_probs = torch.randn(4)

            # Execute step
            loss = model.train_step(states, actions, advantages, returns, old_log_probs)

            # Simulated performance reward metric (minimize loss -> maximize reward)
            performance = 1.0 / (0.1 + loss) + np.random.normal(0.0, 0.1)

            # End update
            new_state, new_hp, checkpoint_format = coordinator.on_member_update_end(
                member, performance=performance, step=step, model_state_dict=model.state_dict()
            )

            print(
                f"  Member {member.member_id}: Performance={performance:.4f}, LR={member.hyperparams['learning_rate']:.2e}, Loss={loss:.4f}"
            )

            if new_state is not None:
                print(
                    f"  >> Exploited! Member {member.member_id} updated weights and hyperparams: {new_hp}"
                )
                coordinator.apply_exploited_parameters(model, new_state, member)

    print("\n" + "=" * 80)
    print("PBT + ADVERSARIAL COORDINATOR DEMO RUN COMPLETE")
    print("=" * 80)
    print(f"Final stats: {coordinator.get_stats()}")
