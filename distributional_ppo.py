import itertools
import math
from collections import deque
from typing import Any, Optional, Sequence, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.type_aliases import GymEnv

try:
    from stable_baselines3.common.vec_env.vec_normalize import unwrap_vec_normalize as _sb3_unwrap
except Exception:  # pragma: no cover - backcompat guard
    _sb3_unwrap = None


torch.set_float32_matmul_precision("high")


def unwrap_vec_normalize(env: VecEnv) -> Optional[VecNormalize]:
    """Backwards compatible helper to locate VecNormalize inside wrappers."""

    if _sb3_unwrap is not None:
        return _sb3_unwrap(env)

    try:
        from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
    except Exception:  # pragma: no cover - optional dependency guard
        return None

    candidate: Optional[VecEnv] = env
    while candidate is not None:
        if isinstance(candidate, VecNormalize):
            return candidate
        if not isinstance(candidate, VecEnvWrapper):
            break
        candidate = getattr(candidate, "venv", None)
    return None


def safe_explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Stable explained variance that guards against zero variance targets."""

    y_true64 = y_true.astype(np.float64)
    y_pred64 = y_pred.astype(np.float64)
    var_y = np.var(y_true64)
    if var_y == 0.0:
        return float("nan")
    return float(1.0 - np.var(y_true64 - y_pred64) / var_y)


def calculate_cvar(probs: torch.Tensor, atoms: torch.Tensor, alpha: float) -> torch.Tensor:
    """Vectorized Conditional Value at Risk for batched categorical distributions."""

    if probs.dim() != 2:
        raise ValueError("'probs' must be a 2D tensor")

    batch_size, num_atoms = probs.shape
    if atoms.numel() != num_atoms:
        raise ValueError("'atoms' length must match probability dimension")

    device = probs.device
    dtype = torch.float32

    sorted_atoms, sort_indices = torch.sort(atoms.to(dtype=dtype, device=device))
    expanded_indices = sort_indices.view(1, -1).expand(batch_size, -1)
    sorted_probs = torch.gather(probs.to(dtype=dtype), 1, expanded_indices)
    cumulative_probs = torch.cumsum(sorted_probs, dim=1)

    alpha_tensor = torch.full((batch_size, 1), alpha, dtype=dtype, device=device)
    var_indices = torch.searchsorted(cumulative_probs.detach(), alpha_tensor).clamp(max=num_atoms - 1)

    atom_positions = torch.arange(num_atoms, device=device).view(1, -1)
    tail_mask = atom_positions < var_indices
    masked_probs = sorted_probs * tail_mask.to(dtype)
    tail_expectation = torch.sum(masked_probs * sorted_atoms.view(1, -1), dim=1)

    prev_indices = (var_indices - 1).clamp(min=0)
    prev_cum = torch.where(
        var_indices > 0,
        torch.gather(cumulative_probs, 1, prev_indices),
        torch.zeros_like(alpha_tensor),
    ).squeeze(1)

    weight_on_var = (alpha_tensor.squeeze(1) - prev_cum).clamp(min=0.0)
    var_values = sorted_atoms[var_indices.squeeze(1)]

    cvar = (tail_expectation + weight_on_var * var_values) / (alpha + 1e-8)
    return cvar


class DistributionalPPO(RecurrentPPO):
    """Distributional PPO with CVaR regularisation and entropy scheduling."""

    def __init__(
        self,
        policy: Union[str, Type[RecurrentActorCriticPolicy]],
        env: Union[VecEnv, str],
        cql_alpha: float = 1.0,
        cql_beta: float = 5.0,
        cvar_alpha: float = 0.05,
        cvar_weight: float = 0.5,
        cvar_cap: Optional[float] = None,
        v_range_ema_alpha: float = 0.01,
        bc_warmup_steps: int = 0,
        bc_decay_steps: int = 0,
        bc_final_coef: Optional[float] = None,
        ent_coef_final: Optional[float] = None,
        ent_coef_decay_steps: int = 0,
        ent_coef_plateau_window: int = 0,
        ent_coef_plateau_tolerance: float = 0.0,
        ent_coef_plateau_min_updates: int = 0,
        kl_lr_decay: float = 0.5,
        kl_epoch_decay: float = 0.5,
        kl_lr_scale_min: float = 0.1,
        use_torch_compile: bool = False,
        **kwargs: Any,
    ) -> None:
        self._last_lstm_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._last_rollout_entropy: float = 0.0
        self._update_calls: int = 0

        super().__init__(policy=policy, env=env, **kwargs)

        self.cql_alpha = float(cql_alpha)
        self.cql_beta = float(cql_beta)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_weight = float(cvar_weight)
        if cvar_cap is not None and cvar_cap <= 0.0:
            raise ValueError("'cvar_cap' must be positive when provided")
        self.cvar_cap = float(cvar_cap) if cvar_cap is not None else None

        self.v_range_ema_alpha = float(v_range_ema_alpha)
        self.running_v_min = 0.0
        self.running_v_max = 0.0
        self.v_range_initialized = False

        self.bc_warmup_steps = max(0, int(bc_warmup_steps))
        self.bc_decay_steps = max(0, int(bc_decay_steps))
        self.bc_initial_coef = float(self.cql_alpha)
        self.bc_final_coef = float(bc_final_coef) if bc_final_coef is not None else 0.0
        self._current_bc_coef = float(self.bc_initial_coef)

        self.ent_coef_initial = float(self.ent_coef)
        self.ent_coef_final = (
            float(ent_coef_final) if ent_coef_final is not None else float(self.ent_coef_initial)
        )
        self.ent_coef_decay_steps = max(0, int(ent_coef_decay_steps))
        self.entropy_plateau_window = max(0, int(ent_coef_plateau_window))
        self.entropy_plateau_tolerance = float(ent_coef_plateau_tolerance)
        self.entropy_plateau_min_updates = max(0, int(ent_coef_plateau_min_updates))
        self._entropy_window: Optional[deque[tuple[int, float]]] = (
            deque(maxlen=self.entropy_plateau_window) if self.entropy_plateau_window > 0 else None
        )
        self._entropy_plateau = False
        self._entropy_decay_start_update: Optional[int] = None
        self._last_entropy_slope = 0.0

        self.lr_scheduler = getattr(self.policy, "optimizer_scheduler", None)

        if use_torch_compile and self.device.type == "cuda":
            self.policy = torch.compile(self.policy, mode="reduce-overhead")

        # --- KL-adaptive training controls ----------------------------------------------------
        self.kl_lr_decay = float(kl_lr_decay)
        if not (0.0 < self.kl_lr_decay < 1.0):
            raise ValueError("'kl_lr_decay' must be in (0, 1)")
        self.kl_epoch_decay = float(kl_epoch_decay)
        if not (0.0 < self.kl_epoch_decay <= 1.0):
            raise ValueError("'kl_epoch_decay' must be in (0, 1]")
        self._kl_min_lr = 1e-6
        self._kl_lr_scale = 1.0
        self._kl_lr_scale_min = float(kl_lr_scale_min)
        if not (0.0 < self._kl_lr_scale_min <= 1.0):
            raise ValueError("'kl_lr_scale_min' must be in (0, 1]")
        self._base_lr_schedule = self.lr_schedule

        def _scaled_lr_schedule(progress_remaining: float) -> float:
            base_lr = self._base_lr_schedule(progress_remaining)
            scaled_lr = base_lr * self._kl_lr_scale
            return float(max(scaled_lr, self._kl_min_lr))

        self.lr_schedule = _scaled_lr_schedule
        self._base_n_epochs = max(1, int(self.n_epochs))
        self._kl_epoch_factor = 1.0
        self._kl_epoch_factor_min = 1.0 / float(self._base_n_epochs)
        self._kl_base_param_lrs: list[float] = []
        self._refresh_kl_base_lrs()

    def _refresh_kl_base_lrs(self) -> None:
        """Cache optimiser base LRs before KL scaling is applied."""

        optimizer = getattr(self.policy, "optimizer", None)
        if optimizer is None:
            self._kl_base_param_lrs = []
            return

        scale = float(getattr(self, "_kl_lr_scale", 1.0))
        if scale <= 0.0:
            scale = 1.0

        base_lrs: list[float] = []
        for group in getattr(optimizer, "param_groups", []):
            current_lr = float(group.get("lr", 0.0))
            base_lr = current_lr / scale if scale != 0.0 else current_lr
            group["_kl_base_lr"] = base_lr
            base_lrs.append(base_lr)

        self._kl_base_param_lrs = base_lrs

    def _apply_lr_decay(self, requested_decay: float) -> float:
        """Reduce optimizer LR by ``requested_decay`` (capped by ``_kl_lr_scale_min``)."""

        if requested_decay <= 0.0:
            return 1.0

        previous_scale = self._kl_lr_scale
        proposed_scale = previous_scale * requested_decay
        if proposed_scale < self._kl_lr_scale_min:
            self._kl_lr_scale = self._kl_lr_scale_min
        else:
            self._kl_lr_scale = proposed_scale

        if previous_scale <= 0.0:
            actual_decay = 1.0
        else:
            actual_decay = self._kl_lr_scale / previous_scale

        for param_group in self.policy.optimizer.param_groups:
            new_lr = max(param_group["lr"] * actual_decay, self._kl_min_lr)
            param_group["lr"] = new_lr
            if "initial_lr" in param_group:
                param_group["initial_lr"] = max(param_group["initial_lr"] * actual_decay, self._kl_min_lr)

        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, "base_lrs"):
            self.lr_scheduler.base_lrs = [
                max(lr * actual_decay, self._kl_min_lr) for lr in self.lr_scheduler.base_lrs
            ]

        self._refresh_kl_base_lrs()
        return actual_decay

    def _apply_epoch_decay(self) -> float:
        """Shrink effective epoch multiplier respecting minimum of one epoch."""

        previous_factor = self._kl_epoch_factor
        proposed_factor = previous_factor * self.kl_epoch_decay
        proposed_factor = min(proposed_factor, 1.0)
        if proposed_factor < self._kl_epoch_factor_min:
            self._kl_epoch_factor = self._kl_epoch_factor_min
        else:
            self._kl_epoch_factor = proposed_factor

        if previous_factor <= 0.0:
            return 1.0
        return self._kl_epoch_factor / previous_factor

    def _handle_kl_divergence(self, approx_kl: float) -> tuple[float, float]:
        """React to KL overshoot by reducing LR and future epoch budget."""

        lr_decay = self._apply_lr_decay(self.kl_lr_decay)
        epoch_decay = self._apply_epoch_decay()
        self.logger.record("train/kl_last_exceeded", approx_kl)
        self.logger.record("train/kl_lr_decay_applied", lr_decay)
        self.logger.record("train/kl_epoch_decay_applied", epoch_decay)
        return lr_decay, epoch_decay

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return self.policy.parameters(recurse)

    def _update_bc_coef(self) -> float:
        if self.bc_decay_steps <= 0:
            self._current_bc_coef = self.bc_initial_coef
            return self._current_bc_coef

        timesteps_after_warmup = max(0, self.num_timesteps - self.bc_warmup_steps)
        if timesteps_after_warmup <= 0:
            self._current_bc_coef = self.bc_initial_coef
            return self._current_bc_coef

        decay_progress = min(1.0, timesteps_after_warmup / max(1, self.bc_decay_steps))
        self._current_bc_coef = (
            self.bc_initial_coef + (self.bc_final_coef - self.bc_initial_coef) * decay_progress
        )
        return self._current_bc_coef

    def _update_ent_coef(self, update_index: int) -> float:
        if self._entropy_decay_start_update is None:
            self.ent_coef = float(self.ent_coef_initial)
            return self.ent_coef

        if self.ent_coef_decay_steps <= 0:
            self.ent_coef = float(self.ent_coef_final)
            return self.ent_coef

        steps_since_start = max(0, update_index - self._entropy_decay_start_update)
        progress = min(1.0, steps_since_start / float(self.ent_coef_decay_steps))
        self.ent_coef = float(
            self.ent_coef_initial + (self.ent_coef_final - self.ent_coef_initial) * progress
        )
        return self.ent_coef

    def _maybe_update_entropy_schedule(self, update_index: int, avg_entropy: float) -> None:
        if self._entropy_window is None:
            self._last_entropy_slope = 0.0
            return

        self._entropy_window.append((update_index, avg_entropy))
        if len(self._entropy_window) < 2:
            self._last_entropy_slope = 0.0
            return

        xs = torch.tensor([item[0] for item in self._entropy_window], dtype=torch.float32)
        ys = torch.tensor([item[1] for item in self._entropy_window], dtype=torch.float32)
        xs_centered = xs - xs.mean()
        ys_centered = ys - ys.mean()
        denom = torch.sum(xs_centered * xs_centered).item()
        if denom <= 1e-8:
            self._last_entropy_slope = 0.0
        else:
            self._last_entropy_slope = float(torch.sum(xs_centered * ys_centered).item() / (denom + 1e-8))

        if (
            not self._entropy_plateau
            and len(self._entropy_window) == self.entropy_plateau_window
            and update_index >= self.entropy_plateau_min_updates
            and abs(self._last_entropy_slope) <= self.entropy_plateau_tolerance
        ):
            self._entropy_plateau = True
            self._entropy_decay_start_update = update_index

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RecurrentRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        if not isinstance(rollout_buffer, RecurrentRolloutBuffer):
            raise TypeError(
                "DistributionalPPO requires a RecurrentRolloutBuffer for recurrent policies"
            )
        assert self._last_obs is not None, "No previous observation was provided"

        if self._last_lstm_states is None:
            init_states = self.policy.recurrent_initial_state
            self._last_lstm_states = (init_states[0].to(self.device), init_states[1].to(self.device))

        self.policy.set_training_mode(False)

        vec_normalize_env: Optional[VecNormalize] = None
        for candidate_env in (env, getattr(self, "env", None)):
            if candidate_env is None:
                continue
            if isinstance(candidate_env, VecNormalize):
                vec_normalize_env = candidate_env
            else:
                try:
                    vec_normalize_env = unwrap_vec_normalize(candidate_env)
                except ValueError:
                    vec_normalize_env = None
            if vec_normalize_env is not None:
                break

        if vec_normalize_env is not None and getattr(vec_normalize_env, "norm_reward", False):
            raise AssertionError("VecNormalize reward normalization must be disabled to recover raw ΔPnL.")

        entropy_loss_total = 0.0
        entropy_loss_count = 0

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                obs_tensor = self.policy.obs_to_tensor(self._last_obs)[0]
                episode_starts = torch.as_tensor(
                    self._last_episode_starts, dtype=torch.float32, device=self.device
                )
                actions, _, log_probs, self._last_lstm_states = self.policy.forward(
                    obs_tensor, self._last_lstm_states, episode_starts
                )
                value_logits = self.policy.last_value_logits

            if value_logits is None:
                raise RuntimeError("Policy did not cache value logits during forward pass")

            probs = torch.softmax(value_logits, dim=1)
            scalar_values = (probs * self.policy.atoms).sum(dim=1, keepdim=True).detach()

            actions_np = actions.cpu().numpy()
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions_np, self.action_space.low, self.action_space.high)
            else:
                clipped_actions = actions_np

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            raw_rewards = rewards
            if vec_normalize_env is not None and hasattr(vec_normalize_env, "get_original_reward"):
                raw_rewards = vec_normalize_env.get_original_reward()
            raw_rewards = np.asarray(raw_rewards, dtype=np.float32)
            if raw_rewards.size > 0:
                self.logger.record("rollout/raw_reward_mean", float(np.mean(raw_rewards)))

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                raw_rewards,
                self._last_episode_starts,
                scalar_values.squeeze(-1),
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            entropy_loss_total += float(-log_probs.mean().item())
            entropy_loss_count += 1

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            obs_tensor = self.policy.obs_to_tensor(new_obs)[0]
            episode_starts = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            _, _, _, _ = self.policy.forward(obs_tensor, self._last_lstm_states, episode_starts)
            last_value_logits = self.policy.last_value_logits

        if last_value_logits is None:
            raise RuntimeError("Policy did not cache value logits during terminal forward pass")

        last_probs = torch.softmax(last_value_logits, dim=1)
        last_scalar_values = (last_probs * self.policy.atoms).sum(dim=1)

        rollout_buffer.compute_returns_and_advantage(last_values=last_scalar_values, dones=dones)
        callback.on_rollout_end()

        if entropy_loss_count > 0:
            self._last_rollout_entropy = entropy_loss_total / float(entropy_loss_count)
            self.logger.record("rollout/policy_entropy", self._last_rollout_entropy)
        else:
            self._last_rollout_entropy = 0.0

        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        self._refresh_kl_base_lrs()
        clip_range = self.clip_range(self._current_progress_remaining)

        current_update = self._update_calls
        self._update_ent_coef(current_update)

        returns_tensor = torch.as_tensor(
            self.rollout_buffer.returns, device=self.device, dtype=torch.float32
        ).flatten()

        policy_clip_limit = getattr(self.policy, "value_clip_limit", None)
        if policy_clip_limit is not None:
            min_half_range = float(policy_clip_limit)
            if not math.isfinite(min_half_range) or min_half_range < 0.0:
                raise RuntimeError(
                    f"Invalid 'value_clip_limit' for distributional value head: {policy_clip_limit}"
                )
        else:
            with torch.no_grad():
                min_half_range = float(torch.max(torch.abs(self.policy.atoms)).item())
        if not math.isfinite(min_half_range):
            min_half_range = 0.0

        v_min = float(torch.min(returns_tensor).item())
        v_max = float(torch.max(returns_tensor).item())
        if not math.isfinite(v_min) or not math.isfinite(v_max):
            raise RuntimeError(
                f"Encountered non-finite return bounds when updating value support: {v_min}, {v_max}"
            )

        if v_max <= v_min:
            center = v_min
            half_range = 0.0
        else:
            center = 0.5 * (v_max + v_min)
            half_range = 0.5 * (v_max - v_min)

        half_range = max(half_range, min_half_range)
        padding = max(1e-6, half_range * 0.05)
        half_range += padding
        v_min = center - half_range
        v_max = center + half_range

        if v_max <= v_min:
            raise RuntimeError(
                f"Failed to compute a valid value support range: v_min={v_min}, v_max={v_max}"
            )

        if not self.v_range_initialized:
            self.running_v_min = v_min
            self.running_v_max = v_max
            self.v_range_initialized = True
        else:
            alpha = self.v_range_ema_alpha
            self.running_v_min = float((1.0 - alpha) * self.running_v_min + alpha * v_min)
            self.running_v_max = float((1.0 - alpha) * self.running_v_max + alpha * v_max)

        if self.running_v_max <= self.running_v_min:
            self.running_v_min = v_min
            self.running_v_max = v_max

        self.policy.update_atoms(self.running_v_min, self.running_v_max)

        self.logger.record("train/v_min", self.running_v_min)
        self.logger.record("train/v_max", self.running_v_max)

        if not (0.0 < float(self.gamma) <= 1.0):
            raise RuntimeError(f"Invalid discount factor 'gamma': {self.gamma}")
        if not (0.0 <= float(self.gae_lambda) <= 1.0):
            raise RuntimeError(f"Invalid GAE lambda 'gae_lambda': {self.gae_lambda}")

        self.logger.record("train/gamma", float(self.gamma))
        self.logger.record("train/gae_lambda", float(self.gae_lambda))

        bc_coef = self._update_bc_coef()
        self.logger.record("train/policy_bc_coef", bc_coef)

        entropy_loss_total = 0.0
        entropy_loss_count = 0
        approx_kl_divs: list[float] = []
        clamped_return_batches: list[torch.Tensor] = []
        mean_value_batches: list[torch.Tensor] = []
        last_optimizer_lr: Optional[float] = None
        last_scheduler_lr: Optional[float] = None

        policy_loss_value = 0.0
        policy_loss_ppo_value = 0.0
        policy_loss_bc_value = 0.0
        policy_loss_bc_weighted_value = 0.0
        critic_loss_value = 0.0
        cvar_raw_value = 0.0
        cvar_loss_value = 0.0
        cvar_term_value = 0.0
        total_loss_value = 0.0

        adv_mean_accum = 0.0
        adv_std_accum = 0.0
        adv_batch_count = 0

        value_logits_final: Optional[torch.Tensor] = None

        base_n_epochs = max(1, int(self.n_epochs))
        if base_n_epochs != self._base_n_epochs:
            self._base_n_epochs = base_n_epochs
            self._kl_epoch_factor_min = 1.0 / float(self._base_n_epochs)
            self._kl_epoch_factor = min(max(self._kl_epoch_factor, self._kl_epoch_factor_min), 1.0)

        effective_n_epochs = max(1, int(round(self._base_n_epochs * self._kl_epoch_factor)))
        kl_early_stop_triggered = False
        epochs_completed = 0

        rollout_n_envs = int(getattr(self.rollout_buffer, "n_envs", 1)) or 1

        def _prepare_minibatch_iterator(desired_batch_size: Optional[int]):
            batch_size_local = desired_batch_size
            if batch_size_local is None or batch_size_local <= 0:
                batch_size_local = rollout_n_envs
            batch_size_local = int(batch_size_local)
            iterator = self.rollout_buffer.get(batch_size_local)
            try:
                first_minibatch = next(iterator)
            except StopIteration:
                return None, batch_size_local
            return itertools.chain([first_minibatch], iterator), batch_size_local

        def _fallback_batch_size() -> int:
            buffer_size_local = int(getattr(self.rollout_buffer, "buffer_size", 0))
            if buffer_size_local <= 0:
                return rollout_n_envs
            fallback_local = buffer_size_local
            divisor = max(1, rollout_n_envs)
            if fallback_local % divisor != 0:
                fallback_local = (fallback_local // divisor) * divisor
                if fallback_local <= 0:
                    fallback_local = divisor
            return fallback_local

        for _ in range(effective_n_epochs):
            minibatch_iterator, actual_batch_size = _prepare_minibatch_iterator(self.batch_size)
            if minibatch_iterator is None:
                fallback_batch_size = _fallback_batch_size()
                minibatch_iterator, actual_batch_size = _prepare_minibatch_iterator(fallback_batch_size)
                if minibatch_iterator is not None and fallback_batch_size != self.batch_size:
                    self.logger.record("warn/train_batch_size_adjusted", float(fallback_batch_size))

            if minibatch_iterator is None:
                self.logger.record("warn/empty_rollout_buffer", 1.0)
                break

            epochs_completed += 1
            self.logger.record("train/actual_batch_size", float(actual_batch_size))

            for rollout_data in minibatch_iterator:
                _values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                advantages = rollout_data.advantages
                with torch.no_grad():
                    adv_mean_tensor = advantages.mean()
                    adv_std_tensor = advantages.std(unbiased=False)
                    adv_std_tensor_clamped = torch.clamp(adv_std_tensor, min=1e-8)
                adv_mean = float(adv_mean_tensor.item())
                adv_std = float(adv_std_tensor.item())
                adv_mean_accum += adv_mean
                adv_std_accum += adv_std
                adv_batch_count += 1

                advantages = (advantages - adv_mean_tensor) / adv_std_tensor_clamped

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss_ppo = -torch.min(policy_loss_1, policy_loss_2).mean()

                if bc_coef <= 0.0:
                    policy_loss_bc = advantages.new_zeros(())
                    policy_loss_bc_weighted = policy_loss_bc
                else:
                    with torch.no_grad():
                        weights = torch.exp(advantages / self.cql_beta)
                        weights = torch.clamp(weights, max=100.0)
                    policy_loss_bc = (-log_prob * weights).mean()
                    policy_loss_bc_weighted = policy_loss_bc * bc_coef

                policy_loss = policy_loss_ppo + policy_loss_bc_weighted

                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                mean_entropy = -entropy_loss.item()
                entropy_loss_total += mean_entropy
                entropy_loss_count += 1

                value_logits = self.policy.last_value_logits
                if value_logits is None:
                    raise RuntimeError(
                        "Policy did not cache value logits during training forward pass"
                    )

                value_logits_fp32 = value_logits.to(dtype=torch.float32)
                with torch.no_grad():
                    target_returns = rollout_data.returns.to(dtype=torch.float32)
                    delta_z = (self.policy.v_max - self.policy.v_min) / float(self.policy.num_atoms - 1)
                    clamped_targets = target_returns.clamp(self.policy.v_min, self.policy.v_max)
                    b = (clamped_targets - self.policy.v_min) / (delta_z + 1e-8)
                    lower_bound = b.floor().long().clamp(min=0, max=self.policy.num_atoms - 1)
                    upper_bound = b.ceil().long().clamp(min=0, max=self.policy.num_atoms - 1)

                    same_bounds = lower_bound == upper_bound
                    lower_bound = torch.where(
                        same_bounds & (lower_bound > 0), lower_bound - 1, lower_bound
                    )
                    upper_bound = torch.where(
                        same_bounds & (upper_bound < self.policy.num_atoms - 1),
                        upper_bound + 1,
                        upper_bound,
                    )

                    target_distribution = torch.zeros_like(value_logits_fp32)
                    lower_prob = (upper_bound.to(torch.float32) - b).clamp(min=0.0)
                    upper_prob = (b - lower_bound.to(torch.float32)).clamp(min=0.0)
                    target_distribution.scatter_add_(1, lower_bound.view(-1, 1), lower_prob.view(-1, 1))
                    target_distribution.scatter_add_(1, upper_bound.view(-1, 1), upper_prob.view(-1, 1))

                pred_probs_fp32 = torch.softmax(value_logits_fp32, dim=1).clamp(min=1e-8, max=1.0)
                log_predictions = torch.log(pred_probs_fp32)
                critic_loss = -(target_distribution * log_predictions).sum(dim=1).mean()

                with torch.no_grad():
                    mean_values_batch = (pred_probs_fp32 * self.policy.atoms).sum(dim=1, keepdim=True)
                    clamped_return_batches.append(clamped_targets.detach())
                    mean_value_batches.append(mean_values_batch.detach())

                predicted_cvar = calculate_cvar(pred_probs_fp32, self.policy.atoms, self.cvar_alpha)
                cvar_raw = predicted_cvar.mean()
                cvar_loss = -cvar_raw
                cvar_term = self.cvar_weight * cvar_loss
                if self.cvar_cap is not None:
                    cvar_term = torch.clamp(cvar_term, min=-self.cvar_cap, max=self.cvar_cap)

                loss = (
                    policy_loss.to(dtype=torch.float32)
                    + self.ent_coef * entropy_loss.to(dtype=torch.float32)
                    + self.vf_coef * critic_loss
                    + cvar_term
                )

                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                max_grad_norm = (
                    0.5 if self.max_grad_norm is None else float(self.max_grad_norm)
                )
                if max_grad_norm <= 0.0:
                    self.logger.record("warn/max_grad_norm_nonpos", float(max_grad_norm))
                    max_grad_norm = 0.5
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), max_grad_norm
                )
                grad_norm_value = (
                    total_grad_norm.item()
                    if isinstance(total_grad_norm, torch.Tensor)
                    else float(total_grad_norm)
                )
                self.logger.record("train/grad_norm_pre_clip", float(grad_norm_value))
                self.logger.record("train/max_grad_norm_used", float(max_grad_norm))

                if hasattr(self, "_kl_lr_scale"):
                    scale = float(self._kl_lr_scale)
                    for group in self.policy.optimizer.param_groups:
                        base_lr = float(group.get("_kl_base_lr", group.get("lr", 0.0)))
                        scaled_lr = max(base_lr * scale, self._kl_min_lr)
                        group["lr"] = scaled_lr
                        if "initial_lr" in group:
                            group["initial_lr"] = scaled_lr

                self.policy.optimizer.step()

                if len(self.policy.optimizer.param_groups) > 0:
                    lrs = [float(g["lr"]) for g in self.policy.optimizer.param_groups]
                    last_optimizer_lr = lrs[0]
                    self.logger.record("train/optimizer_lr", last_optimizer_lr)
                    self.logger.record("train/optimizer_lr_min", min(lrs))
                    self.logger.record("train/optimizer_lr_max", max(lrs))

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    get_last_lr = getattr(self.lr_scheduler, "get_last_lr", None)
                    if callable(get_last_lr):
                        try:
                            scheduler_lrs = get_last_lr()
                        except TypeError:
                            scheduler_lrs = None
                        if scheduler_lrs:
                            last_scheduler_lr = float(scheduler_lrs[0])
                            self.logger.record("train/scheduler_lr", last_scheduler_lr)
                    self._refresh_kl_base_lrs()

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl = (
                        ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    )
                    approx_kl_divs.append(approx_kl)

                if (
                    self.target_kl is not None
                    and self.target_kl > 0.0
                    and approx_kl > float(self.target_kl)
                ):
                    kl_early_stop_triggered = True
                    self._handle_kl_divergence(approx_kl)
                    break

                policy_loss_value = float(policy_loss.item())
                policy_loss_ppo_value = float(policy_loss_ppo.item())
                policy_loss_bc_value = float(policy_loss_bc.item())
                policy_loss_bc_weighted_value = float(policy_loss_bc_weighted.item())
                critic_loss_value = float(critic_loss.item())
                cvar_raw_value = float(cvar_raw.item())
                cvar_loss_value = float(cvar_loss.item())
                cvar_term_value = float(cvar_term.item())
                total_loss_value = float(loss.item())

                value_logits_final = value_logits_fp32.detach()

            if kl_early_stop_triggered:
                break

        self._n_updates += epochs_completed
        self._update_calls += 1

        avg_policy_entropy = (
            entropy_loss_total / float(entropy_loss_count) if entropy_loss_count > 0 else self._last_rollout_entropy
        )
        self._maybe_update_entropy_schedule(current_update, avg_policy_entropy)

        if value_logits_final is None:
            cached_logits = getattr(self.policy, "last_value_logits", None)
            if cached_logits is not None:
                value_logits_final = cached_logits.detach().to(dtype=torch.float32)

        if value_logits_final is None:
            raise RuntimeError("No value logits captured during training loop")
        if len(clamped_return_batches) == 0 or len(mean_value_batches) == 0:
            rollout_returns = torch.as_tensor(
                self.rollout_buffer.returns, device=self.device, dtype=torch.float32
            ).clamp(self.policy.v_min, self.policy.v_max)
            y_true_tensor = rollout_returns.reshape(-1, 1)
            with torch.no_grad():
                pred_probs = torch.softmax(value_logits_final, dim=1)
                y_pred_tensor = (pred_probs * self.policy.atoms).sum(dim=1, keepdim=True)
        else:
            y_true_tensor = torch.cat([t.reshape(-1, 1) for t in clamped_return_batches], dim=0)
            y_pred_tensor = torch.cat([t.reshape(-1, 1) for t in mean_value_batches], dim=0)

        if y_true_tensor.numel() == 0 or y_pred_tensor.numel() == 0:
            explained_var = 0.0
        else:
            if y_true_tensor.shape != y_pred_tensor.shape:
                min_elems = min(y_true_tensor.shape[0], y_pred_tensor.shape[0])
                y_true_tensor = y_true_tensor[:min_elems]
                y_pred_tensor = y_pred_tensor[:min_elems]

            y_true_np = y_true_tensor.flatten().detach().cpu().numpy()
            y_pred_np = y_pred_tensor.flatten().detach().cpu().numpy()
            explained_var = np.nan_to_num(safe_explained_variance(y_true_np, y_pred_np))

        bc_ratio = abs(policy_loss_bc_weighted_value) / (abs(policy_loss_ppo_value) + 1e-8)

        self.logger.record("train/policy_loss", policy_loss_value)
        self.logger.record("train/policy_loss_ppo", policy_loss_ppo_value)
        self.logger.record("train/policy_loss_bc", policy_loss_bc_value)
        self.logger.record("train/policy_loss_bc_weighted", policy_loss_bc_weighted_value)
        self.logger.record("train/policy_bc_vs_ppo_ratio", bc_ratio)
        self.logger.record("train/critic_loss", critic_loss_value)
        self.logger.record("train/cvar_raw", cvar_raw_value)
        self.logger.record("train/cvar_loss", cvar_loss_value)
        self.logger.record("train/cvar_term", cvar_term_value)
        if self.cvar_cap is not None:
            self.logger.record("train/cvar_cap", self.cvar_cap)

        self.logger.record("train/entropy_loss", -avg_policy_entropy)
        self.logger.record("train/policy_entropy", avg_policy_entropy)
        self.logger.record("train/policy_entropy_slope", self._last_entropy_slope)
        self.logger.record("train/entropy_plateau", float(self._entropy_plateau))
        decay_start = self._entropy_decay_start_update if self._entropy_decay_start_update is not None else -1
        self.logger.record("train/entropy_decay_start_update", float(decay_start))

        self.logger.record("train/ent_coef", float(self.ent_coef))
        self.logger.record("train/ent_coef_initial", float(self.ent_coef_initial))
        self.logger.record("train/ent_coef_final", float(self.ent_coef_final))

        if len(approx_kl_divs) > 0:
            self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
            self.logger.record("train/approx_kl_median", float(np.median(approx_kl_divs)))
        if last_optimizer_lr is not None:
            self.logger.record("train/optimizer_lr", last_optimizer_lr)
        if last_scheduler_lr is not None:
            self.logger.record("train/scheduler_lr", last_scheduler_lr)
        self.logger.record("train/loss", total_loss_value)
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/clip_range", float(clip_range))
        if adv_batch_count > 0:
            self.logger.record("train/adv_mean", adv_mean_accum / adv_batch_count)
            self.logger.record("train/adv_std", adv_std_accum / adv_batch_count)
        self.logger.record("train/n_epochs_effective", float(effective_n_epochs))
        self.logger.record("train/n_epochs_completed", float(epochs_completed))
        self.logger.record("train/kl_early_stop", float(1.0 if kl_early_stop_triggered else 0.0))
        if self.target_kl is not None and self.target_kl > 0.0:
            self.logger.record("train/target_kl", float(self.target_kl))
            self.logger.record("train/kl_lr_scale", float(self._kl_lr_scale))
            self.logger.record("train/kl_epoch_factor", float(self._kl_epoch_factor))

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Union[BaseCallback, Sequence[BaseCallback]]] = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "DistributionalPPO":
        callbacks: list[BaseCallback] = []

        if callback is not None:
            if isinstance(callback, BaseCallback):
                callbacks.append(callback)
            else:
                callbacks.extend(callback)

        callback_for_super: Optional[BaseCallback]
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                warn=True,
            )
            callbacks.append(eval_callback)
            callback_for_super = (
                callbacks[0]
                if len(callbacks) == 1
                else CallbackList(callbacks)
            )
        else:
            if not callbacks:
                callback_for_super = None
            elif len(callbacks) == 1:
                callback_for_super = callbacks[0]
            else:
                callback_for_super = CallbackList(callbacks)

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback_for_super,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
