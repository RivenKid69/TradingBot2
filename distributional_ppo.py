import itertools
import math
from collections import deque
from typing import Any, Iterable, Optional, Sequence, Type, Union

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

    @staticmethod
    def _coerce_value_target_scale(value_target_scale: Union[str, float]) -> float:
        if isinstance(value_target_scale, str):
            normalized = value_target_scale.strip().lower()
            if normalized in {"percent", "percents", "percentage", "%"}:
                scale = 100.0
            elif normalized in {"bps", "basis_points", "basis-point", "basis points"}:
                scale = 10000.0
            else:
                raise ValueError(
                    "Unsupported value_target_scale string. Use 'percent', 'bps', or provide a positive float."
                )
        else:
            try:
                scale = float(value_target_scale)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid value_target_scale={value_target_scale!r}; expected positive float or recognised alias."
                ) from exc

        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"'value_target_scale' must be a positive finite value, got {value_target_scale!r}"
            )
        return scale

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
        value_target_scale: Union[str, float] = "percent",
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
        kl_lr_scale_min: float = 0.01,
        kl_exceed_stop_fraction: float = 0.4,
        kl_penalty_beta: float = 0.0,
        kl_penalty_beta_min: float = 0.0,
        kl_penalty_beta_max: float = 0.1,
        kl_penalty_increase: float = 1.5,
        kl_penalty_decrease: float = 0.75,
        ppo_clip_range: float = 0.05,
        use_torch_compile: bool = False,
        microbatch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._last_lstm_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._last_rollout_entropy: float = 0.0
        self._update_calls: int = 0

        if not math.isfinite(kl_lr_scale_min):
            raise ValueError("'kl_lr_scale_min' must be finite")
        kl_lr_scale_min_requested: Optional[float] = None
        if abs(float(kl_lr_scale_min) - 0.01) > 1e-12:
            kl_lr_scale_min_requested = float(kl_lr_scale_min)

        kwargs_local = dict(kwargs)
        clip_range_value = float(ppo_clip_range)
        if not math.isfinite(clip_range_value) or clip_range_value <= 0.0:
            raise ValueError("'ppo_clip_range' must be a positive finite value")
        kwargs_local["clip_range"] = clip_range_value
        kwargs_local["target_kl"] = 0.5

        self.cql_alpha = float(cql_alpha)
        self.cql_beta = float(cql_beta)
        self.cvar_alpha = float(cvar_alpha)
        self.cvar_weight = float(cvar_weight)
        if cvar_cap is not None and cvar_cap <= 0.0:
            raise ValueError("'cvar_cap' must be positive when provided")
        self.cvar_cap = float(cvar_cap) if cvar_cap is not None else None

        self.v_range_ema_alpha = float(v_range_ema_alpha)
        if not (0.0 < self.v_range_ema_alpha <= 1.0):
            raise ValueError("'v_range_ema_alpha' must be in (0, 1]")

        self.value_target_scale = self._coerce_value_target_scale(value_target_scale)
        self._value_clip_limit_scaled: Optional[float] = None

        super().__init__(policy=policy, env=env, **kwargs_local)

        self._configure_gradient_accumulation(
            microbatch_size=microbatch_size,
            grad_steps=gradient_accumulation_steps,
        )

        self._fixed_clip_range = clip_range_value
        self.clip_range = lambda _: self._fixed_clip_range
        self.target_kl = 0.5

        if kl_lr_scale_min_requested is not None:
            self.logger.record("warn/kl_lr_scale_min_requested", float(kl_lr_scale_min_requested))
            self.logger.record("warn/kl_lr_scale_min_effective", 0.01)

    def _configure_gradient_accumulation(
        self,
        microbatch_size: Optional[int],
        grad_steps: Optional[int],
    ) -> None:
        batch_size = int(self.batch_size)
        if batch_size <= 0:
            raise ValueError("'batch_size' must be positive to configure gradient accumulation")

        grad_steps_local = None if grad_steps is None else int(grad_steps)
        micro_size_local = None if microbatch_size is None else int(microbatch_size)

        if grad_steps_local is not None and grad_steps_local <= 0:
            raise ValueError("'gradient_accumulation_steps' must be a positive integer")
        if micro_size_local is not None and micro_size_local <= 0:
            raise ValueError("'microbatch_size' must be a positive integer")

        if grad_steps_local is None and micro_size_local is None:
            micro_size_local = batch_size
            grad_steps_local = 1
        elif grad_steps_local is None:
            if batch_size % micro_size_local != 0:
                raise ValueError("'microbatch_size' must evenly divide batch_size")
            grad_steps_local = batch_size // micro_size_local
        elif micro_size_local is None:
            if batch_size % grad_steps_local != 0:
                raise ValueError("'gradient_accumulation_steps' must evenly divide batch_size")
            micro_size_local = batch_size // grad_steps_local
        else:
            if micro_size_local * grad_steps_local != batch_size:
                if batch_size % micro_size_local != 0 or batch_size // micro_size_local != grad_steps_local:
                    raise ValueError(
                        "microbatch_size * gradient_accumulation_steps must equal batch_size"
                    )

        self._microbatch_size = int(micro_size_local)
        self._grad_accumulation_steps = int(grad_steps_local)


        self.running_v_min = 0.0
        self.running_v_max = 0.0
        self.v_range_initialized = False

        clip_limit_unscaled = getattr(self.policy, "value_clip_limit", None)
        self._value_clip_limit_unscaled: Optional[float]
        if clip_limit_unscaled is None:
            self._value_clip_limit_unscaled = None

            self._value_clip_limit_scaled = None


        else:
            clip_limit_unscaled_f = float(clip_limit_unscaled)
            if clip_limit_unscaled_f <= 0.0 or not math.isfinite(clip_limit_unscaled_f):
                raise ValueError(
                    f"Invalid 'value_clip_limit' for distributional value head: {clip_limit_unscaled}"
                )
            self._value_clip_limit_unscaled = clip_limit_unscaled_f

            self._value_clip_limit_scaled = clip_limit_unscaled_f * self.value_target_scale




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
        self._kl_lr_scale_min = 0.01
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

        self.kl_exceed_stop_fraction = float(kl_exceed_stop_fraction)
        if not (0.0 <= self.kl_exceed_stop_fraction <= 1.0):
            raise ValueError("'kl_exceed_stop_fraction' must be within [0, 1]")

        beta_min = max(0.0, float(kl_penalty_beta_min))
        beta_max = max(beta_min, float(kl_penalty_beta_max))
        beta_initial = float(kl_penalty_beta)
        if not math.isfinite(beta_initial):
            raise ValueError("'kl_penalty_beta' must be finite")
        beta_initial = min(max(beta_initial, beta_min), beta_max)
        self._kl_penalty_beta = beta_initial
        self.kl_penalty_beta_min = beta_min
        self.kl_penalty_beta_max = beta_max
        self.kl_penalty_increase = max(1.0, float(kl_penalty_increase))
        self.kl_penalty_decrease = float(kl_penalty_decrease)
        if not (0.0 < self.kl_penalty_decrease <= 1.0):
            raise ValueError("'kl_penalty_decrease' must be in (0, 1]")

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

    def _adjust_kl_penalty(self, observed_kl: float) -> None:
        """Adapt the KL penalty strength based on observed divergence."""

        if self._kl_penalty_beta <= 0.0:
            return
        if self.target_kl is None or self.target_kl <= 0.0:
            return

        if not math.isfinite(observed_kl):
            return

        if observed_kl > float(self.target_kl):
            updated_beta = self._kl_penalty_beta * self.kl_penalty_increase
            updated_beta = min(updated_beta, self.kl_penalty_beta_max)
        else:
            updated_beta = self._kl_penalty_beta * self.kl_penalty_decrease
            updated_beta = max(updated_beta, self.kl_penalty_beta_min)

        if not math.isfinite(updated_beta):
            updated_beta = self.kl_penalty_beta_max

        self._kl_penalty_beta = updated_beta
        self.logger.record("train/kl_penalty_beta", float(self._kl_penalty_beta))

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

            if self._value_clip_limit_scaled is not None:
                scalar_values = torch.clamp(
                    scalar_values,
                    min=-self._value_clip_limit_scaled,
                    max=self._value_clip_limit_scaled,
                )

            scalar_values = scalar_values / self.value_target_scale
            if self._value_clip_limit_unscaled is not None:
                scalar_values = torch.clamp(
                    scalar_values,
                    min=-self._value_clip_limit_unscaled,
                    max=self._value_clip_limit_unscaled,

                )

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

            scaled_rewards = raw_rewards / self.value_target_scale

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
                scaled_rewards,
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

        if self._value_clip_limit_scaled is not None:
            last_scalar_values = torch.clamp(
                last_scalar_values,
                min=-self._value_clip_limit_scaled,
                max=self._value_clip_limit_scaled,
            )

        last_scalar_values = last_scalar_values / self.value_target_scale
        if self._value_clip_limit_unscaled is not None:
            last_scalar_values = torch.clamp(
                last_scalar_values,
                min=-self._value_clip_limit_unscaled,
                max=self._value_clip_limit_unscaled,
            )

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
        scaled_returns_tensor = returns_tensor * self.value_target_scale

        if self._value_clip_limit_unscaled is not None:

            min_half_range = self._value_clip_limit_unscaled / self.value_target_scale

            min_half_range = self._value_clip_limit_unscaled * self.value_target_scale

        else:
            with torch.no_grad():
                min_half_range = float(torch.max(torch.abs(self.policy.atoms)).item())
        if not math.isfinite(min_half_range):
            min_half_range = 0.0

        if scaled_returns_tensor.numel() == 0:
            v_min = -min_half_range
            v_max = min_half_range
        else:
            quantile_bounds = torch.tensor(
                [0.05, 0.95], device=scaled_returns_tensor.device, dtype=scaled_returns_tensor.dtype
            )
            v_low, v_high = torch.quantile(scaled_returns_tensor, quantile_bounds)
            v_min = float(v_low.item())
            v_max = float(v_high.item())

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

        padding = max(1e-6 / self.value_target_scale, half_range * 0.05)

        padding = max(1e-6 * self.value_target_scale, half_range * 0.05)

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


        running_v_min_unscaled = self.running_v_min * self.value_target_scale
        running_v_max_unscaled = self.running_v_max * self.value_target_scale

        running_v_min_unscaled = self.running_v_min / self.value_target_scale
        running_v_max_unscaled = self.running_v_max / self.value_target_scale

        self.logger.record("train/v_min", running_v_min_unscaled)
        self.logger.record("train/v_max", running_v_max_unscaled)
        self.logger.record("train/v_min_scaled", self.running_v_min)
        self.logger.record("train/v_max_scaled", self.running_v_max)
        self.logger.record("train/value_target_scale", float(self.value_target_scale))
        if self._value_clip_limit_unscaled is not None:
            self.logger.record("train/value_clip_limit", float(self._value_clip_limit_unscaled))

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
        target_return_batches: list[torch.Tensor] = []
        mean_value_batches: list[torch.Tensor] = []
        last_optimizer_lr: Optional[float] = None
        last_scheduler_lr: Optional[float] = None
        kl_exceed_fraction_latest = 0.0

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

        clip_fraction_numer = 0.0
        clip_fraction_denom = 0
        ratio_sum = 0.0
        ratio_sq_sum = 0.0
        ratio_count = 0
        log_prob_sum = 0.0
        log_prob_count = 0
        adv_z_values: list[torch.Tensor] = []
        approx_kl_exceed_count = 0
        minibatches_processed = 0
        kl_penalty_component_total = 0.0
        kl_penalty_component_count = 0

        base_n_epochs = max(1, int(self.n_epochs))
        if base_n_epochs != self._base_n_epochs:
            self._base_n_epochs = base_n_epochs
            self._kl_epoch_factor_min = 1.0 / float(self._base_n_epochs)
            self._kl_epoch_factor = min(max(self._kl_epoch_factor, self._kl_epoch_factor_min), 1.0)

        effective_n_epochs = max(1, int(round(self._base_n_epochs * self._kl_epoch_factor)))
        kl_early_stop_triggered = False
        epochs_completed = 0
        approx_kl_latest = 0.0




        effective_batch_size = int(self.batch_size)
        if effective_batch_size <= 0:
            raise RuntimeError("PPO batch_size must be positive for training")
        microbatch_size_effective = max(
            1, int(getattr(self, "_microbatch_size", effective_batch_size))
        )
        grad_accum_steps = max(
            1, int(getattr(self, "_grad_accumulation_steps", 1))
        )
        if effective_batch_size % microbatch_size_effective != 0:
            raise RuntimeError(
                "Configured batch_size must be divisible by microbatch_size; adjust n_steps, n_envs, or microbatch_size"
            )

        def _prepare_minibatch_iterator() -> tuple[Optional[Iterable[tuple[Any, ...]]], int]:
            microbatches = list(self.rollout_buffer.get(microbatch_size_effective))
            if not microbatches:
                return None, effective_batch_size
            total_micro = len(microbatches)
            if total_micro % grad_accum_steps != 0:
                raise RuntimeError(
                    "Rollout buffer produced incomplete micro-batch bucket; ensure n_steps * n_envs is divisible by batch_size and microbatch_size"
                )

            def _grouped_microbatches() -> Iterable[tuple[Any, ...]]:
                for start_idx in range(0, total_micro, grad_accum_steps):
                    yield tuple(microbatches[start_idx:start_idx + grad_accum_steps])

            return _grouped_microbatches(), effective_batch_size

        for _ in range(effective_n_epochs):
            minibatch_iterator, actual_batch_size = _prepare_minibatch_iterator()
            if minibatch_iterator is None:
                self.logger.record("warn/empty_rollout_buffer", 1.0)
                break

            epochs_completed += 1
            self.logger.record("train/actual_batch_size", float(actual_batch_size))
            self.logger.record("train/microbatch_size", float(microbatch_size_effective))
            self.logger.record("train/grad_accum_steps", float(grad_accum_steps))

            for microbatch_group in minibatch_iterator:
                minibatches_processed += 1
                clip_range = float(self.clip_range(self._current_progress_remaining))
                self.policy.optimizer.zero_grad(set_to_none=True)

                bucket_policy_loss_value = 0.0
                bucket_policy_loss_ppo_value = 0.0
                bucket_policy_loss_bc_value = 0.0
                bucket_policy_loss_bc_weighted_value = 0.0
                bucket_critic_loss_value = 0.0
                bucket_cvar_raw_value = 0.0
                bucket_cvar_loss_value = 0.0
                bucket_cvar_term_value = 0.0
                bucket_total_loss_value = 0.0
                bucket_value_logits_fp32: Optional[torch.Tensor] = None
                approx_kl_weighted_sum = 0.0
                bucket_sample_count = 0

                for rollout_data in microbatch_group:
                    _values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        rollout_data.actions,
                        rollout_data.lstm_states,
                        rollout_data.episode_starts,
                    )

                    advantages = rollout_data.advantages
                    sample_count = int(advantages.shape[0])
                    if sample_count <= 0:
                        continue
                    bucket_sample_count += sample_count
                    weight = float(sample_count) / float(actual_batch_size)

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
                    with torch.no_grad():
                        adv_z_values.append(advantages.detach().cpu())

                    log_ratio = log_prob - rollout_data.old_log_prob
                    ratio = torch.exp(log_ratio)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss_ppo = -torch.min(policy_loss_1, policy_loss_2).mean()

                    with torch.no_grad():
                        ratio_detached = ratio.detach()
                        ratio_sum += float(ratio_detached.sum().item())
                        ratio_sq_sum += float((ratio_detached.square()).sum().item())
                        ratio_count += int(ratio_detached.numel())
                        clip_mask = ratio_detached.sub(1.0).abs() > clip_range
                        clip_fraction_numer += float(clip_mask.sum().item())
                        clip_fraction_denom += int(clip_mask.numel())

                        log_prob_detached = log_prob.detach()
                        log_prob_sum += float(log_prob_detached.sum().item())
                        log_prob_count += int(log_prob_detached.numel())

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
                    if self._kl_penalty_beta > 0.0:
                        kl_penalty_sample = (rollout_data.old_log_prob - log_prob).mean()
                        kl_penalty_component = self._kl_penalty_beta * kl_penalty_sample
                        policy_loss = policy_loss + kl_penalty_component
                        kl_penalty_component_total += float(kl_penalty_component.item())
                        kl_penalty_component_count += 1

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

                        if self._value_clip_limit_unscaled is not None:
                            target_returns = torch.clamp(
                                target_returns,
                                min=-self._value_clip_limit_unscaled,
                                max=self._value_clip_limit_unscaled,
                            )

                        target_returns_scaled = target_returns * self.value_target_scale

                        if self._value_clip_limit_scaled is not None:
                            target_returns_scaled = torch.clamp(
                                target_returns_scaled,
                                min=-self._value_clip_limit_scaled,
                                max=self._value_clip_limit_scaled,
                            )

                        delta_z = (self.policy.v_max - self.policy.v_min) / float(
                            self.policy.num_atoms - 1
                        )
                        clamped_targets = target_returns_scaled.clamp(
                            self.policy.v_min, self.policy.v_max
                        )
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
                        target_return_batches.append(target_returns.reshape(-1, 1).detach())

                    pred_probs_fp32 = torch.softmax(value_logits_fp32, dim=1).clamp(min=1e-8, max=1.0)
                    log_predictions = torch.log(pred_probs_fp32)
                    critic_loss = -(target_distribution * log_predictions).sum(dim=1).mean()

                    with torch.no_grad():

                        mean_values_batch = (pred_probs_fp32 * self.policy.atoms).sum(dim=1, keepdim=True)
                        mean_values_unscaled = mean_values_batch / self.value_target_scale

                        if self._value_clip_limit_unscaled is not None:
                            mean_values_unscaled = torch.clamp(
                                mean_values_unscaled,
                                min=-self._value_clip_limit_unscaled,
                                max=self._value_clip_limit_unscaled,
                            )
                        mean_value_batches.append(mean_values_unscaled.detach())

                    predicted_cvar = calculate_cvar(
                        pred_probs_fp32, self.policy.atoms, self.cvar_alpha
                    )
                    cvar_raw = (predicted_cvar / self.value_target_scale).mean()

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

                    loss_weighted = loss * loss.new_tensor(weight)
                    loss_weighted.backward()

                    bucket_policy_loss_value += float(policy_loss.item()) * weight
                    bucket_policy_loss_ppo_value += float(policy_loss_ppo.item()) * weight
                    bucket_policy_loss_bc_value += float(policy_loss_bc.item()) * weight
                    bucket_policy_loss_bc_weighted_value += float(policy_loss_bc_weighted.item()) * weight
                    bucket_critic_loss_value += float(critic_loss.item()) * weight
                    bucket_cvar_raw_value += float(cvar_raw.item()) * weight
                    bucket_cvar_loss_value += float(cvar_loss.item()) * weight
                    bucket_cvar_term_value += float(cvar_term.item()) * weight
                    bucket_total_loss_value += float(loss.item()) * weight
                    bucket_value_logits_fp32 = value_logits_fp32.detach()

                    approx_kl_component = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    approx_kl_weighted_sum += approx_kl_component * float(sample_count)

                if bucket_sample_count != actual_batch_size:
                    raise RuntimeError(
                        "Accumulated micro-batch size does not match expected batch_size; check microbatch_size configuration"
                    )

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

                post_clip_norm_sq = 0.0
                for param in self.policy.parameters():
                    grad = param.grad
                    if grad is None:
                        continue
                    post_clip_norm_sq += float(grad.detach().to(dtype=torch.float32).pow(2).sum().item())
                post_clip_norm = math.sqrt(post_clip_norm_sq) if post_clip_norm_sq > 0.0 else 0.0
                self.logger.record("train/grad_norm_post_clip", float(post_clip_norm))

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

                approx_kl = approx_kl_weighted_sum / float(bucket_sample_count)
                approx_kl_latest = approx_kl
                approx_kl_divs.append(approx_kl)
                if (
                    self.target_kl is not None
                    and self.target_kl > 0.0
                    and approx_kl > float(self.target_kl)
                ):
                    approx_kl_exceed_count += 1

                policy_loss_value = bucket_policy_loss_value
                policy_loss_ppo_value = bucket_policy_loss_ppo_value
                policy_loss_bc_value = bucket_policy_loss_bc_value
                policy_loss_bc_weighted_value = bucket_policy_loss_bc_weighted_value
                critic_loss_value = bucket_critic_loss_value
                cvar_raw_value = bucket_cvar_raw_value
                cvar_loss_value = bucket_cvar_loss_value
                cvar_term_value = bucket_cvar_term_value
                total_loss_value = bucket_total_loss_value

                if bucket_value_logits_fp32 is not None:
                    value_logits_final = bucket_value_logits_fp32

                if (
                    self.target_kl is not None
                    and self.target_kl > 0.0
                    and self.kl_exceed_stop_fraction > 0.0
                    and minibatches_processed > 0
                ):
                    exceed_fraction = float(approx_kl_exceed_count) / float(minibatches_processed)
                    kl_exceed_fraction_latest = exceed_fraction
                    if exceed_fraction >= self.kl_exceed_stop_fraction:
                        kl_early_stop_triggered = True
                        self._handle_kl_divergence(approx_kl_latest)
                        break

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
        if len(target_return_batches) == 0 or len(mean_value_batches) == 0:
            rollout_returns = torch.as_tensor(
                self.rollout_buffer.returns, device=self.device, dtype=torch.float32
            )
            with torch.no_grad():
                pred_probs = torch.softmax(value_logits_final, dim=1)
                y_pred_tensor = (
                    (pred_probs * self.policy.atoms).sum(dim=1, keepdim=True)
                    / self.value_target_scale
                )
                if self._value_clip_limit_unscaled is not None:
                    y_pred_tensor = torch.clamp(
                        y_pred_tensor,
                        min=-self._value_clip_limit_unscaled,
                        max=self._value_clip_limit_unscaled,
                    )

            y_true_tensor = rollout_returns.reshape(-1, 1)

        else:
            y_true_tensor = torch.cat([t.reshape(-1, 1) for t in target_return_batches], dim=0)
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
        if kl_penalty_component_count > 0:
            avg_kl_penalty_component = kl_penalty_component_total / float(kl_penalty_component_count)
        else:
            avg_kl_penalty_component = 0.0
        self.logger.record("train/policy_loss_kl_penalty", avg_kl_penalty_component)
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

        approx_kl_exceed_frac = (
            float(approx_kl_exceed_count) / float(minibatches_processed)
            if minibatches_processed > 0
            else 0.0
        )
        if len(approx_kl_divs) > 0:
            approx_kl_array = np.asarray(approx_kl_divs, dtype=np.float64)
            approx_kl_mean = float(np.mean(approx_kl_array))
            self.logger.record("train/approx_kl", approx_kl_mean)
            self.logger.record("train/approx_kl_median", float(np.median(approx_kl_array)))
            self.logger.record("train/approx_kl_p90", float(np.quantile(approx_kl_array, 0.9)))
            self.logger.record("train/approx_kl_max", float(np.max(approx_kl_array)))
            if self.target_kl is not None and self.target_kl > 0.0:
                self._adjust_kl_penalty(approx_kl_mean)
        if self.target_kl is not None and self.target_kl > 0.0:
            self.logger.record("train/kl_exceed_frac", approx_kl_exceed_frac)
            self.logger.record("train/kl_exceed_stop_fraction", float(self.kl_exceed_stop_fraction))
            if kl_early_stop_triggered:
                self.logger.record("train/kl_exceed_frac_at_stop", float(kl_exceed_fraction_latest))
        if last_optimizer_lr is not None:
            self.logger.record("train/optimizer_lr", last_optimizer_lr)
        if last_scheduler_lr is not None:
            self.logger.record("train/scheduler_lr", last_scheduler_lr)
        self.logger.record("train/loss", total_loss_value)
        self.logger.record("train/explained_variance", explained_var)
        clip_range_for_log = float(self.clip_range(self._current_progress_remaining))
        self.logger.record("train/clip_range", clip_range_for_log)
        if clip_fraction_denom > 0:
            clip_fraction = float(clip_fraction_numer) / float(clip_fraction_denom)
            self.logger.record("train/clip_fraction", clip_fraction)
        if ratio_count > 0:
            ratio_mean = ratio_sum / float(ratio_count)
            ratio_var = max(ratio_sq_sum / float(ratio_count) - ratio_mean**2, 0.0)
            ratio_std = math.sqrt(ratio_var)
            self.logger.record("train/ratio_mean", float(ratio_mean))
            self.logger.record("train/ratio_std", float(ratio_std))
        if log_prob_count > 0:
            self.logger.record("train/log_prob_mean", float(log_prob_sum / float(log_prob_count)))
        if adv_batch_count > 0:
            self.logger.record("train/adv_mean", adv_mean_accum / adv_batch_count)
            self.logger.record("train/adv_std", adv_std_accum / adv_batch_count)
        if adv_z_values:
            adv_z_tensor = torch.cat(adv_z_values)
            if adv_z_tensor.numel() > 0:
                adv_z_tensor = adv_z_tensor.to(dtype=torch.float32)
                quantiles = torch.quantile(
                    adv_z_tensor,
                    torch.tensor([0.1, 0.5, 0.9], dtype=adv_z_tensor.dtype, device=adv_z_tensor.device),
                )
                self.logger.record("train/adv_z_p10", float(quantiles[0].item()))
                self.logger.record("train/adv_z_p50", float(quantiles[1].item()))
                self.logger.record("train/adv_z_p90", float(quantiles[2].item()))
        self.logger.record("train/n_epochs_effective", float(effective_n_epochs))
        self.logger.record("train/n_epochs_completed", float(epochs_completed))
        self.logger.record("train/n_minibatches_done", float(minibatches_processed))
        self.logger.record("train/kl_early_stop", float(1.0 if kl_early_stop_triggered else 0.0))
        if self.target_kl is not None and self.target_kl > 0.0:
            self.logger.record("train/target_kl", float(self.target_kl))
            self.logger.record("train/kl_lr_scale", float(self._kl_lr_scale))
            self.logger.record("train/kl_epoch_factor", float(self._kl_epoch_factor))
            self.logger.record("train/kl_penalty_beta", float(self._kl_penalty_beta))

        if y_true_tensor.numel() > 0 and y_pred_tensor.numel() > 0:
            y_true_np = y_true_tensor.flatten().detach().cpu().numpy().astype(np.float64)
            y_pred_np = y_pred_tensor.flatten().detach().cpu().numpy().astype(np.float64)
            self.logger.record("train/value_pred_mean", float(np.mean(y_pred_np)))
            self.logger.record("train/value_pred_std", float(np.std(y_pred_np)))
            self.logger.record("train/target_return_mean", float(np.mean(y_true_np)))
            self.logger.record("train/target_return_std", float(np.std(y_true_np)))
            diff_np = y_pred_np - y_true_np
            self.logger.record("train/value_mae", float(np.mean(np.abs(diff_np))))
            self.logger.record(
                "train/value_rmse", float(math.sqrt(np.mean(np.square(diff_np))))
            )

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
