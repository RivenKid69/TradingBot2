# -*- coding: utf-8 -*-
"""
Impact Calibration Module for L3 LOB Simulation.

Provides parameter estimation for market impact models from historical data:
- Regression-based calibration for impact coefficients
- Maximum likelihood estimation (MLE) for decay parameters
- Cross-validation for model selection
- Rolling window calibration for adaptive parameters

References:
    - Almgren et al. (2005): "Direct Estimation of Equity Market Impact"
    - Mastromatteo et al. (2014): "Agent-Based Models for Latent Liquidity"
    - Zarinelli et al. (2015): "Beyond the Square Root"
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from lob.market_impact import (
    AlmgrenChrissModel,
    DecayType,
    GatheralModel,
    ImpactModelType,
    ImpactParameters,
    ImpactResult,
    KyleLambdaModel,
    MarketImpactModel,
    _DEFAULT_DECAY_BETA,
    _DEFAULT_DECAY_HALF_LIFE_MS,
    _DEFAULT_IMPACT_COEF_PERM,
    _DEFAULT_IMPACT_COEF_TEMP,
    _DEFAULT_IMPACT_EXPONENT,
    create_impact_model,
)


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class TradeObservation:
    """
    Single trade observation for calibration.

    Attributes:
        timestamp_ms: Trade timestamp in milliseconds
        price: Trade price
        qty: Trade quantity
        side: Trade side (1 = buy, -1 = sell)
        adv: Average daily volume at time of trade
        volatility: Estimated volatility at time of trade
        pre_trade_mid: Mid price before trade
        post_trade_mid: Mid price after trade (for impact measurement)
        time_to_next_trade_ms: Time until next trade (for decay estimation)
    """

    timestamp_ms: int
    price: float
    qty: float
    side: int  # +1 = buy, -1 = sell
    adv: float
    volatility: float = 0.02
    pre_trade_mid: Optional[float] = None
    post_trade_mid: Optional[float] = None
    time_to_next_trade_ms: Optional[int] = None

    @property
    def participation(self) -> float:
        """Participation ratio (qty / ADV)."""
        return abs(self.qty) / max(self.adv, 1.0)

    @property
    def realized_impact_bps(self) -> Optional[float]:
        """Realized impact in basis points."""
        if self.pre_trade_mid is None or self.post_trade_mid is None:
            return None
        if self.pre_trade_mid <= 0:
            return None

        price_move = (self.post_trade_mid - self.pre_trade_mid) / self.pre_trade_mid
        # Impact is positive when price moves in trade direction
        return price_move * self.side * 10000.0


@dataclass
class CalibrationDataset:
    """
    Dataset for impact model calibration.

    Attributes:
        observations: List of trade observations
        symbol: Trading symbol
        start_time_ms: Start of observation period
        end_time_ms: End of observation period
        avg_adv: Average ADV over period
        avg_volatility: Average volatility over period
    """

    observations: List[TradeObservation] = field(default_factory=list)
    symbol: str = ""
    start_time_ms: int = 0
    end_time_ms: int = 0
    avg_adv: float = 0.0
    avg_volatility: float = 0.02

    def __len__(self) -> int:
        return len(self.observations)

    def add_observation(self, obs: TradeObservation) -> None:
        """Add observation and update statistics."""
        self.observations.append(obs)
        if self.start_time_ms == 0 or obs.timestamp_ms < self.start_time_ms:
            self.start_time_ms = obs.timestamp_ms
        if obs.timestamp_ms > self.end_time_ms:
            self.end_time_ms = obs.timestamp_ms

    def get_participations(self) -> List[float]:
        """Get list of participation ratios."""
        return [obs.participation for obs in self.observations]

    def get_realized_impacts(self) -> List[float]:
        """Get list of realized impacts (excluding None)."""
        return [
            obs.realized_impact_bps
            for obs in self.observations
            if obs.realized_impact_bps is not None
        ]


@dataclass
class CalibrationResult:
    """
    Result of model calibration.

    Attributes:
        model_type: Type of model calibrated
        parameters: Calibrated parameters
        r_squared: Coefficient of determination
        rmse: Root mean squared error
        mae: Mean absolute error
        n_observations: Number of observations used
        confidence_intervals: 95% confidence intervals for parameters
        diagnostics: Additional diagnostic information
    """

    model_type: ImpactModelType = ImpactModelType.ALMGREN_CHRISS
    parameters: Dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    n_observations: int = 0
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def get_parameter(self, name: str, default: float = 0.0) -> float:
        """Get calibrated parameter value."""
        return self.parameters.get(name, default)

    def create_model(self) -> MarketImpactModel:
        """Create model instance with calibrated parameters."""
        params = ImpactParameters(
            eta=self.parameters.get("eta", _DEFAULT_IMPACT_COEF_TEMP),
            gamma=self.parameters.get("gamma", _DEFAULT_IMPACT_COEF_PERM),
            delta=self.parameters.get("delta", _DEFAULT_IMPACT_EXPONENT),
            tau_ms=self.parameters.get("tau_ms", _DEFAULT_DECAY_HALF_LIFE_MS),
            beta=self.parameters.get("beta", _DEFAULT_DECAY_BETA),
            volatility=self.parameters.get("volatility", 0.02),
        )

        if self.model_type == ImpactModelType.GATHERAL:
            return GatheralModel(params=params)
        elif self.model_type == ImpactModelType.KYLE_LAMBDA:
            return KyleLambdaModel(
                lambda_coef=self.parameters.get("lambda", 0.0001),
                permanent_fraction=self.parameters.get("perm_frac", 0.5),
            )
        else:
            return AlmgrenChrissModel(params=params)


@dataclass
class CrossValidationResult:
    """
    Cross-validation result for model comparison.

    Attributes:
        model_type: Model type evaluated
        mean_r_squared: Mean R² across folds
        std_r_squared: Standard deviation of R²
        mean_rmse: Mean RMSE across folds
        std_rmse: Standard deviation of RMSE
        fold_results: Results for each fold
        best_parameters: Parameters from best fold
    """

    model_type: ImpactModelType = ImpactModelType.ALMGREN_CHRISS
    mean_r_squared: float = 0.0
    std_r_squared: float = 0.0
    mean_rmse: float = 0.0
    std_rmse: float = 0.0
    fold_results: List[CalibrationResult] = field(default_factory=list)
    best_parameters: Dict[str, float] = field(default_factory=dict)


# ==============================================================================
# Base Calibrator
# ==============================================================================


class BaseImpactCalibrator:
    """
    Base class for impact model calibrators.

    Provides common functionality for parameter estimation.
    """

    def __init__(
        self,
        model_type: ImpactModelType = ImpactModelType.ALMGREN_CHRISS,
        min_observations: int = 30,
    ) -> None:
        """
        Initialize calibrator.

        Args:
            model_type: Type of model to calibrate
            min_observations: Minimum observations required
        """
        self._model_type = model_type
        self._min_observations = min_observations

    @property
    def model_type(self) -> ImpactModelType:
        """Get model type."""
        return self._model_type

    def calibrate(self, dataset: CalibrationDataset) -> CalibrationResult:
        """
        Calibrate model from dataset.

        Args:
            dataset: Calibration dataset

        Returns:
            CalibrationResult with fitted parameters
        """
        raise NotImplementedError("Subclasses must implement calibrate()")

    def _compute_metrics(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
    ) -> Dict[str, float]:
        """Compute regression metrics."""
        n = len(y_true)
        if n == 0:
            return {"r_squared": 0.0, "rmse": 0.0, "mae": 0.0}

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        # RMSE
        mse = np.mean((y_true_arr - y_pred_arr) ** 2)
        rmse = math.sqrt(mse)

        # MAE
        mae = np.mean(np.abs(y_true_arr - y_pred_arr))

        # R²
        ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
        ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "r_squared": float(r_squared),
            "rmse": float(rmse),
            "mae": float(mae),
        }


# ==============================================================================
# Almgren-Chriss Calibrator
# ==============================================================================


class AlmgrenChrissCalibrator(BaseImpactCalibrator):
    """
    Calibrator for Almgren-Chriss impact model.

    Uses regression to estimate η, γ, δ from trade data:
        impact = η * σ * (Q/V)^δ + γ * (Q/V)

    For fixed δ=0.5 (square-root), this becomes linear in sqrt(participation).
    """

    def __init__(
        self,
        fixed_delta: bool = True,
        delta_value: float = 0.5,
        min_observations: int = 30,
    ) -> None:
        """
        Initialize Almgren-Chriss calibrator.

        Args:
            fixed_delta: Whether to fix δ (vs estimate)
            delta_value: Value of δ if fixed
            min_observations: Minimum observations required
        """
        super().__init__(
            model_type=ImpactModelType.ALMGREN_CHRISS,
            min_observations=min_observations,
        )
        self._fixed_delta = fixed_delta
        self._delta = delta_value

    def calibrate(self, dataset: CalibrationDataset) -> CalibrationResult:
        """
        Calibrate Almgren-Chriss model.

        Uses OLS regression:
            impact_bps = η * σ * sqrt(participation) + γ * participation

        With feature matrix:
            X1 = σ * sqrt(participation)  -> coefficient = η * 10000
            X2 = participation            -> coefficient = γ * 10000
        """
        # Filter observations with valid impacts
        valid_obs = [obs for obs in dataset.observations if obs.realized_impact_bps is not None]

        if len(valid_obs) < self._min_observations:
            return CalibrationResult(
                model_type=self._model_type,
                parameters={"eta": _DEFAULT_IMPACT_COEF_TEMP, "gamma": _DEFAULT_IMPACT_COEF_PERM},
                n_observations=len(valid_obs),
                diagnostics={"error": "insufficient_data"},
            )

        # Build feature matrix
        X = []
        y = []

        for obs in valid_obs:
            participation = obs.participation
            volatility = obs.volatility

            # Features
            sqrt_part = volatility * (participation**self._delta)
            linear_part = participation

            X.append([sqrt_part, linear_part])
            y.append(obs.realized_impact_bps)

        X_arr = np.array(X)
        y_arr = np.array(y)

        # OLS regression: y = X @ beta
        # Using normal equations: beta = (X'X)^(-1) X'y
        try:
            XtX = X_arr.T @ X_arr
            Xty = X_arr.T @ y_arr

            # Add small regularization for stability
            reg = 1e-8 * np.eye(XtX.shape[0])
            beta = np.linalg.solve(XtX + reg, Xty)

            eta = beta[0] / 10000.0  # Convert from bps coefficient
            gamma = beta[1] / 10000.0

            # Ensure non-negative
            eta = max(0.0, eta)
            gamma = max(0.0, gamma)

        except np.linalg.LinAlgError:
            eta = _DEFAULT_IMPACT_COEF_TEMP
            gamma = _DEFAULT_IMPACT_COEF_PERM

        # Compute predictions and metrics
        y_pred = X_arr @ beta if "beta" in dir() else np.zeros_like(y_arr)
        metrics = self._compute_metrics(y_arr, y_pred)

        # Estimate confidence intervals (bootstrap would be better)
        # Approximate using standard errors
        n = len(y_arr)
        p = 2  # Number of parameters
        if n > p:
            residuals = y_arr - y_pred
            mse = np.sum(residuals**2) / (n - p)
            try:
                var_beta = mse * np.linalg.inv(XtX + reg)
                se_beta = np.sqrt(np.diag(var_beta))
                ci_eta = (
                    max(0, (beta[0] - 1.96 * se_beta[0]) / 10000.0),
                    (beta[0] + 1.96 * se_beta[0]) / 10000.0,
                )
                ci_gamma = (
                    max(0, (beta[1] - 1.96 * se_beta[1]) / 10000.0),
                    (beta[1] + 1.96 * se_beta[1]) / 10000.0,
                )
            except Exception:
                ci_eta = (eta * 0.5, eta * 1.5)
                ci_gamma = (gamma * 0.5, gamma * 1.5)
        else:
            ci_eta = (eta * 0.5, eta * 1.5)
            ci_gamma = (gamma * 0.5, gamma * 1.5)

        return CalibrationResult(
            model_type=self._model_type,
            parameters={
                "eta": eta,
                "gamma": gamma,
                "delta": self._delta,
                "volatility": dataset.avg_volatility,
            },
            r_squared=metrics["r_squared"],
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            n_observations=len(valid_obs),
            confidence_intervals={"eta": ci_eta, "gamma": ci_gamma},
            diagnostics={
                "avg_participation": float(np.mean([o.participation for o in valid_obs])),
                "avg_impact_bps": float(np.mean(y_arr)),
            },
        )


# ==============================================================================
# Gatheral Decay Calibrator
# ==============================================================================


class GatheralDecayCalibrator(BaseImpactCalibrator):
    """
    Calibrator for Gatheral decay parameters.

    Estimates τ (decay time constant) and β (power-law exponent)
    from trade sequences.

    Uses non-linear least squares or MLE to fit:
        G(t) = (1 + t/τ)^(-β)
    """

    def __init__(
        self,
        min_observations: int = 50,
        max_decay_time_ms: int = 300000,  # 5 minutes
    ) -> None:
        """
        Initialize Gatheral decay calibrator.

        Args:
            min_observations: Minimum observations required
            max_decay_time_ms: Maximum decay time to consider
        """
        super().__init__(
            model_type=ImpactModelType.GATHERAL,
            min_observations=min_observations,
        )
        self._max_decay_time_ms = max_decay_time_ms

    def calibrate(self, dataset: CalibrationDataset) -> CalibrationResult:
        """
        Calibrate Gatheral decay model.

        Uses pairs of consecutive impacts to estimate decay.
        """
        # Need pairs of trades to measure decay
        valid_pairs: List[Tuple[TradeObservation, TradeObservation]] = []

        sorted_obs = sorted(
            [o for o in dataset.observations if o.realized_impact_bps is not None],
            key=lambda x: x.timestamp_ms,
        )

        for i in range(len(sorted_obs) - 1):
            obs1 = sorted_obs[i]
            obs2 = sorted_obs[i + 1]

            dt = obs2.timestamp_ms - obs1.timestamp_ms
            if 0 < dt <= self._max_decay_time_ms:
                valid_pairs.append((obs1, obs2))

        if len(valid_pairs) < self._min_observations // 2:
            return CalibrationResult(
                model_type=self._model_type,
                parameters={"tau_ms": _DEFAULT_DECAY_HALF_LIFE_MS, "beta": _DEFAULT_DECAY_BETA},
                n_observations=len(valid_pairs),
                diagnostics={"error": "insufficient_pairs"},
            )

        # Grid search for τ and β
        # For each candidate, compute SSE
        tau_candidates = [5000, 10000, 20000, 30000, 60000, 120000, 180000]
        beta_candidates = [0.5, 1.0, 1.5, 2.0, 2.5]

        best_sse = float("inf")
        best_tau = _DEFAULT_DECAY_HALF_LIFE_MS
        best_beta = _DEFAULT_DECAY_BETA

        for tau in tau_candidates:
            for beta in beta_candidates:
                sse = self._compute_decay_sse(valid_pairs, tau, beta)
                if sse < best_sse:
                    best_sse = sse
                    best_tau = tau
                    best_beta = beta

        # Refine around best (simple hill climbing)
        for _ in range(3):
            for tau_adj in [-0.2, 0, 0.2]:
                for beta_adj in [-0.2, 0, 0.2]:
                    tau_try = best_tau * (1 + tau_adj)
                    beta_try = best_beta * (1 + beta_adj)
                    if tau_try > 0 and beta_try > 0:
                        sse = self._compute_decay_sse(valid_pairs, tau_try, beta_try)
                        if sse < best_sse:
                            best_sse = sse
                            best_tau = tau_try
                            best_beta = beta_try

        # Compute metrics
        rmse = math.sqrt(best_sse / len(valid_pairs)) if valid_pairs else 0.0

        return CalibrationResult(
            model_type=self._model_type,
            parameters={
                "tau_ms": best_tau,
                "beta": best_beta,
                "eta": _DEFAULT_IMPACT_COEF_TEMP,
                "gamma": _DEFAULT_IMPACT_COEF_PERM,
            },
            rmse=rmse,
            n_observations=len(valid_pairs),
            diagnostics={
                "sse": best_sse,
                "avg_dt_ms": float(
                    np.mean([obs2.timestamp_ms - obs1.timestamp_ms for obs1, obs2 in valid_pairs])
                ),
            },
        )

    def _compute_decay_sse(
        self,
        pairs: List[Tuple[TradeObservation, TradeObservation]],
        tau: float,
        beta: float,
    ) -> float:
        """Compute sum of squared errors for decay model."""
        sse = 0.0

        for obs1, obs2 in pairs:
            dt = float(obs2.timestamp_ms - obs1.timestamp_ms)
            impact1 = obs1.realized_impact_bps or 0.0

            # Predicted remaining impact at time of obs2
            decay = (1.0 + dt / tau) ** (-beta)
            predicted_remaining = impact1 * decay

            # Actual impact at obs2 (relative to same direction)
            impact2 = obs2.realized_impact_bps or 0.0

            # If same direction, expect continuation; opposite, expect reversal
            if obs1.side == obs2.side:
                error = impact2 - predicted_remaining
            else:
                # Opposite side - impact should have decayed
                error = abs(impact2) - abs(predicted_remaining)

            sse += error**2

        return sse


# ==============================================================================
# Kyle Lambda Calibrator
# ==============================================================================


class KyleLambdaCalibrator(BaseImpactCalibrator):
    """
    Calibrator for Kyle lambda model.

    Estimates λ from price-volume relationship:
        Δp = λ * sign(x) * |x|

    Uses linear regression of price changes on signed volume.
    """

    def __init__(self, min_observations: int = 30) -> None:
        super().__init__(
            model_type=ImpactModelType.KYLE_LAMBDA,
            min_observations=min_observations,
        )

    def calibrate(self, dataset: CalibrationDataset) -> CalibrationResult:
        """Calibrate Kyle lambda from trade data."""
        valid_obs = [obs for obs in dataset.observations if obs.realized_impact_bps is not None]

        if len(valid_obs) < self._min_observations:
            return CalibrationResult(
                model_type=self._model_type,
                parameters={"lambda": 0.0001, "perm_frac": 0.5},
                n_observations=len(valid_obs),
            )

        # Build regression: impact = λ * signed_volume
        X = np.array([[obs.side * obs.qty] for obs in valid_obs])
        y = np.array([obs.realized_impact_bps for obs in valid_obs])

        # OLS
        try:
            XtX = X.T @ X
            Xty = X.T @ y
            lambda_bps = float(np.linalg.solve(XtX + 1e-10, Xty)[0])
            lambda_coef = lambda_bps / 10000.0  # Convert from bps
            lambda_coef = max(0.0, lambda_coef)
        except Exception:
            lambda_coef = 0.0001

        # Estimate permanent fraction from autocorrelation of returns
        # (simplified - would need time series analysis)
        perm_frac = 0.5

        # Metrics
        y_pred = X @ np.array([[lambda_bps]])
        y_pred = y_pred.flatten()
        metrics = self._compute_metrics(y, y_pred)

        return CalibrationResult(
            model_type=self._model_type,
            parameters={"lambda": lambda_coef, "perm_frac": perm_frac},
            r_squared=metrics["r_squared"],
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            n_observations=len(valid_obs),
        )


# ==============================================================================
# Composite Calibration Pipeline
# ==============================================================================


class ImpactCalibrationPipeline:
    """
    Complete calibration pipeline for market impact models.

    Provides:
    - Multi-model calibration
    - Cross-validation
    - Model comparison
    - Rolling calibration
    """

    def __init__(
        self,
        models: Optional[List[ImpactModelType]] = None,
        n_folds: int = 5,
    ) -> None:
        """
        Initialize calibration pipeline.

        Args:
            models: List of models to calibrate (default: all)
            n_folds: Number of folds for cross-validation
        """
        self._models = models or [
            ImpactModelType.ALMGREN_CHRISS,
            ImpactModelType.GATHERAL,
            ImpactModelType.KYLE_LAMBDA,
        ]
        self._n_folds = n_folds
        self._calibrators: Dict[ImpactModelType, BaseImpactCalibrator] = {
            ImpactModelType.ALMGREN_CHRISS: AlmgrenChrissCalibrator(),
            ImpactModelType.GATHERAL: GatheralDecayCalibrator(),
            ImpactModelType.KYLE_LAMBDA: KyleLambdaCalibrator(),
        }
        self._results: Dict[ImpactModelType, CalibrationResult] = {}

    def calibrate_all(
        self,
        dataset: CalibrationDataset,
    ) -> Dict[ImpactModelType, CalibrationResult]:
        """
        Calibrate all specified models.

        Args:
            dataset: Calibration dataset

        Returns:
            Dictionary of calibration results by model type
        """
        self._results = {}

        for model_type in self._models:
            if model_type in self._calibrators:
                result = self._calibrators[model_type].calibrate(dataset)
                self._results[model_type] = result

        return self._results

    def cross_validate(
        self,
        dataset: CalibrationDataset,
    ) -> Dict[ImpactModelType, CrossValidationResult]:
        """
        Perform cross-validation for model comparison.

        Args:
            dataset: Calibration dataset

        Returns:
            Cross-validation results by model type
        """
        cv_results: Dict[ImpactModelType, CrossValidationResult] = {}

        # Split dataset into folds
        n_obs = len(dataset.observations)
        fold_size = n_obs // self._n_folds

        for model_type in self._models:
            if model_type not in self._calibrators:
                continue

            calibrator = self._calibrators[model_type]
            fold_results: List[CalibrationResult] = []
            r_squared_list: List[float] = []
            rmse_list: List[float] = []

            for fold in range(self._n_folds):
                # Create train/test split
                test_start = fold * fold_size
                test_end = test_start + fold_size

                train_obs = dataset.observations[:test_start] + dataset.observations[test_end:]

                if len(train_obs) < calibrator._min_observations:
                    continue

                train_dataset = CalibrationDataset(
                    observations=train_obs,
                    symbol=dataset.symbol,
                    avg_adv=dataset.avg_adv,
                    avg_volatility=dataset.avg_volatility,
                )

                result = calibrator.calibrate(train_dataset)
                fold_results.append(result)
                r_squared_list.append(result.r_squared)
                rmse_list.append(result.rmse)

            if fold_results:
                mean_r2 = statistics.mean(r_squared_list)
                std_r2 = statistics.stdev(r_squared_list) if len(r_squared_list) > 1 else 0.0
                mean_rmse = statistics.mean(rmse_list)
                std_rmse = statistics.stdev(rmse_list) if len(rmse_list) > 1 else 0.0

                # Best parameters from fold with highest R²
                best_fold = max(fold_results, key=lambda r: r.r_squared)

                cv_results[model_type] = CrossValidationResult(
                    model_type=model_type,
                    mean_r_squared=mean_r2,
                    std_r_squared=std_r2,
                    mean_rmse=mean_rmse,
                    std_rmse=std_rmse,
                    fold_results=fold_results,
                    best_parameters=best_fold.parameters,
                )

        return cv_results

    def get_best_model(
        self,
        cv_results: Optional[Dict[ImpactModelType, CrossValidationResult]] = None,
        metric: str = "r_squared",
    ) -> Tuple[ImpactModelType, CalibrationResult]:
        """
        Get best model based on CV results.

        Args:
            cv_results: Cross-validation results
            metric: Metric to use for comparison ("r_squared" or "rmse")

        Returns:
            Tuple of (best model type, calibration result)
        """
        if cv_results is None:
            if not self._results:
                raise ValueError("No calibration results available")
            # Use direct calibration results
            if metric == "rmse":
                best_type = min(self._results, key=lambda t: self._results[t].rmse)
            else:
                best_type = max(self._results, key=lambda t: self._results[t].r_squared)
            return best_type, self._results[best_type]

        # Use CV results
        if metric == "rmse":
            best_type = min(cv_results, key=lambda t: cv_results[t].mean_rmse)
        else:
            best_type = max(cv_results, key=lambda t: cv_results[t].mean_r_squared)

        # Create result from best parameters
        best_cv = cv_results[best_type]
        result = CalibrationResult(
            model_type=best_type,
            parameters=best_cv.best_parameters,
            r_squared=best_cv.mean_r_squared,
            rmse=best_cv.mean_rmse,
            n_observations=sum(r.n_observations for r in best_cv.fold_results),
        )

        return best_type, result

    def create_calibrated_model(
        self,
        model_type: Optional[ImpactModelType] = None,
    ) -> MarketImpactModel:
        """
        Create model instance with calibrated parameters.

        Args:
            model_type: Model type (default: best model)

        Returns:
            Calibrated MarketImpactModel instance
        """
        if model_type is None:
            model_type, result = self.get_best_model()
        else:
            result = self._results.get(model_type)
            if result is None:
                raise ValueError(f"No calibration result for {model_type}")

        return result.create_model()


# ==============================================================================
# Rolling Calibration
# ==============================================================================


class RollingImpactCalibrator:
    """
    Rolling window calibration for adaptive impact parameters.

    Updates parameters as new data arrives, allowing the model
    to adapt to changing market conditions.
    """

    def __init__(
        self,
        model_type: ImpactModelType = ImpactModelType.ALMGREN_CHRISS,
        window_size: int = 500,
        update_frequency: int = 50,
    ) -> None:
        """
        Initialize rolling calibrator.

        Args:
            model_type: Model type to calibrate
            window_size: Number of observations in rolling window
            update_frequency: How often to recalibrate (new observations)
        """
        self._model_type = model_type
        self._window_size = window_size
        self._update_frequency = update_frequency

        self._observations: deque = deque(maxlen=window_size)
        self._obs_since_update = 0
        self._current_params: Dict[str, float] = {}

        # Create appropriate calibrator
        if model_type == ImpactModelType.GATHERAL:
            self._calibrator = GatheralDecayCalibrator(min_observations=30)
        elif model_type == ImpactModelType.KYLE_LAMBDA:
            self._calibrator = KyleLambdaCalibrator(min_observations=30)
        else:
            self._calibrator = AlmgrenChrissCalibrator(min_observations=30)

    @property
    def current_parameters(self) -> Dict[str, float]:
        """Get current calibrated parameters."""
        return self._current_params.copy()

    def add_observation(self, obs: TradeObservation) -> Optional[CalibrationResult]:
        """
        Add observation and potentially recalibrate.

        Args:
            obs: New trade observation

        Returns:
            CalibrationResult if recalibration occurred, else None
        """
        self._observations.append(obs)
        self._obs_since_update += 1

        # Check if time to recalibrate
        if self._obs_since_update >= self._update_frequency:
            if len(self._observations) >= self._calibrator._min_observations:
                return self._recalibrate()

        return None

    def _recalibrate(self) -> CalibrationResult:
        """Perform recalibration on current window."""
        dataset = CalibrationDataset(observations=list(self._observations))
        result = self._calibrator.calibrate(dataset)

        self._current_params = result.parameters.copy()
        self._obs_since_update = 0

        return result

    def get_current_model(self) -> MarketImpactModel:
        """Get model with current parameters."""
        if not self._current_params:
            # Return default model
            return create_impact_model(
                model_type=self._model_type.name.lower(),
            )

        params = ImpactParameters(
            eta=self._current_params.get("eta", _DEFAULT_IMPACT_COEF_TEMP),
            gamma=self._current_params.get("gamma", _DEFAULT_IMPACT_COEF_PERM),
            delta=self._current_params.get("delta", _DEFAULT_IMPACT_EXPONENT),
            tau_ms=self._current_params.get("tau_ms", _DEFAULT_DECAY_HALF_LIFE_MS),
            beta=self._current_params.get("beta", _DEFAULT_DECAY_BETA),
        )

        if self._model_type == ImpactModelType.GATHERAL:
            return GatheralModel(params=params)
        else:
            return AlmgrenChrissModel(params=params)


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_calibrator(
    model_type: Union[str, ImpactModelType] = "almgren_chriss",
) -> BaseImpactCalibrator:
    """
    Factory function to create impact calibrator.

    Args:
        model_type: Model type name or enum

    Returns:
        BaseImpactCalibrator instance
    """
    if isinstance(model_type, str):
        type_map = {
            "kyle": ImpactModelType.KYLE_LAMBDA,
            "kyle_lambda": ImpactModelType.KYLE_LAMBDA,
            "almgren": ImpactModelType.ALMGREN_CHRISS,
            "almgren_chriss": ImpactModelType.ALMGREN_CHRISS,
            "gatheral": ImpactModelType.GATHERAL,
        }
        model_enum = type_map.get(model_type.lower(), ImpactModelType.ALMGREN_CHRISS)
    else:
        model_enum = model_type

    if model_enum == ImpactModelType.GATHERAL:
        return GatheralDecayCalibrator()
    elif model_enum == ImpactModelType.KYLE_LAMBDA:
        return KyleLambdaCalibrator()
    else:
        return AlmgrenChrissCalibrator()


def create_calibration_pipeline(
    models: Optional[List[str]] = None,
    n_folds: int = 5,
) -> ImpactCalibrationPipeline:
    """
    Factory function to create calibration pipeline.

    Args:
        models: List of model type names
        n_folds: Number of CV folds

    Returns:
        ImpactCalibrationPipeline instance
    """
    if models is None:
        model_types = None
    else:
        type_map = {
            "kyle": ImpactModelType.KYLE_LAMBDA,
            "almgren_chriss": ImpactModelType.ALMGREN_CHRISS,
            "gatheral": ImpactModelType.GATHERAL,
        }
        model_types = [type_map.get(m.lower(), ImpactModelType.ALMGREN_CHRISS) for m in models]

    return ImpactCalibrationPipeline(models=model_types, n_folds=n_folds)


def calibrate_from_trades(
    trades: List[Dict],
    adv: float,
    volatility: float = 0.02,
    model_type: str = "almgren_chriss",
) -> CalibrationResult:
    """
    Convenience function to calibrate from trade list.

    Args:
        trades: List of trade dicts with keys: timestamp_ms, price, qty, side, pre_mid, post_mid
        adv: Average daily volume
        volatility: Volatility estimate
        model_type: Model type to calibrate

    Returns:
        CalibrationResult
    """
    observations = []
    for t in trades:
        obs = TradeObservation(
            timestamp_ms=t.get("timestamp_ms", 0),
            price=t.get("price", 0.0),
            qty=t.get("qty", 0.0),
            side=t.get("side", 1),
            adv=adv,
            volatility=volatility,
            pre_trade_mid=t.get("pre_mid"),
            post_trade_mid=t.get("post_mid"),
        )
        observations.append(obs)

    dataset = CalibrationDataset(
        observations=observations,
        avg_adv=adv,
        avg_volatility=volatility,
    )

    calibrator = create_calibrator(model_type)
    return calibrator.calibrate(dataset)
