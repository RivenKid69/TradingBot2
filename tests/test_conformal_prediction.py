# -*- coding: utf-8 -*-
"""
Comprehensive tests for conformal prediction integration.

Tests:
1. Configuration validation
2. CQR calibration and coverage guarantee
3. EnbPI time series handling
4. ACI adaptive behavior
5. CVaR bounds computation
6. Uncertainty tracking and escalation
7. Service integration
8. Edge cases and error handling
"""

import pytest
import numpy as np
from typing import Optional
from unittest.mock import Mock, patch, MagicMock

# Import core types
from core_conformal import (
    ConformalConfig,
    CalibrationConfig,
    TimeSeriesConfig,
    CVaRBoundsConfig,
    RiskIntegrationConfig,
    EscalationConfig,
    TradingSignalsConfig,
    PerformanceConfig,
    DebugConfig,
    ConformalMethod,
    EscalationLevel,
    EscalationAction,
    PredictionInterval,
    CVaRBounds,
    UncertaintyState,
    CalibrationResult,
    create_conformal_config,
    create_disabled_config,
    validate_config,
)

# Import implementations
from impl_conformal import (
    CQRCalibrator,
    EnbPICalibrator,
    ACICalibrator,
    NaiveCalibrator,
    ConformalCVaREstimator,
    UncertaintyTrackerImpl,
    create_calibrator,
    create_cvar_estimator,
    create_uncertainty_tracker,
)

# Import service
from service_conformal import (
    ConformalPredictionService,
    create_conformal_service,
    wrap_cvar_with_bounds,
    create_risk_guard_integration,
    create_escalation_handler,
    TrainingConformalIntegration,
    LiveTradingConformalIntegration,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Create default conformal config."""
    return ConformalConfig()


@pytest.fixture
def disabled_config():
    """Create disabled conformal config."""
    return ConformalConfig(enabled=False)


@pytest.fixture
def cqr_config():
    """Create CQR-specific config with lower sample requirements for testing."""
    return ConformalConfig(
        enabled=True,
        calibration=CalibrationConfig(
            method=ConformalMethod.CQR,
            coverage_target=0.90,
            min_calibration_samples=50,
        ),
    )


@pytest.fixture
def enbpi_config():
    """Create EnbPI config."""
    return ConformalConfig(
        enabled=True,
        calibration=CalibrationConfig(
            method=ConformalMethod.ENBPI,
            min_calibration_samples=20,
        ),
        time_series=TimeSeriesConfig(
            enabled=True,
            lookback_window=100,
        ),
    )


@pytest.fixture
def aci_config():
    """Create ACI config."""
    return ConformalConfig(
        enabled=True,
        calibration=CalibrationConfig(
            method=ConformalMethod.ACI,
            min_calibration_samples=20,
        ),
        time_series=TimeSeriesConfig(
            enabled=True,
            aci_gamma=0.05,
            lookback_window=100,
        ),
    )


@pytest.fixture
def cqr_calibrator(cqr_config):
    """Create CQR calibrator."""
    return CQRCalibrator(cqr_config)


@pytest.fixture
def enbpi_calibrator(enbpi_config):
    """Create EnbPI calibrator."""
    return EnbPICalibrator(enbpi_config)


@pytest.fixture
def aci_calibrator(aci_config):
    """Create ACI calibrator."""
    return ACICalibrator(aci_config)


@pytest.fixture
def service(cqr_config):
    """Create conformal prediction service."""
    return ConformalPredictionService(cqr_config)


@pytest.fixture
def disabled_service(disabled_config):
    """Create disabled conformal prediction service."""
    return ConformalPredictionService(disabled_config)


# =============================================================================
# Test Configuration
# =============================================================================


class TestConformalConfig:
    """Test configuration validation."""

    def test_default_config(self, default_config):
        """Test default config values."""
        assert default_config.enabled is True
        assert default_config.method == ConformalMethod.CQR
        assert default_config.coverage_target == 0.90

    def test_disabled_config(self, disabled_config):
        """Test disabled config."""
        assert disabled_config.enabled is False

    def test_invalid_coverage_target_high(self):
        """Test invalid coverage target above 1."""
        with pytest.raises(ValueError, match="coverage_target"):
            CalibrationConfig(coverage_target=1.5)

    def test_invalid_coverage_target_low(self):
        """Test invalid coverage target at or below 0."""
        with pytest.raises(ValueError, match="coverage_target"):
            CalibrationConfig(coverage_target=0.0)

    def test_invalid_calibration_fraction(self):
        """Test invalid calibration fraction."""
        with pytest.raises(ValueError, match="calibration_fraction"):
            CalibrationConfig(calibration_fraction=1.5)

    def test_invalid_min_calibration_samples(self):
        """Test invalid min calibration samples."""
        with pytest.raises(ValueError, match="min_calibration_samples"):
            CalibrationConfig(min_calibration_samples=5)

    def test_invalid_aci_gamma(self):
        """Test invalid ACI gamma."""
        with pytest.raises(ValueError, match="aci_gamma"):
            TimeSeriesConfig(aci_gamma=1.5)

    def test_invalid_escalation_percentiles(self):
        """Test invalid escalation percentiles."""
        with pytest.raises(ValueError, match="warning_percentile"):
            EscalationConfig(warning_percentile=95, critical_percentile=90)

    def test_create_conformal_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "enabled": True,
            "calibration": {
                "method": "cqr",
                "coverage_target": 0.95,
            },
            "risk_integration": {
                "enabled": True,
                "uncertainty_position_scaling": True,
            },
        }
        config = create_conformal_config(config_dict)

        assert config.enabled is True
        assert config.method == ConformalMethod.CQR
        assert config.coverage_target == 0.95
        assert config.risk_integration_enabled is True

    def test_create_disabled_config(self):
        """Test creating disabled config."""
        config = create_disabled_config()
        assert config.enabled is False

    def test_validate_config_warnings(self):
        """Test config validation warnings."""
        config = ConformalConfig(
            calibration=CalibrationConfig(
                coverage_target=0.995,
                min_calibration_samples=50,
                method=ConformalMethod.NAIVE,
            ),
            cvar_bounds=CVaRBoundsConfig(use_for_gae=True),
        )
        warnings = validate_config(config)

        assert len(warnings) >= 3  # High coverage, low samples, naive method, use_for_gae


# =============================================================================
# Test CQR Calibrator
# =============================================================================


class TestCQRCalibrator:
    """Test Conformalized Quantile Regression."""

    def test_uncalibrated_returns_raw_quantiles(self, cqr_calibrator):
        """Test uncalibrated predictor returns raw quantiles."""
        interval = cqr_calibrator.predict_interval(
            predicted_lower=0.0,
            predicted_upper=1.0,
            point_estimate=0.5,
        )
        assert not interval.is_calibrated
        assert interval.lower_bound == 0.0
        assert interval.upper_bound == 1.0
        assert interval.point_estimate == 0.5

    def test_calibration_achieves_coverage(self, cqr_calibrator):
        """Test CQR achieves target coverage after calibration."""
        np.random.seed(42)
        n = 1000

        # Generate data: Y ~ N(0, 1)
        true_values = np.random.randn(n)
        # Predicted quantiles (5th and 95th percentile estimates with noise)
        predicted_lower = true_values - 1.64 + np.random.randn(n) * 0.2
        predicted_upper = true_values + 1.64 + np.random.randn(n) * 0.2

        # Split into calibration and test
        cal_idx = int(n * 0.5)
        result = cqr_calibrator.calibrate(
            predicted_lower[:cal_idx],
            predicted_upper[:cal_idx],
            true_values[:cal_idx],
        )

        assert result.success
        assert cqr_calibrator.is_calibrated

        # Test coverage
        covered = 0
        for i in range(cal_idx, n):
            interval = cqr_calibrator.predict_interval(
                predicted_lower[i],
                predicted_upper[i],
                point_estimate=(predicted_lower[i] + predicted_upper[i]) / 2,
            )
            if interval.contains(true_values[i]):
                covered += 1

        coverage = covered / (n - cal_idx)
        # Should be close to 90% (allow some variance)
        assert 0.85 <= coverage <= 0.95, f"Coverage {coverage} outside [0.85, 0.95]"

    def test_insufficient_samples_warning(self, cqr_calibrator, caplog):
        """Test warning for insufficient calibration samples."""
        result = cqr_calibrator.calibrate(
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.5]),
        )
        assert not result.success
        assert not cqr_calibrator.is_calibrated
        assert "Insufficient" in result.error_message

    def test_calibration_result_structure(self, cqr_calibrator):
        """Test calibration result has correct structure."""
        np.random.seed(42)
        n = 100
        true_values = np.random.randn(n)
        predicted_lower = true_values - 1.0
        predicted_upper = true_values + 1.0

        result = cqr_calibrator.calibrate(predicted_lower, predicted_upper, true_values)

        assert result.success
        assert result.samples_used == n
        assert result.method == ConformalMethod.CQR
        assert result.calibration_quantile is not None
        assert result.empirical_coverage is not None
        assert 0.0 <= result.empirical_coverage <= 1.0

    def test_prediction_interval_structure(self, cqr_calibrator):
        """Test prediction interval has correct structure."""
        interval = cqr_calibrator.predict_interval(
            predicted_lower=0.0,
            predicted_upper=1.0,
            point_estimate=0.5,
        )

        assert isinstance(interval, PredictionInterval)
        assert interval.point_estimate == 0.5
        assert interval.lower_bound <= interval.upper_bound
        assert interval.coverage_target == 0.90
        assert interval.method == ConformalMethod.CQR
        assert interval.interval_width == 1.0

    def test_get_state(self, cqr_calibrator):
        """Test state serialization."""
        state = cqr_calibrator.get_state()
        assert state["method"] == "CQR"
        assert state["is_calibrated"] is False


# =============================================================================
# Test EnbPI Calibrator
# =============================================================================


class TestEnbPICalibrator:
    """Test Ensemble batch Prediction Intervals for time series."""

    def test_partial_fit_updates_residuals(self, enbpi_calibrator):
        """Test partial_fit updates residual buffer."""
        for i in range(30):
            enbpi_calibrator.partial_fit(predicted=float(i), true_value=float(i + 0.1))

        assert enbpi_calibrator.is_calibrated
        assert enbpi_calibrator.calibration_samples == 30

    def test_interval_adapts_to_residuals(self, enbpi_calibrator):
        """Test interval width adapts to residual distribution."""
        # Low noise data
        for i in range(50):
            enbpi_calibrator.partial_fit(float(i), float(i + 0.01))

        interval_low_noise = enbpi_calibrator.predict_interval(100.0)

        # Reset and use high noise data
        enbpi_calibrator._residuals.clear()
        for i in range(50):
            enbpi_calibrator.partial_fit(float(i), float(i + np.random.randn()))

        interval_high_noise = enbpi_calibrator.predict_interval(100.0)

        # High noise should give wider intervals
        assert interval_high_noise.interval_width > interval_low_noise.interval_width

    def test_batch_calibration(self, enbpi_calibrator):
        """Test batch calibration from arrays."""
        np.random.seed(42)
        predictions = np.arange(100, dtype=float)
        true_values = predictions + np.random.randn(100) * 0.5

        result = enbpi_calibrator.calibrate(predictions, true_values)

        assert result.success
        assert enbpi_calibrator.is_calibrated

    def test_uncalibrated_returns_point_estimate(self, enbpi_calibrator):
        """Test uncalibrated returns point estimate as bounds."""
        interval = enbpi_calibrator.predict_interval(100.0)

        assert not interval.is_calibrated
        assert interval.lower_bound == interval.point_estimate
        assert interval.upper_bound == interval.point_estimate


# =============================================================================
# Test ACI Calibrator
# =============================================================================


class TestACICalibrator:
    """Test Adaptive Conformal Inference."""

    def test_alpha_adapts_to_miscoverage(self, aci_calibrator):
        """Test alpha adapts when coverage is wrong."""
        np.random.seed(42)
        initial_alpha = aci_calibrator.current_alpha

        # Populate residual buffer first
        for i in range(30):
            aci_calibrator._residuals.append(0.1)

        # Simulate consistent under-coverage (true values outside intervals)
        for i in range(50):
            pred = float(i)
            interval = aci_calibrator.predict_interval(pred)
            # Report true value way outside
            aci_calibrator.update(interval, pred + 10.0)

        # Alpha should have increased (widening intervals)
        assert aci_calibrator.current_alpha > initial_alpha

    def test_empirical_coverage_tracking(self, aci_calibrator):
        """Test empirical coverage is tracked correctly."""
        # All covered
        for i in range(20):
            interval = PredictionInterval(
                point_estimate=0.0,
                lower_bound=-1.0,
                upper_bound=1.0,
                coverage_target=0.9,
                method=ConformalMethod.ACI,
                is_calibrated=True,
            )
            aci_calibrator.update(interval, 0.5)  # Inside interval

        assert aci_calibrator.empirical_coverage == 1.0

    def test_partial_fit(self, aci_calibrator):
        """Test partial_fit updates state."""
        for i in range(30):
            aci_calibrator.partial_fit(float(i), float(i + 0.1))

        assert aci_calibrator.is_calibrated
        assert aci_calibrator.calibration_samples == 30


# =============================================================================
# Test Naive Calibrator
# =============================================================================


class TestNaiveCalibrator:
    """Test naive baseline calibrator."""

    def test_calibration_stores_residuals(self, cqr_config):
        """Test calibration stores residuals."""
        cqr_config = ConformalConfig(calibration=CalibrationConfig(method=ConformalMethod.NAIVE))
        calibrator = NaiveCalibrator(cqr_config)

        predictions = np.array([1.0, 2.0, 3.0])
        true_values = np.array([1.1, 2.2, 3.3])

        result = calibrator.calibrate(predictions, true_values)

        assert result.success
        assert calibrator.is_calibrated

    def test_predict_interval_uses_percentiles(self, cqr_config):
        """Test prediction uses percentiles."""
        config = ConformalConfig(calibration=CalibrationConfig(method=ConformalMethod.NAIVE))
        calibrator = NaiveCalibrator(config)

        # Add some residuals
        for i in range(50):
            calibrator.partial_fit(float(i), float(i + np.random.randn()))

        interval = calibrator.predict_interval(100.0)

        assert interval.is_calibrated
        assert interval.lower_bound < interval.point_estimate < interval.upper_bound


# =============================================================================
# Test CVaR Estimator
# =============================================================================


class TestConformalCVaREstimator:
    """Test CVaR with conformal bounds."""

    def test_cvar_bounds_structure(self, cqr_config):
        """Test CVaR bounds have correct structure."""
        estimator = ConformalCVaREstimator(cqr_config)
        quantiles = np.linspace(-2.0, 2.0, 21)

        bounds = estimator.compute_cvar_with_bounds(quantiles, alpha=0.05)

        assert isinstance(bounds, CVaRBounds)
        assert bounds.alpha == 0.05
        assert bounds.cvar_lower <= bounds.cvar_point <= bounds.cvar_upper
        assert bounds.var_lower <= bounds.var_point <= bounds.var_upper

    def test_worst_case_cvar(self, cqr_config):
        """Test worst case CVaR computation."""
        estimator = ConformalCVaREstimator(cqr_config)
        quantiles = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

        bounds = estimator.compute_cvar_with_bounds(quantiles, alpha=0.30)

        # Worst case should be the more conservative (more negative) bound
        assert bounds.worst_case_cvar <= bounds.cvar_point

    def test_empty_quantiles(self, cqr_config):
        """Test empty quantiles handling."""
        estimator = ConformalCVaREstimator(cqr_config)
        quantiles = np.array([])

        bounds = estimator.compute_cvar_with_bounds(quantiles, alpha=0.05)

        assert bounds.cvar_point == 0.0
        assert bounds.var_point == 0.0


# =============================================================================
# Test Uncertainty Tracker
# =============================================================================


class TestUncertaintyTracker:
    """Test uncertainty tracking and escalation."""

    def test_escalation_levels(self, cqr_config):
        """Test escalation level detection."""
        tracker = UncertaintyTrackerImpl(cqr_config)

        # Fill with normal intervals
        for _ in range(100):
            interval = PredictionInterval(
                point_estimate=1.0,
                lower_bound=0.9,
                upper_bound=1.1,
                coverage_target=0.9,
                method=ConformalMethod.CQR,
                is_calibrated=True,
            )
            state = tracker.record_interval(interval)

        assert state.escalation_level == EscalationLevel.NORMAL

        # Very wide interval should trigger higher escalation
        wide_interval = PredictionInterval(
            point_estimate=1.0,
            lower_bound=0.0,
            upper_bound=2.0,  # Width = 2.0, way above normal
            coverage_target=0.9,
            method=ConformalMethod.CQR,
            is_calibrated=True,
        )
        state = tracker.record_interval(wide_interval)
        assert state.escalation_level in (EscalationLevel.WARNING, EscalationLevel.CRITICAL)

    def test_position_scaling(self, cqr_config):
        """Test position scaling based on uncertainty."""
        tracker = UncertaintyTrackerImpl(cqr_config)

        # Normal interval - should get full scale
        normal = PredictionInterval(
            point_estimate=1.0,
            lower_bound=0.95,
            upper_bound=1.05,  # Width = 0.1 = baseline
            coverage_target=0.9,
            method=ConformalMethod.CQR,
            is_calibrated=True,
        )
        state = tracker.record_interval(normal)
        assert state.recommended_position_scale >= 0.9

        # Wide interval should reduce position
        wide = PredictionInterval(
            point_estimate=1.0,
            lower_bound=0.5,
            upper_bound=1.5,  # Width = 1.0
            coverage_target=0.9,
            method=ConformalMethod.CQR,
            is_calibrated=True,
        )
        state = tracker.record_interval(wide)
        assert state.recommended_position_scale < 1.0

    def test_get_statistics(self, cqr_config):
        """Test statistics computation."""
        tracker = UncertaintyTrackerImpl(cqr_config)

        for i in range(50):
            interval = PredictionInterval(
                point_estimate=1.0,
                lower_bound=0.9,
                upper_bound=1.1 + i * 0.01,
                coverage_target=0.9,
                method=ConformalMethod.CQR,
                is_calibrated=True,
            )
            tracker.record_interval(interval)

        stats = tracker.get_statistics()

        assert stats["count"] == 50
        assert stats["mean_width"] is not None
        assert stats["std_width"] is not None


# =============================================================================
# Test Conformal Prediction Service
# =============================================================================


class TestConformalPredictionService:
    """Test main service integration."""

    def test_disabled_service_returns_defaults(self, disabled_service):
        """Test disabled service returns safe defaults."""
        interval = disabled_service.predict_interval(1.0)
        assert interval.lower_bound == 1.0
        assert interval.upper_bound == 1.0
        assert not interval.is_calibrated

        state = disabled_service.get_uncertainty_state()
        assert state.escalation_level == EscalationLevel.NORMAL
        assert state.recommended_position_scale == 1.0

    def test_calibration_and_prediction(self, service):
        """Test calibration followed by prediction."""
        np.random.seed(42)

        # Generate calibration data
        predictions = np.random.randn(100)
        true_values = predictions + np.random.randn(100) * 0.2

        result = service.calibrate(predictions, true_values)
        assert result.success

        # Predict
        interval = service.predict_interval(0.5)
        assert interval.is_calibrated
        assert interval.lower_bound <= 0.5 <= interval.upper_bound

    def test_record_observation(self, service):
        """Test online observation recording."""
        for i in range(100):
            service.record_observation(float(i), float(i + 0.1))

        # Should have accumulated observations
        assert len(service._calibration_data["predictions"]) == 100

    def test_cvar_bounds_disabled(self, cqr_config):
        """Test CVaR bounds when disabled."""
        config = ConformalConfig(
            enabled=True,
            cvar_bounds=CVaRBoundsConfig(enabled=False),
        )
        service = ConformalPredictionService(config)

        quantiles = np.linspace(-1, 1, 21)
        bounds = service.compute_cvar_bounds(quantiles)

        # Should return point estimate as bounds
        assert bounds.cvar_lower == bounds.cvar_point
        assert bounds.cvar_upper == bounds.cvar_point

    def test_escalation_callbacks(self, service):
        """Test escalation callback registration and triggering."""
        callback_called = {"warning": False, "critical": False}

        def warning_callback(state):
            callback_called["warning"] = True

        def critical_callback(state):
            callback_called["critical"] = True

        service.register_escalation_callback(EscalationLevel.WARNING, warning_callback)
        service.register_escalation_callback(EscalationLevel.CRITICAL, critical_callback)

        # Generate normal intervals to establish baseline
        for _ in range(100):
            interval = PredictionInterval(
                point_estimate=1.0,
                lower_bound=0.9,
                upper_bound=1.1,
                coverage_target=0.9,
                method=ConformalMethod.CQR,
                is_calibrated=True,
            )
            service.get_uncertainty_state(interval)

        # Now trigger with very wide interval
        wide_interval = PredictionInterval(
            point_estimate=1.0,
            lower_bound=-10.0,
            upper_bound=10.0,
            coverage_target=0.9,
            method=ConformalMethod.CQR,
            is_calibrated=True,
        )
        service.get_uncertainty_state(wide_interval)

        # At least one callback should have been triggered
        assert callback_called["warning"] or callback_called["critical"]

    def test_get_position_scale(self, service):
        """Test position scale computation."""
        scale = service.get_position_scale(point_estimate=1.0)
        assert 0.0 <= scale <= 1.0

    def test_save_state(self, service, tmp_path):
        """Test state serialization."""
        state_file = tmp_path / "conformal_state.json"
        service.save_state(state_file)

        assert state_file.exists()
        import json

        with open(state_file) as f:
            state = json.load(f)

        assert "config" in state
        assert "calibrator" in state
        assert "uncertainty" in state


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_calibrator_cqr(self, cqr_config):
        """Test creating CQR calibrator."""
        calibrator = create_calibrator(cqr_config)
        assert isinstance(calibrator, CQRCalibrator)

    def test_create_calibrator_enbpi(self, enbpi_config):
        """Test creating EnbPI calibrator."""
        calibrator = create_calibrator(enbpi_config)
        assert isinstance(calibrator, EnbPICalibrator)

    def test_create_calibrator_aci(self, aci_config):
        """Test creating ACI calibrator."""
        calibrator = create_calibrator(aci_config)
        assert isinstance(calibrator, ACICalibrator)

    def test_create_calibrator_disabled_raises(self, disabled_config):
        """Test creating calibrator with disabled config raises."""
        with pytest.raises(ValueError, match="disabled"):
            create_calibrator(disabled_config)

    def test_create_conformal_service_from_dict(self):
        """Test creating service from dictionary."""
        config_dict = {
            "enabled": True,
            "calibration": {
                "method": "cqr",
                "coverage_target": 0.90,
                "min_calibration_samples": 50,
            },
        }
        service = create_conformal_service(config_dict)

        assert service.is_enabled()
        assert service.config.method == ConformalMethod.CQR


# =============================================================================
# Test Integration Helpers
# =============================================================================


class TestIntegrationHelpers:
    """Test integration helper functions."""

    def test_create_risk_guard_integration(self, service):
        """Test risk guard integration helper."""
        base_scale = lambda: 1000.0

        scaled_func = create_risk_guard_integration(service, base_scale)
        result = scaled_func()

        assert 0.0 <= result <= 1000.0

    def test_create_escalation_handler(self):
        """Test escalation handler creation."""
        reduce_called = {"value": None}
        halt_called = {"value": False}

        def reduce_callback(scale):
            reduce_called["value"] = scale

        def halt_callback():
            halt_called["value"] = True

        handlers = create_escalation_handler(
            reduce_position_callback=reduce_callback,
            halt_trading_callback=halt_callback,
        )

        assert EscalationLevel.WARNING in handlers
        assert EscalationLevel.CRITICAL in handlers

    def test_training_integration(self, service):
        """Test training integration helper."""
        integration = TrainingConformalIntegration(service, calibration_frequency=1)

        # Record some batches
        for _ in range(10):
            predictions = np.random.randn(32)
            true_values = predictions + np.random.randn(32) * 0.1
            integration.record_batch(predictions, true_values)

        # End epoch
        result = integration.on_epoch_end()

        metrics = integration.get_metrics()
        assert "conformal_enabled" in metrics
        assert "epoch_count" in metrics

    def test_live_trading_integration(self, service):
        """Test live trading integration helper."""
        reduce_called = {"value": None}

        def reduce_callback(scale):
            reduce_called["value"] = scale

        integration = LiveTradingConformalIntegration(
            service,
            reduce_position_callback=reduce_callback,
        )

        # Test position scale
        scale = integration.get_position_scale(1.0)
        assert 0.0 <= scale <= 1.0

        # Test recording trade result
        state = integration.record_trade_result(1.0, 1.1)
        assert isinstance(state, UncertaintyState)


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_quantiles_cvar(self, service):
        """Test CVaR with empty quantiles."""
        bounds = service.compute_cvar_bounds(np.array([]))
        assert bounds.cvar_point == 0.0

    def test_single_quantile_cvar(self, service):
        """Test CVaR with single quantile."""
        bounds = service.compute_cvar_bounds(np.array([1.0]))
        assert bounds.cvar_point == 1.0

    def test_nan_handling_in_interval(self, cqr_calibrator):
        """Test NaN handling in prediction intervals."""
        interval = cqr_calibrator.predict_interval(
            predicted_lower=float("nan"),
            predicted_upper=1.0,
            point_estimate=0.5,
        )
        # Should not crash, but interval may not be valid
        assert not interval.is_valid or np.isnan(interval.lower_bound)

    def test_inf_handling_in_interval(self, cqr_calibrator):
        """Test infinity handling in prediction intervals."""
        interval = cqr_calibrator.predict_interval(
            predicted_lower=float("-inf"),
            predicted_upper=float("inf"),
            point_estimate=0.0,
        )
        # Should not crash
        assert interval.interval_width == float("inf")

    def test_calibration_with_constant_values(self, cqr_calibrator):
        """Test calibration with constant values."""
        predictions = np.ones(100)
        true_values = np.ones(100)

        result = cqr_calibrator.calibrate(predictions, predictions, true_values)

        assert result.success
        assert result.calibration_quantile == 0.0  # No residuals

    def test_recalibration_trigger(self, cqr_config):
        """Test automatic recalibration trigger."""
        config = ConformalConfig(
            enabled=True,
            calibration=CalibrationConfig(
                min_calibration_samples=10,
                recalibrate_interval=20,
            ),
        )
        service = ConformalPredictionService(config)

        # Initial calibration
        predictions = np.random.randn(50)
        true_values = predictions + np.random.randn(50) * 0.1
        service.calibrate(predictions, true_values)

        # Record observations to trigger recalibration
        for i in range(25):
            service.record_observation(float(i), float(i + 0.1))

        # Should have triggered recalibration at step 20, then 5 more steps
        # recalibrate_interval=20 → triggers at step 20, resets to 0, then 5 more = 5
        assert service._steps_since_calibration == 5


# =============================================================================
# Test Data Types
# =============================================================================


class TestDataTypes:
    """Test data type serialization and methods."""

    def test_prediction_interval_to_dict(self):
        """Test PredictionInterval serialization."""
        interval = PredictionInterval(
            point_estimate=1.0,
            lower_bound=0.5,
            upper_bound=1.5,
            coverage_target=0.90,
            method=ConformalMethod.CQR,
            is_calibrated=True,
            calibration_samples=100,
        )

        d = interval.to_dict()

        assert d["point_estimate"] == 1.0
        assert d["lower_bound"] == 0.5
        assert d["upper_bound"] == 1.5
        assert d["interval_width"] == 1.0
        assert d["method"] == "CQR"

    def test_prediction_interval_contains(self):
        """Test PredictionInterval.contains method."""
        interval = PredictionInterval(
            point_estimate=1.0,
            lower_bound=0.5,
            upper_bound=1.5,
            coverage_target=0.90,
            method=ConformalMethod.CQR,
        )

        assert interval.contains(1.0)
        assert interval.contains(0.5)
        assert interval.contains(1.5)
        assert not interval.contains(0.0)
        assert not interval.contains(2.0)

    def test_cvar_bounds_to_dict(self):
        """Test CVaRBounds serialization."""
        interval = PredictionInterval(
            point_estimate=-0.5,
            lower_bound=-1.0,
            upper_bound=0.0,
            coverage_target=0.90,
            method=ConformalMethod.CQR,
        )

        bounds = CVaRBounds(
            cvar_point=-0.5,
            cvar_lower=-1.0,
            cvar_upper=0.0,
            var_point=-0.3,
            var_lower=-0.5,
            var_upper=-0.1,
            alpha=0.05,
            interval=interval,
        )

        d = bounds.to_dict()

        assert d["cvar_point"] == -0.5
        assert d["alpha"] == 0.05
        assert "worst_case_cvar" in d
        assert "interval" in d

    def test_uncertainty_state_to_dict(self):
        """Test UncertaintyState serialization."""
        state = UncertaintyState(
            current_interval_width=0.5,
            historical_percentile=75.0,
            escalation_level=EscalationLevel.WARNING,
            recommended_action=EscalationAction.REDUCE_POSITION,
            recommended_position_scale=0.7,
            samples_since_calibration=100,
            is_calibrated=True,
        )

        d = state.to_dict()

        assert d["current_interval_width"] == 0.5
        assert d["escalation_level"] == "WARNING"
        assert d["recommended_action"] == "REDUCE_POSITION"

    def test_calibration_result_to_dict(self):
        """Test CalibrationResult serialization."""
        result = CalibrationResult(
            success=True,
            samples_used=500,
            calibration_quantile=0.15,
            empirical_coverage=0.91,
            method=ConformalMethod.CQR,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["samples_used"] == 500
        assert d["method"] == "CQR"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
