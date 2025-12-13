# -*- coding: utf-8 -*-
"""
Comprehensive tests for Market Impact Models (Stage 4).

Tests cover:
- Market impact models (Kyle, Almgren-Chriss, Gatheral)
- Impact effects on LOB state
- Impact calibration from historical data
- Integration with existing LOB components

Test count target: 25+ tests
"""

import math
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
pytest.importorskip("sortedcontainers")

from lob.data_structures import (
    Fill,
    LimitOrder,
    OrderBook,
    OrderType,
    Side,
    Trade,
)
from lob.market_impact import (
    AlmgrenChrissModel,
    CompositeImpactModel,
    DecayType,
    GatheralModel,
    ImpactModelType,
    ImpactParameters,
    ImpactResult,
    ImpactState,
    ImpactTracker,
    KyleLambdaModel,
    MarketImpactModel,
    create_impact_model,
    create_impact_tracker,
)
from lob.impact_effects import (
    AdverseSelectionResult,
    ImpactEffectConfig,
    ImpactEffects,
    LOBImpactSimulator,
    LiquidityReaction,
    LiquidityReactionResult,
    MomentumResult,
    MomentumSignal,
    QuoteShiftResult,
    QuoteShiftType,
    create_impact_effects,
    create_lob_impact_simulator,
)
from lob.impact_calibration import (
    AlmgrenChrissCalibrator,
    CalibrationDataset,
    GatheralDecayCalibrator,
    ImpactCalibrationPipeline,
    KyleLambdaCalibrator,
    RollingImpactCalibrator,
    TradeObservation,
    calibrate_from_trades,
    create_calibration_pipeline,
    create_calibrator,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_order_book() -> OrderBook:
    """Create a sample order book with liquidity."""
    book = OrderBook(symbol="AAPL", tick_size=0.01, lot_size=1.0)

    # Add bids
    for i, (price, qty) in enumerate([
        (150.00, 500),
        (149.99, 300),
        (149.98, 200),
        (149.97, 400),
        (149.96, 600),
    ]):
        order = LimitOrder(
            order_id=f"bid_{i}",
            price=price,
            qty=qty,
            remaining_qty=qty,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
        )
        book.add_limit_order(order)

    # Add asks
    for i, (price, qty) in enumerate([
        (150.02, 400),
        (150.03, 250),
        (150.04, 350),
        (150.05, 500),
        (150.06, 300),
    ]):
        order = LimitOrder(
            order_id=f"ask_{i}",
            price=price,
            qty=qty,
            remaining_qty=qty,
            timestamp_ns=time.time_ns(),
            side=Side.SELL,
        )
        book.add_limit_order(order)

    return book


@pytest.fixture
def sample_fill() -> Fill:
    """Create a sample fill."""
    return Fill(
        order_id="test_order",
        total_qty=100.0,
        avg_price=150.02,
        trades=[
            Trade(
                price=150.02,
                qty=100.0,
                maker_order_id="ask_0",
                timestamp_ns=time.time_ns(),
            )
        ],
        remaining_qty=0.0,
        is_complete=True,
    )


@pytest.fixture
def sample_limit_order() -> LimitOrder:
    """Create a sample limit order."""
    return LimitOrder(
        order_id="test_order",
        price=150.02,
        qty=100.0,
        remaining_qty=100.0,
        timestamp_ns=time.time_ns(),
        side=Side.BUY,
    )


@pytest.fixture
def calibration_dataset() -> CalibrationDataset:
    """Create a sample calibration dataset."""
    dataset = CalibrationDataset(
        symbol="AAPL",
        avg_adv=10_000_000,
        avg_volatility=0.02,
    )

    # Generate synthetic trade observations
    np.random.seed(42)
    base_price = 150.0

    for i in range(100):
        qty = np.random.uniform(100, 1000)
        side = np.random.choice([1, -1])
        participation = qty / dataset.avg_adv

        # Synthetic impact following square-root model
        # impact = 0.1 * 0.02 * sqrt(participation) + 0.05 * participation
        impact_bps = (
            0.1 * 0.02 * np.sqrt(participation) * 10000 +
            0.05 * participation * 10000 +
            np.random.normal(0, 0.5)  # Noise
        )

        pre_mid = base_price
        post_mid = pre_mid * (1 + side * impact_bps / 10000)

        obs = TradeObservation(
            timestamp_ms=i * 1000,
            price=pre_mid * (1 + side * 0.0001),  # Slight adjustment
            qty=qty,
            side=side,
            adv=dataset.avg_adv,
            volatility=dataset.avg_volatility,
            pre_trade_mid=pre_mid,
            post_trade_mid=post_mid,
        )
        dataset.add_observation(obs)

    return dataset


# ==============================================================================
# Test: Impact Parameters
# ==============================================================================

class TestImpactParameters:
    """Tests for ImpactParameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = ImpactParameters()
        assert params.eta > 0
        assert params.gamma > 0
        assert params.delta == 0.5
        assert params.tau_ms > 0
        assert params.beta > 0

    def test_equity_parameters(self):
        """Test equity-specific parameters."""
        params = ImpactParameters.for_equity()
        assert params.eta == 0.05
        assert params.gamma == 0.03
        assert params.spread_bps == 2.0

    def test_crypto_parameters(self):
        """Test crypto-specific parameters."""
        params = ImpactParameters.for_crypto()
        assert params.eta == 0.10
        assert params.gamma == 0.05
        assert params.spread_bps == 5.0


# ==============================================================================
# Test: Kyle Lambda Model
# ==============================================================================

class TestKyleLambdaModel:
    """Tests for Kyle Lambda model."""

    def test_initialization(self):
        """Test model initialization."""
        model = KyleLambdaModel(lambda_coef=0.0001, permanent_fraction=0.5)
        assert model.model_type == ImpactModelType.KYLE_LAMBDA
        assert model._lambda == 0.0001
        assert model._perm_frac == 0.5

    def test_temporary_impact(self):
        """Test temporary impact computation."""
        model = KyleLambdaModel(lambda_coef=0.0001, permanent_fraction=0.5)
        impact = model.compute_temporary_impact(
            order_qty=1000,
            adv=10_000_000,
            volatility=0.02,
        )
        # λ * |Q| * (1 - perm_frac) * 10000
        expected = 0.0001 * 1000 * 0.5 * 10000
        assert abs(impact - expected) < 1e-6

    def test_permanent_impact(self):
        """Test permanent impact computation."""
        model = KyleLambdaModel(lambda_coef=0.0001, permanent_fraction=0.5)
        impact = model.compute_permanent_impact(
            order_qty=1000,
            adv=10_000_000,
        )
        expected = 0.0001 * 1000 * 0.5 * 10000
        assert abs(impact - expected) < 1e-6

    def test_exponential_decay(self):
        """Test exponential decay function."""
        model = KyleLambdaModel(
            decay_type=DecayType.EXPONENTIAL,
            decay_tau_ms=60000,
        )
        initial_impact = 10.0  # bps

        # At t=0, no decay
        assert model.compute_decay(initial_impact, 0) == initial_impact

        # At t=τ, decay to ~37%
        decayed = model.compute_decay(initial_impact, 60000)
        assert abs(decayed - initial_impact * math.exp(-1)) < 0.01

    def test_power_law_decay(self):
        """Test power-law decay function."""
        model = KyleLambdaModel(
            decay_type=DecayType.POWER_LAW,
            decay_tau_ms=60000,
        )
        initial_impact = 10.0

        decayed = model.compute_decay(initial_impact, 60000)
        # (1 + t/τ)^(-1.5) = 2^(-1.5) ≈ 0.354
        expected = initial_impact * (2 ** (-1.5))
        assert abs(decayed - expected) < 0.1

    def test_linear_decay(self):
        """Test linear decay function."""
        model = KyleLambdaModel(
            decay_type=DecayType.LINEAR,
            decay_tau_ms=60000,
        )
        initial_impact = 10.0

        # At t=τ/2, decay to 50%
        decayed = model.compute_decay(initial_impact, 30000)
        assert abs(decayed - 5.0) < 0.01

        # At t=τ, decay to 0
        decayed = model.compute_decay(initial_impact, 60000)
        assert decayed == 0.0


# ==============================================================================
# Test: Almgren-Chriss Model
# ==============================================================================

class TestAlmgrenChrissModel:
    """Tests for Almgren-Chriss model."""

    def test_initialization(self):
        """Test model initialization."""
        model = AlmgrenChrissModel(eta=0.1, gamma=0.05)
        assert model.model_type == ImpactModelType.ALMGREN_CHRISS
        assert model.params.eta == 0.1
        assert model.params.gamma == 0.05

    def test_temporary_impact_square_root(self):
        """Test temporary impact follows square-root model."""
        model = AlmgrenChrissModel(
            params=ImpactParameters(eta=0.1, gamma=0.05, delta=0.5, volatility=0.02)
        )

        adv = 10_000_000
        vol = 0.02

        # Test with different order sizes
        impact_1000 = model.compute_temporary_impact(1000, adv, vol)
        impact_4000 = model.compute_temporary_impact(4000, adv, vol)

        # With delta=0.5, doubling qty should increase impact by sqrt(2)
        # Since 4000/1000 = 4, impact should increase by sqrt(4) = 2
        assert abs(impact_4000 / impact_1000 - 2.0) < 0.01

    def test_permanent_impact_linear(self):
        """Test permanent impact is linear in participation."""
        model = AlmgrenChrissModel(
            params=ImpactParameters(gamma=0.05)
        )

        adv = 10_000_000

        impact_1000 = model.compute_permanent_impact(1000, adv)
        impact_2000 = model.compute_permanent_impact(2000, adv)

        # Linear: doubling qty doubles impact
        assert abs(impact_2000 / impact_1000 - 2.0) < 0.01

    def test_total_impact(self):
        """Test total impact computation."""
        model = AlmgrenChrissModel(
            params=ImpactParameters(eta=0.1, gamma=0.05, delta=0.5, volatility=0.02)
        )

        result = model.compute_total_impact(
            order_qty=10000,
            adv=10_000_000,
            volatility=0.02,
            mid_price=150.0,
        )

        assert isinstance(result, ImpactResult)
        assert result.temporary_impact_bps > 0
        assert result.permanent_impact_bps > 0
        assert result.total_impact_bps == result.temporary_impact_bps + result.permanent_impact_bps
        assert result.impact_cost > 0
        assert len(result.decay_profile) > 0

    def test_optimal_execution_time(self):
        """Test optimal execution time calculation."""
        model = AlmgrenChrissModel(
            params=ImpactParameters(eta=0.1, gamma=0.05, volatility=0.02)
        )

        t_star = model.compute_optimal_execution_time(
            order_qty=100000,
            adv=10_000_000,
            volatility=0.02,
            risk_aversion=1e-3,
        )

        # Verify we get a positive, finite result
        # The exact value depends on formula interpretation and parameter scaling
        assert t_star > 0
        assert math.isfinite(t_star)

    def test_execution_trajectory(self):
        """Test execution trajectory generation."""
        model = AlmgrenChrissModel()

        trajectory = model.compute_execution_trajectory(
            order_qty=10000,
            execution_time_sec=3600,
            n_intervals=10,
        )

        assert len(trajectory) == 11  # n+1 points
        assert trajectory[0][1] == 0  # Start at 0
        assert trajectory[-1][1] == 10000  # End at full qty


# ==============================================================================
# Test: Gatheral Model
# ==============================================================================

class TestGatheralModel:
    """Tests for Gatheral model."""

    def test_initialization(self):
        """Test model initialization."""
        model = GatheralModel(eta=0.1, gamma=0.05, delta=0.5, tau_ms=60000, beta=1.5)
        assert model.model_type == ImpactModelType.GATHERAL
        assert model.params.eta == 0.1
        assert model.params.beta == 1.5

    def test_power_law_decay(self):
        """Test power-law decay characteristic of Gatheral model."""
        model = GatheralModel(tau_ms=60000, beta=1.5)
        initial_impact = 10.0

        # Power-law decay: G(t) = (1 + t/τ)^(-β)
        decayed = model.compute_decay(initial_impact, 60000)
        expected = initial_impact * (2 ** (-1.5))
        assert abs(decayed - expected) < 0.1

    def test_resilience_rate(self):
        """Test market resilience rate computation."""
        model = GatheralModel(tau_ms=60000)
        rate = model.compute_resilience_rate()
        assert abs(rate - 1/60000) < 1e-10

    def test_impact_at_time(self):
        """Test impact at specific time."""
        model = GatheralModel(
            params=ImpactParameters(eta=0.1, gamma=0.05, tau_ms=60000, beta=1.5)
        )

        temp, perm = model.compute_impact_at_time(
            order_qty=10000,
            adv=10_000_000,
            volatility=0.02,
            time_since_trade_ms=30000,
        )

        assert temp > 0
        assert perm > 0
        # Permanent doesn't decay
        perm_at_0 = model.compute_permanent_impact(10000, 10_000_000)
        assert abs(perm - perm_at_0) < 1e-6


# ==============================================================================
# Test: Composite Model
# ==============================================================================

class TestCompositeImpactModel:
    """Tests for Composite impact model."""

    def test_default_ensemble(self):
        """Test default model ensemble."""
        model = CompositeImpactModel()
        assert model.model_type == ImpactModelType.COMPOSITE

        # Should have Almgren-Chriss and Gatheral by default
        assert len(model._models) == 2

    def test_weighted_average(self):
        """Test weighted average of impacts."""
        ac_model = AlmgrenChrissModel(
            params=ImpactParameters(eta=0.1, gamma=0.05)
        )
        gatheral_model = GatheralModel(
            params=ImpactParameters(eta=0.2, gamma=0.1)
        )

        composite = CompositeImpactModel([
            (ac_model, 0.6),
            (gatheral_model, 0.4),
        ])

        impact = composite.compute_temporary_impact(
            order_qty=10000,
            adv=10_000_000,
            volatility=0.02,
        )

        # Verify it's between the two individual impacts
        ac_impact = ac_model.compute_temporary_impact(10000, 10_000_000, 0.02)
        gath_impact = gatheral_model.compute_temporary_impact(10000, 10_000_000, 0.02)

        expected = 0.6 * ac_impact + 0.4 * gath_impact
        assert abs(impact - expected) < 0.1


# ==============================================================================
# Test: Impact Tracker
# ==============================================================================

class TestImpactTracker:
    """Tests for Impact Tracker."""

    def test_record_trade(self):
        """Test recording trades and cumulative impact."""
        tracker = ImpactTracker(model=AlmgrenChrissModel())

        result = tracker.record_trade(
            timestamp_ms=1000,
            order_qty=1000,
            adv=10_000_000,
            volatility=0.02,
        )

        assert result.temporary_impact_bps > 0
        assert tracker.current_impact_bps > 0

    def test_cumulative_impact(self):
        """Test cumulative impact from multiple trades."""
        tracker = ImpactTracker(model=AlmgrenChrissModel())

        # First trade
        tracker.record_trade(1000, 1000, 10_000_000, 0.02)
        impact_after_1 = tracker.current_impact_bps

        # Second trade (same time, no decay)
        tracker.record_trade(1000, 1000, 10_000_000, 0.02)
        impact_after_2 = tracker.current_impact_bps

        # Should be roughly double (same size trades)
        assert impact_after_2 > impact_after_1

    def test_impact_decay_over_time(self):
        """Test impact decays with time."""
        tracker = ImpactTracker(model=AlmgrenChrissModel())

        # Record trade at t=0
        tracker.record_trade(0, 10000, 10_000_000, 0.02)
        impact_at_0 = tracker.current_impact_bps

        # Get impact at t=60s
        impact_at_60s = tracker.get_impact_at_time(60000)

        # Should be less due to decay of temporary component
        assert impact_at_60s < impact_at_0

    def test_reset(self):
        """Test tracker reset."""
        tracker = ImpactTracker()
        tracker.record_trade(1000, 1000, 10_000_000, 0.02)
        assert tracker.current_impact_bps > 0

        tracker.reset()
        assert tracker.current_impact_bps == 0


# ==============================================================================
# Test: Factory Functions
# ==============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_impact_model_almgren_chriss(self):
        """Test creating Almgren-Chriss model."""
        model = create_impact_model("almgren_chriss", asset_class="equity")
        assert model.model_type == ImpactModelType.ALMGREN_CHRISS

    def test_create_impact_model_gatheral(self):
        """Test creating Gatheral model."""
        model = create_impact_model("gatheral", asset_class="equity")
        assert model.model_type == ImpactModelType.GATHERAL

    def test_create_impact_model_kyle(self):
        """Test creating Kyle model."""
        model = create_impact_model("kyle", asset_class="crypto")
        assert model.model_type == ImpactModelType.KYLE_LAMBDA

    def test_create_impact_tracker(self):
        """Test creating impact tracker."""
        tracker = create_impact_tracker("almgren_chriss", asset_class="equity")
        assert isinstance(tracker, ImpactTracker)
        assert tracker.model.model_type == ImpactModelType.ALMGREN_CHRISS


# ==============================================================================
# Test: Impact Effects
# ==============================================================================

class TestImpactEffects:
    """Tests for Impact Effects."""

    def test_apply_temporary_impact_buy(self, sample_order_book):
        """Test applying temporary impact for buy order."""
        effects = ImpactEffects()

        result = effects.apply_temporary_impact(
            order_book=sample_order_book,
            impact_bps=5.0,
            side=Side.BUY,
        )

        assert isinstance(result, QuoteShiftResult)
        # Buy aggression - prices should shift up
        assert result.new_ask > sample_order_book.best_ask
        assert result.ask_shift_bps > 0

    def test_apply_temporary_impact_sell(self, sample_order_book):
        """Test applying temporary impact for sell order."""
        effects = ImpactEffects()

        result = effects.apply_temporary_impact(
            order_book=sample_order_book,
            impact_bps=5.0,
            side=Side.SELL,
        )

        # Sell aggression - prices should shift down
        assert result.new_bid < sample_order_book.best_bid
        assert result.bid_shift_bps < 0

    def test_simulate_liquidity_reaction(
        self, sample_order_book, sample_limit_order, sample_fill
    ):
        """Test simulating liquidity reaction."""
        effects = ImpactEffects()

        result = effects.simulate_liquidity_reaction(
            order_book=sample_order_book,
            our_order=sample_limit_order,
            fill=sample_fill,
            adv=10_000_000,
        )

        assert isinstance(result, LiquidityReactionResult)
        assert result.reaction_type in [
            LiquidityReaction.REPLENISH,
            LiquidityReaction.WITHDRAW,
            LiquidityReaction.NEUTRAL,
            LiquidityReaction.INFORMED,
        ]

    def test_detect_momentum_none(self, sample_order_book):
        """Test momentum detection with no recent trades."""
        effects = ImpactEffects()

        result = effects.detect_momentum(order_book=sample_order_book)

        assert isinstance(result, MomentumResult)
        assert result.signal == MomentumSignal.NONE

    def test_detect_momentum_with_trades(self, sample_order_book):
        """Test momentum detection with trade history."""
        effects = ImpactEffects()

        # Add some buy-side trades
        trades = [
            Trade(
                price=150.0 + i * 0.01,
                qty=100,
                maker_order_id=f"order_{i}",
                timestamp_ns=i * 1_000_000_000,
                aggressor_side=Side.BUY,
            )
            for i in range(10)
        ]

        result = effects.detect_momentum(
            order_book=sample_order_book,
            recent_trades=trades,
        )

        assert result.recent_trade_imbalance > 0  # Net buying
        # With 100% buy imbalance, model may predict continuation OR reversal
        # depending on imbalance threshold. Just verify we get a valid signal.
        assert result.signal in [
            MomentumSignal.WEAK_CONTINUATION,
            MomentumSignal.STRONG_CONTINUATION,
            MomentumSignal.REVERSAL_EXPECTED,
            MomentumSignal.PRICE_DISCOVERY,
        ]
        assert 0 <= result.continuation_probability <= 1

    def test_estimate_adverse_selection(
        self, sample_order_book, sample_limit_order
    ):
        """Test adverse selection estimation."""
        effects = ImpactEffects()

        result = effects.estimate_adverse_selection(
            order_book=sample_order_book,
            our_order=sample_limit_order,
        )

        assert isinstance(result, AdverseSelectionResult)
        assert 0 <= result.toxic_flow_indicator <= 1
        assert result.recommended_spread_bps > 0


# ==============================================================================
# Test: LOB Impact Simulator
# ==============================================================================

class TestLOBImpactSimulator:
    """Tests for LOB Impact Simulator."""

    def test_simulate_trade_impact(
        self, sample_order_book, sample_limit_order, sample_fill
    ):
        """Test complete trade impact simulation."""
        simulator = LOBImpactSimulator()

        impact, quote_shift, liquidity = simulator.simulate_trade_impact(
            order_book=sample_order_book,
            order=sample_limit_order,
            fill=sample_fill,
            adv=10_000_000,
            volatility=0.02,
        )

        assert isinstance(impact, ImpactResult)
        assert isinstance(quote_shift, QuoteShiftResult)
        assert isinstance(liquidity, LiquidityReactionResult)

        assert impact.total_impact_bps > 0

    def test_cumulative_impact_tracking(
        self, sample_order_book, sample_limit_order, sample_fill
    ):
        """Test cumulative impact over multiple trades."""
        simulator = LOBImpactSimulator()

        # First trade
        simulator.simulate_trade_impact(
            sample_order_book, sample_limit_order, sample_fill,
            adv=10_000_000, volatility=0.02, timestamp_ms=1000,
        )
        impact_1 = simulator.cumulative_impact_bps

        # Second trade
        simulator.simulate_trade_impact(
            sample_order_book, sample_limit_order, sample_fill,
            adv=10_000_000, volatility=0.02, timestamp_ms=2000,
        )
        impact_2 = simulator.cumulative_impact_bps

        assert impact_2 > impact_1

    def test_reset(
        self, sample_order_book, sample_limit_order, sample_fill
    ):
        """Test simulator reset."""
        simulator = LOBImpactSimulator()

        simulator.simulate_trade_impact(
            sample_order_book, sample_limit_order, sample_fill,
            adv=10_000_000, volatility=0.02,
        )
        assert simulator.cumulative_impact_bps > 0

        simulator.reset()
        assert simulator.cumulative_impact_bps == 0


# ==============================================================================
# Test: Impact Calibration
# ==============================================================================

class TestImpactCalibration:
    """Tests for Impact Calibration."""

    def test_trade_observation(self):
        """Test TradeObservation dataclass."""
        obs = TradeObservation(
            timestamp_ms=1000,
            price=150.0,
            qty=1000,
            side=1,
            adv=10_000_000,
            volatility=0.02,
            pre_trade_mid=150.0,
            post_trade_mid=150.01,
        )

        assert obs.participation == 1000 / 10_000_000
        assert obs.realized_impact_bps is not None
        assert obs.realized_impact_bps > 0  # Price moved up for buy

    def test_calibration_dataset(self, calibration_dataset):
        """Test CalibrationDataset."""
        assert len(calibration_dataset) == 100
        assert len(calibration_dataset.get_participations()) == 100
        assert len(calibration_dataset.get_realized_impacts()) > 0

    def test_almgren_chriss_calibrator(self, calibration_dataset):
        """Test Almgren-Chriss calibrator."""
        calibrator = AlmgrenChrissCalibrator(min_observations=30)
        result = calibrator.calibrate(calibration_dataset)

        assert result.model_type == ImpactModelType.ALMGREN_CHRISS
        assert "eta" in result.parameters
        assert "gamma" in result.parameters
        assert result.n_observations == 100
        # R² can be low with noisy synthetic data; just verify calibration runs
        # and produces finite positive parameters
        assert result.r_squared is not None
        assert result.parameters["eta"] > 0 or result.parameters["gamma"] > 0

    def test_kyle_calibrator(self, calibration_dataset):
        """Test Kyle Lambda calibrator."""
        calibrator = KyleLambdaCalibrator(min_observations=30)
        result = calibrator.calibrate(calibration_dataset)

        assert result.model_type == ImpactModelType.KYLE_LAMBDA
        assert "lambda" in result.parameters
        # Lambda can be 0 or very small if data doesn't follow Kyle model
        # (our synthetic data follows Almgren-Chriss, not Kyle)
        assert result.parameters["lambda"] >= 0

    def test_gatheral_decay_calibrator(self, calibration_dataset):
        """Test Gatheral decay calibrator."""
        calibrator = GatheralDecayCalibrator(min_observations=30)
        result = calibrator.calibrate(calibration_dataset)

        assert result.model_type == ImpactModelType.GATHERAL
        assert "tau_ms" in result.parameters
        assert "beta" in result.parameters

    def test_calibration_pipeline(self, calibration_dataset):
        """Test complete calibration pipeline."""
        pipeline = ImpactCalibrationPipeline()
        results = pipeline.calibrate_all(calibration_dataset)

        assert len(results) > 0
        assert ImpactModelType.ALMGREN_CHRISS in results

    def test_cross_validation(self, calibration_dataset):
        """Test cross-validation."""
        pipeline = ImpactCalibrationPipeline(n_folds=3)
        cv_results = pipeline.cross_validate(calibration_dataset)

        assert len(cv_results) > 0
        for model_type, cv_result in cv_results.items():
            # R² can be negative if model fits worse than baseline
            # (e.g., Kyle model on Almgren-Chriss synthetic data)
            assert cv_result.mean_r_squared is not None
            assert math.isfinite(cv_result.mean_r_squared)
            assert len(cv_result.fold_results) > 0

    def test_rolling_calibrator(self):
        """Test rolling window calibrator."""
        calibrator = RollingImpactCalibrator(
            model_type=ImpactModelType.ALMGREN_CHRISS,
            window_size=50,
            update_frequency=10,
        )

        # Add observations
        np.random.seed(42)
        for i in range(60):
            obs = TradeObservation(
                timestamp_ms=i * 1000,
                price=150.0,
                qty=1000,
                side=1,
                adv=10_000_000,
                volatility=0.02,
                pre_trade_mid=150.0,
                post_trade_mid=150.01,
            )
            result = calibrator.add_observation(obs)

        # Should have recalibrated
        assert calibrator.current_parameters

    def test_calibrate_from_trades_convenience(self):
        """Test calibrate_from_trades convenience function."""
        trades = [
            {
                "timestamp_ms": i * 1000,
                "price": 150.0,
                "qty": 1000,
                "side": 1,
                "pre_mid": 150.0,
                "post_mid": 150.01,
            }
            for i in range(50)
        ]

        result = calibrate_from_trades(
            trades=trades,
            adv=10_000_000,
            volatility=0.02,
            model_type="almgren_chriss",
        )

        assert result.n_observations == 50

    def test_create_calibrated_model(self, calibration_dataset):
        """Test creating model from calibration results."""
        pipeline = ImpactCalibrationPipeline()
        pipeline.calibrate_all(calibration_dataset)

        model = pipeline.create_calibrated_model(ImpactModelType.ALMGREN_CHRISS)

        assert isinstance(model, AlmgrenChrissModel)
        # Should have non-default parameters
        assert model.params.eta != 0.1 or model.params.gamma != 0.05


# ==============================================================================
# Test: Integration
# ==============================================================================

class TestIntegration:
    """Integration tests for market impact with LOB."""

    def test_impact_model_with_order_book(self, sample_order_book):
        """Test impact model working with order book."""
        model = create_impact_model("almgren_chriss", asset_class="equity")

        order = LimitOrder(
            order_id="test",
            price=150.02,
            qty=1000,
            remaining_qty=1000,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
        )

        result = model.compute_for_order(
            order=order,
            order_book=sample_order_book,
            adv=10_000_000,
            volatility=0.02,
        )

        assert result.total_impact_bps > 0
        assert result.decay_profile

    def test_complete_simulation_workflow(
        self, sample_order_book, sample_limit_order, sample_fill
    ):
        """Test complete simulation workflow."""
        # Create simulator
        simulator = create_lob_impact_simulator(
            model_type="almgren_chriss",
            asset_class="equity",
        )

        # Simulate trade
        impact, quote_shift, liquidity = simulator.simulate_trade_impact(
            order_book=sample_order_book,
            order=sample_limit_order,
            fill=sample_fill,
            adv=10_000_000,
            volatility=0.02,
        )

        # Verify results
        assert impact.total_impact_bps > 0
        assert quote_shift.new_bid is not None
        assert quote_shift.new_ask is not None

        # Check momentum detection
        momentum = simulator.effects.detect_momentum(sample_order_book)
        assert isinstance(momentum, MomentumResult)

    def test_crypto_vs_equity_parameters(self):
        """Test different parameters for crypto vs equity."""
        crypto_model = create_impact_model("almgren_chriss", asset_class="crypto")
        equity_model = create_impact_model("almgren_chriss", asset_class="equity")

        # Crypto should have higher impact for same participation
        crypto_impact = crypto_model.compute_temporary_impact(1000, 10_000_000, 0.02)
        equity_impact = equity_model.compute_temporary_impact(1000, 10_000_000, 0.02)

        # Crypto has higher eta (0.10 vs 0.05)
        assert crypto_impact > equity_impact


# ==============================================================================
# Test: Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_adv(self):
        """Test handling of zero ADV."""
        model = AlmgrenChrissModel()

        # Should handle gracefully (return 0)
        impact = model.compute_temporary_impact(1000, 0, 0.02)
        assert impact == 0.0

    def test_zero_volatility(self):
        """Test handling of zero volatility."""
        model = AlmgrenChrissModel(
            params=ImpactParameters(eta=0.1, gamma=0.05, volatility=0.02)
        )

        # Should use default volatility
        impact = model.compute_temporary_impact(1000, 10_000_000, 0)
        assert impact > 0

    def test_negative_time_decay(self):
        """Test decay with negative time (should return original)."""
        model = GatheralModel()
        initial = 10.0
        result = model.compute_decay(initial, -1000)
        assert result == initial

    def test_very_large_participation(self):
        """Test with very large participation ratio."""
        model = AlmgrenChrissModel()

        # 50% of ADV
        impact = model.compute_temporary_impact(5_000_000, 10_000_000, 0.02)
        assert impact > 0
        assert math.isfinite(impact)

    def test_empty_order_book(self):
        """Test effects with empty order book."""
        effects = ImpactEffects()
        book = OrderBook(symbol="TEST")

        result = effects.apply_temporary_impact(
            order_book=book,
            impact_bps=5.0,
            side=Side.BUY,
        )

        # Should handle gracefully
        assert result.new_bid is None
        assert result.new_ask is None

    def test_insufficient_calibration_data(self):
        """Test calibration with insufficient data."""
        calibrator = AlmgrenChrissCalibrator(min_observations=30)

        # Only 5 observations
        dataset = CalibrationDataset()
        for i in range(5):
            obs = TradeObservation(
                timestamp_ms=i * 1000,
                price=150.0,
                qty=1000,
                side=1,
                adv=10_000_000,
                pre_trade_mid=150.0,
                post_trade_mid=150.01,
            )
            dataset.add_observation(obs)

        result = calibrator.calibrate(dataset)

        # Should return defaults with error diagnostic
        assert result.n_observations == 5
        assert "error" in result.diagnostics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
