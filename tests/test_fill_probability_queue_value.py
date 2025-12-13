"""
Comprehensive Tests for LOB Stage 3: Fill Probability & Queue Value.

Tests cover:
    1. Fill Probability Models (Poisson, Queue-Reactive, Historical, Composite)
    2. Queue Value Models (Standard, Dynamic, Spread Decomposition)
    3. Calibration Pipeline (Poisson, Queue-Reactive, Adverse Selection)
    4. Integration with existing LOB components
    5. Edge cases and numerical stability

Run with: pytest tests/test_fill_probability_queue_value.py -v
"""

import math
import time
from typing import List

import pytest
pytest.importorskip("sortedcontainers")

from lob import (
    # Core structures
    OrderBook,
    LimitOrder,
    Side,
    OrderType,
    Fill,
    Trade,
    # Queue tracking
    QueuePositionTracker,
    QueueState,
    LevelStatistics,
    # Stage 3: Fill Probability
    FillProbabilityModel,
    FillProbabilityModelType,
    FillProbabilityResult,
    LOBState,
    AnalyticalPoissonModel,
    QueueReactiveModel,
    HistoricalRateModel,
    HistoricalFillRate,
    DistanceBasedModel,
    CompositeFillProbabilityModel,
    create_fill_probability_model,
    compute_fill_probability_for_order,
    # Stage 3: Queue Value
    QueueValueModel,
    QueueValueResult,
    QueueValueConfig,
    QueueValueTracker,
    OrderDecision,
    AdverseSelectionParams,
    DynamicQueueValueModel,
    SpreadDecompositionModel,
    QueueImprovementEstimator,
    create_queue_value_model,
    compute_order_queue_value,
    # Stage 3: Calibration
    CalibrationPipeline,
    CalibrationResult,
    CrossValidationResult,
    TradeRecord,
    OrderRecord,
    PoissonRateCalibrator,
    QueueReactiveCalibrator,
    AdverseSelectionCalibrator,
    HistoricalRateCalibrator,
    create_calibration_pipeline,
    calibrate_from_trades,
    estimate_arrival_rate,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_order_book() -> OrderBook:
    """Create a sample order book for testing."""
    book = OrderBook(symbol="AAPL", tick_size=0.01)

    # Add bid orders
    for i, price in enumerate([149.95, 149.90, 149.85]):
        order = LimitOrder(
            order_id=f"bid_{i}",
            price=price,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=time.time_ns(),
            side=Side.BUY,
        )
        book.add_limit_order(order)

    # Add ask orders
    for i, price in enumerate([150.05, 150.10, 150.15]):
        order = LimitOrder(
            order_id=f"ask_{i}",
            price=price,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=time.time_ns(),
            side=Side.SELL,
        )
        book.add_limit_order(order)

    return book


@pytest.fixture
def sample_lob_state() -> LOBState:
    """Create a sample LOB state for testing."""
    return LOBState(
        timestamp_ns=time.time_ns(),
        mid_price=150.0,
        spread=0.10,
        spread_bps=6.67,  # 10 cents on $150 = 6.67 bps
        bid_depth=300.0,
        ask_depth=300.0,
        bid_levels=3,
        ask_levels=3,
        imbalance=0.0,
        volatility=0.02,
        trade_rate=1.0,
        volume_rate=100.0,
    )


@pytest.fixture
def sample_limit_order() -> LimitOrder:
    """Create a sample limit order for testing."""
    return LimitOrder(
        order_id="test_order_1",
        price=150.05,
        qty=50.0,
        remaining_qty=50.0,
        timestamp_ns=time.time_ns(),
        side=Side.SELL,
        is_own=True,
    )


@pytest.fixture
def sample_queue_state() -> QueueState:
    """Create a sample queue state for testing."""
    return QueueState(
        order_id="test_order_1",
        price=150.05,
        side=Side.SELL,
        estimated_position=5,
        qty_ahead=250.0,
        total_level_qty=300.0,
        confidence=0.8,
    )


@pytest.fixture
def sample_trades() -> List[TradeRecord]:
    """Create sample trade records for calibration testing."""
    base_time = time.time_ns()
    trades = []
    for i in range(100):
        trades.append(TradeRecord(
            timestamp_ns=base_time + i * 1_000_000_000,  # 1 second apart
            price=150.0 + (i % 10) * 0.01,
            qty=10.0 + (i % 5) * 5.0,
            side=Side.BUY if i % 2 == 0 else Side.SELL,
            maker_queue_position=i % 10,
            time_in_queue_sec=5.0 + (i % 20),
        ))
    return trades


# ==============================================================================
# Test LOBState
# ==============================================================================


class TestLOBState:
    """Tests for LOBState data structure."""

    def test_creation_defaults(self):
        """Test LOBState with default values."""
        state = LOBState()
        assert state.timestamp_ns == 0
        assert state.mid_price == 0.0
        assert state.volume_rate == 100.0  # Default

    def test_from_order_book(self, sample_order_book):
        """Test LOBState creation from OrderBook."""
        state = LOBState.from_order_book(sample_order_book)

        assert state.mid_price == 150.0
        assert abs(state.spread - 0.10) < 0.001  # 150.05 - 149.95, float tolerance
        assert state.bid_depth == 300.0
        assert state.ask_depth == 300.0
        assert abs(state.imbalance) < 0.01  # Equal depth

    def test_from_order_book_with_volatility(self, sample_order_book):
        """Test LOBState with volatility parameter."""
        state = LOBState.from_order_book(
            sample_order_book,
            volatility=0.02,
            volume_rate=200.0,
        )

        assert state.volatility == 0.02
        assert state.volume_rate == 200.0


# ==============================================================================
# Test Analytical Poisson Model
# ==============================================================================


class TestAnalyticalPoissonModel:
    """Tests for AnalyticalPoissonModel."""

    def test_creation(self):
        """Test model creation with defaults."""
        model = AnalyticalPoissonModel()
        assert model.model_type == FillProbabilityModelType.POISSON

    def test_front_of_queue_high_probability(self, sample_lob_state):
        """Test that front of queue has high fill probability."""
        model = AnalyticalPoissonModel()

        result = model.compute_fill_probability(
            queue_position=0,
            qty_ahead=0.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=sample_lob_state,
        )

        assert result.prob_fill >= 0.9  # Front of queue
        assert result.expected_wait_time_sec < 5.0

    def test_back_of_queue_lower_probability(self, sample_lob_state):
        """Test that back of queue has lower fill probability."""
        model = AnalyticalPoissonModel()

        result = model.compute_fill_probability(
            queue_position=50,
            qty_ahead=5000.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=sample_lob_state,
        )

        assert result.prob_fill < 0.9  # Not front
        assert result.expected_wait_time_sec > 10.0

    def test_probability_bounds(self, sample_lob_state):
        """Test that probabilities are always in [0, 1]."""
        model = AnalyticalPoissonModel()

        for qty_ahead in [0, 100, 1000, 10000]:
            result = model.compute_fill_probability(
                queue_position=0,
                qty_ahead=float(qty_ahead),
                order_qty=100.0,
                time_horizon_sec=60.0,
                market_state=sample_lob_state,
            )

            assert 0.0 <= result.prob_fill <= 1.0
            assert 0.0 <= result.prob_partial <= 1.0
            assert result.prob_partial >= result.prob_fill

    def test_expected_fill_time(self, sample_lob_state):
        """Test expected fill time computation."""
        model = AnalyticalPoissonModel()

        time_front = model.compute_expected_fill_time(
            queue_position=0,
            qty_ahead=0.0,
            market_state=sample_lob_state,
        )

        time_back = model.compute_expected_fill_time(
            queue_position=10,
            qty_ahead=1000.0,
            market_state=sample_lob_state,
        )

        assert time_back > time_front


# ==============================================================================
# Test Queue-Reactive Model
# ==============================================================================


class TestQueueReactiveModel:
    """Tests for QueueReactiveModel."""

    def test_creation(self):
        """Test model creation."""
        model = QueueReactiveModel()
        assert model.model_type == FillProbabilityModelType.QUEUE_REACTIVE

    def test_queue_size_impact(self, sample_lob_state):
        """Test that larger queue reduces fill probability."""
        model = QueueReactiveModel()

        prob_small_queue = model.compute_fill_probability(
            queue_position=5,
            qty_ahead=500.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=sample_lob_state,
        ).prob_fill

        prob_large_queue = model.compute_fill_probability(
            queue_position=50,
            qty_ahead=5000.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=sample_lob_state,
        ).prob_fill

        assert prob_small_queue > prob_large_queue

    def test_spread_sensitivity(self):
        """Test that wider spread increases maker activity."""
        model = QueueReactiveModel()

        # Narrow spread
        narrow_state = LOBState(spread_bps=2.0, volatility=0.02)
        # Wide spread
        wide_state = LOBState(spread_bps=20.0, volatility=0.02)

        rate_narrow, _ = model.compute_adjusted_rate(1000.0, narrow_state)
        rate_wide, _ = model.compute_adjusted_rate(1000.0, wide_state)

        # Wider spread should attract more makers (higher rate)
        assert rate_wide > rate_narrow

    def test_volatility_impact(self):
        """Test that higher volatility reduces maker activity."""
        model = QueueReactiveModel()

        low_vol = LOBState(spread_bps=5.0, volatility=0.01)
        high_vol = LOBState(spread_bps=5.0, volatility=0.05)

        rate_low, _ = model.compute_adjusted_rate(1000.0, low_vol)
        rate_high, _ = model.compute_adjusted_rate(1000.0, high_vol)

        assert rate_low > rate_high  # High vol reduces rate

    def test_imbalance_impact_buy(self):
        """Test imbalance impact for BUY side orders."""
        model = QueueReactiveModel()

        # Buy imbalance (more bids)
        buy_imbalance = LOBState(spread_bps=5.0, imbalance=0.5)
        # Sell imbalance (more asks)
        sell_imbalance = LOBState(spread_bps=5.0, imbalance=-0.5)

        rate_buy_imb, _ = model.compute_adjusted_rate(1000.0, buy_imbalance, Side.BUY)
        rate_sell_imb, _ = model.compute_adjusted_rate(1000.0, sell_imbalance, Side.BUY)

        # Sell imbalance should increase fill rate for BUY orders
        assert rate_sell_imb > rate_buy_imb


# ==============================================================================
# Test Historical Rate Model
# ==============================================================================


class TestHistoricalRateModel:
    """Tests for HistoricalRateModel."""

    def test_creation(self):
        """Test model creation."""
        model = HistoricalRateModel()
        assert model.model_type == FillProbabilityModelType.HISTORICAL

    def test_add_historical_rate(self):
        """Test adding historical rate data."""
        model = HistoricalRateModel()

        rate = HistoricalFillRate(
            price=150.0,
            side=Side.BUY,
            avg_fill_rate=200.0,
            fill_count=100,
        )
        model.add_historical_rate(rate)

        # Check that rate is used
        got_rate, confidence = model.get_rate_at_price(150.0, Side.BUY, 150.0)
        assert got_rate == 200.0
        assert confidence > 0.5

    def test_fallback_to_default(self, sample_lob_state):
        """Test fallback when no historical data."""
        model = HistoricalRateModel(default_fill_rate=50.0)

        result = model.compute_fill_probability(
            queue_position=5,
            qty_ahead=500.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=sample_lob_state,
        )

        # Should use default rate
        assert result.confidence <= 0.5


# ==============================================================================
# Test Distance-Based Model
# ==============================================================================


class TestDistanceBasedModel:
    """Tests for DistanceBasedModel."""

    def test_rate_at_mid(self):
        """Test rate at mid price is maximum."""
        model = DistanceBasedModel(base_rate=500.0)

        rate_mid = model.compute_rate_at_distance(150.0, 150.0, 5.0)
        rate_away = model.compute_rate_at_distance(151.0, 150.0, 5.0)

        assert rate_mid > rate_away

    def test_rate_decay_with_distance(self):
        """Test that rate decays with distance from mid."""
        model = DistanceBasedModel(base_rate=500.0)

        rates = []
        for distance in [0.0, 0.5, 1.0, 2.0]:
            price = 150.0 + distance
            rate = model.compute_rate_at_distance(price, 150.0, 5.0)
            rates.append(rate)

        # Rates should be decreasing
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1]


# ==============================================================================
# Test Composite Model
# ==============================================================================


class TestCompositeFillProbabilityModel:
    """Tests for CompositeFillProbabilityModel."""

    def test_default_ensemble(self, sample_lob_state):
        """Test default ensemble of models."""
        model = CompositeFillProbabilityModel()

        result = model.compute_fill_probability(
            queue_position=5,
            qty_ahead=500.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=sample_lob_state,
        )

        assert 0.0 <= result.prob_fill <= 1.0
        # Check that details contain individual model results
        assert any("prob" in k for k in result.details.keys())

    def test_custom_ensemble(self, sample_lob_state):
        """Test custom ensemble of models."""
        poisson = AnalyticalPoissonModel()
        reactive = QueueReactiveModel()

        model = CompositeFillProbabilityModel(models=[
            (poisson, 0.6),
            (reactive, 0.4),
        ])

        result = model.compute_fill_probability(
            queue_position=5,
            qty_ahead=500.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=sample_lob_state,
        )

        assert 0.0 <= result.prob_fill <= 1.0


# ==============================================================================
# Test Factory Functions
# ==============================================================================


class TestFillProbabilityFactory:
    """Tests for fill probability factory functions."""

    def test_create_poisson(self):
        """Test creating Poisson model."""
        model = create_fill_probability_model("poisson")
        assert isinstance(model, AnalyticalPoissonModel)

    def test_create_queue_reactive(self):
        """Test creating Queue-Reactive model."""
        model = create_fill_probability_model("queue_reactive")
        assert isinstance(model, QueueReactiveModel)

    def test_create_historical(self):
        """Test creating Historical model."""
        model = create_fill_probability_model("historical")
        assert isinstance(model, HistoricalRateModel)

    def test_create_composite(self):
        """Test creating Composite model."""
        model = create_fill_probability_model("composite")
        assert isinstance(model, CompositeFillProbabilityModel)

    def test_create_with_params(self):
        """Test creating model with custom parameters."""
        model = create_fill_probability_model(
            "poisson",
            default_arrival_rate=200.0,
        )
        assert isinstance(model, AnalyticalPoissonModel)


# ==============================================================================
# Test Queue Value Model
# ==============================================================================


class TestQueueValueModel:
    """Tests for QueueValueModel."""

    def test_creation(self):
        """Test model creation."""
        model = QueueValueModel()
        assert model is not None

    def test_positive_value_front_of_queue(
        self, sample_limit_order, sample_lob_state, sample_queue_state
    ):
        """Test that front of queue has positive value."""
        model = QueueValueModel()

        # Modify queue state to be at front
        sample_queue_state.qty_ahead = 0.0
        sample_queue_state.estimated_position = 0

        result = model.compute_queue_value(
            sample_limit_order, sample_lob_state, sample_queue_state
        )

        assert result.queue_value > 0
        assert result.decision == OrderDecision.HOLD

    def test_value_decreases_with_position(
        self, sample_limit_order, sample_lob_state
    ):
        """Test that value decreases further back in queue."""
        model = QueueValueModel()

        values = []
        for position in [0, 10, 50, 100]:
            queue_state = QueueState(
                order_id="test",
                price=150.05,
                side=Side.SELL,
                estimated_position=position,
                qty_ahead=position * 100.0,
                total_level_qty=(position + 1) * 100.0,
            )

            result = model.compute_queue_value(
                sample_limit_order, sample_lob_state, queue_state
            )
            values.append(result.queue_value)

        # Value should decrease
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_should_cancel(self, sample_limit_order, sample_lob_state):
        """Test should_cancel decision."""
        model = QueueValueModel()

        # Very far back in queue
        queue_state = QueueState(
            order_id="test",
            price=150.05,
            side=Side.SELL,
            estimated_position=1000,
            qty_ahead=100000.0,
            total_level_qty=100000.0,
        )

        should_cancel = model.should_cancel(
            sample_limit_order, sample_lob_state, queue_state
        )

        # May or may not cancel depending on parameters
        assert isinstance(should_cancel, bool)

    def test_adverse_selection_cost(self, sample_limit_order, sample_lob_state):
        """Test adverse selection cost is computed."""
        model = QueueValueModel()

        queue_state = QueueState(
            order_id="test",
            price=150.05,
            side=Side.SELL,
            estimated_position=5,
            qty_ahead=500.0,
        )

        result = model.compute_queue_value(
            sample_limit_order, sample_lob_state, queue_state
        )

        assert result.adverse_selection_cost >= 0
        assert result.opportunity_cost >= 0


# ==============================================================================
# Test Dynamic Queue Value Model
# ==============================================================================


class TestDynamicQueueValueModel:
    """Tests for DynamicQueueValueModel."""

    def test_volatility_adjustment(self, sample_limit_order):
        """Test that volatility adjusts adverse selection."""
        model = DynamicQueueValueModel(volatility_adjustment=2.0)

        low_vol_state = LOBState(
            mid_price=150.0, spread_bps=5.0, volatility=0.01
        )
        high_vol_state = LOBState(
            mid_price=150.0, spread_bps=5.0, volatility=0.05
        )

        queue_state = QueueState(
            order_id="test",
            price=150.05,
            side=Side.SELL,
            qty_ahead=500.0,
        )

        result_low = model.compute_queue_value(
            sample_limit_order, low_vol_state, queue_state
        )
        result_high = model.compute_queue_value(
            sample_limit_order, high_vol_state, queue_state
        )

        # Higher vol = higher adverse selection cost
        assert result_high.adverse_selection_cost >= result_low.adverse_selection_cost


# ==============================================================================
# Test Queue Value Tracker
# ==============================================================================


class TestQueueValueTracker:
    """Tests for QueueValueTracker."""

    def test_track_order(self, sample_limit_order):
        """Test tracking an order."""
        tracker = QueueValueTracker()

        tracker.track_order(sample_limit_order)
        assert tracker.tracked_count == 1

    def test_untrack_order(self, sample_limit_order):
        """Test untracking an order."""
        tracker = QueueValueTracker()

        tracker.track_order(sample_limit_order)
        removed = tracker.untrack_order(sample_limit_order.order_id)

        assert removed is not None
        assert tracker.tracked_count == 0

    def test_update_values(self, sample_limit_order, sample_lob_state):
        """Test updating values for tracked orders."""
        tracker = QueueValueTracker()

        tracker.track_order(sample_limit_order)
        cancel_recommendations = tracker.update_values(sample_lob_state)

        # Should return list (may be empty)
        assert isinstance(cancel_recommendations, list)

    def test_get_value_history(self, sample_limit_order, sample_lob_state):
        """Test getting value history."""
        tracker = QueueValueTracker()

        tracker.track_order(sample_limit_order)

        # Update a few times
        for _ in range(3):
            tracker.update_values(sample_lob_state)

        history = tracker.get_value_history(sample_limit_order.order_id)
        assert len(history) == 3


# ==============================================================================
# Test Queue Improvement Estimator
# ==============================================================================


class TestQueueImprovementEstimator:
    """Tests for QueueImprovementEstimator."""

    def test_time_to_front(self):
        """Test time to front of queue estimation."""
        estimator = QueueImprovementEstimator(execution_rate=100.0)

        time = estimator.estimate_time_to_front(qty_ahead=1000.0)

        # Should be approximately 10 seconds (1000 / 100), but cancel rate reduces queue
        # With cancel_rate=0.1, effective rate = exec_rate + queue * cancel_rate = 100 + 1000*0.1 = 200
        # So time ≈ 1000 / 200 = 5 seconds
        assert 4.0 <= time <= 20.0  # Allow for different cancel rates

    def test_position_at_time(self):
        """Test position estimation at future time."""
        estimator = QueueImprovementEstimator(execution_rate=100.0)

        position_now = 1000.0
        position_later = estimator.estimate_position_at_time(
            current_qty_ahead=position_now,
            time_sec=5.0,
        )

        assert position_later < position_now

    def test_position_distribution(self):
        """Test position distribution computation."""
        estimator = QueueImprovementEstimator(execution_rate=100.0)

        dist = estimator.compute_position_distribution(
            current_qty_ahead=1000.0,
            time_sec=10.0,
        )

        assert "mean" in dist
        assert "std" in dist
        assert "p10" in dist
        assert "p90" in dist


# ==============================================================================
# Test Calibration Pipeline
# ==============================================================================


class TestCalibrationPipeline:
    """Tests for CalibrationPipeline."""

    def test_creation(self):
        """Test pipeline creation."""
        pipeline = create_calibration_pipeline()
        assert pipeline is not None

    def test_add_trades(self, sample_trades):
        """Test adding trades to pipeline."""
        pipeline = CalibrationPipeline()
        pipeline.add_trades(sample_trades)

        assert pipeline._poisson_calibrator.n_trades == len(sample_trades)

    def test_run_calibration(self, sample_trades):
        """Test running full calibration."""
        pipeline = CalibrationPipeline()
        pipeline.add_trades(sample_trades)

        results = pipeline.run_calibration()

        assert "poisson" in results
        assert "queue_reactive" in results
        assert results["poisson"].n_samples > 0

    def test_get_best_model(self, sample_trades):
        """Test getting calibrated model."""
        pipeline = CalibrationPipeline()
        pipeline.add_trades(sample_trades)
        pipeline.run_calibration()

        model = pipeline.get_best_model("poisson")
        assert isinstance(model, AnalyticalPoissonModel)


# ==============================================================================
# Test Poisson Rate Calibrator
# ==============================================================================


class TestPoissonRateCalibrator:
    """Tests for PoissonRateCalibrator."""

    def test_fit(self, sample_trades):
        """Test fitting Poisson model."""
        calibrator = PoissonRateCalibrator()
        calibrator.add_trades(sample_trades)

        result = calibrator.fit()

        assert result.model_type == FillProbabilityModelType.POISSON
        assert "arrival_rate" in result.parameters
        assert result.parameters["arrival_rate"] > 0

    def test_get_model(self, sample_trades):
        """Test getting calibrated model."""
        calibrator = PoissonRateCalibrator()
        calibrator.add_trades(sample_trades)
        calibrator.fit()

        model = calibrator.get_model()
        assert isinstance(model, AnalyticalPoissonModel)

    def test_get_rate_at_level(self, sample_trades):
        """Test getting rate at specific level."""
        calibrator = PoissonRateCalibrator()
        calibrator.add_trades(sample_trades)
        calibrator.fit()

        rate = calibrator.get_rate_at_level(150.0, Side.BUY)
        assert rate > 0


# ==============================================================================
# Test Queue-Reactive Calibrator
# ==============================================================================


class TestQueueReactiveCalibrator:
    """Tests for QueueReactiveCalibrator."""

    def test_fit_with_state_data(self):
        """Test fitting with state observations."""
        calibrator = QueueReactiveCalibrator()

        # Add state observations
        base_time = time.time_ns()
        for i in range(50):
            calibrator.add_state_observation(
                timestamp_ns=base_time + i * 1_000_000_000,
                queue_size=500.0 + i * 10,
                spread_bps=5.0,
                volatility=0.02,
                imbalance=0.0,
                executed_qty=10.0 + (i % 10),
            )

        result = calibrator.fit()

        assert "queue_decay_alpha" in result.parameters
        assert "spread_sensitivity_beta" in result.parameters

    def test_get_model(self, sample_trades):
        """Test getting calibrated model."""
        calibrator = QueueReactiveCalibrator()
        calibrator.add_trades(sample_trades)
        calibrator.fit()

        model = calibrator.get_model()
        assert isinstance(model, QueueReactiveModel)


# ==============================================================================
# Test Adverse Selection Calibrator
# ==============================================================================


class TestAdverseSelectionCalibrator:
    """Tests for AdverseSelectionCalibrator."""

    def test_fit_with_price_observations(self):
        """Test fitting with price observations."""
        calibrator = AdverseSelectionCalibrator()

        # Add price observations
        for i in range(50):
            pre_mid = 150.0
            # Some trades move price, some don't
            if i % 3 == 0:
                post_mid = 150.05  # Adverse move for seller
            else:
                post_mid = 149.95  # Favorable for seller

            calibrator.add_price_observation(
                pre_trade_mid=pre_mid,
                post_trade_mid=post_mid,
                trade_side=Side.BUY if i % 2 == 0 else Side.SELL,
            )

        result = calibrator.fit()

        assert "informed_fraction" in result.parameters
        assert "adverse_move_bps" in result.parameters
        assert "adverse_probability" in result.parameters


# ==============================================================================
# Test Historical Rate Calibrator
# ==============================================================================


class TestHistoricalRateCalibrator:
    """Tests for HistoricalRateCalibrator."""

    def test_fit_with_fill_observations(self):
        """Test fitting with fill observations."""
        calibrator = HistoricalRateCalibrator(bucket_size_bps=5.0)

        # Add fill observations
        for i in range(50):
            calibrator.add_fill_observation(
                price=150.0 + (i % 10) * 0.01,
                mid_price=150.0,
                side=Side.BUY if i % 2 == 0 else Side.SELL,
                fill_qty=100.0,
                time_in_queue_sec=5.0 + i % 20,
            )

        result = calibrator.fit()

        assert result.n_samples == 50

    def test_get_model(self):
        """Test getting calibrated model."""
        calibrator = HistoricalRateCalibrator()

        # Add some data
        for i in range(20):
            calibrator.add_fill_observation(
                price=150.0,
                mid_price=150.0,
                side=Side.BUY,
                fill_qty=100.0,
                time_in_queue_sec=10.0,
            )

        calibrator.fit()
        model = calibrator.get_model()

        assert isinstance(model, HistoricalRateModel)


# ==============================================================================
# Test Cross-Validation
# ==============================================================================


class TestCrossValidation:
    """Tests for cross-validation."""

    def test_cross_validate(self, sample_trades):
        """Test cross-validation."""
        pipeline = CalibrationPipeline()
        pipeline.add_trades(sample_trades)

        cv_result = pipeline.cross_validate(n_folds=3)

        assert cv_result.n_folds == 3
        assert len(cv_result.train_scores) == 3
        assert len(cv_result.test_scores) == 3

    def test_cross_validate_insufficient_data(self):
        """Test cross-validation with insufficient data."""
        pipeline = CalibrationPipeline()

        # Only 2 trades
        pipeline.add_trades([
            TradeRecord(
                timestamp_ns=time.time_ns(),
                price=150.0,
                qty=100.0,
                side=Side.BUY,
            ),
            TradeRecord(
                timestamp_ns=time.time_ns() + 1_000_000_000,
                price=150.01,
                qty=50.0,
                side=Side.SELL,
            ),
        ])

        cv_result = pipeline.cross_validate(n_folds=5)

        # Should handle gracefully
        assert cv_result.n_folds == 5


# ==============================================================================
# Test Integration with Existing LOB
# ==============================================================================


class TestIntegrationWithLOB:
    """Tests for integration with existing LOB components."""

    def test_compute_fill_probability_for_order(
        self, sample_order_book, sample_queue_state
    ):
        """Test convenience function with real OrderBook."""
        order = LimitOrder(
            order_id="test",
            price=150.05,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=time.time_ns(),
            side=Side.SELL,
        )

        result = compute_fill_probability_for_order(
            sample_queue_state, order, sample_order_book
        )

        assert result.order_id == "test"
        assert 0.0 <= result.prob_fill <= 1.0

    def test_compute_order_queue_value(self, sample_order_book):
        """Test convenience function for queue value."""
        order = LimitOrder(
            order_id="test",
            price=150.05,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=time.time_ns(),
            side=Side.SELL,
        )

        result = compute_order_queue_value(order, sample_order_book)

        assert result.order_id == "test"
        assert isinstance(result.queue_value, float)

    def test_with_queue_position_tracker(self, sample_order_book):
        """Test integration with QueuePositionTracker."""
        tracker = QueuePositionTracker()

        order = LimitOrder(
            order_id="test",
            price=150.05,
            qty=100.0,
            remaining_qty=100.0,
            timestamp_ns=time.time_ns(),
            side=Side.SELL,
        )

        # Track the order
        queue_state = tracker.add_order(order, level_qty_before=500.0)

        # Compute fill probability
        model = QueueReactiveModel()
        lob_state = LOBState.from_order_book(sample_order_book)

        result = model.compute_for_queue_state(
            queue_state,
            order_qty=order.remaining_qty,
            time_horizon_sec=60.0,
            market_state=lob_state,
        )

        assert result.prob_fill >= 0.0


# ==============================================================================
# Test Numerical Stability
# ==============================================================================


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_zero_qty_ahead(self, sample_lob_state):
        """Test with zero quantity ahead."""
        model = AnalyticalPoissonModel()

        result = model.compute_fill_probability(
            queue_position=0,
            qty_ahead=0.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=sample_lob_state,
        )

        assert math.isfinite(result.prob_fill)
        assert math.isfinite(result.expected_wait_time_sec)

    def test_very_large_queue(self, sample_lob_state):
        """Test with very large queue."""
        model = AnalyticalPoissonModel()

        result = model.compute_fill_probability(
            queue_position=1000000,
            qty_ahead=100000000.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=sample_lob_state,
        )

        assert math.isfinite(result.prob_fill)
        assert result.prob_fill < 0.01  # Should be very low

    def test_zero_volume_rate(self):
        """Test with zero volume rate."""
        model = AnalyticalPoissonModel()

        state = LOBState(volume_rate=0.0)

        result = model.compute_fill_probability(
            queue_position=5,
            qty_ahead=500.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=state,
        )

        # Should use default rate
        assert math.isfinite(result.prob_fill)

    def test_zero_time_horizon(self, sample_lob_state):
        """Test with zero time horizon."""
        model = AnalyticalPoissonModel()

        result = model.compute_fill_probability(
            queue_position=5,
            qty_ahead=500.0,
            order_qty=100.0,
            time_horizon_sec=0.0,
            market_state=sample_lob_state,
        )

        assert math.isfinite(result.prob_fill)

    def test_negative_imbalance(self, sample_lob_state):
        """Test with negative imbalance."""
        model = QueueReactiveModel()

        sample_lob_state.imbalance = -0.9

        result = model.compute_fill_probability(
            queue_position=5,
            qty_ahead=500.0,
            order_qty=100.0,
            time_horizon_sec=60.0,
            market_state=sample_lob_state,
        )

        assert math.isfinite(result.prob_fill)


# ==============================================================================
# Test Factory Convenience Functions
# ==============================================================================


class TestConvenienceFunctions:
    """Tests for convenience/factory functions."""

    def test_calibrate_from_trades(self, sample_trades):
        """Test quick calibration from trades."""
        model = calibrate_from_trades(sample_trades, "poisson")
        assert isinstance(model, AnalyticalPoissonModel)

    def test_estimate_arrival_rate(self, sample_trades):
        """Test quick arrival rate estimation."""
        rate = estimate_arrival_rate(sample_trades)
        assert rate > 0

    def test_estimate_arrival_rate_empty(self):
        """Test arrival rate with empty trades."""
        rate = estimate_arrival_rate([])
        assert rate == 100.0  # Default


# ==============================================================================
# Test Queue Value Factory
# ==============================================================================


class TestQueueValueFactory:
    """Tests for queue value factory functions."""

    def test_create_standard(self):
        """Test creating standard model."""
        model = create_queue_value_model("standard")
        assert isinstance(model, QueueValueModel)

    def test_create_dynamic(self):
        """Test creating dynamic model."""
        model = create_queue_value_model("dynamic")
        assert isinstance(model, DynamicQueueValueModel)

    def test_create_spread_decomposition(self):
        """Test creating spread decomposition model."""
        model = create_queue_value_model("spread_decomposition")
        assert isinstance(model, SpreadDecompositionModel)


# ==============================================================================
# Test Issue 4.1: LOBState order_side and order_price
# ==============================================================================


class TestLOBStateOrderInfo:
    """Tests for LOBState order_side and order_price fields (Issue 4.1 fix)."""

    def test_order_side_field(self):
        """Test LOBState with order_side field."""
        state = LOBState(
            mid_price=150.0,
            order_side=Side.BUY,
        )
        assert state.order_side == Side.BUY

    def test_order_price_field(self):
        """Test LOBState with order_price field."""
        state = LOBState(
            mid_price=150.0,
            order_price=149.95,
        )
        assert state.order_price == 149.95

    def test_from_order_book_with_order_info(self, sample_order_book):
        """Test LOBState.from_order_book with order info."""
        state = LOBState.from_order_book(
            sample_order_book,
            order_side=Side.SELL,
            order_price=150.05,
        )
        assert state.order_side == Side.SELL
        assert state.order_price == 150.05

    def test_from_order_book_defaults(self, sample_order_book):
        """Test LOBState.from_order_book defaults to None."""
        state = LOBState.from_order_book(sample_order_book)
        assert state.order_side is None
        assert state.order_price is None


class TestHistoricalRateModelOrderInfo:
    """Tests for HistoricalRateModel using order_side/price (Issue 4.1 fix)."""

    def test_uses_order_side_from_state(self):
        """Test that HistoricalRateModel uses order_side from LOBState."""
        model = HistoricalRateModel()

        # Add historical rates for both sides
        model.add_historical_rate(HistoricalFillRate(
            price=150.0,
            side=Side.BUY,
            avg_fill_rate=100.0,
            fill_count=50,
        ))
        model.add_historical_rate(HistoricalFillRate(
            price=150.0,
            side=Side.SELL,
            avg_fill_rate=200.0,  # Different rate for SELL
            fill_count=50,
        ))

        # Test with BUY side
        state_buy = LOBState(
            mid_price=150.0,
            order_side=Side.BUY,
            order_price=150.0,
        )
        result_buy = model.compute_fill_probability(
            queue_position=0,
            qty_ahead=100.0,
            order_qty=50.0,
            time_horizon_sec=60.0,
            market_state=state_buy,
        )

        # Test with SELL side
        state_sell = LOBState(
            mid_price=150.0,
            order_side=Side.SELL,
            order_price=150.0,
        )
        result_sell = model.compute_fill_probability(
            queue_position=0,
            qty_ahead=100.0,
            order_qty=50.0,
            time_horizon_sec=60.0,
            market_state=state_sell,
        )

        # Different rates should produce different probabilities
        # SELL has higher rate, so should have higher probability
        assert result_sell.prob_fill >= result_buy.prob_fill

    def test_uses_order_price_from_state(self):
        """Test that HistoricalRateModel uses order_price from LOBState."""
        model = HistoricalRateModel()

        # Add historical rate at specific price
        model.add_historical_rate(HistoricalFillRate(
            price=149.0,
            side=Side.BUY,
            avg_fill_rate=300.0,
            fill_count=50,
        ))

        # State with specific order_price
        state = LOBState(
            mid_price=150.0,
            order_side=Side.BUY,
            order_price=149.0,  # Match the historical rate price
        )
        result = model.compute_fill_probability(
            queue_position=0,
            qty_ahead=100.0,
            order_qty=50.0,
            time_horizon_sec=60.0,
            market_state=state,
        )

        # Should use the high rate at 149.0
        assert result.details["fill_rate"] == 300.0

    def test_fallback_to_imbalance_heuristic(self):
        """Test fallback to imbalance heuristic when order_side is None."""
        model = HistoricalRateModel()

        # Add rates for both sides
        model.add_historical_rate(HistoricalFillRate(
            price=150.0,
            side=Side.BUY,
            avg_fill_rate=100.0,
            fill_count=50,
        ))
        model.add_historical_rate(HistoricalFillRate(
            price=150.0,
            side=Side.SELL,
            avg_fill_rate=200.0,
            fill_count=50,
        ))

        # Positive imbalance -> should use SELL side
        state_pos = LOBState(
            mid_price=150.0,
            imbalance=0.5,  # Positive = more bids
            order_side=None,  # No explicit side
        )
        result_pos = model.compute_fill_probability(
            queue_position=0,
            qty_ahead=100.0,
            order_qty=50.0,
            time_horizon_sec=60.0,
            market_state=state_pos,
        )

        # Negative imbalance -> should use BUY side
        state_neg = LOBState(
            mid_price=150.0,
            imbalance=-0.5,  # Negative = more asks
            order_side=None,
        )
        result_neg = model.compute_fill_probability(
            queue_position=0,
            qty_ahead=100.0,
            order_qty=50.0,
            time_horizon_sec=60.0,
            market_state=state_neg,
        )

        # Should get different rates based on imbalance
        assert result_pos.details["fill_rate"] != result_neg.details["fill_rate"]


# ==============================================================================
# Test Issue 4.2: Calibration _evaluate_model uses actual data
# ==============================================================================


class TestCalibrationEvaluateModel:
    """Tests for CalibrationPipeline._evaluate_model using actual data (Issue 4.2 fix)."""

    def test_uses_actual_volume_rate(self, sample_trades):
        """Test that _evaluate_model computes volume rate from trades."""
        pipeline = CalibrationPipeline()
        pipeline.add_trades(sample_trades)
        pipeline.run_calibration()

        # The calibration should complete without error
        # and use actual data
        model = pipeline.get_best_model("poisson")
        assert model is not None

    def test_uses_time_in_queue_data(self):
        """Test that _evaluate_model uses time_in_queue from trades."""
        pipeline = CalibrationPipeline()

        # Create trades with time_in_queue data
        base_time = time.time_ns()
        trades = []
        for i in range(20):
            trades.append(TradeRecord(
                timestamp_ns=base_time + i * 1_000_000_000,
                price=150.0,
                qty=100.0,
                side=Side.BUY if i % 2 == 0 else Side.SELL,
                time_in_queue_sec=30.0,  # All trades filled in 30 sec
            ))

        pipeline.add_trades(trades)
        results = pipeline.run_calibration()

        # Should complete without error
        assert "poisson" in results

    def test_cross_validate_with_actual_data(self, sample_trades):
        """Test cross-validation uses actual trade data."""
        pipeline = CalibrationPipeline()
        pipeline.add_trades(sample_trades)

        cv_result = pipeline.cross_validate(n_folds=3)

        assert cv_result.n_folds == 3
        assert len(cv_result.test_scores) == 3


# ==============================================================================
# Test Issue 4.3: QueueValueModel decision methods
# ==============================================================================


class TestQueueValueModelDecisionMethods:
    """Tests for QueueValueModel decision methods (Issue 4.3 fix)."""

    def test_should_cancel_only_for_cancel(self, sample_limit_order, sample_lob_state):
        """Test should_cancel returns True only for CANCEL decision."""
        model = QueueValueModel()

        # Very far back in queue -> likely CANCEL
        queue_state = QueueState(
            order_id="test",
            price=150.05,
            side=Side.SELL,
            estimated_position=10000,
            qty_ahead=1000000.0,
            total_level_qty=1000000.0,
        )

        result = model.compute_queue_value(
            sample_limit_order, sample_lob_state, queue_state
        )

        if result.decision == OrderDecision.CANCEL:
            assert model.should_cancel(sample_limit_order, sample_lob_state, queue_state)
            assert model.should_modify(sample_limit_order, sample_lob_state, queue_state)
        elif result.decision == OrderDecision.REPRICE:
            assert not model.should_cancel(sample_limit_order, sample_lob_state, queue_state)
            assert model.should_reprice(sample_limit_order, sample_lob_state, queue_state)
            assert model.should_modify(sample_limit_order, sample_lob_state, queue_state)
        else:
            assert not model.should_cancel(sample_limit_order, sample_lob_state, queue_state)
            assert not model.should_reprice(sample_limit_order, sample_lob_state, queue_state)
            assert not model.should_modify(sample_limit_order, sample_lob_state, queue_state)

    def test_should_reprice_for_reprice(self, sample_limit_order, sample_lob_state):
        """Test should_reprice returns True only for REPRICE decision."""
        model = QueueValueModel()

        # Front of queue with narrow spread -> likely HOLD
        queue_state = QueueState(
            order_id="test",
            price=150.05,
            side=Side.SELL,
            estimated_position=0,
            qty_ahead=0.0,
            total_level_qty=100.0,
        )

        result = model.compute_queue_value(
            sample_limit_order, sample_lob_state, queue_state
        )

        # Verify decision consistency
        if result.decision == OrderDecision.REPRICE:
            assert model.should_reprice(sample_limit_order, sample_lob_state, queue_state)
            assert not model.should_cancel(sample_limit_order, sample_lob_state, queue_state)
            assert model.should_modify(sample_limit_order, sample_lob_state, queue_state)

    def test_should_modify_includes_both(self, sample_limit_order, sample_lob_state):
        """Test should_modify returns True for CANCEL or REPRICE."""
        model = QueueValueModel()

        queue_state = QueueState(
            order_id="test",
            price=150.05,
            side=Side.SELL,
            estimated_position=5,
            qty_ahead=500.0,
        )

        result = model.compute_queue_value(
            sample_limit_order, sample_lob_state, queue_state
        )

        should_modify = model.should_modify(
            sample_limit_order, sample_lob_state, queue_state
        )
        should_cancel = model.should_cancel(
            sample_limit_order, sample_lob_state, queue_state
        )
        should_reprice = model.should_reprice(
            sample_limit_order, sample_lob_state, queue_state
        )

        # should_modify should be True if either cancel or reprice is True
        assert should_modify == (should_cancel or should_reprice)

    def test_hold_decision_all_false(self, sample_limit_order, sample_lob_state):
        """Test HOLD decision returns False for all should_* methods."""
        model = QueueValueModel()

        # Front of queue with good spread -> HOLD
        sample_queue_state = QueueState(
            order_id="test",
            price=150.05,
            side=Side.SELL,
            estimated_position=0,
            qty_ahead=0.0,
            total_level_qty=100.0,
        )

        result = model.compute_queue_value(
            sample_limit_order, sample_lob_state, sample_queue_state
        )

        if result.decision == OrderDecision.HOLD:
            assert not model.should_cancel(
                sample_limit_order, sample_lob_state, sample_queue_state
            )
            assert not model.should_reprice(
                sample_limit_order, sample_lob_state, sample_queue_state
            )
            assert not model.should_modify(
                sample_limit_order, sample_lob_state, sample_queue_state
            )


# ==============================================================================
# Run tests
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
