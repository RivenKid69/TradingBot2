# -*- coding: utf-8 -*-
"""
Backtest Runner - Runs historical backtests.

CLOUD ZONE ONLY.

Uses SimExecutionEngine from shared package.
Does NOT execute live trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from packages.shared.simulation.engine import (
    SimExecutionEngine,
    SimulationConfig,
    SimulatedFill,
)
from packages.shared.contracts.strategy import StrategyContract, StrategyContext, MarketSnapshot
from packages.shared.contracts.intent import OrderIntent


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""

    strategy_id: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime

    # Capital
    initial_capital: Decimal = Decimal("100000")

    # Simulation settings
    slippage_bps: Decimal = Decimal("5")
    commission_pct: Decimal = Decimal("0.001")

    # Data
    data_frequency: str = "1h"  # 1m, 5m, 1h, 1d, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "symbols": self.symbols,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": str(self.initial_capital),
            "slippage_bps": str(self.slippage_bps),
            "commission_pct": str(self.commission_pct),
            "data_frequency": self.data_frequency,
        }


@dataclass
class BacktestMetrics:
    """Backtest performance metrics."""

    total_return: Decimal = Decimal("0")
    total_return_pct: Decimal = Decimal("0")
    sharpe_ratio: Optional[Decimal] = None
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    profit_factor: Optional[Decimal] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": str(self.total_return),
            "total_return_pct": str(self.total_return_pct),
            "sharpe_ratio": str(self.sharpe_ratio) if self.sharpe_ratio else None,
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "win_rate": str(self.win_rate),
            "profit_factor": str(self.profit_factor) if self.profit_factor else None,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
        }


@dataclass
class BacktestResult:
    """Result of backtest run."""

    success: bool = False
    config: Optional[BacktestConfig] = None
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    run_time_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "config": self.config.to_dict() if self.config else None,
            "metrics": self.metrics.to_dict(),
            "equity_curve_length": len(self.equity_curve),
            "trade_count": len(self.trades),
            "errors": self.errors,
            "run_time_ms": self.run_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class BacktestRunner:
    """
    Runs historical backtests.

    Uses SimExecutionEngine to process strategy intents
    against historical data.

    Usage:
        runner = BacktestRunner()

        config = BacktestConfig(
            strategy_id="momentum",
            symbols=["BTCUSDT"],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 1),
        )

        result = runner.run(config, strategy, historical_data)
    """

    def __init__(self):
        """Initialize runner."""
        self._engine: Optional[SimExecutionEngine] = None

    def run(
        self,
        config: BacktestConfig,
        strategy: StrategyContract,
        historical_data: List[Dict[str, Any]],
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            config: Backtest configuration
            strategy: Strategy to test
            historical_data: Historical price data

        Returns:
            BacktestResult with performance metrics
        """
        import time

        start_time = time.time()
        result = BacktestResult(config=config)

        try:
            # Initialize simulation engine
            sim_config = SimulationConfig(
                slippage_bps=config.slippage_bps,
                initial_capital=config.initial_capital,
            )
            self._engine = SimExecutionEngine(config=sim_config)

            # Initialize strategy
            strategy.initialize({
                "symbols": config.symbols,
            })

            # Track equity
            equity_curve = []
            peak_equity = config.initial_capital
            max_drawdown = Decimal("0")

            # Process each bar
            for bar in historical_data:
                timestamp = bar.get("timestamp")
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)

                # Skip if outside date range
                if timestamp < config.start_date or timestamp > config.end_date:
                    continue

                # Create market snapshot
                symbol = bar.get("symbol", config.symbols[0])
                price = Decimal(str(bar.get("close", 0)))

                snapshot = MarketSnapshot(
                    timestamp=timestamp,
                    symbol=symbol,
                    last=price,
                    bid=price * Decimal("0.9999"),
                    ask=price * Decimal("1.0001"),
                )

                # Set price in engine
                self._engine.set_price(symbol, price)

                # Get position
                position = self._engine.get_position(symbol)
                snapshot.position_qty = position.quantity
                snapshot.position_avg_price = position.avg_price

                # Create context
                context = StrategyContext(
                    market=snapshot,
                    is_live=False,
                )

                # Get strategy decision
                strategy_result = strategy.on_data(context)

                # Process intents
                for intent in strategy_result.intents:
                    fill = self._engine.process_intent(intent)
                    if fill:
                        result.trades.append(fill.to_dict())

                # Track equity
                equity = self._engine.get_equity()
                equity_curve.append({
                    "timestamp": timestamp.isoformat(),
                    "equity": str(equity),
                })

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                drawdown = peak_equity - equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            # Calculate metrics
            result.equity_curve = equity_curve
            result.metrics = self._calculate_metrics(
                config.initial_capital,
                self._engine.get_equity(),
                max_drawdown,
                result.trades,
            )

            # Shutdown strategy
            strategy.shutdown()

            result.success = True

        except Exception as e:
            result.errors.append(str(e))

        result.run_time_ms = int((time.time() - start_time) * 1000)
        return result

    def _calculate_metrics(
        self,
        initial_capital: Decimal,
        final_equity: Decimal,
        max_drawdown: Decimal,
        trades: List[Dict[str, Any]],
    ) -> BacktestMetrics:
        """Calculate backtest metrics."""
        total_return = final_equity - initial_capital
        total_return_pct = (
            total_return / initial_capital * 100
            if initial_capital > 0
            else Decimal("0")
        )

        max_drawdown_pct = (
            max_drawdown / initial_capital * 100
            if initial_capital > 0
            else Decimal("0")
        )

        # Trade statistics
        total_trades = len(trades)
        winning_trades = sum(
            1 for t in trades
            if Decimal(t.get("commission", "0")) < Decimal(t.get("notional", "0")) * Decimal("0.01")
        )
        losing_trades = total_trades - winning_trades

        win_rate = (
            Decimal(winning_trades) / Decimal(total_trades) * 100
            if total_trades > 0
            else Decimal("0")
        )

        return BacktestMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
        )
