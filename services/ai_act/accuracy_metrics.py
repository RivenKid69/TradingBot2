# -*- coding: utf-8 -*-
"""
EU AI Act Article 15 - Accuracy Metrics Declaration.

This module implements accuracy metrics tracking and declaration
for high-risk AI systems as required by Article 15 of the EU AI Act.

Article 15 Requirements:
- High-risk AI systems shall be designed and developed in such a way
  that they achieve an appropriate level of accuracy, robustness and
  cybersecurity, and perform consistently in those respects throughout
  their lifecycle.
- The levels of accuracy and the relevant accuracy metrics shall be
  declared in the accompanying instructions of use.

Trading-Specific Metrics (from EU_AI_ACT_INTEGRATION_PLAN.md):
| Metric              | Expected | Minimum | Measurement Method    |
|---------------------|----------|---------|----------------------|
| Sharpe Ratio        | > 1.5    | > 0.5   | Rolling 252-day      |
| Max Drawdown        | < 15%    | < 25%   | Historical max       |
| Win Rate            | > 55%    | > 45%   | Trade-level          |
| Prediction Accuracy | > 52%    | > 50%   | Direction accuracy   |

References:
- EU AI Act Article 15: https://artificialintelligenceact.eu/article/15/
- ISO 22989:2022 AI concepts and terminology
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of accuracy metrics tracked.

    Categories based on EU AI Act and trading-specific requirements.
    """

    # Performance Metrics
    SHARPE_RATIO = auto()
    SORTINO_RATIO = auto()
    CALMAR_RATIO = auto()
    MAX_DRAWDOWN = auto()

    # Trade-Level Metrics
    WIN_RATE = auto()
    PROFIT_FACTOR = auto()
    AVERAGE_WIN_LOSS_RATIO = auto()

    # Prediction Metrics
    DIRECTION_ACCURACY = auto()
    MAGNITUDE_ACCURACY = auto()
    SIGNAL_QUALITY = auto()

    # Risk-Adjusted Metrics
    VAR_95 = auto()
    CVAR_95 = auto()
    INFORMATION_RATIO = auto()

    # Stability Metrics
    CONSISTENCY_SCORE = auto()
    REGIME_ADAPTABILITY = auto()


class MetricStatus(Enum):
    """Status of a metric relative to declared thresholds."""

    EXCEEDS_EXPECTED = "exceeds_expected"
    MEETS_EXPECTED = "meets_expected"
    MEETS_MINIMUM = "meets_minimum"
    BELOW_MINIMUM = "below_minimum"
    CRITICAL = "critical"
    NOT_MEASURED = "not_measured"


@dataclass
class AccuracyMetric:
    """Single accuracy metric with declared thresholds.

    Implements Article 15 requirement for declared accuracy levels.

    Attributes:
        metric_type: Type of the metric
        name: Human-readable metric name
        description: Detailed description of what the metric measures
        expected_value: Target value for optimal performance
        minimum_value: Minimum acceptable value (threshold for alerts)
        critical_value: Value below which system should be halted
        measurement_method: How the metric is calculated
        measurement_window: Time window for measurement (e.g., "252-day")
        unit: Unit of measurement (e.g., "ratio", "percentage")
        higher_is_better: Whether higher values indicate better performance
    """

    metric_type: MetricType
    name: str
    description: str
    expected_value: float
    minimum_value: float
    critical_value: Optional[float] = None
    measurement_method: str = ""
    measurement_window: str = ""
    unit: str = ""
    higher_is_better: bool = True

    def evaluate_value(self, current_value: float) -> MetricStatus:
        """Evaluate current value against declared thresholds.

        Args:
            current_value: Current measured value of the metric

        Returns:
            MetricStatus indicating performance relative to thresholds
        """
        if not math.isfinite(current_value):
            return MetricStatus.NOT_MEASURED

        if self.higher_is_better:
            if self.critical_value is not None and current_value < self.critical_value:
                return MetricStatus.CRITICAL
            if current_value < self.minimum_value:
                return MetricStatus.BELOW_MINIMUM
            if current_value >= self.expected_value:
                return MetricStatus.EXCEEDS_EXPECTED
            if current_value >= (self.expected_value + self.minimum_value) / 2:
                return MetricStatus.MEETS_EXPECTED
            return MetricStatus.MEETS_MINIMUM
        else:
            # For metrics where lower is better (e.g., max drawdown)
            if self.critical_value is not None and current_value > self.critical_value:
                return MetricStatus.CRITICAL
            if current_value > self.minimum_value:
                return MetricStatus.BELOW_MINIMUM
            if current_value <= self.expected_value:
                return MetricStatus.EXCEEDS_EXPECTED
            if current_value <= (self.expected_value + self.minimum_value) / 2:
                return MetricStatus.MEETS_EXPECTED
            return MetricStatus.MEETS_MINIMUM

    def get_compliance_text(self, current_value: float) -> str:
        """Generate Article 15 compliant description of metric status.

        Args:
            current_value: Current measured value

        Returns:
            Human-readable compliance status text
        """
        status = self.evaluate_value(current_value)
        direction = "above" if self.higher_is_better else "below"

        if status == MetricStatus.EXCEEDS_EXPECTED:
            return (
                f"{self.name}: {current_value:.4f} {self.unit} - EXCELLENT. "
                f"Exceeds expected threshold ({self.expected_value} {self.unit})."
            )
        elif status == MetricStatus.MEETS_EXPECTED:
            return (
                f"{self.name}: {current_value:.4f} {self.unit} - GOOD. " f"Within expected range."
            )
        elif status == MetricStatus.MEETS_MINIMUM:
            return (
                f"{self.name}: {current_value:.4f} {self.unit} - ACCEPTABLE. "
                f"Above minimum threshold ({self.minimum_value} {self.unit})."
            )
        elif status == MetricStatus.BELOW_MINIMUM:
            return (
                f"{self.name}: {current_value:.4f} {self.unit} - WARNING. "
                f"Below minimum threshold ({self.minimum_value} {self.unit}). "
                f"Review required per Article 15."
            )
        elif status == MetricStatus.CRITICAL:
            return (
                f"{self.name}: {current_value:.4f} {self.unit} - CRITICAL. "
                f"Below critical threshold ({self.critical_value} {self.unit}). "
                f"Immediate action required per Article 15."
            )
        else:
            return f"{self.name}: Not measured or invalid value."


@dataclass
class MetricMeasurement:
    """Single measurement of a metric at a point in time.

    Attributes:
        metric_type: Type of metric measured
        value: Measured value
        timestamp: When measurement was taken
        measurement_id: Unique identifier for this measurement
        window_start: Start of measurement window
        window_end: End of measurement window
        sample_size: Number of data points used
        confidence_interval: Optional (lower, upper) confidence bounds
        metadata: Additional measurement context
    """

    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    measurement_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    sample_size: int = 0
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeclaredAccuracyMetrics:
    """Complete declaration of accuracy metrics per Article 15.

    This class represents the formal declaration of accuracy levels
    that must be included in the system's instructions of use.

    Attributes:
        declaration_id: Unique identifier for this declaration
        system_name: Name of the AI system
        system_version: Version of the AI system
        declaration_date: When this declaration was made
        metrics: List of declared accuracy metrics
        measurement_conditions: Description of measurement conditions
        limitations: Known limitations affecting accuracy
        regulatory_notes: Additional regulatory compliance notes
    """

    declaration_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    system_name: str = "TradingBot2"
    system_version: str = "1.0.0"
    declaration_date: datetime = field(default_factory=datetime.utcnow)
    metrics: List[AccuracyMetric] = field(default_factory=list)
    measurement_conditions: str = ""
    limitations: List[str] = field(default_factory=list)
    regulatory_notes: str = ""

    def generate_instructions_of_use_section(self) -> str:
        """Generate the accuracy section for instructions of use.

        Article 15 requires accuracy levels to be declared in
        the accompanying instructions of use.

        Returns:
            Formatted text suitable for instructions of use document
        """
        lines = [
            "=" * 70,
            "ACCURACY METRICS DECLARATION",
            f"Per EU AI Act Article 15 (Regulation (EU) 2024/1689)",
            "=" * 70,
            "",
            f"System: {self.system_name}",
            f"Version: {self.system_version}",
            f"Declaration ID: {self.declaration_id}",
            f"Declaration Date: {self.declaration_date.isoformat()}",
            "",
            "-" * 70,
            "DECLARED ACCURACY LEVELS",
            "-" * 70,
            "",
        ]

        for metric in self.metrics:
            lines.extend(
                [
                    f"Metric: {metric.name}",
                    f"  Type: {metric.metric_type.name}",
                    f"  Description: {metric.description}",
                    f"  Expected Value: {metric.expected_value} {metric.unit}",
                    f"  Minimum Acceptable: {metric.minimum_value} {metric.unit}",
                ]
            )
            if metric.critical_value is not None:
                lines.append(f"  Critical Threshold: {metric.critical_value} {metric.unit}")
            lines.extend(
                [
                    f"  Measurement Method: {metric.measurement_method}",
                    f"  Measurement Window: {metric.measurement_window}",
                    "",
                ]
            )

        lines.extend(
            [
                "-" * 70,
                "MEASUREMENT CONDITIONS",
                "-" * 70,
                self.measurement_conditions or "Standard market conditions.",
                "",
                "-" * 70,
                "KNOWN LIMITATIONS",
                "-" * 70,
            ]
        )

        if self.limitations:
            for i, limitation in enumerate(self.limitations, 1):
                lines.append(f"  {i}. {limitation}")
        else:
            lines.append("  No specific limitations declared.")

        lines.extend(
            [
                "",
                "-" * 70,
                "REGULATORY COMPLIANCE NOTES",
                "-" * 70,
                self.regulatory_notes or "System complies with EU AI Act Article 15 requirements.",
                "",
                "=" * 70,
            ]
        )

        return "\n".join(lines)


class AccuracyMonitor:
    """Monitors and tracks accuracy metrics over time.

    Implements continuous monitoring of declared accuracy levels
    as required by Article 15 ("perform consistently throughout
    their lifecycle").

    Thread-safe for production use.
    """

    def __init__(
        self,
        declared_metrics: Optional[DeclaredAccuracyMetrics] = None,
        alert_callback: Optional[Callable[[AccuracyMetric, float, MetricStatus], None]] = None,
        history_limit: int = 10000,
    ):
        """Initialize accuracy monitor.

        Args:
            declared_metrics: The declared accuracy metrics to monitor
            alert_callback: Called when metric status changes
            history_limit: Maximum measurements to retain per metric
        """
        self._lock = threading.RLock()
        self._declared_metrics = declared_metrics or self._create_default_declaration()
        self._alert_callback = alert_callback
        self._history_limit = history_limit

        # Metric lookup
        self._metrics_by_type: Dict[MetricType, AccuracyMetric] = {
            m.metric_type: m for m in self._declared_metrics.metrics
        }

        # Measurement history
        self._history: Dict[MetricType, List[MetricMeasurement]] = {
            m.metric_type: [] for m in self._declared_metrics.metrics
        }

        # Current status cache
        self._current_status: Dict[MetricType, MetricStatus] = {
            m.metric_type: MetricStatus.NOT_MEASURED for m in self._declared_metrics.metrics
        }

        # Statistics
        self._total_measurements = 0
        self._alerts_triggered = 0

    def _create_default_declaration(self) -> DeclaredAccuracyMetrics:
        """Create default accuracy metrics declaration for trading.

        Based on EU_AI_ACT_INTEGRATION_PLAN.md section 1.3.1.

        Returns:
            DeclaredAccuracyMetrics with trading-specific metrics
        """
        return DeclaredAccuracyMetrics(
            system_name="TradingBot2",
            system_version="1.0.0",
            metrics=get_default_trading_metrics(),
            measurement_conditions=(
                "Metrics are measured under standard market conditions. "
                "Extreme volatility events, market circuit breakers, and "
                "force majeure events are excluded from measurements. "
                "All measurements use UTC timestamps."
            ),
            limitations=[
                "Accuracy may degrade during extreme market volatility",
                "Past performance does not guarantee future results",
                "Metrics assume normal market liquidity conditions",
                "Direction accuracy measured on close-to-close basis",
                "Sharpe ratio assumes risk-free rate of 0%",
            ],
            regulatory_notes=(
                "This accuracy declaration is provided in compliance with "
                "EU AI Act Article 15 (Regulation (EU) 2024/1689). "
                "Accuracy levels are monitored continuously and any "
                "significant degradation will trigger appropriate alerts. "
                "System is classified as HIGH-RISK AI SYSTEM per Annex III "
                "(algorithmic trading in financial services)."
            ),
        )

    def record_measurement(
        self,
        metric_type: MetricType,
        value: float,
        window_start: Optional[datetime] = None,
        window_end: Optional[datetime] = None,
        sample_size: int = 0,
        confidence_interval: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MetricMeasurement:
        """Record a new metric measurement.

        Args:
            metric_type: Type of metric being measured
            value: Measured value
            window_start: Start of measurement window
            window_end: End of measurement window
            sample_size: Number of data points used
            confidence_interval: Optional (lower, upper) confidence bounds
            metadata: Additional context

        Returns:
            The recorded measurement
        """
        with self._lock:
            measurement = MetricMeasurement(
                metric_type=metric_type,
                value=value,
                window_start=window_start,
                window_end=window_end,
                sample_size=sample_size,
                confidence_interval=confidence_interval,
                metadata=metadata or {},
            )

            # Add to history
            if metric_type in self._history:
                self._history[metric_type].append(measurement)

                # Trim history if needed
                if len(self._history[metric_type]) > self._history_limit:
                    self._history[metric_type] = self._history[metric_type][-self._history_limit :]
            else:
                self._history[metric_type] = [measurement]

            self._total_measurements += 1

            # Evaluate and potentially alert
            if metric_type in self._metrics_by_type:
                metric = self._metrics_by_type[metric_type]
                old_status = self._current_status.get(metric_type, MetricStatus.NOT_MEASURED)
                new_status = metric.evaluate_value(value)
                self._current_status[metric_type] = new_status

                # Alert on status change or critical
                if new_status != old_status or new_status in (
                    MetricStatus.BELOW_MINIMUM,
                    MetricStatus.CRITICAL,
                ):
                    self._trigger_alert(metric, value, new_status)

            return measurement

    def _trigger_alert(
        self,
        metric: AccuracyMetric,
        value: float,
        status: MetricStatus,
    ) -> None:
        """Trigger alert for metric status change.

        Args:
            metric: The metric that triggered the alert
            value: Current value
            status: New status
        """
        self._alerts_triggered += 1

        logger.warning(
            "Accuracy metric alert: %s = %.4f (%s)",
            metric.name,
            value,
            status.value,
        )

        if self._alert_callback:
            try:
                self._alert_callback(metric, value, status)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)

    def get_current_status(self, metric_type: MetricType) -> MetricStatus:
        """Get current status of a metric.

        Args:
            metric_type: Type of metric

        Returns:
            Current MetricStatus
        """
        with self._lock:
            return self._current_status.get(metric_type, MetricStatus.NOT_MEASURED)

    def get_latest_measurement(
        self,
        metric_type: MetricType,
    ) -> Optional[MetricMeasurement]:
        """Get most recent measurement for a metric.

        Args:
            metric_type: Type of metric

        Returns:
            Most recent MetricMeasurement or None
        """
        with self._lock:
            history = self._history.get(metric_type, [])
            return history[-1] if history else None

    def get_measurement_history(
        self,
        metric_type: MetricType,
        limit: Optional[int] = None,
    ) -> List[MetricMeasurement]:
        """Get measurement history for a metric.

        Args:
            metric_type: Type of metric
            limit: Maximum measurements to return

        Returns:
            List of MetricMeasurement, newest last
        """
        with self._lock:
            history = self._history.get(metric_type, [])
            if limit:
                return list(history[-limit:])
            return list(history)

    def compute_metric_statistics(
        self,
        metric_type: MetricType,
    ) -> Dict[str, float]:
        """Compute statistics for a metric's measurement history.

        Args:
            metric_type: Type of metric

        Returns:
            Dictionary with mean, std, min, max, trend
        """
        with self._lock:
            history = self._history.get(metric_type, [])

            if not history:
                return {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "count": 0,
                    "trend": float("nan"),
                }

            values = [m.value for m in history if math.isfinite(m.value)]

            if not values:
                return {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "count": 0,
                    "trend": float("nan"),
                }

            n = len(values)
            mean_val = sum(values) / n

            if n > 1:
                variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
                std_val = math.sqrt(variance)

                # Simple trend (slope of linear regression)
                x_mean = (n - 1) / 2
                numerator = sum((i - x_mean) * (v - mean_val) for i, v in enumerate(values))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                trend = numerator / denominator if denominator > 0 else 0.0
            else:
                std_val = 0.0
                trend = 0.0

            return {
                "mean": mean_val,
                "std": std_val,
                "min": min(values),
                "max": max(values),
                "count": n,
                "trend": trend,
            }

    def generate_compliance_report(self) -> str:
        """Generate Article 15 compliance report.

        Returns:
            Formatted compliance report text
        """
        with self._lock:
            lines = [
                "=" * 70,
                "ARTICLE 15 ACCURACY COMPLIANCE REPORT",
                f"Generated: {datetime.utcnow().isoformat()}",
                "=" * 70,
                "",
                f"Total Measurements: {self._total_measurements}",
                f"Alerts Triggered: {self._alerts_triggered}",
                "",
                "-" * 70,
                "METRIC STATUS SUMMARY",
                "-" * 70,
                "",
            ]

            for metric_type, metric in self._metrics_by_type.items():
                status = self._current_status.get(metric_type, MetricStatus.NOT_MEASURED)
                latest = self.get_latest_measurement(metric_type)
                stats = self.compute_metric_statistics(metric_type)

                lines.extend(
                    [
                        f"Metric: {metric.name}",
                        f"  Current Status: {status.value}",
                    ]
                )

                if latest:
                    lines.append(f"  Latest Value: {latest.value:.4f} {metric.unit}")
                    lines.append(f"  Measured At: {latest.timestamp.isoformat()}")
                    lines.append(metric.get_compliance_text(latest.value))

                if stats["count"] > 0:
                    lines.extend(
                        [
                            f"  Mean: {stats['mean']:.4f}",
                            f"  Std Dev: {stats['std']:.4f}",
                            f"  Min: {stats['min']:.4f}",
                            f"  Max: {stats['max']:.4f}",
                            f"  Trend: {stats['trend']:+.6f}",
                            f"  Sample Size: {stats['count']}",
                        ]
                    )

                lines.append("")

            # Overall compliance assessment
            statuses = list(self._current_status.values())
            critical_count = sum(1 for s in statuses if s == MetricStatus.CRITICAL)
            below_min_count = sum(1 for s in statuses if s == MetricStatus.BELOW_MINIMUM)

            lines.extend(
                [
                    "-" * 70,
                    "OVERALL COMPLIANCE ASSESSMENT",
                    "-" * 70,
                    "",
                ]
            )

            if critical_count > 0:
                lines.append(
                    f"⚠️  CRITICAL: {critical_count} metric(s) below critical threshold. "
                    "Immediate review required per Article 15."
                )
            elif below_min_count > 0:
                lines.append(
                    f"⚠️  WARNING: {below_min_count} metric(s) below minimum threshold. "
                    "Review recommended per Article 15."
                )
            else:
                lines.append(
                    "✅ All metrics within acceptable ranges. "
                    "System compliant with Article 15 accuracy requirements."
                )

            lines.extend(
                [
                    "",
                    "=" * 70,
                ]
            )

            return "\n".join(lines)

    def export_for_audit(self) -> Dict[str, Any]:
        """Export all data for regulatory audit.

        Returns:
            Complete audit export dictionary
        """
        with self._lock:
            return {
                "export_timestamp": datetime.utcnow().isoformat(),
                "declaration": {
                    "declaration_id": self._declared_metrics.declaration_id,
                    "system_name": self._declared_metrics.system_name,
                    "system_version": self._declared_metrics.system_version,
                    "declaration_date": self._declared_metrics.declaration_date.isoformat(),
                    "measurement_conditions": self._declared_metrics.measurement_conditions,
                    "limitations": self._declared_metrics.limitations,
                    "regulatory_notes": self._declared_metrics.regulatory_notes,
                },
                "declared_metrics": [
                    {
                        "metric_type": m.metric_type.name,
                        "name": m.name,
                        "description": m.description,
                        "expected_value": m.expected_value,
                        "minimum_value": m.minimum_value,
                        "critical_value": m.critical_value,
                        "measurement_method": m.measurement_method,
                        "measurement_window": m.measurement_window,
                        "unit": m.unit,
                        "higher_is_better": m.higher_is_better,
                    }
                    for m in self._declared_metrics.metrics
                ],
                "current_status": {
                    mt.name: status.value for mt, status in self._current_status.items()
                },
                "measurement_history": {
                    mt.name: [
                        {
                            "measurement_id": m.measurement_id,
                            "value": m.value,
                            "timestamp": m.timestamp.isoformat(),
                            "window_start": m.window_start.isoformat() if m.window_start else None,
                            "window_end": m.window_end.isoformat() if m.window_end else None,
                            "sample_size": m.sample_size,
                            "confidence_interval": m.confidence_interval,
                            "metadata": m.metadata,
                        }
                        for m in measurements
                    ]
                    for mt, measurements in self._history.items()
                },
                "statistics": {
                    "total_measurements": self._total_measurements,
                    "alerts_triggered": self._alerts_triggered,
                },
            }

    @property
    def declared_metrics(self) -> DeclaredAccuracyMetrics:
        """Get the declared accuracy metrics."""
        return self._declared_metrics


def get_default_trading_metrics() -> List[AccuracyMetric]:
    """Get default trading accuracy metrics per EU_AI_ACT_INTEGRATION_PLAN.md.

    Returns:
        List of AccuracyMetric for trading systems
    """
    return [
        # Performance Metrics
        AccuracyMetric(
            metric_type=MetricType.SHARPE_RATIO,
            name="Sharpe Ratio",
            description=(
                "Risk-adjusted return metric. Ratio of excess returns to "
                "standard deviation of returns. Measures return per unit of risk."
            ),
            expected_value=1.5,
            minimum_value=0.5,
            critical_value=0.0,
            measurement_method="Rolling calculation of (returns - rf_rate) / std(returns)",
            measurement_window="252 trading days",
            unit="ratio",
            higher_is_better=True,
        ),
        AccuracyMetric(
            metric_type=MetricType.MAX_DRAWDOWN,
            name="Maximum Drawdown",
            description=(
                "Maximum peak-to-trough decline in portfolio value. "
                "Measures worst historical loss from a peak."
            ),
            expected_value=0.15,  # 15%
            minimum_value=0.25,  # 25%
            critical_value=0.35,  # 35%
            measurement_method="Historical max (peak - trough) / peak",
            measurement_window="Since inception or specified period",
            unit="percentage",
            higher_is_better=False,  # Lower drawdown is better
        ),
        # Trade-Level Metrics
        AccuracyMetric(
            metric_type=MetricType.WIN_RATE,
            name="Win Rate",
            description=(
                "Percentage of profitable trades. Number of winning trades "
                "divided by total number of trades."
            ),
            expected_value=0.55,  # 55%
            minimum_value=0.45,  # 45%
            critical_value=0.40,  # 40%
            measurement_method="(Winning trades / Total trades) × 100",
            measurement_window="Rolling 100 trades or specified period",
            unit="percentage",
            higher_is_better=True,
        ),
        # Prediction Metrics
        AccuracyMetric(
            metric_type=MetricType.DIRECTION_ACCURACY,
            name="Prediction Accuracy",
            description=(
                "Accuracy of price direction predictions. Percentage of "
                "correct directional forecasts (up/down)."
            ),
            expected_value=0.52,  # 52%
            minimum_value=0.50,  # 50%
            critical_value=0.48,  # 48%
            measurement_method="(Correct direction predictions / Total predictions) × 100",
            measurement_window="Rolling 252 days",
            unit="percentage",
            higher_is_better=True,
        ),
        # Additional Risk Metrics
        AccuracyMetric(
            metric_type=MetricType.SORTINO_RATIO,
            name="Sortino Ratio",
            description=(
                "Risk-adjusted return using downside deviation. "
                "More appropriate than Sharpe for non-normal return distributions."
            ),
            expected_value=2.0,
            minimum_value=1.0,
            critical_value=0.5,
            measurement_method="(returns - target) / downside_deviation",
            measurement_window="252 trading days",
            unit="ratio",
            higher_is_better=True,
        ),
        AccuracyMetric(
            metric_type=MetricType.PROFIT_FACTOR,
            name="Profit Factor",
            description=(
                "Ratio of gross profits to gross losses. "
                "Values above 1.0 indicate profitable system."
            ),
            expected_value=1.5,
            minimum_value=1.1,
            critical_value=0.9,
            measurement_method="Gross Profit / Gross Loss",
            measurement_window="Rolling 100 trades",
            unit="ratio",
            higher_is_better=True,
        ),
        AccuracyMetric(
            metric_type=MetricType.CVAR_95,
            name="Conditional Value at Risk (95%)",
            description=(
                "Expected loss in the worst 5% of scenarios. " "Measures tail risk beyond VaR."
            ),
            expected_value=0.03,  # 3%
            minimum_value=0.05,  # 5%
            critical_value=0.10,  # 10%
            measurement_method="Mean of losses beyond 95th percentile VaR",
            measurement_window="252 trading days",
            unit="percentage",
            higher_is_better=False,  # Lower CVaR is better
        ),
        AccuracyMetric(
            metric_type=MetricType.CONSISTENCY_SCORE,
            name="Consistency Score",
            description=(
                "Measure of return consistency across time periods. "
                "Higher scores indicate more stable performance."
            ),
            expected_value=0.7,  # 70%
            minimum_value=0.5,  # 50%
            critical_value=0.3,  # 30%
            measurement_method="Percentage of profitable rolling periods",
            measurement_window="Monthly over 12 months",
            unit="percentage",
            higher_is_better=True,
        ),
    ]


def create_accuracy_monitor(
    alert_callback: Optional[Callable[[AccuracyMetric, float, MetricStatus], None]] = None,
    history_limit: int = 10000,
    custom_metrics: Optional[List[AccuracyMetric]] = None,
) -> AccuracyMonitor:
    """Factory function to create an AccuracyMonitor.

    Args:
        alert_callback: Optional callback for alerts
        history_limit: Maximum measurements to retain per metric
        custom_metrics: Optional custom metrics to use instead of defaults

    Returns:
        Configured AccuracyMonitor instance
    """
    declaration = DeclaredAccuracyMetrics(
        system_name="TradingBot2",
        system_version="1.0.0",
        metrics=custom_metrics or get_default_trading_metrics(),
        measurement_conditions=(
            "Metrics are measured under standard market conditions. "
            "Extreme volatility events, market circuit breakers, and "
            "force majeure events are excluded from measurements."
        ),
        limitations=[
            "Accuracy may degrade during extreme market volatility",
            "Past performance does not guarantee future results",
            "Metrics assume normal market liquidity conditions",
        ],
        regulatory_notes=(
            "This accuracy declaration is provided in compliance with "
            "EU AI Act Article 15 (Regulation (EU) 2024/1689)."
        ),
    )

    return AccuracyMonitor(
        declared_metrics=declaration,
        alert_callback=alert_callback,
        history_limit=history_limit,
    )


__all__ = [
    "MetricType",
    "MetricStatus",
    "AccuracyMetric",
    "MetricMeasurement",
    "DeclaredAccuracyMetrics",
    "AccuracyMonitor",
    "create_accuracy_monitor",
    "get_default_trading_metrics",
]
