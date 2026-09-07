# -*- coding: utf-8 -*-
"""
MiFID II Execution Quality Reporting.

This module generates execution quality reports as required by MiFID II
for best execution compliance demonstration.

Key Features:
    - Monthly execution quality summary
    - Quarterly execution quality report
    - Annual best execution review
    - Venue-level execution analysis
    - Top 5 venues per asset class reporting
    - Client notification support

Note: RTS 28 (annual execution quality reports) was repealed by EU 2023/2869.
However, best execution documentation requirements under Article 27 remain.
This module provides internal reporting for compliance demonstration.

References:
    - MiFID II Article 27: Best Execution
    - MiFID II Article 27(6): Best execution policy review
    - Delegated Regulation 2017/565 Article 65: Order execution policy
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Callable,
)

from services.algo_integration.best_execution import (
    ExecutionVenue,
    AssetClass,
    ExecutionQualityLevel,
    ExecutionAnalysis,
    BestExecutionPolicy,
    BestExecutionAnalyzer,
)
from services.algo_integration.venue_analysis import (
    VenueAnalyzer,
    VenuePerformanceMetrics,
)
from services.algo_integration.tca_compliance import (
    TCAComplianceWrapper,
    TCAAggregateMetrics,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ReportPeriod(Enum):
    """Report period types."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


class ReportFormat(Enum):
    """Report output formats."""

    JSON = "json"
    TEXT = "text"
    CSV = "csv"
    HTML = "html"


class ReportStatus(Enum):
    """Report generation status."""

    DRAFT = "draft"
    GENERATED = "generated"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    PUBLISHED = "published"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class VenueExecutionSummary:
    """
    Summary of execution quality at a single venue.

    Per MiFID II Article 27, firms must monitor execution quality
    at each venue used for order execution.
    """

    venue_mic: str = ""
    venue_name: str = ""
    venue_type: str = ""
    country: str = ""

    # Volume metrics
    total_orders: int = 0
    total_fills: int = 0
    total_quantity: Decimal = Decimal("0")
    total_notional: Decimal = Decimal("0")
    volume_share_pct: float = 0.0

    # Quality metrics
    avg_spread_bps: Decimal = Decimal("0")
    avg_latency_ms: float = 0.0
    fill_rate_pct: float = 0.0
    rejection_rate_pct: float = 0.0
    avg_price_improvement_bps: Decimal = Decimal("0")
    avg_cost_bps: Decimal = Decimal("0")

    # Scores
    quality_score: float = 0.0
    ranking: int = 0

    # Period comparison (vs. previous period)
    volume_change_pct: Optional[float] = None
    quality_change: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "venue_mic": self.venue_mic,
            "venue_name": self.venue_name,
            "venue_type": self.venue_type,
            "country": self.country,
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "total_quantity": str(self.total_quantity),
            "total_notional": str(self.total_notional),
            "volume_share_pct": self.volume_share_pct,
            "avg_spread_bps": str(self.avg_spread_bps),
            "avg_latency_ms": self.avg_latency_ms,
            "fill_rate_pct": self.fill_rate_pct,
            "rejection_rate_pct": self.rejection_rate_pct,
            "avg_price_improvement_bps": str(self.avg_price_improvement_bps),
            "avg_cost_bps": str(self.avg_cost_bps),
            "quality_score": self.quality_score,
            "ranking": self.ranking,
            "volume_change_pct": self.volume_change_pct,
            "quality_change": self.quality_change,
        }


@dataclass
class AssetClassExecutionSummary:
    """
    Summary of execution quality for an asset class.

    Includes top venues and aggregate metrics.
    """

    asset_class: AssetClass = AssetClass.EQUITY

    # Aggregate metrics
    total_orders: int = 0
    total_fills: int = 0
    total_notional: Decimal = Decimal("0")

    # Quality metrics
    avg_execution_score: float = 0.0
    avg_price_improvement_bps: Decimal = Decimal("0")
    avg_total_cost_bps: Decimal = Decimal("0")
    avg_latency_ms: float = 0.0
    avg_fill_rate: float = 0.0

    # Quality distribution
    excellent_pct: float = 0.0
    good_pct: float = 0.0
    acceptable_pct: float = 0.0
    below_standard_pct: float = 0.0
    poor_pct: float = 0.0

    # Top venues (ranked)
    top_venues: List[VenueExecutionSummary] = field(default_factory=list)

    # Venue count
    venues_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "asset_class": self.asset_class.value,
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "total_notional": str(self.total_notional),
            "avg_execution_score": self.avg_execution_score,
            "avg_price_improvement_bps": str(self.avg_price_improvement_bps),
            "avg_total_cost_bps": str(self.avg_total_cost_bps),
            "avg_latency_ms": self.avg_latency_ms,
            "avg_fill_rate": self.avg_fill_rate,
            "quality_distribution": {
                "excellent": self.excellent_pct,
                "good": self.good_pct,
                "acceptable": self.acceptable_pct,
                "below_standard": self.below_standard_pct,
                "poor": self.poor_pct,
            },
            "top_venues": [v.to_dict() for v in self.top_venues],
            "venues_used": self.venues_used,
        }


@dataclass
class ExecutionQualityReportMetadata:
    """
    Metadata for execution quality report.

    Includes document control and audit information.
    """

    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str = "execution_quality"
    period: ReportPeriod = ReportPeriod.MONTHLY
    period_start: Optional[date] = None
    period_end: Optional[date] = None

    # Firm information
    firm_lei: str = ""
    firm_name: str = ""

    # Generation info
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = "system"
    generation_duration_ms: float = 0.0

    # Status
    status: ReportStatus = ReportStatus.DRAFT

    # Review/approval
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    # Document control
    version: str = "1.0"
    report_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "period": self.period.value,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "firm_lei": self.firm_lei,
            "firm_name": self.firm_name,
            "generated_at": self.generated_at.isoformat(),
            "generated_by": self.generated_by,
            "status": self.status.value,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "version": self.version,
            "report_hash": self.report_hash,
        }


@dataclass
class ExecutionQualityReport:
    """
    Complete execution quality report.

    Provides comprehensive execution quality data for compliance
    demonstration per MiFID II Article 27.
    """

    metadata: ExecutionQualityReportMetadata = field(default_factory=ExecutionQualityReportMetadata)

    # Executive summary
    executive_summary: str = ""

    # Overall metrics
    total_orders: int = 0
    total_fills: int = 0
    total_notional: Decimal = Decimal("0")
    total_venues_used: int = 0
    avg_execution_score: float = 0.0
    avg_price_improvement_bps: Decimal = Decimal("0")
    avg_total_cost_bps: Decimal = Decimal("0")

    # Quality distribution
    quality_distribution: Dict[str, float] = field(default_factory=dict)

    # Asset class breakdowns
    asset_class_summaries: Dict[str, AssetClassExecutionSummary] = field(default_factory=dict)

    # Top venues overall
    top_venues_overall: List[VenueExecutionSummary] = field(default_factory=list)

    # TCA metrics
    tca_metrics: Optional[TCAAggregateMetrics] = None

    # Policy compliance
    policy_version: str = ""
    policy_review_due: bool = False
    compliance_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Period comparison
    vs_previous_period: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "executive_summary": self.executive_summary,
            "total_orders": self.total_orders,
            "total_fills": self.total_fills,
            "total_notional": str(self.total_notional),
            "total_venues_used": self.total_venues_used,
            "avg_execution_score": self.avg_execution_score,
            "avg_price_improvement_bps": str(self.avg_price_improvement_bps),
            "avg_total_cost_bps": str(self.avg_total_cost_bps),
            "quality_distribution": self.quality_distribution,
            "asset_class_summaries": {
                k: v.to_dict() for k, v in self.asset_class_summaries.items()
            },
            "top_venues_overall": [v.to_dict() for v in self.top_venues_overall],
            "policy_version": self.policy_version,
            "policy_review_due": self.policy_review_due,
            "compliance_issues": self.compliance_issues,
            "recommendations": self.recommendations,
            "vs_previous_period": self.vs_previous_period,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def compute_hash(self) -> str:
        """Compute report hash for integrity."""
        data = self.to_dict()
        data["metadata"]["report_hash"] = ""  # Exclude hash from hash
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass
class ReportGeneratorConfig:
    """Configuration for report generator."""

    firm_lei: str = ""
    firm_name: str = ""

    # Report settings
    top_venues_count: int = 5  # Top N venues per asset class
    include_tca_metrics: bool = True
    include_recommendations: bool = True

    # Thresholds for compliance issues
    min_execution_score: float = 50.0
    max_avg_cost_bps: Decimal = Decimal("100")
    min_fill_rate_pct: float = 90.0
    max_latency_ms: float = 1000.0

    # Output settings
    default_format: ReportFormat = ReportFormat.JSON
    output_directory: str = "state/compliance/reports"


# =============================================================================
# Report Generator
# =============================================================================


class ExecutionQualityReportGenerator:
    """
    Generates execution quality reports for MiFID II compliance.

    Aggregates data from best execution analyzer, TCA wrapper, and
    venue analyzer to create comprehensive reports.
    """

    def __init__(
        self,
        config: ReportGeneratorConfig,
        best_execution_analyzer: Optional[BestExecutionAnalyzer] = None,
        tca_wrapper: Optional[TCAComplianceWrapper] = None,
        venue_analyzer: Optional[VenueAnalyzer] = None,
        best_execution_policy: Optional[BestExecutionPolicy] = None,
        audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize report generator.

        Args:
            config: Generator configuration
            best_execution_analyzer: For execution analysis data
            tca_wrapper: For TCA metrics
            venue_analyzer: For venue performance data
            best_execution_policy: For policy compliance checking
            audit_callback: Callback for audit trail
        """
        self.config = config
        self.be_analyzer = best_execution_analyzer
        self.tca_wrapper = tca_wrapper
        self.venue_analyzer = venue_analyzer
        self.policy = best_execution_policy
        self.audit_callback = audit_callback

        self._reports: List[ExecutionQualityReport] = []
        self._lock = threading.Lock()

    def generate_report(
        self,
        period: ReportPeriod = ReportPeriod.MONTHLY,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        asset_classes: Optional[List[AssetClass]] = None,
    ) -> ExecutionQualityReport:
        """
        Generate execution quality report for a period.

        Args:
            period: Report period type
            start_date: Period start (auto-calculated if None)
            end_date: Period end (default: yesterday)
            asset_classes: Filter to specific asset classes

        Returns:
            ExecutionQualityReport for the period
        """
        start_time = time.time()

        # Calculate period dates
        if end_date is None:
            end_date = date.today() - timedelta(days=1)

        if start_date is None:
            start_date = self._calculate_period_start(period, end_date)

        # Create report with metadata
        report = ExecutionQualityReport()
        report.metadata = ExecutionQualityReportMetadata(
            period=period,
            period_start=start_date,
            period_end=end_date,
            firm_lei=self.config.firm_lei,
            firm_name=self.config.firm_name,
        )

        # Convert to datetime for queries
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        # Collect data from analyzers
        analyses = self._collect_analyses(start_dt, end_dt)

        if analyses:
            self._populate_overall_metrics(report, analyses)
            self._populate_quality_distribution(report, analyses)

            # Asset class breakdowns
            if asset_classes is None:
                asset_classes = list(AssetClass)

            for ac in asset_classes:
                ac_analyses = [a for a in analyses if self._get_asset_class(a) == ac]
                if ac_analyses:
                    summary = self._generate_asset_class_summary(ac, ac_analyses, start_dt, end_dt)
                    report.asset_class_summaries[ac.value] = summary

        # Top venues overall
        if self.venue_analyzer:
            report.top_venues_overall = self._get_top_venues(start_dt, end_dt)
            report.total_venues_used = len(report.top_venues_overall)

        # TCA metrics
        if self.config.include_tca_metrics and self.tca_wrapper:
            report.tca_metrics = self.tca_wrapper.get_aggregate_metrics(start_dt, end_dt)

        # Policy compliance
        if self.policy:
            report.policy_version = self.policy.version
            report.policy_review_due = self.policy.is_review_due()
            report.compliance_issues = self._check_compliance_issues(report)

        # Recommendations
        if self.config.include_recommendations:
            report.recommendations = self._generate_recommendations(report)

        # Executive summary
        report.executive_summary = self._generate_executive_summary(report)

        # Finalize
        report.metadata.generation_duration_ms = (time.time() - start_time) * 1000
        report.metadata.report_hash = report.compute_hash()
        report.metadata.status = ReportStatus.GENERATED

        # Store report
        with self._lock:
            self._reports.append(report)

        # Audit callback
        if self.audit_callback:
            self.audit_callback(
                {
                    "type": "execution_quality_report_generated",
                    "report_id": report.metadata.report_id,
                    "period": period.value,
                    "period_start": start_date.isoformat(),
                    "period_end": end_date.isoformat(),
                    "total_orders": report.total_orders,
                }
            )

        return report

    def _calculate_period_start(self, period: ReportPeriod, end_date: date) -> date:
        """Calculate period start date."""
        if period == ReportPeriod.DAILY:
            return end_date
        elif period == ReportPeriod.WEEKLY:
            return end_date - timedelta(days=6)
        elif period == ReportPeriod.MONTHLY:
            # First day of the month
            return end_date.replace(day=1)
        elif period == ReportPeriod.QUARTERLY:
            # First day of the quarter
            quarter_start_month = ((end_date.month - 1) // 3) * 3 + 1
            return end_date.replace(month=quarter_start_month, day=1)
        elif period == ReportPeriod.ANNUAL:
            return end_date.replace(month=1, day=1)
        else:
            return end_date - timedelta(days=30)  # Default 30 days

    def _collect_analyses(
        self,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[ExecutionAnalysis]:
        """Collect execution analyses for period."""
        if self.be_analyzer:
            return [
                a
                for a in self.be_analyzer.get_analyses()
                if start_dt.timestamp() * 1e9 <= a.analysis_timestamp_ns <= end_dt.timestamp() * 1e9
            ]
        return []

    def _populate_overall_metrics(
        self,
        report: ExecutionQualityReport,
        analyses: List[ExecutionAnalysis],
    ) -> None:
        """Populate overall metrics from analyses."""
        report.total_orders = len(analyses)
        report.total_fills = sum(1 for a in analyses if a.filled_size > 0)

        notionals = [a.execution_price * a.filled_size for a in analyses if a.filled_size > 0]
        report.total_notional = sum(notionals, Decimal("0"))

        scores = [a.weighted_score for a in analyses]
        if scores:
            report.avg_execution_score = statistics.mean(scores)

        pi_values = [float(a.price_improvement_bps) for a in analyses]
        if pi_values:
            report.avg_price_improvement_bps = Decimal(str(statistics.mean(pi_values)))

        cost_values = [float(a.total_cost_bps) for a in analyses]
        if cost_values:
            report.avg_total_cost_bps = Decimal(str(statistics.mean(cost_values)))

    def _populate_quality_distribution(
        self,
        report: ExecutionQualityReport,
        analyses: List[ExecutionAnalysis],
    ) -> None:
        """Populate quality distribution."""
        if not analyses:
            return

        total = len(analyses)
        dist = {}
        for level in ExecutionQualityLevel:
            count = sum(1 for a in analyses if a.quality_level == level)
            dist[level.value] = count / total * 100

        report.quality_distribution = dist

    def _get_asset_class(self, analysis: ExecutionAnalysis) -> AssetClass:
        """Determine asset class from analysis (simplified)."""
        # In a real implementation, this would look up the instrument
        # For now, default to EQUITY
        return AssetClass.EQUITY

    def _generate_asset_class_summary(
        self,
        asset_class: AssetClass,
        analyses: List[ExecutionAnalysis],
        start_dt: datetime,
        end_dt: datetime,
    ) -> AssetClassExecutionSummary:
        """Generate summary for an asset class."""
        summary = AssetClassExecutionSummary(asset_class=asset_class)

        summary.total_orders = len(analyses)
        summary.total_fills = sum(1 for a in analyses if a.filled_size > 0)

        notionals = [a.execution_price * a.filled_size for a in analyses if a.filled_size > 0]
        summary.total_notional = sum(notionals, Decimal("0"))

        scores = [a.weighted_score for a in analyses]
        if scores:
            summary.avg_execution_score = statistics.mean(scores)

        pi_values = [float(a.price_improvement_bps) for a in analyses]
        if pi_values:
            summary.avg_price_improvement_bps = Decimal(str(statistics.mean(pi_values)))

        cost_values = [float(a.total_cost_bps) for a in analyses]
        if cost_values:
            summary.avg_total_cost_bps = Decimal(str(statistics.mean(cost_values)))

        latencies = [a.latency_ms for a in analyses if a.latency_ms > 0]
        if latencies:
            summary.avg_latency_ms = statistics.mean(latencies)

        fill_rates = [a.fill_rate for a in analyses]
        if fill_rates:
            summary.avg_fill_rate = statistics.mean(fill_rates)

        # Quality distribution
        total = len(analyses)
        summary.excellent_pct = (
            sum(1 for a in analyses if a.quality_level == ExecutionQualityLevel.EXCELLENT)
            / total
            * 100
        )
        summary.good_pct = (
            sum(1 for a in analyses if a.quality_level == ExecutionQualityLevel.GOOD) / total * 100
        )
        summary.acceptable_pct = (
            sum(1 for a in analyses if a.quality_level == ExecutionQualityLevel.ACCEPTABLE)
            / total
            * 100
        )
        summary.below_standard_pct = (
            sum(1 for a in analyses if a.quality_level == ExecutionQualityLevel.BELOW_STANDARD)
            / total
            * 100
        )
        summary.poor_pct = (
            sum(1 for a in analyses if a.quality_level == ExecutionQualityLevel.POOR) / total * 100
        )

        # Top venues for this asset class
        if self.venue_analyzer:
            venues_used = set(a.venue_mic for a in analyses if a.venue_mic)
            summary.venues_used = len(venues_used)

            venue_summaries = []
            for venue_mic in venues_used:
                venue_analyses = [a for a in analyses if a.venue_mic == venue_mic]
                if venue_analyses:
                    vs = self._create_venue_summary(venue_mic, venue_analyses, start_dt, end_dt)
                    venue_summaries.append(vs)

            # Sort by quality score and take top N
            venue_summaries.sort(key=lambda v: v.quality_score, reverse=True)
            summary.top_venues = venue_summaries[: self.config.top_venues_count]

            # Assign rankings
            for i, vs in enumerate(summary.top_venues):
                vs.ranking = i + 1

        return summary

    def _create_venue_summary(
        self,
        venue_mic: str,
        analyses: List[ExecutionAnalysis],
        start_dt: datetime,
        end_dt: datetime,
    ) -> VenueExecutionSummary:
        """Create venue execution summary."""
        summary = VenueExecutionSummary(venue_mic=venue_mic)

        # Get venue info if available
        if self.venue_analyzer and venue_mic in self.venue_analyzer._venues:
            venue = self.venue_analyzer._venues[venue_mic]
            summary.venue_name = venue.name
            summary.venue_type = venue.venue_type.value
            summary.country = venue.country

        summary.total_orders = len(analyses)
        summary.total_fills = sum(1 for a in analyses if a.filled_size > 0)

        quantities = [a.filled_size for a in analyses]
        summary.total_quantity = sum(quantities, Decimal("0"))

        notionals = [a.execution_price * a.filled_size for a in analyses if a.filled_size > 0]
        summary.total_notional = sum(notionals, Decimal("0"))

        # Quality metrics
        spreads = [float(a.total_cost_bps) for a in analyses if a.total_cost_bps > 0]
        if spreads:
            summary.avg_spread_bps = Decimal(str(statistics.mean(spreads)))

        latencies = [a.latency_ms for a in analyses if a.latency_ms > 0]
        if latencies:
            summary.avg_latency_ms = statistics.mean(latencies)

        fill_rates = [a.fill_rate for a in analyses]
        if fill_rates:
            summary.fill_rate_pct = statistics.mean(fill_rates) * 100

        pi_values = [float(a.price_improvement_bps) for a in analyses]
        if pi_values:
            summary.avg_price_improvement_bps = Decimal(str(statistics.mean(pi_values)))

        cost_values = [float(a.total_cost_bps) for a in analyses]
        if cost_values:
            summary.avg_cost_bps = Decimal(str(statistics.mean(cost_values)))

        # Quality score
        scores = [a.weighted_score for a in analyses]
        if scores:
            summary.quality_score = statistics.mean(scores) * 100

        return summary

    def _get_top_venues(
        self,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[VenueExecutionSummary]:
        """Get top venues overall."""
        if not self.venue_analyzer:
            return []

        rankings = self.venue_analyzer.get_venue_rankings(start_dt, end_dt)

        top_venues = []
        for i, (mic, score) in enumerate(rankings[: self.config.top_venues_count]):
            metrics = self.venue_analyzer.get_venue_metrics(mic, start_dt, end_dt)

            summary = VenueExecutionSummary(
                venue_mic=mic,
                total_orders=metrics.total_orders,
                total_fills=metrics.total_fills,
                total_quantity=metrics.total_quantity_filled,
                total_notional=metrics.total_notional,
                avg_spread_bps=metrics.avg_spread_bps,
                avg_latency_ms=metrics.avg_latency_ms,
                fill_rate_pct=metrics.avg_fill_rate * 100,
                rejection_rate_pct=metrics.rejection_rate,
                avg_price_improvement_bps=metrics.avg_price_improvement_bps,
                avg_cost_bps=metrics.avg_cost_bps,
                quality_score=score,
                ranking=i + 1,
            )

            if mic in self.venue_analyzer._venues:
                venue = self.venue_analyzer._venues[mic]
                summary.venue_name = venue.name
                summary.venue_type = venue.venue_type.value
                summary.country = venue.country

            top_venues.append(summary)

        return top_venues

    def _check_compliance_issues(self, report: ExecutionQualityReport) -> List[str]:
        """Check for compliance issues."""
        issues = []

        # Execution score below threshold
        if report.avg_execution_score < self.config.min_execution_score:
            issues.append(
                f"Average execution score ({report.avg_execution_score:.1f}) "
                f"below threshold ({self.config.min_execution_score})"
            )

        # Cost above threshold
        if report.avg_total_cost_bps > self.config.max_avg_cost_bps:
            issues.append(
                f"Average total cost ({report.avg_total_cost_bps}bps) "
                f"exceeds threshold ({self.config.max_avg_cost_bps}bps)"
            )

        # Policy review due
        if report.policy_review_due:
            issues.append("Best execution policy review is due")

        # High percentage of poor quality
        poor_pct = report.quality_distribution.get("poor", 0)
        if poor_pct > 10:
            issues.append(f"High percentage of poor quality executions ({poor_pct:.1f}%)")

        return issues

    def _generate_recommendations(self, report: ExecutionQualityReport) -> List[str]:
        """Generate recommendations based on report data."""
        recommendations = []

        # Based on quality distribution
        poor_pct = report.quality_distribution.get("poor", 0) + report.quality_distribution.get(
            "below_standard", 0
        )
        if poor_pct > 20:
            recommendations.append(
                "Review execution strategy - significant portion of executions below standard. "
                "Consider adjusting venue selection or order routing rules."
            )

        # Based on cost
        if report.avg_total_cost_bps > Decimal("50"):
            recommendations.append(
                "Total execution costs are elevated. Consider negotiating lower commissions "
                "or exploring alternative venues with lower implicit costs."
            )

        # Based on TCA
        if report.tca_metrics and report.tca_metrics.pct_within_estimate < 80:
            recommendations.append(
                "Pre-trade cost estimates are frequently exceeded. "
                "Review and calibrate estimation models."
            )

        # Policy review
        if report.policy_review_due:
            recommendations.append(
                "Best execution policy is due for annual review. "
                "Schedule review with compliance team."
            )

        return recommendations

    def _generate_executive_summary(self, report: ExecutionQualityReport) -> str:
        """Generate executive summary text."""
        period_str = f"{report.metadata.period_start} to {report.metadata.period_end}"

        summary = f"""
EXECUTION QUALITY REPORT - EXECUTIVE SUMMARY
Period: {period_str}

OVERALL METRICS:
- Total Orders: {report.total_orders:,}
- Total Notional: {report.total_notional:,.2f} {self.config.firm_name}
- Average Execution Score: {report.avg_execution_score:.1f}/100
- Average Price Improvement: {report.avg_price_improvement_bps}bps
- Average Total Cost: {report.avg_total_cost_bps}bps

QUALITY DISTRIBUTION:
- Excellent: {report.quality_distribution.get('excellent', 0):.1f}%
- Good: {report.quality_distribution.get('good', 0):.1f}%
- Acceptable: {report.quality_distribution.get('acceptable', 0):.1f}%
- Below Standard: {report.quality_distribution.get('below_standard', 0):.1f}%
- Poor: {report.quality_distribution.get('poor', 0):.1f}%

VENUES:
- Total Venues Used: {report.total_venues_used}
"""

        if report.top_venues_overall:
            summary += "- Top Venues: "
            summary += ", ".join(
                f"{v.venue_mic} ({v.quality_score:.1f})" for v in report.top_venues_overall[:3]
            )
            summary += "\n"

        if report.compliance_issues:
            summary += f"\nCOMPLIANCE ISSUES: {len(report.compliance_issues)}\n"
            for issue in report.compliance_issues:
                summary += f"  - {issue}\n"

        if report.recommendations:
            summary += f"\nRECOMMENDATIONS:\n"
            for rec in report.recommendations:
                summary += f"  - {rec}\n"

        summary += f"\nPolicy Version: {report.policy_version}\n"
        summary += f"Report ID: {report.metadata.report_id}\n"

        return summary.strip()

    def mark_reviewed(self, report_id: str, reviewer: str) -> bool:
        """Mark report as reviewed."""
        with self._lock:
            for report in self._reports:
                if report.metadata.report_id == report_id:
                    report.metadata.reviewed_by = reviewer
                    report.metadata.reviewed_at = datetime.utcnow()
                    report.metadata.status = ReportStatus.REVIEWED
                    return True
        return False

    def mark_approved(self, report_id: str, approver: str) -> bool:
        """Mark report as approved."""
        with self._lock:
            for report in self._reports:
                if report.metadata.report_id == report_id:
                    report.metadata.approved_by = approver
                    report.metadata.approved_at = datetime.utcnow()
                    report.metadata.status = ReportStatus.APPROVED
                    return True
        return False

    def get_report(self, report_id: str) -> Optional[ExecutionQualityReport]:
        """Get report by ID."""
        with self._lock:
            for report in self._reports:
                if report.metadata.report_id == report_id:
                    return report
        return None

    def get_reports(
        self,
        period: Optional[ReportPeriod] = None,
        status: Optional[ReportStatus] = None,
    ) -> List[ExecutionQualityReport]:
        """Get reports with optional filters."""
        with self._lock:
            reports = self._reports.copy()

        if period:
            reports = [r for r in reports if r.metadata.period == period]
        if status:
            reports = [r for r in reports if r.metadata.status == status]

        return sorted(reports, key=lambda r: r.metadata.generated_at, reverse=True)

    def export_report(
        self,
        report: ExecutionQualityReport,
        format: ReportFormat = ReportFormat.JSON,
    ) -> str:
        """Export report to specified format."""
        if format == ReportFormat.JSON:
            return report.to_json()
        elif format == ReportFormat.TEXT:
            return report.executive_summary
        elif format == ReportFormat.CSV:
            return self._export_csv(report)
        elif format == ReportFormat.HTML:
            return self._export_html(report)
        else:
            return report.to_json()

    def _export_csv(self, report: ExecutionQualityReport) -> str:
        """Export report to CSV format."""
        lines = [
            "metric,value",
            f"report_id,{report.metadata.report_id}",
            f"period,{report.metadata.period.value}",
            f"period_start,{report.metadata.period_start}",
            f"period_end,{report.metadata.period_end}",
            f"total_orders,{report.total_orders}",
            f"total_notional,{report.total_notional}",
            f"avg_execution_score,{report.avg_execution_score}",
            f"avg_price_improvement_bps,{report.avg_price_improvement_bps}",
            f"avg_total_cost_bps,{report.avg_total_cost_bps}",
        ]

        # Add quality distribution
        for level, pct in report.quality_distribution.items():
            lines.append(f"quality_{level}_pct,{pct}")

        # Add top venues
        lines.append("")
        lines.append("venue_mic,venue_name,quality_score,total_orders,avg_cost_bps")
        for venue in report.top_venues_overall:
            lines.append(
                f"{venue.venue_mic},{venue.venue_name},{venue.quality_score},{venue.total_orders},{venue.avg_cost_bps}"
            )

        return "\n".join(lines)

    def _export_html(self, report: ExecutionQualityReport) -> str:
        """Export report to HTML format."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Execution Quality Report - {report.metadata.period_start} to {report.metadata.period_end}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        .metric {{ font-weight: bold; }}
        .excellent {{ color: green; }}
        .good {{ color: #4CAF50; }}
        .poor {{ color: red; }}
    </style>
</head>
<body>
    <h1>Execution Quality Report</h1>
    <p><strong>Period:</strong> {report.metadata.period_start} to {report.metadata.period_end}</p>
    <p><strong>Generated:</strong> {report.metadata.generated_at}</p>
    <p><strong>Report ID:</strong> {report.metadata.report_id}</p>

    <h2>Summary Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Orders</td><td>{report.total_orders:,}</td></tr>
        <tr><td>Total Notional</td><td>{report.total_notional:,.2f}</td></tr>
        <tr><td>Average Execution Score</td><td>{report.avg_execution_score:.1f}</td></tr>
        <tr><td>Average Price Improvement</td><td>{report.avg_price_improvement_bps}bps</td></tr>
        <tr><td>Average Total Cost</td><td>{report.avg_total_cost_bps}bps</td></tr>
    </table>

    <h2>Top Venues</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Venue</th>
            <th>Quality Score</th>
            <th>Orders</th>
            <th>Avg Cost (bps)</th>
        </tr>
"""

        for venue in report.top_venues_overall:
            html += f"""
        <tr>
            <td>{venue.ranking}</td>
            <td>{venue.venue_mic} - {venue.venue_name}</td>
            <td>{venue.quality_score:.1f}</td>
            <td>{venue.total_orders:,}</td>
            <td>{venue.avg_cost_bps}</td>
        </tr>
"""

        html += """
    </table>
"""

        if report.compliance_issues:
            html += "<h2>Compliance Issues</h2><ul>"
            for issue in report.compliance_issues:
                html += f"<li class='poor'>{issue}</li>"
            html += "</ul>"

        if report.recommendations:
            html += "<h2>Recommendations</h2><ul>"
            for rec in report.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul>"

        html += f"""
    <hr>
    <p><small>Policy Version: {report.policy_version} | Report Hash: {report.metadata.report_hash[:16]}...</small></p>
</body>
</html>
"""
        return html


# =============================================================================
# Factory Functions
# =============================================================================


def create_report_generator(
    firm_lei: str = "",
    firm_name: str = "",
    best_execution_analyzer: Optional[BestExecutionAnalyzer] = None,
    tca_wrapper: Optional[TCAComplianceWrapper] = None,
    venue_analyzer: Optional[VenueAnalyzer] = None,
    best_execution_policy: Optional[BestExecutionPolicy] = None,
    audit_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> ExecutionQualityReportGenerator:
    """
    Factory function to create report generator.

    Args:
        firm_lei: Firm's LEI
        firm_name: Firm's legal name
        best_execution_analyzer: For execution analysis data
        tca_wrapper: For TCA metrics
        venue_analyzer: For venue performance data
        best_execution_policy: For policy compliance checking
        audit_callback: Callback for audit trail

    Returns:
        Configured ExecutionQualityReportGenerator instance.
    """
    config = ReportGeneratorConfig(
        firm_lei=firm_lei,
        firm_name=firm_name,
    )

    return ExecutionQualityReportGenerator(
        config=config,
        best_execution_analyzer=best_execution_analyzer,
        tca_wrapper=tca_wrapper,
        venue_analyzer=venue_analyzer,
        best_execution_policy=best_execution_policy,
        audit_callback=audit_callback,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "ReportPeriod",
    "ReportFormat",
    "ReportStatus",
    # Data classes
    "VenueExecutionSummary",
    "AssetClassExecutionSummary",
    "ExecutionQualityReportMetadata",
    "ExecutionQualityReport",
    "ReportGeneratorConfig",
    # Main class
    "ExecutionQualityReportGenerator",
    # Factory
    "create_report_generator",
]
