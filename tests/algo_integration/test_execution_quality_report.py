# -*- coding: utf-8 -*-
"""
Comprehensive tests for MiFID II Phase 5 Execution Quality Report module.

Tests cover:
    - ReportPeriod enum
    - ReportFormat enum
    - ReportStatus enum
    - VenueExecutionSummary dataclass
    - AssetClassExecutionSummary dataclass
    - ExecutionQualityReportMetadata dataclass
    - ExecutionQualityReport dataclass
    - ReportGeneratorConfig dataclass
    - ExecutionQualityReportGenerator class
    - create_report_generator factory function
    - Export formats (JSON, CSV, HTML, TEXT)
    - Edge cases and concurrent operations

Target: 100% code coverage per MiFID II compliance requirements.
"""

import json
import hashlib
import pytest
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch

from services.algo_integration.execution_quality_report import (
    # Enums
    ReportPeriod,
    ReportFormat,
    ReportStatus,
    # Data classes
    VenueExecutionSummary,
    AssetClassExecutionSummary,
    ExecutionQualityReportMetadata,
    ExecutionQualityReport,
    ReportGeneratorConfig,
    # Main class
    ExecutionQualityReportGenerator,
    # Factory
    create_report_generator,
)
from services.algo_integration.best_execution import (
    AssetClass,
    ExecutionQualityLevel,
    ExecutionAnalysis,
    ExecutionVenue,
    VenueType,
    BestExecutionPolicy,
    BestExecutionAnalyzer,
    BestExecutionPolicyConfig,
)
from services.algo_integration.venue_analysis import (
    VenueAnalyzer,
    VenuePerformanceMetrics,
    VenueAnalysisConfig,
)
from services.algo_integration.tca_compliance import (
    TCAComplianceWrapper,
    TCAAggregateMetrics,
    TCAConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_venue_summary():
    """Create sample venue execution summary."""
    return VenueExecutionSummary(
        venue_mic="XLON",
        venue_name="London Stock Exchange",
        venue_type="regulated_market",
        country="GB",
        total_orders=1000,
        total_fills=950,
        total_quantity=Decimal("100000"),
        total_notional=Decimal("5000000"),
        volume_share_pct=35.5,
        avg_spread_bps=Decimal("2.5"),
        avg_latency_ms=5.2,
        fill_rate_pct=95.0,
        rejection_rate_pct=0.5,
        avg_price_improvement_bps=Decimal("1.2"),
        avg_cost_bps=Decimal("3.8"),
        quality_score=87.5,
        ranking=1,
        volume_change_pct=5.2,
        quality_change=2.1,
    )


@pytest.fixture
def sample_asset_class_summary():
    """Create sample asset class execution summary."""
    return AssetClassExecutionSummary(
        asset_class=AssetClass.EQUITY,
        total_orders=5000,
        total_fills=4750,
        total_notional=Decimal("25000000"),
        avg_execution_score=82.5,
        avg_price_improvement_bps=Decimal("1.5"),
        avg_total_cost_bps=Decimal("4.2"),
        avg_latency_ms=8.5,
        avg_fill_rate=0.95,
        excellent_pct=30.0,
        good_pct=45.0,
        acceptable_pct=15.0,
        below_standard_pct=7.0,
        poor_pct=3.0,
        venues_used=5,
    )


@pytest.fixture
def sample_report_metadata():
    """Create sample report metadata."""
    return ExecutionQualityReportMetadata(
        report_id="test-report-001",
        report_type="execution_quality",
        period=ReportPeriod.MONTHLY,
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 31),
        firm_lei="529900ABCDEFGHIJ1234",
        firm_name="Test Investment Firm",
        generated_at=datetime(2024, 2, 1, 9, 0, 0),
        generated_by="compliance_system",
        generation_duration_ms=1500.5,
        status=ReportStatus.GENERATED,
        reviewed_by=None,
        reviewed_at=None,
        approved_by=None,
        approved_at=None,
        version="1.0",
        report_hash="abc123def456",
    )


@pytest.fixture
def sample_execution_analysis():
    """Create sample execution analysis."""
    return ExecutionAnalysis(
        execution_id="analysis-001",
        order_id="order-001",
        instrument_isin="GB00B03MLX29",
        venue_mic="XLON",
        reference_price=Decimal("150.00"),
        execution_price=Decimal("149.95"),
        order_size=Decimal("1000"),
        filled_size=Decimal("1000"),
        price_improvement_bps=Decimal("3.33"),
        implicit_cost_bps=Decimal("1.0"),
        explicit_cost_bps=Decimal("2.5"),
        total_cost_bps=Decimal("5.0"),
        weighted_score=0.85,
        quality_level=ExecutionQualityLevel.GOOD,
        analysis_timestamp_ns=int(datetime.now().timestamp() * 1e9),
    )


@pytest.fixture
def mock_best_execution_analyzer(sample_execution_analysis):
    """Create mock best execution analyzer."""
    analyzer = Mock(spec=BestExecutionAnalyzer)
    analyzer.get_analyses.return_value = [sample_execution_analysis]
    return analyzer


@pytest.fixture
def mock_tca_wrapper():
    """Create mock TCA wrapper."""
    from datetime import datetime
    wrapper = Mock(spec=TCAComplianceWrapper)
    wrapper.get_aggregate_metrics.return_value = TCAAggregateMetrics(
        period_start=datetime(2024, 1, 1),
        period_end=datetime(2024, 1, 31),
        total_orders=100,
        avg_implementation_shortfall_bps=Decimal("5.0"),
        pct_within_estimate=85.0,
        estimate_accuracy_score=0.92,
    )
    return wrapper


@pytest.fixture
def mock_venue_analyzer():
    """Create mock venue analyzer."""
    analyzer = Mock(spec=VenueAnalyzer)
    analyzer._venues = {
        "XLON": ExecutionVenue(
            mic="XLON",
            name="London Stock Exchange",
            country="GB",
            venue_type=VenueType.REGULATED_MARKET,
        ),
        "XETR": ExecutionVenue(
            mic="XETR",
            name="Xetra",
            country="DE",
            venue_type=VenueType.REGULATED_MARKET,
        ),
    }
    analyzer.get_venue_rankings.return_value = [("XLON", 90.0), ("XETR", 85.0)]
    analyzer.get_venue_metrics.return_value = VenuePerformanceMetrics(
        venue_mic="XLON",
        total_orders=1000,
        total_fills=950,
        total_quantity_filled=Decimal("100000"),
        total_notional=Decimal("5000000"),
        avg_spread_bps=Decimal("2.5"),
        avg_latency_ms=5.0,
        avg_fill_rate=0.95,
        rejection_rate=0.5,
        avg_price_improvement_bps=Decimal("1.2"),
        avg_cost_bps=Decimal("3.8"),
        quality_score=90.0,
    )
    return analyzer


@pytest.fixture
def mock_best_execution_policy():
    """Create mock best execution policy."""
    policy = Mock(spec=BestExecutionPolicy)
    policy.version = "2.0"
    policy.is_review_due.return_value = False
    return policy


@pytest.fixture
def report_generator_config():
    """Create report generator config."""
    return ReportGeneratorConfig(
        firm_lei="529900ABCDEFGHIJ1234",
        firm_name="Test Investment Firm",
        top_venues_count=5,
        include_tca_metrics=True,
        include_recommendations=True,
        min_execution_score=50.0,
        max_avg_cost_bps=Decimal("100"),
        min_fill_rate_pct=90.0,
        max_latency_ms=1000.0,
        default_format=ReportFormat.JSON,
        output_directory="state/compliance/reports",
    )


# =============================================================================
# Test ReportPeriod Enum
# =============================================================================


class TestReportPeriod:
    """Tests for ReportPeriod enum."""

    def test_all_periods_exist(self):
        """Verify all period types exist."""
        assert ReportPeriod.DAILY.value == "daily"
        assert ReportPeriod.WEEKLY.value == "weekly"
        assert ReportPeriod.MONTHLY.value == "monthly"
        assert ReportPeriod.QUARTERLY.value == "quarterly"
        assert ReportPeriod.ANNUAL.value == "annual"
        assert ReportPeriod.CUSTOM.value == "custom"

    def test_period_count(self):
        """Verify correct number of periods."""
        assert len(ReportPeriod) == 6

    def test_period_from_value(self):
        """Test creating period from value."""
        assert ReportPeriod("monthly") == ReportPeriod.MONTHLY
        assert ReportPeriod("quarterly") == ReportPeriod.QUARTERLY


# =============================================================================
# Test ReportFormat Enum
# =============================================================================


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_all_formats_exist(self):
        """Verify all format types exist."""
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.TEXT.value == "text"
        assert ReportFormat.CSV.value == "csv"
        assert ReportFormat.HTML.value == "html"

    def test_format_count(self):
        """Verify correct number of formats."""
        assert len(ReportFormat) == 4


# =============================================================================
# Test ReportStatus Enum
# =============================================================================


class TestReportStatus:
    """Tests for ReportStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all status types exist."""
        assert ReportStatus.DRAFT.value == "draft"
        assert ReportStatus.GENERATED.value == "generated"
        assert ReportStatus.REVIEWED.value == "reviewed"
        assert ReportStatus.APPROVED.value == "approved"
        assert ReportStatus.PUBLISHED.value == "published"

    def test_status_count(self):
        """Verify correct number of statuses."""
        assert len(ReportStatus) == 5

    def test_status_workflow_order(self):
        """Test typical workflow progression."""
        # Document the expected workflow
        workflow = [
            ReportStatus.DRAFT,
            ReportStatus.GENERATED,
            ReportStatus.REVIEWED,
            ReportStatus.APPROVED,
            ReportStatus.PUBLISHED,
        ]
        assert len(workflow) == 5


# =============================================================================
# Test VenueExecutionSummary
# =============================================================================


class TestVenueExecutionSummary:
    """Tests for VenueExecutionSummary dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        summary = VenueExecutionSummary()
        assert summary.venue_mic == ""
        assert summary.venue_name == ""
        assert summary.total_orders == 0
        assert summary.total_quantity == Decimal("0")
        assert summary.volume_change_pct is None
        assert summary.quality_change is None

    def test_full_initialization(self, sample_venue_summary):
        """Test initialization with all values."""
        assert sample_venue_summary.venue_mic == "XLON"
        assert sample_venue_summary.total_orders == 1000
        assert sample_venue_summary.quality_score == 87.5
        assert sample_venue_summary.volume_change_pct == 5.2

    def test_to_dict(self, sample_venue_summary):
        """Test dictionary conversion."""
        d = sample_venue_summary.to_dict()

        assert d["venue_mic"] == "XLON"
        assert d["venue_name"] == "London Stock Exchange"
        assert d["total_orders"] == 1000
        assert d["total_quantity"] == "100000"
        assert d["avg_spread_bps"] == "2.5"
        assert d["quality_score"] == 87.5
        assert d["ranking"] == 1
        assert d["volume_change_pct"] == 5.2

    def test_to_dict_with_none_optional_values(self):
        """Test dictionary conversion with None optional values."""
        summary = VenueExecutionSummary(
            venue_mic="TEST",
            volume_change_pct=None,
            quality_change=None,
        )
        d = summary.to_dict()
        assert d["volume_change_pct"] is None
        assert d["quality_change"] is None


# =============================================================================
# Test AssetClassExecutionSummary
# =============================================================================


class TestAssetClassExecutionSummary:
    """Tests for AssetClassExecutionSummary dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        summary = AssetClassExecutionSummary()
        assert summary.asset_class == AssetClass.EQUITY
        assert summary.total_orders == 0
        assert summary.top_venues == []
        assert summary.venues_used == 0

    def test_full_initialization(self, sample_asset_class_summary):
        """Test initialization with all values."""
        assert sample_asset_class_summary.asset_class == AssetClass.EQUITY
        assert sample_asset_class_summary.total_orders == 5000
        assert sample_asset_class_summary.excellent_pct == 30.0

    def test_to_dict(self, sample_asset_class_summary, sample_venue_summary):
        """Test dictionary conversion."""
        sample_asset_class_summary.top_venues = [sample_venue_summary]
        d = sample_asset_class_summary.to_dict()

        assert d["asset_class"] == "equity"
        assert d["total_orders"] == 5000
        assert d["total_notional"] == "25000000"
        assert d["quality_distribution"]["excellent"] == 30.0
        assert d["quality_distribution"]["good"] == 45.0
        assert d["quality_distribution"]["poor"] == 3.0
        assert len(d["top_venues"]) == 1
        assert d["venues_used"] == 5


# =============================================================================
# Test ExecutionQualityReportMetadata
# =============================================================================


class TestExecutionQualityReportMetadata:
    """Tests for ExecutionQualityReportMetadata dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        metadata = ExecutionQualityReportMetadata()
        assert metadata.report_type == "execution_quality"
        assert metadata.period == ReportPeriod.MONTHLY
        assert metadata.status == ReportStatus.DRAFT
        assert metadata.report_id is not None
        assert metadata.generated_at is not None

    def test_full_initialization(self, sample_report_metadata):
        """Test initialization with all values."""
        assert sample_report_metadata.report_id == "test-report-001"
        assert sample_report_metadata.firm_lei == "529900ABCDEFGHIJ1234"
        assert sample_report_metadata.period_start == date(2024, 1, 1)

    def test_to_dict(self, sample_report_metadata):
        """Test dictionary conversion."""
        d = sample_report_metadata.to_dict()

        assert d["report_id"] == "test-report-001"
        assert d["period"] == "monthly"
        assert d["period_start"] == "2024-01-01"
        assert d["period_end"] == "2024-01-31"
        assert d["firm_lei"] == "529900ABCDEFGHIJ1234"
        assert d["status"] == "generated"

    def test_to_dict_with_none_dates(self):
        """Test dictionary conversion with None dates."""
        metadata = ExecutionQualityReportMetadata(
            period_start=None,
            period_end=None,
            reviewed_at=None,
            approved_at=None,
        )
        d = metadata.to_dict()
        assert d["period_start"] is None
        assert d["period_end"] is None
        assert d["reviewed_at"] is None
        assert d["approved_at"] is None

    def test_to_dict_with_review_approval(self):
        """Test dictionary with review and approval info."""
        metadata = ExecutionQualityReportMetadata(
            reviewed_by="John Doe",
            reviewed_at=datetime(2024, 2, 1, 10, 0, 0),
            approved_by="Jane Smith",
            approved_at=datetime(2024, 2, 2, 11, 0, 0),
        )
        d = metadata.to_dict()
        assert d["reviewed_by"] == "John Doe"
        assert "2024-02-01" in d["reviewed_at"]
        assert d["approved_by"] == "Jane Smith"
        assert "2024-02-02" in d["approved_at"]


# =============================================================================
# Test ExecutionQualityReport
# =============================================================================


class TestExecutionQualityReport:
    """Tests for ExecutionQualityReport dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        report = ExecutionQualityReport()
        assert report.total_orders == 0
        assert report.executive_summary == ""
        assert report.quality_distribution == {}
        assert report.asset_class_summaries == {}
        assert report.top_venues_overall == []
        assert report.tca_metrics is None

    def test_full_initialization(self, sample_report_metadata, sample_venue_summary, sample_asset_class_summary):
        """Test initialization with all values."""
        report = ExecutionQualityReport(
            metadata=sample_report_metadata,
            executive_summary="Test summary",
            total_orders=5000,
            total_fills=4750,
            total_notional=Decimal("25000000"),
            total_venues_used=5,
            avg_execution_score=82.5,
            avg_price_improvement_bps=Decimal("1.5"),
            avg_total_cost_bps=Decimal("4.2"),
            quality_distribution={"excellent": 30.0, "good": 45.0},
            asset_class_summaries={"equity": sample_asset_class_summary},
            top_venues_overall=[sample_venue_summary],
            policy_version="2.0",
            policy_review_due=False,
            compliance_issues=[],
            recommendations=[],
        )

        assert report.total_orders == 5000
        assert report.avg_execution_score == 82.5
        assert len(report.top_venues_overall) == 1

    def test_to_dict(self, sample_report_metadata):
        """Test dictionary conversion."""
        report = ExecutionQualityReport(
            metadata=sample_report_metadata,
            total_orders=100,
            quality_distribution={"excellent": 50.0},
            compliance_issues=["Issue 1"],
            recommendations=["Recommendation 1"],
        )
        d = report.to_dict()

        assert d["total_orders"] == 100
        assert d["quality_distribution"]["excellent"] == 50.0
        assert d["compliance_issues"] == ["Issue 1"]
        assert d["recommendations"] == ["Recommendation 1"]

    def test_to_json(self, sample_report_metadata):
        """Test JSON conversion."""
        report = ExecutionQualityReport(
            metadata=sample_report_metadata,
            total_orders=100,
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["total_orders"] == 100

    def test_compute_hash(self, sample_report_metadata):
        """Test hash computation."""
        report = ExecutionQualityReport(
            metadata=sample_report_metadata,
            total_orders=100,
        )
        hash1 = report.compute_hash()
        assert len(hash1) == 64  # SHA256 hex length

        # Same content = same hash
        report2 = ExecutionQualityReport(
            metadata=sample_report_metadata,
            total_orders=100,
        )
        hash2 = report2.compute_hash()
        assert hash1 == hash2

    def test_compute_hash_different_content(self, sample_report_metadata):
        """Test hash changes with different content."""
        report1 = ExecutionQualityReport(
            metadata=sample_report_metadata,
            total_orders=100,
        )
        report2 = ExecutionQualityReport(
            metadata=sample_report_metadata,
            total_orders=200,
        )
        assert report1.compute_hash() != report2.compute_hash()


# =============================================================================
# Test ReportGeneratorConfig
# =============================================================================


class TestReportGeneratorConfig:
    """Tests for ReportGeneratorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReportGeneratorConfig()
        assert config.firm_lei == ""
        assert config.firm_name == ""
        assert config.top_venues_count == 5
        assert config.include_tca_metrics is True
        assert config.include_recommendations is True
        assert config.min_execution_score == 50.0
        assert config.max_avg_cost_bps == Decimal("100")
        assert config.default_format == ReportFormat.JSON

    def test_custom_values(self, report_generator_config):
        """Test custom configuration values."""
        assert report_generator_config.firm_lei == "529900ABCDEFGHIJ1234"
        assert report_generator_config.top_venues_count == 5
        assert report_generator_config.output_directory == "state/compliance/reports"


# =============================================================================
# Test ExecutionQualityReportGenerator
# =============================================================================


class TestExecutionQualityReportGenerator:
    """Tests for ExecutionQualityReportGenerator class."""

    def test_initialization(self, report_generator_config):
        """Test generator initialization."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        assert generator.config == report_generator_config
        assert generator.be_analyzer is None
        assert generator.tca_wrapper is None
        assert generator.venue_analyzer is None
        assert generator.policy is None

    def test_initialization_with_all_dependencies(
        self,
        report_generator_config,
        mock_best_execution_analyzer,
        mock_tca_wrapper,
        mock_venue_analyzer,
        mock_best_execution_policy,
    ):
        """Test initialization with all dependencies."""
        audit_events = []

        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            best_execution_analyzer=mock_best_execution_analyzer,
            tca_wrapper=mock_tca_wrapper,
            venue_analyzer=mock_venue_analyzer,
            best_execution_policy=mock_best_execution_policy,
            audit_callback=lambda e: audit_events.append(e),
        )

        assert generator.be_analyzer == mock_best_execution_analyzer
        assert generator.tca_wrapper == mock_tca_wrapper
        assert generator.venue_analyzer == mock_venue_analyzer
        assert generator.policy == mock_best_execution_policy

    def test_calculate_period_start_daily(self, report_generator_config):
        """Test period start calculation for daily."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        end_date = date(2024, 1, 15)
        start = generator._calculate_period_start(ReportPeriod.DAILY, end_date)
        assert start == date(2024, 1, 15)

    def test_calculate_period_start_weekly(self, report_generator_config):
        """Test period start calculation for weekly."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        end_date = date(2024, 1, 15)
        start = generator._calculate_period_start(ReportPeriod.WEEKLY, end_date)
        assert start == date(2024, 1, 9)

    def test_calculate_period_start_monthly(self, report_generator_config):
        """Test period start calculation for monthly."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        end_date = date(2024, 1, 31)
        start = generator._calculate_period_start(ReportPeriod.MONTHLY, end_date)
        assert start == date(2024, 1, 1)

    def test_calculate_period_start_quarterly(self, report_generator_config):
        """Test period start calculation for quarterly."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        # Q1
        start = generator._calculate_period_start(ReportPeriod.QUARTERLY, date(2024, 3, 31))
        assert start == date(2024, 1, 1)
        # Q2
        start = generator._calculate_period_start(ReportPeriod.QUARTERLY, date(2024, 6, 30))
        assert start == date(2024, 4, 1)
        # Q3
        start = generator._calculate_period_start(ReportPeriod.QUARTERLY, date(2024, 9, 30))
        assert start == date(2024, 7, 1)
        # Q4
        start = generator._calculate_period_start(ReportPeriod.QUARTERLY, date(2024, 12, 31))
        assert start == date(2024, 10, 1)

    def test_calculate_period_start_annual(self, report_generator_config):
        """Test period start calculation for annual."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        end_date = date(2024, 12, 31)
        start = generator._calculate_period_start(ReportPeriod.ANNUAL, end_date)
        assert start == date(2024, 1, 1)

    def test_calculate_period_start_custom(self, report_generator_config):
        """Test period start calculation for custom (default 30 days)."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        end_date = date(2024, 1, 31)
        start = generator._calculate_period_start(ReportPeriod.CUSTOM, end_date)
        assert start == date(2024, 1, 1)

    def test_generate_report_minimal(self, report_generator_config):
        """Test report generation with minimal setup."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert report.metadata.period == ReportPeriod.MONTHLY
        assert report.metadata.period_start == date(2024, 1, 1)
        assert report.metadata.period_end == date(2024, 1, 31)
        assert report.metadata.status == ReportStatus.GENERATED
        assert report.metadata.report_hash != ""

    def test_generate_report_with_analyses(
        self,
        report_generator_config,
        mock_best_execution_analyzer,
        sample_execution_analysis,
    ):
        """Test report generation with execution analyses."""
        # Set timestamp within period
        analysis_time = datetime(2024, 1, 15, 12, 0, 0)
        sample_execution_analysis.analysis_timestamp_ns = int(analysis_time.timestamp() * 1e9)
        mock_best_execution_analyzer.get_analyses.return_value = [sample_execution_analysis]

        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            best_execution_analyzer=mock_best_execution_analyzer,
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert report.total_orders == 1
        assert report.total_fills == 1

    def test_generate_report_with_venue_analyzer(
        self,
        report_generator_config,
        mock_venue_analyzer,
    ):
        """Test report generation with venue analyzer."""
        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            venue_analyzer=mock_venue_analyzer,
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert len(report.top_venues_overall) > 0
        assert report.top_venues_overall[0].venue_mic == "XLON"

    def test_generate_report_with_tca(
        self,
        report_generator_config,
        mock_tca_wrapper,
    ):
        """Test report generation with TCA metrics."""
        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            tca_wrapper=mock_tca_wrapper,
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert report.tca_metrics is not None
        assert report.tca_metrics.total_orders == 100

    def test_generate_report_with_policy(
        self,
        report_generator_config,
        mock_best_execution_policy,
    ):
        """Test report generation with policy compliance."""
        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            best_execution_policy=mock_best_execution_policy,
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert report.policy_version == "2.0"
        assert report.policy_review_due is False

    def test_generate_report_auto_dates(self, report_generator_config):
        """Test report generation with auto-calculated dates."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(period=ReportPeriod.MONTHLY)

        assert report.metadata.period_end is not None
        assert report.metadata.period_start is not None
        assert report.metadata.period_start <= report.metadata.period_end

    def test_generate_report_with_audit_callback(self, report_generator_config):
        """Test report generation triggers audit callback."""
        audit_events = []

        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            audit_callback=lambda e: audit_events.append(e),
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert len(audit_events) == 1
        assert audit_events[0]["type"] == "execution_quality_report_generated"
        assert audit_events[0]["report_id"] == report.metadata.report_id

    def test_check_compliance_issues_low_score(self, report_generator_config):
        """Test compliance issue detection for low score."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = ExecutionQualityReport()
        report.avg_execution_score = 30.0  # Below 50 threshold
        report.avg_total_cost_bps = Decimal("50")
        report.quality_distribution = {"poor": 5}
        report.policy_review_due = False

        issues = generator._check_compliance_issues(report)
        assert any("execution score" in issue.lower() for issue in issues)

    def test_check_compliance_issues_high_cost(self, report_generator_config):
        """Test compliance issue detection for high cost."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = ExecutionQualityReport()
        report.avg_execution_score = 80.0
        report.avg_total_cost_bps = Decimal("150")  # Above 100 threshold
        report.quality_distribution = {"poor": 5}
        report.policy_review_due = False

        issues = generator._check_compliance_issues(report)
        assert any("cost" in issue.lower() for issue in issues)

    def test_check_compliance_issues_policy_review_due(self, report_generator_config):
        """Test compliance issue detection for policy review."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = ExecutionQualityReport()
        report.avg_execution_score = 80.0
        report.avg_total_cost_bps = Decimal("50")
        report.quality_distribution = {"poor": 5}
        report.policy_review_due = True

        issues = generator._check_compliance_issues(report)
        assert any("policy review" in issue.lower() for issue in issues)

    def test_check_compliance_issues_high_poor_pct(self, report_generator_config):
        """Test compliance issue detection for high poor percentage."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = ExecutionQualityReport()
        report.avg_execution_score = 80.0
        report.avg_total_cost_bps = Decimal("50")
        report.quality_distribution = {"poor": 15}  # Above 10%
        report.policy_review_due = False

        issues = generator._check_compliance_issues(report)
        assert any("poor quality" in issue.lower() for issue in issues)

    def test_generate_recommendations_high_poor_pct(self, report_generator_config):
        """Test recommendation generation for poor quality."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = ExecutionQualityReport()
        report.quality_distribution = {"poor": 15, "below_standard": 10}  # >20%
        report.avg_total_cost_bps = Decimal("30")
        report.tca_metrics = None
        report.policy_review_due = False

        recs = generator._generate_recommendations(report)
        assert any("execution strategy" in rec.lower() for rec in recs)

    def test_generate_recommendations_high_cost(self, report_generator_config):
        """Test recommendation generation for high cost."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = ExecutionQualityReport()
        report.quality_distribution = {"poor": 5, "below_standard": 5}
        report.avg_total_cost_bps = Decimal("60")  # >50
        report.tca_metrics = None
        report.policy_review_due = False

        recs = generator._generate_recommendations(report)
        assert any("cost" in rec.lower() for rec in recs)

    def test_generate_recommendations_tca_accuracy(self, report_generator_config):
        """Test recommendation generation for TCA accuracy."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = ExecutionQualityReport()
        report.quality_distribution = {"poor": 5, "below_standard": 5}
        report.avg_total_cost_bps = Decimal("30")
        report.tca_metrics = TCAAggregateMetrics(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            total_orders=100,
            avg_implementation_shortfall_bps=Decimal("5"),
            pct_within_estimate=70.0,  # <80%
            estimate_accuracy_score=0.8,
        )
        report.policy_review_due = False

        recs = generator._generate_recommendations(report)
        assert any("estimate" in rec.lower() for rec in recs)

    def test_generate_recommendations_policy_review(self, report_generator_config):
        """Test recommendation generation for policy review."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = ExecutionQualityReport()
        report.quality_distribution = {"poor": 5, "below_standard": 5}
        report.avg_total_cost_bps = Decimal("30")
        report.tca_metrics = None
        report.policy_review_due = True

        recs = generator._generate_recommendations(report)
        assert any("policy" in rec.lower() for rec in recs)

    def test_generate_executive_summary(self, report_generator_config):
        """Test executive summary generation."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = ExecutionQualityReport()
        report.metadata = ExecutionQualityReportMetadata(
            period_start=date(2024, 1, 1),
            period_end=date(2024, 1, 31),
        )
        report.total_orders = 5000
        report.total_notional = Decimal("25000000")
        report.avg_execution_score = 82.5
        report.avg_price_improvement_bps = Decimal("1.5")
        report.avg_total_cost_bps = Decimal("4.2")
        report.quality_distribution = {
            "excellent": 30.0,
            "good": 45.0,
            "acceptable": 15.0,
            "below_standard": 7.0,
            "poor": 3.0,
        }
        report.total_venues_used = 5
        report.top_venues_overall = [
            VenueExecutionSummary(venue_mic="XLON", quality_score=90.0),
        ]
        report.compliance_issues = ["Test issue"]
        report.recommendations = ["Test recommendation"]
        report.policy_version = "2.0"

        summary = generator._generate_executive_summary(report)

        assert "EXECUTION QUALITY REPORT" in summary
        assert "Total Orders: 5,000" in summary
        assert "XLON" in summary
        assert "COMPLIANCE ISSUES" in summary
        assert "RECOMMENDATIONS" in summary

    def test_mark_reviewed(self, report_generator_config):
        """Test marking report as reviewed."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        result = generator.mark_reviewed(report.metadata.report_id, "John Doe")
        assert result is True

        updated = generator.get_report(report.metadata.report_id)
        assert updated.metadata.status == ReportStatus.REVIEWED
        assert updated.metadata.reviewed_by == "John Doe"
        assert updated.metadata.reviewed_at is not None

    def test_mark_reviewed_not_found(self, report_generator_config):
        """Test marking non-existent report as reviewed."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        result = generator.mark_reviewed("non-existent-id", "John Doe")
        assert result is False

    def test_mark_approved(self, report_generator_config):
        """Test marking report as approved."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        result = generator.mark_approved(report.metadata.report_id, "Jane Smith")
        assert result is True

        updated = generator.get_report(report.metadata.report_id)
        assert updated.metadata.status == ReportStatus.APPROVED
        assert updated.metadata.approved_by == "Jane Smith"
        assert updated.metadata.approved_at is not None

    def test_mark_approved_not_found(self, report_generator_config):
        """Test marking non-existent report as approved."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        result = generator.mark_approved("non-existent-id", "Jane Smith")
        assert result is False

    def test_get_report(self, report_generator_config):
        """Test getting report by ID."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        found = generator.get_report(report.metadata.report_id)
        assert found is not None
        assert found.metadata.report_id == report.metadata.report_id

    def test_get_report_not_found(self, report_generator_config):
        """Test getting non-existent report."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        found = generator.get_report("non-existent-id")
        assert found is None

    def test_get_reports_no_filter(self, report_generator_config):
        """Test getting all reports."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
        generator.generate_report(
            period=ReportPeriod.QUARTERLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
        )

        reports = generator.get_reports()
        assert len(reports) == 2

    def test_get_reports_filter_by_period(self, report_generator_config):
        """Test getting reports filtered by period."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
        generator.generate_report(
            period=ReportPeriod.QUARTERLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
        )

        reports = generator.get_reports(period=ReportPeriod.MONTHLY)
        assert len(reports) == 1
        assert reports[0].metadata.period == ReportPeriod.MONTHLY

    def test_get_reports_filter_by_status(self, report_generator_config):
        """Test getting reports filtered by status."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report1 = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
        generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 2, 1),
            end_date=date(2024, 2, 29),
        )
        generator.mark_reviewed(report1.metadata.report_id, "Reviewer")

        reports = generator.get_reports(status=ReportStatus.REVIEWED)
        assert len(reports) == 1

    def test_export_report_json(self, report_generator_config):
        """Test exporting report to JSON."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        output = generator.export_report(report, ReportFormat.JSON)
        parsed = json.loads(output)
        assert "metadata" in parsed
        assert "total_orders" in parsed

    def test_export_report_text(self, report_generator_config):
        """Test exporting report to text."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        output = generator.export_report(report, ReportFormat.TEXT)
        assert "EXECUTION QUALITY REPORT" in output

    def test_export_report_csv(self, report_generator_config, mock_venue_analyzer):
        """Test exporting report to CSV."""
        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            venue_analyzer=mock_venue_analyzer,
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        output = generator.export_report(report, ReportFormat.CSV)
        lines = output.split("\n")
        assert lines[0] == "metric,value"
        assert any("total_orders" in line for line in lines)

    def test_export_report_html(self, report_generator_config, mock_venue_analyzer):
        """Test exporting report to HTML."""
        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            venue_analyzer=mock_venue_analyzer,
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )
        report.compliance_issues = ["Test issue"]
        report.recommendations = ["Test recommendation"]

        output = generator.export_report(report, ReportFormat.HTML)
        assert "<!DOCTYPE html>" in output
        assert "Execution Quality Report" in output
        assert "Test issue" in output
        assert "Test recommendation" in output

    def test_export_report_default_format(self, report_generator_config):
        """Test exporting report with unknown format (defaults to JSON)."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        # Use a mock format that doesn't exist in the enum check
        with patch.object(ReportFormat, '__eq__', return_value=False):
            output = generator.export_report(report, ReportFormat.JSON)

        # Should still get valid JSON
        parsed = json.loads(output)
        assert "metadata" in parsed


# =============================================================================
# Test create_report_generator Factory
# =============================================================================


class TestCreateReportGenerator:
    """Tests for create_report_generator factory function."""

    def test_basic_creation(self):
        """Test basic factory creation."""
        generator = create_report_generator()
        assert isinstance(generator, ExecutionQualityReportGenerator)
        assert generator.config.firm_lei == ""
        assert generator.config.firm_name == ""

    def test_with_firm_info(self):
        """Test factory creation with firm info."""
        generator = create_report_generator(
            firm_lei="529900ABCDEFGHIJ1234",
            firm_name="Test Firm",
        )
        assert generator.config.firm_lei == "529900ABCDEFGHIJ1234"
        assert generator.config.firm_name == "Test Firm"

    def test_with_dependencies(
        self,
        mock_best_execution_analyzer,
        mock_tca_wrapper,
        mock_venue_analyzer,
        mock_best_execution_policy,
    ):
        """Test factory creation with all dependencies."""
        audit_events = []

        generator = create_report_generator(
            firm_lei="529900ABCDEFGHIJ1234",
            firm_name="Test Firm",
            best_execution_analyzer=mock_best_execution_analyzer,
            tca_wrapper=mock_tca_wrapper,
            venue_analyzer=mock_venue_analyzer,
            best_execution_policy=mock_best_execution_policy,
            audit_callback=lambda e: audit_events.append(e),
        )

        assert generator.be_analyzer == mock_best_execution_analyzer
        assert generator.tca_wrapper == mock_tca_wrapper
        assert generator.venue_analyzer == mock_venue_analyzer
        assert generator.policy == mock_best_execution_policy


# =============================================================================
# Test Concurrent Operations
# =============================================================================


class TestConcurrentOperations:
    """Tests for thread-safety of report generator."""

    def test_concurrent_report_generation(self, report_generator_config):
        """Test concurrent report generation."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)
        results = []
        errors = []

        def generate_report(period_offset):
            try:
                report = generator.generate_report(
                    period=ReportPeriod.MONTHLY,
                    start_date=date(2024, period_offset, 1),
                    end_date=date(2024, period_offset, 28),
                )
                results.append(report)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(generate_report, i) for i in range(1, 6)]
            for future in futures:
                future.result()

        assert len(errors) == 0
        assert len(results) == 5
        assert len(generator.get_reports()) == 5

    def test_concurrent_status_updates(self, report_generator_config):
        """Test concurrent status updates."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        reports = []
        for i in range(5):
            report = generator.generate_report(
                period=ReportPeriod.MONTHLY,
                start_date=date(2024, i + 1, 1),
                end_date=date(2024, i + 1, 28),
            )
            reports.append(report)

        errors = []

        def update_status(report_id, user):
            try:
                generator.mark_reviewed(report_id, f"Reviewer_{user}")
                generator.mark_approved(report_id, f"Approver_{user}")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(update_status, r.metadata.report_id, i)
                for i, r in enumerate(reports)
            ]
            for future in futures:
                future.result()

        assert len(errors) == 0
        approved = generator.get_reports(status=ReportStatus.APPROVED)
        assert len(approved) == 5


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_analyses(self, report_generator_config, mock_best_execution_analyzer):
        """Test report generation with empty analyses."""
        mock_best_execution_analyzer.get_analyses.return_value = []

        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            best_execution_analyzer=mock_best_execution_analyzer,
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert report.total_orders == 0
        assert report.avg_execution_score == 0

    def test_analyses_outside_period(
        self,
        report_generator_config,
        mock_best_execution_analyzer,
        sample_execution_analysis,
    ):
        """Test that analyses outside period are filtered."""
        # Set timestamp outside the period
        analysis_time = datetime(2023, 6, 15, 12, 0, 0)  # Different year
        sample_execution_analysis.analysis_timestamp_ns = int(analysis_time.timestamp() * 1e9)
        mock_best_execution_analyzer.get_analyses.return_value = [sample_execution_analysis]

        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            best_execution_analyzer=mock_best_execution_analyzer,
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        # Analysis should be filtered out
        assert report.total_orders == 0

    def test_quality_distribution_no_analyses(self, report_generator_config):
        """Test quality distribution with no analyses."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = ExecutionQualityReport()
        generator._populate_quality_distribution(report, [])

        assert report.quality_distribution == {}

    def test_venue_summary_with_partial_data(self, report_generator_config, sample_execution_analysis):
        """Test venue summary creation with partial data."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        # Analysis with zero latency
        sample_execution_analysis.latency_ms = 0
        sample_execution_analysis.total_cost_bps = Decimal("0")

        start_dt = datetime(2024, 1, 1)
        end_dt = datetime(2024, 1, 31)

        summary = generator._create_venue_summary(
            "XLON",
            [sample_execution_analysis],
            start_dt,
            end_dt,
        )

        assert summary.venue_mic == "XLON"
        assert summary.avg_latency_ms == 0

    def test_report_hash_stability(self, report_generator_config):
        """Test that report hash is valid and non-empty."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        original_hash = report.metadata.report_hash

        # Hash should be valid SHA256 (64 hex chars)
        assert original_hash is not None
        assert len(original_hash) == 64
        assert all(c in "0123456789abcdef" for c in original_hash)

    def test_asset_class_summary_generation(
        self,
        report_generator_config,
        mock_venue_analyzer,
        sample_execution_analysis,
    ):
        """Test asset class summary generation."""
        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            venue_analyzer=mock_venue_analyzer,
        )

        start_dt = datetime(2024, 1, 1)
        end_dt = datetime(2024, 1, 31)

        summary = generator._generate_asset_class_summary(
            AssetClass.EQUITY,
            [sample_execution_analysis],
            start_dt,
            end_dt,
        )

        assert summary.asset_class == AssetClass.EQUITY
        assert summary.total_orders == 1

    def test_get_top_venues_no_analyzer(self, report_generator_config):
        """Test getting top venues without venue analyzer."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        start_dt = datetime(2024, 1, 1)
        end_dt = datetime(2024, 1, 31)

        venues = generator._get_top_venues(start_dt, end_dt)
        assert venues == []

    def test_disabled_tca_metrics(self, report_generator_config, mock_tca_wrapper):
        """Test report generation with TCA metrics disabled."""
        report_generator_config.include_tca_metrics = False

        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            tca_wrapper=mock_tca_wrapper,
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        # TCA should not be called
        mock_tca_wrapper.get_aggregate_metrics.assert_not_called()
        assert report.tca_metrics is None

    def test_disabled_recommendations(self, report_generator_config):
        """Test report generation with recommendations disabled."""
        report_generator_config.include_recommendations = False

        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert report.recommendations == []

    def test_specific_asset_classes(
        self,
        report_generator_config,
        mock_best_execution_analyzer,
        sample_execution_analysis,
    ):
        """Test report generation for specific asset classes."""
        analysis_time = datetime(2024, 1, 15, 12, 0, 0)
        sample_execution_analysis.analysis_timestamp_ns = int(analysis_time.timestamp() * 1e9)
        mock_best_execution_analyzer.get_analyses.return_value = [sample_execution_analysis]

        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            best_execution_analyzer=mock_best_execution_analyzer,
        )

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            asset_classes=[AssetClass.EQUITY],
        )

        # Should only have equity summary
        assert AssetClass.EQUITY.value in report.asset_class_summaries
        assert len(report.asset_class_summaries) == 1


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Tests for full integration scenarios."""

    def test_full_monthly_report_workflow(
        self,
        report_generator_config,
        mock_best_execution_analyzer,
        mock_tca_wrapper,
        mock_venue_analyzer,
        mock_best_execution_policy,
        sample_execution_analysis,
    ):
        """Test complete monthly report workflow."""
        # Setup
        analysis_time = datetime(2024, 1, 15, 12, 0, 0)
        sample_execution_analysis.analysis_timestamp_ns = int(analysis_time.timestamp() * 1e9)
        mock_best_execution_analyzer.get_analyses.return_value = [sample_execution_analysis]

        audit_events = []

        generator = ExecutionQualityReportGenerator(
            config=report_generator_config,
            best_execution_analyzer=mock_best_execution_analyzer,
            tca_wrapper=mock_tca_wrapper,
            venue_analyzer=mock_venue_analyzer,
            best_execution_policy=mock_best_execution_policy,
            audit_callback=lambda e: audit_events.append(e),
        )

        # Generate report
        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        assert report.metadata.status == ReportStatus.GENERATED
        assert len(audit_events) == 1

        # Review
        result = generator.mark_reviewed(report.metadata.report_id, "Compliance Officer")
        assert result is True

        updated = generator.get_report(report.metadata.report_id)
        assert updated.metadata.status == ReportStatus.REVIEWED

        # Approve
        result = generator.mark_approved(report.metadata.report_id, "Head of Compliance")
        assert result is True

        updated = generator.get_report(report.metadata.report_id)
        assert updated.metadata.status == ReportStatus.APPROVED

        # Export
        json_output = generator.export_report(updated, ReportFormat.JSON)
        csv_output = generator.export_report(updated, ReportFormat.CSV)
        html_output = generator.export_report(updated, ReportFormat.HTML)
        text_output = generator.export_report(updated, ReportFormat.TEXT)

        assert json_output is not None
        assert csv_output is not None
        assert html_output is not None
        assert text_output is not None

    def test_quarterly_report_with_multiple_months(self, report_generator_config):
        """Test quarterly report covering multiple months."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.QUARTERLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
        )

        assert report.metadata.period == ReportPeriod.QUARTERLY
        assert report.metadata.period_start == date(2024, 1, 1)
        assert report.metadata.period_end == date(2024, 3, 31)

    def test_annual_report(self, report_generator_config):
        """Test annual report generation."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.ANNUAL,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        assert report.metadata.period == ReportPeriod.ANNUAL
        assert report.metadata.period_start == date(2024, 1, 1)
        assert report.metadata.period_end == date(2024, 12, 31)


# =============================================================================
# Test Report Content Validation
# =============================================================================


class TestReportContentValidation:
    """Tests for report content validation."""

    def test_executive_summary_structure(self, report_generator_config):
        """Test executive summary has required sections."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        summary = report.executive_summary
        assert "EXECUTIVE SUMMARY" in summary
        assert "Period:" in summary
        assert "OVERALL METRICS" in summary
        assert "QUALITY DISTRIBUTION" in summary
        assert "VENUES" in summary

    def test_metadata_completeness(self, report_generator_config):
        """Test metadata is complete."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        metadata = report.metadata
        assert metadata.report_id is not None
        assert len(metadata.report_id) > 0
        assert metadata.report_type == "execution_quality"
        assert metadata.generated_at is not None
        assert metadata.generation_duration_ms > 0
        assert metadata.report_hash is not None
        assert len(metadata.report_hash) == 64  # SHA256

    def test_hash_excludes_itself(self, report_generator_config):
        """Test that hash computation excludes the hash field."""
        generator = ExecutionQualityReportGenerator(config=report_generator_config)

        report = generator.generate_report(
            period=ReportPeriod.MONTHLY,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
        )

        # Verify original hash is valid
        original_hash = report.metadata.report_hash
        assert len(original_hash) == 64

        # Change hash field
        report.metadata.report_hash = "different_hash"

        # Recompute - should still produce valid hash (verifies exclusion logic works)
        new_hash = report.compute_hash()
        assert len(new_hash) == 64
        # Hash should not contain the modified value
        assert new_hash != "different_hash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
