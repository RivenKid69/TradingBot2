# -*- coding: utf-8 -*-
"""
Tests for EU AI Act Data Governance Framework (Article 10).

Tests cover:
- DataGovernanceFramework - Main governance orchestrator
- DataQualityAssessor - Quality assessment
- BiasDetector - Bias detection
- DataGapAnalyzer - Gap analysis
- DataValidator - Data validation
- Factory functions
- Thread safety
- Article 10 compliance requirements
"""

import json
import shutil
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from services.ai_act.data_governance import (
    # Enums
    DataQualityDimension,
    BiasType,
    DataGapType,
    DatasetRole,
    # Data structures
    QualityCheckResult,
    BiasCheckResult,
    DataGap,
    DataQualityReport,
    DatasetMetadata,
    # Configuration
    DataGovernanceConfig,
    # Main classes
    DataQualityAssessor,
    BiasDetector,
    DataGapAnalyzer,
    DataValidator,
    DataGovernanceFramework,
    # Factory functions
    create_data_governance_framework,
    create_bias_detector,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def governance_config(temp_log_dir):
    """Create a test governance configuration."""
    return DataGovernanceConfig(
        completeness_threshold=0.95,
        accuracy_threshold=0.95,
        consistency_threshold=0.99,
        timeliness_max_staleness_hours=24.0,
        temporal_bias_threshold=0.3,
        asset_bias_threshold=0.3,
        regime_bias_threshold=0.3,
        survivorship_bias_check_enabled=True,
        min_records_per_asset=100,  # Lower for testing
        min_time_coverage_days=30,  # Lower for testing
        log_path=temp_log_dir,
        log_all_assessments=True,
    )


@pytest.fixture
def sample_trading_df():
    """Create a sample trading DataFrame."""
    np.random.seed(42)
    n_records = 1000

    # Use recent dates to pass staleness checks
    from datetime import datetime, timedelta

    end_date = datetime.now() - timedelta(hours=1)
    start_date = end_date - timedelta(hours=n_records)
    dates = pd.date_range(start=start_date, periods=n_records, freq="1h")
    base_price = 50000

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": np.random.choice(["BTCUSDT", "ETHUSDT", "SOLUSDT"], n_records),
            "open": base_price + np.random.randn(n_records) * 100,
            "high": base_price + np.random.randn(n_records) * 100 + 50,
            "low": base_price + np.random.randn(n_records) * 100 - 50,
            "close": base_price + np.random.randn(n_records) * 100,
            "volume": np.abs(np.random.randn(n_records) * 1000 + 5000),
        }
    )

    # Ensure OHLC consistency
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def biased_df():
    """Create a DataFrame with intentional biases for testing."""
    np.random.seed(42)

    # Temporal bias: heavy in first month, sparse later
    dates_early = pd.date_range(start="2024-01-01", periods=800, freq="1h")
    dates_late = pd.date_range(start="2024-06-01", periods=200, freq="1h")
    dates = pd.concat([pd.Series(dates_early), pd.Series(dates_late)]).reset_index(drop=True)

    # Asset bias: BTCUSDT heavily over-represented
    symbols = ["BTCUSDT"] * 900 + ["ETHUSDT"] * 80 + ["SOLUSDT"] * 20

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": symbols,
            "open": 50000 + np.random.randn(1000) * 100,
            "high": 50050 + np.random.randn(1000) * 100,
            "low": 49950 + np.random.randn(1000) * 100,
            "close": 50000 + np.random.randn(1000) * 100,
            "volume": np.abs(np.random.randn(1000) * 1000 + 5000),
        }
    )

    return df


@pytest.fixture
def incomplete_df():
    """Create a DataFrame with missing values."""
    np.random.seed(42)
    n_records = 1000

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=n_records, freq="1h"),
            "symbol": np.random.choice(["BTCUSDT", "ETHUSDT"], n_records),
            "open": 50000 + np.random.randn(n_records) * 100,
            "high": 50050 + np.random.randn(n_records) * 100,
            "low": 49950 + np.random.randn(n_records) * 100,
            "close": 50000 + np.random.randn(n_records) * 100,
            "volume": np.abs(np.random.randn(n_records) * 1000 + 5000),
        }
    )

    # Introduce missing values (>10% to trigger gap detection)
    df.loc[np.random.choice(n_records, 150, replace=False), "close"] = np.nan
    df.loc[np.random.choice(n_records, 120, replace=False), "volume"] = np.nan

    return df


@pytest.fixture
def governance_framework(governance_config):
    """Create a test governance framework."""
    return DataGovernanceFramework(governance_config)


# =============================================================================
# Enum Tests
# =============================================================================


class TestDataQualityDimension:
    """Tests for DataQualityDimension enum."""

    def test_dimension_values(self):
        """Test all dimension values."""
        assert DataQualityDimension.COMPLETENESS.value == "completeness"
        assert DataQualityDimension.ACCURACY.value == "accuracy"
        assert DataQualityDimension.TIMELINESS.value == "timeliness"
        assert DataQualityDimension.CONSISTENCY.value == "consistency"
        assert DataQualityDimension.VALIDITY.value == "validity"
        assert DataQualityDimension.UNIQUENESS.value == "uniqueness"
        assert DataQualityDimension.REPRESENTATIVENESS.value == "representativeness"


class TestBiasType:
    """Tests for BiasType enum."""

    def test_bias_type_values(self):
        """Test all bias type values."""
        assert BiasType.TEMPORAL_BIAS.value == "temporal_bias"
        assert BiasType.SELECTION_BIAS.value == "selection_bias"
        assert BiasType.SURVIVORSHIP_BIAS.value == "survivorship_bias"
        assert BiasType.LOOK_AHEAD_BIAS.value == "look_ahead_bias"
        assert BiasType.ASSET_BIAS.value == "asset_bias"
        assert BiasType.REGIME_BIAS.value == "regime_bias"


class TestDataGapType:
    """Tests for DataGapType enum."""

    def test_gap_type_values(self):
        """Test all gap type values."""
        assert DataGapType.MISSING_TIME_PERIODS.value == "missing_time_periods"
        assert DataGapType.MISSING_FEATURES.value == "missing_features"
        assert DataGapType.INSUFFICIENT_VOLUME.value == "insufficient_volume"
        assert DataGapType.ASSET_COVERAGE_GAP.value == "asset_coverage_gap"


class TestDatasetRole:
    """Tests for DatasetRole enum."""

    def test_role_values(self):
        """Test all role values."""
        assert DatasetRole.TRAINING.value == "training"
        assert DatasetRole.VALIDATION.value == "validation"
        assert DatasetRole.TESTING.value == "testing"
        assert DatasetRole.PRODUCTION.value == "production"


# =============================================================================
# Data Structure Tests
# =============================================================================


class TestQualityCheckResult:
    """Tests for QualityCheckResult dataclass."""

    def test_create_result(self):
        """Test creating a quality check result."""
        result = QualityCheckResult(
            dimension="completeness",
            check_name="missing_values",
            passed=True,
            score=0.99,
            threshold=0.95,
        )

        assert result.check_id.startswith("QC-")
        assert result.dimension == "completeness"
        assert result.passed is True
        assert result.score == 0.99

    def test_result_auto_id(self):
        """Test result auto-generates ID."""
        result = QualityCheckResult()

        assert result.check_id.startswith("QC-")

    def test_result_defaults(self):
        """Test result default values."""
        result = QualityCheckResult()

        assert result.passed is True
        assert result.score == 1.0
        assert result.threshold == 0.9
        assert result.affected_columns == []


class TestBiasCheckResult:
    """Tests for BiasCheckResult dataclass."""

    def test_create_result(self):
        """Test creating a bias check result."""
        result = BiasCheckResult(
            bias_type="temporal_bias",
            detected=True,
            severity="medium",
            score=0.45,
        )

        assert result.check_id.startswith("BC-")
        assert result.detected is True
        assert result.severity == "medium"

    def test_result_with_suggestions(self):
        """Test result with mitigation suggestions."""
        result = BiasCheckResult(
            bias_type="asset_bias",
            detected=True,
            mitigation_suggestions=["Balance dataset", "Use weighted sampling"],
        )

        assert len(result.mitigation_suggestions) == 2


class TestDataGap:
    """Tests for DataGap dataclass."""

    def test_create_gap(self):
        """Test creating a data gap."""
        gap = DataGap(
            gap_type="missing_time_periods",
            description="Missing weekend data",
            severity="medium",
        )

        assert gap.gap_id.startswith("GAP-")
        assert gap.gap_type == "missing_time_periods"

    def test_gap_remediation(self):
        """Test gap with remediation plan."""
        gap = DataGap(
            gap_type="insufficient_volume",
            remediation_status="planned",
            remediation_plan="Collect more historical data",
        )

        assert gap.remediation_status == "planned"


class TestDataQualityReport:
    """Tests for DataQualityReport dataclass."""

    def test_create_report(self):
        """Test creating a quality report."""
        report = DataQualityReport(
            dataset_name="training_data",
            dataset_role="training",
        )

        assert report.report_id.startswith("DQR-")
        assert report.dataset_name == "training_data"

    def test_report_overall_score(self):
        """Test report with overall score."""
        report = DataQualityReport(
            overall_quality_score=0.95,
            passed=True,
        )

        assert report.overall_quality_score == 0.95
        assert report.passed is True


class TestDatasetMetadata:
    """Tests for DatasetMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating dataset metadata."""
        metadata = DatasetMetadata(
            name="training_data",
            role="training",
            record_count=10000,
            feature_count=50,
        )

        assert metadata.dataset_id.startswith("DS-")
        assert metadata.name == "training_data"
        assert metadata.record_count == 10000

    def test_metadata_versioning(self):
        """Test metadata versioning."""
        metadata = DatasetMetadata(
            name="test",
            version="2.0.0",
        )

        assert metadata.version == "2.0.0"
        assert metadata.created_at != ""


# =============================================================================
# DataGovernanceConfig Tests
# =============================================================================


class TestDataGovernanceConfig:
    """Tests for DataGovernanceConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataGovernanceConfig()

        assert config.completeness_threshold == 0.95
        assert config.accuracy_threshold == 0.95
        assert config.temporal_bias_threshold == 0.3
        assert config.min_records_per_asset == 1000
        assert config.min_time_coverage_days == 365

    def test_custom_config(self):
        """Test custom configuration."""
        config = DataGovernanceConfig(
            completeness_threshold=0.99,
            min_records_per_asset=500,
        )

        assert config.completeness_threshold == 0.99
        assert config.min_records_per_asset == 500


# =============================================================================
# DataQualityAssessor Tests
# =============================================================================


class TestDataQualityAssessor:
    """Tests for DataQualityAssessor class."""

    def test_assess_complete_data(self, governance_config, sample_trading_df):
        """Test assessing complete data."""
        assessor = DataQualityAssessor(governance_config)
        results = assessor.assess(sample_trading_df)

        assert len(results) > 0
        # All checks should pass for clean data
        passed_checks = [r for r in results if r.passed]
        assert len(passed_checks) == len(results)

    def test_assess_incomplete_data(self, governance_config, incomplete_df):
        """Test assessing incomplete data."""
        assessor = DataQualityAssessor(governance_config)
        results = assessor.assess(incomplete_df)

        # Should have completeness check failures
        completeness_results = [r for r in results if r.dimension == "completeness"]
        assert len(completeness_results) > 0

    def test_check_missing_values(self, governance_config, incomplete_df):
        """Test missing values check."""
        assessor = DataQualityAssessor(governance_config)
        result = assessor._check_missing_values(incomplete_df)

        assert result is not None
        assert result.dimension == "completeness"
        assert result.score < 1.0  # Should have some missing

    def test_check_ohlc_consistency(self, governance_config, sample_trading_df):
        """Test OHLC consistency check."""
        assessor = DataQualityAssessor(governance_config)
        result = assessor._check_ohlc_consistency(sample_trading_df)

        assert result is not None
        assert result.dimension == "consistency"
        assert result.passed is True

    def test_check_ohlc_inconsistency(self, governance_config):
        """Test OHLC inconsistency detection."""
        # Create inconsistent data
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [99, 100, 101],  # High < Open - invalid!
                "low": [98, 99, 100],
                "close": [99, 100, 101],
            }
        )

        assessor = DataQualityAssessor(governance_config)
        result = assessor._check_ohlc_consistency(df)

        assert result is not None
        assert not result.passed  # Use boolean comparison instead of 'is False'

    def test_check_outliers(self, governance_config, sample_trading_df):
        """Test outlier detection."""
        assessor = DataQualityAssessor(governance_config)
        result = assessor._check_outliers(sample_trading_df)

        assert result is not None
        assert result.dimension == "accuracy"

    def test_check_distribution(self, governance_config, sample_trading_df):
        """Test distribution check."""
        assessor = DataQualityAssessor(governance_config)
        result = assessor._check_distribution(sample_trading_df)

        assert result is not None
        assert result.dimension == "representativeness"


# =============================================================================
# BiasDetector Tests
# =============================================================================


class TestBiasDetector:
    """Tests for BiasDetector class."""

    def test_detect_biases(self, governance_config, sample_trading_df):
        """Test detecting biases in clean data."""
        detector = BiasDetector(governance_config)
        results = detector.detect_biases(sample_trading_df)

        assert isinstance(results, list)

    def test_detect_temporal_bias(self, governance_config, biased_df):
        """Test detecting temporal bias."""
        detector = BiasDetector(governance_config)
        result = detector._check_temporal_bias(biased_df, "timestamp")

        assert result is not None
        assert result.bias_type == "temporal_bias"
        # Biased data should show higher score
        assert result.score > 0

    def test_detect_asset_bias(self, governance_config, biased_df):
        """Test detecting asset bias."""
        detector = BiasDetector(governance_config)
        result = detector._check_asset_bias(biased_df, "symbol")

        assert result is not None
        assert result.bias_type == "asset_bias"
        # Heavy BTCUSDT bias should be detected
        assert result.detected  # Use boolean comparison instead of 'is True'

    def test_detect_regime_bias(self, governance_config, sample_trading_df):
        """Test detecting regime bias."""
        detector = BiasDetector(governance_config)
        result = detector._check_regime_bias(sample_trading_df)

        assert result is not None
        assert result.bias_type == "regime_bias"

    def test_check_look_ahead_bias(self, governance_config):
        """Test detecting look-ahead bias patterns."""
        # Create data with suspicious columns
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
                "close": np.random.randn(100),
                "future_return": np.random.randn(100),  # Suspicious!
                "next_close": np.random.randn(100),  # Suspicious!
            }
        )

        detector = BiasDetector(governance_config)
        result = detector._check_look_ahead_bias_structural(df)

        assert result is not None
        assert result.bias_type == "look_ahead_bias"
        assert result.detected is True
        assert len(result.affected_segments) > 0

    def test_check_survivorship_bias(self, governance_config, sample_trading_df):
        """Test survivorship bias check."""
        detector = BiasDetector(governance_config)
        result = detector._check_survivorship_bias(sample_trading_df, "symbol", "timestamp")

        assert result is not None
        assert result.bias_type == "survivorship_bias"


# =============================================================================
# DataGapAnalyzer Tests
# =============================================================================


class TestDataGapAnalyzer:
    """Tests for DataGapAnalyzer class."""

    def test_analyze_gaps(self, governance_config, sample_trading_df):
        """Test analyzing gaps in data."""
        analyzer = DataGapAnalyzer(governance_config)
        gaps = analyzer.analyze_gaps(sample_trading_df)

        assert isinstance(gaps, list)

    def test_check_time_period_gaps(self, governance_config):
        """Test checking for time period gaps."""
        # Create data with gaps
        dates1 = pd.date_range("2024-01-01", periods=100, freq="1h")
        dates2 = pd.date_range("2024-06-01", periods=100, freq="1h")
        dates = pd.concat([pd.Series(dates1), pd.Series(dates2)]).reset_index(drop=True)

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "close": np.random.randn(200),
            }
        )

        analyzer = DataGapAnalyzer(governance_config)
        gap = analyzer._check_time_period_gaps(df, "timestamp")

        # Should detect the large gap
        assert gap is not None

    def test_check_asset_coverage_gaps(self, governance_config):
        """Test checking for asset coverage gaps."""
        # Create data with under-represented asset
        df = pd.DataFrame(
            {
                "symbol": ["BTCUSDT"] * 500 + ["ETHUSDT"] * 10,  # ETHUSDT under-represented
                "close": np.random.randn(510),
            }
        )

        analyzer = DataGapAnalyzer(governance_config)
        gap = analyzer._check_asset_coverage_gaps(df, "symbol")

        assert gap is not None
        assert gap.gap_type == "asset_coverage_gap"

    def test_check_feature_gaps(self, governance_config, incomplete_df):
        """Test checking for feature gaps."""
        analyzer = DataGapAnalyzer(governance_config)
        gap = analyzer._check_feature_gaps(incomplete_df)

        # Should detect high missing rates
        assert gap is not None


# =============================================================================
# DataValidator Tests
# =============================================================================


class TestDataValidator:
    """Tests for DataValidator class."""

    def test_validate_valid_data(self, governance_config, sample_trading_df):
        """Test validating valid data."""
        validator = DataValidator(governance_config)
        is_valid, errors = validator.validate(sample_trading_df)

        assert is_valid  # Use boolean comparison instead of 'is True'
        assert len(errors) == 0

    def test_validate_empty_data(self, governance_config):
        """Test validating empty DataFrame."""
        df = pd.DataFrame()

        validator = DataValidator(governance_config)
        is_valid, errors = validator.validate(df)

        assert is_valid is False
        assert len(errors) > 0

    def test_validate_missing_columns(self, governance_config):
        """Test validating data with missing required columns."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
                "close": np.random.randn(10),
                # Missing: open, high, low, volume
            }
        )

        validator = DataValidator(governance_config)
        is_valid, errors = validator.validate(df)

        assert is_valid is False
        assert any("Missing" in e for e in errors)

    def test_validate_infinite_values(self, governance_config):
        """Test validating data with infinite values."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
                "open": [100] * 10,
                "high": [101] * 10,
                "low": [99] * 10,
                "close": [100] * 9 + [np.inf],  # Infinite value!
                "volume": [1000] * 10,
            }
        )

        validator = DataValidator(governance_config)
        is_valid, errors = validator.validate(df)

        assert is_valid is False
        assert any("Infinite" in e for e in errors)

    def test_add_custom_validator(self, governance_config, sample_trading_df):
        """Test adding custom validator."""
        validator = DataValidator(governance_config)

        def check_symbol_count(df):
            if df["symbol"].nunique() < 2:
                return False, "Need at least 2 symbols"
            return True, ""

        validator.add_validator(check_symbol_count)
        is_valid, errors = validator.validate(sample_trading_df)

        # Sample data has 3 symbols
        assert is_valid is True


# =============================================================================
# DataGovernanceFramework Tests
# =============================================================================


class TestDataGovernanceFramework:
    """Tests for DataGovernanceFramework class."""

    def test_framework_initialization(self, governance_framework):
        """Test framework initialization."""
        assert governance_framework is not None

    def test_validate(self, governance_framework, sample_trading_df):
        """Test validation method."""
        is_valid, errors = governance_framework.validate(sample_trading_df)

        assert is_valid is True

    def test_assess_data_quality(self, governance_framework, sample_trading_df):
        """Test comprehensive quality assessment."""
        report = governance_framework.assess_data_quality(
            sample_trading_df,
            dataset_name="test_data",
            role=DatasetRole.TRAINING,
        )

        assert report is not None
        assert report.dataset_name == "test_data"
        assert report.overall_quality_score > 0
        assert len(report.quality_checks) > 0

    def test_assess_data_quality_with_biases(self, governance_framework, biased_df):
        """Test quality assessment with biased data."""
        report = governance_framework.assess_data_quality(
            biased_df,
            dataset_name="biased_data",
            role=DatasetRole.TRAINING,
        )

        assert report.biases_detected > 0

    def test_detect_biases(self, governance_framework, sample_trading_df):
        """Test bias detection method."""
        biases = governance_framework.detect_biases(sample_trading_df)

        assert isinstance(biases, list)

    def test_analyze_gaps(self, governance_framework, sample_trading_df):
        """Test gap analysis method."""
        gaps = governance_framework.analyze_gaps(sample_trading_df)

        assert isinstance(gaps, list)

    def test_generate_metadata(self, governance_framework, sample_trading_df):
        """Test metadata generation."""
        metadata = governance_framework.generate_metadata(
            sample_trading_df,
            name="training_data",
            role=DatasetRole.TRAINING,
            sources=["binance"],
            preprocessing_steps=["winsorization", "normalization"],
        )

        assert metadata is not None
        assert metadata.name == "training_data"
        assert metadata.record_count == len(sample_trading_df)
        assert metadata.feature_count == len(sample_trading_df.columns)
        assert "binance" in metadata.sources

    def test_get_dataset_metadata(self, governance_framework, sample_trading_df):
        """Test retrieving stored metadata."""
        governance_framework.generate_metadata(
            sample_trading_df,
            name="test_data",
            role=DatasetRole.TRAINING,
        )

        metadata = governance_framework.get_dataset_metadata("test_data")

        assert metadata is not None
        assert metadata.name == "test_data"

    def test_get_all_datasets(self, governance_framework, sample_trading_df):
        """Test retrieving all datasets."""
        governance_framework.generate_metadata(
            sample_trading_df, name="data1", role=DatasetRole.TRAINING
        )
        governance_framework.generate_metadata(
            sample_trading_df, name="data2", role=DatasetRole.VALIDATION
        )

        all_datasets = governance_framework.get_all_datasets()

        assert len(all_datasets) >= 2
        assert "data1" in all_datasets
        assert "data2" in all_datasets

    def test_export_governance_report(self, governance_framework, sample_trading_df, temp_log_dir):
        """Test exporting governance report."""
        governance_framework.generate_metadata(
            sample_trading_df, name="test", role=DatasetRole.TRAINING
        )

        output_path = f"{temp_log_dir}/governance_report.json"
        governance_framework.export_governance_report(output_path)

        assert Path(output_path).exists()

        with open(output_path) as f:
            report = json.load(f)

        assert "export_date" in report
        assert "datasets" in report


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_data_governance_framework_default(self):
        """Test creating framework with defaults."""
        framework = create_data_governance_framework()

        assert framework is not None

    def test_create_data_governance_framework_with_config(self, governance_config):
        """Test creating framework with config object."""
        framework = create_data_governance_framework(governance_config)

        assert framework is not None

    def test_create_data_governance_framework_with_dict(self):
        """Test creating framework with dict config."""
        config = {
            "completeness_threshold": 0.99,
            "min_records_per_asset": 500,
        }
        framework = create_data_governance_framework(config)

        assert framework is not None
        assert framework.config.completeness_threshold == 0.99

    def test_create_bias_detector(self):
        """Test creating bias detector."""
        detector = create_bias_detector()

        assert detector is not None


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_assessments(self, governance_framework, sample_trading_df):
        """Test concurrent quality assessments."""
        results = []
        errors = []

        def run_assessment():
            try:
                report = governance_framework.assess_data_quality(
                    sample_trading_df,
                    dataset_name=f"test_{threading.current_thread().name}",
                    role=DatasetRole.TRAINING,
                )
                results.append(report)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(5):
            t = threading.Thread(target=run_assessment)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5


# =============================================================================
# Article 10 Compliance Tests
# =============================================================================


class TestArticle10Compliance:
    """Tests for Article 10 compliance requirements."""

    def test_data_quality_criteria(self, governance_framework, sample_trading_df):
        """Test Article 10(2)(3) - quality criteria assessment."""
        report = governance_framework.assess_data_quality(
            sample_trading_df,
            dataset_name="test",
            role=DatasetRole.TRAINING,
        )

        # Should assess completeness (free of errors)
        completeness_checks = [c for c in report.quality_checks if c.dimension == "completeness"]
        assert len(completeness_checks) > 0

        # Should assess accuracy
        accuracy_checks = [c for c in report.quality_checks if c.dimension == "accuracy"]
        assert len(accuracy_checks) > 0

        # Should assess representativeness
        representativeness_checks = [
            c for c in report.quality_checks if c.dimension == "representativeness"
        ]
        assert len(representativeness_checks) > 0

    def test_bias_examination(self, governance_framework, sample_trading_df):
        """Test Article 10(2)(e) - examination of possible biases."""
        report = governance_framework.assess_data_quality(
            sample_trading_df,
            dataset_name="test",
            role=DatasetRole.TRAINING,
        )

        # Should include bias checks
        assert len(report.bias_checks) > 0

        # Should check for temporal bias
        temporal_checks = [b for b in report.bias_checks if b.bias_type == "temporal_bias"]
        assert len(temporal_checks) > 0

    def test_data_gap_identification(self, governance_framework, sample_trading_df):
        """Test Article 10(2)(f) - identification of data gaps."""
        report = governance_framework.assess_data_quality(
            sample_trading_df,
            dataset_name="test",
            role=DatasetRole.TRAINING,
        )

        # Should analyze data gaps
        # (gaps list may be empty if data is complete)
        assert isinstance(report.data_gaps, list)

    def test_statistical_properties(self, governance_framework, sample_trading_df):
        """Test Article 10(4) - appropriate statistical properties."""
        report = governance_framework.assess_data_quality(
            sample_trading_df,
            dataset_name="test",
            role=DatasetRole.TRAINING,
        )

        # Should include statistical properties
        assert "statistical_properties" in report.__dict__
        assert report.statistical_properties is not None
        assert "record_count" in report.statistical_properties

    def test_preparation_process_documentation(self, governance_framework, sample_trading_df):
        """Test Article 10(2)(b) - preparation process documentation."""
        metadata = governance_framework.generate_metadata(
            sample_trading_df,
            name="test",
            role=DatasetRole.TRAINING,
            preprocessing_steps=["winsorization", "normalization", "feature_shift"],
        )

        # Should document preprocessing steps
        assert len(metadata.preprocessing_steps) == 3
        assert "winsorization" in metadata.preprocessing_steps

    def test_data_source_documentation(self, governance_framework, sample_trading_df):
        """Test Article 10(2)(a) - data collection documentation."""
        metadata = governance_framework.generate_metadata(
            sample_trading_df,
            name="test",
            role=DatasetRole.TRAINING,
            sources=["binance", "alpaca"],
            description="Training data for algorithmic trading model",
        )

        # Should document data sources
        assert len(metadata.sources) == 2
        assert metadata.description != ""


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self, governance_framework):
        """Test handling empty DataFrame."""
        df = pd.DataFrame()

        is_valid, errors = governance_framework.validate(df)

        assert is_valid is False

    def test_single_row_dataframe(self, governance_config):
        """Test handling single-row DataFrame."""
        df = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100],
                "volume": [1000],
                "symbol": ["BTCUSDT"],
            }
        )

        framework = DataGovernanceFramework(governance_config)
        # Should not crash
        report = framework.assess_data_quality(df, "test", DatasetRole.TESTING)

        assert report is not None

    def test_all_null_column(self, governance_config):
        """Test handling column with all null values."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
                "close": np.random.randn(100),
                "all_null": [np.nan] * 100,
            }
        )

        assessor = DataQualityAssessor(governance_config)
        results = assessor.assess(df)

        # Should handle gracefully
        assert len(results) > 0

    def test_non_numeric_ohlc(self, governance_config):
        """Test handling non-numeric OHLC columns."""
        df = pd.DataFrame(
            {
                "open": ["100", "101", "102"],  # Strings instead of numbers
                "high": ["101", "102", "103"],
                "low": ["99", "100", "101"],
                "close": ["100", "101", "102"],
            }
        )

        validator = DataValidator(governance_config)
        is_valid, errors = validator.validate(df)

        assert is_valid is False
        assert any("Non-numeric" in e for e in errors)
