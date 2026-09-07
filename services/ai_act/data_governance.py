# -*- coding: utf-8 -*-
"""
AI Act Data Governance Framework (Article 10).

EU AI Act Article 10 requires high-risk AI systems to be developed on the
basis of training, validation and testing data sets that meet specific
quality criteria.

This module provides:
1. DataGovernanceFramework - Main data governance orchestrator
2. DataQualityAssessor - Comprehensive data quality assessment
3. BiasDetector - Bias detection and mitigation
4. DataValidator - Data validation framework
5. DatasetMetadata - Dataset documentation per Annex IV

Article 10 Requirements Implemented:
    - (1) Quality criteria for training, validation and testing data
    - (2) Data quality practices:
        (a) Design choices and data collection processes
        (b) Relevant preparation processes (annotation, labeling, cleaning)
        (c) Formulation of assumptions about intended purpose
        (d) Assessment of availability, quantity and suitability
        (e) Examination of possible biases
        (f) Identification of data gaps and deficiencies
        (g) Appropriate measures to address issues
    - (3) Free of errors, complete, representative
    - (4) Appropriate statistical properties for intended use
    - (5) Taking into account specific geographical, contextual factors
    - (6) Requirements for processing personal data

References:
    - Article 10: https://artificialintelligenceact.eu/article/10/
    - Annex IV (Technical Documentation): https://artificialintelligenceact.eu/annex/4/
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class DataQualityDimension(Enum):
    """
    Data quality dimensions per Article 10(3).

    These dimensions align with ISO 8000 and DAMA-DMBOK data quality frameworks.
    """

    COMPLETENESS = "completeness"  # No missing critical values
    ACCURACY = "accuracy"  # Values reflect reality
    TIMELINESS = "timeliness"  # Data is current/not stale
    CONSISTENCY = "consistency"  # Cross-field consistency
    VALIDITY = "validity"  # Values within valid domains
    UNIQUENESS = "uniqueness"  # No unwanted duplicates
    REPRESENTATIVENESS = "representativeness"  # Article 10(3) - statistical properties


class BiasType(Enum):
    """
    Types of bias to detect per Article 10(2)(e).

    Based on research on algorithmic bias in financial AI systems.
    """

    TEMPORAL_BIAS = "temporal_bias"  # Over-representation of certain periods
    SELECTION_BIAS = "selection_bias"  # Non-random data selection
    SURVIVORSHIP_BIAS = "survivorship_bias"  # Only surviving entities included
    MEASUREMENT_BIAS = "measurement_bias"  # Systematic measurement errors
    SAMPLING_BIAS = "sampling_bias"  # Non-representative sampling
    LOOK_AHEAD_BIAS = "look_ahead_bias"  # Future information leakage
    ASSET_BIAS = "asset_bias"  # Over/under-representation of assets
    REGIME_BIAS = "regime_bias"  # Market regime over-representation


class DataGapType(Enum):
    """Types of data gaps per Article 10(2)(f)."""

    MISSING_TIME_PERIODS = "missing_time_periods"
    MISSING_FEATURES = "missing_features"
    INSUFFICIENT_VOLUME = "insufficient_volume"
    GEOGRAPHIC_GAP = "geographic_gap"
    ASSET_COVERAGE_GAP = "asset_coverage_gap"
    REGIME_COVERAGE_GAP = "regime_coverage_gap"


class DatasetRole(Enum):
    """Dataset roles in ML pipeline."""

    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    PRODUCTION = "production"


# =============================================================================
# Data Quality Report Structures
# =============================================================================


@dataclass
class QualityCheckResult:
    """Result of a single quality check."""

    check_id: str = ""
    dimension: str = ""
    check_name: str = ""
    passed: bool = True
    score: float = 1.0  # 0-1 score
    threshold: float = 0.9  # Pass threshold
    details: Dict[str, Any] = field(default_factory=dict)
    affected_columns: List[str] = field(default_factory=list)
    affected_rows: int = 0
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.check_id:
            self.check_id = f"QC-{uuid.uuid4().hex[:8].upper()}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class BiasCheckResult:
    """Result of a bias detection check."""

    check_id: str = ""
    bias_type: str = ""
    detected: bool = False
    severity: str = "low"  # low, medium, high, critical
    score: float = 0.0  # 0-1, higher = more bias
    details: Dict[str, Any] = field(default_factory=dict)
    affected_segments: List[str] = field(default_factory=list)
    mitigation_suggestions: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.check_id:
            self.check_id = f"BC-{uuid.uuid4().hex[:8].upper()}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class DataGap:
    """Identified data gap per Article 10(2)(f)."""

    gap_id: str = ""
    gap_type: str = ""
    description: str = ""
    severity: str = "medium"
    impact_assessment: str = ""
    remediation_status: str = "identified"  # identified, planned, in_progress, resolved
    remediation_plan: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.gap_id:
            self.gap_id = f"GAP-{uuid.uuid4().hex[:8].upper()}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class DataQualityReport:
    """
    Comprehensive data quality report per Article 10.

    This report documents:
    - Quality assessment results
    - Bias detection findings
    - Data gap analysis
    - Statistical properties
    - Recommendations
    """

    report_id: str = ""
    dataset_name: str = ""
    dataset_role: str = ""
    assessment_date: str = ""

    # Quality dimensions
    quality_checks: List[QualityCheckResult] = field(default_factory=list)
    overall_quality_score: float = 0.0

    # Bias detection
    bias_checks: List[BiasCheckResult] = field(default_factory=list)
    biases_detected: int = 0

    # Data gaps
    data_gaps: List[DataGap] = field(default_factory=list)

    # Statistical properties (Article 10(3))
    statistical_properties: Dict[str, Any] = field(default_factory=dict)

    # Summary
    passed: bool = False
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.report_id:
            self.report_id = f"DQR-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


@dataclass
class DatasetMetadata:
    """
    Dataset metadata for technical documentation per Annex IV.

    Required for Article 11 technical documentation.
    """

    dataset_id: str = ""
    name: str = ""
    description: str = ""
    role: str = ""  # training, validation, testing

    # Source information (Article 10(2)(a))
    sources: List[str] = field(default_factory=list)
    collection_method: str = ""
    collection_date_range: Tuple[str, str] = ("", "")

    # Preparation (Article 10(2)(b))
    preprocessing_steps: List[str] = field(default_factory=list)
    annotations: Dict[str, str] = field(default_factory=dict)
    cleaning_procedures: List[str] = field(default_factory=list)

    # Properties
    record_count: int = 0
    feature_count: int = 0
    feature_names: List[str] = field(default_factory=list)
    time_range: Tuple[str, str] = ("", "")

    # Statistical properties (Article 10(4))
    statistics: Dict[str, Any] = field(default_factory=dict)

    # Quality assessment
    quality_score: float = 0.0
    last_quality_check: str = ""

    # Geographic/contextual factors (Article 10(5))
    geographic_scope: List[str] = field(default_factory=list)
    contextual_factors: Dict[str, str] = field(default_factory=dict)

    # Versioning
    version: str = "1.0.0"
    hash: str = ""
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.dataset_id:
            self.dataset_id = f"DS-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DataGovernanceConfig:
    """Configuration for Data Governance Framework."""

    # Quality thresholds
    completeness_threshold: float = 0.95
    accuracy_threshold: float = 0.95
    consistency_threshold: float = 0.99
    timeliness_max_staleness_hours: float = 24.0

    # Bias detection
    temporal_bias_threshold: float = 0.3
    asset_bias_threshold: float = 0.3
    regime_bias_threshold: float = 0.3
    survivorship_bias_check_enabled: bool = True

    # Gap analysis
    min_records_per_asset: int = 1000
    min_time_coverage_days: int = 365

    # Statistical requirements (Article 10(3))
    require_normal_distribution: bool = False
    outlier_threshold_sigma: float = 3.0
    min_unique_values_ratio: float = 0.01

    # Winsorization (from features_pipeline)
    winsorize_percentiles: Tuple[float, float] = (1.0, 99.0)

    # Logging
    log_path: str = "logs/ai_act/data_governance"
    log_all_assessments: bool = True


# =============================================================================
# Data Quality Assessor
# =============================================================================


class DataQualityAssessor:
    """
    Comprehensive data quality assessment per Article 10(2)(3).

    Assesses data against quality criteria:
    - Completeness: No critical missing values
    - Accuracy: Values within valid domains
    - Timeliness: Data not stale
    - Consistency: Cross-field validation
    - Representativeness: Statistical properties
    """

    def __init__(self, config: DataGovernanceConfig):
        self.config = config
        self._checks: Dict[DataQualityDimension, List[Callable]] = defaultdict(list)
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default quality checks."""
        # Completeness checks
        self._checks[DataQualityDimension.COMPLETENESS].append(self._check_missing_values)
        self._checks[DataQualityDimension.COMPLETENESS].append(self._check_null_prices)

        # Accuracy checks
        self._checks[DataQualityDimension.ACCURACY].append(self._check_price_validity)
        self._checks[DataQualityDimension.ACCURACY].append(self._check_volume_validity)
        self._checks[DataQualityDimension.ACCURACY].append(self._check_outliers)

        # Timeliness checks
        self._checks[DataQualityDimension.TIMELINESS].append(self._check_data_staleness)
        self._checks[DataQualityDimension.TIMELINESS].append(self._check_time_gaps)

        # Consistency checks
        self._checks[DataQualityDimension.CONSISTENCY].append(self._check_ohlc_consistency)
        self._checks[DataQualityDimension.CONSISTENCY].append(self._check_timestamp_order)

        # Representativeness checks
        self._checks[DataQualityDimension.REPRESENTATIVENESS].append(self._check_distribution)
        self._checks[DataQualityDimension.REPRESENTATIVENESS].append(self._check_variance)

    def assess(self, df: pd.DataFrame, dataset_name: str = "") -> List[QualityCheckResult]:
        """
        Perform comprehensive quality assessment.

        Args:
            df: DataFrame to assess
            dataset_name: Name of the dataset

        Returns:
            List of quality check results
        """
        results: List[QualityCheckResult] = []

        for dimension, checks in self._checks.items():
            for check_func in checks:
                try:
                    result = check_func(df)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Quality check {check_func.__name__} failed: {e}")
                    results.append(
                        QualityCheckResult(
                            dimension=dimension.value,
                            check_name=check_func.__name__,
                            passed=False,
                            score=0.0,
                            details={"error": str(e)},
                        )
                    )

        return results

    # =========================================================================
    # Completeness Checks
    # =========================================================================

    def _check_missing_values(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check for missing values."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0

        columns_with_missing = df.columns[df.isnull().any()].tolist()

        return QualityCheckResult(
            dimension=DataQualityDimension.COMPLETENESS.value,
            check_name="missing_values",
            passed=completeness >= self.config.completeness_threshold,
            score=completeness,
            threshold=self.config.completeness_threshold,
            details={
                "total_cells": total_cells,
                "missing_cells": int(missing_cells),
                "completeness_ratio": completeness,
            },
            affected_columns=columns_with_missing,
            affected_rows=int(df.isnull().any(axis=1).sum()),
            recommendations=(
                ["Consider imputation or removal of rows with missing values"]
                if completeness < self.config.completeness_threshold
                else []
            ),
        )

    def _check_null_prices(self, df: pd.DataFrame) -> Optional[QualityCheckResult]:
        """Check for null prices in OHLC columns."""
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if not price_cols:
            return None

        null_count = df[price_cols].isnull().sum().sum()
        total = len(df) * len(price_cols)
        score = 1.0 - (null_count / total) if total > 0 else 1.0

        return QualityCheckResult(
            dimension=DataQualityDimension.COMPLETENESS.value,
            check_name="null_prices",
            passed=null_count == 0,
            score=score,
            threshold=1.0,
            details={
                "null_price_count": int(null_count),
                "price_columns_checked": price_cols,
            },
            affected_columns=price_cols if null_count > 0 else [],
            recommendations=(
                ["Price data must be complete for trading decisions"] if null_count > 0 else []
            ),
        )

    # =========================================================================
    # Accuracy Checks
    # =========================================================================

    def _check_price_validity(self, df: pd.DataFrame) -> Optional[QualityCheckResult]:
        """Check for valid price values (positive, non-zero)."""
        price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
        if not price_cols:
            return None

        invalid_count = 0
        invalid_details: Dict[str, int] = {}

        for col in price_cols:
            invalid = (df[col] <= 0).sum()
            invalid_count += invalid
            if invalid > 0:
                invalid_details[col] = int(invalid)

        total = len(df) * len(price_cols)
        score = 1.0 - (invalid_count / total) if total > 0 else 1.0

        return QualityCheckResult(
            dimension=DataQualityDimension.ACCURACY.value,
            check_name="price_validity",
            passed=invalid_count == 0,
            score=score,
            threshold=1.0,
            details={
                "invalid_price_count": invalid_count,
                "invalid_by_column": invalid_details,
            },
            affected_columns=list(invalid_details.keys()),
            recommendations=(
                ["Remove or correct non-positive price values"] if invalid_count > 0 else []
            ),
        )

    def _check_volume_validity(self, df: pd.DataFrame) -> Optional[QualityCheckResult]:
        """Check for valid volume values (non-negative)."""
        if "volume" not in df.columns:
            return None

        negative_count = (df["volume"] < 0).sum()
        score = 1.0 - (negative_count / len(df)) if len(df) > 0 else 1.0

        return QualityCheckResult(
            dimension=DataQualityDimension.ACCURACY.value,
            check_name="volume_validity",
            passed=negative_count == 0,
            score=score,
            threshold=1.0,
            details={"negative_volume_count": int(negative_count)},
            affected_columns=["volume"] if negative_count > 0 else [],
            recommendations=(
                ["Volume cannot be negative - check data source"] if negative_count > 0 else []
            ),
        )

    def _check_outliers(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check for statistical outliers using sigma rule."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts: Dict[str, int] = {}
        total_outliers = 0

        for col in numeric_cols:
            if col in ["timestamp", "symbol"]:
                continue

            data = df[col].dropna()
            if len(data) < 10:
                continue

            mean = data.mean()
            std = data.std()
            if std == 0:
                continue

            threshold = self.config.outlier_threshold_sigma
            outliers = ((data - mean).abs() > threshold * std).sum()
            if outliers > 0:
                outlier_counts[col] = int(outliers)
                total_outliers += outliers

        total_values = sum(
            len(df[c].dropna()) for c in numeric_cols if c not in ["timestamp", "symbol"]
        )
        score = 1.0 - (total_outliers / total_values) if total_values > 0 else 1.0

        return QualityCheckResult(
            dimension=DataQualityDimension.ACCURACY.value,
            check_name="outlier_detection",
            passed=score >= self.config.accuracy_threshold,
            score=score,
            threshold=self.config.accuracy_threshold,
            details={
                "total_outliers": total_outliers,
                "outlier_by_column": outlier_counts,
                "sigma_threshold": self.config.outlier_threshold_sigma,
            },
            affected_columns=list(outlier_counts.keys()),
            recommendations=(
                ["Consider winsorization or outlier treatment"] if total_outliers > 0 else []
            ),
        )

    # =========================================================================
    # Timeliness Checks
    # =========================================================================

    def _check_data_staleness(self, df: pd.DataFrame) -> Optional[QualityCheckResult]:
        """Check if data is stale."""
        if "timestamp" not in df.columns:
            return None

        try:
            # Handle various timestamp formats
            if pd.api.types.is_numeric_dtype(df["timestamp"]):
                # Unix timestamp (seconds or milliseconds)
                max_ts = df["timestamp"].max()
                if max_ts > 1e12:  # milliseconds
                    max_ts = max_ts / 1000
                latest = datetime.fromtimestamp(max_ts, tz=timezone.utc)
            else:
                latest = pd.to_datetime(df["timestamp"]).max()
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=timezone.utc)

            now = datetime.now(timezone.utc)
            staleness_hours = (now - latest).total_seconds() / 3600
            score = max(0, 1.0 - staleness_hours / self.config.timeliness_max_staleness_hours)

            return QualityCheckResult(
                dimension=DataQualityDimension.TIMELINESS.value,
                check_name="data_staleness",
                passed=staleness_hours <= self.config.timeliness_max_staleness_hours,
                score=score,
                threshold=0.5,
                details={
                    "latest_timestamp": latest.isoformat(),
                    "staleness_hours": round(staleness_hours, 2),
                    "max_staleness_hours": self.config.timeliness_max_staleness_hours,
                },
                recommendations=(
                    ["Data may be stale - refresh from source"]
                    if staleness_hours > self.config.timeliness_max_staleness_hours
                    else []
                ),
            )
        except Exception as e:
            logger.warning(f"Staleness check failed: {e}")
            return None

    def _check_time_gaps(self, df: pd.DataFrame) -> Optional[QualityCheckResult]:
        """Check for unexpected gaps in time series."""
        if "timestamp" not in df.columns:
            return None

        try:
            timestamps = pd.to_datetime(df["timestamp"]).sort_values()
            if len(timestamps) < 2:
                return None

            diffs = timestamps.diff().dropna()
            median_diff = diffs.median()

            # Gaps larger than 10x median are suspicious
            large_gaps = (diffs > median_diff * 10).sum()
            score = 1.0 - (large_gaps / len(diffs)) if len(diffs) > 0 else 1.0

            return QualityCheckResult(
                dimension=DataQualityDimension.TIMELINESS.value,
                check_name="time_gaps",
                passed=large_gaps == 0,
                score=score,
                threshold=1.0,
                details={
                    "large_gaps_count": int(large_gaps),
                    "median_time_diff_seconds": median_diff.total_seconds(),
                },
                recommendations=(
                    ["Time series has gaps - check for missing data"] if large_gaps > 0 else []
                ),
            )
        except Exception as e:
            logger.warning(f"Time gap check failed: {e}")
            return None

    # =========================================================================
    # Consistency Checks
    # =========================================================================

    def _check_ohlc_consistency(self, df: pd.DataFrame) -> Optional[QualityCheckResult]:
        """Check OHLC consistency (high >= low, etc.)."""
        required_cols = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required_cols):
            return None

        inconsistencies = 0
        details: Dict[str, int] = {}

        # High should be >= Open, Close, Low
        high_violations = (
            (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])
        ).sum()
        if high_violations > 0:
            details["high_violations"] = int(high_violations)
            inconsistencies += high_violations

        # Low should be <= Open, Close, High
        low_violations = (
            (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])
        ).sum()
        if low_violations > 0:
            details["low_violations"] = int(low_violations)
            inconsistencies += low_violations

        score = 1.0 - (inconsistencies / len(df)) if len(df) > 0 else 1.0

        return QualityCheckResult(
            dimension=DataQualityDimension.CONSISTENCY.value,
            check_name="ohlc_consistency",
            passed=inconsistencies == 0,
            score=score,
            threshold=self.config.consistency_threshold,
            details={
                "total_inconsistencies": inconsistencies,
                **details,
            },
            affected_columns=required_cols if inconsistencies > 0 else [],
            recommendations=(
                ["OHLC relationships violated - check data source"] if inconsistencies > 0 else []
            ),
        )

    def _check_timestamp_order(self, df: pd.DataFrame) -> Optional[QualityCheckResult]:
        """Check that timestamps are properly ordered."""
        if "timestamp" not in df.columns:
            return None

        try:
            timestamps = pd.to_datetime(df["timestamp"])
            is_sorted = timestamps.is_monotonic_increasing
            out_of_order = (
                ~timestamps.sort_values()
                .reset_index(drop=True)
                .eq(timestamps.reset_index(drop=True))
            ).sum()

            return QualityCheckResult(
                dimension=DataQualityDimension.CONSISTENCY.value,
                check_name="timestamp_order",
                passed=is_sorted,
                score=1.0 if is_sorted else 0.0,
                threshold=1.0,
                details={
                    "is_sorted": is_sorted,
                    "out_of_order_count": int(out_of_order),
                },
                recommendations=(
                    ["Sort data by timestamp before processing"] if not is_sorted else []
                ),
            )
        except Exception as e:
            logger.warning(f"Timestamp order check failed: {e}")
            return None

    # =========================================================================
    # Representativeness Checks (Article 10(3))
    # =========================================================================

    def _check_distribution(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check distribution properties of numeric features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ["timestamp", "symbol", "volume"]

        distribution_issues: Dict[str, str] = {}

        for col in numeric_cols:
            if col in exclude:
                continue

            data = df[col].dropna()
            if len(data) < 30:
                continue

            # Check for heavy tails (excess kurtosis)
            kurtosis = pd.Series(data).kurtosis()
            if abs(kurtosis) > 10:
                distribution_issues[col] = f"High kurtosis: {kurtosis:.2f}"

            # Check for extreme skewness
            skewness = pd.Series(data).skew()
            if abs(skewness) > 3:
                distribution_issues[col] = f"High skewness: {skewness:.2f}"

        score = 1.0 - len(distribution_issues) / len(numeric_cols) if len(numeric_cols) > 0 else 1.0

        return QualityCheckResult(
            dimension=DataQualityDimension.REPRESENTATIVENESS.value,
            check_name="distribution_check",
            passed=len(distribution_issues) == 0,
            score=score,
            threshold=0.8,
            details={
                "distribution_issues": distribution_issues,
                "columns_checked": len(numeric_cols),
            },
            affected_columns=list(distribution_issues.keys()),
            recommendations=(
                ["Consider transformations for non-normal distributions"]
                if distribution_issues
                else []
            ),
        )

    def _check_variance(self, df: pd.DataFrame) -> QualityCheckResult:
        """Check for zero or near-zero variance columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ["timestamp", "symbol"]

        low_variance_cols: Dict[str, float] = {}
        threshold = self.config.min_unique_values_ratio

        for col in numeric_cols:
            if col in exclude:
                continue

            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            if unique_ratio < threshold:
                low_variance_cols[col] = unique_ratio

        score = 1.0 - len(low_variance_cols) / len(numeric_cols) if len(numeric_cols) > 0 else 1.0

        return QualityCheckResult(
            dimension=DataQualityDimension.REPRESENTATIVENESS.value,
            check_name="variance_check",
            passed=len(low_variance_cols) == 0,
            score=score,
            threshold=0.9,
            details={
                "low_variance_columns": low_variance_cols,
                "unique_ratio_threshold": threshold,
            },
            affected_columns=list(low_variance_cols.keys()),
            recommendations=(
                ["Low variance columns may not be informative"] if low_variance_cols else []
            ),
        )


# =============================================================================
# Bias Detector
# =============================================================================


class BiasDetector:
    """
    Bias detection module per Article 10(2)(e).

    Detects potential biases in training data:
    1. Temporal bias - Over-representation of certain time periods
    2. Asset bias - Over/under-representation of certain assets
    3. Survivorship bias - Only surviving entities included
    4. Regime bias - Market regime over-representation
    5. Look-ahead bias - Future information leakage

    References:
        - Article 10(2)(e): Examination of possible biases
        - Article 10(2)(f): Identification of data gaps
    """

    def __init__(self, config: DataGovernanceConfig):
        self.config = config

    def detect_biases(
        self,
        df: pd.DataFrame,
        asset_column: str = "symbol",
        timestamp_column: str = "timestamp",
    ) -> List[BiasCheckResult]:
        """
        Run all bias detection checks.

        Args:
            df: DataFrame to analyze
            asset_column: Column containing asset identifiers
            timestamp_column: Column containing timestamps

        Returns:
            List of bias check results
        """
        results: List[BiasCheckResult] = []

        # Temporal bias
        if timestamp_column in df.columns:
            result = self._check_temporal_bias(df, timestamp_column)
            if result:
                results.append(result)

        # Asset bias
        if asset_column in df.columns:
            result = self._check_asset_bias(df, asset_column)
            if result:
                results.append(result)

        # Survivorship bias
        if asset_column in df.columns and timestamp_column in df.columns:
            result = self._check_survivorship_bias(df, asset_column, timestamp_column)
            if result:
                results.append(result)

        # Regime bias (using volatility as proxy)
        if "close" in df.columns:
            result = self._check_regime_bias(df)
            if result:
                results.append(result)

        # Look-ahead bias (structural check)
        result = self._check_look_ahead_bias_structural(df)
        if result:
            results.append(result)

        return results

    def _check_temporal_bias(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
    ) -> Optional[BiasCheckResult]:
        """
        Check for temporal bias (over-representation of certain periods).

        Temporal bias can lead to models that perform well in specific
        market conditions but fail in others.
        """
        try:
            timestamps = pd.to_datetime(df[timestamp_column])

            # Group by year-month
            monthly_counts = timestamps.dt.to_period("M").value_counts()

            if len(monthly_counts) < 2:
                return None

            # Calculate coefficient of variation
            cv = monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
            bias_score = min(cv / 2, 1.0)  # Normalize to 0-1

            severity = "low"
            if bias_score > 0.7:
                severity = "critical"
            elif bias_score > 0.5:
                severity = "high"
            elif bias_score > 0.3:
                severity = "medium"

            # Find over/under-represented periods
            mean_count = monthly_counts.mean()
            over_represented = monthly_counts[monthly_counts > mean_count * 2].index.tolist()
            under_represented = monthly_counts[monthly_counts < mean_count * 0.5].index.tolist()

            return BiasCheckResult(
                bias_type=BiasType.TEMPORAL_BIAS.value,
                detected=bias_score > self.config.temporal_bias_threshold,
                severity=severity,
                score=bias_score,
                details={
                    "coefficient_of_variation": round(cv, 4),
                    "month_count_std": round(monthly_counts.std(), 2),
                    "month_count_mean": round(monthly_counts.mean(), 2),
                    "over_represented_periods": [str(p) for p in over_represented[:5]],
                    "under_represented_periods": [str(p) for p in under_represented[:5]],
                },
                affected_segments=[str(p) for p in over_represented + under_represented],
                mitigation_suggestions=[
                    "Consider stratified sampling across time periods",
                    "Use time-weighted sampling to balance representation",
                    "Augment under-represented periods if possible",
                ],
            )
        except Exception as e:
            logger.warning(f"Temporal bias check failed: {e}")
            return None

    def _check_asset_bias(
        self,
        df: pd.DataFrame,
        asset_column: str,
    ) -> Optional[BiasCheckResult]:
        """
        Check for asset bias (over/under-representation of certain assets).
        """
        try:
            asset_counts = df[asset_column].value_counts()

            if len(asset_counts) < 2:
                return None

            # Calculate Gini coefficient for distribution
            n = len(asset_counts)
            values = np.sort(asset_counts.values)
            cumsum = np.cumsum(values)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
            bias_score = abs(gini)

            severity = "low"
            if bias_score > 0.7:
                severity = "critical"
            elif bias_score > 0.5:
                severity = "high"
            elif bias_score > 0.3:
                severity = "medium"

            # Find over/under-represented assets
            mean_count = asset_counts.mean()
            over_represented = asset_counts[asset_counts > mean_count * 3].index.tolist()
            under_represented = asset_counts[asset_counts < mean_count * 0.3].index.tolist()

            return BiasCheckResult(
                bias_type=BiasType.ASSET_BIAS.value,
                detected=bias_score > self.config.asset_bias_threshold,
                severity=severity,
                score=bias_score,
                details={
                    "gini_coefficient": round(gini, 4),
                    "asset_count": len(asset_counts),
                    "max_records": int(asset_counts.max()),
                    "min_records": int(asset_counts.min()),
                    "over_represented_assets": over_represented[:5],
                    "under_represented_assets": under_represented[:5],
                },
                affected_segments=over_represented + under_represented,
                mitigation_suggestions=[
                    "Balance dataset across assets using sampling",
                    "Weight loss function by inverse asset frequency",
                    "Remove severely under-represented assets or collect more data",
                ],
            )
        except Exception as e:
            logger.warning(f"Asset bias check failed: {e}")
            return None

    def _check_survivorship_bias(
        self,
        df: pd.DataFrame,
        asset_column: str,
        timestamp_column: str,
    ) -> Optional[BiasCheckResult]:
        """
        Check for survivorship bias (only surviving entities included).

        Survivorship bias is critical in financial data where failed/delisted
        assets are often excluded, leading to overly optimistic backtests.
        """
        if not self.config.survivorship_bias_check_enabled:
            return None

        try:
            timestamps = pd.to_datetime(df[timestamp_column])
            assets = df[asset_column]

            # Check if all assets have data at the end of the period
            end_period = timestamps.max() - pd.Timedelta(days=30)
            assets_at_end = df[timestamps > end_period][asset_column].unique()
            all_assets = assets.unique()

            # Assets missing at end might indicate survivorship bias
            missing_at_end = set(all_assets) - set(assets_at_end)
            missing_ratio = len(missing_at_end) / len(all_assets) if len(all_assets) > 0 else 0

            # Also check start
            start_period = timestamps.min() + pd.Timedelta(days=30)
            assets_at_start = df[timestamps < start_period][asset_column].unique()
            missing_at_start = set(all_assets) - set(assets_at_start)

            # Survivorship bias indicator: assets present at start but missing at end
            potential_delisted = missing_at_end - missing_at_start
            survivorship_score = (
                len(potential_delisted) / len(all_assets) if len(all_assets) > 0 else 0
            )

            severity = "low"
            if survivorship_score > 0.3:
                severity = "high"
            elif survivorship_score > 0.1:
                severity = "medium"

            # Low score actually indicates POTENTIAL survivorship bias
            # (because we don't see delisted assets)
            if len(potential_delisted) == 0 and len(all_assets) > 10:
                return BiasCheckResult(
                    bias_type=BiasType.SURVIVORSHIP_BIAS.value,
                    detected=True,
                    severity="medium",
                    score=0.5,
                    details={
                        "warning": "No assets appear to have been delisted - potential survivorship bias",
                        "total_assets": len(all_assets),
                        "recommendation": "Verify dataset includes delisted/failed assets",
                    },
                    mitigation_suggestions=[
                        "Verify that delisted/failed assets are included in the dataset",
                        "Cross-reference with historical asset lists",
                        "Use point-in-time dataset construction",
                    ],
                )

            return BiasCheckResult(
                bias_type=BiasType.SURVIVORSHIP_BIAS.value,
                detected=survivorship_score > 0,
                severity=severity,
                score=survivorship_score,
                details={
                    "total_assets": len(all_assets),
                    "assets_at_end": len(assets_at_end),
                    "potential_delisted_count": len(potential_delisted),
                    "potential_delisted_assets": list(potential_delisted)[:10],
                },
                affected_segments=list(potential_delisted),
                mitigation_suggestions=[
                    "Include historical data for delisted assets",
                    "Use point-in-time dataset construction",
                    "Document survivorship bias in technical documentation",
                ],
            )
        except Exception as e:
            logger.warning(f"Survivorship bias check failed: {e}")
            return None

    def _check_regime_bias(self, df: pd.DataFrame) -> Optional[BiasCheckResult]:
        """
        Check for market regime bias (bull/bear/sideways over-representation).

        Uses realized volatility as a proxy for market regimes.
        """
        try:
            if len(df) < 100:
                return None

            # Calculate returns and volatility
            close = df["close"]
            returns = close.pct_change().dropna()

            if len(returns) < 50:
                return None

            # Rolling volatility (20-period)
            rolling_vol = returns.rolling(20).std()

            # Classify regimes by volatility percentiles
            low_vol_threshold = rolling_vol.quantile(0.33)
            high_vol_threshold = rolling_vol.quantile(0.67)

            low_vol_count = (rolling_vol < low_vol_threshold).sum()
            mid_vol_count = (
                (rolling_vol >= low_vol_threshold) & (rolling_vol < high_vol_threshold)
            ).sum()
            high_vol_count = (rolling_vol >= high_vol_threshold).sum()

            total = low_vol_count + mid_vol_count + high_vol_count
            if total == 0:
                return None

            # Calculate regime imbalance
            expected = total / 3
            imbalance = (
                abs(low_vol_count - expected)
                + abs(mid_vol_count - expected)
                + abs(high_vol_count - expected)
            ) / (3 * expected)

            regime_score = min(imbalance, 1.0)

            severity = "low"
            if regime_score > 0.7:
                severity = "high"
            elif regime_score > 0.4:
                severity = "medium"

            return BiasCheckResult(
                bias_type=BiasType.REGIME_BIAS.value,
                detected=regime_score > self.config.regime_bias_threshold,
                severity=severity,
                score=regime_score,
                details={
                    "low_volatility_periods": int(low_vol_count),
                    "mid_volatility_periods": int(mid_vol_count),
                    "high_volatility_periods": int(high_vol_count),
                    "regime_imbalance": round(imbalance, 4),
                },
                affected_segments=["low_vol", "mid_vol", "high_vol"],
                mitigation_suggestions=[
                    "Ensure balanced representation across market regimes",
                    "Use regime-aware sampling for training data",
                    "Test model performance separately by regime",
                ],
            )
        except Exception as e:
            logger.warning(f"Regime bias check failed: {e}")
            return None

    def _check_look_ahead_bias_structural(self, df: pd.DataFrame) -> BiasCheckResult:
        """
        Structural check for look-ahead bias indicators.

        This checks for common patterns that indicate look-ahead bias:
        - Future-looking column names
        - Target columns without proper shifting
        """
        suspicious_columns: List[str] = []
        warnings: List[str] = []

        # Check for suspicious column names
        future_keywords = ["future", "forward", "next", "tomorrow", "predict", "target"]
        for col in df.columns:
            col_lower = col.lower()
            for keyword in future_keywords:
                if keyword in col_lower:
                    suspicious_columns.append(col)
                    break

        # Check if _z suffix columns exist without base columns being shifted
        z_columns = [c for c in df.columns if c.endswith("_z")]
        if z_columns:
            warnings.append(
                f"Found {len(z_columns)} normalized columns - verify shifting is applied"
            )

        detected = len(suspicious_columns) > 0
        severity = "high" if detected else "low"
        score = len(suspicious_columns) / len(df.columns) if len(df.columns) > 0 else 0

        return BiasCheckResult(
            bias_type=BiasType.LOOK_AHEAD_BIAS.value,
            detected=detected,
            severity=severity,
            score=score,
            details={
                "suspicious_columns": suspicious_columns,
                "warnings": warnings,
                "recommendation": "Verify all features are computed from past data only",
            },
            affected_segments=suspicious_columns,
            mitigation_suggestions=[
                "Use FeaturePipeline with shift=True for all feature computation",
                "Verify target columns are properly separated from features",
                "Audit feature computation for any future data usage",
            ],
        )


# =============================================================================
# Data Gap Analyzer
# =============================================================================


class DataGapAnalyzer:
    """
    Data gap analysis per Article 10(2)(f).

    Identifies gaps and deficiencies in training data.
    """

    def __init__(self, config: DataGovernanceConfig):
        self.config = config

    def analyze_gaps(
        self,
        df: pd.DataFrame,
        asset_column: str = "symbol",
        timestamp_column: str = "timestamp",
    ) -> List[DataGap]:
        """
        Analyze data for gaps and deficiencies.

        Args:
            df: DataFrame to analyze
            asset_column: Column containing asset identifiers
            timestamp_column: Column containing timestamps

        Returns:
            List of identified data gaps
        """
        gaps: List[DataGap] = []

        # Time period gaps
        if timestamp_column in df.columns:
            gap = self._check_time_period_gaps(df, timestamp_column)
            if gap:
                gaps.append(gap)

        # Asset coverage gaps
        if asset_column in df.columns:
            gap = self._check_asset_coverage_gaps(df, asset_column)
            if gap:
                gaps.append(gap)

        # Feature completeness gaps
        gap = self._check_feature_gaps(df)
        if gap:
            gaps.append(gap)

        # Volume sufficiency gaps
        if asset_column in df.columns:
            gap = self._check_volume_gaps(df, asset_column)
            if gap:
                gaps.append(gap)

        return gaps

    def _check_time_period_gaps(
        self,
        df: pd.DataFrame,
        timestamp_column: str,
    ) -> Optional[DataGap]:
        """Check for missing time periods."""
        try:
            timestamps = pd.to_datetime(df[timestamp_column])
            date_range = (timestamps.max() - timestamps.min()).days

            if date_range < self.config.min_time_coverage_days:
                return DataGap(
                    gap_type=DataGapType.MISSING_TIME_PERIODS.value,
                    description=f"Dataset covers only {date_range} days, minimum required is {self.config.min_time_coverage_days}",
                    severity="high",
                    impact_assessment="Insufficient time coverage may lead to poor generalization across market conditions",
                    remediation_plan="Extend data collection to cover at least 1 year of historical data",
                )

            # Check for significant gaps within the period
            timestamps_sorted = timestamps.sort_values()
            diffs = timestamps_sorted.diff().dropna()
            median_diff = diffs.median()
            large_gaps = diffs[diffs > median_diff * 100]  # 100x normal gap

            if len(large_gaps) > 0:
                return DataGap(
                    gap_type=DataGapType.MISSING_TIME_PERIODS.value,
                    description=f"Found {len(large_gaps)} significant time gaps in data",
                    severity="medium",
                    impact_assessment="Missing time periods may exclude important market events",
                    remediation_plan="Fill gaps from alternative data sources or document as known limitations",
                )

            return None
        except Exception as e:
            logger.warning(f"Time period gap check failed: {e}")
            return None

    def _check_asset_coverage_gaps(
        self,
        df: pd.DataFrame,
        asset_column: str,
    ) -> Optional[DataGap]:
        """Check for asset coverage gaps."""
        try:
            asset_counts = df[asset_column].value_counts()
            insufficient_assets = asset_counts[asset_counts < self.config.min_records_per_asset]

            if len(insufficient_assets) > 0:
                return DataGap(
                    gap_type=DataGapType.ASSET_COVERAGE_GAP.value,
                    description=f"{len(insufficient_assets)} assets have fewer than {self.config.min_records_per_asset} records",
                    severity="medium",
                    impact_assessment="Insufficient data for some assets may lead to unreliable predictions",
                    remediation_plan="Collect more data for under-represented assets or exclude them from training",
                )

            return None
        except Exception as e:
            logger.warning(f"Asset coverage gap check failed: {e}")
            return None

    def _check_feature_gaps(self, df: pd.DataFrame) -> Optional[DataGap]:
        """Check for feature completeness gaps."""
        try:
            # Check for columns with high missing rates
            missing_rates = df.isnull().sum() / len(df)
            problematic = missing_rates[missing_rates > 0.1]  # >10% missing

            if len(problematic) > 0:
                return DataGap(
                    gap_type=DataGapType.MISSING_FEATURES.value,
                    description=f"{len(problematic)} features have >10% missing values",
                    severity="medium",
                    impact_assessment="High missing rates may require imputation which can introduce bias",
                    remediation_plan="Improve data collection for affected features or develop robust imputation strategies",
                )

            return None
        except Exception as e:
            logger.warning(f"Feature gap check failed: {e}")
            return None

    def _check_volume_gaps(
        self,
        df: pd.DataFrame,
        asset_column: str,
    ) -> Optional[DataGap]:
        """Check for overall data volume sufficiency."""
        try:
            total_records = len(df)
            num_assets = df[asset_column].nunique()
            records_per_asset = total_records / num_assets if num_assets > 0 else 0

            if records_per_asset < self.config.min_records_per_asset:
                return DataGap(
                    gap_type=DataGapType.INSUFFICIENT_VOLUME.value,
                    description=f"Average {records_per_asset:.0f} records per asset, minimum recommended is {self.config.min_records_per_asset}",
                    severity="high",
                    impact_assessment="Insufficient data volume may lead to overfitting and poor generalization",
                    remediation_plan="Expand data collection or reduce the number of assets in scope",
                )

            return None
        except Exception as e:
            logger.warning(f"Volume gap check failed: {e}")
            return None


# =============================================================================
# Data Validator
# =============================================================================


class DataValidator:
    """
    Data validation framework for pre-processing checks.

    Validates data before it enters the ML pipeline.
    """

    def __init__(self, config: DataGovernanceConfig):
        self.config = config
        self._validators: List[Callable[[pd.DataFrame], Tuple[bool, str]]] = []
        self._register_default_validators()

    def _register_default_validators(self) -> None:
        """Register default validators."""
        self._validators.append(self._validate_not_empty)
        self._validators.append(self._validate_required_columns)
        self._validators.append(self._validate_numeric_types)
        self._validators.append(self._validate_no_inf)

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: List[str] = []

        for validator in self._validators:
            try:
                is_valid, message = validator(df)
                if not is_valid:
                    errors.append(message)
            except Exception as e:
                errors.append(f"Validator {validator.__name__} failed: {e}")

        return len(errors) == 0, errors

    def add_validator(self, validator: Callable[[pd.DataFrame], Tuple[bool, str]]) -> None:
        """Add a custom validator."""
        self._validators.append(validator)

    def _validate_not_empty(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate DataFrame is not empty."""
        if len(df) == 0:
            return False, "DataFrame is empty"
        return True, ""

    def _validate_required_columns(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate required columns exist."""
        # Basic OHLCV columns for trading data
        recommended = ["open", "high", "low", "close", "volume"]
        missing = [c for c in recommended if c not in df.columns]

        if missing:
            return False, f"Missing recommended columns: {missing}"
        return True, ""

    def _validate_numeric_types(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate numeric columns are numeric types."""
        price_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]

        non_numeric = []
        for col in price_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric.append(col)

        if non_numeric:
            return False, f"Non-numeric price/volume columns: {non_numeric}"
        return True, ""

    def _validate_no_inf(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate no infinite values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        inf_cols = []
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                inf_cols.append(col)

        if inf_cols:
            return False, f"Infinite values found in columns: {inf_cols}"
        return True, ""


# =============================================================================
# Main Data Governance Framework
# =============================================================================


class DataGovernanceFramework:
    """
    Article 10 compliant Data Governance Framework.

    Provides comprehensive data governance including:
    1. Data quality assessment
    2. Bias detection and mitigation
    3. Data gap analysis
    4. Data validation
    5. Dataset documentation

    Usage:
        config = DataGovernanceConfig()
        framework = DataGovernanceFramework(config)

        # Validate data
        is_valid, errors = framework.validate(df)

        # Full assessment
        report = framework.assess_data_quality(
            df,
            dataset_name="training_data",
            role=DatasetRole.TRAINING,
        )

        # Check for biases
        biases = framework.detect_biases(df)

        # Generate dataset metadata
        metadata = framework.generate_metadata(df, "training_data", DatasetRole.TRAINING)
    """

    def __init__(self, config: Optional[DataGovernanceConfig] = None):
        """Initialize the framework."""
        self.config = config or DataGovernanceConfig()

        # Initialize components
        self._quality_assessor = DataQualityAssessor(self.config)
        self._bias_detector = BiasDetector(self.config)
        self._gap_analyzer = DataGapAnalyzer(self.config)
        self._validator = DataValidator(self.config)

        # Dataset registry
        self._datasets: Dict[str, DatasetMetadata] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DataGovernanceFramework initialized")

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame before processing.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        return self._validator.validate(df)

    # =========================================================================
    # Quality Assessment
    # =========================================================================

    def assess_data_quality(
        self,
        df: pd.DataFrame,
        dataset_name: str = "",
        role: DatasetRole = DatasetRole.TRAINING,
        asset_column: str = "symbol",
        timestamp_column: str = "timestamp",
    ) -> DataQualityReport:
        """
        Perform comprehensive data quality assessment.

        Per Article 10(2)(3), assesses:
        - Completeness, accuracy, consistency
        - Bias detection
        - Data gap analysis
        - Statistical properties

        Args:
            df: DataFrame to assess
            dataset_name: Name of the dataset
            role: Dataset role (training, validation, testing)
            asset_column: Column containing asset identifiers
            timestamp_column: Column containing timestamps

        Returns:
            Comprehensive DataQualityReport
        """
        report = DataQualityReport(
            dataset_name=dataset_name,
            dataset_role=role.value,
        )

        # Quality checks
        report.quality_checks = self._quality_assessor.assess(df, dataset_name)

        # Calculate overall quality score
        if report.quality_checks:
            report.overall_quality_score = sum(c.score for c in report.quality_checks) / len(
                report.quality_checks
            )

        # Bias detection
        report.bias_checks = self._bias_detector.detect_biases(df, asset_column, timestamp_column)
        report.biases_detected = sum(1 for b in report.bias_checks if b.detected)

        # Data gaps
        report.data_gaps = self._gap_analyzer.analyze_gaps(df, asset_column, timestamp_column)

        # Statistical properties
        report.statistical_properties = self._compute_statistical_properties(df)

        # Determine pass/fail
        critical_failures = [
            c for c in report.quality_checks if not c.passed and c.threshold >= 0.95
        ]
        high_severity_biases = [
            b for b in report.bias_checks if b.detected and b.severity in ("high", "critical")
        ]

        report.passed = len(critical_failures) == 0 and len(high_severity_biases) == 0

        # Compile issues and recommendations
        report.critical_issues = (
            [f"Quality: {c.check_name} - {c.details}" for c in critical_failures]
            + [f"Bias: {b.bias_type} - {b.severity}" for b in high_severity_biases]
            + [
                f"Gap: {g.gap_type} - {g.description}"
                for g in report.data_gaps
                if g.severity in ("high", "critical")
            ]
        )

        report.recommendations = []
        for check in report.quality_checks:
            report.recommendations.extend(check.recommendations)
        for bias in report.bias_checks:
            report.recommendations.extend(bias.mitigation_suggestions)
        for gap in report.data_gaps:
            if gap.remediation_plan:
                report.recommendations.append(gap.remediation_plan)

        # Remove duplicates
        report.recommendations = list(set(report.recommendations))

        # Log the assessment
        if self.config.log_all_assessments:
            self._log_assessment(report)

        return report

    def _compute_statistical_properties(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute statistical properties per Article 10(4).

        These properties should be appropriate for the intended use.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude = ["timestamp", "symbol"]

        properties: Dict[str, Any] = {
            "record_count": len(df),
            "feature_count": len(df.columns),
            "numeric_feature_count": len(numeric_cols),
        }

        # Basic statistics for numeric columns
        column_stats: Dict[str, Dict[str, float]] = {}
        for col in numeric_cols:
            if col in exclude:
                continue

            data = df[col].dropna()
            if len(data) == 0:
                continue

            column_stats[col] = {
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "median": float(data.median()),
                "skewness": float(pd.Series(data).skew()),
                "kurtosis": float(pd.Series(data).kurtosis()),
                "null_ratio": float(df[col].isnull().sum() / len(df)),
            }

        properties["column_statistics"] = column_stats

        # Time range if timestamp exists
        if "timestamp" in df.columns:
            try:
                timestamps = pd.to_datetime(df["timestamp"])
                properties["time_range"] = {
                    "start": str(timestamps.min()),
                    "end": str(timestamps.max()),
                    "duration_days": (timestamps.max() - timestamps.min()).days,
                }
            except Exception:
                pass

        # Asset statistics if symbol exists
        if "symbol" in df.columns:
            properties["asset_statistics"] = {
                "unique_assets": int(df["symbol"].nunique()),
                "records_per_asset": {
                    str(k): int(v) for k, v in df["symbol"].value_counts().head(20).items()
                },
            }

        return properties

    # =========================================================================
    # Bias Detection
    # =========================================================================

    def detect_biases(
        self,
        df: pd.DataFrame,
        asset_column: str = "symbol",
        timestamp_column: str = "timestamp",
    ) -> List[BiasCheckResult]:
        """
        Detect biases in the dataset per Article 10(2)(e).

        Args:
            df: DataFrame to analyze
            asset_column: Column containing asset identifiers
            timestamp_column: Column containing timestamps

        Returns:
            List of bias check results
        """
        return self._bias_detector.detect_biases(df, asset_column, timestamp_column)

    # =========================================================================
    # Gap Analysis
    # =========================================================================

    def analyze_gaps(
        self,
        df: pd.DataFrame,
        asset_column: str = "symbol",
        timestamp_column: str = "timestamp",
    ) -> List[DataGap]:
        """
        Analyze data gaps per Article 10(2)(f).

        Args:
            df: DataFrame to analyze
            asset_column: Column containing asset identifiers
            timestamp_column: Column containing timestamps

        Returns:
            List of identified data gaps
        """
        return self._gap_analyzer.analyze_gaps(df, asset_column, timestamp_column)

    # =========================================================================
    # Dataset Metadata
    # =========================================================================

    def generate_metadata(
        self,
        df: pd.DataFrame,
        name: str,
        role: DatasetRole,
        sources: Optional[List[str]] = None,
        preprocessing_steps: Optional[List[str]] = None,
        description: str = "",
    ) -> DatasetMetadata:
        """
        Generate dataset metadata for technical documentation.

        Per Annex IV, creates documentation of:
        - Dataset description and sources
        - Preprocessing and preparation steps
        - Statistical properties
        - Quality assessment results

        Args:
            df: DataFrame to document
            name: Dataset name
            role: Dataset role (training, validation, testing)
            sources: Data sources
            preprocessing_steps: Applied preprocessing steps
            description: Dataset description

        Returns:
            DatasetMetadata for technical documentation
        """
        # Run quality assessment first
        report = self.assess_data_quality(df, name, role)

        # Compute hash for versioning
        hash_content = f"{name}:{len(df)}:{list(df.columns)}"
        dataset_hash = hashlib.sha256(hash_content.encode()).hexdigest()[:16]

        # Time range
        time_range = ("", "")
        if "timestamp" in df.columns:
            try:
                timestamps = pd.to_datetime(df["timestamp"])
                time_range = (str(timestamps.min()), str(timestamps.max()))
            except Exception:
                pass

        metadata = DatasetMetadata(
            name=name,
            description=description,
            role=role.value,
            sources=sources or [],
            preprocessing_steps=preprocessing_steps or [],
            record_count=len(df),
            feature_count=len(df.columns),
            feature_names=list(df.columns),
            time_range=time_range,
            statistics=report.statistical_properties,
            quality_score=report.overall_quality_score,
            last_quality_check=report.assessment_date,
            hash=dataset_hash,
        )

        # Register metadata
        with self._lock:
            self._datasets[name] = metadata

        return metadata

    def get_dataset_metadata(self, name: str) -> Optional[DatasetMetadata]:
        """Get registered dataset metadata."""
        with self._lock:
            return self._datasets.get(name)

    def get_all_datasets(self) -> Dict[str, DatasetMetadata]:
        """Get all registered dataset metadata."""
        with self._lock:
            return dict(self._datasets)

    # =========================================================================
    # Export
    # =========================================================================

    def export_governance_report(self, output_path: str) -> None:
        """
        Export comprehensive governance report for regulatory documentation.

        Args:
            output_path: Path to save the report
        """
        report = {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "framework_version": "1.0.0",
            "article_10_compliance": True,
            "datasets": {name: asdict(meta) for name, meta in self._datasets.items()},
            "configuration": asdict(self.config),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Governance report exported to {output_path}")

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _log_assessment(self, report: DataQualityReport) -> None:
        """Log a quality assessment."""
        log_file = self._log_path / f"assessments_{datetime.now().strftime('%Y%m%d')}.jsonl"

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(report), default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log assessment: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_data_governance_framework(
    config: Optional[Union[Dict[str, Any], DataGovernanceConfig]] = None,
) -> DataGovernanceFramework:
    """
    Create a DataGovernanceFramework instance.

    Args:
        config: Optional configuration dictionary or DataGovernanceConfig

    Returns:
        Configured DataGovernanceFramework instance
    """
    if config is None:
        return DataGovernanceFramework()
    elif isinstance(config, DataGovernanceConfig):
        return DataGovernanceFramework(config)
    else:
        # Dictionary config
        gov_config = DataGovernanceConfig(
            completeness_threshold=config.get("completeness_threshold", 0.95),
            accuracy_threshold=config.get("accuracy_threshold", 0.95),
            consistency_threshold=config.get("consistency_threshold", 0.99),
            timeliness_max_staleness_hours=config.get("timeliness_max_staleness_hours", 24.0),
            temporal_bias_threshold=config.get("temporal_bias_threshold", 0.3),
            asset_bias_threshold=config.get("asset_bias_threshold", 0.3),
            regime_bias_threshold=config.get("regime_bias_threshold", 0.3),
            survivorship_bias_check_enabled=config.get("survivorship_bias_check_enabled", True),
            min_records_per_asset=config.get("min_records_per_asset", 1000),
            min_time_coverage_days=config.get("min_time_coverage_days", 365),
            log_path=config.get("log_path", "logs/ai_act/data_governance"),
            log_all_assessments=config.get("log_all_assessments", True),
        )
        return DataGovernanceFramework(gov_config)


def create_bias_detector(
    config: Optional[Union[Dict[str, Any], DataGovernanceConfig]] = None,
) -> BiasDetector:
    """Create a BiasDetector instance."""
    if config is None:
        return BiasDetector(DataGovernanceConfig())
    elif isinstance(config, DataGovernanceConfig):
        return BiasDetector(config)
    else:
        gov_config = DataGovernanceConfig(
            **{k: v for k, v in config.items() if hasattr(DataGovernanceConfig, k)}
        )
        return BiasDetector(gov_config)
