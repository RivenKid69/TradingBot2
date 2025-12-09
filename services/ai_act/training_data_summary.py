# -*- coding: utf-8 -*-
"""
Article 53(1)(d) EU AI Act - Training Data Summary.

This module implements the training data summary requirements for GPAI models
as mandated by Article 53(1)(d) of the EU AI Act.

Key Requirements:
- Draw up and make publicly available a sufficiently detailed summary
- Content used for training the general-purpose AI model
- Summary according to template provided by AI Office
- Public disclosure for transparency

References:
    - EU AI Act Article 53(1)(d): https://artificialintelligenceact.eu/article/53/
    - Annex XII: Training Data Summary Template
    - GPAI Code of Practice - Training Data Transparency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
import json


class DataCategory(Enum):
    """
    Categories of training data per Annex XII template.

    These categories help structure the training data summary
    in compliance with Article 53(1)(d).
    """
    MARKET_DATA = "market_data"  # Price, volume, order book
    TECHNICAL_INDICATORS = "technical_indicators"  # Computed features
    FUNDAMENTAL_DATA = "fundamental_data"  # Company financials
    ALTERNATIVE_DATA = "alternative_data"  # Non-traditional sources
    SYNTHETIC_DATA = "synthetic_data"  # Generated data
    METADATA = "metadata"  # Data about data


class DataQualityLevel(Enum):
    """Quality assessment level for datasets."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNVERIFIED = "unverified"


@dataclass
class DatasetInfo:
    """
    Detailed information about a training dataset.

    Per Article 53(1)(d), each dataset used in training must be
    documented with sufficient detail.

    Attributes:
        name: Human-readable dataset name
        category: Category per Annex XII
        description: Detailed description of the dataset
        time_range_start: Start of temporal coverage
        time_range_end: End of temporal coverage
        size_rows: Number of data points/samples
        size_gb: Size in gigabytes
        assets_covered: List of assets/instruments covered
        update_frequency: How often data is updated
        source_provider: Data provider name
        preprocessing: List of preprocessing steps applied
        quality_level: Quality assessment
        personal_data_included: Whether PII is included
        geographic_coverage: Geographic scope
    """
    name: str
    category: DataCategory
    description: str
    time_range_start: datetime
    time_range_end: datetime
    size_rows: int
    size_gb: float
    assets_covered: List[str]
    update_frequency: str
    source_provider: str
    preprocessing: List[str]
    quality_level: DataQualityLevel = DataQualityLevel.HIGH
    personal_data_included: bool = False
    geographic_coverage: str = "Global"
    data_format: str = "Tabular"
    sampling_methodology: str = "Complete historical coverage"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "time_range_start": self.time_range_start.isoformat(),
            "time_range_end": self.time_range_end.isoformat(),
            "size_rows": self.size_rows,
            "size_gb": self.size_gb,
            "assets_covered": self.assets_covered,
            "update_frequency": self.update_frequency,
            "source_provider": self.source_provider,
            "preprocessing": self.preprocessing,
            "quality_level": self.quality_level.value,
            "personal_data_included": self.personal_data_included,
            "geographic_coverage": self.geographic_coverage,
            "data_format": self.data_format,
            "sampling_methodology": self.sampling_methodology,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            category=DataCategory(data["category"]),
            description=data["description"],
            time_range_start=datetime.fromisoformat(data["time_range_start"]),
            time_range_end=datetime.fromisoformat(data["time_range_end"]),
            size_rows=data["size_rows"],
            size_gb=data["size_gb"],
            assets_covered=data["assets_covered"],
            update_frequency=data["update_frequency"],
            source_provider=data["source_provider"],
            preprocessing=data["preprocessing"],
            quality_level=DataQualityLevel(data.get("quality_level", "high")),
            personal_data_included=data.get("personal_data_included", False),
            geographic_coverage=data.get("geographic_coverage", "Global"),
            data_format=data.get("data_format", "Tabular"),
            sampling_methodology=data.get("sampling_methodology", "Complete historical coverage"),
        )


@dataclass
class TrainingDataSummary:
    """
    Article 53(1)(d) compliant training data summary.

    Per EU AI Act: "draw up and make publicly available a
    sufficiently detailed summary about the content used
    for training the general-purpose AI model"

    This class represents the complete training data summary
    that must be published for GPAI compliance.

    Attributes:
        model_name: Name of the AI model
        model_version: Version of the model
        summary_date: Date of summary generation
        datasets: List of datasets used in training
        total_training_samples: Total number of training samples
        total_data_size_gb: Total data size in GB
        training_period_start: Start of training period
        training_period_end: End of training period
        data_quality_measures: List of quality measures applied
        bias_mitigation_steps: List of bias mitigation steps
    """
    model_name: str
    model_version: str
    summary_date: datetime
    datasets: List[DatasetInfo]
    total_training_samples: int
    total_data_size_gb: float
    training_period_start: datetime
    training_period_end: datetime
    data_quality_measures: List[str]
    bias_mitigation_steps: List[str]
    personal_data_statement: str = "No personal data is used for training."
    copyright_compliance_reference: str = "See COPYRIGHT_POLICY.md"
    data_collection_methodology: str = "Automated collection via licensed APIs"
    labeling_methodology: str = "N/A - Reinforcement learning from rewards"

    def generate_public_summary(self) -> str:
        """
        Generate public summary document per Article 53(1)(d).

        Returns:
            Markdown formatted summary document
        """
        datasets_text = "\n".join([
            f"- **{d.name}**: {d.description} "
            f"({d.size_rows:,} samples, {d.time_range_start.year}-{d.time_range_end.year})"
            for d in self.datasets
        ])

        category_counts = {}
        for d in self.datasets:
            cat = d.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        category_table = "\n".join([
            f"| {cat.replace('_', ' ').title()} | {count} dataset(s) |"
            for cat, count in category_counts.items()
        ])

        return f"""# Training Data Summary

**Model**: {self.model_name}
**Version**: {self.model_version}
**Summary Date**: {self.summary_date.strftime("%Y-%m-%d")}
**Regulation Reference**: EU AI Act Article 53(1)(d)

---

## 1. Overview

| Metric | Value |
|--------|-------|
| Total Training Samples | {self.total_training_samples:,} |
| Total Data Size | {self.total_data_size_gb:.1f} GB |
| Training Period | {self.training_period_start.strftime("%Y-%m")} to {self.training_period_end.strftime("%Y-%m")} |
| Number of Datasets | {len(self.datasets)} |

## 2. Data Categories

| Category | Count |
|----------|-------|
{category_table}

## 3. Datasets Used

{datasets_text}

## 4. Data Quality Measures

{chr(10).join(f"- {m}" for m in self.data_quality_measures)}

## 5. Bias Mitigation

{chr(10).join(f"- {s}" for s in self.bias_mitigation_steps)}

## 6. Data Collection Methodology

{self.data_collection_methodology}

## 7. Labeling Methodology

{self.labeling_methodology}

## 8. Personal Data Statement

{self.personal_data_statement}

## 9. Copyright Compliance

{self.copyright_compliance_reference}

---

## Detailed Dataset Information

"""
        # Add detailed dataset tables
        result = f"""# Training Data Summary

**Model**: {self.model_name}
**Version**: {self.model_version}
**Summary Date**: {self.summary_date.strftime("%Y-%m-%d")}
**Regulation Reference**: EU AI Act Article 53(1)(d)

---

## 1. Overview

| Metric | Value |
|--------|-------|
| Total Training Samples | {self.total_training_samples:,} |
| Total Data Size | {self.total_data_size_gb:.1f} GB |
| Training Period | {self.training_period_start.strftime("%Y-%m")} to {self.training_period_end.strftime("%Y-%m")} |
| Number of Datasets | {len(self.datasets)} |

## 2. Data Categories

| Category | Count |
|----------|-------|
{category_table}

## 3. Datasets Used

{datasets_text}

## 4. Data Quality Measures

{chr(10).join(f"- {m}" for m in self.data_quality_measures)}

## 5. Bias Mitigation

{chr(10).join(f"- {s}" for s in self.bias_mitigation_steps)}

## 6. Data Collection Methodology

{self.data_collection_methodology}

## 7. Labeling Methodology

{self.labeling_methodology}

## 8. Personal Data Statement

{self.personal_data_statement}

## 9. Copyright Compliance

{self.copyright_compliance_reference}

---

## Appendix: Detailed Dataset Information

"""

        for i, dataset in enumerate(self.datasets, 1):
            result += f"""
### A{i}. {dataset.name}

| Property | Value |
|----------|-------|
| Category | {dataset.category.value.replace('_', ' ').title()} |
| Provider | {dataset.source_provider} |
| Time Range | {dataset.time_range_start.strftime("%Y-%m-%d")} to {dataset.time_range_end.strftime("%Y-%m-%d")} |
| Size | {dataset.size_rows:,} rows ({dataset.size_gb:.1f} GB) |
| Geographic Coverage | {dataset.geographic_coverage} |
| Update Frequency | {dataset.update_frequency} |
| Data Format | {dataset.data_format} |
| Quality Level | {dataset.quality_level.value.title()} |
| Personal Data | {"Yes" if dataset.personal_data_included else "No"} |

**Description**: {dataset.description}

**Assets Covered**: {', '.join(dataset.assets_covered)}

**Preprocessing Steps**:
{chr(10).join(f"- {p}" for p in dataset.preprocessing)}

"""

        result += """
---

*This summary is provided in accordance with Article 53(1)(d) of the EU AI Act
(Regulation (EU) 2024/1689). For questions regarding this summary, please
contact: compliance@[company].com*
"""

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "summary_date": self.summary_date.isoformat(),
            "datasets": [d.to_dict() for d in self.datasets],
            "total_training_samples": self.total_training_samples,
            "total_data_size_gb": self.total_data_size_gb,
            "training_period_start": self.training_period_start.isoformat(),
            "training_period_end": self.training_period_end.isoformat(),
            "data_quality_measures": self.data_quality_measures,
            "bias_mitigation_steps": self.bias_mitigation_steps,
            "personal_data_statement": self.personal_data_statement,
            "copyright_compliance_reference": self.copyright_compliance_reference,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingDataSummary":
        """Create from dictionary."""
        return cls(
            model_name=data["model_name"],
            model_version=data["model_version"],
            summary_date=datetime.fromisoformat(data["summary_date"]),
            datasets=[DatasetInfo.from_dict(d) for d in data["datasets"]],
            total_training_samples=data["total_training_samples"],
            total_data_size_gb=data["total_data_size_gb"],
            training_period_start=datetime.fromisoformat(data["training_period_start"]),
            training_period_end=datetime.fromisoformat(data["training_period_end"]),
            data_quality_measures=data["data_quality_measures"],
            bias_mitigation_steps=data["bias_mitigation_steps"],
            personal_data_statement=data.get(
                "personal_data_statement",
                "No personal data is used for training."
            ),
            copyright_compliance_reference=data.get(
                "copyright_compliance_reference",
                "See COPYRIGHT_POLICY.md"
            ),
        )


def create_default_summary() -> TrainingDataSummary:
    """
    Create default training data summary for the platform.

    Returns:
        TrainingDataSummary with platform's default datasets
    """
    datasets = [
        DatasetInfo(
            name="Binance Spot OHLCV",
            category=DataCategory.MARKET_DATA,
            description="Cryptocurrency spot market OHLCV (Open, High, Low, Close, Volume) data for major trading pairs",
            time_range_start=datetime(2017, 1, 1),
            time_range_end=datetime(2024, 12, 1),
            size_rows=50_000_000,
            size_gb=15.0,
            assets_covered=["BTC/USDT", "ETH/USDT", "BNB/USDT", "Major altcoins"],
            update_frequency="1-minute bars",
            source_provider="Binance API",
            preprocessing=[
                "Outlier detection and removal (>5 sigma)",
                "Gap filling using forward-fill (max 5 bars)",
                "Volume normalization per asset",
                "Z-score normalization for model input"
            ],
            quality_level=DataQualityLevel.HIGH,
            personal_data_included=False,
            geographic_coverage="Global",
            data_format="Tabular CSV/Parquet",
        ),
        DatasetInfo(
            name="US Equity Data",
            category=DataCategory.MARKET_DATA,
            description="US stock market data including OHLCV, trades, and quotes via Polygon.io",
            time_range_start=datetime(2010, 1, 1),
            time_range_end=datetime(2024, 12, 1),
            size_rows=100_000_000,
            size_gb=25.0,
            assets_covered=["S&P 500 constituents", "Russell 2000 constituents"],
            update_frequency="1-minute bars",
            source_provider="Polygon.io",
            preprocessing=[
                "Corporate actions adjustment (splits, dividends)",
                "Exchange code normalization",
                "Timestamp alignment to market hours",
                "Duplicate removal"
            ],
            quality_level=DataQualityLevel.HIGH,
            personal_data_included=False,
            geographic_coverage="United States",
            data_format="Tabular CSV/Parquet",
        ),
        DatasetInfo(
            name="Forex Major Pairs",
            category=DataCategory.MARKET_DATA,
            description="Foreign exchange data for major currency pairs",
            time_range_start=datetime(2010, 1, 1),
            time_range_end=datetime(2024, 12, 1),
            size_rows=30_000_000,
            size_gb=8.0,
            assets_covered=["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"],
            update_frequency="1-minute bars",
            source_provider="OANDA / Alpha Vantage",
            preprocessing=[
                "Weekend gap handling",
                "Spread validation",
                "Tick-to-bar aggregation"
            ],
            quality_level=DataQualityLevel.HIGH,
            personal_data_included=False,
            geographic_coverage="Global",
        ),
        DatasetInfo(
            name="Technical Indicators",
            category=DataCategory.TECHNICAL_INDICATORS,
            description="Computed technical analysis features derived from price and volume data",
            time_range_start=datetime(2010, 1, 1),
            time_range_end=datetime(2024, 12, 1),
            size_rows=150_000_000,
            size_gb=10.0,
            assets_covered=["All traded assets"],
            update_frequency="Computed from OHLCV",
            source_provider="Internal computation",
            preprocessing=[
                "Rolling window calculation",
                "Z-score normalization",
                "Winsorization at 1st/99th percentile",
                "NaN handling (forward-fill then backfill)"
            ],
            quality_level=DataQualityLevel.HIGH,
            personal_data_included=False,
            geographic_coverage="Global",
            data_format="Tabular feature matrices",
        ),
        DatasetInfo(
            name="Adversarial Scenarios",
            category=DataCategory.SYNTHETIC_DATA,
            description="Synthetically generated adversarial market scenarios for model robustness training (SA-PPO)",
            time_range_start=datetime(2024, 1, 1),
            time_range_end=datetime(2024, 12, 1),
            size_rows=10_000_000,
            size_gb=2.0,
            assets_covered=["Simulated assets (no direct real-world mapping)"],
            update_frequency="Generated per training run",
            source_provider="Internal generation (SA-PPO adversarial framework)",
            preprocessing=[
                "Scenario validation against statistical bounds",
                "Distribution alignment with real market moments",
                "Extreme event calibration"
            ],
            quality_level=DataQualityLevel.HIGH,
            personal_data_included=False,
            geographic_coverage="N/A (synthetic)",
            data_format="Numpy arrays",
            sampling_methodology="Adversarial generation via learned perturbation policy"
        ),
    ]

    return TrainingDataSummary(
        model_name="Distributional PPO Trading Model",
        model_version="4.0",
        summary_date=datetime.utcnow(),
        datasets=datasets,
        total_training_samples=340_000_000,
        total_data_size_gb=60.0,
        training_period_start=datetime(2010, 1, 1),
        training_period_end=datetime(2024, 12, 1),
        data_quality_measures=[
            "Automated outlier detection and removal (statistical and ML-based)",
            "Data completeness checks (>99.5% required for inclusion)",
            "Temporal consistency validation (no look-ahead bias)",
            "Cross-source reconciliation for overlapping data",
            "Feature distribution monitoring for data drift",
            "Unit tests for data pipeline integrity",
            "Manual review of edge cases and anomalies"
        ],
        bias_mitigation_steps=[
            "Temporal sampling across market regimes (bull/bear/sideways)",
            "Asset class balancing to prevent over-representation",
            "Survivorship bias correction (include delisted/bankrupt assets)",
            "Look-ahead bias prevention in feature engineering (point-in-time data)",
            "Selection bias mitigation via stratified sampling",
            "Regular bias audits using statistical tests (KS, Chi-square)",
            "Adversarial testing across demographic and market segments"
        ],
        personal_data_statement=(
            "No personal data is used for training. All training data consists of:\n"
            "- Aggregated market statistics\n"
            "- Price and volume information\n"
            "- Computed technical indicators\n"
            "- Synthetically generated scenarios\n\n"
            "The model does not process, store, or learn from any personally "
            "identifiable information (PII)."
        ),
        copyright_compliance_reference=(
            "For detailed copyright compliance information, see COPYRIGHT_POLICY.md.\n"
            "All data sources are either:\n"
            "- Licensed under commercial agreements\n"
            "- Public market data (not subject to copyright)\n"
            "- Internally generated (synthetic)\n\n"
            "Compliance with Article 53(1)(c) EU AI Act is documented separately."
        ),
        data_collection_methodology=(
            "Data is collected via:\n"
            "1. Licensed API connections to market data providers\n"
            "2. Direct exchange feeds where available\n"
            "3. Internal computation for derived features\n"
            "4. Algorithmic generation for synthetic scenarios\n\n"
            "All data collection respects rate limits, terms of service, "
            "and applicable data protection regulations."
        ),
        labeling_methodology=(
            "This model uses reinforcement learning (RL) and does not require "
            "traditional labeled data. The learning signal comes from:\n"
            "1. Trading rewards (P&L, risk-adjusted returns)\n"
            "2. Risk penalties (drawdown, volatility)\n"
            "3. Constraint satisfaction (position limits, exposure)\n\n"
            "No human labeling or annotation is performed."
        )
    )


class TrainingDataSummaryManager:
    """
    Manager for training data summaries.

    Handles creation, updates, and retrieval of Article 53(1)(d)
    compliant training data summaries.

    Example:
        >>> manager = create_summary_manager()
        >>> summary = manager.get_public_summary()
        >>> print(summary[:100])
        # Training Data Summary...
    """

    def __init__(self):
        """Initialize the summary manager with default summary."""
        self.current_summary = create_default_summary()
        self._history: List[TrainingDataSummary] = []

    def get_public_summary(self) -> str:
        """
        Get public summary document.

        Returns:
            Markdown formatted summary document
        """
        return self.current_summary.generate_public_summary()

    def get_summary_metadata(self) -> Dict[str, Any]:
        """
        Get summary metadata for API.

        Returns:
            Dictionary with summary metadata
        """
        return {
            "model_name": self.current_summary.model_name,
            "model_version": self.current_summary.model_version,
            "summary_date": self.current_summary.summary_date.isoformat(),
            "total_samples": self.current_summary.total_training_samples,
            "total_size_gb": self.current_summary.total_data_size_gb,
            "datasets_count": len(self.current_summary.datasets),
            "training_period": {
                "start": self.current_summary.training_period_start.isoformat(),
                "end": self.current_summary.training_period_end.isoformat(),
            },
            "article_reference": "EU AI Act Article 53(1)(d)"
        }

    def get_datasets(self) -> List[Dict[str, Any]]:
        """
        Get list of datasets.

        Returns:
            List of dataset dictionaries
        """
        return [d.to_dict() for d in self.current_summary.datasets]

    def get_dataset_by_name(self, name: str) -> Optional[DatasetInfo]:
        """
        Get a specific dataset by name.

        Args:
            name: Dataset name to find

        Returns:
            DatasetInfo if found, None otherwise
        """
        for dataset in self.current_summary.datasets:
            if dataset.name == name:
                return dataset
        return None

    def get_datasets_by_category(
        self,
        category: DataCategory
    ) -> List[DatasetInfo]:
        """
        Get datasets by category.

        Args:
            category: Category to filter by

        Returns:
            List of matching datasets
        """
        return [
            d for d in self.current_summary.datasets
            if d.category == category
        ]

    def add_dataset(self, dataset: DatasetInfo) -> None:
        """
        Add a new dataset to the summary.

        Args:
            dataset: Dataset to add
        """
        self.current_summary.datasets.append(dataset)
        self._update_totals()

    def remove_dataset(self, name: str) -> bool:
        """
        Remove a dataset from the summary.

        Args:
            name: Name of dataset to remove

        Returns:
            True if removed, False if not found
        """
        for i, dataset in enumerate(self.current_summary.datasets):
            if dataset.name == name:
                del self.current_summary.datasets[i]
                self._update_totals()
                return True
        return False

    def update_summary(self, summary: TrainingDataSummary) -> None:
        """
        Update the current summary.

        Args:
            summary: New summary to use
        """
        # Save current to history
        self._history.append(self.current_summary)
        self.current_summary = summary

    def get_summary_history(self) -> List[Dict[str, Any]]:
        """
        Get history of summaries.

        Returns:
            List of historical summary metadata
        """
        return [
            {
                "model_version": s.model_version,
                "summary_date": s.summary_date.isoformat(),
                "datasets_count": len(s.datasets),
            }
            for s in self._history
        ]

    def validate_summary(self) -> Dict[str, Any]:
        """
        Validate current summary for Article 53(1)(d) compliance.

        Returns:
            Dictionary with validation results
        """
        summary = self.current_summary

        checks = {
            "has_model_name": bool(summary.model_name),
            "has_model_version": bool(summary.model_version),
            "has_datasets": len(summary.datasets) > 0,
            "has_quality_measures": len(summary.data_quality_measures) > 0,
            "has_bias_mitigation": len(summary.bias_mitigation_steps) > 0,
            "has_personal_data_statement": bool(summary.personal_data_statement),
            "has_copyright_reference": bool(summary.copyright_compliance_reference),
            "all_datasets_valid": all(
                self._validate_dataset(d) for d in summary.datasets
            ),
        }

        checks["all_valid"] = all(checks.values())

        return {
            "compliant": checks["all_valid"],
            "checks": checks,
            "validation_date": datetime.utcnow().isoformat(),
            "article_reference": "EU AI Act Article 53(1)(d)"
        }

    def _validate_dataset(self, dataset: DatasetInfo) -> bool:
        """Validate a single dataset."""
        return all([
            bool(dataset.name),
            bool(dataset.description),
            dataset.size_rows > 0,
            bool(dataset.source_provider),
            len(dataset.preprocessing) > 0,
        ])

    def _update_totals(self) -> None:
        """Update total counts after dataset changes."""
        self.current_summary.total_training_samples = sum(
            d.size_rows for d in self.current_summary.datasets
        )
        self.current_summary.total_data_size_gb = sum(
            d.size_gb for d in self.current_summary.datasets
        )
        self.current_summary.summary_date = datetime.utcnow()

    def export_json(self) -> str:
        """
        Export summary as JSON.

        Returns:
            JSON string representation
        """
        return json.dumps(self.current_summary.to_dict(), indent=2)

    def import_json(self, json_str: str) -> None:
        """
        Import summary from JSON.

        Args:
            json_str: JSON string to import
        """
        data = json.loads(json_str)
        self.update_summary(TrainingDataSummary.from_dict(data))


def create_summary_manager() -> TrainingDataSummaryManager:
    """
    Factory function to create TrainingDataSummaryManager.

    Returns:
        Configured TrainingDataSummaryManager instance
    """
    return TrainingDataSummaryManager()


def get_data_categories() -> List[str]:
    """
    Get list of available data categories.

    Returns:
        List of category values
    """
    return [c.value for c in DataCategory]


def validate_dataset_info(dataset: DatasetInfo) -> Dict[str, bool]:
    """
    Validate a dataset info record.

    Args:
        dataset: DatasetInfo to validate

    Returns:
        Dictionary with validation results
    """
    checks = {
        "has_name": bool(dataset.name),
        "has_description": bool(dataset.description),
        "has_provider": bool(dataset.source_provider),
        "has_valid_size": dataset.size_rows > 0 and dataset.size_gb > 0,
        "has_preprocessing": len(dataset.preprocessing) > 0,
        "has_time_range": (
            dataset.time_range_start is not None and
            dataset.time_range_end is not None and
            dataset.time_range_start < dataset.time_range_end
        ),
        "has_assets": len(dataset.assets_covered) > 0,
    }

    checks["all_valid"] = all(checks.values())

    return checks
