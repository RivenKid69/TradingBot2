# -*- coding: utf-8 -*-
"""
Tests for Article 53(1)(d) EU AI Act - Training Data Summary.

This module provides comprehensive tests for the training data summary
functionality required by Article 53(1)(d) of the EU AI Act.

Coverage includes:
- DatasetInfo management
- TrainingDataSummary generation
- TrainingDataSummaryManager operations
- Public summary document generation
- Compliance validation
"""

import pytest
import json
from datetime import datetime
from typing import Dict, Any, List

from services.ai_act.training_data_summary import (
    # Enums
    DataCategory,
    DataQualityLevel,
    # Data structures
    DatasetInfo,
    TrainingDataSummary,
    # Main class
    TrainingDataSummaryManager,
    # Factory functions
    create_default_summary,
    create_summary_manager,
    get_data_categories,
    validate_dataset_info,
)


class TestDataCategory:
    """Test DataCategory enum."""

    def test_all_categories_defined(self):
        """Test all categories are defined."""
        categories = [
            DataCategory.MARKET_DATA,
            DataCategory.TECHNICAL_INDICATORS,
            DataCategory.FUNDAMENTAL_DATA,
            DataCategory.ALTERNATIVE_DATA,
            DataCategory.SYNTHETIC_DATA,
            DataCategory.METADATA,
        ]
        for cat in categories:
            assert cat is not None

    def test_category_values(self):
        """Test category values are strings."""
        for cat in DataCategory:
            assert isinstance(cat.value, str)


class TestDataQualityLevel:
    """Test DataQualityLevel enum."""

    def test_all_levels_defined(self):
        """Test all quality levels are defined."""
        levels = [
            DataQualityLevel.HIGH,
            DataQualityLevel.MEDIUM,
            DataQualityLevel.LOW,
            DataQualityLevel.UNVERIFIED,
        ]
        for level in levels:
            assert level is not None


class TestDatasetInfo:
    """Test DatasetInfo dataclass."""

    @pytest.fixture
    def sample_dataset(self) -> DatasetInfo:
        """Create a sample dataset for testing."""
        return DatasetInfo(
            name="Test Dataset",
            category=DataCategory.MARKET_DATA,
            description="Test market data",
            time_range_start=datetime(2020, 1, 1),
            time_range_end=datetime(2024, 1, 1),
            size_rows=1_000_000,
            size_gb=1.5,
            assets_covered=["BTC", "ETH"],
            update_frequency="1-minute",
            source_provider="Test Provider",
            preprocessing=["Normalization", "Outlier removal"]
        )

    def test_create_dataset_info(self, sample_dataset):
        """Test creating dataset info."""
        assert sample_dataset.name == "Test Dataset"
        assert sample_dataset.category == DataCategory.MARKET_DATA
        assert sample_dataset.size_rows == 1_000_000

    def test_default_values(self, sample_dataset):
        """Test default values."""
        assert sample_dataset.quality_level == DataQualityLevel.HIGH
        assert sample_dataset.personal_data_included is False
        assert sample_dataset.geographic_coverage == "Global"

    def test_to_dict(self, sample_dataset):
        """Test serialization to dictionary."""
        data = sample_dataset.to_dict()
        assert data["name"] == "Test Dataset"
        assert data["category"] == "market_data"
        assert data["size_rows"] == 1_000_000
        assert data["personal_data_included"] is False

    def test_from_dict(self, sample_dataset):
        """Test deserialization from dictionary."""
        data = sample_dataset.to_dict()
        restored = DatasetInfo.from_dict(data)

        assert restored.name == sample_dataset.name
        assert restored.category == sample_dataset.category
        assert restored.size_rows == sample_dataset.size_rows

    def test_time_range_validation(self, sample_dataset):
        """Test time range is valid."""
        assert sample_dataset.time_range_start < sample_dataset.time_range_end

    def test_assets_covered_list(self, sample_dataset):
        """Test assets covered is a list."""
        assert isinstance(sample_dataset.assets_covered, list)
        assert len(sample_dataset.assets_covered) > 0

    def test_preprocessing_steps(self, sample_dataset):
        """Test preprocessing steps are documented."""
        assert len(sample_dataset.preprocessing) > 0


class TestTrainingDataSummary:
    """Test TrainingDataSummary dataclass."""

    def test_create_default_summary(self):
        """Test default summary creation."""
        summary = create_default_summary()
        assert summary.model_name == "Distributional PPO Trading Model"
        assert len(summary.datasets) > 0
        assert summary.total_training_samples > 0

    def test_summary_has_required_fields(self):
        """Test summary has all required fields."""
        summary = create_default_summary()
        assert summary.model_name is not None
        assert summary.model_version is not None
        assert summary.summary_date is not None
        assert summary.total_training_samples > 0
        assert summary.total_data_size_gb > 0

    def test_generate_public_summary(self):
        """Test public summary generation."""
        summary = create_default_summary()
        doc = summary.generate_public_summary()
        assert "Training Data Summary" in doc
        assert "Article 53(1)(d)" in doc
        assert summary.model_name in doc

    def test_summary_includes_datasets(self):
        """Test summary includes dataset information."""
        summary = create_default_summary()
        doc = summary.generate_public_summary()
        for dataset in summary.datasets:
            assert dataset.name in doc

    def test_summary_includes_quality_measures(self):
        """Test summary includes quality measures."""
        summary = create_default_summary()
        doc = summary.generate_public_summary()
        assert "Data Quality" in doc or "quality" in doc.lower()

    def test_summary_includes_bias_mitigation(self):
        """Test summary includes bias mitigation."""
        summary = create_default_summary()
        doc = summary.generate_public_summary()
        assert "Bias" in doc or "bias" in doc.lower()

    def test_summary_to_dict(self):
        """Test summary serialization."""
        summary = create_default_summary()
        data = summary.to_dict()
        assert data["model_name"] == summary.model_name
        assert len(data["datasets"]) == len(summary.datasets)

    def test_summary_from_dict(self):
        """Test summary deserialization."""
        original = create_default_summary()
        data = original.to_dict()
        restored = TrainingDataSummary.from_dict(data)

        assert restored.model_name == original.model_name
        assert len(restored.datasets) == len(original.datasets)


class TestTrainingDataSummaryManager:
    """Test TrainingDataSummaryManager."""

    @pytest.fixture
    def manager(self) -> TrainingDataSummaryManager:
        """Create manager instance."""
        return create_summary_manager()

    def test_get_public_summary(self, manager):
        """Test getting public summary."""
        summary = manager.get_public_summary()
        assert isinstance(summary, str)
        assert "Training Data" in summary

    def test_get_summary_metadata(self, manager):
        """Test getting summary metadata."""
        metadata = manager.get_summary_metadata()
        assert "model_name" in metadata
        assert "total_samples" in metadata
        assert "article_reference" in metadata
        assert "53(1)(d)" in metadata["article_reference"]

    def test_get_datasets(self, manager):
        """Test getting datasets list."""
        datasets = manager.get_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert "name" in datasets[0]

    def test_get_dataset_by_name(self, manager):
        """Test getting dataset by name."""
        dataset = manager.get_dataset_by_name("Binance Spot OHLCV")
        assert dataset is not None
        assert dataset.name == "Binance Spot OHLCV"

    def test_get_nonexistent_dataset(self, manager):
        """Test getting nonexistent dataset."""
        dataset = manager.get_dataset_by_name("Nonexistent")
        assert dataset is None

    def test_get_datasets_by_category(self, manager):
        """Test filtering datasets by category."""
        datasets = manager.get_datasets_by_category(DataCategory.MARKET_DATA)
        assert len(datasets) > 0
        assert all(d.category == DataCategory.MARKET_DATA for d in datasets)

    def test_add_dataset(self, manager):
        """Test adding new dataset."""
        initial_count = len(manager.current_summary.datasets)

        new_dataset = DatasetInfo(
            name="New Test Dataset",
            category=DataCategory.ALTERNATIVE_DATA,
            description="Test alternative data",
            time_range_start=datetime(2023, 1, 1),
            time_range_end=datetime(2024, 1, 1),
            size_rows=100_000,
            size_gb=0.5,
            assets_covered=["TEST"],
            update_frequency="daily",
            source_provider="Test",
            preprocessing=["Test"]
        )
        manager.add_dataset(new_dataset)

        assert len(manager.current_summary.datasets) == initial_count + 1

    def test_add_dataset_updates_totals(self, manager):
        """Test adding dataset updates totals."""
        initial_samples = manager.current_summary.total_training_samples

        new_dataset = DatasetInfo(
            name="Addition Test",
            category=DataCategory.SYNTHETIC_DATA,
            description="Test",
            time_range_start=datetime(2024, 1, 1),
            time_range_end=datetime(2024, 6, 1),
            size_rows=1_000_000,
            size_gb=1.0,
            assets_covered=["TEST"],
            update_frequency="once",
            source_provider="Internal",
            preprocessing=["None"]
        )
        manager.add_dataset(new_dataset)

        assert manager.current_summary.total_training_samples > initial_samples

    def test_remove_dataset(self, manager):
        """Test removing dataset."""
        initial_count = len(manager.current_summary.datasets)
        first_dataset_name = manager.current_summary.datasets[0].name

        result = manager.remove_dataset(first_dataset_name)
        assert result is True
        assert len(manager.current_summary.datasets) == initial_count - 1

    def test_remove_nonexistent_dataset(self, manager):
        """Test removing nonexistent dataset."""
        result = manager.remove_dataset("Nonexistent Dataset")
        assert result is False

    def test_validate_summary(self, manager):
        """Test summary validation."""
        result = manager.validate_summary()
        assert "compliant" in result
        assert "checks" in result
        assert result["compliant"] is True

    def test_export_json(self, manager):
        """Test JSON export."""
        json_str = manager.export_json()
        data = json.loads(json_str)
        assert data["model_name"] == manager.current_summary.model_name

    def test_import_json(self, manager):
        """Test JSON import."""
        json_str = manager.export_json()

        # Modify the JSON
        data = json.loads(json_str)
        data["model_version"] = "5.0"
        modified_json = json.dumps(data)

        manager.import_json(modified_json)
        assert manager.current_summary.model_version == "5.0"

    def test_update_summary_saves_history(self, manager):
        """Test updating summary saves to history."""
        original_version = manager.current_summary.model_version

        new_summary = create_default_summary()
        new_summary.model_version = "6.0"
        manager.update_summary(new_summary)

        history = manager.get_summary_history()
        assert len(history) > 0
        assert any(h["model_version"] == original_version for h in history)


class TestValidateDatasetInfo:
    """Test dataset info validation."""

    def test_validate_complete_dataset(self):
        """Test validating complete dataset."""
        dataset = DatasetInfo(
            name="Test",
            category=DataCategory.MARKET_DATA,
            description="Test description",
            time_range_start=datetime(2020, 1, 1),
            time_range_end=datetime(2024, 1, 1),
            size_rows=1000,
            size_gb=0.1,
            assets_covered=["TEST"],
            update_frequency="daily",
            source_provider="Provider",
            preprocessing=["Step 1"]
        )
        result = validate_dataset_info(dataset)
        assert result["all_valid"] is True

    def test_validate_missing_name(self):
        """Test validation fails with empty name."""
        dataset = DatasetInfo(
            name="",
            category=DataCategory.MARKET_DATA,
            description="Test",
            time_range_start=datetime(2020, 1, 1),
            time_range_end=datetime(2024, 1, 1),
            size_rows=1000,
            size_gb=0.1,
            assets_covered=["TEST"],
            update_frequency="daily",
            source_provider="Provider",
            preprocessing=["Step 1"]
        )
        result = validate_dataset_info(dataset)
        assert result["has_name"] is False
        assert result["all_valid"] is False

    def test_validate_invalid_time_range(self):
        """Test validation fails with invalid time range."""
        dataset = DatasetInfo(
            name="Test",
            category=DataCategory.MARKET_DATA,
            description="Test",
            time_range_start=datetime(2024, 1, 1),
            time_range_end=datetime(2020, 1, 1),  # Before start!
            size_rows=1000,
            size_gb=0.1,
            assets_covered=["TEST"],
            update_frequency="daily",
            source_provider="Provider",
            preprocessing=["Step 1"]
        )
        result = validate_dataset_info(dataset)
        assert result["has_time_range"] is False

    def test_validate_zero_size(self):
        """Test validation fails with zero size."""
        dataset = DatasetInfo(
            name="Test",
            category=DataCategory.MARKET_DATA,
            description="Test",
            time_range_start=datetime(2020, 1, 1),
            time_range_end=datetime(2024, 1, 1),
            size_rows=0,
            size_gb=0,
            assets_covered=["TEST"],
            update_frequency="daily",
            source_provider="Provider",
            preprocessing=["Step 1"]
        )
        result = validate_dataset_info(dataset)
        assert result["has_valid_size"] is False

    def test_validate_empty_preprocessing(self):
        """Test validation fails with no preprocessing."""
        dataset = DatasetInfo(
            name="Test",
            category=DataCategory.MARKET_DATA,
            description="Test",
            time_range_start=datetime(2020, 1, 1),
            time_range_end=datetime(2024, 1, 1),
            size_rows=1000,
            size_gb=0.1,
            assets_covered=["TEST"],
            update_frequency="daily",
            source_provider="Provider",
            preprocessing=[]
        )
        result = validate_dataset_info(dataset)
        assert result["has_preprocessing"] is False


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_default_summary(self):
        """Test default summary creation."""
        summary = create_default_summary()
        assert isinstance(summary, TrainingDataSummary)
        assert len(summary.datasets) > 0

    def test_create_summary_manager(self):
        """Test manager factory."""
        manager = create_summary_manager()
        assert isinstance(manager, TrainingDataSummaryManager)

    def test_get_data_categories(self):
        """Test getting data categories."""
        categories = get_data_categories()
        assert isinstance(categories, list)
        assert "market_data" in categories
        assert "synthetic_data" in categories


class TestArticle53dCompliance:
    """Integration tests for Article 53(1)(d) compliance."""

    @pytest.fixture
    def summary(self) -> TrainingDataSummary:
        """Create summary for testing."""
        return create_default_summary()

    @pytest.fixture
    def manager(self) -> TrainingDataSummaryManager:
        """Create manager for testing."""
        return create_summary_manager()

    def test_summary_includes_all_required_elements(self, summary):
        """Test summary has all required elements per Article 53(1)(d)."""
        doc = summary.generate_public_summary()
        required_elements = [
            "Training Data",
            "datasets" if "dataset" in doc.lower() else "Dataset",
            "Data Quality" if "quality" in doc.lower() else "quality",
            "Bias" if "bias" in doc.lower() else "bias",
        ]
        for element in required_elements:
            assert element.lower() in doc.lower(), f"Missing: {element}"

    def test_all_datasets_have_required_info(self, summary):
        """Test all datasets have required information."""
        for dataset in summary.datasets:
            assert dataset.name is not None
            assert dataset.description is not None
            assert dataset.time_range_start is not None
            assert dataset.size_rows > 0
            assert dataset.source_provider is not None

    def test_bias_mitigation_documented(self, summary):
        """Test bias mitigation is documented."""
        assert len(summary.bias_mitigation_steps) > 0
        assert any("bias" in step.lower() for step in summary.bias_mitigation_steps)

    def test_data_quality_measures_documented(self, summary):
        """Test data quality measures are documented."""
        assert len(summary.data_quality_measures) > 0

    def test_personal_data_statement_present(self, summary):
        """Test personal data statement is present."""
        assert summary.personal_data_statement is not None
        assert len(summary.personal_data_statement) > 0
        assert "personal data" in summary.personal_data_statement.lower()

    def test_copyright_reference_present(self, summary):
        """Test copyright reference is present."""
        assert summary.copyright_compliance_reference is not None

    def test_public_summary_is_sufficiently_detailed(self, summary):
        """Test public summary is 'sufficiently detailed' per Article 53(1)(d)."""
        doc = summary.generate_public_summary()

        # Should include model info
        assert summary.model_name in doc
        assert summary.model_version in doc

        # Should include quantitative metrics (formatted with commas)
        assert f"{summary.total_training_samples:,}" in doc or str(summary.total_training_samples) in doc

        # Should include dataset details (names are in the summary)
        for dataset in summary.datasets:
            assert dataset.name in doc

    def test_manager_validation_passes(self, manager):
        """Test manager validation passes for default summary."""
        result = manager.validate_summary()
        assert result["compliant"] is True


class TestDatasetCategories:
    """Test dataset categories are properly represented."""

    @pytest.fixture
    def summary(self) -> TrainingDataSummary:
        """Create summary for testing."""
        return create_default_summary()

    def test_market_data_present(self, summary):
        """Test market data datasets are present."""
        market_datasets = [
            d for d in summary.datasets
            if d.category == DataCategory.MARKET_DATA
        ]
        assert len(market_datasets) > 0

    def test_synthetic_data_present(self, summary):
        """Test synthetic data is present."""
        synthetic_datasets = [
            d for d in summary.datasets
            if d.category == DataCategory.SYNTHETIC_DATA
        ]
        assert len(synthetic_datasets) > 0

    def test_technical_indicators_present(self, summary):
        """Test technical indicators dataset is present."""
        tech_datasets = [
            d for d in summary.datasets
            if d.category == DataCategory.TECHNICAL_INDICATORS
        ]
        assert len(tech_datasets) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_datasets_list(self):
        """Test summary with empty datasets list."""
        summary = TrainingDataSummary(
            model_name="Test",
            model_version="1.0",
            summary_date=datetime.utcnow(),
            datasets=[],
            total_training_samples=0,
            total_data_size_gb=0,
            training_period_start=datetime(2020, 1, 1),
            training_period_end=datetime(2024, 1, 1),
            data_quality_measures=["Test measure"],
            bias_mitigation_steps=["Test step"]
        )
        doc = summary.generate_public_summary()
        assert "Training Data Summary" in doc

    def test_special_characters_in_dataset_name(self):
        """Test dataset with special characters in name."""
        dataset = DatasetInfo(
            name="Test/Dataset:v1.0 (beta)",
            category=DataCategory.MARKET_DATA,
            description="Test",
            time_range_start=datetime(2020, 1, 1),
            time_range_end=datetime(2024, 1, 1),
            size_rows=1000,
            size_gb=0.1,
            assets_covered=["TEST"],
            update_frequency="daily",
            source_provider="Provider",
            preprocessing=["Step 1"]
        )
        summary = TrainingDataSummary(
            model_name="Test",
            model_version="1.0",
            summary_date=datetime.utcnow(),
            datasets=[dataset],
            total_training_samples=1000,
            total_data_size_gb=0.1,
            training_period_start=datetime(2020, 1, 1),
            training_period_end=datetime(2024, 1, 1),
            data_quality_measures=["Test"],
            bias_mitigation_steps=["Test"]
        )
        doc = summary.generate_public_summary()
        assert "Test/Dataset:v1.0 (beta)" in doc

    def test_large_numbers_formatting(self):
        """Test large numbers are formatted correctly."""
        summary = create_default_summary()
        doc = summary.generate_public_summary()
        # Large numbers should be formatted with commas or similar
        assert str(summary.total_training_samples) in doc.replace(",", "")


class TestMultipleVersions:
    """Test handling of multiple summary versions."""

    @pytest.fixture
    def manager(self) -> TrainingDataSummaryManager:
        """Create manager for testing."""
        return create_summary_manager()

    def test_version_history_tracking(self, manager):
        """Test version history is tracked."""
        # Create new versions
        for i in range(3):
            new_summary = create_default_summary()
            new_summary.model_version = f"{4 + i}.0"
            manager.update_summary(new_summary)

        history = manager.get_summary_history()
        assert len(history) >= 3

    def test_history_preserves_version_info(self, manager):
        """Test history preserves version information."""
        manager.update_summary(create_default_summary())

        history = manager.get_summary_history()
        for entry in history:
            assert "model_version" in entry
            assert "summary_date" in entry
