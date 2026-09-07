# -*- coding: utf-8 -*-
"""
Tests for Article 53(1)(b) EU AI Act - GPAI Model Card.

This module provides comprehensive tests for the GPAI Model Card
functionality required by Article 53(1)(b) of the EU AI Act.

Coverage includes:
- GPAIModelCard dataclass
- ModelCardManager functionality
- Enums and data structures
- Serialization/deserialization
- Compliance validation
- Factory functions
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from services.ai_act.gpai_model_card import (
    # Enums
    IntendedUse,
    LimitationType,
    RiskLevel,
    EvaluationDataset,
    # Data structures
    ModelLimitation,
    PerformanceMetric,
    BiasAssessment,
    EthicalConsideration,
    DownstreamRequirement,
    GPAIModelCard,
    # Main class
    ModelCardManager,
    # Factory functions
    create_default_model_card,
    create_model_card_manager,
    get_default_limitations,
    get_default_biases,
    get_default_downstream_requirements,
    validate_model_card,
)


class TestIntendedUse:
    """Test IntendedUse enum."""

    def test_all_uses_defined(self):
        """Test all intended use categories are defined."""
        uses = [
            IntendedUse.TRADING_SIGNALS,
            IntendedUse.PORTFOLIO_OPTIMIZATION,
            IntendedUse.RISK_ASSESSMENT,
            IntendedUse.RESEARCH,
            IntendedUse.MARKET_ANALYSIS,
            IntendedUse.STRATEGY_DEVELOPMENT,
        ]
        for use in uses:
            assert use is not None

    def test_use_values(self):
        """Test intended use values are strings."""
        assert IntendedUse.TRADING_SIGNALS.value == "trading_signal_generation"
        assert IntendedUse.RESEARCH.value == "research_and_backtesting"


class TestLimitationType:
    """Test LimitationType enum."""

    def test_all_types_defined(self):
        """Test all limitation types are defined."""
        types = [
            LimitationType.TECHNICAL,
            LimitationType.PERFORMANCE,
            LimitationType.ETHICAL,
            LimitationType.REGULATORY,
            LimitationType.DATA,
            LimitationType.OPERATIONAL,
        ]
        for t in types:
            assert t is not None

    def test_type_values(self):
        """Test limitation type values."""
        assert LimitationType.TECHNICAL.value == "technical"
        assert LimitationType.PERFORMANCE.value == "performance"


class TestRiskLevel:
    """Test RiskLevel enum."""

    def test_all_levels_defined(self):
        """Test all risk levels are defined."""
        levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert len(levels) == 4

    def test_level_values(self):
        """Test risk level values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.CRITICAL.value == "critical"


class TestEvaluationDataset:
    """Test EvaluationDataset enum."""

    def test_all_datasets_defined(self):
        """Test all evaluation dataset types are defined."""
        datasets = [
            EvaluationDataset.TRAINING,
            EvaluationDataset.VALIDATION,
            EvaluationDataset.TEST,
            EvaluationDataset.PRODUCTION,
            EvaluationDataset.OUT_OF_DISTRIBUTION,
        ]
        assert len(datasets) == 5


class TestModelLimitation:
    """Test ModelLimitation dataclass."""

    def test_create_limitation(self):
        """Test creating a model limitation."""
        limitation = ModelLimitation(
            limitation_type=LimitationType.TECHNICAL,
            description="Requires 100ms latency",
            severity="medium",
        )
        assert limitation.limitation_type == LimitationType.TECHNICAL
        assert limitation.description == "Requires 100ms latency"
        assert limitation.severity == "medium"

    def test_limitation_with_mitigation(self):
        """Test limitation with mitigation strategy."""
        limitation = ModelLimitation(
            limitation_type=LimitationType.PERFORMANCE,
            description="May fail in extreme markets",
            severity="high",
            mitigation="Implement kill switch",
        )
        assert limitation.mitigation == "Implement kill switch"

    def test_limitation_to_dict(self):
        """Test limitation serialization."""
        limitation = ModelLimitation(
            limitation_type=LimitationType.TECHNICAL,
            description="Test",
            severity="low",
            affected_uses=["trading"],
        )
        data = limitation.to_dict()
        assert data["limitation_type"] == "technical"
        assert data["description"] == "Test"
        assert "trading" in data["affected_uses"]


class TestPerformanceMetric:
    """Test PerformanceMetric dataclass."""

    def test_create_metric(self):
        """Test creating a performance metric."""
        metric = PerformanceMetric(name="Sharpe Ratio", value=1.5, unit="", context="BTC 2020-2024")
        assert metric.name == "Sharpe Ratio"
        assert metric.value == 1.5
        assert metric.context == "BTC 2020-2024"

    def test_metric_with_confidence_interval(self):
        """Test metric with confidence interval."""
        metric = PerformanceMetric(
            name="Sharpe", value=1.2, unit="", context="Test", confidence_interval=(0.9, 1.5)
        )
        assert metric.confidence_interval == (0.9, 1.5)

    def test_metric_to_dict(self):
        """Test metric serialization."""
        metric = PerformanceMetric(
            name="Win Rate", value=52.0, unit="%", context="Test", dataset=EvaluationDataset.TEST
        )
        data = metric.to_dict()
        assert data["name"] == "Win Rate"
        assert data["value"] == 52.0
        assert data["unit"] == "%"
        assert data["dataset"] == "test"


class TestBiasAssessment:
    """Test BiasAssessment dataclass."""

    def test_create_bias(self):
        """Test creating a bias assessment."""
        bias = BiasAssessment(
            bias_type="Temporal",
            description="Better in trending markets",
            impact="May generate false signals",
            mitigation_status="Partially mitigated",
        )
        assert bias.bias_type == "Temporal"
        assert "trending" in bias.description

    def test_bias_to_dict(self):
        """Test bias serialization."""
        bias = BiasAssessment(
            bias_type="Asset",
            description="Test",
            impact="Test impact",
            mitigation_status="Documented",
            affected_groups=["Users"],
        )
        data = bias.to_dict()
        assert data["bias_type"] == "Asset"
        assert "Users" in data["affected_groups"]


class TestEthicalConsideration:
    """Test EthicalConsideration dataclass."""

    def test_create_consideration(self):
        """Test creating ethical consideration."""
        eth = EthicalConsideration(
            category="Financial Risk", description="May cause losses", guidance="Use with caution"
        )
        assert eth.category == "Financial Risk"
        assert eth.guidance == "Use with caution"

    def test_consideration_to_dict(self):
        """Test consideration serialization."""
        eth = EthicalConsideration(
            category="Test",
            description="Test desc",
            guidance="Test guidance",
            relevant_articles=["Article 50"],
        )
        data = eth.to_dict()
        assert data["category"] == "Test"
        assert "Article 50" in data["relevant_articles"]


class TestDownstreamRequirement:
    """Test DownstreamRequirement dataclass."""

    def test_create_requirement(self):
        """Test creating downstream requirement."""
        req = DownstreamRequirement(
            requirement_id="DR-001",
            description="Implement kill switch",
            article_reference="Article 14(4)(f)",
            mandatory=True,
            implementation_guidance="Add stop button",
        )
        assert req.requirement_id == "DR-001"
        assert req.mandatory is True
        assert "14" in req.article_reference

    def test_requirement_to_dict(self):
        """Test requirement serialization."""
        req = DownstreamRequirement(
            requirement_id="DR-002",
            description="Log outputs",
            article_reference="Article 12",
            mandatory=True,
            implementation_guidance="Store for 5 years",
        )
        data = req.to_dict()
        assert data["requirement_id"] == "DR-002"
        assert data["mandatory"] is True


class TestGPAIModelCard:
    """Test GPAIModelCard dataclass."""

    @pytest.fixture
    def sample_card(self) -> GPAIModelCard:
        """Create a sample model card for testing."""
        return create_default_model_card()

    def test_default_card_creation(self, sample_card):
        """Test default model card creation."""
        assert sample_card.model_name == "Distributional PPO Trading Model"
        assert sample_card.model_version == "4.0"
        assert sample_card.provider is not None

    def test_card_has_intended_uses(self, sample_card):
        """Test card has intended uses defined."""
        assert len(sample_card.intended_uses) > 0
        assert IntendedUse.TRADING_SIGNALS in sample_card.intended_uses

    def test_card_has_out_of_scope_uses(self, sample_card):
        """Test card has out of scope uses defined."""
        assert len(sample_card.out_of_scope_uses) > 0
        assert any("advice" in u.lower() for u in sample_card.out_of_scope_uses)

    def test_card_has_limitations(self, sample_card):
        """Test card has limitations documented."""
        assert len(sample_card.limitations) > 0

    def test_card_has_biases(self, sample_card):
        """Test card has biases documented."""
        assert len(sample_card.known_biases) > 0

    def test_card_has_metrics(self, sample_card):
        """Test card has performance metrics."""
        assert len(sample_card.performance_metrics) > 0
        # Check for key metrics
        metric_names = [m.name for m in sample_card.performance_metrics]
        assert "Sharpe Ratio" in metric_names

    def test_card_has_downstream_requirements(self, sample_card):
        """Test card has downstream requirements."""
        assert len(sample_card.downstream_requirements) > 0

    def test_card_has_human_oversight(self, sample_card):
        """Test card has human oversight recommendations."""
        assert len(sample_card.human_oversight_recommendations) > 0

    def test_card_classification(self, sample_card):
        """Test card has correct classification."""
        assert "GPAI" in sample_card.eu_ai_act_classification

    def test_card_article_references(self, sample_card):
        """Test card has article references."""
        assert len(sample_card.article_references) > 0
        assert "Article 53" in sample_card.article_references

    def test_generate_card_document(self, sample_card):
        """Test card document generation."""
        doc = sample_card.generate_card()
        assert "GPAI Model Card" in doc
        assert "Article 53(1)(b)" in doc
        assert sample_card.model_name in doc
        assert "Intended Use" in doc
        assert "Limitations" in doc
        assert "Performance" in doc

    def test_card_includes_ethics(self, sample_card):
        """Test card includes ethical considerations."""
        doc = sample_card.generate_card()
        assert "Ethical" in doc

    def test_card_includes_downstream_info(self, sample_card):
        """Test card includes downstream provider info."""
        doc = sample_card.generate_card()
        assert "Downstream" in doc

    def test_card_to_dict(self, sample_card):
        """Test card serialization to dictionary."""
        data = sample_card.to_dict()
        assert data["model_name"] == sample_card.model_name
        assert data["model_version"] == sample_card.model_version
        assert len(data["intended_uses"]) > 0
        assert len(data["limitations"]) > 0

    def test_card_from_dict(self, sample_card):
        """Test card deserialization from dictionary."""
        data = sample_card.to_dict()
        restored = GPAIModelCard.from_dict(data)
        assert restored.model_name == sample_card.model_name
        assert restored.model_version == sample_card.model_version
        assert len(restored.intended_uses) == len(sample_card.intended_uses)

    def test_card_summary(self, sample_card):
        """Test card summary generation."""
        summary = sample_card.get_summary()
        assert "model_name" in summary
        assert "limitations_count" in summary
        assert summary["limitations_count"] > 0


class TestModelCardManager:
    """Test ModelCardManager."""

    @pytest.fixture
    def manager(self) -> ModelCardManager:
        """Create a manager instance."""
        return create_model_card_manager()

    def test_create_manager(self, manager):
        """Test manager creation."""
        assert isinstance(manager, ModelCardManager)
        assert manager.current_card is not None

    def test_get_model_card(self, manager):
        """Test getting model card document."""
        card = manager.get_model_card()
        assert isinstance(card, str)
        assert "GPAI Model Card" in card

    def test_get_model_card_json(self, manager):
        """Test getting model card as JSON."""
        json_str = manager.get_model_card_json()
        assert isinstance(json_str, str)
        assert "model_name" in json_str

    def test_get_card_metadata(self, manager):
        """Test getting card metadata."""
        metadata = manager.get_card_metadata()
        assert "model_name" in metadata
        assert "model_version" in metadata
        assert "classification" in metadata
        assert "GPAI" in metadata["classification"]
        assert "Article 53(1)(b)" in metadata["article_reference"]

    def test_get_summary(self, manager):
        """Test getting card summary."""
        summary = manager.get_summary()
        assert "model_name" in summary
        assert "limitations_count" in summary

    def test_validate_compliance(self, manager):
        """Test compliance validation."""
        result = manager.validate_compliance()
        assert "compliant" in result
        assert "checks" in result
        assert result["compliant"] is True

    def test_validate_compliance_checks(self, manager):
        """Test specific compliance checks."""
        result = manager.validate_compliance()
        checks = result["checks"]
        assert checks["has_model_name"] is True
        assert checks["has_intended_uses"] is True
        assert checks["has_limitations"] is True
        assert checks["has_downstream_requirements"] is True

    def test_update_card(self, manager):
        """Test updating model card."""
        original_version = manager.current_card.card_version
        success = manager.update_card({"model_version": "5.0"})
        assert success is True
        assert manager.current_card.model_version == "5.0"
        assert manager.current_card.card_version != original_version

    def test_version_history(self, manager):
        """Test version history tracking."""
        manager.update_card({"model_version": "4.1"})
        history = manager.get_version_history()
        assert len(history) > 0

    def test_downstream_checklist(self, manager):
        """Test getting downstream checklist."""
        checklist = manager.get_downstream_checklist()
        assert len(checklist) > 0
        assert "requirement_id" in checklist[0]
        assert "status" in checklist[0]


class TestFactoryFunctions:
    """Test factory and utility functions."""

    def test_create_default_model_card(self):
        """Test default model card factory."""
        card = create_default_model_card()
        assert isinstance(card, GPAIModelCard)
        assert card.model_name is not None

    def test_create_model_card_manager(self):
        """Test model card manager factory."""
        manager = create_model_card_manager()
        assert isinstance(manager, ModelCardManager)

    def test_create_manager_with_custom_card(self):
        """Test manager with custom card."""
        custom_card = GPAIModelCard(
            model_name="Custom Model",
            model_version="1.0",
            provider="Test",
            release_date=datetime.utcnow(),
        )
        manager = create_model_card_manager(model_card=custom_card)
        assert manager.current_card.model_name == "Custom Model"

    def test_get_default_limitations(self):
        """Test getting default limitations."""
        limitations = get_default_limitations()
        assert len(limitations) > 0
        assert all(isinstance(l, ModelLimitation) for l in limitations)

    def test_get_default_biases(self):
        """Test getting default biases."""
        biases = get_default_biases()
        assert len(biases) > 0
        assert all(isinstance(b, BiasAssessment) for b in biases)

    def test_get_default_downstream_requirements(self):
        """Test getting default downstream requirements."""
        requirements = get_default_downstream_requirements()
        assert len(requirements) > 0
        assert all(isinstance(r, DownstreamRequirement) for r in requirements)


class TestValidateModelCard:
    """Test model card validation function."""

    def test_validate_complete_card(self):
        """Test validation of complete card."""
        card = create_default_model_card()
        result = validate_model_card(card)
        assert result["has_identity"] is True
        assert result["has_technical_details"] is True
        assert result["has_intended_uses"] is True
        assert result["has_limitations"] is True

    def test_validate_incomplete_card(self):
        """Test validation of incomplete card."""
        card = GPAIModelCard(
            model_name="", model_version="", provider="", release_date=datetime.utcnow()
        )
        result = validate_model_card(card)
        assert result["has_identity"] is False
        assert result["has_intended_uses"] is False


class TestArticle53Compliance:
    """Integration tests for Article 53(1)(b) compliance."""

    @pytest.fixture
    def card(self) -> GPAIModelCard:
        """Create card for testing."""
        return create_default_model_card()

    @pytest.fixture
    def manager(self) -> ModelCardManager:
        """Create manager for testing."""
        return create_model_card_manager()

    def test_downstream_provider_info_complete(self, card):
        """Test information for downstream providers is complete."""
        assert len(card.downstream_requirements) > 0
        assert len(card.human_oversight_recommendations) > 0
        assert len(card.limitations) > 0

    def test_all_required_articles_referenced(self, card):
        """Test all required articles are referenced."""
        refs = " ".join(card.article_references)
        assert "53" in refs  # GPAI requirements
        assert "50" in refs  # Transparency

    def test_mandatory_downstream_requirements(self, card):
        """Test mandatory requirements are documented."""
        mandatory = [r for r in card.downstream_requirements if r.mandatory]
        assert len(mandatory) > 0
        # Check key requirements
        req_descriptions = " ".join([r.description for r in mandatory])
        assert "kill switch" in req_descriptions.lower() or "halt" in req_descriptions.lower()
        assert "log" in req_descriptions.lower()
        assert "disclosure" in req_descriptions.lower()

    def test_compliance_validation_passes(self, manager):
        """Test full compliance validation."""
        result = manager.validate_compliance()
        assert result["compliant"] is True
        assert len(result["missing"]) == 0

    def test_document_mentions_article_53(self, card):
        """Test generated document mentions Article 53."""
        doc = card.generate_card()
        assert "Article 53" in doc
        assert "EU AI Act" in doc

    def test_all_biases_have_mitigation_status(self, card):
        """Test all biases have mitigation status."""
        for bias in card.known_biases:
            assert bias.mitigation_status is not None
            assert len(bias.mitigation_status) > 0

    def test_all_limitations_have_severity(self, card):
        """Test all limitations have severity."""
        for limitation in card.limitations:
            assert limitation.severity is not None
            assert limitation.severity in ["low", "medium", "high"]


class TestSerialization:
    """Test serialization and deserialization."""

    @pytest.fixture
    def card(self) -> GPAIModelCard:
        """Create card for testing."""
        return create_default_model_card()

    def test_round_trip_serialization(self, card):
        """Test card survives round-trip serialization."""
        data = card.to_dict()
        restored = GPAIModelCard.from_dict(data)

        assert restored.model_name == card.model_name
        assert restored.model_version == card.model_version
        assert len(restored.intended_uses) == len(card.intended_uses)
        assert len(restored.limitations) == len(card.limitations)
        assert len(restored.performance_metrics) == len(card.performance_metrics)
        assert len(restored.known_biases) == len(card.known_biases)
        assert len(restored.downstream_requirements) == len(card.downstream_requirements)

    def test_json_export_import(self):
        """Test JSON export and import."""
        manager = create_model_card_manager()
        json_str = manager.get_model_card_json()

        import json

        data = json.loads(json_str)
        restored = GPAIModelCard.from_dict(data)

        assert restored.model_name == manager.current_card.model_name


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_model_card(self):
        """Test model card with minimal data."""
        card = GPAIModelCard(
            model_name="Test", model_version="1.0", provider="Test", release_date=datetime.utcnow()
        )
        doc = card.generate_card()
        assert "GPAI Model Card" in doc
        assert "Test" in doc

    def test_manager_with_empty_card(self):
        """Test manager with minimal card."""
        card = GPAIModelCard(
            model_name="Empty", model_version="0.1", provider="Test", release_date=datetime.utcnow()
        )
        manager = create_model_card_manager(model_card=card)
        result = manager.validate_compliance()
        assert result["compliant"] is False
        assert len(result["missing"]) > 0

    def test_multiple_updates(self):
        """Test multiple sequential updates."""
        manager = create_model_card_manager()
        original_version = manager.current_card.card_version

        for i in range(3):
            manager.update_card({"model_version": f"4.{i}"})

        history = manager.get_version_history()
        assert len(history) == 3
        assert manager.current_card.model_version == "4.2"


class TestDocumentGeneration:
    """Test document generation quality."""

    @pytest.fixture
    def card(self) -> GPAIModelCard:
        """Create card for testing."""
        return create_default_model_card()

    def test_document_structure(self, card):
        """Test document has proper structure."""
        doc = card.generate_card()
        sections = [
            "Model Overview",
            "Intended Use",
            "Performance",
            "Limitations",
            "Known Biases",
            "Ethical Considerations",
            "Downstream Providers",
            "Human Oversight",
            "EU AI Act Classification",
        ]
        for section in sections:
            assert section in doc

    def test_document_formatting(self, card):
        """Test document has proper markdown formatting."""
        doc = card.generate_card()
        # Check for headers
        assert "# " in doc
        assert "## " in doc
        # Check for tables
        assert "|" in doc
        # Check for lists
        assert "- " in doc

    def test_document_contains_all_metrics(self, card):
        """Test document contains all metrics."""
        doc = card.generate_card()
        for metric in card.performance_metrics:
            assert metric.name in doc

    def test_document_contains_all_limitations(self, card):
        """Test document contains all limitations."""
        doc = card.generate_card()
        for limitation in card.limitations:
            assert limitation.description in doc
