"""
Comprehensive tests for EU AI Act Decision Explainability module.

Tests Article 13 (Transparency) and Article 14 (Human Oversight) explainability requirements.

Coverage target: 95%+
"""

import pytest
import threading
import tempfile
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, patch

from services.ai_act.explainability import (
    ExplanationType,
    FeatureContribution,
    CounterfactualExplanation,
    DecisionExplanation,
    DecisionExplainer,
    create_decision_explainer,
)


# =============================================================================
# Test ExplanationType Enum
# =============================================================================


class TestExplanationType:
    """Tests for ExplanationType enum."""

    def test_explanation_type_values(self):
        """Test all explanation type values exist."""
        assert ExplanationType.FEATURE_ATTRIBUTION.value == "feature_attribution"
        assert ExplanationType.COUNTERFACTUAL.value == "counterfactual"
        assert ExplanationType.RULE_BASED.value == "rule_based"
        assert ExplanationType.CONFIDENCE_BASED.value == "confidence_based"
        assert ExplanationType.COMPARATIVE.value == "comparative"

    def test_explanation_type_membership(self):
        """Test all expected types are in the enum."""
        expected_types = [
            "feature_attribution",
            "counterfactual",
            "rule_based",
            "confidence_based",
            "comparative",
        ]
        actual_types = [t.value for t in ExplanationType]
        for expected in expected_types:
            assert expected in actual_types

    def test_explanation_type_count(self):
        """Test there are exactly 5 explanation types."""
        assert len(ExplanationType) == 5


# =============================================================================
# Test FeatureContribution Dataclass
# =============================================================================


class TestFeatureContribution:
    """Tests for FeatureContribution dataclass."""

    def test_create_positive_contribution(self):
        """Test creating a feature contribution with positive value."""
        fc = FeatureContribution(feature_name="momentum", feature_value=0.8, contribution=0.35)
        assert fc.feature_name == "momentum"
        assert fc.feature_value == 0.8
        assert fc.contribution == 0.35
        assert fc.direction == "positive"  # Auto-set by __post_init__

    def test_create_negative_contribution(self):
        """Test creating a feature contribution with negative value."""
        fc = FeatureContribution(feature_name="volatility", feature_value=0.9, contribution=-0.25)
        assert fc.feature_name == "volatility"
        assert fc.contribution == -0.25
        assert fc.direction == "negative"

    def test_create_neutral_contribution(self):
        """Test creating a feature contribution with neutral value."""
        fc = FeatureContribution(
            feature_name="volume", feature_value=0.5, contribution=0.02  # Below 0.05 threshold
        )
        assert fc.direction == "neutral"

    def test_direction_threshold_positive(self):
        """Test direction threshold at positive boundary."""
        fc_below = FeatureContribution("x", 0.5, 0.04)
        assert fc_below.direction == "neutral"

        fc_at = FeatureContribution("x", 0.5, 0.05)
        assert fc_at.direction == "neutral"  # > not >=

        fc_above = FeatureContribution("x", 0.5, 0.06)
        assert fc_above.direction == "positive"

    def test_direction_threshold_negative(self):
        """Test direction threshold at negative boundary."""
        fc_below = FeatureContribution("x", 0.5, -0.04)
        assert fc_below.direction == "neutral"

        fc_at = FeatureContribution("x", 0.5, -0.05)
        assert fc_at.direction == "neutral"  # < not <=

        fc_above = FeatureContribution("x", 0.5, -0.06)
        assert fc_above.direction == "negative"

    def test_human_readable_generated_strongly_supports(self):
        """Test human-readable for strong positive contribution."""
        fc = FeatureContribution("momentum", 0.8, 0.4)  # > 0.3 = strongly
        assert "strongly supports" in fc.human_readable

    def test_human_readable_generated_moderately_supports(self):
        """Test human-readable for moderate positive contribution."""
        fc = FeatureContribution("momentum", 0.6, 0.2)  # 0.1-0.3 = moderately
        assert "moderately supports" in fc.human_readable

    def test_human_readable_generated_slightly_supports(self):
        """Test human-readable for slight positive contribution."""
        fc = FeatureContribution("momentum", 0.55, 0.08)  # < 0.1 = slightly
        assert "slightly supports" in fc.human_readable

    def test_human_readable_generated_opposes(self):
        """Test human-readable for negative contribution."""
        fc = FeatureContribution("volatility", 0.9, -0.35)
        assert "opposes" in fc.human_readable

    def test_human_readable_generated_neutral(self):
        """Test human-readable for neutral contribution."""
        fc = FeatureContribution("volume", 0.5, 0.02)
        assert "neutral" in fc.human_readable

    def test_human_readable_custom(self):
        """Test custom human-readable description."""
        fc = FeatureContribution(
            feature_name="rsi",
            feature_value=70.0,
            contribution=0.3,
            human_readable="RSI is overbought",
        )
        assert fc.human_readable == "RSI is overbought"

    def test_importance_rank_default(self):
        """Test default importance rank."""
        fc = FeatureContribution("x", 0.5, 0.1)
        assert fc.importance_rank == 0

    def test_importance_rank_custom(self):
        """Test custom importance rank."""
        fc = FeatureContribution("x", 0.5, 0.1, importance_rank=3)
        assert fc.importance_rank == 3

    def test_to_dict(self):
        """Test conversion to dictionary."""
        fc = FeatureContribution("momentum", 0.8, 0.35, importance_rank=1)
        d = fc.to_dict()

        assert d["feature_name"] == "momentum"
        assert d["feature_value"] == 0.8
        assert d["contribution"] == 0.35
        assert d["direction"] == "positive"
        assert d["importance_rank"] == 1
        assert "human_readable" in d


# =============================================================================
# Test CounterfactualExplanation Dataclass
# =============================================================================


class TestCounterfactualExplanation:
    """Tests for CounterfactualExplanation dataclass."""

    def test_create_counterfactual_minimal(self):
        """Test creating a counterfactual with minimal data."""
        cf = CounterfactualExplanation(
            original_decision="BUY",
            alternative_decision="HOLD",
            feature_changes={"momentum": (0.8, 0.5)},
        )
        assert cf.original_decision == "BUY"
        assert cf.alternative_decision == "HOLD"
        assert cf.feature_changes == {"momentum": (0.8, 0.5)}
        assert cf.distance == 0.0
        assert cf.feasibility_score == 1.0

    def test_create_counterfactual_full(self):
        """Test creating a counterfactual with all fields."""
        cf = CounterfactualExplanation(
            original_decision="SELL",
            alternative_decision="BUY",
            feature_changes={
                "price_trend": (0.3, 0.8),
                "volume": (0.4, 0.6),
            },
            distance=0.35,
            feasibility_score=0.65,
        )
        assert cf.original_decision == "SELL"
        assert len(cf.feature_changes) == 2
        assert cf.distance == 0.35
        assert cf.feasibility_score == 0.65

    def test_counterfactual_feasibility_bounds(self):
        """Test counterfactual with edge case feasibility scores."""
        cf_zero = CounterfactualExplanation(
            original_decision="BUY",
            alternative_decision="SELL",
            feature_changes={"x": (0.0, 1.0)},
            feasibility_score=0.0,
        )
        assert cf_zero.feasibility_score == 0.0

        cf_one = CounterfactualExplanation(
            original_decision="HOLD",
            alternative_decision="BUY",
            feature_changes={"x": (0.5, 0.51)},
            feasibility_score=1.0,
        )
        assert cf_one.feasibility_score == 1.0

    def test_summary_property_increase(self):
        """Test summary property with increasing change."""
        cf = CounterfactualExplanation(
            original_decision="BUY",
            alternative_decision="HOLD",
            feature_changes={"momentum": (0.3, 0.6)},
        )
        summary = cf.summary
        assert "BUY" in summary
        assert "HOLD" in summary
        assert "increase" in summary
        assert "momentum" in summary

    def test_summary_property_decrease(self):
        """Test summary property with decreasing change."""
        cf = CounterfactualExplanation(
            original_decision="SELL",
            alternative_decision="HOLD",
            feature_changes={"volatility": (0.8, 0.5)},
        )
        summary = cf.summary
        assert "decrease" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cf = CounterfactualExplanation(
            original_decision="BUY",
            alternative_decision="HOLD",
            feature_changes={"momentum": (0.8, 0.5)},
            distance=0.3,
            feasibility_score=0.7,
        )
        d = cf.to_dict()

        assert d["original_decision"] == "BUY"
        assert d["alternative_decision"] == "HOLD"
        assert d["feature_changes"]["momentum"]["original"] == 0.8
        assert d["feature_changes"]["momentum"]["counterfactual"] == 0.5
        assert d["distance"] == 0.3
        assert d["feasibility_score"] == 0.7


# =============================================================================
# Test DecisionExplanation Dataclass
# =============================================================================


class TestDecisionExplanation:
    """Tests for DecisionExplanation dataclass."""

    def test_create_decision_explanation_minimal(self):
        """Test creating a decision explanation with minimal data."""
        exp = DecisionExplanation(
            decision_id="dec_001",
            timestamp=datetime.now(timezone.utc),
            action="BUY",
            symbol="BTCUSDT",
            position_size=0.5,
            confidence=0.85,
        )
        assert exp.decision_id == "dec_001"
        assert exp.action == "BUY"
        assert exp.symbol == "BTCUSDT"
        assert exp.position_size == 0.5
        assert exp.confidence == 0.85
        assert exp.uncertainty == 0.0
        assert exp.explanation_type == ExplanationType.FEATURE_ATTRIBUTION
        assert exp.feature_contributions == []
        assert exp.counterfactuals == []
        assert exp.summary != ""  # Auto-generated
        assert exp.regulatory_text != ""  # Auto-generated

    def test_create_with_feature_contributions(self):
        """Test creating with feature contributions."""
        features = [
            FeatureContribution("momentum", 0.8, 0.4),
            FeatureContribution("volume", 0.6, 0.2),
        ]
        exp = DecisionExplanation(
            decision_id="dec_002",
            timestamp=datetime.now(timezone.utc),
            action="BUY",
            symbol="ETHUSDT",
            position_size=0.3,
            confidence=0.75,
            feature_contributions=features,
        )
        assert len(exp.feature_contributions) == 2
        assert exp.feature_contributions[0].feature_name == "momentum"

    def test_create_with_counterfactuals(self):
        """Test creating with counterfactual explanations."""
        cf = CounterfactualExplanation(
            original_decision="BUY",
            alternative_decision="HOLD",
            feature_changes={"momentum": (0.8, 0.4)},
        )
        exp = DecisionExplanation(
            decision_id="dec_003",
            timestamp=datetime.now(timezone.utc),
            action="BUY",
            symbol="BTCUSDT",
            position_size=0.5,
            confidence=0.8,
            counterfactuals=[cf],
        )
        assert len(exp.counterfactuals) == 1

    def test_create_with_risk_factors(self):
        """Test creating with risk factors."""
        exp = DecisionExplanation(
            decision_id="dec_004",
            timestamp=datetime.now(timezone.utc),
            action="SELL",
            symbol="ETHUSDT",
            position_size=0.2,
            confidence=0.7,
            risk_factors={"volatility_status": "HIGH", "risk_budget_status": "NEAR_LIMIT"},
        )
        assert exp.risk_factors["volatility_status"] == "HIGH"

    def test_generate_summary_no_contributions(self):
        """Test auto-generated summary without contributions."""
        exp = DecisionExplanation(
            decision_id="dec_005",
            timestamp=datetime.now(timezone.utc),
            action="HOLD",
            symbol="BTCUSDT",
            position_size=0.0,
            confidence=0.6,
        )
        assert "HOLD" in exp.summary
        assert "BTCUSDT" in exp.summary
        assert "60" in exp.summary  # 60%

    def test_generate_summary_with_contributions(self):
        """Test auto-generated summary with contributions."""
        features = [
            FeatureContribution("momentum", 0.8, 0.4),  # positive
            FeatureContribution("volatility", 0.9, -0.3),  # negative
        ]
        exp = DecisionExplanation(
            decision_id="dec_006",
            timestamp=datetime.now(timezone.utc),
            action="BUY",
            symbol="ETHUSDT",
            position_size=0.5,
            confidence=0.85,
            feature_contributions=features,
        )
        assert "Supporting factors" in exp.summary or "momentum" in exp.summary
        assert "Opposing factors" in exp.summary or "volatility" in exp.summary

    def test_generate_regulatory_text(self):
        """Test auto-generated regulatory text."""
        exp = DecisionExplanation(
            decision_id="dec_007",
            timestamp=datetime.now(timezone.utc),
            action="BUY",
            symbol="BTCUSDT",
            position_size=0.5,
            confidence=0.85,
            feature_contributions=[FeatureContribution("momentum", 0.8, 0.4)],
        )
        text = exp.regulatory_text

        assert "AI Trading Decision Explanation" in text
        assert "dec_007" in text
        assert "BUY" in text
        assert "BTCUSDT" in text
        assert "85" in text  # 85%
        assert "EU AI Act Article 13" in text

    def test_regulatory_text_with_risk_factors(self):
        """Test regulatory text includes risk factors."""
        exp = DecisionExplanation(
            decision_id="dec_008",
            timestamp=datetime.now(timezone.utc),
            action="SELL",
            symbol="ETHUSDT",
            position_size=0.3,
            confidence=0.7,
            risk_factors={"volatility_status": "HIGH"},
        )
        assert "RISK ASSESSMENT" in exp.regulatory_text
        assert "volatility_status" in exp.regulatory_text

    def test_regulatory_text_with_counterfactuals(self):
        """Test regulatory text includes counterfactuals."""
        cf = CounterfactualExplanation(
            original_decision="BUY",
            alternative_decision="HOLD",
            feature_changes={"momentum": (0.8, 0.4)},
        )
        exp = DecisionExplanation(
            decision_id="dec_009",
            timestamp=datetime.now(timezone.utc),
            action="BUY",
            symbol="BTCUSDT",
            position_size=0.5,
            confidence=0.8,
            counterfactuals=[cf],
        )
        assert "ALTERNATIVE SCENARIOS" in exp.regulatory_text

    def test_to_dict(self):
        """Test conversion to dictionary."""
        exp = DecisionExplanation(
            decision_id="dec_010",
            timestamp=datetime.now(timezone.utc),
            action="BUY",
            symbol="BTCUSDT",
            position_size=0.5,
            confidence=0.8,
        )
        d = exp.to_dict()

        assert d["decision_id"] == "dec_010"
        assert d["action"] == "BUY"
        assert d["symbol"] == "BTCUSDT"
        assert d["position_size"] == 0.5
        assert d["confidence"] == 0.8
        assert d["explanation_type"] == "feature_attribution"
        assert "timestamp" in d
        assert "summary" in d
        assert "regulatory_text" in d

    def test_top_factors(self):
        """Test top_factors field."""
        exp = DecisionExplanation(
            decision_id="dec_011",
            timestamp=datetime.now(timezone.utc),
            action="BUY",
            symbol="BTCUSDT",
            position_size=0.5,
            confidence=0.8,
            top_factors=["momentum", "volume", "rsi"],
        )
        assert exp.top_factors == ["momentum", "volume", "rsi"]


# =============================================================================
# Test DecisionExplainer Class
# =============================================================================


class TestDecisionExplainerCreation:
    """Tests for DecisionExplainer instantiation."""

    def test_create_explainer_default(self):
        """Test creating an explainer with default settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))
            assert explainer._storage_path == Path(tmpdir)
            assert len(explainer._feature_weights) > 0

    def test_create_explainer_custom_weights(self):
        """Test creating an explainer with custom weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_weights = {"custom_feature": 0.5, "another_feature": 0.3}
            explainer = DecisionExplainer(storage_path=Path(tmpdir), feature_weights=custom_weights)
            assert "custom_feature" in explainer._feature_weights
            assert explainer._feature_weights["custom_feature"] == 0.5

    def test_factory_function(self):
        """Test the create_decision_explainer factory function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = create_decision_explainer(
                storage_path=Path(tmpdir), feature_weights={"test": 0.1}
            )
            assert isinstance(explainer, DecisionExplainer)

    def test_storage_directory_created(self):
        """Test that storage directory is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = Path(tmpdir) / "nested" / "explainability"
            explainer = DecisionExplainer(storage_path=new_path)
            assert new_path.exists()

    def test_default_feature_weights(self):
        """Test default feature weights are set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))
            # Should have standard trading features
            assert "price_momentum" in explainer._feature_weights
            assert "rsi" in explainer._feature_weights
            assert "volatility_regime" in explainer._feature_weights


class TestDecisionExplainerExplainDecision:
    """Tests for DecisionExplainer.explain_decision method."""

    def test_explain_decision_basic(self):
        """Test basic decision explanation generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-001",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"momentum": 0.8, "volume": 0.6},
                confidence=0.85,
            )

            assert explanation.decision_id == "DEC-001"
            assert explanation.action == "BUY"
            assert explanation.symbol == "BTCUSDT"
            assert explanation.position_size == 0.5
            assert explanation.confidence == 0.85

    def test_explain_decision_generates_contributions(self):
        """Test that feature contributions are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-002",
                action="BUY",
                symbol="ETHUSDT",
                position_size=0.3,
                features={"price_momentum": 0.8, "volume_ratio": 0.6, "rsi": 0.5},
                confidence=0.75,
            )

            assert len(explanation.feature_contributions) > 0
            # Contributions should be sorted by importance
            assert explanation.feature_contributions[0].importance_rank == 1

    def test_explain_decision_generates_counterfactual_for_buy(self):
        """Test counterfactual generation for BUY action."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-003",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"price_momentum": 0.8},
                confidence=0.85,
            )

            # BUY should generate counterfactual to HOLD
            if explanation.counterfactuals:
                assert explanation.counterfactuals[0].alternative_decision == "HOLD"

    def test_explain_decision_no_counterfactual_for_hold(self):
        """Test no counterfactual generation for HOLD action."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-004",
                action="HOLD",
                symbol="BTCUSDT",
                position_size=0.0,
                features={"price_momentum": 0.5},
                confidence=0.6,
            )

            assert len(explanation.counterfactuals) == 0

    def test_explain_decision_with_risk_state(self):
        """Test explanation with risk state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-005",
                action="SELL",
                symbol="ETHUSDT",
                position_size=0.2,
                features={"volatility_regime": 0.8, "risk_utilization": 0.9},
                confidence=0.7,
                risk_state={"utilization": 0.85, "drawdown": 0.05, "position_count": 3},
            )

            assert (
                "risk_utilization" in explanation.risk_factors
                or "volatility_status" in explanation.risk_factors
            )

    def test_explain_decision_with_model_outputs(self):
        """Test explanation with model outputs (attributions)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-006",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"momentum": 0.8, "volume": 0.6},
                confidence=0.85,
                model_outputs={"feature_attributions": {"momentum": 0.5, "volume": 0.2}},
            )

            # Should use provided attributions
            assert len(explanation.feature_contributions) >= 2

    def test_explain_decision_stored_in_history(self):
        """Test that explanations are stored in history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explainer.explain_decision(
                decision_id="DEC-007",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"momentum": 0.7},
                confidence=0.8,
            )

            history = explainer.get_recent_explanations()
            assert len(history) == 1
            assert history[0].decision_id == "DEC-007"

    def test_explain_decision_persisted_to_disk(self):
        """Test that explanations are persisted to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            explainer = DecisionExplainer(storage_path=storage_path)

            explanation = explainer.explain_decision(
                decision_id="DEC-008",
                action="SELL",
                symbol="ETHUSDT",
                position_size=0.3,
                features={"momentum": -0.5},
                confidence=0.75,
            )

            # Check file was created
            date_str = explanation.timestamp.strftime("%Y%m%d")
            expected_file = storage_path / f"explanations_{date_str}.jsonl"
            assert expected_file.exists()

    def test_explain_decision_with_uncertainty(self):
        """Test explanation with uncertainty value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-009",
                action="HOLD",
                symbol="BTCUSDT",
                position_size=0.0,
                features={"momentum": 0.5},
                confidence=0.6,
                uncertainty=0.15,
            )

            assert explanation.uncertainty == 0.15


class TestDecisionExplainerHelperMethods:
    """Tests for helper methods."""

    def test_update_feature_weights(self):
        """Test updating feature weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            new_weights = {"new_feature": 0.5, "price_momentum": 0.25}
            explainer.update_feature_weights(new_weights)

            assert explainer._feature_weights["new_feature"] == 0.5
            assert explainer._feature_weights["price_momentum"] == 0.25

    def test_get_explanation_by_id(self):
        """Test getting explanation by ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explainer.explain_decision(
                decision_id="DEC-FIND",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"momentum": 0.7},
                confidence=0.8,
            )

            found = explainer.get_explanation("DEC-FIND")
            assert found is not None
            assert found.decision_id == "DEC-FIND"

    def test_get_explanation_not_found(self):
        """Test getting non-existent explanation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            found = explainer.get_explanation("DOES-NOT-EXIST")
            assert found is None

    def test_get_recent_explanations_empty(self):
        """Test getting recent explanations when empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            history = explainer.get_recent_explanations()
            assert history == []

    def test_get_recent_explanations_with_limit(self):
        """Test getting recent explanations with limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            # Generate 10 explanations
            for i in range(10):
                explainer.explain_decision(
                    decision_id=f"DEC-{i:03d}",
                    action="BUY" if i % 2 == 0 else "SELL",
                    symbol="BTCUSDT",
                    position_size=0.5,
                    features={"momentum": 0.5 + i * 0.05},
                    confidence=0.7,
                )

            # Get last 3
            history = explainer.get_recent_explanations(limit=3)
            assert len(history) == 3
            assert history[-1].decision_id == "DEC-009"


class TestDecisionExplainerStatistics:
    """Tests for statistics methods."""

    def test_get_statistics_empty(self):
        """Test statistics when no explanations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            stats = explainer.get_explanation_statistics()

            assert stats["total_explanations"] == 0
            assert stats["average_confidence"] == 0.0
            assert stats["action_distribution"] == {}

    def test_get_statistics_with_data(self):
        """Test statistics with explanations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            # Generate explanations
            for i in range(5):
                explainer.explain_decision(
                    decision_id=f"DEC-{i}",
                    action="BUY" if i < 3 else "SELL",
                    symbol="BTCUSDT",
                    position_size=0.5,
                    features={"price_momentum": 0.5 + i * 0.1},
                    confidence=0.7 + i * 0.05,
                )

            stats = explainer.get_explanation_statistics()

            assert stats["total_explanations"] == 5
            assert stats["average_confidence"] > 0
            assert stats["action_distribution"]["BUY"] == 3
            assert stats["action_distribution"]["SELL"] == 2
            assert "top_influential_features" in stats


class TestDecisionExplainerExport:
    """Tests for export functionality."""

    def test_export_explanations_default_path(self):
        """Test exporting explanations with default path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            explainer = DecisionExplainer(storage_path=storage_path)

            explainer.explain_decision(
                decision_id="DEC-001",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"momentum": 0.7},
                confidence=0.8,
            )

            output_path = explainer.export_explanations()

            assert output_path.exists()
            with open(output_path) as f:
                data = json.load(f)
                assert data["total_explanations"] == 1
                assert len(data["explanations"]) == 1

    def test_export_explanations_custom_path(self):
        """Test exporting explanations with custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            explainer = DecisionExplainer(storage_path=storage_path)

            explainer.explain_decision(
                decision_id="DEC-001",
                action="SELL",
                symbol="ETHUSDT",
                position_size=0.3,
                features={"momentum": -0.5},
                confidence=0.75,
            )

            custom_path = storage_path / "custom_export.json"
            output_path = explainer.export_explanations(output_path=custom_path)

            assert output_path == custom_path
            assert custom_path.exists()

    def test_export_explanations_with_time_filter(self):
        """Test exporting with time filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            explainer = DecisionExplainer(storage_path=storage_path)

            # Generate explanation
            explainer.explain_decision(
                decision_id="DEC-001",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"momentum": 0.7},
                confidence=0.8,
            )

            # Export with time filter
            start_time = datetime.now(timezone.utc)
            output_path = explainer.export_explanations(start_time=start_time)

            # May or may not include the explanation depending on timing
            assert output_path.exists()

    def test_export_structure(self):
        """Test export has required structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            explainer = DecisionExplainer(storage_path=storage_path)

            explainer.explain_decision(
                decision_id="DEC-001",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"momentum": 0.7},
                confidence=0.8,
            )

            output_path = explainer.export_explanations()

            with open(output_path) as f:
                data = json.load(f)

                assert "export_timestamp" in data
                assert "ai_act_reference" in data
                assert "Article 13" in data["ai_act_reference"]
                assert "total_explanations" in data
                assert "statistics" in data
                assert "explanations" in data


# =============================================================================
# Test Thread Safety
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe operation of DecisionExplainer."""

    def test_concurrent_explanations(self):
        """Test generating explanations from multiple threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))
            results = []
            errors = []

            def generate_explanation(idx):
                try:
                    exp = explainer.explain_decision(
                        decision_id=f"DEC-{idx:03d}",
                        action="BUY" if idx % 2 == 0 else "SELL",
                        symbol="BTCUSDT",
                        position_size=0.5,
                        features={"feature": float(idx) / 10},
                        confidence=0.5 + (idx % 5) * 0.1,
                    )
                    results.append(exp)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=generate_explanation, args=(i,)) for i in range(20)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 20

    def test_concurrent_history_access(self):
        """Test accessing history from multiple threads while generating."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))
            errors = []

            def generate_and_read(idx):
                try:
                    explainer.explain_decision(
                        decision_id=f"DEC-{idx:03d}",
                        action="BUY",
                        symbol="BTCUSDT",
                        position_size=0.5,
                        features={"x": float(idx)},
                        confidence=0.7,
                    )
                    _ = explainer.get_recent_explanations()
                    _ = explainer.get_explanation_statistics()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=generate_and_read, args=(i,)) for i in range(15)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0

    def test_concurrent_weight_updates(self):
        """Test updating weights from multiple threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))
            errors = []

            def update_weights(idx):
                try:
                    explainer.update_feature_weights({f"feature_{idx}": 0.1 * idx})
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=update_weights, args=(i,)) for i in range(10)]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0


# =============================================================================
# Test Risk Factor Extraction
# =============================================================================


class TestRiskFactorExtraction:
    """Tests for risk factor extraction."""

    def test_extract_volatility_high(self):
        """Test volatility status extraction - HIGH."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-001",
                action="SELL",
                symbol="BTCUSDT",
                position_size=0.3,
                features={"volatility_regime": 0.8},
                confidence=0.7,
            )

            assert explanation.risk_factors.get("volatility_status") == "HIGH"

    def test_extract_volatility_moderate(self):
        """Test volatility status extraction - MODERATE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-002",
                action="HOLD",
                symbol="BTCUSDT",
                position_size=0.0,
                features={"volatility_regime": 0.5},
                confidence=0.6,
            )

            assert explanation.risk_factors.get("volatility_status") == "MODERATE"

    def test_extract_volatility_low(self):
        """Test volatility status extraction - LOW."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-003",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"volatility_regime": 0.3},
                confidence=0.8,
            )

            assert explanation.risk_factors.get("volatility_status") == "LOW"

    def test_extract_risk_budget_near_limit(self):
        """Test risk budget status - NEAR_LIMIT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-004",
                action="HOLD",
                symbol="BTCUSDT",
                position_size=0.0,
                features={"risk_utilization": 0.85},
                confidence=0.6,
            )

            assert explanation.risk_factors.get("risk_budget_status") == "NEAR_LIMIT"

    def test_extract_risk_budget_moderate(self):
        """Test risk budget status - MODERATE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-005",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.3,
                features={"risk_utilization": 0.6},
                confidence=0.75,
            )

            assert explanation.risk_factors.get("risk_budget_status") == "MODERATE"

    def test_extract_risk_budget_comfortable(self):
        """Test risk budget status - COMFORTABLE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-006",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"risk_utilization": 0.3},
                confidence=0.85,
            )

            assert explanation.risk_factors.get("risk_budget_status") == "COMFORTABLE"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_features(self):
        """Test explanation with empty features dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-001",
                action="HOLD",
                symbol="BTCUSDT",
                position_size=0.0,
                features={},
                confidence=0.5,
            )

            assert explanation.action == "HOLD"
            assert len(explanation.feature_contributions) == 0

    def test_very_many_features(self):
        """Test explanation with many features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            features = {f"feature_{i}": 0.5 + (i % 10) * 0.05 for i in range(100)}

            explanation = explainer.explain_decision(
                decision_id="DEC-002",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features=features,
                confidence=0.8,
            )

            assert len(explanation.feature_contributions) == 100

    def test_confidence_bounds(self):
        """Test explanations with extreme confidence values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            exp_low = explainer.explain_decision(
                decision_id="DEC-003",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.1,
                features={"x": 0.5},
                confidence=0.0,
            )
            assert exp_low.confidence == 0.0

            exp_high = explainer.explain_decision(
                decision_id="DEC-004",
                action="SELL",
                symbol="ETHUSDT",
                position_size=0.9,
                features={"x": 0.5},
                confidence=1.0,
            )
            assert exp_high.confidence == 1.0

    def test_special_characters_in_symbol(self):
        """Test handling of special characters in symbol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-005",
                action="BUY",
                symbol="BTC/USDT",  # With slash
                position_size=0.5,
                features={"momentum": 0.7},
                confidence=0.8,
            )

            assert explanation.symbol == "BTC/USDT"


# =============================================================================
# Test Compliance Integration
# =============================================================================


class TestComplianceIntegration:
    """Tests for EU AI Act compliance integration."""

    def test_explanation_supports_article_13(self):
        """Test that explanations support Article 13 transparency requirements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-001",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"momentum": 0.7, "volume": 0.6},
                confidence=0.85,
            )

            # Article 13 requires transparency - explanation should provide:
            # 1. Summary of the decision
            assert explanation.summary != ""
            # 2. Feature contributions
            assert len(explanation.feature_contributions) > 0
            # 3. Regulatory text
            assert "Article 13" in explanation.regulatory_text

    def test_explanation_supports_article_14(self):
        """Test that explanations support Article 14 human oversight requirements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            explanation = explainer.explain_decision(
                decision_id="DEC-002",
                action="SELL",
                symbol="ETHUSDT",
                position_size=0.3,
                features={"momentum": -0.5, "trend": -0.3},
                confidence=0.75,
            )

            # Human-readable explanations for oversight
            assert explanation.summary != ""
            for fc in explanation.feature_contributions:
                assert fc.human_readable != ""

    def test_audit_trail_compliance(self):
        """Test that audit trail is maintained for compliance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir)
            explainer = DecisionExplainer(storage_path=storage_path)

            # Generate multiple decisions
            for i in range(5):
                explainer.explain_decision(
                    decision_id=f"DEC-{i:03d}",
                    action="BUY" if i % 2 == 0 else "HOLD",
                    symbol="BTCUSDT",
                    position_size=0.5 if i % 2 == 0 else 0.0,
                    features={"signal": float(i) / 10},
                    confidence=0.6 + i * 0.05,
                )

            # Should be able to export complete audit trail
            export_path = explainer.export_explanations()
            assert export_path.exists()

            with open(export_path) as f:
                data = json.load(f)
                assert data["total_explanations"] == 5
                assert "ai_act_reference" in data

    def test_regulatory_text_complete(self):
        """Test that regulatory text is complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            explainer = DecisionExplainer(storage_path=Path(tmpdir))

            cf = CounterfactualExplanation(
                original_decision="BUY",
                alternative_decision="HOLD",
                feature_changes={"momentum": (0.8, 0.4)},
            )

            explanation = explainer.explain_decision(
                decision_id="DEC-003",
                action="BUY",
                symbol="BTCUSDT",
                position_size=0.5,
                features={"volatility_regime": 0.8, "risk_utilization": 0.7},
                confidence=0.85,
                risk_state={"utilization": 0.7, "drawdown": 0.03},
            )

            text = explanation.regulatory_text

            # Should have all required sections
            assert "AI Trading Decision Explanation" in text
            assert "ACTION:" in text
            assert "DECISION FACTORS:" in text
            # Risk factors should be included if present
            assert "RISK ASSESSMENT" in text or "volatility_status" in text
