# -*- coding: utf-8 -*-
"""
AI Act Decision Explainability Module.

EU AI Act Requirements for Transparency and Explainability.

This module provides decision explainability capabilities for algorithmic
trading systems, enabling operators to understand AI trading decisions
and meet regulatory transparency requirements.

EU AI Act Context:
    Article 13 (Transparency): High-risk AI systems shall be designed and
    developed in such a way to ensure that their operation is sufficiently
    transparent to enable users to interpret the system's output.

    Article 14 (Human Oversight): Requires that AI systems be understandable
    to enable effective human oversight.

Explainability Approaches:
    1. Feature Attribution: Which features influenced the decision
    2. Counterfactual Explanations: What would change the decision
    3. Rule Extraction: Simplified rules approximating the model
    4. Confidence and Uncertainty: How confident is the model

References:
    - EU AI Act Articles 13, 14
    - OECD AI Principles (Transparency)
    - Ribeiro et al. (2016) "Why Should I Trust You?" (LIME)
    - Lundberg & Lee (2017) "SHAP Values"

Example:
    >>> from services.ai_act.explainability import (
    ...     DecisionExplainer,
    ...     create_decision_explainer,
    ... )
    >>> explainer = create_decision_explainer()
    >>> explanation = explainer.explain_decision(
    ...     decision_id="DEC-001",
    ...     action="BUY",
    ...     features=feature_dict,
    ...     confidence=0.85,
    ... )
    >>> print(explanation.summary)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading


logger = logging.getLogger(__name__)


class ExplanationType(Enum):
    """Type of explanation provided."""

    FEATURE_ATTRIBUTION = "feature_attribution"
    """Explanation based on feature contributions."""

    COUNTERFACTUAL = "counterfactual"
    """Explanation showing what would change the decision."""

    RULE_BASED = "rule_based"
    """Simplified rule-based explanation."""

    CONFIDENCE_BASED = "confidence_based"
    """Explanation based on model confidence/uncertainty."""

    COMPARATIVE = "comparative"
    """Explanation comparing to similar historical decisions."""


@dataclass
class FeatureContribution:
    """
    Contribution of a single feature to a decision.

    Attributes:
        feature_name: Name of the feature.
        feature_value: Current value of the feature.
        contribution: Contribution to the decision (-1 to 1).
        direction: Direction of influence ("positive", "negative", "neutral").
        importance_rank: Rank by importance (1 = most important).
        human_readable: Human-readable description of contribution.
    """

    feature_name: str
    feature_value: float
    contribution: float
    direction: str = "neutral"
    importance_rank: int = 0
    human_readable: str = ""

    def __post_init__(self) -> None:
        """Set direction based on contribution."""
        if self.contribution > 0.05:
            self.direction = "positive"
        elif self.contribution < -0.05:
            self.direction = "negative"
        else:
            self.direction = "neutral"

        if not self.human_readable:
            self.human_readable = self._generate_description()

    def _generate_description(self) -> str:
        """Generate human-readable description."""
        strength = abs(self.contribution)
        if strength > 0.3:
            strength_word = "strongly"
        elif strength > 0.1:
            strength_word = "moderately"
        else:
            strength_word = "slightly"

        if self.direction == "positive":
            return f"{self.feature_name} ({self.feature_value:.4f}) {strength_word} supports the decision"
        elif self.direction == "negative":
            return f"{self.feature_name} ({self.feature_value:.4f}) {strength_word} opposes the decision"
        else:
            return f"{self.feature_name} ({self.feature_value:.4f}) has neutral influence"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "feature_value": self.feature_value,
            "contribution": self.contribution,
            "direction": self.direction,
            "importance_rank": self.importance_rank,
            "human_readable": self.human_readable,
        }


@dataclass
class CounterfactualExplanation:
    """
    Counterfactual explanation for a decision.

    Shows what minimal changes to inputs would lead to a different decision.

    Attributes:
        original_decision: The original decision made.
        alternative_decision: The alternative decision.
        feature_changes: Required changes to features.
        distance: How different the counterfactual is.
        feasibility_score: How feasible is this counterfactual (0-1).
    """

    original_decision: str
    alternative_decision: str
    feature_changes: Dict[str, Tuple[float, float]]  # feature: (original, counterfactual)
    distance: float = 0.0
    feasibility_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_decision": self.original_decision,
            "alternative_decision": self.alternative_decision,
            "feature_changes": {
                k: {"original": v[0], "counterfactual": v[1]}
                for k, v in self.feature_changes.items()
            },
            "distance": self.distance,
            "feasibility_score": self.feasibility_score,
        }

    @property
    def summary(self) -> str:
        """Generate human-readable summary."""
        changes = []
        for feature, (original, counterfactual) in self.feature_changes.items():
            direction = "increase" if counterfactual > original else "decrease"
            diff = abs(counterfactual - original)
            changes.append(f"{feature} {direction} by {diff:.4f}")

        return (
            f"To change from {self.original_decision} to {self.alternative_decision}, "
            f"would need: {', '.join(changes)}"
        )


@dataclass
class DecisionExplanation:
    """
    Complete explanation for a trading decision.

    Attributes:
        decision_id: Unique decision identifier.
        timestamp: When the decision was made.
        action: The action taken (BUY, SELL, HOLD).
        symbol: Trading symbol.
        position_size: Position size.
        confidence: Model confidence (0-1).
        uncertainty: Model uncertainty estimate.
        explanation_type: Primary explanation type.
        feature_contributions: Feature-level contributions.
        top_factors: Top contributing factors (summary).
        counterfactuals: Alternative scenarios.
        risk_factors: Risk-related factors.
        summary: Human-readable summary.
        regulatory_text: Regulatory-compliant explanation text.
    """

    decision_id: str
    timestamp: datetime
    action: str
    symbol: str
    position_size: float
    confidence: float
    uncertainty: float = 0.0
    explanation_type: ExplanationType = ExplanationType.FEATURE_ATTRIBUTION
    feature_contributions: List[FeatureContribution] = field(default_factory=list)
    top_factors: List[str] = field(default_factory=list)
    counterfactuals: List[CounterfactualExplanation] = field(default_factory=list)
    risk_factors: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    regulatory_text: str = ""

    def __post_init__(self) -> None:
        """Generate summary if not provided."""
        if not self.summary:
            self.summary = self._generate_summary()
        if not self.regulatory_text:
            self.regulatory_text = self._generate_regulatory_text()

    def _generate_summary(self) -> str:
        """Generate human-readable summary."""
        if not self.feature_contributions:
            return f"Decision: {self.action} {self.symbol} with confidence {self.confidence:.1%}"

        # Get top 3 positive and negative factors
        positive = [
            fc for fc in self.feature_contributions
            if fc.direction == "positive"
        ][:3]
        negative = [
            fc for fc in self.feature_contributions
            if fc.direction == "negative"
        ][:3]

        parts = [f"Decision: {self.action} {self.symbol}"]
        parts.append(f"Confidence: {self.confidence:.1%}")

        if positive:
            supporting = ", ".join(fc.feature_name for fc in positive)
            parts.append(f"Supporting factors: {supporting}")

        if negative:
            opposing = ", ".join(fc.feature_name for fc in negative)
            parts.append(f"Opposing factors: {opposing}")

        return ". ".join(parts) + "."

    def _generate_regulatory_text(self) -> str:
        """Generate EU AI Act-aligned explanation text."""
        lines = [
            "=== AI Trading Decision Explanation ===",
            f"Decision ID: {self.decision_id}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"",
            f"ACTION: {self.action}",
            f"Symbol: {self.symbol}",
            f"Position Size: {self.position_size:.4f}",
            f"Model Confidence: {self.confidence:.2%}",
            f"Uncertainty Estimate: {self.uncertainty:.2%}",
            f"",
            "DECISION FACTORS:",
        ]

        for i, fc in enumerate(self.feature_contributions[:10], 1):
            lines.append(f"  {i}. {fc.human_readable}")

        if self.risk_factors:
            lines.append("")
            lines.append("RISK ASSESSMENT:")
            for key, value in self.risk_factors.items():
                lines.append(f"  - {key}: {value}")

        if self.counterfactuals:
            lines.append("")
            lines.append("ALTERNATIVE SCENARIOS:")
            for cf in self.counterfactuals[:3]:
                lines.append(f"  - {cf.summary}")

        lines.append("")
        lines.append("This explanation is provided in compliance with EU AI Act Article 13")
        lines.append("(Transparency and provision of information to users).")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "symbol": self.symbol,
            "position_size": self.position_size,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "explanation_type": self.explanation_type.value,
            "feature_contributions": [fc.to_dict() for fc in self.feature_contributions],
            "top_factors": self.top_factors,
            "counterfactuals": [cf.to_dict() for cf in self.counterfactuals],
            "risk_factors": self.risk_factors,
            "summary": self.summary,
            "regulatory_text": self.regulatory_text,
        }


class DecisionExplainer:
    """
    Decision Explainability Engine for AI Trading System.

    Provides multiple explanation methods for trading decisions:
    1. Feature attribution (LIME/SHAP-style)
    2. Counterfactual explanations
    3. Rule extraction
    4. Confidence/uncertainty explanations

    Thread-safe for production use.

    Attributes:
        storage_path: Path for explanation storage.
        feature_metadata: Metadata about features.

    Example:
        >>> explainer = DecisionExplainer()
        >>> explanation = explainer.explain_decision(
        ...     decision_id="DEC-001",
        ...     action="BUY",
        ...     symbol="BTCUSDT",
        ...     position_size=0.5,
        ...     features={"momentum": 0.8, "volatility": 0.3},
        ...     confidence=0.85,
        ... )
    """

    # Default feature importance weights (can be updated from model)
    DEFAULT_FEATURE_WEIGHTS = {
        # Price features
        "price_momentum": 0.15,
        "price_return_1h": 0.10,
        "price_return_4h": 0.08,
        "price_return_24h": 0.05,
        # Volume features
        "volume_ratio": 0.08,
        "volume_momentum": 0.05,
        # Volatility features
        "volatility_regime": 0.10,
        "atr_normalized": 0.05,
        "bb_position": 0.04,
        # Momentum indicators
        "rsi": 0.08,
        "macd_signal": 0.06,
        # Risk features
        "risk_utilization": 0.08,
        "drawdown_current": 0.05,
        # External features
        "fear_greed_index": 0.03,
    }

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        feature_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize DecisionExplainer.

        Args:
            storage_path: Path for storing explanations.
            feature_weights: Custom feature importance weights.
        """
        self._lock = threading.RLock()  # Reentrant lock to allow nested calls

        if storage_path is None:
            storage_path = Path("logs/ai_act/explainability")
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._feature_weights = feature_weights or self.DEFAULT_FEATURE_WEIGHTS.copy()
        self._explanation_history: List[DecisionExplanation] = []

        # Feature metadata for human-readable names
        self._feature_metadata = {
            "price_momentum": {
                "display_name": "Price Momentum",
                "description": "Short-term price trend strength",
                "good_range": (0.3, 0.7),
            },
            "volatility_regime": {
                "display_name": "Volatility Regime",
                "description": "Current market volatility state",
                "good_range": (0.2, 0.5),
            },
            "rsi": {
                "display_name": "RSI",
                "description": "Relative Strength Index",
                "good_range": (30, 70),
            },
            "risk_utilization": {
                "display_name": "Risk Utilization",
                "description": "Current risk budget usage",
                "good_range": (0.0, 0.8),
            },
        }

        logger.info(f"DecisionExplainer initialized at {storage_path}")

    def update_feature_weights(self, weights: Dict[str, float]) -> None:
        """Update feature importance weights."""
        with self._lock:
            self._feature_weights.update(weights)
            logger.info(f"Updated {len(weights)} feature weights")

    def explain_decision(
        self,
        decision_id: str,
        action: str,
        symbol: str,
        position_size: float,
        features: Dict[str, float],
        confidence: float,
        uncertainty: float = 0.0,
        risk_state: Optional[Dict[str, Any]] = None,
        model_outputs: Optional[Dict[str, Any]] = None,
    ) -> DecisionExplanation:
        """
        Generate comprehensive explanation for a trading decision.

        Args:
            decision_id: Unique decision identifier.
            action: Trading action (BUY, SELL, HOLD).
            symbol: Trading symbol.
            position_size: Position size (0-1).
            features: Feature values used in decision.
            confidence: Model confidence (0-1).
            uncertainty: Uncertainty estimate.
            risk_state: Current risk management state.
            model_outputs: Raw model outputs if available.

        Returns:
            DecisionExplanation with full analysis.
        """
        timestamp = datetime.now(timezone.utc)

        # Calculate feature contributions
        contributions = self._calculate_feature_contributions(
            features, action, model_outputs
        )

        # Extract top factors
        top_factors = [
            fc.feature_name for fc in contributions[:5]
        ]

        # Generate counterfactual if not HOLD
        counterfactuals = []
        if action != "HOLD":
            cf = self._generate_counterfactual(
                action, features, position_size, contributions
            )
            if cf:
                counterfactuals.append(cf)

        # Extract risk factors
        risk_factors = self._extract_risk_factors(risk_state, features)

        explanation = DecisionExplanation(
            decision_id=decision_id,
            timestamp=timestamp,
            action=action,
            symbol=symbol,
            position_size=position_size,
            confidence=confidence,
            uncertainty=uncertainty,
            explanation_type=ExplanationType.FEATURE_ATTRIBUTION,
            feature_contributions=contributions,
            top_factors=top_factors,
            counterfactuals=counterfactuals,
            risk_factors=risk_factors,
        )

        # Store explanation
        with self._lock:
            self._explanation_history.append(explanation)

        # Persist to disk
        self._persist_explanation(explanation)

        logger.info(
            f"Generated explanation for {decision_id}: {action} {symbol} "
            f"(confidence={confidence:.2%})"
        )

        return explanation

    def _calculate_feature_contributions(
        self,
        features: Dict[str, float],
        action: str,
        model_outputs: Optional[Dict[str, Any]] = None,
    ) -> List[FeatureContribution]:
        """Calculate contribution of each feature to the decision."""
        contributions = []

        # If model provides gradients/attributions, use them
        if model_outputs and "feature_attributions" in model_outputs:
            attributions = model_outputs["feature_attributions"]
        else:
            # Otherwise, use weighted heuristic approach
            attributions = {}
            for feature, value in features.items():
                weight = self._feature_weights.get(feature, 0.01)
                # Simple heuristic: contribution = weight * normalized_value * action_sign
                action_sign = 1.0 if action == "BUY" else (-1.0 if action == "SELL" else 0.0)
                normalized_value = (value - 0.5) * 2  # Assume features normalized to [0,1]
                attributions[feature] = weight * normalized_value * action_sign

        for feature, attribution in attributions.items():
            feature_value = features.get(feature, 0.0)
            fc = FeatureContribution(
                feature_name=feature,
                feature_value=feature_value,
                contribution=attribution,
            )
            contributions.append(fc)

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Assign ranks
        for i, fc in enumerate(contributions):
            fc.importance_rank = i + 1

        return contributions

    def _generate_counterfactual(
        self,
        action: str,
        features: Dict[str, float],
        position_size: float,
        contributions: List[FeatureContribution],
    ) -> Optional[CounterfactualExplanation]:
        """Generate counterfactual explanation."""
        alternative = "HOLD" if action in ("BUY", "SELL") else "BUY"

        # Find features that would need to change
        # Focus on top contributing features
        changes = {}
        distance = 0.0

        for fc in contributions[:3]:
            if abs(fc.contribution) < 0.05:
                continue

            feature_name = fc.feature_name
            original_value = fc.feature_value

            # Calculate counterfactual value (flip direction)
            if fc.direction == "positive":
                cf_value = max(0.0, original_value - 0.3)
            else:
                cf_value = min(1.0, original_value + 0.3)

            changes[feature_name] = (original_value, cf_value)
            distance += abs(cf_value - original_value)

        if not changes:
            return None

        return CounterfactualExplanation(
            original_decision=action,
            alternative_decision=alternative,
            feature_changes=changes,
            distance=distance,
            feasibility_score=max(0.0, 1.0 - distance / len(changes)),
        )

    def _extract_risk_factors(
        self,
        risk_state: Optional[Dict[str, Any]],
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """Extract risk-relevant factors for explanation."""
        risk_factors = {}

        if risk_state:
            risk_factors["risk_utilization"] = risk_state.get("utilization", 0.0)
            risk_factors["current_drawdown"] = risk_state.get("drawdown", 0.0)
            risk_factors["position_count"] = risk_state.get("position_count", 0)

        # Extract from features
        if "volatility_regime" in features:
            vol = features["volatility_regime"]
            if vol > 0.7:
                risk_factors["volatility_status"] = "HIGH"
            elif vol > 0.4:
                risk_factors["volatility_status"] = "MODERATE"
            else:
                risk_factors["volatility_status"] = "LOW"

        if "risk_utilization" in features:
            util = features["risk_utilization"]
            if util > 0.8:
                risk_factors["risk_budget_status"] = "NEAR_LIMIT"
            elif util > 0.5:
                risk_factors["risk_budget_status"] = "MODERATE"
            else:
                risk_factors["risk_budget_status"] = "COMFORTABLE"

        return risk_factors

    def _persist_explanation(self, explanation: DecisionExplanation) -> None:
        """Persist explanation to disk."""
        date_str = explanation.timestamp.strftime("%Y%m%d")
        daily_file = self._storage_path / f"explanations_{date_str}.jsonl"

        with open(daily_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(explanation.to_dict()) + "\n")

    def get_explanation(self, decision_id: str) -> Optional[DecisionExplanation]:
        """Get explanation by decision ID."""
        with self._lock:
            for exp in self._explanation_history:
                if exp.decision_id == decision_id:
                    return exp
        return None

    def get_recent_explanations(self, limit: int = 100) -> List[DecisionExplanation]:
        """Get recent explanations."""
        with self._lock:
            return self._explanation_history[-limit:]

    def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get statistics about explanations."""
        with self._lock:
            if not self._explanation_history:
                return {
                    "total_explanations": 0,
                    "average_confidence": 0.0,
                    "action_distribution": {},
                }

            explanations = self._explanation_history

            action_counts = {}
            total_confidence = 0.0
            top_features = {}

            for exp in explanations:
                action_counts[exp.action] = action_counts.get(exp.action, 0) + 1
                total_confidence += exp.confidence

                for fc in exp.feature_contributions[:5]:
                    top_features[fc.feature_name] = (
                        top_features.get(fc.feature_name, 0) + 1
                    )

            return {
                "total_explanations": len(explanations),
                "average_confidence": total_confidence / len(explanations),
                "action_distribution": action_counts,
                "top_influential_features": dict(
                    sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:10]
                ),
            }

    def export_explanations(
        self,
        output_path: Optional[Path] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Path:
        """
        Export explanations for regulatory audit.

        Args:
            output_path: Output file path.
            start_time: Filter start time.
            end_time: Filter end time.

        Returns:
            Path to exported file.
        """
        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = self._storage_path / f"explanations_export_{timestamp}.json"

        with self._lock:
            explanations = self._explanation_history

            if start_time:
                explanations = [e for e in explanations if e.timestamp >= start_time]
            if end_time:
                explanations = [e for e in explanations if e.timestamp <= end_time]

            export_data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "ai_act_reference": "Article 13 - Transparency",
                "total_explanations": len(explanations),
                "statistics": self.get_explanation_statistics(),
                "explanations": [e.to_dict() for e in explanations],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(explanations)} explanations to {output_path}")
        return output_path


def create_decision_explainer(
    storage_path: Optional[Path] = None,
    feature_weights: Optional[Dict[str, float]] = None,
) -> DecisionExplainer:
    """
    Factory function to create DecisionExplainer.

    Args:
        storage_path: Path for storing explanations.
        feature_weights: Custom feature importance weights.

    Returns:
        Configured DecisionExplainer instance.
    """
    return DecisionExplainer(
        storage_path=storage_path,
        feature_weights=feature_weights,
    )
