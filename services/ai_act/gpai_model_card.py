# -*- coding: utf-8 -*-
"""
Article 53(1)(b) EU AI Act - GPAI Model Card.

This module implements the GPAI Model Card requirements for downstream
providers as mandated by Article 53(1)(b) of the EU AI Act.

Key Requirements:
- Article 53(1)(b): Provide information for downstream providers to comply
  with their own obligations under the AI Act
- GPAI Code of Practice: Model documentation best practices
- ML Model Cards: Industry standard documentation (Mitchell et al., 2019)

The Model Card provides:
1. Model identity and technical details
2. Intended uses and out-of-scope applications
3. Performance metrics and evaluation methodology
4. Known limitations and biases
5. Ethical considerations
6. Integration requirements for downstream providers
7. Human oversight recommendations

References:
    - EU AI Act Article 53: https://artificialintelligenceact.eu/article/53/
    - GPAI Code of Practice: https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai
    - Model Cards for Model Reporting: https://arxiv.org/abs/1810.03993
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Any, Tuple
import hashlib
import json


class IntendedUse(Enum):
    """
    Intended use categories for the AI model.

    Defines the approved applications for the model.
    """
    TRADING_SIGNALS = "trading_signal_generation"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    RISK_ASSESSMENT = "risk_assessment"
    RESEARCH = "research_and_backtesting"
    MARKET_ANALYSIS = "market_analysis"
    STRATEGY_DEVELOPMENT = "strategy_development"


class LimitationType(Enum):
    """
    Types of model limitations.

    Categorizes limitations for clear documentation.
    """
    TECHNICAL = "technical"
    PERFORMANCE = "performance"
    ETHICAL = "ethical"
    REGULATORY = "regulatory"
    DATA = "data"
    OPERATIONAL = "operational"


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvaluationDataset(Enum):
    """Types of evaluation datasets."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"
    OUT_OF_DISTRIBUTION = "out_of_distribution"


@dataclass
class ModelLimitation:
    """
    Document a model limitation per Article 53(1)(b).

    Limitations must be clearly documented to enable downstream
    providers to understand model boundaries.

    Attributes:
        limitation_type: Category of the limitation
        description: Clear description of the limitation
        severity: Impact severity (low, medium, high)
        mitigation: Optional mitigation strategy
        affected_uses: Which use cases are affected
        technical_details: Additional technical information
    """
    limitation_type: LimitationType
    description: str
    severity: str
    mitigation: Optional[str] = None
    affected_uses: List[str] = field(default_factory=list)
    technical_details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "limitation_type": self.limitation_type.value,
            "description": self.description,
            "severity": self.severity,
            "mitigation": self.mitigation,
            "affected_uses": self.affected_uses,
            "technical_details": self.technical_details,
        }


@dataclass
class PerformanceMetric:
    """
    Performance metric with context per Article 53(1)(b).

    Metrics must include evaluation context for proper interpretation.

    Attributes:
        name: Metric name
        value: Metric value
        unit: Unit of measurement
        context: Evaluation context (e.g., "BTC/USDT 2020-2024")
        dataset: Which dataset the metric was computed on
        confidence_interval: Optional 95% CI bounds
        standard_error: Optional standard error
    """
    name: str
    value: float
    unit: str
    context: str
    dataset: EvaluationDataset = EvaluationDataset.TEST
    confidence_interval: Optional[Tuple[float, float]] = None
    standard_error: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "context": self.context,
            "dataset": self.dataset.value,
            "confidence_interval": self.confidence_interval,
            "standard_error": self.standard_error,
        }


@dataclass
class BiasAssessment:
    """
    Documented bias in the model.

    Biases must be documented per GPAI Code of Practice.

    Attributes:
        bias_type: Type of bias identified
        description: Description of the bias
        impact: Impact on predictions
        mitigation_status: Current mitigation status
        affected_groups: Groups affected by this bias
    """
    bias_type: str
    description: str
    impact: str
    mitigation_status: str
    affected_groups: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bias_type": self.bias_type,
            "description": self.description,
            "impact": self.impact,
            "mitigation_status": self.mitigation_status,
            "affected_groups": self.affected_groups,
        }


@dataclass
class EthicalConsideration:
    """
    Ethical consideration for the model.

    Documents ethical aspects per EU AI Act principles.

    Attributes:
        category: Category of ethical concern
        description: Description of the consideration
        guidance: Guidance for addressing it
        relevant_articles: Relevant EU AI Act articles
    """
    category: str
    description: str
    guidance: str
    relevant_articles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "description": self.description,
            "guidance": self.guidance,
            "relevant_articles": self.relevant_articles,
        }


@dataclass
class DownstreamRequirement:
    """
    Requirement for downstream providers.

    Per Article 53(1)(b), GPAI providers must give downstream
    providers information to comply with their obligations.

    Attributes:
        requirement_id: Unique identifier
        description: Description of the requirement
        article_reference: Relevant EU AI Act article
        mandatory: Whether requirement is mandatory
        implementation_guidance: How to implement
    """
    requirement_id: str
    description: str
    article_reference: str
    mandatory: bool
    implementation_guidance: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requirement_id": self.requirement_id,
            "description": self.description,
            "article_reference": self.article_reference,
            "mandatory": self.mandatory,
            "implementation_guidance": self.implementation_guidance,
        }


@dataclass
class GPAIModelCard:
    """
    GPAI Model Card per Article 53(1)(b).

    Provides downstream providers information to:
    - Understand capabilities and limitations
    - Enable compliance with their own obligations
    - Assess suitability for intended use
    - Implement appropriate safeguards

    This is a mandatory document for GPAI providers.

    Attributes:
        model_name: Name of the AI model
        model_version: Version string
        provider: Provider/company name
        release_date: Model release date
        card_version: Version of this model card
        architecture: Model architecture description
        training_approach: Training methodology
        parameters_count: Number of parameters
        input_format: Expected input format
        output_format: Model output format
        intended_uses: List of intended use cases
        out_of_scope_uses: Uses explicitly not supported
        performance_metrics: Performance measurements
        evaluation_methodology: How model was evaluated
        limitations: Known limitations
        known_biases: Documented biases
        ethical_considerations: Ethical aspects
        downstream_requirements: Requirements for integrators
        human_oversight_recommendations: Oversight guidance
        eu_ai_act_classification: Classification under AI Act
        article_references: Relevant AI Act articles
        contact_info: Contact information
    """
    # Identity
    model_name: str
    model_version: str
    provider: str
    release_date: datetime
    card_version: str = "1.0"

    # Technical details
    architecture: str = ""
    training_approach: str = ""
    parameters_count: int = 0
    input_format: str = ""
    output_format: str = ""

    # Intended use
    intended_uses: List[IntendedUse] = field(default_factory=list)
    out_of_scope_uses: List[str] = field(default_factory=list)

    # Performance
    performance_metrics: List[PerformanceMetric] = field(default_factory=list)
    evaluation_methodology: str = ""

    # Limitations and biases
    limitations: List[ModelLimitation] = field(default_factory=list)
    known_biases: List[BiasAssessment] = field(default_factory=list)

    # Ethical considerations
    ethical_considerations: List[EthicalConsideration] = field(default_factory=list)

    # Downstream guidance
    downstream_requirements: List[DownstreamRequirement] = field(default_factory=list)
    human_oversight_recommendations: List[str] = field(default_factory=list)

    # Compliance
    eu_ai_act_classification: str = "General-Purpose AI Model (GPAI)"
    article_references: List[str] = field(default_factory=list)

    # Contact
    contact_info: Dict[str, str] = field(default_factory=dict)

    def generate_card(self) -> str:
        """
        Generate model card document in Markdown format.

        Returns:
            Formatted markdown document
        """
        limitations_text = "\n".join([
            f"- **{l.limitation_type.value.title()}** ({l.severity}): {l.description}"
            + (f"\n  - *Mitigation*: {l.mitigation}" if l.mitigation else "")
            for l in self.limitations
        ]) if self.limitations else "No limitations documented."

        biases_text = "\n".join([
            f"- **{b.bias_type}**: {b.description}\n  - Impact: {b.impact}\n  - Status: {b.mitigation_status}"
            for b in self.known_biases
        ]) if self.known_biases else "No biases documented."

        ethics_text = "\n".join([
            f"- **{e.category}**: {e.description}\n  - Guidance: {e.guidance}"
            for e in self.ethical_considerations
        ]) if self.ethical_considerations else "No ethical considerations documented."

        downstream_text = "\n".join([
            f"- **{r.requirement_id}** ({r.article_reference}): {r.description}\n  - {'Mandatory' if r.mandatory else 'Recommended'}: {r.implementation_guidance}"
            for r in self.downstream_requirements
        ]) if self.downstream_requirements else "No specific requirements."

        metrics_rows = "\n".join([
            f"| {m.name} | {m.value} {m.unit} | {m.context} | {m.dataset.value} |"
            for m in self.performance_metrics
        ]) if self.performance_metrics else "| No metrics | - | - | - |"

        return f"""# GPAI Model Card

**Model**: {self.model_name}
**Version**: {self.model_version}
**Provider**: {self.provider}
**Release Date**: {self.release_date.strftime("%Y-%m-%d")}
**Card Version**: {self.card_version}
**EU AI Act Reference**: Article 53(1)(b)

---

## 1. Model Overview

| Attribute | Value |
|-----------|-------|
| Architecture | {self.architecture} |
| Training Approach | {self.training_approach} |
| Parameters | {self.parameters_count:,} |
| Input Format | {self.input_format} |
| Output Format | {self.output_format} |

## 2. Intended Use

### 2.1 Primary Uses

{chr(10).join(f"- {u.value.replace('_', ' ').title()}" for u in self.intended_uses) if self.intended_uses else "- Not specified"}

### 2.2 Out of Scope Uses

{chr(10).join(f"- {u}" for u in self.out_of_scope_uses) if self.out_of_scope_uses else "- None specified"}

**Important**: Using this model for out-of-scope applications may result in
unreliable outputs and potential harm. Users are responsible for ensuring
appropriate use.

## 3. Performance

### 3.1 Metrics

| Metric | Value | Context | Dataset |
|--------|-------|---------|---------|
{metrics_rows}

### 3.2 Evaluation Methodology

{self.evaluation_methodology if self.evaluation_methodology else "Not specified."}

## 4. Limitations

{limitations_text}

## 5. Known Biases

{biases_text}

## 6. Ethical Considerations

{ethics_text}

## 7. Requirements for Downstream Providers

Per Article 53(1)(b), downstream providers integrating this model must:

{downstream_text}

## 8. Human Oversight Recommendations

Per Article 14 of the EU AI Act, the following oversight measures are recommended:

{chr(10).join(f"- {r}" for r in self.human_oversight_recommendations) if self.human_oversight_recommendations else "- Follow standard AI oversight practices"}

## 9. EU AI Act Classification

| Attribute | Value |
|-----------|-------|
| Classification | {self.eu_ai_act_classification} |
| Relevant Articles | {", ".join(self.article_references) if self.article_references else "Article 53"} |
| Compliance Deadline | August 2, 2026 |

## 10. Contact Information

| Type | Contact |
|------|---------|
| Technical Support | {self.contact_info.get("technical", "Not specified")} |
| Compliance Inquiries | {self.contact_info.get("compliance", "Not specified")} |
| General | {self.contact_info.get("general", "Not specified")} |

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| {self.card_version} | {datetime.utcnow().strftime("%Y-%m-%d")} | Initial release |

---

*This model card is provided in accordance with Article 53(1)(b) of Regulation (EU) 2024/1689 (EU AI Act).*

*Last Updated: {datetime.utcnow().strftime("%Y-%m-%d")}*
"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert model card to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "provider": self.provider,
            "release_date": self.release_date.isoformat(),
            "card_version": self.card_version,
            "architecture": self.architecture,
            "training_approach": self.training_approach,
            "parameters_count": self.parameters_count,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "intended_uses": [u.value for u in self.intended_uses],
            "out_of_scope_uses": self.out_of_scope_uses,
            "performance_metrics": [m.to_dict() for m in self.performance_metrics],
            "evaluation_methodology": self.evaluation_methodology,
            "limitations": [l.to_dict() for l in self.limitations],
            "known_biases": [b.to_dict() for b in self.known_biases],
            "ethical_considerations": [e.to_dict() for e in self.ethical_considerations],
            "downstream_requirements": [r.to_dict() for r in self.downstream_requirements],
            "human_oversight_recommendations": self.human_oversight_recommendations,
            "eu_ai_act_classification": self.eu_ai_act_classification,
            "article_references": self.article_references,
            "contact_info": self.contact_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPAIModelCard":
        """Create model card from dictionary."""
        return cls(
            model_name=data["model_name"],
            model_version=data["model_version"],
            provider=data["provider"],
            release_date=datetime.fromisoformat(data["release_date"]),
            card_version=data.get("card_version", "1.0"),
            architecture=data.get("architecture", ""),
            training_approach=data.get("training_approach", ""),
            parameters_count=data.get("parameters_count", 0),
            input_format=data.get("input_format", ""),
            output_format=data.get("output_format", ""),
            intended_uses=[IntendedUse(u) for u in data.get("intended_uses", [])],
            out_of_scope_uses=data.get("out_of_scope_uses", []),
            performance_metrics=[
                PerformanceMetric(
                    name=m["name"],
                    value=m["value"],
                    unit=m["unit"],
                    context=m["context"],
                    dataset=EvaluationDataset(m.get("dataset", "test")),
                    confidence_interval=m.get("confidence_interval"),
                    standard_error=m.get("standard_error"),
                )
                for m in data.get("performance_metrics", [])
            ],
            evaluation_methodology=data.get("evaluation_methodology", ""),
            limitations=[
                ModelLimitation(
                    limitation_type=LimitationType(l["limitation_type"]),
                    description=l["description"],
                    severity=l["severity"],
                    mitigation=l.get("mitigation"),
                    affected_uses=l.get("affected_uses", []),
                    technical_details=l.get("technical_details"),
                )
                for l in data.get("limitations", [])
            ],
            known_biases=[
                BiasAssessment(
                    bias_type=b["bias_type"],
                    description=b["description"],
                    impact=b["impact"],
                    mitigation_status=b["mitigation_status"],
                    affected_groups=b.get("affected_groups", []),
                )
                for b in data.get("known_biases", [])
            ],
            ethical_considerations=[
                EthicalConsideration(
                    category=e["category"],
                    description=e["description"],
                    guidance=e["guidance"],
                    relevant_articles=e.get("relevant_articles", []),
                )
                for e in data.get("ethical_considerations", [])
            ],
            downstream_requirements=[
                DownstreamRequirement(
                    requirement_id=r["requirement_id"],
                    description=r["description"],
                    article_reference=r["article_reference"],
                    mandatory=r["mandatory"],
                    implementation_guidance=r["implementation_guidance"],
                )
                for r in data.get("downstream_requirements", [])
            ],
            human_oversight_recommendations=data.get("human_oversight_recommendations", []),
            eu_ai_act_classification=data.get("eu_ai_act_classification", "General-Purpose AI Model (GPAI)"),
            article_references=data.get("article_references", []),
            contact_info=data.get("contact_info", {}),
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the model card for quick reference."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "provider": self.provider,
            "classification": self.eu_ai_act_classification,
            "intended_uses_count": len(self.intended_uses),
            "limitations_count": len(self.limitations),
            "biases_count": len(self.known_biases),
            "metrics_count": len(self.performance_metrics),
            "downstream_requirements_count": len(self.downstream_requirements),
        }


def create_default_model_card() -> GPAIModelCard:
    """
    Create default model card for the platform.

    Returns:
        Pre-configured GPAIModelCard with platform-specific information
    """
    return GPAIModelCard(
        model_name="Distributional PPO Trading Model",
        model_version="4.0",
        provider="AI-Powered Quantitative Research Platform",
        release_date=datetime(2025, 12, 1),
        card_version="1.0",
        architecture="LSTM + Distributional Value Network (C51-style, 21 quantiles)",
        training_approach="Reinforcement Learning (PPO with Twin Critics, Self-Adversarial Training)",
        parameters_count=2_500_000,
        input_format="Normalized feature vector (OHLCV + 50 technical indicators, shape: [sequence_length, 55])",
        output_format="Action distribution (buy/hold/sell probs) + value distribution (21 quantiles) + uncertainty estimate",
        intended_uses=[
            IntendedUse.TRADING_SIGNALS,
            IntendedUse.RESEARCH,
            IntendedUse.RISK_ASSESSMENT,
            IntendedUse.MARKET_ANALYSIS,
        ],
        out_of_scope_uses=[
            "Personalized investment advice for individuals",
            "Credit scoring or lending decisions",
            "Insurance pricing or underwriting",
            "Fully autonomous trading without human oversight",
            "High-frequency trading (latency requirements not met)",
            "Trading of illiquid or exotic assets",
            "Regulatory compliance decisions",
            "Financial advice requiring fiduciary duty",
        ],
        performance_metrics=[
            PerformanceMetric(
                name="Sharpe Ratio",
                value=1.2,
                unit="",
                context="BTC/USDT, ETH/USDT backtest",
                dataset=EvaluationDataset.TEST,
                confidence_interval=(0.9, 1.5),
            ),
            PerformanceMetric(
                name="Max Drawdown",
                value=15.0,
                unit="%",
                context="Worst-case across all test periods",
                dataset=EvaluationDataset.TEST,
            ),
            PerformanceMetric(
                name="Win Rate",
                value=52.0,
                unit="%",
                context="Directional accuracy",
                dataset=EvaluationDataset.TEST,
            ),
            PerformanceMetric(
                name="Sortino Ratio",
                value=1.8,
                unit="",
                context="Downside risk-adjusted return",
                dataset=EvaluationDataset.TEST,
            ),
            PerformanceMetric(
                name="Calmar Ratio",
                value=0.8,
                unit="",
                context="Return/Max Drawdown",
                dataset=EvaluationDataset.TEST,
            ),
        ],
        evaluation_methodology=(
            "Walk-forward validation with 70/15/15 train/val/test split. "
            "Test period: 2023-01 to 2024-12 (out-of-sample). "
            "Metrics computed across 10 random seeds for statistical significance. "
            "Transaction costs of 0.1% included in all calculations."
        ),
        limitations=[
            ModelLimitation(
                limitation_type=LimitationType.TECHNICAL,
                description="Requires minimum 100ms inference latency for real-time use",
                severity="medium",
                mitigation="Use batched inference or deploy on GPU for lower latency",
                affected_uses=["high-frequency applications"],
            ),
            ModelLimitation(
                limitation_type=LimitationType.PERFORMANCE,
                description="May underperform during unprecedented market events (black swans)",
                severity="high",
                mitigation="Implement kill switch and human oversight per Article 14",
                affected_uses=["live trading"],
            ),
            ModelLimitation(
                limitation_type=LimitationType.PERFORMANCE,
                description="Trained primarily on liquid assets; illiquid assets not recommended",
                severity="medium",
                mitigation="Only use for assets with daily volume > $10M",
                affected_uses=["illiquid asset trading"],
            ),
            ModelLimitation(
                limitation_type=LimitationType.DATA,
                description="Training data ends December 2024; may not capture recent market dynamics",
                severity="medium",
                mitigation="Retrain periodically with updated data",
                affected_uses=["all uses"],
            ),
            ModelLimitation(
                limitation_type=LimitationType.OPERATIONAL,
                description="Requires stable data feed; missing data degrades performance",
                severity="high",
                mitigation="Implement data quality checks and fallback mechanisms",
                affected_uses=["live trading"],
            ),
        ],
        known_biases=[
            BiasAssessment(
                bias_type="Temporal Bias",
                description="Better performance in trending vs. ranging markets",
                impact="May generate false signals during low-volatility periods",
                mitigation_status="Partially mitigated through regime detection",
                affected_groups=["Users trading in ranging markets"],
            ),
            BiasAssessment(
                bias_type="Asset Bias",
                description="Optimized for major cryptocurrencies (BTC, ETH), may not generalize",
                impact="Lower accuracy on smaller altcoins or traditional assets",
                mitigation_status="Documented; users advised to validate on target assets",
                affected_groups=["Users trading non-primary assets"],
            ),
            BiasAssessment(
                bias_type="Regime Bias",
                description="Trained mostly on 2020-2024 data (post-COVID bull market dominated)",
                impact="May underperform in prolonged bear markets",
                mitigation_status="Mitigated through adversarial training (SA-PPO)",
                affected_groups=["All users during regime changes"],
            ),
        ],
        ethical_considerations=[
            EthicalConsideration(
                category="Financial Risk",
                description="Model outputs should not be used as sole decision basis for trading",
                guidance="Always combine with human judgment and risk management",
                relevant_articles=["Article 14", "Article 50"],
            ),
            EthicalConsideration(
                category="User Understanding",
                description="Users must understand AI limitations before live trading",
                guidance="Implement mandatory disclosure per Article 50",
                relevant_articles=["Article 50(1)"],
            ),
            EthicalConsideration(
                category="Loss Potential",
                description="Significant financial losses are possible",
                guidance="Users should only trade with funds they can afford to lose",
                relevant_articles=["Article 13"],
            ),
            EthicalConsideration(
                category="Vulnerable Users",
                description="Not suitable for users without trading experience",
                guidance="Implement user qualification checks",
                relevant_articles=["Article 5", "Recital 28"],
            ),
        ],
        downstream_requirements=[
            DownstreamRequirement(
                requirement_id="DR-001",
                description="Implement kill switch for immediate trading halt",
                article_reference="Article 14(4)(f)",
                mandatory=True,
                implementation_guidance="Provide UI button and API endpoint for immediate halt",
            ),
            DownstreamRequirement(
                requirement_id="DR-002",
                description="Log all model outputs for audit trail",
                article_reference="Article 12",
                mandatory=True,
                implementation_guidance="Store predictions, confidence, timestamp for minimum 5 years",
            ),
            DownstreamRequirement(
                requirement_id="DR-003",
                description="Display AI disclosure to end users",
                article_reference="Article 50",
                mandatory=True,
                implementation_guidance="Show disclosure before first use and in all AI outputs",
            ),
            DownstreamRequirement(
                requirement_id="DR-004",
                description="Maintain human oversight capability",
                article_reference="Article 14",
                mandatory=True,
                implementation_guidance="Human must be able to review and override all decisions",
            ),
            DownstreamRequirement(
                requirement_id="DR-005",
                description="Implement position and drawdown limits",
                article_reference="Article 9(2)",
                mandatory=True,
                implementation_guidance="Set maximum position size and daily loss limits",
            ),
            DownstreamRequirement(
                requirement_id="DR-006",
                description="Monitor model performance in production",
                article_reference="Article 72",
                mandatory=True,
                implementation_guidance="Track accuracy drift and alert on degradation",
            ),
        ],
        human_oversight_recommendations=[
            "Monitor model performance daily with automated alerts",
            "Review all anomalous signals before execution (>2 std dev from mean)",
            "Set position limits per trade (recommend max 5% of portfolio)",
            "Set daily drawdown limit (recommend max 3% of portfolio)",
            "Maintain ability to halt all trading within 1 second",
            "Review weekly performance reports for drift detection",
            "Conduct monthly model performance review meetings",
            "Implement escalation procedures for model failures",
        ],
        eu_ai_act_classification="General-Purpose AI Model (GPAI)",
        article_references=["Article 53", "Article 50", "Article 52", "Article 12", "Article 14"],
        contact_info={
            "technical": "ai-support@platform.com",
            "compliance": "compliance@platform.com",
            "general": "info@platform.com",
        },
    )


class ModelCardManager:
    """
    Manager for GPAI model cards.

    Provides functionality to:
    - Generate and update model cards
    - Export in various formats
    - Track model card versions
    - Validate compliance

    Example:
        >>> manager = create_model_card_manager()
        >>> card_md = manager.get_model_card()
        >>> metadata = manager.get_card_metadata()
    """

    def __init__(self, model_card: Optional[GPAIModelCard] = None):
        """
        Initialize the model card manager.

        Args:
            model_card: Optional pre-configured model card
        """
        self.current_card = model_card or create_default_model_card()
        self._version_history: List[Dict[str, Any]] = []

    def get_model_card(self) -> str:
        """
        Get model card document in Markdown format.

        Returns:
            Formatted markdown document
        """
        return self.current_card.generate_card()

    def get_model_card_json(self) -> str:
        """
        Get model card as JSON string.

        Returns:
            JSON-formatted model card
        """
        return json.dumps(self.current_card.to_dict(), indent=2, default=str)

    def get_card_metadata(self) -> Dict[str, Any]:
        """
        Get card metadata for API responses.

        Returns:
            Dictionary with model card metadata
        """
        return {
            "model_name": self.current_card.model_name,
            "model_version": self.current_card.model_version,
            "card_version": self.current_card.card_version,
            "provider": self.current_card.provider,
            "classification": self.current_card.eu_ai_act_classification,
            "intended_uses": [u.value for u in self.current_card.intended_uses],
            "limitations_count": len(self.current_card.limitations),
            "biases_count": len(self.current_card.known_biases),
            "metrics_count": len(self.current_card.performance_metrics),
            "article_reference": "EU AI Act Article 53(1)(b)",
            "last_updated": datetime.utcnow().isoformat(),
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get model card summary.

        Returns:
            Summary dictionary
        """
        return self.current_card.get_summary()

    def validate_compliance(self) -> Dict[str, Any]:
        """
        Validate model card compliance with Article 53(1)(b).

        Returns:
            Dictionary with validation results
        """
        checks = {
            "has_model_name": bool(self.current_card.model_name),
            "has_version": bool(self.current_card.model_version),
            "has_provider": bool(self.current_card.provider),
            "has_architecture": bool(self.current_card.architecture),
            "has_intended_uses": len(self.current_card.intended_uses) > 0,
            "has_out_of_scope_uses": len(self.current_card.out_of_scope_uses) > 0,
            "has_performance_metrics": len(self.current_card.performance_metrics) > 0,
            "has_limitations": len(self.current_card.limitations) > 0,
            "has_ethical_considerations": len(self.current_card.ethical_considerations) > 0,
            "has_downstream_requirements": len(self.current_card.downstream_requirements) > 0,
            "has_human_oversight": len(self.current_card.human_oversight_recommendations) > 0,
            "has_classification": bool(self.current_card.eu_ai_act_classification),
        }

        all_valid = all(checks.values())

        return {
            "compliant": all_valid,
            "checks": checks,
            "missing": [k for k, v in checks.items() if not v],
            "article_reference": "Article 53(1)(b)",
            "validation_timestamp": datetime.utcnow().isoformat(),
        }

    def update_card(self, updates: Dict[str, Any]) -> bool:
        """
        Update model card with new values.

        Args:
            updates: Dictionary of fields to update

        Returns:
            True if update successful
        """
        # Store current version in history
        self._version_history.append({
            "version": self.current_card.card_version,
            "timestamp": datetime.utcnow().isoformat(),
            "data": self.current_card.to_dict(),
        })

        # Apply updates
        for key, value in updates.items():
            if hasattr(self.current_card, key):
                setattr(self.current_card, key, value)

        # Increment version
        major, minor = self.current_card.card_version.split(".")
        self.current_card.card_version = f"{major}.{int(minor) + 1}"

        return True

    def get_version_history(self) -> List[Dict[str, Any]]:
        """
        Get version history of the model card.

        Returns:
            List of version records
        """
        return self._version_history.copy()

    def get_downstream_checklist(self) -> List[Dict[str, Any]]:
        """
        Get checklist for downstream providers.

        Returns:
            List of requirements with status fields
        """
        return [
            {
                "requirement_id": r.requirement_id,
                "description": r.description,
                "article_reference": r.article_reference,
                "mandatory": r.mandatory,
                "implementation_guidance": r.implementation_guidance,
                "status": "pending",  # To be filled by downstream provider
            }
            for r in self.current_card.downstream_requirements
        ]


def create_model_card_manager(
    model_card: Optional[GPAIModelCard] = None
) -> ModelCardManager:
    """
    Factory function to create ModelCardManager.

    Args:
        model_card: Optional pre-configured model card

    Returns:
        Configured ModelCardManager instance
    """
    return ModelCardManager(model_card=model_card)


def get_default_limitations() -> List[ModelLimitation]:
    """
    Get default model limitations.

    Returns:
        List of standard ModelLimitation instances
    """
    card = create_default_model_card()
    return card.limitations.copy()


def get_default_biases() -> List[BiasAssessment]:
    """
    Get default bias assessments.

    Returns:
        List of standard BiasAssessment instances
    """
    card = create_default_model_card()
    return card.known_biases.copy()


def get_default_downstream_requirements() -> List[DownstreamRequirement]:
    """
    Get default downstream requirements.

    Returns:
        List of standard DownstreamRequirement instances
    """
    card = create_default_model_card()
    return card.downstream_requirements.copy()


def validate_model_card(card: GPAIModelCard) -> Dict[str, bool]:
    """
    Validate a model card for Article 53(1)(b) compliance.

    Args:
        card: Model card to validate

    Returns:
        Dictionary with validation results
    """
    return {
        "has_identity": bool(card.model_name and card.model_version and card.provider),
        "has_technical_details": bool(card.architecture and card.training_approach),
        "has_intended_uses": len(card.intended_uses) > 0,
        "has_limitations": len(card.limitations) > 0,
        "has_metrics": len(card.performance_metrics) > 0,
        "has_downstream_info": len(card.downstream_requirements) > 0,
        "has_oversight": len(card.human_oversight_recommendations) > 0,
    }
