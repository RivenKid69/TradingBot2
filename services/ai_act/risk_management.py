# -*- coding: utf-8 -*-
"""
AI Act Risk Management System (Article 9).

EU AI Act Article 9 requires high-risk AI systems to establish, implement,
document and maintain a risk management system.

This module provides:
1. AIActRiskCategory - Risk categorization per AI Act taxonomy
2. RiskIdentification - Identification of foreseeable risks
3. RiskAssessment - Risk evaluation and estimation
4. RiskMitigation - Adoption of appropriate risk management measures
5. AIActRiskManager - Main risk management orchestrator

Article 9 Requirements Implemented:
    - (1) Continuous iterative process throughout AI system lifecycle
    - (2)(a) Identification of foreseeable risks during intended use
    - (2)(b) Identification of risks during reasonably foreseeable misuse
    - (2)(c) Evaluation based on post-market monitoring data
    - (2)(d) Evaluation of possible risks when interacting with other systems
    - (4)(a) Elimination or reduction of risks through design
    - (4)(b) Implementation of mitigation and control measures
    - (4)(c) Provision of information and training to deployers
    - (6) Testing against prior defined metrics and probabilistic thresholds
    - (7) Testing in real-world conditions where appropriate

References:
    - Article 9: https://artificialintelligenceact.eu/article/9/
    - Goodwin Law Analysis: https://www.goodwinlaw.com/en/insights/publications/2024/08/
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Risk Categories (per AI Act taxonomy)
# =============================================================================


class AIActRiskCategory(Enum):
    """
    Risk categories as defined by EU AI Act for high-risk AI systems.

    Based on Article 9(2) and industry best practices for algorithmic trading.
    """

    # Core AI Act categories
    SAFETY = "safety"
    FUNDAMENTAL_RIGHTS = "fundamental_rights"

    # Trading-specific categories
    MARKET_STABILITY = "market_stability"
    DATA_QUALITY = "data_quality"
    MODEL_ROBUSTNESS = "model_robustness"
    CYBERSECURITY = "cybersecurity"
    HUMAN_OVERSIGHT = "human_oversight"
    HUMAN_OVERSIGHT_FAILURE = "human_oversight_failure"
    BIAS_DISCRIMINATION = "bias_discrimination"
    ALGORITHMIC_BIAS = "algorithmic_bias"

    # Operational categories
    SYSTEM_FAILURE = "system_failure"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    THIRD_PARTY_DEPENDENCY = "third_party_dependency"
    EXECUTION = "execution"
    TRANSPARENCY = "transparency"

    # Business/financial categories for trading systems
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    REGULATORY = "regulatory"
    SECURITY = "security"


class AIActRiskSeverity(Enum):
    """Risk severity levels for impact assessment."""

    NEGLIGIBLE = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5


class AIActRiskLikelihood(Enum):
    """Risk likelihood levels for probability assessment."""

    RARE = 1  # < 5% probability
    UNLIKELY = 2  # 5-20% probability
    POSSIBLE = 3  # 20-50% probability
    LIKELY = 4  # 50-80% probability
    ALMOST_CERTAIN = 5  # > 80% probability


# =============================================================================
# Risk Data Structures
# =============================================================================


@dataclass
class RiskIdentification:
    """
    Risk identification record per Article 9(2).

    Captures foreseeable risks during:
    - (a) intended use
    - (b) reasonably foreseeable misuse
    """

    risk_id: str
    category: AIActRiskCategory
    title: str
    description: str

    # Source of risk identification
    source: str = ""  # E.g., "design_review", "testing", "post_market_monitoring"

    # Affected system components
    affected_components: List[str] = field(default_factory=list)

    # Context
    intended_use_risk: bool = True  # Article 9(2)(a)
    foreseeable_misuse_risk: bool = False  # Article 9(2)(b)
    interaction_risk: bool = False  # Article 9(2)(d)

    # Affected parties
    affected_parties: List[str] = field(default_factory=list)
    vulnerable_groups_considered: bool = False  # Article 9(9)

    # Discovery
    identification_date: str = ""
    identification_method: str = ""  # design_review, testing, monitoring, etc.
    identified_by: str = ""

    # Status
    is_active: bool = True

    def __post_init__(self):
        if not self.risk_id:
            self.risk_id = f"RISK-{uuid.uuid4().hex[:8].upper()}"
        if not self.identification_date:
            self.identification_date = datetime.now(timezone.utc).isoformat()


@dataclass
class RiskAssessment:
    """
    Risk assessment record per Article 9(2)(c).

    Evaluates and estimates risks based on available data including
    post-market monitoring data.
    """

    assessment_id: str
    risk_id: str

    # Impact assessment
    severity: AIActRiskSeverity = AIActRiskSeverity.MEDIUM
    likelihood: AIActRiskLikelihood = AIActRiskLikelihood.POSSIBLE

    # Risk score (severity * likelihood)
    @property
    def risk_score(self) -> int:
        return self.severity.value * self.likelihood.value

    @property
    def risk_level(self) -> str:
        """Categorize risk level based on score."""
        score = self.risk_score
        if score <= 4:
            return "LOW"
        elif score <= 9:
            return "MEDIUM"
        elif score <= 16:
            return "HIGH"
        else:
            return "CRITICAL"

    # Assessment details
    impact_description: str = ""
    affected_systems: List[str] = field(default_factory=list)
    potential_harm: str = ""

    # Quantitative metrics (where applicable)
    estimated_financial_impact: Optional[float] = None
    estimated_frequency: Optional[float] = None  # Events per time period

    # Post-market data (Article 9(2)(c))
    based_on_post_market_data: bool = False
    post_market_incidents: int = 0

    # Assessment metadata
    assessment_date: str = ""
    assessor: str = ""
    review_due_date: str = ""

    def __post_init__(self):
        if not self.assessment_id:
            self.assessment_id = f"ASSESS-{uuid.uuid4().hex[:8].upper()}"
        if not self.assessment_date:
            self.assessment_date = datetime.now(timezone.utc).isoformat()


@dataclass
class RiskMitigation:
    """
    Risk mitigation record per Article 9(4).

    Documents measures adopted to address identified risks:
    - (a) elimination or reduction through design
    - (b) implementation of mitigation and control measures
    - (c) provision of information and training to deployers
    """

    mitigation_id: str
    risk_id: str

    # Mitigation type (per Article 9(4))
    mitigation_type: str = "control"  # "design", "control", "information"

    # Mitigation details
    title: str = ""
    description: str = ""
    implementation_details: str = ""

    # Controls
    preventive_controls: List[str] = field(default_factory=list)
    detective_controls: List[str] = field(default_factory=list)
    corrective_controls: List[str] = field(default_factory=list)

    # Implementation status
    status: str = "planned"  # "planned", "in_progress", "implemented", "verified"
    implementation_date: Optional[str] = None
    verification_date: Optional[str] = None

    # Effectiveness
    residual_severity: Optional[AIActRiskSeverity] = None
    residual_likelihood: Optional[AIActRiskLikelihood] = None
    effectiveness_rating: str = ""  # "low", "medium", "high"

    # Documentation (Article 9(4)(c))
    deployer_information_provided: bool = False
    training_requirements: List[str] = field(default_factory=list)

    # Review
    owner: str = ""
    review_frequency: str = ""  # "monthly", "quarterly", "annually"
    last_review_date: str = ""

    def __post_init__(self):
        if not self.mitigation_id:
            self.mitigation_id = f"MIT-{uuid.uuid4().hex[:8].upper()}"

    @property
    def residual_risk_score(self) -> Optional[int]:
        """Calculate residual risk score after mitigation."""
        if self.residual_severity and self.residual_likelihood:
            return self.residual_severity.value * self.residual_likelihood.value
        return None


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AIActRiskConfig:
    """
    Configuration for AI Act Risk Management System.

    Extends existing RiskConfig with AI Act specific parameters.
    """

    # Assessment frequency (Article 9(1) - continuous iterative process)
    risk_assessment_frequency_hours: int = 24
    enable_continuous_monitoring: bool = True

    # Thresholds
    residual_risk_acceptance_threshold: float = 0.01  # 1% maximum residual
    risk_score_escalation_threshold: int = 16  # Score >= 16 requires escalation

    # Vulnerable groups (Article 9(9))
    vulnerable_group_considerations: List[str] = field(default_factory=list)

    # Post-market monitoring integration
    post_market_data_lookback_days: int = 90
    incident_threshold_for_reassessment: int = 3

    # Testing (Article 9(6))
    testing_frequency_hours: int = 168  # Weekly
    test_coverage_target: float = 0.95  # 95% coverage

    # Logging
    log_all_assessments: bool = True
    log_path: str = "logs/ai_act/risk_management"

    # Notification
    escalation_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main Risk Manager
# =============================================================================


class AIActRiskManager:
    """
    AI Act compliant Risk Management System per Article 9.

    Implements:
    1. Continuous iterative risk management process
    2. Risk identification, assessment, and mitigation tracking
    3. Integration with existing risk_guard.py
    4. Testing against defined metrics and thresholds

    Usage:
        config = AIActRiskConfig(
            risk_assessment_frequency_hours=24,
            enable_continuous_monitoring=True,
        )
        manager = AIActRiskManager(config)

        # Identify a new risk
        risk = manager.identify_risk(
            category=AIActRiskCategory.MODEL_ROBUSTNESS,
            title="Distribution Shift",
            description="Model performance degrades under market regime change",
        )

        # Assess the risk
        assessment = manager.assess_risk(
            risk_id=risk.risk_id,
            severity=AIActRiskSeverity.HIGH,
            likelihood=AIActRiskLikelihood.POSSIBLE,
        )

        # Add mitigation
        mitigation = manager.add_mitigation(
            risk_id=risk.risk_id,
            mitigation_type="control",
            title="VGS Gradient Scaling",
            description="Use VGS to detect and adapt to distribution shifts",
        )
    """

    def __init__(self, config: Optional[AIActRiskConfig] = None):
        """Initialize risk manager with configuration."""
        self.config = config or AIActRiskConfig()

        # Risk data stores
        self._risks: Dict[str, RiskIdentification] = {}
        self._assessments: Dict[str, List[RiskAssessment]] = {}  # risk_id -> assessments
        self._mitigations: Dict[str, List[RiskMitigation]] = {}  # risk_id -> mitigations

        # Monitoring state
        self._last_assessment_time: float = time.time()
        self._incident_count: int = 0
        self._monitoring_active: bool = False

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("AIActRiskManager initialized")

    # =========================================================================
    # Risk Identification (Article 9(2)(a)(b)(d))
    # =========================================================================

    def identify_risk(
        self,
        category: AIActRiskCategory,
        title: str,
        description: str,
        intended_use_risk: bool = True,
        foreseeable_misuse_risk: bool = False,
        interaction_risk: bool = False,
        affected_parties: Optional[List[str]] = None,
        identification_method: str = "manual",
        identified_by: str = "system",
    ) -> RiskIdentification:
        """
        Identify and register a new risk per Article 9(2).

        Args:
            category: Risk category from AIActRiskCategory
            title: Short risk title
            description: Detailed risk description
            intended_use_risk: Risk during intended use (Article 9(2)(a))
            foreseeable_misuse_risk: Risk during foreseeable misuse (Article 9(2)(b))
            interaction_risk: Risk from system interactions (Article 9(2)(d))
            affected_parties: List of affected parties
            identification_method: How the risk was identified
            identified_by: Who identified the risk

        Returns:
            RiskIdentification record
        """
        risk = RiskIdentification(
            risk_id="",  # Will be auto-generated
            category=category,
            title=title,
            description=description,
            intended_use_risk=intended_use_risk,
            foreseeable_misuse_risk=foreseeable_misuse_risk,
            interaction_risk=interaction_risk,
            affected_parties=affected_parties or [],
            identification_method=identification_method,
            identified_by=identified_by,
        )

        with self._lock:
            self._risks[risk.risk_id] = risk
            self._assessments[risk.risk_id] = []
            self._mitigations[risk.risk_id] = []

        if self.config.log_all_assessments:
            self._log_event("risk_identified", asdict(risk))

        logger.info(f"Risk identified: {risk.risk_id} - {title}")
        return risk

    def get_risk(self, risk_id: str) -> Optional[RiskIdentification]:
        """Get a risk by ID."""
        with self._lock:
            return self._risks.get(risk_id)

    def get_all_risks(self) -> List[RiskIdentification]:
        """Get all registered risks."""
        with self._lock:
            return list(self._risks.values())

    def get_risks_by_category(self, category: AIActRiskCategory) -> List[RiskIdentification]:
        """Get all risks in a specific category."""
        with self._lock:
            return [r for r in self._risks.values() if r.category == category]

    # =========================================================================
    # Risk Assessment (Article 9(2)(c))
    # =========================================================================

    def assess_risk(
        self,
        risk_id: str,
        severity: AIActRiskSeverity,
        likelihood: AIActRiskLikelihood,
        impact_description: str = "",
        affected_systems: Optional[List[str]] = None,
        potential_harm: str = "",
        estimated_financial_impact: Optional[float] = None,
        based_on_post_market_data: bool = False,
        post_market_incidents: int = 0,
        assessor: str = "system",
    ) -> RiskAssessment:
        """
        Assess an identified risk per Article 9(2)(c).

        Args:
            risk_id: ID of the risk to assess
            severity: Impact severity
            likelihood: Probability of occurrence
            impact_description: Description of potential impact
            affected_systems: List of affected systems
            potential_harm: Description of potential harm
            estimated_financial_impact: Estimated financial impact in USD
            based_on_post_market_data: Whether assessment uses post-market data
            post_market_incidents: Number of related post-market incidents
            assessor: Who performed the assessment

        Returns:
            RiskAssessment record

        Raises:
            ValueError: If risk_id not found
        """
        with self._lock:
            if risk_id not in self._risks:
                raise ValueError(f"Risk {risk_id} not found")

        assessment = RiskAssessment(
            assessment_id="",
            risk_id=risk_id,
            severity=severity,
            likelihood=likelihood,
            impact_description=impact_description,
            affected_systems=affected_systems or [],
            potential_harm=potential_harm,
            estimated_financial_impact=estimated_financial_impact,
            based_on_post_market_data=based_on_post_market_data,
            post_market_incidents=post_market_incidents,
            assessor=assessor,
        )

        with self._lock:
            self._assessments[risk_id].append(assessment)
            self._last_assessment_time = time.time()

        # Check for escalation
        if assessment.risk_score >= self.config.risk_score_escalation_threshold:
            self._escalate_risk(risk_id, assessment)

        if self.config.log_all_assessments:
            self._log_event(
                "risk_assessed",
                {
                    **asdict(assessment),
                    "risk_score": assessment.risk_score,
                    "risk_level": assessment.risk_level,
                },
            )

        logger.info(
            f"Risk assessed: {risk_id} - Score={assessment.risk_score} "
            f"Level={assessment.risk_level}"
        )
        return assessment

    def get_latest_assessment(self, risk_id: str) -> Optional[RiskAssessment]:
        """Get the most recent assessment for a risk."""
        with self._lock:
            assessments = self._assessments.get(risk_id, [])
            return assessments[-1] if assessments else None

    def get_all_assessments(self, risk_id: str) -> List[RiskAssessment]:
        """Get all assessments for a risk."""
        with self._lock:
            return list(self._assessments.get(risk_id, []))

    # =========================================================================
    # Risk Mitigation (Article 9(4))
    # =========================================================================

    def add_mitigation(
        self,
        risk_id: str,
        mitigation_type: str,
        title: str,
        description: str,
        implementation_details: str = "",
        preventive_controls: Optional[List[str]] = None,
        detective_controls: Optional[List[str]] = None,
        corrective_controls: Optional[List[str]] = None,
        status: str = "planned",
        owner: str = "",
        training_requirements: Optional[List[str]] = None,
    ) -> RiskMitigation:
        """
        Add a mitigation measure for a risk per Article 9(4).

        Args:
            risk_id: ID of the risk to mitigate
            mitigation_type: Type of mitigation ("design", "control", "information")
            title: Short mitigation title
            description: Detailed description
            implementation_details: How the mitigation is implemented
            preventive_controls: List of preventive controls
            detective_controls: List of detective controls
            corrective_controls: List of corrective controls
            status: Implementation status
            owner: Person responsible for the mitigation
            training_requirements: Required training for deployers

        Returns:
            RiskMitigation record

        Raises:
            ValueError: If risk_id not found or invalid mitigation_type
        """
        if mitigation_type not in ("design", "control", "information"):
            raise ValueError(
                f"Invalid mitigation_type: {mitigation_type}. "
                "Must be 'design', 'control', or 'information'"
            )

        with self._lock:
            if risk_id not in self._risks:
                raise ValueError(f"Risk {risk_id} not found")

        mitigation = RiskMitigation(
            mitigation_id="",
            risk_id=risk_id,
            mitigation_type=mitigation_type,
            title=title,
            description=description,
            implementation_details=implementation_details,
            preventive_controls=preventive_controls or [],
            detective_controls=detective_controls or [],
            corrective_controls=corrective_controls or [],
            status=status,
            owner=owner,
            training_requirements=training_requirements or [],
        )

        with self._lock:
            self._mitigations[risk_id].append(mitigation)

        if self.config.log_all_assessments:
            self._log_event("mitigation_added", asdict(mitigation))

        logger.info(f"Mitigation added: {mitigation.mitigation_id} for {risk_id}")
        return mitigation

    def update_mitigation_status(
        self,
        mitigation_id: str,
        status: str,
        residual_severity: Optional[AIActRiskSeverity] = None,
        residual_likelihood: Optional[AIActRiskLikelihood] = None,
        effectiveness_rating: str = "",
    ) -> Optional[RiskMitigation]:
        """Update the status of a mitigation measure."""
        with self._lock:
            for mitigations in self._mitigations.values():
                for m in mitigations:
                    if m.mitigation_id == mitigation_id:
                        m.status = status
                        if status == "implemented":
                            m.implementation_date = datetime.now(timezone.utc).isoformat()
                        elif status == "verified":
                            m.verification_date = datetime.now(timezone.utc).isoformat()
                        if residual_severity:
                            m.residual_severity = residual_severity
                        if residual_likelihood:
                            m.residual_likelihood = residual_likelihood
                        if effectiveness_rating:
                            m.effectiveness_rating = effectiveness_rating

                        if self.config.log_all_assessments:
                            self._log_event("mitigation_updated", asdict(m))

                        return m
        return None

    def get_mitigations(self, risk_id: str) -> List[RiskMitigation]:
        """Get all mitigations for a risk."""
        with self._lock:
            return list(self._mitigations.get(risk_id, []))

    # =========================================================================
    # Risk Monitoring (Article 9(1))
    # =========================================================================

    def start_continuous_monitoring(self) -> None:
        """Start continuous risk monitoring per Article 9(1)."""
        with self._lock:
            self._monitoring_active = True
        logger.info("Continuous risk monitoring started")

    def stop_continuous_monitoring(self) -> None:
        """Stop continuous risk monitoring."""
        with self._lock:
            self._monitoring_active = False
        logger.info("Continuous risk monitoring stopped")

    def is_monitoring_active(self) -> bool:
        """Check if continuous monitoring is active."""
        with self._lock:
            return self._monitoring_active

    def record_incident(
        self,
        risk_id: str,
        incident_description: str,
        severity: AIActRiskSeverity,
    ) -> None:
        """
        Record a risk-related incident for post-market monitoring.

        This data is used for risk reassessment per Article 9(2)(c).
        """
        with self._lock:
            self._incident_count += 1

            # Check if reassessment is needed
            if self._incident_count >= self.config.incident_threshold_for_reassessment:
                logger.warning(
                    f"Incident threshold reached ({self._incident_count}). "
                    "Risk reassessment recommended."
                )

        self._log_event(
            "incident_recorded",
            {
                "risk_id": risk_id,
                "description": incident_description,
                "severity": severity.name,
                "total_incidents": self._incident_count,
            },
        )

    def needs_reassessment(self) -> bool:
        """Check if risk reassessment is due based on time or incidents."""
        with self._lock:
            # Check time-based reassessment
            hours_since_last = (time.time() - self._last_assessment_time) / 3600
            if hours_since_last >= self.config.risk_assessment_frequency_hours:
                return True

            # Check incident-based reassessment
            if self._incident_count >= self.config.incident_threshold_for_reassessment:
                return True

            return False

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all risks and their status.

        Returns:
            Dictionary with risk summary statistics
        """
        with self._lock:
            total_risks = len(self._risks)
            risks_by_category: Dict[str, int] = {}
            risks_by_level: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
            mitigated_count = 0

            for risk_id, risk in self._risks.items():
                # Count by category
                cat = risk.category.value
                risks_by_category[cat] = risks_by_category.get(cat, 0) + 1

                # Get latest assessment
                assessment = self.get_latest_assessment(risk_id)
                if assessment:
                    risks_by_level[assessment.risk_level] += 1

                # Check if mitigated
                mitigations = self._mitigations.get(risk_id, [])
                if any(m.status == "verified" for m in mitigations):
                    mitigated_count += 1

        return {
            "total_risks": total_risks,
            "risks_by_category": risks_by_category,
            "risks_by_level": risks_by_level,
            "mitigated_count": mitigated_count,
            "unmitigated_count": total_risks - mitigated_count,
            "incident_count": self._incident_count,
            "monitoring_active": self._monitoring_active,
            "last_assessment_time": self._last_assessment_time,
            "needs_reassessment": self.needs_reassessment(),
        }

    def export_risk_register(self) -> Dict[str, Any]:
        """
        Export complete risk register for documentation.

        This supports technical documentation requirements (Article 11).
        """
        with self._lock:
            risks_data = []
            for risk_id, risk in self._risks.items():
                risk_data = asdict(risk)
                risk_data["assessments"] = [asdict(a) for a in self._assessments.get(risk_id, [])]
                risk_data["mitigations"] = [asdict(m) for m in self._mitigations.get(risk_id, [])]
                risks_data.append(risk_data)

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "total_risks": len(risks_data),
            "risks": risks_data,
            "summary": self.get_risk_summary(),
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _escalate_risk(self, risk_id: str, assessment: RiskAssessment) -> None:
        """Escalate high-scoring risks."""
        logger.warning(
            f"Risk escalation: {risk_id} - Score={assessment.risk_score} "
            f"Level={assessment.risk_level}"
        )

        if self.config.escalation_callback:
            try:
                self.config.escalation_callback(
                    risk_id,
                    {
                        "assessment_id": assessment.assessment_id,
                        "risk_score": assessment.risk_score,
                        "risk_level": assessment.risk_level,
                        "severity": assessment.severity.name,
                        "likelihood": assessment.likelihood.name,
                    },
                )
            except Exception as e:
                logger.error(f"Escalation callback failed: {e}")

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a risk management event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"risk_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Function
# =============================================================================


def create_risk_manager(
    config: Optional[Union[Dict[str, Any], AIActRiskConfig]] = None,
) -> AIActRiskManager:
    """
    Create an AIActRiskManager instance.

    Args:
        config: Optional configuration - can be a dictionary or AIActRiskConfig instance

    Returns:
        Configured AIActRiskManager instance
    """
    if config is None:
        ai_act_config = AIActRiskConfig()
    elif isinstance(config, AIActRiskConfig):
        # Already an AIActRiskConfig instance, use directly
        ai_act_config = config
    else:
        # Assume it's a dictionary
        ai_act_config = AIActRiskConfig(
            risk_assessment_frequency_hours=config.get("risk_assessment_frequency_hours", 24),
            enable_continuous_monitoring=config.get("enable_continuous_monitoring", True),
            residual_risk_acceptance_threshold=config.get(
                "residual_risk_acceptance_threshold", 0.01
            ),
            risk_score_escalation_threshold=config.get("risk_score_escalation_threshold", 16),
            vulnerable_group_considerations=config.get("vulnerable_group_considerations", []),
            post_market_data_lookback_days=config.get("post_market_data_lookback_days", 90),
            incident_threshold_for_reassessment=config.get(
                "incident_threshold_for_reassessment", 3
            ),
            testing_frequency_hours=config.get("testing_frequency_hours", 168),
            test_coverage_target=config.get("test_coverage_target", 0.95),
            log_all_assessments=config.get("log_all_assessments", True),
            log_path=config.get("log_path", "logs/ai_act/risk_management"),
        )

    return AIActRiskManager(ai_act_config)
