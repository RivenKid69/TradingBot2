# -*- coding: utf-8 -*-
"""
AI Act Risk Registry Implementation.

EU AI Act Article 9 - Risk Registry for High-Risk AI Systems.

This module implements a comprehensive risk registry for tracking, managing,
and auditing risks identified in the algorithmic trading system.

References:
    - EU AI Act Article 9: Risk Management System
    - ISO 31000:2018 Risk Management Guidelines
    - NIST AI RMF (Risk Management Framework)

Example:
    >>> from services.ai_act.risk_registry import (
    ...     RiskRegistry,
    ...     create_risk_registry,
    ...     get_default_trading_risks,
    ... )
    >>> registry = create_risk_registry()
    >>> registry.load_default_risks()
    >>> risks = registry.get_all_risks()
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from services.ai_act.risk_management import (
    AIActRiskCategory,
    AIActRiskSeverity,
    AIActRiskLikelihood,
    RiskIdentification,
    RiskAssessment,
    RiskMitigation,
)


logger = logging.getLogger(__name__)


class AIActJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles enum values and datetime objects."""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class RiskStatus(Enum):
    """Risk lifecycle status per ISO 31000."""

    IDENTIFIED = "identified"
    """Risk has been identified but not yet assessed."""

    ASSESSED = "assessed"
    """Risk has been assessed with severity and likelihood."""

    MITIGATING = "mitigating"
    """Mitigation measures are being implemented."""

    MITIGATED = "mitigated"
    """Risk has been mitigated to acceptable level."""

    ACCEPTED = "accepted"
    """Risk has been accepted (residual risk within tolerance)."""

    TRANSFERRED = "transferred"
    """Risk has been transferred (e.g., via insurance)."""

    AVOIDED = "avoided"
    """Risk has been avoided by eliminating the activity."""

    CLOSED = "closed"
    """Risk is no longer applicable."""

    ESCALATED = "escalated"
    """Risk has been escalated for management decision."""


@dataclass
class RiskEntry:
    """
    Complete risk entry for the registry.

    Combines identification, assessment, and mitigation information
    with lifecycle tracking and audit trail.

    Attributes:
        risk_id: Unique identifier (e.g., R001).
        identification: Risk identification details.
        assessment: Risk assessment with severity/likelihood.
        mitigations: List of applied mitigations.
        status: Current lifecycle status.
        owner: Person/team responsible for the risk.
        created_at: When the risk was first identified.
        updated_at: Last modification timestamp.
        review_date: Next scheduled review date.
        notes: Additional notes and comments.
        audit_trail: History of status changes.
    """

    risk_id: str
    identification: RiskIdentification
    assessment: Optional[RiskAssessment] = None
    mitigations: List[RiskMitigation] = field(default_factory=list)
    status: RiskStatus = RiskStatus.IDENTIFIED
    owner: str = "AI Act Compliance Team"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    review_date: Optional[datetime] = None
    notes: str = ""
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize audit trail with creation entry."""
        if not self.audit_trail:
            self.audit_trail.append(
                {
                    "timestamp": self.created_at.isoformat(),
                    "action": "created",
                    "status": self.status.value,
                    "user": "system",
                }
            )

    @property
    def risk_score(self) -> int:
        """Get risk score from assessment (0 if not assessed)."""
        if self.assessment is None:
            return 0
        return self.assessment.risk_score

    @property
    def is_high_risk(self) -> bool:
        """Check if risk score indicates high risk (>= 12)."""
        return self.risk_score >= 12

    @property
    def is_critical(self) -> bool:
        """Check if risk has critical severity."""
        if self.assessment is None:
            return False
        return self.assessment.severity == AIActRiskSeverity.CRITICAL

    @property
    def requires_review(self) -> bool:
        """Check if risk requires review based on review_date."""
        if self.review_date is None:
            return False
        return datetime.now(timezone.utc) >= self.review_date

    def update_status(
        self,
        new_status: RiskStatus,
        user: str = "system",
        reason: str = "",
    ) -> None:
        """
        Update risk status with audit trail entry.

        Args:
            new_status: New status to set.
            user: User making the change.
            reason: Reason for status change.
        """
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)

        self.audit_trail.append(
            {
                "timestamp": self.updated_at.isoformat(),
                "action": "status_change",
                "old_status": old_status.value,
                "new_status": new_status.value,
                "user": user,
                "reason": reason,
            }
        )

        logger.info(f"Risk {self.risk_id} status changed: {old_status.value} -> {new_status.value}")

    def add_mitigation(self, mitigation: RiskMitigation) -> None:
        """
        Add mitigation measure to the risk.

        Args:
            mitigation: Mitigation to add.
        """
        self.mitigations.append(mitigation)
        self.updated_at = datetime.now(timezone.utc)

        self.audit_trail.append(
            {
                "timestamp": self.updated_at.isoformat(),
                "action": "mitigation_added",
                "mitigation_id": mitigation.mitigation_id,
                "description": mitigation.description,
                "user": "system",
            }
        )

        # Auto-update status if not already mitigated
        if self.status in (RiskStatus.IDENTIFIED, RiskStatus.ASSESSED):
            self.update_status(RiskStatus.MITIGATING)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        return {
            "risk_id": self.risk_id,
            "identification": asdict(self.identification),
            "assessment": asdict(self.assessment) if self.assessment else None,
            "mitigations": [asdict(m) for m in self.mitigations],
            "status": self.status.value,
            "owner": self.owner,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "review_date": self.review_date.isoformat() if self.review_date else None,
            "notes": self.notes,
            "audit_trail": self.audit_trail,
            "risk_score": self.risk_score,
            "is_high_risk": self.is_high_risk,
            "is_critical": self.is_critical,
        }


class RiskRegistry:
    """
    Comprehensive Risk Registry for EU AI Act Compliance.

    Manages the complete lifecycle of identified risks including:
    - Risk registration and tracking
    - Status management and audit trail
    - Export for regulatory reporting
    - Integration with risk management system

    Thread-safe implementation for production use.

    Attributes:
        storage_path: Path for persistent storage.
        auto_save: Whether to auto-save on changes.

    Example:
        >>> registry = RiskRegistry()
        >>> registry.load_default_risks()
        >>> entry = registry.get_risk("R001")
        >>> entry.update_status(RiskStatus.MITIGATED)
        >>> registry.save()
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        auto_save: bool = True,
    ) -> None:
        """
        Initialize RiskRegistry.

        Args:
            storage_path: Path for persistent storage. Defaults to
                logs/ai_act/risk_registry/registry.json
            auto_save: Whether to auto-save on changes.
        """
        self._lock = threading.RLock()
        self._risks: Dict[str, RiskEntry] = {}
        self._next_id: int = 1
        self._auto_save = auto_save

        if storage_path is None:
            storage_path = Path("logs/ai_act/risk_registry")
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._registry_file = self._storage_path / "registry.json"

        # Callbacks for risk events
        self._on_high_risk_callbacks: List[Callable[[RiskEntry], None]] = []
        self._on_critical_risk_callbacks: List[Callable[[RiskEntry], None]] = []

        logger.info(f"RiskRegistry initialized at {self._storage_path}")

    def generate_risk_id(self) -> str:
        """Generate next sequential risk ID."""
        with self._lock:
            risk_id = f"R{self._next_id:03d}"
            self._next_id += 1
            return risk_id

    def register_risk(
        self,
        identification: RiskIdentification,
        assessment: Optional[RiskAssessment] = None,
        owner: str = "AI Act Compliance Team",
        notes: str = "",
    ) -> RiskEntry:
        """
        Register a new risk in the registry.

        Args:
            identification: Risk identification details.
            assessment: Optional initial assessment.
            owner: Risk owner.
            notes: Additional notes.

        Returns:
            Created RiskEntry.
        """
        with self._lock:
            risk_id = self.generate_risk_id()

            entry = RiskEntry(
                risk_id=risk_id,
                identification=identification,
                assessment=assessment,
                owner=owner,
                notes=notes,
            )

            if assessment is not None:
                entry.status = RiskStatus.ASSESSED

            self._risks[risk_id] = entry

            # Trigger callbacks for high/critical risks
            if entry.is_critical:
                self._trigger_critical_risk_callbacks(entry)
            elif entry.is_high_risk:
                self._trigger_high_risk_callbacks(entry)

            if self._auto_save:
                self._save_internal()

            logger.info(
                f"Registered risk {risk_id}: {identification.title} "
                f"(category={identification.category.value})"
            )

            return entry

    def get_risk(self, risk_id: str) -> Optional[RiskEntry]:
        """Get risk entry by ID."""
        with self._lock:
            return self._risks.get(risk_id)

    def get_all_risks(self) -> List[RiskEntry]:
        """Get all registered risks."""
        with self._lock:
            return list(self._risks.values())

    def get_risks_by_status(self, status: RiskStatus) -> List[RiskEntry]:
        """Get risks filtered by status."""
        with self._lock:
            return [r for r in self._risks.values() if r.status == status]

    def get_risks_by_category(self, category: AIActRiskCategory) -> List[RiskEntry]:
        """Get risks filtered by category."""
        with self._lock:
            return [r for r in self._risks.values() if r.identification.category == category]

    def get_high_risks(self) -> List[RiskEntry]:
        """Get all high-risk entries (score >= 12)."""
        with self._lock:
            return [r for r in self._risks.values() if r.is_high_risk]

    def get_critical_risks(self) -> List[RiskEntry]:
        """Get all critical severity risks."""
        with self._lock:
            return [r for r in self._risks.values() if r.is_critical]

    def get_risks_requiring_review(self) -> List[RiskEntry]:
        """Get risks that require review."""
        with self._lock:
            return [r for r in self._risks.values() if r.requires_review]

    def get_high_priority_risks(self, threshold: int = 12) -> List[RiskEntry]:
        """
        Get all high-priority risks (score >= threshold).

        Args:
            threshold: Minimum risk score to be considered high priority (default: 12)

        Returns:
            List of high-priority risk entries, sorted by score descending.
        """
        with self._lock:
            high_priority = [
                r
                for r in self._risks.values()
                if r.risk_score is not None and r.risk_score >= threshold
            ]
            return sorted(high_priority, key=lambda r: r.risk_score or 0, reverse=True)

    def update_risk_status(
        self,
        risk_id: str,
        new_status: RiskStatus,
        user: str = "system",
        reason: str = "",
    ) -> bool:
        """
        Update status of a specific risk.

        Args:
            risk_id: Risk identifier.
            new_status: New status to set.
            user: User making the change.
            reason: Reason for change.

        Returns:
            True if updated successfully.
        """
        with self._lock:
            entry = self._risks.get(risk_id)
            if entry is None:
                logger.warning(f"Risk {risk_id} not found for status update")
                return False

            entry.update_status(new_status, user, reason)

            if self._auto_save:
                self._save_internal()

            return True

    def add_mitigation_to_risk(
        self,
        risk_id: str,
        mitigation: RiskMitigation,
    ) -> bool:
        """
        Add mitigation to a specific risk.

        Args:
            risk_id: Risk identifier.
            mitigation: Mitigation to add.

        Returns:
            True if added successfully.
        """
        with self._lock:
            entry = self._risks.get(risk_id)
            if entry is None:
                logger.warning(f"Risk {risk_id} not found for mitigation")
                return False

            entry.add_mitigation(mitigation)

            if self._auto_save:
                self._save_internal()

            return True

    def update_assessment(
        self,
        risk_id: str,
        assessment: RiskAssessment,
    ) -> bool:
        """
        Update assessment for a specific risk.

        Args:
            risk_id: Risk identifier.
            assessment: New assessment to set.

        Returns:
            True if updated successfully.
        """
        with self._lock:
            entry = self._risks.get(risk_id)
            if entry is None:
                logger.warning(f"Risk {risk_id} not found for assessment update")
                return False

            entry.assessment = assessment

            if self._auto_save:
                self._save_internal()

            return True

    def register_high_risk_callback(
        self,
        callback: Callable[[RiskEntry], None],
    ) -> None:
        """Register callback for high-risk entries."""
        self._on_high_risk_callbacks.append(callback)

    def register_critical_risk_callback(
        self,
        callback: Callable[[RiskEntry], None],
    ) -> None:
        """Register callback for critical risks."""
        self._on_critical_risk_callbacks.append(callback)

    def _trigger_high_risk_callbacks(self, entry: RiskEntry) -> None:
        """Trigger high-risk callbacks."""
        for callback in self._on_high_risk_callbacks:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"Error in high-risk callback: {e}")

    def _trigger_critical_risk_callbacks(self, entry: RiskEntry) -> None:
        """Trigger critical risk callbacks."""
        for callback in self._on_critical_risk_callbacks:
            try:
                callback(entry)
            except Exception as e:
                logger.error(f"Error in critical-risk callback: {e}")

    def load_default_risks(self) -> int:
        """
        Load predefined trading system risks.

        Returns:
            Number of risks loaded.
        """
        default_risks = get_default_trading_risks()
        count = 0

        for risk_data in default_risks:
            identification = RiskIdentification(
                risk_id="",  # Auto-generate
                category=risk_data["category"],
                title=risk_data["title"],
                description=risk_data["description"],
                source=risk_data["source"],
                affected_components=risk_data["affected_components"],
            )

            assessment = RiskAssessment(
                assessment_id=f"A{count + 1:03d}",
                risk_id=identification.risk_id,
                severity=risk_data["severity"],
                likelihood=risk_data["likelihood"],
            )

            entry = self.register_risk(
                identification=identification,
                assessment=assessment,
                notes=risk_data.get("notes", ""),
            )

            # Add mitigations
            for mit_data in risk_data.get("mitigations", []):
                mitigation = RiskMitigation(
                    mitigation_id=str(uuid.uuid4())[:8],
                    risk_id=entry.risk_id,
                    description=mit_data["description"],
                    status=mit_data["status"],
                    effectiveness_rating=mit_data.get("effectiveness", "medium"),
                )
                entry.add_mitigation(mitigation)

            # Update status based on mitigations
            if entry.mitigations:
                all_implemented = all(m.status == "implemented" for m in entry.mitigations)
                if all_implemented:
                    entry.update_status(
                        RiskStatus.MITIGATED,
                        reason="All mitigations implemented",
                    )

            count += 1

        logger.info(f"Loaded {count} default trading risks")
        return count

    def get_summary(self) -> Dict[str, Any]:
        """
        Get registry summary statistics.

        Returns:
            Summary dictionary with counts and statistics.
        """
        with self._lock:
            risks = list(self._risks.values())

            status_counts = {}
            for status in RiskStatus:
                status_counts[status.value] = sum(1 for r in risks if r.status == status)

            category_counts = {}
            for category in AIActRiskCategory:
                category_counts[category.value] = sum(
                    1 for r in risks if r.identification.category == category
                )

            return {
                "total_risks": len(risks),
                "high_risks": sum(1 for r in risks if r.is_high_risk),
                "critical_risks": sum(1 for r in risks if r.is_critical),
                "requiring_review": sum(1 for r in risks if r.requires_review),
                "status_breakdown": status_counts,
                "category_breakdown": category_counts,
                "average_risk_score": (
                    sum(r.risk_score for r in risks) / len(risks) if risks else 0.0
                ),
                "last_updated": max(
                    (r.updated_at for r in risks),
                    default=datetime.now(timezone.utc),
                ).isoformat(),
            }

    def export_for_audit(self, output_path: Optional[Path] = None) -> Path:
        """
        Export registry for regulatory audit.

        Args:
            output_path: Output file path. Defaults to storage_path/audit_export.json

        Returns:
            Path to exported file.
        """
        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = self._storage_path / f"audit_export_{timestamp}.json"

        with self._lock:
            export_data = {
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "ai_act_compliance_version": "1.0.0",
                "system_classification": "HIGH-RISK AI SYSTEM",
                "regulatory_basis": "EU AI Act (Regulation (EU) 2024/1689) Article 9",
                "summary": self.get_summary(),
                "risks": [r.to_dict() for r in self._risks.values()],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, cls=AIActJSONEncoder)

        logger.info(f"Exported risk registry to {output_path}")
        return output_path

    def save(self) -> None:
        """Manually save registry to storage."""
        with self._lock:
            self._save_internal()

    def _save_internal(self) -> None:
        """Internal save without lock acquisition."""
        data = {
            "next_id": self._next_id,
            "risks": {rid: r.to_dict() for rid, r in self._risks.items()},
        }

        with open(self._registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=AIActJSONEncoder)

    def load(self) -> bool:
        """
        Load registry from storage.

        Returns:
            True if loaded successfully.
        """
        if not self._registry_file.exists():
            logger.info("No existing registry file found")
            return False

        try:
            with open(self._registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            with self._lock:
                self._next_id = data.get("next_id", 1)
                self._risks.clear()

                for risk_id, risk_data in data.get("risks", {}).items():
                    # Reconstruct RiskEntry from dict
                    id_data = risk_data["identification"]
                    identification = RiskIdentification(
                        risk_id=id_data.get("risk_id", ""),
                        source=id_data.get("source", ""),
                        category=AIActRiskCategory(id_data["category"]),
                        title=id_data["title"],
                        description=id_data["description"],
                        affected_components=id_data.get("affected_components", []),
                        identification_date=id_data.get("identification_date", ""),
                        identified_by=id_data.get("identified_by", "system"),
                    )

                    assessment = None
                    if risk_data.get("assessment"):
                        ass_data = risk_data["assessment"]
                        assessment = RiskAssessment(
                            assessment_id=ass_data.get("assessment_id", ""),
                            risk_id=risk_id,
                            severity=AIActRiskSeverity(ass_data["severity"]),
                            likelihood=AIActRiskLikelihood(ass_data["likelihood"]),
                            assessor=ass_data.get("assessor", "system"),
                            assessment_date=ass_data.get("assessment_date", ""),
                        )

                    mitigations = []
                    for mit_data in risk_data.get("mitigations", []):
                        # Handle optional enum fields
                        residual_sev = None
                        if mit_data.get("residual_severity"):
                            residual_sev = AIActRiskSeverity(mit_data["residual_severity"])
                        residual_lik = None
                        if mit_data.get("residual_likelihood"):
                            residual_lik = AIActRiskLikelihood(mit_data["residual_likelihood"])

                        mitigations.append(
                            RiskMitigation(
                                mitigation_id=mit_data.get("mitigation_id", ""),
                                risk_id=risk_id,
                                mitigation_type=mit_data.get("mitigation_type", "control"),
                                title=mit_data.get("title", ""),
                                description=mit_data.get("description", ""),
                                implementation_details=mit_data.get("implementation_details", ""),
                                preventive_controls=mit_data.get("preventive_controls", []),
                                detective_controls=mit_data.get("detective_controls", []),
                                corrective_controls=mit_data.get("corrective_controls", []),
                                status=mit_data.get("status", "planned"),
                                implementation_date=mit_data.get("implementation_date"),
                                verification_date=mit_data.get("verification_date"),
                                residual_severity=residual_sev,
                                residual_likelihood=residual_lik,
                                effectiveness_rating=mit_data.get("effectiveness_rating", ""),
                                deployer_information_provided=mit_data.get(
                                    "deployer_information_provided", False
                                ),
                                training_requirements=mit_data.get("training_requirements", []),
                                owner=mit_data.get("owner", ""),
                                review_frequency=mit_data.get("review_frequency", ""),
                                last_review_date=mit_data.get("last_review_date", ""),
                            )
                        )

                    entry = RiskEntry(
                        risk_id=risk_id,
                        identification=identification,
                        assessment=assessment,
                        mitigations=mitigations,
                        status=RiskStatus(risk_data["status"]),
                        owner=risk_data.get("owner", "AI Act Compliance Team"),
                        created_at=datetime.fromisoformat(risk_data["created_at"]),
                        updated_at=datetime.fromisoformat(risk_data["updated_at"]),
                        review_date=(
                            datetime.fromisoformat(risk_data["review_date"])
                            if risk_data.get("review_date")
                            else None
                        ),
                        notes=risk_data.get("notes", ""),
                        audit_trail=risk_data.get("audit_trail", []),
                    )

                    self._risks[risk_id] = entry

            logger.info(f"Loaded {len(self._risks)} risks from registry")
            return True

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return False


def get_default_trading_risks() -> List[Dict[str, Any]]:
    """
    Get predefined trading system risks per EU AI Act requirements.

    Returns list of risk data dictionaries matching the plan's risk table.

    Returns:
        List of risk definitions.
    """
    return [
        {
            "source": "system_analysis",
            "category": AIActRiskCategory.MODEL_ROBUSTNESS,
            "title": "Distributional Shift in Market Data",
            "description": (
                "Market regime changes or black swan events may cause model "
                "performance degradation due to distribution shift between "
                "training and inference data."
            ),
            "severity": AIActRiskSeverity.HIGH,
            "likelihood": AIActRiskLikelihood.POSSIBLE,
            "affected_components": [
                "distributional_ppo.py",
                "features_pipeline.py",
                "service_conformal.py",
            ],
            "mitigations": [
                {
                    "description": "VGS (Variance Gradient Scaler) for gradient stability",
                    "status": "implemented",
                    "effectiveness": 0.85,
                },
                {
                    "description": "Conformal prediction for uncertainty quantification",
                    "status": "implemented",
                    "effectiveness": 0.80,
                },
                {
                    "description": "Continuous model monitoring with drift detection",
                    "status": "implemented",
                    "effectiveness": 0.75,
                },
            ],
            "notes": "Primary trading risk. Covered by existing VGS and conformal prediction systems.",
        },
        {
            "source": "operational_review",
            "category": AIActRiskCategory.OPERATIONAL,
            "title": "System Downtime During Trading Hours",
            "description": (
                "Unexpected system failures during active trading hours may "
                "lead to inability to manage positions, resulting in uncontrolled "
                "exposure and potential financial losses."
            ),
            "severity": AIActRiskSeverity.CRITICAL,
            "likelihood": AIActRiskLikelihood.UNLIKELY,
            "affected_components": [
                "script_live.py",
                "services/ops_kill_switch.py",
                "risk_guard.py",
            ],
            "mitigations": [
                {
                    "description": "Redundant system architecture with failover",
                    "status": "planned",
                    "effectiveness": 0.90,
                },
                {
                    "description": "Operational kill switch with persistent state",
                    "status": "implemented",
                    "effectiveness": 0.95,
                },
                {
                    "description": "Automated position unwinding on failure",
                    "status": "implemented",
                    "effectiveness": 0.85,
                },
            ],
            "notes": "Kill switch system provides immediate position protection.",
        },
        {
            "source": "financial_analysis",
            "category": AIActRiskCategory.FINANCIAL,
            "title": "Excessive Portfolio Drawdown",
            "description": (
                "Sustained adverse market conditions combined with model "
                "misjudgment may lead to drawdowns exceeding risk tolerance, "
                "potentially triggering margin calls or fund closure."
            ),
            "severity": AIActRiskSeverity.HIGH,
            "likelihood": AIActRiskLikelihood.POSSIBLE,
            "affected_components": [
                "risk_guard.py",
                "trading_patchnew.py",
                "execution_sim.py",
            ],
            "mitigations": [
                {
                    "description": "Dynamic position limits based on volatility",
                    "status": "implemented",
                    "effectiveness": 0.80,
                },
                {
                    "description": "Rolling drawdown monitoring with automatic reduction",
                    "status": "implemented",
                    "effectiveness": 0.85,
                },
                {
                    "description": "CVaR-based risk budgeting in training",
                    "status": "implemented",
                    "effectiveness": 0.75,
                },
            ],
            "notes": "Risk guards provide multi-layer protection against excessive losses.",
        },
        {
            "source": "compliance_review",
            "category": AIActRiskCategory.REGULATORY,
            "title": "Non-Compliant Trading Behavior",
            "description": (
                "AI system may generate trading signals that violate market "
                "regulations, including market manipulation, wash trading, "
                "or exceeding position limits."
            ),
            "severity": AIActRiskSeverity.CRITICAL,
            "likelihood": AIActRiskLikelihood.UNLIKELY,
            "affected_components": [
                "risk_guard.py",
                "execution_sim.py",
                "services/ai_act/",
            ],
            "mitigations": [
                {
                    "description": "Pre-trade compliance checks",
                    "status": "implemented",
                    "effectiveness": 0.90,
                },
                {
                    "description": "Comprehensive audit logging",
                    "status": "implemented",
                    "effectiveness": 0.85,
                },
                {
                    "description": "Position limit enforcement",
                    "status": "implemented",
                    "effectiveness": 0.95,
                },
            ],
            "notes": "MiFID II and exchange-specific rules enforced at pre-trade stage.",
        },
        {
            "source": "security_audit",
            "category": AIActRiskCategory.SECURITY,
            "title": "Unauthorized System Access",
            "description": (
                "Malicious actors may attempt to gain unauthorized access to "
                "the trading system, potentially manipulating trading signals "
                "or extracting proprietary strategies."
            ),
            "severity": AIActRiskSeverity.CRITICAL,
            "likelihood": AIActRiskLikelihood.UNLIKELY,
            "affected_components": [
                "script_live.py",
                "adapters/",
                "configs/",
            ],
            "mitigations": [
                {
                    "description": "API key encryption and secure storage",
                    "status": "implemented",
                    "effectiveness": 0.90,
                },
                {
                    "description": "Network isolation and firewall rules",
                    "status": "implemented",
                    "effectiveness": 0.85,
                },
                {
                    "description": "Access logging and anomaly detection",
                    "status": "planned",
                    "effectiveness": 0.80,
                },
            ],
            "notes": "Security measures follow industry best practices.",
        },
        {
            "source": "technical_review",
            "category": AIActRiskCategory.DATA_QUALITY,
            "title": "Data Feed Corruption or Gaps",
            "description": (
                "Market data feeds may contain errors, gaps, or latency spikes "
                "that lead to incorrect feature computation and trading signals."
            ),
            "severity": AIActRiskSeverity.MEDIUM,
            "likelihood": AIActRiskLikelihood.POSSIBLE,
            "affected_components": [
                "features_pipeline.py",
                "adapters/",
                "data_loader_multi_asset.py",
            ],
            "mitigations": [
                {
                    "description": "Data validation and sanity checks",
                    "status": "implemented",
                    "effectiveness": 0.85,
                },
                {
                    "description": "Multiple data source cross-validation",
                    "status": "partial",
                    "effectiveness": 0.70,
                },
                {
                    "description": "Gap-filling and outlier detection",
                    "status": "implemented",
                    "effectiveness": 0.75,
                },
            ],
            "notes": "Features pipeline includes NaN handling and winsorization.",
        },
        {
            "source": "algorithm_review",
            "category": AIActRiskCategory.ALGORITHMIC_BIAS,
            "title": "Biased Trading Decisions",
            "description": (
                "Model may develop biases toward certain market conditions, "
                "asset classes, or time periods that reduce generalization "
                "and create blind spots in risk assessment."
            ),
            "severity": AIActRiskSeverity.MEDIUM,
            "likelihood": AIActRiskLikelihood.POSSIBLE,
            "affected_components": [
                "distributional_ppo.py",
                "train_model_multi_patch.py",
                "features_pipeline.py",
            ],
            "mitigations": [
                {
                    "description": "Diverse training data across market regimes",
                    "status": "implemented",
                    "effectiveness": 0.75,
                },
                {
                    "description": "Adversarial training (SA-PPO) for robustness",
                    "status": "implemented",
                    "effectiveness": 0.80,
                },
                {
                    "description": "Regular model retraining and evaluation",
                    "status": "implemented",
                    "effectiveness": 0.70,
                },
            ],
            "notes": "PBT and SA-PPO provide regime-robust training.",
        },
        {
            "source": "operational_review",
            "category": AIActRiskCategory.HUMAN_OVERSIGHT,
            "title": "Insufficient Human Oversight",
            "description": (
                "Lack of adequate human oversight mechanisms may prevent "
                "timely intervention when the AI system behaves unexpectedly "
                "or market conditions require human judgment."
            ),
            "severity": AIActRiskSeverity.HIGH,
            "likelihood": AIActRiskLikelihood.UNLIKELY,
            "affected_components": [
                "services/ops_kill_switch.py",
                "services/ai_act/human_oversight.py",
                "script_live.py",
            ],
            "mitigations": [
                {
                    "description": "Multi-level kill switch with manual override",
                    "status": "implemented",
                    "effectiveness": 0.95,
                },
                {
                    "description": "Real-time monitoring dashboard",
                    "status": "planned",
                    "effectiveness": 0.85,
                },
                {
                    "description": "Alert system for anomalous behavior",
                    "status": "implemented",
                    "effectiveness": 0.80,
                },
            ],
            "notes": "AI Act Article 14 compliance being enhanced in Phase 1.",
        },
        {
            "source": "execution_analysis",
            "category": AIActRiskCategory.EXECUTION,
            "title": "Slippage and Execution Failures",
            "description": (
                "Order execution may differ significantly from expected prices "
                "due to market impact, latency, or exchange issues, eroding "
                "strategy profitability."
            ),
            "severity": AIActRiskSeverity.MEDIUM,
            "likelihood": AIActRiskLikelihood.LIKELY,
            "affected_components": [
                "execution_sim.py",
                "execution_providers.py",
                "adapters/",
            ],
            "mitigations": [
                {
                    "description": "Realistic slippage modeling (L2+/L3)",
                    "status": "implemented",
                    "effectiveness": 0.85,
                },
                {
                    "description": "Smart order routing",
                    "status": "partial",
                    "effectiveness": 0.70,
                },
                {
                    "description": "Execution quality monitoring",
                    "status": "implemented",
                    "effectiveness": 0.75,
                },
            ],
            "notes": "Multi-level execution providers (L2, L2+, L3) model realistic costs.",
        },
        {
            "source": "legal_review",
            "category": AIActRiskCategory.TRANSPARENCY,
            "title": "Insufficient Decision Explainability",
            "description": (
                "Inability to explain AI trading decisions may violate "
                "regulatory requirements and hinder risk assessment by "
                "human operators."
            ),
            "severity": AIActRiskSeverity.MEDIUM,
            "likelihood": AIActRiskLikelihood.POSSIBLE,
            "affected_components": [
                "services/ai_act/explainability.py",
                "distributional_ppo.py",
                "custom_policy_patch1.py",
            ],
            "mitigations": [
                {
                    "description": "Feature importance tracking",
                    "status": "planned",
                    "effectiveness": 0.75,
                },
                {
                    "description": "Decision logging with confidence scores",
                    "status": "implemented",
                    "effectiveness": 0.70,
                },
                {
                    "description": "Post-hoc explanation generation",
                    "status": "planned",
                    "effectiveness": 0.80,
                },
            ],
            "notes": "Phase 1 adds explainability module for AI Act compliance.",
        },
    ]


def create_risk_registry(
    storage_path: Optional[Path] = None,
    auto_save: bool = True,
    load_existing: bool = True,
    include_defaults: bool = False,
) -> RiskRegistry:
    """
    Factory function to create a RiskRegistry.

    Args:
        storage_path: Path for persistent storage.
        auto_save: Whether to auto-save on changes.
        load_existing: Whether to load existing registry if present.
        include_defaults: Whether to load default trading risks.

    Returns:
        Configured RiskRegistry instance.
    """
    registry = RiskRegistry(
        storage_path=storage_path,
        auto_save=auto_save,
    )

    if load_existing:
        registry.load()

    if include_defaults:
        registry.load_default_risks()

    return registry
