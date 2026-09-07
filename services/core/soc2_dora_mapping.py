# -*- coding: utf-8 -*-
"""
SOC2-DORA Control Mapping (Block 2.7).

Maps SOC2 Trust Services Criteria to DORA Articles:
- Control correlation for shared evidence
- Gap analysis
- Unified audit support

DORA References:
    - All Articles (mapping to SOC2 TSC)
    - RTS CDR 2024/1774: ICT Risk Management Framework

SOC2 Trust Services Criteria:
    - CC: Common Criteria (Security)
    - A: Availability
    - PI: Processing Integrity
    - C: Confidentiality
    - P: Privacy
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SOC2Category(Enum):
    """SOC2 Trust Services Categories."""

    CC_SECURITY = "CC"  # Common Criteria - Security
    AVAILABILITY = "A"  # Availability
    PROCESSING_INTEGRITY = "PI"  # Processing Integrity
    CONFIDENTIALITY = "C"  # Confidentiality
    PRIVACY = "P"  # Privacy


class DORAArticle(Enum):
    """DORA Article references."""

    ART_5 = "Article 5"  # Governance
    ART_6 = "Article 6"  # ICT Risk Management Framework
    ART_7 = "Article 7"  # ICT Systems, Protocols and Tools
    ART_8 = "Article 8"  # Identification
    ART_9 = "Article 9"  # Protection and Prevention
    ART_10 = "Article 10"  # Detection
    ART_11 = "Article 11"  # Response and Recovery
    ART_12 = "Article 12"  # Backup Policies
    ART_13 = "Article 13"  # Learning and Evolving
    ART_14 = "Article 14"  # Communication
    ART_15 = "Article 15"  # ICT Business Continuity
    ART_17 = "Article 17"  # ICT Incident Management
    ART_19 = "Article 19"  # Incident Reporting
    ART_24 = "Article 24"  # Digital Resilience Testing
    ART_28 = "Article 28"  # Third-Party Risk
    ART_30 = "Article 30"  # Contractual Arrangements


class ControlStatus(Enum):
    """Control implementation status."""

    IMPLEMENTED = "implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    PLANNED = "planned"
    NOT_APPLICABLE = "not_applicable"
    GAP = "gap"


class EvidenceStatus(Enum):
    """Evidence collection status."""

    COLLECTED = "collected"
    PENDING = "pending"
    NOT_REQUIRED = "not_required"
    MISSING = "missing"


@dataclass
class ControlMapping:
    """Mapping between SOC2 and DORA controls."""

    mapping_id: str = ""
    soc2_control: str = ""  # e.g., "CC6.1"
    soc2_category: SOC2Category = SOC2Category.CC_SECURITY
    soc2_description: str = ""
    dora_article: DORAArticle = DORAArticle.ART_9
    dora_requirement: str = ""
    alignment_notes: str = ""
    shared_evidence: bool = True
    implementation_status: ControlStatus = ControlStatus.IMPLEMENTED

    def __post_init__(self):
        if not self.mapping_id:
            self.mapping_id = f"MAP-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class SharedControl:
    """Shared control between SOC2 and DORA."""

    control_id: str = ""
    name: str = ""
    description: str = ""
    soc2_mappings: List[str] = field(default_factory=list)  # SOC2 control IDs
    dora_articles: List[DORAArticle] = field(default_factory=list)
    evidence_requirements: List[str] = field(default_factory=list)
    status: ControlStatus = ControlStatus.IMPLEMENTED
    owner: str = ""
    last_tested: str = ""

    def __post_init__(self):
        if not self.control_id:
            self.control_id = f"CTRL-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class EvidenceRequirement:
    """Evidence requirement for audit."""

    evidence_id: str = ""
    name: str = ""
    description: str = ""
    control_ids: List[str] = field(default_factory=list)
    evidence_type: str = ""  # document, screenshot, log, report
    frequency: str = ""  # annual, quarterly, continuous
    status: EvidenceStatus = EvidenceStatus.PENDING
    collected_date: str = ""
    artifact_location: str = ""

    def __post_init__(self):
        if not self.evidence_id:
            self.evidence_id = f"EVD-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ComplianceGap:
    """Identified compliance gap."""

    gap_id: str = ""
    framework: str = ""  # SOC2, DORA, or both
    control_reference: str = ""
    description: str = ""
    severity: str = "medium"  # critical, high, medium, low
    remediation_plan: str = ""
    target_date: str = ""
    status: str = "open"  # open, in_progress, closed
    owner: str = ""

    def __post_init__(self):
        if not self.gap_id:
            self.gap_id = f"GAP-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class SOC2DORAMappingConfig:
    """Configuration for SOC2DORAMapper."""

    log_all_events: bool = True
    log_path: str = "logs/core/soc2_dora"


# Pre-defined control mappings
CONTROL_MAPPINGS = [
    # Security (CC) -> Multiple DORA Articles
    {
        "soc2_control": "CC1.1",
        "soc2_category": SOC2Category.CC_SECURITY,
        "soc2_description": "COSO Principle 1: The entity demonstrates commitment to integrity and ethical values",
        "dora_article": DORAArticle.ART_5,
        "dora_requirement": "Management body oversight and ICT governance",
        "shared_evidence": True,
    },
    {
        "soc2_control": "CC3.1",
        "soc2_category": SOC2Category.CC_SECURITY,
        "soc2_description": "Risk assessment and management",
        "dora_article": DORAArticle.ART_6,
        "dora_requirement": "ICT Risk Management Framework",
        "shared_evidence": True,
    },
    {
        "soc2_control": "CC6.1",
        "soc2_category": SOC2Category.CC_SECURITY,
        "soc2_description": "Logical and physical access controls",
        "dora_article": DORAArticle.ART_9,
        "dora_requirement": "Protection and Prevention",
        "shared_evidence": True,
    },
    {
        "soc2_control": "CC6.6",
        "soc2_category": SOC2Category.CC_SECURITY,
        "soc2_description": "Restrict access to system components",
        "dora_article": DORAArticle.ART_9,
        "dora_requirement": "Access control and authentication",
        "shared_evidence": True,
    },
    {
        "soc2_control": "CC7.1",
        "soc2_category": SOC2Category.CC_SECURITY,
        "soc2_description": "Detect and monitor anomalies",
        "dora_article": DORAArticle.ART_10,
        "dora_requirement": "Detection of anomalous activities",
        "shared_evidence": True,
    },
    {
        "soc2_control": "CC7.2",
        "soc2_category": SOC2Category.CC_SECURITY,
        "soc2_description": "Security incident handling",
        "dora_article": DORAArticle.ART_17,
        "dora_requirement": "ICT-related incident management",
        "shared_evidence": True,
    },
    {
        "soc2_control": "CC7.3",
        "soc2_category": SOC2Category.CC_SECURITY,
        "soc2_description": "Incident response procedures",
        "dora_article": DORAArticle.ART_11,
        "dora_requirement": "Response and recovery procedures",
        "shared_evidence": True,
    },
    {
        "soc2_control": "CC9.1",
        "soc2_category": SOC2Category.CC_SECURITY,
        "soc2_description": "Risk mitigation activities",
        "dora_article": DORAArticle.ART_28,
        "dora_requirement": "Third-party ICT risk management",
        "shared_evidence": True,
    },
    # Availability (A) -> DORA Continuity
    {
        "soc2_control": "A1.1",
        "soc2_category": SOC2Category.AVAILABILITY,
        "soc2_description": "Maintain system availability commitments",
        "dora_article": DORAArticle.ART_15,
        "dora_requirement": "ICT business continuity policy",
        "shared_evidence": True,
    },
    {
        "soc2_control": "A1.2",
        "soc2_category": SOC2Category.AVAILABILITY,
        "soc2_description": "Backup and recovery procedures",
        "dora_article": DORAArticle.ART_12,
        "dora_requirement": "Backup policies and procedures",
        "shared_evidence": True,
    },
    # Confidentiality (C) -> DORA Protection
    {
        "soc2_control": "C1.1",
        "soc2_category": SOC2Category.CONFIDENTIALITY,
        "soc2_description": "Identification of confidential information",
        "dora_article": DORAArticle.ART_9,
        "dora_requirement": "Data classification and protection",
        "shared_evidence": True,
    },
]


def get_control_mappings() -> List[Dict[str, Any]]:
    """Get pre-defined control mappings."""
    return CONTROL_MAPPINGS.copy()


class SOC2DORAMapper:
    """SOC2-DORA Control Mapper."""

    def __init__(self, config: Optional[SOC2DORAMappingConfig] = None):
        self.config = config or SOC2DORAMappingConfig()
        self._mappings: Dict[str, ControlMapping] = {}
        self._controls: Dict[str, SharedControl] = {}
        self._evidence: Dict[str, EvidenceRequirement] = {}
        self._gaps: Dict[str, ComplianceGap] = {}
        self._lock = threading.RLock()
        self._init_default_mappings()
        logger.info("SOC2DORAMapper initialized")

    def _init_default_mappings(self) -> None:
        """Initialize default control mappings."""
        for mapping_data in CONTROL_MAPPINGS:
            self.add_mapping(
                soc2_control=mapping_data["soc2_control"],
                soc2_category=mapping_data["soc2_category"],
                soc2_description=mapping_data["soc2_description"],
                dora_article=mapping_data["dora_article"],
                dora_requirement=mapping_data["dora_requirement"],
                shared_evidence=mapping_data.get("shared_evidence", True),
            )

    def add_mapping(
        self,
        soc2_control: str,
        soc2_category: SOC2Category,
        soc2_description: str,
        dora_article: DORAArticle,
        dora_requirement: str,
        shared_evidence: bool = True,
    ) -> ControlMapping:
        """Add a control mapping."""
        mapping = ControlMapping(
            soc2_control=soc2_control,
            soc2_category=soc2_category,
            soc2_description=soc2_description,
            dora_article=dora_article,
            dora_requirement=dora_requirement,
            shared_evidence=shared_evidence,
        )
        with self._lock:
            self._mappings[mapping.mapping_id] = mapping
        return mapping

    def get_mappings_by_soc2(self, soc2_control: str) -> List[ControlMapping]:
        """Get DORA mappings for a SOC2 control."""
        with self._lock:
            return [m for m in self._mappings.values() if m.soc2_control == soc2_control]

    def get_mappings_by_dora(self, dora_article: DORAArticle) -> List[ControlMapping]:
        """Get SOC2 mappings for a DORA article."""
        with self._lock:
            return [m for m in self._mappings.values() if m.dora_article == dora_article]

    def record_gap(
        self,
        framework: str,
        control_reference: str,
        description: str,
        severity: str = "medium",
        remediation_plan: str = "",
        owner: str = "",
    ) -> ComplianceGap:
        """Record a compliance gap."""
        gap = ComplianceGap(
            framework=framework,
            control_reference=control_reference,
            description=description,
            severity=severity,
            remediation_plan=remediation_plan,
            owner=owner,
        )
        with self._lock:
            self._gaps[gap.gap_id] = gap
        return gap

    def get_open_gaps(self) -> List[ComplianceGap]:
        """Get all open gaps."""
        with self._lock:
            return [g for g in self._gaps.values() if g.status != "closed"]

    def get_mapping_summary(self) -> Dict[str, Any]:
        """Get mapping summary."""
        with self._lock:
            mappings = list(self._mappings.values())
            gaps = list(self._gaps.values())

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_mappings": len(mappings),
            "by_soc2_category": {
                cat.value: sum(1 for m in mappings if m.soc2_category == cat)
                for cat in SOC2Category
            },
            "by_dora_article": {
                art.value: sum(1 for m in mappings if m.dora_article == art) for art in DORAArticle
            },
            "shared_evidence_count": sum(1 for m in mappings if m.shared_evidence),
            "gaps": {
                "total": len(gaps),
                "open": sum(1 for g in gaps if g.status == "open"),
                "by_severity": {
                    sev: sum(1 for g in gaps if g.severity == sev)
                    for sev in ["critical", "high", "medium", "low"]
                },
            },
        }

    def export_mapping_matrix(self) -> Dict[str, Any]:
        """Export full mapping matrix for audit."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "mappings": [asdict(m) for m in self._mappings.values()],
                "gaps": [asdict(g) for g in self._gaps.values()],
                "summary": self.get_mapping_summary(),
            }


def create_soc2_dora_mapper(
    config: Optional[SOC2DORAMappingConfig] = None,
) -> SOC2DORAMapper:
    """Create a SOC2DORAMapper instance."""
    return SOC2DORAMapper(config=config)
