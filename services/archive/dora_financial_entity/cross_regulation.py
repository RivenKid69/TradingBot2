# -*- coding: utf-8 -*-
"""
DORA Cross-Regulation Integration (Phase 5).

Implements alignment between DORA and neighbouring regulatory regimes:
    - EU AI Act Article 73 (serious incident reporting)
    - NIS2 incident notification timelines
    - MiFID II business continuity / kill-switch expectations

Focus areas:
    1. Incident reporting timeline harmonisation
    2. Risk framework alignment and deduplication
    3. Logging alignment to avoid duplicated controls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class Regulation(Enum):
    """Supported regulations for alignment."""

    DORA = "dora"
    AI_ACT = "ai_act"
    NIS2 = "nis2"
    MIFID_II = "mifid_ii"


@dataclass
class ReportingRequirement:
    """Regulatory reporting deadline."""

    regulation: Regulation
    stage: str
    deadline: datetime
    rationale: str
    required_fields: List[str] = field(default_factory=list)


@dataclass
class IncidentAlignmentResult:
    """Aggregate reporting deadlines across regimes."""

    detection_time: datetime
    classification_time: datetime
    requirements: List[ReportingRequirement] = field(default_factory=list)
    alignment_id: str = field(default_factory=lambda: f"ALIGN-{uuid4().hex[:8].upper()}")

    @property
    def earliest_deadline(self) -> datetime:
        return min(req.deadline for req in self.requirements)

    def schedule(self) -> List[ReportingRequirement]:
        """Return requirements sorted by deadline."""
        return sorted(self.requirements, key=lambda req: req.deadline)


@dataclass
class RiskFrameworkAlignment:
    """Combined risk registry view across regulations."""

    combined_risks: Dict[str, Dict[str, str]]  # risk_id -> {source: level}
    overlaps: Set[str] = field(default_factory=set)
    only_dora: Set[str] = field(default_factory=set)
    only_ai_act: Set[str] = field(default_factory=set)


@dataclass
class LoggingAlignmentResult:
    """Alignment result for logging obligations."""

    combined_categories: Set[str]
    missing_in_dora: Set[str]
    missing_in_ai_act: Set[str]
    unified_schema: Dict[str, str]


class DORARegulationIntegration:
    """
    Cross-regulation integration coordinator.
    """

    def __init__(self):
        self.nis2_initial_deadline_hours = 24
        self.nis2_intermediate_deadline_hours = 72
        self.nis2_final_deadline_days = 30

        self.ai_act_deadline_hours = 24  # Article 73 serious incident

        self.dora_detection_deadline_hours = 24  # Article 19 initial notification
        self.dora_classification_deadline_hours = 4  # After classification as major

    # ------------------------------------------------------------------ #
    # Incident Reporting Alignment
    # ------------------------------------------------------------------ #
    def _build_dora_requirements(
        self,
        detection_time: datetime,
        classification_time: datetime,
    ) -> List[ReportingRequirement]:
        detection_deadline = detection_time + timedelta(hours=self.dora_detection_deadline_hours)
        classification_deadline = classification_time + timedelta(
            hours=self.dora_classification_deadline_hours
        )
        earliest = min(detection_deadline, classification_deadline)

        requirements = [
            ReportingRequirement(
                regulation=Regulation.DORA,
                stage="initial_notification",
                deadline=earliest,
                rationale="Article 19: earlier of detection+24h or classification+4h",
                required_fields=["incident_id", "classification", "services_affected"],
            ),
            ReportingRequirement(
                regulation=Regulation.DORA,
                stage="intermediate",
                deadline=earliest + timedelta(hours=72),
                rationale="Article 19: 72h after initial notification",
                required_fields=["updates", "mitigation_progress"],
            ),
            ReportingRequirement(
                regulation=Regulation.DORA,
                stage="final",
                deadline=earliest + timedelta(days=30),
                rationale="Article 19: final report within one month",
                required_fields=["root_cause", "lessons_learned"],
            ),
        ]
        return requirements

    def _build_nis2_requirements(self, detection_time: datetime) -> List[ReportingRequirement]:
        return [
            ReportingRequirement(
                regulation=Regulation.NIS2,
                stage="early_warning",
                deadline=detection_time + timedelta(hours=self.nis2_initial_deadline_hours),
                rationale="NIS2 Article 23: early warning within 24h",
                required_fields=["incident_id", "initial_assessment"],
            ),
            ReportingRequirement(
                regulation=Regulation.NIS2,
                stage="incident_notification",
                deadline=detection_time + timedelta(hours=self.nis2_intermediate_deadline_hours),
                rationale="NIS2 Article 23: detailed notification within 72h",
                required_fields=["impact", "mitigation_actions"],
            ),
            ReportingRequirement(
                regulation=Regulation.NIS2,
                stage="final_report",
                deadline=detection_time + timedelta(days=self.nis2_final_deadline_days),
                rationale="NIS2 Article 23: final report within 1 month",
                required_fields=["root_cause", "preventive_measures"],
            ),
        ]

    def _build_ai_act_requirements(self, detection_time: datetime) -> List[ReportingRequirement]:
        return [
            ReportingRequirement(
                regulation=Regulation.AI_ACT,
                stage="serious_incident",
                deadline=detection_time + timedelta(hours=self.ai_act_deadline_hours),
                rationale="AI Act Article 73: 24h serious incident notification",
                required_fields=["ai_system_id", "incident_description", "harm_assessment"],
            )
        ]

    def align_incident_reporting(
        self,
        detection_time: datetime,
        classification_time: datetime,
    ) -> IncidentAlignmentResult:
        """Create unified schedule across DORA, AI Act and NIS2."""
        requirements: List[ReportingRequirement] = []
        requirements.extend(self._build_dora_requirements(detection_time, classification_time))
        requirements.extend(self._build_nis2_requirements(detection_time))
        requirements.extend(self._build_ai_act_requirements(detection_time))
        return IncidentAlignmentResult(
            detection_time=detection_time,
            classification_time=classification_time,
            requirements=requirements,
        )

    # ------------------------------------------------------------------ #
    # Risk Framework Alignment
    # ------------------------------------------------------------------ #
    def integrate_risk_frameworks(
        self,
        dora_risks: Dict[str, str],
        ai_act_risks: Dict[str, str],
    ) -> RiskFrameworkAlignment:
        """
        Merge AI Act and DORA risk registries.

        Args:
            dora_risks: mapping risk_id -> risk_level
            ai_act_risks: mapping risk_id -> risk_level
        """
        combined: Dict[str, Dict[str, str]] = {}
        overlaps: Set[str] = set()

        for risk_id, level in dora_risks.items():
            combined[risk_id] = {"dora": level}

        for risk_id, level in ai_act_risks.items():
            if risk_id in combined:
                combined[risk_id]["ai_act"] = level
                overlaps.add(risk_id)
            else:
                combined[risk_id] = {"ai_act": level}

        only_dora = (
            set(dora_risks.keys()) - overlaps - (set(ai_act_risks.keys()) & set(dora_risks.keys()))
        )
        only_ai_act = (
            set(ai_act_risks.keys())
            - overlaps
            - (set(ai_act_risks.keys()) & set(dora_risks.keys()))
        )

        return RiskFrameworkAlignment(
            combined_risks=combined,
            overlaps=overlaps,
            only_dora=only_dora,
            only_ai_act=only_ai_act,
        )

    # ------------------------------------------------------------------ #
    # Logging Alignment
    # ------------------------------------------------------------------ #
    def align_logging_systems(
        self,
        dora_logs: Set[str],
        ai_act_logs: Set[str],
    ) -> LoggingAlignmentResult:
        """
        Align logging obligations and identify gaps.
        """
        minimum_schema = {
            "timestamp": "ISO8601",
            "actor": "string",
            "event_type": "string",
            "system": "string",
            "regulation_source": "string",
        }

        required_dora = {"ict_events", "security_events", "incident_logs"}
        required_ai_act = {"ai_events", "model_decisions", "human_override"}

        missing_in_dora = required_dora - dora_logs
        missing_in_ai_act = required_ai_act - ai_act_logs

        combined = dora_logs.union(ai_act_logs).union(required_dora).union(required_ai_act)

        return LoggingAlignmentResult(
            combined_categories=combined,
            missing_in_dora=missing_in_dora,
            missing_in_ai_act=missing_in_ai_act,
            unified_schema=minimum_schema,
        )
