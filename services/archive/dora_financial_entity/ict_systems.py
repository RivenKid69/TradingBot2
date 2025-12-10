# -*- coding: utf-8 -*-
"""
DORA ICT Systems, Protocols and Tools Module (Article 7).

Regulation (EU) 2022/2554 Article 7 defines requirements for:
    - Use of appropriate and up-to-date ICT systems
    - Reliable, resilient and sufficient capacity systems
    - Automation for high degree of digital operational resilience
    - Sound and comprehensive strategies, policies and procedures

This module provides:
    1. ICTSystemRegistry - System inventory and classification
    2. CapacityManager - Capacity monitoring and planning
    3. SystemReliability - Reliability metrics and monitoring
    4. AutomationAssessment - Automation level tracking
    5. ICTSystemsManager - Main systems management orchestrator

References:
    - Article 7 DORA: https://www.digital-operational-resilience-act.com/Article_7.html
    - RTS on ICT Risk Management: CDR 2024/1774
    - ISO 22301:2019 Business Continuity Management
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class SystemCriticality(Enum):
    """System criticality levels per DORA."""
    CRITICAL = "critical"  # Supports critical/important functions
    IMPORTANT = "important"  # Supports important functions
    STANDARD = "standard"  # Normal business operations
    LOW = "low"  # Non-essential systems


class SystemType(Enum):
    """ICT system types."""
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    MIDDLEWARE = "middleware"
    OPERATING_SYSTEM = "operating_system"
    CLOUD_SERVICE = "cloud_service"
    EXTERNAL_SERVICE = "external_service"


class SystemStatus(Enum):
    """System operational status."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    FAILED = "failed"
    DECOMMISSIONED = "decommissioned"


class CapacityStatus(Enum):
    """Capacity status levels."""
    OPTIMAL = "optimal"  # Under normal thresholds
    WARNING = "warning"  # Approaching limits
    CRITICAL = "critical"  # Near capacity limits
    EXCEEDED = "exceeded"  # Over capacity


class AutomationLevel(Enum):
    """Automation maturity levels."""
    MANUAL = "manual"  # Fully manual processes
    PARTIALLY_AUTOMATED = "partially_automated"  # Some automation
    MOSTLY_AUTOMATED = "mostly_automated"  # High automation
    FULLY_AUTOMATED = "fully_automated"  # Complete automation


class SupportLevel(Enum):
    """Vendor support levels."""
    ACTIVE_SUPPORT = "active_support"  # Full vendor support
    EXTENDED_SUPPORT = "extended_support"  # Limited support
    END_OF_LIFE = "end_of_life"  # No support
    UNSUPPORTED = "unsupported"  # Community only


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ICTSystem:
    """
    ICT System record per DORA Article 7.

    Documents systems that support financial entity operations.
    """
    system_id: str = ""
    name: str = ""
    description: str = ""
    system_type: SystemType = SystemType.APPLICATION
    criticality: SystemCriticality = SystemCriticality.STANDARD

    # Classification
    supports_critical_function: bool = False
    supports_important_function: bool = False
    critical_function_ids: List[str] = field(default_factory=list)

    # Technical details
    version: str = ""
    vendor: str = ""
    technology_stack: List[str] = field(default_factory=list)
    deployment_model: str = ""  # "on_premise", "cloud", "hybrid"
    cloud_provider: str = ""

    # Dependencies
    upstream_dependencies: List[str] = field(default_factory=list)
    downstream_dependencies: List[str] = field(default_factory=list)
    third_party_provider_id: str = ""

    # Support status
    support_level: SupportLevel = SupportLevel.ACTIVE_SUPPORT
    support_end_date: Optional[str] = None
    next_upgrade_date: Optional[str] = None

    # Capacity
    capacity_metrics: Dict[str, float] = field(default_factory=dict)
    capacity_thresholds: Dict[str, float] = field(default_factory=dict)

    # Availability
    target_availability_pct: float = 99.9
    rto_hours: float = 4.0
    rpo_hours: float = 1.0

    # Ownership
    system_owner: str = ""
    technical_owner: str = ""
    business_owner: str = ""

    # Status
    status: SystemStatus = SystemStatus.OPERATIONAL
    last_updated: str = ""

    def __post_init__(self):
        if not self.system_id:
            self.system_id = f"SYS-{uuid.uuid4().hex[:8].upper()}"
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()

    @property
    def is_critical(self) -> bool:
        """Check if system is critical."""
        return (
            self.criticality == SystemCriticality.CRITICAL or
            self.supports_critical_function
        )


@dataclass
class CapacityMetric:
    """
    Capacity metric measurement.
    """
    metric_id: str = ""
    system_id: str = ""
    metric_name: str = ""  # "cpu", "memory", "storage", "network", "connections"
    metric_type: str = ""  # "utilization", "count", "throughput"

    # Values
    current_value: float = 0.0
    threshold_warning: float = 70.0
    threshold_critical: float = 85.0
    maximum_capacity: float = 100.0

    # Trend
    trend_direction: str = ""  # "increasing", "stable", "decreasing"
    forecast_days_to_threshold: Optional[int] = None

    # Status
    status: CapacityStatus = CapacityStatus.OPTIMAL
    measurement_time: str = ""

    def __post_init__(self):
        if not self.metric_id:
            self.metric_id = f"CAP-{uuid.uuid4().hex[:8].upper()}"
        if not self.measurement_time:
            self.measurement_time = datetime.now(timezone.utc).isoformat()

    def evaluate_status(self) -> CapacityStatus:
        """Evaluate capacity status based on current value."""
        utilization = (self.current_value / self.maximum_capacity) * 100 if self.maximum_capacity > 0 else 0

        if utilization >= 100:
            return CapacityStatus.EXCEEDED
        elif utilization >= self.threshold_critical:
            return CapacityStatus.CRITICAL
        elif utilization >= self.threshold_warning:
            return CapacityStatus.WARNING
        else:
            return CapacityStatus.OPTIMAL


@dataclass
class ReliabilityMetric:
    """
    System reliability metric.
    """
    metric_id: str = ""
    system_id: str = ""
    period_start: str = ""
    period_end: str = ""

    # Availability
    uptime_hours: float = 0.0
    downtime_hours: float = 0.0
    availability_pct: float = 100.0
    target_availability_pct: float = 99.9

    # Incidents
    incident_count: int = 0
    critical_incident_count: int = 0
    mean_time_between_failures_hours: float = 0.0
    mean_time_to_repair_hours: float = 0.0

    # Performance
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    error_rate_pct: float = 0.0

    def __post_init__(self):
        if not self.metric_id:
            self.metric_id = f"REL-{uuid.uuid4().hex[:8].upper()}"

    def calculate_availability(self) -> float:
        """Calculate availability percentage."""
        total_hours = self.uptime_hours + self.downtime_hours
        if total_hours > 0:
            self.availability_pct = (self.uptime_hours / total_hours) * 100
        return self.availability_pct

    @property
    def meets_target(self) -> bool:
        """Check if availability meets target."""
        return self.availability_pct >= self.target_availability_pct


@dataclass
class AutomationCapability:
    """
    Automation capability assessment.
    """
    capability_id: str = ""
    system_id: str = ""
    process_name: str = ""
    process_category: str = ""  # "deployment", "monitoring", "backup", "recovery", "patching"

    # Automation level
    current_level: AutomationLevel = AutomationLevel.MANUAL
    target_level: AutomationLevel = AutomationLevel.MOSTLY_AUTOMATED
    automation_score: float = 0.0  # 0-100

    # Details
    manual_steps: List[str] = field(default_factory=list)
    automated_steps: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)

    # Improvement
    improvement_plan: str = ""
    target_date: Optional[str] = None

    def __post_init__(self):
        if not self.capability_id:
            self.capability_id = f"AUTO-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class SystemUpgrade:
    """
    System upgrade/update record.
    """
    upgrade_id: str = ""
    system_id: str = ""
    upgrade_type: str = ""  # "version_upgrade", "patch", "security_update", "migration"

    # Versions
    current_version: str = ""
    target_version: str = ""

    # Schedule
    planned_date: Optional[str] = None
    executed_date: Optional[str] = None
    rollback_date: Optional[str] = None

    # Status
    status: str = ""  # "planned", "approved", "in_progress", "completed", "rolled_back", "cancelled"

    # Details
    change_description: str = ""
    risk_assessment: str = ""
    test_results: str = ""
    approval_reference: str = ""

    # Rollback
    rollback_plan: str = ""
    rollback_executed: bool = False

    def __post_init__(self):
        if not self.upgrade_id:
            self.upgrade_id = f"UPG-{uuid.uuid4().hex[:8].upper()}"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ICTSystemsConfig:
    """Configuration for ICT Systems Management."""
    # Capacity management
    default_warning_threshold_pct: float = 70.0
    default_critical_threshold_pct: float = 85.0
    capacity_check_interval_minutes: int = 5
    capacity_forecast_days: int = 30

    # Reliability targets
    default_target_availability_pct: float = 99.9
    default_rto_hours: float = 4.0
    default_rpo_hours: float = 1.0

    # Support requirements
    warn_days_before_eol: int = 180
    critical_days_before_eol: int = 90

    # Automation targets
    target_automation_score: float = 80.0

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/dora/ict_systems"

    # Callbacks
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Main ICT Systems Manager
# =============================================================================

class DORAICTSystemsManager:
    """
    DORA Article 7 compliant ICT Systems Manager.

    Manages:
    - ICT system inventory and classification
    - Capacity monitoring and planning
    - Reliability metrics tracking
    - Automation assessment
    - System lifecycle management

    Usage:
        config = ICTSystemsConfig()
        manager = DORAICTSystemsManager(config)

        # Register a system
        system = manager.register_system(
            name="Trading Platform",
            system_type=SystemType.APPLICATION,
            criticality=SystemCriticality.CRITICAL,
        )

        # Record capacity metrics
        manager.record_capacity_metric(
            system_id=system.system_id,
            metric_name="cpu",
            current_value=45.0,
            maximum_capacity=100.0,
        )

        # Get system health
        health = manager.get_system_health(system.system_id)
    """

    def __init__(self, config: Optional[ICTSystemsConfig] = None):
        """Initialize ICT Systems Manager."""
        self.config = config or ICTSystemsConfig()

        # Data stores
        self._systems: Dict[str, ICTSystem] = {}
        self._capacity_metrics: Dict[str, List[CapacityMetric]] = {}  # system_id -> metrics
        self._reliability_metrics: Dict[str, List[ReliabilityMetric]] = {}
        self._automation_capabilities: Dict[str, List[AutomationCapability]] = {}
        self._upgrades: Dict[str, SystemUpgrade] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAICTSystemsManager initialized")

    # =========================================================================
    # System Registry (Article 7)
    # =========================================================================

    def register_system(
        self,
        name: str,
        system_type: SystemType,
        criticality: SystemCriticality,
        description: str = "",
        version: str = "",
        vendor: str = "",
        supports_critical_function: bool = False,
        supports_important_function: bool = False,
        critical_function_ids: Optional[List[str]] = None,
        deployment_model: str = "on_premise",
        system_owner: str = "",
        target_availability_pct: Optional[float] = None,
        rto_hours: Optional[float] = None,
        rpo_hours: Optional[float] = None,
    ) -> ICTSystem:
        """
        Register an ICT system.

        Per Article 7, entities must use appropriate ICT systems with
        sufficient capacity and reliability.

        Args:
            name: System name
            system_type: Type of system
            criticality: Criticality level
            description: System description
            version: Current version
            vendor: Vendor name
            supports_critical_function: Supports critical functions
            supports_important_function: Supports important functions
            critical_function_ids: IDs of supported critical functions
            deployment_model: Deployment model
            system_owner: System owner
            target_availability_pct: Target availability
            rto_hours: Recovery time objective
            rpo_hours: Recovery point objective

        Returns:
            Registered ICTSystem
        """
        system = ICTSystem(
            name=name,
            system_type=system_type,
            criticality=criticality,
            description=description,
            version=version,
            vendor=vendor,
            supports_critical_function=supports_critical_function,
            supports_important_function=supports_important_function,
            critical_function_ids=critical_function_ids or [],
            deployment_model=deployment_model,
            system_owner=system_owner,
            target_availability_pct=target_availability_pct or self.config.default_target_availability_pct,
            rto_hours=rto_hours or self.config.default_rto_hours,
            rpo_hours=rpo_hours or self.config.default_rpo_hours,
        )

        with self._lock:
            self._systems[system.system_id] = system
            self._capacity_metrics[system.system_id] = []
            self._reliability_metrics[system.system_id] = []
            self._automation_capabilities[system.system_id] = []

        self._log_event("system_registered", {
            "system_id": system.system_id,
            "name": name,
            "criticality": criticality.value,
        })

        logger.info(f"System registered: {name} ({criticality.value})")
        return system

    def get_system(self, system_id: str) -> Optional[ICTSystem]:
        """Get system by ID."""
        with self._lock:
            return self._systems.get(system_id)

    def get_all_systems(self) -> List[ICTSystem]:
        """Get all registered systems."""
        with self._lock:
            return list(self._systems.values())

    def get_critical_systems(self) -> List[ICTSystem]:
        """Get all critical systems."""
        with self._lock:
            return [s for s in self._systems.values() if s.is_critical]

    def get_systems_by_type(self, system_type: SystemType) -> List[ICTSystem]:
        """Get systems by type."""
        with self._lock:
            return [s for s in self._systems.values() if s.system_type == system_type]

    def update_system_status(
        self,
        system_id: str,
        status: SystemStatus,
        reason: str = "",
    ) -> Optional[ICTSystem]:
        """Update system operational status."""
        with self._lock:
            if system_id not in self._systems:
                return None

            system = self._systems[system_id]
            system.status = status
            system.last_updated = datetime.now(timezone.utc).isoformat()

        self._log_event("system_status_updated", {
            "system_id": system_id,
            "status": status.value,
            "reason": reason,
        })

        return system

    def add_system_dependency(
        self,
        system_id: str,
        depends_on_system_id: str,
    ) -> Tuple[Optional[ICTSystem], Optional[ICTSystem]]:
        """Add dependency between systems."""
        with self._lock:
            if system_id not in self._systems or depends_on_system_id not in self._systems:
                return None, None

            system = self._systems[system_id]
            dependency = self._systems[depends_on_system_id]

            if depends_on_system_id not in system.upstream_dependencies:
                system.upstream_dependencies.append(depends_on_system_id)

            if system_id not in dependency.downstream_dependencies:
                dependency.downstream_dependencies.append(system_id)

        return system, dependency

    def get_system_dependencies(
        self,
        system_id: str,
        include_transitive: bool = False,
    ) -> List[ICTSystem]:
        """Get system dependencies."""
        with self._lock:
            if system_id not in self._systems:
                return []

            system = self._systems[system_id]
            dependencies = []

            for dep_id in system.upstream_dependencies:
                if dep_id in self._systems:
                    dependencies.append(self._systems[dep_id])

                    if include_transitive:
                        # Recursively get transitive dependencies
                        transitive = self.get_system_dependencies(dep_id, True)
                        for t in transitive:
                            if t.system_id not in [d.system_id for d in dependencies]:
                                dependencies.append(t)

        return dependencies

    # =========================================================================
    # Capacity Management (Article 7)
    # =========================================================================

    def record_capacity_metric(
        self,
        system_id: str,
        metric_name: str,
        current_value: float,
        maximum_capacity: float,
        metric_type: str = "utilization",
        threshold_warning: Optional[float] = None,
        threshold_critical: Optional[float] = None,
    ) -> Optional[CapacityMetric]:
        """
        Record a capacity metric for a system.

        Per Article 7, systems must have sufficient capacity.

        Args:
            system_id: System ID
            metric_name: Metric name (cpu, memory, storage, etc.)
            current_value: Current value
            maximum_capacity: Maximum capacity
            metric_type: Type of metric
            threshold_warning: Warning threshold
            threshold_critical: Critical threshold

        Returns:
            Recorded CapacityMetric
        """
        if system_id not in self._systems:
            return None

        metric = CapacityMetric(
            system_id=system_id,
            metric_name=metric_name,
            metric_type=metric_type,
            current_value=current_value,
            maximum_capacity=maximum_capacity,
            threshold_warning=threshold_warning or self.config.default_warning_threshold_pct,
            threshold_critical=threshold_critical or self.config.default_critical_threshold_pct,
        )

        metric.status = metric.evaluate_status()

        with self._lock:
            self._capacity_metrics[system_id].append(metric)

            # Update system capacity metrics
            system = self._systems[system_id]
            system.capacity_metrics[metric_name] = current_value

        # Alert if critical
        if metric.status == CapacityStatus.CRITICAL or metric.status == CapacityStatus.EXCEEDED:
            self._alert_capacity_issue(system_id, metric)

        return metric

    def get_capacity_metrics(
        self,
        system_id: str,
        metric_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[CapacityMetric]:
        """Get capacity metrics for a system."""
        with self._lock:
            metrics = self._capacity_metrics.get(system_id, [])

            if metric_name:
                metrics = [m for m in metrics if m.metric_name == metric_name]

            return metrics[-limit:]

    def get_systems_with_capacity_issues(self) -> List[Tuple[ICTSystem, List[CapacityMetric]]]:
        """Get systems with capacity warnings or critical status."""
        issues = []

        with self._lock:
            for system_id, metrics in self._capacity_metrics.items():
                system = self._systems.get(system_id)
                if not system:
                    continue

                # Get latest metrics by name
                latest_metrics: Dict[str, CapacityMetric] = {}
                for metric in metrics:
                    latest_metrics[metric.metric_name] = metric

                problem_metrics = [
                    m for m in latest_metrics.values()
                    if m.status in (CapacityStatus.WARNING, CapacityStatus.CRITICAL, CapacityStatus.EXCEEDED)
                ]

                if problem_metrics:
                    issues.append((system, problem_metrics))

        return issues

    def _alert_capacity_issue(
        self,
        system_id: str,
        metric: CapacityMetric,
    ) -> None:
        """Alert on capacity issue."""
        if self.config.alert_callback:
            try:
                system = self._systems.get(system_id)
                self.config.alert_callback("capacity_issue", {
                    "system_id": system_id,
                    "system_name": system.name if system else "Unknown",
                    "metric_name": metric.metric_name,
                    "status": metric.status.value,
                    "current_value": metric.current_value,
                    "maximum_capacity": metric.maximum_capacity,
                })
            except Exception as e:
                logger.error(f"Capacity alert callback failed: {e}")

        logger.warning(
            f"Capacity issue: {system_id} {metric.metric_name} "
            f"at {metric.current_value}/{metric.maximum_capacity} ({metric.status.value})"
        )

    # =========================================================================
    # Reliability Tracking (Article 7)
    # =========================================================================

    def record_reliability_metric(
        self,
        system_id: str,
        period_start: str,
        period_end: str,
        uptime_hours: float,
        downtime_hours: float,
        incident_count: int = 0,
        critical_incident_count: int = 0,
        avg_response_time_ms: float = 0.0,
        error_rate_pct: float = 0.0,
    ) -> Optional[ReliabilityMetric]:
        """
        Record reliability metrics for a system.

        Per Article 7, systems must be reliable and resilient.

        Args:
            system_id: System ID
            period_start: Period start date
            period_end: Period end date
            uptime_hours: Hours of uptime
            downtime_hours: Hours of downtime
            incident_count: Number of incidents
            critical_incident_count: Number of critical incidents
            avg_response_time_ms: Average response time
            error_rate_pct: Error rate percentage

        Returns:
            Recorded ReliabilityMetric
        """
        if system_id not in self._systems:
            return None

        system = self._systems[system_id]

        metric = ReliabilityMetric(
            system_id=system_id,
            period_start=period_start,
            period_end=period_end,
            uptime_hours=uptime_hours,
            downtime_hours=downtime_hours,
            incident_count=incident_count,
            critical_incident_count=critical_incident_count,
            avg_response_time_ms=avg_response_time_ms,
            error_rate_pct=error_rate_pct,
            target_availability_pct=system.target_availability_pct,
        )

        metric.calculate_availability()

        # Calculate MTBF
        if incident_count > 0:
            total_hours = uptime_hours + downtime_hours
            metric.mean_time_between_failures_hours = total_hours / incident_count

        # Calculate MTTR
        if incident_count > 0 and downtime_hours > 0:
            metric.mean_time_to_repair_hours = downtime_hours / incident_count

        with self._lock:
            self._reliability_metrics[system_id].append(metric)

        self._log_event("reliability_metric_recorded", {
            "system_id": system_id,
            "availability_pct": metric.availability_pct,
            "meets_target": metric.meets_target,
        })

        return metric

    def get_reliability_metrics(
        self,
        system_id: str,
        limit: int = 12,
    ) -> List[ReliabilityMetric]:
        """Get reliability metrics for a system."""
        with self._lock:
            metrics = self._reliability_metrics.get(system_id, [])
            return metrics[-limit:]

    def get_systems_below_availability_target(self) -> List[Tuple[ICTSystem, ReliabilityMetric]]:
        """Get systems not meeting availability targets."""
        below_target = []

        with self._lock:
            for system_id, metrics in self._reliability_metrics.items():
                if not metrics:
                    continue

                system = self._systems.get(system_id)
                if not system:
                    continue

                latest_metric = metrics[-1]
                if not latest_metric.meets_target:
                    below_target.append((system, latest_metric))

        return below_target

    # =========================================================================
    # Automation Assessment (Article 7)
    # =========================================================================

    def record_automation_capability(
        self,
        system_id: str,
        process_name: str,
        process_category: str,
        current_level: AutomationLevel,
        target_level: AutomationLevel = AutomationLevel.MOSTLY_AUTOMATED,
        manual_steps: Optional[List[str]] = None,
        automated_steps: Optional[List[str]] = None,
        tools_used: Optional[List[str]] = None,
        improvement_plan: str = "",
    ) -> Optional[AutomationCapability]:
        """
        Record automation capability for a system process.

        Per Article 7, systems should support automation for
        high degree of digital operational resilience.

        Args:
            system_id: System ID
            process_name: Process name
            process_category: Category (deployment, monitoring, etc.)
            current_level: Current automation level
            target_level: Target automation level
            manual_steps: Manual steps
            automated_steps: Automated steps
            tools_used: Tools used for automation
            improvement_plan: Plan to improve automation

        Returns:
            Recorded AutomationCapability
        """
        if system_id not in self._systems:
            return None

        # Calculate automation score
        total_steps = len(manual_steps or []) + len(automated_steps or [])
        automated_count = len(automated_steps or [])
        automation_score = (automated_count / total_steps * 100) if total_steps > 0 else 0.0

        capability = AutomationCapability(
            system_id=system_id,
            process_name=process_name,
            process_category=process_category,
            current_level=current_level,
            target_level=target_level,
            automation_score=automation_score,
            manual_steps=manual_steps or [],
            automated_steps=automated_steps or [],
            tools_used=tools_used or [],
            improvement_plan=improvement_plan,
        )

        with self._lock:
            self._automation_capabilities[system_id].append(capability)

        return capability

    def get_automation_capabilities(
        self,
        system_id: str,
        category: Optional[str] = None,
    ) -> List[AutomationCapability]:
        """Get automation capabilities for a system."""
        with self._lock:
            capabilities = self._automation_capabilities.get(system_id, [])

            if category:
                capabilities = [c for c in capabilities if c.process_category == category]

            return capabilities

    def get_automation_summary(self) -> Dict[str, Any]:
        """Get overall automation summary."""
        with self._lock:
            total_capabilities = 0
            total_score = 0.0
            below_target = 0

            for capabilities in self._automation_capabilities.values():
                for cap in capabilities:
                    total_capabilities += 1
                    total_score += cap.automation_score
                    if cap.automation_score < self.config.target_automation_score:
                        below_target += 1

            avg_score = total_score / total_capabilities if total_capabilities > 0 else 0.0

            return {
                "total_capabilities": total_capabilities,
                "average_automation_score": avg_score,
                "target_score": self.config.target_automation_score,
                "below_target_count": below_target,
                "meets_target": avg_score >= self.config.target_automation_score,
            }

    # =========================================================================
    # System Lifecycle (Updates and EOL)
    # =========================================================================

    def plan_upgrade(
        self,
        system_id: str,
        upgrade_type: str,
        target_version: str,
        planned_date: str,
        change_description: str = "",
        risk_assessment: str = "",
        rollback_plan: str = "",
    ) -> Optional[SystemUpgrade]:
        """Plan a system upgrade."""
        if system_id not in self._systems:
            return None

        system = self._systems[system_id]

        upgrade = SystemUpgrade(
            system_id=system_id,
            upgrade_type=upgrade_type,
            current_version=system.version,
            target_version=target_version,
            planned_date=planned_date,
            status="planned",
            change_description=change_description,
            risk_assessment=risk_assessment,
            rollback_plan=rollback_plan,
        )

        with self._lock:
            self._upgrades[upgrade.upgrade_id] = upgrade

        self._log_event("upgrade_planned", {
            "upgrade_id": upgrade.upgrade_id,
            "system_id": system_id,
            "target_version": target_version,
        })

        return upgrade

    def execute_upgrade(
        self,
        upgrade_id: str,
        test_results: str = "",
    ) -> Optional[SystemUpgrade]:
        """Execute a planned upgrade."""
        with self._lock:
            if upgrade_id not in self._upgrades:
                return None

            upgrade = self._upgrades[upgrade_id]
            upgrade.status = "completed"
            upgrade.executed_date = datetime.now(timezone.utc).isoformat()
            upgrade.test_results = test_results

            # Update system version
            if upgrade.system_id in self._systems:
                system = self._systems[upgrade.system_id]
                system.version = upgrade.target_version
                system.last_updated = upgrade.executed_date

        self._log_event("upgrade_executed", {
            "upgrade_id": upgrade_id,
            "system_id": upgrade.system_id,
            "new_version": upgrade.target_version,
        })

        return upgrade

    def set_support_end_date(
        self,
        system_id: str,
        support_end_date: str,
        support_level: SupportLevel = SupportLevel.END_OF_LIFE,
    ) -> Optional[ICTSystem]:
        """Set support end date for a system."""
        with self._lock:
            if system_id not in self._systems:
                return None

            system = self._systems[system_id]
            system.support_end_date = support_end_date
            system.support_level = support_level
            system.last_updated = datetime.now(timezone.utc).isoformat()

        return system

    def get_systems_approaching_eol(self) -> List[Tuple[ICTSystem, int]]:
        """Get systems approaching end of life."""
        now = datetime.now(timezone.utc)
        approaching_eol = []

        with self._lock:
            for system in self._systems.values():
                if not system.support_end_date:
                    continue

                eol_date = datetime.fromisoformat(
                    system.support_end_date.replace("Z", "+00:00")
                )
                days_until_eol = (eol_date - now).days

                if days_until_eol <= self.config.warn_days_before_eol:
                    approaching_eol.append((system, days_until_eol))

        # Sort by days until EOL (ascending)
        approaching_eol.sort(key=lambda x: x[1])

        return approaching_eol

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_systems_summary(self) -> Dict[str, Any]:
        """Get comprehensive systems summary."""
        with self._lock:
            all_systems = list(self._systems.values())
            critical_systems = [s for s in all_systems if s.is_critical]
            operational_systems = [s for s in all_systems if s.status == SystemStatus.OPERATIONAL]

            capacity_issues = self.get_systems_with_capacity_issues()
            availability_issues = self.get_systems_below_availability_target()
            eol_systems = self.get_systems_approaching_eol()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "systems": {
                    "total": len(all_systems),
                    "critical": len(critical_systems),
                    "operational": len(operational_systems),
                    "by_type": {
                        t.value: sum(1 for s in all_systems if s.system_type == t)
                        for t in SystemType
                    },
                    "by_criticality": {
                        c.value: sum(1 for s in all_systems if s.criticality == c)
                        for c in SystemCriticality
                    },
                },
                "capacity": {
                    "systems_with_issues": len(capacity_issues),
                    "issues_detail": [
                        {
                            "system_id": sys.system_id,
                            "system_name": sys.name,
                            "metrics": [m.metric_name for m in metrics],
                        }
                        for sys, metrics in capacity_issues
                    ],
                },
                "availability": {
                    "systems_below_target": len(availability_issues),
                    "issues_detail": [
                        {
                            "system_id": sys.system_id,
                            "system_name": sys.name,
                            "availability_pct": metric.availability_pct,
                            "target_pct": metric.target_availability_pct,
                        }
                        for sys, metric in availability_issues
                    ],
                },
                "automation": self.get_automation_summary(),
                "lifecycle": {
                    "systems_approaching_eol": len(eol_systems),
                    "pending_upgrades": sum(
                        1 for u in self._upgrades.values() if u.status == "planned"
                    ),
                },
            }

    def export_systems_inventory(self) -> Dict[str, Any]:
        """Export complete systems inventory."""
        with self._lock:
            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 7",
                "summary": self.get_systems_summary(),
                "systems": [asdict(s) for s in self._systems.values()],
                "capacity_metrics": {
                    sys_id: [asdict(m) for m in metrics[-10:]]
                    for sys_id, metrics in self._capacity_metrics.items()
                },
                "reliability_metrics": {
                    sys_id: [asdict(m) for m in metrics[-12:]]
                    for sys_id, metrics in self._reliability_metrics.items()
                },
                "automation_capabilities": {
                    sys_id: [asdict(c) for c in caps]
                    for sys_id, caps in self._automation_capabilities.items()
                },
                "upgrades": [asdict(u) for u in self._upgrades.values()],
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        if not self.config.log_all_events:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"ict_systems_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_ict_systems_manager(
    config: Optional[ICTSystemsConfig] = None,
) -> DORAICTSystemsManager:
    """
    Create a DORAICTSystemsManager instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAICTSystemsManager instance
    """
    return DORAICTSystemsManager(config=config)
