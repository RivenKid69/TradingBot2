# -*- coding: utf-8 -*-
"""
On-Premises Deployment Service.

DORA Phase 3 Block 3.6: On-prem deployment guide

Provides on-premises deployment management:
- Deployment planning and execution
- Component deployment tracking
- Validation and verification
- Rollback procedures

DORA References:
    - Art. 30(2)(b): Data location provisions
    - Art. 28(8): Exit strategies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


# =============================================================================
# Enums
# =============================================================================


class DeploymentType(Enum):
    """On-premises deployment types."""

    FULL = "full"  # Complete platform deployment
    PARTIAL = "partial"  # Selected components only
    HYBRID = "hybrid"  # Some on-prem, some cloud
    AIR_GAPPED = "air_gapped"  # No internet connectivity


class DeploymentStatus(Enum):
    """Deployment status."""

    PLANNING = "planning"
    PREREQUISITES = "prerequisites"
    IN_PROGRESS = "in_progress"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ComponentType(Enum):
    """Deployable component types."""

    API_SERVER = "api_server"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    WORKER = "worker"
    SCHEDULER = "scheduler"
    MONITORING = "monitoring"
    LOGGING = "logging"
    LOAD_BALANCER = "load_balancer"
    STORAGE = "storage"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class DeploymentRequirement:
    """Deployment requirement."""

    requirement_id: str
    name: str
    description: str
    category: str  # hardware, software, network, security
    is_mandatory: bool
    status: str = "pending"  # pending, met, not_met
    notes: str = ""


@dataclass
class DeploymentComponent:
    """Deployable component."""

    component_id: str
    component_type: ComponentType
    name: str
    version: str
    description: str
    dependencies: list[str] = field(default_factory=list)  # Other component IDs
    configuration: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, deploying, deployed, failed
    deployed_at: datetime | None = None
    health_check_endpoint: str = ""
    is_healthy: bool = False


@dataclass
class DeploymentChecklist:
    """Deployment checklist item."""

    item_id: str
    phase: str  # pre, during, post
    description: str
    is_completed: bool = False
    completed_by: str | None = None
    completed_at: datetime | None = None
    notes: str = ""

    def complete(self, user: str) -> None:
        """Mark item as completed."""
        self.is_completed = True
        self.completed_by = user
        self.completed_at = datetime.utcnow()


@dataclass
class OnPremDeployment:
    """On-premises deployment record."""

    deployment_id: str
    client_id: str
    client_name: str
    deployment_type: DeploymentType
    status: DeploymentStatus
    version: str
    created_at: datetime
    created_by: str
    requirements: list[DeploymentRequirement] = field(default_factory=list)
    components: list[DeploymentComponent] = field(default_factory=list)
    checklist: list[DeploymentChecklist] = field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None
    failure_reason: str = ""
    notes: str = ""

    @property
    def requirements_met(self) -> bool:
        """Check if all mandatory requirements are met."""
        mandatory = [r for r in self.requirements if r.is_mandatory]
        return all(r.status == "met" for r in mandatory)

    @property
    def components_deployed(self) -> int:
        """Count deployed components."""
        return sum(1 for c in self.components if c.status == "deployed")

    @property
    def checklist_progress(self) -> float:
        """Calculate checklist completion percentage."""
        if not self.checklist:
            return 0.0
        completed = sum(1 for c in self.checklist if c.is_completed)
        return (completed / len(self.checklist)) * 100


@dataclass
class DeploymentConfig:
    """Deployment service configuration."""

    supported_versions: list[str] = field(default_factory=lambda: ["1.0.0", "1.1.0", "1.2.0"])
    default_deployment_type: DeploymentType = DeploymentType.FULL
    require_prerequisites_check: bool = True
    auto_rollback_on_failure: bool = True
    health_check_timeout_seconds: int = 60


# =============================================================================
# Main Service Class
# =============================================================================


class OnPremDeploymentService:
    """
    On-Premises Deployment Service.

    Manages on-premises deployments per DORA data residency requirements.
    """

    def __init__(self, config: DeploymentConfig | None = None) -> None:
        """Initialize on-prem deployment service."""
        self.config = config or DeploymentConfig()
        self._deployments: dict[str, OnPremDeployment] = {}

    def _get_default_components(self) -> list[DeploymentComponent]:
        """Get default components for deployment."""
        return [
            DeploymentComponent(
                component_id=str(uuid4()),
                component_type=ComponentType.DATABASE,
                name="PostgreSQL Database",
                version="15.x",
                description="Primary relational database",
                health_check_endpoint="/health/db",
            ),
            DeploymentComponent(
                component_id=str(uuid4()),
                component_type=ComponentType.CACHE,
                name="Redis Cache",
                version="7.x",
                description="In-memory cache and session store",
                health_check_endpoint="/health/cache",
            ),
            DeploymentComponent(
                component_id=str(uuid4()),
                component_type=ComponentType.MESSAGE_QUEUE,
                name="RabbitMQ",
                version="3.12.x",
                description="Message broker for async processing",
                health_check_endpoint="/health/mq",
            ),
            DeploymentComponent(
                component_id=str(uuid4()),
                component_type=ComponentType.API_SERVER,
                name="API Server",
                version="latest",
                description="Main API server",
                health_check_endpoint="/health",
            ),
            DeploymentComponent(
                component_id=str(uuid4()),
                component_type=ComponentType.WORKER,
                name="Background Worker",
                version="latest",
                description="Async task processor",
                health_check_endpoint="/health/worker",
            ),
            DeploymentComponent(
                component_id=str(uuid4()),
                component_type=ComponentType.MONITORING,
                name="Prometheus + Grafana",
                version="latest",
                description="Metrics and monitoring",
                health_check_endpoint="/health/monitoring",
            ),
            DeploymentComponent(
                component_id=str(uuid4()),
                component_type=ComponentType.LOGGING,
                name="ELK Stack",
                version="8.x",
                description="Centralized logging",
                health_check_endpoint="/health/logging",
            ),
        ]

    def _get_default_checklist(self) -> list[DeploymentChecklist]:
        """Get default deployment checklist."""
        return [
            # Pre-deployment
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="pre",
                description="Verify hardware meets minimum requirements",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="pre",
                description="Verify network connectivity and firewall rules",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="pre",
                description="Install required operating system packages",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="pre",
                description="Configure SSL/TLS certificates",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="pre",
                description="Set up database server and create databases",
            ),
            # During deployment
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="during",
                description="Deploy database schemas and migrations",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="during",
                description="Deploy cache server",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="during",
                description="Deploy message queue",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="during",
                description="Deploy API server",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="during",
                description="Deploy background workers",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="during",
                description="Configure load balancer",
            ),
            # Post-deployment
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="post",
                description="Run health checks on all components",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="post",
                description="Verify data encryption at rest",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="post",
                description="Test backup and restore procedures",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="post",
                description="Configure monitoring and alerting",
            ),
            DeploymentChecklist(
                item_id=str(uuid4()),
                phase="post",
                description="Document deployment configuration",
            ),
        ]

    # =========================================================================
    # Deployment Management
    # =========================================================================

    def create_deployment(
        self,
        client_id: str,
        client_name: str,
        version: str,
        deployment_type: DeploymentType | None = None,
        created_by: str = "system",
    ) -> OnPremDeployment:
        """Create a new on-premises deployment."""
        if version not in self.config.supported_versions:
            raise ValueError(f"Version not supported: {version}")

        deployment = OnPremDeployment(
            deployment_id=str(uuid4()),
            client_id=client_id,
            client_name=client_name,
            deployment_type=deployment_type or self.config.default_deployment_type,
            status=DeploymentStatus.PLANNING,
            version=version,
            created_at=datetime.utcnow(),
            created_by=created_by,
            components=self._get_default_components(),
            checklist=self._get_default_checklist(),
        )
        self._deployments[deployment.deployment_id] = deployment
        return deployment

    def get_deployment(self, deployment_id: str) -> OnPremDeployment | None:
        """Get deployment by ID."""
        return self._deployments.get(deployment_id)

    def list_deployments(
        self,
        client_id: str | None = None,
        status: DeploymentStatus | None = None,
    ) -> list[OnPremDeployment]:
        """List deployments with optional filters."""
        deployments = list(self._deployments.values())

        if client_id:
            deployments = [d for d in deployments if d.client_id == client_id]
        if status:
            deployments = [d for d in deployments if d.status == status]

        return deployments

    def update_status(
        self,
        deployment_id: str,
        status: DeploymentStatus,
        reason: str = "",
    ) -> OnPremDeployment | None:
        """Update deployment status."""
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            return None

        deployment.status = status

        if status == DeploymentStatus.IN_PROGRESS:
            deployment.started_at = datetime.utcnow()
        elif status == DeploymentStatus.COMPLETED:
            deployment.completed_at = datetime.utcnow()
        elif status == DeploymentStatus.FAILED:
            deployment.failed_at = datetime.utcnow()
            deployment.failure_reason = reason

        return deployment

    # =========================================================================
    # Requirements Management
    # =========================================================================

    def add_requirement(
        self,
        deployment_id: str,
        name: str,
        description: str,
        category: str,
        is_mandatory: bool = True,
    ) -> DeploymentRequirement:
        """Add requirement to deployment."""
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment not found: {deployment_id}")

        requirement = DeploymentRequirement(
            requirement_id=str(uuid4()),
            name=name,
            description=description,
            category=category,
            is_mandatory=is_mandatory,
        )
        deployment.requirements.append(requirement)
        return requirement

    def update_requirement_status(
        self,
        deployment_id: str,
        requirement_id: str,
        status: str,
        notes: str = "",
    ) -> bool:
        """Update requirement status."""
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            return False

        for req in deployment.requirements:
            if req.requirement_id == requirement_id:
                req.status = status
                req.notes = notes
                return True

        return False

    # =========================================================================
    # Component Management
    # =========================================================================

    def deploy_component(
        self,
        deployment_id: str,
        component_id: str,
    ) -> DeploymentComponent | None:
        """Mark component as deploying."""
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            return None

        for component in deployment.components:
            if component.component_id == component_id:
                component.status = "deploying"
                return component

        return None

    def complete_component_deployment(
        self,
        deployment_id: str,
        component_id: str,
        is_healthy: bool = True,
    ) -> DeploymentComponent | None:
        """Mark component as deployed."""
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            return None

        for component in deployment.components:
            if component.component_id == component_id:
                component.status = "deployed" if is_healthy else "failed"
                component.deployed_at = datetime.utcnow()
                component.is_healthy = is_healthy
                return component

        return None

    def get_component_status(self, deployment_id: str) -> dict[str, Any]:
        """Get component deployment status summary."""
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            return {}

        return {
            "total": len(deployment.components),
            "pending": sum(1 for c in deployment.components if c.status == "pending"),
            "deploying": sum(1 for c in deployment.components if c.status == "deploying"),
            "deployed": sum(1 for c in deployment.components if c.status == "deployed"),
            "failed": sum(1 for c in deployment.components if c.status == "failed"),
            "healthy": sum(1 for c in deployment.components if c.is_healthy),
        }

    # =========================================================================
    # Checklist Management
    # =========================================================================

    def complete_checklist_item(
        self,
        deployment_id: str,
        item_id: str,
        user: str,
        notes: str = "",
    ) -> bool:
        """Complete a checklist item."""
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            return False

        for item in deployment.checklist:
            if item.item_id == item_id:
                item.complete(user)
                item.notes = notes
                return True

        return False

    def get_checklist_status(self, deployment_id: str) -> dict[str, Any]:
        """Get checklist completion status."""
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            return {}

        by_phase = {}
        for phase in ["pre", "during", "post"]:
            phase_items = [c for c in deployment.checklist if c.phase == phase]
            completed = sum(1 for c in phase_items if c.is_completed)
            by_phase[phase] = {
                "total": len(phase_items),
                "completed": completed,
                "progress_percent": (completed / len(phase_items) * 100) if phase_items else 0,
            }

        return {
            "total_items": len(deployment.checklist),
            "completed_items": sum(1 for c in deployment.checklist if c.is_completed),
            "overall_progress": deployment.checklist_progress,
            "by_phase": by_phase,
        }

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_deployment_summary(self, deployment_id: str) -> dict[str, Any]:
        """Get deployment summary."""
        deployment = self._deployments.get(deployment_id)
        if not deployment:
            return {}

        return {
            "deployment_id": deployment.deployment_id,
            "client_id": deployment.client_id,
            "client_name": deployment.client_name,
            "deployment_type": deployment.deployment_type.value,
            "status": deployment.status.value,
            "version": deployment.version,
            "requirements_met": deployment.requirements_met,
            "components": self.get_component_status(deployment_id),
            "checklist": self.get_checklist_status(deployment_id),
            "created_at": deployment.created_at.isoformat(),
            "started_at": deployment.started_at.isoformat() if deployment.started_at else None,
            "completed_at": (
                deployment.completed_at.isoformat() if deployment.completed_at else None
            ),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_onprem_deployment(
    supported_versions: list[str] | None = None,
    **kwargs: Any,
) -> OnPremDeploymentService:
    """Create on-prem deployment service instance."""
    config = DeploymentConfig(
        supported_versions=supported_versions or ["1.0.0", "1.1.0", "1.2.0"],
        **kwargs,
    )
    return OnPremDeploymentService(config)
