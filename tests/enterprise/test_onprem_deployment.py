# -*- coding: utf-8 -*-
"""
Comprehensive tests for On-Premises Deployment Service.

Tests DORA Phase 3 Block 3.6: On-prem deployment guide.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from services.enterprise.onprem.deployment import (
    ComponentType,
    DeploymentChecklist,
    DeploymentComponent,
    DeploymentConfig,
    DeploymentRequirement,
    DeploymentStatus,
    DeploymentType,
    OnPremDeployment,
    OnPremDeploymentService,
    create_onprem_deployment,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestDeploymentType:
    """Tests for DeploymentType enum."""

    def test_enum_values(self) -> None:
        """Test all deployment types exist."""
        assert DeploymentType.FULL.value == "full"
        assert DeploymentType.PARTIAL.value == "partial"
        assert DeploymentType.HYBRID.value == "hybrid"
        assert DeploymentType.AIR_GAPPED.value == "air_gapped"

    def test_enum_count(self) -> None:
        """Test correct number of deployment types."""
        assert len(DeploymentType) == 4


class TestDeploymentStatus:
    """Tests for DeploymentStatus enum."""

    def test_enum_values(self) -> None:
        """Test all deployment statuses exist."""
        assert DeploymentStatus.PLANNING.value == "planning"
        assert DeploymentStatus.PREREQUISITES.value == "prerequisites"
        assert DeploymentStatus.IN_PROGRESS.value == "in_progress"
        assert DeploymentStatus.VALIDATION.value == "validation"
        assert DeploymentStatus.COMPLETED.value == "completed"
        assert DeploymentStatus.FAILED.value == "failed"
        assert DeploymentStatus.ROLLED_BACK.value == "rolled_back"

    def test_enum_count(self) -> None:
        """Test correct number of deployment statuses."""
        assert len(DeploymentStatus) == 7


class TestComponentType:
    """Tests for ComponentType enum."""

    def test_enum_values(self) -> None:
        """Test all component types exist."""
        assert ComponentType.API_SERVER.value == "api_server"
        assert ComponentType.DATABASE.value == "database"
        assert ComponentType.CACHE.value == "cache"
        assert ComponentType.MESSAGE_QUEUE.value == "message_queue"
        assert ComponentType.WORKER.value == "worker"
        assert ComponentType.SCHEDULER.value == "scheduler"
        assert ComponentType.MONITORING.value == "monitoring"
        assert ComponentType.LOGGING.value == "logging"
        assert ComponentType.LOAD_BALANCER.value == "load_balancer"
        assert ComponentType.STORAGE.value == "storage"

    def test_enum_count(self) -> None:
        """Test correct number of component types."""
        assert len(ComponentType) == 10


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestDeploymentRequirement:
    """Tests for DeploymentRequirement dataclass."""

    def test_creation(self) -> None:
        """Test requirement creation with all fields."""
        req = DeploymentRequirement(
            requirement_id="req-001",
            name="Minimum CPU",
            description="CPU requirement for deployment",
            category="hardware",
            is_mandatory=True,
            status="pending",
            notes="At least 8 cores",
        )
        assert req.requirement_id == "req-001"
        assert req.name == "Minimum CPU"
        assert req.category == "hardware"
        assert req.is_mandatory is True
        assert req.status == "pending"

    def test_default_values(self) -> None:
        """Test requirement default values."""
        req = DeploymentRequirement(
            requirement_id="req-001",
            name="Test Requirement",
            description="Test",
            category="software",
            is_mandatory=False,
        )
        assert req.status == "pending"
        assert req.notes == ""


class TestDeploymentComponent:
    """Tests for DeploymentComponent dataclass."""

    def test_creation(self) -> None:
        """Test component creation with all fields."""
        component = DeploymentComponent(
            component_id="comp-001",
            component_type=ComponentType.DATABASE,
            name="PostgreSQL",
            version="15.x",
            description="Primary database",
            dependencies=["comp-000"],
            configuration={"port": 5432},
            status="pending",
            health_check_endpoint="/health/db",
        )
        assert component.component_id == "comp-001"
        assert component.component_type == ComponentType.DATABASE
        assert component.name == "PostgreSQL"
        assert component.version == "15.x"
        assert "comp-000" in component.dependencies
        assert component.configuration["port"] == 5432

    def test_default_values(self) -> None:
        """Test component default values."""
        component = DeploymentComponent(
            component_id="comp-001",
            component_type=ComponentType.API_SERVER,
            name="API",
            version="1.0",
            description="API server",
        )
        assert component.dependencies == []
        assert component.configuration == {}
        assert component.status == "pending"
        assert component.deployed_at is None
        assert component.health_check_endpoint == ""
        assert component.is_healthy is False


class TestDeploymentChecklist:
    """Tests for DeploymentChecklist dataclass."""

    def test_creation(self) -> None:
        """Test checklist item creation."""
        item = DeploymentChecklist(
            item_id="check-001",
            phase="pre",
            description="Verify hardware requirements",
        )
        assert item.item_id == "check-001"
        assert item.phase == "pre"
        assert item.description == "Verify hardware requirements"
        assert item.is_completed is False

    def test_complete_method(self) -> None:
        """Test checklist complete method."""
        item = DeploymentChecklist(
            item_id="check-001",
            phase="pre",
            description="Test item",
        )
        assert item.is_completed is False
        assert item.completed_by is None
        assert item.completed_at is None

        item.complete("admin@test.com")

        assert item.is_completed is True
        assert item.completed_by == "admin@test.com"
        assert item.completed_at is not None
        assert isinstance(item.completed_at, datetime)


class TestOnPremDeployment:
    """Tests for OnPremDeployment dataclass."""

    def test_creation(self) -> None:
        """Test deployment creation with all fields."""
        deployment = OnPremDeployment(
            deployment_id="deploy-001",
            client_id="client-001",
            client_name="Test Client",
            deployment_type=DeploymentType.FULL,
            status=DeploymentStatus.PLANNING,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="admin@test.com",
        )
        assert deployment.deployment_id == "deploy-001"
        assert deployment.client_name == "Test Client"
        assert deployment.deployment_type == DeploymentType.FULL
        assert deployment.status == DeploymentStatus.PLANNING

    def test_requirements_met_property(self) -> None:
        """Test requirements_met property."""
        deployment = OnPremDeployment(
            deployment_id="deploy-001",
            client_id="client-001",
            client_name="Test",
            deployment_type=DeploymentType.FULL,
            status=DeploymentStatus.PLANNING,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="admin",
            requirements=[
                DeploymentRequirement(
                    requirement_id="req-001",
                    name="CPU",
                    description="CPU requirement",
                    category="hardware",
                    is_mandatory=True,
                    status="met",
                ),
                DeploymentRequirement(
                    requirement_id="req-002",
                    name="RAM",
                    description="RAM requirement",
                    category="hardware",
                    is_mandatory=True,
                    status="met",
                ),
            ],
        )
        assert deployment.requirements_met is True

    def test_requirements_not_met(self) -> None:
        """Test requirements_met when not all met."""
        deployment = OnPremDeployment(
            deployment_id="deploy-001",
            client_id="client-001",
            client_name="Test",
            deployment_type=DeploymentType.FULL,
            status=DeploymentStatus.PLANNING,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="admin",
            requirements=[
                DeploymentRequirement(
                    requirement_id="req-001",
                    name="CPU",
                    description="CPU",
                    category="hardware",
                    is_mandatory=True,
                    status="met",
                ),
                DeploymentRequirement(
                    requirement_id="req-002",
                    name="RAM",
                    description="RAM",
                    category="hardware",
                    is_mandatory=True,
                    status="not_met",
                ),
            ],
        )
        assert deployment.requirements_met is False

    def test_components_deployed_property(self) -> None:
        """Test components_deployed property."""
        deployment = OnPremDeployment(
            deployment_id="deploy-001",
            client_id="client-001",
            client_name="Test",
            deployment_type=DeploymentType.FULL,
            status=DeploymentStatus.IN_PROGRESS,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="admin",
            components=[
                DeploymentComponent(
                    component_id="comp-001",
                    component_type=ComponentType.DATABASE,
                    name="DB",
                    version="15",
                    description="Database",
                    status="deployed",
                ),
                DeploymentComponent(
                    component_id="comp-002",
                    component_type=ComponentType.CACHE,
                    name="Cache",
                    version="7",
                    description="Cache",
                    status="pending",
                ),
            ],
        )
        assert deployment.components_deployed == 1

    def test_checklist_progress_property(self) -> None:
        """Test checklist_progress property."""
        deployment = OnPremDeployment(
            deployment_id="deploy-001",
            client_id="client-001",
            client_name="Test",
            deployment_type=DeploymentType.FULL,
            status=DeploymentStatus.IN_PROGRESS,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="admin",
            checklist=[
                DeploymentChecklist(
                    item_id="check-001",
                    phase="pre",
                    description="Item 1",
                    is_completed=True,
                ),
                DeploymentChecklist(
                    item_id="check-002",
                    phase="pre",
                    description="Item 2",
                    is_completed=False,
                ),
                DeploymentChecklist(
                    item_id="check-003",
                    phase="during",
                    description="Item 3",
                    is_completed=True,
                ),
                DeploymentChecklist(
                    item_id="check-004",
                    phase="post",
                    description="Item 4",
                    is_completed=False,
                ),
            ],
        )
        assert deployment.checklist_progress == 50.0

    def test_checklist_progress_empty(self) -> None:
        """Test checklist_progress with empty checklist."""
        deployment = OnPremDeployment(
            deployment_id="deploy-001",
            client_id="client-001",
            client_name="Test",
            deployment_type=DeploymentType.FULL,
            status=DeploymentStatus.PLANNING,
            version="1.0.0",
            created_at=datetime.utcnow(),
            created_by="admin",
            checklist=[],
        )
        assert deployment.checklist_progress == 0.0


class TestDeploymentConfig:
    """Tests for DeploymentConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DeploymentConfig()
        assert config.supported_versions == ["1.0.0", "1.1.0", "1.2.0"]
        assert config.default_deployment_type == DeploymentType.FULL
        assert config.require_prerequisites_check is True
        assert config.auto_rollback_on_failure is True
        assert config.health_check_timeout_seconds == 60

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = DeploymentConfig(
            supported_versions=["2.0.0", "2.1.0"],
            default_deployment_type=DeploymentType.HYBRID,
            require_prerequisites_check=False,
            health_check_timeout_seconds=120,
        )
        assert config.supported_versions == ["2.0.0", "2.1.0"]
        assert config.default_deployment_type == DeploymentType.HYBRID
        assert config.require_prerequisites_check is False
        assert config.health_check_timeout_seconds == 120


# =============================================================================
# Service Tests
# =============================================================================


class TestOnPremDeploymentService:
    """Tests for OnPremDeploymentService."""

    @pytest.fixture
    def service(self) -> OnPremDeploymentService:
        """Create service instance for testing."""
        return OnPremDeploymentService()

    @pytest.fixture
    def custom_service(self) -> OnPremDeploymentService:
        """Create service with custom config."""
        config = DeploymentConfig(
            supported_versions=["1.0.0", "1.1.0", "2.0.0"],
            default_deployment_type=DeploymentType.HYBRID,
        )
        return OnPremDeploymentService(config)

    def test_initialization(self, service: OnPremDeploymentService) -> None:
        """Test service initialization."""
        assert service.config is not None
        assert service.config.supported_versions == ["1.0.0", "1.1.0", "1.2.0"]
        assert len(service._deployments) == 0

    def test_create_deployment(self, service: OnPremDeploymentService) -> None:
        """Test deployment creation."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin@test.com",
        )
        assert deployment.deployment_id is not None
        assert deployment.client_id == "client-001"
        assert deployment.client_name == "Test Client"
        assert deployment.version == "1.0.0"
        assert deployment.status == DeploymentStatus.PLANNING
        assert deployment.deployment_type == DeploymentType.FULL
        assert len(deployment.components) > 0
        assert len(deployment.checklist) > 0

    def test_create_deployment_custom_type(self, service: OnPremDeploymentService) -> None:
        """Test deployment creation with custom type."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            deployment_type=DeploymentType.AIR_GAPPED,
            created_by="admin@test.com",
        )
        assert deployment.deployment_type == DeploymentType.AIR_GAPPED

    def test_create_deployment_unsupported_version(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test deployment creation with unsupported version."""
        with pytest.raises(ValueError, match="Version not supported"):
            service.create_deployment(
                client_id="client-001",
                client_name="Test Client",
                version="99.99.99",
                created_by="admin@test.com",
            )

    def test_get_deployment(self, service: OnPremDeploymentService) -> None:
        """Test getting deployment by ID."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        retrieved = service.get_deployment(deployment.deployment_id)
        assert retrieved is not None
        assert retrieved.deployment_id == deployment.deployment_id

    def test_get_deployment_not_found(self, service: OnPremDeploymentService) -> None:
        """Test getting non-existent deployment."""
        result = service.get_deployment("non-existent")
        assert result is None

    def test_list_deployments(self, service: OnPremDeploymentService) -> None:
        """Test listing deployments."""
        service.create_deployment(
            client_id="client-001",
            client_name="Client 1",
            version="1.0.0",
            created_by="admin",
        )
        service.create_deployment(
            client_id="client-002",
            client_name="Client 2",
            version="1.1.0",
            created_by="admin",
        )
        deployments = service.list_deployments()
        assert len(deployments) == 2

    def test_list_deployments_by_client(self, service: OnPremDeploymentService) -> None:
        """Test listing deployments filtered by client."""
        service.create_deployment(
            client_id="client-001",
            client_name="Client 1",
            version="1.0.0",
            created_by="admin",
        )
        service.create_deployment(
            client_id="client-002",
            client_name="Client 2",
            version="1.1.0",
            created_by="admin",
        )
        deployments = service.list_deployments(client_id="client-001")
        assert len(deployments) == 1
        assert deployments[0].client_id == "client-001"

    def test_list_deployments_by_status(self, service: OnPremDeploymentService) -> None:
        """Test listing deployments filtered by status."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Client 1",
            version="1.0.0",
            created_by="admin",
        )
        service.update_status(deployment.deployment_id, DeploymentStatus.IN_PROGRESS)

        service.create_deployment(
            client_id="client-002",
            client_name="Client 2",
            version="1.1.0",
            created_by="admin",
        )

        deployments = service.list_deployments(status=DeploymentStatus.PLANNING)
        assert len(deployments) == 1
        assert deployments[0].status == DeploymentStatus.PLANNING

    def test_update_status_in_progress(self, service: OnPremDeploymentService) -> None:
        """Test updating status to in_progress."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        updated = service.update_status(
            deployment.deployment_id, DeploymentStatus.IN_PROGRESS
        )
        assert updated is not None
        assert updated.status == DeploymentStatus.IN_PROGRESS
        assert updated.started_at is not None

    def test_update_status_completed(self, service: OnPremDeploymentService) -> None:
        """Test updating status to completed."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        updated = service.update_status(
            deployment.deployment_id, DeploymentStatus.COMPLETED
        )
        assert updated is not None
        assert updated.status == DeploymentStatus.COMPLETED
        assert updated.completed_at is not None

    def test_update_status_failed(self, service: OnPremDeploymentService) -> None:
        """Test updating status to failed."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        updated = service.update_status(
            deployment.deployment_id,
            DeploymentStatus.FAILED,
            reason="Database connection failed",
        )
        assert updated is not None
        assert updated.status == DeploymentStatus.FAILED
        assert updated.failed_at is not None
        assert updated.failure_reason == "Database connection failed"

    def test_update_status_not_found(self, service: OnPremDeploymentService) -> None:
        """Test updating status for non-existent deployment."""
        result = service.update_status("non-existent", DeploymentStatus.COMPLETED)
        assert result is None


class TestRequirementManagement:
    """Tests for requirement management functionality."""

    @pytest.fixture
    def service(self) -> OnPremDeploymentService:
        """Create service instance for testing."""
        return OnPremDeploymentService()

    def test_add_requirement(self, service: OnPremDeploymentService) -> None:
        """Test adding a requirement to deployment."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        req = service.add_requirement(
            deployment_id=deployment.deployment_id,
            name="8 CPU Cores",
            description="Minimum 8 CPU cores required",
            category="hardware",
            is_mandatory=True,
        )
        assert req.requirement_id is not None
        assert req.name == "8 CPU Cores"
        assert req.category == "hardware"
        assert req.is_mandatory is True

    def test_add_requirement_not_found(self, service: OnPremDeploymentService) -> None:
        """Test adding requirement to non-existent deployment."""
        with pytest.raises(ValueError, match="Deployment not found"):
            service.add_requirement(
                deployment_id="non-existent",
                name="Test",
                description="Test",
                category="hardware",
            )

    def test_update_requirement_status(self, service: OnPremDeploymentService) -> None:
        """Test updating requirement status."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        req = service.add_requirement(
            deployment_id=deployment.deployment_id,
            name="CPU",
            description="CPU requirement",
            category="hardware",
        )
        result = service.update_requirement_status(
            deployment_id=deployment.deployment_id,
            requirement_id=req.requirement_id,
            status="met",
            notes="Verified 16 cores available",
        )
        assert result is True
        assert req.status == "met"
        assert req.notes == "Verified 16 cores available"

    def test_update_requirement_status_deployment_not_found(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test updating requirement status for non-existent deployment."""
        result = service.update_requirement_status(
            deployment_id="non-existent",
            requirement_id="req-001",
            status="met",
        )
        assert result is False

    def test_update_requirement_status_requirement_not_found(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test updating non-existent requirement."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        result = service.update_requirement_status(
            deployment_id=deployment.deployment_id,
            requirement_id="non-existent",
            status="met",
        )
        assert result is False


class TestComponentManagement:
    """Tests for component management functionality."""

    @pytest.fixture
    def service(self) -> OnPremDeploymentService:
        """Create service instance for testing."""
        return OnPremDeploymentService()

    def test_deploy_component(self, service: OnPremDeploymentService) -> None:
        """Test starting component deployment."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        component = deployment.components[0]
        result = service.deploy_component(
            deployment_id=deployment.deployment_id,
            component_id=component.component_id,
        )
        assert result is not None
        assert result.status == "deploying"

    def test_deploy_component_deployment_not_found(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test deploying component for non-existent deployment."""
        result = service.deploy_component(
            deployment_id="non-existent",
            component_id="comp-001",
        )
        assert result is None

    def test_complete_component_deployment(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test completing component deployment."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        component = deployment.components[0]
        service.deploy_component(deployment.deployment_id, component.component_id)

        result = service.complete_component_deployment(
            deployment_id=deployment.deployment_id,
            component_id=component.component_id,
            is_healthy=True,
        )
        assert result is not None
        assert result.status == "deployed"
        assert result.is_healthy is True
        assert result.deployed_at is not None

    def test_complete_component_deployment_unhealthy(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test completing component deployment with unhealthy status."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        component = deployment.components[0]
        result = service.complete_component_deployment(
            deployment_id=deployment.deployment_id,
            component_id=component.component_id,
            is_healthy=False,
        )
        assert result is not None
        assert result.status == "failed"
        assert result.is_healthy is False

    def test_get_component_status(self, service: OnPremDeploymentService) -> None:
        """Test getting component status summary."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        component = deployment.components[0]
        service.complete_component_deployment(
            deployment.deployment_id, component.component_id, is_healthy=True
        )

        status = service.get_component_status(deployment.deployment_id)
        assert status["total"] == len(deployment.components)
        assert status["deployed"] >= 1
        assert status["healthy"] >= 1

    def test_get_component_status_not_found(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test getting component status for non-existent deployment."""
        status = service.get_component_status("non-existent")
        assert status == {}


class TestChecklistManagement:
    """Tests for checklist management functionality."""

    @pytest.fixture
    def service(self) -> OnPremDeploymentService:
        """Create service instance for testing."""
        return OnPremDeploymentService()

    def test_complete_checklist_item(self, service: OnPremDeploymentService) -> None:
        """Test completing a checklist item."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        item = deployment.checklist[0]
        result = service.complete_checklist_item(
            deployment_id=deployment.deployment_id,
            item_id=item.item_id,
            user="admin@test.com",
            notes="Verified hardware meets requirements",
        )
        assert result is True
        assert item.is_completed is True
        assert item.completed_by == "admin@test.com"
        assert item.notes == "Verified hardware meets requirements"

    def test_complete_checklist_item_deployment_not_found(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test completing checklist item for non-existent deployment."""
        result = service.complete_checklist_item(
            deployment_id="non-existent",
            item_id="check-001",
            user="admin",
        )
        assert result is False

    def test_complete_checklist_item_not_found(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test completing non-existent checklist item."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        result = service.complete_checklist_item(
            deployment_id=deployment.deployment_id,
            item_id="non-existent",
            user="admin",
        )
        assert result is False

    def test_get_checklist_status(self, service: OnPremDeploymentService) -> None:
        """Test getting checklist status."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        # Complete first pre item
        pre_items = [c for c in deployment.checklist if c.phase == "pre"]
        if pre_items:
            service.complete_checklist_item(
                deployment.deployment_id, pre_items[0].item_id, "admin"
            )

        status = service.get_checklist_status(deployment.deployment_id)
        assert "total_items" in status
        assert "completed_items" in status
        assert "overall_progress" in status
        assert "by_phase" in status
        assert "pre" in status["by_phase"]
        assert "during" in status["by_phase"]
        assert "post" in status["by_phase"]

    def test_get_checklist_status_not_found(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test getting checklist status for non-existent deployment."""
        status = service.get_checklist_status("non-existent")
        assert status == {}


class TestDeploymentReporting:
    """Tests for deployment reporting functionality."""

    @pytest.fixture
    def service(self) -> OnPremDeploymentService:
        """Create service instance for testing."""
        return OnPremDeploymentService()

    def test_get_deployment_summary(self, service: OnPremDeploymentService) -> None:
        """Test getting deployment summary."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        summary = service.get_deployment_summary(deployment.deployment_id)
        assert summary["deployment_id"] == deployment.deployment_id
        assert summary["client_id"] == "client-001"
        assert summary["client_name"] == "Test Client"
        assert summary["deployment_type"] == "full"
        assert summary["status"] == "planning"
        assert summary["version"] == "1.0.0"
        assert "requirements_met" in summary
        assert "components" in summary
        assert "checklist" in summary
        assert "created_at" in summary

    def test_get_deployment_summary_not_found(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test getting summary for non-existent deployment."""
        summary = service.get_deployment_summary("non-existent")
        assert summary == {}


class TestDefaultComponents:
    """Tests for default components initialization."""

    @pytest.fixture
    def service(self) -> OnPremDeploymentService:
        """Create service instance for testing."""
        return OnPremDeploymentService()

    def test_default_components_created(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test that default components are created."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        assert len(deployment.components) > 0
        component_types = [c.component_type for c in deployment.components]
        assert ComponentType.DATABASE in component_types
        assert ComponentType.CACHE in component_types
        assert ComponentType.MESSAGE_QUEUE in component_types
        assert ComponentType.API_SERVER in component_types

    def test_default_checklist_created(
        self, service: OnPremDeploymentService
    ) -> None:
        """Test that default checklist is created."""
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Test Client",
            version="1.0.0",
            created_by="admin",
        )
        assert len(deployment.checklist) > 0
        phases = {c.phase for c in deployment.checklist}
        assert "pre" in phases
        assert "during" in phases
        assert "post" in phases


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_onprem_deployment_default(self) -> None:
        """Test creating service with default config."""
        service = create_onprem_deployment()
        assert service is not None
        assert service.config.supported_versions == ["1.0.0", "1.1.0", "1.2.0"]

    def test_create_onprem_deployment_custom_versions(self) -> None:
        """Test creating service with custom versions."""
        service = create_onprem_deployment(supported_versions=["2.0.0", "2.1.0"])
        assert service.config.supported_versions == ["2.0.0", "2.1.0"]

    def test_create_onprem_deployment_with_kwargs(self) -> None:
        """Test creating service with additional kwargs."""
        service = create_onprem_deployment(
            supported_versions=["1.0.0"],
            auto_rollback_on_failure=False,
            health_check_timeout_seconds=120,
        )
        assert service.config.auto_rollback_on_failure is False
        assert service.config.health_check_timeout_seconds == 120


# =============================================================================
# Integration Tests
# =============================================================================


class TestDeploymentWorkflow:
    """Integration tests for full deployment workflow."""

    @pytest.fixture
    def service(self) -> OnPremDeploymentService:
        """Create service instance for testing."""
        return OnPremDeploymentService()

    def test_full_deployment_workflow(self, service: OnPremDeploymentService) -> None:
        """Test complete deployment workflow."""
        # 1. Create deployment
        deployment = service.create_deployment(
            client_id="client-001",
            client_name="Enterprise Client",
            version="1.0.0",
            deployment_type=DeploymentType.FULL,
            created_by="admin@enterprise.com",
        )
        assert deployment.status == DeploymentStatus.PLANNING

        # 2. Add requirements
        cpu_req = service.add_requirement(
            deployment_id=deployment.deployment_id,
            name="CPU Cores",
            description="Minimum 16 CPU cores",
            category="hardware",
            is_mandatory=True,
        )
        ram_req = service.add_requirement(
            deployment_id=deployment.deployment_id,
            name="RAM",
            description="Minimum 64GB RAM",
            category="hardware",
            is_mandatory=True,
        )

        # 3. Verify requirements
        service.update_requirement_status(
            deployment.deployment_id, cpu_req.requirement_id, "met"
        )
        service.update_requirement_status(
            deployment.deployment_id, ram_req.requirement_id, "met"
        )
        assert deployment.requirements_met is True

        # 4. Complete pre-deployment checklist
        pre_items = [c for c in deployment.checklist if c.phase == "pre"]
        for item in pre_items:
            service.complete_checklist_item(
                deployment.deployment_id, item.item_id, "admin@enterprise.com"
            )

        # 5. Start deployment
        service.update_status(deployment.deployment_id, DeploymentStatus.IN_PROGRESS)
        assert deployment.status == DeploymentStatus.IN_PROGRESS
        assert deployment.started_at is not None

        # 6. Deploy components
        for component in deployment.components:
            service.deploy_component(deployment.deployment_id, component.component_id)
            service.complete_component_deployment(
                deployment.deployment_id, component.component_id, is_healthy=True
            )

        # 7. Complete during checklist
        during_items = [c for c in deployment.checklist if c.phase == "during"]
        for item in during_items:
            service.complete_checklist_item(
                deployment.deployment_id, item.item_id, "admin@enterprise.com"
            )

        # 8. Complete post-deployment checklist
        post_items = [c for c in deployment.checklist if c.phase == "post"]
        for item in post_items:
            service.complete_checklist_item(
                deployment.deployment_id, item.item_id, "admin@enterprise.com"
            )

        # 9. Mark deployment as completed
        service.update_status(deployment.deployment_id, DeploymentStatus.COMPLETED)
        assert deployment.status == DeploymentStatus.COMPLETED
        assert deployment.completed_at is not None

        # 10. Verify final state
        summary = service.get_deployment_summary(deployment.deployment_id)
        assert summary["status"] == "completed"
        assert summary["checklist"]["overall_progress"] == 100.0
        assert all(c.is_healthy for c in deployment.components)
