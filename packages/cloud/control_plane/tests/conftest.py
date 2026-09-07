# -*- coding: utf-8 -*-
"""
Pytest fixtures for Cloud Control Plane tests.

Provides:
- Async test client with httpx
- In-memory SQLite database for testing
- Authentication fixtures (user/agent tokens)
- Sample data fixtures (organizations, workspaces, users, etc.)
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import AsyncGenerator, Dict, Generator, List, Optional
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

# Set test mode before importing app modules
os.environ["CCEA_TEST_MODE"] = "true"
os.environ["CCEA_JWT_SECRET"] = "test-secret-key-for-testing-only"

from ..app import create_app
from ..database import (
    get_session,
    get_session_factory,
    set_engine,
    set_session_factory,
)
from ..dependencies import create_access_token, create_agent_token
from ..models import (
    Agent,
    AgentEnrollmentToken,
    Alert,
    AlertSeverity,
    ApprovalRecord,
    Artifact,
    AuditAction,
    Base,
    Build,
    ChangeClass,
    Command,
    CommandStatus,
    ConfigBlob,
    DataRetentionPolicy,
    Deployment,
    DeploymentState,
    Organization,
    Permission,
    Role,
    Run,
    RunState,
    Strategy,
    StrategyVersion,
    TelemetryEvent,
    TelemetryLevel,
    TrustState,
    User,
    Workspace,
    compute_token_hash,
    generate_enrollment_token,
)


# =============================================================================
# Helper Functions
# =============================================================================


async def get_or_create_permission(
    session: AsyncSession,
    name: str,
    description: str = "",
    resource: str = "",
    action: str = "",
) -> Permission:
    """Get or create a permission by name to avoid duplicates."""
    from sqlalchemy import select

    result = await session.execute(select(Permission).where(Permission.name == name))
    existing = result.scalar_one_or_none()
    if existing:
        return existing

    perm = Permission(
        name=name,
        description=description or f"Permission {name}",
        resource=resource or name.split(":")[0] if ":" in name else name,
        action=action or name.split(":")[1] if ":" in name else "default",
    )
    session.add(perm)
    await session.commit()
    await session.refresh(perm)
    return perm


# =============================================================================
# Event Loop Fixture
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create an in-memory SQLite engine for testing."""
    # Use SQLite with aiosqlite for async testing
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        echo=False,
        connect_args={"check_same_thread": False},
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to the test engine."""
    factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    return factory


@pytest_asyncio.fixture(scope="function")
async def db_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Get a database session for testing."""
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def setup_db(
    engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[None, None]:
    """Set up the database for app testing."""
    # Inject test engine and session factory
    set_engine(engine)
    set_session_factory(session_factory)
    yield
    # Reset after test
    set_engine(None)
    set_session_factory(None)


# =============================================================================
# Test Client Fixture
# =============================================================================


@pytest_asyncio.fixture(scope="function")
async def client(setup_db: None) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    app = create_app(debug=True)

    # Skip lifespan context for testing (we set up DB manually)
    app.router.lifespan_context = None

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


# =============================================================================
# Sample IDs
# =============================================================================


@pytest.fixture
def org_id() -> UUID:
    """Sample organization ID."""
    return uuid4()


@pytest.fixture
def workspace_id() -> UUID:
    """Sample workspace ID."""
    return uuid4()


@pytest.fixture
def user_id() -> UUID:
    """Sample user ID."""
    return uuid4()


@pytest.fixture
def agent_id() -> UUID:
    """Sample agent ID."""
    return uuid4()


@pytest.fixture
def strategy_id() -> UUID:
    """Sample strategy ID."""
    return uuid4()


@pytest.fixture
def deployment_id() -> UUID:
    """Sample deployment ID."""
    return uuid4()


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def sample_organization(
    db_session: AsyncSession,
    org_id: UUID,
) -> Organization:
    """Create a sample organization."""
    org = Organization(
        id=org_id,
        name="test-org",
        slug="test-org",
    )
    db_session.add(org)
    await db_session.commit()
    await db_session.refresh(org)
    return org


@pytest_asyncio.fixture
async def sample_workspace(
    db_session: AsyncSession,
    sample_organization: Organization,
    workspace_id: UUID,
) -> Workspace:
    """Create a sample workspace."""
    workspace = Workspace(
        id=workspace_id,
        organization_id=sample_organization.id,
        name="test-workspace",
        slug="test-workspace",
        settings={"env": "test"},
    )
    db_session.add(workspace)
    await db_session.commit()
    await db_session.refresh(workspace)
    return workspace


@pytest_asyncio.fixture
async def sample_permission(db_session: AsyncSession) -> Permission:
    """Create a sample permission."""
    permission = Permission(
        name="config:create",
        description="Create configuration",
        resource="config",
        action="create",
    )
    db_session.add(permission)
    await db_session.commit()
    await db_session.refresh(permission)
    return permission


@pytest_asyncio.fixture
async def sample_role(
    db_session: AsyncSession,
    sample_organization: Organization,
    sample_permission: Permission,
) -> Role:
    """Create a sample role with permission."""
    role = Role(
        organization_id=sample_organization.id,
        name="admin",
        description="Administrator role",
    )
    role.permissions.append(sample_permission)
    db_session.add(role)
    await db_session.commit()
    await db_session.refresh(role)
    return role


@pytest_asyncio.fixture
async def sample_user(
    db_session: AsyncSession,
    sample_organization: Organization,
    sample_workspace: Workspace,
    sample_role: Role,
    user_id: UUID,
) -> User:
    """Create a sample user with role."""
    user = User(
        id=user_id,
        email="testuser@example.com",
        password_hash="hashed_password_here",
        display_name="Test User",
        organization_id=sample_organization.id,
        default_workspace_id=sample_workspace.id,
    )
    user.roles.append(sample_role)
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest_asyncio.fixture
async def sample_agent(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    agent_id: UUID,
) -> Agent:
    """Create a sample agent."""
    agent = Agent(
        id=agent_id,
        workspace_id=sample_workspace.id,
        name="test-agent",
        agent_version="1.0.0",
        public_key="a" * 64,  # Min 64 chars for public key
        trust_state=TrustState.ENROLLED,
        capabilities=["trading", "backtest"],
        os_info={"hostname": "test-host"},
    )
    db_session.add(agent)
    await db_session.commit()
    await db_session.refresh(agent)
    return agent


@pytest_asyncio.fixture
async def sample_strategy(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_user: User,
    strategy_id: UUID,
) -> Strategy:
    """Create a sample strategy."""
    strategy = Strategy(
        id=strategy_id,
        workspace_id=sample_workspace.id,
        name="test-strategy",
        description="A test trading strategy",
        # Note: Strategy model doesn't have created_by_user_id field
    )
    db_session.add(strategy)
    await db_session.commit()
    await db_session.refresh(strategy)
    return strategy


@pytest_asyncio.fixture
async def sample_strategy_version(
    db_session: AsyncSession,
    sample_strategy: Strategy,
    sample_user: User,
) -> StrategyVersion:
    """Create a sample strategy version."""
    version = StrategyVersion(
        workspace_id=sample_strategy.workspace_id,
        strategy_id=sample_strategy.id,
        version="1.0.0",
        git_sha="abc123def456",
        # Note: parameters and created_by_user_id don't exist in model
    )
    db_session.add(version)
    await db_session.commit()
    await db_session.refresh(version)
    return version


@pytest_asyncio.fixture
async def sample_build(
    db_session: AsyncSession,
    sample_strategy_version: StrategyVersion,
    sample_user: User,
) -> Build:
    """Create a sample build."""
    build = Build(
        workspace_id=sample_strategy_version.workspace_id,
        strategy_version_id=sample_strategy_version.id,
        build_number=1,
        status="completed",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        # Note: Build model doesn't have started_by_user_id or completed_at
    )
    db_session.add(build)
    await db_session.commit()
    await db_session.refresh(build)
    return build


@pytest_asyncio.fixture
async def sample_artifact(
    db_session: AsyncSession,
    sample_build: Build,
) -> Artifact:
    """Create a sample artifact."""
    artifact = Artifact(
        workspace_id=sample_build.workspace_id,
        build_id=sample_build.id,
        name="test.whl",
        format="wheel",
        digest="sha256:artifact123",
        size_bytes=1024000,
    )
    db_session.add(artifact)
    await db_session.commit()
    await db_session.refresh(artifact)
    return artifact


@pytest_asyncio.fixture
async def sample_deployment(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_agent: Agent,
    sample_artifact: Artifact,
    deployment_id: UUID,
) -> Deployment:
    """Create a sample deployment."""
    deployment = Deployment(
        id=deployment_id,
        workspace_id=sample_workspace.id,
        agent_id=sample_agent.id,
        artifact_id=sample_artifact.id,
        state=DeploymentState.DEPLOYED.value,
    )
    db_session.add(deployment)
    await db_session.commit()
    await db_session.refresh(deployment)
    return deployment


@pytest_asyncio.fixture
async def sample_run(
    db_session: AsyncSession,
    sample_deployment: Deployment,
) -> Run:
    """Create a sample run."""
    run = Run(
        workspace_id=sample_deployment.workspace_id,
        deployment_id=sample_deployment.id,
        state=RunState.RUNNING,
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(run)
    await db_session.commit()
    await db_session.refresh(run)
    return run


@pytest_asyncio.fixture
async def sample_command(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_agent: Agent,
    sample_user: User,
) -> Command:
    """Create a sample command."""
    command = Command(
        workspace_id=sample_workspace.id,
        agent_id=sample_agent.id,
        command_type="STOP",
        payload={"reason": "manual"},
        change_class=ChangeClass.OPERATIONAL,
        status=CommandStatus.PENDING,
        created_by_user_id=sample_user.id,
    )
    db_session.add(command)
    await db_session.commit()
    await db_session.refresh(command)
    return command


@pytest_asyncio.fixture
async def sample_config_blob(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_user: User,
) -> ConfigBlob:
    """Create a sample config blob."""
    import hashlib
    import json

    content = {"setting1": "value1", "setting2": 42}
    content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
    digest = f"sha256:{hashlib.sha256(content_str.encode()).hexdigest()}"

    blob = ConfigBlob(
        workspace_id=sample_workspace.id,
        digest=digest,
        content=content,
        size_bytes=len(content_str.encode()),
        config_type="strategy",
        schema_version="1.0.0",
        created_by_user_id=sample_user.id,
    )
    db_session.add(blob)
    await db_session.commit()
    await db_session.refresh(blob)
    return blob


@pytest_asyncio.fixture
async def sample_telemetry_event(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_agent: Agent,
) -> TelemetryEvent:
    """Create a sample telemetry event."""
    event = TelemetryEvent(
        workspace_id=sample_workspace.id,
        agent_id=sample_agent.id,
        level=TelemetryLevel.INFO,
        category="performance",
        event_type="backtest_completed",
        payload={"duration_sec": 123.45},
    )
    db_session.add(event)
    await db_session.commit()
    await db_session.refresh(event)
    return event


@pytest_asyncio.fixture
async def sample_alert(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_agent: Agent,
) -> Alert:
    """Create a sample alert."""
    alert = Alert(
        workspace_id=sample_workspace.id,
        agent_id=sample_agent.id,
        severity=AlertSeverity.WARNING,
        category="risk",
        title="Drawdown Warning",
        message="Drawdown exceeded 10%",
    )
    db_session.add(alert)
    await db_session.commit()
    await db_session.refresh(alert)
    return alert


@pytest_asyncio.fixture
async def sample_enrollment_token(
    db_session: AsyncSession,
    sample_workspace: Workspace,
    sample_user: User,
) -> tuple[str, AgentEnrollmentToken]:
    """Create a sample enrollment token (returns raw token and record)."""
    raw_token = generate_enrollment_token()
    token_hash = compute_token_hash(raw_token)

    token_record = AgentEnrollmentToken(
        workspace_id=sample_workspace.id,
        name="test-token",
        token_hash=token_hash,
        capabilities=["trading"],
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        created_by_user_id=sample_user.id,
    )
    db_session.add(token_record)
    await db_session.commit()
    await db_session.refresh(token_record)
    return raw_token, token_record


# =============================================================================
# Authentication Fixtures
# =============================================================================


@pytest.fixture
def user_token(
    sample_user: User,  # Ensure user exists in DB
    workspace_id: UUID,
    org_id: UUID,
) -> str:
    """Create a user JWT token for testing (no permissions - for permission denial tests)."""
    return create_access_token(
        user_id=sample_user.id,
        email=sample_user.email,
        workspace_id=workspace_id,
        org_id=org_id,
        permissions=[],  # No permissions - tests that need perms create their own tokens
    )


@pytest.fixture
def superuser_token(
    sample_user: User,  # Ensure user exists in DB
    workspace_id: UUID,
    org_id: UUID,
) -> str:
    """Create a superuser JWT token for testing."""
    return create_access_token(
        user_id=sample_user.id,
        email=sample_user.email,
        workspace_id=workspace_id,
        org_id=org_id,
        permissions=["superuser"],
        is_superuser=True,
    )


@pytest.fixture
def superuser_id(sample_user: User) -> UUID:
    """Get superuser ID (same as sample_user for testing)."""
    return sample_user.id


@pytest.fixture
def agent_token(
    agent_id: UUID,
    workspace_id: UUID,
    org_id: UUID,
) -> str:
    """Create an agent JWT token for testing."""
    return create_agent_token(
        agent_id=agent_id,
        workspace_id=workspace_id,
        org_id=org_id,
        capabilities=["trading", "backtest"],
    )


@pytest.fixture
def auth_headers(user_token: str) -> Dict[str, str]:
    """Get authorization headers for user requests."""
    return {"Authorization": f"Bearer {user_token}"}


@pytest.fixture
def superuser_headers(superuser_token: str) -> Dict[str, str]:
    """Get authorization headers for superuser requests."""
    return {"Authorization": f"Bearer {superuser_token}"}


@pytest.fixture
def agent_headers(agent_token: str) -> Dict[str, str]:
    """Get authorization headers for agent requests."""
    return {"Authorization": f"Bearer {agent_token}"}


# =============================================================================
# Helper Functions
# =============================================================================


def make_auth_headers(token: str) -> Dict[str, str]:
    """Create authorization headers from a token."""
    return {"Authorization": f"Bearer {token}"}


async def create_test_user(
    session: AsyncSession,
    org: Organization,
    workspace: Optional[Workspace] = None,
    email: str = "test@example.com",
    is_superuser: bool = False,
) -> tuple[User, str]:
    """Create a test user and return with token."""
    user = User(
        email=email,
        password_hash="hashed",
        organization_id=org.id,
        default_workspace_id=workspace.id if workspace else None,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    permissions = ["superuser"] if is_superuser else []
    token = create_access_token(
        user_id=user.id,
        email=user.email,
        workspace_id=workspace.id if workspace else None,
        org_id=org.id,
        permissions=permissions,
    )
    return user, token


async def create_test_agent(
    session: AsyncSession,
    workspace: Workspace,
    org: Organization,
    name: str = "test-agent",
) -> tuple[Agent, str]:
    """Create a test agent and return with token."""
    agent = Agent(
        workspace_id=workspace.id,
        name=name,
        public_key="a" * 64,  # Min 64 chars for public key
        trust_state=TrustState.ENROLLED,
    )
    session.add(agent)
    await session.commit()
    await session.refresh(agent)

    token = create_agent_token(
        agent_id=agent.id,
        workspace_id=workspace.id,
        org_id=org.id,
        capabilities=["trading"],
    )
    return agent, token
