"""
Tests for GDPR Article 20 - Right to Data Portability implementation.

References:
    - GDPR Regulation (EU) 2016/679 Article 20
    - EDPB Guidelines on Data Portability (WP242)
"""

import io
import json
import zipfile
import pytest
from datetime import datetime, timedelta, timezone

from services.gdpr.data_export import (
    GDPRExportService,
    PortableDataPackage,
    ExportRequest,
    ExportStatus,
    ExportError,
    ExportNotFoundError,
    InMemoryExportRequestStorage,
    InMemoryUserRepository,
    InMemoryStrategiesRepository,
    InMemoryBacktestsRepository,
    InMemoryExecutionsRepository,
    InMemorySettingsRepository,
    MockUser,
    MockStrategy,
    MockBacktest,
    MockExecution,
    MockSettings,
)


class TestExportStatus:
    """Tests for ExportStatus enum."""

    def test_all_statuses_defined(self):
        """Verify all required statuses exist."""
        assert ExportStatus.PENDING.value == "pending"
        assert ExportStatus.IN_PROGRESS.value == "in_progress"
        assert ExportStatus.COMPLETED.value == "completed"
        assert ExportStatus.FAILED.value == "failed"
        assert ExportStatus.EXPIRED.value == "expired"


class TestPortableDataPackage:
    """Tests for PortableDataPackage dataclass."""

    def test_creation(self):
        """Test PortableDataPackage creation."""
        package = PortableDataPackage(
            user_id="user_001",
            exported_at=datetime.now(timezone.utc),
        )
        assert package.user_id == "user_001"
        assert package.format_version == "1.0"
        assert package.strategies == []
        assert package.backtests == []

    def test_to_dict(self):
        """Test serialization to dictionary."""
        package = PortableDataPackage(
            user_id="user_001",
            exported_at=datetime.now(timezone.utc),
            account={"email": "user@example.com"},
            strategies=[{"name": "Strategy 1"}],
        )
        data = package.to_dict()

        assert data["user_id"] == "user_001"
        assert data["format_version"] == "1.0"
        assert data["account"]["email"] == "user@example.com"
        assert len(data["strategies"]) == 1


class TestExportRequest:
    """Tests for ExportRequest dataclass."""

    def test_creation(self):
        """Test ExportRequest creation."""
        request = ExportRequest(
            request_id="exp_123",
            user_id="user_001",
            requested_at=datetime.now(timezone.utc),
            status=ExportStatus.PENDING,
        )
        assert request.request_id == "exp_123"
        assert request.status == ExportStatus.PENDING

    def test_to_dict(self):
        """Test serialization to dictionary."""
        request = ExportRequest(
            request_id="exp_123",
            user_id="user_001",
            requested_at=datetime.now(timezone.utc),
            status=ExportStatus.COMPLETED,
            file_size_bytes=1024,
        )
        data = request.to_dict()

        assert data["request_id"] == "exp_123"
        assert data["status"] == "completed"
        assert data["file_size_bytes"] == 1024


class TestInMemoryRepositories:
    """Tests for in-memory repository implementations."""

    def test_user_repository(self):
        """Test user repository operations."""
        repo = InMemoryUserRepository()
        user = MockUser(
            user_id="user_001",
            email="user@example.com",
            name="Test User",
            created_at=datetime.now(timezone.utc),
        )
        repo.add(user)

        retrieved = repo.get("user_001")
        assert retrieved is not None
        assert retrieved.email == "user@example.com"

    def test_strategies_repository(self):
        """Test strategies repository operations."""
        repo = InMemoryStrategiesRepository()
        strategy = MockStrategy(
            strategy_id="strat_001",
            name="Test Strategy",
            description="A test strategy",
            parameters={"window": 20},
            code="def run(): pass",
            created_at=datetime.now(timezone.utc),
        )
        repo.add("user_001", strategy)

        strategies = repo.get_by_user("user_001")
        assert len(strategies) == 1
        assert strategies[0].name == "Test Strategy"

    def test_backtests_repository(self):
        """Test backtests repository operations."""
        repo = InMemoryBacktestsRepository()
        backtest = MockBacktest(
            backtest_id="bt_001",
            strategy_name="Test Strategy",
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
            results={"total_return": 0.15},
            ran_at=datetime.now(timezone.utc),
        )
        repo.add("user_001", backtest)

        backtests = repo.get_by_user("user_001")
        assert len(backtests) == 1
        assert backtests[0].results["total_return"] == 0.15

    def test_executions_repository(self):
        """Test executions repository operations."""
        repo = InMemoryExecutionsRepository()
        execution = MockExecution(
            order_id="ord_001",
            symbol="AAPL",
            side="buy",
            quantity=100.0,
            price=150.0,
            broker="alpaca",
            executed_at=datetime.now(timezone.utc),
            status="filled",
        )
        repo.add("user_001", execution)

        executions = repo.get_by_user("user_001")
        assert len(executions) == 1
        assert executions[0].symbol == "AAPL"

    def test_settings_repository(self):
        """Test settings repository operations."""
        repo = InMemorySettingsRepository()
        settings = MockSettings(
            user_id="user_001",
            default_broker="alpaca",
            risk_parameters={"max_position_size": 0.1},
            notification_preferences={"email": True},
        )
        repo.set(settings)

        retrieved = repo.get("user_001")
        assert retrieved is not None
        assert retrieved.default_broker == "alpaca"


class TestGDPRExportService:
    """Tests for GDPRExportService."""

    @pytest.fixture
    def mock_repos(self):
        """Create mock repositories with test data."""
        repos = {
            "export_requests": InMemoryExportRequestStorage(),
            "users": InMemoryUserRepository(),
            "strategies": InMemoryStrategiesRepository(),
            "backtests": InMemoryBacktestsRepository(),
            "executions": InMemoryExecutionsRepository(),
            "settings": InMemorySettingsRepository(),
        }

        # Add test data
        repos["users"].add(MockUser(
            user_id="user_001",
            email="user@example.com",
            name="Test User",
            created_at=datetime.now(timezone.utc),
        ))

        repos["strategies"].add("user_001", MockStrategy(
            strategy_id="strat_001",
            name="Test Strategy",
            description="A test strategy",
            parameters={"window": 20},
            code="def run(): pass",
            created_at=datetime.now(timezone.utc),
        ))

        repos["backtests"].add("user_001", MockBacktest(
            backtest_id="bt_001",
            strategy_name="Test Strategy",
            start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
            results={"total_return": 0.15},
            ran_at=datetime.now(timezone.utc),
        ))

        repos["executions"].add("user_001", MockExecution(
            order_id="ord_001",
            symbol="AAPL",
            side="buy",
            quantity=100.0,
            price=150.0,
            broker="alpaca",
            executed_at=datetime.now(timezone.utc),
            status="filled",
        ))

        repos["settings"].set(MockSettings(
            user_id="user_001",
            default_broker="alpaca",
            risk_parameters={"max_position_size": 0.1},
            notification_preferences={"email": True},
        ))

        return repos

    @pytest.fixture
    def service(self, mock_repos):
        """Create service with mock repositories."""
        return GDPRExportService(mock_repos)

    def test_export_returns_zip(self, service):
        """Test that export returns a valid ZIP file."""
        data = service.export_user_data("user_001")

        # Check ZIP magic bytes
        assert data[:2] == b"PK"

    def test_export_contains_required_files(self, service):
        """Test that export contains all required files."""
        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            assert "metadata.json" in names
            assert "account.json" in names
            assert "strategies.json" in names
            assert "backtests.json" in names
            assert "execution_history.json" in names
            assert "settings.json" in names
            assert "README.txt" in names

    def test_export_json_valid(self, service):
        """Test that exported JSON files are valid."""
        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if name.endswith(".json"):
                    content = zf.read(name)
                    # Should not raise
                    json.loads(content.decode("utf-8"))

    def test_export_account_data(self, service):
        """Test that account data is exported correctly."""
        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            account = json.loads(zf.read("account.json"))
            assert account["email"] == "user@example.com"
            assert account["name"] == "Test User"

    def test_export_excludes_password(self, service, mock_repos):
        """Test that password hash is not exported."""
        # Add a user with password-like field
        user = mock_repos["users"].get("user_001")

        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            account = json.loads(zf.read("account.json"))
            assert "password" not in account
            assert "password_hash" not in account

    def test_export_strategies(self, service):
        """Test that strategies are exported correctly."""
        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            strategies = json.loads(zf.read("strategies.json"))
            assert len(strategies) == 1
            assert strategies[0]["name"] == "Test Strategy"
            assert strategies[0]["code"] == "def run(): pass"

    def test_export_backtests(self, service):
        """Test that backtests are exported correctly."""
        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            backtests = json.loads(zf.read("backtests.json"))
            assert len(backtests) == 1
            assert backtests[0]["strategy_name"] == "Test Strategy"
            assert backtests[0]["results"]["total_return"] == 0.15

    def test_export_executions(self, service):
        """Test that execution history is exported correctly."""
        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            executions = json.loads(zf.read("execution_history.json"))
            assert len(executions) == 1
            assert executions[0]["symbol"] == "AAPL"
            assert executions[0]["side"] == "buy"

    def test_export_settings(self, service):
        """Test that settings are exported correctly."""
        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            settings = json.loads(zf.read("settings.json"))
            assert settings["default_broker"] == "alpaca"

    def test_export_metadata(self, service):
        """Test that metadata is included."""
        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            metadata = json.loads(zf.read("metadata.json"))
            assert metadata["user_id"] == "user_001"
            assert metadata["format_version"] == "1.0"
            assert "Article 20" in metadata["gdpr_article"]

    def test_export_readme(self, service):
        """Test that README is included."""
        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            readme = zf.read("README.txt").decode("utf-8")
            assert "GDPR Article 20" in readme
            assert "Data Portability" in readme

    def test_create_request(self, service):
        """Test creating an export request."""
        request = service.create_request("user_001")

        assert request is not None
        assert request.user_id == "user_001"
        assert request.status == ExportStatus.PENDING

    def test_execute_export_request(self, service):
        """Test executing an export request."""
        request = service.create_request("user_001")
        result = service.execute_export_request(request)

        assert result.status == ExportStatus.COMPLETED
        assert result.file_size_bytes > 0
        assert result.expires_at is not None

    def test_get_export_request(self, service):
        """Test getting an export request."""
        created = service.create_request("user_001")
        retrieved = service.get_export_request(created.request_id)

        assert retrieved.request_id == created.request_id

    def test_get_export_request_not_found(self, service):
        """Test getting a nonexistent request."""
        with pytest.raises(ExportNotFoundError):
            service.get_export_request("nonexistent")

    def test_user_export_history(self, service):
        """Test getting export history for a user."""
        service.create_request("user_001")
        service.create_request("user_001")

        history = service.get_user_export_history("user_001")
        assert len(history) == 2

    def test_generate_export_report(self, service):
        """Test generating an export report."""
        data = service.export_user_data("user_001")
        report = service.generate_export_report("user_001", data)

        assert report["user_id"] == "user_001"
        assert report["total_size_bytes"] > 0
        assert report["file_count"] > 0
        assert report["format"] == "ZIP containing JSON files"
        assert "Article 20" in report["compliance"]["gdpr_article"]

    def test_export_empty_user(self, mock_repos):
        """Test exporting data for a user with no data."""
        service = GDPRExportService(mock_repos)
        data = service.export_user_data("nonexistent_user")

        # Should still return a valid ZIP
        assert data[:2] == b"PK"

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            strategies = json.loads(zf.read("strategies.json"))
            assert strategies == []


class TestGDPRExportServiceCompliance:
    """Compliance-focused tests for GDPRExportService."""

    @pytest.fixture
    def service(self):
        repos = {
            "export_requests": InMemoryExportRequestStorage(),
            "users": InMemoryUserRepository(),
        }
        return GDPRExportService(repos)

    def test_format_is_machine_readable(self, service):
        """GDPR Art. 20 - data in machine-readable format."""
        data = service.export_user_data("user_001")

        # JSON is machine-readable
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            metadata = json.loads(zf.read("metadata.json"))
            # If we can parse it, it's machine-readable
            assert metadata is not None

    def test_format_is_commonly_used(self, service):
        """GDPR Art. 20 - data in commonly used format."""
        data = service.export_user_data("user_001")

        # ZIP and JSON are commonly used formats
        assert data[:2] == b"PK"  # ZIP
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert any(f.endswith(".json") for f in zf.namelist())

    def test_export_is_structured(self, service):
        """GDPR Art. 20 - data in structured format."""
        data = service.export_user_data("user_001")

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            metadata = json.loads(zf.read("metadata.json"))
            # Structured means organized with clear schema
            assert "contents" in metadata
            assert "format_version" in metadata

    def test_deadline_within_30_days(self, service):
        """GDPR Art. 12(3) - respond within one month."""
        assert service.EXPORT_DEADLINE_DAYS <= 30

    def test_download_expiry_reasonable(self, service):
        """Test download expiry is reasonable."""
        assert service.DOWNLOAD_EXPIRY_DAYS >= 1
        assert service.DOWNLOAD_EXPIRY_DAYS <= 30
