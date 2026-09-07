"""
GDPR Article 20 - Right to Data Portability implementation.

Provides comprehensive data export functionality in machine-readable format,
allowing users to receive their personal data for transfer to another service.

References:
    - GDPR Regulation (EU) 2016/679 Article 20
    - EDPB Guidelines on Data Portability (WP242)
    - ICO Guide to the Right to Data Portability

Format: JSON (machine-readable, commonly used, structured)

Data Included:
    - Account information (excluding password hashes)
    - Trading strategies (user-created)
    - Backtest results
    - Execution history
    - Platform settings
    - Acknowledgment records

Data Excluded (per GDPR Art. 20 scope):
    - System-generated analytics
    - Internal audit logs
    - Derived/inferred data

Example:
    >>> service = GDPRExportService(repositories)
    >>> zip_bytes = service.export_user_data("user_001")
    >>> with open("user_data.zip", "wb") as f:
    ...     f.write(zip_bytes)
"""

from __future__ import annotations

import io
import json
import logging
import secrets
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger("gdpr.export")


class ExportStatus(Enum):
    """Status of an export request."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class ExportError(Exception):
    """Base exception for export errors."""

    pass


class ExportNotFoundError(ExportError):
    """Raised when an export request is not found."""

    pass


class ExportInProgressError(ExportError):
    """Raised when an export is already in progress."""

    pass


@dataclass
class PortableDataPackage:
    """
    Container for user's portable data.

    Attributes:
        user_id: Owner of the data
        exported_at: When the export was generated
        format_version: Version of the export format
        account: User account information
        strategies: Trading strategies
        backtests: Backtest results
        execution_history: Order execution logs
        settings: Platform settings
        acknowledgments: Disclaimer acknowledgments
    """

    user_id: str
    exported_at: datetime
    format_version: str = "1.0"
    account: Optional[Dict[str, Any]] = None
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    backtests: List[Dict[str, Any]] = field(default_factory=list)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    settings: Optional[Dict[str, Any]] = None
    acknowledgments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "user_id": self.user_id,
            "exported_at": self.exported_at.isoformat(),
            "format_version": self.format_version,
            "account": self.account,
            "strategies": self.strategies,
            "backtests": self.backtests,
            "execution_history": self.execution_history,
            "settings": self.settings,
            "acknowledgments": self.acknowledgments,
        }


@dataclass
class ExportRequest:
    """
    Represents a GDPR data export request.

    Attributes:
        request_id: Unique identifier
        user_id: User requesting export
        requested_at: When request was made
        status: Current status
        completed_at: When export completed
        download_url: URL to download export (if applicable)
        expires_at: When the export download expires
        file_size_bytes: Size of the export file
        error_message: Error details if failed
    """

    request_id: str
    user_id: str
    requested_at: datetime
    status: ExportStatus
    completed_at: Optional[datetime] = None
    download_url: Optional[str] = None
    expires_at: Optional[datetime] = None
    file_size_bytes: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "requested_at": self.requested_at.isoformat(),
            "status": self.status.value,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "download_url": self.download_url,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "file_size_bytes": self.file_size_bytes,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class UserRepositoryProtocol(Protocol):
    """Protocol for user repository."""

    def get(self, user_id: str) -> Optional[Any]:
        """Get user by ID."""
        ...


class StrategiesRepositoryProtocol(Protocol):
    """Protocol for strategies repository."""

    def get_by_user(self, user_id: str) -> List[Any]:
        """Get all strategies for a user."""
        ...


class ExportRequestStorageProtocol(Protocol):
    """Protocol for export request storage."""

    def save(self, request: ExportRequest) -> None:
        """Save an export request."""
        ...

    def update(self, request: ExportRequest) -> None:
        """Update an existing request."""
        ...

    def get(self, request_id: str) -> Optional[ExportRequest]:
        """Get a request by ID."""
        ...

    def get_by_user(self, user_id: str) -> List[ExportRequest]:
        """Get all requests for a user."""
        ...


class InMemoryExportRequestStorage:
    """In-memory implementation for testing."""

    def __init__(self):
        self._requests: Dict[str, ExportRequest] = {}

    def save(self, request: ExportRequest) -> None:
        self._requests[request.request_id] = request

    def update(self, request: ExportRequest) -> None:
        self._requests[request.request_id] = request

    def get(self, request_id: str) -> Optional[ExportRequest]:
        return self._requests.get(request_id)

    def get_by_user(self, user_id: str) -> List[ExportRequest]:
        return [r for r in self._requests.values() if r.user_id == user_id]


# Mock data classes for repositories
@dataclass
class MockUser:
    """Mock user for testing."""

    user_id: str
    email: str
    name: str
    created_at: datetime


@dataclass
class MockStrategy:
    """Mock strategy for testing."""

    strategy_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    code: str
    created_at: datetime


@dataclass
class MockBacktest:
    """Mock backtest for testing."""

    backtest_id: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    results: Dict[str, Any]
    ran_at: datetime


@dataclass
class MockExecution:
    """Mock execution for testing."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    broker: str
    executed_at: datetime
    status: str


@dataclass
class MockSettings:
    """Mock settings for testing."""

    user_id: str
    default_broker: str
    risk_parameters: Dict[str, Any]
    notification_preferences: Dict[str, Any]


class InMemoryUserRepository:
    """In-memory user repository for testing."""

    def __init__(self):
        self._users: Dict[str, MockUser] = {}

    def add(self, user: MockUser) -> None:
        self._users[user.user_id] = user

    def get(self, user_id: str) -> Optional[MockUser]:
        return self._users.get(user_id)


class InMemoryStrategiesRepository:
    """In-memory strategies repository for testing."""

    def __init__(self):
        self._strategies: Dict[str, List[MockStrategy]] = {}

    def add(self, user_id: str, strategy: MockStrategy) -> None:
        if user_id not in self._strategies:
            self._strategies[user_id] = []
        self._strategies[user_id].append(strategy)

    def get_by_user(self, user_id: str) -> List[MockStrategy]:
        return self._strategies.get(user_id, [])


class InMemoryBacktestsRepository:
    """In-memory backtests repository for testing."""

    def __init__(self):
        self._backtests: Dict[str, List[MockBacktest]] = {}

    def add(self, user_id: str, backtest: MockBacktest) -> None:
        if user_id not in self._backtests:
            self._backtests[user_id] = []
        self._backtests[user_id].append(backtest)

    def get_by_user(self, user_id: str) -> List[MockBacktest]:
        return self._backtests.get(user_id, [])


class InMemoryExecutionsRepository:
    """In-memory executions repository for testing."""

    def __init__(self):
        self._executions: Dict[str, List[MockExecution]] = {}

    def add(self, user_id: str, execution: MockExecution) -> None:
        if user_id not in self._executions:
            self._executions[user_id] = []
        self._executions[user_id].append(execution)

    def get_by_user(self, user_id: str) -> List[MockExecution]:
        return self._executions.get(user_id, [])


class InMemorySettingsRepository:
    """In-memory settings repository for testing."""

    def __init__(self):
        self._settings: Dict[str, MockSettings] = {}

    def set(self, settings: MockSettings) -> None:
        self._settings[settings.user_id] = settings

    def get(self, user_id: str) -> Optional[MockSettings]:
        return self._settings.get(user_id)


class GDPRExportService:
    """
    Export user data in portable format per GDPR Article 20.

    Features:
        - Complete data export in machine-readable JSON format
        - ZIP archive with organized file structure
        - Excludes sensitive internal data (password hashes, internal IDs)
        - README with format documentation
        - Metadata file for import validation

    Export Contents:
        - metadata.json: Export information and schema version
        - account.json: User account details (excluding passwords)
        - strategies.json: User's trading strategies
        - backtests.json: Backtest results
        - execution_history.json: Order execution logs
        - settings.json: Platform settings
        - acknowledgments.json: Disclaimer acknowledgments
        - README.txt: Documentation for the export

    Example:
        >>> repos = {
        ...     "users": users_repo,
        ...     "strategies": strategies_repo,
        ...     "backtests": backtests_repo,
        ...     "executions": executions_repo,
        ...     "settings": settings_repo,
        ...     "export_requests": export_storage,
        ... }
        >>> service = GDPRExportService(repos)
        >>> zip_data = service.export_user_data("user_001")
    """

    # GDPR Art. 12(3) - respond within one month
    EXPORT_DEADLINE_DAYS = 30

    # How long the export download is available
    DOWNLOAD_EXPIRY_DAYS = 7

    def __init__(
        self,
        repositories: Dict[str, Any],
        on_export_complete: Optional[Callable[[ExportRequest], None]] = None,
    ):
        """
        Initialize the GDPR export service.

        Args:
            repositories: Dictionary mapping category names to repository instances
            on_export_complete: Optional callback when export completes
        """
        self._repos = repositories
        self._on_complete = on_export_complete
        self._request_storage: ExportRequestStorageProtocol = repositories.get(
            "export_requests", InMemoryExportRequestStorage()
        )

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return f"exp_{secrets.token_hex(8)}"

    def create_request(
        self,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExportRequest:
        """
        Create an export request.

        Args:
            user_id: User requesting export
            metadata: Additional context

        Returns:
            Created ExportRequest
        """
        now = datetime.now(timezone.utc)

        request = ExportRequest(
            request_id=self._generate_request_id(),
            user_id=user_id,
            requested_at=now,
            status=ExportStatus.PENDING,
            metadata=metadata or {},
        )

        self._request_storage.save(request)

        logger.info(
            f"EXPORT_REQUEST_CREATED | " f"request_id={request.request_id} | " f"user_id={user_id}"
        )

        return request

    def export_user_data(self, user_id: str) -> bytes:
        """
        Export all user data as ZIP containing JSON files.

        Args:
            user_id: User to export data for

        Returns:
            ZIP file as bytes

        Raises:
            ExportError: If export fails
        """
        logger.info(f"EXPORT_STARTED | user_id={user_id}")

        try:
            package = PortableDataPackage(
                user_id=user_id,
                exported_at=datetime.now(timezone.utc),
            )

            # Collect data from all repositories
            package.account = self._export_account(user_id)
            package.strategies = self._export_strategies(user_id)
            package.backtests = self._export_backtests(user_id)
            package.execution_history = self._export_execution_history(user_id)
            package.settings = self._export_settings(user_id)
            package.acknowledgments = self._export_acknowledgments(user_id)

            zip_bytes = self._create_zip(package)

            logger.info(
                f"EXPORT_COMPLETED | " f"user_id={user_id} | " f"size_bytes={len(zip_bytes)}"
            )

            return zip_bytes

        except Exception as e:
            logger.error(f"EXPORT_FAILED | " f"user_id={user_id} | " f"error={str(e)}")
            raise ExportError(f"Failed to export data for user {user_id}: {str(e)}") from e

    def _export_account(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Export user account information."""
        users_repo = self._repos.get("users")
        if users_repo is None:
            return None

        user = users_repo.get(user_id)
        if user is None:
            return None

        # Extract safe fields, exclude password hash
        return {
            "user_id": getattr(user, "user_id", user_id),
            "email": getattr(user, "email", None),
            "name": getattr(user, "name", None),
            "created_at": (
                getattr(user, "created_at", datetime.now(timezone.utc)).isoformat()
                if hasattr(user, "created_at")
                else None
            ),
            # Explicitly excluded: password_hash, internal IDs
        }

    def _export_strategies(self, user_id: str) -> List[Dict[str, Any]]:
        """Export user's trading strategies."""
        strategies_repo = self._repos.get("strategies")
        if strategies_repo is None:
            return []

        strategies = strategies_repo.get_by_user(user_id)
        return [
            {
                "strategy_id": getattr(s, "strategy_id", None),
                "name": getattr(s, "name", None),
                "description": getattr(s, "description", None),
                "parameters": getattr(s, "parameters", {}),
                "code": getattr(s, "code", None),  # User's own code
                "created_at": (
                    getattr(s, "created_at", datetime.now(timezone.utc)).isoformat()
                    if hasattr(s, "created_at")
                    else None
                ),
            }
            for s in strategies
        ]

    def _export_backtests(self, user_id: str) -> List[Dict[str, Any]]:
        """Export user's backtest results."""
        backtests_repo = self._repos.get("backtests")
        if backtests_repo is None:
            return []

        backtests = backtests_repo.get_by_user(user_id)
        return [
            {
                "backtest_id": getattr(b, "backtest_id", None),
                "strategy_name": getattr(b, "strategy_name", None),
                "start_date": (
                    getattr(b, "start_date", datetime.now(timezone.utc)).isoformat()
                    if hasattr(b, "start_date")
                    else None
                ),
                "end_date": (
                    getattr(b, "end_date", datetime.now(timezone.utc)).isoformat()
                    if hasattr(b, "end_date")
                    else None
                ),
                "results": getattr(b, "results", {}),
                "ran_at": (
                    getattr(b, "ran_at", datetime.now(timezone.utc)).isoformat()
                    if hasattr(b, "ran_at")
                    else None
                ),
            }
            for b in backtests
        ]

    def _export_execution_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Export user's order execution history."""
        executions_repo = self._repos.get("executions")
        if executions_repo is None:
            return []

        executions = executions_repo.get_by_user(user_id)
        return [
            {
                "order_id": getattr(e, "order_id", None),
                "symbol": getattr(e, "symbol", None),
                "side": getattr(e, "side", None),
                "quantity": getattr(e, "quantity", None),
                "price": getattr(e, "price", None),
                "broker": getattr(e, "broker", None),
                "executed_at": (
                    getattr(e, "executed_at", datetime.now(timezone.utc)).isoformat()
                    if hasattr(e, "executed_at")
                    else None
                ),
                "status": getattr(e, "status", None),
            }
            for e in executions
        ]

    def _export_settings(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Export user's platform settings."""
        settings_repo = self._repos.get("settings")
        if settings_repo is None:
            return None

        settings = settings_repo.get(user_id)
        if settings is None:
            return None

        return {
            "default_broker": getattr(settings, "default_broker", None),
            "risk_parameters": getattr(settings, "risk_parameters", {}),
            "notification_preferences": getattr(settings, "notification_preferences", {}),
        }

    def _export_acknowledgments(self, user_id: str) -> List[Dict[str, Any]]:
        """Export user's disclaimer acknowledgments."""
        ack_repo = self._repos.get("acknowledgments")
        if ack_repo is None:
            return []

        # Handle both get_all_for_user and get_by_user methods
        if hasattr(ack_repo, "get_all_for_user"):
            acks = ack_repo.get_all_for_user(user_id)
        elif hasattr(ack_repo, "get_by_user"):
            acks = ack_repo.get_by_user(user_id)
        else:
            return []

        return [
            {
                "acknowledgment_id": getattr(a, "acknowledgment_id", None),
                "disclaimer_type": (
                    getattr(a, "disclaimer_type", None).value
                    if hasattr(getattr(a, "disclaimer_type", None), "value")
                    else None
                ),
                "acknowledged_at": (
                    getattr(a, "acknowledged_at", datetime.now(timezone.utc)).isoformat()
                    if hasattr(a, "acknowledged_at")
                    else None
                ),
            }
            for a in acks
        ]

    def _create_zip(self, package: PortableDataPackage) -> bytes:
        """
        Create ZIP file with JSON exports.

        Args:
            package: Data package to export

        Returns:
            ZIP file as bytes
        """
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Metadata
            metadata = {
                "user_id": package.user_id,
                "exported_at": package.exported_at.isoformat(),
                "format_version": package.format_version,
                "gdpr_article": "Article 20 - Right to Data Portability",
                "contents": [
                    "account.json",
                    "strategies.json",
                    "backtests.json",
                    "execution_history.json",
                    "settings.json",
                    "acknowledgments.json",
                ],
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))

            # Data files
            if package.account:
                zf.writestr("account.json", json.dumps(package.account, indent=2))

            zf.writestr("strategies.json", json.dumps(package.strategies, indent=2))
            zf.writestr("backtests.json", json.dumps(package.backtests, indent=2))
            zf.writestr("execution_history.json", json.dumps(package.execution_history, indent=2))

            if package.settings:
                zf.writestr("settings.json", json.dumps(package.settings, indent=2))

            zf.writestr("acknowledgments.json", json.dumps(package.acknowledgments, indent=2))

            # README
            zf.writestr("README.txt", self._generate_readme())

        buffer.seek(0)
        return buffer.read()

    def _generate_readme(self) -> str:
        """Generate README for the export archive."""
        return """DATA EXPORT - GDPR Article 20 (Right to Data Portability)
==========================================================

This archive contains your personal data in machine-readable format (JSON).

FILES INCLUDED
--------------
- metadata.json     : Export information and schema version
- account.json      : Your account details (email, name, creation date)
- strategies.json   : Your trading strategies (including code)
- backtests.json    : Your backtest results and history
- execution_history.json : Your order execution logs
- settings.json     : Your platform settings and preferences
- acknowledgments.json : Your disclaimer acknowledgments

FORMAT DETAILS
--------------
Format: JSON (JavaScript Object Notation)
Encoding: UTF-8
Schema Version: 1.0

DATA EXCLUDED
-------------
The following data is NOT included per GDPR Article 20 scope:
- Password hashes (security)
- Internal system IDs
- System-generated analytics
- Audit logs (legally required retention)

HOW TO USE THIS DATA
--------------------
This data is provided in a structured, machine-readable format that
can be imported into other services or analyzed using standard tools.

You can open JSON files with:
- Any text editor
- Online JSON viewers (e.g., jsonviewer.stack.hu)
- Programming languages (Python, JavaScript, etc.)
- Spreadsheet software (some support JSON import)

LEGAL NOTICE
------------
This export was generated pursuant to your rights under:
- GDPR Article 20 (Right to Data Portability)
- GDPR Article 15 (Right of Access)

If you have questions about your data, please contact our
Data Protection Officer at: [DPO Email]

Export Generated: [See metadata.json for timestamp]
Platform: AI-Powered Quantitative Research Platform
"""

    def execute_export_request(self, request: ExportRequest) -> ExportRequest:
        """
        Execute an export request.

        Args:
            request: Export request to execute

        Returns:
            Updated ExportRequest
        """
        request.status = ExportStatus.IN_PROGRESS
        self._request_storage.update(request)

        try:
            zip_bytes = self.export_user_data(request.user_id)

            request.status = ExportStatus.COMPLETED
            request.completed_at = datetime.now(timezone.utc)
            request.file_size_bytes = len(zip_bytes)
            request.expires_at = request.completed_at + timedelta(days=self.DOWNLOAD_EXPIRY_DAYS)

            # In a real implementation, you would store the file and set download_url
            # request.download_url = f"/api/gdpr/exports/{request.request_id}/download"

        except Exception as e:
            request.status = ExportStatus.FAILED
            request.error_message = str(e)
            request.completed_at = datetime.now(timezone.utc)

        self._request_storage.update(request)

        if self._on_complete:
            self._on_complete(request)

        return request

    def get_export_request(self, request_id: str) -> ExportRequest:
        """
        Get an export request by ID.

        Args:
            request_id: Request ID

        Returns:
            ExportRequest

        Raises:
            ExportNotFoundError: If request not found
        """
        request = self._request_storage.get(request_id)
        if request is None:
            raise ExportNotFoundError(f"Export request {request_id} not found")
        return request

    def get_user_export_history(self, user_id: str) -> List[ExportRequest]:
        """
        Get export history for a user.

        Args:
            user_id: User to query

        Returns:
            List of export requests
        """
        return self._request_storage.get_by_user(user_id)

    def generate_export_report(self, user_id: str, zip_bytes: bytes) -> Dict[str, Any]:
        """
        Generate a report about the export.

        Args:
            user_id: User whose data was exported
            zip_bytes: The export file

        Returns:
            Report dictionary
        """
        # Parse ZIP to count files and sizes
        file_details = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            for info in zf.infolist():
                file_details.append(
                    {
                        "filename": info.filename,
                        "size_bytes": info.file_size,
                        "compressed_size_bytes": info.compress_size,
                    }
                )

        return {
            "user_id": user_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_size_bytes": len(zip_bytes),
            "file_count": len(file_details),
            "files": file_details,
            "format": "ZIP containing JSON files",
            "encoding": "UTF-8",
            "compliance": {
                "gdpr_article": "Article 20 - Right to Data Portability",
                "format_standard": "JSON (machine-readable)",
            },
        }
