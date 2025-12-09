"""
GDPR Article 17 - Right to Erasure implementation.

Provides comprehensive data deletion functionality ensuring compliance with
GDPR requirements while respecting legal retention obligations.

References:
    - GDPR Regulation (EU) 2016/679 Article 17
    - EDPB Guidelines on the Right to Erasure
    - ICO Right to Erasure Guide
    - MiFID II Art. 25 (audit trail retention requirements)

Retention Exceptions (GDPR Art. 17(3)):
    - Legal obligation compliance (e.g., financial audit trails)
    - Public interest archiving
    - Legal claims defense

Example:
    >>> service = GDPRDeletionService(repositories)
    >>> request = service.create_request("user_001")
    >>> result = service.execute_deletion(request)
    >>> report = service.get_deletion_report(result.request_id)
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set

logger = logging.getLogger("gdpr.deletion")


class DeletionStatus(Enum):
    """Status of a deletion request."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataCategory(Enum):
    """Categories of user data subject to GDPR deletion."""

    ACCOUNT = "account"  # Email, name, password hash
    PROFILE = "profile"  # Profile settings, preferences
    STRATEGIES = "strategies"  # User's trading strategies
    BACKTESTS = "backtests"  # Backtest results and history
    EXECUTION_LOGS = "execution_logs"  # Order execution history
    BROKER_CREDENTIALS = "broker_credentials"  # Encrypted API keys
    ANALYTICS = "analytics"  # Usage data, metrics
    NOTIFICATIONS = "notifications"  # Notification history
    SESSIONS = "sessions"  # Login sessions
    DISCLAIMERS = "disclaimers"  # Disclaimer acknowledgments
    AUDIT_LOGS = "audit_logs"  # Security/access logs (retention exception)


class DeletionError(Exception):
    """Base exception for deletion errors."""
    pass


class DeletionInProgressError(DeletionError):
    """Raised when a deletion is already in progress for the user."""
    pass


class DeletionNotFoundError(DeletionError):
    """Raised when a deletion request is not found."""
    pass


@dataclass
class RetentionException:
    """
    Defines a retention exception for data that cannot be immediately deleted.

    Attributes:
        category: Data category subject to retention
        reason: Legal/business reason for retention
        retention_period_years: How long data must be retained
        legal_basis: GDPR article or other legal reference
        anonymization_required: Whether data should be anonymized instead of deleted
    """

    category: DataCategory
    reason: str
    retention_period_years: int
    legal_basis: str
    anonymization_required: bool = True


@dataclass
class DeletionRequest:
    """
    Represents a GDPR deletion request.

    Attributes:
        request_id: Unique identifier for the request
        user_id: User requesting deletion
        user_email: User's email (for confirmation)
        requested_at: When the request was made
        categories: Data categories to delete
        status: Current status of the request
        completed_at: When deletion completed
        retention_exceptions: Categories kept for legal reasons
        deleted_categories: Categories successfully deleted
        anonymized_categories: Categories anonymized instead of deleted
        error_message: Error details if deletion failed
        metadata: Additional context
    """

    request_id: str
    user_id: str
    user_email: Optional[str]
    requested_at: datetime
    categories: List[DataCategory]
    status: DeletionStatus
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    retention_exceptions: List[str] = field(default_factory=list)
    deleted_categories: List[str] = field(default_factory=list)
    anonymized_categories: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "requested_at": self.requested_at.isoformat(),
            "categories": [c.value for c in self.categories],
            "status": self.status.value,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "retention_exceptions": self.retention_exceptions,
            "deleted_categories": self.deleted_categories,
            "anonymized_categories": self.anonymized_categories,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeletionRequest":
        """Deserialize from dictionary."""
        return cls(
            request_id=data["request_id"],
            user_id=data["user_id"],
            user_email=data.get("user_email"),
            requested_at=datetime.fromisoformat(data["requested_at"]),
            categories=[DataCategory(c) for c in data["categories"]],
            status=DeletionStatus(data["status"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            retention_exceptions=data.get("retention_exceptions", []),
            deleted_categories=data.get("deleted_categories", []),
            anonymized_categories=data.get("anonymized_categories", []),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


class DataRepositoryProtocol(Protocol):
    """Protocol for data repository operations."""

    def delete_by_user(self, user_id: str) -> int:
        """Delete all records for a user. Returns count of deleted records."""
        ...

    def anonymize_by_user(self, user_id: str) -> int:
        """Anonymize all records for a user. Returns count of anonymized records."""
        ...


class DeletionRequestStorageProtocol(Protocol):
    """Protocol for deletion request storage."""

    def save(self, request: DeletionRequest) -> None:
        """Save a deletion request."""
        ...

    def update(self, request: DeletionRequest) -> None:
        """Update an existing request."""
        ...

    def get(self, request_id: str) -> Optional[DeletionRequest]:
        """Get a deletion request by ID."""
        ...

    def get_by_user(self, user_id: str) -> List[DeletionRequest]:
        """Get all deletion requests for a user."""
        ...

    def get_pending(self) -> List[DeletionRequest]:
        """Get all pending deletion requests."""
        ...


class InMemoryDeletionRequestStorage:
    """In-memory implementation for testing."""

    def __init__(self):
        self._requests: Dict[str, DeletionRequest] = {}

    def save(self, request: DeletionRequest) -> None:
        """Save a deletion request."""
        self._requests[request.request_id] = request

    def update(self, request: DeletionRequest) -> None:
        """Update an existing request."""
        self._requests[request.request_id] = request

    def get(self, request_id: str) -> Optional[DeletionRequest]:
        """Get a deletion request by ID."""
        return self._requests.get(request_id)

    def get_by_user(self, user_id: str) -> List[DeletionRequest]:
        """Get all deletion requests for a user."""
        return [r for r in self._requests.values() if r.user_id == user_id]

    def get_pending(self) -> List[DeletionRequest]:
        """Get all pending deletion requests."""
        return [
            r for r in self._requests.values()
            if r.status in (DeletionStatus.PENDING, DeletionStatus.IN_PROGRESS)
        ]


class InMemoryDataRepository:
    """In-memory data repository for testing."""

    def __init__(self):
        self._data: Dict[str, List[Dict[str, Any]]] = {}

    def add_record(self, user_id: str, record: Dict[str, Any]) -> None:
        """Add a record for a user."""
        if user_id not in self._data:
            self._data[user_id] = []
        self._data[user_id].append(record)

    def delete_by_user(self, user_id: str) -> int:
        """Delete all records for a user."""
        count = len(self._data.get(user_id, []))
        self._data.pop(user_id, None)
        return count

    def anonymize_by_user(self, user_id: str) -> int:
        """Anonymize all records for a user."""
        records = self._data.get(user_id, [])
        count = len(records)
        for record in records:
            # Replace identifiable fields with anonymized values
            if "user_id" in record:
                record["user_id"] = f"anon_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"
            if "email" in record:
                record["email"] = "anonymized@deleted.local"
            if "ip_address" in record:
                record["ip_address"] = "0.0.0.0"
            if "name" in record:
                record["name"] = "ANONYMIZED"
        return count


class GDPRDeletionService:
    """
    Handles user data deletion requests per GDPR Article 17.

    Features:
        - Complete deletion across all data categories
        - Retention exceptions for legally required data
        - Anonymization for data that must be retained
        - Audit trail of deletion operations
        - Deadline enforcement (30 days per GDPR Art. 12(3))

    Retention Exceptions (GDPR Art. 17(3)):
        - Audit logs: 5 years for financial audit trail (MiFID II Art. 25)

    Example:
        >>> repos = {
        ...     "account": account_repo,
        ...     "strategies": strategies_repo,
        ...     "deletion_requests": deletion_storage,
        ... }
        >>> service = GDPRDeletionService(repos)
        >>> request = service.create_request("user_001", "user@example.com")
        >>> result = service.execute_deletion(request)
    """

    # GDPR Art. 12(3) - respond within one month
    DELETION_DEADLINE_DAYS = 30

    # Categories exempt from immediate deletion per GDPR Art. 17(3)
    RETENTION_EXCEPTIONS: Dict[DataCategory, RetentionException] = {
        DataCategory.AUDIT_LOGS: RetentionException(
            category=DataCategory.AUDIT_LOGS,
            reason="Legal obligation - financial audit trail requirement",
            retention_period_years=5,
            legal_basis="GDPR Art. 17(3)(b), MiFID II Art. 25",
            anonymization_required=True,
        ),
    }

    def __init__(
        self,
        repositories: Dict[str, Any],
        on_deletion_complete: Optional[Callable[[DeletionRequest], None]] = None,
    ):
        """
        Initialize the GDPR deletion service.

        Args:
            repositories: Dictionary mapping category names to repository instances
                         Must include 'deletion_requests' key for request storage
            on_deletion_complete: Optional callback when deletion completes
        """
        self._repos = repositories
        self._on_complete = on_deletion_complete
        self._request_storage: DeletionRequestStorageProtocol = repositories.get(
            "deletion_requests", InMemoryDeletionRequestStorage()
        )

    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        return f"del_{secrets.token_hex(8)}"

    def create_request(
        self,
        user_id: str,
        user_email: Optional[str] = None,
        categories: Optional[List[DataCategory]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeletionRequest:
        """
        Create a deletion request.

        Must be completed within 30 days per GDPR Art. 12(3).

        Args:
            user_id: User requesting deletion
            user_email: User's email for confirmation
            categories: Specific categories to delete (all if None)
            metadata: Additional context

        Returns:
            Created DeletionRequest

        Raises:
            DeletionInProgressError: If deletion is already in progress
        """
        # Check for existing pending requests
        existing = self._request_storage.get_by_user(user_id)
        for req in existing:
            if req.status in (DeletionStatus.PENDING, DeletionStatus.IN_PROGRESS):
                raise DeletionInProgressError(
                    f"Deletion request {req.request_id} already in progress for user {user_id}"
                )

        # Default to all categories
        if categories is None:
            categories = list(DataCategory)

        now = datetime.now(timezone.utc)
        deadline = now + timedelta(days=self.DELETION_DEADLINE_DAYS)

        request = DeletionRequest(
            request_id=self._generate_request_id(),
            user_id=user_id,
            user_email=user_email,
            requested_at=now,
            categories=categories,
            status=DeletionStatus.PENDING,
            deadline=deadline,
            metadata=metadata or {},
        )

        self._request_storage.save(request)

        logger.info(
            f"DELETION_REQUEST_CREATED | "
            f"request_id={request.request_id} | "
            f"user_id={user_id} | "
            f"categories={len(categories)} | "
            f"deadline={deadline.isoformat()}"
        )

        return request

    def execute_deletion(self, request: DeletionRequest) -> DeletionRequest:
        """
        Execute deletion across all data stores.

        Args:
            request: The deletion request to execute

        Returns:
            Updated DeletionRequest with results

        Raises:
            DeletionError: If deletion fails
        """
        request.status = DeletionStatus.IN_PROGRESS
        self._request_storage.update(request)

        logger.info(
            f"DELETION_STARTED | "
            f"request_id={request.request_id} | "
            f"user_id={request.user_id}"
        )

        errors: List[str] = []

        for category in request.categories:
            try:
                if category in self.RETENTION_EXCEPTIONS:
                    # Handle retention exception - anonymize instead
                    exception = self.RETENTION_EXCEPTIONS[category]
                    request.retention_exceptions.append(
                        f"{category.value}: {exception.reason} "
                        f"(retained for {exception.retention_period_years} years per {exception.legal_basis})"
                    )

                    if exception.anonymization_required:
                        count = self._anonymize_category(request.user_id, category)
                        request.anonymized_categories.append(category.value)
                        logger.info(
                            f"CATEGORY_ANONYMIZED | "
                            f"request_id={request.request_id} | "
                            f"category={category.value} | "
                            f"records={count}"
                        )
                else:
                    # Standard deletion
                    count = self._delete_category(request.user_id, category)
                    request.deleted_categories.append(category.value)
                    logger.info(
                        f"CATEGORY_DELETED | "
                        f"request_id={request.request_id} | "
                        f"category={category.value} | "
                        f"records={count}"
                    )

            except Exception as e:
                error_msg = f"Failed to process {category.value}: {str(e)}"
                errors.append(error_msg)
                logger.error(
                    f"DELETION_ERROR | "
                    f"request_id={request.request_id} | "
                    f"category={category.value} | "
                    f"error={str(e)}"
                )

        # Determine final status
        request.completed_at = datetime.now(timezone.utc)

        if errors:
            request.status = DeletionStatus.PARTIALLY_COMPLETED
            request.error_message = "; ".join(errors)
        else:
            request.status = DeletionStatus.COMPLETED

        self._request_storage.update(request)

        logger.info(
            f"DELETION_COMPLETED | "
            f"request_id={request.request_id} | "
            f"status={request.status.value} | "
            f"deleted={len(request.deleted_categories)} | "
            f"anonymized={len(request.anonymized_categories)} | "
            f"exceptions={len(request.retention_exceptions)}"
        )

        # Call completion callback
        if self._on_complete:
            self._on_complete(request)

        return request

    def _delete_category(self, user_id: str, category: DataCategory) -> int:
        """
        Delete all data in category for user.

        Args:
            user_id: User to delete data for
            category: Data category to delete

        Returns:
            Number of records deleted
        """
        repo = self._repos.get(category.value)
        if repo is None:
            logger.warning(f"No repository found for category {category.value}")
            return 0

        count = repo.delete_by_user(user_id)
        return count

    def _anonymize_category(self, user_id: str, category: DataCategory) -> int:
        """
        Anonymize data that must be retained.

        Args:
            user_id: User to anonymize data for
            category: Data category to anonymize

        Returns:
            Number of records anonymized
        """
        repo = self._repos.get(category.value)
        if repo is None:
            logger.warning(f"No repository found for category {category.value}")
            return 0

        count = repo.anonymize_by_user(user_id)
        return count

    def get_deletion_report(self, request_id: str) -> Dict[str, Any]:
        """
        Generate report for user showing what was deleted.

        Args:
            request_id: ID of the deletion request

        Returns:
            Report dictionary with deletion details

        Raises:
            DeletionNotFoundError: If request not found
        """
        request = self._request_storage.get(request_id)
        if request is None:
            raise DeletionNotFoundError(f"Deletion request {request_id} not found")

        # Calculate time taken
        processing_time = None
        if request.completed_at:
            processing_time = (request.completed_at - request.requested_at).total_seconds()

        # Check if completed within deadline
        within_deadline = True
        if request.deadline and request.completed_at:
            within_deadline = request.completed_at <= request.deadline

        return {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "status": request.status.value,
            "requested_at": request.requested_at.isoformat(),
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "deadline": request.deadline.isoformat() if request.deadline else None,
            "within_deadline": within_deadline,
            "processing_time_seconds": processing_time,
            "deleted_categories": request.deleted_categories,
            "anonymized_categories": request.anonymized_categories,
            "retention_exceptions": request.retention_exceptions,
            "error_message": request.error_message,
            "compliance": {
                "gdpr_article": "Article 17 - Right to Erasure",
                "deadline_days": self.DELETION_DEADLINE_DAYS,
                "retention_basis": "GDPR Art. 17(3)",
            },
        }

    def cancel_request(self, request_id: str) -> DeletionRequest:
        """
        Cancel a pending deletion request.

        Args:
            request_id: ID of the request to cancel

        Returns:
            Updated DeletionRequest

        Raises:
            DeletionNotFoundError: If request not found
            DeletionError: If request cannot be cancelled
        """
        request = self._request_storage.get(request_id)
        if request is None:
            raise DeletionNotFoundError(f"Deletion request {request_id} not found")

        if request.status not in (DeletionStatus.PENDING,):
            raise DeletionError(
                f"Cannot cancel request in {request.status.value} status"
            )

        request.status = DeletionStatus.CANCELLED
        request.completed_at = datetime.now(timezone.utc)
        self._request_storage.update(request)

        logger.info(
            f"DELETION_CANCELLED | "
            f"request_id={request_id} | "
            f"user_id={request.user_id}"
        )

        return request

    def get_pending_requests(self) -> List[DeletionRequest]:
        """
        Get all pending deletion requests.

        Useful for background processing jobs.

        Returns:
            List of pending requests
        """
        return self._request_storage.get_pending()

    def get_overdue_requests(self) -> List[DeletionRequest]:
        """
        Get requests that are past their deadline.

        Returns:
            List of overdue requests
        """
        now = datetime.now(timezone.utc)
        pending = self.get_pending_requests()
        return [r for r in pending if r.deadline and r.deadline < now]

    def get_user_deletion_history(self, user_id: str) -> List[DeletionRequest]:
        """
        Get deletion history for a user.

        Args:
            user_id: User to query

        Returns:
            List of deletion requests for the user
        """
        return self._request_storage.get_by_user(user_id)

    def process_pending_deletions(self) -> List[DeletionRequest]:
        """
        Process all pending deletion requests.

        Useful for scheduled background jobs.

        Returns:
            List of processed requests
        """
        pending = self.get_pending_requests()
        results = []

        for request in pending:
            if request.status == DeletionStatus.PENDING:
                try:
                    result = self.execute_deletion(request)
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"DELETION_PROCESSING_FAILED | "
                        f"request_id={request.request_id} | "
                        f"error={str(e)}"
                    )
                    request.status = DeletionStatus.FAILED
                    request.error_message = str(e)
                    self._request_storage.update(request)
                    results.append(request)

        return results
