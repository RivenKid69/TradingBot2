# -*- coding: utf-8 -*-
"""
Agent Update Manager - Signed agent updates distribution.

CLOUD ZONE ONLY.

Manages agent software updates with:
- Signed update packages
- Staged rollout (canary → early adopters → general)
- Rollback capability
- Version pinning for enterprise
- Change windows enforcement

Design Doc Reference:
    - Phase 9: Enterprise/on-prem pack
    - Design Doc 15.2/5.2: Agent updates
    - Design Doc 9: Agent runtime
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple
from uuid import UUID, uuid4

# Local crypto module
try:
    from .crypto import Ed25519Signer, SigningKey, Signature, CRYPTO_AVAILABLE
except ImportError:
    CRYPTO_AVAILABLE = False
    Ed25519Signer = None
    SigningKey = None
    Signature = None

logger = logging.getLogger(__name__)

# Constants
CURRENT_AGENT_VERSION: Final[str] = "1.0.0"
MIN_SUPPORTED_VERSION: Final[str] = "0.9.0"


class UpdateChannel(Enum):
    """Update channel for agent updates."""
    STABLE = "stable"
    BETA = "beta"
    CANARY = "canary"
    ENTERPRISE = "enterprise"  # Pinned versions


class UpdateState(Enum):
    """State of an update for an agent."""
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    INSTALLING = "installing"
    INSTALLED = "installed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


class UpdatePriority(Enum):
    """Update priority level."""
    CRITICAL = "critical"  # Security fix, immediate
    HIGH = "high"  # Important bug fix
    NORMAL = "normal"  # Standard update
    LOW = "low"  # Minor improvement


@dataclass
class AgentUpdate:
    """Represents an agent software update."""
    id: UUID = field(default_factory=uuid4)
    version: str = ""
    channel: UpdateChannel = UpdateChannel.STABLE

    # Update metadata
    title: str = ""
    description: str = ""
    release_notes: str = ""
    priority: UpdatePriority = UpdatePriority.NORMAL

    # Artifacts
    artifact_digest: str = ""  # sha256:...
    artifact_url: str = ""
    artifact_size_bytes: int = 0

    # Signature
    signature: str = ""
    signature_algorithm: str = "ed25519"
    signed_by: str = ""
    signed_at: Optional[datetime] = None

    # Requirements
    min_current_version: str = ""  # Minimum version to update from
    max_current_version: str = ""  # Maximum version to update from
    requires_restart: bool = True
    requires_approval: bool = True
    is_trading_impacting: bool = True

    # Rollout
    rollout_percentage: float = 0.0
    rollout_stage: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    released_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None

    # Status
    is_active: bool = True
    is_mandatory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "version": self.version,
            "channel": self.channel.value,
            "title": self.title,
            "description": self.description,
            "release_notes": self.release_notes,
            "priority": self.priority.value,
            "artifact_digest": self.artifact_digest,
            "artifact_url": self.artifact_url,
            "artifact_size_bytes": self.artifact_size_bytes,
            "signature": self.signature,
            "signature_algorithm": self.signature_algorithm,
            "signed_by": self.signed_by,
            "signed_at": self.signed_at.isoformat() if self.signed_at else None,
            "min_current_version": self.min_current_version,
            "max_current_version": self.max_current_version,
            "requires_restart": self.requires_restart,
            "requires_approval": self.requires_approval,
            "is_trading_impacting": self.is_trading_impacting,
            "rollout_percentage": self.rollout_percentage,
            "rollout_stage": self.rollout_stage,
            "created_at": self.created_at.isoformat(),
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "is_active": self.is_active,
            "is_mandatory": self.is_mandatory,
        }


@dataclass
class AgentUpdateStatus:
    """Status of an update for a specific agent."""
    agent_id: UUID
    update_id: UUID
    state: UpdateState = UpdateState.AVAILABLE
    current_version: str = ""
    target_version: str = ""

    # Progress
    download_progress: float = 0.0
    install_progress: float = 0.0

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Approval
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    evidence_hash: Optional[str] = None

    # Rollback
    rollback_version: Optional[str] = None
    rolled_back_at: Optional[datetime] = None
    rollback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": str(self.agent_id),
            "update_id": str(self.update_id),
            "state": self.state.value,
            "current_version": self.current_version,
            "target_version": self.target_version,
            "download_progress": self.download_progress,
            "install_progress": self.install_progress,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rollback_version": self.rollback_version,
            "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
        }


@dataclass
class AgentUpdateConfig:
    """Configuration for agent update manager."""
    # Channels
    default_channel: UpdateChannel = UpdateChannel.STABLE
    allowed_channels: List[UpdateChannel] = field(
        default_factory=lambda: [UpdateChannel.STABLE, UpdateChannel.ENTERPRISE]
    )

    # Auto-update settings
    auto_download: bool = True
    auto_install: bool = False  # Always require approval for install
    auto_approve_non_trading: bool = True

    # Version constraints
    min_supported_version: str = MIN_SUPPORTED_VERSION
    max_supported_version: str = ""  # Empty = no limit

    # Rollout
    enable_staged_rollout: bool = True
    rollout_stages: List[str] = field(
        default_factory=lambda: ["canary", "early_adopters", "general"]
    )

    # Change windows
    enable_change_windows: bool = False
    change_window_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    change_window_start_hour: int = 9
    change_window_end_hour: int = 17
    change_window_timezone: str = "UTC"

    # Verification
    require_signatures: bool = True
    trusted_signing_keys: List[str] = field(default_factory=list)

    # Rollback
    enable_auto_rollback: bool = True
    rollback_on_failure: bool = True
    rollback_health_check_timeout: int = 300  # 5 minutes
    keep_previous_versions: int = 3

    # Notifications
    notify_on_available: bool = True
    notify_on_installed: bool = True
    notify_on_failed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "default_channel": self.default_channel.value,
            "allowed_channels": [c.value for c in self.allowed_channels],
            "auto_download": self.auto_download,
            "auto_install": self.auto_install,
            "min_supported_version": self.min_supported_version,
            "max_supported_version": self.max_supported_version,
            "enable_staged_rollout": self.enable_staged_rollout,
            "enable_change_windows": self.enable_change_windows,
            "require_signatures": self.require_signatures,
            "enable_auto_rollback": self.enable_auto_rollback,
        }


class AgentUpdateManager:
    """
    Agent Update Manager for signed software distribution.

    Manages agent updates with:
    - Signed update packages (ed25519/sigstore)
    - Staged rollout with automatic progression
    - Rollback on failure
    - Version constraints and pinning
    - Change window enforcement

    Usage:
        config = AgentUpdateConfig(
            enable_staged_rollout=True,
            require_signatures=True,
        )
        manager = AgentUpdateManager(config)

        # Create update
        update = manager.create_update(
            version="1.1.0",
            artifact_digest="sha256:abc123...",
            artifact_url="https://...",
        )

        # Sign update
        await manager.sign_update(update.id, signing_key)

        # Release update
        await manager.release_update(update.id)

        # Check for updates for an agent
        available = await manager.check_for_updates(agent_id, current_version)
    """

    def __init__(
        self,
        config: Optional[AgentUpdateConfig] = None,
        db_session: Optional[Any] = None,
        on_update_available: Optional[Callable[[UUID, AgentUpdate], None]] = None,
        on_update_installed: Optional[Callable[[UUID, AgentUpdate], None]] = None,
        on_update_failed: Optional[Callable[[UUID, AgentUpdate, str], None]] = None,
    ):
        """
        Initialize update manager.

        Args:
            config: Update configuration
            db_session: Database session
            on_update_available: Callback when update available
            on_update_installed: Callback when update installed
            on_update_failed: Callback when update failed
        """
        self.config = config or AgentUpdateConfig()
        self._db = db_session

        # Callbacks
        self._on_update_available = on_update_available
        self._on_update_installed = on_update_installed
        self._on_update_failed = on_update_failed

        # In-memory state (would be in database in production)
        self._updates: Dict[UUID, AgentUpdate] = {}
        self._agent_status: Dict[Tuple[UUID, UUID], AgentUpdateStatus] = {}

    def create_update(
        self,
        version: str,
        artifact_digest: str,
        artifact_url: str,
        channel: UpdateChannel = UpdateChannel.STABLE,
        title: str = "",
        description: str = "",
        release_notes: str = "",
        priority: UpdatePriority = UpdatePriority.NORMAL,
        artifact_size_bytes: int = 0,
        min_current_version: str = "",
        max_current_version: str = "",
        requires_restart: bool = True,
        is_mandatory: bool = False,
    ) -> AgentUpdate:
        """
        Create a new agent update.

        Args:
            version: Target version
            artifact_digest: Artifact digest (sha256:...)
            artifact_url: Download URL
            channel: Update channel
            title: Update title
            description: Description
            release_notes: Release notes
            priority: Update priority
            artifact_size_bytes: Artifact size
            min_current_version: Minimum version to update from
            max_current_version: Maximum version to update from
            requires_restart: Whether restart is required
            is_mandatory: Whether update is mandatory

        Returns:
            Created update
        """
        update = AgentUpdate(
            version=version,
            channel=channel,
            title=title or f"Agent Update {version}",
            description=description,
            release_notes=release_notes,
            priority=priority,
            artifact_digest=artifact_digest,
            artifact_url=artifact_url,
            artifact_size_bytes=artifact_size_bytes,
            min_current_version=min_current_version or self.config.min_supported_version,
            max_current_version=max_current_version,
            requires_restart=requires_restart,
            requires_approval=True,  # Always require approval
            is_trading_impacting=requires_restart,  # Restart = trading impacting
            is_mandatory=is_mandatory,
        )

        self._updates[update.id] = update
        logger.info(f"Created update: {update.version} ({update.id})")

        return update

    async def sign_update(
        self,
        update_id: UUID,
        signing_key: bytes,
        signer_id: str = "ccea-update-signer",
    ) -> bool:
        """
        Sign an update with the given key.

        Args:
            update_id: Update ID
            signing_key: Ed25519 signing key
            signer_id: Signer identifier

        Returns:
            True if signed successfully
        """
        update = self._updates.get(update_id)
        if not update:
            logger.error(f"Update not found: {update_id}")
            return False

        try:
            # Create signature payload
            payload = self._create_signature_payload(update)

            # Sign payload
            signature = await self._sign_payload(payload, signing_key)

            update.signature = signature
            update.signed_by = signer_id
            update.signed_at = datetime.utcnow()

            logger.info(f"Signed update: {update.version} by {signer_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to sign update {update_id}: {e}")
            return False

    async def verify_update_signature(
        self,
        update_id: UUID,
        public_key: Optional[bytes] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify update signature.

        Args:
            update_id: Update ID
            public_key: Public key (uses trusted keys if not provided)

        Returns:
            (is_valid, error_message)
        """
        update = self._updates.get(update_id)
        if not update:
            return False, "Update not found"

        if not update.signature:
            return False, "Update is not signed"

        try:
            payload = self._create_signature_payload(update)
            is_valid = await self._verify_signature(
                payload,
                update.signature,
                public_key,
            )

            if is_valid:
                return True, None
            else:
                return False, "Invalid signature"

        except Exception as e:
            return False, f"Verification error: {e}"

    async def release_update(
        self,
        update_id: UUID,
        initial_rollout_percentage: float = 5.0,
    ) -> bool:
        """
        Release an update for distribution.

        Args:
            update_id: Update ID
            initial_rollout_percentage: Initial rollout percentage (for staged)

        Returns:
            True if released
        """
        update = self._updates.get(update_id)
        if not update:
            logger.error(f"Update not found: {update_id}")
            return False

        # Verify signature if required
        if self.config.require_signatures:
            is_valid, error = await self.verify_update_signature(update_id)
            if not is_valid:
                logger.error(f"Cannot release unsigned/invalid update: {error}")
                return False

        # Set rollout
        if self.config.enable_staged_rollout:
            update.rollout_percentage = initial_rollout_percentage
            update.rollout_stage = self.config.rollout_stages[0]
        else:
            update.rollout_percentage = 100.0
            update.rollout_stage = "general"

        update.released_at = datetime.utcnow()
        update.is_active = True

        logger.info(
            f"Released update: {update.version} at {update.rollout_percentage}% "
            f"({update.rollout_stage})"
        )

        return True

    async def check_for_updates(
        self,
        agent_id: UUID,
        current_version: str,
        channel: Optional[UpdateChannel] = None,
    ) -> Optional[AgentUpdate]:
        """
        Check for available updates for an agent.

        Args:
            agent_id: Agent ID
            current_version: Agent's current version
            channel: Preferred channel

        Returns:
            Available update or None
        """
        channel = channel or self.config.default_channel

        # Check if within change window
        if self.config.enable_change_windows:
            if not self._is_within_change_window():
                logger.debug("Outside change window, no updates available")
                return None

        # Find applicable updates
        for update in self._updates.values():
            if not update.is_active or not update.released_at:
                continue

            if update.channel != channel:
                continue

            # Check version constraints
            if not self._is_version_applicable(
                current_version,
                update.min_current_version,
                update.max_current_version,
                update.version,
            ):
                continue

            # Check rollout eligibility
            if self.config.enable_staged_rollout:
                if not self._is_in_rollout(agent_id, update):
                    continue

            # Check if already processed
            status_key = (agent_id, update.id)
            if status_key in self._agent_status:
                status = self._agent_status[status_key]
                if status.state in (
                    UpdateState.INSTALLED,
                    UpdateState.SKIPPED,
                ):
                    continue

            # Found applicable update
            logger.info(f"Update available for agent {agent_id}: {update.version}")

            # Create or get status
            if status_key not in self._agent_status:
                self._agent_status[status_key] = AgentUpdateStatus(
                    agent_id=agent_id,
                    update_id=update.id,
                    current_version=current_version,
                    target_version=update.version,
                )

            # Notify
            if self._on_update_available:
                try:
                    self._on_update_available(agent_id, update)
                except Exception as e:
                    logger.error(f"Update available callback error: {e}")

            return update

        return None

    async def download_update(
        self,
        agent_id: UUID,
        update_id: UUID,
    ) -> Tuple[bool, Optional[str]]:
        """
        Mark update as downloading/downloaded.

        In practice, the agent downloads the update locally.
        This tracks the state in the cloud.

        Args:
            agent_id: Agent ID
            update_id: Update ID

        Returns:
            (success, error_message)
        """
        status_key = (agent_id, update_id)
        if status_key not in self._agent_status:
            return False, "No update status found"

        status = self._agent_status[status_key]

        if status.state not in (UpdateState.AVAILABLE, UpdateState.FAILED):
            return False, f"Cannot download from state: {status.state.value}"

        status.state = UpdateState.DOWNLOADING
        status.started_at = datetime.utcnow()

        logger.info(f"Agent {agent_id} downloading update {update_id}")
        return True, None

    async def mark_downloaded(
        self,
        agent_id: UUID,
        update_id: UUID,
        verified_digest: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Mark update as downloaded and verified.

        Args:
            agent_id: Agent ID
            update_id: Update ID
            verified_digest: Digest verified by agent

        Returns:
            (success, error_message)
        """
        status_key = (agent_id, update_id)
        if status_key not in self._agent_status:
            return False, "No update status found"

        status = self._agent_status[status_key]
        update = self._updates.get(update_id)

        if not update:
            return False, "Update not found"

        # Verify digest matches
        if verified_digest != update.artifact_digest:
            status.state = UpdateState.FAILED
            status.error_message = "Digest mismatch"
            return False, "Artifact digest mismatch"

        status.state = UpdateState.DOWNLOADED
        status.download_progress = 100.0

        # Determine if approval needed
        if update.requires_approval:
            status.state = UpdateState.PENDING_APPROVAL

        logger.info(f"Agent {agent_id} downloaded update {update_id}")
        return True, None

    async def approve_update(
        self,
        agent_id: UUID,
        update_id: UUID,
        approved_by: str,
        evidence_hash: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Approve update installation.

        Args:
            agent_id: Agent ID
            update_id: Update ID
            approved_by: Approver identifier
            evidence_hash: Evidence hash for audit

        Returns:
            (success, error_message)
        """
        status_key = (agent_id, update_id)
        if status_key not in self._agent_status:
            return False, "No update status found"

        status = self._agent_status[status_key]

        if status.state not in (
            UpdateState.DOWNLOADED,
            UpdateState.PENDING_APPROVAL,
        ):
            return False, f"Cannot approve from state: {status.state.value}"

        status.state = UpdateState.APPROVED
        status.approved_by = approved_by
        status.approved_at = datetime.utcnow()
        status.evidence_hash = evidence_hash

        logger.info(
            f"Update {update_id} approved for agent {agent_id} by {approved_by}"
        )
        return True, None

    async def install_update(
        self,
        agent_id: UUID,
        update_id: UUID,
    ) -> Tuple[bool, Optional[str]]:
        """
        Begin update installation.

        Args:
            agent_id: Agent ID
            update_id: Update ID

        Returns:
            (success, error_message)
        """
        status_key = (agent_id, update_id)
        if status_key not in self._agent_status:
            return False, "No update status found"

        status = self._agent_status[status_key]
        update = self._updates.get(update_id)

        if status.state != UpdateState.APPROVED:
            return False, f"Cannot install from state: {status.state.value}"

        # Check change window again
        if self.config.enable_change_windows:
            if not self._is_within_change_window():
                return False, "Outside change window"

        status.state = UpdateState.INSTALLING
        status.install_progress = 0.0

        logger.info(f"Agent {agent_id} installing update {update_id}")
        return True, None

    async def complete_installation(
        self,
        agent_id: UUID,
        update_id: UUID,
        new_version: str,
        success: bool,
        error_message: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Complete update installation.

        Args:
            agent_id: Agent ID
            update_id: Update ID
            new_version: New agent version after update
            success: Whether installation succeeded
            error_message: Error message if failed

        Returns:
            (success, error_message)
        """
        status_key = (agent_id, update_id)
        if status_key not in self._agent_status:
            return False, "No update status found"

        status = self._agent_status[status_key]
        update = self._updates.get(update_id)

        if success:
            status.state = UpdateState.INSTALLED
            status.install_progress = 100.0
            status.completed_at = datetime.utcnow()
            status.current_version = new_version

            logger.info(f"Agent {agent_id} updated to {new_version}")

            # Notify
            if self._on_update_installed and update:
                try:
                    self._on_update_installed(agent_id, update)
                except Exception as e:
                    logger.error(f"Update installed callback error: {e}")

        else:
            status.state = UpdateState.FAILED
            status.error_message = error_message
            status.retry_count += 1

            logger.error(
                f"Agent {agent_id} update failed: {error_message}"
            )

            # Notify
            if self._on_update_failed and update:
                try:
                    self._on_update_failed(agent_id, update, error_message or "Unknown error")
                except Exception as e:
                    logger.error(f"Update failed callback error: {e}")

            # Auto-rollback if configured
            if self.config.rollback_on_failure and status.rollback_version:
                await self.initiate_rollback(
                    agent_id,
                    update_id,
                    reason=error_message or "Installation failed",
                )

        return True, None

    async def initiate_rollback(
        self,
        agent_id: UUID,
        update_id: UUID,
        reason: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Initiate rollback to previous version.

        Args:
            agent_id: Agent ID
            update_id: Update ID
            reason: Rollback reason

        Returns:
            (success, error_message)
        """
        status_key = (agent_id, update_id)
        if status_key not in self._agent_status:
            return False, "No update status found"

        status = self._agent_status[status_key]

        if not status.rollback_version:
            return False, "No rollback version available"

        status.state = UpdateState.ROLLED_BACK
        status.rolled_back_at = datetime.utcnow()
        status.rollback_reason = reason

        logger.warning(
            f"Agent {agent_id} rolling back from {status.target_version} "
            f"to {status.rollback_version}: {reason}"
        )

        return True, None

    async def progress_rollout(
        self,
        update_id: UUID,
    ) -> bool:
        """
        Progress staged rollout to next stage.

        Args:
            update_id: Update ID

        Returns:
            True if progressed
        """
        update = self._updates.get(update_id)
        if not update:
            return False

        stages = self.config.rollout_stages
        current_idx = -1

        for i, stage in enumerate(stages):
            if stage == update.rollout_stage:
                current_idx = i
                break

        if current_idx < 0 or current_idx >= len(stages) - 1:
            # Already at final stage or not found
            return False

        # Progress to next stage
        next_stage = stages[current_idx + 1]

        # Set percentage based on stage
        if next_stage == "canary":
            update.rollout_percentage = 5.0
        elif next_stage == "early_adopters":
            update.rollout_percentage = 25.0
        elif next_stage == "general":
            update.rollout_percentage = 100.0
        else:
            update.rollout_percentage = min(
                update.rollout_percentage * 2,
                100.0,
            )

        update.rollout_stage = next_stage

        logger.info(
            f"Progressed rollout for {update.version}: "
            f"{next_stage} ({update.rollout_percentage}%)"
        )

        return True

    def get_update(self, update_id: UUID) -> Optional[AgentUpdate]:
        """Get update by ID."""
        return self._updates.get(update_id)

    def get_update_status(
        self,
        agent_id: UUID,
        update_id: UUID,
    ) -> Optional[AgentUpdateStatus]:
        """Get update status for an agent."""
        return self._agent_status.get((agent_id, update_id))

    def list_updates(
        self,
        channel: Optional[UpdateChannel] = None,
        active_only: bool = True,
    ) -> List[AgentUpdate]:
        """List all updates."""
        updates = list(self._updates.values())

        if channel:
            updates = [u for u in updates if u.channel == channel]

        if active_only:
            updates = [u for u in updates if u.is_active]

        return sorted(updates, key=lambda u: u.created_at, reverse=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get update statistics."""
        total_updates = len(self._updates)
        active_updates = sum(1 for u in self._updates.values() if u.is_active)
        released_updates = sum(
            1 for u in self._updates.values() if u.released_at
        )

        # Status breakdown
        status_counts: Dict[str, int] = {}
        for status in self._agent_status.values():
            state = status.state.value
            status_counts[state] = status_counts.get(state, 0) + 1

        return {
            "total_updates": total_updates,
            "active_updates": active_updates,
            "released_updates": released_updates,
            "agent_status_counts": status_counts,
            "config": self.config.to_dict(),
        }

    # ===== Private Methods =====

    def _create_signature_payload(self, update: AgentUpdate) -> bytes:
        """Create payload for signing."""
        data = {
            "version": update.version,
            "artifact_digest": update.artifact_digest,
            "artifact_url": update.artifact_url,
            "created_at": update.created_at.isoformat(),
        }
        return json.dumps(data, sort_keys=True).encode()

    async def _sign_payload(
        self,
        payload: bytes,
        signing_key: bytes,
    ) -> str:
        """
        Sign payload with Ed25519 key.

        Uses real Ed25519 signing when cryptography library is available.

        Args:
            payload: Data to sign
            signing_key: Raw Ed25519 private key (32 bytes)

        Returns:
            JSON-encoded signature
        """
        if not CRYPTO_AVAILABLE:
            # SECURITY: Fail-closed per CCEA Design Doc Section 15.2
            # Cryptography library is mandatory for agent update signing
            logger.error(
                "SECURITY: Cryptography library not available. "
                "Agent update signing requires proper Ed25519 cryptography. "
                "Install cryptography package: pip install cryptography"
            )
            raise RuntimeError(
                "Cryptography library required for agent update signing. "
                "This is a security requirement per CCEA Design Doc Section 15.2."
            )

        try:
            signer = Ed25519Signer(default_signer_id="ccea-agent-update-signer")

            # Import the raw signing key
            key = signer.import_private_key(signing_key, format="raw")

            # Sign the payload
            signature = signer.sign(
                payload,
                key,
                payload_type="agent-update",
                signer_id="ccea-agent-update-signer",
            )

            return json.dumps(signature.to_dict())

        except Exception as e:
            logger.error(f"Failed to sign payload: {e}")
            raise

    async def _verify_signature(
        self,
        payload: bytes,
        signature: str,
        public_key: Optional[bytes] = None,
    ) -> bool:
        """
        Verify Ed25519 signature.

        Args:
            payload: Original payload
            signature: JSON-encoded signature or base64 raw signature
            public_key: Raw Ed25519 public key (32 bytes)

        Returns:
            True if signature is valid
        """
        if not CRYPTO_AVAILABLE:
            # SECURITY: Fail-closed per CCEA Design Doc Section 15.2
            # Cryptography library is mandatory for signature verification
            logger.error(
                "SECURITY: Cryptography library not available. "
                "Agent update verification requires proper Ed25519 cryptography. "
                "Signature verification FAILED (fail-closed)."
            )
            return False

        try:
            # Try to parse as JSON signature
            try:
                sig_data = json.loads(signature)
                if "signature" in sig_data:
                    sig_obj = Signature.from_dict(sig_data)

                    signer = Ed25519Signer()

                    # Use provided public key or trusted keys
                    if public_key:
                        key = signer.import_public_key(
                            public_key,
                            key_id=sig_obj.key_id,
                            format="raw"
                        )
                        signer.add_trusted_key(key)
                    elif self.config.trusted_signing_keys:
                        # Add trusted keys from config
                        for key_b64 in self.config.trusted_signing_keys:
                            import base64
                            key_bytes = base64.b64decode(key_b64)
                            key = signer.import_public_key(key_bytes, format="raw")
                            signer.add_trusted_key(key)

                    return signer.verify(payload, sig_obj)

            except json.JSONDecodeError:
                # Legacy base64 signature format
                pass

            # Fallback: if we have a public key but no structured signature,
            # we can't verify properly
            logger.warning("Could not verify signature - unrecognized format")
            return False

        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    def _is_version_applicable(
        self,
        current: str,
        min_version: str,
        max_version: str,
        target: str,
    ) -> bool:
        """Check if update is applicable for current version."""
        from packaging import version as pkg_version

        try:
            current_v = pkg_version.parse(current)
            target_v = pkg_version.parse(target)

            # Must be upgrade
            if current_v >= target_v:
                return False

            # Check min constraint
            if min_version:
                min_v = pkg_version.parse(min_version)
                if current_v < min_v:
                    return False

            # Check max constraint
            if max_version:
                max_v = pkg_version.parse(max_version)
                if current_v > max_v:
                    return False

            return True

        except Exception:
            return False

    def _is_in_rollout(self, agent_id: UUID, update: AgentUpdate) -> bool:
        """Check if agent is in rollout percentage."""
        # Use agent_id hash for deterministic rollout
        hash_value = int(hashlib.sha256(str(agent_id).encode()).hexdigest(), 16)
        percentage = (hash_value % 10000) / 100  # 0-100

        return percentage < update.rollout_percentage

    def _is_within_change_window(self) -> bool:
        """Check if current time is within change window."""
        from datetime import timezone

        now = datetime.now(timezone.utc)

        # Check day of week (0 = Monday)
        if now.weekday() not in self.config.change_window_days:
            return False

        # Check hour
        if not (
            self.config.change_window_start_hour
            <= now.hour
            < self.config.change_window_end_hour
        ):
            return False

        return True
