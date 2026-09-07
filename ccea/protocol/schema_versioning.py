# -*- coding: utf-8 -*-
"""
CCEA Schema Version Negotiation.

Implements schema version negotiation between Cloud and Agent per Design Doc.
Supports semantic versioning with min/max supported version ranges.

Version Compatibility Rules (SemVer):
- MAJOR: Breaking changes, not backward compatible
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

Negotiation Protocol:
1. Agent sends supported version range in POLL_COMMANDS
2. Cloud responds with selected version in COMMAND_BATCH
3. If no compatible version exists, negotiation fails

SECURITY: Version negotiation must complete before any commands are accepted.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Current schema version (from protocol_messages.schema.json)
CURRENT_SCHEMA_VERSION = "1.0.0"
MIN_SUPPORTED_VERSION = "1.0.0"
MAX_SUPPORTED_VERSION = "1.0.0"

# Semantic version pattern
SEMVER_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


# ============================================================================
# Enums
# ============================================================================


class NegotiationStatus(str, Enum):
    """Status of version negotiation."""

    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"


class VersionCompatibility(str, Enum):
    """Result of version compatibility check."""

    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE_MAJOR = "INCOMPATIBLE_MAJOR"
    INCOMPATIBLE_TOO_OLD = "INCOMPATIBLE_TOO_OLD"
    INCOMPATIBLE_TOO_NEW = "INCOMPATIBLE_TOO_NEW"
    INVALID_VERSION = "INVALID_VERSION"


# ============================================================================
# Schema Version Model
# ============================================================================


class SchemaVersion(BaseModel):
    """
    Semantic version representation for schema versioning.

    Follows SemVer 2.0.0 specification:
    - MAJOR version for incompatible API changes
    - MINOR version for backward-compatible new functionality
    - PATCH version for backward-compatible bug fixes
    """

    major: int = Field(..., ge=0, description="Major version (breaking changes)")
    minor: int = Field(..., ge=0, description="Minor version (new features)")
    patch: int = Field(..., ge=0, description="Patch version (bug fixes)")

    model_config = {"extra": "forbid"}

    @classmethod
    def parse(cls, version_str: str) -> "SchemaVersion":
        """
        Parse version string into SchemaVersion.

        Args:
            version_str: Version string in format "MAJOR.MINOR.PATCH"

        Returns:
            SchemaVersion instance

        Raises:
            ValueError: If version string is invalid
        """
        match = SEMVER_PATTERN.match(version_str.strip())
        if not match:
            raise ValueError(
                f"Invalid version format: '{version_str}'. "
                "Expected MAJOR.MINOR.PATCH (e.g., '1.0.0')"
            )
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3)),
        )

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        return f"SchemaVersion({self.major}.{self.minor}.{self.patch})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = SchemaVersion.parse(other)
            except ValueError:
                return False
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return self.major == other.major and self.minor == other.minor and self.patch == other.patch

    def __lt__(self, other: "SchemaVersion") -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "SchemaVersion") -> bool:
        return self == other or self < other

    def __gt__(self, other: "SchemaVersion") -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)

    def __ge__(self, other: "SchemaVersion") -> bool:
        return self == other or self > other

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """
        Check if this version is compatible with another version.

        Compatibility rules (SemVer):
        - Same MAJOR version is required
        - Higher MINOR is backward compatible
        - PATCH differences are always compatible

        Args:
            other: Version to check compatibility with

        Returns:
            True if versions are compatible
        """
        # Different MAJOR versions are never compatible
        if self.major != other.major:
            return False
        return True

    def to_tuple(self) -> Tuple[int, int, int]:
        """Return version as tuple (major, minor, patch)."""
        return (self.major, self.minor, self.patch)


# ============================================================================
# Negotiation Request/Result Models
# ============================================================================


class VersionNegotiationRequest(BaseModel):
    """
    Version negotiation request from Agent to Cloud.

    Sent in POLL_COMMANDS message to indicate supported versions.
    """

    agent_id: str = Field(..., pattern=r"^agent_[a-zA-Z0-9]{16,32}$")
    min_supported: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    max_supported: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    preferred: Optional[str] = Field(None, pattern=r"^\d+\.\d+\.\d+$")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"extra": "forbid"}

    @field_validator("max_supported")
    @classmethod
    def validate_version_range(cls, v: str, info) -> str:
        """Ensure max_supported >= min_supported."""
        min_ver_str = info.data.get("min_supported")
        if min_ver_str:
            min_ver = SchemaVersion.parse(min_ver_str)
            max_ver = SchemaVersion.parse(v)
            if max_ver < min_ver:
                raise ValueError(f"max_supported ({v}) must be >= min_supported ({min_ver_str})")
        return v

    def get_min_version(self) -> SchemaVersion:
        """Get min_supported as SchemaVersion."""
        return SchemaVersion.parse(self.min_supported)

    def get_max_version(self) -> SchemaVersion:
        """Get max_supported as SchemaVersion."""
        return SchemaVersion.parse(self.max_supported)

    def get_preferred_version(self) -> Optional[SchemaVersion]:
        """Get preferred as SchemaVersion if set."""
        if self.preferred:
            return SchemaVersion.parse(self.preferred)
        return None


@dataclass
class VersionNegotiationResult:
    """
    Result of version negotiation.

    Contains selected version or error information if negotiation failed.
    """

    status: NegotiationStatus
    selected_version: Optional[SchemaVersion] = None
    compatibility: VersionCompatibility = VersionCompatibility.COMPATIBLE
    error_message: Optional[str] = None
    agent_min: Optional[SchemaVersion] = None
    agent_max: Optional[SchemaVersion] = None
    cloud_min: Optional[SchemaVersion] = None
    cloud_max: Optional[SchemaVersion] = None
    negotiated_at: datetime = field(default_factory=datetime.utcnow)

    def is_success(self) -> bool:
        """Check if negotiation was successful."""
        return self.status == NegotiationStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "selected_version": str(self.selected_version) if self.selected_version else None,
            "compatibility": self.compatibility.value,
            "error_message": self.error_message,
            "agent_range": {
                "min": str(self.agent_min) if self.agent_min else None,
                "max": str(self.agent_max) if self.agent_max else None,
            },
            "cloud_range": {
                "min": str(self.cloud_min) if self.cloud_min else None,
                "max": str(self.cloud_max) if self.cloud_max else None,
            },
            "negotiated_at": self.negotiated_at.isoformat(),
        }


# ============================================================================
# Version Negotiator
# ============================================================================


class SchemaVersionNegotiator:
    """
    Schema version negotiator for Cloud-Agent protocol.

    Handles version negotiation between Cloud and Agent, ensuring
    both parties use a compatible schema version for communication.

    Usage:
        negotiator = SchemaVersionNegotiator(
            min_supported="1.0.0",
            max_supported="1.2.0",
        )

        result = negotiator.negotiate(agent_request)
        if result.is_success():
            use_version = result.selected_version
    """

    def __init__(
        self,
        min_supported: str = MIN_SUPPORTED_VERSION,
        max_supported: str = MAX_SUPPORTED_VERSION,
        prefer_latest: bool = True,
    ):
        """
        Initialize version negotiator.

        Args:
            min_supported: Minimum supported schema version
            max_supported: Maximum supported schema version
            prefer_latest: If True, prefer latest compatible version
        """
        self.min_supported = SchemaVersion.parse(min_supported)
        self.max_supported = SchemaVersion.parse(max_supported)
        self.prefer_latest = prefer_latest

        if self.max_supported < self.min_supported:
            raise ValueError(
                f"max_supported ({max_supported}) must be >= " f"min_supported ({min_supported})"
            )

        self._negotiation_cache: Dict[str, VersionNegotiationResult] = {}

    def negotiate(
        self,
        request: VersionNegotiationRequest,
    ) -> VersionNegotiationResult:
        """
        Negotiate schema version with agent.

        Args:
            request: Version negotiation request from agent

        Returns:
            VersionNegotiationResult with selected version or error
        """
        agent_min = request.get_min_version()
        agent_max = request.get_max_version()
        agent_preferred = request.get_preferred_version()

        # Check for overlap in version ranges
        overlap_min = max(agent_min, self.min_supported)
        overlap_max = min(agent_max, self.max_supported)

        if overlap_min > overlap_max:
            # No overlap - determine incompatibility reason
            if agent_max < self.min_supported:
                return VersionNegotiationResult(
                    status=NegotiationStatus.FAILED,
                    compatibility=VersionCompatibility.INCOMPATIBLE_TOO_OLD,
                    error_message=(
                        f"Agent max version {agent_max} is below "
                        f"Cloud min version {self.min_supported}"
                    ),
                    agent_min=agent_min,
                    agent_max=agent_max,
                    cloud_min=self.min_supported,
                    cloud_max=self.max_supported,
                )
            elif agent_min > self.max_supported:
                return VersionNegotiationResult(
                    status=NegotiationStatus.FAILED,
                    compatibility=VersionCompatibility.INCOMPATIBLE_TOO_NEW,
                    error_message=(
                        f"Agent min version {agent_min} is above "
                        f"Cloud max version {self.max_supported}"
                    ),
                    agent_min=agent_min,
                    agent_max=agent_max,
                    cloud_min=self.min_supported,
                    cloud_max=self.max_supported,
                )
            else:
                # Major version mismatch
                return VersionNegotiationResult(
                    status=NegotiationStatus.FAILED,
                    compatibility=VersionCompatibility.INCOMPATIBLE_MAJOR,
                    error_message=(
                        f"No compatible version found between "
                        f"Agent [{agent_min}-{agent_max}] and "
                        f"Cloud [{self.min_supported}-{self.max_supported}]"
                    ),
                    agent_min=agent_min,
                    agent_max=agent_max,
                    cloud_min=self.min_supported,
                    cloud_max=self.max_supported,
                )

        # Select version from overlap range
        if agent_preferred and overlap_min <= agent_preferred <= overlap_max:
            selected = agent_preferred
        elif self.prefer_latest:
            selected = overlap_max
        else:
            selected = overlap_min

        logger.info(
            "Version negotiation successful",
            extra={
                "agent_id": request.agent_id,
                "selected_version": str(selected),
                "agent_range": f"[{agent_min}-{agent_max}]",
                "cloud_range": f"[{self.min_supported}-{self.max_supported}]",
            },
        )

        result = VersionNegotiationResult(
            status=NegotiationStatus.SUCCESS,
            selected_version=selected,
            compatibility=VersionCompatibility.COMPATIBLE,
            agent_min=agent_min,
            agent_max=agent_max,
            cloud_min=self.min_supported,
            cloud_max=self.max_supported,
        )

        # Cache successful negotiation
        self._negotiation_cache[request.agent_id] = result

        return result

    def get_cached_negotiation(
        self,
        agent_id: str,
    ) -> Optional[VersionNegotiationResult]:
        """
        Get cached negotiation result for agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Cached result or None if not found
        """
        return self._negotiation_cache.get(agent_id)

    def clear_cache(self, agent_id: Optional[str] = None) -> None:
        """
        Clear negotiation cache.

        Args:
            agent_id: Specific agent to clear, or None for all
        """
        if agent_id:
            self._negotiation_cache.pop(agent_id, None)
        else:
            self._negotiation_cache.clear()

    def get_supported_range(self) -> Tuple[SchemaVersion, SchemaVersion]:
        """Get supported version range as tuple (min, max)."""
        return (self.min_supported, self.max_supported)


# ============================================================================
# Utility Functions
# ============================================================================


def check_version_compatibility(
    version: str,
    min_supported: str = MIN_SUPPORTED_VERSION,
    max_supported: str = MAX_SUPPORTED_VERSION,
) -> VersionCompatibility:
    """
    Check if a version is compatible with supported range.

    Args:
        version: Version string to check
        min_supported: Minimum supported version
        max_supported: Maximum supported version

    Returns:
        VersionCompatibility status
    """
    try:
        ver = SchemaVersion.parse(version)
        min_ver = SchemaVersion.parse(min_supported)
        max_ver = SchemaVersion.parse(max_supported)
    except ValueError:
        return VersionCompatibility.INVALID_VERSION

    if ver < min_ver:
        if ver.major < min_ver.major:
            return VersionCompatibility.INCOMPATIBLE_MAJOR
        return VersionCompatibility.INCOMPATIBLE_TOO_OLD

    if ver > max_ver:
        if ver.major > max_ver.major:
            return VersionCompatibility.INCOMPATIBLE_MAJOR
        return VersionCompatibility.INCOMPATIBLE_TOO_NEW

    return VersionCompatibility.COMPATIBLE


def negotiate_version(
    agent_min: str,
    agent_max: str,
    cloud_min: str = MIN_SUPPORTED_VERSION,
    cloud_max: str = MAX_SUPPORTED_VERSION,
    prefer_latest: bool = True,
) -> Optional[str]:
    """
    Simple version negotiation function.

    Args:
        agent_min: Agent's minimum supported version
        agent_max: Agent's maximum supported version
        cloud_min: Cloud's minimum supported version
        cloud_max: Cloud's maximum supported version
        prefer_latest: If True, select latest compatible version

    Returns:
        Selected version string, or None if incompatible
    """
    try:
        a_min = SchemaVersion.parse(agent_min)
        a_max = SchemaVersion.parse(agent_max)
        c_min = SchemaVersion.parse(cloud_min)
        c_max = SchemaVersion.parse(cloud_max)
    except ValueError:
        return None

    # Find overlap
    overlap_min = max(a_min, c_min)
    overlap_max = min(a_max, c_max)

    if overlap_min > overlap_max:
        return None

    selected = overlap_max if prefer_latest else overlap_min
    return str(selected)


def get_current_schema_version() -> SchemaVersion:
    """Get current schema version."""
    return SchemaVersion.parse(CURRENT_SCHEMA_VERSION)


def get_supported_version_range() -> Tuple[SchemaVersion, SchemaVersion]:
    """Get supported schema version range."""
    return (
        SchemaVersion.parse(MIN_SUPPORTED_VERSION),
        SchemaVersion.parse(MAX_SUPPORTED_VERSION),
    )
