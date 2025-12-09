# -*- coding: utf-8 -*-
"""
Feature Flag System for Enterprise Features.

DORA Phase 3 Block 3.8: Feature flag system for Enterprise

Provides controlled feature rollout capabilities:
- Tier-based feature access (Standard, Professional, Enterprise)
- Gradual rollout strategies
- A/B testing support
- Real-time feature toggling

DORA References:
    - Supports tiered service offerings per Art. 30(2)(e)
    - Enables controlled deployment per Art. 11 resilience requirements
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


class FeatureTier(Enum):
    """Service tier levels."""

    STANDARD = "standard"  # Basic tier
    PROFESSIONAL = "professional"  # Mid tier
    ENTERPRISE = "enterprise"  # Top tier
    INTERNAL = "internal"  # Internal testing only


class FeatureStatus(Enum):
    """Feature flag status."""

    DISABLED = "disabled"  # Feature is off
    ENABLED = "enabled"  # Feature is on for all eligible
    BETA = "beta"  # Beta testing phase
    DEPRECATED = "deprecated"  # Scheduled for removal


class RolloutStrategy(Enum):
    """Feature rollout strategies."""

    ALL_AT_ONCE = "all_at_once"  # Enable for everyone immediately
    PERCENTAGE = "percentage"  # Enable for X% of users
    ALLOWLIST = "allowlist"  # Enable only for specific clients
    CANARY = "canary"  # Gradual increase over time
    RING = "ring"  # Ring-based deployment


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class FeatureFlag:
    """Feature flag definition."""

    flag_id: str
    name: str
    description: str
    minimum_tier: FeatureTier
    status: FeatureStatus
    rollout_strategy: RolloutStrategy
    rollout_percentage: float = 100.0  # 0-100
    allowlist: list[str] = field(default_factory=list)  # Client IDs
    blocklist: list[str] = field(default_factory=list)  # Client IDs
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    dependencies: list[str] = field(default_factory=list)  # Other flag IDs

    def is_enabled_for_tier(self, tier: FeatureTier) -> bool:
        """Check if feature is enabled for a tier."""
        if self.status in (FeatureStatus.DISABLED, FeatureStatus.DEPRECATED):
            return False

        tier_order = [
            FeatureTier.STANDARD,
            FeatureTier.PROFESSIONAL,
            FeatureTier.ENTERPRISE,
            FeatureTier.INTERNAL,
        ]
        min_index = tier_order.index(self.minimum_tier)
        tier_index = tier_order.index(tier)
        return tier_index >= min_index


@dataclass
class FeatureGate:
    """Feature gate evaluation result."""

    flag_id: str
    client_id: str
    is_enabled: bool
    reason: str
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    tier: FeatureTier | None = None
    rollout_bucket: float | None = None  # 0-100


@dataclass
class ClientFeatureAccess:
    """Client's feature access configuration."""

    client_id: str
    tier: FeatureTier
    custom_features: list[str] = field(default_factory=list)  # Override enabled
    disabled_features: list[str] = field(default_factory=list)  # Override disabled
    beta_participant: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FeatureFlagConfig:
    """Feature flag service configuration."""

    cache_ttl_seconds: int = 60
    enable_analytics: bool = True
    default_tier: FeatureTier = FeatureTier.STANDARD


# =============================================================================
# Main Service Class
# =============================================================================


class FeatureFlagService:
    """
    Feature Flag Service.

    Provides controlled feature rollout for enterprise features.
    """

    def __init__(self, config: FeatureFlagConfig | None = None) -> None:
        """Initialize feature flag service."""
        self.config = config or FeatureFlagConfig()
        self._flags: dict[str, FeatureFlag] = {}
        self._client_access: dict[str, ClientFeatureAccess] = {}
        self._evaluation_log: list[FeatureGate] = []
        self._initialize_default_flags()

    def _initialize_default_flags(self) -> None:
        """Initialize default enterprise feature flags."""
        default_flags = [
            # Extended Reporting (Block 3.2)
            FeatureFlag(
                flag_id="extended_reporting",
                name="Extended Incident Reporting",
                description="PDF/JSON incident report generation",
                minimum_tier=FeatureTier.PROFESSIONAL,
                status=FeatureStatus.ENABLED,
                rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
            ),
            # Client Metrics (Block 3.3)
            FeatureFlag(
                flag_id="client_metrics",
                name="Per-Client Metrics Dashboard",
                description="Dedicated metrics and dashboards per client",
                minimum_tier=FeatureTier.PROFESSIONAL,
                status=FeatureStatus.ENABLED,
                rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
            ),
            # SIEM Integration (Block 3.4)
            FeatureFlag(
                flag_id="siem_integration",
                name="SIEM Integration",
                description="Splunk/ELK event export",
                minimum_tier=FeatureTier.ENTERPRISE,
                status=FeatureStatus.ENABLED,
                rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
            ),
            # TLPT Support (Block 3.5)
            FeatureFlag(
                flag_id="tlpt_support",
                name="TLPT Cooperation Support",
                description="Support for client TLPT engagements",
                minimum_tier=FeatureTier.ENTERPRISE,
                status=FeatureStatus.ENABLED,
                rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
            ),
            # Multi-Region (Block 3.9)
            FeatureFlag(
                flag_id="multi_region",
                name="Multi-Region Deployment",
                description="Multi-region deployment with failover",
                minimum_tier=FeatureTier.ENTERPRISE,
                status=FeatureStatus.ENABLED,
                rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
            ),
            # Dedicated Region (Block 3.13)
            FeatureFlag(
                flag_id="dedicated_region",
                name="Dedicated Region",
                description="Dedicated isolated region deployment",
                minimum_tier=FeatureTier.ENTERPRISE,
                status=FeatureStatus.ENABLED,
                rollout_strategy=RolloutStrategy.ALLOWLIST,
            ),
            # Pooled Audit (Block 3.12)
            FeatureFlag(
                flag_id="pooled_audit",
                name="Pooled Audit Coordination",
                description="Multi-client audit coordination",
                minimum_tier=FeatureTier.PROFESSIONAL,
                status=FeatureStatus.ENABLED,
                rollout_strategy=RolloutStrategy.ALL_AT_ONCE,
            ),
        ]

        for flag in default_flags:
            self._flags[flag.flag_id] = flag

    def create_flag(
        self,
        name: str,
        description: str,
        minimum_tier: FeatureTier,
        status: FeatureStatus = FeatureStatus.DISABLED,
        rollout_strategy: RolloutStrategy = RolloutStrategy.ALL_AT_ONCE,
        rollout_percentage: float = 100.0,
        created_by: str = "system",
    ) -> FeatureFlag:
        """Create a new feature flag."""
        flag = FeatureFlag(
            flag_id=str(uuid4()),
            name=name,
            description=description,
            minimum_tier=minimum_tier,
            status=status,
            rollout_strategy=rollout_strategy,
            rollout_percentage=rollout_percentage,
            created_by=created_by,
        )
        self._flags[flag.flag_id] = flag
        return flag

    def get_flag(self, flag_id: str) -> FeatureFlag | None:
        """Get flag by ID."""
        return self._flags.get(flag_id)

    def list_flags(
        self,
        status: FeatureStatus | None = None,
        tier: FeatureTier | None = None,
    ) -> list[FeatureFlag]:
        """List flags with optional filters."""
        flags = list(self._flags.values())

        if status:
            flags = [f for f in flags if f.status == status]
        if tier:
            flags = [f for f in flags if f.is_enabled_for_tier(tier)]

        return flags

    def update_flag(
        self,
        flag_id: str,
        status: FeatureStatus | None = None,
        rollout_percentage: float | None = None,
        allowlist: list[str] | None = None,
        blocklist: list[str] | None = None,
    ) -> FeatureFlag | None:
        """Update a feature flag."""
        flag = self._flags.get(flag_id)
        if not flag:
            return None

        if status is not None:
            flag.status = status
        if rollout_percentage is not None:
            flag.rollout_percentage = rollout_percentage
        if allowlist is not None:
            flag.allowlist = allowlist
        if blocklist is not None:
            flag.blocklist = blocklist

        flag.updated_at = datetime.utcnow()
        return flag

    def set_client_access(
        self,
        client_id: str,
        tier: FeatureTier,
        custom_features: list[str] | None = None,
        disabled_features: list[str] | None = None,
        beta_participant: bool = False,
    ) -> ClientFeatureAccess:
        """Set client feature access configuration."""
        access = ClientFeatureAccess(
            client_id=client_id,
            tier=tier,
            custom_features=custom_features or [],
            disabled_features=disabled_features or [],
            beta_participant=beta_participant,
        )
        self._client_access[client_id] = access
        return access

    def get_client_access(self, client_id: str) -> ClientFeatureAccess | None:
        """Get client access configuration."""
        return self._client_access.get(client_id)

    def evaluate(self, flag_id: str, client_id: str) -> FeatureGate:
        """Evaluate if a feature is enabled for a client."""
        flag = self._flags.get(flag_id)
        if not flag:
            return FeatureGate(
                flag_id=flag_id,
                client_id=client_id,
                is_enabled=False,
                reason="Flag not found",
            )

        access = self._client_access.get(client_id)
        tier = access.tier if access else self.config.default_tier

        # Check blocklist first
        if client_id in flag.blocklist:
            result = FeatureGate(
                flag_id=flag_id,
                client_id=client_id,
                is_enabled=False,
                reason="Client in blocklist",
                tier=tier,
            )
            self._log_evaluation(result)
            return result

        # Check client disabled features
        if access and flag_id in access.disabled_features:
            result = FeatureGate(
                flag_id=flag_id,
                client_id=client_id,
                is_enabled=False,
                reason="Disabled for client",
                tier=tier,
            )
            self._log_evaluation(result)
            return result

        # Check if feature is disabled globally
        if flag.status == FeatureStatus.DISABLED:
            result = FeatureGate(
                flag_id=flag_id,
                client_id=client_id,
                is_enabled=False,
                reason="Feature disabled",
                tier=tier,
            )
            self._log_evaluation(result)
            return result

        # Check if feature is deprecated
        if flag.status == FeatureStatus.DEPRECATED:
            result = FeatureGate(
                flag_id=flag_id,
                client_id=client_id,
                is_enabled=False,
                reason="Feature deprecated",
                tier=tier,
            )
            self._log_evaluation(result)
            return result

        # Check custom features override
        if access and flag_id in access.custom_features:
            result = FeatureGate(
                flag_id=flag_id,
                client_id=client_id,
                is_enabled=True,
                reason="Custom feature enabled",
                tier=tier,
            )
            self._log_evaluation(result)
            return result

        # Check allowlist for ALLOWLIST strategy
        if flag.rollout_strategy == RolloutStrategy.ALLOWLIST:
            is_allowed = client_id in flag.allowlist
            result = FeatureGate(
                flag_id=flag_id,
                client_id=client_id,
                is_enabled=is_allowed,
                reason="Allowlist check" if is_allowed else "Not in allowlist",
                tier=tier,
            )
            self._log_evaluation(result)
            return result

        # Check beta status
        if flag.status == FeatureStatus.BETA:
            if access and access.beta_participant:
                result = FeatureGate(
                    flag_id=flag_id,
                    client_id=client_id,
                    is_enabled=True,
                    reason="Beta participant",
                    tier=tier,
                )
                self._log_evaluation(result)
                return result
            else:
                result = FeatureGate(
                    flag_id=flag_id,
                    client_id=client_id,
                    is_enabled=False,
                    reason="Not a beta participant",
                    tier=tier,
                )
                self._log_evaluation(result)
                return result

        # Check tier requirement
        if not flag.is_enabled_for_tier(tier):
            result = FeatureGate(
                flag_id=flag_id,
                client_id=client_id,
                is_enabled=False,
                reason=f"Requires {flag.minimum_tier.value} tier",
                tier=tier,
            )
            self._log_evaluation(result)
            return result

        # Check percentage rollout
        if flag.rollout_strategy == RolloutStrategy.PERCENTAGE:
            bucket = hash(client_id) % 100
            is_enabled = bucket < flag.rollout_percentage
            result = FeatureGate(
                flag_id=flag_id,
                client_id=client_id,
                is_enabled=is_enabled,
                reason=f"Percentage rollout ({flag.rollout_percentage}%)",
                tier=tier,
                rollout_bucket=bucket,
            )
            self._log_evaluation(result)
            return result

        # Feature is enabled
        result = FeatureGate(
            flag_id=flag_id,
            client_id=client_id,
            is_enabled=True,
            reason="Feature enabled",
            tier=tier,
        )
        self._log_evaluation(result)
        return result

    def _log_evaluation(self, gate: FeatureGate) -> None:
        """Log feature evaluation for analytics."""
        if self.config.enable_analytics:
            self._evaluation_log.append(gate)

    def is_enabled(self, flag_id: str, client_id: str) -> bool:
        """Quick check if feature is enabled for client."""
        return self.evaluate(flag_id, client_id).is_enabled

    def get_client_features(self, client_id: str) -> dict[str, bool]:
        """Get all feature states for a client."""
        return {
            flag_id: self.is_enabled(flag_id, client_id)
            for flag_id in self._flags.keys()
        }

    def get_evaluation_stats(
        self,
        flag_id: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Get evaluation statistics."""
        logs = self._evaluation_log

        if flag_id:
            logs = [l for l in logs if l.flag_id == flag_id]
        if client_id:
            logs = [l for l in logs if l.client_id == client_id]

        enabled_count = sum(1 for l in logs if l.is_enabled)
        disabled_count = len(logs) - enabled_count

        return {
            "total_evaluations": len(logs),
            "enabled_count": enabled_count,
            "disabled_count": disabled_count,
            "enabled_rate": enabled_count / len(logs) * 100 if logs else 0,
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_feature_flag_service(
    default_tier: FeatureTier = FeatureTier.STANDARD,
    enable_analytics: bool = True,
    **kwargs: Any,
) -> FeatureFlagService:
    """Create feature flag service instance."""
    config = FeatureFlagConfig(
        default_tier=default_tier,
        enable_analytics=enable_analytics,
        **kwargs,
    )
    return FeatureFlagService(config)
