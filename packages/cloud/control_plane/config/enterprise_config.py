# -*- coding: utf-8 -*-
"""
Enterprise Configuration Handler.

CCEA Phase 10: Design Doc compliance for enterprise/on-prem/airgapped deployments.

This module reads environment variables from Helm/Docker configs and applies
the correct behavior in code. It bridges the gap between declarative config
(docker-compose.airgapped.yml, values-enterprise.yaml) and actual code behavior.

Design Doc Requirements:
- mTLS enforcement when configured
- Offline verification mode
- Telemetry local-only mode
- Registry mirror support
- Air-gapped signing mode

Environment Variables:
- CCEA_AIR_GAPPED_MODE: Enable air-gapped mode (no external connectivity)
- CCEA_OFFLINE_VERIFICATION: Enable offline signature verification
- CCEA_MTLS_ENABLED: Enable mTLS for all connections
- CCEA_TELEMETRY_LOCAL_ONLY: Keep telemetry local
- CCEA_TELEMETRY_EXTERNAL_EXPORT: Allow external telemetry export
- CCEA_SIGNING_OFFLINE_MODE: Offline signing without keyless
- CCEA_REGISTRY_URL: Local registry URL
- CCEA_REGISTRY_MIRROR_ENABLED: Use registry mirror
- CCEA_REGISTRY_MIRROR_URL: Registry mirror URL

CLOUD ZONE ONLY.
"""

from __future__ import annotations

import logging
import os
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# Environment Variable Names
# ============================================================================

class EnvVar(str, Enum):
    """Environment variable names for enterprise configuration."""
    # Mode flags
    AIR_GAPPED_MODE = "CCEA_AIR_GAPPED_MODE"
    ENTERPRISE_MODE = "CCEA_ENTERPRISE_MODE"
    ENV = "CCEA_ENV"

    # Verification
    OFFLINE_VERIFICATION = "CCEA_OFFLINE_VERIFICATION"
    SIGNATURE_OFFLINE_MODE = "CCEA_SIGNATURE_OFFLINE_MODE"
    SKIP_UPDATE_CHECK = "CCEA_SKIP_UPDATE_CHECK"
    TRUSTED_ROOT_PATH = "CCEA_TRUSTED_ROOT_PATH"

    # mTLS
    MTLS_ENABLED = "CCEA_MTLS_ENABLED"
    TLS_CERT_PATH = "CCEA_TLS_CERT_PATH"
    TLS_KEY_PATH = "CCEA_TLS_KEY_PATH"
    TLS_CA_PATH = "CCEA_TLS_CA_PATH"
    TLS_CLIENT_CA_PATH = "CCEA_TLS_CLIENT_CA_PATH"

    # Telemetry
    TELEMETRY_ENABLED = "CCEA_TELEMETRY_ENABLED"
    TELEMETRY_LOCAL_ONLY = "CCEA_TELEMETRY_LOCAL_ONLY"
    TELEMETRY_EXTERNAL_EXPORT = "CCEA_TELEMETRY_EXTERNAL_EXPORT"
    TELEMETRY_REDACTION_MANDATORY = "CCEA_TELEMETRY_REDACTION_MANDATORY"
    TELEMETRY_DEFAULT_LEVEL = "CCEA_TELEMETRY_DEFAULT_LEVEL"
    TELEMETRY_LOCAL_RETENTION_DAYS = "CCEA_TELEMETRY_LOCAL_RETENTION_DAYS"

    # Registry
    REGISTRY_URL = "CCEA_REGISTRY_URL"
    REGISTRY_MIRROR_ENABLED = "CCEA_REGISTRY_MIRROR_ENABLED"
    REGISTRY_MIRROR_URL = "CCEA_REGISTRY_MIRROR_URL"

    # Signing
    SIGNING_ENABLED = "CCEA_SIGNING_ENABLED"
    SIGNING_KEYLESS = "CCEA_SIGNING_KEYLESS"
    SIGNING_KEY_PATH = "CCEA_SIGNING_KEY_PATH"
    SIGNING_OFFLINE_MODE = "CCEA_SIGNING_OFFLINE_MODE"

    # SBOM
    SBOM_ENABLED = "CCEA_SBOM_ENABLED"
    SBOM_OFFLINE_MODE = "CCEA_SBOM_OFFLINE_MODE"
    SBOM_LOCAL_DB_PATH = "CCEA_SBOM_LOCAL_DB_PATH"
    SBOM_FORMAT = "CCEA_SBOM_FORMAT"

    # Data residency
    DATA_RESIDENCY = "CCEA_DATA_RESIDENCY"
    EVIDENCE_EXPORT_LOCAL_ONLY = "CCEA_EVIDENCE_EXPORT_LOCAL_ONLY"
    EVIDENCE_EXPORT_PATH = "CCEA_EVIDENCE_EXPORT_PATH"

    # Time sync
    TIME_SYNC_SERVERS = "CCEA_TIME_SYNC_SERVERS"


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class AirGappedConfig:
    """Air-gapped deployment configuration."""
    enabled: bool = False
    skip_update_check: bool = True
    offline_verification: bool = True
    local_registry_only: bool = True
    telemetry_local_only: bool = True
    evidence_local_only: bool = True
    signing_offline: bool = True
    sbom_offline: bool = True
    trusted_root_path: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "AirGappedConfig":
        """Load from environment variables."""
        enabled = _get_bool_env(EnvVar.AIR_GAPPED_MODE, False)

        if not enabled:
            return cls(enabled=False)

        return cls(
            enabled=True,
            skip_update_check=_get_bool_env(EnvVar.SKIP_UPDATE_CHECK, True),
            offline_verification=_get_bool_env(EnvVar.OFFLINE_VERIFICATION, True),
            local_registry_only=True,
            telemetry_local_only=_get_bool_env(EnvVar.TELEMETRY_LOCAL_ONLY, True),
            evidence_local_only=_get_bool_env(EnvVar.EVIDENCE_EXPORT_LOCAL_ONLY, True),
            signing_offline=_get_bool_env(EnvVar.SIGNING_OFFLINE_MODE, True),
            sbom_offline=_get_bool_env(EnvVar.SBOM_OFFLINE_MODE, True),
            trusted_root_path=_get_path_env(EnvVar.TRUSTED_ROOT_PATH),
        )


@dataclass
class MTLSConfig:
    """mTLS configuration."""
    enabled: bool = False
    cert_path: Optional[Path] = None
    key_path: Optional[Path] = None
    ca_path: Optional[Path] = None
    client_ca_path: Optional[Path] = None
    verify_mode: ssl.VerifyMode = ssl.CERT_REQUIRED

    @classmethod
    def from_env(cls) -> "MTLSConfig":
        """Load from environment variables."""
        enabled = _get_bool_env(EnvVar.MTLS_ENABLED, False)

        return cls(
            enabled=enabled,
            cert_path=_get_path_env(EnvVar.TLS_CERT_PATH),
            key_path=_get_path_env(EnvVar.TLS_KEY_PATH),
            ca_path=_get_path_env(EnvVar.TLS_CA_PATH),
            client_ca_path=_get_path_env(EnvVar.TLS_CLIENT_CA_PATH),
            verify_mode=ssl.CERT_REQUIRED if enabled else ssl.CERT_NONE,
        )

    def create_ssl_context(self, purpose: str = "server") -> Optional[ssl.SSLContext]:
        """Create SSL context for mTLS."""
        if not self.enabled:
            return None

        if purpose == "server":
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.verify_mode = self.verify_mode

            if self.cert_path and self.key_path:
                ctx.load_cert_chain(str(self.cert_path), str(self.key_path))

            if self.client_ca_path:
                ctx.load_verify_locations(str(self.client_ca_path))

        else:  # client
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.check_hostname = True

            if self.cert_path and self.key_path:
                ctx.load_cert_chain(str(self.cert_path), str(self.key_path))

            if self.ca_path:
                ctx.load_verify_locations(str(self.ca_path))

        return ctx


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""
    enabled: bool = True
    local_only: bool = False
    external_export: bool = True
    redaction_mandatory: bool = True
    default_level: str = "aggregated"
    local_retention_days: int = 90

    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        """Load from environment variables."""
        return cls(
            enabled=_get_bool_env(EnvVar.TELEMETRY_ENABLED, True),
            local_only=_get_bool_env(EnvVar.TELEMETRY_LOCAL_ONLY, False),
            external_export=_get_bool_env(EnvVar.TELEMETRY_EXTERNAL_EXPORT, True),
            redaction_mandatory=_get_bool_env(EnvVar.TELEMETRY_REDACTION_MANDATORY, True),
            default_level=os.getenv(EnvVar.TELEMETRY_DEFAULT_LEVEL.value, "aggregated"),
            local_retention_days=_get_int_env(EnvVar.TELEMETRY_LOCAL_RETENTION_DAYS, 90),
        )


@dataclass
class RegistryConfig:
    """Artifact registry configuration."""
    url: str = "https://registry.ccea.io"
    mirror_enabled: bool = False
    mirror_url: Optional[str] = None
    use_local_only: bool = False

    @classmethod
    def from_env(cls) -> "RegistryConfig":
        """Load from environment variables."""
        return cls(
            url=os.getenv(EnvVar.REGISTRY_URL.value, "https://registry.ccea.io"),
            mirror_enabled=_get_bool_env(EnvVar.REGISTRY_MIRROR_ENABLED, False),
            mirror_url=os.getenv(EnvVar.REGISTRY_MIRROR_URL.value),
            use_local_only=_get_bool_env(EnvVar.AIR_GAPPED_MODE, False),
        )


@dataclass
class SigningConfig:
    """Signing configuration."""
    enabled: bool = True
    keyless: bool = True
    offline_mode: bool = False
    key_path: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "SigningConfig":
        """Load from environment variables."""
        return cls(
            enabled=_get_bool_env(EnvVar.SIGNING_ENABLED, True),
            keyless=_get_bool_env(EnvVar.SIGNING_KEYLESS, True),
            offline_mode=_get_bool_env(EnvVar.SIGNING_OFFLINE_MODE, False),
            key_path=_get_path_env(EnvVar.SIGNING_KEY_PATH),
        )


@dataclass
class SBOMConfig:
    """SBOM configuration."""
    enabled: bool = True
    offline_mode: bool = False
    local_db_path: Optional[Path] = None
    format: str = "cyclonedx"

    @classmethod
    def from_env(cls) -> "SBOMConfig":
        """Load from environment variables."""
        return cls(
            enabled=_get_bool_env(EnvVar.SBOM_ENABLED, True),
            offline_mode=_get_bool_env(EnvVar.SBOM_OFFLINE_MODE, False),
            local_db_path=_get_path_env(EnvVar.SBOM_LOCAL_DB_PATH),
            format=os.getenv(EnvVar.SBOM_FORMAT.value, "cyclonedx"),
        )


# ============================================================================
# Main Enterprise Configuration
# ============================================================================

@dataclass
class EnterpriseConfig:
    """
    Complete enterprise configuration.

    Loads all configuration from environment variables and provides
    a unified interface for checking enterprise features.

    Usage:
        config = EnterpriseConfig.load()

        if config.is_air_gapped:
            # Use offline verification
            pass

        if config.mtls.enabled:
            # Create mTLS context
            ctx = config.mtls.create_ssl_context()
    """
    # Mode
    environment: str = "development"
    enterprise_mode: bool = False

    # Sub-configs
    airgapped: AirGappedConfig = field(default_factory=AirGappedConfig)
    mtls: MTLSConfig = field(default_factory=MTLSConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    signing: SigningConfig = field(default_factory=SigningConfig)
    sbom: SBOMConfig = field(default_factory=SBOMConfig)

    # Data residency
    data_residency: str = "cloud"
    evidence_export_local_only: bool = False
    evidence_export_path: Optional[Path] = None

    # Time sync
    time_sync_servers: List[str] = field(default_factory=list)

    # Load timestamp
    loaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Apply cascading overrides after initialization."""
        self._apply_airgapped_overrides()

    def _apply_airgapped_overrides(self):
        """Apply air-gapped mode cascading overrides."""
        if self.airgapped.enabled:
            self.telemetry.local_only = True
            self.telemetry.external_export = False
            self.registry.use_local_only = True
            self.signing.offline_mode = True
            self.signing.keyless = False
            self.sbom.offline_mode = True
            self.evidence_export_local_only = True

    @classmethod
    def load(cls) -> "EnterpriseConfig":
        """
        Load enterprise configuration from environment.

        This should be called once at startup and cached.
        """
        airgapped = AirGappedConfig.from_env()

        config = cls(
            environment=os.getenv(EnvVar.ENV.value, "development"),
            enterprise_mode=_get_bool_env(EnvVar.ENTERPRISE_MODE, False) or airgapped.enabled,
            airgapped=airgapped,
            mtls=MTLSConfig.from_env(),
            telemetry=TelemetryConfig.from_env(),
            registry=RegistryConfig.from_env(),
            signing=SigningConfig.from_env(),
            sbom=SBOMConfig.from_env(),
            data_residency=os.getenv(EnvVar.DATA_RESIDENCY.value, "cloud"),
            evidence_export_local_only=_get_bool_env(EnvVar.EVIDENCE_EXPORT_LOCAL_ONLY, False),
            evidence_export_path=_get_path_env(EnvVar.EVIDENCE_EXPORT_PATH),
            time_sync_servers=_get_list_env(EnvVar.TIME_SYNC_SERVERS),
        )

        # Note: air-gapped overrides are applied in __post_init__

        logger.info(
            f"Enterprise config loaded: env={config.environment}, "
            f"enterprise={config.enterprise_mode}, airgapped={config.is_air_gapped}, "
            f"mtls={config.mtls.enabled}"
        )

        return config

    @property
    def is_air_gapped(self) -> bool:
        """Check if running in air-gapped mode."""
        return self.airgapped.enabled

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

    @property
    def requires_offline_verification(self) -> bool:
        """Check if offline verification is required."""
        return self.airgapped.offline_verification

    @property
    def allows_external_connectivity(self) -> bool:
        """Check if external connectivity is allowed."""
        return not self.is_air_gapped

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # mTLS validation
        if self.mtls.enabled:
            if not self.mtls.cert_path or not self.mtls.cert_path.exists():
                errors.append("mTLS enabled but cert_path not found")
            if not self.mtls.key_path or not self.mtls.key_path.exists():
                errors.append("mTLS enabled but key_path not found")

        # Air-gapped validation
        if self.is_air_gapped:
            if not self.registry.url:
                errors.append("Air-gapped mode requires local REGISTRY_URL")
            if self.signing.keyless:
                errors.append("Keyless signing not available in air-gapped mode")

        # Offline signing validation
        if self.signing.offline_mode:
            if not self.signing.key_path or not self.signing.key_path.exists():
                errors.append("Offline signing requires SIGNING_KEY_PATH")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "environment": self.environment,
            "enterprise_mode": self.enterprise_mode,
            "is_air_gapped": self.is_air_gapped,
            "mtls_enabled": self.mtls.enabled,
            "telemetry_local_only": self.telemetry.local_only,
            "registry_url": self.registry.url,
            "signing_offline": self.signing.offline_mode,
            "data_residency": self.data_residency,
            "loaded_at": self.loaded_at.isoformat(),
        }


# ============================================================================
# Helper Functions
# ============================================================================

def _get_bool_env(var: EnvVar, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(var.value, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    if value in ("false", "0", "no", "off"):
        return False
    return default


def _get_int_env(var: EnvVar, default: int = 0) -> int:
    """Get integer environment variable."""
    value = os.getenv(var.value)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return default


def _get_path_env(var: EnvVar) -> Optional[Path]:
    """Get path environment variable."""
    value = os.getenv(var.value)
    if value:
        return Path(value)
    return None


def _get_list_env(var: EnvVar, separator: str = ",") -> List[str]:
    """Get list environment variable."""
    value = os.getenv(var.value, "")
    if value:
        return [item.strip() for item in value.split(separator) if item.strip()]
    return []


# ============================================================================
# Global Configuration Singleton
# ============================================================================

_enterprise_config: Optional[EnterpriseConfig] = None


def get_enterprise_config() -> EnterpriseConfig:
    """
    Get enterprise configuration singleton.

    Configuration is loaded once and cached. Call reload_enterprise_config()
    to force reload.
    """
    global _enterprise_config
    if _enterprise_config is None:
        _enterprise_config = EnterpriseConfig.load()
    return _enterprise_config


def reload_enterprise_config() -> EnterpriseConfig:
    """Force reload enterprise configuration."""
    global _enterprise_config
    _enterprise_config = EnterpriseConfig.load()
    return _enterprise_config


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "EnvVar",
    "AirGappedConfig",
    "MTLSConfig",
    "TelemetryConfig",
    "RegistryConfig",
    "SigningConfig",
    "SBOMConfig",
    "EnterpriseConfig",
    "get_enterprise_config",
    "reload_enterprise_config",
]
