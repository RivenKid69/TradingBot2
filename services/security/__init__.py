"""
Security Services Module.

Provides secure credential storage, audit logging, access control, and geo-blocking.

Components:
    - CredentialVault: AES-256-GCM encrypted credential storage
    - CredentialAuditLogger: Audit trail for credential access
    - GeoBlockingService: Geographic access restrictions for sanctions compliance

References:
    - NIST SP 800-57 Part 1 Rev 5: Key Management
    - OWASP Cryptographic Storage Cheat Sheet
    - ISO 27001 A.12.4: Logging and monitoring
    - OFAC/EU Sanctions Programs
"""

from services.security.credential_vault import (
    CredentialVault,
    CredentialType,
    EncryptedCredential,
    CredentialVaultError,
    InvalidMasterKeyError,
    CredentialDecryptionError,
)

from services.security.credential_audit import (
    CredentialAuditLogger,
    CredentialAccessType,
    CredentialAccessEvent,
    InMemoryAuditStorage,
    AnomalyAlert,
)

from services.security.geo_blocking import (
    GeoBlockingService,
    GeoCheckResult,
    BlockReason,
    Country,
    MockGeoIPProvider,
    BLOCKED_COUNTRIES,
    HIGH_RISK_COUNTRIES,
)

__all__ = [
    # Vault
    "CredentialVault",
    "CredentialType",
    "EncryptedCredential",
    "CredentialVaultError",
    "InvalidMasterKeyError",
    "CredentialDecryptionError",
    # Audit
    "CredentialAuditLogger",
    "CredentialAccessType",
    "CredentialAccessEvent",
    "InMemoryAuditStorage",
    "AnomalyAlert",
    # Geo-blocking
    "GeoBlockingService",
    "GeoCheckResult",
    "BlockReason",
    "Country",
    "MockGeoIPProvider",
    "BLOCKED_COUNTRIES",
    "HIGH_RISK_COUNTRIES",
]
