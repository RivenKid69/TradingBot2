"""
Security Services Module.

Provides secure credential storage, audit logging, and access control.

Components:
    - CredentialVault: AES-256-GCM encrypted credential storage
    - CredentialAuditLogger: Audit trail for credential access

References:
    - NIST SP 800-57 Part 1 Rev 5: Key Management
    - OWASP Cryptographic Storage Cheat Sheet
    - ISO 27001 A.12.4: Logging and monitoring
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
]
