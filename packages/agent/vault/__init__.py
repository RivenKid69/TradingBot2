# -*- coding: utf-8 -*-
"""
CCEA Agent Local Vault.

Secure storage for broker credentials and sensitive data.
Credentials NEVER leave the agent - Cloud has NO access.

Key Features:
- Encrypted storage (AES-256-GCM)
- OS keychain integration (preferred)
- Master key from environment or keychain
- Rotation support
- No cloud backup (by design)

Security Design Commitments:
- Credentials encrypted at rest
- Master key never in memory longer than needed
- No logging of credential values
- No telemetry containing credentials
"""

from typing import Final

ZONE: Final[str] = "agent"

from .local_vault import (
    LocalVault,
    VaultConfig,
    CredentialEntry,
    VaultError,
    CredentialNotFoundError,
    VaultLockedError,
)

from .credential_manager import (
    CredentialManager,
    BrokerCredential,
    CredentialType,
)

__all__ = [
    # Vault
    "LocalVault",
    "VaultConfig",
    "CredentialEntry",
    "VaultError",
    "CredentialNotFoundError",
    "VaultLockedError",
    # Credential Manager
    "CredentialManager",
    "BrokerCredential",
    "CredentialType",
]
