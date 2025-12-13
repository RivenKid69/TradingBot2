# -*- coding: utf-8 -*-
"""
Credential Manager - High-level credential management.

Provides broker-specific credential handling on top of LocalVault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Final, List, Optional

from .local_vault import LocalVault, VaultConfig, CredentialNotFoundError


class CredentialType(str, Enum):
    """Types of broker credentials."""

    API_KEY = "api_key"
    API_SECRET = "api_secret"
    PASSPHRASE = "passphrase"
    OAUTH_TOKEN = "oauth_token"
    OAUTH_REFRESH = "oauth_refresh"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"


# Required credentials per broker
BROKER_CREDENTIALS: Final[Dict[str, List[CredentialType]]] = {
    "binance": [CredentialType.API_KEY, CredentialType.API_SECRET],
    "alpaca": [CredentialType.API_KEY, CredentialType.API_SECRET],
    "oanda": [CredentialType.API_KEY],
    "ib": [CredentialType.API_KEY],  # Simplified for demo
    "deribit": [CredentialType.API_KEY, CredentialType.API_SECRET],
}


@dataclass
class BrokerCredential:
    """
    Complete broker credential set.

    Contains all credentials needed for a broker connection.
    Values are retrieved from vault on demand.
    """

    broker: str
    account_id: Optional[str] = None
    is_paper: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # These are populated from vault
    _api_key: Optional[str] = field(default=None, repr=False)
    _api_secret: Optional[str] = field(default=None, repr=False)
    _passphrase: Optional[str] = field(default=None, repr=False)

    @property
    def api_key(self) -> Optional[str]:
        """Get API key (must be loaded from vault first)."""
        return self._api_key

    @property
    def api_secret(self) -> Optional[str]:
        """Get API secret (must be loaded from vault first)."""
        return self._api_secret

    @property
    def passphrase(self) -> Optional[str]:
        """Get passphrase (must be loaded from vault first)."""
        return self._passphrase

    def is_complete(self) -> bool:
        """Check if all required credentials are present."""
        required = BROKER_CREDENTIALS.get(self.broker, [])
        for cred_type in required:
            if cred_type == CredentialType.API_KEY and not self._api_key:
                return False
            if cred_type == CredentialType.API_SECRET and not self._api_secret:
                return False
            if cred_type == CredentialType.PASSPHRASE and not self._passphrase:
                return False
        return True

    def clear(self) -> None:
        """Clear credential values from memory."""
        self._api_key = None
        self._api_secret = None
        self._passphrase = None


class CredentialManager:
    """
    High-level credential management.

    Wraps LocalVault with broker-specific logic.

    Usage:
        manager = CredentialManager()
        manager.unlock("master_password")

        # Store credentials
        manager.store_broker_credentials("binance", api_key="...", api_secret="...")

        # Get credentials
        creds = manager.get_broker_credentials("binance")
        print(creds.api_key)
    """

    def __init__(self, vault: Optional[LocalVault] = None):
        """Initialize credential manager."""
        self._vault = vault or LocalVault()

    @property
    def is_locked(self) -> bool:
        """Check if vault is locked."""
        return self._vault.is_locked

    def initialize(self, master_password: str) -> None:
        """Initialize vault with master password."""
        self._vault.initialize(master_password)

    def unlock(self, master_password: str) -> None:
        """Unlock vault."""
        self._vault.unlock(master_password)

    def lock(self) -> None:
        """Lock vault."""
        self._vault.lock()

    def store_broker_credentials(
        self,
        broker: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        account_id: Optional[str] = None,
        is_paper: bool = False,
        **kwargs,
    ) -> None:
        """
        Store credentials for a broker.

        Args:
            broker: Broker identifier
            api_key: API key
            api_secret: API secret
            passphrase: Passphrase (if required)
            account_id: Account ID
            is_paper: Whether this is paper trading account
            **kwargs: Additional credentials
        """
        metadata = {
            "account_id": account_id,
            "is_paper": is_paper,
        }

        if api_key:
            self._vault.store(
                broker=broker,
                credential_type=CredentialType.API_KEY.value,
                value=api_key,
                metadata=metadata,
            )

        if api_secret:
            self._vault.store(
                broker=broker,
                credential_type=CredentialType.API_SECRET.value,
                value=api_secret,
                metadata=metadata,
            )

        if passphrase:
            self._vault.store(
                broker=broker,
                credential_type=CredentialType.PASSPHRASE.value,
                value=passphrase,
                metadata=metadata,
            )

        # Store additional credentials
        for key, value in kwargs.items():
            if value:
                self._vault.store(
                    broker=broker,
                    credential_type=key,
                    value=value,
                    metadata=metadata,
                )

    def get_broker_credentials(
        self,
        broker: str,
        account_id: Optional[str] = None,
    ) -> BrokerCredential:
        """
        Get credentials for a broker.

        Args:
            broker: Broker identifier
            account_id: Optional account ID filter

        Returns:
            BrokerCredential with values populated
        """
        cred = BrokerCredential(broker=broker, account_id=account_id)

        # Load credentials from vault
        try:
            cred._api_key = self._vault.retrieve(broker, CredentialType.API_KEY.value)
        except CredentialNotFoundError:
            pass

        try:
            cred._api_secret = self._vault.retrieve(broker, CredentialType.API_SECRET.value)
        except CredentialNotFoundError:
            pass

        try:
            cred._passphrase = self._vault.retrieve(broker, CredentialType.PASSPHRASE.value)
        except CredentialNotFoundError:
            pass

        # Load metadata
        creds_list = self._vault.list_credentials(broker=broker)
        if creds_list:
            metadata = creds_list[0].get("metadata", {})
            cred.is_paper = metadata.get("is_paper", False)
            if account_id is None:
                cred.account_id = metadata.get("account_id")

        return cred

    def delete_broker_credentials(self, broker: str) -> bool:
        """
        Delete all credentials for a broker.

        Args:
            broker: Broker identifier

        Returns:
            True if any credentials deleted
        """
        deleted = False
        for cred_type in CredentialType:
            if self._vault.delete(broker, cred_type.value):
                deleted = True
        return deleted

    def list_brokers(self) -> List[str]:
        """
        List brokers with stored credentials.

        Returns:
            List of broker identifiers
        """
        creds = self._vault.list_credentials()
        brokers = set(c["broker"] for c in creds)
        return sorted(brokers)

    def has_credentials(self, broker: str) -> bool:
        """
        Check if broker has credentials stored.

        Args:
            broker: Broker identifier

        Returns:
            True if credentials exist
        """
        creds = self._vault.list_credentials(broker=broker)
        return len(creds) > 0

    def validate_broker_credentials(self, broker: str) -> tuple[bool, List[str]]:
        """
        Validate broker has all required credentials.

        Args:
            broker: Broker identifier

        Returns:
            Tuple of (is_valid, missing_credentials)
        """
        required = BROKER_CREDENTIALS.get(broker, [])
        missing = []

        for cred_type in required:
            try:
                self._vault.retrieve(broker, cred_type.value)
            except CredentialNotFoundError:
                missing.append(cred_type.value)

        return len(missing) == 0, missing

    def rotate_master_password(self, old_password: str, new_password: str) -> None:
        """
        Rotate vault master password.

        Args:
            old_password: Current password
            new_password: New password
        """
        self._vault.rotate_master_key(old_password, new_password)
