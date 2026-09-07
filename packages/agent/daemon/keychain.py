# -*- coding: utf-8 -*-
"""
Keychain Manager - OS keychain integration for master key.

Design Doc Phase 5:
- Зафиксировать источник master key (предпочтительно OS keychain; fallback: encrypted file + env var)
- Cloud никогда не получает ни master key, ни "backup"

CCEA Phase 5 Component.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import secrets
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, Optional


# Constants
SERVICE_NAME: Final[str] = "ccea-agent"
ACCOUNT_NAME: Final[str] = "vault-master-key"
KEY_SIZE: Final[int] = 32  # 256 bits


class KeychainError(Exception):
    """Keychain operation error."""

    pass


class KeychainNotAvailableError(KeychainError):
    """OS keychain not available."""

    pass


class KeyNotFoundError(KeychainError):
    """Key not found in keychain."""

    pass


@dataclass
class KeychainConfig:
    """
    Keychain configuration.
    """

    # Service identification
    service_name: str = SERVICE_NAME
    account_name: str = ACCOUNT_NAME

    # Fallback options
    use_keychain: bool = True  # Prefer OS keychain
    fallback_to_env: bool = True  # Fallback to environment variable
    fallback_to_file: bool = True  # Fallback to encrypted file
    env_var_name: str = "CCEA_VAULT_KEY"
    key_file_path: Path = field(default_factory=lambda: Path.home() / ".ccea" / "vault.key")

    # Security
    require_keychain_for_live: bool = False  # Require keychain for live trading
    allow_key_generation: bool = True  # Auto-generate key if not found

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "account_name": self.account_name,
            "use_keychain": self.use_keychain,
            "fallback_to_env": self.fallback_to_env,
            "fallback_to_file": self.fallback_to_file,
            "env_var_name": self.env_var_name,
            "key_file_path": str(self.key_file_path),
        }


class KeychainManager:
    """
    Manages master key storage with OS keychain support.

    Design Doc Phase 5:
    - Prefers OS keychain for secure storage
    - Fallback to environment variable or encrypted file
    - Cloud NEVER has access to master key

    Supported platforms:
    - macOS: Keychain Access
    - Linux: Secret Service (GNOME Keyring, KWallet)
    - Windows: Credential Manager

    Usage:
        manager = KeychainManager()

        # Get or generate master key
        key = manager.get_master_key()

        # Store new key
        manager.store_master_key(new_key)

        # Rotate key
        manager.rotate_master_key()
    """

    def __init__(self, config: Optional[KeychainConfig] = None):
        """
        Initialize keychain manager.

        Args:
            config: Configuration
        """
        self.config = config or KeychainConfig()
        self._platform = platform.system()
        self._keychain_available: Optional[bool] = None

    @property
    def is_keychain_available(self) -> bool:
        """Check if OS keychain is available."""
        if self._keychain_available is not None:
            return self._keychain_available

        self._keychain_available = self._check_keychain_available()
        return self._keychain_available

    def _check_keychain_available(self) -> bool:
        """Check if keychain is available on this platform."""
        if self._platform == "Darwin":
            # macOS - check security command
            try:
                subprocess.run(
                    ["security", "help"],
                    capture_output=True,
                    timeout=5,
                )
                return True
            except Exception:
                return False

        elif self._platform == "Linux":
            # Linux - check for secret-tool (Secret Service)
            try:
                result = subprocess.run(
                    ["which", "secret-tool"],
                    capture_output=True,
                    timeout=5,
                )
                return result.returncode == 0
            except Exception:
                return False

        elif self._platform == "Windows":
            # Windows - always available via pywin32 or cmdkey
            try:
                subprocess.run(
                    ["cmdkey", "/list"],
                    capture_output=True,
                    timeout=5,
                )
                return True
            except Exception:
                return False

        return False

    def get_master_key(self) -> bytes:
        """
        Get master key from most secure available source.

        Priority:
        1. OS Keychain
        2. Environment variable
        3. Key file
        4. Generate new (if allowed)

        Returns:
            Master key bytes
        """
        # Try keychain first
        if self.config.use_keychain and self.is_keychain_available:
            try:
                key = self._get_from_keychain()
                if key:
                    return key
            except KeyNotFoundError:
                pass
            except Exception:
                pass  # Fall through to other methods

        # Try environment variable
        if self.config.fallback_to_env:
            key = self._get_from_env()
            if key:
                return key

        # Try key file
        if self.config.fallback_to_file:
            key = self._get_from_file()
            if key:
                return key

        # Generate new key if allowed
        if self.config.allow_key_generation:
            key = self._generate_and_store_key()
            return key

        raise KeyNotFoundError("Master key not found in keychain, environment, or file")

    def store_master_key(self, key: bytes) -> bool:
        """
        Store master key in most secure available storage.

        Args:
            key: Master key to store

        Returns:
            True if stored successfully
        """
        if len(key) != KEY_SIZE:
            raise ValueError(f"Key must be {KEY_SIZE} bytes")

        stored = False

        # Try keychain first
        if self.config.use_keychain and self.is_keychain_available:
            try:
                self._store_in_keychain(key)
                stored = True
            except Exception:
                pass

        # Also store in file as backup (if enabled)
        if self.config.fallback_to_file:
            try:
                self._store_in_file(key)
                if not stored:
                    stored = True
            except Exception:
                pass

        return stored

    def delete_master_key(self) -> bool:
        """
        Delete master key from all storage locations.

        Returns:
            True if deleted from at least one location
        """
        deleted = False

        # Delete from keychain
        if self.is_keychain_available:
            try:
                self._delete_from_keychain()
                deleted = True
            except Exception:
                pass

        # Delete key file
        if self.config.key_file_path.exists():
            try:
                self.config.key_file_path.unlink()
                deleted = True
            except Exception:
                pass

        return deleted

    def rotate_master_key(self) -> bytes:
        """
        Rotate master key.

        Generates new key and stores it.

        Returns:
            New master key
        """
        new_key = secrets.token_bytes(KEY_SIZE)
        self.store_master_key(new_key)
        return new_key

    def get_key_info(self) -> Dict[str, Any]:
        """
        Get information about key storage (without exposing key).

        Returns:
            Key storage info
        """
        info = {
            "platform": self._platform,
            "keychain_available": self.is_keychain_available,
            "keychain_enabled": self.config.use_keychain,
            "env_fallback_enabled": self.config.fallback_to_env,
            "file_fallback_enabled": self.config.fallback_to_file,
            "key_file_exists": self.config.key_file_path.exists(),
            "env_var_set": bool(os.environ.get(self.config.env_var_name)),
        }

        # Check keychain
        if self.is_keychain_available:
            try:
                key = self._get_from_keychain()
                info["keychain_has_key"] = key is not None
            except Exception:
                info["keychain_has_key"] = False
        else:
            info["keychain_has_key"] = False

        return info

    # ===== Platform-Specific Implementations =====

    def _get_from_keychain(self) -> Optional[bytes]:
        """Get key from OS keychain."""
        if self._platform == "Darwin":
            return self._get_from_macos_keychain()
        elif self._platform == "Linux":
            return self._get_from_linux_keychain()
        elif self._platform == "Windows":
            return self._get_from_windows_keychain()
        return None

    def _store_in_keychain(self, key: bytes) -> None:
        """Store key in OS keychain."""
        if self._platform == "Darwin":
            self._store_in_macos_keychain(key)
        elif self._platform == "Linux":
            self._store_in_linux_keychain(key)
        elif self._platform == "Windows":
            self._store_in_windows_keychain(key)

    def _delete_from_keychain(self) -> None:
        """Delete key from OS keychain."""
        if self._platform == "Darwin":
            self._delete_from_macos_keychain()
        elif self._platform == "Linux":
            self._delete_from_linux_keychain()
        elif self._platform == "Windows":
            self._delete_from_windows_keychain()

    # ===== macOS Keychain =====

    def _get_from_macos_keychain(self) -> Optional[bytes]:
        """Get key from macOS Keychain."""
        try:
            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    self.config.service_name,
                    "-a",
                    self.config.account_name,
                    "-w",  # Output password only
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                # Key is base64 encoded
                return base64.b64decode(result.stdout.strip())
            elif "could not be found" in result.stderr:
                raise KeyNotFoundError("Key not found in macOS Keychain")
            else:
                raise KeychainError(f"Keychain error: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise KeychainError("Keychain operation timed out")

    def _store_in_macos_keychain(self, key: bytes) -> None:
        """Store key in macOS Keychain."""
        # First try to delete existing
        try:
            self._delete_from_macos_keychain()
        except Exception:
            pass

        # Store as base64
        key_b64 = base64.b64encode(key).decode()

        result = subprocess.run(
            [
                "security",
                "add-generic-password",
                "-s",
                self.config.service_name,
                "-a",
                self.config.account_name,
                "-w",
                key_b64,
                "-U",  # Update if exists
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise KeychainError(f"Failed to store in Keychain: {result.stderr}")

    def _delete_from_macos_keychain(self) -> None:
        """Delete key from macOS Keychain."""
        subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-s",
                self.config.service_name,
                "-a",
                self.config.account_name,
            ],
            capture_output=True,
            timeout=10,
        )

    # ===== Linux Secret Service =====

    def _get_from_linux_keychain(self) -> Optional[bytes]:
        """Get key from Linux Secret Service."""
        try:
            result = subprocess.run(
                [
                    "secret-tool",
                    "lookup",
                    "service",
                    self.config.service_name,
                    "account",
                    self.config.account_name,
                ],
                capture_output=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout:
                return base64.b64decode(result.stdout.strip())
            else:
                raise KeyNotFoundError("Key not found in Secret Service")

        except subprocess.TimeoutExpired:
            raise KeychainError("Secret Service operation timed out")

    def _store_in_linux_keychain(self, key: bytes) -> None:
        """Store key in Linux Secret Service."""
        key_b64 = base64.b64encode(key).decode()

        result = subprocess.run(
            [
                "secret-tool",
                "store",
                "--label",
                f"{self.config.service_name} - {self.config.account_name}",
                "service",
                self.config.service_name,
                "account",
                self.config.account_name,
            ],
            input=key_b64.encode(),
            capture_output=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise KeychainError(f"Failed to store in Secret Service: {result.stderr}")

    def _delete_from_linux_keychain(self) -> None:
        """Delete key from Linux Secret Service."""
        subprocess.run(
            [
                "secret-tool",
                "clear",
                "service",
                self.config.service_name,
                "account",
                self.config.account_name,
            ],
            capture_output=True,
            timeout=10,
        )

    # ===== Windows Credential Manager =====

    def _get_from_windows_keychain(self) -> Optional[bytes]:
        """Get key from Windows Credential Manager."""
        try:
            # Try using cmdkey to list and then use powershell to read
            target = f"{self.config.service_name}:{self.config.account_name}"

            # Use PowerShell to read credential
            ps_script = f"""
                $cred = Get-StoredCredential -Target "{target}"
                if ($cred) {{
                    [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
                        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($cred.Password)
                    )
                }}
            """

            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0 and result.stdout.strip():
                return base64.b64decode(result.stdout.strip())
            else:
                raise KeyNotFoundError("Key not found in Credential Manager")

        except subprocess.TimeoutExpired:
            raise KeychainError("Credential Manager operation timed out")
        except Exception as e:
            # Fallback: try with cmdkey
            raise KeyNotFoundError(f"Could not read from Credential Manager: {e}")

    def _store_in_windows_keychain(self, key: bytes) -> None:
        """Store key in Windows Credential Manager."""
        target = f"{self.config.service_name}:{self.config.account_name}"
        key_b64 = base64.b64encode(key).decode()

        # Use cmdkey
        result = subprocess.run(
            ["cmdkey", "/add:" + target, "/user:" + self.config.account_name, "/pass:" + key_b64],
            capture_output=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise KeychainError(f"Failed to store in Credential Manager")

    def _delete_from_windows_keychain(self) -> None:
        """Delete key from Windows Credential Manager."""
        target = f"{self.config.service_name}:{self.config.account_name}"
        subprocess.run(
            ["cmdkey", "/delete:" + target],
            capture_output=True,
            timeout=10,
        )

    # ===== Fallback Methods =====

    def _get_from_env(self) -> Optional[bytes]:
        """Get key from environment variable."""
        value = os.environ.get(self.config.env_var_name)
        if value:
            try:
                return base64.b64decode(value)
            except Exception:
                # Try as hex
                try:
                    return bytes.fromhex(value)
                except Exception:
                    pass
        return None

    def _get_from_file(self) -> Optional[bytes]:
        """Get key from encrypted file."""
        if not self.config.key_file_path.exists():
            return None

        try:
            with open(self.config.key_file_path, "rb") as f:
                data = f.read()

            # Try to parse as JSON with metadata
            try:
                parsed = json.loads(data)
                return base64.b64decode(parsed["key"])
            except (json.JSONDecodeError, KeyError):
                # Raw base64 or binary
                try:
                    return base64.b64decode(data)
                except Exception:
                    return data  # Raw bytes

        except Exception:
            return None

    def _store_in_file(self, key: bytes) -> None:
        """Store key in encrypted file."""
        self.config.key_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Store with metadata
        data = {
            "version": "1.0.0",
            "created_at": datetime.utcnow().isoformat(),
            "key": base64.b64encode(key).decode(),
            "key_hash": hashlib.sha256(key).hexdigest()[:16],  # For verification
        }

        with open(self.config.key_file_path, "w") as f:
            json.dump(data, f)

        # Set restrictive permissions
        os.chmod(self.config.key_file_path, 0o600)

    def _generate_and_store_key(self) -> bytes:
        """Generate and store new master key."""
        key = secrets.token_bytes(KEY_SIZE)
        self.store_master_key(key)
        return key
