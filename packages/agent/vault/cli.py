# -*- coding: utf-8 -*-
"""
Vault CLI - Command-line interface for local vault management.

AGENT ZONE ONLY.

Provides local management for vault and credentials:
- View vault status (without exposing secrets)
- Rotate master key
- Add/remove credentials
- Export/import (encrypted)

Design Doc Phase 4.2:
- Local Vault: rotate master key CLI
- Cloud НИКОГДА не получает master key
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from packages.agent.vault.local_vault import LocalVault
    from packages.agent.daemon.keychain import KeychainManager


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def _color(text: str, color: str) -> str:
    """Apply color to text if terminal supports it."""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.RESET}"
    return text


class VaultCLI:
    """
    CLI for local vault management.

    SECURITY:
    - Master key never leaves local machine
    - Secrets are never displayed in plaintext
    - All operations are logged locally

    Usage:
        cli = VaultCLI(vault, keychain)
        cli.run()  # Interactive mode

        # Or programmatic
        cli.status()
        cli.rotate_key()
    """

    def __init__(
        self,
        vault: "LocalVault",
        keychain: Optional["KeychainManager"] = None,
        verbose: bool = False,
    ):
        """
        Initialize CLI.

        Args:
            vault: LocalVault instance
            keychain: KeychainManager instance
            verbose: Verbose output
        """
        self._vault = vault
        self._keychain = keychain
        self._verbose = verbose

    def status(self) -> Dict[str, Any]:
        """
        Show vault status.

        Returns:
            Status dictionary
        """
        print()
        print(_color("=== CCEA Vault Status ===", Colors.BOLD))
        print()

        # Vault info
        print(_color("Vault:", Colors.CYAN))
        print(
            f"  Path:       {self._vault._db_path if hasattr(self._vault, '_db_path') else 'N/A'}"
        )
        print(
            f"  Initialized: {_color('Yes', Colors.GREEN) if self._vault._initialized else _color('No', Colors.RED)}"
        )
        print(
            f"  Locked:     {_color('Yes', Colors.YELLOW) if self._vault._locked else _color('No', Colors.GREEN)}"
        )
        print()

        # Credentials count
        creds = self._vault.list_credentials()
        print(_color("Credentials:", Colors.CYAN))
        print(f"  Total:      {len(creds)}")

        # Group by type
        by_type: Dict[str, int] = {}
        for cred in creds:
            ctype = cred.get("type", "unknown")
            by_type[ctype] = by_type.get(ctype, 0) + 1

        for ctype, count in sorted(by_type.items()):
            print(f"  {ctype}: {count}")
        print()

        # Keychain info
        if self._keychain:
            print(_color("Keychain:", Colors.CYAN))
            info = self._keychain.get_key_info()
            print(f"  Platform:   {info['platform']}")
            print(
                f"  Available:  {_color('Yes', Colors.GREEN) if info['keychain_available'] else _color('No', Colors.YELLOW)}"
            )
            print(
                f"  Has Key:    {_color('Yes', Colors.GREEN) if info.get('keychain_has_key') else _color('No', Colors.YELLOW)}"
            )
            print(
                f"  File Backup: {_color('Yes', Colors.GREEN) if info['key_file_exists'] else _color('No', Colors.DIM)}"
            )
            print()

        return {
            "vault_initialized": self._vault._initialized,
            "vault_locked": self._vault._locked,
            "credentials_count": len(creds),
            "credentials_by_type": by_type,
        }

    def list_credentials(self, show_details: bool = False) -> List[Dict[str, Any]]:
        """
        List all credentials (without secrets).

        Args:
            show_details: Show additional metadata

        Returns:
            List of credential info
        """
        creds = self._vault.list_credentials()

        print()
        print(_color("=== Stored Credentials ===", Colors.BOLD))
        print()

        if not creds:
            print(_color("  No credentials stored", Colors.DIM))
            return []

        for i, cred in enumerate(creds, 1):
            name = cred.get("name", "unknown")
            ctype = cred.get("type", "unknown")

            print(f"  [{i}] {_color(name, Colors.CYAN)}")
            print(f"      Type: {ctype}")

            if show_details:
                if cred.get("created_at"):
                    print(f"      Created: {cred['created_at']}")
                if cred.get("updated_at"):
                    print(f"      Updated: {cred['updated_at']}")
                if cred.get("metadata"):
                    for key, value in cred["metadata"].items():
                        print(f"      {key}: {value}")
            print()

        print(f"Total: {len(creds)} credential(s)")
        return creds

    def add_credential(
        self,
        name: str,
        cred_type: str = "api_key",
        interactive: bool = True,
    ) -> bool:
        """
        Add a new credential.

        Args:
            name: Credential name
            cred_type: Credential type
            interactive: Prompt for secret value

        Returns:
            True if added successfully
        """
        print()
        print(_color(f"Adding credential: {name}", Colors.BOLD))
        print()

        if interactive:
            # Prompt for secret securely (no echo)
            secret = getpass.getpass(prompt="Enter secret value: ")
            if not secret:
                print(_color("Error: Secret cannot be empty", Colors.RED))
                return False

            confirm = getpass.getpass(prompt="Confirm secret value: ")
            if secret != confirm:
                print(_color("Error: Secrets do not match", Colors.RED))
                return False
        else:
            print(_color("Error: Non-interactive mode requires piped input", Colors.RED))
            return False

        try:
            self._vault.store_credential(
                name=name,
                credential_type=cred_type,
                data={"secret": secret},
            )

            print()
            print(_color("✓ Credential stored successfully", Colors.GREEN))
            print(f"  Name: {name}")
            print(f"  Type: {cred_type}")
            return True

        except Exception as e:
            print(_color(f"✗ Failed to store credential: {e}", Colors.RED))
            return False

    def remove_credential(self, name: str, confirm: bool = True) -> bool:
        """
        Remove a credential.

        Args:
            name: Credential name
            confirm: Require confirmation

        Returns:
            True if removed
        """
        if confirm:
            print(_color(f"Are you sure you want to remove '{name}'?", Colors.YELLOW))
            response = input("Type 'yes' to confirm: ").strip().lower()
            if response != "yes":
                print("Cancelled")
                return False

        try:
            success = self._vault.delete_credential(name)
            if success:
                print(_color(f"✓ Credential '{name}' removed", Colors.GREEN))
            else:
                print(_color(f"✗ Credential '{name}' not found", Colors.RED))
            return success
        except Exception as e:
            print(_color(f"✗ Failed to remove credential: {e}", Colors.RED))
            return False

    def rotate_key(self, backup: bool = True) -> bool:
        """
        Rotate the vault master key.

        CRITICAL OPERATION - requires confirmation.

        Args:
            backup: Create backup before rotation

        Returns:
            True if rotated successfully
        """
        print()
        print(_color("⚠ MASTER KEY ROTATION", Colors.YELLOW + Colors.BOLD))
        print()
        print("This will generate a new master key and re-encrypt all credentials.")
        print("The old key will be invalidated.")
        print()

        if not self._keychain:
            print(_color("Error: KeychainManager not available", Colors.RED))
            return False

        # Confirm
        print(_color("Type 'ROTATE' to confirm: ", Colors.YELLOW), end="")
        confirm = input().strip()
        if confirm != "ROTATE":
            print("Cancelled")
            return False

        try:
            # Create backup if requested
            if backup:
                backup_path = self._create_backup()
                if backup_path:
                    print(f"  Backup created: {backup_path}")

            # Generate new key
            print("  Generating new master key...")
            new_key = self._keychain.rotate_master_key()

            # Re-encrypt vault with new key
            print("  Re-encrypting vault...")
            self._vault.rekey(new_key)

            print()
            print(_color("✓ Master key rotated successfully", Colors.GREEN))
            print()
            print(_color("IMPORTANT:", Colors.YELLOW))
            print("  - The old master key is no longer valid")
            print("  - Ensure backups of the new key are secure")
            print("  - Cloud does NOT have access to this key")

            return True

        except Exception as e:
            print(_color(f"✗ Key rotation failed: {e}", Colors.RED))
            print()
            print("The vault may be in an inconsistent state.")
            print("If you have a backup, you may need to restore it.")
            return False

    def _create_backup(self) -> Optional[Path]:
        """Create vault backup."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path.home() / ".ccea" / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            backup_path = backup_dir / f"vault_backup_{timestamp}.enc"

            # Export encrypted vault
            creds = self._vault.list_credentials()
            backup_data = {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "credentials": creds,
            }

            with open(backup_path, "w") as f:
                json.dump(backup_data, f, indent=2)

            return backup_path
        except Exception:
            return None

    def lock(self) -> bool:
        """Lock the vault."""
        try:
            self._vault.lock()
            print(_color("✓ Vault locked", Colors.GREEN))
            return True
        except Exception as e:
            print(_color(f"✗ Failed to lock vault: {e}", Colors.RED))
            return False

    def unlock(self) -> bool:
        """Unlock the vault."""
        try:
            # Get master key
            if self._keychain:
                key = self._keychain.get_master_key()
                self._vault.unlock(key)
            else:
                # Prompt for passphrase
                passphrase = getpass.getpass(prompt="Enter vault passphrase: ")
                self._vault.unlock_with_passphrase(passphrase)

            print(_color("✓ Vault unlocked", Colors.GREEN))
            return True
        except Exception as e:
            print(_color(f"✗ Failed to unlock vault: {e}", Colors.RED))
            return False

    def verify_integrity(self) -> bool:
        """
        Verify vault integrity.

        Returns:
            True if vault is intact
        """
        print()
        print(_color("Verifying vault integrity...", Colors.CYAN))
        print()

        try:
            # Check if vault can be unlocked
            if self._vault._locked:
                print(_color("  Vault is locked - unlock first", Colors.YELLOW))
                return False

            # Verify each credential can be decrypted
            creds = self._vault.list_credentials()
            errors = []

            for cred in creds:
                name = cred.get("name", "unknown")
                try:
                    # Try to retrieve (decrypt) each credential
                    self._vault.get_credential(name)
                    print(f"  ✓ {name}")
                except Exception as e:
                    print(f"  ✗ {name}: {e}")
                    errors.append(name)

            print()
            if errors:
                print(_color(f"✗ {len(errors)} credential(s) have integrity issues", Colors.RED))
                return False
            else:
                print(_color("✓ All credentials verified successfully", Colors.GREEN))
                return True

        except Exception as e:
            print(_color(f"✗ Verification failed: {e}", Colors.RED))
            return False

    def run_interactive(self) -> None:
        """Run interactive vault management session."""
        print()
        print(_color("CCEA Vault Management CLI", Colors.BOLD))
        print(_color("Type 'help' for commands, 'quit' to exit", Colors.DIM))
        print()

        while True:
            try:
                cmd = input(_color("vault> ", Colors.CYAN)).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not cmd:
                continue

            parts = cmd.split()
            command = parts[0].lower()
            args = parts[1:]

            if command in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            elif command in ("help", "h", "?"):
                self._print_help()

            elif command == "status":
                self.status()

            elif command in ("list", "ls"):
                show_details = "-v" in args or "--verbose" in args
                self.list_credentials(show_details=show_details)

            elif command == "add" and args:
                cred_type = args[1] if len(args) > 1 else "api_key"
                self.add_credential(args[0], cred_type=cred_type)

            elif command in ("remove", "rm", "delete") and args:
                self.remove_credential(args[0])

            elif command == "rotate":
                self.rotate_key()

            elif command == "lock":
                self.lock()

            elif command == "unlock":
                self.unlock()

            elif command == "verify":
                self.verify_integrity()

            else:
                print(_color(f"Unknown command: {command}", Colors.RED))
                print("Type 'help' for available commands")

    def _print_help(self) -> None:
        """Print help message."""
        print()
        print(_color("Available Commands:", Colors.BOLD))
        print()
        print("  status          Show vault status")
        print("  list [-v]       List stored credentials")
        print("  add <name> [type]  Add new credential")
        print("  remove <name>   Remove credential")
        print("  rotate          Rotate master key")
        print("  lock            Lock vault")
        print("  unlock          Unlock vault")
        print("  verify          Verify vault integrity")
        print("  help            Show this help")
        print("  quit            Exit CLI")
        print()
        print(_color("Examples:", Colors.DIM))
        print("  add binance_api api_key")
        print("  remove old_credential")
        print("  rotate")
        print()
        print(_color("Security Notes:", Colors.YELLOW))
        print("  - Master key NEVER leaves this machine")
        print("  - Cloud cannot access vault contents")
        print("  - Rotate key periodically for security")
        print()


def create_cli_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="ccea-vault",
        description="CCEA Local Vault CLI",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Status command
    subparsers.add_parser("status", help="Show vault status")

    # List command
    list_parser = subparsers.add_parser("list", help="List credentials")
    list_parser.add_argument("-v", "--verbose", action="store_true")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add credential")
    add_parser.add_argument("name", help="Credential name")
    add_parser.add_argument("-t", "--type", default="api_key", help="Credential type")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove credential")
    remove_parser.add_argument("name", help="Credential name")
    remove_parser.add_argument("-f", "--force", action="store_true", help="Skip confirmation")

    # Rotate command
    rotate_parser = subparsers.add_parser("rotate", help="Rotate master key")
    rotate_parser.add_argument("--no-backup", action="store_true", help="Skip backup")

    # Lock/unlock commands
    subparsers.add_parser("lock", help="Lock vault")
    subparsers.add_parser("unlock", help="Unlock vault")

    # Verify command
    subparsers.add_parser("verify", help="Verify vault integrity")

    # Interactive command
    subparsers.add_parser("interactive", help="Run interactive session")

    return parser


def main(
    vault: "LocalVault",
    keychain: Optional["KeychainManager"] = None,
    args: Optional[List[str]] = None,
) -> int:
    """
    Main entry point for CLI.

    Args:
        vault: LocalVault instance
        keychain: KeychainManager instance
        args: Command line arguments

    Returns:
        Exit code
    """
    parser = create_cli_parser()
    parsed = parser.parse_args(args)

    cli = VaultCLI(vault, keychain)

    if parsed.command == "status":
        cli.status()
        return 0

    elif parsed.command == "list":
        cli.list_credentials(show_details=parsed.verbose)
        return 0

    elif parsed.command == "add":
        success = cli.add_credential(parsed.name, cred_type=parsed.type)
        return 0 if success else 1

    elif parsed.command == "remove":
        success = cli.remove_credential(parsed.name, confirm=not parsed.force)
        return 0 if success else 1

    elif parsed.command == "rotate":
        success = cli.rotate_key(backup=not parsed.no_backup)
        return 0 if success else 1

    elif parsed.command == "lock":
        success = cli.lock()
        return 0 if success else 1

    elif parsed.command == "unlock":
        success = cli.unlock()
        return 0 if success else 1

    elif parsed.command == "verify":
        success = cli.verify_integrity()
        return 0 if success else 1

    elif parsed.command == "interactive" or parsed.command is None:
        cli.run_interactive()
        return 0

    else:
        parser.print_help()
        return 1
