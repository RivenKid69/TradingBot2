# -*- coding: utf-8 -*-
"""
Approval Record Persistence - File-based storage for approval records.

AGENT ZONE ONLY.

Provides durable storage for approval requests and decisions,
ensuring audit trail survives restarts.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, List, Optional
from uuid import UUID

from packages.shared.contracts.config import ChangeClass

# Default storage path
DEFAULT_STORAGE_PATH: Final[Path] = Path.home() / ".ccea" / "approvals"


@dataclass
class ApprovalPersistenceConfig:
    """Configuration for approval persistence."""

    storage_dir: Path = field(default_factory=lambda: DEFAULT_STORAGE_PATH)
    max_history_size: int = 10000
    auto_flush: bool = True
    flush_interval_seconds: int = 30
    enable_compression: bool = False


class ApprovalPersistence:
    """
    File-based persistence for approval records.

    Stores:
    - Pending requests: {storage_dir}/pending/{request_id}.json
    - History: {storage_dir}/history/{YYYY-MM}/{request_id}.json
    - Index: {storage_dir}/index.json (fast lookup)

    Thread-safe implementation.
    """

    def __init__(self, config: Optional[ApprovalPersistenceConfig] = None):
        """Initialize persistence layer."""
        self._config = config or ApprovalPersistenceConfig()
        self._lock = threading.RLock()
        self._index: Dict[str, Dict[str, Any]] = {}
        self._dirty = False

        # Ensure directories exist
        self._ensure_dirs()
        self._load_index()

    def _ensure_dirs(self) -> None:
        """Ensure storage directories exist."""
        (self._config.storage_dir / "pending").mkdir(parents=True, exist_ok=True)
        (self._config.storage_dir / "history").mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> None:
        """Load index from disk."""
        index_path = self._config.storage_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save index to disk."""
        index_path = self._config.storage_dir / "index.json"
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, indent=2, default=str)
            self._dirty = False
        except IOError:
            pass  # Best effort

    def save_pending(self, request_id: UUID, data: Dict[str, Any]) -> bool:
        """
        Save pending approval request.

        Args:
            request_id: Request UUID
            data: Request data as dict

        Returns:
            True if saved successfully
        """
        with self._lock:
            try:
                path = self._config.storage_dir / "pending" / f"{request_id}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)

                # Update index
                self._index[str(request_id)] = {
                    "status": "pending",
                    "path": str(path),
                    "created_at": data.get("created_at"),
                    "command_type": data.get("command_type"),
                }
                self._dirty = True

                if self._config.auto_flush:
                    self._save_index()

                return True
            except IOError:
                return False

    def save_history(self, request_id: UUID, data: Dict[str, Any]) -> bool:
        """
        Save completed approval request to history.

        Args:
            request_id: Request UUID
            data: Request data with decision

        Returns:
            True if saved successfully
        """
        with self._lock:
            try:
                # Organize by year-month
                decided_at = data.get("decided_at")
                if isinstance(decided_at, str):
                    dt = datetime.fromisoformat(decided_at.replace("Z", "+00:00"))
                elif isinstance(decided_at, datetime):
                    dt = decided_at
                else:
                    dt = datetime.utcnow()

                month_dir = self._config.storage_dir / "history" / dt.strftime("%Y-%m")
                month_dir.mkdir(parents=True, exist_ok=True)

                path = month_dir / f"{request_id}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)

                # Remove from pending if exists
                pending_path = self._config.storage_dir / "pending" / f"{request_id}.json"
                if pending_path.exists():
                    pending_path.unlink()

                # Update index
                self._index[str(request_id)] = {
                    "status": data.get("status", "completed"),
                    "path": str(path),
                    "decided_at": str(dt),
                    "command_type": data.get("command_type"),
                    "approved": data.get("status") == "approved",
                }
                self._dirty = True

                if self._config.auto_flush:
                    self._save_index()

                return True
            except IOError:
                return False

    def load_pending(self, request_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Load pending request by ID.

        Args:
            request_id: Request UUID

        Returns:
            Request data or None
        """
        with self._lock:
            path = self._config.storage_dir / "pending" / f"{request_id}.json"
            if not path.exists():
                return None

            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None

    def load_all_pending(self) -> List[Dict[str, Any]]:
        """
        Load all pending requests.

        Returns:
            List of pending request data
        """
        with self._lock:
            pending_dir = self._config.storage_dir / "pending"
            results = []

            for path in pending_dir.glob("*.json"):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        results.append(json.load(f))
                except (json.JSONDecodeError, IOError):
                    continue

            return results

    def load_history(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
        command_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load history records.

        Args:
            limit: Maximum records to return
            since: Only records after this time
            command_type: Filter by command type

        Returns:
            List of history records
        """
        with self._lock:
            history_dir = self._config.storage_dir / "history"
            results = []

            # Get month directories in reverse order
            month_dirs = sorted(history_dir.glob("*-*"), reverse=True)

            for month_dir in month_dirs:
                if not month_dir.is_dir():
                    continue

                for path in sorted(month_dir.glob("*.json"), reverse=True):
                    if len(results) >= limit:
                        break

                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        # Apply filters
                        if command_type and data.get("command_type") != command_type:
                            continue

                        if since:
                            decided_at = data.get("decided_at")
                            if decided_at:
                                dt = datetime.fromisoformat(
                                    decided_at.replace("Z", "+00:00")
                                    if isinstance(decided_at, str)
                                    else str(decided_at)
                                )
                                if dt < since:
                                    continue

                        results.append(data)
                    except (json.JSONDecodeError, IOError):
                        continue

                if len(results) >= limit:
                    break

            return results[:limit]

    def delete_pending(self, request_id: UUID) -> bool:
        """
        Delete pending request.

        Args:
            request_id: Request UUID

        Returns:
            True if deleted
        """
        with self._lock:
            path = self._config.storage_dir / "pending" / f"{request_id}.json"
            if path.exists():
                path.unlink()
                self._index.pop(str(request_id), None)
                self._dirty = True
                if self._config.auto_flush:
                    self._save_index()
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        with self._lock:
            pending_count = len(list((self._config.storage_dir / "pending").glob("*.json")))
            history_count = sum(1 for _ in (self._config.storage_dir / "history").rglob("*.json"))

            return {
                "pending_count": pending_count,
                "history_count": history_count,
                "index_size": len(self._index),
                "storage_dir": str(self._config.storage_dir),
                "dirty": self._dirty,
            }

    def flush(self) -> None:
        """Force flush index to disk."""
        with self._lock:
            if self._dirty:
                self._save_index()

    def cleanup_old_history(self, days: int = 365) -> int:
        """
        Clean up history older than specified days.

        Args:
            days: Delete records older than this

        Returns:
            Number of records deleted
        """
        with self._lock:
            cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff = cutoff.replace(
                year=cutoff.year if cutoff.month > 1 else cutoff.year - 1,
                month=(cutoff.month - 1) if cutoff.month > 1 else 12,
            )

            deleted = 0
            history_dir = self._config.storage_dir / "history"

            for month_dir in history_dir.glob("*-*"):
                if not month_dir.is_dir():
                    continue

                try:
                    # Parse year-month from directory name
                    parts = month_dir.name.split("-")
                    if len(parts) != 2:
                        continue

                    year, month = int(parts[0]), int(parts[1])
                    dir_date = datetime(year, month, 1)

                    # Delete if older than cutoff
                    from datetime import timedelta

                    if dir_date < cutoff - timedelta(days=days):
                        for path in month_dir.glob("*.json"):
                            path.unlink()
                            self._index.pop(path.stem, None)
                            deleted += 1

                        # Remove empty directory
                        if not any(month_dir.iterdir()):
                            month_dir.rmdir()

                except (ValueError, OSError):
                    continue

            if deleted > 0:
                self._dirty = True
                if self._config.auto_flush:
                    self._save_index()

            return deleted
