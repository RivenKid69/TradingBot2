# -*- coding: utf-8 -*-
"""
Order Journal - Durable order logging for crash recovery.

AGENT ZONE ONLY.

Maintains a persistent log of all orders for:
- Crash recovery
- Duplicate detection
- Audit trail
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional
from uuid import UUID


class JournalStatus(str, Enum):
    """Journal entry status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class JournalEntry:
    """
    Journal entry for an order.
    """

    entry_id: str
    client_order_id: str
    intent_id: str
    symbol: str
    side: str
    quantity: str
    order_type: str
    status: JournalStatus = JournalStatus.PENDING
    broker_order_id: Optional[str] = None
    filled_quantity: str = "0"
    avg_price: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "client_order_id": self.client_order_id,
            "intent_id": self.intent_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status.value,
            "broker_order_id": self.broker_order_id,
            "filled_quantity": self.filled_quantity,
            "avg_price": self.avg_price,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> JournalEntry:
        """Create from dictionary."""
        return cls(
            entry_id=data["entry_id"],
            client_order_id=data["client_order_id"],
            intent_id=data["intent_id"],
            symbol=data["symbol"],
            side=data["side"],
            quantity=data["quantity"],
            order_type=data["order_type"],
            status=JournalStatus(data.get("status", "pending")),
            broker_order_id=data.get("broker_order_id"),
            filled_quantity=data.get("filled_quantity", "0"),
            avg_price=data.get("avg_price"),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


class OrderJournal:
    """
    Persistent order journal for crash recovery.

    Uses SQLite for durable storage.

    Usage:
        journal = OrderJournal()

        # Log order
        entry = journal.log_order(order)

        # Update status
        journal.update_status(entry.entry_id, JournalStatus.CONFIRMED)

        # Check for duplicates
        if journal.is_duplicate(client_order_id):
            skip_order()

        # Recover after crash
        pending = journal.get_pending_orders()
        for entry in pending:
            reconcile_with_broker(entry)
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize journal.

        Args:
            db_path: Path to SQLite database
        """
        self._db_path = db_path or Path.home() / ".ccea" / "order_journal.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                entry_id TEXT PRIMARY KEY,
                client_order_id TEXT UNIQUE NOT NULL,
                intent_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity TEXT NOT NULL,
                order_type TEXT NOT NULL,
                status TEXT NOT NULL,
                broker_order_id TEXT,
                filled_quantity TEXT DEFAULT '0',
                avg_price TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT
            )
            """
        )

        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_client_order_id
            ON orders(client_order_id)
            """
        )

        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_status
            ON orders(status)
            """
        )

        self._conn.commit()

    def log_order(
        self,
        client_order_id: str,
        intent_id: str,
        symbol: str,
        side: str,
        quantity: Decimal,
        order_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JournalEntry:
        """
        Log new order to journal.

        Args:
            client_order_id: Client order ID (for idempotency)
            intent_id: Source intent ID
            symbol: Trading symbol
            side: buy/sell
            quantity: Order quantity
            order_type: market/limit/etc
            metadata: Additional metadata

        Returns:
            Created journal entry
        """
        import uuid

        entry = JournalEntry(
            entry_id=str(uuid.uuid4()),
            client_order_id=client_order_id,
            intent_id=intent_id,
            symbol=symbol,
            side=side,
            quantity=str(quantity),
            order_type=order_type,
            status=JournalStatus.PENDING,
            metadata=metadata or {},
        )

        self._conn.execute(
            """
            INSERT INTO orders (
                entry_id, client_order_id, intent_id, symbol, side,
                quantity, order_type, status, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.entry_id,
                entry.client_order_id,
                entry.intent_id,
                entry.symbol,
                entry.side,
                entry.quantity,
                entry.order_type,
                entry.status.value,
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                json.dumps(entry.metadata),
            ),
        )
        self._conn.commit()

        return entry

    def update_status(
        self,
        entry_id: str,
        status: JournalStatus,
        broker_order_id: Optional[str] = None,
        filled_quantity: Optional[Decimal] = None,
        avg_price: Optional[Decimal] = None,
    ) -> bool:
        """
        Update order status.

        Returns:
            True if updated, False if not found
        """
        updates = ["status = ?", "updated_at = ?"]
        values = [status.value, datetime.utcnow().isoformat()]

        if broker_order_id is not None:
            updates.append("broker_order_id = ?")
            values.append(broker_order_id)

        if filled_quantity is not None:
            updates.append("filled_quantity = ?")
            values.append(str(filled_quantity))

        if avg_price is not None:
            updates.append("avg_price = ?")
            values.append(str(avg_price))

        values.append(entry_id)

        cursor = self._conn.execute(
            f"UPDATE orders SET {', '.join(updates)} WHERE entry_id = ?",
            values,
        )
        self._conn.commit()

        return cursor.rowcount > 0

    def is_duplicate(self, client_order_id: str) -> bool:
        """Check if order already exists."""
        cursor = self._conn.execute(
            "SELECT 1 FROM orders WHERE client_order_id = ?",
            (client_order_id,),
        )
        return cursor.fetchone() is not None

    def get_entry(self, entry_id: str) -> Optional[JournalEntry]:
        """Get journal entry by ID."""
        cursor = self._conn.execute(
            "SELECT * FROM orders WHERE entry_id = ?",
            (entry_id,),
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_entry(row)
        return None

    def get_by_client_id(self, client_order_id: str) -> Optional[JournalEntry]:
        """Get journal entry by client order ID."""
        cursor = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id = ?",
            (client_order_id,),
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_entry(row)
        return None

    def get_pending_orders(self) -> List[JournalEntry]:
        """Get all pending orders (for recovery)."""
        cursor = self._conn.execute(
            """
            SELECT * FROM orders
            WHERE status IN (?, ?)
            ORDER BY created_at ASC
            """,
            (JournalStatus.PENDING.value, JournalStatus.SUBMITTED.value),
        )
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def get_orders_with_status(self, statuses: List[JournalStatus]) -> List[JournalEntry]:
        """Get all orders matching any of the provided statuses."""
        if not statuses:
            return []

        placeholders = ", ".join(["?"] * len(statuses))
        values = tuple(s.value for s in statuses)
        cursor = self._conn.execute(
            f"""
            SELECT * FROM orders
            WHERE status IN ({placeholders})
            ORDER BY created_at ASC
            """,
            values,
        )
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def get_unresolved_orders(self) -> List[JournalEntry]:
        """
        Get orders that are unresolved/uncertain and require reconciliation.

        Unresolved statuses:
        - PENDING / SUBMITTED: potentially inflight orders (restart uncertainty)
        - UNKNOWN: broker state unknown (must safe-halt until resolved)
        """
        return self.get_orders_with_status(
            [JournalStatus.PENDING, JournalStatus.SUBMITTED, JournalStatus.UNKNOWN]
        )

    def get_recent_orders(self, limit: int = 100) -> List[JournalEntry]:
        """Get recent orders."""
        cursor = self._conn.execute(
            """
            SELECT * FROM orders
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def _row_to_entry(self, row: sqlite3.Row) -> JournalEntry:
        """Convert database row to entry."""
        return JournalEntry(
            entry_id=row["entry_id"],
            client_order_id=row["client_order_id"],
            intent_id=row["intent_id"],
            symbol=row["symbol"],
            side=row["side"],
            quantity=row["quantity"],
            order_type=row["order_type"],
            status=JournalStatus(row["status"]),
            broker_order_id=row["broker_order_id"],
            filled_quantity=row["filled_quantity"] or "0",
            avg_price=row["avg_price"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
