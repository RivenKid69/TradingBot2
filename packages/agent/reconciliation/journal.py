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
import threading
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
        *,
        hmac_key: Optional[bytes] = None,
    ):
        """
        Initialize journal.

        Args:
            db_path: Path to SQLite database
            hmac_key: optional key for the tamper-evident audit chain (from the
                Agent vault). When omitted the chain is unkeyed SHA-256 (still
                tamper-evident; keyed makes it tamper-proof against forgery).
        """
        self._db_path = db_path or Path.home() / ".ccea" / "order_journal.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._hmac_key = hmac_key
        # OMS-запросы (submit/cancel/fill) приходят на разных потоках (uvicorn
        # threadpool для sync-эндпоинтов), поэтому соединение переносимо между
        # потоками, а запись в append-only hash-chain сериализуется локом.
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
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

        # Tamper-evident append-only audit chain: an immutable hash-linked log of
        # every order lifecycle event (logged / status changes). The `orders` table
        # above remains the mutable working state for fast lookups; this table is
        # the WORM record of record. INSERT-only — never UPDATE/DELETE.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS order_audit (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                event_type TEXT NOT NULL,
                client_order_id TEXT,
                entry_id TEXT,
                payload TEXT NOT NULL,
                prev_hash TEXT NOT NULL,
                entry_hash TEXT NOT NULL
            )
            """
        )

        self._conn.commit()

    # ---- tamper-evident audit chain -------------------------------------
    def _audit_head_hash(self) -> str:
        from packages.agent.audit.hash_chain import GENESIS_HASH
        row = self._conn.execute(
            "SELECT entry_hash FROM order_audit ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row["entry_hash"] if row else GENESIS_HASH

    def _append_audit(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Append a hash-chained audit event (best-effort; never breaks the order)."""
        try:
            from datetime import datetime as _dt

            from packages.agent.audit.hash_chain import chain_hash
            seq = int(self._conn.execute("SELECT COUNT(*) c FROM order_audit").fetchone()["c"]) + 1
            prev = self._audit_head_hash()
            body = {"event_type": event_type, **payload}
            h = chain_hash(prev, body, seq, key=self._hmac_key)
            self._conn.execute(
                "INSERT INTO order_audit(ts, event_type, client_order_id, entry_id, payload, prev_hash, entry_hash) "
                "VALUES(?,?,?,?,?,?,?)",
                (_dt.utcnow().isoformat(), event_type, payload.get("client_order_id"),
                 payload.get("entry_id"), json.dumps(body, sort_keys=True), prev, h),
            )
            self._conn.commit()
        except Exception:  # pragma: no cover - audit must never break execution
            pass

    def verify_audit_chain(self) -> Dict[str, Any]:
        """Recompute the audit chain and report integrity (tamper-evidence)."""
        from packages.agent.audit.hash_chain import ChainRecord, verify_chain
        rows = self._conn.execute(
            "SELECT seq, payload, prev_hash, entry_hash FROM order_audit ORDER BY seq ASC"
        ).fetchall()
        recs = [ChainRecord(seq=r["seq"], payload=json.loads(r["payload"]),
                            prev_hash=r["prev_hash"], entry_hash=r["entry_hash"]) for r in rows]
        out = verify_chain(recs, key=self._hmac_key)
        out["keyed"] = self._hmac_key is not None
        out["head_hash"] = self._audit_head_hash()
        return out

    def get_audit_events(self, *, limit: int = 200) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT seq, ts, event_type, client_order_id, entry_id, entry_hash "
            "FROM order_audit ORDER BY seq DESC LIMIT ?", (int(limit),)
        ).fetchall()
        return [dict(r) for r in rows]

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

        with self._lock:
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

            self._append_audit("order_logged", {
                "client_order_id": entry.client_order_id, "entry_id": entry.entry_id,
                "intent_id": entry.intent_id, "symbol": entry.symbol, "side": entry.side,
                "quantity": entry.quantity, "order_type": entry.order_type,
                "status": entry.status.value,
            })
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

        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE orders SET {', '.join(updates)} WHERE entry_id = ?",
                values,
            )
            self._conn.commit()

            if cursor.rowcount > 0:
                self._append_audit(f"status_{status.value}", {
                    "entry_id": entry_id, "status": status.value,
                    "broker_order_id": broker_order_id,
                    "filled_quantity": str(filled_quantity) if filled_quantity is not None else None,
                    "avg_price": str(avg_price) if avg_price is not None else None,
                })
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

    def get_all_entries(self) -> List[JournalEntry]:
        """
        Get all journal entries.

        Design Doc Phase 8 WI-AGENT-06:
        Used for sequence recovery on restart - scans all entries
        to find the highest sequence number for the current deployment/run.

        Returns:
            List of all journal entries
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM orders
            ORDER BY created_at ASC
            """
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

    def __enter__(self) -> "OrderJournal":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        try:
            self.close()
        except Exception:
            pass


class CommandStatus(str, Enum):
    """Cloud command execution status."""

    RECEIVED = "received"      # Command received from Cloud
    IN_PROGRESS = "in_progress"  # Execution started
    COMPLETED = "completed"    # Successfully executed
    FAILED = "failed"         # Execution failed
    SKIPPED = "skipped"       # Skipped (duplicate/expired)


@dataclass
class CommandEntry:
    """
    Journal entry for a cloud command.

    Design Doc 10.4.2: Tracks command idempotency across restarts.
    """

    command_id: str
    idempotency_key: str
    command_type: str
    status: CommandStatus
    payload_ref: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    received_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "command_id": self.command_id,
            "idempotency_key": self.idempotency_key,
            "command_type": self.command_type,
            "status": self.status.value,
            "payload_ref": self.payload_ref,
            "result": self.result,
            "error_message": self.error_message,
            "received_at": self.received_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommandEntry":
        """Create from dictionary."""
        return cls(
            command_id=data["command_id"],
            idempotency_key=data["idempotency_key"],
            command_type=data["command_type"],
            status=CommandStatus(data.get("status", "received")),
            payload_ref=data.get("payload_ref"),
            result=data.get("result"),
            error_message=data.get("error_message"),
            received_at=datetime.fromisoformat(data["received_at"])
            if "received_at" in data
            else datetime.utcnow(),
            executed_at=datetime.fromisoformat(data["executed_at"])
            if data.get("executed_at")
            else None,
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.utcnow(),
        )


class CommandJournal:
    """
    Persistent journal for cloud command execution.

    Design Doc 10.4.2/9.5:
    - Provides idempotency across agent restarts
    - Tracks command execution status
    - Prevents duplicate execution of the same command

    AGENT ZONE ONLY.

    Usage:
        journal = CommandJournal()

        # Before executing command
        if journal.is_executed(command_id, idempotency_key):
            skip_command()  # Already executed

        # Mark command received
        journal.log_received(command_id, idempotency_key, command_type)

        # Execute command
        try:
            result = execute_command(...)
            journal.mark_completed(command_id, result)
        except Exception as e:
            journal.mark_failed(command_id, str(e))

        # On restart: recover state
        executed_ids = journal.get_executed_command_ids()
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize command journal.

        Args:
            db_path: Path to SQLite database
        """
        self._db_path = db_path or Path.home() / ".ccea" / "command_journal.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS commands (
                command_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL,
                command_type TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_ref TEXT,
                result TEXT,
                error_message TEXT,
                received_at TEXT NOT NULL,
                executed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_idempotency_key
            ON commands(idempotency_key)
            """
        )

        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_command_status
            ON commands(status)
            """
        )

        self._conn.commit()

    def log_received(
        self,
        command_id: str,
        idempotency_key: str,
        command_type: str,
        payload_ref: Optional[str] = None,
    ) -> CommandEntry:
        """
        Log command receipt.

        Design Doc 10.4.2: Mark command RECEIVED before execution.

        Args:
            command_id: Cloud command ID
            idempotency_key: Idempotency key
            command_type: Command type (e.g., REQUEST_START_RUN)
            payload_ref: Reference to command payload

        Returns:
            Created journal entry
        """
        now = datetime.utcnow()
        entry = CommandEntry(
            command_id=command_id,
            idempotency_key=idempotency_key,
            command_type=command_type,
            status=CommandStatus.RECEIVED,
            payload_ref=payload_ref,
            received_at=now,
            created_at=now,
            updated_at=now,
        )

        try:
            self._conn.execute(
                """
                INSERT INTO commands (
                    command_id, idempotency_key, command_type, status,
                    payload_ref, received_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.command_id,
                    entry.idempotency_key,
                    entry.command_type,
                    entry.status.value,
                    entry.payload_ref,
                    entry.received_at.isoformat(),
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            # Already exists - fetch existing entry
            return self.get_entry(command_id)

        return entry

    def mark_in_progress(self, command_id: str) -> bool:
        """Mark command as in progress."""
        return self._update_status(command_id, CommandStatus.IN_PROGRESS)

    def mark_completed(
        self,
        command_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Mark command as completed.

        Design Doc 10.4.2: Mark COMPLETED after successful execution.
        """
        now = datetime.utcnow()
        cursor = self._conn.execute(
            """
            UPDATE commands
            SET status = ?, result = ?, executed_at = ?, updated_at = ?
            WHERE command_id = ?
            """,
            (
                CommandStatus.COMPLETED.value,
                json.dumps(result) if result else None,
                now.isoformat(),
                now.isoformat(),
                command_id,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def mark_failed(
        self,
        command_id: str,
        error_message: str,
    ) -> bool:
        """
        Mark command as failed.

        Design Doc 10.4.2: Mark FAILED if execution fails.
        """
        now = datetime.utcnow()
        cursor = self._conn.execute(
            """
            UPDATE commands
            SET status = ?, error_message = ?, executed_at = ?, updated_at = ?
            WHERE command_id = ?
            """,
            (
                CommandStatus.FAILED.value,
                error_message,
                now.isoformat(),
                now.isoformat(),
                command_id,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def mark_skipped(
        self,
        command_id: str,
        reason: str = "duplicate",
    ) -> bool:
        """Mark command as skipped (duplicate/expired)."""
        now = datetime.utcnow()
        cursor = self._conn.execute(
            """
            UPDATE commands
            SET status = ?, error_message = ?, updated_at = ?
            WHERE command_id = ?
            """,
            (
                CommandStatus.SKIPPED.value,
                reason,
                now.isoformat(),
                command_id,
            ),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def _update_status(self, command_id: str, status: CommandStatus) -> bool:
        """Update command status."""
        cursor = self._conn.execute(
            """
            UPDATE commands
            SET status = ?, updated_at = ?
            WHERE command_id = ?
            """,
            (status.value, datetime.utcnow().isoformat(), command_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def is_executed(self, command_id: str, idempotency_key: Optional[str] = None) -> bool:
        """
        Check if command has already been executed.

        Design Doc 10.4.2: Prevent duplicate execution.

        Args:
            command_id: Command ID to check
            idempotency_key: Optional idempotency key to check

        Returns:
            True if command already executed (COMPLETED or FAILED)
        """
        # Check by command_id
        cursor = self._conn.execute(
            """
            SELECT status FROM commands
            WHERE command_id = ?
            """,
            (command_id,),
        )
        row = cursor.fetchone()
        if row:
            status = row["status"]
            return status in (CommandStatus.COMPLETED.value, CommandStatus.FAILED.value, CommandStatus.SKIPPED.value)

        # Also check by idempotency_key if provided
        if idempotency_key:
            cursor = self._conn.execute(
                """
                SELECT status FROM commands
                WHERE idempotency_key = ? AND status IN (?, ?, ?)
                """,
                (
                    idempotency_key,
                    CommandStatus.COMPLETED.value,
                    CommandStatus.FAILED.value,
                    CommandStatus.SKIPPED.value,
                ),
            )
            return cursor.fetchone() is not None

        return False

    def get_entry(self, command_id: str) -> Optional[CommandEntry]:
        """Get command entry by ID."""
        cursor = self._conn.execute(
            "SELECT * FROM commands WHERE command_id = ?",
            (command_id,),
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_entry(row)
        return None

    def get_by_idempotency_key(self, idempotency_key: str) -> Optional[CommandEntry]:
        """Get command entry by idempotency key."""
        cursor = self._conn.execute(
            "SELECT * FROM commands WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_entry(row)
        return None

    def get_executed_command_ids(self) -> set[str]:
        """
        Get all executed command IDs.

        Design Doc 10.4.2: Used on startup to restore state.

        Returns:
            Set of command IDs that have been executed
        """
        cursor = self._conn.execute(
            """
            SELECT command_id FROM commands
            WHERE status IN (?, ?, ?)
            """,
            (
                CommandStatus.COMPLETED.value,
                CommandStatus.FAILED.value,
                CommandStatus.SKIPPED.value,
            ),
        )
        return {row["command_id"] for row in cursor.fetchall()}

    def get_pending_commands(self) -> List[CommandEntry]:
        """Get commands that are pending (RECEIVED or IN_PROGRESS)."""
        cursor = self._conn.execute(
            """
            SELECT * FROM commands
            WHERE status IN (?, ?)
            ORDER BY received_at ASC
            """,
            (CommandStatus.RECEIVED.value, CommandStatus.IN_PROGRESS.value),
        )
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def get_recent_commands(self, limit: int = 100) -> List[CommandEntry]:
        """Get recent commands."""
        cursor = self._conn.execute(
            """
            SELECT * FROM commands
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def _row_to_entry(self, row: sqlite3.Row) -> CommandEntry:
        """Convert database row to entry."""
        return CommandEntry(
            command_id=row["command_id"],
            idempotency_key=row["idempotency_key"],
            command_type=row["command_type"],
            status=CommandStatus(row["status"]),
            payload_ref=row["payload_ref"],
            result=json.loads(row["result"]) if row["result"] else None,
            error_message=row["error_message"],
            received_at=datetime.fromisoformat(row["received_at"]),
            executed_at=datetime.fromisoformat(row["executed_at"]) if row["executed_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "CommandJournal":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        try:
            self.close()
        except Exception:
            pass
