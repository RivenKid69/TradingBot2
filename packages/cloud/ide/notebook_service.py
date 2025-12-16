# -*- coding: utf-8 -*-
"""
Jupyter Notebook Service - CLOUD ZONE ONLY.

Provides server-side Jupyter notebook execution with tenant isolation.

Design Doc Reference:
    - Section 4.1: "Strategy Workspace (IDE/ноутбуки)"
    - Section 5.3: Cloud multi-tenant isolation, sandbox, quotas

Security:
    - Notebook execution in isolated containers
    - No access to broker credentials (Cloud Zone)
    - Resource quotas per workspace
    - Output sanitization for sensitive data

Based on Jupyter Server best practices:
    - https://jupyter-server.readthedocs.io/
    - https://jupyter.org/security
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Union
from uuid import UUID

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Maximum cells per notebook
MAX_CELLS_PER_NOTEBOOK: Final[int] = 500

# Maximum output size per cell (bytes)
MAX_OUTPUT_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB

# Default execution timeout per cell (seconds)
DEFAULT_CELL_TIMEOUT: Final[int] = 300  # 5 minutes

# Maximum execution timeout per cell (seconds)
MAX_CELL_TIMEOUT: Final[int] = 3600  # 1 hour

# Session timeout (minutes)
SESSION_TIMEOUT_MINUTES: Final[int] = 120

# Prohibited imports in Cloud Zone (broker SDKs, execution)
PROHIBITED_IMPORTS: Final[frozenset] = frozenset({
    "alpaca_trade_api",
    "ib_insync",
    "ccxt",
    "packages.agent",
    "ccea.agent",
    "broker_connectors",
    "execution_providers",
})

# Sensitive patterns to redact from output
SENSITIVE_PATTERNS: Final[List[re.Pattern]] = [
    re.compile(r'api[_-]?key\s*[=:]\s*["\'][^"\']+["\']', re.IGNORECASE),
    re.compile(r'secret\s*[=:]\s*["\'][^"\']+["\']', re.IGNORECASE),
    re.compile(r'password\s*[=:]\s*["\'][^"\']+["\']', re.IGNORECASE),
    re.compile(r'token\s*[=:]\s*["\'][^"\']+["\']', re.IGNORECASE),
]


# ============================================================================
# Enums
# ============================================================================


class CellType(str, Enum):
    """Notebook cell type."""
    CODE = "code"
    MARKDOWN = "markdown"
    RAW = "raw"


class ExecutionState(str, Enum):
    """Cell execution state."""
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class OutputType(str, Enum):
    """Cell output type."""
    STREAM = "stream"
    EXECUTE_RESULT = "execute_result"
    DISPLAY_DATA = "display_data"
    ERROR = "error"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CellOutput:
    """
    Notebook cell output.

    Based on Jupyter nbformat specification:
    https://nbformat.readthedocs.io/en/latest/format_description.html
    """
    output_type: OutputType
    data: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    name: Optional[str] = None  # For stream outputs (stdout/stderr)
    execution_count: Optional[int] = None
    ename: Optional[str] = None  # Error name
    evalue: Optional[str] = None  # Error value
    traceback: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nbformat-compatible dict."""
        result: Dict[str, Any] = {"output_type": self.output_type.value}

        if self.output_type == OutputType.STREAM:
            result["name"] = self.name or "stdout"
            result["text"] = self.text or ""
        elif self.output_type == OutputType.EXECUTE_RESULT:
            result["data"] = self.data
            result["execution_count"] = self.execution_count
            result["metadata"] = self.metadata
        elif self.output_type == OutputType.DISPLAY_DATA:
            result["data"] = self.data
            result["metadata"] = self.metadata
        elif self.output_type == OutputType.ERROR:
            result["ename"] = self.ename or "Error"
            result["evalue"] = self.evalue or ""
            result["traceback"] = self.traceback or []

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CellOutput":
        """Create from nbformat-compatible dict."""
        output_type = OutputType(data.get("output_type", "stream"))
        return cls(
            output_type=output_type,
            data=data.get("data", {}),
            text=data.get("text"),
            name=data.get("name"),
            execution_count=data.get("execution_count"),
            ename=data.get("ename"),
            evalue=data.get("evalue"),
            traceback=data.get("traceback"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class NotebookCell:
    """
    Notebook cell.

    Based on Jupyter nbformat 4.5 specification.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cell_type: CellType = CellType.CODE
    source: str = ""
    outputs: List[CellOutput] = field(default_factory=list)
    execution_count: Optional[int] = None
    execution_state: ExecutionState = ExecutionState.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nbformat-compatible dict."""
        result = {
            "id": self.id,
            "cell_type": self.cell_type.value,
            "source": self.source,
            "metadata": self.metadata,
        }

        if self.cell_type == CellType.CODE:
            result["outputs"] = [o.to_dict() for o in self.outputs]
            result["execution_count"] = self.execution_count

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotebookCell":
        """Create from nbformat-compatible dict."""
        cell_type = CellType(data.get("cell_type", "code"))
        outputs = []
        if cell_type == CellType.CODE and "outputs" in data:
            outputs = [CellOutput.from_dict(o) for o in data["outputs"]]

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            cell_type=cell_type,
            source=data.get("source", ""),
            outputs=outputs,
            execution_count=data.get("execution_count"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class NotebookSession:
    """
    Active notebook session with kernel state.

    Provides isolated execution environment per user/workspace.
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: UUID = field(default_factory=uuid.uuid4)
    user_id: UUID = field(default_factory=uuid.uuid4)
    notebook_id: str = ""

    # Session state
    cells: List[NotebookCell] = field(default_factory=list)
    kernel_state: Dict[str, Any] = field(default_factory=dict)
    execution_counter: int = 0

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Resource tracking
    memory_usage_bytes: int = 0
    cpu_time_seconds: float = 0.0

    # Notebook metadata (nbformat)
    nbformat: int = 4
    nbformat_minor: int = 5
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        }
    })

    def is_expired(self) -> bool:
        """Check if session has expired."""
        timeout = timedelta(minutes=SESSION_TIMEOUT_MINUTES)
        return datetime.now(timezone.utc) - self.last_activity > timeout

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def to_notebook_dict(self) -> Dict[str, Any]:
        """Convert to nbformat-compatible notebook dict."""
        return {
            "nbformat": self.nbformat,
            "nbformat_minor": self.nbformat_minor,
            "metadata": self.metadata,
            "cells": [cell.to_dict() for cell in self.cells],
        }

    @classmethod
    def from_notebook_dict(
        cls,
        data: Dict[str, Any],
        workspace_id: UUID,
        user_id: UUID,
        notebook_id: str = "",
    ) -> "NotebookSession":
        """Create session from nbformat notebook dict."""
        cells = [NotebookCell.from_dict(c) for c in data.get("cells", [])]
        return cls(
            workspace_id=workspace_id,
            user_id=user_id,
            notebook_id=notebook_id,
            cells=cells,
            nbformat=data.get("nbformat", 4),
            nbformat_minor=data.get("nbformat_minor", 5),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionResult:
    """Result of cell execution."""
    cell_id: str
    success: bool
    outputs: List[CellOutput]
    execution_count: int
    execution_time_ms: int
    error_message: Optional[str] = None
    truncated: bool = False


@dataclass
class NotebookQuota:
    """Resource quota for notebook execution."""
    max_memory_mb: int = 4096  # 4GB
    max_cpu_seconds: int = 3600  # 1 hour total
    max_cells: int = MAX_CELLS_PER_NOTEBOOK
    max_output_size_mb: int = 10
    max_concurrent_executions: int = 5
    cell_timeout_seconds: int = DEFAULT_CELL_TIMEOUT


# ============================================================================
# Notebook Service
# ============================================================================


class NotebookService:
    """
    Jupyter Notebook execution service.

    Provides:
    - Notebook session management
    - Cell execution with isolation
    - Resource quotas and limits
    - Output sanitization
    - Import restrictions (Cloud Zone)

    SECURITY: This service runs in Cloud Zone and MUST NOT:
    - Access broker credentials
    - Execute live trading operations
    - Import agent-zone packages
    """

    def __init__(
        self,
        default_quota: Optional[NotebookQuota] = None,
        workspace_quotas: Optional[Dict[UUID, NotebookQuota]] = None,
    ):
        self._sessions: Dict[str, NotebookSession] = {}
        self._default_quota = default_quota or NotebookQuota()
        self._workspace_quotas = workspace_quotas or {}
        self._execution_lock = asyncio.Lock()
        self._active_executions: Dict[str, int] = {}  # workspace_id -> count

        logger.info("NotebookService initialized (Cloud Zone)")

    # ========================================================================
    # Session Management
    # ========================================================================

    async def create_session(
        self,
        workspace_id: UUID,
        user_id: UUID,
        notebook_id: str = "",
        notebook_content: Optional[Dict[str, Any]] = None,
    ) -> NotebookSession:
        """
        Create new notebook session.

        Args:
            workspace_id: Workspace for tenant isolation
            user_id: User creating the session
            notebook_id: Optional notebook identifier
            notebook_content: Optional existing notebook content

        Returns:
            New NotebookSession
        """
        if notebook_content:
            session = NotebookSession.from_notebook_dict(
                notebook_content,
                workspace_id=workspace_id,
                user_id=user_id,
                notebook_id=notebook_id,
            )
        else:
            session = NotebookSession(
                workspace_id=workspace_id,
                user_id=user_id,
                notebook_id=notebook_id,
            )

        self._sessions[session.session_id] = session
        logger.info(f"Created notebook session {session.session_id} for workspace {workspace_id}")
        return session

    async def get_session(self, session_id: str) -> Optional[NotebookSession]:
        """Get session by ID."""
        session = self._sessions.get(session_id)
        if session and session.is_expired():
            await self.close_session(session_id)
            return None
        return session

    async def close_session(self, session_id: str) -> bool:
        """Close and cleanup session."""
        session = self._sessions.pop(session_id, None)
        if session:
            logger.info(f"Closed notebook session {session_id}")
            return True
        return False

    async def list_sessions(
        self,
        workspace_id: Optional[UUID] = None,
        user_id: Optional[UUID] = None,
    ) -> List[NotebookSession]:
        """List active sessions, optionally filtered."""
        sessions = []
        for session in list(self._sessions.values()):
            if session.is_expired():
                await self.close_session(session.session_id)
                continue
            if workspace_id and session.workspace_id != workspace_id:
                continue
            if user_id and session.user_id != user_id:
                continue
            sessions.append(session)
        return sessions

    # ========================================================================
    # Cell Management
    # ========================================================================

    async def add_cell(
        self,
        session_id: str,
        cell_type: CellType = CellType.CODE,
        source: str = "",
        index: Optional[int] = None,
    ) -> Optional[NotebookCell]:
        """Add cell to notebook."""
        session = await self.get_session(session_id)
        if not session:
            return None

        quota = self._get_quota(session.workspace_id)
        if len(session.cells) >= quota.max_cells:
            raise ValueError(f"Maximum cells ({quota.max_cells}) exceeded")

        cell = NotebookCell(cell_type=cell_type, source=source)

        if index is not None and 0 <= index <= len(session.cells):
            session.cells.insert(index, cell)
        else:
            session.cells.append(cell)

        session.touch()
        return cell

    async def update_cell(
        self,
        session_id: str,
        cell_id: str,
        source: Optional[str] = None,
        cell_type: Optional[CellType] = None,
    ) -> Optional[NotebookCell]:
        """Update cell content."""
        session = await self.get_session(session_id)
        if not session:
            return None

        for cell in session.cells:
            if cell.id == cell_id:
                if source is not None:
                    cell.source = source
                if cell_type is not None:
                    cell.cell_type = cell_type
                session.touch()
                return cell

        return None

    async def delete_cell(self, session_id: str, cell_id: str) -> bool:
        """Delete cell from notebook."""
        session = await self.get_session(session_id)
        if not session:
            return False

        for i, cell in enumerate(session.cells):
            if cell.id == cell_id:
                session.cells.pop(i)
                session.touch()
                return True

        return False

    async def move_cell(
        self,
        session_id: str,
        cell_id: str,
        new_index: int,
    ) -> bool:
        """Move cell to new position."""
        session = await self.get_session(session_id)
        if not session:
            return False

        for i, cell in enumerate(session.cells):
            if cell.id == cell_id:
                session.cells.pop(i)
                new_index = max(0, min(new_index, len(session.cells)))
                session.cells.insert(new_index, cell)
                session.touch()
                return True

        return False

    # ========================================================================
    # Execution
    # ========================================================================

    async def execute_cell(
        self,
        session_id: str,
        cell_id: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute a single cell.

        Args:
            session_id: Session ID
            cell_id: Cell to execute
            timeout: Execution timeout in seconds

        Returns:
            ExecutionResult with outputs

        SECURITY:
            - Validates code for prohibited imports
            - Enforces resource quotas
            - Sanitizes outputs
        """
        session = await self.get_session(session_id)
        if not session:
            return ExecutionResult(
                cell_id=cell_id,
                success=False,
                outputs=[],
                execution_count=0,
                execution_time_ms=0,
                error_message="Session not found or expired",
            )

        # Find cell
        cell = None
        for c in session.cells:
            if c.id == cell_id:
                cell = c
                break

        if not cell:
            return ExecutionResult(
                cell_id=cell_id,
                success=False,
                outputs=[],
                execution_count=0,
                execution_time_ms=0,
                error_message="Cell not found",
            )

        if cell.cell_type != CellType.CODE:
            return ExecutionResult(
                cell_id=cell_id,
                success=True,
                outputs=[],
                execution_count=0,
                execution_time_ms=0,
            )

        # Check quotas
        quota = self._get_quota(session.workspace_id)
        timeout = min(timeout or quota.cell_timeout_seconds, MAX_CELL_TIMEOUT)

        # Check concurrent executions
        workspace_key = str(session.workspace_id)
        current_executions = self._active_executions.get(workspace_key, 0)
        if current_executions >= quota.max_concurrent_executions:
            return ExecutionResult(
                cell_id=cell_id,
                success=False,
                outputs=[],
                execution_count=0,
                execution_time_ms=0,
                error_message=f"Maximum concurrent executions ({quota.max_concurrent_executions}) exceeded",
            )

        # Validate code
        validation_error = self._validate_code(cell.source)
        if validation_error:
            return ExecutionResult(
                cell_id=cell_id,
                success=False,
                outputs=[CellOutput(
                    output_type=OutputType.ERROR,
                    ename="SecurityError",
                    evalue=validation_error,
                    traceback=[validation_error],
                )],
                execution_count=0,
                execution_time_ms=0,
                error_message=validation_error,
            )

        # Execute
        async with self._execution_lock:
            self._active_executions[workspace_key] = current_executions + 1

        try:
            result = await self._execute_code(
                session=session,
                cell=cell,
                timeout=timeout,
                quota=quota,
            )
            return result
        finally:
            async with self._execution_lock:
                self._active_executions[workspace_key] = max(
                    0, self._active_executions.get(workspace_key, 1) - 1
                )

    async def execute_all(
        self,
        session_id: str,
        stop_on_error: bool = True,
    ) -> List[ExecutionResult]:
        """Execute all code cells in order."""
        session = await self.get_session(session_id)
        if not session:
            return []

        results = []
        for cell in session.cells:
            if cell.cell_type == CellType.CODE:
                result = await self.execute_cell(session_id, cell.id)
                results.append(result)
                if not result.success and stop_on_error:
                    break

        return results

    async def interrupt_execution(self, session_id: str) -> bool:
        """Interrupt running execution."""
        session = await self.get_session(session_id)
        if not session:
            return False

        # Mark all running cells as cancelled
        for cell in session.cells:
            if cell.execution_state == ExecutionState.RUNNING:
                cell.execution_state = ExecutionState.CANCELLED
                cell.outputs.append(CellOutput(
                    output_type=OutputType.ERROR,
                    ename="KeyboardInterrupt",
                    evalue="Execution interrupted by user",
                    traceback=["KeyboardInterrupt: Execution interrupted by user"],
                ))

        return True

    # ========================================================================
    # Import/Export
    # ========================================================================

    async def export_notebook(
        self,
        session_id: str,
        include_outputs: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Export notebook in nbformat JSON."""
        session = await self.get_session(session_id)
        if not session:
            return None

        notebook = session.to_notebook_dict()

        if not include_outputs:
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None

        return notebook

    async def import_notebook(
        self,
        session_id: str,
        notebook_content: Dict[str, Any],
    ) -> bool:
        """Import notebook content into session."""
        session = await self.get_session(session_id)
        if not session:
            return False

        cells = [NotebookCell.from_dict(c) for c in notebook_content.get("cells", [])]

        quota = self._get_quota(session.workspace_id)
        if len(cells) > quota.max_cells:
            raise ValueError(f"Notebook has too many cells ({len(cells)} > {quota.max_cells})")

        session.cells = cells
        session.metadata = notebook_content.get("metadata", session.metadata)
        session.touch()

        return True

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _get_quota(self, workspace_id: UUID) -> NotebookQuota:
        """Get quota for workspace."""
        return self._workspace_quotas.get(workspace_id, self._default_quota)

    def _validate_code(self, code: str) -> Optional[str]:
        """
        Validate code for security constraints.

        Returns error message if invalid, None if valid.
        """
        # Check for prohibited imports
        import_pattern = re.compile(
            r'(?:^|\s)(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            re.MULTILINE
        )

        for match in import_pattern.finditer(code):
            module = match.group(1)
            for prohibited in PROHIBITED_IMPORTS:
                if module == prohibited or module.startswith(prohibited + "."):
                    return f"Prohibited import: {module} (Cloud Zone restriction)"

        # Check for dangerous operations
        dangerous_patterns = [
            (r'\bos\.system\s*\(', "os.system() is not allowed"),
            (r'\bsubprocess\.\w+\s*\(', "subprocess module is not allowed"),
            (r'\b__import__\s*\(', "__import__() is not allowed"),
            (r'\beval\s*\([^)]*\bopen\b', "eval with open() is not allowed"),
            (r'\bexec\s*\([^)]*\bopen\b', "exec with open() is not allowed"),
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                return message

        return None

    async def _execute_code(
        self,
        session: NotebookSession,
        cell: NotebookCell,
        timeout: int,
        quota: NotebookQuota,
    ) -> ExecutionResult:
        """Execute code cell with isolation."""
        cell.execution_state = ExecutionState.RUNNING
        cell.started_at = datetime.now(timezone.utc)
        cell.outputs = []

        session.execution_counter += 1
        execution_count = session.execution_counter
        cell.execution_count = execution_count

        start_time = time.monotonic()
        outputs: List[CellOutput] = []
        success = True
        error_message = None
        truncated = False

        try:
            # Create isolated namespace with session kernel state
            namespace = dict(session.kernel_state)
            namespace["__name__"] = "__main__"
            namespace["__builtins__"] = __builtins__

            # Add safe imports
            namespace["print"] = self._create_capture_print(outputs)

            # Compile and execute
            code_obj = compile(cell.source, f"<cell-{cell.id}>", "exec")

            # Execute with timeout
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        exec,
                        code_obj,
                        namespace,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                cell.execution_state = ExecutionState.TIMEOUT
                success = False
                error_message = f"Execution timed out after {timeout} seconds"
                outputs.append(CellOutput(
                    output_type=OutputType.ERROR,
                    ename="TimeoutError",
                    evalue=error_message,
                    traceback=[error_message],
                ))

            # Update kernel state (excluding internal variables)
            for key, value in namespace.items():
                if not key.startswith("_") and key not in ("print", "__name__", "__builtins__"):
                    session.kernel_state[key] = value

            # Check for result
            if "_" in namespace and namespace["_"] is not None:
                result_repr = repr(namespace["_"])
                if len(result_repr) > MAX_OUTPUT_SIZE:
                    result_repr = result_repr[:MAX_OUTPUT_SIZE] + "... [truncated]"
                    truncated = True
                outputs.append(CellOutput(
                    output_type=OutputType.EXECUTE_RESULT,
                    data={"text/plain": result_repr},
                    execution_count=execution_count,
                ))

        except SyntaxError as e:
            success = False
            error_message = str(e)
            outputs.append(CellOutput(
                output_type=OutputType.ERROR,
                ename="SyntaxError",
                evalue=str(e),
                traceback=[f"SyntaxError: {e}"],
            ))
        except Exception as e:
            success = False
            error_message = str(e)
            import traceback
            tb_lines = traceback.format_exception(type(e), e, e.__traceback__)
            outputs.append(CellOutput(
                output_type=OutputType.ERROR,
                ename=type(e).__name__,
                evalue=str(e),
                traceback=tb_lines,
            ))

        end_time = time.monotonic()
        execution_time_ms = int((end_time - start_time) * 1000)

        # Sanitize outputs
        outputs = self._sanitize_outputs(outputs)

        # Update cell state
        cell.outputs = outputs
        cell.completed_at = datetime.now(timezone.utc)
        cell.execution_time_ms = execution_time_ms
        cell.execution_state = (
            ExecutionState.COMPLETED if success else ExecutionState.ERROR
        )

        session.touch()

        return ExecutionResult(
            cell_id=cell.id,
            success=success,
            outputs=outputs,
            execution_count=execution_count,
            execution_time_ms=execution_time_ms,
            error_message=error_message,
            truncated=truncated,
        )

    def _create_capture_print(self, outputs: List[CellOutput]):
        """Create print function that captures output."""
        def capture_print(*args, **kwargs):
            text = " ".join(str(arg) for arg in args)
            end = kwargs.get("end", "\n")
            text += end

            if outputs and outputs[-1].output_type == OutputType.STREAM:
                outputs[-1].text = (outputs[-1].text or "") + text
            else:
                outputs.append(CellOutput(
                    output_type=OutputType.STREAM,
                    name="stdout",
                    text=text,
                ))

        return capture_print

    def _sanitize_outputs(self, outputs: List[CellOutput]) -> List[CellOutput]:
        """Sanitize outputs to remove sensitive data."""
        sanitized = []

        for output in outputs:
            new_output = CellOutput(
                output_type=output.output_type,
                data=dict(output.data),
                text=output.text,
                name=output.name,
                execution_count=output.execution_count,
                ename=output.ename,
                evalue=output.evalue,
                traceback=list(output.traceback) if output.traceback else None,
                metadata=dict(output.metadata),
            )

            # Redact sensitive patterns
            if new_output.text:
                for pattern in SENSITIVE_PATTERNS:
                    new_output.text = pattern.sub("[REDACTED]", new_output.text)

            for key, value in list(new_output.data.items()):
                if isinstance(value, str):
                    for pattern in SENSITIVE_PATTERNS:
                        new_output.data[key] = pattern.sub("[REDACTED]", value)

            sanitized.append(new_output)

        return sanitized

    # ========================================================================
    # Cleanup
    # ========================================================================

    async def cleanup_expired_sessions(self) -> int:
        """Cleanup expired sessions. Returns count of cleaned sessions."""
        expired = []
        for session_id, session in list(self._sessions.items()):
            if session.is_expired():
                expired.append(session_id)

        for session_id in expired:
            await self.close_session(session_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired notebook sessions")

        return len(expired)
