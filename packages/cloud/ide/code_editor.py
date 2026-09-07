# -*- coding: utf-8 -*-
"""
Code Editor Service - CLOUD ZONE ONLY.

Web-based code editing with syntax highlighting and real-time collaboration.

Design Doc Reference:
    - Section 4.1: "Strategy Workspace (IDE/ноутбуки, репо стратегий, версии)"

Based on:
    - Monaco Editor architecture
    - Language Server Protocol (LSP)
    - Operational Transformation for collaboration
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Set, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Maximum file size for editing (bytes)
MAX_FILE_SIZE: Final[int] = 5 * 1024 * 1024  # 5MB

# Maximum concurrent editors per file
MAX_CONCURRENT_EDITORS: Final[int] = 10

# Session timeout (minutes)
EDITOR_SESSION_TIMEOUT: Final[int] = 60

# Supported languages
SUPPORTED_LANGUAGES: Final[frozenset] = frozenset(
    {
        "python",
        "yaml",
        "json",
        "markdown",
        "toml",
        "ini",
        "shell",
        "dockerfile",
    }
)


# ============================================================================
# Enums
# ============================================================================


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"
    TOML = "toml"
    INI = "ini"
    SHELL = "shell"
    DOCKERFILE = "dockerfile"
    UNKNOWN = "unknown"


class ChangeType(str, Enum):
    """Type of file change."""

    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"


class DiagnosticSeverity(str, Enum):
    """Diagnostic message severity (LSP-compatible)."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Position:
    """Position in a text document (0-indexed)."""

    line: int
    character: int

    def to_dict(self) -> Dict[str, int]:
        return {"line": self.line, "character": self.character}

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "Position":
        return cls(line=data["line"], character=data["character"])


@dataclass
class Range:
    """Range in a text document."""

    start: Position
    end: Position

    def to_dict(self) -> Dict[str, Any]:
        return {"start": self.start.to_dict(), "end": self.end.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Range":
        return cls(
            start=Position.from_dict(data["start"]),
            end=Position.from_dict(data["end"]),
        )


@dataclass
class FileChange:
    """
    A single change to a file.

    Supports Operational Transformation for collaborative editing.
    """

    change_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    change_type: ChangeType = ChangeType.INSERT
    range: Range = field(default_factory=lambda: Range(Position(0, 0), Position(0, 0)))
    text: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[UUID] = None
    version: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "change_id": self.change_id,
            "change_type": self.change_type.value,
            "range": self.range.to_dict(),
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "user_id": str(self.user_id) if self.user_id else None,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileChange":
        return cls(
            change_id=data.get("change_id", str(uuid.uuid4())),
            change_type=ChangeType(data["change_type"]),
            range=Range.from_dict(data["range"]),
            text=data.get("text", ""),
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now(timezone.utc)
            ),
            user_id=UUID(data["user_id"]) if data.get("user_id") else None,
            version=data.get("version", 0),
        )


@dataclass
class Diagnostic:
    """Diagnostic message (error, warning, etc.)."""

    range: Range
    severity: DiagnosticSeverity
    message: str
    source: str = "ccea"
    code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "range": self.range.to_dict(),
            "severity": self.severity.value,
            "message": self.message,
            "source": self.source,
            "code": self.code,
        }


@dataclass
class CompletionItem:
    """Code completion suggestion."""

    label: str
    kind: str = "text"  # text, function, class, variable, etc.
    detail: Optional[str] = None
    documentation: Optional[str] = None
    insert_text: Optional[str] = None
    sort_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "detail": self.detail,
            "documentation": self.documentation,
            "insertText": self.insert_text or self.label,
            "sortText": self.sort_text,
        }


@dataclass
class EditorSession:
    """
    Active code editor session.

    Tracks file content, changes, and collaborative cursors.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: UUID = field(default_factory=uuid.uuid4)
    user_id: UUID = field(default_factory=uuid.uuid4)
    file_path: str = ""
    language: Language = Language.PYTHON

    # Content
    content: str = ""
    version: int = 0
    content_hash: str = ""

    # Change tracking
    pending_changes: List[FileChange] = field(default_factory=list)
    applied_changes: List[FileChange] = field(default_factory=list)

    # Cursor positions (user_id -> position)
    cursors: Dict[str, Position] = field(default_factory=dict)

    # Diagnostics
    diagnostics: List[Diagnostic] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_dirty: bool = False

    def is_expired(self) -> bool:
        """Check if session has expired."""
        timeout = timedelta(minutes=EDITOR_SESSION_TIMEOUT)
        return datetime.now(timezone.utc) - self.last_activity > timeout

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def compute_hash(self) -> str:
        """Compute content hash."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def update_hash(self) -> None:
        """Update stored hash from current content."""
        self.content_hash = self.compute_hash()


@dataclass
class SyntaxToken:
    """Token for syntax highlighting."""

    start: int
    end: int
    token_type: str  # keyword, string, comment, number, etc.
    modifiers: List[str] = field(default_factory=list)


# ============================================================================
# Syntax Highlighter
# ============================================================================


class SyntaxHighlighter:
    """
    Syntax highlighting for supported languages.

    Provides token-based highlighting compatible with Monaco/VSCode themes.
    """

    # Python syntax patterns
    PYTHON_PATTERNS = [
        (
            r"\b(def|class|if|elif|else|for|while|try|except|finally|with|as|import|from|return|yield|raise|break|continue|pass|lambda|and|or|not|in|is|True|False|None|async|await)\b",
            "keyword",
        ),
        (r"\b(self|cls)\b", "variable.language"),
        (r"@\w+", "decorator"),
        (r"#.*$", "comment"),
        (r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "string.docstring"),
        (r'"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'', "string"),
        (r"\b\d+\.?\d*(?:[eE][+-]?\d+)?\b", "number"),
        (r"\b[A-Z][A-Z0-9_]*\b", "constant"),
        (r"\b[A-Z][a-zA-Z0-9]*\b", "type"),
        (r"\b__\w+__\b", "variable.special"),
    ]

    # YAML syntax patterns
    YAML_PATTERNS = [
        (r"#.*$", "comment"),
        (r"^[\s]*[\w_-]+(?=\s*:)", "property"),
        (r":\s*$", "punctuation"),
        (r'"[^"]*"|\'[^\']*\'', "string"),
        (r"\b(true|false|null|yes|no|on|off)\b", "keyword"),
        (r"\b\d+\.?\d*\b", "number"),
        (r"^\s*-\s", "punctuation"),
    ]

    # JSON syntax patterns
    JSON_PATTERNS = [
        (r'"[^"\\]*(?:\\.[^"\\]*)*"\s*:', "property"),
        (r'"[^"\\]*(?:\\.[^"\\]*)*"', "string"),
        (r"\b(true|false|null)\b", "keyword"),
        (r"-?\b\d+\.?\d*(?:[eE][+-]?\d+)?\b", "number"),
    ]

    def __init__(self):
        self._compiled_patterns: Dict[Language, List[Tuple[re.Pattern, str]]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for performance."""
        self._compiled_patterns[Language.PYTHON] = [
            (re.compile(pattern, re.MULTILINE), token_type)
            for pattern, token_type in self.PYTHON_PATTERNS
        ]
        self._compiled_patterns[Language.YAML] = [
            (re.compile(pattern, re.MULTILINE), token_type)
            for pattern, token_type in self.YAML_PATTERNS
        ]
        self._compiled_patterns[Language.JSON] = [
            (re.compile(pattern, re.MULTILINE), token_type)
            for pattern, token_type in self.JSON_PATTERNS
        ]

    def tokenize(self, content: str, language: Language) -> List[SyntaxToken]:
        """
        Tokenize content for syntax highlighting.

        Args:
            content: Source code content
            language: Programming language

        Returns:
            List of syntax tokens
        """
        tokens: List[SyntaxToken] = []
        patterns = self._compiled_patterns.get(language, [])

        if not patterns:
            return tokens

        # Find all matches
        matches: List[Tuple[int, int, str]] = []
        for pattern, token_type in patterns:
            for match in pattern.finditer(content):
                matches.append((match.start(), match.end(), token_type))

        # Sort by position and remove overlaps
        matches.sort(key=lambda x: (x[0], -x[1]))
        filtered: List[Tuple[int, int, str]] = []
        last_end = 0

        for start, end, token_type in matches:
            if start >= last_end:
                filtered.append((start, end, token_type))
                last_end = end

        # Convert to tokens
        tokens = [
            SyntaxToken(start=start, end=end, token_type=token_type)
            for start, end, token_type in filtered
        ]

        return tokens

    def get_semantic_tokens(
        self,
        content: str,
        language: Language,
    ) -> Dict[str, Any]:
        """
        Get semantic tokens in LSP format.

        Returns data compatible with Monaco's setSemanticTokens.
        """
        tokens = self.tokenize(content, language)

        # Convert to line-relative format
        data: List[int] = []
        prev_line = 0
        prev_char = 0

        lines = content.split("\n")
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line) + 1)

        for token in tokens:
            # Find line number
            line_num = 0
            for i, offset in enumerate(line_offsets[:-1]):
                if offset <= token.start < line_offsets[i + 1]:
                    line_num = i
                    break

            char = token.start - line_offsets[line_num]
            length = token.end - token.start

            # Delta encoding
            delta_line = line_num - prev_line
            delta_char = char if delta_line > 0 else char - prev_char

            # Token type index (simplified)
            type_index = self._get_token_type_index(token.token_type)

            data.extend([delta_line, delta_char, length, type_index, 0])

            prev_line = line_num
            prev_char = char

        return {
            "data": data,
            "resultId": str(uuid.uuid4()),
        }

    def _get_token_type_index(self, token_type: str) -> int:
        """Map token type to index."""
        type_map = {
            "keyword": 0,
            "string": 1,
            "comment": 2,
            "number": 3,
            "type": 4,
            "variable": 5,
            "function": 6,
            "property": 7,
            "decorator": 8,
            "constant": 9,
        }
        base_type = token_type.split(".")[0]
        return type_map.get(base_type, 5)


# ============================================================================
# Code Editor Service
# ============================================================================


class CodeEditorService:
    """
    Code editor service with collaborative editing support.

    Provides:
    - Session management
    - Real-time content synchronization
    - Syntax highlighting
    - Basic code intelligence (completion, diagnostics)

    SECURITY: Cloud Zone only - no access to secrets or broker APIs.
    """

    def __init__(self):
        self._sessions: Dict[str, EditorSession] = {}
        self._file_sessions: Dict[str, Set[str]] = {}  # file_path -> session_ids
        self._highlighter = SyntaxHighlighter()
        self._lock = asyncio.Lock()

        logger.info("CodeEditorService initialized (Cloud Zone)")

    # ========================================================================
    # Session Management
    # ========================================================================

    async def create_session(
        self,
        workspace_id: UUID,
        user_id: UUID,
        file_path: str,
        content: str = "",
        language: Optional[Language] = None,
    ) -> EditorSession:
        """Create new editor session."""
        if len(content) > MAX_FILE_SIZE:
            raise ValueError(f"File too large (max {MAX_FILE_SIZE} bytes)")

        # Detect language from extension
        if language is None:
            language = self._detect_language(file_path)

        session = EditorSession(
            workspace_id=workspace_id,
            user_id=user_id,
            file_path=file_path,
            language=language,
            content=content,
        )
        session.update_hash()

        async with self._lock:
            # Check concurrent editors limit
            existing = self._file_sessions.get(file_path, set())
            if len(existing) >= MAX_CONCURRENT_EDITORS:
                raise ValueError(f"Maximum concurrent editors ({MAX_CONCURRENT_EDITORS}) reached")

            self._sessions[session.session_id] = session
            if file_path not in self._file_sessions:
                self._file_sessions[file_path] = set()
            self._file_sessions[file_path].add(session.session_id)

        logger.info(f"Created editor session {session.session_id} for {file_path}")
        return session

    async def get_session(self, session_id: str) -> Optional[EditorSession]:
        """Get session by ID."""
        session = self._sessions.get(session_id)
        if session and session.is_expired():
            await self.close_session(session_id)
            return None
        return session

    async def close_session(self, session_id: str) -> bool:
        """Close editor session."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                if session.file_path in self._file_sessions:
                    self._file_sessions[session.file_path].discard(session_id)
                    if not self._file_sessions[session.file_path]:
                        del self._file_sessions[session.file_path]
                logger.info(f"Closed editor session {session_id}")
                return True
        return False

    async def get_sessions_for_file(self, file_path: str) -> List[EditorSession]:
        """Get all active sessions for a file."""
        session_ids = self._file_sessions.get(file_path, set())
        sessions = []
        for sid in list(session_ids):
            session = await self.get_session(sid)
            if session:
                sessions.append(session)
        return sessions

    # ========================================================================
    # Content Operations
    # ========================================================================

    async def apply_change(
        self,
        session_id: str,
        change: FileChange,
    ) -> Optional[EditorSession]:
        """Apply a change to the document."""
        session = await self.get_session(session_id)
        if not session:
            return None

        # Apply the change
        content = session.content
        lines = content.split("\n")

        start = change.range.start
        end = change.range.end

        # Compute character offsets
        start_offset = sum(len(lines[i]) + 1 for i in range(start.line)) + start.character
        end_offset = sum(len(lines[i]) + 1 for i in range(end.line)) + end.character

        if change.change_type == ChangeType.INSERT:
            new_content = content[:start_offset] + change.text + content[start_offset:]
        elif change.change_type == ChangeType.DELETE:
            new_content = content[:start_offset] + content[end_offset:]
        elif change.change_type == ChangeType.REPLACE:
            new_content = content[:start_offset] + change.text + content[end_offset:]
        else:
            new_content = content

        # Update session
        session.content = new_content
        session.version += 1
        change.version = session.version
        session.applied_changes.append(change)
        session.is_dirty = True
        session.touch()
        session.update_hash()

        # Broadcast to other sessions on same file
        await self._broadcast_change(session.file_path, session_id, change)

        return session

    async def set_content(
        self,
        session_id: str,
        content: str,
    ) -> Optional[EditorSession]:
        """Set entire file content."""
        session = await self.get_session(session_id)
        if not session:
            return None

        if len(content) > MAX_FILE_SIZE:
            raise ValueError(f"Content too large (max {MAX_FILE_SIZE} bytes)")

        session.content = content
        session.version += 1
        session.is_dirty = True
        session.touch()
        session.update_hash()

        return session

    async def get_content(self, session_id: str) -> Optional[str]:
        """Get current file content."""
        session = await self.get_session(session_id)
        return session.content if session else None

    async def update_cursor(
        self,
        session_id: str,
        position: Position,
    ) -> None:
        """Update cursor position for collaborative display."""
        session = await self.get_session(session_id)
        if session:
            user_key = str(session.user_id)
            session.cursors[user_key] = position
            session.touch()

    # ========================================================================
    # Code Intelligence
    # ========================================================================

    async def get_completions(
        self,
        session_id: str,
        position: Position,
    ) -> List[CompletionItem]:
        """Get code completion suggestions."""
        session = await self.get_session(session_id)
        if not session:
            return []

        completions: List[CompletionItem] = []

        if session.language == Language.PYTHON:
            completions = self._get_python_completions(session.content, position)

        return completions

    async def get_diagnostics(
        self,
        session_id: str,
    ) -> List[Diagnostic]:
        """Get diagnostic messages for the file."""
        session = await self.get_session(session_id)
        if not session:
            return []

        diagnostics: List[Diagnostic] = []

        if session.language == Language.PYTHON:
            diagnostics = self._check_python_syntax(session.content)

        session.diagnostics = diagnostics
        return diagnostics

    async def get_tokens(self, session_id: str) -> List[SyntaxToken]:
        """Get syntax highlighting tokens."""
        session = await self.get_session(session_id)
        if not session:
            return []

        return self._highlighter.tokenize(session.content, session.language)

    async def get_semantic_tokens(self, session_id: str) -> Dict[str, Any]:
        """Get semantic tokens in LSP format."""
        session = await self.get_session(session_id)
        if not session:
            return {"data": [], "resultId": ""}

        return self._highlighter.get_semantic_tokens(session.content, session.language)

    # ========================================================================
    # Private Methods
    # ========================================================================

    def _detect_language(self, file_path: str) -> Language:
        """Detect language from file extension."""
        ext_map = {
            ".py": Language.PYTHON,
            ".yaml": Language.YAML,
            ".yml": Language.YAML,
            ".json": Language.JSON,
            ".md": Language.MARKDOWN,
            ".toml": Language.TOML,
            ".ini": Language.INI,
            ".sh": Language.SHELL,
            ".bash": Language.SHELL,
            "Dockerfile": Language.DOCKERFILE,
        }

        for ext, lang in ext_map.items():
            if file_path.endswith(ext) or file_path.endswith(ext.lower()):
                return lang

        return Language.UNKNOWN

    def _get_python_completions(
        self,
        content: str,
        position: Position,
    ) -> List[CompletionItem]:
        """Get Python code completions."""
        completions: List[CompletionItem] = []

        # Get the current line and word being typed
        lines = content.split("\n")
        if position.line >= len(lines):
            return completions

        line = lines[position.line][: position.character]

        # Extract word prefix
        word_match = re.search(r"(\w+)$", line)
        prefix = word_match.group(1).lower() if word_match else ""

        # Built-in completions
        builtins = [
            ("print", "function", "Print to stdout"),
            ("len", "function", "Return length of object"),
            ("range", "function", "Return range iterator"),
            ("list", "class", "List type"),
            ("dict", "class", "Dictionary type"),
            ("str", "class", "String type"),
            ("int", "class", "Integer type"),
            ("float", "class", "Float type"),
            ("True", "keyword", "Boolean true"),
            ("False", "keyword", "Boolean false"),
            ("None", "keyword", "Null value"),
            ("if", "keyword", "Conditional statement"),
            ("for", "keyword", "For loop"),
            ("while", "keyword", "While loop"),
            ("def", "keyword", "Function definition"),
            ("class", "keyword", "Class definition"),
            ("import", "keyword", "Import statement"),
            ("from", "keyword", "Import from"),
            ("return", "keyword", "Return statement"),
            ("async", "keyword", "Async function"),
            ("await", "keyword", "Await expression"),
        ]

        for label, kind, doc in builtins:
            if label.lower().startswith(prefix):
                completions.append(
                    CompletionItem(
                        label=label,
                        kind=kind,
                        documentation=doc,
                    )
                )

        # Add variables from content
        var_pattern = re.compile(r"\b([a-z_][a-z0-9_]*)\s*=", re.IGNORECASE)
        for match in var_pattern.finditer(content):
            var_name = match.group(1)
            if var_name.lower().startswith(prefix) and var_name not in [
                c.label for c in completions
            ]:
                completions.append(
                    CompletionItem(
                        label=var_name,
                        kind="variable",
                    )
                )

        return completions[:50]  # Limit suggestions

    def _check_python_syntax(self, content: str) -> List[Diagnostic]:
        """Check Python syntax and return diagnostics."""
        diagnostics: List[Diagnostic] = []

        try:
            compile(content, "<string>", "exec")
        except SyntaxError as e:
            line = (e.lineno or 1) - 1
            col = (e.offset or 1) - 1
            diagnostics.append(
                Diagnostic(
                    range=Range(
                        start=Position(line=line, character=col),
                        end=Position(line=line, character=col + 1),
                    ),
                    severity=DiagnosticSeverity.ERROR,
                    message=str(e.msg),
                    source="python",
                )
            )

        # Check for common issues
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Trailing whitespace
            if line.rstrip() != line and line.strip():
                diagnostics.append(
                    Diagnostic(
                        range=Range(
                            start=Position(line=i, character=len(line.rstrip())),
                            end=Position(line=i, character=len(line)),
                        ),
                        severity=DiagnosticSeverity.HINT,
                        message="Trailing whitespace",
                        source="style",
                    )
                )

            # Line too long
            if len(line) > 120:
                diagnostics.append(
                    Diagnostic(
                        range=Range(
                            start=Position(line=i, character=120),
                            end=Position(line=i, character=len(line)),
                        ),
                        severity=DiagnosticSeverity.WARNING,
                        message="Line too long (>120 characters)",
                        source="style",
                    )
                )

        return diagnostics

    async def _broadcast_change(
        self,
        file_path: str,
        source_session_id: str,
        change: FileChange,
    ) -> None:
        """Broadcast change to other sessions editing same file."""
        session_ids = self._file_sessions.get(file_path, set())
        for sid in session_ids:
            if sid != source_session_id:
                session = self._sessions.get(sid)
                if session:
                    session.pending_changes.append(change)

    # ========================================================================
    # Cleanup
    # ========================================================================

    async def cleanup_expired_sessions(self) -> int:
        """Cleanup expired sessions."""
        expired = []
        for session_id, session in list(self._sessions.items()):
            if session.is_expired():
                expired.append(session_id)

        for session_id in expired:
            await self.close_session(session_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired editor sessions")

        return len(expired)
