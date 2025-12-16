# -*- coding: utf-8 -*-
"""
Strategy IDE Package - CLOUD ZONE ONLY.

Web-based IDE for strategy development, Jupyter notebooks, and code editing.

Design Doc Reference:
    - Section 4.1: "Strategy Workspace (IDE/ноутбуки, репо стратегий, версии)"
    - Section 3.A: "Research / Validate / Stress-test / Sim (cloud)"

Components:
    - notebook_service: Jupyter notebook execution and management
    - code_editor: Web-based code editing with syntax highlighting
    - file_manager: Virtual filesystem for strategy files
    - session_manager: User session and workspace isolation
"""

from .notebook_service import (
    NotebookService,
    NotebookSession,
    NotebookCell,
    CellType,
    CellOutput,
    ExecutionState,
    OutputType,
)
from .code_editor import (
    CodeEditorService,
    EditorSession,
    FileChange,
    SyntaxHighlighter,
)
from .file_manager import (
    VirtualFileSystem,
    StrategyFile,
    FilePermission,
    FileType,
)
from .session_manager import (
    IDESessionManager,
    IDESession,
    SessionConfig,
    SessionState,
)

__all__ = [
    # Notebook
    "NotebookService",
    "NotebookSession",
    "NotebookCell",
    "CellType",
    "CellOutput",
    "ExecutionState",
    "OutputType",
    # Editor
    "CodeEditorService",
    "EditorSession",
    "FileChange",
    "SyntaxHighlighter",
    # File system
    "VirtualFileSystem",
    "StrategyFile",
    "FilePermission",
    "FileType",
    # Sessions
    "IDESessionManager",
    "IDESession",
    "SessionConfig",
    "SessionState",
]
