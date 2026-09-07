# -*- coding: utf-8 -*-
"""
Tests for IDE Package.

Tests for:
- NotebookService
- CodeEditorService
- VirtualFileSystem
- IDESessionManager
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from packages.cloud.ide import (
    NotebookService,
    NotebookSession,
    NotebookCell,
    CellType,
    CellOutput,
    CodeEditorService,
    EditorSession,
    VirtualFileSystem,
    StrategyFile,
    FileType,
    IDESessionManager,
    IDESession,
    SessionConfig,
    SessionState,
)


# ============================================================================
# NotebookService Tests
# ============================================================================


class TestNotebookService:
    """Tests for NotebookService."""

    @pytest.fixture
    def service(self):
        """Create NotebookService instance."""
        return NotebookService()

    @pytest.fixture
    def workspace_id(self):
        """Create test workspace ID."""
        return uuid4()

    @pytest.fixture
    def user_id(self):
        """Create test user ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_create_session(self, service, workspace_id, user_id):
        """Test creating a notebook session."""
        session = await service.create_session(workspace_id, user_id)

        assert session is not None
        assert session.workspace_id == workspace_id
        assert session.user_id == user_id
        assert len(session.cells) == 0

    @pytest.mark.asyncio
    async def test_add_cell(self, service, workspace_id, user_id):
        """Test adding cells to notebook."""
        session = await service.create_session(workspace_id, user_id)

        # Add code cell
        cell = await service.add_cell(session.session_id, CellType.CODE, "print('hello world')")

        assert cell is not None
        assert cell.cell_type == CellType.CODE
        assert cell.source == "print('hello world')"
        assert len(session.cells) == 1

    @pytest.mark.asyncio
    async def test_add_markdown_cell(self, service, workspace_id, user_id):
        """Test adding markdown cell."""
        session = await service.create_session(workspace_id, user_id)

        cell = await service.add_cell(
            session.session_id, CellType.MARKDOWN, "# Header\n\nSome text"
        )

        assert cell.cell_type == CellType.MARKDOWN

    @pytest.mark.asyncio
    async def test_execute_cell_simple(self, service, workspace_id, user_id):
        """Test executing a simple code cell."""
        session = await service.create_session(workspace_id, user_id)
        cell = await service.add_cell(
            session.session_id, CellType.CODE, "result = 2 + 2\nprint(result)"
        )

        output = await service.execute_cell(session.session_id, cell.id)

        assert output is not None
        assert cell.execution_count == 1

    @pytest.mark.asyncio
    async def test_execute_cell_prohibited_import(self, service, workspace_id, user_id):
        """Test execution with prohibited import (os module)."""
        session = await service.create_session(workspace_id, user_id)
        cell = await service.add_cell(
            session.session_id, CellType.CODE, "import os\nos.system('ls')"
        )

        output = await service.execute_cell(session.session_id, cell.id)

        # Execution happens but with restricted builtins
        assert output is not None

    @pytest.mark.asyncio
    async def test_update_cell(self, service, workspace_id, user_id):
        """Test updating cell content."""
        session = await service.create_session(workspace_id, user_id)
        cell = await service.add_cell(session.session_id, CellType.CODE, "original code")

        updated_cell = await service.update_cell(session.session_id, cell.id, "updated code")

        # update_cell returns the updated cell object
        assert updated_cell is not None
        assert updated_cell.source == "updated code"

    @pytest.mark.asyncio
    async def test_delete_cell(self, service, workspace_id, user_id):
        """Test deleting a cell."""
        session = await service.create_session(workspace_id, user_id)
        cell = await service.add_cell(session.session_id, CellType.CODE, "some code")

        deleted = await service.delete_cell(session.session_id, cell.id)

        assert deleted is True
        assert len(session.cells) == 0

    @pytest.mark.asyncio
    async def test_close_session(self, service, workspace_id, user_id):
        """Test closing a session."""
        session = await service.create_session(workspace_id, user_id)

        closed = await service.close_session(session.session_id)

        assert closed is True

    @pytest.mark.asyncio
    async def test_session_isolation(self, service):
        """Test that sessions are isolated by workspace."""
        ws1 = uuid4()
        ws2 = uuid4()
        user = uuid4()

        session1 = await service.create_session(ws1, user)
        session2 = await service.create_session(ws2, user)

        sessions_ws1 = await service.list_sessions(workspace_id=ws1)

        assert len(sessions_ws1) == 1
        assert sessions_ws1[0].session_id == session1.session_id


# ============================================================================
# CodeEditorService Tests
# ============================================================================


class TestCodeEditorService:
    """Tests for CodeEditorService."""

    @pytest.fixture
    def service(self):
        """Create CodeEditorService instance."""
        return CodeEditorService()

    @pytest.fixture
    def workspace_id(self):
        """Create test workspace ID."""
        return uuid4()

    @pytest.fixture
    def user_id(self):
        """Create test user ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_create_session(self, service, workspace_id, user_id):
        """Test creating editor session."""
        session = await service.create_session(
            workspace_id, user_id, "strategy.py", "def run():\n    pass"
        )

        assert session is not None
        assert session.file_path == "strategy.py"
        assert session.content == "def run():\n    pass"

    @pytest.mark.asyncio
    async def test_set_content(self, service, workspace_id, user_id):
        """Test updating editor content."""
        session = await service.create_session(workspace_id, user_id, "test.py", "original")

        updated_session = await service.set_content(session.session_id, "updated content")

        assert updated_session is not None
        assert updated_session.content == "updated content"

    @pytest.mark.asyncio
    async def test_get_completions(self, service, workspace_id, user_id):
        """Test getting code completions."""
        from packages.cloud.ide.code_editor import Position

        session = await service.create_session(
            workspace_id, user_id, "test.py", "import pandas as pd\npd."
        )

        completions = await service.get_completions(
            session.session_id, Position(line=1, character=3)
        )

        assert isinstance(completions, list)

    @pytest.mark.asyncio
    async def test_get_diagnostics(self, service, workspace_id, user_id):
        """Test getting diagnostics."""
        session = await service.create_session(
            workspace_id, user_id, "test.py", "def broken(\n    pass"  # Syntax error
        )

        diagnostics = await service.get_diagnostics(session.session_id)

        assert isinstance(diagnostics, list)

    @pytest.mark.asyncio
    async def test_close_session(self, service, workspace_id, user_id):
        """Test closing editor session."""
        session = await service.create_session(workspace_id, user_id, "test.py", "content")

        closed = await service.close_session(session.session_id)

        assert closed is True


# ============================================================================
# VirtualFileSystem Tests
# ============================================================================


class TestVirtualFileSystem:
    """Tests for VirtualFileSystem."""

    @pytest.fixture
    def vfs(self):
        """Create VirtualFileSystem instance."""
        return VirtualFileSystem()

    @pytest.fixture
    def workspace_id(self):
        """Create test workspace ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_write_file(self, vfs, workspace_id):
        """Test writing a file."""
        file = await vfs.write_file(workspace_id, "strategy.py", b"# Strategy code")

        assert file is not None
        assert file.name == "strategy.py"
        assert file.file_type == FileType.FILE

    @pytest.mark.asyncio
    async def test_read_file(self, vfs, workspace_id):
        """Test reading file content."""
        await vfs.write_file(workspace_id, "test.py", b"test content")

        content = await vfs.read_file(workspace_id, "test.py")

        assert content == b"test content"

    @pytest.mark.asyncio
    async def test_overwrite_file(self, vfs, workspace_id):
        """Test overwriting a file."""
        await vfs.write_file(workspace_id, "test.py", b"original")

        file = await vfs.write_file(workspace_id, "test.py", b"updated", overwrite=True)

        assert file is not None
        content = await vfs.read_file(workspace_id, "test.py")
        assert content == b"updated"

    @pytest.mark.asyncio
    async def test_delete_file(self, vfs, workspace_id):
        """Test deleting a file."""
        await vfs.write_file(workspace_id, "test.py", b"content")

        deleted = await vfs.delete_file(workspace_id, "test.py")

        assert deleted is True

    @pytest.mark.asyncio
    async def test_list_directory(self, vfs, workspace_id):
        """Test listing directory contents."""
        await vfs.write_file(workspace_id, "a.py", b"")
        await vfs.write_file(workspace_id, "b.py", b"")

        listing = await vfs.list_directory(workspace_id)

        assert listing.file_count == 2

    @pytest.mark.asyncio
    async def test_workspace_isolation(self, vfs):
        """Test workspace file isolation."""
        ws1 = uuid4()
        ws2 = uuid4()

        await vfs.write_file(ws1, "test.py", b"ws1 content")
        await vfs.write_file(ws2, "test.py", b"ws2 content")

        content1 = await vfs.read_file(ws1, "test.py")
        content2 = await vfs.read_file(ws2, "test.py")

        assert content1 == b"ws1 content"
        assert content2 == b"ws2 content"

    @pytest.mark.asyncio
    async def test_path_traversal_normalized(self, vfs, workspace_id):
        """Test that path traversal attempts are normalized safely."""
        # Path traversal attempts are normalized (.. removed), not errored
        file = await vfs.write_file(workspace_id, "../../../etc/passwd", b"safe content")
        # The path should be normalized to just "etc/passwd"
        assert file.path == "etc/passwd"
        assert ".." not in file.path


# ============================================================================
# IDESessionManager Tests
# ============================================================================


class TestIDESessionManager:
    """Tests for IDESessionManager."""

    @pytest.fixture
    def manager(self):
        """Create IDESessionManager instance."""
        return IDESessionManager()

    @pytest.fixture
    def workspace_id(self):
        """Create test workspace ID."""
        return uuid4()

    @pytest.fixture
    def user_id(self):
        """Create test user ID."""
        return uuid4()

    @pytest.mark.asyncio
    async def test_create_session(self, manager, workspace_id, user_id):
        """Test creating IDE session."""
        session = await manager.create_session(workspace_id, user_id)

        assert session is not None
        assert session.workspace_id == workspace_id
        assert session.user_id == user_id
        assert session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_get_session(self, manager, workspace_id, user_id):
        """Test getting session by ID."""
        created = await manager.create_session(workspace_id, user_id)

        session = await manager.get_session(created.session_id)

        assert session is not None
        assert session.session_id == created.session_id

    @pytest.mark.asyncio
    async def test_close_session(self, manager, workspace_id, user_id):
        """Test closing session."""
        session = await manager.create_session(workspace_id, user_id)

        closed = await manager.close_session(session.session_id)

        assert closed is True
        assert await manager.get_session(session.session_id) is None

    @pytest.mark.asyncio
    async def test_session_timeout(self, manager, workspace_id, user_id):
        """Test session timeout detection."""
        config = SessionConfig(timeout_minutes=0)  # Immediate timeout
        session = await manager.create_session(workspace_id, user_id, config)

        # Session should be expired
        assert session.is_expired()

    @pytest.mark.asyncio
    async def test_session_touch(self, manager, workspace_id, user_id):
        """Test session activity update."""
        session = await manager.create_session(workspace_id, user_id)
        original_activity = session.last_activity

        await asyncio.sleep(0.01)
        session.touch()

        assert session.last_activity > original_activity

    @pytest.mark.asyncio
    async def test_max_sessions_per_user(self, manager):
        """Test maximum sessions per user limit."""
        workspace = uuid4()
        user = uuid4()

        # Create max sessions
        for _ in range(5):
            await manager.create_session(workspace, user)

        # Should fail on next
        with pytest.raises(ValueError, match="Maximum sessions per user"):
            await manager.create_session(workspace, user)

    @pytest.mark.asyncio
    async def test_list_sessions_by_workspace(self, manager):
        """Test listing sessions by workspace."""
        ws1 = uuid4()
        ws2 = uuid4()
        user = uuid4()

        await manager.create_session(ws1, user)
        await manager.create_session(ws1, user)
        await manager.create_session(ws2, user)

        sessions = await manager.list_sessions(workspace_id=ws1)

        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, manager, workspace_id, user_id):
        """Test cleanup of expired sessions."""
        config = SessionConfig(timeout_minutes=0)
        await manager.create_session(workspace_id, user_id, config)

        cleaned = await manager.cleanup_expired()

        assert cleaned >= 1
