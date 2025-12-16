# -*- coding: utf-8 -*-
"""
Virtual File System - CLOUD ZONE ONLY.

Provides isolated virtual filesystem for strategy development.

Design Doc Reference:
    - Section 4.1: "Strategy Workspace (репо стратегий, версии)"
    - Section 5.3: Multi-tenant isolation

Security:
    - Workspace-scoped file access
    - No access to system files
    - Size and path validation
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, Flag, auto
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Final, List, Optional, Set, Tuple, Union
from uuid import UUID

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# Maximum file size (bytes)
MAX_FILE_SIZE: Final[int] = 50 * 1024 * 1024  # 50MB

# Maximum total storage per workspace (bytes)
MAX_WORKSPACE_STORAGE: Final[int] = 1024 * 1024 * 1024  # 1GB

# Maximum path length
MAX_PATH_LENGTH: Final[int] = 500

# Maximum filename length
MAX_FILENAME_LENGTH: Final[int] = 255

# Maximum directory depth
MAX_DIRECTORY_DEPTH: Final[int] = 20

# Allowed file extensions
ALLOWED_EXTENSIONS: Final[frozenset] = frozenset({
    ".py", ".pyx", ".pxd",  # Python
    ".yaml", ".yml",  # YAML
    ".json",  # JSON
    ".toml",  # TOML
    ".ini", ".cfg",  # Config
    ".md", ".rst", ".txt",  # Docs
    ".sh", ".bash",  # Scripts
    ".csv", ".tsv",  # Data
    ".pkl", ".pickle",  # Pickled data
    ".pt", ".pth", ".onnx", ".h5",  # Models
    ".npy", ".npz",  # NumPy
    ".parquet",  # Columnar
    ".sql",  # SQL
    ".dockerfile",  # Docker
    ".gitignore", ".env.example",  # Git/Config
})

# Prohibited path patterns
PROHIBITED_PATTERNS: Final[List[re.Pattern]] = [
    re.compile(r'\.\.'),  # Path traversal
    re.compile(r'^/'),  # Absolute path
    re.compile(r'[<>:"|?*\x00-\x1f]'),  # Invalid characters
    re.compile(r'(^|/)\.git(/|$)'),  # Git internals
    re.compile(r'(^|/)__pycache__(/|$)'),  # Python cache
    re.compile(r'(^|/)node_modules(/|$)'),  # Node modules
]


# ============================================================================
# Enums
# ============================================================================


class FileType(str, Enum):
    """File type."""
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"


class FilePermission(Flag):
    """File permissions (Unix-like)."""
    NONE = 0
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()
    READ_WRITE = READ | WRITE
    ALL = READ | WRITE | EXECUTE


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class StrategyFile:
    """
    Virtual file in strategy workspace.

    Provides file metadata and content access.
    """
    path: str
    file_type: FileType = FileType.FILE
    size_bytes: int = 0
    content_hash: str = ""
    mime_type: str = "application/octet-stream"

    # Permissions
    permissions: FilePermission = FilePermission.READ_WRITE

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Ownership
    workspace_id: Optional[UUID] = None
    created_by: Optional[UUID] = None
    modified_by: Optional[UUID] = None

    # Git info (if tracked)
    git_sha: Optional[str] = None
    git_status: Optional[str] = None

    def __post_init__(self):
        # Normalize path
        self.path = self._normalize_path(self.path)

        # Detect MIME type
        if self.file_type == FileType.FILE:
            mime, _ = mimetypes.guess_type(self.path)
            self.mime_type = mime or "application/octet-stream"

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize file path."""
        # Use POSIX paths internally
        p = PurePosixPath(path)
        # Remove leading slash
        parts = [part for part in p.parts if part != "/"]
        return "/".join(parts) if parts else ""

    @property
    def name(self) -> str:
        """Get filename."""
        return PurePosixPath(self.path).name

    @property
    def parent(self) -> str:
        """Get parent directory path."""
        parent = str(PurePosixPath(self.path).parent)
        return "" if parent == "." else parent

    @property
    def extension(self) -> str:
        """Get file extension."""
        return PurePosixPath(self.path).suffix.lower()

    @property
    def is_directory(self) -> bool:
        """Check if this is a directory."""
        return self.file_type == FileType.DIRECTORY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "type": self.file_type.value,
            "size": self.size_bytes,
            "hash": self.content_hash,
            "mime_type": self.mime_type,
            "permissions": self.permissions.value,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "git_sha": self.git_sha,
            "git_status": self.git_status,
        }


@dataclass
class DirectoryListing:
    """Directory listing result."""
    path: str
    files: List[StrategyFile]
    total_size: int = 0
    file_count: int = 0
    directory_count: int = 0


@dataclass
class StorageQuota:
    """Storage quota for workspace."""
    max_storage_bytes: int = MAX_WORKSPACE_STORAGE
    max_file_size_bytes: int = MAX_FILE_SIZE
    max_files: int = 10000
    used_storage_bytes: int = 0
    file_count: int = 0

    @property
    def available_bytes(self) -> int:
        """Get available storage."""
        return max(0, self.max_storage_bytes - self.used_storage_bytes)

    @property
    def usage_percent(self) -> float:
        """Get usage percentage."""
        if self.max_storage_bytes == 0:
            return 100.0
        return (self.used_storage_bytes / self.max_storage_bytes) * 100


# ============================================================================
# Virtual File System
# ============================================================================


class VirtualFileSystem:
    """
    Virtual filesystem for strategy workspaces.

    Provides:
    - Workspace-isolated file storage
    - Path validation and sanitization
    - Storage quotas
    - File metadata tracking

    SECURITY:
    - No access to host filesystem
    - Strict path validation
    - Workspace tenant isolation
    """

    def __init__(
        self,
        storage_root: Optional[Path] = None,
        default_quota: Optional[StorageQuota] = None,
    ):
        """
        Initialize virtual filesystem.

        Args:
            storage_root: Root directory for file storage (None for in-memory)
            default_quota: Default storage quota
        """
        self._storage_root = storage_root
        self._default_quota = default_quota or StorageQuota()
        self._workspace_quotas: Dict[UUID, StorageQuota] = {}

        # In-memory storage (workspace_id -> path -> content)
        self._memory_storage: Dict[UUID, Dict[str, bytes]] = {}
        self._file_metadata: Dict[UUID, Dict[str, StrategyFile]] = {}

        self._lock = asyncio.Lock()

        logger.info(f"VirtualFileSystem initialized (storage_root={storage_root})")

    # ========================================================================
    # Path Validation
    # ========================================================================

    def validate_path(self, path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file path.

        Returns:
            (is_valid, error_message)
        """
        # Check length
        if len(path) > MAX_PATH_LENGTH:
            return False, f"Path too long (max {MAX_PATH_LENGTH} characters)"

        # Check for prohibited patterns
        for pattern in PROHIBITED_PATTERNS:
            if pattern.search(path):
                return False, f"Path contains prohibited pattern: {pattern.pattern}"

        # Check filename length
        name = PurePosixPath(path).name
        if len(name) > MAX_FILENAME_LENGTH:
            return False, f"Filename too long (max {MAX_FILENAME_LENGTH} characters)"

        # Check directory depth
        parts = [p for p in path.split("/") if p]
        if len(parts) > MAX_DIRECTORY_DEPTH:
            return False, f"Directory too deep (max {MAX_DIRECTORY_DEPTH} levels)"

        # Check extension for files (not directories)
        if "." in name:
            ext = PurePosixPath(path).suffix.lower()
            # Allow files without extension or with allowed extensions
            if ext and ext not in ALLOWED_EXTENSIONS:
                return False, f"File extension not allowed: {ext}"

        return True, None

    def normalize_path(self, path: str) -> str:
        """Normalize and sanitize path."""
        # Use POSIX paths
        p = PurePosixPath(path)
        # Remove leading slash and normalize
        parts = [part for part in p.parts if part not in ("/", "..", ".")]
        return "/".join(parts) if parts else ""

    # ========================================================================
    # File Operations
    # ========================================================================

    async def write_file(
        self,
        workspace_id: UUID,
        path: str,
        content: bytes,
        user_id: Optional[UUID] = None,
        overwrite: bool = True,
    ) -> StrategyFile:
        """
        Write file to workspace.

        Args:
            workspace_id: Workspace ID
            path: File path (relative)
            content: File content
            user_id: User performing the operation
            overwrite: Allow overwriting existing files

        Returns:
            StrategyFile metadata
        """
        # Validate path
        path = self.normalize_path(path)
        valid, error = self.validate_path(path)
        if not valid:
            raise ValueError(error)

        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise ValueError(f"File too large (max {MAX_FILE_SIZE} bytes)")

        # Check quota
        quota = self._get_quota(workspace_id)
        existing_size = 0

        async with self._lock:
            # Get existing file size if overwriting
            if workspace_id in self._file_metadata:
                existing = self._file_metadata[workspace_id].get(path)
                if existing:
                    if not overwrite:
                        raise ValueError(f"File already exists: {path}")
                    existing_size = existing.size_bytes

            # Check quota
            new_usage = quota.used_storage_bytes - existing_size + len(content)
            if new_usage > quota.max_storage_bytes:
                raise ValueError("Storage quota exceeded")

            # Create parent directories
            await self._ensure_parent_dirs(workspace_id, path, user_id)

            # Store content
            if workspace_id not in self._memory_storage:
                self._memory_storage[workspace_id] = {}
            self._memory_storage[workspace_id][path] = content

            # Create metadata
            content_hash = hashlib.sha256(content).hexdigest()
            now = datetime.now(timezone.utc)

            file_meta = StrategyFile(
                path=path,
                file_type=FileType.FILE,
                size_bytes=len(content),
                content_hash=content_hash,
                workspace_id=workspace_id,
                created_by=user_id,
                modified_by=user_id,
                created_at=now,
                modified_at=now,
            )

            if workspace_id not in self._file_metadata:
                self._file_metadata[workspace_id] = {}
            self._file_metadata[workspace_id][path] = file_meta

            # Update quota
            quota.used_storage_bytes = new_usage
            if not existing_size:
                quota.file_count += 1

        logger.debug(f"Wrote file {path} ({len(content)} bytes) to workspace {workspace_id}")
        return file_meta

    async def read_file(
        self,
        workspace_id: UUID,
        path: str,
    ) -> Optional[bytes]:
        """Read file content."""
        path = self.normalize_path(path)

        async with self._lock:
            storage = self._memory_storage.get(workspace_id, {})
            content = storage.get(path)

            if content is not None:
                # Update access time
                if workspace_id in self._file_metadata:
                    meta = self._file_metadata[workspace_id].get(path)
                    if meta:
                        meta.accessed_at = datetime.now(timezone.utc)

            return content

    async def delete_file(
        self,
        workspace_id: UUID,
        path: str,
    ) -> bool:
        """Delete file."""
        path = self.normalize_path(path)

        async with self._lock:
            if workspace_id not in self._memory_storage:
                return False

            content = self._memory_storage[workspace_id].pop(path, None)
            if content is None:
                return False

            # Remove metadata
            if workspace_id in self._file_metadata:
                meta = self._file_metadata[workspace_id].pop(path, None)
                if meta:
                    # Update quota
                    quota = self._get_quota(workspace_id)
                    quota.used_storage_bytes -= meta.size_bytes
                    quota.file_count -= 1

        logger.debug(f"Deleted file {path} from workspace {workspace_id}")
        return True

    async def get_file_info(
        self,
        workspace_id: UUID,
        path: str,
    ) -> Optional[StrategyFile]:
        """Get file metadata."""
        path = self.normalize_path(path)

        async with self._lock:
            if workspace_id not in self._file_metadata:
                return None
            return self._file_metadata[workspace_id].get(path)

    async def file_exists(
        self,
        workspace_id: UUID,
        path: str,
    ) -> bool:
        """Check if file exists."""
        path = self.normalize_path(path)

        async with self._lock:
            if workspace_id not in self._memory_storage:
                return False
            return path in self._memory_storage[workspace_id]

    # ========================================================================
    # Directory Operations
    # ========================================================================

    async def create_directory(
        self,
        workspace_id: UUID,
        path: str,
        user_id: Optional[UUID] = None,
    ) -> StrategyFile:
        """Create directory."""
        path = self.normalize_path(path)
        valid, error = self.validate_path(path)
        if not valid:
            raise ValueError(error)

        async with self._lock:
            # Check if already exists
            if workspace_id in self._file_metadata:
                existing = self._file_metadata[workspace_id].get(path)
                if existing:
                    if existing.is_directory:
                        return existing
                    raise ValueError(f"Path already exists as file: {path}")

            # Create directory metadata
            dir_meta = StrategyFile(
                path=path,
                file_type=FileType.DIRECTORY,
                workspace_id=workspace_id,
                created_by=user_id,
                permissions=FilePermission.ALL,
            )

            if workspace_id not in self._file_metadata:
                self._file_metadata[workspace_id] = {}
            self._file_metadata[workspace_id][path] = dir_meta

        logger.debug(f"Created directory {path} in workspace {workspace_id}")
        return dir_meta

    async def list_directory(
        self,
        workspace_id: UUID,
        path: str = "",
        recursive: bool = False,
    ) -> DirectoryListing:
        """
        List directory contents.

        Args:
            workspace_id: Workspace ID
            path: Directory path (empty for root)
            recursive: Include subdirectories

        Returns:
            DirectoryListing with files and directories
        """
        path = self.normalize_path(path)

        async with self._lock:
            files: List[StrategyFile] = []
            total_size = 0
            file_count = 0
            dir_count = 0

            metadata = self._file_metadata.get(workspace_id, {})

            for file_path, file_meta in metadata.items():
                # Check if in directory
                if path:
                    if not file_path.startswith(path + "/"):
                        continue
                    relative = file_path[len(path) + 1:]
                else:
                    relative = file_path

                # Check depth for non-recursive listing
                if not recursive and "/" in relative:
                    # Only include immediate children
                    continue

                files.append(file_meta)
                if file_meta.is_directory:
                    dir_count += 1
                else:
                    file_count += 1
                    total_size += file_meta.size_bytes

            return DirectoryListing(
                path=path,
                files=sorted(files, key=lambda f: (not f.is_directory, f.path)),
                total_size=total_size,
                file_count=file_count,
                directory_count=dir_count,
            )

    async def delete_directory(
        self,
        workspace_id: UUID,
        path: str,
        recursive: bool = False,
    ) -> int:
        """
        Delete directory.

        Args:
            workspace_id: Workspace ID
            path: Directory path
            recursive: Delete contents recursively

        Returns:
            Number of items deleted
        """
        path = self.normalize_path(path)

        async with self._lock:
            if workspace_id not in self._file_metadata:
                return 0

            metadata = self._file_metadata[workspace_id]
            storage = self._memory_storage.get(workspace_id, {})

            # Find items to delete
            to_delete = []
            for file_path in list(metadata.keys()):
                if file_path == path or file_path.startswith(path + "/"):
                    to_delete.append(file_path)

            if not recursive and len(to_delete) > 1:
                raise ValueError("Directory not empty (use recursive=True)")

            # Delete items
            quota = self._get_quota(workspace_id)
            for file_path in to_delete:
                meta = metadata.pop(file_path, None)
                storage.pop(file_path, None)
                if meta and not meta.is_directory:
                    quota.used_storage_bytes -= meta.size_bytes
                    quota.file_count -= 1

            return len(to_delete)

    # ========================================================================
    # Copy/Move Operations
    # ========================================================================

    async def copy_file(
        self,
        workspace_id: UUID,
        src_path: str,
        dst_path: str,
        user_id: Optional[UUID] = None,
    ) -> StrategyFile:
        """Copy file to new location."""
        src_path = self.normalize_path(src_path)
        dst_path = self.normalize_path(dst_path)

        content = await self.read_file(workspace_id, src_path)
        if content is None:
            raise ValueError(f"Source file not found: {src_path}")

        return await self.write_file(workspace_id, dst_path, content, user_id)

    async def move_file(
        self,
        workspace_id: UUID,
        src_path: str,
        dst_path: str,
        user_id: Optional[UUID] = None,
    ) -> StrategyFile:
        """Move file to new location."""
        new_file = await self.copy_file(workspace_id, src_path, dst_path, user_id)
        await self.delete_file(workspace_id, src_path)
        return new_file

    async def rename_file(
        self,
        workspace_id: UUID,
        path: str,
        new_name: str,
        user_id: Optional[UUID] = None,
    ) -> StrategyFile:
        """Rename file."""
        path = self.normalize_path(path)
        parent = str(PurePosixPath(path).parent)
        if parent == ".":
            parent = ""
        new_path = f"{parent}/{new_name}" if parent else new_name
        return await self.move_file(workspace_id, path, new_path, user_id)

    # ========================================================================
    # Quota Management
    # ========================================================================

    def _get_quota(self, workspace_id: UUID) -> StorageQuota:
        """Get quota for workspace."""
        if workspace_id not in self._workspace_quotas:
            self._workspace_quotas[workspace_id] = StorageQuota(
                max_storage_bytes=self._default_quota.max_storage_bytes,
                max_file_size_bytes=self._default_quota.max_file_size_bytes,
                max_files=self._default_quota.max_files,
            )
        return self._workspace_quotas[workspace_id]

    async def get_quota(self, workspace_id: UUID) -> StorageQuota:
        """Get storage quota for workspace."""
        return self._get_quota(workspace_id)

    async def set_quota(
        self,
        workspace_id: UUID,
        max_storage_bytes: Optional[int] = None,
        max_file_size_bytes: Optional[int] = None,
        max_files: Optional[int] = None,
    ) -> StorageQuota:
        """Update workspace quota."""
        quota = self._get_quota(workspace_id)
        if max_storage_bytes is not None:
            quota.max_storage_bytes = max_storage_bytes
        if max_file_size_bytes is not None:
            quota.max_file_size_bytes = max_file_size_bytes
        if max_files is not None:
            quota.max_files = max_files
        return quota

    # ========================================================================
    # Private Methods
    # ========================================================================

    async def _ensure_parent_dirs(
        self,
        workspace_id: UUID,
        path: str,
        user_id: Optional[UUID] = None,
    ) -> None:
        """Ensure parent directories exist."""
        parts = path.split("/")
        for i in range(1, len(parts)):
            dir_path = "/".join(parts[:i])
            if workspace_id not in self._file_metadata:
                self._file_metadata[workspace_id] = {}

            if dir_path not in self._file_metadata[workspace_id]:
                self._file_metadata[workspace_id][dir_path] = StrategyFile(
                    path=dir_path,
                    file_type=FileType.DIRECTORY,
                    workspace_id=workspace_id,
                    created_by=user_id,
                    permissions=FilePermission.ALL,
                )

    # ========================================================================
    # Cleanup
    # ========================================================================

    async def clear_workspace(self, workspace_id: UUID) -> int:
        """Clear all files in workspace."""
        async with self._lock:
            count = 0
            if workspace_id in self._memory_storage:
                count = len(self._memory_storage[workspace_id])
                del self._memory_storage[workspace_id]
            if workspace_id in self._file_metadata:
                del self._file_metadata[workspace_id]
            if workspace_id in self._workspace_quotas:
                del self._workspace_quotas[workspace_id]

        logger.info(f"Cleared workspace {workspace_id} ({count} files)")
        return count
