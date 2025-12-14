# -*- coding: utf-8 -*-
"""
Tests for Registry Mirror.

Design Doc Reference: Phase 9 - Enterprise/on-prem pack
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from packages.cloud.enterprise.registry_mirror import (
    RegistryMirror,
    RegistryMirrorConfig,
    MirrorStatus,
    MirrorSyncResult,
    ArtifactInfo,
    SyncMode,
)


class TestRegistryMirrorConfig:
    """Tests for RegistryMirrorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RegistryMirrorConfig()

        assert config.sync_interval_seconds == 3600
        assert config.sync_mode == SyncMode.INCREMENTAL
        assert config.auto_sync is True
        assert config.air_gapped is False
        assert config.require_signatures is True
        assert config.retention_days == 90

    def test_custom_config(self):
        """Test custom configuration."""
        config = RegistryMirrorConfig(
            upstream_url="https://registry.example.com",
            sync_interval_seconds=1800,
            air_gapped=True,
            require_signatures=False,
        )

        assert config.upstream_url == "https://registry.example.com"
        assert config.sync_interval_seconds == 1800
        assert config.air_gapped is True
        assert config.require_signatures is False

    def test_config_to_dict(self):
        """Test config serialization."""
        config = RegistryMirrorConfig()
        data = config.to_dict()

        assert "storage_path" in data
        assert "sync_mode" in data
        assert data["auto_sync"] is True


class TestArtifactInfo:
    """Tests for ArtifactInfo."""

    def test_artifact_info_creation(self):
        """Test creating artifact info."""
        artifact = ArtifactInfo(
            digest="sha256:abc123",
            name="strategy-v1",
            tag="1.0.0",
            size_bytes=1024,
            created_at=datetime.utcnow(),
        )

        assert artifact.digest == "sha256:abc123"
        assert artifact.name == "strategy-v1"
        assert artifact.tag == "1.0.0"
        assert artifact.size_bytes == 1024
        assert artifact.signature_verified is False

    def test_artifact_info_to_dict(self):
        """Test artifact info serialization."""
        artifact = ArtifactInfo(
            digest="sha256:abc123",
            name="strategy-v1",
            tag="1.0.0",
            size_bytes=1024,
            created_at=datetime.utcnow(),
            synced_at=datetime.utcnow(),
            signature_verified=True,
        )

        data = artifact.to_dict()

        assert data["digest"] == "sha256:abc123"
        assert data["signature_verified"] is True
        assert "created_at" in data
        assert "synced_at" in data


class TestMirrorSyncResult:
    """Tests for MirrorSyncResult."""

    def test_sync_result_success(self):
        """Test successful sync result."""
        result = MirrorSyncResult(
            success=True,
            synced_count=5,
            skipped_count=2,
            failed_count=0,
        )

        assert result.success is True
        assert result.synced_count == 5
        assert result.failed_count == 0

    def test_sync_result_failure(self):
        """Test failed sync result."""
        result = MirrorSyncResult(
            success=False,
            synced_count=3,
            failed_count=2,
            errors=["Error 1", "Error 2"],
            failed_artifacts=["artifact1", "artifact2"],
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert len(result.failed_artifacts) == 2

    def test_sync_result_to_dict(self):
        """Test sync result serialization."""
        result = MirrorSyncResult(
            success=True,
            synced_count=10,
            bytes_transferred=1024000,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["synced_count"] == 10
        assert "timestamp" in data


class TestRegistryMirror:
    """Tests for RegistryMirror."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mirror_config(self, temp_storage):
        """Create mirror configuration."""
        return RegistryMirrorConfig(
            storage_path=temp_storage,
            upstream_url="https://registry.example.com",
            auto_sync=False,  # Disable for tests
        )

    @pytest.fixture
    def mirror(self, mirror_config):
        """Create registry mirror instance."""
        return RegistryMirror(mirror_config)

    def test_mirror_initialization(self, mirror):
        """Test mirror initialization."""
        assert mirror.status == MirrorStatus.IDLE
        assert mirror.artifact_count == 0
        assert mirror.last_sync is None

    @pytest.mark.asyncio
    async def test_mirror_start_stop(self, mirror):
        """Test starting and stopping mirror."""
        await mirror.start()

        assert mirror.status == MirrorStatus.IDLE
        assert mirror._running is True

        await mirror.stop()

        assert mirror._running is False
        assert mirror.status == MirrorStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_mirror_sync_air_gapped(self, temp_storage):
        """Test sync fails in air-gapped mode."""
        config = RegistryMirrorConfig(
            storage_path=temp_storage,
            air_gapped=True,
        )
        mirror = RegistryMirror(config)

        result = await mirror.sync()

        assert result.success is False
        assert "air-gapped" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_import_artifact(self, mirror, temp_storage):
        """Test importing artifact for air-gapped mode."""
        await mirror.start()

        # Create test artifact
        artifact_content = b"test artifact content"
        artifact_path = temp_storage / "test_artifact.bin"
        with open(artifact_path, "wb") as f:
            f.write(artifact_content)

        # Import artifact
        artifact = await mirror.import_artifact(
            source_path=artifact_path,
            name="test-strategy",
            tag="1.0.0",
        )

        assert artifact is not None
        assert artifact.name == "test-strategy"
        assert artifact.tag == "1.0.0"
        assert artifact.size_bytes == len(artifact_content)
        assert mirror.artifact_count == 1

        await mirror.stop()

    @pytest.mark.asyncio
    async def test_get_artifact(self, mirror, temp_storage):
        """Test getting artifact by digest."""
        await mirror.start()

        # Import artifact
        artifact_content = b"test content"
        artifact_path = temp_storage / "test.bin"
        with open(artifact_path, "wb") as f:
            f.write(artifact_content)

        imported = await mirror.import_artifact(
            source_path=artifact_path,
            name="test",
            tag="latest",
        )

        # Get artifact
        artifact = mirror.get_artifact(imported.digest)

        assert artifact is not None
        assert artifact.digest == imported.digest

        await mirror.stop()

    @pytest.mark.asyncio
    async def test_list_artifacts_with_filter(self, mirror, temp_storage):
        """Test listing artifacts with filters."""
        await mirror.start()

        # Import multiple artifacts
        for i in range(3):
            artifact_path = temp_storage / f"artifact{i}.bin"
            with open(artifact_path, "wb") as f:
                f.write(f"content{i}".encode())

            await mirror.import_artifact(
                source_path=artifact_path,
                name=f"strategy-{i}",
                tag=f"1.{i}.0",
            )

        # List all
        all_artifacts = mirror.list_artifacts()
        assert len(all_artifacts) == 3

        # Filter by name
        filtered = mirror.list_artifacts(name_filter="strategy-1")
        assert len(filtered) == 1
        assert filtered[0].name == "strategy-1"

        await mirror.stop()

    @pytest.mark.asyncio
    async def test_verify_artifact_success(self, mirror, temp_storage):
        """Test successful artifact verification."""
        await mirror.start()

        # Import artifact
        artifact_content = b"verifiable content"
        artifact_path = temp_storage / "verify.bin"
        with open(artifact_path, "wb") as f:
            f.write(artifact_content)

        imported = await mirror.import_artifact(
            source_path=artifact_path,
            name="verify-test",
            tag="1.0.0",
        )

        # Verify
        is_valid, error = await mirror.verify_artifact(imported.digest)

        assert is_valid is True
        assert error is None

        await mirror.stop()

    @pytest.mark.asyncio
    async def test_verify_artifact_not_found(self, mirror):
        """Test verification of non-existent artifact."""
        await mirror.start()

        is_valid, error = await mirror.verify_artifact("sha256:nonexistent")

        assert is_valid is False
        assert "not found" in error.lower()

        await mirror.stop()

    @pytest.mark.asyncio
    async def test_delete_artifact(self, mirror, temp_storage):
        """Test deleting artifact."""
        await mirror.start()

        # Import artifact
        artifact_path = temp_storage / "delete.bin"
        with open(artifact_path, "wb") as f:
            f.write(b"to be deleted")

        imported = await mirror.import_artifact(
            source_path=artifact_path,
            name="delete-test",
            tag="1.0.0",
        )

        assert mirror.artifact_count == 1

        # Delete
        deleted = await mirror.delete_artifact(imported.digest)

        assert deleted is True
        assert mirror.artifact_count == 0

        await mirror.stop()

    @pytest.mark.asyncio
    async def test_cleanup_old_artifacts(self, mirror, temp_storage):
        """Test cleanup of old artifacts."""
        await mirror.start()

        # Import artifact and manually set old date
        artifact_path = temp_storage / "old.bin"
        with open(artifact_path, "wb") as f:
            f.write(b"old content")

        imported = await mirror.import_artifact(
            source_path=artifact_path,
            name="old-artifact",
            tag="1.0.0",
        )

        # Set synced_at to old date
        imported.synced_at = datetime.utcnow() - timedelta(days=100)

        # Cleanup (retention is 90 days by default)
        deleted_count = await mirror.cleanup_old_artifacts()

        assert deleted_count == 1
        assert mirror.artifact_count == 0

        await mirror.stop()

    def test_get_status(self, mirror):
        """Test getting mirror status."""
        status = mirror.get_status()

        assert "status" in status
        assert "artifact_count" in status
        assert "storage_used_bytes" in status
        assert "air_gapped" in status

    @pytest.mark.asyncio
    async def test_sync_callback(self, mirror_config):
        """Test sync completion callback."""
        callback_called = False
        callback_result = None

        def on_sync_complete(result):
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = result

        mirror = RegistryMirror(
            mirror_config,
            on_sync_complete=on_sync_complete,
        )

        await mirror.start()

        # Sync (will fail gracefully with empty upstream)
        await mirror.sync()

        # Callback should have been called
        assert callback_called is True
        assert callback_result is not None

        await mirror.stop()

    @pytest.mark.asyncio
    async def test_storage_used_bytes(self, mirror, temp_storage):
        """Test storage usage calculation."""
        await mirror.start()

        initial_usage = mirror.storage_used_bytes

        # Import artifact
        artifact_path = temp_storage / "size.bin"
        content = b"x" * 1024  # 1KB
        with open(artifact_path, "wb") as f:
            f.write(content)

        await mirror.import_artifact(
            source_path=artifact_path,
            name="size-test",
            tag="1.0.0",
        )

        # Storage should have increased
        assert mirror.storage_used_bytes >= initial_usage + 1024

        await mirror.stop()

    @pytest.mark.asyncio
    async def test_artifact_path(self, mirror, temp_storage):
        """Test getting artifact blob path."""
        await mirror.start()

        # Import artifact
        artifact_path = temp_storage / "path.bin"
        with open(artifact_path, "wb") as f:
            f.write(b"path test")

        imported = await mirror.import_artifact(
            source_path=artifact_path,
            name="path-test",
            tag="1.0.0",
        )

        # Get path
        blob_path = mirror.get_artifact_path(imported.digest)

        assert blob_path is not None
        assert blob_path.exists()

        await mirror.stop()

    @pytest.mark.asyncio
    async def test_index_persistence(self, temp_storage):
        """Test that index is persisted and loaded."""
        config = RegistryMirrorConfig(
            storage_path=temp_storage,
            auto_sync=False,
        )

        # Create mirror and import artifact
        mirror1 = RegistryMirror(config)
        await mirror1.start()

        artifact_path = temp_storage / "persist.bin"
        with open(artifact_path, "wb") as f:
            f.write(b"persistent content")

        imported = await mirror1.import_artifact(
            source_path=artifact_path,
            name="persist-test",
            tag="1.0.0",
        )

        await mirror1.stop()

        # Create new mirror instance - should load index
        mirror2 = RegistryMirror(config)
        await mirror2.start()

        # Should have the artifact
        assert mirror2.artifact_count == 1
        artifact = mirror2.get_artifact(imported.digest)
        assert artifact is not None

        await mirror2.stop()


class TestRegistryMirrorFiltering:
    """Tests for artifact filtering."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_include_patterns(self, temp_storage):
        """Test include pattern filtering."""
        config = RegistryMirrorConfig(
            storage_path=temp_storage,
            include_patterns=["ccea/*", "strategy-*"],
        )
        mirror = RegistryMirror(config)

        artifacts = ["ccea/agent:1.0", "strategy-btc:latest", "other:1.0"]
        filtered = mirror._filter_artifacts(artifacts)

        assert len(filtered) == 2
        assert "other:1.0" not in filtered

    def test_exclude_patterns(self, temp_storage):
        """Test exclude pattern filtering."""
        config = RegistryMirrorConfig(
            storage_path=temp_storage,
            exclude_patterns=["*-dev*", "*-test*"],
        )
        mirror = RegistryMirror(config)

        artifacts = ["strategy:1.0", "strategy-dev:1.0", "agent-test:latest"]
        filtered = mirror._filter_artifacts(artifacts)

        assert len(filtered) == 1
        assert filtered[0] == "strategy:1.0"

    def test_combined_patterns(self, temp_storage):
        """Test combined include and exclude patterns."""
        config = RegistryMirrorConfig(
            storage_path=temp_storage,
            include_patterns=["strategy-*"],
            exclude_patterns=["*-test*"],
        )
        mirror = RegistryMirror(config)

        artifacts = [
            "strategy-btc:1.0",
            "strategy-eth:1.0",
            "strategy-test:1.0",
            "other:1.0",
        ]
        filtered = mirror._filter_artifacts(artifacts)

        assert len(filtered) == 2
        assert "strategy-btc:1.0" in filtered
        assert "strategy-eth:1.0" in filtered
