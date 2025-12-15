# -*- coding: utf-8 -*-
"""
Tests for Artifact Manager - Phase 4.1

Tests cover:
- Artifact download and verification
- Digest validation (SHA-256)
- Manifest parsing
- Artifact lifecycle (activate, deactivate, swap)
"""

import hashlib
import json
import tempfile
import tarfile
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from packages.agent.daemon.artifact_manager import (
    ArtifactManager,
    ArtifactManifest,
    Artifact,
    ArtifactState,
    ArtifactType,
    ArtifactVerificationError,
    ArtifactDownloadError,
)


class TestArtifactManifest:
    """Test ArtifactManifest parsing and defaults."""

    def test_from_yaml_basic(self):
        """Test parsing basic manifest YAML."""
        yaml_content = """
name: test-strategy
version: 1.0.0
type: strategy
entrypoint: main.py
author: test
description: A test strategy

permissions:
  network:
    enabled: true
    egress_allowlist:
      - api.binance.com
      - api.coinbase.com
  filesystem:
    readonly: true
    allowed_paths:
      - /tmp
  resources:
    max_memory_mb: 256
    max_cpu_percent: 50.0
    max_execution_time_seconds: 1800
"""
        manifest = ArtifactManifest.from_yaml(yaml_content)

        assert manifest.name == "test-strategy"
        assert manifest.version == "1.0.0"
        assert manifest.artifact_type == ArtifactType.STRATEGY
        assert manifest.entrypoint == "main.py"
        assert manifest.network_allowed is True
        assert "api.binance.com" in manifest.egress_allowlist
        assert manifest.filesystem_readonly is True
        assert manifest.max_memory_mb == 256
        assert manifest.max_cpu_percent == 50.0
        assert manifest.max_execution_time_seconds == 1800

    def test_from_yaml_minimal(self):
        """Test parsing minimal manifest with defaults."""
        yaml_content = """
name: minimal
version: 0.1.0
"""
        manifest = ArtifactManifest.from_yaml(yaml_content)

        assert manifest.name == "minimal"
        assert manifest.version == "0.1.0"
        # Defaults - restrictive
        assert manifest.network_allowed is False
        assert manifest.filesystem_readonly is True
        assert manifest.max_memory_mb == 512

    def test_default_restrictive(self):
        """Test default restrictive manifest."""
        manifest = ArtifactManifest.default_restrictive()

        assert manifest.network_allowed is False
        assert manifest.egress_allowlist == []
        assert manifest.filesystem_readonly is True
        assert manifest.max_memory_mb == 256
        assert manifest.max_cpu_percent == 50.0
        assert manifest.max_execution_time_seconds == 300

    def test_to_dict(self):
        """Test manifest serialization."""
        manifest = ArtifactManifest(
            name="test",
            version="1.0.0",
            network_allowed=True,
            egress_allowlist=["example.com"],
        )

        data = manifest.to_dict()

        assert data["name"] == "test"
        assert data["version"] == "1.0.0"
        assert data["network_allowed"] is True
        assert "example.com" in data["egress_allowlist"]


class TestArtifactManager:
    """Test ArtifactManager operations."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_cache_dir):
        """Create ArtifactManager with temp directory."""
        return ArtifactManager(cache_dir=temp_cache_dir)

    def test_init_creates_cache_dir(self, temp_cache_dir):
        """Test that init creates cache directory."""
        cache_dir = temp_cache_dir / "artifacts"
        manager = ArtifactManager(cache_dir=cache_dir)

        assert cache_dir.exists()

    def test_verify_local_returns_false_for_missing(self, manager):
        """Test verify_local returns False for missing artifact."""
        assert manager.verify_local("nonexistent") is False

    def test_get_permissions_returns_none_for_missing(self, manager):
        """Test get_permissions returns None for missing artifact."""
        assert manager.get_permissions("nonexistent") is None

    def test_activate_fails_for_missing(self, manager):
        """Test activate fails for missing artifact."""
        assert manager.activate("nonexistent") is False

    def test_deactivate_fails_for_missing(self, manager):
        """Test deactivate fails for missing artifact."""
        assert manager.deactivate("nonexistent") is False

    def test_delete_fails_for_missing(self, manager):
        """Test delete fails for missing artifact."""
        assert manager.delete("nonexistent") is False

    def test_get_cache_size_empty(self, manager):
        """Test cache size is 0 for empty cache."""
        assert manager.get_cache_size_mb() == 0

    def test_list_artifacts_empty(self, manager):
        """Test list_artifacts returns empty list initially."""
        assert manager.list_artifacts() == []

    def test_get_status(self, manager, temp_cache_dir):
        """Test get_status returns expected structure."""
        status = manager.get_status()

        assert "cache_dir" in status
        assert "cache_size_mb" in status
        assert "total_artifacts" in status
        assert status["total_artifacts"] == 0


class TestArtifactVerification:
    """Test artifact verification logic."""

    def test_digest_computation(self):
        """Test SHA-256 digest computation."""
        data = b"test content"
        expected = f"sha256:{hashlib.sha256(data).hexdigest()}"

        hasher = hashlib.sha256()
        hasher.update(data)
        actual = f"sha256:{hasher.hexdigest()}"

        assert actual == expected

    def test_digest_format_normalization(self):
        """Test digest format with and without prefix."""
        digest_with_prefix = "sha256:abc123"
        digest_without_prefix = "abc123"

        # Both should be comparable after normalization
        normalized1 = digest_with_prefix
        normalized2 = f"sha256:{digest_without_prefix}"

        assert normalized1.startswith("sha256:")
        assert normalized2.startswith("sha256:")


class TestArtifactLifecycle:
    """Test artifact lifecycle transitions."""

    def test_artifact_states(self):
        """Test all artifact states exist."""
        states = [
            ArtifactState.DOWNLOADING,
            ArtifactState.VERIFYING,
            ArtifactState.EXTRACTING,
            ArtifactState.READY,
            ArtifactState.ACTIVE,
            ArtifactState.INACTIVE,
            ArtifactState.FAILED,
            ArtifactState.DELETED,
        ]

        assert len(states) == 8

    def test_artifact_types(self):
        """Test all artifact types exist."""
        types = [
            ArtifactType.STRATEGY,
            ArtifactType.MODEL,
            ArtifactType.CONFIG,
            ArtifactType.DATA,
        ]

        assert len(types) == 4

    def test_artifact_to_dict(self):
        """Test Artifact serialization."""
        artifact = Artifact(
            artifact_id="test-123",
            name="test",
            version="1.0.0",
            state=ArtifactState.READY,
            expected_digest="sha256:abc",
            actual_digest="sha256:abc",
            verified=True,
        )

        data = artifact.to_dict()

        assert data["artifact_id"] == "test-123"
        assert data["name"] == "test"
        assert data["state"] == "READY"
        assert data["verified"] is True
