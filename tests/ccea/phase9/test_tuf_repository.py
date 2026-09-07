# -*- coding: utf-8 -*-
"""
Tests for TUF Repository.

Design Doc Reference: Phase 9 - Enterprise/on-prem pack, Design Doc 15.2/5.2
TUF Specification: https://theupdateframework.github.io/specification/latest/
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from packages.cloud.enterprise.tuf_repository import (
    TUFRepository,
    TUFConfig,
    TUFKey,
    TUFSignature,
    TUFMetadata,
    RootMetadata,
    TimestampMetadata,
    SnapshotMetadata,
    TargetsMetadata,
    SignedMetadata,
    TargetInfo,
    RoleInfo,
    TUFRole,
    KeyType,
    ROOT_VERSION,
    TIMESTAMP_EXPIRY_DAYS,
    SNAPSHOT_EXPIRY_DAYS,
    TARGETS_EXPIRY_DAYS,
    ROOT_EXPIRY_DAYS,
)


class TestTUFKey:
    """Tests for TUFKey dataclass."""

    def test_key_creation(self):
        """Test creating TUF key."""
        key = TUFKey(
            key_id="abc123",
            key_type=KeyType.ED25519,
            public_key="base64encodedkey",
        )

        assert key.key_id == "abc123"
        assert key.key_type == KeyType.ED25519
        assert key.scheme == "ed25519"

    def test_key_to_dict(self):
        """Test key serialization to TUF format."""
        key = TUFKey(
            key_id="abc123",
            key_type=KeyType.ED25519,
            public_key="cHVibGljX2tleQ==",
        )

        data = key.to_dict()

        assert data["keytype"] == "ed25519"
        assert data["scheme"] == "ed25519"
        assert data["keyval"]["public"] == "cHVibGljX2tleQ=="

    def test_key_from_dict(self):
        """Test creating key from TUF format."""
        data = {
            "keytype": "ed25519",
            "scheme": "ed25519",
            "keyval": {
                "public": "cHVibGljX2tleQ==",
            },
        }

        key = TUFKey.from_dict("abc123", data)

        assert key.key_id == "abc123"
        assert key.key_type == KeyType.ED25519
        assert key.public_key == "cHVibGljX2tleQ=="


class TestTUFSignature:
    """Tests for TUFSignature dataclass."""

    def test_signature_creation(self):
        """Test creating TUF signature."""
        sig = TUFSignature(
            key_id="key123",
            signature="c2lnbmF0dXJl",
        )

        assert sig.key_id == "key123"
        assert sig.signature == "c2lnbmF0dXJl"

    def test_signature_to_dict(self):
        """Test signature serialization."""
        sig = TUFSignature(
            key_id="key123",
            signature="c2lnbmF0dXJl",
        )

        data = sig.to_dict()

        assert data["keyid"] == "key123"
        assert data["sig"] == "c2lnbmF0dXJl"


class TestRoleInfo:
    """Tests for RoleInfo dataclass."""

    def test_role_info_creation(self):
        """Test creating role info."""
        role = RoleInfo(
            key_ids=["key1", "key2"],
            threshold=2,
        )

        assert role.key_ids == ["key1", "key2"]
        assert role.threshold == 2

    def test_role_info_to_dict(self):
        """Test role info serialization."""
        role = RoleInfo(
            key_ids=["key1"],
            threshold=1,
        )

        data = role.to_dict()

        assert data["keyids"] == ["key1"]
        assert data["threshold"] == 1


class TestTargetInfo:
    """Tests for TargetInfo dataclass."""

    def test_target_info_creation(self):
        """Test creating target info."""
        info = TargetInfo(
            length=1024,
            hashes={
                "sha256": "abc123",
                "sha512": "def456",
            },
        )

        assert info.length == 1024
        assert info.hashes["sha256"] == "abc123"

    def test_target_info_with_custom(self):
        """Test target info with custom metadata."""
        info = TargetInfo(
            length=2048,
            hashes={"sha256": "abc"},
            custom={"version": "1.0.0", "platform": "linux"},
        )

        assert info.custom["version"] == "1.0.0"

    def test_target_info_to_dict(self):
        """Test target info serialization."""
        info = TargetInfo(
            length=1024,
            hashes={"sha256": "abc123"},
            custom={"version": "1.0.0"},
        )

        data = info.to_dict()

        assert data["length"] == 1024
        assert data["hashes"]["sha256"] == "abc123"
        assert data["custom"]["version"] == "1.0.0"


class TestMetadataClasses:
    """Tests for metadata classes."""

    def test_root_metadata(self):
        """Test RootMetadata."""
        now = datetime.utcnow()
        root = RootMetadata(
            version=1,
            expires=now + timedelta(days=365),
            consistent_snapshot=True,
        )

        assert root.role == TUFRole.ROOT
        assert root.version == 1
        assert root.consistent_snapshot is True

    def test_root_metadata_to_dict(self):
        """Test RootMetadata serialization."""
        now = datetime.utcnow()
        key = TUFKey(
            key_id="key1",
            key_type=KeyType.ED25519,
            public_key="abc",
        )
        root = RootMetadata(
            version=1,
            expires=now + timedelta(days=365),
            keys={"key1": key},
            roles={"root": RoleInfo(["key1"], 1)},
        )

        data = root.to_signed_dict()

        assert data["_type"] == "root"
        assert data["version"] == 1
        assert "keys" in data
        assert "roles" in data

    def test_timestamp_metadata(self):
        """Test TimestampMetadata."""
        now = datetime.utcnow()
        ts = TimestampMetadata(
            version=5,
            expires=now + timedelta(days=1),
            snapshot_info=TargetInfo(
                length=1024,
                hashes={"sha256": "abc"},
            ),
        )

        assert ts.role == TUFRole.TIMESTAMP
        assert ts.version == 5

    def test_timestamp_metadata_to_dict(self):
        """Test TimestampMetadata serialization."""
        now = datetime.utcnow()
        ts = TimestampMetadata(
            version=1,
            expires=now + timedelta(days=1),
            snapshot_info=TargetInfo(
                length=512,
                hashes={"sha256": "xyz"},
            ),
        )

        data = ts.to_signed_dict()

        assert data["_type"] == "timestamp"
        assert "meta" in data
        assert "snapshot.json" in data["meta"]

    def test_snapshot_metadata(self):
        """Test SnapshotMetadata."""
        now = datetime.utcnow()
        snap = SnapshotMetadata(
            version=3,
            expires=now + timedelta(days=7),
            meta={
                "targets.json": TargetInfo(length=2048, hashes={"sha256": "abc"}),
            },
        )

        assert snap.role == TUFRole.SNAPSHOT
        assert "targets.json" in snap.meta

    def test_targets_metadata(self):
        """Test TargetsMetadata."""
        now = datetime.utcnow()
        targets = TargetsMetadata(
            version=10,
            expires=now + timedelta(days=365),
            targets={
                "agent-1.0.0.tar.gz": TargetInfo(
                    length=10240,
                    hashes={"sha256": "abc123"},
                    custom={"version": "1.0.0"},
                ),
            },
        )

        assert targets.role == TUFRole.TARGETS
        assert "agent-1.0.0.tar.gz" in targets.targets


class TestSignedMetadata:
    """Tests for SignedMetadata container."""

    def test_signed_metadata_creation(self):
        """Test creating signed metadata."""
        signed = SignedMetadata(
            signed={"_type": "root", "version": 1},
            signatures=[
                TUFSignature(key_id="key1", signature="sig1"),
            ],
        )

        assert signed.signed["version"] == 1
        assert len(signed.signatures) == 1

    def test_signed_metadata_to_dict(self):
        """Test signed metadata serialization."""
        signed = SignedMetadata(
            signed={"_type": "timestamp", "version": 5},
            signatures=[
                TUFSignature(key_id="key1", signature="sig1"),
                TUFSignature(key_id="key2", signature="sig2"),
            ],
        )

        data = signed.to_dict()

        assert data["signed"]["version"] == 5
        assert len(data["signatures"]) == 2

    def test_canonical_bytes(self):
        """Test canonical JSON representation."""
        signed = SignedMetadata(
            signed={"z": 1, "a": 2, "m": 3},
            signatures=[],
        )

        canonical = signed.canonical_bytes

        # Keys should be sorted
        assert b'"a":2' in canonical
        assert canonical.index(b'"a"') < canonical.index(b'"m"')
        assert canonical.index(b'"m"') < canonical.index(b'"z"')


class TestTUFConfig:
    """Tests for TUFConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = TUFConfig()

        assert config.timestamp_expiry_days == TIMESTAMP_EXPIRY_DAYS
        assert config.snapshot_expiry_days == SNAPSHOT_EXPIRY_DAYS
        assert config.targets_expiry_days == TARGETS_EXPIRY_DAYS
        assert config.root_expiry_days == ROOT_EXPIRY_DAYS
        assert config.root_threshold == 2
        assert config.consistent_snapshot is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TUFConfig(
            timestamp_expiry_days=2,
            root_threshold=3,
            consistent_snapshot=False,
        )

        assert config.timestamp_expiry_days == 2
        assert config.root_threshold == 3
        assert config.consistent_snapshot is False

    def test_config_to_dict(self):
        """Test config serialization."""
        config = TUFConfig()
        data = config.to_dict()

        assert "timestamp_expiry_days" in data
        assert "root_threshold" in data
        assert "consistent_snapshot" in data


class TestTUFRepository:
    """Tests for TUFRepository."""

    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_repo):
        """Create repository config."""
        return TUFConfig(
            repository_path=temp_repo,
            root_threshold=1,  # Simpler for testing
        )

    @pytest.fixture
    def key_pairs(self):
        """Generate test key pairs (simplified for testing)."""

        # In real implementation, use proper ed25519 keys
        # Here we use placeholder bytes
        def make_key():
            pub = os.urandom(32)
            priv = os.urandom(64)  # Ed25519 private key is 64 bytes
            return (pub, priv)

        return {
            "root": [make_key()],
            "targets": make_key(),
            "snapshot": make_key(),
            "timestamp": make_key(),
        }

    @pytest.fixture
    async def initialized_repo(self, config, key_pairs):
        """Create initialized repository."""
        repo = TUFRepository(config)
        await repo.initialize(
            root_keys=key_pairs["root"],
            targets_key=key_pairs["targets"],
            snapshot_key=key_pairs["snapshot"],
            timestamp_key=key_pairs["timestamp"],
        )
        return repo

    def test_repository_creation(self, config):
        """Test creating repository."""
        repo = TUFRepository(config)

        assert repo.is_initialized is False
        assert repo.config == config

    @pytest.mark.asyncio
    async def test_initialize_repository(self, config, key_pairs):
        """Test initializing repository."""
        repo = TUFRepository(config)

        result = await repo.initialize(
            root_keys=key_pairs["root"],
            targets_key=key_pairs["targets"],
            snapshot_key=key_pairs["snapshot"],
            timestamp_key=key_pairs["timestamp"],
        )

        assert result is True
        assert repo.is_initialized is True
        assert config.repository_path.exists()

    @pytest.mark.asyncio
    async def test_initialize_creates_metadata(self, initialized_repo):
        """Test that initialization creates all metadata."""
        repo = initialized_repo

        versions = repo.get_metadata_versions()

        assert versions["root"] == ROOT_VERSION
        assert versions["timestamp"] == 1
        assert versions["snapshot"] == 1
        assert versions["targets"] == 1

    @pytest.mark.asyncio
    async def test_add_target(self, initialized_repo):
        """Test adding target to repository."""
        repo = initialized_repo
        content = b"This is the agent binary content"

        result = await repo.add_target(
            name="agent-1.0.0.tar.gz",
            content=content,
            custom={"version": "1.0.0"},
        )

        assert result is True

        info = repo.get_target_info("agent-1.0.0.tar.gz")
        assert info is not None
        assert info.length == len(content)
        assert "sha256" in info.hashes
        assert info.custom["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_add_multiple_targets(self, initialized_repo):
        """Test adding multiple targets."""
        repo = initialized_repo

        await repo.add_target("agent-1.0.0.tar.gz", b"v1 content")
        await repo.add_target("agent-1.1.0.tar.gz", b"v1.1 content")
        await repo.add_target("agent-1.2.0.tar.gz", b"v1.2 content")

        targets = repo.list_targets()

        assert len(targets) == 3
        assert "agent-1.0.0.tar.gz" in targets
        assert "agent-1.1.0.tar.gz" in targets

    @pytest.mark.asyncio
    async def test_remove_target(self, initialized_repo):
        """Test removing target from repository."""
        repo = initialized_repo

        await repo.add_target("to-remove.tar.gz", b"content")
        assert repo.get_target_info("to-remove.tar.gz") is not None

        result = await repo.remove_target("to-remove.tar.gz")

        assert result is True
        assert repo.get_target_info("to-remove.tar.gz") is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_target(self, initialized_repo):
        """Test removing non-existent target."""
        repo = initialized_repo

        result = await repo.remove_target("nonexistent.tar.gz")

        assert result is False

    @pytest.mark.asyncio
    async def test_publish(self, initialized_repo):
        """Test publishing repository."""
        repo = initialized_repo

        await repo.add_target("agent-1.0.0.tar.gz", b"content")

        result = await repo.publish()

        assert result is True

        # Check files written
        repo_path = repo.config.repository_path
        assert (repo_path / "root.json").exists()
        assert (repo_path / "timestamp.json").exists()

    @pytest.mark.asyncio
    async def test_publish_increments_versions(self, initialized_repo):
        """Test that publish increments metadata versions."""
        repo = initialized_repo

        initial_versions = repo.get_metadata_versions()

        await repo.add_target("agent.tar.gz", b"content")
        await repo.publish()

        new_versions = repo.get_metadata_versions()

        # All except root should increment
        assert new_versions["targets"] == initial_versions["targets"] + 1
        assert new_versions["snapshot"] == initial_versions["snapshot"] + 1
        assert new_versions["timestamp"] == initial_versions["timestamp"] + 1

    @pytest.mark.asyncio
    async def test_verify_target_success(self, initialized_repo):
        """Test successful target verification."""
        repo = initialized_repo
        content = b"agent binary content here"

        await repo.add_target("agent.tar.gz", content)
        await repo.publish()

        import hashlib

        content_hash = hashlib.sha256(content).hexdigest()

        is_valid, error = await repo.verify_target(
            name="agent.tar.gz",
            content_hash=content_hash,
            content_length=len(content),
        )

        assert is_valid is True
        assert error is None

    @pytest.mark.asyncio
    async def test_verify_target_wrong_hash(self, initialized_repo):
        """Test target verification with wrong hash."""
        repo = initialized_repo

        await repo.add_target("agent.tar.gz", b"real content")
        await repo.publish()

        is_valid, error = await repo.verify_target(
            name="agent.tar.gz",
            content_hash="wrong_hash",
            content_length=12,
        )

        assert is_valid is False
        assert "Hash mismatch" in error

    @pytest.mark.asyncio
    async def test_verify_target_wrong_length(self, initialized_repo):
        """Test target verification with wrong length."""
        repo = initialized_repo
        content = b"content"

        await repo.add_target("agent.tar.gz", content)
        await repo.publish()

        import hashlib

        content_hash = hashlib.sha256(content).hexdigest()

        is_valid, error = await repo.verify_target(
            name="agent.tar.gz",
            content_hash=content_hash,
            content_length=999,  # Wrong length
        )

        assert is_valid is False
        assert "Length mismatch" in error

    @pytest.mark.asyncio
    async def test_verify_target_not_found(self, initialized_repo):
        """Test verification of non-existent target."""
        repo = initialized_repo
        await repo.publish()

        is_valid, error = await repo.verify_target(
            name="nonexistent.tar.gz",
            content_hash="abc",
            content_length=100,
        )

        assert is_valid is False
        assert "not found" in error

    @pytest.mark.asyncio
    async def test_verify_expired_metadata(self, initialized_repo):
        """Test verification fails with expired metadata."""
        repo = initialized_repo

        await repo.add_target("agent.tar.gz", b"content")
        await repo.publish()

        # Force expiration
        repo._timestamp.expires = datetime.utcnow() - timedelta(days=1)

        is_valid, error = await repo.verify_target(
            name="agent.tar.gz",
            content_hash="abc",
            content_length=7,
        )

        assert is_valid is False
        assert "expired" in error.lower()

    @pytest.mark.asyncio
    async def test_rotate_root(self, initialized_repo, key_pairs):
        """Test root key rotation."""
        repo = initialized_repo
        initial_version = repo._root.version

        # Generate new root keys
        new_root_key = (os.urandom(32), os.urandom(64))

        result = await repo.rotate_root(
            new_root_keys=[new_root_key],
        )

        assert result is True
        assert repo._root.version == initial_version + 1

    @pytest.mark.asyncio
    async def test_load_repository(self, config, key_pairs):
        """Test loading existing repository."""
        # First create and publish
        repo1 = TUFRepository(config)
        await repo1.initialize(
            root_keys=key_pairs["root"],
            targets_key=key_pairs["targets"],
            snapshot_key=key_pairs["snapshot"],
            timestamp_key=key_pairs["timestamp"],
        )
        await repo1.add_target("agent.tar.gz", b"content")
        await repo1.publish()

        # Now load in new instance
        repo2 = TUFRepository(config)
        result = await repo2.load()

        assert result is True
        assert repo2.is_initialized is True

        # Targets should be loaded
        targets = repo2.list_targets()
        assert "agent.tar.gz" in targets

    def test_get_metadata_expiry(self, config):
        """Test getting metadata expiry times."""
        repo = TUFRepository(config)

        expiry = repo.get_metadata_expiry()

        # Not initialized, all should be None
        assert expiry["root"] is None
        assert expiry["timestamp"] is None

    @pytest.mark.asyncio
    async def test_get_metadata_expiry_after_init(self, initialized_repo):
        """Test getting expiry after initialization."""
        repo = initialized_repo

        expiry = repo.get_metadata_expiry()

        assert expiry["root"] is not None
        assert expiry["timestamp"] is not None
        assert expiry["root"] > datetime.utcnow()

    def test_get_statistics(self, config):
        """Test getting repository statistics."""
        repo = TUFRepository(config)

        stats = repo.get_statistics()

        assert stats["initialized"] is False
        assert stats["target_count"] == 0
        assert "versions" in stats
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_get_statistics_after_init(self, initialized_repo):
        """Test statistics after initialization."""
        repo = initialized_repo

        await repo.add_target("t1.tar.gz", b"1")
        await repo.add_target("t2.tar.gz", b"2")

        stats = repo.get_statistics()

        assert stats["initialized"] is True
        assert stats["target_count"] == 2
        assert stats["key_count"] > 0


class TestRollbackProtection:
    """Tests for rollback attack protection."""

    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_repo):
        """Create repository config."""
        return TUFConfig(
            repository_path=temp_repo,
            root_threshold=1,
        )

    @pytest.fixture
    def key_pairs(self):
        """Generate test key pairs."""

        def make_key():
            return (os.urandom(32), os.urandom(64))

        return {
            "root": [make_key()],
            "targets": make_key(),
            "snapshot": make_key(),
            "timestamp": make_key(),
        }

    @pytest.mark.asyncio
    async def test_version_numbers_increase(self, config, key_pairs):
        """Test that version numbers always increase."""
        repo = TUFRepository(config)
        await repo.initialize(
            root_keys=key_pairs["root"],
            targets_key=key_pairs["targets"],
            snapshot_key=key_pairs["snapshot"],
            timestamp_key=key_pairs["timestamp"],
        )

        versions_history = []
        for i in range(3):
            await repo.add_target(f"agent-{i}.tar.gz", f"content {i}".encode())
            await repo.publish()
            versions_history.append(repo.get_metadata_versions())

        # Each publish should increment versions
        for i in range(1, len(versions_history)):
            assert versions_history[i]["targets"] > versions_history[i - 1]["targets"]
            assert versions_history[i]["snapshot"] > versions_history[i - 1]["snapshot"]
            assert versions_history[i]["timestamp"] > versions_history[i - 1]["timestamp"]


class TestFreezeProtection:
    """Tests for freeze attack protection."""

    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_repo):
        """Create config with short expiry for testing."""
        return TUFConfig(
            repository_path=temp_repo,
            root_threshold=1,
            timestamp_expiry_days=1,
        )

    @pytest.fixture
    def key_pairs(self):
        """Generate test key pairs."""

        def make_key():
            return (os.urandom(32), os.urandom(64))

        return {
            "root": [make_key()],
            "targets": make_key(),
            "snapshot": make_key(),
            "timestamp": make_key(),
        }

    @pytest.mark.asyncio
    async def test_expiration_enforced(self, config, key_pairs):
        """Test that expiration is enforced."""
        repo = TUFRepository(config)
        await repo.initialize(
            root_keys=key_pairs["root"],
            targets_key=key_pairs["targets"],
            snapshot_key=key_pairs["snapshot"],
            timestamp_key=key_pairs["timestamp"],
        )

        await repo.add_target("agent.tar.gz", b"content")
        await repo.publish()

        # Simulate expired metadata
        repo._timestamp.expires = datetime.utcnow() - timedelta(hours=1)

        is_valid, error = await repo.verify_target("agent.tar.gz", "hash", 7)

        assert is_valid is False
        assert "expired" in error.lower()

    @pytest.mark.asyncio
    async def test_metadata_has_expiry(self, config, key_pairs):
        """Test all metadata has expiry dates."""
        repo = TUFRepository(config)
        await repo.initialize(
            root_keys=key_pairs["root"],
            targets_key=key_pairs["targets"],
            snapshot_key=key_pairs["snapshot"],
            timestamp_key=key_pairs["timestamp"],
        )

        expiry = repo.get_metadata_expiry()

        for role, exp_time in expiry.items():
            assert exp_time is not None, f"{role} should have expiry"
            assert exp_time > datetime.utcnow(), f"{role} should not be expired"


class TestConsistentSnapshots:
    """Tests for consistent snapshot feature."""

    @pytest.fixture
    def temp_repo(self):
        """Create temporary repository directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config_consistent(self, temp_repo):
        """Create config with consistent snapshots enabled."""
        return TUFConfig(
            repository_path=temp_repo,
            root_threshold=1,
            consistent_snapshot=True,
        )

    @pytest.fixture
    def config_no_consistent(self, temp_repo):
        """Create config without consistent snapshots."""
        return TUFConfig(
            repository_path=temp_repo,
            root_threshold=1,
            consistent_snapshot=False,
        )

    @pytest.fixture
    def key_pairs(self):
        """Generate test key pairs."""

        def make_key():
            return (os.urandom(32), os.urandom(64))

        return {
            "root": [make_key()],
            "targets": make_key(),
            "snapshot": make_key(),
            "timestamp": make_key(),
        }

    @pytest.mark.asyncio
    async def test_consistent_snapshot_versioned_files(self, config_consistent, key_pairs):
        """Test consistent snapshots create versioned files."""
        repo = TUFRepository(config_consistent)
        await repo.initialize(
            root_keys=key_pairs["root"],
            targets_key=key_pairs["targets"],
            snapshot_key=key_pairs["snapshot"],
            timestamp_key=key_pairs["timestamp"],
        )

        await repo.add_target("agent.tar.gz", b"content")
        await repo.publish()

        repo_path = repo.config.repository_path
        versions = repo.get_metadata_versions()

        # Versioned files should exist
        versioned_snapshot = repo_path / f"{versions['snapshot']}.snapshot.json"
        versioned_targets = repo_path / f"{versions['targets']}.targets.json"

        assert versioned_snapshot.exists()
        assert versioned_targets.exists()
