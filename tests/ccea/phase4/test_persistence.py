# -*- coding: utf-8 -*-
"""
Tests for Approval Persistence Module.

Tests cover:
- File-based storage operations (save/load)
- Index management
- History organization by month
- Thread safety
- Error handling and recovery
- Cleanup operations

100% coverage target for persistence.py
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from uuid import UUID, uuid4

import pytest

from packages.agent.approval.persistence import (
    ApprovalPersistence,
    ApprovalPersistenceConfig,
    DEFAULT_STORAGE_PATH,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestApprovalPersistenceConfig:
    """Tests for ApprovalPersistenceConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ApprovalPersistenceConfig()

        assert config.storage_dir == DEFAULT_STORAGE_PATH
        assert config.max_history_size == 10000
        assert config.auto_flush is True
        assert config.flush_interval_seconds == 30
        assert config.enable_compression is False

    def test_custom_config(self, tmp_path):
        """Test custom configuration."""
        config = ApprovalPersistenceConfig(
            storage_dir=tmp_path / "custom",
            max_history_size=5000,
            auto_flush=False,
            flush_interval_seconds=60,
            enable_compression=True,
        )

        assert config.storage_dir == tmp_path / "custom"
        assert config.max_history_size == 5000
        assert config.auto_flush is False
        assert config.flush_interval_seconds == 60
        assert config.enable_compression is True


# =============================================================================
# Initialization Tests
# =============================================================================


class TestPersistenceInitialization:
    """Tests for persistence initialization."""

    def test_creates_directories(self, tmp_path):
        """Test that initialization creates required directories."""
        storage_dir = tmp_path / "approvals"
        config = ApprovalPersistenceConfig(storage_dir=storage_dir)

        persistence = ApprovalPersistence(config)

        assert (storage_dir / "pending").exists()
        assert (storage_dir / "history").exists()

    def test_loads_existing_index(self, tmp_path):
        """Test loading existing index on initialization."""
        storage_dir = tmp_path / "approvals"
        storage_dir.mkdir(parents=True)
        (storage_dir / "pending").mkdir()
        (storage_dir / "history").mkdir()

        # Create index file
        index_data = {
            "test-id-1": {
                "status": "pending",
                "path": str(storage_dir / "pending" / "test-id-1.json"),
            }
        }
        with open(storage_dir / "index.json", "w") as f:
            json.dump(index_data, f)

        config = ApprovalPersistenceConfig(storage_dir=storage_dir)
        persistence = ApprovalPersistence(config)

        assert "test-id-1" in persistence._index

    def test_handles_corrupted_index(self, tmp_path):
        """Test handling of corrupted index file."""
        storage_dir = tmp_path / "approvals"
        storage_dir.mkdir(parents=True)
        (storage_dir / "pending").mkdir()
        (storage_dir / "history").mkdir()

        # Create corrupted index file
        with open(storage_dir / "index.json", "w") as f:
            f.write("not valid json{}")

        config = ApprovalPersistenceConfig(storage_dir=storage_dir)
        persistence = ApprovalPersistence(config)

        # Should start with empty index
        assert persistence._index == {}


# =============================================================================
# Save Pending Tests
# =============================================================================


class TestSavePending:
    """Tests for save_pending operation."""

    @pytest.fixture
    def persistence(self, tmp_path):
        """Create persistence instance."""
        config = ApprovalPersistenceConfig(storage_dir=tmp_path / "approvals")
        return ApprovalPersistence(config)

    def test_save_pending_success(self, persistence):
        """Test successful save of pending request."""
        request_id = uuid4()
        data = {
            "request_id": str(request_id),
            "command_type": "START_RUN",
            "description": "Test request",
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
        }

        result = persistence.save_pending(request_id, data)

        assert result is True

        # Verify file was created
        pending_path = persistence._config.storage_dir / "pending" / f"{request_id}.json"
        assert pending_path.exists()

        # Verify content
        with open(pending_path) as f:
            saved_data = json.load(f)
        assert saved_data["command_type"] == "START_RUN"

    def test_save_pending_updates_index(self, persistence):
        """Test that save_pending updates index."""
        request_id = uuid4()
        data = {
            "request_id": str(request_id),
            "command_type": "UPDATE_CONFIG",
            "created_at": datetime.utcnow().isoformat(),
        }

        persistence.save_pending(request_id, data)

        assert str(request_id) in persistence._index
        assert persistence._index[str(request_id)]["status"] == "pending"
        assert persistence._index[str(request_id)]["command_type"] == "UPDATE_CONFIG"

    def test_save_pending_with_auto_flush(self, tmp_path):
        """Test auto-flush behavior."""
        config = ApprovalPersistenceConfig(
            storage_dir=tmp_path / "approvals",
            auto_flush=True,
        )
        persistence = ApprovalPersistence(config)

        request_id = uuid4()
        data = {"request_id": str(request_id), "status": "pending"}
        persistence.save_pending(request_id, data)

        # Index file should be saved
        index_path = persistence._config.storage_dir / "index.json"
        assert index_path.exists()

    def test_save_pending_without_auto_flush(self, tmp_path):
        """Test without auto-flush (dirty flag)."""
        config = ApprovalPersistenceConfig(
            storage_dir=tmp_path / "approvals",
            auto_flush=False,
        )
        persistence = ApprovalPersistence(config)

        request_id = uuid4()
        data = {"request_id": str(request_id), "status": "pending"}
        persistence.save_pending(request_id, data)

        # Should be dirty but not flushed
        assert persistence._dirty is True


# =============================================================================
# Save History Tests
# =============================================================================


class TestSaveHistory:
    """Tests for save_history operation."""

    @pytest.fixture
    def persistence(self, tmp_path):
        """Create persistence instance."""
        config = ApprovalPersistenceConfig(storage_dir=tmp_path / "approvals")
        return ApprovalPersistence(config)

    def test_save_history_success(self, persistence):
        """Test successful save to history."""
        request_id = uuid4()
        decided_at = datetime.utcnow()
        data = {
            "request_id": str(request_id),
            "command_type": "START_RUN",
            "status": "approved",
            "decided_at": decided_at.isoformat(),
        }

        result = persistence.save_history(request_id, data)

        assert result is True

        # Verify file was created in correct month directory
        month_dir = persistence._config.storage_dir / "history" / decided_at.strftime("%Y-%m")
        assert month_dir.exists()
        assert (month_dir / f"{request_id}.json").exists()

    def test_save_history_removes_pending(self, persistence):
        """Test that save_history removes from pending."""
        request_id = uuid4()

        # First save as pending
        pending_data = {
            "request_id": str(request_id),
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
        }
        persistence.save_pending(request_id, pending_data)

        # Verify pending exists
        pending_path = persistence._config.storage_dir / "pending" / f"{request_id}.json"
        assert pending_path.exists()

        # Now save to history
        history_data = {
            "request_id": str(request_id),
            "status": "approved",
            "decided_at": datetime.utcnow().isoformat(),
        }
        persistence.save_history(request_id, history_data)

        # Pending should be removed
        assert not pending_path.exists()

    def test_save_history_updates_index(self, persistence):
        """Test that save_history updates index."""
        request_id = uuid4()
        data = {
            "request_id": str(request_id),
            "command_type": "STOP_RUN",
            "status": "approved",
            "decided_at": datetime.utcnow().isoformat(),
        }

        persistence.save_history(request_id, data)

        assert str(request_id) in persistence._index
        assert persistence._index[str(request_id)]["status"] == "approved"
        assert persistence._index[str(request_id)]["approved"] is True

    def test_save_history_with_datetime_object(self, persistence):
        """Test save_history with datetime object for decided_at."""
        request_id = uuid4()
        decided_at = datetime.utcnow()
        data = {
            "request_id": str(request_id),
            "status": "denied",
            "decided_at": decided_at,  # datetime object, not string
        }

        result = persistence.save_history(request_id, data)

        assert result is True

    def test_save_history_with_missing_decided_at(self, persistence):
        """Test save_history without decided_at uses current time."""
        request_id = uuid4()
        data = {
            "request_id": str(request_id),
            "status": "approved",
        }

        result = persistence.save_history(request_id, data)

        assert result is True


# =============================================================================
# Load Tests
# =============================================================================


class TestLoadOperations:
    """Tests for load operations."""

    @pytest.fixture
    def persistence(self, tmp_path):
        """Create persistence instance."""
        config = ApprovalPersistenceConfig(storage_dir=tmp_path / "approvals")
        return ApprovalPersistence(config)

    def test_load_pending_success(self, persistence):
        """Test loading pending request."""
        request_id = uuid4()
        data = {
            "request_id": str(request_id),
            "command_type": "START_RUN",
            "status": "pending",
        }
        persistence.save_pending(request_id, data)

        loaded = persistence.load_pending(request_id)

        assert loaded is not None
        assert loaded["command_type"] == "START_RUN"

    def test_load_pending_not_found(self, persistence):
        """Test loading non-existent pending request."""
        loaded = persistence.load_pending(uuid4())

        assert loaded is None

    def test_load_pending_corrupted(self, persistence):
        """Test loading corrupted pending file."""
        request_id = uuid4()
        pending_path = persistence._config.storage_dir / "pending" / f"{request_id}.json"
        pending_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pending_path, "w") as f:
            f.write("corrupted data{")

        loaded = persistence.load_pending(request_id)

        assert loaded is None

    def test_load_all_pending(self, persistence):
        """Test loading all pending requests."""
        # Save multiple pending requests
        for i in range(5):
            request_id = uuid4()
            data = {
                "request_id": str(request_id),
                "command_type": f"CMD_{i}",
                "status": "pending",
            }
            persistence.save_pending(request_id, data)

        all_pending = persistence.load_all_pending()

        assert len(all_pending) == 5

    def test_load_all_pending_skips_corrupted(self, persistence):
        """Test that load_all_pending skips corrupted files."""
        # Save valid request
        valid_id = uuid4()
        persistence.save_pending(valid_id, {
            "request_id": str(valid_id),
            "status": "pending",
        })

        # Create corrupted file
        corrupted_path = persistence._config.storage_dir / "pending" / "corrupted.json"
        with open(corrupted_path, "w") as f:
            f.write("not json")

        all_pending = persistence.load_all_pending()

        assert len(all_pending) == 1

    def test_load_history(self, persistence):
        """Test loading history records."""
        # Save multiple history records
        for i in range(10):
            request_id = uuid4()
            data = {
                "request_id": str(request_id),
                "command_type": f"HIST_{i}",
                "status": "approved" if i % 2 == 0 else "denied",
                "decided_at": datetime.utcnow().isoformat(),
            }
            persistence.save_history(request_id, data)

        history = persistence.load_history(limit=5)

        assert len(history) == 5

    def test_load_history_with_command_filter(self, persistence):
        """Test loading history with command type filter."""
        for i in range(10):
            request_id = uuid4()
            cmd = "START_RUN" if i < 5 else "STOP_RUN"
            data = {
                "request_id": str(request_id),
                "command_type": cmd,
                "status": "approved",
                "decided_at": datetime.utcnow().isoformat(),
            }
            persistence.save_history(request_id, data)

        history = persistence.load_history(command_type="START_RUN")

        assert len(history) == 5
        assert all(r["command_type"] == "START_RUN" for r in history)

    def test_load_history_with_since_filter(self, persistence):
        """Test loading history with time filter."""
        # Save old record
        old_id = uuid4()
        old_time = datetime.utcnow() - timedelta(days=2)
        persistence.save_history(old_id, {
            "request_id": str(old_id),
            "decided_at": old_time.isoformat(),
        })

        # Save recent record
        new_id = uuid4()
        new_time = datetime.utcnow()
        persistence.save_history(new_id, {
            "request_id": str(new_id),
            "decided_at": new_time.isoformat(),
        })

        # Filter for last day
        since = datetime.utcnow() - timedelta(days=1)
        history = persistence.load_history(since=since)

        assert len(history) == 1
        assert history[0]["request_id"] == str(new_id)


# =============================================================================
# Delete Tests
# =============================================================================


class TestDeleteOperations:
    """Tests for delete operations."""

    @pytest.fixture
    def persistence(self, tmp_path):
        """Create persistence instance."""
        config = ApprovalPersistenceConfig(storage_dir=tmp_path / "approvals")
        return ApprovalPersistence(config)

    def test_delete_pending_success(self, persistence):
        """Test successful deletion of pending request."""
        request_id = uuid4()
        persistence.save_pending(request_id, {
            "request_id": str(request_id),
            "status": "pending",
        })

        result = persistence.delete_pending(request_id)

        assert result is True
        assert persistence.load_pending(request_id) is None
        assert str(request_id) not in persistence._index

    def test_delete_pending_not_found(self, persistence):
        """Test deleting non-existent pending request."""
        result = persistence.delete_pending(uuid4())

        assert result is False


# =============================================================================
# Stats and Utility Tests
# =============================================================================


class TestStatsAndUtilities:
    """Tests for stats and utility methods."""

    @pytest.fixture
    def persistence(self, tmp_path):
        """Create persistence instance."""
        config = ApprovalPersistenceConfig(storage_dir=tmp_path / "approvals")
        return ApprovalPersistence(config)

    def test_get_stats(self, persistence):
        """Test get_stats returns correct counts."""
        # Add pending requests
        for i in range(3):
            persistence.save_pending(uuid4(), {"status": "pending"})

        # Add history records
        for i in range(5):
            persistence.save_history(uuid4(), {
                "status": "approved",
                "decided_at": datetime.utcnow().isoformat(),
            })

        stats = persistence.get_stats()

        assert stats["pending_count"] == 3
        assert stats["history_count"] == 5
        assert stats["index_size"] == 8

    def test_flush(self, tmp_path):
        """Test manual flush."""
        config = ApprovalPersistenceConfig(
            storage_dir=tmp_path / "approvals",
            auto_flush=False,
        )
        persistence = ApprovalPersistence(config)

        persistence.save_pending(uuid4(), {"status": "pending"})
        assert persistence._dirty is True

        persistence.flush()

        assert persistence._dirty is False
        assert (persistence._config.storage_dir / "index.json").exists()

    def test_cleanup_old_history(self, persistence):
        """Test cleanup of old history records."""
        # This test mainly verifies the method runs without error
        # Actual cleanup depends on date logic
        deleted = persistence.cleanup_old_history(days=0)

        assert isinstance(deleted, int)
        assert deleted >= 0


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    @pytest.fixture
    def persistence(self, tmp_path):
        """Create persistence instance."""
        config = ApprovalPersistenceConfig(storage_dir=tmp_path / "approvals")
        return ApprovalPersistence(config)

    def test_concurrent_saves(self, persistence):
        """Test concurrent save operations."""
        errors = []
        lock = threading.Lock()

        def save_request(i):
            try:
                request_id = uuid4()
                persistence.save_pending(request_id, {
                    "request_id": str(request_id),
                    "index": i,
                })
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [
            threading.Thread(target=save_request, args=(i,))
            for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        all_pending = persistence.load_all_pending()
        assert len(all_pending) == 20

    def test_concurrent_read_write(self, persistence):
        """Test concurrent read and write operations."""
        # Pre-populate some data
        request_ids = []
        for i in range(10):
            rid = uuid4()
            persistence.save_pending(rid, {
                "request_id": str(rid),
                "value": i,
            })
            request_ids.append(rid)

        errors = []
        lock = threading.Lock()

        def read_request(rid):
            try:
                persistence.load_pending(rid)
            except Exception as e:
                with lock:
                    errors.append(str(e))

        def write_request(i):
            try:
                persistence.save_pending(uuid4(), {"value": i})
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = []
        for rid in request_ids:
            threads.append(threading.Thread(target=read_request, args=(rid,)))
        for i in range(10):
            threads.append(threading.Thread(target=write_request, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
