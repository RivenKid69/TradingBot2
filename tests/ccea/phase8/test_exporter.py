# -*- coding: utf-8 -*-
"""
Tests for TelemetryExporter.

CCEA Phase 8 - Privacy-controlled export tests.
"""

import json
import gzip
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# Skip entire module if cryptography is not available
try:
    from packages.agent.telemetry.exporter import (
        TelemetryExporter,
        ExportConfig,
        ExportFormat,
        ExportResult,
        ExportScope,
        MAX_EXPORT_SIZE_BYTES,
    )
except ImportError:
    pytestmark = pytest.mark.skip(reason="cryptography not available")


class TestExporterBasic:
    """Basic exporter tests."""

    def test_create_exporter(self):
        """Test creating exporter."""
        exporter = TelemetryExporter()
        assert exporter is not None

    def test_export_to_memory_json(self):
        """Test exporting to memory as JSON."""
        exporter = TelemetryExporter()
        data = [{"status": "ok", "count": 42}]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSON),
        )

        assert result.success is True
        assert result.content is not None
        assert result.record_count == 1
        assert result.format == ExportFormat.JSON

    def test_export_to_memory_jsonl(self):
        """Test exporting to memory as JSONL."""
        exporter = TelemetryExporter()
        data = [
            {"status": "ok"},
            {"status": "running"},
        ]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSONL),
        )

        assert result.success is True
        # Each record on a line
        lines = result.content.decode().strip().split("\n")
        assert len(lines) >= 2  # Data lines + potential metadata


class TestExportFormats:
    """Export format tests."""

    def test_json_format(self):
        """Test JSON format output."""
        exporter = TelemetryExporter()
        data = [{"key": "value"}]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSON, include_metadata=False),
        )

        parsed = json.loads(result.content.decode())
        assert "records" in parsed
        assert parsed["records"][0]["key"] == "value"

    def test_json_compact_format(self):
        """Test compact JSON format."""
        exporter = TelemetryExporter()
        data = [{"key": "value"}]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSON_COMPACT, include_metadata=False),
        )

        content = result.content.decode()
        # Compact should have minimal whitespace
        assert "\n" not in content or content.count("\n") <= 1

    def test_csv_format(self):
        """Test CSV format output."""
        exporter = TelemetryExporter()
        data = [
            {"name": "test", "value": 123},
            {"name": "test2", "value": 456},
        ]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.CSV),
        )

        assert result.success is True
        content = result.content.decode()
        assert "name" in content
        assert "value" in content

    def test_jsonl_format(self):
        """Test JSONL format output."""
        exporter = TelemetryExporter()
        data = [{"id": 1}, {"id": 2}]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSONL, include_metadata=False),
        )

        lines = result.content.decode().strip().split("\n")
        for line in lines:
            # Each line should be valid JSON
            parsed = json.loads(line)
            assert "id" in parsed


class TestExportToFile:
    """Export to file tests."""

    def test_export_to_file(self):
        """Test exporting to file."""
        exporter = TelemetryExporter()
        data = [{"status": "ok"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "export.json"

            result = exporter.export_to_file(
                data=data,
                path=path,
                config=ExportConfig(format=ExportFormat.JSON),
            )

            assert result.success is True
            assert result.file_path == path
            assert path.exists()

            # Verify content
            with open(path) as f:
                content = json.load(f)
            assert "records" in content

    def test_export_creates_directories(self):
        """Test export creates parent directories."""
        exporter = TelemetryExporter()
        data = [{"status": "ok"}]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "export.json"

            result = exporter.export_to_file(
                data=data,
                path=path,
            )

            assert result.success is True
            assert path.exists()


class TestCompression:
    """Compression tests."""

    def test_gzip_compression(self):
        """Test gzip compression."""
        exporter = TelemetryExporter()
        data = [{"status": "ok", "data": "x" * 1000}]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(compress=True),
        )

        assert result.success is True
        assert result.compressed is True

        # Verify it's valid gzip
        decompressed = gzip.decompress(result.content)
        assert len(decompressed) > len(result.content)  # Uncompressed is larger


class TestEncryption:
    """Encryption tests."""

    def test_encrypted_export(self):
        """Test encrypted export."""
        exporter = TelemetryExporter()
        data = [{"secret_data": "value"}]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(
                format=ExportFormat.ENCRYPTED,
                encrypt=True,
                encryption_password="test-password-123",
            ),
        )

        assert result.success is True
        assert result.encrypted is True

        # Content should not contain plaintext
        assert b"secret_data" not in result.content

    def test_decrypt_export(self):
        """Test decrypting exported data."""
        exporter = TelemetryExporter()
        password = "secure-password-123"
        data = [{"message": "hello world"}]

        # Encrypt
        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(
                format=ExportFormat.JSON,
                encrypt=True,
                encryption_password=password,
            ),
        )

        # Decrypt
        decrypted = exporter.decrypt_content(result.content, password)

        parsed = json.loads(decrypted.decode())
        assert "records" in parsed
        assert parsed["records"][0]["message"] == "hello world"


class TestRedaction:
    """Redaction during export tests."""

    def test_secrets_redacted_on_export(self):
        """Test secrets are redacted during export."""
        exporter = TelemetryExporter()
        data = [{"api_key": "secret123", "status": "ok"}]

        result = exporter.export_to_memory(data=data)

        content = result.content.decode()
        assert "secret123" not in content
        assert "status" in content


class TestPrivacyTransforms:
    """Privacy transformation tests."""

    def test_anonymize_ids(self):
        """Test ID anonymization."""
        exporter = TelemetryExporter()
        data = [
            {"id": "user-123", "agent_id": "agent-456"},
        ]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSON, anonymize_ids=True),
        )

        content = json.loads(result.content.decode())
        records = content.get("records", [])

        # IDs should be anonymized
        assert records[0]["id"].startswith("anon_")
        assert records[0]["agent_id"].startswith("anon_")

    def test_consistent_anonymization(self):
        """Test IDs are consistently anonymized."""
        exporter = TelemetryExporter()
        data = [
            {"id": "user-123"},
            {"id": "user-123"},  # Same ID
            {"id": "user-456"},  # Different ID
        ]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSON, anonymize_ids=True),
        )

        content = json.loads(result.content.decode())
        records = content.get("records", [])

        # Same original ID should have same anonymized ID
        assert records[0]["id"] == records[1]["id"]
        # Different original ID should have different anonymized ID
        assert records[0]["id"] != records[2]["id"]

    def test_remove_timestamps(self):
        """Test timestamp removal."""
        exporter = TelemetryExporter()
        data = [
            {"status": "ok", "timestamp": "2024-01-01T00:00:00", "created_at": "2024-01-01"},
        ]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSON, remove_timestamps=True),
        )

        content = json.loads(result.content.decode())
        records = content.get("records", [])

        assert "timestamp" not in records[0]
        assert "created_at" not in records[0]
        assert "status" in records[0]


class TestFiltering:
    """Export filtering tests."""

    def test_filter_by_time_range(self):
        """Test filtering by time range."""
        exporter = TelemetryExporter()
        now = datetime.utcnow()
        data = [
            {"id": 1, "timestamp": (now - timedelta(days=5)).isoformat()},
            {"id": 2, "timestamp": (now - timedelta(days=1)).isoformat()},
            {"id": 3, "timestamp": (now + timedelta(days=1)).isoformat()},
        ]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(
                format=ExportFormat.JSON,
                time_range=(now - timedelta(days=3), now),
            ),
        )

        content = json.loads(result.content.decode())
        records = content.get("records", [])

        # Only record 2 should be in range
        assert len(records) == 1
        assert records[0]["id"] == 2

    def test_filter_by_run_ids(self):
        """Test filtering by run IDs."""
        exporter = TelemetryExporter()
        data = [
            {"run_id": "run-1", "status": "ok"},
            {"run_id": "run-2", "status": "ok"},
            {"run_id": "run-3", "status": "ok"},
        ]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSON, run_ids=["run-1", "run-3"]),
        )

        content = json.loads(result.content.decode())
        records = content.get("records", [])

        assert len(records) == 2
        run_ids = [r["run_id"] for r in records]
        assert "run-1" in run_ids
        assert "run-3" in run_ids
        assert "run-2" not in run_ids

    def test_filter_by_agent_ids(self):
        """Test filtering by agent IDs."""
        exporter = TelemetryExporter()
        data = [
            {"agent_id": "agent-1"},
            {"agent_id": "agent-2"},
        ]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSON, agent_ids=["agent-1"]),
        )

        content = json.loads(result.content.decode())
        records = content.get("records", [])

        assert len(records) == 1
        assert records[0]["agent_id"] == "agent-1"

    def test_max_records_limit(self):
        """Test max records limit."""
        exporter = TelemetryExporter()
        data = [{"id": i} for i in range(100)]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(max_records=10),
        )

        assert result.record_count == 10


class TestMetadata:
    """Export metadata tests."""

    def test_metadata_included(self):
        """Test metadata is included by default."""
        exporter = TelemetryExporter()
        data = [{"status": "ok"}]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(format=ExportFormat.JSON, include_metadata=True),
        )

        content = json.loads(result.content.decode())
        assert "_metadata" in content
        assert "export_id" in content["_metadata"]
        assert "redaction_applied" in content["_metadata"]

    def test_metadata_excluded(self):
        """Test metadata can be excluded."""
        exporter = TelemetryExporter()
        data = [{"status": "ok"}]

        result = exporter.export_to_memory(
            data=data,
            config=ExportConfig(
                format=ExportFormat.JSON,
                include_metadata=False,
            ),
        )

        content = json.loads(result.content.decode())
        assert "_metadata" not in content


class TestExportResult:
    """Export result tests."""

    def test_result_checksum(self):
        """Test checksum is generated."""
        exporter = TelemetryExporter()
        result = exporter.export_to_memory(data=[{"status": "ok"}])

        assert result.checksum is not None
        assert len(result.checksum) == 64  # SHA-256 hex

    def test_result_to_dict(self):
        """Test result serialization."""
        exporter = TelemetryExporter()
        result = exporter.export_to_memory(data=[{"status": "ok"}])

        data = result.to_dict()

        assert "export_id" in data
        assert "success" in data
        assert "format" in data
        assert "record_count" in data
        assert "checksum" in data


class TestExportLog:
    """Export audit log tests."""

    def test_export_logged(self):
        """Test exports are logged."""
        exporter = TelemetryExporter()

        exporter.export_to_memory(data=[{"status": "ok"}])

        log = exporter.get_export_log()
        assert len(log) > 0

    def test_log_limit(self):
        """Test log respects limit."""
        exporter = TelemetryExporter()

        # Generate multiple exports
        for _ in range(10):
            exporter.export_to_memory(data=[{"status": "ok"}])

        log = exporter.get_export_log(limit=5)
        assert len(log) <= 5


class TestStreamExport:
    """Stream export tests."""

    def test_export_stream(self):
        """Test streaming export."""
        import io
        exporter = TelemetryExporter()

        def data_iterator():
            for i in range(5):
                yield {"id": i}

        output = io.BytesIO()
        result = exporter.export_stream(
            data_iterator=data_iterator(),
            output=output,
            config=ExportConfig(format=ExportFormat.JSONL),
        )

        assert result.success is True
        assert result.record_count == 5
        assert result.size_bytes > 0

    def test_stream_respects_max_records(self):
        """Test stream export respects max records."""
        import io
        exporter = TelemetryExporter()

        def data_iterator():
            for i in range(100):
                yield {"id": i}

        output = io.BytesIO()
        result = exporter.export_stream(
            data_iterator=data_iterator(),
            output=output,
            config=ExportConfig(max_records=10),
        )

        assert result.record_count == 10


class TestAnonymizationMapping:
    """Anonymization mapping tests."""

    def test_clear_mapping(self):
        """Test clearing anonymization mapping."""
        exporter = TelemetryExporter()

        # Generate some mappings
        exporter.export_to_memory(
            data=[{"id": "user-123"}],
            config=ExportConfig(anonymize_ids=True),
        )

        # Clear
        exporter.clear_anonymization_mapping()

        # Verify mapping was cleared by checking internal state
        assert len(exporter._id_mapping) == 0, (
            "ID anonymization mapping should be empty after clear()"
        )
