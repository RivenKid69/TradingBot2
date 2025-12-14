# -*- coding: utf-8 -*-
"""
Tests for Evidence Pack Exporter.

Design Doc Reference: Phase 9 - Enterprise/on-prem pack, Design Doc 16.1
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from packages.cloud.enterprise.evidence_pack import (
    EvidencePackExporter,
    EvidencePackConfig,
    EvidencePack,
    EvidenceItem,
    EvidenceType,
    ExportFormat,
    ExportDestination,
)


class TestEvidencePackConfig:
    """Tests for EvidencePackConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EvidencePackConfig()

        assert config.output_format == ExportFormat.JSON
        assert config.compress is True
        assert config.redact_pii is True
        assert config.days_back == 30

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvidencePackConfig(
            output_format=ExportFormat.JSON_LINES,
            compress=False,
            include_sensitive=True,
            days_back=90,
        )

        assert config.output_format == ExportFormat.JSON_LINES
        assert config.compress is False
        assert config.include_sensitive is True

    def test_config_to_dict(self):
        """Test config serialization."""
        config = EvidencePackConfig()
        data = config.to_dict()

        assert "output_format" in data
        assert "evidence_types" in data
        assert "redact_pii" in data


class TestEvidenceItem:
    """Tests for EvidenceItem."""

    def test_evidence_item_creation(self):
        """Test creating evidence item."""
        item = EvidenceItem(
            type=EvidenceType.APPROVAL_RECORDS,
            id="test-123",
            timestamp=datetime.utcnow(),
            data={"approved": True},
            source="cloud",
        )

        assert item.type == EvidenceType.APPROVAL_RECORDS
        assert item.id == "test-123"
        assert item.data["approved"] is True

    def test_evidence_item_to_dict(self):
        """Test evidence item serialization."""
        item = EvidenceItem(
            type=EvidenceType.DEPLOYMENT_LOGS,
            id="deploy-456",
            timestamp=datetime.utcnow(),
            data={"status": "success"},
            source="agent",
            workspace_id=uuid4(),
        )

        data = item.to_dict()

        assert data["type"] == "deployment_logs"
        assert data["source"] == "agent"
        assert "workspace_id" in data


class TestEvidencePack:
    """Tests for EvidencePack."""

    def test_evidence_pack_creation(self):
        """Test creating evidence pack."""
        pack = EvidencePack(
            description="Test pack",
            created_by="test_user",
        )

        assert pack.description == "Test pack"
        assert pack.created_by == "test_user"
        assert pack.is_complete is False
        assert pack.version == "1.0.0"

    def test_evidence_pack_to_dict(self):
        """Test evidence pack serialization."""
        pack = EvidencePack(
            evidence_types=[EvidenceType.ARTIFACT_DIGESTS],
            item_count=10,
        )

        data = pack.to_dict()

        assert "id" in data
        assert data["item_count"] == 10
        assert "artifact_digests" in data["evidence_types"]


class TestEvidencePackExporter:
    """Tests for EvidencePackExporter."""

    @pytest.fixture
    def temp_output(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def exporter_config(self, temp_output):
        """Create exporter configuration."""
        return EvidencePackConfig(
            output_path=temp_output,
            evidence_types=[
                EvidenceType.ARTIFACT_DIGESTS,
                EvidenceType.APPROVAL_RECORDS,
            ],
            days_back=30,
        )

    @pytest.fixture
    def exporter(self, exporter_config):
        """Create exporter instance."""
        return EvidencePackExporter(exporter_config)

    @pytest.mark.asyncio
    async def test_export_empty_pack(self, exporter):
        """Test exporting empty evidence pack."""
        pack = await exporter.export(
            description="Empty test pack",
            created_by="test",
        )

        assert pack.is_complete is True
        assert pack.item_count == 0
        assert pack.archive_path is not None
        assert pack.archive_path.exists()

    @pytest.mark.asyncio
    async def test_export_with_time_range(self, temp_output):
        """Test export with custom time range."""
        config = EvidencePackConfig(
            output_path=temp_output,
            start_time=datetime.utcnow() - timedelta(days=7),
            end_time=datetime.utcnow(),
        )
        exporter = EvidencePackExporter(config)

        pack = await exporter.export()

        assert pack.start_time is not None
        assert pack.end_time is not None
        assert (pack.end_time - pack.start_time).days == 7

    @pytest.mark.asyncio
    async def test_export_compressed(self, temp_output):
        """Test compressed export."""
        config = EvidencePackConfig(
            output_path=temp_output,
            compress=True,
        )
        exporter = EvidencePackExporter(config)

        pack = await exporter.export()

        assert pack.is_complete is True
        assert pack.archive_path is not None
        # Compressed files end with .zip.gz
        assert str(pack.archive_path).endswith(".zip.gz")

    @pytest.mark.asyncio
    async def test_export_uncompressed(self, temp_output):
        """Test uncompressed export."""
        config = EvidencePackConfig(
            output_path=temp_output,
            compress=False,
        )
        exporter = EvidencePackExporter(config)

        pack = await exporter.export()

        assert pack.archive_path.suffix == ".zip"

    @pytest.mark.asyncio
    async def test_export_signed(self, temp_output):
        """Test signed export."""
        config = EvidencePackConfig(
            output_path=temp_output,
            sign_pack=True,
        )
        exporter = EvidencePackExporter(config)

        pack = await exporter.export()

        assert pack.signature is not None
        assert pack.signed_by is not None

    @pytest.mark.asyncio
    async def test_incremental_export(self, temp_output):
        """Test incremental export."""
        config = EvidencePackConfig(output_path=temp_output)
        exporter = EvidencePackExporter(config)

        since = datetime.utcnow() - timedelta(hours=1)
        pack = await exporter.export_incremental(since=since)

        assert pack.start_time is not None
        assert pack.start_time >= since

    @pytest.mark.asyncio
    async def test_verify_pack_success(self, temp_output):
        """Test successful pack verification."""
        config = EvidencePackConfig(
            output_path=temp_output,
            compress=False,
        )
        exporter = EvidencePackExporter(config)

        # Export pack
        pack = await exporter.export()

        # Verify pack
        is_valid, errors = await exporter.verify_pack(pack.archive_path)

        assert is_valid is True
        assert len(errors) == 0

    def test_list_packs(self, temp_output):
        """Test listing exported packs."""
        config = EvidencePackConfig(output_path=temp_output)
        exporter = EvidencePackExporter(config)

        # Initially empty
        packs = exporter.list_packs()
        assert len(packs) == 0

    @pytest.mark.asyncio
    async def test_list_packs_after_export(self, temp_output):
        """Test listing packs after export."""
        config = EvidencePackConfig(
            output_path=temp_output,
            compress=False,
        )
        exporter = EvidencePackExporter(config)

        # Export a pack
        await exporter.export()

        # List packs
        packs = exporter.list_packs()
        assert len(packs) >= 1


class TestEvidenceRedaction:
    """Tests for PII redaction."""

    @pytest.fixture
    def temp_output(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_redact_sensitive_fields(self, temp_output):
        """Test redaction of sensitive fields."""
        config = EvidencePackConfig(
            output_path=temp_output,
            redact_pii=True,
        )
        exporter = EvidencePackExporter(config)

        item = EvidenceItem(
            type=EvidenceType.ACCESS_AUDIT,
            id="audit-1",
            timestamp=datetime.utcnow(),
            data={
                "user": "test",
                "password": "secret123",
                "api_key": "key123",
                "email": "test@example.com",
            },
            source="test",
        )

        redacted = exporter._redact_item(item)

        assert redacted.data["password"] == "[REDACTED]"
        assert redacted.data["api_key"] == "[REDACTED]"
        assert redacted.data["email"] == "[REDACTED]"
        assert redacted.data["user"] == "test"

    def test_redact_nested_fields(self, temp_output):
        """Test redaction in nested structures."""
        config = EvidencePackConfig(
            output_path=temp_output,
            redact_pii=True,
        )
        exporter = EvidencePackExporter(config)

        data = {
            "outer": {
                "secret_key": "value",
                "normal": "ok",
            },
            "auth_info": {
                "api_token": "abc123",
                "user_id": "user-123",
            },
        }

        redacted = exporter._redact_dict(data)

        assert redacted["outer"]["secret_key"] == "[REDACTED]"
        assert redacted["outer"]["normal"] == "ok"
        # "api_token" contains "token" which is sensitive
        assert redacted["auth_info"]["api_token"] == "[REDACTED]"
        # "user_id" doesn't match sensitive patterns
        assert redacted["auth_info"]["user_id"] == "user-123"


class TestEvidenceTypes:
    """Tests for different evidence types."""

    @pytest.fixture
    def temp_output(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_all_evidence_types_have_collectors(self, temp_output):
        """Test that all evidence types have collectors."""
        config = EvidencePackConfig(output_path=temp_output)
        exporter = EvidencePackExporter(config)

        for evidence_type in EvidenceType:
            assert evidence_type in exporter._collectors

    @pytest.mark.asyncio
    async def test_sensitive_types_require_include_sensitive(self, temp_output):
        """Test that sensitive types respect include_sensitive flag."""
        config = EvidencePackConfig(
            output_path=temp_output,
            evidence_types=[EvidenceType.TELEMETRY_RAW],
            include_sensitive=False,
        )
        exporter = EvidencePackExporter(config)

        now = datetime.utcnow()
        items = await exporter._collect_telemetry_raw(
            now - timedelta(days=1), now
        )

        # Should return empty when include_sensitive is False
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_artifact_evidence_types(self, temp_output):
        """Test artifact-related evidence types."""
        config = EvidencePackConfig(
            output_path=temp_output,
            evidence_types=[
                EvidenceType.ARTIFACT_DIGESTS,
                EvidenceType.ARTIFACT_SIGNATURES,
                EvidenceType.ARTIFACT_SBOM,
            ],
        )
        exporter = EvidencePackExporter(config)

        pack = await exporter.export()

        # Should complete without errors
        assert pack.is_complete is True
        assert len(pack.errors) == 0


class TestExportDestinations:
    """Tests for export destinations."""

    def test_local_destination(self):
        """Test local destination config."""
        config = EvidencePackConfig(
            destination=ExportDestination.LOCAL,
        )

        assert config.destination == ExportDestination.LOCAL

    def test_s3_destination(self):
        """Test S3 destination config."""
        config = EvidencePackConfig(
            destination=ExportDestination.S3,
            destination_config={
                "bucket": "my-bucket",
                "prefix": "evidence/",
            },
        )

        assert config.destination == ExportDestination.S3
        assert config.destination_config["bucket"] == "my-bucket"

    def test_gcs_destination(self):
        """Test GCS destination config."""
        config = EvidencePackConfig(
            destination=ExportDestination.GCS,
            destination_config={
                "bucket": "my-gcs-bucket",
            },
        )

        assert config.destination == ExportDestination.GCS


class TestManifestCreation:
    """Tests for manifest creation."""

    @pytest.fixture
    def temp_output(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_manifest(self, temp_output):
        """Test manifest creation."""
        config = EvidencePackConfig(output_path=temp_output)
        exporter = EvidencePackExporter(config)

        pack = EvidencePack(
            evidence_types=[EvidenceType.APPROVAL_RECORDS],
            item_count=5,
        )

        items = [
            EvidenceItem(
                type=EvidenceType.APPROVAL_RECORDS,
                id=f"item-{i}",
                timestamp=datetime.utcnow(),
                data={},
                source="test",
            )
            for i in range(5)
        ]

        manifest = exporter._create_manifest(pack, items)

        assert manifest["version"] == "1.0.0"
        assert manifest["item_count"] == 5
        assert "summary" in manifest
        assert manifest["summary"]["total_items"] == 5

    def test_create_summary(self, temp_output):
        """Test summary creation."""
        config = EvidencePackConfig(output_path=temp_output)
        exporter = EvidencePackExporter(config)

        items = [
            EvidenceItem(
                type=EvidenceType.APPROVAL_RECORDS,
                id="1",
                timestamp=datetime.utcnow(),
                data={},
                source="test",
            ),
            EvidenceItem(
                type=EvidenceType.APPROVAL_RECORDS,
                id="2",
                timestamp=datetime.utcnow(),
                data={},
                source="test",
            ),
            EvidenceItem(
                type=EvidenceType.DEPLOYMENT_LOGS,
                id="3",
                timestamp=datetime.utcnow(),
                data={},
                source="test",
            ),
        ]

        summary = exporter._create_summary(items)

        assert summary["total_items"] == 3
        assert summary["by_type"]["approval_records"] == 2
        assert summary["by_type"]["deployment_logs"] == 1
