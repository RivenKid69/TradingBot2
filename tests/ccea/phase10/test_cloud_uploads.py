# -*- coding: utf-8 -*-
"""
Tests for Evidence Pack cloud uploads (S3/GCS/Azure).

Phase 10: Enterprise cloud storage.
"""

import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock
from uuid import uuid4
import pytest

from packages.cloud.enterprise.evidence_pack import (
    EvidencePackExporter,
    EvidencePackConfig,
    EvidencePack,
    ExportDestination,
    ExportFormat,
    BOTO3_AVAILABLE,
    GCS_AVAILABLE,
    AZURE_AVAILABLE,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pack(temp_dir):
    """Create a sample evidence pack for testing."""
    archive_path = temp_dir / "evidence-pack.zip"
    archive_path.write_bytes(b"test archive content")

    return EvidencePack(
        id=uuid4(),
        version="1.0.0",
        created_at=datetime.now(timezone.utc),
        checksum="sha256:abc123",
        archive_path=archive_path,
    )


class TestS3Upload:
    """Tests for S3 upload functionality."""

    @pytest.mark.asyncio
    async def test_upload_to_s3_requires_boto3(self, temp_dir, sample_pack):
        """Test S3 upload requires boto3."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.S3,
            destination_config={"bucket": "test-bucket"},
        )
        exporter = EvidencePackExporter(config)

        with patch("packages.cloud.enterprise.evidence_pack.BOTO3_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="boto3"):
                await exporter._upload_to_s3(sample_pack)

    @pytest.mark.asyncio
    async def test_upload_to_s3_requires_bucket(self, temp_dir, sample_pack):
        """Test S3 upload requires bucket configuration."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.S3,
            destination_config={},  # No bucket
        )
        exporter = EvidencePackExporter(config)

        with patch("packages.cloud.enterprise.evidence_pack.BOTO3_AVAILABLE", True):
            with pytest.raises(ValueError, match="bucket"):
                await exporter._upload_to_s3(sample_pack)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    async def test_upload_to_s3_success(self, temp_dir, sample_pack):
        """Test successful S3 upload with mocked boto3."""
        import boto3

        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.S3,
            destination_config={
                "bucket": "test-bucket",
                "prefix": "evidence-packs",
                "region": "us-east-1",
            },
        )
        exporter = EvidencePackExporter(config)

        mock_s3_client = MagicMock()

        with patch.object(boto3, "client", return_value=mock_s3_client):
            await exporter._upload_to_s3(sample_pack)

            # Verify upload was called
            mock_s3_client.upload_file.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    async def test_upload_to_s3_with_encryption(self, temp_dir, sample_pack):
        """Test S3 upload includes server-side encryption."""
        import boto3

        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.S3,
            destination_config={"bucket": "secure-bucket"},
        )
        exporter = EvidencePackExporter(config)

        mock_s3_client = MagicMock()

        with patch.object(boto3, "client", return_value=mock_s3_client):
            await exporter._upload_to_s3(sample_pack)

            call_args = mock_s3_client.upload_file.call_args
            extra_args = call_args[1]["ExtraArgs"]

            assert extra_args["ServerSideEncryption"] == "AES256"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not available")
    async def test_upload_to_s3_with_metadata(self, temp_dir, sample_pack):
        """Test S3 upload includes metadata."""
        import boto3

        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.S3,
            destination_config={"bucket": "metadata-bucket"},
        )
        exporter = EvidencePackExporter(config)

        mock_s3_client = MagicMock()

        with patch.object(boto3, "client", return_value=mock_s3_client):
            await exporter._upload_to_s3(sample_pack)

            call_args = mock_s3_client.upload_file.call_args
            metadata = call_args[1]["ExtraArgs"]["Metadata"]

            assert "pack-id" in metadata
            assert "pack-version" in metadata
            assert "checksum" in metadata

    @pytest.mark.asyncio
    async def test_upload_to_s3_requires_archive(self, temp_dir):
        """Test S3 upload requires archive path."""
        pack_without_archive = EvidencePack(
            id=uuid4(),
            version="1.0.0",
            created_at=datetime.now(timezone.utc),
            checksum="sha256:noarchive",
            archive_path=None,
        )

        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.S3,
            destination_config={"bucket": "test-bucket"},
        )
        exporter = EvidencePackExporter(config)

        with patch("packages.cloud.enterprise.evidence_pack.BOTO3_AVAILABLE", True):
            with pytest.raises(ValueError, match="archive"):
                await exporter._upload_to_s3(pack_without_archive)


class TestGCSUpload:
    """Tests for Google Cloud Storage upload functionality."""

    @pytest.mark.asyncio
    async def test_upload_to_gcs_requires_library(self, temp_dir, sample_pack):
        """Test GCS upload requires google-cloud-storage."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.GCS,
            destination_config={"bucket": "test-bucket"},
        )
        exporter = EvidencePackExporter(config)

        with patch("packages.cloud.enterprise.evidence_pack.GCS_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="google-cloud-storage"):
                await exporter._upload_to_gcs(sample_pack)

    @pytest.mark.asyncio
    async def test_upload_to_gcs_requires_bucket(self, temp_dir, sample_pack):
        """Test GCS upload requires bucket configuration."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.GCS,
            destination_config={},
        )
        exporter = EvidencePackExporter(config)

        with patch("packages.cloud.enterprise.evidence_pack.GCS_AVAILABLE", True):
            with pytest.raises(ValueError, match="bucket"):
                await exporter._upload_to_gcs(sample_pack)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not GCS_AVAILABLE, reason="google-cloud-storage not available")
    async def test_upload_to_gcs_success(self, temp_dir, sample_pack):
        """Test successful GCS upload."""
        from google.cloud import storage as gcs_storage

        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.GCS,
            destination_config={
                "bucket": "gcs-bucket",
                "prefix": "packs",
            },
        )
        exporter = EvidencePackExporter(config)

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        with patch.object(gcs_storage, "Client", return_value=mock_client):
            await exporter._upload_to_gcs(sample_pack)

            mock_client.bucket.assert_called_once_with("gcs-bucket")
            mock_blob.upload_from_filename.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not GCS_AVAILABLE, reason="google-cloud-storage not available")
    async def test_upload_to_gcs_with_metadata(self, temp_dir, sample_pack):
        """Test GCS upload includes metadata."""
        from google.cloud import storage as gcs_storage

        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.GCS,
            destination_config={"bucket": "gcs-bucket"},
        )
        exporter = EvidencePackExporter(config)

        mock_client = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        mock_client.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        with patch.object(gcs_storage, "Client", return_value=mock_client):
            await exporter._upload_to_gcs(sample_pack)

            # Check metadata was set on blob
            assert mock_blob.metadata is not None
            assert "pack-id" in mock_blob.metadata


class TestAzureUpload:
    """Tests for Azure Blob Storage upload functionality."""

    @pytest.mark.asyncio
    async def test_upload_to_azure_requires_library(self, temp_dir, sample_pack):
        """Test Azure upload requires azure-storage-blob."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.AZURE,
            destination_config={"container": "test-container"},
        )
        exporter = EvidencePackExporter(config)

        with patch("packages.cloud.enterprise.evidence_pack.AZURE_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="azure-storage-blob"):
                await exporter._upload_to_azure(sample_pack)

    @pytest.mark.asyncio
    async def test_upload_to_azure_requires_container(self, temp_dir, sample_pack):
        """Test Azure upload requires container configuration."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.AZURE,
            destination_config={},
        )
        exporter = EvidencePackExporter(config)

        with patch("packages.cloud.enterprise.evidence_pack.AZURE_AVAILABLE", True):
            with pytest.raises(ValueError, match="container"):
                await exporter._upload_to_azure(sample_pack)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not AZURE_AVAILABLE, reason="azure-storage-blob not available")
    async def test_upload_to_azure_requires_connection(self, temp_dir, sample_pack):
        """Test Azure upload requires connection string or account URL."""
        from azure.storage.blob import BlobServiceClient

        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.AZURE,
            destination_config={"container": "test-container"},
        )
        exporter = EvidencePackExporter(config)

        with patch.object(
            BlobServiceClient, "from_connection_string", side_effect=ValueError("connection_string")
        ):
            with pytest.raises(ValueError, match="connection_string"):
                await exporter._upload_to_azure(sample_pack)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not AZURE_AVAILABLE, reason="azure-storage-blob not available")
    async def test_upload_to_azure_with_connection_string(self, temp_dir, sample_pack):
        """Test Azure upload with connection string."""
        from azure.storage.blob import BlobServiceClient

        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.AZURE,
            destination_config={
                "container": "azure-container",
                "connection_string": "DefaultEndpointsProtocol=https;...",
                "prefix": "packs",
            },
        )
        exporter = EvidencePackExporter(config)

        mock_blob_service = MagicMock()
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()

        mock_blob_service.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client

        with patch.object(
            BlobServiceClient, "from_connection_string", return_value=mock_blob_service
        ):
            await exporter._upload_to_azure(sample_pack)

            BlobServiceClient.from_connection_string.assert_called_once()
            mock_blob_client.upload_blob.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not AZURE_AVAILABLE, reason="azure-storage-blob not available")
    async def test_upload_to_azure_with_account_url(self, temp_dir, sample_pack):
        """Test Azure upload with account URL."""
        from azure.storage.blob import BlobServiceClient

        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.AZURE,
            destination_config={
                "container": "azure-container",
                "account_url": "https://myaccount.blob.core.windows.net",
            },
        )
        exporter = EvidencePackExporter(config)

        mock_blob_service = MagicMock()
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()

        mock_blob_service.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client

        with patch.object(BlobServiceClient, "__init__", return_value=None):
            with patch.object(
                BlobServiceClient, "get_container_client", return_value=mock_container_client
            ):
                mock_container_client.get_blob_client.return_value = mock_blob_client

                await exporter._upload_to_azure(sample_pack)

                mock_blob_client.upload_blob.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.skipif(not AZURE_AVAILABLE, reason="azure-storage-blob not available")
    async def test_upload_to_azure_with_metadata(self, temp_dir, sample_pack):
        """Test Azure upload includes metadata."""
        from azure.storage.blob import BlobServiceClient

        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.AZURE,
            destination_config={
                "container": "azure-container",
                "connection_string": "connection-string",
            },
        )
        exporter = EvidencePackExporter(config)

        mock_blob_service = MagicMock()
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()

        mock_blob_service.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client

        with patch.object(
            BlobServiceClient, "from_connection_string", return_value=mock_blob_service
        ):
            await exporter._upload_to_azure(sample_pack)

            call_args = mock_blob_client.upload_blob.call_args
            metadata = call_args[1]["metadata"]

            assert "pack_id" in metadata
            assert "pack_version" in metadata
            assert "checksum" in metadata


class TestDestinationRouting:
    """Tests for destination routing."""

    @pytest.mark.asyncio
    async def test_route_to_s3(self, temp_dir, sample_pack):
        """Test routing to S3 destination."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.S3,
            destination_config={"bucket": "test"},
        )
        exporter = EvidencePackExporter(config)

        with patch.object(exporter, "_upload_to_s3", new_callable=AsyncMock) as mock_s3:
            await exporter._upload_to_destination(sample_pack)
            mock_s3.assert_called_once_with(sample_pack)

    @pytest.mark.asyncio
    async def test_route_to_gcs(self, temp_dir, sample_pack):
        """Test routing to GCS destination."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.GCS,
            destination_config={"bucket": "test"},
        )
        exporter = EvidencePackExporter(config)

        with patch.object(exporter, "_upload_to_gcs", new_callable=AsyncMock) as mock_gcs:
            await exporter._upload_to_destination(sample_pack)
            mock_gcs.assert_called_once_with(sample_pack)

    @pytest.mark.asyncio
    async def test_route_to_azure(self, temp_dir, sample_pack):
        """Test routing to Azure destination."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.AZURE,
            destination_config={"container": "test"},
        )
        exporter = EvidencePackExporter(config)

        with patch.object(exporter, "_upload_to_azure", new_callable=AsyncMock) as mock_azure:
            await exporter._upload_to_destination(sample_pack)
            mock_azure.assert_called_once_with(sample_pack)

    @pytest.mark.asyncio
    async def test_local_destination_no_upload(self, temp_dir, sample_pack):
        """Test local destination doesn't upload."""
        config = EvidencePackConfig(
            output_path=temp_dir,
            destination=ExportDestination.LOCAL,
        )
        exporter = EvidencePackExporter(config)

        # Should not raise any errors
        await exporter._upload_to_destination(sample_pack)


class TestCloudLibraryAvailability:
    """Tests for cloud library availability flags."""

    def test_boto3_availability_flag(self):
        """Test BOTO3_AVAILABLE flag exists."""
        from packages.cloud.enterprise.evidence_pack import BOTO3_AVAILABLE

        assert isinstance(BOTO3_AVAILABLE, bool)

    def test_gcs_availability_flag(self):
        """Test GCS_AVAILABLE flag exists."""
        from packages.cloud.enterprise.evidence_pack import GCS_AVAILABLE

        assert isinstance(GCS_AVAILABLE, bool)

    def test_azure_availability_flag(self):
        """Test AZURE_AVAILABLE flag exists."""
        from packages.cloud.enterprise.evidence_pack import AZURE_AVAILABLE

        assert isinstance(AZURE_AVAILABLE, bool)
