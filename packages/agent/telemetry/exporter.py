# -*- coding: utf-8 -*-
"""
Telemetry Exporter with Privacy Controls.

CCEA Phase 8 Implementation.

This module provides controlled export of telemetry data with:
    - Multiple export formats (JSON, CSV, JSONL, encrypted)
    - Privacy-preserving transformations
    - Redaction enforcement on export
    - Audit logging of exports

Design Doc Reference:
    - Phase 8 (13.3): "Cloud governance: retention, export/delete (DSAR)"
    - Privacy-controlled export with redaction

SECURITY:
    - All exports pass through redaction middleware
    - Export operations are logged for audit
    - Encrypted export option for sensitive data
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, Final, Iterator, List, Optional, TextIO, Union
from uuid import uuid4

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .redaction import TelemetryRedactionMiddleware, RedactionResult, ensure_redaction_applied


# ============================================================================
# Enums and Constants
# ============================================================================


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = "json"  # Pretty-printed JSON
    JSON_COMPACT = "json_compact"  # Minified JSON
    JSONL = "jsonl"  # JSON Lines (one record per line)
    CSV = "csv"  # Comma-separated values
    ENCRYPTED = "encrypted"  # Encrypted JSON


class ExportScope(Enum):
    """Scope of data to export."""

    ALL = auto()  # All telemetry
    TIME_RANGE = auto()  # Specific time range
    RUN = auto()  # Specific run
    AGENT = auto()  # Specific agent


# Maximum export size (100MB uncompressed)
MAX_EXPORT_SIZE_BYTES: Final[int] = 100 * 1024 * 1024

# Default encryption key derivation iterations
KDF_ITERATIONS: Final[int] = 480000


@dataclass
class ExportConfig:
    """
    Configuration for telemetry export.

    Attributes:
        format: Output format
        include_metadata: Include export metadata
        compress: Gzip compress output
        encrypt: Encrypt output (requires password)
        max_records: Maximum records to export
        time_range: Optional time range filter
        run_ids: Optional run ID filter
        agent_ids: Optional agent ID filter
    """

    format: ExportFormat = ExportFormat.JSONL
    include_metadata: bool = True
    compress: bool = False
    encrypt: bool = False
    encryption_password: Optional[str] = None
    max_records: int = 100000
    time_range: Optional[tuple[datetime, datetime]] = None
    run_ids: Optional[List[str]] = None
    agent_ids: Optional[List[str]] = None

    # Privacy options
    anonymize_ids: bool = False
    remove_timestamps: bool = False
    aggregate_only: bool = False


@dataclass
class ExportResult:
    """
    Result of an export operation.

    Attributes:
        success: Whether export succeeded
        export_id: Unique export identifier
        format: Export format used
        record_count: Number of records exported
        file_path: Path to exported file (if file export)
        content: Export content (if memory export)
        size_bytes: Size of exported data
        checksum: SHA-256 checksum of content
        encrypted: Whether output is encrypted
        compressed: Whether output is compressed
    """

    success: bool = True
    export_id: str = field(default_factory=lambda: str(uuid4()))
    format: ExportFormat = ExportFormat.JSONL
    record_count: int = 0
    file_path: Optional[Path] = None
    content: Optional[bytes] = None
    size_bytes: int = 0
    checksum: str = ""
    encrypted: bool = False
    compressed: bool = False
    error: Optional[str] = None
    exported_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "export_id": self.export_id,
            "success": self.success,
            "format": self.format.value,
            "record_count": self.record_count,
            "file_path": str(self.file_path) if self.file_path else None,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "encrypted": self.encrypted,
            "compressed": self.compressed,
            "error": self.error,
            "exported_at": self.exported_at.isoformat(),
        }


@dataclass
class ExportMetadata:
    """Metadata included in export."""

    export_id: str
    exported_at: datetime
    format: str
    record_count: int
    time_range: Optional[tuple[str, str]]
    redaction_applied: bool = True
    redaction_version: str = "1.0.0"
    agent_id: Optional[str] = None
    run_ids: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "export_id": self.export_id,
            "exported_at": self.exported_at.isoformat(),
            "format": self.format,
            "record_count": self.record_count,
            "time_range": self.time_range,
            "redaction_applied": self.redaction_applied,
            "redaction_version": self.redaction_version,
            "agent_id": self.agent_id,
            "run_ids": self.run_ids,
        }


class TelemetryExporter:
    """
    Export telemetry data with privacy controls.

    This exporter ensures all exported data passes through
    redaction and provides multiple output formats.

    Usage:
        exporter = TelemetryExporter()

        # Export to file
        result = exporter.export_to_file(
            data=telemetry_events,
            path=Path("/tmp/export.jsonl"),
            config=ExportConfig(format=ExportFormat.JSONL),
        )

        # Export to memory
        result = exporter.export_to_memory(
            data=telemetry_events,
            config=ExportConfig(format=ExportFormat.JSON),
        )

        # Export encrypted
        result = exporter.export_to_file(
            data=telemetry_events,
            path=Path("/tmp/export.enc"),
            config=ExportConfig(
                format=ExportFormat.ENCRYPTED,
                encrypt=True,
                encryption_password="secure-password",
            ),
        )

    SECURITY:
        - All exports are redacted
        - Export operations are logged
        - Encryption available for sensitive exports
    """

    def __init__(self):
        """Initialize exporter."""
        self._redaction = TelemetryRedactionMiddleware()
        self._export_log: List[ExportResult] = []
        self._lock = threading.Lock()
        self._id_mapping: Dict[str, str] = {}  # For anonymization

    def export_to_file(
        self,
        data: List[Dict[str, Any]],
        path: Path,
        config: Optional[ExportConfig] = None,
    ) -> ExportResult:
        """
        Export telemetry data to file.

        Args:
            data: Telemetry records to export
            path: Output file path
            config: Export configuration

        Returns:
            ExportResult
        """
        config = config or ExportConfig()
        export_id = str(uuid4())

        try:
            # Apply redaction to all records
            redacted_data = self._redact_data(data)

            # Apply privacy transformations
            processed_data = self._apply_privacy_transforms(redacted_data, config)

            # Apply filters
            filtered_data = self._apply_filters(processed_data, config)

            # Limit records
            limited_data = filtered_data[: config.max_records]

            # Generate content
            content = self._format_data(limited_data, config, export_id)

            # Compress if requested
            if config.compress:
                content = gzip.compress(content)

            # Encrypt if requested
            if config.encrypt and config.encryption_password:
                content = self._encrypt_content(content, config.encryption_password)

            # Write to file
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(content)

            # Calculate checksum
            checksum = hashlib.sha256(content).hexdigest()

            result = ExportResult(
                success=True,
                export_id=export_id,
                format=config.format,
                record_count=len(limited_data),
                file_path=path,
                size_bytes=len(content),
                checksum=checksum,
                encrypted=config.encrypt,
                compressed=config.compress,
            )

        except Exception as e:
            result = ExportResult(
                success=False,
                export_id=export_id,
                format=config.format,
                error=str(e),
            )

        self._log_export(result)
        return result

    def export_to_memory(
        self,
        data: List[Dict[str, Any]],
        config: Optional[ExportConfig] = None,
    ) -> ExportResult:
        """
        Export telemetry data to memory.

        Args:
            data: Telemetry records to export
            config: Export configuration

        Returns:
            ExportResult with content
        """
        config = config or ExportConfig()
        export_id = str(uuid4())

        try:
            # Apply redaction
            redacted_data = self._redact_data(data)

            # Apply privacy transformations
            processed_data = self._apply_privacy_transforms(redacted_data, config)

            # Apply filters
            filtered_data = self._apply_filters(processed_data, config)

            # Limit records
            limited_data = filtered_data[: config.max_records]

            # Check size limit
            if len(limited_data) > 0:
                sample_size = len(json.dumps(limited_data[0]).encode())
                estimated_size = sample_size * len(limited_data)
                if estimated_size > MAX_EXPORT_SIZE_BYTES:
                    raise ValueError(f"Export would exceed {MAX_EXPORT_SIZE_BYTES} bytes")

            # Generate content
            content = self._format_data(limited_data, config, export_id)

            # Compress if requested
            if config.compress:
                content = gzip.compress(content)

            # Encrypt if requested
            if config.encrypt and config.encryption_password:
                content = self._encrypt_content(content, config.encryption_password)

            # Calculate checksum
            checksum = hashlib.sha256(content).hexdigest()

            result = ExportResult(
                success=True,
                export_id=export_id,
                format=config.format,
                record_count=len(limited_data),
                content=content,
                size_bytes=len(content),
                checksum=checksum,
                encrypted=config.encrypt,
                compressed=config.compress,
            )

        except Exception as e:
            result = ExportResult(
                success=False,
                export_id=export_id,
                format=config.format,
                error=str(e),
            )

        self._log_export(result)
        return result

    def export_stream(
        self,
        data_iterator: Iterator[Dict[str, Any]],
        output: BinaryIO,
        config: Optional[ExportConfig] = None,
    ) -> ExportResult:
        """
        Export telemetry data as stream.

        Useful for large exports that don't fit in memory.

        Args:
            data_iterator: Iterator of telemetry records
            output: Output stream
            config: Export configuration

        Returns:
            ExportResult
        """
        config = config or ExportConfig()
        export_id = str(uuid4())
        record_count = 0
        total_size = 0

        try:
            # Write metadata header for JSONL
            if config.include_metadata and config.format == ExportFormat.JSONL:
                metadata = ExportMetadata(
                    export_id=export_id,
                    exported_at=datetime.utcnow(),
                    format=config.format.value,
                    record_count=0,  # Unknown at start
                    time_range=None,
                    redaction_applied=True,
                )
                header = json.dumps({"_metadata": metadata.to_dict()}).encode() + b"\n"
                output.write(header)
                total_size += len(header)

            # Process records one by one
            for record in data_iterator:
                if record_count >= config.max_records:
                    break

                # Redact
                redacted = self._redaction.redact(record)

                # Apply privacy transforms
                processed = self._apply_privacy_transforms([redacted.data], config)[0]

                # Format and write
                if config.format == ExportFormat.JSONL:
                    line = json.dumps(processed).encode() + b"\n"
                    output.write(line)
                    total_size += len(line)

                record_count += 1

            result = ExportResult(
                success=True,
                export_id=export_id,
                format=config.format,
                record_count=record_count,
                size_bytes=total_size,
            )

        except Exception as e:
            result = ExportResult(
                success=False,
                export_id=export_id,
                format=config.format,
                record_count=record_count,
                error=str(e),
            )

        self._log_export(result)
        return result

    def _redact_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply redaction to all records."""
        return [self._redaction.redact(record).data for record in data]

    def _apply_privacy_transforms(
        self,
        data: List[Dict[str, Any]],
        config: ExportConfig,
    ) -> List[Dict[str, Any]]:
        """Apply privacy transformations."""
        result = []

        for record in data:
            processed = dict(record)

            # Anonymize IDs
            if config.anonymize_ids:
                processed = self._anonymize_record(processed)

            # Remove timestamps
            if config.remove_timestamps:
                processed = self._remove_timestamps(processed)

            result.append(processed)

        return result

    def _anonymize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize identifiers in a record."""
        result = {}

        for key, value in record.items():
            if isinstance(value, dict):
                result[key] = self._anonymize_record(value)
            elif isinstance(value, list):
                result[key] = [
                    self._anonymize_record(v) if isinstance(v, dict) else v for v in value
                ]
            elif key.endswith("_id") or key == "id":
                # Anonymize IDs consistently
                result[key] = self._get_anonymized_id(str(value))
            else:
                result[key] = value

        return result

    def _get_anonymized_id(self, original_id: str) -> str:
        """Get consistent anonymized ID."""
        if original_id not in self._id_mapping:
            # Create deterministic but non-reversible mapping
            hashed = hashlib.sha256(original_id.encode()).hexdigest()[:12]
            self._id_mapping[original_id] = f"anon_{hashed}"
        return self._id_mapping[original_id]

    def _remove_timestamps(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Remove timestamp fields from record."""
        timestamp_keys = {"timestamp", "created_at", "updated_at", "sent_at", "received_at"}
        result = {}

        for key, value in record.items():
            if key.lower() in timestamp_keys:
                continue
            if isinstance(value, dict):
                result[key] = self._remove_timestamps(value)
            else:
                result[key] = value

        return result

    def _apply_filters(
        self,
        data: List[Dict[str, Any]],
        config: ExportConfig,
    ) -> List[Dict[str, Any]]:
        """Apply filters based on config."""
        result = data

        # Filter by time range
        if config.time_range:
            start, end = config.time_range
            result = [r for r in result if self._record_in_time_range(r, start, end)]

        # Filter by run IDs
        if config.run_ids:
            result = [r for r in result if r.get("run_id") in config.run_ids]

        # Filter by agent IDs
        if config.agent_ids:
            result = [r for r in result if r.get("agent_id") in config.agent_ids]

        return result

    def _record_in_time_range(
        self,
        record: Dict[str, Any],
        start: datetime,
        end: datetime,
    ) -> bool:
        """Check if record timestamp is in range."""
        timestamp_str = record.get("timestamp")
        if not timestamp_str:
            return True  # Include records without timestamp

        try:
            if isinstance(timestamp_str, datetime):
                ts = timestamp_str
            else:
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return start <= ts <= end
        except (ValueError, TypeError):
            return True

    def _format_data(
        self,
        data: List[Dict[str, Any]],
        config: ExportConfig,
        export_id: str,
    ) -> bytes:
        """Format data according to export format."""
        if config.format == ExportFormat.JSON:
            output = {
                "records": data,
            }
            if config.include_metadata:
                output["_metadata"] = self._create_metadata(export_id, config, len(data)).to_dict()
            return json.dumps(output, indent=2, default=str).encode()

        elif config.format == ExportFormat.JSON_COMPACT:
            output = {
                "records": data,
            }
            if config.include_metadata:
                output["_metadata"] = self._create_metadata(export_id, config, len(data)).to_dict()
            return json.dumps(output, separators=(",", ":"), default=str).encode()

        elif config.format == ExportFormat.JSONL:
            lines = []
            if config.include_metadata:
                metadata = self._create_metadata(export_id, config, len(data))
                lines.append(json.dumps({"_metadata": metadata.to_dict()}, default=str))
            for record in data:
                lines.append(json.dumps(record, default=str))
            return "\n".join(lines).encode()

        elif config.format == ExportFormat.CSV:
            return self._format_csv(data, config, export_id)

        elif config.format == ExportFormat.ENCRYPTED:
            # Use JSON format for encrypted
            output = {
                "records": data,
            }
            if config.include_metadata:
                output["_metadata"] = self._create_metadata(export_id, config, len(data)).to_dict()
            return json.dumps(output, default=str).encode()

        else:
            raise ValueError(f"Unsupported format: {config.format}")

    def _format_csv(
        self,
        data: List[Dict[str, Any]],
        config: ExportConfig,
        export_id: str,
    ) -> bytes:
        """Format data as CSV."""
        if not data:
            return b""

        output = io.StringIO()
        writer = None

        for record in data:
            # Flatten nested structures
            flat = self._flatten_dict(record)

            if writer is None:
                writer = csv.DictWriter(output, fieldnames=sorted(flat.keys()))
                writer.writeheader()

            writer.writerow(flat)

        return output.getvalue().encode()

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "_",
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
        return dict(items)

    def _create_metadata(
        self,
        export_id: str,
        config: ExportConfig,
        record_count: int,
    ) -> ExportMetadata:
        """Create export metadata."""
        time_range = None
        if config.time_range:
            time_range = (
                config.time_range[0].isoformat(),
                config.time_range[1].isoformat(),
            )

        return ExportMetadata(
            export_id=export_id,
            exported_at=datetime.utcnow(),
            format=config.format.value,
            record_count=record_count,
            time_range=time_range,
            redaction_applied=True,
            redaction_version=self._redaction.redaction_version,
            run_ids=config.run_ids,
        )

    def _encrypt_content(self, content: bytes, password: str) -> bytes:
        """Encrypt content using Fernet (AES-128-CBC)."""
        # Derive key from password
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=KDF_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

        # Encrypt
        fernet = Fernet(key)
        encrypted = fernet.encrypt(content)

        # Prepend salt for decryption
        return salt + encrypted

    def decrypt_content(self, encrypted: bytes, password: str) -> bytes:
        """
        Decrypt exported content.

        Args:
            encrypted: Encrypted content (with salt prefix)
            password: Encryption password

        Returns:
            Decrypted content
        """
        # Extract salt
        salt = encrypted[:16]
        ciphertext = encrypted[16:]

        # Derive key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=KDF_ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))

        # Decrypt
        fernet = Fernet(key)
        return fernet.decrypt(ciphertext)

    def _log_export(self, result: ExportResult) -> None:
        """Log export operation for audit."""
        with self._lock:
            self._export_log.append(result)

    def get_export_log(self, limit: int = 100) -> List[ExportResult]:
        """Get export audit log."""
        with self._lock:
            return self._export_log[-limit:]

    def clear_anonymization_mapping(self) -> None:
        """Clear ID anonymization mapping."""
        self._id_mapping.clear()


# ============================================================================
# Convenience Functions
# ============================================================================


def export_telemetry_json(
    data: List[Dict[str, Any]],
    path: Path,
    compress: bool = False,
) -> ExportResult:
    """Export telemetry to JSON file."""
    exporter = TelemetryExporter()
    return exporter.export_to_file(
        data=data,
        path=path,
        config=ExportConfig(
            format=ExportFormat.JSON,
            compress=compress,
        ),
    )


def export_telemetry_encrypted(
    data: List[Dict[str, Any]],
    path: Path,
    password: str,
) -> ExportResult:
    """Export telemetry encrypted."""
    exporter = TelemetryExporter()
    return exporter.export_to_file(
        data=data,
        path=path,
        config=ExportConfig(
            format=ExportFormat.ENCRYPTED,
            encrypt=True,
            encryption_password=password,
        ),
    )
