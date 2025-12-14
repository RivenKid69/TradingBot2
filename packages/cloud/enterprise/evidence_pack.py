# -*- coding: utf-8 -*-
"""
Evidence Pack Exporter - Audit trail and compliance documentation.

CLOUD ZONE ONLY.

Exports comprehensive evidence packs for enterprise compliance:
- Artifact digests and signatures
- SBOM (Software Bill of Materials)
- Deploy/upgrade/approval journals
- Command logs with idempotency tracking
- Halt reasons and incident logs
- Telemetry exports (by sensitivity level)

Design Doc Reference:
    - Phase 9: Enterprise/on-prem pack
    - Design Doc 16.1: Evidence pack exporter
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Set, Tuple, Union
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# Constants
EVIDENCE_PACK_VERSION: Final[str] = "1.0.0"
MAX_EXPORT_SIZE_BYTES: Final[int] = 10 * 1024 * 1024 * 1024  # 10GB


class EvidenceType(Enum):
    """Types of evidence that can be exported."""
    # Artifacts
    ARTIFACT_DIGESTS = "artifact_digests"
    ARTIFACT_SIGNATURES = "artifact_signatures"
    ARTIFACT_SBOM = "artifact_sbom"
    ARTIFACT_PROVENANCE = "artifact_provenance"

    # Deployment
    DEPLOYMENT_LOGS = "deployment_logs"
    UPGRADE_LOGS = "upgrade_logs"
    APPROVAL_RECORDS = "approval_records"
    COMMAND_LOGS = "command_logs"

    # Incidents
    HALT_REASONS = "halt_reasons"
    INCIDENT_LOGS = "incident_logs"
    KILL_SWITCH_EVENTS = "kill_switch_events"

    # Telemetry
    TELEMETRY_AGGREGATED = "telemetry_aggregated"
    TELEMETRY_DETAILED = "telemetry_detailed"
    TELEMETRY_RAW = "telemetry_raw"

    # Audit
    ACCESS_AUDIT = "access_audit"
    BREAK_GLASS_EVENTS = "break_glass_events"
    CONFIG_CHANGES = "config_changes"

    # System
    AGENT_HEALTH = "agent_health"
    SYSTEM_METRICS = "system_metrics"


class ExportFormat(Enum):
    """Export format options."""
    JSON = "json"
    JSON_LINES = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


class ExportDestination(Enum):
    """Export destination types."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


@dataclass
class EvidencePackConfig:
    """Configuration for evidence pack export."""
    # Output settings
    output_path: Path = field(default_factory=lambda: Path.home() / ".ccea" / "evidence")
    output_format: ExportFormat = ExportFormat.JSON
    compress: bool = True
    encrypt: bool = False
    encryption_key_id: Optional[str] = None

    # Content settings
    evidence_types: List[EvidenceType] = field(default_factory=lambda: list(EvidenceType))
    include_sensitive: bool = False
    redact_pii: bool = True

    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    days_back: int = 30

    # Filtering
    workspace_ids: Optional[List[UUID]] = None
    agent_ids: Optional[List[UUID]] = None
    deployment_ids: Optional[List[UUID]] = None

    # Signing
    sign_pack: bool = True
    signing_key_path: Optional[Path] = None

    # Destination
    destination: ExportDestination = ExportDestination.LOCAL
    destination_config: Dict[str, Any] = field(default_factory=dict)

    # Retention
    retention_days: int = 365

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_path": str(self.output_path),
            "output_format": self.output_format.value,
            "compress": self.compress,
            "encrypt": self.encrypt,
            "evidence_types": [e.value for e in self.evidence_types],
            "include_sensitive": self.include_sensitive,
            "redact_pii": self.redact_pii,
            "days_back": self.days_back,
            "sign_pack": self.sign_pack,
            "destination": self.destination.value,
            "retention_days": self.retention_days,
        }


@dataclass
class EvidenceItem:
    """A single piece of evidence."""
    type: EvidenceType
    id: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    workspace_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None
    checksum: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "checksum": self.checksum,
        }


@dataclass
class EvidencePack:
    """A complete evidence pack."""
    id: UUID = field(default_factory=uuid4)
    version: str = EVIDENCE_PACK_VERSION
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"

    # Metadata
    workspace_id: Optional[UUID] = None
    organization_id: Optional[UUID] = None
    description: str = ""

    # Time range
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Content
    evidence_types: List[EvidenceType] = field(default_factory=list)
    item_count: int = 0
    total_size_bytes: int = 0

    # Files
    manifest_path: Optional[Path] = None
    archive_path: Optional[Path] = None

    # Integrity
    checksum: str = ""
    signature: Optional[str] = None
    signed_by: Optional[str] = None

    # Status
    is_complete: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "organization_id": str(self.organization_id) if self.organization_id else None,
            "description": self.description,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "evidence_types": [e.value for e in self.evidence_types],
            "item_count": self.item_count,
            "total_size_bytes": self.total_size_bytes,
            "checksum": self.checksum,
            "signature": self.signature,
            "signed_by": self.signed_by,
            "is_complete": self.is_complete,
            "errors": self.errors,
        }


class EvidencePackExporter:
    """
    Evidence Pack Exporter for enterprise compliance.

    Exports comprehensive audit trails and compliance documentation:
    - Artifact digests, signatures, and SBOM
    - Deployment and approval journals
    - Incident logs and halt reasons
    - Telemetry exports (by sensitivity)

    Usage:
        config = EvidencePackConfig(
            evidence_types=[
                EvidenceType.ARTIFACT_DIGESTS,
                EvidenceType.APPROVAL_RECORDS,
                EvidenceType.INCIDENT_LOGS,
            ],
            days_back=30,
        )
        exporter = EvidencePackExporter(config)

        # Export evidence pack
        pack = await exporter.export()

        # Get pack location
        print(f"Evidence pack: {pack.archive_path}")
    """

    def __init__(
        self,
        config: Optional[EvidencePackConfig] = None,
        db_session: Optional[Any] = None,
    ):
        """
        Initialize evidence pack exporter.

        Args:
            config: Export configuration
            db_session: Database session for queries
        """
        self.config = config or EvidencePackConfig()
        self._db = db_session

        # Collectors for each evidence type
        self._collectors: Dict[EvidenceType, Callable] = {
            EvidenceType.ARTIFACT_DIGESTS: self._collect_artifact_digests,
            EvidenceType.ARTIFACT_SIGNATURES: self._collect_artifact_signatures,
            EvidenceType.ARTIFACT_SBOM: self._collect_artifact_sbom,
            EvidenceType.ARTIFACT_PROVENANCE: self._collect_artifact_provenance,
            EvidenceType.DEPLOYMENT_LOGS: self._collect_deployment_logs,
            EvidenceType.UPGRADE_LOGS: self._collect_upgrade_logs,
            EvidenceType.APPROVAL_RECORDS: self._collect_approval_records,
            EvidenceType.COMMAND_LOGS: self._collect_command_logs,
            EvidenceType.HALT_REASONS: self._collect_halt_reasons,
            EvidenceType.INCIDENT_LOGS: self._collect_incident_logs,
            EvidenceType.KILL_SWITCH_EVENTS: self._collect_kill_switch_events,
            EvidenceType.TELEMETRY_AGGREGATED: self._collect_telemetry_aggregated,
            EvidenceType.TELEMETRY_DETAILED: self._collect_telemetry_detailed,
            EvidenceType.TELEMETRY_RAW: self._collect_telemetry_raw,
            EvidenceType.ACCESS_AUDIT: self._collect_access_audit,
            EvidenceType.BREAK_GLASS_EVENTS: self._collect_break_glass_events,
            EvidenceType.CONFIG_CHANGES: self._collect_config_changes,
            EvidenceType.AGENT_HEALTH: self._collect_agent_health,
            EvidenceType.SYSTEM_METRICS: self._collect_system_metrics,
        }

    async def export(
        self,
        description: str = "",
        created_by: str = "system",
    ) -> EvidencePack:
        """
        Export evidence pack.

        Args:
            description: Pack description
            created_by: User/system creating the pack

        Returns:
            Evidence pack with archive location
        """
        pack = EvidencePack(
            description=description,
            created_by=created_by,
            evidence_types=self.config.evidence_types,
            workspace_id=self.config.workspace_ids[0] if self.config.workspace_ids else None,
        )

        # Calculate time range
        if self.config.start_time:
            pack.start_time = self.config.start_time
        else:
            pack.start_time = datetime.utcnow() - timedelta(days=self.config.days_back)

        pack.end_time = self.config.end_time or datetime.utcnow()

        logger.info(
            f"Exporting evidence pack: {len(self.config.evidence_types)} types, "
            f"{pack.start_time} to {pack.end_time}"
        )

        # Create temp directory for export
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                # Collect all evidence
                all_items: List[EvidenceItem] = []

                for evidence_type in self.config.evidence_types:
                    collector = self._collectors.get(evidence_type)
                    if collector:
                        try:
                            items = await collector(pack.start_time, pack.end_time)
                            all_items.extend(items)
                            logger.debug(f"Collected {len(items)} items for {evidence_type.value}")
                        except Exception as e:
                            pack.errors.append(f"Failed to collect {evidence_type.value}: {e}")
                            logger.error(f"Failed to collect {evidence_type.value}: {e}")

                pack.item_count = len(all_items)

                # Write evidence files
                evidence_dir = temp_path / "evidence"
                evidence_dir.mkdir(parents=True, exist_ok=True)

                for evidence_type in self.config.evidence_types:
                    type_items = [i for i in all_items if i.type == evidence_type]
                    if type_items:
                        file_path = evidence_dir / f"{evidence_type.value}.json"
                        self._write_evidence_file(file_path, type_items)

                # Calculate checksum of evidence BEFORE adding manifest
                pack.checksum = self._calculate_directory_checksum(
                    temp_path, exclude=["manifest.json", "signature.sig"]
                )

                # Write manifest with checksum
                manifest = self._create_manifest(pack, all_items)
                manifest_path = temp_path / "manifest.json"
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)

                # Sign pack if configured
                if self.config.sign_pack:
                    signature = await self._sign_pack(pack.checksum)
                    if signature:
                        pack.signature = signature
                        pack.signed_by = "ccea-evidence-signer"

                        # Write signature file
                        sig_path = temp_path / "signature.sig"
                        with open(sig_path, "w") as f:
                            f.write(signature)

                # Create archive
                archive_name = f"evidence-pack-{pack.id}-{pack.created_at.strftime('%Y%m%d-%H%M%S')}"
                archive_path = self._create_archive(temp_path, archive_name)

                pack.archive_path = archive_path
                pack.total_size_bytes = archive_path.stat().st_size

                # Upload to destination if not local
                if self.config.destination != ExportDestination.LOCAL:
                    await self._upload_to_destination(pack)

                pack.is_complete = True
                logger.info(
                    f"Evidence pack exported: {pack.id}, {pack.item_count} items, "
                    f"{pack.total_size_bytes / 1024 / 1024:.1f}MB"
                )

            except Exception as e:
                pack.errors.append(str(e))
                logger.error(f"Evidence pack export failed: {e}")

        return pack

    async def export_incremental(
        self,
        since: datetime,
        description: str = "",
    ) -> EvidencePack:
        """
        Export incremental evidence pack since last export.

        Args:
            since: Export evidence since this timestamp
            description: Pack description

        Returns:
            Evidence pack
        """
        self.config.start_time = since
        self.config.end_time = datetime.utcnow()
        return await self.export(description=description)

    async def verify_pack(self, pack_path: Path) -> Tuple[bool, List[str]]:
        """
        Verify evidence pack integrity.

        Args:
            pack_path: Path to evidence pack archive

        Returns:
            (is_valid, errors)
        """
        errors = []

        try:
            # Extract archive
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Extract
                if str(pack_path).endswith(".zip.gz"):
                    # First decompress the gzip, then extract the zip
                    with gzip.open(pack_path, "rb") as f_in:
                        zip_content = f_in.read()
                    import io
                    with zipfile.ZipFile(io.BytesIO(zip_content), "r") as zf:
                        zf.extractall(temp_path)
                else:
                    with zipfile.ZipFile(pack_path, "r") as zf:
                        zf.extractall(temp_path)

                # Read manifest
                manifest_path = temp_path / "manifest.json"
                if not manifest_path.exists():
                    errors.append("Missing manifest.json")
                    return False, errors

                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                # Verify checksum (excluding manifest and signature)
                computed_checksum = self._calculate_directory_checksum(
                    temp_path, exclude=["manifest.json", "signature.sig"]
                )
                if manifest.get("checksum") != computed_checksum:
                    errors.append("Checksum mismatch")

                # Verify signature if present
                sig_path = temp_path / "signature.sig"
                if sig_path.exists():
                    with open(sig_path, "r") as f:
                        signature = f.read()

                    sig_valid = await self._verify_signature(
                        manifest.get("checksum", ""),
                        signature,
                    )
                    if not sig_valid:
                        errors.append("Invalid signature")

                # Verify individual evidence files
                evidence_dir = temp_path / "evidence"
                if evidence_dir.exists():
                    for file_path in evidence_dir.glob("*.json"):
                        file_valid = self._verify_evidence_file(file_path)
                        if not file_valid:
                            errors.append(f"Invalid evidence file: {file_path.name}")

        except Exception as e:
            errors.append(f"Verification error: {e}")

        return len(errors) == 0, errors

    def list_packs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        List exported evidence packs.

        Args:
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of pack metadata
        """
        packs = []
        output_path = self.config.output_path

        if not output_path.exists():
            return packs

        for archive_path in output_path.glob("evidence-pack-*.zip*"):
            try:
                # Parse pack info from filename
                name_parts = archive_path.stem.replace(".zip", "").split("-")
                pack_id = name_parts[2] if len(name_parts) > 2 else ""

                stat = archive_path.stat()
                created_at = datetime.fromtimestamp(stat.st_mtime)

                if start_date and created_at < start_date:
                    continue
                if end_date and created_at > end_date:
                    continue

                packs.append({
                    "id": pack_id,
                    "path": str(archive_path),
                    "created_at": created_at.isoformat(),
                    "size_bytes": stat.st_size,
                })

            except Exception as e:
                logger.debug(f"Error parsing pack {archive_path}: {e}")

        return sorted(packs, key=lambda x: x["created_at"], reverse=True)

    # ===== Collectors =====

    async def _collect_artifact_digests(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect artifact digest evidence."""
        items = []
        # In real implementation: query artifacts table
        # SELECT id, digest, name, created_at FROM artifacts WHERE created_at BETWEEN ...
        return items

    async def _collect_artifact_signatures(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect artifact signature evidence."""
        items = []
        # Query signature records
        return items

    async def _collect_artifact_sbom(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect SBOM evidence."""
        items = []
        # Query SBOM records
        return items

    async def _collect_artifact_provenance(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect provenance evidence."""
        items = []
        # Query provenance records (SLSA)
        return items

    async def _collect_deployment_logs(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect deployment log evidence."""
        items = []
        # Query deployments table with state changes
        return items

    async def _collect_upgrade_logs(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect upgrade log evidence."""
        items = []
        # Query upgrade events
        return items

    async def _collect_approval_records(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect approval record evidence."""
        items = []
        # Query approval_records table
        return items

    async def _collect_command_logs(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect command log evidence."""
        items = []
        # Query commands table with full lifecycle
        return items

    async def _collect_halt_reasons(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect halt reason evidence."""
        items = []
        # Query halt events from telemetry/alerts
        return items

    async def _collect_incident_logs(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect incident log evidence."""
        items = []
        # Query alerts with severity >= ERROR
        return items

    async def _collect_kill_switch_events(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect kill switch event evidence."""
        items = []
        # Query kill switch events from telemetry
        return items

    async def _collect_telemetry_aggregated(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect aggregated telemetry evidence."""
        items = []
        # Query telemetry_events WHERE level = 'aggregated'
        return items

    async def _collect_telemetry_detailed(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect detailed telemetry evidence."""
        if not self.config.include_sensitive:
            return []
        items = []
        # Query telemetry_events WHERE level = 'detailed_non_sensitive'
        return items

    async def _collect_telemetry_raw(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect raw telemetry evidence."""
        if not self.config.include_sensitive:
            return []
        items = []
        # Query telemetry_events WHERE level = 'raw_order_events'
        # IMPORTANT: This requires special permission and is enterprise-only
        return items

    async def _collect_access_audit(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect access audit evidence."""
        items = []
        # Query access_audits table
        return items

    async def _collect_break_glass_events(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect break-glass event evidence."""
        items = []
        # Query access_audits WHERE is_break_glass = TRUE
        return items

    async def _collect_config_changes(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect config change evidence."""
        items = []
        # Query config_blobs with changes
        return items

    async def _collect_agent_health(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect agent health evidence."""
        items = []
        # Query agent heartbeats and health status
        return items

    async def _collect_system_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> List[EvidenceItem]:
        """Collect system metrics evidence."""
        items = []
        # Query system metrics from monitoring
        return items

    # ===== Helper Methods =====

    def _write_evidence_file(
        self,
        file_path: Path,
        items: List[EvidenceItem],
    ) -> None:
        """Write evidence items to file."""
        # Redact PII if configured
        if self.config.redact_pii:
            items = [self._redact_item(item) for item in items]

        # Write based on format
        if self.config.output_format == ExportFormat.JSON:
            with open(file_path, "w") as f:
                json.dump([i.to_dict() for i in items], f, indent=2)

        elif self.config.output_format == ExportFormat.JSON_LINES:
            file_path = file_path.with_suffix(".jsonl")
            with open(file_path, "w") as f:
                for item in items:
                    f.write(json.dumps(item.to_dict()) + "\n")

    def _redact_item(self, item: EvidenceItem) -> EvidenceItem:
        """Redact PII from evidence item."""
        # Deep copy and redact sensitive fields
        redacted_data = self._redact_dict(item.data)
        return EvidenceItem(
            type=item.type,
            id=item.id,
            timestamp=item.timestamp,
            data=redacted_data,
            source=item.source,
            workspace_id=item.workspace_id,
            agent_id=item.agent_id,
            checksum=item.checksum,
        )

    def _redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from dictionary."""
        import copy
        import re

        result = copy.deepcopy(data)
        sensitive_patterns = [
            r"password", r"secret", r"key", r"token", r"credential",
            r"api_key", r"private", r"ssn", r"social_security",
            r"credit_card", r"card_number", r"cvv", r"expiry",
            r"account.*number", r"routing.*number", r"email",
            r"phone", r"address", r"ip_address",
        ]
        pattern = re.compile("|".join(sensitive_patterns), re.IGNORECASE)

        def redact_recursive(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: "[REDACTED]" if pattern.search(k) else redact_recursive(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [redact_recursive(item) for item in obj]
            return obj

        return redact_recursive(result)

    def _create_manifest(
        self,
        pack: EvidencePack,
        items: List[EvidenceItem],
    ) -> Dict[str, Any]:
        """Create pack manifest."""
        return {
            "version": EVIDENCE_PACK_VERSION,
            "pack_id": str(pack.id),
            "created_at": pack.created_at.isoformat(),
            "created_by": pack.created_by,
            "description": pack.description,
            "time_range": {
                "start": pack.start_time.isoformat() if pack.start_time else None,
                "end": pack.end_time.isoformat() if pack.end_time else None,
            },
            "evidence_types": [e.value for e in pack.evidence_types],
            "item_count": pack.item_count,
            "checksum": pack.checksum,
            "config": {
                "include_sensitive": self.config.include_sensitive,
                "redact_pii": self.config.redact_pii,
                "output_format": self.config.output_format.value,
            },
            "summary": self._create_summary(items),
        }

    def _create_summary(self, items: List[EvidenceItem]) -> Dict[str, Any]:
        """Create evidence summary."""
        by_type: Dict[str, int] = {}
        for item in items:
            by_type[item.type.value] = by_type.get(item.type.value, 0) + 1

        return {
            "total_items": len(items),
            "by_type": by_type,
        }

    def _calculate_directory_checksum(
        self,
        dir_path: Path,
        exclude: Optional[List[str]] = None,
    ) -> str:
        """Calculate checksum of all files in directory."""
        exclude = exclude or []
        sha256 = hashlib.sha256()

        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file() and file_path.name not in exclude:
                sha256.update(file_path.name.encode())
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        sha256.update(chunk)

        return f"sha256:{sha256.hexdigest()}"

    def _create_archive(self, source_path: Path, archive_name: str) -> Path:
        """Create ZIP archive of evidence."""
        output_path = self.config.output_path
        output_path.mkdir(parents=True, exist_ok=True)

        zip_path = output_path / f"{archive_name}.zip"

        # Create ZIP file first
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_path.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_path)
                    zf.write(file_path, arcname)

        # Optionally gzip the ZIP file
        if self.config.compress:
            gz_path = zip_path.with_suffix(".zip.gz")
            with open(zip_path, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Remove the uncompressed ZIP
            zip_path.unlink()
            return gz_path

        return zip_path

    async def _sign_pack(self, checksum: str) -> Optional[str]:
        """Sign the evidence pack checksum."""
        # Placeholder - would use actual signing implementation
        # In real implementation: use sigstore/cosign or GPG
        import base64
        sig_data = f"CCEA-EVIDENCE-SIG::{checksum}::{datetime.utcnow().isoformat()}"
        return base64.b64encode(sig_data.encode()).decode()

    async def _verify_signature(self, checksum: str, signature: str) -> bool:
        """Verify pack signature."""
        # Placeholder - would verify actual signature
        return True

    async def _upload_to_destination(self, pack: EvidencePack) -> None:
        """Upload pack to configured destination."""
        if self.config.destination == ExportDestination.S3:
            await self._upload_to_s3(pack)
        elif self.config.destination == ExportDestination.GCS:
            await self._upload_to_gcs(pack)
        elif self.config.destination == ExportDestination.AZURE:
            await self._upload_to_azure(pack)

    async def _upload_to_s3(self, pack: EvidencePack) -> None:
        """Upload to S3."""
        # Placeholder - would use boto3
        pass

    async def _upload_to_gcs(self, pack: EvidencePack) -> None:
        """Upload to Google Cloud Storage."""
        # Placeholder - would use google-cloud-storage
        pass

    async def _upload_to_azure(self, pack: EvidencePack) -> None:
        """Upload to Azure Blob Storage."""
        # Placeholder - would use azure-storage-blob
        pass

    def _verify_evidence_file(self, file_path: Path) -> bool:
        """Verify evidence file integrity."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            return isinstance(data, list)
        except Exception:
            return False
