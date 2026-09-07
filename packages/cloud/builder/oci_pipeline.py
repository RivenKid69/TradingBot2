# -*- coding: utf-8 -*-
"""
OCI-First Artifact Pipeline Wrapper.

CLOUD ZONE ONLY.

Design Doc 8.1: OCI image as primary format.
- Immutable via digest
- Signable (Sigstore/Cosign compatible)
- Portable (local/VPS/on-prem)
- Strict environment isolation

This module wraps ccea.artifact.pipeline.PublishPipeline to provide
Cloud-side artifact building with OCI as the canonical format.

Usage:
    pipeline = CloudArtifactPipeline()
    result = pipeline.build_and_publish(
        source_dir=Path("/path/to/strategy"),
        entrypoint_module="strategy",
        entrypoint_class="MyStrategy",
        signing_key=keypair,
    )

    if result.success:
        print(f"Artifact: {result.artifact_digest}")
        print(f"OCI: {result.oci_result.image_digest}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ccea.artifact.pipeline import (
    PublishPipeline,
    PipelineConfig,
    PipelineResult,
    ArtifactFormat,
    PipelineStage,
)
from ccea.artifact.registry import ArtifactRegistry
from ccea.artifact.sbom import SBOMFormat
from ccea.crypto.keys import KeyPair
from ccea.crypto.key_manager import KeyManager

logger = logging.getLogger(__name__)


@dataclass
class CloudBuildConfig:
    """
    Cloud artifact build configuration.

    Design Doc 8.1: OCI as primary format with ZIP fallback for dev.
    """

    # Source
    source_dir: Path
    entrypoint_module: str
    entrypoint_class: str
    entrypoint_function: Optional[str] = None

    # Artifact metadata
    artifact_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0.0"
    workspace_id: Optional[str] = None

    # Format (Design Doc 8.1: OCI primary, ZIP fallback)
    use_oci_format: bool = True  # Primary format
    fallback_to_zip: bool = True  # Fallback for dev/simple cases

    # Runtime
    python_version: str = "3.11"
    min_memory_mb: int = 1024
    gpu_required: bool = False

    # Dependencies
    requirements_file: Optional[Path] = None
    include_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", ".git"])

    # Live capabilities
    requires_broker_access: bool = False
    supported_brokers: List[str] = field(default_factory=list)

    # Output
    output_dir: Optional[Path] = None
    labels: Dict[str, str] = field(default_factory=dict)

    # SBOM (Design Doc 8.4: mandatory)
    generate_sbom: bool = True
    sbom_format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON

    # Signing (Design Doc: MANDATORY)
    sign_artifact: bool = True  # Cannot be disabled in production


@dataclass
class CloudBuildResult:
    """
    Result of Cloud artifact build.

    Contains both OCI and ZIP results if both were generated.
    """

    success: bool
    artifact_id: str
    version: str
    artifact_digest: str
    manifest_digest: str
    artifact_path: Path
    manifest_path: Path

    # Format used
    format: ArtifactFormat
    oci_image_digest: Optional[str] = None  # OCI image digest if OCI was used

    # SBOM
    sbom_digest: Optional[str] = None
    sbom_path: Optional[Path] = None

    # Signature
    signed: bool = False
    key_id: Optional[str] = None

    # Registry
    registry_ref: Optional[str] = None

    # Errors/warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "artifact_digest": self.artifact_digest,
            "manifest_digest": self.manifest_digest,
            "artifact_path": str(self.artifact_path),
            "manifest_path": str(self.manifest_path),
            "format": self.format.value,
            "oci_image_digest": self.oci_image_digest,
            "sbom_digest": self.sbom_digest,
            "sbom_path": str(self.sbom_path) if self.sbom_path else None,
            "signed": self.signed,
            "key_id": self.key_id,
            "registry_ref": self.registry_ref,
            "errors": self.errors,
            "warnings": self.warnings,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


class CloudArtifactPipeline:
    """
    Cloud-side artifact pipeline with OCI as primary format.

    Design Doc 8.1:
    - OCI image is primary format for production
    - ZIP bundle available for development/simple cases
    - All artifacts MUST be signed
    - SBOM mandatory

    This wraps ccea.artifact.pipeline.PublishPipeline which provides:
    - build → sbom → sign → validate → publish
    - Strict signature enforcement
    - OCI builder + verifier integration
    """

    def __init__(
        self,
        registry: Optional[ArtifactRegistry] = None,
        key_manager: Optional[KeyManager] = None,
        signing_key: Optional[KeyPair] = None,
        git_repo_path: Optional[Path] = None,
    ):
        """
        Initialize pipeline.

        Args:
            registry: Artifact registry for publishing
            key_manager: Key manager for signing
            signing_key: Direct signing key (alternative to key_manager)
            git_repo_path: Git repo path for provenance
        """
        self._registry = registry
        self._key_manager = key_manager
        self._signing_key = signing_key
        self._git_repo_path = git_repo_path

        # Create underlying pipeline
        self._pipeline = PublishPipeline(
            registry=registry,
            key_manager=key_manager,
            signing_key=signing_key,
            git_repo_path=git_repo_path,
            require_signature=True,  # ALWAYS required
        )

    def build_and_publish(self, config: CloudBuildConfig) -> CloudBuildResult:
        """
        Build and publish artifact using OCI-first pipeline.

        Args:
            config: Build configuration

        Returns:
            CloudBuildResult with build details

        Note:
            OCI format is attempted first. If it fails and fallback_to_zip is True,
            ZIP format is used as fallback.
        """
        started_at = datetime.now(timezone.utc)

        # Build pipeline config
        output_format = ArtifactFormat.OCI if config.use_oci_format else ArtifactFormat.ZIP

        pipeline_config = PipelineConfig(
            source_dir=config.source_dir,
            entrypoint_module=config.entrypoint_module,
            entrypoint_class=config.entrypoint_class,
            entrypoint_function=config.entrypoint_function,
            artifact_id=config.artifact_id,
            name=config.name,
            description=config.description,
            version=config.version,
            output_format=output_format,
            fallback_to_zip=config.fallback_to_zip,
            python_version=config.python_version,
            min_memory_mb=config.min_memory_mb,
            gpu_required=config.gpu_required,
            requirements_file=config.requirements_file,
            include_patterns=config.include_patterns,
            exclude_patterns=config.exclude_patterns,
            requires_broker_access=config.requires_broker_access,
            supported_brokers=config.supported_brokers,
            output_dir=config.output_dir,
            workspace_id=config.workspace_id,
            labels=config.labels,
            generate_sbom=config.generate_sbom,
            sbom_format=config.sbom_format,
            validate_manifest=True,
            sign_artifact=config.sign_artifact,
        )

        # Execute pipeline
        result = self._pipeline.execute(pipeline_config)

        # Convert to CloudBuildResult
        completed_at = datetime.now(timezone.utc)

        # Determine format used
        actual_format = ArtifactFormat.OCI if result.oci_result else ArtifactFormat.ZIP

        # Get OCI digest if available
        oci_image_digest = None
        if result.oci_result:
            oci_image_digest = result.oci_result.image_digest

        return CloudBuildResult(
            success=result.success,
            artifact_id=result.artifact_id,
            version=result.version,
            artifact_digest=result.artifact_digest,
            manifest_digest=result.manifest_digest,
            artifact_path=result.artifact_path,
            manifest_path=result.manifest_path,
            format=actual_format,
            oci_image_digest=oci_image_digest,
            sbom_digest=result.sbom_digest,
            sbom_path=result.sbom_path,
            signed=result.signature is not None,
            key_id=result.signature.key_id if result.signature else None,
            registry_ref=result.registry_entry.digest_ref if result.registry_entry else None,
            errors=result.errors,
            warnings=result.warnings,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
        )

    def build_oci(self, config: CloudBuildConfig) -> CloudBuildResult:
        """
        Build artifact using OCI format only (no ZIP fallback).

        Args:
            config: Build configuration

        Returns:
            CloudBuildResult

        Raises:
            Fails if OCI build fails (no fallback)
        """
        # Force OCI only
        config.use_oci_format = True
        config.fallback_to_zip = False
        return self.build_and_publish(config)

    def build_zip(self, config: CloudBuildConfig) -> CloudBuildResult:
        """
        Build artifact using ZIP format (for development/simple cases).

        Args:
            config: Build configuration

        Returns:
            CloudBuildResult
        """
        # Force ZIP
        config.use_oci_format = False
        config.fallback_to_zip = False
        return self.build_and_publish(config)


def build_cloud_artifact(
    source_dir: Path,
    entrypoint_module: str,
    entrypoint_class: str,
    signing_key: KeyPair,
    registry: Optional[ArtifactRegistry] = None,
    artifact_id: Optional[str] = None,
    version: str = "1.0.0",
    use_oci: bool = True,
    **kwargs,
) -> CloudBuildResult:
    """
    Convenience function to build Cloud artifact.

    Design Doc 8.1: Uses OCI as primary format by default.

    Args:
        source_dir: Source directory
        entrypoint_module: Entry module
        entrypoint_class: Entry class
        signing_key: Key for signing
        registry: Optional registry
        artifact_id: Optional artifact ID
        version: Version string
        use_oci: Use OCI format (default True per Design Doc)
        **kwargs: Additional config parameters

    Returns:
        CloudBuildResult
    """
    config = CloudBuildConfig(
        source_dir=source_dir,
        entrypoint_module=entrypoint_module,
        entrypoint_class=entrypoint_class,
        artifact_id=artifact_id,
        version=version,
        use_oci_format=use_oci,
        **kwargs,
    )

    pipeline = CloudArtifactPipeline(
        registry=registry,
        signing_key=signing_key,
    )

    return pipeline.build_and_publish(config)


# Export
__all__ = [
    "CloudBuildConfig",
    "CloudBuildResult",
    "CloudArtifactPipeline",
    "build_cloud_artifact",
]
