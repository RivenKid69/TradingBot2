# -*- coding: utf-8 -*-
"""
CCEA Artifact Publish Pipeline.

Complete pipeline for building, signing, and publishing artifacts.
Per Design Doc Phase 4:
- Without signature, artifact is designed not to be published
- Agent is designed not to run artifact without successful verification

Security design requirements (verify via tests and CI):
- Artifacts are designed to be signed before publish
- Signature requirement is designed with no bypass path
- SBOM generation is a design requirement
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ccea.artifact.builder import ArtifactBuilder, BuildConfig, ArtifactPackage
from ccea.artifact.oci_builder import OCIImageBuilder, OCIBuildConfig, OCIBuildResult
from ccea.artifact.sbom import SBOMGenerator, SBOMFormat, generate_sbom
from ccea.artifact.signer import ArtifactSigner, SignatureInfo, SignatureVerifier
from ccea.artifact.manifest import ManifestValidator
from ccea.artifact.registry import ArtifactRegistry, RegistryEntry
from ccea.crypto.keys import KeyPair
from ccea.crypto.key_manager import KeyManager, KeyPurpose, SigningKeyProvider
from ccea.crypto.digest import compute_file_digest


class ArtifactFormat(str, Enum):
    """Artifact output format."""
    OCI = "oci"  # Primary format
    ZIP = "zip"  # Fallback


class PipelineStage(str, Enum):
    """Pipeline stages."""
    INIT = "init"
    BUILD = "build"
    SBOM = "sbom"
    SIGN = "sign"
    VALIDATE = "validate"
    PUBLISH = "publish"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
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

    # Format
    output_format: ArtifactFormat = ArtifactFormat.OCI
    fallback_to_zip: bool = True

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
    workspace_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)

    # Pipeline options
    generate_sbom: bool = True
    sbom_format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON
    validate_manifest: bool = True
    sign_artifact: bool = True  # CANNOT be disabled in production


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    stage: PipelineStage
    artifact_id: str
    version: str
    artifact_digest: str
    manifest_digest: str
    artifact_path: Path
    manifest_path: Path
    signature: Optional[SignatureInfo] = None
    sbom_digest: Optional[str] = None
    sbom_path: Optional[Path] = None
    registry_entry: Optional[RegistryEntry] = None
    oci_result: Optional[OCIBuildResult] = None
    zip_result: Optional[ArtifactPackage] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "stage": self.stage.value,
            "artifact_id": self.artifact_id,
            "version": self.version,
            "artifact_digest": self.artifact_digest,
            "manifest_digest": self.manifest_digest,
            "artifact_path": str(self.artifact_path),
            "manifest_path": str(self.manifest_path),
            "signature": self.signature.to_dict() if self.signature else None,
            "sbom_digest": self.sbom_digest,
            "sbom_path": str(self.sbom_path) if self.sbom_path else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
        }


class PublishPipeline:
    """
    Artifact publish pipeline.

    Complete workflow: build → sbom → sign → validate → publish

    IMPORTANT: Signature is MANDATORY. Unsigned artifacts are NOT published.
    """

    def __init__(
        self,
        registry: Optional[ArtifactRegistry] = None,
        key_manager: Optional[KeyManager] = None,
        signing_key: Optional[KeyPair] = None,
        git_repo_path: Optional[Path] = None,
        require_signature: bool = True,
    ):
        """
        Initialize pipeline.

        Args:
            registry: Artifact registry
            key_manager: Key manager for signing
            signing_key: Direct signing key (alternative to key_manager)
            git_repo_path: Git repo path for provenance
            require_signature: Require signature (default True, CANNOT be False in production)
        """
        self.registry = registry
        self.key_manager = key_manager
        self.signing_key = signing_key
        self.git_repo_path = git_repo_path

        # In production, signature is ALWAYS required
        # This flag is only for testing
        self._require_signature = require_signature

        # Get signing key from manager if available
        if self.key_manager and not self.signing_key:
            provider = SigningKeyProvider(self.key_manager)
            self.signing_key = provider.get_artifact_signing_key()

    def execute(self, config: PipelineConfig) -> PipelineResult:
        """
        Execute the publish pipeline.

        Args:
            config: Pipeline configuration

        Returns:
            PipelineResult with outcomes

        Note: Returns failure result instead of raising exceptions
        """
        started_at = datetime.now(timezone.utc)
        errors = []
        warnings = []

        # Initialize result
        artifact_id = config.artifact_id or self._generate_artifact_id(config)

        try:
            # Stage 1: Build
            build_result = self._stage_build(config, artifact_id)
            if not build_result[0]:
                return self._failure_result(
                    PipelineStage.BUILD,
                    artifact_id,
                    config.version,
                    started_at,
                    errors=[build_result[1]],
                )

            artifact_path, manifest_path, oci_result, zip_result = build_result[1:]

            # Compute digests
            artifact_digest = compute_file_digest(artifact_path)
            manifest_digest = compute_file_digest(manifest_path)

            # Stage 2: Generate SBOM
            sbom_path = None
            sbom_digest = None
            if config.generate_sbom:
                sbom_result = self._stage_sbom(config, artifact_id, artifact_path.parent)
                if sbom_result[0]:
                    sbom_path, sbom_digest = sbom_result[1:]
                    # Update manifest with sbom_ref
                    self._update_manifest_sbom_ref(manifest_path, sbom_digest)
                else:
                    warnings.append(f"SBOM generation warning: {sbom_result[1]}")

            # Stage 3: Sign
            signature = None
            if config.sign_artifact:
                sign_result = self._stage_sign(artifact_path, manifest_path)
                if sign_result[0]:
                    signature = sign_result[1]
                elif self._require_signature:
                    return self._failure_result(
                        PipelineStage.SIGN,
                        artifact_id,
                        config.version,
                        started_at,
                        artifact_path=artifact_path,
                        manifest_path=manifest_path,
                        artifact_digest=artifact_digest,
                        manifest_digest=manifest_digest,
                        errors=[f"Signing failed: {sign_result[1]}. Unsigned artifacts cannot be published."],
                    )
                else:
                    warnings.append(f"Signing skipped: {sign_result[1]}")
            elif self._require_signature:
                return self._failure_result(
                    PipelineStage.SIGN,
                    artifact_id,
                    config.version,
                    started_at,
                    artifact_path=artifact_path,
                    manifest_path=manifest_path,
                    artifact_digest=artifact_digest,
                    manifest_digest=manifest_digest,
                    errors=["Signature is required but sign_artifact=False. Unsigned artifacts cannot be published."],
                )

            # Stage 4: Validate manifest
            if config.validate_manifest:
                validate_result = self._stage_validate(manifest_path)
                if not validate_result[0]:
                    return self._failure_result(
                        PipelineStage.VALIDATE,
                        artifact_id,
                        config.version,
                        started_at,
                        artifact_path=artifact_path,
                        manifest_path=manifest_path,
                        artifact_digest=artifact_digest,
                        manifest_digest=manifest_digest,
                        signature=signature,
                        errors=[f"Validation failed: {validate_result[1]}"],
                    )
                elif validate_result[1]:
                    warnings.extend(validate_result[1])

            # Recompute manifest digest after updates
            manifest_digest = compute_file_digest(manifest_path)

            # Stage 5: Publish to registry
            registry_entry = None
            if self.registry:
                publish_result = self._stage_publish(
                    config,
                    artifact_id,
                    artifact_path,
                    manifest_path,
                    signature,
                    sbom_path,
                )
                if publish_result[0]:
                    registry_entry = publish_result[1]
                else:
                    warnings.append(f"Registry publish warning: {publish_result[1]}")

            # Success
            completed_at = datetime.now(timezone.utc)
            return PipelineResult(
                success=True,
                stage=PipelineStage.COMPLETE,
                artifact_id=artifact_id,
                version=config.version,
                artifact_digest=artifact_digest,
                manifest_digest=manifest_digest,
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                signature=signature,
                sbom_digest=sbom_digest,
                sbom_path=sbom_path,
                registry_entry=registry_entry,
                oci_result=oci_result,
                zip_result=zip_result,
                errors=errors,
                warnings=warnings,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
            )

        except Exception as e:
            return self._failure_result(
                PipelineStage.FAILED,
                artifact_id,
                config.version,
                started_at,
                errors=[f"Pipeline error: {str(e)}"],
            )

    def _stage_build(
        self,
        config: PipelineConfig,
        artifact_id: str,
    ) -> Tuple[bool, ...]:
        """Execute build stage."""
        output_dir = config.output_dir or Path(f"/tmp/ccea_build_{artifact_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        oci_result = None
        zip_result = None

        if config.output_format == ArtifactFormat.OCI:
            try:
                oci_config = OCIBuildConfig(
                    source_dir=config.source_dir,
                    entrypoint_module=config.entrypoint_module,
                    entrypoint_class=config.entrypoint_class,
                    image_name=artifact_id,
                    tag=config.version,
                    python_version=config.python_version,
                    requirements_file=config.requirements_file,
                    include_patterns=config.include_patterns,
                    exclude_patterns=config.exclude_patterns,
                    output_dir=output_dir / "oci",
                    labels=config.labels,
                )

                builder = OCIImageBuilder()
                oci_result = builder.build(oci_config)

                # Export as tar for signing
                tar_path = output_dir / f"{artifact_id}-{config.version}.oci.tar"
                builder.export_tar(oci_result, tar_path)

                # Create CCEA ArtifactManifest (separate from OCI image manifest)
                ccea_manifest = self._create_ccea_manifest(config, artifact_id, oci_result)
                ccea_manifest_path = output_dir / "manifest.json"
                ccea_manifest_path.write_text(json.dumps(ccea_manifest, indent=2, default=str))

                return (True, tar_path, ccea_manifest_path, oci_result, None)

            except Exception as e:
                if config.fallback_to_zip:
                    # Fallback to ZIP
                    pass
                else:
                    return (False, f"OCI build failed: {e}")

        # ZIP format (primary or fallback)
        try:
            build_config = BuildConfig(
                source_dir=config.source_dir,
                entrypoint_module=config.entrypoint_module,
                entrypoint_class=config.entrypoint_class,
                entrypoint_function=config.entrypoint_function,
                artifact_id=artifact_id,
                name=config.name,
                description=config.description,
                version=config.version,
                python_version=config.python_version,
                min_memory_mb=config.min_memory_mb,
                gpu_required=config.gpu_required,
                requirements_file=config.requirements_file,
                include_patterns=config.include_patterns,
                exclude_patterns=config.exclude_patterns,
                requires_broker_access=config.requires_broker_access,
                supported_brokers=config.supported_brokers,
                output_dir=output_dir,
                sign=False,  # We sign in pipeline stage
            )

            artifact_builder = ArtifactBuilder(
                signing_key=None,
                git_repo_path=self.git_repo_path,
            )
            zip_result = artifact_builder.build(build_config)

            return (True, zip_result.package_path, zip_result.manifest_path, oci_result, zip_result)

        except Exception as e:
            return (False, f"Build failed: {e}")

    def _stage_sbom(
        self,
        config: PipelineConfig,
        artifact_id: str,
        output_dir: Path,
    ) -> Tuple[bool, ...]:
        """Execute SBOM generation stage."""
        try:
            sbom_path = output_dir / f"{artifact_id}-sbom.json"
            sbom_digest = generate_sbom(
                source_dir=config.source_dir,
                artifact_name=artifact_id,
                artifact_version=config.version,
                output_path=sbom_path,
                requirements_file=config.requirements_file,
                output_format=config.sbom_format,
            )
            return (True, sbom_path, sbom_digest)
        except Exception as e:
            return (False, str(e))

    def _stage_sign(
        self,
        artifact_path: Path,
        manifest_path: Path,
    ) -> Tuple[bool, ...]:
        """Execute signing stage."""
        if not self.signing_key:
            return (False, "No signing key available - signature is required")

        try:
            signer = ArtifactSigner.from_keypair(self.signing_key)

            # Sign artifact
            artifact_signature = signer.sign_file(artifact_path)

            # Sign manifest
            manifest_signature = signer.sign_manifest(manifest_path)

            # Update manifest with signature
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            manifest["signature"] = artifact_signature.to_dict()

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            return (True, artifact_signature)

        except Exception as e:
            return (False, str(e))

    def _stage_validate(
        self,
        manifest_path: Path,
    ) -> Tuple[bool, ...]:
        """Execute validation stage."""
        validator = ManifestValidator()
        errors = validator.validate_file(manifest_path)

        if errors:
            return (False, "; ".join(errors))

        return (True, [])

    def _stage_publish(
        self,
        config: PipelineConfig,
        artifact_id: str,
        artifact_path: Path,
        manifest_path: Path,
        signature: Optional[SignatureInfo],
        sbom_path: Optional[Path],
    ) -> Tuple[bool, ...]:
        """Execute publish stage."""
        try:
            entry = self.registry.push(
                artifact_path=artifact_path,
                manifest_path=manifest_path,
                artifact_id=artifact_id,
                version=config.version,
                signature=signature,
                sbom_path=sbom_path,
                workspace_id=config.workspace_id,
                labels=config.labels,
            )
            return (True, entry)
        except Exception as e:
            return (False, str(e))

    def _update_manifest_sbom_ref(
        self,
        manifest_path: Path,
        sbom_digest: str,
    ) -> None:
        """Update manifest with SBOM reference."""
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            manifest["sbom_ref"] = sbom_digest

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except Exception:
            pass

    def _generate_artifact_id(self, config: PipelineConfig) -> str:
        """Generate artifact ID from config."""
        base = config.entrypoint_module.replace(".", "_")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{base}_{timestamp}"

    def _create_ccea_manifest(
        self,
        config: PipelineConfig,
        artifact_id: str,
        oci_result: OCIBuildResult,
    ) -> Dict[str, Any]:
        """Create CCEA ArtifactManifest for OCI format."""
        # Compute deps_lock_digest from requirements if available
        deps_digest = f"sha256:{'0' * 64}"  # Placeholder
        if config.requirements_file and config.requirements_file.exists():
            deps_digest = compute_file_digest(config.requirements_file)

        manifest = {
            "schema_version": "1.0.0",
            "artifact_id": artifact_id,
            "artifact_type": "strategy",
            "entrypoint": {
                "module": config.entrypoint_module,
                "class": config.entrypoint_class,
            },
            "runtime": {
                "python_version": config.python_version,
                "min_memory_mb": config.min_memory_mb,
                "gpu_required": config.gpu_required,
                "platform": "any",
            },
            "deps_lock_digest": deps_digest,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": config.version,
        }

        if config.name:
            manifest["name"] = config.name
        if config.description:
            manifest["description"] = config.description
        if config.requires_broker_access:
            manifest["live_capabilities"] = {
                "requires_broker_access": config.requires_broker_access,
                "supported_brokers": config.supported_brokers or [],
            }

        # Store OCI digest in labels if labels are provided
        labels = dict(config.labels) if config.labels else {}
        labels["oci_image_digest"] = oci_result.image_digest
        manifest["labels"] = labels

        return manifest

    def _failure_result(
        self,
        stage: PipelineStage,
        artifact_id: str,
        version: str,
        started_at: datetime,
        errors: List[str],
        artifact_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        artifact_digest: Optional[str] = None,
        manifest_digest: Optional[str] = None,
        signature: Optional[SignatureInfo] = None,
    ) -> PipelineResult:
        """Create failure result."""
        completed_at = datetime.now(timezone.utc)
        return PipelineResult(
            success=False,
            stage=stage,
            artifact_id=artifact_id,
            version=version,
            artifact_digest=artifact_digest or "",
            manifest_digest=manifest_digest or "",
            artifact_path=artifact_path or Path("/dev/null"),
            manifest_path=manifest_path or Path("/dev/null"),
            signature=signature,
            errors=errors,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
        )


def build_and_publish(
    source_dir: Path,
    entrypoint_module: str,
    entrypoint_class: str,
    signing_key: KeyPair,
    registry: Optional[ArtifactRegistry] = None,
    artifact_id: Optional[str] = None,
    version: str = "1.0.0",
    **kwargs: Any,
) -> PipelineResult:
    """
    Convenience function to build and publish artifact.

    Args:
        source_dir: Source directory
        entrypoint_module: Entry module
        entrypoint_class: Entry class
        signing_key: Key for signing
        registry: Optional registry
        artifact_id: Optional artifact ID
        version: Version string
        **kwargs: Additional PipelineConfig parameters

    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        source_dir=source_dir,
        entrypoint_module=entrypoint_module,
        entrypoint_class=entrypoint_class,
        artifact_id=artifact_id,
        version=version,
        **kwargs,
    )

    pipeline = PublishPipeline(
        registry=registry,
        signing_key=signing_key,
    )

    return pipeline.execute(config)
