# -*- coding: utf-8 -*-
"""
Artifact Builder - Builds strategy artifacts.

CLOUD ZONE ONLY.

Builds artifacts that can be deployed to agents.
Artifacts are signed and contain manifest with metadata.
"""

from __future__ import annotations

import os
import socket
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ccea.artifact.sbom import SBOMFormat, generate_sbom
from ccea.crypto.keys import KeyPair
from ccea.crypto.signing import sign_message
from packages.shared.contracts.manifest import (
    ArtifactManifest,
    ArtifactFormat,
    Provenance,
    RuntimeRequirements,
    Signature,
    SignatureAlgorithm,
)
from packages.shared.utils.hashing import compute_file_hash


@dataclass
class BuildConfig:
    """Configuration for artifact build."""

    strategy_id: str
    strategy_name: str
    version: str
    entrypoint: str

    # Source
    source_path: Path = Path(".")
    requirements_file: Optional[Path] = None

    # Output
    # IMPORTANT: Format MUST match actual artifact type (Phase 3, 3.1)
    # ZIP_BUNDLE is default for Python strategies
    # OCI_IMAGE requires actual OCI image building
    output_format: ArtifactFormat = ArtifactFormat.ZIP_BUNDLE
    output_path: Optional[Path] = None

    # Signing
    sign_artifact: bool = True
    signature_algorithm: SignatureAlgorithm = SignatureAlgorithm.SIGSTORE

    # Runtime
    runtime_requirements: RuntimeRequirements = field(
        default_factory=RuntimeRequirements
    )

    # Git info (for provenance)
    git_repo: str = ""
    git_sha: str = ""
    git_branch: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "version": self.version,
            "entrypoint": self.entrypoint,
            "source_path": str(self.source_path),
            "requirements_file": str(self.requirements_file) if self.requirements_file else None,
            "output_format": self.output_format.value,
            "output_path": str(self.output_path) if self.output_path else None,
            "sign_artifact": self.sign_artifact,
            "signature_algorithm": self.signature_algorithm.value,
            "runtime_requirements": self.runtime_requirements.to_dict(),
            "git_repo": self.git_repo,
            "git_sha": self.git_sha,
            "git_branch": self.git_branch,
        }


@dataclass
class BuildResult:
    """Result of artifact build."""

    success: bool = False
    manifest: Optional[ArtifactManifest] = None
    artifact_path: Optional[Path] = None
    artifact_digest: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    build_time_ms: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "manifest": self.manifest.to_dict() if self.manifest else None,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "artifact_digest": self.artifact_digest,
            "errors": self.errors,
            "warnings": self.warnings,
            "build_time_ms": self.build_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class ArtifactBuilder:
    """
    Builds strategy artifacts.

    Creates signed, versioned artifacts that can be deployed to agents.

    Usage:
        builder = ArtifactBuilder()

        config = BuildConfig(
            strategy_id="momentum_btc",
            strategy_name="BTC Momentum Strategy",
            version="1.0.0",
            entrypoint="strategy:MomentumStrategy",
        )

        result = builder.build(config)
        if result.success:
            print(f"Artifact: {result.artifact_digest}")
    """

    def __init__(
        self,
        registry_url: str = "",
        builder_id: str = "",
        signing_key: Optional[KeyPair] = None,
    ):
        """Initialize builder."""
        self._registry_url = registry_url
        self._builder_id = builder_id or f"builder-{uuid4().hex[:8]}"
        self._signing_key = signing_key

    def build(self, config: BuildConfig) -> BuildResult:
        """
        Build artifact from configuration.

        Args:
            config: Build configuration

        Returns:
            BuildResult with artifact details
        """
        import time

        start_time = time.time()
        result = BuildResult()

        try:
            if not config.sign_artifact:
                result.errors.append("Unsigned artifacts are not allowed (sign_artifact must be True)")
                return result

            if self._signing_key is None:
                result.errors.append("No signing key configured for artifact builder")
                return result

            # Validate source
            if not self._validate_source(config, result):
                return result

            # Create provenance
            provenance = self._create_provenance(config)

            # Determine output directory
            output_dir = config.output_path or Path(
                tempfile.mkdtemp(prefix="ccea_cloud_artifact_")
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # Package artifact (zip bundle baseline; OCI image can be added later)
            artifact_file = output_dir / "artifact.zip"
            self._package_artifact(config, artifact_file, result)
            if result.errors:
                return result

            # Phase 3 (3.1): Validate format honesty
            if not self._validate_format_honesty(config, artifact_file, result):
                return result

            # Compute digest
            artifact_digest = compute_file_hash(artifact_file, with_prefix=True)

            # Generate SBOM (required)
            sbom_file = output_dir / "sbom.cyclonedx.json"
            try:
                generate_sbom(
                    source_dir=config.source_path,
                    artifact_name=config.strategy_name or config.strategy_id,
                    artifact_version=config.version,
                    output_path=sbom_file,
                    requirements_file=config.requirements_file,
                    output_format=SBOMFormat.CYCLONEDX_JSON,
                )
            except Exception as e:
                result.errors.append(f"SBOM generation failed: {e}")
                return result

            # Create signature
            signature = self._sign_artifact(artifact_file, artifact_digest, config)
            if signature.algorithm == SignatureAlgorithm.NONE or not signature.signature_value:
                result.errors.append("Artifact signature is required but missing")
                return result

            # Create manifest
            manifest = ArtifactManifest(
                strategy_id=config.strategy_id,
                strategy_name=config.strategy_name,
                version=config.version,
                format=config.output_format,
                artifact_digest=artifact_digest,
                registry_url=self._registry_url,
                entrypoint=config.entrypoint,
                signature=signature,
                sbom_ref=str(sbom_file.name),
                provenance=provenance,
                runtime=config.runtime_requirements,
                change_class="trading_impacting",
            )

            # Always write manifest alongside artifact bundle
            self._write_manifest(output_dir, manifest)
            result.artifact_path = output_dir

            result.success = True
            result.manifest = manifest
            result.artifact_digest = artifact_digest

        except Exception as e:
            result.errors.append(f"Build failed: {e}")

        result.build_time_ms = int((time.time() - start_time) * 1000)
        return result

    def _validate_source(self, config: BuildConfig, result: BuildResult) -> bool:
        """Validate source files exist."""
        if not config.source_path.exists():
            result.errors.append(f"Source path not found: {config.source_path}")
            return False
        if config.requirements_file and not config.requirements_file.exists():
            result.errors.append(f"Requirements file not found: {config.requirements_file}")
            return False
        return True

    def _create_provenance(self, config: BuildConfig) -> Provenance:
        """Create build provenance."""
        deps_lock_digest = ""
        if config.requirements_file and config.requirements_file.exists():
            deps_lock_digest = compute_file_hash(config.requirements_file, with_prefix=True)

        return Provenance(
            git_repo=config.git_repo,
            git_sha=config.git_sha,
            git_branch=config.git_branch,
            builder_id=self._builder_id,
            build_timestamp=datetime.utcnow(),
            build_host=socket.gethostname(),
            ci_job_id=os.getenv("GITHUB_RUN_ID") or os.getenv("CI_JOB_ID"),
            ci_pipeline_url=os.getenv("GITHUB_SERVER_URL") and os.getenv("GITHUB_REPOSITORY") and os.getenv("GITHUB_RUN_ID")
            and f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}/actions/runs/{os.getenv('GITHUB_RUN_ID')}",
            deps_lock_digest=deps_lock_digest,
        )

    def _validate_format_honesty(
        self,
        config: BuildConfig,
        artifact_path: Path,
        result: BuildResult,
    ) -> bool:
        """
        Validate that declared format matches actual artifact.

        Phase 3 (3.1): Format field MUST be honest.
        """
        import zipfile
        import tarfile

        actual_format = None

        # Detect actual format
        if artifact_path.exists():
            if zipfile.is_zipfile(artifact_path):
                actual_format = ArtifactFormat.ZIP_BUNDLE
            elif tarfile.is_tarfile(artifact_path):
                # Could be OCI image tarball
                actual_format = ArtifactFormat.OCI_IMAGE
            elif artifact_path.suffix == ".whl":
                actual_format = ArtifactFormat.WHEEL

        if actual_format is None:
            result.warnings.append(
                f"Could not determine actual artifact format for {artifact_path}"
            )
            return True  # Allow, but warn

        if config.output_format != actual_format:
            result.errors.append(
                f"Format mismatch: declared format is {config.output_format.value}, "
                f"but actual artifact is {actual_format.value}. "
                f"Format MUST match actual artifact type (Phase 3, 3.1)."
            )
            return False

        return True

    def _package_artifact(
        self,
        config: BuildConfig,
        output_file: Path,
        result: BuildResult,
    ) -> None:
        """
        Package artifact content.

        Baseline packaging: ZIP bundle.

        The produced artifact is what gets digest-pinned and signed.
        """
        root = config.source_path.resolve()

        excluded_dirs = {".git", "__pycache__", ".venv", "venv", "build", "dist"}

        def should_skip(path: Path) -> bool:
            parts = set(path.parts)
            return bool(parts & excluded_dirs)

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for file_path in sorted(root.rglob("*")):
                    if file_path.is_dir():
                        continue
                    if should_skip(file_path.relative_to(root)):
                        continue
                    rel = file_path.relative_to(root).as_posix()
                    zf.write(file_path, arcname=f"src/{rel}")

                if config.requirements_file:
                    zf.write(config.requirements_file, arcname="requirements.txt")
        except Exception as e:
            result.errors.append(f"Packaging failed: {e}")

    def _sign_artifact(
        self,
        artifact_file: Path,
        digest: str,
        config: BuildConfig,
    ) -> Signature:
        """Sign artifact."""
        if not config.sign_artifact or self._signing_key is None:
            return Signature(algorithm=SignatureAlgorithm.NONE)

        artifact_bytes = artifact_file.read_bytes()
        signature_value = sign_message(artifact_bytes, self._signing_key.private_key)

        return Signature(
            algorithm=config.signature_algorithm,
            signature_value=signature_value,
            timestamp=datetime.utcnow(),
            key_id=self._signing_key.key_id,
        )

    def _write_manifest(self, output_dir: Path, manifest: ArtifactManifest) -> None:
        """Write manifest to disk."""
        import json

        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = output_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
