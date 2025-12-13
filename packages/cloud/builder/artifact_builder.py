# -*- coding: utf-8 -*-
"""
Artifact Builder - Builds strategy artifacts.

CLOUD ZONE ONLY.

Builds artifacts that can be deployed to agents.
Artifacts are signed and contain manifest with metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, List, Optional
from uuid import UUID, uuid4

from packages.shared.contracts.manifest import (
    ArtifactManifest,
    ArtifactFormat,
    Provenance,
    RuntimeRequirements,
    Signature,
    SignatureAlgorithm,
)
from packages.shared.utils.hashing import compute_sha256, compute_file_hash


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
    output_format: ArtifactFormat = ArtifactFormat.OCI_IMAGE
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
    ):
        """Initialize builder."""
        self._registry_url = registry_url
        self._builder_id = builder_id or f"builder-{uuid4().hex[:8]}"

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
            # Validate source
            if not self._validate_source(config, result):
                return result

            # Create provenance
            provenance = self._create_provenance(config)

            # Build artifact (simplified - in reality this would package the code)
            artifact_content = self._package_artifact(config, result)
            if not artifact_content:
                return result

            # Compute digest
            artifact_digest = compute_sha256(artifact_content, with_prefix=True)

            # Create signature
            signature = self._sign_artifact(
                artifact_content, artifact_digest, config
            )

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
                provenance=provenance,
                runtime=config.runtime_requirements,
                change_class="trading_impacting",
            )

            # Write artifact if output path specified
            if config.output_path:
                self._write_artifact(config.output_path, artifact_content, manifest)
                result.artifact_path = config.output_path

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
        return True

    def _create_provenance(self, config: BuildConfig) -> Provenance:
        """Create build provenance."""
        return Provenance(
            git_repo=config.git_repo,
            git_sha=config.git_sha,
            git_branch=config.git_branch,
            builder_id=self._builder_id,
            build_timestamp=datetime.utcnow(),
        )

    def _package_artifact(
        self,
        config: BuildConfig,
        result: BuildResult,
    ) -> Optional[bytes]:
        """
        Package artifact content.

        In real implementation, this would:
        1. Collect source files
        2. Resolve dependencies
        3. Create OCI image or zip bundle
        """
        # Simplified: create placeholder content
        import json

        content = {
            "strategy_id": config.strategy_id,
            "version": config.version,
            "entrypoint": config.entrypoint,
            "built_at": datetime.utcnow().isoformat(),
        }
        return json.dumps(content).encode()

    def _sign_artifact(
        self,
        content: bytes,
        digest: str,
        config: BuildConfig,
    ) -> Signature:
        """Sign artifact."""
        if not config.sign_artifact:
            return Signature(algorithm=SignatureAlgorithm.NONE)

        # In real implementation, this would use sigstore or GPG
        # For now, create placeholder signature
        signature_value = compute_sha256(content + digest.encode())

        return Signature(
            algorithm=config.signature_algorithm,
            signature_value=signature_value,
            timestamp=datetime.utcnow(),
        )

    def _write_artifact(
        self,
        output_path: Path,
        content: bytes,
        manifest: ArtifactManifest,
    ) -> None:
        """Write artifact to disk."""
        import json

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write artifact
        artifact_file = output_path / "artifact.bin"
        with open(artifact_file, "wb") as f:
            f.write(content)

        # Write manifest
        manifest_file = output_path / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)
