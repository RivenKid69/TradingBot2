# -*- coding: utf-8 -*-
"""
CCEA Artifact Builder.

Creates deployable artifact packages from strategy code:
- Packages code and dependencies
- Generates manifest
- Computes digest
- Signs artifact

Security:
- All artifacts must be signed
- Dependencies are pinned by digest
- No secrets in artifacts
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ccea.crypto.digest import compute_file_digest, compute_digest
from ccea.crypto.keys import KeyPair
from ccea.models.manifest import (
    ArtifactManifest,
    ArtifactType,
    Entrypoint,
    Runtime,
    Provenance,
    LiveCapabilities,
    ChangeClass,
)


@dataclass
class BuildConfig:
    """
    Artifact build configuration.
    """
    # Source
    source_dir: Path
    entrypoint_module: str
    entrypoint_class: str
    entrypoint_function: Optional[str] = None

    # Artifact info
    artifact_id: str = ""
    artifact_type: ArtifactType = ArtifactType.STRATEGY
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0.0"

    # Runtime
    python_version: str = "3.11"
    min_memory_mb: int = 1024
    gpu_required: bool = False

    # Dependencies
    requirements_file: Optional[Path] = None
    include_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    exclude_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", ".git"])

    # Live capabilities (for agent validation)
    requires_broker_access: bool = False
    supported_brokers: List[str] = field(default_factory=list)

    # Build settings
    output_dir: Optional[Path] = None
    sign: bool = True


@dataclass
class ArtifactPackage:
    """
    Built artifact package.
    """
    artifact_id: str
    version: str
    package_path: Path
    manifest_path: Path
    digest: str
    manifest_digest: str
    manifest: ArtifactManifest
    signature: Optional[str] = None
    sbom_path: Optional[Path] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class ArtifactBuilder:
    """
    Artifact builder.

    Creates signed, verifiable artifact packages.
    """

    def __init__(
        self,
        signing_key: Optional[KeyPair] = None,
        git_repo_path: Optional[Path] = None,
    ):
        """
        Initialize artifact builder.

        Args:
            signing_key: Key pair for signing
            git_repo_path: Git repository path for provenance
        """
        self.signing_key = signing_key
        self.git_repo_path = git_repo_path

    def build(self, config: BuildConfig) -> ArtifactPackage:
        """
        Build artifact package.

        Args:
            config: Build configuration

        Returns:
            ArtifactPackage with all metadata

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If build fails
        """
        # Validate config
        self._validate_config(config)

        # Generate artifact ID if not provided
        artifact_id = config.artifact_id or self._generate_artifact_id(config)

        # Setup output directory
        output_dir = config.output_dir or Path(tempfile.mkdtemp(prefix="ccea_build_"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create build directory
        build_dir = output_dir / "build"
        build_dir.mkdir(exist_ok=True)

        # Copy source files
        self._copy_source(config, build_dir)

        # Generate deps lock
        deps_lock_digest = self._generate_deps_lock(config, build_dir)

        # Get provenance info
        provenance = self._get_provenance(config)

        # Create manifest
        manifest = self._create_manifest(
            config=config,
            artifact_id=artifact_id,
            deps_lock_digest=deps_lock_digest,
            provenance=provenance,
        )

        # Write manifest
        manifest_path = build_dir / "manifest.json"
        manifest_data = manifest.model_dump(mode="json")
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2, default=str)

        manifest_digest = compute_file_digest(manifest_path)

        # Create package (zip)
        package_path = output_dir / f"{artifact_id}-{config.version}.zip"
        self._create_zip_package(build_dir, package_path)

        # Compute package digest
        package_digest = compute_file_digest(package_path)

        # Sign if key provided
        signature = None
        if config.sign and self.signing_key:
            from ccea.crypto.signing import sign_message
            signature = sign_message(
                open(package_path, "rb").read(),
                self.signing_key.private_key,
            )

            # Update manifest with signature
            manifest.signature = {
                "algorithm": self.signing_key.algorithm.value,
                "signature_value": signature,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Rewrite manifest with signature
            with open(manifest_path, "w") as f:
                json.dump(manifest.model_dump(mode="json"), f, indent=2, default=str)

        # Generate SBOM (minimal)
        sbom_path = output_dir / f"{artifact_id}-sbom.json"
        self._generate_sbom(config, sbom_path)

        return ArtifactPackage(
            artifact_id=artifact_id,
            version=config.version,
            package_path=package_path,
            manifest_path=manifest_path,
            digest=package_digest,
            manifest_digest=manifest_digest,
            manifest=manifest,
            signature=signature,
            sbom_path=sbom_path,
        )

    def _validate_config(self, config: BuildConfig) -> None:
        """Validate build configuration."""
        if not config.source_dir.exists():
            raise ValueError(f"Source directory not found: {config.source_dir}")

        if not config.entrypoint_module:
            raise ValueError("Entrypoint module is required")

        if not config.entrypoint_class:
            raise ValueError("Entrypoint class is required")

    def _generate_artifact_id(self, config: BuildConfig) -> str:
        """Generate artifact ID from config."""
        base = config.entrypoint_module.replace(".", "_")
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{base}_{timestamp}"

    def _copy_source(self, config: BuildConfig, build_dir: Path) -> None:
        """Copy source files to build directory."""
        import fnmatch

        src_dir = build_dir / "src"
        src_dir.mkdir(exist_ok=True)

        for root, dirs, files in os.walk(config.source_dir):
            # Filter directories
            dirs[:] = [
                d for d in dirs
                if not any(fnmatch.fnmatch(d, p) for p in config.exclude_patterns)
            ]

            for file in files:
                # Check include/exclude patterns
                if not any(fnmatch.fnmatch(file, p) for p in config.include_patterns):
                    continue
                if any(fnmatch.fnmatch(file, p) for p in config.exclude_patterns):
                    continue

                src_path = Path(root) / file
                rel_path = src_path.relative_to(config.source_dir)
                dst_path = src_dir / rel_path

                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)

    def _generate_deps_lock(self, config: BuildConfig, build_dir: Path) -> str:
        """Generate dependencies lock digest."""
        deps_content = ""

        if config.requirements_file and config.requirements_file.exists():
            deps_content = config.requirements_file.read_text()
        else:
            # Generate minimal requirements
            deps_content = f"# Auto-generated\npython=={config.python_version}\n"

        # Write to build dir
        deps_path = build_dir / "requirements.lock"
        with open(deps_path, "w") as f:
            f.write(deps_content)

        return compute_file_digest(deps_path)

    def _get_provenance(self, config: BuildConfig) -> Optional[Provenance]:
        """Get git provenance information."""
        if not self.git_repo_path:
            return None

        try:
            # Get git SHA
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.git_repo_path,
                capture_output=True,
                text=True,
            )
            git_sha = result.stdout.strip() if result.returncode == 0 else None

            if not git_sha:
                return None

            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.git_repo_path,
                capture_output=True,
                text=True,
            )
            git_branch = result.stdout.strip() if result.returncode == 0 else None

            return Provenance(
                git_sha=git_sha,
                git_branch=git_branch,
            )
        except Exception:
            return None

    def _create_manifest(
        self,
        config: BuildConfig,
        artifact_id: str,
        deps_lock_digest: str,
        provenance: Optional[Provenance],
    ) -> ArtifactManifest:
        """Create artifact manifest."""
        return ArtifactManifest(
            schema_version="1.0.0",
            artifact_id=artifact_id,
            artifact_type=config.artifact_type,
            name=config.name,
            description=config.description,
            version=config.version,
            entrypoint=Entrypoint(
                module=config.entrypoint_module,
                class_name=config.entrypoint_class,
                function=config.entrypoint_function,
            ),
            runtime=Runtime(
                python_version=config.python_version,
                min_memory_mb=config.min_memory_mb,
                gpu_required=config.gpu_required,
            ),
            deps_lock_digest=deps_lock_digest,
            live_capabilities=LiveCapabilities(
                requires_broker_access=config.requires_broker_access,
                supported_brokers=config.supported_brokers if config.supported_brokers else None,
            ) if config.requires_broker_access else None,
            provenance=provenance,
            created_at=datetime.utcnow(),
            change_class=ChangeClass.TRADING_IMPACTING if config.requires_broker_access else ChangeClass.NON_TRADING_IMPACTING,
        )

    def _create_zip_package(self, build_dir: Path, package_path: Path) -> None:
        """Create zip package from build directory."""
        with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(build_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_name = file_path.relative_to(build_dir)
                    zf.write(file_path, arc_name)

    def _generate_sbom(self, config: BuildConfig, sbom_path: Path) -> None:
        """Generate minimal SBOM (CycloneDX format)."""
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "version": 1,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "component": {
                    "type": "application",
                    "name": config.artifact_id or config.entrypoint_module,
                    "version": config.version,
                }
            },
            "components": []
        }

        # Add Python runtime
        sbom["components"].append({
            "type": "library",
            "name": "python",
            "version": config.python_version,
        })

        # Add dependencies from requirements if available
        if config.requirements_file and config.requirements_file.exists():
            for line in config.requirements_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    # Parse requirement
                    parts = line.split("==")
                    name = parts[0].strip()
                    version = parts[1].strip() if len(parts) > 1 else "unknown"
                    sbom["components"].append({
                        "type": "library",
                        "name": name,
                        "version": version,
                    })

        with open(sbom_path, "w") as f:
            json.dump(sbom, f, indent=2)


def build_hello_strategy(output_dir: Path, signing_key: Optional[KeyPair] = None) -> ArtifactPackage:
    """
    Build the hello strategy artifact for Phase 1 testing.

    Args:
        output_dir: Output directory
        signing_key: Optional signing key

    Returns:
        ArtifactPackage
    """
    # Create temporary source directory
    src_dir = output_dir / "hello_strategy_src"
    src_dir.mkdir(parents=True, exist_ok=True)

    # Write hello strategy
    strategy_code = '''
"""
Hello Strategy - CCEA Phase 1 Test Artifact.

This strategy does NOT use any broker connection.
It simply demonstrates the CCEA artifact execution flow.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict


class HelloStrategy:
    """
    Minimal test strategy for CCEA Phase 1.

    NO BROKER CONNECTION - Research/Test Only
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize strategy."""
        self.config = config or {}
        self.iteration = 0
        self.started_at = datetime.utcnow()

    def run_iteration(self) -> Dict[str, Any]:
        """Run one iteration of the strategy."""
        self.iteration += 1
        return {
            "iteration": self.iteration,
            "message": f"Hello from CCEA Phase 1! Iteration {self.iteration}",
            "timestamp": datetime.utcnow().isoformat(),
            "config": self.config,
        }

    def stop(self) -> None:
        """Stop the strategy."""
        pass


def main():
    """Main entry point."""
    strategy = HelloStrategy()
    print("Hello Strategy Starting...")
    print(f"Config: {strategy.config}")

    for i in range(5):
        result = strategy.run_iteration()
        print(f"Result: {json.dumps(result)}")
        time.sleep(1)

    print("Hello Strategy Complete")
    return {"status": "success", "iterations": strategy.iteration}


if __name__ == "__main__":
    main()
'''

    (src_dir / "hello_strategy.py").write_text(strategy_code)
    (src_dir / "__init__.py").write_text("# Hello Strategy Package\n")

    # Build config
    config = BuildConfig(
        source_dir=src_dir,
        entrypoint_module="hello_strategy",
        entrypoint_class="HelloStrategy",
        artifact_id="hello_strategy",
        artifact_type=ArtifactType.STRATEGY,
        name="Hello Strategy",
        description="CCEA Phase 1 test strategy - NO broker connection",
        version="1.0.0",
        python_version="3.11",
        requires_broker_access=False,
        output_dir=output_dir,
        sign=signing_key is not None,
    )

    builder = ArtifactBuilder(signing_key=signing_key)
    return builder.build(config)
