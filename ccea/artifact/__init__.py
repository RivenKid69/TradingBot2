# -*- coding: utf-8 -*-
"""
CCEA Artifact Builder.

Provides:
- Artifact packaging (OCI image or zip bundle)
- Manifest generation
- Signature creation and verification
- SBOM generation (CycloneDX/SPDX)
- Digest pinning
- Publish pipeline with mandatory signing
- Agent verification

Security:
- All artifacts signed before publishing
- Agent verifies signature before execution
- No unsigned artifacts allowed
- SBOM required for supply chain verification
"""

from ccea.artifact.builder import (
    ArtifactBuilder,
    ArtifactPackage,
    BuildConfig,
    build_hello_strategy,
)

from ccea.artifact.manifest import (
    ManifestBuilder,
    ManifestValidator,
)

from ccea.artifact.signer import (
    ArtifactSigner,
    SignatureVerifier,
    SignatureInfo,
)

from ccea.artifact.registry import (
    ArtifactRegistry,
    RegistryEntry,
)

from ccea.artifact.oci_builder import (
    OCIImageBuilder,
    OCIBuildConfig,
    OCIBuildResult,
    OCIManifest,
    OCILayer,
    create_oci_artifact,
)

from ccea.artifact.sbom import (
    SBOMGenerator,
    SBOMFormat,
    SBOM,
    Component,
    generate_sbom,
)

from ccea.artifact.verifier import (
    ArtifactVerifier,
    VerificationReport,
    VerificationResult,
    RejectionReason,
    SchemaVersionPolicy,
    create_verifier_from_key_manager,
)

from ccea.artifact.pipeline import (
    PublishPipeline,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    ArtifactFormat,
    build_and_publish,
)

__all__ = [
    # Builder
    "ArtifactBuilder",
    "ArtifactPackage",
    "BuildConfig",
    "build_hello_strategy",
    # OCI Builder
    "OCIImageBuilder",
    "OCIBuildConfig",
    "OCIBuildResult",
    "OCIManifest",
    "OCILayer",
    "create_oci_artifact",
    # Manifest
    "ManifestBuilder",
    "ManifestValidator",
    # Signer
    "ArtifactSigner",
    "SignatureVerifier",
    "SignatureInfo",
    # Registry
    "ArtifactRegistry",
    "RegistryEntry",
    # SBOM
    "SBOMGenerator",
    "SBOMFormat",
    "SBOM",
    "Component",
    "generate_sbom",
    # Verifier
    "ArtifactVerifier",
    "VerificationReport",
    "VerificationResult",
    "RejectionReason",
    "SchemaVersionPolicy",
    "create_verifier_from_key_manager",
    # Pipeline
    "PublishPipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
    "ArtifactFormat",
    "build_and_publish",
]
