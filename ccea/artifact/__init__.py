# -*- coding: utf-8 -*-
"""
CCEA Artifact Builder.

Provides:
- Artifact packaging (OCI image or zip bundle)
- Manifest generation
- Signature creation and verification
- SBOM generation
- Digest pinning

Security:
- All artifacts signed before publishing
- Agent verifies signature before execution
- No unsigned artifacts allowed
"""

from ccea.artifact.builder import (
    ArtifactBuilder,
    ArtifactPackage,
    BuildConfig,
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

__all__ = [
    # Builder
    "ArtifactBuilder",
    "ArtifactPackage",
    "BuildConfig",
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
]
