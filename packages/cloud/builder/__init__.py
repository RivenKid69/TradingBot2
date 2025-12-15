# -*- coding: utf-8 -*-
"""
CCEA Cloud Builder - Artifact building and registry.

Builds strategy artifacts for deployment to agents.
Artifacts are signed and stored in registry.

Design Doc 8.1: OCI as primary format.
- CloudArtifactPipeline: OCI-first pipeline (canonical)
- ArtifactBuilder: ZIP bundle builder (legacy/fallback)

Key Components:
- CloudArtifactPipeline: OCI-first artifact pipeline (Design Doc 8.1)
- ArtifactBuilder: ZIP bundle builder (legacy)
- StrategyRegistry: Stores artifact metadata

IMPORTANT: Builder creates artifacts but DOES NOT execute trading.
Artifacts are deployed to Agent for execution.
"""

from typing import Final

ZONE: Final[str] = "cloud"

from .artifact_builder import (
    ArtifactBuilder,
    BuildConfig,
    BuildResult,
)

from .registry import (
    StrategyRegistry,
    RegistryEntry,
)

# Design Doc 8.1: OCI-first pipeline (canonical path)
from .oci_pipeline import (
    CloudArtifactPipeline,
    CloudBuildConfig,
    CloudBuildResult,
    build_cloud_artifact,
)

__all__ = [
    # Design Doc 8.1: OCI-first (primary/canonical)
    "CloudArtifactPipeline",
    "CloudBuildConfig",
    "CloudBuildResult",
    "build_cloud_artifact",
    # Legacy ZIP builder (fallback)
    "ArtifactBuilder",
    "BuildConfig",
    "BuildResult",
    # Registry
    "StrategyRegistry",
    "RegistryEntry",
]
