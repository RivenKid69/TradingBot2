# -*- coding: utf-8 -*-
"""
CCEA Cloud Builder - Artifact building and registry.

Builds strategy artifacts for deployment to agents.
Artifacts are signed and stored in registry.

Key Components:
- ArtifactBuilder: Builds strategy packages
- StrategyRegistry: Stores artifact metadata
- SigningService: Signs artifacts

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

__all__ = [
    "ArtifactBuilder",
    "BuildConfig",
    "BuildResult",
    "StrategyRegistry",
    "RegistryEntry",
]
