# -*- coding: utf-8 -*-
"""
Provenance Validator - Ensures mandatory provenance fields are present.

Phase 3 (3.3): SBOM + provenance with mandatory fields.

Design Doc:
- SBOM must always be generated
- Provenance must include: git_sha, deps_lock_digest, builder_id, build_timestamp
- For ML models: dataset_refs, training_run_id, params_hash

CLOUD ZONE ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Final, FrozenSet, List, Optional, Set

from packages.shared.contracts.manifest import ArtifactManifest, Provenance


class ProvenanceValidationLevel(str, Enum):
    """Validation strictness levels."""

    STRICT = "strict"  # All mandatory fields required
    STANDARD = "standard"  # Core fields required
    RELAXED = "relaxed"  # Minimum fields required (dev only)


class ArtifactType(str, Enum):
    """Artifact types with different provenance requirements."""

    STRATEGY = "strategy"
    MODEL = "model"
    FEATURE_PIPELINE = "feature_pipeline"
    ENSEMBLE = "ensemble"
    DATA_TRANSFORM = "data_transform"


# Core mandatory provenance fields for all artifacts
CORE_MANDATORY_FIELDS: Final[FrozenSet[str]] = frozenset(
    [
        "builder_id",
        "build_timestamp",
    ]
)

# Standard mandatory fields (for production builds)
STANDARD_MANDATORY_FIELDS: Final[FrozenSet[str]] = frozenset(
    [
        "builder_id",
        "build_timestamp",
        "deps_lock_digest",
    ]
)

# Strict mandatory fields (for CI/CD builds)
STRICT_MANDATORY_FIELDS: Final[FrozenSet[str]] = frozenset(
    [
        "builder_id",
        "build_timestamp",
        "deps_lock_digest",
        "git_sha",
    ]
)

# Additional mandatory fields for ML model artifacts
MODEL_MANDATORY_FIELDS: Final[FrozenSet[str]] = frozenset(
    [
        "training_run_id",
        "params_hash",
    ]
)

# Recommended (but not mandatory) fields
RECOMMENDED_FIELDS: Final[FrozenSet[str]] = frozenset(
    [
        "git_repo",
        "git_branch",
        "git_tag",
        "ci_job_id",
        "ci_pipeline_url",
    ]
)


@dataclass
class ProvenanceValidationResult:
    """Result of provenance validation."""

    valid: bool = False
    missing_mandatory: List[str] = field(default_factory=list)
    missing_recommended: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    artifact_type: Optional[ArtifactType] = None
    validation_level: ProvenanceValidationLevel = ProvenanceValidationLevel.STANDARD

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "missing_mandatory": self.missing_mandatory,
            "missing_recommended": self.missing_recommended,
            "warnings": self.warnings,
            "errors": self.errors,
            "artifact_type": self.artifact_type.value if self.artifact_type else None,
            "validation_level": self.validation_level.value,
        }


@dataclass
class SBOMValidationResult:
    """Result of SBOM validation."""

    valid: bool = False
    sbom_present: bool = False
    sbom_format: Optional[str] = None
    component_count: int = 0
    has_licenses: bool = False
    has_vulnerabilities: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "sbom_present": self.sbom_present,
            "sbom_format": self.sbom_format,
            "component_count": self.component_count,
            "has_licenses": self.has_licenses,
            "has_vulnerabilities": self.has_vulnerabilities,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ProvenanceValidator:
    """
    Validates provenance information in artifact manifests.

    Phase 3 (3.3): Ensures mandatory fields are present for supply chain integrity.

    Usage:
        validator = ProvenanceValidator(level=ProvenanceValidationLevel.STRICT)

        result = validator.validate(manifest)
        if not result.valid:
            for error in result.errors:
                print(f"Error: {error}")
    """

    def __init__(
        self,
        level: ProvenanceValidationLevel = ProvenanceValidationLevel.STANDARD,
        artifact_type: Optional[ArtifactType] = None,
    ):
        """
        Initialize validator.

        Args:
            level: Validation strictness level
            artifact_type: Type of artifact being validated
        """
        self._level = level
        self._artifact_type = artifact_type

    def validate(
        self,
        manifest: ArtifactManifest,
        artifact_type: Optional[ArtifactType] = None,
    ) -> ProvenanceValidationResult:
        """
        Validate provenance in manifest.

        Args:
            manifest: Artifact manifest
            artifact_type: Override artifact type

        Returns:
            ProvenanceValidationResult
        """
        result = ProvenanceValidationResult(
            artifact_type=artifact_type or self._artifact_type,
            validation_level=self._level,
        )

        provenance = manifest.provenance
        if provenance is None:
            result.errors.append("Provenance is missing from manifest")
            result.valid = False
            return result

        # Get mandatory fields based on level
        mandatory = self._get_mandatory_fields(result.artifact_type)

        # Check mandatory fields
        provenance_dict = provenance.to_dict()
        for field_name in mandatory:
            value = provenance_dict.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                result.missing_mandatory.append(field_name)

        # Check model-specific fields
        if result.artifact_type == ArtifactType.MODEL:
            for field_name in MODEL_MANDATORY_FIELDS:
                value = provenance_dict.get(field_name)
                if value is None or (isinstance(value, str) and not value.strip()):
                    result.missing_mandatory.append(field_name)
                    result.warnings.append(
                        f"Model artifact missing {field_name} - "
                        "required for ML provenance tracking"
                    )

        # Check recommended fields
        for field_name in RECOMMENDED_FIELDS:
            value = provenance_dict.get(field_name)
            if value is None or (isinstance(value, str) and not value.strip()):
                result.missing_recommended.append(field_name)

        # Build errors list
        if result.missing_mandatory:
            result.errors.append(
                f"Missing mandatory provenance fields: {', '.join(result.missing_mandatory)}"
            )

        # Add warnings for missing recommended
        if result.missing_recommended:
            result.warnings.append(
                f"Missing recommended provenance fields: {', '.join(result.missing_recommended)}"
            )

        # Validate timestamp format
        if provenance.build_timestamp:
            try:
                if isinstance(provenance.build_timestamp, str):
                    datetime.fromisoformat(provenance.build_timestamp.replace("Z", "+00:00"))
            except ValueError as e:
                result.errors.append(f"Invalid build_timestamp format: {e}")

        # Validate git_sha format (if present)
        if provenance.git_sha:
            if len(provenance.git_sha) not in [40, 64]:  # SHA-1 or SHA-256
                result.warnings.append(
                    f"Unusual git_sha length: {len(provenance.git_sha)}. "
                    "Expected 40 (SHA-1) or 64 (SHA-256) characters."
                )

        # Validate deps_lock_digest format (if present)
        if provenance.deps_lock_digest:
            if not provenance.deps_lock_digest.startswith("sha256:"):
                result.warnings.append(
                    "deps_lock_digest should have 'sha256:' prefix for consistency"
                )

        result.valid = len(result.errors) == 0
        return result

    def validate_provenance_dict(
        self,
        provenance_dict: Dict[str, Any],
        artifact_type: Optional[ArtifactType] = None,
    ) -> ProvenanceValidationResult:
        """
        Validate provenance from dictionary.

        Args:
            provenance_dict: Provenance as dictionary
            artifact_type: Artifact type

        Returns:
            ProvenanceValidationResult
        """
        provenance = Provenance.from_dict(provenance_dict)
        manifest = ArtifactManifest(provenance=provenance)
        return self.validate(manifest, artifact_type)

    def _get_mandatory_fields(
        self,
        artifact_type: Optional[ArtifactType],
    ) -> FrozenSet[str]:
        """Get mandatory fields based on validation level."""
        if self._level == ProvenanceValidationLevel.STRICT:
            return STRICT_MANDATORY_FIELDS
        elif self._level == ProvenanceValidationLevel.STANDARD:
            return STANDARD_MANDATORY_FIELDS
        else:
            return CORE_MANDATORY_FIELDS


class SBOMValidator:
    """
    Validates SBOM presence and content.

    Phase 3 (3.3): SBOM must always be generated and valid.

    Usage:
        validator = SBOMValidator()
        result = validator.validate(sbom_path)
    """

    # Supported SBOM formats
    SUPPORTED_FORMATS: Final[Set[str]] = {
        "CycloneDX",
        "SPDX",
    }

    def __init__(self, require_licenses: bool = False):
        """
        Initialize validator.

        Args:
            require_licenses: Whether to require license information
        """
        self._require_licenses = require_licenses

    def validate_from_path(self, sbom_path: str) -> SBOMValidationResult:
        """
        Validate SBOM from file path.

        Args:
            sbom_path: Path to SBOM file

        Returns:
            SBOMValidationResult
        """
        import json
        from pathlib import Path

        result = SBOMValidationResult()

        path = Path(sbom_path)
        if not path.exists():
            result.errors.append(f"SBOM file not found: {sbom_path}")
            return result

        result.sbom_present = True

        try:
            with open(path) as f:
                sbom_data = json.load(f)
        except json.JSONDecodeError as e:
            result.errors.append(f"Invalid SBOM JSON: {e}")
            return result

        return self.validate(sbom_data)

    def validate(self, sbom_data: Dict[str, Any]) -> SBOMValidationResult:
        """
        Validate SBOM content.

        Args:
            sbom_data: SBOM as dictionary

        Returns:
            SBOMValidationResult
        """
        result = SBOMValidationResult()
        result.sbom_present = True

        # Detect format
        if "bomFormat" in sbom_data and sbom_data.get("bomFormat") == "CycloneDX":
            result.sbom_format = "CycloneDX"
            self._validate_cyclonedx(sbom_data, result)
        elif "spdxVersion" in sbom_data:
            result.sbom_format = "SPDX"
            self._validate_spdx(sbom_data, result)
        else:
            result.errors.append("Unknown SBOM format. Supported: CycloneDX, SPDX")
            return result

        result.valid = len(result.errors) == 0
        return result

    def _validate_cyclonedx(
        self,
        sbom_data: Dict[str, Any],
        result: SBOMValidationResult,
    ) -> None:
        """Validate CycloneDX format."""
        # Check version
        spec_version = sbom_data.get("specVersion", "")
        if not spec_version:
            result.errors.append("Missing specVersion in CycloneDX SBOM")

        # Check components
        components = sbom_data.get("components", [])
        result.component_count = len(components)

        if result.component_count == 0:
            result.warnings.append("SBOM has no components listed")

        # Check for licenses
        has_licenses = False
        for comp in components:
            if comp.get("licenses"):
                has_licenses = True
                break
        result.has_licenses = has_licenses

        if self._require_licenses and not has_licenses:
            result.errors.append("SBOM missing license information (required)")

        # Check for vulnerabilities (optional)
        vulnerabilities = sbom_data.get("vulnerabilities", [])
        result.has_vulnerabilities = len(vulnerabilities) > 0

    def _validate_spdx(
        self,
        sbom_data: Dict[str, Any],
        result: SBOMValidationResult,
    ) -> None:
        """Validate SPDX format."""
        # Check version
        spdx_version = sbom_data.get("spdxVersion", "")
        if not spdx_version:
            result.errors.append("Missing spdxVersion in SPDX SBOM")

        # Check packages
        packages = sbom_data.get("packages", [])
        result.component_count = len(packages)

        if result.component_count == 0:
            result.warnings.append("SBOM has no packages listed")

        # Check for licenses
        has_licenses = False
        for pkg in packages:
            if pkg.get("licenseConcluded") or pkg.get("licenseDeclared"):
                has_licenses = True
                break
        result.has_licenses = has_licenses

        if self._require_licenses and not has_licenses:
            result.errors.append("SBOM missing license information (required)")


def validate_manifest_provenance(
    manifest: ArtifactManifest,
    level: ProvenanceValidationLevel = ProvenanceValidationLevel.STANDARD,
    artifact_type: Optional[ArtifactType] = None,
) -> ProvenanceValidationResult:
    """
    Convenience function to validate manifest provenance.

    Args:
        manifest: Artifact manifest
        level: Validation level
        artifact_type: Artifact type

    Returns:
        ProvenanceValidationResult
    """
    validator = ProvenanceValidator(level=level, artifact_type=artifact_type)
    return validator.validate(manifest, artifact_type)


def validate_sbom(
    sbom_path_or_data: Any,
    require_licenses: bool = False,
) -> SBOMValidationResult:
    """
    Convenience function to validate SBOM.

    Args:
        sbom_path_or_data: Path to SBOM file or SBOM dictionary
        require_licenses: Whether to require license information

    Returns:
        SBOMValidationResult
    """
    validator = SBOMValidator(require_licenses=require_licenses)

    if isinstance(sbom_path_or_data, str):
        return validator.validate_from_path(sbom_path_or_data)
    elif isinstance(sbom_path_or_data, dict):
        return validator.validate(sbom_path_or_data)
    else:
        result = SBOMValidationResult()
        result.errors.append("Invalid SBOM input: expected path or dictionary")
        return result
