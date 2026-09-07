# -*- coding: utf-8 -*-
"""
CCEA Manifest Builder and Validator.

Creates and validates artifact manifests per schema.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ccea.models.manifest import (
    ArtifactManifest,
    ArtifactType,
    Entrypoint,
    Runtime,
    Provenance,
    LiveCapabilities,
    ChangeClass,
)
from ccea.crypto.digest import compute_json_digest, compute_file_digest


class ManifestBuilder:
    """
    Builder for artifact manifests.

    Provides fluent interface for manifest creation.
    """

    def __init__(self):
        """Initialize builder."""
        self._data: Dict[str, Any] = {
            "schema_version": "1.0.0",
        }

    def set_artifact_id(self, artifact_id: str) -> "ManifestBuilder":
        """Set artifact ID."""
        self._data["artifact_id"] = artifact_id
        return self

    def set_type(self, artifact_type: ArtifactType) -> "ManifestBuilder":
        """Set artifact type."""
        self._data["artifact_type"] = artifact_type
        return self

    def set_name(self, name: str) -> "ManifestBuilder":
        """Set artifact name."""
        self._data["name"] = name
        return self

    def set_description(self, description: str) -> "ManifestBuilder":
        """Set description."""
        self._data["description"] = description
        return self

    def set_version(self, version: str) -> "ManifestBuilder":
        """Set version."""
        self._data["version"] = version
        return self

    def set_entrypoint(
        self,
        module: str,
        class_name: str,
        function: Optional[str] = None,
    ) -> "ManifestBuilder":
        """Set entrypoint."""
        self._data["entrypoint"] = Entrypoint(
            module=module,
            class_name=class_name,
            function=function,
        )
        return self

    def set_runtime(
        self,
        python_version: str,
        min_memory_mb: int = 1024,
        gpu_required: bool = False,
    ) -> "ManifestBuilder":
        """Set runtime requirements."""
        self._data["runtime"] = Runtime(
            python_version=python_version,
            min_memory_mb=min_memory_mb,
            gpu_required=gpu_required,
        )
        return self

    def set_deps_lock_digest(self, digest: str) -> "ManifestBuilder":
        """Set dependencies lock digest."""
        self._data["deps_lock_digest"] = digest
        return self

    def set_provenance(
        self,
        git_sha: str,
        git_branch: Optional[str] = None,
        build_id: Optional[str] = None,
    ) -> "ManifestBuilder":
        """Set provenance."""
        self._data["provenance"] = Provenance(
            git_sha=git_sha,
            git_branch=git_branch,
            build_id=build_id,
        )
        return self

    def set_live_capabilities(
        self,
        requires_broker_access: bool,
        supported_brokers: Optional[List[str]] = None,
    ) -> "ManifestBuilder":
        """Set live capabilities."""
        self._data["live_capabilities"] = LiveCapabilities(
            requires_broker_access=requires_broker_access,
            supported_brokers=supported_brokers,
        )
        return self

    def set_change_class(self, change_class: ChangeClass) -> "ManifestBuilder":
        """Set change class."""
        self._data["change_class"] = change_class
        return self

    def build(self) -> ArtifactManifest:
        """
        Build the manifest.

        Returns:
            ArtifactManifest

        Raises:
            ValueError: If required fields are missing
        """
        # Set created_at if not set
        if "created_at" not in self._data:
            self._data["created_at"] = datetime.utcnow()

        return ArtifactManifest(**self._data)


class ManifestValidator:
    """
    Validator for artifact manifests.

    Validates manifests against schema and business rules.
    """

    # Prohibited fields in manifest (order-like)
    PROHIBITED_FIELDS = {
        "side",
        "quantity",
        "qty",
        "price",
        "order_type",
        "target_position",
        "execute_order",
        "place_order",
        "submit_order",
        "intent",
        "signal",
    }

    def __init__(self, min_schema_version: str = "1.0.0"):
        """
        Initialize validator.

        Args:
            min_schema_version: Minimum supported schema version
        """
        self.min_schema_version = min_schema_version

    def validate(self, manifest: ArtifactManifest) -> List[str]:
        """
        Validate manifest.

        Args:
            manifest: Manifest to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check schema version
        if not self._check_version(manifest.schema_version, self.min_schema_version):
            errors.append(
                f"Schema version {manifest.schema_version} is below minimum {self.min_schema_version}"
            )

        # Check required fields
        if not manifest.artifact_id:
            errors.append("artifact_id is required")

        if not manifest.entrypoint:
            errors.append("entrypoint is required")
        elif not manifest.entrypoint.module or not manifest.entrypoint.class_name:
            errors.append("entrypoint.module and entrypoint.class are required")

        if not manifest.runtime:
            errors.append("runtime is required")
        elif not manifest.runtime.python_version:
            errors.append("runtime.python_version is required")

        if not manifest.deps_lock_digest:
            errors.append("deps_lock_digest is required")
        elif not manifest.deps_lock_digest.startswith("sha256:"):
            errors.append("deps_lock_digest must be a sha256 digest")

        # Check for prohibited fields
        manifest_dict = manifest.model_dump()
        prohibited = self._find_prohibited_fields(manifest_dict)
        if prohibited:
            errors.append(f"Prohibited fields found: {prohibited}")

        return errors

    def validate_file(self, manifest_path: Path) -> List[str]:
        """
        Validate manifest from file.

        Args:
            manifest_path: Path to manifest file

        Returns:
            List of validation errors
        """
        errors = []

        if not manifest_path.exists():
            return ["Manifest file not found"]

        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
            manifest = ArtifactManifest(**data)
            errors = self.validate(manifest)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
        except Exception as e:
            errors.append(f"Invalid manifest: {e}")

        return errors

    def _check_version(self, version: str, min_version: str) -> bool:
        """Check if version meets minimum."""
        try:
            v_parts = [int(x) for x in version.split(".")]
            min_parts = [int(x) for x in min_version.split(".")]

            for v, m in zip(v_parts, min_parts):
                if v > m:
                    return True
                if v < m:
                    return False
            return True
        except ValueError:
            return False

    def _find_prohibited_fields(
        self,
        data: Any,
        path: str = "",
    ) -> List[str]:
        """Recursively find prohibited fields."""
        found = []

        if isinstance(data, dict):
            for key, value in data.items():
                if key in self.PROHIBITED_FIELDS:
                    found.append(f"{path}.{key}" if path else key)
                found.extend(self._find_prohibited_fields(value, f"{path}.{key}" if path else key))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                found.extend(self._find_prohibited_fields(item, f"{path}[{i}]"))

        return found
