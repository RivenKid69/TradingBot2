# -*- coding: utf-8 -*-
"""
Artifact Digest Pinning and Registry Allowlist Check.

GDPR Phase 2 Implementation: Enforces artifact security guardrails.

This module:
    - Verifies all artifact references use sha256 digests (no "latest")
    - Enforces registry allowlist (only approved registries)
    - Validates artifact manifest integrity

Design Doc Reference:
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L911 (digest pinning)
    - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L913 (registry allowlist)
    - docs/design/CCEA_CLOUD/CI_GUARDRAILS.md (BT-003, RT-001)

Usage:
    python -m ccea.guardrails.artifact_digest_check

Exit codes:
    0 - Check passed
    1 - Check failed (violations found)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Pattern, Tuple


# ============================================================================
# Constants
# ============================================================================

# Valid digest pattern (sha256:hex64)
DIGEST_PATTERN: Pattern = re.compile(r"^sha256:[a-f0-9]{64}$")

# Invalid reference patterns (no "latest" or bare tags)
INVALID_REF_PATTERNS: List[Tuple[Pattern, str]] = [
    (re.compile(r":latest\s*$"), "Using ':latest' tag is forbidden"),
    (re.compile(r":[0-9]+\.[0-9]+\.[0-9]+\s*$"), "Bare version tags forbidden - use digest"),
    (re.compile(r":v[0-9]+\s*$"), "Bare version tags forbidden - use digest"),
    (re.compile(r":stable\s*$"), "Using ':stable' tag is forbidden"),
    (re.compile(r":edge\s*$"), "Using ':edge' tag is forbidden"),
    (re.compile(r":dev\s*$"), "Using ':dev' tag is forbidden"),
]

# Allowed registries (EU-only for GDPR compliance)
ALLOWED_REGISTRIES: FrozenSet[str] = frozenset({
    # Platform registries (EU)
    "registry.ccea.io",
    "eu.registry.ccea.io",
    "ghcr.io/ccea",
    # Enterprise registries (customer-controlled, must be EU)
    "registry.enterprise.ccea.io",
    # Local development
    "localhost",
    "127.0.0.1",
})

# Files to check for artifact references
ARTIFACT_REFERENCE_FILES: List[str] = [
    # Deployment configs
    "deploy/docker/docker-compose.yml",
    "deploy/docker/docker-compose.*.yml",
    "deploy/helm/**/values.yaml",
    "deploy/helm/**/values-*.yaml",
    # Kubernetes manifests
    "deploy/k8s/**/*.yaml",
    "deploy/k8s/**/*.yml",
    # CI/CD configs
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    # Protocol and schema files
    "docs/schemas/artifact_manifest.schema.json",
    # Application code with artifact references
    "packages/cloud/builder/*.py",
    "packages/cloud/enterprise/*.py",
    "packages/agent/updater/*.py",
]

# Field names that should contain digests
DIGEST_FIELD_NAMES: FrozenSet[str] = frozenset({
    "artifact_digest",
    "config_digest",
    "new_artifact_digest",
    "old_artifact_digest",
    "new_config_digest",
    "old_config_digest",
    "image_digest",
    "digest",
    "sha256",
    "content_digest",
    "evidence_hash",
})


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DigestViolation:
    """A digest pinning or registry violation."""
    file_path: str
    line_number: int
    violation_type: str  # "no_digest", "invalid_digest", "invalid_registry", "latest_tag"
    message: str
    value: str
    severity: str = "HIGH"


@dataclass
class CheckResult:
    """Result of artifact digest check."""
    passed: bool
    message: str
    violations: List[DigestViolation] = field(default_factory=list)
    files_checked: int = 0
    digests_found: int = 0


# ============================================================================
# Validation Functions
# ============================================================================

def is_valid_digest(value: str) -> bool:
    """
    Check if value is a valid SHA256 digest.

    Valid format: sha256:[64 hex characters]
    """
    return bool(DIGEST_PATTERN.match(value))


def extract_registry(image_ref: str) -> Optional[str]:
    """
    Extract registry from image reference.

    Examples:
        registry.ccea.io/image:tag -> registry.ccea.io
        ghcr.io/ccea/image@sha256:... -> ghcr.io/ccea
    """
    # Remove digest/tag
    ref = image_ref.split("@")[0].split(":")[0]

    # Extract registry (first component if contains .)
    parts = ref.split("/")
    if len(parts) >= 2 and ("." in parts[0] or ":" in parts[0] or parts[0] == "localhost"):
        # First two parts for namespaced registries like ghcr.io/ccea
        if len(parts) >= 3 and parts[0] in ("ghcr.io", "docker.io", "gcr.io"):
            return f"{parts[0]}/{parts[1]}"
        return parts[0]

    return None


def is_allowed_registry(registry: str) -> bool:
    """Check if registry is in allowlist."""
    if registry is None:
        return True  # Local references

    registry_lower = registry.lower()
    for allowed in ALLOWED_REGISTRIES:
        if registry_lower == allowed.lower() or registry_lower.startswith(f"{allowed.lower()}/"):
            return True

    return False


def check_for_invalid_patterns(
    value: str,
) -> Optional[str]:
    """
    Check if value matches any invalid patterns.

    Returns error message if invalid, None if ok.
    """
    for pattern, message in INVALID_REF_PATTERNS:
        if pattern.search(value):
            return message
    return None


# ============================================================================
# File Checking Functions
# ============================================================================

def find_files_to_check() -> List[Path]:
    """Find all files that should be checked for artifact references."""
    import glob

    files = []
    for pattern in ARTIFACT_REFERENCE_FILES:
        matches = glob.glob(pattern, recursive=True)
        files.extend(Path(m) for m in matches if Path(m).is_file())

    return list(set(files))


def check_yaml_file(file_path: Path) -> List[DigestViolation]:
    """Check a YAML file for artifact reference violations."""
    violations = []

    try:
        with open(file_path, "r") as f:
            content = f.read()
            lines = content.split("\n")
    except Exception:
        return violations

    for line_num, line in enumerate(lines, 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Check for image references
        if "image:" in line:
            # Extract image value
            match = re.search(r"image:\s*['\"]?([^'\"#\n]+)", line)
            if match:
                image_ref = match.group(1).strip()

                # Check for invalid patterns (latest, etc.)
                error = check_for_invalid_patterns(image_ref)
                if error:
                    violations.append(DigestViolation(
                        file_path=str(file_path),
                        line_number=line_num,
                        violation_type="latest_tag",
                        message=error,
                        value=image_ref,
                    ))

                # Check for digest
                if "@sha256:" not in image_ref and not is_valid_digest(image_ref):
                    # Check registry
                    registry = extract_registry(image_ref)
                    if registry and not is_allowed_registry(registry):
                        violations.append(DigestViolation(
                            file_path=str(file_path),
                            line_number=line_num,
                            violation_type="invalid_registry",
                            message=f"Registry '{registry}' not in allowlist",
                            value=image_ref,
                        ))

        # Check for digest fields
        for field_name in DIGEST_FIELD_NAMES:
            if f"{field_name}:" in line or f'"{field_name}"' in line:
                # Extract value
                match = re.search(rf'{field_name}["\']?\s*:\s*["\']?([^"\'#\n,}}]+)', line)
                if match:
                    value = match.group(1).strip()
                    if value and value != "null" and not is_valid_digest(value):
                        violations.append(DigestViolation(
                            file_path=str(file_path),
                            line_number=line_num,
                            violation_type="invalid_digest",
                            message=f"Field '{field_name}' must be a valid SHA256 digest",
                            value=value,
                        ))

    return violations


def check_json_file(file_path: Path) -> List[DigestViolation]:
    """Check a JSON file for artifact reference violations."""
    violations = []

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception:
        return violations

    def check_recursive(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key

                # Check digest fields
                if key.lower() in DIGEST_FIELD_NAMES:
                    if isinstance(value, str) and value and not is_valid_digest(value):
                        violations.append(DigestViolation(
                            file_path=str(file_path),
                            line_number=0,  # JSON doesn't have line numbers easily
                            violation_type="invalid_digest",
                            message=f"Field '{key}' must be a valid SHA256 digest",
                            value=value,
                        ))

                # Check image references
                if key.lower() in ("image", "artifact", "artifact_ref"):
                    if isinstance(value, str):
                        error = check_for_invalid_patterns(value)
                        if error:
                            violations.append(DigestViolation(
                                file_path=str(file_path),
                                line_number=0,
                                violation_type="latest_tag",
                                message=error,
                                value=value,
                            ))

                check_recursive(value, current_path)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_recursive(item, f"{path}[{i}]")

    check_recursive(data)
    return violations


def check_python_file(file_path: Path) -> List[DigestViolation]:
    """Check a Python file for artifact reference violations."""
    violations = []

    try:
        with open(file_path, "r") as f:
            content = f.read()
            lines = content.split("\n")
    except Exception:
        return violations

    for line_num, line in enumerate(lines, 1):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#"):
            continue

        # Check for digest field assignments
        for field_name in DIGEST_FIELD_NAMES:
            # Pattern: field_name = "value" or field_name="value"
            pattern = rf'{field_name}\s*=\s*["\']([^"\']+)["\']'
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                if value and not is_valid_digest(value):
                    # Skip if it's a variable reference or pattern
                    if not value.startswith("{") and not value.startswith("$"):
                        violations.append(DigestViolation(
                            file_path=str(file_path),
                            line_number=line_num,
                            violation_type="invalid_digest",
                            message=f"Field '{field_name}' must be a valid SHA256 digest",
                            value=value,
                        ))

        # Check for :latest patterns in strings
        if ":latest" in line and not "forbidden" in line.lower():
            match = re.search(r'["\']([^"\']*:latest[^"\']*)["\']', line)
            if match:
                value = match.group(1)
                violations.append(DigestViolation(
                    file_path=str(file_path),
                    line_number=line_num,
                    violation_type="latest_tag",
                    message="Using ':latest' tag is forbidden",
                    value=value,
                ))

    return violations


def check_file(file_path: Path) -> List[DigestViolation]:
    """Check a file for artifact reference violations."""
    suffix = file_path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return check_yaml_file(file_path)
    elif suffix == ".json":
        return check_json_file(file_path)
    elif suffix == ".py":
        return check_python_file(file_path)

    return []


# ============================================================================
# Main Check Function
# ============================================================================

def check_artifact_digests() -> CheckResult:
    """
    Check all artifact references for digest pinning and registry compliance.

    Returns:
        CheckResult with pass/fail status and violations
    """
    files = find_files_to_check()
    all_violations = []
    digests_found = 0

    for file_path in files:
        violations = check_file(file_path)
        all_violations.extend(violations)

    if not all_violations:
        return CheckResult(
            passed=True,
            message="All artifact references are properly pinned with digests",
            violations=[],
            files_checked=len(files),
            digests_found=digests_found,
        )

    return CheckResult(
        passed=False,
        message=f"Found {len(all_violations)} artifact reference violations",
        violations=all_violations,
        files_checked=len(files),
        digests_found=digests_found,
    )


# ============================================================================
# Validation API (for runtime use)
# ============================================================================

def validate_artifact_reference(
    reference: str,
    require_digest: bool = True,
    check_registry: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Validate an artifact reference at runtime.

    Args:
        reference: Artifact reference string
        require_digest: Whether to require digest (default True)
        check_registry: Whether to check registry allowlist (default True)

    Returns:
        (is_valid, error_message)
    """
    # Check for invalid patterns
    error = check_for_invalid_patterns(reference)
    if error:
        return False, error

    # Check for digest
    if require_digest:
        if "@sha256:" not in reference:
            return False, "Artifact reference must include @sha256: digest"

        # Extract and validate digest
        parts = reference.split("@")
        if len(parts) == 2:
            if not is_valid_digest(parts[1]):
                return False, f"Invalid digest format: {parts[1]}"

    # Check registry
    if check_registry:
        registry = extract_registry(reference)
        if registry and not is_allowed_registry(registry):
            return False, f"Registry '{registry}' not in allowlist"

    return True, None


def validate_digest(digest: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a digest string.

    Args:
        digest: Digest string (should be sha256:...)

    Returns:
        (is_valid, error_message)
    """
    if not digest:
        return False, "Digest cannot be empty"

    if not is_valid_digest(digest):
        return False, f"Invalid digest format: {digest}. Expected: sha256:<64 hex chars>"

    return True, None


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """Main entry point for CI."""
    print("=" * 60)
    print("ARTIFACT DIGEST PINNING CHECK")
    print("=" * 60)

    result = check_artifact_digests()

    print(f"\nFiles checked: {result.files_checked}")

    if result.passed:
        print(f"\n[PASS] {result.message}")
        return 0
    else:
        print(f"\n[FAIL] {result.message}")
        print("\nViolations:")
        for v in result.violations:
            print(f"  [{v.severity}] {v.file_path}:{v.line_number}")
            print(f"         {v.violation_type}: {v.message}")
            print(f"         Value: {v.value}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
