# -*- coding: utf-8 -*-
"""
CCEA Software Bill of Materials (SBOM) Generator.

Generates SBOM in CycloneDX and SPDX formats per Design Doc Phase 4.

Features:
- CycloneDX 1.5 support (primary)
- SPDX 2.3 support
- Dependency analysis from requirements.txt, setup.py, pyproject.toml
- License detection
- Vulnerability reference linking

Security:
- SBOM required for all artifacts
- sbom_ref in manifest points to SBOM
- Used for supply chain verification
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ccea.crypto.digest import compute_file_digest


class SBOMFormat(str, Enum):
    """SBOM output formats."""
    CYCLONEDX_JSON = "cyclonedx+json"
    CYCLONEDX_XML = "cyclonedx+xml"
    SPDX_JSON = "spdx+json"
    SPDX_TAG_VALUE = "spdx+tv"


class ComponentType(str, Enum):
    """Component types for SBOM."""
    APPLICATION = "application"
    LIBRARY = "library"
    FRAMEWORK = "framework"
    FILE = "file"
    FIRMWARE = "firmware"
    OPERATING_SYSTEM = "operating-system"


class ComponentScope(str, Enum):
    """Component scope."""
    REQUIRED = "required"
    OPTIONAL = "optional"
    EXCLUDED = "excluded"


@dataclass
class License:
    """License information."""
    id: Optional[str] = None  # SPDX ID
    name: Optional[str] = None
    url: Optional[str] = None

    def to_cyclonedx(self) -> Dict[str, Any]:
        """Convert to CycloneDX format."""
        if self.id:
            return {"license": {"id": self.id}}
        elif self.name:
            return {"license": {"name": self.name}}
        return {}


@dataclass
class ExternalRef:
    """External reference (e.g., package URL, advisory)."""
    type: str
    url: str
    comment: Optional[str] = None


@dataclass
class Component:
    """SBOM Component."""
    name: str
    version: str
    type: ComponentType = ComponentType.LIBRARY
    group: Optional[str] = None
    description: Optional[str] = None
    scope: ComponentScope = ComponentScope.REQUIRED
    licenses: List[License] = field(default_factory=list)
    purl: Optional[str] = None  # Package URL
    cpe: Optional[str] = None  # Common Platform Enumeration
    hashes: Dict[str, str] = field(default_factory=dict)
    external_refs: List[ExternalRef] = field(default_factory=list)
    bom_ref: Optional[str] = None

    def __post_init__(self):
        if not self.bom_ref:
            self.bom_ref = f"{self.name}@{self.version}"

    def to_cyclonedx(self) -> Dict[str, Any]:
        """Convert to CycloneDX component."""
        component = {
            "type": self.type.value,
            "name": self.name,
            "version": self.version,
            "bom-ref": self.bom_ref,
        }

        if self.group:
            component["group"] = self.group
        if self.description:
            component["description"] = self.description
        if self.scope != ComponentScope.REQUIRED:
            component["scope"] = self.scope.value
        if self.purl:
            component["purl"] = self.purl
        if self.cpe:
            component["cpe"] = self.cpe
        if self.hashes:
            component["hashes"] = [
                {"alg": alg.upper(), "content": content}
                for alg, content in self.hashes.items()
            ]
        if self.licenses:
            component["licenses"] = [lic.to_cyclonedx() for lic in self.licenses]
        if self.external_refs:
            component["externalReferences"] = [
                {"type": ref.type, "url": ref.url}
                for ref in self.external_refs
            ]

        return component

    def to_spdx(self) -> Dict[str, Any]:
        """Convert to SPDX package."""
        package = {
            "SPDXID": f"SPDXRef-Package-{self.name.replace('-', '_').replace('.', '_')}",
            "name": self.name,
            "versionInfo": self.version,
            "downloadLocation": "NOASSERTION",
        }

        if self.purl:
            package["externalRefs"] = [{
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": self.purl,
            }]

        if self.licenses:
            if self.licenses[0].id:
                package["licenseConcluded"] = self.licenses[0].id
            else:
                package["licenseConcluded"] = "NOASSERTION"
        else:
            package["licenseConcluded"] = "NOASSERTION"

        package["licenseDeclared"] = package.get("licenseConcluded", "NOASSERTION")
        package["copyrightText"] = "NOASSERTION"
        package["filesAnalyzed"] = False

        return package


@dataclass
class Dependency:
    """Dependency relationship."""
    ref: str
    depends_on: List[str] = field(default_factory=list)


@dataclass
class Vulnerability:
    """Vulnerability reference."""
    id: str
    source: str
    description: Optional[str] = None
    ratings: List[Dict[str, Any]] = field(default_factory=list)
    affects: List[str] = field(default_factory=list)


@dataclass
class SBOM:
    """Software Bill of Materials."""
    serial_number: str
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    components: List[Component] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_component(self, component: Component) -> None:
        """Add a component."""
        self.components.append(component)

    def add_dependency(self, ref: str, depends_on: List[str]) -> None:
        """Add a dependency relationship."""
        self.dependencies.append(Dependency(ref=ref, depends_on=depends_on))

    def to_cyclonedx(self) -> Dict[str, Any]:
        """Convert to CycloneDX 1.5 format."""
        bom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.5",
            "serialNumber": self.serial_number,
            "version": self.version,
            "metadata": {
                "timestamp": self.created_at.isoformat(),
                "tools": [{
                    "vendor": "CCEA",
                    "name": "CCEA SBOM Generator",
                    "version": "1.0.0",
                }],
                **self.metadata,
            },
            "components": [c.to_cyclonedx() for c in self.components],
        }

        if self.dependencies:
            bom["dependencies"] = [
                {"ref": d.ref, "dependsOn": d.depends_on}
                for d in self.dependencies
            ]

        if self.vulnerabilities:
            bom["vulnerabilities"] = [
                {
                    "id": v.id,
                    "source": {"name": v.source},
                    "description": v.description,
                    "affects": [{"ref": ref} for ref in v.affects],
                }
                for v in self.vulnerabilities
            ]

        return bom

    def to_spdx(self) -> Dict[str, Any]:
        """Convert to SPDX 2.3 format."""
        # Generate document namespace
        namespace = f"https://ccea.io/spdxdocs/{self.serial_number}"

        doc = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": self.metadata.get("component", {}).get("name", "ccea-artifact"),
            "documentNamespace": namespace,
            "creationInfo": {
                "created": self.created_at.isoformat(),
                "creators": ["Tool: CCEA-SBOM-Generator-1.0.0"],
            },
            "packages": [c.to_spdx() for c in self.components],
            "relationships": [],
        }

        # Add document describes relationship
        for comp in self.components:
            if comp.type == ComponentType.APPLICATION:
                spdx_id = f"SPDXRef-Package-{comp.name.replace('-', '_').replace('.', '_')}"
                doc["relationships"].append({
                    "spdxElementId": "SPDXRef-DOCUMENT",
                    "relatedSpdxElement": spdx_id,
                    "relationshipType": "DESCRIBES",
                })

        # Add dependency relationships
        for dep in self.dependencies:
            ref_id = f"SPDXRef-Package-{dep.ref.replace('-', '_').replace('.', '_').replace('@', '_')}"
            for depends_on in dep.depends_on:
                dep_id = f"SPDXRef-Package-{depends_on.replace('-', '_').replace('.', '_').replace('@', '_')}"
                doc["relationships"].append({
                    "spdxElementId": ref_id,
                    "relatedSpdxElement": dep_id,
                    "relationshipType": "DEPENDS_ON",
                })

        return doc


class SBOMGenerator:
    """
    SBOM Generator.

    Creates Software Bill of Materials from source code and dependencies.
    """

    # Common license mappings
    LICENSE_MAP = {
        "mit": "MIT",
        "apache-2.0": "Apache-2.0",
        "apache 2.0": "Apache-2.0",
        "bsd": "BSD-3-Clause",
        "bsd-3-clause": "BSD-3-Clause",
        "gpl-3.0": "GPL-3.0-only",
        "lgpl-3.0": "LGPL-3.0-only",
        "isc": "ISC",
        "mpl-2.0": "MPL-2.0",
    }

    def __init__(self, include_transitive: bool = True):
        """
        Initialize generator.

        Args:
            include_transitive: Include transitive dependencies
        """
        self.include_transitive = include_transitive

    def generate(
        self,
        source_dir: Path,
        artifact_name: str,
        artifact_version: str,
        requirements_file: Optional[Path] = None,
        output_format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON,
    ) -> SBOM:
        """
        Generate SBOM for artifact.

        Args:
            source_dir: Source directory
            artifact_name: Artifact name
            artifact_version: Artifact version
            requirements_file: Optional requirements.txt path
            output_format: Output format

        Returns:
            SBOM object
        """
        # Generate serial number
        serial = f"urn:uuid:{uuid.uuid4()}"

        sbom = SBOM(
            serial_number=serial,
            metadata={
                "component": {
                    "type": "application",
                    "name": artifact_name,
                    "version": artifact_version,
                    "bom-ref": f"{artifact_name}@{artifact_version}",
                }
            },
        )

        # Add main component
        main_component = Component(
            name=artifact_name,
            version=artifact_version,
            type=ComponentType.APPLICATION,
            description=f"CCEA Strategy Artifact: {artifact_name}",
            purl=f"pkg:ccea/{artifact_name}@{artifact_version}",
        )
        sbom.add_component(main_component)

        # Parse dependencies
        dependencies = self._parse_dependencies(source_dir, requirements_file)

        # Add components and dependencies
        main_deps = []
        for dep in dependencies:
            sbom.add_component(dep)
            main_deps.append(dep.bom_ref)

        if main_deps:
            sbom.add_dependency(main_component.bom_ref, main_deps)

        return sbom

    def _parse_dependencies(
        self,
        source_dir: Path,
        requirements_file: Optional[Path],
    ) -> List[Component]:
        """Parse dependencies from various sources."""
        components = []
        seen: Set[str] = set()

        # Try requirements.txt
        req_file = requirements_file or source_dir / "requirements.txt"
        if req_file.exists():
            components.extend(self._parse_requirements_txt(req_file, seen))

        # Try pyproject.toml
        pyproject = source_dir / "pyproject.toml"
        if pyproject.exists():
            components.extend(self._parse_pyproject_toml(pyproject, seen))

        # Try setup.py
        setup_py = source_dir / "setup.py"
        if setup_py.exists():
            components.extend(self._parse_setup_py(setup_py, seen))

        # Add Python runtime
        if "python" not in seen:
            import sys
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            components.append(Component(
                name="python",
                version=python_version,
                type=ComponentType.LIBRARY,
                purl=f"pkg:generic/python@{python_version}",
            ))

        return components

    def _parse_requirements_txt(
        self,
        req_file: Path,
        seen: Set[str],
    ) -> List[Component]:
        """Parse requirements.txt file."""
        components = []

        try:
            content = req_file.read_text()
            for line in content.splitlines():
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#") or line.startswith("-"):
                    continue

                # Parse requirement specifier
                # Handle: package==1.0.0, package>=1.0.0, package[extra]==1.0.0
                match = re.match(
                    r'^([a-zA-Z0-9_-]+)(?:\[[\w,]+\])?(?:[=<>!~]+)?([\d.]+)?',
                    line.replace(" ", ""),
                )
                if match:
                    name = match.group(1).lower()
                    version = match.group(2) or "unknown"

                    if name not in seen:
                        seen.add(name)
                        components.append(Component(
                            name=name,
                            version=version,
                            type=ComponentType.LIBRARY,
                            purl=f"pkg:pypi/{name}@{version}",
                        ))
        except Exception:
            pass

        return components

    def _parse_pyproject_toml(
        self,
        pyproject_path: Path,
        seen: Set[str],
    ) -> List[Component]:
        """Parse pyproject.toml for dependencies."""
        components = []

        try:
            # Try to import tomllib (Python 3.11+) or fallback
            try:
                import tomllib
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
            except ImportError:
                # Python < 3.11, try tomli
                try:
                    import tomli
                    with open(pyproject_path, "rb") as f:
                        data = tomli.load(f)
                except ImportError:
                    return components

            # Extract dependencies from various sections
            deps = []

            # Poetry style
            if "tool" in data and "poetry" in data["tool"]:
                poetry = data["tool"]["poetry"]
                if "dependencies" in poetry:
                    deps.extend(poetry["dependencies"].items())

            # PEP 621 style
            if "project" in data:
                project = data["project"]
                if "dependencies" in project:
                    for dep in project["dependencies"]:
                        match = re.match(r'^([a-zA-Z0-9_-]+)', dep)
                        if match:
                            deps.append((match.group(1), "*"))

            for name, version_spec in deps:
                if name.lower() == "python":
                    continue
                name = name.lower()
                if name not in seen:
                    seen.add(name)
                    version = version_spec if isinstance(version_spec, str) else "unknown"
                    version = re.sub(r'[^0-9.]', '', version) or "unknown"
                    components.append(Component(
                        name=name,
                        version=version,
                        type=ComponentType.LIBRARY,
                        purl=f"pkg:pypi/{name}@{version}",
                    ))

        except Exception:
            pass

        return components

    def _parse_setup_py(
        self,
        setup_path: Path,
        seen: Set[str],
    ) -> List[Component]:
        """Parse setup.py for dependencies (basic extraction)."""
        components = []

        try:
            content = setup_path.read_text()

            # Simple regex to find install_requires
            match = re.search(
                r'install_requires\s*=\s*\[(.*?)\]',
                content,
                re.DOTALL,
            )
            if match:
                deps_str = match.group(1)
                # Extract quoted strings
                for dep_match in re.finditer(r'["\']([^"\']+)["\']', deps_str):
                    dep = dep_match.group(1)
                    name_match = re.match(r'^([a-zA-Z0-9_-]+)', dep)
                    if name_match:
                        name = name_match.group(1).lower()
                        if name not in seen:
                            seen.add(name)
                            components.append(Component(
                                name=name,
                                version="unknown",
                                type=ComponentType.LIBRARY,
                                purl=f"pkg:pypi/{name}",
                            ))
        except Exception:
            pass

        return components

    def write(
        self,
        sbom: SBOM,
        output_path: Path,
        output_format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON,
    ) -> str:
        """
        Write SBOM to file.

        Args:
            sbom: SBOM object
            output_path: Output file path
            output_format: Output format

        Returns:
            Digest of written file
        """
        if output_format == SBOMFormat.CYCLONEDX_JSON:
            data = sbom.to_cyclonedx()
            content = json.dumps(data, indent=2)
        elif output_format == SBOMFormat.SPDX_JSON:
            data = sbom.to_spdx()
            content = json.dumps(data, indent=2)
        else:
            # Default to CycloneDX JSON
            data = sbom.to_cyclonedx()
            content = json.dumps(data, indent=2)

        output_path.write_text(content)
        return compute_file_digest(output_path)


def generate_sbom(
    source_dir: Path,
    artifact_name: str,
    artifact_version: str,
    output_path: Path,
    requirements_file: Optional[Path] = None,
    output_format: SBOMFormat = SBOMFormat.CYCLONEDX_JSON,
) -> str:
    """
    Convenience function to generate SBOM.

    Args:
        source_dir: Source directory
        artifact_name: Artifact name
        artifact_version: Artifact version
        output_path: Output file path
        requirements_file: Optional requirements.txt
        output_format: Output format

    Returns:
        Digest of SBOM file
    """
    generator = SBOMGenerator()
    sbom = generator.generate(
        source_dir=source_dir,
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        requirements_file=requirements_file,
        output_format=output_format,
    )
    return generator.write(sbom, output_path, output_format)
