# -*- coding: utf-8 -*-
"""
Build zone-separated distributions (Cloud vs Agent).

Phase 9 (P1) / WI-BUILD-01:
- Produce separate Cloud and Agent artifacts
- Allow post-build scanning of the Cloud artifact for prohibited trading code

This intentionally builds "wheel-like" zip artifacts to keep the implementation
pure-Python and CI-friendly (no packaging backend required).
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    include_paths: List[str]


def read_project_version(pyproject_path: Path) -> str:
    try:
        import tomllib
    except ImportError:  # pragma: no cover (Py<3.11)
        import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(((data.get("project") or {}).get("version") or "0.0.0"))


def iter_files(root: Path, relative_paths: Sequence[str]) -> Iterable[tuple[Path, str]]:
    """
    Yield (absolute_path, archive_name) pairs for all files under the provided directories.
    """
    for rel in relative_paths:
        base = (root / rel).resolve()
        if not base.exists():
            continue

        if base.is_file():
            arcname = base.relative_to(root.resolve()).as_posix()
            yield base, arcname
            continue

        for file_path in sorted(base.rglob("*")):
            if not file_path.is_file():
                continue
            arcname = file_path.relative_to(root.resolve()).as_posix()
            yield file_path, arcname


def _write_dist_info(zf: zipfile.ZipFile, dist_name: str, version: str) -> None:
    dist_info = f"{dist_name}-{version}.dist-info"
    wheel = "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: tools/build_zone_distributions.py",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )
    metadata = "\n".join(
        [
            "Metadata-Version: 2.1",
            f"Name: {dist_name}",
            f"Version: {version}",
            "Summary: Zone-separated distribution (CCEA)",
            "",
        ]
    )
    zf.writestr(f"{dist_info}/WHEEL", wheel)
    zf.writestr(f"{dist_info}/METADATA", metadata)


def build_wheel_like_artifact(
    *,
    root: Path,
    output_file: Path,
    dist_name: str,
    version: str,
    include_paths: Sequence[str],
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_dist_info(zf, dist_name, version)
        for file_path, arcname in iter_files(root, include_paths):
            zf.write(file_path, arcname=arcname)

    return output_file


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build zone-separated distributions")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument("--out", type=Path, default=Path("dist"), help="Output directory")
    parser.add_argument("--cloud", action="store_true", help="Build cloud distribution")
    parser.add_argument("--agent", action="store_true", help="Build agent distribution")
    args = parser.parse_args(argv)

    if not args.cloud and not args.agent:
        parser.error("At least one of --cloud or --agent is required")

    root = args.root.resolve()
    version = read_project_version(root / "pyproject.toml")

    # Exclude CCEA guardrails + deprecated shims from runtime artifacts:
    # - guardrails contain prohibited-pattern strings used for scanning
    # - ccea/agent references packages.agent (cloud-prohibited import)
    cloud_spec = DistributionSpec(
        name="ccea_cloud",
        include_paths=[
            "packages/cloud",
            "packages/shared",
            "ccea/artifact",
            "ccea/crypto",
            "ccea/models",
            "ccea/contracts",
            "ccea/protocol",
            "ccea/telemetry",
        ],
    )
    agent_spec = DistributionSpec(
        name="ccea_agent",
        include_paths=[
            "packages/agent",
            "packages/shared",
            "ccea/artifact",
            "ccea/crypto",
            "ccea/models",
            "ccea/contracts",
            "ccea/protocol",
            "ccea/telemetry",
        ],
    )

    built: list[Path] = []
    if args.cloud:
        built.append(
            build_wheel_like_artifact(
                root=root,
                output_file=args.out / f"{cloud_spec.name}-{version}-py3-none-any.whl",
                dist_name=cloud_spec.name,
                version=version,
                include_paths=cloud_spec.include_paths,
            )
        )
    if args.agent:
        built.append(
            build_wheel_like_artifact(
                root=root,
                output_file=args.out / f"{agent_spec.name}-{version}-py3-none-any.whl",
                dist_name=agent_spec.name,
                version=version,
                include_paths=agent_spec.include_paths,
            )
        )

    for p in built:
        sys.stdout.write(str(p) + os.linesep)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
