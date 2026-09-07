"""Validate build_hash_report.json contents against built artifacts.

The report is produced by setup.py during `make build`. This validator is
invoked by `make verify-hash` and CI to guarantee that:
- the report exists and references real artifacts
- recorded SHA256 hashes match the on-disk files
- all built extension modules are covered by the report
- builds are performed with Python 3.12.x (reproducibility policy)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Iterable, Sequence


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Hash report missing: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# The report describes the extensions built in place. Everything below is either
# a copy of one of them (setup.py leaves duplicates under build/) or belongs to
# an installed dependency, so counting them as unreported artifacts only ever
# produced false failures.
IGNORED_DIRS = frozenset(
    {
        "build",
        "dist",
        ".venv",
        "venv",
        "env",
        "site-packages",
        ".git",
        "node_modules",
        ".eggs",
        "__pycache__",
    }
)


def find_built_artifacts(roots: Iterable[Path]) -> set[Path]:
    artifacts: set[Path] = set()
    suffixes = set(EXTENSION_SUFFIXES)
    for root in roots:
        for suffix in suffixes:
            for path in root.rglob(f"*{suffix}"):
                if IGNORED_DIRS.intersection(path.parts):
                    continue
                artifacts.add(path)
    return {path.resolve() for path in artifacts if path.exists()}


def validate_report(report_path: Path, artifact_roots: Iterable[Path], require_all: bool) -> dict:
    data = load_report(report_path)

    build_info = data.get("build_info", {})
    python_version = build_info.get("python_version", "")
    if not python_version.startswith("3.12"):
        raise ValueError(
            f"build_hash_report.json must be generated with Python 3.12.x, found {python_version!r}"
        )

    extensions = data.get("extensions")
    if not isinstance(extensions, dict) or not extensions:
        raise ValueError("Hash report must include at least one compiled extension.")

    recorded_paths = {}
    for name, info in extensions.items():
        raw_path = info.get("path")
        if not raw_path:
            raise ValueError(f"Extension {name} is missing a path entry.")
        path = (report_path.parent / Path(raw_path)).resolve()
        recorded_paths[name] = path

    mismatches: list[str] = []
    for name, path in recorded_paths.items():
        if not path.exists():
            mismatches.append(f"{name}: missing file at {path}")
            continue
        recorded_hash = extensions[name].get("sha256")
        actual_hash = compute_sha256(path)
        if recorded_hash != actual_hash:
            mismatches.append(
                f"{name}: hash mismatch (report={recorded_hash}, actual={actual_hash})"
            )

    if require_all:
        built_artifacts = find_built_artifacts(artifact_roots)
        unreported = [
            str(path) for path in built_artifacts if path not in set(recorded_paths.values())
        ]
        if unreported:
            mismatches.append(
                "Built artifacts missing from report: " + ", ".join(sorted(unreported))
            )

    if mismatches:
        raise ValueError("Hash verification failed:\n- " + "\n- ".join(mismatches))

    return {
        "extensions": len(extensions),
        "python_version": python_version,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify build hash report.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("build_hash_report.json"),
        help="Path to build_hash_report.json",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Directory to search for built artifacts (defaults to cwd).",
    )
    parser.add_argument(
        "--require-all-artifacts",
        action="store_true",
        help="Fail if compiled extension files exist that are not captured in the report.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = args.report.resolve()
    roots = [args.root.resolve()]

    result = validate_report(report_path, roots, require_all=args.require_all_artifacts)
    print(
        f"Verified hash report for {result['extensions']} extensions "
        f"(Python {result['python_version']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
