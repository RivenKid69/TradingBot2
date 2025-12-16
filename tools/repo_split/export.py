#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export git-tracked files into repo seeds according to tools/repo_split/mapping.yaml.

This does NOT create git history. It creates a clean directory tree suitable for:
  - initializing new repositories (ccea-sdk, ccea-agent, ccea-cloud)
  - then adding fresh LICENSE/README/CI per repo
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class RepoMapping:
    name: str
    include_globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]


def _run_git_ls_files(repo_root: Path) -> list[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=str(repo_root))
    files = out.decode("utf-8").splitlines()
    return [f for f in files if f and not f.endswith("/")]


def _matches_any(path: str, globs: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in globs)


def _load_mapping(
    mapping_path: Path,
) -> tuple[Mapping[str, RepoMapping], tuple[str, ...], tuple[str, ...]]:
    data = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    exclude_globs = tuple(defaults.get("exclude_globs", []))
    public_safety = data.get("public_safety", {}) or {}
    forbidden_globs = tuple(public_safety.get("forbidden_globs", []))
    forbidden_content_regex = tuple(public_safety.get("forbidden_content_regex", []))

    repos: dict[str, RepoMapping] = {}
    for name, spec in (data.get("repos") or {}).items():
        repo_excludes = tuple(spec.get("exclude_globs", []))
        repos[name] = RepoMapping(
            name=name,
            include_globs=tuple(spec.get("include_globs", [])),
            exclude_globs=tuple(exclude_globs + repo_excludes),
        )
    return repos, exclude_globs, forbidden_globs, forbidden_content_regex


def _enforce_public_safety(repo_name: str, selected_files: Sequence[str], forbidden_globs: Sequence[str]) -> None:
    if repo_name not in {"ccea-sdk", "ccea-agent"}:
        return
    violations = [f for f in selected_files if _matches_any(f, forbidden_globs)]
    if violations:
        preview = "\n".join(f"  - {v}" for v in violations[:25])
        more = "" if len(violations) <= 25 else f"\n  ... +{len(violations) - 25} more"
        raise SystemExit(
            f"[FATAL] public safety violation: forbidden files selected for {repo_name}:\n{preview}{more}\n"
            "Fix tools/repo_split/mapping.yaml (include/exclude) before exporting."
        )


def _enforce_public_content_safety(
    repo_root: Path,
    repo_name: str,
    selected_files: Sequence[str],
    forbidden_regex: Sequence[str],
) -> None:
    if repo_name not in {"ccea-sdk", "ccea-agent"}:
        return
    if not forbidden_regex:
        return

    compiled = [re.compile(pat) for pat in forbidden_regex]
    violations: list[tuple[str, str]] = []

    # Only scan code-like text files; keep this fast and deterministic.
    scan_exts = {".py", ".pyi", ".md", ".toml", ".yaml", ".yml", ".json", ".ini"}
    for rel in selected_files:
        if Path(rel).suffix.lower() not in scan_exts:
            continue
        src = repo_root / rel
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for rx in compiled:
            m = rx.search(text)
            if m:
                snippet = (m.group(0) or "").strip()
                violations.append((rel, snippet[:120]))
                break

    if violations:
        preview = "\n".join(f"  - {path}: {snippet}" for path, snippet in violations[:25])
        more = "" if len(violations) <= 25 else f"\n  ... +{len(violations) - 25} more"
        raise SystemExit(
            f"[FATAL] public content safety violation in {repo_name} export:\n{preview}{more}\n"
            "Fix tools/repo_split/mapping.yaml (include/exclude) or refactor code before exporting."
        )


def _select_files(
    tracked_files: Sequence[str],
    include_globs: Sequence[str],
    exclude_globs: Sequence[str],
) -> list[str]:
    selected: list[str] = []
    for rel in tracked_files:
        if _matches_any(rel, exclude_globs):
            continue
        if _matches_any(rel, include_globs):
            selected.append(rel)
    return sorted(set(selected))


def _copy_files(repo_root: Path, out_dir: Path, files: Sequence[str]) -> None:
    for rel in files:
        src = repo_root / rel
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)


def _apply_templates(repo_root: Path, repo_name: str, out_dir: Path) -> None:
    templates_root = repo_root / "tools" / "repo_split" / "templates" / repo_name
    if not templates_root.is_dir():
        return

    for root, _dirs, files in os.walk(templates_root):
        root_path = Path(root)
        rel_root = root_path.relative_to(templates_root)
        for filename in files:
            src = root_path / filename
            dst = out_dir / rel_root / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst, follow_symlinks=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export repo split seeds from mapping.yaml (git-tracked files only).")
    parser.add_argument("--repo", choices=["ccea-sdk", "ccea-agent", "ccea-cloud", "all"], required=True)
    parser.add_argument("--mapping", default="tools/repo_split/mapping.yaml")
    parser.add_argument("--out", default="", help="Output directory (ignored for --repo all unless you pass --out-root)")
    parser.add_argument("--out-root", default="dist/repo-split", help="Output root for --repo all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Delete output directory before export")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mapping_path = repo_root / args.mapping
    repos, _, forbidden_globs, forbidden_content_regex = _load_mapping(mapping_path)

    tracked = _run_git_ls_files(repo_root)

    def export_one(repo_name: str, out_dir: Path) -> None:
        mapping = repos[repo_name]
        files = _select_files(tracked, mapping.include_globs, mapping.exclude_globs)
        _enforce_public_safety(repo_name, files, forbidden_globs)
        _enforce_public_content_safety(repo_root, repo_name, files, forbidden_content_regex)
        if args.dry_run:
            print(f"[dry-run] {repo_name}: {len(files)} files")
            for sample in files[:25]:
                print(f"  {sample}")
            if len(files) > 25:
                print(f"  ... +{len(files) - 25} more")
            return

        if args.clean and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _copy_files(repo_root, out_dir, files)
        _apply_templates(repo_root, repo_name, out_dir)
        print(f"[ok] {repo_name}: wrote {len(files)} files to {out_dir}")

    if args.repo == "all":
        out_root = repo_root / args.out_root
        export_one("ccea-sdk", out_root / "ccea-sdk")
        export_one("ccea-agent", out_root / "ccea-agent")
        export_one("ccea-cloud", out_root / "ccea-cloud")
        return 0

    if args.repo not in repos:
        raise SystemExit(f"Unknown repo in mapping: {args.repo}")

    out_dir = Path(args.out) if args.out else (repo_root / args.out_root / args.repo)
    export_one(args.repo, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
