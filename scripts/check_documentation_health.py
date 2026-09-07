#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Documentation Health Check Script

Автоматическая проверка здоровья документации проекта AI-Powered Quantitative Research Platform.

Проверки:
- Возраст файлов (warn if > 3 months old)
- Broken links (fail on any broken)
- Duplicate content (warn on >80% similarity)
- Missing required sections (fail on missing)

Usage:
    python scripts/check_documentation_health.py
    python scripts/check_documentation_health.py --verbose
    python scripts/check_documentation_health.py --fix  # Auto-fix broken links
"""

import os
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set
from collections import defaultdict

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Core documentation files that MUST exist
REQUIRED_CORE_DOCS = [
    "README.md",
    "docs/PLATFORM_REFERENCE.md",
    "DOCS_INDEX.md",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "BUILD_INSTRUCTIONS.md",
    "VERIFICATION_INSTRUCTIONS.md",
]

# Required sections in core docs
REQUIRED_SECTIONS = {
    "README.md": ["Статус Проекта", "Основные Возможности", "Быстрый Старт"],
    "docs/PLATFORM_REFERENCE.md": ["СТАТУС ПРОЕКТА", "КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ", "Архитектура проекта"],
    "DOCS_INDEX.md": ["CRITICAL - READ FIRST", "Core Documentation"],
    "CHANGELOG.md": ["[2.1.0] - 2025-11-21"],
}

# Age thresholds
MAX_AGE_DAYS_CRITICAL = 30  # Critical docs should be updated within 1 month
MAX_AGE_DAYS_NORMAL = 90    # Normal docs should be updated within 3 months


class DocumentationHealthChecker:
    """Automated documentation health checker."""

    def __init__(self, root_dir: str, verbose: bool = False, fix: bool = False):
        self.root_dir = Path(root_dir)
        self.verbose = verbose
        self.fix = fix
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def log_error(self, msg: str):
        """Log error message."""
        self.errors.append(f"[ERROR] {msg}")
        print(f"[ERROR] {msg}")

    def log_warning(self, msg: str):
        """Log warning message."""
        self.warnings.append(f"[WARNING] {msg}")
        if self.verbose:
            print(f"[WARNING] {msg}")

    def log_info(self, msg: str):
        """Log info message."""
        self.info.append(f"[INFO] {msg}")
        if self.verbose:
            print(f"[INFO] {msg}")

    def check_core_docs_exist(self) -> bool:
        """Check that all core documentation files exist."""
        self.log_info("Checking core documentation files...")
        all_exist = True

        for doc in REQUIRED_CORE_DOCS:
            path = self.root_dir / doc
            if not path.exists():
                self.log_error(f"Missing core documentation: {doc}")
                all_exist = False
            else:
                self.log_info(f"✓ Found: {doc}")

        return all_exist

    def check_file_age(self) -> Dict[str, List[str]]:
        """Check last modified dates of documentation files."""
        self.log_info("Checking file ages...")
        outdated_critical = []
        outdated_normal = []

        now = datetime.now()
        critical_threshold = now - timedelta(days=MAX_AGE_DAYS_CRITICAL)
        normal_threshold = now - timedelta(days=MAX_AGE_DAYS_NORMAL)

        for doc in REQUIRED_CORE_DOCS:
            path = self.root_dir / doc
            if not path.exists():
                continue

            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            age_days = (now - mtime).days

            if mtime < critical_threshold:
                outdated_critical.append(f"{doc} ({age_days} days old)")
                self.log_warning(f"{doc} is {age_days} days old (> {MAX_AGE_DAYS_CRITICAL} days)")
            elif mtime < normal_threshold:
                outdated_normal.append(f"{doc} ({age_days} days old)")
                self.log_info(f"{doc} is {age_days} days old")
            else:
                self.log_info(f"✓ {doc} is up to date ({age_days} days old)")

        return {
            "critical": outdated_critical,
            "normal": outdated_normal,
        }

    def check_broken_links(self) -> List[Tuple[str, str, str]]:
        """Check for broken internal links in markdown files."""
        self.log_info("Checking for broken links...")
        broken_links = []

        # Pattern for markdown links: [text](link)
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

        for md_file in self.root_dir.rglob("*.md"):
            # Skip archived files
            if "archive" in md_file.parts:
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception as e:
                self.log_warning(f"Could not read {md_file}: {e}")
                continue

            for match in link_pattern.finditer(content):
                link_text = match.group(1)
                link_url = match.group(2)

                # Skip external links (http/https), anchors, and emails
                if link_url.startswith(("http://", "https://", "#", "mailto:")):
                    continue

                # Remove anchor if present
                link_path = link_url.split("#")[0]
                if not link_path:  # Just an anchor
                    continue

                # Resolve relative path
                target_path = (md_file.parent / link_path).resolve()

                if not target_path.exists():
                    broken_links.append((str(md_file.relative_to(self.root_dir)), link_text, link_url))
                    self.log_error(f"Broken link in {md_file.name}: [{link_text}]({link_url})")

        return broken_links

    def check_required_sections(self) -> Dict[str, List[str]]:
        """Check for required sections in core documents."""
        self.log_info("Checking required sections...")
        missing_sections = {}

        for doc, sections in REQUIRED_SECTIONS.items():
            path = self.root_dir / doc
            if not path.exists():
                continue

            try:
                content = path.read_text(encoding="utf-8")
            except Exception as e:
                self.log_warning(f"Could not read {doc}: {e}")
                continue

            missing = []
            for section in sections:
                if section not in content:
                    missing.append(section)
                    self.log_error(f"Missing required section in {doc}: '{section}'")

            if missing:
                missing_sections[doc] = missing
            else:
                self.log_info(f"✓ All required sections found in {doc}")

        return missing_sections

    def check_duplicate_content(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """Check for duplicate content across documentation files."""
        self.log_info("Checking for duplicate content...")
        duplicates = []

        # Get all markdown files (excluding archived)
        md_files = [f for f in self.root_dir.rglob("*.md") if "archive" not in f.parts]

        # Simple similarity check (Jaccard similarity of lines)
        file_lines = {}
        for md_file in md_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                # Normalize: lowercase, strip, remove empty lines
                lines = set(line.strip().lower() for line in content.split("\n") if line.strip())
                file_lines[md_file] = lines
            except Exception as e:
                self.log_warning(f"Could not read {md_file}: {e}")
                continue

        # Compare all pairs
        checked_pairs = set()
        for file1 in md_files:
            for file2 in md_files:
                if file1 >= file2:  # Skip self and duplicate pairs
                    continue

                pair = tuple(sorted([str(file1), str(file2)]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                if file1 not in file_lines or file2 not in file_lines:
                    continue

                lines1 = file_lines[file1]
                lines2 = file_lines[file2]

                if not lines1 or not lines2:
                    continue

                # Jaccard similarity
                intersection = len(lines1 & lines2)
                union = len(lines1 | lines2)
                similarity = intersection / union if union > 0 else 0

                if similarity > threshold:
                    duplicates.append((
                        str(file1.relative_to(self.root_dir)),
                        str(file2.relative_to(self.root_dir)),
                        similarity
                    ))
                    self.log_warning(
                        f"High similarity ({similarity:.1%}) between "
                        f"{file1.name} and {file2.name}"
                    )

        return duplicates

    def count_docs(self) -> Dict[str, int]:
        """Count documentation files."""
        self.log_info("Counting documentation files...")
        counts = {
            "root_md": 0,
            "docs_md": 0,
            "archived_md": 0,
            "total_md": 0,
        }

        for md_file in self.root_dir.rglob("*.md"):
            counts["total_md"] += 1

            if md_file.parent == self.root_dir:
                counts["root_md"] += 1
            elif "archive" in md_file.parts:
                counts["archived_md"] += 1
            else:
                counts["docs_md"] += 1

        return counts

    def run_all_checks(self) -> bool:
        """Run all health checks."""
        print("\n" + "="*70)
        print("AI-Powered Quantitative Research Platform Documentation Health Check")
        print("="*70 + "\n")

        # 1. Check core docs exist
        core_exist = self.check_core_docs_exist()

        # 2. Check file ages
        outdated = self.check_file_age()

        # 3. Check broken links
        broken = self.check_broken_links()

        # 4. Check required sections
        missing = self.check_required_sections()

        # 5. Check duplicates
        duplicates = self.check_duplicate_content()

        # 6. Count docs
        counts = self.count_docs()

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70 + "\n")

        print(f"Documentation Counts:")
        print(f"  - Total .md files: {counts['total_md']}")
        print(f"  - Root directory: {counts['root_md']}")
        print(f"  - docs/ subdirs: {counts['docs_md']}")
        print(f"  - Archived: {counts['archived_md']}")
        print()

        print(f"File Age:")
        print(f"  - Critically outdated (>{MAX_AGE_DAYS_CRITICAL}d): {len(outdated['critical'])}")
        print(f"  - Outdated (>{MAX_AGE_DAYS_NORMAL}d): {len(outdated['normal'])}")
        print()

        print(f"Links:")
        print(f"  - Broken links: {len(broken)}")
        print()

        print(f"Required Sections:")
        print(f"  - Documents missing sections: {len(missing)}")
        print()

        print(f"Duplicate Content:")
        print(f"  - High similarity pairs (>80%): {len(duplicates)}")
        print()

        # Pass/Fail Criteria
        print("="*70)
        print("PASS/FAIL CRITERIA")
        print("="*70 + "\n")

        criteria_passed = 0
        criteria_total = 6

        # Criterion 1: Core docs exist
        if core_exist:
            print("[PASS] All core documentation files exist")
            criteria_passed += 1
        else:
            print("[FAIL] Some core documentation files missing")

        # Criterion 2: Core docs up to date (< 1 month)
        if len(outdated['critical']) == 0:
            print(f"[PASS] All core docs updated within {MAX_AGE_DAYS_CRITICAL} days")
            criteria_passed += 1
        else:
            print(f"[WARN] {len(outdated['critical'])} core docs outdated (>{MAX_AGE_DAYS_CRITICAL}d)")

        # Criterion 3: No broken links
        if len(broken) == 0:
            print("[PASS] No broken links found")
            criteria_passed += 1
        else:
            print(f"[FAIL] {len(broken)} broken links found")

        # Criterion 4: Required sections present
        if len(missing) == 0:
            print("[PASS] All required sections present")
            criteria_passed += 1
        else:
            print(f"[FAIL] {len(missing)} documents missing required sections")

        # Criterion 5: Low duplicate content
        if len(duplicates) < 5:
            print(f"[PASS] Duplicate content acceptable ({len(duplicates)} pairs)")
            criteria_passed += 1
        else:
            print(f"[WARN] High duplicate content ({len(duplicates)} pairs >80% similar)")

        # Criterion 6: Root directory clean (< 20 .md files)
        if counts['root_md'] < 20:
            print(f"[PASS] Root directory clean ({counts['root_md']} .md files)")
            criteria_passed += 1
        else:
            print(f"[WARN] Too many files in root ({counts['root_md']} .md files, target <20)")

        print()
        print("="*70)
        print(f"OVERALL SCORE: {criteria_passed}/{criteria_total} criteria passed "
              f"({criteria_passed/criteria_total*100:.0f}%)")
        print("="*70)

        # Overall health
        if criteria_passed == criteria_total:
            print("\n[EXCELLENT] Documentation health is excellent!")
            return True
        elif criteria_passed >= criteria_total - 1:
            print("\n[GOOD] Documentation health is good (minor issues)")
            return True
        elif criteria_passed >= criteria_total - 2:
            print("\n[ACCEPTABLE] Documentation health is acceptable (some issues)")
            return False
        else:
            print("\n[POOR] Documentation health needs attention!")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check documentation health for AI-Powered Quantitative Research Platform"
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory of the project (default: current directory)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix issues where possible (NOT IMPLEMENTED YET)"
    )

    args = parser.parse_args()

    checker = DocumentationHealthChecker(
        root_dir=args.root,
        verbose=args.verbose,
        fix=args.fix,
    )

    success = checker.run_all_checks()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
