#!/usr/bin/env python3
"""
Project Cleanup Utility

Removes build artifacts, backup files, and organizes reports.
Safe to run - will ask for confirmation before destructive operations.

Usage:
    python tools/cleanup_project.py [--dry-run] [--force]

Options:
    --dry-run: Show what would be deleted without actually deleting
    --force: Skip confirmation prompts
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple


class ProjectCleanup:
    """Clean up build artifacts and organize project files."""

    def __init__(self, root_dir: Path, dry_run: bool = False, force: bool = False):
        self.root_dir = root_dir
        self.dry_run = dry_run
        self.force = force
        self.stats = {
            "compiled_removed": 0,
            "backups_removed": 0,
            "reports_moved": 0,
        }

    def find_files_to_remove(self) -> Tuple[List[Path], List[Path]]:
        """Find files to remove.

        Returns:
            (compiled_files, backup_files)
        """
        compiled_patterns = ["*.so", "*.c"]
        backup_patterns = ["*.backup", "*.v2_backup", "*.bak"]

        compiled_files = []
        backup_files = []

        # Only check root directory and immediate subdirectories
        for pattern in compiled_patterns:
            compiled_files.extend(self.root_dir.glob(pattern))

        for pattern in backup_patterns:
            backup_files.extend(self.root_dir.glob(pattern))
            # Also check subdirectories
            backup_files.extend(self.root_dir.glob(f"*/{pattern}"))

        return compiled_files, backup_files

    def find_reports_to_move(self) -> List[Path]:
        """Find report files in root to move to docs/reports/."""
        report_patterns = [
            "*_REPORT*.md",
            "*_REPORT*.txt",
            "*_SUMMARY*.md",
            "*_SUMMARY*.txt",
            "*_FINDINGS*.md",
            "*_FINDINGS*.txt",
            "*_ANALYSIS*.md",
            "*_ANALYSIS*.txt",
            "*_INDEX.txt",
        ]

        # Exclude main documentation files
        exclude_files = {
            "README.md",
            "ARCHITECTURE.md",
            "BUILD_INSTRUCTIONS.md",
            "docs/PLATFORM_REFERENCE.md",
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            "DOCS_INDEX.md",
            "FOREX_INTEGRATION.md",
            "QUICK_START_REFERENCE.md",
        }

        # Use a set to avoid duplicates
        reports_set = set()
        for pattern in report_patterns:
            for file_path in self.root_dir.glob(pattern):
                if file_path.name not in exclude_files:
                    reports_set.add(file_path)

        return sorted(list(reports_set))

    def remove_files(self, files: List[Path], category: str) -> None:
        """Remove files safely.

        Args:
            files: List of file paths to remove
            category: Category name for stats tracking
        """
        if not files:
            print(f"[OK] No {category} files to remove")
            return

        print(f"\n{category.upper()} FILES TO REMOVE:")
        for f in files:
            rel_path = f.relative_to(self.root_dir)
            print(f"  - {rel_path}")

        if self.dry_run:
            print(f"[DRY-RUN] Would remove {len(files)} files")
            return

        if not self.force:
            response = input(f"\nRemove {len(files)} files? [y/N]: ")
            if response.lower() != "y":
                print("Skipped.")
                return

        for f in files:
            try:
                f.unlink()
                self.stats[f"{category}_removed"] += 1
            except Exception as e:
                print(f"[ERROR] Error removing {f}: {e}")

        print(f"[OK] Removed {self.stats[f'{category}_removed']} files")

    def move_reports(self, reports: List[Path]) -> None:
        """Move report files to docs/reports/."""
        if not reports:
            print("[OK] No reports to move")
            return

        reports_dir = self.root_dir / "docs" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        print("\nREPORTS TO MOVE:")
        for f in reports:
            print(f"  - {f.name} -> docs/reports/")

        if self.dry_run:
            print(f"[DRY-RUN] Would move {len(reports)} reports")
            return

        if not self.force:
            response = input(f"\nMove {len(reports)} reports? [y/N]: ")
            if response.lower() != "y":
                print("Skipped.")
                return

        for f in reports:
            dest = reports_dir / f.name
            try:
                if dest.exists():
                    print(f"[SKIP] {f.name} (already exists in destination)")
                    continue
                shutil.move(str(f), str(dest))
                self.stats["reports_moved"] += 1
            except Exception as e:
                print(f"[ERROR] Error moving {f.name}: {e}")

        print(f"[OK] Moved {self.stats['reports_moved']} reports")

    def run(self) -> None:
        """Run cleanup process."""
        print("=" * 60)
        print("PROJECT CLEANUP UTILITY")
        print("=" * 60)

        if self.dry_run:
            print("[DRY-RUN] No files will be modified\n")

        # Find files
        print("Scanning project...")
        compiled_files, backup_files = self.find_files_to_remove()
        reports = self.find_reports_to_move()

        # Summary
        print("\nFOUND:")
        print(f"  - {len(compiled_files)} compiled artifacts (*.so, *.c)")
        print(f"  - {len(backup_files)} backup files (*.backup, *.bak)")
        print(f"  - {len(reports)} reports to organize")

        if not any([compiled_files, backup_files, reports]):
            print("\n[OK] Project is already clean!")
            return

        # Execute cleanup
        self.remove_files(compiled_files, "compiled")
        self.remove_files(backup_files, "backups")
        self.move_reports(reports)

        # Final summary
        print("\n" + "=" * 60)
        print("CLEANUP SUMMARY")
        print("=" * 60)
        print(f"  Compiled artifacts removed: {self.stats['compiled_removed']}")
        print(f"  Backup files removed: {self.stats['backups_removed']}")
        print(f"  Reports moved: {self.stats['reports_moved']}")
        print()

        if not self.dry_run:
            print("[OK] Cleanup complete!")
            print("\nNOTE: Compiled files will be regenerated on next build.")
            print("      Run 'python setup.py build_ext --inplace' if needed.")


def main():
    parser = argparse.ArgumentParser(
        description="Clean up build artifacts and organize project files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts",
    )

    args = parser.parse_args()

    # Get project root (parent of tools/ directory)
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    cleanup = ProjectCleanup(root_dir, dry_run=args.dry_run, force=args.force)
    cleanup.run()


if __name__ == "__main__":
    main()
