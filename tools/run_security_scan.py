#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_security_scan.py
Run security scans locally before committing.

Usage:
    python tools/run_security_scan.py              # Run all scans
    python tools/run_security_scan.py --bandit     # Run only bandit
    python tools/run_security_scan.py --semgrep    # Run only semgrep
    python tools/run_security_scan.py --quick      # Quick scan (high severity only)

Requirements:
    pip install bandit
    # For semgrep: https://semgrep.dev/docs/getting-started/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
BANDIT_CONFIG = PROJECT_ROOT / ".bandit"
SEMGREP_CONFIG = PROJECT_ROOT / ".semgrep.yml"


# =============================================================================
# Color Output
# =============================================================================


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_header(msg: str) -> None:
    """Print a header message."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")


def print_success(msg: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_warning(msg: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")


def print_error(msg: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


# =============================================================================
# Bandit Scanner
# =============================================================================


def check_bandit_installed() -> bool:
    """Check if bandit is installed."""
    try:
        result = subprocess.run(
            ["bandit", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_bandit(quick: bool = False) -> Tuple[int, List[Dict]]:
    """
    Run bandit security scan.

    Args:
        quick: If True, only report high severity issues.

    Returns:
        Tuple of (exit_code, list_of_issues).
    """
    print_header("Running Bandit Security Scan")

    if not check_bandit_installed():
        print_error("Bandit not installed. Run: pip install bandit")
        return 1, []

    cmd = [
        "bandit",
        "-r",
        str(PROJECT_ROOT),
        "-f",
        "json",
    ]

    if BANDIT_CONFIG.exists():
        cmd.extend(["-c", str(BANDIT_CONFIG)])

    if quick:
        cmd.extend(["-ll"])  # Only high confidence, high severity

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        issues = []
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                issues = data.get("results", [])
            except json.JSONDecodeError:
                pass

        # Count by severity
        high_count = len([i for i in issues if i.get("issue_severity") == "HIGH"])
        medium_count = len([i for i in issues if i.get("issue_severity") == "MEDIUM"])
        low_count = len([i for i in issues if i.get("issue_severity") == "LOW"])

        print(f"Issues found: {len(issues)} total")
        print(f"  - High:   {high_count}")
        print(f"  - Medium: {medium_count}")
        print(f"  - Low:    {low_count}")

        if high_count > 0:
            print_error(f"Found {high_count} HIGH severity issues!")
            for issue in issues:
                if issue.get("issue_severity") == "HIGH":
                    print(
                        f"\n  {Colors.RED}[HIGH]{Colors.RESET} {issue.get('filename')}:{issue.get('line_number')}"
                    )
                    print(f"    {issue.get('issue_text')}")
                    print(f"    Test ID: {issue.get('test_id')}")
        elif medium_count > 0:
            print_warning(f"Found {medium_count} MEDIUM severity issues")
        else:
            print_success("No high/medium severity issues found")

        return (1 if high_count > 0 else 0), issues

    except Exception as e:
        print_error(f"Error running bandit: {e}")
        return 1, []


# =============================================================================
# Semgrep Scanner
# =============================================================================


def check_semgrep_installed() -> bool:
    """Check if semgrep is installed."""
    try:
        result = subprocess.run(
            ["semgrep", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def run_semgrep(quick: bool = False) -> Tuple[int, List[Dict]]:
    """
    Run semgrep security scan.

    Args:
        quick: If True, only use local rules.

    Returns:
        Tuple of (exit_code, list_of_issues).
    """
    print_header("Running Semgrep Security Scan")

    if not check_semgrep_installed():
        print_warning("Semgrep not installed. Skipping.")
        print("Install: pip install semgrep  OR  https://semgrep.dev/docs/getting-started/")
        return 0, []

    all_issues = []
    exit_code = 0

    # Run local rules
    if SEMGREP_CONFIG.exists():
        print("Running local rules...")
        cmd = [
            "semgrep",
            "scan",
            "--config",
            str(SEMGREP_CONFIG),
            "--json",
            str(PROJECT_ROOT),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    issues = data.get("results", [])
                    all_issues.extend(issues)

                    errors = [i for i in issues if i.get("extra", {}).get("severity") == "ERROR"]
                    if errors:
                        exit_code = 1
                        print_error(f"Found {len(errors)} ERROR severity issues from local rules")
                    else:
                        print_success(f"Local rules: {len(issues)} issues (no errors)")
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print_error(f"Error running semgrep local rules: {e}")

    # Run community rules (unless quick mode)
    if not quick:
        print("\nRunning community rules (p/python, p/secrets)...")
        cmd = [
            "semgrep",
            "scan",
            "--config",
            "p/python",
            "--config",
            "p/secrets",
            "--json",
            str(PROJECT_ROOT),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=300,  # 5 minute timeout
            )

            if result.stdout:
                try:
                    data = json.loads(result.stdout)
                    issues = data.get("results", [])
                    all_issues.extend(issues)

                    errors = [i for i in issues if i.get("extra", {}).get("severity") == "ERROR"]
                    if errors:
                        exit_code = 1
                        print_error(
                            f"Found {len(errors)} ERROR severity issues from community rules"
                        )
                    else:
                        print_success(f"Community rules: {len(issues)} issues (no errors)")
                except json.JSONDecodeError:
                    pass
        except subprocess.TimeoutExpired:
            print_warning("Community rules scan timed out (5 min limit)")
        except Exception as e:
            print_error(f"Error running semgrep community rules: {e}")

    # Summary
    print(f"\nTotal Semgrep issues: {len(all_issues)}")

    return exit_code, all_issues


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run security scans locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tools/run_security_scan.py              # Run all scans
    python tools/run_security_scan.py --bandit     # Run only bandit
    python tools/run_security_scan.py --semgrep    # Run only semgrep
    python tools/run_security_scan.py --quick      # Quick scan (high severity only)
        """,
    )
    parser.add_argument(
        "--bandit",
        action="store_true",
        help="Run only bandit scan",
    )
    parser.add_argument(
        "--semgrep",
        action="store_true",
        help="Run only semgrep scan",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick scan (high severity only, local rules only)",
    )

    args = parser.parse_args()

    # If no specific scanner selected, run all
    run_all = not args.bandit and not args.semgrep

    exit_code = 0

    if run_all or args.bandit:
        code, _ = run_bandit(quick=args.quick)
        exit_code = max(exit_code, code)

    if run_all or args.semgrep:
        code, _ = run_semgrep(quick=args.quick)
        exit_code = max(exit_code, code)

    # Final summary
    print_header("Security Scan Complete")

    if exit_code == 0:
        print_success("No critical security issues found!")
    else:
        print_error("Security issues found. Please fix before committing.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
