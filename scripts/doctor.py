#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/doctor.py
Environment self-check utility for the trading bot.

Usage:
    python scripts/doctor.py              # Run all checks
    python scripts/doctor.py --verbose    # Show detailed output
    python scripts/doctor.py --fix        # Attempt to fix issues
    python scripts/doctor.py --json       # Output JSON report

Checks performed:
    1. Python version (3.12 required)
    2. Core dependencies installed
    3. API credentials configured (existence only, not validity)
    4. Data directories exist
    5. Config files present and valid
    6. Network connectivity (optional)
    7. System resources (disk space, memory)
    8. GPU availability (for ML training)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# Constants
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

REQUIRED_PYTHON = (3, 12)

CORE_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pydantic", "pydantic"),
    ("yaml", "pyyaml"),
    ("gymnasium", "gymnasium"),
]

ML_PACKAGES = [
    "torch",
    "stable_baselines3",
    "sb3_contrib",
]

EXCHANGE_PACKAGES = [
    ("alpaca_py", "alpaca-py"),
    ("polygon", "polygon-api-client"),
    ("binance", "python-binance"),
]

# Optional packages with security/feature implications
# Tech Debt: docs/reports/TECH_DEBT_REGISTRY.md#dependency-optional-fallbacks
# Each entry: (import_name, pip_name, feature, fallback_behavior)
OPTIONAL_PACKAGES = [
    ("pyotp", "pyotp", "MFA/TOTP", "MFA verification will fail-closed (return False)"),
    ("argon2", "argon2-cffi", "Argon2id password hashing", "Falls back to bcrypt/PBKDF2"),
    ("cryptography", "cryptography", "Artifact signing", "Signature verification will fail-closed"),
    ("requests", "requests", "HTTP client for adapters", "Some adapters may not function"),
]

REQUIRED_DIRS = [
    "configs",
    "data",
    "models",
    "logs",
]

REQUIRED_CONFIGS = [
    "configs/config_train.yaml",
    "configs/config_sim.yaml",
    "configs/risk.yaml",
]

API_KEY_ENV_VARS = {
    "binance": ["BINANCE_API_KEY", "BINANCE_API_SECRET"],
    "alpaca": ["ALPACA_API_KEY", "ALPACA_API_SECRET"],
    "polygon": ["POLYGON_API_KEY"],
    "oanda": ["OANDA_API_KEY", "OANDA_ACCOUNT_ID"],
}

MIN_DISK_GB = 10  # Minimum free disk space in GB
MIN_MEMORY_GB = 8  # Minimum system memory in GB


# =============================================================================
# Check Result Types
# =============================================================================


@dataclass
class CheckResult:
    """Result of a single check."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    details: Dict[str, Any] = field(default_factory=dict)
    fix_command: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
            "fix_command": self.fix_command,
        }


@dataclass
class DoctorReport:
    """Full report from doctor checks."""

    checks: List[CheckResult] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    system_info: Dict[str, Any] = field(default_factory=dict)

    def add_check(self, result: CheckResult) -> None:
        self.checks.append(result)
        if result.passed:
            self.passed += 1
        elif result.severity == "warning":
            self.warnings += 1
        else:
            self.failed += 1

    def is_healthy(self) -> bool:
        """Return True if no errors (warnings OK)."""
        return self.failed == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthy": self.is_healthy(),
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "system_info": self.system_info,
            "checks": [c.to_dict() for c in self.checks],
        }


# =============================================================================
# Color Output
# =============================================================================


class Colors:
    """ANSI color codes."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls) -> None:
        """Disable colors (for non-TTY output)."""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.BOLD = ""
        cls.RESET = ""


# =============================================================================
# Check Functions
# =============================================================================


def check_python_version() -> CheckResult:
    """Check Python version is 3.12+."""
    version = sys.version_info[:2]
    passed = version >= REQUIRED_PYTHON

    return CheckResult(
        name="Python Version",
        passed=passed,
        message=f"Python {version[0]}.{version[1]}"
        + (" ✓" if passed else f" (need {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+)"),
        severity="error",
        details={
            "current": f"{version[0]}.{version[1]}",
            "required": f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+",
            "full_version": sys.version,
        },
    )


def check_package_installed(
    package_name: str, import_name: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if a package is installed.

    Returns:
        Tuple of (is_installed, version_string).
    """
    import_name = import_name or package_name
    try:
        module = __import__(import_name.split(".")[0])
        version = getattr(module, "__version__", "unknown")
        if callable(version):
            try:
                version = version()
            except Exception:
                version = str(version)
        if not isinstance(version, str):
            version = str(version)
        return True, version
    except ImportError:
        return False, None


def check_core_packages() -> CheckResult:
    """Check core packages are installed."""
    missing = []
    installed = []

    for import_name, pip_name in CORE_PACKAGES:
        is_installed, version = check_package_installed(import_name)
        if is_installed:
            installed.append(f"{pip_name}=={version}")
        else:
            missing.append(pip_name)

    passed = len(missing) == 0

    return CheckResult(
        name="Core Packages",
        passed=passed,
        message=f"{len(installed)}/{len(CORE_PACKAGES)} installed"
        + (f", missing: {', '.join(missing)}" if missing else ""),
        severity="error",
        details={"installed": installed, "missing": missing},
        fix_command=f"pip install {' '.join(missing)}" if missing else None,
    )


def check_ml_packages() -> CheckResult:
    """Check ML packages (PyTorch, SB3) are installed."""
    missing = []
    installed = []

    for pkg in ML_PACKAGES:
        is_installed, version = check_package_installed(pkg)
        if is_installed:
            installed.append(f"{pkg}=={version}")
        else:
            missing.append(pkg)

    passed = len(missing) == 0

    return CheckResult(
        name="ML Packages",
        passed=passed,
        message=f"{len(installed)}/{len(ML_PACKAGES)} installed"
        + (f", missing: {', '.join(missing)}" if missing else ""),
        severity="warning",  # Warning since training might not be needed
        details={"installed": installed, "missing": missing},
        fix_command='pip install -e ".[cpu]"' if missing else None,
    )


def check_exchange_packages() -> CheckResult:
    """Check exchange adapter packages are installed."""
    results = {}
    installed_count = 0

    for import_name, pkg_name in EXCHANGE_PACKAGES:
        is_installed, version = check_package_installed(import_name)
        results[pkg_name] = {"installed": is_installed, "version": version}
        if is_installed:
            installed_count += 1

    # At least one exchange package should be installed
    passed = installed_count > 0

    return CheckResult(
        name="Exchange Packages",
        passed=passed,
        message=f"{installed_count}/{len(EXCHANGE_PACKAGES)} installed",
        severity="warning",
        details={"packages": results},
    )


def check_optional_packages() -> CheckResult:
    """Check optional packages with security/feature implications.

    Optional packages that affect security or feature behavior.
    Missing packages result in documented fallback behavior.

    Tech Debt: docs/reports/TECH_DEBT_REGISTRY.md#dependency-optional-fallbacks
    """
    results = {}
    missing_security = []  # Security-critical missing packages

    for import_name, pkg_name, feature, fallback in OPTIONAL_PACKAGES:
        is_installed, version = check_package_installed(import_name)
        results[pkg_name] = {
            "installed": is_installed,
            "version": version,
            "feature": feature,
            "fallback": fallback if not is_installed else None,
        }
        # Track security-related missing packages
        if not is_installed and import_name in ("pyotp", "argon2", "cryptography"):
            missing_security.append(f"{pkg_name} ({feature})")

    installed_count = sum(1 for r in results.values() if r["installed"])

    # Warning if security packages missing, but not error
    passed = len(missing_security) == 0

    message = f"{installed_count}/{len(OPTIONAL_PACKAGES)} optional packages installed"
    if missing_security:
        message += f". Missing security: {', '.join(missing_security)}"

    return CheckResult(
        name="Optional Packages (Security/Features)",
        passed=passed,
        message=message,
        severity="warning",  # Warning, not error - fallbacks exist
        details={"packages": results, "missing_security": missing_security},
    )


def check_api_credentials() -> CheckResult:
    """Check API credentials are configured (not validity, just existence)."""
    configured = {}
    missing = {}

    for exchange, vars_list in API_KEY_ENV_VARS.items():
        env_status = {}
        all_set = True
        for var in vars_list:
            is_set = bool(os.environ.get(var))
            env_status[var] = "✓" if is_set else "✗"
            if not is_set:
                all_set = False

        if all_set:
            configured[exchange] = env_status
        else:
            missing[exchange] = env_status

    # At least one exchange should be configured
    passed = len(configured) > 0

    message = f"{len(configured)}/{len(API_KEY_ENV_VARS)} exchanges configured"
    if configured:
        message += f" ({', '.join(configured.keys())})"

    return CheckResult(
        name="API Credentials",
        passed=passed,
        message=message,
        severity="warning",
        details={"configured": configured, "missing": missing},
    )


def check_directories() -> CheckResult:
    """Check required directories exist."""
    missing = []
    present = []

    for dir_name in REQUIRED_DIRS:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists() and dir_path.is_dir():
            present.append(dir_name)
        else:
            missing.append(dir_name)

    passed = len(missing) == 0

    return CheckResult(
        name="Directories",
        passed=passed,
        message=f"{len(present)}/{len(REQUIRED_DIRS)} present"
        + (f", missing: {', '.join(missing)}" if missing else ""),
        severity="error",
        details={"present": present, "missing": missing},
        fix_command="; ".join([f"mkdir {d}" for d in missing]) if missing else None,
    )


def check_config_files() -> CheckResult:
    """Check required config files exist and are valid YAML."""
    missing = []
    invalid = []
    valid = []

    for config_path in REQUIRED_CONFIGS:
        full_path = PROJECT_ROOT / config_path
        if not full_path.exists():
            missing.append(config_path)
            continue

        # Try to parse YAML
        try:
            import yaml

            with open(full_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            valid.append(config_path)
        except Exception as e:
            invalid.append(f"{config_path}: {str(e)[:50]}")

    passed = len(missing) == 0 and len(invalid) == 0

    message = f"{len(valid)}/{len(REQUIRED_CONFIGS)} valid"
    if missing:
        message += f", missing: {len(missing)}"
    if invalid:
        message += f", invalid: {len(invalid)}"

    return CheckResult(
        name="Config Files",
        passed=passed,
        message=message,
        severity="error",
        details={"valid": valid, "missing": missing, "invalid": invalid},
    )


def check_disk_space() -> CheckResult:
    """Check available disk space."""
    try:
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)

        passed = free_gb >= MIN_DISK_GB

        return CheckResult(
            name="Disk Space",
            passed=passed,
            message=f"{free_gb:.1f} GB free / {total_gb:.1f} GB total"
            + ("" if passed else f" (need {MIN_DISK_GB} GB)"),
            severity="warning",
            details={
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_gb": round((total - free) / (1024**3), 2),
                "min_required_gb": MIN_DISK_GB,
            },
        )
    except Exception as e:
        return CheckResult(
            name="Disk Space",
            passed=False,
            message=f"Unable to check: {e}",
            severity="warning",
        )


def check_memory() -> CheckResult:
    """Check system memory."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)

        passed = total_gb >= MIN_MEMORY_GB

        return CheckResult(
            name="System Memory",
            passed=passed,
            message=f"{available_gb:.1f} GB available / {total_gb:.1f} GB total"
            + ("" if passed else f" (need {MIN_MEMORY_GB} GB)"),
            severity="warning",
            details={
                "total_gb": round(total_gb, 2),
                "available_gb": round(available_gb, 2),
                "used_percent": mem.percent,
                "min_required_gb": MIN_MEMORY_GB,
            },
        )
    except ImportError:
        return CheckResult(
            name="System Memory",
            passed=True,
            message="psutil not installed (skip)",
            severity="info",
            fix_command="pip install psutil",
        )
    except Exception as e:
        return CheckResult(
            name="System Memory",
            passed=True,
            message=f"Unable to check: {e}",
            severity="info",
        )


def check_gpu() -> CheckResult:
    """Check GPU availability for ML training."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()

        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return CheckResult(
                name="GPU",
                passed=True,
                message=f"CUDA available: {gpu_name} ({gpu_memory:.1f} GB)",
                severity="info",
                details={
                    "cuda_available": True,
                    "device_name": gpu_name,
                    "memory_gb": round(gpu_memory, 2),
                    "cuda_version": torch.version.cuda,
                },
            )
        else:
            return CheckResult(
                name="GPU",
                passed=True,  # GPU not strictly required
                message="CUDA not available (CPU only)",
                severity="info",
                details={"cuda_available": False},
            )
    except ImportError:
        return CheckResult(
            name="GPU",
            passed=True,
            message="PyTorch not installed (skip)",
            severity="info",
        )
    except Exception as e:
        return CheckResult(
            name="GPU",
            passed=True,
            message=f"Unable to check: {e}",
            severity="info",
        )


def check_network_connectivity() -> CheckResult:
    """Check network connectivity to key services."""
    import socket

    endpoints = [
        ("api.binance.com", 443),
        ("paper-api.alpaca.markets", 443),
        ("api.polygon.io", 443),
    ]

    reachable = []
    unreachable = []

    for host, port in endpoints:
        try:
            socket.setdefaulttimeout(5)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, port))
            sock.close()
            reachable.append(host)
        except (socket.timeout, socket.error):
            unreachable.append(host)

    passed = len(reachable) > 0

    return CheckResult(
        name="Network",
        passed=passed,
        message=f"{len(reachable)}/{len(endpoints)} endpoints reachable",
        severity="warning",
        details={"reachable": reachable, "unreachable": unreachable},
    )


# =============================================================================
# Main Doctor Function
# =============================================================================


def run_doctor(verbose: bool = False, skip_network: bool = False) -> DoctorReport:
    """
    Run all doctor checks.

    Args:
        verbose: Show detailed output.
        skip_network: Skip network connectivity check.

    Returns:
        DoctorReport with all check results.
    """
    report = DoctorReport()

    # Collect system info
    report.system_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "project_root": str(PROJECT_ROOT),
        "cwd": os.getcwd(),
    }

    # Define checks to run
    checks: List[Callable[[], CheckResult]] = [
        check_python_version,
        check_core_packages,
        check_ml_packages,
        check_exchange_packages,
        check_optional_packages,  # Security/feature optional packages
        check_api_credentials,
        check_directories,
        check_config_files,
        check_disk_space,
        check_memory,
        check_gpu,
    ]

    if not skip_network:
        checks.append(check_network_connectivity)

    # Run all checks
    for check_fn in checks:
        try:
            result = check_fn()
            report.add_check(result)
        except Exception as e:
            report.add_check(
                CheckResult(
                    name=check_fn.__name__.replace("check_", "").replace("_", " ").title(),
                    passed=False,
                    message=f"Check failed: {e}",
                    severity="error",
                )
            )

    return report


def print_report(report: DoctorReport, verbose: bool = False) -> None:
    """Print the doctor report to console."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}  Trading Bot Environment Check{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")

    for check in report.checks:
        if check.passed:
            status = f"{Colors.GREEN}✓ PASS{Colors.RESET}"
        elif check.severity == "warning":
            status = f"{Colors.YELLOW}⚠ WARN{Colors.RESET}"
        else:
            status = f"{Colors.RED}✗ FAIL{Colors.RESET}"

        print(f"  {status}  {check.name}: {check.message}")

        if verbose and check.details:
            for key, value in check.details.items():
                print(f"          {key}: {value}")

        if not check.passed and check.fix_command:
            print(f"          {Colors.YELLOW}Fix: {check.fix_command}{Colors.RESET}")

    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"  Passed:   {Colors.GREEN}{report.passed}{Colors.RESET}")
    print(f"  Warnings: {Colors.YELLOW}{report.warnings}{Colors.RESET}")
    print(f"  Failed:   {Colors.RED}{report.failed}{Colors.RESET}")

    if report.is_healthy():
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Environment is healthy!{Colors.RESET}")
    else:
        print(
            f"\n{Colors.RED}{Colors.BOLD}✗ Environment has issues. Please fix before running.{Colors.RESET}"
        )


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Environment self-check for trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/doctor.py              # Run all checks
    python scripts/doctor.py --verbose    # Show detailed output
    python scripts/doctor.py --json       # Output JSON report
    python scripts/doctor.py --skip-network  # Skip network checks
        """,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network connectivity check",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Show fix commands for failed checks",
    )

    args = parser.parse_args()

    # Disable colors for JSON output or non-TTY
    if args.json or not sys.stdout.isatty():
        Colors.disable()

    # Run checks
    report = run_doctor(verbose=args.verbose, skip_network=args.skip_network)

    # Output
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report, verbose=args.verbose or args.fix)

    return 0 if report.is_healthy() else 1


if __name__ == "__main__":
    sys.exit(main())
