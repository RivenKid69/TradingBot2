#!/usr/bin/env python3
"""
Comprehensive Test Runner for UPGD + PBT + Twin Critics + Variance Scaling

This script runs all integration and validation tests for the advanced
optimization stack and produces a detailed report.
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


class TestRunner:
    """Manages test execution and reporting."""

    def __init__(self):
        self.results = []
        self.start_time = None
        self.total_time = 0

    def run_test_suite(self, test_file: str, description: str) -> Tuple[bool, str]:
        """Run a single test suite and return success status and output."""
        print(f"\n{'=' * 80}")
        print(f"Running: {description}")
        print(f"File: {test_file}")
        print(f"{'=' * 80}\n")

        start = time.time()

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "-x"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per suite
            )

            duration = time.time() - start

            success = result.returncode == 0

            output = result.stdout + "\n" + result.stderr

            print(output)

            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"\n{status} ({duration:.2f}s)")

            return success, output

        except subprocess.TimeoutExpired:
            duration = time.time() - start
            print(f"\n✗ TIMEOUT ({duration:.2f}s)")
            return False, "Test suite timed out"

        except Exception as e:
            duration = time.time() - start
            print(f"\n✗ ERROR: {e} ({duration:.2f}s)")
            return False, str(e)

    def run_all_tests(self):
        """Run all test suites."""
        self.start_time = time.time()

        test_suites = [
            (
                "tests/test_upgd_deep_validation.py",
                "UPGD Deep Validation - Core optimizer mechanics",
            ),
            (
                "tests/test_upgd_pbt_twin_critics_variance_integration.py",
                "Full Integration - UPGD + PBT + Twin Critics + VGS",
            ),
            ("tests/test_upgd_integration.py", "UPGD Integration - PPO integration tests"),
            ("tests/test_variance_gradient_scaler.py", "Variance Gradient Scaler - Unit tests"),
            ("tests/test_vgs_integration.py", "VGS Integration - Integration with training"),
            ("tests/test_pbt_scheduler.py", "PBT Scheduler - Population-based training"),
            ("tests/test_twin_critics.py", "Twin Critics - Basic functionality"),
            (
                "tests/test_twin_critics_integration.py",
                "Twin Critics Integration - Full integration",
            ),
        ]

        for test_file, description in test_suites:
            # Check if test file exists
            if not Path(test_file).exists():
                print(f"\n⚠ SKIPPED: {test_file} (file not found)")
                self.results.append(
                    {
                        "file": test_file,
                        "description": description,
                        "status": "SKIPPED",
                        "output": "File not found",
                    }
                )
                continue

            success, output = self.run_test_suite(test_file, description)

            self.results.append(
                {
                    "file": test_file,
                    "description": description,
                    "status": "PASSED" if success else "FAILED",
                    "output": output,
                }
            )

        self.total_time = time.time() - self.start_time

    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.results if r["status"] == "PASSED")
        failed = sum(1 for r in self.results if r["status"] == "FAILED")
        skipped = sum(1 for r in self.results if r["status"] == "SKIPPED")
        total = len(self.results)

        print(f"\nTotal Suites: {total}")
        print(f"  ✓ Passed:   {passed}")
        print(f"  ✗ Failed:   {failed}")
        print(f"  ⚠ Skipped:  {skipped}")
        print(f"\nTotal Time: {self.total_time:.2f}s")

        if failed > 0:
            print("\n" + "=" * 80)
            print("FAILED TEST SUITES:")
            print("=" * 80)
            for result in self.results:
                if result["status"] == "FAILED":
                    print(f"\n✗ {result['description']}")
                    print(f"  File: {result['file']}")
                    print(f"\n  Last 50 lines of output:")
                    print("  " + "-" * 76)
                    lines = result["output"].split("\n")
                    for line in lines[-50:]:
                        print(f"  {line}")

        print("\n" + "=" * 80)

        if failed == 0 and skipped == 0:
            print("🎉 ALL TESTS PASSED! 🎉")
            print("=" * 80)
            return 0
        elif failed == 0:
            print("⚠ ALL RUN TESTS PASSED (some skipped)")
            print("=" * 80)
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            print("=" * 80)
            return 1

    def save_report(self, filename: str = "test_report.txt"):
        """Save detailed test report to file."""
        with open(filename, "w") as f:
            f.write("COMPREHENSIVE TEST REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Time: {self.total_time:.2f}s\n\n")

            for result in self.results:
                f.write("=" * 80 + "\n")
                f.write(f"Suite: {result['description']}\n")
                f.write(f"File: {result['file']}\n")
                f.write(f"Status: {result['status']}\n")
                f.write("-" * 80 + "\n")
                f.write(result["output"])
                f.write("\n\n")

        print(f"\nDetailed report saved to: {filename}")


def main():
    """Main entry point."""
    print("=" * 80)
    print("UPGD + PBT + Twin Critics + VGS - Comprehensive Test Suite")
    print("=" * 80)
    print("\nThis will run all integration and validation tests.")
    print("Expected duration: 5-15 minutes depending on system.\n")

    runner = TestRunner()
    runner.run_all_tests()
    runner.save_report("upgd_test_report.txt")
    exit_code = runner.print_summary()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
