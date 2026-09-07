"""
Analyze test coverage for PBT + Adversarial Training.

This script:
1. Counts all test functions across test files
2. Analyzes what each test covers
3. Identifies potential gaps
4. Generates a coverage report
"""

import ast
import os
from pathlib import Path
from collections import defaultdict


def analyze_test_file(filepath):
    """Analyze a test file and extract test information."""
    with open(filepath, "r") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return []

    tests = []
    current_class = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            current_class = node.name
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            # Extract docstring
            docstring = ast.get_docstring(node) or "No description"
            tests.append(
                {
                    "name": node.name,
                    "class": current_class,
                    "docstring": docstring,
                    "line": node.lineno,
                }
            )

    return tests


def main():
    """Main analysis function."""
    test_dir = Path("/home/user/AI-Powered Quantitative Research Platform/tests")

    # Find all PBT + Adversarial test files
    test_files = [
        "test_state_perturbation.py",
        "test_sa_ppo.py",
        "test_pbt_scheduler.py",
        "test_integration_pbt_adversarial.py",
        "test_training_pbt_adversarial_integration.py",
        "test_pbt_adversarial_defaults.py",
        "test_pbt_adversarial_deep_validation.py",
        "test_pbt_adversarial_real_integration.py",
    ]

    all_tests = {}
    total_count = 0
    category_counts = defaultdict(int)

    print("=" * 80)
    print("PBT + ADVERSARIAL TRAINING TEST COVERAGE ANALYSIS")
    print("=" * 80)
    print()

    for test_file in test_files:
        filepath = test_dir / test_file
        if not filepath.exists():
            print(f"⚠ Missing: {test_file}")
            continue

        tests = analyze_test_file(filepath)
        all_tests[test_file] = tests
        total_count += len(tests)

        print(f"\n📝 {test_file}")
        print(f"   Tests: {len(tests)}")

        # Categorize tests
        for test in tests:
            test_class = test["class"] or "Global"
            category_counts[test_class] += 1

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total test files: {len([f for f in test_files if (test_dir / f).exists()])}")
    print(f"Total test functions: {total_count}")
    print()

    print("Test distribution by category:")
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {category:40s}: {count:3d} tests")

    print()
    print("=" * 80)
    print("DETAILED TEST LIST")
    print("=" * 80)

    for test_file, tests in all_tests.items():
        if tests:
            print(f"\n{test_file} ({len(tests)} tests)")
            print("-" * 80)

            current_class = None
            for test in tests:
                if test["class"] != current_class:
                    current_class = test["class"]
                    if current_class:
                        print(f"\n  {current_class}:")

                indent = "    " if current_class else "  "
                print(f"{indent}{test['name']}")
                if test["docstring"] != "No description":
                    # Print first line of docstring
                    first_line = test["docstring"].split("\n")[0].strip()
                    print(f"{indent}  → {first_line}")

    print()
    print("=" * 80)
    print("COVERAGE ANALYSIS")
    print("=" * 80)

    # Analyze what's covered
    coverage_areas = {
        "State Perturbation": 0,
        "SA-PPO": 0,
        "PBT Scheduler": 0,
        "Integration": 0,
        "Defaults": 0,
        "Configuration": 0,
        "Edge Cases": 0,
        "Error Handling": 0,
        "Performance": 0,
        "Memory": 0,
        "Real Models": 0,
        "File System": 0,
        "Numerical Stability": 0,
    }

    for test_file, tests in all_tests.items():
        for test in tests:
            test_name = test["name"].lower()
            docstring = test["docstring"].lower()
            combined = test_name + " " + docstring

            if "perturbation" in combined or "fgsm" in combined or "pgd" in combined:
                coverage_areas["State Perturbation"] += 1
            if "sappo" in combined or "sa_ppo" in combined or "adversarial_ppo" in combined:
                coverage_areas["SA-PPO"] += 1
            if "pbt" in combined or "population" in combined:
                coverage_areas["PBT Scheduler"] += 1
            if "integration" in combined or "coordinator" in combined:
                coverage_areas["Integration"] += 1
            if "default" in combined:
                coverage_areas["Defaults"] += 1
            if "config" in combined or "yaml" in combined:
                coverage_areas["Configuration"] += 1
            if "edge" in combined or "boundary" in combined:
                coverage_areas["Edge Cases"] += 1
            if "error" in combined or "exception" in combined or "invalid" in combined:
                coverage_areas["Error Handling"] += 1
            if "performance" in combined or "speed" in combined or "time" in combined:
                coverage_areas["Performance"] += 1
            if "memory" in combined or "leak" in combined:
                coverage_areas["Memory"] += 1
            if "real" in combined or "actual" in combined or "gradient" in combined:
                coverage_areas["Real Models"] += 1
            if (
                "file" in combined
                or "checkpoint" in combined
                or "save" in combined
                or "load" in combined
            ):
                coverage_areas["File System"] += 1
            if "numerical" in combined or "stability" in combined or "precision" in combined:
                coverage_areas["Numerical Stability"] += 1

    print("\nCoverage by area:")
    for area, count in sorted(coverage_areas.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 2)
        print(f"  {area:25s}: {count:3d} tests {bar}")

    print()
    print("=" * 80)
    print("COMPLETENESS ASSESSMENT")
    print("=" * 80)

    completeness_score = 0
    max_score = 0

    checks = [
        ("State perturbation tests", coverage_areas["State Perturbation"] >= 20, 10),
        ("SA-PPO tests", coverage_areas["SA-PPO"] >= 15, 10),
        ("PBT scheduler tests", coverage_areas["PBT Scheduler"] >= 20, 10),
        ("Integration tests", coverage_areas["Integration"] >= 15, 10),
        ("Default settings tests", coverage_areas["Defaults"] >= 10, 10),
        ("Configuration tests", coverage_areas["Configuration"] >= 10, 10),
        ("Edge case tests", coverage_areas["Edge Cases"] >= 15, 10),
        ("Error handling tests", coverage_areas["Error Handling"] >= 10, 5),
        ("Performance tests", coverage_areas["Performance"] >= 3, 5),
        ("Memory management tests", coverage_areas["Memory"] >= 2, 5),
        ("Real model tests", coverage_areas["Real Models"] >= 10, 10),
        ("File system tests", coverage_areas["File System"] >= 5, 5),
        ("Numerical stability tests", coverage_areas["Numerical Stability"] >= 5, 5),
    ]

    for name, passed, points in checks:
        max_score += points
        if passed:
            completeness_score += points
            status = "✓"
        else:
            status = "✗"
        print(f"  {status} {name:40s} ({points} points)")

    print()
    percentage = (completeness_score / max_score) * 100
    print(f"Overall completeness: {completeness_score}/{max_score} points ({percentage:.1f}%)")

    if percentage >= 95:
        print("🎉 EXCELLENT: Comprehensive test coverage!")
    elif percentage >= 80:
        print("✓ GOOD: Test coverage is solid.")
    elif percentage >= 60:
        print("⚠ FAIR: Some areas need more testing.")
    else:
        print("❌ INSUFFICIENT: Significant testing gaps.")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
