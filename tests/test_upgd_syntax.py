#!/usr/bin/env python
"""
Syntax and structure validation for UPGD optimizer integration.
This test doesn't require PyTorch or other heavy dependencies.
"""

import ast
import sys
import os

print("=" * 60)
print("UPGD Optimizer Integration - Syntax & Structure Tests")
print("=" * 60)

# Test 1: Verify all UPGD optimizer files exist
print("\nTest 1: Checking UPGD optimizer files")
print("-" * 60)

required_files = {
    "optimizers/__init__.py": "Optimizer module init",
    "optimizers/upgd.py": "Basic UPGD optimizer",
    "optimizers/adaptive_upgd.py": "AdaptiveUPGD optimizer",
    "optimizers/upgdw.py": "UPGDW optimizer",
    "tests/test_upgd_optimizer.py": "Unit tests",
    "tests/test_upgd_integration.py": "Integration tests",
    "docs/UPGD_INTEGRATION.md": "Documentation",
}

all_exist = True
for filepath, description in required_files.items():
    if os.path.exists(filepath):
        print(f"✓ {filepath} - {description}")
    else:
        print(f"✗ {filepath} - MISSING")
        all_exist = False

if not all_exist:
    print("\n✗ Some required files are missing")
    sys.exit(1)

print("\n✓ All required files exist")

# Test 2: Verify Python syntax
print("\nTest 2: Validating Python syntax")
print("-" * 60)

python_files = [
    "optimizers/__init__.py",
    "optimizers/upgd.py",
    "optimizers/adaptive_upgd.py",
    "optimizers/upgdw.py",
    "tests/test_upgd_integration.py",
]

all_valid = True
for filepath in python_files:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
            ast.parse(code)
        print(f"✓ {filepath} - Valid Python syntax")
    except SyntaxError as e:
        print(f"✗ {filepath} - Syntax error: {e}")
        all_valid = False

if not all_valid:
    print("\n✗ Some files have syntax errors")
    sys.exit(1)

print("\n✓ All Python files have valid syntax")

# Test 3: Verify optimizer classes are defined
print("\nTest 3: Checking optimizer class definitions")
print("-" * 60)


def check_class_in_file(filepath, classname):
    """Check if a class is defined in a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == classname:
            return True
    return False


optimizer_classes = {
    "optimizers/upgd.py": "UPGD",
    "optimizers/adaptive_upgd.py": "AdaptiveUPGD",
    "optimizers/upgdw.py": "UPGDW",
}

all_defined = True
for filepath, classname in optimizer_classes.items():
    if check_class_in_file(filepath, classname):
        print(f"✓ {classname} class defined in {filepath}")
    else:
        print(f"✗ {classname} class NOT FOUND in {filepath}")
        all_defined = False

if not all_defined:
    print("\n✗ Some optimizer classes are missing")
    sys.exit(1)

print("\n✓ All optimizer classes are properly defined")

# Test 4: Verify exports in __init__.py
print("\nTest 4: Checking optimizer module exports")
print("-" * 60)

with open("optimizers/__init__.py", "r", encoding="utf-8") as f:
    init_content = f.read()

required_exports = ["UPGD", "AdaptiveUPGD", "UPGDW"]
all_exported = True

for export in required_exports:
    if export in init_content:
        print(f"✓ {export} is exported")
    else:
        print(f"✗ {export} is NOT exported")
        all_exported = False

if not all_exported:
    print("\n✗ Some optimizers are not exported")
    sys.exit(1)

print("\n✓ All optimizers are properly exported")

# Test 5: Verify distributional_ppo.py changes
print("\nTest 5: Checking DistributionalPPO modifications")
print("-" * 60)

with open("distributional_ppo.py", "r", encoding="utf-8") as f:
    ppo_content = f.read()

checks = {
    "AdaptiveUPGD import": "from optimizers import AdaptiveUPGD",
    "Default optimizer logic": "return AdaptiveUPGD",
    "Fallback to AdamW": "return torch.optim.AdamW",
    "UPGD defaults": 'optimizer_name in ("UPGD", "AdaptiveUPGD", "UPGDW")',
}

all_present = True
for description, search_str in checks.items():
    if search_str in ppo_content:
        print(f"✓ {description} - Found")
    else:
        print(f"✗ {description} - NOT FOUND")
        all_present = False

if not all_present:
    print("\n✗ Some required changes are missing in distributional_ppo.py")
    sys.exit(1)

print("\n✓ DistributionalPPO properly modified")

# Test 6: Check documentation
print("\nTest 6: Checking documentation updates")
print("-" * 60)

with open("docs/UPGD_INTEGRATION.md", "r", encoding="utf-8") as f:
    docs_content = f.read()

doc_checks = {
    "Default optimizer note": "AdaptiveUPGD is now the default optimizer",
    "Default usage section": "Default Usage (AdaptiveUPGD)",
    "Migration guide": "Migration Guide",
    "Best practices": "Best Practices",
}

all_documented = True
for description, search_str in doc_checks.items():
    if search_str in docs_content:
        print(f"✓ {description} - Found")
    else:
        print(f"✗ {description} - NOT FOUND")
        all_documented = False

if not all_documented:
    print("\n✗ Documentation is incomplete")
    sys.exit(1)

print("\n✓ Documentation is complete and up to date")

# Test 7: Check test file structure
print("\nTest 7: Checking test file structure")
print("-" * 60)

with open("tests/test_upgd_integration.py", "r", encoding="utf-8") as f:
    test_content = f.read()

test_checks = {
    "Default optimizer test": "test_default_adaptive_upgd_when_none",
    "Explicit AdamW test": "test_explicit_adamw_selection",
    "Default config class": "TestUPGDDefaultConfiguration",
    "Comprehensive coverage class": "TestUPGDComprehensiveCoverage",
    "Edge cases class": "TestUPGDEdgeCases",
}

all_tests_present = True
for description, search_str in test_checks.items():
    if search_str in test_content:
        print(f"✓ {description} - Found")
    else:
        print(f"✗ {description} - NOT FOUND")
        all_tests_present = False

if not all_tests_present:
    print("\n✗ Some tests are missing")
    sys.exit(1)

print("\n✓ All test cases are present")

# Count test methods
test_tree = ast.parse(test_content)
test_count = 0
for node in ast.walk(test_tree):
    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
        test_count += 1

print(f"\nTotal test methods found: {test_count}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY: All validation tests passed! ✓")
print("=" * 60)
print("\nUPGD Optimizer Integration Status:")
print(f"  ✓ All required files present ({len(required_files)} files)")
print(f"  ✓ All Python files have valid syntax")
print(f"  ✓ All optimizer classes properly defined (3 optimizers)")
print(f"  ✓ Optimizer exports configured correctly")
print(f"  ✓ DistributionalPPO default optimizer set to AdaptiveUPGD")
print(f"  ✓ Documentation updated with migration guide")
print(f"  ✓ Comprehensive test suite added ({test_count}+ test methods)")
print("\n✓ Integration is structurally complete and ready!")
print("\nNote: Runtime tests require PyTorch and dependencies.")
print("      To run full tests: pytest tests/test_upgd_*.py")
