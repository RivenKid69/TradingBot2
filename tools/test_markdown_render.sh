#!/bin/bash
# Test markdown rendering with pandoc
# This script tests if key markdown files render correctly

set -e

KEY_FILES=(
    "README.md"
    "ARCHITECTURE.md"
    "docs/PLATFORM_REFERENCE.md"
    "BUILD_INSTRUCTIONS.md"
    "DOCS_INDEX.md"
)

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "Error: pandoc is not installed"
    echo "Install with: sudo apt-get install pandoc"
    exit 1
fi

echo "Testing markdown files with pandoc..."
echo "======================================"

failed=0
passed=0

for file in "${KEY_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠ SKIP: $file (not found)"
        continue
    fi

    echo -n "Testing: $file ... "
    if pandoc "$file" -t html -o /dev/null 2>&1; then
        echo "✓ OK"
        ((passed++))
    else
        echo "✗ FAILED"
        ((failed++))
    fi
done

echo "======================================"
echo "Results: $passed passed, $failed failed"

if [ $failed -gt 0 ]; then
    exit 1
fi

echo "All markdown files render successfully!"
exit 0
