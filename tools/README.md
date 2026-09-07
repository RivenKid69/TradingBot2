# Tools Directory

Utility scripts for maintaining and managing the CustodiaCloud repository.

## 🧹 Project Maintenance

### cleanup_project.py

**Purpose**: Remove build artifacts, backup files, and organize reports.

**What it cleans:**

- ✅ Compiled Cython artifacts (`*.so`, `*.c`)
- ✅ Backup files (`*.backup`, `*.v2_backup`, `*.bak`)
- ✅ Organizes reports from root → `docs/reports/`

**Usage:**

```bash
# Dry-run (preview what would be cleaned)
python tools/cleanup_project.py --dry-run

# Interactive cleanup (asks for confirmation)
python tools/cleanup_project.py

# Force cleanup (skip confirmations)
python tools/cleanup_project.py --force
```

**Safe to run**: Won't delete important files. Compiled artifacts will be regenerated on next build.

### Encoding Tools

See [README_ENCODING.md](README_ENCODING.md) for encoding normalization tools:

- `normalize_encoding.py` - Normalize Unicode to ASCII-safe
- `check_encoding.py` - Check for encoding issues (CI/CD)
- `fix_broken_encoding.py` - Convert non-UTF-8 files to UTF-8

## 📊 Testing & Validation

### check_feature_parity.py

Verify feature consistency between online and offline environments.

```bash
python tools/check_feature_parity.py
```

### analyze_test_coverage.py

Analyze test coverage across the codebase.

```bash
python tools/analyze_test_coverage.py
```

### check_imports.py

Check for circular imports and dependency violations.

```bash
python tools/check_import.py
```

## 🔍 Debugging & Analysis

### debug_vgs_load.py

Debug VGS (Variance Gradient Scaler) loading issues.

```bash
python tools/debug_vgs_load.py
```

### analyze_upgd_vgs_issues.py

Analyze UPGD optimizer and VGS integration issues.

```bash
python tools/analyze_upgd_vgs_issues.py
```

### check_drift.py

Check for data drift in features.

```bash
python tools/check_drift.py
```

## 🧪 Test Utilities

### run_comprehensive_upgd_tests.py

Run comprehensive UPGD optimizer tests.

```bash
python tools/run_comprehensive_upgd_tests.py
```

### quick_test_twin_critics_default.py

Quick test for Twin Critics default configuration.

```bash
python tools/quick_test_twin_critics_default.py
```

## 📋 Best Practices

### Regular Maintenance

**Weekly:**

```bash
# 1. Clean up artifacts
python tools/cleanup_project.py --dry-run
python tools/cleanup_project.py

# 2. Check encoding
python tools/check_encoding.py

# 3. Check feature parity
python tools/check_feature_parity.py
```

**Before Committing:**

```bash
# 1. Normalize encoding
python tools/normalize_encoding.py . --dry-run
python tools/normalize_encoding.py .

# 2. Check imports
python tools/check_imports.py
```

**After Major Changes:**

```bash
# 1. Run comprehensive tests
pytest tests/

# 2. Check coverage
python tools/analyze_test_coverage.py

# 3. Verify fixes
python tools/comprehensive_target_fix_verification.py
```

### Safe Cleanup Workflow

1. **Dry-run first**: Always run with `--dry-run` to preview changes
2. **Review output**: Check what will be deleted/moved
3. **Confirm changes**: Run without `--dry-run` when ready
4. **Test build**: Run `python setup.py build_ext --inplace` after cleanup
5. **Run tests**: Ensure nothing broke with `pytest tests/`

### CI/CD Integration

The project uses GitHub Actions for automated checks:

```yaml
# .github/workflows/maintenance.yml
- name: Check encoding
  run: python tools/check_encoding.py

- name: Check feature parity
  run: python tools/check_feature_parity.py

- name: Check imports
  run: python tools/check_imports.py
```

## 🔧 Development Tools

### audit_feature_indices.py

Audit feature indices for consistency.

### precise_index_count.py

Count and verify feature indices.

### deep_verification_63.py

Deep verification of 63-feature pipeline.

### manual_audit_63.py

Manual audit of 63-feature configuration.

## 📝 Notes

- All tools are safe to run in dry-run mode
- Most tools support `--help` for detailed usage
- Backup files are automatically ignored by git (see `.gitignore`)
- Compiled artifacts are regenerated automatically on build

## 🐛 Troubleshooting

### "No module named 'XXX'"

Install dependencies:

```bash
pip install -r requirements.txt
```

### "Permission denied"

On Windows, run with proper permissions or use Git Bash:

```bash
python tools/cleanup_project.py
```

### "File already exists"

When moving reports, if a file already exists in `docs/reports/`, it will be skipped.
Check manually if you need to replace it.

### After cleanup, build fails

Rebuild Cython extensions:

```bash
python setup.py build_ext --inplace
```

## 📚 Related Documentation

- [BUILD_INSTRUCTIONS.md](../BUILD_INSTRUCTIONS.md) - Build setup
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Project architecture
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [docs/pipeline.md](../docs/pipeline.md) - Feature pipeline

## 🔄 Maintenance Schedule

| Task | Frequency | Command |
|------|-----------|---------|
| Cleanup artifacts | Weekly | `python tools/cleanup_project.py` |
| Check encoding | Before commit | `python tools/check_encoding.py` |
| Check feature parity | After feature changes | `python tools/check_feature_parity.py` |
| Analyze coverage | After major changes | `python tools/analyze_test_coverage.py` |

---

**Last Updated**: 2025-12-03
**Maintained by**: CustodiaCloud maintainers
