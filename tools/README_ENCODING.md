# Encoding Normalization Tools

This directory contains tools for maintaining consistent UTF-8 encoding across all markdown and YAML files in the project.

## Tools

### 1. `normalize_encoding.py`

Automatically normalizes problematic Unicode characters to ASCII-safe alternatives.

**Replacements:**

- Em-dash (--) → Double hyphen (--)
- En-dash (-) → Single hyphen (-)
- Non-breaking hyphen (-) → Regular hyphen (-)
- Non-breaking space → Regular space
- Zero-width space → Removed
- BOM → Removed

**Usage:**

```bash
# Dry run (preview changes)
python tools/normalize_encoding.py --dry-run

# Normalize specific files
python tools/normalize_encoding.py README.md `../docs/PLATFORM_REFERENCE.md`

# Normalize entire directory
python tools/normalize_encoding.py docs/

# Normalize all markdown and YAML files
python tools/normalize_encoding.py .
```

### 2. `check_encoding.py`

Checks for encoding issues without modifying files. Used in CI/CD.

**Usage:**

```bash
# Check all files
python tools/check_encoding.py

# Exit code 0 = no issues
# Exit code 1 = issues found
```

### 3. `fix_broken_encoding.py`

Converts files with incorrect encoding (e.g., Windows-1251, CP1252) to UTF-8.

**Usage:**

```bash
# Dry run
python tools/fix_broken_encoding.py file.md --dry-run

# Convert files
python tools/fix_broken_encoding.py file1.md file2.md
```

## Best Practices

### Why Normalize Encoding?

1. **CommonMark Compatibility**: ASCII-safe markdown renders correctly everywhere
2. **YAML Spec**: UTF-8 without BOM is the standard
3. **Editor Compatibility**: Works in all editors (VS Code, Vim, Emacs, etc.)
4. **CI/CD**: Prevents encoding-related failures in pipelines
5. **Search**: Better full-text search indexing
6. **Copy-Paste**: No weird characters when copying code

### When to Use

- Before committing documentation changes
- After importing content from external sources
- When seeing "кракозябры" (garbled text)
- In CI/CD pre-commit hooks

### Supported Encodings (Detection)

- UTF-8 (target)
- Windows-1251 (Cyrillic)
- CP1252 (Windows Western European)
- ISO-8859-1 (Latin-1)
- CP866 (DOS Cyrillic)
- KOI8-R (Russian)

## Integration with CI/CD

The project uses GitHub Actions to automatically check encoding on every push/PR:

```yaml
# .github/workflows/docs-quality.yml
- name: Check file encoding
  run: python tools/check_encoding.py
```

If the check fails, run normalization locally and push the fix:

```bash
python tools/normalize_encoding.py .
git add .
git commit -m "fix: normalize encoding in documentation"
git push
```

## Configuration

### Markdownlint

The project uses `.markdownlint.json` to enforce consistent markdown style:

```json
{
  "default": true,
  "MD013": {
    "line_length": 120
  },
  "MD033": {
    "allowed_elements": ["antml:function_calls"]
  }
}
```

## References

- [CommonMark Spec](https://spec.commonmark.org/)
- [YAML 1.2 Spec](https://yaml.org/spec/1.2/spec.html)
- [RFC 3629: UTF-8](https://www.rfc-editor.org/rfc/rfc3629)
- [Markdownlint Rules](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md)

## Troubleshooting

### "UnicodeDecodeError: 'utf-8' codec can't decode byte"

File is not UTF-8. Use `fix_broken_encoding.py`:

```bash
python tools/fix_broken_encoding.py path/to/file.md
```

### "File already UTF-8 but contains problematic characters"

Use `normalize_encoding.py`:

```bash
python tools/normalize_encoding.py path/to/file.md
```

### "Console output shows garbled text"

This is a Windows console (CP1251) display issue. The files are correct.
Use Git Bash or WSL for proper UTF-8 display.

## Statistics (Project-Wide)

After normalization (2025-12-03):

- **Files modified**: 42
- **Total changes**: 651
- **Broken encoding files fixed**: 3 (Windows-1251 → UTF-8)
- **Em-dashes replaced**: ~400
- **Non-breaking hyphens**: ~60
- **Non-breaking spaces**: ~30

## Maintenance

Run normalization regularly:

```bash
# Weekly maintenance
python tools/normalize_encoding.py . --dry-run
# Review changes, then:
python tools/normalize_encoding.py .
```

Add to pre-commit hook:

```bash
# .git/hooks/pre-commit
#!/bin/bash
python tools/check_encoding.py || {
  echo "Encoding issues found. Run: python tools/normalize_encoding.py"
  exit 1
}
```
