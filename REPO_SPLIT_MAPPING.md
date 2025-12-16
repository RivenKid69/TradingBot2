# Repo Split Mapping (CCEA Variant A)

Authoritative mapping and export tooling lives in:

- `tools/repo_split/mapping.yaml`
- `tools/repo_split/MAPPING.md`
- `tools/repo_split/export.py`

Quick dry-run:

```bash
python3 tools/repo_split/export.py --repo ccea-sdk --dry-run
python3 tools/repo_split/export.py --repo ccea-agent --dry-run
python3 tools/repo_split/export.py --repo ccea-cloud --dry-run
```

