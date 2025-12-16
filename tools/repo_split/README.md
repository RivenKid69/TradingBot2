# Repo Split Toolkit (CCEA Variant A)

This directory contains the **authoritative mapping** and helper scripts to split this monorepo into:

- `ccea-sdk` (public OSS): protocol/contracts/schemas + crypto/signing + artifact verification + guardrails
- `ccea-agent` (public OSS): local execution agent (vault, policy firewall, approvals, telemetry redaction, sandbox)
- `ccea-cloud` (private proprietary): everything else (RL/training, backtest/sim, execution models, UI/IDE, enterprise governance)

## Files

- `mapping.yaml`: source-of-truth include/exclude patterns per target repo
- `export.py`: copies **git-tracked files** into `dist/repo-split/<repo>` according to the mapping (no history)
- `MAPPING.md`: human-readable rationale + post-split fixups checklist

## Usage

Dry-run (shows counts and a sample):

```bash
python3 tools/repo_split/export.py --repo ccea-sdk --dry-run
python3 tools/repo_split/export.py --repo ccea-agent --dry-run
python3 tools/repo_split/export.py --repo ccea-cloud --dry-run
```

Export to local folders (git-tracked files only):

```bash
python3 tools/repo_split/export.py --repo ccea-sdk --out dist/repo-split/ccea-sdk --clean
python3 tools/repo_split/export.py --repo ccea-agent --out dist/repo-split/ccea-agent --clean
python3 tools/repo_split/export.py --repo ccea-cloud --out dist/repo-split/ccea-cloud --clean
python3 tools/repo_split/export.py --repo all --clean
```

Notes:
- Repo scaffolding (`LICENSE`, `pyproject.toml`, basic CI, etc.) is applied from `tools/repo_split/templates/<repo>/`.
- This tool intentionally excludes runtime artifacts (`data/`, `logs/`, `.venv/`, `__pycache__/`, etc.).
