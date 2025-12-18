# Licensing & IP (Monorepo)

This **monorepo is proprietary** (see `LICENSE` and `pyproject.toml`).

## Open-core split (CCEA Variant A) — optional future path

To provide legal clarity and make the security substrate auditable, a future repo split can be considered:

- **`ccea-sdk` (candidate public repo; license TBD):** protocol/contracts/schemas + crypto/signing + artifact verification + portable guardrails.
- **`ccea-agent` (candidate public repo; license TBD):** local execution agent (vault, sandbox, policy firewall, telemetry redaction).
- **`ccea-cloud` (private proprietary):** cloud/orchestration + enterprise features + all competitive trading/ML IP.

The local export mapping and tool live in `tools/repo_split/` and can produce ready-to-init repo seeds in `dist/repo-split/` (for future use).

## How to export the split

- Export all seeds: `python3 tools/repo_split/export.py --repo all --clean`
- Mapping: `tools/repo_split/mapping.yaml`
