# Licensing & IP (Monorepo)

This **monorepo is proprietary** (see `LICENSE` and `pyproject.toml`).

## Open-core split (CCEA Variant A)

To provide legal clarity and make the security substrate auditable, the project is intentionally split into:

- **`ccea-sdk` (public OSS, Apache-2.0):** protocol/contracts/schemas + crypto/signing + artifact verification + portable guardrails.
- **`ccea-agent` (public OSS, Apache-2.0):** local execution agent (vault, sandbox, policy firewall, telemetry redaction).
- **`ccea-cloud` (private proprietary):** cloud/orchestration + enterprise features + all competitive trading/ML IP.

The local export mapping and tool live in `tools/repo_split/` and can produce ready-to-init repo seeds in `dist/repo-split/`.

## How to export the split

- Export all seeds: `python3 tools/repo_split/export.py --repo all --clean`
- Mapping: `tools/repo_split/mapping.yaml`

