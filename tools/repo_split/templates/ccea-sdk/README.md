# CCEA SDK (`ccea-sdk`)

Open-source trust layer for the CCEA architecture: protocol contracts/schemas, crypto primitives, artifact manifests + signature verification, and CI guardrails.

## Scope

Included:
- Protocol models + JSON Schemas
- Cryptographic keys + signing/verification primitives
- Artifact manifest/SBOM helpers + verifier
- Guardrails for protocol/schema integrity

Explicitly **not included** (kept proprietary in the private Cloud/Core repos):
- Trading strategies and signals
- Backtesting/simulation engines
- Reinforcement learning/training code
- Broker/exchange execution logic

## Development

- Install: `pip install -e .`
- Run schema guardrail: `python -m ccea.guardrails.schema_check docs/schemas/`
- Run contract drift check: `python -m ccea.contracts.validation`

## Trademark

See `TRADEMARK.md`.

