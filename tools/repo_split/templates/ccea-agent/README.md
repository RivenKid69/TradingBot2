# CCEA Agent (`ccea-agent`)

Open-source local execution agent for the CCEA architecture. The agent runs on a user-controlled host and enforces local safety constraints (vault, sandboxing, policy firewall, hard caps, telemetry redaction).

## Scope

Included:
- Agent daemon runtime (local lifecycle loop)
- Local Vault (secrets at rest)
- Telemetry buffering + redaction
- Risk controls (policy firewall, hard caps, preflight)

Explicitly **not included** (kept proprietary in the private Cloud/Core repos):
- Trading strategies and trade-decision logic
- Training pipelines / RL code
- Cloud orchestration and enterprise features

## Quickstart (dev)

- Install: `pip install -e .`
- Copy config example: `cp configs/examples/agent.yaml ~/.ccea/agent.yaml`
- Run daemon (foreground): `ccea-agentd --config ~/.ccea/agent.yaml --foreground`

## Dependency

`ccea-agent` depends on `ccea-sdk` for protocol/contracts/crypto verification.
