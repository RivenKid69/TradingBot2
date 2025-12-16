# Repo Split Mapping (Exact Paths) — CCEA Variant A

This mapping is optimized for two simultaneous goals:

1) **Max IP protection** (keep RL/simulation/execution intelligence proprietary).
2) **Max EU/enterprise trust** (open-source the “infrastructure of trust”: Agent + SDK + protocol/guardrails).

## Target Repos

### 1) `ccea-sdk` (PUBLIC OSS)
Purpose: **Protocol + cryptography + guardrails + artifact verification**. This is the “CCEA trust substrate”.

**Move to `ccea-sdk`:**
- `ccea/crypto/**`
- `ccea/guardrails/**`
- `ccea/artifact/**`
- `ccea/contracts/**`
- `ccea/models/**`
- `ccea/protocol/**`
- `docs/schemas/**`
- `packages/shared/__init__.py`
- `packages/shared/contracts/**`
- `packages/shared/utils/**`
- `packages/shared/models/**`
- `packages/shared/adapters/**`

**Deliberately NOT in SDK (keep private or in agent):**
- `ccea/telemetry/**` (agent already enforces mandatory redaction; avoid duplicate surface)
- `ccea/control_plane/**`, `ccea/agent/**` (legacy/prototype)

### 2) `ccea-agent` (PUBLIC OSS)
Purpose: **Local execution agent** running in customer environment.

**Move to `ccea-agent`:**
- `packages/agent/**`
- `docs/agent/**`

**Immediate post-split fixups (required):**
- Replace imports from `packages.shared.*` / `ccea.*` with `ccea-sdk` public modules where appropriate.
- Provide a real CLI matching docs (currently docs mention `ccea-agent ...` but this monorepo runs `python -m packages.agent.daemon`).
- Fix missing `configs/agent.yaml` references across docs: it’s referenced but not present.

### 3) `ccea-cloud` (PRIVATE, proprietary)
Purpose: **Commercial Cloud control plane + all secret sauce** (training, sim, execution models, enterprise posture).

**Move to `ccea-cloud` (broad bucket):**
- Cloud stack: `packages/cloud/**`, `deploy/**`
- Compliance/enterprise docs: `docs/cloud/**`, `docs/compliance/**`, `docs/enterprise/**`, `docs/legal/**`, `docs/architecture/**`, `docs/design/**`, `docs/business/**`, `docs/security/**`, `docs/runbooks/**`, `docs/operations/**`, `docs/ui/**`
- Core IP + legacy runtime: `services/**`, `adapters/**`, `strategies/**`, `optimizers/**`, `lob/**`, `execution/**`, `wrappers/**`, `domain/**`, `api/**`, `backtest/**`, `scripts/**`, plus root-level training/backtest scripts/modules (`*.py`, `*.pyx`, etc.)

**Why broad?**
To avoid accidental IP leakage and keep the “open” repos clean. You can later refactor private code into a dedicated private “core” package if desired.

## Public Safety Guardrails

The following must **never** appear in public repos (defense-in-depth):
- RL/training: `distributional_ppo.py`, `train_model_*.py`, `optimizers/**`
- Execution/simulation: `lob/**`, `execution/**`, `service_*.py`, `script_live*.py`
- Integrations: `adapters/**`, broker private clients
- Any runtime data: `data/**`, `logs/**`, `state/**`, `artifacts/**`

## Export Helper

Use `tools/repo_split/export.py` to export only git-tracked files for each repo into `dist/repo-split/*`.
This is intended to create clean “seed” folders for new repositories (without history).

