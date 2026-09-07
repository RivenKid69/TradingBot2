# P0-A / P0-D / P0-E closure (2026-07-16)

> Closes the remaining P0 blockers from
> [PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md](PLATFORM_FULL_GAP_ANALYSIS_2026-07-15.md):
> broken imports (P0-A), missing agent config (P0-D), and full model-signature
> coverage on the daemon load path (P0-E).
> Tests: `tests/test_agent_config.py` (6), `tests/test_agent_model_signature.py`
> (10), `tests/test_backward_compatibility_facade.py` (29 pass / 9 skip) + live smoke.

---

## P0-A — broken imports

Three modules failed to import (confirmed by reproduction). All fixed at the
correct layer.

| Module | Root cause | Fix |
|--------|-----------|-----|
| `packages.shared.models` | Re-exported `TimeFrame`/`OrderSide`/`PositionSide` from `core_models`, which never had them; also `packages.shared.contracts` didn't export `MarketSnapshot`/`ExecutionMode`/`RiskLevel`/`ChangeClass` | Added canonical `TimeFrame` enum to `core_models.py`; sourced `OrderSide`/`PositionSide` from their real home `core_futures`; exported the 4 missing names from `packages/shared/contracts/__init__.py` |
| `adapters.theta_data` | `options.py` imported `Bar` from `adapters.models` (not there) | Import `Bar` from `core_models` — the convention every other adapter already uses |
| `services.compliance` | Deprecated shim hard-imported `services.archive.mifid_financial_entity`, a module intentionally removed (platform is an **ICT Provider**, not a MiFID Financial Entity — only `dora_financial_entity` ships) | Graceful degrade: the archive block is now `try/except ImportError` → `ARCHIVE_AVAILABLE = False` + a `RuntimeWarning`; CORE risk-controls + INTEGRATION toolkit still re-export; `__all__` drops absent names so `import *` stays honest. Stale archive tests skip on `not ARCHIVE_AVAILABLE` |

`TimeFrame` values are exchange-style (`1m`…`1w`) with a `.seconds` helper.

## P0-D — `configs/agent.yaml`

The documented launch `python -m packages.agent.daemon --config configs/agent.yaml`
had **no config file**, and the config builder passed **stale field names** to
`DegradedModeConfig` (`cloud_unreachable_threshold_seconds` /
`data_feed_stale_threshold_seconds`) → it crashed on any real config.

- Created configs/agent.yaml: a documented reference
  config covering every section the builder reads (`agent`, `cloud`,
  `components.{kill_switch,time_sync,degraded_mode,telemetry,sandbox,keychain}`),
  with safe defaults and CCEA notes (secrets stay in Agent zone, standalone by
  default).
- Fixed `packages/agent/daemon/__main__.py::build_daemon_config`:
  - `DegradedModeConfig` now gets the real fields `cloud_timeout_seconds` /
    `data_stale_threshold_seconds` (old key names accepted as aliases).
  - Kill-switch pct thresholds are converted to `Decimal` (exact `0.30`, no
    float/Decimal mixing in the hard-cap comparisons).
- Fixed the launch command in `PLATFORM_REFERENCE.md` (`packages.agent.daemon`, not
  `…daemon.agentd` — `agentd.py` has no `__main__`).

Verified: `python -m packages.agent.daemon --config configs/agent.yaml --dry-run`
→ "config validation successful"; `--dump-config` → exit 0.

## P0-E — model-signature gate on **all** daemon load paths

The Ed25519 gate (`services/model_signature_gate.py`) was wired into the RL
inference loader (`service_rl_inference`) and the XS rebalance path, but the
**daemon's own** artifact-activation path (`RunController.initialize` →
`_init_live_runner`) only verified the SHA-256 **digest** (integrity), never the
**signature** (authenticity). An SB3 `.zip` checkpoint is pickle: deserializing
an unsigned/tampered artifact is arbitrary code execution in the process that
holds broker keys (CCEA design doc §15: "Artifact Signature Verification:
REQUIRED").

- New `packages/agent/daemon/model_gate.py`:
  - `find_model_files(path)` — locate checkpoints (`.zip/.pt/.pth/.pkl/.ckpt/.onnx/.safetensors`).
  - `verify_artifact_models(extracted_path, *, live, policy, registry, context)`
    — run each checkpoint through the **same** `verify_model_artifact` the RL
    loader uses. Enforce (default for LIVE) → `ModelSignatureError` on the first
    failure, **before any pickle is read**. No checkpoint (code-only strategy) →
    `[]` (manifest digest + sandbox controls still apply), logged.
- `RunController` (`agentd.py`): new `RunControllerConfig.require_model_signature`
  (default `True`) + `model_signature_policy`; `initialize()` calls
  `_verify_model_signature()` before `_init_live_runner()`. A failure in enforce
  (LIVE) makes `initialize()` return `False` — **fail-closed, the run never
  starts**.

## MVP

`GET /api/agent/daemon/config` validates `configs/agent.yaml` through the daemon's
own builder and reports the live-safety posture: config validity, components
present, kill-switch thresholds, and `model_signature.{enforced_on_load,
live_policy}`. Surfaced in the Pro **Security → Верификатор подписей** tab as a
card **«Agent-демон: конфиг и гейт подписи модели»** (VALID/INVALID badge +
enforce status + the documented launch command).

## Files

- `core_models.py` — `TimeFrame` enum.
- `packages/shared/models/__init__.py`, `packages/shared/contracts/__init__.py` — re-export fixes.
- `adapters/theta_data/options.py` — `Bar` from `core_models`.
- `services/compliance/__init__.py` — graceful archive degrade + `ARCHIVE_AVAILABLE`.
- `configs/agent.yaml` — new daemon reference config.
- `packages/agent/daemon/__main__.py` — builder field-name + Decimal fixes.
- `packages/agent/daemon/model_gate.py` — new signature helper.
- `packages/agent/daemon/agentd.py` — `RunController` signature enforcement.
- `app.py` — `/api/agent/daemon/config`.
- `index.html` — Pro Security posture card.
- Tests: `tests/test_agent_config.py`, `tests/test_agent_model_signature.py`,
  `tests/test_backward_compatibility_facade.py` (guarded archive tests).
