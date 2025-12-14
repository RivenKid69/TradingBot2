# CCEA Master Remediation Plan (single doc for AI agent)

Issues source: `Список проблем и несогсасованностей.txt` (expected=67, actual=67)

Hard invariants (Design Doc): `Design Doc CCEA Cloud.txt:64`, `Design Doc CCEA Cloud.txt:66`, `Design Doc CCEA Cloud.txt:68`, `Design Doc CCEA Cloud.txt:78`, `Design Doc CCEA Cloud.txt:1025`

## Phases
- P0: truth + CI safety rails (legal/docs/traceability/guardrails/protocol/deps/windows CI)
- P1: working Cloud↔Agent lifecycle (enroll/heartbeat/poll/ack/approval/result + reconciliation/idempotency)
- P2: enterprise pack + signed updates + evidence pack + cloud research job isolation/anti-abuse

## Gates
- Gate P0→P1: CI green (Ubuntu+Windows), docs-quality green, guardrails enforced fail-closed, protocol consistent+enforced, deps pinned/locked, legal/marketing aligned.
- Gate P1→P2: E2E lifecycle green, boundary enforced, no duplicate orders after retries/restarts, safe-halt on uncertainty.
- Exit P2: on-prem pack + signed evidence pack + signed updates+rollback protection + cloud research isolation/anti-abuse.

## Work Items Catalog (authoritative)
```yaml
work_items:
  WI-DEPS-01:
    phase: P0
    goal: |-
      Make required runtime deps explicit and reproducible (pin+lock+CI).
    touch:
      - pyproject.toml
      - requirements-*.lock.txt
      - requirements-dev.txt
      - packages/agent/vault/local_vault.py:30
      - ccea/crypto/keys.py:19
      - packages/cloud/control_plane/database.py:29
      - packages/cloud/control_plane/dependencies.py:21
    steps:
      - Add/pin cryptography (agent vault/signing).
      - Add/pin asyncpg or change cloud DB default for dev/test (never silently missing).
      - Add/pin PyJWT (or replace with explicit JWT lib) and lock version.
      - Update lockfiles; ensure CI installs locked deps.
    acceptance:
      - .github/workflows/build-and-test.yml (Ubuntu+Windows) passes dependency install.
      - .github/workflows/security-sast.yml dependency-audit stays green.
    standards:
      - SLSA/SBOM basics
      - OWASP supply-chain hygiene (general practice)
    effort: S
  WI-AGENT-01:
    phase: P0
    goal: |-
      Unblock Windows CI by making sandbox import/behavior Windows-safe.
    touch:
      - packages/agent/daemon/sandbox.py:18
    steps:
      - Guard POSIX-only imports (resource) behind platform checks.
      - Implement Windows path (Job Objects/psutil) OR make sandbox Linux-only and skip tests on Windows with explicit marker.
      - Document behavior in agent docs (link from docs/agent/* once created).
    acceptance:
      - .github/workflows/build-and-test.yml windows-latest passes pytest import stage.
    standards:
      - Operational reliability (cross-platform CI)
    effort: M
  WI-AGENT-02:
    phase: P0
    goal: |-
      Fix SQLite lifecycle to avoid Windows lockups; ensure clean shutdown.
    touch:
      - packages/agent/daemon/telemetry_buffer.py:246
      - packages/agent/daemon/telemetry_buffer.py:400
    steps:
      - Replace sqlite connect context misuse with explicit close()/contextlib.closing.
      - Ensure background flusher thread stops and DB connections are closed deterministically.
      - Add regression test for Windows locking (best-effort) and run in CI.
    acceptance:
      - tests/ccea/phase5 telemetry buffer tests pass on windows-latest.
      - No sqlite “database is locked” flake in CI.
    standards:
      - Reliability; safe-degraded prerequisites
    effort: M
  WI-CI-02:
    phase: P0
    goal: |-
      Make guardrails actually usable: fix current failures and fail-open gaps.
    touch:
      - packages/cloud/control_plane/tests/test_telemetry.py:45
      - ccea/guardrails/intent_prohibition.py:512
      - ccea/guardrails/intent_prohibition.py:515
      - ccea/guardrails/cloud_allowlist.py:371
    steps:
      - Remove prohibited intent-like fields from cloud telemetry test fixtures (or mark fixtures excluded) so boundary guardrail remains strict.
      - Make CLI output ASCII-safe in non-UTF8 Windows consoles.
      - Fix cloud_allowlist discovery so modules_checked>0; add hard fail if files_checked>0 but modules_checked==0.
    acceptance:
      - Guardrail suite passes locally and in CI; “0 checked => FAIL” enforced.
    standards:
      - Design Doc CI guardrails: Design Doc CCEA Cloud.txt:1025
    effort: M
  WI-CI-01:
    phase: P0
    goal: |-
      Enforce CCEA guardrails in CI and pre-commit (fail-closed).
    touch:
      - .pre-commit-config.yaml
      - .github/workflows/build-and-test.yml
      - ccea/guardrails/*
    steps:
      - Add pre-commit local hooks for schema/protocol/import/intent/cloud_allowlist/build_artifact checks.
      - Add CI steps/jobs that run these guardrails and fail merge on violation.
    acceptance:
      - pre-commit run -a passes
      - CI fails on injected prohibited command/field/import.
    standards:
      - Design Doc CCEA Cloud.txt:1025
      - OWASP ASVS (build integrity concept)
    effort: M
  WI-PROTOCOL-01:
    phase: P0
    goal: |-
      Make protocol self-consistent: schema version negotiation and constants align and are enforced.
    touch:
      - docs/schemas/protocol_messages.schema.json
      - ccea/models/protocol.py
      - ccea/protocol/schema_versioning.py
      - ccea/__init__.py
    steps:
      - Pick ONE negotiation mechanism: min_supported/max_supported (Design Doc).
      - Update schema + pydantic models to match; remove/replace supported_schema_versions.
      - Add CI check: ccea/__init__.py version constants match schema metadata.
      - Enforce negotiation before accepting/processing any command batch.
    acceptance:
      - schema_check and protocol_check are green; negotiation tests cover incompatible versions.
    standards:
      - Robustness/compatibility; Design Doc CCEA Cloud.txt:615
    effort: M
  WI-PROTOCOL-02:
    phase: P0
    goal: |-
      Eliminate allowlist drift for command types (REQUEST_RESUME_RUN).
    touch:
      - packages/cloud/control_plane/commands.py:40
      - packages/cloud/control_plane/boundary.py:47
      - docs/schemas/protocol_messages.schema.json:273
      - ccea/models/protocol.py:93
    steps:
      - Default action: delete REQUEST_RESUME_RUN if not in Design Doc allowlist.
      - If needed: add it to schema + models + boundary allowlists + guardrails + tests, with explicit security review note.
    acceptance:
      - protocol allowlist check ensures parity across schema/models/boundary.
    standards:
      - Least privilege / minimize surface area
    effort: S
  WI-CLOUD-01:
    phase: P0
    goal: |-
      Fail-closed Cloud boundary in runtime: enum allowlist + boundary validator everywhere.
    touch:
      - packages/cloud/control_plane/routers/commands.py:111
      - packages/cloud/control_plane/boundary.py
      - packages/cloud/control_plane/routers/config_blobs.py:360
    steps:
      - Replace free-form command_type string with enum allowlist.
      - Apply CloudBoundaryValidator on command create/send and config blob writes.
      - Add tests: reject unknown commands; reject order-like payloads; reject secrets in blobs (paired with WI-CLOUD-05).
    acceptance:
      - API rejects unknown command_type and any order-like payload; guardrails detect regressions.
    standards:
      - OWASP ASVS (input validation)
      - Design Doc “no order-like payloads”
    effort: M
  WI-TRACE-01:
    phase: P0
    goal: |-
      Make Design Doc verifiable: snapshot + SHA in CI.
    touch:
      - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt
      - docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.md:6
    steps:
      - Add snapshot file required by plan.
      - Recompute SHA256 and update recorded value.
      - Add CI check that recomputes SHA and compares.
    acceptance:
      - CI fails if snapshot SHA mismatches recorded SHA.
    standards:
      - Provenance hygiene (documented architecture version)
    effort: S
  WI-TRACE-02:
    phase: P0
    goal: |-
      Make traceability truthful: DONE requires real artifact+check.
    touch:
      - docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md
    steps:
      - For each PLANNED vs claimed DONE mismatch: either implement and link CI proof OR downgrade plan status.
      - Add a mechanical check (lint/script) that forbids DONE with empty Artifact/Check fields.
    acceptance:
      - Traceability matrix has no DONE without a concrete CI/test artifact reference.
    standards:
      - SOC2/ISO27001 evidence practice (auditability)
    effort: M
  WI-LEGAL-01:
    phase: P0
    goal: |-
      Remove legal/marketing contradictions to CCEA boundary.
    touch:
      - docs/legal/TERMS_OF_SERVICE.md
      - docs/legal/PRIVACY_POLICY.md
      - docs/legal/DPA_TEMPLATE.md
      - docs/legal/AUP.md
      - docs/BUSINESS_PLAN_EU_VISA.md
      - docs/INVESTOR_BRIEF.md
      - docs/ENTERPRISE_ADOPTION_RISK_MITIGATION.md
      - docs/business/IP_PROTECTION_STRATEGY.md
      - docs/DORA_OPERATIONAL_RESILIENCE_PLAN.md
    steps:
      - Rewrite ToS/Privacy/DPA to reflect: keys in Agent only; cloud never executes or sends orders.
      - Add AUP aligned to cloud compute abuse controls (ties into P2 cloud isolation).
      - Fix compliance/ops docs claiming key storage/broker registration.
      - Add phrase-guard checks (ban “we trade for you”, “cloud auto-execution”, “keys stored”).
    acceptance:
      - docs-quality CI green; phrase guard green; no conflicting claims remain.
    standards:
      - GDPR minimization/retention (Design Doc)
      - DORA/NIS2 posture consistency (if referenced)
    effort: M
  WI-DOCS-01:
    phase: P0
    goal: |-
      Create required CCEA docs structure for cloud/agent/runbooks.
    touch:
      - docs/CCEA_OVERVIEW.md
      - docs/cloud/*
      - docs/agent/*
      - docs/runbooks/*
    steps:
      - Add docs/CCEA_OVERVIEW.md: boundary, threat model, legal posture, product modes.
      - Add docs/cloud/*: control plane API, builder/registry, governance/privacy/residency, research isolation.
      - Add docs/agent/*: install/upgrade, local vault, approvals, policies/hard caps, degraded modes, recovery.
      - Add docs/runbooks/*: incident/kill-switch, revoke/rotation, safe-degraded, recovery.
    acceptance:
      - docs-quality CI green; README links to overview; docs match Design Doc invariants.
    standards:
      - SOC2-style runbooks and change control practices
    effort: M
  WI-DOCS-02:
    phase: P0/P1
    goal: |-
      Remove script_live.py as recommended path; move legacy to archive.
    touch:
      - README.md
      - ARCHITECTURE.md
      - docs/* (multiple)
      - docs/archive/*
    steps:
      - Rewrite docs to describe CCEA workflow (Agent daemon + cloud control plane) instead of script_live.py.
      - Mark legacy entrypoints as archived; ensure no “live commands” via cloud are suggested.
    acceptance:
      - docs-quality CI green; grep finds no recommended script_live path (except archive).
    standards:
      - Misrepresentation risk control (Design Doc marketing section)
    effort: M
  WI-DEDRIFT-01:
    phase: P0
    goal: |-
      Remove duplicate Agent/Control Plane stacks; one canonical implementation each.
    touch:
      - ccea/agent/*
      - packages/agent/*
      - ccea/control_plane/*
      - packages/cloud/control_plane/*
    steps:
      - Select canonical stacks (recommended: packages/agent/* and packages/cloud/control_plane/*).
      - Deprecate/remove the non-canonical stacks or reduce them to thin compatibility shims.
      - Update docs and import boundaries accordingly.
    acceptance:
      - Import boundary checks enforce single stack usage; docs reference only canonical entrypoints.
    standards:
      - Attack surface reduction; operational clarity
    effort: L
  WI-CONTRACTS-01:
    phase: P0/P1
    goal: |-
      Eliminate enum/state-machine drift between protocol models and cloud DB.
    touch:
      - ccea/models/protocol.py
      - packages/cloud/control_plane/models.py
      - docs/schemas/*
    steps:
      - Make schema the source of truth for enums/state machines.
      - Generate or centralize shared contracts; map DB stored values explicitly to protocol enums.
      - Add contract tests: schema ↔ models ↔ DB mapping.
    acceptance:
      - Contract tests prevent drift; schema and runtime share the same states/values.
    standards:
      - Robustness; change control
    effort: L
  WI-CLOUD-04:
    phase: P1
    goal: |-
      Introduce migrations and enforce tenant isolation via RLS.
    touch:
      - packages/cloud/control_plane/models.py:1330
      - alembic/*
    steps:
      - Add Alembic
      - Create migrations
      - Enable RLS + policies in migrations
      - Add tenant isolation tests
    acceptance:
      - Migrations run in CI; RLS verified (negative tests).
    standards:
      - Multi-tenant isolation best practice (SOC2/ISO style)
    effort: L
  WI-CLOUD-02:
    phase: P1
    goal: |-
      Implement agent-auth lifecycle endpoints and delivery semantics.
    touch:
      - packages/cloud/control_plane/routers/auth.py
      - packages/cloud/control_plane/routers/commands.py
      - packages/cloud/control_plane/routers/*
    steps:
      - Implement AgentDep auth for heartbeat/poll/ack/approval/result endpoints.
      - Implement long-poll command retrieval and persistence.
      - Return accurate pending_commands counts; update last_seen.
      - Pick one transport (DB+poll baseline; MQ only if justified).
    acceptance:
      - E2E tests cover enroll/heartbeat/poll/ack/approval/result; idempotency verified.
    standards:
      - Design Doc protocol section; OWASP ASVS (authn/input validation)
    effort: L
  WI-CLOUD-03:
    phase: P1
    goal: |-
      Remove duplicate command layer (in-memory vs DB).
    touch:
      - packages/cloud/control_plane/commands.py
      - packages/cloud/control_plane/routers/commands.py
    steps:
      - Pick DB-backed command store with long-poll (baseline).
      - Delete/retire in-memory dispatcher or DB duplication.
    acceptance:
      - Only one command dispatch path remains; tests use it.
    standards:
      - Reliability and auditability
    effort: M
  WI-CLOUD-05:
    phase: P1
    goal: |-
      Config blob validation and DLP/secret scanning (cloud never stores secrets).
    touch:
      - packages/cloud/control_plane/routers/config_blobs.py:360
    steps:
      - Validate by schema per config_type
      - Secret/DLP scan before storing
      - Reject secrets / forbidden patterns
    acceptance:
      - Tests: secrets rejected; schema enforced; no bypass in routers/services.
    standards:
      - OWASP ASVS (input validation)
      - GDPR minimization
    effort: L
  WI-CLOUD-06:
    phase: P1
    goal: |-
      Real cloud artifact builder (signed + SBOM + provenance).
    touch:
      - packages/cloud/builder/artifact_builder.py
      - ccea/artifact/*
    steps:
      - Replace placeholder builder
      - Ensure signature required
      - Generate/attach SBOM
      - Record provenance metadata
    acceptance:
      - CI blocks unsigned artifacts; agent rejects unsigned; SBOM artifact exists.
    standards:
      - SLSA, SBOM (CycloneDX/SPDX), signing/provenance
    effort: L
  WI-AUTH-01:
    phase: P0/P1
    goal: |-
      Production-grade auth: password hashing + session revocation + rate limiting.
    touch:
      - packages/cloud/control_plane/routers/auth.py
      - packages/cloud/control_plane/routers/users.py
      - packages/cloud/control_plane/dependencies.py
    steps:
      - Replace SHA256 with Argon2id/bcrypt
      - Add password policy
      - Add rate limits/lockout
      - Implement JWT revocation (jti blocklist or refresh rotation)
    acceptance:
      - Unit/integration tests; security-sast green; logout actually revokes/invalidates tokens.
    standards:
      - OWASP ASVS
      - NIST 800-63B
    effort: M
  WI-AGENT-03:
    phase: P1
    goal: |-
      Replace agent “connected=True” placeholder with real CloudClient.
    touch:
      - packages/agent/daemon/agentd.py:731
    steps:
      - Implement enroll/heartbeat/poll/ack/approval/result
      - Use outbound-only transport
      - Sign/auth messages
    acceptance:
      - E2E lifecycle test passes; degraded mode reports real connectivity.
    standards:
      - Design Doc protocol: outbound-only + auth
    effort: L
  WI-AGENT-04:
    phase: P1
    goal: |-
      Ensure all TRADING_IMPACTING changes require local approval by default.
    touch:
      - packages/agent/approval/*
      - packages/agent/daemon/agentd.py
    steps:
      - Wire approval into start/upgrade/update_config
      - Record evidence_hash
      - Allow auto-approve only via local policy
    acceptance:
      - Tests prove cloud cannot bypass approval; approval diff shown/recorded.
    standards:
      - Design Doc CCEA Cloud.txt:78
    effort: L
  WI-AGENT-06:
    phase: P1
    goal: |-
      Deterministic client_order_id + persistent dedup (no duplicate orders on retry).
    touch:
      - packages/agent/execution/engine.py
    steps:
      - Replace counter-based ID with deterministic scheme
      - Persist dedup state (journal/sqlite)
      - Add restart/retry tests
    acceptance:
      - Idempotency tests prove no duplicate order submissions.
    standards:
      - Design Doc “no duplicate orders due to retries”
    effort: L
  WI-AGENT-05:
    phase: P1
    goal: |-
      Reconciliation wired and safe: reconcile on start/periodic; safe halt on uncertainty.
    touch:
      - packages/agent/runner/live.py
      - packages/agent/reconciliation/reconciler.py
    steps:
      - Call reconciler from runner
      - Replace reconciler stub
      - Implement safe-halt path
      - Add tests
    acceptance:
      - Reconcile tests: restart path halts safely if uncertain.
    standards:
      - Design Doc CCEA Cloud.txt:581
    effort: L
  WI-BUILD-01:
    phase: P1
    goal: |-
      Separate cloud/agent distributions and verify cloud artifact contents in CI.
    touch:
      - docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md:98
      - Makefile
      - pyproject.toml
      - .github/workflows/build-and-test.yml
    steps:
      - Produce separate cloud and agent artifacts
      - Run build_artifact_check on produced cloud artifact
      - Fail if live/broker code present
    acceptance:
      - CI has artifact-contents-check and it fails on intentional violations.
    standards:
      - Design Doc CI guardrails: no trading libs in cloud
    effort: L
  WI-CLOUD-RESEARCH-01:
    phase: P2
    goal: |-
      Cloud research job isolation/anti-abuse (sandbox/quotas/egress allowlist/abuse detection).
    touch:
      - packages/cloud/research/__init__.py:13
      - packages/cloud/research/*
    steps:
      - Sandbox runner (container/VM)
      - CPU/RAM/time quotas
      - Egress allowlist deny-by-default
      - Abuse detection (mining/scanning/botnet)
    acceptance:
      - tests/ccea/phase10 cover isolation and abuse controls.
    standards:
      - OWASP cloud isolation best practice (general)
      - Design Doc CCEA Cloud.txt:927
    effort: L
  WI-ENTERPRISE-01:
    phase: P2
    goal: |-
      On-prem pack (docker-compose/Helm) + docs.
    touch:
      - docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md:316
      - deploy/*
    steps:
      - Add compose/helm manifests
      - Add docs and air-gapped notes if needed
    acceptance:
      - onprem smoke deploy checklist/test passes.
    standards:
      - Enterprise readiness: vendor pack
    effort: L
  WI-ENTERPRISE-02:
    phase: P2
    goal: |-
      Evidence pack exporter (signed): digests/signatures/SBOM + approvals/commands/halt logs.
    touch:
      - docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md:68
      - packages/cloud/enterprise/*
    steps:
      - Export required evidence set
      - Redact sensitive data
      - Sign the evidence pack
    acceptance:
      - Evidence pack test: content complete + signature valid.
    standards:
      - SOC2 evidence practice
      - Design Doc CCEA Cloud.txt:946
    effort: L
  WI-ENTERPRISE-03:
    phase: P2
    goal: |-
      Signed agent updates + staged rollout + rollback protection.
    touch:
      - docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md:22
      - packages/cloud/enterprise/*
    steps:
      - Implement signed update metadata
      - Stage rollout controls
      - Rollback/freeze protection
    acceptance:
      - Update security tests: rollback attack blocked; unsigned update rejected.
    standards:
      - SLSA + TUF-like update metadata (recommended)
      - Design Doc CCEA Cloud.txt:917
    effort: L
  WI-VAULT-01:
    phase: P2
    goal: |-
      Keychain-backed master key + local rotation (finish integration; reduce password usage).
    touch:
      - packages/agent/daemon/keychain.py
      - packages/agent/vault/*
    steps:
      - Use OS keychain as master key source
      - Implement local key rotation
      - Update docs/runbooks for rotation
    acceptance:
      - Keychain tests pass; rotation runbook exists; no secret leakage to cloud.
    standards:
      - Secret management best practice
    effort: L
```

## Issue Registry (authoritative; do not skip)
```yaml
issues:
  -
    src: Список проблем и несогсасованностей.txt:5
    severity: CRITICAL
    phase: P0
    work_items:
      - WI-LEGAL-01
    depends_on:

    id: |-
      [CRITICAL] ToS описывает модель “пользователь передаёт Broker API Keys платформе, платформа исполняет/меняет/отменяет ордера” (docs/legal/TERMS_OF_SERVICE.md (line 24), docs/legal/TERMS_OF_SERVICE.md (line 185), docs/legal/TERMS_OF_SERVICE.md (line 188), docs/legal/TERMS_OF_SERVICE.md (line 197)) — Fix: переписать ToS под CCEA (ключи только в Agent, cloud не исполняет и не отправляет ордера; поправить определения и разделы ответственности/рисков).
  -
    src: Список проблем и несогсасованностей.txt:6
    severity: CRITICAL
    phase: P0
    work_items:
      - WI-LEGAL-01
    depends_on:

    id: |-
      [CRITICAL] Privacy Policy заявляет хранение/дешифровку broker credentials “для order execution” и “credential storage” (docs/legal/PRIVACY_POLICY.md (line 68), docs/legal/PRIVACY_POLICY.md (line 72), docs/legal/PRIVACY_POLICY.md (line 468)) — Fix: переписать Privacy Policy: креды хранятся локально в Agent; cloud не получает/не хранит; пересобрать таблицы категорий данных/ретеншн/шэринг.
  -
    src: Список проблем и несогсасованностей.txt:7
    severity: CRITICAL
    phase: P0
    work_items:
      - WI-LEGAL-01
    depends_on:

    id: |-
      [CRITICAL] DPA Template предполагает “Encrypted broker API keys” и “Order execution on user's behalf” (docs/legal/DPA_TEMPLATE.md (line 58), docs/legal/DPA_TEMPLATE.md (line 73)) — Fix: обновить DPA: обработка ключей/исполнение происходят в среде контроллера (customer‑managed agent), cloud — control/monitoring только.
  -
    src: Список проблем и несогсасованностей.txt:8
    severity: HIGH
    phase: P0
    work_items:
      - WI-LEGAL-01
    depends_on:

    id: |-
      [HIGH] Compliance/ops документ явно пишет “Client API keys stored” и даже “SEC/FINRA registered broker” (docs/DORA_OPERATIONAL_RESILIENCE_PLAN.md (line 2322), docs/DORA_OPERATIONAL_RESILIENCE_PLAN.md (line 2325)) — Fix: привести к posture “software provider / cloud ≠ execution”, убрать утверждения про брокерскую регистрацию; вынести старое в legacy/архив.
  -
    src: Список проблем и несогсасованностей.txt:9
    severity: HIGH
    phase: P0
    work_items:
      - WI-LEGAL-01
    depends_on:

    id: |-
      [HIGH] Упоминание Vault “для API keys” противоречит “cloud never stores keys” (docs/business/IP_PROTECTION_STRATEGY.md (line 118)) — Fix: заменить на “Local Agent Vault + OS keychain/CMK на customer‑managed host”; если Vault допустим только on‑prem у клиента — явно это указать.
  -
    src: Список проблем и несогсасованностей.txt:10
    severity: MEDIUM
    phase: P0
    work_items:
      - WI-LEGAL-01
    depends_on:

    id: |-
      [MEDIUM] Enterprise‑материал продолжает подразумевать API keys как хранимые платформой данные (docs/ENTERPRISE_ADOPTION_RISK_MITIGATION.md (line 406)) — Fix: явно указать “API keys остаются в Agent; в cloud не попадают”.
  -
    src: Список проблем и несогсасованностей.txt:11
    severity: HIGH
    phase: P0
    work_items:
      - WI-LEGAL-01
    depends_on:

    id: |-
      [HIGH] Маркетинг формулирует “execute trades on behalf of clients” (docs/BUSINESS_PLAN_EU_VISA.md (line 135), docs/INVESTOR_BRIEF.md (line 688)) — Fix: заменить на “execution runs on customer‑managed agent; platform provides software + monitoring/control plane; no brokerage/custody”.
  -
    src: Список проблем и несогсасованностей.txt:12
    severity: HIGH
    phase: P0
    work_items:
      - WI-LEGAL-01
    depends_on:

    id: |-
      [HIGH] AUP отсутствует, хотя план требует ToS/Privacy/AUP синхронизацию (docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 358)) — Fix: добавить docs/legal/AUP.md и согласовать формулировки с CCEA.
  -
    src: Список проблем и несогсасованностей.txt:15
    severity: CRITICAL
    phase: P0
    work_items:
      - WI-DOCS-01
    depends_on:
      - WI-LEGAL-01
    id: |-
      [CRITICAL] Нет требуемого docs/CCEA_OVERVIEW.md (docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 351)) — Fix: добавить overview (boundary, threat model, legal posture, product modes) и сделать ссылку из README.
  -
    src: Список проблем и несогсасованностей.txt:16
    severity: CRITICAL
    phase: P0
    work_items:
      - WI-DOCS-01
    depends_on:
      - WI-LEGAL-01
    id: |-
      [CRITICAL] Нет docs/cloud/* (docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 352)) — Fix: завести раздел под Cloud control plane API, builder/registry, governance/privacy/residency, job isolation.
  -
    src: Список проблем и несогсасованностей.txt:17
    severity: CRITICAL
    phase: P0
    work_items:
      - WI-DOCS-01
    depends_on:
      - WI-LEGAL-01
    id: |-
      [CRITICAL] Нет docs/agent/* (docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 353)) — Fix: завести install/upgrade, local vault, approvals, policies/hard caps, degraded modes, recovery.
  -
    src: Список проблем и несогсасованностей.txt:18
    severity: CRITICAL
    phase: P0
    work_items:
      - WI-DOCS-01
    depends_on:
      - WI-LEGAL-01
    id: |-
      [CRITICAL] Нет docs/runbooks/ (в плане “runbooks”) (docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 355)) — Fix: перенести/переписать runbooks под CCEA (incident/kill‑switch, revoke/rotation, safe‑degraded).
  -
    src: Список проблем и несогсасованностей.txt:19
    severity: HIGH
    phase: P0
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [HIGH] README не переориентирован на CCEA и всё ещё учит live через script_live.py (README.md (line 43), README.md (line 48), README.md (line 170)) — Fix: переписать README: Cloud research/build/monitoring + Local Agent execution; “keys stay local”.
  -
    src: Список проблем и несогсасованностей.txt:20
    severity: HIGH
    phase: P0
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [HIGH] ARCHITECTURE.md содержит legacy контракт Decision(side="BUY") и legacy entrypoint script_live.py (ARCHITECTURE.md (line 69), ARCHITECTURE.md (line 223), ARCHITECTURE.md (line 355)) — Fix: обновить архитектуру под CCEA; legacy вынести в docs/archive/.
  -
    src: Список проблем и несогсасованностей.txt:21
    severity: HIGH
    phase: P0
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [HIGH] docs/GETTING_STARTED.md описывает live запуск через script_live.py без CCEA‑границы (docs/GETTING_STARTED.md (line 336), docs/GETTING_STARTED.md (line 350)) — Fix: заменить на “Agent daemon + deployment/run из cloud control plane”.
  -
    src: Список проблем и несогсасованностей.txt:22
    severity: HIGH
    phase: P0
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [HIGH] docs/OPERATIONS_RUNBOOK.md использует script_live.py как операционный путь (docs/OPERATIONS_RUNBOOK.md (line 172), docs/OPERATIONS_RUNBOOK.md (line 190)) — Fix: переписать под Agent‑рантайм, добавить процедуры revoke/rotation и safe‑degraded.
  -
    src: Список проблем и несогсасованностей.txt:23
    severity: HIGH
    phase: P0
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [HIGH] docs/RECOVERY_PROCEDURES.md строится вокруг script_live.py (docs/RECOVERY_PROCEDURES.md (line 132), docs/RECOVERY_PROCEDURES.md (line 413)) — Fix: заменить на runbook для Agent daemon (journal/telemetry buffer/reconcile/kill‑switch).
  -
    src: Список проблем и несогсасованностей.txt:24
    severity: HIGH
    phase: P0
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [HIGH] docs/STOCK_TRADING_GUIDE.md даёт live команды через script_live.py (docs/STOCK_TRADING_GUIDE.md (line 93), docs/STOCK_TRADING_GUIDE.md (line 315)) — Fix: переписать под CCEA “live via local agent”.
  -
    src: Список проблем и несогсасованностей.txt:25
    severity: HIGH
    phase: P0
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [HIGH] docs/futures/deployment.md опирается на script_live.py (docs/futures/deployment.md (line 138)) — Fix: перевести на Agent deployment/run workflow.
  -
    src: Список проблем и несогсасованностей.txt:26
    severity: HIGH
    phase: P0
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [HIGH] docs/futures/configuration.md опирается на script_live.py (docs/futures/configuration.md (line 457)) — Fix: перевести конфиг‑инструкции на config layering CCEA (cloud desired vs local secrets).
  -
    src: Список проблем и несогсасованностей.txt:27
    severity: MEDIUM
    phase: P1
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [MEDIUM] docs/futures/migration_guide.md продолжает ссылаться на script_live.py (docs/futures/migration_guide.md (line 527)) — Fix: обновить migration guide под новый entrypoint (agent).
  -
    src: Список проблем и несогсасованностей.txt:28
    severity: MEDIUM
    phase: P1
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [MEDIUM] docs/futures/overview.md фиксирует старые entrypoints (docs/futures/overview.md (line 23)) — Fix: обновить overview под CCEA.
  -
    src: Список проблем и несогсасованностей.txt:29
    severity: MEDIUM
    phase: P1
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [MEDIUM] docs/FOREX_INTEGRATION_QUICK_REF.md содержит script_live.py как основной путь (docs/FOREX_INTEGRATION_QUICK_REF.md (line 110)) — Fix: адаптировать под Agent‑рантайм.
  -
    src: Список проблем и несогсасованностей.txt:30
    severity: MEDIUM
    phase: P1
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [MEDIUM] docs/SERVICE_DEPENDENCY_MAP.md позиционирует script_live.py/service_signal_runner.py как центральные (docs/SERVICE_DEPENDENCY_MAP.md (line 68), docs/SERVICE_DEPENDENCY_MAP.md (line 70)) — Fix: обновить карту под Cloud/Agent separation.
  -
    src: Список проблем и несогсасованностей.txt:31
    severity: LOW
    phase: P1
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [LOW] docs/codex/README.md советует отладку через service_signal_runner.py (docs/codex/README.md (line 98)) — Fix: отметить как legacy или переписать под agent CLI.
  -
    src: Список проблем и несогсасованностей.txt:32
    severity: LOW
    phase: P1
    work_items:
      - WI-DOCS-02
    depends_on:

    id: |-
      [LOW] docs/reports/FOREX_INDEX.txt перечисляет script_live.py как entrypoint (docs/reports/FOREX_INDEX.txt (line 184)) — Fix: пометить как legacy/обновить индекс.
  -
    src: Список проблем и несогсасованностей.txt:35
    severity: HIGH
    phase: P0
    work_items:
      - WI-TRACE-01
    depends_on:

    id: |-
      [HIGH] В Design_Doc_CCEA_Cloud.md указан SHA256, который не совпадает ни с файлом, ни с Design Doc CCEA Cloud.txt (docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.md (line 6)) — Fix: пересчитать SHA и зафиксировать корректно.
  -
    src: Список проблем и несогсасованностей.txt:36
    severity: HIGH
    phase: P0
    work_items:
      - WI-TRACE-01
    depends_on:

    id: |-
      [HIGH] Требуемый снапшот Design Doc для CI отсутствует (docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 9)) — Fix: добавить docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt (или ссылку+sha+дату) и синхронизировать.
  -
    src: Список проблем и несогсасованностей.txt:37
    severity: HIGH
    phase: P0
    work_items:
      - WI-TRACE-02
    depends_on:
      - WI-CLOUD-02
      - WI-AUTH-01
      - WI-CLOUD-04
      - WI-AGENT-05
      - WI-ENTERPRISE-01
      - WI-ENTERPRISE-02
      - WI-ENTERPRISE-03
      - WI-CLOUD-RESEARCH-01
      - WI-LEGAL-01
    id: |-
      [HIGH] Матрица трассируемости фиксирует PLANNED там, где план заявляет Done (то есть 100% соответствие не подтверждено): monitoring/alerts (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 20)), agent updates (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 22)), cloud job isolation (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 23)), data model org/workspace (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 25)), agent/deployment/run (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 27)), telemetry/access audit (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 29)), reconciliation (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 45)), retention/residency (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 61), docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 62)), evidence/on‑prem (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 68), docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 69)), AI Act/not advice (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 71), docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 72)) — Fix: либо реализовать и сменить статус, либо откорректировать “Done” в плане.
  -
    src: Список проблем и несогсасованностей.txt:38
    severity: HIGH
    phase: P0
    work_items:
      - WI-CI-01
    depends_on:
      - WI-CI-02
    id: |-
      [HIGH] CI guardrails в traceability тоже PLANNED (интеграция в CI не сделана): BT‑001..PM‑002 (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 115), docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 120)) — Fix: реально внедрить jobs/хуки, иначе guardrails существуют только “на бумаге”.
  -
    src: Список проблем и несогсасованностей.txt:41
    severity: HIGH
    phase: P0
    work_items:
      - WI-CI-02
    depends_on:
      - WI-DEPS-01
    id: |-
      [HIGH] Guardrail intent_prohibition падает из-за запретного поля signal в cloud тесте (packages/cloud/control_plane/tests/test_telemetry.py (line 45)) — Fix: переименовать поле/тип события или исключить тестовые фикстуры из скана.
  -
    src: Список проблем и несогсасованностей.txt:42
    severity: MEDIUM
    phase: P0
    work_items:
      - WI-CI-02
    depends_on:
      - WI-DEPS-01
    id: |-
      [MEDIUM] CLI intent_prohibition печатает Unicode (галочка/крестик), что ломает вывод в не‑UTF8 окружениях Windows (ccea/guardrails/intent_prohibition.py (line 512), ccea/guardrails/intent_prohibition.py (line 515)) — Fix: ASCII‑вывод или принудительный UTF‑8 stdout + PYTHONUTF8=1 в CI.
  -
    src: Список проблем и несогсасованностей.txt:43
    severity: HIGH
    phase: P0
    work_items:
      - WI-CI-02
    depends_on:
      - WI-DEPS-01
    id: |-
      [HIGH] cloud_allowlist даёт “Modules checked: 0” при 40 файлах (guardrail не даёт гарантии) — Fix: исправить подсчёт/разбор импортов и добавить тест‑предохранитель “0 modules => FAIL”.
  -
    src: Список проблем и несогсасованностей.txt:44
    severity: HIGH
    phase: P1
    work_items:
      - WI-BUILD-01
    depends_on:
      - WI-DEDRIFT-01
    id: |-
      [HIGH] План требует раздельных сборок cloud/agent и build‑time проверки состава cloud‑артефакта, но в репо нет артефактного разделения (только общий make build/test) (docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 98), docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 101), docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 102)) — Fix: разделить дистрибутивы (wheels/OCI) и запускать build_artifact_check на готовых артефактах в CI.
  -
    src: Список проблем и несогсасованностей.txt:45
    severity: HIGH
    phase: P0
    work_items:
      - WI-CI-01
    depends_on:

    id: |-
      [HIGH] Pre-commit не включает CCEA guardrails (.pre-commit-config.yaml) — Fix: добавить hooks для schema/protocol/import/intent/build_artifact checks.
  -
    src: Список проблем и несогсасованностей.txt:48
    severity: HIGH
    phase: P0
    work_items:
      - WI-PROTOCOL-01
    depends_on:

    id: |-
      [HIGH] Схемы/модели/код по schema version negotiation противоречат друг другу: schema содержит version_negotiation и комментарий про schema_version (docs/schemas/protocol_messages.schema.json (line 7), docs/schemas/protocol_messages.schema.json (line 17)), но POLL_COMMANDS — supported_schema_versions (docs/schemas/protocol_messages.schema.json (line 254), ccea/models/protocol.py (line 275)), а negotiation код ждёт min/max (ccea/protocol/schema_versioning.py (line 190), ccea/protocol/schema_versioning.py (line 191)) — Fix: выбрать один механизм и реально включить negotiation в runtime до приёма команд.
  -
    src: Список проблем и несогсасованностей.txt:49
    severity: HIGH
    phase: P0
    work_items:
      - WI-PROTOCOL-01
    depends_on:

    id: |-
      [HIGH] Константы поддерживаемых версий не согласованы со схемой (ccea/__init__.py (line 46), docs/schemas/protocol_messages.schema.json (line 12)) — Fix: единый источник истины для версий и генерация метаданных.
  -
    src: Список проблем и несогсасованностей.txt:50
    severity: HIGH
    phase: P0
    work_items:
      - WI-PROTOCOL-02
    depends_on:
      - WI-PROTOCOL-01
    id: |-
      [HIGH] Дрейф allowlist: REQUEST_RESUME_RUN есть в cloud коде, но его нет в boundary/schema (packages/cloud/control_plane/commands.py (line 40), packages/cloud/control_plane/boundary.py (line 47), docs/schemas/protocol_messages.schema.json (line 273)) — Fix: удалить или провести полный security‑review и добавить во все allowlists/схемы/тесты.
  -
    src: Список проблем и несогсасованностей.txt:51
    severity: HIGH
    phase: P0
    work_items:
      - WI-CLOUD-01
    depends_on:
      - WI-PROTOCOL-02
    id: |-
      [HIGH] Cloud API принимает произвольный command_type (строка) (packages/cloud/control_plane/routers/commands.py (line 111)) — Fix: enum allowlist + отказ на неизвестные.
  -
    src: Список проблем и несогсасованностей.txt:52
    severity: HIGH
    phase: P0
    work_items:
      - WI-CLOUD-01
      - WI-CLOUD-05
    depends_on:
      - WI-PROTOCOL-01
    id: |-
      [HIGH] Boundary validator объявлен, но не внедрён в API/dispatch (нет интеграции с router/service) — Fix: применять CloudBoundaryValidator при создании/отправке команд и при записи payload/config blobs.
  -
    src: Список проблем и несогсасованностей.txt:53
    severity: HIGH
    phase: P1
    work_items:
      - WI-CLOUD-02
      - WI-CLOUD-03
    depends_on:
      - WI-AUTH-01
      - WI-CLOUD-04
    id: |-
      [HIGH] План требует long‑poll commands + ack/approval/result endpoints (docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 56), docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 246)), но в cloud реализациях этого нет (есть только user CRUD команд и заглушка heartbeat) — Fix: реализовать агент‑аутентифицированные poll/ack/result/approval endpoints и связать их с хранилищем команд.
  -
    src: Список проблем и несогсасованностей.txt:56
    severity: HIGH
    phase: P1
    work_items:
      - WI-CLOUD-02
    depends_on:
      - WI-AUTH-01
    id: |-
      [HIGH] Agent heartbeat endpoint — заглушка (нет AgentDep auth, нет last_seen, pending_commands=0) (packages/cloud/control_plane/routers/auth.py (line 310), packages/cloud/control_plane/routers/auth.py (line 316)) — Fix: сделать нормальный heartbeat (auth+DB update+pending count).
  -
    src: Список проблем и несогсасованностей.txt:57
    severity: HIGH
    phase: P1
    work_items:
      - WI-CLOUD-02
      - WI-CLOUD-03
    depends_on:
      - WI-CLOUD-04
    id: |-
      [HIGH] Dispatch команд — заглушка “в реальной реализации через очередь” (packages/cloud/control_plane/commands.py (line 312)) — Fix: выбрать и реализовать один транспорт (DB+poll или MQ).
  -
    src: Список проблем и несогсасованностей.txt:58
    severity: HIGH
    phase: P1
    work_items:
      - WI-CLOUD-02
      - WI-CLOUD-03
    depends_on:
      - WI-CLOUD-04
    id: |-
      [HIGH] Дублирование командного слоя: in-memory packages/cloud/control_plane/commands.py vs DB routers/models (packages/cloud/control_plane/routers/commands.py) — Fix: оставить один вариант, второй удалить/архивировать.
  -
    src: Список проблем и несогсасованностей.txt:59
    severity: MEDIUM
    phase: P0
    work_items:
      - WI-AUTH-01
    depends_on:
      - WI-DEPS-01
    id: |-
      [MEDIUM] Пароли хэшируются SHA256 “for demo” (packages/cloud/control_plane/routers/auth.py (line 108), packages/cloud/control_plane/routers/users.py (line 30)) — Fix: bcrypt/argon2 + политика паролей + rate limits.
  -
    src: Список проблем и несогсасованностей.txt:60
    severity: MEDIUM
    phase: P0
    work_items:
      - WI-AUTH-01
    depends_on:
      - WI-CLOUD-04
    id: |-
      [MEDIUM] JWT revocation помечен TODO (packages/cloud/control_plane/routers/auth.py (line 194)) — Fix: jti‑blocklist/rotation либо refresh‑токены + короткий TTL.
  -
    src: Список проблем и несогсасованностей.txt:61
    severity: HIGH
    phase: P1
    work_items:
      - WI-CLOUD-04
    depends_on:
      - WI-DEPS-01
    id: |-
      [HIGH] RLS указан как SQL‑строка “to be executed during migration”, но миграций нет (packages/cloud/control_plane/models.py (line 1330), packages/cloud/control_plane/models.py (line 1333)) — Fix: Alembic migrations + реальное включение RLS/policies.
  -
    src: Список проблем и несогсасованностей.txt:62
    severity: HIGH
    phase: P1
    work_items:
      - WI-CLOUD-05
    depends_on:
      - WI-CLOUD-01
    id: |-
      [HIGH] Config blobs принимают любой JSON без secret/DLP scan и без schema validation (packages/cloud/control_plane/routers/config_blobs.py (line 360), packages/cloud/control_plane/routers/config_blobs.py (line 363)) — Fix: валидировать и сканировать; запретить секретные поля/паттерны; отделить secrets local‑only.
  -
    src: Список проблем и несогсасованностей.txt:63
    severity: HIGH
    phase: P1
    work_items:
      - WI-CLOUD-06
    depends_on:
      - WI-DEPS-01
    id: |-
      [HIGH] Cloud artifact builder в packages/cloud/builder — placeholder (упаковка и подпись “в реальной реализации…”) (packages/cloud/builder/artifact_builder.py (line 229), packages/cloud/builder/artifact_builder.py (line 255)) — Fix: заменить на ccea/artifact/* или довести до OCI+SBOM+sigstore и связать с control plane.
  -
    src: Список проблем и несогсасованностей.txt:64
    severity: HIGH
    phase: P2
    work_items:
      - WI-CLOUD-RESEARCH-01
    depends_on:

    id: |-
      [HIGH] Cloud research job isolation не реализована: ResearchJobRunner упомянут, но отсутствует (packages/cloud/research/__init__.py (line 13)) — Fix: реализовать sandbox runner с квотами и egress‑политикой.
  -
    src: Список проблем и несогсасованностей.txt:67
    severity: CRITICAL
    phase: P0
    work_items:
      - WI-AGENT-01
    depends_on:
      - WI-DEPS-01
    id: |-
      [CRITICAL] packages/agent/daemon/sandbox.py не работает на Windows из-за resource (packages/agent/daemon/sandbox.py (line 18)), из-за чего падают CCEA tests (pytest tests/ccea падает на импорте) — Fix: Windows‑ветка (Job Objects/psutil) или явный “Linux-only” + skipping tests.
  -
    src: Список проблем и несогсасованностей.txt:68
    severity: HIGH
    phase: P0
    work_items:
      - WI-AGENT-02
    depends_on:

    id: |-
      [HIGH] sqlite3.connect используется как context manager (не закрывает conn), что приводит к lock’ам на Windows (packages/agent/daemon/telemetry_buffer.py (line 246), packages/agent/daemon/telemetry_buffer.py (line 400)) — Fix: contextlib.closing(...)/явный close() и корректная остановка фоновых потоков.
  -
    src: Список проблем и несогсасованностей.txt:69
    severity: HIGH
    phase: P1
    work_items:
      - WI-AGENT-03
      - WI-CLOUD-02
    depends_on:
      - WI-AUTH-01
    id: |-
      [HIGH] Heartbeat “to cloud” в agentd не реализован (ставит connected=True “in real implementation…”) (packages/agent/daemon/agentd.py (line 731), packages/agent/daemon/agentd.py (line 732)) — Fix: CloudClient с реальными enroll/heartbeat/poll/ack/result.
  -
    src: Список проблем и несогсасованностей.txt:70
    severity: HIGH
    phase: P1
    work_items:
      - WI-AGENT-04
    depends_on:
      - WI-AGENT-03
    id: |-
      [HIGH] Local approval для TRADING_IMPACTING не включён в рабочий путь (модуль есть, wiring нет; план требует) (docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 277)) — Fix: интегрировать packages/agent/approval/* в применение start/upgrade/update_config.
  -
    src: Список проблем и несогсасованностей.txt:71
    severity: HIGH
    phase: P1
    work_items:
      - WI-AGENT-05
    depends_on:
      - WI-AGENT-06
    id: |-
      [HIGH] Reconciliation заявлен, но не доведён до рабочего контура (в runner нет вызовов, а в reconciler есть заглушка) (packages/agent/runner/live.py (line 14), packages/agent/runner/live.py (line 91), packages/agent/reconciliation/reconciler.py (line 210)) — Fix: подключить reconciler+journal, делать reconcile на старте/периодически, safe halt при неопределённости.
  -
    src: Список проблем и несогсасованностей.txt:72
    severity: HIGH
    phase: P1
    work_items:
      - WI-AGENT-06
    depends_on:
      - WI-AGENT-02
    id: |-
      [HIGH] “Детерминированный client_order_id” не выполнен: используется счётчик, idempotency — только in‑memory (packages/agent/execution/engine.py (line 314), packages/agent/execution/engine.py (line 316), packages/agent/execution/engine.py (line 266)) — Fix: детерминированный ID + персистентный dedup (journal/sqlite).
  -
    src: Список проблем и несогсасованностей.txt:73
    severity: MEDIUM
    phase: P2
    work_items:
      - WI-VAULT-01
    depends_on:
      - WI-DEPS-01
    id: |-
      [MEDIUM] Keychain integration существует, но vault/credential manager остаются password‑based (интеграция не завершена) — Fix: использовать keychain как источник master key и реализовать локальную ротацию.
  -
    src: Список проблем и несогсасованностей.txt:76
    severity: HIGH
    phase: P0
    work_items:
      - WI-DEDRIFT-01
    depends_on:
      - WI-CLOUD-02
      - WI-AGENT-03
    id: |-
      [HIGH] Два параллельных “Agent” стека: ccea/agent/daemon.py (реально делает HTTP‑протокол) vs packages/agent/daemon/agentd.py (локальный/частично заглушки) (ccea/agent/daemon.py (line 33), packages/agent/daemon/agentd.py (line 731)) — Fix: выбрать канонический стек и убрать второй.
  -
    src: Список проблем и несогсасованностей.txt:77
    severity: HIGH
    phase: P0
    work_items:
      - WI-DEDRIFT-01
    depends_on:
      - WI-CLOUD-02
      - WI-AGENT-03
    id: |-
      [HIGH] Два параллельных “Control Plane” стека: ccea/control_plane/* vs packages/cloud/control_plane/* + ещё packages/cloud/control_plane/commands.py — Fix: унифицировать и оставить один источник истины для протокола/статусов/доставки команд.
  -
    src: Список проблем и несогсасованностей.txt:78
    severity: HIGH
    phase: P0
    work_items:
      - WI-CONTRACTS-01
    depends_on:
      - WI-PROTOCOL-01
    id: |-
      [HIGH] Дрейф enum/state machines между слоями (протокол vs cloud DB) (ccea/models/protocol.py (line 40), ccea/models/protocol.py (line 68), packages/cloud/control_plane/models.py (line 95), packages/cloud/control_plane/models.py (line 109)) — Fix: централизовать контракты (shared) и/или генерировать из schema; в БД хранить значения с чётким маппингом к протоколу.
  -
    src: Список проблем и несогсасованностей.txt:81
    severity: HIGH
    phase: P2
    work_items:
      - WI-ENTERPRISE-01
    depends_on:
      - WI-CLOUD-02
    id: |-
      [HIGH] On‑prem pack (docker-compose/Helm) отсутствует, хотя заявлен (docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 316), PLANNED: docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 69)) — Fix: добавить манифесты развёртывания + документацию.
  -
    src: Список проблем и несогсасованностей.txt:82
    severity: HIGH
    phase: P2
    work_items:
      - WI-ENTERPRISE-02
    depends_on:
      - WI-CLOUD-02
      - WI-CLOUD-06
      - WI-AGENT-05
    id: |-
      [HIGH] Evidence pack exporter отсутствует (PLANNED) (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 68)) — Fix: реализовать экспорт digests/signatures/SBOM + approvals/commands/halt logs; подписывать пакет.
  -
    src: Список проблем и несогсасованностей.txt:83
    severity: HIGH
    phase: P2
    work_items:
      - WI-ENTERPRISE-03
    depends_on:
      - WI-CLOUD-06
    id: |-
      [HIGH] Signed agent updates / staged rollout отсутствуют (PLANNED) (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 22)) — Fix: реализовать механизм обновлений (подпись, pinning, rollback protection).
  -
    src: Список проблем и несогсасованностей.txt:84
    severity: HIGH
    phase: P2
    work_items:
      - WI-CLOUD-RESEARCH-01
    depends_on:

    id: |-
      [HIGH] Cloud research job isolation/anti‑abuse отсутствует (PLANNED) (docs/design/CCEA_CLOUD/CCEA_TRACEABILITY_MATRIX.md (line 23), docs/plans/CCEA_CLOUD_ALIGNMENT_PLAN.md (line 337)) — Fix: sandbox runner (контейнер/VM), квоты, egress allowlist, abuse detection.
  -
    src: Список проблем и несогсасованностей.txt:87
    severity: HIGH
    phase: P0
    work_items:
      - WI-DEPS-01
    depends_on:

    id: |-
      [HIGH] cryptography обязателен для vault/signing (packages/agent/vault/local_vault.py (line 30), ccea/crypto/keys.py (line 19)) — Fix: добавить в pyproject.toml/lockfiles и проверять в CI.
  -
    src: Список проблем и несогсасованностей.txt:88
    severity: HIGH
    phase: P0
    work_items:
      - WI-DEPS-01
    depends_on:

    id: |-
      [HIGH] Cloud DB по умолчанию требует asyncpg (URL содержит postgresql+asyncpg) (packages/cloud/control_plane/database.py (line 29)) — Fix: добавить asyncpg в зависимости или сменить дефолт на sqlite только для dev/test.
  -
    src: Список проблем и несогсасованностей.txt:89
    severity: HIGH
    phase: P0
    work_items:
      - WI-DEPS-01
    depends_on:

    id: |-
      [HIGH] Cloud auth требует PyJWT (packages/cloud/control_plane/dependencies.py (line 21)) — Fix: явно добавить PyJWT в зависимости/lockfiles и зафиксировать версию.
```

## Execution Order (agent-optimized)
- P0: WI-DEPS-01 → WI-AGENT-01/WI-AGENT-02 → WI-CI-02 → WI-CI-01 → WI-PROTOCOL-01/WI-PROTOCOL-02 → WI-CLOUD-01 → WI-TRACE-01/WI-TRACE-02 → WI-LEGAL-01 → WI-DOCS-01/WI-DOCS-02 → WI-DEDRIFT-01/WI-CONTRACTS-01
- P1: WI-CLOUD-04 → WI-AUTH-01 → WI-CLOUD-02/WI-CLOUD-03 → WI-AGENT-03 → WI-AGENT-04 → WI-AGENT-06 → WI-AGENT-05 → WI-CLOUD-05 → WI-CLOUD-06 → WI-BUILD-01
- P2: WI-CLOUD-RESEARCH-01 → WI-ENTERPRISE-01 → WI-ENTERPRISE-03 → WI-ENTERPRISE-02 → WI-VAULT-01

