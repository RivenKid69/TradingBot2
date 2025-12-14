# CCEA Traceability Matrix

> **Version**: 1.0.0
> **Date**: 2025-12-13
> **Status**: APPROVED

Матрица трассируемости связывает требования Design Doc с фазами плана, кодом и проверками.

## 1. Design Doc Requirements → Plan Phases

| Design Doc Section | Requirement | Plan Phase | Artifact/Check | Status |
|--------------------|-------------|------------|----------------|--------|
| **0. Key Principle** |||||
| 0.1 | Cloud = research/control, Agent = secrets/execution | Phase 0 | TARGET_CCEA_ARCHITECTURE.md | DONE |
| 0.2 | Cloud never stores keys | Phase 0 | DECISION_LOG.md (OQ-005) | DONE |
| 0.3 | Cloud never sends orders | Phase 0 | protocol_messages.schema.json | DONE |
| **3-5. Requirements** |||||
| 3.1 | Product modes (Retail/Enterprise) | Phase 0 | Design_Doc_CCEA_Cloud.md §3 | DONE |
| 3.2 | Zone separation (Cloud/Agent/Shared) | Phase 0, 2 | TARGET_CCEA_ARCHITECTURE.md | DONE |
| 4.1 | Monitoring/alerts | Phase 8 | - | PLANNED |
| 5.1 | Artifact signing | Phase 1, 4 | artifact_manifest.schema.json | DONE |
| 5.2 | Agent updates | Phase 9 | - | PLANNED |
| 5.3 | Cloud job isolation | Phase 10 | - | PLANNED |
| **6. Data Model** |||||
| 6.1 | Org/Workspace/User/Roles | Phase 6 | - | PLANNED |
| 6.2 | Strategy/Build/Artifact | Phase 4, 6 | artifact_manifest.schema.json | DONE |
| 6.3 | Agent/Deployment/Run | Phase 6 | - | PLANNED |
| 6.4 | Command/ApprovalRecord | Phase 6, 7 | protocol_messages.schema.json | DONE |
| 6.5 | TelemetryEvent/AccessAudit | Phase 8 | - | PLANNED |
| **7. Change Class/Policy Firewall** |||||
| 7.1 | TRADING_IMPACTING classification | Phase 0, 7 | DECISION_LOG.md | DONE |
| 7.2 | Local approval required | Phase 0, 7 | protocol_messages.schema.json | DONE |
| 7.3 | Hard caps (local priority) | Phase 5 | Design_Doc_CCEA_Cloud.md §6 | DONE |
| **8. Artifacts** |||||
| 8.1 | Manifest schema | Phase 0, 4 | artifact_manifest.schema.json | DONE |
| 8.2 | Digest pinning | Phase 4 | artifact_manifest.schema.json | DONE |
| 8.3 | Signature (sigstore/GPG) | Phase 4 | DECISION_LOG.md (OQ-007) | DONE |
| 8.4 | SBOM | Phase 4 | artifact_manifest.schema.json (sbom_ref) | DONE |
| 8.5 | Provenance | Phase 4 | artifact_manifest.schema.json | DONE |
| **9. Agent Runtime** |||||
| 9.1 | Local Vault | Phase 5 | Design_Doc_CCEA_Cloud.md §1.2 | DONE |
| 9.2 | Sandbox/isolation | Phase 5 | DECISION_LOG.md (OQ-001) | DONE |
| 9.3 | Policy Firewall | Phase 5 | Design_Doc_CCEA_Cloud.md §1.2 | DONE |
| 9.4 | Kill Switch | Phase 5 | CCEA_SEQUENCE_DIAGRAMS.md §7 | DONE |
| 9.5 | Reconciliation | Phase 5 | - | PLANNED |
| **10. Protocol** |||||
| 10.1 | Message types (allowlist) | Phase 0, 7 | protocol_messages.schema.json | DONE |
| 10.2 | Authentication (mTLS/JWT) | Phase 1, 7 | Design_Doc_CCEA_Cloud.md §4.2 | DONE |
| 10.3 | Idempotency | Phase 7 | protocol_messages.schema.json | DONE |
| 10.4 | Schema versioning | Phase 7 | protocol_messages.schema.json | DONE |
| 10.5 | No order-like payload | Phase 0 | protocol_messages.schema.json | DONE |
| **11. State Machines** |||||
| 11.1 | Deployment states | Phase 0, 7 | Design_Doc_CCEA_Cloud.md §5.1 | DONE |
| 11.2 | Run states | Phase 0, 7 | Design_Doc_CCEA_Cloud.md §5.2 | DONE |
| **12. Config Layering** |||||
| 12.1 | Priority order | Phase 0 | Design_Doc_CCEA_Cloud.md §6 | DONE |
| 12.2 | Local hard caps priority | Phase 0 | Design_Doc_CCEA_Cloud.md §6.1 | DONE |
| **13-14. Telemetry/Privacy/Residency** |||||
| 13.1 | Telemetry levels | Phase 0, 8 | DECISION_LOG.md (OQ-002) | DONE |
| 13.2 | Mandatory redaction | Phase 0, 8 | Design_Doc_CCEA_Cloud.md §8.2 | DONE |
| 14.1 | Data retention | Phase 8 | - | PLANNED |
| 14.2 | Data residency | Phase 8 | - | PLANNED |
| **15. Security** |||||
| 15.1 | Threat model | Phase 0 | Design_Doc_CCEA_Cloud.md §7 | DONE |
| 15.2 | Safe defaults | Phase 0 | DECISION_LOG.md | DONE |
| 15.3 | Secret hygiene | Phase 0 | Design_Doc_CCEA_Cloud.md §2.4 | DONE |
| **16. Enterprise/Evidence Pack** |||||
| 16.1 | Evidence pack exporter | Phase 9 | - | PLANNED |
| 16.2 | On-prem/VPC deployment | Phase 9 | - | PLANNED |
| **17-18. AI Act/Not Advice** |||||
| 17.1 | AI Act posture | Phase 11 | - | PLANNED |
| 18.1 | "Not advice" disclaimers | Phase 11 | - | PLANNED |
| **19. CI Guardrails** |||||
| 19.1 | No trading libs in cloud | Phase 0, 1 | CI_GUARDRAILS.md (BT-001) | DONE |
| 19.2 | No order payload in schema | Phase 0, 1 | CI_GUARDRAILS.md (BT-002) | DONE |
| 19.3 | Artifact signature required | Phase 1, 4 | CI_GUARDRAILS.md (BT-003) | DONE |
| 19.4 | Import boundary check | Phase 2 | CI_GUARDRAILS.md (BT-005) | DONE |
| **20. Rollout Plan** |||||
| 20.1 | Phase mapping | Phase 0 | TARGET_CCEA_ARCHITECTURE.md §5 | DONE |
| **21. Open Questions** |||||
| 21.1 | Sandbox mode | Phase 0 | DECISION_LOG.md (OQ-001) | DONE |
| 21.2 | RAW telemetry policy | Phase 0 | DECISION_LOG.md (OQ-002) | DONE |
| 21.3 | Remote flatten | Phase 0 | DECISION_LOG.md (OQ-003) | DONE |
| 21.4 | Cloud inference | Phase 0 | DECISION_LOG.md (OQ-004) | DONE |
| 21.5 | Data classification | Phase 0 | DECISION_LOG.md (OQ-005) | DONE |
| 21.6 | BYO verification | Phase 0 | DECISION_LOG.md (OQ-006) | DONE |
| 21.7 | Key management | Phase 0 | DECISION_LOG.md (OQ-007) | DONE |
| **22. Sequence Diagrams** |||||
| 22.1 | Enrollment flow | Phase 0 | CCEA_SEQUENCE_DIAGRAMS.md §1 | DONE |
| 22.2 | Deploy+Start+Approve | Phase 0 | CCEA_SEQUENCE_DIAGRAMS.md §2 | DONE |
| 22.3 | Upgrade flow | Phase 0 | CCEA_SEQUENCE_DIAGRAMS.md §3 | DONE |
| 22.4 | Stop/Pause flow | Phase 0 | CCEA_SEQUENCE_DIAGRAMS.md §4 | DONE |
| 22.5 | Key rotation | Phase 0 | CCEA_SEQUENCE_DIAGRAMS.md §5 | DONE |
| 22.6 | Export logs | Phase 0 | CCEA_SEQUENCE_DIAGRAMS.md §6 | DONE |

## 2. Plan Phase 0 Done Criteria → Artifacts

| Done Criterion | Artifact | Location | Status |
|----------------|----------|----------|--------|
| Target CCEA Architecture | TARGET_CCEA_ARCHITECTURE.md | docs/design/CCEA_CLOUD/ | DONE |
| Module zone mapping | TARGET_CCEA_ARCHITECTURE.md §2 | docs/design/CCEA_CLOUD/ | DONE |
| Decision Log (Open Questions) | DECISION_LOG.md | docs/design/CCEA_CLOUD/ | DONE |
| JSON Schema (manifest) | artifact_manifest.schema.json | docs/schemas/ | DONE |
| JSON Schema (protocol) | protocol_messages.schema.json | docs/schemas/ | DONE |
| CI Guardrails list | CI_GUARDRAILS.md | docs/design/CCEA_CLOUD/ | DONE |
| Design Doc snapshot | Design_Doc_CCEA_Cloud.md | docs/design/CCEA_CLOUD/ | DONE |
| Rollout mapping | TARGET_CCEA_ARCHITECTURE.md §5 | docs/design/CCEA_CLOUD/ | DONE |
| Sequence diagrams | CCEA_SEQUENCE_DIAGRAMS.md | docs/design/CCEA_CLOUD/ | DONE |
| Traceability matrix | CCEA_TRACEABILITY_MATRIX.md | docs/design/CCEA_CLOUD/ | DONE |

## 3. CI Guardrails → Implementation

| Guardrail ID | Name | Implementation File | Test File | Status |
|--------------|------|---------------------|-----------|--------|
| BT-001 | no-trading-libs-in-cloud | ccea/guardrails/import_check.py | tests/ccea/guardrails/test_import_check.py | DONE |
| BT-002 | no-order-payloads-in-schema | ccea/guardrails/schema_check.py | tests/ccea/guardrails/test_schema_check.py | DONE |
| BT-003 | artifact-signature-required | ccea/guardrails/artifact_check.py | tests/ccea/guardrails/test_artifact_check.py | DONE |
| BT-005 | import-boundary-check | ccea/guardrails/import_check.py, importlinter.ini | tests/ccea/guardrails/test_import_check.py | DONE |
| PM-001 | schema-validation | ccea/guardrails/schema_check.py | tests/ccea/guardrails/test_schema_check.py | DONE |
| PM-002 | protocol-allowlist | ccea/guardrails/protocol_check.py | tests/ccea/guardrails/test_protocol_check.py | DONE |
| PM-003 | intent-prohibition | ccea/guardrails/intent_prohibition.py | tests/ccea/guardrails/test_intent_prohibition.py | DONE |
| PM-004 | secret-scan | .gitleaks.toml, .secrets.baseline | build-and-test.yml | DONE |
| PM-005 | cloud-allowlist | ccea/guardrails/cloud_allowlist.py | tests/ccea/guardrails/test_cloud_allowlist.py | DONE |
| PM-006 | design-doc-sha | ccea/guardrails/design_doc_check.py | tests/ccea/guardrails/test_design_doc_check.py | DONE |
| PM-007 | traceability-check | ccea/guardrails/traceability_check.py | tests/ccea/guardrails/test_traceability_check.py | DONE |
| RT-001 | signature-verification | packages/agent/artifact_verifier.py | tests/agent/test_artifact_verifier.py | PLANNED |
| RT-005 | redaction-middleware | packages/agent/telemetry/redaction.py | tests/agent/test_redaction.py | PLANNED |

## 4. JSON Schema → Prohibited Fields Verification

| Schema | Prohibited Field | Check Location | Test | Status |
|--------|------------------|----------------|------|--------|
| artifact_manifest.schema.json | side | $defs.prohibited_fields | test_manifest_prohibits_side | DONE |
| artifact_manifest.schema.json | quantity | $defs.prohibited_fields | test_manifest_prohibits_quantity | DONE |
| artifact_manifest.schema.json | price | $defs.prohibited_fields | test_manifest_prohibits_price | DONE |
| artifact_manifest.schema.json | order_type | $defs.prohibited_fields | test_manifest_prohibits_order_type | DONE |
| artifact_manifest.schema.json | target_position | $defs.prohibited_fields | test_manifest_prohibits_target_position | DONE |
| protocol_messages.schema.json | side | prohibited_order_fields | test_protocol_prohibits_side | DONE |
| protocol_messages.schema.json | quantity/qty | prohibited_order_fields | test_protocol_prohibits_quantity | DONE |
| protocol_messages.schema.json | price | prohibited_order_fields | test_protocol_prohibits_price | DONE |
| protocol_messages.schema.json | order_type | prohibited_order_fields | test_protocol_prohibits_order_type | DONE |
| protocol_messages.schema.json | intent | prohibited_order_fields | test_protocol_prohibits_intent | DONE |
| protocol_messages.schema.json | signal | prohibited_order_fields | test_protocol_prohibits_signal | DONE |

## 5. Sequence Diagram → Test Scenario Mapping

| Diagram | Scenario | Test File | Status |
|---------|----------|-----------|--------|
| §1 Enrollment | Agent enrollment happy path | tests/ccea/test_enrollment.py | PLANNED |
| §1 Enrollment | Token TTL expiration | tests/ccea/test_enrollment.py | PLANNED |
| §2 Deploy+Start | Start with approval | tests/ccea/test_deployment.py | PLANNED |
| §2 Deploy+Start | Signature verification failure | tests/ccea/test_deployment.py | PLANNED |
| §3 Upgrade | Upgrade with diff approval | tests/ccea/test_upgrade.py | PLANNED |
| §4 Stop/Pause | Stop without approval | tests/ccea/test_lifecycle.py | PLANNED |
| §5 Key Rotation | Key rotation with approval | tests/ccea/test_rotation.py | PLANNED |
| §6 Export Logs | Export with redaction | tests/ccea/test_export.py | PLANNED |
| §7 Kill Switch | Kill switch trigger | tests/ccea/test_kill_switch.py | PLANNED |
| §8 Heartbeat | Heartbeat + telemetry | tests/ccea/test_telemetry.py | PLANNED |
| §9 Preflight | Preflight checks | tests/ccea/test_preflight.py | PLANNED |

## 6. Module Zone Verification

| Module Pattern | Expected Zone | Verification Check | Status |
|----------------|---------------|-------------------|--------|
| `core_*.py` | SHARED | import_check --zone=shared | PLANNED |
| `impl_*.py` | SHARED | import_check --zone=shared | PLANNED |
| `adapters/*/market_data.py` | SHARED | import_check --zone=shared | PLANNED |
| `adapters/*/fees.py` | SHARED | import_check --zone=shared | PLANNED |
| `adapters/*/order_execution.py` | AGENT | import_check --zone=agent | PLANNED |
| `adapters/*/options_execution.py` | AGENT | import_check --zone=agent | PLANNED |
| `execution_providers.py` (live) | AGENT | import_check --zone=agent | PLANNED |
| `service_signal_runner.py` (live) | AGENT | import_check --zone=agent | PLANNED |
| `risk_guard.py` | AGENT | import_check --zone=agent | PLANNED |
| `app.py` | CLOUD | import_check --zone=cloud | PLANNED |
| `service_backtest.py` | CLOUD | import_check --zone=cloud | PLANNED |
| `service_train.py` | CLOUD | import_check --zone=cloud | PLANNED |

## 7. Decision Log → Test Verification

| Decision ID | Decision | Verification Test | Status |
|-------------|----------|-------------------|--------|
| OQ-001 | Sandbox: process-ok, docker recommended | test_sandbox_fallback | PLANNED |
| OQ-002 | RAW telemetry: enterprise-only, opt-in | test_telemetry_level_restrictions | PLANNED |
| OQ-003 | Remote flatten: disabled by default | test_flatten_policy | PLANNED |
| OQ-004 | Cloud inference: disabled | test_no_cloud_inference | PLANNED |
| OQ-005 | Data classification: 4 levels | test_data_classification | PLANNED |
| OQ-006 | BYO verification: ToS + attestation | test_byo_verification | PLANNED |
| OQ-007 | Signing: sigstore default, keyful for enterprise | test_signing_methods | PLANNED |

## 8. Coverage Summary

### Phase 0 Requirements

| Category | Total | Documented | Tested | Coverage |
|----------|-------|------------|--------|----------|
| Design Doc sections covered | 22 | 22 | - | 100% |
| Open Questions resolved | 7 | 7 | - | 100% |
| JSON Schemas created | 2 | 2 | - | 100% |
| Sequence diagrams | 9 | 9 | - | 100% |
| CI Guardrails defined | 9 | 9 | - | 100% |
| Done criteria met | 10 | 10 | - | 100% |

### Phase 3 Requirements (CI/Traceability Hardening)

| Category | Total | Implemented | Tested | Coverage |
|----------|-------|-------------|--------|----------|
| CI Guardrails enforcement | 11 | 11 | 11 | 100% |
| Pre-commit hooks added | 8 | 8 | 8 | 100% |
| Design Doc SHA verification | 1 | 1 | 1 | 100% |
| Traceability matrix check | 1 | 1 | 1 | 100% |

### Documentation Status

| Document | Created | Reviewed | Approved |
|----------|---------|----------|----------|
| Design_Doc_CCEA_Cloud.md | Yes | Yes | Yes |
| TARGET_CCEA_ARCHITECTURE.md | Yes | Yes | Yes |
| DECISION_LOG.md | Yes | Yes | Yes |
| CI_GUARDRAILS.md | Yes | Yes | Yes |
| CCEA_SEQUENCE_DIAGRAMS.md | Yes | Yes | Yes |
| CCEA_TRACEABILITY_MATRIX.md | Yes | Yes | Yes |
| artifact_manifest.schema.json | Yes | Yes | Yes |
| protocol_messages.schema.json | Yes | Yes | Yes |

---

**Document Control:**
- Author: CCEA Architecture Team
- Last Updated: 2025-12-15
- Phase 3 Completion: CI/Traceability Hardening
- Next Review: After Phase 4 completion
