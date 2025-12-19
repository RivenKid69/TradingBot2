# CCEA Traceability Matrix

> **Version**: 2.0.0
> **Date**: 2025-12-16
> **Status**: INTERNAL REVIEW COMPLETE (engineering team approval; not independently audited)

Матрица трассируемости связывает требования Design Doc с фазами плана, кодом и проверками.

## 1. Design Doc Requirements → Plan Phases

| Design Doc Section | Requirement | Plan Phase | Artifact/Check | Status |
|--------------------|-------------|------------|----------------|--------|
| **0. Key Principle** |||||
| 0.1 | Cloud = research/control, Agent = secrets/execution | Phase 0 | TARGET_CCEA_ARCHITECTURE.md | DONE |
| 0.2 | Cloud designed to not store keys | Phase 0 | DECISION_LOG.md (OQ-005) | IMPLEMENTED (verify via tests) |
| 0.3 | Cloud never sends orders | Phase 0 | protocol_messages.schema.json | DONE |
| **3-5. Requirements** |||||
| 3.1 | Product modes (Retail/Enterprise) | Phase 0 | Design_Doc_CCEA_Cloud.md §3 | DONE |
| 3.2 | Zone separation (Cloud/Agent/Shared) | Phase 0, 2 | TARGET_CCEA_ARCHITECTURE.md | DONE |
| 4.1 | Monitoring/alerts | Phase 8 | packages/cloud/governance/alert_rules.py, tests/ccea/phase8/test_alert_rules.py | DONE |
| 5.1 | Artifact signing | Phase 1, 4 | artifact_manifest.schema.json | DONE |
| 5.2 | Agent updates | Phase 9 | packages/cloud/enterprise/agent_updates.py, tests/ccea/phase9/test_agent_updates.py | DONE |
| 5.3 | Cloud job isolation | Phase 10 | packages/cloud/research/sandbox/, tests/ccea/phase10/test_cloud_sandbox.py | DONE |
| **6. Data Model** |||||
| 6.1 | Org/Workspace/User/Roles | Phase 6, 7 | packages/cloud/control_plane/routers/organizations.py, tests/ccea/phase6/ | DONE |
| 6.2 | Strategy/Build/Artifact | Phase 4, 6 | artifact_manifest.schema.json | DONE |
| 6.3 | Agent/Deployment/Run | Phase 6, 7 | packages/cloud/control_plane/routers/deployments.py | DONE |
| 6.4 | Command/ApprovalRecord | Phase 6, 7 | protocol_messages.schema.json, packages/agent/approval/ | DONE |
| 6.5 | TelemetryEvent/AccessAudit | Phase 8 | packages/agent/telemetry/, tests/ccea/phase8/test_level_manager.py | DONE |
| **7. Change Class/Policy Firewall** |||||
| 7.1 | TRADING_IMPACTING classification | Phase 0, 7 | DECISION_LOG.md, packages/cloud/control_plane/services/change_class_enforcer.py | DONE |
| 7.2 | Local approval required | Phase 0, 7 | protocol_messages.schema.json, packages/agent/approval/manager.py | DONE |
| 7.3 | Hard caps (local priority) | Phase 5 | packages/agent/policy/hard_caps.py, tests/ccea/phase4/test_layered_policy.py | DONE |
| **8. Artifacts** |||||
| 8.1 | Manifest schema | Phase 0, 4 | artifact_manifest.schema.json | DONE |
| 8.2 | Digest pinning | Phase 4 | artifact_manifest.schema.json, ccea/artifact/verifier.py | DONE |
| 8.3 | Signature (sigstore/GPG) | Phase 4 | ccea/artifact/signer.py, tests/ccea/phase4/test_verifier.py | DONE |
| 8.4 | SBOM | Phase 4 | ccea/artifact/sbom.py, tests/ccea/phase4/test_sbom.py | DONE |
| 8.5 | Provenance | Phase 4 | packages/cloud/builder/provenance_validator.py | DONE |
| **9. Agent Runtime** |||||
| 9.1 | Local Vault | Phase 5 | packages/agent/vault/local_vault.py, tests/ccea/phase5/test_keychain.py | DONE |
| 9.2 | Sandbox/isolation | Phase 5 | packages/agent/daemon/sandbox.py, tests/ccea/phase5/test_sandbox.py | DONE |
| 9.3 | Policy Firewall | Phase 5 | packages/agent/policy/firewall.py, tests/ccea/phase4/test_layered_policy.py | DONE |
| 9.4 | Kill Switch | Phase 5 | packages/agent/daemon/kill_switch.py, tests/ccea/phase5/test_kill_switch.py | DONE |
| 9.5 | Reconciliation | Phase 5, 9 | packages/agent/reconciliation/, tests/ccea/phase4/test_reconciler.py | DONE |
| **10. Protocol** |||||
| 10.1 | Message types (allowlist) | Phase 0, 7 | protocol_messages.schema.json | DONE |
| 10.2 | Authentication (mTLS/JWT) | Phase 1, 7 | packages/cloud/control_plane/middleware/agent_signature.py | DONE |
| 10.3 | Idempotency | Phase 7 | packages/cloud/control_plane/services/idempotency.py, tests/ccea/phase8_p1/test_order_idempotency.py | DONE |
| 10.4 | Schema versioning | Phase 7 | ccea/protocol/schema_versioning.py | DONE |
| 10.5 | No order-like payload | Phase 0 | protocol_messages.schema.json, ccea/guardrails/intent_prohibition.py | DONE |
| **11. State Machines** |||||
| 11.1 | Deployment states | Phase 0, 7 | ccea/models/state_machines.py | DONE |
| 11.2 | Run states | Phase 0, 7 | ccea/models/state_machines.py | DONE |
| **12. Config Layering** |||||
| 12.1 | Priority order | Phase 0, 5 | packages/agent/daemon/config_manager.py, tests/ccea/phase4/test_config_manager.py | DONE |
| 12.2 | Local hard caps priority | Phase 0, 5 | packages/agent/policy/layered_policy.py | DONE |
| **13-14. Telemetry/Privacy/Residency** |||||
| 13.1 | Telemetry levels | Phase 0, 8 | packages/agent/telemetry/level_manager.py, tests/ccea/phase8/test_level_manager.py | DONE |
| 13.2 | Mandatory redaction | Phase 0, 8 | packages/agent/telemetry/redaction.py, tests/ccea/phase8/test_redaction.py | DONE |
| 14.1 | Data retention | Phase 8 | packages/cloud/governance/retention.py, tests/ccea/phase8/test_retention.py | DONE |
| 14.2 | Data residency | Phase 8 | packages/cloud/governance/residency.py, tests/ccea/phase8/test_residency.py | DONE |
| **15. Security** |||||
| 15.1 | Threat model | Phase 0 | Design_Doc_CCEA_Cloud.md §7 | DONE |
| 15.2 | Safe defaults | Phase 0 | DECISION_LOG.md | DONE |
| 15.3 | Secret hygiene | Phase 0, 8 | packages/agent/telemetry/dlp.py, tests/ccea/phase8/test_dlp.py | DONE |
| **16. Enterprise/Evidence Pack** |||||
| 16.1 | Evidence pack exporter | Phase 9 | packages/cloud/enterprise/evidence_pack.py, tests/ccea/phase9/test_evidence_pack.py | DONE |
| 16.2 | On-prem/VPC deployment | Phase 9 | deploy/helm/ccea-cloud/, tests/ccea/phase9/test_registry_mirror.py | DONE |
| **17-18. AI Act/Not Advice** |||||
| 17.1 | AI Act posture | Phase 11 | docs/CCEA_OVERVIEW.md §3, docs/legal/TERMS_OF_SERVICE.md | DONE |
| 18.1 | "Not advice" disclaimers | Phase 11 | docs/legal/TERMS_OF_SERVICE.md, docs/ui/ONBOARDING_GUARDRAILS.md | DONE |
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
| RT-001 | signature-verification | packages/agent/cloud/signature_verifier.py | tests/ccea/phase5/test_cloud_signature_verifier.py | DONE |
| RT-005 | redaction-middleware | packages/agent/telemetry/redaction.py | tests/ccea/phase8/test_redaction.py | DONE |

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
| §1 Enrollment | Agent enrollment happy path | tests/ccea/phase4/test_cloud_client.py | DONE |
| §1 Enrollment | Token TTL expiration | tests/ccea/phase4/test_cloud_client.py | DONE |
| §2 Deploy+Start | Start with approval | tests/ccea/phase4/test_approval_workflow.py | DONE |
| §2 Deploy+Start | Signature verification failure | tests/ccea/phase5/test_cloud_signature_verifier.py | DONE |
| §3 Upgrade | Upgrade with diff approval | tests/ccea/phase9/test_agent_updates.py | DONE |
| §4 Stop/Pause | Stop without approval | packages/cloud/control_plane/tests/test_phase7_agent_lifecycle.py | DONE |
| §5 Key Rotation | Key rotation with approval | tests/ccea/phase4/test_key_manager.py | DONE |
| §6 Export Logs | Export with redaction | tests/ccea/phase8/test_exporter.py | DONE |
| §7 Kill Switch | Kill switch trigger | tests/ccea/phase5/test_kill_switch.py, tests/ccea/phase5/test_kill_switch_executor.py | DONE |
| §8 Heartbeat | Heartbeat + telemetry | tests/ccea/phase4/test_telemetry_buffer.py, tests/ccea/phase8_p1/test_cloud_client.py | DONE |
| §9 Preflight | Preflight checks | tests/ccea/phase5/test_preflight.py, tests/ccea/phase9_p1/test_preflight_signature_required.py | DONE |

## 6. Module Zone Verification

| Module Pattern | Expected Zone | Verification Check | Status |
|----------------|---------------|-------------------|--------|
| `core_*.py` | SHARED | importlinter.ini, tests/ccea/guardrails/test_import_check.py | DONE |
| `impl_*.py` | SHARED | importlinter.ini, tests/ccea/guardrails/test_import_check.py | DONE |
| `adapters/*/market_data.py` | SHARED | importlinter.ini | DONE |
| `adapters/*/fees.py` | SHARED | importlinter.ini | DONE |
| `adapters/*/order_execution.py` | AGENT | importlinter.ini, packages/cloud/CLOUD_EXCLUSION_MANIFEST.yaml | DONE |
| `adapters/*/options_execution.py` | AGENT | importlinter.ini, packages/cloud/CLOUD_EXCLUSION_MANIFEST.yaml | DONE |
| `execution_providers.py` (live) | AGENT | importlinter.ini | DONE |
| `packages/agent/*` | AGENT | importlinter.ini, ccea/guardrails/import_check.py | DONE |
| `packages/agent/policy/*` | AGENT | importlinter.ini | DONE |
| `packages/cloud/control_plane/*` | CLOUD | importlinter.ini | DONE |
| `packages/cloud/builder/*` | CLOUD | importlinter.ini | DONE |
| `packages/shared/*` | SHARED | importlinter.ini | DONE |

## 7. Decision Log → Test Verification

| Decision ID | Decision | Verification Test | Status |
|-------------|----------|-------------------|--------|
| OQ-001 | Sandbox: process-ok, docker recommended | tests/ccea/phase5/test_sandbox.py, tests/ccea/phase10/test_cloud_sandbox.py | DONE |
| OQ-002 | RAW telemetry: enterprise-only, opt-in | tests/ccea/phase8/test_level_manager.py | DONE |
| OQ-003 | Remote flatten: disabled by default | tests/ccea/phase4/test_layered_policy.py | DONE |
| OQ-004 | Cloud inference: disabled | tests/ccea/phase10/test_design_doc_phase11_compliance.py | DONE |
| OQ-005 | Data classification: 4 levels | tests/ccea/phase8/test_dlp.py | DONE |
| OQ-006 | BYO verification: ToS + attestation | docs/legal/TERMS_OF_SERVICE.md | DONE |
| OQ-007 | Signing: sigstore default, keyful for enterprise | tests/ccea/phase4/test_verifier.py, tests/ccea/phase10/test_crypto.py | DONE |

## 8. Coverage Summary

### Phase 0 Requirements

| Category | Total | Documented | Tested | Coverage |
|----------|-------|------------|--------|----------|
| Design Doc sections covered | 22 | 22 | 22 | 100% |
| Open Questions resolved | 7 | 7 | 7 | 100% |
| JSON Schemas created | 2 | 2 | 2 | 100% |
| Sequence diagrams | 9 | 9 | 9 | 100% |
| CI Guardrails defined | 11 | 11 | 11 | 100% |
| Done criteria met | 10 | 10 | 10 | 100% |

### Phase 1-10 Implementation Status

| Phase | Focus Area | Status | Test Coverage |
|-------|-----------|--------|---------------|
| Phase 1 | Dependencies baseline | DONE | tests/ccea/phase1/ |
| Phase 3 | Strategy API, Protocol guardrails | DONE | tests/ccea/phase3/ (4 test files) |
| Phase 4 | Artifact building, Contracts, Approvals | DONE | tests/ccea/phase4/ (25+ test files) |
| Phase 5 | Agent daemon, Sandbox, Kill switch | DONE | tests/ccea/phase5/ (15 test files) |
| Phase 6 | Legal compliance, Docs structure | DONE | tests/ccea/phase6/ (4 test files) |
| Phase 8 | Telemetry, Alerts, DLP, CMK | DONE | tests/ccea/phase8/ (11 test files) |
| Phase 8_P1 | Cloud client, Order idempotency | DONE | tests/ccea/phase8_p1/ (3 test files) |
| Phase 9 | Enterprise: TUF, Staged rollout, Evidence | DONE | tests/ccea/phase9/ (6 test files) |
| Phase 9_P1 | Cloud artifact builder, Zone distributions | DONE | tests/ccea/phase9_p1/ (4 test files) |
| Phase 10 | Cloud sandbox, Abuse detection, Tenant isolation | DONE | tests/ccea/phase10/ (13 test files) |

### Guardrails Implementation Status

| Category | Total | Implemented | Tested | Coverage |
|----------|-------|-------------|--------|----------|
| Build-time (BT-*) | 4 | 4 | 4 | 100% |
| Pre-merge (PM-*) | 7 | 7 | 7 | 100% |
| Runtime (RT-*) | 2 | 2 | 2 | 100% |
| **Total Guardrails** | **13** | **13** | **13** | **100%** |

### Documentation Status

| Document | Created | Reviewed | Approved |
|----------|---------|----------|----------|
| Design_Doc_CCEA_Cloud.md | Yes | Yes | Yes |
| TARGET_CCEA_ARCHITECTURE.md | Yes | Yes | Yes |
| DECISION_LOG.md | Yes | Yes | Yes |
| CI_GUARDRAILS.md | Yes | Yes | Yes |
| CCEA_SEQUENCE_DIAGRAMS.md | Yes | Yes | Yes |
| CCEA_TRACEABILITY_MATRIX.md | Yes | Yes | Yes |
| CCEA_OVERVIEW.md | Yes | Yes | Yes |
| artifact_manifest.schema.json | Yes | Yes | Yes |
| protocol_messages.schema.json | Yes | Yes | Yes |
| docs/cloud/* | Yes | Yes | Yes |
| docs/agent/* | Yes | Yes | Yes |
| docs/runbooks/* | Yes | Yes | Yes |
| docs/legal/* | Yes | Yes | Yes |

### Business Documentation Alignment (Design Doc CCEA Cloud)

All business documentation has been synchronized with Design Doc CCEA Cloud (2025-12-16):

| Document | CCEA Section Added | Key Additions | Status |
|----------|-------------------|---------------|--------|
| docs/BUSINESS_PLAN_EU_VISA.md | §2.4, §3.4 | Legal Structure with CCEA design commitments, CCEA Architecture section | ✅ Aligned |
| docs/REGULATORY_COMPLIANCE_STRATEGY.md | §1.4, §1.5, §2.1.1-2.1.2 | CCEA Architecture diagram, CCEA Technical Enforcement, DORA/AI Act compliance | ✅ Aligned |
| docs/ENTERPRISE_DEPLOYMENT_ARCHITECTURE.md | Executive Summary, CCEA Zone Overview | CCEA Zone Architecture, Enterprise Deployment Options | ✅ Aligned |
| docs/CYBERSECURITY_FRAMEWORK.md | §1.1 | CCEA Security Model, Threat Model, CI/CD Guardrails | ✅ Aligned |
| docs/DATA_PROTECTION_POLICY.md | §1.1 | CCEA Data Protection Model, Telemetry Redaction Pipeline | ✅ Aligned |
| README.md | Architecture section | CCEA overview, Security Design Commitments, Product Modes | ✅ Aligned |
| claude.md | §CCEA | Legal Posture, Threat Model, Safe Defaults | ✅ Aligned |
| docs/CCEA_OVERVIEW.md | Full document | Comprehensive CCEA documentation | ✅ Aligned |

**Alignment Criteria Met:**
- All documents reference CCEA architecture
- Zone separation (Cloud/Agent) documented in all relevant files
- Security design commitments (no secrets in Cloud, no orders from Cloud) stated consistently
- Legal posture (Software Provider, not Investment Adviser) documented
- Telemetry redaction requirements documented
- CI Guardrails referenced where applicable

### Test Statistics

| Metric | Count |
|--------|-------|
| Total CCEA test files | 117 |
| Test directories | 10 phases |
| Guardrail tests | tests/ccea/guardrails/ |
| Integration tests | packages/cloud/control_plane/tests/ |

---

**Document Control:**
- Author: CCEA Architecture Team
- Last Updated: 2025-12-16
- All Phases Completed: Phase 1-10, Phase 11 (Docs)
- Implementation Status: **Implementation aligns with Design Doc specifications**
