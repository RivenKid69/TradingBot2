# CCEA Decision Log

> **Version**: 2.0.0
> **Date**: 2025-12-16
> **Status**: APPROVED | **All Decisions Implemented**

Этот документ фиксирует решения по Open Questions из Design Doc CCEA Cloud (Section 21).

## Decision Format

Каждое решение содержит:
- **ID**: Уникальный идентификатор
- **Question**: Исходный вопрос
- **Decision**: Принятое решение
- **Rationale**: Обоснование
- **Implications**: Последствия
- **Owner**: Ответственный
- **Date**: Дата решения

---

## OQ-001: Минимальный sandbox для retail agent

**Question:** Docker-required или process-ok? Что делать при отсутствии docker?

**Decision:** Process-ok с fallback, Docker рекомендован.

**Rationale:**
1. Требование Docker создаёт барьер для retail пользователей
2. Process isolation достаточна для базовой безопасности
3. Docker обеспечивает лучшую изоляцию, но опционален

**Implementation:**
```yaml
sandbox:
  mode: "auto"  # auto | docker | process
  fallback_order:
    - docker
    - process
  docker:
    required: false
    recommended: true
  process:
    cpu_limit: 2
    memory_limit_mb: 4096
    timeout_s: 3600
```

**Implications:**
- При отсутствии Docker: предупреждение пользователю, работа в process mode
- Enterprise: Docker обязателен
- Документация должна рекомендовать Docker

**Owner:** Platform Team
**Date:** 2025-12-13

---

## OQ-002: Политика RAW order telemetry

**Question:** Кто может использовать RAW order telemetry, когда, по умолчанию?

**Decision:** Enterprise-only, opt-in, выключено по умолчанию.

**Rationale:**
1. RAW order data содержит sensitive информацию
2. GDPR требует минимизации данных
3. Retail пользователям AGGREGATED достаточно
4. Enterprise нужны детальные данные для compliance

**Implementation:**
```yaml
telemetry:
  levels:
    AGGREGATED:
      default: true
      available_to: ["retail", "enterprise"]
      description: "PnL, win rate, drawdown (no trade details)"

    DETAILED_NON_SENSITIVE:
      default: false
      available_to: ["retail", "enterprise"]
      requires_opt_in: true
      description: "Trade counts, timing, latency"

    RAW_ORDER_EVENTS:
      default: false
      available_to: ["enterprise"]
      requires_opt_in: true
      requires_contract: true
      description: "Full order details, fills, modifications"
```

**Implications:**
- Retail: только AGGREGATED и DETAILED
- Enterprise: все уровни по контракту
- UI должен явно показывать уровень телеметрии

**Owner:** Privacy Team
**Date:** 2025-12-13

---

## OQ-003: Flatten позиции удалённо

**Question:** Разрешаем ли flatten как удалённый request?

**Decision:** По умолчанию только локально; enterprise — по договору.

**Rationale:**
1. Flatten = торговая операция = должна быть локальной
2. Cloud не должен иметь возможность исполнять ордера
3. Enterprise может требовать remote flatten для risk management
4. Всегда требуется local approval или explicit policy

**Implementation:**
```yaml
flatten_policy:
  retail:
    remote_flatten: disabled
    local_flatten: enabled
    triggers:
      - kill_switch
      - manual_ui

  enterprise:
    remote_flatten: by_contract  # requires explicit contract clause
    local_flatten: enabled
    remote_requires:
      - break_glass_reason
      - dual_approval
      - audit_log
```

**Implications:**
- Retail: flatten только через local UI или kill switch
- Enterprise: remote flatten требует dual approval
- Все flatten операции логируются

**Owner:** Security Team
**Date:** 2025-12-13

---

## OQ-004: Remote brain, local finger (cloud inference)

**Question:** Разрешаем ли cloud inference для принятия торговых решений?

**Decision:** Не делаем в этой итерации.

**Rationale:**
1. Создаёт зависимость от cloud для live trading
2. Усложняет архитектуру безопасности
3. Latency concerns для HFT
4. Можно добавить позже как опцию

**Implementation:**
```yaml
cloud_inference:
  enabled: false
  rationale: "Deferred to future iteration"
  future_considerations:
    - latency_requirements
    - fallback_strategy
    - security_implications
```

**Implications:**
- Все inference происходит локально в Agent
- Model artifacts загружаются из Registry
- Нет realtime dependency на Cloud для trading decisions

**Owner:** Architecture Team
**Date:** 2025-12-13

---

## OQ-005: Data sensitivity классификация

**Question:** Какие поля персональные/чувствительные, что считается trading-sensitive/IP?

**Decision:** Явная классификация с 4 уровнями.

**Classification:**

### Level 1: PUBLIC
Данные, которые можно свободно передавать:
- Рыночные цены (public)
- Торговые часы
- Fee structures
- Exchange info

### Level 2: INTERNAL
Данные для внутреннего использования:
- Aggregated performance metrics
- Strategy configurations (non-secret)
- Deployment states

### Level 3: SENSITIVE
Требует защиты и redaction:
- Account identifiers (masked)
- IP addresses
- Trade counts/volumes
- Position sizes (aggregated)
- User IDs

### Level 4: SECRET
Никогда не покидает Agent:
- Broker API keys
- Master vault keys
- Private keys
- Raw order details (fills, modifications)

**Implementation:**
```yaml
data_classification:
  PUBLIC:
    retention: unlimited
    redaction: none
    can_export: true

  INTERNAL:
    retention: 7_years
    redaction: none
    can_export: with_approval

  SENSITIVE:
    retention: 5_years
    redaction: required
    can_export: redacted_only

  SECRET:
    retention: local_only
    redaction: full_mask
    can_export: never
```

**Implications:**
- Telemetry pipeline применяет классификацию
- Export tools автоматически redact
- Audit trail для доступа к SENSITIVE+

**Owner:** Privacy Team
**Date:** 2025-12-13

---

## OQ-006: Доказательство customer-managed host

**Question:** Что считаем достаточным доказательством "customer-managed host" в BYO/VPS сценарии?

**Decision:** Процесс онбординга + attestation + ToS acceptance.

**Implementation:**

### Retail (BYO Agent):
```yaml
byo_verification:
  required:
    - tos_acceptance: true
    - self_attestation: "I confirm this is my own infrastructure"
  optional:
    - infrastructure_type: ["local_machine", "vps", "cloud_vm"]
```

### Enterprise (VPS/Self-hosted):
```yaml
enterprise_verification:
  required:
    - contract_signed: true
    - infrastructure_attestation: true
    - security_questionnaire: completed
    - designated_contact: true
  verification_methods:
    - ip_allowlist
    - domain_verification
    - organization_verification
```

**Onboarding Flow:**
1. User creates account
2. User acknowledges ToS (includes BYO clause)
3. User downloads/deploys Agent
4. Agent enrolls with attestation
5. First deployment requires explicit approval

**Implications:**
- Мы не несём ответственности за customer infrastructure
- ToS явно указывает на customer responsibility
- Audit trail фиксирует attestation

**Owner:** Legal + Product Team
**Date:** 2025-12-13

---

## OQ-007: Key management для artifact signing

**Question:** Как храним/ротируем ключи подписи? Keyless sigstore vs keyful?

**Decision:** Dual approach: keyless sigstore для public, keyful для enterprise/offline.

**Implementation:**
```yaml
artifact_signing:
  default: sigstore_keyless

  sigstore_keyless:
    description: "OIDC-based signing via Sigstore"
    use_cases: ["public_artifacts", "saas"]
    benefits:
      - no_key_management
      - transparency_log
      - automatic_rotation

  keyful:
    description: "Traditional key-based signing"
    use_cases: ["enterprise", "air_gapped", "offline"]
    key_storage:
      - hsm
      - kms
      - encrypted_file
    rotation_policy:
      frequency: yearly
      on_compromise: immediate

  trust_root:
    sigstore: "fulcio root CA"
    enterprise: "customer-provided root or our enterprise CA"
```

**Implications:**
- SaaS: sigstore keyless по умолчанию
- Enterprise: поддержка customer CA
- Agent: проверяет подпись против trust root

**Owner:** Security Team
**Date:** 2025-12-13

---

## Default Values Summary

| Setting | Default Value | Rationale |
|---------|--------------|-----------|
| Sandbox mode | `auto` (prefer docker) | Balance accessibility/security |
| RAW telemetry | `disabled` | Privacy by default |
| Remote flatten | `disabled` | Security by default |
| Cloud inference | `disabled` | Simplicity, no dependency |
| Redaction | `enabled` (cannot disable) | Mandatory protection |
| Local approval | `required` for trading_impacting | User control |
| Auto-approve | `disabled` | Explicit consent |
| Signing method | `sigstore_keyless` | Modern, no key management |

---

## Threat Model Decisions

### TM-001: RCE in Cloud

**Mitigation:** Cloud build не содержит trading libraries.

**Implementation:**
- CI check: `no-trading-libs-in-cloud`
- Build-time dependency scan
- Runtime: no broker connectors

### TM-002: Key Exfiltration

**Mitigation:** Keys never leave Agent.

**Implementation:**
- Vault only in Agent
- Redaction mandatory
- No key backup to Cloud

### TM-003: Artifact Tampering

**Mitigation:** Digest pinning + signature verification.

**Implementation:**
- Artifacts stored by digest
- Signature required for publish
- Agent rejects unsigned

### TM-004: Cloud Becomes Execution

**Mitigation:** No order-like payloads in protocol.

**Implementation:**
- Schema prohibits side/qty/price
- CI check for schema
- Runtime validation

### TM-005: Replay Attacks

**Mitigation:** Idempotency keys + timestamps.

**Implementation:**
- Unique idempotency_key per command
- Timestamp validation
- Dedup in Agent

---

## Change Log

| Date | Decision | Change |
|------|----------|--------|
| 2025-12-13 | OQ-001 | Initial decision |
| 2025-12-13 | OQ-002 | Initial decision |
| 2025-12-13 | OQ-003 | Initial decision |
| 2025-12-13 | OQ-004 | Initial decision |
| 2025-12-13 | OQ-005 | Initial decision |
| 2025-12-13 | OQ-006 | Initial decision |
| 2025-12-13 | OQ-007 | Initial decision |

---

**Document Control:**
- Author: CCEA Architecture Team
- Reviewers: Security, Legal, Privacy, Engineering
- Approval: Architecture Review Board
- Last Updated: 2025-12-16
- Implementation Status: **100% Design Doc Compliance**
