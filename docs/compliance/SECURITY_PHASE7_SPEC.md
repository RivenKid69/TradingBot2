# Security Controls & Breach Workflow - Phase 7 Specification

**Version**: 1.0.0
**Date**: 2025-12-17
**Status**: Implementation
**GDPR Articles**: Art. 32 (Security of Processing), Art. 33-34 (Breach Notification)

## 1. Overview

This specification defines the implementation of GDPR Phase 7: Security Controls and Breach Workflow for the CCEA (Cloud-Controlled Execution Architecture) platform. It covers:

1. **Security Baseline** (Art. 32) - Encryption, key management, MFA, secrets management
2. **Supply Chain Security** (Design Doc 15.1) - Signed artifacts, digest pinning, SBOM
3. **Agent Update Management** (Design Doc 15.2) - Staged rollout, rollback, version pinning
4. **Research Sandbox Isolation** (Design Doc 15.3) - Sandboxing, quotas, egress control, abuse detection
5. **Breach Notification Workflow** (Art. 33-34) - Decision tree, 72h notification, tabletop exercises
6. **Evidence Pack Export** - Comprehensive audit artifacts for compliance

## 2. GDPR Article 32 Requirements

### 2.1 Security of Processing (Art. 32(1))

The controller and processor shall implement appropriate technical and organisational measures to ensure a level of security appropriate to the risk, including as appropriate:

| Requirement | Implementation |
|-------------|----------------|
| **(a)** Pseudonymisation and encryption of personal data | Encryption at rest (AES-256), in transit (TLS 1.3), key management with rotation |
| **(b)** Confidentiality, integrity, availability, resilience | RBAC, audit trails, hash chains, backup/recovery procedures |
| **(c)** Ability to restore access to data | Backup verification, recovery testing, documented procedures |
| **(d)** Regular testing and evaluation | Security assessments, penetration testing, tabletop exercises |

### 2.2 GDPR Article 33 - Notification to Supervisory Authority

- **Deadline**: 72 hours from becoming aware of a personal data breach
- **Exception**: Unless unlikely to result in a risk to rights and freedoms
- **Content**: Nature of breach, categories/numbers affected, DPO contact, likely consequences, remediation measures

### 2.3 GDPR Article 34 - Communication to Data Subject

- **Trigger**: When breach is likely to result in high risk to rights and freedoms
- **Exceptions**: Encryption renders data unintelligible, subsequent measures prevent high risk, disproportionate effort (use public communication)

## 3. Security Baseline Service

### 3.1 Encryption Controls

```python
@dataclass
class EncryptionConfig:
    algorithm: EncryptionAlgorithm  # AES_256_GCM, CHACHA20_POLY1305
    key_derivation: KeyDerivation   # PBKDF2, ARGON2, SCRYPT
    key_rotation_days: int          # Default: 90
    at_rest_enabled: bool           # Always True
    in_transit_min_tls: str         # "1.3"
```

**Encryption Requirements**:
- All data at rest: AES-256-GCM encryption
- All data in transit: TLS 1.3 minimum
- Key rotation: Every 90 days (configurable)
- Key derivation: PBKDF2 with 100,000+ iterations or Argon2id

### 3.2 Key Management

```python
class KeyManagementPolicy(str, Enum):
    PLATFORM_MANAGED = "platform_managed"      # Cloud-managed keys
    CUSTOMER_MANAGED = "customer_managed"      # BYOK/CMK
    HSM_BACKED = "hsm_backed"                  # Hardware security module
```

**Key Management Requirements**:
- Master keys stored in HSM or cloud KMS
- Data encryption keys (DEK) wrapped by master keys
- Key rotation audit trail
- Emergency key revocation procedure
- Key access logging

### 3.3 MFA Enforcement

```python
class MFARequirement(str, Enum):
    DISABLED = "disabled"           # Not required (non-sensitive)
    OPTIONAL = "optional"           # User choice
    REQUIRED_SENSITIVE = "required_sensitive"  # For sensitive resources
    REQUIRED_ALWAYS = "required_always"        # Always required
```

**MFA Policy**:
- Required for: Break-glass access, admin operations, DSAR processing
- Methods: TOTP, WebAuthn/FIDO2, SMS (deprecated), Hardware tokens
- Session timeout: 8 hours standard, 1 hour for elevated access

### 3.4 Secrets Management

```python
@dataclass
class SecretPolicy:
    min_length: int = 16
    complexity_required: bool = True
    rotation_days: int = 90
    expiry_warning_days: int = 14
    storage_backend: str = "vault"  # vault, kms, sealed_secrets
```

**Secrets Requirements**:
- No secrets in code, config files, or logs (CCEA constraint)
- Secrets stored in dedicated vault (HashiCorp Vault, AWS Secrets Manager)
- Automatic rotation support
- Secret access audit logging

## 4. Supply Chain Security Service

### 4.1 Artifact Signing

```python
@dataclass
class SignedArtifact:
    artifact_id: str
    digest: str                    # sha256:hexdigest
    signature: str                 # Base64-encoded signature
    signer_id: str                 # Key/identity that signed
    signing_timestamp: datetime
    algorithm: SigningAlgorithm    # RSA_PSS_SHA256, ED25519, ECDSA_P256
    certificate_chain: List[str]   # For X.509 verification
```

**Signing Requirements**:
- All artifacts must be signed before deployment
- Unsigned artifacts are rejected (fail closed)
- Signature verification at deploy time
- Supported algorithms: RSA-PSS-SHA256, Ed25519, ECDSA-P256

### 4.2 Digest Pinning

```python
@dataclass
class DigestPin:
    artifact_type: ArtifactType    # CONTAINER_IMAGE, CONFIG_BLOB, BINARY
    artifact_name: str
    pinned_digest: str             # sha256:hexdigest (NEVER "latest")
    pinned_at: datetime
    pinned_by: str
    expires_at: Optional[datetime]
```

**Pinning Policy**:
- All artifact references by digest ONLY (no "latest", no tags)
- Digest verification at pull/deploy time
- Pin expiration with mandatory review
- Change audit trail

### 4.3 Registry Allowlist

```python
@dataclass
class RegistryAllowlist:
    allowed_registries: Set[str]   # e.g., {"gcr.io/project", "ecr.eu-west-1"}
    denied_registries: Set[str]    # Explicit blocklist
    require_signature: bool        # All artifacts must be signed
    require_sbom: bool             # SBOM must accompany artifact
```

**Registry Policy**:
- Only allowlisted registries permitted
- Default deny for unknown registries
- EU-only registry locations enforced
- Mirror support for air-gapped deployments

### 4.4 SBOM Management

```python
@dataclass
class SBOM:
    sbom_id: str
    artifact_digest: str           # Links to signed artifact
    format: SBOMFormat             # SPDX, CYCLONEDX
    version: str
    created_at: datetime
    components: List[SBOMComponent]
    vulnerabilities: List[Vulnerability]  # CVE references
    integrity_hash: str            # sha256 of SBOM content
```

**SBOM Requirements**:
- SBOM required for all deployable artifacts
- Stored with artifact digest reference
- Exportable for evidence pack
- Vulnerability tracking (CVE references)
- SPDX or CycloneDX format

## 5. Agent Update Service

### 5.1 Update Signing

```python
@dataclass
class AgentUpdate:
    update_id: str
    version: str                   # semver
    artifact_digest: str
    signature: str
    release_notes: str
    min_agent_version: str         # Compatibility
    max_agent_version: Optional[str]
    published_at: datetime
    published_by: str
```

**Update Signing Requirements**:
- All agent updates cryptographically signed
- Agents reject unsigned updates
- Signature verification before installation
- Key pinning for update verification keys

### 5.2 Staged Rollout

```python
@dataclass
class RolloutStage:
    stage_id: str
    stage_name: str               # e.g., "canary", "early_adopter", "general"
    percentage: int               # 0-100
    started_at: datetime
    criteria: RolloutCriteria     # Conditions to proceed
    status: RolloutStatus         # PENDING, IN_PROGRESS, COMPLETED, FAILED, PAUSED
```

```python
@dataclass
class RolloutPlan:
    rollout_id: str
    update_id: str
    stages: List[RolloutStage]
    auto_proceed: bool            # Auto-advance on success
    pause_on_error_rate: float    # Pause if error rate exceeds (e.g., 0.01 = 1%)
    rollback_on_critical: bool    # Auto-rollback on critical errors
```

**Staged Rollout Policy**:
- Default stages: canary (1%) → early (10%) → general (100%)
- Automatic pause on error rate threshold
- Manual approval option between stages
- Metrics collection at each stage

### 5.3 Rollback Support

```python
@dataclass
class RollbackRequest:
    rollback_id: str
    update_id: str                # Update to rollback from
    target_version: str           # Version to rollback to
    reason: str
    requested_by: str
    requested_at: datetime
    scope: RollbackScope          # ALL, FAILED_ONLY, SPECIFIC_AGENTS
    affected_agents: Optional[List[str]]
```

**Rollback Requirements**:
- Previous version retained for rollback
- Rollback can be triggered manually or automatically
- Rollback audit trail
- Notification to affected tenants

### 5.4 Enterprise Version Pinning

```python
@dataclass
class VersionPin:
    workspace_id: str
    pinned_version: str
    pinned_at: datetime
    pinned_by: str
    reason: str
    expires_at: Optional[datetime]
    change_window: Optional[ChangeWindow]  # When updates allowed
```

```python
@dataclass
class ChangeWindow:
    allowed_days: Set[int]        # 0=Monday, 6=Sunday
    start_hour: int               # UTC hour (0-23)
    end_hour: int                 # UTC hour (0-23)
    timezone: str                 # e.g., "Europe/London"
```

**Enterprise Controls**:
- Version pinning by workspace
- Change windows for update application
- Pin expiration with mandatory review
- Audit trail for pin changes

## 6. Research Sandbox Service

### 6.1 Sandbox Isolation

```python
class IsolationLevel(str, Enum):
    CONTAINER = "container"        # Container-level isolation
    VM = "vm"                      # VM-level isolation
    FIRECRACKER = "firecracker"    # microVM isolation
    KATA = "kata"                  # Kata containers
```

```python
@dataclass
class SandboxConfig:
    isolation_level: IsolationLevel
    seccomp_profile: str           # Path to seccomp profile
    capabilities_drop: List[str]   # Linux capabilities to drop
    read_only_rootfs: bool
    no_new_privileges: bool
    user_namespace: bool
```

**Isolation Requirements**:
- Research jobs run in isolated sandbox
- No access to host filesystem (except designated mounts)
- No network access to internal services
- No privilege escalation possible

### 6.2 Resource Quotas

```python
@dataclass
class ResourceQuota:
    cpu_limit: float              # CPU cores (e.g., 2.0)
    cpu_request: float            # Guaranteed CPU
    memory_limit_mb: int          # Hard limit in MB
    memory_request_mb: int        # Guaranteed memory
    disk_limit_mb: int            # Ephemeral storage
    gpu_limit: int                # Number of GPUs (0 = none)
    max_runtime_seconds: int      # Job timeout
    max_processes: int            # PID limit
    max_open_files: int           # File descriptor limit
```

**Quota Policy**:
- Per-workspace quota limits
- Per-job resource allocation
- Automatic termination on limit breach
- Resource usage monitoring and alerting

### 6.3 Egress Allowlist

```python
@dataclass
class EgressPolicy:
    policy_id: str
    workspace_id: str
    allowed_domains: Set[str]     # e.g., {"api.polygon.io", "data.nasdaq.com"}
    allowed_ips: Set[str]         # CIDR notation
    allowed_ports: Set[int]       # e.g., {443, 80}
    deny_all_by_default: bool     # True = allowlist mode
    log_all_connections: bool
```

**Egress Control**:
- Default deny for all outbound connections
- Allowlist for approved data sources
- Connection logging for audit
- Rate limiting per destination

### 6.4 Abuse Detection

```python
class AbuseType(str, Enum):
    CRYPTO_MINING = "crypto_mining"
    BOTNET_ACTIVITY = "botnet_activity"
    NETWORK_SCANNING = "network_scanning"
    DATA_EXFILTRATION = "data_exfiltration"
    RESOURCE_ABUSE = "resource_abuse"
    CREDENTIAL_STUFFING = "credential_stuffing"
```

```python
@dataclass
class AbuseEvent:
    event_id: str
    workspace_id: str
    job_id: str
    abuse_type: AbuseType
    confidence: float             # 0.0 - 1.0
    detected_at: datetime
    indicators: List[str]         # What triggered detection
    action_taken: AbuseAction     # ALERT, THROTTLE, TERMINATE, BLOCK
    evidence_hash: str
```

**Detection Mechanisms**:
- CPU pattern analysis (crypto mining signatures)
- Network behavior analysis (scanning, C2 communication)
- Resource consumption anomalies
- Known malicious pattern matching
- Automatic response actions

## 7. Breach Notification Workflow

### 7.1 Breach Assessment

```python
class BreachSeverity(str, Enum):
    LOW = "low"                   # Unlikely risk to individuals
    MEDIUM = "medium"             # Some risk, monitor situation
    HIGH = "high"                 # Likely risk, notification required
    CRITICAL = "critical"         # High risk, immediate notification
```

```python
class BreachCategory(str, Enum):
    CONFIDENTIALITY = "confidentiality"  # Unauthorized disclosure
    INTEGRITY = "integrity"              # Unauthorized modification
    AVAILABILITY = "availability"        # Loss of access
```

```python
@dataclass
class BreachAssessment:
    breach_id: str
    detected_at: datetime
    assessed_at: datetime
    assessed_by: str
    category: BreachCategory
    severity: BreachSeverity
    data_categories_affected: List[str]
    individuals_affected_count: Optional[int]
    individuals_affected_estimate: str   # e.g., "100-500"
    cross_border: bool                   # Affects multiple EU countries
    contains_special_categories: bool    # Art. 9 data
    likely_consequences: List[str]
    mitigating_factors: List[str]
    risk_score: float                    # 0.0 - 10.0
```

### 7.2 Notification Decision Tree

```python
@dataclass
class NotificationDecision:
    decision_id: str
    breach_id: str
    authority_notification_required: bool
    authority_notification_deadline: datetime  # 72h from awareness
    subject_notification_required: bool
    subject_notification_reason: Optional[str]
    exemption_applied: Optional[str]          # Art. 34(3) exemption
    decision_rationale: str
    decided_by: str
    decided_at: datetime
    approved_by: Optional[str]
    evidence_hash: str
```

**Decision Criteria**:

| Condition | Authority (Art. 33) | Subject (Art. 34) |
|-----------|--------------------|--------------------|
| Risk score < 3.0 | No (document) | No |
| Risk score 3.0-6.0 | Yes (72h) | Evaluate |
| Risk score > 6.0 | Yes (72h) | Yes |
| Encrypted data only | May be exempt | Exempt (Art. 34(3)(a)) |
| Subsequent measures remove risk | Yes (72h) | May be exempt (Art. 34(3)(b)) |
| Special category data | Yes (72h) | Yes |

### 7.3 Notification Templates

```python
@dataclass
class AuthorityNotification:
    notification_id: str
    breach_id: str
    supervisory_authority: str         # e.g., "ICO", "CNIL", "BfDI"
    nature_of_breach: str
    categories_of_data: List[str]
    approximate_number_of_subjects: str
    approximate_number_of_records: str
    dpo_contact: DPOContact
    likely_consequences: List[str]
    measures_taken: List[str]
    measures_proposed: List[str]
    submitted_at: Optional[datetime]
    submission_reference: Optional[str]
```

```python
@dataclass
class SubjectNotification:
    notification_id: str
    breach_id: str
    notification_method: str           # email, postal, public
    plain_language_description: str
    dpo_contact: DPOContact
    likely_consequences: List[str]
    measures_taken: List[str]
    recommendations_for_subject: List[str]
    sent_at: Optional[datetime]
```

### 7.4 Breach Response Timeline

```python
@dataclass
class BreachTimeline:
    breach_id: str
    events: List[TimelineEvent]
    current_status: BreachStatus
    authority_deadline: datetime
    authority_notified_at: Optional[datetime]
    subjects_notified_at: Optional[datetime]
    resolution_target: Optional[datetime]
    resolved_at: Optional[datetime]
```

**Required Timeline Events**:
1. `DETECTED` - Breach first detected
2. `AWARENESS` - Became aware of breach (starts 72h clock)
3. `ASSESSMENT_STARTED` - Risk assessment begun
4. `ASSESSMENT_COMPLETED` - Risk assessment completed
5. `DECISION_MADE` - Notification decision made
6. `AUTHORITY_NOTIFIED` - Supervisory authority notified
7. `SUBJECTS_NOTIFIED` - Data subjects notified (if required)
8. `CONTAINMENT_COMPLETED` - Breach contained
9. `REMEDIATION_COMPLETED` - Remediation measures implemented
10. `RESOLVED` - Breach fully resolved

### 7.5 Tabletop Exercises

```python
@dataclass
class TabletopExercise:
    exercise_id: str
    scenario_name: str
    scenario_description: str
    conducted_at: datetime
    participants: List[str]
    duration_minutes: int
    breach_type: BreachCategory
    severity_simulated: BreachSeverity
    timeline_achieved: TabletopTimeline
    gaps_identified: List[str]
    improvements_recommended: List[str]
    next_exercise_due: datetime
    evidence_hash: str
```

**Tabletop Requirements**:
- Quarterly tabletop exercises
- Simulate various breach scenarios
- Document gaps and improvements
- Produce draft notification within 24h
- Store as evidence artifact

## 8. Evidence Pack Service

### 8.1 Evidence Categories

```python
class EvidenceCategory(str, Enum):
    ARTIFACT_INVENTORY = "artifact_inventory"
    SBOM = "sbom"
    CHANGE_JOURNAL = "change_journal"
    ROLLOUT_RECORDS = "rollout_records"
    SANDBOX_POLICIES = "sandbox_policies"
    SANDBOX_VIOLATIONS = "sandbox_violations"
    SECURITY_CONTROLS = "security_controls"
    BREACH_RECORDS = "breach_records"
    TABLETOP_REPORTS = "tabletop_reports"
    ACCESS_AUDIT = "access_audit"
    KEY_ROTATION = "key_rotation"
```

### 8.2 Evidence Pack Export

```python
@dataclass
class EvidencePack:
    pack_id: str
    workspace_id: str
    exported_at: datetime
    exported_by: str
    period_start: datetime
    period_end: datetime
    categories: Set[EvidenceCategory]
    artifacts: List[EvidenceArtifact]
    manifest: EvidenceManifest
    integrity_hash: str           # SHA-256 of entire pack
    signature: Optional[str]      # Optional signing
```

```python
@dataclass
class EvidenceArtifact:
    artifact_id: str
    category: EvidenceCategory
    filename: str
    content_type: str             # application/json, text/csv
    size_bytes: int
    content_hash: str
    created_at: datetime
```

```python
@dataclass
class EvidenceManifest:
    manifest_version: str
    pack_id: str
    exported_at: datetime
    artifact_count: int
    artifacts: List[Dict[str, str]]  # id -> hash mapping
    total_size_bytes: int
    manifest_hash: str
```

### 8.3 Export Contents

**Artifact Inventory Export**:
- All deployed artifact versions
- Digest references
- Signatures and certificates
- Deployment timestamps

**SBOM Export**:
- Component lists per artifact
- Vulnerability assessments
- License information
- Dependency trees

**Change Journal Export**:
- All deploy/upgrade/rollback events
- Approval records
- Config blob digests
- Who requested/approved

**Rollout Records Export**:
- Staged rollout plans
- Stage progression history
- Error rates and metrics
- Rollback events

**Sandbox Policy Export**:
- Egress allowlists
- Resource quotas
- Isolation configurations
- Policy change history

**Sandbox Violations Export**:
- Abuse detection events
- Actions taken
- Evidence of violations
- Resolution records

## 9. Integration Points

### 9.1 Service Dependencies

```
SecurityBaselineService
    ↓ provides encryption/key status
    ↓
SupplyChainService
    ↓ validates artifacts
    ↓
AgentUpdateService
    ↓ uses signed artifacts
    ↓
ResearchSandboxService
    ↓ enforces policies
    ↓
BreachWorkflowService
    ← uses all services for breach context
    ↓
EvidencePackService
    ← exports from all services
    ↓
AccessAuditService (from Phase 6)
    ← logs all operations
```

### 9.2 Callback Integration

All Phase 7 services integrate with:
- `AccessAuditService` for audit logging
- `RBACService` for authorization
- Alert systems for notifications

## 10. API Endpoints (Reference)

```
# Security Baseline
GET  /api/v1/security/status
GET  /api/v1/security/encryption
POST /api/v1/security/keys/rotate
GET  /api/v1/security/mfa/status

# Supply Chain
GET  /api/v1/supply-chain/artifacts
GET  /api/v1/supply-chain/artifacts/{digest}
POST /api/v1/supply-chain/artifacts/verify
GET  /api/v1/supply-chain/sbom/{digest}
GET  /api/v1/supply-chain/registries

# Agent Updates
GET  /api/v1/updates/available
POST /api/v1/updates/rollout
GET  /api/v1/updates/rollout/{rollout_id}
POST /api/v1/updates/rollback
GET  /api/v1/updates/pins
POST /api/v1/updates/pins

# Research Sandbox
GET  /api/v1/sandbox/policies
GET  /api/v1/sandbox/quotas
GET  /api/v1/sandbox/egress
GET  /api/v1/sandbox/violations
POST /api/v1/sandbox/jobs/{job_id}/terminate

# Breach Workflow
POST /api/v1/breach/report
GET  /api/v1/breach/{breach_id}
POST /api/v1/breach/{breach_id}/assess
POST /api/v1/breach/{breach_id}/notify
GET  /api/v1/breach/{breach_id}/timeline
GET  /api/v1/breach/tabletops

# Evidence Pack
POST /api/v1/evidence/export
GET  /api/v1/evidence/packs
GET  /api/v1/evidence/packs/{pack_id}
GET  /api/v1/evidence/packs/{pack_id}/download
```

## 11. Testing Requirements

### 11.1 Unit Tests

- Encryption/decryption operations
- Key rotation logic
- Signature verification
- Digest validation
- SBOM parsing
- Rollout stage transitions
- Rollback procedures
- Quota enforcement
- Egress filtering
- Abuse detection patterns
- Breach assessment scoring
- Timeline management
- Evidence export

### 11.2 Integration Tests

- End-to-end artifact signing and verification
- Staged rollout with simulated failures
- Sandbox isolation verification
- Breach simulation with notification generation
- Evidence pack completeness

### 11.3 Security Tests

- Signature bypass attempts
- Unsigned artifact rejection
- Sandbox escape attempts
- Quota bypass attempts
- Egress circumvention
- Key exposure prevention

## 12. Definition of Done

- [ ] A simulated breach produces complete notification decision package within 24h
- [ ] Evidence trail maintained for all breach response actions
- [ ] 72h authority notification deadline tracked and enforced
- [ ] Evidence pack can export: signed artifact inventory + SBOM + change journal + rollout records + sandbox policies/violations
- [ ] All services have 100% test coverage
- [ ] No regressions in existing governance tests
- [ ] Documentation complete (SOP, Art. 32 checklist)

## 13. References

- GDPR Regulation (EU) 2016/679 - Articles 32, 33, 34
- EDPB Guidelines on personal data breach notification (WP250 rev.01)
- Design Doc CCEA_CLOUD - Sections 15.1, 15.2, 15.3
- ISO/IEC 27001:2022 - Information security controls
- NIST CSF 2.0 - Respond and Recover functions
- OWASP Software Component Verification Standard (SCVS)
- SLSA (Supply-chain Levels for Software Artifacts)
