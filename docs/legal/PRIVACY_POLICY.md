# Privacy Policy

**CustodiaCloud**

**Effective Date:** December 2024
**Version:** 3.0.0

**Legal Framework:** General Data Protection Regulation (EU) 2016/679 (GDPR)

**Architecture:** Cloud-Controlled Execution Architecture (CCEA)

---

## 1. DATA CONTROLLER

### 1.1 Identity

The data controller responsible for your personal data is:

**[Company Name]** *(template; finalize upon entity formation)*
- **Registered Address:** [Address]
- **Registration Number:** [Number]
- **Country:** [Final jurisdiction TBD upon entity formation]

### 1.2 Contact Information

For privacy-related inquiries:

- **Email:** privacy@[company-domain].com
- **Data Protection Officer (DPO):** dpo@[company-domain].com
- **Postal Address:** [Company Address]

### 1.3 EU Representative

For Users outside the EU, our EU representative is:
- [Representative details if applicable]

---

## 1A. CCEA ARCHITECTURE AND DATA ZONES

### 1A.1 Architectural Overview

**IMPORTANT - DATA RESIDENCY BY DESIGN (design intent; implementation may vary by deployment):**

The Platform implements **CCEA (Cloud-Controlled Execution Architecture)**, which is designed to separate data processing between Cloud and Agent zones:

| Zone | Data Processed | Credentials Access | Your Sensitive Data |
|------|---------------|-------------------|---------------------|
| **Cloud** | Research, backtesting, monitoring | **None (by design)** | Aggregated telemetry only |
| **Agent** | Execution, risk enforcement | **YES (Local Only)** | Designed to stay on YOUR hardware |

### 1A.2 Cloud Zone Data (Processed by Us)

Data processed in our Cloud infrastructure:

| Data Type | Processing Location (design target) | Retention (target) |
|-----------|-------------------|-----------|
| Account information | Cloud (EU region by design) | Until deletion |
| Research/backtest jobs | Cloud (EU region by design) | Per retention policy |
| Strategy source code | Cloud (EU region by design) | Until deletion |
| Aggregated telemetry | Cloud (EU region by design) | 90 days (target) |
| Audit logs | Cloud (EU region by design) | 5 years (target) |

**Cloud is designed not to receive or process (enforced via CCEA architecture):**
- Broker API keys or secrets (secrets stay in customer-controlled Agent)
- Trading credentials of any kind
- Order details (side, quantity, price)
- Real-time position data (only aggregated metrics)

### 1A.3 Agent Zone Data (Processed Locally on Your Hardware)

Data processed by the optional local Agent (if you deploy one):

| Data Type | Storage Location | Our Access |
|-----------|-----------------|------------|
| Broker API credentials | Your hardware (encrypted vault) | **NONE** |
| Order execution data | Your hardware | **NONE** |
| Position details | Your hardware | **NONE** |
| Local approval records | Your hardware | **NONE** |

**Agent-zone data is designed to remain under your control:**
1. By design, Agent-zone data is intended to stay on your hardware (verify via architecture review)
2. Encryption is designed to use keys derived locally, which we are designed not to possess (verify via threat model/security review)
3. The Agent is designed to operate autonomously on your infrastructure

### 1A.4 Telemetry Redaction

Telemetry transmitted from Agent to Cloud is redacted/bucketed according to the Platform's telemetry controls:

| Original Data | Transmitted Data | Redaction |
|--------------|-----------------|-----------|
| `symbol: "AAPL"` | `symbol: "[REDACTED]"` | Mandatory |
| `quantity: 1.5` | Not transmitted | Blocked |
| `price: 50000` | Not transmitted | Blocked |
| `equity: 100500.25` | `equity_bucket: "100K-500K"` | Bucketed |
| `pnl: 5.23%` | `pnl_percent: 5.2` | Rounded |

Redaction is enforced by schema/controls for data transmitted to Cloud.

---

## 2. DATA WE COLLECT

### 2.1 Account Data

When you create an account, we collect:

| Data Type | Purpose | Legal Basis |
|-----------|---------|-------------|
| Email address | Account identification, communication | Contract (Art. 6(1)(b)) |
| Name (optional) | Personalization | Consent (Art. 6(1)(a)) |
| Password (hashed) | Authentication | Contract (Art. 6(1)(b)) |
| Account creation date | Record keeping | Contract (Art. 6(1)(b)) |

### 2.2 Strategy and Research Data

When you use research/backtesting features, we process:

| Data Type | Purpose | Legal Basis |
|-----------|---------|-------------|
| Strategies (code/config) | Service provision, backtesting | Contract (Art. 6(1)(b)) |
| Backtest/simulation results | Research workflows | Contract (Art. 6(1)(b)) |
| Aggregated telemetry (if enabled) | Monitoring and reliability | Contract (Art. 6(1)(b)) |

### 2.3 Broker Credentials (CCEA Architecture)

**IMPORTANT - CCEA ARCHITECTURE: CREDENTIALS ARE DESIGNED TO STAY LOCAL**

Under CCEA architecture, broker credentials are handled differently depending on your usage mode:

| Mode | Credential Storage | Who Processes | Our Access |
|------|-------------------|--------------|------------|
| **Research SaaS** | Not applicable | Not applicable | N/A |
| **Live Execution (Agent)** | Your local Agent | Your hardware | **NONE** |
| **Enterprise (Self-Hosted)** | Your infrastructure | Your servers | **NONE** |

#### 2.3.1 Agent Local Vault (Live Execution Mode)

When you deploy a local Agent for live execution:

| Data Type | Storage Location | Encryption | Our Access |
|-----------|-----------------|------------|------------|
| API Key | Agent local vault | AES-256-GCM | **NONE** |
| API Secret | Agent local vault | AES-256-GCM | **NONE** |
| Passphrase | Agent local vault | AES-256-GCM | **NONE** |

**How the Agent Vault Works (by design):**
1. Credentials are designed to be encrypted on YOUR hardware with keys derived from YOUR passphrase
2. Master key is designed to remain within your Agent's local environment
3. Credentials are designed to be decrypted only in-memory inside the Agent when executing orders
4. CustodiaCloud Cloud is designed without the ability to decrypt or access these credentials

#### 2.3.2 What We DO NOT Store

Our Cloud infrastructure is designed to not store or process:
- Broker API keys
- Broker API secrets
- Exchange credentials
- Trading passphrases
- Any authentication tokens for brokers

**This is enforced by architecture** - Cloud has no API or mechanism to receive credentials.

### 2.4 Technical Data

We automatically collect:

| Data Type | Purpose | Legal Basis |
|-----------|---------|-------------|
| IP address | Security, geo-verification | Legitimate interest (Art. 6(1)(f)) |
| Device information | Service optimization | Legitimate interest (Art. 6(1)(f)) |
| Browser type | Compatibility | Legitimate interest (Art. 6(1)(f)) |
| Access timestamps | Security monitoring | Legitimate interest (Art. 6(1)(f)) |
| Error logs | Debugging, service improvement | Legitimate interest (Art. 6(1)(f)) |

### 2.5 Usage Analytics

With your consent, we may collect:

| Data Type | Purpose | Legal Basis |
|-----------|---------|-------------|
| Feature usage | Service improvement | Consent (Art. 6(1)(a)) |
| Performance metrics | Optimization | Consent (Art. 6(1)(a)) |
| User interactions | UX improvement | Consent (Art. 6(1)(a)) |

---

## 3. LEGAL BASIS FOR PROCESSING

### 3.1 Contract Performance (Article 6(1)(b))

We process data necessary to provide our services:

- Account management
- Strategy storage and execution
- Backtest computation
- Monitoring (where enabled) and customer support

### 3.2 Legitimate Interest (Article 6(1)(f))

We process data for our legitimate business interests:

- **Security**: Protecting the platform and users from unauthorized access
- **Fraud prevention**: Detecting and preventing malicious activities
- **Service improvement**: Analyzing usage patterns to improve functionality
- **Communication**: Sending service-related notifications

**Balancing Test:** We have conducted legitimate interest assessments intended to ensure our interests do not override your rights and freedoms (assessment documentation available upon request).

### 3.3 Consent (Article 6(1)(a))

For optional processing, we obtain your explicit consent:

- Marketing communications
- Analytics cookies
- Optional data collection

You can withdraw consent at any time through account settings.

### 3.4 Legal Obligation (Article 6(1)(c))

We may process data to comply with legal requirements:

- Operational/security record-keeping (e.g., access logs)
- Tax reporting obligations
- Law enforcement requests (with valid legal process)
- Regulatory inquiries

---

## 4. DATA RETENTION

### 4.1 Retention Periods

| Data Category | Retention Period | Reason |
|---------------|------------------|--------|
| Account data | Until deletion request + 30 days | Account recovery grace period |
| Strategies | Until user deletes or account closure | Service provision |
| Backtest results | 2 years or until user deletes | Historical reference |
| Telemetry (redacted/aggregated) | 90 days (default) | Monitoring and reliability |
| Security logs | 2 years | Security investigation |
| Audit logs | 5 years (anonymized after deletion) | Legal compliance |

### 4.2 Deletion Process

When you delete data or your account:

1. **Immediate**: Data is marked for deletion
2. **Within 30 days**: Data is permanently deleted from primary systems
3. **Within 90 days**: Data is removed from backups
4. **Exception**: Anonymized audit logs may be retained per legal requirements

---

## 5. DATA SHARING

### 5.1 Third-Party Brokers (CCEA Architecture)

**IMPORTANT: Under CCEA architecture, WE do not share data with brokers - YOUR Agent does.**

| Operation | Who Sends | What is Sent | Our Involvement |
|-----------|----------|--------------|-----------------|
| Order execution | Your local Agent | Order details, credentials | **NONE** |
| Position queries | Your local Agent | API credentials | **NONE** |
| Account status | Your local Agent | API credentials | **NONE** |

**Data flow for broker connections:**
```
Your Agent (local) → Broker API (direct connection)
     ↓
Cloud receives ONLY redacted telemetry (no order details)
```

We do NOT:
- Store your broker credentials
- Transmit orders on your behalf
- Have any connection to your broker accounts
- Receive order details (side, quantity, price)

### 5.2 No Sale of Data

**WE DO NOT SELL YOUR PERSONAL DATA.**

We do not sell, rent, or trade your personal information to third parties for marketing or any other purposes.

### 5.3 Sub-Processors (EU-only)

**Planned sub-processor configuration: EU-only.** Our current infrastructure design specifies EU-region sub-processors. Actual sub-processor list is maintained at [docs/compliance/SUBPROCESSORS_REGISTER.md](../compliance/SUBPROCESSORS_REGISTER.md).

| Provider | Purpose | Region (EU-only) | DPA Status | Notes |
|----------|---------|------------------|------------|-------|
| AWS (Amazon Web Services) | Cloud infrastructure (RDS, S3, ElastiCache, CloudWatch) | eu-central-1 (Frankfurt), eu-west-1 (Ireland) | Template available | Standard AWS DPA |
| Supabase | Database hosting (PostgreSQL alternative) | EU (Germany) | Template available | Standard Supabase DPA |
| Stripe | Payment processing | EU (Ireland) | Template available | Standard Stripe DPA |
| AWS SES / SendGrid | Transactional email | EU | Template available | Standard Twilio DPA |
| Sentry | Error monitoring (redacted, no PII) | EU (Germany) | Template available | Standard Sentry DPA |

**Sub-processor change notification:**
- **Notification period:** 30 days prior to new sub-processor engagement
- **Method:** Email to billing contact + in-app notification
- **Objection process:** Customer may object within 30 days; if unresolved, termination right applies

**Updated List:** The current sub-processor register with EU-only evidence is maintained at [docs/compliance/SUBPROCESSORS_REGISTER.md](../compliance/SUBPROCESSORS_REGISTER.md).

### 5.4 Telemetry Data (Agent → Cloud)

When you operate a local Agent, it may send telemetry to Cloud for monitoring:

#### 5.4.1 What Telemetry Contains (After Mandatory Redaction)

| Metric Type | Data Transmitted | Sensitive Details |
|-------------|-----------------|-------------------|
| Performance | PnL %, drawdown % | Rounded/bucketed |
| Health | Heartbeat, state | No trading data |
| Errors | Error codes | No order details |
| Equity | Bucketed range | No exact values |

#### 5.4.2 What Telemetry Is Designed Not to Contain

These fields are **designed to be blocked at the protocol level** (per CCEA architecture):
- Order side (buy/sell)
- Order quantity
- Order price
- Symbol/asset identifiers
- Exact position sizes
- Exact equity values
- Broker account identifiers

#### 5.4.3 Telemetry Sensitivity Levels (CCEA Design)

The Platform implements three distinct telemetry sensitivity levels as defined in the CCEA architecture. **Redaction is designed to be always active**; the architecture is designed not to expose configuration options or feature flags to disable it (enforced via CI guardrails and runtime checks; design intent).

| Level | Description | Data Included | Opt-in Required |
|-------|-------------|---------------|-----------------|
| **`AGGREGATED`** | Default for all tiers | PnL %, drawdown %, exposure (bucketed), error rates, health status, latency percentiles | No (default) |
| **`DETAILED_NON_SENSITIVE`** | Technical debugging | All AGGREGATED plus: timestamps, state transitions, signal metrics, queue depths, memory/CPU | Yes (explicit) |
| **`RAW_ORDER_EVENTS`** | Enterprise-only | All DETAILED plus: order events (masked account IDs), fill events, position changes | Yes (enterprise + explicit consent) |

**Level-specific restrictions:**

| Level | Order/Fill Data | Account Identifiers | Retention | Access |
|-------|-----------------|---------------------|-----------|--------|
| `AGGREGATED` | **Forbidden** | **Forbidden** | 90 days (configurable) | Workspace members |
| `DETAILED_NON_SENSITIVE` | **Forbidden** | **Forbidden** | 30 days (configurable) | Authorized workspace members |
| `RAW_ORDER_EVENTS` | Allowed (masked) | Masked only | 7 days (max 30) | Workspace admins + break-glass |

**RAW_ORDER_EVENTS requirements:**
- Available **only** to enterprise tier customers
- Requires **explicit per-workspace opt-in** (audited)
- Consent record must exist with: who, what, when, scope, expiry
- Minimal retention enforced (7 days default, maximum 30 days)
- Access restricted to workspace admins with audit trail
- Alternative: "telemetry stays local" mode (no Cloud transmission)

**What RAW_ORDER_EVENTS is designed to exclude** (even at enterprise level):
- API keys, secrets, or credentials (blocked by mandatory redaction)
- Unmasked account identifiers (masked by design)
- Environment variables (forbidden in telemetry schema)

#### 5.4.4 Telemetry Retention

| Sensitivity Level | Default Retention | Maximum Retention | Deletion |
|-------------------|-------------------|-------------------|----------|
| `AGGREGATED` | 90 days | Configurable per tenant | Auto-purged |
| `DETAILED_NON_SENSITIVE` | 30 days | 90 days | Auto-purged |
| `RAW_ORDER_EVENTS` | 7 days | 30 days | Auto-purged + audit event |

All telemetry is subject to:
- **Tenant-specific retention policies** (can reduce but not exceed maximums)
- **Auto-purge with auditable records** (purge job logs counts, timestamps)
- **Legal hold capability** (suspends deletion when active)
- **DSAR export/delete** (data subject rights honored within retention period)

### 5.5 Legal Disclosure

We may disclose data when required by:

- Valid court orders or subpoenas
- Regulatory requirements
- Protection of our legal rights
- Emergency situations involving safety

We will notify you of such disclosures unless legally prohibited.

---

## 6. INTERNATIONAL TRANSFERS

### 6.1 EU Storage

Your data is stored and processed in the European Union:

- **Primary Region:** AWS eu-central-1 (Frankfurt, Germany)
- **Backup Region:** AWS eu-west-1 (Dublin, Ireland)

### 6.2 Non-EU Transfers

If data transfer outside the EU is necessary, we ensure adequate protection through:

- **Standard Contractual Clauses (SCCs)** approved by the European Commission
- **Adequacy decisions** (for countries with adequate protection)
- **Binding Corporate Rules** (where applicable)

### 6.3 Broker Connections

When you connect to non-EU brokers/exchanges, order data may be transmitted to broker/exchange servers outside the EU by the broker relationship you operate. This is:

- Necessary for contract performance
- Initiated and controlled by you
- Processed by your broker/exchange counterparties (CustodiaCloud Cloud does not receive broker order payloads under CCEA)

---

## 7. YOUR RIGHTS UNDER GDPR

### 7.1 Right of Access (Article 15)

You have the right to:

- Confirm whether we process your personal data
- Obtain a copy of your personal data
- Information about processing purposes, categories, recipients

**How to exercise:** Account Settings > Privacy > Download My Data

### 7.2 Right to Rectification (Article 16)

You have the right to:

- Correct inaccurate personal data
- Complete incomplete personal data

**How to exercise:** Account Settings > Profile, or contact support

### 7.3 Right to Erasure (Article 17)

You have the right to request deletion of your personal data when:

- Data is no longer necessary for the original purpose
- You withdraw consent (for consent-based processing)
- You object to processing (and there are no overriding legitimate grounds)
- Data has been unlawfully processed

**Exceptions:** We may retain data required for legal compliance (e.g., audit logs for 5 years).

**How to exercise:** Account Settings > Privacy > Delete Account, or contact DPO

### 7.4 Right to Restriction (Article 18)

You have the right to restrict processing when:

- You contest data accuracy (pending verification)
- Processing is unlawful but you prefer restriction over erasure
- We no longer need the data but you need it for legal claims
- You have objected to processing (pending verification)

### 7.5 Right to Data Portability (Article 20)

You have the right to:

- Receive your personal data in a structured, commonly used, machine-readable format (JSON)
- Transmit that data to another controller

**Scope:** Applies to data you provided to us, processed by automated means, based on consent or contract.

**How to exercise:** Account Settings > Privacy > Export My Data

### 7.6 Right to Object (Article 21)

You have the right to object to processing based on legitimate interests. We will cease processing unless we demonstrate compelling legitimate grounds.

### 7.7 Right Regarding Automated Decision-Making (Article 22)

You have the right not to be subject to decisions based solely on automated processing that produce legal or significant effects.

**Note:** Your local Agent executes orders based on YOUR strategies, not automated decisions by us. The Cloud Platform manages strategy deployment and monitoring only.

### 7.8 Right to Withdraw Consent (Article 7(3))

Where processing is based on consent, you can withdraw consent at any time. Withdrawal does not affect the lawfulness of prior processing.

### 7.9 Right to Lodge a Complaint

You have the right to lodge a complaint with your local data protection supervisory authority (EEA/UK as applicable).

### 7.10 DSAR Scope Boundaries (CCEA-specific)

**Important:** Due to the CCEA architecture, DSAR (Data Subject Access Request) scope is limited to Cloud-controlled data:

**IN SCOPE (Cloud-controlled, we can export/delete):**
- User account data (email, display_name, preferences)
- Organization membership records
- Workspace membership and roles
- Strategy metadata (owned by user/workspace)
- Telemetry data (at enabled sensitivity level)
- Command history and approval records
- Access audit logs (where user is subject)
- Support interaction records

**OUT OF SCOPE (Agent-controlled, customer responsibility):**
- Broker credentials (never in Cloud)
- Local execution logs (unless exported via REQUEST_EXPORT_LOGS)
- Order/fill data (unless RAW_ORDER_EVENTS enabled and transmitted)
- Local vault contents
- Position data (unless transmitted via telemetry)

**Standard DSAR response:**
> "Your request has been processed for all personal data held in our Cloud systems. Data stored in your local Agent environment (including broker credentials, local logs, and order data) is under your control and not accessible to us. Please contact your system administrator for access to Agent-local data."

---

## 7A. CCEA PRIVACY DESIGN COMMITMENTS CHECKLIST

This section provides an explicit checklist of privacy design commitments for the CCEA architecture. These commitments describe the intended Cloud/Agent boundary and telemetry controls.

### 7A.1 Cloud Does Not Receive Secrets

| Commitment | Enforcement | Verification |
|-----------|-------------|--------------|
| **No broker API keys in Cloud** | Schema validation, CI guardrails | Build-time + runtime |
| **No API secrets in Cloud** | Redaction middleware (mandatory) | Cannot be disabled |
| **No OAuth tokens in Cloud** | Protocol prohibition | Schema enforcement |
| **No environment variables in Cloud** | Forbidden in telemetry schema | CI tests |

### 7A.2 No Order-like Payloads in Protocol

| Commitment | Enforcement | Verification |
|-----------|-------------|--------------|
| **No side (buy/sell) in commands** | JSON Schema prohibition | Build-time CI |
| **No quantity in commands** | JSON Schema prohibition | Build-time CI |
| **No price in commands** | JSON Schema prohibition | Build-time CI |
| **No order_id in commands** | JSON Schema prohibition | Build-time CI |
| **No target_position in commands** | JSON Schema prohibition | Build-time CI |

### 7A.3 Telemetry Controls

| Commitment | Enforcement | Verification |
|-----------|-------------|--------------|
| **Default is AGGREGATED** | Config default | Runtime check |
| **DETAILED_NON_SENSITIVE requires opt-in** | Explicit configuration | Audit event |
| **RAW_ORDER_EVENTS requires enterprise + opt-in** | Tier check + consent record | Audit trail |
| **Redaction designed as always on** | Designed not to be disabled by flag | CI test + runtime (verify via test reports) |
| **Order data forbidden in non-RAW** | Schema validation | Runtime rejection |

### 7A.4 EU-Priority Data Residency

| Design Goal | Enforcement | Verification |
|-------------|-------------|--------------|
| **Core storage in EU** | Region configuration (design goal) | Drift check (fail-closed; pending infrastructure deployment) |
| **Core backups in EU** | Backup region policy | Automated verification |
| **Core logs in EU** | CloudWatch region lock | Infrastructure audit |
| **Sub-processors: EU where possible** | Contractual + DPA; SCCs/DPF for non-EU | Quarterly review |

**Note:** Some sub-processors (e.g., payment, email, error monitoring) may process data outside the EU under Standard Contractual Clauses (SCCs) and/or Data Privacy Framework (DPF). See SUBPROCESSORS_REGISTER.md for details.

### 7A.5 DSAR Boundaries

| Commitment | Enforcement | Verification |
|-----------|-------------|--------------|
| **DSAR scope is Cloud-only** | Architecture | Process documentation |
| **Agent data is customer-controlled** | No Cloud access | Cannot export what we don't have |
| **Response includes boundary explanation** | Template enforcement | SOP compliance |

---

## 7B. SUPPORT WITH CONSENT

### 7B.1 Support Data Access Policy

Support staff access to customer data requires **explicit consent** with auditable records.

**Consent requirements:**
- **Who:** Identity of the user granting consent
- **What:** Specific data or scope of access
- **When:** Timestamp of consent grant
- **Scope:** Bounded by workspace/data type
- **Expiry:** Time-limited (default: 72 hours, max: 30 days)

### 7B.2 Consent Record Structure

Each support consent record contains:

| Field | Description | Required |
|-------|-------------|----------|
| `consent_id` | Unique identifier | Yes |
| `user_id` | User granting consent | Yes |
| `workspace_id` | Scope of access | Yes |
| `granted_at` | Timestamp (UTC) | Yes |
| `expires_at` | Expiry timestamp | Yes |
| `scope` | Data types accessible | Yes |
| `purpose` | Reason for access | Yes |
| `support_ticket_id` | Associated ticket | Yes |
| `revoked_at` | Revocation timestamp | If revoked |

### 7B.3 Consent Revocation

You can revoke support consent at any time:

- **Method:** Account Settings > Privacy > Support Access, or contact DPO
- **Effect:** Immediate (access blocked within seconds)
- **Audit:** Revocation is logged with timestamp and actor

**Enforcement:** Support data export is blocked without active, non-expired consent.

### 7B.4 Auditable Evidence

All support access is logged in the governance audit trail:
- Every data access during support session
- Every export generated
- Support session start/end timestamps
- Consent verification at access time

---

## 8. COOKIES AND TRACKING

### 8.1 Essential Cookies

We use essential cookies required for the Platform to function:

| Cookie | Purpose | Duration |
|--------|---------|----------|
| session_id | Session management | Session |
| csrf_token | Security | Session |
| auth_token | Authentication | 7 days |

**These cookies do not require consent** under ePrivacy Directive Art. 5(3).

### 8.2 Analytics Cookies (Optional)

With your consent, we may use analytics cookies:

| Cookie | Purpose | Duration |
|--------|---------|----------|
| _analytics_id | Usage analytics | 1 year |

**How to manage:** Cookie preferences are available in account settings.

### 8.3 No Third-Party Advertising

We do not use advertising cookies or allow third-party advertising trackers.

---

## 9. SECURITY MEASURES

### 9.1 Technical Measures

We implement comprehensive security controls:

- **Encryption at rest**: AES-256 for sensitive Cloud data (e.g., account data, audit logs, redacted telemetry); broker credentials remain in the customer-controlled Agent environment
- **Encryption in transit**: TLS 1.3 for all communications
- **Key management**: Hardware Security Modules (HSM) for master keys
- **Access controls**: Role-based access, multi-factor authentication
- **Network security**: Firewalls, intrusion detection, DDoS protection

### 9.2 Organizational Measures

- **Security training**: Regular employee security awareness training
- **Access policies**: Principle of least privilege
- **Incident response**: Documented procedures for security incidents
- **Vendor assessment**: Security evaluation of all sub-processors

### 9.3 Audit and Monitoring

- Security assessments (internal) and penetration testing (planned; see Trust Center for roadmap)
- Continuous monitoring for suspicious activities
- Audit logging of all sensitive data access

### 9.4 Breach Notification

In the event of a personal data breach:

- We will notify the supervisory authority within 72 hours (GDPR Art. 33)
- We will notify affected users without undue delay if high risk (GDPR Art. 34)
- We will document all breaches in our breach register

---

## 10. CHILDREN'S PRIVACY

The Platform is not intended for individuals under 18 years of age. We do not knowingly collect personal data from children.

If we become aware that we have collected data from a child, we will delete it promptly.

---

## 11. CHANGES TO THIS POLICY

### 11.1 Notification of Changes

We will notify you of material changes to this Privacy Policy:

- Via email to your registered address
- Through a prominent notice on the Platform
- At least 30 days before changes take effect

### 11.2 Version History

| Version | Date | Summary of Changes |
|---------|------|--------------------|
| 1.0.0 | December 2024 | Initial release |
| 2.0.0 | December 2024 | Added CCEA architecture sections: data zones, credential handling, telemetry redaction |
| 3.0.0 | December 2024 | GDPR Phase 1: Added CCEA telemetry levels (AGGREGATED/DETAILED_NON_SENSITIVE/RAW_ORDER_EVENTS), CCEA Privacy Design Commitments Checklist (Section 7A), Support-with-Consent policy (Section 7B), DSAR scope boundaries, EU-only sub-processor list with review timestamps |

### 11.3 Review

We review this Privacy Policy annually and update it as necessary to reflect changes in our practices or legal requirements.

---

## 12. CONTACT US

### 12.1 Privacy Inquiries

For questions about this Privacy Policy or your personal data:

- **Email:** privacy@[company-domain].com
- **DPO Email:** dpo@[company-domain].com

### 12.2 Data Subject Requests

To exercise your GDPR rights:

- **Self-service:** Account Settings > Privacy
- **Email:** dpo@[company-domain].com
- **Response time:** Within 30 days (extendable by 60 days for complex requests)

### 12.3 Complaints

If you are unsatisfied with our response, you may:

1. Escalate to our DPO
2. Lodge a complaint with your supervisory authority

---

## 13. LEGAL REFERENCES

| Reference | Description |
|-----------|-------------|
| GDPR Article 13 | Information for direct collection |
| GDPR Article 14 | Information for indirect collection |
| GDPR Article 15 | Right of access |
| GDPR Article 16 | Right to rectification |
| GDPR Article 17 | Right to erasure |
| GDPR Article 18 | Right to restriction |
| GDPR Article 20 | Right to data portability |
| GDPR Article 21 | Right to object |
| GDPR Article 22 | Automated decision-making |
| GDPR Article 33-34 | Breach notification |
| ePrivacy Directive Art. 5(3) | Cookie consent |
| EDPB Guidelines WP260 | Transparency |
| EDPB Guidelines WP242 | Data Portability |

---

**Last Updated:** December 2024
**Document Version:** 3.0.0

---

## APPENDIX A: DATA PROCESSING ACTIVITIES REGISTER

| Activity | Data Categories | Legal Basis | Retention | Recipients | Zone |
|----------|----------------|-------------|-----------|------------|------|
| Account Management | Email, name, password hash | Contract | Until deletion | Internal | Cloud |
| Strategy Storage | Code, parameters | Contract | Until deletion | Internal | Cloud |
| Backtesting | Historical results | Contract | 2 years | Internal | Cloud |
| Security Monitoring | IP, logs | Legitimate Interest | 2 years | Internal | Cloud |
| Telemetry Ingestion | Redacted metrics | Contract | 90 days | Internal | Cloud |
| Audit Logging | Access records | Legal obligation | 5 years | Internal | Cloud |

**Agent-Zone Activities (Processed Locally, NOT by Us):**

| Activity | Data Categories | Storage | Our Access |
|----------|----------------|---------|------------|
| Credential Storage | API keys (encrypted) | Agent vault | **NONE** |
| Order Execution | Orders, positions | Agent local | **NONE** |
| Position Tracking | Real-time positions | Agent memory | **NONE** |
| Approval Records | Local evidence | Agent storage | **NONE** |

---

## APPENDIX B: CCEA DATA FLOW DIAGRAM

```
┌──────────────────────────────────────────────────────────────────────┐
│                        YOUR INFRASTRUCTURE                            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     LOCAL AGENT                                   │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │ │
│  │  │ Credentials  │  │   Orders     │  │   Position Data      │  │ │
│  │  │ (Encrypted)  │  │ (Generated)  │  │   (Real-time)        │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │ │
│  │         │                 │                                      │ │
│  │         │     ┌───────────┴───────────┐                         │ │
│  │         │     │     BROKER API        │ ← Direct connection     │ │
│  │         └─────┤    (Your account)     │   (We have NO access)   │ │
│  │               └───────────────────────┘                         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                               │                                       │
│                    Redacted Telemetry Only                           │
│                      (No credentials,                                │
│                       no order details)                              │
│                               │                                       │
└───────────────────────────────┼───────────────────────────────────────┘
                                ↓
┌───────────────────────────────────────────────────────────────────────┐
│                          CCEA CLOUD                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐       │
│  │  Research    │  │  Aggregated  │  │    Audit Logs        │       │
│  │  Jobs        │  │  Telemetry   │  │  (Access records)    │       │
│  └──────────────┘  └──────────────┘  └──────────────────────┘       │
│                                                                       │
│  ❌ NO Credentials  ❌ NO Orders  ❌ NO Position Details            │
└───────────────────────────────────────────────────────────────────────┘
```
