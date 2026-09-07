# Cloud Governance: RBAC, Privacy, Data Residency

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Overview

Cloud Governance provides multi-tenant data management with:

- Role-Based Access Control (RBAC)
- Data retention policies
- EU data residency
- Access audit logging
- Break-glass procedures for emergency access

---

## 1. Role-Based Access Control (RBAC)

### 1.1 Organizational Hierarchy

```
Organization
    │
    ├── Workspace 1
    │   ├── Users (with roles)
    │   ├── Strategies
    │   ├── Deployments
    │   └── Agents
    │
    └── Workspace 2
        ├── Users (with roles)
        └── ...
```

### 1.2 Built-in Roles

| Role | Description | Key Permissions |
|------|-------------|-----------------|
| `org_admin` | Organization administrator | Full access to all workspaces |
| `workspace_admin` | Workspace administrator | Full access within workspace |
| `developer` | Strategy developer | Create/edit strategies, view telemetry |
| `operator` | Operations | Manage deployments, view monitoring |
| `viewer` | Read-only | View strategies, telemetry (no edit) |
| `auditor` | Compliance auditor | Read access + audit logs |

### 1.3 Permission Matrix

| Permission | org_admin | ws_admin | developer | operator | viewer | auditor |
|------------|-----------|----------|-----------|----------|--------|---------|
| Create workspace | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Manage users | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Create strategies | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Edit strategies | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| View strategies | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Create deployments | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| Start/stop runs | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| View telemetry | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RAW telemetry | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| View audit logs | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| Break-glass access | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

### 1.4 Custom Roles

Organizations can define custom roles:

```json
{
  "role_name": "quant_researcher",
  "description": "Quantitative researcher with backtest access",
  "permissions": [
    "strategies.read",
    "strategies.create",
    "strategies.edit.own",
    "backtests.run",
    "telemetry.read.aggregated"
  ],
  "constraints": {
    "max_concurrent_backtests": 5,
    "max_strategy_versions": 10
  }
}
```

---

## 2. Data Retention

### 2.1 Default Retention Periods

| Data Type | Default Retention | Minimum | Maximum |
|-----------|-------------------|---------|---------|
| Account data | Until deletion + 30 days | 30 days | Unlimited |
| Strategies | Until user deletes | N/A | Unlimited |
| Strategy versions | 2 years | 90 days | 7 years |
| Backtest results | 90 days | 7 days | 2 years |
| Execution logs | 5 years | 5 years | 7 years |
| Telemetry (AGGREGATED) | 90 days | 7 days | 2 years |
| Telemetry (DETAILED) | 30 days | 7 days | 1 year |
| Telemetry (RAW) | 7 days | 1 day | 90 days |
| Audit logs | 5 years | 5 years | 7 years |
| Security logs | 2 years | 1 year | 5 years |

### 2.2 Configuring Retention

```yaml
# workspace-config.yaml
retention:
  backtest_results_days: 90
  telemetry_aggregated_days: 90
  telemetry_detailed_days: 30
  telemetry_raw_days: 7
  strategy_versions_days: 730  # 2 years
```

### 2.3 Retention Jobs

Automated retention jobs run daily:

```sql
-- Example: Purge expired telemetry
DELETE FROM telemetry_events
WHERE workspace_id = $1
  AND created_at < NOW() - INTERVAL '90 days'
  AND level = 'AGGREGATED';
```

### 2.4 Legal Hold

For legal/compliance requirements:

```bash
# Enable legal hold (prevents deletion)
ccea-admin retention legal-hold enable \
  --workspace ws_abc123 \
  --reason "regulatory_investigation" \
  --until 2026-01-01
```

---

## 3. Data Residency

### 3.1 Region Configuration

| Region | Primary DC | Backup DC | Default For |
|--------|------------|-----------|-------------|
| EU | eu-central-1 (Frankfurt) | eu-west-1 (Dublin) | EU users |
| US | us-east-1 (Virginia) | us-west-2 (Oregon) | US users |
| APAC | ap-southeast-1 (Singapore) | ap-northeast-1 (Tokyo) | APAC users |

### 3.2 EU Data Residency (Default for EU)

```yaml
# workspace-config.yaml
data_residency:
  region: eu
  storage_locations:
    - eu-central-1
    - eu-west-1
  cross_region_replication: eu_only
  encryption:
    at_rest: AES-256
    key_management: aws_kms_eu
```

### 3.3 Data Flow Restrictions

| Data Type | Allowed Regions | Cross-Border Transfer |
|-----------|-----------------|----------------------|
| Personal data | Configured region | SCCs required |
| Trading telemetry | Configured region | Prohibited (default) |
| Aggregated metrics | Any | Allowed |
| Audit logs | Configured region | Prohibited |

### 3.4 Enterprise: Custom Residency

```yaml
# enterprise-residency.yaml
data_residency:
  mode: customer_controlled
  primary_region: eu-central-1
  allowed_regions:
    - eu-central-1
    - eu-west-1
  prohibited_regions:
    - us-*
    - cn-*
  encryption:
    mode: customer_managed_key
    kms_key_arn: arn:aws:kms:eu-central-1:xxx:key/yyy
```

---

## 4. Access Audit

### 4.1 Audited Events

| Event Type | Description | Retention |
|------------|-------------|-----------|
| `user.login` | User authentication | 2 years |
| `user.logout` | User logout | 2 years |
| `user.mfa_challenge` | MFA verification | 2 years |
| `resource.create` | Resource creation | 5 years |
| `resource.update` | Resource modification | 5 years |
| `resource.delete` | Resource deletion | 5 years |
| `resource.access` | Resource read (sensitive) | 5 years |
| `admin.action` | Administrative action | 5 years |
| `break_glass.access` | Emergency access | 7 years |

### 4.2 Audit Log Format

```json
{
  "event_id": "evt_abc123",
  "timestamp": "2025-12-14T12:00:00Z",
  "event_type": "resource.access",
  "actor": {
    "user_id": "user_xyz",
    "email": "user@example.com",
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0..."
  },
  "resource": {
    "type": "strategy",
    "id": "strat_abc",
    "workspace_id": "ws_xyz"
  },
  "action": "read",
  "result": "success",
  "metadata": {
    "reason": "deployment_review",
    "request_id": "req_xxx"
  }
}
```

### 4.3 Querying Audit Logs

```bash
# CLI query
ccea-admin audit query \
  --workspace ws_abc123 \
  --event-type "resource.*" \
  --from "2025-12-01" \
  --to "2025-12-14"

# API query
GET /v1/audit/events?workspace_id=ws_abc123&event_type=resource.*
```

### 4.4 Audit Export

```bash
# Export for compliance
ccea-admin audit export \
  --workspace ws_abc123 \
  --format json \
  --from "2025-01-01" \
  --to "2025-12-31" \
  --output audit_2025.json
```

---

## 5. Break-Glass Procedures

### 5.1 What is Break-Glass?

Emergency access to bypass normal RBAC controls when:

- Critical incident requires immediate action
- Normal access paths unavailable
- Time-sensitive compliance requirement

### 5.2 Break-Glass Requirements

1. **Justification**: Written reason required
2. **Approval**: Second org_admin approval (dual control)
3. **Time-limited**: Maximum 4-hour session
4. **Audited**: All actions logged with enhanced detail
5. **Notification**: All workspace admins notified

### 5.3 Initiating Break-Glass

```bash
# Request break-glass access
ccea-admin break-glass request \
  --workspace ws_abc123 \
  --reason "critical_security_incident" \
  --duration 2h

# Approve (by different admin)
ccea-admin break-glass approve \
  --request-id bg_xyz \
  --approver-notes "Confirmed incident, approved"
```

### 5.4 Break-Glass Audit

```json
{
  "event_id": "evt_break_glass_123",
  "timestamp": "2025-12-14T12:00:00Z",
  "event_type": "break_glass.access",
  "actor": {
    "user_id": "admin_xyz",
    "email": "admin@company.com"
  },
  "break_glass": {
    "request_id": "bg_xyz",
    "reason": "critical_security_incident",
    "approved_by": "admin_abc",
    "approved_at": "2025-12-14T11:55:00Z",
    "expires_at": "2025-12-14T14:00:00Z"
  },
  "actions_taken": [
    {
      "action": "read",
      "resource": "deployment_dep_abc",
      "timestamp": "2025-12-14T12:01:00Z"
    }
  ]
}
```

---

## 6. Multi-Factor Authentication

### 6.1 MFA Requirements

| Role | MFA Required | Methods |
|------|--------------|---------|
| org_admin | Yes (by default) | TOTP, WebAuthn |
| workspace_admin | Yes (by default) | TOTP, WebAuthn |
| developer | Configurable | TOTP, WebAuthn |
| operator | Configurable | TOTP, WebAuthn |
| viewer | Configurable | TOTP |

### 6.2 Configuring MFA Policy

```yaml
# org-security.yaml
mfa:
  required_roles:
    - org_admin
    - workspace_admin
  optional_roles:
    - developer
    - operator
  exempt_roles: []  # No exemptions
  allowed_methods:
    - totp
    - webauthn
  session_timeout_minutes: 60
  remember_device_days: 30
```

---

## 7. Tenant Isolation

### 7.1 Database-Level Isolation

```sql
-- Postgres Row-Level Security
CREATE POLICY workspace_isolation ON strategies
  FOR ALL
  USING (workspace_id = current_setting('app.current_workspace_id'));

-- All queries automatically filtered
SELECT * FROM strategies;  -- Only returns current workspace
```

### 7.2 API-Level Isolation

```python
# All endpoints include workspace context
@router.get("/strategies")
async def list_strategies(
    workspace_id: str = Depends(get_current_workspace),
    db: Session = Depends(get_db)
):
    # workspace_id is validated and enforced
    return db.query(Strategy).filter(
        Strategy.workspace_id == workspace_id
    ).all()
```

### 7.3 Network Isolation (Enterprise)

```yaml
# enterprise-isolation.yaml
network:
  mode: dedicated_vpc
  vpc_cidr: 10.100.0.0/16
  private_subnets: true
  nat_gateway: true
  vpc_endpoints:
    - s3
    - dynamodb
    - kms
```

---

## 8. Compliance Reporting

### 8.1 Available Reports

| Report | Description | Format |
|--------|-------------|--------|
| Access Report | User access summary | PDF, CSV |
| Audit Trail | Complete audit log | JSON, CSV |
| Data Inventory | Personal data mapping | JSON |
| Retention Report | Data retention status | PDF |
| DSAR Export | GDPR data subject request | ZIP |

### 8.2 Generating Reports

```bash
# Access report
ccea-admin compliance report access \
  --workspace ws_abc123 \
  --period monthly \
  --format pdf

# DSAR export
ccea-admin compliance dsar export \
  --user user@example.com \
  --include personal_data telemetry strategies \
  --format zip
```

---

## 9. Configuration Reference

### Complete Governance Config

```yaml
# governance-config.yaml
organization:
  id: org_abc123
  name: "Example Corp"

security:
  mfa:
    required: true
    methods: [totp, webauthn]
  session:
    timeout_minutes: 60
    max_concurrent: 3
  password:
    min_length: 12
    require_special: true
    rotation_days: 90

retention:
  telemetry_aggregated_days: 90
  telemetry_detailed_days: 30
  telemetry_raw_days: 7
  backtest_results_days: 90
  audit_logs_years: 5

residency:
  region: eu
  cross_border: prohibited
  encryption: aes256_kms

audit:
  enabled: true
  sensitive_access_logging: true
  export_format: json
  retention_years: 5

compliance:
  frameworks: [gdpr, dora]
  auto_reports: true
  report_schedule: monthly
```

---

**Related Documentation:**

- [Privacy Policy](../legal/PRIVACY_POLICY.md)
- [Terms of Service](../legal/TERMS_OF_SERVICE.md)
- [DORA Compliance](../compliance/DORA_INTEGRATION_PLAN.md)
- [Security Trust Center](../security/TRUST_CENTER.md)
