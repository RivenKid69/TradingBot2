# Telemetry Data Dictionary (Cloud Ingestion)

**Document Version**: 1.0.0
**Effective Date**: 2025-12-16
**Classification**: INTERNAL / COMPLIANCE
**Related Documents**:
- `docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md`
- `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt`
- `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md`

## 1. Overview

This document defines the canonical telemetry data dictionary for Cloud ingestion in the CCEA architecture. It specifies allowed and forbidden fields for each telemetry sensitivity level per GDPR data minimization requirements (Art. 5(1)(c)) and the Design Doc constraints.

### 1.1 Canonical Telemetry Levels

| Level | ID | Default | Access | Storage | Redaction |
|-------|----|---------| -------|---------|-----------|
| Aggregated | `AGGREGATED` | Yes (retail/pro) | All users | Standard Cloud | N/A (no PII) |
| Detailed Non-Sensitive | `DETAILED_NON_SENSITIVE` | Opt-in | All users | Standard Cloud | Mandatory |
| Raw Order Events | `RAW_ORDER_EVENTS` | Enterprise-only | Restricted RBAC | Isolated Enterprise Storage | Mandatory (except order fields) |

**Reference**: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L846`, `#L853`

### 1.2 Governing Principles

1. **Data Minimization**: Collect only what's necessary for monitoring, billing, and support (with consent).
2. **Redaction Is Designed to Be Mandatory**: Architecture is designed so that redaction cannot be disabled by configuration or feature flag (verify via CI guardrails and runtime tests).
3. **Order-Like Fields Forbidden**: Except for `RAW_ORDER_EVENTS` with enterprise opt-in.
4. **EU-Priority Residency**: Telemetry is designed to be stored in EU regions (verify via infrastructure configuration and drift checks).
5. **Retention Per Tenant**: Auto-purge according to retention policies.

---

## 2. Field Classification

### 2.1 Classification Categories

| Category | Description | Handling |
|----------|-------------|----------|
| `ALLOWED` | Field is permitted at this telemetry level | Accept and store |
| `FORBIDDEN` | Field is prohibited at this level | Reject ingestion (HTTP 422) |
| `REDACT` | Field must be redacted before transmission | Verify redaction markers present |
| `ENTERPRISE_ONLY` | Field requires enterprise license | Reject unless enterprise + opt-in |

### 2.2 Sensitivity Tags

| Tag | Description | Examples |
|-----|-------------|----------|
| `PII` | Personally Identifiable Information | email, phone, IP address |
| `FINANCIAL` | Financial/trading data | order details, positions |
| `SECRET` | Credentials and secrets | API keys, tokens, passwords |
| `TECHNICAL` | Technical metrics | CPU, memory, latency |
| `AGGREGATE` | Pre-aggregated statistics | counts, averages, percentiles |

---

## 3. AGGREGATED Level (Default)

**Purpose**: Operational monitoring with minimal data exposure.

**Default**: Yes - this is the default for all retail and professional users.

**Reference**: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L855`

### 3.1 Allowed Fields

#### Temporal Fields
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `timestamp` | datetime | Event timestamp (ISO 8601) | `2025-12-16T10:30:00Z` |
| `event_time` | datetime | When event occurred | `2025-12-16T10:30:00Z` |
| `start_time` | datetime | Period start time | `2025-12-16T00:00:00Z` |
| `end_time` | datetime | Period end time | `2025-12-16T23:59:59Z` |
| `period` | string | Time period identifier | `1h`, `1d` |
| `window` | string | Aggregation window | `5m`, `1h` |
| `interval` | string | Reporting interval | `PT5M` |
| `duration_seconds` | integer | Duration in seconds | `3600` |

#### Statistical Aggregates
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `count` | integer | Count of events | `1500` |
| `sum` | float | Sum of values | `45000.50` |
| `avg` | float | Average value | `30.0` |
| `min` | float | Minimum value | `10.5` |
| `max` | float | Maximum value | `95.2` |
| `median` | float | Median value | `28.5` |
| `p50` | float | 50th percentile | `28.5` |
| `p90` | float | 90th percentile | `75.0` |
| `p95` | float | 95th percentile | `85.0` |
| `p99` | float | 99th percentile | `92.0` |
| `stddev` | float | Standard deviation | `15.3` |
| `variance` | float | Variance | `234.09` |
| `histogram` | object | Histogram buckets | `{"0-10": 5, "10-20": 15}` |
| `buckets` | array | Bucket values | `[0, 10, 20, 50, 100]` |

#### System Metrics
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `cpu_percent` | float | CPU utilization (0-100) | `45.2` |
| `cpu_usage` | float | CPU usage | `0.452` |
| `memory_percent` | float | Memory utilization (0-100) | `62.8` |
| `memory_usage` | float | Memory usage | `0.628` |
| `memory_mb` | integer | Memory in MB | `4096` |
| `disk_usage` | float | Disk usage | `0.45` |
| `disk_percent` | float | Disk utilization (0-100) | `45.0` |
| `network_bytes_sent` | integer | Bytes sent | `1048576` |
| `network_bytes_recv` | integer | Bytes received | `2097152` |
| `latency_ms` | float | Latency in milliseconds | `15.5` |
| `latency_p50` | float | p50 latency | `10.2` |
| `latency_p95` | float | p95 latency | `25.8` |
| `latency_p99` | float | p99 latency | `45.0` |

#### Trading Metrics (Aggregated Only)
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `trade_count` | integer | Number of trades | `150` |
| `order_count` | integer | Number of orders | `200` |
| `fill_rate` | float | Order fill rate (0-1) | `0.85` |
| `win_rate` | float | Win rate (0-1) | `0.55` |
| `sharpe_ratio` | float | Sharpe ratio | `1.8` |
| `sortino_ratio` | float | Sortino ratio | `2.1` |
| `max_drawdown` | float | Maximum drawdown | `0.15` |
| `daily_return` | float | Daily return | `0.012` |
| `cumulative_return` | float | Cumulative return | `0.185` |
| `volatility` | float | Volatility | `0.18` |
| `alpha` | float | Alpha | `0.05` |
| `beta` | float | Beta | `0.85` |
| `information_ratio` | float | Information ratio | `0.65` |

#### Identifiers (Non-PII)
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `strategy_id` | string | Strategy identifier | `strat_abc123` |
| `strategy_name` | string | Strategy name | `momentum_v2` |
| `run_id` | string | Run identifier | `run_xyz789` |
| `deployment_id` | string | Deployment identifier | `deploy_456` |
| `agent_id` | string | Agent identifier | `agent_abc123def456` |
| `workspace_id` | string | Workspace identifier | `ws_123` |
| `version` | string | Version string | `1.2.3` |
| `status` | string | Status indicator | `RUNNING` |
| `state` | string | State indicator | `ACTIVE` |
| `is_paper_trading` | boolean | Paper trading flag | `true` |

#### Error Metrics
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `error_count` | integer | Number of errors | `5` |
| `error_rate` | float | Error rate | `0.02` |
| `warning_count` | integer | Number of warnings | `12` |
| `success_count` | integer | Success count | `245` |
| `failure_count` | integer | Failure count | `5` |
| `retry_count` | integer | Retry count | `3` |
| `timeout_count` | integer | Timeout count | `2` |

#### Resource Usage
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `active_connections` | integer | Active connections | `15` |
| `queue_depth` | integer | Queue depth | `42` |
| `processing_time_ms` | float | Processing time | `125.5` |
| `request_count` | integer | Request count | `10000` |
| `response_count` | integer | Response count | `9995` |
| `cache_hit_rate` | float | Cache hit rate | `0.85` |
| `cache_miss_rate` | float | Cache miss rate | `0.15` |

### 3.2 Forbidden Fields at AGGREGATED Level

All fields from sections 4.2, 5.2, and 6 are **FORBIDDEN** at AGGREGATED level.

---

## 4. DETAILED_NON_SENSITIVE Level (Opt-In)

**Purpose**: Operational debugging and performance analysis without sensitive data.

**Default**: No - requires explicit opt-in.

**Reference**: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L855`

### 4.1 Additional Allowed Fields

All fields from AGGREGATED level **plus** the following:

#### Event Details
| Field | Type | Description | Example | Redaction |
|-------|------|-------------|---------|-----------|
| `event_type` | string | Type of event | `heartbeat` | None |
| `event_name` | string | Event name | `agent_started` | None |
| `event_category` | string | Event category | `lifecycle` | None |
| `source` | string | Event source | `agent` | None |
| `component` | string | Component name | `telemetry` | None |
| `module` | string | Module name | `ingester` | None |
| `function` | string | Function name | `process_batch` | None |
| `level` | string | Log level | `INFO` | None |
| `severity` | string | Severity level | `WARNING` | None |
| `message` | string | Event message | `Batch processed` | PII scan |
| `description` | string | Description | `Processing complete` | PII scan |
| `details` | object | Additional details | `{}` | Deep scan |
| `context` | object | Context data | `{}` | Deep scan |
| `metadata` | object | Metadata | `{}` | Deep scan |
| `tags` | array | Tags | `["prod", "us"]` | None |
| `labels` | object | Labels | `{"env": "prod"}` | None |
| `annotations` | object | Annotations | `{}` | Deep scan |

#### Error Details (Non-PII)
| Field | Type | Description | Example | Redaction |
|-------|------|-------------|---------|-----------|
| `error_type` | string | Error type | `ValidationError` | None |
| `error_code` | string | Error code | `E4001` | None |
| `error_class` | string | Error class | `ClientError` | None |
| `stack_trace_hash` | string | Hash of stack trace | `sha256:abc...` | Hash only |

#### Tracing Fields
| Field | Type | Description | Example | Redaction |
|-------|------|-------------|---------|-----------|
| `operation` | string | Operation name | `process_command` | None |
| `operation_type` | string | Operation type | `COMMAND` | None |
| `resource_type` | string | Resource type | `deployment` | None |
| `resource_id` | string | Resource ID | `res_123` | None |
| `correlation_id` | string | Correlation ID | `corr_abc` | Hash if needed |
| `trace_id` | string | Trace ID | `trace_xyz` | None |
| `span_id` | string | Span ID | `span_123` | None |
| `parent_span_id` | string | Parent span ID | `span_000` | None |

### 4.2 Forbidden Fields at DETAILED_NON_SENSITIVE Level

#### Order/Intent Fields (ALWAYS FORBIDDEN unless RAW)
| Field | Reason | Violation Type |
|-------|--------|----------------|
| `side` | Trading intent | `PROHIBITED_FIELD` |
| `quantity` / `qty` | Order quantity | `PROHIBITED_FIELD` |
| `price` | Order price | `PROHIBITED_FIELD` |
| `order_type` | Order type | `PROHIBITED_FIELD` |
| `limit_price` | Limit price | `PROHIBITED_FIELD` |
| `stop_price` | Stop price | `PROHIBITED_FIELD` |
| `take_profit` | Take profit | `PROHIBITED_FIELD` |
| `stop_loss` | Stop loss | `PROHIBITED_FIELD` |
| `order_id` | Order identifier | `PROHIBITED_FIELD` |
| `client_order_id` | Client order ID | `PROHIBITED_FIELD` |
| `filled_qty` | Filled quantity | `PROHIBITED_FIELD` |
| `remaining_qty` | Remaining quantity | `PROHIBITED_FIELD` |
| `average_price` | Average fill price | `PROHIBITED_FIELD` |
| `commission` | Commission | `PROHIBITED_FIELD` |
| `fills` | Fill details | `PROHIBITED_FIELD` |
| `intent` | Trading intent | `INTENT_INJECTION` |
| `signal` | Trading signal | `INTENT_INJECTION` |
| `target_position` | Target position | `INTENT_INJECTION` |
| `target_qty` | Target quantity | `INTENT_INJECTION` |
| `target_allocation` | Target allocation | `INTENT_INJECTION` |
| `execute_order` | Execute flag | `INTENT_INJECTION` |
| `place_order` | Place flag | `INTENT_INJECTION` |
| `submit_order` | Submit flag | `INTENT_INJECTION` |
| `cancel_order` | Cancel flag | `INTENT_INJECTION` |
| `modify_order` | Modify flag | `INTENT_INJECTION` |
| `position_side` | Position side | `PROHIBITED_FIELD` |
| `position_size` | Position size | `PROHIBITED_FIELD` |
| `entry_price` | Entry price | `PROHIBITED_FIELD` |
| `exit_price` | Exit price | `PROHIBITED_FIELD` |
| `unrealized_pnl` | Unrealized P&L | `PROHIBITED_FIELD` |
| `realized_pnl` | Realized P&L | `PROHIBITED_FIELD` |
| `execution_id` | Execution ID | `PROHIBITED_FIELD` |
| `trade_id` | Trade ID | `PROHIBITED_FIELD` |
| `fill_price` | Fill price | `PROHIBITED_FIELD` |
| `fill_qty` | Fill quantity | `PROHIBITED_FIELD` |
| `slippage` | Slippage | `PROHIBITED_FIELD` |

#### PII Fields (Designed to be FORBIDDEN in Cloud; verify via redaction tests)
| Field | Reason | Violation Type |
|-------|--------|----------------|
| `email` | Personal data | `SENSITIVE_PII` |
| `phone` / `phone_number` | Personal data | `SENSITIVE_PII` |
| `address` | Personal data | `SENSITIVE_PII` |
| `ssn` / `social_security` | Personal data | `SENSITIVE_PII` |
| `tax_id` | Personal data | `SENSITIVE_PII` |
| `passport` | Personal data | `SENSITIVE_PII` |
| `driver_license` | Personal data | `SENSITIVE_PII` |
| `credit_card` / `card_number` | Financial data | `SENSITIVE_PII` |
| `cvv` | Financial data | `SENSITIVE_PII` |
| `account_number` | Financial data | `SENSITIVE_PII` |
| `routing_number` | Financial data | `SENSITIVE_PII` |
| `iban` / `swift` | Financial data | `SENSITIVE_PII` |
| `bank_account` | Financial data | `SENSITIVE_PII` |
| `date_of_birth` / `dob` | Personal data | `SENSITIVE_PII` |
| `ip_address` | Unless aggregated | `SENSITIVE_PII` |
| `mac_address` | Device data | `SENSITIVE_PII` |
| `device_id` | Unless hashed | `SENSITIVE_PII` |
| `user_agent` | Unless aggregated | `SENSITIVE_PII` |
| `password` | Credential | `BROKER_CREDENTIALS` |
| `secret` | Credential | `BROKER_CREDENTIALS` |

---

## 5. RAW_ORDER_EVENTS Level (Enterprise Only)

**Purpose**: Compliance audit trail for regulated enterprise customers.

**Default**: No - enterprise license + explicit opt-in required.

**Access Control**: Restricted RBAC + break-glass for support access.

**Storage**: Isolated enterprise storage with encryption.

**Retention**: Minimized per contract (typically 7-30 days).

**Reference**: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L853`, `#L1739`

### 5.1 Enterprise Gating Requirements

To use `RAW_ORDER_EVENTS`, ALL of the following must be true:

1. **Enterprise License**: Active enterprise subscription verified
2. **Explicit Opt-In**: Workspace-level opt-in recorded with audit trail
3. **Legal Agreement**: Enterprise DPA signed with RAW data addendum
4. **Access Controls**: RBAC configured for RAW data access
5. **Retention Policy**: Custom retention <= 30 days configured
6. **Encryption**: Customer-managed keys (CMK) enabled (optional)

### 5.2 Additional Allowed Fields (Enterprise RAW Only)

All fields from AGGREGATED and DETAILED_NON_SENSITIVE levels **plus**:

#### Order Fields
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `side` | string | Order side | `BUY`, `SELL` |
| `quantity` / `qty` | float | Order quantity | `100.0` |
| `price` | float | Order price | `150.50` |
| `order_type` | string | Order type | `LIMIT`, `MARKET` |
| `limit_price` | float | Limit price | `150.00` |
| `stop_price` | float | Stop price | `145.00` |
| `order_id` | string | Exchange order ID | `ord_123` |
| `client_order_id` | string | Client order ID | `client_ord_456` |
| `filled_qty` | float | Filled quantity | `100.0` |
| `remaining_qty` | float | Remaining quantity | `0.0` |
| `average_price` | float | Average fill price | `150.25` |
| `fill_price` | float | Individual fill price | `150.25` |
| `fill_qty` | float | Individual fill quantity | `50.0` |
| `execution_id` | string | Execution ID | `exec_789` |
| `trade_id` | string | Trade ID | `trade_abc` |
| `commission` | float | Commission charged | `1.50` |
| `slippage` | float | Slippage | `0.25` |

#### Position Fields
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `position_side` | string | Position side | `LONG`, `SHORT` |
| `position_size` | float | Position size | `500.0` |
| `entry_price` | float | Average entry price | `148.50` |
| `exit_price` | float | Exit price | `155.00` |
| `unrealized_pnl` | float | Unrealized P&L | `3250.00` |
| `realized_pnl` | float | Realized P&L | `1500.00` |

#### Signal/Intent Fields (Audit Trail)
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `signal` | string | Trading signal | `ENTER_LONG` |
| `intent` | string | Trading intent | `open_position` |
| `target_position` | float | Target position size | `1000.0` |
| `target_qty` | float | Target quantity | `500.0` |

#### Order Lifecycle
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `order_status` | string | Order status | `FILLED` |
| `order_created_at` | datetime | Order creation time | `2025-12-16T10:30:00Z` |
| `order_submitted_at` | datetime | Order submission time | `2025-12-16T10:30:01Z` |
| `order_filled_at` | datetime | Order fill time | `2025-12-16T10:30:02Z` |
| `order_cancelled_at` | datetime | Order cancellation time | `null` |

#### Execution Timing
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `signal_timestamp` | datetime | Signal generation time | `2025-12-16T10:29:59Z` |
| `submit_timestamp` | datetime | Order submit time | `2025-12-16T10:30:01Z` |
| `ack_timestamp` | datetime | Exchange ACK time | `2025-12-16T10:30:01.500Z` |
| `fill_timestamp` | datetime | Fill time | `2025-12-16T10:30:02Z` |
| `exchange_timestamp` | datetime | Exchange timestamp | `2025-12-16T10:30:02Z` |

### 5.3 Still Forbidden at RAW Level

Even with enterprise RAW access, the following are **ALWAYS FORBIDDEN**:

#### Credentials (CRITICAL)
| Field | Pattern | Reason |
|-------|---------|--------|
| `api_key` | `api[_-]?key` | Broker credential |
| `api_secret` | `api[_-]?secret` | Broker credential |
| `secret_key` | `secret[_-]?key` | Broker credential |
| `private_key` | `private[_-]?key` | Cryptographic key |
| `access_token` | `access[_-]?token` | Access token |
| `refresh_token` | `refresh[_-]?token` | Refresh token |
| `bearer_token` | `bearer[_-]?token` | Bearer token |
| `password` | - | Password |
| `passphrase` | - | Passphrase |
| Broker-specific | `(binance|alpaca|...)_(key|secret|token)` | Broker credential |

#### Environment Variables
| Pattern | Reason |
|---------|--------|
| `AWS_*` | Cloud credentials |
| `AZURE_*` | Cloud credentials |
| `GCP_*` | Cloud credentials |
| `DATABASE_*` | Database credentials |
| `REDIS_*` | Redis credentials |
| `*_PASSWORD` | Password |
| `*_TOKEN` | Token |
| `*_SECRET` | Secret |

---

## 6. Credential Detection Patterns

These patterns trigger `CRITICAL` violations and immediate rejection:

```python
BROKER_CREDENTIAL_PATTERNS = [
    (r"api[_-]?key", "API key"),
    (r"api[_-]?secret", "API secret"),
    (r"secret[_-]?key", "Secret key"),
    (r"private[_-]?key", "Private key"),
    (r"access[_-]?token", "Access token"),
    (r"refresh[_-]?token", "Refresh token"),
    (r"bearer[_-]?token", "Bearer token"),
    (r"password", "Password"),
    (r"passphrase", "Passphrase"),
    (r"(binance|alpaca|deribit|oanda|interactive_brokers|ib)[_-]?(key|secret|token)",
     "Broker credential"),
]
```

### 6.1 Credential Value Detection

Values matching these patterns are blocked regardless of field name:

| Pattern | Description |
|---------|-------------|
| `^[A-Za-z0-9+/]{32,}={0,2}$` | Base64 encoded (32+ chars) |
| `^[a-f0-9]{32,}$` | Hex string (32+ chars) |
| `^sk-[a-zA-Z0-9]{20,}$` | API key pattern (sk-...) |
| `^pk-[a-zA-Z0-9]{20,}$` | Public key pattern (pk-...) |
| `^[A-Z0-9]{20,}$` | AWS-style key |

---

## 7. Intent Injection Detection

The validator performs deep scanning to detect order-like structures:

### 7.1 Order Structure Detection

A payload containing all of these fields is considered an order structure:
- `side` AND `quantity` AND `price`

### 7.2 Intent Structure Detection

A payload containing:
- (`signal` OR `target`) AND (`position` OR `allocation` OR `order`)

---

## 8. Validation Rules Summary

| Rule ID | Rule | Severity | Telemetry Level | Action |
|---------|------|----------|-----------------|--------|
| `VAL-001` | Redaction required for non-AGGREGATED | CRITICAL | DETAILED, RAW | Block |
| `VAL-002` | Order fields in non-RAW | CRITICAL | AGGREGATED, DETAILED | Block |
| `VAL-003` | RAW without enterprise license | CRITICAL | RAW | Block |
| `VAL-004` | RAW without explicit opt-in | CRITICAL | RAW | Block |
| `VAL-005` | Credentials detected | CRITICAL | ALL | Block |
| `VAL-006` | PII in non-AGGREGATED | CRITICAL | DETAILED, RAW | Block |
| `VAL-007` | Intent injection | CRITICAL | AGGREGATED, DETAILED | Block |
| `VAL-008` | Unknown field (strict mode) | MEDIUM | DETAILED | Warn |
| `VAL-009` | Unredacted sensitive field | HIGH | DETAILED, RAW | Block |

---

## 9. Implementation References

### 9.1 Code Locations

| Component | File | Purpose |
|-----------|------|---------|
| Telemetry Validator | `packages/cloud/control_plane/middleware/telemetry_validation.py` | Cloud-side validation |
| Redaction Middleware | `ccea/telemetry/redaction.py` | Mandatory redaction |
| Protocol Models | `ccea/models/protocol.py` | Schema definitions |
| Telemetry Ingester | `packages/cloud/control_plane/telemetry_ingester.py` | Ingestion with validation |

### 9.2 CI Guardrails

| Guardrail | ID | Reference |
|-----------|-----|-----------|
| No order payloads in schema | BT-002 | `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md` |
| Redaction test | PM-003 | `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md` |
| Secret scan | PM-004 | `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md` |
| Redaction middleware | RT-005 | `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md` |

---

## 10. Change History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | CCEA Team | Initial version for GDPR Phase 2 |
