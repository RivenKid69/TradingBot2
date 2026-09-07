# Distributed Security Requirements

**Version**: 1.0
**Date**: 2025-12-20
**Status**: Active
**Canon Reference**: `docs/DOCUMENTATION_CANON_DESIGN.md`

---

## Purpose

This document defines security requirements for multi-instance/distributed deployments of CustodiaCloud. These requirements ensure consistent security controls across all instances.

---

## 1. MFA Token Storage

### Current Implementation

- In-memory storage in single process
- Location: `packages/cloud/control_plane/routers/auth.py`

### Single-Instance Deployment

- In-memory storage is acceptable
- Tokens survive for process lifetime
- Restart clears pending MFA tokens (users must re-authenticate)

### Multi-Instance Deployment Requirements

1. **Backend**: Redis or PostgreSQL for token storage
2. **TTL**: Tokens expire after 5 minutes (configurable)
3. **Monitoring**: Track `mfa_pending_token_count` metric
4. **Audit**: Log token issuance and redemption

### Control Metrics

- `mfa_pending_token_count`: Current pending tokens
- `mfa_token_issued_total`: Total tokens issued
- `mfa_token_redeemed_total`: Total tokens successfully redeemed
- `mfa_token_expired_total`: Total tokens that expired unused

---

## 2. JWT Revocation (JTI Blocklist)

### Current Implementation

- In-memory LRU blocklist
- Location: `packages/cloud/control_plane/security/jwt_revocation.py`

### Single-Instance Deployment

- In-memory blocklist is acceptable
- Revocations are immediate within the instance
- Restart clears blocklist (tokens resume validity until expiry)

### Multi-Instance Deployment Requirements

1. **Backend**: Redis with pub/sub for revocation propagation
2. **Consistency**: Revocation must propagate within 1 second
3. **Monitoring**: Track revocation distribution latency
4. **Fallback**: Short JWT TTL (15 minutes) as defense-in-depth

### Control Metrics

- `jwt_revoked_count`: Total revocations
- `jwt_revocation_check_count`: Total revocation checks
- `jwt_revocation_propagation_latency_ms`: Time to propagate across instances

---

## 3. Rate Limiting

### Current Implementation

- In-memory rate limiting with account lockout
- Location: `packages/cloud/control_plane/security/rate_limiter.py`

### Single-Instance Deployment

- In-memory rate limiting is acceptable
- All requests hit the same counter

### Multi-Instance Deployment Requirements

1. **Backend**: Redis with atomic operations (INCR, EXPIRE)
2. **Consistency**: Rate counts must be synchronized across instances
3. **Algorithm**: Sliding window or token bucket with Redis
4. **Monitoring**: Track bypass attempts

### Security Risk Without Distributed State

Attackers can bypass rate limits by distributing requests across instances.
Without Redis backend, effective rate limit = configured_limit * instance_count.

### Control Metrics

- `rate_limit_exceeded_count`: Total rate limit violations
- `account_lockout_count`: Total account lockouts
- `account_lockout_duration_seconds`: Distribution of lockout durations

---

## 4. Artifact Signature Verification

### Current Implementation

- Fail-closed implementation
- Location: `packages/cloud/enterprise/registry_mirror.py`

### Production Requirements

1. **Cosign/Sigstore**: Use sigstore for production verification
2. **Key Management**: Trusted keys in secure key store
3. **Development Only**: `CCEA_SKIP_SIGNATURE_VERIFICATION=DEVELOPMENT_ONLY`
4. **Monitoring**: Alert on any signature verification failures

### Control Metrics

- `signature_verification_success_total`: Successful verifications
- `signature_verification_failed_total`: Failed verifications
- `signature_verification_skipped_total`: Skipped (dev only)

---

## 5. Agent Update Signing

### Current Implementation

- Fail-closed implementation requiring cryptography library
- Location: `packages/cloud/enterprise/agent_updates.py`

### Production Requirements

1. **Cryptography Library**: Mandatory (no fallback)
2. **Key Management**: Ed25519 signing keys in HSM or secure vault
3. **Rotation**: Key rotation schedule (annually minimum)
4. **Audit**: Log all signing operations

### Control Metrics

- `agent_update_signed_total`: Total updates signed
- `agent_update_signature_verified_total`: Total verifications
- `agent_update_signature_failed_total`: Failed verifications

---

## Deployment Checklist

### Single-Instance (Development/Staging)

- [ ] In-memory storage acceptable for all components
- [ ] Restart clears state (acceptable for non-production)
- [ ] Monitor memory usage for blocklist/rate limit storage

### Multi-Instance (Production)

- [ ] Redis deployed and configured
- [ ] MFA tokens in Redis with TTL
- [ ] JWT revocation in Redis with pub/sub
- [ ] Rate limiting in Redis with atomic operations
- [ ] Monitoring dashboards for all control metrics
- [ ] Alerting configured for security events

---

## Document Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-20 | Initial version covering MFA, JWT revocation, rate limiting |

**Review Frequency**: Quarterly or upon architectural changes
**Owner**: Security Engineering
**Classification**: Internal

---

*This document follows the Documentation Canon - no absolute claims, honest disclosure of limitations.*
