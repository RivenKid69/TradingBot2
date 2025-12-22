# Production Deployment Security Checklist

**Version**: 1.0
**Date**: 2025-12-21
**Status**: Active
**Canon Reference**: `docs/DOCUMENTATION_CANON_DESIGN.md`

---

## Purpose

This checklist defines mandatory security requirements for production deployment of CustodiaCloud Cloud Control Plane. Deployment without completing these checks may result in security vulnerabilities.

**Important**: This checklist is designed to support secure deployment practices. Actual security depends on proper implementation and ongoing operational controls.

---

## Mandatory Environment Variables

The following environment variables MUST be set in production. The application will **fail to start** (fail-closed) if these are not configured.

### Authentication

| Variable | Requirement | Validation |
|----------|-------------|------------|
| `CCEA_ENV` | Set to `production` | Enables production security checks |
| `CCEA_JWT_SECRET` | Minimum 32 bytes, cryptographically random | Fail-closed: app refuses to start with default value in production |
| `CCEA_JWT_ALGORITHM` | `HS256` (default) or `RS256` for asymmetric | Validated at startup |

**Generate secure JWT secret**:
```bash
# Option 1: OpenSSL
openssl rand -base64 32

# Option 2: Python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Database

| Variable | Requirement | Notes |
|----------|-------------|-------|
| `CCEA_DATABASE_URL` | PostgreSQL connection string | Must not use SQLite in production |
| `CCEA_DB_POOL_SIZE` | Integer (recommended: 10-20) | Based on expected load |
| `CCEA_DB_MAX_OVERFLOW` | Integer (recommended: 20-40) | For burst handling |

### Security Features

| Variable | Requirement | Notes |
|----------|-------------|-------|
| `CCEA_SKIP_SIGNATURE_VERIFICATION` | Must NOT be set in production | CI check enforces this |
| `ALLOW_UNSAFE_MODEL_LOAD` | Must NOT be set in production | Model loading security |

---

## Pre-Deployment Checklist

### 1. Secrets Management

- [ ] JWT_SECRET generated and stored in secure secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)
- [ ] Database credentials stored in secrets manager
- [ ] No secrets in environment files committed to git
- [ ] `.secrets.baseline` updated and reviewed

### 2. Network Security

- [ ] TLS/HTTPS enabled for all endpoints
- [ ] Database connections use TLS
- [ ] Firewall rules restrict access to necessary ports only
- [ ] Private subnets for database and internal services

### 3. Authentication/Authorization

- [ ] MFA enabled for administrative accounts
- [ ] JWT token expiration configured appropriately (default: 24h)
- [ ] Rate limiting enabled
- [ ] Account lockout policy configured

### 4. Monitoring and Logging

- [ ] Audit logging enabled
- [ ] Security event alerting configured
- [ ] Log retention policy defined
- [ ] No sensitive data in logs (redaction verified)

### 5. Backup and Recovery

- [ ] Database backup schedule configured
- [ ] Backup encryption enabled
- [ ] Recovery procedure documented and tested
- [ ] DR drill schedule established

---

## CI/CD Enforcement

The following checks are enforced in CI to prevent insecure deployments:

1. **Secret Detection**: `detect-secrets` scan blocks commits with potential secrets
2. **Signature Verification Bypass**: CI fails if `CCEA_SKIP_SIGNATURE_VERIFICATION` is detected in production configs
3. **Model Loading Security**: Static analysis checks for unsafe `torch.load` calls
4. **Dependency Audit**: `pip-audit` and SBOM generation

---

## Validation Commands

Run these commands before deployment:

```bash
# Verify no default secrets
grep -r "dev-secret-change" . --include="*.py" --include="*.yaml" --include="*.yml"

# Verify no bypass flags in production configs
grep -r "CCEA_SKIP_SIGNATURE" deploy/production/

# Run security scans (via CI: .github/workflows/security-sast.yml)
# Local equivalents:
python -m bandit -r . -c .bandit -f txt
# Or run full CI workflow locally via act (if installed)

# Verify SBOM generated (via CI: security-sast.yml uses cyclonedx-py)
# Local equivalent (generates sbom.json from lockfiles):
cyclonedx-py --format json --output sbom.json --requirements requirements-cpu.lock.txt

# CI artifacts available after workflow run:
# - bandit-results.json, semgrep-results.json (security scans)
# - sbom.json, sbom-verification.json (SBOM with hash)
# See: .github/workflows/security-sast.yml for full pipeline
```

---

## Fail-Closed Behavior

The following components implement fail-closed behavior:

| Component | Behavior | Code Location |
|-----------|----------|---------------|
| JWT Secret | App refuses to start with default secret in production | `packages/cloud/control_plane/dependencies.py:44-50` |
| MFA Verification | Returns False if pyotp unavailable (not bypassed) | `packages/cloud/control_plane/routers/auth.py:256-263` |
| Signature Verification | Returns False if cryptography unavailable | `packages/cloud/enterprise/registry_mirror.py:751-801` |
| Model Loading | `weights_only=True` by default | `docs/security/THREAT_MODEL_MODEL_LOADING.md` |

---

## Document Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-21 | Initial version - JWT fail-closed implementation |

**Owner**: Engineering / Security
**Review Frequency**: Quarterly or upon security-relevant changes
**Classification**: Internal

---

*This document follows the Documentation Canon - no absolute claims about security guarantees.*
