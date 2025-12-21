# Artifact Builder & Registry Guide

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Overview

The Artifact Builder is designed to create immutable, signed trading strategy artifacts. Artifacts are designed to be:
- **Digest-pinned**: Content-addressable by SHA256
- **Signed**: Cryptographically signed with cosign/GPG
- **Documented**: Includes SBOM and provenance
- **Versioned**: Schema version for compatibility

## Build Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ARTIFACT BUILD PIPELINE                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  1. Source        2. Build          3. Sign           4. Publish             │
│  ┌─────────┐     ┌─────────┐       ┌─────────┐       ┌─────────┐            │
│  │ Git Repo│────▶│ Builder │──────▶│ Signing │──────▶│Registry │            │
│  │ (pinned)│     │ (OCI)   │       │(cosign) │       │(digest) │            │
│  └─────────┘     └─────────┘       └─────────┘       └─────────┘            │
│       │               │                 │                 │                  │
│       ▼               ▼                 ▼                 ▼                  │
│  ┌─────────┐     ┌─────────┐       ┌─────────┐       ┌─────────┐            │
│  │ Deps    │     │Manifest │       │  SBOM   │       │Artifact │            │
│  │ Lock    │     │ (JSON)  │       │(CycloneDX)      │  Ready  │            │
│  └─────────┘     └─────────┘       └─────────┘       └─────────┘            │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Artifact Formats

### OCI Image (Preferred)

The primary artifact format is an OCI-compliant container image:

```dockerfile
# Example Dockerfile for strategy artifact
FROM python:3.12-slim

# Install dependencies (digest-pinned)
COPY requirements.lock /app/
RUN pip install --no-cache-dir -r requirements.lock

# Copy strategy code
COPY strategy/ /app/strategy/

# Security: Non-root user
RUN useradd -r strategy
USER strategy

WORKDIR /app
ENTRYPOINT ["python", "-m", "strategy.main"]
```

**Benefits:**
- Immutable by content hash
- Reproducible builds
- Sandboxing support
- Standard tooling

### ZIP Bundle (Fallback)

For environments without container support:

```
strategy-v1.0.0.zip
├── manifest.json       # Required: artifact metadata
├── requirements.lock   # Pinned dependencies
├── strategy/
│   ├── __init__.py
│   ├── main.py        # Entrypoint
│   └── models/
├── signature.sig       # Detached signature
└── sbom.json          # Software Bill of Materials
```

## Manifest Schema

Every artifact includes a `manifest.json`:

```json
{
  "schema_version": "1.0.0",
  "artifact_id": "strategy_momentum_v1",
  "artifact_digest": "sha256:a1b2c3d4...",

  "entrypoint": "strategy.main:run",
  "runtime": "python:3.12",

  "deps_lock_digest": "sha256:deps123...",
  "model_refs": [
    {
      "name": "policy_network",
      "digest": "sha256:model456..."
    }
  ],

  "data_contract": {
    "required_features": ["close", "volume", "ma_20"],
    "output_type": "OrderIntent"
  },

  "permissions": {
    "filesystem": "read_only",
    "network": "none",
    "max_memory_mb": 1024,
    "max_cpu_percent": 50
  },

  "risk_profile_suggested": {
    "max_position_pct": 10,
    "max_daily_loss_pct": 2,
    "allowed_order_types": ["LIMIT", "MARKET"]
  },

  "live_capabilities": {
    "requires_broker_access": true,
    "supported_brokers": ["binance", "alpaca"],
    "required_sandbox": "docker"
  },

  "telemetry_schema_version": "1.0.0",
  "change_class": "TRADING_IMPACTING",

  "provenance": {
    "git_repo": "https://github.com/org/strategies",
    "git_sha": "abc123def456...",
    "git_branch": "main",
    "build_timestamp": "2025-12-14T10:00:00Z",
    "builder_version": "1.0.0",
    "dataset_refs": [],
    "training_run_id": null,
    "params_hash": null
  },

  "signature": "...",
  "sbom_ref": "sbom:sha256:xxx"
}
```

### Schema Version Compatibility

| Agent Schema | Manifest Schema | Compatible? |
|--------------|-----------------|-------------|
| 1.0.x | 1.0.x | Yes |
| 1.0.x | 1.1.x | Yes (forward compatible) |
| 1.0.x | 2.0.x | No (major version mismatch) |
| 1.1.x | 1.0.x | Yes (backward compatible) |

## Signing Process

### Using Cosign (Recommended)

```bash
# Generate key pair (one-time setup)
cosign generate-key-pair

# Sign the artifact
cosign sign --key cosign.key \
  registry.ccea.cloud/strategies/momentum@sha256:a1b2c3d4

# Verify signature
cosign verify --key cosign.pub \
  registry.ccea.cloud/strategies/momentum@sha256:a1b2c3d4
```

### Using GPG (Alternative)

```bash
# Sign manifest
gpg --armor --detach-sign manifest.json

# Verify
gpg --verify manifest.json.asc manifest.json
```

### Keyless Signing (Sigstore)

For public transparency log:

```bash
cosign sign --keyless \
  registry.ccea.cloud/strategies/momentum@sha256:a1b2c3d4
```

## SBOM Generation

Every artifact includes a Software Bill of Materials:

```bash
# Generate SBOM with syft
syft packages dir:./strategy -o cyclonedx-json > sbom.json

# Or with trivy
trivy sbom --format cyclonedx ./strategy > sbom.json
```

### SBOM Contents

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "serialNumber": "urn:uuid:xxx",
  "version": 1,
  "components": [
    {
      "type": "library",
      "name": "numpy",
      "version": "1.26.0",
      "purl": "pkg:pypi/numpy@1.26.0",
      "hashes": [
        {
          "alg": "SHA-256",
          "content": "abc123..."
        }
      ]
    }
  ]
}
```

## Registry Operations

### Publishing Artifacts

```bash
# Build OCI image
docker build -t strategy:v1.0.0 .

# Tag with digest
docker tag strategy:v1.0.0 registry.ccea.cloud/ws_abc/strategy@sha256:xxx

# Push to registry
docker push registry.ccea.cloud/ws_abc/strategy@sha256:xxx

# Sign after push
cosign sign --key cosign.key registry.ccea.cloud/ws_abc/strategy@sha256:xxx
```

### Registry API

```bash
# List artifacts
curl https://registry.ccea.cloud/v2/ws_abc/strategy/tags/list

# Get manifest
curl https://registry.ccea.cloud/v2/ws_abc/strategy/manifests/sha256:xxx

# Get SBOM
curl https://registry.ccea.cloud/v2/ws_abc/strategy/sbom/sha256:xxx
```

### Access Control

| Role | Permissions |
|------|-------------|
| `viewer` | Pull artifacts |
| `developer` | Pull, push artifacts |
| `admin` | Pull, push, delete, manage keys |

## Agent Verification

Agents MUST verify artifacts before execution:

### Verification Checklist

```python
def verify_artifact(artifact_ref: str, manifest: dict) -> bool:
    """
    Agent-side artifact verification.
    All checks must pass before execution.
    """
    checks = [
        # 1. Digest matches
        verify_digest(artifact_ref, manifest["artifact_digest"]),

        # 2. Signature valid
        verify_signature(artifact_ref, manifest["signature"]),

        # 3. Registry in allowlist
        verify_registry_allowlist(artifact_ref),

        # 4. Schema version compatible
        verify_schema_version(manifest["schema_version"]),

        # 5. SBOM exists and valid
        verify_sbom(manifest["sbom_ref"]),

        # 6. No unsigned dependencies
        verify_deps_signatures(manifest["deps_lock_digest"]),
    ]

    return all(checks)
```

### Rejection Reasons

| Reason | Action |
|--------|--------|
| Invalid digest | Reject, alert |
| Invalid signature | Reject, alert |
| Unknown registry | Reject |
| Incompatible schema | Reject |
| Missing SBOM | Reject (configurable) |
| Expired signature | Reject (enterprise) |

## Key Management

### Trust Hierarchy

```
Root CA (offline, HSM)
    │
    ├── Builder Signing Key
    │   └── Signs artifacts
    │
    └── Registry Signing Key
        └── Signs manifests
```

### Key Rotation

```bash
# Generate new key
cosign generate-key-pair --output-key-prefix new

# Update trust root
ccea-admin trust add-key --key new.pub --effective-date 2025-01-01

# Deprecate old key (grace period)
ccea-admin trust deprecate-key --key old.pub --sunset-date 2025-02-01
```

### Enterprise: Customer-Managed Keys

```yaml
# enterprise-config.yaml
signing:
  mode: customer_managed
  kms_provider: aws
  key_arn: arn:aws:kms:eu-central-1:123456:key/xxx
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Sign Artifact

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build OCI Image
        run: |
          docker build -t strategy:${{ github.ref_name }} .

      - name: Generate SBOM
        uses: anchore/sbom-action@v0
        with:
          image: strategy:${{ github.ref_name }}
          format: cyclonedx-json

      - name: Push to Registry
        run: |
          docker tag strategy:${{ github.ref_name }} \
            registry.ccea.cloud/strategies/${{ github.ref_name }}
          docker push registry.ccea.cloud/strategies/${{ github.ref_name }}

      - name: Sign Artifact
        uses: sigstore/cosign-installer@main
        run: |
          cosign sign --key ${{ secrets.COSIGN_KEY }} \
            registry.ccea.cloud/strategies/${{ github.ref_name }}
```

## Best Practices

### DO

- Always pin dependencies by digest
- Include comprehensive SBOM
- Sign with short-lived keys (enterprise)
- Verify artifacts on every deployment
- Use minimal base images
- Run as non-root user

### DON'T

- Include secrets in artifacts
- Use `latest` tags
- Skip signature verification
- Include unnecessary dependencies
- Use writable filesystems
- Run with elevated privileges

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Signature verification failed | Check key is in trust store |
| Schema version incompatible | Update agent or rebuild artifact |
| SBOM missing | Regenerate with syft/trivy |
| Registry access denied | Check RBAC permissions |
| Digest mismatch | Rebuild with pinned deps |

---

**Related Documentation:**
- [JSON Schemas](../schemas/README.md)
- [Agent Verification](../agent/README.md)
- [Security Best Practices](../security/TRUST_CENTER.md)
