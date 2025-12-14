# Key Rotation Runbook

## Overview

This runbook provides procedures for rotating cryptographic keys used in CCEA.
Key rotation is a critical security practice that limits the impact of key compromise.

**Phase 10 Component: WI-VAULT-01**

## Key Types

| Key Type | Purpose | Storage Location | Rotation Frequency |
|----------|---------|------------------|-------------------|
| Master Vault Key | Encrypts broker credentials | OS Keychain | Annually or on compromise |
| Ed25519 Signing Key | Signs evidence packs | Cloud secret store | Annually |
| Agent Update Key | Signs agent updates | Cloud secret store | Annually |
| Artifact Signing Key | Signs build artifacts | Cloud secret store | Annually |

## Prerequisites

- Administrative access to the CCEA platform
- Access to OS keychain (for vault keys)
- Access to cloud secret store (for signing keys)
- Backup of current keys (encrypted)
- Maintenance window scheduled

## 1. Master Vault Key Rotation

The master vault key encrypts broker credentials stored locally on the agent.
This key NEVER leaves the agent machine.

### 1.1 Preparation

```bash
# Verify current key status
python -c "
from packages.agent.daemon.keychain import KeychainManager
km = KeychainManager()
print(km.get_key_info())
"
```

Expected output:
```
{
  'platform': 'Linux',
  'keychain_available': true,
  'keychain_has_key': true,
  'key_file_exists': false
}
```

### 1.2 Pre-Rotation Checklist

- [ ] Agent is not actively trading
- [ ] No pending trades or orders
- [ ] Broker connections are closed
- [ ] Backup of encrypted credentials exists

### 1.3 Rotation Procedure

```python
from packages.agent.daemon.keychain import KeychainManager
from packages.agent.vault.credential_manager import CredentialManager

# Initialize managers
km = KeychainManager()
cm = CredentialManager()

# Step 1: Get current key and decrypt credentials
old_key = km.get_master_key()

# Step 2: Export all credentials (in-memory only)
credentials = {}
for broker in cm.list_brokers():
    credentials[broker] = cm.get_broker_credentials(broker)

# Step 3: Rotate the master key
new_key = km.rotate_master_key()

# Step 4: Re-encrypt credentials with new key
cm.rotate_master_password(new_password_derived_from_new_key)

# Step 5: Verify rotation
for broker in credentials:
    assert cm.get_broker_credentials(broker) == credentials[broker]

print("Key rotation successful")
```

### 1.4 CLI Command

```bash
# Using the agent CLI
ccea-agent vault rotate-key

# Verify
ccea-agent vault status
```

### 1.5 Rollback Procedure

If rotation fails:

```python
from packages.agent.daemon.keychain import KeychainManager

km = KeychainManager()

# Restore from backup key file
backup_key_path = Path("~/.ccea/vault.key.backup")
if backup_key_path.exists():
    with open(backup_key_path, 'rb') as f:
        old_key = base64.b64decode(json.load(f)['key'])
    km.store_master_key(old_key)
```

## 2. Ed25519 Signing Key Rotation

Signing keys are used to cryptographically sign evidence packs and agent updates.

### 2.1 Generate New Key Pair

```python
from packages.cloud.enterprise.crypto import Ed25519Signer

signer = Ed25519Signer()

# Generate new key with 1-year expiry
new_key = signer.generate_key(
    key_id="evidence-pack-signer-v2",
    expires_in_days=365
)

# Export public key for distribution
public_key_pem = signer.export_public_key(new_key, format="pem")
print(public_key_pem.decode())

# Save private key securely
signer.save_key(
    new_key,
    Path("/etc/ccea/keys/evidence-pack-signer-v2.pem"),
    include_private=True,
    password=b"secure-password"  # Use strong password from secret store
)
```

### 2.2 Update Kubernetes Secret

```yaml
# signing-keys-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ccea-signing-keys
  namespace: ccea
type: Opaque
data:
  evidence-pack-signer.pem: <base64-encoded-private-key>
  evidence-pack-signer-v2.pem: <base64-encoded-new-private-key>
```

```bash
kubectl apply -f signing-keys-secret.yaml
```

### 2.3 Transition Period

During the transition period, both old and new keys should be valid for verification:

```python
from packages.cloud.enterprise.crypto import Ed25519Signer

signer = Ed25519Signer()

# Load both keys as trusted
old_key = signer.load_key(Path("/etc/ccea/keys/evidence-pack-signer.pem"))
new_key = signer.load_key(Path("/etc/ccea/keys/evidence-pack-signer-v2.pem"))

signer.add_trusted_key(old_key)
signer.add_trusted_key(new_key)

# Both keys can verify signatures
```

### 2.4 Remove Old Key

After transition period (typically 30-90 days):

```bash
# Remove old key from secret
kubectl patch secret ccea-signing-keys -n ccea \
  --type='json' \
  -p='[{"op": "remove", "path": "/data/evidence-pack-signer.pem"}]'

# Remove trusted key from config
# Update Helm values.yaml to remove old key from trustedKeys list
```

## 3. Agent Update Signing Key Rotation

Agent update signing keys are critical for supply chain security.

### 3.1 Pre-Rotation Steps

1. **Notify Stakeholders**: Announce key rotation schedule
2. **Update Documentation**: Document new key fingerprint
3. **Stage New Key**: Deploy new key to signing infrastructure

### 3.2 Rotation Procedure

```python
from packages.cloud.enterprise.crypto import Ed25519Signer

signer = Ed25519Signer(default_signer_id="ccea-agent-update-signer")

# Generate new key
new_key = signer.generate_key(
    key_id="agent-update-signer-2025",
    expires_in_days=365
)

# Export for distribution
public_key_b64 = new_key.public_key_base64
print(f"New public key (base64): {public_key_b64}")

# Save private key
signer.save_key(
    new_key,
    Path("/etc/ccea/keys/agent-update-signer-2025.pem"),
    include_private=True,
    password=b"strong-password"
)
```

### 3.3 Update Agent Configuration

Update agents to trust both old and new keys:

```yaml
# Agent config (Helm values or direct config)
agentUpdates:
  trustedSigningKeys:
    - "old-public-key-base64"
    - "new-public-key-base64"
```

### 3.4 Sign New Updates

All new updates should be signed with the new key:

```python
from packages.cloud.enterprise import AgentUpdateManager

manager = AgentUpdateManager()
update = manager.create_update(
    version="1.2.0",
    artifact_digest="sha256:...",
    artifact_url="https://...",
)

# Sign with new key
await manager.sign_update(
    update.id,
    new_signing_key_bytes,
    signer_id="ccea-agent-update-signer-2025"
)
```

## 4. Emergency Key Revocation

If a key is compromised, follow this emergency procedure:

### 4.1 Immediate Actions

1. **Alert Security Team**: Notify immediately
2. **Generate New Key**: Create replacement key
3. **Revoke Old Key**: Remove from all trust stores

### 4.2 Revocation Procedure

```python
from packages.cloud.enterprise.crypto import Ed25519Signer

signer = Ed25519Signer()

# Remove compromised key from trusted keys
signer.remove_trusted_key("compromised-key-id")

# Update all configuration
# - Kubernetes secrets
# - Helm values
# - Agent configs
```

### 4.3 Audit Trail

Document the incident:

```json
{
  "incident_type": "key_compromise",
  "key_id": "compromised-key-id",
  "detected_at": "2025-01-15T10:30:00Z",
  "revoked_at": "2025-01-15T10:35:00Z",
  "replacement_key_id": "new-key-id",
  "affected_systems": ["evidence-pack-signer"],
  "remediation_steps": [
    "Generated new key pair",
    "Updated Kubernetes secrets",
    "Deployed to production",
    "Verified all services"
  ]
}
```

## 5. Verification Procedures

### 5.1 Verify Key Rotation

```python
from packages.cloud.enterprise.crypto import Ed25519Signer, verify_data

# Test signing with new key
signer = Ed25519Signer()
key = signer.load_key(Path("/etc/ccea/keys/new-key.pem"))

test_data = b"test message"
signature = signer.sign(test_data, key)

# Verify
assert signer.verify(test_data, signature, key)
print("Key verification successful")
```

### 5.2 Verify Evidence Pack Signing

```python
from packages.cloud.enterprise import EvidencePackExporter, EvidencePackConfig

config = EvidencePackConfig(
    signing_enabled=True,
    signing_key_path=Path("/etc/ccea/keys/new-key.pem")
)
exporter = EvidencePackExporter(config)

# Create test pack
pack = await exporter.create_pack(
    evidence_types=[EvidenceType.SYSTEM_METRICS],
    time_range=(start_time, end_time)
)

# Verify pack signature
is_valid = await exporter.verify_pack(pack)
assert is_valid, "Pack signature verification failed"
```

## 6. Scheduled Maintenance

### 6.1 Annual Key Rotation Schedule

| Quarter | Keys to Rotate |
|---------|----------------|
| Q1 | Agent Update Signing Key |
| Q2 | Evidence Pack Signing Key |
| Q3 | Artifact Signing Key |
| Q4 | Master Vault Key (optional) |

### 6.2 Pre-Rotation Checklist

- [ ] Notify stakeholders 2 weeks in advance
- [ ] Schedule maintenance window
- [ ] Generate new keys in staging
- [ ] Test rotation procedure in staging
- [ ] Backup current keys
- [ ] Document new key fingerprints
- [ ] Update runbook if needed

### 6.3 Post-Rotation Checklist

- [ ] Verify all services using new keys
- [ ] Update documentation with new key fingerprints
- [ ] Archive old keys (encrypted, offline)
- [ ] Update monitoring alerts
- [ ] Notify stakeholders of completion

## 7. Troubleshooting

### 7.1 Common Issues

**Issue: Keychain not available**
```
KeychainNotAvailableError: OS keychain not available
```
Solution: Install OS keychain support
- macOS: Built-in
- Linux: `apt install gnome-keyring` or `secret-tool`
- Windows: Built-in Credential Manager

**Issue: Signature verification fails**
```
VerificationError: Invalid signature
```
Solution: Check that the correct public key is configured

**Issue: Key file permission denied**
```
PermissionError: [Errno 13] Permission denied
```
Solution: Check file permissions (should be 0600 for private keys)

### 7.2 Recovery Procedures

If key rotation fails mid-process:

1. Check key backup exists
2. Restore old key from backup
3. Verify services are functional
4. Investigate failure cause
5. Retry rotation with fixes

## 8. Security Considerations

- **Never log private keys** or key material
- **Use secure channels** for key distribution
- **Encrypt backups** with strong passwords
- **Limit access** to key rotation procedures
- **Audit all key operations** in security logs
- **Test recovery procedures** regularly

## References

- [RFC 8032: Edwards-Curve Digital Signature Algorithm (EdDSA)](https://tools.ietf.org/html/rfc8032)
- [NIST SP 800-57: Key Management Guidelines](https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final)
- [CIS Controls v8: Control 10 - Malware Defenses](https://www.cisecurity.org/controls/malware-defenses)
