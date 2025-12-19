# Local Vault: Credential Storage

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Overview

The Local Vault securely stores broker API credentials on the agent's local system. The CCEA architecture is **designed so credentials remain in the customer-controlled Agent** and are **designed not to be transmitted to Cloud** (enforced via protocol schema and Agent implementation).

## Security Design Commitments

```
Local Vault DESIGN COMMITMENTS (enforced at architecture level):
  - Credentials designed to be stored locally (architecture designed so they are not transmitted to Cloud)
  - Encryption at rest (AES-256-GCM)
  - OS keychain integration when available
  - Automatic redaction in logs/telemetry
  - Master key designed not to be transmitted to Cloud (enforced via Agent implementation)
```

---

## Storage Backends

### 1. OS Keychain (Recommended)

Uses the operating system's secure credential storage:

| OS | Keychain |
|----|----------|
| macOS | Keychain (Security.framework) |
| Linux | GNOME Keyring / KDE Wallet / libsecret |
| Windows | Credential Manager (DPAPI) |

**Advantages:**
- Protected by OS-level security
- Integrates with biometrics (Touch ID, Windows Hello)
- No separate master key needed
- Hardware-backed on supported devices

**Configuration:**
```yaml
vault:
  backend: keychain
```

### 2. Encrypted File (Fallback)

For environments without keychain support:

**Configuration:**
```yaml
vault:
  backend: encrypted_file
  file_path: ~/.ccea/vault.enc
  encryption_key_source: env  # or: file, prompt
```

**Encryption:**
- Algorithm: AES-256-GCM
- Key derivation: Argon2id
- Unique salt per vault

---

## Credential Management

### Adding Broker Credentials

**Interactive:**
```bash
ccea-agent vault add-broker
# Prompts for:
# - Broker name (binance, alpaca, etc.)
# - API key
# - API secret
# - Passphrase (if applicable)
# - Label (for identification)
```

**Non-interactive:**
```bash
ccea-agent vault add-broker \
  --broker binance \
  --api-key "your_api_key" \
  --api-secret "your_api_secret" \
  --label "main-trading"
```

**From environment:**
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"

ccea-agent vault add-broker \
  --broker binance \
  --api-key-env BINANCE_API_KEY \
  --api-secret-env BINANCE_API_SECRET \
  --label "main-trading"
```

### Listing Credentials

```bash
ccea-agent vault list

# Output:
# ID          BROKER    LABEL           ADDED
# cred_abc    binance   main-trading    2025-12-14
# cred_xyz    alpaca    paper-trading   2025-12-10
```

**Note:** API keys/secrets are designed not to be displayed (redacted by default).

### Removing Credentials

```bash
ccea-agent vault remove --id cred_abc

# Or by label
ccea-agent vault remove --label "main-trading"
```

### Verifying Credentials

```bash
ccea-agent vault verify --id cred_abc

# Tests:
# - API key format valid
# - Broker connectivity
# - Permissions check
```

---

## Multi-Account Support

The vault supports multiple accounts per broker:

```bash
# Add production account
ccea-agent vault add-broker \
  --broker binance \
  --api-key $PROD_KEY \
  --api-secret $PROD_SECRET \
  --label "production"

# Add paper trading account
ccea-agent vault add-broker \
  --broker binance \
  --api-key $PAPER_KEY \
  --api-secret $PAPER_SECRET \
  --label "paper-trading"
```

**Selecting Account for Deployment:**
```yaml
# deployment config
broker:
  name: binance
  credential_label: production  # or credential_id
```

---

## Encryption Key Management

### Environment Variable (Default)

```bash
# Generate key
openssl rand -base64 32 > ~/.ccea/vault.key
chmod 600 ~/.ccea/vault.key

# Set environment variable
export CCEA_VAULT_KEY=$(cat ~/.ccea/vault.key)
```

### File-Based Key

```yaml
vault:
  backend: encrypted_file
  encryption_key_source: file
  encryption_key_file: ~/.ccea/vault.key
```

### Prompt for Key

```yaml
vault:
  backend: encrypted_file
  encryption_key_source: prompt
```

Agent will prompt for the key at startup.

### Key Rotation

```bash
# Rotate vault encryption key
ccea-agent vault rotate-key

# This will:
# 1. Decrypt vault with old key
# 2. Re-encrypt with new key
# 3. Update key storage
```

---

## Security Best Practices

### DO

1. **Use OS keychain when available**
   ```yaml
   vault:
     backend: keychain
   ```

2. **Restrict API key permissions**
   - Trading only (no withdrawal)
   - IP whitelist if supported
   - Read-only for data-only operations

3. **Use separate keys per environment**
   - Production vs paper trading
   - Different strategies

4. **Rotate credentials regularly**
   ```bash
   ccea-agent vault rotate-credentials --id cred_abc
   ```

5. **Audit vault access**
   ```bash
   ccea-agent vault audit-log
   ```

### DON'T

1. **Do not share vault files**
   - Vault is tied to local machine
   - Should not be transferred

2. **Do not log credentials**
   - Agent automatically redacts
   - Custom code must also redact

3. **Do not store in version control**
   - Add `~/.ccea/` to `.gitignore`
   - Do not commit `.env` files

4. **Credentials designed not to transmit to Cloud**
   - Agent architecture enforces this
   - Cloud API designed without credential-accepting endpoints

---

## Redaction

All credential values are automatically redacted:

### Log Redaction

```
# Before redaction (internal):
Connecting to broker with API key: abc123def456...

# After redaction (in logs):
Connecting to broker with API key: [REDACTED:API_KEY]
```

### Telemetry Redaction

```python
# Telemetry event before:
{
  "broker": "binance",
  "api_key": "abc123def456",
  "account_id": "12345678"
}

# After redaction:
{
  "broker": "binance",
  "api_key": "[REDACTED]",
  "account_id": "[REDACTED:4_CHARS]78"
}
```

### Redaction Patterns

| Pattern | Example | Redacted |
|---------|---------|----------|
| API key | `abc123def456` | `[REDACTED:API_KEY]` |
| API secret | `secret123` | `[REDACTED:API_SECRET]` |
| Account ID | `12345678` | `[REDACTED:4_CHARS]78` |
| IP address | `192.168.1.1` | `192.168.x.x` |
| Email | `user@example.com` | `u***@example.com` |

---

## Vault File Format

The encrypted vault file structure:

```
┌─────────────────────────────────────────────────────┐
│ Header (32 bytes)                                   │
│  - Magic: "CCEA_VAULT" (10 bytes)                  │
│  - Version: 1 (2 bytes)                            │
│  - Salt: (16 bytes)                                │
│  - Reserved: (4 bytes)                             │
├─────────────────────────────────────────────────────┤
│ Encrypted Payload                                   │
│  - AES-256-GCM encrypted JSON                      │
│  - Contains: credentials, metadata                 │
├─────────────────────────────────────────────────────┤
│ Auth Tag (16 bytes)                                │
│  - GCM authentication tag                          │
└─────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Vault locked" | Missing encryption key | Set CCEA_VAULT_KEY |
| "Keychain access denied" | OS permission | Grant keychain access |
| "Credential invalid" | Expired/revoked API key | Regenerate at broker |
| "Broker connection failed" | Network/IP whitelist | Check network settings |

### Diagnostics

```bash
# Check vault status
ccea-agent vault status

# Verify specific credential
ccea-agent vault verify --id cred_abc --verbose

# Test broker connectivity
ccea-agent doctor --check broker

# View vault audit log
ccea-agent vault audit-log --tail 50
```

### Recovery

If vault is corrupted:

```bash
# Export credentials (requires unlock)
ccea-agent vault export --output credentials.enc

# Reset vault
ccea-agent vault reset

# Re-import
ccea-agent vault import --input credentials.enc
```

---

## API Reference

### Python API

```python
from ccea_agent.vault import CredentialVault

# Initialize vault
vault = CredentialVault()

# Add credential
vault.add_broker_credential(
    broker="binance",
    api_key="your_key",
    api_secret="your_secret",
    label="main"
)

# Get credential (for internal use only)
cred = vault.get_credential("cred_abc")

# Use credential with broker client
client = vault.get_broker_client("cred_abc")
```

### CLI Commands

```bash
# Vault management
ccea-agent vault status          # Show vault status
ccea-agent vault list            # List credentials
ccea-agent vault add-broker      # Add broker credential
ccea-agent vault remove          # Remove credential
ccea-agent vault verify          # Verify credential
ccea-agent vault rotate-key      # Rotate encryption key
ccea-agent vault export          # Export vault (encrypted)
ccea-agent vault import          # Import vault
ccea-agent vault audit-log       # View access log
ccea-agent vault reset           # Reset vault (DANGER)
```

---

**Related Documentation:**
- [Installation](./INSTALLATION.md)
- [Risk Controls](./RISK_CONTROLS.md)
- [Security Trust Center](../security/TRUST_CENTER.md)
