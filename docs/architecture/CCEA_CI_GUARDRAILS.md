# CCEA CI Guardrails

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16
>
> **Reference**: Design Doc CCEA Cloud.txt (canonical source) - Section 19

## Overview

This document defines the CI/CD guardrails that enforce the CCEA security model at build time. These are not tests - they are hard blocks that prevent the possibility of violating the Cloud/Agent boundary.

**Philosophy:** Cut the possibility of violation at the build level, not just test for it.

---

## 1. Core Guardrails

### 1.1 No Broker Clients in Cloud

**Rule:** Cloud services cannot depend on broker/exchange trading client packages.

**Implementation:**

> **DOCS/DRIFT Note (CCEA-DOC-001):** The workflow below is a *recommended* configuration.
> Current CI uses `security-sast.yml` for security checks. Full guardrails workflow
> to be created per this specification.

```yaml
# .github/workflows/guardrails.yml (RECOMMENDED - to be created)
name: CCEA Guardrails

on: [push, pull_request]

jobs:
  no-broker-in-cloud:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check Cloud imports
        run: |
          echo "Checking for prohibited imports in Cloud packages..."

          # Prohibited patterns
          PROHIBITED=(
            "from adapters.binance.order"
            "from adapters.alpaca.order"
            "from adapters.oanda.order"
            "import ccxt"
            "from ibapi"
            "from alpaca_trade_api"
            "BrokerOrderSubmitter"
            "LiveOrderExecutor"
          )

          # Cloud packages to check
          CLOUD_PATHS=(
            "packages/cloud/"
            "services/control_plane/"
            "services/backtest/"
            "services/training/"
          )

          VIOLATIONS=0

          for path in "${CLOUD_PATHS[@]}"; do
            for pattern in "${PROHIBITED[@]}"; do
              if grep -r "$pattern" "$path" 2>/dev/null; then
                echo "VIOLATION: '$pattern' found in $path"
                VIOLATIONS=$((VIOLATIONS + 1))
              fi
            done
          done

          if [ $VIOLATIONS -gt 0 ]; then
            echo "ERROR: $VIOLATIONS prohibited import(s) found in Cloud packages"
            exit 1
          fi

          echo "OK: No prohibited imports in Cloud packages"
```

**Python Implementation:**

```python
# scripts/check_cloud_imports.py
"""
CI Guardrail: Ensure Cloud packages never import trading clients.
"""

import ast
import sys
from pathlib import Path

PROHIBITED_IMPORTS = {
    # Broker trading clients
    'ccxt',
    'alpaca_trade_api',
    'ibapi',
    'oandapyV20',

    # Internal trading modules
    'adapters.binance.order_executor',
    'adapters.alpaca.order_submitter',
    'adapters.oanda.live_client',
    'packages.agent.execution',
}

PROHIBITED_PATTERNS = [
    'BrokerOrderSubmitter',
    'LiveOrderExecutor',
    'OrderExecutor',
    'submit_order',
    'place_order',
]

CLOUD_PATHS = [
    'packages/cloud/',
    'services/control_plane/',
    'services/backtest/',
    'services/training/',
    'services/artifact_builder/',
]


def check_file(filepath: Path) -> list[str]:
    """Check a single Python file for violations."""
    violations = []

    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in PROHIBITED_IMPORTS:
                    violations.append(f"{filepath}:{node.lineno}: import {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if module in PROHIBITED_IMPORTS:
                violations.append(f"{filepath}:{node.lineno}: from {module}")
            for alias in node.names:
                full_name = f"{module}.{alias.name}"
                if full_name in PROHIBITED_IMPORTS:
                    violations.append(f"{filepath}:{node.lineno}: from {full_name}")

        # Check for prohibited patterns in names
        elif isinstance(node, ast.Name):
            if node.id in PROHIBITED_PATTERNS:
                violations.append(f"{filepath}:{node.lineno}: {node.id}")

    return violations


def main():
    violations = []

    for cloud_path in CLOUD_PATHS:
        path = Path(cloud_path)
        if not path.exists():
            continue

        for py_file in path.rglob('*.py'):
            violations.extend(check_file(py_file))

    if violations:
        print("GUARDRAIL VIOLATION: Broker clients found in Cloud packages!")
        print()
        for v in violations:
            print(f"  {v}")
        print()
        print(f"Total violations: {len(violations)}")
        sys.exit(1)

    print("OK: No broker clients in Cloud packages")
    sys.exit(0)


if __name__ == '__main__':
    main()
```

### 1.2 No Order-Like Commands in Protocol

**Rule:** Protocol schema must prohibit fields that look like orders.

**Implementation:**

```python
# scripts/check_protocol_schema.py
"""
CI Guardrail: Ensure protocol schema prohibits order-like payloads.
"""

import json
import sys
from pathlib import Path

PROHIBITED_FIELDS = [
    'side',
    'quantity',
    'qty',
    'price',
    'order_type',
    'target_position',
    'symbol',  # in order context
    'limit_price',
    'stop_price',
    'time_in_force',
]

PROHIBITED_COMMAND_TYPES = [
    'PLACE_ORDER',
    'SUBMIT_ORDER',
    'EXECUTE_SIGNAL',
    'SET_TARGET_POSITION',
    'CANCEL_ORDER',
    'MODIFY_ORDER',
    'FLATTEN_POSITION',
]


def check_schema(schema_path: Path) -> list[str]:
    """Check schema for prohibited elements."""
    violations = []

    with open(schema_path) as f:
        schema = json.load(f)

    # Check command types
    if 'definitions' in schema:
        for name, definition in schema['definitions'].items():
            if 'enum' in definition:
                for value in definition['enum']:
                    if value in PROHIBITED_COMMAND_TYPES:
                        violations.append(f"Prohibited command type: {value}")

    # Check for prohibited fields
    def check_properties(obj, path=''):
        if isinstance(obj, dict):
            if 'properties' in obj:
                for field in obj['properties']:
                    if field.lower() in PROHIBITED_FIELDS:
                        violations.append(f"Prohibited field '{field}' at {path}")
            for key, value in obj.items():
                check_properties(value, f"{path}.{key}")

    check_properties(schema)

    return violations


def main():
    schema_files = [
        'docs/schemas/protocol_messages.schema.json',
        'docs/schemas/artifact_manifest.schema.json',
    ]

    violations = []
    for schema_file in schema_files:
        path = Path(schema_file)
        if path.exists():
            violations.extend(check_schema(path))

    if violations:
        print("GUARDRAIL VIOLATION: Order-like elements in protocol schema!")
        print()
        for v in violations:
            print(f"  {v}")
        sys.exit(1)

    print("OK: No order-like elements in protocol schema")
    sys.exit(0)


if __name__ == '__main__':
    main()
```

### 1.3 Signature Required for Artifacts

**Rule:** Pipeline cannot publish artifacts without signatures.

**Implementation:**

```yaml
# .github/workflows/artifact-build.yml
name: Build Artifact

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build artifact
        run: |
          python scripts/build_artifact.py --output dist/

      - name: Sign artifact (MANDATORY)
        run: |
          # This step MUST succeed for pipeline to continue
          cosign sign-blob \
            --key env://COSIGN_PRIVATE_KEY \
            --output-signature dist/artifact.sig \
            dist/artifact.tar.gz

          # Verify signature was created
          if [ ! -f dist/artifact.sig ]; then
            echo "ERROR: Signature file not created"
            exit 1
          fi

      - name: Verify signature before publish
        run: |
          cosign verify-blob \
            --key env://COSIGN_PUBLIC_KEY \
            --signature dist/artifact.sig \
            dist/artifact.tar.gz

      - name: Publish to registry
        run: |
          # Only runs if signature verification passed
          python scripts/publish_artifact.py --artifact dist/
```

**Agent Verification:**

```python
# packages/agent/security/signature.py
"""
Agent-side signature verification.
NEVER run unsigned artifacts.
"""

import subprocess
from pathlib import Path


def verify_artifact_signature(
    artifact_path: Path,
    signature_path: Path,
    public_key: str
) -> bool:
    """
    Verify artifact signature using cosign.

    CRITICAL: This function MUST be called before running ANY artifact.
    Agent rejects unsigned artifacts unconditionally.
    """
    result = subprocess.run(
        [
            'cosign', 'verify-blob',
            '--key', public_key,
            '--signature', str(signature_path),
            str(artifact_path)
        ],
        capture_output=True
    )

    if result.returncode != 0:
        raise SecurityError(
            f"Artifact signature verification FAILED: {result.stderr.decode()}"
        )

    return True


class SecurityError(Exception):
    """Raised when security check fails. Non-recoverable."""
    pass
```

### 1.4 Telemetry Redaction Mandatory

**Rule:** Agent cannot send telemetry without redaction middleware enabled.

**Implementation:**

```python
# packages/agent/telemetry/buffer.py
"""
Telemetry buffer with mandatory redaction.
"""

from typing import Any
import re


class TelemetryBuffer:
    """
    Buffer for telemetry events before transmission.

    CRITICAL: Redaction is enabled by default and designed with no disable flag.
    The redaction_enabled flag does not exist by design (verify via code review).
    """

    REDACT_PATTERNS = [
        r'api[_-]?key',
        r'api[_-]?secret',
        r'secret[_-]?key',
        r'password',
        r'token',
        r'credential',
        r'private[_-]?key',
        r'auth[_-]?token',
    ]

    def __init__(self):
        # No option to disable redaction
        self._buffer = []

    def add(self, event: dict) -> None:
        """Add event to buffer after mandatory redaction."""
        redacted = self._redact(event)
        self._buffer.append(redacted)

    def _redact(self, data: Any) -> Any:
        """
        Recursively redact sensitive fields.

        This method is ALWAYS called. There is no bypass.
        """
        if isinstance(data, dict):
            return {
                k: self._redact_value(k, v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._redact(item) for item in data]
        return data

    def _redact_value(self, key: str, value: Any) -> Any:
        """Redact value if key matches sensitive pattern."""
        for pattern in self.REDACT_PATTERNS:
            if re.match(pattern, key, re.IGNORECASE):
                return '[REDACTED]'

        if isinstance(value, (dict, list)):
            return self._redact(value)

        return value
```

---

## 2. Dependency Allowlists

### 2.1 Cloud Dependencies

```yaml
# cloud_dependencies_allowlist.yaml
# Only these packages are allowed in Cloud builds

allowed:
  # Web framework
  - fastapi
  - uvicorn
  - starlette

  # Database
  - sqlalchemy
  - asyncpg
  - redis

  # Core
  - pydantic
  - numpy
  - pandas

  # ML (training/backtest only)
  - torch
  - ray

  # Utilities
  - httpx
  - aiofiles

forbidden:
  # Trading clients (NEVER in Cloud)
  - ccxt
  - alpaca-trade-api
  - ibapi
  - oandapyV20
  - python-binance

  # Agent-only packages
  - keyring
  - secretstorage
```

### 2.2 Agent Dependencies

```yaml
# agent_dependencies_allowlist.yaml
# Agent is more permissive but still controlled

allowed:
  # Trading clients (Agent ONLY)
  - ccxt
  - alpaca-trade-api
  - python-binance

  # Secrets management
  - keyring
  - secretstorage
  - cryptography

  # Core
  - pydantic
  - numpy
  - httpx
  - aiofiles

  # Monitoring
  - prometheus-client

forbidden:
  # Cloud-only packages
  - ray  # No distributed training in Agent
  - mlflow  # No experiment tracking in Agent
```

### 2.3 Dependency Check Script

```python
# scripts/check_dependencies.py
"""
CI Guardrail: Verify dependencies against allowlist.
"""

import sys
import yaml
from pathlib import Path


def load_allowlist(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_requirements(path: str) -> set[str]:
    """Extract package names from requirements file."""
    packages = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before ==, >=, etc.)
                pkg = line.split('==')[0].split('>=')[0].split('[')[0]
                packages.add(pkg.lower())
    return packages


def main():
    # Check Cloud
    cloud_allowlist = load_allowlist('cloud_dependencies_allowlist.yaml')
    cloud_reqs = get_requirements('requirements-cloud.txt')

    violations = []

    for pkg in cloud_reqs:
        if pkg in cloud_allowlist.get('forbidden', []):
            violations.append(f"Cloud: Forbidden package '{pkg}'")

    # Check Agent
    agent_allowlist = load_allowlist('agent_dependencies_allowlist.yaml')
    agent_reqs = get_requirements('requirements-agent.txt')

    for pkg in agent_reqs:
        if pkg in agent_allowlist.get('forbidden', []):
            violations.append(f"Agent: Forbidden package '{pkg}'")

    if violations:
        print("GUARDRAIL VIOLATION: Forbidden dependencies!")
        for v in violations:
            print(f"  {v}")
        sys.exit(1)

    print("OK: All dependencies within allowlist")
    sys.exit(0)


if __name__ == '__main__':
    main()
```

---

## 3. Code Review Requirements

### 3.1 Protected Files

Files that require additional review:

```yaml
# .github/CODEOWNERS
# CCEA Security-Critical Paths

# Protocol definitions - require security review
docs/schemas/*.json @security-team
packages/cloud/control_plane/boundary.py @security-team
packages/agent/security/ @security-team

# Telemetry - require privacy review
packages/agent/telemetry/ @privacy-team @security-team

# Broker connectors - require trading review
packages/agent/execution/ @trading-team @security-team
adapters/*/order*.py @trading-team @security-team
```

### 3.2 Review Checklist

```markdown
## Security Review Checklist

For any PR touching CCEA security paths:

### Protocol Changes
- [ ] No new command types that look like orders
- [ ] No new payload fields that could convey trading instructions
- [ ] Schema version updated appropriately
- [ ] Backward compatibility maintained

### Cloud Changes
- [ ] No imports from agent-only packages
- [ ] No direct broker/exchange client usage
- [ ] No credential storage or handling
- [ ] Telemetry properly sanitized

### Agent Changes
- [ ] Signature verification cannot be bypassed
- [ ] Redaction middleware cannot be disabled
- [ ] Kill switch cannot be circumvented
- [ ] Local approval cannot be skipped for TRADING_IMPACTING

### All Changes
- [ ] No secrets in code or config
- [ ] No hardcoded credentials
- [ ] Logging does not leak sensitive data
- [ ] Error messages do not leak sensitive data
```

---

## 4. CI Pipeline Integration

### 4.1 Full Guardrails Workflow

```yaml
# .github/workflows/guardrails.yml
name: CCEA Guardrails

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  guardrails:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install pyyaml

      # Core guardrails - MUST ALL PASS
      - name: "Guardrail: No broker clients in Cloud"
        run: python scripts/check_cloud_imports.py

      - name: "Guardrail: No order-like protocol elements"
        run: python scripts/check_protocol_schema.py

      - name: "Guardrail: Dependencies allowlist"
        run: python scripts/check_dependencies.py

      - name: "Guardrail: No secrets in code"
        run: |
          # Check for common secret patterns
          if grep -rE "(api_key|api_secret|password)\s*=\s*['\"]" \
            --include="*.py" --exclude-dir=.venv .; then
            echo "ERROR: Potential hardcoded secrets found"
            exit 1
          fi
          echo "OK: No hardcoded secrets found"

      - name: "Guardrail: Schema prohibited fields"
        run: |
          # Verify schema has "not" constraint for order fields
          python -c "
          import json
          schema = json.load(open('docs/schemas/protocol_messages.schema.json'))
          assert 'not' in str(schema), 'Schema must have NOT constraint for prohibited fields'
          print('OK: Schema has prohibited field constraints')
          "

  # Only runs if all guardrails pass
  tests:
    needs: guardrails
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest tests/
```

### 4.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: no-broker-in-cloud
        name: Check no broker clients in Cloud
        entry: python scripts/check_cloud_imports.py
        language: python
        pass_filenames: false
        files: ^packages/cloud/

      - id: no-secrets
        name: Check no hardcoded secrets
        entry: python scripts/check_secrets.py
        language: python
        types: [python]

      - id: check-protocol-schema
        name: Check protocol schema
        entry: python scripts/check_protocol_schema.py
        language: python
        pass_filenames: false
        files: ^docs/schemas/
```

---

## 5. Guardrail Summary

| Guardrail | Purpose | Enforcement Level |
|-----------|---------|-------------------|
| No broker in Cloud | Prevent Cloud from executing trades | CI block |
| No order commands | Prevent order-like protocol messages | CI block |
| Signature required | Ensure artifact integrity | CI block + Agent reject |
| Redaction mandatory | Prevent credential leakage | Code design (no flag) |
| Dependency allowlist | Control package usage | CI block |
| Protected files | Require security review | GitHub CODEOWNERS |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | CCEA Team | Initial CI guardrails per Design Doc |

---

**Related Documentation:**

- [CCEA Overview](./CCEA_OVERVIEW.md)
- [Protocol](./CCEA_PROTOCOL.md)
- [Security](../cloud/RESEARCH_JOB_ISOLATION.md)
