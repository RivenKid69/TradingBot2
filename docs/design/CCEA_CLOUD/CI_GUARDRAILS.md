# CCEA CI Guardrails

> **Version**: 2.1.0
> **Date**: 2025-12-22
> **Status**: APPROVED | **Core Guardrails Implemented** (PM-005 coverage gate: TRACKED, not enforced)

Этот документ определяет CI/CD guardrails для обеспечения архитектурной целостности CCEA.

## 1. Категории проверок

### 1.1 Build-time Guardrails (блокируют сборку)

| ID | Check | Scope | Failure Action |
|----|-------|-------|----------------|
| BT-001 | `no-trading-libs-in-cloud` | Cloud build | Block build |
| BT-002 | `no-order-payloads-in-schema` | JSON schemas | Block merge |
| BT-003 | `artifact-signature-required` | Artifact publish | Block publish |
| BT-004 | `sbom-generation` | All builds | Block publish |
| BT-005 | `import-boundary-check` | All packages | Block build |
| BT-006 | `dependency-allowlist` | Cloud/Agent | Block build |

### 1.2 Pre-merge Guardrails (блокируют merge)

| ID | Check | Scope | Failure Action |
|----|-------|-------|----------------|
| PM-001 | `schema-validation` | Schema changes | Block merge |
| PM-002 | `protocol-allowlist` | Protocol changes | Block merge + Security review |
| PM-003 | `redaction-test` | Telemetry code | Block merge |
| PM-004 | `secret-scan` | All files | Block merge |
| PM-005 | `test-coverage` | All code | **TARGET**: Block if < 80% (see note) |

> **PM-005 Implementation Note**: Test coverage is now tracked in CI with artifact upload (`coverage.xml`, `coverage-report.json`, `htmlcov/`). The 80% threshold is a target goal; enforcement as a merge-blocking gate is planned when baseline coverage stabilizes above 70%. Coverage metrics are generated via `pytest --cov` in `.github/workflows/build-and-test.yml` and available as downloadable CI artifacts.
>
> **Control Artifacts**: `coverage.xml` (Cobertura format), `coverage-report.json` (summary with timestamp)
> **Tech Debt Tracking**: `docs/reports/TECH_DEBT_REGISTRY.md#docs-ci-coverage-gate`
> **Status**: Coverage TRACKED (artifact generated); threshold enforcement is TARGET per Documentation Canon

### 1.3 Runtime Guardrails (Agent)

| ID | Check | Scope | Failure Action |
|----|-------|-------|----------------|
| RT-001 | `signature-verification` | Artifact pull | Reject artifact |
| RT-002 | `schema-version-compat` | Command receive | Reject command |
| RT-003 | `approval-enforcement` | Trading-impacting | Queue for approval |
| RT-004 | `hard-cap-enforcement` | Order creation | Reject/limit order |
| RT-005 | `redaction-middleware` | Telemetry send | Block if disabled |

## 2. Детальные спецификации проверок

### BT-001: no-trading-libs-in-cloud

**Цель:** Cloud build не содержит order execution библиотек.

**Реализация:**

```python
# ccea/guardrails/import_check.py

PROHIBITED_IN_CLOUD = [
    "adapters.*.order_execution",
    "adapters.*.options_execution",
    "execution_providers",  # in live mode
    "service_signal_runner",  # in live mode
]

PROHIBITED_PACKAGES = [
    # Third-party trading libs
    "ccxt",  # если используется для order submission
    "alpaca-trade-api",  # submission endpoints
    "ib_insync",  # order submission
]

def check_cloud_build(build_dir: str) -> list[str]:
    """
    Scan cloud build for prohibited imports.
    Returns list of violations.
    """
    violations = []
    for py_file in glob.glob(f"{build_dir}/**/*.py", recursive=True):
        with open(py_file) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = get_module_name(node)
                if matches_prohibited(module, PROHIBITED_IN_CLOUD):
                    violations.append(f"{py_file}: prohibited import {module}")
    return violations
```

**CI Integration:**

```yaml
# .github/workflows/ci.yml
jobs:
  cloud-build-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check cloud imports
        run: python -m ccea.guardrails.import_check --target cloud
        env:
          FAIL_ON_VIOLATION: "true"
```

### BT-002: no-order-payloads-in-schema

**Цель:** JSON schemas не содержат order-like полей.

**Реализация:**

```python
# ccea/guardrails/schema_check.py

PROHIBITED_FIELDS = [
    "side",
    "quantity",
    "qty",
    "price",
    "order_type",
    "target_position",
    "execute_order",
    "place_order",
    "submit_order",
    "intent",
    "signal",
]

PROHIBITED_VALUES = {
    "side": ["BUY", "SELL", "buy", "sell"],
    "order_type": ["MARKET", "LIMIT", "market", "limit"],
}

def validate_schema(schema_path: str) -> list[str]:
    """
    Validate schema doesn't contain order-like payloads.
    """
    violations = []
    with open(schema_path) as f:
        schema = json.load(f)

    # Deep scan for prohibited fields
    def scan_properties(obj, path=""):
        if isinstance(obj, dict):
            if "properties" in obj:
                for field, spec in obj["properties"].items():
                    if field in PROHIBITED_FIELDS:
                        # Check if it's in allowed context
                        if not is_in_prohibited_context(path):
                            continue
                        violations.append(
                            f"{path}.{field}: prohibited order-like field"
                        )
                    scan_properties(spec, f"{path}.{field}")
            for key, value in obj.items():
                scan_properties(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                scan_properties(item, f"{path}[{i}]")

    scan_properties(schema)
    return violations
```

### BT-003: artifact-signature-required

**Цель:** Артефакты публикуются только с подписью.

**Реализация:**

```python
# ccea/guardrails/artifact_check.py

def verify_artifact_signature(artifact_path: str, manifest_path: str) -> bool:
    """
    Verify artifact has valid signature.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    if "signature" not in manifest:
        raise ValueError("Artifact manifest missing signature")

    signature = manifest["signature"]

    if signature["algorithm"] == "sigstore":
        return verify_sigstore(artifact_path, signature)
    elif signature["algorithm"] == "gpg":
        return verify_gpg(artifact_path, signature)
    else:
        raise ValueError(f"Unknown signature algorithm: {signature['algorithm']}")

# CI hook
def pre_publish_check(artifact_path: str) -> None:
    """Block publish if signature missing or invalid."""
    manifest_path = f"{artifact_path}/manifest.json"
    if not verify_artifact_signature(artifact_path, manifest_path):
        raise RuntimeError("Artifact signature verification failed")
```

### BT-005: import-boundary-check

**Цель:** Enforce architectural boundaries между слоями.

**Реализация:**

```ini
# importlinter.ini (расширение для CCEA)

[importlinter]
root_package = .
include_external_packages = True

[importlinter:contract:ccea-cloud-agent-boundary]
name = Cloud cannot import Agent modules
type = forbidden
source_modules =
    packages.cloud
forbidden_modules =
    packages.agent
    adapters.*.order_execution
    adapters.*.options_execution

[importlinter:contract:ccea-agent-no-cloud]
name = Agent should not depend on Cloud internals
type = forbidden
source_modules =
    packages.agent
forbidden_modules =
    packages.cloud.internal

[importlinter:contract:ccea-shared-independence]
name = Shared modules cannot import Cloud or Agent
type = forbidden
source_modules =
    packages.shared
    core_*
    impl_*
forbidden_modules =
    packages.cloud
    packages.agent
```

### PM-002: protocol-allowlist

**Цель:** Новые типы команд требуют security review.

**Реализация:**

```python
# ccea/guardrails/protocol_check.py

ALLOWED_COMMAND_TYPES = frozenset([
    "REQUEST_START_RUN",
    "REQUEST_STOP_RUN",
    "REQUEST_PAUSE_RUN",
    "REQUEST_UPGRADE_ARTIFACT",
    "REQUEST_UPDATE_CONFIG",
    "REQUEST_ROTATE_AGENT_SESSION",
    "REQUEST_EXPORT_LOGS",
])

ALLOWED_MESSAGE_TYPES = frozenset([
    "HEARTBEAT",
    "POLL_COMMANDS",
    "COMMAND_BATCH",
    "COMMAND_ACK",
    "COMMAND_APPROVAL",
    "COMMAND_RESULT",
    "TELEMETRY",
])

def check_protocol_changes(base_schema: dict, new_schema: dict) -> dict:
    """
    Check for new command/message types.
    Returns dict with changes requiring review.
    """
    changes = {
        "new_commands": [],
        "new_messages": [],
        "requires_security_review": False,
    }

    base_commands = extract_command_types(base_schema)
    new_commands = extract_command_types(new_schema)

    added_commands = new_commands - base_commands
    for cmd in added_commands:
        if cmd not in ALLOWED_COMMAND_TYPES:
            changes["new_commands"].append(cmd)
            changes["requires_security_review"] = True

    return changes
```

### PM-004: secret-scan

**Цель:** Предотвращение коммита секретов.

**Реализация:**

```yaml
# .github/workflows/secret-scan.yml
jobs:
  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: TruffleHog scan
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          extra_args: --only-verified
      - name: Gitleaks scan
        uses: gitleaks/gitleaks-action@v2
```

**Patterns:**

```toml
# .gitleaks.toml
[extend]
useDefault = true

[[rules]]
description = "Broker API Key"
regex = '''(?i)(binance|alpaca|oanda|ib).*['\"]?[a-zA-Z0-9]{20,}['\"]?'''
tags = ["broker", "api", "key"]

[[rules]]
description = "Private Key"
regex = '''-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'''
tags = ["private", "key"]
```

### RT-001: signature-verification

**Цель:** Agent проверяет подпись артефакта.

**Реализация:**

```python
# packages/agent/artifact_verifier.py

class ArtifactVerifier:
    def __init__(self, trust_roots: list[str]):
        self.trust_roots = trust_roots

    def verify(self, artifact_path: str, manifest: dict) -> VerificationResult:
        """
        Verify artifact signature against trust roots.

        MUST pass before artifact can be loaded.
        """
        # 1. Verify digest
        computed_digest = compute_digest(artifact_path)
        if computed_digest != manifest.get("artifact_digest"):
            return VerificationResult(
                success=False,
                error="Digest mismatch"
            )

        # 2. Verify signature
        signature = manifest.get("signature")
        if not signature:
            return VerificationResult(
                success=False,
                error="Missing signature"
            )

        # 3. Verify against trust root
        if not self._verify_signature(artifact_path, signature):
            return VerificationResult(
                success=False,
                error="Signature verification failed"
            )

        # 4. Verify schema version compatibility
        schema_version = manifest.get("schema_version")
        if not self._is_compatible_version(schema_version):
            return VerificationResult(
                success=False,
                error=f"Incompatible schema version: {schema_version}"
            )

        return VerificationResult(success=True)
```

### RT-005: redaction-middleware

**Цель:** Телеметрия всегда проходит redaction.

**Реализация:**

```python
# packages/agent/telemetry/redaction.py

class RedactionMiddleware:
    """
    MANDATORY middleware for all telemetry.
    Cannot be disabled.
    """

    PATTERNS = [
        (r'[A-Za-z0-9]{20,}', '[REDACTED_KEY]'),  # API keys
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[REDACTED_IP]'),  # IPs
        (r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[REDACTED_EMAIL]'),
    ]

    SENSITIVE_FIELDS = [
        'api_key', 'secret', 'password', 'token',
        'private_key', 'credential', 'auth',
    ]

    def __init__(self):
        self._enabled = True  # Cannot be disabled

    @property
    def enabled(self) -> bool:
        return True  # Always True, setter is no-op

    @enabled.setter
    def enabled(self, value: bool) -> None:
        # Intentionally ignore - redaction cannot be disabled
        pass

    def redact(self, data: dict) -> dict:
        """
        Apply redaction to telemetry data.
        """
        return self._deep_redact(data)

    def _deep_redact(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: self._redact_value(k, v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._deep_redact(item) for item in obj]
        elif isinstance(obj, str):
            return self._redact_string(obj)
        return obj

    def _redact_value(self, key: str, value: Any) -> Any:
        # Full redaction for sensitive fields
        if any(s in key.lower() for s in self.SENSITIVE_FIELDS):
            return '[REDACTED]'
        return self._deep_redact(value)

    def _redact_string(self, s: str) -> str:
        for pattern, replacement in self.PATTERNS:
            s = re.sub(pattern, replacement, s)
        return s
```

## 3. CI Pipeline Integration

### 3.1 GitHub Actions Workflow

> **DOCS/DRIFT Note (CCEA-DOC-002):** The workflow below is a *recommended* configuration.
> Current CI uses `security-sast.yml` and `build-and-test.yml` for checks.
> Dedicated guardrails workflow to be created per this specification.

```yaml
# .github/workflows/ccea-guardrails.yml (RECOMMENDED - to be created)
name: CCEA Guardrails

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  import-boundary-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install import-linter
      - name: Check import boundaries
        run: lint-imports

  schema-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate schemas
        run: python -m ccea.guardrails.schema_check docs/schemas/

  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Run Gitleaks
        uses: gitleaks/gitleaks-action@v2

  cloud-build-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check cloud imports
        run: python -m ccea.guardrails.import_check --target cloud

  protocol-check:
    runs-on: ubuntu-latest
    if: contains(github.event.pull_request.labels.*.name, 'protocol-change')
    steps:
      - uses: actions/checkout@v4
      - name: Check protocol changes
        run: python -m ccea.guardrails.protocol_check
      - name: Request security review
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.addLabels({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              labels: ['security-review-required']
            })

  test-coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests with coverage
        run: pytest --cov=ccea --cov-report=xml
      - name: Check coverage threshold
        run: |
          coverage=$(python -c "import xml.etree.ElementTree as ET; print(ET.parse('coverage.xml').getroot().attrib['line-rate'])")
          if (( $(echo "$coverage < 0.80" | bc -l) )); then
            echo "Coverage $coverage is below 80%"
            exit 1
          fi
```

### 3.2 Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ccea-import-check
        name: CCEA Import Boundary Check
        entry: python -m ccea.guardrails.import_check
        language: system
        types: [python]
        pass_filenames: false

      - id: ccea-schema-check
        name: CCEA Schema Validation
        entry: python -m ccea.guardrails.schema_check
        language: system
        files: \.schema\.json$

      - id: secret-scan
        name: Secret Scan
        entry: gitleaks protect --staged
        language: system
        pass_filenames: false
```

## 4. Enforcement Matrix

| Guardrail | Pre-commit | CI (PR) | CI (merge) | Runtime |
|-----------|------------|---------|------------|---------|
| BT-001 no-trading-libs | Yes | Yes | Yes | - |
| BT-002 no-order-payloads | Yes | Yes | Yes | - |
| BT-003 artifact-signature | - | Yes | Yes | Yes |
| BT-005 import-boundary | Yes | Yes | Yes | - |
| PM-001 schema-validation | Yes | Yes | Yes | - |
| PM-002 protocol-allowlist | - | Yes | Yes | - |
| PM-004 secret-scan | Yes | Yes | Yes | - |
| RT-001 signature-verify | - | - | - | Yes |
| RT-005 redaction | - | - | - | Yes |

## 5. Bypass Process

### 5.1 Emergency Bypass

В экстренных случаях возможен bypass с:

1. Approval от 2+ Senior Engineers
2. Security team sign-off
3. Documented justification в issue
4. Post-incident review

### 5.2 Bypass Log

Все bypass логируются:

```json
{
  "timestamp": "2025-12-13T10:00:00Z",
  "guardrail": "BT-001",
  "approvers": ["engineer1", "engineer2"],
  "security_signoff": "security_lead",
  "justification": "Emergency hotfix for...",
  "issue_ref": "#1234",
  "expires_at": "2025-12-14T10:00:00Z"
}
```

---

**Document Control:**

- Author: CCEA Platform Team
- Reviewers: Security, DevOps
- Approval: Engineering Lead
