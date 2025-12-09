# Software Provider Compliance Implementation Plan

**Target:** SaaS platform where users develop strategies and trade via their own broker API keys.
**Regulatory Position:** Software vendor (not investment firm, not broker).

---

## Phase 1: Legal & Security Foundation (Critical)

### 1.1 Terms of Service

**Location:** `docs/legal/TERMS_OF_SERVICE.md` + integration in app

**Required Sections (per EU E-Commerce Directive 2000/31/EC):**

```
1. DEFINITIONS
   - "Platform", "User", "Strategy", "Broker API Keys"

2. SERVICE DESCRIPTION
   - Software tool for strategy development, backtesting, execution
   - NOT investment advice, NOT portfolio management
   - User controls all trading decisions

3. USER RESPONSIBILITIES
   - Provide own broker account and API keys
   - Ensure compliance with broker's terms
   - Understand trading risks
   - Age 18+ (or legal age in jurisdiction)

4. BROKER API KEYS
   - User grants platform permission to execute orders via their keys
   - Keys encrypted at rest (AES-256)
   - User can revoke access anytime
   - Platform never has withdrawal rights

5. NO INVESTMENT ADVICE (MiFID II Article 4(1)(4) exclusion)
   - Platform provides tools, not recommendations
   - Backtests are simulations, not predictions
   - User solely responsible for strategy decisions

6. LIMITATION OF LIABILITY
   - Direct damages capped at fees paid (12 months)
   - No liability for: market losses, broker failures, strategy performance
   - Force majeure clause

7. DISCLAIMERS
   - "Past performance does not guarantee future results"
   - "Trading involves substantial risk of loss"
   - "This is software, not financial advice"

8. DATA & PRIVACY
   - Reference to Privacy Policy
   - GDPR rights summary

9. TERMINATION
   - User can terminate anytime
   - Data deletion on request (GDPR Art. 17)

10. GOVERNING LAW
    - EU jurisdiction (recommend: Netherlands/Ireland)
    - Dispute resolution mechanism
```

**Sources:**
- EU E-Commerce Directive 2000/31/EC Art. 5, 6
- ESMA Q&A MiFID II (ESMA35-43-349) - software vendor exclusion
- GDPR Art. 13-14 transparency requirements

**Tests:**
```python
# tests/test_legal_compliance.py
class TestTermsOfService:
    def test_tos_file_exists(self):
        assert Path("docs/legal/TERMS_OF_SERVICE.md").exists()

    def test_tos_contains_required_sections(self):
        content = Path("docs/legal/TERMS_OF_SERVICE.md").read_text()
        required = [
            "NO INVESTMENT ADVICE",
            "LIMITATION OF LIABILITY",
            "BROKER API KEYS",
            "past performance",
            "substantial risk"
        ]
        for section in required:
            assert section.lower() in content.lower()

    def test_tos_acceptance_endpoint_exists(self):
        # API must have ToS acceptance tracking
        pass
```

---

### 1.2 Privacy Policy

**Location:** `docs/legal/PRIVACY_POLICY.md`

**Required Sections (GDPR Art. 13-14):**

```
1. DATA CONTROLLER
   - Company name, address, contact
   - DPO contact (if applicable)

2. DATA COLLECTED
   - Account: email, name, password hash
   - Trading: strategies, backtests, execution logs
   - Technical: IP, device, usage analytics
   - Broker keys: encrypted, purpose-limited

3. LEGAL BASIS (GDPR Art. 6)
   - Contract performance (Art. 6(1)(b)): core service
   - Legitimate interest (Art. 6(1)(f)): security, fraud prevention
   - Consent (Art. 6(1)(a)): marketing (optional)

4. DATA RETENTION
   - Account data: until deletion request + 30 days
   - Trading logs: 5 years (audit trail)
   - Broker keys: until user revokes

5. DATA SHARING
   - Brokers: only API calls with user's keys
   - No selling data to third parties
   - Sub-processors list

6. INTERNATIONAL TRANSFERS
   - EU-only storage (AWS eu-central-1)
   - If non-EU: SCCs or adequacy decision

7. USER RIGHTS (GDPR Art. 15-22)
   - Access (Art. 15)
   - Rectification (Art. 16)
   - Erasure (Art. 17)
   - Portability (Art. 20)
   - Object (Art. 21)
   - Complaint to supervisory authority

8. COOKIES
   - Essential only (no consent needed)
   - Analytics (with consent)

9. SECURITY MEASURES
   - Encryption at rest and in transit
   - Access controls
   - Regular audits

10. CHANGES
    - Notification of material changes
    - Version history
```

**Sources:**
- GDPR Regulation (EU) 2016/679
- EDPB Guidelines on Transparency (WP260)
- ICO Guide to Privacy Notices

**Tests:**
```python
class TestPrivacyPolicy:
    def test_privacy_policy_exists(self):
        assert Path("docs/legal/PRIVACY_POLICY.md").exists()

    def test_gdpr_articles_referenced(self):
        content = Path("docs/legal/PRIVACY_POLICY.md").read_text()
        assert "Article 15" in content  # Access
        assert "Article 17" in content  # Erasure
        assert "Article 20" in content  # Portability

    def test_data_controller_specified(self):
        content = Path("docs/legal/PRIVACY_POLICY.md").read_text()
        assert "data controller" in content.lower()
```

---

### 1.3 API Key Encryption at Rest

**Location:** `services/security/credential_vault.py`

**Implementation (NIST SP 800-57, OWASP):**

```python
# services/security/credential_vault.py
"""
Broker API Key Vault with AES-256-GCM encryption.

References:
- NIST SP 800-57 Part 1 Rev 5: Key Management
- OWASP Cryptographic Storage Cheat Sheet
- PCI DSS Requirement 3.4: Render PAN unreadable
"""
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import base64
import hashlib
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from enum import Enum

class CredentialType(Enum):
    BROKER_API_KEY = "broker_api_key"
    BROKER_API_SECRET = "broker_api_secret"
    BROKER_PASSPHRASE = "broker_passphrase"

@dataclass
class EncryptedCredential:
    credential_id: str
    user_id: str
    credential_type: CredentialType
    broker: str
    ciphertext: bytes
    nonce: bytes
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0

class CredentialVault:
    """
    Secure storage for broker API credentials.

    Security features:
    - AES-256-GCM encryption (authenticated encryption)
    - Unique nonce per encryption
    - Key derivation from master key + user_id (isolation)
    - Access logging for audit trail
    """

    def __init__(self, master_key: bytes):
        """
        Args:
            master_key: 32-byte master encryption key
                        Source: environment variable or AWS KMS
        """
        if len(master_key) != 32:
            raise ValueError("Master key must be 32 bytes (256 bits)")
        self._master_key = master_key
        self._access_log: list = []

    def _derive_user_key(self, user_id: str) -> bytes:
        """Derive user-specific key from master key."""
        kdf = PBKDF2HMAC(
            algorithm=hashlib.sha256(),
            length=32,
            salt=user_id.encode(),
            iterations=100_000,
        )
        return kdf.derive(self._master_key)

    def encrypt(
        self,
        user_id: str,
        credential_type: CredentialType,
        broker: str,
        plaintext: str
    ) -> EncryptedCredential:
        """Encrypt a broker credential."""
        user_key = self._derive_user_key(user_id)
        aesgcm = AESGCM(user_key)
        nonce = os.urandom(12)  # 96-bit nonce for GCM

        ciphertext = aesgcm.encrypt(
            nonce,
            plaintext.encode(),
            associated_data=f"{user_id}:{broker}:{credential_type.value}".encode()
        )

        credential_id = hashlib.sha256(
            f"{user_id}:{broker}:{credential_type.value}".encode()
        ).hexdigest()[:16]

        return EncryptedCredential(
            credential_id=credential_id,
            user_id=user_id,
            credential_type=credential_type,
            broker=broker,
            ciphertext=ciphertext,
            nonce=nonce,
            created_at=datetime.utcnow()
        )

    def decrypt(
        self,
        credential: EncryptedCredential,
        purpose: str
    ) -> str:
        """
        Decrypt a broker credential.

        Args:
            credential: Encrypted credential object
            purpose: Reason for access (logged)
        """
        user_key = self._derive_user_key(credential.user_id)
        aesgcm = AESGCM(user_key)

        plaintext = aesgcm.decrypt(
            credential.nonce,
            credential.ciphertext,
            associated_data=f"{credential.user_id}:{credential.broker}:{credential.credential_type.value}".encode()
        )

        # Log access
        self._log_access(credential, purpose)

        return plaintext.decode()

    def _log_access(self, credential: EncryptedCredential, purpose: str):
        """Log credential access for audit trail."""
        self._access_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "credential_id": credential.credential_id,
            "user_id": credential.user_id,
            "broker": credential.broker,
            "credential_type": credential.credential_type.value,
            "purpose": purpose
        })
        credential.last_accessed = datetime.utcnow()
        credential.access_count += 1

    def delete(self, credential: EncryptedCredential) -> bool:
        """Securely delete credential (GDPR Art. 17)."""
        # Overwrite with random data before deletion
        credential.ciphertext = os.urandom(len(credential.ciphertext))
        credential.nonce = os.urandom(12)
        self._log_access(credential, "DELETION")
        return True

    def get_access_log(self, user_id: Optional[str] = None) -> list:
        """Get access log, optionally filtered by user."""
        if user_id:
            return [e for e in self._access_log if e["user_id"] == user_id]
        return self._access_log
```

**Sources:**
- NIST SP 800-57 Part 1 Rev 5
- OWASP Cryptographic Storage Cheat Sheet
- RFC 5116 (AEAD)

**Tests:**
```python
# tests/test_credential_vault.py
import pytest
import os

class TestCredentialVault:
    @pytest.fixture
    def vault(self):
        master_key = os.urandom(32)
        return CredentialVault(master_key)

    def test_encrypt_decrypt_roundtrip(self, vault):
        plaintext = "sk-live-abc123xyz"
        encrypted = vault.encrypt(
            user_id="user_001",
            credential_type=CredentialType.BROKER_API_KEY,
            broker="alpaca",
            plaintext=plaintext
        )
        decrypted = vault.decrypt(encrypted, purpose="test")
        assert decrypted == plaintext

    def test_different_users_different_ciphertext(self, vault):
        plaintext = "same_key"
        enc1 = vault.encrypt("user_1", CredentialType.BROKER_API_KEY, "alpaca", plaintext)
        enc2 = vault.encrypt("user_2", CredentialType.BROKER_API_KEY, "alpaca", plaintext)
        assert enc1.ciphertext != enc2.ciphertext

    def test_access_logging(self, vault):
        encrypted = vault.encrypt("user_001", CredentialType.BROKER_API_KEY, "binance", "key")
        vault.decrypt(encrypted, purpose="order_execution")
        log = vault.get_access_log("user_001")
        assert len(log) == 1
        assert log[0]["purpose"] == "order_execution"

    def test_tampered_ciphertext_fails(self, vault):
        encrypted = vault.encrypt("user_001", CredentialType.BROKER_API_KEY, "alpaca", "key")
        encrypted.ciphertext = b"tampered" + encrypted.ciphertext[8:]
        with pytest.raises(Exception):  # InvalidTag
            vault.decrypt(encrypted, purpose="test")

    def test_secure_deletion(self, vault):
        encrypted = vault.encrypt("user_001", CredentialType.BROKER_API_KEY, "alpaca", "key")
        original_ciphertext = encrypted.ciphertext
        vault.delete(encrypted)
        assert encrypted.ciphertext != original_ciphertext
```

---

### 1.4 UI Disclaimer (Pre-Live Trading)

**Location:** Frontend component + API endpoint

**Implementation:**

```python
# services/api/disclaimer_service.py
"""
User acknowledgment tracking for legal disclaimers.

References:
- MiFID II Article 24(4): Fair, clear, not misleading
- ESMA Guidelines on Product Governance
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import hashlib

class DisclaimerType(Enum):
    PRE_LIVE_TRADING = "pre_live_trading"
    BACKTEST_RESULTS = "backtest_results"
    TERMS_OF_SERVICE = "terms_of_service"
    PRIVACY_POLICY = "privacy_policy"

@dataclass
class DisclaimerAcknowledgment:
    user_id: str
    disclaimer_type: DisclaimerType
    disclaimer_version: str
    acknowledged_at: datetime
    ip_address: str
    user_agent: str

PRE_LIVE_TRADING_DISCLAIMER = """
IMPORTANT: READ BEFORE ENABLING LIVE TRADING

1. RISK WARNING
   Trading financial instruments involves substantial risk of loss.
   You may lose more than your initial investment.

2. NO INVESTMENT ADVICE
   This platform is a software tool. It does NOT provide investment advice,
   recommendations, or portfolio management services.

3. YOUR RESPONSIBILITY
   - You are solely responsible for your trading decisions
   - You are using YOUR OWN broker account and API keys
   - You must ensure your strategies comply with applicable regulations

4. PAST PERFORMANCE
   Backtest results are simulations based on historical data.
   Past performance does NOT guarantee future results.

5. TECHNICAL RISKS
   - Software may contain bugs
   - Network failures may affect order execution
   - Market conditions may differ from simulations

By clicking "I Understand and Accept", you confirm that:
- You have read and understood the above warnings
- You are 18 years or older
- You accept full responsibility for your trading activities
"""

class DisclaimerService:
    def __init__(self, storage):
        self._storage = storage
        self._current_versions = {
            DisclaimerType.PRE_LIVE_TRADING: "1.0.0",
            DisclaimerType.BACKTEST_RESULTS: "1.0.0",
        }

    def get_disclaimer_text(self, disclaimer_type: DisclaimerType) -> str:
        if disclaimer_type == DisclaimerType.PRE_LIVE_TRADING:
            return PRE_LIVE_TRADING_DISCLAIMER
        # ... other types

    def record_acknowledgment(
        self,
        user_id: str,
        disclaimer_type: DisclaimerType,
        ip_address: str,
        user_agent: str
    ) -> DisclaimerAcknowledgment:
        """Record user's disclaimer acknowledgment."""
        ack = DisclaimerAcknowledgment(
            user_id=user_id,
            disclaimer_type=disclaimer_type,
            disclaimer_version=self._current_versions[disclaimer_type],
            acknowledged_at=datetime.utcnow(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        self._storage.save(ack)
        return ack

    def has_valid_acknowledgment(
        self,
        user_id: str,
        disclaimer_type: DisclaimerType
    ) -> bool:
        """Check if user has acknowledged current version."""
        latest = self._storage.get_latest(user_id, disclaimer_type)
        if not latest:
            return False
        return latest.disclaimer_version == self._current_versions[disclaimer_type]

    def require_acknowledgment(
        self,
        user_id: str,
        disclaimer_type: DisclaimerType
    ):
        """Raise if user hasn't acknowledged disclaimer."""
        if not self.has_valid_acknowledgment(user_id, disclaimer_type):
            raise DisclaimerNotAcknowledgedError(
                f"User must acknowledge {disclaimer_type.value} before proceeding"
            )

class DisclaimerNotAcknowledgedError(Exception):
    pass
```

**Integration point:**
```python
# In order execution flow
def execute_live_order(user_id: str, order: Order):
    disclaimer_service.require_acknowledgment(
        user_id,
        DisclaimerType.PRE_LIVE_TRADING
    )
    # ... proceed with execution
```

**Tests:**
```python
class TestDisclaimerService:
    def test_fresh_user_has_no_acknowledgment(self, service):
        assert not service.has_valid_acknowledgment("new_user", DisclaimerType.PRE_LIVE_TRADING)

    def test_acknowledgment_recorded(self, service):
        service.record_acknowledgment(
            "user_001",
            DisclaimerType.PRE_LIVE_TRADING,
            "127.0.0.1",
            "Mozilla/5.0"
        )
        assert service.has_valid_acknowledgment("user_001", DisclaimerType.PRE_LIVE_TRADING)

    def test_version_change_requires_reacknowledgment(self, service):
        service.record_acknowledgment("user_001", DisclaimerType.PRE_LIVE_TRADING, "", "")
        service._current_versions[DisclaimerType.PRE_LIVE_TRADING] = "2.0.0"
        assert not service.has_valid_acknowledgment("user_001", DisclaimerType.PRE_LIVE_TRADING)

    def test_require_acknowledgment_raises(self, service):
        with pytest.raises(DisclaimerNotAcknowledgedError):
            service.require_acknowledgment("new_user", DisclaimerType.PRE_LIVE_TRADING)
```

---

### 1.5 Backtest Disclaimers

**Location:** `services/backtest/result_formatter.py`

**Implementation:**
```python
# services/backtest/disclaimer_injection.py
"""
Automatic disclaimer injection into backtest results.

References:
- SEC Rule 206(4)-1: Performance advertising
- FCA COBS 4.6: Past performance
- ESMA Guidelines on Marketing Communications
"""

BACKTEST_DISCLAIMER = {
    "warning": "SIMULATION ONLY - NOT ACTUAL TRADING RESULTS",
    "legal": (
        "IMPORTANT: These results are based on historical simulation and do NOT "
        "represent actual trading. Past performance does NOT guarantee future results. "
        "Simulated results have inherent limitations: (1) they are prepared with benefit "
        "of hindsight, (2) they do not reflect actual slippage, fees, or market impact, "
        "(3) they assume perfect execution which rarely occurs in live trading. "
        "Trading involves substantial risk of loss."
    ),
    "version": "1.0.0"
}

def inject_disclaimer(backtest_result: dict) -> dict:
    """Add mandatory disclaimer to backtest results."""
    return {
        "disclaimer": BACKTEST_DISCLAIMER,
        "results": backtest_result,
        "generated_at": datetime.utcnow().isoformat(),
        "is_simulation": True,
        "is_investment_advice": False
    }

# Integration in backtest service
class BacktestService:
    def run_backtest(self, strategy, data) -> dict:
        raw_results = self._execute_backtest(strategy, data)
        return inject_disclaimer(raw_results)
```

**Tests:**
```python
class TestBacktestDisclaimer:
    def test_disclaimer_always_present(self, backtest_service):
        result = backtest_service.run_backtest(mock_strategy, mock_data)
        assert "disclaimer" in result
        assert "past performance" in result["disclaimer"]["legal"].lower()

    def test_is_simulation_flag(self, backtest_service):
        result = backtest_service.run_backtest(mock_strategy, mock_data)
        assert result["is_simulation"] is True
        assert result["is_investment_advice"] is False
```

---

## Phase 2: GDPR & Additional Protections

### 2.1 GDPR Delete (Right to Erasure)

**Location:** `services/gdpr/data_deletion.py`

**Implementation (GDPR Art. 17):**

```python
# services/gdpr/data_deletion.py
"""
GDPR Article 17 - Right to Erasure implementation.

References:
- GDPR Regulation (EU) 2016/679 Article 17
- EDPB Guidelines on the Right to Erasure
- ICO Right to Erasure Guide
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class DeletionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"

class DataCategory(Enum):
    ACCOUNT = "account"              # Email, name, password
    STRATEGIES = "strategies"         # User's trading strategies
    BACKTESTS = "backtests"          # Backtest results
    EXECUTION_LOGS = "execution_logs" # Order execution history
    BROKER_CREDENTIALS = "broker_credentials"
    ANALYTICS = "analytics"           # Usage data
    AUDIT_LOGS = "audit_logs"        # Security logs (retention exception)

@dataclass
class DeletionRequest:
    request_id: str
    user_id: str
    requested_at: datetime
    categories: List[DataCategory]
    status: DeletionStatus
    completed_at: Optional[datetime] = None
    retention_exceptions: List[str] = None  # Categories kept for legal reasons

class GDPRDeletionService:
    """
    Handles user data deletion requests per GDPR Art. 17.

    Retention exceptions (GDPR Art. 17(3)):
    - Legal obligation compliance
    - Public interest archiving
    - Legal claims defense
    """

    # Categories exempt from immediate deletion
    RETENTION_EXCEPTIONS = {
        DataCategory.AUDIT_LOGS: {
            "reason": "Legal obligation - financial audit trail",
            "retention_period_years": 5,
            "legal_basis": "GDPR Art. 17(3)(b), MiFID II Art. 25"
        }
    }

    DELETION_DEADLINE_DAYS = 30  # GDPR Art. 12(3)

    def __init__(self, repositories: dict):
        self._repos = repositories

    def create_request(self, user_id: str) -> DeletionRequest:
        """Create deletion request - must complete within 30 days."""
        request = DeletionRequest(
            request_id=self._generate_id(),
            user_id=user_id,
            requested_at=datetime.utcnow(),
            categories=list(DataCategory),
            status=DeletionStatus.PENDING,
            retention_exceptions=[]
        )
        self._repos["deletion_requests"].save(request)
        logger.info(f"Deletion request created: {request.request_id}")
        return request

    def execute_deletion(self, request: DeletionRequest) -> DeletionRequest:
        """Execute deletion across all data stores."""
        request.status = DeletionStatus.IN_PROGRESS

        for category in request.categories:
            if category in self.RETENTION_EXCEPTIONS:
                request.retention_exceptions.append(
                    f"{category.value}: {self.RETENTION_EXCEPTIONS[category]['reason']}"
                )
                # Anonymize instead of delete
                self._anonymize_category(request.user_id, category)
            else:
                self._delete_category(request.user_id, category)

        request.status = DeletionStatus.COMPLETED
        request.completed_at = datetime.utcnow()
        self._repos["deletion_requests"].update(request)

        logger.info(f"Deletion completed: {request.request_id}")
        return request

    def _delete_category(self, user_id: str, category: DataCategory):
        """Delete all data in category for user."""
        repo = self._repos.get(category.value)
        if repo:
            count = repo.delete_by_user(user_id)
            logger.info(f"Deleted {count} records from {category.value}")

    def _anonymize_category(self, user_id: str, category: DataCategory):
        """Anonymize data that must be retained."""
        repo = self._repos.get(category.value)
        if repo:
            repo.anonymize_by_user(user_id)
            logger.info(f"Anonymized {category.value} for user {user_id}")

    def get_deletion_report(self, request_id: str) -> dict:
        """Generate report for user showing what was deleted."""
        request = self._repos["deletion_requests"].get(request_id)
        return {
            "request_id": request.request_id,
            "status": request.status.value,
            "requested_at": request.requested_at.isoformat(),
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "deleted_categories": [
                c.value for c in request.categories
                if c not in self.RETENTION_EXCEPTIONS
            ],
            "retention_exceptions": request.retention_exceptions,
            "compliance": "GDPR Article 17"
        }
```

**Tests:**
```python
class TestGDPRDeletion:
    def test_deletion_request_created(self, service):
        request = service.create_request("user_001")
        assert request.status == DeletionStatus.PENDING
        assert request.user_id == "user_001"

    def test_deletion_completes_within_deadline(self, service):
        request = service.create_request("user_001")
        result = service.execute_deletion(request)
        assert result.status == DeletionStatus.COMPLETED
        delta = result.completed_at - result.requested_at
        assert delta.days <= service.DELETION_DEADLINE_DAYS

    def test_audit_logs_anonymized_not_deleted(self, service, mock_repos):
        request = service.create_request("user_001")
        service.execute_deletion(request)
        # Audit logs should be anonymized, not deleted
        assert mock_repos["audit_logs"].anonymize_by_user.called
        assert not mock_repos["audit_logs"].delete_by_user.called

    def test_broker_credentials_deleted(self, service, mock_repos):
        request = service.create_request("user_001")
        service.execute_deletion(request)
        assert mock_repos["broker_credentials"].delete_by_user.called

    def test_deletion_report_generated(self, service):
        request = service.create_request("user_001")
        service.execute_deletion(request)
        report = service.get_deletion_report(request.request_id)
        assert "deleted_categories" in report
        assert "retention_exceptions" in report
```

---

### 2.2 GDPR Export (Right to Portability)

**Location:** `services/gdpr/data_export.py`

**Implementation (GDPR Art. 20):**

```python
# services/gdpr/data_export.py
"""
GDPR Article 20 - Right to Data Portability.

Format: JSON (machine-readable, commonly used)
References:
- GDPR Art. 20
- EDPB Guidelines on Data Portability (WP242)
"""
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List
import zipfile
import io

@dataclass
class PortableDataPackage:
    user_id: str
    exported_at: datetime
    format_version: str = "1.0"

    account: dict = None
    strategies: List[dict] = None
    backtests: List[dict] = None
    execution_history: List[dict] = None
    settings: dict = None

class GDPRExportService:
    """Export user data in portable format per GDPR Art. 20."""

    EXPORT_DEADLINE_DAYS = 30

    def __init__(self, repositories: dict):
        self._repos = repositories

    def export_user_data(self, user_id: str) -> bytes:
        """
        Export all user data as ZIP containing JSON files.

        Returns: ZIP file as bytes
        """
        package = PortableDataPackage(
            user_id=user_id,
            exported_at=datetime.utcnow()
        )

        # Collect data from all repositories
        package.account = self._export_account(user_id)
        package.strategies = self._export_strategies(user_id)
        package.backtests = self._export_backtests(user_id)
        package.execution_history = self._export_execution_history(user_id)
        package.settings = self._export_settings(user_id)

        return self._create_zip(package)

    def _export_account(self, user_id: str) -> dict:
        user = self._repos["users"].get(user_id)
        return {
            "email": user.email,
            "name": user.name,
            "created_at": user.created_at.isoformat(),
            # Exclude: password hash, internal IDs
        }

    def _export_strategies(self, user_id: str) -> List[dict]:
        strategies = self._repos["strategies"].get_by_user(user_id)
        return [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
                "created_at": s.created_at.isoformat(),
                "code": s.code  # User's own code
            }
            for s in strategies
        ]

    def _export_backtests(self, user_id: str) -> List[dict]:
        backtests = self._repos["backtests"].get_by_user(user_id)
        return [
            {
                "strategy_name": b.strategy_name,
                "start_date": b.start_date.isoformat(),
                "end_date": b.end_date.isoformat(),
                "results": b.results,
                "ran_at": b.ran_at.isoformat()
            }
            for b in backtests
        ]

    def _export_execution_history(self, user_id: str) -> List[dict]:
        executions = self._repos["executions"].get_by_user(user_id)
        return [
            {
                "order_id": e.order_id,
                "symbol": e.symbol,
                "side": e.side,
                "quantity": e.quantity,
                "price": e.price,
                "broker": e.broker,
                "executed_at": e.executed_at.isoformat(),
                "status": e.status
            }
            for e in executions
        ]

    def _export_settings(self, user_id: str) -> dict:
        settings = self._repos["settings"].get(user_id)
        return {
            "default_broker": settings.default_broker,
            "risk_parameters": settings.risk_parameters,
            "notification_preferences": settings.notification_preferences
        }

    def _create_zip(self, package: PortableDataPackage) -> bytes:
        """Create ZIP file with JSON exports."""
        buffer = io.BytesIO()

        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Metadata
            zf.writestr("metadata.json", json.dumps({
                "user_id": package.user_id,
                "exported_at": package.exported_at.isoformat(),
                "format_version": package.format_version,
                "gdpr_article": "Article 20 - Right to Data Portability"
            }, indent=2))

            # Data files
            zf.writestr("account.json", json.dumps(package.account, indent=2))
            zf.writestr("strategies.json", json.dumps(package.strategies, indent=2))
            zf.writestr("backtests.json", json.dumps(package.backtests, indent=2))
            zf.writestr("execution_history.json", json.dumps(package.execution_history, indent=2))
            zf.writestr("settings.json", json.dumps(package.settings, indent=2))

            # README
            zf.writestr("README.txt", self._generate_readme())

        buffer.seek(0)
        return buffer.read()

    def _generate_readme(self) -> str:
        return """
DATA EXPORT - GDPR Article 20 (Right to Data Portability)

This archive contains your personal data in machine-readable format (JSON).

Files included:
- metadata.json: Export information
- account.json: Your account details
- strategies.json: Your trading strategies
- backtests.json: Your backtest results
- execution_history.json: Your order execution history
- settings.json: Your platform settings

Format: JSON (JavaScript Object Notation)
Encoding: UTF-8

For questions, contact: [DPO email]
"""
```

**Tests:**
```python
class TestGDPRExport:
    def test_export_returns_zip(self, service):
        data = service.export_user_data("user_001")
        assert data[:2] == b'PK'  # ZIP magic bytes

    def test_export_contains_required_files(self, service):
        data = service.export_user_data("user_001")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            assert "metadata.json" in names
            assert "account.json" in names
            assert "strategies.json" in names

    def test_export_json_valid(self, service):
        data = service.export_user_data("user_001")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            account = json.loads(zf.read("account.json"))
            assert "email" in account

    def test_export_excludes_sensitive_data(self, service):
        data = service.export_user_data("user_001")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            account = json.loads(zf.read("account.json"))
            assert "password" not in account
            assert "password_hash" not in account
```

---

### 2.3 Geo-Blocking (Sanctioned Countries)

**Location:** `services/security/geo_blocking.py`

**Implementation:**

```python
# services/security/geo_blocking.py
"""
Geographic access restrictions for sanctioned jurisdictions.

References:
- EU Council Regulations (sanctions)
- OFAC Sanctions Programs (US)
- UK Financial Sanctions
"""
from enum import Enum
from typing import Set, Optional
from dataclasses import dataclass
import ipaddress

class BlockReason(Enum):
    OFAC_SANCTIONS = "US OFAC Comprehensive Sanctions"
    EU_SANCTIONS = "EU Council Sanctions"
    UK_SANCTIONS = "UK Financial Sanctions"
    PLATFORM_POLICY = "Platform Policy"

# ISO 3166-1 alpha-2 codes
BLOCKED_COUNTRIES: dict[str, BlockReason] = {
    # Comprehensive US/EU sanctions
    "CU": BlockReason.OFAC_SANCTIONS,  # Cuba
    "IR": BlockReason.OFAC_SANCTIONS,  # Iran
    "KP": BlockReason.OFAC_SANCTIONS,  # North Korea
    "SY": BlockReason.OFAC_SANCTIONS,  # Syria
    "RU": BlockReason.EU_SANCTIONS,    # Russia (partial)
    "BY": BlockReason.EU_SANCTIONS,    # Belarus

    # Crimea, Donetsk, Luhansk (special territories)
    # Handled via IP range blocking
}

@dataclass
class GeoCheckResult:
    allowed: bool
    country_code: Optional[str]
    block_reason: Optional[BlockReason] = None

class GeoBlockingService:
    """Check if user's location is allowed."""

    def __init__(self, geoip_provider):
        """
        Args:
            geoip_provider: GeoIP lookup service (MaxMind, IP2Location)
        """
        self._geoip = geoip_provider
        self._blocked = BLOCKED_COUNTRIES

    def check_ip(self, ip_address: str) -> GeoCheckResult:
        """Check if IP address is from allowed jurisdiction."""
        try:
            country = self._geoip.lookup(ip_address)

            if country.code in self._blocked:
                return GeoCheckResult(
                    allowed=False,
                    country_code=country.code,
                    block_reason=self._blocked[country.code]
                )

            return GeoCheckResult(
                allowed=True,
                country_code=country.code
            )
        except Exception:
            # Fail-open for unknown IPs (log for review)
            return GeoCheckResult(allowed=True, country_code=None)

    def check_registration(self, ip_address: str, declared_country: str) -> GeoCheckResult:
        """Check both IP and declared country during registration."""
        ip_check = self.check_ip(ip_address)

        # Block if either IP or declared country is sanctioned
        if not ip_check.allowed:
            return ip_check

        if declared_country in self._blocked:
            return GeoCheckResult(
                allowed=False,
                country_code=declared_country,
                block_reason=self._blocked[declared_country]
            )

        return GeoCheckResult(allowed=True, country_code=declared_country)
```

**Tests:**
```python
class TestGeoBlocking:
    def test_allowed_country(self, service):
        result = service.check_ip("8.8.8.8")  # US
        assert result.allowed

    def test_blocked_country_iran(self, service, mock_geoip):
        mock_geoip.lookup.return_value = Country(code="IR")
        result = service.check_ip("1.2.3.4")
        assert not result.allowed
        assert result.block_reason == BlockReason.OFAC_SANCTIONS

    def test_blocked_country_russia(self, service, mock_geoip):
        mock_geoip.lookup.return_value = Country(code="RU")
        result = service.check_ip("1.2.3.4")
        assert not result.allowed

    def test_registration_checks_declared_country(self, service, mock_geoip):
        mock_geoip.lookup.return_value = Country(code="DE")  # IP says Germany
        result = service.check_registration("1.2.3.4", "KP")  # Declares North Korea
        assert not result.allowed
```

---

### 2.4 API Key Access Logging

**Location:** `services/security/credential_audit.py`

Integrated in `CredentialVault` (section 1.3). Additional structured logging:

```python
# services/security/credential_audit.py
"""
Detailed audit logging for broker credential access.

References:
- ISO 27001 A.12.4: Logging and monitoring
- SOC 2 CC6.1: Logical access security
- PCI DSS 10.2: Audit trail
"""
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class CredentialAccessType(Enum):
    READ = "read"
    DECRYPT = "decrypt"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    FAILED_ATTEMPT = "failed_attempt"

@dataclass
class CredentialAccessEvent:
    timestamp: datetime
    user_id: str
    credential_id: str
    broker: str
    access_type: CredentialAccessType
    purpose: str
    source_ip: str
    success: bool
    error_message: str = None

class CredentialAuditLogger:
    """Structured audit logging for credential access."""

    def __init__(self, storage):
        self._storage = storage
        self._logger = logging.getLogger("credential_audit")

    def log_access(self, event: CredentialAccessEvent):
        """Log credential access event."""
        self._storage.append(event)
        self._logger.info(
            f"CREDENTIAL_ACCESS | "
            f"user={event.user_id} | "
            f"broker={event.broker} | "
            f"type={event.access_type.value} | "
            f"purpose={event.purpose} | "
            f"success={event.success}"
        )

    def get_user_access_history(
        self,
        user_id: str,
        days: int = 90
    ) -> list[CredentialAccessEvent]:
        """Get access history for user (for GDPR requests)."""
        return self._storage.query(
            user_id=user_id,
            since=datetime.utcnow() - timedelta(days=days)
        )

    def detect_anomalies(self, user_id: str) -> list[str]:
        """Detect suspicious access patterns."""
        anomalies = []
        history = self.get_user_access_history(user_id, days=1)

        # Check for unusual volume
        if len(history) > 1000:
            anomalies.append(f"High access volume: {len(history)} in 24h")

        # Check for failures
        failures = [e for e in history if not e.success]
        if len(failures) > 10:
            anomalies.append(f"Multiple failures: {len(failures)}")

        return anomalies
```

**Tests:**
```python
class TestCredentialAuditLogger:
    def test_access_logged(self, logger, mock_storage):
        event = CredentialAccessEvent(
            timestamp=datetime.utcnow(),
            user_id="user_001",
            credential_id="cred_123",
            broker="alpaca",
            access_type=CredentialAccessType.DECRYPT,
            purpose="order_execution",
            source_ip="127.0.0.1",
            success=True
        )
        logger.log_access(event)
        assert mock_storage.append.called

    def test_anomaly_detection_high_volume(self, logger, mock_storage):
        # Simulate 1500 accesses
        mock_storage.query.return_value = [mock_event] * 1500
        anomalies = logger.detect_anomalies("user_001")
        assert any("High access volume" in a for a in anomalies)
```

---

### 2.5 DPA Template (B2B Only)

**Location:** `docs/legal/DPA_TEMPLATE.md`

**Required Sections (GDPR Art. 28):**

```markdown
# Data Processing Agreement

## Parties
- Controller: [Client Company]
- Processor: [Your Company]

## 1. Subject Matter & Duration
- Processing of personal data for trading platform services
- Duration: term of service agreement

## 2. Nature & Purpose of Processing
- Strategy development and backtesting
- Order execution via client's broker API
- Analytics and reporting

## 3. Types of Personal Data
- User identifiers
- Trading strategies (may contain personal logic)
- Execution logs
- Broker API credentials (encrypted)

## 4. Categories of Data Subjects
- Client's employees/traders
- Client's end users (if applicable)

## 5. Processor Obligations (Art. 28(3))
a) Process only on documented instructions
b) Ensure personnel confidentiality
c) Implement appropriate security measures
d) Sub-processor conditions (prior authorization)
e) Assist with data subject rights
f) Assist with DPIA if required
g) Delete/return data on termination
h) Demonstrate compliance, allow audits

## 6. Sub-Processors
- List: [AWS, etc.]
- Prior written authorization required for changes
- 30-day objection period

## 7. Security Measures (Art. 32)
- Encryption at rest (AES-256)
- Encryption in transit (TLS 1.3)
- Access controls
- Regular security testing
- Incident response procedures

## 8. Data Breach Notification
- Notify Controller within 24 hours
- Provide: nature of breach, categories affected, mitigation measures

## 9. International Transfers
- EU-only storage (default)
- If transfer needed: SCCs in Annex

## 10. Audit Rights
- Annual audit right
- 30-day notice required
- Confidentiality of audit findings

## 11. Termination
- Return or delete all data within 30 days
- Certification of deletion provided

## Signatures
[Signature blocks]

## Annex A: Technical and Organizational Measures
## Annex B: Sub-Processor List
## Annex C: Standard Contractual Clauses (if applicable)
```

---

## Summary: Test Coverage

```
tests/
├── test_legal_compliance.py          # ToS, Privacy Policy checks
├── test_credential_vault.py          # Encryption, access logging
├── test_disclaimer_service.py        # UI disclaimers
├── test_backtest_disclaimer.py       # Backtest warnings
├── test_gdpr_deletion.py             # Right to erasure
├── test_gdpr_export.py               # Right to portability
├── test_geo_blocking.py              # Sanctions screening
└── test_credential_audit.py          # Access logging
```

**Coverage targets:**
- Phase 1 (Critical): 100% coverage
- Phase 2 (GDPR): 95% coverage

---

## Implementation Order

```
Week 1-2 (Phase 1):
├── Day 1-2: Terms of Service document
├── Day 3-4: Privacy Policy document
├── Day 5-7: CredentialVault (encryption)
├── Day 8-10: DisclaimerService (UI + backtest)
└── Day 11-14: Integration + tests

Week 3-4 (Phase 2):
├── Day 15-17: GDPRDeletionService
├── Day 18-20: GDPRExportService
├── Day 21-23: GeoBlockingService
├── Day 24-26: CredentialAuditLogger
├── Day 27-28: DPA Template (if B2B)
```

---

## References

1. **GDPR**: Regulation (EU) 2016/679
2. **MiFID II**: Directive 2014/65/EU
3. **E-Commerce Directive**: 2000/31/EC
4. **ESMA Q&A**: ESMA35-43-349
5. **EDPB Guidelines**: WP260 (Transparency), WP242 (Portability)
6. **NIST**: SP 800-57 (Key Management)
7. **OWASP**: Cryptographic Storage Cheat Sheet
8. **OFAC**: Sanctions Programs and Country Information
9. **EU Sanctions**: Council Regulation (EU) 833/2014
