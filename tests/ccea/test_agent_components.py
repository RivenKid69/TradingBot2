# -*- coding: utf-8 -*-
"""
Tests for packages/agent components.

Phase 2 Implementation: Tests for Agent-only components.
"""

from __future__ import annotations

import pytest
import tempfile
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path


class TestLocalVault:
    """Tests for LocalVault credential storage."""

    @pytest.fixture
    def vault(self, tmp_path):
        """Create a vault for testing."""
        from packages.agent.vault.local_vault import LocalVault, VaultConfig
        config = VaultConfig(vault_path=tmp_path / "vault.enc")
        vault = LocalVault(config=config)
        vault.unlock("test_master_key_12345678901234567890")
        return vault

    def test_vault_store_and_retrieve(self, vault):
        """Test storing and retrieving credentials."""
        # Store credential
        cred_id = vault.store(
            broker="alpaca",
            credential_type="api_key",
            value="test_api_key_12345",
            metadata={"environment": "paper"},
        )

        assert cred_id is not None

        # Retrieve credential
        retrieved = vault.retrieve(broker="alpaca", credential_type="api_key")
        assert retrieved == "test_api_key_12345"

    def test_vault_encryption(self, tmp_path):
        """Test that credentials are encrypted at rest."""
        from packages.agent.vault.local_vault import LocalVault, VaultConfig

        vault_path = tmp_path / "vault.enc"
        config = VaultConfig(vault_path=vault_path)
        vault = LocalVault(config=config)
        vault.unlock("test_master_key_12345678901234567890")

        # Store credential
        vault.store(
            broker="binance",
            credential_type="api_secret",
            value="super_secret_value_12345",
        )
        vault.save()

        # Read raw file content
        with open(vault_path, "rb") as f:
            raw_content = f.read()

        # Secret should NOT appear in plaintext
        assert b"super_secret_value_12345" not in raw_content

    def test_vault_delete(self, vault):
        """Test credential deletion."""
        vault.store(broker="test", credential_type="key", value="secret")
        assert vault.retrieve(broker="test", credential_type="key") == "secret"

        vault.delete(broker="test", credential_type="key")
        assert vault.retrieve(broker="test", credential_type="key") is None

    def test_vault_list_credentials(self, vault):
        """Test listing credentials."""
        vault.store(broker="alpaca", credential_type="api_key", value="key1")
        vault.store(broker="alpaca", credential_type="api_secret", value="secret1")
        vault.store(broker="binance", credential_type="api_key", value="key2")

        creds = vault.list_credentials()
        assert len(creds) == 3
        assert any(c["broker"] == "alpaca" and c["credential_type"] == "api_key" for c in creds)


class TestCredentialManager:
    """Tests for CredentialManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        """Create credential manager for testing."""
        from packages.agent.vault.credential_manager import CredentialManager
        from packages.agent.vault.local_vault import LocalVault, VaultConfig

        config = VaultConfig(vault_path=tmp_path / "vault.enc")
        vault = LocalVault(config=config)
        vault.unlock("test_master_key_12345678901234567890")
        return CredentialManager(vault=vault)

    def test_credential_manager_get_broker_credentials(self, manager):
        """Test getting broker credentials."""
        # Store credentials
        manager.vault.store(broker="alpaca", credential_type="api_key", value="APCA123")
        manager.vault.store(broker="alpaca", credential_type="api_secret", value="SECRET456")

        # Get broker credential
        cred = manager.get_broker_credential("alpaca")
        assert cred is not None
        assert cred.api_key == "APCA123"
        assert cred.api_secret == "SECRET456"

    def test_credential_manager_missing_broker(self, manager):
        """Test handling missing broker."""
        cred = manager.get_broker_credential("nonexistent")
        assert cred is None


class TestPolicyFirewall:
    """Tests for PolicyFirewall."""

    def test_firewall_allows_valid_config(self):
        """Test that firewall allows valid config changes."""
        from packages.agent.policy.firewall import PolicyFirewall, PolicyConfig
        from packages.shared.contracts.config import RiskConfig

        firewall = PolicyFirewall(
            policy_config=PolicyConfig(
                max_position_pct_ceiling=Decimal("0.10"),
                max_daily_loss_pct_ceiling=Decimal("0.05"),
            )
        )

        new_config = RiskConfig(
            max_position_pct=Decimal("0.05"),  # Below ceiling
            max_daily_loss_pct=Decimal("0.02"),  # Below ceiling
        )

        result = firewall.check_config_change(new_config, source="cloud")
        assert result.allowed is True

    def test_firewall_blocks_excessive_risk(self):
        """Test that firewall blocks excessive risk limits."""
        from packages.agent.policy.firewall import PolicyFirewall, PolicyConfig
        from packages.shared.contracts.config import RiskConfig

        firewall = PolicyFirewall(
            policy_config=PolicyConfig(
                max_position_pct_ceiling=Decimal("0.10"),
            )
        )

        new_config = RiskConfig(
            max_position_pct=Decimal("0.15"),  # Above ceiling!
        )

        result = firewall.check_config_change(new_config, source="cloud")
        assert result.allowed is False

    def test_firewall_trading_impacting_requires_approval(self):
        """Test that trading-impacting changes require approval."""
        from packages.agent.policy.firewall import PolicyFirewall, PolicyConfig
        from packages.shared.contracts.config import ExecutionConfig

        firewall = PolicyFirewall(
            policy_config=PolicyConfig(
                require_approval_for_trading_impacting=True,
            )
        )

        new_config = ExecutionConfig(
            enable_live_trading=True,  # Trading-impacting!
        )

        result = firewall.check_config_change(new_config, source="cloud")
        assert result.requires_approval is True


class TestHardCapEnforcer:
    """Tests for HardCapEnforcer."""

    def test_hard_caps_position_size(self):
        """Test hard cap on position size."""
        from packages.agent.policy.hard_caps import HardCapEnforcer, HardCaps
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        enforcer = HardCapEnforcer(
            hard_caps=HardCaps(
                max_position_size=Decimal("1000"),
                max_order_size=Decimal("100"),
            )
        )

        # Valid order
        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.OPEN,
            side=IntentSide.LONG,
            target_quantity=Decimal("50"),
        )

        result = enforcer.check(intent)
        assert result.passed is True

    def test_hard_caps_reject_excessive_order(self):
        """Test hard cap rejects excessive order."""
        from packages.agent.policy.hard_caps import HardCapEnforcer, HardCaps
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide

        enforcer = HardCapEnforcer(
            hard_caps=HardCaps(
                max_order_size=Decimal("100"),
            )
        )

        # Excessive order
        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.OPEN,
            side=IntentSide.LONG,
            target_quantity=Decimal("500"),  # Exceeds max!
        )

        result = enforcer.check(intent)
        assert result.passed is False


class TestRiskChecker:
    """Tests for RiskChecker."""

    def test_risk_checker_pre_trade(self):
        """Test pre-trade risk checks."""
        from packages.agent.policy.risk_checker import RiskChecker, PreTradeCheck
        from packages.shared.contracts.intent import OrderIntent, IntentType, IntentSide
        from packages.shared.contracts.config import RiskConfig

        checker = RiskChecker(
            risk_config=RiskConfig(
                max_position_pct=Decimal("0.05"),
            )
        )

        intent = OrderIntent(
            strategy_id="test",
            symbol="AAPL",
            intent_type=IntentType.OPEN,
            side=IntentSide.LONG,
            target_quantity=Decimal("100"),
        )

        result = checker.pre_trade_check(
            intent=intent,
            account_value=Decimal("100000"),
            current_price=Decimal("150"),
        )

        assert isinstance(result, PreTradeCheck)


class TestOrderJournal:
    """Tests for OrderJournal."""

    @pytest.fixture
    def journal(self, tmp_path):
        """Create order journal for testing."""
        from packages.agent.reconciliation.journal import OrderJournal
        return OrderJournal(journal_path=tmp_path / "journal.db")

    def test_journal_record_order(self, journal):
        """Test recording order in journal."""
        order_id = journal.record_order(
            strategy_id="test",
            symbol="AAPL",
            side="long",
            quantity=Decimal("100"),
            order_type="market",
        )

        assert order_id is not None

        # Retrieve order
        order = journal.get_order(order_id)
        assert order is not None
        assert order["symbol"] == "AAPL"

    def test_journal_update_fill(self, journal):
        """Test updating order with fill."""
        order_id = journal.record_order(
            strategy_id="test",
            symbol="AAPL",
            side="long",
            quantity=Decimal("100"),
            order_type="market",
        )

        journal.record_fill(
            order_id=order_id,
            fill_quantity=Decimal("100"),
            fill_price=Decimal("150.25"),
        )

        order = journal.get_order(order_id)
        assert order["status"] == "filled"
        assert order["filled_quantity"] == Decimal("100")


class TestApprovalManager:
    """Tests for ApprovalManager."""

    def test_approval_request_creation(self):
        """Test creating approval request."""
        from packages.agent.approval.manager import ApprovalManager

        manager = ApprovalManager()

        request = manager.create_request(
            change_type="enable_live_trading",
            source="cloud",
            details={"new_value": True},
        )

        assert request is not None
        assert request.status == "pending"

    def test_approval_approve(self):
        """Test approving request."""
        from packages.agent.approval.manager import ApprovalManager

        manager = ApprovalManager()

        request = manager.create_request(
            change_type="risk_limit_increase",
            source="cloud",
            details={},
        )

        result = manager.approve(request.request_id, approver="admin")
        assert result.status == "approved"

    def test_approval_reject(self):
        """Test rejecting request."""
        from packages.agent.approval.manager import ApprovalManager

        manager = ApprovalManager()

        request = manager.create_request(
            change_type="risk_limit_increase",
            source="cloud",
            details={},
        )

        result = manager.reject(request.request_id, reason="Too risky")
        assert result.status == "rejected"


class TestEvidenceRecord:
    """Tests for EvidenceRecord."""

    def test_evidence_record_creation(self):
        """Test creating evidence record."""
        from packages.agent.approval.evidence import EvidenceRecord

        evidence = EvidenceRecord(
            action_type="order_submitted",
            actor="agent",
            details={"order_id": "12345", "symbol": "AAPL"},
        )

        assert evidence.action_type == "order_submitted"
        assert evidence.hash is not None

    def test_evidence_hash_consistency(self):
        """Test that evidence hash is consistent."""
        from packages.agent.approval.evidence import compute_evidence_hash

        data = {"action": "test", "value": 123}
        hash1 = compute_evidence_hash(data)
        hash2 = compute_evidence_hash(data)

        assert hash1 == hash2
