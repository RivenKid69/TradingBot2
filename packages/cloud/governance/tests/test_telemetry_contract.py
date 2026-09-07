# -*- coding: utf-8 -*-
"""
Tests for Telemetry Contract Enforcement.

GDPR Phase 2: Tests for TelemetryLevelContract, RawOrderEventsGate,
and TelemetryContractValidator.

Coverage:
    - Contract field validation per level
    - RAW_ORDER_EVENTS enterprise gating
    - Credential detection
    - Forbidden field rejection
    - Opt-in workflow
"""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from ccea.models.protocol import TelemetryLevel
from packages.cloud.governance.telemetry_contract import (
    TelemetryLevelContract,
    TelemetryContractValidator,
    RawOrderEventsGate,
    EnterpriseRawOptIn,
    ContractViolation,
    ContractValidationResult,
    ContractViolationType,
    ViolationSeverity,
    validate_telemetry_contract,
    get_allowed_fields_for_level,
    get_forbidden_fields_for_level,
    AGGREGATED_ALLOWED_FIELDS,
    DETAILED_ALLOWED_FIELDS,
    RAW_ORDER_ALLOWED_FIELDS,
    ALWAYS_FORBIDDEN_FIELDS,
    ORDER_LIKE_FIELDS,
    PII_FIELDS,
)


# ============================================================================
# Test TelemetryLevelContract
# ============================================================================


class TestTelemetryLevelContract:
    """Tests for TelemetryLevelContract."""

    def test_aggregated_contract_defaults(self):
        """Test AGGREGATED level contract has correct defaults."""
        contract = TelemetryLevelContract.for_aggregated()

        assert contract.level == TelemetryLevel.AGGREGATED
        assert contract.requires_enterprise is False
        assert contract.requires_explicit_opt_in is False
        assert contract.requires_redaction is False  # Aggregated doesn't need redaction
        assert contract.max_retention_days == 365

    def test_aggregated_allows_metrics(self):
        """Test AGGREGATED allows standard metrics."""
        contract = TelemetryLevelContract.for_aggregated()

        # Standard metrics should be allowed
        assert "timestamp" in contract.allowed_fields
        assert "cpu_percent" in contract.allowed_fields
        assert "memory_percent" in contract.allowed_fields
        assert "trade_count" in contract.allowed_fields
        assert "sharpe_ratio" in contract.allowed_fields

    def test_aggregated_forbids_order_fields(self):
        """Test AGGREGATED forbids order-like fields."""
        contract = TelemetryLevelContract.for_aggregated()

        # Order fields should be forbidden
        assert "side" in contract.forbidden_fields
        assert "quantity" in contract.forbidden_fields
        assert "price" in contract.forbidden_fields
        assert "order_id" in contract.forbidden_fields
        assert "intent" in contract.forbidden_fields
        assert "signal" in contract.forbidden_fields

    def test_detailed_contract_requires_opt_in(self):
        """Test DETAILED_NON_SENSITIVE requires opt-in."""
        contract = TelemetryLevelContract.for_detailed_non_sensitive()

        assert contract.level == TelemetryLevel.DETAILED_NON_SENSITIVE
        assert contract.requires_enterprise is False
        assert contract.requires_explicit_opt_in is True
        assert contract.requires_redaction is True
        assert contract.max_retention_days == 90

    def test_detailed_allows_more_fields(self):
        """Test DETAILED allows additional fields beyond AGGREGATED."""
        contract = TelemetryLevelContract.for_detailed_non_sensitive()

        # Additional detailed fields
        assert "event_type" in contract.allowed_fields
        assert "message" in contract.allowed_fields
        assert "trace_id" in contract.allowed_fields
        assert "error_code" in contract.allowed_fields

        # Still includes aggregated fields
        assert "timestamp" in contract.allowed_fields
        assert "cpu_percent" in contract.allowed_fields

    def test_detailed_still_forbids_orders(self):
        """Test DETAILED still forbids order fields."""
        contract = TelemetryLevelContract.for_detailed_non_sensitive()

        assert "side" in contract.forbidden_fields
        assert "quantity" in contract.forbidden_fields
        assert "intent" in contract.forbidden_fields

    def test_raw_contract_requires_enterprise(self):
        """Test RAW_ORDER_EVENTS requires enterprise."""
        contract = TelemetryLevelContract.for_raw_order_events()

        assert contract.level == TelemetryLevel.RAW_ORDER_EVENTS
        assert contract.requires_enterprise is True
        assert contract.requires_explicit_opt_in is True
        assert contract.requires_redaction is True
        assert contract.max_retention_days == 30

    def test_raw_allows_order_fields(self):
        """Test RAW allows order fields for enterprise."""
        contract = TelemetryLevelContract.for_raw_order_events()

        # Order fields allowed in RAW
        assert "side" in contract.allowed_fields
        assert "quantity" in contract.allowed_fields
        assert "price" in contract.allowed_fields
        assert "order_id" in contract.allowed_fields
        assert "signal" in contract.allowed_fields
        assert "intent" in contract.allowed_fields

    def test_raw_still_forbids_credentials(self):
        """Test RAW still forbids credentials."""
        contract = TelemetryLevelContract.for_raw_order_events()

        # Credentials always forbidden
        assert "api_key" in contract.forbidden_fields
        assert "api_secret" in contract.forbidden_fields
        assert "password" in contract.forbidden_fields
        assert "private_key" in contract.forbidden_fields

    def test_get_contract_by_level(self):
        """Test getting contract by level."""
        agg = TelemetryLevelContract.get_contract(TelemetryLevel.AGGREGATED)
        assert agg.level == TelemetryLevel.AGGREGATED

        det = TelemetryLevelContract.get_contract(TelemetryLevel.DETAILED_NON_SENSITIVE)
        assert det.level == TelemetryLevel.DETAILED_NON_SENSITIVE

        raw = TelemetryLevelContract.get_contract(TelemetryLevel.RAW_ORDER_EVENTS)
        assert raw.level == TelemetryLevel.RAW_ORDER_EVENTS

    def test_get_contract_invalid_level(self):
        """Test getting contract with invalid level raises error."""
        with pytest.raises(ValueError, match="Unknown telemetry level"):
            TelemetryLevelContract.get_contract("INVALID_LEVEL")


# ============================================================================
# Test RawOrderEventsGate
# ============================================================================


class TestRawOrderEventsGate:
    """Tests for RawOrderEventsGate."""

    def test_gate_requires_enterprise_license(self):
        """Test gate requires enterprise license."""

        def enterprise_check(ws_id, org_id):
            return False, "Enterprise license required"

        gate = RawOrderEventsGate(enterprise_checker=enterprise_check)

        is_allowed, error = gate.check_enterprise_license("ws_1", "org_1")
        assert is_allowed is False
        assert "Enterprise" in error

    def test_gate_requires_opt_in(self):
        """Test gate requires explicit opt-in."""
        gate = RawOrderEventsGate()

        # No opt-in registered
        is_allowed, error = gate.check_explicit_opt_in("ws_1")
        assert is_allowed is False
        assert "opt-in" in error.lower()

    def test_gate_passes_with_valid_opt_in(self):
        """Test gate passes with valid opt-in."""
        gate = RawOrderEventsGate()

        # Register opt-in
        opt_in = gate.register_opt_in(
            workspace_id="ws_1",
            organization_id="org_1",
            principal_id="user_1",
            acknowledgment_text="I acknowledge RAW data risks",
        )

        assert opt_in.is_valid
        is_allowed, error = gate.check_explicit_opt_in("ws_1")
        assert is_allowed is True
        assert error is None

    def test_gate_rejects_revoked_opt_in(self):
        """Test gate rejects revoked opt-in."""
        gate = RawOrderEventsGate()

        # Register and then revoke
        gate.register_opt_in(
            workspace_id="ws_1",
            organization_id="org_1",
            principal_id="user_1",
            acknowledgment_text="I acknowledge",
        )
        gate.revoke_opt_in("ws_1", "admin_1")

        is_allowed, error = gate.check_explicit_opt_in("ws_1")
        assert is_allowed is False
        assert "revoked" in error.lower()

    def test_gate_rejects_expired_opt_in(self):
        """Test gate rejects expired opt-in."""
        gate = RawOrderEventsGate()

        # Register with past expiry
        opt_in = EnterpriseRawOptIn(
            workspace_id="ws_1",
            organization_id="org_1",
            opted_in_at=datetime.now(timezone.utc) - timedelta(days=60),
            opted_in_by="user_1",
            acknowledgment_hash="sha256:abc",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        gate._opt_in_store["ws_1"] = opt_in

        is_allowed, error = gate.check_explicit_opt_in("ws_1")
        assert is_allowed is False
        assert "expired" in error.lower()

    def test_gate_requires_dpa_signed(self):
        """Test gate requires DPA signed."""
        gate = RawOrderEventsGate()

        opt_in = EnterpriseRawOptIn(
            workspace_id="ws_1",
            organization_id="org_1",
            opted_in_at=datetime.now(timezone.utc),
            opted_in_by="user_1",
            acknowledgment_hash="sha256:abc",
            dpa_signed=False,  # DPA not signed
        )
        gate._opt_in_store["ws_1"] = opt_in

        is_allowed, error = gate.check_explicit_opt_in("ws_1")
        assert is_allowed is False
        assert "DPA" in error

    def test_full_gate_validation(self):
        """Test full gate validation flow."""
        gate = RawOrderEventsGate()

        # Without opt-in
        ok, errors = gate.validate_gate("ws_1", "org_1")
        assert ok is False
        assert len(errors) > 0

        # With opt-in
        gate.register_opt_in(
            workspace_id="ws_1",
            organization_id="org_1",
            principal_id="user_1",
            acknowledgment_text="I acknowledge",
        )

        ok, errors = gate.validate_gate("ws_1", "org_1")
        assert ok is True
        assert len(errors) == 0


# ============================================================================
# Test TelemetryContractValidator
# ============================================================================


class TestTelemetryContractValidator:
    """Tests for TelemetryContractValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gate = RawOrderEventsGate()
        self.validator = TelemetryContractValidator(raw_gate=self.gate)

    def test_validate_aggregated_valid_payload(self):
        """Test validating valid AGGREGATED payload."""
        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "cpu_percent": 45.0,
            "memory_percent": 60.0,
            "trade_count": 100,
            "sharpe_ratio": 1.5,
        }

        result = self.validator.validate(
            payload,
            TelemetryLevel.AGGREGATED,
            "ws_1",
            "org_1",
            redaction_applied=True,
        )

        assert result.valid is True
        assert result.blocked is False
        assert len(result.violations) == 0

    def test_validate_aggregated_rejects_order_fields(self):
        """Test AGGREGATED rejects order-like fields."""
        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "side": "BUY",  # Forbidden!
            "quantity": 100,  # Forbidden!
            "price": 150.0,  # Forbidden!
        }

        result = self.validator.validate(
            payload,
            TelemetryLevel.AGGREGATED,
            "ws_1",
            "org_1",
        )

        assert result.valid is False
        assert result.blocked is True
        assert len(result.violations) >= 3
        assert any(
            v.violation_type == ContractViolationType.FORBIDDEN_FIELD for v in result.violations
        )

    def test_validate_detailed_requires_redaction(self):
        """Test DETAILED requires redaction_applied=True."""
        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "event_type": "heartbeat",
        }

        # Without redaction
        result = self.validator.validate(
            payload,
            TelemetryLevel.DETAILED_NON_SENSITIVE,
            "ws_1",
            "org_1",
            redaction_applied=False,
        )

        assert result.valid is False
        assert any(
            v.violation_type == ContractViolationType.REDACTION_MISSING for v in result.violations
        )

    def test_validate_raw_requires_enterprise(self):
        """Test RAW requires enterprise + opt-in."""
        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "side": "BUY",
            "quantity": 100,
        }

        # Without enterprise opt-in
        result = self.validator.validate(
            payload,
            TelemetryLevel.RAW_ORDER_EVENTS,
            "ws_1",
            "org_1",
        )

        assert result.valid is False
        assert result.enterprise_verified is False
        assert any(
            v.violation_type == ContractViolationType.ENTERPRISE_REQUIRED for v in result.violations
        )

    def test_validate_raw_passes_with_enterprise(self):
        """Test RAW passes with enterprise opt-in."""
        # Register opt-in
        self.gate.register_opt_in(
            workspace_id="ws_1",
            organization_id="org_1",
            principal_id="user_1",
            acknowledgment_text="I acknowledge",
        )

        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "side": "BUY",
            "quantity": 100,
            "price": 150.0,
        }

        result = self.validator.validate(
            payload,
            TelemetryLevel.RAW_ORDER_EVENTS,
            "ws_1",
            "org_1",
            redaction_applied=True,
        )

        assert result.valid is True
        assert result.enterprise_verified is True
        assert result.opt_in_verified is True

    def test_validate_detects_credentials(self):
        """Test validator detects credentials in all levels."""
        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "api_key": "sk-1234567890abcdef",
        }

        # Credentials forbidden even in AGGREGATED
        result = self.validator.validate(
            payload,
            TelemetryLevel.AGGREGATED,
            "ws_1",
            "org_1",
        )

        assert result.valid is False
        assert any(
            v.violation_type == ContractViolationType.CREDENTIAL_DETECTED for v in result.violations
        )

    def test_validate_detects_credential_patterns(self):
        """Test validator detects credential patterns."""
        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "binance_secret": "secret123",
        }

        result = self.validator.validate(
            payload,
            TelemetryLevel.AGGREGATED,
            "ws_1",
            "org_1",
        )

        assert result.valid is False
        assert any(
            v.violation_type == ContractViolationType.CREDENTIAL_DETECTED for v in result.violations
        )

    def test_validate_detects_credential_values(self):
        """Test validator detects credential-like values."""
        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "some_field": "sk-abcdefghijklmnopqrstuvwxyz123456",
        }

        result = self.validator.validate(
            payload,
            TelemetryLevel.AGGREGATED,
            "ws_1",
            "org_1",
        )

        assert result.valid is False
        assert any(
            v.violation_type == ContractViolationType.CREDENTIAL_DETECTED for v in result.violations
        )

    def test_validate_nested_forbidden_fields(self):
        """Test validator catches forbidden fields in nested structures."""
        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "nested": {
                "data": {
                    "side": "BUY",  # Nested forbidden field
                }
            },
        }

        result = self.validator.validate(
            payload,
            TelemetryLevel.AGGREGATED,
            "ws_1",
            "org_1",
        )

        assert result.valid is False
        assert any("nested.data.side" in v.field_path for v in result.violations)

    def test_raw_still_forbids_credentials(self):
        """Test RAW still forbids credentials even with enterprise."""
        self.gate.register_opt_in(
            workspace_id="ws_1",
            organization_id="org_1",
            principal_id="user_1",
            acknowledgment_text="I acknowledge",
        )

        payload = {
            "timestamp": "2025-12-16T10:00:00Z",
            "side": "BUY",
            "api_key": "secret123",  # Still forbidden!
        }

        result = self.validator.validate(
            payload,
            TelemetryLevel.RAW_ORDER_EVENTS,
            "ws_1",
            "org_1",
            redaction_applied=True,
        )

        assert result.valid is False
        assert any(
            v.violation_type == ContractViolationType.CREDENTIAL_DETECTED for v in result.violations
        )


# ============================================================================
# Test Convenience Functions
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_telemetry_contract_function(self):
        """Test validate_telemetry_contract function."""
        payload = {"timestamp": "2025-12-16T10:00:00Z", "cpu_percent": 50}

        result = validate_telemetry_contract(
            payload,
            "AGGREGATED",
            "ws_1",
            "org_1",
        )

        assert result.valid is True

    def test_validate_with_invalid_level(self):
        """Test validation with invalid level string."""
        payload = {"timestamp": "2025-12-16T10:00:00Z"}

        result = validate_telemetry_contract(
            payload,
            "INVALID_LEVEL",
            "ws_1",
            "org_1",
        )

        assert result.valid is False
        assert "Invalid telemetry level" in result.violations[0].message

    def test_get_allowed_fields(self):
        """Test get_allowed_fields_for_level."""
        agg_fields = get_allowed_fields_for_level("AGGREGATED")
        assert "timestamp" in agg_fields
        assert "cpu_percent" in agg_fields
        assert "side" not in agg_fields

        det_fields = get_allowed_fields_for_level("DETAILED_NON_SENSITIVE")
        assert "event_type" in det_fields
        assert "timestamp" in det_fields

        raw_fields = get_allowed_fields_for_level("RAW_ORDER_EVENTS")
        assert "side" in raw_fields
        assert "quantity" in raw_fields

    def test_get_forbidden_fields(self):
        """Test get_forbidden_fields_for_level."""
        agg_forbidden = get_forbidden_fields_for_level("AGGREGATED")
        assert "side" in agg_forbidden
        assert "api_key" in agg_forbidden

        raw_forbidden = get_forbidden_fields_for_level("RAW_ORDER_EVENTS")
        assert "api_key" in raw_forbidden
        assert "side" not in raw_forbidden  # RAW allows order fields

    def test_get_fields_invalid_level(self):
        """Test get fields with invalid level returns empty set."""
        allowed = get_allowed_fields_for_level("INVALID")
        assert allowed == frozenset()

        forbidden = get_forbidden_fields_for_level("INVALID")
        assert forbidden == frozenset()


# ============================================================================
# Test Field Sets
# ============================================================================


class TestFieldSets:
    """Tests for field set constants."""

    def test_aggregated_fields_not_empty(self):
        """Test AGGREGATED_ALLOWED_FIELDS is not empty."""
        assert len(AGGREGATED_ALLOWED_FIELDS) > 0
        assert "timestamp" in AGGREGATED_ALLOWED_FIELDS

    def test_detailed_fields_not_empty(self):
        """Test DETAILED_ALLOWED_FIELDS is not empty."""
        assert len(DETAILED_ALLOWED_FIELDS) > 0
        assert "event_type" in DETAILED_ALLOWED_FIELDS

    def test_raw_fields_not_empty(self):
        """Test RAW_ORDER_ALLOWED_FIELDS is not empty."""
        assert len(RAW_ORDER_ALLOWED_FIELDS) > 0
        assert "side" in RAW_ORDER_ALLOWED_FIELDS
        assert "quantity" in RAW_ORDER_ALLOWED_FIELDS

    def test_always_forbidden_not_empty(self):
        """Test ALWAYS_FORBIDDEN_FIELDS is not empty."""
        assert len(ALWAYS_FORBIDDEN_FIELDS) > 0
        assert "api_key" in ALWAYS_FORBIDDEN_FIELDS
        assert "password" in ALWAYS_FORBIDDEN_FIELDS

    def test_pii_fields_not_empty(self):
        """Test PII_FIELDS is not empty."""
        assert len(PII_FIELDS) > 0
        assert "email" in PII_FIELDS
        assert "ssn" in PII_FIELDS

    def test_order_like_fields_not_empty(self):
        """Test ORDER_LIKE_FIELDS is not empty."""
        assert len(ORDER_LIKE_FIELDS) > 0
        assert "side" in ORDER_LIKE_FIELDS
        assert "quantity" in ORDER_LIKE_FIELDS
        assert "intent" in ORDER_LIKE_FIELDS

    def test_no_overlap_between_aggregated_and_order(self):
        """Test no overlap between AGGREGATED allowed and ORDER_LIKE."""
        overlap = AGGREGATED_ALLOWED_FIELDS & ORDER_LIKE_FIELDS
        assert len(overlap) == 0, f"Unexpected overlap: {overlap}"

    def test_credentials_not_in_any_allowed(self):
        """Test credentials not in any allowed set."""
        all_allowed = AGGREGATED_ALLOWED_FIELDS | DETAILED_ALLOWED_FIELDS | RAW_ORDER_ALLOWED_FIELDS

        for cred_field in ALWAYS_FORBIDDEN_FIELDS:
            assert (
                cred_field not in all_allowed
            ), f"Credential field {cred_field} should not be allowed"


# ============================================================================
# Test EnterpriseRawOptIn
# ============================================================================


class TestEnterpriseRawOptIn:
    """Tests for EnterpriseRawOptIn."""

    def test_valid_opt_in(self):
        """Test valid opt-in is_valid."""
        opt_in = EnterpriseRawOptIn(
            workspace_id="ws_1",
            organization_id="org_1",
            opted_in_at=datetime.now(timezone.utc),
            opted_in_by="user_1",
            acknowledgment_hash="sha256:abc",
        )

        assert opt_in.is_valid is True

    def test_revoked_opt_in_invalid(self):
        """Test revoked opt-in is invalid."""
        opt_in = EnterpriseRawOptIn(
            workspace_id="ws_1",
            organization_id="org_1",
            opted_in_at=datetime.now(timezone.utc),
            opted_in_by="user_1",
            acknowledgment_hash="sha256:abc",
            revoked=True,
            revoked_at=datetime.now(timezone.utc),
        )

        assert opt_in.is_valid is False

    def test_expired_opt_in_invalid(self):
        """Test expired opt-in is invalid."""
        opt_in = EnterpriseRawOptIn(
            workspace_id="ws_1",
            organization_id="org_1",
            opted_in_at=datetime.now(timezone.utc) - timedelta(days=60),
            opted_in_by="user_1",
            acknowledgment_hash="sha256:abc",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )

        assert opt_in.is_valid is False

    def test_unsigned_dpa_invalid(self):
        """Test unsigned DPA makes opt-in invalid."""
        opt_in = EnterpriseRawOptIn(
            workspace_id="ws_1",
            organization_id="org_1",
            opted_in_at=datetime.now(timezone.utc),
            opted_in_by="user_1",
            acknowledgment_hash="sha256:abc",
            dpa_signed=False,
        )

        assert opt_in.is_valid is False


# ============================================================================
# Test ContractViolation
# ============================================================================


class TestContractViolation:
    """Tests for ContractViolation."""

    def test_to_dict(self):
        """Test ContractViolation.to_dict()."""
        violation = ContractViolation(
            violation_type=ContractViolationType.FORBIDDEN_FIELD,
            severity=ViolationSeverity.CRITICAL,
            field_path="data.side",
            message="Field 'side' is forbidden",
            telemetry_level="AGGREGATED",
        )

        d = violation.to_dict()

        assert d["violation_type"] == "forbidden_field"
        assert d["severity"] == "critical"
        assert d["field_path"] == "data.side"
        assert d["blocked"] is True


# ============================================================================
# Test ContractValidationResult
# ============================================================================


class TestContractValidationResult:
    """Tests for ContractValidationResult."""

    def test_critical_violations_filter(self):
        """Test critical_violations property."""
        result = ContractValidationResult(
            valid=False,
            telemetry_level="AGGREGATED",
            violations=[
                ContractViolation(
                    violation_type=ContractViolationType.FORBIDDEN_FIELD,
                    severity=ViolationSeverity.CRITICAL,
                    field_path="a",
                    message="m1",
                    telemetry_level="AGGREGATED",
                ),
                ContractViolation(
                    violation_type=ContractViolationType.FORBIDDEN_FIELD,
                    severity=ViolationSeverity.MEDIUM,
                    field_path="b",
                    message="m2",
                    telemetry_level="AGGREGATED",
                ),
            ],
        )

        assert len(result.critical_violations) == 1
        assert result.critical_violations[0].field_path == "a"

    def test_to_dict(self):
        """Test ContractValidationResult.to_dict()."""
        result = ContractValidationResult(
            valid=True,
            telemetry_level="AGGREGATED",
            enterprise_verified=True,
        )

        d = result.to_dict()

        assert d["valid"] is True
        assert d["telemetry_level"] == "AGGREGATED"
        assert d["enterprise_verified"] is True
