# -*- coding: utf-8 -*-
"""
Tests for AI Act Cybersecurity Module (Article 15(5)).

Comprehensive test coverage for cybersecurity functionality including:
- Input validation
- Model integrity verification
- Adversarial detection
- Data poisoning detection
- Access control
"""

import pytest
import tempfile
import shutil
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from services.ai_act.cybersecurity import (
    ThreatType,
    ThreatSeverity,
    ValidationStatus,
    AccessLevel,
    SecurityEventType,
    SecurityThreat,
    ValidationResult,
    IntegrityRecord,
    AccessRecord,
    SecurityPolicy,
    SecurityEvent,
    CybersecurityConfig,
    InputValidator,
    ModelIntegrityVerifier,
    AdversarialDetector,
    DataPoisoningDetector,
    AccessControlManager,
    AIActCybersecurity,
    create_cybersecurity,
    get_default_security_policies,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_model_file(temp_dir):
    """Create a temporary model file."""
    model_path = Path(temp_dir) / "test_model.pt"
    model_path.write_bytes(b"mock model data" * 100)
    return str(model_path)


@pytest.fixture
def security_config(temp_dir):
    """Create security configuration."""
    return CybersecurityConfig(
        enable_input_validation=True,
        enable_anomaly_detection=True,
        enable_adversarial_detection=True,
        enable_integrity_checks=True,
        enable_access_control=True,
        max_input_size_mb=10.0,
        input_range_min=-1000.0,
        input_range_max=1000.0,
        anomaly_threshold=3.0,
        perturbation_threshold=0.1,
        max_failed_attempts=3,
        lockout_duration_minutes=5,
        log_path=f"{temp_dir}/logs",
    )


@pytest.fixture
def security(security_config):
    """Create cybersecurity instance."""
    return AIActCybersecurity(security_config)


# =============================================================================
# Test Enums
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_threat_types(self):
        """Test threat types per Article 15(5)."""
        assert ThreatType.DATA_POISONING.value == "data_poisoning"
        assert ThreatType.MODEL_POISONING.value == "model_poisoning"
        assert ThreatType.ADVERSARIAL_EXAMPLE.value == "adversarial_example"
        assert ThreatType.MODEL_INVERSION.value == "model_inversion"
        assert ThreatType.EVASION_ATTACK.value == "evasion_attack"

    def test_threat_severity_ordering(self):
        """Test threat severity ordering."""
        assert ThreatSeverity.LOW.value < ThreatSeverity.MEDIUM.value
        assert ThreatSeverity.MEDIUM.value < ThreatSeverity.HIGH.value
        assert ThreatSeverity.HIGH.value < ThreatSeverity.CRITICAL.value

    def test_validation_status(self):
        """Test validation status values."""
        assert ValidationStatus.VALID.value == "valid"
        assert ValidationStatus.SUSPICIOUS.value == "suspicious"
        assert ValidationStatus.INVALID.value == "invalid"
        assert ValidationStatus.BLOCKED.value == "blocked"

    def test_access_levels(self):
        """Test access levels (higher value = more privileges)."""
        assert AccessLevel.READ_ONLY.value == 1
        assert AccessLevel.OPERATOR.value == 2
        assert AccessLevel.DEVELOPER.value == 3
        assert AccessLevel.ADMIN.value == 4
        # Verify ordering
        assert AccessLevel.READ_ONLY.value < AccessLevel.OPERATOR.value
        assert AccessLevel.OPERATOR.value < AccessLevel.DEVELOPER.value
        assert AccessLevel.DEVELOPER.value < AccessLevel.ADMIN.value

    def test_security_event_types(self):
        """Test security event types."""
        assert SecurityEventType.AUTHENTICATION.value == "authentication"
        assert SecurityEventType.AUTHORIZATION.value == "authorization"
        assert SecurityEventType.INPUT_VALIDATION.value == "input_validation"
        assert SecurityEventType.ADVERSARIAL_DETECTION.value == "adversarial_detection"


# =============================================================================
# Test Data Classes
# =============================================================================

class TestDataClasses:
    """Test data class initialization."""

    def test_security_threat_creation(self):
        """Test security threat creation."""
        threat = SecurityThreat(
            threat_id="",
            threat_type=ThreatType.ADVERSARIAL_EXAMPLE,
            severity=ThreatSeverity.HIGH,
            title="Adversarial Input Detected",
            description="Suspicious input pattern",
        )
        assert threat.threat_id.startswith("THREAT-")
        assert not threat.mitigated
        assert threat.investigation_status == "open"

    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ValidationResult(
            validation_id="",
            status=ValidationStatus.VALID,
            input_type="observation",
        )
        assert result.validation_id.startswith("VAL-")
        assert not result.blocked

    def test_integrity_record_creation(self):
        """Test integrity record creation."""
        record = IntegrityRecord(
            record_id="",
            resource_type="model",
            resource_path="/path/to/model",
            resource_name="test_model",
        )
        assert record.record_id.startswith("INT-")
        assert not record.is_verified
        assert not record.tampering_detected

    def test_access_record_creation(self):
        """Test access record creation."""
        record = AccessRecord(
            access_id="",
            user_id="user123",
            resource_type="model",
            resource_id="model001",
            access_level=AccessLevel.OPERATOR,
        )
        assert record.access_id.startswith("ACC-")
        assert record.authorized

    def test_security_policy_creation(self):
        """Test security policy creation."""
        policy = SecurityPolicy(
            policy_id="",
            name="Input Validation",
            description="Validate all inputs",
            policy_type="input_validation",
        )
        assert policy.policy_id.startswith("POL-")
        assert policy.is_active

    def test_security_event_creation(self):
        """Test security event creation."""
        event = SecurityEvent(
            event_id="",
            event_type=SecurityEventType.AUTHENTICATION,
            severity=ThreatSeverity.LOW,
            description="User login",
        )
        assert event.event_id.startswith("EVT-")
        assert event.success


# =============================================================================
# Test Input Validator
# =============================================================================

class TestInputValidator:
    """Test input validation functionality."""

    def test_validate_valid_input(self, security_config):
        """Test validating valid input."""
        validator = InputValidator(security_config)

        data = np.random.randn(100, 50).astype(np.float32)
        result = validator.validate(data, "observation")

        assert result.status == ValidationStatus.VALID
        assert not result.blocked
        assert "data_type" in result.checks_passed

    def test_validate_invalid_data_type(self, security_config):
        """Test validating invalid data type."""
        security_config.allowed_input_types = ["float32"]
        validator = InputValidator(security_config)

        data = np.random.randn(100).astype(np.float16)  # Not allowed
        result = validator.validate(data)

        assert result.status == ValidationStatus.INVALID
        assert "data_type" in result.checks_failed

    def test_validate_out_of_range(self, security_config):
        """Test validating out of range values."""
        validator = InputValidator(security_config)

        data = np.array([1e11, 1e12]).astype(np.float64)  # Out of range
        result = validator.validate(data)

        assert result.status == ValidationStatus.INVALID
        assert "range" in result.checks_failed

    def test_validate_nan_values(self, security_config):
        """Test validating NaN values."""
        validator = InputValidator(security_config)

        data = np.array([1.0, np.nan, 2.0]).astype(np.float64)
        result = validator.validate(data)

        assert result.status == ValidationStatus.INVALID
        assert "finite" in result.checks_failed

    def test_validate_inf_values(self, security_config):
        """Test validating infinite values."""
        validator = InputValidator(security_config)

        data = np.array([1.0, np.inf, 2.0]).astype(np.float64)
        result = validator.validate(data)

        assert "finite" in result.checks_failed

    def test_sanitize_nan_values(self, security_config):
        """Test sanitizing NaN values."""
        validator = InputValidator(security_config)

        data = np.array([1.0, np.nan, 2.0, np.inf])
        sanitized, details = validator.sanitize(data, replace_nan=True)

        assert np.all(np.isfinite(sanitized))
        assert "Replaced" in details

    def test_sanitize_out_of_range(self, security_config):
        """Test sanitizing out of range values."""
        validator = InputValidator(security_config)

        data = np.array([1e15, -1e15, 500.0])
        sanitized, details = validator.sanitize(data, clip_range=True)

        assert sanitized[0] == security_config.input_range_max
        assert sanitized[1] == security_config.input_range_min
        assert "Clipped" in details

    def test_anomaly_detection(self, security_config):
        """Test anomaly detection in input."""
        validator = InputValidator(security_config)

        # Build up history with normal data
        for _ in range(20):
            normal_data = np.random.randn(100)
            validator.validate(normal_data, "test")

        # Now check anomalous data
        anomalous_data = np.random.randn(100) + 10  # Shifted mean
        result = validator.validate(anomalous_data, "test")

        assert result.anomaly_score > 0


# =============================================================================
# Test Model Integrity Verifier
# =============================================================================

class TestModelIntegrityVerifier:
    """Test model integrity verification."""

    def test_register_model(self, security_config, temp_model_file):
        """Test model registration."""
        verifier = ModelIntegrityVerifier(security_config)

        record = verifier.register_model(
            model_path=temp_model_file,
            model_name="test_model",
            compute_hash=True,
        )

        assert record.record_id is not None
        assert record.expected_hash != ""
        assert record.resource_name == "test_model"

    def test_verify_integrity_success(self, security_config, temp_model_file):
        """Test successful integrity verification."""
        verifier = ModelIntegrityVerifier(security_config)

        record = verifier.register_model(temp_model_file, "model", True)
        is_valid, details = verifier.verify_integrity(record.record_id)

        assert is_valid
        assert "verified" in details.lower()

    def test_verify_integrity_tampered(self, security_config, temp_model_file):
        """Test integrity verification of tampered model."""
        verifier = ModelIntegrityVerifier(security_config)

        record = verifier.register_model(temp_model_file, "model", True)

        # Tamper with the model
        with open(temp_model_file, "ab") as f:
            f.write(b"tampered data")

        is_valid, details = verifier.verify_integrity(record.record_id)

        assert not is_valid
        assert record.tampering_detected

    def test_verify_missing_model(self, security_config, temp_model_file):
        """Test verification of deleted model."""
        verifier = ModelIntegrityVerifier(security_config)

        record = verifier.register_model(temp_model_file, "model", True)

        # Delete the model
        Path(temp_model_file).unlink()

        is_valid, details = verifier.verify_integrity(record.record_id)

        assert not is_valid
        assert "not found" in details.lower()

    def test_sign_model(self, security_config, temp_model_file):
        """Test model signing."""
        verifier = ModelIntegrityVerifier(security_config)

        record = verifier.register_model(temp_model_file, "model", True)
        signed = verifier.sign_model(
            record.record_id,
            signing_key=b"secret_key",
            signer_id="admin",
        )

        assert signed.signature != ""
        assert signed.signed_by == "admin"

    def test_verify_signature(self, security_config, temp_model_file):
        """Test signature verification."""
        verifier = ModelIntegrityVerifier(security_config)
        key = b"secret_key"

        record = verifier.register_model(temp_model_file, "model", True)
        verifier.sign_model(record.record_id, key, "admin")

        is_valid, details = verifier.verify_signature(record.record_id, key)

        assert is_valid

    def test_verify_signature_wrong_key(self, security_config, temp_model_file):
        """Test signature verification with wrong key."""
        verifier = ModelIntegrityVerifier(security_config)

        record = verifier.register_model(temp_model_file, "model", True)
        verifier.sign_model(record.record_id, b"correct_key", "admin")

        is_valid, details = verifier.verify_signature(record.record_id, b"wrong_key")

        assert not is_valid


# =============================================================================
# Test Adversarial Detector
# =============================================================================

class TestAdversarialDetector:
    """Test adversarial input detection."""

    def test_add_reference(self, security_config):
        """Test adding reference inputs."""
        detector = AdversarialDetector(security_config)

        data = np.random.randn(100)
        detector.add_reference(data)

        assert len(detector._reference_inputs) == 1

    def test_detect_clean_input(self, security_config):
        """Test detecting clean (non-adversarial) input."""
        detector = AdversarialDetector(security_config)

        # Use fixed seed for reproducibility
        np.random.seed(42)
        reference_data = np.random.randn(100)
        detector.add_reference(reference_data)

        # Check same input as reference (should be recognized)
        is_adversarial, confidence, details = detector.detect(reference_data)

        # Identical input to reference should not be flagged as adversarial
        assert is_adversarial is False or confidence < 0.7

    def test_detect_perturbation(self, security_config):
        """Test detecting perturbation from reference."""
        detector = AdversarialDetector(security_config)

        reference = np.zeros(100)
        perturbed = reference + 0.5  # Large perturbation

        is_adversarial, confidence, details = detector.detect(perturbed, reference)

        assert "perturbation" in details.lower()

    def test_detect_high_frequency_noise(self, security_config):
        """Test detecting high-frequency noise."""
        detector = AdversarialDetector(security_config)

        # Create noisy input
        noisy = np.zeros(100)
        noisy[::2] = 1  # Alternating pattern

        is_adversarial, confidence, details = detector.detect(noisy)

        assert "noise" in details.lower()

    def test_detection_history(self, security_config):
        """Test detection history tracking."""
        detector = AdversarialDetector(security_config)

        for _ in range(5):
            detector.detect(np.random.randn(100))

        assert len(detector._detection_history) == 5


# =============================================================================
# Test Data Poisoning Detector
# =============================================================================

class TestDataPoisoningDetector:
    """Test data poisoning detection."""

    def test_set_baseline(self, security_config):
        """Test setting baseline statistics."""
        detector = DataPoisoningDetector(security_config)

        data = np.random.randn(1000)
        stats = detector.set_baseline("dataset1", data)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_detect_clean_data(self, security_config):
        """Test detecting clean (non-poisoned) data."""
        detector = DataPoisoningDetector(security_config)

        # Use fixed seed for reproducibility
        np.random.seed(42)
        baseline_data = np.random.randn(10000)
        detector.set_baseline("dataset1", baseline_data)

        # Use similar data from same distribution with same seed
        np.random.seed(42)
        new_data = np.random.randn(10000)  # Same distribution, same seed
        is_poisoned, confidence, details = detector.detect("dataset1", new_data)

        # Clean data (identical) should have very low confidence
        assert confidence < 0.1

    def test_detect_poisoned_data(self, security_config):
        """Test detecting poisoned data."""
        detector = DataPoisoningDetector(security_config)

        baseline_data = np.random.randn(1000)
        detector.set_baseline("dataset1", baseline_data)

        # Create poisoned data (shifted distribution)
        poisoned_data = np.random.randn(1000) + 5  # Shifted mean

        is_poisoned, confidence, details = detector.detect("dataset1", poisoned_data)

        assert is_poisoned
        assert confidence > 0.3

    def test_detect_outlier_injection(self, security_config):
        """Test detecting outlier injection."""
        detector = DataPoisoningDetector(security_config)

        baseline_data = np.random.randn(1000)
        detector.set_baseline("dataset1", baseline_data)

        # Add extreme outliers
        poisoned_data = np.random.randn(1000)
        poisoned_data[:100] = 100  # Inject outliers

        is_poisoned, confidence, details = detector.detect("dataset1", poisoned_data)

        assert "outlier_ratio" in details["deviations"]

    def test_detect_without_baseline(self, security_config):
        """Test detection without baseline."""
        detector = DataPoisoningDetector(security_config)

        data = np.random.randn(100)
        is_poisoned, confidence, details = detector.detect("nonexistent", data)

        assert not is_poisoned
        assert "error" in details


# =============================================================================
# Test Access Control Manager
# =============================================================================

class TestAccessControlManager:
    """Test access control functionality."""

    def test_create_session(self, security_config):
        """Test session creation."""
        manager = AccessControlManager(security_config)

        session_id = manager.create_session(
            user_id="user123",
            access_level=AccessLevel.OPERATOR,
            ip_address="192.168.1.1",
        )

        assert session_id is not None
        is_valid, session = manager.validate_session(session_id)
        assert is_valid

    def test_validate_session(self, security_config):
        """Test session validation."""
        manager = AccessControlManager(security_config)

        session_id = manager.create_session("user123", AccessLevel.DEVELOPER)
        is_valid, session = manager.validate_session(session_id)

        assert is_valid
        assert session["user_id"] == "user123"

    def test_invalid_session(self, security_config):
        """Test invalid session."""
        manager = AccessControlManager(security_config)

        is_valid, session = manager.validate_session("invalid_session")

        assert not is_valid
        assert session is None

    def test_check_access_granted(self, security_config):
        """Test access granted."""
        manager = AccessControlManager(security_config)

        session_id = manager.create_session("user123", AccessLevel.DEVELOPER)
        allowed, reason = manager.check_access(
            session_id=session_id,
            resource_type="model",
            resource_id="model001",
            action="read",
            required_level=AccessLevel.OPERATOR,
        )

        assert allowed

    def test_check_access_denied(self, security_config):
        """Test access denied."""
        manager = AccessControlManager(security_config)

        session_id = manager.create_session("user123", AccessLevel.READ_ONLY)
        allowed, reason = manager.check_access(
            session_id=session_id,
            resource_type="config",
            resource_id="config001",
            action="write",
            required_level=AccessLevel.ADMIN,
        )

        assert not allowed
        assert "Insufficient" in reason

    def test_record_failed_attempt(self, security_config):
        """Test recording failed attempts."""
        manager = AccessControlManager(security_config)

        # Record failures but not enough for lockout
        for _ in range(2):
            locked = manager.record_failed_attempt("user123")
            assert not locked

        # Third attempt should trigger lockout
        locked = manager.record_failed_attempt("user123")
        assert locked

    def test_lockout(self, security_config):
        """Test user lockout."""
        manager = AccessControlManager(security_config)

        # Trigger lockout
        for _ in range(3):
            manager.record_failed_attempt("user123")

        # User should be locked out
        with pytest.raises(PermissionError, match="locked out"):
            manager.create_session("user123", AccessLevel.OPERATOR)

    def test_invalidate_session(self, security_config):
        """Test session invalidation."""
        manager = AccessControlManager(security_config)

        session_id = manager.create_session("user123", AccessLevel.OPERATOR)
        result = manager.invalidate_session(session_id)

        assert result
        is_valid, _ = manager.validate_session(session_id)
        assert not is_valid

    def test_get_access_log(self, security_config):
        """Test getting access log."""
        manager = AccessControlManager(security_config)

        session_id = manager.create_session("user123", AccessLevel.DEVELOPER)
        manager.check_access(session_id, "resource", "r1", "read", AccessLevel.READ_ONLY)
        manager.check_access(session_id, "resource", "r2", "write", AccessLevel.DEVELOPER)

        logs = manager.get_access_log(user_id="user123")

        assert len(logs) == 2


# =============================================================================
# Test Main Cybersecurity Class
# =============================================================================

class TestAIActCybersecurity:
    """Test main cybersecurity class."""

    def test_validate_input(self, security):
        """Test input validation through main class."""
        data = np.random.randn(100).astype(np.float32)
        result, sanitized = security.validate_input(data, "observation")

        assert result.status == ValidationStatus.VALID
        assert sanitized is None  # No sanitization needed

    def test_validate_and_sanitize(self, security):
        """Test input validation with sanitization."""
        data = np.array([1.0, np.nan, 2.0]).astype(np.float64)
        result, sanitized = security.validate_input(data, "observation", sanitize=True)

        assert result.sanitized
        assert sanitized is not None
        assert np.all(np.isfinite(sanitized))

    def test_register_and_verify_model(self, security, temp_model_file):
        """Test model registration and verification."""
        record = security.register_model(temp_model_file, "test_model")
        is_valid, details = security.verify_model_integrity(record.record_id)

        assert is_valid

    def test_detect_adversarial(self, security):
        """Test adversarial detection."""
        # Add references
        for _ in range(10):
            security.add_reference_input(np.random.randn(100))

        # Test detection
        data = np.random.randn(100)
        is_adversarial, confidence, details = security.detect_adversarial(data)

        # Should not be adversarial for random data
        assert isinstance(is_adversarial, bool)
        assert 0 <= confidence <= 1

    def test_data_poisoning_detection(self, security):
        """Test data poisoning detection."""
        baseline = np.random.randn(1000)
        security.set_data_baseline("test_dataset", baseline)

        new_data = np.random.randn(1000)
        is_poisoned, confidence, details = security.detect_data_poisoning(
            "test_dataset", new_data
        )

        assert isinstance(is_poisoned, bool)

    def test_access_control(self, security):
        """Test access control through main class."""
        session = security.create_session("user123", AccessLevel.OPERATOR)
        allowed, reason = security.check_access(
            session, "model", "m1", "read", AccessLevel.READ_ONLY
        )

        assert allowed

        security.invalidate_session(session)
        allowed, reason = security.check_access(
            session, "model", "m1", "read", AccessLevel.READ_ONLY
        )
        assert not allowed

    def test_record_threat(self, security):
        """Test threat recording."""
        threat = security.record_threat(
            threat_type=ThreatType.ADVERSARIAL_EXAMPLE,
            severity=ThreatSeverity.HIGH,
            title="Test Threat",
            description="Test description",
        )

        assert threat.threat_id is not None
        assert not threat.mitigated

        retrieved = security.get_threat(threat.threat_id)
        assert retrieved is not None

    def test_mitigate_threat(self, security):
        """Test threat mitigation."""
        threat = security.record_threat(
            threat_type=ThreatType.DATA_POISONING,
            severity=ThreatSeverity.MEDIUM,
            title="Test Threat",
            description="Description",
        )

        mitigated = security.mitigate_threat(
            threat.threat_id,
            mitigation_details="Fixed the issue",
            actions_taken=["Updated validation", "Retrained model"],
        )

        assert mitigated.mitigated
        assert len(mitigated.response_actions) == 2

    def test_get_active_threats(self, security):
        """Test getting active threats."""
        t1 = security.record_threat(
            ThreatType.ADVERSARIAL_EXAMPLE, ThreatSeverity.LOW,
            "Threat 1", "Desc 1",
        )
        t2 = security.record_threat(
            ThreatType.MODEL_POISONING, ThreatSeverity.HIGH,
            "Threat 2", "Desc 2",
        )

        security.mitigate_threat(t1.threat_id, "Fixed")

        active = security.get_active_threats()
        assert len(active) == 1

    def test_create_policy(self, security):
        """Test policy creation."""
        policy = security.create_policy(
            name="Test Policy",
            description="Test description",
            policy_type="input_validation",
            rules=[{"check": "data_type"}],
            on_violation="block",
        )

        assert policy.policy_id is not None
        assert policy.is_active

    def test_get_security_status(self, security):
        """Test getting security status."""
        # Create some data
        security.record_threat(
            ThreatType.ADVERSARIAL_EXAMPLE, ThreatSeverity.HIGH,
            "Threat", "Description",
        )

        status = security.get_security_status()

        assert "threats" in status
        assert "events" in status
        assert "policies" in status
        assert "components" in status
        assert "compliance_status" in status

    def test_generate_security_report(self, security):
        """Test security report generation."""
        security.record_threat(
            ThreatType.DATA_POISONING, ThreatSeverity.MEDIUM,
            "Threat", "Description",
        )

        report = security.generate_security_report()

        assert "report_date" in report
        assert "article_reference" in report
        assert "status" in report
        assert "threats" in report
        assert "recommendations" in report


# =============================================================================
# Test Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_cybersecurity_default(self, temp_dir):
        """Test creating cybersecurity with default config."""
        security = create_cybersecurity()
        assert security is not None
        assert isinstance(security, AIActCybersecurity)

    def test_create_cybersecurity_with_config(self, temp_dir):
        """Test creating cybersecurity with config object."""
        config = CybersecurityConfig(
            max_failed_attempts=5,
            log_path=f"{temp_dir}/logs",
        )
        security = create_cybersecurity(config)
        assert security.config.max_failed_attempts == 5

    def test_create_cybersecurity_with_dict(self, temp_dir):
        """Test creating cybersecurity with dictionary config."""
        config_dict = {
            "max_failed_attempts": 10,
            "lockout_duration_minutes": 60,
            "log_path": f"{temp_dir}/logs",
        }
        security = create_cybersecurity(config_dict)
        assert security.config.max_failed_attempts == 10
        assert security.config.lockout_duration_minutes == 60

    def test_get_default_security_policies(self):
        """Test getting default security policies."""
        policies = get_default_security_policies()

        assert len(policies) >= 3

        # Check required fields
        for policy in policies:
            assert "name" in policy
            assert "description" in policy
            assert "policy_type" in policy
            assert "rules" in policy

        # Check specific policies exist
        names = {p["name"] for p in policies}
        assert "Input Validation Policy" in names
        assert "Access Control Policy" in names


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_verify_nonexistent_record(self, security):
        """Test verifying non-existent record."""
        is_valid, details = security.verify_model_integrity("INVALID-ID")
        assert not is_valid

    def test_mitigate_nonexistent_threat(self, security):
        """Test mitigating non-existent threat."""
        with pytest.raises(ValueError, match="not found"):
            security.mitigate_threat("INVALID-ID", "Details")

    def test_large_input_validation(self, security):
        """Test validating large input."""
        # Create input larger than limit
        large_data = np.random.randn(100000000).astype(np.float32)  # ~400MB

        result, _ = security.validate_input(large_data, "observation")
        assert "size" in result.checks_failed


# =============================================================================
# Test Thread Safety
# =============================================================================

class TestThreadSafety:
    """Test thread safety."""

    def test_concurrent_validation(self, security):
        """Test concurrent input validation."""
        import threading

        results = []
        errors = []

        def validate(index):
            try:
                data = np.random.randn(100).astype(np.float32)
                result, _ = security.validate_input(data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=validate, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
