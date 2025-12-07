# -*- coding: utf-8 -*-
"""
AI Act Cybersecurity Module (Article 15(5)).

EU AI Act Article 15(5) requires high-risk AI systems to be resilient
against attempts by unauthorized third parties to exploit vulnerabilities.

This module provides:
1. InputValidator - Input validation and sanitization
2. ModelIntegrityVerifier - Model integrity verification
3. AdversarialDetector - Adversarial input detection
4. DataPoisoningDetector - Data poisoning detection
5. AIActCybersecurity - Main cybersecurity orchestrator
6. AccessControlManager - Access control and audit

Article 15(5) Requirements Implemented:
    - Resilience against unauthorized exploitation
    - Protection from data poisoning
    - Protection from model poisoning
    - Protection from adversarial examples
    - Protection from model flaws exploitation
    - Protection from confidentiality attacks

References:
    - Article 15: https://artificialintelligenceact.eu/article/15/
    - NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework
    - ENISA AI Threat Landscape: https://www.enisa.europa.eu/
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ThreatType(Enum):
    """Types of AI-specific security threats per Article 15(5)."""
    DATA_POISONING = "data_poisoning"
    MODEL_POISONING = "model_poisoning"
    ADVERSARIAL_EXAMPLE = "adversarial_example"
    MODEL_INVERSION = "model_inversion"
    MEMBERSHIP_INFERENCE = "membership_inference"
    MODEL_EXTRACTION = "model_extraction"
    INPUT_MANIPULATION = "input_manipulation"
    EVASION_ATTACK = "evasion_attack"
    TROJAN_ATTACK = "trojan_attack"
    SUPPLY_CHAIN = "supply_chain"
    UNAUTHORIZED_ACCESS = "unauthorized_access"


class ThreatSeverity(Enum):
    """Severity levels for security threats."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ValidationStatus(Enum):
    """Status of input validation."""
    VALID = "valid"
    SUSPICIOUS = "suspicious"
    INVALID = "invalid"
    BLOCKED = "blocked"


class AccessLevel(Enum):
    """Access control levels (higher value = more privileges)."""
    READ_ONLY = 1
    OPERATOR = 2
    DEVELOPER = 3
    ADMIN = 4


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    INTEGRITY_CHECK = "integrity_check"
    ADVERSARIAL_DETECTION = "adversarial_detection"
    ANOMALY_DETECTION = "anomaly_detection"
    ACCESS_CONTROL = "access_control"
    RATE_LIMITING = "rate_limiting"
    POLICY_VIOLATION = "policy_violation"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SecurityThreat:
    """
    Security threat record.

    Documents detected or potential security threats.
    """
    threat_id: str
    threat_type: ThreatType
    severity: ThreatSeverity
    title: str
    description: str

    # Detection details
    detection_method: str = ""
    detection_time: str = ""
    detection_source: str = ""

    # Evidence
    affected_components: List[str] = field(default_factory=list)
    indicators: Dict[str, Any] = field(default_factory=dict)
    evidence: str = ""

    # Impact
    potential_impact: str = ""
    affected_data: List[str] = field(default_factory=list)

    # Response
    response_actions: List[str] = field(default_factory=list)
    mitigated: bool = False
    mitigation_date: Optional[str] = None
    mitigation_details: str = ""

    # Status
    is_confirmed: bool = False
    is_false_positive: bool = False
    investigation_status: str = "open"  # open, investigating, resolved, closed

    # Metadata
    created_at: str = ""
    reported_by: str = ""

    def __post_init__(self):
        if not self.threat_id:
            self.threat_id = f"THREAT-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.detection_time:
            self.detection_time = datetime.now(timezone.utc).isoformat()


@dataclass
class ValidationResult:
    """
    Input validation result.

    Documents the result of input validation.
    """
    validation_id: str
    status: ValidationStatus
    input_type: str

    # Details
    checks_performed: List[str] = field(default_factory=list)
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)

    # Risk assessment
    risk_score: float = 0.0
    anomaly_score: float = 0.0

    # Actions taken
    sanitized: bool = False
    sanitization_details: str = ""
    blocked: bool = False
    blocking_reason: str = ""

    # Metadata
    timestamp: str = ""
    processing_time_ms: float = 0.0

    def __post_init__(self):
        if not self.validation_id:
            self.validation_id = f"VAL-{uuid.uuid4().hex[:8].upper()}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class IntegrityRecord:
    """
    Model/data integrity record.

    Documents integrity verification of models and data.
    """
    record_id: str
    resource_type: str  # model, weights, data, config
    resource_path: str
    resource_name: str

    # Hash values
    hash_algorithm: str = "sha256"
    expected_hash: str = ""
    actual_hash: str = ""

    # Signature (if applicable)
    signature: str = ""
    signature_valid: Optional[bool] = None
    signed_by: str = ""

    # Verification
    is_verified: bool = False
    verification_time: str = ""
    verification_result: str = ""

    # Tampering detection
    tampering_detected: bool = False
    tampering_details: str = ""

    # Metadata
    created_at: str = ""
    last_verified: str = ""

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"INT-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class AccessRecord:
    """
    Access control record.

    Documents access to protected resources.
    """
    access_id: str
    user_id: str
    resource_type: str
    resource_id: str
    access_level: AccessLevel

    # Access details
    action: str = ""  # read, write, execute, delete
    access_time: str = ""
    ip_address: str = ""
    user_agent: str = ""

    # Authorization
    authorized: bool = True
    authorization_reason: str = ""

    # Session
    session_id: str = ""
    session_start: str = ""

    # Audit
    success: bool = True
    failure_reason: str = ""

    def __post_init__(self):
        if not self.access_id:
            self.access_id = f"ACC-{uuid.uuid4().hex[:8].upper()}"
        if not self.access_time:
            self.access_time = datetime.now(timezone.utc).isoformat()


@dataclass
class SecurityPolicy:
    """
    Security policy definition.

    Defines security rules and constraints.
    """
    policy_id: str
    name: str
    description: str
    policy_type: str  # input_validation, access_control, rate_limiting

    # Rules
    rules: List[Dict[str, Any]] = field(default_factory=list)

    # Scope
    applies_to: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)

    # Actions
    on_violation: str = "block"  # block, warn, log
    alert_on_violation: bool = True

    # Status
    is_active: bool = True
    effective_date: str = ""
    expiry_date: Optional[str] = None

    # Metadata
    created_at: str = ""
    created_by: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = f"POL-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SecurityEvent:
    """
    Security event record.

    Documents security-related events for audit.
    """
    event_id: str
    event_type: SecurityEventType
    severity: ThreatSeverity
    description: str

    # Context
    user_id: str = ""
    resource_id: str = ""
    action: str = ""

    # Details
    success: bool = True
    failure_reason: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)

    # Source
    source_ip: str = ""
    source_component: str = ""

    # Timestamps
    timestamp: str = ""

    # Correlation
    correlation_id: str = ""
    session_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CybersecurityConfig:
    """
    Configuration for AI Act Cybersecurity module.
    """
    # Input validation
    enable_input_validation: bool = True
    max_input_size_mb: float = 100.0
    allowed_input_types: List[str] = field(default_factory=lambda: ["float32", "float64", "int32", "int64"])
    input_range_min: float = -1e10
    input_range_max: float = 1e10

    # Anomaly detection
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 3.0  # Standard deviations
    statistical_window_size: int = 1000

    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 1000
    max_requests_per_hour: int = 10000

    # Model integrity
    enable_integrity_checks: bool = True
    integrity_check_interval_hours: int = 24
    hash_algorithm: str = "sha256"

    # Access control
    enable_access_control: bool = True
    session_timeout_minutes: int = 60
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30

    # Adversarial detection
    enable_adversarial_detection: bool = True
    perturbation_threshold: float = 0.1

    # Logging
    log_all_events: bool = True
    log_path: str = "logs/ai_act/security"

    # Alerting
    alert_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None


# =============================================================================
# Input Validator
# =============================================================================

class InputValidator:
    """
    Input validation and sanitization.

    Validates and sanitizes inputs to protect against:
    - Adversarial examples
    - Input manipulation
    - Data injection
    """

    def __init__(self, config: CybersecurityConfig):
        """Initialize input validator."""
        self.config = config
        self._statistics: Dict[str, List[float]] = {}
        self._lock = threading.RLock()

    def validate(
        self,
        data: np.ndarray,
        input_type: str = "observation",
    ) -> ValidationResult:
        """
        Validate input data.

        Args:
            data: Input data array
            input_type: Type of input

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(
            validation_id="",
            status=ValidationStatus.VALID,
            input_type=input_type,
        )

        checks_passed = []
        checks_failed = []

        # Check 1: Data type
        result.checks_performed.append("data_type")
        if str(data.dtype) in self.config.allowed_input_types:
            checks_passed.append("data_type")
        else:
            checks_failed.append("data_type")

        # Check 2: Size
        result.checks_performed.append("size")
        size_mb = data.nbytes / (1024 * 1024)
        if size_mb <= self.config.max_input_size_mb:
            checks_passed.append("size")
        else:
            checks_failed.append("size")

        # Check 3: Range
        result.checks_performed.append("range")
        if np.all(data >= self.config.input_range_min) and np.all(data <= self.config.input_range_max):
            checks_passed.append("range")
        else:
            checks_failed.append("range")

        # Check 4: NaN/Inf
        result.checks_performed.append("finite")
        if np.all(np.isfinite(data)):
            checks_passed.append("finite")
        else:
            checks_failed.append("finite")

        # Check 5: Statistical anomaly
        if self.config.enable_anomaly_detection:
            result.checks_performed.append("anomaly")
            anomaly_score = self._check_statistical_anomaly(data, input_type)
            result.anomaly_score = anomaly_score
            if anomaly_score <= self.config.anomaly_threshold:
                checks_passed.append("anomaly")
            else:
                checks_failed.append("anomaly")

        result.checks_passed = checks_passed
        result.checks_failed = checks_failed

        # Determine status
        if not checks_failed:
            result.status = ValidationStatus.VALID
            result.risk_score = 0.0
        elif len(checks_failed) == 1 and "anomaly" in checks_failed:
            result.status = ValidationStatus.SUSPICIOUS
            result.risk_score = 0.5
        else:
            result.status = ValidationStatus.INVALID
            result.risk_score = 1.0
            result.blocked = True
            result.blocking_reason = f"Failed checks: {', '.join(checks_failed)}"

        return result

    def sanitize(
        self,
        data: np.ndarray,
        clip_range: bool = True,
        replace_nan: bool = True,
    ) -> Tuple[np.ndarray, str]:
        """
        Sanitize input data.

        Args:
            data: Input data array
            clip_range: Whether to clip to valid range
            replace_nan: Whether to replace NaN/Inf values

        Returns:
            Tuple of (sanitized data, sanitization details)
        """
        sanitized = data.copy()
        details = []

        if replace_nan:
            nan_count = np.sum(~np.isfinite(sanitized))
            if nan_count > 0:
                sanitized = np.nan_to_num(sanitized, nan=0.0, posinf=0.0, neginf=0.0)
                details.append(f"Replaced {nan_count} non-finite values")

        if clip_range:
            clipped = np.clip(
                sanitized,
                self.config.input_range_min,
                self.config.input_range_max,
            )
            clip_count = np.sum(sanitized != clipped)
            if clip_count > 0:
                sanitized = clipped
                details.append(f"Clipped {clip_count} out-of-range values")

        return sanitized, "; ".join(details) if details else "No sanitization needed"

    def _check_statistical_anomaly(
        self,
        data: np.ndarray,
        input_type: str,
    ) -> float:
        """Check for statistical anomalies in input."""
        with self._lock:
            if input_type not in self._statistics:
                self._statistics[input_type] = []

            stats = self._statistics[input_type]

            # Compute data statistics
            data_mean = float(np.mean(data))
            data_std = float(np.std(data))

            # If not enough history, just store and return 0
            if len(stats) < 10:
                stats.append(data_mean)
                if len(stats) > self.config.statistical_window_size:
                    stats.pop(0)
                return 0.0

            # Compute historical statistics
            hist_mean = np.mean(stats)
            hist_std = np.std(stats) + 1e-8  # Avoid division by zero

            # Z-score of current data mean
            z_score = abs((data_mean - hist_mean) / hist_std)

            # Update history
            stats.append(data_mean)
            if len(stats) > self.config.statistical_window_size:
                stats.pop(0)

            return float(z_score)


# =============================================================================
# Model Integrity Verifier
# =============================================================================

class ModelIntegrityVerifier:
    """
    Model integrity verification.

    Verifies that models have not been tampered with.
    """

    def __init__(self, config: CybersecurityConfig):
        """Initialize integrity verifier."""
        self.config = config
        self._records: Dict[str, IntegrityRecord] = {}
        self._lock = threading.RLock()

    def register_model(
        self,
        model_path: str,
        model_name: str,
        compute_hash: bool = True,
    ) -> IntegrityRecord:
        """
        Register a model for integrity monitoring.

        Args:
            model_path: Path to model file
            model_name: Model name
            compute_hash: Whether to compute hash now

        Returns:
            IntegrityRecord for the model
        """
        record = IntegrityRecord(
            record_id="",
            resource_type="model",
            resource_path=model_path,
            resource_name=model_name,
            hash_algorithm=self.config.hash_algorithm,
        )

        if compute_hash and Path(model_path).exists():
            record.expected_hash = self._compute_file_hash(model_path)

        with self._lock:
            self._records[record.record_id] = record

        return record

    def verify_integrity(
        self,
        record_id: str,
    ) -> Tuple[bool, str]:
        """
        Verify model integrity.

        Args:
            record_id: ID of the integrity record

        Returns:
            Tuple of (is_valid, details)
        """
        with self._lock:
            if record_id not in self._records:
                return False, f"Record {record_id} not found"

            record = self._records[record_id]

        if not Path(record.resource_path).exists():
            record.is_verified = False
            record.verification_result = "File not found"
            record.tampering_detected = True
            return False, "Model file not found"

        # Compute current hash
        current_hash = self._compute_file_hash(record.resource_path)
        record.actual_hash = current_hash
        record.verification_time = datetime.now(timezone.utc).isoformat()
        record.last_verified = record.verification_time

        if current_hash == record.expected_hash:
            record.is_verified = True
            record.verification_result = "Integrity verified"
            record.tampering_detected = False
            return True, "Model integrity verified"
        else:
            record.is_verified = False
            record.verification_result = "Hash mismatch"
            record.tampering_detected = True
            record.tampering_details = f"Expected: {record.expected_hash[:16]}..., Got: {current_hash[:16]}..."
            return False, "Model integrity check failed - possible tampering"

    def sign_model(
        self,
        record_id: str,
        signing_key: bytes,
        signer_id: str,
    ) -> IntegrityRecord:
        """
        Sign a model for additional integrity verification.

        Args:
            record_id: ID of the integrity record
            signing_key: Key for HMAC signature
            signer_id: ID of the signer

        Returns:
            Updated IntegrityRecord
        """
        with self._lock:
            if record_id not in self._records:
                raise ValueError(f"Record {record_id} not found")

            record = self._records[record_id]

        if not record.expected_hash:
            raise ValueError("Model hash must be computed before signing")

        # Create HMAC signature
        signature = hmac.new(
            signing_key,
            record.expected_hash.encode(),
            hashlib.sha256,
        ).hexdigest()

        record.signature = signature
        record.signed_by = signer_id

        return record

    def verify_signature(
        self,
        record_id: str,
        signing_key: bytes,
    ) -> Tuple[bool, str]:
        """
        Verify model signature.

        Args:
            record_id: ID of the integrity record
            signing_key: Key for HMAC verification

        Returns:
            Tuple of (is_valid, details)
        """
        with self._lock:
            if record_id not in self._records:
                return False, f"Record {record_id} not found"

            record = self._records[record_id]

        if not record.signature:
            return False, "Model is not signed"

        # Verify HMAC
        expected_sig = hmac.new(
            signing_key,
            record.expected_hash.encode(),
            hashlib.sha256,
        ).hexdigest()

        is_valid = hmac.compare_digest(record.signature, expected_sig)
        record.signature_valid = is_valid

        if is_valid:
            return True, "Signature verified"
        else:
            return False, "Signature verification failed"

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash of a file."""
        hash_obj = hashlib.new(self.config.hash_algorithm)

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def get_record(self, record_id: str) -> Optional[IntegrityRecord]:
        """Get integrity record by ID."""
        with self._lock:
            return self._records.get(record_id)


# =============================================================================
# Adversarial Detector
# =============================================================================

class AdversarialDetector:
    """
    Adversarial input detection.

    Detects potential adversarial examples designed to fool the model.
    """

    def __init__(self, config: CybersecurityConfig):
        """Initialize adversarial detector."""
        self.config = config
        self._reference_inputs: List[np.ndarray] = []
        self._detection_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()

    def add_reference(self, data: np.ndarray) -> None:
        """Add reference input for comparison."""
        with self._lock:
            self._reference_inputs.append(data.copy())
            # Keep only recent references
            if len(self._reference_inputs) > 100:
                self._reference_inputs.pop(0)

    def detect(
        self,
        data: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float, str]:
        """
        Detect potential adversarial input.

        Args:
            data: Input to check
            reference: Optional reference input for comparison

        Returns:
            Tuple of (is_adversarial, confidence, details)
        """
        results = []

        # Check 1: High-frequency noise detection
        noise_score = self._detect_high_frequency_noise(data)
        results.append(("noise", noise_score))

        # Check 2: Perturbation magnitude
        if reference is not None:
            pert_score = self._detect_perturbation(data, reference)
            results.append(("perturbation", pert_score))

        # Check 3: Statistical anomaly
        anomaly_score = self._detect_statistical_anomaly(data)
        results.append(("anomaly", anomaly_score))

        # Check 4: Distribution shift
        shift_score = self._detect_distribution_shift(data)
        results.append(("distribution", shift_score))

        # Aggregate scores
        max_score = max(score for _, score in results)
        avg_score = sum(score for _, score in results) / len(results)

        # Determine if adversarial
        threshold = self.config.perturbation_threshold
        is_adversarial = max_score > threshold or avg_score > threshold * 0.7

        # Build details
        details = ", ".join([f"{name}:{score:.3f}" for name, score in results])

        # Store detection
        with self._lock:
            self._detection_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "is_adversarial": is_adversarial,
                "confidence": max_score,
                "details": details,
            })
            if len(self._detection_history) > 1000:
                self._detection_history.pop(0)

        return is_adversarial, max_score, details

    def _detect_high_frequency_noise(self, data: np.ndarray) -> float:
        """Detect high-frequency noise patterns."""
        if data.ndim == 1:
            # 1D: check consecutive differences
            diff = np.diff(data)
            high_freq_ratio = np.mean(np.abs(diff) > np.std(data) * 2)
            return float(high_freq_ratio)
        else:
            # Multi-dimensional: check gradient magnitude
            gradients = np.gradient(data.flatten())
            high_freq_ratio = np.mean(np.abs(gradients) > np.std(data) * 2)
            return float(high_freq_ratio)

    def _detect_perturbation(self, data: np.ndarray, reference: np.ndarray) -> float:
        """Detect perturbation from reference."""
        if data.shape != reference.shape:
            return 0.0

        # L2 distance normalized by input magnitude
        l2_dist = np.linalg.norm(data - reference)
        ref_norm = np.linalg.norm(reference) + 1e-8
        relative_perturbation = l2_dist / ref_norm

        return float(min(1.0, relative_perturbation))

    def _detect_statistical_anomaly(self, data: np.ndarray) -> float:
        """Detect statistical anomalies."""
        # Check for unusual kurtosis (heavy tails)
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        kurtosis = np.mean(((data - mean) / std) ** 4) - 3

        # High kurtosis can indicate adversarial perturbations
        return float(min(1.0, abs(kurtosis) / 10))

    def _detect_distribution_shift(self, data: np.ndarray) -> float:
        """Detect distribution shift from references."""
        with self._lock:
            if not self._reference_inputs:
                return 0.0

            # Compare to reference distribution
            data_mean = np.mean(data)
            data_std = np.std(data)

            ref_means = [np.mean(r) for r in self._reference_inputs]
            ref_stds = [np.std(r) for r in self._reference_inputs]

            mean_shift = abs(data_mean - np.mean(ref_means)) / (np.std(ref_means) + 1e-8)
            std_shift = abs(data_std - np.mean(ref_stds)) / (np.std(ref_stds) + 1e-8)

            return float(min(1.0, (mean_shift + std_shift) / 4))


# =============================================================================
# Data Poisoning Detector
# =============================================================================

class DataPoisoningDetector:
    """
    Data poisoning detection.

    Detects potential data poisoning attempts.
    """

    def __init__(self, config: CybersecurityConfig):
        """Initialize data poisoning detector."""
        self.config = config
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        self._lock = threading.RLock()

    def set_baseline(
        self,
        dataset_id: str,
        data: np.ndarray,
    ) -> Dict[str, float]:
        """
        Set baseline statistics for a dataset.

        Args:
            dataset_id: Dataset identifier
            data: Baseline data

        Returns:
            Computed baseline statistics
        """
        stats = {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q1": float(np.percentile(data, 25)),
            "q3": float(np.percentile(data, 75)),
            "size": int(data.size),
        }

        with self._lock:
            self._baseline_stats[dataset_id] = stats

        return stats

    def detect(
        self,
        dataset_id: str,
        new_data: np.ndarray,
        threshold: float = 0.3,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect potential data poisoning.

        Args:
            dataset_id: Dataset identifier
            new_data: New data to check
            threshold: Detection threshold

        Returns:
            Tuple of (is_poisoned, confidence, details)
        """
        with self._lock:
            if dataset_id not in self._baseline_stats:
                return False, 0.0, {"error": "No baseline set"}

            baseline = self._baseline_stats[dataset_id]

        # Compute new data statistics
        new_stats = {
            "mean": float(np.mean(new_data)),
            "std": float(np.std(new_data)),
            "min": float(np.min(new_data)),
            "max": float(np.max(new_data)),
            "median": float(np.median(new_data)),
        }

        # Compare statistics
        deviations = {}
        for key in ["mean", "std", "median"]:
            if key in new_stats and key in baseline:
                baseline_val = baseline[key]
                new_val = new_stats[key]
                if abs(baseline_val) > 1e-8:
                    deviation = abs(new_val - baseline_val) / abs(baseline_val)
                else:
                    deviation = abs(new_val - baseline_val)
                deviations[key] = deviation

        # Check for outlier injection
        baseline_iqr = baseline["q3"] - baseline["q1"]
        outlier_threshold = baseline["q3"] + 1.5 * baseline_iqr
        outlier_ratio = np.mean(new_data > outlier_threshold)
        deviations["outlier_ratio"] = float(outlier_ratio)

        # Aggregate score
        max_deviation = max(deviations.values())
        avg_deviation = sum(deviations.values()) / len(deviations)

        is_poisoned = max_deviation > threshold or avg_deviation > threshold * 0.5
        confidence = min(1.0, max_deviation)

        return is_poisoned, confidence, {
            "deviations": deviations,
            "baseline_stats": baseline,
            "new_stats": new_stats,
        }


# =============================================================================
# Access Control Manager
# =============================================================================

class AccessControlManager:
    """
    Access control and audit management.

    Manages access to protected resources.
    """

    def __init__(self, config: CybersecurityConfig):
        """Initialize access control manager."""
        self.config = config
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._access_log: List[AccessRecord] = []
        self._failed_attempts: Dict[str, List[str]] = {}  # user_id -> timestamps
        self._locked_users: Dict[str, str] = {}  # user_id -> lockout_end
        self._lock = threading.RLock()

    def create_session(
        self,
        user_id: str,
        access_level: AccessLevel,
        ip_address: str = "",
    ) -> str:
        """
        Create a new session.

        Args:
            user_id: User identifier
            access_level: Access level
            ip_address: Client IP address

        Returns:
            Session ID
        """
        # Check if user is locked out
        if self._is_locked_out(user_id):
            raise PermissionError(f"User {user_id} is locked out")

        session_id = secrets.token_urlsafe(32)
        session_start = datetime.now(timezone.utc)
        session_expiry = session_start + timedelta(minutes=self.config.session_timeout_minutes)

        with self._lock:
            self._sessions[session_id] = {
                "user_id": user_id,
                "access_level": access_level,
                "ip_address": ip_address,
                "created_at": session_start.isoformat(),
                "expires_at": session_expiry.isoformat(),
                "is_active": True,
            }

        return session_id

    def validate_session(self, session_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate a session.

        Args:
            session_id: Session to validate

        Returns:
            Tuple of (is_valid, session_data)
        """
        with self._lock:
            if session_id not in self._sessions:
                return False, None

            session = self._sessions[session_id]

            if not session["is_active"]:
                return False, None

            expiry = datetime.fromisoformat(session["expires_at"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > expiry:
                session["is_active"] = False
                return False, None

            return True, session

    def check_access(
        self,
        session_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        required_level: AccessLevel = AccessLevel.READ_ONLY,
    ) -> Tuple[bool, str]:
        """
        Check if access is allowed.

        Args:
            session_id: Session ID
            resource_type: Type of resource
            resource_id: Resource identifier
            action: Action to perform
            required_level: Minimum required access level

        Returns:
            Tuple of (is_allowed, reason)
        """
        is_valid, session = self.validate_session(session_id)

        if not is_valid:
            return False, "Invalid or expired session"

        user_level = session["access_level"]

        # Check access level
        if user_level.value < required_level.value:
            self._log_access(
                session=session,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                authorized=False,
                reason="Insufficient access level",
            )
            return False, "Insufficient access level"

        # Access granted
        self._log_access(
            session=session,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            authorized=True,
        )
        return True, "Access granted"

    def record_failed_attempt(self, user_id: str) -> bool:
        """
        Record a failed access attempt.

        Args:
            user_id: User identifier

        Returns:
            True if user is now locked out
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            if user_id not in self._failed_attempts:
                self._failed_attempts[user_id] = []

            self._failed_attempts[user_id].append(now)

            # Clean old attempts
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            self._failed_attempts[user_id] = [
                t for t in self._failed_attempts[user_id]
                if t > cutoff
            ]

            # Check if lockout needed
            if len(self._failed_attempts[user_id]) >= self.config.max_failed_attempts:
                lockout_end = (
                    datetime.now(timezone.utc) +
                    timedelta(minutes=self.config.lockout_duration_minutes)
                ).isoformat()
                self._locked_users[user_id] = lockout_end
                return True

        return False

    def _is_locked_out(self, user_id: str) -> bool:
        """Check if user is locked out."""
        with self._lock:
            if user_id not in self._locked_users:
                return False

            lockout_end = datetime.fromisoformat(
                self._locked_users[user_id].replace("Z", "+00:00")
            )
            if datetime.now(timezone.utc) > lockout_end:
                del self._locked_users[user_id]
                return False

            return True

    def _log_access(
        self,
        session: Dict[str, Any],
        resource_type: str,
        resource_id: str,
        action: str,
        authorized: bool,
        reason: str = "",
    ) -> None:
        """Log access attempt."""
        record = AccessRecord(
            access_id="",
            user_id=session["user_id"],
            resource_type=resource_type,
            resource_id=resource_id,
            access_level=session["access_level"],
            action=action,
            ip_address=session.get("ip_address", ""),
            authorized=authorized,
            authorization_reason=reason,
            session_id=session.get("session_id", ""),
            session_start=session.get("created_at", ""),
            success=authorized,
            failure_reason=reason if not authorized else "",
        )

        with self._lock:
            self._access_log.append(record)
            if len(self._access_log) > 10000:
                self._access_log.pop(0)

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["is_active"] = False
                return True
            return False

    def get_access_log(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AccessRecord]:
        """Get access log entries."""
        with self._lock:
            if user_id:
                logs = [r for r in self._access_log if r.user_id == user_id]
            else:
                logs = list(self._access_log)

            return logs[-limit:]


# =============================================================================
# Main Cybersecurity Class
# =============================================================================

class AIActCybersecurity:
    """
    AI Act compliant Cybersecurity system per Article 15(5).

    Provides comprehensive cybersecurity functionality including:
    - Input validation and sanitization
    - Model integrity verification
    - Adversarial input detection
    - Data poisoning detection
    - Access control
    - Security event logging

    Usage:
        config = CybersecurityConfig(
            enable_input_validation=True,
            enable_adversarial_detection=True,
        )
        security = AIActCybersecurity(config)

        # Validate input
        result = security.validate_input(observation)

        # Check model integrity
        is_valid, details = security.verify_model_integrity(model_path)

        # Detect adversarial inputs
        is_adversarial, score, details = security.detect_adversarial(input_data)
    """

    def __init__(self, config: Optional[CybersecurityConfig] = None):
        """Initialize cybersecurity system."""
        self.config = config or CybersecurityConfig()

        # Components
        self._input_validator = InputValidator(self.config)
        self._integrity_verifier = ModelIntegrityVerifier(self.config)
        self._adversarial_detector = AdversarialDetector(self.config)
        self._poisoning_detector = DataPoisoningDetector(self.config)
        self._access_manager = AccessControlManager(self.config)

        # Threat tracking
        self._threats: Dict[str, SecurityThreat] = {}
        self._events: List[SecurityEvent] = []
        self._policies: Dict[str, SecurityPolicy] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("AIActCybersecurity initialized")

    # =========================================================================
    # Input Validation
    # =========================================================================

    def validate_input(
        self,
        data: np.ndarray,
        input_type: str = "observation",
        sanitize: bool = True,
    ) -> Tuple[ValidationResult, Optional[np.ndarray]]:
        """
        Validate and optionally sanitize input.

        Args:
            data: Input data
            input_type: Type of input
            sanitize: Whether to sanitize invalid inputs

        Returns:
            Tuple of (validation result, sanitized data or None)
        """
        result = self._input_validator.validate(data, input_type)

        sanitized_data = None
        if result.status != ValidationStatus.VALID and sanitize:
            sanitized_data, details = self._input_validator.sanitize(data)
            result.sanitized = True
            result.sanitization_details = details

        # Log event
        self._log_security_event(
            event_type=SecurityEventType.INPUT_VALIDATION,
            severity=ThreatSeverity.LOW if result.status == ValidationStatus.VALID else ThreatSeverity.MEDIUM,
            description=f"Input validation: {result.status.value}",
            success=result.status == ValidationStatus.VALID,
            additional_data={
                "input_type": input_type,
                "checks_failed": result.checks_failed,
            },
        )

        return result, sanitized_data

    # =========================================================================
    # Model Integrity
    # =========================================================================

    def register_model(
        self,
        model_path: str,
        model_name: str,
    ) -> IntegrityRecord:
        """Register a model for integrity monitoring."""
        return self._integrity_verifier.register_model(model_path, model_name)

    def verify_model_integrity(
        self,
        record_id: str,
    ) -> Tuple[bool, str]:
        """Verify model integrity."""
        is_valid, details = self._integrity_verifier.verify_integrity(record_id)

        # Log event
        self._log_security_event(
            event_type=SecurityEventType.INTEGRITY_CHECK,
            severity=ThreatSeverity.LOW if is_valid else ThreatSeverity.CRITICAL,
            description=f"Model integrity check: {details}",
            success=is_valid,
            additional_data={"record_id": record_id},
        )

        if not is_valid:
            # Record threat
            self.record_threat(
                threat_type=ThreatType.MODEL_POISONING,
                severity=ThreatSeverity.CRITICAL,
                title="Model Integrity Violation",
                description=details,
            )

        return is_valid, details

    # =========================================================================
    # Adversarial Detection
    # =========================================================================

    def detect_adversarial(
        self,
        data: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> Tuple[bool, float, str]:
        """Detect adversarial inputs."""
        is_adversarial, confidence, details = self._adversarial_detector.detect(data, reference)

        # Log event
        self._log_security_event(
            event_type=SecurityEventType.ADVERSARIAL_DETECTION,
            severity=ThreatSeverity.HIGH if is_adversarial else ThreatSeverity.LOW,
            description=f"Adversarial detection: {'DETECTED' if is_adversarial else 'clean'}",
            success=not is_adversarial,
            additional_data={
                "confidence": confidence,
                "details": details,
            },
        )

        if is_adversarial:
            self.record_threat(
                threat_type=ThreatType.ADVERSARIAL_EXAMPLE,
                severity=ThreatSeverity.HIGH,
                title="Adversarial Input Detected",
                description=f"Confidence: {confidence:.2f}, Details: {details}",
            )

        return is_adversarial, confidence, details

    def add_reference_input(self, data: np.ndarray) -> None:
        """Add reference input for adversarial detection."""
        self._adversarial_detector.add_reference(data)

    # =========================================================================
    # Data Poisoning Detection
    # =========================================================================

    def set_data_baseline(
        self,
        dataset_id: str,
        data: np.ndarray,
    ) -> Dict[str, float]:
        """Set baseline for data poisoning detection."""
        return self._poisoning_detector.set_baseline(dataset_id, data)

    def detect_data_poisoning(
        self,
        dataset_id: str,
        new_data: np.ndarray,
        threshold: float = 0.3,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect data poisoning."""
        is_poisoned, confidence, details = self._poisoning_detector.detect(
            dataset_id, new_data, threshold
        )

        self._log_security_event(
            event_type=SecurityEventType.ANOMALY_DETECTION,
            severity=ThreatSeverity.CRITICAL if is_poisoned else ThreatSeverity.LOW,
            description=f"Data poisoning detection: {'DETECTED' if is_poisoned else 'clean'}",
            success=not is_poisoned,
            additional_data=details,
        )

        if is_poisoned:
            self.record_threat(
                threat_type=ThreatType.DATA_POISONING,
                severity=ThreatSeverity.CRITICAL,
                title="Data Poisoning Detected",
                description=f"Dataset: {dataset_id}, Confidence: {confidence:.2f}",
            )

        return is_poisoned, confidence, details

    # =========================================================================
    # Access Control
    # =========================================================================

    def create_session(
        self,
        user_id: str,
        access_level: AccessLevel,
        ip_address: str = "",
    ) -> str:
        """Create access session."""
        return self._access_manager.create_session(user_id, access_level, ip_address)

    def check_access(
        self,
        session_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        required_level: AccessLevel = AccessLevel.READ_ONLY,
    ) -> Tuple[bool, str]:
        """Check access permission."""
        return self._access_manager.check_access(
            session_id, resource_type, resource_id, action, required_level
        )

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session."""
        return self._access_manager.invalidate_session(session_id)

    # =========================================================================
    # Threat Management
    # =========================================================================

    def record_threat(
        self,
        threat_type: ThreatType,
        severity: ThreatSeverity,
        title: str,
        description: str,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> SecurityThreat:
        """Record a security threat."""
        threat = SecurityThreat(
            threat_id="",
            threat_type=threat_type,
            severity=severity,
            title=title,
            description=description,
            indicators=indicators or {},
        )

        with self._lock:
            self._threats[threat.threat_id] = threat

        self._log_event("threat_recorded", asdict(threat))

        # Alert if configured
        if self.config.alert_callback and severity.value >= ThreatSeverity.HIGH.value:
            try:
                self.config.alert_callback("security_threat", asdict(threat))
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Security threat recorded: {threat.threat_id} - {title}")
        return threat

    def get_threat(self, threat_id: str) -> Optional[SecurityThreat]:
        """Get threat by ID."""
        with self._lock:
            return self._threats.get(threat_id)

    def get_active_threats(self) -> List[SecurityThreat]:
        """Get all active (non-mitigated) threats."""
        with self._lock:
            return [t for t in self._threats.values() if not t.mitigated]

    def mitigate_threat(
        self,
        threat_id: str,
        mitigation_details: str,
        actions_taken: Optional[List[str]] = None,
    ) -> SecurityThreat:
        """Mark a threat as mitigated."""
        with self._lock:
            if threat_id not in self._threats:
                raise ValueError(f"Threat {threat_id} not found")

            threat = self._threats[threat_id]
            threat.mitigated = True
            threat.mitigation_date = datetime.now(timezone.utc).isoformat()
            threat.mitigation_details = mitigation_details
            threat.response_actions = actions_taken or []
            threat.investigation_status = "resolved"

        self._log_event("threat_mitigated", {
            "threat_id": threat_id,
            "mitigation_details": mitigation_details,
        })
        return threat

    # =========================================================================
    # Policy Management
    # =========================================================================

    def create_policy(
        self,
        name: str,
        description: str,
        policy_type: str,
        rules: Optional[List[Dict[str, Any]]] = None,
        applies_to: Optional[List[str]] = None,
        on_violation: str = "block",
    ) -> SecurityPolicy:
        """Create a security policy."""
        policy = SecurityPolicy(
            policy_id="",
            name=name,
            description=description,
            policy_type=policy_type,
            rules=rules or [],
            applies_to=applies_to or [],
            on_violation=on_violation,
        )

        with self._lock:
            self._policies[policy.policy_id] = policy

        return policy

    def get_policy(self, policy_id: str) -> Optional[SecurityPolicy]:
        """Get policy by ID."""
        with self._lock:
            return self._policies.get(policy_id)

    def get_active_policies(self) -> List[SecurityPolicy]:
        """Get all active policies."""
        with self._lock:
            return [p for p in self._policies.values() if p.is_active]

    # =========================================================================
    # Compliance Reporting
    # =========================================================================

    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status."""
        with self._lock:
            active_threats = self.get_active_threats()
            critical_threats = [t for t in active_threats if t.severity == ThreatSeverity.CRITICAL]
            high_threats = [t for t in active_threats if t.severity == ThreatSeverity.HIGH]

            recent_events = self._events[-100:] if self._events else []
            failed_events = [e for e in recent_events if not e.success]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threats": {
                "total": len(self._threats),
                "active": len(active_threats),
                "critical": len(critical_threats),
                "high": len(high_threats),
            },
            "events": {
                "total_recent": len(recent_events),
                "failed_recent": len(failed_events),
            },
            "policies": {
                "total": len(self._policies),
                "active": len(self.get_active_policies()),
            },
            "components": {
                "input_validation": self.config.enable_input_validation,
                "anomaly_detection": self.config.enable_anomaly_detection,
                "adversarial_detection": self.config.enable_adversarial_detection,
                "integrity_checks": self.config.enable_integrity_checks,
                "access_control": self.config.enable_access_control,
                "rate_limiting": self.config.enable_rate_limiting,
            },
            "compliance_status": (
                "compliant" if not critical_threats else "non-compliant"
            ),
        }

    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        status = self.get_security_status()

        with self._lock:
            return {
                "report_date": datetime.now(timezone.utc).isoformat(),
                "article_reference": "Article 15(5)",
                "status": status,
                "threats": [asdict(t) for t in list(self._threats.values())[-50:]],
                "policies": [asdict(p) for p in self._policies.values()],
                "recent_events": [asdict(e) for e in self._events[-100:]],
                "recommendations": self._generate_recommendations(status),
            }

    def _generate_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []

        if status["threats"]["critical"] > 0:
            recommendations.append("Immediately address critical security threats")

        if not status["components"]["adversarial_detection"]:
            recommendations.append("Enable adversarial input detection")

        if not status["components"]["integrity_checks"]:
            recommendations.append("Enable model integrity verification")

        if status["events"]["failed_recent"] > 10:
            recommendations.append("Investigate high rate of failed security events")

        return recommendations

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_security_event(
        self,
        event_type: SecurityEventType,
        severity: ThreatSeverity,
        description: str,
        success: bool = True,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a security event."""
        event = SecurityEvent(
            event_id="",
            event_type=event_type,
            severity=severity,
            description=description,
            success=success,
            additional_data=additional_data or {},
        )

        with self._lock:
            self._events.append(event)
            if len(self._events) > 10000:
                self._events.pop(0)

        if self.config.log_all_events:
            self._log_event("security_event", asdict(event))

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log event to file."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"security_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_cybersecurity(
    config: Optional[Union[Dict[str, Any], CybersecurityConfig]] = None,
) -> AIActCybersecurity:
    """
    Create an AIActCybersecurity instance.

    Args:
        config: Optional configuration

    Returns:
        Configured AIActCybersecurity instance
    """
    if config is None:
        security_config = CybersecurityConfig()
    elif isinstance(config, CybersecurityConfig):
        security_config = config
    else:
        security_config = CybersecurityConfig(
            enable_input_validation=config.get("enable_input_validation", True),
            max_input_size_mb=config.get("max_input_size_mb", 100.0),
            input_range_min=config.get("input_range_min", -1e10),
            input_range_max=config.get("input_range_max", 1e10),
            enable_anomaly_detection=config.get("enable_anomaly_detection", True),
            anomaly_threshold=config.get("anomaly_threshold", 3.0),
            enable_rate_limiting=config.get("enable_rate_limiting", True),
            max_requests_per_minute=config.get("max_requests_per_minute", 1000),
            enable_integrity_checks=config.get("enable_integrity_checks", True),
            integrity_check_interval_hours=config.get("integrity_check_interval_hours", 24),
            enable_access_control=config.get("enable_access_control", True),
            session_timeout_minutes=config.get("session_timeout_minutes", 60),
            max_failed_attempts=config.get("max_failed_attempts", 5),
            lockout_duration_minutes=config.get("lockout_duration_minutes", 30),
            enable_adversarial_detection=config.get("enable_adversarial_detection", True),
            perturbation_threshold=config.get("perturbation_threshold", 0.1),
            log_all_events=config.get("log_all_events", True),
            log_path=config.get("log_path", "logs/ai_act/security"),
        )

    return AIActCybersecurity(security_config)


def get_default_security_policies() -> List[Dict[str, Any]]:
    """
    Get default security policies.

    Returns:
        List of policy definitions
    """
    return [
        {
            "name": "Input Validation Policy",
            "description": "Validate all inputs before processing",
            "policy_type": "input_validation",
            "rules": [
                {"check": "data_type", "action": "block"},
                {"check": "size", "action": "block"},
                {"check": "range", "action": "sanitize"},
                {"check": "finite", "action": "sanitize"},
            ],
            "on_violation": "block",
        },
        {
            "name": "Model Integrity Policy",
            "description": "Verify model integrity before loading",
            "policy_type": "integrity",
            "rules": [
                {"check": "hash", "action": "block"},
                {"check": "signature", "action": "warn"},
            ],
            "on_violation": "block",
        },
        {
            "name": "Access Control Policy",
            "description": "Enforce access control on protected resources",
            "policy_type": "access_control",
            "rules": [
                {"resource": "model", "min_level": "operator"},
                {"resource": "config", "min_level": "admin"},
                {"resource": "logs", "min_level": "read_only"},
            ],
            "on_violation": "block",
        },
        {
            "name": "Rate Limiting Policy",
            "description": "Limit request rates to prevent abuse",
            "policy_type": "rate_limiting",
            "rules": [
                {"max_per_minute": 1000},
                {"max_per_hour": 10000},
            ],
            "on_violation": "block",
        },
    ]
