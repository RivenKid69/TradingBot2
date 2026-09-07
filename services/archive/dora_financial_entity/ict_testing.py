# -*- coding: utf-8 -*-
"""
DORA ICT System Testing Module (Article 25).

Regulation (EU) 2022/2554 Article 25 defines requirements for testing
of ICT tools and systems:
    - Digital operational resilience testing programme applied to all ICT systems
    - Risk-based approach to testing
    - Independent parties for testing (internal or external)
    - Testing of critical ICT systems and applications at least yearly
    - Documented procedures for prioritizing, classifying, and remedying issues

Key Requirements per Article 25:
    1. Test all ICT systems supporting critical or important functions (Article 25(1))
    2. Risk-based approach considering evolving ICT risks (Article 25(1))
    3. Independent testing by qualified testers (Article 25(2))
    4. Testing at least yearly for systems supporting critical functions (Article 25(1))
    5. Documented procedures for assessment, classification, remediation (Article 25(3))
    6. Establish mechanisms to prioritize, classify, and remedy issues (Article 25(3))

References:
    - Article 25 DORA: https://www.digital-operational-resilience-act.com/Article_25.html
    - RTS on ICT Risk Management Framework (CDR 2024/1774)
    - Technical implementation standards
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class ICTSystemType(Enum):
    """Types of ICT systems per DORA."""
    TRADING_SYSTEM = "trading_system"
    MARKET_DATA = "market_data"
    ORDER_MANAGEMENT = "order_management"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    EXECUTION_MANAGEMENT = "execution_management"
    POST_TRADE = "post_trade"
    REPORTING = "reporting"
    COMPLIANCE = "compliance"
    AUTHENTICATION = "authentication"
    DATABASE = "database"
    MESSAGING = "messaging"
    API_GATEWAY = "api_gateway"
    NETWORK = "network"
    INFRASTRUCTURE = "infrastructure"
    THIRD_PARTY_INTERFACE = "third_party_interface"
    MONITORING = "monitoring"
    BACKUP = "backup"
    OTHER = "other"


class SystemCriticality(Enum):
    """System criticality levels per Article 25(1)."""
    CRITICAL = "critical"      # Supports critical functions
    IMPORTANT = "important"    # Supports important functions
    STANDARD = "standard"      # Standard business functions
    SUPPORT = "support"        # Support/administrative functions


class TestingPriority(Enum):
    """Testing priority based on risk assessment."""
    P1_IMMEDIATE = "P1"    # Test immediately
    P2_HIGH = "P2"         # Test within 1 month
    P3_MEDIUM = "P3"       # Test within quarter
    P4_LOW = "P4"          # Test within year
    P5_MINIMAL = "P5"      # As resources permit


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"     # CVSS 9.0-10.0
    HIGH = "high"             # CVSS 7.0-8.9
    MEDIUM = "medium"         # CVSS 4.0-6.9
    LOW = "low"               # CVSS 0.1-3.9
    INFORMATIONAL = "info"    # CVSS 0.0


class VulnerabilityStatus(Enum):
    """Vulnerability remediation status."""
    IDENTIFIED = "identified"
    CONFIRMED = "confirmed"
    IN_REMEDIATION = "in_remediation"
    REMEDIATED = "remediated"
    VERIFIED = "verified"
    ACCEPTED = "accepted"       # Risk accepted
    FALSE_POSITIVE = "false_positive"
    DEFERRED = "deferred"


class TestType(Enum):
    """Types of ICT system tests per Article 25."""
    VULNERABILITY_SCAN = "vulnerability_scan"
    PENETRATION_TEST = "penetration_test"
    CONFIGURATION_REVIEW = "configuration_review"
    CODE_REVIEW = "code_review"
    ARCHITECTURE_REVIEW = "architecture_review"
    INTERFACE_TEST = "interface_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    STRESS_TEST = "stress_test"
    FAILOVER_TEST = "failover_test"
    BACKUP_TEST = "backup_test"
    RECOVERY_TEST = "recovery_test"
    ACCESS_CONTROL_TEST = "access_control_test"
    DATA_INTEGRITY_TEST = "data_integrity_test"


class TestStatus(Enum):
    """Test execution status."""
    PLANNED = "planned"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TestResult(Enum):
    """Test result outcomes."""
    PASSED = "passed"
    PASSED_WITH_FINDINGS = "passed_with_findings"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class RemediationStatus(Enum):
    """Remediation plan status."""
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    CLOSED = "closed"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ICTSystemProfile:
    """
    ICT System profile for testing per Article 25(1).
    """
    system_id: str = ""
    system_name: str = ""
    system_type: ICTSystemType = ICTSystemType.OTHER
    description: str = ""

    # Classification
    criticality: SystemCriticality = SystemCriticality.STANDARD
    testing_priority: TestingPriority = TestingPriority.P3_MEDIUM

    # Functions supported
    business_functions: List[str] = field(default_factory=list)
    critical_functions: List[str] = field(default_factory=list)
    important_functions: List[str] = field(default_factory=list)

    # Technical details
    technology_stack: List[str] = field(default_factory=list)
    hosting_type: str = ""  # on_premise, cloud, hybrid
    environment: str = ""   # production, staging, development

    # Interfaces
    internal_interfaces: List[str] = field(default_factory=list)
    external_interfaces: List[str] = field(default_factory=list)
    third_party_providers: List[str] = field(default_factory=list)

    # Data
    data_types_processed: List[str] = field(default_factory=list)
    data_sensitivity: str = ""  # high, medium, low

    # Risk profile
    risk_score: float = 0.0
    last_risk_assessment: str = ""
    risk_factors: List[str] = field(default_factory=list)

    # Testing requirements
    minimum_test_frequency: str = "annual"  # annual, semi_annual, quarterly
    required_test_types: List[TestType] = field(default_factory=list)
    last_test_date: str = ""
    next_test_due: str = ""

    # Change tracking
    last_significant_change: str = ""
    change_history: List[Dict[str, Any]] = field(default_factory=list)

    # Owner and contacts
    system_owner: str = ""
    technical_contact: str = ""
    business_contact: str = ""

    # Status
    is_active: bool = True

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.system_id:
            self.system_id = f"SYS-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SystemTestPlan:
    """
    Test plan for an ICT system per Article 25(1).
    """
    plan_id: str = ""
    system_id: str = ""
    system_name: str = ""
    plan_name: str = ""
    description: str = ""

    # Planning
    planned_period_start: str = ""
    planned_period_end: str = ""
    planning_approach: str = "risk_based"  # risk_based, comprehensive, targeted

    # Tests included
    test_types: List[TestType] = field(default_factory=list)
    test_objectives: List[str] = field(default_factory=list)
    test_scope: List[str] = field(default_factory=list)

    # Resources
    estimated_effort_hours: float = 0.0
    estimated_cost_eur: float = 0.0
    assigned_testers: List[str] = field(default_factory=list)
    external_provider: str = ""

    # Prerequisites
    prerequisites: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    # Risk considerations
    risk_factors_addressed: List[str] = field(default_factory=list)

    # Approval
    prepared_by: str = ""
    approved_by: str = ""
    approval_date: str = ""

    # Status
    status: str = "draft"  # draft, approved, in_progress, completed

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = f"PLAN-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class SystemTest:
    """
    Individual system test execution per Article 25.
    """
    test_id: str = ""
    plan_id: str = ""
    system_id: str = ""

    # Test details
    test_type: TestType = TestType.VULNERABILITY_SCAN
    test_name: str = ""
    test_description: str = ""
    test_objectives: List[str] = field(default_factory=list)

    # Scope
    components_tested: List[str] = field(default_factory=list)
    interfaces_tested: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0

    # Schedule
    scheduled_start: str = ""
    scheduled_end: str = ""
    actual_start: str = ""
    actual_end: str = ""

    # Execution
    status: TestStatus = TestStatus.PLANNED
    result: TestResult = TestResult.INCONCLUSIVE

    # Testers
    tester_type: str = "internal"  # internal, external
    testers: List[str] = field(default_factory=list)
    tester_organization: str = ""

    # Methodology
    methodology: str = ""
    tools_used: List[str] = field(default_factory=list)
    test_cases_executed: int = 0
    test_cases_passed: int = 0
    test_cases_failed: int = 0

    # Findings
    vulnerabilities_found: int = 0
    critical_vulnerabilities: int = 0
    high_vulnerabilities: int = 0
    medium_vulnerabilities: int = 0
    low_vulnerabilities: int = 0

    # Results
    executive_summary: str = ""
    detailed_findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Evidence
    evidence_collected: List[str] = field(default_factory=list)
    report_path: str = ""

    # Approval
    reviewed_by: str = ""
    approved_by: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.test_id:
            self.test_id = f"TST-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class Vulnerability:
    """
    Vulnerability identified during testing per Article 25(3).
    """
    vulnerability_id: str = ""
    test_id: str = ""
    system_id: str = ""

    # Identification
    title: str = ""
    description: str = ""
    technical_details: str = ""

    # Classification
    severity: VulnerabilitySeverity = VulnerabilitySeverity.MEDIUM
    status: VulnerabilityStatus = VulnerabilityStatus.IDENTIFIED

    # Scoring
    cvss_score: Optional[float] = None
    cvss_vector: str = ""
    risk_rating: str = ""

    # References
    cve_ids: List[str] = field(default_factory=list)
    cwe_ids: List[str] = field(default_factory=list)

    # Affected components
    affected_components: List[str] = field(default_factory=list)
    affected_versions: List[str] = field(default_factory=list)
    affected_environments: List[str] = field(default_factory=list)

    # Impact
    confidentiality_impact: str = ""  # none, low, high
    integrity_impact: str = ""        # none, low, high
    availability_impact: str = ""     # none, low, high
    business_impact: str = ""

    # Exploitation
    exploit_available: bool = False
    exploit_complexity: str = ""      # low, high
    privileges_required: str = ""     # none, low, high
    user_interaction: str = ""        # none, required

    # Evidence
    evidence: str = ""
    proof_of_concept: str = ""
    screenshots: List[str] = field(default_factory=list)

    # Remediation
    remediation_recommendation: str = ""
    remediation_steps: List[str] = field(default_factory=list)
    remediation_complexity: str = ""  # low, medium, high
    remediation_deadline: str = ""
    remediation_owner: str = ""

    # Resolution
    resolution_details: str = ""
    resolved_at: str = ""
    resolved_by: str = ""
    verification_test_id: str = ""
    verified_at: str = ""
    verified_by: str = ""

    # Tracking
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""

    def __post_init__(self):
        if not self.vulnerability_id:
            self.vulnerability_id = f"VULN-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class RemediationPlan:
    """
    Remediation plan per Article 25(3).
    """
    plan_id: str = ""
    vulnerability_id: str = ""
    system_id: str = ""

    # Plan details
    title: str = ""
    description: str = ""

    # Remediation approach
    remediation_approach: str = ""  # patch, upgrade, configuration, compensating_control
    remediation_steps: List[Dict[str, Any]] = field(default_factory=list)

    # Resources
    estimated_effort_hours: float = 0.0
    assigned_to: List[str] = field(default_factory=list)
    required_resources: List[str] = field(default_factory=list)

    # Schedule
    planned_start: str = ""
    planned_completion: str = ""
    actual_start: str = ""
    actual_completion: str = ""

    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    change_request_id: str = ""

    # Impact
    implementation_impact: str = ""
    rollback_plan: str = ""

    # Approval
    prepared_by: str = ""
    approved_by: str = ""
    approval_date: str = ""

    # Status
    status: RemediationStatus = RemediationStatus.DRAFT

    # Verification
    verification_required: bool = True
    verification_steps: List[str] = field(default_factory=list)
    verified_by: str = ""
    verified_at: str = ""

    # Metadata
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = f"REM-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class ThirdPartyInterfaceTest:
    """
    Testing of third-party provider interfaces per Article 25.
    """
    test_id: str = ""
    system_id: str = ""
    provider_name: str = ""
    provider_type: str = ""  # exchange, data_provider, cloud, etc.

    # Interface details
    interface_type: str = ""  # API, FTP, messaging, etc.
    interface_protocol: str = ""
    interface_endpoints: List[str] = field(default_factory=list)

    # Test scope
    test_type: TestType = TestType.INTERFACE_TEST
    test_objectives: List[str] = field(default_factory=list)

    # Test cases
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    # Format: {"id": str, "name": str, "expected": str, "actual": str, "passed": bool}

    # Security tests
    authentication_tested: bool = False
    authorization_tested: bool = False
    encryption_tested: bool = False
    data_validation_tested: bool = False

    # Results
    status: TestStatus = TestStatus.PLANNED
    result: TestResult = TestResult.INCONCLUSIVE
    findings: List[Dict[str, Any]] = field(default_factory=list)

    # SLA validation
    sla_response_time_ms: float = 0.0
    sla_availability_percent: float = 0.0
    sla_met: bool = True

    # Execution
    executed_at: str = ""
    executed_by: str = ""

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.test_id:
            self.test_id = f"INT-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ICTTestingConfig:
    """Configuration for ICT System Testing."""
    # Testing frequencies (days)
    critical_system_test_frequency: int = 365    # At least yearly
    important_system_test_frequency: int = 365   # At least yearly
    standard_system_test_frequency: int = 365
    support_system_test_frequency: int = 730     # Every 2 years

    # Vulnerability SLA (days)
    critical_vuln_sla_days: int = 7
    high_vuln_sla_days: int = 14
    medium_vuln_sla_days: int = 30
    low_vuln_sla_days: int = 90

    # Risk scoring
    enable_risk_scoring: bool = True
    risk_score_weights: Dict[str, float] = field(default_factory=lambda: {
        "criticality": 0.3,
        "exposure": 0.2,
        "vulnerability_history": 0.2,
        "change_frequency": 0.15,
        "third_party_dependency": 0.15,
    })

    # Independence requirements
    require_independent_testing: bool = True
    external_testing_for_critical: bool = True

    # Reporting
    auto_generate_reports: bool = True
    report_retention_years: int = 5

    # Alerts
    alert_on_critical_vulnerability: bool = True
    alert_on_overdue_test: bool = True
    alert_threshold_days: int = 30

    # Logging
    log_path: str = "logs/dora/ict_testing"


# =============================================================================
# Main Class Implementation
# =============================================================================

class DORAICTSystemTesting:
    """
    DORA Article 25 compliant ICT System Testing.

    Provides:
    - ICT system inventory and profiling for testing
    - Risk-based test planning and prioritization
    - Vulnerability identification and tracking
    - Remediation management per Article 25(3)
    - Third-party interface testing
    - Compliance reporting

    Article 25 Requirements:
    1. Test all ICT systems supporting critical/important functions (Art. 25(1))
    2. Risk-based approach with consideration of evolving ICT risks (Art. 25(1))
    3. Independent testing by qualified parties (Art. 25(2))
    4. Annual testing minimum for critical systems (Art. 25(1))
    5. Documented procedures for prioritizing and remedying issues (Art. 25(3))

    Usage:
        config = ICTTestingConfig()
        ict_testing = DORAICTSystemTesting(config)

        # Register system for testing
        system = ict_testing.register_system(
            system_name="Trading Engine",
            system_type=ICTSystemType.TRADING_SYSTEM,
            criticality=SystemCriticality.CRITICAL,
            critical_functions=["order_execution", "market_data"],
        )

        # Create test plan
        plan = ict_testing.create_test_plan(
            system_id=system.system_id,
            test_types=[TestType.VULNERABILITY_SCAN, TestType.PENETRATION_TEST],
        )

        # Execute test and record vulnerabilities
        test = ict_testing.schedule_test(plan.plan_id, TestType.VULNERABILITY_SCAN)
    """

    def __init__(self, config: Optional[ICTTestingConfig] = None):
        """Initialize ICT System Testing."""
        self.config = config or ICTTestingConfig()

        # Data stores
        self._systems: Dict[str, ICTSystemProfile] = {}
        self._test_plans: Dict[str, SystemTestPlan] = {}
        self._tests: Dict[str, SystemTest] = {}
        self._vulnerabilities: Dict[str, Vulnerability] = {}
        self._remediation_plans: Dict[str, RemediationPlan] = {}
        self._interface_tests: Dict[str, ThirdPartyInterfaceTest] = {}

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        logger.info("DORAICTSystemTesting initialized")

    # =========================================================================
    # System Management
    # =========================================================================

    def register_system(
        self,
        system_name: str,
        system_type: ICTSystemType,
        criticality: SystemCriticality = SystemCriticality.STANDARD,
        description: str = "",
        business_functions: Optional[List[str]] = None,
        critical_functions: Optional[List[str]] = None,
        important_functions: Optional[List[str]] = None,
        technology_stack: Optional[List[str]] = None,
        hosting_type: str = "on_premise",
        data_types_processed: Optional[List[str]] = None,
        data_sensitivity: str = "medium",
        third_party_providers: Optional[List[str]] = None,
        system_owner: str = "",
        technical_contact: str = "",
    ) -> ICTSystemProfile:
        """
        Register an ICT system for testing per Article 25(1).

        Args:
            system_name: System name
            system_type: Type of ICT system
            criticality: System criticality level
            description: System description
            business_functions: Business functions supported
            critical_functions: Critical functions supported
            important_functions: Important functions supported
            technology_stack: Technology stack
            hosting_type: Hosting type
            data_types_processed: Data types processed
            data_sensitivity: Data sensitivity level
            third_party_providers: Third-party providers
            system_owner: System owner
            technical_contact: Technical contact

        Returns:
            Created ICTSystemProfile
        """
        system = ICTSystemProfile(
            system_name=system_name,
            system_type=system_type,
            criticality=criticality,
            description=description,
            business_functions=business_functions or [],
            critical_functions=critical_functions or [],
            important_functions=important_functions or [],
            technology_stack=technology_stack or [],
            hosting_type=hosting_type,
            environment="production",
            data_types_processed=data_types_processed or [],
            data_sensitivity=data_sensitivity,
            third_party_providers=third_party_providers or [],
            system_owner=system_owner,
            technical_contact=technical_contact,
        )

        # Calculate testing priority
        system.testing_priority = self._calculate_testing_priority(system)

        # Determine required test types based on criticality
        system.required_test_types = self._determine_required_tests(system)

        # Set minimum test frequency
        system.minimum_test_frequency = self._get_test_frequency(criticality)

        # Calculate next test due
        freq_days = self._frequency_to_days(system.minimum_test_frequency)
        system.next_test_due = (
            datetime.now(timezone.utc) + timedelta(days=freq_days)
        ).isoformat()

        # Calculate risk score
        if self.config.enable_risk_scoring:
            system.risk_score = self._calculate_risk_score(system)

        self._systems[system.system_id] = system

        self._log_event("system_registered", {
            "system_id": system.system_id,
            "system_name": system_name,
            "criticality": criticality.value,
        })

        logger.info(f"System registered: {system.system_id} - {system_name}")
        return system

    def _calculate_testing_priority(
        self,
        system: ICTSystemProfile,
    ) -> TestingPriority:
        """Calculate testing priority based on system profile."""
        if system.criticality == SystemCriticality.CRITICAL:
            return TestingPriority.P1_IMMEDIATE
        elif system.criticality == SystemCriticality.IMPORTANT:
            return TestingPriority.P2_HIGH
        elif system.criticality == SystemCriticality.STANDARD:
            return TestingPriority.P3_MEDIUM
        else:
            return TestingPriority.P4_LOW

    def _determine_required_tests(
        self,
        system: ICTSystemProfile,
    ) -> List[TestType]:
        """Determine required test types based on system profile."""
        required_tests = [TestType.VULNERABILITY_SCAN]

        if system.criticality in (SystemCriticality.CRITICAL, SystemCriticality.IMPORTANT):
            required_tests.extend([
                TestType.PENETRATION_TEST,
                TestType.CONFIGURATION_REVIEW,
                TestType.ACCESS_CONTROL_TEST,
            ])

        if system.criticality == SystemCriticality.CRITICAL:
            required_tests.extend([
                TestType.CODE_REVIEW,
                TestType.ARCHITECTURE_REVIEW,
                TestType.FAILOVER_TEST,
                TestType.RECOVERY_TEST,
            ])

        if system.third_party_providers:
            required_tests.append(TestType.INTERFACE_TEST)

        if system.data_sensitivity == "high":
            required_tests.append(TestType.DATA_INTEGRITY_TEST)

        return list(set(required_tests))

    def _get_test_frequency(self, criticality: SystemCriticality) -> str:
        """Get minimum test frequency based on criticality."""
        mapping = {
            SystemCriticality.CRITICAL: "annual",
            SystemCriticality.IMPORTANT: "annual",
            SystemCriticality.STANDARD: "annual",
            SystemCriticality.SUPPORT: "biennial",
        }
        return mapping.get(criticality, "annual")

    def _frequency_to_days(self, frequency: str) -> int:
        """Convert frequency string to days."""
        mapping = {
            "monthly": 30,
            "quarterly": 90,
            "semi_annual": 180,
            "annual": 365,
            "biennial": 730,
        }
        return mapping.get(frequency, 365)

    def _calculate_risk_score(
        self,
        system: ICTSystemProfile,
    ) -> float:
        """Calculate system risk score (0-100)."""
        weights = self.config.risk_score_weights
        score = 0.0

        # Criticality score
        criticality_scores = {
            SystemCriticality.CRITICAL: 100,
            SystemCriticality.IMPORTANT: 75,
            SystemCriticality.STANDARD: 50,
            SystemCriticality.SUPPORT: 25,
        }
        score += criticality_scores.get(system.criticality, 50) * weights.get("criticality", 0.3)

        # Exposure score (based on interfaces)
        interface_count = len(system.external_interfaces) + len(system.third_party_providers)
        exposure_score = min(100, interface_count * 20)
        score += exposure_score * weights.get("exposure", 0.2)

        # Third-party dependency score
        third_party_score = min(100, len(system.third_party_providers) * 25)
        score += third_party_score * weights.get("third_party_dependency", 0.15)

        # Data sensitivity score
        sensitivity_scores = {"high": 100, "medium": 60, "low": 30}
        score += sensitivity_scores.get(system.data_sensitivity, 50) * 0.15

        return round(score, 2)

    def get_system(self, system_id: str) -> Optional[ICTSystemProfile]:
        """Get system by ID."""
        return self._systems.get(system_id)

    def get_systems_by_criticality(
        self,
        criticality: SystemCriticality,
    ) -> List[ICTSystemProfile]:
        """Get systems by criticality level."""
        return [
            s for s in self._systems.values()
            if s.criticality == criticality and s.is_active
        ]

    def get_critical_systems(self) -> List[ICTSystemProfile]:
        """Get all critical systems requiring testing."""
        return self.get_systems_by_criticality(SystemCriticality.CRITICAL)

    def get_systems_requiring_testing(self) -> List[ICTSystemProfile]:
        """Get systems with overdue or upcoming tests."""
        now = datetime.now(timezone.utc)
        systems = []

        for system in self._systems.values():
            if not system.is_active:
                continue

            if system.next_test_due:
                due_date = datetime.fromisoformat(
                    system.next_test_due.replace("Z", "+00:00")
                )
                # Include if due within alert threshold
                threshold = now + timedelta(days=self.config.alert_threshold_days)
                if due_date <= threshold:
                    systems.append(system)
            else:
                # Never tested
                systems.append(system)

        return sorted(systems, key=lambda s: s.next_test_due or "")

    def update_system(
        self,
        system_id: str,
        **updates: Any,
    ) -> ICTSystemProfile:
        """Update system profile."""
        if system_id not in self._systems:
            raise ValueError(f"System {system_id} not found")

        system = self._systems[system_id]
        for key, value in updates.items():
            if hasattr(system, key):
                setattr(system, key, value)

        system.updated_at = datetime.now(timezone.utc).isoformat()

        # Recalculate priority and required tests
        system.testing_priority = self._calculate_testing_priority(system)
        system.required_test_types = self._determine_required_tests(system)

        return system

    # =========================================================================
    # Test Planning
    # =========================================================================

    def create_test_plan(
        self,
        system_id: str,
        plan_name: str = "",
        test_types: Optional[List[TestType]] = None,
        test_objectives: Optional[List[str]] = None,
        planned_period_start: Optional[str] = None,
        planned_period_end: Optional[str] = None,
        estimated_effort_hours: float = 0.0,
        assigned_testers: Optional[List[str]] = None,
        external_provider: str = "",
        prepared_by: str = "",
    ) -> SystemTestPlan:
        """
        Create a test plan for a system per Article 25.

        Args:
            system_id: System ID
            plan_name: Plan name
            test_types: Types of tests to include
            test_objectives: Test objectives
            planned_period_start: Start date
            planned_period_end: End date
            estimated_effort_hours: Estimated effort
            assigned_testers: Assigned testers
            external_provider: External provider
            prepared_by: Preparer

        Returns:
            Created SystemTestPlan
        """
        if system_id not in self._systems:
            raise ValueError(f"System {system_id} not found")

        system = self._systems[system_id]

        # Default to required tests if not specified
        if not test_types:
            test_types = system.required_test_types

        plan = SystemTestPlan(
            system_id=system_id,
            system_name=system.system_name,
            plan_name=plan_name or f"Test Plan for {system.system_name}",
            test_types=test_types,
            test_objectives=test_objectives or [],
            planned_period_start=planned_period_start or datetime.now(timezone.utc).isoformat(),
            planned_period_end=planned_period_end or "",
            estimated_effort_hours=estimated_effort_hours,
            assigned_testers=assigned_testers or [],
            external_provider=external_provider,
            prepared_by=prepared_by,
            risk_factors_addressed=system.risk_factors.copy(),
        )

        self._test_plans[plan.plan_id] = plan

        self._log_event("test_plan_created", {
            "plan_id": plan.plan_id,
            "system_id": system_id,
            "test_types": [t.value for t in test_types],
        })

        return plan

    def approve_test_plan(
        self,
        plan_id: str,
        approved_by: str,
    ) -> SystemTestPlan:
        """Approve a test plan."""
        if plan_id not in self._test_plans:
            raise ValueError(f"Plan {plan_id} not found")

        plan = self._test_plans[plan_id]
        plan.status = "approved"
        plan.approved_by = approved_by
        plan.approval_date = datetime.now(timezone.utc).isoformat()

        self._log_event("test_plan_approved", {
            "plan_id": plan_id,
            "approved_by": approved_by,
        })

        return plan

    def get_test_plan(self, plan_id: str) -> Optional[SystemTestPlan]:
        """Get test plan by ID."""
        return self._test_plans.get(plan_id)

    # =========================================================================
    # Test Execution
    # =========================================================================

    def schedule_test(
        self,
        plan_id: str,
        test_type: TestType,
        test_name: str = "",
        scheduled_start: Optional[str] = None,
        scheduled_end: Optional[str] = None,
        testers: Optional[List[str]] = None,
        tester_organization: str = "",
        methodology: str = "",
        tools: Optional[List[str]] = None,
    ) -> SystemTest:
        """
        Schedule a test execution.

        Args:
            plan_id: Test plan ID
            test_type: Type of test
            test_name: Test name
            scheduled_start: Scheduled start
            scheduled_end: Scheduled end
            testers: Assigned testers
            tester_organization: Tester organization
            methodology: Testing methodology
            tools: Testing tools

        Returns:
            Created SystemTest
        """
        if plan_id not in self._test_plans:
            raise ValueError(f"Plan {plan_id} not found")

        plan = self._test_plans[plan_id]

        test = SystemTest(
            plan_id=plan_id,
            system_id=plan.system_id,
            test_type=test_type,
            test_name=test_name or f"{test_type.value} for {plan.system_name}",
            scheduled_start=scheduled_start or datetime.now(timezone.utc).isoformat(),
            scheduled_end=scheduled_end or "",
            status=TestStatus.SCHEDULED,
            testers=testers or plan.assigned_testers,
            tester_organization=tester_organization or plan.external_provider,
            methodology=methodology,
            tools_used=tools or [],
        )

        # Set tester type based on configuration
        system = self._systems.get(plan.system_id)
        if system and system.criticality == SystemCriticality.CRITICAL and self.config.external_testing_for_critical:
            test.tester_type = "external"
        elif tester_organization:
            test.tester_type = "external"
        else:
            test.tester_type = "internal"

        self._tests[test.test_id] = test

        self._log_event("test_scheduled", {
            "test_id": test.test_id,
            "plan_id": plan_id,
            "test_type": test_type.value,
        })

        return test

    def start_test(
        self,
        test_id: str,
        testers: Optional[List[str]] = None,
    ) -> SystemTest:
        """Start test execution."""
        if test_id not in self._tests:
            raise ValueError(f"Test {test_id} not found")

        test = self._tests[test_id]
        test.status = TestStatus.IN_PROGRESS
        test.actual_start = datetime.now(timezone.utc).isoformat()

        if testers:
            test.testers = testers

        test.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("test_started", {
            "test_id": test_id,
            "started_at": test.actual_start,
        })

        return test

    def complete_test(
        self,
        test_id: str,
        result: TestResult,
        executive_summary: str = "",
        recommendations: Optional[List[str]] = None,
        test_cases_executed: int = 0,
        test_cases_passed: int = 0,
        test_cases_failed: int = 0,
        coverage_percentage: float = 100.0,
        report_path: str = "",
        reviewed_by: str = "",
        approved_by: str = "",
    ) -> SystemTest:
        """
        Complete a test execution.

        Args:
            test_id: Test ID
            result: Test result
            executive_summary: Executive summary
            recommendations: Recommendations
            test_cases_executed: Test cases executed
            test_cases_passed: Test cases passed
            test_cases_failed: Test cases failed
            coverage_percentage: Coverage percentage
            report_path: Path to report
            reviewed_by: Reviewer
            approved_by: Approver

        Returns:
            Updated SystemTest
        """
        if test_id not in self._tests:
            raise ValueError(f"Test {test_id} not found")

        test = self._tests[test_id]
        now = datetime.now(timezone.utc).isoformat()

        test.status = TestStatus.COMPLETED
        test.result = result
        test.actual_end = now
        test.executive_summary = executive_summary
        test.recommendations = recommendations or []
        test.test_cases_executed = test_cases_executed
        test.test_cases_passed = test_cases_passed
        test.test_cases_failed = test_cases_failed
        test.coverage_percentage = coverage_percentage
        test.report_path = report_path
        test.reviewed_by = reviewed_by
        test.approved_by = approved_by
        test.updated_at = now

        # Update vulnerability counts
        vulnerabilities = self.get_vulnerabilities_for_test(test_id)
        test.vulnerabilities_found = len(vulnerabilities)
        test.critical_vulnerabilities = sum(1 for v in vulnerabilities if v.severity == VulnerabilitySeverity.CRITICAL)
        test.high_vulnerabilities = sum(1 for v in vulnerabilities if v.severity == VulnerabilitySeverity.HIGH)
        test.medium_vulnerabilities = sum(1 for v in vulnerabilities if v.severity == VulnerabilitySeverity.MEDIUM)
        test.low_vulnerabilities = sum(1 for v in vulnerabilities if v.severity == VulnerabilitySeverity.LOW)

        # Update system last test date
        system = self._systems.get(test.system_id)
        if system:
            system.last_test_date = now
            # Calculate next test due
            freq_days = self._frequency_to_days(system.minimum_test_frequency)
            system.next_test_due = (
                datetime.now(timezone.utc) + timedelta(days=freq_days)
            ).isoformat()

        self._log_event("test_completed", {
            "test_id": test_id,
            "result": result.value,
            "vulnerabilities_found": test.vulnerabilities_found,
        })

        logger.info(f"Test completed: {test_id} - Result: {result.value}")
        return test

    def get_test(self, test_id: str) -> Optional[SystemTest]:
        """Get test by ID."""
        return self._tests.get(test_id)

    def get_tests_for_system(self, system_id: str) -> List[SystemTest]:
        """Get all tests for a system."""
        return [t for t in self._tests.values() if t.system_id == system_id]

    def get_tests_by_status(self, status: TestStatus) -> List[SystemTest]:
        """Get tests by status."""
        return [t for t in self._tests.values() if t.status == status]

    # =========================================================================
    # Vulnerability Management (Article 25(3))
    # =========================================================================

    def record_vulnerability(
        self,
        test_id: str,
        title: str,
        description: str,
        severity: VulnerabilitySeverity,
        affected_components: Optional[List[str]] = None,
        technical_details: str = "",
        cvss_score: Optional[float] = None,
        cvss_vector: str = "",
        cve_ids: Optional[List[str]] = None,
        cwe_ids: Optional[List[str]] = None,
        evidence: str = "",
        remediation_recommendation: str = "",
        remediation_steps: Optional[List[str]] = None,
        created_by: str = "",
    ) -> Vulnerability:
        """
        Record a vulnerability per Article 25(3).

        Args:
            test_id: Test ID
            title: Vulnerability title
            description: Description
            severity: Severity level
            affected_components: Affected components
            technical_details: Technical details
            cvss_score: CVSS score
            cvss_vector: CVSS vector
            cve_ids: CVE IDs
            cwe_ids: CWE IDs
            evidence: Evidence
            remediation_recommendation: Remediation recommendation
            remediation_steps: Remediation steps
            created_by: Creator

        Returns:
            Created Vulnerability
        """
        if test_id not in self._tests:
            raise ValueError(f"Test {test_id} not found")

        test = self._tests[test_id]

        vulnerability = Vulnerability(
            test_id=test_id,
            system_id=test.system_id,
            title=title,
            description=description,
            severity=severity,
            status=VulnerabilityStatus.IDENTIFIED,
            affected_components=affected_components or [],
            technical_details=technical_details,
            cvss_score=cvss_score,
            cvss_vector=cvss_vector,
            cve_ids=cve_ids or [],
            cwe_ids=cwe_ids or [],
            evidence=evidence,
            remediation_recommendation=remediation_recommendation,
            remediation_steps=remediation_steps or [],
            created_by=created_by,
        )

        # Calculate risk rating from CVSS if available
        if cvss_score is not None:
            vulnerability.risk_rating = self._cvss_to_risk_rating(cvss_score)

        # Set remediation deadline based on severity
        sla_days = self._get_vulnerability_sla(severity)
        if sla_days > 0:
            vulnerability.remediation_deadline = (
                datetime.now(timezone.utc) + timedelta(days=sla_days)
            ).isoformat()

        self._vulnerabilities[vulnerability.vulnerability_id] = vulnerability

        self._log_event("vulnerability_recorded", {
            "vulnerability_id": vulnerability.vulnerability_id,
            "test_id": test_id,
            "severity": severity.value,
            "title": title,
        })

        # Alert on critical vulnerabilities
        if severity == VulnerabilitySeverity.CRITICAL and self.config.alert_on_critical_vulnerability:
            logger.critical(f"CRITICAL vulnerability: {vulnerability.vulnerability_id} - {title}")

        return vulnerability

    def _cvss_to_risk_rating(self, cvss_score: float) -> str:
        """Convert CVSS score to risk rating."""
        if cvss_score >= 9.0:
            return "critical"
        elif cvss_score >= 7.0:
            return "high"
        elif cvss_score >= 4.0:
            return "medium"
        else:
            return "low"

    def _get_vulnerability_sla(self, severity: VulnerabilitySeverity) -> int:
        """Get SLA days for vulnerability severity."""
        sla_mapping = {
            VulnerabilitySeverity.CRITICAL: self.config.critical_vuln_sla_days,
            VulnerabilitySeverity.HIGH: self.config.high_vuln_sla_days,
            VulnerabilitySeverity.MEDIUM: self.config.medium_vuln_sla_days,
            VulnerabilitySeverity.LOW: self.config.low_vuln_sla_days,
            VulnerabilitySeverity.INFORMATIONAL: 0,
        }
        return sla_mapping.get(severity, 30)

    def update_vulnerability_status(
        self,
        vulnerability_id: str,
        status: VulnerabilityStatus,
        notes: str = "",
        updated_by: str = "",
    ) -> Vulnerability:
        """Update vulnerability status."""
        if vulnerability_id not in self._vulnerabilities:
            raise ValueError(f"Vulnerability {vulnerability_id} not found")

        vuln = self._vulnerabilities[vulnerability_id]
        vuln.status = status
        vuln.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("vulnerability_status_updated", {
            "vulnerability_id": vulnerability_id,
            "status": status.value,
            "updated_by": updated_by,
        })

        return vuln

    def resolve_vulnerability(
        self,
        vulnerability_id: str,
        resolution_details: str,
        resolved_by: str,
    ) -> Vulnerability:
        """Mark vulnerability as remediated."""
        if vulnerability_id not in self._vulnerabilities:
            raise ValueError(f"Vulnerability {vulnerability_id} not found")

        vuln = self._vulnerabilities[vulnerability_id]
        now = datetime.now(timezone.utc).isoformat()

        vuln.status = VulnerabilityStatus.REMEDIATED
        vuln.resolution_details = resolution_details
        vuln.resolved_at = now
        vuln.resolved_by = resolved_by
        vuln.updated_at = now

        self._log_event("vulnerability_resolved", {
            "vulnerability_id": vulnerability_id,
            "resolved_by": resolved_by,
        })

        return vuln

    def verify_vulnerability_fix(
        self,
        vulnerability_id: str,
        verification_test_id: str,
        verified_by: str,
    ) -> Vulnerability:
        """Verify that vulnerability fix is effective."""
        if vulnerability_id not in self._vulnerabilities:
            raise ValueError(f"Vulnerability {vulnerability_id} not found")

        vuln = self._vulnerabilities[vulnerability_id]
        now = datetime.now(timezone.utc).isoformat()

        vuln.status = VulnerabilityStatus.VERIFIED
        vuln.verification_test_id = verification_test_id
        vuln.verified_at = now
        vuln.verified_by = verified_by
        vuln.updated_at = now

        self._log_event("vulnerability_verified", {
            "vulnerability_id": vulnerability_id,
            "verification_test_id": verification_test_id,
            "verified_by": verified_by,
        })

        return vuln

    def get_vulnerability(self, vulnerability_id: str) -> Optional[Vulnerability]:
        """Get vulnerability by ID."""
        return self._vulnerabilities.get(vulnerability_id)

    def get_vulnerabilities_for_test(self, test_id: str) -> List[Vulnerability]:
        """Get all vulnerabilities for a test."""
        return [v for v in self._vulnerabilities.values() if v.test_id == test_id]

    def get_vulnerabilities_for_system(self, system_id: str) -> List[Vulnerability]:
        """Get all vulnerabilities for a system."""
        return [v for v in self._vulnerabilities.values() if v.system_id == system_id]

    def get_open_vulnerabilities(self) -> List[Vulnerability]:
        """Get all open vulnerabilities."""
        open_statuses = (
            VulnerabilityStatus.IDENTIFIED,
            VulnerabilityStatus.CONFIRMED,
            VulnerabilityStatus.IN_REMEDIATION,
        )
        return [v for v in self._vulnerabilities.values() if v.status in open_statuses]

    def get_vulnerabilities_by_severity(
        self,
        severity: VulnerabilitySeverity,
    ) -> List[Vulnerability]:
        """Get vulnerabilities by severity."""
        return [v for v in self._vulnerabilities.values() if v.severity == severity]

    def get_overdue_vulnerabilities(self) -> List[Vulnerability]:
        """Get vulnerabilities past their remediation deadline."""
        now = datetime.now(timezone.utc)
        overdue = []

        for vuln in self._vulnerabilities.values():
            if vuln.status in (VulnerabilityStatus.VERIFIED, VulnerabilityStatus.ACCEPTED,
                              VulnerabilityStatus.FALSE_POSITIVE):
                continue
            if not vuln.remediation_deadline:
                continue

            deadline = datetime.fromisoformat(
                vuln.remediation_deadline.replace("Z", "+00:00")
            )
            if now > deadline:
                overdue.append(vuln)

        return overdue

    # =========================================================================
    # Remediation Management (Article 25(3))
    # =========================================================================

    def create_remediation_plan(
        self,
        vulnerability_id: str,
        title: str,
        remediation_approach: str,
        remediation_steps: List[Dict[str, Any]],
        estimated_effort_hours: float = 0.0,
        assigned_to: Optional[List[str]] = None,
        planned_start: Optional[str] = None,
        planned_completion: Optional[str] = None,
        prepared_by: str = "",
    ) -> RemediationPlan:
        """
        Create a remediation plan per Article 25(3).

        Args:
            vulnerability_id: Vulnerability ID
            title: Plan title
            remediation_approach: Approach (patch, upgrade, etc.)
            remediation_steps: Remediation steps
            estimated_effort_hours: Estimated effort
            assigned_to: Assigned team members
            planned_start: Planned start date
            planned_completion: Planned completion date
            prepared_by: Preparer

        Returns:
            Created RemediationPlan
        """
        if vulnerability_id not in self._vulnerabilities:
            raise ValueError(f"Vulnerability {vulnerability_id} not found")

        vuln = self._vulnerabilities[vulnerability_id]

        plan = RemediationPlan(
            vulnerability_id=vulnerability_id,
            system_id=vuln.system_id,
            title=title,
            remediation_approach=remediation_approach,
            remediation_steps=remediation_steps,
            estimated_effort_hours=estimated_effort_hours,
            assigned_to=assigned_to or [],
            planned_start=planned_start or datetime.now(timezone.utc).isoformat(),
            planned_completion=planned_completion or vuln.remediation_deadline,
            prepared_by=prepared_by,
        )

        self._remediation_plans[plan.plan_id] = plan

        # Update vulnerability status
        vuln.status = VulnerabilityStatus.IN_REMEDIATION
        vuln.updated_at = datetime.now(timezone.utc).isoformat()

        self._log_event("remediation_plan_created", {
            "plan_id": plan.plan_id,
            "vulnerability_id": vulnerability_id,
            "approach": remediation_approach,
        })

        return plan

    def approve_remediation_plan(
        self,
        plan_id: str,
        approved_by: str,
    ) -> RemediationPlan:
        """Approve a remediation plan."""
        if plan_id not in self._remediation_plans:
            raise ValueError(f"Plan {plan_id} not found")

        plan = self._remediation_plans[plan_id]
        plan.status = RemediationStatus.APPROVED
        plan.approved_by = approved_by
        plan.approval_date = datetime.now(timezone.utc).isoformat()
        plan.updated_at = datetime.now(timezone.utc).isoformat()

        return plan

    def complete_remediation(
        self,
        plan_id: str,
        completion_notes: str = "",
    ) -> RemediationPlan:
        """Mark remediation as completed."""
        if plan_id not in self._remediation_plans:
            raise ValueError(f"Plan {plan_id} not found")

        plan = self._remediation_plans[plan_id]
        now = datetime.now(timezone.utc).isoformat()

        plan.status = RemediationStatus.COMPLETED
        plan.actual_completion = now
        plan.updated_at = now

        return plan

    def verify_remediation(
        self,
        plan_id: str,
        verified_by: str,
    ) -> RemediationPlan:
        """Verify remediation is effective."""
        if plan_id not in self._remediation_plans:
            raise ValueError(f"Plan {plan_id} not found")

        plan = self._remediation_plans[plan_id]
        now = datetime.now(timezone.utc).isoformat()

        plan.status = RemediationStatus.VERIFIED
        plan.verified_by = verified_by
        plan.verified_at = now
        plan.updated_at = now

        # Update associated vulnerability
        vuln = self._vulnerabilities.get(plan.vulnerability_id)
        if vuln:
            vuln.status = VulnerabilityStatus.VERIFIED
            vuln.verified_by = verified_by
            vuln.verified_at = now

        return plan

    def get_remediation_plan(self, plan_id: str) -> Optional[RemediationPlan]:
        """Get remediation plan by ID."""
        return self._remediation_plans.get(plan_id)

    # =========================================================================
    # Third-Party Interface Testing
    # =========================================================================

    def create_interface_test(
        self,
        system_id: str,
        provider_name: str,
        provider_type: str,
        interface_type: str = "API",
        interface_protocol: str = "REST",
        interface_endpoints: Optional[List[str]] = None,
        test_objectives: Optional[List[str]] = None,
    ) -> ThirdPartyInterfaceTest:
        """
        Create a third-party interface test.

        Args:
            system_id: System ID
            provider_name: Provider name
            provider_type: Provider type
            interface_type: Interface type
            interface_protocol: Protocol
            interface_endpoints: Endpoints to test
            test_objectives: Test objectives

        Returns:
            Created ThirdPartyInterfaceTest
        """
        if system_id not in self._systems:
            raise ValueError(f"System {system_id} not found")

        test = ThirdPartyInterfaceTest(
            system_id=system_id,
            provider_name=provider_name,
            provider_type=provider_type,
            interface_type=interface_type,
            interface_protocol=interface_protocol,
            interface_endpoints=interface_endpoints or [],
            test_objectives=test_objectives or [],
        )

        self._interface_tests[test.test_id] = test

        self._log_event("interface_test_created", {
            "test_id": test.test_id,
            "system_id": system_id,
            "provider_name": provider_name,
        })

        return test

    def execute_interface_test(
        self,
        test_id: str,
        test_cases: List[Dict[str, Any]],
        authentication_tested: bool = False,
        authorization_tested: bool = False,
        encryption_tested: bool = False,
        data_validation_tested: bool = False,
        executed_by: str = "",
    ) -> ThirdPartyInterfaceTest:
        """
        Execute interface test and record results.

        Args:
            test_id: Test ID
            test_cases: Test case results
            authentication_tested: Whether auth was tested
            authorization_tested: Whether authz was tested
            encryption_tested: Whether encryption was tested
            data_validation_tested: Whether data validation was tested
            executed_by: Executor

        Returns:
            Updated ThirdPartyInterfaceTest
        """
        if test_id not in self._interface_tests:
            raise ValueError(f"Test {test_id} not found")

        test = self._interface_tests[test_id]
        now = datetime.now(timezone.utc).isoformat()

        test.test_cases = test_cases
        test.authentication_tested = authentication_tested
        test.authorization_tested = authorization_tested
        test.encryption_tested = encryption_tested
        test.data_validation_tested = data_validation_tested
        test.executed_at = now
        test.executed_by = executed_by
        test.status = TestStatus.COMPLETED

        # Determine result
        passed_cases = sum(1 for tc in test_cases if tc.get("passed", False))
        total_cases = len(test_cases)

        if total_cases > 0:
            if passed_cases == total_cases:
                test.result = TestResult.PASSED
            elif passed_cases > 0:
                test.result = TestResult.PASSED_WITH_FINDINGS
            else:
                test.result = TestResult.FAILED
        else:
            test.result = TestResult.INCONCLUSIVE

        self._log_event("interface_test_executed", {
            "test_id": test_id,
            "result": test.result.value,
            "passed_cases": passed_cases,
            "total_cases": total_cases,
        })

        return test

    def get_interface_test(self, test_id: str) -> Optional[ThirdPartyInterfaceTest]:
        """Get interface test by ID."""
        return self._interface_tests.get(test_id)

    def get_interface_tests_for_system(
        self,
        system_id: str,
    ) -> List[ThirdPartyInterfaceTest]:
        """Get all interface tests for a system."""
        return [t for t in self._interface_tests.values() if t.system_id == system_id]

    # =========================================================================
    # Reporting and Compliance
    # =========================================================================

    def get_compliance_status(self) -> Dict[str, Any]:
        """
        Get Article 25 compliance status.

        Returns:
            Compliance status dictionary
        """
        now = datetime.now(timezone.utc)

        # Check systems with overdue tests
        systems_overdue = []
        systems_compliant = []

        for system in self._systems.values():
            if not system.is_active:
                continue

            if system.next_test_due:
                due_date = datetime.fromisoformat(
                    system.next_test_due.replace("Z", "+00:00")
                )
                if now > due_date:
                    systems_overdue.append(system.system_id)
                else:
                    systems_compliant.append(system.system_id)
            else:
                systems_overdue.append(system.system_id)

        # Critical systems
        critical_systems = self.get_critical_systems()
        critical_tested = [
            s for s in critical_systems
            if s.last_test_date and
            (now - datetime.fromisoformat(s.last_test_date.replace("Z", "+00:00"))).days <= 365
        ]

        # Vulnerability metrics
        open_vulns = self.get_open_vulnerabilities()
        overdue_vulns = self.get_overdue_vulnerabilities()

        return {
            "article_reference": "Article 25",
            "assessment_date": now.isoformat(),
            "system_testing": {
                "total_systems": len(self._systems),
                "active_systems": sum(1 for s in self._systems.values() if s.is_active),
                "systems_compliant": len(systems_compliant),
                "systems_overdue": len(systems_overdue),
                "compliance_rate": (
                    len(systems_compliant) / (len(systems_compliant) + len(systems_overdue)) * 100
                    if (systems_compliant or systems_overdue) else 100
                ),
            },
            "critical_systems": {
                "total": len(critical_systems),
                "tested_within_year": len(critical_tested),
                "compliance_rate": (
                    len(critical_tested) / len(critical_systems) * 100
                    if critical_systems else 100
                ),
            },
            "vulnerabilities": {
                "total_identified": len(self._vulnerabilities),
                "open": len(open_vulns),
                "overdue": len(overdue_vulns),
                "critical_open": len(self.get_vulnerabilities_by_severity(VulnerabilitySeverity.CRITICAL)),
                "high_open": len(self.get_vulnerabilities_by_severity(VulnerabilitySeverity.HIGH)),
            },
            "overall_compliant": len(systems_overdue) == 0 and len(overdue_vulns) == 0,
        }

    def generate_testing_report(
        self,
        system_id: Optional[str] = None,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate testing report per Article 25.

        Args:
            system_id: Optional system ID to filter
            period_start: Report period start
            period_end: Report period end

        Returns:
            Testing report dictionary
        """
        if not period_end:
            period_end = datetime.now(timezone.utc).isoformat()
        if not period_start:
            period_start = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()

        start_dt = datetime.fromisoformat(period_start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(period_end.replace("Z", "+00:00"))

        # Filter tests
        period_tests = [
            t for t in self._tests.values()
            if t.created_at and
            start_dt <= datetime.fromisoformat(t.created_at.replace("Z", "+00:00")) <= end_dt
        ]

        if system_id:
            period_tests = [t for t in period_tests if t.system_id == system_id]

        # Calculate metrics
        completed_tests = [t for t in period_tests if t.status == TestStatus.COMPLETED]
        passed_tests = [t for t in completed_tests if t.result in (TestResult.PASSED, TestResult.PASSED_WITH_FINDINGS)]

        # Aggregate vulnerabilities
        period_vulns = []
        for test in completed_tests:
            period_vulns.extend(self.get_vulnerabilities_for_test(test.test_id))

        return {
            "report_type": "ict_system_testing_report",
            "article_reference": "Article 25",
            "period_start": period_start,
            "period_end": period_end,
            "system_id": system_id,
            "report_date": datetime.now(timezone.utc).isoformat(),
            "testing_summary": {
                "total_tests": len(period_tests),
                "completed_tests": len(completed_tests),
                "passed_tests": len(passed_tests),
                "pass_rate": (len(passed_tests) / len(completed_tests) * 100) if completed_tests else 0,
                "tests_by_type": self._count_tests_by_type(completed_tests),
            },
            "vulnerability_summary": {
                "total_found": len(period_vulns),
                "by_severity": {
                    "critical": sum(1 for v in period_vulns if v.severity == VulnerabilitySeverity.CRITICAL),
                    "high": sum(1 for v in period_vulns if v.severity == VulnerabilitySeverity.HIGH),
                    "medium": sum(1 for v in period_vulns if v.severity == VulnerabilitySeverity.MEDIUM),
                    "low": sum(1 for v in period_vulns if v.severity == VulnerabilitySeverity.LOW),
                },
                "resolved": sum(1 for v in period_vulns if v.status == VulnerabilityStatus.VERIFIED),
            },
            "compliance_status": self.get_compliance_status(),
        }

    def _count_tests_by_type(self, tests: List[SystemTest]) -> Dict[str, int]:
        """Count tests by type."""
        counts = {}
        for test in tests:
            type_name = test.test_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def export_test_data(
        self,
        system_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export test data for regulatory purposes.

        Args:
            system_id: Optional system ID to filter

        Returns:
            Export data dictionary
        """
        if system_id:
            systems = [self._systems[system_id]] if system_id in self._systems else []
            tests = [t for t in self._tests.values() if t.system_id == system_id]
            vulns = [v for v in self._vulnerabilities.values() if v.system_id == system_id]
        else:
            systems = list(self._systems.values())
            tests = list(self._tests.values())
            vulns = list(self._vulnerabilities.values())

        return {
            "export_date": datetime.now(timezone.utc).isoformat(),
            "article_reference": "Article 25",
            "systems": [asdict(s) for s in systems],
            "tests": [asdict(t) for t in tests],
            "vulnerabilities": [asdict(v) for v in vulns],
            "remediation_plans": [asdict(p) for p in self._remediation_plans.values()],
        }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an ICT testing event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"ict_testing_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_ict_system_testing(
    config: Optional[ICTTestingConfig] = None,
) -> DORAICTSystemTesting:
    """
    Create a DORAICTSystemTesting instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAICTSystemTesting instance
    """
    return DORAICTSystemTesting(config=config)


def get_system_criticality_levels() -> List[SystemCriticality]:
    """
    Get list of system criticality levels.

    Returns:
        List of SystemCriticality enums
    """
    return list(SystemCriticality)


def get_vulnerability_severities() -> List[VulnerabilitySeverity]:
    """
    Get list of vulnerability severity levels.

    Returns:
        List of VulnerabilitySeverity enums
    """
    return list(VulnerabilitySeverity)


def get_test_types() -> List[TestType]:
    """
    Get list of test types per Article 25.

    Returns:
        List of TestType enums
    """
    return list(TestType)
