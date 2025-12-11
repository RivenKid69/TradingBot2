# -*- coding: utf-8 -*-
"""
Comprehensive Tests for DORA Phase 3 - ICT System Testing (Article 25).

Tests for Testing of ICT Tools and Systems per Article 25 of DORA.

Coverage includes:
- ICT System Type enumeration
- System Criticality classification
- Testing Priority management
- Vulnerability Severity and Status
- Test Type categorization
- ICT System Profile management
- System Test Plan creation
- System Test execution
- Vulnerability management
- Remediation planning
- Third-Party Interface testing

References:
- Article 25 DORA: https://www.digital-operational-resilience-act.com/Article_25.html
- RTS on ICT Risk Management Framework (CDR 2024/1774)
"""


import pytest
pytest.skip(
    "Legacy DORA test - uses deprecated imports from services.dora.* "
    "These modules have been migrated to services.dora_integration.*. "
    "See tests/dora/ and tests/dora_integration/ for current tests.",
    allow_module_level=True
)


import pytest
from datetime import datetime, timedelta, timezone

from services.dora.ict_testing import (
    ICTSystemType,
    SystemCriticality,
    TestingPriority,
    VulnerabilitySeverity,
    VulnerabilityStatus,
    TestType,
    TestStatus,
    TestResult,
    RemediationStatus,
    ICTSystemProfile,
    SystemTestPlan,
    SystemTest,
    Vulnerability,
    RemediationPlan,
    ThirdPartyInterfaceTest,
    ICTTestingConfig,
    DORAICTSystemTesting,
    create_ict_system_testing,
)


# =============================================================================
# ICT System Type Tests
# =============================================================================

class TestICTSystemType:
    """Tests for ICT system type enumeration."""

    def test_trading_system_type(self):
        """Test trading system type."""
        assert ICTSystemType.TRADING_SYSTEM.value == "trading_system"

    def test_market_data_type(self):
        """Test market data system type."""
        assert ICTSystemType.MARKET_DATA.value == "market_data"

    def test_order_management_type(self):
        """Test order management system type."""
        assert ICTSystemType.ORDER_MANAGEMENT.value == "order_management"

    def test_risk_management_type(self):
        """Test risk management system type."""
        assert ICTSystemType.RISK_MANAGEMENT.value == "risk_management"

    def test_portfolio_management_type(self):
        """Test portfolio management system type."""
        assert ICTSystemType.PORTFOLIO_MANAGEMENT.value == "portfolio_management"

    def test_execution_management_type(self):
        """Test execution management system type."""
        assert ICTSystemType.EXECUTION_MANAGEMENT.value == "execution_management"

    def test_authentication_type(self):
        """Test authentication system type."""
        assert ICTSystemType.AUTHENTICATION.value == "authentication"

    def test_database_type(self):
        """Test database system type."""
        assert ICTSystemType.DATABASE.value == "database"

    def test_api_gateway_type(self):
        """Test API gateway system type."""
        assert ICTSystemType.API_GATEWAY.value == "api_gateway"

    def test_third_party_interface_type(self):
        """Test third-party interface type."""
        assert ICTSystemType.THIRD_PARTY_INTERFACE.value == "third_party_interface"


# =============================================================================
# System Criticality Tests (Article 25(1))
# =============================================================================

class TestSystemCriticality:
    """Tests for system criticality per Article 25(1)."""

    def test_critical_level(self):
        """Test critical level (supports critical functions)."""
        assert SystemCriticality.CRITICAL.value == "critical"

    def test_important_level(self):
        """Test important level (supports important functions)."""
        assert SystemCriticality.IMPORTANT.value == "important"

    def test_standard_level(self):
        """Test standard level (standard business functions)."""
        assert SystemCriticality.STANDARD.value == "standard"

    def test_support_level(self):
        """Test support level (support/administrative functions)."""
        assert SystemCriticality.SUPPORT.value == "support"


# =============================================================================
# Testing Priority Tests
# =============================================================================

class TestTestingPriority:
    """Tests for testing priority enumeration."""

    def test_p1_immediate(self):
        """Test P1 immediate priority."""
        assert TestingPriority.P1_IMMEDIATE.value == "P1"

    def test_p2_high(self):
        """Test P2 high priority (test within 1 month)."""
        assert TestingPriority.P2_HIGH.value == "P2"

    def test_p3_medium(self):
        """Test P3 medium priority (test within quarter)."""
        assert TestingPriority.P3_MEDIUM.value == "P3"

    def test_p4_low(self):
        """Test P4 low priority (test within year)."""
        assert TestingPriority.P4_LOW.value == "P4"

    def test_p5_minimal(self):
        """Test P5 minimal priority (as resources permit)."""
        assert TestingPriority.P5_MINIMAL.value == "P5"


# =============================================================================
# Vulnerability Severity Tests
# =============================================================================

class TestVulnerabilitySeverity:
    """Tests for vulnerability severity enumeration."""

    def test_critical_severity(self):
        """Test critical severity (CVSS 9.0-10.0)."""
        assert VulnerabilitySeverity.CRITICAL.value == "critical"

    def test_high_severity(self):
        """Test high severity (CVSS 7.0-8.9)."""
        assert VulnerabilitySeverity.HIGH.value == "high"

    def test_medium_severity(self):
        """Test medium severity (CVSS 4.0-6.9)."""
        assert VulnerabilitySeverity.MEDIUM.value == "medium"

    def test_low_severity(self):
        """Test low severity (CVSS 0.1-3.9)."""
        assert VulnerabilitySeverity.LOW.value == "low"

    def test_informational_severity(self):
        """Test informational severity (CVSS 0.0)."""
        assert VulnerabilitySeverity.INFORMATIONAL.value == "info"


# =============================================================================
# Vulnerability Status Tests
# =============================================================================

class TestVulnerabilityStatus:
    """Tests for vulnerability status enumeration."""

    def test_identified_status(self):
        """Test identified status."""
        assert VulnerabilityStatus.IDENTIFIED.value == "identified"

    def test_confirmed_status(self):
        """Test confirmed status."""
        assert VulnerabilityStatus.CONFIRMED.value == "confirmed"

    def test_in_remediation_status(self):
        """Test in remediation status."""
        assert VulnerabilityStatus.IN_REMEDIATION.value == "in_remediation"

    def test_remediated_status(self):
        """Test remediated status."""
        assert VulnerabilityStatus.REMEDIATED.value == "remediated"

    def test_verified_status(self):
        """Test verified status."""
        assert VulnerabilityStatus.VERIFIED.value == "verified"

    def test_accepted_status(self):
        """Test accepted (risk accepted) status."""
        assert VulnerabilityStatus.ACCEPTED.value == "accepted"

    def test_false_positive_status(self):
        """Test false positive status."""
        assert VulnerabilityStatus.FALSE_POSITIVE.value == "false_positive"


# =============================================================================
# Test Type Tests
# =============================================================================

class TestTestType:
    """Tests for test type enumeration."""

    def test_vulnerability_scan(self):
        """Test vulnerability scan type."""
        assert TestType.VULNERABILITY_SCAN.value == "vulnerability_scan"

    def test_penetration_test(self):
        """Test penetration test type."""
        assert TestType.PENETRATION_TEST.value == "penetration_test"

    def test_configuration_review(self):
        """Test configuration review type."""
        assert TestType.CONFIGURATION_REVIEW.value == "configuration_review"

    def test_code_review(self):
        """Test code review type."""
        assert TestType.CODE_REVIEW.value == "code_review"

    def test_performance_test(self):
        """Test performance test type."""
        assert TestType.PERFORMANCE_TEST.value == "performance_test"

    def test_stress_test(self):
        """Test stress test type."""
        assert TestType.STRESS_TEST.value == "stress_test"

    def test_failover_test(self):
        """Test failover test type."""
        assert TestType.FAILOVER_TEST.value == "failover_test"

    def test_backup_test(self):
        """Test backup test type."""
        assert TestType.BACKUP_TEST.value == "backup_test"

    def test_recovery_test(self):
        """Test recovery test type."""
        assert TestType.RECOVERY_TEST.value == "recovery_test"


# =============================================================================
# Test Status Tests
# =============================================================================

class TestTestStatus:
    """Tests for test status enumeration."""

    def test_planned_status(self):
        """Test planned status."""
        assert TestStatus.PLANNED.value == "planned"

    def test_scheduled_status(self):
        """Test scheduled status."""
        assert TestStatus.SCHEDULED.value == "scheduled"

    def test_in_progress_status(self):
        """Test in progress status."""
        assert TestStatus.IN_PROGRESS.value == "in_progress"

    def test_completed_status(self):
        """Test completed status."""
        assert TestStatus.COMPLETED.value == "completed"


# =============================================================================
# Test Result Tests
# =============================================================================

class TestTestResult:
    """Tests for test result enumeration."""

    def test_passed_result(self):
        """Test passed result."""
        assert TestResult.PASSED.value == "passed"

    def test_passed_with_findings_result(self):
        """Test passed with findings result."""
        assert TestResult.PASSED_WITH_FINDINGS.value == "passed_with_findings"

    def test_failed_result(self):
        """Test failed result."""
        assert TestResult.FAILED.value == "failed"

    def test_inconclusive_result(self):
        """Test inconclusive result."""
        assert TestResult.INCONCLUSIVE.value == "inconclusive"


# =============================================================================
# Remediation Status Tests
# =============================================================================

class TestRemediationStatus:
    """Tests for remediation status enumeration."""

    def test_draft_status(self):
        """Test draft status."""
        assert RemediationStatus.DRAFT.value == "draft"

    def test_pending_approval_status(self):
        """Test pending approval status."""
        assert RemediationStatus.PENDING_APPROVAL.value == "pending_approval"

    def test_approved_status(self):
        """Test approved status."""
        assert RemediationStatus.APPROVED.value == "approved"

    def test_in_progress_status(self):
        """Test in progress status."""
        assert RemediationStatus.IN_PROGRESS.value == "in_progress"

    def test_completed_status(self):
        """Test completed status."""
        assert RemediationStatus.COMPLETED.value == "completed"


# =============================================================================
# ICT System Profile Tests
# =============================================================================

class TestICTSystemProfile:
    """Tests for ICT system profile structure."""

    def test_profile_creation_with_defaults(self):
        """Test profile creation with default values."""
        profile = ICTSystemProfile()
        assert profile.system_id != ""
        assert profile.criticality == SystemCriticality.STANDARD

    def test_profile_with_name(self):
        """Test profile with specific name."""
        profile = ICTSystemProfile(
            name="Trading Engine",
            system_type=ICTSystemType.TRADING_SYSTEM
        )
        assert profile.name == "Trading Engine"
        assert profile.system_type == ICTSystemType.TRADING_SYSTEM

    def test_profile_with_criticality(self):
        """Test profile with specific criticality."""
        profile = ICTSystemProfile(
            name="Risk Engine",
            criticality=SystemCriticality.CRITICAL
        )
        assert profile.criticality == SystemCriticality.CRITICAL

    def test_profile_auto_generates_id(self):
        """Test that profile auto-generates a unique ID."""
        profile1 = ICTSystemProfile()
        profile2 = ICTSystemProfile()
        assert profile1.system_id != profile2.system_id


# =============================================================================
# System Test Plan Tests
# =============================================================================

class TestSystemTestPlan:
    """Tests for system test plan structure."""

    def test_plan_creation_with_defaults(self):
        """Test plan creation with default values."""
        plan = SystemTestPlan()
        assert plan.plan_id != ""
        assert plan.tests == []

    def test_plan_with_name(self):
        """Test plan with specific name."""
        plan = SystemTestPlan(
            name="Q1 2025 Test Plan",
            description="Quarterly testing plan"
        )
        assert plan.name == "Q1 2025 Test Plan"

    def test_plan_auto_generates_id(self):
        """Test that plan auto-generates a unique ID."""
        plan1 = SystemTestPlan()
        plan2 = SystemTestPlan()
        assert plan1.plan_id != plan2.plan_id


# =============================================================================
# System Test Tests
# =============================================================================

class TestSystemTest:
    """Tests for system test structure."""

    def test_test_creation_with_defaults(self):
        """Test creation with default values."""
        test = SystemTest()
        assert test.test_id != ""
        assert test.status == TestStatus.PLANNED

    def test_test_with_type(self):
        """Test with specific type."""
        test = SystemTest(
            name="Vulnerability Scan",
            test_type=TestType.VULNERABILITY_SCAN
        )
        assert test.test_type == TestType.VULNERABILITY_SCAN

    def test_test_auto_generates_id(self):
        """Test that test auto-generates a unique ID."""
        test1 = SystemTest()
        test2 = SystemTest()
        assert test1.test_id != test2.test_id


# =============================================================================
# Vulnerability Tests
# =============================================================================

class TestVulnerability:
    """Tests for vulnerability structure."""

    def test_vulnerability_creation_with_defaults(self):
        """Test vulnerability creation with default values."""
        vuln = Vulnerability()
        assert vuln.vulnerability_id != ""
        assert vuln.severity == VulnerabilitySeverity.MEDIUM
        assert vuln.status == VulnerabilityStatus.IDENTIFIED

    def test_vulnerability_with_severity(self):
        """Test vulnerability with specific severity."""
        vuln = Vulnerability(
            title="SQL Injection",
            severity=VulnerabilitySeverity.CRITICAL
        )
        assert vuln.severity == VulnerabilitySeverity.CRITICAL

    def test_vulnerability_with_cvss(self):
        """Test vulnerability with CVSS score."""
        vuln = Vulnerability(
            title="XSS Vulnerability",
            cvss_score=7.5,
            severity=VulnerabilitySeverity.HIGH
        )
        assert vuln.cvss_score == 7.5

    def test_vulnerability_auto_generates_id(self):
        """Test that vulnerability auto-generates a unique ID."""
        vuln1 = Vulnerability()
        vuln2 = Vulnerability()
        assert vuln1.vulnerability_id != vuln2.vulnerability_id


# =============================================================================
# Remediation Plan Tests
# =============================================================================

class TestRemediationPlan:
    """Tests for remediation plan structure."""

    def test_plan_creation_with_defaults(self):
        """Test plan creation with default values."""
        plan = RemediationPlan()
        assert plan.plan_id != ""
        assert plan.status == RemediationStatus.DRAFT

    def test_plan_with_vulnerability_id(self):
        """Test plan with specific vulnerability ID."""
        plan = RemediationPlan(
            vulnerability_id="VULN-001",
            description="Patch SQL injection vulnerability"
        )
        assert plan.vulnerability_id == "VULN-001"

    def test_plan_auto_generates_id(self):
        """Test that plan auto-generates a unique ID."""
        plan1 = RemediationPlan()
        plan2 = RemediationPlan()
        assert plan1.plan_id != plan2.plan_id


# =============================================================================
# Third-Party Interface Test Tests
# =============================================================================

class TestThirdPartyInterfaceTest:
    """Tests for third-party interface test structure."""

    def test_interface_test_creation_with_defaults(self):
        """Test interface test creation with default values."""
        test = ThirdPartyInterfaceTest()
        assert test.test_id != ""

    def test_interface_test_with_provider(self):
        """Test interface test with specific provider."""
        test = ThirdPartyInterfaceTest(
            provider_name="Bloomberg",
            interface_type="Market Data API"
        )
        assert test.provider_name == "Bloomberg"

    def test_interface_test_auto_generates_id(self):
        """Test that interface test auto-generates a unique ID."""
        test1 = ThirdPartyInterfaceTest()
        test2 = ThirdPartyInterfaceTest()
        assert test1.test_id != test2.test_id


# =============================================================================
# ICT Testing Config Tests
# =============================================================================

class TestICTTestingConfig:
    """Tests for ICT testing configuration."""

    def test_config_creation_with_defaults(self):
        """Test config creation with default values."""
        config = ICTTestingConfig()
        assert config is not None

    def test_config_with_entity_id(self):
        """Test config with entity ID."""
        config = ICTTestingConfig(
            entity_id="ENT-001"
        )
        assert config.entity_id == "ENT-001"


# =============================================================================
# DORAICTSystemTesting Creation Tests
# =============================================================================

class TestDORAICTSystemTestingCreation:
    """Tests for DORAICTSystemTesting creation."""

    def test_create_with_factory_function(self):
        """Test creation via factory function."""
        testing = create_ict_system_testing()
        assert testing is not None
        assert isinstance(testing, DORAICTSystemTesting)

    def test_create_with_config(self):
        """Test creation with custom config."""
        config = ICTTestingConfig(entity_id="TEST-001")
        testing = create_ict_system_testing(config)
        assert testing is not None
        assert testing.config.entity_id == "TEST-001"

    def test_has_required_methods(self):
        """Test that system testing has required methods."""
        testing = create_ict_system_testing()
        assert hasattr(testing, 'register_system')
        assert hasattr(testing, 'create_test_plan')
        assert hasattr(testing, 'execute_test')


# =============================================================================
# DORAICTSystemTesting System Registration Tests
# =============================================================================

class TestDORAICTSystemTestingRegistration:
    """Tests for system registration functionality."""

    def test_register_system(self):
        """Test registering a new system."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Trading Engine",
            system_type=ICTSystemType.TRADING_SYSTEM,
            criticality=SystemCriticality.CRITICAL
        )
        assert system is not None
        assert system.name == "Trading Engine"

    def test_get_system(self):
        """Test retrieving a registered system."""
        testing = create_ict_system_testing()
        created = testing.register_system(
            name="Risk Engine",
            system_type=ICTSystemType.RISK_MANAGEMENT
        )
        retrieved = testing.get_system(created.system_id)
        assert retrieved is not None
        assert retrieved.name == "Risk Engine"

    def test_list_systems(self):
        """Test listing all systems."""
        testing = create_ict_system_testing()
        testing.register_system(name="System 1", system_type=ICTSystemType.DATABASE)
        testing.register_system(name="System 2", system_type=ICTSystemType.API_GATEWAY)
        systems = testing.list_systems()
        assert len(systems) >= 2


# =============================================================================
# DORAICTSystemTesting Test Plan Tests
# =============================================================================

class TestDORAICTSystemTestingPlan:
    """Tests for test plan management."""

    def test_create_test_plan(self):
        """Test creating a test plan."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Test System",
            system_type=ICTSystemType.TRADING_SYSTEM
        )
        plan = testing.create_test_plan(
            system_id=system.system_id,
            name="Annual Test Plan 2025"
        )
        assert plan is not None
        assert plan.name == "Annual Test Plan 2025"

    def test_get_test_plans(self):
        """Test retrieving test plans for a system."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Test System",
            system_type=ICTSystemType.DATABASE
        )
        testing.create_test_plan(
            system_id=system.system_id,
            name="Plan 1"
        )
        plans = testing.get_test_plans(system.system_id)
        assert len(plans) >= 1


# =============================================================================
# DORAICTSystemTesting Test Execution Tests
# =============================================================================

class TestDORAICTSystemTestingExecution:
    """Tests for test execution management."""

    def test_add_test_to_plan(self):
        """Test adding a test to a plan."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Test System",
            system_type=ICTSystemType.AUTHENTICATION
        )
        plan = testing.create_test_plan(
            system_id=system.system_id,
            name="Security Test Plan"
        )
        test = testing.add_test(
            plan_id=plan.plan_id,
            name="Authentication Pentest",
            test_type=TestType.PENETRATION_TEST
        )
        assert test is not None
        assert test.test_type == TestType.PENETRATION_TEST

    def test_execute_test(self):
        """Test executing a test."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Test System",
            system_type=ICTSystemType.DATABASE
        )
        plan = testing.create_test_plan(
            system_id=system.system_id,
            name="DB Test Plan"
        )
        test = testing.add_test(
            plan_id=plan.plan_id,
            name="Vuln Scan",
            test_type=TestType.VULNERABILITY_SCAN
        )
        executed = testing.execute_test(test.test_id)
        assert executed.status == TestStatus.IN_PROGRESS

    def test_complete_test(self):
        """Test completing a test."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Test System",
            system_type=ICTSystemType.API_GATEWAY
        )
        plan = testing.create_test_plan(
            system_id=system.system_id,
            name="API Test Plan"
        )
        test = testing.add_test(
            plan_id=plan.plan_id,
            name="API Security Test",
            test_type=TestType.PENETRATION_TEST
        )
        testing.execute_test(test.test_id)
        completed = testing.complete_test(
            test_id=test.test_id,
            result=TestResult.PASSED
        )
        assert completed.status == TestStatus.COMPLETED
        assert completed.result == TestResult.PASSED


# =============================================================================
# DORAICTSystemTesting Vulnerability Management Tests
# =============================================================================

class TestDORAICTSystemTestingVulnerabilities:
    """Tests for vulnerability management."""

    def test_add_vulnerability(self):
        """Test adding a vulnerability."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Test System",
            system_type=ICTSystemType.TRADING_SYSTEM
        )
        vuln = testing.add_vulnerability(
            system_id=system.system_id,
            title="SQL Injection in Order API",
            severity=VulnerabilitySeverity.CRITICAL,
            cvss_score=9.8
        )
        assert vuln is not None
        assert vuln.severity == VulnerabilitySeverity.CRITICAL

    def test_get_vulnerabilities(self):
        """Test retrieving vulnerabilities for a system."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Test System",
            system_type=ICTSystemType.DATABASE
        )
        testing.add_vulnerability(
            system_id=system.system_id,
            title="Weak Password Policy",
            severity=VulnerabilitySeverity.MEDIUM
        )
        vulns = testing.get_vulnerabilities(system.system_id)
        assert len(vulns) >= 1

    def test_update_vulnerability_status(self):
        """Test updating vulnerability status."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Test System",
            system_type=ICTSystemType.AUTHENTICATION
        )
        vuln = testing.add_vulnerability(
            system_id=system.system_id,
            title="Insecure Session Handling",
            severity=VulnerabilitySeverity.HIGH
        )
        updated = testing.update_vulnerability_status(
            vulnerability_id=vuln.vulnerability_id,
            status=VulnerabilityStatus.IN_REMEDIATION
        )
        assert updated.status == VulnerabilityStatus.IN_REMEDIATION


# =============================================================================
# DORAICTSystemTesting Remediation Tests
# =============================================================================

class TestDORAICTSystemTestingRemediation:
    """Tests for remediation plan management."""

    def test_create_remediation_plan(self):
        """Test creating a remediation plan."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Test System",
            system_type=ICTSystemType.TRADING_SYSTEM
        )
        vuln = testing.add_vulnerability(
            system_id=system.system_id,
            title="Critical Vulnerability",
            severity=VulnerabilitySeverity.CRITICAL
        )
        plan = testing.create_remediation_plan(
            vulnerability_id=vuln.vulnerability_id,
            description="Apply security patch",
            target_date=(datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
        )
        assert plan is not None
        assert plan.vulnerability_id == vuln.vulnerability_id

    def test_approve_remediation_plan(self):
        """Test approving a remediation plan."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Test System",
            system_type=ICTSystemType.DATABASE
        )
        vuln = testing.add_vulnerability(
            system_id=system.system_id,
            title="DB Vulnerability",
            severity=VulnerabilitySeverity.HIGH
        )
        plan = testing.create_remediation_plan(
            vulnerability_id=vuln.vulnerability_id,
            description="Upgrade database version"
        )
        approved = testing.approve_remediation_plan(plan.plan_id)
        assert approved.status == RemediationStatus.APPROVED


# =============================================================================
# Risk-Based Testing Tests (Article 25(1))
# =============================================================================

class TestRiskBasedSystemTesting:
    """Tests for risk-based testing approach per Article 25(1)."""

    def test_prioritize_systems_by_criticality(self):
        """Test prioritizing systems based on criticality."""
        testing = create_ict_system_testing()
        testing.register_system(
            name="Critical Trading System",
            system_type=ICTSystemType.TRADING_SYSTEM,
            criticality=SystemCriticality.CRITICAL
        )
        testing.register_system(
            name="Support Tool",
            system_type=ICTSystemType.OTHER,
            criticality=SystemCriticality.SUPPORT
        )
        prioritized = testing.get_prioritized_systems()
        assert prioritized[0].criticality == SystemCriticality.CRITICAL

    def test_determine_testing_priority(self):
        """Test determining testing priority based on system criticality."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Critical System",
            system_type=ICTSystemType.RISK_MANAGEMENT,
            criticality=SystemCriticality.CRITICAL
        )
        priority = testing.determine_testing_priority(system.system_id)
        assert priority == TestingPriority.P1_IMMEDIATE


# =============================================================================
# Annual Testing Requirement Tests (Article 25(1))
# =============================================================================

class TestAnnualSystemTesting:
    """Tests for annual testing requirement per Article 25(1)."""

    def test_critical_systems_require_annual_testing(self):
        """Test that critical systems require annual testing."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Critical System",
            system_type=ICTSystemType.TRADING_SYSTEM,
            criticality=SystemCriticality.CRITICAL
        )
        requirement = testing.get_testing_requirement(system.system_id)
        assert "annual" in str(requirement).lower() or requirement.get("minimum_frequency") == "annual"


# =============================================================================
# Reporting Tests
# =============================================================================

class TestICTSystemTestingReporting:
    """Tests for reporting functionality."""

    def test_generate_system_report(self):
        """Test generating a system test report."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Report Test System",
            system_type=ICTSystemType.TRADING_SYSTEM
        )
        report = testing.generate_system_report(system.system_id)
        assert report is not None
        assert "system_id" in report

    def test_get_vulnerability_summary(self):
        """Test getting vulnerability summary."""
        testing = create_ict_system_testing()
        system = testing.register_system(
            name="Vuln Test System",
            system_type=ICTSystemType.DATABASE
        )
        testing.add_vulnerability(
            system_id=system.system_id,
            title="Test Vuln",
            severity=VulnerabilitySeverity.HIGH
        )
        summary = testing.get_vulnerability_summary(system.system_id)
        assert summary is not None
