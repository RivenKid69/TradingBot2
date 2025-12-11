# -*- coding: utf-8 -*-
"""
Comprehensive Tests for DORA Phase 3 - Digital Operational Resilience Testing (Article 24).

Tests for the Digital Operational Resilience Testing Programme per Article 24 of DORA.

Coverage includes:
- Test Category enumeration (Article 24(1)(a))
- Test Frequency and Status management
- Test Scope definition
- Test Definition management
- Test Execution lifecycle
- Test Finding management
- Testing Programme creation and management
- Testing Cycle execution
- Risk-based test prioritization
- Annual testing requirements

References:
- Article 24 DORA: https://www.digital-operational-resilience-act.com/Article_24.html
- RTS on TLPT (CDR 2024/2698)
- TIBER-EU Framework
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

from services.dora.resilience_testing import (
    TestCategory,
    TestFrequency,
    TestStatus,
    TestResult,
    FindingSeverity,
    FindingStatus,
    TesterType,
    SystemCriticality,
    TestScope,
    TestDefinition,
    TestExecution,
    TestFinding,
    TestingProgramme,
    TestingCycle,
    ResilienceTestingConfig,
    DORAResilienceTestingProgramme,
    create_resilience_testing_programme,
)


# =============================================================================
# Article 24(1)(a): Test Category Tests
# =============================================================================

class TestTestCategories:
    """Tests for test categories per Article 24(1)(a)."""

    def test_vulnerability_assessment_category(self):
        """Test vulnerability assessment category exists."""
        assert TestCategory.VULNERABILITY_ASSESSMENT.value == "vulnerability_assessment"

    def test_open_source_analysis_category(self):
        """Test open source analysis category exists."""
        assert TestCategory.OPEN_SOURCE_ANALYSIS.value == "open_source_analysis"

    def test_network_security_category(self):
        """Test network security assessment category exists."""
        assert TestCategory.NETWORK_SECURITY.value == "network_security"

    def test_gap_analysis_category(self):
        """Test gap analysis category exists."""
        assert TestCategory.GAP_ANALYSIS.value == "gap_analysis"

    def test_physical_security_category(self):
        """Test physical security review category exists."""
        assert TestCategory.PHYSICAL_SECURITY.value == "physical_security"

    def test_questionnaire_category(self):
        """Test questionnaire category exists."""
        assert TestCategory.QUESTIONNAIRE.value == "questionnaire"

    def test_source_code_review_category(self):
        """Test source code review category exists."""
        assert TestCategory.SOURCE_CODE_REVIEW.value == "source_code_review"

    def test_scenario_based_category(self):
        """Test scenario-based test category exists."""
        assert TestCategory.SCENARIO_BASED.value == "scenario_based"

    def test_compatibility_category(self):
        """Test compatibility testing category exists."""
        assert TestCategory.COMPATIBILITY.value == "compatibility"

    def test_performance_category(self):
        """Test performance testing category exists."""
        assert TestCategory.PERFORMANCE.value == "performance"

    def test_end_to_end_category(self):
        """Test end-to-end testing category exists."""
        assert TestCategory.END_TO_END.value == "end_to_end"

    def test_penetration_testing_category(self):
        """Test penetration testing category exists."""
        assert TestCategory.PENETRATION_TESTING.value == "penetration_testing"

    def test_all_twelve_categories_defined(self):
        """Verify all 12 test categories per Article 24(1)(a) are defined."""
        assert len(TestCategory) == 12


# =============================================================================
# Test Frequency Tests
# =============================================================================

class TestTestFrequency:
    """Tests for test frequency enumeration."""

    def test_continuous_frequency(self):
        """Test continuous frequency value."""
        assert TestFrequency.CONTINUOUS.value == "continuous"

    def test_daily_frequency(self):
        """Test daily frequency value."""
        assert TestFrequency.DAILY.value == "daily"

    def test_weekly_frequency(self):
        """Test weekly frequency value."""
        assert TestFrequency.WEEKLY.value == "weekly"

    def test_monthly_frequency(self):
        """Test monthly frequency value."""
        assert TestFrequency.MONTHLY.value == "monthly"

    def test_quarterly_frequency(self):
        """Test quarterly frequency value."""
        assert TestFrequency.QUARTERLY.value == "quarterly"

    def test_annual_frequency(self):
        """Test annual frequency value."""
        assert TestFrequency.ANNUAL.value == "annual"

    def test_triennial_frequency(self):
        """Test triennial frequency (TLPT minimum) value."""
        assert TestFrequency.TRIENNIAL.value == "triennial"

    def test_on_change_frequency(self):
        """Test on-change trigger frequency value."""
        assert TestFrequency.ON_CHANGE.value == "on_change"


# =============================================================================
# Test Status Tests
# =============================================================================

class TestTestStatus:
    """Tests for test status enumeration."""

    def test_planned_status(self):
        """Test planned status value."""
        assert TestStatus.PLANNED.value == "planned"

    def test_scheduled_status(self):
        """Test scheduled status value."""
        assert TestStatus.SCHEDULED.value == "scheduled"

    def test_in_progress_status(self):
        """Test in progress status value."""
        assert TestStatus.IN_PROGRESS.value == "in_progress"

    def test_completed_status(self):
        """Test completed status value."""
        assert TestStatus.COMPLETED.value == "completed"

    def test_cancelled_status(self):
        """Test cancelled status value."""
        assert TestStatus.CANCELLED.value == "cancelled"

    def test_failed_status(self):
        """Test failed status value."""
        assert TestStatus.FAILED.value == "failed"


# =============================================================================
# Test Result Tests
# =============================================================================

class TestTestResult:
    """Tests for test result enumeration."""

    def test_passed_result(self):
        """Test passed result value."""
        assert TestResult.PASSED.value == "passed"

    def test_passed_with_findings_result(self):
        """Test passed with findings result value."""
        assert TestResult.PASSED_WITH_FINDINGS.value == "passed_with_findings"

    def test_failed_result(self):
        """Test failed result value."""
        assert TestResult.FAILED.value == "failed"

    def test_inconclusive_result(self):
        """Test inconclusive result value."""
        assert TestResult.INCONCLUSIVE.value == "inconclusive"


# =============================================================================
# Finding Severity Tests
# =============================================================================

class TestFindingSeverity:
    """Tests for finding severity enumeration."""

    def test_critical_severity(self):
        """Test critical severity (immediate action required)."""
        assert FindingSeverity.CRITICAL.value == "critical"

    def test_high_severity(self):
        """Test high severity (action within 7 days)."""
        assert FindingSeverity.HIGH.value == "high"

    def test_medium_severity(self):
        """Test medium severity (action within 30 days)."""
        assert FindingSeverity.MEDIUM.value == "medium"

    def test_low_severity(self):
        """Test low severity (action within 90 days)."""
        assert FindingSeverity.LOW.value == "low"

    def test_informational_severity(self):
        """Test informational severity (no action required)."""
        assert FindingSeverity.INFORMATIONAL.value == "informational"


# =============================================================================
# Finding Status Tests
# =============================================================================

class TestFindingStatus:
    """Tests for finding status enumeration."""

    def test_open_status(self):
        """Test open finding status."""
        assert FindingStatus.OPEN.value == "open"

    def test_in_progress_status(self):
        """Test in progress finding status."""
        assert FindingStatus.IN_PROGRESS.value == "in_progress"

    def test_resolved_status(self):
        """Test resolved finding status."""
        assert FindingStatus.RESOLVED.value == "resolved"

    def test_accepted_status(self):
        """Test accepted (risk accepted) finding status."""
        assert FindingStatus.ACCEPTED.value == "accepted"

    def test_false_positive_status(self):
        """Test false positive finding status."""
        assert FindingStatus.FALSE_POSITIVE.value == "false_positive"


# =============================================================================
# Tester Type Tests (Article 24(4))
# =============================================================================

class TestTesterType:
    """Tests for tester type per Article 24(4)."""

    def test_internal_tester(self):
        """Test internal tester type."""
        assert TesterType.INTERNAL.value == "internal"

    def test_external_tester(self):
        """Test external tester type."""
        assert TesterType.EXTERNAL.value == "external"

    def test_mixed_tester(self):
        """Test mixed (internal + external) tester type."""
        assert TesterType.MIXED.value == "mixed"


# =============================================================================
# System Criticality Tests
# =============================================================================

class TestSystemCriticality:
    """Tests for system criticality levels."""

    def test_critical_level(self):
        """Test critical system level."""
        assert SystemCriticality.CRITICAL.value == "critical"

    def test_high_level(self):
        """Test high importance level."""
        assert SystemCriticality.HIGH.value == "high"

    def test_medium_level(self):
        """Test medium importance level."""
        assert SystemCriticality.MEDIUM.value == "medium"

    def test_low_level(self):
        """Test low importance level."""
        assert SystemCriticality.LOW.value == "low"


# =============================================================================
# Test Scope Tests (Article 24(3))
# =============================================================================

class TestTestScope:
    """Tests for test scope definition per Article 24(3)."""

    def test_scope_creation_with_defaults(self):
        """Test scope creation with default values."""
        scope = TestScope()
        assert scope.scope_id != ""
        assert scope.name == ""
        assert scope.systems == []

    def test_scope_creation_with_values(self):
        """Test scope creation with specific values."""
        scope = TestScope(
            name="Critical Systems Scope",
            description="Testing of critical ICT systems",
            systems=["trading_platform", "risk_engine"]
        )
        assert scope.name == "Critical Systems Scope"
        assert "trading_platform" in scope.systems

    def test_scope_auto_generates_id(self):
        """Test that scope auto-generates a unique ID."""
        scope1 = TestScope()
        scope2 = TestScope()
        assert scope1.scope_id != scope2.scope_id

    def test_scope_id_format(self):
        """Test scope ID format starts with SCOPE-."""
        scope = TestScope()
        assert scope.scope_id.startswith("SCOPE-")


# =============================================================================
# Test Definition Tests
# =============================================================================

class TestTestDefinition:
    """Tests for test definition structure."""

    def test_definition_creation_with_defaults(self):
        """Test definition creation with default values."""
        definition = TestDefinition()
        assert definition.test_id != ""
        assert definition.category == TestCategory.VULNERABILITY_ASSESSMENT

    def test_definition_with_category(self):
        """Test definition with specific category."""
        definition = TestDefinition(
            name="Network Security Scan",
            category=TestCategory.NETWORK_SECURITY
        )
        assert definition.category == TestCategory.NETWORK_SECURITY

    def test_definition_with_frequency(self):
        """Test definition with specific frequency."""
        definition = TestDefinition(
            name="Quarterly Pentest",
            frequency=TestFrequency.QUARTERLY
        )
        assert definition.frequency == TestFrequency.QUARTERLY

    def test_definition_auto_generates_id(self):
        """Test that definition auto-generates a unique ID."""
        def1 = TestDefinition()
        def2 = TestDefinition()
        assert def1.test_id != def2.test_id


# =============================================================================
# Test Execution Tests
# =============================================================================

class TestTestExecution:
    """Tests for test execution structure."""

    def test_execution_creation_with_defaults(self):
        """Test execution creation with default values."""
        execution = TestExecution()
        assert execution.execution_id != ""
        assert execution.status == TestStatus.PLANNED

    def test_execution_with_status(self):
        """Test execution with specific status."""
        execution = TestExecution(
            status=TestStatus.IN_PROGRESS
        )
        assert execution.status == TestStatus.IN_PROGRESS

    def test_execution_with_result(self):
        """Test execution with specific result."""
        execution = TestExecution(
            status=TestStatus.COMPLETED,
            result=TestResult.PASSED
        )
        assert execution.result == TestResult.PASSED

    def test_execution_auto_generates_id(self):
        """Test that execution auto-generates a unique ID."""
        exec1 = TestExecution()
        exec2 = TestExecution()
        assert exec1.execution_id != exec2.execution_id


# =============================================================================
# Test Finding Tests
# =============================================================================

class TestTestFinding:
    """Tests for test finding structure."""

    def test_finding_creation_with_defaults(self):
        """Test finding creation with default values."""
        finding = TestFinding()
        assert finding.finding_id != ""
        assert finding.severity == FindingSeverity.MEDIUM
        assert finding.status == FindingStatus.OPEN

    def test_finding_with_severity(self):
        """Test finding with specific severity."""
        finding = TestFinding(
            title="SQL Injection Vulnerability",
            severity=FindingSeverity.CRITICAL
        )
        assert finding.severity == FindingSeverity.CRITICAL

    def test_finding_with_status(self):
        """Test finding with specific status."""
        finding = TestFinding(
            title="Resolved Issue",
            status=FindingStatus.RESOLVED
        )
        assert finding.status == FindingStatus.RESOLVED

    def test_finding_auto_generates_id(self):
        """Test that finding auto-generates a unique ID."""
        find1 = TestFinding()
        find2 = TestFinding()
        assert find1.finding_id != find2.finding_id


# =============================================================================
# Testing Programme Tests
# =============================================================================

class TestTestingProgramme:
    """Tests for testing programme structure."""

    def test_programme_creation_with_defaults(self):
        """Test programme creation with default values."""
        programme = TestingProgramme()
        assert programme.programme_id != ""
        assert programme.name == ""
        assert programme.test_definitions == []

    def test_programme_with_name(self):
        """Test programme with specific name."""
        programme = TestingProgramme(
            name="2025 Digital Resilience Testing Programme"
        )
        assert programme.name == "2025 Digital Resilience Testing Programme"

    def test_programme_auto_generates_id(self):
        """Test that programme auto-generates a unique ID."""
        prog1 = TestingProgramme()
        prog2 = TestingProgramme()
        assert prog1.programme_id != prog2.programme_id


# =============================================================================
# Testing Cycle Tests
# =============================================================================

class TestTestingCycle:
    """Tests for testing cycle structure."""

    def test_cycle_creation_with_defaults(self):
        """Test cycle creation with default values."""
        cycle = TestingCycle()
        assert cycle.cycle_id != ""

    def test_cycle_with_year(self):
        """Test cycle with specific year."""
        cycle = TestingCycle(
            cycle_year=2025,
            name="Annual Testing Cycle 2025"
        )
        assert cycle.cycle_year == 2025

    def test_cycle_auto_generates_id(self):
        """Test that cycle auto-generates a unique ID."""
        cyc1 = TestingCycle()
        cyc2 = TestingCycle()
        assert cyc1.cycle_id != cyc2.cycle_id


# =============================================================================
# Resilience Testing Config Tests
# =============================================================================

class TestResilienceTestingConfig:
    """Tests for resilience testing configuration."""

    def test_config_creation_with_defaults(self):
        """Test config creation with default values."""
        config = ResilienceTestingConfig()
        assert config is not None

    def test_config_has_properties(self):
        """Test config has expected properties."""
        config = ResilienceTestingConfig()
        assert hasattr(config, 'require_independent_testing')
        assert hasattr(config, 'default_review_frequency_days')


# =============================================================================
# DORAResilienceTestingProgramme Creation Tests
# =============================================================================

class TestDORAResilienceTestingProgrammeCreation:
    """Tests for DORAResilienceTestingProgramme creation."""

    def test_create_with_factory_function(self):
        """Test creation via factory function."""
        programme = create_resilience_testing_programme()
        assert programme is not None
        assert isinstance(programme, DORAResilienceTestingProgramme)

    def test_create_with_config(self):
        """Test creation with custom config."""
        config = ResilienceTestingConfig()
        programme = create_resilience_testing_programme(config)
        assert programme is not None
        assert programme.config is not None

    def test_has_required_methods(self):
        """Test that programme has required methods."""
        programme = create_resilience_testing_programme()
        assert hasattr(programme, 'create_programme')
        assert hasattr(programme, 'create_test_definition')
        assert hasattr(programme, 'execute_test')


# =============================================================================
# DORAResilienceTestingProgramme Programme Management Tests
# =============================================================================

class TestDORAResilienceTestingProgrammeManagement:
    """Tests for programme management functionality."""

    def test_create_programme(self):
        """Test creating a new programme."""
        programme_mgr = create_resilience_testing_programme()
        programme = programme_mgr.create_programme(
            name="2025 Resilience Testing",
            entity_name="Test Financial Entity",
            entity_type="investment_firm",
            description="Annual testing programme"
        )
        assert programme is not None
        assert programme.name == "2025 Resilience Testing"

    def test_get_programme(self):
        """Test retrieving a programme."""
        programme_mgr = create_resilience_testing_programme()
        created = programme_mgr.create_programme(
            name="Test Programme",
            entity_name="Test Entity",
            entity_type="investment_firm"
        )
        retrieved = programme_mgr.get_programme(created.programme_id)
        assert retrieved is not None
        assert retrieved.name == "Test Programme"

    def test_get_all_programmes(self):
        """Test getting all test definitions."""
        programme_mgr = create_resilience_testing_programme()
        definitions = programme_mgr.get_all_test_definitions()
        # Should have default test definitions
        assert len(definitions) >= 0


# =============================================================================
# DORAResilienceTestingProgramme Test Definition Tests
# =============================================================================

class TestDORAResilienceTestingDefinitions:
    """Tests for test definition management."""

    def test_create_test_definition(self):
        """Test creating a test definition."""
        programme_mgr = create_resilience_testing_programme()
        definition = programme_mgr.create_test_definition(
            name="Vulnerability Scan",
            category=TestCategory.VULNERABILITY_ASSESSMENT,
            frequency=TestFrequency.MONTHLY
        )
        assert definition is not None
        assert definition.name == "Vulnerability Scan"

    def test_get_test_definitions_by_category(self):
        """Test retrieving test definitions by category."""
        programme_mgr = create_resilience_testing_programme()
        programme_mgr.create_test_definition(
            name="Test 1",
            category=TestCategory.VULNERABILITY_ASSESSMENT
        )
        definitions = programme_mgr.get_test_definitions_by_category(
            TestCategory.VULNERABILITY_ASSESSMENT
        )
        assert len(definitions) >= 1


# =============================================================================
# DORAResilienceTestingProgramme Test Execution Tests
# =============================================================================

class TestDORAResilienceTestingExecution:
    """Tests for test execution management."""

    def test_execute_test(self):
        """Test executing a test."""
        programme_mgr = create_resilience_testing_programme()
        programme = programme_mgr.create_programme(
            name="Test Programme",
            entity_name="Test Entity",
            entity_type="investment_firm"
        )
        scope = programme_mgr.create_scope(
            name="Test Scope",
            systems=["trading_system"]
        )
        definition = programme_mgr.create_test_definition(
            name="Pentest",
            category=TestCategory.PENETRATION_TESTING
        )
        execution = programme_mgr.execute_test(
            test_id=definition.test_id,
            scope_id=scope.scope_id,
            tester_type=TesterType.EXTERNAL
        )
        assert execution is not None

    def test_get_execution(self):
        """Test getting test execution."""
        programme_mgr = create_resilience_testing_programme()
        programme = programme_mgr.create_programme(
            name="Test Programme",
            entity_name="Test Entity",
            entity_type="investment_firm"
        )
        scope = programme_mgr.create_scope(name="Test Scope")
        definition = programme_mgr.create_test_definition(
            name="Test",
            category=TestCategory.VULNERABILITY_ASSESSMENT
        )
        execution = programme_mgr.execute_test(
            test_id=definition.test_id,
            scope_id=scope.scope_id
        )
        retrieved = programme_mgr.get_execution(execution.execution_id)
        assert retrieved is not None

    def test_complete_execution(self):
        """Test completing a test execution."""
        programme_mgr = create_resilience_testing_programme()
        programme = programme_mgr.create_programme(
            name="Test Programme",
            entity_name="Test Entity",
            entity_type="investment_firm"
        )
        scope = programme_mgr.create_scope(name="Test Scope")
        definition = programme_mgr.create_test_definition(
            name="Test",
            category=TestCategory.VULNERABILITY_ASSESSMENT
        )
        execution = programme_mgr.execute_test(
            test_id=definition.test_id,
            scope_id=scope.scope_id
        )
        completed = programme_mgr.complete_execution(
            execution_id=execution.execution_id,
            result=TestResult.PASSED
        )
        assert completed.status == TestStatus.COMPLETED


# =============================================================================
# DORAResilienceTestingProgramme Finding Tests
# =============================================================================

class TestDORAResilienceTestingFindings:
    """Tests for finding management."""

    def test_record_finding(self):
        """Test recording a finding from a test execution."""
        programme_mgr = create_resilience_testing_programme()
        programme = programme_mgr.create_programme(
            name="Test Programme",
            entity_name="Test Entity",
            entity_type="investment_firm"
        )
        scope = programme_mgr.create_scope(name="Test Scope")
        definition = programme_mgr.create_test_definition(
            name="Test",
            category=TestCategory.PENETRATION_TESTING
        )
        execution = programme_mgr.execute_test(
            test_id=definition.test_id,
            scope_id=scope.scope_id
        )
        finding = programme_mgr.record_finding(
            execution_id=execution.execution_id,
            title="XSS Vulnerability",
            description="Cross-site scripting found in login form",
            severity=FindingSeverity.HIGH
        )
        assert finding is not None
        assert finding.title == "XSS Vulnerability"

    def test_get_open_findings(self):
        """Test retrieving open findings."""
        programme_mgr = create_resilience_testing_programme()
        findings = programme_mgr.get_open_findings()
        # Should return list (may be empty)
        assert isinstance(findings, list)

    def test_get_findings_by_severity(self):
        """Test getting findings by severity."""
        programme_mgr = create_resilience_testing_programme()
        findings = programme_mgr.get_findings_by_severity(FindingSeverity.CRITICAL)
        assert isinstance(findings, list)


# =============================================================================
# DORAResilienceTestingProgramme Testing Cycle Tests
# =============================================================================

class TestDORAResilienceTestingCycle:
    """Tests for testing cycle management."""

    def test_create_testing_cycle(self):
        """Test creating a testing cycle."""
        programme_mgr = create_resilience_testing_programme()
        programme = programme_mgr.create_programme(
            name="Test Programme",
            entity_name="Test Entity",
            entity_type="investment_firm"
        )
        cycle = programme_mgr.create_testing_cycle(
            programme_id=programme.programme_id,
            cycle_year=2025,
            name="Q1 2025 Testing"
        )
        assert cycle is not None
        assert cycle.cycle_year == 2025

    def test_get_cycle(self):
        """Test retrieving a testing cycle."""
        programme_mgr = create_resilience_testing_programme()
        programme = programme_mgr.create_programme(
            name="Test Programme",
            entity_name="Test Entity",
            entity_type="investment_firm"
        )
        cycle = programme_mgr.create_testing_cycle(
            programme_id=programme.programme_id,
            cycle_year=2025
        )
        retrieved = programme_mgr.get_cycle(cycle.cycle_id)
        assert retrieved is not None


# =============================================================================
# Annual Testing Requirements Tests (Article 24)
# =============================================================================

class TestAnnualTestingRequirements:
    """Tests for annual testing requirements per Article 24."""

    def test_critical_systems_annual_testing(self):
        """Test that critical systems must be tested annually."""
        programme_mgr = create_resilience_testing_programme()
        definition = programme_mgr.create_test_definition(
            name="Critical System Test",
            category=TestCategory.END_TO_END,
            frequency=TestFrequency.ANNUAL
        )
        assert definition.frequency == TestFrequency.ANNUAL

    def test_default_test_definitions_exist(self):
        """Test that default test definitions are created."""
        programme_mgr = create_resilience_testing_programme()
        definitions = programme_mgr.get_all_test_definitions()
        # Should have default tests for each Article 24(1)(a) category
        assert len(definitions) >= 12


# =============================================================================
# Risk-Based Testing Tests (Article 24(1))
# =============================================================================

class TestRiskBasedTesting:
    """Tests for risk-based testing approach per Article 24(1)."""

    def test_risk_based_approach_enabled(self):
        """Test that risk-based approach is enabled by default."""
        programme_mgr = create_resilience_testing_programme()
        programme = programme_mgr.create_programme(
            name="Risk-Based Testing",
            entity_name="Test Entity",
            entity_type="investment_firm"
        )
        assert programme.risk_based_approach == True


# =============================================================================
# Reporting and Export Tests
# =============================================================================

class TestResilienceTestingReporting:
    """Tests for reporting and export functionality."""

    def test_get_compliance_status(self):
        """Test getting compliance status."""
        programme_mgr = create_resilience_testing_programme()
        status = programme_mgr.get_compliance_status()
        assert status is not None

    def test_get_testing_statistics(self):
        """Test getting testing statistics."""
        programme_mgr = create_resilience_testing_programme()
        stats = programme_mgr.get_testing_statistics()
        assert stats is not None
