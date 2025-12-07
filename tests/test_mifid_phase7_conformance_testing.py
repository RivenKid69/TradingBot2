# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 7 - Conformance Testing (RTS 6 Article 5).

Comprehensive test coverage for:
- ConformanceTest dataclass
- ConformanceTestSuite functionality
- ConformanceTestRunner execution
- TestEvidence collection
- Standard test templates
"""

import json
import time
import pytest
from datetime import datetime, date, timedelta
from typing import Dict, Any

from services.compliance.conformance_testing import (
    # Enums
    TestResult,
    TestCategory,
    TestPriority,
    TestEnvironment,
    ConformanceSuiteStatus,
    CertificationStatus,
    # Data classes
    TestEvidence,
    ConformanceTest,
    ConformanceTestSuite,
    # Runner
    TestExecutorConfig,
    ConformanceTestRunner,
    # Factory functions
    create_conformance_suite,
    create_test_runner,
    get_standard_conformance_tests,
)


# =============================================================================
# Test Enums
# =============================================================================


class TestEnums:
    """Test enum definitions."""

    def test_test_result_values(self):
        """Test TestResult enum has all expected values."""
        assert TestResult.PASS.value == "pass"
        assert TestResult.FAIL.value == "fail"
        assert TestResult.SKIP.value == "skip"
        assert TestResult.ERROR.value == "error"
        assert TestResult.WARNING.value == "warning"
        assert TestResult.PENDING.value == "pending"

    def test_test_category_values(self):
        """Test TestCategory enum has required categories."""
        categories = [c.value for c in TestCategory]
        assert "risk_controls" in categories
        assert "kill_switch" in categories
        assert "pre_trade" in categories
        assert "clock_sync" in categories
        assert "record_keeping" in categories
        assert "stress_test" in categories
        assert "best_execution" in categories
        assert "transaction_reporting" in categories

    def test_test_priority_values(self):
        """Test TestPriority enum values."""
        assert TestPriority.CRITICAL.value == "critical"
        assert TestPriority.HIGH.value == "high"
        assert TestPriority.MEDIUM.value == "medium"
        assert TestPriority.LOW.value == "low"

    def test_test_environment_values(self):
        """Test TestEnvironment enum values."""
        assert TestEnvironment.SANDBOX.value == "sandbox"
        assert TestEnvironment.UAT.value == "uat"
        assert TestEnvironment.STAGING.value == "staging"
        assert TestEnvironment.PRODUCTION_REPLICA.value == "production_replica"
        assert TestEnvironment.PRODUCTION.value == "production"

    def test_suite_status_values(self):
        """Test ConformanceSuiteStatus enum values."""
        assert ConformanceSuiteStatus.NOT_STARTED.value == "not_started"
        assert ConformanceSuiteStatus.IN_PROGRESS.value == "in_progress"
        assert ConformanceSuiteStatus.COMPLETED.value == "completed"
        assert ConformanceSuiteStatus.PASSED.value == "passed"
        assert ConformanceSuiteStatus.FAILED.value == "failed"


# =============================================================================
# Test TestEvidence
# =============================================================================


class TestTestEvidence:
    """Test TestEvidence dataclass."""

    def test_create_evidence(self):
        """Test creating evidence."""
        evidence = TestEvidence(
            evidence_type="log",
            description="Test log output",
            content="Log content here",
        )
        assert evidence.evidence_id
        assert evidence.evidence_type == "log"
        assert evidence.description == "Test log output"
        assert evidence.content == "Log content here"
        assert evidence.timestamp_ns > 0

    def test_evidence_to_dict(self):
        """Test evidence serialization."""
        evidence = TestEvidence(
            evidence_type="screenshot",
            description="UI screenshot",
            file_path="/path/to/screenshot.png",
        )
        data = evidence.to_dict()
        assert data["evidence_type"] == "screenshot"
        assert data["description"] == "UI screenshot"
        assert data["file_path"] == "/path/to/screenshot.png"

    def test_evidence_content_truncation(self):
        """Test evidence content is truncated in to_dict."""
        long_content = "x" * 2000
        evidence = TestEvidence(content=long_content)
        data = evidence.to_dict()
        assert len(data["content"]) == 1000  # Truncated


# =============================================================================
# Test ConformanceTest
# =============================================================================


class TestConformanceTest:
    """Test ConformanceTest dataclass."""

    def test_create_test(self):
        """Test creating a conformance test."""
        test = ConformanceTest(
            test_id="CT-001",
            name="Test Kill Switch",
            category=TestCategory.KILL_SWITCH,
            description="Verify kill switch functionality",
            rts_reference="RTS 6 Article 12",
            priority=TestPriority.CRITICAL,
        )
        assert test.test_id == "CT-001"
        assert test.name == "Test Kill Switch"
        assert test.category == TestCategory.KILL_SWITCH
        assert test.priority == TestPriority.CRITICAL
        assert test.result == TestResult.PENDING

    def test_test_is_critical(self):
        """Test is_critical method."""
        critical_test = ConformanceTest(priority=TestPriority.CRITICAL)
        high_test = ConformanceTest(priority=TestPriority.HIGH)

        assert critical_test.is_critical()
        assert not high_test.is_critical()

    def test_test_passed(self):
        """Test passed method."""
        test_pass = ConformanceTest(result=TestResult.PASS)
        test_warning = ConformanceTest(result=TestResult.WARNING)
        test_fail = ConformanceTest(result=TestResult.FAIL)

        assert test_pass.passed()
        assert test_warning.passed()
        assert not test_fail.passed()

    def test_test_failed(self):
        """Test failed method."""
        test_fail = ConformanceTest(result=TestResult.FAIL)
        test_error = ConformanceTest(result=TestResult.ERROR)
        test_pass = ConformanceTest(result=TestResult.PASS)

        assert test_fail.failed()
        assert test_error.failed()
        assert not test_pass.failed()

    def test_test_is_pending(self):
        """Test is_pending method."""
        pending_test = ConformanceTest(result=TestResult.PENDING)
        completed_test = ConformanceTest(result=TestResult.PASS)

        assert pending_test.is_pending()
        assert not completed_test.is_pending()

    def test_add_evidence(self):
        """Test adding evidence to test."""
        test = ConformanceTest(test_id="CT-001")
        evidence = TestEvidence(description="Test evidence")
        test.add_evidence(evidence)

        assert len(test.evidence) == 1
        assert test.evidence[0].description == "Test evidence"

    def test_test_to_dict(self):
        """Test serialization."""
        test = ConformanceTest(
            test_id="CT-001",
            name="Test Name",
            category=TestCategory.KILL_SWITCH,
            priority=TestPriority.CRITICAL,
            result=TestResult.PASS,
            duration_ms=100.5,
        )
        data = test.to_dict()

        assert data["test_id"] == "CT-001"
        assert data["category"] == "kill_switch"
        assert data["priority"] == "critical"
        assert data["result"] == "pass"
        assert data["duration_ms"] == 100.5

    def test_test_from_dict(self):
        """Test deserialization."""
        data = {
            "test_id": "CT-002",
            "name": "Test from dict",
            "category": "pre_trade",
            "priority": "high",
            "result": "fail",
            "details": "Test failed",
        }
        test = ConformanceTest.from_dict(data)

        assert test.test_id == "CT-002"
        assert test.category == TestCategory.PRE_TRADE
        assert test.priority == TestPriority.HIGH
        assert test.result == TestResult.FAIL

    def test_test_roundtrip(self):
        """Test serialization roundtrip."""
        test = ConformanceTest(
            test_id="CT-003",
            name="Roundtrip Test",
            category=TestCategory.CLOCK_SYNC,
            priority=TestPriority.MEDIUM,
            preconditions=["Condition 1", "Condition 2"],
            test_steps=["Step 1", "Step 2"],
            expected_result="Expected outcome",
        )

        data = test.to_dict()
        restored = ConformanceTest.from_dict(data)

        assert restored.test_id == test.test_id
        assert restored.name == test.name
        assert restored.category == test.category
        assert restored.preconditions == test.preconditions


# =============================================================================
# Test ConformanceTestSuite
# =============================================================================


class TestConformanceTestSuite:
    """Test ConformanceTestSuite dataclass."""

    @pytest.fixture
    def sample_suite(self) -> ConformanceTestSuite:
        """Create sample test suite."""
        suite = ConformanceTestSuite(
            name="Sample Suite",
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
            environment=TestEnvironment.UAT,
            created_by="tester",
        )

        # Add tests
        suite.add_test(ConformanceTest(
            test_id="CT-001",
            name="Critical Test 1",
            category=TestCategory.KILL_SWITCH,
            priority=TestPriority.CRITICAL,
            result=TestResult.PASS,
        ))
        suite.add_test(ConformanceTest(
            test_id="CT-002",
            name="High Test 1",
            category=TestCategory.PRE_TRADE,
            priority=TestPriority.HIGH,
            result=TestResult.PASS,
        ))
        suite.add_test(ConformanceTest(
            test_id="CT-003",
            name="Critical Test 2",
            category=TestCategory.KILL_SWITCH,
            priority=TestPriority.CRITICAL,
            result=TestResult.FAIL,
        ))

        return suite

    def test_create_suite(self):
        """Test creating a test suite."""
        suite = ConformanceTestSuite(
            name="Test Suite",
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
        )
        assert suite.suite_id
        assert suite.name == "Test Suite"
        assert suite.algorithm_id == "ALGO-001"
        assert suite.status == ConformanceSuiteStatus.NOT_STARTED

    def test_add_test(self):
        """Test adding tests to suite."""
        suite = ConformanceTestSuite()
        test = ConformanceTest(test_id="CT-001")
        suite.add_test(test)

        assert len(suite.tests) == 1
        assert suite.tests[0].test_id == "CT-001"

    def test_get_test(self):
        """Test getting test by ID."""
        suite = ConformanceTestSuite()
        test = ConformanceTest(test_id="CT-001")
        suite.add_test(test)

        found = suite.get_test("CT-001")
        assert found is not None
        assert found.test_id == "CT-001"

        not_found = suite.get_test("CT-999")
        assert not_found is None

    def test_get_tests_by_category(self, sample_suite):
        """Test filtering tests by category."""
        kill_switch_tests = sample_suite.get_tests_by_category(TestCategory.KILL_SWITCH)
        assert len(kill_switch_tests) == 2

        pre_trade_tests = sample_suite.get_tests_by_category(TestCategory.PRE_TRADE)
        assert len(pre_trade_tests) == 1

    def test_get_critical_tests(self, sample_suite):
        """Test getting critical tests."""
        critical = sample_suite.get_critical_tests()
        assert len(critical) == 2
        assert all(t.is_critical() for t in critical)

    def test_get_failed_tests(self, sample_suite):
        """Test getting failed tests."""
        failed = sample_suite.get_failed_tests()
        assert len(failed) == 1
        assert failed[0].test_id == "CT-003"

    def test_get_pending_tests(self):
        """Test getting pending tests."""
        suite = ConformanceTestSuite()
        suite.add_test(ConformanceTest(test_id="CT-001", result=TestResult.PENDING))
        suite.add_test(ConformanceTest(test_id="CT-002", result=TestResult.PASS))

        pending = suite.get_pending_tests()
        assert len(pending) == 1
        assert pending[0].test_id == "CT-001"

    def test_get_metrics(self, sample_suite):
        """Test metrics calculation."""
        metrics = sample_suite.get_metrics()

        assert metrics["total_tests"] == 3
        assert metrics["passed"] == 2
        assert metrics["failed"] == 1
        assert metrics["pending"] == 0
        assert metrics["critical_total"] == 2
        assert metrics["critical_passed"] == 1
        assert metrics["critical_failed"] == 1
        assert 66.0 < metrics["pass_rate_pct"] < 67.0  # ~66.67%

    def test_get_category_summary(self, sample_suite):
        """Test category summary."""
        summary = sample_suite.get_category_summary()

        assert "kill_switch" in summary
        assert summary["kill_switch"]["total"] == 2
        assert summary["kill_switch"]["passed"] == 1
        assert summary["kill_switch"]["failed"] == 1

    def test_suite_start(self):
        """Test starting suite execution."""
        suite = ConformanceTestSuite()
        suite.start("tester@example.com")

        assert suite.status == ConformanceSuiteStatus.IN_PROGRESS
        assert suite.executed_by == "tester@example.com"
        assert suite.start_timestamp_ns is not None

    def test_suite_complete_passed(self):
        """Test completing suite with passed status."""
        suite = ConformanceTestSuite()
        suite.add_test(ConformanceTest(
            test_id="CT-001",
            priority=TestPriority.CRITICAL,
            result=TestResult.PASS,
        ))
        suite.complete()

        assert suite.status == ConformanceSuiteStatus.PASSED
        assert suite.end_timestamp_ns is not None

    def test_suite_complete_failed(self, sample_suite):
        """Test completing suite with failed status."""
        sample_suite.complete()
        assert sample_suite.status == ConformanceSuiteStatus.FAILED

    def test_suite_complete_with_warnings(self):
        """Test completing suite with warnings."""
        suite = ConformanceTestSuite()
        suite.add_test(ConformanceTest(
            test_id="CT-001",
            priority=TestPriority.CRITICAL,
            result=TestResult.PASS,
        ))
        suite.add_test(ConformanceTest(
            test_id="CT-002",
            priority=TestPriority.HIGH,
            result=TestResult.WARNING,
        ))
        suite.complete()

        assert suite.status == ConformanceSuiteStatus.PASSED_WITH_WARNINGS

    def test_suite_abort(self):
        """Test aborting suite."""
        suite = ConformanceTestSuite()
        suite.abort("Test aborted")

        assert suite.status == ConformanceSuiteStatus.ABORTED
        assert suite.end_timestamp_ns is not None

    def test_can_certify(self, sample_suite):
        """Test certification eligibility."""
        # Suite with failed critical test cannot certify
        sample_suite.complete()
        assert not sample_suite.can_certify()

        # Suite with all tests passed can certify
        passed_suite = ConformanceTestSuite()
        passed_suite.add_test(ConformanceTest(
            test_id="CT-001",
            priority=TestPriority.CRITICAL,
            result=TestResult.PASS,
        ))
        passed_suite.complete()
        assert passed_suite.can_certify()

    def test_generate_certificate_data(self):
        """Test certificate data generation."""
        suite = ConformanceTestSuite(
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
            environment=TestEnvironment.UAT,
        )
        suite.add_test(ConformanceTest(
            test_id="CT-001",
            priority=TestPriority.CRITICAL,
            result=TestResult.PASS,
        ))
        suite.start("tester")
        suite.complete()

        cert_data = suite.generate_certificate_data()
        assert cert_data is not None
        assert cert_data["algorithm_id"] == "ALGO-001"
        assert cert_data["algorithm_version"] == "1.0.0"
        assert cert_data["critical_tests_passed"] is True

    def test_generate_summary_report(self, sample_suite):
        """Test summary report generation."""
        sample_suite.complete()
        report = sample_suite.generate_summary_report()

        assert "CONFORMANCE TEST SUITE REPORT" in report
        assert "ALGO-001" in report
        assert "TEST RESULTS SUMMARY" in report
        assert "FAILED TESTS" in report

    def test_suite_to_dict(self, sample_suite):
        """Test suite serialization."""
        data = sample_suite.to_dict()

        assert data["name"] == "Sample Suite"
        assert data["algorithm_id"] == "ALGO-001"
        assert len(data["tests"]) == 3

    def test_suite_from_dict(self):
        """Test suite deserialization."""
        data = {
            "suite_id": "test-suite-id",
            "name": "From Dict Suite",
            "algorithm_id": "ALGO-002",
            "algorithm_version": "2.0.0",
            "environment": "staging",
            "status": "passed",
            "tests": [
                {"test_id": "CT-001", "category": "kill_switch", "result": "pass"},
            ],
        }

        suite = ConformanceTestSuite.from_dict(data)
        assert suite.suite_id == "test-suite-id"
        assert suite.name == "From Dict Suite"
        assert suite.environment == TestEnvironment.STAGING
        assert len(suite.tests) == 1

    def test_suite_json_roundtrip(self, sample_suite):
        """Test JSON serialization roundtrip."""
        json_str = sample_suite.to_json()
        restored = ConformanceTestSuite.from_json(json_str)

        assert restored.suite_id == sample_suite.suite_id
        assert restored.name == sample_suite.name
        assert len(restored.tests) == len(sample_suite.tests)


# =============================================================================
# Test ConformanceTestRunner
# =============================================================================


class TestConformanceTestRunner:
    """Test ConformanceTestRunner."""

    def test_create_runner(self):
        """Test creating test runner."""
        runner = ConformanceTestRunner()
        assert runner.config is not None
        assert not runner._running

    def test_create_runner_with_config(self):
        """Test creating runner with custom config."""
        config = TestExecutorConfig(
            timeout_seconds=120.0,
            max_retries=5,
            stop_on_critical_failure=False,
        )
        runner = ConformanceTestRunner(config=config)

        assert runner.config.timeout_seconds == 120.0
        assert runner.config.max_retries == 5
        assert not runner.config.stop_on_critical_failure

    def test_run_suite(self):
        """Test running a test suite."""
        runner = ConformanceTestRunner()
        suite = ConformanceTestSuite(
            name="Test Suite",
            algorithm_id="ALGO-001",
        )
        suite.add_test(ConformanceTest(
            test_id="CT-001",
            name="Test 1",
            category=TestCategory.KILL_SWITCH,
            priority=TestPriority.HIGH,
        ))

        result = runner.run_suite(suite, executed_by="tester")

        assert result.executed_by == "tester"
        assert result.status in {
            ConformanceSuiteStatus.PASSED,
            ConformanceSuiteStatus.PASSED_WITH_WARNINGS,
            ConformanceSuiteStatus.FAILED,
        }

    def test_run_suite_with_category_filter(self):
        """Test running suite with category filter."""
        runner = ConformanceTestRunner()
        suite = ConformanceTestSuite()
        suite.add_test(ConformanceTest(
            test_id="CT-001",
            category=TestCategory.KILL_SWITCH,
        ))
        suite.add_test(ConformanceTest(
            test_id="CT-002",
            category=TestCategory.PRE_TRADE,
        ))

        result = runner.run_suite(
            suite,
            executed_by="tester",
            categories={TestCategory.KILL_SWITCH},
        )

        # Only kill switch test should be executed
        assert result.status != ConformanceSuiteStatus.NOT_STARTED

    def test_run_single_test(self):
        """Test running a single test."""
        runner = ConformanceTestRunner()
        test = ConformanceTest(
            test_id="CT-001",
            name="Single Test",
            category=TestCategory.KILL_SWITCH,
        )

        result = runner.run_single_test(test, "tester")

        assert result.tester == "tester"
        assert result.timestamp is not None
        assert result.duration_ms >= 0

    def test_abort_runner(self):
        """Test aborting runner."""
        runner = ConformanceTestRunner()
        runner.abort()
        assert runner._aborted


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_conformance_suite(self):
        """Test create_conformance_suite factory."""
        suite = create_conformance_suite(
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
            environment=TestEnvironment.STAGING,
            created_by="developer",
            include_standard_tests=True,
        )

        assert suite.algorithm_id == "ALGO-001"
        assert suite.algorithm_version == "1.0.0"
        assert suite.environment == TestEnvironment.STAGING
        assert suite.created_by == "developer"
        assert len(suite.tests) > 0  # Standard tests included

    def test_create_conformance_suite_without_standard_tests(self):
        """Test creating suite without standard tests."""
        suite = create_conformance_suite(
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
            include_standard_tests=False,
        )

        assert len(suite.tests) == 0

    def test_create_conformance_suite_filtered_categories(self):
        """Test creating suite with filtered categories."""
        suite = create_conformance_suite(
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
            categories={TestCategory.KILL_SWITCH},
            include_standard_tests=True,
        )

        # All tests should be kill switch category
        assert all(
            t.category == TestCategory.KILL_SWITCH
            for t in suite.tests
        )

    def test_create_test_runner(self):
        """Test create_test_runner factory."""
        runner = create_test_runner()
        assert isinstance(runner, ConformanceTestRunner)

    def test_create_test_runner_with_config(self):
        """Test create_test_runner with config."""
        config = TestExecutorConfig(timeout_seconds=300.0)
        runner = create_test_runner(config=config)
        assert runner.config.timeout_seconds == 300.0

    def test_get_standard_conformance_tests(self):
        """Test getting standard conformance tests."""
        tests = get_standard_conformance_tests()

        assert len(tests) > 0

        # Check required categories are covered
        categories = {t.category for t in tests}
        assert TestCategory.KILL_SWITCH in categories
        assert TestCategory.PRE_TRADE in categories
        assert TestCategory.CLOCK_SYNC in categories
        assert TestCategory.RECORD_KEEPING in categories

    def test_standard_tests_have_required_fields(self):
        """Test standard tests have all required fields."""
        tests = get_standard_conformance_tests()

        for test in tests:
            assert test.test_id, f"Test missing test_id"
            assert test.name, f"Test {test.test_id} missing name"
            assert test.description, f"Test {test.test_id} missing description"
            assert test.rts_reference, f"Test {test.test_id} missing rts_reference"

    def test_standard_tests_have_critical_priority(self):
        """Test some standard tests are marked critical."""
        tests = get_standard_conformance_tests()
        critical = [t for t in tests if t.priority == TestPriority.CRITICAL]

        assert len(critical) > 0, "No critical tests defined"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests."""

    def test_full_conformance_workflow(self):
        """Test complete conformance testing workflow."""
        # 1. Create suite with standard tests
        suite = create_conformance_suite(
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
            environment=TestEnvironment.UAT,
            created_by="developer",
            include_standard_tests=True,
            categories={TestCategory.KILL_SWITCH},
        )

        # 2. Create runner
        runner = create_test_runner()

        # 3. Run suite
        result = runner.run_suite(suite, executed_by="qa_team")

        # 4. Check results
        assert result.executed_by == "qa_team"
        assert result.status != ConformanceSuiteStatus.NOT_STARTED

        # 5. Get metrics
        metrics = result.get_metrics()
        assert metrics["total_tests"] > 0

        # 6. Generate report
        report = result.generate_summary_report()
        assert "CONFORMANCE TEST SUITE REPORT" in report
        assert "ALGO-001" in report

    def test_certification_eligibility_workflow(self):
        """Test certification eligibility determination."""
        # Create suite with all passing tests
        suite = ConformanceTestSuite(
            algorithm_id="ALGO-001",
            algorithm_version="1.0.0",
        )

        # Add critical tests that pass
        suite.add_test(ConformanceTest(
            test_id="CT-001",
            name="Critical Test",
            category=TestCategory.KILL_SWITCH,
            priority=TestPriority.CRITICAL,
            result=TestResult.PASS,
        ))
        suite.add_test(ConformanceTest(
            test_id="CT-002",
            name="High Priority Test",
            category=TestCategory.PRE_TRADE,
            priority=TestPriority.HIGH,
            result=TestResult.PASS,
        ))

        suite.start("developer")
        suite.complete()

        # Should be eligible for certification
        assert suite.can_certify()

        # Generate certificate data
        cert_data = suite.generate_certificate_data()
        assert cert_data is not None
        assert cert_data["critical_tests_passed"] is True
        assert cert_data["pass_rate_pct"] == 100.0
