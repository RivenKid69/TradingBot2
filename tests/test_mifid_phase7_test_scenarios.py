# -*- coding: utf-8 -*-
"""
Tests for MiFID II Phase 7 - Test Scenarios (RTS 6 Articles 5-8).

Comprehensive test coverage for:
- ScenarioStep dataclass
- TestScenario functionality
- ScenarioExecutor execution
- Standard scenario templates
"""

import json
import time
import pytest
from datetime import datetime, date

from services.compliance.test_scenarios import (
    # Enums
    ScenarioType,
    ScenarioSeverity,
    ExecutionPhase,
    ScenarioStatus,
    # Data classes
    ScenarioStep,
    TestScenario,
    # Executor
    ScenarioExecutor,
    # Factory functions
    create_test_scenario,
    create_scenario_executor,
    # Standard scenarios
    get_kill_switch_scenarios,
    get_pre_trade_scenarios,
    get_stress_test_scenarios,
    get_business_continuity_scenarios,
    get_all_standard_scenarios,
)
from services.compliance.conformance_testing import (
    TestCategory,
    TestEnvironment,
    TestResult,
)


# =============================================================================
# Test Enums
# =============================================================================


class TestEnums:
    """Test enum definitions."""

    def test_scenario_type_values(self):
        """Test ScenarioType enum values."""
        assert ScenarioType.FUNCTIONAL.value == "functional"
        assert ScenarioType.STRESS.value == "stress"
        assert ScenarioType.INTEGRATION.value == "integration"
        assert ScenarioType.NEGATIVE.value == "negative"
        assert ScenarioType.DISASTER_RECOVERY.value == "disaster_recovery"

    def test_scenario_severity_values(self):
        """Test ScenarioSeverity enum values."""
        assert ScenarioSeverity.CRITICAL.value == "critical"
        assert ScenarioSeverity.MAJOR.value == "major"
        assert ScenarioSeverity.MINOR.value == "minor"

    def test_execution_phase_values(self):
        """Test ExecutionPhase enum values."""
        assert ExecutionPhase.SETUP.value == "setup"
        assert ExecutionPhase.EXECUTION.value == "execution"
        assert ExecutionPhase.VERIFICATION.value == "verification"
        assert ExecutionPhase.TEARDOWN.value == "teardown"

    def test_scenario_status_values(self):
        """Test ScenarioStatus enum values."""
        assert ScenarioStatus.PENDING.value == "pending"
        assert ScenarioStatus.SETUP_IN_PROGRESS.value == "setup_in_progress"
        assert ScenarioStatus.EXECUTING.value == "executing"
        assert ScenarioStatus.PASSED.value == "passed"
        assert ScenarioStatus.FAILED.value == "failed"


# =============================================================================
# Test ScenarioStep
# =============================================================================


class TestScenarioStep:
    """Test ScenarioStep dataclass."""

    def test_create_step(self):
        """Test creating a scenario step."""
        step = ScenarioStep(
            step_id="STEP-001",
            description="Test step description",
            action="Perform test action",
            expected_result="Expected outcome",
        )
        assert step.step_id == "STEP-001"
        assert step.description == "Test step description"
        assert step.action == "Perform test action"
        assert step.expected_result == "Expected outcome"
        assert step.passed is None

    def test_step_to_dict(self):
        """Test step serialization."""
        step = ScenarioStep(
            step_id="STEP-001",
            description="Test",
            action="Do something",
            expected_result="Something happens",
            actual_result="It happened",
            passed=True,
            duration_ms=50.5,
        )
        data = step.to_dict()

        assert data["step_id"] == "STEP-001"
        assert data["passed"] is True
        assert data["duration_ms"] == 50.5

    def test_step_from_dict(self):
        """Test step deserialization."""
        data = {
            "step_id": "STEP-002",
            "description": "From dict",
            "action": "Test action",
            "expected_result": "Expected",
            "actual_result": "Actual",
            "passed": False,
        }
        step = ScenarioStep.from_dict(data)

        assert step.step_id == "STEP-002"
        assert step.passed is False


# =============================================================================
# Test TestScenario
# =============================================================================


class TestTestScenario:
    """Test TestScenario dataclass."""

    @pytest.fixture
    def sample_scenario(self) -> TestScenario:
        """Create sample test scenario."""
        scenario = TestScenario(
            name="Sample Scenario",
            scenario_type=ScenarioType.FUNCTIONAL,
            category=TestCategory.KILL_SWITCH,
            severity=ScenarioSeverity.CRITICAL,
            description="Sample scenario for testing",
            rts_reference="RTS 6 Article 12",
            objective="Test kill switch functionality",
            preconditions=["System running", "Test mode enabled"],
            pass_criteria=["All orders cancelled", "Audit logged"],
            created_by="developer",
        )

        # Add steps
        scenario.add_setup_step(ScenarioStep(
            step_id="S1",
            description="Setup",
            action="Initialize system",
            expected_result="System ready",
        ))
        scenario.add_execution_step(ScenarioStep(
            step_id="E1",
            description="Execute",
            action="Trigger kill switch",
            expected_result="Kill switch activated",
        ))
        scenario.add_verification_step(ScenarioStep(
            step_id="V1",
            description="Verify",
            action="Check orders cancelled",
            expected_result="All orders cancelled",
        ))
        scenario.add_teardown_step(ScenarioStep(
            step_id="T1",
            description="Teardown",
            action="Reset system",
            expected_result="System reset",
        ))

        return scenario

    def test_create_scenario(self):
        """Test creating a scenario."""
        scenario = TestScenario(
            name="Test Scenario",
            scenario_type=ScenarioType.STRESS,
            category=TestCategory.STRESS_TEST,
            severity=ScenarioSeverity.MAJOR,
        )
        assert scenario.scenario_id
        assert scenario.name == "Test Scenario"
        assert scenario.scenario_type == ScenarioType.STRESS
        assert scenario.status == ScenarioStatus.PENDING

    def test_is_critical(self, sample_scenario):
        """Test is_critical method."""
        assert sample_scenario.is_critical()

        minor_scenario = TestScenario(severity=ScenarioSeverity.MINOR)
        assert not minor_scenario.is_critical()

    def test_passed(self):
        """Test passed method."""
        passed_scenario = TestScenario(status=ScenarioStatus.PASSED)
        failed_scenario = TestScenario(status=ScenarioStatus.FAILED)

        assert passed_scenario.passed()
        assert not failed_scenario.passed()

    def test_failed(self):
        """Test failed method."""
        failed_scenario = TestScenario(status=ScenarioStatus.FAILED)
        error_scenario = TestScenario(status=ScenarioStatus.ERROR)

        assert failed_scenario.failed()
        assert error_scenario.failed()

    def test_is_complete(self):
        """Test is_complete method."""
        complete_statuses = [
            ScenarioStatus.PASSED,
            ScenarioStatus.FAILED,
            ScenarioStatus.ERROR,
            ScenarioStatus.SKIPPED,
            ScenarioStatus.ABORTED,
        ]
        incomplete_statuses = [
            ScenarioStatus.PENDING,
            ScenarioStatus.SETUP_IN_PROGRESS,
            ScenarioStatus.EXECUTING,
        ]

        for status in complete_statuses:
            scenario = TestScenario(status=status)
            assert scenario.is_complete(), f"{status} should be complete"

        for status in incomplete_statuses:
            scenario = TestScenario(status=status)
            assert not scenario.is_complete(), f"{status} should not be complete"

    def test_all_steps(self, sample_scenario):
        """Test all_steps method."""
        all_steps = sample_scenario.all_steps()
        assert len(all_steps) == 4
        step_ids = [s.step_id for s in all_steps]
        assert "S1" in step_ids
        assert "E1" in step_ids
        assert "V1" in step_ids
        assert "T1" in step_ids

    def test_add_steps(self):
        """Test adding steps to scenario."""
        scenario = TestScenario()

        scenario.add_setup_step(ScenarioStep(step_id="S1"))
        scenario.add_execution_step(ScenarioStep(step_id="E1"))
        scenario.add_verification_step(ScenarioStep(step_id="V1"))
        scenario.add_teardown_step(ScenarioStep(step_id="T1"))

        assert len(scenario.setup_steps) == 1
        assert len(scenario.execution_steps) == 1
        assert len(scenario.verification_steps) == 1
        assert len(scenario.teardown_steps) == 1

    def test_start(self):
        """Test starting scenario."""
        scenario = TestScenario()
        scenario.start("tester@example.com")

        assert scenario.executed_by == "tester@example.com"
        assert scenario.start_timestamp_ns is not None
        assert scenario.status == ScenarioStatus.SETUP_IN_PROGRESS

    def test_complete_passed(self):
        """Test completing scenario as passed."""
        scenario = TestScenario()
        scenario.complete(passed=True, summary="All tests passed")

        assert scenario.status == ScenarioStatus.PASSED
        assert scenario.result_summary == "All tests passed"
        assert scenario.end_timestamp_ns is not None

    def test_complete_failed(self):
        """Test completing scenario as failed."""
        scenario = TestScenario()
        scenario.complete(passed=False, summary="Test failed")

        assert scenario.status == ScenarioStatus.FAILED
        assert scenario.result_summary == "Test failed"

    def test_abort(self):
        """Test aborting scenario."""
        scenario = TestScenario()
        scenario.abort("Aborted by user")

        assert scenario.status == ScenarioStatus.ABORTED
        assert scenario.result_summary == "Aborted by user"

    def test_error(self):
        """Test marking scenario as errored."""
        scenario = TestScenario()
        scenario.error("Exception occurred")

        assert scenario.status == ScenarioStatus.ERROR
        assert scenario.result_summary == "Exception occurred"

    def test_skip(self):
        """Test skipping scenario."""
        scenario = TestScenario()
        scenario.skip("Not applicable")

        assert scenario.status == ScenarioStatus.SKIPPED
        assert scenario.result_summary == "Not applicable"

    def test_evaluate_all_passed(self):
        """Test evaluation when all steps pass."""
        scenario = TestScenario()
        scenario.add_execution_step(ScenarioStep(
            step_id="E1",
            passed=True,
        ))
        scenario.add_verification_step(ScenarioStep(
            step_id="V1",
            passed=True,
        ))

        assert scenario.evaluate() is True

    def test_evaluate_with_failure(self):
        """Test evaluation when a step fails."""
        scenario = TestScenario()
        scenario.add_execution_step(ScenarioStep(
            step_id="E1",
            passed=True,
        ))
        scenario.add_verification_step(ScenarioStep(
            step_id="V1",
            passed=False,
        ))

        assert scenario.evaluate() is False

    def test_to_conformance_test(self, sample_scenario):
        """Test converting scenario to conformance test."""
        sample_scenario.status = ScenarioStatus.PASSED
        sample_scenario.start_timestamp_ns = time.time_ns()

        test = sample_scenario.to_conformance_test()

        assert test.name == sample_scenario.name
        assert test.category == sample_scenario.category
        assert test.result == TestResult.PASS

    def test_get_duration_ms(self):
        """Test duration calculation."""
        scenario = TestScenario()
        scenario.start_timestamp_ns = 1000000000000
        scenario.end_timestamp_ns = 1000100000000  # 100ms = 100,000,000 ns later

        duration = scenario.get_duration_ms()
        assert duration == 100.0  # 100ms difference

    def test_to_dict(self, sample_scenario):
        """Test scenario serialization."""
        data = sample_scenario.to_dict()

        assert data["name"] == "Sample Scenario"
        assert data["scenario_type"] == "functional"
        assert data["category"] == "kill_switch"
        assert data["severity"] == "critical"
        assert len(data["setup_steps"]) == 1
        assert len(data["execution_steps"]) == 1

    def test_from_dict(self):
        """Test scenario deserialization."""
        data = {
            "scenario_id": "test-id",
            "name": "From Dict Scenario",
            "scenario_type": "stress",
            "category": "stress_test",
            "severity": "major",
            "status": "passed",
            "setup_steps": [{"step_id": "S1", "action": "Setup"}],
            "execution_steps": [{"step_id": "E1", "action": "Execute"}],
        }

        scenario = TestScenario.from_dict(data)

        assert scenario.scenario_id == "test-id"
        assert scenario.name == "From Dict Scenario"
        assert scenario.scenario_type == ScenarioType.STRESS
        assert scenario.status == ScenarioStatus.PASSED
        assert len(scenario.setup_steps) == 1
        assert len(scenario.execution_steps) == 1

    def test_json_roundtrip(self, sample_scenario):
        """Test JSON serialization roundtrip."""
        json_str = sample_scenario.to_json()
        restored = TestScenario.from_dict(json.loads(json_str))

        assert restored.name == sample_scenario.name
        assert restored.scenario_type == sample_scenario.scenario_type
        assert len(restored.all_steps()) == len(sample_scenario.all_steps())


# =============================================================================
# Test ScenarioExecutor
# =============================================================================


class TestScenarioExecutor:
    """Test ScenarioExecutor."""

    def test_create_executor(self):
        """Test creating scenario executor."""
        executor = ScenarioExecutor()
        assert executor.timeout_seconds == 300.0
        assert executor.stop_on_step_failure is True

    def test_create_executor_with_params(self):
        """Test creating executor with parameters."""
        executor = ScenarioExecutor(
            timeout_seconds=600.0,
            stop_on_step_failure=False,
        )
        assert executor.timeout_seconds == 600.0
        assert executor.stop_on_step_failure is False

    def test_execute_scenario(self):
        """Test executing a scenario."""
        executor = ScenarioExecutor()
        scenario = TestScenario(
            name="Test Scenario",
            category=TestCategory.KILL_SWITCH,
        )
        scenario.add_execution_step(ScenarioStep(
            step_id="E1",
            action="Test action",
            expected_result="Expected",
        ))
        scenario.add_verification_step(ScenarioStep(
            step_id="V1",
            action="Verify",
            expected_result="Verified",
        ))

        result = executor.execute_scenario(scenario, "tester")

        assert result.executed_by == "tester"
        assert result.is_complete()

    def test_abort_executor(self):
        """Test aborting executor."""
        executor = ScenarioExecutor()
        executor.abort()
        assert executor._aborted


# =============================================================================
# Test Standard Scenario Templates
# =============================================================================


class TestStandardScenarios:
    """Test standard scenario templates."""

    def test_get_kill_switch_scenarios(self):
        """Test kill switch scenarios."""
        scenarios = get_kill_switch_scenarios()

        assert len(scenarios) >= 2
        for scenario in scenarios:
            assert scenario.category == TestCategory.KILL_SWITCH
            assert scenario.rts_reference  # Should have RTS reference

    def test_get_pre_trade_scenarios(self):
        """Test pre-trade scenarios."""
        scenarios = get_pre_trade_scenarios()

        assert len(scenarios) >= 2
        for scenario in scenarios:
            assert scenario.category == TestCategory.PRE_TRADE

    def test_get_stress_test_scenarios(self):
        """Test stress test scenarios."""
        scenarios = get_stress_test_scenarios()

        assert len(scenarios) >= 2
        for scenario in scenarios:
            assert scenario.category == TestCategory.STRESS_TEST
            assert "RTS 6 Article 8" in scenario.rts_reference

    def test_get_business_continuity_scenarios(self):
        """Test business continuity scenarios."""
        scenarios = get_business_continuity_scenarios()

        assert len(scenarios) >= 1
        for scenario in scenarios:
            assert scenario.category == TestCategory.BUSINESS_CONTINUITY
            assert "RTS 6 Article 3" in scenario.rts_reference

    def test_get_all_standard_scenarios(self):
        """Test getting all standard scenarios."""
        scenarios = get_all_standard_scenarios()

        # Should include scenarios from all categories
        categories = {s.category for s in scenarios}
        assert TestCategory.KILL_SWITCH in categories
        assert TestCategory.PRE_TRADE in categories
        assert TestCategory.STRESS_TEST in categories
        assert TestCategory.BUSINESS_CONTINUITY in categories

    def test_scenarios_have_steps(self):
        """Test that standard scenarios have steps defined."""
        scenarios = get_all_standard_scenarios()

        for scenario in scenarios:
            # Each scenario should have at least execution steps
            assert len(scenario.execution_steps) > 0, (
                f"Scenario {scenario.name} has no execution steps"
            )

    def test_scenarios_have_pass_criteria(self):
        """Test that standard scenarios have pass criteria."""
        scenarios = get_all_standard_scenarios()

        for scenario in scenarios:
            assert len(scenario.pass_criteria) > 0, (
                f"Scenario {scenario.name} has no pass criteria"
            )


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_test_scenario(self):
        """Test create_test_scenario factory."""
        scenario = create_test_scenario(
            name="Custom Scenario",
            category=TestCategory.INTEGRATION,
            severity=ScenarioSeverity.MAJOR,
            rts_reference="RTS 6 Article 5",
            created_by="developer",
        )

        assert scenario.name == "Custom Scenario"
        assert scenario.category == TestCategory.INTEGRATION
        assert scenario.severity == ScenarioSeverity.MAJOR
        assert scenario.rts_reference == "RTS 6 Article 5"
        assert scenario.created_by == "developer"

    def test_create_scenario_executor(self):
        """Test create_scenario_executor factory."""
        executor = create_scenario_executor(timeout_seconds=600.0)

        assert isinstance(executor, ScenarioExecutor)
        assert executor.timeout_seconds == 600.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests."""

    def test_full_scenario_execution_workflow(self):
        """Test complete scenario execution workflow."""
        # 1. Create scenario
        scenario = create_test_scenario(
            name="Integration Test Scenario",
            category=TestCategory.KILL_SWITCH,
            severity=ScenarioSeverity.CRITICAL,
            rts_reference="RTS 6 Article 12",
        )

        # 2. Add steps
        scenario.add_setup_step(ScenarioStep(
            step_id="S1",
            description="Initialize test environment",
            action="Start test mode",
            expected_result="Test mode active",
        ))
        scenario.add_execution_step(ScenarioStep(
            step_id="E1",
            description="Trigger kill switch",
            action="Call kill_switch.trigger()",
            expected_result="Kill switch activated",
        ))
        scenario.add_verification_step(ScenarioStep(
            step_id="V1",
            description="Verify orders cancelled",
            action="Check order status",
            expected_result="All orders in CANCELLED state",
        ))
        scenario.add_teardown_step(ScenarioStep(
            step_id="T1",
            description="Reset system",
            action="Deactivate test mode",
            expected_result="System reset to normal",
        ))

        scenario.pass_criteria = [
            "Kill switch activates within 1 second",
            "All orders cancelled",
            "Audit trail recorded",
        ]

        # 3. Execute scenario
        executor = create_scenario_executor()
        result = executor.execute_scenario(scenario, "qa_team")

        # 4. Check results
        assert result.executed_by == "qa_team"
        assert result.is_complete()
        assert result.start_timestamp_ns is not None
        assert result.end_timestamp_ns is not None

    def test_standard_scenarios_executable(self):
        """Test that all standard scenarios can be executed."""
        executor = create_scenario_executor()
        scenarios = get_all_standard_scenarios()

        for scenario in scenarios[:3]:  # Test first 3 for speed
            result = executor.execute_scenario(scenario, "test_runner")
            assert result.is_complete()
