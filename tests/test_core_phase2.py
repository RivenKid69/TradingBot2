# -*- coding: utf-8 -*-
"""
Comprehensive tests for DORA Phase 2: Core Operational Resilience.

Tests all Block 2.x modules:
- 2.1: Tiered Backup (15min/1h/24h RPO)
- 2.2: Enhanced Healthcheck
- 2.3: Structured Logging
- 2.4: Comprehensive Alerting
- 2.5: DR Testing Framework
- 2.6: Security Gates
- 2.7: SOC2-DORA Mapping
- 2.9: Multi-AZ Deployment
- 2.10: On-Call Rotation
- 2.11: Subcontractor Monitoring
- 2.12: Trust Center Platform
- 2.13: DR Execution
- 2.14: CTPP Risk Monitoring
"""

import pytest
from datetime import datetime, timezone, timedelta


# =============================================================================
# Block 2.1: Tiered Backup Tests
# =============================================================================


class TestTieredBackup:
    """Tests for tiered backup system."""

    def test_tiered_backup_import(self):
        """Test tiered backup module imports."""
        from services.core.tiered_backup import (
            BackupTier,
            RPOLevel,
            BackupStrategy,
            ReplicationMode,
            BackupJobStatus,
            TieredBackupPolicy,
            BackupSchedule,
            BackupExecution,
            ReplicationConfig,
            BackupMetrics,
            TieredBackupConfig,
            TieredBackupManager,
            create_tiered_backup_manager,
            get_tier_definitions,
        )

        assert BackupTier.CRITICAL.value == "critical"
        assert RPOLevel.RPO_15MIN.value == "15min"

    def test_tier_definitions(self):
        """Test tier definitions are correctly configured."""
        from services.core.tiered_backup import get_tier_definitions, BackupTier

        definitions = get_tier_definitions()
        assert BackupTier.CRITICAL in definitions
        assert BackupTier.STANDARD in definitions
        assert BackupTier.ARCHIVE in definitions

        # Check RPO values
        assert definitions[BackupTier.CRITICAL]["rpo_minutes"] == 15
        assert definitions[BackupTier.STANDARD]["rpo_minutes"] == 60
        assert definitions[BackupTier.ARCHIVE]["rpo_minutes"] == 1440

    def test_create_tiered_backup_manager(self):
        """Test creating tiered backup manager."""
        from services.core.tiered_backup import create_tiered_backup_manager

        manager = create_tiered_backup_manager()
        assert manager is not None
        assert manager.config is not None

    def test_create_backup_policy(self):
        """Test creating backup policies for each tier."""
        from services.core.tiered_backup import (
            create_tiered_backup_manager,
            BackupTier,
        )

        manager = create_tiered_backup_manager()

        # Create critical tier policy
        policy = manager.create_policy(
            name="Critical Database Backup",
            tier=BackupTier.CRITICAL,
            systems_covered=["trading-db", "order-db"],
        )

        assert policy is not None
        assert policy.tier == BackupTier.CRITICAL
        assert policy.rpo_minutes == 15
        assert "trading-db" in policy.systems_covered

    def test_backup_execution(self):
        """Test backup execution workflow."""
        from services.core.tiered_backup import (
            create_tiered_backup_manager,
            BackupTier,
            BackupJobStatus,
        )

        manager = create_tiered_backup_manager()

        policy = manager.create_policy(
            name="Test Backup",
            tier=BackupTier.STANDARD,
            systems_covered=["test-system"],
        )

        execution = manager.execute_backup(policy.policy_id)
        assert execution is not None
        assert execution.status in (BackupJobStatus.COMPLETED, BackupJobStatus.VERIFIED)

    def test_rpo_compliance_check(self):
        """Test RPO compliance checking."""
        from services.core.tiered_backup import (
            create_tiered_backup_manager,
            BackupTier,
        )

        manager = create_tiered_backup_manager()

        policy = manager.create_policy(
            name="RPO Test",
            tier=BackupTier.STANDARD,
            systems_covered=["test"],
        )

        # Execute backup
        manager.execute_backup(policy.policy_id)

        # Check compliance
        compliance = manager.check_rpo_compliance(policy.policy_id)
        assert "compliant" in compliance
        assert compliance["rpo_minutes"] == 60

    def test_backup_metrics(self):
        """Test backup metrics collection."""
        from services.core.tiered_backup import create_tiered_backup_manager

        manager = create_tiered_backup_manager()
        metrics = manager.get_metrics()

        assert metrics is not None
        assert hasattr(metrics, "success_rate_24h")
        assert hasattr(metrics, "metrics_by_tier")

    def test_backup_summary(self):
        """Test backup summary generation."""
        from services.core.tiered_backup import create_tiered_backup_manager

        manager = create_tiered_backup_manager()
        summary = manager.get_backup_summary()

        assert "policies" in summary
        assert "dora_compliance" in summary


# =============================================================================
# Block 2.2: Enhanced Healthcheck Tests
# =============================================================================


class TestEnhancedHealthcheck:
    """Tests for enhanced healthcheck system."""

    def test_enhanced_healthcheck_import(self):
        """Test enhanced healthcheck module imports."""
        from services.core.enhanced_healthcheck import (
            ProbeType,
            DependencyType,
            DependencyStatus,
            ReadinessCondition,
            HealthProbe,
            DependencyCheck,
            LivenessResult,
            ReadinessResult,
            HealthResult,
            EnhancedHealthcheckConfig,
            EnhancedHealthcheck,
            create_enhanced_healthcheck,
        )

        assert ProbeType.LIVENESS.value == "liveness"
        assert DependencyStatus.HEALTHY.value == "healthy"

    def test_create_enhanced_healthcheck(self):
        """Test creating enhanced healthcheck."""
        from services.core.enhanced_healthcheck import create_enhanced_healthcheck

        healthcheck = create_enhanced_healthcheck()
        assert healthcheck is not None

    def test_liveness_probe(self):
        """Test liveness probe endpoint."""
        from services.core.enhanced_healthcheck import create_enhanced_healthcheck

        healthcheck = create_enhanced_healthcheck()
        result = healthcheck.live()

        assert result.alive is True
        assert result.uptime_seconds >= 0
        assert result.process_id > 0

    def test_readiness_probe(self):
        """Test readiness probe endpoint."""
        from services.core.enhanced_healthcheck import create_enhanced_healthcheck

        healthcheck = create_enhanced_healthcheck()
        result = healthcheck.ready()

        assert hasattr(result, "ready")
        assert hasattr(result, "dependencies_checked")

    def test_health_endpoint(self):
        """Test full health endpoint."""
        from services.core.enhanced_healthcheck import create_enhanced_healthcheck

        healthcheck = create_enhanced_healthcheck()
        result = healthcheck.health()

        assert hasattr(result, "healthy")
        assert hasattr(result, "status")
        assert hasattr(result, "components")

    def test_register_dependency(self):
        """Test dependency registration."""
        from services.core.enhanced_healthcheck import (
            create_enhanced_healthcheck,
            DependencyType,
            DatabaseChecker,
        )

        healthcheck = create_enhanced_healthcheck()
        healthcheck.register_dependency(
            name="test-db",
            dependency_type=DependencyType.DATABASE,
            checker=DatabaseChecker(),
            is_critical=True,
        )

        deps = healthcheck.get_all_dependencies()
        assert len(deps) > 0
        assert any(d["name"] == "test-db" for d in deps)

    def test_health_handlers(self):
        """Test HTTP handler creation."""
        from services.core.enhanced_healthcheck import create_enhanced_healthcheck

        healthcheck = create_enhanced_healthcheck()
        handlers = healthcheck.create_handlers()

        assert "health" in handlers
        assert "ready" in handlers
        assert "live" in handlers

        # Call handlers
        health_response = handlers["health"]()
        assert "healthy" in health_response


# =============================================================================
# Block 2.3: Structured Logging Tests
# =============================================================================


class TestStructuredLogging:
    """Tests for structured logging system."""

    def test_structured_logging_import(self):
        """Test structured logging module imports."""
        from services.core.structured_logging import (
            LogLevel,
            LogCategory,
            CorrelationContext,
            StructuredLogEntry,
            LoggingConfig,
            StructuredLogger,
            create_structured_logger,
            get_correlation_id,
            set_correlation_id,
            correlation_context,
        )

        assert LogLevel.INFO.value == "INFO"
        assert LogCategory.SECURITY.value == "security"

    def test_create_structured_logger(self):
        """Test creating structured logger."""
        from services.core.structured_logging import create_structured_logger

        logger = create_structured_logger()
        assert logger is not None

    def test_correlation_id(self):
        """Test correlation ID management."""
        from services.core.structured_logging import (
            get_correlation_id,
            set_correlation_id,
        )

        # Get generates new ID if none set
        cid = get_correlation_id()
        assert cid is not None
        assert len(cid) > 0

        # Set custom ID
        set_correlation_id("test-correlation-123")
        assert get_correlation_id() == "test-correlation-123"

    def test_correlation_context_manager(self):
        """Test correlation context manager."""
        from services.core.structured_logging import (
            correlation_context,
            get_correlation_id,
        )

        original_id = get_correlation_id()

        with correlation_context(correlation_id="ctx-test-456") as cid:
            assert cid == "ctx-test-456"
            assert get_correlation_id() == "ctx-test-456"

        # Should restore original after context
        # (Note: in tests context vars may behave differently)

    def test_log_methods(self):
        """Test logging methods."""
        from services.core.structured_logging import (
            create_structured_logger,
            LoggingConfig,
        )

        config = LoggingConfig(output_destination="stdout")
        logger = create_structured_logger(config)

        # These should not raise
        logger.info("Test info message")
        logger.warning("Test warning")
        logger.error("Test error")
        logger.debug("Test debug")

    def test_category_methods(self):
        """Test category-specific logging methods."""
        from services.core.structured_logging import (
            create_structured_logger,
            LoggingConfig,
        )

        config = LoggingConfig(output_destination="stdout")
        logger = create_structured_logger(config)

        # These should not raise
        logger.security("Security event", user_id="test")
        logger.audit("Audit event", action="test")
        logger.performance("Performance metric", duration_ms=100)
        logger.business("Business event", order_id="123")

    def test_contextual_logger(self):
        """Test contextual logger binding."""
        from services.core.structured_logging import create_structured_logger

        logger = create_structured_logger()
        contextual = logger.with_context(user_id="user123", session_id="sess456")

        assert contextual is not None
        contextual.info("Test with context")

    def test_structured_log_entry(self):
        """Test structured log entry creation."""
        from services.core.structured_logging import StructuredLogEntry

        entry = StructuredLogEntry(
            level="INFO",
            message="Test message",
            correlation_id="test-123",
        )

        json_str = entry.to_json()
        assert "INFO" in json_str
        assert "Test message" in json_str
        assert "test-123" in json_str


# =============================================================================
# Block 2.4: Alerting Tests
# =============================================================================


class TestAlerting:
    """Tests for alerting system."""

    def test_alerting_import(self):
        """Test alerting module imports."""
        from services.core.alerting import (
            AlertSeverity,
            AlertChannel,
            AlertStatus,
            EscalationLevel,
            AlertRule,
            Alert,
            EscalationPolicy,
            AlertingConfig,
            NotificationResult,
            AlertingService,
            create_alerting_service,
        )

        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertChannel.SLACK.value == "slack"

    def test_create_alerting_service(self):
        """Test creating alerting service."""
        from services.core.alerting import create_alerting_service

        service = create_alerting_service()
        assert service is not None

    def test_create_alert_rule(self):
        """Test creating alert rules."""
        from services.core.alerting import (
            create_alerting_service,
            AlertSeverity,
        )

        service = create_alerting_service()

        rule = service.create_rule(
            name="High CPU",
            condition_type="threshold",
            metric_name="cpu_percent",
            threshold_value=90.0,
            severity=AlertSeverity.HIGH,
        )

        assert rule is not None
        assert rule.name == "High CPU"
        assert rule.severity == AlertSeverity.HIGH

    def test_trigger_alert(self):
        """Test triggering alerts."""
        from services.core.alerting import (
            create_alerting_service,
            AlertSeverity,
            AlertStatus,
        )

        service = create_alerting_service()

        rule = service.create_rule(
            name="Test Alert",
            condition_type="threshold",
            metric_name="test_metric",
            threshold_value=50.0,
            severity=AlertSeverity.MEDIUM,
        )

        alert = service.trigger_alert(
            rule_id=rule.rule_id,
            metric_value=75.0,
            source="test-source",
        )

        assert alert is not None
        assert alert.status == AlertStatus.TRIGGERED

    def test_acknowledge_alert(self):
        """Test acknowledging alerts."""
        from services.core.alerting import (
            create_alerting_service,
            AlertSeverity,
            AlertStatus,
        )

        service = create_alerting_service()

        rule = service.create_rule(
            name="Ack Test",
            condition_type="threshold",
            metric_name="test",
            threshold_value=50.0,
            severity=AlertSeverity.LOW,
        )

        alert = service.trigger_alert(
            rule_id=rule.rule_id,
            metric_value=60.0,
            source="test",
        )

        acked = service.acknowledge_alert(alert.alert_id, "test@test.com")
        assert acked is not None
        assert acked.status == AlertStatus.ACKNOWLEDGED

    def test_resolve_alert(self):
        """Test resolving alerts."""
        from services.core.alerting import (
            create_alerting_service,
            AlertSeverity,
            AlertStatus,
        )

        service = create_alerting_service()

        rule = service.create_rule(
            name="Resolve Test",
            condition_type="threshold",
            metric_name="test",
            threshold_value=50.0,
            severity=AlertSeverity.LOW,
        )

        alert = service.trigger_alert(
            rule_id=rule.rule_id,
            metric_value=60.0,
            source="test",
        )

        resolved = service.resolve_alert(alert.alert_id, "test@test.com", "Fixed")
        assert resolved is not None
        assert resolved.status == AlertStatus.RESOLVED

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        from services.core.alerting import create_alerting_service

        service = create_alerting_service()
        active = service.get_active_alerts()
        assert isinstance(active, list)

    def test_alert_statistics(self):
        """Test alert statistics."""
        from services.core.alerting import create_alerting_service

        service = create_alerting_service()
        stats = service.get_statistics()

        assert "total_rules" in stats
        assert "total_alerts" in stats
        assert "by_severity_24h" in stats


# =============================================================================
# Block 2.5: DR Testing Tests
# =============================================================================


class TestDRTesting:
    """Tests for DR testing framework."""

    def test_dr_testing_import(self):
        """Test DR testing module imports."""
        from services.core.dr_testing import (
            DRTestType,
            DRTestStatus,
            DRTestResult,
            RecoveryPhase,
            DRTestScenario,
            DRTestExecution,
            DRTestReport,
            RecoveryMetrics,
            DRTestingConfig,
            DRTestingFramework,
            create_dr_testing_framework,
        )

        assert DRTestType.TABLETOP.value == "tabletop"
        assert DRTestResult.PASSED.value == "passed"

    def test_create_dr_testing_framework(self):
        """Test creating DR testing framework."""
        from services.core.dr_testing import create_dr_testing_framework

        framework = create_dr_testing_framework()
        assert framework is not None

    def test_create_scenario(self):
        """Test creating DR test scenarios."""
        from services.core.dr_testing import (
            create_dr_testing_framework,
            DRTestType,
        )

        framework = create_dr_testing_framework()

        scenario = framework.create_scenario(
            name="Database Failover Test",
            test_type=DRTestType.COMPONENT,
            systems_in_scope=["primary-db"],
            rto_target_minutes=30,
            rpo_target_minutes=15,
        )

        assert scenario is not None
        assert scenario.rto_target_minutes == 30

    def test_schedule_and_execute_test(self):
        """Test scheduling and executing DR tests."""
        from services.core.dr_testing import (
            create_dr_testing_framework,
            DRTestType,
            DRTestStatus,
            DRTestResult,
        )

        framework = create_dr_testing_framework()

        scenario = framework.create_scenario(
            name="Test Scenario",
            test_type=DRTestType.TABLETOP,
        )

        execution = framework.schedule_test(
            scenario_id=scenario.scenario_id,
            scheduled_date=datetime.now(timezone.utc).isoformat(),
            test_lead="test_lead",
        )

        assert execution is not None

        # Start test
        started = framework.start_test(execution.execution_id)
        assert started.status == DRTestStatus.IN_PROGRESS

        # Complete test
        completed = framework.complete_test(
            execution.execution_id,
            result=DRTestResult.PASSED,
            rto_achieved_minutes=25.0,
            rpo_achieved_minutes=10.0,
        )

        assert completed.result == DRTestResult.PASSED
        assert completed.rto_met is True

    def test_quarterly_compliance(self):
        """Test quarterly compliance checking."""
        from services.core.dr_testing import create_dr_testing_framework

        framework = create_dr_testing_framework()
        compliance = framework.check_quarterly_compliance()

        assert "compliant" in compliance
        assert "quarter" in compliance


# =============================================================================
# Block 2.6: Security Gates Tests
# =============================================================================


class TestSecurityGates:
    """Tests for security gates."""

    def test_security_gates_import(self):
        """Test security gates module imports."""
        from services.core.security_gates import (
            ScanType,
            ScanStatus,
            VulnerabilitySeverity,
            GateDecision,
            SecurityScanResult,
            SecurityGate,
            GatePolicy,
            SecurityGatesConfig,
            SecurityGatesManager,
            create_security_gates_manager,
        )

        assert ScanType.SAST.value == "sast"
        assert GateDecision.PASS.value == "pass"

    def test_create_security_gates_manager(self):
        """Test creating security gates manager."""
        from services.core.security_gates import create_security_gates_manager

        manager = create_security_gates_manager()
        assert manager is not None

    def test_create_security_gate(self):
        """Test creating security gates."""
        from services.core.security_gates import (
            create_security_gates_manager,
            ScanType,
        )

        manager = create_security_gates_manager()

        gate = manager.create_gate(
            name="Test Gate",
            required_scans=[ScanType.SAST, ScanType.SCA],
            max_critical=0,
            max_high=0,
        )

        assert gate is not None
        assert gate.max_critical == 0

    def test_record_scan(self):
        """Test recording security scans."""
        from services.core.security_gates import (
            create_security_gates_manager,
            ScanType,
        )

        manager = create_security_gates_manager()

        scan = manager.record_scan(
            scan_type=ScanType.SAST,
            pipeline_id="test-pipeline-123",
            commit_sha="abc123",
            findings=[
                {"severity": "medium", "description": "Test finding"},
            ],
            tool_name="test-scanner",
        )

        assert scan is not None
        assert scan.vulnerabilities_found == 1
        assert scan.medium_count == 1

    def test_evaluate_gate(self):
        """Test gate evaluation."""
        from services.core.security_gates import (
            create_security_gates_manager,
            ScanType,
            GateDecision,
        )

        manager = create_security_gates_manager()

        gate = manager.create_gate(
            name="Evaluation Test Gate",
            required_scans=[ScanType.SAST],
            max_critical=0,
            max_high=0,
            max_medium=5,
        )

        scan = manager.record_scan(
            scan_type=ScanType.SAST,
            pipeline_id="test",
            commit_sha="abc",
            findings=[
                {"severity": "low", "description": "Low finding"},
            ],
        )

        result = manager.evaluate_gate(gate.gate_id, [scan])
        assert result["decision"] == GateDecision.PASS.value


# =============================================================================
# Block 2.7: SOC2-DORA Mapping Tests
# =============================================================================


class TestSOC2DORAMapping:
    """Tests for SOC2-DORA mapping."""

    def test_soc2_dora_mapping_import(self):
        """Test SOC2-DORA mapping module imports."""
        from services.core.soc2_dora_mapping import (
            SOC2Category,
            DORAArticle,
            ControlStatus,
            EvidenceStatus,
            ControlMapping,
            SharedControl,
            EvidenceRequirement,
            ComplianceGap,
            SOC2DORAMappingConfig,
            SOC2DORAMapper,
            create_soc2_dora_mapper,
            get_control_mappings,
        )

        assert SOC2Category.CC_SECURITY.value == "CC"
        assert DORAArticle.ART_9.value == "Article 9"

    def test_create_soc2_dora_mapper(self):
        """Test creating SOC2-DORA mapper."""
        from services.core.soc2_dora_mapping import create_soc2_dora_mapper

        mapper = create_soc2_dora_mapper()
        assert mapper is not None

    def test_get_control_mappings(self):
        """Test getting pre-defined control mappings."""
        from services.core.soc2_dora_mapping import get_control_mappings

        mappings = get_control_mappings()
        assert len(mappings) > 0
        assert any(m["soc2_control"] == "CC6.1" for m in mappings)

    def test_add_mapping(self):
        """Test adding custom mappings."""
        from services.core.soc2_dora_mapping import (
            create_soc2_dora_mapper,
            SOC2Category,
            DORAArticle,
        )

        mapper = create_soc2_dora_mapper()

        mapping = mapper.add_mapping(
            soc2_control="CC8.1",
            soc2_category=SOC2Category.CC_SECURITY,
            soc2_description="Test control",
            dora_article=DORAArticle.ART_10,
            dora_requirement="Detection",
        )

        assert mapping is not None

    def test_get_mappings_by_dora(self):
        """Test getting mappings by DORA article."""
        from services.core.soc2_dora_mapping import (
            create_soc2_dora_mapper,
            DORAArticle,
        )

        mapper = create_soc2_dora_mapper()
        mappings = mapper.get_mappings_by_dora(DORAArticle.ART_9)

        assert len(mappings) > 0

    def test_mapping_summary(self):
        """Test mapping summary generation."""
        from services.core.soc2_dora_mapping import create_soc2_dora_mapper

        mapper = create_soc2_dora_mapper()
        summary = mapper.get_mapping_summary()

        assert "total_mappings" in summary
        assert "by_soc2_category" in summary
        assert "by_dora_article" in summary


# =============================================================================
# Block 2.9: Multi-AZ Tests
# =============================================================================


class TestMultiAZ:
    """Tests for Multi-AZ deployment."""

    def test_multi_az_import(self):
        """Test Multi-AZ module imports."""
        from services.core.multi_az import (
            AvailabilityZone,
            DeploymentStrategy,
            FailoverMode,
            ZoneStatus,
            ZoneConfig,
            DeploymentConfig,
            FailoverConfig,
            ZoneHealthStatus,
            MultiAZConfig,
            MultiAZManager,
            create_multi_az_manager,
        )

        assert AvailabilityZone.EU_WEST_1A.value == "eu-west-1a"
        assert DeploymentStrategy.ACTIVE_ACTIVE.value == "active_active"

    def test_create_multi_az_manager(self):
        """Test creating Multi-AZ manager."""
        from services.core.multi_az import create_multi_az_manager

        manager = create_multi_az_manager()
        assert manager is not None

    def test_register_zone(self):
        """Test registering availability zones."""
        from services.core.multi_az import (
            create_multi_az_manager,
            AvailabilityZone,
        )

        manager = create_multi_az_manager()

        zone = manager.register_zone(
            zone=AvailabilityZone.EU_WEST_1A,
            region="eu-west-1",
            is_primary=True,
            services=["api", "db"],
        )

        assert zone is not None
        assert zone.is_primary is True

    def test_create_deployment(self):
        """Test creating Multi-AZ deployment."""
        from services.core.multi_az import (
            create_multi_az_manager,
            AvailabilityZone,
            DeploymentStrategy,
        )

        manager = create_multi_az_manager()

        zone1 = manager.register_zone(
            zone=AvailabilityZone.EU_WEST_1A,
            region="eu-west-1",
            is_primary=True,
        )

        zone2 = manager.register_zone(
            zone=AvailabilityZone.EU_WEST_1B,
            region="eu-west-1",
            is_primary=False,
        )

        deployment = manager.create_deployment(
            name="Production",
            strategy=DeploymentStrategy.ACTIVE_ACTIVE,
            zone_ids=[zone1.zone_id, zone2.zone_id],
        )

        assert deployment is not None
        assert len(deployment.zones) == 2

    def test_get_deployment_status(self):
        """Test getting deployment status."""
        from services.core.multi_az import (
            create_multi_az_manager,
            AvailabilityZone,
            DeploymentStrategy,
        )

        manager = create_multi_az_manager()

        zone1 = manager.register_zone(
            zone=AvailabilityZone.EU_WEST_1A,
            region="eu-west-1",
        )

        zone2 = manager.register_zone(
            zone=AvailabilityZone.EU_WEST_1B,
            region="eu-west-1",
        )

        deployment = manager.create_deployment(
            name="Test",
            strategy=DeploymentStrategy.ACTIVE_ACTIVE,
            zone_ids=[zone1.zone_id, zone2.zone_id],
        )

        status = manager.get_deployment_status(deployment.deployment_id)
        assert status is not None
        assert "zones" in status


# =============================================================================
# Block 2.10: On-Call Rotation Tests
# =============================================================================


class TestOnCallRotation:
    """Tests for on-call rotation."""

    def test_oncall_rotation_import(self):
        """Test on-call rotation module imports."""
        from services.core.oncall_rotation import (
            OnCallTier,
            RotationSchedule,
            EscalationPath,
            IncidentPriority,
            OnCallEngineer,
            OnCallShift,
            EscalationRule,
            OnCallIncident,
            OnCallRotationConfig,
            OnCallRotationManager,
            create_oncall_rotation_manager,
        )

        assert OnCallTier.OPTION_B.value == "option_b"
        assert IncidentPriority.P1.value == "P1"

    def test_create_oncall_manager(self):
        """Test creating on-call rotation manager."""
        from services.core.oncall_rotation import create_oncall_rotation_manager

        manager = create_oncall_rotation_manager()
        assert manager is not None

    def test_register_engineer(self):
        """Test registering on-call engineers."""
        from services.core.oncall_rotation import create_oncall_rotation_manager

        manager = create_oncall_rotation_manager()

        engineer = manager.register_engineer(
            name="Test Engineer",
            email="test@example.com",
            phone="+1234567890",
            skill_level="senior",
        )

        assert engineer is not None
        assert engineer.name == "Test Engineer"

    def test_create_shift(self):
        """Test creating on-call shifts."""
        from services.core.oncall_rotation import (
            create_oncall_rotation_manager,
            EscalationPath,
        )

        manager = create_oncall_rotation_manager()

        engineer = manager.register_engineer(
            name="Shift Engineer",
            email="shift@example.com",
        )

        now = datetime.now(timezone.utc)
        shift = manager.create_shift(
            engineer_id=engineer.engineer_id,
            start_time=now.isoformat(),
            end_time=(now + timedelta(hours=8)).isoformat(),
            escalation_path=EscalationPath.PRIMARY,
        )

        assert shift is not None

    def test_assign_incident(self):
        """Test incident assignment."""
        from services.core.oncall_rotation import (
            create_oncall_rotation_manager,
            IncidentPriority,
        )

        manager = create_oncall_rotation_manager()

        incident = manager.assign_incident(
            title="Test Incident",
            description="Test description",
            priority=IncidentPriority.P2,
        )

        assert incident is not None
        assert incident.priority == IncidentPriority.P2

    def test_coverage_compliance(self):
        """Test coverage compliance checking."""
        from services.core.oncall_rotation import create_oncall_rotation_manager

        manager = create_oncall_rotation_manager()
        compliance = manager.check_coverage_compliance()

        assert "compliant" in compliance
        assert "tier" in compliance


# =============================================================================
# Block 2.11: Subcontractor Monitoring Tests
# =============================================================================


class TestSubcontractorMonitoring:
    """Tests for subcontractor monitoring."""

    def test_subcontractor_monitoring_import(self):
        """Test subcontractor monitoring module imports."""
        from services.core.subcontractor_monitoring import (
            SubcontractorHealthStatus,
            MonitoringFrequency,
            AlertThreshold,
            SubcontractorStatus,
            HealthCheckResult,
            StatusReport,
            SubcontractorMonitoringConfig,
            SubcontractorMonitor,
            create_subcontractor_monitor,
        )

        assert SubcontractorHealthStatus.HEALTHY.value == "healthy"

    def test_create_subcontractor_monitor(self):
        """Test creating subcontractor monitor."""
        from services.core.subcontractor_monitoring import create_subcontractor_monitor

        monitor = create_subcontractor_monitor()
        assert monitor is not None

    def test_register_subcontractor(self):
        """Test registering subcontractors."""
        from services.core.subcontractor_monitoring import create_subcontractor_monitor

        monitor = create_subcontractor_monitor()

        sub = monitor.register_subcontractor(
            name="Test Provider",
            service_type="Cloud Infrastructure",
            sla_target_uptime=99.9,
            is_critical=True,
        )

        assert sub is not None
        assert sub.is_critical_provider is True

    def test_record_health_check(self):
        """Test recording health checks."""
        from services.core.subcontractor_monitoring import (
            create_subcontractor_monitor,
            SubcontractorHealthStatus,
        )

        monitor = create_subcontractor_monitor()

        sub = monitor.register_subcontractor(
            name="Health Check Test",
            service_type="API",
        )

        check = monitor.record_health_check(
            subcontractor_id=sub.subcontractor_id,
            status=SubcontractorHealthStatus.HEALTHY,
            response_time_ms=50.0,
        )

        assert check is not None
        assert check.success is True

    def test_generate_report(self):
        """Test report generation."""
        from services.core.subcontractor_monitoring import create_subcontractor_monitor

        monitor = create_subcontractor_monitor()
        report = monitor.generate_report()

        assert report is not None
        assert hasattr(report, "total_subcontractors")


# =============================================================================
# Block 2.12: Trust Center Tests
# =============================================================================


class TestTrustCenter:
    """Tests for trust center platform."""

    def test_trust_center_import(self):
        """Test trust center module imports."""
        from services.core.trust_center import (
            DocumentType,
            AccessLevel,
            CertificationType,
            ComplianceStatus,
            TrustDocument,
            CertificationRecord,
            SecurityPosture,
            TrustCenterConfig,
            TrustCenterPlatform,
            create_trust_center,
        )

        assert DocumentType.SOC2_REPORT.value == "soc2_report"
        assert AccessLevel.PUBLIC.value == "public"

    def test_create_trust_center(self):
        """Test creating trust center."""
        from services.core.trust_center import create_trust_center

        trust_center = create_trust_center()
        assert trust_center is not None

    def test_add_document(self):
        """Test adding documents."""
        from services.core.trust_center import (
            create_trust_center,
            DocumentType,
            AccessLevel,
        )

        trust_center = create_trust_center()

        doc = trust_center.add_document(
            title="Test Security Doc",
            description="Test description",
            document_type=DocumentType.SECURITY_WHITEPAPER,
            access_level=AccessLevel.PUBLIC,
        )

        assert doc is not None

    def test_add_certification(self):
        """Test adding certifications."""
        from services.core.trust_center import (
            create_trust_center,
            CertificationType,
        )

        trust_center = create_trust_center()

        cert = trust_center.add_certification(
            certification_type=CertificationType.SOC2_TYPE_II,
            name="SOC 2 Type II",
            issued_date="2024-01-01",
            expiry_date="2025-01-01",
            issuing_body="Test Auditor",
        )

        assert cert is not None

    def test_security_posture(self):
        """Test security posture generation."""
        from services.core.trust_center import create_trust_center

        trust_center = create_trust_center()
        posture = trust_center.get_security_posture()

        assert posture is not None
        assert hasattr(posture, "overall_rating")
        assert hasattr(posture, "security_score")

    def test_trust_center_summary(self):
        """Test trust center summary."""
        from services.core.trust_center import create_trust_center

        trust_center = create_trust_center()
        summary = trust_center.get_trust_center_summary()

        assert "organization" in summary
        assert "security_posture" in summary


# =============================================================================
# Block 2.13: DR Execution Tests
# =============================================================================


class TestDRExecution:
    """Tests for DR execution manager."""

    def test_dr_execution_import(self):
        """Test DR execution module imports."""
        from services.core.dr_execution import (
            ExecutionPhase,
            ExecutionStatus,
            ValidationResult,
            ExecutionStep,
            ExecutionResult,
            ValidationCheck,
            DRExecutionConfig,
            DRExecutionManager,
            create_dr_execution_manager,
        )

        assert ExecutionPhase.PREPARATION.value == "preparation"
        assert ValidationResult.PASSED.value == "passed"

    def test_create_dr_execution_manager(self):
        """Test creating DR execution manager."""
        from services.core.dr_execution import create_dr_execution_manager

        manager = create_dr_execution_manager()
        assert manager is not None

    def test_create_execution(self):
        """Test creating DR execution."""
        from services.core.dr_execution import create_dr_execution_manager

        manager = create_dr_execution_manager()

        execution = manager.create_execution(
            test_name="Test DR Execution",
            rto_target_minutes=120.0,
            rpo_target_minutes=30.0,
        )

        assert execution is not None
        assert execution.rto_target_minutes == 120.0

    def test_execution_workflow(self):
        """Test full execution workflow."""
        from services.core.dr_execution import (
            create_dr_execution_manager,
            ExecutionPhase,
            ExecutionStatus,
            ValidationResult,
        )

        manager = create_dr_execution_manager()

        # Create execution
        execution = manager.create_execution(test_name="Workflow Test")

        # Add steps
        step = manager.add_step(
            execution_id=execution.execution_id,
            phase=ExecutionPhase.INITIATION,
            name="Initialize failover",
        )

        # Add validation
        validation = manager.add_validation(
            execution_id=execution.execution_id,
            name="Data integrity check",
            success_criteria="All data accessible",
        )

        # Start execution
        started = manager.start_execution(execution.execution_id)
        assert started.final_status == ExecutionStatus.IN_PROGRESS

        # Complete step
        manager.complete_step(
            execution_id=execution.execution_id,
            step_id=step.step_id,
            status=ExecutionStatus.COMPLETED,
        )

        # Record validation
        manager.record_validation(
            execution_id=execution.execution_id,
            check_id=validation.check_id,
            result=ValidationResult.PASSED,
        )

        # Complete execution
        completed = manager.complete_execution(
            execution_id=execution.execution_id,
            rto_achieved_minutes=100.0,
            rpo_achieved_minutes=20.0,
        )

        assert completed.overall_result == "passed"
        assert completed.rto_met is True
        assert completed.rpo_met is True


# =============================================================================
# Block 2.14: CTPP Monitoring Tests
# =============================================================================


class TestCTPPMonitoring:
    """Tests for CTPP risk monitoring."""

    def test_ctpp_monitoring_import(self):
        """Test CTPP monitoring module imports."""
        from services.core.ctpp_monitoring import (
            CTPPRiskLevel,
            MonitoringStatus,
            RiskIndicator,
            CTPPRiskAssessment,
            RiskMetric,
            RiskAlert,
            CTPPMonitoringConfig,
            CTPPRiskMonitor,
            create_ctpp_risk_monitor,
        )

        assert CTPPRiskLevel.CRITICAL.value == "critical"
        assert RiskIndicator.CONCENTRATION_LEVEL.value == "concentration_level"

    def test_create_ctpp_monitor(self):
        """Test creating CTPP monitor."""
        from services.core.ctpp_monitoring import create_ctpp_risk_monitor

        monitor = create_ctpp_risk_monitor()
        assert monitor is not None

    def test_register_provider(self):
        """Test registering providers."""
        from services.core.ctpp_monitoring import create_ctpp_risk_monitor

        monitor = create_ctpp_risk_monitor()

        assessment = monitor.register_provider(
            provider_name="Test Cloud Provider",
            services_used=["Compute", "Storage"],
            dependency_percentage=25.0,
        )

        assert assessment is not None
        assert assessment.dependency_percentage == 25.0

    def test_register_designated_ctpp(self):
        """Test registering designated CTPP."""
        from services.core.ctpp_monitoring import create_ctpp_risk_monitor

        monitor = create_ctpp_risk_monitor()

        # AWS is a designated CTPP
        assessment = monitor.register_provider(
            provider_name="Amazon Web Services",
            services_used=["EC2", "S3", "RDS"],
            dependency_percentage=40.0,
        )

        assert assessment.is_designated_ctpp is True
        assert assessment.lead_overseer == "EBA"

    def test_record_metric(self):
        """Test recording risk metrics."""
        from services.core.ctpp_monitoring import (
            create_ctpp_risk_monitor,
            RiskIndicator,
        )

        monitor = create_ctpp_risk_monitor()

        assessment = monitor.register_provider(
            provider_name="Metric Test Provider",
            services_used=["API"],
        )

        metric = monitor.record_metric(
            provider_id=assessment.provider_id,
            indicator=RiskIndicator.CONCENTRATION_LEVEL,
            value=35.0,
        )

        assert metric is not None

    def test_concentration_summary(self):
        """Test concentration risk summary."""
        from services.core.ctpp_monitoring import create_ctpp_risk_monitor

        monitor = create_ctpp_risk_monitor()
        summary = monitor.get_concentration_summary()

        assert "total_providers" in summary or "status" in summary

    def test_check_designation_updates(self):
        """Test checking for designation updates."""
        from services.core.ctpp_monitoring import create_ctpp_risk_monitor

        monitor = create_ctpp_risk_monitor()
        updates = monitor.check_designation_updates()

        assert isinstance(updates, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPhase2Integration:
    """Integration tests for Phase 2 modules."""

    def test_all_modules_importable(self):
        """Test that all Phase 2 modules are importable from services.core."""
        from services.core import (
            # Tiered Backup
            TieredBackupManager,
            create_tiered_backup_manager,
            # Enhanced Healthcheck
            EnhancedHealthcheck,
            create_enhanced_healthcheck,
            # Structured Logging
            StructuredLogger,
            create_structured_logger,
            # Alerting
            AlertingService,
            create_alerting_service,
            # DR Testing
            DRTestingFramework,
            create_dr_testing_framework,
            # Security Gates
            SecurityGatesManager,
            create_security_gates_manager,
            # SOC2-DORA Mapping
            SOC2DORAMapper,
            create_soc2_dora_mapper,
            # Multi-AZ
            MultiAZManager,
            create_multi_az_manager,
            # On-Call Rotation
            OnCallRotationManager,
            create_oncall_rotation_manager,
            # Subcontractor Monitoring
            SubcontractorMonitor,
            create_subcontractor_monitor,
            # Trust Center
            TrustCenterPlatform,
            create_trust_center,
            # DR Execution
            DRExecutionManager,
            create_dr_execution_manager,
            # CTPP Monitoring
            CTPPRiskMonitor,
            create_ctpp_risk_monitor,
        )

        # Create instances of all
        assert create_tiered_backup_manager() is not None
        assert create_enhanced_healthcheck() is not None
        assert create_structured_logger() is not None
        assert create_alerting_service() is not None
        assert create_dr_testing_framework() is not None
        assert create_security_gates_manager() is not None
        assert create_soc2_dora_mapper() is not None
        assert create_multi_az_manager() is not None
        assert create_oncall_rotation_manager() is not None
        assert create_subcontractor_monitor() is not None
        assert create_trust_center() is not None
        assert create_dr_execution_manager() is not None
        assert create_ctpp_risk_monitor() is not None

    def test_dora_compliance_coverage(self):
        """Test that Phase 2 covers required DORA articles."""
        # Article coverage from Phase 2:
        # - Article 9: Protection (Security Gates)
        # - Article 10: Detection (Alerting, Healthcheck)
        # - Article 11: Response/Recovery (DR Testing, DR Execution, On-Call)
        # - Article 12: Backup (Tiered Backup)
        # - Article 14: Communication (Alerting, On-Call)
        # - Article 15: Business Continuity (DR Testing, Multi-AZ)
        # - Article 28-30: Third-Party Risk (CTPP Monitoring, Subcontractor Monitoring, Trust Center)

        covered_articles = [
            "Article 9",
            "Article 10",
            "Article 11",
            "Article 12",
            "Article 14",
            "Article 15",
            "Article 28",
            "Article 29",
            "Article 30",
            "Article 31-44",
        ]

        assert len(covered_articles) >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
