# -*- coding: utf-8 -*-
"""
Tests for Research Job Database Models.

CCEA Phase 10: Data models for research job management.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from packages.cloud.research.models import (
    # Enums
    ResearchJobState,
    JobTerminationReason,
    IsolationLevel,
    QuotaTier,
    AbuseType,
    AlertSeverity,
    # Models
    ResearchJob,
    JobQuota,
    QuotaUsageRecord,
    EgressPolicyRecord,
    EgressViolationRecord,
    AbuseIncident,
    JobArtifact,
    # Helpers
    create_research_job,
    create_job_quota,
)


class TestResearchJobState:
    """Tests for ResearchJobState enum."""

    def test_all_states_exist(self):
        """Test all job states exist."""
        assert ResearchJobState.PENDING == "pending"
        assert ResearchJobState.VALIDATING == "validating"
        assert ResearchJobState.QUEUED == "queued"
        assert ResearchJobState.STARTING == "starting"
        assert ResearchJobState.RUNNING == "running"
        assert ResearchJobState.STOPPING == "stopping"
        assert ResearchJobState.COMPLETED == "completed"
        assert ResearchJobState.FAILED == "failed"
        assert ResearchJobState.TERMINATED == "terminated"
        assert ResearchJobState.CANCELLED == "cancelled"

    def test_state_is_string_enum(self):
        """Test states are string enums."""
        assert isinstance(ResearchJobState.PENDING.value, str)


class TestJobTerminationReason:
    """Tests for JobTerminationReason enum."""

    def test_all_reasons_exist(self):
        """Test all termination reasons exist."""
        assert JobTerminationReason.COMPLETED == "completed"
        assert JobTerminationReason.TIMEOUT == "timeout"
        assert JobTerminationReason.OOM == "oom"
        assert JobTerminationReason.ABUSE_DETECTED == "abuse_detected"
        assert JobTerminationReason.QUOTA_EXCEEDED == "quota_exceeded"
        assert JobTerminationReason.USER_CANCELLED == "user_cancelled"
        assert JobTerminationReason.SYSTEM_ERROR == "system_error"
        assert JobTerminationReason.EGRESS_VIOLATION == "egress_violation"
        assert JobTerminationReason.TENANT_VIOLATION == "tenant_violation"


class TestIsolationLevel:
    """Tests for IsolationLevel enum."""

    def test_all_levels_exist(self):
        """Test all isolation levels exist."""
        assert IsolationLevel.NONE == "none"
        assert IsolationLevel.PROCESS == "process"
        assert IsolationLevel.CONTAINER == "container"
        assert IsolationLevel.GVISOR == "gvisor"
        assert IsolationLevel.MICROVM == "microvm"


class TestQuotaTier:
    """Tests for QuotaTier enum."""

    def test_all_tiers_exist(self):
        """Test all quota tiers exist."""
        assert QuotaTier.FREE == "free"
        assert QuotaTier.PREMIUM == "premium"
        assert QuotaTier.ENTERPRISE == "enterprise"
        assert QuotaTier.CUSTOM == "custom"


class TestAbuseType:
    """Tests for AbuseType enum."""

    def test_all_types_exist(self):
        """Test all abuse types exist."""
        assert AbuseType.CRYPTOCURRENCY_MINING == "cryptocurrency_mining"
        assert AbuseType.PORT_SCANNING == "port_scanning"
        assert AbuseType.NETWORK_SCANNING == "network_scanning"
        assert AbuseType.BOTNET_C2 == "botnet_c2"
        assert AbuseType.DATA_EXFILTRATION == "data_exfiltration"
        assert AbuseType.RESOURCE_EXHAUSTION == "resource_exhaustion"
        assert AbuseType.MALWARE_EXECUTION == "malware_execution"
        assert AbuseType.REVERSE_SHELL == "reverse_shell"


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_all_severities_exist(self):
        """Test all severities exist."""
        assert AlertSeverity.LOW == "low"
        assert AlertSeverity.MEDIUM == "medium"
        assert AlertSeverity.HIGH == "high"
        assert AlertSeverity.CRITICAL == "critical"


class TestResearchJobModel:
    """Tests for ResearchJob model."""

    def test_table_name(self):
        """Test table name."""
        assert ResearchJob.__tablename__ == "research_jobs"

    def test_column_defaults(self):
        """Test column defaults exist."""
        # Check column info exists (model class metadata)
        columns = {c.name: c for c in ResearchJob.__table__.columns}

        assert "id" in columns
        assert "state" in columns
        assert "cpu_limit" in columns
        assert "memory_limit_mb" in columns
        assert "timeout_seconds" in columns

        # Check default values
        assert columns["state"].default.arg == ResearchJobState.PENDING.value
        assert columns["isolation_level"].default.arg == IsolationLevel.CONTAINER.value
        assert columns["cpu_limit"].default.arg == 1.0
        assert columns["memory_limit_mb"].default.arg == 1024

    def test_relationships_defined(self):
        """Test relationships are defined."""
        assert "artifacts" in dir(ResearchJob)
        assert "abuse_incidents" in dir(ResearchJob)


class TestJobQuotaModel:
    """Tests for JobQuota model."""

    def test_table_name(self):
        """Test table name."""
        assert JobQuota.__tablename__ == "job_quotas"

    def test_column_defaults(self):
        """Test column defaults."""
        columns = {c.name: c for c in JobQuota.__table__.columns}

        assert "tier" in columns
        assert "cpu_hours_daily" in columns
        assert "max_concurrent_jobs" in columns
        assert "is_active" in columns

        # Check defaults
        assert columns["tier"].default.arg == QuotaTier.FREE.value
        assert columns["cpu_hours_daily"].default.arg == 2.0
        assert columns["max_concurrent_jobs"].default.arg == 2
        assert columns["is_active"].default.arg is True


class TestQuotaUsageRecordModel:
    """Tests for QuotaUsageRecord model."""

    def test_table_name(self):
        """Test table name."""
        assert QuotaUsageRecord.__tablename__ == "quota_usage_records"

    def test_column_defaults(self):
        """Test column defaults."""
        columns = {c.name: c for c in QuotaUsageRecord.__table__.columns}

        assert "period_type" in columns
        assert "cpu_hours_used" in columns
        assert "jobs_run" in columns

        # Check defaults
        assert columns["period_type"].default.arg == "daily"
        assert columns["cpu_hours_used"].default.arg == 0.0


class TestEgressPolicyRecordModel:
    """Tests for EgressPolicyRecord model."""

    def test_table_name(self):
        """Test table name."""
        assert EgressPolicyRecord.__tablename__ == "egress_policies"

    def test_column_defaults(self):
        """Test column defaults."""
        columns = {c.name: c for c in EgressPolicyRecord.__table__.columns}

        assert "name" in columns
        assert "allowlist_only" in columns
        assert "max_requests_per_minute" in columns

        # Check defaults
        assert columns["name"].default.arg == "default"
        assert columns["allowlist_only"].default.arg is True
        assert columns["max_requests_per_minute"].default.arg == 60


class TestEgressViolationRecordModel:
    """Tests for EgressViolationRecord model."""

    def test_table_name(self):
        """Test table name."""
        assert EgressViolationRecord.__tablename__ == "egress_violations"

    def test_column_defaults(self):
        """Test column defaults."""
        columns = {c.name: c for c in EgressViolationRecord.__table__.columns}

        assert "destination" in columns
        assert "port" in columns
        assert "protocol" in columns
        assert "action_taken" in columns

        # Check defaults
        assert columns["protocol"].default.arg == "TCP"
        assert columns["action_taken"].default.arg == "deny"


class TestAbuseIncidentModel:
    """Tests for AbuseIncident model."""

    def test_table_name(self):
        """Test table name."""
        assert AbuseIncident.__tablename__ == "abuse_incidents"

    def test_column_defaults(self):
        """Test column defaults."""
        columns = {c.name: c for c in AbuseIncident.__table__.columns}

        assert "abuse_type" in columns
        assert "severity" in columns
        assert "confidence" in columns
        assert "job_terminated" in columns

        # Check defaults
        assert columns["severity"].default.arg == AlertSeverity.MEDIUM.value
        assert columns["job_terminated"].default.arg is False

    def test_relationship_defined(self):
        """Test job relationship is defined."""
        assert "job" in dir(AbuseIncident)


class TestJobArtifactModel:
    """Tests for JobArtifact model."""

    def test_table_name(self):
        """Test table name."""
        assert JobArtifact.__tablename__ == "job_artifacts"

    def test_column_defaults(self):
        """Test column defaults."""
        columns = {c.name: c for c in JobArtifact.__table__.columns}

        assert "name" in columns
        assert "artifact_type" in columns
        assert "storage_path" in columns
        assert "size_bytes" in columns
        assert "content_type" in columns

        # Check defaults
        assert columns["size_bytes"].default.arg == 0
        assert columns["content_type"].default.arg == "application/octet-stream"

    def test_relationship_defined(self):
        """Test job relationship is defined."""
        assert "job" in dir(JobArtifact)


class TestCreateResearchJob:
    """Tests for create_research_job helper."""

    def test_create_with_minimal_args(self):
        """Test creating job with minimal arguments."""
        workspace_id = uuid4()
        user_id = uuid4()

        job = create_research_job(
            workspace_id=workspace_id,
            user_id=user_id,
            name="Test Job",
        )

        assert job.workspace_id == workspace_id
        assert job.user_id == user_id
        assert job.name == "Test Job"
        assert job.cpu_limit == 1.0
        assert job.memory_limit_mb == 1024
        assert job.timeout_seconds == 3600

    def test_create_with_custom_resources(self):
        """Test creating job with custom resources."""
        workspace_id = uuid4()
        user_id = uuid4()

        job = create_research_job(
            workspace_id=workspace_id,
            user_id=user_id,
            name="Custom Job",
            cpu_limit=4.0,
            memory_limit_mb=8192,
            timeout_seconds=7200,
        )

        assert job.cpu_limit == 4.0
        assert job.memory_limit_mb == 8192
        assert job.timeout_seconds == 7200

    def test_create_with_extra_kwargs(self):
        """Test creating job with extra kwargs."""
        workspace_id = uuid4()
        user_id = uuid4()

        job = create_research_job(
            workspace_id=workspace_id,
            user_id=user_id,
            name="Extra Job",
            description="A test job",
            docker_image="python:3.10",
        )

        assert job.description == "A test job"
        assert job.docker_image == "python:3.10"


class TestCreateJobQuota:
    """Tests for create_job_quota helper."""

    def test_create_free_tier(self):
        """Test creating free tier quota."""
        workspace_id = uuid4()

        quota = create_job_quota(workspace_id, QuotaTier.FREE)

        assert quota.workspace_id == workspace_id
        assert quota.tier == QuotaTier.FREE.value
        assert quota.cpu_hours_daily == 2.0
        assert quota.memory_gb_hours_daily == 4.0
        assert quota.max_concurrent_jobs == 2

    def test_create_premium_tier(self):
        """Test creating premium tier quota."""
        workspace_id = uuid4()

        quota = create_job_quota(workspace_id, QuotaTier.PREMIUM)

        assert quota.tier == QuotaTier.PREMIUM.value
        assert quota.cpu_hours_daily == 24.0
        assert quota.memory_gb_hours_daily == 48.0
        assert quota.max_concurrent_jobs == 10

    def test_create_enterprise_tier(self):
        """Test creating enterprise tier quota."""
        workspace_id = uuid4()

        quota = create_job_quota(workspace_id, QuotaTier.ENTERPRISE)

        assert quota.tier == QuotaTier.ENTERPRISE.value
        assert quota.cpu_hours_daily == 1000.0
        assert quota.memory_gb_hours_daily == 2000.0
        assert quota.max_concurrent_jobs == 100

    def test_create_with_overrides(self):
        """Test creating quota with overrides."""
        workspace_id = uuid4()

        quota = create_job_quota(
            workspace_id,
            QuotaTier.FREE,
            cpu_hours_daily=10.0,  # Override default
            storage_mb=5120,  # Override default
        )

        assert quota.cpu_hours_daily == 10.0
        assert quota.storage_mb == 5120
        # Other defaults should still apply
        assert quota.max_concurrent_jobs == 2


class TestModelIndexes:
    """Tests for model indexes."""

    def test_research_job_indexes(self):
        """Test ResearchJob has required indexes."""
        indexes = {idx.name: idx for idx in ResearchJob.__table__.indexes}

        assert "ix_research_job_state" in indexes
        assert "ix_research_job_user" in indexes
        assert "ix_research_job_created" in indexes

    def test_quota_usage_indexes(self):
        """Test QuotaUsageRecord has required indexes."""
        indexes = {idx.name: idx for idx in QuotaUsageRecord.__table__.indexes}

        assert "ix_quota_usage_period" in indexes

    def test_egress_violation_indexes(self):
        """Test EgressViolationRecord has required indexes."""
        indexes = {idx.name: idx for idx in EgressViolationRecord.__table__.indexes}

        assert "ix_egress_violation_job" in indexes
        assert "ix_egress_violation_time" in indexes

    def test_abuse_incident_indexes(self):
        """Test AbuseIncident has required indexes."""
        indexes = {idx.name: idx for idx in AbuseIncident.__table__.indexes}

        assert "ix_abuse_incident_job" in indexes
        assert "ix_abuse_incident_type" in indexes
        assert "ix_abuse_incident_severity" in indexes


class TestModelConstraints:
    """Tests for model constraints."""

    def test_job_quota_unique_workspace(self):
        """Test JobQuota has unique workspace constraint."""
        constraints = {c.name: c for c in JobQuota.__table__.constraints}

        assert "uq_job_quota_workspace" in constraints

    def test_egress_policy_unique_workspace_name(self):
        """Test EgressPolicyRecord has unique workspace+name constraint."""
        constraints = {c.name: c for c in EgressPolicyRecord.__table__.constraints}

        assert "uq_egress_policy_workspace_name" in constraints

    def test_job_artifact_unique_job_name(self):
        """Test JobArtifact has unique job+name constraint."""
        constraints = {c.name: c for c in JobArtifact.__table__.constraints}

        assert "uq_job_artifact_job_name" in constraints


class TestModelForeignKeys:
    """Tests for model foreign keys."""

    def test_research_job_user_fk(self):
        """Test ResearchJob has user foreign key."""
        columns = {c.name: c for c in ResearchJob.__table__.columns}

        assert columns["user_id"].foreign_keys

    def test_egress_violation_job_fk(self):
        """Test EgressViolationRecord has job foreign key."""
        columns = {c.name: c for c in EgressViolationRecord.__table__.columns}

        assert columns["job_id"].foreign_keys

    def test_abuse_incident_job_fk(self):
        """Test AbuseIncident has job foreign key."""
        columns = {c.name: c for c in AbuseIncident.__table__.columns}

        assert columns["job_id"].foreign_keys

    def test_job_artifact_job_fk(self):
        """Test JobArtifact has job foreign key."""
        columns = {c.name: c for c in JobArtifact.__table__.columns}

        assert columns["job_id"].foreign_keys
