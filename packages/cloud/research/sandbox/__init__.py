# -*- coding: utf-8 -*-
"""
Cloud Research Job Sandbox - CLOUD ZONE ONLY.

Phase 10 Implementation: Isolated execution environment for research jobs.

Design Doc Requirements (15.3/5.3):
- Sandbox for research jobs (container/VM)
- Quotas CPU/RAM/time
- Egress allowlist + abuse detection
- Tenant isolation at job execution level

SECURITY CRITICAL:
    - User code runs in isolated containers/VMs
    - Network egress is restricted by allowlist
    - Abuse patterns are detected and blocked
    - Tenant isolation is enforced at all levels
"""

from packages.cloud.research.sandbox.cloud_sandbox import (
    CloudResearchSandbox,
    CloudSandboxConfig,
    CloudSandboxState,
    CloudSandboxResult,
    CloudSandboxMetrics,
    IsolationLevel,
)

from packages.cloud.research.sandbox.egress_firewall import (
    EgressFirewall,
    EgressRule,
    EgressPolicy,
    EgressViolation,
)

from packages.cloud.research.sandbox.abuse_detector import (
    AbuseDetector,
    AbuseType,
    AbuseAlert,
    AbuseDetectorConfig,
)

from packages.cloud.research.sandbox.resource_quota import (
    ResourceQuotaManager,
    TenantQuota,
    QuotaUsage,
    QuotaExceededError,
)

from packages.cloud.research.sandbox.tenant_isolation import (
    TenantJobIsolation,
    IsolatedJobContext,
    TenantBoundaryViolation,
)

from packages.cloud.research.sandbox.job_executor import (
    ResearchJobExecutor,
    ResearchJob,
    JobConfig,
    JobResult,
    JobState,
)

__all__ = [
    # CloudSandbox
    "CloudResearchSandbox",
    "CloudSandboxConfig",
    "CloudSandboxState",
    "CloudSandboxResult",
    "CloudSandboxMetrics",
    "IsolationLevel",
    # Egress
    "EgressFirewall",
    "EgressRule",
    "EgressPolicy",
    "EgressViolation",
    # Abuse
    "AbuseDetector",
    "AbuseType",
    "AbuseAlert",
    "AbuseDetectorConfig",
    # Quota
    "ResourceQuotaManager",
    "TenantQuota",
    "QuotaUsage",
    "QuotaExceededError",
    # Tenant Isolation
    "TenantJobIsolation",
    "IsolatedJobContext",
    "TenantBoundaryViolation",
    # Job Executor
    "ResearchJobExecutor",
    "ResearchJob",
    "JobConfig",
    "JobResult",
    "JobState",
]
