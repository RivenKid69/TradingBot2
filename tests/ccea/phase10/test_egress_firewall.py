# -*- coding: utf-8 -*-
"""
Tests for EgressFirewall.

CCEA Phase 10: Network egress control.
"""

import pytest
from datetime import datetime, timedelta

from packages.cloud.research.sandbox.egress_firewall import (
    EgressFirewall,
    EgressRule,
    EgressPolicy,
    EgressViolation,
    EgressAction,
    EgressProtocol,
    create_restrictive_policy,
    create_permissive_policy,
    DEFAULT_ALLOWED_PORTS,
    BLOCKED_PORTS,
    BLOCKED_METADATA_IPS,
)


class TestEgressRule:
    """Tests for EgressRule."""

    def test_default_rule(self):
        """Test default rule values."""
        rule = EgressRule()

        assert rule.action == EgressAction.DENY
        assert rule.protocol == EgressProtocol.TCP
        assert rule.enabled is True

    def test_rule_with_destination(self):
        """Test rule with destination."""
        rule = EgressRule(
            name="allow-github",
            destination="api.github.com",
            port=443,
            action=EgressAction.ALLOW,
        )

        assert rule.destination == "api.github.com"
        assert rule.port == 443
        assert rule.action == EgressAction.ALLOW

    def test_rule_matches_exact_ip(self):
        """Test rule matches exact IP."""
        rule = EgressRule(
            destination="1.2.3.4",
            port=443,
        )

        assert rule.matches("1.2.3.4", 443) is True
        assert rule.matches("1.2.3.5", 443) is False

    def test_rule_matches_cidr(self):
        """Test rule matches CIDR range."""
        rule = EgressRule(
            destination="10.0.0.0/8",
            port=443,
        )

        assert rule.matches("10.0.0.1", 443) is True
        assert rule.matches("10.255.255.255", 443) is True
        assert rule.matches("11.0.0.1", 443) is False

    def test_rule_matches_domain(self):
        """Test rule matches domain."""
        rule = EgressRule(
            destination="api.github.com",
            port=443,
        )

        assert rule.matches("api.github.com", 443) is True
        assert rule.matches("api.gitlab.com", 443) is False

    def test_rule_matches_subdomain(self):
        """Test rule matches subdomain."""
        rule = EgressRule(
            destination="github.com",
            port=443,
        )

        # Subdomain should match parent domain rule
        assert rule.matches("api.github.com", 443) is True
        assert rule.matches("raw.github.com", 443) is True

    def test_rule_matches_wildcard(self):
        """Test rule matches wildcard pattern."""
        rule = EgressRule(
            destination="*.github.com",
            port=443,
        )

        assert rule.matches("api.github.com", 443) is True
        assert rule.matches("raw.github.com", 443) is True
        assert rule.matches("github.com", 443) is False

    def test_rule_matches_any_port(self):
        """Test rule matches any port when port is None."""
        rule = EgressRule(
            destination="api.github.com",
            port=None,  # Any port
        )

        assert rule.matches("api.github.com", 80) is True
        assert rule.matches("api.github.com", 443) is True
        assert rule.matches("api.github.com", 8080) is True

    def test_rule_matches_port_range(self):
        """Test rule matches port range."""
        rule = EgressRule(
            destination="*",
            port_range=(8000, 9000),
        )

        assert rule.matches("any.host", 8000) is True
        assert rule.matches("any.host", 8500) is True
        assert rule.matches("any.host", 9000) is True
        assert rule.matches("any.host", 7999) is False
        assert rule.matches("any.host", 9001) is False

    def test_rule_disabled(self):
        """Test disabled rule doesn't match."""
        rule = EgressRule(
            destination="*",
            port=443,
            enabled=False,
        )

        assert rule.matches("any.host", 443) is False

    def test_rule_expired(self):
        """Test expired rule doesn't match."""
        rule = EgressRule(
            destination="*",
            port=443,
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )

        assert rule.matches("any.host", 443) is False

    def test_rule_to_dict(self):
        """Test rule serialization."""
        rule = EgressRule(
            name="test-rule",
            destination="api.github.com",
            port=443,
            action=EgressAction.ALLOW,
        )

        data = rule.to_dict()

        assert data["name"] == "test-rule"
        assert data["destination"] == "api.github.com"
        assert data["port"] == 443
        assert data["action"] == "ALLOW"


class TestEgressPolicy:
    """Tests for EgressPolicy."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = EgressPolicy(tenant_id="tenant-123")

        assert policy.tenant_id == "tenant-123"
        assert policy.default_action == EgressAction.LOG_AND_DENY
        assert policy.allowlist_only is True
        assert len(policy.rules) == 0

    def test_add_rule(self):
        """Test adding rule to policy."""
        policy = EgressPolicy(tenant_id="tenant-123")

        rule = EgressRule(
            name="allow-github",
            destination="api.github.com",
            port=443,
            action=EgressAction.ALLOW,
            priority=50,
        )

        policy.add_rule(rule)

        assert len(policy.rules) == 1
        assert policy.rules[0].name == "allow-github"

    def test_rules_sorted_by_priority(self):
        """Test rules are sorted by priority."""
        policy = EgressPolicy(tenant_id="tenant-123")

        policy.add_rule(EgressRule(name="low", priority=100))
        policy.add_rule(EgressRule(name="high", priority=10))
        policy.add_rule(EgressRule(name="medium", priority=50))

        assert policy.rules[0].name == "high"
        assert policy.rules[1].name == "medium"
        assert policy.rules[2].name == "low"

    def test_remove_rule(self):
        """Test removing rule from policy."""
        policy = EgressPolicy(tenant_id="tenant-123")

        rule = EgressRule(name="test")
        policy.add_rule(rule)

        result = policy.remove_rule(rule.rule_id)

        assert result is True
        assert len(policy.rules) == 0

    def test_remove_nonexistent_rule(self):
        """Test removing non-existent rule."""
        policy = EgressPolicy(tenant_id="tenant-123")

        result = policy.remove_rule("nonexistent")

        assert result is False


class TestEgressFirewall:
    """Tests for EgressFirewall."""

    def test_firewall_creation(self):
        """Test firewall creation."""
        firewall = EgressFirewall()

        assert firewall.get_stats()["total_checks"] == 0

    def test_create_policy(self):
        """Test creating policy."""
        firewall = EgressFirewall()

        policy = firewall.create_policy(
            tenant_id="tenant-123",
            allowlist=["api.github.com", "api.binance.com"],
        )

        assert policy.tenant_id == "tenant-123"
        assert len(policy.rules) > 0

    def test_create_policy_with_default_allowlist(self):
        """Test policy includes default allowlist."""
        firewall = EgressFirewall()

        policy = firewall.create_policy(
            tenant_id="tenant-123",
            include_default_allowlist=True,
        )

        # Should have rules for default data sources
        destinations = [r.destination for r in policy.rules if r.action == EgressAction.ALLOW]
        assert any("binance" in d for d in destinations)

    def test_get_policy(self):
        """Test getting policy."""
        firewall = EgressFirewall()
        firewall.create_policy(tenant_id="tenant-123")

        policy = firewall.get_policy("tenant-123")

        assert policy is not None
        assert policy.tenant_id == "tenant-123"

    def test_get_nonexistent_policy(self):
        """Test getting non-existent policy."""
        firewall = EgressFirewall()

        policy = firewall.get_policy("nonexistent")

        assert policy is None

    def test_check_egress_allowed(self):
        """Test allowed egress."""
        firewall = EgressFirewall()
        firewall.create_policy(
            tenant_id="tenant-123",
            allowlist=["api.github.com"],
        )

        allowed, violation = firewall.check_egress(
            tenant_id="tenant-123",
            destination="api.github.com",
            port=443,
        )

        assert allowed is True
        assert violation is None

    def test_check_egress_denied_no_policy(self):
        """Test egress denied without policy."""
        firewall = EgressFirewall()

        allowed, violation = firewall.check_egress(
            tenant_id="tenant-123",
            destination="api.github.com",
            port=443,
        )

        assert allowed is False
        assert violation is not None
        assert "No egress policy" in violation.reason

    def test_check_egress_denied_not_in_allowlist(self):
        """Test egress denied when not in allowlist."""
        firewall = EgressFirewall()
        firewall.create_policy(
            tenant_id="tenant-123",
            allowlist=["api.github.com"],
            include_default_allowlist=False,
        )

        allowed, violation = firewall.check_egress(
            tenant_id="tenant-123",
            destination="malicious.site",
            port=443,
        )

        assert allowed is False
        assert violation is not None

    def test_check_egress_blocks_metadata(self):
        """Test egress blocks cloud metadata endpoints."""
        firewall = EgressFirewall()
        firewall.create_policy(tenant_id="tenant-123")

        # AWS/GCP/Azure metadata endpoint
        allowed, violation = firewall.check_egress(
            tenant_id="tenant-123",
            destination="169.254.169.254",
            port=80,
        )

        assert allowed is False
        assert violation is not None
        assert "metadata" in violation.reason.lower() or "SSRF" in violation.reason

    def test_check_egress_blocks_private_ip(self):
        """Test egress blocks private IP ranges."""
        firewall = EgressFirewall()
        firewall.create_policy(tenant_id="tenant-123")

        # Private IP
        allowed, violation = firewall.check_egress(
            tenant_id="tenant-123",
            destination="192.168.1.1",
            port=443,
        )

        assert allowed is False

    def test_check_egress_blocks_dangerous_ports(self):
        """Test egress blocks dangerous ports."""
        firewall = EgressFirewall()
        firewall.create_policy(
            tenant_id="tenant-123",
            allowlist=["*"],  # Allow all destinations
        )

        # SSH port
        allowed, violation = firewall.check_egress(
            tenant_id="tenant-123",
            destination="external.host",
            port=22,
        )

        assert allowed is False
        assert "port" in violation.reason.lower()

    def test_check_egress_rate_limit(self):
        """Test egress rate limiting."""
        firewall = EgressFirewall()
        policy = firewall.create_policy(
            tenant_id="tenant-123",
            allowlist=["api.github.com"],
        )
        policy.max_requests_per_minute = 5

        # Make requests up to limit
        for _ in range(5):
            allowed, _ = firewall.check_egress(
                tenant_id="tenant-123",
                destination="api.github.com",
                port=443,
                job_id="job-456",
            )
            assert allowed is True

        # Next request should be rate limited
        allowed, violation = firewall.check_egress(
            tenant_id="tenant-123",
            destination="api.github.com",
            port=443,
            job_id="job-456",
        )

        assert allowed is False
        assert "rate limit" in violation.reason.lower()

    def test_get_violations(self):
        """Test getting violation history."""
        firewall = EgressFirewall()

        # Create some violations
        firewall.check_egress(
            tenant_id="tenant-123",
            destination="blocked.site",
            port=443,
            job_id="job-456",
        )

        violations = firewall.get_violations(tenant_id="tenant-123")

        assert len(violations) > 0
        assert violations[0].tenant_id == "tenant-123"

    def test_get_violations_filtered_by_job(self):
        """Test getting violations filtered by job."""
        firewall = EgressFirewall()

        firewall.check_egress(
            tenant_id="tenant-123",
            destination="blocked.site",
            port=443,
            job_id="job-1",
        )
        firewall.check_egress(
            tenant_id="tenant-123",
            destination="blocked.site",
            port=443,
            job_id="job-2",
        )

        violations = firewall.get_violations(job_id="job-1")

        assert all(v.job_id == "job-1" for v in violations)

    def test_get_stats(self):
        """Test getting firewall stats."""
        firewall = EgressFirewall()

        stats = firewall.get_stats()

        assert "total_checks" in stats
        assert "allowed" in stats
        assert "denied" in stats
        assert "rate_limited" in stats

    def test_update_policy(self):
        """Test updating policy."""
        firewall = EgressFirewall()
        firewall.create_policy(
            tenant_id="tenant-123",
            allowlist=["api.github.com"],
        )

        policy = firewall.update_policy(
            tenant_id="tenant-123",
            add_destinations=["api.gitlab.com"],
        )

        assert policy is not None

    def test_delete_policy(self):
        """Test deleting policy."""
        firewall = EgressFirewall()
        firewall.create_policy(tenant_id="tenant-123")

        result = firewall.delete_policy("tenant-123")

        assert result is True
        assert firewall.get_policy("tenant-123") is None

    def test_violation_callback(self):
        """Test violation callback is triggered."""
        violations_received = []

        def on_violation(v):
            violations_received.append(v)

        firewall = EgressFirewall(on_violation=on_violation)

        firewall.check_egress(
            tenant_id="tenant-123",
            destination="blocked.site",
            port=443,
        )

        assert len(violations_received) > 0


class TestCreatePolicyHelpers:
    """Tests for policy creation helpers."""

    def test_create_restrictive_policy(self):
        """Test creating restrictive policy."""
        policy = create_restrictive_policy("tenant-123")

        assert policy.tenant_id == "tenant-123"
        assert policy.allowlist_only is True
        assert policy.default_action == EgressAction.DENY

    def test_create_permissive_policy(self):
        """Test creating permissive policy."""
        policy = create_permissive_policy(
            "tenant-123",
            allowed_domains=["api.github.com"],
        )

        assert policy.tenant_id == "tenant-123"


class TestEgressViolation:
    """Tests for EgressViolation."""

    def test_violation_creation(self):
        """Test violation creation."""
        violation = EgressViolation(
            tenant_id="tenant-123",
            job_id="job-456",
            destination="blocked.site",
            port=443,
            reason="Not in allowlist",
        )

        assert violation.tenant_id == "tenant-123"
        assert violation.destination == "blocked.site"
        assert violation.action_taken == EgressAction.DENY

    def test_violation_to_dict(self):
        """Test violation serialization."""
        violation = EgressViolation(
            tenant_id="tenant-123",
            destination="blocked.site",
            port=443,
            reason="Test",
        )

        data = violation.to_dict()

        assert data["tenant_id"] == "tenant-123"
        assert data["destination"] == "blocked.site"


class TestBlockedResources:
    """Tests for blocked resources constants."""

    def test_blocked_ports(self):
        """Test blocked ports are defined."""
        assert 22 in BLOCKED_PORTS  # SSH
        assert 3306 in BLOCKED_PORTS  # MySQL
        assert 5432 in BLOCKED_PORTS  # PostgreSQL
        assert 6379 in BLOCKED_PORTS  # Redis

    def test_blocked_metadata_ips(self):
        """Test cloud metadata IPs are blocked."""
        assert "169.254.169.254" in BLOCKED_METADATA_IPS

    def test_default_allowed_ports(self):
        """Test default allowed ports."""
        assert 443 in DEFAULT_ALLOWED_PORTS
