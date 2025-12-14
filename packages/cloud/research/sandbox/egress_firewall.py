# -*- coding: utf-8 -*-
"""
Egress Firewall - Network egress control for cloud research jobs.

CCEA Phase 10: Egress allowlist + abuse prevention.

Design Doc 15.3/5.3:
- Egress allowlist
- Deny-by-default outbound network
- Prevent unauthorized data exfiltration

This module implements network egress control for research jobs:
- Domain/IP allowlisting
- Port restrictions
- Protocol filtering
- Rate limiting
- Violation logging and alerting

Security Model:
- Default deny all outbound traffic
- Explicit allowlist for permitted destinations
- Block known malicious IPs/domains
- Log all egress attempts for audit

Reference:
- Network Policy (Kubernetes): https://kubernetes.io/docs/concepts/services-networking/network-policies/
- iptables: https://netfilter.org/projects/iptables/
- nftables: https://netfilter.org/projects/nftables/
"""

from __future__ import annotations

import ipaddress
import re
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Final, FrozenSet, List, Optional, Set, Tuple
from uuid import uuid4
import hashlib
import fnmatch


# ============================================================================
# Constants
# ============================================================================

# Default allowed ports for HTTPS only
DEFAULT_ALLOWED_PORTS: Final[FrozenSet[int]] = frozenset({443})

# Extended allowed ports (when explicitly enabled)
EXTENDED_ALLOWED_PORTS: Final[FrozenSet[int]] = frozenset({80, 443, 8080, 8443})

# Always blocked ports (security-sensitive)
BLOCKED_PORTS: Final[FrozenSet[int]] = frozenset({
    22,    # SSH
    23,    # Telnet
    25,    # SMTP (spam)
    135,   # RPC
    137,   # NetBIOS
    138,   # NetBIOS
    139,   # NetBIOS
    445,   # SMB
    1433,  # MSSQL
    1434,  # MSSQL Browser
    3306,  # MySQL
    3389,  # RDP
    5432,  # PostgreSQL
    6379,  # Redis
    27017, # MongoDB
})

# Always blocked IP ranges (private, loopback, link-local)
BLOCKED_IP_RANGES: Final[List[str]] = [
    "10.0.0.0/8",       # Private Class A
    "172.16.0.0/12",    # Private Class B
    "192.168.0.0/16",   # Private Class C
    "127.0.0.0/8",      # Loopback
    "169.254.0.0/16",   # Link-local
    "224.0.0.0/4",      # Multicast
    "240.0.0.0/4",      # Reserved
    "100.64.0.0/10",    # Carrier-grade NAT
    "0.0.0.0/8",        # Current network
    "255.255.255.255/32",  # Broadcast
]

# Cloud metadata endpoints (MUST BLOCK)
BLOCKED_METADATA_IPS: Final[Set[str]] = {
    "169.254.169.254",   # AWS/GCP/Azure metadata
    "metadata.google.internal",
    "metadata.azure.internal",
}

# Rate limiting defaults
DEFAULT_REQUESTS_PER_MINUTE: Final[int] = 60
DEFAULT_BYTES_PER_MINUTE: Final[int] = 50 * 1024 * 1024  # 50 MB/min

# Default public data sources allowlist (research-friendly)
DEFAULT_ALLOWLIST: Final[List[str]] = [
    # Financial data APIs
    "api.binance.com",
    "api.binance.us",
    "api.kraken.com",
    "api.coinbase.com",
    "api.coingecko.com",
    "api.cryptocompare.com",
    "www.alphavantage.co",
    "api.polygon.io",
    "query1.finance.yahoo.com",
    "query2.finance.yahoo.com",

    # Public data sources
    "api.github.com",
    "raw.githubusercontent.com",
    "pypi.org",
    "files.pythonhosted.org",
]


class EgressAction(Enum):
    """Action to take on egress attempt."""
    ALLOW = auto()
    DENY = auto()
    LOG_AND_ALLOW = auto()
    LOG_AND_DENY = auto()
    RATE_LIMIT = auto()


class EgressProtocol(Enum):
    """Network protocol."""
    TCP = auto()
    UDP = auto()
    ICMP = auto()
    ANY = auto()


@dataclass
class EgressRule:
    """
    Single egress rule.

    Rules are evaluated in order; first match wins.
    """
    rule_id: str = field(default_factory=lambda: str(uuid4())[:8])
    name: str = ""
    description: str = ""

    # Match criteria
    destination: str = ""  # IP, CIDR, domain, or wildcard
    port: Optional[int] = None  # None = any port
    port_range: Optional[Tuple[int, int]] = None  # (start, end)
    protocol: EgressProtocol = EgressProtocol.TCP

    # Action
    action: EgressAction = EgressAction.DENY

    # Priority (lower = higher priority)
    priority: int = 100

    # Rate limiting (if action is RATE_LIMIT)
    requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE
    bytes_per_minute: int = DEFAULT_BYTES_PER_MINUTE

    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def matches(
        self,
        destination: str,
        port: int,
        protocol: EgressProtocol = EgressProtocol.TCP,
    ) -> bool:
        """
        Check if rule matches the egress request.

        Args:
            destination: Target IP or domain
            port: Target port
            protocol: Protocol

        Returns:
            True if rule matches
        """
        if not self.enabled:
            return False

        # Check expiration
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False

        # Check protocol
        if self.protocol != EgressProtocol.ANY and self.protocol != protocol:
            return False

        # Check port
        if not self._matches_port(port):
            return False

        # Check destination
        if not self._matches_destination(destination):
            return False

        return True

    def _matches_port(self, port: int) -> bool:
        """Check if port matches rule."""
        if self.port is None and self.port_range is None:
            return True  # Any port
        if self.port is not None and self.port == port:
            return True
        if self.port_range is not None:
            return self.port_range[0] <= port <= self.port_range[1]
        return False

    def _matches_destination(self, destination: str) -> bool:
        """Check if destination matches rule."""
        if not self.destination or self.destination == "*":
            return True

        # Try IP matching
        try:
            dest_ip = ipaddress.ip_address(destination)

            # CIDR match
            if "/" in self.destination:
                network = ipaddress.ip_network(self.destination, strict=False)
                return dest_ip in network

            # Exact IP match
            rule_ip = ipaddress.ip_address(self.destination)
            return dest_ip == rule_ip

        except ValueError:
            pass

        # Domain matching with wildcards
        rule_dest = self.destination.lower()
        dest_lower = destination.lower()

        # Wildcard pattern
        if "*" in rule_dest:
            return fnmatch.fnmatch(dest_lower, rule_dest)

        # Exact match
        if rule_dest == dest_lower:
            return True

        # Subdomain match (rule "example.com" matches "sub.example.com")
        if dest_lower.endswith("." + rule_dest):
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "destination": self.destination,
            "port": self.port,
            "port_range": list(self.port_range) if self.port_range else None,
            "protocol": self.protocol.name,
            "action": self.action.name,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class EgressPolicy:
    """
    Egress policy consisting of multiple rules.

    Policies are tenant-specific and define allowed egress patterns.
    """
    policy_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""
    name: str = "default"
    description: str = ""

    # Rules (evaluated in priority order)
    rules: List[EgressRule] = field(default_factory=list)

    # Default action (if no rule matches)
    default_action: EgressAction = EgressAction.LOG_AND_DENY

    # Global rate limits
    max_requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE
    max_bytes_per_minute: int = DEFAULT_BYTES_PER_MINUTE

    # Allowlist mode (only allow explicit destinations)
    allowlist_only: bool = True

    # Logging
    log_all_attempts: bool = True
    log_denied_only: bool = False

    # Metadata
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_rule(self, rule: EgressRule) -> None:
        """Add a rule to the policy."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)
        self.updated_at = datetime.utcnow()

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID."""
        original_len = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        if len(self.rules) < original_len:
            self.updated_at = datetime.utcnow()
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "policy_id": self.policy_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "rules_count": len(self.rules),
            "default_action": self.default_action.name,
            "allowlist_only": self.allowlist_only,
            "enabled": self.enabled,
        }


@dataclass
class EgressViolation:
    """Record of an egress policy violation."""
    violation_id: str = field(default_factory=lambda: str(uuid4()))
    tenant_id: str = ""
    job_id: str = ""
    sandbox_id: str = ""

    # Request details
    destination: str = ""
    port: int = 0
    protocol: EgressProtocol = EgressProtocol.TCP
    bytes_attempted: int = 0

    # Rule that blocked it (if any)
    matched_rule_id: Optional[str] = None
    matched_rule_name: str = ""

    # Action taken
    action_taken: EgressAction = EgressAction.DENY
    reason: str = ""

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Risk assessment
    risk_score: float = 0.0  # 0-100
    is_suspicious: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "violation_id": self.violation_id,
            "tenant_id": self.tenant_id,
            "job_id": self.job_id,
            "destination": self.destination,
            "port": self.port,
            "protocol": self.protocol.name,
            "action_taken": self.action_taken.name,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "risk_score": self.risk_score,
            "is_suspicious": self.is_suspicious,
        }


class EgressFirewall:
    """
    Egress firewall for cloud research jobs.

    Enforces network egress policies, blocks unauthorized destinations,
    and logs all egress attempts for audit.

    Security Features:
    - Default deny all outbound traffic
    - Explicit allowlist for permitted destinations
    - Block cloud metadata endpoints (prevent SSRF)
    - Block private IP ranges
    - Block security-sensitive ports
    - Rate limiting per tenant/job
    - Violation logging and alerting

    Usage:
        firewall = EgressFirewall()

        # Create policy
        policy = firewall.create_policy(
            tenant_id="tenant-123",
            allowlist=["api.binance.com", "api.github.com"],
        )

        # Check egress
        allowed, violation = firewall.check_egress(
            tenant_id="tenant-123",
            destination="api.binance.com",
            port=443,
        )

        if not allowed:
            print(f"Blocked: {violation.reason}")
    """

    def __init__(
        self,
        on_violation: Optional[Callable[[EgressViolation], None]] = None,
        on_suspicious_activity: Optional[Callable[[EgressViolation], None]] = None,
    ):
        """
        Initialize firewall.

        Args:
            on_violation: Callback for policy violations
            on_suspicious_activity: Callback for suspicious activity
        """
        self._on_violation = on_violation
        self._on_suspicious_activity = on_suspicious_activity

        # Policy storage (tenant_id -> policy)
        self._policies: Dict[str, EgressPolicy] = {}
        self._lock = threading.RLock()

        # Violation history (for rate limiting and analysis)
        self._violations: List[EgressViolation] = []
        self._max_violation_history = 10000

        # Rate limiting state
        self._request_counts: Dict[str, List[datetime]] = {}  # key -> timestamps
        self._byte_counts: Dict[str, List[Tuple[datetime, int]]] = {}  # key -> (timestamp, bytes)

        # Pre-compiled blocked IP networks
        self._blocked_networks = [
            ipaddress.ip_network(cidr, strict=False)
            for cidr in BLOCKED_IP_RANGES
        ]

        # Statistics
        self._stats = {
            "total_checks": 0,
            "allowed": 0,
            "denied": 0,
            "rate_limited": 0,
        }

    def create_policy(
        self,
        tenant_id: str,
        name: str = "default",
        allowlist: Optional[List[str]] = None,
        include_default_allowlist: bool = True,
        allowed_ports: Optional[Set[int]] = None,
    ) -> EgressPolicy:
        """
        Create an egress policy for a tenant.

        Args:
            tenant_id: Tenant identifier
            name: Policy name
            allowlist: List of allowed destinations
            include_default_allowlist: Include default data sources
            allowed_ports: Set of allowed ports (default: 443 only)

        Returns:
            Created EgressPolicy
        """
        policy = EgressPolicy(
            tenant_id=tenant_id,
            name=name,
            allowlist_only=True,
            default_action=EgressAction.LOG_AND_DENY,
        )

        # Build destination allowlist
        destinations = set()

        if include_default_allowlist:
            destinations.update(DEFAULT_ALLOWLIST)

        if allowlist:
            destinations.update(allowlist)

        # Create allow rules for each destination
        ports = allowed_ports or DEFAULT_ALLOWED_PORTS

        for dest in destinations:
            for port in ports:
                rule = EgressRule(
                    name=f"allow-{dest}:{port}",
                    destination=dest,
                    port=port,
                    protocol=EgressProtocol.TCP,
                    action=EgressAction.ALLOW,
                    priority=50,
                )
                policy.add_rule(rule)

        # Add blocked rules (highest priority)
        self._add_blocked_rules(policy)

        with self._lock:
            self._policies[tenant_id] = policy

        return policy

    def _add_blocked_rules(self, policy: EgressPolicy) -> None:
        """Add rules for blocked destinations."""
        # Block metadata endpoints
        for ip in BLOCKED_METADATA_IPS:
            rule = EgressRule(
                name=f"block-metadata-{ip}",
                destination=ip,
                port=None,  # All ports
                action=EgressAction.LOG_AND_DENY,
                priority=1,  # Highest priority
            )
            policy.add_rule(rule)

        # Block private networks
        for cidr in BLOCKED_IP_RANGES:
            rule = EgressRule(
                name=f"block-private-{cidr}",
                destination=cidr,
                port=None,
                action=EgressAction.DENY,
                priority=5,
            )
            policy.add_rule(rule)

        # Block dangerous ports
        for port in BLOCKED_PORTS:
            rule = EgressRule(
                name=f"block-port-{port}",
                destination="*",
                port=port,
                action=EgressAction.LOG_AND_DENY,
                priority=10,
            )
            policy.add_rule(rule)

    def get_policy(self, tenant_id: str) -> Optional[EgressPolicy]:
        """Get policy for a tenant."""
        with self._lock:
            return self._policies.get(tenant_id)

    def update_policy(
        self,
        tenant_id: str,
        add_destinations: Optional[List[str]] = None,
        remove_destinations: Optional[List[str]] = None,
        add_ports: Optional[Set[int]] = None,
    ) -> Optional[EgressPolicy]:
        """
        Update an existing policy.

        Args:
            tenant_id: Tenant identifier
            add_destinations: Destinations to add
            remove_destinations: Destinations to remove
            add_ports: Ports to add

        Returns:
            Updated policy or None if not found
        """
        with self._lock:
            policy = self._policies.get(tenant_id)
            if not policy:
                return None

            # Add new destinations
            if add_destinations:
                for dest in add_destinations:
                    for port in (add_ports or DEFAULT_ALLOWED_PORTS):
                        rule = EgressRule(
                            name=f"allow-{dest}:{port}",
                            destination=dest,
                            port=port,
                            action=EgressAction.ALLOW,
                            priority=50,
                        )
                        policy.add_rule(rule)

            # Remove destinations (find and remove matching rules)
            if remove_destinations:
                for dest in remove_destinations:
                    policy.rules = [
                        r for r in policy.rules
                        if r.destination.lower() != dest.lower()
                        or r.action != EgressAction.ALLOW
                    ]

            policy.updated_at = datetime.utcnow()
            return policy

    def delete_policy(self, tenant_id: str) -> bool:
        """Delete a tenant's policy."""
        with self._lock:
            if tenant_id in self._policies:
                del self._policies[tenant_id]
                return True
            return False

    def check_egress(
        self,
        tenant_id: str,
        destination: str,
        port: int,
        protocol: EgressProtocol = EgressProtocol.TCP,
        job_id: str = "",
        sandbox_id: str = "",
        bytes_count: int = 0,
    ) -> Tuple[bool, Optional[EgressViolation]]:
        """
        Check if egress is allowed.

        Args:
            tenant_id: Tenant identifier
            destination: Target IP or domain
            port: Target port
            protocol: Protocol (TCP/UDP)
            job_id: Job identifier for logging
            sandbox_id: Sandbox identifier
            bytes_count: Number of bytes (for rate limiting)

        Returns:
            Tuple of (allowed, violation if denied)
        """
        with self._lock:
            self._stats["total_checks"] += 1

            # Resolve domain to IP if needed
            resolved_ip = self._resolve_destination(destination)

            # Check built-in blocks first (always enforced)
            blocked_reason = self._check_builtin_blocks(destination, resolved_ip, port)
            if blocked_reason:
                violation = self._create_violation(
                    tenant_id, job_id, sandbox_id,
                    destination, port, protocol, bytes_count,
                    EgressAction.DENY, blocked_reason,
                    risk_score=80.0, is_suspicious=True,
                )
                self._record_violation(violation)
                self._stats["denied"] += 1
                return False, violation

            # Get tenant policy
            policy = self._policies.get(tenant_id)
            if not policy or not policy.enabled:
                # No policy = default deny
                violation = self._create_violation(
                    tenant_id, job_id, sandbox_id,
                    destination, port, protocol, bytes_count,
                    EgressAction.DENY, "No egress policy defined",
                    risk_score=50.0,
                )
                self._record_violation(violation)
                self._stats["denied"] += 1
                return False, violation

            # Check rate limits
            rate_key = f"{tenant_id}:{job_id}"
            if not self._check_rate_limit(rate_key, bytes_count, policy):
                violation = self._create_violation(
                    tenant_id, job_id, sandbox_id,
                    destination, port, protocol, bytes_count,
                    EgressAction.RATE_LIMIT, "Rate limit exceeded",
                    risk_score=30.0,
                )
                self._record_violation(violation)
                self._stats["rate_limited"] += 1
                return False, violation

            # Evaluate rules
            for rule in policy.rules:
                if rule.matches(destination, port, protocol):
                    if rule.action in (EgressAction.ALLOW, EgressAction.LOG_AND_ALLOW):
                        self._stats["allowed"] += 1

                        # Update rate limit counters
                        self._update_rate_counters(rate_key, bytes_count)

                        # Log if required
                        if rule.action == EgressAction.LOG_AND_ALLOW or policy.log_all_attempts:
                            violation = self._create_violation(
                                tenant_id, job_id, sandbox_id,
                                destination, port, protocol, bytes_count,
                                EgressAction.ALLOW, f"Allowed by rule: {rule.name}",
                                matched_rule_id=rule.rule_id,
                                matched_rule_name=rule.name,
                            )
                            self._record_violation(violation)

                        return True, None

                    else:
                        # Deny
                        violation = self._create_violation(
                            tenant_id, job_id, sandbox_id,
                            destination, port, protocol, bytes_count,
                            rule.action, f"Denied by rule: {rule.name}",
                            matched_rule_id=rule.rule_id,
                            matched_rule_name=rule.name,
                            risk_score=60.0,
                        )
                        self._record_violation(violation)
                        self._stats["denied"] += 1
                        return False, violation

            # No rule matched, apply default action
            if policy.default_action in (EgressAction.ALLOW, EgressAction.LOG_AND_ALLOW):
                self._stats["allowed"] += 1
                self._update_rate_counters(rate_key, bytes_count)
                return True, None
            else:
                violation = self._create_violation(
                    tenant_id, job_id, sandbox_id,
                    destination, port, protocol, bytes_count,
                    policy.default_action, "No matching rule, default deny",
                    risk_score=40.0,
                )
                self._record_violation(violation)
                self._stats["denied"] += 1
                return False, violation

    def _resolve_destination(self, destination: str) -> Optional[str]:
        """Resolve domain to IP address."""
        try:
            # Check if already an IP
            ipaddress.ip_address(destination)
            return destination
        except ValueError:
            pass

        # Try DNS resolution
        try:
            result = socket.gethostbyname(destination)
            return result
        except socket.gaierror:
            return None

    def _check_builtin_blocks(
        self,
        destination: str,
        resolved_ip: Optional[str],
        port: int,
    ) -> Optional[str]:
        """Check built-in security blocks."""
        # Block metadata endpoints (SSRF prevention)
        if destination in BLOCKED_METADATA_IPS:
            return "Cloud metadata endpoint blocked (SSRF prevention)"

        if resolved_ip and resolved_ip in BLOCKED_METADATA_IPS:
            return "Resolved to cloud metadata endpoint (SSRF prevention)"

        # Block private IP ranges
        if resolved_ip:
            try:
                ip = ipaddress.ip_address(resolved_ip)
                for network in self._blocked_networks:
                    if ip in network:
                        return f"Private/reserved IP range blocked: {network}"
            except ValueError:
                pass

        # Block dangerous ports
        if port in BLOCKED_PORTS:
            return f"Security-sensitive port blocked: {port}"

        return None

    def _check_rate_limit(
        self,
        key: str,
        bytes_count: int,
        policy: EgressPolicy,
    ) -> bool:
        """Check if request is within rate limits."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        # Request count check
        if key in self._request_counts:
            # Remove old entries
            self._request_counts[key] = [
                ts for ts in self._request_counts[key]
                if ts > window_start
            ]
            if len(self._request_counts[key]) >= policy.max_requests_per_minute:
                return False

        # Bytes check
        if key in self._byte_counts:
            self._byte_counts[key] = [
                (ts, b) for ts, b in self._byte_counts[key]
                if ts > window_start
            ]
            total_bytes = sum(b for _, b in self._byte_counts[key])
            if total_bytes + bytes_count > policy.max_bytes_per_minute:
                return False

        return True

    def _update_rate_counters(self, key: str, bytes_count: int) -> None:
        """Update rate limit counters."""
        now = datetime.utcnow()

        if key not in self._request_counts:
            self._request_counts[key] = []
        self._request_counts[key].append(now)

        if bytes_count > 0:
            if key not in self._byte_counts:
                self._byte_counts[key] = []
            self._byte_counts[key].append((now, bytes_count))

    def _create_violation(
        self,
        tenant_id: str,
        job_id: str,
        sandbox_id: str,
        destination: str,
        port: int,
        protocol: EgressProtocol,
        bytes_count: int,
        action: EgressAction,
        reason: str,
        matched_rule_id: Optional[str] = None,
        matched_rule_name: str = "",
        risk_score: float = 0.0,
        is_suspicious: bool = False,
    ) -> EgressViolation:
        """Create a violation record."""
        return EgressViolation(
            tenant_id=tenant_id,
            job_id=job_id,
            sandbox_id=sandbox_id,
            destination=destination,
            port=port,
            protocol=protocol,
            bytes_attempted=bytes_count,
            matched_rule_id=matched_rule_id,
            matched_rule_name=matched_rule_name,
            action_taken=action,
            reason=reason,
            risk_score=risk_score,
            is_suspicious=is_suspicious,
        )

    def _record_violation(self, violation: EgressViolation) -> None:
        """Record a violation and trigger callbacks."""
        self._violations.append(violation)

        # Trim history
        if len(self._violations) > self._max_violation_history:
            self._violations = self._violations[-self._max_violation_history:]

        # Trigger callbacks
        if violation.action_taken in (EgressAction.DENY, EgressAction.LOG_AND_DENY):
            if self._on_violation:
                try:
                    self._on_violation(violation)
                except Exception:
                    pass

        if violation.is_suspicious and self._on_suspicious_activity:
            try:
                self._on_suspicious_activity(violation)
            except Exception:
                pass

    def get_violations(
        self,
        tenant_id: Optional[str] = None,
        job_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[EgressViolation]:
        """
        Get violation history.

        Args:
            tenant_id: Filter by tenant
            job_id: Filter by job
            since: Filter by time
            limit: Maximum results

        Returns:
            List of violations
        """
        with self._lock:
            results = self._violations.copy()

        # Apply filters
        if tenant_id:
            results = [v for v in results if v.tenant_id == tenant_id]
        if job_id:
            results = [v for v in results if v.job_id == job_id]
        if since:
            results = [v for v in results if v.timestamp >= since]

        # Sort by timestamp descending
        results.sort(key=lambda v: v.timestamp, reverse=True)

        return results[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get firewall statistics."""
        with self._lock:
            return {
                **self._stats,
                "policies_count": len(self._policies),
                "violations_in_history": len(self._violations),
            }

    def get_tenant_stats(self, tenant_id: str) -> Dict[str, Any]:
        """Get statistics for a specific tenant."""
        violations = self.get_violations(tenant_id=tenant_id)

        return {
            "tenant_id": tenant_id,
            "total_violations": len(violations),
            "denied_count": sum(
                1 for v in violations
                if v.action_taken in (EgressAction.DENY, EgressAction.LOG_AND_DENY)
            ),
            "suspicious_count": sum(1 for v in violations if v.is_suspicious),
            "rate_limited_count": sum(
                1 for v in violations
                if v.action_taken == EgressAction.RATE_LIMIT
            ),
        }


def create_restrictive_policy(tenant_id: str) -> EgressPolicy:
    """
    Create a highly restrictive policy (no external access).

    Args:
        tenant_id: Tenant identifier

    Returns:
        Restrictive EgressPolicy
    """
    policy = EgressPolicy(
        tenant_id=tenant_id,
        name="restrictive",
        allowlist_only=True,
        default_action=EgressAction.DENY,
        log_all_attempts=True,
    )

    # Only blocked rules, no allow rules
    firewall = EgressFirewall()
    firewall._add_blocked_rules(policy)

    return policy


def create_permissive_policy(
    tenant_id: str,
    allowed_domains: List[str],
) -> EgressPolicy:
    """
    Create a permissive policy with explicit allowlist.

    Args:
        tenant_id: Tenant identifier
        allowed_domains: List of allowed domains

    Returns:
        EgressPolicy with allowlist
    """
    firewall = EgressFirewall()
    return firewall.create_policy(
        tenant_id=tenant_id,
        name="permissive",
        allowlist=allowed_domains,
        include_default_allowlist=True,
        allowed_ports={80, 443},
    )
