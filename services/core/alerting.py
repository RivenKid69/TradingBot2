# -*- coding: utf-8 -*-
"""
Comprehensive Alerting System (Block 2.4).

Implements multi-channel alerting with escalation:
- Multi-channel notifications (Email, Slack, PagerDuty, SMS)
- Escalation policies
- Alert deduplication and grouping
- SLA-aware alerting

DORA References:
    - Article 10: Detection of Anomalous Activities
    - Article 11: Response and Recovery
    - Article 14: Communication
    - RTS CDR 2024/1774: ICT Risk Management Framework

Best Practices:
    - PagerDuty Alerting Best Practices
    - Google SRE Book: Practical Alerting
    - NIST SP 800-61: Computer Security Incident Handling Guide
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
import dataclasses
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"  # P1 - Immediate action required
    HIGH = "high"  # P2 - Urgent attention needed
    MEDIUM = "medium"  # P3 - Should be addressed soon
    LOW = "low"  # P4 - Informational
    INFO = "info"  # P5 - For awareness


class AlertChannel(Enum):
    """Alert notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    SMS = "sms"
    WEBHOOK = "webhook"
    TEAMS = "teams"
    OPSGENIE = "opsgenie"
    LOG = "log"


class AlertStatus(Enum):
    """Alert lifecycle status."""

    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


class EscalationLevel(Enum):
    """Escalation levels."""

    L1 = "L1"  # First responder
    L2 = "L2"  # Senior engineer
    L3 = "L3"  # Team lead / Manager
    L4 = "L4"  # Director / VP
    L5 = "L5"  # Executive


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AlertRule:
    """Alert rule definition."""

    rule_id: str = ""
    name: str = ""
    description: str = ""

    # Condition
    condition_type: str = ""  # threshold, anomaly, pattern
    metric_name: str = ""
    threshold_value: float = 0.0
    comparison: str = ">"  # >, <, >=, <=, ==, !=

    # Classification
    severity: AlertSeverity = AlertSeverity.MEDIUM
    category: str = ""
    tags: List[str] = field(default_factory=list)

    # Notification
    channels: List[AlertChannel] = field(default_factory=list)
    escalation_policy_id: str = ""

    # Timing
    evaluation_interval_seconds: int = 60
    pending_duration_seconds: int = 0  # Time to wait before alerting

    # Grouping
    group_by: List[str] = field(default_factory=list)
    group_wait_seconds: int = 30
    group_interval_seconds: int = 300

    # Status
    is_enabled: bool = True
    created_at: str = ""

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = f"RULE-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class Alert:
    """Alert instance."""

    alert_id: str = ""
    rule_id: str = ""
    name: str = ""
    description: str = ""

    # Classification
    severity: AlertSeverity = AlertSeverity.MEDIUM
    status: AlertStatus = AlertStatus.TRIGGERED
    category: str = ""

    # Timing
    triggered_at: str = ""
    acknowledged_at: str = ""
    resolved_at: str = ""
    last_escalation_at: str = ""

    # Escalation
    escalation_level: EscalationLevel = EscalationLevel.L1
    escalation_count: int = 0

    # Context
    source: str = ""
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)

    # Notification
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)

    # Resolution
    acknowledged_by: str = ""
    resolved_by: str = ""
    resolution_notes: str = ""

    # Fingerprint for deduplication
    fingerprint: str = ""

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"ALERT-{uuid.uuid4().hex[:8].upper()}"
        if not self.triggered_at:
            self.triggered_at = datetime.now(timezone.utc).isoformat()
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Generate deduplication fingerprint."""
        components = [self.rule_id, self.source, self.metric_name]
        components.extend(sorted(f"{k}={v}" for k, v in self.labels.items()))
        return uuid.uuid5(uuid.NAMESPACE_OID, ":".join(components)).hex[:16]


@dataclass
class EscalationPolicy:
    """Escalation policy definition."""

    policy_id: str = ""
    name: str = ""
    description: str = ""

    # Escalation steps
    steps: List[Dict[str, Any]] = field(default_factory=list)
    # Each step: {"level": "L1", "delay_minutes": 15, "channels": ["slack"], "targets": ["team-oncall"]}

    # Repeat
    repeat_enabled: bool = True
    repeat_interval_minutes: int = 30
    max_escalations: int = 5

    # Status
    is_active: bool = True
    created_at: str = ""

    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = f"ESCPOL-{uuid.uuid4().hex[:8].upper()}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class NotificationResult:
    """Result of sending a notification."""

    notification_id: str = ""
    alert_id: str = ""
    channel: AlertChannel = AlertChannel.LOG
    target: str = ""

    # Status
    success: bool = False
    sent_at: str = ""
    delivered_at: str = ""
    error: str = ""

    # Response
    response_code: int = 0
    response_message: str = ""

    def __post_init__(self):
        if not self.notification_id:
            self.notification_id = f"NOTIF-{uuid.uuid4().hex[:8].upper()}"
        if not self.sent_at:
            self.sent_at = datetime.now(timezone.utc).isoformat()


@dataclass
class AlertingConfig:
    """Configuration for AlertingService."""

    # Deduplication
    dedup_window_seconds: int = 3600  # 1 hour
    dedup_enabled: bool = True

    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_channel: Dict[str, int] = field(default_factory=dict)

    # Default channels by severity
    default_channels: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "critical": ["pagerduty", "slack", "email"],
            "high": ["slack", "email"],
            "medium": ["slack"],
            "low": ["log"],
            "info": ["log"],
        }
    )

    # Silence periods
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)

    # Notification settings
    email_from: str = "alerts@quantitative-platform.com"
    slack_webhook_url: str = ""
    pagerduty_api_key: str = ""
    pagerduty_service_id: str = ""

    # Logging
    log_all_alerts: bool = True
    log_path: str = "logs/core/alerts"

    # Callbacks
    notification_callback: Optional[Callable[[AlertChannel, Alert], NotificationResult]] = None


# =============================================================================
# Default Notification Handlers
# =============================================================================


class NotificationHandler:
    """Base notification handler."""

    def send(self, alert: Alert, target: str) -> NotificationResult:
        raise NotImplementedError


class LogNotificationHandler(NotificationHandler):
    """Log notification handler."""

    def send(self, alert: Alert, target: str = "") -> NotificationResult:
        logger.warning(
            f"ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.description} "
            f"(source={alert.source}, value={alert.metric_value})"
        )
        return NotificationResult(
            alert_id=alert.alert_id,
            channel=AlertChannel.LOG,
            target="log",
            success=True,
        )


class SlackNotificationHandler(NotificationHandler):
    """Slack notification handler with real webhook delivery.

    Supports both production mode (real HTTP POST to Slack webhook) and
    simulation mode (logging only, for testing/development).

    Args:
        webhook_url: Slack Incoming Webhook URL.
        simulation_mode: If True, logs messages instead of sending HTTP requests.
            Defaults to False (production mode).
        timeout_seconds: HTTP request timeout. Defaults to 10 seconds.
    """

    def __init__(
        self,
        webhook_url: str,
        simulation_mode: bool = False,
        timeout_seconds: float = 10.0,
    ):
        self.webhook_url = webhook_url
        self.simulation_mode = simulation_mode
        self.timeout_seconds = timeout_seconds

    def send(self, alert: Alert, target: str = "") -> NotificationResult:
        """Send alert to Slack webhook.

        In production mode, performs HTTP POST to webhook URL.
        In simulation mode, logs the message for testing purposes.
        """
        severity_emoji = {
            AlertSeverity.CRITICAL: ":rotating_light:",
            AlertSeverity.HIGH: ":warning:",
            AlertSeverity.MEDIUM: ":large_orange_diamond:",
            AlertSeverity.LOW: ":information_source:",
            AlertSeverity.INFO: ":speech_balloon:",
        }

        message = {
            "text": f"{severity_emoji.get(alert.severity, ':bell:')} *{alert.name}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{alert.name}*\n{alert.description}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Severity:* {alert.severity.value}"},
                        {"type": "mrkdwn", "text": f"*Source:* {alert.source}"},
                        {"type": "mrkdwn", "text": f"*Value:* {alert.metric_value}"},
                        {"type": "mrkdwn", "text": f"*Threshold:* {alert.threshold}"},
                    ],
                },
            ],
        }

        webhook_target = target or self.webhook_url

        if self.simulation_mode:
            logger.info(f"[SIMULATION] Slack notification to {webhook_target}: {message['text']}")
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.SLACK,
                target=webhook_target,
                success=True,
            )

        # Production mode: send real HTTP POST to Slack webhook
        try:
            import requests

            response = requests.post(
                webhook_target,
                json=message,
                timeout=self.timeout_seconds,
                headers={"Content-Type": "application/json"},
            )

            success = response.status_code == 200
            error_msg = (
                None
                if success
                else f"Slack API returned {response.status_code}: {response.text[:200]}"
            )

            if success:
                logger.info(f"Slack notification sent successfully: {alert.alert_id}")
            else:
                logger.error(f"Slack notification failed: {error_msg}")

            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.SLACK,
                target=webhook_target,
                success=success,
                error=error_msg,
            )
        except requests.exceptions.Timeout:
            error_msg = f"Slack webhook request timed out after {self.timeout_seconds}s"
            logger.error(error_msg)
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.SLACK,
                target=webhook_target,
                success=False,
                error=error_msg,
            )
        except requests.exceptions.RequestException as e:
            error_msg = f"Slack webhook request failed: {str(e)}"
            logger.error(error_msg)
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.SLACK,
                target=webhook_target,
                success=False,
                error=error_msg,
            )


class EmailNotificationHandler(NotificationHandler):
    """Email notification handler with SMTP delivery.

    Supports both production mode (real SMTP sending) and simulation mode
    (logging only, for testing/development).

    Args:
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port. Defaults to 587 (TLS).
        from_addr: Sender email address.
        username: SMTP authentication username (optional).
        password: SMTP authentication password (optional).
        use_tls: Whether to use TLS. Defaults to True.
        simulation_mode: If True, logs messages instead of sending. Defaults to False.
        timeout_seconds: SMTP connection timeout. Defaults to 30 seconds.
    """

    def __init__(
        self,
        smtp_host: str = "",
        smtp_port: int = 587,
        from_addr: str = "",
        username: str = "",
        password: str = "",
        use_tls: bool = True,
        simulation_mode: bool = False,
        timeout_seconds: float = 30.0,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_addr = from_addr
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.simulation_mode = simulation_mode
        self.timeout_seconds = timeout_seconds

    def send(self, alert: Alert, target: str = "") -> NotificationResult:
        """Send alert via email.

        In production mode, sends via SMTP.
        In simulation mode, logs the message for testing purposes.
        """
        subject = f"[{alert.severity.value.upper()}] Alert: {alert.name}"
        body = f"""Alert: {alert.name}
Severity: {alert.severity.value}
Source: {alert.source}

{alert.description}

Metric: {alert.metric_name} = {alert.metric_value} (threshold: {alert.threshold})

Triggered at: {alert.triggered_at}
"""

        if self.simulation_mode:
            logger.info(f"[SIMULATION] Email notification to {target}: {subject}")
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.EMAIL,
                target=target,
                success=True,
            )

        # Production mode: send real email via SMTP
        if not self.smtp_host or not self.from_addr:
            error_msg = "SMTP host and from_addr must be configured for production mode"
            logger.error(error_msg)
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.EMAIL,
                target=target,
                success=False,
                error=error_msg,
            )

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg["From"] = self.from_addr
            msg["To"] = target
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(
                self.smtp_host, self.smtp_port, timeout=self.timeout_seconds
            ) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.from_addr, target, msg.as_string())

            logger.info(f"Email notification sent successfully to {target}: {alert.alert_id}")
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.EMAIL,
                target=target,
                success=True,
            )
        except smtplib.SMTPException as e:
            error_msg = f"SMTP error sending email: {str(e)}"
            logger.error(error_msg)
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.EMAIL,
                target=target,
                success=False,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Failed to send email: {str(e)}"
            logger.error(error_msg)
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.EMAIL,
                target=target,
                success=False,
                error=error_msg,
            )


class PagerDutyNotificationHandler(NotificationHandler):
    """PagerDuty notification handler with Events API v2 delivery.

    Supports both production mode (real HTTP POST to PagerDuty Events API)
    and simulation mode (logging only, for testing/development).

    Args:
        api_key: PagerDuty Integration/Routing Key.
        service_id: PagerDuty Service ID for reference.
        simulation_mode: If True, logs messages instead of sending. Defaults to False.
        timeout_seconds: HTTP request timeout. Defaults to 10 seconds.
    """

    PAGERDUTY_EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(
        self,
        api_key: str,
        service_id: str,
        simulation_mode: bool = False,
        timeout_seconds: float = 10.0,
    ):
        self.api_key = api_key
        self.service_id = service_id
        self.simulation_mode = simulation_mode
        self.timeout_seconds = timeout_seconds

    def send(self, alert: Alert, target: str = "") -> NotificationResult:
        """Send alert to PagerDuty Events API v2.

        In production mode, sends HTTP POST to PagerDuty.
        In simulation mode, logs the event for testing purposes.
        """
        event = {
            "routing_key": self.api_key,
            "event_action": "trigger",
            "dedup_key": alert.fingerprint,
            "payload": {
                "summary": f"{alert.name}: {alert.description}",
                "severity": self._map_severity(alert.severity),
                "source": alert.source,
                "custom_details": {
                    "metric": alert.metric_name,
                    "value": alert.metric_value,
                    "threshold": alert.threshold,
                    "alert_id": alert.alert_id,
                },
            },
        }

        if self.simulation_mode:
            logger.info(f"[SIMULATION] PagerDuty notification: {event['payload']['summary']}")
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.PAGERDUTY,
                target=self.service_id,
                success=True,
            )

        # Production mode: send real HTTP POST to PagerDuty Events API v2
        try:
            import requests

            response = requests.post(
                self.PAGERDUTY_EVENTS_URL,
                json=event,
                timeout=self.timeout_seconds,
                headers={"Content-Type": "application/json"},
            )

            # PagerDuty returns 202 Accepted for successful events
            success = response.status_code in (200, 202)
            error_msg = (
                None
                if success
                else f"PagerDuty API returned {response.status_code}: {response.text[:200]}"
            )

            if success:
                logger.info(f"PagerDuty notification sent successfully: {alert.alert_id}")
            else:
                logger.error(f"PagerDuty notification failed: {error_msg}")

            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.PAGERDUTY,
                target=self.service_id,
                success=success,
                error=error_msg,
            )
        except requests.exceptions.Timeout:
            error_msg = f"PagerDuty request timed out after {self.timeout_seconds}s"
            logger.error(error_msg)
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.PAGERDUTY,
                target=self.service_id,
                success=False,
                error=error_msg,
            )
        except requests.exceptions.RequestException as e:
            error_msg = f"PagerDuty request failed: {str(e)}"
            logger.error(error_msg)
            return NotificationResult(
                alert_id=alert.alert_id,
                channel=AlertChannel.PAGERDUTY,
                target=self.service_id,
                success=False,
                error=error_msg,
            )

    def _map_severity(self, severity: AlertSeverity) -> str:
        mapping = {
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.HIGH: "error",
            AlertSeverity.MEDIUM: "warning",
            AlertSeverity.LOW: "info",
            AlertSeverity.INFO: "info",
        }
        return mapping.get(severity, "info")


# =============================================================================
# Main Class
# =============================================================================


class AlertingService:
    """
    Comprehensive Alerting Service per DORA Article 14.

    Features:
    - Multi-channel notifications
    - Escalation policies
    - Alert deduplication
    - Rate limiting
    - Maintenance windows

    Usage:
        config = AlertingConfig()
        service = AlertingService(config)

        # Create alert rule
        rule = service.create_rule(
            name="High CPU Usage",
            condition_type="threshold",
            metric_name="cpu_percent",
            threshold_value=90.0,
            severity=AlertSeverity.HIGH,
        )

        # Trigger alert
        alert = service.trigger_alert(
            rule_id=rule.rule_id,
            metric_value=95.0,
            source="web-server-1",
        )

        # Acknowledge
        service.acknowledge_alert(alert.alert_id, "user@example.com")
    """

    def __init__(self, config: Optional[AlertingConfig] = None):
        """Initialize Alerting Service."""
        self.config = config or AlertingConfig()

        # Data stores
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._policies: Dict[str, EscalationPolicy] = {}
        self._active_fingerprints: Dict[str, str] = {}  # fingerprint -> alert_id

        # Notification handlers
        self._handlers: Dict[AlertChannel, NotificationHandler] = {
            AlertChannel.LOG: LogNotificationHandler(),
        }

        # Rate limiting
        self._rate_limit_counters: Dict[str, List[float]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Logging
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Initialize default handlers
        self._init_handlers()

        # Durable persistence (P2 #21): alerts survive restarts.
        self._durable = None
        try:
            from services.core.durable_store import DurableAlertStore

            self._durable = DurableAlertStore(str(self._log_path / "alerts.db"))
            self._rehydrate_alerts()
        except Exception as e:  # pragma: no cover - persistence must not block startup
            logger.warning("durable alert store unavailable: %s", e)

        logger.info("AlertingService initialized")

    def _rehydrate_alerts(self) -> None:
        """Reload persisted alerts into memory after a restart (P2 #21)."""
        if self._durable is None:
            return
        loaded = 0
        for d in self._durable.load_all():
            try:
                a = self._alert_from_dict(d)
                self._alerts[a.alert_id] = a
                if a.status == AlertStatus.TRIGGERED and a.fingerprint:
                    self._active_fingerprints[a.fingerprint] = a.alert_id
                loaded += 1
            except Exception:
                continue
        if loaded:
            logger.info("rehydrated %d persisted alerts", loaded)

    @staticmethod
    def _alert_from_dict(d: Dict[str, Any]) -> "Alert":
        d = dict(d)

        def _enum(cls, val, default):
            try:
                return cls(val)
            except Exception:
                return default

        d["severity"] = _enum(AlertSeverity, d.get("severity"), AlertSeverity.MEDIUM)
        d["status"] = _enum(AlertStatus, d.get("status"), AlertStatus.TRIGGERED)
        d["escalation_level"] = _enum(
            EscalationLevel, d.get("escalation_level"), EscalationLevel.L1
        )
        known = {f.name for f in dataclasses.fields(Alert)}
        return Alert(**{k: v for k, v in d.items() if k in known})

    def _persist_alert(self, alert: "Alert") -> None:
        if self._durable is not None:
            try:
                self._durable.save(alert)
            except Exception:  # pragma: no cover
                pass

    def _init_handlers(self) -> None:
        """Initialize notification handlers."""
        if self.config.slack_webhook_url:
            self._handlers[AlertChannel.SLACK] = SlackNotificationHandler(
                self.config.slack_webhook_url
            )

        if self.config.pagerduty_api_key:
            self._handlers[AlertChannel.PAGERDUTY] = PagerDutyNotificationHandler(
                self.config.pagerduty_api_key,
                self.config.pagerduty_service_id,
            )

        self._handlers[AlertChannel.EMAIL] = EmailNotificationHandler(
            from_addr=self.config.email_from,
        )

    # =========================================================================
    # Rule Management
    # =========================================================================

    def create_rule(
        self,
        name: str,
        condition_type: str,
        metric_name: str,
        threshold_value: float,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        comparison: str = ">",
        channels: Optional[List[AlertChannel]] = None,
        escalation_policy_id: str = "",
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> AlertRule:
        """Create an alert rule."""
        # Get default channels for severity if not specified
        if channels is None:
            default_channels = self.config.default_channels.get(severity.value, ["log"])
            channels = [AlertChannel(c) for c in default_channels]

        rule = AlertRule(
            name=name,
            description=description or f"Alert when {metric_name} {comparison} {threshold_value}",
            condition_type=condition_type,
            metric_name=metric_name,
            threshold_value=threshold_value,
            comparison=comparison,
            severity=severity,
            channels=channels,
            escalation_policy_id=escalation_policy_id,
            tags=tags or [],
        )

        with self._lock:
            self._rules[rule.rule_id] = rule

        self._log_event("rule_created", {"rule_id": rule.rule_id, "name": name})
        return rule

    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get rule by ID."""
        with self._lock:
            return self._rules.get(rule_id)

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                self._log_event("rule_deleted", {"rule_id": rule_id})
                return True
            return False

    # =========================================================================
    # Escalation Policies
    # =========================================================================

    def create_escalation_policy(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        repeat_interval_minutes: int = 30,
        max_escalations: int = 5,
    ) -> EscalationPolicy:
        """Create an escalation policy."""
        policy = EscalationPolicy(
            name=name,
            steps=steps,
            repeat_interval_minutes=repeat_interval_minutes,
            max_escalations=max_escalations,
        )

        with self._lock:
            self._policies[policy.policy_id] = policy

        return policy

    def get_escalation_policy(self, policy_id: str) -> Optional[EscalationPolicy]:
        """Get escalation policy by ID."""
        with self._lock:
            return self._policies.get(policy_id)

    # =========================================================================
    # Alert Lifecycle
    # =========================================================================

    def trigger_alert(
        self,
        rule_id: str,
        metric_value: float,
        source: str,
        labels: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> Optional[Alert]:
        """
        Trigger an alert from a rule evaluation.

        Args:
            rule_id: Rule that triggered
            metric_value: Current metric value
            source: Alert source
            labels: Additional labels
            description: Override description

        Returns:
            Alert if triggered, None if deduplicated or rate limited
        """
        with self._lock:
            if rule_id not in self._rules:
                logger.error(f"Rule not found: {rule_id}")
                return None

            rule = self._rules[rule_id]

        if not rule.is_enabled:
            return None

        # Check maintenance windows
        if self._in_maintenance_window():
            logger.info(f"Alert suppressed due to maintenance window: {rule.name}")
            return None

        # Create alert
        alert = Alert(
            rule_id=rule_id,
            name=rule.name,
            description=description or rule.description,
            severity=rule.severity,
            category=rule.category,
            source=source,
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold=rule.threshold_value,
            labels=labels or {},
        )

        # Check deduplication
        if self.config.dedup_enabled:
            existing_alert_id = self._active_fingerprints.get(alert.fingerprint)
            if existing_alert_id:
                logger.debug(f"Alert deduplicated: {alert.fingerprint}")
                return None

        # Check rate limiting
        if not self._check_rate_limit():
            logger.warning("Alert rate limited")
            return None

        # Store alert
        with self._lock:
            self._alerts[alert.alert_id] = alert
            self._active_fingerprints[alert.fingerprint] = alert.alert_id
            self._persist_alert(alert)  # P2 #21: durable

        # Send notifications
        self._send_notifications(alert, rule.channels)

        self._log_event(
            "alert_triggered",
            {
                "alert_id": alert.alert_id,
                "rule_id": rule_id,
                "severity": alert.severity.value,
                "source": source,
            },
        )

        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        notes: str = "",
    ) -> Optional[Alert]:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return None

            alert = self._alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(timezone.utc).isoformat()
            alert.acknowledged_by = acknowledged_by
            if notes:
                alert.resolution_notes = notes
            self._persist_alert(alert)  # P2 #21: durable

        self._log_event(
            "alert_acknowledged",
            {
                "alert_id": alert_id,
                "by": acknowledged_by,
            },
        )

        return alert

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_notes: str = "",
    ) -> Optional[Alert]:
        """Resolve an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                return None

            alert = self._alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc).isoformat()
            alert.resolved_by = resolved_by
            alert.resolution_notes = resolution_notes

            # Remove from active fingerprints
            if alert.fingerprint in self._active_fingerprints:
                del self._active_fingerprints[alert.fingerprint]
            self._persist_alert(alert)  # P2 #21: durable

        self._log_event(
            "alert_resolved",
            {
                "alert_id": alert_id,
                "by": resolved_by,
            },
        )

        return alert

    def escalate_alert(self, alert_id: str) -> Optional[Alert]:
        """Escalate an alert to the next level."""
        with self._lock:
            if alert_id not in self._alerts:
                return None

            alert = self._alerts[alert_id]

            # Get escalation policy
            rule = self._rules.get(alert.rule_id)
            if not rule or not rule.escalation_policy_id:
                return alert

            policy = self._policies.get(rule.escalation_policy_id)
            if not policy:
                return alert

            # Check max escalations
            if alert.escalation_count >= policy.max_escalations:
                logger.warning(f"Max escalations reached for alert {alert_id}")
                return alert

            # Escalate
            alert.escalation_count += 1
            current_level = min(alert.escalation_count, len(EscalationLevel))
            alert.escalation_level = list(EscalationLevel)[current_level - 1]
            alert.status = AlertStatus.ESCALATED
            alert.last_escalation_at = datetime.now(timezone.utc).isoformat()

            # Get step configuration
            if alert.escalation_count <= len(policy.steps):
                step = policy.steps[alert.escalation_count - 1]
                channels = [AlertChannel(c) for c in step.get("channels", ["slack"])]
                self._send_notifications(alert, channels, escalation=True)

        self._log_event(
            "alert_escalated",
            {
                "alert_id": alert_id,
                "level": alert.escalation_level.value,
                "count": alert.escalation_count,
            },
        )

        return alert

    # =========================================================================
    # Notifications
    # =========================================================================

    def _send_notifications(
        self,
        alert: Alert,
        channels: List[AlertChannel],
        escalation: bool = False,
    ) -> List[NotificationResult]:
        """Send notifications through specified channels."""
        results = []

        for channel in channels:
            handler = self._handlers.get(channel)
            if not handler:
                logger.warning(f"No handler for channel: {channel.value}")
                continue

            try:
                result = handler.send(alert, "")
                results.append(result)

                # Record notification
                with self._lock:
                    if alert.alert_id in self._alerts:
                        self._alerts[alert.alert_id].notifications_sent.append(
                            {
                                "channel": channel.value,
                                "success": result.success,
                                "sent_at": result.sent_at,
                                "escalation": escalation,
                            }
                        )

            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification: {e}")
                results.append(
                    NotificationResult(
                        alert_id=alert.alert_id,
                        channel=channel,
                        success=False,
                        error=str(e),
                    )
                )

        return results

    # =========================================================================
    # Queries
    # =========================================================================

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        with self._lock:
            return self._alerts.get(alert_id)

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None,
    ) -> List[Alert]:
        """Get active (non-resolved) alerts."""
        with self._lock:
            alerts = [
                a
                for a in self._alerts.values()
                if a.status not in (AlertStatus.RESOLVED, AlertStatus.EXPIRED)
            ]

            if severity:
                alerts = [a for a in alerts if a.severity == severity]

            if source:
                alerts = [a for a in alerts if a.source == source]

            return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)

    def get_alert_history(
        self,
        hours: int = 24,
        status: Optional[AlertStatus] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alert history."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

        with self._lock:
            alerts = [a for a in self._alerts.values() if a.triggered_at > cutoff]

            if status:
                alerts = [a for a in alerts if a.status == status]

            alerts.sort(key=lambda a: a.triggered_at, reverse=True)
            return alerts[:limit]

    # =========================================================================
    # Utilities
    # =========================================================================

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now(timezone.utc).timestamp()
        window_start = now - 60  # 1 minute window

        with self._lock:
            # Clean old entries
            if "global" not in self._rate_limit_counters:
                self._rate_limit_counters["global"] = []

            self._rate_limit_counters["global"] = [
                t for t in self._rate_limit_counters["global"] if t > window_start
            ]

            # Check limit
            if len(self._rate_limit_counters["global"]) >= self.config.rate_limit_per_minute:
                return False

            # Record this alert
            self._rate_limit_counters["global"].append(now)
            return True

    def _in_maintenance_window(self) -> bool:
        """Check if currently in a maintenance window."""
        now = datetime.now(timezone.utc)

        for window in self.config.maintenance_windows:
            start = datetime.fromisoformat(window.get("start", ""))
            end = datetime.fromisoformat(window.get("end", ""))
            if start <= now <= end:
                return True

        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        with self._lock:
            all_alerts = list(self._alerts.values())

        now = datetime.now(timezone.utc)
        last_24h = (now - timedelta(hours=24)).isoformat()
        last_7d = (now - timedelta(days=7)).isoformat()

        alerts_24h = [a for a in all_alerts if a.triggered_at > last_24h]
        alerts_7d = [a for a in all_alerts if a.triggered_at > last_7d]

        return {
            "timestamp": now.isoformat(),
            "total_rules": len(self._rules),
            "active_rules": sum(1 for r in self._rules.values() if r.is_enabled),
            "total_alerts": len(all_alerts),
            "active_alerts": len(
                [
                    a
                    for a in all_alerts
                    if a.status not in (AlertStatus.RESOLVED, AlertStatus.EXPIRED)
                ]
            ),
            "alerts_24h": len(alerts_24h),
            "alerts_7d": len(alerts_7d),
            "by_severity_24h": {
                s.value: sum(1 for a in alerts_24h if a.severity == s) for s in AlertSeverity
            },
            "by_status": {
                s.value: sum(1 for a in all_alerts if a.status == s) for s in AlertStatus
            },
            "mttr_minutes": self._calculate_mttr(alerts_7d),
        }

    def _calculate_mttr(self, alerts: List[Alert]) -> float:
        """Calculate Mean Time to Resolve."""
        resolved = [a for a in alerts if a.status == AlertStatus.RESOLVED and a.resolved_at]

        if not resolved:
            return 0.0

        total_minutes = 0.0
        for alert in resolved:
            triggered = datetime.fromisoformat(alert.triggered_at.replace("Z", "+00:00"))
            resolved_time = datetime.fromisoformat(alert.resolved_at.replace("Z", "+00:00"))
            total_minutes += (resolved_time - triggered).total_seconds() / 60

        return round(total_minutes / len(resolved), 2)

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        if not self.config.log_all_alerts:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_alerting_service(
    config: Optional[AlertingConfig] = None,
) -> AlertingService:
    """
    Create an AlertingService instance.

    Args:
        config: Optional configuration

    Returns:
        Configured AlertingService instance
    """
    return AlertingService(config=config)
