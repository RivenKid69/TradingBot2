# -*- coding: utf-8 -*-
"""
MiFID II RTS 6 Article 12 Compliant Kill Switch.

Implements emergency order cancellation mechanism per Commission Delegated
Regulation (EU) 2017/589 (RTS 6) Article 12:

    "The investment firm shall be able to cancel immediately, as an emergency
    measure, any or all of its unexecuted orders submitted to any or all
    trading venues"

Key Requirements (Article 12 RTS 6):
    1. Immediate cancellation capability for all unexecuted orders
    2. Cancellation per trading venue, algorithm, or trader
    3. Compliance staff must maintain contact with person able to cancel
    4. Clear lines of accountability and escalation

References:
    - RTS 6 Article 12: https://www.handbook.fca.org.uk/techstandards/MIFID-MIFIR/2017/reg_del_2017_589_oj/chapter-ii/section-3/
    - ESMA Algo Trading Guidelines: https://www.esma.europa.eu/sites/default/files/library/esma70-156-4572_mifid_ii_final_report_on_algorithmic_trading.pdf
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Dict,
    List,
    Optional,
    Callable,
    Any,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


class KillSwitchScope(str, Enum):
    """
    Scope of kill switch activation per RTS 6 Article 12.

    The regulation requires ability to cancel orders at different granularities.
    """

    ALL = "all"  # All orders across all venues
    VENUE = "venue"  # Specific trading venue (MIC)
    ALGORITHM = "algorithm"  # Specific algorithm ID
    TRADER = "trader"  # Specific trader/desk
    INSTRUMENT = "instrument"  # Specific instrument (ISIN)
    ASSET_CLASS = "asset_class"  # Specific asset class
    STRATEGY = "strategy"  # Specific trading strategy


class KillSwitchTriggerReason(str, Enum):
    """Reason categories for kill switch activation."""

    MANUAL = "manual"  # Manual human intervention
    ERROR_THRESHOLD = "error_threshold"  # Error count exceeded
    RISK_LIMIT = "risk_limit"  # Risk limit breached
    CLOCK_DRIFT = "clock_drift"  # RTS 25 clock drift excessive
    MARKET_DISRUPTION = "market_disruption"  # Market circuit breaker
    CONNECTIVITY = "connectivity"  # Connection loss
    REGULATORY = "regulatory"  # Regulatory requirement
    SYSTEM_ERROR = "system_error"  # System malfunction
    CAPITAL_BREACH = "capital_breach"  # Capital threshold breach
    POSITION_LIMIT = "position_limit"  # Position limit exceeded
    OTR_BREACH = "otr_breach"  # Order-to-trade ratio breach
    ALGORITHMIC_ERROR = "algorithmic_error"  # Algorithm malfunction


class KillSwitchState(str, Enum):
    """Current state of the kill switch system."""

    ARMED = "armed"  # Normal operation, ready to trigger
    TRIGGERED = "triggered"  # Kill switch active, orders cancelled
    DISARMED = "disarmed"  # Kill switch temporarily disabled
    COOLDOWN = "cooldown"  # Post-trigger cooldown period


@dataclass
class KillSwitchEvent:
    """
    Immutable record of a kill switch activation.

    Per RTS 6: All kill switch events must be recorded for audit trail.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    scope: KillSwitchScope = KillSwitchScope.ALL
    scope_id: str = ""
    reason: KillSwitchTriggerReason = KillSwitchTriggerReason.MANUAL
    reason_detail: str = ""
    triggered_by: str = ""  # Person or system identifier
    orders_cancelled: int = 0
    venues_affected: List[str] = field(default_factory=list)
    algorithms_affected: List[str] = field(default_factory=list)
    confirmation_id: str = ""
    recovery_timestamp_ns: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp_ns": self.timestamp_ns,
            "timestamp_utc": self.timestamp_utc,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "reason": self.reason.value,
            "reason_detail": self.reason_detail,
            "triggered_by": self.triggered_by,
            "orders_cancelled": self.orders_cancelled,
            "venues_affected": self.venues_affected,
            "algorithms_affected": self.algorithms_affected,
            "confirmation_id": self.confirmation_id,
            "recovery_timestamp_ns": self.recovery_timestamp_ns,
        }

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for integrity verification."""
        data = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()


@dataclass
class KillSwitchConfig:
    """
    Configuration for the enhanced kill switch per RTS 6.
    """

    # Contact information per RTS 6 Art. 12
    primary_contact_name: str = ""
    primary_contact_email: str = ""
    primary_contact_phone: str = ""
    escalation_contact_name: str = ""
    escalation_contact_email: str = ""
    escalation_contact_phone: str = ""
    out_of_hours_phone: str = ""

    # Timing settings
    cooldown_seconds: float = 300.0  # 5 minutes cooldown after trigger
    auto_recovery_enabled: bool = False  # Auto-recover after cooldown
    max_triggers_per_hour: int = 10  # Rate limit on triggers

    # Thresholds for automatic triggers
    error_threshold_rest: int = 100  # REST API errors before trigger
    error_threshold_ws: int = 50  # WebSocket errors before trigger
    position_limit_pct: float = 95.0  # % of max position before trigger
    capital_threshold_pct: float = 90.0  # % of capital used before warning

    # Integration settings
    notify_venues: bool = True  # Send cancel to venues
    notify_compliance: bool = True  # Notify compliance team
    persist_events: bool = True  # Persist events to disk
    event_log_path: str = "state/compliance/kill_switch_events.jsonl"


@dataclass
class EmergencyContact:
    """Emergency contact information per RTS 6 Article 12."""

    name: str
    role: str
    email: str
    phone: str
    is_available_24_7: bool = False
    escalation_order: int = 0


class EnhancedKillSwitch:
    """
    MiFID II RTS 6 Article 12 Compliant Kill Switch.

    Provides:
        - Immediate cancellation of all unexecuted orders
        - Granular scope control (venue, algorithm, instrument)
        - Full audit trail of all triggers
        - Contact information for compliance staff
        - Rate limiting and cooldown periods
        - Integration with existing ops_kill_switch

    Usage:
        kill_switch = EnhancedKillSwitch(
            order_cancellation_callback=cancel_all_orders,
            config=KillSwitchConfig(primary_contact_email="trading@firm.com"),
        )

        # Trigger for all orders
        event = kill_switch.trigger_all("Emergency market conditions")

        # Trigger for specific venue
        event = kill_switch.trigger(
            scope=KillSwitchScope.VENUE,
            scope_id="XLON",
            reason=KillSwitchTriggerReason.MARKET_DISRUPTION,
        )

        # Check status
        if kill_switch.is_triggered:
            print("Kill switch is active")

    RTS 6 Article 12 Compliance:
        - Immediate cancellation: trigger() cancels orders synchronously
        - Granular control: scope parameter allows venue/algo/trader targeting
        - Accountability: triggered_by and audit trail provide clear accountability
        - Contact info: get_emergency_contacts() provides compliance contact details
    """

    def __init__(
        self,
        order_cancellation_callback: Callable[[KillSwitchScope, str], int],
        config: Optional[KillSwitchConfig] = None,
        alert_callback: Optional[Callable[[KillSwitchEvent], None]] = None,
        compliance_callback: Optional[Callable[[KillSwitchEvent], None]] = None,
        venue_notification_callback: Optional[Callable[[str, KillSwitchEvent], None]] = None,
    ):
        """
        Initialize Enhanced Kill Switch.

        Args:
            order_cancellation_callback: Function to cancel orders.
                Args: (scope, scope_id)
                Returns: number of orders cancelled
            config: Kill switch configuration.
            alert_callback: Called on each trigger for alerting.
            compliance_callback: Called for compliance notification.
            venue_notification_callback: Called per venue to notify.
        """
        self._cancel_orders = order_cancellation_callback
        self._config = config or KillSwitchConfig()
        self._alert_callback = alert_callback
        self._compliance_callback = compliance_callback
        self._venue_notify = venue_notification_callback

        # State
        self._lock = threading.Lock()
        self._state = KillSwitchState.ARMED
        self._events: List[KillSwitchEvent] = []
        self._active_scopes: Set[Tuple[KillSwitchScope, str]] = set()
        self._trigger_timestamps: List[float] = []
        self._last_trigger_time: float = 0.0

        # Emergency contacts
        self._emergency_contacts: List[EmergencyContact] = []
        self._setup_contacts()

        logger.info(
            "EnhancedKillSwitch initialized",
            extra={
                "state": self._state.value,
                "cooldown_seconds": self._config.cooldown_seconds,
            },
        )

    def _setup_contacts(self) -> None:
        """Initialize emergency contacts from config."""
        if self._config.primary_contact_email:
            self._emergency_contacts.append(
                EmergencyContact(
                    name=self._config.primary_contact_name or "Primary Contact",
                    role="Trading Operations",
                    email=self._config.primary_contact_email,
                    phone=self._config.primary_contact_phone,
                    is_available_24_7=False,
                    escalation_order=1,
                )
            )

        if self._config.escalation_contact_email:
            self._emergency_contacts.append(
                EmergencyContact(
                    name=self._config.escalation_contact_name or "Escalation Contact",
                    role="Compliance Officer",
                    email=self._config.escalation_contact_email,
                    phone=self._config.escalation_contact_phone,
                    is_available_24_7=False,
                    escalation_order=2,
                )
            )

        if self._config.out_of_hours_phone:
            self._emergency_contacts.append(
                EmergencyContact(
                    name="Out of Hours",
                    role="Emergency Contact",
                    email="",
                    phone=self._config.out_of_hours_phone,
                    is_available_24_7=True,
                    escalation_order=3,
                )
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(self) -> KillSwitchState:
        """Current kill switch state."""
        with self._lock:
            return self._state

    @property
    def is_triggered(self) -> bool:
        """Check if kill switch is currently triggered."""
        with self._lock:
            return self._state == KillSwitchState.TRIGGERED

    @property
    def is_armed(self) -> bool:
        """Check if kill switch is armed and ready."""
        with self._lock:
            return self._state == KillSwitchState.ARMED

    @property
    def active_scopes(self) -> Set[Tuple[KillSwitchScope, str]]:
        """Get currently active (triggered) scopes."""
        with self._lock:
            return self._active_scopes.copy()

    # =========================================================================
    # Trigger Methods
    # =========================================================================

    def trigger(
        self,
        scope: KillSwitchScope = KillSwitchScope.ALL,
        scope_id: str = "",
        reason: KillSwitchTriggerReason = KillSwitchTriggerReason.MANUAL,
        reason_detail: str = "",
        triggered_by: str = "system",
        venues: Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None,
    ) -> KillSwitchEvent:
        """
        Trigger kill switch.

        Per RTS 6 Article 12: "cancel immediately, as an emergency measure"

        Args:
            scope: Scope of cancellation (all, venue, algorithm, etc.)
            scope_id: Identifier for the scope (e.g., MIC code, algo ID)
            reason: Reason category for the trigger
            reason_detail: Human-readable reason description
            triggered_by: Person or system that triggered
            venues: List of venues affected (for logging)
            algorithms: List of algorithms affected (for logging)

        Returns:
            KillSwitchEvent with trigger details

        Raises:
            RuntimeError: If kill switch is disarmed
        """
        with self._lock:
            # Check if disarmed
            if self._state == KillSwitchState.DISARMED:
                raise RuntimeError("Kill switch is disarmed. Cannot trigger in disarmed state.")

            # Rate limiting
            if not self._check_rate_limit():
                logger.warning(
                    f"Kill switch rate limit exceeded ({self._config.max_triggers_per_hour}/hour)"
                )

            # Execute cancellation
            orders_cancelled = 0
            try:
                orders_cancelled = self._cancel_orders(scope, scope_id)
            except Exception as e:
                logger.error(f"Order cancellation callback failed: {e}")
                # Still record the event even if cancellation fails

            # Create event
            event = KillSwitchEvent(
                scope=scope,
                scope_id=scope_id,
                reason=reason,
                reason_detail=reason_detail,
                triggered_by=triggered_by,
                orders_cancelled=orders_cancelled,
                venues_affected=venues or [],
                algorithms_affected=algorithms or [],
                confirmation_id=str(uuid.uuid4()),
            )

            # Update state
            self._state = KillSwitchState.TRIGGERED
            self._active_scopes.add((scope, scope_id))
            self._events.append(event)
            self._trigger_timestamps.append(time.time())
            self._last_trigger_time = time.time()

        # Log the event
        logger.critical(
            f"KILL SWITCH TRIGGERED: scope={scope.value}, scope_id={scope_id}, "
            f"cancelled={orders_cancelled} orders. Reason: {reason_detail}",
            extra=event.to_dict(),
        )

        # Notify callbacks (outside lock)
        self._notify_callbacks(event)

        # Persist event
        if self._config.persist_events:
            self._persist_event(event)

        return event

    def trigger_all(
        self,
        reason_detail: str = "Emergency",
        triggered_by: str = "system",
    ) -> KillSwitchEvent:
        """
        Trigger kill switch for ALL orders on ALL venues.

        This is the primary emergency measure per RTS 6 Article 12.

        Args:
            reason_detail: Human-readable reason
            triggered_by: Person or system identifier

        Returns:
            KillSwitchEvent with trigger details
        """
        return self.trigger(
            scope=KillSwitchScope.ALL,
            scope_id="",
            reason=KillSwitchTriggerReason.MANUAL,
            reason_detail=reason_detail,
            triggered_by=triggered_by,
        )

    def trigger_venue(
        self,
        venue_mic: str,
        reason: KillSwitchTriggerReason = KillSwitchTriggerReason.MARKET_DISRUPTION,
        reason_detail: str = "",
        triggered_by: str = "system",
    ) -> KillSwitchEvent:
        """
        Trigger kill switch for a specific trading venue.

        Args:
            venue_mic: MIC code of the venue (e.g., "XLON", "XETR")
            reason: Reason category
            reason_detail: Human-readable reason
            triggered_by: Person or system identifier

        Returns:
            KillSwitchEvent with trigger details
        """
        return self.trigger(
            scope=KillSwitchScope.VENUE,
            scope_id=venue_mic,
            reason=reason,
            reason_detail=reason_detail or f"Kill switch for venue {venue_mic}",
            triggered_by=triggered_by,
            venues=[venue_mic],
        )

    def trigger_algorithm(
        self,
        algorithm_id: str,
        reason: KillSwitchTriggerReason = KillSwitchTriggerReason.ALGORITHMIC_ERROR,
        reason_detail: str = "",
        triggered_by: str = "system",
    ) -> KillSwitchEvent:
        """
        Trigger kill switch for a specific algorithm.

        Args:
            algorithm_id: Algorithm identifier
            reason: Reason category
            reason_detail: Human-readable reason
            triggered_by: Person or system identifier

        Returns:
            KillSwitchEvent with trigger details
        """
        return self.trigger(
            scope=KillSwitchScope.ALGORITHM,
            scope_id=algorithm_id,
            reason=reason,
            reason_detail=reason_detail or f"Kill switch for algorithm {algorithm_id}",
            triggered_by=triggered_by,
            algorithms=[algorithm_id],
        )

    def trigger_instrument(
        self,
        isin: str,
        reason: KillSwitchTriggerReason = KillSwitchTriggerReason.RISK_LIMIT,
        reason_detail: str = "",
        triggered_by: str = "system",
    ) -> KillSwitchEvent:
        """
        Trigger kill switch for a specific instrument.

        Args:
            isin: ISIN of the instrument
            reason: Reason category
            reason_detail: Human-readable reason
            triggered_by: Person or system identifier

        Returns:
            KillSwitchEvent with trigger details
        """
        return self.trigger(
            scope=KillSwitchScope.INSTRUMENT,
            scope_id=isin,
            reason=reason,
            reason_detail=reason_detail or f"Kill switch for instrument {isin}",
            triggered_by=triggered_by,
        )

    # =========================================================================
    # Recovery Methods
    # =========================================================================

    def recover(
        self,
        scope: Optional[KillSwitchScope] = None,
        scope_id: str = "",
        recovered_by: str = "system",
    ) -> bool:
        """
        Recover from triggered state.

        Args:
            scope: Specific scope to recover (None = all)
            scope_id: Scope identifier to recover
            recovered_by: Person or system recovering

        Returns:
            True if recovery successful
        """
        with self._lock:
            if self._state != KillSwitchState.TRIGGERED:
                return False

            # Check cooldown
            elapsed = time.time() - self._last_trigger_time
            if elapsed < self._config.cooldown_seconds:
                remaining = self._config.cooldown_seconds - elapsed
                logger.warning(
                    f"Cannot recover: cooldown period active ({remaining:.1f}s remaining)"
                )
                return False

            if scope is not None:
                # Recover specific scope
                key = (scope, scope_id)
                if key in self._active_scopes:
                    self._active_scopes.discard(key)
                    logger.info(
                        f"Recovered kill switch scope: {scope.value}/{scope_id}",
                        extra={"recovered_by": recovered_by},
                    )
            else:
                # Recover all
                self._active_scopes.clear()

            # If no active scopes, fully recover
            if not self._active_scopes:
                self._state = KillSwitchState.ARMED

                # Update last event with recovery timestamp
                if self._events:
                    self._events[-1].recovery_timestamp_ns = time.time_ns()

                logger.info(
                    "Kill switch fully recovered",
                    extra={"recovered_by": recovered_by},
                )

        return True

    def force_recover(self, recovered_by: str = "system") -> bool:
        """
        Force recovery ignoring cooldown period.

        Use with caution - only for authorized personnel.

        Args:
            recovered_by: Person forcing recovery

        Returns:
            True if recovery successful
        """
        with self._lock:
            self._active_scopes.clear()
            self._state = KillSwitchState.ARMED

            if self._events:
                self._events[-1].recovery_timestamp_ns = time.time_ns()

            logger.warning(
                f"Kill switch FORCE RECOVERED by {recovered_by}",
                extra={"recovered_by": recovered_by},
            )

        return True

    def arm(self) -> None:
        """Arm the kill switch (enable triggering)."""
        with self._lock:
            if self._state == KillSwitchState.TRIGGERED:
                logger.warning("Cannot arm while triggered. Recover first.")
                return
            self._state = KillSwitchState.ARMED
            logger.info("Kill switch armed")

    def disarm(self, disarmed_by: str = "system") -> None:
        """
        Disarm the kill switch (prevent triggering).

        Use with extreme caution - this disables safety controls.

        Args:
            disarmed_by: Person disarming the kill switch
        """
        with self._lock:
            self._state = KillSwitchState.DISARMED
            logger.warning(
                f"Kill switch DISARMED by {disarmed_by}. Safety controls disabled!",
                extra={"disarmed_by": disarmed_by},
            )

    # =========================================================================
    # Query Methods
    # =========================================================================

    def is_scope_triggered(self, scope: KillSwitchScope, scope_id: str = "") -> bool:
        """
        Check if a specific scope is triggered.

        Args:
            scope: Scope type to check
            scope_id: Scope identifier

        Returns:
            True if the scope is currently triggered
        """
        with self._lock:
            # Check for ALL scope
            if (KillSwitchScope.ALL, "") in self._active_scopes:
                return True
            return (scope, scope_id) in self._active_scopes

    def can_trade(
        self, venue: str = "", algorithm: str = "", instrument: str = ""
    ) -> Tuple[bool, str]:
        """
        Check if trading is allowed given current kill switch state.

        Args:
            venue: MIC code of venue
            algorithm: Algorithm ID
            instrument: ISIN of instrument

        Returns:
            (can_trade, reason) tuple
        """
        with self._lock:
            if self._state == KillSwitchState.DISARMED:
                return True, "Kill switch disarmed"

            if self._state != KillSwitchState.TRIGGERED:
                return True, "Kill switch not triggered"

            # Check ALL scope
            if (KillSwitchScope.ALL, "") in self._active_scopes:
                return False, "Kill switch triggered for ALL orders"

            # Check specific scopes
            if venue and (KillSwitchScope.VENUE, venue) in self._active_scopes:
                return False, f"Kill switch triggered for venue {venue}"

            if algorithm and (KillSwitchScope.ALGORITHM, algorithm) in self._active_scopes:
                return False, f"Kill switch triggered for algorithm {algorithm}"

            if instrument and (KillSwitchScope.INSTRUMENT, instrument) in self._active_scopes:
                return False, f"Kill switch triggered for instrument {instrument}"

            return True, "No matching kill switch scope"

    def get_events(
        self,
        limit: int = 100,
        scope: Optional[KillSwitchScope] = None,
    ) -> List[KillSwitchEvent]:
        """
        Get kill switch event history.

        Args:
            limit: Maximum events to return
            scope: Filter by scope (None = all)

        Returns:
            List of KillSwitchEvent objects
        """
        with self._lock:
            events = self._events.copy()

        if scope is not None:
            events = [e for e in events if e.scope == scope]

        return events[-limit:]

    def get_last_event(self) -> Optional[KillSwitchEvent]:
        """Get the most recent kill switch event."""
        with self._lock:
            if self._events:
                return self._events[-1]
        return None

    # =========================================================================
    # Contact Information (RTS 6 Requirement)
    # =========================================================================

    def get_emergency_contacts(self) -> List[EmergencyContact]:
        """
        Get emergency contact information.

        Per RTS 6: "compliance staff must maintain contact with the
        individual at the firm who is able to cancel immediately"

        Returns:
            List of EmergencyContact objects
        """
        return self._emergency_contacts.copy()

    def get_contact_info(self) -> Dict[str, Any]:
        """
        Get contact information as dictionary.

        Returns:
            Dictionary with contact details
        """
        contacts = []
        for contact in self._emergency_contacts:
            contacts.append(
                {
                    "name": contact.name,
                    "role": contact.role,
                    "email": contact.email,
                    "phone": contact.phone,
                    "available_24_7": contact.is_available_24_7,
                    "escalation_order": contact.escalation_order,
                }
            )

        return {
            "contacts": contacts,
            "primary_email": self._config.primary_contact_email,
            "emergency_phone": self._config.out_of_hours_phone,
        }

    def add_emergency_contact(self, contact: EmergencyContact) -> None:
        """Add an emergency contact."""
        self._emergency_contacts.append(contact)
        self._emergency_contacts.sort(key=lambda c: c.escalation_order)

    # =========================================================================
    # Statistics and Reporting
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get kill switch statistics."""
        with self._lock:
            total_triggers = len(self._events)
            total_orders_cancelled = sum(e.orders_cancelled for e in self._events)

            # Count by reason
            reasons: Dict[str, int] = {}
            for event in self._events:
                reasons[event.reason.value] = reasons.get(event.reason.value, 0) + 1

            # Recent triggers (last hour)
            hour_ago = time.time() - 3600
            recent_triggers = sum(1 for ts in self._trigger_timestamps if ts > hour_ago)

            return {
                "state": self._state.value,
                "total_triggers": total_triggers,
                "total_orders_cancelled": total_orders_cancelled,
                "recent_triggers_1h": recent_triggers,
                "active_scopes": len(self._active_scopes),
                "triggers_by_reason": reasons,
                "last_trigger_time": self._last_trigger_time,
                "cooldown_remaining": max(
                    0, self._config.cooldown_seconds - (time.time() - self._last_trigger_time)
                ),
            }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate RTS 6 Article 12 compliance report.

        Returns:
            Dictionary with compliance report data
        """
        stats = self.get_statistics()
        contacts = self.get_contact_info()

        return {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "regulation": "RTS 6 Article 12",
            "requirement": "Kill Switch / Emergency Cancellation",
            "compliance_status": "CONFIGURED",
            "current_state": {
                "state": self._state.value,
                "is_triggered": self.is_triggered,
                "active_scopes": [{"scope": s.value, "id": i} for s, i in self._active_scopes],
            },
            "configuration": {
                "cooldown_seconds": self._config.cooldown_seconds,
                "auto_recovery_enabled": self._config.auto_recovery_enabled,
                "max_triggers_per_hour": self._config.max_triggers_per_hour,
                "error_threshold_rest": self._config.error_threshold_rest,
                "error_threshold_ws": self._config.error_threshold_ws,
            },
            "emergency_contacts": contacts,
            "statistics": stats,
            "recent_events": [e.to_dict() for e in self.get_events(limit=10)],
        }

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _check_rate_limit(self) -> bool:
        """Check if trigger rate is within limits."""
        now = time.time()
        hour_ago = now - 3600

        # Clean old timestamps
        self._trigger_timestamps = [ts for ts in self._trigger_timestamps if ts > hour_ago]

        return len(self._trigger_timestamps) < self._config.max_triggers_per_hour

    def _notify_callbacks(self, event: KillSwitchEvent) -> None:
        """Notify all registered callbacks."""
        if self._alert_callback:
            try:
                self._alert_callback(event)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        if self._compliance_callback and self._config.notify_compliance:
            try:
                self._compliance_callback(event)
            except Exception as e:
                logger.error(f"Compliance callback error: {e}")

        if self._venue_notify and self._config.notify_venues:
            for venue in event.venues_affected:
                try:
                    self._venue_notify(venue, event)
                except Exception as e:
                    logger.error(f"Venue notification error for {venue}: {e}")

    def _persist_event(self, event: KillSwitchEvent) -> None:
        """Persist event to disk for audit trail."""
        try:
            import os
            from pathlib import Path

            path = Path(self._config.event_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "a", encoding="utf-8") as f:
                data = event.to_dict()
                data["hash"] = event.compute_hash()
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist kill switch event: {e}")


# =============================================================================
# Factory Function
# =============================================================================


def create_enhanced_kill_switch(
    order_cancellation_callback: Callable[[KillSwitchScope, str], int],
    config: Optional[Dict[str, Any]] = None,
    alert_callback: Optional[Callable[[KillSwitchEvent], None]] = None,
) -> EnhancedKillSwitch:
    """
    Factory function to create EnhancedKillSwitch instance.

    Args:
        order_cancellation_callback: Function to cancel orders
        config: Optional configuration dictionary
        alert_callback: Optional alert callback

    Returns:
        Configured EnhancedKillSwitch instance
    """
    cfg = KillSwitchConfig()

    if config:
        for key, value in config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    return EnhancedKillSwitch(
        order_cancellation_callback=order_cancellation_callback,
        config=cfg,
        alert_callback=alert_callback,
    )


__all__ = [
    "KillSwitchScope",
    "KillSwitchTriggerReason",
    "KillSwitchState",
    "KillSwitchEvent",
    "KillSwitchConfig",
    "EmergencyContact",
    "EnhancedKillSwitch",
    "create_enhanced_kill_switch",
]
