# -*- coding: utf-8 -*-
"""
MiFID II NCA Notification - Article 17(2) Compliance.

This module handles notifications to National Competent Authorities (NCAs)
per MiFID II Article 17(2) requirements:

"An investment firm that engages in algorithmic trading shall notify the
competent authorities of its home Member State and of the trading venues
at which the investment firm engages in algorithmic trading as a member
or participant of those trading venues."

Key Requirements:
    - Notify NCA of algorithmic trading activities
    - Provide algorithm descriptions and strategies
    - Notify before deployment of new algorithms
    - Notify on material changes
    - Support NCA information requests

References:
    - MiFID II (Directive 2014/65/EU) Article 17(2)
    - RTS 6 (Regulation 2017/589) Article 1: Notification requirements
    - ESMA Guidelines on algorithmic trading (ESMA/2015/1464)
    - FCA SUP 10A (UK-specific requirements)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Tuple,
    Protocol,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class NCAJurisdiction(Enum):
    """National Competent Authority jurisdictions."""

    # Major EU NCAs
    FCA = "fca"
    """Financial Conduct Authority (United Kingdom)."""

    BAFIN = "bafin"
    """Bundesanstalt für Finanzdienstleistungsaufsicht (Germany)."""

    AMF = "amf"
    """Autorité des marchés financiers (France)."""

    AFM = "afm"
    """Autoriteit Financiële Markten (Netherlands)."""

    CONSOB = "consob"
    """Commissione Nazionale per le Società e la Borsa (Italy)."""

    CNMV = "cnmv"
    """Comisión Nacional del Mercado de Valores (Spain)."""

    CSSF = "cssf"
    """Commission de Surveillance du Secteur Financier (Luxembourg)."""

    FSMA = "fsma"
    """Financial Services and Markets Authority (Belgium)."""

    CBI = "cbi"
    """Central Bank of Ireland (Ireland)."""

    CMVM = "cmvm"
    """Comissão do Mercado de Valores Mobiliários (Portugal)."""

    KNF = "knf"
    """Komisja Nadzoru Finansowego (Poland)."""

    FIN = "fin"
    """Finansinspektionen (Sweden)."""

    DFSA = "dfsa"
    """Finanstilsynet (Denmark)."""

    NFSA = "nfsa"
    """Finanstilsynet (Norway)."""

    # ESMA (for pan-EU matters)
    ESMA = "esma"
    """European Securities and Markets Authority."""


class NotificationType(Enum):
    """Type of NCA notification."""

    INITIAL = "initial"
    """Initial notification of algorithmic trading activities."""

    NEW_ALGORITHM = "new_algorithm"
    """Notification of new algorithm deployment."""

    MATERIAL_CHANGE = "material_change"
    """Material change to existing algorithm."""

    DECOMMISSION = "decommission"
    """Algorithm decommissioning notification."""

    ANNUAL_REVIEW = "annual_review"
    """Annual review and update notification."""

    INCIDENT = "incident"
    """Incident notification (market disruption, etc.)."""

    INFORMATION_REQUEST = "information_request"
    """Response to NCA information request."""


class NotificationStatus(Enum):
    """Status of NCA notification."""

    DRAFT = "draft"
    """Notification being prepared."""

    PENDING_REVIEW = "pending_review"
    """Awaiting internal compliance review."""

    APPROVED = "approved"
    """Approved for submission."""

    SUBMITTED = "submitted"
    """Submitted to NCA."""

    ACKNOWLEDGED = "acknowledged"
    """Acknowledged by NCA."""

    REJECTED = "rejected"
    """Rejected (needs revision)."""

    COMPLETED = "completed"
    """Notification process completed."""


class AlgorithmCategory(Enum):
    """Algorithm category per RTS 6."""

    EXECUTION = "execution"
    """Order execution algorithm (TWAP, VWAP, etc.)."""

    MARKET_MAKING = "market_making"
    """Market making strategy."""

    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    """Statistical arbitrage strategy."""

    MOMENTUM = "momentum"
    """Momentum/trend following strategy."""

    MEAN_REVERSION = "mean_reversion"
    """Mean reversion strategy."""

    NEWS_TRADING = "news_trading"
    """News-based trading."""

    HFT = "hft"
    """High-frequency trading (special requirements)."""

    SMART_ORDER_ROUTING = "smart_order_routing"
    """Smart order routing."""

    OTHER = "other"
    """Other algorithm type."""


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NCAContact:
    """Contact information for an NCA."""

    jurisdiction: NCAJurisdiction = NCAJurisdiction.FCA
    authority_name: str = ""
    department: str = "Market Supervision"
    email: str = ""
    phone: str = ""
    address: str = ""
    submission_portal: str = ""
    reference_number: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "jurisdiction": self.jurisdiction.value,
            "authority_name": self.authority_name,
            "department": self.department,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "submission_portal": self.submission_portal,
            "reference_number": self.reference_number,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NCAContact":
        """Create from dictionary."""
        return cls(
            jurisdiction=NCAJurisdiction(data.get("jurisdiction", "fca")),
            authority_name=data.get("authority_name", ""),
            department=data.get("department", "Market Supervision"),
            email=data.get("email", ""),
            phone=data.get("phone", ""),
            address=data.get("address", ""),
            submission_portal=data.get("submission_portal", ""),
            reference_number=data.get("reference_number", ""),
        )


@dataclass
class AlgorithmDescription:
    """
    Algorithm description for NCA notification.

    Per MiFID II Article 17(2), firms must provide descriptions
    of algorithmic trading strategies and risk controls.
    """

    algorithm_id: str = ""
    algorithm_name: str = ""
    version: str = ""
    category: AlgorithmCategory = AlgorithmCategory.EXECUTION
    description: str = ""
    strategy_summary: str = ""

    # Operational details
    asset_classes: List[str] = field(default_factory=list)
    trading_venues: List[str] = field(default_factory=list)
    markets: List[str] = field(default_factory=list)

    # Risk controls
    risk_controls_summary: str = ""
    max_order_value_eur: float = 0.0
    max_order_volume: float = 0.0
    max_daily_notional_eur: float = 0.0
    kill_switch_enabled: bool = True

    # Technical details
    responsible_person: str = ""
    responsible_email: str = ""
    deployment_date: Optional[date] = None
    last_material_change: Optional[date] = None

    # HFT specific (if applicable)
    is_hft: bool = False
    hft_strategy_type: str = ""
    messages_per_second: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm_id": self.algorithm_id,
            "algorithm_name": self.algorithm_name,
            "version": self.version,
            "category": self.category.value,
            "description": self.description,
            "strategy_summary": self.strategy_summary,
            "asset_classes": self.asset_classes,
            "trading_venues": self.trading_venues,
            "markets": self.markets,
            "risk_controls_summary": self.risk_controls_summary,
            "max_order_value_eur": self.max_order_value_eur,
            "max_order_volume": self.max_order_volume,
            "max_daily_notional_eur": self.max_daily_notional_eur,
            "kill_switch_enabled": self.kill_switch_enabled,
            "responsible_person": self.responsible_person,
            "responsible_email": self.responsible_email,
            "deployment_date": self.deployment_date.isoformat() if self.deployment_date else None,
            "last_material_change": self.last_material_change.isoformat() if self.last_material_change else None,
            "is_hft": self.is_hft,
            "hft_strategy_type": self.hft_strategy_type,
            "messages_per_second": self.messages_per_second,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmDescription":
        """Create from dictionary."""
        desc = cls(
            algorithm_id=data.get("algorithm_id", ""),
            algorithm_name=data.get("algorithm_name", ""),
            version=data.get("version", ""),
            category=AlgorithmCategory(data.get("category", "execution")),
            description=data.get("description", ""),
            strategy_summary=data.get("strategy_summary", ""),
            asset_classes=data.get("asset_classes", []),
            trading_venues=data.get("trading_venues", []),
            markets=data.get("markets", []),
            risk_controls_summary=data.get("risk_controls_summary", ""),
            max_order_value_eur=data.get("max_order_value_eur", 0.0),
            max_order_volume=data.get("max_order_volume", 0.0),
            max_daily_notional_eur=data.get("max_daily_notional_eur", 0.0),
            kill_switch_enabled=data.get("kill_switch_enabled", True),
            responsible_person=data.get("responsible_person", ""),
            responsible_email=data.get("responsible_email", ""),
            is_hft=data.get("is_hft", False),
            hft_strategy_type=data.get("hft_strategy_type", ""),
            messages_per_second=data.get("messages_per_second", 0),
        )

        if data.get("deployment_date"):
            desc.deployment_date = date.fromisoformat(data["deployment_date"])
        if data.get("last_material_change"):
            desc.last_material_change = date.fromisoformat(data["last_material_change"])

        return desc


@dataclass
class NCANotification:
    """
    MiFID II Article 17(2) NCA Notification.

    Official notification to National Competent Authority about
    algorithmic trading activities.

    Attributes:
        notification_id: Unique identifier
        notification_reference: Human-readable reference
        notification_type: Type of notification
        nca_jurisdiction: Target NCA
        firm_lei: Firm's LEI
        firm_name: Firm's legal name
        algorithms: Algorithm descriptions included
        status: Notification status
        submission_date: Date submitted
        acknowledgment_date: Date acknowledged by NCA
        nca_reference: NCA's reference number
    """

    notification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    notification_reference: str = ""
    notification_type: NotificationType = NotificationType.INITIAL

    # NCA details
    nca_jurisdiction: NCAJurisdiction = NCAJurisdiction.FCA
    nca_contact: Optional[NCAContact] = None

    # Firm identification
    firm_lei: str = ""
    firm_name: str = ""
    firm_address: str = ""
    authorized_person: str = ""
    authorized_email: str = ""
    authorized_phone: str = ""

    # Notification content
    algorithms: List[AlgorithmDescription] = field(default_factory=list)
    summary: str = ""
    change_description: str = ""  # For material change notifications
    effective_date: Optional[date] = None

    # Status
    status: NotificationStatus = NotificationStatus.DRAFT

    # Dates
    prepared_date: Optional[date] = None
    reviewed_date: Optional[date] = None
    approved_date: Optional[date] = None
    submission_date: Optional[date] = None
    acknowledgment_date: Optional[date] = None

    # NCA response
    nca_reference: str = ""
    nca_response: str = ""
    follow_up_required: bool = False
    follow_up_deadline: Optional[date] = None

    # Approvals
    prepared_by: str = ""
    reviewed_by: str = ""
    approved_by: str = ""

    # Metadata
    created_timestamp_ns: int = field(default_factory=time.time_ns)
    modified_timestamp_ns: int = field(default_factory=time.time_ns)

    def __post_init__(self) -> None:
        """Initialize notification reference if not set."""
        if not self.notification_reference:
            today = date.today().strftime("%Y%m%d")
            short_id = self.notification_id[:4].upper()
            self.notification_reference = f"NCA-{self.nca_jurisdiction.value.upper()}-{today}-{short_id}"

    # =========================================================================
    # Status Management
    # =========================================================================

    def prepare(self, prepared_by: str) -> None:
        """Mark notification as prepared."""
        self.prepared_by = prepared_by
        self.prepared_date = date.today()
        self.status = NotificationStatus.PENDING_REVIEW
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Notification {self.notification_reference} prepared by {prepared_by}")

    def review(self, reviewed_by: str) -> None:
        """Mark notification as reviewed."""
        self.reviewed_by = reviewed_by
        self.reviewed_date = date.today()
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Notification {self.notification_reference} reviewed by {reviewed_by}")

    def approve(self, approved_by: str) -> None:
        """Approve notification for submission."""
        self.approved_by = approved_by
        self.approved_date = date.today()
        self.status = NotificationStatus.APPROVED
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Notification {self.notification_reference} approved by {approved_by}")

    def submit(self, nca_reference: str = "") -> None:
        """Mark notification as submitted."""
        self.submission_date = date.today()
        self.nca_reference = nca_reference
        self.status = NotificationStatus.SUBMITTED
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Notification {self.notification_reference} submitted to {self.nca_jurisdiction.value}")

    def acknowledge(self, nca_reference: str = "", response: str = "") -> None:
        """Mark notification as acknowledged by NCA."""
        self.acknowledgment_date = date.today()
        if nca_reference:
            self.nca_reference = nca_reference
        self.nca_response = response
        self.status = NotificationStatus.ACKNOWLEDGED
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Notification {self.notification_reference} acknowledged by NCA")

    def complete(self) -> None:
        """Mark notification process as complete."""
        self.status = NotificationStatus.COMPLETED
        self.modified_timestamp_ns = time.time_ns()
        logger.info(f"Notification {self.notification_reference} completed")

    def reject(self, reason: str) -> None:
        """Mark notification as rejected."""
        self.status = NotificationStatus.REJECTED
        self.nca_response = reason
        self.modified_timestamp_ns = time.time_ns()
        logger.warning(f"Notification {self.notification_reference} rejected: {reason}")

    # =========================================================================
    # Algorithm Management
    # =========================================================================

    def add_algorithm(self, algorithm: AlgorithmDescription) -> None:
        """Add algorithm to notification."""
        self.algorithms.append(algorithm)
        self.modified_timestamp_ns = time.time_ns()

    def remove_algorithm(self, algorithm_id: str) -> bool:
        """Remove algorithm from notification."""
        for i, algo in enumerate(self.algorithms):
            if algo.algorithm_id == algorithm_id:
                self.algorithms.pop(i)
                self.modified_timestamp_ns = time.time_ns()
                return True
        return False

    def get_algorithm(self, algorithm_id: str) -> Optional[AlgorithmDescription]:
        """Get algorithm by ID."""
        for algo in self.algorithms:
            if algo.algorithm_id == algorithm_id:
                return algo
        return None

    # =========================================================================
    # Document Generation
    # =========================================================================

    def generate_notification_document(self) -> str:
        """
        Generate formal NCA notification document.

        Returns text representation of the notification suitable
        for submission to the NCA.
        """
        doc = f"""
================================================================================
                    NCA NOTIFICATION
                    MiFID II Article 17(2)
================================================================================

NOTIFICATION DETAILS
--------------------
Reference:              {self.notification_reference}
Type:                   {self.notification_type.value.upper()}
Status:                 {self.status.value.upper()}
NCA Jurisdiction:       {self.nca_jurisdiction.value.upper()}

--------------------------------------------------------------------------------
                         FIRM IDENTIFICATION
--------------------------------------------------------------------------------
Legal Entity Name:      {self.firm_name}
Legal Entity Identifier: {self.firm_lei}
Address:                {self.firm_address}

Contact Person:         {self.authorized_person}
Email:                  {self.authorized_email}
Phone:                  {self.authorized_phone}

--------------------------------------------------------------------------------
                         NOTIFICATION SUMMARY
--------------------------------------------------------------------------------
{self.summary or 'Notification of algorithmic trading activities per MiFID II Article 17(2).'}

"""

        if self.change_description:
            doc += f"""
CHANGE DESCRIPTION
------------------
{self.change_description}

"""

        doc += f"""
--------------------------------------------------------------------------------
                       ALGORITHM DESCRIPTIONS
--------------------------------------------------------------------------------
Total Algorithms: {len(self.algorithms)}

"""

        for i, algo in enumerate(self.algorithms, 1):
            hft_marker = " [HFT]" if algo.is_hft else ""
            doc += f"""
ALGORITHM {i}: {algo.algorithm_name}{hft_marker}
{'=' * 70}
Algorithm ID:           {algo.algorithm_id}
Version:                {algo.version}
Category:               {algo.category.value}

Description:
{algo.description}

Strategy Summary:
{algo.strategy_summary}

Asset Classes:          {', '.join(algo.asset_classes) or 'N/A'}
Trading Venues:         {', '.join(algo.trading_venues) or 'N/A'}
Markets:                {', '.join(algo.markets) or 'N/A'}

Risk Controls:
{algo.risk_controls_summary}

Limits:
  - Max Order Value:    EUR {algo.max_order_value_eur:,.2f}
  - Max Order Volume:   {algo.max_order_volume:,.0f} units
  - Max Daily Notional: EUR {algo.max_daily_notional_eur:,.2f}
  - Kill Switch:        {'Enabled' if algo.kill_switch_enabled else 'Disabled'}

Responsible Person:     {algo.responsible_person}
Contact Email:          {algo.responsible_email}
Deployment Date:        {algo.deployment_date or 'Pending'}
Last Material Change:   {algo.last_material_change or 'N/A'}

"""
            if algo.is_hft:
                doc += f"""
HFT Details:
  - Strategy Type:      {algo.hft_strategy_type}
  - Msg/Second:         {algo.messages_per_second:,}

"""

        doc += f"""
--------------------------------------------------------------------------------
                         DECLARATION
--------------------------------------------------------------------------------
I hereby certify that:

1. The firm engages in algorithmic trading as defined in Article 4(1)(39)
   of MiFID II.

2. The information provided in this notification is accurate and complete
   to the best of my knowledge.

3. The firm has implemented the organizational requirements, risk controls,
   and systems required by Article 17 of MiFID II and RTS 6.

4. The firm will notify the NCA of any material changes to the information
   provided in this notification.

5. The firm maintains records of all algorithmic trading activities as
   required by MiFIR Article 25.

Effective Date:         {self.effective_date or date.today()}

--------------------------------------------------------------------------------
                         APPROVALS
--------------------------------------------------------------------------------
Prepared By:            {self.prepared_by or '___________________'}
Date:                   {self.prepared_date or '___________________'}

Reviewed By:            {self.reviewed_by or '___________________'}
Date:                   {self.reviewed_date or '___________________'}

Approved By:            {self.approved_by or '___________________'}
Date:                   {self.approved_date or '___________________'}

Signature:              ___________________

"""

        if self.nca_reference:
            doc += f"""
--------------------------------------------------------------------------------
                        NCA ACKNOWLEDGMENT
--------------------------------------------------------------------------------
NCA Reference:          {self.nca_reference}
Submission Date:        {self.submission_date or 'N/A'}
Acknowledgment Date:    {self.acknowledgment_date or 'N/A'}
NCA Response:           {self.nca_response or 'N/A'}

"""

        doc += """
================================================================================
END OF NOTIFICATION
================================================================================
"""
        return doc

    def generate_xml_notification(self) -> str:
        """
        Generate XML format notification for electronic submission.

        Returns XML representation per ESMA technical standards.
        """
        # Build XML structure
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<AlgoTradingNotification xmlns="urn:esma:mifid2:rts6">',
            f'  <NotificationReference>{self.notification_reference}</NotificationReference>',
            f'  <NotificationType>{self.notification_type.value}</NotificationType>',
            f'  <NCAJurisdiction>{self.nca_jurisdiction.value}</NCAJurisdiction>',
            '  <Firm>',
            f'    <LEI>{self.firm_lei}</LEI>',
            f'    <Name>{self.firm_name}</Name>',
            f'    <Address>{self.firm_address}</Address>',
            f'    <ContactPerson>{self.authorized_person}</ContactPerson>',
            f'    <ContactEmail>{self.authorized_email}</ContactEmail>',
            f'    <ContactPhone>{self.authorized_phone}</ContactPhone>',
            '  </Firm>',
            '  <Algorithms>',
        ]

        for algo in self.algorithms:
            xml_lines.extend([
                '    <Algorithm>',
                f'      <AlgorithmID>{algo.algorithm_id}</AlgorithmID>',
                f'      <Name>{algo.algorithm_name}</Name>',
                f'      <Version>{algo.version}</Version>',
                f'      <Category>{algo.category.value}</Category>',
                f'      <Description>{algo.description}</Description>',
                f'      <IsHFT>{str(algo.is_hft).lower()}</IsHFT>',
                f'      <KillSwitchEnabled>{str(algo.kill_switch_enabled).lower()}</KillSwitchEnabled>',
                f'      <MaxOrderValueEUR>{algo.max_order_value_eur}</MaxOrderValueEUR>',
                f'      <MaxOrderVolume>{algo.max_order_volume}</MaxOrderVolume>',
                f'      <ResponsiblePerson>{algo.responsible_person}</ResponsiblePerson>',
                '      <TradingVenues>',
            ])
            for venue in algo.trading_venues:
                xml_lines.append(f'        <Venue>{venue}</Venue>')
            xml_lines.extend([
                '      </TradingVenues>',
                '    </Algorithm>',
            ])

        xml_lines.extend([
            '  </Algorithms>',
            f'  <EffectiveDate>{self.effective_date or date.today()}</EffectiveDate>',
            f'  <PreparedBy>{self.prepared_by}</PreparedBy>',
            f'  <ApprovedBy>{self.approved_by}</ApprovedBy>',
            '</AlgoTradingNotification>',
        ])

        return '\n'.join(xml_lines)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "notification_id": self.notification_id,
            "notification_reference": self.notification_reference,
            "notification_type": self.notification_type.value,
            "nca_jurisdiction": self.nca_jurisdiction.value,
            "nca_contact": self.nca_contact.to_dict() if self.nca_contact else None,
            "firm_lei": self.firm_lei,
            "firm_name": self.firm_name,
            "firm_address": self.firm_address,
            "authorized_person": self.authorized_person,
            "authorized_email": self.authorized_email,
            "authorized_phone": self.authorized_phone,
            "algorithms": [a.to_dict() for a in self.algorithms],
            "summary": self.summary,
            "change_description": self.change_description,
            "effective_date": self.effective_date.isoformat() if self.effective_date else None,
            "status": self.status.value,
            "prepared_date": self.prepared_date.isoformat() if self.prepared_date else None,
            "reviewed_date": self.reviewed_date.isoformat() if self.reviewed_date else None,
            "approved_date": self.approved_date.isoformat() if self.approved_date else None,
            "submission_date": self.submission_date.isoformat() if self.submission_date else None,
            "acknowledgment_date": self.acknowledgment_date.isoformat() if self.acknowledgment_date else None,
            "nca_reference": self.nca_reference,
            "nca_response": self.nca_response,
            "follow_up_required": self.follow_up_required,
            "follow_up_deadline": self.follow_up_deadline.isoformat() if self.follow_up_deadline else None,
            "prepared_by": self.prepared_by,
            "reviewed_by": self.reviewed_by,
            "approved_by": self.approved_by,
            "created_timestamp_ns": self.created_timestamp_ns,
            "modified_timestamp_ns": self.modified_timestamp_ns,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NCANotification":
        """Create from dictionary."""
        notification = cls(
            notification_id=data.get("notification_id", str(uuid.uuid4())),
            notification_reference=data.get("notification_reference", ""),
            notification_type=NotificationType(data.get("notification_type", "initial")),
            nca_jurisdiction=NCAJurisdiction(data.get("nca_jurisdiction", "fca")),
            firm_lei=data.get("firm_lei", ""),
            firm_name=data.get("firm_name", ""),
            firm_address=data.get("firm_address", ""),
            authorized_person=data.get("authorized_person", ""),
            authorized_email=data.get("authorized_email", ""),
            authorized_phone=data.get("authorized_phone", ""),
            summary=data.get("summary", ""),
            change_description=data.get("change_description", ""),
            status=NotificationStatus(data.get("status", "draft")),
            nca_reference=data.get("nca_reference", ""),
            nca_response=data.get("nca_response", ""),
            follow_up_required=data.get("follow_up_required", False),
            prepared_by=data.get("prepared_by", ""),
            reviewed_by=data.get("reviewed_by", ""),
            approved_by=data.get("approved_by", ""),
            created_timestamp_ns=data.get("created_timestamp_ns", time.time_ns()),
            modified_timestamp_ns=data.get("modified_timestamp_ns", time.time_ns()),
        )

        # Parse NCA contact
        if data.get("nca_contact"):
            notification.nca_contact = NCAContact.from_dict(data["nca_contact"])

        # Parse algorithms
        notification.algorithms = [
            AlgorithmDescription.from_dict(a)
            for a in data.get("algorithms", [])
        ]

        # Parse dates
        date_fields = [
            "effective_date", "prepared_date", "reviewed_date", "approved_date",
            "submission_date", "acknowledgment_date", "follow_up_deadline"
        ]
        for field_name in date_fields:
            if data.get(field_name):
                setattr(notification, field_name, date.fromisoformat(data[field_name]))

        return notification


# =============================================================================
# NCA Notification Manager
# =============================================================================


class NCANotificationManager:
    """
    Manager for NCA notifications.

    Handles notification lifecycle including preparation,
    submission tracking, and record keeping.
    """

    def __init__(
        self,
        storage_path: str = "state/compliance/nca_notifications",
        default_jurisdiction: NCAJurisdiction = NCAJurisdiction.FCA,
    ):
        """
        Initialize notification manager.

        Args:
            storage_path: Path for notification storage.
            default_jurisdiction: Default NCA jurisdiction.
        """
        self.storage_path = Path(storage_path)
        self.default_jurisdiction = default_jurisdiction
        self._notifications: Dict[str, NCANotification] = {}
        self._nca_contacts: Dict[NCAJurisdiction, NCAContact] = {}
        self._ensure_storage()
        self._load_default_contacts()

    def _ensure_storage(self) -> None:
        """Ensure storage directory exists."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _load_default_contacts(self) -> None:
        """Load default NCA contact information."""
        self._nca_contacts = {
            NCAJurisdiction.FCA: NCAContact(
                jurisdiction=NCAJurisdiction.FCA,
                authority_name="Financial Conduct Authority",
                department="Market Oversight Division",
                email="algorithm.trading@fca.org.uk",
                submission_portal="https://connect.fca.org.uk",
            ),
            NCAJurisdiction.BAFIN: NCAContact(
                jurisdiction=NCAJurisdiction.BAFIN,
                authority_name="BaFin",
                department="Wertpapieraufsicht",
                email="algo-trading@bafin.de",
            ),
            NCAJurisdiction.AMF: NCAContact(
                jurisdiction=NCAJurisdiction.AMF,
                authority_name="Autorité des marchés financiers",
                department="Direction de la surveillance des marchés",
                email="trading-algorithmique@amf-france.org",
            ),
        }

    # =========================================================================
    # Notification Creation
    # =========================================================================

    def create_notification(
        self,
        notification_type: NotificationType,
        firm_lei: str,
        firm_name: str,
        jurisdiction: Optional[NCAJurisdiction] = None,
        authorized_person: str = "",
    ) -> NCANotification:
        """
        Create a new NCA notification.

        Args:
            notification_type: Type of notification.
            firm_lei: Firm's LEI.
            firm_name: Firm's legal name.
            jurisdiction: Target NCA jurisdiction.
            authorized_person: Authorized contact person.

        Returns:
            New NCANotification instance.
        """
        if jurisdiction is None:
            jurisdiction = self.default_jurisdiction

        notification = NCANotification(
            notification_type=notification_type,
            nca_jurisdiction=jurisdiction,
            nca_contact=self._nca_contacts.get(jurisdiction),
            firm_lei=firm_lei,
            firm_name=firm_name,
            authorized_person=authorized_person,
        )

        self._notifications[notification.notification_id] = notification
        logger.info(f"Created notification {notification.notification_reference}")

        return notification

    def create_initial_notification(
        self,
        firm_lei: str,
        firm_name: str,
        algorithms: List[AlgorithmDescription],
        jurisdiction: Optional[NCAJurisdiction] = None,
    ) -> NCANotification:
        """
        Create initial algorithmic trading notification.

        Args:
            firm_lei: Firm's LEI.
            firm_name: Firm's legal name.
            algorithms: Algorithms to include.
            jurisdiction: Target NCA.

        Returns:
            New notification for initial registration.
        """
        notification = self.create_notification(
            notification_type=NotificationType.INITIAL,
            firm_lei=firm_lei,
            firm_name=firm_name,
            jurisdiction=jurisdiction,
        )

        notification.summary = (
            "Initial notification of algorithmic trading activities per "
            "MiFID II Article 17(2). The firm hereby notifies the competent "
            "authority that it engages in algorithmic trading."
        )

        for algo in algorithms:
            notification.add_algorithm(algo)

        return notification

    def create_new_algorithm_notification(
        self,
        firm_lei: str,
        firm_name: str,
        algorithm: AlgorithmDescription,
        effective_date: Optional[date] = None,
        jurisdiction: Optional[NCAJurisdiction] = None,
    ) -> NCANotification:
        """
        Create notification for new algorithm deployment.

        Args:
            firm_lei: Firm's LEI.
            firm_name: Firm's legal name.
            algorithm: New algorithm description.
            effective_date: Planned deployment date.
            jurisdiction: Target NCA.

        Returns:
            New algorithm notification.
        """
        notification = self.create_notification(
            notification_type=NotificationType.NEW_ALGORITHM,
            firm_lei=firm_lei,
            firm_name=firm_name,
            jurisdiction=jurisdiction,
        )

        notification.add_algorithm(algorithm)
        notification.effective_date = effective_date or date.today()
        notification.summary = (
            f"Notification of new algorithm deployment: {algorithm.algorithm_name} "
            f"(ID: {algorithm.algorithm_id}). The firm intends to deploy this "
            f"algorithm effective {notification.effective_date}."
        )

        return notification

    def create_material_change_notification(
        self,
        firm_lei: str,
        firm_name: str,
        algorithm: AlgorithmDescription,
        change_description: str,
        effective_date: Optional[date] = None,
        jurisdiction: Optional[NCAJurisdiction] = None,
    ) -> NCANotification:
        """
        Create notification for material change to algorithm.

        Args:
            firm_lei: Firm's LEI.
            firm_name: Firm's legal name.
            algorithm: Updated algorithm description.
            change_description: Description of changes.
            effective_date: Effective date of changes.
            jurisdiction: Target NCA.

        Returns:
            Material change notification.
        """
        notification = self.create_notification(
            notification_type=NotificationType.MATERIAL_CHANGE,
            firm_lei=firm_lei,
            firm_name=firm_name,
            jurisdiction=jurisdiction,
        )

        notification.add_algorithm(algorithm)
        notification.change_description = change_description
        notification.effective_date = effective_date or date.today()
        notification.summary = (
            f"Notification of material change to algorithm {algorithm.algorithm_name} "
            f"(ID: {algorithm.algorithm_id}). Changes effective {notification.effective_date}."
        )

        return notification

    # =========================================================================
    # Notification Management
    # =========================================================================

    def get_notification(self, notification_id: str) -> Optional[NCANotification]:
        """Get notification by ID."""
        return self._notifications.get(notification_id)

    def get_notification_by_reference(self, reference: str) -> Optional[NCANotification]:
        """Get notification by reference number."""
        for notif in self._notifications.values():
            if notif.notification_reference == reference:
                return notif
        return None

    def get_notifications_by_status(
        self,
        status: NotificationStatus,
    ) -> List[NCANotification]:
        """Get notifications by status."""
        return [n for n in self._notifications.values() if n.status == status]

    def get_pending_notifications(self) -> List[NCANotification]:
        """Get all pending notifications."""
        pending_statuses = {
            NotificationStatus.DRAFT,
            NotificationStatus.PENDING_REVIEW,
            NotificationStatus.APPROVED,
        }
        return [n for n in self._notifications.values() if n.status in pending_statuses]

    def get_notifications_requiring_followup(self) -> List[NCANotification]:
        """Get notifications requiring follow-up."""
        return [
            n for n in self._notifications.values()
            if n.follow_up_required and n.status != NotificationStatus.COMPLETED
        ]

    def get_all_notifications(self) -> List[NCANotification]:
        """Get all notifications."""
        return sorted(
            self._notifications.values(),
            key=lambda n: n.created_timestamp_ns,
            reverse=True,
        )

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_notification(self, notification: NCANotification) -> None:
        """Save notification to storage."""
        file_path = self.storage_path / f"{notification.notification_id}.json"
        with open(file_path, "w") as f:
            f.write(notification.to_json())
        logger.debug(f"Saved notification {notification.notification_reference}")

    def load_notification(self, notification_id: str) -> Optional[NCANotification]:
        """Load notification from storage."""
        file_path = self.storage_path / f"{notification_id}.json"
        if file_path.exists():
            with open(file_path, "r") as f:
                notif = NCANotification.from_dict(json.load(f))
                self._notifications[notif.notification_id] = notif
                return notif
        return None

    def load_all_notifications(self) -> int:
        """Load all notifications from storage."""
        count = 0
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    notif = NCANotification.from_dict(json.load(f))
                    self._notifications[notif.notification_id] = notif
                    count += 1
            except Exception as e:
                logger.error(f"Failed to load notification {file_path}: {e}")

        logger.info(f"Loaded {count} notifications")
        return count

    def save_all_notifications(self) -> int:
        """Save all notifications to storage."""
        count = 0
        for notif in self._notifications.values():
            try:
                self.save_notification(notif)
                count += 1
            except Exception as e:
                logger.error(f"Failed to save notification {notif.notification_reference}: {e}")

        return count

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_notification_summary(self) -> Dict[str, Any]:
        """Get summary of all notifications."""
        status_counts = {status: 0 for status in NotificationStatus}
        type_counts = {notif_type: 0 for notif_type in NotificationType}
        jurisdiction_counts: Dict[str, int] = {}

        for notif in self._notifications.values():
            status_counts[notif.status] += 1
            type_counts[notif.notification_type] += 1
            juris = notif.nca_jurisdiction.value
            jurisdiction_counts[juris] = jurisdiction_counts.get(juris, 0) + 1

        pending = self.get_pending_notifications()
        followup = self.get_notifications_requiring_followup()

        return {
            "total_notifications": len(self._notifications),
            "status_counts": {k.value: v for k, v in status_counts.items()},
            "type_counts": {k.value: v for k, v in type_counts.items()},
            "jurisdiction_counts": jurisdiction_counts,
            "pending_count": len(pending),
            "followup_required": len(followup),
        }


# =============================================================================
# Factory Functions
# =============================================================================


def create_algorithm_description(
    algorithm_id: str,
    algorithm_name: str,
    version: str,
    category: AlgorithmCategory,
    description: str,
    strategy_summary: str,
    responsible_person: str,
    responsible_email: str,
    trading_venues: Optional[List[str]] = None,
    asset_classes: Optional[List[str]] = None,
    max_order_value_eur: float = 1_000_000.0,
    is_hft: bool = False,
) -> AlgorithmDescription:
    """
    Create an algorithm description for NCA notification.

    Args:
        algorithm_id: Algorithm identifier.
        algorithm_name: Human-readable name.
        version: Algorithm version.
        category: Algorithm category.
        description: Full description.
        strategy_summary: Brief strategy summary.
        responsible_person: Responsible person name.
        responsible_email: Contact email.
        trading_venues: List of venue MICs.
        asset_classes: List of asset classes.
        max_order_value_eur: Maximum order value limit.
        is_hft: Whether algorithm is HFT.

    Returns:
        AlgorithmDescription instance.
    """
    return AlgorithmDescription(
        algorithm_id=algorithm_id,
        algorithm_name=algorithm_name,
        version=version,
        category=category,
        description=description,
        strategy_summary=strategy_summary,
        responsible_person=responsible_person,
        responsible_email=responsible_email,
        trading_venues=trading_venues or [],
        asset_classes=asset_classes or [],
        max_order_value_eur=max_order_value_eur,
        is_hft=is_hft,
        kill_switch_enabled=True,
    )


def create_nca_notification_manager(
    storage_path: str = "state/compliance/nca_notifications",
    default_jurisdiction: NCAJurisdiction = NCAJurisdiction.FCA,
) -> NCANotificationManager:
    """
    Create an NCA notification manager.

    Args:
        storage_path: Path for notification storage.
        default_jurisdiction: Default NCA jurisdiction.

    Returns:
        NCANotificationManager instance.
    """
    return NCANotificationManager(
        storage_path=storage_path,
        default_jurisdiction=default_jurisdiction,
    )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Enums
    "NCAJurisdiction",
    "NotificationType",
    "NotificationStatus",
    "AlgorithmCategory",
    # Data classes
    "NCAContact",
    "AlgorithmDescription",
    "NCANotification",
    # Manager
    "NCANotificationManager",
    # Factory functions
    "create_algorithm_description",
    "create_nca_notification_manager",
]
