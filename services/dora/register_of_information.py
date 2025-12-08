# -*- coding: utf-8 -*-
"""
DORA Register of Information Module (Article 28(3)).

Regulation (EU) 2022/2554 Article 28(3) requires financial entities to
maintain and update a register of information relating to all contractual
arrangements on the use of ICT services provided by ICT third-party providers.

Key Requirements:
    - Maintain register at entity, sub-consolidated and consolidated levels
    - Update on material changes
    - Make available to NCA upon request
    - Submit to NCA annually (by 30 April per ESA Decision)
    - Follow ITS templates (CIR 2024/2956)

ITS Template Structure (DPM 4.0):
    - B_01.01: Entity maintaining register
    - B_01.02: Branch information
    - B_02.01: Contractual arrangement level
    - B_02.02: Contractual arrangement functions
    - B_03.01: ICT third-party service provider (direct)
    - B_04.01: Subcontracting chain
    - B_05.01: Entity making use of ICT services (group)
    - B_06.01: ICT services supporting functions
    - B_99.01: Totals

References:
    - Article 28(3) DORA: https://www.digital-operational-resilience-act.com/Article_28.html
    - ITS on Register of Information: CIR 2024/2956
    - ESA Guidelines on ROI: JC 2023/86
    - Reference Date: 31 March 2025
    - Submission Deadline: 30 April 2025 (first submission)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations per ITS
# =============================================================================

class ContractType(Enum):
    """Contract type classification per ITS."""
    OUTSOURCING = "outsourcing"
    PROCUREMENT = "procurement"
    INTRA_GROUP = "intra_group"
    MIXED = "mixed"


class ServiceType(Enum):
    """ICT service type per ITS."""
    CLOUD_COMPUTING = "cloud_computing"
    DATA_CENTERS = "data_centers"
    SOFTWARE = "software"
    HARDWARE = "hardware"
    NETWORK = "network"
    SECURITY = "security"
    DATA_ANALYTICS = "data_analytics"
    MARKET_DATA = "market_data"
    TRADING = "trading"
    PAYMENT_PROCESSING = "payment_processing"
    OTHER = "other"


class FunctionType(Enum):
    """Business function type per ITS."""
    CRITICAL = "critical"  # Critical or Important function
    IMPORTANT = "important"
    STANDARD = "standard"


class DataLocation(Enum):
    """Data location classification."""
    EU = "eu"
    EEA = "eea"
    ADEQUACY_DECISION = "adequacy_decision"
    THIRD_COUNTRY = "third_country"


class ProviderLocationType(Enum):
    """Provider location classification."""
    EU_MEMBER_STATE = "eu_member_state"
    EEA_COUNTRY = "eea_country"
    THIRD_COUNTRY = "third_country"


class SubcontractingLevel(Enum):
    """Level in subcontracting chain."""
    DIRECT = "direct"
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3_PLUS = "level_3_plus"


class RegisterStatus(Enum):
    """Register entry status."""
    ACTIVE = "active"
    PENDING = "pending"
    TERMINATED = "terminated"
    UNDER_REVIEW = "under_review"


class ExportFormat(Enum):
    """Export format for ITS submission."""
    CSV = "csv"
    XML = "xml"
    JSON = "json"


# =============================================================================
# Data Structures per ITS Templates
# =============================================================================

@dataclass
class EntityInformation:
    """
    B_01.01 - Entity maintaining the register.

    Identification of the financial entity maintaining the register.
    """
    entity_id: str = ""
    lei: str = ""  # Legal Entity Identifier (mandatory if available)
    entity_name: str = ""
    entity_type: str = ""  # Per Article 2(1) DORA
    country_of_incorporation: str = ""  # ISO 3166-1 alpha-2
    competent_authority: str = ""  # NCA
    parent_lei: str = ""  # For group structures
    consolidation_level: str = ""  # entity, sub_consolidated, consolidated

    def __post_init__(self):
        if not self.entity_id:
            self.entity_id = f"ENT-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class BranchInformation:
    """
    B_01.02 - Branch information.

    Information on branches of the financial entity.
    """
    branch_id: str = ""
    entity_lei: str = ""
    branch_country: str = ""  # ISO 3166-1 alpha-2
    branch_name: str = ""
    branch_address: str = ""
    local_competent_authority: str = ""

    def __post_init__(self):
        if not self.branch_id:
            self.branch_id = f"BRN-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ContractualArrangement:
    """
    B_02.01 - Contractual arrangement level.

    Core contractual arrangement information per ITS template.
    """
    arrangement_id: str = ""
    arrangement_reference: str = ""  # Internal reference number

    # Counterparty identification
    provider_lei: str = ""
    provider_name: str = ""
    provider_country: str = ""
    provider_type_location: ProviderLocationType = ProviderLocationType.EU_MEMBER_STATE

    # Contract details
    contract_type: ContractType = ContractType.PROCUREMENT
    contract_start_date: Optional[str] = None  # ISO format
    contract_end_date: Optional[str] = None  # ISO format (None = indefinite)
    contract_value_eur: float = 0.0
    notice_period_days: int = 30
    renewal_type: str = ""  # automatic, manual, fixed_term

    # Criticality
    supports_critical_function: bool = False
    critical_function_ids: List[str] = field(default_factory=list)

    # Data handling
    data_processing_countries: List[str] = field(default_factory=list)
    data_storage_countries: List[str] = field(default_factory=list)
    personal_data_processed: bool = False
    sensitive_data_processed: bool = False

    # Subcontracting
    subcontracting_permitted: bool = False
    subcontracting_approved: bool = False
    subcontractor_chain_known: bool = True

    # Audit and termination
    audit_rights: bool = False
    nca_access_rights: bool = False
    exit_plan_exists: bool = False

    # Status and dates
    status: RegisterStatus = RegisterStatus.ACTIVE
    entry_date: str = ""
    last_update_date: str = ""

    def __post_init__(self):
        if not self.arrangement_id:
            self.arrangement_id = f"CA-{uuid.uuid4().hex[:8].upper()}"
        if not self.entry_date:
            self.entry_date = datetime.now(timezone.utc).isoformat()
        if not self.last_update_date:
            self.last_update_date = self.entry_date


@dataclass
class ContractualArrangementFunction:
    """
    B_02.02 - Contractual arrangement functions.

    Links contractual arrangements to business functions.
    """
    link_id: str = ""
    arrangement_id: str = ""
    function_id: str = ""
    function_name: str = ""
    function_type: FunctionType = FunctionType.STANDARD
    is_critical_or_important: bool = False
    date_of_last_assessment: Optional[str] = None

    def __post_init__(self):
        if not self.link_id:
            self.link_id = f"CAF-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ICTServiceProvider:
    """
    B_03.01 - ICT third-party service provider (direct).

    Detailed information about direct ICT service providers.
    """
    provider_id: str = ""
    lei: str = ""  # Legal Entity Identifier
    alternative_id: str = ""  # If no LEI (e.g., non-EU entity)
    alternative_id_type: str = ""  # Type of alternative ID

    # Names
    legal_name: str = ""
    trading_name: str = ""

    # Location
    headquarters_country: str = ""  # ISO 3166-1 alpha-2
    headquarters_address: str = ""
    location_type: ProviderLocationType = ProviderLocationType.EU_MEMBER_STATE

    # Classification
    is_ict_service_provider: bool = True
    is_ict_intra_group_provider: bool = False
    provider_category: str = ""

    # Parent company (if applicable)
    parent_lei: str = ""
    parent_name: str = ""
    ultimate_parent_lei: str = ""
    ultimate_parent_name: str = ""
    ultimate_parent_country: str = ""

    # CTPP status
    is_designated_ctpp: bool = False
    ctpp_lead_overseer: str = ""

    # Contact
    primary_contact_name: str = ""
    primary_contact_email: str = ""

    # Metadata
    entry_date: str = ""
    last_update_date: str = ""

    def __post_init__(self):
        if not self.provider_id:
            self.provider_id = f"PRV-{uuid.uuid4().hex[:8].upper()}"
        if not self.entry_date:
            self.entry_date = datetime.now(timezone.utc).isoformat()


@dataclass
class SubcontractorEntry:
    """
    B_04.01 - Subcontracting chain.

    Information on subcontractors in the chain.
    """
    subcontractor_id: str = ""
    arrangement_id: str = ""  # Parent arrangement
    direct_provider_id: str = ""  # Direct provider

    # Subcontractor identification
    lei: str = ""
    alternative_id: str = ""
    legal_name: str = ""
    country: str = ""

    # Chain position
    subcontracting_level: SubcontractingLevel = SubcontractingLevel.LEVEL_1
    chain_rank: int = 1  # Position in chain

    # Services
    services_subcontracted: List[str] = field(default_factory=list)
    supports_critical_function: bool = False

    # Data handling
    data_processing_countries: List[str] = field(default_factory=list)
    personal_data_access: bool = False

    # Notification
    entity_notified: bool = True
    notification_date: Optional[str] = None

    def __post_init__(self):
        if not self.subcontractor_id:
            self.subcontractor_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class ICTServiceRecord:
    """
    B_06.01 - ICT services supporting functions.

    ICT services and their relationship to business functions.
    """
    service_id: str = ""
    arrangement_id: str = ""
    provider_id: str = ""

    # Service details
    service_name: str = ""
    service_type: ServiceType = ServiceType.OTHER
    service_description: str = ""

    # Function support
    supported_functions: List[str] = field(default_factory=list)
    supports_critical_function: bool = False

    # Service levels
    availability_target_pct: float = 99.0
    rpo_hours: int = 24  # Recovery Point Objective
    rto_hours: int = 4  # Recovery Time Objective

    # Data classification
    data_classification: str = ""  # public, internal, confidential, restricted
    personal_data_involved: bool = False

    def __post_init__(self):
        if not self.service_id:
            self.service_id = f"SVC-{uuid.uuid4().hex[:8].upper()}"


@dataclass
class RegisterSubmission:
    """
    Record of register submission to NCA.
    """
    submission_id: str = ""
    submission_date: str = ""
    reference_date: str = ""  # Data as of date
    competent_authority: str = ""
    submission_format: ExportFormat = ExportFormat.CSV

    # Submission details
    entity_lei: str = ""
    entity_name: str = ""
    consolidation_level: str = ""

    # Contents
    arrangements_count: int = 0
    providers_count: int = 0
    functions_count: int = 0

    # Status
    submission_status: str = ""  # submitted, acknowledged, accepted, rejected
    acknowledgment_reference: str = ""
    feedback_received: bool = False
    feedback_details: str = ""

    def __post_init__(self):
        if not self.submission_id:
            self.submission_id = f"REG-{uuid.uuid4().hex[:8].upper()}"
        if not self.submission_date:
            self.submission_date = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RegisterOfInformationConfig:
    """Configuration for Register of Information."""

    # Entity information
    entity_lei: str = ""
    entity_name: str = ""
    entity_type: str = "investment_firm"  # Per Article 2(1)
    country: str = ""
    competent_authority: str = ""

    # Submission settings
    annual_submission_deadline_month: int = 4  # April
    annual_submission_deadline_day: int = 30
    reference_date_month: int = 3  # March
    reference_date_day: int = 31

    # Thresholds
    materiality_threshold_eur: float = 10000.0
    auto_update_on_material_change: bool = True

    # Validation
    require_lei_for_eu_providers: bool = True
    require_exit_plan_for_critical: bool = True
    validate_data_locations: bool = True

    # Export settings
    default_export_format: ExportFormat = ExportFormat.CSV
    include_inactive_in_export: bool = False

    # Storage
    storage_path: str = "state/dora/register_of_information"
    log_path: str = "logs/dora/register_of_information"

    # Retention
    historical_retention_years: int = 5


# =============================================================================
# Main Implementation
# =============================================================================

class DORARegisterOfInformation:
    """
    DORA Article 28(3) compliant Register of Information.

    Maintains comprehensive register of all ICT third-party arrangements
    with full ITS template compliance for regulatory submission.

    Key Features:
    - Full ITS template structure (B_01 to B_99)
    - Multi-level consolidation support
    - Automatic validation
    - CSV/XML export for NCA submission
    - Submission tracking
    - Material change monitoring

    Usage:
        config = RegisterOfInformationConfig(
            entity_lei="YOUR_LEI",
            entity_name="Your Company",
        )
        register = DORARegisterOfInformation(config)

        # Register entity information
        register.set_entity_information(
            lei="YOUR_LEI",
            entity_name="Your Company",
            entity_type="investment_firm",
            country="DE",
        )

        # Add contractual arrangement
        arrangement = register.add_contractual_arrangement(
            provider_name="Cloud Provider",
            provider_country="DE",
            contract_type=ContractType.PROCUREMENT,
        )

        # Generate annual submission
        submission = register.generate_annual_submission(
            reference_date="2025-03-31",
        )

        # Export to ITS format
        csv_data = register.export_to_csv()
    """

    def __init__(self, config: Optional[RegisterOfInformationConfig] = None):
        """Initialize Register of Information."""
        self.config = config or RegisterOfInformationConfig()

        # Entity information
        self._entity: Optional[EntityInformation] = None
        self._branches: Dict[str, BranchInformation] = {}

        # Core registers
        self._arrangements: Dict[str, ContractualArrangement] = {}
        self._arrangement_functions: Dict[str, ContractualArrangementFunction] = {}
        self._providers: Dict[str, ICTServiceProvider] = {}
        self._subcontractors: Dict[str, SubcontractorEntry] = {}
        self._services: Dict[str, ICTServiceRecord] = {}

        # Submission history
        self._submissions: Dict[str, RegisterSubmission] = {}

        # Indexes
        self._arrangements_by_provider: Dict[str, Set[str]] = {}
        self._functions_by_arrangement: Dict[str, Set[str]] = {}
        self._services_by_arrangement: Dict[str, Set[str]] = {}
        self._subcontractors_by_arrangement: Dict[str, Set[str]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Storage paths
        self._storage_path = Path(self.config.storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Initialize entity if config provided
        if self.config.entity_lei:
            self.set_entity_information(
                lei=self.config.entity_lei,
                entity_name=self.config.entity_name,
                entity_type=self.config.entity_type,
                country=self.config.country,
                competent_authority=self.config.competent_authority,
            )

        logger.info("DORARegisterOfInformation initialized")

    # =========================================================================
    # Entity Information (B_01.01)
    # =========================================================================

    def set_entity_information(
        self,
        lei: str,
        entity_name: str,
        entity_type: str,
        country: str,
        competent_authority: str = "",
        parent_lei: str = "",
        consolidation_level: str = "entity",
    ) -> EntityInformation:
        """
        Set the entity maintaining the register.

        Args:
            lei: Legal Entity Identifier
            entity_name: Entity legal name
            entity_type: Entity type per Article 2(1)
            country: Country of incorporation (ISO 3166-1 alpha-2)
            competent_authority: Competent NCA
            parent_lei: Parent LEI for group structures
            consolidation_level: entity, sub_consolidated, or consolidated

        Returns:
            EntityInformation
        """
        entity = EntityInformation(
            lei=lei,
            entity_name=entity_name,
            entity_type=entity_type,
            country_of_incorporation=country.upper(),
            competent_authority=competent_authority,
            parent_lei=parent_lei,
            consolidation_level=consolidation_level,
        )

        with self._lock:
            self._entity = entity

        self._log_event("entity_set", {
            "lei": lei,
            "entity_name": entity_name,
        })

        return entity

    def get_entity_information(self) -> Optional[EntityInformation]:
        """Get entity information."""
        with self._lock:
            return self._entity

    def add_branch(
        self,
        branch_country: str,
        branch_name: str,
        branch_address: str = "",
        local_competent_authority: str = "",
    ) -> BranchInformation:
        """Add branch information."""
        with self._lock:
            if not self._entity:
                raise ValueError("Entity information must be set first")

            branch = BranchInformation(
                entity_lei=self._entity.lei,
                branch_country=branch_country.upper(),
                branch_name=branch_name,
                branch_address=branch_address,
                local_competent_authority=local_competent_authority,
            )

            self._branches[branch.branch_id] = branch

        return branch

    # =========================================================================
    # Contractual Arrangements (B_02.01)
    # =========================================================================

    def add_contractual_arrangement(
        self,
        provider_name: str,
        provider_country: str,
        contract_type: ContractType,
        arrangement_reference: str = "",
        provider_lei: str = "",
        contract_start_date: Optional[str] = None,
        contract_end_date: Optional[str] = None,
        contract_value_eur: float = 0.0,
        notice_period_days: int = 30,
        supports_critical_function: bool = False,
        critical_function_ids: Optional[List[str]] = None,
        data_processing_countries: Optional[List[str]] = None,
        data_storage_countries: Optional[List[str]] = None,
        personal_data_processed: bool = False,
        subcontracting_permitted: bool = False,
        audit_rights: bool = False,
        exit_plan_exists: bool = False,
    ) -> ContractualArrangement:
        """
        Add a contractual arrangement to the register.

        Args:
            provider_name: Provider legal name
            provider_country: Provider country (ISO 3166-1 alpha-2)
            contract_type: Type of contract
            arrangement_reference: Internal reference
            provider_lei: Provider LEI (if available)
            contract_start_date: Contract start date (ISO format)
            contract_end_date: Contract end date (None = indefinite)
            contract_value_eur: Annual contract value in EUR
            notice_period_days: Termination notice period
            supports_critical_function: Whether supports critical function
            critical_function_ids: IDs of critical functions supported
            data_processing_countries: Countries where data is processed
            data_storage_countries: Countries where data is stored
            personal_data_processed: Whether personal data is processed
            subcontracting_permitted: Whether subcontracting is permitted
            audit_rights: Whether audit rights are granted
            exit_plan_exists: Whether exit plan exists

        Returns:
            ContractualArrangement
        """
        # Validate
        if self.config.require_lei_for_eu_providers:
            if self._is_eu_country(provider_country) and not provider_lei:
                logger.warning(
                    f"LEI recommended for EU provider: {provider_name}"
                )

        if supports_critical_function and self.config.require_exit_plan_for_critical:
            if not exit_plan_exists:
                logger.warning(
                    f"Exit plan required for critical function provider: {provider_name}"
                )

        # Determine provider location type
        if self._is_eu_country(provider_country):
            location_type = ProviderLocationType.EU_MEMBER_STATE
        elif provider_country in ["IS", "LI", "NO"]:  # EEA
            location_type = ProviderLocationType.EEA_COUNTRY
        else:
            location_type = ProviderLocationType.THIRD_COUNTRY

        arrangement = ContractualArrangement(
            arrangement_reference=arrangement_reference or f"ARR-{datetime.now().strftime('%Y%m%d')}",
            provider_lei=provider_lei,
            provider_name=provider_name,
            provider_country=provider_country.upper(),
            provider_type_location=location_type,
            contract_type=contract_type,
            contract_start_date=contract_start_date,
            contract_end_date=contract_end_date,
            contract_value_eur=contract_value_eur,
            notice_period_days=notice_period_days,
            supports_critical_function=supports_critical_function,
            critical_function_ids=critical_function_ids or [],
            data_processing_countries=[c.upper() for c in (data_processing_countries or [])],
            data_storage_countries=[c.upper() for c in (data_storage_countries or [])],
            personal_data_processed=personal_data_processed,
            subcontracting_permitted=subcontracting_permitted,
            audit_rights=audit_rights,
            exit_plan_exists=exit_plan_exists,
        )

        with self._lock:
            self._arrangements[arrangement.arrangement_id] = arrangement

            # Update provider index
            provider_key = provider_lei or provider_name
            if provider_key not in self._arrangements_by_provider:
                self._arrangements_by_provider[provider_key] = set()
            self._arrangements_by_provider[provider_key].add(arrangement.arrangement_id)

            # Initialize other indexes
            self._functions_by_arrangement[arrangement.arrangement_id] = set()
            self._services_by_arrangement[arrangement.arrangement_id] = set()
            self._subcontractors_by_arrangement[arrangement.arrangement_id] = set()

        self._log_event("arrangement_added", {
            "arrangement_id": arrangement.arrangement_id,
            "provider_name": provider_name,
            "contract_type": contract_type.value,
            "supports_critical_function": supports_critical_function,
        })

        return arrangement

    def update_arrangement(
        self,
        arrangement_id: str,
        **updates: Any,
    ) -> Optional[ContractualArrangement]:
        """Update a contractual arrangement."""
        with self._lock:
            if arrangement_id not in self._arrangements:
                return None

            arrangement = self._arrangements[arrangement_id]

            for key, value in updates.items():
                if hasattr(arrangement, key):
                    setattr(arrangement, key, value)

            arrangement.last_update_date = datetime.now(timezone.utc).isoformat()

        self._log_event("arrangement_updated", {
            "arrangement_id": arrangement_id,
            "updates": list(updates.keys()),
        })

        return arrangement

    def terminate_arrangement(
        self,
        arrangement_id: str,
        termination_date: str,
        termination_reason: str = "",
    ) -> Optional[ContractualArrangement]:
        """Terminate a contractual arrangement."""
        with self._lock:
            if arrangement_id not in self._arrangements:
                return None

            arrangement = self._arrangements[arrangement_id]
            arrangement.status = RegisterStatus.TERMINATED
            arrangement.contract_end_date = termination_date
            arrangement.last_update_date = datetime.now(timezone.utc).isoformat()

        self._log_event("arrangement_terminated", {
            "arrangement_id": arrangement_id,
            "termination_date": termination_date,
            "reason": termination_reason,
        })

        return arrangement

    def get_arrangement(
        self,
        arrangement_id: str,
    ) -> Optional[ContractualArrangement]:
        """Get arrangement by ID."""
        with self._lock:
            return self._arrangements.get(arrangement_id)

    def get_all_arrangements(
        self,
        include_terminated: bool = False,
    ) -> List[ContractualArrangement]:
        """Get all arrangements."""
        with self._lock:
            arrangements = list(self._arrangements.values())
            if not include_terminated:
                arrangements = [
                    a for a in arrangements
                    if a.status != RegisterStatus.TERMINATED
                ]
            return arrangements

    def get_arrangements_for_critical_functions(self) -> List[ContractualArrangement]:
        """Get arrangements supporting critical functions."""
        with self._lock:
            return [
                a for a in self._arrangements.values()
                if a.supports_critical_function and a.status == RegisterStatus.ACTIVE
            ]

    def get_arrangements_by_provider(
        self,
        provider_identifier: str,
    ) -> List[ContractualArrangement]:
        """Get arrangements for a specific provider."""
        with self._lock:
            arrangement_ids = self._arrangements_by_provider.get(provider_identifier, set())
            return [
                self._arrangements[aid]
                for aid in arrangement_ids
                if aid in self._arrangements
            ]

    # =========================================================================
    # Function Links (B_02.02)
    # =========================================================================

    def link_function_to_arrangement(
        self,
        arrangement_id: str,
        function_id: str,
        function_name: str,
        function_type: FunctionType,
        is_critical_or_important: bool = False,
    ) -> Optional[ContractualArrangementFunction]:
        """Link a business function to an arrangement."""
        with self._lock:
            if arrangement_id not in self._arrangements:
                return None

            link = ContractualArrangementFunction(
                arrangement_id=arrangement_id,
                function_id=function_id,
                function_name=function_name,
                function_type=function_type,
                is_critical_or_important=is_critical_or_important,
                date_of_last_assessment=datetime.now(timezone.utc).isoformat(),
            )

            self._arrangement_functions[link.link_id] = link
            self._functions_by_arrangement[arrangement_id].add(link.link_id)

            # Update arrangement criticality if needed
            if is_critical_or_important:
                arrangement = self._arrangements[arrangement_id]
                arrangement.supports_critical_function = True
                if function_id not in arrangement.critical_function_ids:
                    arrangement.critical_function_ids.append(function_id)

        return link

    def get_functions_for_arrangement(
        self,
        arrangement_id: str,
    ) -> List[ContractualArrangementFunction]:
        """Get all functions linked to an arrangement."""
        with self._lock:
            link_ids = self._functions_by_arrangement.get(arrangement_id, set())
            return [
                self._arrangement_functions[lid]
                for lid in link_ids
                if lid in self._arrangement_functions
            ]

    # =========================================================================
    # Provider Information (B_03.01)
    # =========================================================================

    def add_provider(
        self,
        legal_name: str,
        headquarters_country: str,
        lei: str = "",
        alternative_id: str = "",
        alternative_id_type: str = "",
        trading_name: str = "",
        headquarters_address: str = "",
        is_intra_group: bool = False,
        parent_lei: str = "",
        parent_name: str = "",
        is_designated_ctpp: bool = False,
        ctpp_lead_overseer: str = "",
        primary_contact_email: str = "",
    ) -> ICTServiceProvider:
        """
        Add ICT service provider information.

        Args:
            legal_name: Legal entity name
            headquarters_country: HQ country (ISO 3166-1 alpha-2)
            lei: Legal Entity Identifier
            alternative_id: Alternative identifier if no LEI
            alternative_id_type: Type of alternative ID
            trading_name: Trading name
            headquarters_address: HQ address
            is_intra_group: Whether intra-group provider
            parent_lei: Parent company LEI
            parent_name: Parent company name
            is_designated_ctpp: Whether designated as CTPP
            ctpp_lead_overseer: CTPP lead overseer (if designated)
            primary_contact_email: Primary contact

        Returns:
            ICTServiceProvider
        """
        # Determine location type
        if self._is_eu_country(headquarters_country):
            location_type = ProviderLocationType.EU_MEMBER_STATE
        elif headquarters_country in ["IS", "LI", "NO"]:
            location_type = ProviderLocationType.EEA_COUNTRY
        else:
            location_type = ProviderLocationType.THIRD_COUNTRY

        provider = ICTServiceProvider(
            lei=lei,
            alternative_id=alternative_id,
            alternative_id_type=alternative_id_type,
            legal_name=legal_name,
            trading_name=trading_name or legal_name,
            headquarters_country=headquarters_country.upper(),
            headquarters_address=headquarters_address,
            location_type=location_type,
            is_ict_intra_group_provider=is_intra_group,
            parent_lei=parent_lei,
            parent_name=parent_name,
            is_designated_ctpp=is_designated_ctpp,
            ctpp_lead_overseer=ctpp_lead_overseer,
            primary_contact_email=primary_contact_email,
        )

        with self._lock:
            self._providers[provider.provider_id] = provider

        self._log_event("provider_added", {
            "provider_id": provider.provider_id,
            "legal_name": legal_name,
            "is_ctpp": is_designated_ctpp,
        })

        return provider

    def get_provider(
        self,
        provider_id: str,
    ) -> Optional[ICTServiceProvider]:
        """Get provider by ID."""
        with self._lock:
            return self._providers.get(provider_id)

    def get_provider_by_lei(
        self,
        lei: str,
    ) -> Optional[ICTServiceProvider]:
        """Get provider by LEI."""
        with self._lock:
            for provider in self._providers.values():
                if provider.lei == lei:
                    return provider
            return None

    def get_all_providers(self) -> List[ICTServiceProvider]:
        """Get all providers."""
        with self._lock:
            return list(self._providers.values())

    def get_ctpp_providers(self) -> List[ICTServiceProvider]:
        """Get all designated CTPPs."""
        with self._lock:
            return [p for p in self._providers.values() if p.is_designated_ctpp]

    # =========================================================================
    # Subcontracting Chain (B_04.01)
    # =========================================================================

    def add_subcontractor(
        self,
        arrangement_id: str,
        direct_provider_id: str,
        legal_name: str,
        country: str,
        lei: str = "",
        subcontracting_level: SubcontractingLevel = SubcontractingLevel.LEVEL_1,
        services_subcontracted: Optional[List[str]] = None,
        supports_critical_function: bool = False,
        data_processing_countries: Optional[List[str]] = None,
        personal_data_access: bool = False,
    ) -> Optional[SubcontractorEntry]:
        """Add subcontractor to the chain."""
        with self._lock:
            if arrangement_id not in self._arrangements:
                return None

            subcontractor = SubcontractorEntry(
                arrangement_id=arrangement_id,
                direct_provider_id=direct_provider_id,
                lei=lei,
                legal_name=legal_name,
                country=country.upper(),
                subcontracting_level=subcontracting_level,
                services_subcontracted=services_subcontracted or [],
                supports_critical_function=supports_critical_function,
                data_processing_countries=[
                    c.upper() for c in (data_processing_countries or [])
                ],
                personal_data_access=personal_data_access,
                notification_date=datetime.now(timezone.utc).isoformat(),
            )

            self._subcontractors[subcontractor.subcontractor_id] = subcontractor
            self._subcontractors_by_arrangement[arrangement_id].add(
                subcontractor.subcontractor_id
            )

        return subcontractor

    def get_subcontractors_for_arrangement(
        self,
        arrangement_id: str,
    ) -> List[SubcontractorEntry]:
        """Get all subcontractors for an arrangement."""
        with self._lock:
            sub_ids = self._subcontractors_by_arrangement.get(arrangement_id, set())
            return [
                self._subcontractors[sid]
                for sid in sub_ids
                if sid in self._subcontractors
            ]

    def get_full_subcontracting_chain(
        self,
        arrangement_id: str,
    ) -> Dict[str, Any]:
        """Get full subcontracting chain visualization."""
        with self._lock:
            if arrangement_id not in self._arrangements:
                return {}

            arrangement = self._arrangements[arrangement_id]
            subcontractors = self.get_subcontractors_for_arrangement(arrangement_id)

            return {
                "arrangement_id": arrangement_id,
                "direct_provider": arrangement.provider_name,
                "chain_known": arrangement.subcontractor_chain_known,
                "subcontracting_permitted": arrangement.subcontracting_permitted,
                "chain": [
                    {
                        "level": s.subcontracting_level.value,
                        "name": s.legal_name,
                        "country": s.country,
                        "services": s.services_subcontracted,
                        "critical": s.supports_critical_function,
                    }
                    for s in sorted(subcontractors, key=lambda x: x.chain_rank)
                ],
            }

    # =========================================================================
    # ICT Services (B_06.01)
    # =========================================================================

    def add_service(
        self,
        arrangement_id: str,
        provider_id: str,
        service_name: str,
        service_type: ServiceType,
        service_description: str = "",
        supported_functions: Optional[List[str]] = None,
        supports_critical_function: bool = False,
        availability_target_pct: float = 99.0,
        rpo_hours: int = 24,
        rto_hours: int = 4,
        data_classification: str = "",
        personal_data_involved: bool = False,
    ) -> Optional[ICTServiceRecord]:
        """Add ICT service record."""
        with self._lock:
            if arrangement_id not in self._arrangements:
                return None

            service = ICTServiceRecord(
                arrangement_id=arrangement_id,
                provider_id=provider_id,
                service_name=service_name,
                service_type=service_type,
                service_description=service_description,
                supported_functions=supported_functions or [],
                supports_critical_function=supports_critical_function,
                availability_target_pct=availability_target_pct,
                rpo_hours=rpo_hours,
                rto_hours=rto_hours,
                data_classification=data_classification,
                personal_data_involved=personal_data_involved,
            )

            self._services[service.service_id] = service
            self._services_by_arrangement[arrangement_id].add(service.service_id)

        return service

    def get_services_for_arrangement(
        self,
        arrangement_id: str,
    ) -> List[ICTServiceRecord]:
        """Get all services for an arrangement."""
        with self._lock:
            service_ids = self._services_by_arrangement.get(arrangement_id, set())
            return [
                self._services[sid]
                for sid in service_ids
                if sid in self._services
            ]

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_register(self) -> Dict[str, Any]:
        """
        Validate register completeness and compliance.

        Returns comprehensive validation report.
        """
        with self._lock:
            issues = []
            warnings = []

            # Check entity information
            if not self._entity:
                issues.append("Entity information not set (B_01.01)")
            elif not self._entity.lei:
                warnings.append("Entity LEI not provided")

            # Check arrangements
            for arr_id, arr in self._arrangements.items():
                if arr.status != RegisterStatus.ACTIVE:
                    continue

                # LEI requirement for EU providers
                if self._is_eu_country(arr.provider_country) and not arr.provider_lei:
                    if self.config.require_lei_for_eu_providers:
                        warnings.append(
                            f"LEI missing for EU provider in arrangement {arr_id}"
                        )

                # Exit plan for critical
                if arr.supports_critical_function and not arr.exit_plan_exists:
                    if self.config.require_exit_plan_for_critical:
                        issues.append(
                            f"Exit plan missing for critical arrangement {arr_id}"
                        )

                # Data location
                if self.config.validate_data_locations:
                    if not arr.data_processing_countries:
                        warnings.append(
                            f"Data processing locations not specified for {arr_id}"
                        )

                # Subcontracting chain
                if arr.subcontracting_permitted and not arr.subcontractor_chain_known:
                    warnings.append(
                        f"Subcontracting chain not fully known for {arr_id}"
                    )

            # Summary
            return {
                "validation_date": datetime.now(timezone.utc).isoformat(),
                "is_valid": len(issues) == 0,
                "issues_count": len(issues),
                "warnings_count": len(warnings),
                "issues": issues,
                "warnings": warnings,
                "summary": {
                    "entity_set": self._entity is not None,
                    "total_arrangements": len(self._arrangements),
                    "active_arrangements": sum(
                        1 for a in self._arrangements.values()
                        if a.status == RegisterStatus.ACTIVE
                    ),
                    "total_providers": len(self._providers),
                    "critical_arrangements": sum(
                        1 for a in self._arrangements.values()
                        if a.supports_critical_function
                    ),
                },
            }

    # =========================================================================
    # Export Functions (ITS Compliance)
    # =========================================================================

    def export_to_csv(
        self,
        reference_date: Optional[str] = None,
        include_terminated: bool = False,
    ) -> Dict[str, str]:
        """
        Export register to CSV format per ITS requirements.

        Returns dictionary with template name as key and CSV content as value.
        """
        with self._lock:
            result = {}

            # B_01.01 - Entity information
            if self._entity:
                result["B_01_01"] = self._export_entity_csv()

            # B_01.02 - Branch information
            if self._branches:
                result["B_01_02"] = self._export_branches_csv()

            # B_02.01 - Contractual arrangements
            result["B_02_01"] = self._export_arrangements_csv(include_terminated)

            # B_02.02 - Arrangement functions
            result["B_02_02"] = self._export_arrangement_functions_csv()

            # B_03.01 - Providers
            result["B_03_01"] = self._export_providers_csv()

            # B_04.01 - Subcontractors
            result["B_04_01"] = self._export_subcontractors_csv()

            # B_06.01 - Services
            result["B_06_01"] = self._export_services_csv()

            # B_99.01 - Totals
            result["B_99_01"] = self._export_totals_csv()

            return result

    def export_to_json(
        self,
        include_terminated: bool = False,
    ) -> Dict[str, Any]:
        """Export register to JSON format."""
        with self._lock:
            arrangements = [
                asdict(a) for a in self._arrangements.values()
                if include_terminated or a.status != RegisterStatus.TERMINATED
            ]

            return {
                "export_date": datetime.now(timezone.utc).isoformat(),
                "entity": asdict(self._entity) if self._entity else None,
                "branches": [asdict(b) for b in self._branches.values()],
                "contractual_arrangements": arrangements,
                "arrangement_functions": [
                    asdict(f) for f in self._arrangement_functions.values()
                ],
                "providers": [asdict(p) for p in self._providers.values()],
                "subcontractors": [asdict(s) for s in self._subcontractors.values()],
                "services": [asdict(s) for s in self._services.values()],
            }

    def _export_entity_csv(self) -> str:
        """Export entity information to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "Entity_ID", "LEI", "Entity_Name", "Entity_Type",
            "Country", "Competent_Authority", "Parent_LEI", "Consolidation_Level"
        ])

        # Data
        e = self._entity
        writer.writerow([
            e.entity_id, e.lei, e.entity_name, e.entity_type,
            e.country_of_incorporation, e.competent_authority,
            e.parent_lei, e.consolidation_level
        ])

        return output.getvalue()

    def _export_branches_csv(self) -> str:
        """Export branches to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Branch_ID", "Entity_LEI", "Branch_Country",
            "Branch_Name", "Branch_Address", "Local_NCA"
        ])

        for b in self._branches.values():
            writer.writerow([
                b.branch_id, b.entity_lei, b.branch_country,
                b.branch_name, b.branch_address, b.local_competent_authority
            ])

        return output.getvalue()

    def _export_arrangements_csv(
        self,
        include_terminated: bool,
    ) -> str:
        """Export arrangements to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Arrangement_ID", "Reference", "Provider_LEI", "Provider_Name",
            "Provider_Country", "Location_Type", "Contract_Type",
            "Start_Date", "End_Date", "Value_EUR", "Notice_Days",
            "Supports_Critical", "Critical_Functions",
            "Data_Processing_Countries", "Data_Storage_Countries",
            "Personal_Data", "Subcontracting_Permitted",
            "Audit_Rights", "Exit_Plan", "Status"
        ])

        for a in self._arrangements.values():
            if not include_terminated and a.status == RegisterStatus.TERMINATED:
                continue

            writer.writerow([
                a.arrangement_id, a.arrangement_reference,
                a.provider_lei, a.provider_name,
                a.provider_country, a.provider_type_location.value,
                a.contract_type.value,
                a.contract_start_date, a.contract_end_date,
                a.contract_value_eur, a.notice_period_days,
                a.supports_critical_function,
                ";".join(a.critical_function_ids),
                ";".join(a.data_processing_countries),
                ";".join(a.data_storage_countries),
                a.personal_data_processed, a.subcontracting_permitted,
                a.audit_rights, a.exit_plan_exists, a.status.value
            ])

        return output.getvalue()

    def _export_arrangement_functions_csv(self) -> str:
        """Export arrangement-function links to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Link_ID", "Arrangement_ID", "Function_ID", "Function_Name",
            "Function_Type", "Is_Critical", "Last_Assessment"
        ])

        for f in self._arrangement_functions.values():
            writer.writerow([
                f.link_id, f.arrangement_id, f.function_id, f.function_name,
                f.function_type.value, f.is_critical_or_important,
                f.date_of_last_assessment
            ])

        return output.getvalue()

    def _export_providers_csv(self) -> str:
        """Export providers to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Provider_ID", "LEI", "Alt_ID", "Alt_ID_Type",
            "Legal_Name", "Trading_Name", "HQ_Country", "HQ_Address",
            "Location_Type", "Is_Intra_Group",
            "Parent_LEI", "Parent_Name",
            "Is_CTPP", "CTPP_Overseer", "Contact_Email"
        ])

        for p in self._providers.values():
            writer.writerow([
                p.provider_id, p.lei, p.alternative_id, p.alternative_id_type,
                p.legal_name, p.trading_name,
                p.headquarters_country, p.headquarters_address,
                p.location_type.value, p.is_ict_intra_group_provider,
                p.parent_lei, p.parent_name,
                p.is_designated_ctpp, p.ctpp_lead_overseer,
                p.primary_contact_email
            ])

        return output.getvalue()

    def _export_subcontractors_csv(self) -> str:
        """Export subcontractors to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Subcontractor_ID", "Arrangement_ID", "Direct_Provider_ID",
            "LEI", "Legal_Name", "Country", "Level", "Chain_Rank",
            "Services", "Supports_Critical",
            "Data_Processing_Countries", "Personal_Data_Access"
        ])

        for s in self._subcontractors.values():
            writer.writerow([
                s.subcontractor_id, s.arrangement_id, s.direct_provider_id,
                s.lei, s.legal_name, s.country,
                s.subcontracting_level.value, s.chain_rank,
                ";".join(s.services_subcontracted), s.supports_critical_function,
                ";".join(s.data_processing_countries), s.personal_data_access
            ])

        return output.getvalue()

    def _export_services_csv(self) -> str:
        """Export services to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Service_ID", "Arrangement_ID", "Provider_ID",
            "Service_Name", "Service_Type", "Description",
            "Supported_Functions", "Supports_Critical",
            "Availability_Target", "RPO_Hours", "RTO_Hours",
            "Data_Classification", "Personal_Data"
        ])

        for s in self._services.values():
            writer.writerow([
                s.service_id, s.arrangement_id, s.provider_id,
                s.service_name, s.service_type.value, s.service_description,
                ";".join(s.supported_functions), s.supports_critical_function,
                s.availability_target_pct, s.rpo_hours, s.rto_hours,
                s.data_classification, s.personal_data_involved
            ])

        return output.getvalue()

    def _export_totals_csv(self) -> str:
        """Export totals summary to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        active_arrangements = [
            a for a in self._arrangements.values()
            if a.status == RegisterStatus.ACTIVE
        ]

        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total_Arrangements", len(active_arrangements)])
        writer.writerow(["Critical_Arrangements", sum(
            1 for a in active_arrangements if a.supports_critical_function
        )])
        writer.writerow(["Total_Providers", len(self._providers)])
        writer.writerow(["CTPP_Providers", sum(
            1 for p in self._providers.values() if p.is_designated_ctpp
        )])
        writer.writerow(["Total_Subcontractors", len(self._subcontractors)])
        writer.writerow(["Total_Services", len(self._services)])

        return output.getvalue()

    # =========================================================================
    # Submission Management
    # =========================================================================

    def generate_annual_submission(
        self,
        reference_date: str,
        export_format: ExportFormat = ExportFormat.CSV,
    ) -> RegisterSubmission:
        """
        Generate annual submission package for NCA.

        Per DORA Article 28(3), financial entities shall make available
        to the competent authority the register of information.

        Args:
            reference_date: Data reference date (typically 31 March)
            export_format: Export format

        Returns:
            RegisterSubmission record
        """
        with self._lock:
            # Validate before submission
            validation = self.validate_register()
            if not validation["is_valid"]:
                logger.warning(
                    f"Generating submission with {validation['issues_count']} issues"
                )

            active_arrangements = [
                a for a in self._arrangements.values()
                if a.status == RegisterStatus.ACTIVE
            ]

            submission = RegisterSubmission(
                reference_date=reference_date,
                competent_authority=self._entity.competent_authority if self._entity else "",
                submission_format=export_format,
                entity_lei=self._entity.lei if self._entity else "",
                entity_name=self._entity.entity_name if self._entity else "",
                consolidation_level=self._entity.consolidation_level if self._entity else "",
                arrangements_count=len(active_arrangements),
                providers_count=len(self._providers),
                functions_count=len(self._arrangement_functions),
                submission_status="prepared",
            )

            self._submissions[submission.submission_id] = submission

        self._log_event("submission_generated", {
            "submission_id": submission.submission_id,
            "reference_date": reference_date,
            "arrangements_count": submission.arrangements_count,
        })

        return submission

    def record_submission_acknowledgment(
        self,
        submission_id: str,
        acknowledgment_reference: str,
        status: str = "acknowledged",
    ) -> Optional[RegisterSubmission]:
        """Record NCA acknowledgment of submission."""
        with self._lock:
            if submission_id not in self._submissions:
                return None

            submission = self._submissions[submission_id]
            submission.acknowledgment_reference = acknowledgment_reference
            submission.submission_status = status

        return submission

    def record_submission_feedback(
        self,
        submission_id: str,
        feedback_details: str,
        status: str = "accepted",
    ) -> Optional[RegisterSubmission]:
        """Record NCA feedback on submission."""
        with self._lock:
            if submission_id not in self._submissions:
                return None

            submission = self._submissions[submission_id]
            submission.feedback_received = True
            submission.feedback_details = feedback_details
            submission.submission_status = status

        return submission

    def get_submission_history(self) -> List[RegisterSubmission]:
        """Get submission history."""
        with self._lock:
            return sorted(
                self._submissions.values(),
                key=lambda s: s.submission_date,
                reverse=True
            )

    def get_next_submission_deadline(self) -> Tuple[str, int]:
        """Get next submission deadline and days remaining."""
        now = datetime.now(timezone.utc)
        current_year = now.year

        # Next deadline
        deadline = datetime(
            year=current_year,
            month=self.config.annual_submission_deadline_month,
            day=self.config.annual_submission_deadline_day,
            tzinfo=timezone.utc
        )

        # If past this year's deadline, use next year
        if now > deadline:
            deadline = datetime(
                year=current_year + 1,
                month=self.config.annual_submission_deadline_month,
                day=self.config.annual_submission_deadline_day,
                tzinfo=timezone.utc
            )

        days_remaining = (deadline - now).days

        return deadline.isoformat(), days_remaining

    # =========================================================================
    # Statistics and Reporting
    # =========================================================================

    def get_register_statistics(self) -> Dict[str, Any]:
        """Get comprehensive register statistics."""
        with self._lock:
            active_arrangements = [
                a for a in self._arrangements.values()
                if a.status == RegisterStatus.ACTIVE
            ]

            # Country breakdown
            provider_countries = {}
            for arr in active_arrangements:
                country = arr.provider_country
                provider_countries[country] = provider_countries.get(country, 0) + 1

            # Contract type breakdown
            contract_types = {}
            for arr in active_arrangements:
                ct = arr.contract_type.value
                contract_types[ct] = contract_types.get(ct, 0) + 1

            deadline_date, days_to_deadline = self.get_next_submission_deadline()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "entity": {
                    "lei": self._entity.lei if self._entity else None,
                    "name": self._entity.entity_name if self._entity else None,
                },
                "arrangements": {
                    "total": len(self._arrangements),
                    "active": len(active_arrangements),
                    "terminated": len(self._arrangements) - len(active_arrangements),
                    "supporting_critical_functions": sum(
                        1 for a in active_arrangements if a.supports_critical_function
                    ),
                    "by_contract_type": contract_types,
                    "total_value_eur": sum(a.contract_value_eur for a in active_arrangements),
                },
                "providers": {
                    "total": len(self._providers),
                    "ctpp_designated": sum(
                        1 for p in self._providers.values() if p.is_designated_ctpp
                    ),
                    "by_country": provider_countries,
                    "eu_providers": sum(
                        1 for a in active_arrangements
                        if a.provider_type_location == ProviderLocationType.EU_MEMBER_STATE
                    ),
                    "third_country_providers": sum(
                        1 for a in active_arrangements
                        if a.provider_type_location == ProviderLocationType.THIRD_COUNTRY
                    ),
                },
                "subcontracting": {
                    "total_subcontractors": len(self._subcontractors),
                    "arrangements_with_subcontracting": sum(
                        1 for a in active_arrangements if a.subcontracting_permitted
                    ),
                    "chains_fully_known": sum(
                        1 for a in active_arrangements
                        if a.subcontracting_permitted and a.subcontractor_chain_known
                    ),
                },
                "services": {
                    "total": len(self._services),
                    "supporting_critical": sum(
                        1 for s in self._services.values() if s.supports_critical_function
                    ),
                },
                "compliance": {
                    "arrangements_with_audit_rights": sum(
                        1 for a in active_arrangements if a.audit_rights
                    ),
                    "arrangements_with_exit_plan": sum(
                        1 for a in active_arrangements if a.exit_plan_exists
                    ),
                    "personal_data_arrangements": sum(
                        1 for a in active_arrangements if a.personal_data_processed
                    ),
                },
                "submissions": {
                    "total": len(self._submissions),
                    "last_submission": max(
                        (s.submission_date for s in self._submissions.values()),
                        default=None
                    ),
                    "next_deadline": deadline_date,
                    "days_to_deadline": days_to_deadline,
                },
            }

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _is_eu_country(self, country_code: str) -> bool:
        """Check if country is EU member state."""
        EU_COUNTRIES = {
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE"
        }
        return country_code.upper() in EU_COUNTRIES

    def _log_event(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Log an event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"register_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_register_of_information(
    config: Optional[RegisterOfInformationConfig] = None,
) -> DORARegisterOfInformation:
    """
    Create a DORARegisterOfInformation instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORARegisterOfInformation instance
    """
    return DORARegisterOfInformation(config=config)


def get_contract_types() -> List[ContractType]:
    """Get all contract types."""
    return list(ContractType)


def get_service_types() -> List[ServiceType]:
    """Get all service types."""
    return list(ServiceType)


def get_its_templates() -> List[str]:
    """Get list of ITS template identifiers."""
    return [
        "B_01_01",  # Entity information
        "B_01_02",  # Branch information
        "B_02_01",  # Contractual arrangements
        "B_02_02",  # Arrangement functions
        "B_03_01",  # ICT service providers
        "B_04_01",  # Subcontracting chain
        "B_05_01",  # Entity making use (group)
        "B_06_01",  # ICT services
        "B_99_01",  # Totals
    ]
