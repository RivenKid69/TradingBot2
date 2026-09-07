# -*- coding: utf-8 -*-
"""
DORA Register of Information (ROI) Data Generator - Integration Layer (Article 28(3)).

This module generates ROI data packages for financial entity clients to populate
their Article 28(3) Registers of Information for NCA submissions.

CRITICAL DISTINCTION - ICT Provider Role:
    We GENERATE data packages for client ROI submissions.
    We DO NOT maintain client registers (that's THEIR obligation).
    We DO NOT submit to NCAs (clients submit via their NCAs).

    Financial entities must maintain their own ROI per Art. 28(3).
    We provide them structured data packages with:
    - Our provider identification (B_03.01)
    - Our service details (B_06.01)
    - Our subcontractor chain (B_04.01)
    - Contract reference data (B_02.01)

Regulation (EU) 2022/2554 Article 28(3) requires financial entities to:
    - Maintain register at entity, sub-consolidated and consolidated levels
    - Update on material changes
    - Make available to NCA upon request
    - Submit to NCA annually (by 30 April per ESA Decision)
    - Follow ITS templates (CIR 2024/2956)

ITS Template Structure (DPM 4.0) - Data we provide:
    - B_02.01: Contractual arrangement level (reference data)
    - B_03.01: ICT third-party service provider identification (OUR data)
    - B_04.01: Subcontracting chain (OUR subcontractors)
    - B_06.01: ICT services supporting functions (OUR services)

    Data client must provide themselves:
    - B_01.01: Entity maintaining register (CLIENT data)
    - B_01.02: Branch information (CLIENT data)
    - B_02.02: Contractual arrangement functions (CLIENT mapping)
    - B_05.01: Entity making use of ICT services (CLIENT data)
    - B_99.01: Totals (CLIENT calculates)

References:
    - Article 28(3) DORA: https://www.digital-operational-resilience-act.com/Article_28.html
    - ITS on Register of Information: CIR 2024/2956
    - ESA Guidelines on ROI: JC 2023/86
    - Reference Date: 31 March 2025
    - Submission Deadline: 30 April 2025 (first submission)

Migration: services/dora/register_of_information.py -> services/dora_integration/reporting/
Refactored: Full register -> ROI Data Generator for clients
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import uuid

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
    CRITICAL = "critical"
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


class ExportFormat(Enum):
    """Export format for ROI data packages."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    DICT = "dict"


# =============================================================================
# ITS Template Data Structures - Provider Data for Client ROI
# =============================================================================

@dataclass
class ProviderIdentification:
    """
    B_03.01 - ICT third-party service provider identification.

    OUR identification data for client ROI population.
    """
    # Provider identifiers
    provider_id: str = ""
    lei: str = ""  # Our LEI (mandatory if we have one)
    alternative_id: str = ""  # If no LEI
    alternative_id_type: str = ""

    # Names
    legal_name: str = ""
    trading_name: str = ""

    # Location
    headquarters_country: str = ""  # ISO 3166-1 alpha-2
    headquarters_address: str = ""
    location_type: ProviderLocationType = ProviderLocationType.EU_MEMBER_STATE

    # Parent company (if applicable)
    parent_lei: str = ""
    parent_name: str = ""
    ultimate_parent_lei: str = ""
    ultimate_parent_name: str = ""
    ultimate_parent_country: str = ""

    # Classification
    is_intra_group_provider: bool = False
    is_designated_ctpp: bool = False
    ctpp_lead_overseer: str = ""

    # Contact
    primary_contact_name: str = ""
    primary_contact_email: str = ""
    primary_contact_phone: str = ""

    # Metadata
    data_as_of_date: str = ""
    generated_at: str = ""

    def __post_init__(self):
        if not self.provider_id:
            self.provider_id = f"PRV-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider_id": self.provider_id,
            "lei": self.lei,
            "alternative_id": self.alternative_id,
            "alternative_id_type": self.alternative_id_type,
            "legal_name": self.legal_name,
            "trading_name": self.trading_name,
            "headquarters_country": self.headquarters_country,
            "headquarters_address": self.headquarters_address,
            "location_type": self.location_type.value,
            "parent_lei": self.parent_lei,
            "parent_name": self.parent_name,
            "ultimate_parent_lei": self.ultimate_parent_lei,
            "ultimate_parent_name": self.ultimate_parent_name,
            "ultimate_parent_country": self.ultimate_parent_country,
            "is_intra_group_provider": self.is_intra_group_provider,
            "is_designated_ctpp": self.is_designated_ctpp,
            "ctpp_lead_overseer": self.ctpp_lead_overseer,
            "primary_contact_name": self.primary_contact_name,
            "primary_contact_email": self.primary_contact_email,
            "primary_contact_phone": self.primary_contact_phone,
            "data_as_of_date": self.data_as_of_date,
            "generated_at": self.generated_at,
        }


@dataclass
class ContractReferenceData:
    """
    B_02.01 - Contract reference data.

    Our data about the contract for client ROI population.
    Client adds their arrangement IDs and function mappings.
    """
    # Reference (our internal)
    contract_reference: str = ""

    # Provider identification
    provider_lei: str = ""
    provider_name: str = ""

    # Contract basics
    contract_type: ContractType = ContractType.PROCUREMENT
    contract_start_date: str = ""
    contract_end_date: str = ""  # Empty = indefinite
    annual_value_eur: float = 0.0
    notice_period_days: int = 30
    renewal_type: str = ""  # automatic, manual, fixed_term

    # Service information
    service_types_provided: List[str] = field(default_factory=list)
    service_descriptions: List[str] = field(default_factory=list)

    # Data handling (for client to assess criticality)
    data_processing_countries: List[str] = field(default_factory=list)
    data_storage_countries: List[str] = field(default_factory=list)
    personal_data_processed: bool = False
    sensitive_data_processed: bool = False

    # Subcontracting
    subcontracting_permitted: bool = False
    subcontractor_chain_provided: bool = True

    # Contractual rights (Art. 30 compliance)
    audit_rights_granted: bool = True
    nca_access_rights: bool = True
    exit_plan_provided: bool = True
    data_portability_supported: bool = True

    # Metadata
    data_as_of_date: str = ""
    generated_at: str = ""

    def __post_init__(self):
        if not self.contract_reference:
            self.contract_reference = f"CTR-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SubcontractorData:
    """
    B_04.01 - Subcontracting chain data.

    Our subcontractor chain data for client ROI population.
    """
    # Subcontractor identification
    subcontractor_id: str = ""
    lei: str = ""
    alternative_id: str = ""
    legal_name: str = ""
    country: str = ""

    # Chain position
    parent_contract_reference: str = ""
    subcontracting_level: SubcontractingLevel = SubcontractingLevel.LEVEL_1
    chain_rank: int = 1

    # Services
    services_subcontracted: List[str] = field(default_factory=list)
    services_description: str = ""

    # Data handling
    data_processing_countries: List[str] = field(default_factory=list)
    personal_data_access: bool = False

    # Notification
    notified_to_clients: bool = True
    notification_date: str = ""

    # Metadata
    data_as_of_date: str = ""
    generated_at: str = ""

    def __post_init__(self):
        if not self.subcontractor_id:
            self.subcontractor_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["subcontracting_level"] = self.subcontracting_level.value
        return data


@dataclass
class ServiceRecord:
    """
    B_06.01 - ICT service record.

    Our service details for client ROI population.
    """
    # Service identification
    service_id: str = ""
    contract_reference: str = ""

    # Service details
    service_name: str = ""
    service_type: ServiceType = ServiceType.OTHER
    service_description: str = ""

    # Service levels
    availability_target_pct: float = 99.0
    rpo_hours: int = 24  # Recovery Point Objective
    rto_hours: int = 4   # Recovery Time Objective

    # For client criticality assessment
    supports_trading_functions: bool = False
    supports_payment_functions: bool = False
    supports_custody_functions: bool = False
    supports_settlement_functions: bool = False
    supports_risk_management: bool = False
    supports_regulatory_reporting: bool = False

    # Data classification (for client assessment)
    data_classification: str = ""
    personal_data_involved: bool = False

    # Metadata
    data_as_of_date: str = ""
    generated_at: str = ""

    def __post_init__(self):
        if not self.service_id:
            self.service_id = f"SVC-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["service_type"] = self.service_type.value
        return data


@dataclass
class ROIDataPackage:
    """
    Complete ROI data package for client.

    Contains all our data that clients need to populate their ROI.
    """
    package_id: str = ""
    generated_at: str = ""
    reference_date: str = ""

    # Provider info
    provider: Optional[ProviderIdentification] = None

    # Contract data
    contracts: List[ContractReferenceData] = field(default_factory=list)

    # Service records
    services: List[ServiceRecord] = field(default_factory=list)

    # Subcontractor chain
    subcontractors: List[SubcontractorData] = field(default_factory=list)

    # Package metadata
    format_version: str = "1.0"
    its_template_version: str = "DPM_4.0"

    # Validation
    is_validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.package_id:
            self.package_id = f"ROI-PKG-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "package_id": self.package_id,
            "generated_at": self.generated_at,
            "reference_date": self.reference_date,
            "format_version": self.format_version,
            "its_template_version": self.its_template_version,
            "provider": self.provider.to_dict() if self.provider else None,
            "contracts": [c.to_dict() for c in self.contracts],
            "services": [s.to_dict() for s in self.services],
            "subcontractors": [s.to_dict() for s in self.subcontractors],
            "is_validated": self.is_validated,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
        }


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ROIDataGeneratorConfig:
    """Configuration for ROI Data Generator."""

    # Our provider information
    provider_lei: str = ""
    provider_name: str = ""
    provider_country: str = ""
    provider_address: str = ""

    # Parent company info
    parent_lei: str = ""
    parent_name: str = ""

    # Contact defaults
    default_contact_name: str = ""
    default_contact_email: str = ""
    default_contact_phone: str = ""

    # CTPP status
    is_designated_ctpp: bool = False
    ctpp_lead_overseer: str = ""

    # Validation settings
    require_lei: bool = True
    validate_countries: bool = True

    # Export settings
    default_export_format: ExportFormat = ExportFormat.JSON


# =============================================================================
# Main Implementation - ROI Data Generator
# =============================================================================

class DORARegisterOfInformation:
    """
    DORA Article 28(3) ROI Data Generator for ICT Service Providers.

    Generates ROI data packages for financial entity clients to populate
    their Register of Information submissions to NCAs.

    CRITICAL: This is a DATA GENERATOR, not a register maintainer.
    Clients maintain their own registers. We provide them data.

    Key Features:
    - Provider identification data (B_03.01)
    - Contract reference data (B_02.01)
    - Service records (B_06.01)
    - Subcontractor chain data (B_04.01)
    - Multi-format export (JSON, CSV, XML)
    - ITS template compliance validation

    Usage:
        config = ROIDataGeneratorConfig(
            provider_lei="549300EXAMPLE0000",
            provider_name="ICT Provider Ltd",
            provider_country="DE",
        )
        generator = DORARegisterOfInformation(config)

        # Add contract
        contract = generator.add_contract(
            contract_type=ContractType.PROCUREMENT,
            service_types_provided=[ServiceType.CLOUD_COMPUTING.value],
        )

        # Add services
        generator.add_service(
            contract_reference=contract.contract_reference,
            service_name="Cloud Hosting",
            service_type=ServiceType.CLOUD_COMPUTING,
        )

        # Generate package for client
        package = generator.generate_roi_data_package(
            reference_date="2025-03-31",
        )

        # Export
        json_data = generator.export_package_to_json(package)
    """

    def __init__(self, config: Optional[ROIDataGeneratorConfig] = None):
        """Initialize ROI Data Generator."""
        self.config = config or ROIDataGeneratorConfig()

        # Our provider identification (constant)
        self._provider_identification = self._create_provider_identification()

        # Contract data
        self._contracts: Dict[str, ContractReferenceData] = {}

        # Services
        self._services: Dict[str, ServiceRecord] = {}

        # Subcontractors
        self._subcontractors: Dict[str, SubcontractorData] = {}

        # Indexes
        self._services_by_contract: Dict[str, set] = {}
        self._subcontractors_by_contract: Dict[str, set] = {}

        logger.info("DORARegisterOfInformation (ROI Data Generator) initialized")

    def _create_provider_identification(self) -> ProviderIdentification:
        """Create our provider identification record."""
        location_type = self._determine_location_type(self.config.provider_country)

        return ProviderIdentification(
            lei=self.config.provider_lei,
            legal_name=self.config.provider_name,
            trading_name=self.config.provider_name,
            headquarters_country=self.config.provider_country,
            headquarters_address=self.config.provider_address,
            location_type=location_type,
            parent_lei=self.config.parent_lei,
            parent_name=self.config.parent_name,
            is_designated_ctpp=self.config.is_designated_ctpp,
            ctpp_lead_overseer=self.config.ctpp_lead_overseer,
            primary_contact_name=self.config.default_contact_name,
            primary_contact_email=self.config.default_contact_email,
            primary_contact_phone=self.config.default_contact_phone,
        )

    def _determine_location_type(self, country_code: str) -> ProviderLocationType:
        """Determine location type from country code."""
        EU_COUNTRIES = {
            "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
            "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
            "PL", "PT", "RO", "SK", "SI", "ES", "SE"
        }
        EEA_COUNTRIES = {"IS", "LI", "NO"}

        code = country_code.upper()
        if code in EU_COUNTRIES:
            return ProviderLocationType.EU_MEMBER_STATE
        elif code in EEA_COUNTRIES:
            return ProviderLocationType.EEA_COUNTRY
        else:
            return ProviderLocationType.THIRD_COUNTRY

    # =========================================================================
    # Provider Data
    # =========================================================================

    def get_provider_identification(self) -> ProviderIdentification:
        """Get our provider identification data."""
        return self._provider_identification

    def update_provider_identification(
        self,
        **updates: Any,
    ) -> ProviderIdentification:
        """
        Update provider identification data.

        Args:
            **updates: Fields to update

        Returns:
            Updated ProviderIdentification
        """
        for key, value in updates.items():
            if hasattr(self._provider_identification, key):
                setattr(self._provider_identification, key, value)

        self._provider_identification.generated_at = (
            datetime.now(timezone.utc).isoformat()
        )

        return self._provider_identification

    # =========================================================================
    # Contract Management
    # =========================================================================

    def add_contract(
        self,
        contract_type: ContractType,
        service_types_provided: Optional[List[str]] = None,
        contract_start_date: str = "",
        contract_end_date: str = "",
        annual_value_eur: float = 0.0,
        notice_period_days: int = 30,
        data_processing_countries: Optional[List[str]] = None,
        data_storage_countries: Optional[List[str]] = None,
        personal_data_processed: bool = False,
        subcontracting_permitted: bool = False,
        audit_rights_granted: bool = True,
        exit_plan_provided: bool = True,
        **kwargs: Any,
    ) -> ContractReferenceData:
        """
        Add a contract to our data.

        Args:
            contract_type: Type of contract
            service_types_provided: Service types
            contract_start_date: Start date
            contract_end_date: End date (empty = indefinite)
            annual_value_eur: Annual contract value
            notice_period_days: Notice period
            data_processing_countries: Countries where data processed
            data_storage_countries: Countries where data stored
            personal_data_processed: Whether personal data processed
            subcontracting_permitted: Whether subcontracting permitted
            audit_rights_granted: Whether audit rights granted
            exit_plan_provided: Whether exit plan provided

        Returns:
            ContractReferenceData
        """
        contract = ContractReferenceData(
            provider_lei=self.config.provider_lei,
            provider_name=self.config.provider_name,
            contract_type=contract_type,
            service_types_provided=service_types_provided or [],
            contract_start_date=contract_start_date,
            contract_end_date=contract_end_date,
            annual_value_eur=annual_value_eur,
            notice_period_days=notice_period_days,
            data_processing_countries=[
                c.upper() for c in (data_processing_countries or [])
            ],
            data_storage_countries=[
                c.upper() for c in (data_storage_countries or [])
            ],
            personal_data_processed=personal_data_processed,
            subcontracting_permitted=subcontracting_permitted,
            audit_rights_granted=audit_rights_granted,
            exit_plan_provided=exit_plan_provided,
        )

        self._contracts[contract.contract_reference] = contract
        self._services_by_contract[contract.contract_reference] = set()
        self._subcontractors_by_contract[contract.contract_reference] = set()

        logger.info(f"Contract added: {contract.contract_reference}")

        return contract

    def get_contract(
        self,
        contract_reference: str,
    ) -> Optional[ContractReferenceData]:
        """Get contract by reference."""
        return self._contracts.get(contract_reference)

    def get_all_contracts(self) -> List[ContractReferenceData]:
        """Get all contracts."""
        return list(self._contracts.values())

    def update_contract(
        self,
        contract_reference: str,
        **updates: Any,
    ) -> Optional[ContractReferenceData]:
        """Update contract data."""
        contract = self._contracts.get(contract_reference)
        if not contract:
            return None

        for key, value in updates.items():
            if hasattr(contract, key):
                setattr(contract, key, value)

        contract.generated_at = datetime.now(timezone.utc).isoformat()

        return contract

    # =========================================================================
    # Service Management
    # =========================================================================

    def add_service(
        self,
        contract_reference: str,
        service_name: str,
        service_type: ServiceType,
        service_description: str = "",
        availability_target_pct: float = 99.0,
        rpo_hours: int = 24,
        rto_hours: int = 4,
        supports_trading_functions: bool = False,
        supports_payment_functions: bool = False,
        supports_custody_functions: bool = False,
        supports_settlement_functions: bool = False,
        supports_risk_management: bool = False,
        supports_regulatory_reporting: bool = False,
        personal_data_involved: bool = False,
        **kwargs: Any,
    ) -> Optional[ServiceRecord]:
        """
        Add a service record.

        Args:
            contract_reference: Contract this service belongs to
            service_name: Service name
            service_type: Service type
            service_description: Description
            availability_target_pct: Availability target
            rpo_hours: Recovery Point Objective
            rto_hours: Recovery Time Objective
            supports_trading_functions: Supports trading
            supports_payment_functions: Supports payments
            supports_custody_functions: Supports custody
            supports_settlement_functions: Supports settlement
            supports_risk_management: Supports risk management
            supports_regulatory_reporting: Supports regulatory reporting
            personal_data_involved: Personal data involved

        Returns:
            ServiceRecord
        """
        if contract_reference not in self._contracts:
            logger.warning(f"Contract not found: {contract_reference}")
            return None

        service = ServiceRecord(
            contract_reference=contract_reference,
            service_name=service_name,
            service_type=service_type,
            service_description=service_description,
            availability_target_pct=availability_target_pct,
            rpo_hours=rpo_hours,
            rto_hours=rto_hours,
            supports_trading_functions=supports_trading_functions,
            supports_payment_functions=supports_payment_functions,
            supports_custody_functions=supports_custody_functions,
            supports_settlement_functions=supports_settlement_functions,
            supports_risk_management=supports_risk_management,
            supports_regulatory_reporting=supports_regulatory_reporting,
            personal_data_involved=personal_data_involved,
        )

        self._services[service.service_id] = service
        self._services_by_contract[contract_reference].add(service.service_id)

        logger.info(f"Service added: {service.service_id}")

        return service

    def get_services_for_contract(
        self,
        contract_reference: str,
    ) -> List[ServiceRecord]:
        """Get services for a contract."""
        service_ids = self._services_by_contract.get(contract_reference, set())
        return [
            self._services[sid]
            for sid in service_ids
            if sid in self._services
        ]

    def get_all_services(self) -> List[ServiceRecord]:
        """Get all services."""
        return list(self._services.values())

    # =========================================================================
    # Subcontractor Management
    # =========================================================================

    def add_subcontractor(
        self,
        parent_contract_reference: str,
        legal_name: str,
        country: str,
        lei: str = "",
        subcontracting_level: SubcontractingLevel = SubcontractingLevel.LEVEL_1,
        services_subcontracted: Optional[List[str]] = None,
        data_processing_countries: Optional[List[str]] = None,
        personal_data_access: bool = False,
        **kwargs: Any,
    ) -> Optional[SubcontractorData]:
        """
        Add subcontractor to the chain.

        Args:
            parent_contract_reference: Parent contract reference
            legal_name: Subcontractor name
            country: Subcontractor country
            lei: LEI (if available)
            subcontracting_level: Level in chain
            services_subcontracted: Services subcontracted
            data_processing_countries: Data processing countries
            personal_data_access: Personal data access

        Returns:
            SubcontractorData
        """
        if parent_contract_reference not in self._contracts:
            logger.warning(f"Contract not found: {parent_contract_reference}")
            return None

        subcontractor = SubcontractorData(
            lei=lei,
            legal_name=legal_name,
            country=country.upper(),
            parent_contract_reference=parent_contract_reference,
            subcontracting_level=subcontracting_level,
            services_subcontracted=services_subcontracted or [],
            data_processing_countries=[
                c.upper() for c in (data_processing_countries or [])
            ],
            personal_data_access=personal_data_access,
            notified_to_clients=True,
            notification_date=datetime.now(timezone.utc).isoformat(),
        )

        self._subcontractors[subcontractor.subcontractor_id] = subcontractor
        self._subcontractors_by_contract[parent_contract_reference].add(
            subcontractor.subcontractor_id
        )

        logger.info(f"Subcontractor added: {subcontractor.subcontractor_id}")

        return subcontractor

    def get_subcontractors_for_contract(
        self,
        contract_reference: str,
    ) -> List[SubcontractorData]:
        """Get subcontractors for a contract."""
        sub_ids = self._subcontractors_by_contract.get(contract_reference, set())
        return [
            self._subcontractors[sid]
            for sid in sub_ids
            if sid in self._subcontractors
        ]

    def get_full_subcontracting_chain(
        self,
        contract_reference: str,
    ) -> Dict[str, Any]:
        """Get full subcontracting chain visualization."""
        contract = self._contracts.get(contract_reference)
        if not contract:
            return {}

        subcontractors = self.get_subcontractors_for_contract(contract_reference)

        return {
            "contract_reference": contract_reference,
            "provider": self.config.provider_name,
            "subcontracting_permitted": contract.subcontracting_permitted,
            "chain": [
                {
                    "level": s.subcontracting_level.value,
                    "name": s.legal_name,
                    "country": s.country,
                    "services": s.services_subcontracted,
                    "personal_data_access": s.personal_data_access,
                }
                for s in sorted(subcontractors, key=lambda x: x.chain_rank)
            ],
        }

    def get_all_subcontractors(self) -> List[SubcontractorData]:
        """Get all subcontractors."""
        return list(self._subcontractors.values())

    # =========================================================================
    # ROI Data Package Generation
    # =========================================================================

    def generate_roi_data_package(
        self,
        reference_date: str = "",
        contract_references: Optional[List[str]] = None,
    ) -> ROIDataPackage:
        """
        Generate ROI data package for client.

        Creates a complete package containing all our data
        that clients need to populate their ROI submissions.

        Args:
            reference_date: Reference date for the data
            contract_references: Specific contracts (None = all)

        Returns:
            ROIDataPackage
        """
        if not reference_date:
            reference_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Determine contracts to include
        if contract_references:
            contracts = [
                self._contracts[ref]
                for ref in contract_references
                if ref in self._contracts
            ]
        else:
            contracts = list(self._contracts.values())

        # Gather services for included contracts
        services = []
        for contract in contracts:
            services.extend(
                self.get_services_for_contract(contract.contract_reference)
            )

        # Gather subcontractors for included contracts
        subcontractors = []
        for contract in contracts:
            subcontractors.extend(
                self.get_subcontractors_for_contract(contract.contract_reference)
            )

        # Update data_as_of_date
        self._provider_identification.data_as_of_date = reference_date

        for contract in contracts:
            contract.data_as_of_date = reference_date

        for service in services:
            service.data_as_of_date = reference_date

        for sub in subcontractors:
            sub.data_as_of_date = reference_date

        # Create package
        package = ROIDataPackage(
            reference_date=reference_date,
            provider=self._provider_identification,
            contracts=contracts,
            services=services,
            subcontractors=subcontractors,
        )

        # Validate
        package = self._validate_package(package)

        logger.info(
            f"ROI data package generated: {package.package_id} "
            f"({len(contracts)} contracts, {len(services)} services, "
            f"{len(subcontractors)} subcontractors)"
        )

        return package

    def _validate_package(self, package: ROIDataPackage) -> ROIDataPackage:
        """Validate ROI data package."""
        errors = []
        warnings = []

        # Validate provider
        if self.config.require_lei and not package.provider.lei:
            warnings.append("Provider LEI not provided (recommended)")

        if not package.provider.legal_name:
            errors.append("Provider legal_name is required")

        # Validate contracts
        for contract in package.contracts:
            if not contract.service_types_provided:
                warnings.append(
                    f"Contract {contract.contract_reference}: "
                    f"no service types specified"
                )

            if contract.subcontracting_permitted:
                subs = [
                    s for s in package.subcontractors
                    if s.parent_contract_reference == contract.contract_reference
                ]
                if not subs:
                    warnings.append(
                        f"Contract {contract.contract_reference}: "
                        f"subcontracting permitted but no subcontractors listed"
                    )

        # Validate data locations
        if self.config.validate_countries:
            for contract in package.contracts:
                for country in contract.data_processing_countries:
                    if not self._is_valid_country(country):
                        warnings.append(
                            f"Contract {contract.contract_reference}: "
                            f"unknown country code {country}"
                        )

        package.is_validated = len(errors) == 0
        package.validation_errors = errors
        package.validation_warnings = warnings

        return package

    def _is_valid_country(self, country_code: str) -> bool:
        """Validate country code (basic check)."""
        return len(country_code) == 2 and country_code.isalpha()

    # =========================================================================
    # Export Functions
    # =========================================================================

    def export_package_to_json(
        self,
        package: ROIDataPackage,
        indent: int = 2,
    ) -> str:
        """Export package to JSON."""
        return json.dumps(package.to_dict(), indent=indent, default=str)

    def export_package_to_csv(
        self,
        package: ROIDataPackage,
    ) -> Dict[str, str]:
        """
        Export package to CSV format.

        Returns dictionary with ITS template name as key and CSV as value.
        """
        result = {}

        # B_03.01 - Provider
        result["B_03_01_Provider"] = self._export_provider_csv(package.provider)

        # B_02.01 - Contracts
        result["B_02_01_Contracts"] = self._export_contracts_csv(package.contracts)

        # B_06.01 - Services
        result["B_06_01_Services"] = self._export_services_csv(package.services)

        # B_04.01 - Subcontractors
        result["B_04_01_Subcontractors"] = self._export_subcontractors_csv(
            package.subcontractors
        )

        return result

    def _export_provider_csv(self, provider: ProviderIdentification) -> str:
        """Export provider to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Provider_ID", "LEI", "Alternative_ID", "Legal_Name", "Trading_Name",
            "HQ_Country", "HQ_Address", "Location_Type", "Is_CTPP",
            "CTPP_Overseer", "Contact_Name", "Contact_Email", "Data_As_Of"
        ])

        writer.writerow([
            provider.provider_id, provider.lei, provider.alternative_id,
            provider.legal_name, provider.trading_name,
            provider.headquarters_country, provider.headquarters_address,
            provider.location_type.value, provider.is_designated_ctpp,
            provider.ctpp_lead_overseer, provider.primary_contact_name,
            provider.primary_contact_email, provider.data_as_of_date
        ])

        return output.getvalue()

    def _export_contracts_csv(
        self,
        contracts: List[ContractReferenceData],
    ) -> str:
        """Export contracts to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Contract_Reference", "Provider_LEI", "Provider_Name",
            "Contract_Type", "Start_Date", "End_Date", "Value_EUR",
            "Notice_Days", "Service_Types", "Data_Processing_Countries",
            "Personal_Data", "Subcontracting_Permitted", "Audit_Rights",
            "Exit_Plan", "Data_As_Of"
        ])

        for c in contracts:
            writer.writerow([
                c.contract_reference, c.provider_lei, c.provider_name,
                c.contract_type.value, c.contract_start_date, c.contract_end_date,
                c.annual_value_eur, c.notice_period_days,
                ";".join(c.service_types_provided),
                ";".join(c.data_processing_countries),
                c.personal_data_processed, c.subcontracting_permitted,
                c.audit_rights_granted, c.exit_plan_provided, c.data_as_of_date
            ])

        return output.getvalue()

    def _export_services_csv(self, services: List[ServiceRecord]) -> str:
        """Export services to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Service_ID", "Contract_Reference", "Service_Name", "Service_Type",
            "Availability_Target", "RPO_Hours", "RTO_Hours",
            "Supports_Trading", "Supports_Payments", "Supports_Custody",
            "Personal_Data", "Data_As_Of"
        ])

        for s in services:
            writer.writerow([
                s.service_id, s.contract_reference, s.service_name,
                s.service_type.value, s.availability_target_pct,
                s.rpo_hours, s.rto_hours,
                s.supports_trading_functions, s.supports_payment_functions,
                s.supports_custody_functions, s.personal_data_involved,
                s.data_as_of_date
            ])

        return output.getvalue()

    def _export_subcontractors_csv(
        self,
        subcontractors: List[SubcontractorData],
    ) -> str:
        """Export subcontractors to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "Subcontractor_ID", "Contract_Reference", "LEI", "Legal_Name",
            "Country", "Level", "Chain_Rank", "Services",
            "Data_Processing_Countries", "Personal_Data_Access",
            "Notified_Date", "Data_As_Of"
        ])

        for s in subcontractors:
            writer.writerow([
                s.subcontractor_id, s.parent_contract_reference, s.lei,
                s.legal_name, s.country, s.subcontracting_level.value,
                s.chain_rank, ";".join(s.services_subcontracted),
                ";".join(s.data_processing_countries), s.personal_data_access,
                s.notification_date, s.data_as_of_date
            ])

        return output.getvalue()

    def export_package_to_xml(self, package: ROIDataPackage) -> str:
        """Export package to XML format."""
        def dict_to_xml(d: Dict[str, Any], root_tag: str) -> str:
            xml_parts = [f"<{root_tag}>"]
            for key, value in d.items():
                if isinstance(value, list):
                    xml_parts.append(f"<{key}>")
                    for item in value:
                        if isinstance(item, dict):
                            xml_parts.append(dict_to_xml(item, "item"))
                        else:
                            xml_parts.append(
                                f"<item>{_escape_xml(str(item))}</item>"
                            )
                    xml_parts.append(f"</{key}>")
                elif isinstance(value, dict):
                    xml_parts.append(dict_to_xml(value, key))
                else:
                    xml_parts.append(f"<{key}>{_escape_xml(str(value))}</{key}>")
            xml_parts.append(f"</{root_tag}>")
            return "".join(xml_parts)

        xml_content = dict_to_xml(package.to_dict(), "ROI_DataPackage")
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_content}'

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            "provider": {
                "lei": self.config.provider_lei,
                "name": self.config.provider_name,
                "is_ctpp": self.config.is_designated_ctpp,
            },
            "data_counts": {
                "contracts": len(self._contracts),
                "services": len(self._services),
                "subcontractors": len(self._subcontractors),
            },
            "contracts_by_type": self._count_contracts_by_type(),
            "services_by_type": self._count_services_by_type(),
        }

    def _count_contracts_by_type(self) -> Dict[str, int]:
        """Count contracts by type."""
        counts = {}
        for contract in self._contracts.values():
            ct = contract.contract_type.value
            counts[ct] = counts.get(ct, 0) + 1
        return counts

    def _count_services_by_type(self) -> Dict[str, int]:
        """Count services by type."""
        counts = {}
        for service in self._services.values():
            st = service.service_type.value
            counts[st] = counts.get(st, 0) + 1
        return counts


# =============================================================================
# Helper Functions
# =============================================================================

def _escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )


def create_register_of_information(
    config: Optional[ROIDataGeneratorConfig] = None,
) -> DORARegisterOfInformation:
    """
    Create a DORARegisterOfInformation (ROI Data Generator) instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORARegisterOfInformation instance
    """
    return DORARegisterOfInformation(config=config)


def create_roi_data_generator(
    provider_lei: str = "",
    provider_name: str = "",
    provider_country: str = "",
    **kwargs: Any,
) -> DORARegisterOfInformation:
    """
    Create ROI Data Generator with provider info.

    Args:
        provider_lei: Provider LEI
        provider_name: Provider name
        provider_country: Provider country
        **kwargs: Additional config options

    Returns:
        Configured DORARegisterOfInformation instance
    """
    config = ROIDataGeneratorConfig(
        provider_lei=provider_lei,
        provider_name=provider_name,
        provider_country=provider_country,
        **kwargs,
    )
    return DORARegisterOfInformation(config=config)


def get_contract_types() -> List[ContractType]:
    """Get all contract types."""
    return list(ContractType)


def get_service_types() -> List[ServiceType]:
    """Get all service types."""
    return list(ServiceType)


def get_subcontracting_levels() -> List[SubcontractingLevel]:
    """Get all subcontracting levels."""
    return list(SubcontractingLevel)


def get_its_templates_provided() -> List[str]:
    """Get list of ITS template identifiers we provide data for."""
    return [
        "B_02_01",  # Contractual arrangements (reference data)
        "B_03_01",  # ICT service providers (our identification)
        "B_04_01",  # Subcontracting chain (our subcontractors)
        "B_06_01",  # ICT services (our services)
    ]


def get_its_templates_client_provides() -> List[str]:
    """Get list of ITS template identifiers clients must provide themselves."""
    return [
        "B_01_01",  # Entity maintaining register (client data)
        "B_01_02",  # Branch information (client data)
        "B_02_02",  # Contractual arrangement functions (client mapping)
        "B_05_01",  # Entity making use of ICT services (client data)
        "B_99_01",  # Totals (client calculates)
    ]
