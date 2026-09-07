# -*- coding: utf-8 -*-
"""
DORA Provider Information Package Module.

For ICT Third-Party Service Providers: Generates data packages that clients
need to populate their Register of Information (ROI) per Article 28(3).

DORA Context:
    - Financial entities must maintain a Register of Information (ROI)
    - ROI includes information about ALL ICT third-party service providers
    - WE provide the DATA; CLIENTS submit to their NCA
    - CIR 2024/2956 defines the ITS templates (B_01 to B_99)

Key Templates We Support:
    - B_02.01: ICT third-party service provider information
    - B_03.01: Type of ICT services
    - B_05.01/02: Function identification
    - B_06.01: ICT service assessment
    - B_99.01: Subcontractor chains

References:
    - DORA Article 28(3): Register of information requirement
    - CIR 2024/2956: ITS on Register of Information templates
    - DORA Article 30(2)(b): Data location disclosure
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations (Aligned with ITS templates)
# =============================================================================


class ICTServiceType(Enum):
    """ITS-aligned ICT service type codes."""

    # From B_03.01 template
    CLOUD_SERVICES = "CS"  # Cloud computing services
    DATA_ANALYTICS = "DA"  # Data analytics services
    DATA_CENTRE = "DC"  # Data centre services
    ICT_PLATFORM = "IP"  # ICT platform operation
    ICT_SECURITY = "IS"  # ICT security services
    NETWORK_INFRA = "NI"  # Network infrastructure
    SOFTWARE_DEV = "SD"  # Software development/maintenance
    OTHER = "OT"  # Other ICT services


class FunctionCriticality(Enum):
    """Function criticality classification."""

    CRITICAL = "critical"
    IMPORTANT = "important"
    NEITHER = "neither"


class SubstitutabilityLevel(Enum):
    """Ease of substitution assessment."""

    EASY = "easy"  # Multiple alternatives readily available
    MEDIUM = "medium"  # Some alternatives with transition effort
    DIFFICULT = "difficult"  # Few alternatives, significant effort
    NOT_SUBSTITUTABLE = "not_substitutable"  # No alternatives available


class DataSensitivity(Enum):
    """Data sensitivity classification."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


# =============================================================================
# Data Structures (ITS-aligned)
# =============================================================================


@dataclass
class ProviderIdentification:
    """
    Provider identification information (B_02.01 aligned).

    Required fields for client ROI.
    """

    # Entity identification
    provider_name: str = ""
    legal_name: str = ""
    lei_code: str = ""  # Legal Entity Identifier (if available)
    alternative_id: str = ""  # If no LEI
    alternative_id_type: str = ""  # Type of alternative ID

    # Registration
    registration_country: str = ""  # ISO 3166-1 alpha-2
    registration_number: str = ""

    # Parent company (if applicable)
    parent_company_name: str = ""
    parent_company_lei: str = ""
    parent_company_country: str = ""

    # Contact
    headquarters_address: str = ""
    headquarters_country: str = ""
    contact_email: str = ""
    contact_phone: str = ""

    # Provider type
    is_intra_group: bool = False
    is_ctpp_designated: bool = False

    def to_its_format(self) -> Dict[str, Any]:
        """Export in ITS B_02.01 format."""
        return {
            "B_02_01_0010": self.provider_name,
            "B_02_01_0020": self.lei_code or self.alternative_id,
            "B_02_01_0030": "LEI" if self.lei_code else self.alternative_id_type,
            "B_02_01_0040": self.registration_country,
            "B_02_01_0050": self.headquarters_country,
            "B_02_01_0060": "Y" if self.is_intra_group else "N",
            "B_02_01_0070": "Y" if self.is_ctpp_designated else "N",
            "B_02_01_0080": self.parent_company_lei or "",
        }


@dataclass
class ServiceDescription:
    """
    ICT service description (B_03.01 aligned).
    """

    service_id: str = ""
    service_name: str = ""
    service_type: ICTServiceType = ICTServiceType.OTHER
    service_description: str = ""

    # Features
    features: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)

    # Supported functions (client maps these to their functions)
    function_types_supported: List[str] = field(default_factory=list)

    # Dependencies
    third_party_dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.service_id:
            self.service_id = f"SVC-{uuid.uuid4().hex[:8].upper()}"

    def to_its_format(self) -> Dict[str, Any]:
        """Export in ITS B_03.01 format."""
        return {
            "B_03_01_0010": self.service_id,
            "B_03_01_0020": self.service_type.value,
            "B_03_01_0030": self.service_name,
            "B_03_01_0040": self.service_description,
        }


@dataclass
class DataLocation:
    """
    Data processing and storage location (B_04.01 aligned).
    """

    location_id: str = ""
    location_type: str = ""  # processing, storage, backup
    country_code: str = ""  # ISO 3166-1 alpha-2
    region: str = ""  # AWS region, Azure region, etc.
    data_center_name: str = ""
    cloud_provider: str = ""  # If applicable

    # Data types at this location
    data_types: List[str] = field(default_factory=list)
    data_sensitivity: DataSensitivity = DataSensitivity.CONFIDENTIAL

    # Compliance
    eu_adequacy_decision: bool = True  # Is data transfer lawful?
    transfer_mechanism: str = ""  # SCCs, BCRs, etc. if outside EEA

    def __post_init__(self):
        if not self.location_id:
            self.location_id = f"LOC-{uuid.uuid4().hex[:8].upper()}"

    def to_its_format(self) -> Dict[str, Any]:
        """Export in ITS B_04.01 format."""
        return {
            "B_04_01_0010": self.location_id,
            "B_04_01_0020": self.location_type,
            "B_04_01_0030": self.country_code,
            "B_04_01_0040": self.region,
            "B_04_01_0050": "Y" if self.eu_adequacy_decision else "N",
        }


@dataclass
class SubcontractorInfo:
    """
    Subcontractor information (B_99.01 aligned).
    """

    subcontractor_id: str = ""
    subcontractor_name: str = ""
    subcontractor_lei: str = ""
    subcontractor_country: str = ""

    # Services provided
    services_provided: List[str] = field(default_factory=list)
    service_description: str = ""

    # Data handling
    data_locations: List[str] = field(default_factory=list)  # Country codes
    has_data_access: bool = False
    data_types_accessed: List[str] = field(default_factory=list)

    # Criticality
    is_material: bool = False  # Material subcontracting?
    supports_critical_function: bool = False

    # Chain
    subcontractor_chain_level: int = 1  # 1 = direct, 2 = sub-sub, etc.

    # Certifications
    certifications: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.subcontractor_id:
            self.subcontractor_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"

    def to_its_format(self) -> Dict[str, Any]:
        """Export in ITS B_99.01 format."""
        return {
            "B_99_01_0010": self.subcontractor_id,
            "B_99_01_0020": self.subcontractor_name,
            "B_99_01_0030": self.subcontractor_lei or "",
            "B_99_01_0040": self.subcontractor_country,
            "B_99_01_0050": self.service_description,
            "B_99_01_0060": ",".join(self.data_locations),
            "B_99_01_0070": "Y" if self.is_material else "N",
            "B_99_01_0080": str(self.subcontractor_chain_level),
        }


@dataclass
class CertificationInfo:
    """
    Certification and security attestation information.
    """

    certification_type: str = ""  # SOC2, ISO27001, C5, etc.
    certification_scope: str = ""
    issuing_body: str = ""
    issue_date: str = ""
    expiry_date: str = ""
    certificate_reference: str = ""

    # Availability
    available_on_request: bool = True
    available_in_portal: bool = False
    portal_url: str = ""


@dataclass
class ContractSummary:
    """
    Contract summary information for client ROI.
    """

    contract_reference: str = ""
    contract_start_date: str = ""
    contract_end_date: str = ""
    renewal_type: str = ""  # automatic, manual
    notice_period_days: int = 90

    # Services in scope
    services_in_scope: List[str] = field(default_factory=list)

    # Criticality (as assessed by client)
    supports_critical_function: bool = False
    critical_functions: List[str] = field(default_factory=list)

    # SLA
    sla_tier: str = ""  # standard, professional, enterprise


@dataclass
class ProviderInfoPackage:
    """
    Complete provider information package for client ROI.

    This is what we provide to clients to help them populate
    their Register of Information submission.
    """

    package_id: str = ""
    package_version: str = "1.0"
    generated_at: str = ""
    valid_until: str = ""

    # For which client
    client_id: str = ""
    client_name: str = ""

    # Provider information
    provider: ProviderIdentification = field(default_factory=ProviderIdentification)

    # Services
    services: List[ServiceDescription] = field(default_factory=list)

    # Data locations
    data_locations: List[DataLocation] = field(default_factory=list)

    # Subcontractors
    subcontractors: List[SubcontractorInfo] = field(default_factory=list)

    # Certifications
    certifications: List[CertificationInfo] = field(default_factory=list)

    # Contract
    contract: ContractSummary = field(default_factory=ContractSummary)

    # Additional information
    additional_notes: str = ""
    regulatory_status: str = ""  # Any regulatory designations

    def __post_init__(self):
        if not self.package_id:
            self.package_id = f"PKG-{uuid.uuid4().hex[:8].upper()}"
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ProviderInfoConfig:
    """Configuration for provider info package generation."""

    # Our company information
    company_name: str = "AI-Powered Quantitative Research Platform"
    legal_name: str = ""
    lei_code: str = ""
    registration_country: str = "NL"  # Netherlands
    registration_number: str = ""
    headquarters_country: str = "NL"

    # Package settings
    default_validity_days: int = 365
    include_detailed_locations: bool = True
    include_subcontractor_chain: bool = True

    # Storage
    output_path: str = "data/dora/provider_packages"
    log_path: str = "logs/dora/provider_packages"


# =============================================================================
# Main Implementation
# =============================================================================


class DORAProviderInfoPackage:
    """
    DORA Provider Information Package Generator.

    Generates data packages for financial entity clients to use in their
    Register of Information (ROI) submissions per Article 28(3).

    Key Features:
    - ITS-aligned data export (CIR 2024/2956 templates)
    - Per-client customization
    - Subcontractor chain documentation
    - Multiple export formats (JSON, CSV, PDF)
    - Version tracking

    Usage:
        config = ProviderInfoConfig(
            company_name="My Company",
            lei_code="549300...",
        )
        generator = DORAProviderInfoPackage(config)

        # Generate package for a client
        package = generator.generate_package(
            client_id="client1",
            client_name="Bank XYZ",
            services_in_scope=["trading_platform", "market_data"],
        )

        # Export in ITS format
        its_data = generator.export_its_format(package)
    """

    def __init__(self, config: Optional[ProviderInfoConfig] = None):
        """Initialize provider info package generator."""
        self.config = config or ProviderInfoConfig()

        # Standard provider identification
        self._provider = ProviderIdentification(
            provider_name=self.config.company_name,
            legal_name=self.config.legal_name or self.config.company_name,
            lei_code=self.config.lei_code,
            registration_country=self.config.registration_country,
            registration_number=self.config.registration_number,
            headquarters_country=self.config.headquarters_country,
        )

        # Standard services we offer
        self._services: List[ServiceDescription] = []

        # Data locations
        self._data_locations: List[DataLocation] = []

        # Subcontractors
        self._subcontractors: List[SubcontractorInfo] = []

        # Certifications
        self._certifications: List[CertificationInfo] = []

        # Generated packages
        self._packages: Dict[str, ProviderInfoPackage] = {}

        # Setup directories
        self._output_path = Path(self.config.output_path)
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._log_path = Path(self.config.log_path)
        self._log_path.mkdir(parents=True, exist_ok=True)

        # Initialize with our standard services
        self._initialize_standard_services()
        self._initialize_standard_locations()
        self._initialize_standard_subcontractors()

        logger.info("DORAProviderInfoPackage initialized")

    # =========================================================================
    # Service Definition
    # =========================================================================

    def _initialize_standard_services(self) -> None:
        """Initialize our standard service offerings."""
        self._services = [
            ServiceDescription(
                service_id="SVC-TRADING",
                service_name="Algorithmic Trading Platform",
                service_type=ICTServiceType.ICT_PLATFORM,
                service_description="Cloud-based algorithmic and AI-powered trading platform "
                "supporting strategy development, backtesting, paper trading, "
                "and live trading integration with multiple brokers.",
                features=[
                    "Strategy development IDE",
                    "Historical backtesting engine",
                    "Paper trading simulation",
                    "Live trading execution",
                    "Multi-broker integration (IB, Alpaca, Binance)",
                    "Real-time market data",
                ],
                capabilities=[
                    "Order routing and execution",
                    "Position management",
                    "Risk monitoring",
                    "Performance analytics",
                ],
                function_types_supported=[
                    "trading_and_execution",
                    "risk_management",
                    "market_data",
                    "analytics",
                ],
            ),
            ServiceDescription(
                service_id="SVC-RESEARCH",
                service_name="Quantitative Research Platform",
                service_type=ICTServiceType.DATA_ANALYTICS,
                service_description="AI/ML-powered quantitative research environment "
                "for financial data analysis, model training, and signal generation.",
                features=[
                    "Jupyter notebook environment",
                    "Pre-built ML models",
                    "Factor library",
                    "Signal generation",
                    "Portfolio optimization",
                ],
                capabilities=[
                    "Data analysis",
                    "Model training",
                    "Backtesting",
                    "Research collaboration",
                ],
                function_types_supported=[
                    "quantitative_research",
                    "risk_modeling",
                    "analytics",
                ],
            ),
            ServiceDescription(
                service_id="SVC-DATA",
                service_name="Market Data Services",
                service_type=ICTServiceType.DATA_ANALYTICS,
                service_description="Real-time and historical market data aggregation "
                "from multiple sources with normalization and storage.",
                features=[
                    "Real-time price feeds",
                    "Historical data access",
                    "Alternative data integration",
                    "Data normalization",
                ],
                capabilities=[
                    "Multi-source aggregation",
                    "Low-latency delivery",
                    "Data quality checks",
                ],
                function_types_supported=[
                    "market_data",
                    "analytics",
                ],
                third_party_dependencies=[
                    "Polygon.io",
                    "Alpaca Market Data",
                    "Binance",
                ],
            ),
        ]

    def _initialize_standard_locations(self) -> None:
        """Initialize our standard data locations."""
        self._data_locations = [
            DataLocation(
                location_id="LOC-PRIMARY",
                location_type="processing",
                country_code="IE",
                region="eu-west-1",
                data_center_name="AWS Dublin",
                cloud_provider="AWS",
                data_types=["trading_data", "user_data", "market_data"],
                data_sensitivity=DataSensitivity.CONFIDENTIAL,
                eu_adequacy_decision=True,
            ),
            DataLocation(
                location_id="LOC-BACKUP",
                location_type="backup",
                country_code="DE",
                region="eu-central-1",
                data_center_name="AWS Frankfurt",
                cloud_provider="AWS",
                data_types=["trading_data", "user_data"],
                data_sensitivity=DataSensitivity.CONFIDENTIAL,
                eu_adequacy_decision=True,
            ),
            DataLocation(
                location_id="LOC-DR",
                location_type="disaster_recovery",
                country_code="NL",
                region="eu-west-3",
                data_center_name="AWS Paris",
                cloud_provider="AWS",
                data_types=["critical_backups"],
                data_sensitivity=DataSensitivity.CONFIDENTIAL,
                eu_adequacy_decision=True,
            ),
        ]

    def _initialize_standard_subcontractors(self) -> None:
        """Initialize our standard subcontractor list."""
        self._subcontractors = [
            SubcontractorInfo(
                subcontractor_id="SUB-AWS",
                subcontractor_name="Amazon Web Services EMEA SARL",
                subcontractor_lei="",  # AWS doesn't have LEI
                subcontractor_country="LU",
                services_provided=["cloud_infrastructure", "compute", "storage", "networking"],
                service_description="Cloud infrastructure (IaaS) including compute, storage, "
                "networking, and managed database services",
                data_locations=["IE", "DE", "FR"],
                has_data_access=True,
                data_types_accessed=["encrypted_storage", "compute_workloads"],
                is_material=True,
                supports_critical_function=True,
                subcontractor_chain_level=1,
                certifications=["SOC2", "ISO27001", "C5", "ENS High"],
            ),
            SubcontractorInfo(
                subcontractor_id="SUB-POLYGON",
                subcontractor_name="Polygon.io, Inc.",
                subcontractor_country="US",
                services_provided=["market_data"],
                service_description="Real-time and historical market data API for stocks, "
                "options, forex, and crypto",
                data_locations=["US"],
                has_data_access=False,
                is_material=False,
                supports_critical_function=False,
                subcontractor_chain_level=1,
                certifications=["SOC2"],
            ),
            SubcontractorInfo(
                subcontractor_id="SUB-ALPACA",
                subcontractor_name="AlpacaDB, Inc.",
                subcontractor_country="US",
                services_provided=["brokerage_api", "market_data"],
                service_description="Commission-free stock and crypto trading API",
                data_locations=["US"],
                has_data_access=True,
                data_types_accessed=["trading_orders", "account_data"],
                is_material=True,
                supports_critical_function=True,
                subcontractor_chain_level=1,
                certifications=["SOC2", "FINRA Member"],
            ),
        ]

    def add_service(self, service: ServiceDescription) -> None:
        """Add a service to the standard offerings."""
        self._services.append(service)

    def add_location(self, location: DataLocation) -> None:
        """Add a data location."""
        self._data_locations.append(location)

    def add_subcontractor(self, subcontractor: SubcontractorInfo) -> None:
        """Add a subcontractor."""
        self._subcontractors.append(subcontractor)

    def add_certification(self, certification: CertificationInfo) -> None:
        """Add a certification."""
        self._certifications.append(certification)

    # =========================================================================
    # Package Generation
    # =========================================================================

    def generate_package(
        self,
        client_id: str,
        client_name: str,
        contract_reference: str = "",
        services_in_scope: Optional[List[str]] = None,
        supports_critical_function: bool = False,
        critical_functions: Optional[List[str]] = None,
        sla_tier: str = "standard",
        contract_start_date: str = "",
        contract_end_date: str = "",
    ) -> ProviderInfoPackage:
        """
        Generate a provider information package for a specific client.

        Args:
            client_id: Client identifier
            client_name: Client display name
            contract_reference: Contract reference number
            services_in_scope: List of service IDs in scope
            supports_critical_function: Whether we support client's critical function
            critical_functions: List of critical functions supported
            sla_tier: SLA tier (standard, professional, enterprise)
            contract_start_date: Contract start date
            contract_end_date: Contract end date

        Returns:
            Generated ProviderInfoPackage
        """
        # Filter services by scope
        if services_in_scope:
            services = [
                s
                for s in self._services
                if s.service_id in services_in_scope or s.service_name in services_in_scope
            ]
        else:
            services = self._services.copy()

        # Filter subcontractors relevant to selected services
        relevant_subcontractors = self._get_relevant_subcontractors(services)

        # Calculate validity
        valid_until = datetime.now(timezone.utc).replace(year=datetime.now().year + 1).isoformat()

        package = ProviderInfoPackage(
            client_id=client_id,
            client_name=client_name,
            provider=self._provider,
            services=services,
            data_locations=self._data_locations.copy(),
            subcontractors=relevant_subcontractors,
            certifications=self._certifications.copy(),
            contract=ContractSummary(
                contract_reference=contract_reference,
                contract_start_date=contract_start_date,
                contract_end_date=contract_end_date,
                services_in_scope=[s.service_id for s in services],
                supports_critical_function=supports_critical_function,
                critical_functions=critical_functions or [],
                sla_tier=sla_tier,
            ),
            valid_until=valid_until,
        )

        self._packages[package.package_id] = package

        self._log_event(
            "package_generated",
            {
                "package_id": package.package_id,
                "client_id": client_id,
                "services_count": len(services),
            },
        )

        return package

    def _get_relevant_subcontractors(
        self,
        services: List[ServiceDescription],
    ) -> List[SubcontractorInfo]:
        """Get subcontractors relevant to the selected services."""
        # AWS is always relevant (infrastructure)
        relevant = [s for s in self._subcontractors if s.subcontractor_id == "SUB-AWS"]

        # Check service dependencies
        for service in services:
            for dep in service.third_party_dependencies:
                for sub in self._subcontractors:
                    if dep.lower() in sub.subcontractor_name.lower():
                        if sub not in relevant:
                            relevant.append(sub)

        return relevant

    # =========================================================================
    # Export Functions
    # =========================================================================

    def export_its_format(
        self,
        package: ProviderInfoPackage,
    ) -> Dict[str, Any]:
        """
        Export package in ITS template format (CIR 2024/2956).

        Returns data structured for client to import into their ROI system.
        """
        return {
            "metadata": {
                "package_id": package.package_id,
                "package_version": package.package_version,
                "generated_at": package.generated_at,
                "valid_until": package.valid_until,
                "its_regulation": "CIR 2024/2956",
            },
            "client": {
                "client_id": package.client_id,
                "client_name": package.client_name,
            },
            "templates": {
                "B_02_01": package.provider.to_its_format(),
                "B_03_01": [s.to_its_format() for s in package.services],
                "B_04_01": [l.to_its_format() for l in package.data_locations],
                "B_99_01": [s.to_its_format() for s in package.subcontractors],
            },
            "contract_summary": asdict(package.contract),
            "certifications": [asdict(c) for c in package.certifications],
        }

    def export_json(
        self,
        package: ProviderInfoPackage,
        file_path: Optional[str] = None,
    ) -> str:
        """Export package as JSON file."""
        data = {
            "package_id": package.package_id,
            "package_version": package.package_version,
            "generated_at": package.generated_at,
            "valid_until": package.valid_until,
            "client": {
                "client_id": package.client_id,
                "client_name": package.client_name,
            },
            "provider": asdict(package.provider),
            "services": [asdict(s) for s in package.services],
            "data_locations": [asdict(l) for l in package.data_locations],
            "subcontractors": [asdict(s) for s in package.subcontractors],
            "certifications": [asdict(c) for c in package.certifications],
            "contract": asdict(package.contract),
        }

        if file_path is None:
            file_path = str(self._output_path / f"{package.client_id}_{package.package_id}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return file_path

    def export_csv(
        self,
        package: ProviderInfoPackage,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Export package as multiple CSV files (one per ITS template).

        Returns dict of template name to file path.
        """
        import csv

        if output_dir is None:
            output_dir = str(self._output_path / package.package_id)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        files = {}

        # B_02_01 - Provider info
        provider_file = output_path / "B_02_01_provider.csv"
        with open(provider_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=package.provider.to_its_format().keys())
            writer.writeheader()
            writer.writerow(package.provider.to_its_format())
        files["B_02_01"] = str(provider_file)

        # B_03_01 - Services
        if package.services:
            services_file = output_path / "B_03_01_services.csv"
            with open(services_file, "w", newline="", encoding="utf-8") as f:
                fieldnames = package.services[0].to_its_format().keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for service in package.services:
                    writer.writerow(service.to_its_format())
            files["B_03_01"] = str(services_file)

        # B_04_01 - Locations
        if package.data_locations:
            locations_file = output_path / "B_04_01_locations.csv"
            with open(locations_file, "w", newline="", encoding="utf-8") as f:
                fieldnames = package.data_locations[0].to_its_format().keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for location in package.data_locations:
                    writer.writerow(location.to_its_format())
            files["B_04_01"] = str(locations_file)

        # B_99_01 - Subcontractors
        if package.subcontractors:
            subs_file = output_path / "B_99_01_subcontractors.csv"
            with open(subs_file, "w", newline="", encoding="utf-8") as f:
                fieldnames = package.subcontractors[0].to_its_format().keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for sub in package.subcontractors:
                    writer.writerow(sub.to_its_format())
            files["B_99_01"] = str(subs_file)

        return files

    def get_package(self, package_id: str) -> Optional[ProviderInfoPackage]:
        """Get a generated package by ID."""
        return self._packages.get(package_id)

    def get_packages_for_client(self, client_id: str) -> List[ProviderInfoPackage]:
        """Get all packages generated for a client."""
        return [p for p in self._packages.values() if p.client_id == client_id]

    # =========================================================================
    # Summary and Reporting
    # =========================================================================

    def get_provider_summary(self) -> Dict[str, Any]:
        """Get summary of provider information for quick reference."""
        return {
            "provider": {
                "name": self._provider.provider_name,
                "legal_name": self._provider.legal_name,
                "lei": self._provider.lei_code,
                "country": self._provider.registration_country,
            },
            "services_count": len(self._services),
            "services": [
                {
                    "id": s.service_id,
                    "name": s.service_name,
                    "type": s.service_type.value,
                }
                for s in self._services
            ],
            "data_locations": [
                {
                    "country": l.country_code,
                    "type": l.location_type,
                    "provider": l.cloud_provider,
                }
                for l in self._data_locations
            ],
            "subcontractors_count": len(self._subcontractors),
            "material_subcontractors": [
                s.subcontractor_name for s in self._subcontractors if s.is_material
            ],
            "certifications": [c.certification_type for c in self._certifications],
            "packages_generated": len(self._packages),
        }

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
        }

        log_file = self._log_path / f"packages_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


# =============================================================================
# Factory Functions
# =============================================================================


def create_provider_info_package(
    config: Optional[ProviderInfoConfig] = None,
) -> DORAProviderInfoPackage:
    """
    Create a DORAProviderInfoPackage instance.

    Args:
        config: Optional configuration

    Returns:
        Configured DORAProviderInfoPackage instance
    """
    return DORAProviderInfoPackage(config=config)
