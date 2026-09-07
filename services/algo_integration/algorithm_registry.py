# -*- coding: utf-8 -*-
"""
Algorithm Registration Module per Article 17(2) MiFID II.

Investment firms engaged in algorithmic trading must notify the
competent authority (NCA) and maintain records of all deployed algorithms.

Requirements (Article 17 MiFID II & RTS 6):
    - Notify NCA of algorithmic trading activities
    - Maintain algorithm inventory with unique identifiers
    - Record responsible persons for each algorithm
    - Document algorithm purpose and risk controls
    - Track modifications and version history
    - Annual self-assessment (RTS 6 Article 9)

References:
    - Article 17 MiFID II: Algorithmic trading notification
    - RTS 6 Article 8: Testing and deployment requirements
    - RTS 6 Article 9: Annual self-assessment
    - Kroll Guide: https://www.kroll.com/en/publications/financial-compliance-regulation/algorithmic-trading-under-mifid-ii
"""

from __future__ import annotations

import logging
import json
import uuid
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional, Dict, List, Any, Set
from enum import Enum
from pathlib import Path
from threading import RLock

logger = logging.getLogger(__name__)


class AlgorithmType(str, Enum):
    """
    Algorithm classification per MiFID II taxonomy.

    Categories help NCAs understand the nature and risk profile
    of deployed algorithmic trading strategies.
    """

    EXECUTION = "execution"
    """Execution algorithms: TWAP, VWAP, POV, Implementation Shortfall"""

    DECISION = "decision"
    """Decision-making algorithms: Signal generation, portfolio optimization"""

    MARKET_MAKING = "market_making"
    """Market making algorithms: Quote provision, inventory management"""

    ARBITRAGE = "arbitrage"
    """Arbitrage algorithms: Statistical arbitrage, cross-asset, latency"""

    HIGH_FREQUENCY = "high_frequency"
    """High-frequency trading algorithms (subject to additional RTS 6 requirements)"""

    HYBRID = "hybrid"
    """Hybrid algorithms combining multiple strategy types"""


class AlgorithmStatus(str, Enum):
    """Algorithm lifecycle status."""

    DEVELOPMENT = "development"
    """Under development, not yet deployed"""

    TESTING = "testing"
    """In testing/UAT environment"""

    PAPER_TRADING = "paper_trading"
    """Paper trading validation"""

    PRODUCTION = "production"
    """Live production deployment"""

    SUSPENDED = "suspended"
    """Temporarily suspended (e.g., for review)"""

    RETIRED = "retired"
    """Permanently retired from use"""


@dataclass
class AlgorithmRiskControl:
    """
    Risk control configuration for an algorithm.

    Per RTS 6 Article 15, algorithms must have appropriate risk controls.
    """

    control_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_enabled: bool = True
    last_tested: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_type": self.control_type,
            "description": self.description,
            "parameters": self.parameters,
            "is_enabled": self.is_enabled,
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmRiskControl":
        last_tested = data.get("last_tested")
        if isinstance(last_tested, str):
            last_tested = datetime.fromisoformat(last_tested)
        return cls(
            control_type=data.get("control_type", ""),
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            is_enabled=data.get("is_enabled", True),
            last_tested=last_tested,
        )


@dataclass
class AlgorithmRecord:
    """
    Comprehensive algorithm registration record per MiFID II Article 17.

    Contains all required information for NCA notification and
    internal compliance documentation.
    """

    # Identification
    algorithm_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"
    version_hash: str = ""

    # Classification
    algorithm_type: AlgorithmType = AlgorithmType.DECISION
    sub_type: str = ""
    description: str = ""

    # Responsibility (RTS 6 requirement)
    responsible_person: str = ""
    responsible_email: str = ""
    responsible_phone: str = ""
    development_team: str = ""
    compliance_reviewer: str = ""

    # Deployment
    status: AlgorithmStatus = AlgorithmStatus.DEVELOPMENT
    deployment_date: Optional[datetime] = None
    last_modification: Optional[datetime] = None
    retirement_date: Optional[datetime] = None

    # Trading scope
    asset_classes: List[str] = field(default_factory=list)
    instruments: List[str] = field(default_factory=list)
    venues: List[str] = field(default_factory=list)
    markets: List[str] = field(default_factory=list)

    # Risk controls (RTS 6 Article 15)
    risk_controls: List[AlgorithmRiskControl] = field(default_factory=list)
    max_order_size: Optional[float] = None
    max_position_size: Optional[float] = None
    max_daily_turnover: Optional[float] = None

    # Testing (RTS 6 Article 8)
    last_conformance_test: Optional[datetime] = None
    conformance_test_passed: bool = False
    test_environment: str = ""
    test_report_path: str = ""

    # HFT-specific (if applicable)
    is_hft: bool = False
    median_order_lifetime_ms: Optional[float] = None
    message_rate_per_second: Optional[float] = None

    # Audit
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    modification_history: List[Dict[str, Any]] = field(default_factory=list)

    def is_active(self) -> bool:
        """Check if algorithm is in active status."""
        return self.status in (
            AlgorithmStatus.PRODUCTION,
            AlgorithmStatus.PAPER_TRADING,
        )

    def requires_hft_notification(self) -> bool:
        """
        Check if algorithm requires HFT notification per RTS 6.

        Per RTS 6 Article 19, HFT is defined by:
            - High message rate (> 2 per second on average)
            - Short order lifetime (median < 500ms)
        """
        if self.is_hft:
            return True
        if self.median_order_lifetime_ms is not None and self.median_order_lifetime_ms < 500:
            return True
        if self.message_rate_per_second is not None and self.message_rate_per_second > 2:
            return True
        return False

    def calculate_version_hash(self) -> str:
        """Calculate hash of algorithm configuration for change detection."""
        data = {
            "name": self.name,
            "version": self.version,
            "algorithm_type": self.algorithm_type.value,
            "risk_controls": [rc.to_dict() for rc in self.risk_controls],
            "max_order_size": self.max_order_size,
            "max_position_size": self.max_position_size,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "algorithm_id": self.algorithm_id,
            "name": self.name,
            "version": self.version,
            "version_hash": self.version_hash,
            "algorithm_type": self.algorithm_type.value,
            "sub_type": self.sub_type,
            "description": self.description,
            "responsible_person": self.responsible_person,
            "responsible_email": self.responsible_email,
            "responsible_phone": self.responsible_phone,
            "development_team": self.development_team,
            "compliance_reviewer": self.compliance_reviewer,
            "status": self.status.value,
            "deployment_date": self.deployment_date.isoformat() if self.deployment_date else None,
            "last_modification": self.last_modification.isoformat() if self.last_modification else None,
            "retirement_date": self.retirement_date.isoformat() if self.retirement_date else None,
            "asset_classes": self.asset_classes,
            "instruments": self.instruments,
            "venues": self.venues,
            "markets": self.markets,
            "risk_controls": [rc.to_dict() for rc in self.risk_controls],
            "max_order_size": self.max_order_size,
            "max_position_size": self.max_position_size,
            "max_daily_turnover": self.max_daily_turnover,
            "last_conformance_test": self.last_conformance_test.isoformat() if self.last_conformance_test else None,
            "conformance_test_passed": self.conformance_test_passed,
            "test_environment": self.test_environment,
            "test_report_path": self.test_report_path,
            "is_hft": self.is_hft,
            "median_order_lifetime_ms": self.median_order_lifetime_ms,
            "message_rate_per_second": self.message_rate_per_second,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "modification_history": self.modification_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmRecord":
        """Create AlgorithmRecord from dictionary."""

        def parse_datetime(value: Any) -> Optional[datetime]:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                return datetime.fromisoformat(value)
            return None

        algo_type = data.get("algorithm_type", "decision")
        try:
            algo_type = AlgorithmType(algo_type)
        except ValueError:
            algo_type = AlgorithmType.DECISION

        status = data.get("status", "development")
        try:
            status = AlgorithmStatus(status)
        except ValueError:
            status = AlgorithmStatus.DEVELOPMENT

        risk_controls = [
            AlgorithmRiskControl.from_dict(rc)
            for rc in data.get("risk_controls", [])
        ]

        return cls(
            algorithm_id=data.get("algorithm_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            version_hash=data.get("version_hash", ""),
            algorithm_type=algo_type,
            sub_type=data.get("sub_type", ""),
            description=data.get("description", ""),
            responsible_person=data.get("responsible_person", ""),
            responsible_email=data.get("responsible_email", ""),
            responsible_phone=data.get("responsible_phone", ""),
            development_team=data.get("development_team", ""),
            compliance_reviewer=data.get("compliance_reviewer", ""),
            status=status,
            deployment_date=parse_datetime(data.get("deployment_date")),
            last_modification=parse_datetime(data.get("last_modification")),
            retirement_date=parse_datetime(data.get("retirement_date")),
            asset_classes=data.get("asset_classes", []),
            instruments=data.get("instruments", []),
            venues=data.get("venues", []),
            markets=data.get("markets", []),
            risk_controls=risk_controls,
            max_order_size=data.get("max_order_size"),
            max_position_size=data.get("max_position_size"),
            max_daily_turnover=data.get("max_daily_turnover"),
            last_conformance_test=parse_datetime(data.get("last_conformance_test")),
            conformance_test_passed=data.get("conformance_test_passed", False),
            test_environment=data.get("test_environment", ""),
            test_report_path=data.get("test_report_path", ""),
            is_hft=data.get("is_hft", False),
            median_order_lifetime_ms=data.get("median_order_lifetime_ms"),
            message_rate_per_second=data.get("message_rate_per_second"),
            created_at=parse_datetime(data.get("created_at")) or datetime.utcnow(),
            updated_at=parse_datetime(data.get("updated_at")) or datetime.utcnow(),
            modification_history=data.get("modification_history", []),
        )

    def to_nca_report_format(self) -> Dict[str, Any]:
        """
        Format record for NCA (National Competent Authority) reporting.

        Returns simplified format suitable for regulatory submission.
        """
        return {
            "algorithm_id": self.algorithm_id,
            "algorithm_name": self.name,
            "algorithm_version": self.version,
            "algorithm_type": self.algorithm_type.value,
            "is_hft": self.requires_hft_notification(),
            "responsible_person": self.responsible_person,
            "contact_email": self.responsible_email,
            "contact_phone": self.responsible_phone,
            "deployment_date": self.deployment_date.date().isoformat() if self.deployment_date else None,
            "status": self.status.value,
            "asset_classes": self.asset_classes,
            "trading_venues": self.venues,
            "risk_controls_count": len(self.risk_controls),
            "last_conformance_test": self.last_conformance_test.date().isoformat() if self.last_conformance_test else None,
            "conformance_passed": self.conformance_test_passed,
        }


class AlgorithmRegistry:
    """
    MiFID II Algorithm Registry for regulatory compliance.

    Maintains comprehensive records of all deployed algorithms
    as required by Article 17(2) MiFID II.

    Features:
        - Algorithm registration and lifecycle management
        - Version tracking and modification history
        - NCA reporting format generation
        - Annual self-assessment data
        - Persistence to disk

    Example:
        registry = AlgorithmRegistry()

        # Register new algorithm
        algo = AlgorithmRecord(
            name="PPO_Momentum",
            algorithm_type=AlgorithmType.DECISION,
            responsible_person="John Smith",
        )
        registry.register(algo)

        # Get active algorithms
        active = registry.get_active_algorithms()

        # Generate NCA report
        report = registry.generate_nca_report()
    """

    def __init__(
        self,
        registry_path: str = "state/compliance/algorithm_registry.json",
        auto_register: bool = True,
        require_responsible_person: bool = True,
        version_on_modification: bool = True,
        nca_jurisdiction: str = "",
        firm_name: str = "",
    ):
        """
        Initialize Algorithm Registry.

        Args:
            registry_path: Path to persist registry data.
            auto_register: Auto-register algorithms on first use.
            require_responsible_person: Require responsible person for production.
            version_on_modification: Auto-increment version on modifications.
            nca_jurisdiction: NCA jurisdiction code (e.g., 'FCA', 'BaFin').
            firm_name: Legal name of the investment firm.
        """
        self._registry_path = Path(registry_path)
        self._auto_register = auto_register
        self._require_responsible_person = require_responsible_person
        self._version_on_modification = version_on_modification
        self._nca_jurisdiction = nca_jurisdiction
        self._firm_name = firm_name

        # In-memory registry
        self._algorithms: Dict[str, AlgorithmRecord] = {}
        self._lock = RLock()

        # Load existing registry
        self._load()

        logger.info(
            "AlgorithmRegistry initialized",
            extra={
                "registry_path": str(self._registry_path),
                "algorithm_count": len(self._algorithms),
                "nca_jurisdiction": nca_jurisdiction,
            },
        )

    def _load(self) -> None:
        """Load registry from disk."""
        if not self._registry_path.exists():
            return

        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for algo_data in data.get("algorithms", []):
                algo = AlgorithmRecord.from_dict(algo_data)
                self._algorithms[algo.algorithm_id] = algo

            logger.info(f"Loaded {len(self._algorithms)} algorithms from registry")

        except Exception as e:
            logger.error(f"Failed to load algorithm registry: {e}")

    def _save(self) -> None:
        """Save registry to disk."""
        try:
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "firm_name": self._firm_name,
                "nca_jurisdiction": self._nca_jurisdiction,
                "algorithms": [algo.to_dict() for algo in self._algorithms.values()],
            }

            with open(self._registry_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save algorithm registry: {e}")

    def register(self, algo: AlgorithmRecord) -> str:
        """
        Register a new algorithm.

        Args:
            algo: AlgorithmRecord to register.

        Returns:
            Algorithm ID.

        Raises:
            ValueError: If validation fails.
        """
        # Validate
        if not algo.name:
            raise ValueError("Algorithm name is required")

        if self._require_responsible_person and algo.status == AlgorithmStatus.PRODUCTION:
            if not algo.responsible_person:
                raise ValueError("Responsible person required for production algorithms")

        # Generate ID if not set
        if not algo.algorithm_id:
            algo.algorithm_id = str(uuid.uuid4())

        # Calculate version hash
        algo.version_hash = algo.calculate_version_hash()
        algo.created_at = datetime.utcnow()
        algo.updated_at = datetime.utcnow()

        with self._lock:
            self._algorithms[algo.algorithm_id] = algo
            self._save()

        logger.info(
            f"Registered algorithm: {algo.name} ({algo.algorithm_id})",
            extra={"algorithm_id": algo.algorithm_id, "type": algo.algorithm_type.value},
        )

        return algo.algorithm_id

    def update(self, algo: AlgorithmRecord, reason: str = "") -> None:
        """
        Update an existing algorithm record.

        Args:
            algo: Updated AlgorithmRecord.
            reason: Reason for modification (for audit trail).
        """
        with self._lock:
            if algo.algorithm_id not in self._algorithms:
                raise ValueError(f"Algorithm not found: {algo.algorithm_id}")

            old_algo = self._algorithms[algo.algorithm_id]
            old_hash = old_algo.version_hash

            # Calculate new hash
            new_hash = algo.calculate_version_hash()

            # Track modification
            if old_hash != new_hash:
                modification = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "previous_version": old_algo.version,
                    "new_version": algo.version,
                    "previous_hash": old_hash,
                    "new_hash": new_hash,
                    "reason": reason,
                }
                algo.modification_history = old_algo.modification_history + [modification]

                # Auto-increment version if enabled
                if self._version_on_modification:
                    parts = algo.version.split(".")
                    if len(parts) >= 3:
                        try:
                            parts[-1] = str(int(parts[-1]) + 1)
                            algo.version = ".".join(parts)
                        except ValueError:
                            pass

            algo.version_hash = new_hash
            algo.updated_at = datetime.utcnow()
            algo.last_modification = datetime.utcnow()

            self._algorithms[algo.algorithm_id] = algo
            self._save()

        logger.info(f"Updated algorithm: {algo.name}", extra={"algorithm_id": algo.algorithm_id})

    def get(self, algorithm_id: str) -> Optional[AlgorithmRecord]:
        """Get algorithm by ID."""
        with self._lock:
            return self._algorithms.get(algorithm_id)

    def get_by_name(self, name: str) -> Optional[AlgorithmRecord]:
        """Get algorithm by name (returns first match)."""
        with self._lock:
            for algo in self._algorithms.values():
                if algo.name == name:
                    return algo
        return None

    def get_all(self) -> List[AlgorithmRecord]:
        """Get all registered algorithms."""
        with self._lock:
            return list(self._algorithms.values())

    def get_active_algorithms(self) -> List[AlgorithmRecord]:
        """Get all active (production/paper_trading) algorithms."""
        with self._lock:
            return [algo for algo in self._algorithms.values() if algo.is_active()]

    def get_by_status(self, status: AlgorithmStatus) -> List[AlgorithmRecord]:
        """Get algorithms by status."""
        with self._lock:
            return [algo for algo in self._algorithms.values() if algo.status == status]

    def get_by_type(self, algo_type: AlgorithmType) -> List[AlgorithmRecord]:
        """Get algorithms by type."""
        with self._lock:
            return [algo for algo in self._algorithms.values() if algo.algorithm_type == algo_type]

    def retire(self, algorithm_id: str, reason: str = "") -> None:
        """
        Retire an algorithm (mark as no longer in use).

        Args:
            algorithm_id: ID of algorithm to retire.
            reason: Reason for retirement.
        """
        with self._lock:
            if algorithm_id not in self._algorithms:
                raise ValueError(f"Algorithm not found: {algorithm_id}")

            algo = self._algorithms[algorithm_id]
            algo.status = AlgorithmStatus.RETIRED
            algo.retirement_date = datetime.utcnow()
            algo.updated_at = datetime.utcnow()

            modification = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "retirement",
                "reason": reason,
            }
            algo.modification_history.append(modification)

            self._save()

        logger.info(f"Retired algorithm: {algorithm_id}", extra={"reason": reason})

    def generate_nca_report(self) -> Dict[str, Any]:
        """
        Generate report for National Competent Authority.

        Per Article 17(2), firms must notify NCA of algorithmic trading
        activities and be able to provide information on request.
        """
        with self._lock:
            active = self.get_active_algorithms()
            hft_algos = [a for a in active if a.requires_hft_notification()]

        return {
            "report_type": "NCA_ALGORITHMIC_TRADING_NOTIFICATION",
            "report_date": datetime.utcnow().isoformat(),
            "firm_name": self._firm_name,
            "nca_jurisdiction": self._nca_jurisdiction,
            "regulation_reference": "Article 17(2) MiFID II (Directive 2014/65/EU)",
            "summary": {
                "total_algorithms": len(self._algorithms),
                "active_algorithms": len(active),
                "hft_algorithms": len(hft_algos),
                "algorithm_types": list(set(a.algorithm_type.value for a in active)),
            },
            "algorithms": [algo.to_nca_report_format() for algo in active],
            "hft_notification_required": len(hft_algos) > 0,
        }

    def generate_annual_self_assessment(self) -> Dict[str, Any]:
        """
        Generate annual self-assessment data per RTS 6 Article 9.

        Investment firms must perform annual self-assessment of their
        algorithmic trading systems and strategies.
        """
        with self._lock:
            all_algos = list(self._algorithms.values())
            active = [a for a in all_algos if a.is_active()]
            retired_this_year = [
                a for a in all_algos
                if a.retirement_date and a.retirement_date.year == datetime.utcnow().year
            ]

        # Calculate compliance metrics
        tested_algos = [a for a in active if a.last_conformance_test and a.conformance_test_passed]
        with_risk_controls = [a for a in active if len(a.risk_controls) > 0]
        with_responsible_person = [a for a in active if a.responsible_person]

        return {
            "assessment_type": "ANNUAL_SELF_ASSESSMENT",
            "assessment_date": datetime.utcnow().isoformat(),
            "assessment_period": {
                "start": f"{datetime.utcnow().year}-01-01",
                "end": datetime.utcnow().date().isoformat(),
            },
            "regulation_reference": "RTS 6 Article 9",
            "firm_name": self._firm_name,
            "metrics": {
                "total_algorithms": len(all_algos),
                "active_algorithms": len(active),
                "retired_this_year": len(retired_this_year),
                "conformance_tested": len(tested_algos),
                "conformance_test_rate": len(tested_algos) / len(active) if active else 0,
                "with_risk_controls": len(with_risk_controls),
                "risk_control_rate": len(with_risk_controls) / len(active) if active else 0,
                "with_responsible_person": len(with_responsible_person),
                "responsibility_rate": len(with_responsible_person) / len(active) if active else 0,
            },
            "compliance_status": {
                "risk_controls_adequate": len(with_risk_controls) == len(active),
                "testing_current": all(
                    a.last_conformance_test and
                    (datetime.utcnow() - a.last_conformance_test).days < 365
                    for a in active
                ),
                "documentation_complete": len(with_responsible_person) == len(active),
            },
            "recommendations": [],
            "algorithm_inventory": [algo.to_dict() for algo in active],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            all_algos = list(self._algorithms.values())

        by_status = {}
        for status in AlgorithmStatus:
            count = len([a for a in all_algos if a.status == status])
            if count > 0:
                by_status[status.value] = count

        by_type = {}
        for algo_type in AlgorithmType:
            count = len([a for a in all_algos if a.algorithm_type == algo_type])
            if count > 0:
                by_type[algo_type.value] = count

        return {
            "total_algorithms": len(all_algos),
            "by_status": by_status,
            "by_type": by_type,
            "hft_algorithms": len([a for a in all_algos if a.requires_hft_notification()]),
            "registry_path": str(self._registry_path),
            "firm_name": self._firm_name,
            "nca_jurisdiction": self._nca_jurisdiction,
        }


def create_algorithm_registry(
    config: Optional[Dict[str, Any]] = None,
) -> AlgorithmRegistry:
    """
    Factory function to create AlgorithmRegistry instance.

    Args:
        config: Optional configuration dictionary.

    Returns:
        Configured AlgorithmRegistry instance.
    """
    config = config or {}

    return AlgorithmRegistry(
        registry_path=config.get("registry_path", "state/compliance/algorithm_registry.json"),
        auto_register=config.get("auto_register", True),
        require_responsible_person=config.get("require_responsible_person", True),
        version_on_modification=config.get("version_on_modification", True),
        nca_jurisdiction=config.get("nca_jurisdiction", ""),
        firm_name=config.get("firm_name", ""),
    )


def get_default_algorithm_types() -> List[Dict[str, str]]:
    """
    Get default algorithm type definitions for documentation.

    Returns:
        List of algorithm type descriptions.
    """
    return [
        {
            "type": AlgorithmType.EXECUTION.value,
            "description": "Execution algorithms (TWAP, VWAP, POV, IS)",
            "examples": ["TWAP", "VWAP", "POV", "Implementation Shortfall"],
        },
        {
            "type": AlgorithmType.DECISION.value,
            "description": "Decision-making algorithms for signal generation",
            "examples": ["PPO", "DQN", "Rule-based signals"],
        },
        {
            "type": AlgorithmType.MARKET_MAKING.value,
            "description": "Market making and quote provision",
            "examples": ["Avellaneda-Stoikov", "Basic MM"],
        },
        {
            "type": AlgorithmType.ARBITRAGE.value,
            "description": "Arbitrage strategies",
            "examples": ["Statistical arbitrage", "Pairs trading"],
        },
        {
            "type": AlgorithmType.HIGH_FREQUENCY.value,
            "description": "High-frequency trading (subject to RTS 6 HFT requirements)",
            "examples": ["Latency arbitrage", "Market microstructure"],
        },
    ]


__all__ = [
    "AlgorithmType",
    "AlgorithmStatus",
    "AlgorithmRiskControl",
    "AlgorithmRecord",
    "AlgorithmRegistry",
    "create_algorithm_registry",
    "get_default_algorithm_types",
]
