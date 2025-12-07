# -*- coding: utf-8 -*-
"""
Tests for MiFID II Algorithm Registry Module.

Comprehensive tests for algorithm registration per Article 17(2) MiFID II.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, date
from pathlib import Path
from unittest.mock import Mock, patch

from services.compliance.algorithm_registry import (
    AlgorithmType,
    AlgorithmStatus,
    AlgorithmRiskControl,
    AlgorithmRecord,
    AlgorithmRegistry,
    create_algorithm_registry,
    get_default_algorithm_types,
)


class TestAlgorithmType:
    """Tests for AlgorithmType enum."""

    def test_execution_value(self):
        assert AlgorithmType.EXECUTION.value == "execution"

    def test_decision_value(self):
        assert AlgorithmType.DECISION.value == "decision"

    def test_market_making_value(self):
        assert AlgorithmType.MARKET_MAKING.value == "market_making"

    def test_arbitrage_value(self):
        assert AlgorithmType.ARBITRAGE.value == "arbitrage"

    def test_high_frequency_value(self):
        assert AlgorithmType.HIGH_FREQUENCY.value == "high_frequency"

    def test_hybrid_value(self):
        assert AlgorithmType.HYBRID.value == "hybrid"


class TestAlgorithmStatus:
    """Tests for AlgorithmStatus enum."""

    def test_development_value(self):
        assert AlgorithmStatus.DEVELOPMENT.value == "development"

    def test_testing_value(self):
        assert AlgorithmStatus.TESTING.value == "testing"

    def test_paper_trading_value(self):
        assert AlgorithmStatus.PAPER_TRADING.value == "paper_trading"

    def test_production_value(self):
        assert AlgorithmStatus.PRODUCTION.value == "production"

    def test_suspended_value(self):
        assert AlgorithmStatus.SUSPENDED.value == "suspended"

    def test_retired_value(self):
        assert AlgorithmStatus.RETIRED.value == "retired"


class TestAlgorithmRiskControl:
    """Tests for AlgorithmRiskControl dataclass."""

    def test_create_risk_control(self):
        control = AlgorithmRiskControl(
            control_type="price_collar",
            description="Max 5% price deviation",
            parameters={"max_deviation_pct": 5.0},
        )
        assert control.control_type == "price_collar"
        assert control.is_enabled is True

    def test_to_dict(self):
        control = AlgorithmRiskControl(
            control_type="max_order_size",
            description="Maximum order size limit",
            parameters={"max_size": 1000},
            is_enabled=True,
            last_tested=datetime(2024, 1, 1, 12, 0, 0),
        )
        data = control.to_dict()
        assert data["control_type"] == "max_order_size"
        assert data["is_enabled"] is True
        assert data["last_tested"] is not None

    def test_from_dict(self):
        data = {
            "control_type": "rate_limit",
            "description": "Message rate limit",
            "parameters": {"max_per_second": 100},
            "is_enabled": True,
            "last_tested": "2024-01-01T12:00:00",
        }
        control = AlgorithmRiskControl.from_dict(data)
        assert control.control_type == "rate_limit"
        assert control.last_tested == datetime(2024, 1, 1, 12, 0, 0)

    def test_from_dict_without_last_tested(self):
        data = {
            "control_type": "test",
            "description": "Test control",
        }
        control = AlgorithmRiskControl.from_dict(data)
        assert control.last_tested is None


class TestAlgorithmRecord:
    """Tests for AlgorithmRecord dataclass."""

    def test_create_basic_record(self):
        record = AlgorithmRecord(name="Test Algorithm")
        assert record.name == "Test Algorithm"
        assert record.algorithm_type == AlgorithmType.DECISION
        assert record.status == AlgorithmStatus.DEVELOPMENT
        assert record.algorithm_id is not None

    def test_create_full_record(self):
        record = AlgorithmRecord(
            name="PPO Momentum Strategy",
            version="2.1.0",
            algorithm_type=AlgorithmType.DECISION,
            description="PPO-based momentum trading",
            responsible_person="John Smith",
            responsible_email="john@example.com",
            status=AlgorithmStatus.PRODUCTION,
            deployment_date=datetime(2024, 1, 1),
            asset_classes=["equity", "forex"],
            venues=["NYSE", "NASDAQ"],
        )
        assert record.name == "PPO Momentum Strategy"
        assert record.responsible_person == "John Smith"
        assert "equity" in record.asset_classes

    def test_is_active_production(self):
        record = AlgorithmRecord(
            name="Test",
            status=AlgorithmStatus.PRODUCTION,
        )
        assert record.is_active() is True

    def test_is_active_paper_trading(self):
        record = AlgorithmRecord(
            name="Test",
            status=AlgorithmStatus.PAPER_TRADING,
        )
        assert record.is_active() is True

    def test_is_active_development(self):
        record = AlgorithmRecord(
            name="Test",
            status=AlgorithmStatus.DEVELOPMENT,
        )
        assert record.is_active() is False

    def test_is_active_retired(self):
        record = AlgorithmRecord(
            name="Test",
            status=AlgorithmStatus.RETIRED,
        )
        assert record.is_active() is False

    def test_requires_hft_notification_explicit(self):
        record = AlgorithmRecord(
            name="HFT Strategy",
            is_hft=True,
        )
        assert record.requires_hft_notification() is True

    def test_requires_hft_notification_by_lifetime(self):
        record = AlgorithmRecord(
            name="Fast Strategy",
            median_order_lifetime_ms=100,
        )
        assert record.requires_hft_notification() is True

    def test_requires_hft_notification_by_message_rate(self):
        record = AlgorithmRecord(
            name="High Rate Strategy",
            message_rate_per_second=5.0,
        )
        assert record.requires_hft_notification() is True

    def test_requires_hft_notification_false(self):
        record = AlgorithmRecord(
            name="Normal Strategy",
            is_hft=False,
            median_order_lifetime_ms=1000,
            message_rate_per_second=1.0,
        )
        assert record.requires_hft_notification() is False

    def test_calculate_version_hash(self):
        record = AlgorithmRecord(
            name="Test",
            version="1.0.0",
            algorithm_type=AlgorithmType.DECISION,
        )
        hash1 = record.calculate_version_hash()
        assert len(hash1) == 16

        # Different config should produce different hash
        record.max_order_size = 1000
        hash2 = record.calculate_version_hash()
        assert hash1 != hash2

    def test_to_dict(self):
        record = AlgorithmRecord(
            name="Test Algorithm",
            algorithm_type=AlgorithmType.EXECUTION,
            status=AlgorithmStatus.TESTING,
        )
        data = record.to_dict()
        assert data["name"] == "Test Algorithm"
        assert data["algorithm_type"] == "execution"
        assert data["status"] == "testing"
        assert "algorithm_id" in data
        assert "created_at" in data

    def test_from_dict(self):
        data = {
            "algorithm_id": "test-id-123",
            "name": "Test Algorithm",
            "version": "1.2.3",
            "algorithm_type": "execution",
            "status": "production",
            "responsible_person": "Jane Doe",
            "deployment_date": "2024-01-01T00:00:00",
            "asset_classes": ["equity"],
            "risk_controls": [
                {"control_type": "max_size", "description": "Max size", "parameters": {}}
            ],
        }
        record = AlgorithmRecord.from_dict(data)
        assert record.algorithm_id == "test-id-123"
        assert record.name == "Test Algorithm"
        assert record.algorithm_type == AlgorithmType.EXECUTION
        assert record.status == AlgorithmStatus.PRODUCTION
        assert len(record.risk_controls) == 1

    def test_from_dict_invalid_type(self):
        data = {"name": "Test", "algorithm_type": "invalid_type"}
        record = AlgorithmRecord.from_dict(data)
        assert record.algorithm_type == AlgorithmType.DECISION  # Default

    def test_to_nca_report_format(self):
        record = AlgorithmRecord(
            name="Test Strategy",
            algorithm_type=AlgorithmType.DECISION,
            responsible_person="John Smith",
            responsible_email="john@example.com",
            status=AlgorithmStatus.PRODUCTION,
            deployment_date=datetime(2024, 1, 1),
        )
        report = record.to_nca_report_format()
        assert report["algorithm_name"] == "Test Strategy"
        assert report["responsible_person"] == "John Smith"
        assert report["contact_email"] == "john@example.com"


class TestAlgorithmRegistry:
    """Tests for AlgorithmRegistry class."""

    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary file for registry."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            yield f.name
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass

    def test_init_default(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        assert registry._auto_register is True
        assert len(registry._algorithms) == 0

    def test_init_with_firm_info(self, temp_registry_path):
        registry = AlgorithmRegistry(
            registry_path=temp_registry_path,
            firm_name="Test Firm Ltd",
            nca_jurisdiction="FCA",
        )
        assert registry._firm_name == "Test Firm Ltd"
        assert registry._nca_jurisdiction == "FCA"

    def test_register_algorithm(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        algo = AlgorithmRecord(
            name="Test Algorithm",
            algorithm_type=AlgorithmType.DECISION,
        )
        algo_id = registry.register(algo)
        assert algo_id is not None
        assert algo_id == algo.algorithm_id

    def test_register_algorithm_without_name_fails(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        algo = AlgorithmRecord(name="")
        with pytest.raises(ValueError, match="name is required"):
            registry.register(algo)

    def test_register_production_without_responsible_fails(self, temp_registry_path):
        registry = AlgorithmRegistry(
            registry_path=temp_registry_path,
            require_responsible_person=True,
        )
        algo = AlgorithmRecord(
            name="Test",
            status=AlgorithmStatus.PRODUCTION,
            responsible_person="",
        )
        with pytest.raises(ValueError, match="Responsible person required"):
            registry.register(algo)

    def test_get_algorithm(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        algo = AlgorithmRecord(name="Test Algorithm")
        algo_id = registry.register(algo)

        retrieved = registry.get(algo_id)
        assert retrieved is not None
        assert retrieved.name == "Test Algorithm"

    def test_get_algorithm_not_found(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        retrieved = registry.get("non-existent-id")
        assert retrieved is None

    def test_get_by_name(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        algo = AlgorithmRecord(name="Unique Name")
        registry.register(algo)

        retrieved = registry.get_by_name("Unique Name")
        assert retrieved is not None
        assert retrieved.name == "Unique Name"

    def test_get_by_name_not_found(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        retrieved = registry.get_by_name("Non Existent")
        assert retrieved is None

    def test_get_all(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        registry.register(AlgorithmRecord(name="Algo 1"))
        registry.register(AlgorithmRecord(name="Algo 2"))
        registry.register(AlgorithmRecord(name="Algo 3"))

        all_algos = registry.get_all()
        assert len(all_algos) == 3

    def test_get_active_algorithms(self, temp_registry_path):
        registry = AlgorithmRegistry(
            registry_path=temp_registry_path,
            require_responsible_person=False,
        )
        registry.register(AlgorithmRecord(name="Dev", status=AlgorithmStatus.DEVELOPMENT))
        registry.register(AlgorithmRecord(name="Prod", status=AlgorithmStatus.PRODUCTION))
        registry.register(AlgorithmRecord(name="Paper", status=AlgorithmStatus.PAPER_TRADING))

        active = registry.get_active_algorithms()
        assert len(active) == 2
        names = [a.name for a in active]
        assert "Prod" in names
        assert "Paper" in names

    def test_get_by_status(self, temp_registry_path):
        registry = AlgorithmRegistry(
            registry_path=temp_registry_path,
            require_responsible_person=False,
        )
        registry.register(AlgorithmRecord(name="Dev1", status=AlgorithmStatus.DEVELOPMENT))
        registry.register(AlgorithmRecord(name="Dev2", status=AlgorithmStatus.DEVELOPMENT))
        registry.register(AlgorithmRecord(name="Prod", status=AlgorithmStatus.PRODUCTION))

        dev_algos = registry.get_by_status(AlgorithmStatus.DEVELOPMENT)
        assert len(dev_algos) == 2

    def test_get_by_type(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        registry.register(AlgorithmRecord(name="Exec", algorithm_type=AlgorithmType.EXECUTION))
        registry.register(AlgorithmRecord(name="Decision", algorithm_type=AlgorithmType.DECISION))
        registry.register(AlgorithmRecord(name="HFT", algorithm_type=AlgorithmType.HIGH_FREQUENCY))

        exec_algos = registry.get_by_type(AlgorithmType.EXECUTION)
        assert len(exec_algos) == 1
        assert exec_algos[0].name == "Exec"

    def test_update_algorithm(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        algo = AlgorithmRecord(name="Test", version="1.0.0")
        algo_id = registry.register(algo)

        # Update
        algo.description = "Updated description"
        registry.update(algo, reason="Added description")

        retrieved = registry.get(algo_id)
        assert retrieved.description == "Updated description"

    def test_update_algorithm_not_found(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        algo = AlgorithmRecord(algorithm_id="non-existent", name="Test")
        with pytest.raises(ValueError, match="not found"):
            registry.update(algo)

    def test_update_increments_version(self, temp_registry_path):
        registry = AlgorithmRegistry(
            registry_path=temp_registry_path,
            version_on_modification=True,
        )
        algo = AlgorithmRecord(name="Test", version="1.0.0")
        registry.register(algo)

        # Modify something that changes the hash
        algo.max_order_size = 1000
        registry.update(algo)

        retrieved = registry.get(algo.algorithm_id)
        assert retrieved.version == "1.0.1"

    def test_retire_algorithm(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        algo = AlgorithmRecord(name="Test")
        algo_id = registry.register(algo)

        registry.retire(algo_id, reason="End of life")

        retrieved = registry.get(algo_id)
        assert retrieved.status == AlgorithmStatus.RETIRED
        assert retrieved.retirement_date is not None

    def test_retire_algorithm_not_found(self, temp_registry_path):
        registry = AlgorithmRegistry(registry_path=temp_registry_path)
        with pytest.raises(ValueError, match="not found"):
            registry.retire("non-existent")

    def test_generate_nca_report(self, temp_registry_path):
        registry = AlgorithmRegistry(
            registry_path=temp_registry_path,
            firm_name="Test Firm",
            nca_jurisdiction="FCA",
            require_responsible_person=False,
        )
        registry.register(AlgorithmRecord(name="Algo1", status=AlgorithmStatus.PRODUCTION))
        registry.register(AlgorithmRecord(name="Algo2", status=AlgorithmStatus.PRODUCTION))

        report = registry.generate_nca_report()
        assert report["firm_name"] == "Test Firm"
        assert report["nca_jurisdiction"] == "FCA"
        assert report["summary"]["active_algorithms"] == 2
        assert len(report["algorithms"]) == 2

    def test_generate_annual_self_assessment(self, temp_registry_path):
        registry = AlgorithmRegistry(
            registry_path=temp_registry_path,
            firm_name="Test Firm",
            require_responsible_person=False,
        )
        registry.register(AlgorithmRecord(
            name="Algo1",
            status=AlgorithmStatus.PRODUCTION,
            responsible_person="John",
            risk_controls=[AlgorithmRiskControl(control_type="test", description="test")],
        ))

        report = registry.generate_annual_self_assessment()
        assert "assessment_type" in report
        assert "metrics" in report
        assert "compliance_status" in report

    def test_get_statistics(self, temp_registry_path):
        registry = AlgorithmRegistry(
            registry_path=temp_registry_path,
            require_responsible_person=False,
        )
        registry.register(AlgorithmRecord(name="Algo1", status=AlgorithmStatus.PRODUCTION))
        registry.register(AlgorithmRecord(name="Algo2", status=AlgorithmStatus.DEVELOPMENT))

        stats = registry.get_statistics()
        assert stats["total_algorithms"] == 2
        assert "by_status" in stats
        assert "by_type" in stats

    def test_persistence(self, temp_registry_path):
        # Create registry and add algorithms
        registry1 = AlgorithmRegistry(registry_path=temp_registry_path)
        registry1.register(AlgorithmRecord(name="Persistent Algo"))

        # Create new registry instance and verify data loaded
        registry2 = AlgorithmRegistry(registry_path=temp_registry_path)
        all_algos = registry2.get_all()
        assert len(all_algos) == 1
        assert all_algos[0].name == "Persistent Algo"


class TestCreateAlgorithmRegistry:
    """Tests for create_algorithm_registry factory function."""

    def test_create_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"registry_path": os.path.join(tmpdir, "test_registry.json")}
            registry = create_algorithm_registry(config)
            assert isinstance(registry, AlgorithmRegistry)

    def test_create_with_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "registry_path": os.path.join(tmpdir, "test_registry.json"),
                "firm_name": "Test Firm",
                "nca_jurisdiction": "FCA",
                "auto_register": False,
            }
            registry = create_algorithm_registry(config)
            assert registry._firm_name == "Test Firm"
            assert registry._auto_register is False


class TestGetDefaultAlgorithmTypes:
    """Tests for get_default_algorithm_types function."""

    def test_returns_list(self):
        types = get_default_algorithm_types()
        assert isinstance(types, list)
        assert len(types) > 0

    def test_contains_required_fields(self):
        types = get_default_algorithm_types()
        for t in types:
            assert "type" in t
            assert "description" in t
            assert "examples" in t

    def test_contains_execution_type(self):
        types = get_default_algorithm_types()
        type_values = [t["type"] for t in types]
        assert "execution" in type_values
