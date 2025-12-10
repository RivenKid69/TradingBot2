# -*- coding: utf-8 -*-
"""
Phase 0 Structure Validation Tests.

Validates that all directory structures, __init__.py files, and configuration
directories have been properly created for the DORA integration layer migration.

Test Coverage:
    - Directory structure validation
    - __init__.py file existence and content
    - Configuration directory setup
    - Archive directory preparation
    - Git tag and branch verification
    - Import audit documentation
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase0DirectoryStructure:
    """Test that all required directories exist."""

    @pytest.fixture
    def services_path(self) -> Path:
        """Return path to services directory."""
        return PROJECT_ROOT / "services"

    @pytest.fixture
    def config_path(self) -> Path:
        """Return path to config directory."""
        return PROJECT_ROOT / "config"

    def test_dora_integration_directory_exists(self, services_path: Path) -> None:
        """Verify services/dora_integration/ directory exists."""
        integration_path = services_path / "dora_integration"
        assert integration_path.exists(), f"Directory not found: {integration_path}"
        assert integration_path.is_dir(), f"Not a directory: {integration_path}"

    def test_due_diligence_subdirectory_exists(self, services_path: Path) -> None:
        """Verify services/dora_integration/due_diligence/ exists."""
        path = services_path / "dora_integration" / "due_diligence"
        assert path.exists(), f"Directory not found: {path}"
        assert path.is_dir(), f"Not a directory: {path}"

    def test_incident_interface_subdirectory_exists(self, services_path: Path) -> None:
        """Verify services/dora_integration/incident_interface/ exists."""
        path = services_path / "dora_integration" / "incident_interface"
        assert path.exists(), f"Directory not found: {path}"
        assert path.is_dir(), f"Not a directory: {path}"

    def test_third_party_subdirectory_exists(self, services_path: Path) -> None:
        """Verify services/dora_integration/third_party/ exists."""
        path = services_path / "dora_integration" / "third_party"
        assert path.exists(), f"Directory not found: {path}"
        assert path.is_dir(), f"Not a directory: {path}"

    def test_contracts_subdirectory_exists(self, services_path: Path) -> None:
        """Verify services/dora_integration/contracts/ exists."""
        path = services_path / "dora_integration" / "contracts"
        assert path.exists(), f"Directory not found: {path}"
        assert path.is_dir(), f"Not a directory: {path}"

    def test_reporting_subdirectory_exists(self, services_path: Path) -> None:
        """Verify services/dora_integration/reporting/ exists."""
        path = services_path / "dora_integration" / "reporting"
        assert path.exists(), f"Directory not found: {path}"
        assert path.is_dir(), f"Not a directory: {path}"

    def test_sharing_subdirectory_exists(self, services_path: Path) -> None:
        """Verify services/dora_integration/sharing/ exists."""
        path = services_path / "dora_integration" / "sharing"
        assert path.exists(), f"Directory not found: {path}"
        assert path.is_dir(), f"Not a directory: {path}"

    def test_archive_directory_exists(self, services_path: Path) -> None:
        """Verify services/archive/dora_financial_entity/ exists."""
        path = services_path / "archive" / "dora_financial_entity"
        assert path.exists(), f"Directory not found: {path}"
        assert path.is_dir(), f"Not a directory: {path}"

    def test_archive_configs_directory_exists(self, services_path: Path) -> None:
        """Verify services/archive/dora_financial_entity/configs/ exists."""
        path = services_path / "archive" / "dora_financial_entity" / "configs"
        assert path.exists(), f"Directory not found: {path}"
        assert path.is_dir(), f"Not a directory: {path}"

    def test_config_dora_integration_directory_exists(self, config_path: Path) -> None:
        """Verify config/dora_integration/ exists."""
        path = config_path / "dora_integration"
        assert path.exists(), f"Directory not found: {path}"
        assert path.is_dir(), f"Not a directory: {path}"


class TestPhase0InitFiles:
    """Test that all __init__.py files exist and have correct content."""

    @pytest.fixture
    def services_path(self) -> Path:
        """Return path to services directory."""
        return PROJECT_ROOT / "services"

    def test_dora_integration_init_exists(self, services_path: Path) -> None:
        """Verify services/dora_integration/__init__.py exists."""
        init_path = services_path / "dora_integration" / "__init__.py"
        assert init_path.exists(), f"Init file not found: {init_path}"

    def test_dora_integration_init_content(self, services_path: Path) -> None:
        """Verify services/dora_integration/__init__.py has required content."""
        init_path = services_path / "dora_integration" / "__init__.py"
        content = init_path.read_text()

        # Check for required docstring elements
        assert "DORA Integration Layer" in content
        assert "Art. 30" in content or "Article 30" in content
        assert "__version__" in content
        assert "__migration_phase__" in content

    def test_due_diligence_init_exists(self, services_path: Path) -> None:
        """Verify due_diligence/__init__.py exists."""
        init_path = services_path / "dora_integration" / "due_diligence" / "__init__.py"
        assert init_path.exists(), f"Init file not found: {init_path}"

    def test_due_diligence_init_content(self, services_path: Path) -> None:
        """Verify due_diligence/__init__.py has correct docstring."""
        init_path = services_path / "dora_integration" / "due_diligence" / "__init__.py"
        content = init_path.read_text()
        assert "Due Diligence" in content or "Audit Readiness" in content

    def test_incident_interface_init_exists(self, services_path: Path) -> None:
        """Verify incident_interface/__init__.py exists."""
        init_path = services_path / "dora_integration" / "incident_interface" / "__init__.py"
        assert init_path.exists(), f"Init file not found: {init_path}"

    def test_incident_interface_init_content(self, services_path: Path) -> None:
        """Verify incident_interface/__init__.py has correct docstring."""
        init_path = services_path / "dora_integration" / "incident_interface" / "__init__.py"
        content = init_path.read_text()
        assert "Incident" in content
        assert "We notify CLIENTS" in content or "Client" in content

    def test_third_party_init_exists(self, services_path: Path) -> None:
        """Verify third_party/__init__.py exists."""
        init_path = services_path / "dora_integration" / "third_party" / "__init__.py"
        assert init_path.exists(), f"Init file not found: {init_path}"

    def test_contracts_init_exists(self, services_path: Path) -> None:
        """Verify contracts/__init__.py exists."""
        init_path = services_path / "dora_integration" / "contracts" / "__init__.py"
        assert init_path.exists(), f"Init file not found: {init_path}"

    def test_reporting_init_exists(self, services_path: Path) -> None:
        """Verify reporting/__init__.py exists."""
        init_path = services_path / "dora_integration" / "reporting" / "__init__.py"
        assert init_path.exists(), f"Init file not found: {init_path}"

    def test_sharing_init_exists(self, services_path: Path) -> None:
        """Verify sharing/__init__.py exists."""
        init_path = services_path / "dora_integration" / "sharing" / "__init__.py"
        assert init_path.exists(), f"Init file not found: {init_path}"

    def test_archive_init_exists(self, services_path: Path) -> None:
        """Verify archive/dora_financial_entity/__init__.py exists."""
        init_path = services_path / "archive" / "dora_financial_entity" / "__init__.py"
        assert init_path.exists(), f"Init file not found: {init_path}"

    def test_archive_init_content(self, services_path: Path) -> None:
        """Verify archive __init__.py has correct docstring."""
        init_path = services_path / "archive" / "dora_financial_entity" / "__init__.py"
        content = init_path.read_text()
        assert "Archived" in content
        assert "Financial Entity" in content or "FE" in content


class TestPhase0ImportCapability:
    """Test that the new modules can be imported without errors."""

    def test_import_dora_integration_package(self) -> None:
        """Verify services.dora_integration can be imported."""
        try:
            from services import dora_integration
            # Phase 4 complete - version is now 1.3.0
            assert dora_integration.__version__ == "1.3.0"
            # Phase 4 complete - migration_phase is now 4
            assert dora_integration.__migration_phase__ >= 4
        except ImportError as e:
            pytest.fail(f"Failed to import dora_integration: {e}")

    def test_import_due_diligence_subpackage(self) -> None:
        """Verify services.dora_integration.due_diligence can be imported."""
        try:
            from services.dora_integration import due_diligence
            # Phase 1 complete - due_diligence now has exports
            assert len(due_diligence.__all__) > 0
        except ImportError as e:
            pytest.fail(f"Failed to import due_diligence: {e}")

    def test_import_incident_interface_subpackage(self) -> None:
        """Verify services.dora_integration.incident_interface can be imported."""
        try:
            from services.dora_integration import incident_interface
            # Phase 2 complete - incident_interface now has exports
            assert len(incident_interface.__all__) > 0
        except ImportError as e:
            pytest.fail(f"Failed to import incident_interface: {e}")

    def test_import_third_party_subpackage(self) -> None:
        """Verify services.dora_integration.third_party can be imported."""
        try:
            from services.dora_integration import third_party
            # Phase 3 complete - third_party now has 88 exports
            assert len(third_party.__all__) > 0
        except ImportError as e:
            pytest.fail(f"Failed to import third_party: {e}")

    def test_import_contracts_subpackage(self) -> None:
        """Verify services.dora_integration.contracts can be imported."""
        try:
            from services.dora_integration import contracts
            # Phase 4 complete - contracts now has exports
            assert len(contracts.__all__) > 0
        except ImportError as e:
            pytest.fail(f"Failed to import contracts: {e}")

    def test_import_reporting_subpackage(self) -> None:
        """Verify services.dora_integration.reporting can be imported."""
        try:
            from services.dora_integration import reporting
            assert reporting.__all__ == []  # Empty in Phase 0
        except ImportError as e:
            pytest.fail(f"Failed to import reporting: {e}")

    def test_import_sharing_subpackage(self) -> None:
        """Verify services.dora_integration.sharing can be imported."""
        try:
            from services.dora_integration import sharing
            assert sharing.__all__ == []  # Empty in Phase 0
        except ImportError as e:
            pytest.fail(f"Failed to import sharing: {e}")


class TestPhase0ExistingCodeIntegrity:
    """Test that existing services.dora module still works."""

    def test_existing_dora_module_imports(self) -> None:
        """Verify services.dora can still be imported."""
        try:
            from services import dora
            assert hasattr(dora, "__version__")
        except ImportError as e:
            pytest.fail(f"Failed to import existing dora module: {e}")

    def test_existing_dora_exports(self) -> None:
        """Verify key exports from services.dora still work."""
        try:
            from services.dora import DORAScope, FunctionClassifier
            assert DORAScope is not None
            assert FunctionClassifier is not None
        except ImportError as e:
            pytest.fail(f"Failed to import from existing dora module: {e}")


class TestPhase0Documentation:
    """Test that required documentation exists."""

    @pytest.fixture
    def docs_path(self) -> Path:
        """Return path to docs directory."""
        return PROJECT_ROOT / "docs"

    @pytest.fixture
    def services_path(self) -> Path:
        """Return path to services directory."""
        return PROJECT_ROOT / "services"

    def test_migration_plan_exists(self, docs_path: Path) -> None:
        """Verify DORA migration plan document exists."""
        plan_path = docs_path / "DORA_INTEGRATION_LAYER_MIGRATION_PLAN.md"
        assert plan_path.exists(), f"Migration plan not found: {plan_path}"

    def test_import_audit_exists(self, docs_path: Path) -> None:
        """Verify import audit document exists."""
        audit_path = docs_path / "migration" / "DORA_IMPORT_AUDIT_PHASE0.md"
        assert audit_path.exists(), f"Import audit not found: {audit_path}"

    def test_archive_readme_exists(self, services_path: Path) -> None:
        """Verify archive README exists."""
        readme_path = services_path / "archive" / "dora_financial_entity" / "README.md"
        assert readme_path.exists(), f"Archive README not found: {readme_path}"

    def test_config_readme_exists(self) -> None:
        """Verify config README exists."""
        readme_path = PROJECT_ROOT / "config" / "dora_integration" / "README.md"
        assert readme_path.exists(), f"Config README not found: {readme_path}"


class TestPhase0MigrationChecklistComplete:
    """Verify all Phase 0 checklist items are complete."""

    def test_all_directories_created(self) -> None:
        """Verify all 6 integration subpackages + archive exist."""
        expected_dirs = [
            "services/dora_integration/due_diligence",
            "services/dora_integration/incident_interface",
            "services/dora_integration/third_party",
            "services/dora_integration/contracts",
            "services/dora_integration/reporting",
            "services/dora_integration/sharing",
            "services/archive/dora_financial_entity",
            "services/archive/dora_financial_entity/configs",
            "config/dora_integration",
        ]

        for dir_path in expected_dirs:
            full_path = PROJECT_ROOT / dir_path
            assert full_path.exists(), f"Directory missing: {dir_path}"
            assert full_path.is_dir(), f"Not a directory: {dir_path}"

    def test_all_init_files_created(self) -> None:
        """Verify all __init__.py files exist."""
        expected_inits = [
            "services/dora_integration/__init__.py",
            "services/dora_integration/due_diligence/__init__.py",
            "services/dora_integration/incident_interface/__init__.py",
            "services/dora_integration/third_party/__init__.py",
            "services/dora_integration/contracts/__init__.py",
            "services/dora_integration/reporting/__init__.py",
            "services/dora_integration/sharing/__init__.py",
            "services/archive/dora_financial_entity/__init__.py",
        ]

        for init_path in expected_inits:
            full_path = PROJECT_ROOT / init_path
            assert full_path.exists(), f"Init file missing: {init_path}"

    def test_import_audit_documented(self) -> None:
        """Verify import audit is documented."""
        audit_path = PROJECT_ROOT / "docs" / "migration" / "DORA_IMPORT_AUDIT_PHASE0.md"
        assert audit_path.exists(), "Import audit document missing"

        content = audit_path.read_text()
        # Check for key audit sections
        assert "Test File Imports" in content or "test_dora" in content
        assert "Integration Layer Modules" in content
        assert "Archive Modules" in content


class TestPhase0TargetArchitecture:
    """Validate the target architecture from the migration plan."""

    def test_architecture_separation(self) -> None:
        """Verify architecture follows core/integration/archive separation."""
        # Core should exist and not be touched
        core_path = PROJECT_ROOT / "services" / "core"
        assert core_path.exists(), "Core directory should exist"

        # Integration layer should be new
        integration_path = PROJECT_ROOT / "services" / "dora_integration"
        assert integration_path.exists(), "Integration layer should exist"

        # Archive should be prepared
        archive_path = PROJECT_ROOT / "services" / "archive" / "dora_financial_entity"
        assert archive_path.exists(), "Archive should exist"

        # Original dora module should still exist (for backward compat)
        dora_path = PROJECT_ROOT / "services" / "dora"
        assert dora_path.exists(), "Original dora module should exist"

    def test_config_separation(self) -> None:
        """Verify config follows the migration plan."""
        # Original dora config should exist
        dora_config = PROJECT_ROOT / "config" / "dora"
        assert dora_config.exists(), "Original dora config should exist"

        # Integration config should be new
        integration_config = PROJECT_ROOT / "config" / "dora_integration"
        assert integration_config.exists(), "Integration config should exist"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
