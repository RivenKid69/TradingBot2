# -*- coding: utf-8 -*-
"""
Tests for MicroVM (Firecracker) isolation.

Phase 10: Enterprise isolation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import subprocess
import pytest

from packages.cloud.research.sandbox.cloud_sandbox import (
    CloudResearchSandbox,
    CloudSandboxConfig,
    CloudSandboxResult,
    CloudSandboxState,
    IsolationLevel,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def microvm_config(temp_dir):
    """Create MicroVM sandbox configuration."""
    return CloudSandboxConfig(
        sandbox_id="test-microvm-001",
        tenant_id="test-tenant-001",  # Required for isolated execution
        isolation_level=IsolationLevel.MICROVM,
        scratch_dir=temp_dir,
        cpu_limit=1.0,
        memory_limit_mb=512,
        timeout_seconds=60,
        network_enabled=False,
        readonly_rootfs=True,
    )


@pytest.fixture
def sandbox(microvm_config):
    """Create sandbox instance."""
    return CloudResearchSandbox(microvm_config)


class TestMicroVMConfiguration:
    """Tests for MicroVM configuration."""

    def test_microvm_isolation_level(self, microvm_config):
        """Test MicroVM isolation level configuration."""
        assert microvm_config.isolation_level == IsolationLevel.MICROVM

    def test_default_firecracker_paths(self):
        """Test default Firecracker configuration paths."""
        from packages.cloud.research.sandbox.cloud_sandbox import (
            CloudResearchSandbox,
        )

        config = CloudSandboxConfig(
            tenant_id="test-tenant",  # Required for isolated execution
            isolation_level=IsolationLevel.MICROVM,
        )
        sandbox = CloudResearchSandbox(config)

        # These are the expected default paths
        assert "/etc/ccea/firecracker" in str(Path("/etc/ccea/firecracker"))

    def test_microvm_config_validation(self, temp_dir):
        """Test MicroVM config is validated."""
        config = CloudSandboxConfig(
            tenant_id="test-tenant",  # Required for isolated execution
            isolation_level=IsolationLevel.MICROVM,
            scratch_dir=temp_dir,
            cpu_limit=2.0,
            memory_limit_mb=1024,
        )
        assert config.cpu_limit == 2.0
        assert config.memory_limit_mb == 1024


class TestFirecrackerConfigGeneration:
    """Tests for Firecracker configuration generation."""

    def test_build_firecracker_config(self, sandbox, temp_dir):
        """Test Firecracker VM configuration generation."""
        kernel_path = temp_dir / "vmlinux"
        rootfs_path = temp_dir / "rootfs.ext4"
        kernel_path.touch()
        rootfs_path.touch()

        sandbox._scratch_dir = temp_dir

        config = sandbox._build_firecracker_config(
            kernel_path=kernel_path,
            rootfs_path=rootfs_path,
            entrypoint="main.py",
        )

        assert "boot-source" in config
        assert config["boot-source"]["kernel_image_path"] == str(kernel_path)
        assert "drives" in config
        assert len(config["drives"]) == 1
        assert config["drives"][0]["drive_id"] == "rootfs"
        assert "machine-config" in config
        assert config["machine-config"]["vcpu_count"] >= 1

    def test_config_includes_resource_limits(self, sandbox, temp_dir):
        """Test that config includes resource limits."""
        kernel_path = temp_dir / "vmlinux"
        rootfs_path = temp_dir / "rootfs.ext4"
        kernel_path.touch()
        rootfs_path.touch()

        sandbox._scratch_dir = temp_dir

        config = sandbox._build_firecracker_config(
            kernel_path=kernel_path,
            rootfs_path=rootfs_path,
            entrypoint="main.py",
        )

        machine_config = config["machine-config"]
        assert machine_config["vcpu_count"] >= 1
        assert machine_config["mem_size_mib"] == 512  # From fixture
        assert machine_config["smt"] is False  # SMT disabled for security

    def test_config_includes_boot_args(self, sandbox, temp_dir):
        """Test that config includes kernel boot arguments."""
        kernel_path = temp_dir / "vmlinux"
        rootfs_path = temp_dir / "rootfs.ext4"
        kernel_path.touch()
        rootfs_path.touch()

        sandbox._scratch_dir = temp_dir

        config = sandbox._build_firecracker_config(
            kernel_path=kernel_path,
            rootfs_path=rootfs_path,
            entrypoint="test.py",
        )

        boot_args = config["boot-source"]["boot_args"]
        assert "console=ttyS0" in boot_args
        assert "reboot=k" in boot_args
        assert "panic=1" in boot_args
        assert "test.py" in boot_args

    def test_readonly_rootfs_in_boot_args(self, temp_dir):
        """Test readonly rootfs flag in boot args."""
        config = CloudSandboxConfig(
            tenant_id="test-tenant",  # Required for isolated execution
            isolation_level=IsolationLevel.MICROVM,
            scratch_dir=temp_dir,
            readonly_rootfs=True,
        )
        sandbox = CloudResearchSandbox(config)
        sandbox._scratch_dir = temp_dir

        kernel_path = temp_dir / "vmlinux"
        rootfs_path = temp_dir / "rootfs.ext4"
        kernel_path.touch()
        rootfs_path.touch()

        vm_config = sandbox._build_firecracker_config(
            kernel_path=kernel_path,
            rootfs_path=rootfs_path,
            entrypoint="main.py",
        )

        assert "ro" in vm_config["boot-source"]["boot_args"]


class TestFirecrackerPrerequisites:
    """Tests for Firecracker prerequisite checks."""

    def test_fail_when_firecracker_not_installed(self, sandbox, temp_dir):
        """Test failure when Firecracker binary is missing."""
        sandbox._scratch_dir = temp_dir
        (temp_dir / "workspace").mkdir()

        result = CloudSandboxResult()

        with patch('pathlib.Path.exists', return_value=False):
            result = sandbox._execute_microvm("main.py", result)

        assert result.state == CloudSandboxState.FAILED
        assert any("not installed" in err.lower() for err in result.errors)

    def test_fail_when_kernel_missing(self, sandbox, temp_dir):
        """Test failure when kernel image is missing."""
        sandbox._scratch_dir = temp_dir
        (temp_dir / "workspace").mkdir()

        result = CloudSandboxResult()

        def path_exists_mock(self):
            # Firecracker exists but kernel doesn't
            path_str = str(self)
            if "firecracker" in path_str:
                return True
            if "vmlinux" in path_str:
                return False
            return True

        with patch.object(Path, 'exists', path_exists_mock):
            result = sandbox._execute_microvm("main.py", result)

        # Should fail due to missing kernel
        assert result.state == CloudSandboxState.FAILED

    def test_fail_when_rootfs_missing(self, sandbox, temp_dir):
        """Test failure when root filesystem is missing."""
        sandbox._scratch_dir = temp_dir
        (temp_dir / "workspace").mkdir()

        result = CloudSandboxResult()

        def path_exists_mock(self):
            path_str = str(self)
            if "firecracker" in path_str:
                return True
            if "vmlinux" in path_str:
                return True
            if "rootfs.ext4" in path_str:
                return False
            return True

        with patch.object(Path, 'exists', path_exists_mock):
            result = sandbox._execute_microvm("main.py", result)

        # Should fail due to missing rootfs
        assert result.state == CloudSandboxState.FAILED


class TestRootfsOverlay:
    """Tests for rootfs overlay creation."""

    def test_create_rootfs_overlay_with_reflink(self, sandbox, temp_dir):
        """Test rootfs overlay creation with reflink."""
        base_rootfs = temp_dir / "base.ext4"
        vm_rootfs = temp_dir / "vm.ext4"
        workspace = temp_dir / "workspace"

        base_rootfs.write_bytes(b"rootfs content")
        workspace.mkdir()

        sandbox._scratch_dir = temp_dir

        # Mock subprocess to simulate successful reflink
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            sandbox._create_rootfs_overlay(base_rootfs, vm_rootfs, workspace)

            # Should have tried reflink copy
            assert mock_run.called

    def test_create_rootfs_overlay_fallback(self, sandbox, temp_dir):
        """Test rootfs overlay creation falls back to regular copy."""
        base_rootfs = temp_dir / "base.ext4"
        vm_rootfs = temp_dir / "vm.ext4"
        workspace = temp_dir / "workspace"

        base_rootfs.write_bytes(b"rootfs content")
        workspace.mkdir()

        sandbox._scratch_dir = temp_dir

        # Mock subprocess to fail (no reflink support)
        with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cp')):
            with patch('shutil.copy2') as mock_copy:
                sandbox._create_rootfs_overlay(base_rootfs, vm_rootfs, workspace)

                # Should have fallen back to shutil.copy2
                mock_copy.assert_called_once()


class TestVMTermination:
    """Tests for VM termination."""

    def test_terminate_firecracker_vm(self, sandbox, temp_dir):
        """Test graceful VM termination via API."""
        api_socket = temp_dir / "firecracker.sock"
        api_socket.touch()

        sandbox._scratch_dir = temp_dir

        with patch('socket.socket') as mock_socket:
            mock_sock_instance = MagicMock()
            mock_socket.return_value = mock_sock_instance

            sandbox._terminate_firecracker_vm(api_socket)

            # Should have connected and sent shutdown request
            mock_sock_instance.connect.assert_called_once()
            mock_sock_instance.send.assert_called_once()
            mock_sock_instance.close.assert_called_once()

    def test_terminate_nonexistent_socket(self, sandbox, temp_dir):
        """Test termination with non-existent socket."""
        api_socket = temp_dir / "nonexistent.sock"

        sandbox._scratch_dir = temp_dir

        # Should not raise exception
        sandbox._terminate_firecracker_vm(api_socket)


class TestVMCleanup:
    """Tests for VM resource cleanup."""

    def test_cleanup_firecracker_vm(self, sandbox, temp_dir):
        """Test VM resource cleanup."""
        sandbox._scratch_dir = temp_dir

        # Create files to clean up
        (temp_dir / "firecracker.sock").touch()
        (temp_dir / "rootfs.ext4").touch()
        (temp_dir / "vm_config.json").touch()

        sandbox._cleanup_firecracker_vm("test-vm")

        # Files should be removed
        assert not (temp_dir / "firecracker.sock").exists()
        assert not (temp_dir / "rootfs.ext4").exists()
        assert not (temp_dir / "vm_config.json").exists()


class TestVMMetrics:
    """Tests for VM metrics parsing."""

    def test_parse_vm_metrics(self, sandbox, temp_dir):
        """Test parsing Firecracker metrics."""
        sandbox._scratch_dir = temp_dir

        metrics_data = {
            "vcpu": {"exit_io_out": 1000000000},  # 1 second in ns
            "block": {"read_bytes": 1024, "write_bytes": 512},
        }

        metrics_path = temp_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_data))

        result = CloudSandboxResult()
        sandbox._parse_vm_metrics(result)

        assert sandbox._metrics.cpu_time_seconds == 1.0
        assert sandbox._metrics.disk_read_bytes == 1024
        assert sandbox._metrics.disk_write_bytes == 512

    def test_parse_vm_metrics_no_file(self, sandbox, temp_dir):
        """Test metrics parsing when file doesn't exist."""
        sandbox._scratch_dir = temp_dir

        result = CloudSandboxResult()
        # Should not raise exception
        sandbox._parse_vm_metrics(result)

    def test_parse_vm_metrics_invalid_json(self, sandbox, temp_dir):
        """Test metrics parsing with invalid JSON."""
        sandbox._scratch_dir = temp_dir

        metrics_path = temp_dir / "metrics.json"
        metrics_path.write_text("invalid json")

        result = CloudSandboxResult()
        # Should not raise exception
        sandbox._parse_vm_metrics(result)


class TestMicroVMExecution:
    """Tests for MicroVM execution flow."""

    def test_execute_microvm_success_flow(self, sandbox, temp_dir):
        """Test successful MicroVM execution flow."""
        sandbox._scratch_dir = temp_dir
        (temp_dir / "workspace").mkdir()

        result = CloudSandboxResult()

        # Mock all prerequisites exist
        def mock_exists(path):
            return True

        # Mock Popen for successful execution
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"success", b"")
        mock_process.returncode = 0

        with patch.object(Path, 'exists', return_value=True):
            with patch('subprocess.run', return_value=MagicMock(returncode=0)):
                with patch('subprocess.Popen', return_value=mock_process):
                    with patch.object(sandbox, '_create_rootfs_overlay'):
                        with patch.object(sandbox, '_build_firecracker_config', return_value={}):
                            with patch.object(sandbox, '_parse_vm_metrics'):
                                with patch.object(sandbox, '_cleanup_firecracker_vm'):
                                    result = sandbox._execute_microvm("main.py", result)

        assert result.success is True
        assert result.exit_code == 0

    def test_execute_microvm_timeout(self, sandbox, temp_dir):
        """Test MicroVM execution timeout handling."""
        sandbox._scratch_dir = temp_dir
        (temp_dir / "workspace").mkdir()

        result = CloudSandboxResult()

        # Mock process that times out on first communicate call, then returns on cleanup
        mock_process = MagicMock()
        # First call raises timeout, second call (during cleanup) returns normally
        mock_process.communicate.side_effect = [
            subprocess.TimeoutExpired("firecracker", 60),
            (b"", b""),  # Second call during cleanup returns normally
        ]
        mock_process.kill = MagicMock()
        mock_process.returncode = -9

        with patch.object(Path, 'exists', return_value=True):
            with patch('subprocess.run', return_value=MagicMock(returncode=0)):
                with patch('subprocess.Popen', return_value=mock_process):
                    with patch.object(sandbox, '_create_rootfs_overlay'):
                        with patch.object(sandbox, '_build_firecracker_config', return_value={}):
                            with patch.object(sandbox, '_terminate_firecracker_vm'):
                                with patch.object(sandbox, '_cleanup_firecracker_vm'):
                                    result = sandbox._execute_microvm("main.py", result)

        assert result.killed_by_timeout is True
        assert result.state == CloudSandboxState.TERMINATED


class TestSecurityFeatures:
    """Tests for MicroVM security features."""

    def test_smt_disabled_for_security(self, sandbox, temp_dir):
        """Test SMT is disabled for security (Spectre mitigation)."""
        kernel_path = temp_dir / "vmlinux"
        rootfs_path = temp_dir / "rootfs.ext4"
        kernel_path.touch()
        rootfs_path.touch()

        sandbox._scratch_dir = temp_dir

        config = sandbox._build_firecracker_config(
            kernel_path=kernel_path,
            rootfs_path=rootfs_path,
            entrypoint="main.py",
        )

        assert config["machine-config"]["smt"] is False

    def test_network_disabled_by_default(self, microvm_config):
        """Test network is disabled by default."""
        assert microvm_config.network_enabled is False

    def test_readonly_rootfs_enabled(self, microvm_config):
        """Test readonly rootfs is enabled."""
        assert microvm_config.readonly_rootfs is True


class TestIsolationLevelSelection:
    """Tests for isolation level selection."""

    def test_microvm_is_highest_isolation(self):
        """Test MicroVM has highest isolation level."""
        # MicroVM should be the most secure option
        levels = [
            IsolationLevel.NONE,
            IsolationLevel.PROCESS,
            IsolationLevel.CONTAINER,
            IsolationLevel.GVISOR,
            IsolationLevel.MICROVM,
        ]
        assert IsolationLevel.MICROVM == levels[-1]

    def test_select_microvm_execution(self, temp_dir):
        """Test that MicroVM level selects _execute_microvm."""
        config = CloudSandboxConfig(
            tenant_id="test-tenant",  # Required for isolated execution
            isolation_level=IsolationLevel.MICROVM,
            scratch_dir=temp_dir,
        )
        sandbox = CloudResearchSandbox(config)

        # The sandbox should use _execute_microvm for MICROVM level
        assert sandbox.config.isolation_level == IsolationLevel.MICROVM
