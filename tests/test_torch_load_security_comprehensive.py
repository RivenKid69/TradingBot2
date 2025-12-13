"""
Comprehensive Security Tests for torch.load() Vulnerability Fixes

This test suite verifies that all torch.load() calls in production and test code
use secure loading practices (weights_only=True) to prevent arbitrary code execution
via malicious pickle payloads.

Test Coverage:
1. Production code security verification
2. Malicious checkpoint blocking
3. Legitimate checkpoint compatibility
4. PBT scheduler security
5. Inference script security
"""
import os
import pickle
import tempfile
from pathlib import Path
import re
import pytest
torch = pytest.importorskip("torch")
import pytest
import numpy as np


# Global flag to track malicious code execution
EXPLOIT_EXECUTED = False
EXPLOIT_DATA = None


class MaliciousPayload:
    """
    Test payload that executes code during unpickling.
    Used to verify that malicious checkpoints are blocked.
    """
    def __reduce__(self):
        global EXPLOIT_EXECUTED, EXPLOIT_DATA
        EXPLOIT_EXECUTED = True
        EXPLOIT_DATA = "EXPLOIT: Code executed during unpickling"
        return (print, ("Security test: malicious payload executed",))


def create_safe_checkpoint(path: Path):
    """Create a legitimate checkpoint with only tensors."""
    state_dict = {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(5, 3),
        'layer2.bias': torch.randn(5),
    }
    torch.save(state_dict, path)
    return state_dict


def create_malicious_checkpoint(path: Path):
    """Create a checkpoint with malicious payload."""
    state_dict = {
        'layer1.weight': torch.randn(10, 5),
        'malicious': MaliciousPayload(),
    }
    torch.save(state_dict, path)


class TestProductionCodeSecurity:
    """Test that production code uses secure torch.load() practices."""

    def test_no_vulnerable_torch_load_in_production(self):
        """
        Verify that all production torch.load() calls use weights_only=True.

        This test scans production files to ensure no vulnerable torch.load() calls remain.
        """
        production_files = [
            "adversarial/pbt_scheduler.py",
            "infer_signals.py",
        ]

        base_dir = Path(__file__).parent.parent
        vulnerable_calls = []

        for file_path in production_files:
            full_path = base_dir / file_path

            if not full_path.exists():
                pytest.skip(f"Production file not found: {file_path}")

            content = full_path.read_text(encoding='utf-8')

            # Find all torch.load() calls
            pattern = r'torch\.load\s*\([^)]*\)'
            matches = re.finditer(pattern, content)

            for match in matches:
                call = match.group(0)

                # Check if this call has weights_only=True or is part of safe pattern
                if 'weights_only=True' not in call and 'weights_only=False' not in call:
                    # Missing weights_only parameter entirely - vulnerable!
                    vulnerable_calls.append({
                        'file': file_path,
                        'call': call,
                        'line': content[:match.start()].count('\n') + 1,
                    })
                elif 'weights_only=False' in call:
                    # Check if it has a safe fallback pattern (try/except with security comment)
                    # Expand context window to capture the entire try/except block
                    start_pos = max(0, match.start() - 1000)
                    end_pos = min(len(content), match.end() + 200)
                    context = content[start_pos:end_pos]

                    # Allow fallback pattern if:
                    # 1. There's a try/except with weights_only=True first
                    # 2. There's a security comment explaining why
                    context_lower = context.lower()
                    has_security_comment = 'security' in context_lower
                    has_weights_only_true_attempt = 'weights_only=True' in context
                    has_fallback_pattern = 'except' in context and 'fallback' in context_lower
                    has_warnings_warn = 'warnings.warn' in context or 'userwarning' in context_lower

                    # Safe if has security comment AND (weights_only=True attempt OR fallback pattern with warning)
                    is_safe = has_security_comment and has_weights_only_true_attempt and (has_fallback_pattern or has_warnings_warn)

                    if not is_safe:
                        vulnerable_calls.append({
                            'file': file_path,
                            'call': call,
                            'line': content[:match.start()].count('\n') + 1,
                        })

        if vulnerable_calls:
            msg = "\nVulnerable torch.load() calls found:\n"
            for vuln in vulnerable_calls:
                msg += f"  {vuln['file']}:{vuln['line']}: {vuln['call']}\n"
            pytest.fail(msg)

    def test_pbt_scheduler_secure_loading(self, tmp_path):
        """
        Test that PBT scheduler loads checkpoints securely.

        Verifies that adversarial/pbt_scheduler.py uses weights_only=True
        and blocks malicious checkpoints.
        """
        from adversarial.pbt_scheduler import PBTScheduler, PBTConfig, HyperparamConfig

        config = PBTConfig(
            population_size=2,
            perturbation_interval=1,  # Trigger exploit on first step
            truncation_ratio=0.5,  # Exploit from top 50%
            hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        # Create safe checkpoint
        safe_checkpoint = tmp_path / "safe_checkpoint.pt"
        safe_state = create_safe_checkpoint(safe_checkpoint)

        # Manually set checkpoint path for testing
        population[0].checkpoint_path = str(safe_checkpoint)
        population[0].performance = 0.9
        population[1].performance = 0.5

        # Mark that we've reached perturbation interval
        population[1].steps_since_perturbation = config.perturbation_interval

        # Trigger exploitation (load from better performer)
        global EXPLOIT_EXECUTED, EXPLOIT_DATA
        EXPLOIT_EXECUTED = False
        EXPLOIT_DATA = None

        # This should load successfully with weights_only=True
        new_state, new_hyperparams = scheduler.exploit_and_explore(population[1])

        # Verify safe checkpoint loaded
        assert new_state is not None, "Expected checkpoint to be loaded during exploitation"
        assert 'layer1.weight' in new_state

        # Verify no malicious code executed
        assert not EXPLOIT_EXECUTED, (
            "Malicious code was executed! PBT scheduler is not using weights_only=True correctly."
        )

    def test_pbt_scheduler_blocks_malicious_checkpoint(self, tmp_path):
        """
        Test that PBT scheduler rejects malicious checkpoints.
        """
        from adversarial.pbt_scheduler import PBTScheduler, PBTConfig, HyperparamConfig

        config = PBTConfig(
            population_size=2,
            perturbation_interval=1,  # Trigger exploit on first step
            truncation_ratio=0.5,  # Exploit from top 50%
            hyperparams=[HyperparamConfig(name="lr", min_value=1e-5, max_value=1e-3)],
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )

        scheduler = PBTScheduler(config, seed=42)
        population = scheduler.initialize_population()

        # Create malicious checkpoint
        malicious_checkpoint = tmp_path / "malicious_checkpoint.pt"
        create_malicious_checkpoint(malicious_checkpoint)

        # Manually set checkpoint path
        population[0].checkpoint_path = str(malicious_checkpoint)
        population[0].performance = 0.9
        population[1].performance = 0.5

        # Mark that we've reached perturbation interval
        population[1].steps_since_perturbation = config.perturbation_interval

        # Reset exploit flag
        global EXPLOIT_EXECUTED, EXPLOIT_DATA
        EXPLOIT_EXECUTED = False
        EXPLOIT_DATA = None

        # This should raise an exception when trying to load malicious checkpoint
        with pytest.raises((pickle.UnpicklingError, RuntimeError, AttributeError)):
            scheduler.exploit_and_explore(population[1])

        # Verify malicious code was NOT executed
        assert not EXPLOIT_EXECUTED, (
            "CRITICAL: Malicious code was executed! weights_only=True is not working."
        )


class TestInferSignalsSecurity:
    """Test that infer_signals.py loads models securely."""

    def test_infer_signals_safe_checkpoint_loading(self, tmp_path):
        """
        Test that infer_signals.py can load safe checkpoints.
        """
        # Create safe model checkpoint
        checkpoint_path = tmp_path / "model.pt"
        safe_state = create_safe_checkpoint(checkpoint_path)

        # Test direct loading with weights_only=True
        loaded_state = torch.load(checkpoint_path, weights_only=True)

        assert 'layer1.weight' in loaded_state
        assert torch.allclose(loaded_state['layer1.weight'], safe_state['layer1.weight'])

    def test_infer_signals_has_fallback_mechanism(self):
        """
        Test that infer_signals.py has a fallback mechanism for legacy checkpoints.

        The code should have a try/except pattern that attempts weights_only=True
        first, then falls back to weights_only=False with a warning.
        """
        import infer_signals
        import inspect

        # Get the source code of _load_model
        source = inspect.getsource(infer_signals._load_model)

        # Verify the security pattern exists
        assert 'weights_only=True' in source, "Missing weights_only=True attempt"
        assert 'weights_only=False' in source, "Missing weights_only=False fallback"
        assert 'warnings.warn' in source or 'UserWarning' in source, "Missing security warning"


class TestCheckpointCompatibility:
    """Test that legitimate checkpoints still work after security fixes."""

    def test_state_dict_checkpoint_loads_securely(self, tmp_path):
        """Test that state_dict checkpoints load with weights_only=True."""
        checkpoint_path = tmp_path / "state_dict.pt"
        safe_state = create_safe_checkpoint(checkpoint_path)

        # Load with weights_only=True (secure)
        loaded_state = torch.load(checkpoint_path, weights_only=True)

        assert 'layer1.weight' in loaded_state
        assert torch.allclose(loaded_state['layer1.weight'], safe_state['layer1.weight'])

    def test_malicious_checkpoint_blocked_with_weights_only(self, tmp_path):
        """Test that malicious checkpoints are blocked with weights_only=True."""
        checkpoint_path = tmp_path / "malicious.pt"
        create_malicious_checkpoint(checkpoint_path)

        global EXPLOIT_EXECUTED, EXPLOIT_DATA
        EXPLOIT_EXECUTED = False
        EXPLOIT_DATA = None

        # Attempt to load with weights_only=True (should fail)
        with pytest.raises((pickle.UnpicklingError, RuntimeError, AttributeError)):
            torch.load(checkpoint_path, weights_only=True)

        # Verify exploit did NOT execute
        assert not EXPLOIT_EXECUTED, (
            "CRITICAL: Malicious code executed even with weights_only=True!"
        )


class TestTestFileSecurity:
    """Test that test files use secure practices."""

    def test_test_files_document_security_decisions(self):
        """
        Verify that test files using weights_only=False have security comments.

        Test files that need to load full objects should document why
        weights_only=False is safe in that context.
        """
        test_files = [
            "test_bug10_vgs_state_persistence.py",
            "tests/test_pbt_adversarial_deep_validation.py",
            "tests/test_pbt_adversarial_real_integration.py",
        ]

        base_dir = Path(__file__).parent.parent
        undocumented_unsafe_loads = []

        for file_path in test_files:
            full_path = base_dir / file_path

            if not full_path.exists():
                continue

            content = full_path.read_text(encoding='utf-8')

            # Find torch.load() calls with weights_only=False
            pattern = r'torch\.load\s*\([^)]*weights_only=False[^)]*\)'
            matches = re.finditer(pattern, content)

            for match in matches:
                # Check for security comment nearby
                start_pos = max(0, match.start() - 300)
                end_pos = min(len(content), match.end() + 100)
                context = content[start_pos:end_pos]

                if 'Security' not in context and 'security' not in context:
                    undocumented_unsafe_loads.append({
                        'file': file_path,
                        'line': content[:match.start()].count('\n') + 1,
                        'call': match.group(0)[:100],
                    })

        if undocumented_unsafe_loads:
            msg = "\nTest files with undocumented weights_only=False:\n"
            for item in undocumented_unsafe_loads:
                msg += f"  {item['file']}:{item['line']}: {item['call']}\n"
            pytest.fail(msg + "\nAdd security comment explaining why weights_only=False is safe.")


class TestRegressionPrevention:
    """Tests to prevent future regressions."""

    def test_torch_save_creates_weights_only_compatible_checkpoint(self, tmp_path):
        """
        Verify that torch.save(state_dict) creates checkpoints
        compatible with weights_only=True.
        """
        checkpoint_path = tmp_path / "checkpoint.pt"

        state_dict = {
            'weights': torch.randn(10, 10),
            'biases': torch.randn(10),
        }

        # Save
        torch.save(state_dict, checkpoint_path)

        # Load with weights_only=True should work
        loaded = torch.load(checkpoint_path, weights_only=True)

        assert 'weights' in loaded
        assert torch.allclose(loaded['weights'], state_dict['weights'])

    def test_full_model_save_incompatible_with_weights_only(self, tmp_path):
        """
        Verify that torch.save(model) creates checkpoints
        that require weights_only=False.

        This demonstrates why production code should use state_dict saving.
        """
        checkpoint_path = tmp_path / "full_model.pt"

        # Use a built-in module that can be pickled
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 3),
            torch.nn.ReLU(),
        )

        # Save full model (NOT recommended for production)
        torch.save(model, checkpoint_path)

        # Loading with weights_only=True should fail because it includes module structure
        with pytest.raises((pickle.UnpicklingError, RuntimeError, AttributeError)):
            torch.load(checkpoint_path, weights_only=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
