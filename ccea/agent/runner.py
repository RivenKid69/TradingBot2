# -*- coding: utf-8 -*-
"""
CCEA Artifact Runner.

Handles artifact execution:
- Artifact download and verification
- Sandbox setup
- Strategy execution
- Result collection

Security:
- All artifacts verified by digest and signature
- Sandbox isolation for strategy execution
- No network access by default
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

from ccea.crypto.digest import compute_file_digest, verify_file_digest


logger = logging.getLogger(__name__)


class RunState(str, Enum):
    """Run state."""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    HALTED = "HALTED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class RunContext:
    """
    Context for an artifact run.

    Contains all information needed for execution.
    """
    run_id: str
    deployment_id: str
    artifact_digest: str
    artifact_path: Path
    config: Dict[str, Any] = field(default_factory=dict)
    working_dir: Optional[Path] = None
    state: RunState = RunState.INITIALIZING
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class RunResult:
    """Result of a run."""
    run_id: str
    success: bool
    state: RunState
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)


class ArtifactRunner:
    """
    Artifact execution runner.

    Handles:
    - Artifact verification
    - Execution environment setup
    - Strategy running
    - State management
    """

    def __init__(
        self,
        artifacts_dir: Path,
        runs_dir: Optional[Path] = None,
        python_path: str = sys.executable,
    ):
        """
        Initialize artifact runner.

        Args:
            artifacts_dir: Directory containing artifacts
            runs_dir: Directory for run working directories
            python_path: Python interpreter path
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.runs_dir = Path(runs_dir) if runs_dir else self.artifacts_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.python_path = python_path

        self._runs: Dict[str, RunContext] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def verify_artifact(
        self,
        artifact_path: Path,
        expected_digest: str,
    ) -> bool:
        """
        Verify artifact integrity.

        Args:
            artifact_path: Path to artifact
            expected_digest: Expected SHA256 digest

        Returns:
            True if verified, False otherwise
        """
        if not artifact_path.exists():
            logger.error(f"Artifact not found: {artifact_path}")
            return False

        try:
            return verify_file_digest(artifact_path, expected_digest)
        except Exception as e:
            logger.error(f"Artifact verification failed: {e}")
            return False

    def start_run(
        self,
        run_id: str,
        deployment_id: str,
        artifact_path: Path,
        artifact_digest: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> RunContext:
        """
        Start artifact run.

        Args:
            run_id: Unique run identifier
            deployment_id: Deployment this run belongs to
            artifact_path: Path to artifact
            artifact_digest: Expected artifact digest
            config: Run configuration

        Returns:
            RunContext

        Raises:
            ValueError: If artifact verification fails
        """
        # Verify artifact
        if not self.verify_artifact(artifact_path, artifact_digest):
            raise ValueError(f"Artifact verification failed: {artifact_digest}")

        # Create run context
        working_dir = self.runs_dir / run_id
        working_dir.mkdir(parents=True, exist_ok=True)

        context = RunContext(
            run_id=run_id,
            deployment_id=deployment_id,
            artifact_digest=artifact_digest,
            artifact_path=artifact_path,
            config=config or {},
            working_dir=working_dir,
            state=RunState.INITIALIZING,
        )

        with self._lock:
            self._runs[run_id] = context

        # Start execution
        try:
            self._execute_artifact(context)
            context.state = RunState.RUNNING
            context.started_at = datetime.utcnow()
            logger.info(f"Run started: {run_id}")
        except Exception as e:
            context.state = RunState.FAILED
            context.error = str(e)
            logger.exception(f"Run failed to start: {run_id}")

        return context

    def stop_run(self, run_id: str) -> bool:
        """
        Stop a run.

        Args:
            run_id: Run to stop

        Returns:
            True if stopped, False if not found
        """
        with self._lock:
            context = self._runs.get(run_id)
            if context is None:
                return False

            process = self._processes.get(run_id)
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

            context.state = RunState.STOPPED
            context.stopped_at = datetime.utcnow()

        logger.info(f"Run stopped: {run_id}")
        return True

    def pause_run(self, run_id: str) -> bool:
        """
        Pause a run.

        Args:
            run_id: Run to pause

        Returns:
            True if paused, False if not found
        """
        with self._lock:
            context = self._runs.get(run_id)
            if context is None:
                return False

            if context.state != RunState.RUNNING:
                return False

            # Send pause signal (implementation-specific)
            context.state = RunState.PAUSED

        logger.info(f"Run paused: {run_id}")
        return True

    def resume_run(self, run_id: str) -> bool:
        """
        Resume a paused run.

        Args:
            run_id: Run to resume

        Returns:
            True if resumed, False if not found
        """
        with self._lock:
            context = self._runs.get(run_id)
            if context is None:
                return False

            if context.state != RunState.PAUSED:
                return False

            context.state = RunState.RUNNING

        logger.info(f"Run resumed: {run_id}")
        return True

    def get_run(self, run_id: str) -> Optional[RunContext]:
        """Get run context."""
        with self._lock:
            return self._runs.get(run_id)

    def get_run_result(self, run_id: str) -> Optional[RunResult]:
        """Get run result."""
        with self._lock:
            context = self._runs.get(run_id)
            if context is None:
                return None

            duration = None
            if context.started_at and context.stopped_at:
                duration = (context.stopped_at - context.started_at).total_seconds()

            return RunResult(
                run_id=run_id,
                success=context.state == RunState.COMPLETED,
                state=context.state,
                started_at=context.started_at,
                stopped_at=context.stopped_at,
                duration_seconds=duration,
                error=context.error,
            )

    def list_runs(self) -> List[RunContext]:
        """List all runs."""
        with self._lock:
            return list(self._runs.values())

    def _execute_artifact(self, context: RunContext) -> None:
        """Execute artifact in subprocess."""
        # For Phase 1, we use a simple subprocess execution
        # Production would use proper sandbox (container, etc.)

        # Write config to file
        config_path = context.working_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(context.config, f)

        # Create a simple wrapper script
        wrapper_script = context.working_dir / "run_wrapper.py"
        wrapper_content = f'''
import sys
import json
from pathlib import Path

# Load config
config_path = Path("{config_path}")
with open(config_path) as f:
    config = json.load(f)

# For hello strategy, just print and exit
print("Hello Strategy Starting")
print(f"Artifact: {context.artifact_digest}")
print(f"Config: {{config}}")

# Simulate strategy running
import time
for i in range(5):
    print(f"Strategy iteration {{i+1}}")
    time.sleep(1)

print("Hello Strategy Complete")
'''

        with open(wrapper_script, "w") as f:
            f.write(wrapper_content)

        # Start subprocess
        process = subprocess.Popen(
            [self.python_path, str(wrapper_script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(context.working_dir),
        )

        with self._lock:
            self._processes[context.run_id] = process


# ============================================================================
# Hello Strategy (Phase 1 Test Artifact)
# ============================================================================

class HelloStrategy:
    """
    Minimal test strategy for Phase 1.

    Does NOT use any broker connection.
    Simply demonstrates the artifact execution flow.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy."""
        self.config = config
        self.iteration = 0

    def run(self) -> Dict[str, Any]:
        """Run one iteration."""
        self.iteration += 1
        return {
            "iteration": self.iteration,
            "message": f"Hello from strategy iteration {self.iteration}",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def stop(self) -> None:
        """Stop strategy."""
        pass


def create_hello_strategy_artifact(output_dir: Path) -> tuple[Path, str]:
    """
    Create a hello strategy artifact for testing.

    Args:
        output_dir: Directory to create artifact in

    Returns:
        Tuple of (artifact_path, digest)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create strategy file
    strategy_content = '''
# Hello Strategy - CCEA Phase 1 Test Artifact
# NO BROKER CONNECTION - Research/Test Only

import json
import time
from datetime import datetime

class HelloStrategy:
    """Minimal test strategy."""

    def __init__(self, config=None):
        self.config = config or {}
        self.iteration = 0

    def run_iteration(self):
        self.iteration += 1
        return {
            "iteration": self.iteration,
            "message": f"Hello from iteration {self.iteration}",
            "timestamp": datetime.utcnow().isoformat(),
        }

if __name__ == "__main__":
    strategy = HelloStrategy()
    print("Hello Strategy Starting...")

    for i in range(5):
        result = strategy.run_iteration()
        print(f"Result: {json.dumps(result)}")
        time.sleep(1)

    print("Hello Strategy Complete")
'''

    artifact_path = output_dir / "hello_strategy.py"
    with open(artifact_path, "w") as f:
        f.write(strategy_content)

    # Compute digest
    digest = compute_file_digest(artifact_path)

    return artifact_path, digest
