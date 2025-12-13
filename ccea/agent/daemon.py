# -*- coding: utf-8 -*-
"""
CCEA Agent Daemon.

Main agent daemon that orchestrates:
- Enrollment with cloud
- Heartbeat sending
- Command polling
- Approval workflow
- Artifact execution
- Telemetry reporting

Security:
- Outbound-only connections
- Local approval for trading-impacting
- All secrets stored locally
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import requests

from ccea.agent.enrollment import AgentEnrollment, EnrollmentState
from ccea.agent.command_handler import CommandHandler, LocalCommandRecord, LocalCommandStatus
from ccea.agent.approval import ApprovalManager, ApprovalRequest
from ccea.agent.runner import ArtifactRunner, RunState
from ccea.crypto.signing import MessageSigner, sign_json
from ccea.crypto.tokens import generate_idempotency_key
from ccea.models.protocol import (
    HeartbeatMessage,
    AgentState,
    AgentHealth,
    PollCommandsMessage,
    CommandAckMessage,
    CommandApprovalMessage,
    CommandResultMessage,
    CommandStatus,
    DeploymentState,
    RunState as ProtocolRunState,
)


logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Agent daemon state."""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass
class AgentConfig:
    """
    Agent configuration.

    All sensitive settings are local-only.
    """
    # Directories
    data_dir: Path = field(default_factory=lambda: Path("./agent_data"))
    artifacts_dir: Optional[Path] = None

    # Cloud connection
    cloud_base_url: str = "http://localhost:8000"

    # Intervals (seconds)
    heartbeat_interval: int = 30
    command_poll_interval: int = 5
    telemetry_interval: int = 60

    # Timeouts (seconds)
    request_timeout: int = 30
    approval_timeout: int = 300

    # Agent info
    agent_version: str = "1.0.0"
    hostname: Optional[str] = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if self.artifacts_dir is None:
            self.artifacts_dir = self.data_dir / "artifacts"


class AgentDaemon:
    """
    CCEA Agent Daemon.

    Main daemon process that:
    - Manages enrollment
    - Sends heartbeats
    - Polls for commands
    - Handles approvals
    - Runs artifacts
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize agent daemon.

        Args:
            config: Agent configuration
        """
        self.config = config
        self._state = AgentState.STOPPED
        self._start_time: Optional[datetime] = None

        # Setup data directory
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.enrollment = AgentEnrollment(
            data_dir=self.config.data_dir,
            cloud_base_url=self.config.cloud_base_url,
            agent_version=self.config.agent_version,
        )

        self.command_handler = CommandHandler(
            approval_timeout=self.config.approval_timeout,
        )

        self.approval_manager = ApprovalManager(
            data_dir=self.config.data_dir / "approvals",
            timeout_seconds=self.config.approval_timeout,
        )

        self.runner = ArtifactRunner(
            artifacts_dir=self.config.artifacts_dir,
        )

        # Threading
        self._running = False
        self._threads: List[threading.Thread] = []
        self._stop_event = threading.Event()

        # Message signer (initialized after enrollment)
        self._signer: Optional[MessageSigner] = None

        # Current deployment state
        self._deployment_state = DeploymentState.CREATED
        self._run_state: Optional[ProtocolRunState] = None

        # Register command executors
        self._register_executors()

    @property
    def state(self) -> AgentState:
        """Get daemon state."""
        return self._state

    @property
    def agent_id(self) -> Optional[str]:
        """Get agent ID if enrolled."""
        return self.enrollment.agent_id

    def is_enrolled(self) -> bool:
        """Check if agent is enrolled."""
        return self.enrollment.is_enrolled()

    def enroll(self, token: str) -> bool:
        """
        Enroll agent with cloud.

        Args:
            token: Enrollment token

        Returns:
            True if successful
        """
        response = self.enrollment.enroll(
            token=token,
            hostname=self.config.hostname,
        )

        if response.success:
            self._deployment_state = DeploymentState.ENROLLED
            # Initialize signer
            private_key = self.enrollment.get_private_key()
            if private_key:
                self._signer = MessageSigner(private_key, self.agent_id)
            return True

        return False

    def start(self) -> None:
        """Start the agent daemon."""
        if self._state == AgentState.RUNNING:
            logger.warning("Agent already running")
            return

        if not self.is_enrolled():
            raise RuntimeError("Agent must be enrolled before starting")

        self._state = AgentState.STARTING
        self._start_time = datetime.utcnow()
        self._running = True
        self._stop_event.clear()

        # Initialize signer if not done
        if self._signer is None:
            private_key = self.enrollment.get_private_key()
            if private_key:
                self._signer = MessageSigner(private_key, self.agent_id)

        # Start heartbeat thread
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="heartbeat",
            daemon=True,
        )
        heartbeat_thread.start()
        self._threads.append(heartbeat_thread)

        # Start command poll thread
        command_thread = threading.Thread(
            target=self._command_poll_loop,
            name="command_poll",
            daemon=True,
        )
        command_thread.start()
        self._threads.append(command_thread)

        self._state = AgentState.RUNNING
        self._deployment_state = DeploymentState.RUNNING

        logger.info(f"Agent daemon started: {self.agent_id}")

    def stop(self) -> None:
        """Stop the agent daemon."""
        if self._state != AgentState.RUNNING:
            return

        self._state = AgentState.STOPPING
        self._running = False
        self._stop_event.set()

        # Stop all runs
        for run in self.runner.list_runs():
            if run.state == RunState.RUNNING:
                self.runner.stop_run(run.run_id)

        # Wait for threads
        for thread in self._threads:
            thread.join(timeout=5)

        self._threads.clear()
        self._state = AgentState.STOPPED
        self._deployment_state = DeploymentState.STOPPED

        logger.info("Agent daemon stopped")

    def run(self) -> None:
        """
        Run the agent daemon (blocking).

        Sets up signal handlers and blocks until stopped.
        """
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.start()

        try:
            while self._running:
                self._stop_event.wait(timeout=1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, stopping...")
        self._running = False
        self._stop_event.set()

    def _heartbeat_loop(self) -> None:
        """Heartbeat sending loop."""
        while self._running:
            try:
                self._send_heartbeat()
            except Exception as e:
                logger.exception("Heartbeat error")

            self._stop_event.wait(timeout=self.config.heartbeat_interval)

    def _send_heartbeat(self) -> None:
        """Send heartbeat to cloud."""
        if not self.enrollment.credentials:
            return

        uptime = None
        if self._start_time:
            uptime = int((datetime.utcnow() - self._start_time).total_seconds())

        # Get health metrics
        health = self._get_health_metrics()

        # Create heartbeat message
        heartbeat = HeartbeatMessage(
            agent_id=self.agent_id,
            state=AgentState(
                deployment_state=self._deployment_state,
                run_state=self._run_state,
                agent_version=self.config.agent_version,
                uptime_seconds=uptime,
            ),
            health=health,
        )

        # Sign and send
        data = heartbeat.model_dump(mode="json")
        if self._signer:
            data = self._signer.sign(data)

        try:
            endpoint = self.enrollment.credentials.heartbeat_endpoint
            resp = requests.post(
                endpoint,
                json=data,
                timeout=self.config.request_timeout,
            )
            if resp.status_code != 200:
                logger.warning(f"Heartbeat failed: {resp.status_code}")
        except requests.RequestException as e:
            logger.warning(f"Heartbeat request failed: {e}")

    def _command_poll_loop(self) -> None:
        """Command polling loop."""
        while self._running:
            try:
                self._poll_commands()
            except Exception as e:
                logger.exception("Command poll error")

            self._stop_event.wait(timeout=self.config.command_poll_interval)

    def _poll_commands(self) -> None:
        """Poll for commands from cloud."""
        if not self.enrollment.credentials:
            return

        # Create poll request
        poll = PollCommandsMessage(
            agent_id=self.agent_id,
            supported_schema_versions=["1.0.0"],
        )

        data = poll.model_dump(mode="json")
        if self._signer:
            data = self._signer.sign(data)

        try:
            endpoint = self.enrollment.credentials.commands_endpoint
            resp = requests.post(
                endpoint,
                json=data,
                timeout=self.config.request_timeout,
            )

            if resp.status_code != 200:
                return

            result = resp.json()
            commands = result.get("commands", [])

            for cmd in commands:
                self._handle_command(cmd)

        except requests.RequestException as e:
            logger.debug(f"Command poll request failed: {e}")

    def _handle_command(self, command: Dict[str, Any]) -> None:
        """Handle incoming command."""
        # Process through command handler
        record = self.command_handler.handle_command(command)

        # Send ack
        self._send_ack(record)

        # If requires approval, create request
        if record.status == LocalCommandStatus.AWAITING_APPROVAL:
            self._create_approval_request(record)
        elif record.status == LocalCommandStatus.APPROVED:
            # Execute immediately (safety commands)
            self._execute_and_report(record)

    def _send_ack(self, record: LocalCommandRecord) -> None:
        """Send command acknowledgment."""
        if not self.enrollment.credentials:
            return

        # Map local status to protocol status
        status_map = {
            LocalCommandStatus.RECEIVED: CommandStatus.RECEIVED,
            LocalCommandStatus.AWAITING_APPROVAL: CommandStatus.AWAITING_APPROVAL,
            LocalCommandStatus.APPROVED: CommandStatus.APPROVED,
            LocalCommandStatus.REJECTED: CommandStatus.REJECTED,
            LocalCommandStatus.FILTERED: CommandStatus.REJECTED,
        }

        ack = CommandAckMessage(
            agent_id=self.agent_id,
            command_id=record.command_id,
            idempotency_key=record.idempotency_key,
            status=status_map.get(record.status, CommandStatus.RECEIVED),
            message=record.filter_reason if record.status == LocalCommandStatus.FILTERED else None,
        )

        data = ack.model_dump(mode="json")
        if self._signer:
            data = self._signer.sign(data)

        try:
            endpoint = self.enrollment.credentials.commands_endpoint.replace("/commands", "/ack")
            requests.post(
                endpoint,
                json=data,
                timeout=self.config.request_timeout,
            )
        except requests.RequestException:
            pass

    def _create_approval_request(self, record: LocalCommandRecord) -> None:
        """Create approval request for command."""
        request = self.approval_manager.create_request(
            command_id=record.command_id,
            command_type=record.command_type,
            deployment_id=record.payload.get("deployment_id"),
            artifact_digest=record.payload.get("artifact_digest"),
            config_digest=record.payload.get("config_digest") or record.payload.get("new_config_digest"),
            change_class=record.payload.get("change_class"),
        )

        # If auto-approved, execute
        if request.status.value == "AUTO_APPROVED":
            self.command_handler.approve_command(
                record.command_id,
                evidence_hash=request.evidence_hash,
            )
            self._execute_and_report(record)

    def _execute_and_report(self, record: LocalCommandRecord) -> None:
        """Execute command and report result."""
        result = self.command_handler.execute_command(record.command_id)

        if result and self.enrollment.credentials:
            # Send result
            result_msg = CommandResultMessage(
                agent_id=self.agent_id,
                command_id=record.command_id,
                idempotency_key=record.idempotency_key,
                status=CommandStatus.COMPLETED if "error" not in result else CommandStatus.FAILED,
                result=result if "error" not in result else None,
                error={"code": "EXEC_ERROR", "message": result.get("error")} if "error" in result else None,
            )

            data = result_msg.model_dump(mode="json")
            if self._signer:
                data = self._signer.sign(data)

            try:
                endpoint = self.enrollment.credentials.commands_endpoint.replace("/commands", "/result")
                requests.post(
                    endpoint,
                    json=data,
                    timeout=self.config.request_timeout,
                )
            except requests.RequestException:
                pass

    def _get_health_metrics(self) -> AgentHealth:
        """Get system health metrics."""
        import psutil

        try:
            return AgentHealth(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                disk_percent=psutil.disk_usage("/").percent,
            )
        except Exception:
            return AgentHealth()

    def _register_executors(self) -> None:
        """Register command executors."""
        from ccea.models.protocol import CommandType

        def execute_start_run(record: LocalCommandRecord) -> Dict[str, Any]:
            deployment_id = record.payload.get("deployment_id")
            artifact_digest = record.payload.get("artifact_digest")
            config_digest = record.payload.get("config_digest")

            # For Phase 1, just acknowledge
            self._run_state = ProtocolRunState.RUNNING
            return {
                "action": "started",
                "deployment_id": deployment_id,
                "artifact_digest": artifact_digest,
            }

        def execute_stop_run(record: LocalCommandRecord) -> Dict[str, Any]:
            deployment_id = record.payload.get("deployment_id")
            self._run_state = ProtocolRunState.STOPPED
            return {"action": "stopped", "deployment_id": deployment_id}

        def execute_pause_run(record: LocalCommandRecord) -> Dict[str, Any]:
            deployment_id = record.payload.get("deployment_id")
            self._run_state = ProtocolRunState.PAUSED
            return {"action": "paused", "deployment_id": deployment_id}

        self.command_handler.register_executor(
            CommandType.REQUEST_START_RUN.value,
            execute_start_run,
        )
        self.command_handler.register_executor(
            CommandType.REQUEST_STOP_RUN.value,
            execute_stop_run,
        )
        self.command_handler.register_executor(
            CommandType.REQUEST_PAUSE_RUN.value,
            execute_pause_run,
        )

    def approve_pending(self, request_id: str) -> bool:
        """
        Approve a pending approval request.

        Args:
            request_id: Request to approve

        Returns:
            True if approved
        """
        result = self.approval_manager.approve(request_id)
        if result:
            # Find and execute the command
            record = self.command_handler.get_command(result.command_id)
            if record:
                self.command_handler.approve_command(
                    record.command_id,
                    evidence_hash=result.evidence_hash,
                )
                self._execute_and_report(record)
            return True
        return False

    def reject_pending(self, request_id: str, reason: Optional[str] = None) -> bool:
        """
        Reject a pending approval request.

        Args:
            request_id: Request to reject
            reason: Rejection reason

        Returns:
            True if rejected
        """
        result = self.approval_manager.reject(request_id, reason=reason)
        if result:
            record = self.command_handler.get_command(result.command_id)
            if record:
                self.command_handler.reject_command(record.command_id, reason)
            return True
        return False

    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """Get pending approval requests."""
        return self.approval_manager.get_pending()
