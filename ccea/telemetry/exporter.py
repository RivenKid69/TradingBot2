# -*- coding: utf-8 -*-
"""
CCEA Telemetry Exporter.

Exports telemetry events to cloud with MANDATORY redaction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
import threading

from ccea.telemetry.collector import TelemetryCollector, TelemetryEvent, TelemetryLevel
from ccea.telemetry.redaction import RedactionMiddleware, redact_data
from ccea.crypto.signing import MessageSigner


logger = logging.getLogger(__name__)


@dataclass
class TelemetryBatch:
    """Batch of telemetry events for export."""
    agent_id: str
    batch_id: str
    events: List[TelemetryEvent]
    level: TelemetryLevel
    timestamp: datetime = field(default_factory=datetime.utcnow)
    redaction_applied: bool = True  # Always True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "batch_id": self.batch_id,
            "events": [e.to_dict() for e in self.events],
            "level": self.level.value,
            "timestamp": self.timestamp.isoformat(),
            "redaction_applied": self.redaction_applied,
        }


class TelemetryExporter:
    """
    Telemetry exporter.

    Exports events to cloud with mandatory redaction.
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        endpoint: str,
        signer: Optional[MessageSigner] = None,
        batch_size: int = 100,
        export_interval: int = 60,
    ):
        """
        Initialize exporter.

        Args:
            collector: Telemetry collector
            endpoint: Cloud telemetry endpoint
            signer: Message signer for authentication
            batch_size: Max events per batch
            export_interval: Export interval in seconds
        """
        self.collector = collector
        self.endpoint = endpoint
        self.signer = signer
        self.batch_size = batch_size
        self.export_interval = export_interval

        # MANDATORY redaction
        self._redaction = RedactionMiddleware()

        # Export state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._batch_counter = 0

    def start(self) -> None:
        """Start background export."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._export_loop,
            name="telemetry_export",
            daemon=True,
        )
        self._thread.start()

        logger.info("Telemetry exporter started")

    def stop(self) -> None:
        """Stop background export."""
        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        # Final export
        self.export_now()

        logger.info("Telemetry exporter stopped")

    def export_now(self) -> bool:
        """
        Export events immediately.

        Returns:
            True if export successful
        """
        events = self.collector.flush()
        if not events:
            return True

        # Create batch
        self._batch_counter += 1
        batch = TelemetryBatch(
            agent_id=self.collector.agent_id,
            batch_id=f"batch_{self._batch_counter}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            events=events,
            level=self.collector.level,
        )

        # Export
        return self._send_batch(batch)

    def _export_loop(self) -> None:
        """Background export loop."""
        while self._running:
            try:
                self.export_now()
            except Exception as e:
                logger.exception("Telemetry export error")

            self._stop_event.wait(timeout=self.export_interval)

    def _send_batch(self, batch: TelemetryBatch) -> bool:
        """Send batch to cloud."""
        # MANDATORY: Apply redaction to entire batch
        batch_data = batch.to_dict()
        redacted_data = self._redaction.process(batch_data)

        # Ensure redaction_applied is True
        redacted_data["redaction_applied"] = True

        # Sign if signer available
        if self.signer:
            redacted_data = self.signer.sign(redacted_data)

        try:
            response = requests.post(
                self.endpoint,
                json=redacted_data,
                timeout=30,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                logger.debug(f"Exported batch {batch.batch_id} with {len(batch.events)} events")
                return True
            else:
                logger.warning(f"Telemetry export failed: {response.status_code}")
                return False

        except requests.RequestException as e:
            logger.warning(f"Telemetry export request failed: {e}")
            return False


class LocalTelemetryExporter:
    """
    Local file-based telemetry exporter.

    For enterprise/on-prem deployments where telemetry stays local.
    """

    def __init__(
        self,
        collector: TelemetryCollector,
        output_dir: Path,
        rotate_size_mb: int = 10,
    ):
        """
        Initialize local exporter.

        Args:
            collector: Telemetry collector
            output_dir: Output directory
            rotate_size_mb: File rotation size
        """
        self.collector = collector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_size_mb = rotate_size_mb

        # MANDATORY redaction
        self._redaction = RedactionMiddleware()

        self._current_file: Optional[Path] = None
        self._lock = threading.Lock()

    def export(self) -> int:
        """
        Export events to local file.

        Returns:
            Number of events exported
        """
        events = self.collector.flush()
        if not events:
            return 0

        with self._lock:
            # Get or create output file
            output_file = self._get_output_file()

            # Write events
            with open(output_file, "a") as f:
                for event in events:
                    # MANDATORY: Redact event data
                    event_data = event.to_dict()
                    redacted = self._redaction.process(event_data)
                    redacted["redaction_applied"] = True

                    f.write(json.dumps(redacted) + "\n")

        return len(events)

    def _get_output_file(self) -> Path:
        """Get current output file, rotating if needed."""
        if self._current_file and self._current_file.exists():
            size_mb = self._current_file.stat().st_size / (1024 * 1024)
            if size_mb < self.rotate_size_mb:
                return self._current_file

        # Create new file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._current_file = self.output_dir / f"telemetry_{timestamp}.jsonl"
        return self._current_file
