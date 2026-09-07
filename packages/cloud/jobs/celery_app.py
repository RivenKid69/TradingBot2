# -*- coding: utf-8 -*-
"""
Celery Application Configuration - CLOUD ZONE ONLY.

Distributed task queue for ML training, backtesting, and research jobs.

Design Doc Reference:
    - Section 4.1: "Training Service (RL/ML jobs)"
    - Cloud multi-tenant isolation

Based on Celery best practices:
    - https://docs.celeryq.dev/en/stable/userguide/configuration.html
    - https://docs.celeryq.dev/en/stable/userguide/tasks.html
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, Final, List, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_BROKER_URL: Final[str] = "redis://localhost:6379/0"
DEFAULT_RESULT_BACKEND: Final[str] = "redis://localhost:6379/1"

# Queue names
QUEUE_TRAINING: Final[str] = "training"
QUEUE_BACKTEST: Final[str] = "backtest"
QUEUE_RESEARCH: Final[str] = "research"
QUEUE_ARTIFACT: Final[str] = "artifact"
QUEUE_DEFAULT: Final[str] = "default"

# Task timeouts
TRAINING_TIMEOUT: Final[int] = 86400  # 24 hours
BACKTEST_TIMEOUT: Final[int] = 3600  # 1 hour
RESEARCH_TIMEOUT: Final[int] = 7200  # 2 hours
ARTIFACT_TIMEOUT: Final[int] = 1800  # 30 minutes


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class CeleryConfig:
    """
    Celery configuration.

    Supports Redis and RabbitMQ as message brokers.
    """

    # Broker settings
    broker_url: str = field(
        default_factory=lambda: os.getenv("CELERY_BROKER_URL", DEFAULT_BROKER_URL)
    )
    result_backend: str = field(
        default_factory=lambda: os.getenv("CELERY_RESULT_BACKEND", DEFAULT_RESULT_BACKEND)
    )

    # Task settings
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    timezone: str = "UTC"
    enable_utc: bool = True

    # Task execution
    task_acks_late: bool = True  # Acknowledge after task completes
    task_reject_on_worker_lost: bool = True
    task_time_limit: int = TRAINING_TIMEOUT  # Hard limit
    task_soft_time_limit: int = TRAINING_TIMEOUT - 300  # Soft limit (5 min before hard)

    # Worker settings
    worker_prefetch_multiplier: int = 1  # One task at a time for long-running jobs
    worker_concurrency: int = field(
        default_factory=lambda: int(os.getenv("CELERY_CONCURRENCY", "4"))
    )
    worker_max_tasks_per_child: int = 100  # Restart worker after N tasks (memory leak prevention)

    # Result settings
    result_expires: int = 86400 * 7  # 7 days
    result_extended: bool = True

    # Retry settings
    task_default_retry_delay: int = 60
    task_max_retries: int = 3

    # Rate limiting
    task_default_rate_limit: Optional[str] = None  # e.g., "10/m"

    # Queues
    task_queues: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            QUEUE_TRAINING: {"exchange": "training", "routing_key": "training"},
            QUEUE_BACKTEST: {"exchange": "backtest", "routing_key": "backtest"},
            QUEUE_RESEARCH: {"exchange": "research", "routing_key": "research"},
            QUEUE_ARTIFACT: {"exchange": "artifact", "routing_key": "artifact"},
            QUEUE_DEFAULT: {"exchange": "default", "routing_key": "default"},
        }
    )

    task_routes: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "packages.cloud.jobs.tasks.TrainingTask": {"queue": QUEUE_TRAINING},
            "packages.cloud.jobs.tasks.BacktestTask": {"queue": QUEUE_BACKTEST},
            "packages.cloud.jobs.tasks.ResearchTask": {"queue": QUEUE_RESEARCH},
            "packages.cloud.jobs.tasks.ArtifactBuildTask": {"queue": QUEUE_ARTIFACT},
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Celery config dict."""
        return {
            "broker_url": self.broker_url,
            "result_backend": self.result_backend,
            "task_serializer": self.task_serializer,
            "result_serializer": self.result_serializer,
            "accept_content": self.accept_content,
            "timezone": self.timezone,
            "enable_utc": self.enable_utc,
            "task_acks_late": self.task_acks_late,
            "task_reject_on_worker_lost": self.task_reject_on_worker_lost,
            "task_time_limit": self.task_time_limit,
            "task_soft_time_limit": self.task_soft_time_limit,
            "worker_prefetch_multiplier": self.worker_prefetch_multiplier,
            "worker_concurrency": self.worker_concurrency,
            "worker_max_tasks_per_child": self.worker_max_tasks_per_child,
            "result_expires": self.result_expires,
            "result_extended": self.result_extended,
            "task_default_retry_delay": self.task_default_retry_delay,
            "task_default_rate_limit": self.task_default_rate_limit,
            "task_queues": self.task_queues,
            "task_routes": self.task_routes,
        }


# ============================================================================
# Celery Application
# ============================================================================

# Lazy initialization to avoid import errors when Celery is not installed
_celery_app = None


def get_celery_app(config: Optional[CeleryConfig] = None):
    """
    Get or create Celery application.

    Args:
        config: Optional configuration override

    Returns:
        Celery application instance
    """
    global _celery_app

    if _celery_app is not None:
        return _celery_app

    try:
        from celery import Celery
    except ImportError:
        logger.warning("Celery not installed. Job scheduling unavailable.")
        return None

    config = config or CeleryConfig()

    app = Celery(
        "ccea_jobs",
        broker=config.broker_url,
        backend=config.result_backend,
    )

    # Apply configuration
    app.conf.update(**config.to_dict())

    # Register task classes
    app.autodiscover_tasks(["packages.cloud.jobs"])

    _celery_app = app
    logger.info("Celery application initialized")

    return app


# Convenience alias
celery_app = get_celery_app


# ============================================================================
# Celery Beat Schedule (Periodic Tasks)
# ============================================================================


def get_beat_schedule() -> Dict[str, Dict[str, Any]]:
    """
    Get Celery Beat schedule for periodic tasks.

    Returns:
        Schedule configuration dict
    """
    return {
        # Cleanup expired jobs every hour
        "cleanup-expired-jobs": {
            "task": "packages.cloud.jobs.tasks.cleanup_expired_jobs",
            "schedule": timedelta(hours=1),
            "options": {"queue": QUEUE_DEFAULT},
        },
        # Health check every 5 minutes
        "job-health-check": {
            "task": "packages.cloud.jobs.tasks.job_health_check",
            "schedule": timedelta(minutes=5),
            "options": {"queue": QUEUE_DEFAULT},
        },
        # Retry stalled jobs every 15 minutes
        "retry-stalled-jobs": {
            "task": "packages.cloud.jobs.tasks.retry_stalled_jobs",
            "schedule": timedelta(minutes=15),
            "options": {"queue": QUEUE_DEFAULT},
        },
    }
