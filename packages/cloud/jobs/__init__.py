# -*- coding: utf-8 -*-
"""
Job Scheduling Package - CLOUD ZONE ONLY.

Provides distributed task queue for training, backtest, and research jobs.

Design Doc Reference:
    - Section 4.1: "Training Service (RL/ML jobs)"
    - Section 4.1: "Backtest & Sim Service (job runner)"

Components:
    - celery_app: Celery application configuration
    - tasks: Task definitions for training, backtest, etc.
    - scheduler: Job scheduling and prioritization
    - monitor: Job monitoring and health checks
"""

from .celery_app import celery_app, CeleryConfig
from .tasks import (
    TrainingTask,
    BacktestTask,
    ResearchTask,
    ArtifactBuildTask,
)
from .scheduler import (
    JobScheduler,
    JobPriority,
    JobConfig,
    ScheduledJob,
)
from .monitor import (
    JobMonitor,
    JobStatus,
    JobMetrics,
)

__all__ = [
    "celery_app",
    "CeleryConfig",
    "TrainingTask",
    "BacktestTask",
    "ResearchTask",
    "ArtifactBuildTask",
    "JobScheduler",
    "JobPriority",
    "JobConfig",
    "ScheduledJob",
    "JobMonitor",
    "JobStatus",
    "JobMetrics",
]
