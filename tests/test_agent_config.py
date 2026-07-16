# -*- coding: utf-8 -*-
"""Smoke tests for the CCEA agent daemon config (P0-D closure).

The documented launch ``python -m packages.agent.daemon --config configs/agent.yaml``
had no config file to load, and the config builder passed stale field names to
``DegradedModeConfig`` (would crash on any real config). These tests lock in that
the shipped ``configs/agent.yaml`` loads and validates end-to-end.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from packages.agent.daemon.__main__ import (
    build_daemon_config,
    create_parser,
    load_config_file,
    main,
)

ROOT = Path(__file__).resolve().parents[1]
AGENT_YAML = ROOT / "configs" / "agent.yaml"


def test_agent_yaml_exists():
    assert AGENT_YAML.exists(), "configs/agent.yaml must ship (documented launch target)"


def test_agent_yaml_loads():
    cfg = load_config_file(AGENT_YAML)
    assert isinstance(cfg, dict)
    assert "agent" in cfg and "components" in cfg


def test_build_daemon_config_from_yaml():
    cfg = load_config_file(AGENT_YAML)
    args = create_parser().parse_args([])  # no CLI overrides
    daemon_config = build_daemon_config(cfg, args)

    assert daemon_config.agent_name == "ccea-agent"
    # Kill switch pct thresholds are exact Decimals (not floats).
    ks = daemon_config.kill_switch_config
    assert ks is not None
    assert ks.max_daily_loss_pct == Decimal("0.30")
    assert ks.max_drawdown_pct == Decimal("0.50")
    assert isinstance(ks.max_daily_loss_pct, Decimal)
    # Degraded mode maps to the REAL DegradedModeConfig fields (the old stale
    # field names would have raised TypeError here).
    dg = daemon_config.degraded_mode_config
    assert dg is not None
    assert dg.cloud_timeout_seconds == 120
    assert dg.data_stale_threshold_seconds == 30


def test_degraded_mode_legacy_alias_still_accepted():
    # Backwards-compat: old key names map onto the real fields.
    cfg = {
        "components": {
            "degraded_mode": {
                "cloud_unreachable_threshold_seconds": 99,
                "data_feed_stale_threshold_seconds": 11,
            }
        }
    }
    args = create_parser().parse_args([])
    dc = build_daemon_config(cfg, args)
    assert dc.degraded_mode_config.cloud_timeout_seconds == 99
    assert dc.degraded_mode_config.data_stale_threshold_seconds == 11


def test_dry_run_exits_zero():
    # The documented `--dry-run` validation path returns success (0).
    rc = main(["--config", str(AGENT_YAML), "--dry-run"])
    assert rc == 0


def test_dump_config_exits_zero():
    rc = main(["--config", str(AGENT_YAML), "--dump-config"])
    assert rc == 0
