# -*- coding: utf-8 -*-
"""
script_xs_backtest.py
=====================

CLI cross-sectional бэктеста (Stage A12).

    python script_xs_backtest.py --config configs/config_xs_template.yaml [--out report.json]

Загружает YAML-конфиг, собирает конвейер, прогоняет бэктест и печатает Trust Report.
Источник данных берётся из конфига (synthetic по умолчанию — работает без данных).
Слой ``script_``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

from service_xs_pipeline import XSConfig, run_backtest


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cross-sectional backtest")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--out", default=None, help="optional JSON report output path")
    args = p.parse_args(argv)

    cfg = XSConfig.model_validate(_load_yaml(args.config))
    out = run_backtest(cfg)

    report = {
        "summary": out["summary"],
        "trust_report": out["trust_report"],
        "n_rebalances": out["n_rebalances"],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\nSaved report → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
