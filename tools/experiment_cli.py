# -*- coding: utf-8 -*-
"""
tools/experiment_cli.py
=======================

CLI для experiment-tracking и model-registry (P0: MLOps воспроизводимость).

    python tools/experiment_cli.py experiments
    python tools/experiment_cli.py runs <experiment>
    python tools/experiment_cli.py run <experiment> <run_id>
    python tools/experiment_cli.py models
    python tools/experiment_cli.py versions <model>
    python tools/experiment_cli.py register <model> --artifact <path> [--run <run_id>]
    python tools/experiment_cli.py promote <model> <version> [--stage production]
    python tools/experiment_cli.py rollback <model> [--to <version>]
    python tools/experiment_cli.py verify <model> <version>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service_experiment_tracking import get_tracker, get_registry


def _p(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Experiment tracking & model registry CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("experiments")
    s = sub.add_parser("runs"); s.add_argument("experiment")
    s = sub.add_parser("run"); s.add_argument("experiment"); s.add_argument("run_id")
    sub.add_parser("models")
    s = sub.add_parser("versions"); s.add_argument("model")
    s = sub.add_parser("register"); s.add_argument("model"); s.add_argument("--artifact", required=True)
    s.add_argument("--run", default=None); s.add_argument("--desc", default="")
    s = sub.add_parser("promote"); s.add_argument("model"); s.add_argument("version", type=int)
    s.add_argument("--stage", default="production")
    s = sub.add_parser("rollback"); s.add_argument("model"); s.add_argument("--to", type=int, default=None)
    s = sub.add_parser("verify"); s.add_argument("model"); s.add_argument("version", type=int)

    a = ap.parse_args(argv)
    t = get_tracker(); reg = get_registry()

    if a.cmd == "experiments":
        _p([{"experiment": e, "n_runs": len(t.list_runs(e))} for e in t.list_experiments()])
    elif a.cmd == "runs":
        _p([{"run_id": r.run_id, "status": r.status, "metrics": r.metrics} for r in t.list_runs(a.experiment)])
    elif a.cmd == "run":
        rec = t.get_run(a.experiment, a.run_id)
        if rec is None:
            print("run not found", file=sys.stderr); return 1
        _p(rec.to_dict())
    elif a.cmd == "models":
        names = sorted([d for d in os.listdir(reg.root) if os.path.isdir(os.path.join(reg.root, d))]) \
            if os.path.isdir(reg.root) else []
        out = []
        for n in names:
            prod = reg.get(n, stage="production")
            out.append({"name": n, "n_versions": len(reg.list_versions(n)),
                        "production": prod.version if prod else None})
        _p(out)
    elif a.cmd == "versions":
        _p([{**v.to_dict(), "signature_valid": reg.verify(a.model, v.version)}
            for v in reg.list_versions(a.model)])
    elif a.cmd == "register":
        mv = reg.register(a.model, run_id=a.run, artifact_path=a.artifact, description=a.desc)
        _p({"registered": mv.to_dict(), "signature_valid": reg.verify(a.model, mv.version)})
    elif a.cmd == "promote":
        _p({"promoted": reg.transition(a.model, a.version, a.stage).to_dict()})
    elif a.cmd == "rollback":
        _p({"rolled_back_to": reg.rollback(a.model, to_version=a.to).to_dict()})
    elif a.cmd == "verify":
        _p({"model": a.model, "version": a.version, "signature_valid": reg.verify(a.model, a.version)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
