# -*- coding: utf-8 -*-
"""Закрытие квант-гэпов P2-H/I/M + DAG-пайплайн (§5.21+/22/24/25).

Покрывает четыре модуля:
  * services/hardware.py        — GPU-детекция/выбор устройства (честная);
  * services/premium_data.py    — интрадей-фиды: минутки/тики + entitlement;
  * REST compare/bundle         — сравнение экспериментов + reproducibility;
  * services/research_pipeline  — DAG-оркестратор (deps/blocked/resume/LeakGuard).
"""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

os.environ.setdefault("SEASONALITY_API_TOKEN", "test-token-quant-gaps")
os.environ.setdefault("RIVEN_ENABLE_CCEA", "0")

from fastapi.testclient import TestClient

import app as app_module
from app import api

client = TestClient(api, headers={"X-API-Key": app_module.API_TOKEN})


# ============================================================= GPU (P2-H)


class TestHardware:
    def test_gpu_status_honest_shape(self):
        from services.hardware import gpu_status

        st = gpu_status()
        assert isinstance(st["torch_available"], bool)
        assert isinstance(st["cuda_available"], bool)
        assert st["reason"]  # причина всегда объяснена
        if not st["cuda_available"] and st["torch_available"]:
            # CPU-сборка → честная подсказка как поставить GPU-вариант
            assert st["torch_cuda_build"] is not None or st["install_hint"]

    def test_resolve_cpu_explicit(self):
        from services.hardware import resolve_device

        r = resolve_device("cpu")
        assert r["effective"] == "cpu" and r["requested"] == "cpu"

    def test_resolve_cuda_degrades_honestly_without_gpu(self, monkeypatch):
        import services.hardware as hw

        monkeypatch.setattr(
            hw, "gpu_status", lambda: {"cuda_available": False, "reason": "torch собран без CUDA"}
        )
        r = hw.resolve_device("cuda")
        assert r["effective"] == "cpu"
        assert "cuda" in r["requested"] and "CUDA" in r["reason"] or "cuda" in r["reason"]

    def test_resolve_auto_uses_cuda_when_available(self, monkeypatch):
        import services.hardware as hw

        monkeypatch.setattr(
            hw, "gpu_status", lambda: {"cuda_available": True, "reason": "CUDA доступна: Test GPU"}
        )
        assert hw.resolve_device("auto")["effective"] == "cuda"

    def test_resolve_env_override(self, monkeypatch):
        from services.hardware import resolve_device

        monkeypatch.setenv("RIVEN_TRAIN_DEVICE", "cpu")
        assert resolve_device(None)["effective"] == "cpu"

    def test_api_hardware_gpu(self):
        res = client.get("/api/hardware/gpu")
        assert res.status_code == 200
        d = res.json()
        assert "cuda_available" in d and "resolution" in d
        assert d["resolution"]["cpu"]["effective"] == "cpu"

    def test_train_cli_has_device_flag(self):
        src = Path("train_model_multi_patch.py").read_text(encoding="utf-8")
        assert '"--device"' in src
        assert "RIVEN_TRAIN_DEVICE_EFFECTIVE" in src  # env-мост до ctor
        # ctor реально получает device
        assert 'device=os.environ.get("RIVEN_TRAIN_DEVICE_EFFECTIVE"' in src

    def test_run_train_job_passes_device(self, monkeypatch):
        captured = {}

        def fake_start(cmd, *a, **k):
            captured["cmd"] = [str(c) for c in cmd]
            return 4242

        monkeypatch.setattr(app_module, "start_background", fake_start)
        monkeypatch.setattr(app_module, "background_running", lambda _p: False)
        res = client.post("/api/run_job", json={"job": "run_train", "params": {"device": "cuda"}})
        assert res.status_code == 200, res.text
        assert "--device" in captured["cmd"] and "cuda" in captured["cmd"]


# ==================================================== premium data (P2-M)


class _FakeBar:
    def __init__(self, ts, o, h, l, c, v):
        # canonical core_models.Bar carries volume as volume_base
        self.ts, self.open, self.high, self.low, self.close, self.volume_base = ts, o, h, l, c, v


class _FakeAdapter:
    def __init__(self):
        self.calls = []

    def get_bars(self, symbol, timeframe, *, limit=1000, start_ts=None, end_ts=None):
        self.calls.append((symbol, timeframe, start_ts, end_ts))
        # два окна: во втором дубликат первого бара — проверяем dedupe
        base = start_ts
        return [
            _FakeBar(base, 100.0, 101.0, 99.0, 100.5, 10.0),
            _FakeBar(base + 60_000, 100.5, 102.0, 100.0, 101.0, 12.0),
        ]


class TestPremiumData:
    def test_vendor_status_honest(self, monkeypatch):
        for k in (
            "POLYGON_API_KEY",
            "ALPACA_API_KEY",
            "ALPACA_API_SECRET",
            "OANDA_API_KEY",
            "OANDA_ACCOUNT_ID",
        ):
            monkeypatch.delenv(k, raising=False)
        from services.premium_data import vendor_status

        by = {v["vendor"]: v for v in vendor_status()}
        assert by["binance"]["ready"] is True  # публичный API без ключей
        assert by["binance"]["ticks"] == "history"  # aggTrades-бэкфилл реален
        assert by["polygon"]["ready"] is False and by["polygon"]["paid"] is True
        assert by["alpaca"]["keys_present"] is False
        # ticks честно unavailable там, где адаптер не умеет историю
        assert by["polygon"]["ticks"] == "unavailable"

    def test_download_minute_bars_schema_and_manifest(self, tmp_path):
        from services.premium_data import download_minute_bars
        import pandas as pd

        fake = _FakeAdapter()
        res = download_minute_bars(
            "binance",
            ["BTCUSDT"],
            timeframe="1m",
            start_ts_ms=1_700_000_000_000,
            end_ts_ms=1_700_000_000_000 + 2 * 24 * 3600 * 1000,
            out_dir=str(tmp_path),
            adapter=fake,
        )[0]
        assert res.ok, res.error
        df = pd.read_parquet(res.path)
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        assert df["timestamp"].is_monotonic_increasing
        assert (df["timestamp"] < 10**12).all()  # секунды, не мс
        manifest = json.loads(Path(res.manifest_path).read_text(encoding="utf-8"))
        assert manifest["sha256"] and manifest["rows"] == len(df) == res.rows
        assert len(fake.calls) == 2  # окно по chunk_ms

    def test_download_unknown_vendor_and_bad_timeframe(self, tmp_path):
        from services.premium_data import download_minute_bars

        bad = download_minute_bars(
            "nope", ["X"], start_ts_ms=0, end_ts_ms=1, out_dir=str(tmp_path)
        )[0]
        assert not bad.ok and "неизвестный" in bad.error
        bad_tf = download_minute_bars(
            "binance",
            ["X"],
            timeframe="3s",
            start_ts_ms=0,
            end_ts_ms=1,
            out_dir=str(tmp_path),
            adapter=_FakeAdapter(),
        )[0]
        assert not bad_tf.ok and "таймфрейм" in bad_tf.error

    def test_agg_trades_pagination(self, tmp_path):
        from services.premium_data import download_binance_agg_trades
        import pandas as pd

        pages = [
            [{"a": i, "p": "100.0", "q": "0.1", "T": 1000 + i, "m": False} for i in range(1000)],
            [
                {"a": 1000 + i, "p": "100.5", "q": "0.2", "T": 2000 + i, "m": True}
                for i in range(10)
            ],
        ]
        calls = []

        def fetch(params):
            calls.append(dict(params))
            return pages[len(calls) - 1] if len(calls) <= len(pages) else []

        res = download_binance_agg_trades(
            ["BTCUSDT"], start_ts_ms=1000, end_ts_ms=999_999, out_dir=str(tmp_path), fetch_fn=fetch
        )[0]
        assert res.ok and res.rows == 1010
        assert calls[1].get("fromId") == 1000  # canonical fromId-пагинация
        df = pd.read_parquet(res.path)
        assert set(df.columns) == {"ts_ms", "price", "qty", "agg_id", "is_buyer_maker", "symbol"}

    def test_api_vendors_and_download_validation(self, monkeypatch):
        res = client.get("/api/data/premium/vendors")
        assert res.status_code == 200 and res.json()["vendors"]
        res = client.post(
            "/api/data/premium/download",
            json={
                "kind": "bars",
                "vendor": "binance",
                "symbols": [],
                "start": "2026-07-01",
                "end": "2026-07-02",
            },
        )
        assert res.status_code in (400, 422)
        captured = {}
        monkeypatch.setattr(
            app_module,
            "start_background",
            lambda cmd, pid, log: captured.update(cmd=[str(c) for c in cmd]) or 777,
        )
        monkeypatch.setattr(app_module, "background_running", lambda _p: False)
        res = client.post(
            "/api/data/premium/download",
            json={
                "kind": "bars",
                "vendor": "binance",
                "symbols": ["BTCUSDT"],
                "timeframe": "1m",
                "start": "2026-07-01",
                "end": "2026-07-02",
            },
        )
        assert res.status_code == 200, res.text
        assert any(c.endswith("download_premium_data.py") for c in captured["cmd"])
        assert "--vendor" in captured["cmd"] and "binance" in captured["cmd"]


# ============================================ experiment compare (P2-I)


@pytest.fixture
def temp_tracker(tmp_path, monkeypatch):
    import service_experiment_tracking as et

    tracker = et.ExperimentTracker(root=str(tmp_path / "exp"))
    monkeypatch.setattr(et, "get_tracker", lambda: tracker)
    return tracker


def _mk_run(tracker, exp, params, metrics):
    run = tracker.start_run(exp, params=params)
    for k, v in metrics.items():
        run.log_metric(k, v)
    if hasattr(run, "finish"):
        run.finish()
    elif hasattr(run, "end"):
        run.end()
    return run.record.run_id if hasattr(run, "record") else run.run_id


class TestExperimentCompare:
    def test_compare_params_diff_and_metrics(self, temp_tracker):
        r1 = _mk_run(
            temp_tracker, "cmp", {"lr": 0.001, "gamma": 0.99}, {"sharpe": 1.2, "loss": 0.5}
        )
        r2 = _mk_run(
            temp_tracker, "cmp", {"lr": 0.0005, "gamma": 0.99}, {"sharpe": 1.5, "loss": 0.4}
        )
        res = client.get(f"/api/experiments/cmp/compare?runs={r1},{r2}")
        assert res.status_code == 200, res.text
        d = res.json()
        params = {p["key"]: p for p in d["params"]}
        assert params["lr"]["differs"] is True
        assert params["gamma"]["differs"] is False
        metrics = {m["key"]: m for m in d["metrics"]}
        assert metrics["sharpe"]["values"][r2] == 1.5

    def test_compare_requires_two_runs(self, temp_tracker):
        r1 = _mk_run(temp_tracker, "cmp1", {"a": 1}, {})
        assert client.get(f"/api/experiments/cmp1/compare?runs={r1}").status_code == 400

    def test_compare_unknown_run_404(self, temp_tracker):
        r1 = _mk_run(temp_tracker, "cmp2", {"a": 1}, {})
        assert client.get(f"/api/experiments/cmp2/compare?runs={r1},nope").status_code == 404

    def test_bundle_reproducibility(self, temp_tracker):
        r1 = _mk_run(temp_tracker, "cmpb", {"lr": 0.001}, {"sharpe": 1.0})
        res = client.get(f"/api/experiments/cmpb/runs/{r1}/bundle")
        assert res.status_code == 200, res.text
        b = res.json()
        assert b["bundle_version"] == 1
        assert b["run"]["params"]["lr"] == 0.001
        assert "sharpe" in b["metric_histories"]
        assert "python" in b["environment"]


# ============================================ DAG pipeline (§5.21+)


def _fake_worker(log, fail_on=()):
    def worker(name, params, timeout):
        log.append((name, dict(params)))
        if name in fail_on:
            return "failed", f"{name} упал (тест)", 1
        return "succeeded", f"{name} ok", 0

    return worker


def _spec(tmp_path, extra_step=None):
    from services.research_pipeline import PipelineSpec, StepSpec

    steps = [
        StepSpec(id="a", worker="w_a"),
        StepSpec(id="b", worker="w_b", depends_on=["a"]),
        StepSpec(id="c", worker="w_c", depends_on=["b"]),
        StepSpec(id="d", worker="w_d", depends_on=["a"]),  # параллельная ветка
    ]
    if extra_step:
        steps.append(extra_step)
    return PipelineSpec(name="t", steps=steps)


class TestResearchPipelineDAG:
    def test_spec_validation(self):
        from services.research_pipeline import PipelineSpec, StepSpec

        with pytest.raises(ValueError):
            PipelineSpec(
                name="dup", steps=[StepSpec(id="x", worker="w"), StepSpec(id="x", worker="w")]
            ).validate()
        with pytest.raises(ValueError):
            PipelineSpec(
                name="dep", steps=[StepSpec(id="x", worker="w", depends_on=["nope"])]
            ).validate()
        with pytest.raises(ValueError):
            PipelineSpec(
                name="cyc",
                steps=[
                    StepSpec(id="x", worker="w", depends_on=["y"]),
                    StepSpec(id="y", worker="w", depends_on=["x"]),
                ],
            ).validate()

    def test_reference_spec_loads_and_topo(self):
        from services.research_pipeline import load_spec

        spec = load_spec("research_nightly")
        assert spec is not None
        order = spec.topo_order()
        assert (
            order.index("features")
            < order.index("targets")
            < order.index("no_trade")
            < order.index("splits")
        )
        assert order.index("features") < order.index("training_table")

    def test_run_success_and_order(self, tmp_path):
        from services.research_pipeline import PipelineRunner

        log = []
        runner = PipelineRunner(_fake_worker(log), runs_dir=str(tmp_path))
        state = runner.run(_spec(tmp_path))
        assert state["status"] == "succeeded"
        names = [n for n, _ in log]
        assert names.index("w_a") < names.index("w_b") < names.index("w_c")
        # журнал долговечен
        assert runner.load_run(state["run_id"])["status"] == "succeeded"

    def test_failure_blocks_dependents_fail_closed(self, tmp_path):
        from services.research_pipeline import PipelineRunner

        log = []
        runner = PipelineRunner(_fake_worker(log, fail_on=("w_b",)), runs_dir=str(tmp_path))
        state = runner.run(_spec(tmp_path))
        st = {s["id"]: s["status"] for s in state["steps"]}
        assert state["status"] == "failed"
        assert st["b"] == "failed" and st["c"] == "blocked"  # c не выполнялся
        assert st["a"] == "succeeded" and st["d"] == "succeeded"
        assert all(n != "w_c" for n, _ in log)

    def test_resume_skips_succeeded(self, tmp_path):
        from services.research_pipeline import PipelineRunner

        log1 = []
        runner = PipelineRunner(_fake_worker(log1, fail_on=("w_b",)), runs_dir=str(tmp_path))
        state = runner.run(_spec(tmp_path))
        run_id = state["run_id"]
        # чиним воркер → resume докатывает только b и c
        log2 = []
        runner._worker = _fake_worker(log2)
        state2 = runner.run(_spec(tmp_path), run_id=run_id, resume=True)
        assert state2["status"] == "succeeded" and state2["resumed"] is True
        assert [n for n, _ in log2] == ["w_b", "w_c"]  # a и d не повторялись

    def test_leakguard_floor_clamped_by_engine(self, tmp_path):
        from services.research_pipeline import PipelineRunner, StepSpec

        log = []
        runner = PipelineRunner(_fake_worker(log), runs_dir=str(tmp_path))
        spec = _spec(
            tmp_path,
            StepSpec(
                id="tt",
                worker="run_training_table",
                depends_on=["a"],
                params={"decision_delay_ms": 100},
            ),
        )  # попытка ослабить
        state = runner.run(spec)
        assert state["status"] == "succeeded"
        tt_params = [p for n, p in log if n == "run_training_table"][0]
        assert tt_params["decision_delay_ms"] == 8000  # пол не ослабляется

    def test_api_pipeline_list_and_unknown_run(self):
        res = client.get("/api/pipeline/list")
        assert res.status_code == 200
        names = [p["name"] for p in res.json()["pipelines"]]
        assert "research_nightly" in names
        assert client.post("/api/pipeline/run", json={"name": "no_such"}).status_code == 404

    def test_scheduler_action_registered(self):
        actions = app_module._build_scheduler_actions()
        assert "pipeline.run" in actions
