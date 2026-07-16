# -*- coding: utf-8 -*-
"""
service_experiment_tracking.py
==============================

Слой ``service_`` — файловый, воспроизводимый experiment-tracking + model-registry
(лёгкий аналог MLflow без внешних серверов). Закрывает P0-блокер:
«нельзя защитить выбор модели перед LP/регулятором» — даёт прогоны, метрики,
**lineage** (модель→данные→конфиг→git), **версии**, **стадии**, **rollback** и
**криптоподпись артефактов** (Ed25519, fallback HMAC-SHA256).

Зависимости: stdlib; ``cryptography`` опционально (для Ed25519). Без неё — HMAC.

Хранилище (по умолчанию, переопределяемо):
    experiments/<experiment>/<run_id>/meta.json        # RunRecord
    experiments/<experiment>/<run_id>/metrics.jsonl     # история метрик
    experiments/<experiment>/<run_id>/artifacts/...
    model_registry/<name>/registry.json                 # версии + стадии
    model_registry/<name>/v<version>/<artifact>(.sig)
    state/artifact_ed25519.key | state/artifact_hmac.key # ключи подписи

API:
    tracker = ExperimentTracker()
    with tracker.run("xs_crypto", params={...}) as run:
        run.set_lineage(dataset_uri="data/...", config_uri="configs/...")
        run.log_metric("sharpe", 1.3, step=0)
        ref = run.log_artifact("models/m.json")      # хэшируется и подписывается
    reg = ModelRegistry()
    mv = reg.register("xs_crypto_alpha", run_id=run.run_id, artifact_path="models/m.json",
                      metrics={"sharpe": 1.3})
    reg.transition(mv.name, mv.version, "production")   # архивирует предыдущий production
    reg.rollback(mv.name)                               # вернуть предыдущий production
    assert reg.verify(mv.name, mv.version)              # перепроверка подписи
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from core_experiment import (
    ArtifactRef, Lineage, ModelStage, ModelVersion, RunRecord, RunStatus,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _atomic_write(path: str, data: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_config(obj: Any) -> str:
    """Детерминированный хэш конфига (dict/list/scalar) для lineage."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return sha256_bytes(payload)


def git_lineage(cwd: Optional[str] = None) -> Dict[str, Any]:
    """Текущий git-commit + dirty-флаг (мягко: при отсутствии git → None)."""
    cwd = cwd or _ROOT
    out: Dict[str, Any] = {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        if commit.returncode == 0:
            out["git_commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        if status.returncode == 0:
            out["git_dirty"] = bool(status.stdout.strip())
    except Exception:
        pass
    return out


def capture_environment() -> Dict[str, Any]:
    """Environment fingerprint for reproducibility (P2 #23): python version, platform,
    and versions of key packages (a lightweight in-lineage lockfile)."""
    import platform as _platform
    import sys as _sys
    pkgs: Dict[str, str] = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "cvxpy",
                 "pydantic", "fastapi"):
        try:
            mod = __import__(name.replace("-", "_"))
            pkgs[name] = str(getattr(mod, "__version__", "?"))
        except Exception:
            continue
    return {
        "python_version": _sys.version.split()[0],
        "platform": _platform.platform(),
        "package_versions": pkgs,
    }


# ---------------------------------------------------------------------------
# Подпись артефактов (Ed25519 → fallback HMAC-SHA256)
# ---------------------------------------------------------------------------
class ArtifactSigner:
    """Криптоподпись артефактов для provenance/integrity.

    Предпочтительно Ed25519 (асимметрично; ``cryptography``). Если библиотеки нет —
    HMAC-SHA256 (симметрично; integrity+authenticity внутри организации). Ключи
    создаются и хранятся локально под ``state/`` (никогда не уходят в Cloud).
    """

    def __init__(self, key_dir: Optional[str] = None) -> None:
        self.key_dir = key_dir or os.path.join(_ROOT, "state")
        os.makedirs(self.key_dir, exist_ok=True)
        self._algo = "none"
        self._priv = None
        self._pub_hex: Optional[str] = None
        self._hmac_key: Optional[bytes] = None
        self._init_keys()

    def _init_keys(self) -> None:
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization

            key_path = os.path.join(self.key_dir, "artifact_ed25519.key")
            if os.path.exists(key_path):
                with open(key_path, "rb") as fh:
                    self._priv = serialization.load_pem_private_key(fh.read(), password=None)
            else:
                self._priv = Ed25519PrivateKey.generate()
                pem = self._priv.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                # приватный ключ — read-only владельцу по возможности
                with open(key_path, "wb") as fh:
                    fh.write(pem)
                try:
                    os.chmod(key_path, 0o600)
                except OSError:
                    pass
            pub = self._priv.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            self._pub_hex = pub.hex()
            self._algo = "ed25519"
            # экспорт публичного ключа (для верификации потребителями)
            _atomic_write(os.path.join(self.key_dir, "artifact_ed25519.pub"), self._pub_hex)
            return
        except Exception:
            pass
        # Fallback: HMAC-SHA256
        self._algo = "hmac-sha256"
        env_key = os.environ.get("RIVEN_ARTIFACT_SIGNING_KEY")
        if env_key:
            self._hmac_key = env_key.encode("utf-8")
        else:
            key_path = os.path.join(self.key_dir, "artifact_hmac.key")
            if os.path.exists(key_path):
                with open(key_path, "rb") as fh:
                    self._hmac_key = fh.read()
            else:
                self._hmac_key = os.urandom(32)
                with open(key_path, "wb") as fh:
                    fh.write(self._hmac_key)
                try:
                    os.chmod(key_path, 0o600)
                except OSError:
                    pass

    @property
    def algo(self) -> str:
        return self._algo

    def sign_digest(self, digest_hex: str) -> Dict[str, Optional[str]]:
        msg = digest_hex.encode("utf-8")
        if self._algo == "ed25519" and self._priv is not None:
            sig = self._priv.sign(msg).hex()
            return {"algo": "ed25519", "signature": sig, "public_key": self._pub_hex}
        sig = hmac.new(self._hmac_key or b"", msg, hashlib.sha256).hexdigest()
        return {"algo": "hmac-sha256", "signature": sig, "public_key": None}

    def sign_file(self, path: str) -> Dict[str, Any]:
        digest = sha256_file(path)
        s = self.sign_digest(digest)
        return {"sha256": digest, "size_bytes": os.path.getsize(path), **s}

    def verify(self, digest_hex: str, signature: str, algo: str,
               public_key: Optional[str] = None) -> bool:
        try:
            msg = digest_hex.encode("utf-8")
            if algo == "ed25519":
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
                pub_hex = public_key or self._pub_hex
                if not pub_hex:
                    return False
                pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
                pub.verify(bytes.fromhex(signature), msg)
                return True
            if algo == "hmac-sha256":
                if self._hmac_key is None:
                    return False
                expect = hmac.new(self._hmac_key, msg, hashlib.sha256).hexdigest()
                return hmac.compare_digest(expect, signature)
        except Exception:
            return False
        return False

    def verify_file(self, path: str, ref: ArtifactRef) -> bool:
        if not os.path.exists(path):
            return False
        digest = sha256_file(path)
        if digest != ref.sha256:
            return False
        if ref.algo == "none" or not ref.signature:
            return True  # без подписи проверяем только integrity (хэш совпал)
        return self.verify(digest, ref.signature, ref.algo, ref.public_key)

    def verify_status(self, path: str, ref: ArtifactRef) -> Dict[str, Any]:
        """Honest verification breakdown that does NOT conflate integrity with a
        cryptographic signature. ``signature_valid`` is True only when a real
        signature was present AND verified; an unsigned (integrity-only)
        artifact reports ``status='unsigned'`` and ``signature_valid=False``."""
        if not os.path.exists(path):
            return {"integrity_ok": False, "signed": False,
                    "signature_valid": False, "status": "missing"}
        integrity_ok = sha256_file(path) == ref.sha256
        signed = bool(ref.signature) and ref.algo not in (None, "", "none")
        if not integrity_ok:
            return {"integrity_ok": False, "signed": signed,
                    "signature_valid": False, "status": "hash_mismatch"}
        if not signed:
            return {"integrity_ok": True, "signed": False,
                    "signature_valid": False, "status": "unsigned"}
        sig_ok = self.verify(sha256_file(path), ref.signature, ref.algo, ref.public_key)
        return {"integrity_ok": True, "signed": True,
                "signature_valid": bool(sig_ok),
                "status": "verified" if sig_ok else "signature_mismatch"}


# ---------------------------------------------------------------------------
# Активный прогон (handle)
# ---------------------------------------------------------------------------
class ActiveRun:
    def __init__(self, tracker: "ExperimentTracker", record: RunRecord) -> None:
        self._t = tracker
        self.record = record

    @property
    def run_id(self) -> str:
        return self.record.run_id

    def log_param(self, key: str, value: Any) -> None:
        self.record.params[str(key)] = value
        self._t._save_run(self.record)

    def log_params(self, params: Dict[str, Any]) -> None:
        self.record.params.update({str(k): v for k, v in params.items()})
        self._t._save_run(self.record)

    def set_tags(self, tags: Dict[str, Any]) -> None:
        self.record.tags.update({str(k): v for k, v in tags.items()})
        self._t._save_run(self.record)

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        self.record.metrics[str(key)] = float(value)
        self._t._append_metric(self.record, str(key), float(value), int(step))
        self._t._save_run(self.record)

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def set_lineage(self, *, data_hash: Optional[str] = None, config_hash: Optional[str] = None,
                    dataset_uri: Optional[str] = None, config_uri: Optional[str] = None,
                    parent_run_id: Optional[str] = None, capture_git: bool = True,
                    seed: Optional[int] = None, capture_env: bool = True,
                    **extra: Any) -> None:
        lg = self.record.lineage
        # Reproducibility (P2 #23): capture RNG seed + environment fingerprint.
        if seed is not None:
            lg.seed = int(seed)
        if capture_env and not lg.python_version:
            env = capture_environment()
            lg.python_version = env["python_version"]
            lg.platform = env["platform"]
            lg.package_versions = env["package_versions"]
        if data_hash is not None:
            lg.data_hash = data_hash
        if config_hash is not None:
            lg.config_hash = config_hash
        if dataset_uri is not None:
            lg.dataset_uri = dataset_uri
        if config_uri is not None:
            lg.config_uri = config_uri
        if parent_run_id is not None:
            lg.parent_run_id = parent_run_id
        if capture_git and lg.git_commit is None:
            g = git_lineage()
            lg.git_commit = g["git_commit"]
            lg.git_dirty = g["git_dirty"]
        if extra:
            lg.extra.update(extra)
        self._t._save_run(self.record)

    def log_artifact(self, src_path: str, name: Optional[str] = None) -> ArtifactRef:
        return self._t._log_artifact(self.record, src_path, name=name)

    def end(self, status: str = RunStatus.FINISHED.value) -> None:
        self.record.status = status
        self.record.end_ms = _now_ms()
        self._t._save_run(self.record)

    # context manager
    def __enter__(self) -> "ActiveRun":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.record.status == RunStatus.RUNNING.value:
            self.end(RunStatus.FAILED.value if exc_type else RunStatus.FINISHED.value)


# ---------------------------------------------------------------------------
# Experiment tracker
# ---------------------------------------------------------------------------
class ExperimentTracker:
    def __init__(self, root: Optional[str] = None, signer: Optional[ArtifactSigner] = None) -> None:
        self.root = root or os.path.join(_ROOT, "experiments")
        os.makedirs(self.root, exist_ok=True)
        self.signer = signer or ArtifactSigner()
        self._lock = threading.RLock()

    # -- paths --
    def _run_dir(self, experiment: str, run_id: str) -> str:
        return os.path.join(self.root, experiment, run_id)

    def _meta_path(self, rec: RunRecord) -> str:
        return os.path.join(self._run_dir(rec.experiment, rec.run_id), "meta.json")

    def _metrics_path(self, rec: RunRecord) -> str:
        return os.path.join(self._run_dir(rec.experiment, rec.run_id), "metrics.jsonl")

    # -- persistence --
    def _save_run(self, rec: RunRecord) -> None:
        with self._lock:
            _atomic_write(self._meta_path(rec),
                          json.dumps(rec.to_dict(), indent=2, ensure_ascii=False))

    def _append_metric(self, rec: RunRecord, key: str, value: float, step: int) -> None:
        line = json.dumps({"key": key, "value": value, "step": step, "ts": _now_ms()},
                          ensure_ascii=False)
        path = self._metrics_path(rec)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with self._lock:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def _log_artifact(self, rec: RunRecord, src_path: str, name: Optional[str] = None) -> ArtifactRef:
        if not os.path.exists(src_path):
            raise FileNotFoundError(src_path)
        adir = os.path.join(self._run_dir(rec.experiment, rec.run_id), "artifacts")
        os.makedirs(adir, exist_ok=True)
        base = name or os.path.basename(src_path)
        dst = os.path.join(adir, base)
        with open(src_path, "rb") as r, open(dst, "wb") as w:
            w.write(r.read())
        s = self.signer.sign_file(dst)
        ref = ArtifactRef(path=dst, sha256=s["sha256"], size_bytes=s["size_bytes"],
                          algo=s["algo"], signature=s["signature"],
                          public_key=s.get("public_key"), name=base)
        # сохранить sidecar-подпись
        _atomic_write(dst + ".sig", json.dumps(ref.to_dict(), indent=2, ensure_ascii=False))
        rec.artifacts.append(ref)
        self._save_run(rec)
        return ref

    # -- public API --
    def start_run(self, experiment: str, *, params: Optional[Dict[str, Any]] = None,
                  tags: Optional[Dict[str, Any]] = None,
                  run_name: Optional[str] = None) -> ActiveRun:
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        rid = f"{run_name + '-' if run_name else ''}{ts}-{uuid.uuid4().hex[:6]}"
        rec = RunRecord(
            run_id=rid, experiment=experiment, status=RunStatus.RUNNING.value,
            start_ms=_now_ms(), params=dict(params or {}), tags=dict(tags or {}),
            lineage=Lineage(),
        )
        self._save_run(rec)
        return ActiveRun(self, rec)

    def run(self, experiment: str, **kwargs: Any) -> ActiveRun:
        """Контекст-менеджер: ``with tracker.run('exp', params={...}) as r: ...``"""
        return self.start_run(experiment, **kwargs)

    def get_run(self, experiment: str, run_id: str) -> Optional[RunRecord]:
        path = os.path.join(self._run_dir(experiment, run_id), "meta.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return RunRecord.from_dict(json.load(fh))

    def list_experiments(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted([d for d in os.listdir(self.root)
                       if os.path.isdir(os.path.join(self.root, d))])

    def list_runs(self, experiment: str) -> List[RunRecord]:
        edir = os.path.join(self.root, experiment)
        if not os.path.isdir(edir):
            return []
        out: List[RunRecord] = []
        for rid in os.listdir(edir):
            rec = self.get_run(experiment, rid)
            if rec is not None:
                out.append(rec)
        out.sort(key=lambda r: r.start_ms, reverse=True)
        return out

    def read_metric_history(self, experiment: str, run_id: str, key: str) -> List[Dict[str, Any]]:
        path = os.path.join(self._run_dir(experiment, run_id), "metrics.jsonl")
        if not os.path.exists(path):
            return []
        out = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("key") == key:
                    out.append(rec)
        return out


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self, root: Optional[str] = None, signer: Optional[ArtifactSigner] = None) -> None:
        self.root = root or os.path.join(_ROOT, "model_registry")
        os.makedirs(self.root, exist_ok=True)
        self.signer = signer or ArtifactSigner()
        self._lock = threading.RLock()

    def _name_dir(self, name: str) -> str:
        return os.path.join(self.root, name)

    def _registry_path(self, name: str) -> str:
        return os.path.join(self._name_dir(name), "registry.json")

    def _load(self, name: str) -> Dict[str, Any]:
        path = self._registry_path(name)
        if not os.path.exists(path):
            return {"name": name, "versions": []}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save(self, name: str, data: Dict[str, Any]) -> None:
        _atomic_write(self._registry_path(name),
                      json.dumps(data, indent=2, ensure_ascii=False))

    def register(self, name: str, *, run_id: Optional[str] = None,
                 artifact_path: str, metrics: Optional[Dict[str, float]] = None,
                 lineage: Optional[Lineage] = None, description: str = "") -> ModelVersion:
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(artifact_path)
        with self._lock:
            data = self._load(name)
            next_ver = (max([v["version"] for v in data["versions"]], default=0) + 1)
            vdir = os.path.join(self._name_dir(name), f"v{next_ver}")
            os.makedirs(vdir, exist_ok=True)
            base = os.path.basename(artifact_path)
            dst = os.path.join(vdir, base)
            with open(artifact_path, "rb") as r, open(dst, "wb") as w:
                w.write(r.read())
            s = self.signer.sign_file(dst)
            ref = ArtifactRef(path=dst, sha256=s["sha256"], size_bytes=s["size_bytes"],
                              algo=s["algo"], signature=s["signature"],
                              public_key=s.get("public_key"), name=base)
            _atomic_write(dst + ".sig", json.dumps(ref.to_dict(), indent=2, ensure_ascii=False))
            mv = ModelVersion(
                name=name, version=next_ver, run_id=run_id,
                stage=ModelStage.NONE.value, artifact=ref,
                metrics=dict(metrics or {}), lineage=lineage or Lineage(),
                description=description, created_ms=_now_ms(),
            )
            data["versions"].append(mv.to_dict())
            self._save(name, data)
            return mv

    def list_versions(self, name: str) -> List[ModelVersion]:
        data = self._load(name)
        return [ModelVersion.from_dict(v) for v in data.get("versions", [])]

    def get_version(self, name: str, version: int) -> Optional[ModelVersion]:
        for v in self.list_versions(name):
            if v.version == version:
                return v
        return None

    def get(self, name: str, *, stage: Optional[str] = None,
            version: Optional[int] = None) -> Optional[ModelVersion]:
        if version is not None:
            return self.get_version(name, version)
        stage = stage or ModelStage.PRODUCTION.value
        cands = [v for v in self.list_versions(name) if v.stage == stage]
        if not cands:
            return None
        return max(cands, key=lambda v: v.version)

    def transition(self, name: str, version: int, stage: str, *, force: bool = False) -> ModelVersion:
        stage = ModelStage(stage).value
        with self._lock:
            data = self._load(name)
            target: Optional[Dict[str, Any]] = None
            for v in data["versions"]:
                if v["version"] == version:
                    target = v
                elif stage == ModelStage.PRODUCTION.value and v.get("stage") == ModelStage.PRODUCTION.value:
                    # единственный production: архивируем предыдущий
                    v["stage"] = ModelStage.ARCHIVED.value
            if target is None:
                raise ValueError(f"model '{name}' has no version {version}")
            # Reproducibility guard (P2 #23): refuse to promote a run built from a
            # dirty git tree to PRODUCTION (its exact code state isn't recoverable),
            # unless explicitly forced.
            if stage == ModelStage.PRODUCTION.value and not force:
                lin = (target.get("lineage") or {})
                if lin.get("git_dirty") is True:
                    raise ValueError(
                        f"refusing to promote '{name}' v{version} to production: built from a "
                        f"dirty git tree (not reproducible). Commit changes or pass force=True."
                    )
            target["stage"] = stage
            self._save(name, data)
            return ModelVersion.from_dict(target)

    def rollback(self, name: str, *, to_version: Optional[int] = None) -> ModelVersion:
        """Откат production: на ``to_version`` или на последнюю архивированную версию,
        которая ранее была production (самый свежий предыдущий production)."""
        with self._lock:
            versions = self.list_versions(name)
            if to_version is None:
                archived = sorted(
                    [v for v in versions if v.stage == ModelStage.ARCHIVED.value],
                    key=lambda v: v.version, reverse=True,
                )
                if not archived:
                    raise ValueError(f"no archived version to roll back to for '{name}'")
                to_version = archived[0].version
            return self.transition(name, to_version, ModelStage.PRODUCTION.value)

    def verify(self, name: str, version: int) -> bool:
        mv = self.get_version(name, version)
        if mv is None or mv.artifact is None:
            return False
        return self.signer.verify_file(mv.artifact.path, mv.artifact)

    def verify_status(self, name: str, version: int) -> Dict[str, Any]:
        """Honest verification breakdown (see ArtifactSigner.verify_status)."""
        mv = self.get_version(name, version)
        if mv is None or mv.artifact is None:
            return {"integrity_ok": False, "signed": False,
                    "signature_valid": False, "status": "not_found"}
        return self.signer.verify_status(mv.artifact.path, mv.artifact)


# ---------------------------------------------------------------------------
# Singletons (для API) + helper'ы
# ---------------------------------------------------------------------------
_GLOBAL_TRACKER: Optional[ExperimentTracker] = None
_GLOBAL_REGISTRY: Optional[ModelRegistry] = None


def get_tracker() -> ExperimentTracker:
    global _GLOBAL_TRACKER
    if _GLOBAL_TRACKER is None:
        _GLOBAL_TRACKER = ExperimentTracker()
    return _GLOBAL_TRACKER


def get_registry() -> ModelRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ModelRegistry()
    return _GLOBAL_REGISTRY


__all__ = [
    "ArtifactSigner", "ExperimentTracker", "ModelRegistry", "ActiveRun",
    "get_tracker", "get_registry",
    "sha256_file", "sha256_bytes", "hash_config", "git_lineage",
]
