# -*- coding: utf-8 -*-
"""GPU/устройство обучения: честная детекция и выбор device (P2-H, §5.22).

Закрывает гэп «нет GPU-обучения»: SB3 сам умеет CUDA (``device="auto"``), но
в проекте не было ни детекции (какой torch собран, есть ли CUDA/драйвер), ни
явного контроля устройства в тренировочном CLI/конфиге, ни MVP-поверхности.
Дефолтная установка (`pip install -e ".[cpu]"`) даёт CPU-only torch — обучение
молча шло на CPU, и пользователь не знал почему.

Принципы (как в проф. ML-пайплайнах):
* **Честность**: если CUDA недоступна — говорим ПОЧЕМУ (torch собран без CUDA /
  нет драйвера / нет GPU) и как исправить (``pip install -e ".[gpu]"``).
* **Fail-open на CPU для обучения**: запрошенный ``cuda`` без CUDA не роняет
  тренировку, а деградирует в CPU с явной причиной в статусе и логах
  (research-задача, в отличие от live-риска, не должна падать из-за железа).
* torch импортируется лениво — модуль пригоден и в сборках без torch.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GPU_INSTALL_HINT = 'pip install -e ".[gpu]"  # PyTorch с CUDA 12.1 (см. pyproject.toml)'


def _try_import_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


def _nvidia_smi_gpus() -> List[Dict[str, Any]]:
    """GPU-инвентарь через nvidia-smi (работает и при CPU-only torch)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "vram_total_mb": int(float(parts[2])),
                        "vram_free_mb": int(float(parts[3])),
                        "driver": parts[4],
                    }
                )
        return gpus
    except Exception:
        return []


def gpu_status() -> Dict[str, Any]:
    """Честный снимок GPU-возможностей: torch-сборка, CUDA, устройства, причина."""
    status: Dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "torch_cuda_build": None,  # версия CUDA, с которой собран torch (None = CPU-сборка)
        "cuda_available": False,
        "devices": [],
        "nvidia_smi_devices": _nvidia_smi_gpus(),
        "recommended_device": "cpu",
        "reason": "",
        "install_hint": None,
    }

    torch = _try_import_torch()
    if torch is None:
        status["reason"] = "torch не установлен — обучение недоступно в этой сборке"
        status["install_hint"] = GPU_INSTALL_HINT
        return status

    status["torch_available"] = True
    status["torch_version"] = str(getattr(torch, "__version__", "?"))
    status["torch_cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)

    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    status["cuda_available"] = cuda_ok

    if cuda_ok:
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            try:
                free_b, total_b = torch.cuda.mem_get_info(i)
            except Exception:
                free_b, total_b = 0, getattr(props, "total_memory", 0)
            devices.append(
                {
                    "index": i,
                    "name": props.name,
                    "vram_total_mb": int(total_b / 1e6),
                    "vram_free_mb": int(free_b / 1e6),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        status["devices"] = devices
        status["recommended_device"] = "cuda"
        status["reason"] = f"CUDA доступна: {devices[0]['name']}" if devices else "CUDA доступна"
        return status

    # CUDA недоступна — объясняем почему.
    if status["torch_cuda_build"] is None:
        status["reason"] = (
            "torch собран без CUDA (CPU-сборка) — обучение идёт на CPU. "
            "Для GPU переустановите torch с CUDA."
        )
        status["install_hint"] = GPU_INSTALL_HINT
    elif status["nvidia_smi_devices"]:
        status["reason"] = (
            "torch собран с CUDA, GPU виден драйверу, но torch.cuda.is_available()=False "
            "— проверьте совместимость версии драйвера и CUDA-сборки torch."
        )
    else:
        status["reason"] = (
            "NVIDIA GPU/драйвер не обнаружены (nvidia-smi недоступен) — обучение на CPU."
        )
    return status


def resolve_device(requested: Optional[str] = None) -> Dict[str, Any]:
    """Разрешить запрошенное устройство в эффективное.

    ``auto``/None → cuda при наличии, иначе cpu. Запрошенный ``cuda`` без CUDA
    честно деградирует в cpu (research fail-open) с причиной. Возвращает
    {requested, effective, reason} — то, что уходит и в лог, и в UI.
    """
    req = (requested or os.environ.get("RIVEN_TRAIN_DEVICE") or "auto").strip().lower()
    st = gpu_status()

    if req in ("cpu",):
        return {"requested": req, "effective": "cpu", "reason": "явно запрошен CPU"}

    cuda_ok = st["cuda_available"]
    if req in ("auto", ""):
        eff = "cuda" if cuda_ok else "cpu"
        return {
            "requested": "auto",
            "effective": eff,
            "reason": st["reason"] if not cuda_ok else f"auto → cuda ({st['reason']})",
        }

    if req.startswith("cuda"):
        if cuda_ok:
            return {"requested": req, "effective": req, "reason": st["reason"]}
        return {
            "requested": req,
            "effective": "cpu",
            "reason": f"запрошен {req}, но {st['reason']}",
        }

    # Неизвестное значение — честный auto-фоллбек.
    eff = "cuda" if cuda_ok else "cpu"
    return {"requested": req, "effective": eff, "reason": f"неизвестное устройство {req!r} → {eff}"}


def quick_benchmark(size: int = 2048, repeats: int = 3) -> Dict[str, Any]:
    """Микро-бенчмарк matmul CPU vs GPU (по запросу; честный N/A без CUDA)."""
    torch = _try_import_torch()
    if torch is None:
        return {"ok": False, "reason": "torch не установлен"}
    import time as _t

    def _bench(device: str) -> Optional[float]:
        try:
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            for _ in range(2):
                (a @ b)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = _t.perf_counter()
            for _ in range(repeats):
                (a @ b)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            return (_t.perf_counter() - t0) / repeats * 1000.0
        except Exception as exc:
            logger.warning("gpu-bench(%s): %s", device, exc)
            return None

    out: Dict[str, Any] = {"ok": True, "size": size, "cpu_ms": _bench("cpu")}
    if gpu_status()["cuda_available"]:
        out["cuda_ms"] = _bench("cuda")
        if out.get("cpu_ms") and out.get("cuda_ms"):
            out["speedup"] = round(out["cpu_ms"] / out["cuda_ms"], 1)
    else:
        out["cuda_ms"] = None
    return out


__all__ = ["GPU_INSTALL_HINT", "gpu_status", "quick_benchmark", "resolve_device"]
