from __future__ import annotations

import os
import platform
import signal
import subprocess
import json
import logging
import tempfile
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.fspath(path)) or "."
    os.makedirs(directory, exist_ok=True)


# ------------------------------------------------------------------
# Atomic file writing with retries
ALERT_LEVEL = logging.CRITICAL + 10
logging.addLevelName(ALERT_LEVEL, "ALERT")


def atomic_write_with_retry(
    path: str | Path,
    data: str | bytes | None,
    retries: int = 3,
    backoff: float = 0.1,
    mode: str = "w",
) -> None:
    """Atomically write *data* to *path* with retry logic.

    If ``data`` is ``None`` the function will simply ``fsync`` the existing
    file at ``path``.  When ``data`` is provided it is written to a temporary
    file which is then ``os.replace``'d into place.  In both cases the
    directory containing ``path`` is created if necessary and ``fsync`` is
    attempted for durability.  After all retry attempts are exhausted the
    failure is logged at ALERT level and the exception is re-raised.
    """

    p = Path(path)
    for attempt in range(retries + 1):
        try:
            if data is None:
                fd = os.open(str(p), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                if mode == "a":
                    if isinstance(data, (bytes, bytearray)):
                        with open(p, "ab") as f:
                            f.write(data)
                            f.flush()
                            os.fsync(f.fileno())
                    else:
                        with open(p, "a", encoding="utf-8", newline="") as f:
                            f.write(str(data))
                            f.flush()
                            os.fsync(f.fileno())
                    try:
                        dir_fd = os.open(str(p.parent), os.O_DIRECTORY)
                        os.fsync(dir_fd)
                        os.close(dir_fd)
                    except Exception:
                        pass
                else:
                    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent))
                    try:
                        if isinstance(data, (bytes, bytearray)):
                            with os.fdopen(fd, "wb") as f:
                                f.write(data)
                                f.flush()
                                os.fsync(f.fileno())
                        else:
                            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                                f.write(str(data))
                                f.flush()
                                os.fsync(f.fileno())
                        os.replace(tmp_path, str(p))
                        try:
                            dir_fd = os.open(str(p.parent), os.O_DIRECTORY)
                            os.fsync(dir_fd)
                            os.close(dir_fd)
                        except Exception:
                            pass
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
            return
        except Exception:
            if attempt >= retries:
                logging.getLogger(__name__).log(
                    ALERT_LEVEL, "Failed to write %s", path, exc_info=True
                )
                raise
            time.sleep(backoff)


def atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Durably replace a JSON document used for process status metadata."""
    atomic_write_with_retry(
        path,
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )


def run_cmd(cmd: List[str], cwd: Optional[str] = None, log_path: Optional[str] = None) -> int:
    """Blocking command execution with optional logging."""
    from desktop_job_runtime import prepare_python_command, worker_environment

    cmd = prepare_python_command(cmd)
    env = worker_environment()
    if log_path:
        ensure_dir(log_path)
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            f.write(f"\n$ {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=f, text=True, env=env)
            return int(proc.returncode)
    else:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        if proc.stderr:
            print(proc.stderr)
        return int(proc.returncode)


def start_background(cmd: List[str], pid_file: str, log_file: str) -> int:
    """Start a process and durably track its real completion/exit status."""
    from datetime import datetime, timezone
    from desktop_job_runtime import prepare_python_command, worker_environment

    cmd = prepare_python_command(cmd)
    env = worker_environment()
    ensure_dir(pid_file)
    ensure_dir(log_file)
    if os.path.exists(pid_file):
        raise RuntimeError("Process already running (PID file exists). Stop it first.")
    status_file = pid_file + ".json"
    logf = open(log_file, "a", encoding="utf-8", newline="")
    if platform.system() == "Windows":
        creationflags = 0x00000200
        proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, creationflags=creationflags, env=env)
    else:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, preexec_fn=os.setsid, env=env)
    with open(pid_file, "w", encoding="utf-8") as f:
        f.write(str(proc.pid))

    started_at = datetime.now(timezone.utc).isoformat()
    atomic_write_json(
        status_file,
        {
            "pid": int(proc.pid),
            "state": "running",
            "running": True,
            "exit_code": None,
            "started_at": started_at,
            "finished_at": None,
            "command": cmd,
        },
    )

    def _watch() -> None:
        exit_code = int(proc.wait())
        try:
            logf.flush()
        finally:
            logf.close()
        final_state = "succeeded" if exit_code == 0 else "failed"
        try:
            current_status = read_json(status_file)
            if current_status.get("state") == "stopped":
                final_state = "stopped"
        except Exception:
            pass
        atomic_write_json(
            status_file,
            {
                "pid": int(proc.pid),
                "state": final_state,
                "running": False,
                "exit_code": exit_code,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "command": cmd,
            },
        )
        try:
            if os.path.exists(pid_file):
                with open(pid_file, "r", encoding="utf-8") as fh:
                    if fh.read().strip() == str(proc.pid):
                        os.remove(pid_file)
        except Exception:
            pass

    threading.Thread(target=_watch, name=f"job-watch-{proc.pid}", daemon=True).start()
    return int(proc.pid)


def background_status(pid_file: str) -> Dict[str, Any]:
    """Return the last durable process state, including its exit code."""
    status_file = pid_file + ".json"
    status = read_json(status_file)
    if background_running(pid_file):
        status.update({"state": "running", "running": True, "exit_code": None})
        return status
    if status:
        status["running"] = False
        return status
    return {"state": "idle", "running": False, "exit_code": None}


def stop_background(pid_file: str) -> bool:
    """Stop background process using stored PID."""
    if not os.path.exists(pid_file):
        return False
    try:
        with open(pid_file, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
    except Exception:
        os.remove(pid_file)
        return False
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
    try:
        os.remove(pid_file)
    except Exception:
        pass
    try:
        from datetime import datetime, timezone

        current = read_json(pid_file + ".json")
        current.update(
            {
                "state": "stopped",
                "running": False,
                "exit_code": None,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        atomic_write_json(pid_file + ".json", current)
    except Exception:
        pass
    return True


def background_running(pid_file: str) -> bool:
    if not os.path.exists(pid_file):
        return False
    try:
        with open(pid_file, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        if platform.system() == "Windows":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True
            )
            is_alive = str(pid) in (out.stdout or "")
            if not is_alive:
                # A freshly spawned Windows process can take a moment to appear
                # in tasklist.  Removing its pid file during that window makes
                # the UI report a false completion before the worker has even
                # imported its module.  Keep the durable "running" state for a
                # short, bounded startup grace period; the watcher will still
                # publish the real terminal state and exit code.
                try:
                    if time.time() - os.path.getmtime(pid_file) < 2.0:
                        return True
                except OSError:
                    pass
                try:
                    os.remove(pid_file)
                except Exception:
                    pass
            return is_alive
        else:
            try:
                os.kill(pid, 0)
            except OSError:
                try:
                    os.remove(pid_file)
                except Exception:
                    pass
                return False

            # Check if process is zombie on Linux
            try:
                with open(f"/proc/{pid}/status", "r") as f_proc:
                    for line in f_proc:
                        if line.startswith("State:"):
                            state = line.split()[1]
                            if state.upper() in ("Z", "ZOMBIE"):
                                try:
                                    os.waitpid(pid, os.WNOHANG)
                                except Exception:
                                    pass
                                try:
                                    os.remove(pid_file)
                                except Exception:
                                    pass
                                return False
                            break
            except Exception:
                pass

            return True
    except Exception:
        try:
            os.remove(pid_file)
        except Exception:
            pass
        return False


def tail_file(path: str, n: int = 200) -> str:
    if n <= 0 or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            if end <= 0:
                return ""
            block_size = 8192
            chunks: list[bytes] = []
            lines_found = 0
            while end > 0 and lines_found <= n:
                read_size = block_size if end >= block_size else end
                fh.seek(end - read_size)
                chunk = fh.read(read_size)
                chunks.append(chunk)
                lines_found += chunk.count(b"\n")
                end -= read_size
            data = b"".join(reversed(chunks))
            text = data.decode("utf-8", errors="ignore")
            lines = text.splitlines()
            if len(lines) > n:
                lines = lines[-n:]
            return "\n".join(lines)
    except Exception:
        return ""


def read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read_csv(path: str, n: int = 200) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if len(df) > n:
            return df.tail(n).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def read_signals_csv(path: str, n: int = 200) -> pd.DataFrame:
    df = read_csv(path, n=n)
    if df.empty:
        return df
    try:
        if "uid" not in df.columns:
            df["uid"] = df.apply(lambda r: signal_uid(r.to_dict()), axis=1)
    except Exception:
        pass
    return df


def signal_uid(row: Dict[str, Any]) -> str:
    ts = str(int(row.get("ts_ms", 0)))
    sym = str(row.get("symbol", "")).upper()
    fh = str(row.get("features_hash", ""))
    side = str(row.get("side", ""))
    vol = str(row.get("volume_frac", ""))
    return f"{ts}_{sym}_{fh}_{side}_{vol}"


def append_row_csv(path: str, header: List[str], row: List[Any]) -> None:
    ensure_dir(path)
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        import csv as _csv

        w = _csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def append_jsonl(path: str, data: Dict[str, Any]) -> None:
    atomic_write_with_retry(
        path,
        json.dumps(data, separators=(",", ":")) + "\n",
        mode="a",
    )


def load_signals_full(path: str, max_rows: int = 500) -> pd.DataFrame:
    return read_signals_csv(path, n=max_rows)


__all__ = [
    "ensure_dir",
    "atomic_write_with_retry",
    "atomic_write_json",
    "run_cmd",
    "start_background",
    "stop_background",
    "background_running",
    "background_status",
    "tail_file",
    "read_json",
    "read_csv",
    "read_signals_csv",
    "signal_uid",
    "append_row_csv",
    "append_jsonl",
    "load_signals_full",
]
