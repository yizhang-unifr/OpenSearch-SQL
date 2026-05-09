from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Optional


def run_command_interruptible(
    command: Iterable[str],
    cwd: Path,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Run a subprocess with robust Ctrl-C propagation and process-group cleanup."""
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(env) if env is not None else None,
        start_new_session=True,
    )
    try:
        rc = process.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, list(command))
    except KeyboardInterrupt:
        _terminate_process_group(process)
        raise


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    os.killpg(pgid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    os.killpg(pgid, signal.SIGKILL)
    process.wait(timeout=5)
