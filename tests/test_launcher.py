"""The combined launcher owns both children and must reap them on termination."""

import os
from pathlib import Path
import shutil
import subprocess
import time


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_stopping_combined_launcher_stops_worker_and_api(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(Path(__file__).resolve().parents[1] / "scripts/start.sh", scripts / "start.sh")
    for name in ("api", "worker"):
        child = scripts / f"run-{name}.sh"
        child.write_text(f'#!/usr/bin/env bash\necho $$ > runtime/{name}.pid\nexec sleep 60\n')
        child.chmod(0o755)
    process = subprocess.Popen(["bash", str(scripts / "start.sh")], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    children = []
    try:
        deadline = time.monotonic() + 3
        while not all((tmp_path / f"runtime/{name}.pid").exists() for name in ("api", "worker")):
            assert time.monotonic() < deadline, "launcher failed to start its children"
            time.sleep(.02)
        children = [int((tmp_path / f"runtime/{name}.pid").read_text()) for name in ("api", "worker")]
        process.terminate()
        process.wait(timeout=3)
        assert not any(alive(pid) for pid in children), "launcher left a child process running"
    finally:
        for pid in children:
            if alive(pid):
                os.kill(pid, 15)
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=3)
