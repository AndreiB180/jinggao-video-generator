import os
import subprocess
import sys
import time

from jinggao import instance, paths


def test_worker_releases_queue_lock_when_launcher_is_forcefully_killed(tmp_path, monkeypatch):
    data = tmp_path / "orphan-data"
    monkeypatch.setattr(paths, "DATA", data)
    helper = tmp_path / "launch_worker.py"
    helper.write_text("""import multiprocessing, time
from jinggao.worker import loop
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    p = multiprocessing.Process(target=loop, daemon=True)
    p.start()
    while True:
        time.sleep(1)
""")
    env = os.environ | {"JINGGAO_DATA": str(data), "PYTHONPATH": str(paths.ROOT)}
    launcher = subprocess.Popen([sys.executable, str(helper)], env=env)
    try:
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            if (data / "jinggao.sqlite3").exists():
                break
            time.sleep(.1)
        assert (data / "jinggao.sqlite3").exists()
        assert instance.acquire("worker.lock") is None
        # kill(), unlike terminate(), leaves no opportunity for launcher finally blocks.
        launcher.kill()
        launcher.wait(timeout=5)
        deadline = time.monotonic()+6
        while time.monotonic() < deadline:
            handle = instance.acquire("worker.lock")
            if handle is not None:
                handle.close()
                break
            time.sleep(.1)
        else:
            raise AssertionError("orphan worker kept the queue lock after its launcher died")
    finally:
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=5)
