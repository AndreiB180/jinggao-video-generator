from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jinggao import media, paths, power, store, worker


def test_windows_restores_prior_thread_state_after_failure(monkeypatch):
    setter = Mock(return_value=0x80000002)
    monkeypatch.setattr(power.sys, "platform", "win32")
    monkeypatch.setattr(power.ctypes, "windll", SimpleNamespace(kernel32=SimpleNamespace(
        SetThreadExecutionState=setter)), raising=False)
    with pytest.raises(media.Cancelled):
        with power.keep_awake() as info:
            assert info["active"]
            raise media.Cancelled()
    assert [call.args for call in setter.call_args_list] == [(0x80000001,), (0x80000002,)]


def test_macos_assertion_tracks_worker_and_releases_after_failure(monkeypatch):
    child = Mock()
    child.poll.return_value = None
    launch = Mock(return_value=child)
    monkeypatch.setattr(power.sys, "platform", "darwin")
    monkeypatch.setattr(power.subprocess, "Popen", launch)
    with pytest.raises(ValueError):
        with power.keep_awake() as info:
            assert info["active"]
            raise ValueError("task failed")
    assert launch.call_args.args[0] == ["/usr/bin/caffeinate", "-is", "-w", str(power.os.getpid())]
    child.terminate.assert_called_once()
    child.wait.assert_called_once_with(timeout=2)


def test_unavailable_assertion_does_not_skip_work(monkeypatch):
    monkeypatch.setattr(power.sys, "platform", "darwin")
    monkeypatch.setattr(power.subprocess, "Popen", Mock(side_effect=OSError("unavailable")))
    with power.keep_awake() as info:
        assert info == {"requested": True, "active": False, "method": "caffeinate_idle_system"}


@pytest.mark.parametrize("cancelled", [False, True])
def test_queue_preserves_terminal_result_and_both_clocks(tmp_path, monkeypatch, cancelled):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    p = store.new_project("timing")
    job = store.enqueue(p["id"], "plan")
    monkeypatch.setattr(power.sys, "platform", "linux")
    def process(job, stop):
        if cancelled:
            raise media.Cancelled()
        store.job_update(job["id"], status="completed", progress=100)
    monkeypatch.setattr(worker, "process", process)
    assert worker.run_once()
    assert store.get_job(job["id"])["status"] == ("cancelled" if cancelled else "completed")
    run = store.get_project(p["id"])["execution_info"]
    assert run["job_id"] == job["id"]
    assert run["finished_at"] >= run["started_at"]
    assert run["elapsed_wall_seconds"] >= 0
    assert run["elapsed_monotonic_seconds"] >= 0
    assert run["power_assertion"]["active"] is False
