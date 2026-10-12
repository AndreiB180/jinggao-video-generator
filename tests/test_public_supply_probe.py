import json
import subprocess

import pytest

from jinggao import platform_videos, public_supply_probe as probe


def candidate(ident="c319920qxtj", duration=60):
    return {"source_url": f"https://v.qq.com/x/page/{ident}.html", "title": "做饭演示",
            "duration": duration, "provider": "tencent", "media_streams": [{"url": "https://qq.com/a?auth=secret"}],
            "auth": "not-for-report"}


def test_probe_preserves_rank_caps_attempts_and_does_not_retry_failed_urls(tmp_path):
    calls = []

    def run(c, root, seconds):
        calls.append(c)
        assert root == tmp_path and seconds == 10
        assert "media_streams" not in c and "auth" not in c
        return dict(c, status="unavailable", scene_verified=False)

    rows = [candidate(duration=12), candidate("a3196riehgg"), candidate("a3196riehgg"),
            candidate("t323410xl7m"), candidate("r3550j5mgbc")]
    result = probe.probe_candidates(rows, tmp_path, limit=2, runner=run)
    assert [r["video_id"] for r in calls] == ["a3196riehgg", "t323410xl7m"]
    assert result["decoded_count"] == 0 and result["production_index_modified"] is False
    probe.probe_candidates(rows, tmp_path, limit=2, runner=run)
    assert [r["video_id"] for r in calls] == ["a3196riehgg", "t323410xl7m", "r3550j5mgbc"]
    assert "secret" not in (tmp_path / "report.json").read_text()


def test_interrupted_attempt_is_recorded_and_not_automatically_repeated(tmp_path):
    def interrupted(*_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        probe.probe_candidates([candidate()], tmp_path, runner=interrupted)
    result = probe.probe_candidates([candidate()], tmp_path, runner=lambda *_: pytest.fail("retried"))
    assert result["attempts"][0]["status"] == "started"
    assert result["decoded_count"] == 0


def test_search_stops_on_failure_and_cannot_call_baidu(tmp_path, monkeypatch):
    calls = []

    def fetch(query):
        calls.append(query)
        raise ValueError("公开页需要验证")

    monkeypatch.setattr(platform_videos, "tencent_search", fetch)
    assert probe.search_candidates("tencent", ["做饭", "维修"], tmp_path) == []
    assert probe.search_candidates("tencent", ["做饭", "维修"], tmp_path) == []
    assert calls == ["做饭"]
    with pytest.raises(ValueError):
        probe.search_candidates("baidu", ["做饭"], tmp_path)
    with pytest.raises(ValueError):
        probe.search_candidates("tencent", ["a", "b", "c", "d"], tmp_path)


def test_worker_deadline_stops_only_owned_process_and_does_not_approve(tmp_path, monkeypatch):
    from jinggao import video_sources
    calls = []

    class Process:
        pid = 12345
        returncode = None

        def wait(self, timeout):
            calls.append(("wait", timeout))
            raise subprocess.TimeoutExpired("owned worker", timeout)

        def send_signal(self, sig):
            calls.append(("signal", sig))

        def poll(self):
            return None

    process = Process()

    def popen(args, **kwargs):
        assert args[1:4] == ["-m", "jinggao.public_supply_probe", "_worker"]
        assert "API_KEY" not in kwargs["env"]
        return process

    monkeypatch.setenv("API_KEY", "not-for-worker")
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(video_sources, "_stop_public_extractor", lambda proc: calls.append(("stop", proc.pid)))
    result = probe._bounded_worker(probe.candidate_metadata(candidate()), tmp_path, 10)
    assert ("wait", probe.CANDIDATE_SECONDS) in calls and ("stop", 12345) in calls
    assert result["status"] == "deadline" and result["scene_verified"] is False
    task = next((tmp_path / "workers").glob("*.json"))
    assert "secret" not in json.dumps(json.loads(task.read_text()))


def test_invalid_limits_do_no_work(tmp_path):
    for limit, seconds in ((7, 10), (1, 30)):
        with pytest.raises(ValueError):
            probe.probe_candidates([candidate()], tmp_path, limit=limit, clip_seconds=seconds,
                                   runner=lambda *_: pytest.fail("no network"))
