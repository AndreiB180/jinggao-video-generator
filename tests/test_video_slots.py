import multiprocessing
import threading
import time

from fastapi.testclient import TestClient

from jinggao import config, paths, store, throttle, worker
from jinggao.api import create_app


def test_batch_videos_wait_behind_a_single_video(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    batch = [store.enqueue(store.new_project(f"批量{i}")["id"], "produce", {"batch": "b1"}) for i in range(3)]
    single = store.enqueue(store.new_project("单条")["id"], "produce", {})
    assert store.claim()["id"] == single["id"]
    assert [store.claim()["id"] for _ in range(3)] == [job["id"] for job in batch]
    assert store.claim() is None


def test_cancel_all_stops_the_whole_queue_but_not_the_footage_build(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    jobs = [store.enqueue(store.new_project(f"v{i}")["id"], "produce", {"batch": "b1"}) for i in range(4)]
    index = store.enqueue(store.new_project("素材库")["id"], "footage_index", {})
    running = store.claim()
    assert running["id"] == jobs[0]["id"]
    assert store.cancel_all() == {"queued": 3, "running": 1}
    assert [store.get_job(j["id"])["status"] for j in jobs] == ["running", "cancelled", "cancelled", "cancelled"]
    assert store.get_job(running["id"])["cancel"] == 1
    assert store.get_job(index["id"])["status"] == "queued"
    assert store.claim() is None


def test_cancel_one_batch_only(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    first = store.enqueue(store.new_project("a")["id"], "produce", {"batch": "b1"})
    second = store.enqueue(store.new_project("b")["id"], "produce", {"batch": "b2"})
    assert store.cancel_all("b1") == {"queued": 1, "running": 0}
    assert store.get_job(first["id"])["status"] == "cancelled"
    assert store.get_job(second["id"])["status"] == "queued"


def test_batch_of_a_hundred_with_workers_cancel_and_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    with TestClient(create_app()) as client:
        assert client.patch("/api/settings", json={"model_api_key": "sk-test"}).status_code == 200
        projects = [{"title": f"第{i}篇", "script": f"这是第{i}篇文稿的内容。", "burn_subtitles": False} for i in range(100)]
        r = client.post("/api/batch", json={"projects": projects, "workers": 2})
        assert r.status_code == 202, r.text
        batch = r.json()["batch"]
        assert len(r.json()["projects"]) == 100 and config.read()["video_workers"] == "2"
        queue = client.get("/api/queue").json()
        assert len(queue) == 100 and {row["batch"] for row in queue} == {batch}
        pid = r.json()["projects"][0]["project_id"]
        assert client.get(f"/api/projects/{pid}").json()["burn_subtitles"] is False
        assert client.post("/api/batch", json={"projects": projects * 3}).status_code == 422   # 300 > 200
        assert client.post("/api/queue/cancel", json={}).json() == {"queued": 100, "running": 0}
        # Cancelled videos are not "failed": retry-all leaves them; a failed one comes back with its batch.
        assert client.post("/api/queue/retry", json={}).json() == {"retried": 0}
        job = r.json()["projects"][1]["job_id"]
        store.job_update(job, status="failed", error="B站接口暂时限制访问")
        assert client.post("/api/queue/retry", json={"batch": batch}).json() == {"retried": 1}
        again = [row for row in client.get("/api/queue").json() if row["status"] == "queued"]
        assert len(again) == 1 and again[0]["batch"] == batch


def test_video_workers_setting_is_clamped(monkeypatch):
    for value, want in (("1", 1), ("3", 3), ("9", throttle.MAX_VIDEOS), ("", 1), ("x", 1)):
        monkeypatch.setattr(config, "read", lambda value=value: {"video_workers": value})
        assert worker.video_workers() == want


def test_idle_slot_does_not_claim_when_fewer_videos_are_wanted(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    job = store.enqueue(store.new_project("v")["id"], "produce", {})
    monkeypatch.setattr(config, "read", lambda: config.DEFAULTS | {"video_workers": "1"})
    stop = threading.Event()
    claimed = []
    monkeypatch.setattr(worker, "run_once", lambda *a, **k: claimed.append(1) or False)
    thread = threading.Thread(target=worker.loop, args=(stop, "main", 1, None, False), daemon=True)
    thread.start()
    time.sleep(1.5)
    stop.set()
    thread.join(5)
    assert not claimed and store.get_job(job["id"])["status"] == "queued"


def test_shared_slots_bound_every_worker_together():
    throttle.install(throttle.shared(multiprocessing.get_context("spawn")))
    try:
        held = []
        for _ in range(throttle.LIMITS["render"]):
            context = throttle.slot("render")
            context.__enter__()
            held.append(context)
        got = threading.Event()
        def another():
            with throttle.slot("render"):
                got.set()
        thread = threading.Thread(target=another, daemon=True)
        thread.start()
        assert not got.wait(.6)            # the fourth encode waits for a free session
        held.pop().__exit__(None, None, None)
        assert got.wait(2)
        for context in held:
            context.__exit__(None, None, None)
    finally:
        throttle.install({"slots": {n: threading.BoundedSemaphore(v) for n, v in throttle.LIMITS.items()},
                          "bili_pause": throttle._Pause()})


def test_a_refused_bilibili_request_pauses_every_caller():
    throttle.bili_refused(.6)
    began = time.monotonic()
    with throttle.bili_call():
        pass
    assert time.monotonic() - began >= .5


def test_temporary_names_differ_per_thread_and_publish_replaces(tmp_path):
    target = tmp_path / "meta.json"
    names, together = set(), threading.Barrier(4)
    def name():
        together.wait()          # all four alive at once: thread ids are only unique among live threads
        names.add(paths.temporary(target))
    threads = [threading.Thread(target=name) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(names) == 4
    for i in range(2):
        temp = paths.temporary(target)
        temp.write_text(str(i))
        paths.publish(temp, target)
    assert target.read_text() == "1" and not list(tmp_path.glob(".*.tmp"))


def _cut_runs(monkeypatch, tmp_path, gpu, refuse_gpu=False):
    from jinggao import express_sources as es, media
    runs = []
    def run(args, **kwargs):
        runs.append("nvenc" if "h264_nvenc" in args else "x264")
        if refuse_gpu and "h264_nvenc" in args:
            raise ValueError("媒体处理失败：OpenEncodeSessionEx failed: out of memory")
        from pathlib import Path
        Path(args[-1]).write_bytes(b"clip")
        return ""
    monkeypatch.setattr(media, "run", run)
    monkeypatch.setattr(es, "_nvenc", lambda: gpu)
    es.cut_window("in.mp4", 1, 3, tmp_path / "clip.mp4")
    assert (tmp_path / "clip.mp4").read_bytes() == b"clip"
    return runs


def test_candidate_clips_use_a_free_gpu_session_and_fall_back_to_the_cpu(tmp_path, monkeypatch):
    assert _cut_runs(monkeypatch, tmp_path, gpu=False) == ["x264"]
    assert _cut_runs(monkeypatch, tmp_path, gpu=True) == ["nvenc"]
    assert _cut_runs(monkeypatch, tmp_path, gpu=True, refuse_gpu=True) == ["nvenc", "x264"]
    held = [throttle.slot("render") for _ in range(throttle.LIMITS["render"])]
    for context in held:
        context.__enter__()
    try:   # every NVENC session busy with renders: the clip goes to the CPU instead of waiting
        assert _cut_runs(monkeypatch, tmp_path, gpu=True) == ["x264"]
    finally:
        for context in held:
            context.__exit__(None, None, None)


def test_matcher_time_limits_stretch_with_videos_running_at_once(tmp_path, monkeypatch):
    from jinggao import express_match
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    assert express_match.Matcher.share() == 1
    for i in range(3):
        store.enqueue(store.new_project(f"v{i}")["id"], "produce", {})
        store.claim()
    assert express_match.Matcher.share() == 3
