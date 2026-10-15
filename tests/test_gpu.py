"""NVIDIA card check, GPU/CPU encode lanes with fallback, low priority, and showing finished files."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from jinggao import config, express_sources as es, gpu, media, paths, store, throttle
from jinggao.api import create_app


def test_sessions_follow_the_driver_and_leave_room_for_other_programs():
    assert gpu.sessions_for("552.12") == 6
    assert gpu.sessions_for("546.33") == 4
    assert gpu.sessions_for("472.12") == 3
    assert gpu.sessions_for("") == 3


def test_advice_names_the_driver_version_to_install():
    error = ("Driver does not support the required nvenc API version. Required: 13.1 Found: 12.2\n"
             "The minimum required Nvidia driver for nvenc is 610.00 or newer")
    assert "610.00" in gpu.advice(error, "546.33") and "更新显卡驱动" in gpu.advice(error, "546.33")
    assert "录屏" in gpu.advice("OpenEncodeSessionEx failed: incompatible client key (21)")
    assert "CPU" in gpu.advice("something else")


def _fake_ffmpeg(monkeypatch, *, nvenc=True, full=True, smi="NVIDIA GeForce RTX 3060, 546.33, 12288 MiB"):
    calls = []
    def run(args, **kwargs):
        text = " ".join(map(str, args))
        calls.append(text)
        if "nvidia-smi" in text:
            if smi is None:
                raise ValueError("no smi")
            return smi
        if "h264_nvenc" in args and not nvenc:
            raise ValueError("媒体处理失败：The minimum required Nvidia driver for nvenc is 610.00 or newer")
        if "scale_cuda" in text and not full:
            raise ValueError("媒体处理失败：No such filter: 'scale_cuda'")
        if args[-1] != "-":
            Path(args[-1]).write_bytes(b"video")
        return ""
    monkeypatch.setattr(media, "run", run)
    monkeypatch.setattr(media, "probe", lambda path: {"streams": [{"codec_type": "video", "width": 1280, "height": 720}]})
    monkeypatch.setattr(gpu, "_smi", lambda: None if smi is None else "nvidia-smi")
    return calls


def test_detect_with_a_working_card_writes_the_check_for_every_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    calls = _fake_ffmpeg(monkeypatch)
    state = gpu.detect()
    assert state["nvenc"] and state["full"] and state["sessions"] == 4
    assert state["gpu"] == "NVIDIA GeForce RTX 3060" and state["driver"] == "546.33"
    assert json.loads((tmp_path / gpu.CHECK_FILE).read_text("utf-8"))["full"] is True
    assert any("-hwaccel cuda" in c and "scale_cuda" in c for c in calls)     # the real clip command was tried
    assert gpu.mode("clip") == "full" and gpu.mode("render") == "nvenc"
    assert "显卡加速已开启" in gpu.summary(state)


def test_detect_explains_an_old_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    _fake_ffmpeg(monkeypatch, nvenc=False)
    state = gpu.detect()
    assert state["nvidia"] and not state["nvenc"] and not state["full"]
    assert "610.00" in state["advice"] and gpu.mode("clip") == "cpu" and gpu.mode("render") == "cpu"
    assert "未开启" in gpu.summary(state)


def test_detect_without_a_card_is_quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    _fake_ffmpeg(monkeypatch, nvenc=False, smi=None)
    state = gpu.detect()
    assert not state["nvidia"] and not state["nvenc"] and state["advice"] == ""
    assert gpu.summary(state) == "没有检测到 NVIDIA 显卡，使用 CPU 合成"


def test_nvenc_without_gpu_decoding_still_encodes_on_the_card(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    _fake_ffmpeg(monkeypatch, full=False)
    state = gpu.detect()
    assert state["nvenc"] and not state["full"] and "scale_cuda" in state["reason"]
    assert gpu.mode("clip") == "nvenc"


def test_three_gpu_failures_in_a_row_step_down_to_the_next_way(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    _fake_ffmpeg(monkeypatch)
    gpu.detect()
    gpu.failed("full", "boom")
    gpu.worked("full")                 # a success in between resets the count
    for _ in range(3):
        gpu.failed("full", "boom")
    assert gpu.mode("clip") == "nvenc"
    for _ in range(3):
        gpu.failed("nvenc", "boom")
    assert gpu.mode("clip") == "cpu" and gpu.mode("render") == "cpu"
    assert set(gpu.process_notes()["disabled"]) == {"full", "nvenc"}
    gpu.detect()                       # 「重新检测」 starts afresh
    assert gpu.mode("clip") == "full"


def test_the_cpu_setting_never_uses_the_card(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    _fake_ffmpeg(monkeypatch)
    gpu.detect()
    read = config.read
    monkeypatch.setattr(config, "read", lambda: read() | {"encoder": "cpu"})
    assert gpu.mode("clip") == "cpu" and gpu.mode("render") == "cpu"


def test_clip_commands_for_each_way():
    full = es.clip_args("in.mp4", 2, 3, Path("out.mp4"), "full", short_edge=720)
    assert full[full.index("-hwaccel") + 1] == "cuda" and "scale_cuda" in full[full.index("-vf") + 1]
    assert full[full.index("-r") + 1] == "25" and "h264_nvenc" in full
    cpu = es.clip_args("in.mp4", 2, 3, Path("out.mp4"), "cpu", short_edge=720)
    assert "libx264" in cpu and cpu[cpu.index("-threads") + 1] == "2" and "-hwaccel" not in cpu
    assert cpu.index("-threads") < cpu.index("-i")          # the decoder is limited too, not only the encoder
    nvenc = es.clip_args("in.mp4", 2, 3, Path("out.mp4"), "nvenc", short_edge=720)
    assert "h264_nvenc" in nvenc and "fps=25" in nvenc[nvenc.index("-vf") + 1]


def test_a_real_cpu_clip_is_cut_with_bounded_threads(tmp_path):
    ffmpeg = paths.binary("ffmpeg")
    source = tmp_path / "source.mp4"
    subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30", "-t", "3",
                    "-pix_fmt", "yuv420p", str(source)], check=True)
    assert es.cut_window(str(source), 0.5, 1.2, tmp_path / "clip.mp4") == "cpu"
    stream = next(s for s in media.probe(tmp_path / "clip.mp4")["streams"] if s["codec_type"] == "video")
    assert (stream["width"], stream["height"]) == (1280, 720) and stream["r_frame_rate"] == "25/1"


def test_download_slot_is_free_while_the_clip_is_encoded(tmp_path, monkeypatch):
    seen = {}
    def fetch(stream, url, start, duration, local, check=None):
        local.write_bytes(b"segments")
        return [(0, 10, 4.0, 2.0)]
    def cut(source, start, duration, out, **kwargs):
        free = throttle._slots["bili_download"].acquire(False)
        if free:
            throttle._slots["bili_download"].release()
        seen.update(start=start, free=free)
        out.write_bytes(b"clip")
        return "full"
    monkeypatch.setattr(es, "_fetch_segments", fetch)
    monkeypatch.setattr(es, "cut_window", cut)
    held = [throttle.slot("bili_download") for _ in range(throttle.LIMITS["bili_download"] - 1)]
    for context in held:
        context.__enter__()
    try:
        assert es.dash_window({"urls": ["u"]}, 5.0, 2.0, tmp_path / "w.mp4") == "full"
    finally:
        for context in held:
            context.__exit__(None, None, None)
    assert seen == {"start": 1.0, "free": True}       # window start relative to the first segment
    assert not (tmp_path / "w.src.mp4").exists()


def test_either_prefers_the_card_and_waits_briefly_before_the_cpu(monkeypatch):
    with throttle.either() as lane:
        assert lane == "gpu"
    held = [throttle.slot("gpu") for _ in range(throttle.LIMITS["gpu"])]
    for context in held:
        context.__enter__()
    try:
        released = threading.Timer(.3, lambda: held.pop().__exit__(None, None, None))
        released.start()
        with throttle.either(wait_for_gpu=2) as lane:     # a session frees within the wait: still the card
            assert lane == "gpu"
        released.join()
        again = throttle.slot("gpu")                      # every session busy again
        again.__enter__()
        held.append(again)
        began = time.monotonic()
        with throttle.either(wait_for_gpu=.4) as lane:
            assert lane == "cpu" and time.monotonic() - began >= .35
    finally:
        for context in held:
            context.__exit__(None, None, None)


def test_render_falls_back_to_the_cpu_shot_by_shot_when_the_card_refuses(tmp_path, monkeypatch):
    from PIL import Image
    from jinggao import planner
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    p = store.new_project("显卡回退", media_policy="mixed")
    path = paths.project_dir(p["id"]) / "plain.jpg"
    Image.new("RGB", (1280, 720), "#2c6578").save(path)
    asset = media.inspect_asset(p["id"], path, "纯色")
    shots = [planner.new_shot(f"第{i}镜") for i in range(2)]
    for shot in shots:
        shot.update(duration=1, fit="cover", selected_asset=asset["id"], approved=True, locked=True)
    p.update(shots=shots, assets=[asset])
    real = media.run
    def run(args, **kwargs):
        if "h264_nvenc" in args:
            raise ValueError("媒体处理失败：OpenEncodeSessionEx failed: out of memory")
        return real(args, **kwargs)
    monkeypatch.setattr(media, "run", run)
    monkeypatch.setattr(gpu, "mode", lambda kind="clip": "nvenc")
    result = media.render(p, store.uid(), {"resolution": "720p", "allow_silent": True, "image_motion": False},
                          lambda *_: None, lambda: None)
    assert result["encoders"] == {"cpu": 2} and result["encoder"] == "cpu"
    assert paths.file_path(p["id"], result["path"]).stat().st_size > 1000
    assert gpu.process_notes()["counts"]["nvenc_failed"] == 2


def test_workers_lower_their_own_priority():
    code = ("import os; from jinggao import power; before = os.nice(0); print(power.lower_priority(), before, os.nice(0))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=paths.ROOT,
                         check=True).stdout.split()
    if os.name != "nt":
        assert out[0] == "nice" and int(out[2]) >= 10


def _finished_project(tmp_path):
    p = store.new_project("港口成片")
    folder = paths.project_dir(p["id"]) / "exports" / "e1"
    folder.mkdir(parents=True)
    (folder / "video.mp4").write_bytes(b"mp4")
    saved = tmp_path / "成片" / "港口成片.mp4"
    saved.parent.mkdir()
    saved.write_bytes(b"mp4")
    export = {"id": "e1", "path": "exports/e1/video.mp4", "saved_to": str(saved), "duration": 61.0, "created": 1,
              "width": 1920, "height": 1080, "encoder": "nvenc", "revision": 1}
    store.mutate(p["id"], lambda q: q["exports"].insert(0, export), internal=True)
    return p, saved


def test_finished_videos_can_be_played_revealed_and_listed_in_the_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    opened = []
    monkeypatch.setattr(media, "reveal", lambda target: opened.append(("reveal", target)))
    monkeypatch.setattr(media, "open_path", lambda target: opened.append(("open", target)))
    with TestClient(create_app()) as client:
        p, saved = _finished_project(tmp_path)
        job = store.enqueue(p["id"], "produce", {"batch": "b1"})
        store.claim()
        store.job_update(job["id"], status="completed", progress=100)
        row = next(r for r in client.get("/api/queue").json() if r["id"] == job["id"])
        assert row["video"] == {"path": "exports/e1/video.mp4", "saved_to": str(saved), "duration": 61.0}
        assert client.post(f"/api/projects/{p['id']}/reveal", json={}).json()["path"] == str(saved)
        assert client.post(f"/api/projects/{p['id']}/play", json={"export_id": "e1"}).status_code == 200
        assert opened == [("reveal", saved), ("open", saved)]
        saved.unlink()            # moved away by the user: the copy inside the project is shown instead
        inner = client.post(f"/api/projects/{p['id']}/reveal", json={}).json()["path"]
        assert inner.endswith(os.path.join("exports", "e1", "video.mp4"))
        empty = store.new_project("没做完")
        assert client.post(f"/api/projects/{empty['id']}/play", json={}).status_code == 400
        assert next(r for r in client.get("/api/projects").json() if r["id"] == p["id"])["video"]["id"] == "e1"


def test_folder_picker_changes_where_videos_are_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    chosen = tmp_path / "D盘成片"
    monkeypatch.setattr(media, "choose_folder", lambda initial: str(chosen))
    with TestClient(create_app()) as client:
        r = client.post("/api/output-folder/choose").json()
        assert r == {"path": str(chosen), "changed": True} and config.read()["export_dir"] == str(chosen)
        monkeypatch.setattr(media, "choose_folder", lambda initial: None)      # cancelled: nothing changes
        assert client.post("/api/output-folder/choose").json() == {"path": str(chosen), "changed": False}


def test_system_report_has_the_card_and_recent_videos_but_no_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    with TestClient(create_app()) as client:
        client.patch("/api/settings", json={"model_api_key": "sk-secret-123"})
        _finished_project(tmp_path)
        text = client.get("/api/system/report").text
        assert "显卡" in text and "港口成片" in text and "sk-secret" not in text
        info = client.get("/api/system").json()
        assert {"cores", "cpu_slots", "gpu_slots", "nvenc"} <= set(info)


def test_windows_folder_picker_reads_a_chinese_path_whatever_the_console_code_page(monkeypatch):
    import base64
    chosen = r"D:\视频\镜稿成片"
    seen = {}
    def run(command, **kwargs):
        seen.update(command=command, env=kwargs["env"])
        return subprocess.CompletedProcess(command, 0, base64.b64encode(chosen.encode("utf-8")) + b"\r\n", b"")
    monkeypatch.setattr(media.subprocess, "run", run)
    monkeypatch.setattr(media, "_windows_flags", lambda: 0)
    monkeypatch.setattr(media.os, "name", "nt")
    assert media.choose_folder(r"C:\Users\A\Videos") == chosen
    assert seen["command"][0] == "powershell" and seen["env"]["JINGGAO_FOLDER"] == r"C:\Users\A\Videos"
    monkeypatch.setattr(media.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, b"", b""))
    assert media.choose_folder("") is None          # cancelled
