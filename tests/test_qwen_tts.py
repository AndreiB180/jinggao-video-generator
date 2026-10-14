"""千问配音, 音频成片 and automatic saving, with fake transports and temporary data only."""
import json
from pathlib import Path

import httpx
import pytest

from jinggao import config, costs, media, paths, qwen_tts, store, worker


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    cfg = config.DEFAULTS | {"model_api_key": "test-key", "tts_provider": "qwen_tts", "qwen_tts_voice": "Neil",
                             "tts_speed": "1.0"}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    return cfg


def charges():
    with costs._db() as con:
        return [dict(row) for row in con.execute("SELECT * FROM charges ORDER BY created")]


def fake_service(monkeypatch, *, busy=0):
    calls = {"posts": [], "busy": busy}

    def handler(request):
        if request.method == "POST":
            body = json.loads(request.content)
            calls["posts"].append(body)
            assert request.headers["authorization"] == "Bearer test-key"
            if calls["busy"]:
                calls["busy"] -= 1
                return httpx.Response(429, json={"code": "Throttling", "message": "busy"})
            text = body["input"]["text"]
            return httpx.Response(200, json={"output": {"audio": {"url": f"https://bucket.oss-cn-beijing.aliyuncs.com/{len(calls['posts'])}.wav"}},
                                             "usage": {"characters": qwen_tts.billed(text) - 1}})
        return httpx.Response(200, content=b"RIFF-fake")

    original = httpx.Client
    monkeypatch.setattr(qwen_tts.httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(qwen_tts.time, "sleep", lambda s: None)
    joined = {}

    def run(args, **kw):
        joined["args"] = args
        Path(args[-1]).write_bytes(b"joined")
        return ""
    monkeypatch.setattr(media, "run", run)
    monkeypatch.setattr(media, "probe", lambda path: {"format": {"duration": "12.5"}})
    return calls, joined


def test_split_keeps_every_character_within_the_request_limit():
    text = "范仲淹在杭州救灾。" * 80 + "一句没有句号却很长的话" * 40
    parts = qwen_tts.split(text)
    assert "".join(parts) == text
    assert all(qwen_tts.billed(p) <= qwen_tts.PART for p in parts)
    assert qwen_tts.split("  ") == []
    assert qwen_tts.billed("AI 时代") == 3 + 4


def test_synthesize_voices_parts_in_order_and_settles_on_the_service_count(environment, monkeypatch):
    calls, joined = fake_service(monkeypatch, busy=1)
    text = "第一句话讲清楚背景。" * 60
    target = paths.DATA / "audio" / "speech.wav"
    with costs.scope("p1", "j1", "voice"):
        media.synthesize(text, target)
    assert target.read_bytes() == b"joined"
    assert target.with_suffix(".txt").read_text("utf-8") == text
    parts = qwen_tts.split(text)
    assert sorted(b["input"]["text"] for b in calls["posts"][1:]) == sorted(parts)
    assert {b["input"]["voice"] for b in calls["posts"]} == {"Neil"}
    assert {b["model"] for b in calls["posts"]} == {"qwen3-tts-flash"}
    assert "concat=n=%d" % len(parts) in " ".join(joined["args"]) and "atempo" not in " ".join(joined["args"])
    rows = charges()
    settled = [r for r in rows if r["state"] == "settled"]
    assert len(settled) == len(parts) and {r["category"] for r in rows} == {"tts"}
    assert [r for r in rows if r["state"] == "rejected"]          # the 429 is recorded, at no cost
    assert sum(r["amount"] for r in settled) == sum(80 * (qwen_tts.billed(p) - 1) for p in parts)
    assert {r["project_id"] for r in rows} == {"p1"} and {r["pricing_status"] for r in rows} == {"known"}


def test_speed_and_a_retired_voice_name(environment, monkeypatch):
    environment |= {"tts_speed": "1.2", "qwen_tts_voice": "alloy"}
    monkeypatch.setattr(config, "read", lambda: environment.copy())
    calls, joined = fake_service(monkeypatch)
    media.synthesize("短短一句。", paths.DATA / "a.wav")
    assert "atempo=1.2" in " ".join(joined["args"])
    assert calls["posts"][0]["input"]["voice"] == qwen_tts.DEFAULT_VOICE


def test_speech_has_its_own_daily_cap_and_does_not_use_the_model_caps(environment, monkeypatch):
    fake_service(monkeypatch)
    environment |= {"model_job_budget_cny": "0.00", "model_project_budget_cny": "0.00"}
    monkeypatch.setattr(config, "read", lambda: environment.copy())
    with costs.scope("p1", "j1", "voice"):
        media.synthesize("模型预算为零时配音仍然可以进行。", paths.DATA / "a.wav")
    environment["tts_daily_budget_cny"] = "0.00"
    with costs.scope("p1", "j1", "voice"), pytest.raises(costs.BudgetExceeded, match="今日配音预算"):
        media.synthesize("配音预算用完以后停止。", paths.DATA / "b.wav")


def test_a_missing_key_or_a_foreign_endpoint_stops_before_sending(environment, monkeypatch):
    calls, _ = fake_service(monkeypatch)
    environment["model_api_key"] = ""
    with pytest.raises(ValueError, match="百炼密钥"):
        media.synthesize("一句话。", paths.DATA / "a.wav")
    environment |= {"model_api_key": "k", "model_base_url": "https://api.deepseek.com"}
    with pytest.raises(ValueError, match="DashScope"):
        media.synthesize("一句话。", paths.DATA / "a.wav")
    assert not calls["posts"]


def test_settings_validate_voice_speed_and_migrate_baidu_once(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    paths.DATA.mkdir(parents=True)
    with pytest.raises(ValueError, match="音色"):
        config.update({"qwen_tts_voice": "alloy"})
    with pytest.raises(ValueError, match="语速"):
        config.update({"tts_speed": "2"})
    (paths.DATA / "settings.json").write_text(json.dumps({"tts_provider": "baidu_http"}), "utf-8")
    config.migrate()
    assert config.read()["tts_provider"] == "qwen_tts"
    config.update({"tts_provider": "system"})
    config.migrate()
    assert config.read()["tts_provider"] == "system"


def test_audio_project_is_transcribed_planned_and_keeps_the_upload_as_its_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    paths.prepare()
    store.init()
    p = store.new_project("音频成片")
    audio = {"path": "assets/voice.m4a", "duration": 30.0, "title": "voice.m4a"}
    store.mutate(p["id"], lambda q: q.update(source_audio=audio))
    from jinggao import local_asr, script_norm
    monkeypatch.setattr(media, "run", lambda args, **kw: "")
    monkeypatch.setattr(local_asr, "transcribe", lambda wav, check, progress: [{"text": t} for t in "范仲淹在杭州救灾他反其道而行"])
    monkeypatch.setattr(script_norm, "normalize", lambda raw, check=None: ("范仲淹在杭州救灾，他反其道而行。", {}))
    job = store.enqueue(p["id"], "plan")
    worker.process(job | {"kind": "transcribe", "payload": {}}, finalize=False, progress_sink=lambda *a: None)
    q = store.get_project(p["id"])
    assert q["script"] == "范仲淹在杭州救灾，他反其道而行。" and q["script_original"] == "范仲淹在杭州救灾他反其道而行"
    assert q["title"] == "范仲淹在杭州救灾"


def test_finished_export_is_copied_with_subtitles_one_level_down(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    out = tmp_path / "成片"
    monkeypatch.setattr(config, "read", lambda: config.DEFAULTS | {"export_dir": str(out)})
    pid = store.uid()
    export = paths.project_dir(pid) / "exports" / "e1"
    export.mkdir(parents=True)
    (export / "video.mp4").write_bytes(b"mp4")
    (export / "subtitles.srt").write_text("1", "utf-8")
    project = {"id": pid, "title": "范仲淹/荒政:三策"}
    result = {"path": "exports/e1/video.mp4"}
    first = media.deliver(project, result)
    second = media.deliver(project, result)
    assert Path(first).name == "范仲淹 荒政 三策.mp4" and Path(second).name == "范仲淹 荒政 三策（2）.mp4"
    assert (out / media.SUBTITLE_FOLDER / "范仲淹 荒政 三策.srt").exists()
    assert not list(out.glob("*.srt"))


def test_produce_voices_the_standard_script_while_shots_are_planned(tmp_path, monkeypatch):
    import threading
    import time as clock
    from jinggao import express_plan, planner, script_norm
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    paths.prepare()
    store.init()
    cfg = config.DEFAULTS | {"model_api_key": "k", "tts_provider": "qwen_tts", "pipeline": "express"}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    p = store.new_project("t", "范仲淹在杭州救灾 他反其道而行")
    standard = "范仲淹在杭州救灾，他反其道而行。"
    monkeypatch.setattr(script_norm, "normalize", lambda text, check=None: (standard, {}))
    events, voiced = [], threading.Event()
    def synthesize(text, target, check=None):
        events.append(("voice", text, threading.current_thread().name))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"wav")
        voiced.set()
    def plan(project, check=None):
        assert voiced.wait(5), "the voice did not start before planning finished"
        events.append(("plan", project["script"]))
        return [planner.new_shot("范仲淹在杭州救灾，"), planner.new_shot("他反其道而行。")], {"mode": "express", "script": standard}
    monkeypatch.setattr(media, "synthesize", synthesize)
    monkeypatch.setattr(express_plan, "plan", plan)
    monkeypatch.setattr(media, "probe", lambda path: {"format": {"duration": "6.5"}})
    def stop(*a, **k):
        raise RuntimeError("stop after the voice is attached")
    monkeypatch.setattr(worker, "ensure_narration_alignment", stop)
    job = store.enqueue(p["id"], "produce", {})
    with pytest.raises(RuntimeError, match="stop after"):
        worker.process(job, finalize=False, progress_sink=lambda *a: None)
    q = store.get_project(p["id"])
    assert events[0] == ("voice", standard, "early-voice") and events[1] == ("plan", standard)
    assert q["script"] == standard and q["script_original"] == "范仲淹在杭州救灾 他反其道而行"
    assert q["full_audio"]["duration"] == 6.5 and q["full_audio"]["provider"] == "qwen_tts"
    assert len(q["shots"]) == 2 and all(s["audio"] is None for s in q["shots"])
    assert "voice_full" in q["workflow_info"]["stages"]
