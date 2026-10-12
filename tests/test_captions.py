import json

import pytest

from jinggao import captions, media, planner, store
from fastapi.testclient import TestClient
from jinggao.api import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    from jinggao import paths
    monkeypatch.setattr(paths, "DATA", tmp_path / "字幕")
    with TestClient(create_app()) as result:
        yield result


def test_semantic_captions_keep_dependent_clause_with_its_sentence():
    text = "你有没有注意过，买东西的时候，手指只动了几下，收货的时候，箱子却像自己长了腿一样出现在附近？"
    phrases = captions.parts(text)
    assert "".join(phrases) == text
    assert any("收货的时候，箱子" in phrase for phrase in phrases)
    assert all(not phrase.endswith("时候，") for phrase in phrases)
    assert all("\n" not in phrase for phrase in phrases)


def test_audio_alignment_uses_observed_speech_not_length_and_preserves_original():
    words = [{"text": char, "start": start, "end": end} for char, start, end in
             [("你", .3, .6), ("好", .6, 1.1), ("世", 3.0, 3.2), ("界", 3.2, 3.6)]]
    aligned = captions.align("你好。世界！", words, 4)
    assert aligned["cues"] == [{"start": .3, "end": 1.1, "text": "你好。"},
                               {"start": 3.0, "end": 3.6, "text": "世界！"}]
    assert aligned["api_cost_cny"] == 0
    with pytest.raises(ValueError, match="一致度不足"):
        captions.align("这是一份完全不同的稿件。", words, 4)


def test_line_wrap_prefers_clause_and_does_not_split_daowei():
    text = "东西没收到、包装有损坏、沟通没做到位，都可以把事实说清楚，要求解决。"
    wrapped = captions.wrap(text)
    assert "到位，\n" in wrapped
    assert wrapped.replace("\n", "") == text
    assert max(map(len, wrapped.splitlines())) <= 20


def test_one_caption_survives_a_shot_cut_and_has_same_ass_clock():
    track = {"cues": [{"start": 1.2, "end": 4.8, "text": "字幕不会因为画面切换而断成半句话。"}]}
    first = captions.segments(track, 0, 3, 720, 1280)
    second = captions.segments(track, 3, 3, 720, 1280)
    assert first[0][:2] == (1.2, 3)
    assert second[0][:2] == (0, 1.8)
    assert first[0][2] == second[0][2]
    ass, segments = media.subtitles("旧分镜文字", 3, 720, 1280, timed_segments=second)
    assert segments == second
    assert "0:00:00.00,0:00:01.80" in ass
    assert "旧分镜文字" not in ass


def test_caption_cache_binds_audio_bytes_and_original_narration(client):
    p = store.new_project("字幕缓存", media_policy="mixed")
    shot = planner.new_shot("你好。")
    from jinggao import paths
    root = paths.project_dir(p["id"])
    (root / "voice.wav").write_bytes(b"old audio")
    p.update(shots=[shot], full_audio={"path": "voice.wav", "duration": 2})
    *_, binding, target = captions._source(p)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"binding": binding, "cues": [{"start": 0, "end": 1, "text": "你好。"}]}))
    assert captions.cached(p)
    (root / "voice.wav").write_bytes(b"new audio")
    assert captions.cached(p) is None
    (root / "voice.wav").write_bytes(b"old audio")
    shot["narration"] = "文字修改了。"
    assert captions.cached(p) is None


def test_preview_uses_audio_bound_global_cues(client):
    p = store.new_project("字幕预览", media_policy="mixed")
    shots = [planner.new_shot("第一镜的旧切点"), planner.new_shot("第二镜的旧切点")]
    for shot in shots:
        shot["duration"] = 3
    from jinggao import paths
    (paths.project_dir(p["id"]) / "voice.wav").write_bytes(b"cached voice")
    p = store.mutate(p["id"], lambda p: p.update(shots=shots, full_audio={"path": "voice.wav", "duration": 6}))
    *_, binding, target = captions._source(p)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"binding": binding, "method": "local_paraformer",
                                 "cues": [{"start": 2, "end": 5, "text": "这是跨画面的一整句话。"}]}))
    response = client.get(f"/api/projects/{p['id']}/shots/{shots[1]['id']}/captions")
    assert response.status_code == 200
    assert response.json()["segments"] == [[0, 2, "这是跨画面的一整句话。"]]
    assert response.json()["alignment"] == "local_paraformer"
