"""免费配音 (Edge read-aloud voices) with the service stubbed out: no network."""
import sys
import types
from pathlib import Path

import pytest

from jinggao import config, costs, edge_voice, media, paths


@pytest.fixture
def stub(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    cfg = config.DEFAULTS | {"tts_speed": "1.2", "edge_tts_voice": "zh-CN-XiaoxiaoNeural"}
    monkeypatch.setattr(config, "read", lambda: cfg.copy())
    calls = []
    class Communicate:
        def __init__(self, text, voice, rate="+0%", proxy=None):
            calls.append((text, voice, rate))
            self.text = text
        async def save(self, path):
            Path(path).write_bytes(b"ID3fake")
    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Communicate))
    joined = {}
    def run(args, **kw):
        joined["args"] = args
        Path(args[-1]).write_bytes(b"wav")
        return ""
    monkeypatch.setattr(media, "run", run)
    monkeypatch.setattr(media, "probe", lambda path: {"format": {"duration": "9.5"}})
    return cfg, calls, joined


def test_free_voice_is_the_default_and_costs_nothing(stub):
    cfg, calls, joined = stub
    cfg["tts_provider"] = "edge_tts"
    text = "第一句话讲清楚背景。" * 200
    target = paths.DATA / "audio" / "speech.wav"
    with costs.scope("p1", "j1", "voice"):
        media.synthesize(text, target)
    assert target.read_bytes() == b"wav" and target.with_suffix(".txt").read_text("utf-8") == text
    parts = edge_voice.split(text)
    assert len(parts) > 1 and sorted(c[0] for c in calls) == sorted(parts)
    assert {c[1] for c in calls} == {"zh-CN-XiaoxiaoNeural"} and {c[2] for c in calls} == {"+20%"}
    assert "atempo" not in " ".join(joined["args"])               # speed is the service's rate
    with costs._db() as con:
        assert con.execute("SELECT COUNT(*) FROM charges").fetchone()[0] == 0
    assert config.DEFAULTS["tts_provider"] == "edge_tts"


def test_a_failing_service_says_what_to_do(stub, monkeypatch):
    cfg, calls, joined = stub
    class Broken:
        def __init__(self, *a, **k):
            pass
        async def save(self, path):
            raise ConnectionError("403 Forbidden")
    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Broken))
    monkeypatch.setattr(edge_voice.time, "sleep", lambda s: None)
    with pytest.raises(ValueError, match="千问配音"):
        edge_voice.synthesize("一句话。", paths.DATA / "a.wav")


def test_unknown_voices_fall_back_and_settings_validate(stub, tmp_path, monkeypatch):
    assert edge_voice.voice({"edge_tts_voice": "alloy"}) == edge_voice.DEFAULT_VOICE
    assert edge_voice.rate({"tts_speed": "1.0"}) == "+0%" and edge_voice.rate({"tts_speed": "9"}) == "+0%"
    for voice, *_ in edge_voice.VOICES:
        assert (paths.ROOT / "assets" / "voices" / f"{voice}.mp3").exists()      # bundled samples
