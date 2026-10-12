"""Public audio remains bounded, source-bound, cancellable and locator-only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading
import time
import wave

import httpcore
import httpx
import pytest

from jinggao import local_asr, media, network, paths, segment_discovery, source_audio, sources

_REAL_READ = segment_discovery._read


@pytest.fixture(scope="module")
def audio_bytes(tmp_path_factory):
    file = tmp_path_factory.mktemp("public-source-audio") / "speech.m4a"
    media.run([paths.binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
               "sine=frequency=500:sample_rate=16000:duration=30", "-c:a", "aac", "-b:a", "48k",
               "-movflags", "+faststart", str(file)], timeout=10)
    return file.read_bytes()


@pytest.fixture
def scenario(tmp_path, monkeypatch, audio_bytes):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    source = "https://www.bilibili.com/video/BV1Ns411E7Mw"
    candidate = {
        "source_url": source, "duration": 30, "metadata_checked": True,
        "source_verified": True, "download_status": "available", "media_extractor": "bilibili-public",
        "source_evidence": {"url": source, "video_id": "BV1Ns411E7Mw", "aid": 27090957,
                            "cid": 46677584, "evidence_origin": "platform_metadata", "readable": True},
    }
    audio_url = "https://upos-sz-estgoss.bilivideo.com/upgcxcode/84/75/46677584/46677584_da2-1-30216.m4s?expires=1"
    metadata = {"code": 0, "data": {"timelength": 30000, "dash": {
        "duration": 30, "audio": [
            {"bandwidth": 132805, "mimeType": "audio/mp4", "codecs": "mp4a.40.2",
             "baseUrl": audio_url.replace("30216", "30232")},
            {"bandwidth": 67269, "mimeType": "audio/mp4", "codecs": "mp4a.40.2", "baseUrl": audio_url},
        ]}}}
    state = {"metadata": metadata, "audio_bytes": audio_bytes, "reads": [], "asr_calls": [],
             "model": {"weights": "model-v1"}, "words": [{"text": "检查", "start": 2., "end": 2.6},
                                                         {"text": "刹车", "start": 2.6, "end": 3.2}]}

    def read(url, budget, referer, *, limit, target=None, deadline_supervised=False):
        budget.check()
        assert deadline_supervised
        assert budget.limit == source_audio.MAX_BYTES
        assert referer == source
        state["reads"].append((url, limit, budget))
        raw = json.dumps(state["metadata"]).encode() if target is None else state["audio_bytes"]
        if len(raw) > limit:
            raise segment_discovery.SampleLimit("采样元数据或分段超过下载预算")
        budget.count(raw)
        if target:
            Path(target).write_bytes(raw)
        return raw, url

    def transcribe(file, check, progress):
        check()
        with wave.open(str(file), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        state["asr_calls"].append(str(file))
        progress(1, "done")
        return deepcopy(state["words"])

    monkeypatch.setattr(segment_discovery, "_read", read)
    monkeypatch.setattr(local_asr, "transcribe", transcribe)
    monkeypatch.setattr(source_audio, "_model_identity", lambda check: deepcopy(state["model"]))
    return candidate, state, tmp_path, audio_url


def test_real_decode_lowest_audio_and_bound_cache_without_media_left(scenario):
    candidate, state, root, audio_url = scenario
    before = deepcopy(candidate)
    result = source_audio.transcript(candidate)
    assert candidate == before
    assert len(state["reads"]) == 2 and state["reads"][1][0] == audio_url
    assert state["reads"][0][1] == 128 * 1024
    assert state["reads"][1][1] < source_audio.MAX_BYTES
    assert len(state["asr_calls"]) == 1
    assert result["words"] == state["words"] and result["duration"] == pytest.approx(30, abs=.1)
    assert result["audio_sha256"] == hashlib.sha256(state["audio_bytes"]).hexdigest()
    assert result["sampling_only"] is True and result["model_requests"] == 0
    assert not result["cache_hit"] and result["network_requests"] == 2
    assert len(state["audio_bytes"]) < result["downloaded_bytes"] < source_audio.MAX_BYTES
    assert not {"approved", "verification", "selected_asset"} & result.keys()
    assert list((root / "cache" / "source-audio").iterdir()) == [Path(result["cache_path"])]
    record = json.loads(Path(result["cache_path"]).read_text())
    assert record["binding"]["cid"] == 46677584 and record["binding"]["asr"] == state["model"]
    assert record["payload"]["cache_key"] == result["cache_key"]
    hit = source_audio.transcript(candidate)
    assert hit["cache_hit"] and hit["downloaded_bytes"] == hit["network_requests"] == 0
    assert hit["audio_sha256"] == result["audio_sha256"]
    assert len(state["reads"]) == 2 and len(state["asr_calls"]) == 1


@pytest.mark.parametrize("change", [
    {"metadata_checked": False}, {"source_verified": False}, {"download_status": "preview_only"},
    {"media_extractor": "unknown"}, {"duration": 29}, {"duration": 421}, {"duration": float("nan")},
    {"is_preview": True}, {"rights": {"pay": 1}}, {"aid": 1},
    {"source_url": "https://www.bilibili.com/video/BV1AT4y1n7sx"},
])
def test_ineligible_source_fails_without_network(scenario, change):
    candidate, state, _, _ = scenario
    candidate.update(change)
    assert not source_audio.eligible(candidate)
    with pytest.raises(ValueError):
        source_audio.transcript(candidate)
    assert not state["reads"]


@pytest.mark.parametrize("change", [
    {"video_id": "BV1AT4y1n7sx"}, {"cid": True}, {"aid": -1}, {"readable": False},
    {"url": "https://example.com/source"}, {"evidence_origin": "aggregator_search"},
])
def test_source_evidence_binding_is_required(scenario, change):
    candidate, state, _, _ = scenario
    candidate["source_evidence"].update(change)
    assert not source_audio.eligible(candidate)
    with pytest.raises(ValueError):
        source_audio.transcript(candidate)
    assert not state["reads"]


def test_eligible_has_no_model_or_filesystem_side_effect(scenario, monkeypatch):
    candidate, state, root, _ = scenario
    monkeypatch.setattr(source_audio, "_model_identity", lambda *_: pytest.fail("eligible loaded model"))
    assert source_audio.eligible(candidate)
    assert not state["reads"] and not list(root.iterdir())
    assert not source_audio.eligible(None)


@pytest.mark.parametrize("change", [
    {"is_preview": True}, {"is_drm": True}, {"is_upower_exclusive": True},
    {"need_login": 1}, {"rights": {"pay": 1}}, {"bvid": "BV1AT4y1n7sx"},
    {"cid": 123}, {"aid": 123}, {"timelength": 40000},
])
def test_restricted_or_inconsistent_playback_never_downloads_audio(scenario, change):
    candidate, state, root, _ = scenario
    state["metadata"]["data"].update(change)
    with pytest.raises(ValueError):
        source_audio.transcript(candidate)
    assert len(state["reads"]) == 1 and not state["asr_calls"]
    assert not list((root / "cache" / "source-audio").iterdir())


@pytest.mark.parametrize("field,value", [
    ("baseUrl", "https://evil.test/46677584/46677584_audio.m4s"),
    ("baseUrl", "https://upos.bilivideo.com/other/123/123_audio.m4s"),
    ("baseUrl", "http://upos.bilivideo.com/46677584/46677584_audio.m4s"),
    ("mimeType", "video/mp4"), ("codecs", "ec-3"), ("bandwidth", 8000),
    ("bandwidth", 319130), ("bandwidth", float("inf")), ("has_drm", True),
])
def test_only_bound_plain_aac_is_selected(scenario, field, value):
    candidate, state, _, _ = scenario
    for row in state["metadata"]["data"]["dash"]["audio"]:
        row[field] = value
    with pytest.raises(ValueError):
        source_audio.transcript(candidate)
    assert len(state["reads"]) == 1 and not state["asr_calls"]


def test_no_audio_or_error_does_not_try_alternate_api(scenario):
    candidate, state, _, _ = scenario
    state["metadata"]["code"] = -101
    with pytest.raises(ValueError):
        source_audio.transcript(candidate)
    assert len(state["reads"]) == 1
    state["metadata"]["code"] = 0
    state["metadata"]["data"]["dash"]["audio"] = []
    with pytest.raises(ValueError):
        source_audio.transcript(candidate)
    assert len(state["reads"]) == 2


def test_model_and_source_duration_change_rekey_cache(scenario):
    candidate, state, _, _ = scenario
    first = source_audio.transcript(candidate)
    state["model"] = {"weights": "model-v2"}
    second = source_audio.transcript(candidate)
    candidate["duration"] = 30.5
    third = source_audio.transcript(candidate)
    assert len({row["cache_key"] for row in (first, second, third)}) == 3
    assert len(state["reads"]) == 6 and len(state["asr_calls"]) == 3


def test_modified_cached_words_do_not_replay_as_trusted_transcript(scenario):
    candidate, state, _, _ = scenario
    first = source_audio.transcript(candidate)
    file = Path(first["cache_path"])
    record = json.loads(file.read_text())
    record["payload"]["words"][0]["text"] = "错误的文字"
    file.write_text(json.dumps(record))
    second = source_audio.transcript(candidate)
    assert not second["cache_hit"] and second["words"] == state["words"]
    assert len(state["reads"]) == 4


@pytest.mark.parametrize("words", [
    [], [{"text": "", "start": 0, "end": 1}], [{"text": "错", "start": -1, "end": 1}],
    [{"text": "错", "start": 2, "end": 1}], [{"text": "错", "start": 29, "end": 40}],
    [{"text": "错", "start": float("nan"), "end": 1}],
    [{"text": "甲", "start": 1, "end": 3}, {"text": "乙", "start": 2, "end": 4}],
])
def test_invalid_asr_timestamps_do_not_make_cache(scenario, words):
    candidate, state, root, _ = scenario
    state["words"] = words
    with pytest.raises(ValueError):
        source_audio.transcript(candidate)
    assert not list((root / "cache" / "source-audio").iterdir())


def test_oversized_audio_stops_within_shared_cap_and_cleans_temp(scenario):
    candidate, state, root, _ = scenario
    state["audio_bytes"] = b"x" * (source_audio.MAX_BYTES + 1)
    with pytest.raises(ValueError, match="预算"):
        source_audio.transcript(candidate)
    assert not state["asr_calls"]
    assert len(state["reads"]) == 2 and state["reads"][1][2].bytes <= source_audio.MAX_BYTES
    assert not list((root / "cache" / "source-audio").iterdir())


def test_cancel_during_download_propagates_and_cleans_temp(scenario, monkeypatch):
    candidate, state, root, _ = scenario
    original = segment_discovery._read

    def cancelled(url, budget, referer, *, limit, target=None, deadline_supervised=False):
        if target is not None:
            Path(target).write_bytes(b"partial")
            raise media.Cancelled("stopped")
        return original(url, budget, referer, limit=limit, target=target, deadline_supervised=deadline_supervised)

    monkeypatch.setattr(segment_discovery, "_read", cancelled)
    with pytest.raises(media.Cancelled):
        source_audio.transcript(candidate)
    assert not state["asr_calls"]
    assert not list((root / "cache" / "source-audio").iterdir())


def test_network_errors_do_not_expose_signed_urls(scenario, monkeypatch):
    candidate, _, _, _ = scenario

    def unavailable(*args, **kwargs):
        raise httpx.ReadTimeout("signed https://upos.bilivideo.com/?private-query=1")

    monkeypatch.setattr(segment_discovery, "_read", unavailable)
    with pytest.raises(ValueError) as error:
        source_audio.transcript(candidate)
    assert "http" not in str(error.value) and "private-query" not in str(error.value)


def test_deadline_during_asr_cleans_temp_and_does_not_save(scenario, monkeypatch):
    candidate, state, root, _ = scenario

    def over_deadline(file, check, progress):
        state["reads"][-1][2].deadline = 0
        progress(1, "finished one chunk")

    monkeypatch.setattr(local_asr, "transcribe", over_deadline)
    with pytest.raises(ValueError, match="时间预算"):
        source_audio.transcript(candidate)
    assert not list((root / "cache" / "source-audio").iterdir())


@pytest.mark.parametrize("phase", ["metadata", "audio"])
@pytest.mark.parametrize("stop", ["deadline", "cancel"])
def test_slow_response_headers_are_interrupted_and_workers_reclaimed(scenario, monkeypatch, phase, stop):
    """Real HTTPX/httpcore parsing over an interruptible in-memory transport.

    A header byte every 10 ms defeats a per-read timeout: absent independent
    supervision, these headers take 400 ms although the budget is only 100 ms.
    No real socket, DNS request, model request or production data is involved.
    """
    candidate, state, root, _ = scenario
    streams, budgets = [], []
    started = time.monotonic()
    cancellation = media.Cancelled("original cancellation")
    cancellation_reported = False

    def check():
        nonlocal cancellation_reported
        if stop == "cancel" and time.monotonic() - started >= .06 and not cancellation_reported:
            cancellation_reported = True
            raise cancellation  # Only once: the original exception must survive shutdown.

    class Stream:
        def __init__(self, payload, slow):
            self.closed = threading.Event()
            self.shutdown_count = 0
            headers = b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: " + str(len(payload)).encode() + b"\r\n"
            self.chunks = ([headers + b"X-Slow: ", *([b"x"] * 40), b"\r\n\r\n" + payload]
                           if slow else [headers + b"\r\n" + payload])
            self.slow, self.reads = slow, 0

        def read(self, max_bytes, timeout=None):
            if self.closed.is_set() or (self.slow and self.reads and self.closed.wait(.01)):
                raise httpcore.ReadError("fixture socket was interrupted")
            self.reads += 1
            return self.chunks.pop(0) if self.chunks else b""

        def write(self, data, timeout=None):
            pass

        def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            return self

        def get_extra_info(self, name):
            return self if name == "socket" else None

        def shutdown(self, how):
            self.shutdown_count += 1
            self.closed.set()

        def close(self):
            self.closed.set()

    class Backend:
        def connect_tcp(self, host, port, timeout=None, **kwargs):
            is_metadata = host == "api.bilibili.com"
            assert is_metadata or host == "upos-sz-estgoss.bilivideo.com"
            payload = json.dumps(state["metadata"]).encode() if is_metadata else state["audio_bytes"]
            stream = Stream(payload, ("metadata" if is_metadata else "audio") == phase)
            streams.append(stream)
            return stream

    def client(**kwargs):
        transport = httpx.HTTPTransport(trust_env=False)
        transport._pool._network_backend = Backend()
        return httpx.Client(**kwargs, transport=transport, trust_env=False)

    def read(*args, **kwargs):
        budgets.append(args[1])
        return _REAL_READ(*args, **kwargs)

    monkeypatch.setattr(network, "client", client)
    monkeypatch.setattr(sources, "public_url", lambda url: None)
    monkeypatch.setattr(sources, "download_slot", lambda url: root / "cooldown.json")
    monkeypatch.setattr(segment_discovery, "_read", read)
    monkeypatch.setattr(source_audio, "MAX_SECONDS", .1 if stop == "deadline" else 5)
    before_threads = {thread.ident for thread in threading.enumerate()}
    expected = segment_discovery.SampleLimit if stop == "deadline" else media.Cancelled
    started = time.monotonic()
    with pytest.raises(expected) as error:
        source_audio.transcript(candidate, check)
    elapsed = time.monotonic() - started
    assert elapsed < .25, f"slow response headers outlived shared deadline: {elapsed:.3f}s"
    if stop == "cancel":
        assert error.value is cancellation
    else:
        assert "时间预算" in str(error.value)
    assert len(streams) == (1 if phase == "metadata" else 2)
    assert streams[-1].shutdown_count >= 1
    assert all(stream.closed.is_set() for stream in streams)
    assert all(not budget.clients and not budget.read_sockets for budget in budgets)
    assert not state["asr_calls"] and not list((root / "cache" / "source-audio").iterdir())
    assert not [thread for thread in threading.enumerate()
                if thread.ident not in before_threads and thread.name == "source-audio-deadline"]
