from copy import deepcopy
import hashlib
import json
import socket
import subprocess
import time

import pytest

from jinggao import local_scene_candidates as scenes
from jinggao import media, network, paths


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Local scene tests must not use network or models")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(network, "post", forbidden)


def probe_data(*, duration=4, origin=0):
    return {"streams": [{"index": 0, "codec_type": "video", "width": 1280, "height": 720,
                         "start_time": str(origin), "time_base": "1/1000"}],
            "format": {"duration": str(duration), "start_time": str(origin)}}


def frame_log(times=None, *, origin=0, cuts=(2,), colours=None):
    times = times if times is not None else [i*.04 for i in range(100)]
    text = ["#tb 0: 1/1000", "#dimensions 0: 320x180"]
    for i, value in enumerate(times):
        pts = round((value+origin)*1000)
        colour = colours[i] if colours else str(sum(value >= cut for cut in cuts))
        pixel_hash = hashlib.sha256(colour.encode()).hexdigest()
        score = .6 if any(abs(value-cut) < .000001 for cut in cuts) else 0
        text.extend([f"[metadata] frame:{i} pts:{pts} pts_time:{value+origin}",
                     f"[metadata] lavfi.scene_score={score}", f"0, {pts}, {pts}, 40, 86400, {pixel_hash}"])
    return "\n".join(text)


def install(tmp_path, monkeypatch, *, probe=None, log=None, effect=None):
    file = tmp_path / "local.mp4"
    file.write_bytes(b"A local media mock with a real file SHA; transport is forbidden")
    calls = []
    monkeypatch.setattr(paths, "binary", lambda name: name)
    def run(args, *, timeout, check):
        assert timeout > 0 and timeout <= 3
        assert args[args.index("-protocol_whitelist")+1] == "file"
        assert "-threads" in args and args[args.index("-threads")+1] == "1"
        check()
        calls.append(args)
        if args[0] == "ffprobe":
            return json.dumps(probe if probe is not None else probe_data())
        assert "-copyts" in args and "-enc_time_base" in args
        assert args[args.index("-enc_time_base")+1] == "demux"
        assert args[args.index("-threads:v")+1] == "1"
        assert args[args.index("-loglevel")+1] == "error" and "-nostats" in args and "-xerror" in args
        assert args[args.index("-vf")+1].endswith("metadata=print:key=lavfi.scene_score:file=-:direct=1")
        assert args[-1] == "-" and str(file) in args
        if effect:
            effect(file, check)
        return frame_log() if log is None else log
    monkeypatch.setattr(media, "run", run)
    return file, calls


def invoke(path, **kwargs):
    return scenes.candidates(path, source_base=100, duration=4, deadline=time.monotonic()+10, **kwargs)


def test_real_pts_and_low_resolution_pixel_hash_bind_the_local_file(tmp_path, monkeypatch):
    file, calls = install(tmp_path, monkeypatch)
    originals = [100.4]
    result = invoke(file, existing_points=originals)
    assert result["status"] == "candidates" and len(calls) == 2
    assert originals == [100.4] and result["existing_points"] == originals
    assert result["source_binding"]["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert result["source_binding"]["width"] == 1280 and result["source_binding"]["height"] == 720
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["pts"] == 3000 and candidate["local_at"] == 3 and candidate["source_at"] == 103
    assert candidate["pixel_sha256"] == hashlib.sha256(b"1").hexdigest()
    assert candidate["pixel_size"] == [320, 180] and candidate["pixel_format"] == "yuv420p"
    assert candidate["local_interval"] == [2, 4]
    assert not any(key in json.dumps(result) for key in ("approved", "verified", "action", "usable_window"))


def test_native_nonzero_pts_are_normalized_from_probed_container_origin(tmp_path, monkeypatch):
    file, _ = install(tmp_path, monkeypatch, probe=probe_data(origin=5), log=frame_log(origin=5))
    result = invoke(file, existing_points=[100.4])
    candidate = result["candidates"][0]
    assert result["source_binding"]["origin"] == 5
    assert candidate["pts"] == 8000 and candidate["local_at"] == 3 and candidate["source_at"] == 103


def test_vfr_midpoint_is_not_invented_or_quantized_to_25fps(tmp_path, monkeypatch):
    times = [0, .32, .81, 1.56, 1.64, 2.03, 2.56, 3.14, 3.96]
    file, _ = install(tmp_path, monkeypatch, log=frame_log(times, cuts=(1.56,)))
    result = invoke(file, existing_points=[100.32])
    candidate = result["candidates"][0]
    assert candidate["local_at"] == 2.56 and candidate["pts"] == 2560
    assert candidate["local_at"] != (1.56+4)/2


def test_single_frame_flash_is_not_a_scene_candidate_and_duplicate_return_is_free(tmp_path, monkeypatch):
    colours = ["white" if i == 50 else "red" for i in range(100)]
    file, _ = install(tmp_path, monkeypatch, log=frame_log(cuts=(2, 2.04), colours=colours))
    result = invoke(file, existing_points=[100.5])
    assert result["status"] == "no_candidates" and result["fallback_required"]
    assert result["skipped_short_intervals"] == 1 and not result["candidates"]
    repeated = invoke(file)
    assert len(repeated["candidates"]) == 1  # Exactly identical red pixels around the flash.


def test_static_clip_does_not_consume_another_observation_slot(tmp_path, monkeypatch):
    file, _ = install(tmp_path, monkeypatch, log=frame_log(cuts=()))
    result = invoke(file)
    assert result["status"] == "no_candidates" and result["reason"] == "no_new_stable_scene"
    assert result["candidates"] == [] and result["fallback_required"]


def test_short_clip_retains_caller_fallback_without_decoding(tmp_path, monkeypatch):
    file, calls = install(tmp_path, monkeypatch, probe=probe_data(duration=.4))
    result = scenes.candidates(file, source_base=0, duration=.4, deadline=time.monotonic()+5)
    assert len(calls) == 1 and result["reason"] == "short_window" and result["fallback_required"]


def test_many_cuts_keep_at_most_four_real_interior_frames_in_source_order(tmp_path, monkeypatch):
    times = [i*.04 for i in range(200)]
    cuts = tuple(i*.8 for i in range(1,10))
    file, _ = install(tmp_path, monkeypatch, probe=probe_data(duration=8), log=frame_log(times, cuts=cuts))
    result = scenes.candidates(file, source_base=20, duration=8, deadline=time.monotonic()+5, limit=4)
    assert len(result["candidates"]) == 4
    points = [row["source_at"] for row in result["candidates"]]
    assert points == sorted(set(points)) and all(20 <= p < 28 for p in points)
    for row in result["candidates"]:
        start, end = row["local_interval"]
        assert start+scenes.EDGE_SECONDS <= row["local_at"] <= end-scenes.EDGE_SECONDS


@pytest.mark.parametrize("changes", [
    {"source_base": -1}, {"source_base": float("nan")}, {"source_base": 1e308},
    {"duration": 0}, {"duration": 41}, {"duration": True}, {"duration": None},
    {"limit": 0}, {"limit": 5}, {"limit": True}, {"max_seconds": 0}, {"max_seconds": 11},
    {"deadline": float("inf")}, {"existing_points": [99.9]}, {"existing_points": [104]},
    {"existing_points": [True]}, {"existing_points": [100.1]*25},
])
def test_invalid_caller_bounds_never_launch_a_decoder(tmp_path, monkeypatch, changes):
    file, calls = install(tmp_path, monkeypatch)
    args = {"source_base": 100, "duration": 4, "deadline": time.monotonic()+5} | changes
    with pytest.raises(ValueError):
        scenes.candidates(file, **args)
    assert calls == []


@pytest.mark.parametrize("mutation", ["low_resolution", "long_file", "wrong_duration", "no_video", "two_videos",
                                     "attached_picture", "missing_origin", "bad_timebase", "bad_stream_start"])
def test_invalid_or_unbounded_media_never_enters_scan(tmp_path, monkeypatch, mutation):
    info = probe_data()
    video = info["streams"][0]
    if mutation == "low_resolution":
        video.update(width=640, height=360)
    elif mutation == "long_file":
        info["format"]["duration"] = "3600"
    elif mutation == "wrong_duration":
        info["format"]["duration"] = "10"
    elif mutation == "no_video":
        video["codec_type"] = "audio"
    elif mutation == "two_videos":
        info["streams"].append(deepcopy(video) | {"index": 1})
    elif mutation == "attached_picture":
        video["disposition"] = {"attached_pic": 1}
    elif mutation == "missing_origin":
        info["format"].pop("start_time")
    elif mutation == "bad_timebase":
        video["time_base"] = "1/0"
    else:
        video["start_time"] = "-5"
    file, calls = install(tmp_path, monkeypatch, probe=info)
    result = invoke(file)
    assert len(calls) == 1 and result["status"] == "unavailable"
    assert result["candidates"] == [] and result["fallback_required"]


@pytest.mark.parametrize("mutation", ["truncated", "bad_score", "duplicate_metadata", "wrong_hash_pts",
                                     "bad_hash", "wrong_tb", "missing_dimensions", "out_of_range", "reverse"])
def test_malformed_frame_bindings_cannot_produce_candidates(tmp_path, monkeypatch, mutation):
    log = frame_log()
    if mutation == "truncated":
        # Missing 200 ms exceeds the normal 120 ms container/audio tail allowance.
        log = "\n".join(log.splitlines()[:-15])
    elif mutation == "bad_score":
        log = log.replace("scene_score=0.6", "scene_score=nan")
    elif mutation == "duplicate_metadata":
        log += "\n[metadata] frame:0 pts:0 pts_time:0"
    elif mutation == "wrong_hash_pts":
        log = log.replace("0, 2000, 2000,", "0, 2000, 2100,")
    elif mutation == "bad_hash":
        log = log.replace(hashlib.sha256(b"1").hexdigest(), "not-a-sha")
    elif mutation == "wrong_tb":
        log = log.replace("#tb 0: 1/1000", "#tb 0: 1/90000")
    elif mutation == "missing_dimensions":
        log = log.replace("#dimensions 0: 320x180", "#dimensions absent")
    elif mutation == "out_of_range":
        log = frame_log(origin=-1)
    else:
        log = frame_log([0, .08, .04, 3.96])
    file, _ = install(tmp_path, monkeypatch, log=log)
    result = invoke(file)
    assert result["status"] == "unavailable" and not result["candidates"]


def test_source_change_during_decode_invalidates_every_candidate(tmp_path, monkeypatch):
    file, _ = install(tmp_path, monkeypatch, effect=lambda path, check: path.write_bytes(b"different bytes"))
    result = invoke(file)
    assert result["status"] == "unavailable" and not result["candidates"]


def test_missing_file_and_non_media_input_are_fallbacks_not_network(tmp_path, monkeypatch):
    file, calls = install(tmp_path, monkeypatch)
    for path in [tmp_path / "missing.mp4", tmp_path, "https://example.invalid/remote.mp4"]:
        result = invoke(path)
        assert result["status"] == "unavailable" and result["fallback_required"]
    file = file.with_suffix(".m3u8")
    file.write_text("#EXTM3U\nhttps://example.invalid/stream.ts")
    assert invoke(file)["status"] == "unavailable"
    assert calls == []


def test_expired_deadline_never_probes_or_decodes(tmp_path, monkeypatch):
    file, calls = install(tmp_path, monkeypatch)
    result = scenes.candidates(file, source_base=0, duration=4, deadline=time.monotonic()-1)
    assert result["reason"] == "time_budget" and calls == []


def test_deadline_during_scan_discards_partial_work(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(scenes.time, "monotonic", lambda: now[0])
    def expire(file, check):
        now[0] += 3.1
        check()
    file, calls = install(tmp_path, monkeypatch, effect=expire)
    result = scenes.candidates(file, source_base=0, duration=4, deadline=110)
    assert len(calls) == 2 and result["reason"] == "time_budget" and not result["candidates"]


@pytest.mark.parametrize("error", [media.Cancelled("stop"), ValueError("caller stopped")])
def test_caller_cancellation_and_errors_propagate(tmp_path, monkeypatch, error):
    active = [False]
    def enable(file, check):
        active[0] = True
        check()
    file, _ = install(tmp_path, monkeypatch, effect=enable)
    def check():
        if active[0]:
            raise error
    with pytest.raises(type(error)) as found:
        invoke(file, check=check)
    assert found.value is error


def test_decoder_failure_does_not_retry(tmp_path, monkeypatch):
    def fail(file, check):
        raise ValueError("damaged media")
    file, calls = install(tmp_path, monkeypatch, effect=fail)
    result = invoke(file)
    assert len(calls) == 2 and result["status"] == "unavailable" and not result["candidates"]


@pytest.fixture(scope="module")
def actual_videos(tmp_path_factory):
    root = tmp_path_factory.mktemp("actual-local-scenes")
    result = {}
    for name, colours, suffix, extra, size in (
            ("cut", [("red",2),("blue",2)], ".mp4", [], "1280x720"),
            ("static", [("red",2)], ".mp4", [], "1280x720"),
            ("flash", [("red",1),("white",.04),("red",1)], ".mp4", [], "1280x720"),
            ("offset", [("red",2),("blue",2)], ".ts", ["-output_ts_offset","5"], "1280x720"),
            ("portrait", [("red",2),("blue",2)], ".mp4", [], "720x1280")):
        target = root / (name+suffix)
        args = [paths.binary("ffmpeg"), "-v", "error", "-y", "-filter_complex_threads", "1"]
        for colour, duration in colours:
            args += ["-f", "lavfi", "-i", f"color=c={colour}:s={size}:r=25:d={duration}"]
        graph = "".join(f"[{i}:v]" for i in range(len(colours)))+f"concat=n={len(colours)}:v=1:a=0[v]"
        args += ["-filter_complex", graph, "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                 "-threads", "1", *extra, str(target)]
        media.run(args, timeout=10)
        result[name] = target
    return result


def test_actual_ffmpeg_pixels_and_pts_bind_after_nonzero_source_offset(actual_videos):
    path = actual_videos["cut"]
    result = scenes.candidates(path, source_base=123.4, duration=4, deadline=time.monotonic()+10, max_seconds=8,
                               existing_points=[123.8])
    assert result["status"] == "candidates" and len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["local_at"] == 3 and candidate["source_at"] == 126.4
    assert result["source_binding"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    raw = subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-threads", "1", "-filter_threads", "1",
        "-protocol_whitelist", "file", "-i", str(path), "-vf",
        f"scale=320:-2,format=yuv420p,select=eq(n\\,{candidate['frame_index']})",
        "-frames:v", "1", "-threads:v", "1", "-f", "rawvideo", "-"], capture_output=True, check=True, timeout=10).stdout
    assert hashlib.sha256(raw).hexdigest() == candidate["pixel_sha256"]
    assert not list(path.parent.glob("*.jpg"))  # The production helper writes no previews.


def test_actual_mpegts_origin_is_not_added_twice(actual_videos):
    result = scenes.candidates(actual_videos["offset"], source_base=100, duration=4,
        deadline=time.monotonic()+10, max_seconds=8, existing_points=[100.4])
    assert result["status"] == "candidates"
    assert result["source_binding"]["origin"] > 0
    assert result["candidates"][0]["local_at"] == 3
    assert result["candidates"][0]["source_at"] == 103


def test_actual_portrait_structured_records_keep_every_frame_bound(actual_videos):
    result = scenes.candidates(actual_videos["portrait"], source_base=0, duration=4,
        deadline=time.monotonic()+10, max_seconds=8, existing_points=[.5])
    assert result["status"] == "candidates" and result["decoded_frames"] == 100
    assert result["source_binding"]["width"] == 720 and result["source_binding"]["height"] == 1280
    assert result["candidates"][0]["pixel_size"] == [320, 568]
    assert result["candidates"][0]["source_at"] == 3


@pytest.mark.parametrize("name,duration", [("static",2),("flash",2.04)])
def test_actual_static_and_one_frame_flash_use_original_fallback(actual_videos, name, duration):
    result = scenes.candidates(actual_videos[name], source_base=0, duration=duration,
        deadline=time.monotonic()+10, max_seconds=8, existing_points=[.5])
    assert result["status"] == "no_candidates" and result["fallback_required"]
    assert not result["candidates"]
