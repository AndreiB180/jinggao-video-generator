import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from jinggao import captions, config, local_asr, media, paths, planner, pooled_match, store, worker
from test_curation import completed_pixel_review


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    p = store.new_project("真实时间戳", "你好。世界。你好。", media_policy="mixed")
    shots = [planner.new_shot(text) for text in ("你好。", "世界。", "你好。")]
    for shot in shots:
        shot.update(duration=2, selected_asset="a", approved=True, verification={"verdict": "match"},
                    match_preview={"duration": 2}, source_review_note="人工复核", source_review_url="https://example.org")
        shot["candidates"] = [{"id": "candidate", "preview": {"duration": 2}, "preview_paths": ["old.jpg"],
                               "preflight_status": "reviewed", "preflight_reason": "old window"}]
    (paths.project_dir(p["id"]) / "voice.wav").write_bytes(b"saved full audio")
    return store.mutate(p["id"], lambda p: p.update(shots=shots, assets=[{"id": "a", "kind": "image"}],
                        full_audio={"path": "voice.wav", "duration": 6, "alignment": "pending"}))


def save_track(project, *, starts=(.1, 1.04, 3.04), ends=(1, 3, 5.9)):
    *_, binding, target = captions._source(project)
    track = {"binding": binding, "duration": project["full_audio"]["duration"], "method": "local_paraformer", "coverage": 1,
             "cues": [{"text": s["narration"], "start": begin, "end": end}
                      for s, begin, end in zip(project["shots"], starts, ends, strict=True)]}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(track, ensure_ascii=False))
    return track, target


def test_installing_new_full_audio_invalidates_all_old_window_and_identity_evidence(project):
    original = copy.deepcopy(project)
    for shot in project["shots"]:
        shot["match_result"] = {"verdict": "match"}
        shot["verification"]["identity_evidence"] = {"status": "supported"}
    media.align_narration(project, {"path": "replacement.wav", "duration": 6})
    assert project["full_audio"]["alignment"] == "pending"
    assert [s["duration"] for s in project["shots"]] == [s["duration"] for s in original["shots"]]
    for shot in project["shots"]:
        assert not shot["approved"] and shot["verification"] is None
        assert not shot["source_review_note"] and not shot["source_review_url"]
        assert "match_preview" not in shot and "match_result" not in shot
        assert shot["candidates"] == [{"id": "candidate"}]
        assert shot["selected_asset"] == "a"


def test_single_shot_uses_measured_audio_total_but_cannot_retain_old_approval(project):
    project["shots"] = project["shots"][:1]
    media.align_narration(project, {"path": "replacement.wav", "duration": 6.011})
    assert project["shots"][0]["duration"] == 6.04
    assert not project["shots"][0]["approved"] and project["shots"][0]["verification"] is None
    assert project["full_audio"]["alignment"] == "pending"


def test_legacy_cues_use_ordered_text_offsets_and_invalidate_only_changed_windows(project, monkeypatch):
    original = copy.deepcopy(project)
    save_track(project)
    monkeypatch.setattr(local_asr, "transcribe", lambda *a: pytest.fail("exact cues need no ASR"))
    proposal = media.prepare_narration_alignment(project)
    media.apply_narration_alignment(project, proposal)
    assert [s["duration"] for s in project["shots"]] == [1.04, 2, 2.96]
    assert [s["narration"] for s in project["shots"]] == [s["narration"] for s in original["shots"]]
    assert project["shots"][1] == original["shots"][1]
    assert "字幕句段边界" in project["full_audio"]["note"]
    for index in (0, 2):
        shot = project["shots"][index]
        assert not shot["approved"] and shot["verification"] is None and "match_preview" not in shot
        assert shot["selected_asset"] == "a" and not shot["source_review_note"]
        assert shot["candidates"] == [{"id": "candidate"}]
    media.require_narration_alignment(project)
    snapshot = copy.deepcopy(project)
    media.apply_narration_alignment(project, media.prepare_narration_alignment(project))
    assert project == snapshot


def test_lock_rejects_changed_absolute_position_even_when_duration_unchanged(project):
    project["shots"][1]["locked"] = True
    save_track(project)
    snapshot = copy.deepcopy(project)
    with pytest.raises(ValueError, match="锁定镜头"):
        media.apply_narration_alignment(project, media.prepare_narration_alignment(project))
    assert project == snapshot


def test_lock_at_unchanged_absolute_interval_is_preserved(project):
    project["shots"][0]["locked"] = True
    save_track(project, starts=(.1, 2, 3.04), ends=(1.9, 3, 5.9))
    expected = copy.deepcopy(project["shots"][0])
    media.apply_narration_alignment(project, media.prepare_narration_alignment(project))
    assert project["shots"][0] == expected


@pytest.mark.parametrize("change", ["audio", "narration", "duration", "locked", "order"])
def test_commit_rechecks_audio_bytes_and_shot_inputs(project, change):
    save_track(project)
    proposal = media.prepare_narration_alignment(project)
    if change == "audio":
        paths.file_path(project["id"], "voice.wav").write_bytes(b"different audio")
    elif change == "order":
        project["shots"].reverse()
    else:
        project["shots"][0][change] = {"narration": "改文", "duration": 3, "locked": True}[change]
    snapshot = copy.deepcopy(project)
    with pytest.raises(ValueError, match="发生变化"):
        media.apply_narration_alignment(project, proposal)
    assert project == snapshot


def test_internal_cue_cut_upgrades_cache_once_and_keeps_observed_word_edges(project, monkeypatch):
    project["shots"] = [planner.new_shot("你好，"), planner.new_shot("世界。")]
    project["full_audio"]["duration"] = 4
    *_, binding, target = captions._source(project)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"binding": binding, "duration": 4, "method": "local_paraformer",
                                 "cues": [{"text": "你好，世界。", "start": .2, "end": 3.8}]}))
    calls = []
    def transcribe(*args):
        calls.append(1)
        return [{"text": "你好", "start": .2, "end": 1}, {"text": "世界", "start": 2.4, "end": 3.8}]
    monkeypatch.setattr(local_asr, "transcribe", transcribe)
    monkeypatch.setattr(media, "run", lambda *a, **k: None)
    timing = captions.ensure_shot_timing(project)
    assert timing["durations"] == [2.4, 1.6]
    assert timing["boundaries"][0]["method"] == "token_edges"
    assert captions.ensure_shot_timing(project) == timing and calls == [1]


def test_multi_character_token_interior_is_not_an_acoustic_boundary():
    p = {"shots": [{"narration": "你好"}, {"narration": "世界。"}], "full_audio": {"duration": 4}}
    track = captions.align("你好世界。", [{"text": "你好世界", "start": .1, "end": 3.9}], 4) | {"duration": 4}
    with pytest.raises(ValueError, match="没有实测"):
        captions.shot_timing(p, track)


def test_recognition_gap_interpolation_cannot_be_used_as_shot_boundary():
    text = "一二三四五六七八九十" 
    p = {"shots": [{"narration": text[:5]}, {"narration": text[5:]}], "full_audio": {"duration": 6}}
    words = [{"text": ch, "start": .1+i*.5, "end": .5+i*.5} for i, ch in enumerate(text) if i != 4]
    track = captions.align(text, words, 6) | {"duration": 6}
    with pytest.raises(ValueError, match="没有实测"):
        captions.shot_timing(p, track)


def test_single_character_asr_substitution_keeps_observed_edges_without_inflating_coverage():
    text = "一二三四五六七八九十"
    p = {"shots": [{"narration": text[:4]}, {"narration": text[4:]}], "full_audio": {"duration": 6}}
    words = [{"text": ch, "start": .1+i*.5, "end": .5+i*.5} for i, ch in enumerate(text.replace("五", "伍"))]
    track = captions.align(text, words, 6) | {"duration": 6}
    assert track["coverage"] == .9 and track["matched_characters"] == 9
    assert "".join(c["text"] for c in track["cues"]) == text
    timing = captions.shot_timing(p, track)
    edge = timing["boundaries"][0]
    assert edge["right_start"] == 2.1 and edge["frame"] == 53
    assert edge["recognition_substitutions"] == [{"character_offset": 4, "original": "五", "recognized": "伍",
            "token_index": 4, "start": 2.1, "end": 2.5, "method": "isolated_token_substitution", "version": 2}]


@pytest.mark.parametrize("case", ["multi_character_token", "two_replacements", "short_context", "overlap", "long_token"])
def test_substitution_without_isolated_observed_token_evidence_cannot_prove_a_cut(case):
    text = "一二三四五六七八九十甲乙丙丁戊己庚辛壬癸"
    words = [{"text": ch, "start": .1+i*.5, "end": .5+i*.5} for i, ch in enumerate(text.replace("五", "伍"))]
    offset = 4
    if case == "multi_character_token":
        words[4:6] = [{"text": "伍六", "start": 2.1, "end": 3}]
    elif case == "two_replacements":
        words[5]["text"] = "陆"
    elif case == "short_context":
        words[2]["text"] = "叁"
    elif case == "overlap":
        words[4]["start"] = 1.9
    else:
        # Leave enough silence around an implausibly long individual token;
        # it must not become an observed cut merely because its text matches.
        words[4].update(start=2.05, end=2.75)
        for word in words[5:]:
            word["start"] += .7
            word["end"] += .7
    track = captions.align(text, words, 12) | {"duration": 12}
    assert not track["character_timing"]["spans"][offset][2]
    p = {"shots": [{"narration": text[:offset]}, {"narration": text[offset:]}], "full_audio": {"duration": 12}}
    with pytest.raises(ValueError, match="没有实测"):
        captions.shot_timing(p, track)


def test_real_tea_31_shot_asr_replay_uses_original_token_time_and_preserves_narration():
    fixture = json.loads((Path(__file__).parent / "fixtures/tea_narration_timing.json").read_text())
    p = {"shots": fixture["shots"], "full_audio": {"duration": fixture["duration"]}}
    text = "".join(s["narration"] for s in p["shots"])
    track = captions.align(text, fixture["words"], fixture["duration"]) | {"duration": fixture["duration"]}
    assert track["coverage"] == .9748 and track["matched_characters"] == 1276
    assert "".join(c["text"] for c in track["cues"]) == captions.clean(text)
    timing = captions.shot_timing(p, track)
    assert len(timing["durations"]) == 31 and len(timing["boundaries"]) == 30
    assert timing["frames"][-1] == 7398
    edge = timing["boundaries"][3]
    assert (edge["character_offset"], edge["right_start"], edge["frame"]) == (193, 43.33, 1083)
    assert edge["recognition_substitutions"][0]["recognized"] == "新"
    assert all(edge["method"] == "token_edges" for edge in timing["boundaries"])


def test_failing_old_character_cache_upgrades_once_without_changing_audio(project, monkeypatch):
    text = "一二三四五六七八九十"
    project["shots"] = [planner.new_shot(text[:4]), planner.new_shot(text[4:])]
    words = [{"text": ch, "start": .1+i*.5, "end": .5+i*.5} for i, ch in enumerate(text.replace("五", "伍"))]
    old = captions.align(text, words, 6)
    old.pop("alignment_version")
    old.pop("observed_substitutions")
    old["character_timing"]["spans"][4][2:] = [False, False]
    *_, binding, target = captions._source(project)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(old | {"binding": binding, "duration": 6}))
    calls = []
    monkeypatch.setattr(local_asr, "transcribe", lambda *a: calls.append(1) or words)
    monkeypatch.setattr(media, "run", lambda *a, **k: None)
    monkeypatch.setattr(media, "synthesize", lambda *a, **k: pytest.fail("must reuse audio"))
    timing = captions.ensure_shot_timing(project)
    assert captions.ensure_shot_timing(project) == timing and calls == [1]
    cached = captions.cached(project)
    assert cached["alignment_version"] == 2 and cached["observed_words"] == words
    assert cached["binding"] == binding


def test_new_alignment_does_not_retry_asr_when_observed_boundary_is_still_missing(project, monkeypatch):
    text = "一二三四五六七八九十"
    project["shots"] = [planner.new_shot(text[:4]), planner.new_shot(text[4:])]
    words = [{"text": ch, "start": .1+i*.5, "end": .5+i*.5} for i, ch in enumerate(text) if i != 4]
    *_, binding, target = captions._source(project)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(captions.align(text, words, 6) | {"binding": binding, "duration": 6}))
    monkeypatch.setattr(local_asr, "transcribe", lambda *a: pytest.fail("new policy already attempted"))
    with pytest.raises(ValueError, match="没有实测"):
        captions.ensure_shot_timing(project)


@pytest.mark.parametrize("bad", ["wrong_text", "overlap", "nan", "short", "long", "wrong_duration"])
def test_invalid_timing_never_changes_project(project, bad):
    track, target = save_track(project)
    if bad == "wrong_text":
        track["cues"][1]["text"] = "别的原稿。"
    elif bad == "overlap":
        track["cues"][1]["start"] = .5
    elif bad == "nan":
        track["cues"][1]["start"] = float("nan")
    elif bad == "short":
        track["cues"][0]["end"] = .3
        track["cues"][1]["start"] = .5
    elif bad == "long":
        project["full_audio"]["duration"] = track["duration"] = 130
    else:
        track["duration"] = 5
    target.write_text(json.dumps(track))
    snapshot = copy.deepcopy(project)
    with pytest.raises(ValueError):
        media.prepare_narration_alignment(project)
    assert project == snapshot and paths.file_path(project["id"], "voice.wav").exists()


def test_absolute_frame_rounding_does_not_accumulate_roundup():
    p = {"shots": [{"narration": text} for text in ("甲。", "乙。", "丙。")], "full_audio": {"duration": 4.011}}
    track = {"duration": 4.011, "method": "local_paraformer", "cues": [
        {"text": "甲。", "start": .1, "end": 1.3}, {"text": "乙。", "start": 1.333, "end": 2.6},
        {"text": "丙。", "start": 2.667, "end": 4}]}
    timing = captions.shot_timing(p, track)
    assert timing["frames"] == [0, 33, 67, 101]
    assert timing["durations"] == [1.32, 1.36, 1.36]


def test_nonfinite_existing_duration_cannot_bypass_lock_or_approval(project):
    save_track(project)
    project["shots"][0].update(duration=float("inf"), locked=True)
    snapshot = copy.deepcopy(project)
    with pytest.raises(ValueError, match="当前分镜时长无效"):
        media.apply_narration_alignment(project, media.prepare_narration_alignment(project))
    assert project == snapshot
    with pytest.raises(ValueError, match="尚未按真实"):
        media.require_narration_alignment(project)


def test_real_c919_34_shot_cache_replays_without_asr():
    fixture = json.loads((Path(__file__).parent / "fixtures/c919_narration_timing.json").read_text())
    p = {"shots": fixture["shots"], "full_audio": {"duration": fixture["audio_duration"]}}
    timing = captions.shot_timing(p, fixture["track"])
    assert len(timing["durations"]) == 34 and len(timing["boundaries"]) == 33
    assert timing["frames"][-1] == 6616
    assert min(timing["durations"]) == 3 and max(timing["durations"]) == 11.44
    assert sum(abs(s["duration"]-d) > 1e-6 for s, d in zip(p["shots"], timing["durations"], strict=True)) == 32
    assert all(edge["method"] == "caption_edges" for edge in timing["boundaries"])


def test_identical_caption_generation_is_singleflight(project, monkeypatch):
    calls = []
    monkeypatch.setattr(media, "run", lambda *a, **k: None)
    def transcribe(*a):
        calls.append(1)
        return [{"text": char, "start": i*.8+.1, "end": i*.8+.7} for i, char in enumerate("你好世界你好")]
    monkeypatch.setattr(local_asr, "transcribe", transcribe)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: captions.ensure(project), range(4)))
    assert calls == [1] and all(result == results[0] for result in results)


@pytest.mark.parametrize("kind", ["match", "verify", "select", "preview", "produce", "render"])
def test_worker_stops_before_paid_review_when_local_alignment_fails(project, monkeypatch, kind):
    def fail(*a, **k):
        raise ValueError("未识别到可与原稿对齐的语音")
    monkeypatch.setattr(captions, "ensure", fail)
    monkeypatch.setattr(planner, "verify", lambda *a: pytest.fail("no paid review before alignment"))
    monkeypatch.setattr(pooled_match, "review_batch", lambda *a: pytest.fail("no retained review before alignment"))
    monkeypatch.setattr(pooled_match, "match", lambda *a: pytest.fail("no matching before alignment"))
    monkeypatch.setattr(media, "synthesize", lambda *a: pytest.fail("reuse saved audio"))
    job = store.enqueue(project["id"], kind, {"shot_id": project["shots"][0]["id"], "replace": True})
    with pytest.raises(ValueError, match="未识别到"):
        worker.process(job)
    result = store.get_project(project["id"])
    assert result["full_audio"] == project["full_audio"] and result["shots"] == project["shots"]
    assert paths.file_path(project["id"], "voice.wav").exists()


def test_single_verify_uses_aligned_duration_and_preserves_other_shots(project, monkeypatch):
    save_track(project)
    calls = []
    monkeypatch.setattr(planner, "verify", lambda pid, shot, asset:
                        calls.append(shot["duration"]) or {"verdict": "match"})
    worker.process(store.enqueue(project["id"], "verify", {"shot_id": project["shots"][0]["id"]}))
    result = store.get_project(project["id"])
    assert calls == [1.04] and result["shots"][1] == project["shots"][1]


def test_produce_aligns_before_retained_review_and_reuses_cache_on_resume(project, monkeypatch):
    def reviewed(p):
        for shot in p["shots"]:
            shot["verification"] = completed_pixel_review()
    store.mutate(project["id"], reviewed)
    save_track(project)
    monkeypatch.setattr(local_asr, "transcribe", lambda *a: pytest.fail("use cached cues"))
    seen = []
    def review(p, rows, check):
        seen.extend(row["shot"]["duration"] for row in rows)
        return [row | {"preview": {}, "verification": completed_pixel_review()} for row in rows], [], {}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: pytest.fail("lifecycle test must not call a model"))
    monkeypatch.setattr(pooled_match, "match", lambda *a, **kw: pytest.fail("retained selections passed current review"))
    monkeypatch.setattr(media, "render", lambda *a: {"path": "video.mp4"})
    config.update({"model_api_key": "test-only"})
    worker.process(store.enqueue(project["id"], "produce"))
    assert seen == [1.04, 2.96]
    assert "align" in store.get_project(project["id"])["workflow_info"]["stages"]
    seen.clear()
    worker.process(store.enqueue(project["id"], "produce"))
    assert seen == []


def test_full_audio_render_uses_validated_cached_clock_without_running_asr(project, monkeypatch):
    from PIL import Image
    from test_workflow import audio_bytes
    root = paths.project_dir(project["id"])
    (root / "voice.wav").write_bytes(audio_bytes(6))
    source = root / "source.png"
    Image.new("RGB", (1280, 720), (30, 70, 120)).save(source)
    asset = media.inspect_asset(project["id"], source, "测试画面")
    project["assets"] = [asset]
    save_track(project)
    media.apply_narration_alignment(project, media.prepare_narration_alignment(project))
    for shot in project["shots"]:
        shot.update(selected_asset=asset["id"], approved=True, locked=True)
    monkeypatch.setattr(captions, "ensure", lambda *a, **k: pytest.fail("full-audio clock is already validated"))
    result = media.render(project, store.uid(), {"resolution": "720p", "image_motion": False},
                          lambda *_: None, lambda: None)
    output = paths.file_path(project["id"], result["path"])
    assert abs(float(media.probe(output)["format"]["duration"])-6) < .05
    subtitle = output.with_name("subtitles.srt").read_text(encoding="utf-8-sig")
    assert "00:00:01,040" in subtitle and "00:00:03,040" in subtitle
