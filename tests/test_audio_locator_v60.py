"""Speech earns an image probe, never a visual fact or a selected window."""
from copy import deepcopy

import pytest

from jinggao import audio_locator, media, pooled_match, segment_discovery, source_audio


def words(text, at, step=.2):
    return [{"text": c, "start": at+i*step, "end": at+(i+1)*step} for i, c in enumerate(text)]


def test_concrete_speech_probes_replace_uniform_points_without_extra_images():
    transcript = [*words("大家好今天我们来聊一聊", 0), *words("先检查轮胎看看轮胎有没有漏气", 21),
                  *words("然后开始检查刹车压住刹车看看是否有效", 92),
                  *words("骑车记得戴稳头盔系好头盔", 160)]
    shots = [{"id": "s1", "narration": "看看轮胎，再试一试刹车。"},
             {"id": "s2", "narration": "头盔戴稳。"}]
    before = deepcopy((transcript, shots))
    points, evidence = audio_locator.locate(transcript, shots, 240)
    assert len(points) == len(set(points)) == 4
    assert any(21 <= at <= 25 for at in points)
    assert any(92 <= at <= 97 for at in points)
    assert any(160 <= at <= 166 for at in points)
    assert set(points) & set(segment_discovery._points(240, 4))
    assert len(evidence) == 3 and not any("approved" in r for r in evidence)
    assert before == (transcript, shots)


def test_unrelated_or_only_connective_speech_does_not_replace_sampling():
    assert audio_locator.locate(words("大家我们自己然后可以一个这个", 12),
                                [{"id": "s", "narration": "我们可以自己看看轮胎。"}], 60) == ([], [])
    assert audio_locator.locate(words("检查自行车刹车", 12),
                                [{"id": "s", "narration": "工人搬运快递。"}], 60) == ([], [])


def test_intro_filler_does_not_crowd_out_rare_concrete_compounds():
    transcript = [*words("首先分享给大家小小的常识因为我们的车子", 3),
                  *words("检查刹车看看刹车有没有失效", 39),
                  *words("外胎出现老化开裂就要更换外胎", 83)]
    points, evidence = audio_locator.locate(transcript,
        [{"id": "s", "narration": "准备骑自己的车出门，看看轮胎，再试一试刹车。"}], 150)
    assert any(39 <= at <= 45 for at in points)
    assert any(83 <= at <= 89 for at in points)
    assert not any(e["at"] < 12 for e in evidence)


def test_source_selection_is_bounded_and_keeps_explicit_probes(monkeypatch):
    monkeypatch.setattr(source_audio, "eligible", lambda c: True)
    shots = [{"id": "s", "narration": "检查轮胎和刹车。"}]
    candidates = [{"id": str(i), "title": "自行车刹车轮胎检查"} for i in range(6)]
    candidates[0]["sample_points"] = [10, 20, 30, 40]
    before = deepcopy(candidates)
    plan = audio_locator.candidates([(shots, candidates), (shots, candidates[:2])])
    assert len(plan) == 2 and "0" not in plan
    assert all(v == shots for v in plan.values()) and candidates == before


def test_source_selection_uses_bound_search_hints_for_narration_paraphrases(monkeypatch):
    from jinggao import editorial_contract
    monkeypatch.setattr(source_audio, "eligible", lambda c: True)
    narration = "扶住车把，看看轮胎，再试一试刹车。"
    project = {"script": narration, "requirements": ""}
    shots = [{"id": "s", "narration": narration, "visual_origin": "generated"}]
    bundle = editorial_contract.compile_responses(project, shots, [{"contracts": [{
        "ref": "s1", "use": "illustration", "refs": [], "facts": [], "must": [], "queries": []}]}])
    shots = editorial_contract.attach(project, shots, bundle)
    shots[0]["retrieval_hints"] = {"source_checksum": shots[0]["editorial_contract"]["checksum"],
                                   "queries": ["自行车检查"]}
    candidate = {"id": "c", "title": "车子哪些部位最需要检查"}
    before = deepcopy((shots, candidate))
    assert audio_locator.candidates([(shots, [candidate])]) == {"c": shots}
    assert (shots, candidate) == before
    # Unbound hints must not silently displace actual narrative context.
    shots[0]["retrieval_hints"]["source_checksum"] = "stale"
    assert audio_locator.candidates([(shots, [candidate])]) == {}


def test_index_failure_falls_back_without_swallowing_cancellation(monkeypatch):
    candidate = {"id": "c", "source_url": "https://www.bilibili.com/video/BV1Ns411E7Mw", "duration": 90}
    def unavailable(*args):
        raise ValueError("公开音轨不可读")
    monkeypatch.setattr(source_audio, "transcript", unavailable)
    result, info = audio_locator.refine(candidate, [], lambda: None)
    assert result is candidate and info["status"] == "uniform" and info["model_requests"] == 0
    def cancelled(*args):
        raise media.Cancelled("stop")
    monkeypatch.setattr(source_audio, "transcript", cancelled)
    with pytest.raises(media.Cancelled):
        audio_locator.refine(candidate, [], lambda: None)


def test_speech_stays_out_of_blind_observation_and_ranking(monkeypatch):
    shots = [{"id": "s", "narration": "检查轮胎和刹车。"}]
    candidate = {"id": "c", "source_url": "https://www.bilibili.com/video/BV1Ns411E7Mw"}
    project = {"shots": shots}
    marker = "仅用于查时间的语音，不是视觉事实"
    monkeypatch.setattr(audio_locator, "candidates", lambda rows: {"c": shots})
    monkeypatch.setattr(audio_locator, "refine", lambda c, s, ck: (
        c | {"sample_points": [20, 40, 60, 80]}, {"hints": [marker], "downloaded_bytes": 123}))
    sampled = []
    def sample(c, root, check, **kw):
        sampled.append((deepcopy(c), kw))
        assert marker not in repr(c)
        return {"candidate_id": "c", "frames": [{"id": "f0"}], "downloaded_bytes": 12}
    monkeypatch.setattr(segment_discovery, "sample_candidate", sample)
    def observe(samples, *args):
        assert marker not in repr(samples)
        return {}, {"model_requests": 0, "failures": []}
    monkeypatch.setattr(segment_discovery, "observe_samples", observe)
    monkeypatch.setattr(pooled_match, "shared_observed_samples", lambda s, p, ck: (s, []))
    monkeypatch.setattr(segment_discovery, "shared_rank_input_bound", lambda *args: lambda shots: 100)
    def rank(s, samples, *args, **kwargs):
        assert marker not in repr((s, samples, kwargs))
        return {"proposals": {"s": []}, "model_requests": 0}
    monkeypatch.setattr(segment_discovery, "rank_windows_batch", rank)
    proposals, _, info = pooled_match.discover_windows(project, [(shots, [candidate])], lambda: None,
        lambda *args: None, workers=1, audio_locate=True)
    assert proposals == {"s": []}
    assert sampled[0][1]["count"] == 4 and sampled[0][1]["sample_points"] == [20, 40, 60, 80]
    assert info["audio_location"][0]["hints"] == [marker]
    assert not any(s.get("selected_asset") for s in shots)
