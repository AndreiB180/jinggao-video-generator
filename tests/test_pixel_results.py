import copy
import json

from PIL import Image
import pytest

from jinggao import config, neutral_observer, paths, pixel_results, visual_quality
from test_neutral_observer import group
from test_visual_quality import frame, sticker


@pytest.fixture
def records(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: {"vision_model": "test", "model_api_key": "test",
                                               "model_base_url": "https://model.test"})
    result = []
    for i, color in enumerate(("red", "blue")):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (64, 64), color).save(path)
        result.append({"shot": {"id": f"shot{i}"}, "frame_paths": [path]})
    return result


@pytest.mark.parametrize("change", ["missing", "invalid_frame", "outside_array"])
def test_neutral_failure_keeps_only_the_complete_group_without_retry(records, monkeypatch, tmp_path, change):
    body = {"groups": [group("g0"), group("g1")]}
    if change == "missing":
        body["groups"].pop()
    elif change == "invalid_frame":
        body["groups"][1]["frames"][0].pop("action")
    else:
        body.update(body["groups"].pop())
    calls = []
    def request(content, cfg, check):
        counts = json.loads(content[0]["text"])["observation_groups"]
        calls.append(counts)
        return json.dumps(body if len(calls) == 1 else {"groups": [group("g0")]}), {"finish_reason": "stop"}
    monkeypatch.setattr(neutral_observer, "_request", request)
    with pytest.raises(neutral_observer.ObservationError) as error:
        neutral_observer.observe(records, lambda: None)
    assert error.value.request_count == 1 and len(calls) == 1
    assert len(list(tmp_path.glob("cache/neutral-observer/*.json"))) == 1
    cached, sent = neutral_observer.observe(records[:1], lambda: None)
    assert sent == 0 and cached["shot0"] == {k: v for k, v in group().items() if k != "id"}
    rows, sent = neutral_observer.observe(records, lambda: None)
    assert sent == 1 and len(rows) == 2
    assert len(calls) == 2 and len(calls[1]) == 1


def test_quality_failure_keeps_complete_sibling_and_bad_group_stays_unknown(records, monkeypatch, tmp_path):
    bad = {"id": "g0", "frames": [frame(0, overlay=sticker())]}
    bad["frames"][0]["overlays"][0]["occludes"] = ""
    good = {"id": "g1", "frames": [frame()]}
    calls = []
    def request(content, cfg, check, deadline):
        calls.append(content)
        return json.dumps({"groups": [bad, good]}), {"finish_reason": "stop"}
    monkeypatch.setattr(visual_quality, "_request", request)
    rows, info = visual_quality.observe(records, lambda: None)
    assert info["model_requests"] == 1 and len(calls) == 1
    assert rows["shot0"]["quality_status"] == "uncertain" and rows["shot0"]["frames"] == []
    assert rows["shot1"]["quality_status"] == "not_detected" and not rows["shot1"]["content_approved"]
    assert len(list(tmp_path.glob("cache/visual-quality/*.json"))) == 1
    cached, info = visual_quality.observe(records[1:], lambda: None)
    assert info["model_requests"] == 0 and cached["shot1"]["cached"]


@pytest.mark.parametrize("observer", [neutral_observer, visual_quality])
def test_truncated_provider_result_does_not_preserve_even_well_formed_groups(records, monkeypatch, tmp_path, observer):
    data = [group("g0"), group("g1")] if observer is neutral_observer else [
        {"id": f"g{i}", "frames": [frame()]} for i in range(2)]
    monkeypatch.setattr(observer, "_request", lambda *a: (json.dumps({"groups": data}), {"finish_reason": "length"}))
    if observer is neutral_observer:
        with pytest.raises(neutral_observer.ObservationError):
            observer.observe(records, lambda: None)
    else:
        rows, _ = observer.observe(records, lambda: None)
        assert all(row["quality_status"] == "uncertain" for row in rows.values())
    assert not list(tmp_path.glob("cache/**/*.json"))


@pytest.mark.parametrize("change", ["duplicate", "foreign", "contradictory", "invalid_id", "trailing", "truncated"])
def test_ambiguous_or_incomplete_json_never_preserves_siblings(change):
    body = {"groups": [group("g0"), group("g1")]}
    if change == "duplicate":
        body["groups"][1]["id"] = "g0"
    elif change == "foreign":
        body["groups"][1]["id"] = "g2"
    elif change == "contradictory":
        body["groups"][1]["group"] = "g0"
    elif change == "invalid_id":
        body["groups"][1]["id"] = []
    raw = json.dumps(body)
    raw = raw + " trailing" if change == "trailing" else raw[:-1] if change == "truncated" else raw
    untouched = copy.deepcopy(body)
    assert pixel_results.complete_groups(raw, {"g0": 1, "g1": 1}, neutral_observer.parse_groups) == {}
    assert body == untouched


@pytest.mark.parametrize("observer", [neutral_observer, visual_quality])
@pytest.mark.parametrize("field", ["id", "frame_index", "groups"])
def test_duplicate_object_fields_are_rejected_before_overwriting_evidence(observer, field):
    row = group() if observer is neutral_observer else {"id": "g0", "frames": [frame()]}
    raw = json.dumps({"groups": [row]})
    if field == "id":
        raw = raw.replace('"id": "g0"', '"id": "foreign", "id": "g0"')
    elif field == "frame_index":
        raw = raw.replace('"frame_index": 0', '"frame_index": 7, "frame_index": 0')
    else:
        raw = raw.replace('{"groups":', '{"groups": [], "groups":', 1)
    with pytest.raises(ValueError):
        observer.parse_groups(raw, {"g0": 1})
    assert pixel_results.complete_groups(raw, {"g0": 1, "g1": 1}, observer.parse_groups) == {}
