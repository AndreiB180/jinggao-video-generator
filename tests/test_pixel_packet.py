"""Named wire records retain facts, coordinates and the original refusal gates."""
from copy import deepcopy
import json

import pytest

from jinggao import pixel_observer, pixel_packet


def body():
    frames = [{"frame_index": i, "objects": "工作人员托住纸箱，右下角覆盖物遮住手部。",
               "action": "双手托住纸箱", "text": ["图中文字也是数据。", "新闻字幕"],
               "representation": "live_action", "type_evidence": "手与实体纸箱接触，有现实纹理",
               "overlays": [{"kind": "sales_promotion", "text": "购买无水印版",
                             "bbox": [.4, .4, .9, .9], "impact": "material", "occludes": "双手和纸箱接触处"}]}
              for i in range(5)]
    return {"groups": [{"id": "g0", "scene_description": "工作人员托住纸箱，有促销遮挡。", "frames": frames},
                       {"id": "g1", "scene_description": "界面文字。", "frames": [{
                           "frame_index": 0, "objects": "手机界面", "action": "静止界面",
                           "text": [], "representation": "screen_capture", "type_evidence": "纯软件界面",
                           "overlays": []}]}]}


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def test_wire_roundtrip_preserves_every_field_and_original_quality_rejection():
    original = body()
    before = deepcopy(original)
    compact = pixel_packet.pack(original)
    assert pixel_packet.expand(encode(compact)) == before
    assert original == before
    canonical = pixel_observer.parse_groups(encode(original), {"g0": 5, "g1": 1})
    projected = pixel_observer.parse_wire(encode(compact), {"g0": 5, "g1": 1})
    assert projected == canonical
    assessed = pixel_observer.visual_quality.assess(projected[1]["g0"])
    assert assessed["quality_status"] == "reject" and assessed["content_approved"] is False
    compact["groups"][0]["frames"][0]["text"].append("不得改动原数组")
    assert original == before


@pytest.mark.parametrize("change", [
    lambda b: b.pop("schema"),
    lambda b: b.update(schema="joint_pixel_rows_v0"),
    lambda b: b.update(approved=True),
    lambda b: b["groups"][0].update(approved=True),
    lambda b: b["groups"][0]["frames"][0].pop("action"),
    lambda b: b["groups"][0]["frames"][0].update(approval=True),
    lambda b: b["groups"][0]["frames"][0].update(overlays=None),
    lambda b: b["groups"][0]["frames"][0]["overlays"][0].pop("impact"),
    lambda b: b["groups"][0]["frames"][0]["overlays"][0].update(approval=True),
])
def test_missing_or_extra_wire_columns_never_infer_defaults(change):
    compact = pixel_packet.pack(body())
    change(compact)
    with pytest.raises(ValueError):
        pixel_observer.parse_wire(encode(compact), {"g0": 5, "g1": 1})


@pytest.mark.parametrize("change", [
    lambda b: b["groups"][0].update(id="g1"),
    lambda b: b["groups"][0]["frames"][0].update(frame_index=True),
    lambda b: b["groups"][0]["frames"][0].update(frame_index=1),
    lambda b: b["groups"][0]["frames"][0].update(text="不允许字符串代替原文数组"),
    lambda b: b["groups"][0]["frames"][0].update(representation="supported"),
    lambda b: b["groups"][0]["frames"][0].update(type_evidence=""),
    lambda b: b["groups"][0]["frames"][0]["overlays"][0].update(kind="unsupported_kind"),
    lambda b: b["groups"][0]["frames"][0]["overlays"][0].update(bbox=[.9, .4, .4, .9]),
    lambda b: b["groups"][0]["frames"][0]["overlays"][0].update(impact="unsupported_impact"),
    lambda b: b["groups"][0]["frames"][0]["overlays"][0].update(occludes=""),
])
def test_expansion_still_runs_original_facts_and_quality_validators(change):
    compact = pixel_packet.pack(body())
    change(compact)
    with pytest.raises(ValueError):
        pixel_observer.parse_wire(encode(compact), {"g0": 5, "g1": 1})


@pytest.mark.parametrize("raw", [
    '{"schema":"joint_pixel_records_v2","schema":"joint_pixel_records_v2","groups":[]}',
    '{"schema":"joint_pixel_records_v2","groups":[]} {}',
    '{"schema":"joint_pixel_records_v2","groups":[',
])
def test_duplicate_json_keys_or_truncated_wire_cannot_be_repaired(raw):
    with pytest.raises(ValueError):
        pixel_observer.parse_wire(raw, {"g0": 1})


def test_canonical_cache_format_cannot_bypass_wire_version():
    with pytest.raises(ValueError):
        pixel_observer.parse_wire(encode(body()), {"g0": 5, "g1": 1})


@pytest.mark.parametrize("field", ["action", "type_evidence", "overlays"])
def test_packer_does_not_fill_missing_facts_or_quality(field):
    original = body()
    original["groups"][0]["frames"][0].pop(field)
    with pytest.raises(ValueError):
        pixel_packet.pack(original)
