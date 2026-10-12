"""Preserve stock actions that occur late in tag lists within the input bound."""
from copy import deepcopy
import json

import pytest

from jinggao import paths, planner, source_metadata


@pytest.mark.parametrize("provider", ["pixabay", "pexels"])
def test_late_stock_tags_reach_screen_without_altering_source_or_approving(tmp_path, monkeypatch, provider):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    title = "bike, bicycle, girl, children, female, cycle, outdoor, park, activity, sport, ride, bicyclist, lifestyle, kid, outside, city, people, fun, cycling"
    assert title.index("cycling") > 110
    source = {"id": "stock", "provider": provider, "title": title, "description": "", "duration": 14,
              "source_url": "https://example.com/stock", "download_status": "unchecked"}
    project = {"script": "骑车经过城市街道，再停车休息。", "requirements": ""}
    shots = [{"id": "s1", "narration": "骑车经过城市街道。"}, {"id": "s2", "narration": "停车休息。"}]
    group = {"indices": [0, 1], "queries": ["城市骑行"]}
    original = deepcopy((project, shots, source))
    calls = []
    def chat(system, user, **options):
        request = json.loads(user)
        row = request["groups"][0]
        assert row["candidates"][0]["title"] == title
        assert row["narrations"] == [shot["narration"] for shot in shots]
        assert len(user.encode()) <= source_metadata.MAX_INPUT_BYTES
        calls.append(options)
        return {"groups": [{"group_ref": row["group_ref"], "candidate_refs": [row["candidates"][0]["candidate_ref"]]}]}, {"request_count": 1}
    monkeypatch.setattr(planner, "chat", chat)
    ranked, info = source_metadata.rank(project, shots, [(group, [source])], lambda: None)
    assert ranked == {0: [source]} and info["metadata_only"] and len(calls) == 1
    assert (project, shots, source) == original and "approved" not in ranked[0][0]


def test_oversized_inputs_are_still_rejected_before_spending(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("Oversized input must be rejected before a paid request")
    monkeypatch.setattr(planner, "chat", forbidden)
    with pytest.raises(ValueError, match="24KB"):
        source_metadata.rank({"script": "文案", "requirements": "要求" * 13000},
                             [{"narration": "文案"}], [({"indices": [0], "queries": ["查询"]}, [])], lambda: None)
