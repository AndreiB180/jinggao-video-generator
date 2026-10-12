from copy import deepcopy
import json

import pytest

from jinggao import library_match, planner, rank_packet


def inputs():
    scene = "工人在传送带旁分拣纸箱，墙上贴有安全操作告示。" * 20
    shots = [{"id": f"shot-{i}", "narration": f"独立原文与要求{i}"} for i in range(4)]
    options = {s["id"]: [{"id": f"clip-{i}", "source": {"title": f"Source {i}"},
                         "description": scene, "visible_text": "安全操作"} for i in range(8)] for s in shots}
    return shots, options


def test_real_rank_entry_shares_text_and_preserves_each_candidate_scope(monkeypatch, tmp_path):
    shots, options = inputs()
    original = deepcopy((shots, options))
    requests = []
    def chat(system, content, **kwargs):
        data = json.loads(content)
        requests.append(data)
        assert rank_packet.PROTOCOL in system
        raw = rank_packet.unpack(data)
        assert len(rank_packet.encoded(data)) < len(rank_packet.encoded(raw)) / 2
        assert [s["narration"] for s in raw["shots"]] == [s["narration"] for s in shots]
        for i, shot in enumerate(raw["shots"]):
            assert shot["shot_id"] == shots[i]["id"]
            for j, c in enumerate(shot["candidates"]):
                assert c["candidate_ref"] == f"S{i+1:02d}-C{j+1:02d}"
                assert c["observed_scene"] == original[1][shots[i]["id"]][j]["description"][:700]
        return {"rankings": [{"shot_id": s["shot_id"], "candidate_refs": [s["candidates"][0]["candidate_ref"]]}
                             for s in data["shots"]]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    library_match.rank_windows(shots, options, lambda: None, summary_dir=tmp_path)
    assert len(requests) == 1
    assert shots == original[0]
    assert options == {sid: rows[:1] for sid, rows in original[1].items()}
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert rank_packet.PROTOCOL in saved["system"]
    assert saved["ranking_only"] is True


def test_shared_text_id_cannot_be_selected_in_place_of_a_candidate(monkeypatch, tmp_path):
    shots, options = inputs()
    def chat(_system, content, **kwargs):
        data = json.loads(content)
        return {"rankings": [{"shot_id": s["shot_id"], "candidate_refs": ["T0"]} for s in data["shots"]]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    library_match.rank_windows(shots, options, lambda: None, summary_dir=tmp_path)
    assert all(not rows for rows in options.values())


def test_small_unique_payload_retains_exact_original_input():
    data = {"protocol": "library_shot_candidates_v2", "shots": [{"shot_id": "original-id", "candidates": [
        {"candidate_ref": "S01-C01", "source_title": "独立标题", "observed_scene": "一只猫。", "visible_text": ""}]}]}
    shared, protocol = rank_packet.pack(data)
    assert shared is data and not protocol


def test_prepopulated_or_nontext_evidence_is_not_reinterpreted():
    with pytest.raises(ValueError, match="already supplied"):
        rank_packet.pack({"candidate_texts": {}})
    with pytest.raises(ValueError, match="original text"):
        rank_packet.pack({"shots": [{"candidates": [{"source_title": {"$candidate_text": "T0"}}]}]})
