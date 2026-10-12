"""Retained library spending must use the same source clues as new choices."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from jinggao import editorial_contract, library, library_match, paths, planner, retrieval_terms


@pytest.fixture
def bound_retained(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)

    def unexpected(*_args, **_kwargs):
        pytest.fail("A local retrieval clue check must not request a model")

    monkeypatch.setattr(planner, "chat", unexpected)

    def make(script, queries, *, narration=None, requirements="", description="", visible_text="", **fields):
        narration = narration or script
        before, after = script.split(narration, 1)
        parts = [part for part in (before, narration, after) if part]
        project = {"id": "a" * 32, "script": script, "requirements": requirements, "assets": []}
        shots = [planner.new_shot(part, duration=6, clip_start=1,
                                  visual_origin="generated", **fields) for part in parts]
        response = {"contracts": [{"ref": f"s{i}", "use": "illustration", "refs": [],
            "facts": [], "must": [], "queries": queries} for i in range(1, len(shots) + 1)]}
        bundle = editorial_contract.compile_responses(project, shots, [response])
        project["shots"] = editorial_contract.attach(project, shots, bundle)
        shot = project["shots"][parts.index(narration)]
        video = paths.file_path(project["id"], "asset.mp4")
        video.write_bytes(b"exact indexed media for a source-bound retrieval check")
        digest = hashlib.sha256(video.read_bytes()).hexdigest()
        clip = {"id": "clip", "blob_id": digest, "start": 10, "offset": 0, "duration": 8,
                "description": description, "visible_text": visible_text,
                "source": {"source_url": "https://example.org/clip", "title": "资料来源"}}
        asset = {"id": "asset", "library_clip": clip["id"], "library_blob": digest,
                 "path": "asset.mp4", "source_url": clip["source"]["source_url"], "source_clip_start": 10}
        shot["selected_asset"] = asset["id"]
        project["assets"] = [asset]
        monkeypatch.setattr(library, "get", lambda _: clip)
        monkeypatch.setattr(library_match, "_blind_clip_ids", lambda _: {clip["id"]})
        return project, shot, asset, clip

    return make


def test_historic_packaging_candidate_is_deferred_for_both_new_and_retained_choice(bound_retained):
    fixture = json.loads((Path(__file__).parent / "fixtures/library_tea_rank_inputs.json").read_text())
    row = next(row for row in fixture["rows"] if row["shot_index"] == 30)
    candidate = next(c for c in row["candidates"]
                     if row["candidate_clip_ids"][c["candidate_ref"]] in row["selected_clip_ids"])
    project, shot, asset, clip = bound_retained(
        fixture["script"], row["queries"], narration=row["narration"], requirements=fixture["requirements"],
        description=candidate["observed_scene"], visible_text=candidate["visible_text"])
    original = deepcopy((project, shot, asset, clip))
    # The old gate accepted an inferred food/packaging hint and did not require
    # a source-bound object clue in the actual scene. OCR alone is not that clue.
    assert retrieval_terms.observed_match(row["queries"], clip["description"], clip["visible_text"])
    stats = {}
    chosen, options = library_match.quick_candidates(project, [shot], {shot["id"]: [clip]}, stats)
    assert not chosen and not options[shot["id"]]
    assert stats["library_quick_skips"][0]["reasons"] == {"no_source_observed_clues": 1}
    assert library_match.retained_retrieval_miss(project, shot, asset)
    assert (project, shot, asset, clip) == original


@pytest.mark.parametrize("script,queries,description,text,fields", [
    ("全国快递业务量增长。", ["快递分拣"], "快递员整理包裹。", "", {}),
    ("茶叶经过揉捻。", ["揉捻茶叶"], "手掌推动茶叶进行揉捻。", "", {}),
    ("国产C919客机已经交付。", ["C919客机"], "一架客机停在机坪。", "型号C919客机", {}),
    ("刘翔的训练经历。", ["田径训练现场"], "运动员在场地内移动。", "刘翔",
     {"persons": [{"name": "刘翔", "aliases": []}], "evidence_mode": "person"}),
])
def test_source_clues_keep_new_and_retained_inspection_without_approving(
        bound_retained, script, queries, description, text, fields):
    project, shot, asset, clip = bound_retained(script, queries, description=description, visible_text=text, **fields)
    original = deepcopy((project, shot, asset, clip))
    chosen, options = library_match.quick_candidates(project, [shot], {shot["id"]: [clip]}, {})
    assert chosen == [shot] and [c["id"] for c in options[shot["id"]]] == [clip["id"]]
    assert not library_match.retained_retrieval_miss(project, shot, asset)
    assert not shot.get("approved") and not shot.get("verification")
    assert (project, shot, asset, clip) == original


@pytest.mark.parametrize("protection", ["locked", "approved", "manual"])
def test_source_clue_miss_preserves_protected_choices(bound_retained, protection):
    project, shot, asset, clip = bound_retained("茶叶经过揉捻。", ["揉捻茶叶"], description="快递员打包纸箱。")
    if protection == "manual":
        shot.update(source_review_url="https://example.org/verified", source_review_note="已人工核对来源与实际画面")
    else:
        shot[protection] = True
    original = deepcopy((project, shot, asset, clip))
    assert not library_match.retained_retrieval_miss(project, shot, asset)
    assert (project, shot, asset, clip) == original


@pytest.mark.parametrize("description,miss", [("民航飞机从机场跑道起飞。", False), ("快递员打包纸箱。", True)])
def test_missing_source_profile_retains_legacy_fallback(bound_retained, monkeypatch, description, miss):
    project, shot, asset, _ = bound_retained("民航飞机起飞。", ["民航飞机起飞"], description=description)
    monkeypatch.setattr(editorial_contract, "context", lambda _: {})
    assert library_match.retained_retrieval_miss(project, shot, asset) is miss


def test_retained_source_clues_run_only_after_exact_asset_binding(bound_retained, monkeypatch):
    project, shot, asset, _ = bound_retained("茶叶经过揉捻。", ["揉捻茶叶"], description="快递员打包纸箱。")
    paths.file_path(project["id"], asset["path"]).write_bytes(b"different media")
    monkeypatch.setattr(library_match, "_source_profile", lambda *_: pytest.fail("Unbound pixels must retain normal handling"))
    assert not library_match.retained_retrieval_miss(project, shot, asset)
