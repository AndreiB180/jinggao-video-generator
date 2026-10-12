from copy import deepcopy
import hashlib

import pytest

from jinggao import editorial_contract, library, library_match, paths


@pytest.fixture
def retained(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    project = {"id": "a" * 32}
    video = paths.file_path(project["id"], "asset.mp4")
    video.write_bytes(b"the exact indexed media bytes")
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    clip = {"id": "clip", "blob_id": digest, "start": 10, "offset": 0, "duration": 8,
            "description": "工作人员在仓库分拣包裹。", "visible_text": "",
            "source": {"source_url": "https://example.org/clip"}}
    shot = {"id": "shot", "duration": 6, "clip_start": 1, "approved": False}
    asset = {"library_clip": "clip", "library_blob": digest, "path": "asset.mp4",
             "source_url": clip["source"]["source_url"], "source_clip_start": 10}
    monkeypatch.setattr(editorial_contract, "retrieval_fields", lambda _: {"query_hints": ["民航飞机起飞"]})
    monkeypatch.setattr(library, "get", lambda _: clip)
    monkeypatch.setattr(library_match, "_blind_clip_ids", lambda _: {"clip"})
    return project, shot, asset, clip


def test_retained_miss_is_only_a_retrieval_deferral_and_does_not_mutate(retained):
    project, shot, asset, _ = retained
    before = deepcopy(retained)
    assert library_match.retained_retrieval_miss(project, shot, asset)
    assert retained == before


@pytest.mark.parametrize("change", ["file", "window", "source", "blob", "unobserved", "locked", "approved", "manual", "stale_base", "missing_offset"])
def test_ambiguous_bindings_and_confirmed_choices_still_get_regular_handling(retained, monkeypatch, change):
    project, shot, asset, clip = retained
    if change == "file":
        paths.file_path(project["id"], asset["path"]).write_bytes(b"changed media")
    elif change == "window":
        shot["clip_start"] = 4
    elif change == "source":
        asset["source_url"] = "https://example.org/different"
    elif change == "blob":
        asset["library_blob"] = "0" * 64
    elif change == "unobserved":
        monkeypatch.setattr(library_match, "_blind_clip_ids", lambda _: set())
    elif change == "manual":
        shot.update(source_review_url="https://example.org/checked",
                    source_review_note="已经核对实际画面和来源上下文")
    elif change == "stale_base":
        asset["source_clip_start"] = 0
        shot["clip_start"] = 11
    elif change == "missing_offset":
        clip.pop("offset")
    else:
        shot[change] = True
    assert not library_match.retained_retrieval_miss(project, shot, asset)


def test_retained_specific_match_is_not_a_retrieval_miss(retained):
    project, shot, asset, clip = retained
    clip["description"] = "民航飞机从机场跑道起飞。"
    assert not library_match.retained_retrieval_miss(project, shot, asset)
