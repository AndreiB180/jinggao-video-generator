import json

import pytest
from fastapi.testclient import TestClient

from jinggao import captions, local_asr, media, paths, planner, store
from jinggao.api import create_app
from test_workflow import audio_bytes


@pytest.fixture
def aligned_case(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    monkeypatch.setattr(local_asr, "transcribe", lambda *_a, **_k: pytest.fail("Approval must not run ASR"))
    store.init()
    project = store.new_project("按语音边界批准", "甲。乙。")
    shots = [planner.new_shot(text) for text in ("甲。", "乙。")]
    for shot in shots:
        shot.update(duration=2, selected_asset="a", approved=False)
    audio = paths.project_dir(project["id"]) / "audio" / "voice.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(audio_bytes(4))
    def save(p):
        p["shots"] = shots
        media.align_narration(p, {"path": "audio/voice.wav", "duration": 4, "text": p["script"]})
    project = store.mutate(project["id"], save)
    *_, binding, cache = captions._source(project)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"method": "local_paraformer", "duration": 4, "coverage": 1,
        "binding": binding, "cues": [{"start": .1, "end": .8, "text": "甲。"},
                                     {"start": 3, "end": 3.8, "text": "乙。"}]}))
    with TestClient(create_app()) as client:
        yield project, client


@pytest.mark.parametrize("bulk", [False, True])
def test_approval_rejects_old_proportional_timeline_without_mutating_project(aligned_case, bulk):
    project, client = aligned_case
    pid, sid = project["id"], project["shots"][0]["id"]
    result = (client.post(f"/api/projects/{pid}/approve-verified") if bulk else
              client.patch(f"/api/projects/{pid}/shots/{sid}", json={"approved": True}))
    assert result.status_code == 400
    assert "真实配音边界" in result.json()["detail"]
    assert store.get_project(pid) == project


def test_approval_uses_current_values_and_preserves_the_last_aligned_revision(aligned_case):
    project, client = aligned_case
    proposal = media.prepare_narration_alignment(project)
    project = store.mutate(project["id"], lambda p: media.apply_narration_alignment(p, proposal))
    assert [s["duration"] for s in project["shots"]] == [3, 1]
    route = f"/api/projects/{project['id']}/shots/{project['shots'][0]['id']}"
    result = client.patch(route, json={"duration": 2, "approved": True})
    assert result.status_code == 400
    assert store.get_project(project["id"]) == project
    result = client.patch(route, json={"approved": True})
    assert result.status_code == 200
    assert result.json()["shots"][0]["approved"]
