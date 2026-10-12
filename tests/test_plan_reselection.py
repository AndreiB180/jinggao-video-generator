"""A framing proposal is not approval; source facts and the full audit survive."""
from copy import deepcopy
import json

from PIL import Image
import pytest

from jinggao import frame_contract, neutral_observer, paths, plan_consistency, plan_reselection, planner, pooled_match, store

pytestmark = pytest.mark.usefixtures("isolated_pixel_quality")


NARRATIONS = ["中间那么长一段路，在手机上往往只剩几行字。今天，就把这几行字重新展开。",
              "把他们全部叫成一个模糊的送货人，会漏掉很多具体劳动。"]
OLD = [{"scene": "手机屏幕", "subject": "手机快递物流信息页面", "action": "手指在屏幕上滑动查看物流进度", "query": "手机 物流 滑动"},
       {"scene": "快递网点", "subject": "快递员与快件处理员", "action": "快递员从处理员手中接过包裹", "query": "快递员 交接"}]
NEW = [{"scene": "手机屏幕", "subject": "手机快递物流信息页面", "action": "手指点击查快递查看物流详情", "query": "手机 查快递"},
       {"scene": "快递分拨中心", "subject": "工作人员", "action": "在传送带旁搬放包裹", "query": "工作人员 搬放包裹"}]
FACTS = [("手持手机显示查快递菜单及物流详情页面", "手指点击查快递", ["查快递", "更多物流信息"]),
         ("工作人员站在快递分拣线传送带旁，手接触包裹", "工作人员双手搬放传送带旁的包裹", ["中新视频"])]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()

    def create(count=2):
        narrations = [NARRATIONS[i % 2] for i in range(count)]
        project = store.new_project("取景重选", "\n".join(narrations), media_policy="video_only")
        shots, assets, rows = [], [], []
        source_start = 0
        for i, narration in enumerate(narrations):
            shot = planner.new_shot(narration, event_scope="background", overlay_label="资料画面")
            shot.update(duration=6.88 if i % 2 == 0 else 5.08, event_persons=[],
                        audio={"path": f"voice-{i}.mp3", "duration": 6.88 if i % 2 == 0 else 5.08},
                        planning_review={"purpose": "illustration", "source_span": {
                            "start": source_start, "end": source_start + len(narration)}},
                        visual_plan={"mode": "single", "primary": deepcopy(OLD[i % 2]), "alternatives": []})
            source_start += len(narration) + 1
            plan_consistency.project_visual_plan(project, shot, required=True)
            asset = {"id": store.uid(), "kind": "video", "title": "已经实际看帧的资料",
                     "path": f"assets/clip-{i}.mp4", "duration": shot["duration"] + .12, "width": 1280,
                     "height": 720, "source_clip_start": 10, "source_url": f"https://example.test/video/{i}"}
            video = paths.file_path(project["id"], asset["path"])
            video.parent.mkdir(exist_ok=True)
            video.write_bytes(b"local fixture; decoder mocked")
            frames = []
            for n in range(5):
                frame = paths.file_path(project["id"], f"review/{i}/{n}.jpg")
                frame.parent.mkdir(exist_ok=True, parents=True)
                Image.new("RGB", (64, 64), (20 + i, n * 40, 40)).save(frame)
                frames.append(frame)
            indices = sorted({int(shot["duration"]*25*f) for f in (.05, .25, .5, .75, .93)})
            preview = {"motion_detected": True, "frames": [{"path": f.relative_to(paths.project_dir(project["id"])).as_posix(),
                        "at": round(10 + n/25, 3), "kind": "video_frame"} for f, n in zip(frames, indices, strict=True)]}
            objects, action, text = FACTS[i % 2]
            observations = [{"frame_index": n, "objects": objects, "action": action, "text": text} for n in range(5)]
            evidence = [{"frame_index": 0, "field": "action", "quote": action}]
            checks = [{"key": k, "required": value, "status": "unknown", "reason": "旧取景未被证明", "evidence": evidence}
                      for k, value in frame_contract.requirements(shot).items()]
            verdict = {"verdict": "uncertain", "scene_alignment": "uncertain", "selection_blocked": True,
                       "needs_source_check": True, "reason": "旧取景未被证明", "identity_evidence": {"reason": "无具名人物"},
                       "observations": {"scene_description": objects, "frame_observations": observations,
                                        "observation_method": "blind_actual_frames_v1"},
                       "identity_audit": {"status": "unknown", "primary_checks": checks}}
            rows.append({"shot": deepcopy(shot), "asset": asset, "candidate": {"id": f"c{i}"}, "preview": preview,
                         "frame_paths": frames, "frame_count": 5, "verification": verdict})
            shots.append(shot)
            assets.append(asset)
        project = store.mutate(project["id"], lambda p: p.update(shots=shots, assets=assets,
                                full_audio={"path": "paid-full-track.mp3", "duration": sum(s["duration"] for s in shots)}))
        windows = {}
        for row in rows:
            plan_reselection.remember(project, windows, row)
        return project, rows, windows
    return create


def response(item, index):
    return {"shot_id": item["shot_id"], "status": "propose", "reason": "调整模型自选动作，保留原稿和实际主体",
            "requirements_preserved": True, "window_id": item["windows"][0]["window_id"],
            "primary": deepcopy(NEW[index % 2]),
            "source_evidence": [{"source": "script", "quote": item["readonly_shot"]["narration"]}],
            "frame_evidence": [{"frame_index": 0, "field": "action", "quote": FACTS[index % 2][1]}]}


def commit(row, pid):
    def apply(project):
        current = next(s for s in project["shots"] if s["id"] == row["shot"]["id"])
        current.update(plan_reselection.commit_fields(project, current, row))
        current.update(selected_asset=row["asset"]["id"], clip_start=row["shot"]["clip_start"],
                       verification=row["verification"], approved=False)
    store.mutate(pid, apply, internal=True)
    return True


@pytest.mark.parametrize("through_match", [False, True])
@pytest.mark.parametrize("reject_new_audit", [False, True])
def test_real_02_10_directions_use_shadow_and_full_original_audit(workspace, monkeypatch, reject_new_audit, through_match):
    project, rows, windows = workspace()
    if through_match:
        from jinggao import config
        config.update({"search_provider": "local"})
        def seed(current):
            for shot, row in zip(current["shots"], rows, strict=True):
                shot.update(verification=row["verification"], match_preview=row["preview"] | {
                    "asset_id": row["asset"]["id"], "clip_start": 0, "duration": shot["duration"],
                    "review_context_hash": plan_reselection.fingerprint(current, shot),
                    "frame_sha256": plan_reselection.frame_hashes(row)})
        project = store.mutate(project["id"], seed)
        monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: (
            [{"label": "本地", "indices": [0, 1], "queries": ["快递"]}], {}))
        monkeypatch.setattr(pooled_match.sources, "local_candidates", lambda *a: [])
        monkeypatch.setattr(pooled_match.library, "remember_window", lambda *a, **kw: None)
    before = deepcopy(project)
    by_id = {r["shot"]["id"]: r for r in rows}
    calls = []
    def sample(pid, shot, asset, check):
        original = by_id[shot["id"]]
        assert shot["clip_start"] == original["shot"]["clip_start"]
        assert shot["duration"] == original["shot"]["duration"]
        return original["frame_paths"], original["preview"]
    def observe(records, check):
        return {r["shot"]["id"]: {"scene_description": by_id[r["shot"]["id"]]["verification"]["observations"]["scene_description"],
                "frames": by_id[r["shot"]["id"]]["verification"]["observations"]["frame_observations"]} for r in records}, 0
    def chat(system, content, **kwargs):
        if system.startswith("定位图中完整的中国大陆11位个人手机号码"):
            calls.append("privacy-scan")
            count = len({entry["text"].split(";")[0] for entry in content if entry["type"] == "text"})
            return {"frames": [{"frame_index": i, "boxes": []} for i in range(count)]}, {}
        if system == plan_reselection.SYSTEM:
            calls.append("propose")
            payload = json.loads(content)
            assert payload["full_script"] == before["script"]
            return {"proposals": [response(item, i) for i, item in enumerate(payload["shots"])]}, {"usage": {"total_tokens": 123}}
        calls.append("full-audit")
        saved = store.get_project(project["id"])
        assert saved["full_audio"] == before["full_audio"]
        assert all(plan_reselection.fingerprint(saved, shot) == plan_reselection.fingerprint(before, old)
                   for shot, old in zip(saved["shots"], before["shots"], strict=True))
        diagnostic = next((paths.DATA / "diagnostics" / project["id"]).glob("plan-reselection-*/batch-*.json"))
        assert json.loads(diagnostic.read_text())["status"] == "proposals_saved"
        answers = []
        for entry in content:
            data = json.loads(entry["text"])
            citation = {"frame_index": 0, "field": "action", "quote": data["frame_observations"][0]["action"]}
            caption_citations = {f"caption_{row['caption_id']}": {
                "frame_index": row["frame_indices"][0], "field": "text", "quote": row["text"]}
                for row in data["context_label_context"]["captions"]}
            status = "conflict" if reject_new_audit else "supported"
            answers.append({"shot_id": data["shot_id"], "status": status, "reason": "完整重新核验",
                "labels": [], "evidence": [citation], "primary_checks": [
                    {"key": k, "status": status, "reason": "实际帧依据", "evidence": [caption_citations.get(k, citation)]}
                    for k in data["selected_visual_requirements"]]})
        return {"reviews": answers}, {}
    monkeypatch.setattr(pooled_match, "sample_frames", sample)
    monkeypatch.setattr(neutral_observer, "observe", observe)
    monkeypatch.setattr(planner, "chat", chat)
    if through_match:
        matched = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None)
        result = matched["plan_reselection"]
        assert matched["downloads"] == 0 and matched["selected"] == (0 if reject_new_audit else 2)
    else:
        result = plan_reselection.run(project, windows, lambda: None, review_batch=pooled_match.review_batch,
                                     commit=lambda row: commit(row, project["id"]))
    after = store.get_project(project["id"])
    expected = ["propose", "full-audit"] + ([] if reject_new_audit else ["privacy-scan"])
    assert calls == expected and result["model_requests"] == len(expected)
    assert result["proposal_usage"] == {"total_tokens": 123}
    assert after["script"] == before["script"] and after["full_audio"] == before["full_audio"]
    assert len(result["applied_shots"]) == (0 if reject_new_audit else 2)
    for old, new in zip(before["shots"], after["shots"], strict=True):
        assert all(new[k] == old[k] for k in ("narration", "duration", "audio", "persons", "event", "overlay_label"))
        if reject_new_audit:
            assert all(new[k] == old[k] for k in plan_reselection.DERIVED_FIELDS)
            assert not new["selected_asset"]
        else:
            assert new["selected_asset"] and new["visual_reselection"]["old_primary"] == old["visual_plan"]["primary"]
            assert new["visual_reselection"]["source_evidence"][0]["quote"] == old["narration"]


@pytest.mark.parametrize("change", [{"persons": [{"name": "张三", "aliases": []}]},
                                   {"event": "2024年决赛"}, {"locked": True}, {"visual_origin": "manual"},
                                   {"subject": "用户另行指定的物件"}])
def test_person_event_locked_and_manually_edited_conditions_never_enter(workspace, change):
    project, rows, _ = workspace(1)
    row = rows[0] | {"shot": rows[0]["shot"] | change}
    windows = {}
    plan_reselection.remember(project, windows, row)
    assert not windows


def test_budget_is_eight_shots_two_windows_and_survives_resume(workspace, monkeypatch):
    project, rows, windows = workspace(10)
    for row in rows:
        for i in range(3):
            extra = deepcopy(row)
            extra["asset"]["id"] += str(i)
            plan_reselection.remember(project, windows, extra)
    assert all(len(options) == 2 for options in windows.values())
    requests = []
    def chat(system, content, **kwargs):
        items = json.loads(content)["shots"]
        requests.append(items)
        return {"proposals": [{"shot_id": item["shot_id"], "status": "keep", "reason": "没有合适完整片段",
                    "requirements_preserved": True, "window_id": None, "primary": None,
                    "source_evidence": [], "frame_evidence": []} for item in items]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    def no_audit(*args):
        pytest.fail("Kept primaries must not trigger paid frame review")
    result = plan_reselection.run(project, windows, lambda: None, review_batch=no_audit, commit=no_audit)
    assert result["attempted_shots"] == 8 and result["model_requests"] == 2
    assert sorted(len(items) for items in requests) == [4, 4]
    again = plan_reselection.run(store.get_project(project["id"]), windows, lambda: None, review_batch=no_audit, commit=no_audit)
    assert again["attempted_shots"] == again["model_requests"] == 0 and len(requests) == 2


@pytest.mark.parametrize("corrupt", ["foreign-window", "invented-source", "narration-patch"])
def test_invalid_proposals_preserve_raw_and_never_reach_audit(workspace, monkeypatch, corrupt):
    project, _, windows = workspace(1)
    def chat(system, content, **kwargs):
        row = response(json.loads(content)["shots"][0], 0)
        if corrupt == "foreign-window":
            row["window_id"] = "another-shot:W1"
        elif corrupt == "invented-source":
            row["source_evidence"][0]["quote"] = "原稿没有要求出现任何人物和动作"
        else:
            row["narration"] = "改掉原稿才能配上视频"
        return {"proposals": [row]}, {}
    monkeypatch.setattr(planner, "chat", chat)
    def no_audit(*args):
        pytest.fail("Invalid proposal must not reach full review")
    result = plan_reselection.run(project, windows, lambda: None, review_batch=no_audit, commit=no_audit)
    assert result["failures"] and result["proposed_shots"] == 0
    assert store.get_project(project["id"])["shots"] == project["shots"]
    record = json.loads(next((paths.DATA / "diagnostics" / project["id"]).glob("plan-reselection-*/batch-*.json")).read_text())
    assert record["response"]["proposals"] and record["decisions"][0]["status"] == "invalid"


def test_existing_preview_requires_current_conditions_and_matching_pixel_cache(workspace):
    project, rows, _ = workspace(1)
    row = rows[0]
    def save(p):
        p["shots"][0].update(verification=row["verification"], match_preview=row["preview"] | {
            "asset_id": row["asset"]["id"], "clip_start": 0, "duration": row["shot"]["duration"]})
    project = store.mutate(project["id"], save)
    assert not plan_reselection.existing_windows(project, lambda: None)
    cache = paths.DATA / "cache" / "neutral-observer" / "old.json"
    cache.parent.mkdir(parents=True)
    observation = row["verification"]["observations"]
    cache.write_text(json.dumps({"frame_sha256": plan_reselection.frame_hashes(row), "observation": {
        "scene_description": observation["scene_description"], "frames": observation["frame_observations"]}}))
    assert len(plan_reselection.existing_windows(project, lambda: None)[row["shot"]["id"]]) == 1
    changed = deepcopy(project)
    changed["shots"][0]["verification"] = None
    assert not plan_reselection.existing_windows(changed, lambda: None)
    changed = deepcopy(project)
    changed["requirements"] = "画面必须展示手指滑动"
    assert not plan_reselection.existing_windows(changed, lambda: None)
    row["frame_paths"][0].write_bytes(b"changed pixels")
    assert not plan_reselection.existing_windows(project, lambda: None)


@pytest.mark.parametrize("change", ["user-condition", "occupied"])
def test_atomic_commit_rejects_stale_conditions_or_occupied_window(workspace, monkeypatch, change):
    project, rows, windows = workspace(2)
    original = deepcopy(project["shots"][0])
    def chat(system, content, **kwargs):
        return {"proposals": [response(json.loads(content)["shots"][0], 0)]}, {}
    def audit(p, drafts, check):
        def update(current):
            if change == "user-condition":
                current["requirements"] = "画面必须保留原来的滑动动作"
            else:
                current["shots"][1].update(selected_asset=rows[0]["asset"]["id"], clip_start=0)
        store.mutate(project["id"], update)
        draft = drafts[0]
        return [draft | {"verification": {"verdict": "match", "scene_alignment": "match",
            "selection_blocked": False, "needs_source_check": False, "identity_audit": {"status": "supported"}}}], [], {"model_requests": 1}
    monkeypatch.setattr(planner, "chat", chat)
    result = plan_reselection.run(project, {rows[0]["shot"]["id"]: windows[rows[0]["shot"]["id"]]},
        lambda: None, review_batch=audit, commit=lambda row: commit(row, project["id"]))
    assert not result["applied_shots"] and result["failures"]
    assert store.get_project(project["id"])["shots"][0] == original
