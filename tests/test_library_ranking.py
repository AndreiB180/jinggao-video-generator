"""Library ranking is a shortlist, never an identity or content approval."""
from copy import deepcopy
import json
from pathlib import Path
from threading import Barrier, Lock

import pytest

from jinggao import editorial_contract, library, library_match, media, paths, planner, pooled_match, store


def clip(clip_id):
    return {"id": clip_id, "source": {"title": "比赛资料"},
            "description": "室内球场两位运动员连续移动击球", "visible_text": "决赛"}


def input_rows(count=2, candidates=2):
    shots = [{"id": f"shot-{i}", "narration": f"镜头{i}原文", "visual": "来回击球",
              "persons": ["指定运动员"], "event_scope": "exact", "event": "指定决赛",
              "evidence_mode": "person", "location": "指定球馆", "era": "指定赛季",
              "exclusions": "不能用采访代替比赛", "duration": 6} for i in range(count)]
    return shots, {s["id"]: [clip(f"{s['id']}-clip-{j}") for j in range(candidates)] for s in shots}


def ranked(content, *, selected=6):
    data = json.loads(content)
    return {"rankings": [{"shot_id": row["shot_id"], "candidate_refs": [
        c["candidate_ref"] for c in row["candidates"][:selected]]} for row in data["shots"]]}, {}


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")


def test_shot_scoped_candidates_preserve_constraints_and_never_approve(monkeypatch):
    shots, options = input_rows()
    # Identical bytes / clip IDs offered to two shots still have different refs.
    options["shot-1"][0] = options["shot-0"][0]
    original, requests = deepcopy((shots, options)), []

    def chat(system, content, **kwargs):
        data = json.loads(content)
        requests.append(data)
        assert data["protocol"] == "library_shot_candidates_v2"
        assert "clips" not in data and "没有全局数字索引" in system
        assert kwargs == {"max_tokens": 730, "timeout": 25, "task": "library_ranking"}
        for row, shot in zip(data["shots"], shots, strict=True):
            assert all(row[key] == value for key, value in shot.items() if key != "id")
        return ranked(content, selected=1)

    monkeypatch.setattr(planner, "chat", chat)
    assert library_match.rank_windows(shots, options, lambda: None)
    assert requests[0]["shots"][0]["candidates"][0]["candidate_ref"] == "S01-C01"
    assert requests[0]["shots"][1]["candidates"][0]["candidate_ref"] == "S02-C01"
    assert shots == original[0]
    assert options == {sid: rows[:1] for sid, rows in original[1].items()}
    assert all("approved" not in c and "verification" not in c for rows in options.values() for c in rows)


@pytest.mark.parametrize("refs", [
    ["S02-C01"], ["S01-C01", "S02-C01"], ["C01"], ["S1-C1"], ["S01-C03"],
    [0], ["0"], [True], [None], [["S01-C01"]], [{"candidate_ref": "S01-C01"}],
    ["S01-C01", "S01-C01"], "S01-C01", None,
])
def test_cross_shot_aliases_and_malformed_refs_reject_entire_row(monkeypatch, refs):
    shots, options = input_rows()
    original, errors = deepcopy(options), []

    def chat(_system, content, **_kwargs):
        response, info = ranked(content)
        response["rankings"][0]["candidate_refs"] = refs
        return response, info

    monkeypatch.setattr(planner, "chat", chat)
    assert library_match.rank_windows(shots, options, lambda: None, diagnostics=errors)
    assert options == {"shot-0": [], "shot-1": original["shot-1"]}
    assert len(errors) == 1 and "shot-0" in errors[0]


def test_more_than_six_refs_is_rejected_without_truncation(monkeypatch):
    shots, options = input_rows(1, 12)
    monkeypatch.setattr(planner, "chat", lambda _s, content, **_kw: ranked(content, selected=7))
    library_match.rank_windows(shots, options, lambda: None)
    assert options == {"shot-0": []}


@pytest.mark.parametrize("bad_row", ["missing", "duplicate", "foreign", "wrong-type"])
def test_bad_shot_rows_do_not_discard_independent_valid_shots(monkeypatch, bad_row):
    shots, options = input_rows()
    original, errors = deepcopy(options), []

    def chat(_system, content, **_kwargs):
        response, info = ranked(content)
        if bad_row == "missing":
            response["rankings"].pop(0)
        elif bad_row == "duplicate":
            response["rankings"].append(deepcopy(response["rankings"][0]))
        elif bad_row == "foreign":
            response["rankings"][0]["shot_id"] = "unknown-shot"
        else:
            response["rankings"][0] = ["shot-0", "S01-C01"]
        return response, info

    monkeypatch.setattr(planner, "chat", chat)
    library_match.rank_windows(shots, options, lambda: None, diagnostics=errors)
    assert options == {"shot-0": [], "shot-1": original["shot-1"]}
    assert any("shot-0" in e for e in errors)


def test_33_shots_use_bounded_batches_and_unique_refs_with_exact_request_count(monkeypatch):
    shots, options = input_rows(33, 12)
    original, requests, started = deepcopy(options), [], []

    def chat(_system, content, **kwargs):
        data = json.loads(content)
        requests.append(data)
        assert len(data["shots"]) <= 4
        assert sum(len(r["candidates"]) for r in data["shots"]) <= 48
        assert kwargs["max_tokens"] == 250 + 240 * len(data["shots"])
        return ranked(content)

    monkeypatch.setattr(planner, "chat", chat)
    assert library_match.rank_windows(shots, options, lambda: None, workers=1,
                                      request_started=lambda: started.append(True))
    assert len(requests) == len(started) == 9
    refs = [c["candidate_ref"] for d in requests for r in d["shots"] for c in r["candidates"]]
    assert len(refs) == len(set(refs)) == 396
    assert refs[0] == "S01-C01" and refs[-1] == "S33-C12"
    assert options == {sid: rows[:6] for sid, rows in original.items()}


def test_prior_batch_ref_cannot_be_rebound_in_later_batch(monkeypatch):
    shots, options = input_rows(5, 1)
    original, errors = deepcopy(options), []

    def chat(_system, content, **_kwargs):
        response, info = ranked(content)
        if response["rankings"][0]["shot_id"] == "shot-4":
            response["rankings"][0]["candidate_refs"] = ["S01-C01"]
        return response, info

    monkeypatch.setattr(planner, "chat", chat)
    library_match.rank_windows(shots, options, lambda: None, diagnostics=errors)
    assert options == original | {"shot-4": []}
    assert len(errors) == 1 and "shot-4" in errors[0]


def test_parallel_rank_calls_never_exceed_two(monkeypatch):
    shots, options = input_rows(8, 1)
    barrier, lock, concurrent = Barrier(2, timeout=5), Lock(), []
    active = 0

    def chat(_system, content, **_kwargs):
        nonlocal active
        with lock:
            active += 1
            concurrent.append(active)
        barrier.wait()
        with lock:
            active -= 1
        return ranked(content)

    monkeypatch.setattr(planner, "chat", chat)
    library_match.rank_windows(shots, options, lambda: None, workers=8)
    assert max(concurrent) == 2 and active == 0


@pytest.mark.parametrize("bad_response", [None, [], {"rankings": {}}, {"wrong": []}])
def test_malformed_batch_does_not_discard_other_batches(monkeypatch, bad_response):
    shots, options = input_rows(5, 1)
    original = deepcopy(options)

    def chat(_system, content, **_kwargs):
        data = json.loads(content)
        return (bad_response, {}) if len(data["shots"]) == 4 else ranked(content)

    monkeypatch.setattr(planner, "chat", chat)
    library_match.rank_windows(shots, options, lambda: None)
    assert options == {sid: [] for sid in original} | {"shot-4": original["shot-4"]}


def test_failed_chat_is_counted_once_without_retry_or_reusing_candidates(monkeypatch):
    shots, options = input_rows(5, 1)
    original, attempted, errors = deepcopy(options), [], []

    def chat(_system, content, **_kwargs):
        if len(json.loads(content)["shots"]) == 4:
            raise ValueError("模型响应不是有效JSON")
        return ranked(content)

    monkeypatch.setattr(planner, "chat", chat)
    library_match.rank_windows(shots, options, lambda: None, workers=1, diagnostics=errors,
                               request_started=lambda: attempted.append(True))
    assert len(attempted) == 2 and len(errors) == 1
    assert options == {sid: [] for sid in original} | {"shot-4": original["shot-4"]}


def test_account_failure_stops_remaining_batches_without_changing_options(monkeypatch):
    shots, options = input_rows(5, 1)
    original, attempted = deepcopy(options), []

    def unavailable(*_args, **_kwargs):
        raise planner.ModelUnavailable("HTTP 402")

    monkeypatch.setattr(planner, "chat", unavailable)
    with pytest.raises(planner.ModelUnavailable, match="402"):
        library_match.rank_windows(shots, options, lambda: None, workers=1,
                                   request_started=lambda: attempted.append(True))
    assert len(attempted) == 1 and options == original


def test_cancellation_after_response_never_publishes_partial_options(monkeypatch):
    shots, options = input_rows(5, 1)
    original, cancelled = deepcopy(options), False

    def chat(_system, content, **_kwargs):
        nonlocal cancelled
        cancelled = True
        return ranked(content)

    def check():
        if cancelled:
            raise media.Cancelled()

    monkeypatch.setattr(planner, "chat", chat)
    with pytest.raises(media.Cancelled):
        library_match.rank_windows(shots, options, check, workers=1)
    assert options == original


def test_legacy_numeric_response_is_saved_raw_and_replayed_without_guessing(monkeypatch):
    shots, options = input_rows()
    original, summaries = deepcopy(options), []
    # This is a constructed old-protocol response, not the unavailable round-5
    # raw response. A global ordinal must never become a shot-local selection.
    bad = {"rankings": [{"shot_id": s["id"], "indices": [2]} for s in shots]}
    monkeypatch.setattr(planner, "chat", lambda *_a, **_kw: (bad, {
        "model": "test", "usage": {"total_tokens": 8}, "api_key": "must-not-be-saved"}))
    library_match.rank_windows(shots, options, lambda: None, summary_paths=summaries)
    assert options == {s["id"]: [] for s in shots} and len(summaries) == 1
    saved_text = Path(summaries[0]).read_text()
    saved = json.loads(saved_text)
    assert "must-not-be-saved" not in saved_text and "api_key" not in saved_text
    assert saved["response"] == bad and saved["ranking_only"] is True
    assert saved["model_info"] == {"model": "test", "usage": {"total_tokens": 8}}
    by_clip = {c["id"]: c for rows in original.values() for c in rows}
    allowed = {sid: {ref: by_clip[cid] for ref, cid in refs.items()}
               for sid, refs in saved["candidate_clip_ids"].items()}
    replayed, errors = library_match._rank_response(saved["request"], allowed, saved["response"])
    assert replayed == options and errors == saved["errors"]


@pytest.mark.parametrize("problem", ["duplicate-shot", "duplicate-clip", "too-many-clips"])
def test_invalid_input_fails_before_any_call(monkeypatch, problem):
    shots, options = input_rows(2, 2)
    if problem == "duplicate-shot":
        shots[1]["id"] = shots[0]["id"]
    elif problem == "duplicate-clip":
        options["shot-1"][1] = options["shot-1"][0]
    else:
        options["shot-1"] = [clip(f"clip-{i}") for i in range(13)]
    original = deepcopy(options)
    monkeypatch.setattr(planner, "chat", lambda *_a, **_kw: pytest.fail("invalid input reached model"))
    with pytest.raises(ValueError):
        library_match.rank_windows(shots, options, lambda: None)
    assert options == original


def test_no_observed_descriptions_skips_model_without_counting_a_request(monkeypatch):
    shots, options = input_rows(1, 1)
    options["shot-0"][0]["description"] = ""
    original = deepcopy(options)
    monkeypatch.setattr(planner, "chat", lambda *_a, **_kw: pytest.fail("no observations"))
    assert not library_match.rank_windows(shots, options, lambda: None,
                                          request_started=lambda: pytest.fail("counted an unused request"))
    assert options == original


@pytest.mark.parametrize("subject,action,description,reason", [
    ("快件处理员", "在传送带旁分拣包裹", "空荡荡的仓库内有自动化分拣设备、轨道和货架，全程没有人物出现。",
     "subject_not_observed"),
    ("快递员", "拨打电话联系收件人", "快递员在站点整理包裹，背喷雾器对货车消杀，将纸箱搬到传送带上。",
     "action_not_observed"),
    ("快递员", "在楼道里敲门送件", "交警指挥交通，行人撑伞行走。没有出现快递员整理包裹、按门铃或等待收件人的动作。",
     "no_observed_terms"),
])
def test_round9_known_different_actions_do_not_enter_library_fast_review(subject, action, description, reason):
    shot = {"id": "shot", "duration": 6, "query": subject + " " + action, "subject": subject,
            "visual_plan": {"mode": "single", "primary": {"subject": subject, "action": action}}}
    candidate = clip("observed") | {"description": description,
        "source": {"title": subject + action}, "observation_method": "blind_actual_frames_v1"}
    with library.connection() as con:
        con.execute("INSERT INTO clip_observation_methods VALUES (?,?)", ("observed", "blind_actual_frames_v1"))
    stats = {}
    chosen, options = library_match.quick_candidates({"shots": [shot]}, [shot], {"shot": [candidate]}, stats)
    assert chosen == [] and options == {"shot": []}
    assert stats["library_quick_skips"][0]["reasons"] == {reason: 1}
    assert "approved" not in candidate and "verification" not in candidate


def test_every_shot_gets_a_bounded_shortlist_with_identity_priority_and_blind_marker():
    shots = [{"id": f"shot-{i}", "duration": i+1, "query": "球场 移动 击球"} for i in range(10)]
    shots[0]["persons"] = [{"name": "指定运动员", "aliases": []}]
    options = {s["id"]: [clip(f"{s['id']}-{j}") for j in range(4)] for s in shots}
    forged = clip("unmarked") | {"description": "球场移动击球", "observation_method": "blind_actual_frames_v1"}
    options[shots[0]["id"]].insert(0, forged)
    with library.connection() as con:
        con.executemany("INSERT INTO clip_observation_methods VALUES (?,?)", [
            (c["id"], "blind_actual_frames_v1") for rows in options.values() for c in rows if c is not forged])
    stats = {}
    chosen, filtered = library_match.quick_candidates({"shots": shots}, shots, options, stats)
    assert len(chosen) == len(shots) and chosen[0]["id"] == "shot-0"
    assert sum(map(len, filtered.values())) == 40
    assert all(len(rows) <= 8 and forged not in rows for rows in filtered.values())
    reasons = [r["reasons"] for r in stats["library_quick_skips"]]
    assert not any(r.get("shot_budget", 0) for r in reasons)
    assert sum(r.get("no_blind_observation", 0) for r in reasons) == 1
    assert all(not c.get("approved") and not c.get("verification") for rows in filtered.values() for c in rows)


def test_quick_shortlist_does_not_spend_three_slots_on_the_same_source_window():
    shot = {"id": "shot", "duration": 6, "query": "球场 移动 击球"}
    rows = [clip(name) | {"start": start, "duration": duration,
            "source": {"title": "比赛资料", "source_url": source}}
            for name, start, duration, source in [
                ("best", 12.04, 12.16, "https://v.qq.com/x/page/one.html"),
                ("same-start", 12.04, 11.16, "https://v.qq.com/x/page/one.html"),
                ("near-start", 13.12, 10, "https://v.qq.com/x/page/one.html"),
                ("later-scene", 40, 10, "https://v.qq.com/x/page/one.html"),
                ("other-source", 12.04, 12.16, "https://v.qq.com/x/page/two.html"),
            ]]
    with library.connection() as con:
        con.executemany("INSERT INTO clip_observation_methods VALUES (?,?)",
                        [(c["id"], "blind_actual_frames_v1") for c in rows])
    original = deepcopy(rows)
    stats = {}
    chosen, filtered = library_match.quick_candidates({"shots": [shot]}, [shot], {"shot": rows}, stats)
    assert chosen == [shot]
    assert [c["id"] for c in filtered["shot"]] == ["best", "later-scene", "other-source"]
    assert stats["library_quick_skips"][0]["reasons"]["overlapping_window"] == 2
    assert rows == original


def test_cache_defers_after_one_failed_window_to_preserve_fresh_source_budget(monkeypatch):
    store.init()
    project = store.new_project("快查预算")
    shot = {"id": store.uid(), "query": "球场 移动 击球", "duration": 6, "persons": [],
            "selected_asset": None, "approved": False}
    project = store.mutate(project["id"], lambda p: p["shots"].append(shot))
    candidates = [clip(f"clip-{i}") | {"start": 0, "offset": 0, "duration": 9, "score": 20,
        "source": {"title": "比赛资料", "source_url": f"https://v.qq.com/x/page/{i}.html"}} for i in range(4)]
    with library.connection() as con:
        con.executemany("INSERT INTO clip_observation_methods VALUES (?,?)", [
            (c["id"], "blind_actual_frames_v1") for c in candidates])
    monkeypatch.setattr(library, "search", lambda *a, **kw: candidates)
    calls, reviewed = [], []
    def rank(system, content, **kwargs):
        calls.append(json.loads(content))
        assert "主体、动作及场景的排序依据只能来自observed_scene和visible_text" in system
        assert "普通说明性内容允许同题材、同类主体" in system
        assert "具名人物仍须严格核对" in system
        assert len(calls[-1]["shots"][0]["candidates"]) == 4
        return ranked(content)
    monkeypatch.setattr(planner, "chat", rank)
    monkeypatch.setattr(library, "materialize", lambda *a: {"id": "asset", "kind": "video"})
    def review(project, rows, check):
        reviewed.extend(rows)
        return [r | {"verification": {"verdict": "mismatch", "scene_alignment": "mismatch",
            "selection_blocked": True, "needs_source_check": True}} for r in rows], [], {"model_requests": 2}
    monkeypatch.setattr(pooled_match, "review_batch", review)
    passed, stats = library_match.match(project, [shot], lambda: None, lambda *a: None)
    assert not passed and len(calls) == 1 and len(reviewed) == 1
    assert reviewed[0]["library_clip"]["id"] == candidates[0]["id"]
    assert stats["library_quick_budget"] == {"shots": 1, "windows_per_shot": 8, "review_rounds": 1}
    assert stats["library_quick_reviewed"] == stats["library_quick_deferred"] == 1
    assert stats["library_model_requests"] == 3
    assert any(r["reasons"].get("actual_review_not_matched") for r in stats["library_quick_skips"])
    assert not store.get_project(project["id"])["shots"][0]["selected_asset"]


def source_scenario(script, queries, *, requirements="", **fields):
    project = {"script": script, "requirements": requirements, "assets": []}
    shot = planner.new_shot(script, duration=6, visual_origin="generated", **fields)
    response = {"contracts": [{"ref": "s1", "use": "illustration", "refs": [],
        "facts": [], "must": [], "queries": queries}]}
    bundle = editorial_contract.compile_responses(project, [shot], [response])
    project["shots"] = editorial_contract.attach(project, [shot], bundle)
    return project, project["shots"][0]


def test_source_shortlist_spends_on_bound_scene_clues_not_generic_hints_or_titles(monkeypatch):
    project, shot = source_scenario("茶叶在机器中揉捻。", ["机械揉捻机内部", "食品加工动作"])
    descriptions = [
        ("weak", "设备在工作，人员处理包裹。", ""),
        ("invented", "食品在输送线上移动。", ""),
        ("caption", "人员打开纸箱，字幕介绍茶叶揉捻。", "茶叶揉捻"),
        ("absent", "没有看到揉捻，纸箱在输送线上移动。", ""),
        ("actual", "手掌推动茶叶进行揉捻。", ""),
    ]
    rows = [clip(cid) | {"description": description, "visible_text": text,
                        "source": {"title": "制茶揉捻过程"}} for cid, description, text in descriptions]
    original, stats = deepcopy((project, rows)), {}
    monkeypatch.setattr(library_match, "_blind_clip_ids", lambda ids: ids)
    chosen, options = library_match.quick_candidates(project, [shot], {shot["id"]: rows}, stats)
    assert chosen == [shot] and [r["id"] for r in options[shot["id"]]] == ["actual"]
    assert stats["library_quick_skips"][0]["reasons"] == {"no_source_observed_clues": 4}
    assert stats["library_source_clues"][0]["source_terms"] == ["揉捻"]
    assert options[shot["id"]][0]["retrieval_support"]["observed_terms"] == ["揉捻"]
    assert not options[shot["id"]][0].get("approved") and (project, rows) == original


@pytest.mark.parametrize("script,queries,description,text,fields", [
    ("全国快递业务量达到一千亿件。", ["快递分拣", "快递运输"],
     "快递员整理包裹。", "今年的快递服务", {}),
    ("制度让民航发展更稳健。", ["民航飞机起飞"], "民航飞机停在机场。", "", {}),
    ("国产C919客机已经交付。", ["C919客机"], "一架客机停在机坪。", "机身C919", {}),
    ("刘翔的训练经历。", ["田径训练现场"], "运动员在场内移动。", "刘翔",
     {"persons": [{"name": "刘翔", "aliases": []}], "evidence_mode": "person"}),
])
def test_source_shortlist_keeps_industry_and_identity_inspection_without_approval(
        monkeypatch, script, queries, description, text, fields):
    project, shot = source_scenario(script, queries, **fields)
    row = clip("actual") | {"description": description, "visible_text": text}
    before = deepcopy((project, row))
    monkeypatch.setattr(library_match, "_blind_clip_ids", lambda ids: ids)
    chosen, options = library_match.quick_candidates(project, [shot], {shot["id"]: [row]}, {})
    assert chosen == [shot] and [r["id"] for r in options[shot["id"]]] == ["actual"]
    assert not options[shot["id"]][0].get("approved") and not options[shot["id"]][0].get("verification")
    assert (project, row) == before


def test_deferred_source_candidates_make_zero_requests_and_preserve_existing_review(monkeypatch):
    project, shot = source_scenario("茶叶经过机械揉捻。", ["机械揉捻机内部"])
    # A spending hint cannot erase a previous valid choice, even if this caller
    # includes it in targets. Normal matching callers already omit approved shots.
    shot.update(selected_asset="previous-asset", approved=True, verification={"verdict": "match"})
    row = clip("unrelated") | {"score": 10, "description": "机械设备正在分拣快递包裹。"}
    before = deepcopy(project)
    monkeypatch.setattr(library_match, "_blind_clip_ids", lambda ids: ids)
    monkeypatch.setattr(library, "search", lambda *a, **kw: [row])
    def unexpected(*_args, **_kwargs):
        pytest.fail("Deferred local candidates must not call a model, materialize or mutate a project")
    monkeypatch.setattr(planner, "chat", unexpected)
    monkeypatch.setattr(library, "materialize", unexpected)
    monkeypatch.setattr(pooled_match, "review_batch", unexpected)
    monkeypatch.setattr(store, "mutate", unexpected)
    passed, stats = library_match.match(project, [shot], lambda: None, lambda *a: None)
    assert passed == set() and stats["library_quick_deferred"] == 1
    assert stats["library_model_requests"] == stats["library_quick_reviewed"] == 0
    assert project == before and shot["approved"]


def test_source_candidate_still_requires_a_stored_blind_observation(monkeypatch):
    project, shot = source_scenario("茶叶经过揉捻。", ["揉捻茶叶"])
    row = clip("unmarked") | {"description": "手掌在揉捻茶叶。", "observation_method": "blind_actual_frames_v1"}
    monkeypatch.setattr(library_match, "_blind_clip_ids", lambda ids: set())
    stats = {}
    chosen, options = library_match.quick_candidates(project, [shot], {shot["id"]: [row]}, stats)
    assert not chosen and not options[shot["id"]]
    assert stats["library_quick_skips"][0]["reasons"] == {"no_blind_observation": 1}
