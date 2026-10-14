from copy import deepcopy

from jinggao import config, library_match, paths, planner, pooled_match, store, video_sources


def shot(sid, query, duration=8, **fields):
    return {"id": sid, "query": query, "duration": duration, "search_queries": [query],
            "persons": [], "event": "", "event_scope": "background", **fields}


def video(cid, title, provider="bilibili", **fields):
    return {"id": cid, "title": title, "kind": "video", "provider": provider,
            "width": 1280, "height": 720, "status": "unverified", **fields}


def test_round5_box_and_truck_primaries_get_rescue_before_shared_sorting_words():
    shots = [shot("first", "分拣线 包裹 自动流动", 3),
             shot("05", "旧的宽泛主题", 11.16, visual_plan={"mode": "single", "primary": {
                 "query": "纸箱 打开 快递", "search_queries": ["快递 开箱"]}, "alternatives": [
                     {"query": "纸箱 书籍 快递"}]}),
             shot("32", "快递货车 行驶", 15), shot("done", "已通过场景", 100)]
    groups = [{"indices": [0, 1, 2, 3], "queries": ["快递分拨传送带", "自动化分拣设备"]}]
    found = {"快递分拨传送带": [], "分拣线 包裹 自动流动": []}
    before = deepcopy(shots)
    tasks = pooled_match.online_rescue_queries(groups, shots, shots[:3], found)
    assert list(tasks)[:3] == ["快递货车 行驶", "纸箱 打开 快递", "快递 开箱"]
    assert "自动化分拣设备" == list(tasks)[-1]
    assert not {"旧的宽泛主题", "纸箱 书籍 快递", "已通过场景"} & tasks.keys()
    audit = pooled_match.query_selection("快递货车 行驶", tasks["快递货车 行驶"], shots, shots[:3])
    assert audit == {"reason": "primary_gap", "shot_indices": [2], "gap_seconds": 15}
    assert shots == before


def test_round5_opening_title_survives_the_24_to_12_metadata_cut():
    shots = [shot("05", "纸箱 打开 快递", 11.16), shot("storage", "快递纸箱 存放", 8)]
    group = {"indices": [0, 1], "queries": ["快递纸箱拆箱", "收件人拆快递"]}
    rows = [video(str(i), "快递纸箱长期堆放的危害", "iqiyi") for i in range(28)]
    rows += [video("opening", "六月拆快递", "iqiyi"), video("dog", "警犬查获违禁品", "chinanews"),
             video("rice", "炒饭正确打开方式", "sohu")]
    before = deepcopy(rows)
    pool, audit = pooled_match.primary_candidate_pool({"min_video_height": 720}, group, shots, rows, 24)
    assert len(pool) == 24 and "opening" in {c["id"] for c in pool[:12]}
    assert not {"dog", "rice"} & {c["id"] for c in pool[:12]}
    opening = next(row for row in audit[:12] if row["candidate_id"] == "opening")
    assert opening["shot_indices"] == [0] and opening["reason"] == "primary_query_terms"
    assert rows == before and all(c["status"] == "unverified" for c in pool)


def test_unrelated_provider_cannot_displace_other_agricultural_actions():
    shots = [shot("plough", "拖拉机 耕地", 12), shot("test", "粮食 水分 检测", 9),
             shot("water", "农田 灌溉", 7), shot("done", "农民 收割", 60)]
    group = {"indices": [0, 1, 2, 3], "queries": ["农业机械", "农业工作"]}
    rows = [video(str(i), "拖拉机耕地作业", "cctv") for i in range(30)]
    rows += [video("moisture", "粮食水分检测操作", "cctv"), video("irrigation", "农田灌溉现场", "cctv"),
             video("concert", "歌手演唱会", "bilibili"), video("harvest", "农民收割粮食", "iqiyi"),
             video("low", "农田灌溉", "chinanews", width=480, height=270)]
    pool, audit = pooled_match.primary_candidate_pool({}, group, shots, rows, 12, shots[:3])
    assert {c["id"] for c in pool[:3]} == {"0", "moisture", "irrigation"}
    assert "low" not in {c["id"] for c in pool}
    assert [row["shot_indices"] for row in audit[:3]] == [[0], [1], [2]]
    assert all(3 not in row["shot_indices"] for row in audit)
    assert not any(c.get("approved") or c.get("verification") for c in pool)


def test_sampling_keeps_distinct_pending_primaries_instead_of_provider_heads():
    shots = [shot("phone", "旧主题", visual_plan={"primary": {
                 "query": "快递 手机物流页面", "search_queries": []}}),
             shot("call", "快递员 打电话"), shot("door", "快递员 楼道敲门"),
             shot("done", "快递员 分拣包裹", 60)]
    rows = [video(str(i), "快递员 快递 分拣包裹", provider, download_status="available")
            for i, provider in enumerate(["cctv", "iqiyi", "chinanews"] * 10)]
    rows += [video(cid, title, download_status="available") for cid, title in [
        ("phone", "手机物流页面 查询"), ("call", "快递员打电话联系收件人"), ("door", "快递员在楼道敲门")]]
    group = {"indices": list(range(4)), "queries": ["快递派送", "快递分拨"]}
    pending = {"group": group, "shots": shots[:3], "choices": {c["id"]: c for c in rows}}
    before = deepcopy((shots, rows))
    chosen, budgeted = pooled_match.allocate_sampling_candidates({}, {"g": pending}, shots)
    assert [c["id"] for c in chosen["g"]] == ["phone", "call", "door"]
    assert budgeted == {"phone", "call", "door"} and (shots, rows) == before
    assert all(c["status"] == "unverified" and not c.get("verification") for c in chosen["g"])


def test_large_shared_pool_gets_cohorts_without_larger_requests_or_more_text_batches():
    queries = ['谷物 清洗', '稻谷 脱壳', '粮食 筛分', '大米 包装']
    shots = [shot(str(i), queries[i // 8]) for i in range(30)]
    group = {'indices': list(range(30)), 'queries': ['粮食 加工', '食品 工厂']}
    rows = [video(str(i), query, download_status='available') for i, query in enumerate(queries)]
    pending = {'all': {'group': group, 'shots': shots, 'choices': {c['id']: c for c in rows}}}
    before = deepcopy(pending)
    old, old_sources = pooled_match.allocate_sampling_candidates({}, pending, shots)
    cohorts = pooled_match.sampling_cohorts(pending)
    chosen, new_sources = pooled_match.allocate_sampling_candidates({}, cohorts, shots)
    assert len(old_sources) == 3 and len(new_sources) == 4
    assert len(cohorts) == 4  # Same ceil(30/8) text localization batches.
    assert all(len(row['shots']) <= 8 and len(chosen[key]) <= 3 for key, row in cohorts.items())
    assert [chosen[key][0]['id'] for key in cohorts] == ['0', '1', '2', '3']
    assert len(new_sources) * 4 == 16 > len(old_sources) * 4 == 12
    assert pending == before and all(row['parent_key'] == 'all' for row in cohorts.values())


def test_cohorts_keep_global_source_budget_and_every_pending_shot():
    shots = [shot(str(i), f'具体工序{i}', duration=i+1) for i in range(80)]
    group = {'indices': list(range(80)), 'queries': ['生产 工厂']}
    rows = [video(str(i), f'具体工序{i}', download_status='available') for i in range(80)]
    pending = {'all': {'group': group, 'shots': shots, 'choices': {c['id']: c for c in rows}}}
    cohorts = pooled_match.sampling_cohorts(pending)
    chosen, budgeted = pooled_match.allocate_sampling_candidates({}, cohorts, shots)
    assert len(budgeted) <= 24 and all(len(v) <= 3 for v in chosen.values())
    assert [s['id'] for row in cohorts.values() for s in row['shots']] == [s['id'] for s in shots]
    assert all(chosen[key] for key in cohorts)


def test_sampling_preserves_one_neighbour_slot_named_event_and_hd_gates():
    shots = [shot("named", "梁靖崑 比赛回合", persons=[{"name": "梁靖崑", "aliases": []}],
                  event="2026 WTT中国大满贯 R32", event_scope="exact")]
    title = "2026 WTT中国大满贯 梁靖崑 MS R32"
    rows = [video("wrong-round", "2026 WTT中国大满贯 梁靖崑 MS R16", download_status="available"),
            video("low", title, height=480, download_status="available"),
            video("blocked", title, download_status="blocked"),
            video("neighbour", title, sample_points=[12, 24], download_status="available"),
            video("extra-neighbour", title, sample_points=[32], download_status="available"),
            video("fresh1", title, download_status="available"),
            video("fresh2", title, download_status="available")]
    group = {"indices": [0], "queries": ["梁靖崑 比赛回合"]}
    pending = {"group": group, "shots": shots, "choices": {c["id"]: c for c in rows}}
    chosen, budgeted = pooled_match.allocate_sampling_candidates({}, {"g": pending}, shots)
    assert [c["id"] for c in chosen["g"]] == ["neighbour", "fresh1", "fresh2"]
    assert len(budgeted) == 3 and sum(bool(c.get("sample_points")) for c in chosen["g"]) == 1
    assert shots[0]["persons"][0]["name"] == "梁靖崑" and shots[0]["event_scope"] == "exact"


def test_sampling_round_robin_still_caps_the_whole_pass_at_24_sources():
    shots = [shot(str(i), f"农田第{i}区 灌溉") for i in range(16)]
    pending = {}
    for i, s in enumerate(shots):
        rows = [video(f"{i}-{j}", s["query"], download_status="available") for j in range(3)]
        pending[i] = {"group": {"indices": [i], "queries": [s["query"]]},
                      "shots": [s], "choices": {c["id"]: c for c in rows}}
    chosen, budgeted = pooled_match.allocate_sampling_candidates({}, pending, shots)
    assert len(budgeted) == 24 and len(chosen) == 16
    assert all(1 <= len(rows) <= 3 for rows in chosen.values())
    assert {c["id"] for rows in chosen.values() for c in rows} == budgeted


def test_diagnostics_use_project_indices_after_library_removes_earlier_shot(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    store.init()
    config.update({"search_provider": "video"})
    project = store.new_project("检索索引", "三个镜头。", media_policy="video_only")
    shots = [planner.new_shot("同主题不同动作。", query=query)
             for query in ["完成场景", "拖拉机 耕地", "农田 灌溉"]]
    store.mutate(project["id"], lambda p: p.update(shots=shots))
    project = store.get_project(project["id"])
    monkeypatch.setattr(library_match, "match", lambda *a: ({shots[0]["id"]}, {
        "library_model_requests": 0, "library_failures": []}))
    group = {"label": "农业", "indices": [0, 1], "queries": ["农业", "备用"]}
    monkeypatch.setattr(pooled_match, "retrieval_groups", lambda *a: ([group], {}))
    rows = [video("plough", "拖拉机耕地"), video("water", "农田灌溉")]
    monkeypatch.setattr(video_sources, "search_videos", lambda *a, **kw: (rows, []))
    monkeypatch.setattr(planner, "rank_candidates", lambda *a: [])
    monkeypatch.setattr(video_sources, "inspect_video_candidate", lambda c, *a: c | {"download_status": "blocked"})
    monkeypatch.setattr(pooled_match, "discover_windows", lambda *a, **kw: ({}, {}, {
        "model_requests": 0, "sampled_bytes": 0, "failures": []}))
    monkeypatch.setattr(planner, "chat", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("unexpected model call")))
    stats = pooled_match.match(project, project["shots"], lambda: None, lambda *a: None)
    initial = next(row for row in stats["search_tasks"] if row["phase"] == "initial")
    assert initial["shot_indices"] == [1, 2]
    selections = stats["candidate_screening"][0]["selections"]
    assert {tuple(row["shot_indices"]) for row in selections} == {(1,), (2,)}
    assert all(row["cover"] and row["reason"] == "primary_query_terms" for row in selections)
    assert stats["retrieval_shot_index_base"] == 0 and stats["downloads"] == 0
