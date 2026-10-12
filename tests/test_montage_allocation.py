"""Offline allocation/control-flow regressions, never content approval evidence."""
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal

import pytest

from jinggao import costs, library, library_match, media, montage_match as m, paths, planner, store


def clip(name, duration=8, *, start=0, offset=0, url=None):
    return {"id": name, "start": start, "duration": duration, "offset": offset,
            "source": {"source_url": url or f"https://source.test/{name}", "title": "普通资料"}}


def shot(name="s", duration=9.6):
    return {"id": name, "duration": duration, "narration": "实际口播完整一句，不按画面切字幕。", "query": name,
            "persons": [], "selected_asset": None, "clip_start": 0, "approved": False}


def accepted():
    return {"verdict": "match", "scene_alignment": "match", "selection_blocked": False,
            "needs_source_check": False, "reason": "仅测试桩，不是实际内容核验"}


def assert_layout(plan, durations):
    assert len(plan) == len(durations)
    occupied = defaultdict(list)
    for allocation, duration in zip(plan, durations, strict=True):
        c, (key, (start, stop)) = allocation["clip"], allocation["reserved"]
        assert stop-start == pytest.approx(duration)
        assert start*25 == pytest.approx(round(start*25))
        assert stop*25 == pytest.approx(round(stop*25))
        assert Decimal(str(start)) >= Decimal(str(c["start"]))
        assert Decimal(str(stop)) <= Decimal(str(c["start"]))+Decimal(str(c["duration"]))
        assert not any(start < end and stop > begin for begin, end in occupied[key])
        occupied[key].append((start, stop))


@pytest.mark.parametrize("duration", [9.24, 9.6, 9.92, 10.6, 12])
def test_two_eight_second_sources_fit_fixed_three_parts(duration):
    durations = m.piece_lengths(duration)
    used = defaultdict(list)
    plan, status = m._plan([clip("a"), clip("b")], durations, used, set(), lambda: None)
    assert status == "feasible" and len(durations) == 3
    assert_layout(plan, durations)
    assert not used  # Planning is transactional, including recursive backtracking.
    assert plan[0]["reserved"][1][0] == pytest.approx((200-round(durations[0]*25))//2/25)


@pytest.mark.parametrize("duration", [9.24, 9.6, 9.92, 10.6, 12])
def test_single_eight_second_source_is_proven_insufficient(duration):
    plan, status = m._plan([clip("a")], m.piece_lengths(duration), {}, set(), lambda: None)
    assert plan is None and status == "infeasible"


def test_overlapping_clip_ids_and_bilibili_url_aliases_do_not_double_capacity():
    first = "https://www.bilibili.com/video/BV1xx411c7mD/?share_source=public"
    second = "http://bilibili.com/video/BV1xx411c7mD#t=1"
    clips = [clip("original", url=first), clip("alias", url=second)]
    plan, status = m._plan(clips, m.piece_lengths(9.6), {}, set(), lambda: None)
    assert plan is None and status == "infeasible"
    used = {second: [(0, 4)]}
    plan, status = m._plan(clips, [2, 2], used, set(), lambda: None)
    assert status == "feasible"
    assert_layout(plan, [2, 2])
    assert all(r["reserved"][0] == "https://www.bilibili.com/video/BV1xx411c7mD" for r in plan)
    assert all(r["reserved"][1][0] >= 4 for r in plan)
    assert used == {second: [(0, 4)]}


def test_same_source_different_clips_nonoverlap_and_existing_source_parts():
    clips = [clip("a", 8, start=10, url="https://source.test/s"),
             clip("b", 8, start=14, url="https://source.test/s")]
    used = {"https://source.test/s": [(13.001, 16.001)]}
    before = deepcopy(used)
    plan, status = m._plan(clips, [2.8, 2.8], used, set(), lambda: None)
    assert status == "feasible"
    assert_layout(plan, [2.8, 2.8])
    assert all(not (a < 16.04 and b > 13) for _, (a, b) in (r["reserved"] for r in plan))
    assert used == before


@pytest.mark.parametrize("start,duration", [(0.001, 8), (0.039999999999, 8), (0.040000000001, 8),
                                         (100.019999999999, 7.99999999999)])
def test_fractional_boundaries_round_inward(start, duration):
    c = clip("fractional", duration, start=start, offset=0.37)
    plan, status = m._plan([c], [3.96, 3.96], {}, set(), lambda: None)
    assert status == "feasible"
    assert_layout(plan, [3.96, 3.96])
    for allocation in plan:
        source_start = allocation["reserved"][1][0]
        local_start = c["offset"]+source_start-c["start"]
        assert local_start >= c["offset"]
        assert local_start+3.96 <= c["offset"]+c["duration"]+1e-12
        assert local_start+(c["start"]-c["offset"]) == pytest.approx(source_start)
    if start == 0.001:
        plan, status = m._plan([c], [4, 4], {}, set(), lambda: None)
        assert plan is None and status == "infeasible"  # 7.96 usable, not 8.


def test_failed_exact_source_window_cannot_be_reused_through_alias_clip():
    a, b = clip("original", 4, url="https://source.test/s#one"), clip("alias", 4, url="https://source.test/s#two")
    failed = {("https://source.test/s", 0, 100)}
    plan, status = m._plan([a, b], [4], {}, failed, lambda: None)
    assert plan is None and status == "infeasible"


def test_search_limit_is_distinct_and_check_is_cancellable(monkeypatch):
    monkeypatch.setattr(m, "_SEARCH_LIMIT", 1)
    used = {"https://source.test/occupied": [(0, 1)]}
    before = deepcopy(used)
    plan, status = m._plan([clip("a"), clip("b")], [3.2]*3, used, set(), lambda: None)
    assert plan is None and status == "search_limit"
    assert used == before
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise media.Cancelled("offline cancellation")
    with pytest.raises(media.Cancelled):
        m._plan([clip("a"), clip("b")], [3.2]*3, used, set(), cancel)
    assert used == before and calls == 3


class Run:
    def __init__(self, tmp_path, monkeypatch, shots, options, *, ranking=None, materializing=None,
                 judging=None, combining=None, check=None, existing=None):
        monkeypatch.setattr(paths, "DATA", tmp_path)
        store.init()
        self.project = store.new_project("纯离线蒙太奇分配回归")
        original = {"shots": deepcopy(shots), "full_audio": {"path": "keep-original.wav", "duration": 264},
                    "caption_cues": [{"start": 0, "end": 8, "text": "整句字幕不随内部画面切分"}]}
        if existing:
            original["assets"] = existing["assets"]
            original["shots"].extend(existing["shots"])
        self.project = store.mutate(self.project["id"], lambda p: p.update(original))
        self.shots, self.options = shots, options
        self.rank_calls, self.materializations, self.piece_reviews, self.whole_reviews = [], [], [], []
        self.assembled, self.used_refs, self.used_at_materialize = [], [], []
        original_plan = m._plan
        def planning(clips, durations, used, failed, check, **kwargs):
            self.used_refs.append(used)
            return original_plan(clips, durations, used, failed, check, **kwargs)
        monkeypatch.setattr(m, "_plan", planning)
        monkeypatch.setattr(library, "search", lambda query, **kw: deepcopy(options[query]))
        monkeypatch.setattr(library_match, "quick_candidates", lambda p, s, o, stats: (s, o))
        def rank(shots, candidates, check, **kwargs):
            self.rank_calls.append([s["id"] for s in shots])
            if ranking:
                ranking(shots, candidates)
            return True
        monkeypatch.setattr(library_match, "rank_windows", rank)
        def materialize(pid, c, check):
            check()
            self.materializations.append(c["id"])
            self.used_at_materialize.append(deepcopy(dict(self.used_refs[-1])))
            if materializing:
                materializing(c, self)
            return {"id": "asset-"+c["id"], "kind": "video", "width": 1280, "height": 720,
                    "duration": c["offset"]+c["duration"], "source_url": c["source"]["source_url"],
                    "source_clip_start": c["start"]-c["offset"]}
        monkeypatch.setattr(library, "materialize", materialize)
        def combine(p, s, rows, check):
            check()
            self.assembled.append((s, rows))
            if combining:
                combining(s, rows)
            assert len(rows) == len(m.piece_lengths(s["duration"]))
            assert sum(r["shot"]["duration"] for r in rows) == pytest.approx(s["duration"])
            assert all(r["verification"] == accepted() for r in rows)
            asset = {"id": "assembled-"+s["id"], "kind": "video", "source_parts": [
                {"source_url": r["reserved"][0], "source_at": r["reserved"][1][0],
                 "duration": r["shot"]["duration"]} for r in rows]}
            store.mutate(p["id"], lambda current: current["assets"].append(asset), internal=True)
            return asset
        monkeypatch.setattr(m, "assemble", combine)
        def review(p, rows, check):
            result, errors = [], []
            for row in rows:
                whole = bool(row["asset"].get("source_parts"))
                (self.whole_reviews if whole else self.piece_reviews).append(row)
                verdict = judging(row, whole, self) if judging else accepted()
                if isinstance(verdict, Exception):
                    raise verdict
                if isinstance(verdict, str):
                    errors.append((row, verdict))
                else:
                    result.append(row | {"verification": verdict, "preview": {"frames": []}})
            return result, errors, {"model_requests": 1}
        self.review = review
        self.check = check or (lambda: None)

    def run(self):
        return m.match(self.project, self.shots, self.check, lambda *args: None, review_batch=self.review, workers=1)

    def assert_unchanged_audio(self):
        current = store.get_project(self.project["id"])
        assert current["full_audio"] == self.project["full_audio"]
        assert current["caption_cues"] == self.project["caption_cues"]
        assert [s["narration"] for s in current["shots"]] == [s["narration"] for s in self.project["shots"]]
        assert all(not s.get("approved") for s in current["shots"])

    def assert_clean(self, expected=None):
        if self.used_refs:
            assert {k: v for k, v in self.used_refs[-1].items() if v} == (expected or {})


@pytest.mark.parametrize("duration", [9.24, 9.6, 9.92, 10.6, 12])
def test_match_impossible_preflight_spends_no_rank_or_review(tmp_path, monkeypatch, duration):
    run = Run(tmp_path, monkeypatch, [shot(duration=duration)], {"s": [clip("a")]})
    passed, stats = run.run()
    assert not passed and not run.rank_calls and not run.materializations and not run.piece_reviews
    assert stats["model_requests"] == 0
    assert stats["allocation_diagnostics"][0]["stage"] == "before_rank"
    assert stats["allocation_diagnostics"][0]["status"] == "infeasible"
    run.assert_unchanged_audio()


def test_match_search_limit_is_diagnostic_not_capacity_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "_SEARCH_LIMIT", 1)
    run = Run(tmp_path, monkeypatch, [shot()], {"s": [clip("a"), clip("b")]})
    passed, stats = run.run()
    assert not passed and not run.rank_calls and stats["model_requests"] == 0
    assert stats["allocation_diagnostics"][0]["status"] == "search_limit"


def test_rank_removed_candidates_are_not_restored(tmp_path, monkeypatch):
    def rank(shots, options):
        options["s"] = options["s"][:1]
    run = Run(tmp_path, monkeypatch, [shot()], {"s": [clip("a"), clip("b")]}, ranking=rank)
    passed, stats = run.run()
    assert not passed and len(run.rank_calls) == 1
    assert not run.materializations and not run.piece_reviews
    assert stats["allocation_diagnostics"][0]["stage"] == "allocation"
    run.assert_clean()


@pytest.mark.parametrize("duration", [9.24, 9.6, 9.92, 10.6, 12])
def test_complete_three_piece_and_whole_reviews_still_required(tmp_path, monkeypatch, duration):
    run = Run(tmp_path, monkeypatch, [shot(duration=duration)], {"s": [clip("a"), clip("b")]})
    passed, stats = run.run()
    assert passed == {"s"} and stats["selected"] == 1
    assert len(run.piece_reviews) == 3 and len(run.whole_reviews) == 1
    run.assert_unchanged_audio()
    run.assert_clean()


def test_all_shots_own_full_plans_before_first_materialize(tmp_path, monkeypatch):
    run = Run(tmp_path, monkeypatch, [shot("one", 8), shot("two", 8)],
              {sid: [clip("a"), clip("b")] for sid in ("one", "two")})
    passed, _ = run.run()
    assert passed == {"one", "two"}
    assert sum(map(len, run.used_at_materialize[0].values())) == 4
    assert sum(b-a for spans in run.used_at_materialize[0].values() for a, b in spans) == 16
    first_wave = run.piece_reviews[:2]
    assert first_wave[0]["reserved"][0] != first_wave[1]["reserved"][0]
    run.assert_clean()


@pytest.mark.parametrize("failure", ["download", "review_error", "review_reject"])
def test_failed_owner_releases_unused_tail_to_waiting_shot(tmp_path, monkeypatch, failure):
    same = "https://source.test/shared"
    options = {"one": [clip("long", 8, url=same)], "two": [clip("tail", 4, start=4, url=same)]}
    def materialize(c, run):
        if c["id"] == "long" and failure == "download":
            raise OSError("offline download failed")
    def judge(row, whole, run):
        if row["shot"]["id"] == "one" and not whole:
            if failure == "review_error":
                return "offline malformed review"
            if failure == "review_reject":
                return accepted() | {"verdict": "mismatch", "selection_blocked": True}
        return accepted()
    run = Run(tmp_path, monkeypatch, [shot("one", 8), shot("two", 4)], options,
              materializing=materialize, judging=judge)
    passed, _ = run.run()
    assert passed == {"two"}
    assert len([r for r in run.piece_reviews if r["shot"]["id"] == "two"]) == 2
    run.assert_clean()


def test_passed_prefix_not_paid_reviewed_again_after_second_piece_failure(tmp_path, monkeypatch):
    failed = []
    def judge(row, whole, run):
        if not whole and len(run.piece_reviews) == 2:
            failed.append(m._window_id(row["reserved"]))
            return accepted() | {"selection_blocked": True}
        return accepted()
    clips = [clip("a"), clip("b"), clip("b-alias", url="https://source.test/b#alias")]
    run = Run(tmp_path, monkeypatch, [shot()], {"s": clips}, judging=judge)
    passed, _ = run.run()
    assert passed == {"s"} and len(run.piece_reviews) == 4
    identities = [m._window_id(r["reserved"]) for r in run.piece_reviews]
    assert len(identities) == len(set(identities))
    assert identities.count(failed[0]) == 1
    assert len(run.whole_reviews) == 1
    run.assert_clean()


def test_five_round_cap_and_no_repeat_failed_window(tmp_path, monkeypatch):
    run = Run(tmp_path, monkeypatch, [shot()], {"s": [clip("a"), clip("b")]},
              judging=lambda *args: accepted() | {"verdict": "unknown", "selection_blocked": True})
    passed, _ = run.run()
    assert not passed and len(run.piece_reviews) == 5 and not run.whole_reviews
    identities = [m._window_id(r["reserved"]) for r in run.piece_reviews]
    assert len(identities) == len(set(identities))
    run.assert_clean()


@pytest.mark.parametrize("exception", [media.Cancelled, costs.BudgetExceeded, planner.ModelUnavailable])
@pytest.mark.parametrize("phase", ["materialize", "piece_review", "whole_review"])
def test_fatal_errors_propagate_and_owned_reservations_release(tmp_path, monkeypatch, exception, phase):
    fatal = exception("offline fatal")
    def materialize(c, run):
        if phase == "materialize":
            raise fatal
    def judge(row, whole, run):
        if (phase == "whole_review" and whole) or (phase == "piece_review" and not whole):
            raise fatal
        return accepted()
    run = Run(tmp_path, monkeypatch, [shot()], {"s": [clip("a"), clip("b")]},
              materializing=materialize, judging=judge)
    with pytest.raises(exception) as caught:
        run.run()
    assert caught.value is fatal
    run.assert_clean()
    assert store.get_project(run.project["id"])["shots"][0]["selected_asset"] is None
    run.assert_unchanged_audio()


def test_cancel_after_whole_review_before_commit(tmp_path, monkeypatch):
    cancel = False
    def judge(row, whole, run):
        nonlocal cancel
        if whole:
            cancel = True
        return accepted()
    def check():
        if cancel:
            raise media.Cancelled("cancel after response")
    run = Run(tmp_path, monkeypatch, [shot()], {"s": [clip("a"), clip("b")]}, judging=judge, check=check)
    with pytest.raises(media.Cancelled):
        run.run()
    assert len(run.whole_reviews) == 1
    assert store.get_project(run.project["id"])["shots"][0]["selected_asset"] is None
    run.assert_clean()


@pytest.mark.parametrize("failure", ["assemble", "whole_reject", "whole_error"])
def test_whole_failure_never_selects_and_does_not_restart_waiting_shot(tmp_path, monkeypatch, failure):
    def combine(s, rows):
        if failure == "assemble":
            raise ValueError("offline assembly failed")
    def judge(row, whole, run):
        if whole and failure == "whole_reject":
            return accepted() | {"selection_blocked": True}
        if whole and failure == "whole_error":
            return "offline final format failed"
        return accepted()
    run = Run(tmp_path, monkeypatch, [shot("one", 8), shot("two", 8)],
              {sid: [clip("shared")] for sid in ("one", "two")}, combining=combine, judging=judge)
    passed, _ = run.run()
    assert not passed
    assert all(r["shot"]["id"] == "one" for r in run.piece_reviews)
    assert all(s["selected_asset"] is None for s in store.get_project(run.project["id"])["shots"])
    run.assert_clean()


def test_existing_occupancy_survives_cleanup(tmp_path, monkeypatch):
    url = "https://source.test/shared"
    existing = {"assets": [{"id": "selected", "kind": "video", "source_parts": [
        {"asset_id": "old", "source_url": url, "source_at": 8, "duration": 4}]}],
        "shots": [shot("existing", 4) | {"selected_asset": "selected", "locked": True}]}
    run = Run(tmp_path, monkeypatch, [shot("s", 8)], {"s": [clip("new", 12, url=url)]}, existing=existing)
    passed, _ = run.run()
    assert passed == {"s"}
    assert all(r["reserved"][1][1] <= 8 for r in run.piece_reviews)
    run.assert_clean({url: [(8, 12.0)]})


def test_shared_source_prefers_boundaries_but_exclusive_source_keeps_center():
    a, b = clip("a"), clip("b")
    shots = [shot("one"), shot("two")]
    options = {"one": [a, clip("alias", url="https://source.test/a#public"), b], "two": [a]}
    shared = m._shared_sources(shots, options)
    assert shared == {"https://source.test/a"}
    assert m._shared_sources(shots[:1], options) == set()
    plan, status = m._plan(options["one"], [3.2]*3, {}, set(), lambda: None, compact_sources=shared)
    assert status == "feasible"
    assert [r["reserved"] for r in plan] == [("https://source.test/a", (0.0, 3.2)),
                                            ("https://source.test/a", (3.2, 6.4)),
                                            ("https://source.test/b", (2.4, 5.6))]


def test_sharing_is_recomputed_after_rank_removes_other_shot_options(tmp_path, monkeypatch):
    def rank(shots, options):
        options["two"] = []
    run = Run(tmp_path, monkeypatch, [shot("one"), shot("two", 8)],
              {"one": [clip("a"), clip("b")], "two": [clip("a")]}, ranking=rank)
    passed, _ = run.run()
    assert passed == {"one"}
    assert run.piece_reviews[0]["reserved"] == ("https://source.test/a", (2.4, 5.6))
    run.assert_clean()


def test_search_limit_polls_database_style_check_at_bounded_intervals():
    calls = 0
    def check():
        nonlocal calls
        calls += 1
    failed = {("https://source.test/a", at, at+82) for at in range(219)}
    plan, status = m._plan([clip("a", 12)], m.piece_lengths(9.92), {}, failed, check)
    assert plan is None and status == "search_limit"
    assert calls == 1+1+m._SEARCH_LIMIT//32+1  # entry, clip, polling, exit
