import threading
from concurrent.futures import ThreadPoolExecutor

from jinggao import library, paths, platform_videos, video_sources as video


def setup_search(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(library, "remember_sources", lambda _: None)
    monkeypatch.setattr(video, "_recent_unavailable", lambda rows, _: (rows, {}))


def test_empty_long_query_replaces_later_tencent_call_with_explicit_hint_only(tmp_path, monkeypatch):
    setup_search(tmp_path, monkeypatch)
    tencent, cctv = [], []
    monkeypatch.setattr(platform_videos, "tencent_search", lambda q: tencent.append(q) or [])
    monkeypatch.setattr(video, "cctv_search", lambda q: cctv.append(q) or [])
    state = video.TencentQueryFallback()
    first, second = {}, {}
    video.search_videos("水龙头维修 拆卸阀芯", use_index=False, channel_names={"腾讯视频", "央视"},
                        tencent_short_query="水龙头维修", tencent_query_state=state, diagnostics=first)
    rows, errors = video.search_videos("王楚钦 澳门世界杯 反手接发球", use_index=False,
                        channel_names={"腾讯视频", "央视"}, tencent_short_query="王楚钦 澳门世界杯",
                        tencent_query_state=state, diagnostics=second)
    assert tencent == ["水龙头维修 拆卸阀芯", "王楚钦 澳门世界杯"]
    assert cctv == ["水龙头维修 拆卸阀芯", "王楚钦 澳门世界杯 反手接发球"]
    assert first["providers"][0]["empty_query_trigger"] is True
    assert second["providers"][0]["query_reason"] == "explicit_short_after_empty"
    assert second["providers"][0]["effective_query"] == "王楚钦 澳门世界杯"
    assert not rows and not errors  # An empty search never becomes approval.


def test_failures_and_missing_explicit_hints_do_not_invent_or_activate_short_query(tmp_path, monkeypatch):
    setup_search(tmp_path, monkeypatch)
    state = video.TencentQueryFallback()
    calls = []

    def fetch(query):
        calls.append(query)
        if len(calls) == 1:
            raise ValueError("访问受限")
        return []

    monkeypatch.setattr(platform_videos, "tencent_search", fetch)
    options = dict(use_index=False, channel_names={"腾讯视频"}, tencent_query_state=state)
    video.search_videos("ThinkPad T14 更换电池", tencent_short_query="ThinkPad T14", **options)
    video.search_videos("ThinkPad T14 拆卸后盖", **options)
    video.search_videos("ThinkPad T14 检查接口", tencent_short_query="ThinkPad T14", **options)
    video.search_videos("ThinkPad T14 整机安装", **options)
    assert calls == ["ThinkPad T14 更换电池", "ThinkPad T14 拆卸后盖",
                     "ThinkPad T14 检查接口", "ThinkPad T14 整机安装"]
    assert video.TencentQueryFallback().select("维修 开盖", "维修") == "维修 开盖"


def test_queued_provider_rechecks_feedback_after_slot_without_extra_call(tmp_path, monkeypatch):
    setup_search(tmp_path, monkeypatch)
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(video, "SEARCH_PROVIDER_SLOTS", slots)
    state, entered, calls = video.TencentQueryFallback(), threading.Event(), []
    monkeypatch.setattr(platform_videos, "tencent_search", lambda q: calls.append(q) or [])
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(video.search_videos, "校园图书馆 学生借书", use_index=False,
            channel_names={"腾讯视频"}, tencent_short_query="校园图书馆", tencent_query_state=state,
            check=entered.set)
        assert entered.wait(2)
        state.observe_empty("校园图书馆 办理借阅", "校园图书馆 办理借阅", "校园图书馆")
        slots.release()
        assert future.result(timeout=2) == ([], [])
    assert calls == ["校园图书馆"]
    assert slots.acquire(blocking=False)
    slots.release()
