"""Ordinary anonymous page metadata, isolated browser lifetime and bounded access."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import subprocess

import pytest

from jinggao import bilibili_web_search as web
from jinggao import paths, platform_videos as platform, video_sources as video


@pytest.fixture(autouse=True)
def reset_transient_backoff(monkeypatch):
    monkeypatch.setattr(web, "unavailable_until", 0)


def snapshot():
    return {"url": web.PAGE + "?keyword=城市交通&page=1", "title": "城市交通-哔哩哔哩_bilibili",
        "page": 1, "ready": "complete", "text": "登录后可以同步历史记录", "cards": [{
            "title": "<em>城市</em>交通资料", "url": "https://www.bilibili.com/video/BV1pTap6UEai/?from=search",
            "video_links": ["https://www.bilibili.com/video/BV1pTap6UEai/"],
            "thumbnail": "//i0.hdslb.com/bfs/archive/example.jpg", "publisher": "发布者",
            "publisher_url": "https://space.bilibili.com/123", "duration": "01:23", "date": " · 2025年1月2日",
            "excluded": False, "badges": ""}]}


def test_public_card_metadata_is_bound_and_does_not_claim_decoded_quality():
    result = web.parse_snapshot(snapshot(), "城市交通")
    row = result[0]
    assert row["source_url"] == "https://www.bilibili.com/video/BV1pTap6UEai"
    assert row["title"] == "城市交通资料" and row["duration"] == 83
    assert row["publisher"] == "发布者" and row["publisher_url"] == "https://space.bilibili.com/123"
    assert row["thumbnail_url"] == "https://i0.hdslb.com/bfs/archive/example.jpg"
    assert row["height"] == row["width"] == 0 and not row["metadata_checked"]
    assert row["search_method"] == "anonymous_browser_dom" and not row["publisher_verified"]
    assert row["source_evidence"]["evidence_origin"] == "publisher_search"
    assert row["source_evidence"]["url"] == row["source_url"]


@pytest.mark.parametrize("change", [
    {"url": "https://www.bilibili.com.evil.test/video/BV1pTap6UEai"},
    {"url": "https://www.bilibili.com/bangumi/play/ep123"},
    {"url": "https://www.bilibili.com/cheese/play/ep123"},
    {"url": "https://user:pass@www.bilibili.com/video/BV1pTap6UEai"},
    {"video_links": ["https://www.bilibili.com/video/BV1xR4y147L1/"]},
    {"title": ""}, {"excluded": True}, {"badges": "充电专属"}, {"badges": "付费"}, {"badges": "大会员"},
])
def test_non_video_paid_ads_and_mixed_identity_cards_are_excluded(change):
    data = snapshot()
    data["cards"][0].update(change)
    assert web.parse_snapshot(data, "城市交通") == []


def test_duplicate_cards_and_untrusted_thumbnail_or_author_urls():
    data = snapshot()
    data["cards"][0].update(thumbnail="https://foreign.example/a.jpg", publisher_url="https://evil.test/123")
    data["cards"].append(copy.deepcopy(data["cards"][0]))
    result = web.parse_snapshot(data, "城市交通")
    assert len(result) == 1 and result[0]["thumbnail_url"] == "" and result[0]["publisher_url"] == ""
    data["cards"][0]["thumbnail"] = ""
    assert web.parse_snapshot(data, "城市交通")[0]["thumbnail_url"] == ""


@pytest.mark.parametrize("change", [
    {"url": web.PAGE + "?keyword=其他词&page=1"}, {"url": "https://foreign.example/all?keyword=城市交通"},
    {"page": 2}, {"page": None}, {"page": True}, {"cards": None},
])
def test_keyword_page_and_structure_must_match(change):
    with pytest.raises(ValueError):
        web.parse_snapshot(snapshot() | change, "城市交通")


@pytest.mark.parametrize("change", [
    {"title": "验证码_哔哩哔哩"}, {"url": "https://passport.bilibili.com/login"},
    {"cards": [], "text": "请完成安全验证后继续"}, {"cards": [], "text": "登录后才能搜索"},
])
def test_visible_access_wall_stops_without_retry(change):
    with pytest.raises(web.PublicSearchBlocked):
        web.parse_snapshot(snapshot() | change, "城市交通")


def test_optional_login_prompt_and_security_related_query_are_not_access_walls():
    data = snapshot() | {"url": web.PAGE + "?keyword=安全验证", "title": "安全验证-哔哩哔哩_bilibili"}
    assert len(web.parse_snapshot(data, "安全验证")) == 1


def test_empty_results_need_explicit_page_feedback():
    assert web.parse_snapshot(snapshot() | {"cards": [], "text": "没有找到相关内容"}, "城市交通") == []
    with pytest.raises(ValueError, match="完整"):
        web.parse_snapshot(snapshot() | {"cards": [], "text": "搜索 加载中"}, "城市交通")


@pytest.fixture
def isolated_browser(monkeypatch):
    state = {"processes": [], "calls": [], "snapshot": snapshot()}
    class Process:
        def __init__(self, args, **kwargs):
            state["args"], state["kwargs"] = args, kwargs
            self.code, self.terminated = None, 0
            folder = Path(next(arg.split("=", 1)[1] for arg in args if arg.startswith("--user-data-dir=")))
            state["profile"] = folder
            (folder / "DevToolsActivePort").write_text("12345\n/devtools/browser/test\n")
            state["processes"].append(self)
        def poll(self):
            return self.code
        def terminate(self):
            self.terminated += 1
            self.code = 0
        def kill(self):
            self.code = -9
        def wait(self, **kwargs):
            if self.code is None:
                raise subprocess.TimeoutExpired("own-browser", kwargs.get("timeout", 1))
            return self.code
    class WebSocket:
        def close(self):
            state["ws_closed"] = True
    class CDP:
        def __init__(self, url):
            assert url == "ws://127.0.0.1:12345/devtools/browser/test"
            self.ws = WebSocket()
        def call(self, method, params=None, **kwargs):
            state["calls"].append((method, params))
            if method == "Target.getTargets":
                return {"targetInfos": [{"type": "page", "targetId": "own-tab"}]}
            if method == "Target.attachToTarget":
                return {"sessionId": "own-session"}
            if method == "Browser.close" and state.get("graceful_close", True):
                state["processes"][0].code = 0
            return {}
        def evaluate(self, expression):
            assert expression == web.SNAPSHOT
            return state["snapshot"]
    class Version:
        def json(self):
            return {"webSocketDebuggerUrl": "ws://127.0.0.1:12345/devtools/browser/test"}
    def local_get(url, **kwargs):
        assert url == "http://127.0.0.1:12345/json/version" and kwargs["trust_env"] is False
        return Version()
    monkeypatch.setattr(web.baidu_web, "browser_binary", lambda: "/installed/chrome")
    monkeypatch.setattr(web.baidu_web, "CDP", CDP)
    monkeypatch.setattr(web.subprocess, "Popen", Process)
    monkeypatch.setattr(web.httpx, "get", local_get)
    return state


def test_browser_uses_disposable_profile_one_navigation_and_no_account_state(isolated_browser):
    state = isolated_browser
    result = web.read_public_page("城市交通")
    assert len(result) == 1 and len(state["processes"]) == 1
    assert "--headless=new" in state["args"] and "about:blank" == state["args"][-1]
    assert not any(token in " ".join(state["args"]) for token in ("Default", "baidu-browser", "AutomationControlled", "user-agent"))
    methods = [method for method, _ in state["calls"]]
    assert methods.count("Page.navigate") == 1 and methods[-1] == "Browser.close"
    assert not any("Cookie" in method or "Storage" in method for method in methods)
    assert state["ws_closed"] and state["processes"][0].terminated == 0
    assert not state["profile"].exists()


def test_browser_challenge_stops_and_cleans_up_without_another_navigation(isolated_browser):
    state = isolated_browser
    state["snapshot"] = snapshot() | {"title": "验证码_哔哩哔哩", "cards": []}
    with pytest.raises(web.PublicSearchBlocked):
        web.read_public_page("城市交通")
    assert sum(method == "Page.navigate" for method, _ in state["calls"]) == 1
    assert not state["profile"].exists() and state["ws_closed"]


def test_browser_hard_deadline_terminates_only_its_own_process(isolated_browser):
    state = isolated_browser
    state["snapshot"] = snapshot() | {"cards": [], "ready": "loading"}
    started = time.monotonic()
    with pytest.raises(ValueError, match="时限"):
        web.read_public_page("城市交通", timeout=.03)
    assert time.monotonic() - started < 1
    assert state["processes"][0].terminated == 1 and not state["profile"].exists()


@pytest.mark.parametrize("challenge", [False, True])
def test_windows_profile_lock_preserves_candidates_or_original_access_error(isolated_browser, monkeypatch, tmp_path, challenge):
    state = isolated_browser
    if challenge:
        state["snapshot"] = snapshot() | {"title": "验证码_哔哩哔哩", "cards": []}
    user_profile = tmp_path / "user-Chrome" / "Default"
    user_profile.mkdir(parents=True)
    sentinel = user_profile / "unrelated-user-file"
    sentinel.write_text("must remain untouched")
    original = web.tempfile.TemporaryDirectory
    attempts = []
    class InitiallyLockedDirectory(original):
        def cleanup(self):
            attempts.append(self.name)
            assert Path(self.name) == state["profile"] and Path(self.name) != user_profile
            if len(attempts) < 3:
                error = PermissionError("EdgeCoupons file handle is still closing")
                error.winerror = 32
                raise error
            return super().cleanup()
    monkeypatch.setattr(web.tempfile, "TemporaryDirectory", InitiallyLockedDirectory)
    if challenge:
        with pytest.raises(web.PublicSearchBlocked):
            web.read_public_page("城市交通")
    else:
        assert len(web.read_public_page("城市交通")) == 1
    assert len(attempts) == 3 and not state["profile"].exists()
    assert state["processes"][0].terminated == 0
    assert sentinel.read_text() == "must remain untouched"


def test_persistent_windows_lock_defers_only_owned_directory_and_keeps_result(isolated_browser, monkeypatch):
    state = isolated_browser
    original = web.tempfile.TemporaryDirectory
    locked, scheduled, sleeps = [True], [], []
    class LockedDirectory(original):
        def cleanup(self):
            assert Path(self.name) == state["profile"]
            if locked[0]:
                raise PermissionError("WinError 32")
            return super().cleanup()
    class DeferredCleanup:
        def __init__(self, *, target, args, name, daemon):
            assert name == "jinggao-bili-profile-cleanup" and daemon
            self.target, self.args = target, args
        def start(self):
            scheduled.append(self)
    class Deadline:
        def __init__(self, *args):
            pass
        def start(self):
            pass
        def cancel(self):
            pass
    monkeypatch.setattr(web.tempfile, "TemporaryDirectory", LockedDirectory)
    monkeypatch.setattr(web.threading, "Timer", Deadline)
    monkeypatch.setattr(web.threading, "Thread", DeferredCleanup)
    monkeypatch.setattr(web.time, "sleep", sleeps.append)
    assert len(web.read_public_page("城市交通")) == 1
    assert state["profile"].exists() and len(scheduled) == 1
    assert sleeps == [.1, .3] and scheduled[0].args[1] == (1, 2, 4, 8)
    locked[0] = False
    assert scheduled[0].target(*scheduled[0].args)
    assert not state["profile"].exists() and state["processes"][0].terminated == 0


def test_unresponsive_own_browser_is_stopped_after_grace_period(isolated_browser):
    state = isolated_browser
    state["graceful_close"] = False
    assert len(web.read_public_page("城市交通")) == 1
    assert len(state["processes"]) == 1 and state["processes"][0].terminated == 1
    assert not state["profile"].exists()


def test_new_web_route_ignores_legacy_api_cooldown_and_caches_by_query_and_page(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(platform, "_bili_blocked_until", time.monotonic() + 1800)
    monkeypatch.setattr(web, "blocked_until", 0)
    calls = []
    def read(query, page, *, timeout):
        calls.append((query, page))
        return [{"id": str(page), "source_url": "https://www.bilibili.com/video/BV1pTap6UEai"}]
    monkeypatch.setattr(web, "read_public_page", read)
    monkeypatch.setattr(video, "_json", lambda *a, **kw: pytest.fail("retired API must not be called"))
    assert platform.bilibili_search("城市交通") == platform.bilibili_search("城市交通")
    assert platform.bilibili_search("城市交通", page=2)[0]["id"] == "2"
    assert calls == [("城市交通", 1), ("城市交通", 2)]


def test_search_allows_two_browsers_but_coalesces_same_query(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(web, "blocked_until", 0)
    monkeypatch.setattr(web, "SEARCH_SLOTS", threading.BoundedSemaphore(2))
    lock = threading.Lock()
    active, peak, calls = 0, 0, []
    def read(query, page, *, timeout):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append(query)
        time.sleep(.05)
        with lock:
            active -= 1
        return [{"id": query}]
    monkeypatch.setattr(web, "read_public_page", read)
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(platform.bilibili_search, ["甲", "乙", "甲", "丙", "丁"]))
    assert peak == 2 and sorted(calls) == ["丁", "丙", "乙", "甲"]


def test_queued_search_stops_if_active_request_detected_a_challenge(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(web, "blocked_until", 0)
    class Slot:
        def acquire(self, *, timeout):
            # Simulate another active request discovering the wall while this
            # caller waits for its independent browser slot.
            web.blocked_until = time.monotonic() + 1800
            return True
        def release(self):
            pass
    monkeypatch.setattr(web, "SEARCH_SLOTS", Slot())
    monkeypatch.setattr(web, "read_public_page", lambda *a: pytest.fail("queued request must not navigate"))
    with pytest.raises(ValueError, match="验证"):
        platform.bilibili_search("城市交通")


@pytest.mark.parametrize("failure", ["loading", "connection"])
def test_loading_or_connection_failure_has_short_backoff_and_no_retry(isolated_browser, monkeypatch, tmp_path, failure):
    state = isolated_browser
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(web, "blocked_until", 0)
    monkeypatch.setattr(web, "BROWSER_TIMEOUT", .03)
    state["snapshot"] = snapshot() | {"cards": [], "ready": "complete"}
    if failure == "connection":
        def disconnected(*args, **kwargs):
            raise web.httpx.ConnectTimeout("anonymous browser connection timed out")
        monkeypatch.setattr(web.httpx, "get", disconnected)
    with pytest.raises(web.PublicSearchUnavailable):
        platform.bilibili_search("城市交通")
    until = web.unavailable_until
    assert 59 < until - time.monotonic() <= 60 and web.blocked_until == 0
    with pytest.raises(web.PublicSearchUnavailable, match="60秒"):
        platform.bilibili_search("不同关键词")
    assert len(state["processes"]) == 1 and web.unavailable_until == until
    assert not state["profile"].exists()
    monkeypatch.setattr(web, "unavailable_until", time.monotonic() - 1)
    monkeypatch.setattr(web, "read_public_page", lambda *a, **kw: [])
    assert platform.bilibili_search("城市交通") == []  # Failed searches were not cached.


def test_queued_search_rechecks_short_backoff_without_navigation(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(web, "blocked_until", 0)
    releases, cooldown = [], []
    class Slot:
        def acquire(self, *, timeout):
            web.unavailable_until = time.monotonic() + 60
            cooldown.append(web.unavailable_until)
            return True
        def release(self):
            releases.append(True)
    monkeypatch.setattr(web, "SEARCH_SLOTS", Slot())
    monkeypatch.setattr(web, "read_public_page", lambda *a, **kw: pytest.fail("queued request must not navigate"))
    with pytest.raises(web.PublicSearchUnavailable, match="60秒"):
        platform.bilibili_search("城市交通")
    assert releases == [True] and web.unavailable_until == cooldown[0]
    assert web.blocked_until == 0


def test_queue_and_navigation_share_budget_success_and_empty_do_not_backoff(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(web, "blocked_until", 0)
    clock, waits, navigations, releases = [100.0], [], [], []
    monkeypatch.setattr(platform.time, "monotonic", lambda: clock[0])
    class Slot:
        def acquire(self, *, timeout):
            waits.append(timeout)
            clock[0] += 11 if len(waits) < 3 else 18
            return True
        def release(self):
            releases.append(True)
    def read(query, page, *, timeout):
        navigations.append(timeout)
        return [] if query == "空结果" else [{"id": "public-video"}]
    monkeypatch.setattr(web, "SEARCH_SLOTS", Slot())
    monkeypatch.setattr(web, "read_public_page", read)
    assert platform.bilibili_search("空结果") == []
    assert platform.bilibili_search("城市交通") == [{"id": "public-video"}]
    with pytest.raises(ValueError, match="等待已达到时限"):
        platform.bilibili_search("队列超时")
    assert waits == [18, 18, 18] and navigations == [7, 7] and len(releases) == 3
    assert web.unavailable_until == web.blocked_until == 0
