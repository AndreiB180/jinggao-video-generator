import json
from pathlib import Path

import pytest

from jinggao import baidu_web, config, media, paths


@pytest.fixture
def browser_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(baidu_web, "endpoint", lambda: "ws://127.0.0.1:9222/devtools/browser/test")
    monkeypatch.setattr(config, "read", lambda: {"baidu_voice_name": "测试公共音色"})

    class FakeBrowser:
        free = True
        status = "completed"
        received = 42
        logged_in = True
        cards = ["测试公共音色"]

        def __init__(self, url):
            self.events, self.calls, self.clicks = [], [], []
            self.ws = self
            self.closed = False

        def call(self, method, params=None, **kwargs):
            self.calls.append((method, params))
            if method == "Storage.getCookies":
                return {"cookies": [{"name": "BDUSS", "value": "login", "domain": ".baidu.com"}] if self.logged_in else
                        [{"name": "BAIDUID", "value": "visitor", "domain": ".baidu.com"}]}
            if method == "Browser.setDownloadBehavior" and params["behavior"] == "allowAndName":
                self.folder = Path(params["downloadPath"])
            return {"targetId": "created-tab", "sessionId": "created-session",
                    "frameTree": {"frame": {"id": "our-frame"}}}

        def wait_element(self, expression, check, timeout=35):
            check()
            return {"x": 10, "y": 20, "selector": expression}

        def click(self, expression):
            self.clicks.append(expression["selector"])

        def type_text(self, text):
            self.typed = text

        def evaluate(self, expression):
            if "jinggao-cards" in expression:
                return {"hash": "#/clone-voice", "loading": False, "empty": not self.cards,
                        "cards": [{"name": n, "description": ""} for n in self.cards]}
            if "location.hash" in expression:
                return {"hash": "#/clone-voice"}
            if expression.endswith("textarea')?.value"):
                return self.typed
            return self.free

        def receive(self):
            (self.folder / "our-download").write_bytes(b"expected audio")
            self.events += [
                {"method": "Browser.downloadWillBegin", "params": {"frameId": "other-frame", "guid": "other"}},
                {"method": "Browser.downloadProgress", "params": {"guid": "other", "state": "completed"}},
                {"method": "Browser.downloadWillBegin", "params": {"frameId": "our-frame", "guid": "our-download"}},
                {"method": "Browser.downloadProgress", "params": {"guid": "our-download", "state": self.status,
                                                                    "receivedBytes": self.received}},
            ]

        def close(self):
            self.closed = True

    browser = FakeBrowser("")
    monkeypatch.setattr(baidu_web, "CDP", lambda url: browser)
    return browser


def test_only_our_download_is_imported_and_profile_is_restored(browser_flow, tmp_path, monkeypatch):
    target = tmp_path / "voice.wav"

    def convert(args, **kwargs):
        assert Path(args[args.index("-i") + 1]).read_bytes() == b"expected audio"
        target.write_bytes(b"converted")

    monkeypatch.setattr(media, "run", convert)
    baidu_web.generate("配音测试", target, lambda: None)
    assert target.read_bytes() == b"converted"
    assert browser_flow.closed
    assert ("Target.closeTarget", {"targetId": "created-tab"}) in browser_flow.calls
    assert ("Browser.setDownloadBehavior", {"behavior": "default", "eventsEnabled": False}) in browser_flow.calls
    assert not list((tmp_path / "baidu-downloads").iterdir())


def test_page_quota_response_stops_without_waiting_for_download(browser_flow, tmp_path, monkeypatch):
    original = browser_flow.call
    def call(method, params=None, **kwargs):
        if method == "Network.getResponseBody":
            return {"body": '{"errno":1000000,"errmsg":"今日调用次数已达上限"}'}
        return original(method, params, **kwargs)
    def receive():
        browser_flow.events += [
            {"method": "Network.responseReceived", "params": {"requestId": "our-request", "response": {
                "url": "https://huibo.baidu.com/ai_anchor/tts/check_auth", "status": 200}}},
            {"method": "Network.loadingFinished", "params": {"requestId": "our-request"}},
        ]
    monkeypatch.setattr(browser_flow, "call", call)
    monkeypatch.setattr(browser_flow, "receive", receive)
    with pytest.raises(ValueError, match="今日调用次数已达上限"):
        baidu_web.generate("配音测试", tmp_path/"voice.wav", lambda: None)
    assert not (tmp_path/"voice.wav").exists()
    assert browser_flow.closed


def test_changed_paid_flow_is_not_submitted(browser_flow, tmp_path):
    browser_flow.free = False
    with pytest.raises(ValueError, match="核对收费方式"):
        baidu_web.generate("配音测试", tmp_path / "voice.wav", lambda: None)
    assert not any("button" in expression for expression in browser_flow.clicks)
    assert browser_flow.closed
    assert not any(method == "Target.closeTarget" for method, _ in browser_flow.calls)


@pytest.mark.parametrize("free", [True, False])
def test_cost_hook_runs_only_after_free_check_before_download(browser_flow, tmp_path, monkeypatch, free):
    browser_flow.free = free
    calls = []
    def before_submit():
        assert not any("button" in expression for expression in browser_flow.clicks)
        calls.append(1)
    monkeypatch.setattr(media, "run", lambda *args, **kwargs: None)
    if free:
        baidu_web.generate("配音测试", tmp_path / "voice.wav", lambda: None, before_submit=before_submit)
        assert calls == [1]
    else:
        with pytest.raises(ValueError, match="核对收费方式"):
            baidu_web.generate("配音测试", tmp_path / "voice.wav", lambda: None, before_submit=before_submit)
        assert calls == []


@pytest.mark.parametrize(("status", "size", "message"), [
    ("canceled", 42, "下载已取消"),
    ("inProgress", 101*1024*1024, "超过 100 MB"),
])
def test_download_failure_cancels_own_job(browser_flow, tmp_path, status, size, message):
    browser_flow.status, browser_flow.received = status, size
    with pytest.raises(ValueError, match=message):
        baidu_web.generate("配音测试", tmp_path / "voice.wav", lambda: None)
    assert ("Browser.cancelDownload", {"guid": "our-download"}) in browser_flow.calls
    assert not (tmp_path / "voice.wav").exists()
    assert browser_flow.closed


class Clock:
    """Fake time for the page waits (seconds pass only when the code sleeps or waits for events)."""
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_not_logged_in_shows_the_browser_and_says_what_to_do(browser_flow, tmp_path, monkeypatch):
    browser_flow.logged_in = False
    shown = []
    monkeypatch.setattr(baidu_web, "_present", lambda url, pid=None: shown.append(url))
    with pytest.raises(ValueError, match="扫码登录"):
        baidu_web.generate("配音测试", tmp_path / "voice.wav", lambda: None)
    assert shown == ["ws://127.0.0.1:9222/devtools/browser/test"]
    assert not any(method == "Target.createTarget" for method, _ in browser_flow.calls)


@pytest.mark.parametrize("cards", [["别的音色", "另一个音色"], []])     # other results, or the page's empty state
def test_a_voice_no_longer_listed_fails_clearly_without_submitting(browser_flow, tmp_path, monkeypatch, cards):
    browser_flow.cards = cards
    monkeypatch.setattr(baidu_web, "time", Clock())
    before = []
    with pytest.raises(baidu_web.VoiceMissing, match="找不到「测试公共音色」.*刷新音色列表"):
        baidu_web.generate("配音测试", tmp_path / "voice.wav", lambda: None, before_submit=lambda: before.append(1))
    assert before == [] and not browser_flow.clicks[1:]      # only the search box was clicked
    assert ("Target.closeTarget", {"targetId": "created-tab"}) in browser_flow.calls   # no tab left behind
    assert browser_flow.closed


def test_closed_browser_is_opened_and_a_logged_in_profile_carries_on(browser_flow, tmp_path, monkeypatch):
    addresses = iter([None, "ws://127.0.0.1:9222/devtools/browser/test"])
    monkeypatch.setattr(baidu_web, "endpoint", lambda: next(addresses))
    opened = []
    monkeypatch.setattr(baidu_web, "open_browser", lambda: opened.append(1))
    monkeypatch.setattr(media, "run", lambda *args, **kwargs: None)
    baidu_web.generate("配音测试", tmp_path / "voice.wav", lambda: None)
    assert opened == [1]


def test_browser_order_setting_and_missing_message(monkeypatch, tmp_path):
    edge, chrome = tmp_path / "msedge.exe", tmp_path / "chrome.exe"
    edge.write_text("")
    chrome.write_text("")
    monkeypatch.setattr(baidu_web, "_installed", lambda: {"edge": [tmp_path / "gone.exe", edge], "chrome": [chrome]})
    monkeypatch.setattr(baidu_web.os, "name", "nt")
    settings = {"baidu_browser": "auto"}
    monkeypatch.setattr(config, "read", lambda: settings)
    assert baidu_web.browser_candidates() == [("edge", edge), ("chrome", chrome)]
    settings["baidu_browser"] = "chrome"
    assert baidu_web.browser_candidates() == [("chrome", chrome)]
    monkeypatch.setattr(baidu_web, "_installed", lambda: {"edge": [], "chrome": []})
    with pytest.raises(ValueError, match="没有找到 Google Chrome"):
        baidu_web.browser_candidates()
    settings["baidu_browser"] = "auto"
    with pytest.raises(ValueError, match="没有找到 Microsoft Edge 或 Google Chrome"):
        baidu_web.browser_candidates()


def test_open_browser_presents_a_running_one_and_falls_back_to_the_other_browser(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    presented, started, stopped = [], [], []
    monkeypatch.setattr(baidu_web, "_present", lambda url, pid=None: presented.append(url))
    monkeypatch.setattr(baidu_web, "running", lambda: ("edge", "ws://running"))
    assert "已打开" in baidu_web.open_browser()["message"] and presented == ["ws://running"]

    class Process:
        pid = 7
        def __init__(self, code):
            self.code = code
        def poll(self):
            return self.code
    monkeypatch.setattr(baidu_web, "running", lambda: None)
    monkeypatch.setattr(baidu_web, "browser_candidates", lambda: [("edge", Path("e")), ("chrome", Path("c"))])
    monkeypatch.setattr(baidu_web, "_held", lambda folder: False)
    monkeypatch.setattr(baidu_web, "_stop_profile", lambda folder: stopped.append(folder))
    monkeypatch.setattr(baidu_web, "_start", lambda kind, binary: started.append(kind) or Process(1 if kind == "edge" else None))
    monkeypatch.setattr(baidu_web, "_wait_ready", lambda folder, process: None if process.code == 1 else "ws://chrome")
    result = baidu_web.open_browser()
    assert started == ["edge", "chrome"] and result["browser"] == "Google Chrome"
    assert presented[-1] == "ws://chrome"
    monkeypatch.setattr(baidu_web, "_wait_ready", lambda folder, process: None)
    with pytest.raises(ValueError, match="没有启动成功（试过 Microsoft Edge、Google Chrome）"):
        baidu_web.open_browser()


def test_status_reports_browser_and_login_without_opening_anything(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(config, "read", lambda: {"baidu_voice_name": "磁性专业很灵动", "baidu_browser": "auto"})
    monkeypatch.setattr(baidu_web, "browser_candidates", lambda: [("edge", Path("e"))])
    monkeypatch.setattr(baidu_web, "running", lambda: None)
    monkeypatch.setattr(baidu_web, "open_browser", lambda: pytest.fail("status must not open the browser"))
    closed = baidu_web.status()
    assert closed["open"] is False and closed["logged_in"] is None and "没有打开" in closed["message"]

    class Browser:
        def __init__(self, url):
            self.ws = self
        def call(self, method, params=None, **kwargs):
            assert method == "Storage.getCookies"
            return {"cookies": [{"name": "BDUSS_BFESS", "value": "SECRET-LOGIN", "domain": ".baidu.com"}]}
        def close(self):
            pass
    monkeypatch.setattr(baidu_web, "running", lambda: ("edge", "ws://x"))
    monkeypatch.setattr(baidu_web, "CDP", Browser)
    ready = baidu_web.status()
    assert ready["logged_in"] is True and ready["browser"] == "Microsoft Edge" and "已登录" in ready["message"]
    assert "SECRET-LOGIN" not in json.dumps(ready, ensure_ascii=False)


def test_voice_list_from_the_page_keeps_usable_public_voices():
    body = {"errno": 0, "data": {"list": [
        {"display_name": "磁性专业很灵动", "status": 1, "description": "新闻解说", "category_list": "男、青年",
         "show_url": "https://huibo-static.cdn.bcebos.com/voice/a.mp3"},
        {"display_name": "磁性专业很灵动", "status": 1},
        {"display_name": "已下架", "status": 2},
        {"display_name": "外站样音", "status": 1, "show_url": "https://example.com/a.mp3"},
    ]}}
    voices = baidu_web._voice_items(body)
    assert [v["name"] for v in voices] == ["磁性专业很灵动", "外站样音"]
    assert voices[0]["sample"].startswith("https://huibo-static") and voices[0]["category"] == "男、青年"
    assert voices[1]["sample"] == ""
    assert baidu_web._voice_items({"data": {"list": [{"id": 1}]}}) == []


def test_refresh_reads_the_list_the_page_loads_and_caches_it(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(baidu_web, "endpoint", lambda: "ws://x")
    monkeypatch.setattr(baidu_web, "time", Clock())
    listing = {"errno": 0, "data": {"list": [{"display_name": "甲", "status": 1}, {"display_name": "乙", "status": 1}]}}

    class Browser:
        def __init__(self, url):
            self.ws, self.events, self.session, self.calls = self, [], None, []
        def call(self, method, params=None, **kwargs):
            self.calls.append(method)
            if method == "Storage.getCookies":
                return {"cookies": [{"name": "BDUSS", "value": "x", "domain": ".baidu.com"}]}
            if method == "Network.getResponseBody":
                return {"body": json.dumps(listing if params["requestId"] == "list" else {"errno": 0, "data": {}})}
            return {"targetId": "tab", "sessionId": "s"}
        def receive(self):
            baidu_web.time.sleep(.5)
            if "Page.navigate" in self.calls and not any(e.get("seen") for e in self.events):
                self.events += [
                    {"method": "Network.responseReceived", "params": {"requestId": "other", "response": {
                        "url": "https://huibo.baidu.com/ai_anchor/user/info"}}},
                    {"method": "Network.responseReceived", "params": {"requestId": "list", "response": {
                        "url": "https://huibo.baidu.com/ai_anchor/tts/tts_list"}}},
                    {"method": "Network.loadingFinished", "params": {"requestId": "other"}},
                    {"method": "Network.loadingFinished", "params": {"requestId": "list"}, "seen": True}]
        def evaluate(self, expression):
            return {"hash": "#/clone-voice", "cards": [{"name": "甲"}, {"name": "丙"}]}
        def close(self):
            pass
    monkeypatch.setattr(baidu_web, "CDP", Browser)
    saved = baidu_web.refresh_voices()
    assert [v["name"] for v in saved["voices"]] == ["甲", "乙", "丙"]
    assert baidu_web.cached_voices()["voices"] == saved["voices"]


def test_refresh_needs_a_login(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(baidu_web, "endpoint", lambda: None)
    with pytest.raises(ValueError, match="打开一镜专用浏览器"):
        baidu_web.refresh_voices()


def test_a_fresh_choice_of_baidu_survives_the_one_time_migration(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")
    paths.DATA.mkdir(parents=True)
    config.update({"tts_provider": "baidu_web", "baidu_voice_name": "甲"})     # new install, no flags yet
    config.migrate()
    assert config.read()["tts_provider"] == "baidu_web"
    with pytest.raises(ValueError, match="专用浏览器"):
        config.update({"baidu_browser": "firefox"})


def test_only_our_own_profile_is_ever_stopped():
    folder = Path("/data/baidu-browser")
    assert baidu_web._uses_profile('"/x/msedge" --user-data-dir="/data/baidu-browser" --remote-debugging-port=9334', folder)
    assert baidu_web._uses_profile('"/x/msedge" "--user-data-dir=/data/baidu-browser" --no-first-run', folder)
    assert baidu_web._uses_profile("/x/chrome --type=renderer --user-data-dir=/data/baidu-browser", folder)
    assert not baidu_web._uses_profile("/x/msedge --user-data-dir=/data/baidu-browser-chrome", folder)
    assert not baidu_web._uses_profile("/x/msedge --profile-directory=Default", folder)    # the user's own browser
