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

        def __init__(self, url):
            self.events, self.calls, self.clicks = [], [], []
            self.ws = self
            self.closed = False

        def call(self, method, params=None, **kwargs):
            self.calls.append((method, params))
            if method == "Browser.setDownloadBehavior" and params["behavior"] == "allowAndName":
                self.folder = Path(params["downloadPath"])
            return {"targetId": "created-tab", "sessionId": "created-session",
                    "frameTree": {"frame": {"id": "our-frame"}}}

        def wait_element(self, expression, check):
            check()
            return {"x": 10, "y": 20, "selector": expression}

        def click(self, expression):
            self.clicks.append(expression["selector"])

        def type_text(self, text):
            self.typed = text

        def evaluate(self, expression):
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
