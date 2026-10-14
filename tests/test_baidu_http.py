import json
from contextlib import contextmanager

import httpx
import pytest
import websocket

from jinggao import baidu_http as api
from jinggao import baidu_web, config, media, network


VOICE = {"display_name": "测试公共音色", "id": 3596, "voice_type": 1, "status": 1,
         "clone_type": "ptts", "tts_timbre_id": "7007", "tts_mid": "test", "tts_pdt": "10159"}


@pytest.fixture
def flow(monkeypatch):
    state = {"calls": [], "auth": {"errno": 0, "data": {"auth": "never-log-this"}},
             "events": [b"mp3 data", json.dumps({"type": "FINISH", "err_no": 0})], "sends": []}

    def handler(request):
        state["calls"].append(request.url.path)
        if request.url.path == api.VOICE_LIST:
            value = {"errno": 0, "data": {"list": [VOICE]}}
        elif request.url.path == api.AUTH:
            value = state["auth"]
        elif request.url.path == api.API_STATUS:
            value = state.get("account_status", {"errno": 0, "data": -1})
        else:
            pytest.fail("Unexpected endpoint")
        return httpx.Response(200, json=value)

    @contextmanager
    def session(**kwargs):
        with httpx.Client(base_url=api.ORIGIN, transport=httpx.MockTransport(handler)) as client:
            yield client, "test-device"

    class Socket:
        def send(self, value):
            state["sends"].append(json.loads(value))

        def settimeout(self, value):
            pass

        def recv(self):
            event = state["events"].pop(0) if state["events"] else ""
            if isinstance(event, Exception):
                raise event
            return event

        def close(self):
            state["closed"] = True

    def connect(*args, **kwargs):
        state["connections"] = state.get("connections", 0) + 1
        return Socket()

    def convert(args, **kwargs):
        from pathlib import Path
        assert Path(args[args.index("-i") + 1]).read_bytes() == b"mp3 data"
        Path(args[-1]).write_bytes(b"converted wav")

    monkeypatch.setattr(api, "_session", session)
    monkeypatch.setattr(config, "read", lambda: {"baidu_voice_name": VOICE["display_name"]})
    monkeypatch.setattr(websocket, "create_connection", connect)
    # On macOS network.websocket_connection opens a raw TCP socket to tts.baidu.com
    # before calling websocket.create_connection; fake the whole connector.
    monkeypatch.setattr(network, "websocket_connection", connect)
    monkeypatch.setattr(media, "run", convert)
    monkeypatch.setattr(media, "probe", lambda path: {"format": {"duration": "7.68"}})
    return state


def test_whole_text_is_submitted_once_and_no_page_controls(flow, tmp_path):
    target = tmp_path / "voice.wav"
    api.generate("全文<&>\n测试", target, lambda: None)
    assert flow["calls"] == [api.VOICE_LIST, api.AUTH]
    assert flow["connections"] == 1
    assert len(flow["sends"]) == 1
    data = flow["sends"][0]["data"]
    assert data["tex"] == "<speak>全文&lt;&amp;&gt;测试</speak>"
    assert data["auth"] == "never-log-this"
    assert target.read_bytes() == b"converted wav"
    assert not list(tmp_path.glob(".baidu-*"))
    assert flow["closed"]


def test_cost_hook_precedes_authorization_and_is_called_once(flow, tmp_path):
    calls = []
    def before_submit():
        assert flow["calls"] == [api.VOICE_LIST]
        assert not flow.get("connections")
        calls.append(1)
    api.generate("完整原稿", tmp_path / "voice.wav", lambda: None, before_submit=before_submit)
    assert calls == [1]


@pytest.mark.parametrize(("response", "match"), [
    ({"errno": 1000000, "errmsg": "今日调用次数已达上限"}, "今日调用次数"),
    ({"errno": 0, "data": {"anti_flag": 1, "auth": "blocked"}}, "人机验证"),
    ({"errno": 0, "data": {"anti_flag": "1"}}, "人机验证"),
    ({"errno": 0, "data": {}}, "未返回配音授权"),
    ({"errno": 0, "data": {"auth": ""}}, "有效配音授权"),
    ({"errno": 302, "errmsg": "请登录"}, "请登录"),
])
def test_quota_verification_and_expired_login_never_submit_or_fallback(flow, tmp_path, response, match):
    flow["auth"] = response
    with pytest.raises(ValueError, match=match):
        api.generate("测试", tmp_path / "voice.wav", lambda: None)
    assert flow["calls"].count(api.AUTH) == 1
    assert not flow.get("connections")
    assert not list(tmp_path.iterdir())
    assert not baidu_web._generation_lock.locked()


@pytest.mark.parametrize("events", [
    [b"partial", ""],
    [b"partial", json.dumps({"type": "ERROR", "err_no": 10, "err_msg": "失败"})],
    [b"partial", json.dumps({"type": "FINISH", "err_no": 10, "err_msg": "失败"})],
    [b"partial", websocket.WebSocketConnectionClosedException("connection lost")],
    [json.dumps({"type": "FINISH", "err_no": 0})],
    ["malformed"],
])
def test_failed_or_truncated_stream_preserves_old_output_and_never_resubmits(flow, tmp_path, events):
    flow["events"] = events
    target = tmp_path / "voice.wav"
    target.write_bytes(b"old valid audio")
    with pytest.raises(ValueError):
        api.generate("测试", target, lambda: None)
    assert target.read_bytes() == b"old valid audio"
    assert flow["connections"] == 1
    assert sum(e["type"] == "START" for e in flow["sends"]) == 1
    assert not list(tmp_path.glob(".baidu-*"))
    assert flow["closed"]


def test_cancellation_during_stream_sends_cancel_and_releases_lock(flow, tmp_path):
    checks = 0

    def check():
        nonlocal checks
        checks += 1
        if flow.get("sends") and flow["sends"][0]["type"] == "START":
            raise media.Cancelled()

    with pytest.raises(media.Cancelled):
        api.generate("测试", tmp_path / "voice.wav", check)
    assert flow["sends"][-1]["type"] == "CANCEL"
    assert flow["closed"]
    assert not baidu_web._generation_lock.locked()
    assert not list(tmp_path.iterdir())


def test_status_does_not_synthesize_or_return_credentials(flow):
    result = api.status()
    assert result["ready"]
    assert flow["calls"] == [api.API_STATUS, api.VOICE_LIST]
    assert "never-log-this" not in json.dumps(result)


def test_status_rejects_stale_cookie_instead_of_trusting_public_voice_list(flow):
    flow["account_status"] = {"errno": 302, "errmsg": "请登陆"}
    with pytest.raises(ValueError, match="请登陆"):
        api.status()
    assert flow["calls"] == [api.API_STATUS]
    assert not flow.get("connections")


def test_only_service_cookies_read_from_dedicated_profile(monkeypatch):
    calls = []

    class Browser:
        session = None

        @property
        def ws(self):
            return self

        def call(self, method, params=None, **kwargs):
            calls.append((method, params))
            if method == "Browser.getVersion":
                return {"userAgent": "Chrome test"}
            if method == "Target.getTargets":
                return {"targetInfos": [{"type": "page", "url": api.ORIGIN, "targetId": "service"}]}
            if method == "Target.attachToTarget":
                return {"sessionId": "test"}
            if method == "Network.getCookies":
                return {"cookies": [{"name": "BDUSS", "value": "private", "domain": ".baidu.com", "path": "/"}]}
            pytest.fail(method)

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(baidu_web, "endpoint", lambda: "ws://127.0.0.1:1/dedicated")
    monkeypatch.setattr(baidu_web, "CDP", lambda _: Browser())
    jar, ua, cuid = api._credentials(open_if_needed=False)
    assert jar.get("BDUSS") == "private"
    assert ("Network.getCookies", {"urls": [api.ORIGIN + "/"]}) in calls
    assert not any(method.startswith(("Input.", "Runtime.", "Page.")) for method, _ in calls)
    assert calls[-1] == ("close", None)


def test_http_failure_does_not_leak_signed_url_or_retry():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(403)

    with httpx.Client(base_url=api.ORIGIN, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError) as error:
            api._post(client, api.AUTH, {"text": "私人稿件"})
    assert len(calls) == 1
    assert "sign=" not in str(error.value)
    assert "私人稿件" not in str(error.value)
