"""Experimental same-account Baidu web API, independent of page controls.

CDP is used only to read huibo cookies from Jinggao's dedicated browser profile.
Credentials and short-lived TTS authorization stay in memory. This is not the
commercial OpenAPI. Challenges, quotas and ambiguous submissions are not retried.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode, urlparse
from xml.sax.saxutils import escape

import httpx
import websocket

from . import baidu_web, config, network, paths

ORIGIN = "https://huibo.baidu.com"
VOICE_LIST = "/ai_anchor/tts/tts_list"
AUTH = "/ai_anchor/tts/check_auth"
API_STATUS = "/ai_anchor/video/anchor/rights/video-api/status"
MAX_BYTES = 100 * 1024 * 1024


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _signed_query(path: str, body: str) -> dict:
    # Public frontend protocol constants; these are not the customer's AK/SK.
    stamp = str(int(time.time()))
    query = {"ak": "huibo_fe", "timestamp": stamp}
    value = "huibo_fe123qaz_huibo_fe_321plm" + stamp + path + urlencode(sorted(query.items())) + body
    query["sign"] = hashlib.md5(value.encode()).hexdigest()
    return query


def _data(value: dict):
    if not isinstance(value, dict):
        raise ValueError("一镜接口返回格式已变化，请检查服务或使用导入配音")
    data = value.get("data")
    if isinstance(data, dict) and data.get("anti_flag") not in (None, False, 0, "0", ""):
        raise ValueError("一镜要求人机验证，请在专用浏览器完成验证后再试")
    if value.get("errno") not in (0, "0"):
        error = str(value.get("errmsg") or "登录已失效或接口不可用")[:200]
        raise ValueError("一镜拒绝请求：" + error)
    return data


def _credentials(*, open_if_needed: bool):
    url = baidu_web.endpoint()
    if not url and open_if_needed:
        baidu_web.open_browser()
        url = baidu_web.endpoint()
    if not url:
        raise ValueError("请先打开一镜专用浏览器并登录，再检查接口")
    browser = None
    tab = None
    try:
        browser = baidu_web.CDP(url)
        ua = browser.call("Browser.getVersion")["userAgent"]
        pages = browser.call("Target.getTargets")["targetInfos"]
        page = next((p for p in pages if p["type"] == "page" and
                     urlparse(p["url"]).hostname == "huibo.baidu.com"), None)
        if page:
            target_id = page["targetId"]
        else:
            tab = browser.call("Target.createTarget", {"url": "about:blank"})["targetId"]
            target_id = tab
        browser.session = browser.call("Target.attachToTarget", {
            "targetId": target_id, "flatten": True})["sessionId"]
        cookies = browser.call("Network.getCookies", {"urls": [ORIGIN + "/"]}, page=True)["cookies"]
        if not any(c["name"] in ("BDUSS", "BDUSS_BFESS") and c["value"] for c in cookies):
            raise ValueError("一镜尚未登录，请在专用浏览器登录后再试")
        jar = httpx.Cookies()
        for cookie in cookies:
            jar.set(cookie["name"], cookie["value"], domain=cookie["domain"], path=cookie["path"])
        device = next((c["value"] for name in ("BAIDUID", "BAIDUID_BFESS", "BIDUPSID")
                       for c in cookies if c["name"] == name), "jinggao-local")
        cuid = "pc-" + hashlib.md5(device.encode()).hexdigest()
        return jar, ua, cuid
    except (websocket.WebSocketException, OSError, KeyError):
        raise ValueError("一镜登录浏览器连接失败，请重新打开专用浏览器") from None
    finally:
        if browser:
            try:
                if tab:
                    browser.call("Target.closeTarget", {"targetId": tab})
            finally:
                browser.ws.close()


@contextmanager
def _session(*, open_if_needed=True):
    cookies, ua, cuid = _credentials(open_if_needed=open_if_needed)
    with network.client(base_url=ORIGIN, cookies=cookies, follow_redirects=False, timeout=20,
                      headers={"User-Agent": ua, "Origin": ORIGIN, "Referer": baidu_web.PAGE,
                               "Content-Type": "application/json; charset=utf-8",
                               "hbx-platform-type-header": "1", "hbx-platform-version-header": "2"}) as client:
        yield client, cuid


def _post(client, path, payload):
    body = _json(payload)
    try:
        response = client.post(path, params=_signed_query(path, body), content=body)
        response.raise_for_status()
        value = response.json()
    except (httpx.HTTPError, ValueError):
        raise ValueError("一镜接口连接失败或返回异常；请检查网络和登录状态。未自动重复提交") from None
    return _data(value)


def _voice(client, name: str) -> tuple[dict, int]:
    result = _post(client, VOICE_LIST, {"mood": "", "age": 0, "gender": 0})
    voices = result.get("list", []) if isinstance(result, dict) else []
    matches = [v for v in voices if v.get("display_name") == name]
    if len(matches) != 1:
        raise ValueError("未找到唯一的公共音色，请核对完整名称：" + name)
    voice = matches[0]
    if voice.get("status") != 1 or voice.get("voice_type") != 1:
        raise ValueError("选中的一镜公共音色当前不可用")
    if voice.get("clone_type") != "ptts" or str(voice.get("tts_timbre_id")) not in ("7006", "7007"):
        raise ValueError("该音色尚未接入自动接口，请选「磁性专业很灵动」或使用网页配音")
    return voice, len(voices)


def status() -> dict:
    """No synthesis or quota-consuming request; do not expose auth material."""
    with _session(open_if_needed=False) as (client, _):
        # Public voice metadata alone does not prove the session is still valid.
        # This read-only account route rejects requests without a valid login.
        _post(client, API_STATUS, {})
        voice, count = _voice(client, config.read()["baidu_voice_name"].strip())
        return {"ready": True, "voice": voice["display_name"], "voice_count": count,
                "message": f"登录态接口可用，已找到音色「{voice['display_name']}」。未生成配音。"}


def _params(text: str, voice: dict, cuid: str) -> dict:
    now = str(int(time.time() * 1000))
    mid = voice.get("tts_mid", "")
    return dict(sn=f"huibo-ttsvc_{mid}-{cuid}-{now}", ctp=10, pdt=int(voice.get("tts_pdt") or 10159),
                multiplex=1, tex="<speak>" + escape(text.replace("\r", "").replace("\n", "")) + "</speak>",
                cuid=cuid, lip=0, spd=5, pit=5, vol=5, per=int(voice["tts_timbre_id"]), aue=3,
                ver="3.5.7.eba3e85c", rate=64, lan="ZH", xml=1, mode=0,
                text_ctrl='{"llm_nlu":1}', time=now, audio_ctrl=_json({"mid": mid}), paragraph=1)


def _stream(params: dict, auth: str, target: Path, check, *, timeout: float):
    ws = None
    complete = False
    size = 0
    deadline = time.monotonic() + timeout
    try:
        check()
        ws = network.websocket_connection("wss://tts.baidu.com/opentts?" + urlencode({"sn": params["sn"]}),
                                         timeout=15, subprotocols=["websocket"], origin=ORIGIN)
        check()
        ws.send(_json({"type": "START", "data": params | {"auth": auth}}))
        ws.settimeout(1)
        with target.open("wb") as output:
            while time.monotonic() < deadline:
                check()
                try:
                    message = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not message:
                    break
                if isinstance(message, bytes):
                    size += len(message)
                    if size > MAX_BYTES:
                        raise ValueError("一镜音轨超过 100 MB，请缩短稿件")
                    output.write(message)
                    continue
                try:
                    event = json.loads(message)
                except ValueError:
                    raise ValueError("一镜音频流格式已变化") from None
                if not isinstance(event, dict):
                    raise ValueError("一镜音频流格式已变化")
                if event.get("type") == "ERROR" or event.get("err_no", 0) not in (0, "0"):
                    raise ValueError("一镜音频生成失败：" + str(event.get("err_msg") or "请检查账号额度")[:200])
                if event.get("type") == "FINISH":
                    complete = True
                    break
        if not complete or not size:
            raise ValueError("一镜音频流未完整结束，未保存残缺音轨；请检查网络后重试")
    except (websocket.WebSocketException, OSError):
        raise ValueError("一镜音频连接中断，未自动重复生成；请检查网络后重试") from None
    finally:
        if ws:
            if not complete:
                try:
                    ws.send(_json({"type": "CANCEL"}))
                except (websocket.WebSocketException, OSError):
                    pass
            ws.close()


def generate(text: str, target: Path, check, *, before_submit=None):
    from . import media

    if not 1 <= len(text) <= 5000 or not text.strip():
        raise ValueError("一镜自动配音单次需要 1–5000 字")
    name = config.read()["baidu_voice_name"].strip()
    if not name:
        raise ValueError("请在设置中填写一镜公共库的完整音色名称")
    # Share the existing account lock with the legacy GUI route. No repeated
    # submission, no alternate account, and no GUI fallback after an API error.
    while not baidu_web._generation_lock.acquire(timeout=.2):
        check()
    try:
        check()
        with _session() as (client, cuid):
            voice, _ = _voice(client, name)
            params = _params(text, voice, cuid)
            check()
            # Authorization may itself consume the account's daily allowance;
            # never wait until the later audio stream to record this attempt.
            if before_submit:
                before_submit()
            authorization = _post(client, AUTH, params)
            if not isinstance(authorization, dict) or not isinstance(authorization.get("auth"), str):
                raise ValueError("一镜未返回配音授权，请检查登录、额度或完成网页验证")
            if not authorization["auth"]:
                raise ValueError("一镜未返回有效配音授权")
            check()
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".baidu-", dir=target.parent) as directory:
                audio = Path(directory) / "speech.mp3"
                output = Path(directory) / "speech.wav"
                _stream(params, authorization["auth"], audio, check, timeout=min(600, 120 + len(text) * .1))
                check()
                media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(audio),
                           "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(output)],
                          timeout=60, check=check)
                info = media.probe(output)
                if float(info.get("format", {}).get("duration", 0)) <= 0:
                    raise ValueError("一镜返回的音轨没有有效时长")
                check()
                output.replace(target)
    finally:
        baidu_web._generation_lock.release()
