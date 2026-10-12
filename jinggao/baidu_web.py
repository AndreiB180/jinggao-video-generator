"""Optional browser workflow using a separate local Edge/Chrome login profile.

Uses visible site controls and Chrome DevTools Protocol, never copies Chrome's
personal cookie database or calls undocumented Baidu APIs. Site changes can break
the flow; imported audio remains the fallback. Login/challenges are user actions.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import websocket

from . import config, paths, store

PAGE = "https://huibo.baidu.com/index.html#/clone-voice?platform=video&voice_tab=public"
_lock = threading.Lock()
_generation_lock = threading.Lock()


class PageControlTimeout(ValueError):
    """A page control was unavailable before submitting a generation request."""


def response_error(body: dict) -> str:
    """Read a failure already returned to the page; never call a private API."""
    code = body.get("errno", body.get("error_code", 0))
    message = body.get("errmsg", body.get("message", ""))
    if code not in (0, "0", None) and isinstance(message, str) and message.strip():
        return message[:300]
    return ""


def browser_binary() -> str:
    if sys.platform == "darwin":
        choices = [Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                   Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")]
    elif os.name == "nt":
        choices = []
        for base in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
            if os.environ.get(base):
                choices += [Path(os.environ[base]) / "Microsoft/Edge/Application/msedge.exe",
                            Path(os.environ[base]) / "Google/Chrome/Application/chrome.exe"]
    else:
        choices = [Path(p) for name in ("google-chrome", "chromium", "microsoft-edge") if (p := shutil.which(name))]
    for path in choices:
        if path.exists():
            return str(path)
    raise ValueError("没有找到 Edge 或 Chrome，请先安装一个浏览器，或使用一镜网页手动下载导入")


def endpoint() -> str | None:
    file = paths.DATA / "baidu-browser" / "DevToolsActivePort"
    try:
        port = int(file.read_text().splitlines()[0])
        if not 1024 <= port <= 65535:
            return None
        info = httpx.get(f"http://127.0.0.1:{port}/json/version", timeout=1, trust_env=False).json()
        url = info["webSocketDebuggerUrl"]
        parsed = httpx.URL(url)
        if parsed.scheme == "ws" and parsed.host in ("127.0.0.1", "localhost") and parsed.port == port:
            return url
    except (OSError, ValueError, IndexError, KeyError, httpx.HTTPError):
        pass
    return None


def open_browser() -> dict:
    """Open only the application's profile; never restart the user's browser."""
    with _lock:
        if endpoint():
            return {"opened": True, "message": "一镜专用浏览器已开启，请在该窗口完成登录"}
        profile = paths.DATA / "baidu-browser"
        profile.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            profile.chmod(0o700)
        args = [browser_binary(), "--user-data-dir="+str(profile), "--remote-debugging-port=0",
                "--remote-debugging-address=127.0.0.1", "--no-first-run", "--no-default-browser-check", PAGE]
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for _ in range(50):
            if endpoint():
                return {"opened": True, "message": "请在一镜专用浏览器中登录一次，完成后回工作台生成配音"}
            time.sleep(.2)
        raise ValueError("浏览器未就绪，请关闭镜稿专用浏览器窗口后重试；原有日常浏览器无需关闭")


class CDP:
    def __init__(self, url: str):
        self.ws = websocket.create_connection(url, timeout=15, suppress_origin=True,
                                              http_no_proxy=["127.0.0.1", "localhost"])
        self.seq = 0
        self.events = []
        self.session = None

    def call(self, method: str, params=None, *, page=False, timeout=15):
        self.seq += 1
        request = {"id": self.seq, "method": method, "params": params or {}}
        if page:
            request["sessionId"] = self.session
        self.ws.send(json.dumps(request))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise websocket.WebSocketTimeoutException("Browser command timed out")
            self.ws.settimeout(remaining)
            message = json.loads(self.ws.recv())
            if message.get("id") == self.seq:
                if "error" in message:
                    raise ValueError("一镜浏览器操作未完成，请检查登录状态或改用手动下载")
                return message.get("result", {})
            self.events.append(message)

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True}, page=True)
        if result.get("exceptionDetails"):
            raise ValueError("一镜网页结构已变化，请使用手动配音入口")
        return result.get("result", {}).get("value")

    def receive(self):
        self.ws.settimeout(.5)
        try:
            self.events.append(json.loads(self.ws.recv()))
        except websocket.WebSocketTimeoutException:
            pass

    def wait_element(self, expression: str, check, timeout=35):
        deadline = time.monotonic()+timeout
        previous = None
        while time.monotonic() < deadline:
            check()
            value = self.evaluate("(() => {const e="+expression+"; if(!e)return null;"
                                  "const r=e.getBoundingClientRect();if(!r.width||!r.height||e.disabled)return null;"
                                  "e.scrollIntoView({block:'center',behavior:'instant'});const b=e.getBoundingClientRect();"
                                  "const p={x:b.x+b.width/2,y:b.y+b.height/2};"
                                  "const top=document.elementFromPoint(p.x,p.y);"
                                  "if(!top||!e.contains(top)||getComputedStyle(e).visibility==='hidden')return null;"
                                  "return p;})()")
            if value and value == previous:
                return value
            previous = value
            time.sleep(.4)
        raise PageControlTimeout("未找到一镜控件。请在专用浏览器完成登录/验证，并确认可进入视频音色公共库后重试")

    def click(self, point):
        for kind in ("mouseMoved", "mousePressed", "mouseReleased"):
            args = {"type": kind, **point}
            if kind != "mouseMoved":
                args.update(button="left", clickCount=1)
            self.call("Input.dispatchMouseEvent", args, page=True)

    def type_text(self, text: str):
        self.call("Input.dispatchKeyEvent", {"type": "keyDown", "key": "a", "code": "KeyA",
                  "modifiers": 4 if sys.platform == "darwin" else 2, "windowsVirtualKeyCode": 65}, page=True)
        self.call("Input.dispatchKeyEvent", {"type": "keyUp", "key": "a", "code": "KeyA",
                  "modifiers": 4 if sys.platform == "darwin" else 2, "windowsVirtualKeyCode": 65}, page=True)
        self.call("Input.insertText", {"text": text}, page=True)


def generate(text: str, target: Path, check, *, before_submit=None):
    # DownloadBehavior belongs to the browser, so parallel tabs cannot safely
    # choose different download folders. Only retry controls before submission;
    # never repeat a request after the synthesis/download button was clicked.
    while not _generation_lock.acquire(timeout=.2):
        check()
    try:
        for attempt in range(2):
            check()
            try:
                if before_submit is None:
                    return _generate(text, target, check)
                return _generate(text, target, check, before_submit=before_submit)
            except PageControlTimeout:
                if attempt:
                    raise
    finally:
        _generation_lock.release()


def _generate(text: str, target: Path, check, *, before_submit=None):
    if not 1 <= len(text) <= 5000:
        raise ValueError("一镜网页配音单次需要 1–5000 字")
    url = endpoint()
    if not url:
        open_browser()
        raise ValueError("已打开一镜专用浏览器。请先登录一次，再回到工作台点击重试")
    name = config.read()["baidu_voice_name"].strip()
    if not name:
        raise ValueError("请在设置中填写一镜公共库的完整音色名称")
    try:
        client = CDP(url)
    except (websocket.WebSocketException, OSError):
        raise ValueError("一镜浏览器连接中断，请重新打开专用浏览器或手动下载导入") from None
    folder = paths.DATA / "baidu-downloads" / store.uid()
    folder.mkdir(parents=True)
    tab = None
    success = False
    guid = None
    try:
        client.call("Browser.setDownloadBehavior", {"behavior": "allowAndName", "downloadPath": str(folder),
                                                  "eventsEnabled": True})
        tab = client.call("Target.createTarget", {"url": PAGE})["targetId"]
        client.session = client.call("Target.attachToTarget", {"targetId": tab, "flatten": True})["sessionId"]
        client.call("Runtime.enable", page=True)
        client.call("Network.enable", page=True)
        search = client.wait_element("document.querySelector('input[placeholder=\"输入名称搜索公共音色\"]')", check)
        client.click(search)
        client.type_text(name)
        client.call("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter", "code": "Enter",
                                               "windowsVirtualKeyCode": 13}, page=True)
        client.call("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter", "code": "Enter",
                                               "windowsVirtualKeyCode": 13}, page=True)
        # Use the exact visible title, then reveal the same download icon used manually.
        title = "Array.from(document.querySelectorAll('[class*=voiceCardListItemTitleContentName]')).find(e=>e.textContent.trim()==="+json.dumps(name)+")"
        point = client.wait_element(title, check)
        client.call("Input.dispatchMouseEvent", {"type": "mouseMoved", **point}, page=True)
        download = client.wait_element("("+title+")?.closest('[class*=\"voiceCardListItem_\"]')?.querySelector('img[src*=\"download.\"]')", check)
        client.click(download)
        field = client.wait_element("document.querySelector('.voice-list-download textarea')", check)
        client.click(field)
        client.type_text(text)
        if client.evaluate("document.querySelector('.voice-list-download textarea')?.value") != text:
            raise ValueError("一镜网页没有收到完整口播，请重试或手动导入")
        button = "Array.from(document.querySelectorAll('.voice-list-download button')).find(e=>e.textContent.trim()==='下载'&&e.getBoundingClientRect().width)"
        point = client.wait_element(button, check)
        # A changed paid flow is a user decision; do not click a purchase confirmation.
        free = client.evaluate("document.querySelector('.voice-list-download')?.innerText.includes('限时免费')")
        if not free:
            raise ValueError("一镜当前页面未显示限时免费，请在网页核对收费方式并手动下载导入")
        frame = client.call("Page.getFrameTree", page=True)["frameTree"]["frame"]["id"]
        if before_submit:
            before_submit()
        client.click(point)
        deadline = time.monotonic()+180
        last_feedback = time.monotonic()
        responses = set()
        while time.monotonic() < deadline:
            check()
            client.receive()
            events, client.events = client.events, []
            for event in events:
                method, params = event.get("method"), event.get("params", {})
                if method == "Network.responseReceived":
                    response = params.get("response", {})
                    location = urlparse(response.get("url", ""))
                    if (location.scheme == "https" and location.hostname == "huibo.baidu.com"
                            and location.path.startswith("/ai_anchor/tts/") and params.get("requestId")):
                        responses.add(params["requestId"])
                if method == "Network.loadingFinished" and params.get("requestId") in responses:
                    responses.discard(params["requestId"])
                    try:
                        observed = client.call("Network.getResponseBody", {"requestId": params["requestId"]}, page=True)
                        body = json.loads(observed.get("body", "")) if not observed.get("base64Encoded") else {}
                        error = response_error(body) if isinstance(body, dict) else ""
                    except (ValueError, TypeError):
                        error = ""
                    if error:
                        raise ValueError("一镜网页提示：" + error)
                if method == "Browser.downloadWillBegin" and params.get("frameId") == frame and guid is None:
                    candidate = params.get("guid", "")
                    if candidate and all(c.isascii() and (c.isalnum() or c == "-") for c in candidate):
                        guid = candidate
                if method == "Browser.downloadProgress" and guid and params.get("guid") == guid:
                    if params.get("receivedBytes", 0) > 100*1024*1024:
                        raise ValueError("一镜下载文件超过 100 MB，请手动核对")
                    if params.get("state") == "canceled":
                        raise ValueError("一镜音频下载已取消，请重试或手动导入")
                    if params.get("state") == "completed":
                        downloaded = folder / guid
                        if not downloaded.is_file() or downloaded.stat().st_size > 100*1024*1024:
                            raise ValueError("一镜下载文件异常，请手动核对")
                        from .media import run
                        run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(downloaded),
                             "-vn", "-ar", "24000", "-ac", "1", str(target)], check=check)
                        success = True
                        return
            if time.monotonic() - last_feedback > 1:
                last_feedback = time.monotonic()
                message = client.evaluate("Array.from(document.querySelectorAll('[role=alert],.ant-message-notice-content')).filter(e=>e.getBoundingClientRect().width).map(e=>e.innerText).join(' ').slice(0,300)") or ""
                if any(word in message for word in ("失败", "验证码", "请登录", "不足", "频繁", "异常", "上限", "明天")):
                    raise ValueError("一镜网页提示："+message)
        state = client.evaluate("document.querySelector('.voice-list-download')?.innerText.slice(-350)") or "下载弹窗已关闭"
        raise ValueError("一镜下载超时。网页当前显示："+state)
    except (websocket.WebSocketException, OSError, KeyError):
        raise ValueError("一镜浏览器连接中断，请重新打开专用浏览器或手动下载导入") from None
    finally:
        cleanup = [("Browser.setDownloadBehavior", {"behavior": "default", "eventsEnabled": False})]
        if guid and not success:
            cleanup.insert(0, ("Browser.cancelDownload", {"guid": guid}))
        if success and tab:
            cleanup.append(("Target.closeTarget", {"targetId": tab}))
        for method, params in cleanup:
            try:
                client.call(method, params)
            except Exception:
                pass
        client.ws.close()
        shutil.rmtree(folder, ignore_errors=True)
