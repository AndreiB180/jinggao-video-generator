"""Optional browser workflow using a separate local Edge/Chrome login profile.

Uses visible site controls and Chrome DevTools Protocol, never copies the user's own
browser profile or cookie database and never calls undocumented Baidu APIs itself: the
voice list is what the 一镜 page loads and shows. Site changes can break the flow;
imported audio remains the fallback. Login/challenges are user actions.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
import websocket

from . import config, paths, store

PAGE = "https://huibo.baidu.com/index.html#/clone-voice?platform=video&voice_tab=public"
BROWSERS = {"edge": "Microsoft Edge", "chrome": "Google Chrome"}
LOGIN_COOKIES = ("BDUSS", "BDUSS_BFESS")
# The dedicated profile must not sign in to the Windows Microsoft account or sync the user's own Edge
# data (passwords, history) into it; it must keep working while its window is behind other windows.
FLAGS = ["--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check", "--disable-sync",
         "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
         "--disable-backgrounding-occluded-windows"]
EDGE_FLAGS = ["--disable-features=msImplicitSignin"]
START_SECONDS = 30
NOT_FOUND = ("没有找到 Microsoft Edge 或 Google Chrome。一镜配音要用其中一个浏览器登录百度账号："
             "请安装 Chrome 或 Edge 后重试，或先改用「免费配音」")
NOT_LOGGED_IN = ("一镜还没有登录：已打开一镜专用浏览器，请在那个窗口里用百度 App 扫码登录（登录一次即可），"
                 "然后回到镜稿重试")
LOGIN_EXPIRED = "一镜登录已失效：请在一镜专用浏览器里重新扫码登录，然后重试"
VOICES_FILE = "baidu-voices.json"
SAMPLE_HOSTS = ("baidu.com", "bcebos.com", "bdstatic.com", "baidubce.com", "bdimg.com")
SAMPLE_KEYS = ("show_url", "audition_url", "demo_url", "sample_url", "preview_url", "audio_url", "listen_url")
_lock = threading.Lock()
_generation_lock = threading.Lock()


class PageControlTimeout(ValueError):
    """A page control was unavailable before submitting a generation request."""


class VoiceMissing(ValueError):
    """The chosen voice is not in the 一镜 public library (any more)."""


def response_error(body: dict) -> str:
    """Read a failure already returned to the page; never call a private API."""
    code = body.get("errno", body.get("error_code", 0))
    message = body.get("errmsg", body.get("message", ""))
    if code not in (0, "0", None) and isinstance(message, str) and message.strip():
        return message[:300]
    return ""


def _app_paths(exe: str) -> list[Path]:
    """Where Windows records an installed browser (App Paths): also finds Chrome/Edge installed to another
    drive or folder."""
    if os.name != "nt":
        return []
    import winreg
    found = []
    for root, view in ((winreg.HKEY_CURRENT_USER, 0), (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
                       (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY)):
        try:
            with winreg.OpenKey(root, "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\" + exe, 0,
                                winreg.KEY_READ | view) as key:
                value = winreg.QueryValueEx(key, "")[0]
        except OSError:
            continue
        value = os.path.expandvars(str(value)).strip().strip('"')
        if value:
            found.append(Path(value))
    return found


def _installed() -> dict[str, list[Path]]:
    if sys.platform == "darwin":
        apps = [Path("/Applications"), Path.home() / "Applications"]
        return {"chrome": [a / "Google Chrome.app/Contents/MacOS/Google Chrome" for a in apps],
                "edge": [a / "Microsoft Edge.app/Contents/MacOS/Microsoft Edge" for a in apps]}
    if os.name == "nt":
        bases = [os.environ.get(k) for k in ("PROGRAMFILES(X86)", "PROGRAMFILES", "PROGRAMW6432", "LOCALAPPDATA")]
        bases += ["C:\\Program Files (x86)", "C:\\Program Files"]
        return {"edge": _app_paths("msedge.exe") + [Path(b) / "Microsoft/Edge/Application/msedge.exe" for b in bases if b],
                "chrome": _app_paths("chrome.exe") + [Path(b) / "Google/Chrome/Application/chrome.exe" for b in bases if b]}
    return {"chrome": [Path(p) for n in ("google-chrome", "chromium") if (p := shutil.which(n))],
            "edge": [Path(p) for n in ("microsoft-edge",) if (p := shutil.which(n))]}


def browser_candidates() -> list[tuple[str, Path]]:
    """Installed browsers in the order they are tried: the one chosen in Settings, or Edge (always present on
    Windows 10/11) then Chrome; Chrome first on a Mac."""
    installed = _installed()
    choice = config.read().get("baidu_browser", "auto")
    if choice in BROWSERS:
        order = [choice]
    else:
        order = ["edge", "chrome"] if os.name == "nt" else ["chrome", "edge"]
    found = []
    for kind in order:
        path = next((p for p in installed.get(kind, []) if p.is_file()), None)
        if path:
            found.append((kind, path))
    if not found:
        if choice in BROWSERS:
            raise ValueError(f"没有找到 {BROWSERS[choice]}。请安装它，或在设置 → 配音里把「一镜专用浏览器」改回自动")
        raise ValueError(NOT_FOUND)
    return found


def browser_binary() -> str:
    return str(browser_candidates()[0][1])


def profile(kind: str) -> Path:
    """The application's own login profile for one browser (Edge and Chrome profiles are not interchangeable).
    The browser earlier versions used first (Edge on Windows, Chrome elsewhere) keeps data/baidu-browser, so an
    existing login carries over."""
    legacy = "edge" if os.name == "nt" else "chrome"
    return paths.DATA / ("baidu-browser" if kind == legacy else f"baidu-browser-{kind}")


def _endpoint_for(folder: Path, timeout: float = 1) -> str | None:
    try:
        port = int((folder / "DevToolsActivePort").read_text("ascii").splitlines()[0])
        if not 1024 <= port <= 65535:
            return None
        info = httpx.get(f"http://127.0.0.1:{port}/json/version", timeout=timeout, trust_env=False).json()
        url = info["webSocketDebuggerUrl"]
        parsed = httpx.URL(url)
        if parsed.scheme == "ws" and parsed.host in ("127.0.0.1", "localhost") and parsed.port == port:
            return url
    except (OSError, ValueError, IndexError, KeyError, TypeError, httpx.HTTPError):
        pass
    return None


def running() -> tuple[str, str] | None:
    """(browser, DevTools address) of the dedicated browser that is open now."""
    for kind in BROWSERS:
        url = _endpoint_for(profile(kind))
        if url:
            return kind, url
    return None


def endpoint() -> str | None:
    found = running()
    return found[1] if found else None


def _held(folder: Path) -> bool:
    """A browser process has this profile open (Chromium keeps a lock file while it runs)."""
    return (folder / "lockfile").exists() if os.name == "nt" else (folder / "SingletonLock").is_symlink()


def _uses_profile(command: str, folder: Path) -> bool:
    """The command line opens exactly this profile (quoted or not; not baidu-browser-chrome for baidu-browser)."""
    pattern = r'--user-data-dir="?' + re.escape(str(folder)) + r'"?(?=\s|$)'
    return bool(re.search(pattern, command, re.IGNORECASE if os.name == "nt" else 0))


def _stop_profile(folder: Path) -> None:
    """End browser processes of our own profile only (one that hangs, or runs without the DevTools port)."""
    try:
        if os.name == "nt":
            # PowerShell lists the command lines (with Chinese paths intact); the match is done here.
            script = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-CimInstance Win32_Process -Filter "
                      "\"Name='msedge.exe' OR Name='chrome.exe'\" | ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
            listed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                                     base64.b64encode(script.encode("utf-16-le")).decode()],
                                    capture_output=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
            lines = listed.stdout.decode("utf-8", "replace").splitlines()
        else:
            lines = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True,
                                   timeout=10).stdout.splitlines()
        for line in lines:
            pid, _, command = line.strip().replace("\t", " ", 1).partition(" ")
            if pid.isdigit() and _uses_profile(command, folder):
                try:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True, timeout=10,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        os.kill(int(pid), 9)
                except (OSError, subprocess.SubprocessError):
                    pass
    except (OSError, subprocess.SubprocessError):
        pass
    for _ in range(25):
        if not _held(folder):
            return
        time.sleep(.2)


def _quiet_profile(folder: Path) -> None:
    """Edge keeps running invisibly after its window is closed unless background mode is off; then the next
    click would find a browser without a window. Set it before the browser starts (Local State)."""
    state = folder / "Local State"
    try:
        data = json.loads(state.read_text("utf-8")) if state.exists() else {}
        if not isinstance(data, dict) or (data.get("background_mode") or {}).get("enabled") is False:
            return
        data["background_mode"] = {**(data.get("background_mode") or {}), "enabled": False}
        state.write_text(json.dumps(data), "utf-8")
    except (OSError, ValueError, TypeError, AttributeError):
        pass


def _start(kind: str, binary: Path) -> subprocess.Popen:
    folder = profile(kind)
    folder.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        folder.chmod(0o700)
    _quiet_profile(folder)
    args = [str(binary), "--user-data-dir=" + str(folder), *FLAGS, *(EDGE_FLAGS if kind == "edge" else []), PAGE]
    options: dict = {"start_new_session": True}
    if os.name == "nt":
        # A GUI program, shown normally; never CREATE_NO_WINDOW/DETACHED. Normal priority even when a video
        # worker (below normal) opens it, so the page stays responsive.
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = 1       # SW_SHOWNORMAL
        options = {"startupinfo": info,
                   "creationflags": subprocess.NORMAL_PRIORITY_CLASS | subprocess.CREATE_NEW_PROCESS_GROUP}
    return subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, close_fds=True, **options)


def _wait_ready(folder: Path, process: subprocess.Popen, seconds: float = START_SECONDS) -> str | None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        url = _endpoint_for(folder)
        if url:
            return url
        if process.poll() not in (None, 0):      # crashed or blocked (0: handed over to a running copy)
            return None
        time.sleep(.25)
    return None


def _windows_front(pid: int, wait: float = 5.0) -> bool:
    """Put the browser's window above the others on the user's desktop. Windows does not let a background
    program take the keyboard focus from the program the user is using, so a new window could open behind
    the 镜稿 page; on top and flashing in the taskbar, the user sees it."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback, wintypes.LPARAM]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    for name in ("IsWindowVisible", "IsIconic", "GetWindowTextLengthW", "SetForegroundWindow"):
        getattr(user32, name).argtypes = [wintypes.HWND]
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_uint]
    user32.GetForegroundWindow.restype = wintypes.HWND

    class Flash(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("hwnd", wintypes.HWND), ("dwFlags", wintypes.DWORD),
                    ("uCount", wintypes.UINT), ("dwTimeout", wintypes.DWORD)]

    deadline = time.monotonic() + wait
    while True:
        found = []

        def collect(hwnd, _):
            owner = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid and user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0:
                kind = ctypes.create_unicode_buffer(64)
                user32.GetClassNameW(hwnd, kind, 64)
                if kind.value.startswith("Chrome_WidgetWin"):
                    found.append(hwnd)
            return True
        user32.EnumWindows(callback(collect), 0)
        if found or time.monotonic() >= deadline:
            break
        time.sleep(.2)
    if not found:
        return False
    window = found[0]                    # EnumWindows starts at the top of the z-order
    if user32.IsIconic(window):
        user32.ShowWindow(window, 9)     # SW_RESTORE
    flags = 0x0001 | 0x0002 | 0x0040     # SWP_NOSIZE | SWP_NOMOVE | SWP_SHOWWINDOW
    user32.SetWindowPos(window, wintypes.HWND(-1), 0, 0, 0, 0, flags)    # HWND_TOPMOST, then back to normal:
    user32.SetWindowPos(window, wintypes.HWND(-2), 0, 0, 0, 0, flags)    # it stays above the other windows
    user32.SetForegroundWindow(window)
    if user32.GetForegroundWindow() != window:
        flash = Flash(ctypes.sizeof(Flash), window, 0x3 | 0xC, 0, 0)       # caption + taskbar until focused
        user32.FlashWindowEx(ctypes.byref(flash))
    return True


def _present(url: str, pid: int | None = None) -> None:
    """Show the 一镜 page in a window of the dedicated browser, restored and in front."""
    client = CDP(url)
    try:
        pages = [t for t in client.call("Target.getTargets")["targetInfos"] if t["type"] == "page"]
        page = next((t for t in pages if urlparse(t["url"]).hostname == "huibo.baidu.com"), None)
        # No page at all: the browser runs without a window; open one.
        target = page["targetId"] if page else client.call(
            "Target.createTarget", {"url": PAGE, "newWindow": not pages})["targetId"]
        client.call("Target.activateTarget", {"targetId": target})
        try:
            window = client.call("Browser.getWindowForTarget", {"targetId": target})
            if window.get("bounds", {}).get("windowState") == "minimized":
                client.call("Browser.setWindowBounds", {"windowId": window["windowId"],
                                                        "bounds": {"windowState": "normal"}})
        except (ValueError, KeyError):
            pass
        try:
            processes = client.call("SystemInfo.getProcessInfo")["processInfo"]
            pid = next(p["id"] for p in processes if p.get("type") == "browser")
        except (ValueError, KeyError, StopIteration):
            pass
    finally:
        client.ws.close()
    if os.name == "nt" and pid:
        try:
            _windows_front(pid)
        except (OSError, AttributeError, ValueError):
            pass


def _try_present(url: str, pid: int | None = None) -> bool:
    try:
        _present(url, pid)
        return True
    except (websocket.WebSocketException, OSError, ValueError, KeyError):
        return False        # the browser answers but cannot be driven: treated as not running


def open_browser() -> dict:
    """Open only the application's own profile; never touch the user's everyday browser."""
    with _lock:
        found = running()
        if found and _try_present(found[1]):
            kind = found[0]
            return {"opened": True, "browser": BROWSERS[kind],
                    "message": f"一镜专用浏览器（{BROWSERS[kind]}）已打开，窗口已放到最前面。没有登录的话，"
                               "点页面左下角的「登录/注册」，用百度 App 扫码登录，登录一次即可"}
        tried = []
        for kind, binary in browser_candidates():
            folder = profile(kind)
            if _held(folder) and not _endpoint_for(folder, timeout=3):
                _stop_profile(folder)        # a copy that hangs or has no DevTools port would swallow the start
            process = _start(kind, binary)
            url = _wait_ready(folder, process)
            if not url and process.poll() == 0 and _held(folder):
                _stop_profile(folder)        # it handed over to a copy we cannot reach: restart it once
                process = _start(kind, binary)
                url = _wait_ready(folder, process)
            if url:
                _try_present(url, process.pid)
                return {"opened": True, "browser": BROWSERS[kind],
                        "message": f"已打开一镜专用浏览器（{BROWSERS[kind]} 窗口）。请在窗口里用百度 App 扫码登录，"
                                   "登录一次即可；没看到窗口的话，请看任务栏上闪烁的浏览器图标"}
            tried.append(BROWSERS[kind])
            if process.poll() is None:
                _stop_profile(folder)
        raise ValueError(f"一镜专用浏览器没有启动成功（试过 {'、'.join(tried)}）。请关闭安全软件对浏览器的拦截"
                         "（如 360、火绒的浏览器防护）后重试，或在设置 → 配音里换一个专用浏览器")


def _logged_in(client) -> bool:
    """Whether the dedicated profile holds a Baidu login (cookie presence only; values are never read out)."""
    cookies = client.call("Storage.getCookies").get("cookies", [])
    return any(c.get("name") in LOGIN_COOKIES and c.get("value") and str(c.get("domain", "")).endswith("baidu.com")
               for c in cookies)


def status() -> dict:
    """Browser, login and voice-list state for the settings page. Opens nothing and sends nothing to Baidu."""
    cached = cached_voices()
    info = {"voice": config.read()["baidu_voice_name"].strip(), "voices": len(cached["voices"]),
            "voices_updated": cached["updated"], "browser": None, "open": False, "logged_in": None}
    try:
        info["browser"] = BROWSERS[browser_candidates()[0][0]]
    except ValueError as exc:
        return info | {"message": str(exc)}
    found = running()
    if not found:
        return info | {"message": "一镜专用浏览器没有打开。点「打开一镜专用浏览器」：登录过的账号会保持登录，"
                                  "没登录的话在窗口里用百度 App 扫码登录"}
    kind, url = found
    info.update(browser=BROWSERS[kind], open=True)
    try:
        client = CDP(url)
        try:
            info["logged_in"] = _logged_in(client)
        finally:
            client.ws.close()
    except (websocket.WebSocketException, OSError, ValueError, KeyError):
        return info | {"message": "一镜专用浏览器没有响应，请关闭它后重新点「打开一镜专用浏览器」"}
    if info["logged_in"]:
        return info | {"message": "已登录百度账号，可以用一镜配音"}
    return info | {"message": "一镜专用浏览器已打开，还没有登录：请在那个窗口里用百度 App 扫码登录"}


def cached_voices() -> dict:
    try:
        data = json.loads((paths.DATA / VOICES_FILE).read_text("utf-8"))
        if isinstance(data, dict) and isinstance(data.get("voices"), list):
            return {"voices": [v for v in data["voices"] if isinstance(v, dict) and v.get("name")],
                    "updated": str(data.get("updated", ""))}
    except (OSError, ValueError):
        pass
    return {"voices": [], "updated": ""}


def _sample_url(value) -> str:
    if not isinstance(value, str):
        return ""
    location = urlparse(value.strip())
    host = (location.hostname or "").lower()
    if location.scheme in ("http", "https") and any(host == h or host.endswith("." + h) for h in SAMPLE_HOSTS):
        return value.strip()
    return ""


def _voice_items(body) -> list[dict]:
    """The public voice list in a response the 一镜 page itself received (data.list of voice cards)."""
    data = body.get("data") if isinstance(body, dict) else None
    items = data.get("list") if isinstance(data, dict) else data
    if not isinstance(items, list) or not items or not all(isinstance(i, dict) for i in items):
        return []
    if not any(isinstance(i.get("display_name"), str) for i in items):
        return []
    voices, seen = [], set()
    for item in items:
        name = item.get("display_name")
        if not isinstance(name, str) or not name.strip() or name.strip() in seen:
            continue
        if item.get("status") not in (None, 1, "1"):     # only voices the page offers for use
            continue
        seen.add(name.strip())
        sample = next((u for key in SAMPLE_KEYS if (u := _sample_url(item.get(key)))), "")
        voices.append({"name": name.strip()[:60], "description": str(item.get("description") or "")[:120],
                       "category": str(item.get("category_list") or "")[:60], "sample": sample})
    return voices


CARDS = ("(() => {/*jinggao-cards*/const vis=e=>{const r=e.getBoundingClientRect();return r.width>0&&r.height>0};"
         "const cards=Array.from(document.querySelectorAll('[class*=voiceCardListItem_]')).filter(vis);"
         "const q=(c,s)=>(c.querySelector(s)?.textContent||'').trim();"
         "return {hash: location.hash, loading: !!Array.from(document.querySelectorAll("
         "'[class*=voiceCardContentWrapLoading],.ant-spin-spinning')).find(vis),"
         "empty: !!Array.from(document.querySelectorAll('[class*=mpty]')).find(e=>vis(e)&&/没有|暂无/.test(e.innerText)),"
         "cards: cards.map(c=>({name:q(c,'[class*=voiceCardListItemTitleContentName]'),"
         "description:q(c,'[class*=voiceCardListItemDescriptionContent]')})).filter(c=>c.name)};})()")
# One-time "novice guide" popovers of the 一镜 page can cover the controls: close them (their × only).
DISMISS = ("(() => {let n=0;for(const e of document.querySelectorAll('.huibo-novice-guide-pop .ant-popover-title-close,"
           ".ant-tour .ant-tour-close')){const r=e.getBoundingClientRect();if(r.width&&r.height){e.click();n++}}return n})()")
NEXT_PAGE = ("(() => {const n=document.querySelector('.ant-pagination-next:not(.ant-pagination-disabled)');"
             "if(n){n.click();return true}return false})()")


def _login_gone(state) -> bool:
    """The 一镜 app sends a visitor without a valid login from the voice page to its home page."""
    return isinstance(state, dict) and str(state.get("hash", "")).startswith("#/home")


def refresh_voices(check=lambda: None) -> dict:
    """Read the public voice list through the logged-in dedicated browser and keep it for the settings page."""
    url = endpoint()
    if not url:
        raise ValueError("请先点「打开一镜专用浏览器」并登录百度账号，再刷新音色列表")
    try:
        client = CDP(url)
    except (websocket.WebSocketException, OSError):
        raise ValueError("一镜专用浏览器连接中断，请关闭它后重新打开") from None
    tab = None
    try:
        if not _logged_in(client):
            raise ValueError("一镜还没有登录：请在一镜专用浏览器里用百度 App 扫码登录后，再刷新音色列表")
        tab = client.call("Target.createTarget", {"url": "about:blank", "background": True})["targetId"]
        client.session = client.call("Target.attachToTarget", {"targetId": tab, "flatten": True})["sessionId"]
        client.call("Network.enable", page=True)
        client.call("Runtime.enable", page=True)
        client.call("Page.navigate", {"url": PAGE}, page=True)
        listed, pending, cards = [], set(), []
        began = time.monotonic()
        listed_at = None
        while time.monotonic() - began < 45:
            check()
            client.receive()
            events, client.events = client.events, []
            for event in events:
                method, params = event.get("method"), event.get("params", {})
                if method == "Network.responseReceived":
                    location = urlparse(params.get("response", {}).get("url", ""))
                    if (location.scheme == "https" and location.hostname == "huibo.baidu.com"
                            and location.path.startswith("/ai_anchor/") and "clone" not in location.path):
                        pending.add(params.get("requestId"))
                elif method == "Network.loadingFinished" and params.get("requestId") in pending:
                    pending.discard(params["requestId"])
                    try:
                        body = client.call("Network.getResponseBody", {"requestId": params["requestId"]}, page=True)
                        items = _voice_items(json.loads(body.get("body", ""))) if not body.get("base64Encoded") else []
                    except ValueError:
                        items = []
                    if len(items) > len(listed):
                        listed, listed_at = items, time.monotonic()
            state = client.evaluate(CARDS)
            if _login_gone(state) and time.monotonic() - began > 3:
                raise ValueError(LOGIN_EXPIRED)
            if isinstance(state, dict) and state.get("cards"):
                cards = state["cards"]
            if listed and (cards or time.monotonic() - listed_at > 5):
                break
            if not listed and cards and time.monotonic() - began > 15:
                cards = _all_pages(client, cards, check)       # the page's list response was not seen
                break
        voices = listed or [{"name": c["name"][:60], "description": c.get("description", "")[:120],
                             "category": "", "sample": ""} for c in cards]
        names = {v["name"] for v in voices}
        voices += [{"name": c["name"][:60], "description": c.get("description", "")[:120], "category": "",
                    "sample": ""} for c in cards if c["name"] not in names]
        if not voices:
            raise ValueError("一镜音色库没有加载出来：请在专用浏览器里打开一镜「音色 → 公共音色」确认能看到音色后再刷新")
        saved = {"updated": datetime.now().isoformat(timespec="seconds"), "voices": voices}
        target = paths.DATA / VOICES_FILE
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, ensure_ascii=False, indent=1), "utf-8")
        temporary.replace(target)
        return saved
    except (websocket.WebSocketException, OSError, KeyError):
        raise ValueError("一镜专用浏览器连接中断，请关闭它后重新打开，再刷新音色列表") from None
    finally:
        try:
            if tab:
                client.call("Target.closeTarget", {"targetId": tab})
        except Exception:
            pass
        client.ws.close()


def _all_pages(client, cards: list, check) -> list:
    """Voice names from every page of the visible list (only when the page's own list was not observed)."""
    seen = {c["name"]: c for c in cards}
    for _ in range(40):
        check()
        if not client.evaluate(NEXT_PAGE):
            break
        time.sleep(1.2)
        state = client.evaluate(CARDS)
        new = [c for c in (state or {}).get("cards", []) if c["name"] not in seen] if isinstance(state, dict) else []
        if not new:
            break
        seen.update((c["name"], c) for c in new)
    return list(seen.values())


def sample(name: str) -> Path:
    """A cached copy of the voice's public sample audio (as listed by the 一镜 page)."""
    voice = next((v for v in cached_voices()["voices"] if v.get("name") == name), None)
    url = _sample_url((voice or {}).get("sample"))
    if not url:
        raise KeyError("这个一镜音色没有试听样音")
    folder = paths.DATA / "baidu-voice-samples"
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    cached = sorted(folder.glob(key + ".*"))
    if cached:
        return cached[0]
    from . import network
    try:
        with network.client(timeout=20, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError:
        raise ValueError("一镜试听样音下载失败，请检查网络后重试") from None
    if not _sample_url(str(response.url)) or len(response.content) > 10 * 1024 * 1024 or not response.content:
        raise ValueError("一镜试听样音地址无效")
    kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
    suffix = {"audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav",
              "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".aac", "audio/ogg": ".ogg"}.get(kind)
    suffix = suffix or (Path(urlparse(url).path).suffix.lower() if Path(urlparse(url).path).suffix.lower()
                        in (".mp3", ".wav", ".m4a", ".aac", ".ogg") else ".mp3")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (key + suffix)
    temporary = paths.temporary(target)
    temporary.write_bytes(response.content)
    paths.publish(temporary, target)
    return target


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
    # choose different download folders: one narration at a time across every
    # video worker process (throttle slot) and thread (lock). Only retry controls
    # before submission; never repeat a request after the download button was clicked.
    from . import throttle
    with throttle.slot("baidu_web", check):
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


VOICE_MISSING = "一镜公共音色库里找不到「{}」（可能已下架或改名）。请到 设置 → 配音 点「刷新音色列表」，重新选一个音色"


def _find_voice(client, name: str, title: str, check, timeout: float = 40):
    """After the search: the exact voice card, or a clear error once the results have settled without it."""
    began = time.monotonic()
    seen, since = None, began
    while time.monotonic() - began < timeout:
        check()
        client.evaluate(DISMISS)
        state = client.evaluate(CARDS)
        if isinstance(state, dict):
            names = [c.get("name") for c in state.get("cards", [])]
            if name in names:
                return client.wait_element(title, check)
            if _login_gone(state):
                raise ValueError(LOGIN_EXPIRED)
            key = (tuple(names), bool(state.get("empty")))
            if key != seen or state.get("loading"):
                seen, since = key, time.monotonic()
            elif (names or key[1]) and time.monotonic() - since >= 4 and time.monotonic() - began >= 6:
                raise VoiceMissing(VOICE_MISSING.format(name))
        time.sleep(.5)
    raise PageControlTimeout("一镜音色搜索结果没有加载出来，请检查网络，或在专用浏览器确认能打开一镜音色库后重试")


def _generate(text: str, target: Path, check, *, before_submit=None):
    if not 1 <= len(text) <= 5000:
        raise ValueError("一镜网页配音单次需要 1–5000 字")
    name = config.read()["baidu_voice_name"].strip()
    if not name:
        raise ValueError("请在 设置 → 配音 里选择一个一镜音色")
    url = endpoint()
    if not url:
        open_browser()      # the profile keeps the login: a logged-in browser simply carries on
        url = endpoint()
        if not url:
            raise ValueError("一镜专用浏览器没有打开，请在 设置 → 配音 点「打开一镜专用浏览器」后重试")
    try:
        client = CDP(url)
    except (websocket.WebSocketException, OSError):
        raise ValueError("一镜浏览器连接中断，请重新打开专用浏览器或手动下载导入") from None
    try:
        logged_in = _logged_in(client)
    except (websocket.WebSocketException, OSError, ValueError, KeyError):
        client.ws.close()
        raise ValueError("一镜浏览器连接中断，请重新打开专用浏览器或手动下载导入") from None
    if not logged_in:
        client.ws.close()
        try:
            _present(url)
        except (websocket.WebSocketException, OSError, ValueError, KeyError):
            pass
        raise ValueError(NOT_LOGGED_IN)
    folder = paths.DATA / "baidu-downloads" / store.uid()
    folder.mkdir(parents=True)
    tab = None
    success = False
    discard = False           # a tab that only shows a search or a control that did not load
    guid = None
    try:
        client.call("Browser.setDownloadBehavior", {"behavior": "allowAndName", "downloadPath": str(folder),
                                                  "eventsEnabled": True})
        # A background tab of the dedicated browser: it does not take the user's focus. Focus emulation lets the
        # page's search box and text field take typed text as in a front tab.
        tab = client.call("Target.createTarget", {"url": PAGE, "background": True})["targetId"]
        client.session = client.call("Target.attachToTarget", {"targetId": tab, "flatten": True})["sessionId"]
        client.call("Emulation.setFocusEmulationEnabled", {"enabled": True}, page=True)
        client.call("Runtime.enable", page=True)
        client.call("Network.enable", page=True)

        def page_check():
            check()
            if _login_gone(client.evaluate("({hash: location.hash})")):
                raise ValueError(LOGIN_EXPIRED)
            client.evaluate(DISMISS)
        search = client.wait_element("document.querySelector('input[placeholder=\"输入名称搜索公共音色\"]')", page_check)
        client.click(search)
        client.type_text(name)
        client.call("Input.dispatchKeyEvent", {"type": "keyDown", "key": "Enter", "code": "Enter",
                                               "windowsVirtualKeyCode": 13}, page=True)
        client.call("Input.dispatchKeyEvent", {"type": "keyUp", "key": "Enter", "code": "Enter",
                                               "windowsVirtualKeyCode": 13}, page=True)
        # Use the exact visible title, then reveal the same download icon used manually.
        title = "Array.from(document.querySelectorAll('[class*=voiceCardListItemTitleContentName]')).find(e=>e.textContent.trim()==="+json.dumps(name)+")"
        point = _find_voice(client, name, title, check)
        client.call("Input.dispatchMouseEvent", {"type": "mouseMoved", **point}, page=True)
        try:
            download = client.wait_element("("+title+")?.closest('[class*=\"voiceCardListItem_\"]')?.querySelector('img[src*=\"download.\"]')", check, timeout=15)
        except PageControlTimeout:
            raise PageControlTimeout("一镜音色卡片上没有「下载音色」按钮：这个百度账号可能还没有一镜的音频下载权限。"
                                     "请在专用浏览器打开一镜音色库，确认能手动下载配音后重试") from None
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
    except (PageControlTimeout, VoiceMissing):
        discard = True
        raise
    except (websocket.WebSocketException, OSError, KeyError):
        raise ValueError("一镜浏览器连接中断，请重新打开专用浏览器或手动下载导入") from None
    finally:
        cleanup = [("Browser.setDownloadBehavior", {"behavior": "default", "eventsEnabled": False})]
        if guid and not success:
            cleanup.insert(0, ("Browser.cancelDownload", {"guid": guid}))
        if (success or discard) and tab:
            cleanup.append(("Target.closeTarget", {"targetId": tab}))
        for method, params in cleanup:
            try:
                client.call(method, params)
            except Exception:
                pass
        client.ws.close()
        shutil.rmtree(folder, ignore_errors=True)
