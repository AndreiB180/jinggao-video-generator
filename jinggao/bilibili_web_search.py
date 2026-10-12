"""Read the ordinary public search page in a disposable anonymous browser.

The direct HTTP page currently serves a challenge while a normal anonymous
browser shows public cards. Do not replay the old API, extract cookies, solve a
challenge, or reuse any logged-in browser. Playback is verified independently.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import httpx
import websocket

from . import baidu_web, domestic, video_sources as video

BROWSER_TIMEOUT = 18
TRANSIENT_BACKOFF = 60
SEARCH_SLOTS = threading.BoundedSemaphore(2)
blocked_until = 0.0  # This ordinary webpage's cooldown is independent of the retired API.
unavailable_until = 0.0  # Loading/connection failures do not imply a verification wall.
PAGE = "https://search.bilibili.com/all"

# Selectors and the public SSR page number were observed on the normal page.
# Read only card metadata and visible UI; never enumerate state/storage/auth.
SNAPSHOT = r"""(() => {
  const cards = [...document.querySelectorAll('.bili-video-card')].slice(0, 60).map(card => {
    const title = card.querySelector('.bili-video-card__info--tit');
    return {
      title: title?.getAttribute('title') || title?.innerText || '',
      url: title?.closest('a')?.href || '',
      video_links: [...card.querySelectorAll('a[href]')].map(a => a.href).filter(u => u.includes('/video/BV')),
      thumbnail: card.querySelector('.bili-video-card__cover img')?.getAttribute('src') || '',
      publisher: card.querySelector('.bili-video-card__info--author')?.innerText || '',
      publisher_url: card.querySelector('.bili-video-card__info--owner')?.href || '',
      duration: card.querySelector('.bili-video-card__stats__duration')?.innerText || '',
      date: card.querySelector('.bili-video-card__info--date')?.innerText || '',
      excluded: !!card.querySelector('.bili-video-card__info--ad-creative,.bili-video-card__info--cheese'),
      badges: [...card.querySelectorAll('[class*="badge"],[class*="label"],[class*="tag"]')].map(e => e.innerText).join(' ')
    };
  });
  return {url: location.href, title: document.title, ready: document.readyState,
    page: window.__pinia?.searchResponse?.searchAllResponse?.page ?? null,
    text: document.body?.innerText.slice(0, 6000) || '', cards};
})()"""


class PublicSearchBlocked(ValueError):
    """A visible verification/login wall stops this provider without retry."""


class PublicSearchUnavailable(ValueError):
    """The ordinary page could not finish loading or connect within its budget."""


def _challenge(snapshot):
    title, text = str(snapshot.get("title", "")), str(snapshot.get("text", ""))
    url = urlparse(str(snapshot.get("url", "")))
    if (re.search(r"(?:验证码|安全验证|访问受限|人机验证)(?:[_｜|]|$)", title)
            or url.hostname in {"passport.bilibili.com", "account.bilibili.com"}
            or re.search(r"/(?:captcha|challenge|verify)(?:/|$)", url.path)
            or not snapshot.get("cards") and re.search(r"请完成.{0,12}验证|验证后继续|登录后才能搜索|异常访问", text)):
        raise PublicSearchBlocked("B站普通匿名网页要求验证或登录，本轮已停止该渠道")


def parse_snapshot(snapshot, query, page=1):
    if not isinstance(snapshot, dict):
        raise ValueError("B站公开搜索页面结构已变化")
    _challenge(snapshot)
    location = urlparse(str(snapshot.get("url", "")))
    args = parse_qs(location.query)
    if (location.scheme != "https" or location.hostname != "search.bilibili.com" or location.path != "/all"
            or args.get("keyword") != [query] or type(snapshot.get("page")) is not int
            or snapshot["page"] != page):
        raise ValueError("B站公开搜索页与请求的关键词或页码不一致")
    cards = snapshot.get("cards")
    if not isinstance(cards, list):
        raise ValueError("B站公开搜索缺少视频卡片")
    if not cards:
        if re.search(r"没有找到|暂无相关|无相关结果", str(snapshot.get("text", ""))):
            return []
        raise ValueError("B站公开搜索未加载完整视频卡片")
    result, seen = [], set()
    for card in cards[:60]:
        if not isinstance(card, dict) or card.get("excluded") or re.search(
                r"充电专属|付费|会员专享|VIP|大会员", str(card.get("badges", "")), re.I):
            continue
        url = urlparse(str(card.get("url", "")))
        match = re.fullmatch(r"/video/(BV[0-9A-Za-z]{10})/?", url.path)
        if url.scheme != "https" or url.hostname != "www.bilibili.com" or not match or url.username or url.password:
            continue
        bvid = match[1]
        links = card.get("video_links", [])
        # The title and cover must refer to the same individual video.
        if (not isinstance(links, list) or any(
                re.search(r"/video/(BV[0-9A-Za-z]{10})", str(link)).group(1) != bvid
                for link in links if re.search(r"/video/(BV[0-9A-Za-z]{10})", str(link)))):
            continue
        title = video._clean(card.get("title"), 500)
        if not title or bvid in seen:
            continue
        seen.add(bvid)
        source = "https://www.bilibili.com/video/" + bvid
        raw_thumb = str(card.get("thumbnail", ""))
        thumb = video._http_url(urljoin(PAGE, raw_thumb)) if raw_thumb else ""
        if not domestic.allowed(thumb):
            thumb = ""
        publisher = video._clean(card.get("publisher"), 200)
        publisher_url = str(card.get("publisher_url", ""))
        if not re.fullmatch(r"https://space\.bilibili\.com/\d+/?", publisher_url):
            publisher_url = ""
        published = video._clean(card.get("date"), 80).lstrip(" ·")
        result.append(video._candidate("bilibili", title, source, thumb, video_id=bvid,
            description="", publisher=publisher, publisher_url=publisher_url,
            publisher_verified=False, published_at=published,
            duration=video._duration(card.get("duration")), source_verified=True,
            license="B站公开投稿；转载及商用许可需核实", search_method="anonymous_browser_dom",
            source_evidence=video._evidence(source, title, "", publisher, publisher_url, published, "publisher_search")))
    return result[:40]


def _stop(process):
    if process.poll() is None:
        try:
            process.terminate()  # Only the process created for this disposable profile.
        except OSError:
            pass


def _cleanup_profile(profile, delays):
    """Retry only this TemporaryDirectory; Edge can release DB handles late."""
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            profile.cleanup()
            if not Path(profile.name).exists():
                return True
        except OSError:
            pass
    return False


@contextmanager
def _anonymous_profile():
    profile = tempfile.TemporaryDirectory(prefix="jinggao-bili-search-", ignore_cleanup_errors=True)
    try:
        yield profile.name
    finally:
        if not _cleanup_profile(profile, (0, .1, .3)):
            # Keep a short Windows file-lock race from replacing valid search
            # results (or the original access error). No browser is killed here.
            cleanup = threading.Thread(target=_cleanup_profile, args=(profile, (1, 2, 4, 8)),
                                       name="jinggao-bili-profile-cleanup", daemon=True)
            try:
                cleanup.start()
            except RuntimeError:
                pass  # Shutdown must not turn anonymous cache cleanup into a search failure.


def _finish_process(process):
    # Browser.close initiates a graceful shutdown. Give its children time to
    # close their files before terminating only this launch's parent process.
    try:
        process.wait(timeout=1)
        return
    except subprocess.TimeoutExpired:
        _stop(process)
    except OSError:
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass
    except OSError:
        pass


def read_public_page(query, page=1, *, timeout=BROWSER_TIMEOUT):
    """One navigation, hard process deadline, and no challenge retries."""
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 200 or type(page) is not int or not 1 <= page <= 50:
        raise ValueError("B站检索需要1–200字关键词和1–50页码")
    url = PAGE + "?" + urlencode({"keyword": query, "page": page})
    started, client, process, timer = time.monotonic(), None, None, None
    try:
        binary = baidu_web.browser_binary()
    except ValueError:
        raise ValueError("B站公开网页检索需要本机 Edge 或 Chrome") from None
    with _anonymous_profile() as folder:
        try:
            process = subprocess.Popen([binary, "--headless=new", "--user-data-dir=" + folder,
                "--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1",
                "--no-first-run", "--no-default-browser-check", "about:blank"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            timer = threading.Timer(timeout, _stop, (process,))
            timer.daemon = True
            timer.start()
            portfile = Path(folder) / "DevToolsActivePort"
            while not portfile.exists():
                if process.poll() is not None or time.monotonic() - started >= timeout:
                    raise PublicSearchUnavailable("B站匿名检索浏览器未就绪或已超时")
                time.sleep(.1)
            port = int(portfile.read_text().splitlines()[0])
            if not 1024 <= port <= 65535:
                raise ValueError("匿名检索浏览器端口无效")
            info = httpx.get(f"http://127.0.0.1:{port}/json/version", timeout=2, trust_env=False).json()
            endpoint = urlparse(info["webSocketDebuggerUrl"])
            if endpoint.scheme != "ws" or endpoint.hostname != "127.0.0.1" or endpoint.port != port:
                raise ValueError("匿名检索浏览器地址无效")
            client = baidu_web.CDP(endpoint.geturl())
            targets = client.call("Target.getTargets")["targetInfos"]
            tab = next(t["targetId"] for t in targets if t["type"] == "page")
            client.session = client.call("Target.attachToTarget", {"targetId": tab, "flatten": True})["sessionId"]
            client.call("Page.enable", page=True)
            client.call("Page.navigate", {"url": url}, page=True)
            while time.monotonic() - started < timeout and process.poll() is None:
                snapshot = client.evaluate(SNAPSHOT)
                if isinstance(snapshot, dict):
                    _challenge(snapshot)
                    if snapshot.get("cards") and snapshot.get("page") is not None:
                        return parse_snapshot(snapshot, query, page)
                    if snapshot.get("ready") == "complete" and re.search(
                            r"没有找到|暂无相关|无相关结果", str(snapshot.get("text", ""))):
                        return parse_snapshot(snapshot, query, page)
                time.sleep(.2)
            raise PublicSearchUnavailable("B站普通匿名搜索在时限内未加载完整，未重试")
        except PublicSearchBlocked:
            raise
        except (httpx.HTTPError, websocket.WebSocketException, OSError, KeyError, IndexError, StopIteration):
            raise PublicSearchUnavailable("B站匿名公开搜索连接失败或超时，未重试") from None
        finally:
            if timer:
                timer.cancel()
            if client:
                try:
                    client.call("Browser.close", timeout=1)
                except (ValueError, OSError, websocket.WebSocketException):
                    pass
                try:
                    client.ws.close()
                except (OSError, websocket.WebSocketException):
                    pass
            if process:
                _finish_process(process)
