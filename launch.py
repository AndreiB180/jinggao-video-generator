from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import signal
import socket
import subprocess
import threading
import time
import webbrowser

import httpx
import uvicorn

from jinggao import __version__, config, instance, paths, source_index
from jinggao.api import create_app
from jinggao.worker import loop


def open_when_ready(url: str):
    for _ in range(100):
        try:
            if httpx.get(url+"/api/health", timeout=1, trust_env=False).status_code == 200:
                webbrowser.open(url)
                return
        except httpx.HTTPError:
            pass
        time.sleep(.2)


DEFAULT_PORT = 8770     # 0.3.2+; earlier versions used 8765, whose browser cache could keep showing the old page
LEGACY_PORT = 8765


def netstat_listeners(text: str, port: int) -> set[int]:
    """PIDs listening on port in Windows `netstat -ano` output. The state word may be localised, so a
    listening socket is recognised by its foreign address 0.0.0.0:0."""
    pids = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[1].endswith(f":{port}") \
                and parts[2] in ("0.0.0.0:0", "[::]:0") and parts[-1].isdigit():
            pids.add(int(parts[-1]))
    return pids


def _listening_pids(port: int) -> set[int]:
    """Processes listening on 127.0.0.1:port."""
    if os.name == "nt":
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, errors="replace").stdout
        return netstat_listeners(out, port)
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True).stdout
    return {int(x) for x in out.split() if x.isdigit()}


def _running_copy(port: int) -> dict | None:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1, trust_env=False)
        info = r.json() if r.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return None
    return info if info.get("app") == "jinggao" else None


def _close_copy(port: int, info: dict) -> None:
    """End another running 镜稿 (an old version or another folder) so this one opens instead."""
    for pid in _listening_pids(port) - {os.getpid()}:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    for _ in range(75):     # up to 15 s for it to let go of the port
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                break
        time.sleep(.2)
    print(f"已关闭正在运行的镜稿 {info.get('version', '旧版本')}（端口 {port}），改为启动 {__version__}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="镜稿本地视频工作台")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    paths.prepare()
    config.migrate()
    ownership = instance.acquire()
    if ownership is None:
        # This folder is already running: the same version is simply opened again; another version
        # (e.g. a new one unpacked over the old folder) replaces it.
        for _ in range(30):
            try:
                old = json.loads((paths.DATA / "instance.json").read_text("utf-8"))
                old_port = int(old["port"])
                info = _running_copy(old_port)
                if info and info.get("version") == __version__:
                    url = f"http://127.0.0.1:{old_port}"
                    if not args.no_browser:
                        webbrowser.open(url)
                    print("镜稿已在 " + url + " 运行")
                    return
                if info:
                    _close_copy(old_port, info)
                    ownership = instance.acquire()
                    if ownership is not None:
                        break
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(.2)
            ownership = instance.acquire()
            if ownership is not None:
                break
        if ownership is None:
            raise RuntimeError("已有工作台进程占用此数据目录，请检查原启动窗口")
    port = args.port
    # This folder's data is ours (the lock above), so a 镜稿 answering on our port, or on the port earlier
    # versions used, is another copy (usually the old version): close it so only the new one runs.
    for candidate in dict.fromkeys((args.port, LEGACY_PORT)):
        info = _running_copy(candidate)
        if info:
            _close_copy(candidate, info)
    for offset in range(20):
        with socket.socket() as sock:
            try:
                if os.name != "nt":
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", args.port+offset))
                port = args.port+offset
                break
            except OSError:
                continue
    else:
        raise RuntimeError("本地端口均被占用，请指定 --port")
    def footage_seed():
        # The footage index shipped with the app (assets/footage-index) goes in once, in the background.
        try:
            from jinggao import footage_index
            added = footage_index.install_seed()
            if added:
                print(f"已载入素材库 {added:,} 个视频", flush=True)
        except Exception as exc:
            print(f"素材库种子未能载入：{exc}", flush=True)
    threading.Thread(target=footage_seed, name="footage-seed", daemon=True).start()
    seeded = source_index.install_seed()
    source_index.prepare_entities()
    if seeded:
        print(f"已载入 {seeded:,} 条视频来源索引", flush=True)
    app = create_app()
    stop = multiprocessing.Event()
    worker = multiprocessing.Process(target=loop, args=(stop,), daemon=True)
    worker.start()
    background = multiprocessing.Process(target=loop, args=(stop, "background"), daemon=True)
    background.start()
    url = f"http://127.0.0.1:{port}"
    (paths.DATA / "instance.json").write_text(json.dumps({"port": port}), "utf-8")
    if not args.no_browser:
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()
    print(f"镜稿 {__version__} 已启动：{url}\n关闭浏览器不会停止任务。退出本窗口将停止后台服务。", flush=True)
    def graceful_term(signum, frame):
        raise KeyboardInterrupt
    previous_term = signal.signal(signal.SIGTERM, graceful_term)
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False,
                    timeout_graceful_shutdown=5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for process in (worker, background):
            process.join(3)
            if process.is_alive():
                process.terminate()
                process.join()
        ownership.close()
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
