from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import signal
import socket
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


def main():
    parser = argparse.ArgumentParser(description="镜稿本地视频工作台")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    paths.prepare()
    config.migrate()
    ownership = instance.acquire()
    if ownership is None:
        for _ in range(30):
            try:
                old = json.loads((paths.DATA / "instance.json").read_text("utf-8"))
                url = "http://127.0.0.1:" + str(int(old["port"]))
                if httpx.get(url+"/api/health", timeout=1, trust_env=False).json().get("app") == "jinggao":
                    if not args.no_browser:
                        webbrowser.open(url)
                    print("镜稿已在 " + url + " 运行")
                    return
            except (OSError, ValueError, KeyError, httpx.HTTPError):
                pass
            time.sleep(.2)
        raise RuntimeError("已有工作台进程占用此数据目录，请检查原启动窗口")
    port = args.port
    # This folder's data is ours (the lock above), so a 镜稿 already answering on the port is another copy,
    # usually an older version still running: never open that one; start on the next free port instead.
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=1, trust_env=False)
        if r.status_code == 200 and r.json().get("app") == "jinggao":
            print(f"注意：端口 {port} 上还有另一个镜稿在运行（版本 {r.json().get('version', '未知')}），"
                  f"本版本 {__version__} 改用其他端口。旧版本的黑色窗口可以关掉。", flush=True)
    except (httpx.HTTPError, ValueError):
        pass
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
