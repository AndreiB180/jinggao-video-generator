#!/usr/bin/env python3
"""Isolated, loopback-only visual observation benchmark; never imports jinggao.

Public model files must already exist and match the publisher hashes. This
program neither downloads files nor reads API credentials or production state.
Ground truth in cases.json is copied to the run record, never sent to the model.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
from pathlib import Path
import resource
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {
    "Qwen3.5-2B-Q8_0.gguf": "1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1",
    "mmproj-F16.gguf": "7035e9cb8d7c6a9681d07eef9a364783e86ea4cd73faab2eabb4f43a101830c7",
}
PROMPT = """你是中立的逐帧观察员。依次观察给出的 5 张图片，严格只写各帧实际可见的内容。
不提供故事背景、不判断是否适合某个稿件，不识别人脸身份，不猜人物意图，不用其他帧的内容补写当前帧。
输出一个 JSON 对象，格式为 {"frames":[...]}，恰好 5 项，每项包含：
frame_index（1 到 5）、scene（场景和主要物体，简短）、visible_action（直接看见的动作；看不清就说明）、
overlay_text（画面后加的字幕/水印，能读才抄录，否则空数组）、
native_text（场内标牌/物体或手机屏幕文字，能读才抄录，否则空数组）、
obstructions（黑块、白屏或主体遮挡的实际位置，没有则空数组）、
sensitive_text_regions（可能的私人电话、地址、单号区域，写 kind 和 bbox；bbox 是相对整幅图的 [left,top,right,bottom]，0 到 1；没有则空数组）、
uncertainty（不确定或看不清处，没有则空字符串）。
不抄录私人姓名、私人电话号码、详细地址或运单号的字符，只报告其区域和类别。公开标牌、字幕可以抄录。
若出现整帧空白，明确写空白，不为它推断场景。每帧独立，避免漏掉首尾或将字幕当成实际动作。
保持简短，只输出 JSON，不要解释分析过程。"""


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect forbidden", headers, fp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data/models/qwen3.5-2b-vision")
    parser.add_argument("--port", type=int, default=28935)
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--protocol", default="local_neutral_visual_probe_v1")
    parser.add_argument("--image-min-tokens", type=int, default=0)
    parser.add_argument("--json-mode", action="store_true")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or not 1 <= args.max_cases <= 5:
        parser.error("expected a private nonprivileged port and 1..5 cases")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "run.json").exists():
        parser.error("output run.json exists; use a fresh artifact directory")
    if not 0 <= args.image_min_tokens <= 2048:
        parser.error("image-min-tokens must be between 0 and 2048")
    prompt = args.prompt_file.read_text() if args.prompt_file else PROMPT
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    model_manifest = []
    for name, expected in MODEL_FILES.items():
        path = args.model_dir.resolve() / name
        actual = digest(path)
        if actual != expected:
            raise ValueError(f"publisher hash mismatch: {name}")
        model_manifest.append({"path": str(path), "sha256": actual, "bytes": path.stat().st_size})
    cases = json.loads(args.cases.read_text())[:args.max_cases]
    for case in cases:
        if len(case["frames"]) != 5:
            raise ValueError("each case requires exactly five actual frames")
        for frame in case["frames"]:
            path = (ROOT / frame["path"]).resolve()
            if frame.get("sha256") and frame["sha256"] != digest(path):
                raise ValueError(f"input changed: {path}")
            frame["sha256"] = digest(path)
            frame["bytes"] = path.stat().st_size
            with Image.open(path) as im:
                frame["dimensions"] = list(im.size)
    write_json(args.out / "cases.json", cases)
    (args.out / "prompt.txt").write_text(prompt)
    (args.out / "evaluate_local_vision.py").write_bytes(Path(__file__).read_bytes())
    cmd = [
        "/opt/homebrew/bin/llama-server", "--offline",
        "--model", str(args.model_dir.resolve() / "Qwen3.5-2B-Q8_0.gguf"),
        "--mmproj", str(args.model_dir.resolve() / "mmproj-F16.gguf"),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--alias", "local-qwen3.5-2b-vision", "--ctx-size", "16384",
        "--parallel", "1", "--threads", "8", "--threads-batch", "8",
        "--gpu-layers", "99", "--image-max-tokens", "2048",
        "--batch-size", "512", "--ubatch-size", "256",
        "--cache-ram", "0", "--no-cache-prompt", "--no-webui",
        "--reasoning", "off", "--reasoning-budget", "0",
        "--chat-template-kwargs", '{"enable_thinking": false}',
    ]
    if args.image_min_tokens:
        cmd.extend(["--image-min-tokens", str(args.image_min_tokens)])
    run = {
        "started_utc": utc(), "model_files": model_manifest, "command": cmd,
        "script_sha256": digest(Path(__file__)),
        "protocol": args.protocol, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
                     "presence_penalty": 1.5, "repeat_penalty": 1.0, "seed": 20261009},
        "remote_inference_requests": 0, "api_cost_cny": 0,
        "computer_cost_measured": False, "formal_generation_target_evaluated": False,
        "network_scope": "inference uses 127.0.0.1 only; no redirects or proxy; server offline; child environment PATH only",
        "memory_method": "ps RSS sampled every 0.5 s; child ru_maxrss on macOS; Metal logs separate; not total unified-memory usage",
        "cases": [],
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    base = f"http://127.0.0.1:{args.port}"
    memory = []
    stopped = threading.Event()
    proc = None
    start = time.monotonic()
    log = (args.out / "llama-server.log").open("w")
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env={"PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"})
        run["server_pid"] = proc.pid

        def sample_memory() -> None:
            while not stopped.is_set():
                rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True)
                if rss.returncode == 0 and rss.stdout.strip().isdigit():
                    memory.append({"seconds": round(time.monotonic() - start, 3), "rss_kib": int(rss.stdout.strip())})
                stopped.wait(0.5)

        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
        while time.monotonic() - start < 90:
            if proc.poll() is not None:
                raise RuntimeError(f"local server exited before readiness: {proc.returncode}")
            try:
                with opener.open(base + "/health", timeout=1) as response:
                    if response.status == 200:
                        run["load_to_ready_seconds"] = round(time.monotonic() - start, 3)
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.25)
        else:
            raise TimeoutError("local server readiness exceeded 90 seconds")
        print(json.dumps({"server_ready_seconds": run["load_to_ready_seconds"], "pid": proc.pid}), flush=True)
        for case in cases:
            content = [{"type": "text", "text": prompt}]
            for index, frame in enumerate(case["frames"], 1):
                raw = (ROOT / frame["path"]).read_bytes()
                content.extend([
                    {"type": "text", "text": f"frame_index={index}"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()}},
                ])
            payload = {
                "model": "local-qwen3.5-2b-vision",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 2200, "stream": False, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False},
                **run["sampling"],
            }
            if args.json_mode:
                payload["response_format"] = {"type": "json_object"}
            request_bytes = json.dumps(payload, ensure_ascii=False).encode()
            # Avoid duplicating private screenshots in artifacts. Hash the exact body.
            meta = {k: v for k, v in payload.items() if k != "messages"}
            meta.update({"prompt": prompt, "input_frames": case["frames"],
                         "request_sha256": hashlib.sha256(request_bytes).hexdigest(), "request_bytes": len(request_bytes)})
            write_json(args.out / f'{case["id"]}-request.json', meta)
            result = {"id": case["id"], "started_utc": utc(), "input_sha256": [f["sha256"] for f in case["frames"]]}
            t = time.monotonic()
            memory_start = len(memory)
            try:
                request = urllib.request.Request(base + "/v1/chat/completions", data=request_bytes,
                                                 headers={"Content-Type": "application/json"}, method="POST")
                with opener.open(request, timeout=150) as response:
                    result["http_status"] = response.status
                    raw_response = response.read()
                (args.out / f'{case["id"]}-response.json').write_bytes(raw_response)
                result["response_sha256"] = hashlib.sha256(raw_response).hexdigest()
                body = json.loads(raw_response)
                result["usage"] = body.get("usage")
                result["server_timings"] = body.get("timings")
                choice = body["choices"][0]
                result["finish_reason"] = choice.get("finish_reason")
                message = choice["message"]
                text = message.get("content", "")
                result["reasoning_content_present"] = bool(message.get("reasoning_content"))
                result["content_chars"] = len(text)
                try:
                    parsed = json.loads(text)
                    indices = [f.get("frame_index") for f in parsed.get("frames", [])]
                    result["strict_json_valid"] = True
                    result["exact_five_frame_indices"] = indices == list(range(1, 6))
                except (ValueError, AttributeError, TypeError):
                    result["strict_json_valid"] = False
                    result["exact_five_frame_indices"] = False
            except Exception as exc:
                result["error"] = type(exc).__name__ + ": " + str(exc)
            result["wall_seconds"] = round(time.monotonic() - t, 3)
            result["peak_rss_kib"] = max((m["rss_kib"] for m in memory[memory_start:]), default=None)
            run["cases"].append(result)
            write_json(args.out / "run.json", run)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result.get("error"):
                # Do not leave a timed-out generation running or repeatedly troubleshoot.
                break
    except Exception as exc:
        run["error"] = type(exc).__name__ + ": " + str(exc)
        print(json.dumps({"error": run["error"]}, ensure_ascii=False), flush=True)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        stopped.set()
        if "sampler" in locals():
            sampler.join(timeout=2)
        log.close()
        run["server_exit_code"] = proc.returncode if proc else None
        run["completed_utc"] = utc()
        run["server_lifetime_seconds"] = round(time.monotonic() - start, 3)
        run["peak_sampled_rss_kib"] = max((m["rss_kib"] for m in memory), default=None)
        run["child_ru_maxrss_macos_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        write_json(args.out / "memory-samples.json", memory)
        write_json(args.out / "run.json", run)


if __name__ == "__main__":
    main()
