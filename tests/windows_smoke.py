"""Run with the distribution's embedded Python, from a writable Windows folder.

Uses only local demo assets and system SAPI speech, with a separate test data directory.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    root = Path(sys.executable).resolve().parents[2]
    output = Path(os.environ["JINGGAO_SMOKE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["JINGGAO_DATA"] = str(output / "中文 项目数据")
    port = 18765
    base = f"http://127.0.0.1:{port}/api"
    report = {"os": platform.platform(), "python": sys.version, "machine": platform.machine(),
              "root": str(root), "archive_sha256": os.environ.get("JINGGAO_SMOKE_ARCHIVE_SHA256"), "checks": []}
    log = (output / "launch.log").open("wb")
    proc = subprocess.Popen([sys.executable, "-X", "utf8", str(root / "launch.py"), "--port", str(port),
                             "--no-browser"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)

    def request(path, body=None, method=None):
        payload = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        r = Request(base+path, data=payload, method=method or ("POST" if payload else "GET"),
                    headers={"Content-Type": "application/json"})
        with urlopen(r, timeout=40) as response:
            return json.load(response)

    def job(pid, kind, **options):
        start = time.monotonic()
        j = request(f"/projects/{pid}/jobs", {"kind": kind, **options})
        while time.monotonic()-start < 300:
            current = request("/jobs/"+j["id"])
            if current["status"] == "completed":
                report["checks"].append({"kind": kind, "elapsed": round(time.monotonic()-start, 2), "status": "passed"})
                return
            if current["status"] in ("failed", "cancelled", "interrupted"):
                raise RuntimeError(current["error"] or current["message"])
            time.sleep(.5)
        raise RuntimeError("Job timed out: "+kind)

    try:
        for _ in range(100):
            try:
                if request("/health")["ok"]:
                    break
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError("Launcher exited; inspect launch.log") from None
                time.sleep(.3)
        else:
            raise RuntimeError("Launcher did not become ready")
        report["app_version"] = request("/health")["version"]
        assert report["app_version"] == "0.2.4", report["app_version"]
        report["checks"].append({"kind": "http_startup", "status": "passed"})
        duplicate = subprocess.run([sys.executable, "-X", "utf8", str(root / "launch.py"), "--port", str(port),
                                    "--no-browser"], cwd=root, env=env, capture_output=True, timeout=25)
        assert duplicate.returncode == 0, duplicate.stderr.decode(errors="replace")
        report["checks"].append({"kind": "single_instance", "status": "passed"})
        report["hardware"] = request("/system")
        report["ffmpeg_version"] = subprocess.check_output([str(root / "runtime/ffmpeg/ffmpeg.exe"), "-version"]).decode().splitlines()[0]
        request("/settings", {"encoder": "auto", "tts_provider": "system"}, "PATCH")
        p = request("/demo", method="POST")
        request(f"/projects/{p['id']}/shots/{p['shots'][0]['id']}",
                {"overlay_label": "Windows 验收资料图", "approved": True}, "PATCH")
        job(p["id"], "voice")
        p = request("/projects/"+p["id"])
        for shot in p["shots"]:
            request(f"/projects/{p['id']}/shots/{shot['id']}", {"approved": True}, "PATCH")
        job(p["id"], "render", resolution="1080p")
        p = request("/projects/"+p["id"])
        report["export"] = p["exports"][0]
        report["voice_durations"] = [s["audio"]["duration"] for s in p["shots"]]
        export_path = Path(env["JINGGAO_DATA"]) / "projects" / p["id"] / report["export"]["path"]
        import shutil
        shutil.copy2(export_path, output / "windows-demo-1080p.mp4")
        ffprobe = root / "runtime" / "ffmpeg" / "ffprobe.exe"
        report["probe"] = json.loads(subprocess.check_output([str(ffprobe), "-v", "error", "-show_streams",
                                                            "-show_format", "-of", "json", str(export_path)]))
        assert {s["codec_type"] for s in report["probe"]["streams"]} == {"video", "audio"}
        if os.environ.get("JINGGAO_TEST_BAIDU_AUDIO"):
            # This is already downloaded audio; no Baidu request or login occurs here.
            import runpy
            importer = runpy.run_path(str(Path(__file__).with_name("audio_import_smoke.py")))
            report["baidu_audio"] = importer["run"](base, Path(os.environ["JINGGAO_TEST_BAIDU_AUDIO"]),
                                                    output, root / "assets/fonts/NotoSansSC-Regular.otf")
            report["checks"].append({"kind": "baidu_audio_import_render", "status": "passed"})
        # Simulate closing only the launcher; the worker must not survive it.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/F"], capture_output=True, check=True)
        proc.wait(timeout=20)
        for _ in range(12):
            released = subprocess.run([sys.executable, "-c",
                "from jinggao import instance; import sys; h=instance.acquire('worker.lock'); sys.exit(0 if h else 1)"],
                cwd=root, env=env, capture_output=True, timeout=10)
            if released.returncode == 0:
                break
            time.sleep(.5)
        else:
            raise RuntimeError("Worker survived its launcher")
        report["checks"].append({"kind": "worker_parent_exit", "status": "passed"})
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        # Stop only this test's launcher tree, never other user processes.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
        proc.wait(timeout=20)
        log.close()
        (output / "windows-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        print(json.dumps({"status": report.get("status"), "checks": report["checks"],
                          "error": report.get("error")}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
