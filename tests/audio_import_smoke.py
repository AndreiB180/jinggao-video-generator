"""Render an actual downloaded Baidu audio file through the running local API."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont


def run(base: str, audio: Path, output: Path, font: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=base, timeout=60, trust_env=False) as client:
        def request(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        def job(pid, kind, **options):
            item = request("POST", f"/projects/{pid}/jobs", json={"kind": kind, **options})
            for _ in range(240):
                current = request("GET", "/jobs/"+item["id"])
                if current["status"] == "completed":
                    return
                if current["status"] in ("failed", "cancelled", "interrupted"):
                    raise RuntimeError(current["error"] or current["message"])
                time.sleep(.5)
            raise RuntimeError("Audio import smoke timed out")

        p = request("POST", "/projects", json={"title": "一镜网页登录 · 音画合成实测",
                    "script": "这是一次配音测试。画面与声音将在本地合成。", "media_policy": "mixed"})
        pid = p["id"]
        job(pid, "plan", mode="rules")
        image = Image.new("RGB", (1920, 1080), "#141820")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((120, 100, 1800, 890), radius=36, fill="#1e2733", outline="#445161", width=2)
        for i, height in enumerate((45, 85, 120, 60, 150, 95, 170, 100, 70, 135, 90, 40)):
            x = 230 + i*50
            draw.rounded_rectangle((x, 570-height, x+20, 570+height), radius=10, fill="#f1ab70")
        for y, text, size, color in [(185, "JINGGAO  /  AUDIO WORKFLOW TEST", 32, "#f1ab70"),
                                   (275, "一镜账号配音 · 本地成片", 64, "#edf3fa"),
                                   (780, "网页登录下载 MP3 → 导入工作台 → 字幕与 MP4", 34, "#acbccb")]:
            draw.text((220, y), text, font=ImageFont.truetype(str(font), size), fill=color)
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        p = request("POST", f"/projects/{pid}/upload", files={"file": ("配音流程示意.png", buffer.getvalue(), "image/png")})
        aid = p["assets"][0]["id"]
        for shot in p["shots"]:
            request("POST", f"/projects/{pid}/shots/{shot['id']}/asset/{aid}")
            request("PATCH", f"/projects/{pid}/shots/{shot['id']}", json={"approved": True})
        p = request("POST", f"/projects/{pid}/upload", data={"role": "narration"},
                    files={"file": (audio.name, audio.read_bytes(), "audio/mpeg")})
        assert 4 <= p["full_audio"]["duration"] < 5
        for shot in p["shots"]:
            request("PATCH", f"/projects/{pid}/shots/{shot['id']}", json={"approved": True})
        job(pid, "render", resolution="1080p")
        p = request("GET", f"/projects/{pid}")
        exported = p["exports"][0]
        assert (exported["width"], exported["height"]) == (1920, 1080)
        for key, suffix in (("path", ".mp4"), ("bundle", ".zip")):
            response = client.get(f"/projects/{pid}/files/{exported[key]}")
            response.raise_for_status()
            (output / ("baidu-login-test-1080p"+suffix)).write_bytes(response.content)
        report = {"status": "passed", "project_id": pid, "audio_duration": p["full_audio"]["duration"],
                  "export": exported, "alignment": p["full_audio"]["alignment"]}
        (output / "baidu-audio-import.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.base, args.audio, args.output, args.font), ensure_ascii=False))
