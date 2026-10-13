"""End-to-end acceptance through the normal local API (no manual material choice).

Usage: .venv/bin/python scripts/express_acceptance.py examples/port.txt [--base http://127.0.0.1:8766]
Creates a project, runs `produce`, waits for the export, and writes a report
JSON plus a per-shot contact sheet of the exported MP4 next to the export.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jinggao import paths  # noqa: E402


def contact_sheet(video: Path, project: dict, out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    ffmpeg = paths.binary("ffmpeg")
    font = ImageFont.truetype(str(paths.font_path()), 18)
    tiles, elapsed = [], 0.0
    for index, shot in enumerate(project["shots"], 1):
        duration = float(shot["duration"])
        target = out.parent / f"sheet-{index:03}.jpg"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{elapsed + duration * 0.5:.2f}", "-i", str(video),
                        "-frames:v", "1", "-vf", "scale=384:-2", str(target)], check=True)
        image = Image.open(target).convert("RGB")
        canvas = Image.new("RGB", (384, image.height + 46), (20, 20, 20))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        lane = ((shot.get("verification") or {}).get("express") or {}).get("lane", "?")
        draw.text((4, image.height + 2), f"{index} [{lane}] {shot.get('visual', '')[:18]}", font=font, fill=(255, 220, 0))
        draw.text((4, image.height + 23), shot["narration"][:20], font=font, fill=(230, 230, 230))
        tiles.append(canvas)
        target.unlink()
        elapsed += duration
    cols = 5
    w, h = tiles[0].size
    sheet = Image.new("RGB", (cols * w, -(-len(tiles) // cols) * h), (0, 0, 0))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
    sheet.save(out, quality=85)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script")
    parser.add_argument("--base", default="http://127.0.0.1:8766")
    parser.add_argument("--title", default="")
    parser.add_argument("--aspect", default="16:9")
    parser.add_argument("--requirements", default="")
    parser.add_argument("--resolution", default="720p")
    args = parser.parse_args()
    text = Path(args.script).read_text("utf-8").strip()
    title = args.title or Path(args.script).stem
    client = httpx.Client(base_url=args.base, timeout=60)
    project = client.post("/api/projects", json={"title": title, "script": text, "aspect": args.aspect}).json()
    pid = project["id"]
    if args.requirements:
        client.patch(f"/api/projects/{pid}", json={"requirements": args.requirements}).raise_for_status()
    began = time.time()
    job = client.post(f"/api/projects/{pid}/jobs", json={"kind": "produce", "resolution": args.resolution}).json()
    last = ""
    while True:
        state = client.get(f"/api/jobs/{job['id']}").json()
        line = f"{time.time()-began:6.1f}s {state['status']} {state.get('progress')} {state.get('message', '')[:90]}"
        if line[8:] != last:
            print(line, flush=True)
            last = line[8:]
        if state["status"] in ("completed", "failed", "needs_review", "cancelled", "interrupted"):
            break
        time.sleep(2)
    wall = time.time() - began
    project = client.get(f"/api/projects/{pid}").json()
    from jinggao import costs as ledger
    costs = ledger.summary(pid)
    export = (project.get("exports") or [None])[0]
    report = {"project_id": pid, "title": title, "script_chars": len(text), "status": state["status"],
              "error": state.get("error"), "wall_seconds": round(wall, 1),
              "workflow": project.get("workflow_info"), "express": project.get("express_report"),
              "model_cost_cny": (costs.get("project") or {}).get("total_cny"),
              "cost_tasks": costs.get("tasks"), "shots": len(project["shots"]),
              "timeline_seconds": round(sum(float(s["duration"]) for s in project["shots"]), 2),
              "export": export}
    if export:
        video = paths.DATA / "projects" / pid / export["path"]
        report["video_path"] = str(video)
        info = json.loads(subprocess.run([paths.binary("ffprobe"), "-v", "error", "-show_entries",
                                          "format=duration:stream=width,height,codec_type", "-of", "json", str(video)],
                                         capture_output=True, text=True).stdout)
        report["video_probe"] = info
        sheet = video.parent / "contact-sheet.jpg"
        contact_sheet(video, project, sheet)
        report["contact_sheet"] = str(sheet)
    out = (paths.DATA / "projects" / pid / "acceptance.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({k: report[k] for k in ("status", "error", "wall_seconds", "model_cost_cny", "shots",
                                              "timeline_seconds")}, ensure_ascii=False))
    print("report:", out)


if __name__ == "__main__":
    main()
