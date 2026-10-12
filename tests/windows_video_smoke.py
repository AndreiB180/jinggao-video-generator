"""Exercise public-video preview and bounded download with the packaged runtime."""
from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path


def main():
    output = Path(os.environ["JINGGAO_SMOKE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    os.environ["JINGGAO_DATA"] = str(output / "视频检索 中文数据")
    from jinggao import curation, media, paths, planner, sources, store, video_sources, worker

    report = {"os": platform.platform(), "machine": platform.machine(),
              "archive_sha256": os.environ.get("JINGGAO_SMOKE_ARCHIVE_SHA256"), "checks": []}
    try:
        store.init()
        project = store.new_project("Windows 公开动态视频验收", media_policy="video_only", min_video_height=1080)
        shot = planner.new_shot("官方比赛视频的公开片段。")
        shot.update(duration=3, overlay_label="公开视频下载验收 · WTT", fit="contain")
        candidate = sources.candidate("youtube", "WTT 本场公开视频", "https://www.youtube.com/watch?v=NOfcBrVMNGw",
                                      "", "https://www.youtube.com/watch?v=NOfcBrVMNGw", kind="video")
        candidate = video_sources.inspect_video_candidate(candidate)
        assert candidate["download_status"] == "available", candidate.get("download_reason")
        assert any(s.get("height", 0) >= 1080 for s in candidate["media_streams"])
        report["checks"].append({"kind": "public_youtube_metadata", "status": "passed",
                                  "parser": candidate.get("media_extractor"), "height": candidate["height"]})
        shot["candidates"] = [candidate]
        store.mutate(project["id"], lambda p: p.update(shots=[shot]), internal=True)
        candidate, _ = worker.preview_candidate(project["id"], shot, candidate, lambda: None, clip_start=510)
        preview = candidate["preview"]
        assert preview["motion_detected"] and len(preview["frames"]) == 5
        report["checks"].append({"kind": "bounded_preview", "status": "passed",
                                  "width": preview["width"], "height": preview["height"],
                                  "duration": preview["duration"], "bytes": preview["downloaded_bytes"]})
        asset = worker.materialize(project["id"], candidate, lambda: None, clip_start=510, duration=3.3)
        assert min(asset["width"], asset["height"]) >= 1080 and asset["duration"] >= 3
        store.mutate(project["id"], lambda p: p["shots"][0].update(selected_asset=asset["id"], approved=True), internal=True)
        p = store.get_project(project["id"])
        invalid = dict(p) | {"shots": [dict(p["shots"][0]) | {"duration": 8}]}
        try:
            curation.validate_export(invalid)
        except ValueError:
            report["checks"].append({"kind": "frozen_tail_rejected", "status": "passed"})
        else:
            raise AssertionError("Short video unexpectedly passed the duration gate")
        result = media.render(p, store.uid(), {"resolution": "1080p", "allow_silent": True}, lambda *_: None, lambda: None)
        target = paths.file_path(project["id"], result["path"])
        media.run([paths.binary("ffmpeg"), "-v", "error", "-xerror", "-i", str(target), "-f", "null", "-"], timeout=60)
        shutil.copy2(target, output / "windows-real-video-1080p.mp4")
        frame = output / "windows-real-video-frame.jpg"
        media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-ss", "1", "-i", str(target), "-frames:v", "1", str(frame)])
        report["checks"].append({"kind": "real_video_render", "status": "passed",
                                  "source_dimensions": [asset["width"], asset["height"]],
                                  "source_clip_start": asset["source_clip_start"],
                                  "downloaded_bytes": asset["downloaded_bytes"], "export": result})
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "windows-video-smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
        print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
