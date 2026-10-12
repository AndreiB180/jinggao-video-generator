"""Observe at most four saved selections; never mutate or approve the project.

Run only when the user has authorized the (at most two) configured vision calls.
An existing report is not overwritten or retried. No credentials/settings are
printed or saved; raw model content contains only the pixel observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jinggao import paths, visual_quality  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-snapshot", required=True, type=Path)
    parser.add_argument("--shots", required=True, help="1-based indices, at most four")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result_path = args.out / "results.json"
    if result_path.exists():
        raise SystemExit("已有诊断报告，未覆盖或重新请求")
    indices = [int(v) for v in args.shots.split(",")]
    if not 1 <= len(indices) <= 4 or len(set(indices)) != len(indices):
        raise SystemExit("需要1–4个不重复镜头编号")
    raw = args.project_snapshot.read_bytes()
    project = json.loads(raw)
    project = project.get("project", project)
    records, selections = [], []
    for index in indices:
        if not 1 <= index <= len(project["shots"]):
            raise SystemExit("镜头编号超出快照范围")
        shot = project["shots"][index-1]
        asset = next(a for a in project["assets"] if a["id"] == shot.get("selected_asset"))
        frozen = args.project_snapshot.parent / "evidence-frames" / f"shot-{index:02}"
        frames = sorted(frozen.glob("*.jpg")) if frozen.is_dir() else [
            paths.file_path(project["id"], frame["path"]) for frame in shot["match_preview"]["frames"]]
        if len(frames) != 5:
            raise SystemExit("诊断需要该镜已保存的完整5帧")
        originals = [paths.file_path(project["id"], frame["path"]) for frame in shot["match_preview"]["frames"]]
        if (len(originals) != len(frames) or any(hashlib.sha256(a.read_bytes()).digest() != hashlib.sha256(b.read_bytes()).digest()
                                               for a, b in zip(frames, originals, strict=True))):
            raise SystemExit("存档帧与快照实际预览不同，未发起请求")
        records.append({"shot": {"id": shot["id"]}, "frame_paths": frames})
        selections.append({"shot_index": index, "shot_id": shot["id"], "asset_id": asset["id"],
            "source_url": asset.get("source_url"), "source_start": (asset.get("source_clip_start") or 0)+(shot.get("clip_start") or 0),
            "duration": shot["duration"], "frame_paths": [str(path.resolve()) for path in frames],
            "frame_sha256": [hashlib.sha256(path.read_bytes()).hexdigest() for path in frames]})
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"project_id": project["id"], "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
              "module_sha256": hashlib.sha256(Path(visual_quality.__file__).read_bytes()).hexdigest(),
              "selections": selections, "project_mutated": False, "content_approval_performed": False}
    # Write before calling so interruption cannot silently look like an untried run.
    result_path.write_text(json.dumps(report | {"status": "started"}, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        result, info = visual_quality.observe(records, lambda: None, diagnostics_dir=args.out / "responses")
        report.update(status="completed", results=result, info=info)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      model_requests=getattr(exc, "model_requests", 0))
        raise
    finally:
        result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "info": info,
        "shots": [{"shot": s["shot_index"], "representation": result[s["shot_id"]]["representation"],
                   "quality_status": result[s["shot_id"]]["quality_status"]} for s in selections]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
