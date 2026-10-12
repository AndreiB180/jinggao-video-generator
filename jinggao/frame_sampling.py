"""Bounded local scene coverage for video review, never a content decision."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import time

from . import library, media, paths

VERSION = "scene_coverage_v1"
FPS = 25
LIMIT = 5
ANALYSIS_WIDTH = 192
SCENE_THRESHOLD = .18
DEADLINE_SECONDS = 45


class CoverageError(ValueError):
    def __init__(self, message, sampling):
        super().__init__(message)
        self.sampling = sampling


def policy_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def window(shot, asset):
    start, requested, available, base = (float(shot.get("clip_start", 0)), float(shot["duration"]),
        float(asset["duration"]), float(asset.get("source_clip_start", 0)))
    if (not all(math.isfinite(n) for n in (start, requested, available, base))
            or min(start, base) < 0 or not 0 < requested <= 120 or available-start+.04 < requested):
        raise ValueError("实际视频窗口无效或短于镜头，不能定格补足")
    duration = min(requested, available-start)
    if duration <= 0:
        raise ValueError("实际视频窗口为空")
    return start, duration, base, max(1, math.ceil(duration*FPS-1e-7))


def indices_for(starts, count):
    if (not isinstance(starts, list) or not starts or starts[0] != 0
            or any(type(n) is not int for n in starts) or starts != sorted(set(starts))
            or type(count) is not int or count <= 0 or starts[-1] >= count):
        raise ValueError("本地场景索引无效")
    if len(starts) > LIMIT:
        raise CoverageError(f"窗口检测到 {len(starts)} 个场景，最多 {LIMIT} 张审核帧无法完整覆盖；请重选更连续的片段",
                            {"version": VERSION, "scene_starts": starts, "frame_count": count})
    ends = [*starts[1:], count]
    chosen = {(lo+hi-1)//2 for lo, hi in zip(starts, ends, strict=True)}
    # Fill the unchanged image allowance without dropping a detected scene.
    for index in (0, count-1):
        if len(chosen) < min(LIMIT, count):
            chosen.add(index)
    while len(chosen) < min(LIMIT, count):
        pairs = sorted(zip(sorted(chosen), sorted(chosen)[1:], strict=False),
                       key=lambda pair: (-(pair[1]-pair[0]), pair[0]))
        lo, hi = pairs[0]
        if hi-lo <= 1:
            raise ValueError("本地场景选帧数量不足")
        chosen.add((lo+hi)//2)
    return sorted(chosen)


def parse_scores(output, count):
    """The decoder must report every 25fps frame, not just a few detected cuts."""
    rows, current = [], None
    for line in output.splitlines():
        match = re.search(r"\bframe:(\d+)\s+pts:\s*-?\d+\s+pts_time:([0-9.eE+-]+)", line)
        if match:
            if current is not None:
                raise ValueError("本地场景检测缺少帧分数")
            current = (int(match[1]), float(match[2]))
        score = re.search(r"lavfi\.scene_score=([0-9.eE+-]+)", line)
        if score:
            if current is None:
                raise ValueError("本地场景检测缺少帧时间")
            rows.append((*current, float(score[1])))
            current = None
    if (current is not None or len(rows) != count
            or any(n != i or not math.isfinite(at) or abs(at-i/FPS) > .0001
                   or not math.isfinite(score) or not 0 <= score <= 1
                   for i, (n, at, score) in enumerate(rows))):
        raise ValueError("本地场景检测未覆盖完整窗口，未发送模型审核")
    return [0, *[n for n, _, score in rows if n and score >= SCENE_THRESHOLD]]


def sample(pid, shot, asset, folder, check):
    start, duration, base, count = window(shot, asset)
    deadline = time.monotonic()+DEADLINE_SECONDS

    def guarded():
        check()
        if time.monotonic() >= deadline:
            raise ValueError("本地场景选帧超过总时限，未发送模型审核")

    guarded()
    source = paths.file_path(pid, asset["path"])
    original_hash = library.digest_file(source)
    guarded()
    # Identical input trimming and 25fps sequence for both passes. Output -t
    # instead would drop the final ceil-selected frame at fractional durations.
    input_window = ["-ss", str(start), "-t", str(duration), "-i", str(source), "-an"]
    # Seeking a 30fps source can leave the first decoded timestamp at 0.033s.
    # Normalize both passes identically before the 25fps review grid; otherwise
    # a real first frame appears to be missing and valid middle cuts fail.
    sequence = f"setpts=PTS-STARTPTS,fps={FPS},trim=end_frame={count}"
    output = media.run([paths.binary("ffmpeg"), "-hide_banner", "-v", "info", "-nostdin", "-xerror",
        "-threads", "1", *input_window,
        "-vf", f"{sequence},scale={ANALYSIS_WIDTH}:-2,select='gte(scene,0)',metadata=print",
        "-f", "null", "-"], timeout=DEADLINE_SECONDS, check=guarded)
    guarded()
    starts = parse_scores(output, count)
    indices = indices_for(starts, count)
    select = "+".join(f"eq(n\\,{i})" for i in indices)
    media.run([paths.binary("ffmpeg"), "-v", "error", "-nostdin", "-xerror", "-y", "-threads", "1",
        *input_window, "-vf", f"{sequence},select='{select}',scale=1280:-2", "-fps_mode", "vfr",
        "-frames:v", str(len(indices)), "-q:v", "3", str(folder / "%02d.jpg")],
        timeout=max(.001, deadline-time.monotonic()), check=guarded)
    guarded()
    frames = sorted(folder.glob("*.jpg"))
    if len(frames) != len(indices) or library.digest_file(source) != original_hash:
        raise ValueError("实际视频抽帧不完整或源文件已改变")
    sampling = {"version": VERSION, "policy_sha256": policy_hash(), "fps": FPS, "frame_count": count,
        "analysis_width": ANALYSIS_WIDTH, "scene_threshold": SCENE_THRESHOLD, "scene_starts": starts,
        "indices": indices, "clip_start": start, "duration": duration, "asset_sha256": original_hash,
        "frame_sha256": [library.digest_file(file) for file in frames]}
    guarded()
    return frames, {"sampling": sampling,
        "frames": [{"path": f.relative_to(paths.project_dir(pid)).as_posix(),
                    "at": round(base+start+n/FPS, 3), "kind": "video_frame"}
                   for f, n in zip(frames, indices, strict=True)]}


def validate_preview(shot, asset, preview):
    """Validate code-owned sampling metadata; this cannot prove its semantics."""
    start, duration, base, count = window(shot, asset)
    if not isinstance(preview, dict) or not isinstance(preview.get("sampling"), dict):
        raise ValueError("片段尚未按当前场景覆盖规则抽帧，需要重新核验")
    sampling = preview["sampling"]
    fixed = {"version": VERSION, "policy_sha256": policy_hash(), "fps": FPS, "frame_count": count,
             "analysis_width": ANALYSIS_WIDTH, "scene_threshold": SCENE_THRESHOLD,
             "clip_start": start, "duration": duration}
    if any(sampling.get(k) != v for k, v in fixed.items()):
        raise ValueError("片段采样规则或实际窗口已改变，需要重新核验")
    indices = indices_for(sampling.get("scene_starts"), count)
    frames, hashes = preview.get("frames"), sampling.get("frame_sha256")
    if (sampling.get("indices") != indices or not isinstance(frames, list) or len(frames) != len(indices)
            or not isinstance(hashes, list) or len(hashes) != len(indices)
            or any(not isinstance(h, str) or re.fullmatch(r"[0-9a-f]{64}", h) is None
                   for h in [sampling.get("asset_sha256"), *hashes])
            or any(not isinstance(f, dict) or f.get("kind") != "video_frame"
                   or f.get("at") != round(base+start+n/FPS, 3)
                   for f, n in zip(frames, indices, strict=True))):
        raise ValueError("片段场景覆盖或实际帧绑定无效")
    return sampling


def review_binding(pid, shot, asset):
    """Code-owned file/window identity, never copied from model fields."""
    start, _, base, _ = window(shot, asset)
    fields = (*library.FIELDS, "id", "path", "kind", "width", "height", "duration",
              "source_parts", "source_clip_start", "privacy_redaction", "derived_from_video")
    return {"asset": deepcopy({k: asset.get(k) for k in fields}),
            "sha256": library.digest_file(paths.file_path(pid, asset["path"])),
            "clip_start": start, "duration": shot["duration"], "source_start": base+start}


def verified_selection(project, shot, original, result):
    """Install only the exact reviewed window or its real privacy derivative."""
    pid = project["id"]
    bound = result.get("reviewed_selection") or {}
    before = bound.get("original")
    if before != review_binding(pid, shot, original):
        raise ValueError("核验原素材或窗口已改变，未覆盖当前选择")
    after = bound.get("actual") or {}
    asset = next((a for a in project["assets"] if a["id"] == after.get("asset", {}).get("id")), None)
    if not asset:
        raise ValueError("核验后的实际视频素材不存在")
    actual_shot = shot | {"clip_start": after.get("clip_start")}
    if after != review_binding(pid, actual_shot, asset) or after["duration"] != before["duration"]:
        raise ValueError("核验后的实际素材或窗口已改变")
    # A redaction is a newly encoded copy of the same absolute source interval;
    # it is not permission to select an unrelated file or a different interval.
    if abs(after["source_start"]-before["source_start"]) > .000001:
        raise ValueError("核验后的素材不属于原视频时间窗口")
    if asset["id"] == original["id"]:
        if after != before:
            raise ValueError("同一素材的核验窗口不一致")
    else:
        if after["clip_start"] != 0:
            raise ValueError("隐私派生素材必须从实际审核的本地起点使用")
        current, seen = asset, set()
        while current["id"] != original["id"]:
            if current["id"] in seen:
                raise ValueError("隐私派生素材链循环")
            seen.add(current["id"])
            redaction = current.get("privacy_redaction") or {}
            parent = next((a for a in project["assets"] if a["id"] == redaction.get("original_asset_id")), None)
            expected_start = before["clip_start"] if parent and parent["id"] == original["id"] else 0
            if (not parent or redaction.get("version") != 3
                    or redaction.get("method") != "sampled_union_opaque_boxes"
                    or redaction.get("requires_fresh_review") is not True
                    or redaction.get("duration") != before["duration"]
                    or redaction.get("clip_start") != expected_start
                    or current.get("source_url") != parent.get("source_url")
                    or current.get("kind") != "video" or parent.get("kind") != "video"
                    or abs(float(current.get("source_clip_start", 0))
                           -float(parent.get("source_clip_start", 0))-expected_start) > .000001):
                raise ValueError("核验结果没有绑定原素材的真实隐私派生链")
            current = parent
    preview = result.get("review_preview")
    sampling = validate_preview(actual_shot, asset, preview)
    if (preview.get("asset_id") != asset["id"] or preview.get("clip_start") != after["clip_start"]
            or preview.get("duration") != after["duration"] or sampling["asset_sha256"] != after["sha256"]
            or [library.digest_file(paths.file_path(pid, f["path"])) for f in preview["frames"]]
            != sampling["frame_sha256"]):
        raise ValueError("核验预览没有绑定最终实际视频和完整窗口")
    return asset, after["clip_start"], preview


def selection_error(project, shot, asset):
    if shot.get("locked") or asset.get("kind") != "video":
        return ""
    result = shot.get("verification") or {}
    preview = result.get("review_preview")
    try:
        sampling = validate_preview(shot, asset, preview)
        if (preview.get("asset_id") != asset["id"] or preview.get("clip_start") != shot.get("clip_start", 0)
                or preview.get("duration") != shot["duration"]):
            raise ValueError("片段审核预览与当前选择不一致")
        quality = result.get("visual_quality") or {}
        if quality.get("frame_sha256") != sampling["frame_sha256"]:
            raise ValueError("场景采样与实际画面审核未绑定")
        source = paths.file_path(project["id"], asset["path"])
        if (library.digest_file(source) != sampling["asset_sha256"]
                or [library.digest_file(paths.file_path(project["id"], f["path"])) for f in preview["frames"]]
                != sampling["frame_sha256"]):
            raise ValueError("片段或实际审核帧已改变，需要重新核验")
    except (KeyError, TypeError, OSError, ValueError) as exc:
        return str(exc) or "片段场景覆盖尚未完成核验"
    return ""
