"""Exact local windows awaiting the normal review, never cached approvals.

An interrupted review may already have paid for valid pixel observations. Keep
the source/window/frame binding before that call so recovery can find the same
images without ranking or downloading a different window first.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

from . import costs, execution, frame_sampling, library, paths, pixel_results, planner, window_attempts

VERSION = 1
TTL = 7 * 86400
LIMIT = 128
_collected = ContextVar("pending_window_receipts", default=None)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def uninterrupted(check):
    check()
    # Some legacy observer paths return unknown after an inner failure. The
    # job's real budget-stop event, not a model-written reason, is authoritative.
    stopped = costs._scope.get()[3]
    if stopped is not None and stopped.is_set():
        raise costs.BudgetExceeded("本次任务已触及模型预算，待审窗口已保留；重试不会清零费用")


@contextmanager
def collect():
    receipts = []
    token = _collected.set(receipts)
    try:
        yield receipts
    finally:
        _collected.reset(token)


def folder(pid):
    # Validate the project id using the application's existing path boundary.
    paths.project_dir(pid)
    return paths.DATA / "cache" / "pending-windows" / pid


def context(project, shot, asset):
    original = next(s for s in project["shots"] if s["id"] == shot["id"])
    keys = (*window_attempts.SHOT_FIELDS, "id", "selected_asset", "approved", "source_review_note",
            "source_review_url", "source_unit_id", "source_presentation")
    return {"project": {k: project.get(k) for k in (
                "id", "script", "requirements", "aspect", "media_policy", "min_video_height", "min_video_ratio")},
            "original_shot": {k: original.get(k) for k in keys},
            "narrative_context": planner.narrative_context(project, shot),
            "asset_id": asset["id"], "asset_path": asset["path"],
            "asset": {k: asset.get(k) for k in window_attempts.ASSET_FIELDS},
            "source_url": library.source_url(asset["source_url"]),
            "asset_sha256": library.digest_file(paths.file_path(project["id"], asset["path"])),
            "review_policy": window_attempts.policy(),
            "window": {"clip_start": shot.get("clip_start", 0), "duration": shot["duration"],
                       "fit": shot.get("fit", "cover")}}


def same_context(current, bound, state):
    # Pending means no completed decision exists to reuse. A new reviewer may
    # inspect the same bound bytes under its current rules. Every script,
    # source, window and pixel/sampling check still runs; reviewed/terminal
    # records keep the exact policy binding.
    if state == "pending":
        current = {k:v for k,v in current.items() if k != "review_policy"}
        bound = {k:v for k,v in bound.items() if k != "review_policy"}
    return current == bound


def frame_binding(project, shot, asset, files, preview):
    if not isinstance(preview, dict):
        raise ValueError("待审窗口预览结构损坏")
    start, duration, base = (float(shot.get("clip_start", 0)), float(shot["duration"]),
                             float(asset.get("source_clip_start", 0)))
    available = float(asset["duration"]) - start
    if (not all(math.isfinite(n) for n in (start, duration, base, available))
            or min(start, base) < 0 or not 1 <= duration <= 120 or available + .04 < duration):
        raise ValueError("待审窗口超出真实视频范围")
    sampling = frame_sampling.validate_preview(shot, asset, preview)
    expected = [round(base + start + n / frame_sampling.FPS, 3) for n in sampling["indices"]]
    frames = preview["frames"]
    if len(files) != len(expected):
        raise ValueError("待审窗口帧数量与场景覆盖不一致")
    hashes, relative = [], []
    for file, frame in zip(files, frames, strict=True):
        path = paths.file_path(project["id"], frame["path"])
        if Path(file).resolve() != path:
            raise ValueError("待审窗口帧路径不一致")
        relative.append(frame["path"])
        hashes.append(library.digest_file(path))
    if (hashes != sampling["frame_sha256"] or library.digest_file(paths.file_path(project["id"], asset["path"]))
            != sampling["asset_sha256"]):
        raise ValueError("待审窗口实际视频或采样帧已改变")
    return {"paths": relative, "sha256": hashes, "at": expected, "sampling": digest(sampling)}


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(body, stream, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def read(path):
    try:
        if path.stat().st_size > 2_000_000:
            return None
        body = json.loads(path.read_text("utf-8"), object_pairs_hook=pixel_results.unique_object)
        if (body["version"] != VERSION or body["state"] not in {"pending", "reviewed", "terminal"}
                or not isinstance(body["binding"], dict) or not isinstance(body["frames"], dict)
                or not isinstance(body["preview"], dict)
                or not 0 <= time.time() - body["created_at"] <= TTL
                or body["key"] != path.stem or digest(body["binding"]) != body["key"]):
            return None
        return body
    except (OSError, ValueError, KeyError, TypeError):
        return None


class Receipt:
    def __init__(self, path, body):
        self.path, self.body, self.key = path, body, body["key"]

    def replay(self, project, shot, asset, check, *, allow_terminal=False):
        uninterrupted(check)
        body = read(self.path)
        if (body is None or body["binding"] != self.body["binding"]
                or body["state"] == "terminal" and not allow_terminal):
            raise ValueError("待审窗口记录缺失、损坏或已经完成")
        current = context(project, shot, asset)
        files = [paths.file_path(project["id"], f) for f in body["frames"]["paths"]]
        frames = frame_binding(project, shot, asset, files, body["preview"])
        if not same_context(current, body["binding"]["context"], body["state"]) or frames != body["frames"]:
            raise ValueError("待审窗口的原文、素材或实际帧已改变")
        if frames["sha256"] != body["binding"]["frame_sha256"] or frames["at"] != body["binding"]["at"]:
            raise ValueError("待审窗口帧绑定无效")
        return files, deepcopy(body["preview"])

    def finish(self, state, reason):
        if state not in {"reviewed", "terminal"}:
            raise ValueError("待审窗口状态无效")
        with execution.keyed_lock(("pending-window", str(self.path))):
            body = read(self.path)
            if body is None or body["binding"] != self.body["binding"]:
                return
            body.update(state=state, outcome_reason=str(reason)[:1000], updated_at=time.time())
            write(self.path, body)


def capture(project, row, files, preview, check):
    shot, asset = row["shot"], row["asset"]
    # This first version covers ordinary contiguous public-source video only.
    if asset.get("kind") != "video" or asset.get("source_parts") or not asset.get("source_url"):
        return None
    uninterrupted(check)
    frames = frame_binding(project, shot, asset, files, preview)
    binding = {"version": VERSION, "context": context(project, shot, asset),
               "frame_sha256": frames["sha256"], "at": frames["at"]}
    key = digest(binding)
    path = folder(project["id"]) / (key + ".json")
    with execution.keyed_lock(("pending-window", str(path)), check=check):
        body = read(path)
        if body:
            try:
                Receipt(path, body).replay(project, shot, asset, check, allow_terminal=True)
            except (OSError, ValueError, KeyError, TypeError):
                # A fresh decode may replace missing/corrupt retained JPEGs;
                # it supplies a new pending binding, never a prior decision.
                body = None
        if body and body["state"] == "terminal":
            raise ValueError("同一输入窗口已完成审核，不重复付费；继续寻找其他素材")
        if body is None:
            body = {"version": VERSION, "key": key, "created_at": time.time(), "state": "pending",
                    "binding": binding, "frames": frames, "preview": deepcopy(preview)}
            write(path, body)
    receipt = Receipt(path, body)
    if _collected.get() is not None:
        _collected.get().append(receipt)
    return receipt


def recover(project, shots, check):
    targets = {s["id"]: s for s in shots if not s.get("locked") and not s.get("approved")
               and not s.get("selected_asset")}
    assets = {a["id"]: a for a in project["assets"]}
    rows, seen = [], set()
    diagnostics = {"candidates": 0, "invalid": 0, "terminal": 0, "known_rejections": 0}
    root = folder(project["id"])
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:LIMIT]
    for path in files:
        uninterrupted(check)
        body = read(path)
        if body is None:
            diagnostics["invalid"] += 1
            continue
        try:
            bound = body["binding"]["context"]
            sid = bound["original_shot"]["id"]
            if sid not in targets or sid in seen:
                continue
            asset = assets[bound["asset_id"]]
            shot = targets[sid] | bound["window"]
            # Ignore stale contexts before considering which receipt is newest.
            if not same_context(context(project, shot, asset), bound, body["state"]):
                continue
            receipt = Receipt(path, body)
            receipt.replay(project, shot, asset, check, allow_terminal=True)
            # Only the newest valid window (including a valid terminal one)
            # owns this shot. A corrupt newer record must not hide older work.
            seen.add(sid)
            if body["state"] == "terminal":
                diagnostics["terminal"] += 1
                continue
            attempt = window_attempts.prepare(project, shot, asset)
            rejected = attempt.read() if attempt else None
            if rejected:
                diagnostics["known_rejections"] += 1
                receipt.finish("terminal", rejected["reason"])
                continue
            rows.append({"shot": shot, "asset": asset, "pending_receipt": receipt,
                         "candidate": {"id": asset.get("candidate_id") or "saved-" + asset["id"]}})
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            diagnostics["invalid"] += 1
    diagnostics["candidates"] = len(rows)
    return rows, diagnostics
