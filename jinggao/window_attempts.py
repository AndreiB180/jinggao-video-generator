"""Remember only grounded conflicts for an unchanged, exact local video window.

The lookup binds the original media bytes and sampling policy, so a resume can
avoid decoding the same rejected interval again. Saved sample bytes are checked
as well. This store cannot grant approval or turn missing evidence into conflict.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

from . import config, curation, frame_contract, frame_sampling, library, model_routing, neutral_observer, paths, planner

VERSION = 1
TTL = 7 * 86400
POLICY_FILES = ("planner", "pooled_match", "editorial_contract", "editorial_review", "frame_contract",
                "frame_quality", "visual_quality", "neutral_observer", "review_evidence", "review_packet",
                "identity", "context_labels", "curation", "privacy", "model_protocol", "pixel_results",
                "pixel_observer", "pixel_packet", "source_planner", "source_units", "frame_sampling", "window_attempts")
SHOT_FIELDS = tuple(dict.fromkeys((*planner.SHOT_FIELDS, "duration", "visual_plan", "editorial_contract",
                                  "clip_start", "fit", "fit_origin", "crop", "privacy_masks", "locked", "overlay_label_origin")))
ASSET_FIELDS = tuple(dict.fromkeys((*library.FIELDS, "kind", "width", "height", "duration", "source_parts",
                                  "source_clip_start", "entity_names", "origin_image_url", "url")))


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def policy():
    cfg = config.read()
    return {"files": {name: hashlib.sha256(Path(__file__).with_name(name + ".py").read_bytes()).hexdigest()
                      for name in POLICY_FILES},
            "models": {task: model_routing.model(cfg, task) for task in ("evidence_review", "pixel_observation")},
            "endpoint": cfg["model_base_url"],
            "options": {key: value for key, value in cfg.items() if key.startswith("model_reasoning")},
            "pixel_observer_mode": cfg.get("pixel_observer_mode", "separate")}


def signature(project, shot, asset):
    if asset.get("kind") != "video":
        raise ValueError("Rejected-window reuse requires actual video")
    start = float(shot.get("clip_start", 0))
    duration = float(shot["duration"])
    base = float(asset.get("source_clip_start", 0))
    if not all(math.isfinite(n) for n in (start, duration, base)) or min(start, base) < 0 or duration <= 0:
        raise ValueError("Invalid rejected-window interval")
    media = paths.file_path(project["id"], asset["path"])
    source = library.source_url(asset["source_url"])
    effective = shot | {"fit": curation.preferred_fit(shot, asset, project.get("aspect", "16:9"))}
    return {"version": VERSION, "project_id": project["id"], "source": source,
            "asset_sha256": library.digest_file(media), "source_start": base + start, "duration": duration,
            "project": {k: project.get(k) for k in ("script", "requirements", "aspect", "media_policy", "min_video_height")},
            "shot": {k: effective.get(k) for k in SHOT_FIELDS},
            "context": planner.narrative_context(project, shot),
            "asset": {k: asset.get(k) for k in ASSET_FIELDS}, "policy": policy()}


def valid_rejection(audit, shot, hashes):
    """Revalidate complete, literal evidence; unknown/errors never blacklist."""
    if not isinstance(audit, dict) or audit.get("status") != "conflict":
        return False
    if not isinstance(audit.get("reason"), str) or not audit["reason"].strip():
        return False
    quality = audit.get("visual_quality")
    if not isinstance(quality, dict) or quality.get("frame_sha256") != hashes:
        return False
    try:
        frames = neutral_observer.parse_groups(encoded({"groups": [{"id": "window",
            "scene_description": audit.get("scene_description"), "frames": audit.get("frame_observations")}]}),
            {"window": len(hashes)})["window"]["frames"]
        evidence = audit.get("evidence")
        labels = audit.get("labels")
        if (not isinstance(evidence, list) or not evidence
                or any(not neutral_observer.grounded(row, frames) for row in evidence)
                or not isinstance(labels, list)
                or any(not neutral_observer.grounded(row, frames, label=True) for row in labels)):
            return False
        status, _, _ = frame_contract.validate(shot, audit.get("primary_checks"), frames)
        # An explicit, grounded per-condition contradiction is reusable. Pixel
        # quality rejections have a different contract and are not cached here.
        return status == "conflict"
    except (KeyError, TypeError, ValueError):
        return False


class Rejection:
    def __init__(self, project, shot, asset):
        self.project, self.shot, self.asset = project, shot, asset
        self.binding = signature(project, shot, asset)
        self.key = hashlib.sha256(encoded(self.binding).encode()).hexdigest()
        self.path = paths.DATA / "cache" / "rejected-windows" / project["id"] / (self.key + ".json")

    def read(self):
        try:
            if signature(self.project, self.shot, self.asset) != self.binding:
                return None
            if self.path.stat().st_size > 2_000_000:
                return None
            body = json.loads(self.path.read_text("utf-8"))
            if (body.get("binding") != self.binding or body.get("outcome") != "rejected"
                    or not 0 <= time.time() - body["created_at"] <= TTL):
                return None
            frames, hashes = body["frame_paths"], body["frame_sha256"]
            sampling = frame_sampling.validate_preview(self.shot, self.asset, body.get("preview"))
            expected = len(sampling["indices"])
            if not isinstance(frames, list) or len(frames) != expected or len(hashes) != expected:
                return None
            actual = [library.digest_file(paths.file_path(self.project["id"], file)) for file in frames]
            if (actual != hashes or hashes != sampling["frame_sha256"]
                    or not valid_rejection(body["audit"], self.shot, hashes)):
                return None
            return {"reason": body["audit"]["reason"], "key": self.key}
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, row):
        temporary = None
        try:
            # Privacy repair/cropping/source changes can return a different
            # asset or interval: do not attach that decision to the old input.
            if signature(self.project, row["shot"], row["asset"]) != self.binding:
                return False
            result = row.get("verification") or {}
            audit = result.get("identity_audit")
            if result.get("verdict") != "mismatch" or not result.get("selection_blocked"):
                return False
            files = [Path(file) for file in row.get("frame_paths", [])]
            hashes = [library.digest_file(file) for file in files]
            sampling = frame_sampling.validate_preview(row["shot"], row["asset"], row.get("preview"))
            if (not hashes or hashes != sampling["frame_sha256"]
                    or not valid_rejection(audit, row["shot"], hashes)):
                return False
            root = paths.project_dir(self.project["id"]).resolve()
            body = {"binding": self.binding, "outcome": "rejected", "created_at": time.time(),
                    "frame_paths": [file.resolve().relative_to(root).as_posix() for file in files],
                    "frame_sha256": hashes, "preview": deepcopy(row["preview"]), "audit": deepcopy(audit)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as file:
                temporary = Path(file.name)
                json.dump(body, file, ensure_ascii=False)
            temporary.replace(self.path)
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


def prepare(project, shot, asset):
    try:
        return Rejection(project, shot, asset)
    except (OSError, ValueError, KeyError, TypeError):
        return None
