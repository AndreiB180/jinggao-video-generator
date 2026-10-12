"""Exact request cache for local-window shortlists, never for approvals."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
import tempfile
import time
from pathlib import Path

from . import config, library, model_protocol, model_routing, paths


class Ranking:
    def __init__(self, system, data, candidates, options):
        self.inputs = deepcopy((system, data, candidates, options))
        bindings = {}
        checked = {}
        for sid, refs in candidates.items():
            bindings[sid] = {}
            for ref, clip in refs.items():
                if clip.get("project_id"):
                    from . import project_pool
                    asset = project_pool.resolve(clip, digests=checked)
                    bindings[sid][ref] = {"blob": clip["blob_id"], "project_id": clip["project_id"],
                        "asset_id": asset["id"], "source": clip["source"],
                        "start": clip["start"], "duration": clip["duration"], "offset": clip["offset"],
                        "observation": {k: clip.get(k) for k in ("description", "visible_text", "observation_method")}}
                    continue
                digest = clip["blob_id"]
                if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                    raise ValueError("Ranking reuse requires actual media binding")
                file = library.file_path(clip["path"])
                if file not in checked:
                    checked[file] = library.digest_file(file)
                if checked[file] != digest:
                    raise ValueError("Indexed media changed")
                bindings[sid][ref] = {"blob": digest, "source": library.source_url(clip["source"]["source_url"]),
                    "start": clip["start"], "duration": clip["duration"], "offset": clip["offset"],
                    "observation": {k: clip.get(k) for k in ("description", "visible_text", "observation_method")}}
        cfg = config.read()
        self.model = model_routing.model(cfg, "library_ranking")
        payload = {"model": self.model, "system": system,
                   "data": data, "options": options, "bindings": bindings}
        model_protocol.apply(payload, cfg)
        key = json.dumps({"version": 1, "payload": payload, "endpoint": cfg["model_base_url"]},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self.path = paths.DATA / "cache" / "library-rank-responses" / (hashlib.sha256(key.encode()).hexdigest()+".json")

    def current(self):
        try:
            return Ranking(*self.inputs).path == self.path
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def read(self, valid):
        try:
            if not self.current():
                return None
            if self.path.stat().st_size > 1_000_000:
                return None
            data = json.loads(self.path.read_text("utf-8"))
            if 0 <= time.time()-data["created"] <= 7*86400 and valid(data["response"]):
                return deepcopy(data["response"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def save(self, response, valid):
        if not self.current() or not valid(response):
            return
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as file:
                temporary = Path(file.name)
                json.dump({"created": time.time(), "response": response}, file, ensure_ascii=False)
            temporary.replace(self.path)
        except OSError:
            pass
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


def prepare(*args):
    try:
        return Ranking(*args)
    except (OSError, KeyError, TypeError, ValueError):
        return None
