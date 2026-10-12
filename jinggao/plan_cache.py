"""Reuse validated review batches on resume, keyed by the complete request.

Shot UUIDs are canonicalized by global order, so regenerating identifiers alone
does not discard paid work. Every cache hit still passes current validation.
"""
import hashlib
import json
import os
import tempfile
import time

from . import config, model_routing, paths


def _ids(value, mapping):
    if isinstance(value, list):
        return [_ids(item, mapping) for item in value]
    if isinstance(value, dict):
        return {k: mapping.get(v, v) if k == "shot_id" and isinstance(v, str) else _ids(v, mapping)
                for k, v in value.items()}
    return value


class Batch:
    def __init__(self, project, system, payload, options):
        cfg = config.read()
        self.to_stable = {r["shot_id"]: f"shot-{i}" for i, r in enumerate(payload["readonly_global_plan"])}
        self.to_current = {v: k for k, v in self.to_stable.items()}
        key = {"version": 1, "system": system, "request": _ids(payload, self.to_stable), "options": options,
               "endpoint": cfg.get("model_base_url"),
               "model": model_routing.model(cfg, options.get("task", "requirements"))}
        digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        # Pure callers with no saved project have no durable resume scope.
        self.path = paths.DATA / "cache" / "plan-review" / (digest+".json") if project.get("id") else None

    def read(self):
        try:
            if self.path is None or self.path.stat().st_size > 4_000_000:
                return None
            data = json.loads(self.path.read_text("utf-8"))
            if not 0 <= time.time()-data["created"] <= 7*86400:
                return None
            return _ids(data["response"], self.to_current)
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def save(self, response):
        if self.path is None:
            return
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=".review-", delete=False) as file:
                temporary = file.name
                json.dump({"created": time.time(), "response": _ids(response, self.to_stable)}, file, ensure_ascii=False)
            os.replace(temporary, self.path)
        except OSError:
            pass  # A missing cache cannot invalidate an already validated result.
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


class PlanningBatch(Batch):
    """Cache only completely validated metadata, never approvals or guesses."""
    def __init__(self, project, system, payload, options):
        cfg = config.read()
        self.to_stable = self.to_current = {}
        key = {"version": 1, "system": system, "request": payload, "options": options,
               "endpoint": cfg.get("model_base_url"), "model": model_routing.model(cfg, "planning")}
        digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.path = paths.DATA / "cache" / "planning" / (digest+".json") if project.get("id") else None


class ContractExtraction(Batch):
    """Cache a complete source-contract response set, never a compiled approval.

    The request's opaque snapshot contains random shot IDs. All meaningful
    extraction inputs already occur in its source units, targets and read-only
    fields; those remain in the key. Responses use stable source refs and must
    be compiled against the current shots again before reuse.
    """

    def __init__(self, project, system, requests, options):
        cfg = config.read()
        self.to_stable = self.to_current = {}
        self.task = options.get("task", "requirements")
        self.model = model_routing.model(cfg, self.task)
        self.endpoint = cfg.get("model_base_url")
        key = {"version": 1, "system": system,
               "requests": [{k: v for k, v in request.items() if k != "snapshot"}
                            for request in requests],
               "options": options, "endpoint": self.endpoint, "model": self.model}
        digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.path = paths.DATA / "cache" / "editorial-contracts" / (digest+".json") if project.get("id") else None

    def current(self):
        """Do not read or publish under a route that changed while waiting."""
        cfg = config.read()
        return (cfg.get("model_base_url") == self.endpoint
                and model_routing.model(cfg, self.task) == self.model)
