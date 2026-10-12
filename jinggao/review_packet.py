"""Lossless transport for the text judge, not a replacement for its evidence.

Repeated source paragraphs live once in a table. Short record/citation IDs are
expanded before the existing identity, action and grounding checks run. Cached
raw judgments always go through those checks again; approvals are never cached.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import time

from . import paths, review_evidence

VERSION = 1
PROTOCOL = """\n输入使用无损结构化封包 review_records_v1：records 中每项就是此前要求的一个镜头。
texts 是共享原文表。任何字段的 {\"$text\":\"T0\"} 等值于 texts.T0 的完整原文；先读取对应原文再判断，不把引用对象当成缺失或空值。不得忽略共享全文、要求、事实或反证。
shot_id 和 observation_references.ref 已缩短，只返回输入中的这些短ID。输出要求和全部内容审核规则保持不变。
"""


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _walk(value, transform):
    if isinstance(value, list):
        return [_walk(v, transform) for v in value]
    if isinstance(value, dict):
        return {k: _walk(v, transform) for k, v in value.items()}
    return transform(value)


def _ids(value, mapping):
    if isinstance(value, list):
        return [_ids(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: mapping.get(v, v) if k in {"shot_id", "ref"} and isinstance(v, str)
                else _ids(v, mapping) for k, v in value.items()}
    return value


class Packet:
    def __init__(self, content):
        if (not isinstance(content, list) or not content
                or any(not isinstance(c, dict) or c.get("type") != "text" for c in content)):
            raise ValueError("Text review packets require complete JSON records")
        self.original = [json.loads(c["text"]) for c in content]
        if any(not isinstance(r, dict) or not isinstance(r.get("shot_id"), str) for r in self.original):
            raise ValueError("Missing review record ID")
        ids = [r["shot_id"] for r in self.original]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate review record ID")
        self.forward = {sid: f"r{i}" for i, sid in enumerate(ids)}
        for record in self.original:
            sid = record["shot_id"]
            for ref in record.get("observation_references", []):
                key = ref.get("ref")
                if not isinstance(key, str) or not key.startswith(sid+":"):
                    raise ValueError("Unbound observation reference")
                self.forward[key] = self.forward[sid]+key[len(sid):]
        if len(set(self.forward.values())) != len(self.forward):
            raise ValueError("Ambiguous short reference")
        self.reverse = {v: k for k, v in self.forward.items()}
        records = _ids(self.original, self.forward)
        strings = Counter()
        def count(value):
            if isinstance(value, str) and len(value.encode("utf-8")) >= 48:
                strings[value] += 1
            return value
        _walk(records, count)
        texts = {}
        for value, count in strings.items():
            if count < 2:
                continue
            key = f"T{len(texts)}"
            literal_bytes = len(encoded(value).encode("utf-8"))
            reference_bytes = len(encoded({"$text": key}).encode("utf-8"))
            table_bytes = len(encoded(key).encode("utf-8")) + literal_bytes + 2
            if literal_bytes * count > reference_bytes * count + table_bytes:
                texts[key] = value
        replace = {v: {"$text": k} for k, v in texts.items()}
        self.body = {"schema": "review_records_v1", "texts": texts,
                     "records": _walk(records, lambda v: replace.get(v, v) if isinstance(v, str) else v)}
        if self.unpack() != self.original:
            raise ValueError("Review packet changed source evidence")
        self.text = encoded(self.body)
        self.stats = {"original_bytes": len(encoded(self.original).encode()),
                      "packed_bytes": len(self.text.encode()), "shared_texts": len(texts)}

    def unpack(self):
        def expand(value):
            if isinstance(value, list):
                return [expand(v) for v in value]
            if isinstance(value, dict):
                if set(value) == {"$text"}:
                    return self.body["texts"][value["$text"]]
                return {k: expand(v) for k, v in value.items()}
            return value
        return _ids(expand(self.body["records"]), self.reverse)

    def restore(self, response):
        return _ids(response, self.reverse)

    def valid_response(self, response):
        """Only cache complete, grounded raw responses; still recheck on use."""
        from .neutral_observer import grounded
        rows = response.get("reviews") if isinstance(response, dict) else None
        expected = {self.forward[r["shot_id"]]: r for r in self.original}
        if (not isinstance(rows, list) or len(rows) != len(expected)
                or any(not isinstance(r, dict) or not isinstance(r.get("shot_id"), str) for r in rows)
                or {r["shot_id"] for r in rows} != set(expected)):
            return False
        for raw in rows:
            record = expected[raw["shot_id"]]
            frames = record.get("frame_observations", [])
            _, lookup = review_evidence.references(record["shot_id"], frames)
            row = review_evidence.resolve(self.restore(raw), lookup)
            if (not isinstance(row.get("status"), str) or row["status"] not in {"supported", "unknown", "conflict"}
                    or not isinstance(row.get("reason"), str)):
                return False
            for key in ("labels", "evidence"):
                if not isinstance(row.get(key), list) or any(not grounded(v, frames, label=key=="labels") for v in row[key]):
                    return False
            checks = row.get("primary_checks")
            required = set(record.get("selected_visual_requirements", {}))
            if (not isinstance(checks, list) or len(checks) != len(required)
                    or any(not isinstance(v, dict) or not isinstance(v.get("key"), str) for v in checks)
                    or {v["key"] for v in checks} != required):
                return False
            for check in checks:
                if (not isinstance(check.get("status"), str) or check["status"] not in {"supported", "unknown", "conflict"}
                        or not isinstance(check.get("reason"), str) or not isinstance(check.get("evidence"), list)
                        or any(not review_evidence.condition_grounded(check["key"], v, frames) for v in check["evidence"])
                        or check["status"] != "unknown" and not check["evidence"]):
                    return False
            if row["status"] != "unknown" and not row["evidence"]:
                return False
        return True


class Cache:
    def __init__(self, packet, payload, endpoint):
        self.packet = packet
        key = {"version": VERSION, "payload": payload, "endpoint": endpoint}
        self.path = paths.DATA / "cache" / "text-judgments" / (hashlib.sha256(encoded(key).encode()).hexdigest()+".json")

    def bound(self):
        for record in self.packet.original:
            binding = record.get("evidence_binding", {})
            if not isinstance(binding, dict):
                return False
            hashes = binding.get("frame_sha256")
            frames = record.get("frame_observations")
            if (not isinstance(frames, list) or not frames or not isinstance(hashes, list)
                    or len(hashes) != len(frames) or any(not isinstance(h, str)
                        or re.fullmatch(r"[a-f0-9]{64}", h) is None for h in hashes)):
                return False
            for key in ("clip_start", "duration"):
                value = binding.get(key)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    return False
        return True

    def read(self):
        if not self.bound():
            return None
        try:
            if self.path.stat().st_size > 2_000_000:
                return None
            data = json.loads(self.path.read_text("utf-8"))
            if 0 <= time.time()-data["created"] <= 7*86400 and self.packet.valid_response(data["response"]):
                return deepcopy(data["response"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def save(self, response):
        if not self.bound() or not self.packet.valid_response(response):
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
