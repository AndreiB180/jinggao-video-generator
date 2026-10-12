"""Opt-in joint pixel facts/quality observer.

One transport yields two independently validated projections. Neither projection
is editorial approval. Invalid/truncated responses stay unknown, with no retry
or fallback to the two existing observers.
"""
from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from uuid import uuid4

import httpx

from . import config, costs, execution, model_protocol, network, neutral_observer, paths, pixel_packet, pixel_results, visual_quality

VERSION = 2
MAX_GROUPS = 4
GROUPS_PER_REQUEST = 2
MAX_FRAMES = 5
MAX_TOKENS = 4000
LOW_TIER_INPUT_BOUND = 32_000
TTL = 30 * 86400
TASK = "pixel_observation_joint"
FACT_FIELDS = {"frame_index", "objects", "action", "text"}
QUALITY_FIELDS = {"frame_index", "representation", "type_evidence", "overlays"}
FRAME_FIELDS = FACT_FIELDS | QUALITY_FIELDS
PROTOCOL = (
    neutral_observer.PROTOCOL.split("只返回一个JSON对象", 1)[0]
    + "\n" + visual_quality.PROTOCOL.split("只返回一个JSON对象", 1)[0]
    + "\n以上事实记录和视觉质量检查在同一份输出中完成，不返回两份JSON，不省略其中任一部分。\n"
    + pixel_packet.PROTOCOL
)


class PixelObservationError(ValueError):
    def __init__(self, message, *, request_count=0):
        super().__init__(message)
        self.request_count = self.model_requests = request_count


def parse_groups(raw, expected):
    """Project complete joint records through both unchanged strict parsers."""
    try:
        body = json.loads(raw, object_pairs_hook=pixel_results.unique_object)
    except (ValueError, TypeError):
        raise ValueError("联合观察未返回无歧义的完整JSON") from None
    if not isinstance(body, dict) or set(body) != {"groups"} or not isinstance(body["groups"], list):
        raise ValueError("联合观察分组格式无效")
    facts, quality = [], []
    for group in body["groups"]:
        if (not isinstance(group, dict) or set(group) != {"id", "scene_description", "frames"}
                or not isinstance(group["frames"], list)):
            raise ValueError("联合观察缺少完整分组字段")
        for frame in group["frames"]:
            if not isinstance(frame, dict) or set(frame) != FRAME_FIELDS:
                raise ValueError("联合观察缺少事实或质量字段")
            overlays = frame["overlays"]
            if (not isinstance(overlays, list) or any(not isinstance(row, dict)
                    or set(row) != {"kind", "text", "bbox", "impact", "occludes"} for row in overlays)):
                raise ValueError("联合观察覆盖物字段不完整")
        facts.append({"id": group["id"], "scene_description": group["scene_description"],
                      "frames": [{key: frame[key] for key in FACT_FIELDS} for frame in group["frames"]]})
        quality.append({"id": group["id"],
                        "frames": [{key: frame[key] for key in QUALITY_FIELDS} for frame in group["frames"]]})
    return (neutral_observer.parse_groups(json.dumps({"groups": facts}, ensure_ascii=False), expected),
            visual_quality.parse_groups(json.dumps({"groups": quality}, ensure_ascii=False), expected))


def parse_wire(raw, expected):
    """Validate named records before both existing semantic validators."""
    return parse_groups(json.dumps(pixel_packet.expand(raw), ensure_ascii=False), expected)


def _payload(groups, cfg, *, max_tokens=None):
    content = [{"type": "text", "text": json.dumps({"observation_groups": [
        {"group_id": ident, "frame_indices": list(range(len(frames)))} for ident, frames in groups.items()]})}]
    for ident, frames in groups.items():
        for index, (_, picture) in enumerate(frames):
            content.extend([{"type": "text", "text": json.dumps({"group_id": ident, "frame_index": index})},
                            {"type": "image_url", "image_url": {
                                "url": "data:image/jpeg;base64," + base64.b64encode(picture).decode()}}])
    payload = {"model": cfg["vision_model"], "messages": [
        {"role": "system", "content": PROTOCOL}, {"role": "user", "content": content}],
        "max_tokens": MAX_TOKENS if max_tokens is None else max_tokens, "response_format": {"type": "json_object"}}
    model_protocol.apply(payload, cfg)
    return payload


def _cache_key(cfg, prepared, *, max_tokens=None):
    # Canonical group IDs make a valid group's cache independent of batchmates
    # and shot IDs. Preserve every actual transport option, including max_pixels.
    payload = _payload({"g0": prepared}, cfg, max_tokens=MAX_TOKENS if max_tokens is None else max_tokens)
    for block in payload["messages"][1]["content"]:
        if block["type"] == "image_url":
            encoded = block["image_url"]["url"]
            block["image_url"]["url"] = "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
    binding = {"version": VERSION, "endpoint": cfg["model_base_url"].rstrip("/"), "payload": payload,
               "frame_sha256": [sha for sha, _ in prepared],
               "validators": [neutral_observer.VERSION, neutral_observer.PROTOCOL,
                              visual_quality.VERSION, visual_quality.PROTOCOL]}
    return hashlib.sha256(json.dumps(binding, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _group(cfg, prepared):
    """Keep actual 1/2/4-group output bindings; share one flight per pixel group.

    A four-group response uses 8000, never a fictitious 4000 cache key. Complete
    projections from either real transport are reusable after strict parsing.
    The 4000 key stays the lock identity so overlapping pairs/fours coalesce.
    """
    variants = {limit: _cache_key(cfg, prepared, max_tokens=limit)
                for limit in (MAX_TOKENS, MAX_TOKENS * 2, (MAX_TOKENS + 1) // 2)}
    key = variants[MAX_TOKENS]
    return key, {"frames": prepared, "hashes": [sha for sha, _ in prepared],
                 "target": _cache_path(key), "variants": variants,
                 "cache_key": key, "observation": None, "cached": False}


def _cache_path(key):
    return paths.DATA / "cache" / "pixel-observer" / (key + ".json")


def _read_group_cache(group):
    for key in group["variants"].values():
        observation = _read_cache(_cache_path(key), key, group["hashes"])
        if observation is not None:
            group.update(observation=observation, cache_key=key, cached=True)
            return True
    return False


def _request_limit(count):
    # Four groups retain the same total output allowance as two pairs. This
    # saves repeated input and round trips, not the maximum output reservation.
    return (MAX_TOKENS + 1) // 2 if count == 1 else MAX_TOKENS * (2 if count > GROUPS_PER_REQUEST else 1)


def _supports_four(cfg):
    return costs._provider(cfg) == "dashscope.aliyuncs.com" and cfg["vision_model"] in costs.QWEN_FLASH_MODELS


def _partition(groups, keys, cfg):
    """Choose 4/2/1 before sending, using the same actual bound as the ledger.

    Unknown providers keep the old two-group policy. A single oversized group
    remains subject to the existing pricing/budget guard; nothing is truncated.
    """
    expanded = _supports_four(cfg)
    headroom = costs.project_headroom() if expanded else None
    pending, batches = list(keys), []
    while pending:
        count = 4 if expanded and len(pending) >= 4 else min(GROUPS_PER_REQUEST, len(pending))
        while expanded and count > 1:
            ordered = {f"g{i}": groups[key]["frames"] for i, key in enumerate(pending[:count])}
            limit = _request_limit(count)
            bound = costs._input_bound(cfg, _payload(ordered, cfg, max_tokens=limit))
            reserve = costs._amount(costs._rates(cfg, cfg["vision_model"], bound), bound, limit)
            if bound <= LOW_TIER_INPUT_BOUND and (headroom is None or reserve <= headroom):
                break
            count //= 2
        batches.append(pending[:count])
        del pending[:count]
    return batches


def _joined(facts, quality, ident):
    return {"scene_description": facts[ident]["scene_description"],
            "frames": [fact | quality[ident]["frames"][n]
                       for n, fact in enumerate(facts[ident]["frames"])]}


def _complete_wire_groups(raw, expected):
    """Retain only complete siblings in an unambiguous, complete JSON envelope."""
    try:
        body = json.loads(raw, object_pairs_hook=pixel_results.unique_object)
    except (ValueError, TypeError):
        return {}
    if (not isinstance(body, dict) or set(body) != {"schema", "groups"}
            or body["schema"] != pixel_packet.VERSION):
        return {}

    def parse(single, selected):
        packet = {"schema": pixel_packet.VERSION, **json.loads(single)}
        facts, quality = parse_wire(json.dumps(packet, ensure_ascii=False), selected)
        return {ident: _joined(facts, quality, ident) for ident in selected}

    return pixel_results.complete_groups(json.dumps({"groups": body["groups"]}, ensure_ascii=False), expected, parse)


def _read_cache(target, key, hashes):
    try:
        if target.stat().st_size > 2_000_000:
            return None
        body = json.loads(target.read_text("utf-8"), object_pairs_hook=pixel_results.unique_object)
        if (body["version"] != VERSION or body["key"] != key or body["frame_sha256"] != hashes
                or not 0 <= time.time() - body["created_at"] <= TTL):
            return None
        group = {"id": "cache", **body["observation"]}
        parse_groups(json.dumps({"groups": [group]}, ensure_ascii=False), {"cache": len(hashes)})
        return body["observation"]
    except (OSError, KeyError, ValueError, TypeError):
        return None


def cached(records, check, *, cfg=None):
    """Read complete joint observations without transport, locks or writes.

    Return the same neutral/quality projections as ``observe`` for hits only.
    ``missing_shot_ids`` explicitly identifies records needing the caller's
    normal observation route. A miss never triggers a paid joint fallback.
    API credentials are unnecessary; current model/endpoint, protocol/options
    and ordered actual image bytes still bind every hit.
    """
    started = time.monotonic()
    stats = {"model_requests": 0, "request_count": 0, "cache_hits": 0,
             "singleflight_hits": 0, "unique_groups": 0, "failed_shot_ids": [],
             "missing_shot_ids": [], "all_cache_hit": False, "model_info": [],
             "task": TASK, "diagnostic_paths": [], "cache_only": True}
    if not records:
        return {}, {}, stats | {"elapsed_seconds": 0}
    if not 1 <= len(records) <= MAX_GROUPS:
        raise ValueError("联合观察缓存每批最多4组")
    cfg = deepcopy(config.read() if cfg is None else cfg)
    if not all(cfg.get(key) for key in ("vision_model", "model_base_url")):
        raise PixelObservationError("联合观察缓存需要当前视觉模型和接口")

    def checked():
        try:
            check()
        except Exception as exc:
            exc.request_count = exc.model_requests = 0
            raise

    groups, shots, files_by_shot = {}, {}, {}
    for record in records:
        checked()
        sid, files = record["shot"]["id"], record.get("frame_paths")
        if (not isinstance(sid, str) or not sid or sid in shots or not isinstance(files, (tuple, list))
                or not 1 <= len(files) <= MAX_FRAMES):
            raise ValueError("联合观察镜头编号重复或实际帧数量不在1–5之间")
        shots[sid], files_by_shot[sid] = None, files
        prepared = []
        for path in files:
            checked()
            try:
                prepared.append(neutral_observer._frame(path))
            except (OSError, ValueError):
                break
        if len(prepared) != len(files):
            continue
        key, group = _group(cfg, prepared)
        shots[sid] = key
        if key not in groups:
            _read_group_cache(group)
            groups[key] = group

    facts_out, quality_out, hit_keys, missing = {}, {}, set(), []
    for sid, key in shots.items():
        checked()
        group = groups.get(key)
        if not group or group["observation"] is None:
            missing.append(sid)
            continue
        # Validate both projections again and ensure the actual files have not
        # changed since key construction. Never return a partial cached group.
        try:
            facts, quality = parse_groups(json.dumps({"groups": [
                {"id": "current", **group["observation"]}]}, ensure_ascii=False), {"current": len(group["hashes"])})
            if [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files_by_shot[sid]] != group["hashes"]:
                missing.append(sid)
                continue
            assessed = visual_quality.assess(quality["current"])
        except (OSError, KeyError, ValueError, TypeError):
            missing.append(sid)
            continue
        common = {"frame_sha256": group["hashes"], "observation_model": cfg["vision_model"],
                  "cached": True, "cache_key": group["cache_key"], "observation_status": "complete",
                  "content_approved": False}
        facts_out[sid], quality_out[sid] = facts["current"] | common, assessed | common
        hit_keys.add(key)
    checked()
    return facts_out, quality_out, stats | {
        "cache_hits": len(hit_keys), "unique_groups": len(groups),
        "failed_shot_ids": missing, "missing_shot_ids": missing,
        "all_cache_hit": bool(shots) and not missing, "model": cfg["vision_model"],
        "elapsed_seconds": round(time.monotonic() - started, 3)}


def _request(payload, cfg, check, deadline):
    """Reserve and send the same already-final payload exactly once."""
    from .planner import ModelUnavailable
    sent, acquired, started = False, False, time.monotonic()
    transport_trace = network.ModelTrace()

    def bounded_check():
        check()
        if time.monotonic() >= deadline:
            raise TimeoutError("联合观察总时限已到")

    try:
        while not execution.MODEL_SLOTS.acquire(timeout=.1):
            bounded_check()
        acquired = True
        bounded_check()
        with costs.request(cfg, payload, TASK, check=bounded_check) as charge:
            bounded_check()
            # The caller owns one deadline for locks, transport and all batches.
            # Do not discard its remaining time with a shorter hidden HTTP cap:
            # a request may already be billable even if no response is received.
            remaining = deadline - time.monotonic()

            async def post():
                async with network.model_async_client(cfg["model_base_url"], timeout=remaining) as client:
                    async def send():
                        nonlocal sent
                        bounded_check()
                        charge.mark_sent()
                        sent = True
                        response = await client.post(cfg["model_base_url"].rstrip("/") + "/chat/completions",
                            json=payload, headers={"Authorization": "Bearer " + cfg["model_api_key"]},
                            extensions={"trace": transport_trace.async_record})
                        charge.response(response)
                        return response
                    task = asyncio.create_task(send())
                    try:
                        async with asyncio.timeout(remaining):
                            while not task.done():
                                bounded_check()
                                await asyncio.wait({task}, timeout=.1)
                            bounded_check()
                            return task.result()
                    finally:
                        if not task.done():
                            task.cancel()
                            with suppress(asyncio.CancelledError):
                                await task

            response = asyncio.run(post())
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        raw = choice["message"]["content"]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("联合观察返回空内容")
        info = {"model": cfg["vision_model"], "response_model": body.get("model", ""),
                "usage": body.get("usage", {}), "finish_reason": choice.get("finish_reason"),
                "request_count": 1, "task": TASK, "input_bound": costs._input_bound(cfg, payload),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "request_options": {key: payload[key] for key in
                    ("max_tokens", "thinking", "enable_thinking", "temperature", "reasoning_effort") if key in payload}}
        return raw, info
    except httpx.HTTPStatusError as exc:
        code = exc.response.status_code
        if code in (401, 402, 403, 429):
            failure = ModelUnavailable(f"联合观察模型账户不可用（HTTP {code}），已停止任务")
            failure.request_count = failure.model_requests = int(sent)
            raise failure from None
        raise PixelObservationError(f"联合观察返回HTTP {code}，未重试", request_count=int(sent)) from None
    except (httpx.RequestError, TimeoutError) as exc:
        transport_trace.save(exc)
        failure = PixelObservationError("联合观察连接失败或超时，未重试", request_count=int(sent))
        if hasattr(exc, "transport_diagnostic_path"):
            failure.transport_diagnostic_path = exc.transport_diagnostic_path
        raise failure from None
    except (KeyError, IndexError, TypeError, ValueError):
        raise PixelObservationError("联合观察响应无效，未重试", request_count=int(sent)) from None
    except Exception as exc:
        exc.request_count = exc.model_requests = int(sent)
        raise
    finally:
        if acquired:
            execution.MODEL_SLOTS.release()


def _observe_batch(groups, keys, cfg, check, deadline, diagnostics_dir):
    def bounded_check():
        check()
        if time.monotonic() >= deadline:
            raise TimeoutError("联合观察等待缓存或模型超时，未启动更多请求")

    with neutral_observer._group_flights(groups, keys, bounded_check) as flights:
        missing = []
        for key in keys:
            group, flight = groups[key], flights[key]
            cached = _read_group_cache(group)
            if cached or "observation" in flight:
                if not cached:
                    group.update(observation=deepcopy(flight["observation"]), cache_key=flight["cache_key"])
                group.update(cached=True, shared=True)
            elif "failure" in flight:
                if flight["failure"]["account_unavailable"]:
                    neutral_observer._raise_shared_failure(flight["failure"])
                group.update(error=flight["failure"]["reason"], shared=True)
            else:
                missing.append(key)
        if not missing:
            return 0, [], []
        ordered = {f"g{i}": groups[key]["frames"] for i, key in enumerate(missing)}
        expected = {ident: len(frames) for ident, frames in ordered.items()}
        payload = _payload(ordered, cfg, max_tokens=_request_limit(len(missing)))
        raw, info, calls, error, completed = "", {}, 0, "", {}
        can_retain = False
        try:
            raw, info = _request(payload, cfg, bounded_check, deadline)
            calls = 1
            if info.get("finish_reason") != "stop":
                raise ValueError("联合观察未完整结束，未补写或重试")
            if info.get("response_model") not in (None, "", cfg["vision_model"]):
                raise ValueError("联合观察响应模型与请求不一致")
            can_retain = True
            facts, quality = parse_wire(raw, expected)
            completed = {ident: _joined(facts, quality, ident) for ident in expected}
        except ValueError as exc:
            calls = max(calls, getattr(exc, "request_count", 0))
            error = str(exc)
            if can_retain:
                completed = _complete_wire_groups(raw, expected)
            failed = [key for i, key in enumerate(missing) if f"g{i}" not in completed]
            neutral_observer._flight_failed(flights, failed, exc)
            for key in failed:
                groups[key]["error"] = error
        except Exception as exc:
            neutral_observer._flight_failed(flights, missing, exc)
            exc.request_count = exc.model_requests = max(calls, getattr(exc, "request_count", 0))
            raise
        try:
            # Transport failures may already have exhausted the local deadline;
            # still preserve their diagnostic. A usable projection additionally
            # checks this deadline before any cache publication.
            bounded_check() if completed else check()
            for i, key in enumerate(missing):
                ident = f"g{i}"
                if ident not in completed:
                    continue
                observation = completed[ident]
                cache_key = groups[key]["variants"][payload["max_tokens"]]
                groups[key].update(observation=observation, cache_key=cache_key)
                flights[key].update(observation=deepcopy(observation), cache_key=cache_key)
                with suppress(OSError):
                    visual_quality._write_json(_cache_path(cache_key), {"version": VERSION, "key": cache_key,
                        "created_at": time.time(), "frame_sha256": groups[key]["hashes"], "observation": observation})
        except Exception as exc:
            neutral_observer._flight_failed(flights, missing, exc)
            exc.request_count = exc.model_requests = calls
            raise
        completion = {"batch_complete": not error and len(completed) == len(expected),
                      "retained_group_ids": list(completed),
                      "failed_group_ids": [ident for ident in expected if ident not in completed]}
        if info:
            info = info | completion
        saved = []
        if diagnostics_dir or error:
            target = Path(diagnostics_dir or paths.DATA / "diagnostics" / "pixel-observer") / (uuid4().hex + ".json")
            with suppress(OSError):
                visual_quality._write_json(target, {"version": VERSION, "task": TASK, "model_info": info,
                    "expected_groups": expected, "frame_sha256": {
                        f"g{i}": groups[key]["hashes"] for i, key in enumerate(missing)},
                    "raw_content": raw[:100000], "error": error, "request_count": calls, **completion})
                saved.append(str(target))
        return calls, saved, [info] if info else []


def observe(records, check, *, cfg=None, diagnostics_dir=None, max_seconds=None):
    """Return (neutral_by_shot, quality_by_shot, stats), with no approval.

    At most four records/five frames each. Known Qwen requests are partitioned
    into 4/2/1 groups before sending to avoid crossing its low input-price tier.
    Only pixels and local group/frame IDs are sent. A cache hit is a complete
    previously saved or shared joint group; singleflight_hits identifies sharing
    with another invocation. all_cache_hit means THIS invocation sent zero
    requests and every group reused valid joint facts, not that no peer paid.
    Cancellation, account and budget errors propagate with actual request_count.
    Other transport/schema failures return explicit unknown/uncertain results.
    """
    started, calls = time.monotonic(), 0
    original_check = check

    def check():
        try:
            original_check()
        except Exception as exc:
            # A callback is not a request. Transport/batch owners add their
            # requests once when propagating this cancellation or deadline.
            exc.request_count = exc.model_requests = 0
            raise
    if not records:
        return {}, {}, {"model_requests": 0, "request_count": 0, "cache_hits": 0,
                        "unique_groups": 0, "all_cache_hit": False, "elapsed_seconds": 0}
    # A four-group reply has twice the output allowance. Keep one bounded local
    # deadline (not a fresh timer per partition); the caller's check still owns
    # the earlier whole-job deadline and explicit shorter limits stay shorter.
    cfg = deepcopy(config.read() if cfg is None else cfg)
    if not all(cfg.get(key) for key in ("vision_model", "model_base_url", "model_api_key")):
        raise PixelObservationError("联合观察需要已配置视觉模型")
    if max_seconds is None:
        max_seconds = 60 if len(records) > GROUPS_PER_REQUEST and _supports_four(cfg) else 45
    if not 1 <= len(records) <= MAX_GROUPS or not 1 <= max_seconds <= 60:
        raise ValueError("联合观察每批最多4组，总时限须为1–60秒")
    groups, shots, files_by_shot, diagnostics, infos = {}, {}, {}, [], []
    for record in records:
        check()
        sid, files = record["shot"]["id"], record.get("frame_paths")
        if (not isinstance(sid, str) or not sid or sid in shots or not isinstance(files, (tuple, list))
                or not 1 <= len(files) <= MAX_FRAMES):
            raise ValueError("联合观察镜头编号重复或实际帧数量不在1–5之间")
        prepared = []
        for path in files:
            check()
            prepared.append(neutral_observer._frame(path))
        key, group = _group(cfg, prepared)
        shots[sid], files_by_shot[sid] = key, files
        if key not in groups:
            _read_group_cache(group)
            groups[key] = group
    pending = [key for key, group in groups.items() if group["observation"] is None]
    try:
        batches = _partition(groups, pending, cfg)
    except Exception as exc:
        exc.request_count = exc.model_requests = 0
        raise
    for keys in batches:
        try:
            check()
            added, saved, models = _observe_batch(groups, keys, cfg, check, started + max_seconds, diagnostics_dir)
        except TimeoutError as exc:
            calls += getattr(exc, "request_count", 0)
            for key in keys:
                groups[key]["error"] = str(exc)
            continue
        except Exception as exc:
            exc.request_count = exc.model_requests = calls + getattr(exc, "request_count", 0)
            raise
        calls += added
        diagnostics.extend(saved)
        infos.extend(models)
    facts_out, quality_out, failed = {}, {}, []
    for sid, key in shots.items():
        group, error = groups[key], groups[key].get("error", "")
        try:
            if [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files_by_shot[sid]] != group["hashes"]:
                error = "观察期间实际图片已改变，原结果不能用于当前图片"
        except OSError:
            error = "观察期间实际图片已不可读"
        if group["observation"] is not None and not error:
            facts, quality = parse_groups(json.dumps({"groups": [
                {"id": "current", **group["observation"]}]}, ensure_ascii=False), {"current": len(group["hashes"])})
            fact, assessed, status = facts["current"], visual_quality.assess(quality["current"]), "complete"
        else:
            error = error or "联合观察未完成"
            failed.append(sid)
            fact = {"scene_description": "", "frames": [], "error": error}
            assessed = {"version": visual_quality.VERSION, "representation": "uncertain",
                        "quality_status": "uncertain", "quality_reason": error, "frames": [],
                        "blocking_code": "", "representation_counts": {}, "obstruction_evidence": [],
                        "content_approved": False, "authenticity_assessed": False}
            status = "unknown"
        common = {"frame_sha256": group["hashes"], "observation_model": cfg["vision_model"],
                  "cached": group["cached"], "cache_key": group["cache_key"], "observation_status": status,
                  "content_approved": False}
        facts_out[sid], quality_out[sid] = fact | common, assessed | common
    try:
        check()
    except Exception as exc:
        exc.request_count = exc.model_requests = calls
        raise
    hits = sum(group["cached"] for group in groups.values())
    return facts_out, quality_out, {"model_requests": calls, "request_count": calls,
        "cache_hits": hits, "singleflight_hits": sum(bool(g.get("shared")) for g in groups.values()),
        "unique_groups": len(groups), "failed_shot_ids": failed,
        "all_cache_hit": bool(groups) and not failed and calls == 0 and hits == len(groups),
        "model": cfg["vision_model"], "model_info": infos, "task": TASK, "diagnostic_paths": diagnostics,
        "deadline_seconds": max_seconds,
        "elapsed_seconds": round(time.monotonic() - started, 3)}
