"""Durable model spending guard. Reserve before sending, including parallel calls.

CNY estimates use the peak price, not an invoice. Missing usage/timeouts keep the
reservation: a client disconnect does not prove the provider stopped billing.
No prompts, images, responses, or credentials are stored in this ledger.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from functools import wraps
import hashlib
import json
import sqlite3
import time
from threading import Condition, Event, Lock
from urllib.parse import urlparse
from uuid import uuid4

from . import config, paths

MICROS = 1_000_000
PRICING_DATE = "2026-10-09"
PRICING_URL = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
ALIYUN_PRICING_URL = "https://help.aliyun.com/zh/model-studio/qwen3-7-flash"
QWEN_FLASH_MODELS = {"qwen3.7-flash", "qwen3.7-flash-2026-07-15"}
_scope = ContextVar("model_cost_scope", default=("unassigned", "", "direct", None))
_schema_lock = Lock()
_settlements = Condition()
_active = set()
_cancel_check = ContextVar("model_cost_cancel_check", default=None)


class BudgetExceeded(RuntimeError):
    """Fatal to the job; candidate/retry handlers must not swallow this."""


class PricingUnavailable(RuntimeError):
    pass


@contextmanager
def scope(project_id, job_id="", stage="direct"):
    previous = _scope.get()
    stop = previous[3] if previous[:2] == (project_id, job_id) and previous[3] else Event()
    token = _scope.set((project_id or "unassigned", job_id, stage, stop))
    try:
        yield
    finally:
        _scope.reset(token)


def job(function):
    @wraps(function)
    def wrapped(task, *args, **kwargs):
        from . import media, store
        stop = args[0] if args else kwargs.get("stop")
        def check():
            if (stop is not None and stop.is_set()) or store.get_job(task["id"])["cancel"]:
                raise media.Cancelled()
        token = _cancel_check.set(check)
        try:
            with scope(task["project_id"], task["id"], task["kind"]):
                return function(task, *args, **kwargs)
        finally:
            _cancel_check.reset(token)
    return wrapped


def _day():
    # The same computer's calendar day is used by the worker and the local UI.
    return datetime.now().astimezone().date().isoformat()


def _micros(value):
    return int((Decimal(str(value)) * MICROS).to_integral_value(rounding=ROUND_CEILING))


def _provider(cfg):
    return (urlparse(cfg["model_base_url"]).hostname or "").lower()


def request_hash(endpoint, payload):
    """Identify identical requests without retaining text, media or credentials."""
    canonical = json.dumps({"endpoint": endpoint.rstrip("/"), "payload": payload},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@contextmanager
def _db():
    paths.prepare()
    con = sqlite3.connect(paths.DATA / "model-costs.sqlite3", timeout=30)
    con.row_factory = sqlite3.Row
    try:
        # SQLite's initial journal-mode change can fail immediately when many
        # first requests race, even with busy_timeout. Initialize it serially.
        with _schema_lock:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
            CREATE TABLE IF NOT EXISTS charges (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, job_id TEXT NOT NULL,
                stage TEXT NOT NULL, category TEXT NOT NULL, model TEXT NOT NULL,
                created REAL NOT NULL, day TEXT NOT NULL, state TEXT NOT NULL,
                reserved INTEGER NOT NULL, amount INTEGER NOT NULL,
                input_tokens INTEGER, cached_tokens INTEGER, output_tokens INTEGER,
                rates TEXT NOT NULL, http_status INTEGER
            );
            CREATE INDEX IF NOT EXISTS charges_project ON charges(project_id);
            CREATE INDEX IF NOT EXISTS charges_day ON charges(day);
            """)
            columns = {r[1] for r in con.execute("PRAGMA table_info(charges)")}
            additions = {
                "provider": "TEXT NOT NULL DEFAULT ''",
                "task": "TEXT NOT NULL DEFAULT ''",
                "request_hash": "TEXT NOT NULL DEFAULT ''",
                "cost_basis": "TEXT NOT NULL DEFAULT 'model_tokens'",
                "pricing_status": "TEXT NOT NULL DEFAULT 'known'",
                "unit": "TEXT NOT NULL DEFAULT 'tokens'",
                "usage_units": "INTEGER",
                "pricing_hash": "TEXT NOT NULL DEFAULT ''",
            }
            for name, definition in additions.items():
                if name not in columns:
                    con.execute(f"ALTER TABLE charges ADD COLUMN {name} {definition}")
            con.execute("CREATE INDEX IF NOT EXISTS charges_provider ON charges(provider)")
            con.execute("CREATE INDEX IF NOT EXISTS charges_request_hash ON charges(request_hash)")
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def project_headroom():
    """Advisory batch sizing only; request._reserve remains the atomic guard.

    A batch may fit after smaller responses settle. Include all charged and
    reserved amounts exactly as the guard does; never rewrite past spend.
    """
    limit = (config.DEFAULTS | config.read())["model_project_budget_cny"]
    if limit == "":
        return None
    pid = _scope.get()[0]
    with _db() as con:
        used = con.execute("SELECT COALESCE(SUM(amount),0) FROM charges WHERE project_id=?", (pid,)).fetchone()[0]
    return max(0, _micros(limit)-used)


def _rates(cfg, model, input_tokens=None):
    manual = [cfg.get(k, "") for k in ("cost_input_per_million", "cost_cached_per_million",
                                      "cost_output_per_million")]
    if any(manual):
        try:
            rates = tuple(Decimal(v) for v in manual)
            if not all(r.is_finite() and r >= 0 for r in rates) or rates[0] <= 0 or rates[2] <= 0:
                raise ValueError
            if rates[1] > rates[0]:
                raise ValueError
            return rates
        except (InvalidOperation, ValueError):
            raise PricingUnavailable("费用单价未填完整或无效，已停止模型请求；请检查服务与设置") from None
    if urlparse(cfg["model_base_url"]).hostname == "api.deepseek.com":
        if model in {"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}:
            return Decimal("2"), Decimal("0.04"), Decimal("8")
        if model == "deepseek-v4-pro":
            return Decimal("9"), Decimal("0.30"), Decimal("27")
    if _provider(cfg) == "dashscope.aliyuncs.com" and model in QWEN_FLASH_MODELS:
        # Reserve at the conservative input bound's tier; settle at the actual
        # prompt tier. A tier applies to the entire request, including output.
        count = input_tokens if input_tokens is not None else 1_000_000
        if count <= 32_000:
            return Decimal("0.2"), Decimal("0.04"), Decimal("0.8")
        if count <= 256_000:
            return Decimal("0.6"), Decimal("0.12"), Decimal("2.4")
        return Decimal("1.2"), Decimal("0.24"), Decimal("4.8")
    if (_provider(cfg) == "dashscope.aliyuncs.com"
            and model in {"qwen3.7-plus", "qwen3.7-plus-2026-05-26"}):
        # https://help.aliyun.com/zh/model-studio/qwen3-7-plus (Beijing)
        if input_tokens is not None and input_tokens <= 256_000:
            return Decimal("2"), Decimal("0.4"), Decimal("8")
        return Decimal("6"), Decimal("1.2"), Decimal("24")
    if _provider(cfg) == "dashscope.aliyuncs.com" and model in {"qwen-turbo", "qwen-turbo-latest"}:
        # Beijing, non-thinking; https://help.aliyun.com/zh/model-studio/qwen-turbo
        # Turbo is cheaper only for output-heavy requests, not a blanket upgrade.
        return Decimal("0.3"), Decimal("0.06"), Decimal("0.6")
    raise PricingUnavailable("当前接口或模型没有已知单价，已停止付费请求；请在服务与设置填写人民币单价")


def _input_bound(cfg, payload):
    # UTF-8 bytes bound byte-level text tokens conservatively. Do not count an
    # inline image's base64 bytes as text. DeepSeek documents <=1024/image:
    # https://api-docs.deepseek.com/zh-cn/guides/vision/
    count = 256
    format_spec = payload.get("response_format", {})
    if isinstance(format_spec, dict) and format_spec.get("type") == "json_schema":
        # The server consumes the schema as well as messages. Its complete
        # UTF-8 bytes participate in tier selection and reservation.
        count += len(json.dumps(format_spec, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    for message in payload["messages"]:
        count += 128 + len(message.get("role", "").encode("utf-8"))
        content = message.get("content", "")
        if isinstance(content, str):
            count += len(content.encode("utf-8"))
            continue
        for block in content:
            if block.get("type") == "image_url":
                known = (urlparse(cfg["model_base_url"]).hostname == "api.deepseek.com"
                         and payload["model"] in {"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"})
                qwen = _provider(cfg) == "dashscope.aliyuncs.com" and payload["model"] in QWEN_FLASH_MODELS
                allowance = int(cfg.get("cost_image_tokens", "0"))
                if qwen:
                    # Official Qwen3.7: 32x32 pixels/token; default 2,621,440
                    # pixels/image, high-resolution mode at most 16,384 tokens.
                    # https://help.aliyun.com/zh/model-studio/vision
                    details = block.get("image_url", {})
                    high = any(part.get("vl_high_resolution_images") for part in (payload, block, details)
                               if isinstance(part, dict))
                    pixels = block.get("max_pixels", 2_621_440)
                    if type(pixels) is not int or not 0 < pixels <= 16_777_216:
                        raise PricingUnavailable("图片像素上限无效，已停止付费请求")
                    allowance = max(allowance, 16_384 if high else (pixels+1023)//1024)
                if not known and allowance <= 0:
                    raise PricingUnavailable("该视觉模型需要设置每图 token 预留上限，已停止付费请求")
                count += max(1024 if known else 0, allowance) + 128
            elif block.get("type") == "text":
                count += len(block.get("text", "").encode("utf-8")) + 32
            else:
                raise PricingUnavailable("无法估算该输入类型的费用，已停止付费请求")
    return count


def _amount(rates, input_tokens, output_tokens, cached_tokens=0):
    # A rate in CNY/million multiplied by tokens is already micro-CNY.
    cost = (input_tokens-cached_tokens)*rates[0] + cached_tokens*rates[1] + output_tokens*rates[2]
    return int(cost.to_integral_value(rounding=ROUND_CEILING))


def _usage(body):
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    prompt, output = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(type(v) is not int or v < 0 for v in (prompt, output)):
        return None
    details = usage.get("prompt_tokens_details") or {}
    cached = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0) if isinstance(details, dict) else 0)
    if type(cached) is not int or not 0 <= cached <= prompt:
        cached = 0
    # completion_tokens includes reasoning_tokens; never charge reasoning twice.
    return prompt, output, cached


class request:
    def __init__(self, cfg, payload, category, *, task=None, check=None):
        self.cfg, self.payload, self.category = cfg, payload, category
        self.task = task or category
        self.id, self.sent, self.settled = uuid4().hex, False, False
        self.rates = ()
        self.provider = _provider(cfg)
        self.request_hash = request_hash(cfg["model_base_url"].rstrip("/") + "/chat/completions", payload)
        self.cost_basis, self.pricing_status = "model_tokens", "known"
        self.unit, self.usage_units, self.pricing_hash = "tokens", None, ""
        self.check = check or _cancel_check.get()

    def __enter__(self):
        pid, jid, stage, stopped = _scope.get()
        if stopped is not None and stopped.is_set():
            raise BudgetExceeded("本次任务已触及模型预算，已停止后续请求；已保存进度，重试不会清零费用")
        output = self.payload.get("max_tokens")
        if type(output) is not int or output <= 0:
            raise PricingUnavailable("模型请求缺少输出上限，已停止付费请求")
        if self.provider == "dashscope.aliyuncs.com" and self.payload.get("enable_thinking"):
            thinking = self.payload.get("thinking_budget")
            if type(thinking) is not int or thinking <= 0:
                raise PricingUnavailable("思考请求缺少独立输出上限，已停止付费请求")
            # Qwen's max_tokens excludes reasoning; reserve both components.
            output += thinking
        bound = _input_bound(self.cfg, self.payload)
        self.rates = _rates(self.cfg, self.payload["model"], bound)
        reserve = _amount(self.rates, bound, output)
        return self._reserve(reserve)

    def _insert(self, con, reserve):
        pid, jid, stage, _ = _scope.get()
        con.execute("INSERT INTO charges (id,project_id,job_id,stage,category,model,created,day,state,reserved,amount,rates,provider,task,request_hash,cost_basis,pricing_status,unit,usage_units,pricing_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (self.id, pid, jid, stage, self.category, self.payload["model"], time.time(), _day(),
                     "reserved", reserve, reserve, json.dumps([str(r) for r in self.rates]), self.provider,
                     self.task, self.request_hash, self.cost_basis, self.pricing_status, self.unit,
                     self.usage_units, self.pricing_hash))
        with _settlements:
            _active.add(self.id)

    def _reserve(self, reserve):
        pid, jid, stage, stopped = _scope.get()
        deadline = time.monotonic()+185  # longest bounded model request is 180s
        while True:
            if self.check:
                self.check()
            if stopped is not None and stopped.is_set():
                raise BudgetExceeded("本次任务已触及模型预算，后续请求已停止；已保存进度")
            day = _day()
            # Re-read limits after waits: running jobs respect a lowered cap.
            limits = config.DEFAULTS | config.read()
            waiting = False
            with _db() as con:
                con.execute("BEGIN IMMEDIATE")
                rows = con.execute("SELECT id,project_id,job_id,day,state,amount,provider,category FROM charges "
                                   "WHERE project_id=? OR day=? OR provider=? OR provider='' OR state IN ('reserved','unknown')",
                                   (pid, day, self.provider)).fetchall()
                with _settlements:
                    active = _active.copy()
                # Speech is priced per character and grows with the script, so it has its own daily
                # cap; the model caps cover planning, rewriting and picture checks only.
                speech = self.category == "tts"
                rows = [r for r in rows if (r["category"] == "tts") == speech]
                today = [r for r in rows if r["day"] == day or r["state"] in {"reserved", "unknown"}]
                caps = ((("今日配音", today, limits["tts_daily_budget_cny"]),) if speech else (
                    ("本次任务", [r for r in rows if r["job_id"] == jid], limits["model_job_budget_cny"] if jid else ""),
                    ("项目累计", [r for r in rows if r["project_id"] == pid], limits["model_project_budget_cny"]),
                    ("今日", today, limits["model_daily_budget_cny"]),
                    ("当前服务累计", [r for r in rows if r["provider"] in {self.provider, ""}], limits["model_provider_budget_cny"]),
                ))
                for name, subset, limit in caps:
                    if limit == "":
                        continue
                    used = sum(r["amount"] for r in subset)
                    if used + reserve <= _micros(limit):
                        continue
                    live_reserves = sum(r["amount"] for r in subset if r["id"] in active and r["state"] == "reserved")
                    # Only wait on this process's genuinely in-flight calls.
                    # Crash leftovers and unknown usage never free themselves.
                    if live_reserves and used-live_reserves+reserve <= _micros(limit) and time.monotonic()<deadline:
                        waiting = True
                        continue
                    if stopped is not None:
                        stopped.set()
                    raise BudgetExceeded(f"{name}{'' if speech else '模型'}预算 ¥{Decimal(limit):.2f}：已记账/预留 ¥{used/MICROS:.4f}，"
                                         f"下一请求需预留 ¥{reserve/MICROS:.4f}，已在发送前停止。已保存进度；重试不会清零费用。")
                if not waiting:
                    self._insert(con, reserve)
            if not waiting:
                return self
            with _settlements:
                _settlements.wait(timeout=.2)

    def mark_sent(self):
        self.sent = True

    def response(self, response):
        """Account before caller parses choices or raises on an HTTP error."""
        try:
            body = response.json()
        except ValueError:
            body = None
        usage = _usage(body)
        code = response.status_code
        if usage is not None:
            prompt, output, cached = usage
            self.rates = _rates(self.cfg, self.payload["model"], prompt)
            with _db() as con:
                con.execute("UPDATE charges SET state='settled',amount=?,input_tokens=?,cached_tokens=?,output_tokens=?,http_status=?,rates=? WHERE id=?",
                            (_amount(self.rates, prompt, output, cached), prompt, cached, output, code,
                             json.dumps([str(r) for r in self.rates]), self.id))
            self.settled = True
        elif code in {400, 401, 402, 403, 404, 413, 422, 429}:
            self._close("rejected", 0, code)
        else:
            self._close("unknown", None, code)
        return body

    def _close(self, state, amount=None, code=None):
        with _db() as con:
            con.execute("UPDATE charges SET state=?,amount=COALESCE(?,amount),http_status=? WHERE id=?",
                        (state, amount, code, self.id))
        self.settled = True

    def __exit__(self, exc_type, exc, traceback):
        try:
            if not self.settled:
                self._close("unknown" if self.sent else "not_sent", None if self.sent else 0)
        finally:
            with _settlements:
                _active.discard(self.id)
                _settlements.notify_all()


def summary(project_id=None):
    cfg = config.DEFAULTS | config.read()
    day = _day()
    def totals(con, where, args):
        row = con.execute("SELECT COUNT(*) AS requests, COALESCE(SUM(amount),0) AS total, "
                          "COALESCE(SUM(CASE WHEN state='settled' THEN amount ELSE 0 END),0) AS settled, "
                          "COALESCE(SUM(CASE WHEN state IN ('reserved','unknown') THEN amount ELSE 0 END),0) AS reserved, "
                          "COALESCE(SUM(CASE WHEN state IN ('reserved','unknown') THEN 1 ELSE 0 END),0) AS unresolved, "
                          "COALESCE(SUM(CASE WHEN pricing_status='unpriced' AND state NOT IN ('not_sent','rejected') THEN 1 ELSE 0 END),0) AS unpriced "
                          "FROM charges WHERE "+where, args).fetchone()
        return {"requests": row["requests"], "estimated_cny": row["settled"]/MICROS,
                "reserved_cny": row["reserved"]/MICROS, "total_cny": row["total"]/MICROS,
                "unresolved_requests": row["unresolved"], "unpriced_requests": row["unpriced"],
                "cost_complete": not bool(row["unpriced"]),
                "upper_bound_cny": None if row["unpriced"] else row["total"]/MICROS,
                "complete_total_cny": None if row["unpriced"] else row["total"]/MICROS}
    with _db() as con:
        first = con.execute("SELECT MIN(created) FROM charges").fetchone()[0]
        daily = totals(con, "day=? OR state IN ('reserved','unknown','unpriced')", (day,))
        provider = totals(con, "provider=? OR provider=''", (_provider(cfg),))
        project = totals(con, "project_id=?", (project_id,)) if project_id else None
        # Speech has its own daily cap: show model and voice money side by side against their own limits.
        open_states = "state IN ('reserved','unknown','unpriced')"
        def split(where, args):
            rows = con.execute("SELECT category='tts' AS speech, COALESCE(SUM(amount),0) AS total FROM charges WHERE ("
                               + where + ") GROUP BY category='tts'", args).fetchall()
            found = {bool(r["speech"]): r["total"] / MICROS for r in rows}
            return found.get(False, 0.0), found.get(True, 0.0)
        model_day, speech_day = split("day=? OR " + open_states, (day,))
        model_project, speech_project = split("project_id=?", (project_id,)) if project_id else (None, None)
        where, args = ("project_id=?", (project_id,)) if project_id else ("day=? OR state IN ('reserved','unknown','unpriced')", (day,))
        groups = con.execute("SELECT stage,category,model,COUNT(*) AS requests,SUM(amount) AS amount FROM charges WHERE "+where+" GROUP BY stage,category,model", args).fetchall()
        tasks = con.execute("SELECT stage,category,COALESCE(NULLIF(task,''),category) AS task,model,provider,pricing_status,COUNT(*) AS requests,SUM(amount) AS amount FROM charges WHERE "+where+" GROUP BY stage,category,task,model,provider,pricing_status", args).fetchall()
    daily["limit_cny"] = float(cfg["model_daily_budget_cny"])
    provider.update(host=_provider(cfg), limit_cny=(float(cfg["model_provider_budget_cny"])
                    if cfg["model_provider_budget_cny"] else None))
    if project is not None:
        project["limit_cny"] = float(cfg["model_project_budget_cny"])
    split_totals = {"model": {"daily_cny": model_day, "daily_limit_cny": float(cfg["model_daily_budget_cny"]),
                              "project_cny": model_project, "project_limit_cny": float(cfg["model_project_budget_cny"])},
                    "speech": {"daily_cny": speech_day, "daily_limit_cny": float(cfg["tts_daily_budget_cny"]),
                               "project_cny": speech_project}}
    return {"day": day, "day_basis": "本机日期", "daily": daily, "project": project, "provider": provider, **split_totals,
            "categories": [{"stage": r["stage"], "category": r["category"], "model": r["model"], "requests": r["requests"],
                            "total_cny": r["amount"]/MICROS} for r in groups],
            "tasks": [{"stage": r["stage"], "category": r["category"], "task": r["task"], "model": r["model"],
                       "provider": r["provider"], "pricing_status": r["pricing_status"], "requests": r["requests"],
                       "total_cny": None if r["pricing_status"] == "unpriced" else r["amount"]/MICROS} for r in tasks],
            "cost_complete": (project or daily)["cost_complete"],
            "complete_total_cny": (project or daily)["complete_total_cny"],
            "upper_bound_cny": (project or daily)["upper_bound_cny"],
            "first_recorded_at": first, "pricing_date": PRICING_DATE,
            "pricing_url": ALIYUN_PRICING_URL if _provider(cfg) == "dashscope.aliyuncs.com" else PRICING_URL,
            "note": "模型与已定价配音的保守估算，不是平台账单；未知用量继续预留。未核明单价的配音另行计数，此时 total_cny 只是已记金额下界，完整总额和上界未知。服务累计跨项目、跨日、跨密钥，并保守计入旧版未标注服务的记录。不含账本启用前的配音、付费素材、电费及其他客户端费用；费用完整性仅针对本账本覆盖的请求。"}
