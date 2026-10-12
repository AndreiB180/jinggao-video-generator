"""Public video metadata catalogue. Bulk indexing never opens media streams.

Topics describe the retrieval route, not a visual or identity verification.
Only ordinary public search pages are read; a provider challenge stops that lane.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
import time
from .execution import ThreadPoolExecutor

import httpx

from . import catalog_taxonomy, library, paths, sources, stock_queries, stock_sources, store, video_sources

PROVIDERS = {"tencent": "腾讯视频", "bilibili": "B站", "cctv": "央视", "chinanews": "中新网",
             "iqiyi": "爱奇艺", "sohu": "搜狐", "youku": "优酷", "baidu_video": "百度视频",
             "pexels": "Pexels", "pixabay": "Pixabay", "coverr": "Coverr",
             "commons": "Wikimedia Commons", "nasa": "NASA"}
# Interleave these routes so a partial build already covers every subject area.
TOPICS = {
    "生活家居": "家庭生活|家居收纳|装修改造|超市购物|洗衣清洁|小区生活|租房生活|厨房用品|生活妙招|老人生活|夜市摆摊|农村生活",
    "美食烹饪": "美食制作|家常菜|面食制作|街头小吃|烘焙蛋糕|火锅|烧烤|咖啡制作|包饺子|早餐|海鲜烹饪|地方美食",
    "城市街景": "城市航拍|城市夜景|街头人群|商业街|城市交通|县城街景|北京街景|上海街景|广州街景|重庆街景|老街|高楼建设",
    "家庭情感": "亲子陪伴|婚礼现场|夫妻生活|老人养老|家庭团聚|照顾孩子|父母子女|毕业离别|年轻人独居|社区活动|家庭教育|邻里互助",
    "职场劳动": "工人工作|办公室办公|快递员送货|外卖骑手|建筑工地|工厂流水线|农民工|程序员工作|招聘现场|地铁通勤|创业青年|环卫工人",
    "教育校园": "学校课堂|高考考场|大学校园|学生运动|职业教育|幼儿园|图书馆学习|实验室科研|开学现场|乡村教师|毕业典礼|校园生活",
    "社会民生": "社区养老|居民办事|就业招聘|保障住房|乡村振兴|社会救助|无障碍设施|消费维权|菜市场|公共服务|医疗保障|人口老龄化",
    "时政公共事务": "全国两会|外交部发布会|联合国会议|基层治理|政务服务|城市治理|国际会议|公共政策|中国外交|国家发展|乡村治理|新闻发布会",
    "历史档案": "历史影像|抗战纪录片|新中国成立|改革开放|清朝历史|汉代历史|唐朝历史|民国影像|古代战争|丝绸之路|历史人物|考古发掘",
    "传统文化": "非遗手艺|传统节日|书法创作|中国戏曲|陶瓷制作|中国古建筑|传统服饰|民间艺术|博物馆文物|古籍修复|龙舟比赛|舞龙舞狮",
    "商业财经": "企业生产|股票交易|银行业务|实体店经营|电商直播|商场消费|房地产市场|工业经济|企业家访谈|小微企业|金融市场|跨境贸易",
    "科技互联网": "人工智能|机器人演示|芯片制造|手机评测|数据中心|互联网发展|无人驾驶|量子计算|科技创新|智能家居|无人机演示|电子产品",
    "工业制造": "汽车工厂|机械加工|钢铁冶炼|船舶制造|港口装卸|工业机器人|高铁制造|电力生产|精密制造|矿山开采|自动化生产|大国工匠",
    "农业乡村": "温室种植|水稻收割|果园采摘|农业机械|乡村田野|养殖场|农田灌溉|茶叶采摘|渔民捕鱼|丰收现场|农村集市|现代农业",
    "医疗健康": "医院诊疗|医生科普|康复训练|全民健身|营养饮食|睡眠健康|体检科普|急救演示|护理工作|中医药文化|老年健康|公共卫生",
    "自然环境": "森林航拍|大海波浪|雪山风景|河流湖泊|沙漠戈壁|野生动物|环境保护|气候变化|自然保护区|湿地生态|垃圾分类|清洁能源",
    "动物宠物": "猫咪日常|狗狗训练|熊猫生活|动物救助|宠物护理|鸟类观察|海洋动物|动物园|牧场动物|野生动物保护|小动物|宠物医院",
    "旅游地理": "旅行风景|乡村旅游|西藏风景|新疆风景|云南旅游|海边度假|登山徒步|古镇旅游|地理科普|城市旅行|高原湖泊|世界遗产",
    "交通汽车": "高速公路|高铁运行|机场航班|新能源汽车|汽车驾驶|公共交通|物流运输|轮船航行|大桥航拍|铁路运输|交通安全|自行车骑行",
    "体育赛事": "乒乓球比赛|足球比赛|篮球比赛|游泳比赛|田径比赛|羽毛球比赛|排球比赛|滑雪比赛|马龙乒乓球|孙颖莎比赛|樊振东比赛|全红婵跳水",
    "娱乐文艺": "电影幕后|演员访谈|音乐演出|舞蹈表演|相声表演|综艺现场|剧组拍摄|戏剧演出|歌手采访|舞台排练|文艺演出|艺术展览",
    "人物访谈": "科学家访谈|运动员采访|企业家采访|作家访谈|袁隆平纪录片|屠呦呦访谈|钟南山采访|航天员访谈|劳动模范|雷军采访|董宇辉访谈|普通人故事",
    "航天国防": "火箭发射|空间站|航天员训练|卫星发射|航天工程|航空展览|国防教育|军舰航行|飞行训练|军事博物馆|探月工程|航天科普",
    "灾害救援": "消防救援|防汛救灾|地震救援|应急演练|安全生产|森林防火|海上救援|交通救援|台风防御|暴雨天气|防灾科普|救援队",
}
TOPICS.update(catalog_taxonomy.EXTRA_TOPICS)
PAGE_TTL = 7 * 86400
REQUEST_INTERVAL = 1.2


def routes():
    from .platform_videos import bilibili_search, tencent_search
    from .publisher_videos import chinanews_search
    from .portal_videos import iqiyi_search, sohu_search, youku_search
    from .baidu_videos import baidu_video_search
    public = {"tencent": tencent_search, "bilibili": bilibili_search,
            "cctv": video_sources.cctv_search, "chinanews": chinanews_search,
            "iqiyi": iqiyi_search, "sohu": sohu_search, "youku": youku_search,
            "baidu_video": baidu_video_search}
    for provider in stock_sources.registry():
        public[provider] = lambda query, page=1, provider=provider: stock_sources.search_items(provider, query, page)
    return public


def source_access(platforms=None):
    """Report configuration separately from actual, persisted query outcomes."""
    with library.connection() as con:
        if platforms is None:
            platforms = {r[0]: r[1] for r in con.execute(
                "SELECT provider,COUNT(*) FROM source_catalog GROUP BY provider")}
        recent = {}
        for row in con.execute("SELECT provider,status,updated,error FROM index_pages ORDER BY updated DESC"):
            recent.setdefault(row["provider"], dict(row))
    result = []
    for provider, entry in stock_sources.registry().items():
        row = {k: entry.get(k) for k in ("name", "status", "reason", "website", "configuration_status",
                                        "index_mode", "bulk_index_allowed", "index_policy")}
        row.update(provider=provider, indexed_sources=platforms.get(provider, 0), last_query_status="unchecked")
        last = recent.get(provider)
        if last:
            row.update(last_query_status=last["status"], last_query_at=last["updated"])
            if row["status"] == "configured":
                if last["status"] == "failed":
                    # Historical error text may contain an authenticated URL.
                    # Expose only a known status code and the safe reason.
                    raw_error = str(last["error"])
                    code = re.search(r"\bHTTP\s+([45]\d{2})\b", raw_error, re.I)
                    safe_reason = stock_sources._failure(ValueError(raw_error))[1]
                    row.update(status="unavailable", reason="最近一次索引查询失败" +
                               (f"（HTTP {code[1]}）" if code else "") + "：" + safe_reason)
                else:
                    row["reason"] = "最近一次索引查询成功；条目仍需检查视频流和对应画面"
        if row["index_policy"] and row["index_policy"] not in row["reason"]:
            row["reason"] += "；" + row["index_policy"]
        result.append(row)
    return result


def overview():
    with library.connection() as con:
        platforms = {r[0]: r[1] for r in con.execute(
            "SELECT provider,COUNT(*) FROM source_catalog GROUP BY provider")}
        topics = {r[0]: r[1] for r in con.execute(
            "SELECT topic,COUNT(DISTINCT source_id) FROM source_topics GROUP BY topic")}
        states = {r[0]: r[1] for r in con.execute("SELECT state,COUNT(*) FROM source_catalog GROUP BY state")}
        cached = con.execute("SELECT COUNT(DISTINCT source_id) FROM clips").fetchone()[0]
        entity_counts = {r[0]: r[1] for r in con.execute("SELECT e.kind,COUNT(DISTINCT e.id) FROM catalog_entities e JOIN source_entities se ON se.entity_id=e.id GROUP BY e.kind")}
        planned = {r[0]: r[1] for r in con.execute("SELECT kind,COUNT(*) FROM catalog_entities GROUP BY kind")}
    return {"providers": platforms, "topics": {k: topics.get(k, 0) for k in TOPICS},
            "states": states, "cached_sources": cached, "entities": entity_counts, "planned_entities": planned,
            "source_access": source_access(platforms)}


def prepare_entities():
    """Backfill metadata mentions once per taxonomy revision, including seed imports."""
    definitions = catalog_taxonomy.entities()
    fingerprint = hashlib.sha256(("metadata-mentions-v3"+json.dumps(definitions, sort_keys=True)).encode()).hexdigest()
    marker = library.root() / "entity-taxonomy.sha256"
    if marker.exists() and marker.read_text() == fingerprint:
        return
    with library.connection() as con:
        con.execute("DELETE FROM catalog_entities")
        con.executemany("INSERT OR REPLACE INTO catalog_entities VALUES (?,?,?,?,?)",
            [(e["id"], e["name"], e["kind"], e["domain"], json.dumps(e["aliases"], ensure_ascii=False)) for e in definitions.values()])
        con.execute("DELETE FROM source_entities")
        for row in con.execute("SELECT id,body FROM sources").fetchall():
            body = json.loads(row["body"])
            con.executemany("INSERT OR IGNORE INTO source_entities VALUES (?,?,?)", [(row["id"], eid, mention)
                for eid, mention in catalog_taxonomy.mentions(body.get("title", "")+" "+body.get("description", ""))])
    marker.write_text(fingerprint, "utf-8")


def entity_search(query="", kind="", limit=100):
    with library.connection() as con:
        rows = con.execute("""SELECT e.*,COUNT(DISTINCT se.source_id) AS sources FROM catalog_entities e
            LEFT JOIN source_entities se ON se.entity_id=e.id
            WHERE (?='' OR e.kind=?) AND (?='' OR instr(e.name,?)>0 OR instr(e.aliases,?)>0)
            GROUP BY e.id ORDER BY sources DESC,e.name LIMIT ?""", (kind,kind,query,query,query,min(500,limit))).fetchall()
    return [dict(row) | {"aliases": json.loads(row["aliases"]), "evidence": "metadata_mention"} for row in rows]


def search(query="", *, topic="", provider="", entity="", limit=40, offset=0):
    """BM25 before LIMIT, so old but relevant entries remain retrievable at scale."""
    tokens = sorted(sources.tokens(query))[:60]
    # Exact canonical names and explicit aliases share the same mention index.
    # These are retrieval synonyms only; source text/evidence is never rewritten.
    exact = [e for e in catalog_taxonomy.entities().values()
             if query.strip() in [e["name"],*e["aliases"]]] if query.strip() else []
    if not entity and len(exact)==1:
        entity,tokens = exact[0]["id"],[]
    profile = stock_queries.query_profile(query)
    if not entity and not exact and profile["complete"]:
        tokens = sorted(set(tokens) | sources.tokens(" ".join(profile["queries"])))[:80]
    if query.strip() and not tokens and not entity:
        return {"items": [], "total": 0}
    params, conditions, order_params = [], [], []
    join = "FROM source_catalog sc JOIN sources s ON s.id=sc.source_id"
    order = "s.updated DESC,s.id"
    if tokens:
        join += " JOIN source_terms ON source_terms.rowid=sc.rowid"
        conditions.append("source_terms MATCH ?")
        params.append(" OR ".join('"'+t.replace('"', '""')+'"' for t in tokens))
        order = "CASE WHEN instr(json_extract(s.body,'$.title'),?)>0 THEN 0 ELSE 1 END,bm25(source_terms),s.id"
        order_params.append(query.strip())
    elif query.strip() and entity:
        order = "CASE WHEN instr(json_extract(s.body,'$.title'),?)>0 THEN 0 ELSE 1 END,s.updated DESC,s.id"
        order_params.append(query.strip())
    if topic:
        conditions.append("EXISTS (SELECT 1 FROM source_topics st WHERE st.source_id=s.id AND st.topic=?)")
        params.append(topic)
    if provider:
        conditions.append("sc.provider=?")
        params.append(provider)
    if entity:
        conditions.append("EXISTS (SELECT 1 FROM source_entities se WHERE se.source_id=s.id AND se.entity_id=?)")
        params.append(entity)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    with library.connection() as con:
        total = con.execute("SELECT COUNT(*) " + join + where, params).fetchone()[0]
        rows = con.execute("SELECT s.id,s.body,sc.state,sc.checked,sc.reason " + join + where +
            f" ORDER BY {order} LIMIT ? OFFSET ?",
            params + order_params + [min(200, max(1, limit)), max(0, offset)]).fetchall()
    result = []
    for row in rows:
        body = json.loads(row["body"])
        try:
            library.source_url(body["source_url"])
        except ValueError:
            continue
        result.append(body | {"source_id": row["id"], "index_state": row["state"],
                            "checked_at": row["checked"], "index_reason": row["reason"]})
    return {"items": result, "total": total}


def recall(query, *, limit=40, channel_names=None):
    tokens = sources.tokens(query)
    exact = [e for e in catalog_taxonomy.entities().values() if query.strip() in [e["name"],*e["aliases"]]]
    aliases = [query.strip()]
    if len(exact)==1:
        aliases += [exact[0]["name"],*exact[0]["aliases"]]
    result = []
    candidates = {r["source_id"]: r for r in search(query, limit=200)["items"]}
    # A large Chinese catalogue must not consume every result slot before the
    # equivalent English stock titles can be considered. Keep a bounded lane
    # for each usable stock source; the same relevance checks still apply.
    for provider, access in stock_sources.registry().items():
        if access["status"] == "configured" and (channel_names is None or PROVIDERS[provider] in channel_names):
            candidates.update({r["source_id"]: r for r in search(query, provider=provider, limit=16)["items"]})
    for row in candidates.values():
        name = "哔哩哔哩" if row.get("provider") == "bilibili" else PROVIDERS.get(row.get("provider"))
        if channel_names is not None and name not in channel_names:
            continue
        text = " ".join(str(row.get(k) or "") for k in ("title", "description", "publisher"))
        translated = stock_queries.coverage(query, text) if row.get("provider") in stock_sources.PROVIDERS else 0
        if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", query.strip()) and not any(a in text for a in aliases) and translated < .5:
            continue
        coverage = max(translated, len(tokens & sources.tokens(text)) / max(1, len(tokens)))
        if len(exact)==1 and any(a in text for a in aliases):
            coverage = 1.0
        if coverage < .5:
            continue
        if row["index_state"] in ("blocked", "unsupported", "preview_only") and time.time()-row["checked_at"] < 86400:
            continue
        # A historical inspection is only a hint. Never reuse expiring stream URLs or identity approvals.
        c = video_sources._candidate(row.get("provider", "web_video"), row["title"],
                                      row["source_url"], row.get("thumbnail_url", ""))
        c.update(row, from_source_index=True, index_coverage=coverage)
        result.append(c)
    def priority(c):
        date = str(c.get("published_at", ""))
        year = re.search(r"(?:19|20)\d{2}", c["source_url"]) if c["provider"] == "chinanews" else None
        # Old ChinaNews pages often still have titles/covers but no supported
        # video player. Prefer newer originals without discarding history queries.
        old_page = bool(year and int(year[0]) < 2017)
        return (-c["index_coverage"], c["index_state"] != "available", old_page,
                c["provider"] == "cctv", not bool(re.match(r"20(?:2[0-9]|3[0-9])", date)))
    result.sort(key=priority)
    return result[:limit]


def record_check(candidate):
    """Record observed availability without persisting private/signed stream URLs."""
    try:
        sid, _ = library.source_record(candidate)
        state = candidate.get("download_status", "unchecked")
        if state not in ("available", "preview_only", "blocked", "unsupported"):
            return
        with library.connection() as con:
            con.execute("UPDATE source_catalog SET state=?,checked=?,reason=? WHERE source_id=?",
                (state, time.time(), str(candidate.get("download_reason") or "")[:500], sid))
    except (ValueError, OSError):
        pass


def export_seed(destination):
    """Export only publicly collected metadata, never clips, accounts or project data."""
    with library.connection() as con:
        rows = con.execute("SELECT id,body FROM sources WHERE id IN (SELECT source_id FROM source_topics) ORDER BY id").fetchall()
        tags = {}
        for sid, topic, query in con.execute("SELECT source_id,topic,query FROM source_topics ORDER BY topic,query"):
            tags.setdefault(sid, []).append([topic, query])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8") as f:
        for row in rows:
            body = json.loads(row["body"])
            # Reconstruct the public publisher-search evidence on import. This
            # whitelist deliberately excludes project context and frame/person reviews.
            public = {k: body[k] for k in ("title", "description", "source_url", "thumbnail_url", "provider",
                      "publisher", "publisher_url", "published_at", "duration", "video_id", "license", "license_url",
                      "author", "collection_url") if k in body}
            f.write(json.dumps({"source": public, "topics": tags[row["id"]]}, ensure_ascii=False)+"\n")
    return len(rows)


def install_seed(seed=None):
    """Install a portable metadata seed once on a new/updated Windows installation."""
    seed = seed or paths.ROOT / "assets" / "source-index.jsonl.gz"
    if not seed.is_file():
        return 0
    fingerprint = hashlib.sha256(seed.read_bytes()).hexdigest()
    marker = library.root() / "source-seed.sha256"
    if marker.exists() and marker.read_text().strip() == fingerprint:
        return 0
    entries = []
    with gzip.open(seed, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 501000 or len(line) > 20000:
                raise ValueError("来源索引种子超过限定大小")
            row = json.loads(line)
            body = row["source"]
            library.source_url(body["source_url"])
            if body.get("provider") not in PROVIDERS:
                raise ValueError("来源索引种子包含未知平台")
            c = video_sources._candidate(body["provider"], body["title"], body["source_url"], body.get("thumbnail_url", ""))
            c.update({k: body[k] for k in library.INDEX_FIELDS if k in body})
            c["source_verified"] = True
            c["source_evidence"] = video_sources._evidence(c["source_url"], c["title"], c.get("description", ""),
                c.get("publisher", ""), c.get("publisher_url", ""), c.get("published_at", ""), "publisher_search")
            topics = [(t, q) for t, q in row.get("topics", []) if t in TOPICS and isinstance(q, str) and len(q) <= 100]
            entries.append((c, topics))
    added = library.remember_sources([c for c, _ in entries])
    prepare_entities()
    with library.connection() as con:
        for c, topics in entries:
            sid, _ = library.source_record(c)
            con.executemany("INSERT OR IGNORE INTO source_topics VALUES (?,?,?)", [(sid, t, q) for t, q in topics])
    marker.write_text(fingerprint, "utf-8")
    return added


def run(pid, options, check, progress):
    prepare_entities()
    began = time.monotonic()
    target = int(options.get("index_target", 10000))
    chosen = options.get("index_topics") or list(TOPICS)
    selected_providers = list(dict.fromkeys(options.get("index_providers") or list(PROVIDERS)))
    if not set(chosen) <= set(TOPICS) or not set(selected_providers) <= set(PROVIDERS):
        raise ValueError("未知索引主题或来源")
    seeds = {t: TOPICS[t].split("|") for t in chosen}
    scene_tasks = [(topic, queries[i]) for i in range(max(map(len, seeds.values())))
                   for topic, queries in seeds.items() if i < len(queries)]
    entity_tasks = [(topic, query) for topic, query, _ in catalog_taxonomy.queries()
                    if topic in chosen] if options.get("index_entities", True) else []
    # Every pass alternates broad scene and named-entity discovery. Filling the
    # target cannot silently leave people/events until the very last pages.
    merged = []
    for i in range(max(len(scene_tasks), len(entity_tasks))):
        if i < len(scene_tasks):
            merged.append(scene_tasks[i])
        if i < len(entity_tasks):
            merged.append(entity_tasks[i])
    tasks = [(topic, query, page) for page in range(1, 4) for topic, query in merged]
    adapters = routes()
    lock = threading.Lock()
    abort = threading.Event()
    report = {"action": "index", "target": target, "before": library.stats()["sources"],
              "added": 0, "pages": 0, "skipped_pages": 0, "errors": [], "provider_stops": {},
              "on_demand_providers": [], "downloads": 0, "model_requests": 0}
    report["total"] = report["before"]
    stock_registry = stock_sources.registry()
    for provider in selected_providers:
        access = stock_registry.get(provider)
        if access and not access["bulk_index_allowed"]:
            # Pixabay disallows large automated query sweeps. We also keep
            # Pexels demand-driven to avoid burning its API request allowance.
            # A configured key never authorizes whole-topic prewarming.
            report["provider_stops"][provider] = access["reason"]
            report["on_demand_providers"].append(provider)
        elif access and access["status"] != "configured":
            reason = access["reason"]
            report["provider_stops"][provider] = reason
            report["errors"].append(PROVIDERS[provider] + "：" + reason)
    def snapshot():
        report["elapsed_seconds"] = round(time.monotonic()-began, 3)
        store.mutate(pid, lambda p: p.update(library_report=dict(report)), internal=True)
        progress(min(98, report["total"]/target*98),
                 f"来源索引 {report['total']:,}/{target:,} · 新增 {report['added']:,} · 仅保存文字与封面链接")

    def lane(provider):
        if provider in report["provider_stops"]:
            return
        next_at, errors = 0, 0
        with library.connection() as con:
            recent = con.execute("SELECT error FROM index_pages WHERE provider=? AND status='failed' AND updated>?",
                                 (provider, time.time()-1800)).fetchall()
        blocked = next((r[0] for r in recent if any(s in r[0] for s in ("验证", "429", "412", "403", "限流"))), "")
        if blocked:
            with lock:
                report["provider_stops"][provider] = blocked
                report["errors"].append(PROVIDERS[provider]+"："+blocked)
                snapshot()
            return
        provider_tasks = tasks
        if provider in stock_registry:
            # Stock catalogues use their own language and subject coverage.
            # These routes are retrieval labels, never source/visual evidence.
            provider_tasks = [(topic, query, page) for page in range(1, 4) for topic in chosen
                              for query in stock_queries.queries(topic, provider)]
        for topic, query, page in provider_tasks:
            if abort.is_set():
                return
            check()
            with lock:
                if report["total"] >= target:
                    return
            with library.connection() as con:
                old = con.execute("SELECT status,updated FROM index_pages WHERE provider=? AND query=? AND page=?",
                                  (provider, query, page)).fetchone()
            if old and old["status"] == "completed" and time.time()-old["updated"] < PAGE_TTL:
                with lock:
                    report["skipped_pages"] += 1
                continue
            # Per-provider serial pacing. Cancellation is checked during the wait.
            while time.monotonic() < next_at:
                if abort.is_set():
                    return
                check()
                time.sleep(max(0, min(.2, next_at-time.monotonic())))
            next_at = time.monotonic() + REQUEST_INTERVAL
            failure, rows = "", []
            try:
                rows = adapters[provider](query, page)
                qt = sources.tokens(query)
                rows = [r for r in rows if qt & sources.tokens(r["title"] + " " + r.get("description", ""))]
                errors = 0
            except (ValueError, httpx.HTTPError) as exc:
                failure = str(exc)[:200]
                errors += 1
            check()
            if abort.is_set():
                return
            with lock:
                added = library.remember_sources(rows, topic=topic, query=query)
                with library.connection() as con:
                    con.execute("INSERT OR REPLACE INTO index_pages VALUES (?,?,?,?,?,?,?,?)",
                                (provider, query, page, "failed" if failure else "completed",
                                 len(rows), added, time.time(), failure))
                report["added"] += added
                report["total"] += added
                report["pages"] += 1
                if failure:
                    report["errors"] = (report["errors"] + [PROVIDERS[provider]+"："+failure])[-12:]
                stop = failure and (errors >= 3 or any(s in failure for s in ("验证", "429", "412", "403", "限流", "冷却")))
                if stop:
                    report["provider_stops"][provider] = failure
                snapshot()
            if stop:
                return
    snapshot()
    def guarded_lane(provider):
        try:
            lane(provider)
        except BaseException:
            abort.set()
            raise
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(guarded_lane, provider) for provider in selected_providers]
        for future in futures:
            future.result()
    with lock:
        report["total"] = library.stats()["sources"]
        report["reached_target"] = report["total"] >= target
        report["coverage"] = overview()
        snapshot()
    return report
