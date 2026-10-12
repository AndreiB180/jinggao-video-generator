"""Conservative, non-biometric checks for named-person and event footage.

The vision model may transcribe a scoreboard or name caption. It must never
identify a face. Publisher text comes from the retrieval adapter, not the model;
all model quotations are checked against that independently retrieved text.
"""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit


PUBLISHER_DOMAINS = {
    "cctv.com": "央视", "cntv.cn": "央视", "ittf.com": "国际乒联",
    "worldtabletennis.com": "WTT", "olympics.com": "奥林匹克官网",
    "olympics.org": "国际奥委会",
}
TEXT_KINDS = {"caption", "scoreboard", "nameplate"}
SOURCE_ORIGINS = {"publisher_search", "publisher_page", "platform_metadata"}


def subject_name_label(row: dict) -> bool:
    """A headline mentioning a name does not label the person in the footage."""
    if row.get("relation") not in (None, "", "on_subject", "scoreboard_participant"):
        return False
    quote = str(row.get("quote") or "")
    # Record-comparison headlines often overlay another athlete's race.
    if row.get("kind") == "caption" and re.search(r"纪录.*(?:无人|未能|难以|依然|仍)|(?:对比|相比|致敬|超越|追赶|打破).{0,16}(?:纪录|记录)|(?:纪录|记录).{0,12}(?:打破|保持)", quote):
        return False
    return True
EVENT_NAMES = (
    ("中国大满贯", "chinasmash"), ("新加坡大满贯", "singaporesmash"),
    ("美国大满贯", "ussmash", "unitedstatessmash"),
    ("世乒赛", "worldtabletennischampionships", "worldchampionships"),
    ("世界杯", "worldcup"), ("奥运会", "olympicgames"),
)


def event_text(value: str) -> str:
    # Publisher titles often use traditional Chinese. Only normalize the
    # event vocabulary here; this is not a guessed alias for a person's name.
    return value.translate(str.maketrans("決賽輪滿貫國單體強晉進", "决赛轮满贯国单体强晋进"))


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w\u3400-\u9fff]+", "", text)


def persons(value) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("人物条件需要是不超过 8 项的列表")
    result = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("人物条件格式无效")
        name, aliases = row.get("name"), row.get("aliases", [])
        if not isinstance(name, str) or not 2 <= len(name.strip()) <= 100:
            raise ValueError("人物需要完整姓名，长度为 2–100 字")
        if (not isinstance(aliases, list) or len(aliases) > 8
                or any(not isinstance(a, str) or not 2 <= len(a.strip()) <= 100 for a in aliases)):
            raise ValueError("人物别名格式无效")
        result.append({"name": name.strip(), "aliases": list(dict.fromkeys(a.strip() for a in aliases))})
    return result


def mentions(text: str, person: dict) -> bool:
    """Match whole Latin names and complete Chinese aliases, never surnames alone."""
    for name in [person["name"], *person.get("aliases", [])]:
        if name != person["name"]:
            if re.search(r"(?:先生|女士|老师|教授|医生|医师|博士|教练|同学|队长|球员|选手)$", name):
                continue
            if name in {"大胖", "小胖", "小将", "少年", "老将", "嘉宾", "记者", "国乒"}:
                continue
            han = re.findall(r"[\u3400-\u9fff]", name)
            canonical_han = re.findall(r"[\u3400-\u9fff]", person["name"])
            if han and canonical_han and len(han) < len(canonical_han):
                continue
            if not han and len(re.findall(r"[A-Za-zÀ-ž]+", name)) < 2:
                continue
        if re.search(r"[\u3400-\u9fff]", name):
            if normalize(name) in normalize(text):
                return True
        else:
            # Punctuation/spaces may vary in a scoreboard, but adjacent letters
            # cannot turn e.g. "Ann Lee" into a match for "Joann Leeds".
            parts = re.findall(r"\w+", unicodedata.normalize("NFKC", name), flags=re.UNICODE)
            if parts and re.search(r"(?<!\w)" + r"[\W_]*".join(map(re.escape, parts)) + r"(?!\w)",
                                   unicodedata.normalize("NFKC", text), flags=re.IGNORECASE):
                return True
    return False


def web_host(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            return ""
        return (parsed.hostname or "").lower().rstrip(".")
    except (TypeError, ValueError):
        return ""


def publisher_domain(url: str) -> str:
    host = web_host(url)
    return next((domain for domain in PUBLISHER_DOMAINS if host == domain or host.endswith("." + domain)), "")


def source_record(asset: dict) -> tuple[dict, str]:
    """Accept only retrieval-provided text for the candidate's actual source."""
    record = asset.get("source_evidence")
    if not isinstance(record, dict) or record.get("readable") is not True:
        return {}, ""
    if record.get("evidence_origin") not in SOURCE_ORIGINS:
        return {}, ""
    url, source_url = str(record.get("url") or ""), str(asset.get("source_url") or "")
    # A link to a trusted publisher in an unrelated page does not authenticate it.
    same_host = web_host(url) and web_host(url) == web_host(source_url)
    same_publisher = publisher_domain(url) and publisher_domain(url) == publisher_domain(source_url)
    if not same_host and not same_publisher:
        return {}, ""
    # The URL must identify the same page, allowing only known publisher host
    # aliases to differ. Do not accept arbitrary pages from the same site.
    if urlsplit(url).path.rstrip("/") != urlsplit(source_url).path.rstrip("/"):
        return {}, ""
    if urlsplit(url).query != urlsplit(source_url).query:
        return {}, ""
    text = "\n".join(str(record.get(key) or "")[:10000] for key in ("title", "description"))
    return record, text


def _quoted_source(quote: str, text: str) -> bool:
    needle = normalize(quote)
    return len(needle) >= 4 and needle in normalize(text)


def event_dates(text: str) -> set[tuple[int | None, int, int]]:
    """Extract explicit event dates; a publication timestamp is deliberately absent."""
    dates = set()
    for year, month, day in re.findall(r"(?:(19\d{2}|20\d{2})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]", text):
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            dates.add((int(year) if year else None, int(month), int(day)))
    for year, month, day in re.findall(r"(?<!\d)((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", text):
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            dates.add((int(year), int(month), int(day)))
    months = {name: index for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}
    for month, day, year in re.findall(r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(20\d{2}|19\d{2})\b", text):
        if month.lower()[:3] in months and 1 <= int(day) <= 31:
            dates.add((int(year), months[month.lower()[:3]], int(day)))
    for day, month, year in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\s+(20\d{2}|19\d{2})\b", text):
        if month.lower()[:3] in months and 1 <= int(day) <= 31:
            dates.add((int(year), months[month.lower()[:3]], int(day)))
    return dates


def event_rounds(text: str) -> set[tuple[str, int]]:
    """Keep ordinal rounds separate from draw sizes: round 2 is not always R32."""
    text = event_text(text)
    rounds = set()
    for value in re.findall(r"\b(?:[MW][SD]\s*)?R\s*(128|64|32|16|8)\b|\bround\s+of\s+(128|64|32|16|8)\b", text, re.I):
        rounds.add(("size", int(next(x for x in value if x))))
    for denominator in re.findall(r"(?<!\d)1\s*/\s*(64|32|16|8|4|2)\s*决赛", text):
        rounds.add(("size", int(denominator) * 2))
    for size in re.findall(r"(?:轮次\s*[:：]\s*(?:男单|女单)?|(?:男单|女单)\s*)(128|64|32|16|8)强(?:赛)?", text):
        rounds.add(("size", int(size)))
    for pattern, size in ((r"八分之一决赛", 16), (r"四分之一决赛|八强赛|quarter[ -]?final", 8),
                          (r"半决赛|四强赛|semi[ -]?final", 4)):
        if re.search(pattern, text, re.I):
            rounds.add(("size", size))
    remaining = re.sub(r"[一二四八十六三]+分之一决赛|半决赛|1\s*/\s*\d+\s*决赛|(?:quarter|semi)[ -]?finals?", "", text, flags=re.I)
    if re.search(r"决赛|\bfinals?\b", remaining, re.I):
        rounds.add(("size", 2))
    chinese = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8}
    for value in re.findall(r"第([一二三四五六七八]|\d{1,2})轮", text):
        rounds.add(("ordinal", chinese[value] if value in chinese else int(value)))
    for value in re.findall(r"\bround\s+(\d{1,2})(?!\d)", text, re.I):
        rounds.add(("ordinal", int(value)))
    if "首轮" in text:
        rounds.add(("ordinal", 1))
    # "晋级八强" describes an R16 win, not an R8 match. Bare “八强”
    # is ambiguous and is intentionally not used to assert a playing round.
    for prefix, size in re.findall(r"(晋级|挺进|闯入|进入|止步)\s*(?:男单|女单|男双|女双|混双)?\s*(128|64|32|16|8|八|四)强", text):
        number = {"八": 8, "四": 4}.get(size, int(size) if size.isdigit() else 0)
        rounds.add(("size", number if prefix == "止步" else number * 2))
    return rounds


def event_opponents(event: str, required: list[dict]) -> tuple[list[dict], bool]:
    clause = re.search(r"对阵\s*[:：]\s*([^；;\n]+)", event)
    text = clause.group(1) if clause else event
    parts = re.split(r"\s+vs?\.?\s+|\s+versus\s+|\s+against\s+|对阵|迎战|对战", text, flags=re.I)
    if len(parts) != 2:
        return [], bool(clause)
    result = []
    for part in parts:
        matches = [person for person in required if mentions(part, person)]
        if len(matches) == 1:
            result.append(matches[0])
            continue
        # A structured 对阵 field allows exact names that were not subjects of
        # the shot; unstructured surrounding prose is never guessed into a name.
        candidate = part.strip(" ：:，,。.")
        if clause and (re.fullmatch(r"[\u3400-\u9fff·]{2,10}", candidate)
                       or re.fullmatch(r"[A-Za-zÀ-ž'-]+(?:[ -][A-Za-zÀ-ž'-]+){1,4}", candidate)):
            result.append({"name": candidate, "aliases": []})
        else:
            return [], True
    return result, True


def event_checks(event: str, text: str, required: list[dict]) -> list[dict]:
    """Hard qualifiers can veto an LLM's semantic same-event judgement."""
    event, text = event_text(event), event_text(text)
    checks = []
    expected_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", event))
    actual_years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text))
    if expected_years:
        status = "supported" if expected_years <= actual_years else "conflict" if actual_years and expected_years.isdisjoint(actual_years) else "insufficient"
        checks.append({"field": "年份", "status": status, "expected": sorted(expected_years), "observed": sorted(actual_years)})
    expected_dates, actual_dates = event_dates(event), event_dates(text)
    for expected in sorted(expected_dates, key=str):
        compatible = [actual for actual in actual_dates if actual[1:] == expected[1:]
                      and (actual[0] is None or expected[0] is None or actual[0] == expected[0])]
        status = "supported" if compatible else "conflict" if actual_dates else "insufficient"
        checks.append({"field": "日期", "status": status, "expected": list(expected),
                       "observed": [list(date) for date in sorted(actual_dates, key=str)]})
    expected_rounds, actual_rounds = event_rounds(event), event_rounds(text)
    for kind, value in sorted(expected_rounds):
        same_kind = [actual for actual in actual_rounds if actual[0] == kind]
        status = "supported" if (kind, value) in actual_rounds else "conflict" if same_kind else "insufficient"
        checks.append({"field": "轮次", "status": status, "expected": [kind, value],
                       "observed": [list(round_) for round_ in sorted(actual_rounds)]})
    opponents, has_opponents = event_opponents(event, required)
    if has_opponents:
        missing = [person["name"] for person in opponents if not mentions(text, person)]
        status = "supported" if opponents and not missing else "insufficient"
        checks.append({"field": "对阵", "status": status, "expected": [person["name"] for person in opponents], "missing": missing})
    # General named conferences, announcements, hearings, etc. use an explicit
    # activity field. The same method works without a sports-domain allowlist.
    label = re.search(r"(?:活动|赛事|会议|事件)\s*[:：]\s*([^；;\n]+)", event)
    if label:
        expected = normalize(re.sub(r"(?:19|20)\d{2}年?", "", label.group(1)))
        label_events = {i for i, names in enumerate(EVENT_NAMES) if any(normalize(n) in expected for n in names)}
        source_events = {i for i, names in enumerate(EVENT_NAMES) if any(normalize(n) in normalize(text) for n in names)}
        status = "supported" if ((expected and expected in normalize(text))
                                or (label_events and label_events <= source_events)) else "insufficient"
        checks.append({"field": "活动名称", "status": status, "expected": label.group(1)})
    expected_events = {i for i, names in enumerate(EVENT_NAMES) if any(normalize(n) in normalize(event) for n in names)}
    actual_events = {i for i, names in enumerate(EVENT_NAMES) if any(normalize(n) in normalize(text) for n in names)}
    if expected_events:
        status = "supported" if expected_events <= actual_events else "conflict" if actual_events and expected_events.isdisjoint(actual_events) else "insufficient"
        checks.append({"field": "赛事名称", "status": status, "expected": sorted(expected_events), "observed": sorted(actual_events)})
    return checks


def assess(shot: dict, asset: dict, observations: dict | None = None, *,
           frame_count: int = 0, has_segment_frames: bool = False) -> dict:
    """Combine source text and readable on-screen labels without identifying faces.

    `needs_segment_preview` lets the worker escalate a title-only video candidate
    to cheap temporal previews before it downloads a full clip. `supported` means
    source/label evidence supports use, not that anyone's face was identified.
    """
    observations = observations if isinstance(observations, dict) else {}
    required = persons(shot.get("persons", []))
    mode = shot.get("evidence_mode", "person" if required else "generic")
    event = str(shot.get("event") or "").strip()
    exact_event = bool(event) and shot.get("event_scope", "exact") != "background"
    needs_person = bool(required) or mode == "person"
    needs_event = exact_event or (mode == "event" and shot.get("event_scope", "exact") != "background")
    base = {"status": "supported", "matched_names": [], "evidence": [],
            "reason": "本镜未要求特定人物或赛事；仍需检查画面质量和内容。",
            "event_status": "not_requested", "coverage": "generic",
            "needs_segment_preview": False, "selection_blocked": False, "requires_overlay": False}
    if not needs_person and not needs_event:
        return base
    record, text = source_record(asset)
    source_url = str(asset.get("source_url") or "")
    evidence, matched = [], []
    conflicts = observations.get("context_conflicts", [])
    conflicts = [str(x)[:200] for x in conflicts[:8] if isinstance(x, str) and x.strip()] if isinstance(conflicts, list) else []
    visible = observations.get("visual_evidence", [])
    visible = visible if isinstance(visible, list) else []
    # A Commons file record binds its caption to this exact downloaded image.
    # This is explicit source attribution, never identification from a face or
    # from an unrelated article title/search thumbnail.
    attributed_image = (asset.get("kind") == "image" and asset.get("provider") == "commons"
        and web_host(source_url) == "commons.wikimedia.org"
        and record.get("evidence_origin") == "platform_metadata"
        and record.get("image_url") == (asset.get("origin_image_url") or asset.get("url"))
        and web_host(record.get("image_url", "")) in ("upload.wikimedia.org", "thumb.wikimedia.org"))
    if asset.get("kind") == "image" and asset.get("provider") == "publisher_image":
        from .context_sources import photo_publisher
        attributed_image = (photo_publisher(source_url) and record.get("evidence_origin") == "publisher_page"
            and record.get("photo_attribution") == "publisher_caption"
            and bool(record.get("image_url"))
            and record["image_url"] == (asset.get("origin_image_url") or asset.get("url")))
    for row in visible[:20]:
        if not isinstance(row, dict) or row.get("kind") not in TEXT_KINDS:
            continue
        quote, index, context = row.get("quote"), row.get("frame_index"), row.get("context")
        if (not isinstance(quote, str) or not 1 <= len(quote) <= 300
                or type(index) is not int or not 0 <= index < frame_count
                or not isinstance(context, str) or not context.strip()):
            continue
        evidence.append({"kind": row["kind"], "quote": quote, "source_url": source_url,
                         "frame_index": index, "context": context[:300], "relation": row.get("relation", "")})
    for person in required:
        named_on_screen = any(subject_name_label(row) and mentions(row["quote"], person) for row in evidence)
        named_in_source = bool(text) and mentions(text, person)
        if (named_on_screen or attributed_image) and named_in_source:
            matched.append(person["name"])
            if attributed_image:
                evidence.append({"kind": "image_caption_attribution", "quote": text[:1000],
                                 "source_url": source_url, "image_url": record["image_url"]})
    if record:
        for key in ("title", "description"):
            value = str(record.get(key) or "")
            if value and any(mentions(value, person) for person in required):
                evidence.append({"kind": "source_" + key, "quote": value[:1000], "source_url": source_url})
        domain = publisher_domain(source_url)
        if domain:
            evidence.append({"kind": "publisher_domain", "quote": PUBLISHER_DOMAINS[domain],
                             "source_url": source_url})
    elif asset.get("title"):
        # Keep this as a visible reason to investigate, never as corroboration.
        evidence.append({"kind": "unverified_title", "quote": str(asset["title"])[:500], "source_url": source_url})
    event_status = "not_requested"
    checks = []
    if event or mode == "event":
        event_status = "insufficient"
        event_rows = observations.get("event_evidence", [])
        quoted_event_text = []
        if isinstance(event_rows, list):
            for row in event_rows[:8]:
                if not isinstance(row, dict) or not isinstance(row.get("quote"), str):
                    continue
                if row.get("source_url") != source_url or not _quoted_source(row["quote"], text):
                    continue
                relation = row.get("relation")
                if relation not in {"same_event", "different_event"}:
                    continue
                evidence.append({"kind": "event_source", "quote": row["quote"][:1000],
                                 "source_url": source_url, "relation": relation})
                if relation == "different_event":
                    event_status = "conflict"
                elif event_status != "conflict":
                    event_status = "supported"
                quoted_event_text.append(row["quote"])
        event_people = list({p["name"]: p for p in [*required, *persons(shot.get("event_persons", []))]}.values())
        checks = event_checks(event, "\n".join(quoted_event_text) or text, event_people)
        if event_status != "conflict":
            # A dated report can corroborate the very same fixture, but never
            # merely lend its date to a video of another round or opponent.
            corroboration = corroborate_date(asset, record, event, checks)
            if corroboration:
                evidence.append(corroboration)
                evidence.append({"kind": "event_source", "quote": str(record.get("title") or ""),
                                 "source_url": source_url, "relation": "same_event",
                                 "method": "matching_fixture_and_dated_report"})
                event_status = "supported"
                for check in checks:
                    if check["field"] == "日期" and check["status"] == "insufficient":
                        check.update(status="supported", corroborating_url=corroboration["source_url"])
        if any(row["status"] == "conflict" for row in checks):
            event_status = "conflict"
        elif any(row["status"] == "insufficient" for row in checks) and event_status == "supported":
            event_status = "insufficient"
        if shot.get("event_scope", "exact") == "background":
            event_status = "background"
    missing = [person["name"] for person in required if person["name"] not in matched]
    status, reason = "supported", "已读取来源正文/标题与可见姓名字幕或计分牌互相支持；没有进行人脸识别。"
    coverage = "named_participant" if matched else "event_report"
    if attributed_image and matched:
        coverage, reason = "source_attributed", "该照片的原始文件页与图注明确标注人物；依据来源归属，没有进行人脸识别。"
    if conflicts or (needs_event and event_status == "conflict"):
        status, reason = "conflict", "；".join(conflicts) or "来源显示为不同赛事或年份，不能作为本场画面。"
    elif needs_person and (not required or missing):
        status, reason = "insufficient", "缺少可追溯来源与可见姓名文字的双重证据：" + ("、".join(missing) or "尚未指定完整姓名")
    elif needs_event and event_status != "supported":
        missing_fields = list(dict.fromkeys(row["field"] for row in checks if row["status"] == "insufficient"))
        status, reason = "insufficient", "人物相关性不足以证明同一场活动；来源证据尚未覆盖" + ("、".join(missing_fields) or "具体事件、日期及参与人") + "。"
    requires_overlay = event_status == "background"
    if requires_overlay and not re.search(r"资料|示意", str(shot.get("overlay_label") or "")):
        status, reason = "insufficient", "背景赛事素材必须叠加“资料画面”说明后再使用。"
    needs_preview = status == "insufficient" and asset.get("kind") == "video" and not has_segment_frames
    return base | {"status": status, "matched_names": matched, "evidence": evidence, "reason": reason,
                   "event_status": event_status, "coverage": coverage if status == "supported" else "unverified",
                   "event_checks": checks,
                   "selection_blocked": status != "supported", "needs_segment_preview": needs_preview,
                   "requires_overlay": requires_overlay}


def corroborate_date(asset, video_record, event, checks):
    if (not any(c["field"] == "日期" and c["status"] == "insufficient" for c in checks)
            or any(c["status"] == "conflict" for c in checks)):
        return None
    title = event_text(str(video_record.get("title") or ""))
    people = [p for p in asset.get("entity_names", []) if isinstance(p, dict) and p.get("name") and mentions(title, p)]
    rounds = event_rounds(title)
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", title))
    families = {i for i, names in enumerate(EVENT_NAMES) if any(normalize(n) in normalize(title) for n in names)}
    # Exactly two named opponents and one edition/round/family identify a
    # fixture. A general player highlights montage cannot pass this bridge.
    if len(people) != 2 or len(rounds) != 1 or len(years) != 1 or len(families) != 1:
        return None
    dates = event_dates(event)
    for document in asset.get("context_sources", [])[:12]:
        if (not isinstance(document, dict) or document.get("readable") is not True
                or document.get("evidence_origin") != "publisher_page" or not web_host(document.get("url", ""))):
            continue
        for paragraph in document.get("paragraphs", [])[:12]:
            if not isinstance(paragraph, str) or not 30 <= len(paragraph) <= 1600:
                continue
            text = event_text(paragraph)
            actual = event_dates(text)
            if not all(any(a[1:] == d[1:] and (d[0] is None or a[0] is None or d[0] == a[0]) for a in actual) for d in dates):
                continue
            if (not all(mentions(text, person) for person in people) or not rounds <= event_rounds(text)
                    or not years <= set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text))):
                continue
            article_families = {i for i, names in enumerate(EVENT_NAMES) if any(normalize(n) in normalize(text) for n in names)}
            if not families <= article_families:
                continue
            return {"kind": "event_date_corroboration", "quote": paragraph, "source_url": document["url"],
                    "video_source_url": asset.get("source_url", ""), "reason": "同一赛事届次、轮次与双方姓名均对应，报道正文明确比赛日期"}
    return None
