"""Adapter for the separate news-crawler project.

Settings: news_crawler_dir (the crawler checkout), news_crawler_command (run there to crawl),
news_crawler_output (its result file, relative to the dir: .json / .jsonl / .csv / .db / .sqlite).
Records need a title and the article text; url, source and publish time are used when present.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from . import config, paths

TIMEOUT = 600
PLATFORMS = {"toutiao", "tencent", "sohu", "baidu", "bilibili", "netease", "sina"}
MAX_SCRIPTS = 5         # the crawler's own 口播 per crawl (about 0.02 CNY each on qwen3.7-plus)
TEXT_KEYS = ("text", "content", "body", "article", "正文", "内容")


def _dir() -> Path | None:
    value = config.read().get("news_crawler_dir", "").strip()
    if not value:
        default = paths.ROOT.parent / "news-crawler"
        return default if default.is_dir() else None
    path = Path(value).expanduser()
    return path if path.is_dir() else None


def builtin(folder: Path | None) -> bool:
    """The 全平台新闻爬虫 project (main.py + core/crawler.py + platforms/): run and read it directly."""
    return bool(folder and (folder / "main.py").is_file() and (folder / "core" / "crawler.py").is_file()
                and (folder / "platforms").is_dir())


def _batches(folder: Path, kind: str) -> list[Path]:
    """The crawler's batch folders, newest first: data/采集/采集_* or koubo/口播/口播_*."""
    root = folder / ("data" if kind == "采集" else "koubo") / kind
    return sorted((p for p in root.glob(f"{kind}_*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True) \
        if root.is_dir() else []


def status() -> dict:
    cfg, folder = config.read(), _dir()
    own = builtin(folder)
    ready = bool(folder and (own or cfg.get("news_crawler_command", "").strip()))
    latest = _batches(folder, "采集")[0] if own and _batches(folder, "采集") else None
    return {"ready": ready, "builtin": own, "dir": str(folder or cfg.get("news_crawler_dir", "")),
            "command": cfg.get("news_crawler_command", ""), "output": cfg.get("news_crawler_output", ""),
            "latest": {"name": latest.name, "count": len(list(latest.glob("*.txt")))} if latest else None,
            "message": "" if ready else "新闻爬虫尚未接入：在服务与设置里填写爬虫目录（全平台新闻爬虫可自动识别）"}


# --- the 全平台新闻爬虫 project ---------------------------------------------------------------------

HEADER = {"标题": "title", "类别": "category", "平台": "platform", "作者/来源": "source", "发布时间": "published",
          "原文链接": "url", "浏览量": "views", "抓取时间": "fetched"}


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding).replace("\r\n", "\n")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def _article(path: Path) -> dict:
    """One saved article: 标题：/类别：/… header lines, a dashed line, then the full text."""
    lines = _read_text(path).split("\n")
    row, body = {}, 0
    while body < len(lines):
        line = lines[body].strip()
        match = re.fullmatch(r"([^：:]+)[：:]\s*(.*)", line) if line else None
        if match and match.group(1) in HEADER:
            row[HEADER[match.group(1)]] = match.group(2).strip()
        elif line and re.fullmatch(r"[-─━—_=]{10,}", line):
            body += 1
            break
        elif line:
            break
        body += 1
    row["text"] = "\n".join(lines[body:]).strip() if row else "\n".join(lines).strip()
    row["title"] = row.get("title") or path.stem
    if row.get("published") in ("未知", "未分类"):
        row["published"] = ""
    if row.get("source") in ("未知",):
        row["source"] = ""
    row["source"] = row.get("source") or {"toutiao": "今日头条", "tencent": "腾讯新闻", "sohu": "搜狐网",
                                           "baidu": "百度", "bilibili": "B站", "netease": "网易", "sina": "新浪"}.get(row.get("platform", ""), "")
    return row


def _base(title: str) -> str:
    """The crawler's file-name stem for a title (storage.safe_filename(title, 50))."""
    stem = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", title or "").strip().strip(". ")[:50]
    return stem or "untitled"


LABEL = re.compile(r"^[*#\s>]*(百家号标题|封面标题|视频标题|标题|关键词|标签|话题)[*\s]*[：:]", re.M)


def _split_script(text: str) -> tuple[str, str]:
    """The narration of a crawler 口播稿 and its 百家号标题; label lines (标题/关键词…) are not narration."""
    title = ""
    match = re.search(r"^[*#\s>]*百家号标题[*\s]*[：:][*\s]*(.+)$", text, re.M)
    if match:
        title = re.sub(r"[*#]", "", match.group(1)).strip()
    label = LABEL.search(text)
    return (text[:label.start()].strip() if label else text.strip()), title


def _koubo(folder: Path, since: float = 0) -> dict:
    """The crawler's own 口播稿 (its AI rewrite), keyed by the file-name stem without _views/_n suffixes."""
    found = {}
    for batch in _batches(folder, "口播"):
        if batch.stat().st_mtime < since:
            break
        for path in batch.glob("*.txt"):
            stem = re.sub(r"_\d+(?:_\d+)?$", "", path.stem)
            found.setdefault(stem, _split_script(_read_text(path).strip()))
    return found


def read_batch(folder: Path, batch: Path, since: float = 0) -> list[dict]:
    koubo = _koubo(folder, since)
    rows = []
    for path in sorted(batch.glob("*.txt")):
        try:
            row = _article(path)
        except OSError:
            continue
        base = _base(row["title"])
        # Same title, same file stem; a 50-character stem may be a truncated longer title.
        script = koubo.get(base) or (next((v for k, v in koubo.items() if len(k) == 50 and base.startswith(k)), None)
                                     if len(base) >= 50 else None)
        if script and len(script[0]) >= 50:
            row["koubo"], row["koubo_title"] = script
        rows.append(row)
    return rows


def import_latest() -> list[dict]:
    """Articles of the crawler's newest batch (crawled in its own window), with its 口播稿 if any."""
    folder = _dir()
    if not builtin(folder):
        raise ValueError(status()["message"] or "没有找到全平台新闻爬虫目录")
    batches = _batches(folder, "采集")
    if not batches:
        raise ValueError("爬虫还没有抓取过新闻（data/采集 下没有批次）")
    return read_batch(folder, batches[0])


RUNNER = ("import os,runpy,sys;folder=os.getcwd();sys.path.insert(0,folder);sys.argv=sys.argv[1:];"
          "runpy.run_path(os.path.join(folder,sys.argv[0]),run_name='__main__')")


def _script_args(python: str, script: str, rest: list[str]) -> list[str]:
    """Run a script from the crawler folder even under an isolated embedded Python (a ._pth file puts
    neither the script folder nor the working directory on sys.path, so `import config` would fail)."""
    return [python, "-X", "utf8", "-c", RUNNER, script, *rest]


def _python() -> str:
    """The crawler is standard-library Python. A packaged app's own executable cannot run scripts."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    found = shutil.which("python3") or shutil.which("python") or shutil.which("py")
    if not found:
        raise ValueError("未找到 Python，无法在此运行爬虫；请在爬虫窗口里抓取后点“导入上次抓取结果”")
    return found


def _crawl_builtin(cfg: dict, folder: Path, koubo: bool) -> list[dict]:
    limit = str(max(1, min(50, int(cfg.get("news_crawler_limit") or 5))))
    args = _script_args(_python(), "main.py", ["--limit", limit])
    names = [n for n in re.split(r"[,，、;；\s]+", cfg.get("news_crawler_platforms", "").lower()) if n]
    unknown = [n for n in names if n not in PLATFORMS]
    if unknown:
        raise ValueError(f"未知的爬虫平台：{'、'.join(unknown)}（可选 {', '.join(sorted(PLATFORMS))}）")
    if names:
        args += ["--platforms", ",".join(dict.fromkeys(names))]
    env = os.environ | {"PYTHONIOENCODING": "utf-8"}
    if koubo:
        # The crawler's own 口播 rewrite runs on the same Qwen account as the video tool (passed only to
        # this process, never written to the crawler's files).
        from . import model_routing
        if not cfg.get("model_api_key"):
            raise ValueError("请先在服务与设置中填写模型 API Key，再让爬虫生成口播稿")
        from . import costs
        report = costs.summary("news-crawler")
        model = report.get("model") or {}
        if model.get("daily_limit_cny") is not None and model.get("daily_cny", 0) + 0.03 * MAX_SCRIPTS > float(model["daily_limit_cny"]):
            raise ValueError("今日模型费用预算不足，未让爬虫生成口播稿；可以只抓新闻，出片时再洗稿")
        base = cfg.get("model_base_url", "").rstrip("/")
        env |= {"NEWS_LLM_API_KEY": cfg["model_api_key"], "NEWS_LLM_API_URL": base + "/chat/completions",
                "NEWS_LLM_MODEL": model_routing.model(cfg, "rewrite")}
        args += ["--koubo", "--max-koubo", str(MAX_SCRIPTS)]
    began = time.time() - 1
    try:
        run = subprocess.run(args, cwd=folder, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=TIMEOUT * (3 if koubo else 1), env=env)
    except FileNotFoundError:
        raise ValueError("无法启动 Python 运行爬虫") from None
    except subprocess.TimeoutExpired:
        raise ValueError("新闻爬虫运行时间过长，已停止；可减少每平台条数") from None
    if run.returncode != 0:
        raise ValueError(f"新闻爬虫运行失败（退出码 {run.returncode}）：{(run.stderr or run.stdout).strip()[-600:]}")
    global last_notes
    last_notes = []
    failed = [line.strip() for line in run.stdout.splitlines() if "抓取失败" in line]
    if koubo:
        done = re.search(r"口播稿完成[:：]\s*成功\s*(\d+)\s*篇[,，]\s*失败\s*(\d+)\s*篇", run.stdout)
        errors = [line.strip() for line in run.stdout.splitlines() if "[失败]" in line or line.strip().startswith("[口播]")]
        if done and int(done.group(2)):
            last_notes.append(f"爬虫口播稿失败 {done.group(2)} 篇" + (f"：{errors[-1][:120]}" if errors else ""))
        elif not done and errors:
            last_notes.append("爬虫没有生成口播稿：" + errors[-1][:120])
    fresh = [b for b in _batches(folder, "采集") if b.stat().st_mtime >= began]
    if not fresh:
        if failed:
            raise ValueError("爬虫没有抓到新闻：" + "；".join(failed[:3])[:300])
        if re.search(r"总计保留 0 条", run.stdout):
            return []
        raise ValueError("爬虫运行完成，但没有找到本次的采集结果：" + run.stdout.strip()[-300:])
    if failed:
        last_notes.append("部分平台抓取失败：" + "；".join(failed[:2])[:200])
    return read_batch(folder, fresh[0], since=began if koubo else 0)


last_notes: list[str] = []


def _command(cfg: dict, folder: Path) -> list[str]:
    parts = shlex.split(cfg["news_crawler_command"], posix=os.name != "nt")
    if parts and parts[0] in ("python", "python3"):
        venv = next((p for p in (folder / ".venv/bin/python", folder / "venv/bin/python",
                                 folder / ".venv/Scripts/python.exe") if p.exists()), None)
        if venv:
            parts[0] = str(venv)
        elif len(parts) > 1 and parts[1].endswith(".py"):
            parts = _script_args(_python(), parts[1], parts[2:])
        else:
            parts[0] = _python()
    return parts


def _records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix in (".db", ".sqlite", ".sqlite3"):
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
                columns = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
                text = next((k for k in TEXT_KEYS if k in columns), None)
                if text and "title" in columns:
                    order = next((c for c in ("published", "publish_time", "created_at", "crawled_at", "id", "rowid")
                                  if c in columns or c == "rowid"), "rowid")
                    rows = con.execute(f'SELECT * FROM "{table}" ORDER BY "{order}" DESC LIMIT 200').fetchall()
                    return [dict(r) for r in rows]
        return []
    raw = path.read_text("utf-8-sig")
    if suffix == ".csv":
        return list(csv.DictReader(raw.splitlines()))
    if suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    data = json.loads(raw)
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def _plain(value) -> str:
    if isinstance(value, list):
        return "\n".join(_plain(v) for v in value if v is not None)
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _text_field(row: dict) -> dict:
    text = next((_plain(row[k]) for k in TEXT_KEYS if _plain(row.get(k)).strip()), "")
    return row | {"text": text, "title": _plain(row.get("title"))}


def crawl(koubo: bool = False) -> list[dict]:
    cfg, folder = config.read(), _dir()
    if builtin(folder) and not cfg.get("news_crawler_command", "").strip():
        return _crawl_builtin(cfg, folder, koubo)
    if not folder or not cfg.get("news_crawler_command", "").strip():
        raise ValueError(status()["message"])
    env = os.environ | {"PYTHONIOENCODING": "utf-8"}
    try:
        run = subprocess.run(_command(cfg, folder), cwd=folder, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=TIMEOUT, env=env)
    except FileNotFoundError:
        raise ValueError("找不到爬虫运行命令，请检查“服务与设置 → 新闻爬虫”里的命令") from None
    except subprocess.TimeoutExpired:
        raise ValueError(f"新闻爬虫运行超过 {TIMEOUT // 60} 分钟仍未结束，已停止") from None
    except (OSError, ValueError) as exc:
        raise ValueError(f"新闻爬虫无法启动：{exc}") from None
    if run.returncode != 0:
        tail = (run.stderr or run.stdout).strip()[-600:]
        raise ValueError(f"新闻爬虫运行失败（退出码 {run.returncode}）：{tail}")
    output = cfg.get("news_crawler_output", "").strip()
    if output:
        path = (folder / output).resolve()
        if not path.exists():
            raise ValueError(f"爬虫运行完成，但没有找到结果文件 {output}")
        try:
            rows = _records(path)
        except (OSError, ValueError, sqlite3.Error, csv.Error) as exc:
            raise ValueError(f"爬虫结果文件 {output} 无法读取：{exc}") from None
    else:   # the command printed JSON (possibly after log lines) or JSON lines
        rows = _printed(run.stdout)
    return [_text_field(r) for r in rows if isinstance(r, dict)]


def _rows(data) -> list:
    """Records in a parsed JSON value: a list of objects, an object that is itself a record (has a title
    or text), or a wrapper object holding the list ({"data": [...]})."""
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        # A wrapper holds a list of records (even if it also has its own title); otherwise it is a record.
        for value in data.values():
            if isinstance(value, list) and any(isinstance(r, dict) and ("title" in r or any(k in r for k in TEXT_KEYS))
                                               for r in value):
                return [r for r in value if isinstance(r, dict)]
        return [data] if "title" in data or any(k in data for k in TEXT_KEYS) else []
    return []


def _printed(out: str) -> list:
    """Records from what the crawler printed: one JSON value, JSON lines, or JSON between log lines."""
    out = out.strip()
    decoder = json.JSONDecoder()
    def value_at(offset):       # a JSON value starting here; trailing log text is ignored
        try:
            return decoder.raw_decode(out, offset)[0]
        except ValueError:
            return None
    lines = [line.strip() for line in out.splitlines() if line.strip().startswith("{")]
    if len(lines) > 1:      # JSON lines: one record (or wrapper) per line; lines that are not JSON are skipped
        parsed = []
        for line in lines:
            try:
                parsed += _rows(json.loads(line))
            except ValueError:
                continue
        if len(parsed) > 1:
            return parsed
    # One JSON value, possibly with log lines before or after it (some log lines start with "[INFO]").
    offset, empty = 0, False
    for line in out.splitlines(keepends=True):
        if line.lstrip()[:1] in ("[", "{"):
            value = value_at(offset + len(line) - len(line.lstrip()))
            if _rows(value):        # log lines like "[1] 抓取…" or {"level": "info"} hold no records
                return _rows(value)
            empty = empty or value == [] or (isinstance(value, dict) and any(v == [] for v in value.values()))
        offset += len(line)
    if empty:
        return []                   # the crawler printed an empty result: nothing new this time
    try:
        return _rows(json.loads(out)) if out else []
    except ValueError:
        raise ValueError("爬虫输出不是 JSON，请在设置里填写结果文件路径") from None
