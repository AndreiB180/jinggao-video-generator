from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import (__version__, captions, config, costs, curation, demo, library, library_collect, media, paths, source_index,
               sources, store, subtitle_style)
from .models import (AssetPatch, BatchRequest, FileAction, FootageBuild, JobRequest, LibraryBuild, LibraryUse, NewProject, ProjectPatch, RemoteImport,
                     NewsCrawl, NewsLink, NewsProduce, NewsText, QueueAction, RewriteRequest, ShotPatch)
from .worker import shot_by_id


def create_app() -> FastAPI:
    store.init()
    app = FastAPI(title="镜稿 · 本地视频工作台", docs_url="/api/docs")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost", "::1", "testserver"):
                return JSONResponse({"detail": "仅接受本地工作台请求"}, status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"detail": "不接受跨站写入"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": str(exc).strip("'")}, status_code=404)

    @app.exception_handler(RuntimeError)
    async def runtime_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "app": "jinggao"}

    @app.get("/api/system")
    def system():
        return media.hardware() | {"font": (paths.ROOT/"assets/fonts/NotoSansSC-Regular.otf").exists(),
                                   "version": __version__}

    @app.get("/api/system/report")
    def system_report():
        """Plain text the user can paste to support: the card check, CPU, concurrency and the latest videos'
        step timings and encoders. No keys, scripts or file contents."""
        import platform
        from fastapi.responses import PlainTextResponse
        info, cfg = media.hardware(), config.read()
        lines = [f"镜稿 {__version__} · {platform.platform()} · Python {platform.python_version()}",
                 f"显卡：{info.get('gpu')} · 驱动 {info.get('driver') or '-'} · NVENC {'可用' if info.get('nvenc') else '不可用'}"
                 f" · 全程显卡 {'可用' if info.get('full') else '不可用'} · 显卡编码名额 {info.get('gpu_slots')}",
                 f"原因：{info.get('reason') or '-'}", f"建议：{info.get('advice') or '-'}",
                 f"CPU：{info.get('cores')} 核 · CPU 转码名额 {info.get('cpu_slots')}",
                 "设置：" + " ".join(f"{k}={cfg.get(k)}" for k in ("encoder", "video_workers", "render_workers",
                                                                 "match_workers", "tts_provider", "pipeline")), ""]
        lines.insert(5, f"内存：{info.get('memory_gb')} GB · 这台电脑最多同时 {info.get('max_videos')} 条"
                        f"（按 CPU {info.get('cpu_videos')}、按内存 {info.get('memory_videos')}）"
                        f" · 读字用显卡 {'是' if info.get('ocr_gpu') else '否'}")
        live = {}
        for path in (paths.DATA / "diag").glob("*.json"):
            try:
                item = json.loads(path.read_text("utf-8"))
                if item.get("job") and time.time() - item.get("updated", 0) < 60:
                    live[item["job"]] = item
            except (OSError, ValueError):
                pass
        with store.connection() as con:
            jobs = con.execute("SELECT jobs.*, json_extract(projects.body, '$.title') AS title FROM jobs JOIN projects "
                               "ON projects.id=jobs.project_id WHERE jobs.kind='produce' AND jobs.created>? "
                               "ORDER BY jobs.created DESC LIMIT 40", (time.time() - 86400,)).fetchall()
        now = time.time()
        busy = [j for j in jobs if j["status"] in ("running", "queued")]
        if busy:
            lines.append(f"正在做 / 排队（{len(busy)}）：")
            for j in busy:
                waits = (live.get(j["id"]) or {}).get("waited") or {}
                lines.append(f"  {str(j['title'])[:24]} · {j['status']} {j['progress']:.0f}% · 已 {now - j['created']:.0f} 秒 · "
                             f"{str(j['message'])[:60]}" + (f" · 排队 {waits}" if waits else ""))
        failed = [j for j in jobs if j["status"] in ("failed", "needs_review", "interrupted")][:6]
        if failed:
            lines.append("没做完：")
            for j in failed:
                lines.append(f"  {str(j['title'])[:24]} · {j['status']} · {str(j['message'])[:40]} · {str(j['error'])[:120]}")
        lines.append("")
        rows = [p for p in store.list_projects() if p.get("video")][:12]
        for row in rows:
            p = store.get_project(row["id"])
            video = p["exports"][0]
            wall = (p.get("workflow_info") or {}).get("stage_wall_seconds") or {}
            report = p.get("express_report") or {}
            clips = report.get("clip_encoders") or {}
            timing = report.get("timing") or {}
            lines.append(f"{p['title'][:30]} · {len(p['shots'])} 镜 · 成片 {video.get('duration', 0):.0f} 秒 · "
                         + " ".join(f"{k}={v:.0f}s" for k, v in wall.items() if isinstance(v, (int, float)))
                         + f" · 片段转码 {clips or '-'} · 合成 {video.get('encoders') or video.get('encoder')}")
            lines.append("    匹配内部 " + " ".join(f"{k}={v:.0f}s" for k, v in timing.items() if isinstance(v, (int, float)))
                         + f" · 排队 {report.get('waited') or '-'} · 模型 {report.get('model_seconds') or '-'}"
                         + f" · 下载 {report.get('fetch_seconds') or '-'}")
        return PlainTextResponse("\n".join(lines))

    @app.get("/api/assist")
    def assist_status():
        from . import assist
        return assist.status()

    @app.post("/api/assist/start")
    def assist_start(request: Request):
        from . import assist
        if not assist.available():
            raise HTTPException(404, "这个版本没有带远程协助")
        try:   # the app's own port: a request through the relay arrives on the developer's forwarded port
            port = int(json.loads((paths.DATA / "instance.json").read_text("utf-8"))["port"])
        except (OSError, ValueError, KeyError, TypeError):
            port = request.url.port or 8770
        return assist.start(port)

    @app.post("/api/assist/stop")
    def assist_stop():
        from . import assist
        return assist.stop()

    @app.post("/api/system/gpu-check")
    def gpu_check():
        """Run the card check again (after a driver update or closing a screen recorder)."""
        from . import gpu
        gpu.detect()
        return system()

    @app.get("/api/settings")
    def settings():
        return config.public()

    @app.patch("/api/settings")
    def save_settings(values: dict):
        return config.update(values)

    @app.post("/api/settings/stock/check")
    def check_stock_key(values: dict):
        from . import stock_sources
        if set(values) - {"provider", "api_key"} or values.get("provider") not in ("pexels", "pixabay"):
            raise ValueError("请选择 Pexels 或 Pixabay 素材接口")
        key = values.get("api_key", "")
        if not isinstance(key, str) or len(key) > 4000:
            raise ValueError("素材 Key 格式无效")
        provider = values["provider"]
        settings = config.read()
        if key.strip():
            settings = settings | {provider + "_api_key": key.strip()}
        result = stock_sources.search(provider, "nature", settings=settings)
        # A user-triggered metadata query: no model call, download or settings
        # write. Never return provider headers, temporary credentials or URLs.
        return {"provider": provider, "status": result["status"], "reason": result["reason"],
                "checked": result["checked"], "cache_hit": result["cache_hit"],
                "observed_at": result["observed_at"], "result_count": len(result["items"])}

    @app.get("/api/costs")
    def model_costs():
        return costs.summary()

    @app.get("/api/output-folder")
    def output_folder():
        cfg = config.read()
        return {"path": str(media.output_folder()), "auto_save": cfg["auto_save_exports"] == "1"}

    @app.post("/api/output-folder/open")
    def open_output_folder():
        folder = media.output_folder()
        folder.mkdir(parents=True, exist_ok=True)
        media.open_path(folder)
        return {"path": str(folder)}

    @app.post("/api/output-folder/choose")
    def choose_output_folder():
        """The system folder picker; the chosen folder becomes the place finished videos are saved."""
        chosen = media.choose_folder(str(media.output_folder()))
        if chosen:
            config.update({"export_dir": chosen})
        return {"path": str(media.output_folder()), "changed": bool(chosen)}

    @app.post("/api/projects/{pid}/reveal")
    def reveal_video(pid: str, body: FileAction):
        """Show the finished video selected in its folder."""
        target = media.finished_file(store.get_project(pid), body.export_id)
        media.reveal(target)
        return {"path": str(target)}

    @app.post("/api/projects/{pid}/play")
    def play_video(pid: str, body: FileAction):
        """Open the finished video in the computer's own video player."""
        target = media.finished_file(store.get_project(pid), body.export_id)
        media.open_path(target)
        return {"path": str(target)}

    @app.get("/api/footage-index")
    def footage_index_status():
        from . import footage_index
        pid = library_collect.project_id()
        job = next((j for j in store.jobs(pid) if j["kind"] == "footage_index"), None)
        return footage_index.stats() | {"job": job, "report": store.get_project(pid).get("footage_index_report")}

    @app.post("/api/footage-index/build", status_code=202)
    def footage_index_build(body: FootageBuild):
        from . import footage_index
        unknown = [name for name in body.packs if name not in footage_index.PACKS]
        if unknown:
            raise ValueError("没有这个题材包：" + "、".join(unknown))
        if not body.packs and not body.keywords:
            raise ValueError("请选择题材包或填写关键词")
        return store.enqueue(library_collect.project_id(), "footage_index", body.model_dump())

    @app.get("/api/footage-index/file")
    def footage_index_file():
        import sqlite3
        source = paths.DATA / "footage-index.sqlite"
        if not source.exists():
            raise KeyError("素材库还是空的")
        target = paths.DATA / "footage-index-export.sqlite"
        with sqlite3.connect(source) as con, sqlite3.connect(target) as out:
            con.backup(out)       # a consistent copy while jobs may be writing
        return FileResponse(target, filename="镜稿素材库.sqlite", media_type="application/octet-stream")

    @app.post("/api/footage-index/import")
    def footage_index_import(file: UploadFile = File(...)):
        from . import footage_index
        target = paths.DATA / "footage-index-import.sqlite"
        size = 0
        try:
            with target.open("wb") as out:
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 2 * 1024 ** 3:
                        raise ValueError("素材库文件最大 2 GB")
                    out.write(chunk)
            added = footage_index.merge(target)
        finally:
            file.file.close()
            target.unlink(missing_ok=True)
        return footage_index.stats() | {"added": added}

    def voice_module(provider):
        from . import edge_voice, qwen_tts
        return {"edge_tts": (edge_voice, "edge_tts_voice", 0.0), "qwen_tts": (qwen_tts, "qwen_tts_voice", 0.16)}.get(provider)

    @app.get("/api/tts/voices")
    def tts_voices(provider: str = ""):
        cfg = config.read()
        provider = provider or cfg["tts_provider"]
        found = voice_module(provider)
        if not found:
            return {"provider": provider, "voices": [], "voice": "", "voice_key": "", "speed": cfg["tts_speed"],
                    "speeds": ["1.0", "1.1", "1.2", "1.3"], "price_per_1000_chars": 0, "key_set": bool(cfg["model_api_key"])}
        module, key, price = found
        return {"provider": provider, "voice": module.voice(cfg), "voice_key": key, "speed": cfg["tts_speed"],
                "speeds": list(module.SPEEDS), "voices": module.voices(), "price_per_1000_chars": price,
                "key_set": bool(cfg["model_api_key"])}

    def voice_owner(name):
        from . import edge_voice, qwen_tts
        return edge_voice if name in edge_voice.NAMES else qwen_tts if name in qwen_tts.NAMES else None

    @app.post("/api/tts/preview")
    def tts_preview(values: dict):
        name = values.get("voice")
        module = voice_owner(name) if isinstance(name, str) else None
        if not module:
            raise ValueError("没有这个音色")
        module.preview(name)
        return {"url": "/api/tts/sample/" + name}

    @app.get("/api/tts/sample/{name}")
    def tts_sample(name: str):
        if not voice_owner(name):
            raise KeyError("没有这个音色")
        bundled = paths.ROOT / "assets" / "voices" / f"{name}.mp3"
        if bundled.exists():
            return FileResponse(bundled, media_type="audio/mpeg")
        cached = sorted((paths.DATA / "voice-samples").glob(f"{name}-*.*"))
        if not cached:
            raise KeyError("这个音色还没有试听样音")
        return FileResponse(cached[-1], media_type="audio/mpeg" if cached[-1].suffix == ".mp3" else "audio/wav")

    @app.post("/api/baidu/browser")
    def open_baidu_browser():
        from .baidu_web import open_browser
        return open_browser()

    @app.get("/api/baidu/status")
    def baidu_status():
        from .baidu_http import status
        return status()

    @app.post("/api/bilibili/qr")
    def bilibili_qr():
        import base64
        from .express_sources import qr_login_start
        start = qr_login_start()
        return {"key": start["key"], "image": "data:image/png;base64," + base64.b64encode(start["png"]).decode()}

    @app.get("/api/bilibili/qr/{key}")
    def bilibili_qr_poll(key: str):
        import re
        from .express_sources import qr_login_poll
        if not re.fullmatch(r"[A-Za-z0-9]{8,64}", key):
            raise ValueError("登录二维码无效")
        return qr_login_poll(key)

    @app.get("/api/bilibili/status")
    def bilibili_status():
        from .express_sources import login_status
        return login_status()

    @app.get("/api/projects")
    def list_projects():
        return store.list_projects()

    @app.post("/api/projects", status_code=201)
    def new_project(body: NewProject):
        return store.new_project(**body.model_dump())

    @app.post("/api/batch", status_code=202)
    def new_batch(body: BatchRequest):
        if not config.read()["model_api_key"]:
            raise ValueError("请先在服务与设置中填写模型 API Key，再启动批量制作")
        if body.workers:
            config.update({"video_workers": str(body.workers)})
        batch, result = store.uid()[:12], []
        for item in body.projects:
            p = store.new_project(**item.model_dump())
            store.mutate(p["id"], lambda p: p.update(requirements=body.requirements))
            job = store.enqueue(p["id"], "produce", {"resolution": body.resolution, "mode": "model", "batch": batch})
            result.append({"project_id": p["id"], "title": p["title"], "job_id": job["id"]})
        return {"batch": batch, "projects": result}

    @app.post("/api/batch/audio", status_code=202)
    def batch_audio(file: UploadFile = File(...), batch: str = Form(""), aspect: str = Form("16:9"),
                    resolution: str = Form("1080p"), burn_subtitles: str = Form("1"),
                    media_policy: str = Form("video_preferred")):
        """音频批量成片: one finished narration (audio, or a video whose sound is the narration) per call; the
        page sends a folder's files one after another with the batch id of the first answer. Each becomes its
        own video: the words are recognised locally, then shots, footage and subtitles follow the recording."""
        if not config.read()["model_api_key"]:
            raise ValueError("请先在服务与设置中填写模型 API Key，再启动批量制作")
        if aspect not in ("16:9", "9:16") or resolution not in ("720p", "1080p") or \
                media_policy not in ("video_preferred", "video_only", "mixed") or len(batch) > 40:
            raise ValueError("批量设置无效")
        name = re.sub(r"\.[^.]+$", "", __import__("pathlib").Path(file.filename or "").name).strip()[:100] or "音频成片"
        project = store.new_project(name, "", aspect, media_policy=media_policy, burn_subtitles=burn_subtitles != "0")
        try:
            upload(project["id"], file, "source_audio", "")
        except Exception:
            store.discard_project(project["id"])
            raise
        batch = batch or store.uid()[:12]
        job = store.enqueue(project["id"], "produce", {"resolution": resolution, "mode": "model", "batch": batch})
        return {"batch": batch, "project_id": project["id"], "title": name, "job_id": job["id"]}

    @app.post("/api/queue/cancel")
    def cancel_queue(body: QueueAction):
        return store.cancel_all(body.batch)

    @app.post("/api/queue/retry", status_code=202)
    def retry_queue(body: QueueAction):
        """Start again every failed or interrupted video (of one batch), from where each stopped."""
        with store.connection() as con:
            rows = con.execute("SELECT * FROM jobs WHERE kind='produce' ORDER BY created DESC LIMIT 1000").fetchall()
        latest = {}
        for row in rows:
            latest.setdefault(row["project_id"], row)
        retried = 0
        for row in latest.values():
            payload = json.loads(row["payload"])
            if row["status"] in ("failed", "interrupted") and (not body.batch or payload.get("batch") == body.batch):
                try:
                    store.enqueue(row["project_id"], "produce", payload)
                    retried += 1
                except (KeyError, ValueError):
                    pass      # the project was deleted or already has a job
        return {"retried": retried}

    @app.get("/api/queue")
    def queue():
        with store.connection() as con:
            rows = con.execute("SELECT jobs.*, json_extract(projects.body, '$.title') AS title, "
                               "json_extract(projects.body, '$.aspect') AS aspect, "
                               "json_extract(projects.body, '$.exports[0].path') AS video_path, "
                               "json_extract(projects.body, '$.exports[0].saved_to') AS saved_to, "
                               "json_extract(projects.body, '$.exports[0].duration') AS video_duration FROM jobs "
                               "JOIN projects ON projects.id=jobs.project_id ORDER BY jobs.created DESC LIMIT 400").fetchall()
        return [{"id": r["id"], "project_id": r["project_id"], "title": r["title"],
                 "kind": r["kind"], "status": r["status"], "progress": r["progress"],
                 "message": r["message"], "error": r["error"], "created": r["created"], "updated": r["updated"],
                 "batch": json.loads(r["payload"]).get("batch"), "aspect": r["aspect"],
                 # The finished video of a completed job: where it was saved and the copy inside the project.
                 "video": {"path": r["video_path"], "saved_to": r["saved_to"] or "", "duration": r["video_duration"]}
                 if r["status"] == "completed" and r["video_path"] else None} for r in rows]

    @app.post("/api/projects/{pid}/approve-verified")
    def approve_verified(pid: str):
        def change(project):
            media.require_narration_alignment(project)
            curation.approve_verified(project)
        return store.mutate(pid, change)

    @app.get("/api/projects/{pid}/shots/{sid}/captions")
    def shot_captions(pid: str, sid: str):
        p = store.get_project(pid)
        s = shot_by_id(p, sid)
        w, h = (1920, 1080) if p["aspect"] == "16:9" else (1080, 1920)
        offset = sum(shot["duration"] for shot in p["shots"][:p["shots"].index(s)]) if p.get("full_audio") else 0
        audio = p.get("full_audio") or s.get("audio")
        track = captions.cached(p, None if p.get("full_audio") else s)
        if track and p.get("full_audio"):
            offset = sum(captions.frame_duration(shot["duration"]) for shot in p["shots"][:p["shots"].index(s)])
        st = subtitle_style.style(w, h)
        index = p["shots"].index(s)
        timed = captions.segments(track, offset, s["duration"], w, h, narration=s["narration"],
                                  char_offset=captions.char_offset(p["shots"], index) if p.get("full_audio") else 0,
                                  line_chars=st["line_chars"]) if track else None
        _, segments = media.subtitles(s["narration"], s["duration"], w, h, s.get("overlay_label", ""),
                                      timed_segments=timed, style=st)
        return {"segments": segments, "duration": s["duration"], "audio_offset": offset,
                "audio_path": audio.get("path") if audio else None, "overlay_label": s.get("overlay_label", ""),
                "alignment": track["method"] if track else "proportional", "api_cost_cny": 0,
                # The preview draws the burned-in look: sizes are fractions of the frame height.
                "style": {"font": st["font"], "font_file": st["file"].name, "colour": st["css_colour"],
                          "size": st["size"] * subtitle_style.em_ratio(st["file"]) / h, "outline": st["outline"] / h,
                          "margin": st["margin_v"] / h}}

    @app.get("/api/projects/{pid}/film")
    def film_timeline(pid: str):
        """The whole video as the export will cut it: every shot's start, length, subtitle lines and audio,
        so the workbench can play the shots back to back before rendering."""
        p = store.get_project(pid)
        w, h = (1920, 1080) if p["aspect"] == "16:9" else (1080, 1920)
        st = subtitle_style.style(w, h)
        full = p.get("full_audio")
        full_track = captions.cached(p) if full else None
        aligned, alignment_note = True, ""
        if full:
            try:
                media.require_narration_alignment(p)
            except ValueError as exc:      # durations are still estimates: cuts would drift from the narration
                aligned, alignment_note, full_track = False, str(exc), None
        rows, begin = [], 0.0
        for index, s in enumerate(p["shots"]):
            audio = None if full else s.get("audio")
            duration = captions.frame_duration(float((audio or {}).get("duration") or s["duration"]))
            track = full_track if full else (captions.cached(p, s) if audio else None)
            timed = captions.segments(track, begin if full else 0, duration, w, h, narration=s["narration"],
                                      char_offset=captions.char_offset(p["shots"], index) if full else 0,
                                      line_chars=st["line_chars"]) if track else None
            _, segments = media.subtitles(s["narration"], duration, w, h, s.get("overlay_label", ""),
                                          timed_segments=timed, style=st)
            rows.append({"id": s["id"], "start": round(begin, 3), "duration": duration, "segments": segments,
                         "audio_path": audio.get("path") if audio else None,
                         "overlay_label": s.get("overlay_label", "")})
            begin += duration
        return {"total": round(begin, 3), "audio_path": full.get("path") if full else None,
                "aligned": aligned, "alignment_note": alignment_note,
                "alignment": full_track["method"] if full_track else ("per_shot" if any(r["audio_path"] for r in rows) else "none"),
                "shots": rows,
                "style": {"font": st["font"], "font_file": st["file"].name, "colour": st["css_colour"],
                          "size": st["size"] * subtitle_style.em_ratio(st["file"]) / h, "outline": st["outline"] / h,
                          "margin": st["margin_v"] / h}}

    @app.get("/api/fonts/{name}")
    def font_file(name: str):
        files = {f[0] for f in subtitle_style.FONTS.values()}
        path = paths.ROOT / "assets" / "fonts" / name
        if name not in files or not path.exists():
            raise HTTPException(404, "字体不存在")
        return FileResponse(path, headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/subtitle-styles")
    def subtitle_styles():
        have = subtitle_style.available()
        return {"fonts": [{"key": k, "label": v[2], "file": v[0], "available": have[k],
                           "em": subtitle_style.em_ratio(subtitle_style.font_file(k)) if have[k] else 0.69}
                          for k, v in subtitle_style.FONTS.items()],
                "sizes": [{"key": k, "label": v} for k, v in subtitle_style.SIZES.items()],
                "colours": [{"key": k, "label": v[2], "css": v[1]} for k, v in subtitle_style.COLOURS.items()]}

    @app.post("/api/demo", status_code=201)
    def new_demo():
        return demo.create()

    @app.post("/api/rewrite")
    def rewrite_script(body: RewriteRequest):
        """AI 洗稿: one rewrite and one fact-check request; the caller decides whether to keep it."""
        from . import rewrite
        if body.project_id:
            store.get_project(body.project_id)
        return rewrite.rewrite(body.text, title=body.title, minutes=body.minutes, aspect=body.aspect,
                               project_id=body.project_id or "rewrite")

    @app.get("/api/news")
    def news_list():
        from . import news, news_crawler
        return {"items": news.items(), "crawler": news_crawler.status()}

    @app.post("/api/news/link")
    def news_link(body: NewsLink):
        from . import news
        found, _ = news.add([news.fetch(body.url)], "link")
        if not found:
            raise ValueError("没有从页面中识别出正文，请直接复制新闻原文粘贴")
        return found[0]

    @app.post("/api/news/text")
    def news_text(body: NewsText):
        from . import news
        found, _ = news.add([body.model_dump()], "text")
        if not found:
            raise ValueError("新闻正文至少需要 30 个字")
        return found[0]

    @app.post("/api/news/crawl")
    def news_crawl(body: NewsCrawl | None = None):
        from . import news, news_crawler
        rows = news_crawler.crawl(koubo=bool(body and body.koubo))
        found, new = news.add(rows, "crawler")
        return {"fetched": len(rows), "new": new, "scripts": sum(1 for r in rows if r.get("koubo")),
                "notes": list(news_crawler.last_notes), "items": news.items()}

    @app.post("/api/news/import-latest")
    def news_import_latest():
        from . import news, news_crawler
        rows = news_crawler.import_latest()
        found, new = news.add(rows, "crawler")
        return {"fetched": len(rows), "new": new, "scripts": sum(1 for r in rows if r.get("koubo")), "items": news.items()}

    @app.delete("/api/news/{item_id}")
    def news_delete(item_id: str):
        from . import news
        news.remove(item_id)
        return {"ok": True}

    @app.post("/api/news/produce", status_code=202)
    def news_produce(body: NewsProduce):
        from . import news
        if not config.read()["model_api_key"]:
            raise ValueError("请先在服务与设置中填写模型 API Key")
        batch = store.uid()[:12] if body.auto and len(body.ids) > 1 else None
        return {"batch": batch, "projects": [news.make_project(i, aspect=body.aspect, minutes=body.minutes,
                                                               resolution=body.resolution, auto=body.auto, batch=batch)
                                             for i in body.ids]}

    @app.get("/api/library")
    def get_library(query: str = "", min_height: int = 720):
        if len(query) > 300 or min_height not in (0, 480, 720, 1080):
            raise ValueError("索引查询条件无效")
        with store.connection() as con:
            job = con.execute("SELECT id,status,progress,message,error FROM jobs WHERE kind='library' ORDER BY created DESC LIMIT 1").fetchone()
            row = con.execute("SELECT body FROM projects WHERE json_extract(body,'$.system_library')=1 LIMIT 1").fetchone()
        import json
        report = json.loads(row["body"]).get("library_report") if row else None
        return {"stats": library.stats(), "clips": library.search(query, min_height=min_height),
                "job": dict(job) if job else None, "report": report, "coverage": source_index.overview()}

    @app.get("/api/library/sources")
    def library_sources(query: str = "", topic: str = "", provider: str = "", entity: str = "", offset: int = 0):
        if (len(query) > 300 or (topic and topic not in source_index.TOPICS)
                or (provider and provider not in source_index.PROVIDERS) or len(entity)>32 or not 0 <= offset <= 500000):
            raise ValueError("来源索引查询条件无效")
        return source_index.search(query, topic=topic, provider=provider, entity=entity, offset=offset)

    @app.get("/api/library/entities")
    def library_entities(query: str = "", kind: str = ""):
        if len(query)>100 or kind not in ("", "person", "event"):
            raise ValueError("人物事件查询条件无效")
        return {"items": source_index.entity_search(query,kind,limit=500)}

    @app.post("/api/library/build", status_code=202)
    def build_library(body: LibraryBuild):
        if not set(body.index_topics) <= set(source_index.TOPICS):
            raise ValueError("未知索引主题")
        if body.action == "collect" and not (body.queries or body.urls):
            raise ValueError("请输入至少一个关键词或已接入来源的公开视频链接")
        for url in body.urls:
            library.source_url(url)
        return store.enqueue(library_collect.project_id(), "library", body.model_dump())

    @app.get("/api/library/clips/{clip_id}/{kind}")
    def library_file(clip_id: str, kind: str):
        if kind not in ("video", "thumbnail"):
            raise ValueError("素材库文件类型无效")
        clip = library.get(clip_id)
        target = library.file_path(clip["path" if kind == "video" else "thumbnail"])
        if not target.is_file():
            raise KeyError("索引文件缺失，请重新采集")
        return FileResponse(target, media_type="video/mp4" if kind == "video" else "image/jpeg")

    @app.post("/api/library/clips/{clip_id}/use")
    def use_library_clip(clip_id: str, body: LibraryUse):
        if any(j["status"] in ("queued", "running") for j in store.jobs(body.project_id)):
            raise ValueError("当前项目任务进行中，请完成后添加素材")
        asset = library.materialize(body.project_id, library.get(clip_id), internal=False)
        return {"asset_id": asset["id"], "project": store.get_project(body.project_id)}

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        p = store.get_project(pid)
        track = captions.cached(p) if p.get("full_audio") else None
        return p | {"composition": curation.composition(p), "model_costs": costs.summary(pid),
                    "caption_alignment": {"method": track["method"], "coverage": track["coverage"]} if track else None}

    @app.patch("/api/projects/{pid}")
    def patch_project(pid: str, body: ProjectPatch):
        values = body.model_dump(exclude_none=True)
        def change(p):
            if "script" in values and values["script"] != p["script"] and p["shots"]:
                p["script_changed"] = True
                for shot in p["shots"]:
                    shot.pop("editorial_contract", None)
                    curation.invalidate(shot)
            if "requirements" in values and values["requirements"] != p.get("requirements", ""):
                for shot in p["shots"]:
                    shot.pop("editorial_contract", None)
            if any(k in values and values[k] != p.get(k) for k in ("requirements", "min_video_height")):
                for shot in p["shots"]:
                    # Express picks are re-matched on the next produce (manual picks stay, awaiting a confirm).
                    if not curation.keep_express(p | values, shot, recheck=True):
                        shot.update(approved=False, verification=None, source_review_note="", source_review_url="")
            p.update(values)
        return store.mutate(pid, change)

    @app.patch("/api/projects/{pid}/shots/{sid}")
    def patch_shot(pid: str, sid: str, body: ShotPatch):
        values = body.model_dump(exclude_none=True)
        def change(p):
            s = shot_by_id(p, sid)
            if s["locked"] and set(values) != {"locked"}:
                raise ValueError("请先解锁镜头")
            if "narration" in values and values["narration"] != s["narration"]:
                if p.get("full_audio"):
                    raise ValueError("使用整段配音时请先移除配音，再修改口播")
                s["audio"] = None
            if s.get("audio") and "duration" in values and abs(values["duration"]-s["audio"]["duration"]) > .01:
                raise ValueError("逐镜配音决定实际时长，请更换配音或先修改口播移除旧音频，不能只缩短审核区间")
            changed = {k for k in values if values[k] != s.get(k)}
            before = deepcopy(s)
            review_changed = any(k in values and values[k] != s.get(k) for k in
                   ("narration", "visual", "subject", "location", "era", "exclusions", "overlay_label", "clip_start", "duration", "fit",
                    "persons", "event", "evidence_mode", "event_scope", "preferred_media"))
            if any(k in values and values[k] != s.get(k) for k in
                   ("narration", "visual", "subject", "location", "era", "exclusions", "query",
                    "persons", "event", "evidence_mode", "event_scope", "preferred_media")):
                s.pop("editorial_contract", None)
            if values.get("approved") and not s.get("selected_asset"):
                raise ValueError("请先选择素材")
            if "query" in values and values["query"] != s["query"]:
                s["search_queries"] = [values["query"]]
            if "visual" in values and values["visual"] != s.get("visual"):
                s["visual_origin"] = "manual"
            if "fit" in values and values["fit"] != s.get("fit"):
                s["fit_origin"] = "manual"
            if "overlay_label" in values and values["overlay_label"] != s.get("overlay_label"):
                s["overlay_label_origin"] = "manual"
            s.update(values)
            if review_changed and not curation.express_edit(p, s, before, changed):
                # The editor sends all fields, including stale manual evidence.
                # Invalidate after applying them so that old evidence cannot return.
                curation.invalidate(s)
            if values.get("approved"):
                media.require_narration_alignment(p)
            if "approved" in values and curation.express_result(s):
                # The user looked at it (keep it across re-timing), or took the confirmation back.
                s["verification"]["needs_source_check"] = not values["approved"]
            if values.get("approved") and not curation.identity_ready(s):
                raise ValueError("人物或事件依据不足。请核对原始来源，填写来源链接和至少 8 字的核对说明后再确认")
            if values.get("approved") and s.get("event_scope") == "background" and curation.needs_identity(s):
                if not re.search(r"资料|示意", s.get("overlay_label", "")):
                    raise ValueError("资料画面需要填写画面标注，例如：资料画面 · 非本场比赛")
        return store.mutate(pid, change)

    @app.post("/api/projects/{pid}/shots/{sid}/asset/{aid}")
    def choose_asset(pid: str, sid: str, aid: str):
        def change(p):
            shot = shot_by_id(p, sid)
            if shot["locked"]:
                raise ValueError("请先解锁镜头")
            if not any(a["id"] == aid and a["kind"] in ("image", "video") for a in p["assets"]):
                raise ValueError("素材不存在或类型不符")
            shot.update(selected_asset=aid, approved=False, verification=None, clip_start=0,
                        source_review_note="", source_review_url="")
            curation.manual_pick(p, shot, next(a for a in p["assets"] if a["id"] == aid))
        return store.mutate(pid, change)

    @app.post("/api/projects/{pid}/shots/{sid}/video-link")
    def add_video_link(pid: str, sid: str, body: RemoteImport):
        sources.public_url(body.url)
        source = body.source_url or body.url
        sources.public_url(source)
        candidate = sources.candidate("web_video", body.title, body.url, "", source,
                                      kind="video", license=body.license, download_status="unchecked")
        def change(p):
            shot = shot_by_id(p, sid)
            if shot["locked"]:
                raise ValueError("请先解锁镜头")
            if not any(c["id"] == candidate["id"] for c in shot["candidates"]):
                shot["candidates"].insert(0, candidate)
        return store.mutate(pid, change)

    @app.patch("/api/projects/{pid}/assets/{aid}")
    def patch_asset(pid: str, aid: str, body: AssetPatch):
        def change(p):
            a = next((a for a in p["assets"] if a["id"] == aid), None)
            if not a:
                raise KeyError("素材不存在")
            a.update(body.model_dump(exclude_none=True))
            for shot in p["shots"]:
                # Source/licence notes change no pixels: an express pick stays as it is.
                if shot.get("selected_asset") == aid and not curation.keep_express(p, shot):
                    shot.update(approved=False, verification=None, source_review_note="", source_review_url="")
        return store.mutate(pid, change)

    @app.post("/api/projects/{pid}/upload")
    def upload(pid: str, file: UploadFile = File(...), role: str = Form("asset"), shot_id: str = Form("")):
        store.get_project(pid)
        if role not in ("asset", "voice", "narration", "music", "source_audio"):
            raise ValueError("文件用途无效")
        # Acquire the project's edit check before doing expensive file work.
        store.mutate(pid, lambda p: None)
        suffix = __import__("pathlib").Path(file.filename or "").suffix.lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".mkv", ".webm", ".ogg",
                          ".wav", ".mp3", ".m4a", ".aac", ".flac", ".aiff"):
            raise ValueError("支持 JPG/PNG/WebP、常用视频及音频格式")
        path = paths.project_dir(pid) / "assets" / (store.uid()+suffix)
        path.parent.mkdir(exist_ok=True)
        try:
            size = 0
            with path.open("wb") as output:
                while chunk := file.file.read(1024*1024):
                    size += len(chunk)
                    if size > 500*1024*1024:
                        raise ValueError("单个文件最多 500 MB")
                    output.write(chunk)
            asset = media.inspect_asset(pid, path, file.filename or "导入素材")
            if role != "asset" and asset["kind"] not in ("audio", "video"):
                raise ValueError("请选择含音轨的音频或视频文件")
            if role != "asset":
                info = media.probe(paths.file_path(pid, asset["path"]))
                if not any(s["codec_type"] == "audio" for s in info["streams"]):
                    raise ValueError("文件没有音轨")
            def change(p):
                p["assets"].append(asset)
                audio = {"path": asset["path"], "duration": asset["duration"], "title": asset["title"]}
                if role == "voice":
                    s = shot_by_id(p, shot_id)
                    if s["locked"] or p.get("full_audio"):
                        raise ValueError("请先解锁镜头并移除整段配音")
                    if asset["duration"] > 120:
                        raise ValueError("逐镜音频最多 120 秒，长音频请选择整段配音")
                    retimed = abs(s["duration"] - asset["duration"]) > .01
                    s.update(audio=audio, duration=asset["duration"])
                    if retimed and not curation.keep_express(p, s):
                        curation.invalidate(s)
                elif role == "music":
                    p["music"] = audio
                elif role == "narration":
                    media.align_narration(p, audio)
                elif role == "source_audio":
                    # 音频成片: the produce job transcribes it, plans shots from the text and keeps it as the voice.
                    if p["shots"]:
                        raise ValueError("这个项目已经有分镜；音频成片请新建项目")
                    if not 3 <= asset["duration"] <= 3600:
                        raise ValueError("音频须在 3 秒到 60 分钟之间")
                    p.update(source_audio=audio, script="", script_original="", full_audio=None)
            return store.mutate(pid, change)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            file.file.close()

    @app.delete("/api/projects/{pid}/audio/{role}")
    def detach_audio(pid: str, role: str):
        if role not in ("music", "full_audio"):
            raise ValueError("音轨类型无效")
        return store.mutate(pid, lambda p: p.update({role: None}))

    @app.post("/api/projects/{pid}/jobs", status_code=202)
    def submit_job(pid: str, body: JobRequest):
        payload = body.model_dump(exclude_none=True)
        if body.kind in ("select", "preview", "verify") and not body.shot_id:
            raise ValueError("此操作需要指定分镜")
        if body.kind in ("select", "preview") and not body.candidate_id:
            raise ValueError("请选择候选素材")
        return store.enqueue(pid, body.kind, payload)

    @app.get("/api/projects/{pid}/jobs")
    def project_jobs(pid: str):
        store.get_project(pid)
        return store.jobs(pid)

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        return store.get_job(jid)

    @app.post("/api/jobs/{jid}/cancel")
    def cancel_job(jid: str):
        job = store.get_job(jid)
        if job["status"] == "queued":
            store.job_update(jid, status="cancelled", cancel=1, message="已取消")
        elif job["status"] == "running":
            store.job_update(jid, cancel=1, message="正在取消，等待当前操作结束")
        return store.get_job(jid)

    @app.post("/api/jobs/{jid}/retry", status_code=202)
    def retry_job(jid: str):
        old = store.get_job(jid)
        if old["status"] not in ("failed", "cancelled", "interrupted", "needs_review"):
            raise ValueError("此任务无需重试")
        payload = old["payload"]
        # Preserve completed audio, avoiding a second API charge after interruption.
        if old["kind"] == "voice":
            payload["replace"] = False
        return store.enqueue(old["project_id"], old["kind"], payload)

    @app.get("/api/projects/{pid}/files/{relative:path}")
    def get_file(pid: str, relative: str, download: bool = False):
        store.get_project(pid)
        path = paths.file_path(pid, relative)
        if not path.is_file():
            raise HTTPException(404, "文件不存在")
        return FileResponse(path, filename=path.name if download else None)

    frontend = paths.ROOT / "frontend" / "dist"
    if (frontend / "assets").exists():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

    @app.get("/{route:path}")
    def index(route: str):
        if route.startswith("api/"):
            raise HTTPException(404, "接口不存在")
        if not (frontend / "index.html").exists():
            return JSONResponse({"message": "请先在 frontend 执行 npm run build"}, status_code=503)
        # The page must be fetched again after an update (its scripts have hashed names and may be cached).
        return FileResponse(frontend / "index.html", headers={"Cache-Control": "no-cache"})

    return app
