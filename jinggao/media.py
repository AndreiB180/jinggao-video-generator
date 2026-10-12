from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from collections.abc import Callable
from functools import lru_cache

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import captions, config, curation, execution, paths, store


class Cancelled(Exception):
    pass


def run(args: list[str], *, cwd: Path | None = None, timeout: int = 600,
        check: Callable[[], None] | None = None) -> str:
    import tempfile
    # File-backed stderr avoids pipe deadlocks during long encodes.
    with tempfile.TemporaryFile() as output:
        proc = subprocess.Popen(args, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        start = time.monotonic()
        try:
            while proc.poll() is None:
                if check:
                    check()
                if time.monotonic()-start > timeout:
                    raise ValueError("媒体处理超时，请缩短镜头或使用较小素材")
                time.sleep(0.15)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        output.seek(0)
        result = output.read().decode("utf-8", errors="replace")
        if proc.returncode:
            # Keep diagnostics local and exclude command lines / credentials.
            raise ValueError("媒体处理失败：" + result[-1800:])
        return result


def probe(file: Path) -> dict:
    return json.loads(run([paths.binary("ffprobe"), "-v", "error", "-show_format", "-show_streams",
                           "-of", "json", str(file)], timeout=30))


def inspect_asset(project_id: str, path: Path, title: str, **metadata) -> dict:
    root = paths.project_dir(project_id)
    asset_id = store.uid()
    thumb = root / "thumbs" / f"{asset_id}.jpg"
    thumb.parent.mkdir(exist_ok=True)
    duration = 0
    try:
        with Image.open(path) as im:
            width, height = im.size
            if width * height > 80_000_000:
                raise ValueError("图片尺寸过大")
            im = ImageOps.exif_transpose(im).convert("RGB")
            width, height = im.size
            # Re-encode images to avoid browser handling arbitrary uploaded formats.
            canonical = path.with_suffix(".jpg")
            im.save(canonical, quality=94)
            if canonical != path:
                path.unlink()
                path = canonical
            im.thumbnail((720, 720))
            im.save(thumb, quality=84)
        kind = "image"
    except (OSError, Image.UnidentifiedImageError):
        info = probe(path)
        video = next((s for s in info["streams"] if s["codec_type"] == "video" and
                      not s.get("disposition", {}).get("attached_pic")), None)
        audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
        duration = float(info.get("format", {}).get("duration", 0))
        if not math.isfinite(duration) or duration <= 0 or duration > 7200:
            raise ValueError("媒体时长需要在 0–7200 秒之间") from None
        if video:
            kind, width, height = "video", int(video["width"]), int(video["height"])
            run([paths.binary("ffmpeg"), "-v", "error", "-y", "-ss", str(min(1, duration/2)),
                 "-i", str(path), "-frames:v", "1", "-vf", "scale=720:-2", str(thumb)], timeout=45)
        elif audio:
            kind, width, height = "audio", 0, 0
        else:
            raise ValueError("文件中没有有效的图片、视频或音轨") from None
        if path.suffix == ".media":
            formats = info.get("format", {}).get("format_name", "").split(",")
            suffix = ".webm" if "webm" in formats else ".mp4" if "mp4" in formats else ".ogg" if "ogg" in formats else ".bin"
            canonical = path.with_suffix(suffix)
            path.rename(canonical)
            path = canonical
    return dict(id=asset_id, title=title[:300], path=path.relative_to(root).as_posix(),
                thumbnail=thumb.relative_to(root).as_posix() if thumb.exists() else None,
                kind=kind, duration=duration, width=width, height=height,
                source_url=metadata.get("source_url", ""), license=metadata.get("license", "用户导入，待核实"),
                license_url=metadata.get("license_url", ""),
                author=metadata.get("author", ""), description=metadata.get("description", ""),
                provider=metadata.get("provider", "upload"))


def hardware() -> dict:
    result = {"ffmpeg": False, "nvenc": False, "gpu": "未检测到 NVIDIA GPU", "system": sys.platform}
    try:
        paths.binary("ffmpeg")
        paths.binary("ffprobe")
        result["ffmpeg"] = True
    except RuntimeError:
        return result
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            result["gpu"] = run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=10).strip()
        except ValueError:
            pass
    try:
        run([paths.binary("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "color=s=320x180:r=25",
             "-t", "0.1", "-c:v", "h264_nvenc", "-f", "null", "-"], timeout=15)
        result["nvenc"] = True
    except ValueError:
        pass
    return result


def synthesize(text: str, target: Path, check=None) -> None:
    from .tts_costs import speech_request
    import tempfile

    cfg = config.read()
    if cfg["tts_provider"] == "import":
        raise ValueError("当前选择导入配音。请上传整段音频或逐镜音频")
    if cfg["tts_provider"] == "compatible" and (not cfg["tts_api_key"] or not cfg["tts_model"]):
        raise ValueError("请先填写配音接口密钥和模型")
    with speech_request(cfg, text, check=check) as charge:
        target.parent.mkdir(parents=True, exist_ok=True)
        text_file = target.with_suffix(".txt")
        text_file.write_text(text, "utf-8")
        if cfg["tts_provider"] in {"baidu_web", "baidu_http"}:
            if cfg["tts_provider"] == "baidu_web":
                from .baidu_web import generate
            else:
                from .baidu_http import generate
            generate(text, target, check or (lambda: None), before_submit=charge.mark_sent)
            charge.success()
            return
        if cfg["tts_provider"] == "compatible":
            try:
                with httpx.Client(timeout=180) as client:
                    if check:
                        check()
                    charge.mark_sent()
                    r = client.post(cfg["tts_base_url"].rstrip("/")+"/audio/speech",
                                    headers={"Authorization": "Bearer "+cfg["tts_api_key"]},
                                    json=charge.payload)
                    charge.response(r)
                    r.raise_for_status()
                # A malformed/truncated response must not overwrite a previous
                # valid track; the provider charge still remains in the ledger.
                with tempfile.TemporaryDirectory(prefix=".speech-", dir=target.parent) as directory:
                    output = Path(directory) / "speech.wav"
                    output.write_bytes(r.content)
                    if float(probe(output).get("format", {}).get("duration", 0)) <= 0:
                        raise ValueError("配音接口返回的音轨没有有效时长")
                    if check:
                        check()
                    output.replace(target)
            except httpx.HTTPError:
                raise ValueError("配音接口调用失败，请检查接口兼容性、模型、音色及额度") from None
        elif os.name == "nt":
            charge.mark_sent()
            run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                 str(paths.ROOT / "scripts" / "sapi.ps1"), "-TextFile", str(text_file), "-OutputFile", str(target)],
                check=check)
            charge.success()
        elif sys.platform == "darwin":
            aiff = target.with_suffix(".aiff")
            charge.mark_sent()
            run(["/usr/bin/say", "-v", "Tingting", "-f", str(text_file), "-o", str(aiff)], check=check)
            run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(aiff), "-ar", "24000", str(target)], check=check)
            aiff.unlink(missing_ok=True)
            charge.success()
        else:
            raise ValueError("此系统暂不支持内置语音，请配置兼容配音接口或导入音频")


def _invalidate_narration_window(shot):
    curation.invalidate(shot)
    shot.pop("match_preview", None)
    shot.pop("match_result", None)
    for candidate in shot.get("candidates", []):
        for field in ("preview", "preview_paths", "preflight_status", "preflight_reason"):
            candidate.pop(field, None)


def align_narration(project: dict, audio: dict) -> None:
    """Save the track first; paid media review must wait for acoustic alignment."""
    if not project["shots"]:
        raise ValueError("请先创建分镜，再导入或生成整段配音")
    if any(s.get("locked") for s in project["shots"]):
        raise ValueError("整段配音需要重排镜头时长，请先解锁镜头")
    if not math.isfinite(float(audio["duration"])) or float(audio["duration"]) <= 0:
        raise ValueError("音频没有有效时长")
    for shot in project["shots"]:
        # Until acoustic alignment completes, even an unchanged old duration
        # cannot prove that the new audio uses the previously approved window.
        _invalidate_narration_window(shot)
        shot["audio"] = None
    if len(project["shots"]) == 1:
        # With no interior cuts the audio's measured total is the whole window;
        # semantic/audio validation still remains pending. Never split by text.
        duration = captions.frame_duration(audio["duration"])
        if 1 <= duration <= 120:
            project["shots"][0]["duration"] = duration
    project["full_audio"] = audio | {"alignment": "pending", "note": "音轨已保存；素材核验前将按实际语音对齐镜头"}


def _narration_state(project):
    audio = project.get("full_audio") or {}
    return {"audio": {key: audio.get(key) for key in ("path", "duration", "text")},
            "script_changed": bool(project.get("script_changed")),
            "shots": [{key: shot.get(key) for key in ("id", "narration", "duration", "locked", "audio")}
                      for shot in project["shots"]]}


def prepare_narration_alignment(project, check=lambda: None, progress=lambda *_: None):
    if project.get("script_changed"):
        raise ValueError("原稿已经修改，请重新规划后对齐音轨")
    expected = _narration_state(project)
    timing = captions.ensure_shot_timing(project, check, progress)
    return {"expected": expected, "timing": timing}


def apply_narration_alignment(project, proposal):
    """Validate every boundary before atomically invalidating changed windows."""
    if _narration_state(project) != proposal["expected"]:
        raise ValueError("对齐期间分镜或音轨发生变化，已保留音频，请重试")
    timing = proposal["timing"]
    source = captions._source(project)
    if source is None or source[-2] != timing["binding"]:
        raise ValueError("对齐期间音频内容发生变化，已保留音频，请重试")
    elapsed = 0.0
    for index, shot in enumerate(project["shots"]):
        old_duration = float(shot["duration"])
        if not math.isfinite(old_duration) or not 1 <= old_duration <= 120:
            raise ValueError("当前分镜时长无效，未改动音频或镜头")
        end = elapsed+old_duration
        if shot.get("locked") and (abs(elapsed-timing["frames"][index]/25) > 1e-6
                                  or abs(end-timing["frames"][index+1]/25) > 1e-6):
            raise ValueError("真实语音对齐会移动锁定镜头的起止位置，请先解锁；已保留音频")
        elapsed = end
    for shot, duration in zip(project["shots"], timing["durations"], strict=True):
        if abs(float(shot["duration"])-duration) > 1e-6:
            _invalidate_narration_window(shot)
        shot["duration"] = duration
    legacy = any(edge["method"] == "caption_edges" for edge in timing["boundaries"])
    note = "已按音轨字幕句段边界对齐至 25 帧/秒（复用旧缓存）" if legacy else "镜头已按实测语音边界对齐至 25 帧/秒"
    project["full_audio"].update(alignment="local_paraformer", alignment_info=timing, note=note)


def require_narration_alignment(project):
    """Read-only gate for approval/render paths; never run ASR in an API write."""
    if not project.get("full_audio"):
        return
    if project.get("script_changed"):
        raise ValueError("原稿已经修改，请重新规划后对齐音轨")
    track = captions.cached(project)
    timing = captions.shot_timing(project, track)
    if any(not math.isfinite(float(shot["duration"])) or abs(float(shot["duration"])-duration) > 1e-6
           for shot, duration in zip(project["shots"], timing["durations"], strict=True)):
        raise ValueError("镜头尚未按真实配音边界对齐，请先运行匹配或复核；原审核不能用于新时长")
    return track


def subtitle_parts(text: str, max_chars: int = 18) -> list[str]:
    return captions.parts(text, max_chars)


def stamps(seconds: float, ass=False) -> str:
    total = int(round(seconds * (100 if ass else 1000)))
    base = 100 if ass else 1000
    h, remain = divmod(total, 3600*base)
    m, remain = divmod(remain, 60*base)
    s, fraction = divmod(remain, base)
    return f"{h}:{m:02}:{s:02}.{fraction:02}" if ass else f"{h:02}:{m:02}:{s:02},{fraction:03}"


def subtitles(text: str, duration: float, w: int, h: int, label: str = "", *, timed_segments=None) -> tuple[str, list[tuple]]:
    size = 46 if w > h else 48
    size = round(size * min(w, h) / 1080)
    max_chars = 24 if w > h else 20
    if timed_segments is None:
        lines = subtitle_parts(text, max_chars * 2)
        weight = sum(max(len(x), 1) for x in lines)
        timed_segments, elapsed = [], 0.0
        for i, line in enumerate(lines):
            end = duration if i == len(lines)-1 else elapsed+duration*max(len(line), 1)/weight
            timed_segments.append((elapsed, end, captions.wrap(line, max_chars)))
            elapsed = end
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans SC,{size},&H00FFFFFF,&H00FFFFFF,&H00111111,&H60000000,0,0,0,0,100,100,0,0,1,2,1,2,40,40,{round(h*.075)},1
Style: Label,Noto Sans SC,{round(min(w,h)*.028)},&H00FFFFFF,&H00FFFFFF,&H60111111,&H60111111,0,0,0,0,100,100,0,0,3,8,0,9,36,36,36,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    segments = []
    if label:
        escaped_label = label.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ").replace("\r", " ")
        header += f"Dialogue: 1,0:00:00.00,{stamps(duration,True)},Label,,0,0,0,,{escaped_label}\n"
    for elapsed, end, line in timed_segments:
        # Remove ASS override syntax from untrusted narration.
        escaped = line.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", r"\N")
        header += f"Dialogue: 0,{stamps(elapsed,True)},{stamps(end,True)},Default,,0,0,0,,{escaped}\n"
        segments.append((elapsed, end, line))
    return header, segments


@lru_cache(maxsize=2)
def has_ass(ffmpeg: str) -> bool:
    return bool(re.search(r"\s+ass\s+V->V", run([ffmpeg, "-hide_banner", "-filters"], timeout=15)))


def subtitle_overlays(segments: list[tuple], w: int, h: int, work: Path, visual: str, label: str = "") -> tuple[list[str], str]:
    """Fallback for FFmpeg builds without libass, with the same caption timing."""
    size = round((46 if w > h else 48) * min(w, h) / 1080)
    font = ImageFont.truetype(str(paths.font_path()), size)
    strip_height = size * 3
    inputs, graph = [], [f"[0:v]{visual}[base0]"]
    for i, (begin, end, line) in enumerate(segments):
        overlay = Image.new("RGBA", (w, strip_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.multiline_text((w/2, strip_height/2), line, font=font, fill="white", anchor="mm", align="center",
                  stroke_width=max(1, round(size/16)), stroke_fill=(17, 17, 17, 255))
        name = f"caption-{i}.png"
        overlay.save(work / name)
        inputs += ["-i", name]
        y = h - round(h*.075) - strip_height//2 - size//2
        graph.append(f"[base{i}][{i+2}:v]overlay=0:{y}:enable='gte(t,{begin:.6f})*lt(t,{end:.6f})'[base{i+1}]")
    last = f"base{len(segments)}"
    if label:
        label_font = ImageFont.truetype(str(paths.font_path()), round(min(w,h)*.028))
        overlay = Image.new("RGBA", (w, 110), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        text = label.replace("\n", " ").replace("\r", " ")
        box = draw.textbbox((w-36, 36), text, font=label_font, anchor="rt")
        draw.rounded_rectangle((box[0]-12, box[1]-8, box[2]+12, box[3]+8), radius=5, fill=(17,17,17,180))
        draw.text((w-36, 36), text, font=label_font, fill="white", anchor="rt")
        overlay.save(work / "label.png")
        inputs += ["-i", "label.png"]
        graph.append(f"[{last}][{len(segments)+2}:v]overlay=0:0[labelled]")
        last = "labelled"
    graph.append(f"[{last}]format=yuv420p[captioned]")
    return inputs, ";".join(graph)


def render(project: dict, job_id: str, options: dict, progress, check) -> dict:
    shots = project["shots"]
    if not shots:
        raise ValueError("项目还没有分镜")
    if any(not s.get("selected_asset") for s in shots):
        raise ValueError("请为每个镜头选择素材后再导出")
    if any(not s.get("approved") for s in shots):
        raise ValueError("请逐镜确认画面后再导出")
    full_captions = require_narration_alignment(project)
    composition = curation.validate_export(project, image_motion=options.get("image_motion", True))
    full_audio = project.get("full_audio")
    if not options.get("allow_silent") and not full_audio and any(not s.get("audio") for s in shots):
        raise ValueError("还有镜头缺少配音。请先生成/导入音频，或明确勾选无配音预览")
    root = paths.project_dir(project["id"])
    out = root / "exports" / job_id
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "fonts").mkdir(exist_ok=True)
    # Packaged fonts may be read-only on Windows. The disposable copy must
    # remain writable so that cleanup cannot invalidate a completed export.
    shutil.copyfile(paths.font_path(), work / "fonts" / paths.font_path().name)
    w, h = (1920, 1080) if options.get("resolution", "1080p") == "1080p" else (1280, 720)
    if project["aspect"] == "9:16":
        w, h = h, w
    cfg = config.read()
    ass_available = has_ass(paths.binary("ffmpeg"))
    use_gpu = cfg["encoder"] != "cpu" and hardware()["nvenc"]
    if cfg["encoder"] == "nvenc" and not use_gpu:
        raise ValueError("NVIDIA 编码不可用，请检查驱动或改为自动/CPU")
    encode = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-b:v", "0"] if use_gpu else \
             ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-threads", "4"]
    assets = {a["id"]: a for a in project["assets"]}
    def caption_progress(fraction, message):
        progress(fraction * 4, message)
    timeline, source_list, elapsed, srt_rows = [], [], 0.0, []
    prepared = []
    for i, shot in enumerate(shots):
        check()
        asset = assets.get(shot["selected_asset"])
        if not asset or asset["kind"] not in ("image", "video"):
            raise ValueError(f"镜头 {i+1} 的素材已失效")
        source = paths.file_path(project["id"], asset["path"])
        duration = float(shot["duration"])
        audio = shot.get("audio")
        if full_audio:
            duration = float(shot["duration"])
        elif audio:
            duration = float(probe(paths.file_path(project["id"], audio["path"]))["format"]["duration"])
        if not 0 < duration <= 120:
            raise ValueError(f"镜头 {i+1} 时长超过 120 秒，请拆分")
        # Round up to complete video frames, and pad audio to exactly the same duration.
        duration = math.ceil(duration*25-1e-7)/25
        prepared.append((i, shot, asset, source, audio, duration, elapsed))
        elapsed += duration

    def render_shot(item):
        i, shot, asset, source, audio, duration, begin = item
        check()
        # Every concurrent encoder gets its own subtitle/overlay files.
        shot_work = work / f"shot-{i:04}"
        shot_work.mkdir(exist_ok=True)
        label = shot.get("overlay_label", "")
        track = full_captions or (captions.ensure(project, shot, check, caption_progress) if audio else None)
        timed = captions.segments(track, begin if full_captions else 0, duration, w, h) if track else None
        ass, segments = subtitles(shot["narration"], duration, w, h, label, timed_segments=timed)
        (shot_work / "subtitle.ass").write_text(ass, "utf-8-sig")
        visual = f"scale={w}:{h}:force_original_aspect_ratio="
        if shot.get("fit") == "contain":
            # Preserve all source pixels in the foreground. A dim, blurred
            # copy fills aspect-ratio gaps without cropping evidence or adding
            # unrelated images. Blur at quarter size to keep CPU export cheap.
            iw, ih = w, h
            if asset["kind"] == "image" and options.get("image_motion", True):
                iw, ih = int(w/1.06)//2*2, int(h/1.06)//2*2
            bw, bh = max(2, w//8*2), max(2, h//8*2)
            visual = (f"split=2[backdrop_input][foreground_input];"
                      f"[backdrop_input]scale={bw}:{bh}:force_original_aspect_ratio=increase,"
                      f"crop={bw}:{bh},boxblur=12:2,eq=brightness=-0.22:saturation=0.6,"
                      f"scale={w}:{h},setsar=1[backdrop];"
                      f"[foreground_input]scale={iw}:{ih}:force_original_aspect_ratio=decrease,"
                      f"setsar=1[foreground];"
                      f"[backdrop][foreground]overlay=(W-w)/2:(H-h)/2:shortest=1")
        else:
            visual += f"increase,crop={w}:{h}"
        if asset["kind"] == "image" and options.get("image_motion", True):
            # Supersample before subpixel zoom to keep slow photographic motion smooth.
            visual += (f",scale={w*2}:{h*2},zoompan=z='1+0.055*on/{max(1,round(duration*25))}':"
                       f"x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={w}x{h}:fps=25")
        visual += f",setsar=1,fps=25,tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration}"
        args = [paths.binary("ffmpeg"), "-v", "error", "-y"]
        if asset["kind"] == "image":
            args += ["-loop", "1", "-framerate", "25", "-i", str(source)]
        else:
            start = float(shot.get("clip_start", 0))
            if start >= asset["duration"]:
                raise ValueError(f"镜头 {i+1} 的截取起点超出素材时长")
            if project.get("media_policy", "mixed") != "mixed" and duration > asset["duration"]-start+.16:
                raise ValueError(f"镜头 {i+1} 视频短于实际配音，需补足片段后导出")
            args += ["-ss", str(start), "-i", str(source)]
        if full_audio:
            args += ["-ss", str(begin), "-i", str(paths.file_path(project["id"], full_audio["path"]))]
        elif audio:
            args += ["-i", str(paths.file_path(project["id"], audio["path"]))]
        else:
            args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        filename = f"clip-{i:04}.mp4"
        if ass_available:
            args += ["-map", "0:v:0", "-vf", visual+",ass=subtitle.ass:fontsdir=../fonts,format=yuv420p"]
        else:
            inputs, graph = subtitle_overlays(segments, w, h, shot_work, visual, label)
            args += inputs + ["-filter_complex_threads", "2", "-filter_complex", graph, "-map", "[captioned]"]
        # Encode full narration only once after concatenating the video. AAC
        # priming/padding on every cut otherwise introduces timing drift.
        args += (["-an"] if full_audio else
                 ["-map", "1:a:0", "-af", "apad", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"])
        args += ["-t", str(duration), *encode,
                 "-movflags", "+faststart", str(work / filename)]
        run(args, cwd=shot_work, check=check)
        return item, segments, filename, label

    workers = int(options.get("render_workers") or cfg.get("render_workers", "1"))
    progress(0, f"开始合成 {len(shots)} 镜 · {workers} 路并发 · {'NVIDIA' if use_gpu else 'CPU'}")
    for item, segments, filename, label in execution.ordered_map(render_shot, prepared, workers):
        i, shot, asset, source, audio, duration, begin = item
        check()
        progress((i+1)/len(shots)*85, f"已合成 {i+1}/{len(shots)} 镜 · {workers} 路并发 · {'NVIDIA' if use_gpu else 'CPU'}")
        timeline.append(f"file '{filename}'")
        if not full_captions:
            for begin, end, text in segments:
                srt_rows.append(f"{len(srt_rows)+1}\n{stamps(item[6]+begin)} --> {stamps(item[6]+end)}\n{text}\n")
        source_list.append({"shot": i+1, "narration": shot["narration"], "start": item[6],
                            "duration": duration, "clip_start": shot.get("clip_start", 0),
                            "title": asset["title"], "source_url": asset.get("source_url", ""),
                            "license": asset.get("license", ""), "author": asset.get("author", ""),
                            "license_url": asset.get("license_url", ""),
                            "kind": asset["kind"], "source_clip_start": None if asset.get("source_parts") else asset.get("source_clip_start", 0)+shot.get("clip_start", 0),
                            "source_parts": asset.get("source_parts"),
                            "source_evidence": asset.get("source_evidence"),
                            "derived_from_video": asset.get("derived_from_video"),
                            "privacy_redaction": asset.get("privacy_redaction"),
                            "source_review": {"url": shot.get("source_review_url", ""), "note": shot.get("source_review_note", "")},
                            "overlay_label": label, "image_motion": asset["kind"] == "image" and options.get("image_motion", True),
                            "verification": shot.get("verification")})
    check()
    (work / "concat.txt").write_text("\n".join(timeline), "utf-8")
    joined = work / "joined.mp4"
    run([paths.binary("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "concat.txt",
         "-c", "copy", str(joined)], cwd=work, check=check)
    final = out / "video.mp4"
    progress(92, "正在混音与整理导出文件")
    music = project.get("music")
    if full_audio:
        args = [paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(joined),
                "-i", str(paths.file_path(project["id"], full_audio["path"]))]
        if music:
            args += ["-stream_loop", "-1", "-i", str(paths.file_path(project["id"], music["path"])),
                     "-filter_complex", f"[1:a]apad[voice];[2:a]volume={options.get('music_volume',.12)}[bg];"
                     "[voice][bg]amix=inputs=2:duration=first:normalize=0[a]", "-map", "[a]"]
        else:
            args += ["-map", "1:a:0", "-af", "apad"]
        args += ["-map", "0:v:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2",
                 "-t", str(elapsed), "-movflags", "+faststart", str(final)]
        run(args, check=check)
    elif music:
        run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(joined), "-stream_loop", "-1",
             "-i", str(paths.file_path(project["id"], music["path"])), "-filter_complex",
             f"[1:a]volume={options.get('music_volume',.12)}[bg];[0:a][bg]amix=inputs=2:duration=first:normalize=0[a]",
             "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-t", str(elapsed),
             "-movflags", "+faststart", str(final)], check=check)
    else:
        run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(joined), "-c", "copy",
             "-movflags", "+faststart", str(final)], check=check)
    if full_captions:
        srt_rows = [f"{i+1}\n{stamps(c['start'])} --> {stamps(c['end'])}\n{captions.wrap(c['text'], 24 if w > h else 20)}\n"
                    for i, c in enumerate(full_captions["cues"])]
    (out / "subtitles.srt").write_text("\n".join(srt_rows), "utf-8-sig")
    (out / "sources.json").write_text(json.dumps(source_list, ensure_ascii=False, indent=2), "utf-8")
    (out / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), "utf-8")
    with zipfile.ZipFile(out / "delivery.zip", "w", zipfile.ZIP_STORED) as bundle:
        for name in ("video.mp4", "subtitles.srt", "sources.json", "project.json"):
            bundle.write(out / name, name)
    result = {"id": job_id, "path": final.relative_to(root).as_posix(),
              "bundle": (out / "delivery.zip").relative_to(root).as_posix(), "created": time.time(),
              "duration": elapsed, "width": w, "height": h, "encoder": "nvenc" if use_gpu else "cpu",
              "revision": project["revision"], "silent_preview": bool(options.get("allow_silent"))}
    result["image_motion"] = bool(options.get("image_motion", True))
    result["composition"] = composition
    result["caption_alignment"] = {"method": full_captions["method"], "coverage": full_captions["coverage"],
                                   "api_cost_cny": 0} if full_captions else {"method": "per_shot"}
    # Final outputs are durable; intermediate clips can be recreated on retry.
    def clear_readonly(function, path, error):
        if not isinstance(error, PermissionError):
            raise error
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        function(path)
    try:
        shutil.rmtree(work, onexc=clear_readonly)
    except OSError:
        # MP4 and delivery files are already durable. A cleanup problem must
        # not make the queue lose the finished export or charge for a retry.
        result["cleanup_warning"] = "成片已保存，临时渲染目录未能清理，可稍后关闭播放器后删除"
    return result
