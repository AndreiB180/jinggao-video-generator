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

from . import captions, config, curation, execution, express_clean, gpu, paths, store, subtitle_style, throttle


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


def _ocr_gpu() -> bool:
    """Text reading can use the graphics card (DirectML build of onnxruntime, Windows package)."""
    try:
        import onnxruntime
        return os.environ.get("JINGGAO_OCR_GPU") == "1" and "DmlExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def hardware() -> dict:
    """FFmpeg and the NVIDIA card as checked at start (gpu.detect), plus what this computer offers for CPU work."""
    result = {"ffmpeg": False, "system": sys.platform, "cores": throttle.cores(),
              "cpu_slots": throttle.LIMITS["cpu"], "gpu_slots": throttle.LIMITS["gpu"],
              "memory_gb": round(throttle.memory_gb(), 1), "memory_videos": throttle.memory_videos(),
              "cpu_videos": throttle.cpu_videos(), "max_videos": throttle.max_videos(), "ocr_gpu": _ocr_gpu()}
    try:
        paths.binary("ffmpeg")
        paths.binary("ffprobe")
        result["ffmpeg"] = True
    except RuntimeError:
        return result | {"nvenc": False, "full": False, "gpu": "未检测到 NVIDIA GPU"}
    state = gpu.status()
    return result | state | {"gpu": state.get("gpu") or "未检测到 NVIDIA GPU", "summary": gpu.summary(state)}


def synthesize(text: str, target: Path, check=None) -> None:
    from .tts_costs import speech_request
    import tempfile

    cfg = config.read()
    if cfg["tts_provider"] == "edge_tts":
        from . import edge_voice
        edge_voice.synthesize(text, target, check)     # free: nothing to record in the ledger
        return
    if cfg["tts_provider"] == "qwen_tts":
        from . import qwen_tts
        qwen_tts.synthesize(text, target, check)      # one charge per request part, inside
        target.with_suffix(".txt").write_text(text, "utf-8")
        return
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
            run(["/usr/bin/say", "-v", _say_voice(), "-f", str(text_file), "-o", str(aiff)], check=check)
            run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(aiff), "-ar", "24000", str(target)], check=check)
            aiff.unlink(missing_ok=True)
            charge.success()
        else:
            raise ValueError("此系统暂不支持内置语音，请配置兼容配音接口或导入音频")


SUBTITLE_FOLDER = "外挂字幕（仅上传平台时用，播放时不要加载）"


def _windows_videos() -> Path | None:
    """The user's 视频 folder as Windows knows it, which may have been moved (to D:\\ or OneDrive)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        import uuid
        from ctypes import wintypes
        folder_id = uuid.UUID("18989B1D-99B5-455B-841C-AB7C74E4DDFC")      # FOLDERID_Videos
        guid = (ctypes.c_ubyte * 16).from_buffer_copy(folder_id.bytes_le)
        found = ctypes.c_wchar_p()
        shell = ctypes.windll.shell32
        shell.SHGetKnownFolderPath.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.HANDLE,
                                               ctypes.POINTER(ctypes.c_wchar_p)]
        if shell.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(found)) != 0:
            return None
        try:
            return Path(found.value) if found.value else None
        finally:
            ctypes.windll.ole32.CoTaskMemFree(found)
    except Exception:
        return None


def output_folder() -> Path:
    """Where finished videos are copied: the chosen folder, else 视频/镜稿成片 in the user's home."""
    chosen = config.read().get("export_dir", "").strip()
    if chosen:
        return Path(chosen).expanduser()
    for videos in (_windows_videos(), Path.home() / ("Movies" if sys.platform == "darwin" else "Videos")):
        if videos and videos.is_dir():
            return videos / "镜稿成片"
    return Path.home() / "镜稿成片"


def finished_file(project: dict, export_id: str | None = None) -> Path:
    """A project's finished video on disk: the copy in the output folder if it is still there, else the export
    inside the project."""
    exports = project.get("exports") or []
    export = next((e for e in exports if e.get("id") == export_id), None) if export_id else (exports[0] if exports else None)
    if not export:
        raise ValueError("这个项目还没有成片")
    saved = Path(export["saved_to"]) if export.get("saved_to") else None
    if saved and saved.is_file():
        return saved
    inner = paths.project_dir(project["id"]) / export["path"]
    if not inner.is_file():
        raise ValueError("成片文件已被移动或删除")
    return inner


def _windows_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def open_path(target: Path) -> None:
    """Open a file with its default program (a video plays in the system player) or a folder in the file manager."""
    if os.name == "nt":
        os.startfile(str(target))  # noqa: S606 - the app's own files on the user's machine
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


def reveal(target: Path) -> None:
    """Show the file selected in Explorer / Finder."""
    if os.name == "nt":
        # Windows file names cannot contain quotes, so the path is safe inside them.
        subprocess.Popen(f'explorer /select,"{target}"')
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target.parent)])


def choose_folder(initial: str = "") -> str | None:
    """The system's folder picker; returns the chosen folder, or None when cancelled or unavailable."""
    if os.name == "nt":
        # The chosen path comes back as Base64 of its UTF-8 bytes: a Chinese Windows console would otherwise
        # hand over GBK text.
        script = ("Add-Type -AssemblyName System.Windows.Forms;"
                  "$owner=New-Object System.Windows.Forms.Form -Property @{TopMost=$true;ShowInTaskbar=$false};"
                  "$d=New-Object System.Windows.Forms.FolderBrowserDialog;"
                  "$d.Description='选择成片保存位置';$d.ShowNewFolderButton=$true;"
                  "if($env:JINGGAO_FOLDER){$d.SelectedPath=$env:JINGGAO_FOLDER};"
                  "if($d.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK){"
                  "[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($d.SelectedPath))}")
        command = ["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", script]
    elif sys.platform == "darwin":
        command = ["osascript", "-e", 'POSIX path of (choose folder with prompt "选择成片保存位置")']
    else:
        return None
    try:
        done = subprocess.run(command, capture_output=True, timeout=900, creationflags=_windows_flags(),
                              env=os.environ | {"JINGGAO_FOLDER": initial})
    except (OSError, subprocess.SubprocessError):
        return None
    chosen = done.stdout.decode("utf-8", errors="replace").strip().splitlines()
    if done.returncode != 0 or not chosen:
        return None
    if os.name == "nt":
        import base64
        try:
            return base64.b64decode(chosen[-1].strip()).decode("utf-8") or None
        except ValueError:
            return None
    return chosen[-1].strip() or None


def export_name(project: dict) -> str:
    """The film's file name: the first sentence of the script (the hook the video opens with), else the
    project title."""
    script = re.sub(r"[*#>“”\"'‘’]", "", str(project.get("script") or "")).strip()
    found = re.match(r"[^。！？!?\n]*[！？!?]?", script)
    first = (found.group(0) if found else "").strip(" ，,、；;：:")
    return first or str(project.get("title") or "")


def deliver(project: dict, result: dict) -> str:
    """Copy a finished export to the output folder as <title>.mp4; the subtitle file goes one level down so
    players that auto-load a same-named .srt do not show the subtitles twice."""
    if config.read().get("auto_save_exports", "1") != "1":
        return ""
    folder = output_folder()
    folder.mkdir(parents=True, exist_ok=True)
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", export_name(project)).strip(" .")[:60] or "成片"
    stem, n = name, 2
    while (folder / f"{stem}.mp4").exists():
        stem, n = f"{name}（{n}）", n + 1
    root = paths.project_dir(project["id"])
    video = folder / f"{stem}.mp4"
    shutil.copyfile(root / result["path"], video)
    srt = (root / result["path"]).with_name("subtitles.srt")
    if srt.exists():
        (folder / SUBTITLE_FOLDER).mkdir(exist_ok=True)
        shutil.copyfile(srt, folder / SUBTITLE_FOLDER / f"{stem}.srt")
    try:
        from . import cover
        cover.start(project, video, folder, stem, project.get("headline") or name)
    except Exception:   # noqa: BLE001 — the cover is an extra; the film is saved either way
        pass
    return str(video)


def _invalidate_narration_window(shot, project=None):
    if project is not None and curation.keep_express(project, shot):
        # Express picks keep their footage across a re-timed narration; produce re-approves
        # them (approve_verified) unless the clip no longer covers the new duration.
        shot["approved"] = False
    else:
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
    if len(project["shots"]) == 1:
        # With no interior cuts the audio's measured total is the whole window;
        # semantic/audio validation still remains pending. Never split by text.
        duration = captions.frame_duration(audio["duration"])
        if 1 <= duration <= 120:
            project["shots"][0]["duration"] = duration
    for shot in project["shots"]:
        # Until acoustic alignment completes, even an unchanged old duration
        # cannot prove that the new audio uses the previously approved window.
        _invalidate_narration_window(shot, project)
        shot["audio"] = None
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
        changed = abs(float(shot["duration"])-duration) > 1e-6
        shot["duration"] = duration
        if changed:
            _invalidate_narration_window(shot, project)
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


def subtitles(text: str, duration: float, w: int, h: int, label: str = "", *, timed_segments=None,
              style: dict | None = None) -> tuple[str, list[tuple]]:
    """ASS subtitles: one line at a time in the configured font, size and colour (subtitle_style)."""
    st = style or subtitle_style.style(w, h)
    line_chars = st["line_chars"]
    if timed_segments is None:
        lines = subtitle_parts(text, line_chars)
        weight = sum(max(len(x), 1) for x in lines)
        timed_segments, elapsed = [], 0.0
        for i, line in enumerate(lines):
            end = duration if i == len(lines)-1 else elapsed+duration*max(len(line), 1)/weight
            timed_segments.append((elapsed, end, captions.wrap(line, line_chars)))
            elapsed = end
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{st["family"]},{st["size"]},{st["primary"]},&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,1,{st["outline"]},{st["shadow"]},2,{round(w*.04)},{round(w*.04)},{st["margin_v"]},1
Style: Label,Noto Sans SC,{round(min(w,h)*.028)},&H00FFFFFF,&H00FFFFFF,&H60111111,&H60111111,0,0,0,0,100,100,0,0,3,8,0,9,36,36,36,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    segments = []
    if label:
        escaped_label = label.replace("\\", "＼").replace("{", "｛").replace("}", "｝").replace("\n", " ").replace("\r", " ")
        header += f"Dialogue: 1,0:00:00.00,{stamps(duration,True)},Label,,0,0,0,,{escaped_label}\n"
    for elapsed, end, line in timed_segments:
        # Untrusted narration: ASS override syntax is escaped; glyphs the style font lacks use the fallback.
        header += f"Dialogue: 0,{stamps(elapsed,True)},{stamps(end,True)},Default,,0,0,0,,{subtitle_style.ass_text(line, st)}\n"
        segments.append((elapsed, end, line))
    return header, segments


@lru_cache(maxsize=2)
def has_ass(ffmpeg: str) -> bool:
    return bool(re.search(r"\s+ass\s+V->V", run([ffmpeg, "-hide_banner", "-filters"], timeout=15)))


def subtitle_overlays(segments: list[tuple], w: int, h: int, work: Path, visual: str, label: str = "") -> tuple[list[str], str]:
    """Fallback for FFmpeg builds without libass, with the same caption timing."""
    st = subtitle_style.style(w, h)
    size = st["size"]
    font = ImageFont.truetype(str(st["file"]), size)
    strip_height = size * 3
    inputs, graph = [], [f"[0:v]{visual}[base0]"]
    for i, (begin, end, line) in enumerate(segments):
        overlay = Image.new("RGBA", (w, strip_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.multiline_text((w/2, strip_height/2), line, font=font, fill=st["css_colour"], anchor="mm", align="center",
                  stroke_width=st["outline"], stroke_fill=(0, 0, 0, 255))
        name = f"caption-{i}.png"
        overlay.save(work / name)
        inputs += ["-i", name]
        y = h - st["margin_v"] - strip_height//2 - size//2
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


@lru_cache(maxsize=1)
def _say_voice() -> str:
    """The Mandarin voice is "Tingting" on older macOS and "Ting-Ting" on newer releases."""
    try:
        voices = subprocess.run(["/usr/bin/say", "-v", "?"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return "Tingting"
    return next((v for v in ("Tingting", "Ting-Ting") if re.search(rf"^{v}\s", voices, re.M)), "Tingting")


def _derived_from(assets: dict, asset: dict, target) -> bool:
    """Is asset the target file itself or a privacy-redacted copy of it?"""
    seen = set()
    while asset and asset["id"] not in seen:
        if asset["id"] == target:
            return True
        seen.add(asset["id"])
        asset = assets.get((asset.get("privacy_redaction") or {}).get("original_asset_id"))
    return False


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
    w, h = (1920, 1080) if options.get("resolution", "1080p") == "1080p" else (1280, 720)
    if project["aspect"] == "9:16":
        w, h = h, w
    caption_style = subtitle_style.style(w, h)
    for font in {paths.font_path(), caption_style["file"], caption_style["fallback_file"]}:
        shutil.copyfile(font, work / "fonts" / font.name)
    cfg = config.read()
    ass_available = has_ass(paths.binary("ffmpeg"))
    use_gpu = gpu.mode("render") == "nvenc"
    if cfg["encoder"] == "nvenc" and not use_gpu:
        state = gpu.status()
        raise ValueError("NVIDIA 编码不可用：" + (state.get("advice") or state.get("reason") or "请检查驱动或改为自动/CPU"))
    encoders = {}
    assets = {a["id"]: a for a in project["assets"]}
    def caption_progress(fraction, message):
        progress(fraction * 4, message)
    timeline, source_list, elapsed, srt_rows = [], [], 0.0, []
    burn = options.get("subtitles")
    burn = project.get("burn_subtitles", True) if burn is None else bool(burn)
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
        timed = captions.segments(track, begin if full_captions else 0, duration, w, h, narration=shot["narration"],
                                  char_offset=captions.char_offset(shots, i) if full_captions else 0,
                                  line_chars=caption_style["line_chars"]) if track else None
        ass, segments = subtitles(shot["narration"], duration, w, h, label, timed_segments=timed, style=caption_style)
        drawn = segments
        if not burn:
            # No narration on the picture (the SRT keeps the timed lines); a card's name label stays.
            ass, _ = subtitles("", duration, w, h, label, style=caption_style)
            drawn = []
        (shot_work / "subtitle.ass").write_text(ass, "utf-8-sig")
        visual = f"scale={w}:{h}:force_original_aspect_ratio="
        # Express crops were measured on one downloaded file (or its privacy-processed copy); a different
        # asset chosen later renders uncropped. Shots saved before crop_asset existed keep their crop.
        measured = asset["kind"] == "video" and (shot.get("crop_asset") is None
                                                 or _derived_from(assets, asset, shot["crop_asset"]))
        crop = float(shot.get("crop_bottom") or 0)
        prefix = f"crop=iw:trunc(ih*{1-crop:.3f}/2)*2:0:0," if measured and 0 < crop < .3 else ""
        if measured and shot.get("frame_crop"):
            # Express footage: blur leftover logos, then show the watermark/subtitle-free rectangle.
            clean = express_clean.video_filter(shot["frame_crop"])
            prefix = clean + "," if clean else prefix
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
            # Supersample before subpixel zoom to keep slow photographic motion smooth. Pictures take turns at
            # pushing in, pulling out and panning, so a run of them still looks like footage, not a slideshow.
            n = max(1, round(duration * 25))
            centre = "x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2'"
            motion = [f"z='1+0.13*on/{n}':{centre}", f"z='1.13-0.13*on/{n}':{centre}",
                      f"z='1.13':x='(iw-iw/zoom)*on/{n}':y='ih/2-ih/zoom/2'",
                      f"z='1.13':x='(iw-iw/zoom)*(1-on/{n})':y='ih/2-ih/zoom/2'"][i % 4]
            visual += f",scale={w*2}:{h*2},zoompan={motion}:d=1:s={w}x{h}:fps=25"
        visual = prefix + visual
        visual += f",setsar=1,fps=25,tpad=stop_mode=clone:stop_duration={duration},trim=duration={duration}"
        args = [paths.binary("ffmpeg"), "-v", "error", "-y", "-filter_threads", "2"]
        if asset["kind"] == "image":
            args += ["-loop", "1", "-framerate", "25", "-i", str(source)]
        else:
            start = float(shot.get("clip_start", 0))
            if start >= asset["duration"]:
                raise ValueError(f"镜头 {i+1} 的截取起点超出素材时长")
            if project.get("media_policy", "mixed") != "mixed" and duration > asset["duration"]-start+.16:
                raise ValueError(f"镜头 {i+1} 视频短于实际配音，需补足片段后导出")
            args += ["-threads", "2", "-ss", str(start), "-i", str(source)]
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
            inputs, graph = subtitle_overlays(drawn, w, h, shot_work, visual, label)
            args += inputs + ["-filter_complex_threads", "2", "-filter_complex", graph, "-map", "[captioned]"]
        # Encode full narration only once after concatenating the video. AAC
        # priming/padding on every cut otherwise introduces timing drift.
        args += (["-an"] if full_audio else
                 ["-map", "1:a:0", "-af", "apad", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"])
        def encode(way):
            run(args + ["-t", str(duration), *(gpu.NVENC_RENDER if way == "nvenc" else gpu.X264_RENDER),
                        "-movflags", "+faststart", str(work / filename)], cwd=shot_work, check=check)
        # Slots are shared by every video being made: NVENC sessions on the card, CPU encodes on the cores.
        way = gpu.mode("render")
        if way == "nvenc":
            with throttle.slot("gpu", check):
                try:
                    encode("nvenc")
                    gpu.worked("nvenc")
                except ValueError as exc:     # the card refused this shot: the CPU makes it below
                    gpu.failed("nvenc", exc)
                    way = "cpu"
        if way == "cpu":
            with throttle.slot("cpu", check):
                encode("cpu")
        return item, segments, filename, label, way

    workers = int(options.get("render_workers") or cfg.get("render_workers", "1"))
    progress(0, f"开始合成 {len(shots)} 镜 · {workers} 路并发 · {'NVIDIA' if use_gpu else 'CPU'}")
    for item, segments, filename, label, way in execution.ordered_map(render_shot, prepared, workers):
        i, shot, asset, source, audio, duration, begin = item
        encoders[way] = encoders.get(way, 0) + 1
        check()
        progress((i+1)/len(shots)*85, f"已合成 {i+1}/{len(shots)} 镜 · {workers} 路并发 · {'NVIDIA' if use_gpu else 'CPU'}")
        timeline.append(f"file '{filename}'")
        for begin, end, text in segments:   # the SRT carries exactly the lines burned into the picture
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
    (out / "subtitles.srt").write_text("\n".join(srt_rows), "utf-8-sig")
    (out / "sources.json").write_text(json.dumps(source_list, ensure_ascii=False, indent=2), "utf-8")
    (out / "project.json").write_text(json.dumps(project, ensure_ascii=False, indent=2), "utf-8")
    with zipfile.ZipFile(out / "delivery.zip", "w", zipfile.ZIP_STORED) as bundle:
        for name in ("video.mp4", "subtitles.srt", "sources.json", "project.json"):
            bundle.write(out / name, name)
    result = {"id": job_id, "path": final.relative_to(root).as_posix(),
              "bundle": (out / "delivery.zip").relative_to(root).as_posix(), "created": time.time(),
              "duration": elapsed, "width": w, "height": h,
              "encoder": "nvenc" if encoders.get("nvenc") and not encoders.get("cpu") else "cpu" if not encoders.get("nvenc") else "mixed",
              "encoders": encoders,
              "revision": project["revision"], "silent_preview": bool(options.get("allow_silent")), "subtitles": burn}
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
