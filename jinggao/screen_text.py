"""Text burned into footage: the uploader's contact details (加微信 / 看私信 / phone numbers), drama subtitles,
captions, title bars and scrolling lines. None of it belongs in our videos.

A small local OCR (PP-OCRv6 det + rec, Apache-2.0, run with onnxruntime; no OpenCV, nothing uploaded, free)
reads a few frames of each candidate and of each source's preview frames. classify() turns the lines into
decisions: contact/ad text rejects the footage outright, lines that sit where captions sit become marks the
safe crop cuts away (express_clean), dense text in the middle of the picture or a line that moves makes the
clip dirty so a cleaner one is preferred.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import re
import threading
import zipfile
from functools import lru_cache
from pathlib import Path

from PIL import Image

from . import paths

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy ships with the app
    np = None

WHEEL = ("https://files.pythonhosted.org/packages/59/07/203065283bba0c5e9c10af3602dc25ab3e59ee54dc2d1d22c8faeabf5ed8/"
         "rapidocr-3.10.0-py3-none-any.whl")
WHEEL_SHA256 = "2fc34e26cd0f48514804a94f8832283039c7334076bff60768b1fed9ad068050"
FILES = {
    "PP-OCRv6_det_small.onnx": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    "PP-OCRv6_rec_small.onnx": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
}
DET_SIDE = 960          # longest side for detection (a 16:9 frame becomes 960x544)
DET_THRESH, BOX_THRESH, UNCLIP = 0.3, 0.6, 1.6
REC_HEIGHT, MIN_SCORE = 48, 0.6


def model_dir() -> Path:
    return paths.DATA / "models" / "ocr"


def installed() -> bool:
    return np is not None and all((model_dir() / name).exists() for name in FILES)


def install_model(check=lambda: None) -> None:
    """Fetch the two public models once (from the hash-pinned RapidOCR wheel on PyPI)."""
    import httpx
    root = model_dir()
    root.mkdir(parents=True, exist_ok=True)
    if all((root / n).exists() and hashlib.sha256((root / n).read_bytes()).hexdigest() == d for n, d in FILES.items()):
        return
    buffer = io.BytesIO()
    with httpx.stream("GET", WHEEL, timeout=60, follow_redirects=True) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(1024 * 1024):
            check()
            buffer.write(chunk)
    if hashlib.sha256(buffer.getvalue()).hexdigest() != WHEEL_SHA256:
        raise ValueError("文字识别模型下载校验失败，请重试")
    with zipfile.ZipFile(buffer) as wheel:
        for name, digest in FILES.items():
            data = wheel.read(f"rapidocr/models/{name}")
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("文字识别模型校验失败：" + name)
            temporary = root / (name + ".part")
            temporary.write_bytes(data)
            temporary.replace(root / name)


@lru_cache(maxsize=1)
def _sessions(directory: str):
    """Detection and recognition sessions on the CPU (DirectML only with JINGGAO_OCR_GPU=1)."""
    import onnxruntime as ort
    root = Path(directory)

    def make(gpu: bool):
        options = ort.SessionOptions()
        # Many clips are read at once by different threads and videos: each run stays on two threads.
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False      # keeps each worker near 300 MB instead of growing with every frame
        options.enable_mem_pattern = False        # (DirectML requires this too)
        if gpu:
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        return (ort.InferenceSession(str(root / "PP-OCRv6_det_small.onnx"), options, providers=providers),
                ort.InferenceSession(str(root / "PP-OCRv6_rec_small.onnx"), options, providers=providers))

    # Off unless asked for: DirectML crashed a worker process outright (access violation inside onnxruntime)
    # in a 4-video batch on Windows, and reading text is not where videos wait anyway.
    gpu = os.environ.get("JINGGAO_OCR_GPU") == "1" and "DmlExecutionProvider" in ort.get_available_providers()
    try:
        det, rec = make(gpu)
    except Exception:
        if not gpu:
            raise
        gpu = False
        det, rec = make(False)
    chars = rec.get_modelmeta().custom_metadata_map["character"].splitlines()
    return det, rec, ["blank"] + chars + [" "], threading.Lock() if gpu else None


def _run(session, feeds, lock):
    """One DirectML session is used by one thread at a time; CPU sessions run in parallel."""
    if lock is None:
        return session.run(None, feeds)
    with lock:
        return session.run(None, feeds)


def on_gpu() -> bool:
    try:
        return installed() and _sessions(str(model_dir()))[3] is not None
    except Exception:
        return False


def _bgr(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)[:, :, ::-1]


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, list]]:
    """Connected regions of a binary map as (x0, y0, x1, y1, runs); runs are (row, start, end) pieces.
    Union-find over horizontal runs: text maps are sparse, so this stays quick without OpenCV."""
    parent: list[int] = []
    runs: list[tuple[int, int, int]] = []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    previous: list[int] = []
    for y in range(mask.shape[0]):
        row = mask[y]
        if not row.any():
            previous = []
            continue
        padded = np.concatenate(([False], row, [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        current = []
        for start, end in zip(edges[::2], edges[1::2], strict=True):
            index = len(runs)
            runs.append((y, int(start), int(end)))
            parent.append(index)
            for other in previous:
                _, s, e = runs[other]
                if s <= end and start <= e:          # touching, 8-connected
                    a, b = find(index), find(other)
                    if a != b:
                        parent[b] = a
            current.append(index)
        previous = current
    groups: dict[int, list] = {}
    for index, run in enumerate(runs):
        groups.setdefault(find(index), []).append(run)
    result = []
    for pieces in groups.values():
        ys = [p[0] for p in pieces]
        result.append((min(p[1] for p in pieces), min(ys), max(p[2] for p in pieces), max(ys) + 1, pieces))
    return result


def detect(image: Image.Image) -> list[tuple[float, float, float, float, float]]:
    """Text line boxes as (x0, y0, x1, y1, score) in pixels of image."""
    det, _, _, lock = _sessions(str(model_dir()))
    w, h = image.size
    ratio = min(1.0, DET_SIDE / max(w, h))
    rw, rh = max(32, int(round(w * ratio / 32)) * 32), max(32, int(round(h * ratio / 32)) * 32)
    pixels = _bgr(image.resize((rw, rh), Image.BILINEAR))
    tensor = ((pixels / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)
    prob = _run(det, {det.get_inputs()[0].name: tensor}, lock)[0][0, 0]
    mask = prob > DET_THRESH
    grown = mask.copy()                     # 2x2 dilation, as PaddleOCR does
    grown[1:, :] |= mask[:-1, :]
    grown[:, 1:] |= grown[:, :-1].copy()
    sx, sy = w / rw, h / rh
    boxes = []
    for x0, y0, x1, y1, pieces in _components(grown):
        bw, bh = x1 - x0, y1 - y0
        if min(bw, bh) < 3:
            continue
        score = float(np.mean(np.concatenate([prob[y, s:e] for y, s, e in pieces])))
        if score < BOX_THRESH:
            continue
        grow = bw * bh * UNCLIP / (2 * (bw + bh))
        boxes.append((max(0.0, (x0 - grow) * sx), max(0.0, (y0 - grow) * sy),
                      min(float(w), (x1 + grow) * sx), min(float(h), (y1 + grow) * sy), score))
    return sorted(boxes, key=lambda b: (round(b[1] / 10), b[0]))


def recognise(image: Image.Image, boxes) -> list[tuple[str, float]]:
    """Read each box (horizontal lines; a tall narrow box is read as a column of characters)."""
    _, rec, chars, lock = _sessions(str(model_dir()))
    if not boxes:
        return []
    crops = []
    for x0, y0, x1, y1, *_ in boxes:
        crop = image.crop((int(x0), int(y0), max(int(x0) + 1, int(math.ceil(x1))), max(int(y0) + 1, int(math.ceil(y1)))))
        if crop.height >= crop.width * 1.5:
            crop = crop.rotate(90, expand=True)
        crops.append(crop)
    results: list[tuple[str, float] | None] = [None] * len(crops)
    order = sorted(range(len(crops)), key=lambda i: crops[i].width / crops[i].height)
    for start in range(0, len(order), 8):
        batch = order[start:start + 8]
        widest = max(320 / REC_HEIGHT, max(crops[i].width / crops[i].height for i in batch))
        width = int(REC_HEIGHT * widest)
        tensor = np.zeros((len(batch), 3, REC_HEIGHT, width), dtype=np.float32)
        for row, i in enumerate(batch):
            crop = crops[i]
            resized_w = min(width, int(math.ceil(REC_HEIGHT * crop.width / crop.height)))
            pixels = _bgr(crop.resize((max(1, resized_w), REC_HEIGHT), Image.BILINEAR))
            tensor[row, :, :, :resized_w] = ((pixels / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
        out = _run(rec, {rec.get_inputs()[0].name: tensor}, lock)[0]
        index, prob = out.argmax(axis=2), out.max(axis=2)
        for row, i in enumerate(batch):
            keep = np.ones(index.shape[1], dtype=bool)
            keep[1:] = index[row][1:] != index[row][:-1]
            keep &= index[row] != 0
            text = "".join(chars[k] for k in index[row][keep] if k < len(chars))
            results[i] = (text, float(prob[row][keep].mean()) if keep.any() else 0.0)
    return [r or ("", 0.0) for r in results]


def read(image: Image.Image) -> list[dict]:
    """Lines of text in a picture: [{"text", "score", "box": [x0, y0, x1, y1] as fractions}]. Callers hold the
    shared "frames" slot, which bounds how many run at once across every video."""
    boxes = detect(image)
    recognised = recognise(image, boxes)
    w, h = image.size
    lines = []
    for box, (text, score) in zip(boxes, recognised, strict=True):
        text = text.strip()
        if text and score >= MIN_SCORE:
            lines.append({"text": text, "score": round(score, 3),
                          "box": [round(box[0] / w, 4), round(box[1] / h, 4), round(box[2] / w, 4), round(box[3] / h, 4)]})
    return lines


# ------------------------------------------------------------------ decisions
# Contact details and selling: never acceptable, wherever they are (uploaders of "素材" videos put them over
# the picture: 需要视频私信 / 加微信 / QQ numbers / phone numbers).
CONTACT = re.compile(r"微信|威信|薇信|徽信|[Vv]信|(?<![A-Za-z])(?:[VvWw][Xx])(?![A-Za-z])|加[Vv](?![A-Za-z])|私信|私聊|私我|"
                     r"签名|扣扣|(?<![A-Za-z])[Qq]{2}(?![A-Za-z])|需要视频|需要素材|抖音号|快手号|公众号|直播间|二维码|扫码|"
                     r"联系方式|联系我|再联系|请联系|欢迎联系|需要联系|加我|商务合作|淘宝|天猫|拼多多|店铺")
# Words that only mean selling or contact when a number stands next to them (OCR also misreads 水电站 as 水电话).
WITH_NUMBER = re.compile(r"电话|热线|手机|价格|费用|收费|报价|包邮|购买|下单|订房|订餐")   # not 元: menu boards are scenery
MOBILE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)|(?<![\d.])0\d{2,3}-\d{7,8}(?!\d)")   # mobile; landline with area code
DIGITS = re.compile(r"(?<!\d)\d{7,12}(?!\d)")    # QQ and phone numbers; longer runs are prices, IDs, addresses
# Container numbers, B站 default user names, and the numbers of a market screen (the client lost every crypto
# chart to "contact details": prices 1.63333389100586, wallet addresses 0x1a03…, dates, times, percentages).
NOT_CONTACT = re.compile(r"[A-Z]{3}[UJZ]\s?\d{6,7}|bili[_ ]?\d+|0x[0-9a-f]{6,}|\d+[.,]\d+%?|\d+%|"
                         r"(?<!\d)(19|20)\d{2}[年/.-]\d{1,2}([月/.-]\d{1,2}日?)?|\d{1,2}月\d{1,2}日?|\d{1,2}:\d{2}(:\d{2})?", re.I)
CJK = re.compile(r"[㐀-鿿]")
TOP, BOTTOM = 0.22, 0.62      # within the picture: above TOP is the title/watermark band, below BOTTOM captions


def _cjk(text: str) -> int:
    return len(CJK.findall(text))


def contact(line: dict) -> bool:
    text = line["text"]
    plain = NOT_CONTACT.sub("", text.replace(" ", ""))
    if CONTACT.search(text) or MOBILE.search(plain):
        return True
    if WITH_NUMBER.search(plain) and re.search(r"\d{4,}", plain):
        return True
    digits = DIGITS.search(plain)
    if digits:
        # 8+ digits anywhere (QQ, phone); 7 near the edges, where uploaders put their IDs (TC8888802).
        return len(digits.group(0)) >= 8 or line["box"][3] <= 0.2 or line["box"][1] >= 0.8
    return False


def _relative(box, content):
    x0, y0, x1, y1 = content
    h = max(1e-6, y1 - y0)
    return (box[1] - y0) / h, (box[3] - y0) / h


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def classify(frames: list[list[dict]], content=(0.0, 0.0, 1.0, 1.0)) -> dict:
    """frames: read() of a few frames of one window (or of a source's previews).

    Returns {"contact": [texts], "marks": [{"kind", "box"}], "dense": bool, "moving": bool}: marks are
    caption lines (kind "caption", cut away or blurred), title/watermark text at the top (kind "logo") and
    text across the middle of the picture (kind "center"; the clip counts as dirty)."""
    found, marks, middle = [], [], []
    for lines in frames:
        mid_lines = []
        for line in lines:
            if line["score"] < 0.7:
                continue
            if contact(line):
                found.append(line["text"])
                marks.append({"kind": "center", "box": line["box"]})
                continue
            cjk, text = _cjk(line["text"]), line["text"]
            words = len(text.split())
            top, bottom = _relative(line["box"], content)
            if bottom <= TOP and (cjk >= 2 or len(text) >= 4):
                marks.append({"kind": "logo", "box": line["box"]})
            elif top >= BOTTOM and (cjk >= 2 or words >= 3):
                marks.append({"kind": "caption", "box": line["box"]})
            elif cjk >= 2:
                mid_lines.append(line)
        # Middle of the picture: a shop sign or a short label is scenery; a long line, several lines or a
        # big title is an overlay (花字, formula boards, news title bars, info cards).
        total = sum(_cjk(l["text"]) for l in mid_lines)
        if mid_lines and (total >= 10 or len(mid_lines) >= 3 or max(_cjk(l["text"]) for l in mid_lines) >= 6):
            marks.append({"kind": "center", "box": _union([l["box"] for l in mid_lines])})
            middle.append(True)
    moving = _moving(frames)
    if moving:
        marks.append({"kind": "center", "box": _union(moving)})
    return {"contact": sorted(set(found)), "marks": marks, "dense": bool(middle), "moving": bool(moving)}


def _moving(frames) -> list:
    """A Chinese line seen in two frames at clearly different x and the same height: scrolling text."""
    from difflib import SequenceMatcher
    seen = [[l for l in lines if l["score"] >= 0.7 and _cjk(l["text"]) >= 4] for lines in frames]
    boxes = []
    for i, a_lines in enumerate(seen):
        for b_lines in seen[i + 1:]:
            for a in a_lines:
                for b in b_lines:
                    shift = abs((a["box"][0] + a["box"][2]) - (b["box"][0] + b["box"][2])) / 2
                    if (shift >= 0.06 and abs(a["box"][1] - b["box"][1]) <= 0.03
                            and SequenceMatcher(None, a["text"], b["text"]).ratio() >= 0.8):
                        boxes += [a["box"], b["box"]]
    return boxes


def frames_of(path, duration: float, fractions=(0.3, 0.75), width: int = 960) -> list[Image.Image]:
    """Colour frames of a local clip for reading: one FFmpeg run, the frames stacked into one picture."""
    import subprocess
    from . import media
    info = next(s for s in media.probe(path)["streams"] if s.get("codec_type") == "video")
    w, h = int(info["width"]), int(info["height"])
    if w > width:
        w, h = width, max(2, round(h * width / w / 2) * 2)
    args = [paths.binary("ffmpeg"), "-v", "error"]
    for fraction in fractions:
        args += ["-threads", "1", "-ss", f"{duration * fraction:.2f}", "-i", str(path)]
    scaled = ";".join(f"[{i}:v:0]scale={w}:{h},setsar=1,format=rgb24[f{i}]" for i in range(len(fractions)))
    graph = scaled + ";" + "".join(f"[f{i}]" for i in range(len(fractions))) + \
        (f"vstack=inputs={len(fractions)}[out]" if len(fractions) > 1 else "null[out]")
    args += ["-filter_complex", graph, "-map", "[out]", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(args, capture_output=True, timeout=60).stdout
    size = w * h * 3
    return [Image.frombytes("RGB", (w, h), raw[i * size:(i + 1) * size]) for i in range(len(raw) // size)]
