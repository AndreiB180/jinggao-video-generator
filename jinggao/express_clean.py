"""Find burned-in overlays (channel logos, uploader watermarks, subtitles) and crop them away.

Platform footage often carries a TV-station bug, an uploader watermark or hard
subtitles. The free preview sprites span the whole source: in the median of
many different pictures the scenery washes out while anything printed at a fixed
place (logos, lower thirds, the subtitle line) stays sharp. Each mark found that
way is confirmed in the downloaded window itself before it costs any picture.
Bilibili's own upload watermark is not in the sprites, so the corners of the
window are also checked for strokes that never change while the picture moves.
The render shows the largest output-shaped rectangle that avoids every mark
within a zoom limit; logos still inside it are blurred, and marks in the middle
of the picture make the clip "dirty" so the matcher prefers a cleaner one.
"""
from __future__ import annotations

import base64
import io
import subprocess
from functools import lru_cache

from PIL import Image

from . import paths, throttle

try:
    import numpy as np
except ImportError:  # detection is an enhancement; without numpy the render stays uncropped
    np = None

WIDTH = 480          # analysis width; sprites from Bilibili are 480x270 or 160x90
MAX_ZOOM = 1.42      # never magnify the normal cover crop by more than this
CELL = 8
LOGO_MATCH = 0.62    # normalised correlation; watermarked clips score 0.68-0.98, clean ones <= 0.52

# Bilibili upload watermark logo ("bilibili"), 46x96 grey, as rendered on 720-line video.
LOGO_PNG = (
    "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"
    "Ji+FliYAAAAASUVORK5CYII="
)


def available() -> bool:
    return np is not None


def _gray(image: Image.Image, size) -> np.ndarray:
    return np.asarray(image.convert("L").resize(size, Image.BILINEAR), dtype=np.float32)


def clip_frames(path, duration: float, size, count: int = 10) -> list:
    """Evenly spaced grey frames of a local clip at the analysis size."""
    w, h = size
    rate = count / max(0.5, duration)
    with throttle.slot("frames"):    # shared by every video being made: the computer stays responsive
        raw = subprocess.run(
            [paths.binary("ffmpeg"), "-v", "error", "-threads", "2", "-i", str(path), "-vf",
             f"fps={rate:.4f},scale={w}:{h}:flags=area,format=gray", "-frames:v", str(count),
             "-f", "rawvideo", "-"], capture_output=True, timeout=60, check=True).stdout
    n = len(raw) // (w * h)
    return [np.frombuffer(raw[i * w * h:(i + 1) * w * h], dtype=np.uint8).reshape(h, w).astype(np.float32)
            for i in range(n)]


def _edges(frame, threshold) -> np.ndarray:
    gx = np.zeros_like(frame)
    gy = np.zeros_like(frame)
    gx[:, 1:-1] = np.abs(frame[:, 2:] - frame[:, :-2])
    gy[1:-1, :] = np.abs(frame[2:, :] - frame[:-2, :])
    return np.maximum(gx, gy) > threshold


def _dilate(mask) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    grown = out.copy()
    grown[:, 1:] |= out[:, :-1]
    grown[:, :-1] |= out[:, 1:]
    return grown


def content_rect(frames) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) fractions without black letterbox/pillarbox bars."""
    stack = np.stack(frames)
    h, w = stack.shape[1:]
    # Any bright pixel in a row makes it picture: dark night scenes are not bars. Subtitles printed inside
    # a bar keep that bar as content, where subtitles() finds and crops them.
    rows = np.percentile(stack.max(axis=2), 90, axis=0)
    cols = np.percentile(stack.max(axis=1), 90, axis=0)
    def span(profile, n):
        live = np.nonzero(profile > 40)[0]
        if not len(live):
            return 0, n
        a, b = int(live[0]), int(live[-1]) + 1
        # Bars must be margins at both ends to count; otherwise keep the full frame. They may be wide: a
        # vertical video posted inside a 16:9 frame leaves about a third black on each side (the client got
        # those half-black pictures while the limit was 30%).
        return (a, b) if a < n * 0.42 and b > n * 0.58 else (0, n)
    y0, y1 = span(rows, h)
    x0, x1 = span(cols, w)
    pad = 2
    y0, y1 = (y0 + pad if y0 else 0), (y1 - pad if y1 < h else h)
    x0, x1 = (x0 + pad if x0 else 0), (x1 - pad if x1 < w else w)
    return x0 / w, y0 / h, x1 / w, y1 / h


def _straight(med, border, cols) -> bool:
    """A canvas border is a straight vertical line: in most rows the strongest horizontal step near it sits
    in the same column (a bright subject on a dark sky has a ragged edge instead)."""
    lo, hi = max(1, border - 6), min(cols - 1, border + 6)
    if hi - lo < 4:
        return False
    steps = np.abs(med[:, lo + 1:hi + 1] - med[:, lo - 1:hi - 1])        # (rows, hi-lo), step at column lo..hi-1
    strong = steps.max(axis=1) > 6
    if strong.mean() < 0.5:
        return False
    where = steps.argmax(axis=1)[strong] + lo
    centre = np.median(where)
    return float((np.abs(where - centre) <= 2).mean()) >= 0.7


def _picture_run(pixel, cols, med=None):
    """(a, b) columns of the footage in a frame-in-frame layout, judged by one per-pixel measure, or None."""
    score = pixel.mean(axis=0)
    threshold = 0.45 * np.percentile(score, 85)
    inside = score > threshold
    runs, start, gap = [], None, 0          # runs of picture columns, gaps of up to 3 columns bridged
    for k, flag in enumerate(inside):
        if flag:
            start = k if start is None else start
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 3:
                runs.append((start, k - gap + 1))
                start, gap = None, 0
    if start is not None:
        runs.append((start, cols - gap))
    if not runs:
        return None
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if not 0.22 * cols <= b - a <= 0.8 * cols:
        return None
    outside = np.concatenate([inside[:max(0, a - 3)], inside[b + 3:]])
    if len(outside) < 0.15 * cols or outside.mean() > 0.12:
        return None
    for border, canvas, footage in ((a, slice(max(0, a - 7), max(0, a - 2)), slice(a + 2, a + 7)),
                                    (b, slice(b + 2, b + 7), slice(max(0, b - 7), max(0, b - 2)))):
        if border in (0, cols) or canvas.stop - canvas.start < 3 or canvas.start >= cols:
            continue
        plain, sharp = pixel[:, canvas].mean(axis=1), pixel[:, footage].mean(axis=1)
        if ((plain * 2 + 1) < sharp).mean() < 0.55:
            return None
        if med is not None and not _straight(med, border, cols):
            return None
    return a, b


def inner_picture(frames, content) -> tuple[float, float, float, float] | None:
    """A picture set inside a larger canvas: a vertical video uploaded as landscape, or a template with the
    uploader's name and logo beside the footage. Returns its (x0, y0, x1, y1) fractions, or None.

    Every column gets a "picture" score (detail of the median frame plus change over time). The footage is
    one contiguous run of high-scoring columns covering 22-80% of the width, the canvas outside it scores
    low, and right at each border the canvas side is plainer than the picture side in most rows (a dark,
    still or blurred canvas meets sharp footage along a straight line)."""
    stack = np.stack(frames)
    n, h, w = stack.shape
    cx0, cy0 = round(content[0] * w), round(content[1] * h)
    cx1, cy1 = round(content[2] * w), round(content[3] * h)
    region = stack[:, cy0:cy1, cx0:cx1]
    rows, cols = region.shape[1:]
    if n < 3 or cols < 60 or rows < 40:
        return None
    med = np.median(region, axis=0)
    detail = np.zeros_like(med)
    detail[1:, :] += np.abs(np.diff(med, axis=0))
    detail[:, 1:] += np.abs(np.diff(med, axis=1))
    # A dark or still canvas differs from footage in detail and in change over time; a blurred copy of the
    # video moves like the footage, so there detail alone tells them apart.
    change = region.std(axis=0)
    found = _picture_run(detail + change, cols, med)          # dark, still or text canvas: straight borders
    if not found:
        found = _picture_run(detail, cols)                    # blurred canvas: soft borders, but it moves
        if found:
            a, b = found
            inner = change[:, a:b].mean()
            outer = np.concatenate([change[:, :max(0, a - 3)], change[:, b + 3:]], axis=1)
            if not outer.size or outer.mean() < 0.5 * inner:
                found = None
    if not found:
        return None
    a, b = found
    pad = 2
    x0 = (cx0 + a + (pad if a else 0)) / w
    x1 = (cx0 + b - (pad if b < cols else 0)) / w
    return round(x0, 4), round(cy0 / h, 4), round(x1, 4), round(cy1 / h, 4)


def layout_jump(frames, content) -> bool:
    """The canvas changes during the clip: the source cuts from its intro card (name and logo beside the
    picture) to the footage, or the picture moves to another place. On screen that is a jump, not a cut."""
    half = len(frames) // 2
    if half < 3:
        return False
    first, second = inner_picture(frames[:half], content), inner_picture(frames[half:], content)
    if bool(first) == bool(second) and (not first or (abs(first[0] - second[0]) <= 0.05
                                                       and abs(first[2] - second[2]) <= 0.05)):
        return False
    # The canvas around the picture must really change between the halves (an intro card giving way to the
    # footage); a scene that only looks like a picture on a canvas for a moment stays the same around it.
    picture = first or second
    stack = np.stack(frames)
    h, w = stack.shape[1:]
    diff = np.abs(np.median(stack[:half], axis=0) - np.median(stack[half:], axis=0))
    x0, x1 = int(picture[0] * w), int(picture[2] * w)
    canvas = np.concatenate([diff[:, :max(0, x0 - 2)], diff[:, x1 + 2:]], axis=1)
    return canvas.size > 0 and float(canvas.mean()) > 18


def static_bands(frames, content) -> list[dict]:
    """Still bands of text or graphics across the top or bottom of moving footage (a news channel's data bar,
    a title strip): boxes for the crop to avoid. A band must be still while the picture below or above it
    moves, carry detail (not sky or a plain bar), and end at a straight horizontal edge."""
    stack = np.stack(frames)
    n, h, w = stack.shape
    cx0, cy0 = round(content[0] * w), round(content[1] * h)
    cx1, cy1 = round(content[2] * w), round(content[3] * h)
    region = stack[:, cy0:cy1, cx0:cx1]
    rows, cols = region.shape[1:]
    if n < 3 or rows < 40:
        return []
    activity = region.std(axis=0).mean(axis=1)                          # per row
    detail = np.abs(np.diff(np.median(region, axis=0), axis=1)).mean(axis=1)
    edge = (np.abs(region[:, 2:, :] - region[:, :-2, :]) > 16).mean(axis=2).min(axis=0)   # per row boundary
    out = []
    for top in (True, False):
        limit = int(rows * 0.25)
        span = range(limit) if top else range(rows - 1, rows - 1 - limit, -1)
        body = activity[limit:rows - limit].mean()
        if body < 3:
            break                                                       # the whole picture is still
        depth = 0
        for r in span:
            if activity[r] < 0.3 * body:
                depth += 1
            else:
                break
        if depth < 0.04 * rows:
            continue
        band = slice(0, depth) if top else slice(rows - depth, rows)
        boundary = depth - 1 if top else rows - depth - 2
        near = edge[max(0, boundary - 2):boundary + 3]
        if detail[band].mean() < 4 or not len(near) or near.max() < 0.6:
            continue
        y0 = cy0 + (0 if top else rows - depth)
        y1 = cy0 + (depth if top else rows)
        out.append({"kind": "banner", "box": [round(cx0 / w, 4), round(y0 / h, 4), round(cx1 / w, 4), round(y1 / h, 4)]})
    return out


def _components(cells) -> list[tuple[int, int, int, int, int]]:
    """8-connected components of a boolean grid -> (x0, y0, x1, y1, count) in cells."""
    seen = np.zeros_like(cells, dtype=bool)
    out = []
    rows, cols = cells.shape
    for r in range(rows):
        for c in range(cols):
            if not cells[r, c] or seen[r, c]:
                continue
            stack, members = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                members.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and cells[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            ys = [m[0] for m in members]
            xs = [m[1] for m in members]
            out.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(members)))
    return out


def _boxes_from_mask(mask, min_cells=3, kind="logo") -> list[dict]:
    h, w = mask.shape
    gh, gw = h // CELL, w // CELL
    cells = mask[:gh * CELL, :gw * CELL].reshape(gh, CELL, gw, CELL).sum(axis=(1, 3)) >= 5
    # Join strokes of one logo that sit a cell apart.
    joined = cells.copy()
    joined[:, 1:] |= cells[:, :-1]
    joined[:, :-1] |= cells[:, 1:]
    joined &= np.logical_or(cells, np.roll(cells, 1, axis=1) & np.roll(cells, -1, axis=1))
    boxes = []
    for x0, y0, x1, y1, n in _components(joined):
        box = [x0 * CELL / w, y0 * CELL / h, x1 * CELL / w, y1 * CELL / h]
        if n < min_cells or (box[2] - box[0]) * (box[3] - box[1]) > 0.2:
            continue
        boxes.append({"kind": kind, "box": [round(v, 4) for v in box]})
    return boxes


def _edge_distance(box, content=(0, 0, 1, 1)) -> float:
    return min(box[0] - content[0], box[1] - content[1], content[2] - box[2], content[3] - box[3])


def _distinct(frames) -> list:
    """Drop near-repeats so the median is taken across genuinely different pictures."""
    kept, signatures = [], []
    for frame in frames:
        small = np.asarray(Image.fromarray(frame.astype(np.uint8)).resize((16, 9), Image.BILINEAR), dtype=np.float32)
        if all(np.abs(small - other).mean() > 12 for other in signatures):
            kept.append(frame)
            signatures.append(small)
    return kept


def _join_line(boxes) -> list[dict]:
    """Thin or italic strokes of one mid-picture name break into fragments: join those on the same line."""
    boxes = [dict(b) for b in boxes]
    joined = True
    while joined:
        joined = False
        for i, a in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                A, B = a["box"], boxes[j]["box"]
                if max(_edge_distance(A), _edge_distance(B)) <= 0.12:
                    continue   # edge marks are cropped away whole anyway
                overlap = min(A[3], B[3]) - max(A[1], B[1])
                gap = max(A[0], B[0]) - min(A[2], B[2])
                if overlap >= 0.5 * min(A[3] - A[1], B[3] - B[1]) and gap <= 0.05:
                    a["box"] = [min(A[0], B[0]), min(A[1], B[1]), max(A[2], B[2]), max(A[3], B[3])]
                    a["joined"] = True
                    del boxes[j]
                    joined = True
                    break
            if joined:
                break
    return boxes


def profile(images: list, size: tuple[int, int]) -> dict:
    """Source-wide fixed marks from preview frames (PIL images of one source)."""
    if np is None:
        return {}
    frames = _distinct([_gray(im, size) for im in images])
    if len(frames) < 10:
        return {"frames": len(frames), "boxes": [], "size": size}
    median = np.median(np.stack(frames), axis=0)
    steady = np.stack([_dilate(_edges(f, 12)) for f in frames]).mean(axis=0) >= 0.5
    mask = _inside(_edges(median, 40) & steady, content_rect(frames))
    def texty(box):
        # Mid-picture marks need many distinct frames before the median can be trusted, and must look like
        # a line of text (an uploader's "@name"), not a fragment of a face or a plate rim. With many distinct
        # frames the median leaves nothing but overlays, so a short name ("月色梦光影") is enough.
        x0, y0, x1, y1 = box
        aspect = (x1 - x0) * size[0] / max(1e-6, (y1 - y0) * size[1])
        return len(frames) >= 20 and ((x1 - x0 >= 0.12 and aspect >= 2.5)
                                      or (len(frames) >= 40 and x1 - x0 >= 0.08 and aspect >= 1.8))
    pieces = _boxes_from_mask(mask)
    boxes = []
    for b in _join_line(pieces):
        joined = b.pop("joined", False)
        if joined and (len(frames) < 40 or (_edge_distance(b["box"]) > 0.12 and not texty(b["box"]))):
            # A shape pieced together from fragments needs the cleaner median of many distinct frames and must
            # still read as one line of text; otherwise judge the fragments on their own.
            boxes += [dict(f) for f in pieces if all(b["box"][k] <= f["box"][k] for k in (0, 1))
                      and all(f["box"][k] <= b["box"][k] for k in (2, 3))]
        else:
            boxes.append(b)
    for b in list(boxes):
        x0, y0, x1, y1 = b["box"]
        b["central"] = _edge_distance(b["box"]) > 0.12
        if not b["central"] and not _overlay_position(b["box"], size, len(frames)):
            b["pattern"] = None   # not an overlay position: likely fixed scenery
            continue
        if b["central"] and not texty(b["box"]):
            b["pattern"] = None
            continue
        b["pattern"] = mask[int(y0 * size[1]):int(y1 * size[1]) + 1, int(x0 * size[0]):int(x1 * size[0]) + 1]
    return {"frames": len(frames), "boxes": boxes, "size": size}


def _present(box, pattern, frames, size) -> bool:
    """Is a source-wide mark actually on screen during this window?"""
    if pattern is None or pattern.sum() < 8:
        return False
    x0, y0 = int(box[0] * size[0]), int(box[1] * size[1])
    hits = []
    for f in frames:
        region = f[y0:y0 + pattern.shape[0], x0:x0 + pattern.shape[1]]
        if region.shape != pattern.shape:
            return False
        hits.append((_dilate(_edges(region, 12)) & pattern).sum() / pattern.sum())
    return sum(1 for h in hits if h >= 0.6) >= max(2, 0.3 * len(hits))


def _corner(box, content) -> bool:
    x0, y0, x1, y1 = content
    return ((box[0] < x0 + 0.3 * (x1 - x0) or box[2] > x1 - 0.3 * (x1 - x0))
            and (box[1] < y0 + 0.16 * (y1 - y0) or box[3] > y1 - 0.16 * (y1 - y0)))


def _inside(mask, content, band: int = 4):
    """Keep mask pixels inside the content only, away from the bar/picture boundary (a steady edge)."""
    h, w = mask.shape
    x0, y0, x1, y1 = content
    keep = np.zeros_like(mask)
    top, bottom = int(y0 * h) + (band if y0 > 0 else 0), int(round(y1 * h)) - (band if y1 < 1 else 0)
    left, right = int(x0 * w) + (band if x0 > 0 else 0), int(round(x1 * w)) - (band if x1 < 1 else 0)
    keep[max(0, top):max(0, bottom), max(0, left):max(0, right)] = True
    return mask & keep


def _tight_corner(box) -> bool:
    """Hugging a side and the top or bottom: where stations and stock sites put their marks."""
    x0, y0, x1, y1 = box
    return (x0 <= .06 or x1 >= .94) and (y0 <= .2 or y1 >= .8) and (x1 - x0) * (y1 - y0) <= .05


def _overlay_position(box, size, frames: int) -> bool:
    """Where overlays sit: a tight corner, a top or bottom band, a vertical strip or panel on a side, or a
    wide line of text. Small marks part-way down a side are usually scenery that never moved (a gym)."""
    x0, y0, x1, y1 = box
    w, h = (x1 - x0) * size[0], (y1 - y0) * size[1]
    return (_tight_corner(box) or y1 <= .22 or y0 >= .78
            or ((x0 <= .06 or x1 >= .94) and x1 - x0 <= .1 and h >= 1.8 * w)
            or ((x0 <= .02 or x1 >= .98) and y1 - y0 >= .3)
            or (frames >= 20 and x1 - x0 >= .12 and w >= 2.5 * h))


def _logo_shaped(box, size) -> bool:
    """Station logos and upload marks are short strips (or narrow vertical text along a side), never a tall
    block inside the picture."""
    x0, y0, x1, y1 = box
    if y1 - y0 <= 0.16 and (x1 - x0) * size[0] >= 0.8 * (y1 - y0) * size[1]:
        return True
    return x1 - x0 <= 0.1 and (x0 <= 0.12 or x1 >= 0.88) and y1 - y0 <= 0.45


def corner_overlays(frames, content) -> list[dict]:
    """Upload watermarks: strokes in a corner that never move while the rest of the picture does."""
    if len(frames) < 6:
        return []
    steady = np.stack([_dilate(_edges(f, 14)) for f in frames]).mean(axis=0) >= 0.95
    h, w = steady.shape
    x0, y0, x1, y1 = content
    inner = steady[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    if inner.size == 0 or inner.mean() > 0.08:
        return []   # mostly static picture: persistence cannot tell overlay from scenery
    boxes = []
    steady = _inside(steady, content)
    for b in _boxes_from_mask(steady):
        bx0, by0, bx1, by1 = b["box"]
        region = steady[int(by0 * h):int(by1 * h) + 1, int(bx0 * w):int(bx1 * w) + 1]
        if _corner(b["box"], content) and region.size and region.mean() >= max(0.1, 4 * inner.mean()) \
                and _logo_shaped(b["box"], (w, h)):
            boxes.append(b)
    return boxes


@lru_cache(maxsize=1)
def _logo() -> np.ndarray:
    with Image.open(io.BytesIO(base64.b64decode("".join(LOGO_PNG)))) as im:
        return np.asarray(im.convert("L").resize((48, 23), Image.BILINEAR), dtype=np.float32)


def _ncc(image, template) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(image, template.shape)
    t = template - template.mean()
    centred = windows - windows.mean(axis=(2, 3), keepdims=True)
    num = (centred * t).sum(axis=(2, 3))
    return num / (np.sqrt((centred * centred).sum(axis=(2, 3))) * np.sqrt((t * t).sum()) + 1e-6)


def platform_watermark(path, duration: float, clip_size: tuple[int, int], count: int = 8) -> list[dict]:
    """Bilibili's upload watermark (user name + "bilibili") is burned into the stream but absent from the
    preview sprites. Find its logo in the median of the top band (either corner, any common size)."""
    band = 96
    cw, ch = clip_size
    # Short edge 720, as the logo is sized for it; explicit even sizes keep the raw reshape exact.
    if cw >= ch:
        width, height = round(cw * 720 / ch / 2) * 2, 720
    else:
        width, height = 720, round(ch * 720 / cw / 2) * 2
    with throttle.slot("frames"):
        raw = subprocess.run(
            [paths.binary("ffmpeg"), "-v", "error", "-threads", "2", "-i", str(path), "-vf",
             f"fps={count / max(0.5, duration):.4f},scale={width}:{height},crop=iw:{band}:0:0,format=gray",
             "-frames:v", str(count), "-f", "rawvideo", "-"], capture_output=True, timeout=60).stdout
    n = len(raw) // (width * band)
    if n < 2:
        return []
    top = np.median(np.frombuffer(raw[:n * width * band], dtype=np.uint8).reshape(n, band, width), axis=0)
    logo = _logo()
    found = []
    side = min(576, width // 2)
    for left in (False, True):
        x0 = 0 if left else width - side
        region = np.asarray(Image.fromarray(top[:, x0:x0 + side].astype(np.uint8)).resize(
            (side // 2, band // 2), Image.BILINEAR), dtype=np.float32)
        best = (-1.0, None, None)
        for scale in (0.6, 0.72, 0.85, 1.0, 1.15, 1.3) + ((1.5, 1.78) if ch > cw else ()):
            template = np.asarray(Image.fromarray(logo.astype(np.uint8)).resize(
                (round(48 * scale), round(23 * scale)), Image.BILINEAR), dtype=np.float32)
            score = _ncc(region, template)
            if score.max() > best[0]:
                best = (float(score.max()), np.unravel_index(score.argmax(), score.shape), template.shape)
        score, where, size = best
        if score < LOGO_MATCH:
            continue
        y1 = (where[0] + size[0]) * 2 / height + 0.012
        x1 = (x0 + (where[1] + size[1]) * 2) / width + 0.01
        # The user name sits left of the logo; its length is unknown, so reserve generously.
        box = [0.0, 0.0, x1, y1] if left else [max(0.0, x1 - 0.42), 0.0, min(1.0, x1), y1]
        found.append({"kind": "logo", "box": [round(v, 4) for v in box], "score": round(score, 3)})
    return found


def sprite_corners(frames, content, sprites, window_start: float, duration: float) -> list[dict]:
    """Static pictures: the preview frame of the same moment has no upload watermark, so any
    corner difference between it and the downloaded frame is the watermark."""
    boxes = []
    n = len(frames)
    for t, image in sprites:
        offset = t - window_start
        if not 0 <= offset <= duration:
            continue
        frame = frames[min(n - 1, max(0, round(offset / duration * n - 0.5)))]
        ref = np.asarray(image.convert("L").resize((frame.shape[1], frame.shape[0]), Image.BILINEAR), dtype=np.float32)
        if image.width < frame.shape[1]:   # a 160x90 cell is soft: compare like with like, not every sharp edge
            frame = np.asarray(Image.fromarray(frame.astype(np.uint8)).resize(image.size, Image.BILINEAR).resize(
                (frame.shape[1], frame.shape[0]), Image.BILINEAR), dtype=np.float32)
        diff = np.abs(frame - ref)
        if np.median(diff) > 6:
            continue   # not the same picture (motion or a cut): no evidence either way
        # Upload watermarks are short, wide strips; a tall difference is a re-framed picture, not a mark.
        boxes += [b for b in _boxes_from_mask(diff > 30) if _corner(b["box"], content)
                  and _logo_shaped(b["box"], (frame.shape[1], frame.shape[0]))]
    return _merge(boxes)


def _subtitle_bands(frame, content) -> list[tuple]:
    """Centred bands of dense high-contrast strokes in the lower part of one frame (one per text line)."""
    h, w = frame.shape
    x0, y0, x1, y1 = content
    top, bottom = int((y0 + 0.7 * (y1 - y0)) * h), int(y1 * h)   # subtitle lines sit in the bottom 30%
    left, right = int((x0 + 0.08 * (x1 - x0)) * w), int((x1 - 0.08 * (x1 - x0)) * w)
    region = frame[top:bottom, left:right]
    if region.size == 0:
        return []
    text = _edges(region, 70) & _dilate(_dilate(region > 170))
    density = text.mean(axis=1)
    rows = density > 0.045
    bands, r, n = [], 0, len(rows)
    while r < n:
        if not rows[r]:
            r += 1
            continue
        s = r
        while r < n and (rows[r] or (r + 2 < n and rows[r + 1:r + 3].any())):
            r += 1
        if 0.022 <= (r - s) / h <= 0.13:
            cols = text[s:r].any(axis=0)
            live = np.nonzero(cols)[0]
            if len(live):
                span = (live[-1] - live[0] + 1) / w
                centre = (left + (live[0] + live[-1]) / 2) / w
                fill = cols[live[0]:live[-1] + 1].mean()
                if 0.08 <= span <= 0.9 and abs(centre - (x0 + x1) / 2) < 0.12 and fill > 0.35:
                    bands.append(((top + s) / h, (top + r) / h, (left + live[0]) / w, (left + live[-1] + 1) / w))
    return bands


def _steady_text(frames, content, y0, y1) -> bool:
    """A subtitle keeps the same strokes for a second or more; foam, crowds or containers under a moving
    camera never line up twice (stroke-mask overlap between neighbouring frames stays below ~0.3)."""
    h, w = frames[0].shape
    masks = []
    for f in frames:
        region = f[int(y0 * h):int(y1 * h) + 1, int(content[0] * w):int(content[2] * w)]
        masks.append(_dilate(_edges(region, 70) & _dilate(_dilate(region > 170))))
    for a, b in zip(masks, masks[1:], strict=False):
        if a.sum() >= 20 and b.sum() >= 20 and (a & b).sum() / max(1, (a | b).sum()) >= 0.75:
            return True
    return False


def subtitles(frames, content, source_boxes=()) -> list[dict]:
    """Subtitle lines (up to two, e.g. Chinese + translation) seen in many frames of the window,
    or in a few where the source shows a line there too."""
    per_frame = [_subtitle_bands(f, content) for f in frames]
    bands = [b for found in per_frame for b in found]
    lines = []
    for b in sorted(bands, key=lambda b: -sum(1 for o in bands if abs(o[0] - b[0]) < 0.04)):
        if any(abs(b[0] - line[0][0]) < 0.04 for line in lines):
            continue
        lines.append([o for o in bands if abs(o[0] - b[0]) < 0.04])
        if len(lines) == 2:
            break
    boxes = []
    for group in lines:
        seen = sum(1 for found in per_frame if any(abs(o[0] - group[0][0]) < 0.04 for o in found))
        y0, y1 = min(b[0] for b in group) - 0.012, max(b[1] for b in group) + 0.012
        backed = any(sb["box"][1] < y1 and sb["box"][3] > y0 and sb["box"][2] - sb["box"][0] > 0.2
                     for sb in source_boxes)
        if seen < max(2, (0.2 if backed else 0.45) * len(frames)) or not _steady_text(frames, content, y0, y1):
            continue
        x0 = min(min(b[2] for b in group) - 0.02, 0.2)   # later lines are longer: keep the centred width
        x1 = max(max(b[3] for b in group) + 0.02, 0.8)
        boxes.append({"kind": "subtitle", "box": [round(max(0, x0), 4), round(max(0, y0), 4),
                                                  round(min(1, x1), 4), round(min(1, y1), 4)]})
    return boxes


def _overlap(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def safe_crop(boxes, content, src_aspect: float, out_aspect: float, max_zoom: float = MAX_ZOOM):
    """Largest output-shaped rectangle inside content avoiding boxes.

    Returns (rect [x, y, w, h] fractions, zoom relative to the plain cover crop, residual boxes).
    """
    cx0, cy0, cx1, cy1 = content
    cw, ch = cx1 - cx0, cy1 - cy0
    # Full-content cover crop size in fractions.
    if cw * src_aspect / ch >= out_aspect:
        base_h = ch
        base_w = ch * out_aspect / src_aspect
    else:
        base_w = cw
        base_h = cw * src_aspect / out_aspect
    pad = 0.006
    grown = [[b["box"][0] - pad, b["box"][1] - pad, b["box"][2] + pad, b["box"][3] + pad] for b in boxes]
    best = None
    steps = 22
    for k in range(0, 60):
        scale = 1 - k * 0.01
        if 1 / scale > max_zoom + 1e-6:
            break
        w, h = base_w * scale, base_h * scale
        xs = [cx0 + (cw - w) * i / steps for i in range(steps + 1)]
        ys = [cy0 + (ch - h) * i / steps for i in range(steps + 1)]
        centre = (cx0 + cw / 2, cy0 + ch / 2)
        for x in xs:
            for y in ys:
                rect = (x, y, x + w, y + h)
                hit = sum(_overlap(rect, g) for g in grown)
                dist = abs(x + w / 2 - centre[0]) + abs(y + h / 2 - centre[1])
                key = (hit > 1e-9, -scale if hit <= 1e-9 else 0, hit, dist)
                if best is None or key < best[0]:
                    best = (key, rect, scale)
        if best and not best[0][0]:
            break
    _, rect, scale = best
    residual = []
    for b, g in zip(boxes, grown, strict=True):
        if _overlap(rect, g) > 1e-9:
            clipped = [max(rect[0], g[0]), max(rect[1], g[1]), min(rect[2], g[2]), min(rect[3], g[3])]
            residual.append({"kind": b["kind"], "box": [round(v, 4) for v in clipped]})
    x0, y0, x1, y1 = rect
    return [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)], round(1 / scale, 3), residual


def spread(frames) -> float:
    """Mean grey-level distance of the window's frames from their median (coarse): a fixed camera on one
    scene stays low, cuts and camera moves are high."""
    small = [np.asarray(Image.fromarray(f.astype(np.uint8)).resize((48, 27), Image.BILINEAR), dtype=np.float32)
             for f in frames]
    median = np.median(np.stack(small), axis=0)
    return float(np.mean([np.abs(f - median).mean() for f in small]))


def motion(frames) -> dict:
    """Share of one-second steps where (almost) no pixel changes: slideshow stills, frozen frames.
    dark: a quarter or more of the sampled frames are uniformly black (black screens, fades at an intro)."""
    steps = [float((np.abs(b - a) > 20).mean()) for a, b in zip(frames, frames[1:], strict=False)]
    frozen = sum(1 for f in steps if f < 0.004)
    black = sum(1 for f in frames if float(f.mean()) < 14 and float(f.std()) < 10)
    return {"frozen": frozen, "steps": len(steps), "still": bool(steps) and frozen >= 0.55 * len(steps),
            "dark": bool(frames) and black >= max(1, 0.25 * len(frames))}


# Where a TV station's channel logo sits (CCTV and the provincial 卫视 put it top-left), as a share of the
# picture. These logos are semi-transparent and often missed by the edge checks, so a broadcast source
# gets this box even when nothing was detected there.
CHANNEL_LOGO = (0.0, 0.0, 0.16, 0.13)


def analyse(clip, duration: float, clip_size: tuple[int, int], out_aspect: float, source: dict | None = None,
            sprites: list | None = None, window_start: float = 0.0, channel_logo: bool = False) -> dict:
    """clip_size: (w, h) of the local clip; source: profile() of the preview frames of the same video;
    sprites: [(source_time, PIL image)] preview frames, used for static pictures; channel_logo: the
    source is a TV broadcast, so the top-left channel logo area is treated as a logo."""
    if np is None:
        return {}
    cw, chh = clip_size
    src_aspect = cw / chh
    size = (WIDTH, max(2, round(WIDTH / src_aspect)))
    frames = clip_frames(clip, duration, size)
    if len(frames) < 3:
        return {}
    content = content_rect(frames)
    still = motion(frames)
    pip = inner_picture(frames, content) if not still["still"] else None
    jump = not still["still"] and layout_jump(frames, content)
    if pip:
        content = pip            # the canvas around the picture (bars, the uploader's name and logo) is not footage
    source_boxes = (source or {}).get("boxes", []) if tuple((source or {}).get("size") or ()) == size else []
    # A fixed camera on one scene (a gym, a studio) keeps its background as steady as any overlay: steady
    # edges in the clip then prove little, so only tight corner marks are taken from them. Marks from the
    # preview median (the whole video) are already limited to overlay positions in profile().
    fixed = not still["still"] and spread(frames) < 15
    found = [b for b in source_boxes if _present(b["box"], b.get("pattern"), frames, size)]
    edge = [{"kind": b["kind"], "box": b["box"]} for b in found if not b["central"]]
    central = [{"kind": "center", "box": b["box"]} for b in found if b["central"]]
    corners = platform_watermark(clip, duration, clip_size)
    if not still["still"]:
        corners += [b for b in corner_overlays(frames, content) if not fixed or _tight_corner(b["box"])]
    elif sprites:   # a frozen picture is steady everywhere: compare with the preview frame instead
        corners += sprite_corners(frames, content, sprites, window_start, duration)
    if channel_logo:
        x0, y0, x1, y1 = content
        lx0, ly0, lx1, ly1 = CHANNEL_LOGO
        logo = [round(x0 + lx0 * (x1 - x0), 4), round(y0 + ly0 * (y1 - y0), 4),
                round(x0 + lx1 * (x1 - x0), 4), round(y0 + ly1 * (y1 - y0), 4)]
        if not any(b["box"][0] <= logo[0] + .02 and b["box"][1] <= logo[1] + .02 and b["box"][2] >= logo[2] - .02
                   and b["box"][3] >= logo[3] - .02 for b in edge + corners):
            corners.append({"kind": "logo", "box": logo})
    if not still["still"]:
        corners += static_bands(frames, content)
    edge = _merge(edge + corners)
    subs = subtitles(frames, content, source_boxes)
    # A picture much narrower than the film: a vertical video, or one inside wide black or blurred bars.
    narrow = (content[2] - content[0]) * src_aspect / max(1e-6, content[3] - content[1]) < 0.8 * out_aspect
    if narrow:
        # A picture much narrower than the film (a vertical video): show it whole; the render's contain fit
        # fills the sides with a blurred copy of the picture itself. Marks inside it are blurred.
        x0, y0, x1, y1 = content
        rect, zoom = [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)], 1.0
        residual = [o | {"box": [max(o["box"][0], x0), max(o["box"][1], y0), min(o["box"][2], x1), min(o["box"][3], y1)]}
                    for o in edge + subs if _overlap(o["box"], (x0, y0, x1, y1)) > 1e-9]
    elif edge or subs:
        rect, zoom, residual = safe_crop(edge + subs, content, src_aspect, out_aspect)
    else:   # nothing to hide: only trim black bars and let the render's fit decide the framing
        x0, y0, x1, y1 = content
        rect, zoom, residual = [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)], 1.0, []
    return {"content": [round(v, 4) for v in content], "overlays": edge + subs, "central": central, "size": [cw, chh],
            "crop": rect, "zoom": zoom, "residual": residual + central, "motion": still | {"fixed_camera": fixed},
            "src_aspect": round(src_aspect, 4), "inner_picture": list(pip) if pip else None, "narrow": narrow,
            "layout_jump": jump}


def dirty(result: dict) -> bool:
    """Marks that neither the crop nor painting out removes (text across the middle, a subtitle line, a big
    logo) are left in view."""
    return bool(result) and not painted(result)


def central_marks(source: dict | None) -> list[dict]:
    """Confirmed-shape mid-picture marks of a whole source (decided before any download)."""
    return [b for b in (source or {}).get("boxes", []) if b.get("central") and b.get("pattern") is not None]


def recrop(result: dict, extra: list[dict], out_aspect: float) -> dict:
    """Add boxes found later (by the visual check) and choose the crop again."""
    if not result:
        return result
    overlays = _merge(result["overlays"] + extra)
    if not overlays:
        return result
    rect, zoom, residual = safe_crop(overlays, result["content"], result["src_aspect"], out_aspect)
    return result | {"overlays": overlays, "crop": rect, "zoom": zoom, "residual": residual + result.get("central", [])}


def from_view(box, rect) -> list[float]:
    """Map a box given in the cropped view's fractions back to source-frame fractions."""
    x, y, w, h = rect
    return [round(x + box[0] * w, 4), round(y + box[1] * h, 4), round(x + box[2] * w, 4), round(y + box[3] * h, 4)]


def video_filter(result: dict) -> str:
    """FFmpeg filter prefix: crop to the clean rectangle, then paint out small logos near the edges that the
    crop could not remove (delogo fills them from the pixels around them; the client saw corner watermarks
    left in view). Text across the middle is never painted over (it looked like random mosaic): such clips
    are used only when nothing else is left."""
    if not result or not result.get("crop"):
        return ""
    x, y, w, h = result["crop"]
    parts = []
    if w < 0.995 or h < 0.995:
        parts.append(f"crop=w='trunc(iw*{w:.4f}/2)*2':h='trunc(ih*{h:.4f}/2)*2'"
                     f":x='trunc(iw*{x:.4f})':y='trunc(ih*{y:.4f})'")
    size = result.get("size")
    if size and len(size) == 2:
        iw, ih = int(size[0]), int(size[1])
        cw, ch = int(iw * w / 2) * 2 if w < 0.995 or h < 0.995 else iw, int(ih * h / 2) * 2 if w < 0.995 or h < 0.995 else ih
        x0, y0 = (int(iw * x), int(ih * y)) if w < 0.995 or h < 0.995 else (0, 0)
        for mark in result.get("residual") or []:
            box = paintable(mark, result["crop"])
            if not box:
                continue
            bx0, by0 = max(1, round(box[0] * iw) - x0 - 2), max(1, round(box[1] * ih) - y0 - 2)
            bx1, by1 = min(cw - 2, round(box[2] * iw) - x0 + 2), min(ch - 2, round(box[3] * ih) - y0 + 2)
            if bx1 - bx0 >= 4 and by1 - by0 >= 4:
                parts.append(f"delogo=x={bx0}:y={by0}:w={bx1 - bx0}:h={by1 - by0}")
    return ",".join(parts)


def paintable(mark: dict, crop) -> list | None:
    """A leftover mark small enough and close enough to the picture's edge to paint out: a logo, an "@name",
    a channel bug. Its box in source fractions, or None."""
    if mark.get("kind") in ("center", "subtitle", "caption"):
        return None
    x, y, w, h = crop
    b = [max(mark["box"][0], x), max(mark["box"][1], y), min(mark["box"][2], x + w), min(mark["box"][3], y + h)]
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    area = (b[2] - b[0]) * (b[3] - b[1]) / max(1e-6, w * h)
    # Relative to the cropped view: within the outer fifth on some side.
    near = min((b[0] - x) / w, (x + w - b[2]) / w, (b[1] - y) / h, (y + h - b[3]) / h) <= 0.2
    return b if area <= 0.03 and near else None


def painted(result: dict) -> bool:
    """Whether every mark left in view gets painted out (the clip then counts as clean)."""
    rest = [r for r in (result or {}).get("residual", []) if not (result.get("size") and paintable(r, result["crop"]))]
    return not any(r["kind"] in ("center", "subtitle", "caption") or (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1]) > 0.004
                   for r in rest)


def _merge(boxes):
    merged = []
    for b in sorted(boxes, key=lambda b: b["box"]):
        for m in merged:
            g = [m["box"][0] - 0.01, m["box"][1] - 0.01, m["box"][2] + 0.01, m["box"][3] + 0.01]
            if m["kind"] == b["kind"] and _overlap(g, b["box"]) > 0:
                m["box"] = [min(m["box"][0], b["box"][0]), min(m["box"][1], b["box"][1]),
                            max(m["box"][2], b["box"][2]), max(m["box"][3], b["box"][3])]
                break
        else:
            merged.append({"kind": b["kind"], "box": list(b["box"])})
    return merged
