"""One audio-bound caption clock shared by previews, burned captions and SRT."""
from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import threading
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from . import local_asr, paths, subtitle_style

VERSION = 1
ALIGNMENT_VERSION = 3
_locks_guard = threading.Lock()
_locks = {}


class CharacterTimingRequired(ValueError):
    """An old cue-only cache cannot prove a cut inside a caption."""


class AcousticBoundaryRequired(ValueError):
    """A cut lacks observed token edges; interpolation cannot prove it."""


def width(text):
    return subtitle_style.width(text)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _words(text):
    try:
        import jieba
        jieba.setLogLevel(30)
        return list(jieba.cut(text, HMM=False))
    except ImportError:
        # Sentence boundaries still work without the optional word segmenter.
        return re.findall(r"[A-Za-z0-9]+|.", text)


def parts(text, max_chars=36):
    """Keep whole sentences, then clauses; word breaks are the last resort.

    max_chars is a display width: Latin letters and digits count about half a character."""
    text = clean(text)
    sentences = re.findall(r"[^。！？!?；;]+[。！？!?；;]*|[。！？!?；;]+", text)
    result = []
    for sentence in sentences:
        if width(sentence) <= max_chars:
            result.append(sentence)
            continue
        clauses = re.findall(r"[^，,：:、]+[，,：:、]*|[，,：:、]+", sentence)
        units = []
        for clause in clauses:
            if width(clause) <= max_chars:
                units.append(clause)
                continue
            # Word units; punctuation stays with the word before it so no line starts or ends as a lone mark.
            for word in _words(clause):
                if units and (re.fullmatch(r"[^\w]+", word) or not units[-1]):
                    units[-1] += word
                else:
                    units.append(word)
        if len(units) > 1 and re.fullmatch(r"[^\w]+", units[0]):
            units[1] = units[0] + units[1]
            units = units[1:]
        # Balance groups of clauses instead of greedily filling a fixed width.
        # This keeps e.g. “收货的时候，箱子……” together.
        best = [(0, [])] + [(float("inf"), []) for _ in units]
        target = width(sentence) / math.ceil(width(sentence) / max_chars)
        for end in range(1, len(units) + 1):
            for start in range(end):
                phrase = "".join(units[start:end])
                if width(phrase) > max_chars and end-start > 1:
                    continue
                dangling = bool(re.search(r"(?:时候|之后|之前|以后|来说|而言|的话)[，,：:]$", phrase))
                cost = best[start][0] + (width(phrase)-target)**2 + 80 + (500 if dangling else 0)
                if cost < best[end][0]:
                    best[end] = cost, best[start][1] + [phrase]
        result.extend(best[-1][1])
    return result or [""]


def wrap(text, line_chars=20):
    if width(text) <= line_chars:
        return text
    boundaries, offset = [], 0
    for word in _words(text):
        offset += len(word)
        if offset < len(text) and text[offset] not in "，。！？、；：,.;:!?）》】”’":
            boundaries.append(offset)
    if not boundaries:
        return text
    def penalty(i):
        overflow = max(0, i - line_chars) + max(0, len(text) - i - line_chars)
        return overflow * 100 + abs(i - len(text) / 2) - (6 if text[i-1] in "，：、,:" else 0)
    cut = min(boundaries, key=penalty)
    return text[:cut] + "\n" + text[cut:]


def _normal(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).lower() if c.isalnum())


_ZH_DIGITS = "零一二三四五六七八九"


def _section(n):
    out, zero = "", False
    for power, unit in ((3, "千"), (2, "百"), (1, "十"), (0, "")):
        digit = n // 10 ** power % 10
        if not digit:
            zero = bool(out)
            continue
        if zero:
            out += "零"
            zero = False
        out += _ZH_DIGITS[digit] + unit
    return out


def _cardinal(n):
    if n == 0:
        return "零"
    sections = []
    while n:
        sections.append(n % 10000)
        n //= 10000
    out = ""
    for index in range(len(sections) - 1, -1, -1):
        section = sections[index]
        if not section:
            continue
        if out and section < 1000 and not out.endswith("零"):
            out += "零"
        out += _section(section) + ("", "万", "亿", "万亿")[index]
    return out[1:] if out.startswith("一十") else out


def _spoken(original):
    """Spoken Mandarin form of digit runs for matching ASR output; owner maps back to original."""
    spoken, owner, i = "", [], 0
    while i < len(original):
        if not original[i].isdigit() or not original[i].isascii():
            spoken += original[i]
            owner.append(i)
            i += 1
            continue
        j = i
        while j < len(original) and original[j].isdigit() and original[j].isascii():
            j += 1
        run = original[i:j]
        following = original[j] if j < len(original) else ""
        if (len(run) == 4 and following == "年") or len(run) > 8 or (len(run) > 1 and run.startswith("0")):
            words = "".join(_ZH_DIGITS[int(d)] for d in run)
        else:
            words = _cardinal(int(run))
        for k, char in enumerate(words):
            spoken += char
            owner.append(i + min(len(run) - 1, k * len(run) // len(words)))
        i = j
    return spoken, owner


def align(text, words, duration):
    """Anchor original wording to recognized speech, rejecting broad mismatch."""
    text = clean(text)
    source = _normal(text)
    # Match in spoken form (digits read aloud), then collapse back to source characters.
    original, owner = _spoken(source)
    spoken, timings, acoustic, token_indices = "", [], [], []
    for token_index, word in enumerate(words):
        token = _normal(word["text"])
        if not token:
            continue
        for i in range(len(token)):
            a = word["start"] + (word["end"] - word["start"]) * i / len(token)
            b = word["start"] + (word["end"] - word["start"]) * (i+1) / len(token)
            timings.append((a, b))
            acoustic.append((i == 0, i == len(token)-1))
            token_indices.append(token_index)
        spoken += token
    if not original or not spoken:
        raise ValueError("未识别到可与原稿对齐的语音，已停止字幕生成")
    mapping = [None] * len(original)
    anchors = [(False, False)] * len(original)
    matched = 0
    matcher = SequenceMatcher(None, original, spoken, autojunk=False)
    for block in matcher.get_matching_blocks():
        for i in range(block.size):
            mapping[block.a + i] = timings[block.b + i]
            anchors[block.a + i] = acoustic[block.b + i]
        matched += block.size
    coverage = matched / len(original)
    if coverage < .85 or matched / len(spoken) < .8:
        raise ValueError(f"音轨与原稿一致度不足（{coverage:.0%}），请检查是否选错配音；未按字数伪造同步")
    # A single misrecognized character still has an observed token interval.
    # Preserve that evidence only for an isolated 1:1 replacement between two
    # exact contexts, never for an insertion/deletion or a multi-character
    # token's interpolated interior. This does not increase lexical coverage.
    substitutions = []
    opcodes = matcher.get_opcodes()
    for index, (kind, a, b, c, d) in enumerate(opcodes):
        if kind != "replace" or b-a != 1 or d-c != 1 or not 0 < index < len(opcodes)-1:
            continue
        before, after = opcodes[index-1], opcodes[index+1]
        if (before[0] != "equal" or after[0] != "equal" or before[2]-before[1] < 2
                or after[2]-after[1] < 2 or acoustic[c] != (True, True)
                or not all(re.fullmatch(r"[\u3400-\u9fff]", char) for char in (original[a], spoken[c]))):
            continue
        begin, end = timings[c]
        left, right = mapping[a-1][1], mapping[b][0]
        if not left <= begin < end <= right or end-begin > .600001 or right-left > 3:
            continue
        mapping[a], anchors[a] = (begin, end), acoustic[c]
        substitutions.append({"character_offset": a, "original": original[a], "recognized": spoken[c],
                              "token_index": token_indices[c], "start": begin, "end": end,
                              "method": "isolated_token_substitution", "version": ALIGNMENT_VERSION})
    # Only short recognition errors may use nearby acoustic anchors. Entire
    # sentences or a different narration can never become proportional timing.
    index = 0
    while index < len(mapping):
        if mapping[index] is not None:
            index += 1
            continue
        end = index
        while end < len(mapping) and mapping[end] is None:
            end += 1
        left = mapping[index-1][1] if index else 0
        right = mapping[end][0] if end < len(mapping) else duration
        if end-index > 8 or right-left > 3.0 or right < left:
            raise ValueError("原稿有一段未能与音轨对应，已保留配音并停止字幕生成")
        for i in range(index, end):
            mapping[i] = (left + (right-left)*(i-index)/(end-index),
                          left + (right-left)*(i+1-index)/(end-index))
        index = end
    positions = [[] for _ in source]
    for index, origin in enumerate(owner):
        positions[origin].append(index)
    # A character folded into its neighbour when spoken ("10" is read as one syllable) gets a zero-length
    # time at the end of the character before it (or the start of the next one).
    folded = [i for i, js in enumerate(positions) if not js]
    for i in folded:
        positions[i] = [positions[i-1][-1]] if i and positions[i-1] else \
            next(([later[0]] for later in positions[i+1:] if later), [0])
    mapping = [(mapping[js[0]][0], mapping[js[-1]][1]) for js in positions]
    anchors = [(anchors[js[0]][0], anchors[js[-1]][1]) for js in positions]
    for i in folded:
        edge = mapping[i-1][1] if i else mapping[i][0]
        mapping[i], anchors[i] = (edge, edge), (anchors[i-1][1], anchors[i-1][1]) if i else anchors[i]
    for row in substitutions:
        row["character_offset"] = owner[row["character_offset"]]
    original = source
    cues, offset = [], 0
    for phrase in parts(text):
        count = len(_normal(phrase))
        if not count:
            continue
        begin, end = mapping[offset][0], mapping[offset+count-1][1]
        offset += count
        if end <= begin or end > duration + .1:
            raise ValueError("语音时间戳无效，未生成字幕")
        cues.append({"start": round(max(0, begin), 3), "end": round(min(duration, end), 3), "text": phrase})
    return {"cues": cues, "coverage": round(coverage, 4), "matched_characters": matched,
            "characters": len(original), "method": "local_paraformer", "api_cost_cny": 0,
            "alignment_version": ALIGNMENT_VERSION, "observed_substitutions": substitutions,
            # Caption display may interpolate short recognition gaps or the
            # inside of a token. Shot cuts must use observed token edges only.
            "character_timing": {"version": 1, "text_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "spans": [[a, b, left, right] for (a, b), (left, right) in zip(mapping, anchors, strict=True)]}}


def _source(project, shot=None):
    audio = project.get("full_audio") if shot is None else shot.get("audio")
    if not audio:
        return None
    path = paths.file_path(project["id"], audio["path"])
    text = clean("".join(s["narration"] for s in project["shots"]) if shot is None else shot["narration"])
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    binding = {"version": VERSION, "audio_sha256": digest,
               "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "model_sha256": local_asr.MODEL_HASH}
    key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    return path, text, audio, binding, paths.DATA / "cache" / "captions" / (key + ".json")


def cached(project, shot=None):
    try:
        source = _source(project, shot)
        if not source:
            return None
        *_, binding, target = source
        data = json.loads(target.read_text())
        if data.get("binding") == binding and data.get("cues"):
            return data
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


def ensure(project, shot=None, check=lambda: None, progress=lambda *_: None, *, require_character_timing=False,
           require_alignment_version=0):
    source = _source(project, shot)
    if not source:
        return None
    with _locks_guard:
        lock = _locks.setdefault(str(source[-1]), threading.Lock())
    while not lock.acquire(timeout=.1):
        check()
    try:
        existing = cached(project, shot)
        if (existing and (not require_character_timing or existing.get("character_timing"))
                and existing.get("alignment_version", 1) >= require_alignment_version):
            return existing
        return _generate(project, shot, check, progress)
    finally:
        lock.release()


def _generate(project, shot, check, progress):
    from . import media
    source = _source(project, shot)
    if not source:
        return None
    path, text, audio, binding, target = source
    check()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="caption-", dir=target.parent) as directory:
        wav = Path(directory) / "speech.wav"
        media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-i", str(path),
                   "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], check=check)
        words = local_asr.transcribe(wav, check, progress)
    duration = float(audio["duration"])
    result = align(text, words, duration) | {"binding": binding, "duration": duration,
                                           "observed_words": words}
    check()
    if _source(project, shot)[-2] != binding:
        raise ValueError("对齐期间音频内容发生变化，已保留音频，请重新对齐")
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        temporary = Path(out.name)
    temporary.replace(target)
    return result


def shot_timing(project, track):
    """Derive absolute frame boundaries without any character-length timing."""
    shots = project["shots"]
    duration = float(project["full_audio"]["duration"])
    if not shots or not math.isfinite(duration) or duration <= 0:
        raise ValueError("音频或分镜时长无效，已保留音频")
    if not isinstance(track, dict) or track.get("method") != "local_paraformer":
        raise ValueError("整段音轨尚未完成本地语音对齐，请先运行匹配或复核")
    if not math.isfinite(float(track.get("duration", 0))) or abs(float(track.get("duration", 0))-duration) > .001:
        raise ValueError("字幕缓存与当前音频时长不一致，已停止镜头对齐")
    original = clean("".join(s["narration"] for s in shots))
    cues = track.get("cues", [])
    if not cues or clean("".join(c["text"] for c in cues)) != original:
        raise ValueError("字幕缓存未逐字保留当前分镜原稿，已停止镜头对齐")
    normal = _normal(original)
    counts = [len(_normal(s["narration"])) for s in shots]
    if not all(counts) or sum(counts) != len(normal):
        raise ValueError("分镜缺少可对齐的口播文字，已停止镜头对齐")
    edges, offset, previous_end = {}, 0, 0.0
    for index, cue in enumerate(cues):
        begin, end = float(cue["start"]), float(cue["end"])
        if not all(math.isfinite(t) for t in (begin, end)) or not 0 <= begin < end <= duration+.001:
            raise ValueError("字幕时间戳无效，已停止镜头对齐")
        if begin < previous_end-.001:
            raise ValueError("字幕时间戳重叠，已停止镜头对齐")
        if index:
            edges[offset] = (previous_end, begin)
        offset += len(_normal(cue["text"]))
        previous_end = end
    character = track.get("character_timing")
    spans = None
    if character is not None:
        if not isinstance(character, dict):
            raise ValueError("字符时间映射无效，已停止镜头对齐")
        spans = character.get("spans")
        if (character.get("version") != 1 or character.get("text_sha256") != hashlib.sha256(normal.encode()).hexdigest()
                or not isinstance(spans, list) or len(spans) != len(normal)):
            raise ValueError("字符时间映射与当前原稿不一致，已停止镜头对齐")
        previous_end = 0.0
        for span in spans:
            if not isinstance(span, list) or len(span) != 4 or any(type(flag) is not bool for flag in span[2:]):
                raise ValueError("字符时间映射无效，已停止镜头对齐")
            begin, end = float(span[0]), float(span[1])
            if (not all(math.isfinite(t) for t in (begin, end)) or not 0 <= begin <= end <= duration+.001
                    or begin < previous_end-.001):
                raise ValueError("字符时间映射无效，已停止镜头对齐")
            previous_end = end
    boundaries, evidence, offset = [0], [], 0
    for count in counts[:-1]:
        offset += count
        if spans is not None:
            left, right = spans[offset-1], spans[offset]
            left_end, right_start = float(left[1]), float(right[0])
            # Prefer observed token edges; a cut inside a short recognised gap
            # (numbers, names) uses the bounded interpolation instead of failing.
            kind = "token_edges" if left[3] and right[2] else "interpolated_edge"
        elif offset in edges:
            # Legacy cache reuse is allowed only at an exact whole-cue text
            # offset; never estimate a position within its text or duration.
            left_end, right_start = edges[offset]
            kind = "caption_edges"
        else:
            raise CharacterTimingRequired("镜头切点位于旧字幕内部，需要补充本地字符时间映射")
        if right_start < left_end-.001:
            raise ValueError("镜头切点语音时间重叠，已停止镜头对齐")
        frame = math.floor(right_start*25+.5)
        boundaries.append(frame)
        evidence.append({"character_offset": offset, "left_end": left_end,
                         "right_start": right_start, "frame": frame, "method": kind})
        substitutions = [s for s in track.get("observed_substitutions", [])
                         if s["character_offset"] in (offset-1, offset)]
        if substitutions:
            evidence[-1]["recognition_substitutions"] = substitutions
    boundaries.append(math.ceil(duration*25-1e-7))
    lengths = [end-begin for begin, end in zip(boundaries, boundaries[1:], strict=False)]
    if any(not 25 <= frames <= 3000 for frames in lengths):
        raise ShotLengthOutOfRange([i for i, frames in enumerate(lengths) if frames < 25],
                                   [i for i, frames in enumerate(lengths) if frames > 3000])
    return {"version": 1, "fps": 25, "binding": track.get("binding"), "frames": boundaries,
            "durations": [frames/25 for frames in lengths], "boundaries": evidence}


class ShotLengthOutOfRange(ValueError):
    """Real speech gives some shots under 1 s (short) or over 120 s (long); indices are shot positions."""
    def __init__(self, short, long):
        super().__init__("真实语音对应的镜头不足 1 秒或超过 120 秒，请调整分镜；已保留音频")
        self.short, self.long = short, long


def ensure_shot_timing(project, check=lambda: None, progress=lambda *_: None):
    track = ensure(project, check=check, progress=progress)
    try:
        return shot_timing(project, track)
    except CharacterTimingRequired:
        track = ensure(project, check=check, progress=progress, require_character_timing=True)
        return shot_timing(project, track)
    except AcousticBoundaryRequired:
        if track.get("alignment_version", 1) >= ALIGNMENT_VERSION:
            raise
        # Old caches discarded the observed times of ASR substitutions. Only a
        # failing old cache is upgraded; valid existing clocks remain reusable.
        track = ensure(project, check=check, progress=progress, require_character_timing=True,
                       require_alignment_version=ALIGNMENT_VERSION)
        return shot_timing(project, track)


def segments(track, offset, duration, w, h, narration=None, char_offset=0, line_chars=None):
    """On-screen lines of one shot: one line at a time, each timed to its own spoken characters.

    narration/char_offset locate the shot in the track's text; without character timing
    (old caches) whole cues are split into lines in proportion to their characters."""
    line_chars = line_chars or subtitle_style.line_chars(w, h)
    spans = (track.get("character_timing") or {}).get("spans")
    if narration and isinstance(spans, list):
        pieces = [p for p in parts(narration, line_chars) if _normal(p)]
        total = sum(len(_normal(p)) for p in pieces)
        if pieces and char_offset + total <= len(spans):
            rows, k = [], char_offset
            for piece in pieces:
                n = len(_normal(piece))
                rows.append([float(spans[k][0]) - offset, float(spans[k+n-1][1]) - offset, piece])
                k += n
            # On screen from the cut until the next line starts, so no line flickers off between words.
            rows[0][0] = 0.0
            for before, after in zip(rows, rows[1:], strict=False):
                cut = min(max(after[0], before[0]), duration)
                before[1], after[0] = cut, cut
            rows[-1][1] = duration
            return [(round(max(0.0, a), 3), round(min(duration, b), 3), wrap(text, line_chars))
                    for a, b, text in rows if b > a]
    # Old cue-only caches: split each cue into lines by character share; a line belongs to the shot that
    # holds its middle, and stays up until the next line (no flicker at the cut).
    rows = []
    for cue in track["cues"]:
        if not (cue["end"] > offset and cue["start"] < offset + duration):
            continue
        pieces = parts(cue["text"], line_chars)
        weights = [max(1, len(_normal(p))) for p in pieces]
        start, span, done = float(cue["start"]), float(cue["end"]) - float(cue["start"]), 0
        for piece, weight in zip(pieces, weights, strict=True):
            a = start + span * done / sum(weights)
            done += weight
            b = start + span * done / sum(weights)
            if offset <= (a + b) / 2 < offset + duration:
                rows.append([max(0.0, a - offset), min(duration, b - offset), piece])
    if not rows:
        return []
    for before, after in zip(rows, rows[1:], strict=False):
        before[1] = after[0]
    rows[-1][1] = duration if rows[-1][1] >= duration - .5 else rows[-1][1]
    return [(round(a, 3), round(b, 3), wrap(text, line_chars)) for a, b, text in rows if b > a]


def char_offset(shots, index):
    """Position of shot index within the full narration's normalised characters."""
    return sum(len(_normal(s["narration"])) for s in shots[:index])


def frame_duration(value):
    return math.ceil(float(value) * 25 - 1e-7) / 25
