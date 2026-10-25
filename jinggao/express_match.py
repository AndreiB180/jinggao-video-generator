"""Express matcher: fill the whole narration timeline with real footage, cheaply.

Ordinary shots only need roughly related, natural footage, so they share one
batched visual pick per search group over labelled contact sheets of free
preview thumbnails. Shots that name a real person use a strict lane: the
source text must carry the full name and the visual pick must show a real
person as the subject; otherwise they fall back to people-free context
footage instead of someone else. A final cheap visual QA over the actually
downloaded windows replaces off-topic, black, talking-head or covered clips.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageStat

from . import (config, curation, execution, express_clean, express_sources as es, media, network, news_pool, paths,
               planner, screen_text, store, throttle)

VERSION = 1
BILI_PER_QUERY = 6
PERSON_PER_QUERY = 5   # named people have little real footage: look a little wider before falling back
PIXA_PER_QUERY = 6     # licensed stock: clean, no subtitles or contact text; strong for generic and abstract lines
LOCAL_PER_QUERY = 5     # videos from the local footage index (footage_index) per query, ahead of live results
TILES_PER_BILI = 8
PERSON_TILES = 10
QA_W, QA_H = 320, 180
SHEET_COLS, SHEET_ROWS = 5, 4
TILE_W, TILE_H = es.TILE
LABEL_H = 0
WINDOW_GAP = 1.5
# Repetition: one source may fill at most this share of the timeline before it costs extra; two windows of
# one source closer than SCENE_SECONDS usually show the same scene; tiles this close in dHash bits are the
# same picture (the same footage re-uploaded under another id repeats just as visibly).
SOURCE_SHARE = 0.03
HARD_SHARE = 0.04      # ...and never more than this share (one long stock take filled a third of a film)
HARD_MIN = 2           # ...nor more than twice in a short film: viewers called three windows of one video "翻来覆去"
SCENE_SECONDS = 20.0   # two windows of one video closer than this show the same scene
NEWS_CAP = 6           # windows of one news report about the event (news_pool) per film
NEWS_LATER = (0.0, 5.0, 10.0)
NEWS_LATER_ONSITE = (0.0, 4.0)   # windows already moved onto on-site footage by its preview frames
NEWS_WAIT = 150.0      # seconds the matching waits, after picking, for the news reports to be transcribed
LOW_HEIGHT = 480       # without a B站 login many drama and short-play clips stop at 480p: usable after the sharp
                       # ones (a social story otherwise loses most of its footage and repeats the rest)
STOCK_SHARE = 0.15     # foreign stock (Pixabay) at most this share of a film: for a Chinese story its foreign
                       # couples and weddings read as filler ("都是空镜头"), whatever the planner asked for
TWIN_BITS = 8
SPRITE_SHEETS_KEPT = 12   # preview sheets kept decoded (about 13 MB each in grey)
MARGIN = 0.3   # seconds of extra footage per window; tolerates sources ending early
FOOTAGE_WORDS = re.compile(r"素材|空镜|航拍|实拍|纪录|影像|画面|现场|4K|4k|高清|原声|新闻|珍贵|全景|延时|记录")
NOISE_WORDS = re.compile(
    r"教程|讲解|解说|课程|讲座|课件|PPT|ppt|直播|游戏|动画|MMD|手书|鬼畜|CAD|图纸|三维模型|建模|广告|带货|测评|开箱|"
    r"reaction|反应|翻唱|MV|AMV|搞笑|模拟器|我的世界|原神|王者|AI绘|AI生成|配音秀|考研|考试|答辩|股票|基金|大A|"
    r"财经|行情|招聘|面试|求职|小说|有声书|电台|音乐|歌曲|BGM|bgm|伴奏")
# Uploads that say they are sped up (re-uploaders speed clips up against duplicate checks) or are memes:
# never used.
SKIPPED_UPLOADS = re.compile(r"倍速|加速版|快进|变速|表情包|梗图|斗图")
DRAMA_WORDS = re.compile(r"电视剧|电影|剧照|饰演|扮演|演员|剧情|混剪|AI|特型演员|模仿|cos|动画|动漫|漫画|手绘|配音|片段合集|"
                         r"朗诵|背景视频|LED|舞台|情景")
PERSON_EVIDENCE = re.compile(r"影像|纪录|新闻|讲话|原声|视频|资料|珍贵|历史|现场|采访|发言|报道")
# Named people must never be shown by an actor: drop drama/edit categories, tags and dialogue-style titles.
DRAMA_TYPES = {"影视剪辑", "影视杂谈", "小剧场", "短片", "电影", "电视剧", "综艺", "娱乐杂谈", "搞笑", "鬼畜调教",
               "鬼畜剧场", "动画", "MAD·AMV", "手书", "短剧"}
DRAMA_TAGS = re.compile(r"电影|电视剧|影视剪辑|影视|演技|演员|饰演|扮演|混剪|二创|剧情|短剧|名场面|配音")
DIALOGUE_TITLE = re.compile(r"(说|道|问|答)[：:]|[“\"].{2,}[”\"].*(说|道)")
IDENTITY_REJECTION = re.compile(r"演员|扮演|影视|情景|再现|不是|非|主体为|他人|别人|其他人|冒充|主持|博主|嘉宾|访谈|采访|"
                                r"镜头讲话|对镜|口播画面|演播室|黑屏|纯色|模糊|动画|AI|字幕遮挡|台标|"
                                r"葬礼|遗体|遗像|灵堂|灵柩|追悼|悼念|花圈")
# Rejections that make a clip unusable for any line, even as a last resort before a title card.
BROKEN_REJECTION = re.compile(r"黑屏|黑场|纯色|模糊|动画|游戏|AI|PPT|文字页|图表|截图|截屏|评论区|聊天|界面|马赛克|主持|博主|嘉宾|访谈|采访|出镜|镜头讲话|对镜|"
                              r"对着镜头|面对镜头|口播画面|演播室|遮挡|硬字幕|水印|台标|弹幕|贴纸|广告|联系方式|引流|电话|微信|QQ|私信|号码|打码|色块")
FUNERAL = re.compile(r"葬礼|遗体|遗容|遗像|灵堂|灵柩|追悼|悼念|花圈|讣告|火化|送别|逝世后|告别仪式")
DEATH_LINE = re.compile(r"逝世|去世|病逝|离世|悼念|追悼|葬礼|长眠|告别")
BROADCAST = re.compile(r"CCTV|cctv|央视|中央电视台|卫视|电视台|CGTN|凤凰卫视|凤凰资讯|凤凰网|凤凰视频")   # a channel logo sits top-left
ARCHIVE_TITLE = re.compile(r"珍贵|影像|纪录片|央视|新闻|原声|档案|讲课|讲话|采访|实录")

PICK_SYSTEM = """你在为营销号视频挑选实拍素材。图片是若干视频的缩略图联系表，每格左上角有黄色编号。
请为每个镜头挑出合适的格子编号（最好的在前，最多6个），只输出 JSON：{"S1":["12","7"],"S2":[]}。
普通镜头标准：画面主体要和口播讲的对象一致（画面建议只是参考），不要求型号、年份、地点、动作逐一一致：讲人和人之间的事（夫妻、家人、同事、官员、民众、士兵），画面要有相应的人和场景，不选风景、空镜、物品；讲某个国家、组织的事，画面要是那个国家、组织的，不选别国的同类画面（讲也门战事不选中国火箭军、中国海军）；讲中国国内的人和事，不选明显是外国人或外国街景的画面。情感、家庭、社会故事可以选影视剧、短剧里情节相符的片段（有人物、有情绪）。
必须排除：主持人或博主对镜头口播、PPT/文字页/图表/排行榜截图为主、手机或电脑屏幕截图（评论区、聊天记录、网页/App 界面）、大面积马赛克、片头片尾和转场、频道订阅页、黑屏或严重模糊、大面积硬字幕/弹幕/贴纸/台标遮挡画面、游戏/动画/CG（除非口播就是讲这些）、与口播明显无关的内容、广告。
新闻、纪录片里真实存在的人物的脸部特写，除非该镜头要求此人物，否则尽量不选（容易被当成口播里讲的人）；影视剧、短剧里的演员不受此限。
优先选有明显动作或镜头运动的视频画面；静态插画、照片、幻灯片式的画面尽量不选（口播在讲这幅画、这件文物时除外）。"""

PERSON_RULES = """
本组是具名人物镜头，要求出现的人物：{names}。这些格子来自标题含该姓名的视频。
只选：真实拍摄的历史影像/新闻/纪录片画面，以人物为主体（讲话、会见、视察、出场等），与口播一致。
排除：影视剧/情景再现中的演员扮演（较早年代的事件却是彩色高清、布光像电影的场景，通常是扮演）、动画/手绘/AI图、只有文字或照片墙、多人画面中看不出谁是主体、字幕上写着其他人姓名的画面。
如画面上能读到该人物姓名字幕更好。另外请用键 "_ctx" 列出不含任何可辨认人物、但与口播背景相关的环境/资料画面编号（如场景、建筑、文件），用于无合适人物画面时补位。"""

ANCIENT_RULES = """
本组讲古代人物：{names}（生活在照相术出现以前，不可能有真实影像）。这些格子来自标题含该姓名的视频。
只选：表现此人的画像、雕像、历史题材影视剧/纪录片/动画/插画中以此人为主体的画面，或与口播相关的同时代场景。
排除：现代场景和现代服饰、明显是其他具名人物（字幕或画面写着别人的名字）、主持人或博主对镜头讲解、片头片尾。"""

PRESENTER_SYSTEM = """图片每一行是同一个视频的8张抽帧（行首黄色编号 V1、V2…）。
找出“以主持人/博主/讲解者对着镜头说话为主”的视频：8帧里有3帧及以上是真人（或虚拟主播）正对镜头讲话、坐在桌前或站在背景前讲解的画面。
影视剧、纪录片里的人物对话、新闻现场、采访对象不算。只输出 JSON：{"presenter":["V2","V5"]}，没有就空数组。"""

TOPIC_SHOT = "通用补位画面：与整篇主题相关的环境/场景/物体画面，不出现可辨认人物特写"

QA_SYSTEM = """你在检查营销号视频的成片选镜。每一行是同一个镜头的3张抽帧（左→右，已按成片裁切），行首黄色编号是镜头号。
对照每个镜头的口播内容（画面建议只是参考，不必符合），只输出 JSON：{"bad":{"S3":"原因"},"wm":{"S2":["右上","底部"]}}，没有就留空，原因不超过20字。
bad：只列“明显不合格”的镜头。不合格指：任意一帧出现主持人、博主或嘉宾对着镜头讲话（访谈/口播/演播室；要求的具名人物本人的历史讲话影像除外）；与口播明显无关/跑题；要求具名人物的镜头画面是影视剧/情景再现的演员扮演（老年代事件却是彩色高清电影感画面）；PPT/文字页/图表为主；手机或电脑屏幕截图（评论区、聊天记录、网页/App 界面），即使口播在讲网友评论；大面积马赛克或打码；表情包、卡通表情、贴纸、梗图、对话气泡图标叠在画面上，或画面本身是表情包式的夸张反应镜头（对着镜头狂笑、做鬼脸）；画面上后加的大号花字、标题字（不是场景里本来就有的字）；整段画面静止不动（一张图片或定格）；黑屏、纯色、严重模糊；大面积硬字幕、弹幕、贴纸或台标遮挡主体；画面上有联系方式或引流文字（电话号码、微信号、QQ号、“私信”“加V”“需要视频”等，再小也算）；游戏/动画（口播不是讲这些时）；要求具名人物的镜头却明显是别人或演员。
wm：画面上仍能看到的台标、水印、平台或作者署名（包括叠在画面中间或侧边的半透明频道名/账号名，如“@名字”“某某说”“某某影视”，再淡也要列出）、原视频字幕或文字条、画面中间的花字/标题/滚动文字，写出所有位置：左上/右上/左下/右下/顶部/底部/左侧/右侧/中间（字幕、文字条写“底部”）。小角标也要列出，它们会被裁掉，不算不合格。
若给出“整篇领域”，画面属于其他领域（如讲乒乓球却是电竞、板球、足球等其他运动，或卡通动画、3D建模渲染、频道片尾订阅页、排行榜截图）一律不合格；标为“古代人物”的镜头里，表现此人的动画、插画不算其他领域。
股市K线和走势图、数据图表、商品和品牌logo、街道招牌、包装上的字是画面本身的内容，不算文字页，口播相关就合格。
普通镜头（没有要求具名人物）：用电视剧、短剧、栏目剧片段里的演员表现口播讲的情节（夫妻吵架、婆媳矛盾、查看手机、离婚、职场冲突等）是合格的，不要因为是演员扮演或情景再现判不合格。画面主体和口播讲的对象一致即可，细节不必一致：例如口播讲港口工作人员协作，画面是港口作业远景也算合格；不要因为型号、年份、具体动作、人物细节或特写不一致而判不合格；菜品种类、制作步骤、同一场所的不同环节（如揉面与拉面、后厨与前厅、汤与面）也不必一致。以下算跑题（不合格，原因写“跑题”）：口播讲人和人之间的事（夫妻、家人、同事、官员、民众），画面却是风景、空镜、物品、动物，看不出相关的人；口播讲中国国内的人和事，画面明显是外国人或外国街景；口播讲某个国家、组织，画面明显是另一个国家的（如讲也门战事却是中国火箭军发射、中国海军演习）；科普讲解、课堂课件、带货、游戏解说之类与口播无关的视频画面；画面题材与口播完全无关。
要求具名人物的镜头：只要画面是此人本人的真实影像或照片就合格，即使与口播所讲的具体事件、场景不一致也不要判不合格。
标为“古代人物”的镜头：此人没有真实影像，画像、雕像、影视剧/纪录片/动画/插画里表现此人的形象都合格，不要因演员扮演、动画或插画判不合格；只有现代场景、明显是其他具名人物、主持人对镜头讲解才不合格。
任何镜头：口播没讲到逝世、悼念时，葬礼、遗体、遗像、灵堂、花圈、追悼会画面不合格。"""


BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
           "Chrome/124.0 Safari/537.36")


class BiliUnavailable(RuntimeError):
    """Bilibili refused this computer (searches all empty, or play addresses refused): a film made now would
    be a few stock clips over and over, so the video waits and is made again later."""


class Budget(RuntimeError):
    pass


def intro_seconds(length: float) -> float:
    """How much of a source's start to leave out: uploaders' intro cards run a few seconds."""
    return min(8.0, 0.1 * max(0.0, length))


def _font(size):
    try:
        return ImageFont.truetype(str(paths.font_path()), size)
    except (OSError, RuntimeError):
        return ImageFont.load_default()


def _jpeg(image: Image.Image, quality=80) -> str:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _dark(image: Image.Image) -> bool:
    stat = ImageStat.Stat(image.convert("L").resize((64, 36)))
    return stat.mean[0] < 22 or stat.stddev[0] < 10


def _dhash(image: Image.Image) -> int:
    small = image.convert("L").resize((9, 8))
    pixels = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = bits << 1 | (pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return bits


def _tokens(text: str) -> list[str]:
    import jieba
    words = [w.strip() for w in jieba.lcut(text) if len(w.strip()) >= 2]
    return list(dict.fromkeys(words))


def _sheet(tiles: list[dict], start_label: int) -> tuple[Image.Image, dict]:
    rows = math.ceil(len(tiles) / SHEET_COLS)
    sheet = Image.new("RGB", (SHEET_COLS * TILE_W, rows * TILE_H), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    font = _font(26)
    labels = {}
    for index, tile in enumerate(tiles):
        label = str(start_label + index)
        x, y = (index % SHEET_COLS) * TILE_W, (index // SHEET_COLS) * TILE_H
        sheet.paste(tile["image"].resize((TILE_W, TILE_H)), (x, y))
        box = draw.textbbox((x + 4, y + 2), label, font=font)
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(255, 214, 0))
        draw.text((x + 4, y + 2), label, font=font, fill=(0, 0, 0))
        draw.rectangle((x, y, x + TILE_W - 1, y + TILE_H - 1), outline=(0, 0, 0), width=2)
        labels[label] = tile
    return sheet, labels


class Matcher:
    def __init__(self, project: dict, shots: list[dict], check, progress):
        self.project = project
        self.pid = project["id"]
        self.shots = shots
        self.check = check
        self.progress = progress
        self.started = time.monotonic()
        self.cfg = config.read()
        self.bili = es.Bilibili()
        key = es.stock_key()
        self.pixa = es.Pixabay(key) if key else None
        self.plan = project.get("plan_info") or {}
        self.min_height = int(project.get("min_video_height", 720) or 720)
        self.notes = []
        self.lock = threading.Lock()
        self.calls = {"pick": 0, "qa": 0}
        self.timing = {}
        self.fetch_seconds = []
        self.encoders = {}    # how candidate clips were encoded: full / nvenc / cpu -> count
        self.model_seconds = {}   # task -> seconds spent in model calls (summed over parallel calls)
        try:
            self.elsewhere = store.footage_used_elsewhere(self.pid)
        except Exception:
            self.elsewhere = {}
        self.text_found = {"contact": 0, "caption": 0, "center": 0, "sources_contact": 0, "sources_captioned": 0}
        self.qa_unchecked = set()
        self.job_choice = {}
        self.subtitled = set()
        self.clean = {}       # job id -> express_clean analysis of the downloaded window
        self.pool = None      # downloads outlive a round: on a slow link their progress is kept
        self.inflight = {}    # future -> (job_id, sid, choice)
        self.abandon = threading.Event()
        self.soft = {}        # job id -> {"wm", "still"} flags from the visual check
        self.profiles = {}
        self.sheets = OrderedDict()   # (source, sheet) -> grey preview cells, most recently used last
        self.timeline = project["shots"]
        self.shots_by_id_all = {s["id"]: s for s in project["shots"]}
        self.position = {s["id"]: i for i, s in enumerate(self.timeline)}
        self.placed = self.existing_placements()
        self.looks = {}       # timeline position -> dHash of the chosen picture (repetition check)
        # Every named person in this script: a dramatisation about them must not slip in as filler either.
        self.names = {g.get("person") for g in self.plan.get("groups", []) if g.get("person")}
        self.names |= {p["name"] for s in project["shots"] for p in s.get("persons") or []}
        self.unfilmed = set()   # names no candidate source carries (no real footage of them was found)
        self.ancient = set(self.plan.get("ancient") or [])   # people from before photography (no real footage can exist)
        self.host_hits, self.host_sources = {}, set()   # sources whose windows keep showing a presenter on camera
        self.presenter_sources = set()                  # presenter-led videos found from their preview frames

    @staticmethod
    def share() -> int:
        """Videos being made at the same time share the CPU and the download slots, so every wait of each one
        stretches about that much; the time limits below were tuned for one video alone and scale with it."""
        try:
            with store.connection() as con:
                running = con.execute("SELECT COUNT(*) FROM jobs WHERE status='running' "
                                      "AND kind IN ('produce','match')").fetchone()[0]
        except Exception:
            return 1
        return max(1, min(throttle.MAX_VIDEOS, int(running)))

    def chat(self, system, content, *, max_tokens, task, timeout=40, attempts=3):
        """Vision call with short timeouts and retries; Germany->Beijing calls occasionally stall."""
        last = None
        for attempt in range(attempts):
            self.check()
            began = time.monotonic()
            try:
                return planner.chat(system, content, vision=True, max_tokens=max_tokens, timeout=timeout, task=task)
            except ValueError as exc:
                message = str(exc)
                if not any(k in message for k in ("连接失败或超时", "HTTP 5", "没有返回有效 JSON", "响应必须是 JSON")):
                    raise
                last = exc
                time.sleep(1.5 * (attempt + 1))
            finally:
                with self.lock:
                    self.model_seconds[task] = self.model_seconds.get(task, 0.0) + time.monotonic() - began
        raise last

    # ------------------------------------------------------------------ groups
    def groups(self) -> list[dict]:
        wanted = {s["id"] for s in self.shots}
        groups = []
        for g in self.plan.get("groups", []):
            ids = [sid for sid in g.get("shots", []) if sid in wanted]
            if ids:
                groups.append(g | {"shots": ids})
        known = {sid for g in groups for sid in g["shots"]}
        for shot in self.shots:
            if shot["id"] not in known:
                express = shot.get("express") or {}
                groups.append({"id": f"R{shot['id'][:6]}", "zh": express.get("zh") or [shot.get("query") or shot["narration"][:30]],
                               "en": express.get("en", []),
                               "person": (shot.get("persons") or [{}])[0].get("name", ""), "shots": [shot["id"]]})
        topic = self.plan.get("topic") or {}
        groups.append({"id": "TOPIC", "zh": topic.get("zh", [])[:2], "en": topic.get("en", [])[:2],
                       "person": "", "shots": [], "topic": True})
        return groups

    # ------------------------------------------------------------------ search
    def search(self, groups):
        tasks = []
        for g in groups:
            for q in g["zh"][:2]:
                tasks.append(("bili", q))
            if self.pixa:
                for q in g["en"][:2]:
                    tasks.append(("pixa", q))
        tasks = list(dict.fromkeys(tasks))
        results = {}
        def run(task):
            self.check()
            kind, q = task
            try:
                return task, (self.bili.search(q) if kind == "bili" else self.pixa.search(q))
            except Exception as exc:  # network/provider errors only cost this query
                self.notes.append(f"检索失败 {kind}:{q}: {type(exc).__name__}")
                return task, []
        with execution.ThreadPoolExecutor(max_workers=6) as pool:
            for task, rows in pool.map(run, tasks):
                results[task] = rows
        asked = [rows for (kind, _), rows in results.items() if kind == "bili"]
        if len(asked) >= 4 and not any(asked):
            raise BiliUnavailable(f"B站的 {len(asked)} 次搜索全部没有结果（多半是被限流或连不上B站）")
        # The local index: videos found and described before (no network), with the matching moments.
        from . import footage_index
        for q in dict.fromkeys(q for g in groups for q in g["zh"][:2]):
            try:
                results[("local", q)] = footage_index.search(q, limit=8)
            except Exception as exc:   # a missing or damaged index only costs this lane
                self.notes.append(f"本地素材库检索失败: {type(exc).__name__}")
                break
        return results

    def score(self, row, group, shots_by_id):
        text = " ".join((row.get("title", ""), row.get("tags", ""), row.get("description", "")))
        ancient_names = getattr(self, "ancient", set())
        ancient = group.get("person") in ancient_names
        if group.get("person"):
            name = group["person"]
            title = row.get("title", "")
            if name not in text:
                return None
            # Drama clips are how a pre-photography figure is shown; for anyone else they are actors.
            if not ancient and (DRAMA_WORDS.search(title + row.get("tags", "")) or row.get("typename", "") in DRAMA_TYPES
                                or DRAMA_TAGS.search(row.get("tags", "")) or DIALOGUE_TITLE.search(title)):
                return None
        elif any(name in row.get("title", "") and name not in ancient_names for name in getattr(self, "names", ())) and \
                DRAMA_WORDS.search(row.get("title", "") + row.get("tags", "")):
            return None
        words = []
        for q in group["zh"] + group["en"]:
            words += _tokens(q) if row["provider"] == "bilibili" else [w.lower() for w in re.findall(r"[A-Za-z]{3,}", q)]
        words = list(dict.fromkeys(words))
        haystack = text.lower()
        overlap = sum(1 for w in words if w.lower() in haystack) / max(1, len(words))
        score = overlap * 2
        duration = row.get("duration", 0)
        need = max([float(shots_by_id[s]["duration"]) for s in group["shots"]] or [8])
        if row["provider"] == "bilibili":
            if duration < 12 or duration > 3 * 3600:
                return None
            if SKIPPED_UPLOADS.search(row.get("title", "") + row.get("tags", "")):
                return None
            if NOISE_WORDS.search(row.get("title", "")):
                score -= 1.2
            if FOOTAGE_WORDS.search(text):
                score += 0.5
            if group.get("person") and PERSON_EVIDENCE.search(text):
                score += 0.5
            if group.get("person") and ARCHIVE_TITLE.search(row.get("title", "")):
                score += 0.6
            score += 0.3 if 45 <= duration <= 1200 else 0
            score += min(0.3, math.log10(1 + row.get("play", 0)) / 20)
            if row.get("moments"):
                score += 0.6    # described frames of this video match the words, not just its title
        else:
            if duration + 0.05 < min(need, 6):
                return None
            score += 0.4 if duration >= need + 1 else -0.4
        return score

    def candidates(self, groups, results):
        """Per group: top Bilibili results of EACH Chinese query, plus stock clips whose tags match."""
        shots_by_id = {s["id"]: s for s in self.shots}
        chosen = {}
        for g in groups:
            picked, seen = [], set()
            bili_quota = PERSON_PER_QUERY if g.get("person") else BILI_PER_QUERY
            for kind, queries, quota in (("local", g["zh"][:2], LOCAL_PER_QUERY), ("bili", g["zh"][:2], bili_quota),
                                         ("pixa", g["en"][:2] if self.pixa else [], PIXA_PER_QUERY)):
                for q in queries:
                    ranked = []
                    for rank, row in enumerate(results.get((kind, q), [])):
                        score = self.score(row, g, shots_by_id)
                        if score is None:
                            continue
                        if kind == "pixa" and (not self.stock_fits(row, q) or not self.oriented(row)):
                            continue
                        ranked.append((score - rank * 0.04, row))
                    ranked.sort(key=lambda item: -item[0])
                    taken = 0
                    for _, row in ranked:
                        key = (row["provider"], row["id"])
                        title_key = (row["provider"], row.get("title", "")[:30])
                        if key in seen or title_key in seen:
                            continue
                        seen.update((key, title_key))
                        picked.append(row)
                        taken += 1
                        if taken >= quota:
                            break
            chosen[g["id"]] = picked
        return chosen

    def oriented(self, source) -> bool:
        """A portrait stock clip in a landscape project would be a thin strip; landscape stock crops fine to 9:16."""
        r = source.get("rendition") or {}
        if not (r.get("width") and r.get("height")) or self.project.get("aspect", "16:9") == "9:16":
            return True
        return r["width"] >= r["height"]

    def stock_fits(self, row, query) -> bool:
        """A stock clip whose tags share one word with a three-word query ("warship sailing sea" and a Monaco
        yacht tagged "sea") is off-topic: a query of two or more words needs two of them in the tags."""
        words = len(re.findall(r"[A-Za-z]{3,}", query))
        return self.stock_overlap(row, query) >= min(2, max(1, words))

    @staticmethod
    def stock_overlap(row, query) -> int:
        tags = {t.strip().lower() for t in row.get("tags", "").split(",")}
        words = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", query)}
        tag_words = {w for t in tags for w in t.split()}
        return len(words & tag_words)

    # ----------------------------------------------------------------- tiles
    def tiles_for(self, row, count=TILES_PER_BILI) -> list[dict]:
        if row["provider"] == "pixabay":
            image = self.pixa.thumbnail(row)
            if image is None or _dark(image):
                return []
            return [{"source": row, "t": row["duration"] / 2, "image": image, "hash": _dhash(image)}]
        previews = self.bili.previews(row["id"], row["duration"])
        if not previews:
            return []
        duration = row["duration"] or previews[-1]["t"]
        # The first seconds are often the uploader's intro card (name, logo, a different layout): skip them.
        inner = [p for p in previews if max(duration * 0.04, intro_seconds(duration)) <= p["t"] <= duration * 0.96] or previews
        step = max(1, len(inner) / count)
        picked = [inner[min(len(inner) - 1, int(i * step + step / 2))] for i in range(min(count, len(inner)))]
        if row.get("moments") or row.get("avoid"):
            # Indexed video: the frames whose descriptions matched come first, the even spread fills up;
            # nothing near a frame described as a presenter, a text page or black.
            near = [min(inner, key=lambda p, t=t: abs(p["t"] - t)) for t in row.get("moments") or []]
            avoid = row.get("avoid") or []
            picked = [p for p in {id(p): p for p in near + picked}.values()
                      if not any(abs(p["t"] - t) < 6 for t in avoid)][:count]
        tiles, hashes = [], []
        for p in picked:
            image = es.bilibili_tile(self.bili, p)
            if _dark(image):
                continue
            h = _dhash(image)
            if any(bin(h ^ other).count("1") <= 5 for other in hashes):
                continue
            hashes.append(h)
            tiles.append({"source": row, "t": p["t"], "image": image, "hash": h})
        if tiles and row["provider"] == "bilibili":
            # Kept full size for reading the source's text, which happens only if the picker chooses it.
            row["_screen"] = [t["image"] for t in tiles][:self.SCREEN_FRAMES]
        for tile in tiles:
            # Contact sheets paste tiles at TILE size: keep only that (a full 480x270 frame per tile, ten tiles
            # for each of ~90 sources, was most of a worker's memory with several videos at once).
            tile["image"] = tile["image"].resize((TILE_W, TILE_H))
        if express_clean.available():
            try:
                frames = self.sprite_frames(row, picked[0]["t"] if picked else 0)
                frames = frames[::max(1, len(frames) // 120)]
                size = (express_clean.WIDTH, round(express_clean.WIDTH * 9 / 16))
                profile = express_clean.profile([im for _, im in frames], size)
                with self.lock:
                    self.profiles[(row["id"], size)] = profile
                if express_clean.central_marks(profile):
                    row["dirty"] = True   # an uploader mark across the middle of the picture
            except Exception as exc:  # the check is an optimisation only
                self.notes.append(f"水印预检失败 {row['id']}: {type(exc).__name__}")
        return tiles

    SCREEN_FRAMES = 3

    def screen_source(self, row, images):
        """Read a few preview frames of a whole video before any download: an uploader's contact overlay
        (需要视频私信 / 加微信 / QQ) usually runs through all of it, as do burned-in subtitles. Cached per video."""
        if not screen_text.installed():
            return
        cache = es._cache("text") / f"{row['id']}.json"
        try:
            found = json.loads(cache.read_text("utf-8"))
        except (OSError, ValueError):
            usable = [im for im in images if im.width >= 300][:self.SCREEN_FRAMES]   # 160x90 sheets are too small
            if not usable:
                return
            try:
                with throttle.slot("frames", self.check):
                    reads = [screen_text.classify([screen_text.read(im)]) for im in usable]
            except media.Cancelled:
                raise
            except Exception as exc:
                self.notes.append(f"预览文字识别失败 {row['id']}: {type(exc).__name__}")
                return
            found = {"frames": len(reads), "contact": sorted({t for r in reads for t in r["contact"]}),
                     "caption": sum(1 for r in reads if any(m["kind"] == "caption" for m in r["marks"])),
                     "center": sum(1 for r in reads if r["dense"] and not r["contact"])}
            es._write_json(cache, found)
        if found.get("contact"):
            row["contact"] = True
            with self.lock:
                self.text_found["sources_contact"] += 1
        elif found.get("frames"):
            if found.get("caption", 0) * 2 >= found["frames"]:
                row["captioned"] = True
                with self.lock:
                    self.text_found["sources_captioned"] += 1
            if found.get("center", 0) * 2 >= found["frames"]:
                row["dirty"] = True

    def all_tiles(self, chosen, person_groups=()):
        sources, person = {}, set()
        for gid, rows in chosen.items():
            for row in rows:
                sources[(row["provider"], row["id"])] = row
                if gid in person_groups:
                    person.add((row["provider"], row["id"]))
        tiles = {}
        def run(item):
            key, row = item
            self.check()
            try:
                return key, self.tiles_for(row, PERSON_TILES if key in person else TILES_PER_BILI)
            except Exception as exc:
                self.notes.append(f"预览失败 {key[1]}: {type(exc).__name__}")
                return key, []
        with execution.ThreadPoolExecutor(max_workers=8) as pool:
            for key, rows in pool.map(run, list(sources.items())):
                tiles[key] = rows
        return tiles

    # ------------------------------------------------------------ presenters
    def screen_presenters(self, chosen, tiles=None) -> set:
        """Commentary videos where a presenter talks to camera much of the time: their windows keep failing
        QA (each failure costs a download round). One cheap look at 8 preview frames per video, 8 videos a
        request, cached per video (the produce warm-up fills the cache while the narration is made)."""
        rows = {(r["provider"], r["id"]): r for rs in chosen.values() for r in rs if r["provider"] == "bilibili"
                and (tiles is None or tiles.get((r["provider"], r["id"])))}   # previews that failed are not retried
        found, todo = set(), []
        for key, row in rows.items():
            cache = es._cache("presenter") / f"{row['id']}.json"
            try:
                if json.loads(cache.read_text())["presenter"]:
                    found.add(key)
                continue
            except (OSError, ValueError, KeyError, TypeError):
                todo.append((key, row, cache))
        def frames(row):
            self.check()
            previews = self.bili.previews(row["id"], row["duration"]) or []
            duration = row["duration"] or (previews[-1]["t"] if previews else 0)
            inner = [p for p in previews if duration * 0.04 <= p["t"] <= duration * 0.96] or previews
            step = max(1, len(inner) / 8)
            return [es.bilibili_tile(self.bili, inner[min(len(inner) - 1, int(i * step + step / 2))])
                    for i in range(min(8, len(inner)))]
        def ask(batch):
            self.check()
            sheet = Image.new("RGB", (60 + 8 * 160, len(batch) * 90), (0, 0, 0))
            draw, font, ids = ImageDraw.Draw(sheet), _font(26), {}
            for r, (key, row, cache, images) in enumerate(batch):
                label = f"V{r + 1}"
                ids[label] = (key, cache)
                draw.text((6, r * 90 + 28), label, font=font, fill=(255, 214, 0))
                for c, image in enumerate(images):
                    sheet.paste(image.resize((160, 90)), (60 + c * 160, r * 90))
            result, _ = self.chat(PRESENTER_SYSTEM, [{"type": "text", "text": f"共 {len(batch)} 行（V1-V{len(batch)}）"},
                                                     {"type": "image_url", "image_url": {"url": _jpeg(sheet)}}],
                                  max_tokens=200, task="express_qa", attempts=2)
            flagged = {str(v).strip() for v in (result.get("presenter") or [])} if isinstance(result, dict) else set()
            out = set()
            for label, (key, cache) in ids.items():
                hit = label in flagged
                es._write_json(cache, {"presenter": hit})
                if hit:
                    out.add(key)
            return out
        prepared = []
        for key, row, cache in todo:
            try:
                images = frames(row)
            except Exception:
                images = []
            if len(images) >= 4:
                prepared.append((key, row, cache, images))
        batches = [prepared[i:i + 8] for i in range(0, len(prepared), 8)]
        if batches:
            with execution.ThreadPoolExecutor(max_workers=4) as pool:
                for result in pool.map(lambda b: self._safe(ask, b), batches):
                    found |= result or set()
        return found

    def _safe(self, function, *args):
        try:
            return function(*args)
        except media.Cancelled:
            raise
        except Exception as exc:
            if type(exc).__name__ in ("BudgetExceeded", "ModelUnavailable", "PricingUnavailable"):
                raise
            self.notes.append(f"主持人筛查失败：{type(exc).__name__}")
            return set()

    # ------------------------------------------------------------------ picks
    def pick(self, group, tiles, shots_by_id):
        if not tiles:
            return {}, {}
        tiles = tiles[:3 * SHEET_COLS * SHEET_ROWS]
        sheets, labels = [], {}
        for start in range(0, len(tiles), SHEET_COLS * SHEET_ROWS):
            sheet, part = _sheet(tiles[start:start + SHEET_COLS * SHEET_ROWS], start + 1)
            sheets.append(sheet)
            labels.update(part)
        if group.get("topic"):
            lines = [f"S1：{TOPIC_SHOT}（主题：{'、'.join(group['zh'])}）"]
            ids = {"S1": "TOPIC"}
        else:
            lines, ids = [], {}
            for n, sid in enumerate(group["shots"], 1):
                shot = shots_by_id[sid]
                extra = ""
                names = [x["name"] for x in shot.get("persons") or []]
                if names and group.get("person") not in names:
                    extra = f"｜注意：口播提到{'、'.join(names)}，本组素材不能确认其身份，只选不含可辨认人物的相关画面"
                lines.append(f"S{n}（约{float(shot['duration']):.0f}秒）口播：{shot['narration']}｜画面建议：{shot.get('visual', '')}{extra}")
                ids[f"S{n}"] = sid
        system = PICK_SYSTEM
        if group.get("person"):
            rules = ANCIENT_RULES if group["person"] in self.ancient else PERSON_RULES
            system += rules.format(names=group["person"])
        requirement = (self.project.get("requirements") or "").strip()
        domain = self.plan.get("domain") or ""
        era = ("\n本片是古代历史题材：古装影视剧、纪录片情景再现、历史动画都可以选，优先选有动作的影视画面；不选现代场景、外国场景"
               if self.ancient else "")
        text = ("用户画面要求：" + (requirement or "无") + era + "\n全文主题：" + self.project.get("title", "")
                + (f"\n整篇领域：{domain}（其他领域的画面不选，如电竞、别的运动、卡通、3D渲染）" if domain else "")
                + "\n镜头：\n" + "\n".join(lines))
        content = [{"type": "text", "text": text}] + [
            {"type": "image_url", "image_url": {"url": _jpeg(s)}} for s in sheets]
        cache_key = hashlib.sha256((system + text + "".join(
            f"{t['source']['provider']}:{t['source']['id']}:{t['t']:.1f}" for t in tiles)).encode()).hexdigest()[:40]
        cache = es._cache("picks") / f"{cache_key}.json"
        result = es._read_json(cache, 30 * 24 * 3600)
        if result is None:
            self.check()
            result, _ = self.chat(system, content, max_tokens=120 + 60 * len(ids), task="express_pick")
            with self.lock:
                self.calls["pick"] += 1
            es._write_json(cache, result)
        picks, ctx = {}, []
        for key, value in result.items() if isinstance(result, dict) else []:
            values = [str(v).strip() for v in (value if isinstance(value, list) else []) if str(v).strip() in labels]
            if key == "_ctx":
                ctx = values
            elif key in ids:
                picks[ids[key]] = [labels[v] for v in dict.fromkeys(values)][:6]
        return picks, {"ctx": [labels[v] for v in dict.fromkeys(ctx)]}

    def stock_options(self, shots, groups, results, final) -> dict:
        """Pixabay clips from each open line's own searches that were not offered yet (relevant by their tags)."""
        group_of = {sid: g for g in groups for sid in g["shots"]}
        shown = {c["tile"]["source"]["id"] for c in final.values() if c["tile"]["source"].get("provider") == "pixabay"}
        options = {}
        for shot in shots:
            group = group_of.get(shot["id"])
            if not group:
                continue
            rows = []
            for q in group.get("en", [])[:3]:
                for row in results.get(("pixa", q), []):
                    if (row["id"] not in shown and row not in rows and self.stock_fits(row, q)
                            and self.oriented(row) and float(row.get("duration") or 0) >= float(shot["duration"]) + MARGIN):
                        rows.append(row)
            if rows:
                options[shot["id"]] = [({"source": row, "t": float(row.get("duration") or 0) / 2, "hash": None}, "stock")
                                       for row in rows[:12]]
        return options

    def borrow(self, shot, final, files, limit=None):
        """The nearest accepted clip (never the next one: a clip twice in a row looks like a glitch) that is long
        enough and shows no other named person; anywhere in the video, as one about the same subject. limit: skip
        clips whose video already fills this many shots (borrow from the less used ones first)."""
        position, need = self.position[shot["id"]], float(shot["duration"]) - 0.15
        counts = {}
        for choice in final.values():
            key = (choice["tile"]["source"].get("provider"), choice["tile"]["source"].get("id"))
            counts[key] = counts.get(key, 0) + 1
        for distance in range(2, len(self.timeline)):
            for other in (position - distance, position + distance):
                if not 0 <= other < len(self.timeline):
                    continue
                sid = self.timeline[other]["id"]
                choice = final.get(sid)
                if not choice or choice.get("lane") in ("card", "reuse") or sid not in files:
                    continue
                source = choice["tile"]["source"]
                if limit and counts.get((source.get("provider"), source.get("id")), 0) >= limit:
                    continue
                donor_shot = self.shots_by_id_all.get(sid) or {}
                if shot.get("persons"):
                    if not (self.names_in_source(shot, choice["tile"]["source"]) or choice.get("lane") in ("ctx", "topic")):
                        continue
                elif donor_shot.get("persons") and choice.get("lane") in ("pick", "person", "person_near"):
                    continue    # that clip shows someone by name
                try:
                    if es.probe_duration(files[sid]) < need:
                        continue
                except ValueError:
                    continue
                return sid
        return None

    # ------------------------------------------------------------ pictures
    IMAGE_PICK = """每一行是一句口播（行首黄色编号）和几张网上找来的候选图片（行内从左到右是第1、2、3、4张）。
为每句挑一张最适合放在这句话下面的图片：要和口播讲的人、事、地方对得上，画面干净清楚。
不要：文字多的图（海报、截图、图表、聊天记录）；带明显水印、网站logo或二维码的；表情包、漫画、卡通；
画质很差或模糊的；人脸打了马赛克的；口播讲中国的人和事却明显是外国人或外国街景的。都不合适就填 0。
只输出 JSON：{"S3":2,"S5":0}"""
    IMAGE_WORDS = re.compile(r"电视剧片段|电视剧|短剧|栏目剧|片段|实拍|视频|素材|现场|新闻|航拍|空镜|纪录片|历史影像|画面|镜头")
    # Stock-photo sites whose pictures carry their watermark across the middle.
    IMAGE_SKIP_HOSTS = re.compile(r"699pic|588ku|58pic|ibaotu|nipic|huitu|quanjing|veer|51miz|shutterstock|gettyimages|"
                                  r"istockphoto|dreamstime|alamy|123rf|depositphotos|vcg\.com|ooopic|zcool|sucai")

    def image_fill(self, shots) -> dict:
        """Lines still without footage: related web pictures (shown with slow motion by the render), chosen by
        the vision model from up to four candidates a line. {shot id: (choice, path)}; lines nothing fits are
        left out."""
        from . import publisher_videos, screen_text, sources
        from PIL import ImageOps
        root = paths.project_dir(self.pid) / "images"
        root.mkdir(exist_ok=True)
        groups = {sid: g for g in self.plan.get("groups", []) for sid in g.get("shots", [])}
        def queries(shot):
            words = []
            for q in (groups.get(shot["id"]) or {}).get("zh", [])[:2] + [shot.get("visual") or ""]:
                q = " ".join(self.IMAGE_WORDS.sub(" ", q).split())
                if q and q not in words:
                    words.append(q)
            return words[:2]
        seen_hashes = []
        lock = threading.Lock()
        def candidates(shot):
            rows = []
            for q in queries(shot):
                for search in (sources.bing_images, publisher_videos.chinanews_images):
                    try:
                        rows += search(q)[:8]
                    except Exception:   # noqa: BLE001 — one search less
                        continue
            pictures = []
            for row in rows:
                if len(pictures) >= 4:
                    break
                url = row.get("url") or ""
                if not url or self.IMAGE_SKIP_HOSTS.search(url + " " + str(row.get("source_url") or row.get("source") or "")):
                    continue
                self.check()
                try:
                    with network.client(timeout=12, follow_redirects=True) as client:
                        response = client.get(url, headers={"User-Agent": BROWSER})
                    if response.status_code != 200 or len(response.content) > 12 * 2**20:
                        continue
                    with Image.open(io.BytesIO(response.content)) as raw:
                        image = ImageOps.exif_transpose(raw).convert("RGB")
                except Exception:   # noqa: BLE001 — broken or refused picture
                    continue
                if min(image.size) < 400 or not 0.45 <= image.width / image.height <= 2.6:
                    continue
                h = _dhash(image)
                with lock:
                    if any(bin(h ^ other).count("1") <= 6 for other in seen_hashes):
                        continue
                    seen_hashes.append(h)
                if screen_text.installed():
                    try:
                        found = screen_text.classify([screen_text.read(image)])
                    except Exception:   # noqa: BLE001
                        found = {"contact": [], "dense": False, "moving": False}
                    if found["contact"] or found["dense"]:
                        continue    # a poster, a screenshot or an ad
                image.thumbnail((1920, 1920))
                pictures.append((row, image, h))
            return shot, pictures
        found = {}
        with execution.ThreadPoolExecutor(max_workers=4) as pool:
            for shot, pictures in pool.map(candidates, shots):
                if pictures:
                    found[shot["id"]] = pictures
        if not found:
            return {}
        chosen = {}
        items = list(found.items())
        for start in range(0, len(items), 6):
            batch = items[start:start + 6]
            labels = {f"S{self.position[sid] + 1}": sid for sid, _ in batch}
            w, h = TILE_W, TILE_H
            sheet = Image.new("RGB", (w * 4 + 8 * 3, (h + 8) * len(batch)), (30, 30, 30))
            draw = ImageDraw.Draw(sheet)
            font = _font(24)
            lines = []
            for row_index, (sid, pictures) in enumerate(batch):
                for k, (_, image, _) in enumerate(pictures):
                    sheet.paste(ImageOps.fit(image, (w, h)), (k * (w + 8), row_index * (h + 8)))
                label = f"S{self.position[sid] + 1}"
                box = draw.textbbox((4, row_index * (h + 8) + 2), label, font=font)
                draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(255, 214, 0))
                draw.text((4, row_index * (h + 8) + 2), label, font=font, fill=(0, 0, 0))
                lines.append(f"{label}（{len(pictures)} 张）：{self.shots_by_id_all[sid]['narration'][:60]}")
            content = [{"type": "text", "text": "\n".join(lines)}, {"type": "image_url", "image_url": {"url": _jpeg(sheet)}}]
            try:
                result, _ = self.chat(self.IMAGE_PICK, content, max_tokens=60 + 12 * len(batch), task="express_pick")
            except Exception as exc:   # noqa: BLE001 — these lines go on to borrowing
                self.notes.append(f"挑图失败：{str(exc)[:60]}")
                continue
            for label, value in (result or {}).items() if isinstance(result, dict) else []:
                sid = labels.get(str(label).strip())
                if sid and isinstance(value, int) and 1 <= value <= len(found[sid]):
                    chosen[sid] = found[sid][value - 1]
        out = {}
        for sid, (row, image, h) in chosen.items():
            name = hashlib.sha1((row.get("url") or str(h)).encode()).hexdigest()[:16]
            path = root / f"{name}.jpg"
            image.save(path, "JPEG", quality=92)
            source = {"provider": "web_image", "id": name, "title": (row.get("title") or "网页图片")[:80],
                      "url": row.get("source_url") or row.get("source") or row.get("url") or "",
                      "license": row.get("license") or "网页图片，使用许可待核实"}
            out[sid] = ({"tile": {"source": source, "t": 0.0, "hash": h}, "start": 0.0, "lane": "image", "rank": 0,
                         "qa": "image"}, path)
        return out

    def title_card(self, shot, backdrop=None):
        """Neutral image card (counts as image seconds, never as fake footage): a blurred, darkened frame
        of nearby accepted footage, or a plain gradient, with no text on it; the narration subtitle is burned
        in later, and planner notes never appear on screen."""
        w, h = (1280, 720) if self.project.get("aspect", "16:9") != "9:16" else (720, 1280)
        image = None
        if backdrop:
            try:
                frame = Path(backdrop).with_suffix(".card.jpg")
                subprocess.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-ss", "0.5", "-i", str(backdrop),
                                "-frames:v", "1", str(frame)], check=True, timeout=30)
                with Image.open(frame) as source:
                    image = ImageOps.fit(source.convert("RGB"), (w, h)).filter(ImageFilter.GaussianBlur(24))
                image = ImageEnhance.Brightness(image).enhance(.45)
                frame.unlink(missing_ok=True)
            except (OSError, subprocess.SubprocessError, ValueError):
                image = None
        if image is None:
            image = Image.new("RGB", (w, h), (18, 22, 30))
            draw = ImageDraw.Draw(image)
            for y in range(h):
                shade = int(18 + 22 * y / h)
                draw.line([(0, y), (w, y)], fill=(shade, shade + 4, shade + 12))
        # No words on the card: a name in the middle of the picture reads as junk text (customers asked for
        # nothing but the subtitles); the narration subtitle says who it is.
        root = paths.project_dir(self.pid) / "assets"
        root.mkdir(exist_ok=True)
        path = root / f"express-card-{shot['id'][:8]}.jpg"
        image.save(path, quality=92)
        return path

    def note_presenters(self, bad):
        """A source rejected twice for a presenter talking to camera is a commentary video: take no more
        windows from it (each would cost a download and a QA look for the same result)."""
        for job_id, reason in bad.items():
            # Presenter words only ("与口播内容无关" is an ordinary off-topic rejection).
            if not re.search(r"主持|博主|嘉宾|对镜|对着镜头|面对镜头|镜头讲话|出镜|演播室|口播画面", str(reason)):
                continue
            source = ((self.job_choice.get(job_id) or {}).get("tile") or {}).get("source") or {}
            if not source:
                continue
            key = (source.get("provider"), source.get("id"))
            shots = self.host_hits.setdefault(key, set())
            shots.add(job_id.split("#")[0])          # count distinct shots, not candidate jobs
            if len(shots) >= 2:
                self.host_sources.add(key)

    def ancient_shot(self, shot) -> bool:
        """Every named person of the line lived before photography: portraits and dramatisations stand in."""
        return bool(shot.get("persons")) and all(p["name"] in getattr(self, "ancient", set()) for p in shot["persons"])

    def unfilmed_shot(self, shot) -> bool:
        """A named-person line whose people have no real footage at all (e.g. a Qing-dynasty cook):
        it takes related background footage like an ordinary line instead of falling back to a card."""
        return bool(shot.get("persons")) and all(p["name"] in self.unfilmed for p in shot["persons"])

    @staticmethod
    def names_in_source(shot, source) -> bool:
        text = " ".join(str(source.get(k, "")) for k in ("title", "tags", "description"))
        return all(p["name"] in text for p in shot.get("persons") or [])

    # -------------------------------------------------------------- allocation
    def window(self, tile, duration, reserved):
        source = tile["source"]
        length = float(source.get("duration") or 0)
        need = duration + MARGIN
        if length < need + 0.1:
            return None
        t = float(tile["t"])
        key = (source["provider"], source["id"])
        taken = reserved.get(key, [])
        lo = max(0.0, t - need + 1.0, min(intro_seconds(length), t - 0.5))
        hi = min(length - need - 0.1, t - 1.0 if need > 2.5 else t)
        if hi < lo:
            lo, hi = 0.0, max(0.0, length - need - 0.1)
        preferred = min(max(t - need * 0.3, lo), hi)
        steps = [preferred] + [preferred + d * s for s in range(1, 40) for d in (0.5, -0.5)]
        for start in steps:
            if not lo - 1e-6 <= start <= hi + 1e-6:
                continue
            end = start + need
            if all(end + WINDOW_GAP <= a or start >= b + WINDOW_GAP for a, b in taken):
                return round(start, 2)
        return None

    def existing_placements(self) -> dict:
        assets = {a["id"]: a for a in self.project.get("assets", [])}
        wanted = {s["id"] for s in self.shots}
        placed = {}
        for index, shot in enumerate(self.timeline):
            asset = assets.get(shot.get("selected_asset")) or {}
            candidate = str(asset.get("candidate_id", ""))
            if shot["id"] not in wanted and ":" in candidate:
                placed[index] = tuple(candidate.split(":", 1))
        return placed

    def existing_windows(self) -> dict:
        wanted = {s["id"] for s in self.shots}
        assets = {a["id"]: a for a in self.project.get("assets", [])}
        reserved = {}
        for shot in self.project.get("shots", []):
            if shot["id"] in wanted:
                continue
            asset = assets.get(shot.get("selected_asset")) or {}
            provider_id = str(asset.get("candidate_id", ""))
            if ":" not in provider_id or asset.get("download_mode") != "express":
                continue
            provider, source_id = provider_id.split(":", 1)
            start = float(asset.get("source_clip_start") or 0) + float(shot.get("clip_start") or 0)
            reserved.setdefault((provider, source_id), []).append((start, start + float(shot["duration"])))
        return reserved

    def place(self, sid, choice, key):
        self.placed[self.position[sid]] = key
        if choice["tile"].get("hash") is not None:
            self.looks[self.position[sid]] = choice["tile"]["hash"]

    def allocate(self, options, reserved, usage, shots, exclude=None):
        exclude = exclude or {}
        assigned = {}
        assigned_keys = dict(self.placed)
        looks = dict(self.looks)
        cap = max(HARD_MIN, math.ceil(len(self.timeline) * SOURCE_SHARE))
        hard_cap = max(HARD_MIN, math.ceil(len(self.timeline) * HARD_SHARE))
        # The opening shot chooses first (it decides whether viewers stay), then the shots with fewest options.
        order = sorted(shots, key=lambda s: (self.position[s["id"]] != 0, len(options.get(s["id"], [])),
                                             self.position[s["id"]]))
        stock_cap = max(2, int(len(self.timeline) * STOCK_SHARE))
        stock_used = sum(1 for key in assigned_keys.values() if key and key[0] == "pixabay")
        for shot in order:
            sid = shot["id"]
            best = None
            for rank, (tile, lane) in enumerate(options.get(sid, [])):
                key = (tile["source"]["provider"], tile["source"]["id"])
                if (key, round(tile["t"], 1)) in exclude.get(sid, set()) or tile["source"].get("contact"):
                    continue    # an uploader who puts 加微信 / phone numbers on the picture: never
                start = self.window(tile, float(shot["duration"]), reserved)
                if start is None:
                    continue
                index = self.position[sid]
                adjacent = sum(1 for other in (index - 1, index + 1)
                               if other in assigned_keys and assigned_keys[other] == key)
                used = usage.get(key, 0)
                near = min((abs(start - a) for a, _ in reserved.get(key, [])), default=1e9)
                h = tile.get("hash")
                close = [pos for pos, other in looks.items()
                         if h is not None and pos != index and bin(h ^ other).count("1") <= TWIN_BITS]
                # The same picture further away is a repeat; right next to it from the same video it is the
                # scene continuing, from another video a re-upload of it.
                twin = any(abs(pos - index) > 1 for pos in close)
                reupload = any(abs(pos - index) == 1 and assigned_keys.get(pos) != key for pos in close)
                if key[0] == "pixabay" and stock_used >= stock_cap:
                    continue
                if lane not in ("person", "person_near") and not (shot.get("persons") and not self.unfilmed_shot(shot)):
                    # Hard limits (scores alone let one long stock take fill ten shots in a row): a source at
                    # most HARD_SHARE of the film, never three shots running, never the same picture again.
                    run = [assigned_keys.get(i) for i in (index - 2, index - 1, index + 1, index + 2)]
                    # A news report's windows show different scenes: it may fill more lines (20 s apart).
                    if (used >= (NEWS_CAP if lane == "news" else hard_cap) or twin
                            or (run[1] == key and (run[0] == key or run[2] == key))
                            or (run[2] == key and run[3] == key)):
                        continue
                cost = (rank + 2.0 * used + 6.0 * max(0, used + 1 - cap) + 3.0 * adjacent
                        + (8.0 if near < SCENE_SECONDS else 0) + (8.0 if twin or reupload else 0)
                        # Footage other recent videos already show (a batch on one topic would repeat itself).
                        + min(6.0, 2.0 * self.elsewhere.get(f"{key[0]}:{key[1]}", 0))
                        + {"news": 0, "pick": 0, "person": 1.5, "person_near": 2.5, "ctx": 3, "topic": 4, "neighbor": 4,
                           "near": 6}.get(lane, 6)
                        # A line about a named person should show that person, not generic topic footage.
                        + (5 if lane == "topic" and shot.get("persons") and not self.unfilmed_shot(shot) else 0)
                        + (6 if tile["source"].get("dirty") else 0)
                        # Burned-in subtitles in most of its previews: cropped or blurred, but cleaner footage first.
                        + (3 if tile["source"].get("captioned") else 0)
                        + (4 if tile["source"].get("low_res") else 0)
                        # A presenter-led source is a last resort (its windows keep failing QA).
                        + (8 if key in self.host_sources else 0))
                if best is None or cost < best[0]:
                    best = (cost, tile, start, lane, rank)
                if rank > 8 and best:
                    break
            if best:
                _, tile, start, lane, rank = best
                key = (tile["source"]["provider"], tile["source"]["id"])
                reserved.setdefault(key, []).append((start, start + float(shot["duration"]) + MARGIN))
                usage[key] = usage.get(key, 0) + 1
                assigned[sid] = {"tile": tile, "start": start, "lane": lane, "rank": rank}
                assigned_keys[self.position[sid]] = key
                stock_used += key[0] == "pixabay"
                if tile.get("hash") is not None:
                    looks[self.position[sid]] = tile["hash"]
        return assigned

    # ---------------------------------------------------------------- download
    def fetch(self, shot, choice, streams, check=None):
        check = check or self.check
        source = choice["tile"]["source"]
        duration = float(shot["duration"]) + MARGIN
        root = paths.project_dir(self.pid) / "assets"
        root.mkdir(exist_ok=True)
        out = root / f"express-{shot['id'][:8]}-{source['id']}-{int(choice['start']*100)}.mp4"
        need = float(shot["duration"]) - 0.15   # the render holds the last frame; export allows 0.16 s (validate_export)
        try:
            ready = out.exists() and es.probe_duration(out) >= need
        except ValueError:
            ready = False
        if not ready:
            out.unlink(missing_ok=True)
            if source["provider"] == "pixabay":
                local = self.pixa.download(source, check)
                way = es.cut_window(str(local), choice["start"], duration, out, check=check, short_edge=self.min_height)
            else:
                stream = streams.get(source["id"]) or {}
                if not stream.get("available"):
                    raise ValueError("该视频没有满足最低清晰度的可用流")
                way = es.dash_window(stream, choice["start"], duration, out, check=check, short_edge=self.min_height)
            with self.lock:
                self.encoders[way] = self.encoders.get(way, 0) + 1
            got = es.probe_duration(out)
            if got < need:
                out.unlink(missing_ok=True)
                raise ValueError(f"下载片段时长不足（{got:.1f}/{need:.1f} 秒）")
        return out

    def resolve(self, sources: dict) -> dict:
        streams = {}
        def run(item):
            bvid, source = item
            self.check()
            try:
                stream = self.bili.stream(bvid, self.min_height)
                if (stream and not stream.get("available")
                        and LOW_HEIGHT <= int(stream.get("best_short_edge") or 0) < self.min_height):
                    lower = self.bili.stream(bvid, LOW_HEIGHT)
                    if lower and lower.get("available"):
                        source["low_res"] = True
                        stream = lower
                return bvid, stream
            except media.Cancelled:
                raise
            except Exception as exc:
                self.notes.append(f"播放地址失败 {bvid}: {str(exc)[:100]}")
                return bvid, None
        with execution.ThreadPoolExecutor(max_workers=4) as pool:
            for bvid, stream in pool.map(run, list(sources.items())):
                streams[bvid] = stream
                if stream and stream.get("available") and stream.get("duration"):
                    sources[bvid]["duration"] = min(float(sources[bvid].get("duration") or 1e9), stream["duration"])
        return streams

    def _fetch_job(self, job_id, sid, choice, streams):
        def check():
            self.check()
            if self.abandon.is_set():   # matching is over: stop instead of eating bandwidth
                raise TimeoutError("downloads stopped")
        shot = self.shots_by_id_all[sid]
        self.check()
        began = time.monotonic()
        try:
            path = self.fetch(shot, choice, streams, check)
            with self.lock:
                self.fetch_seconds.append((choice["tile"]["source"]["provider"], round(time.monotonic() - began, 2)))
            clean = self.analyse(shot, choice, path)
            with self.lock:
                self.clean[job_id] = clean
            if clean.get("central"):
                choice["tile"]["source"]["dirty"] = True   # mid-picture watermark: avoid this source
            self.read_text(job_id, shot, choice, path)
            return job_id, sid, path, None
        except media.Cancelled:
            raise
        except Exception as exc:
            return job_id, sid, None, f"{type(exc).__name__}: {str(exc)[:160]}"

    def download(self, jobs, streams, qa=False, deadline=None):
        """jobs: [(job_id, sid, choice)]. Downloads in parallel; QA batches of six as they land.

        Returns {job_id: path}, {job_id: error}, {job_id: qa_reason} and the ids still downloading.
        Results of jobs started by earlier calls are included. Jobs still running at the deadline keep
        going into the next call instead of being thrown away; stop_downloads() ends them.
        """
        from concurrent.futures import FIRST_COMPLETED, wait
        if self.pool is None:
            self.pool = execution.ThreadPoolExecutor(max_workers=16)
        for job in jobs:
            self.inflight[self.pool.submit(self._fetch_job, *job, streams)] = job
        results, failures, bad = {}, {}, {}
        shots_by_id = self.shots_by_id_all
        qa_pool = execution.ThreadPoolExecutor(max_workers=4)
        try:
            pending, reviews, batch = set(self.inflight), [], []
            while pending:
                timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
                done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                if not done:
                    break
                for future in done:
                    self.inflight.pop(future, None)
                    job_id, sid, path, error = future.result()
                    if path:
                        results[job_id] = path
                        batch.append((job_id, sid, path))
                    else:
                        failures[job_id] = error
                if qa and (len(batch) >= 6 or (not pending and batch)):
                    reviews.append(qa_pool.submit(self.qa_batch, batch, shots_by_id))
                    batch = []
            if qa and batch:
                reviews.append(qa_pool.submit(self.qa_batch, batch, shots_by_id))
            for review in reviews:
                bad.update(review.result())
        finally:
            qa_pool.shutdown(wait=True)
        running = {job[0] for job in self.inflight.values()}
        if running:
            self.notes.append(f"{len(running)} 个片段本轮未下载完，继续下载")
        return results, failures, bad, running

    def drop_queued(self) -> list:
        """Cancel downloads that have not started (stale candidates from earlier rounds), so a later lane
        gets the download slots; returns their jobs. Downloads already running carry on."""
        dropped = []
        for future, job in list(self.inflight.items()):
            if future.cancel():
                self.inflight.pop(future, None)
                dropped.append(job)
        if dropped:
            self.notes.append(f"{len(dropped)} 个排队中的候选片段已取消，让补位片段先下载")
        return dropped

    def stop_downloads(self) -> dict:
        """Abandon whatever is still downloading; returns {job_id: reason}."""
        self.abandon.set()
        left = {job_id: "下载超时，已放弃该片段" for job_id, _, _ in self.inflight.values()}
        self.inflight.clear()
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None
        if left:
            self.notes.append(f"{len(left)} 个片段到匹配结束仍未下载完，已放弃")
        return left

    # ------------------------------------------------------------- clean-up
    def out_aspect(self) -> float:
        return 9 / 16 if self.project.get("aspect") == "9:16" else 16 / 9

    def sprite_frames(self, source, start) -> list:
        """Preview frames of a Bilibili source: sheets already on disk plus the one around the window."""
        previews = self.bili.previews(source["id"], float(source.get("duration") or 0))
        if not previews:
            return []
        meta, root = previews[0]["meta"], previews[0]["root"]
        # Sheets fetched for the thumbnails are enough; extra 1-2 MB sheets would compete with clip downloads.
        sheets = sorted({p["sheet"] for p in previews if (Path(root) / f"sprite-{p['sheet']}.jpg").exists()})[:4]
        if not sheets:
            sheets = [min((abs(p["t"] - start), p["sheet"]) for p in previews)[1]]
        frames = []
        for sheet in sheets:
            key = (source["id"], sheet)
            with self.lock:
                cached = self.sheets.get(key)
                if cached is not None:
                    self.sheets.move_to_end(key)
            if cached is None:
                cached = []
                with Image.open(self.bili.sprite(root, meta, sheet)) as im:
                    im = im.convert("L")      # every use is greyscale; a third of the memory of RGB
                    for p in previews:
                        if p["sheet"] != sheet:
                            continue
                        x, y = (p["cell"] % meta["x"]) * meta["w"], (p["cell"] // meta["x"]) * meta["h"]
                        if y + meta["h"] <= im.height:
                            cached.append((p["t"], im.crop((x, y, x + meta["w"], y + meta["h"]))))
                with self.lock:
                    self.sheets[key] = cached
                    while len(self.sheets) > SPRITE_SHEETS_KEPT:
                        self.sheets.popitem(last=False)
            frames += cached
        return frames

    def analyse(self, shot, choice, path) -> dict:
        """Find logos/watermarks/subtitles in a downloaded Bilibili window (stock footage is clean)."""
        source = choice["tile"]["source"]
        if source["provider"] != "bilibili" or not express_clean.available():
            return {}
        try:
            stream = next(s for s in media.probe(path)["streams"] if s.get("codec_type") == "video")
            size = (int(stream["width"]), int(stream["height"]))
            sprites = self.sprite_frames(source, choice["start"])
            if sprites and size[1] > size[0] and sprites[0][1].width > sprites[0][1].height:
                # Portrait videos sit pillarboxed inside landscape preview cells: keep only the picture.
                cw, ch = sprites[0][1].size
                inner = max(2, round(ch * size[0] / size[1]))
                left = (cw - inner) // 2
                sprites = [(t, im.crop((left, 0, left + inner, ch))) for t, im in sprites]
            analysis = (express_clean.WIDTH, max(2, round(express_clean.WIDTH * size[1] / size[0])))
            key = (source["id"], analysis)
            with self.lock:
                profile = self.profiles.get(key)
            if profile is None:
                profile = express_clean.profile([im for _, im in sprites], analysis)
                with self.lock:
                    self.profiles[key] = profile
            broadcast = bool(BROADCAST.search(" ".join(str(source.get(k, "")) for k in ("title", "tags", "author"))))
            return express_clean.analyse(path, float(shot["duration"]), size, self.out_aspect(), profile,
                                         sprites, choice["start"], channel_logo=broadcast)
        except media.Cancelled:
            raise
        except Exception as exc:
            self.notes.append(f"去水印分析失败 {source['id']}: {type(exc).__name__}: {str(exc)[:80]}")
            return {}

    def wants_backup(self, shot, primary, videos: int) -> bool:
        """A second candidate per shot in the same round saves a round when the first fails the check, at the
        cost of one more download, encode, analysis and check. With one video alone, time matters most: always.
        With several videos at once the computer is the limit: only where the first pick is likely to fail
        (named people, a weak match, a flagged source)."""
        if primary is None or videos <= 1 or shot.get("persons"):
            return True
        source = primary["tile"]["source"]
        return (primary.get("lane") not in ("pick", "near") or source.get("dirty") or source.get("captioned")
                or (source["provider"], source["id"]) in self.host_sources)

    def read_text(self, job_id, shot, choice, path):
        """Words burned into a downloaded window (local OCR, free): contact details reject it (and every other
        window of that video); captions and titles are cropped away or blurred; text across the middle or
        scrolling text makes it dirty, so a cleaner clip is preferred."""
        source = choice["tile"]["source"]
        if source["provider"] != "bilibili" or not screen_text.installed():
            return
        try:
            with throttle.slot("frames", self.check):
                frames = screen_text.frames_of(path, float(shot["duration"]))
                lines = [screen_text.read(frame) for frame in frames]
        except media.Cancelled:
            raise
        except Exception as exc:   # reading is a safeguard: the visual check still runs
            self.notes.append(f"文字识别失败 {source['id']}: {type(exc).__name__}")
            return
        clean = self.clean.get(job_id) or {}
        found = screen_text.classify(lines, clean.get("content") or (0.0, 0.0, 1.0, 1.0))
        kinds = {m["kind"] for m in found["marks"]}
        with self.lock:
            self.text_found["caption"] += "caption" in kinds
            self.text_found["center"] += "center" in kinds and not found["contact"]
        if found["contact"]:
            with self.lock:
                self.text_found["contact"] += 1
                if not source.get("contact"):
                    self.text_found["sources_contact"] += 1
            source["contact"] = True
            raise ValueError("画面里有联系方式或广告文字：" + "、".join(found["contact"])[:40])
        if found["moving"]:
            raise ValueError("画面里有滚动字幕，换别的画面")
        if found["dense"]:
            # Text across the picture: a text page or an app screen is unusable, a stock chart, a logo or a
            # sign is footage. Nothing is blurred; the visual check judges it and cleaner clips come first.
            with self.lock:
                self.soft.setdefault(job_id, set()).add("wm")
        if not found["marks"]:
            return
        if clean.get("crop"):
            clean = express_clean.recrop(clean, found["marks"], self.out_aspect())
            with self.lock:
                self.clean[job_id] = clean
        elif kinds - {"logo"}:
            with self.lock:
                self.soft.setdefault(job_id, set()).add("wm")

    def flags(self, job_id) -> set:
        clean = self.clean.get(job_id) or {}
        flags = set(self.soft.get(job_id, ()))
        if (clean.get("motion") or {}).get("still"):
            flags.add("still")
        if (clean.get("motion") or {}).get("dark"):
            flags.add("dark")
        if clean.get("layout_jump"):
            flags.add("jump")
        if express_clean.dirty(clean):
            flags.add("wm")
        return flags

    # ---------------------------------------------------------------------- QA
    def qa_frames(self, job_id, path, shot):
        duration = float(shot["duration"])
        prefix = express_clean.video_filter(self.clean.get(job_id))
        tag = hashlib.sha1(prefix.encode()).hexdigest()[:8] if prefix else "raw"
        fractions = (0.15, 0.5, 0.85)
        targets = [path.parent / f"{path.stem}-qa{int(fraction*100)}-{tag}.jpg" for fraction in fractions]
        missing = [(fraction, target) for fraction, target in zip(fractions, targets) if not target.exists()]
        if missing:
            # One FFmpeg run for all three stills: each is its own fast-seeking input and output.
            args = [paths.binary("ffmpeg"), "-v", "error", "-y"]
            for fraction, _ in missing:
                args += ["-threads", "1", "-ss", f"{duration*fraction:.2f}", "-i", str(path)]
            picture = ((prefix + ",") if prefix else "") + (f"scale={QA_W}:{QA_H}:force_original_aspect_ratio=decrease,"
                                                            f"pad={QA_W}:{QA_H}:(ow-iw)/2:(oh-ih)/2")
            for index, (_, target) in enumerate(missing):
                args += ["-map", f"{index}:v:0", "-frames:v", "1", "-vf", picture, str(target)]
            with throttle.slot("frames", self.check):
                media.run(args, timeout=30)
        frames = []
        for target in targets:
            with Image.open(target) as im:
                frames.append(im.convert("RGB"))
        return frames

    def qa_batch(self, batch, shots_by_id):
        """batch: [(job_id, sid, path)] -> {job_id: reason}; a QA failure leaves the batch unchecked."""
        try:
            return self._qa_batch(batch, shots_by_id)
        except media.Cancelled:
            raise
        except Exception as exc:
            if type(exc).__name__ in ("BudgetExceeded", "ModelUnavailable", "PricingUnavailable"):
                raise
            if "HTTP 400" in str(exc):
                # Usually the content check refusing one picture: check the halves so the others still get QA.
                # A single clip it still refuses (bodies, funerals, violence) is not used.
                if len(batch) > 1:
                    half = len(batch) // 2
                    return self.qa_batch(batch[:half], shots_by_id) | self.qa_batch(batch[half:], shots_by_id)
                return {batch[0][0]: "抽检被内容审核拒绝"}
            with self.lock:
                self.qa_unchecked.update(job_id for job_id, _, _ in batch)
            self.notes.append(f"抽检未完成（{type(exc).__name__}）：{str(exc)[:80]}")
            return {}

    def _qa_batch(self, batch, shots_by_id):
        rows = [(job_id, sid, self.qa_frames(job_id, path, shots_by_id[sid])) for job_id, sid, path in batch]
        bad = {job_id: "黑屏或纯色画面" for job_id, sid, frames in rows if sum(_dark(f) for f in frames) >= 2}
        rows = [(job_id, sid, frames) for job_id, sid, frames in rows if job_id not in bad]
        if not rows:
            return bad
        label_w = 70
        sheet = Image.new("RGB", (label_w + 3 * QA_W, len(rows) * QA_H), (0, 0, 0))
        draw = ImageDraw.Draw(sheet)
        font = _font(30)
        lines, ids = [], {}
        for r, (job_id, sid, frames) in enumerate(rows):
            label = f"S{r+1}"
            ids[label] = job_id
            draw.rectangle((0, r * QA_H, label_w - 4, r * QA_H + 44), fill=(255, 214, 0))
            draw.text((6, r * QA_H + 4), label, font=font, fill=(0, 0, 0))
            for c, frame in enumerate(frames):
                sheet.paste(frame.resize((QA_W, QA_H)), (label_w + c * QA_W, r * QA_H))
            shot = shots_by_id[sid]
            names = "、".join(p["name"] for p in shot.get("persons") or [])
            role = ""
            if names:
                choice = self.job_choice.get(job_id) or {}
                source = choice.get("tile", {}).get("source", {})
                lane = choice.get("base_lane", choice.get("lane"))
                if self.ancient_shot(shot):
                    role = (f"｜古代人物：{names}（没有真实影像；画像、雕像、影视剧或动画中表现此人的形象都合格，"
                            "或与口播相关的同时代场景；现代场景、其他具名人物、主持人讲解不合格）")
                elif self.names_in_source(shot, source) and lane in ("pick", "person", "person_near"):
                    role = f"｜要求人物：{names}（来源标题含此姓名，画面主体应是此人，不能是演员或他人）"
                elif self.unfilmed_shot(shot):
                    role = (f"｜背景画面：找不到{names}本人的真实影像，此镜用与口播主题相关的背景画面即可，不要求出现此人，"
                            "不要因没有此人而判不合格；但不能出现冒充他的人物")
                else:   # people-free context footage for this line
                    role = (f"｜背景画面：口播提到{names}；画面可以是不含人物的相关场景、文件、建筑，也可以是此人本人的真实影像，"
                            "不要因没有此人而判不合格；但不能是演员扮演，也不应以其他可辨认的具名人物为主体")
            lines.append(f"{label} 口播：{shot['narration']}｜画面建议：{shot.get('visual', '')}{role}")
        domain = self.plan.get("domain") or ""
        head = [f"整篇领域：{domain}"] if domain else []
        if self.ancient:
            head.append(f"本片是古代历史题材（{'、'.join(sorted(self.ancient))}）：古装影视剧、纪录片情景再现、历史动画和插画都是合格的画面，"
                        "不要因为是演员扮演、不是真实史料判不合格；只有现代场景、明显是别的朝代或外国、主持人对镜头讲解才不合格")
        content = [{"type": "text", "text": "\n".join(head + lines)}, {"type": "image_url", "image_url": {"url": _jpeg(sheet)}}]
        result, _ = self.chat(QA_SYSTEM, content, max_tokens=900, task="express_qa", attempts=2)
        with self.lock:
            self.calls["qa"] += 1
        result = result if isinstance(result, dict) else {}
        found = result.get("bad") if isinstance(result.get("bad"), dict) else {}
        bad.update({ids[k]: str(v)[:60] for k, v in found.items() if k in ids})
        marks = result.get("wm") if isinstance(result.get("wm"), dict) else {}
        with self.lock:
            for label, where in marks.items():
                if label in ids:
                    self.residual_mark(ids[label], where)
                    # Nothing is blurred any more: an "@name" or 花字 across the middle would stay in the film.
                    if label in ids and ids[label] not in bad and "中间" in str(where):
                        bad[ids[label]] = "画面中间有水印或花字"
        return bad

    VIEW_BOXES = {"左上": [0, 0, 0.36, 0.16], "右上": [0.64, 0, 1, 0.16], "左下": [0, 0.84, 0.36, 1],
                  "右下": [0.64, 0.84, 1, 1], "底部": [0.12, 0.8, 0.88, 1], "顶部": [0.12, 0, 0.88, 0.14],
                  "左侧": [0, 0.2, 0.2, 0.8], "右侧": [0.8, 0.2, 1, 0.8]}

    def residual_mark(self, job_id, where):
        """The visual check still sees marks: crop them too, or mark the clip dirty when that is not certain."""
        where = "、".join(map(str, where)) if isinstance(where, (list, tuple)) else str(where or "")
        if not where.strip() or re.fullmatch(r"\s*(无|没有|否|none|null|None|\[\])\s*", where):
            return   # "nothing found" written as a value
        hits = [(k, b) for k, b in self.VIEW_BOXES.items() if k in where]
        # Mid-picture marks cannot be cropped; side marks have no known extent, so the crop may not clear them.
        if not hits or any(k in where for k in ("中间", "左侧", "右侧")):
            self.soft.setdefault(job_id, set()).add("wm")
        if "中间" in where:   # a mid-picture "@name" is on the whole video: other windows of it are no better
            source = ((self.job_choice.get(job_id) or {}).get("tile") or {}).get("source")
            if source is not None:
                source["dirty"] = True
        if not hits:
            return
        clean = self.clean.get(job_id)
        if not clean:
            if any(k == "底部" for k, _ in hits):
                self.subtitled.add(job_id)   # stock/unanalysed clip: old bottom crop
            if any(k != "底部" for k, _ in hits):
                self.soft.setdefault(job_id, set()).add("wm")
            return
        extra = [{"kind": "subtitle" if k == "底部" else "logo", "box": express_clean.from_view(b, clean["crop"])}
                 for k, b in hits]
        clean = express_clean.recrop(clean, extra, self.out_aspect())
        self.clean[job_id] = clean
        if express_clean.dirty(clean):
            self.soft.setdefault(job_id, set()).add("wm")

    # ------------------------------------------------------------------- save
    def save(self, final, files):
        try:
            store.mark_footage_used(self.pid, [f"{c['tile']['source']['provider']}:{c['tile']['source']['id']}"
                                               for c in final.values() if c["tile"]["source"].get("provider") != "card"])
        except Exception as exc:
            self.notes.append(f"记录已用素材失败: {type(exc).__name__}")
        try:   # library videos used in a film rank a little lower next time (less repetition across videos)
            from . import footage_index
            footage_index.mark_used({f"{c['tile']['source']['provider']}:{c['tile']['source']['id']}"
                                     for c in final.values() if c['tile']['source'].get('indexed')})
        except Exception:
            pass
        def prepare(item):
            sid, choice = item
            source = choice["tile"]["source"]
            path = files[sid]
            asset = media.inspect_asset(self.pid, path, source.get("title") or source["id"],
                                        source_url=source.get("url", ""),
                                        license=source.get("license", "公开平台视频，使用前请自行确认授权"),
                                        license_url=source.get("license_url", ""),
                                        author=source.get("author", ""),
                                        description=(source.get("tags") or "")[:300],
                                        provider=source["provider"])
            asset.update(source_clip_start=choice["start"], download_mode="express",
                         candidate_id=f"{source['provider']}:{source['id']}")
            clean = self.clean.get(choice.get("job")) or {}
            if clean.get("crop") and asset.get("kind") == "video" and asset.get("thumbnail"):
                # The workbench shows what the export shows: thumbnail after the same crop and blur.
                root = paths.project_dir(self.pid)
                target = root / "thumbs" / f"{asset['id']}-clean.jpg"
                cleaned = express_clean.video_filter({"crop": clean["crop"], "residual": clean["residual"]})
                try:
                    media.run([paths.binary("ffmpeg"), "-v", "error", "-y", "-ss",
                               f"{min(1.0, float(asset.get('duration') or 2) / 2):.2f}", "-i", str(path), "-frames:v", "1",
                               "-vf", (cleaned + "," if cleaned else "") + "scale=720:-2", str(target)], timeout=45)
                    asset.update(raw_thumbnail=asset["thumbnail"], thumbnail=target.relative_to(root).as_posix())
                except Exception as exc:   # the raw thumbnail still works
                    self.notes.append(f"去水印缩略图生成失败：{type(exc).__name__}")
            return sid, asset
        # Probing and thumbnails are separate ffmpeg runs per clip: a long script has hundreds.
        with execution.ThreadPoolExecutor(max_workers=4) as pool:
            assets = dict(pool.map(prepare, final.items()))
        def apply(p):
            for sid, asset in assets.items():
                p["assets"].append(asset)
                shot = next(s for s in p["shots"] if s["id"] == sid)
                choice = final[sid]
                source = choice["tile"]["source"]
                person = [x["name"] for x in shot.get("persons") or []]
                if person and choice.get("base_lane", choice["lane"]) in ("pick", "person", "person_near") and \
                        self.names_in_source(shot, source) and not str(choice.get("qa", "")).startswith("rejected"):
                    identity = {"status": "supported", "basis": "source_title_and_frame_subject",
                                "source_text": source.get("title", ""), "names": person}
                elif person:
                    identity = {"status": "not_shown", "basis": "people_free_context_footage", "names": person}
                else:
                    identity = None
                verification = {
                    "verdict": "match", "scene_alignment": "match", "selection_blocked": False,
                    "needs_source_check": False,
                    "reason": {"pick": "快速通道：缩略图批量选片通过", "person": "快速通道：使用同一人物的其他真实影像",
                               "person_near": "快速通道：使用同一人物影像来源中的相邻片段",
                               "ctx": "快速通道：人物镜头无可靠人物素材，改用不含人物的相关画面",
                               "topic": "快速通道：本组素材不足，使用全片通用相关画面补位",
                               "neighbor": "快速通道：使用相邻镜头同组的相关画面补位",
                               "near": "快速通道：使用已选画面相邻片段补位",
                               "repeat": "快速通道：素材不足，复用已通过抽检的相关画面",
                               "reuse": "快速通道：没有新的可用画面，借用附近镜头已通过检查的画面",
                               "stock": "快速通道：用正版图库空镜补位",
                               "image": "快速通道：没有合适的视频，用相关图片加推拉动态补位",
                               "card": "快速通道：没有可用实拍素材，使用模糊背景卡片"}.get(choice["lane"], "快速通道"),
                    "visual_quality": {"representation": "live_action" if choice["lane"] != "card" else "illustration"},
                    "identity_evidence": identity,
                    "express": {"version": VERSION, "lane": choice["lane"], "rank": choice["rank"],
                                "provider": source["provider"], "source_id": source["id"],
                                "source_url": source.get("url", ""), "source_title": source.get("title", ""),
                                "tile_time": round(float(choice["tile"]["t"]), 2),
                                "window": [choice["start"], round(choice["start"] + float(shot["duration"]), 2)],
                                "qa": choice.get("qa", "pass"), "identity": (identity or {}).get("status", ""),
                                "flags": sorted(self.flags(choice.get("job"))) if choice.get("job") else []},
                }
                clean = self.clean.get(choice.get("job")) or {}
                framed = asset
                if clean.get("crop"):
                    shot["frame_crop"] = {"crop": clean["crop"], "zoom": clean["zoom"],
                                          "residual": [r for r in clean["residual"] if r["kind"] != "subtitle"],
                                          "overlays": [o["kind"] for o in clean["overlays"]],
                                          # Where each mark was found (source-frame fractions), shown in the workbench.
                                          "boxes": [{"kind": o["kind"], "box": o["box"]}
                                                    for o in clean["overlays"] + clean.get("central", [])],
                                          "content": clean.get("content")}
                    framed = asset | {"width": asset.get("width", 0) * clean["crop"][2],
                                      "height": asset.get("height", 0) * clean["crop"][3]}
                    shot["crop_bottom"] = 0
                else:
                    shot.pop("frame_crop", None)
                    shot["crop_bottom"] = 0.16 if choice.get("job") in self.subtitled else 0
                if shot.get("selected_asset") != asset["id"]:
                    # A source note written for the previous clip says nothing about this one.
                    shot.update(source_review_note="", source_review_url="")
                shot["crop_asset"] = asset["id"]   # the crop was measured on this file only
                shot.update(selected_asset=asset["id"], clip_start=0, verification=verification, approved=True,
                            search_note=verification["reason"],
                            fit=curation.preferred_fit(shot, framed, p.get("aspect", "16:9")))
        store.mutate(self.pid, apply, internal=True)

    # -------------------------------------------------------------------- run
    def run(self):
        def stamp(name, began):
            self.timing[name] = round(time.monotonic() - began, 2)
        shots_by_id = {s["id"]: s for s in self.shots}
        groups = self.groups()
        began = time.monotonic()
        self.progress(3, f"检索素材：{len(groups)} 组")
        results = self.search(groups)
        # 新闻模式: reports about the same event, transcribed while the searched footage is previewed and picked.
        news, news_thread = {}, None
        event = [q for q in (self.plan.get("event") or []) if q][:2]
        if event and self.bili and config.read().get("news_mode") == "1":
            searched = [row for (kind, _), rows in results.items() if kind == "bili" for row in rows]
            def news_job():
                try:
                    news.update(news_pool.build(self.bili, event, self.shots, self.check, self.notes,
                                                extra_queries=[q for g in groups for q in g.get("zh", [])],
                                                searched=searched))
                except Exception as exc:  # noqa: BLE001 — an extra; the searched footage still works
                    self.notes.append(f"新闻模式未完成：{type(exc).__name__}")
            news_thread = threading.Thread(target=news_job, name="news-pool", daemon=True)
            news_thread.start()
        chosen = self.candidates(groups, results)
        def carries(row, name):
            return name in " ".join(str(row.get(k, "")) for k in ("title", "tags", "description"))
        self.unfilmed = {name for name in self.names
                         if not any(carries(row, name) for rows in chosen.values() for row in rows)}
        stamp("search", began)
        began = time.monotonic()
        self.progress(15, f"读取 {sum(len(v) for v in chosen.values())} 个候选的预览")
        tiles = self.all_tiles(chosen, {g["id"] for g in groups if g.get("person")})
        self.presenter_sources = self.screen_presenters(chosen, tiles)
        try:   # the library grows with every video made (title text and frame times; described in bulk later)
            from . import footage_index
            footage_index.add_seen([row for rows in chosen.values() for row in rows],
                                   [t for rows in tiles.values() for t in rows], topic=self.plan.get("domain") or "")
        except Exception as exc:
            self.notes.append(f"写入本地素材库失败: {type(exc).__name__}")
        stamp("previews", began)
        began = time.monotonic()
        self.progress(30, "批量判断画面")
        picks, ctx_tiles, topic_tiles = {}, {}, []
        def run_pick(group):
            group_tiles = []
            rows = chosen.get(group["id"], [])
            rows = [r for r in rows if not r.get("contact")]   # 加微信 / phone numbers on the picture: never
            clean_rows = [r for r in rows if not r.get("dirty") and tiles.get((r["provider"], r["id"]))]
            if len(clean_rows) >= 2:
                rows = clean_rows   # watermarked-across-the-picture sources only when little else exists
            filmed = [r for r in rows if (r["provider"], r["id"]) not in self.presenter_sources]
            if len(filmed) >= 2:
                rows = filmed       # presenter-led commentary videos only when little else exists
            per_source = [tiles.get((r["provider"], r["id"]), []) for r in rows]
            # pick() looks at 60 tiles at most: give every source a share spread over its whole length
            # (an indexed video's matching frames come first in its list and are kept).
            share = max(2, (3 * SHEET_COLS * SHEET_ROWS) // max(1, len(per_source)))
            per_source = [x if len(x) <= share else x[:share // 2] + [
                x[share // 2 + round(i * (len(x) - share // 2 - 1) / max(1, share - share // 2 - 1))]
                for i in range(share - share // 2)] for x in per_source]
            # Interleave sources so a sheet never shows only one video.
            for i in range(max([len(x) for x in per_source] or [0])):
                group_tiles += [x[i] for x in per_source if i < len(x)]
            try:
                return group, self.pick(group, group_tiles, shots_by_id)
            except media.Cancelled:
                raise
            except Exception as exc:
                if type(exc).__name__ in ("BudgetExceeded", "ModelUnavailable", "PricingUnavailable"):
                    raise
                if "设置中填写" in str(exc):
                    raise
                self.notes.append(f"选片失败 {group['id']}: {str(exc)[:120]}")
                return group, ({}, {})
        with execution.ThreadPoolExecutor(max_workers=6) as pool:
            for group, (group_picks, extra) in pool.map(run_pick, groups):
                if group.get("topic"):
                    topic_tiles = group_picks.get("TOPIC", [])
                    continue
                picks.update(group_picks)
                for sid in group["shots"]:
                    ctx_tiles[sid] = extra.get("ctx", [])
        stamp("pick", began)
        began = time.monotonic()
        chosen_rows = {}
        for group_tiles in list(picks.values()) + [topic_tiles] + list(ctx_tiles.values()):
            for tile in group_tiles or []:
                row = tile["source"]
                if row.get("provider") == "bilibili" and row.get("_screen"):
                    chosen_rows[row["id"]] = row
        def screen(row):
            self.check()
            self.screen_source(row, row.get("_screen") or [])
        with execution.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(screen, chosen_rows.values()))
        for rows in chosen.values():
            for row in rows:
                row.pop("_screen", None)
        stamp("text", began)
        for rows in tiles.values():      # pictures are only needed for the contact sheets
            for tile in rows:
                tile["image"] = None
        if news_thread is not None:
            news_began = time.monotonic()
            self.progress(40, "新闻模式：对照同一事件的新闻报道")
            news_thread.join(timeout=NEWS_WAIT)
            self.timing["news_wait"] = round(time.monotonic() - news_began, 2)
            if news:
                self.timing["news"] = news.get("seconds")
                self.notes.append(f"新闻模式：{news.get('sources', 0)} 条同一事件的报道，{len(news.get('matched') or {})} 镜对上了报道原话")
        # The words are often said by the presenter on camera and the report cuts to the scene after: offer
        # the moment they are said and 5 and 10 s later (QA rejects the presenter, the next one is tried),
        # or, when the preview frames already moved the window onto the scene, that moment and a little later.
        later_times = NEWS_LATER_ONSITE if news.get("onsite") else NEWS_LATER
        news_tiles = {sid: [({"source": row, "t": float(t) + later, "hash": None}, "news") for row, t, _ in found
                            for later in later_times if float(t) + later < float(row.get("duration") or 1e9) - 2]
                      for sid, found in (news.get("matched") or {}).items()}
        # Build ranked options per shot: the matching news report first, own picks, then context/topic/neighbour fills.
        group_of = {sid: g for g in groups for sid in g["shots"]}
        options = {}
        for shot in self.shots:
            sid = shot["id"]
            index = self.position[sid]
            person = bool(shot.get("persons")) and not self.unfilmed_shot(shot)
            own = news_tiles.get(sid, []) + [(t, "pick") for t in picks.get(sid, [])]
            fill = []
            if person:
                # Other footage of the same person (picked for other lines) before any people-free filler.
                names = {p["name"] for p in shot["persons"]}
                for g in groups:
                    if g.get("person") in names:
                        for other in g["shots"]:
                            if other != sid:
                                fill += [(t, "person") for t in picks.get(other, [])
                                         if self.names_in_source(shot, t["source"])]
                fill += [(t, "ctx") for t in ctx_tiles.get(sid, [])]
            fill += [(t, "topic") for t in topic_tiles]
            if not person:
                neighbours = []
                for other in self.timeline[max(0, index - 3):index + 4]:
                    if other["id"] != sid and group_of.get(other["id"], {}).get("id") == group_of.get(sid, {}).get("id"):
                        neighbours += picks.get(other["id"], [])
                for other in self.timeline[max(0, index - 2):index + 3]:
                    if other["id"] != sid and not other.get("persons"):
                        neighbours += picks.get(other["id"], [])
                fill += [(t, "neighbor") for t in neighbours]
            if index == 0:
                # The opening line hooks the viewer and usually sums up what the next lines tell: when it has no
                # pick of its own, the next lines' picks come before any filler.
                own += [(t, "pick") for other in self.timeline[1:4] for t in picks.get(other["id"], [])]
            seen, merged = set(), []
            for tile, lane in own + fill:
                k = (tile["source"]["provider"], tile["source"]["id"], round(tile["t"], 1))
                if k not in seen:
                    seen.add(k)
                    merged.append((tile, lane))
            options[sid] = merged
        began = time.monotonic()
        self.progress(42, "确认清晰度与播放地址")
        referenced = {}
        for rows in options.values():
            for tile, _ in rows:
                if tile["source"]["provider"] == "bilibili":
                    referenced[tile["source"]["id"]] = tile["source"]
        streams = self.resolve(referenced)
        refused = sum(1 for bvid in referenced if not streams.get(bvid))
        if len(referenced) >= 8 and refused >= 0.8 * len(referenced):
            raise BiliUnavailable(f"B站 {len(referenced)} 个视频里有 {refused} 个拿不到播放地址（多半是被限流）")
        landscape = self.project.get("aspect", "16:9") != "9:16"
        def usable(tile):
            source = tile["source"]
            if source["provider"] != "bilibili":
                return self.oriented(source)
            stream = streams.get(source["id"]) or {}
            w, h = stream.get("width") or 0, stream.get("height") or 0
            # A portrait re-upload of landscape footage is a small picture between caption bars.
            return bool(stream.get("available")) and (not (w and h) or (w >= h) == landscape)
        low = sum(1 for st in streams.values() if st and not st.get("available"))
        soft = sum(1 for bvid in streams if referenced[bvid].get("low_res"))
        if soft:
            self.notes.append(f"{soft} 个B站来源只有 {LOW_HEIGHT}p，排在高清画面之后使用" +
                              ("（登录B站可获得 720p/1080p）" if not self.bili.logged_in else ""))
        if low:
            self.notes.append(f"{low} 个B站来源最高清晰度低于 {self.min_height}p，已跳过" +
                              ("（登录B站可获得 720p/1080p）" if not self.bili.logged_in else ""))
        # Spare windows next to tiles the model accepted (same scene is likely), as the final fill lane.
        accepted = {}
        for tile_list in list(picks.values()) + [topic_tiles]:
            for tile in tile_list:
                if usable(tile):
                    accepted[(tile["source"]["provider"], tile["source"]["id"], round(tile["t"], 1))] = tile
        spare_all = []
        for tile in accepted.values():
            length = float(tile["source"].get("duration") or 0)
            for delta in (-12, 12, -24, 24):
                t = float(tile["t"]) + delta
                if 1 <= t <= length - 1:
                    spare_all.append(({"source": tile["source"], "t": t, "near": tile["t"]}, "near"))
        for shot in self.shots:
            sid = shot["id"]
            rows = [(t, lane) for t, lane in options[sid] if usable(t)]
            if not shot.get("persons") or self.unfilmed_shot(shot):
                rows += spare_all
            else:   # more of a documentary about this person, right next to footage the model accepted
                rows += [(t, "person_near") for t, _ in spare_all if self.names_in_source(shot, t["source"])]
            if not DEATH_LINE.search(shot["narration"]):   # a funeral video only for the line about the death
                rows = [(t, lane) for t, lane in rows if not FUNERAL.search(t["source"].get("title", ""))]
            options[sid] = rows
        stamp("streams", began)
        reserved, usage = self.existing_windows(), {}
        tried = {s["id"]: set() for s in self.shots}
        final, files, rejected, failures, qa_bad = {}, {}, {}, {}, {}
        provisional = {}   # sid -> (choice, path, flags): usable but watermarked or a still picture
        def key_of(choice):
            tile = choice["tile"]
            return (tile["source"]["provider"], tile["source"]["id"]), round(float(tile["t"]), 1)
        def release(choice):
            k = key_of(choice)[0]
            reserved[k] = [w for w in reserved.get(k, []) if abs(w[0] - choice["start"]) > 1e-6]
            usage[k] = max(0, usage.get(k, 1) - 1)
        pending = list(self.shots)
        checking = self.cfg.get("express_final_check", "1") != "0"
        carried = []   # (job_id, sid, choice) still downloading from an earlier round
        def settle(jobs, got, bad, running, last):
            """Take the landed jobs; returns those still downloading (their windows stay reserved)."""
            # Clean, moving clips first: in the last round a flagged clip must not beat a clean one of the same shot.
            jobs = sorted(jobs, key=lambda job: bool(got.get(job[0])) and job[0] not in bad
                          and bool(self.flags(job[0])))
            for job_id, sid, choice in jobs:
                if job_id in running:
                    continue
                path = got.get(job_id)
                if not path:
                    release(choice)
                    continue
                if job_id not in bad and "dark" in self.flags(job_id):
                    bad[job_id] = "黑屏或淡入黑场"     # found locally, whatever the sampled QA frames showed
                if job_id not in bad and "jump" in self.flags(job_id):
                    bad[job_id] = "画面版式中途跳变（片头或画中画切换），看起来像闪了一下"
                if job_id in bad:
                    rejected.setdefault(sid, []).append((choice | {"job": job_id}, path, bad[job_id]))   # region stays reserved
                    continue
                if sid in final:
                    release(choice)   # the other candidate already won; free its window
                    continue
                passed = choice | {"qa": "pass" if checking and job_id not in self.qa_unchecked else "unchecked",
                                   "job": job_id}
                flags = self.flags(job_id)
                if "still" in flags:
                    # A still picture (illustration, photo slideshow) is a last resort: moving footage, also
                    # reused from a neighbour, comes first; keepable() may still take it before a title card.
                    rejected.setdefault(sid, []).append((passed, path, "近似静态画面（图片）"))
                    continue
                held = provisional.get(sid)
                if not flags or last:
                    if flags and held and len(held[2]) <= len(flags):
                        release(choice)   # last round: keep the earlier, no worse provisional clip
                        continue
                    if held:
                        release(held[0])
                        provisional.pop(sid)
                    final[sid] = passed
                    files[sid] = path
                    self.place(sid, choice, key_of(choice)[0])
                elif held is None or len(flags) < len(held[2]) or ("wm" in held[2] and "wm" not in flags):
                    if held:
                        release(held[0])
                    provisional[sid] = (passed, path, flags)   # look for a clean, moving clip next round
                else:
                    release(choice)
            return [job for job in jobs if job[0] in running]
        for round_no in range(3):
            if not pending:
                break
            began = time.monotonic()
            self.progress(50 + round_no * 12, f"第 {round_no+1} 轮：下载并抽检 {len(pending)} 镜")
            # A shot whose clip is still downloading waits for it (a slow link gains nothing from more
            # parallel downloads), except in the last round.
            busy = {sid for _, sid, _ in carried}
            fresh = pending if round_no == 2 else [s for s in pending if s["id"] not in busy]
            primary = self.allocate(options, reserved, usage, fresh, exclude=tried)
            for sid, choice in primary.items():
                tried[sid].add(key_of(choice))
            videos = self.share()
            backup = self.allocate(options, reserved, usage,
                                   [s for s in fresh if self.wants_backup(s, primary.get(s["id"]), videos)], exclude=tried)
            for sid, choice in backup.items():
                tried[sid].add(key_of(choice))
            # Job ids must be unique across rounds: per-job analysis and QA results are keyed by them.
            jobs = ([(f"{sid}#{round_no}a", sid, c) for sid, c in primary.items()]
                    + [(f"{sid}#{round_no}b", sid, c) for sid, c in backup.items()])
            self.job_choice.update({job_id: c for job_id, _, c in jobs})
            # Longer scripts (one subtitle line per shot) download proportionally more clips per round.
            extra = max(0, len(pending) - 30)
            share = self.share()
            limit = time.monotonic() + (75 + 0.6 * extra if round_no == 0 else 50 + 0.4 * extra) * share
            got, failed, bad, running = self.download(jobs, streams, qa=checking, deadline=limit)
            failures.update(failed)
            qa_bad.update(bad)
            self.note_presenters(bad)
            carried = settle(carried + jobs, got, bad, running, last=round_no == 2)
            stamp(f"round{round_no+1}", began)
            pending = [s for s in pending if s["id"] not in final]   # provisional shots keep looking
            budget = (300 + 1.2 * max(0, len(self.shots) - 30)) * self.share()
            if pending and time.monotonic() - self.started > budget:
                self.notes.append(f"匹配已超过 {budget:.0f} 秒，停止继续替换")
                break
        for sid, (choice, path, _) in provisional.items():
            if sid not in final:   # no cleaner clip turned up: the provisional one is still relevant footage
                final[sid] = choice
                files[sid] = path
                self.place(sid, choice, key_of(choice)[0])
        pending = [s for s in pending if s["id"] not in final]
        stock = self.stock_options(pending, groups, results, final) if pending and self.pixa else {}
        if stock:
            # Licensed stock further down the same searches (20 results each, only the best few were offered):
            # clean footage for the lines still open, before reusing anything already in the film.
            began = time.monotonic()
            self.progress(86, f"用正版空镜补 {len(stock)} 镜")
            picked = self.allocate(stock, reserved, usage, [s for s in pending if s["id"] in stock], exclude=tried)
            for sid, choice in picked.items():
                tried[sid].add(key_of(choice))
            jobs = [(f"{sid}#s", sid, c) for sid, c in picked.items()]
            self.job_choice.update({job_id: c for job_id, _, c in jobs})
            got, failed, bad, running = self.download(jobs, streams, qa=checking,
                                                       deadline=time.monotonic() + (60 + 0.3 * len(jobs)) * self.share())
            failures.update(failed)
            qa_bad.update(bad)
            carried = settle(carried + jobs, got, bad, running, last=True)
            stamp("stock", began)
            pending = [s for s in pending if s["id"] not in final]
        if pending:
            # Last resort without unchecked footage: reuse a window that already passed QA,
            # preferring the same search group, then the nearest shot on the timeline.
            began = time.monotonic()
            for _, _, choice in self.drop_queued():
                release(choice)
            carried = [job for job in carried if job[0] in {j[0] for j in self.inflight.values()}]
            group_of_id = {sid: g["id"] for g in groups for sid in g["shots"]}
            donors = sorted(final.items(), key=lambda item: self.position[item[0]])
            hard_cap = max(HARD_MIN, math.ceil(len(self.timeline) * HARD_SHARE))
            shots_from = {}
            for choice in final.values():
                shots_from[key_of(choice)[0]] = shots_from.get(key_of(choice)[0], 0) + 1
            jobs = []
            for shot in pending:
                sid = shot["id"]
                person = bool(shot.get("persons")) and not self.unfilmed_shot(shot)
                ranked = sorted(donors, key=lambda item: (group_of_id.get(item[0]) != group_of_id.get(sid),
                                                          abs(self.position[item[0]] - self.position[sid])))
                for donor_id, donor in ranked:
                    # A named-person shot may only reuse footage verified for that name or people-free context.
                    if person and not (self.names_in_source(shot, donor["tile"]["source"])
                                       or donor["lane"] in ("ctx", "topic")):
                        continue
                    if not person and shots_from.get(key_of(donor)[0], 0) >= hard_cap + 1:
                        continue    # that video already fills its share of the film (plus one): try another
                    if donor_id not in self.position:
                        continue
                    if FUNERAL.search(donor["tile"]["source"].get("title", "")) and not DEATH_LINE.search(shot["narration"]):
                        continue
                    source = donor["tile"]["source"]
                    length = float(source.get("duration") or 0)
                    need = float(shot["duration"]) + MARGIN
                    if length < need + 0.1:
                        continue
                    key = key_of(donor)[0]
                    donor_end = donor["start"] + float(self.shots_by_id_all[donor_id]["duration"]) + MARGIN
                    # Same scene, different frames: continue after (or before) the passed window.
                    start = None
                    # Nearest free stretch of the same accepted video first, then anywhere else in it
                    # (each reused window is checked by QA again).
                    spread_out = sorted((x * 1.0 for x in range(0, int(max(0.0, length - need - 0.1)) + 1)),
                                        key=lambda x: abs(x - donor["start"]))
                    for candidate in (donor_end + 0.2, donor["start"] - need - 0.2,
                                      donor_end + need + 0.4, donor["start"] - 2 * need - 0.4, *spread_out):
                        if 0 <= candidate <= length - need - 0.1 and all(
                                candidate + need + 0.2 <= a or candidate >= b + 0.2 for a, b in reserved.get(key, [])):
                            start = candidate
                            break
                    if start is None:
                        continue
                    reserved.setdefault(key, []).append((start, start + need))
                    jobs.append((f"{sid}#r", sid, donor | {"start": round(start, 2), "lane": "repeat",
                                                           "base_lane": donor.get("base_lane", donor["lane"]),
                                                           "repeat_of": donor_id}))
                    shots_from[key_of(donor)[0]] = shots_from.get(key_of(donor)[0], 0) + 1
                    break
            self.job_choice.update({job_id: c for job_id, _, c in jobs})
            got, failed, bad, running = self.download(jobs, streams, qa=checking,
                                                       deadline=time.monotonic() + (45 + 0.3 * max(0, len(jobs) - 10)) * self.share())
            failures.update(failed)
            qa_bad.update(bad)
            carried = settle(carried, got, bad, running, last=True)   # late clips of the main rounds win
            for job_id, sid, choice in jobs:
                if sid in final:
                    continue
                if job_id in got and job_id not in bad and self.flags(job_id) & {"still", "dark", "jump"}:
                    # Reused windows are a last resort: a static one is usually a screenshot or text page.
                    bad[job_id] = "近似静态截图或黑场"
                if job_id in got and job_id in bad:
                    rejected.setdefault(sid, []).append((choice | {"job": job_id}, got[job_id], bad[job_id]))
                elif job_id in got:
                    final[sid] = choice | {"qa": "pass" if checking and job_id not in self.qa_unchecked else "unchecked",
                                           "job": job_id}
                    files[sid] = got[job_id]
            pending = [s for s in pending if s["id"] not in final]
            stamp("repeat", began)
        failures.update(self.stop_downloads())
        def keepable(shot, choice, reason):
            # Clearly wrong footage is worse than a neutral card: badly off-topic anywhere, and for an
            # ancient-history video modern, foreign or other-dynasty scenes.
            if re.search(r"严重不符|完全无关|完全不符|明显无关|毫无关系|毫无关联", reason):
                return False
            if self.ancient and re.search(r"现代|当代|清朝|清代|明朝|明代|民国|外国|国外|西方|欧美|朝代|时代背景|时代错|年代不符", reason):
                return False
            if self.ancient_shot(shot) and not re.search(r"黑屏|黑场|纯色|模糊|主持|博主|截图|界面|马赛克|水印|台标|字幕|弹幕", reason):
                return not re.search(r"现代|其他|他人|别人|不是|非.{0,6}本人|主体为|冒充", reason)
            if BROKEN_REJECTION.search(reason) or (FUNERAL.search(reason) and not DEATH_LINE.search(shot["narration"])):
                return False
            if self.ancient_shot(shot):
                return not re.search(r"现代|其他|他人|别人|不是|非.{0,6}本人|主体为|冒充", reason)
            if not shot.get("persons"):
                return True
            # Real footage of the person passes QA whatever its topic, so a rejection under the "must show
            # this person" role means someone else, an actor or a broken clip: never kept. People-free
            # background footage judged only off-topic may stay rather than a title card.
            lane = choice.get("base_lane", choice["lane"])
            if self.names_in_source(shot, choice["tile"]["source"]) and lane in ("pick", "person", "person_near") \
                    and not self.unfilmed_shot(shot):
                return False
            return not IDENTITY_REJECTION.search(reason) and (lane in ("ctx", "topic") or self.unfilmed_shot(shot))
        for shot in list(pending):
            sid = shot["id"]
            # A still picture waits until borrowing moving footage has been tried (below).
            kept = next(((c, path, reason) for c, path, reason in rejected.get(sid, [])
                         if "静态" not in reason and keepable(shot, c, reason)), None)
            if kept:
                choice, path, reason = kept
                final[sid] = choice | {"qa": "rejected_kept:" + reason}
                files[sid] = path
        cards = [s for s in self.shots if s["id"] not in final]
        if cards:
            # Related pictures with slow motion before any clip is shown again (the client saw one piggy-bank
            # clip twenty times when Bilibili gave nothing).
            began = time.monotonic()
            try:
                pictured = self.image_fill(cards)
            except (media.Cancelled, Budget):
                raise
            except Exception as exc:   # noqa: BLE001 — borrowing and cards still follow
                self.notes.append(f"相关图片补位失败：{type(exc).__name__}")
                pictured = {}
            for sid, (choice, path) in pictured.items():
                final[sid], files[sid] = choice, path
            cards = [s for s in cards if s["id"] not in pictured]
            if pictured:
                self.notes.append(f"{len(pictured)} 镜没有合适的视频，用了相关图片（慢慢推拉的动态画面）")
            stamp("pictures", began)
        footage = sorted((self.position[sid], path) for sid, path in files.items() if sid in self.position)
        # A real clip already accepted elsewhere (never the neighbouring one) beats a blurred card, but each clip
        # is lent out at most twice: one clip all over a film is what viewers notice first.
        reused = 0
        for shot in list(cards):
            donor = self.borrow(shot, final, files, limit=2)
            if donor:
                final[shot["id"]] = final[donor] | {"lane": "reuse", "qa": "reused", "repeat_of": donor}
                files[shot["id"]] = files[donor]
                cards.remove(shot)
                reused += 1
        if reused:
            self.notes.append(f"{reused} 镜没有新的可用画面，借用了附近镜头已通过检查的画面")
        for shot in list(cards):
            # Nothing to borrow: a still picture beats a title card, except as the opening shot.
            sid = shot["id"]
            still = None if self.position[sid] == 0 else next(
                ((c, path, reason) for c, path, reason in rejected.get(sid, [])
                 if "静态" in reason and keepable(shot, c, reason)), None)
            if still:
                choice, path, reason = still
                final[sid] = choice | {"qa": "rejected_kept:" + reason}
                files[sid] = path
                cards.remove(shot)
        for shot in cards:
            # Background: a frame of the nearest accepted clip (blurred), so the card belongs to the video.
            near = min(footage, key=lambda item: abs(item[0] - self.position[shot["id"]]), default=None)
            final[shot["id"]] = {"tile": {"source": {"provider": "card", "id": "card", "title": "文字卡片"}, "t": 0},
                                 "start": 0, "lane": "card", "rank": 0, "qa": "card"}
            files[shot["id"]] = self.title_card(shot, near[1] if near else None)
        if cards:
            self.notes.append(f"{len(cards)} 镜没有可用实拍素材，使用文字卡片补位")
        self.progress(92, "保存选片")
        self.save(final, files)
        unmatched = [s["id"] for s in self.shots if s["id"] not in final]
        report = {"selected": len(final), "unmatched": unmatched, "elapsed_seconds": round(time.monotonic() - self.started, 2),
                  "timing": self.timing, "model_calls": self.calls, "qa_rejected": qa_bad, "download_failures": failures,
                  "lanes": {lane: sum(1 for c in final.values() if c["lane"] == lane)
                            for lane in ("news", "pick", "person", "person_near", "ctx", "topic", "neighbor", "near", "stock", "image",
                                         "repeat", "reuse", "card")},
                  "qa_kept_rejected": [sid for sid, c in final.items() if str(c.get("qa", "")).startswith("rejected")],
                  "flags": {"still": sum(1 for c in final.values() if "still" in self.flags(c.get("job"))),
                            "wm": sum(1 for c in final.values() if "wm" in self.flags(c.get("job"))),
                            "cropped": sum(1 for c in final.values()
                                           if (self.clean.get(c.get("job")) or {}).get("zoom", 1) > 1.001)},
                  "sources": len({key_of(c)[0] for c in final.values()}),
                  "from_index": sum(1 for c in final.values() if c["tile"]["source"].get("indexed")),
                  "providers": {p: sum(1 for c in final.values() if c["tile"]["source"]["provider"] == p)
                                for p in ("bilibili", "pixabay")},
                  "clean": {sid: {"crop": (self.clean.get(c.get("job")) or {}).get("crop"),
                                  "overlays": [(o["kind"], o["box"]) for o in
                                               (self.clean.get(c.get("job")) or {}).get("overlays", [])],
                                  "residual": [(o["kind"], o["box"]) for o in
                                               (self.clean.get(c.get("job")) or {}).get("residual", [])],
                                  "flags": sorted(self.flags(c.get("job")))} for sid, c in final.items()},
                  "bilibili_logged_in": self.bili.logged_in, "notes": self.notes[:40],
                  "clip_encoders": dict(self.encoders), "text": dict(self.text_found),
                  "waited": {k: round(v, 1) for k, v in throttle.waited.items()},
                  "model_seconds": {k: round(v, 1) for k, v in self.model_seconds.items()},
                  "fetch_seconds": {"count": len(self.fetch_seconds),
                                    "max": max([x[1] for x in self.fetch_seconds] or [0]),
                                    "mean": round(sum(x[1] for x in self.fetch_seconds) / max(1, len(self.fetch_seconds)), 2)}}
        store.mutate(self.pid, lambda p: p.update(express_report=report), internal=True)
        return report


def warm(project, shots, check):
    """Footage search and preview sheets only (both cached on disk, no model calls). Produce runs this
    while the narration is generated and aligned, so the matching stage starts with them ready."""
    matcher = Matcher(project, shots, check, lambda *_: None)
    try:
        groups = matcher.groups()
        chosen = matcher.candidates(groups, matcher.search(groups))
        matcher.all_tiles(chosen, {g["id"] for g in groups if g.get("person")})
        matcher.screen_presenters(chosen)
    finally:
        matcher.stop_downloads()
        matcher.bili.close()
        if matcher.pixa:
            matcher.pixa.close()


def match(project, shots, check, progress):
    matcher = Matcher(project, shots, check, progress)
    try:
        return matcher.run()
    finally:
        matcher.stop_downloads()
        matcher.bili.close()
        if matcher.pixa:
            matcher.pixa.close()
