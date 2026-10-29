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
# Sources per group: the picker sees 60 tiles of a group at most, so beyond about 14 videos every extra one is
# fetched, profiled and screened for nothing. 0.4.9's bigger library filled 10 local places a group on top
# of 12 Bilibili ones and made the client's preview step 1.5-12 minutes a film.
BILI_PER_QUERY = 5
BILI_AFTER_LOCAL = 3   # a query the library already answers with LOCAL_PER_QUERY videos takes few live results
                       # (the client: a big library matches well enough; live searches bring the 营销号 junk)
PERSON_PER_QUERY = 5   # named people have little real footage: look a little wider before falling back
PIXA_PER_QUERY = 3     # licensed stock: clean, no subtitles or contact text; strong for generic and abstract lines
LOCAL_PER_QUERY = 4     # videos from the local footage index (footage_index) per query, ahead of live results
GROUP_SOURCES = 20      # at most this many videos a group (stock clips included)
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
# One source, one look (the client, 10-28: the same girl in pink eight times in one film, from other moments of one
# drama). Every lane but a named person's own footage and a news report about the event: one video at most
# SOURCE_MAX windows a film, SCENE_GAP seconds apart and never in neighbouring shots; lines without a clip of
# their own take a NEW video from the library (its frames are described one by one) instead of another moment of
# a video already shown.
SOURCE_MAX = 2
SCENE_GAP = 30.0
LIB_PER_LEVEL = (1, 1)  # library videos offered to the picker per line: from its specific words, from its broader ones
                        # (each costs a B站 preview call; two from the first words made previews 60-75 s a film)
LIB_TILES = 3           # preview frames of such a video: the matching moments first
LIB_FILL_PER_LINE = 2   # library candidates downloaded per open line in one fill pass (one usually passes QA)
AUDIT_SECONDS = 40      # download time the final repeat check's replacements and the topic round get, flat
TOPIC_SECONDS = 25      # the topic round, a last fill before borrowing
LIBRARY_SECONDS = 90    # ...and the library round at most (scaled by films at once it waited up to three minutes)
FRONTIER = 1            # options per line whose play addresses are asked for before the first round (the rest when
                        # a round allocates them: every address is two B站 calls)
SIMILAR_BITS = 8        # two shots whose frames differ by at most this many dHash bits (2 of 3 frame pairs) look alike
# Lines that are about another country may show foreigners; elsewhere frames described as foreign are skipped.
FOREIGN_LINE = re.compile(r"美国|美军|美方|日本|韩国|朝鲜|欧洲|欧盟|英国|法国|德国|意大利|西班牙|俄罗斯|俄方|乌克兰|印度|"
                          r"以色列|伊朗|中东|非洲|巴西|澳大利亚|加拿大|越南|泰国|新加坡|外国|国外|海外|西方|欧美|白宫|特朗普|拜登")
NEWS_CAP = 6          # windows of one news report about the event (news_pool) per film
NEWS_LATER = (0.0, 5.0, 10.0)
NEWS_LATER_ONSITE = (0.0, 4.0)   # windows already moved onto on-site footage by its preview frames
NEWS_WAIT = 150.0      # seconds the matching waits, after picking, for the news reports to be transcribed
LOW_HEIGHT = 480       # without a B站 login many drama and short-play clips stop at 480p: usable after the sharp
                       # ones (a social story otherwise loses most of its footage and repeats the rest)
STOCK_SHARE = 0.15     # foreign stock (Pixabay) at most this share of a film: for a Chinese story its foreign
                       # couples and weddings read as filler ("都是空镜头"), whatever the planner asked for
TWIN_BITS = 8
SPRITE_SHEETS_KEPT = 12   # preview sheets kept decoded (about 13 MB each in grey)
# Footage a line about a named person may borrow as background: it was chosen as context, not as someone else
# (a 70-line film about two named people got 19 title cards while library and picture fill were off limits).
CONTEXT_LANES = ("ctx", "topic", "library", "image", "stock")
PICTURE_SECONDS = 90    # time the picture fallback may take per film (searches and downloads)
PICTURE_LINES = 30      # lines it looks for pictures for
FILMED_KINDS = {"实拍", "影视剧"}     # library frame kinds that are footage (not a cartoon, a presenter or text)
MARGIN = 0.3   # seconds of extra footage per window; tolerates sources ending early
FOOTAGE_WORDS = re.compile(r"素材|空镜|航拍|实拍|纪录|影像|画面|现场|4K|4k|高清|原声|新闻|珍贵|全景|延时|记录")
NOISE_WORDS = re.compile(
    r"教程|讲解|解说|课程|讲座|课件|PPT|ppt|直播|游戏|动画|MMD|手书|鬼畜|CAD|图纸|三维模型|建模|广告|带货|测评|开箱|"
    r"reaction|反应|翻唱|MV|AMV|搞笑|模拟器|我的世界|原神|王者|AI绘|AI生成|配音秀|考研|考试|答辩|股票|基金|大A|"
    r"财经|行情|招聘|面试|求职|小说|有声书|电台|音乐|歌曲|BGM|bgm|伴奏")
# Uploads that say they are sped up (re-uploaders speed clips up against duplicate checks) or are memes:
# never used.
SKIPPED_UPLOADS = re.compile(r"倍速|加速版|快进|变速|表情包|梗图|斗图|擦边|性感|福利|比基尼|丝袜|写真|热舞|美腿|内衣|泳装|"
                             r"吻戏|床戏|亲热|激情|大尺度")
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
# Clips with a watermark or 花字 left in view (the client's biggest complaint, 10-27): used only after related
# pictures and borrowing, an edge watermark (painted out where small) before 花字 across the middle.
MARKED = "有去不掉的水印或文字"
MARKED_MIDDLE = "画面中间有花字或大段文字"
NARROW = "竖屏或只占半边的画面"
MARK_REJECTION = re.compile(r"水印|花字|大字|字幕条|标题字|台标|logo|LOGO|二维码|贴纸|表情包")
FUNERAL = re.compile(r"葬礼|遗体|遗容|遗像|灵堂|灵柩|追悼|悼念|花圈|讣告|火化|送别|逝世后|告别仪式")
DEATH_LINE = re.compile(r"逝世|去世|病逝|离世|悼念|追悼|葬礼|长眠|告别")
BROADCAST = re.compile(r"CCTV|cctv|央视|中央电视台|卫视|电视台|CGTN|凤凰卫视|凤凰资讯|凤凰网|凤凰视频")   # a channel logo sits top-left
ARCHIVE_TITLE = re.compile(r"珍贵|影像|纪录片|央视|新闻|原声|档案|讲课|讲话|采访|实录")

PICK_SYSTEM = """你在为营销号视频挑选实拍素材。图片是若干视频的缩略图联系表，每格左上角有黄色编号。
请为每个镜头挑出合适的格子编号（最好的在前，最多6个），只输出 JSON：{"S1":["12","7"],"S2":[]}。
普通镜头标准：画面主体要和口播讲的对象一致（画面建议只是参考），不要求型号、年份、地点、动作逐一一致：讲人和人之间的事（夫妻、家人、同事、官员、民众、士兵），画面要有相应的人和场景，不选风景、空镜、物品；讲某个国家、组织的事，画面要是那个国家、组织的，不选别国的同类画面（讲也门战事不选中国火箭军、中国海军）；讲中国国内的人和事，不选明显是外国人或外国街景的画面。情感、家庭、社会故事可以选影视剧、短剧里情节相符的片段（有人物、有情绪）。
必须排除：性暗示或暴露的画面（内衣、泳装、裸露身体、亲热接吻、床戏、擦边）、血腥暴力（殴打、流血）、主持人或博主对镜头口播、PPT/文字页/图表/排行榜截图为主、手机或电脑屏幕截图（评论区、聊天记录、网页/App 界面）、大面积马赛克、片头片尾和转场、频道订阅页、黑屏或严重模糊、大面积硬字幕/弹幕/贴纸/台标遮挡画面、游戏/动画/CG（除非口播就是讲这些）、与口播明显无关的内容、广告。
新闻、纪录片里真实存在的人物的脸部特写，除非该镜头要求此人物，否则尽量不选（容易被当成口播里讲的人）；影视剧、短剧里的演员不受此限。
优先选有明显动作或镜头运动的视频画面；静态插画、照片、幻灯片式的画面尽量不选（口播在讲这幅画、这件文物时除外）。
画面干净比贴题更重要：画面上有后加的大字、花字、标题条、水印、表情包、贴纸，或是竖屏、两边黑边/模糊边的，不选；宁可选大致相关但干净的画面。"""

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
bad：只列“明显不合格”的镜头。不合格指：任意一帧有性暗示或暴露（内衣、泳装、裸露身体、亲热接吻、床戏、擦边），或血腥暴力（殴打、拉扯、流血）；任意一帧出现主持人、博主或嘉宾对着镜头讲话（访谈/口播/演播室；要求的具名人物本人的历史讲话影像除外）；与口播明显无关/跑题；要求具名人物的镜头画面是影视剧/情景再现的演员扮演（老年代事件却是彩色高清电影感画面）；PPT/文字页/图表为主；手机或电脑屏幕截图（评论区、聊天记录、网页/App 界面），即使口播在讲网友评论；大面积马赛克或打码；表情包、卡通表情、贴纸、梗图、对话气泡图标叠在画面上，或画面本身是表情包式的夸张反应镜头（对着镜头狂笑、做鬼脸）；画面上后加的大号花字、标题字（不是场景里本来就有的字）；整段画面静止不动（一张图片或定格）；黑屏、纯色、严重模糊；大面积硬字幕、弹幕、贴纸或台标遮挡主体；画面上有联系方式或引流文字（电话号码、微信号、QQ号、“私信”“加V”“需要视频”等，再小也算）；游戏/动画（口播不是讲这些时）；要求具名人物的镜头却明显是别人或演员。
wm：画面上仍能看到的台标、水印、平台或作者署名（包括叠在画面中间或侧边的半透明频道名/账号名，如“@名字”“某某说”“某某影视”，再淡也要列出）、原视频字幕或文字条、画面中间的花字/标题/滚动文字，写出所有位置：左上/右上/左下/右下/顶部/底部/左侧/右侧/中间（字幕、文字条写“底部”）。小角标也要列出，它们会被裁掉，不算不合格。
若给出“整篇领域”，画面属于其他领域（如讲乒乓球却是电竞、板球、足球等其他运动，或卡通动画、3D建模渲染、频道片尾订阅页、排行榜截图）一律不合格；标为“古代人物”的镜头里，表现此人的动画、插画不算其他领域。
股市K线和走势图、数据图表、商品和品牌logo、街道招牌、包装上的字是画面本身的内容，不算文字页，口播相关就合格。
普通镜头（没有要求具名人物）：用电视剧、短剧、栏目剧片段里的演员表现口播讲的情节（夫妻吵架、婆媳矛盾、查看手机、离婚、职场冲突等）是合格的，不要因为是演员扮演或情景再现判不合格。画面主体和口播讲的对象一致即可，细节不必一致：例如口播讲港口工作人员协作，画面是港口作业远景也算合格；不要因为型号、年份、具体动作、人物细节或特写不一致而判不合格；菜品种类、制作步骤、同一场所的不同环节（如揉面与拉面、后厨与前厅、汤与面）也不必一致。以下算跑题（不合格，原因写“跑题”）：口播讲人和人之间的事（夫妻、家人、同事、官员、民众），画面却是风景、空镜、物品、动物，看不出相关的人；口播讲中国国内的人和事，画面明显是外国人或外国街景；口播讲某个国家、组织，画面明显是另一个国家的（如讲也门战事却是中国火箭军发射、中国海军演习）；科普讲解、课堂课件、带货、游戏解说之类与口播无关的视频画面；画面题材与口播完全无关。
要求具名人物的镜头：只要画面是此人本人的真实影像或照片就合格，即使与口播所讲的具体事件、场景不一致也不要判不合格。
标为“古代人物”的镜头：此人没有真实影像，画像、雕像、影视剧/纪录片/动画/插画里表现此人的形象都合格，不要因演员扮演、动画或插画判不合格；只有现代场景、明显是其他具名人物、主持人对镜头讲解才不合格。
任何镜头：口播没讲到逝世、悼念时，葬礼、遗体、遗像、灵堂、花圈、追悼会画面不合格。"""


BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
           "Chrome/124.0 Safari/537.36")


SEARCH_QUIET = 300      # seconds after a refused round of searches that films use the local library only


def _stamp(name: str) -> float:
    """When the note `name` in the data folder was last written (0: never); shared by every worker process."""
    try:
        return float(json.loads((paths.DATA / name).read_text("utf-8"))["at"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def _mark(name: str, why: str) -> None:
    target = paths.DATA / name
    try:
        temporary = paths.temporary(target)
        temporary.write_text(json.dumps({"at": time.time(), "why": why}, ensure_ascii=False), "utf-8")
        paths.publish(temporary, target)
    except OSError:
        pass


class BiliUnavailable(RuntimeError):
    """Bilibili refused this computer (searches all empty, or play addresses refused): a film made now would
    be a few stock clips over and over, so the video waits and is made again later."""


class FootageStarved(BiliUnavailable):
    """This one film found almost no usable footage (a niche topic, or downloads refused just for it): it is
    made again later on its own, without pausing the rest of the computer's videos."""


class TooRepetitive(FootageStarved):
    """The film would show the same pictures again and again because B站 refused the new videos it needed: made
    again a little later instead (the client, 10-28: repeats make a film useless)."""


REPEAT_RETRIES = 0      # films made again for too many repeats: off — with B站 refusing, each redo was a whole film
                        # again plus four minutes' wait (a dorm round took 57 minutes for six films)


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


def _boxed(image: Image.Image) -> bool:
    """A preview frame of a vertical video: taller than wide, or a picture in the middle with both sides
    black (wider than a fifth of the frame each)."""
    if image.width < image.height * 1.2:
        return True
    small = image.convert("L").resize((64, 36))
    cols = [max(small.getpixel((x, y)) for y in range(36)) for x in range(64)]
    side = 13    # a fifth of 64
    if max(cols[:side]) < 40 and max(cols[-side:]) < 40 and max(cols[side:-side]) >= 60:
        return True
    # A vertical video on a blurred, enlarged copy of itself: both sides carry almost no detail while the
    # middle is sharp (the client got an old man between two smeared brown bands).
    edges = small.filter(ImageFilter.FIND_EDGES)
    def detail(x0, x1):
        return ImageStat.Stat(edges.crop((x0, 2, x1, 34))).mean[0]
    middle = detail(24, 40)
    return middle >= 12 and detail(1, 12) < middle * 0.25 and detail(52, 63) < middle * 0.25


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
        self.library_rich = set()   # groups whose lines all found library videos of their own (fewer live searches)
        self.fill_errors = {}   # job id -> why its download or text check failed (written back to the library's checks)
        self.window_notes = []  # library moments judged here, for footage_index.record_checks
        self.similar = []       # sizes of the groups of alike-looking shots in the finished film (audit)
        self.looked, self.relooked = {}, set()   # clip path -> dHash of three frames; jobs analysed again by audit()

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
        # Bilibili refused searches from this computer a few minutes ago: asking again only keeps the refusal
        # going, so this film takes its footage from the local library (downloads usually still work).
        quiet = time.time() - _stamp("bili-search-refused.json") < SEARCH_QUIET
        if quiet:
            self.notes.append("B站搜索刚被限流，这条先用本地素材库的视频")
        # The library first (no network): a group whose every line already has library videos of its own searches
        # Bilibili with one query, not two (every live result costs preview calls, and the calls got the dorm PC
        # refused: one film read 256 candidates' previews in 11 minutes).
        library = self.library_options(groups)
        self.library_rich = {g["id"] for g in groups if not g.get("person") and g.get("shots")
                             and len(library.get(g["id"], [])) >= len(g["shots"])}
        for g in groups:
            live = 1 if g.get("id") in self.library_rich else 2
            for q in g["zh"][:live] if not quiet else []:
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
        refused = len(asked) >= 4 and not any(asked)
        if refused:
            _mark("bili-search-refused.json", f"B站的 {len(asked)} 次搜索全部没有结果")
        # The group's long queries in the library as well, where its lines found little of their own (bigrams of a
        # long query match loosely: the lines' own words come first).
        from . import footage_index
        for q in dict.fromkeys(q for g in groups if g.get("id") not in self.library_rich for q in g["zh"][:2]):
            try:
                results[("local", q)] = footage_index.search(q, limit=8)
            except Exception as exc:   # a missing or damaged index only costs this lane
                self.notes.append(f"本地素材库检索失败: {type(exc).__name__}")
                break
        for gid, rows in library.items():
            results[("lib", gid)] = rows
        local = sum(1 for (kind, _), rows in results.items() if kind in ("local", "lib") and rows)
        if (refused or quiet) and local < max(2, len(groups) // 3) and self.cfg.get("bili_limit_mode", "wait") == "wait":
            raise BiliUnavailable("B站搜索被限流，本地素材库里也没有这条要的画面")
        return results

    def penalties(self) -> dict:
        """Library videos other recent films already show cost a little (a batch on one topic repeats itself)."""
        # Strong: a batch of a hundred films on a few topics must not show the same clips again (a platform reads that
        # as re-posted content); a word score is about 1, so a video another film used comes after unused ones.
        return {key: min(8.0, 3.0 * count) for key, count in (self.elsewhere or {}).items()}

    def foreign_ok(self, shot) -> bool:
        return bool(FOREIGN_LINE.search(shot.get("narration", "") + (self.plan.get("domain") or "")))

    def library_options(self, groups) -> dict:
        """gid -> library videos found by its lines' own short words (express_plan.library_words: specific, then
        broader), each at the frame that matched, for the picker next to the live search results. Every line
        gets different videos, so the picks of a group spread over many sources. Lines naming a person are left
        to their name's searches."""
        from . import express_plan, footage_index
        if not hasattr(footage_index, "find") or not hasattr(express_plan, "library_words"):
            return {}
        jobs = []
        for g in groups:
            if g.get("person") or g.get("topic"):
                continue
            for sid in g.get("shots") or []:
                shot = self.shots_by_id_all.get(sid)
                if shot and not shot.get("persons"):
                    jobs.append((g["id"], shot))
        penalize = self.penalties()
        def run(job):
            gid, shot = job
            self.check()
            found = []
            try:
                for level, words in enumerate(express_plan.library_words(shot)[:len(LIB_PER_LEVEL)]):
                    found += [(level, row) for row in footage_index.find(
                        words, limit=LIB_PER_LEVEL[level] + 3, penalize=penalize, foreign_ok=self.foreign_ok(shot))]
            except media.Cancelled:
                raise
            except Exception as exc:   # noqa: BLE001 — a missing or damaged index only costs this lane
                self.notes.append(f"素材库检索失败: {type(exc).__name__}")
            return gid, shot["id"], found
        out, taken = {}, set()
        with execution.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, jobs))
        for gid, sid, found in sorted(results, key=lambda item: self.position.get(item[1], 0)):
            counts = [0] * len(LIB_PER_LEVEL)
            for level, row in found:
                key = f"{row['provider']}:{row['id']}"
                if key in taken or counts[level] >= LIB_PER_LEVEL[level]:
                    continue
                taken.add(key)
                counts[level] += 1
                out.setdefault(gid, []).append(row | {"lib_for": sid, "lib_level": level})
        if out:
            self.notes.append(f"素材库按每句的画面短词找到 {sum(len(v) for v in out.values())} 个视频供选片")
        return out

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
            local_taken = {}
            # Library videos found by the lines' own words come first: described frame by frame, at the moment that
            # matched, one set of videos per line. With two or more a line, live searches add only a few (the
            # client: the live results bring the 营销号 re-uploads with their marks).
            library = results.get(("lib", g["id"]), [])
            for row in library:
                key, title_key = (row["provider"], row["id"]), (row["provider"], row.get("title", "")[:30])
                if key not in seen and title_key not in seen and self.score(row, g, shots_by_id) is not None:
                    seen.update((key, title_key))
                    picked.append(row)
            if g["id"] in self.library_rich:
                bili_quota = BILI_AFTER_LOCAL
            for kind, queries, quota in (("local", g["zh"][:2], LOCAL_PER_QUERY), ("bili", g["zh"][:2], bili_quota),
                                         ("pixa", g["en"][:2] if self.pixa else [], PIXA_PER_QUERY)):
                for q in queries:
                    if kind == "bili" and not g.get("person") and (local_taken.get(q, 0) >= LOCAL_PER_QUERY
                                                                   or bili_quota == BILI_AFTER_LOCAL):
                        quota = BILI_AFTER_LOCAL
                    elif kind == "bili":
                        quota = bili_quota
                    ranked = []
                    for rank, row in enumerate(results.get((kind, q), [])):
                        score = self.score(row, g, shots_by_id)
                        if score is None:
                            continue
                        if kind == "pixa" and (not self.stock_fits(row, q) or not self.oriented(row)
                                               or (self.stock_people(row) and not self.group_foreign(g))):
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
                    if kind == "local":
                        local_taken[q] = taken
            chosen[g["id"]] = picked[:GROUP_SOURCES + len(library)]
        return chosen

    STOCK_PEOPLE = re.compile(r"\b(man|men|woman|women|girl|boy|people|person|couple|family|child|children|kid|kids|baby|"
                              r"student|students|businessman|lady|guy|friends|mother|father|senior|elderly|teen|teenager|"
                              r"portrait|face|lovers|bride|groom|worker|workers|crowd)\b", re.I)

    def stock_people(self, row) -> bool:
        """Foreign stock footage of people (the dorm test films' foreign couples and silhouettes in a story about
        Chinese students): only for lines about another country."""
        return bool(self.STOCK_PEOPLE.search(" ".join(str(row.get(k) or "") for k in ("tags", "title", "description"))))

    def group_foreign(self, group) -> bool:
        return any(self.foreign_ok(self.shots_by_id_all[sid]) for sid in group.get("shots") or [] if sid in self.shots_by_id_all)

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
        tiles, hashes, boxed = [], [], 0
        landscape = self.project.get("aspect", "16:9") != "9:16"
        for p in picked:
            image = es.bilibili_tile(self.bili, p)
            if _dark(image):
                continue
            if landscape and _boxed(image):
                boxed += 1
                continue
            h = _dhash(image)
            if any(bin(h ^ other).count("1") <= 5 for other in hashes):
                continue
            hashes.append(h)
            tiles.append({"source": row, "t": p["t"], "image": image, "hash": h})
        if boxed and boxed >= len(tiles):
            return []    # a vertical video (or one inside black bars): never shown in a landscape film
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
                frames = frames[::max(1, len(frames) // 48)]    # the median of 48 shows a fixed mark as well as 120
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
                count = PERSON_TILES if key in person else LIB_TILES if row.get("lib_for") else TILES_PER_BILI
                return key, self.tiles_for(row, count)
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
        # Previews that failed are not retried; library videos found by a line's words were already sorted out by
        # their described frames (presenter and text frames counted per video).
        rows = {(r["provider"], r["id"]): r for rs in chosen.values() for r in rs if r["provider"] == "bilibili"
                and not r.get("lib_for") and (tiles is None or tiles.get((r["provider"], r["id"])))}
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
    def pick(self, group, tiles, shots_by_id, halved=False):
        if not tiles:
            return {}, {}
        tiles = tiles[:3 * SHEET_COLS * SHEET_ROWS]
        sheets, labels = [], {}
        for start in range(0, len(tiles), SHEET_COLS * SHEET_ROWS):
            sheet, part = _sheet(tiles[start:start + SHEET_COLS * SHEET_ROWS], start + 1)
            sheets.append(sheet)
            labels.update(part)
        if group.get("topic"):
            # The film-wide reserve: lines without footage of their own take these before any clip is shown
            # again or a marked one is used, so it should be wide and clean (the client, 10-27: "less on topic is
            # fine, but clean and not repeated").
            lines = [f"S1：{TOPIC_SHOT}（主题：{'、'.join(group['zh'])}）。这一项多选，最多12个，尽量来自不同的视频；"
                     "只选画面干净的（没有大字、花字、水印、竖屏黑边、表情包）"]
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
            try:
                result, _ = self.chat(system, content, max_tokens=120 + 60 * len(ids), task="express_pick")
            except Exception as exc:   # noqa: BLE001
                # The service refuses a whole request over one picture now and then (HTTP 400, its content
                # check); without picks the group's lines all fall back to borrowed clips. Halves, one by one.
                if "HTTP 400" not in str(exc) or len(tiles) < 8 or halved:
                    raise
                merged = {}
                for part in (tiles[:len(tiles) // 2], tiles[len(tiles) // 2:]):
                    try:
                        got, _ = self.pick(group, part, shots_by_id, halved=True)
                    except Exception:   # noqa: BLE001 — the other half may still answer
                        continue
                    for sid, picked in got.items():
                        merged.setdefault(sid, []).extend(picked)
                return {sid: v[:12 if group.get("topic") else 6] for sid, v in merged.items()}, {"ctx": []}
            with self.lock:
                self.calls["pick"] += 1
            es._write_json(cache, result)
        picks, ctx = {}, []
        for key, value in result.items() if isinstance(result, dict) else []:
            values = [str(v).strip() for v in (value if isinstance(value, list) else []) if str(v).strip() in labels]
            if key == "_ctx":
                ctx = values
            elif key in ids:
                picks[ids[key]] = [labels[v] for v in dict.fromkeys(values)][:12 if group.get("topic") else 6]
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
                            and self.oriented(row) and float(row.get("duration") or 0) >= float(shot["duration"]) + MARGIN
                            and not (self.stock_people(row) and not self.foreign_ok(shot))):
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
                if str(choice.get("qa", "")).startswith("rejected_kept") or self.flags(choice.get("job")) & {"wm", "narrow"}:
                    continue    # a marked clip is never lent out: it would show its marks again and again
                source = choice["tile"]["source"]
                if limit and counts.get((source.get("provider"), source.get("id")), 0) >= limit:
                    continue
                donor_shot = self.shots_by_id_all.get(sid) or {}
                if shot.get("persons"):
                    if not (self.names_in_source(shot, choice["tile"]["source"]) or choice.get("lane") in CONTEXT_LANES):
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

    def any_clean(self, shot, final, files):
        """The last resort before nothing: a clean clip of this film for a line borrow() found no donor for (too
        short, too near, every clip lent out). Longer first (a short one holds its last frame), then not the
        neighbour, then lent out least; a line about a named person takes no one else's footage."""
        position = self.position[shot["id"]]
        need = float(shot["duration"])
        lent = {}
        for choice in final.values():
            if choice.get("repeat_of"):
                lent[choice["repeat_of"]] = lent.get(choice["repeat_of"], 0) + 1
        best = None
        for sid, choice in final.items():
            if sid not in files or choice.get("lane") in ("card", "reuse") or sid not in self.position:
                continue
            if str(choice.get("qa", "")).startswith("rejected_kept") or self.flags(choice.get("job")) & {"wm", "narrow"}:
                continue
            donor_shot = self.shots_by_id_all.get(sid) or {}
            if shot.get("persons"):
                if not (self.names_in_source(shot, choice["tile"]["source"]) or choice.get("lane") in CONTEXT_LANES):
                    continue
            elif donor_shot.get("persons") and choice.get("lane") in ("pick", "person", "person_near"):
                continue
            try:
                length = es.probe_duration(files[sid])
            except ValueError:
                continue
            if length * 2.0 < need + 0.1:
                continue                     # even at half speed it would not fill the line
            key = (length < need - 0.1, abs(self.position[sid] - position) < 2, lent.get(sid, 0), abs(self.position[sid] - position))
            if best is None or key < best[0]:
                best = (key, sid, length)
        if not best:
            return None
        _, sid, length = best
        return sid, (round(min(2.0, (need + 0.1) / length), 3) if length < need - 0.1 else 1.0)

    # ------------------------------------------------------------- library
    # Words too general to find a line's footage alone ("男子" matches every man in the library).
    VAGUE = {"男子", "女子", "男性", "女性", "人物", "人群", "多人", "画面", "场景", "室内", "户外", "特写", "现代"}
    TOPIC_NOISE = re.compile(r"电视剧片段|电视剧|短剧|栏目剧|片段|实拍|视频|素材|现场|新闻|航拍|空镜|纪录片|历史影像|画面|镜头|高清|4K")

    def library_words(self, shot, levels) -> list[list[str]]:
        """Word sets to look the line up with: its planned levels (express_plan.library_words: specific to broad),
        "wide" = each of its words alone (vague ones last), "topic" = the film's topic and domain words."""
        from . import express_plan
        planned = express_plan.library_words(shot) if hasattr(express_plan, "library_words") else []
        if levels in ("wide", "all"):
            words = list(dict.fromkeys(w for ws in reversed(planned) for w in ws))
            # Alone, a vague person word finds any stranger and a thing or a place finds reviews and scenery (a card
            # reader for a line about a boy's phone): single words are people doing something, or a named role.
            alone = getattr(express_plan, "only_things", None)
            wide = [[w] for w in words if w not in self.VAGUE and not (alone and alone([w]))]
            if levels == "wide":
                return wide
            # Every level at once, specific first: one download round instead of one per level (four rounds that
            # each waited their whole time limit made a film three minutes slower).
            out = []
            for ws in [p for p in planned if p] + wide:
                if ws not in out:
                    out.append(ws)
            return out
        if levels == "topic":
            topic = self.plan.get("topic") or {}
            found = [w for q in (topic.get("zh") or []) + [self.plan.get("domain") or ""]
                     for w in self.TOPIC_NOISE.sub(" ", q).split() if len(w) >= 2]
            words = list(dict.fromkeys(found))
            return [words[:2]] * (len(words) >= 2) + [[w] for w in words if w not in self.VAGUE]
        return [planned[level] for level in levels if level < len(planned) and planned[level]]

    def library_jobs(self, pending, final, reserved, streams, levels=(0, 1), tag="a", per_line=LIB_FILL_PER_LINE,
                     per_set=2) -> list:
        """Lines still without footage: a window from a library video the film does not show yet, whose described
        frame (a filmed one between filmed neighbours, footage_index.find) carries the line's words, at that
        moment. No previews or picker: the library's descriptions choose, QA and the clean-up check the window.
        Videos the dorm PC already checked clean come first. [(job_id, sid, choice)]"""
        from . import footage_index
        if not hasattr(footage_index, "find"):
            return []
        used = {f"{c['tile']['source'].get('provider')}:{c['tile']['source'].get('id')}" for c in final.values()}
        used |= {f"{provider}:{source_id}" for (provider, source_id), windows in reserved.items() if windows}
        penalize = self.penalties()
        planned = []
        for shot in pending:
            sid = shot["id"]
            need = float(shot["duration"]) + MARGIN
            tries = 0
            for words in self.library_words(shot, levels):
                if tries >= per_line:
                    break
                self.check()
                rows = footage_index.find(words, limit=per_set + 4, exclude=used, penalize=penalize,
                                          foreign_ok=self.foreign_ok(shot))
                from_set = 0
                for row in rows:
                    if from_set >= per_set:
                        break
                    key = f"{row.get('provider')}:{row.get('id')}"
                    if row.get("provider") != "bilibili" or key in used or row.get("contact"):
                        continue
                    if FUNERAL.search(row.get("title", "")) and not DEATH_LINE.search(shot["narration"]):
                        continue
                    if SKIPPED_UPLOADS.search(row.get("title", "") + row.get("tags", "")):
                        continue
                    length = float(row.get("duration") or 0)
                    moment = next((float(t) for t in row.get("moments") or [] if max(0.0, float(t) - 1.0) + need <= length - 0.1),
                                  None)
                    if moment is None:
                        continue
                    start = max(0.0, moment - 1.0)
                    used.add(key)            # one new video per window: the point is not to repeat
                    reserved.setdefault(("bilibili", row["id"]), []).append((start, start + need))
                    base = "ctx" if shot.get("persons") else "library"
                    planned.append((f"{sid}#lib{tag}{tries}", sid, {
                        "tile": {"source": row, "t": moment, "hash": None}, "start": round(start, 2),
                        "lane": "library", "base_lane": base, "rank": tries, "moment": moment,
                        "words": " ".join(words)}))
                    tries += 1
                    from_set += 1
                    if tries >= per_line:
                        break
        if not planned:
            return []
        streams.update(self.resolve({choice["tile"]["source"]["id"]: choice["tile"]["source"] for _, _, choice in planned}))
        for _, _, choice in planned:
            stream = streams.get(choice["tile"]["source"]["id"])
            if stream is not None and not stream.get("available"):     # no stream sharp enough (None: refused for now)
                self.window_notes.append({"source": f"bilibili:{choice['tile']['source']['id']}", "t": choice["moment"],
                                          "status": "lowres", "note": "清晰度不够"})
        return [job for job in planned if (streams.get(job[2]["tile"]["source"]["id"]) or {}).get("available")]

    # ------------------------------------------------------------ pictures
    IMAGE_PICK = """每一行是一句口播（行首黄色编号）和几张网上找来的候选图片（行内从左到右是第1、2、3、4张）。
为每句挑一张放在这句话下面的图片。这是没有视频时的补位画面：和口播讲的人、事、地方、话题大致相关就可以选，画面干净清楚最重要。
口播讲到某个具名人物时，优先选这个人本人的照片（新闻照、会议照、肖像都可以），不要选明显是别人的照片。
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
            # A line about a named person: photos of that person first (an editor's answer when there is little
            # footage of someone; a film about two named people borrowed one clip 23 times instead).
            names = [p["name"] for p in shot.get("persons") or [] if p.get("name")]
            words = [f"{names[0]} 照片", names[0]] if names else []
            for q in (groups.get(shot["id"]) or {}).get("zh", [])[:2] + [shot.get("visual") or ""]:
                q = " ".join(self.IMAGE_WORDS.sub(" ", q).split())
                if q and q not in words:
                    words.append(q)
            return words[:3 if names else 2]
        seen_hashes = []
        lock = threading.Lock()
        # A time budget: the client's machines spent up to four and a half minutes here for a few pictures.
        deadline = time.monotonic() + PICTURE_SECONDS
        def candidates(shot):
            rows = []
            for q in queries(shot):
                for search in (sources.bing_images, publisher_videos.chinanews_images):
                    try:
                        rows += search(q)[:12]
                    except Exception:   # noqa: BLE001 — one search less
                        continue
            pictures = []
            for row in rows:
                if len(pictures) >= 3 or time.monotonic() > deadline:
                    break
                url = row.get("url") or ""
                if not url or self.IMAGE_SKIP_HOSTS.search(url + " " + str(row.get("source_url") or row.get("source") or "")):
                    continue
                self.check()
                try:
                    with network.client(timeout=6, follow_redirects=True) as client:
                        response = client.get(url, headers={"User-Agent": BROWSER})
                    if response.status_code != 200 or len(response.content) > 12 * 2**20:
                        continue
                    with Image.open(io.BytesIO(response.content)) as raw:
                        image = ImageOps.exif_transpose(raw).convert("RGB")
                except Exception:   # noqa: BLE001 — broken or refused picture
                    continue
                if min(image.size) < 360 or not 0.45 <= image.width / image.height <= 2.6:
                    continue
                if self.project.get("aspect", "16:9") != "9:16" and image.width < image.height * 1.15:
                    continue    # a tall picture in a landscape film would sit between blurred side bands
                h = _dhash(image)
                with lock:
                    if any(bin(h ^ other).count("1") <= 6 for other in seen_hashes):
                        continue
                    seen_hashes.append(h)
                if screen_text.installed():
                    try:
                        lines = screen_text.read(image)
                        found = screen_text.classify([lines])
                    except Exception:   # noqa: BLE001
                        lines, found = [], {"contact": [], "dense": False, "moving": False}
                    big = any(l["score"] >= 0.7 and screen_text._cjk(l["text"]) >= 1 and l["box"][3] - l["box"][1] >= 0.08
                              for l in lines)
                    if found["contact"] or found["dense"] or big:
                        continue    # a poster, a screenshot, an ad, or a stock picture with big words on it ("房贷")
                image.thumbnail((1920, 1920))
                pictures.append((row, image, h))
            return shot, pictures
        found = {}
        ordered = sorted(shots, key=lambda shot: not shot.get("persons"))   # person lines first
        with execution.ThreadPoolExecutor(max_workers=8) as pool:
            for shot, pictures in pool.map(candidates, ordered[:PICTURE_LINES]):
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
                    # Hard limits (scores alone let one long stock take fill ten shots in a row): a source at most
                    # SOURCE_MAX windows a film, SCENE_GAP seconds apart, never in neighbouring shots, never the same
                    # picture again. A news report's windows show different scenes: it may fill more lines.
                    news = lane == "news"
                    if (used >= (NEWS_CAP if news else min(SOURCE_MAX, hard_cap)) or twin or (adjacent and not news)
                            or near < (SCENE_SECONDS if news else SCENE_GAP)):
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
        # Two at a time: with three films at once B站 counts every call from this computer together.
        with execution.ThreadPoolExecutor(max_workers=2) as pool:
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
            with self.lock:
                self.fill_errors[job_id] = str(exc)[:160]
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
            # Text across the picture (花字, a title board, an app screen): the client's biggest complaint.
            # Used only when neither another clip, a related picture nor a borrowed clip is left.
            with self.lock:
                self.soft.setdefault(job_id, set()).update(("wm", "center"))
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
        if clean.get("narrow"):
            flags.add("narrow")
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
            if not role:
                choice = self.job_choice.get(job_id) or {}
                lane = choice.get("lane")
                if lane == "library":
                    # A new library video found by this line's own picture words: related to the line or the story
                    # is enough, but a line about people shows people (the client: no filler 空镜).
                    role = (f"｜补位画面（按这句的画面词“{choice.get('words', '')}”从素材库找的）：画面和这句口播或整篇讲的人和事"
                            "大致相关就合格，细节、人物身份不必一致；口播讲人和人之间的事，画面要有相应的人，风景空镜判跑题；"
                            "并按画面是否干净（主持人、文字页、花字、水印、表情包、黑屏、模糊）判断")
                elif lane in ("topic", "near", "neighbor", "repeat"):
                    # Filler for a line without footage of its own: a fresh clean picture beats the same clip
                    # again (the client: "less on topic is fine, but clean and not repeated").
                    # The client (10-28): no filler 空镜 — a line about people shows people, scenery only for a place.
                    role = ("｜补位画面：和整篇讲的人和事大致相关就合格，不要因为和这句口播细节对不上而判跑题；"
                            "但口播讲人、讲人的处境时，风景、空镜、航拍、田野山水判跑题；"
                            "并按画面是否干净（主持人、文字页、花字、水印、表情包、黑屏、模糊）判断")
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
            self.soft.setdefault(job_id, set()).add("center")
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
    # ------------------------------------------------------------- final review
    FINAL_SYSTEM = """你在替客户做营销号视频的成片终审：成片要直接发布，不能有任何一个镜头让观众觉得不对劲。
图片每一行是成片里一个镜头的3张截图（左→右，就是观众看到的样子），行首黄色编号是镜头号；文字里有每个镜头的口播。
只输出 JSON：{"bad":{"S3":"原因"}}，没有就 {"bad":{}}，原因不超过15字。
只列有下面这些硬伤、发布出去会出事或一看就是劣质素材的镜头（画面和口播对得不够准不算，不要列）：
- 不得体：性暗示或暴露（内衣、泳装、裸露身体、亲热接吻、床戏、擦边），血腥暴力（殴打、拉扯、流血）；
- 口播讲中国的人和事，画面明显是外国人、外国街景或外国影视剧；
- 表情包式的夸张反应镜头（对着镜头狂哭狂笑、做鬼脸）、卡通动画、游戏画面、AI 图（口播在讲这些时除外）；
- 主持人或博主对着镜头讲话、PPT 文字页、手机电脑屏幕截图；
- 画面上能看到水印、台标、平台名或素材站名（如“光厂素材”）、账号名、花字、大段字幕；竖屏、黑边、画中画、严重模糊、黑屏。
标为“要求人物”的镜头：画面是这个人本人的真实影像就合格。场景里本来就有的招牌、商品包装、钞票上的字不算。
影视剧、短剧里演员表演口播讲的情节是合格的；细节、人物身份和口播不完全一致也合格，只要大致是这类人、这类事。"""
    FINAL_ROWS = 10

    def final_review(self, final, files) -> dict:
        """The film as the viewer sees it, every shot against the client's bar (publish without touching it): three
        stills of each shot as rendered, ten shots a sheet, one cheap look each. {sid: reason} of shots to replace;
        a named person's own footage and news reports are judged but kept (no stand-in would be better)."""
        order = [sid for _, sid in sorted((self.position[sid], sid) for sid in final if sid in files and sid in self.position)]
        rows = []
        for sid in order:
            choice = final[sid]
            if choice.get("lane") == "image":
                continue        # a picture with slow motion: its own check chose it
            try:
                frames = self.qa_frames(choice.get("job") or sid, Path(files[sid]), self.shots_by_id_all[sid])
            except Exception:   # noqa: BLE001 — an unreadable clip is left to the audit
                continue
            rows.append((sid, frames))
        domain = self.plan.get("domain") or ""
        def ask(part):
            self.check()
            label_w = 70
            sheet = Image.new("RGB", (label_w + 3 * QA_W, len(part) * QA_H), (0, 0, 0))
            draw, font, ids, lines = ImageDraw.Draw(sheet), _font(30), {}, []
            for r, (sid, frames) in enumerate(part):
                label = f"S{r + 1}"
                ids[label] = sid
                draw.rectangle((0, r * QA_H, label_w - 4, r * QA_H + 44), fill=(255, 214, 0))
                draw.text((6, r * QA_H + 4), label, font=font, fill=(0, 0, 0))
                for c, frame in enumerate(frames[:3]):
                    sheet.paste(frame.resize((QA_W, QA_H)), (label_w + c * QA_W, r * QA_H))
                shot = self.shots_by_id_all[sid]
                names = "、".join(p["name"] for p in shot.get("persons") or [])
                role = f"｜要求人物：{names}" if names and final[sid].get("base_lane", final[sid].get("lane")) in (
                    "pick", "person", "person_near") else ""
                lines.append(f"{label} 口播：{shot['narration']}{role}")
            head = [f"全文主题：{self.project.get('title', '')}"] + ([f"整篇领域：{domain}"] if domain else [])
            content = [{"type": "text", "text": "\n".join(head + lines)}, {"type": "image_url", "image_url": {"url": _jpeg(sheet)}}]
            result, _ = self.chat(self.FINAL_SYSTEM, content, max_tokens=500, task="express_qa", attempts=2)
            with self.lock:
                self.calls["qa"] += 1
            bad = (result or {}).get("bad") if isinstance(result, dict) else {}
            return {ids[k]: str(v)[:40] for k, v in (bad or {}).items() if k in ids}
        parts = [rows[i:i + self.FINAL_ROWS] for i in range(0, len(rows), self.FINAL_ROWS)]
        verdict = {}
        with execution.ThreadPoolExecutor(max_workers=4) as pool:
            for found in pool.map(lambda part: self._safe_dict(ask, part), parts):
                verdict.update(found)
        return verdict

    def _safe_dict(self, function, *args) -> dict:
        try:
            return function(*args)
        except media.Cancelled:
            raise
        except Exception as exc:
            if type(exc).__name__ in ("BudgetExceeded", "ModelUnavailable", "PricingUnavailable"):
                raise
            self.notes.append(f"成片终审失败：{type(exc).__name__}")
            return {}

    # ------------------------------------------------------------- audit
    PRESENTER_REASON = re.compile(r"主持|博主|嘉宾|对着镜头|对镜|访谈|采访|口播|演播")
    CARTOON_REASON = re.compile(r"动画|卡通|游戏|CG|渲染|漫画")
    TEXT_REASON = re.compile(r"PPT|文字页|截图|截屏|界面|图表|聊天|评论区|花字|标题字|大字")
    MARK_REASON = re.compile(r"水印|台标|logo|LOGO|硬字幕|弹幕|贴纸|表情包")
    # Lanes kept first when shots look alike: the line's own footage before fill.
    KEEP_ORDER = ("news", "pick", "person", "person_near", "ctx", "library", "topic", "stock", "image", "neighbor",
                  "near", "repeat", "reuse")

    def window_status(self, job_id, reason=None) -> tuple:
        """What a downloaded window showed, as a footage_index check status (None: nothing learned)."""
        error = self.fill_errors.get(job_id)
        if error:
            if re.search(r"联系方式|广告|滚动字幕", error):
                return "text", error[:40]
            return None, ""          # a network error or a refusal says nothing about the picture
        clean = self.clean.get(job_id)
        if not clean or not screen_text.installed():
            return None, ""          # not looked at (or not with the same tools as the dorm check)
        flags = self.flags(job_id)
        for flag in ("narrow", "still", "dark", "jump"):
            if flag in flags:
                return flag, ""
        if "wm" in flags:
            return ("text" if "center" in flags else "mark"), ""
        reason = str(reason or "")
        for pattern, status in ((self.PRESENTER_REASON, "presenter"), (self.CARTOON_REASON, "cartoon"),
                                (self.TEXT_REASON, "text"), (self.MARK_REASON, "mark")):
            if pattern.search(reason):
                return status, reason[:40]
        if re.search(r"黑屏|黑场|纯色|严重模糊", reason):
            return "dark", reason[:40]
        return ("crop" if clean.get("zoom", 1) > 1.001 or clean.get("residual") else "clean"), ""

    def record_windows(self, qa_bad) -> int:
        """Teach the library what the windows taken from it looked like here, so later films (and the next seed,
        through the dorm PC) skip the dirty moments and take the clean ones first."""
        from . import footage_index
        if not hasattr(footage_index, "record_checks"):
            return 0
        items = list(self.window_notes)
        for job_id, choice in list(self.job_choice.items()):
            source = choice["tile"]["source"]
            if source.get("provider") != "bilibili" or not source.get("indexed"):
                continue
            moment = choice.get("moment")
            if moment is None:
                t = float(choice["tile"]["t"])
                moment = next((float(m) for m in source.get("moments") or [] if abs(float(m) - t) <= 1.5), None)
            if moment is None:
                continue
            status, note = self.window_status(job_id, qa_bad.get(job_id))
            if status:
                items.append({"source": f"bilibili:{source['id']}", "t": moment, "status": status,
                              "crop": (self.clean.get(job_id) or {}).get("crop"), "note": note})
        try:
            return footage_index.record_checks(items) if items else 0
        except Exception as exc:   # noqa: BLE001 — learning is an extra
            self.notes.append(f"素材库体检记录失败：{type(exc).__name__}")
            return 0

    @staticmethod
    def looks_of(path, duration) -> list:
        """dHash of three frames of a clip (start, middle, end)."""
        frames = express_clean.clip_frames(path, duration, (9, 8), count=3)
        return [int("".join("1" if bit else "0" for bit in (f[:, 1:] > f[:, :-1]).flatten()), 2) for f in frames]

    @staticmethod
    def alike(a, b) -> bool:
        return sum(1 for x in a for y in b if bin(x ^ y).count("1") <= SIMILAR_BITS) >= 2

    def audit(self, final, files) -> dict:
        """The film as it will be seen: shots that look alike (one video's other moments, re-uploads of one
        picture, one clip borrowed again), one video used more than SOURCE_MAX times, alike shots side by side,
        and any Bilibili window that was never checked for marks. Returns {sid: reason} to replace; a named
        person's own footage and news reports are left alone."""
        redo = {}
        exempt = {sid for sid, c in final.items()
                  if c.get("base_lane", c.get("lane")) in ("person", "person_near", "news")
                  or (self.shots_by_id_all.get(sid, {}).get("persons") and not self.unfilmed_shot(self.shots_by_id_all[sid]))}
        for sid, choice in final.items():
            job = choice.get("job")
            if (sid in files and choice["tile"]["source"].get("provider") == "bilibili" and express_clean.available()
                    and job and not self.clean.get(job) and job not in self.relooked):
                self.relooked.add(job)
                # The mark check failed or never ran for this window (a vertical, watermarked clip got through
                # that way): look again, and replace it if it still cannot be judged.
                clean = self.analyse(self.shots_by_id_all[sid], choice, files[sid])
                with self.lock:
                    self.clean[job] = clean
                if not clean or self.flags(job) & {"wm", "narrow"}:
                    redo[sid] = "画面没能检查或有水印/竖屏"
        if not express_clean.available():
            return redo
        order = sorted((self.position[sid], sid) for sid in final if sid in files and sid in self.position)
        looks = {}
        def look(item):
            _, sid = item
            path = str(files[sid])
            if path not in self.looked:
                try:
                    self.looked[path] = self.looks_of(files[sid], float(self.shots_by_id_all[sid]["duration"]))
                except Exception:   # noqa: BLE001 — an unreadable clip is judged by its source alone
                    self.looked[path] = []
            return sid, self.looked[path]
        with execution.ThreadPoolExecutor(max_workers=4) as pool:
            for sid, value in pool.map(look, order):
                if value:
                    looks[sid] = value
        rank = {lane: n for n, lane in enumerate(self.KEEP_ORDER)}
        def keep_key(sid):
            choice = final[sid]
            return (sid not in exempt, rank.get(choice.get("lane"), 99), self.position[sid])
        parent = {sid: sid for _, sid in order}
        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def key_of(sid):
            source = final[sid]["tile"]["source"]
            return source.get("provider"), source.get("id")
        sids = [sid for _, sid in order]
        for i, a in enumerate(sids):
            for b in sids[i + 1:]:
                same_clip = ((final[a].get("lane") == "reuse" and final[a].get("repeat_of") == b)
                             or (final[b].get("lane") == "reuse" and final[b].get("repeat_of") == a))
                if same_clip or (a in looks and b in looks and self.alike(looks[a], looks[b])):
                    parent[root(a)] = root(b)
                    if abs(self.position[a] - self.position[b]) == 1 and b not in exempt:
                        redo.setdefault(b, "和上一镜画面几乎一样")
        clusters = {}
        for sid in sids:
            clusters.setdefault(root(sid), []).append(sid)
        for members in clusters.values():
            if len(members) > SOURCE_MAX:
                for sid in sorted(members, key=keep_key)[SOURCE_MAX:]:
                    if sid not in exempt:
                        redo.setdefault(sid, f"全片有 {len(members)} 个镜头画面几乎一样")
        for a, b in zip(sids, sids[1:], strict=False):
            # The same video in neighbouring shots reads as one shot cut in two (S17/S18 of a dorm test film).
            if key_of(a) == key_of(b) and b not in exempt and abs(self.position[a] - self.position[b]) == 1:
                redo.setdefault(b, "和上一镜是同一个视频")
        for sid in sids:
            if final[sid].get("lane") == "reuse" and sid not in exempt:
                redo.setdefault(sid, "同一段画面第二次出现")
        by_source = {}
        for sid in sids:
            by_source.setdefault(key_of(sid), []).append(sid)
        for members in by_source.values():
            counted = [sid for sid in members if sid not in exempt]
            if len(counted) > SOURCE_MAX:
                for sid in sorted(counted, key=keep_key)[SOURCE_MAX:]:
                    redo.setdefault(sid, f"同一个视频用了 {len(counted)} 次")
        self.similar = sorted((len(m) for m in clusters.values() if len(m) > 1), reverse=True)
        return redo

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
                cleaned = express_clean.video_filter({"crop": clean["crop"], "residual": clean["residual"],
                                                      "size": clean.get("size")})
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
                               "library": "快速通道：素材库里描述相符的新视频补位",
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
                    shot["frame_crop"] = {"crop": clean["crop"], "zoom": clean["zoom"], "size": clean.get("size"),
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
                if choice.get("slow", 1.0) > 1.0:
                    shot["playback_slow"] = choice["slow"]   # a borrowed clip shorter than the line, slowed
                else:
                    shot.pop("playback_slow", None)
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
        # Play addresses only for the videos each line is likely to take (its first options): every address costs
        # two B站 API calls, and asking for all of a film's 80-100 candidate videos at once, three films at a time,
        # got the computer refused (10-29). The rest are asked for when a round actually allocates them.
        referenced, frontier = {}, {}
        for rows in options.values():
            for rank, (tile, _) in enumerate(rows):
                if tile["source"]["provider"] == "bilibili":
                    referenced[tile["source"]["id"]] = tile["source"]
                    if rank < FRONTIER:
                        frontier[tile["source"]["id"]] = tile["source"]
        streams = self.resolve(frontier)
        refused = sum(1 for bvid in frontier if not streams.get(bvid))
        if len(frontier) >= 8 and refused >= 0.8 * len(frontier) and self.cfg.get("bili_limit_mode", "wait") == "wait":
            raise BiliUnavailable(f"B站 {len(frontier)} 个视频里有 {refused} 个拿不到播放地址（多半是被限流）")
        landscape = self.project.get("aspect", "16:9") != "9:16"
        def usable(tile):
            source = tile["source"]
            if source["provider"] != "bilibili":
                return self.oriented(source)
            if source["id"] not in streams:
                return True          # not asked yet: the round that allocates it asks (settle_streams)
            stream = streams.get(source["id"]) or {}
            w, h = stream.get("width") or 0, stream.get("height") or 0
            # A portrait re-upload of landscape footage is a small picture between caption bars.
            return bool(stream.get("available")) and (not (w and h) or (w >= h) == landscape)
        def settle_streams(chosen) -> set:
            """Ask for the play addresses of allocated videos not asked yet; the lines whose video turned out
            unusable (no sharp stream, portrait, refused) are returned and its tiles leave every line's options."""
            ask = {c["tile"]["source"]["id"]: c["tile"]["source"] for c in chosen.values()
                   if c["tile"]["source"]["provider"] == "bilibili" and c["tile"]["source"]["id"] not in streams}
            if ask:
                streams.update(self.resolve(ask))
            gone = {sid for sid, c in chosen.items() if not usable(c["tile"])}
            if gone:
                bad = {chosen[sid]["tile"]["source"]["id"] for sid in gone}
                for sid in options:
                    options[sid] = [(t, lane) for t, lane in options[sid]
                                    if t["source"]["provider"] != "bilibili" or t["source"]["id"] not in bad]
            return gone
        def allocate_usable(lines, exclude):
            """allocate(), then the play addresses of what it chose; lines whose video is unusable choose again."""
            chosen = self.allocate(options, reserved, usage, lines, exclude=exclude)
            for _ in range(2):
                gone = settle_streams(chosen)
                if not gone:
                    break
                for sid in gone:
                    release(chosen.pop(sid))
                chosen.update(self.allocate(options, reserved, usage, [s for s in lines if s["id"] in gone], exclude=exclude))
            for sid in [sid for sid, c in chosen.items() if not usable(c["tile"])]:
                release(chosen.pop(sid))
            return chosen
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
                if flags & {"wm", "narrow"}:
                    # Watermark or 花字 left in view, or a vertical / half-black picture: related pictures and
                    # borrowed clean clips come first.
                    reason = MARKED_MIDDLE if "center" in flags else NARROW if "narrow" in flags else MARKED
                    rejected.setdefault(sid, []).append((passed, path, reason))
                    continue
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
            primary = allocate_usable(fresh, tried)
            for sid, choice in primary.items():
                tried[sid].add(key_of(choice))
            videos = self.share()
            backup = allocate_usable([s for s in fresh if self.wants_backup(s, primary.get(s["id"]), videos)], tried)
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
        # Lines still open take a NEW video from the library before any video already in the film shows another
        # moment (the client, 10-28: one drama's girl in pink eight times in a film, "use the library, don't
        # repeat"): first videos whose described frames carry the line's specific or broader words, then its
        # most general word. Related pictures are looked for at the same time (one after the other they added
        # about two minutes a film); only lines the library leaves open take them.
        picture_thread, early_lines, early_pictures = None, [], {}
        if pending:
            early_lines = list(pending)
            def pictures_early():
                try:
                    early_pictures.update(self.image_fill(early_lines))
                except (media.Cancelled, Budget):
                    pass     # the main line stops on its own check
                except Exception as exc:   # noqa: BLE001 — borrowing follows
                    self.notes.append(f"相关图片补位失败：{type(exc).__name__}")
            picture_thread = threading.Thread(target=pictures_early, name="pictures", daemon=True)
            picture_thread.start()
        library_filled = 0
        def fill_from_library(tag, levels, per_line=LIB_FILL_PER_LINE, seconds=None):
            nonlocal carried
            for _, _, choice in self.drop_queued():
                release(choice)
            try:
                jobs = self.library_jobs(pending, final, reserved, streams, levels=levels, tag=tag, per_line=per_line)
            except (media.Cancelled, Budget):
                raise
            except Exception as exc:   # noqa: BLE001 — the next fills still follow
                self.notes.append(f"素材库补位失败：{type(exc).__name__}")
                return 0
            if not jobs:
                return 0
            self.job_choice.update({job_id: c for job_id, _, c in jobs})
            limit = seconds if seconds is not None else min(LIBRARY_SECONDS, (60 + 0.3 * len(jobs)) * self.share())
            got, failed, bad, running = self.download(jobs, streams, qa=checking, deadline=time.monotonic() + limit)
            failures.update(failed)
            qa_bad.update(bad)
            carried = settle(carried, got, bad, running, last=True)   # late clips of the main rounds win
            filled = 0
            for job_id, sid, choice in sorted(jobs, key=lambda job: job[2]["rank"]):
                if sid in final or job_id not in got or job_id in bad:
                    continue
                if self.flags(job_id) & {"wm", "narrow", "still", "dark", "jump"}:
                    continue
                final[sid] = choice | {"qa": "pass" if checking and job_id not in self.qa_unchecked else "unchecked",
                                       "job": job_id}
                files[sid] = got[job_id]
                self.place(sid, choice, key_of(choice)[0])
                filled += 1
            return filled
        if pending:
            began = time.monotonic()
            library_filled += fill_from_library("a", "all", per_line=LIB_FILL_PER_LINE + 1)
            pending = [s for s in pending if s["id"] not in final]
            stamp("library", began)
        if library_filled:
            self.notes.append(f"{library_filled} 镜用素材库里描述相符的新视频补上，没有重复已经用过的画面")
        if pending:
            # Another moment of a video already in the film, only where the library had nothing new: one video at
            # most SOURCE_MAX windows a film, SCENE_GAP seconds apart, never next to its other window (a named
            # person's own footage may show more of the same documentary).
            began = time.monotonic()
            for _, _, choice in self.drop_queued():
                release(choice)
            carried = [job for job in carried if job[0] in {j[0] for j in self.inflight.values()}]
            group_of_id = {sid: g["id"] for g in groups for sid in g["shots"]}
            donors = sorted(final.items(), key=lambda item: self.position[item[0]])
            shots_from, places = {}, {}
            for donor_id, choice in final.items():
                shots_from[key_of(choice)[0]] = shots_from.get(key_of(choice)[0], 0) + 1
                places.setdefault(key_of(choice)[0], []).append(self.position.get(donor_id, -9))
            jobs = []
            for shot in pending:
                sid = shot["id"]
                person = bool(shot.get("persons")) and not self.unfilmed_shot(shot)
                ranked = sorted(donors, key=lambda item: (group_of_id.get(item[0]) != group_of_id.get(sid),
                                                          abs(self.position[item[0]] - self.position[sid])))
                for donor_id, donor in ranked:
                    # A named-person shot may only reuse footage verified for that name or people-free context.
                    if person and not (self.names_in_source(shot, donor["tile"]["source"])
                                       or donor["lane"] in CONTEXT_LANES):
                        continue
                    key = key_of(donor)[0]
                    if not person and (shots_from.get(key, 0) >= SOURCE_MAX
                                       or any(abs(p - self.position[sid]) < 2 for p in places.get(key, []))):
                        continue    # that video is in the film enough already, or right next to this line
                    if donor_id not in self.position or donor.get("lane") in ("reuse", "card"):
                        continue
                    if FUNERAL.search(donor["tile"]["source"].get("title", "")) and not DEATH_LINE.search(shot["narration"]):
                        continue
                    source = donor["tile"]["source"]
                    length = float(source.get("duration") or 0)
                    need = float(shot["duration"]) + MARGIN
                    if length < need + 0.1:
                        continue
                    gap = SCENE_SECONDS if person else SCENE_GAP
                    taken = reserved.get(key, [])
                    # The free stretch furthest from the windows already shown: another scene, not the same one.
                    spread_out = sorted((x * 1.0 for x in range(0, int(max(0.0, length - need - 0.1)) + 1)),
                                        key=lambda x: -min([abs(x - a) for a, _ in taken] or [1e9]))
                    start = next((c for c in spread_out if all(abs(c - a) >= gap and (c + need + 0.2 <= a or c >= b + 0.2)
                                                               for a, b in taken)), None)
                    if start is None:
                        continue
                    reserved.setdefault(key, []).append((start, start + need))
                    jobs.append((f"{sid}#r", sid, donor | {"start": round(start, 2), "lane": "repeat",
                                                           "base_lane": donor.get("base_lane", donor["lane"]),
                                                           "repeat_of": donor_id}))
                    shots_from[key] = shots_from.get(key, 0) + 1
                    places.setdefault(key, []).append(self.position[sid])
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
                if job_id in got and job_id not in bad and self.flags(job_id) & {"wm", "narrow"}:
                    flags = self.flags(job_id)
                    bad[job_id] = MARKED_MIDDLE if "center" in flags else NARROW if "narrow" in flags else MARKED
                if job_id in got and job_id in bad:
                    rejected.setdefault(sid, []).append((choice | {"job": job_id}, got[job_id], bad[job_id]))
                elif job_id in got:
                    final[sid] = choice | {"qa": "pass" if checking and job_id not in self.qa_unchecked else "unchecked",
                                           "job": job_id}
                    files[sid] = got[job_id]
            pending = [s for s in pending if s["id"] not in final]
            stamp("repeat", began)
        failures.update(self.stop_downloads())
        def keepable(shot, choice, reason, marks_ok=False):
            # Clearly wrong footage is worse than a neutral card: badly off-topic anywhere, and for an
            # ancient-history video modern, foreign or other-dynasty scenes.
            if re.search(r"严重不符|完全无关|完全不符|明显无关|毫无关系|毫无关联", reason):
                return False
            if not marks_ok and (reason in (MARKED, MARKED_MIDDLE, NARROW) or MARK_REJECTION.search(reason)
                                 or self.flags(choice.get("job")) & {"wm", "narrow"}):
                return False    # watermarks, 花字, vertical pictures: after pictures and borrowing (below)
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
            open_ids = {s["id"] for s in cards}
            early_ids = {s["id"] for s in early_lines}
            pictured = {}
            if picture_thread is not None:
                picture_thread.join()
                self.check()
                pictured = {sid: v for sid, v in early_pictures.items() if sid in open_ids}
            later = [s for s in cards if s["id"] not in early_ids]
            if later:
                try:
                    pictured.update(self.image_fill(later))
                except (media.Cancelled, Budget):
                    raise
                except Exception as exc:   # noqa: BLE001 — borrowing still follows
                    self.notes.append(f"相关图片补位失败：{type(exc).__name__}")
            for sid, (choice, path) in pictured.items():
                final[sid], files[sid] = choice, path
            cards = [s for s in cards if s["id"] not in pictured]
            if pictured:
                self.notes.append(f"{len(pictured)} 镜没有合适的视频，用了相关图片（慢慢推拉的动态画面）")
            stamp("pictures", began)
        footage = sorted((self.position[sid], path) for sid, path in files.items() if sid in self.position)
        # A real clip already accepted elsewhere (never the neighbouring one) beats a blurred card, but each clip
        # is lent out at most twice: one clip all over a film is what viewers notice first.
        if cards:
            # Still open after the pictures: any fresh library video about the film's subject (its topic words) before
            # a clip already in the film is shown again.
            self.abandon.clear()
            began = time.monotonic()
            pending = list(cards)
            filled = fill_from_library("d", "topic", seconds=TOPIC_SECONDS)
            failures.update(self.stop_downloads())
            if filled:
                self.notes.append(f"{filled} 镜用素材库里同一主题的新视频补上")
            cards = [s for s in cards if s["id"] not in final]
            stamp("library_d", began)
        reused = 0
        for shot in list(cards):
            # The same clip a second time (never next to itself), from a video the film shows only once so far.
            donor = self.borrow(shot, final, files, limit=SOURCE_MAX)
            if donor:
                final[shot["id"]] = final[donor] | {"lane": "reuse", "qa": "reused", "repeat_of": donor}
                files[shot["id"]] = files[donor]
                cards.remove(shot)
                reused += 1
        if reused:
            self.notes.append(f"{reused} 镜没有新的可用画面，借用了附近镜头已通过检查的画面")
        marked = 0
        # The client (10-28): "can't they just never be used?" — by default a clip with a watermark, 花字 or a
        # vertical picture is never used; the setting lets them in as a last resort before a borrowed clip.
        for shot in list(cards) if self.cfg.get("marked_clips", "never") == "last_resort" else []:
            # Nothing else: a clip with a watermark (edge marks first, then text across the middle) beats a card.
            sid = shot["id"]
            for wanted in (lambda r: r == MARKED, lambda r: r == NARROW,
                           lambda r: r == MARKED_MIDDLE or bool(MARK_REJECTION.search(r))):
                kept = next(((c, path, reason) for c, path, reason in rejected.get(sid, [])
                             if wanted(reason) and keepable(shot, c, "画面合格", marks_ok=True)), None)
                if kept:
                    choice, path, reason = kept
                    final[sid] = choice | {"qa": "rejected_kept:" + reason}
                    files[sid] = path
                    cards.remove(shot)
                    marked += 1
                    break
        if marked:
            self.notes.append(f"{marked} 镜实在没有别的画面，用了带水印、文字或竖屏的片段")
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
        reused_more = 0
        for limit in (SOURCE_MAX + 1, None):    # spread over many clips: never one clip all over the film
            for shot in list(cards):
                # Still nothing: the client reads a blurred card as a broken video (24 of them in one film), so
                # a clip lent out more often comes first, the least-lent ones before the rest.
                donor = self.borrow(shot, final, files, limit=limit)
                if donor:
                    final[shot["id"]] = final[donor] | {"lane": "reuse", "qa": "reused", "repeat_of": donor}
                    files[shot["id"]] = files[donor]
                    cards.remove(shot)
                    reused_more += 1
        if reused_more:
            self.notes.append(f"{reused_more} 镜实在没有别的画面，又借用了已经用过的画面")
        # No title cards at all (the client, 10-28): the last lines take any clean clip of this film.
        last = 0
        for shot in list(cards):
            found = self.any_clean(shot, final, files)
            if found:
                donor, slow = found
                final[shot["id"]] = final[donor] | {"lane": "reuse", "qa": "reused", "repeat_of": donor, "slow": slow}
                files[shot["id"]] = files[donor]
                cards.remove(shot)
                last += 1
        if last:
            self.notes.append(f"{last} 镜实在没有别的画面，用了片子里别处的干净画面（不出文字卡）")
        if cards:
            footage_lines = sum(1 for c in final.values() if c.get("lane") not in ("reuse", "card"))
            if footage_lines < 0.3 * len(self.shots):
                # Hardly any footage at all: Bilibili is refusing downloads for now; made again a few minutes later.
                raise FootageStarved(f"只拿到 {footage_lines} 段可用画面（共 {len(self.shots)} 个镜头），稍后重新找一次")
            raise ValueError(f"有 {len(cards)} 个镜头找不到任何可用画面，没有用文字卡凑；请稍后重试")
        began = time.monotonic()
        self.progress(88, "检查全片有没有重复画面")
        redo = self.audit(final, files)
        replaced = 0
        if redo:
            # Each repeat or unchecked window gets one more go at a fresh library video; where none passes, the
            # shot keeps what it had (better than a gap).
            kept = {sid: (final.pop(sid), files.pop(sid, None)) for sid in redo}
            self.abandon.clear()
            pending = [s for s in self.shots if s["id"] in redo]
            # One round with a short limit: the film is otherwise done, the replacements must not hold it up.
            replaced += fill_from_library("x", "all", per_line=LIB_FILL_PER_LINE + 1, seconds=AUDIT_SECONDS)
            pending = [s for s in pending if s["id"] not in final]
            failures.update(self.stop_downloads())
            for sid, (choice, path) in kept.items():
                if sid not in final:
                    final[sid], files[sid] = choice, path
            if replaced:
                self.notes.append(f"全片检查：{replaced} 个重复或没检查过的镜头换成了素材库里的新视频")
            if len(redo) > replaced:
                self.notes.append(f"全片检查：还有 {len(redo) - replaced} 个镜头没找到更好的画面（"
                                  + "；".join(sorted(set(redo.values())))[:80] + "）")
            left = len(redo) - replaced
            refused = self.bili.blocked > 0 or any("风控" in n for n in self.notes)
            tries = int(self.project.get("repeat_retries") or 0)
            if REPEAT_RETRIES and refused and left >= max(3, 0.08 * len(self.shots)) and tries < REPEAT_RETRIES:
                store.mutate(self.pid, lambda p: p.update(repeat_retries=tries + 1), internal=True)
                raise TooRepetitive(f"B站限流中，{left} 个镜头只能用重复画面，稍后重新找一次")
            self.similar = []
            self.audit(final, files)     # the sizes reported are those of the film as saved
        stamp("audit", began)
        began = time.monotonic()
        final_bad, final_fixed = {}, 0
        if checking:
            # 成片终审: the film as the viewer sees it, against the client's bar ("publish it without looking").
            self.progress(90, "成片终审：逐镜检查")
            final_bad = self.final_review(final, files)
            exempt = {sid for sid, c in final.items() if c.get("base_lane", c.get("lane")) in ("person", "person_near", "news")
                      or (self.shots_by_id_all.get(sid, {}).get("persons") and not self.unfilmed_shot(self.shots_by_id_all[sid]))}
            redo_final = {sid: why for sid, why in final_bad.items() if sid not in exempt}
            if redo_final:
                kept = {sid: (final.pop(sid), files.pop(sid, None)) for sid in redo_final}
                self.abandon.clear()
                pending = [s for s in self.shots if s["id"] in redo_final]
                final_fixed = fill_from_library("f", "all", per_line=LIB_FILL_PER_LINE + 1, seconds=AUDIT_SECONDS)
                failures.update(self.stop_downloads())
                for sid, (choice, path) in kept.items():
                    if sid not in final:
                        final[sid], files[sid] = choice, path
                self.notes.append(f"成片终审：{len(final_bad)} 个镜头不合格（" + "；".join(sorted(set(final_bad.values())))[:80]
                                  + f"），{final_fixed} 个换成了素材库里的新视频")
        stamp("final_review", began)
        self.record_windows(qa_bad)
        self.progress(92, "保存选片")
        self.save(final, files)
        unmatched = [s["id"] for s in self.shots if s["id"] not in final]
        report = {"selected": len(final), "unmatched": unmatched, "elapsed_seconds": round(time.monotonic() - self.started, 2),
                  "timing": self.timing, "model_calls": self.calls, "qa_rejected": qa_bad, "download_failures": failures,
                  "lanes": {lane: sum(1 for c in final.values() if c["lane"] == lane)
                            for lane in ("news", "pick", "person", "person_near", "ctx", "topic", "neighbor", "near", "stock", "image",
                                         "library", "repeat", "reuse", "card")},
                  "qa_kept_rejected": [sid for sid, c in final.items() if str(c.get("qa", "")).startswith("rejected")],
                  "flags": {"still": sum(1 for c in final.values() if "still" in self.flags(c.get("job"))),
                            "wm": sum(1 for c in final.values() if "wm" in self.flags(c.get("job"))),
                            "cropped": sum(1 for c in final.values()
                                           if (self.clean.get(c.get("job")) or {}).get("zoom", 1) > 1.001)},
                  "sources": len({key_of(c)[0] for c in final.values()}),
                  "max_same_source": max([sum(1 for c in final.values() if key_of(c)[0] == k)
                                          for k in {key_of(c)[0] for c in final.values()}] or [0]),
                  "alike_groups": list(self.similar), "audit_replaced": replaced, "audit_found": len(redo),
                  "final_bad": final_bad, "final_fixed": final_fixed,
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
