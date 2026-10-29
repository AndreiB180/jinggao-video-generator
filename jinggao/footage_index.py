"""本地素材索引：B站视频的标题、标签、预览帧和画面描述，不下载视频本身。

Each indexed video keeps its search row (title, tags, length, uploader) and 8-16 "moments": preview
frames from Bilibili's own thumbnail sprites, each with a one-line description written by the vision
model from a contact sheet (about 0.00003 元 a frame). Text is searched with Chinese character
bigrams in SQLite FTS5, so a shot's query finds the video and the moment in it that shows the thing.
Production downloads only the chosen few seconds, as before. Shots look the library up with find(): a few
short words a camera sees, all of them on one filmed frame, widened through the describer's own vocabulary
(expand); the "checks" table holds what the clean-up check (footage_check) found at library moments.

The index grows two ways: bulk builds from topic packs or keywords (build()), and every production run's
live search results (add_seen()). The described part ships with the app as gzip JSON lines
(assets/footage-index, export_seed/install_seed).

Three processes use the file at once (the server, the production worker, the index-building worker):
every write is its own short BEGIN IMMEDIATE transaction, reads need no lock (WAL), and nothing holds
a connection open between calls.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import zlib
from contextlib import contextmanager

from . import costs, execution, paths
from .footage_topics import PACKS   # topic -> Bilibili searches

MOMENTS = 10            # preview frames kept (and described) per video when the length is unknown
SHEET = (6, 4)          # describe sheet: columns x rows of frames
SEARCH_WORKERS = 3      # Bilibili's search API answers bursts with 412 risk control
SEARCH_GAP = 0.4        # seconds between one search's result pages
SHOT_WORKERS = 8        # videoshot metadata (same API gateway as search) and its CDN sprites
DESCRIBE_WORKERS = 10   # the background worker allows 10 model calls at a time (execution.MODEL_SLOTS)
MAX_TRIES = 2           # a video whose sheets failed this often is left out of later builds
SKIP_TITLE = re.compile(r"直播|回放|教程|PPT|课件|歌曲|MV|翻唱|游戏|手游|原神|王者荣耀|和平精英|鬼畜|reaction|开箱|测评|"
                        r"带货|广告|招聘|网课|考研|公务员|英语|单词|倍速|加速版|快进|变速|表情包|梗图|斗图")
BAD_KINDS = {"主持人出镜", "图表文字", "黑屏或模糊"}
DESCRIBE_SYSTEM = """图片是视频预览帧的联系表，每格左上角有黄色编号。为每一格写一句画面描述（最多16个字，不要标点结尾）：主体、动作、场景，能看出时代的写上（古代/近代/现代），能认出地名或事物的写上；再给画面类型，只能是：实拍、影视剧、动画、图表文字、主持人出镜、黑屏或模糊。
只输出 JSON：{"1":["描述","类型"],"2":["描述","类型"],...}，每个编号都要有。"""
SEED = paths.ROOT / "assets" / "footage-index"
SEED_PART_BYTES = 40 * 1024 * 1024     # GitHub refuses files over 100 MB and warns over 50 MB
BATCH = 500                            # videos per write transaction when installing or merging

_lock = threading.Lock()               # one writer at a time inside a process (SQLite serialises processes)
_ready, _ready_lock = set(), threading.Lock()
CJK = re.compile(r"[㐀-鿿]+")
WORD = re.compile(r"[A-Za-z0-9]{2,}")
BV = re.compile(r"BV[0-9A-Za-z]{10}")


def grams(text: str) -> list[str]:
    """Searchable terms: two-character pieces of every Chinese run (one character alone), plus words."""
    out = []
    for run in CJK.findall(text or ""):
        out += [run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)]
    out += [w.lower() for w in WORD.findall(text or "")]
    return list(dict.fromkeys(out))


def _file():
    return paths.DATA / "footage-index.sqlite"


def _schema(con):
    paths.wal(con)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            key TEXT PRIMARY KEY, provider TEXT NOT NULL, id TEXT NOT NULL, row TEXT NOT NULL,
            topic TEXT NOT NULL DEFAULT '', added REAL NOT NULL, described INTEGER NOT NULL DEFAULT 0,
            used INTEGER NOT NULL DEFAULT 0, flags TEXT NOT NULL DEFAULT '', tries INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS moments (
            source TEXT NOT NULL, t REAL NOT NULL, hash TEXT NOT NULL DEFAULT '', text TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL DEFAULT '', PRIMARY KEY (source, t));
        CREATE VIRTUAL TABLE IF NOT EXISTS terms USING fts5(source UNINDEXED, t UNINDEXED, words);
        CREATE TABLE IF NOT EXISTS term_rows (source TEXT NOT NULL, rid INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS term_rows_source ON term_rows(source);
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS checks (
            source TEXT NOT NULL, t REAL NOT NULL, status TEXT NOT NULL, crop TEXT NOT NULL DEFAULT '',
            note TEXT NOT NULL DEFAULT '', checked REAL NOT NULL DEFAULT 0, PRIMARY KEY (source, t));
    """)
    con.execute("BEGIN IMMEDIATE")
    try:
        try:   # indexes made before 'tries' existed
            con.execute("ALTER TABLE sources ADD COLUMN tries INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        # term_rows lets a video's terms be deleted by rowid (an UNINDEXED FTS column would scan the table).
        if (not con.execute("SELECT 1 FROM term_rows LIMIT 1").fetchone()
                and con.execute("SELECT 1 FROM terms LIMIT 1").fetchone()):
            con.execute("INSERT INTO term_rows (source, rid) SELECT source, rowid FROM terms")
        con.commit()
    except BaseException:
        con.rollback()
        raise


@contextmanager
def _conn(write: bool = False, timeout: float = 30):
    """A short-lived connection: commits on success, rolls back on error, always closes."""
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=timeout, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        with _ready_lock:
            if str(path) not in _ready:
                _schema(con)
                _ready.add(str(path))
        if write:
            con.execute("BEGIN IMMEDIATE")     # the guard in _add and its writes see the same state
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def _key(row) -> str:
    return f"{row['provider']}:{row['id']}"


def known(keys) -> set:
    keys = list(keys)
    found = set()
    with _conn() as con:
        for start in range(0, len(keys), 500):
            part = keys[start:start + 500]
            found |= {r["key"] for r in con.execute(
                f"SELECT key FROM sources WHERE key IN ({','.join('?' * len(part))})", part)}
    return found


def add(row: dict, moments: list[dict], *, topic: str = "", described: bool = False) -> None:
    """Save (or refresh) one video: its search row and moments [{t, hash, text, kind}]."""
    with _lock, _conn(write=True) as con:
        _add(con, row, moments, topic=topic, described=described)


def _add(con, row, moments, *, topic="", described=False) -> bool:
    """One video, inside the caller's write transaction; False when a described copy is already there."""
    key = _key(row)
    old = con.execute("SELECT described, used, topic, tries FROM sources WHERE key=?", (key,)).fetchone()
    if old and old["described"] and not described:
        return False      # never replace described moments with bare ones
    keep = {k: row.get(k) for k in ("provider", "id", "url", "title", "tags", "description", "duration", "author",
                                    "typename", "play", "pubdate", "query")}
    usable = [m for m in moments if m.get("kind") not in BAD_KINDS]
    hosted = sum(1 for m in moments if m.get("kind") == "主持人出镜")
    flags = "host" if moments and hosted * 2 >= len(moments) else ""
    con.execute("INSERT OR REPLACE INTO sources (key,provider,id,row,topic,added,described,used,flags,tries) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (key, row["provider"], row["id"], json.dumps(keep, ensure_ascii=False),
                 topic or (old["topic"] if old else ""), time.time(), int(described), old["used"] if old else 0, flags,
                 0 if described or not old else old["tries"]))
    if old:
        con.execute("DELETE FROM moments WHERE source=?", (key,))
        con.execute("DELETE FROM terms WHERE rowid IN (SELECT rid FROM term_rows WHERE source=?)", (key,))
        con.execute("DELETE FROM term_rows WHERE source=?", (key,))
    con.executemany("INSERT OR REPLACE INTO moments (source,t,hash,text,kind) VALUES (?,?,?,?,?)",
                    [(key, round(float(m["t"]), 2), str(m.get("hash", "")), str(m.get("text") or ""),
                      str(m.get("kind") or "")) for m in moments])
    head = " ".join(grams(" ".join(str(row.get(k) or "") for k in ("title", "tags", "query", "typename"))))
    rows = [(-1.0, head)] + [(round(float(m["t"]), 2), " ".join(grams(m["text"]))) for m in usable if m.get("text")]
    rids = []
    for t, words in rows:
        rids.append((key, con.execute("INSERT INTO terms (source,t,words) VALUES (?,?,?)", (key, t, words)).lastrowid))
    con.executemany("INSERT INTO term_rows (source, rid) VALUES (?,?)", rids)
    return True


def add_seen(rows: list[dict], tiles: list[dict], topic: str = "") -> int:
    """Production found these videos live: keep the new ones (title text and frame times; described in a
    later build). One short transaction; a busy index is skipped rather than holding up the film."""
    by_source = {}
    for tile in tiles:
        source = tile["source"]
        if source.get("provider") != "bilibili" or source.get("indexed"):
            continue
        by_source.setdefault(_key(source), []).append({"t": tile["t"], "hash": format(tile.get("hash") or 0, "x")})
    wanted = {}
    for row in rows:
        if row.get("provider") == "bilibili" and not row.get("indexed") and _key(row) in by_source:
            wanted.setdefault(_key(row), row)
    if not wanted:
        return 0
    have = known(wanted)
    new = 0
    with _lock, _conn(write=True, timeout=3) as con:
        for key, row in wanted.items():
            if key not in have and _add(con, row, by_source[key], topic=topic):
                new += 1
    return new


def search(query: str, limit: int = 8, *, exclude: set | None = None) -> list[dict]:
    """Videos whose title or described frames match the query, best first. Each row is shaped like a
    live Bilibili search row plus "indexed", "moments" (matching frame times), "avoid" (frames showing a
    presenter, a text page or black) and "index_score"."""
    words = grams(query)[:60]
    if not words:
        return []
    expression = " OR ".join('"' + w.replace('"', "") + '"' for w in words)
    with _conn() as con:
        try:
            hits = con.execute("SELECT source, t, bm25(terms) AS rank FROM terms WHERE terms MATCH ? "
                               "ORDER BY rank LIMIT 600", (expression,)).fetchall()
        except sqlite3.OperationalError:
            return []
        by = {}
        for hit in hits:
            entry = by.setdefault(hit["source"], {"title": 0.0, "moments": []})
            if float(hit["t"]) < 0:
                entry["title"] = -float(hit["rank"])
            else:
                entry["moments"].append((-float(hit["rank"]), float(hit["t"])))
        scored = []
        for key, entry in by.items():
            if exclude and key in exclude:
                continue
            best = sorted(entry["moments"], reverse=True)
            score = entry["title"] * 0.6 + (best[0][0] if best else 0) + 0.3 * min(3, len(best))
            scored.append((score, key, [t for _, t in best[:6]]))
        scored.sort(reverse=True)
        out = []
        for score, key, moments in scored[:limit * 3]:
            src = con.execute("SELECT row, flags, used FROM sources WHERE key=?", (key,)).fetchone()
            if not src or src["flags"] == "host":
                continue
            row = json.loads(src["row"])
            avoid = [r["t"] for r in con.execute(
                f"SELECT t FROM moments WHERE source=? AND kind IN ({','.join('?' * len(BAD_KINDS))})",
                (key, *sorted(BAD_KINDS)))]
            out.append(row | {"indexed": True, "moments": moments, "avoid": avoid, "index_score": round(score, 3),
                              "index_used": src["used"]})
            if len(out) >= limit:
                break
        return out


def frame_kinds(row: dict) -> list[tuple[float, str]]:
    """[(time, kind)] of a video's described frames (实拍, 影视剧, 动画, 主持人出镜, 图表文字, 黑屏或模糊)."""
    try:
        with _conn() as con:
            return [(float(r["t"]), r["kind"]) for r in con.execute(
                "SELECT t, kind FROM moments WHERE source=? AND t>=0 ORDER BY t", (_key(row),))]
    except sqlite3.Error:
        return []


# ------------------------------------------------------------------ short words
# The planner names what a camera would see in a few short words ("中年男子 摔手机"); find() looks for filmed
# frames whose description carries every one of them. search() above OR-s the bigrams of long queries and
# returned loosely related frames (88% of the library windows it chose failed QA on the dorm runs, 10-28).
# The describer (footage_local) writes what it sees in its own words: a father on screen is a 中年男子, a crying
# mother a 中年女性 who is 哭泣, a 16-year-old a 少年 or 男生. This table maps the words shots ask for to the
# words the library uses, checked against the keyword part ("描述｜关键词") of the 18.1万-video seed's filmed
# frames (10-28: 中年男子 2.9k, 中年女性 1.7k, 父亲 75, 摔手机 0, 出租屋 61, 租房 63, 公寓 235 …).
# "a/b/c = x y" makes a, b and c alternatives of one another and adds x and y one way (a 父亲 is usually
# described as a 中年男子; a 中年男子 is not a 父亲). Earlier alternatives are closer and rank higher.
# "A+B" is one alternative that needs both words on the frame (a phone shop the describer calls 手机 + 柜台).
SYNONYMS = """
男孩/少年/男生/小伙子/男学生 = 小男孩 男童 青少年 中学生 高中生 初中生
女孩/少女/女生/女学生/姑娘 = 小女孩 女童 青少年 中学生 高中生 初中生
小男孩/男童 = 男孩 儿童 小孩
小女孩/女童 = 女孩 儿童 小孩
孩子/小孩/儿童/孩童/小朋友 = 小男孩 小女孩 男孩 女孩 幼儿 少年 少女 小学生
婴儿/宝宝/幼儿/婴幼儿/新生儿 = 襁褓 婴儿车
青少年/未成年人/未成年 = 少年 少女 中学生 高中生 男生 女生
学生/学生们 = 校服 男生 女生 同学
中学生/高中生/初中生 = 学生 校服 男生 女生 少年 少女
小学生 = 红领巾 儿童 学生 孩子 小孩
大学生 = 学生 男生 女生 年轻人 青年 年轻男子 年轻女子 年轻女性
年轻人/青年/年轻男女 = 年轻男子 年轻女子 年轻女性 年轻男性 小伙子 男生 女生
男子/男人/男性/男士
女子/女人/女性/女士 = 妇女
中年男子/中年男人/中年男性/大叔/中年大叔 = 中年男
中年女子/中年女性/中年妇女/中年女人/大妈/阿姨 = 中年女
中年人 = 中年男 中年女 中年夫妻
老人/老年人/老年 = 老奶奶 老爷爷 老太太 老大爷 老汉 老妇 白发
老奶奶/奶奶/老太太/老妇人/老太/外婆/姥姥 = 老年女 老妇 老人
老爷爷/爷爷/老大爷/老头/老汉/外公/姥爷 = 老年男 老人
父亲/爸爸/老爸 = 父子 父女 中年男 中年大叔
母亲/妈妈/老妈 = 母子 母女 中年女 中年妇女
父母/爸妈/双亲 = 中年夫妻 夫妻 夫妇 一家 家长 家人
家长 = 父母 母亲 父亲 妈妈 爸爸 母子 父子 母女 父女 中年男 中年女
儿子 = 父子 母子 男孩 少年 小男孩
女儿 = 父女 母女 女孩 少女 小女孩
夫妻/夫妇/两口子/小两口/伴侣 = 中年夫妻 一男一女 男女
丈夫/老公 = 夫妻 夫妇
妻子/老婆 = 夫妻 夫妇
情侣/恋人/小情侣/男女朋友/男朋友/女朋友/男友/女友 = 年轻男女 一男一女 男女
同居 = 情侣 年轻男女 一男一女
一家人/一家三口/一家四口/家人/全家/家庭 = 一家 亲子
兄弟/兄妹/姐弟/姐妹/兄弟姐妹
亲戚/亲友 = 家人
同学/室友 = 学生 男生 女生
老师/教师/班主任/教授 = 讲台 讲课
老板/领导/上司/经理/总裁 = 西装
员工/职员/白领/上班族/打工人 = 职场
医生/大夫 = 白大褂 医护
护士 = 医护
警察/民警/公安/警员/警官 = 特警 交警
外卖员/骑手/外卖小哥 = 外卖
快递员 = 快递
工人/农民工/建筑工人
农民/村民/农户 = 农村 田间 劳作
司机 = 开车 驾驶
保安/门卫
主播/网红 = 直播
哭泣/流泪/大哭/痛哭/落泪/抹泪/哭着/哭了/啼哭/哭喊/哭闹/哭诉 = 眼泪 泪水 擦眼泪
哭 = 哭泣 流泪 大哭 痛哭 落泪 哭着 眼泪 泪水
伤心/难过/悲伤/悲痛/心碎 = 哭泣 流泪 痛苦 失落 沮丧
生气/愤怒/发火/恼火/气愤/暴怒/怒吼 = 怒视 怒斥 咆哮 大吼 吼叫
争吵/吵架/争执/口角/吵闹/争论/拌嘴 = 冲突 对峙 怒吼
叹气/叹息/唉声叹气/无奈 = 扶额 发愁 愁容 苦恼 担忧 焦虑
担心/担忧/焦虑/着急/忧虑/发愁/犯愁/忧心 = 愁容 苦恼 不安 紧张
害怕/恐惧/惊恐/惊吓/恐慌 = 惊慌 慌张
惊讶/震惊/吃惊/惊愕
开心/高兴/快乐/欢笑/喜悦/欢乐 = 大笑 微笑 笑容 欢呼
甜蜜/幸福/恩爱/亲密 = 依偎 拥抱 牵手 亲吻
沮丧/失落/颓废/消沉/郁闷/抑郁/低落/失意 = 垂头丧气 发呆
疲惫/疲倦/劳累/疲劳 = 困倦 打哈欠
崩溃/绝望 = 痛哭 抱头 痛苦
后悔/懊悔/自责 = 捂脸 抱头
孤独/孤单/独自/一个人 = 背影 独坐
迷茫/茫然/发呆/出神/愣住 = 沉思 呆坐
沉默/一言不发 = 沉思 低头
紧张/不安/忐忑 = 焦虑
冷漠/冷淡 = 无视
玩手机/刷手机/看手机/盯着手机/低头看手机 = 拿着手机 手机屏幕
打电话/接电话/通话/讲电话/打手机 = 视频通话
刷视频/刷短视频/刷抖音/短视频 = 看手机 玩手机 手机屏幕
打游戏/玩游戏/打电游/电子游戏/手游/网游/电竞/沉迷游戏/网瘾/玩电脑 = 手机游戏 电脑游戏 游戏机 手柄 网吧 网咖 电脑前
游戏 = 打游戏 玩游戏 手机游戏 电脑游戏 游戏机 手柄 电竞 网吧
摔手机/砸手机/扔手机 = 摔东西 砸东西 摔碎 砸碎 摔杯 摔碗 扔东西 打砸
摔东西/砸东西/扔东西/发脾气/摔门/掀桌子 = 摔碎 砸碎 摔杯 摔碗 打砸
抢手机/夺手机/没收手机/没收/抢夺/夺走/抢走 = 夺过 争夺 拉扯
打架/斗殴/殴打/打人/动手/扭打/厮打 = 打斗 推搡 拉扯
家暴 = 殴打 打人 扭打 推搡 争吵
拥抱/相拥/抱住/搂抱 = 依偎
牵手/拉手/手牵手
亲吻/接吻/亲嘴
睡觉/睡着/入睡/熟睡/午睡/睡 = 躺在床上 躺床上 卧床
躺/躺着/躺在床上/躺床上/躺平 = 躺在 卧床
失眠/熬夜/睡不着 = 深夜 夜里
起床/赖床 = 床边
吃饭/用餐/就餐/进餐/吃 = 聚餐 餐桌 夹菜 吃面
做饭/烹饪/下厨/炒菜 = 厨房 切菜
学习/写作业/做作业/复习/读书/看书/自习 = 书桌 课本 写字
上课/听课/课堂/讲课 = 教室 讲台 黑板
考试/高考/中考/考场/考生/答题 = 试卷
上学/放学/上学路上/去学校 = 书包 背着书包 校门 学校门口
逃学/旷课/辍学/不上学/拒绝上学/休学/退学
上班/工作/打工/办公 = 工位 写字楼
加班 = 深夜+办公 夜晚+办公 深夜+电脑 工位
开会/会议 = 会议室
面试/求职/找工作/应聘 = 招聘 简历 人才市场
失业/辞职/裁员/离职/下岗/辞退/被辞退
搬家/搬东西/搬进/搬出/搬走/搬运 = 搬箱子 纸箱 打包 收拾行李 行李
行李箱/拉杆箱/旅行箱/拖行李箱/拉行李箱/拖着行李箱 = 行李
收拾行李/整理行李/打包行李 = 打包 行李
打扫/做家务/家务/拖地/扫地/清洁/收拾屋子 = 整理
购物/逛街/买东西 = 超市 商场
散步/走路/行走/漫步/步行
跑步/奔跑/跑 = 跑动
开车/驾驶/驾车 = 方向盘 驾驶座
喝酒/醉酒/喝醉/饮酒/酗酒 = 酒杯 啤酒 白酒 干杯
抽烟/吸烟 = 香烟 烟头
看病/就医/就诊/挂号 = 诊室 医院 输液 打针
住院 = 病床 病房
手术 = 手术室
签字/签合同/签约 = 合同
数钱/点钞 = 钞票 现金 人民币 纸币
转账/付款/支付/扫码 = 手机支付 收款
借钱/欠钱/欠债/还钱/借贷/贷款/网贷/催收/债务/借条/欠条
交谈/聊天/说话/对话/谈话/谈心/沟通
安慰/劝说/开导/劝慰 = 拍肩 拥抱
道歉/认错 = 鞠躬
下跪/跪地/跪下/跪 = 跪着
等待/等候
离家出走/离家/出走/离开家 = 背影 行李箱
回家/到家/进门 = 开门 家门口
接孩子/送孩子/接送 = 校门 学校门口 家长
陪伴/陪着/陪孩子
直播/带货
拍照/自拍
家/家里/家中/在家 = 客厅 卧室 居家
客厅 = 沙发 起居室
卧室/房间/屋里/屋内/房内 = 床边
厨房 = 灶台
餐桌/饭桌 = 吃饭
出租屋/出租房/租房/合租/隔断间/租的房子 = 公寓 单身公寓 城中村
宿舍/寝室/学生宿舍/学生公寓 = 上下铺
学校/校园 = 校门 教学楼 操场 教室
校门/学校门口/校门口/学校大门 = 学校门 大学门口
教室/课室 = 课堂 课桌 黑板 讲台
操场/运动场 = 跑道
图书馆/自习室
网吧/网咖 = 电竞馆
医院 = 病房 诊室
派出所/警局/公安局/警察局
法院/法庭 = 庭审 审判 法官
监狱/看守所/牢房 = 铁窗 囚犯
办公室/公司/单位/写字楼/办公楼 = 工位 职场
工厂/车间 = 流水线 生产线
工地/建筑工地 = 施工
街道/马路/街头/路边/街边/大街 = 街景
农村/乡村/村子/村庄/乡下/老家/村里 = 农家 土路 田间
火车站/车站/高铁站/候车室 = 候车 站台
银行 = 取款机 ATM
超市/商场/商店 = 货架 购物
手机店/手机卖场/手机营业厅/数码店/手机专卖店 = 手机+柜台 手机+店铺 手机+货架 手机+店内 手机+门店 修手机 营业厅
大海/海边/海面/海浪/海洋/海景/海岸/海上 = 海滩 沙滩
河边/江边/湖边 = 河流 江面 湖面
山区/大山 = 山村 山路 山脉
夜晚/深夜/晚上/夜里/半夜 = 夜间
雨天/下雨/雨中/大雨 = 雨夜 撑伞 雨水
小区/楼下/居民楼 = 住宅区
楼道/走廊
门口/门外/家门口 = 门前
阳台/窗边/窗前 = 窗户
天台/楼顶/屋顶
车里/车内/车上 = 驾驶座 副驾
公交/公交车/地铁
手机/智能手机 = 苹果手机 iPhone
苹果手机/iPhone/新手机/新款手机/买手机/换手机 = 智能手机 手机+包装 手机+盒 手机+柜台 手机+货架
电脑/笔记本电脑/台式电脑 = 笔记本 显示器
平板/平板电脑/iPad
钱/现金/钞票/人民币/纸币/百元大钞 = 数钱 红包 钱包
红包/压岁钱
银行卡/信用卡 = 刷卡
合同/协议 = 签字 文件
书包/双肩包 = 背包
课本/教材/书本
试卷/成绩单/分数 = 考试
录取通知书/通知书
药/药品/药物/药片/吃药 = 药瓶 药盒
礼物/礼品 = 礼盒
外卖/外卖盒 = 外卖员
快递/包裹 = 快递员 纸箱
汽车/轿车/私家车/小汽车/车 = 车辆
房子/房产/买房/房价/楼盘/新房/商品房 = 售楼处 住宅 楼房
结婚/婚礼/婚纱/新娘/新郎
彩礼/嫁妆 = 现金 红包
离婚/离婚证/结婚证/民政局
怀孕/孕妇/孕肚 = 验孕棒
"""
FILMED = {"实拍", "影视剧"}
CHECK_GOOD = {"clean", "crop"}
CHECK_VIDEO = {"gone", "lowres"}       # said of the whole video, not of one moment
FOREIGN = re.compile(r"外国|老外|欧美|西方|白人|黑人|金发|碧眼|欧洲|美国|英国|法国|德国|意大利|西班牙|俄罗斯|乌克兰|日本|"
                     r"韩国|朝鲜|印度|泰国|越南|菲律宾|印尼|马来西亚|非洲|中东|阿拉伯|伊朗|以色列|土耳其|巴西|墨西哥|"
                     r"澳大利亚|澳洲|加拿大|纽约|伦敦|巴黎|东京|首尔|迪拜|悉尼|洛杉矶|新加坡|日剧|韩剧|美剧|英剧|泰剧|"
                     r"泰恐|留学|也门|叙利亚|阿富汗|伊拉克|巴基斯坦|孟加拉|尼泊尔|缅甸|柬埔寨|老挝|埃及|南非|尼日利亚|"
                     r"肯尼亚|阿根廷|智利|秘鲁|哥伦比亚|古巴|委内瑞拉|波兰|荷兰|瑞士|瑞典|挪威|芬兰|丹麦|希腊|奥地利|"
                     r"新西兰|夏威夷|旧金山|芝加哥|好莱坞")
PEOPLE = re.compile(r"人|男|女|孩|童|婴|宝宝|学生|老师|医生|护士|警察|士兵|工人|夫妻|情侣|父|母|爸|妈|爷|奶|叔|姨|少年|"
                    r"少女|青年|顾客|乘客|游客|观众|球员|演员|歌手|选手|厨师|司机|农民|村民|员工|患者|病人|群众|他|她")
NO_PEOPLE = re.compile(r"空无一人|无人|人民币|人行道|机器人|人造|人工")
# Animals: a line about people searched with 母子 or 一家三口 found gorilla and giraffe mothers with their young.
ANIMAL = re.compile(r"动物|宠物|猩猩|猴子|长颈鹿|大象|狮子|老虎|豹子|猎豹|熊猫|狗熊|黑熊|棕熊|北极熊|狼群|狐狸|鹿|斑马|河马|"
                    r"犀牛|袋鼠|企鹅|海豚|鲸|鲨鱼|猫|狗|犬|兔子|仓鼠|鹦鹉|小鸟|鸟类|飞鸟|母鸡|小鸡|鸭子|鹅|羊群|山羊|绵羊|"
                    r"牛群|奶牛|小牛|马匹|骏马|小马|猪|鱼群|金鱼|昆虫|蝴蝶|蜜蜂|蛇|蜥蜴|乌龟")
# Costume and war dramas: a modern story's 男子 愤怒 should not get an emperor (unless a word asks for one).
PERIOD = re.compile(r"古装|古代|古风|宫廷|皇帝|皇宫|官服|铠甲|盔甲|武侠|仙侠|清朝|明朝|宋朝|唐朝|汉朝|秦朝|三国|民国|"
                    r"二战|抗战|抗日|长袍马褂")
# What the describer saw on top of the picture: subtitles along the bottom go with production's normal crop,
# captions across the picture and split or vertical screens do not.
OVERLAY = ((re.compile(r"分屏|画中画|拼接|竖屏|黑边"), 1.0), (re.compile(r"叠加|文字信息|标题|弹幕|水印|台标|截图|二维码|花字"), 0.5),
           (re.compile(r"字幕"), 0.2))
FIND_HITS = 6000        # frames read from the full-text index per query (newest first: frame-by-frame descriptions)
FIND_POOL = 40          # unpenalised videos scored before the best are returned
FIND_CHUNK = 150        # videos loaded per round
_PUNCT = re.compile(r"[^\w+]|_")


def _synonyms() -> dict:
    table = {}
    for line in SYNONYMS.strip().splitlines():
        left, _, right = line.partition("=")
        same = [w.strip() for w in left.split("/") if w.strip()]
        for word in same:
            alts = table.setdefault(word, [])
            alts += [w for w in same + right.split() if w != word and w not in alts]
    return table


_SYN = _synonyms()


def _searchable(alternative: str) -> bool:
    """Every part has a full-text term of two characters or more (a single Chinese character is a term only
    when it stands alone, so a phrase cannot find it inside a description)."""
    return all(any(len(g) >= 2 for g in grams(part)) for part in alternative.split("+"))


def _pieces(word) -> list[str]:
    return [p for p in (_PUNCT.sub("", piece) for piece in str(word or "").split("/")) if p]


def expand(word: str) -> list[str]:
    """Alternatives for one query word: the word itself plus synonyms the library's describer uses
    ("男孩" -> 男孩/少年/男生/小男孩…; "父亲" -> 父亲/爸爸/中年男子…). A word may already hold alternatives
    separated by "/" ("男孩/少年"). Words shorter than 2 CJK characters only survive through the table."""
    out = []
    for piece in _pieces(word):
        for alternative in [piece] + _SYN.get(piece, []):
            if alternative not in out and _searchable(alternative):
                out.append(alternative)
    return out


def _spec(word) -> dict | None:
    """One query word ready to search: its alternatives with their parts and weights (the caller's own
    words 1.0, table words less and less down the line) and its full-text expression."""
    own = set(_pieces(word))
    alternatives = []
    for n, alternative in enumerate(expand(word)):
        weight = 1.0 if alternative in own else max(0.55, 0.75 - 0.02 * n)
        alternatives.append((alternative, [p.lower() for p in alternative.split("+")], weight))
    if not alternatives:
        return None
    return {"word": str(word), "own": own, "alternatives": alternatives, "fts": _fts(alternatives)}


def _fts(alternatives) -> str:
    """FTS5 expression: any alternative, each with all its terms (not a phrase: grams() keeps a term once
    per frame, so a word whose bigram came earlier in the text would not match as a phrase; the text itself
    is checked afterwards)."""
    ors = []
    for _, parts, _ in alternatives:
        terms = list(dict.fromkeys(g for part in parts for g in grams(part) if len(g) >= 2))
        ors.append("(" + " AND ".join(f'"{g}"' for g in terms) + ")")
    return "(" + " OR ".join(ors) + ")"


def _weight(low: str, keywords: str, spec) -> float:
    """How well a frame's text carries one word: the best alternative found (a little more when it is one
    of the describer's own search words), 0 when none is."""
    best = 0.0
    for _, parts, weight in spec["alternatives"]:
        if weight + 0.1 > best and all(p in low for p in parts):
            best = max(best, weight + (0.1 if all(p in keywords for p in parts) else 0.0))
    return best


def _people(low: str) -> bool:
    return bool(PEOPLE.search(NO_PEOPLE.sub("", low)))


def _status_at(checked: dict, t: float) -> str:
    """The check made at this moment (or within a second of it), "" when none was."""
    if not checked:
        return ""
    status = checked.get(round(t, 2))
    if status is None:
        near = min(checked, key=lambda c: abs(c - t))
        status = checked[near] if abs(near - t) <= 1.0 else ""
    return status


def find(words: list[str], *, limit: int = 6, exclude: set[str] | None = None,
         penalize: dict[str, float] | None = None, foreign_ok: bool = False, people: bool | None = None) -> list[dict]:
    """Videos with a filmed frame whose description carries every word (each word = any of expand(word)),
    best first, one row per video. exclude: source keys ("bilibili:BV…") not to return; penalize: key -> score
    to subtract (videos other recent films used); foreign_ok False skips frames described as 外国/欧美/西方…
    people True prefers frames with people, False frames without.
    Row = the stored search row | {"indexed": True, "moments": [t of matching usable frames, best first],
    "avoid": [t of presenter/text/black frames], "index_score": float, "index_used": int,
    "match": {"t": float, "text": str, "kind": str, "check": "" | status}}

    A frame counts only when it and its described neighbours are filmed (实拍/影视剧): a filmed frame between
    presenter or text frames is a commentary video cutting away for a second. Videos led by a presenter, with
    over 35% presenter/text/black frames, with skipped titles or outside 20 s-3 h are left out, and so are
    frames (and videos: gone, lowres) a clean-up check found unusable; frames checked clean rank first and
    the bad ones join "avoid". With fewer than `limit` full matches, a video whose title row carries one
    missing word may follow (its match gets "title_word"). An unreadable index answers []."""
    words = list(dict.fromkeys(w for word in words or [] for w in str(word).split() if w.strip()))[:4]
    specs = [s for s in (_spec(w) for w in words) if s]
    if not specs:
        return []
    exclude, penalize = set(exclude or ()), penalize or {}
    foreign_ok = foreign_ok or any(FOREIGN.search(p) for s in specs for p in s["own"])
    period = any(PERIOD.search(a[0]) for s in specs for a in s["alternatives"])
    # Words about people never take an animal frame, unless a word asks for an animal ("男孩 遛狗").
    if not any(ANIMAL.search(p) for s in specs for p in s["own"]) and any(
            _people(p) for s in specs for p in s["own"]):
        period = (period, "no_animals")
    wants = (people, period)
    try:
        with _conn() as con:
            found = _gather(con, specs, None, limit, exclude, penalize, foreign_ok, wants)
            if len(found) < limit and len(specs) > 1:
                taken = exclude | {key for _, key, _ in found}
                extra = []
                for i, spec in enumerate(specs):
                    # Titles are loose text: only the word itself counts there (a title's 砸碎 is no 摔手机).
                    own = [a for a in spec["alternatives"] if a[0] in spec["own"]]
                    if own:
                        own = spec | {"alternatives": own, "fts": _fts(own)}
                        extra += _gather(con, specs[:i] + specs[i + 1:], own, limit, taken, penalize, foreign_ok,
                                         wants)
                for item in sorted(extra, key=lambda e: -e[0]):
                    if item[1] not in taken:
                        taken.add(item[1])
                        found.append(item)
    except sqlite3.Error:
        return []
    return [row for _, _, row in found[:limit]]


def _gather(con, specs, title_spec, limit, exclude, penalize, foreign_ok, wants) -> list:
    """[(score, key, row)] best first: videos with a frame matching every spec (and, with title_spec, whose
    title row carries that word too). The newest FIND_HITS matching frames are read, their videos ordered by
    checked-clean frames and matching-frame count, and loaded in rounds until FIND_POOL unpenalised ones
    qualify, so a common word costs no more than a rare one."""
    titles = None
    if title_spec:      # the videos whose title row has the word, first: usually few
        titles = {r[0] for r in con.execute("SELECT source, t FROM terms WHERE terms MATCH ? ORDER BY rowid DESC "
                                            "LIMIT ?", (title_spec["fts"], 5 * FIND_HITS)) if float(r[1]) < 0}
        if not titles:
            return []
    hits, titled = {}, set()
    if titles is not None and len(titles) <= 600:
        # Few: every frame of these videos is checked against the other words (None: any time), rather than
        # hoping the newest matches of a common word ("男子") include them.
        hits = {key: None for key in titles if key not in exclude}
    else:
        for hit in con.execute("SELECT source, t FROM terms WHERE terms MATCH ? ORDER BY rowid DESC LIMIT ?",
                               (" AND ".join(s["fts"] for s in specs), FIND_HITS)):
            key, t = hit[0], float(hit[1])
            if key in exclude or (titles is not None and key not in titles):
                continue
            if t < 0:
                titled.add(key)
            else:
                hits.setdefault(key, set()).add(round(t, 2))
    if not hits:
        return []
    keys = list(hits)
    checks = _checks(con, keys)
    def prior(item):
        n, key = item
        times = hits[key] if hits[key] is not None else checks.get(key, {})
        good = any(_status_at(checks.get(key, {}), t) in CHECK_GOOD for t in times)
        return (-int(good), -min(3, len(hits[key] or ())), n)
    keys = [key for _, key in sorted(enumerate(keys), key=prior)]
    out, clear = [], 0
    for start in range(0, len(keys), FIND_CHUNK):
        part = keys[start:start + FIND_CHUNK]
        marks = ",".join("?" * len(part))
        sources = {r["key"]: r for r in con.execute(
            f"SELECT key, row, flags, used FROM sources WHERE key IN ({marks})", part)}
        moments = {}
        for m in con.execute(
                f"SELECT source, t, kind, text FROM moments WHERE source IN ({marks}) ORDER BY source, t", part):
            moments.setdefault(m["source"], []).append(m)
        for key in part:
            if key not in sources:
                continue
            got = _video(sources[key], moments.get(key, []), checks.get(key, {}), hits[key], specs, title_spec,
                         foreign_ok, wants)
            if not got:
                continue
            score, row = got
            score += (0.15 if key in titled else 0.0) - float(penalize.get(key, 0.0))
            row["index_score"] = round(score, 3)
            out.append((score, key, row))
            clear += key not in penalize
        # A people preference is only known once the text is read: score more videos then.
        if clear >= max(FIND_POOL * (1 if wants[0] is None else 3), 4 * limit):
            break
    out.sort(key=lambda item: -item[0])
    return out


def _video(src, moments, checked, hit, specs, title_spec, foreign_ok, wants):
    """(score, row) for one candidate video, or None when it or all its matching frames are unusable.
    wants: (people preference, whether a period picture was asked for)."""
    people, period = wants
    no_animals = isinstance(period, tuple)
    period = period[0] if no_animals else period
    row = json.loads(src["row"])
    duration = float(row.get("duration") or 0)
    if (src["flags"] == "host" or SKIP_TITLE.search(row.get("title") or "") or not 20 <= duration <= 3 * 3600
            or any(status in CHECK_VIDEO for status in checked.values())):
        return None
    if title_spec and not _weight(" ".join(str(row.get(k) or "") for k in ("title", "tags", "query", "typename"))
                                  .lower(), "", title_spec):
        return None
    if not foreign_ok and FOREIGN.search(str(row.get("title") or "")):
        return None          # 美国自驾、泰恐怖、留学vlog: the people in it are foreign even where a frame does not say so
    described = [m for m in moments if m["text"]]
    if not described or sum(m["kind"] in BAD_KINDS for m in described) > 0.35 * len(described):
        return None
    if not foreign_ok and sum(bool(FOREIGN.search(m["text"])) for m in described) >= 0.4 * len(described):
        return None          # a foreign vlog or drama: its unmarked frames show foreigners too
    frames = []
    for i, m in enumerate(described):
        t = float(m["t"])
        if (hit is not None and round(t, 2) not in hit) or m["kind"] not in FILMED:
            continue
        if any(n["kind"] not in FILMED for n in described[max(0, i - 1):i + 2]):
            continue
        status = _status_at(checked, t)
        if status and status not in CHECK_GOOD:
            continue
        low = m["text"].lower()
        if not foreign_ok and FOREIGN.search(low):
            continue
        if no_animals and ANIMAL.search(low):
            continue
        keywords = low.split("｜", 1)[1] if "｜" in low else ""
        weights = [_weight(low, keywords, spec) for spec in specs]
        if not all(weights):
            continue
        score = sum(weights)
        if people is not None:
            score += 0.5 if _people(low) == people else -0.5
        if not period and PERIOD.search(low):
            score -= 0.6
        score -= sum(cost for pattern, cost in OVERLAY if pattern.search(low))
        if t > duration - 8:
            score -= 0.5     # the window after it may not fit
        if status:
            score += 10      # checked clean: ranks first
        frames.append((score, t, m, status))
    if not frames:
        return None
    frames.sort(key=lambda f: (-f[0], f[1]))
    best, t, m, status = frames[0]
    bad = [c for c, s in checked.items() if s not in CHECK_GOOD]
    match = {"t": t, "text": m["text"], "kind": m["kind"], "check": status}
    if title_spec:
        match["title_word"] = title_spec["word"]
    score = best + 0.1 * min(2, len(frames) - 1) - min(0.3, 0.03 * src["used"])
    return score, row | {"indexed": True, "moments": [f[1] for f in frames],
                         "avoid": sorted({float(x["t"]) for x in moments if x["kind"] in BAD_KINDS} | set(bad)),
                         "index_score": round(score, 3), "index_used": src["used"], "match": match}


# ------------------------------------------------------------------ clean-up checks
# footage_check downloads a few low-resolution seconds at a library moment and judges them like production
# does: "clean", "crop" (clean after the normal crop/delogo), or why not ("text", "mark", "narrow",
# "presenter", "cartoon", "still", "dark", "jump", "gone", "lowres"). find() ranks clean moments first and
# never returns the others; the results travel with the seed.
def _crop_text(crop) -> str:
    """The crop rectangle [x, y, w, h] in fractions of the source picture (express_clean.safe_crop, as production
    applies it), rounded to 4 places; anything else (pixels, a malformed value) is no crop."""
    try:
        if not crop or len(crop) != 4:
            return ""
        values = [round(float(v), 4) for v in crop]
    except (TypeError, ValueError):
        return ""
    if not all(0.0 <= v <= 1.0001 for v in values) or values[0] + values[2] > 1.0001 or values[1] + values[3] > 1.0001:
        return ""
    return json.dumps(values)


def _upsert_checks(con, rows) -> None:
    """rows: (source, t, status, crop, note, checked); an older answer never replaces a newer one."""
    con.executemany(
        "INSERT INTO checks (source, t, status, crop, note, checked) VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(source, t) DO UPDATE SET status=excluded.status, crop=excluded.crop, note=excluded.note, "
        "checked=excluded.checked WHERE excluded.checked >= checks.checked", rows)


def record_checks(items: list[dict]) -> int:
    """items: {"source": "bilibili:BV…" (or "provider"+"id"), "t": float, "status": str, "crop": [x,y,w,h] in
    fractions of the source picture | None, "note": str}; upsert (newer replaces older); returns rows written.
    Short write transaction; never raises on a busy index (skip and return 0)."""
    now, rows = time.time(), []
    for item in items or []:
        try:
            source = str(item.get("source") or "") or (f"{item['provider']}:{item['id']}"
                                                       if item.get("provider") and item.get("id") else "")
            status = str(item.get("status") or "").strip()[:20]
            t = round(float(item["t"]), 2)
            checked = float(item.get("checked") or now)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if ":" in source and status:
            rows.append((source, t, status, _crop_text(item.get("crop")), str(item.get("note") or "")[:200], checked))
    if not rows or not _lock.acquire(timeout=5):
        return 0
    try:
        with _conn(write=True, timeout=3) as con:
            before = con.total_changes
            _upsert_checks(con, rows)
            return con.total_changes - before
    except sqlite3.OperationalError:
        return 0             # busy (a long install or another process): the caller checks again later
    finally:
        _lock.release()


def _checks(con, keys, since: float = 0.0) -> dict:
    out = {}
    keys = list(keys)
    for start in range(0, len(keys), 500):
        part = keys[start:start + 500]
        for r in con.execute(f"SELECT source, t, status FROM checks WHERE source IN ({','.join('?' * len(part))}) "
                             "AND checked >= ?", [*part, since]):
            out.setdefault(r["source"], {})[round(float(r["t"]), 2)] = r["status"]
    return out


def checks_for(keys: list[str], since: float = 0.0) -> dict[str, dict[float, str]]:
    """key -> {t: status} for the videos that have checks (others are left out); since: only checks made at or
    after this time (the footage_check script asks again what older rules decided)."""
    try:
        with _conn() as con:
            return _checks(con, keys, since)
    except sqlite3.Error:
        return {}


def check_stats() -> dict:
    """{"windows": checked moments, "videos": videos checked, "clean_videos": videos with a clean or crop
    moment, "by_status": {status: moments}}."""
    with _conn() as con:
        by_status = {r["status"]: r["n"] for r in con.execute(
            "SELECT status, COUNT(*) AS n FROM checks GROUP BY status ORDER BY n DESC")}
        videos = con.execute("SELECT COUNT(DISTINCT source) FROM checks").fetchone()[0]
        clean = con.execute("SELECT COUNT(DISTINCT source) FROM checks WHERE status IN ('clean','crop')").fetchone()[0]
    return {"windows": sum(by_status.values()), "videos": videos, "clean_videos": clean, "by_status": by_status}


def _valid(row) -> bool:
    return isinstance(row, dict) and row.get("provider") == "bilibili" and bool(BV.fullmatch(str(row.get("id", ""))))


def _write_batch(items) -> int:
    """[(row, moments, topic, described)] in one write transaction; returns how many were saved."""
    if not items:
        return 0
    with _lock, _conn(write=True) as con:
        return sum(1 for row, moments, topic, described in items
                   if _add(con, row, moments, topic=topic, described=described))


def merge(other_file) -> int:
    """Take in another machine's index: videos this one lacks, or has without descriptions, and its
    clean-up checks."""
    from pathlib import Path
    try:
        # A read-only URI (percent-encoded, forward slashes) also works for Windows paths with Chinese names.
        other = sqlite3.connect(Path(other_file).resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.DatabaseError:
        raise ValueError("素材库文件无法读取") from None
    other.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in other.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"sources", "moments"} <= tables:
            raise ValueError("不是镜稿素材库文件")
        with _conn() as con:
            mine = {r["key"]: r["described"] for r in con.execute("SELECT key, described FROM sources")}
        added, batch = 0, []
        for src in other.execute("SELECT key, row, topic, described FROM sources"):
            if src["key"] in mine and (mine[src["key"]] or not src["described"]):
                continue
            try:
                row = json.loads(src["row"])
            except (TypeError, ValueError):
                continue
            if not _valid(row):
                continue
            moments = [dict(m) for m in other.execute("SELECT t, hash, text, kind FROM moments WHERE source=?",
                                                     (src["key"],))]
            batch.append((row, moments, str(src["topic"] or "")[:30], bool(src["described"])))
            if len(batch) >= BATCH:
                added += _write_batch(batch)
                batch = []
        added += _write_batch(batch)
        if "checks" in tables:      # clean-up checks: the newer answer wins
            rows = [(c["source"], round(float(c["t"]), 2), str(c["status"])[:20], c["crop"] or "", c["note"] or "",
                     float(c["checked"] or 0))
                    for c in other.execute("SELECT source, t, status, crop, note, checked FROM checks")]
            for start in range(0, len(rows), 20 * BATCH):
                _write_checks(rows[start:start + 20 * BATCH])
        return added
    except sqlite3.DatabaseError:
        raise ValueError("素材库文件无法读取") from None
    finally:
        other.close()


def export_seed(folder=SEED, part_bytes: int = SEED_PART_BYTES) -> list:
    """The described part of the index as gzip JSON lines in numbered parts (shipped with the app). Each
    part is written beside its final name and moved into place; parts the new export no longer has go last,
    so an interrupted export never leaves the folder empty or a part cut short. A video with clean-up checks
    carries them as "checks": [{"t", "status", "crop", "checked"}]."""
    import gzip
    import io
    folder.mkdir(parents=True, exist_ok=True)
    for leftover in folder.glob("*.tmp"):
        leftover.unlink()
    with _conn() as con:
        sources = con.execute("SELECT key, row, topic FROM sources WHERE described=1 ORDER BY added").fetchall()
        moments = {}
        for m in con.execute("SELECT source, t, hash, text, kind FROM moments WHERE source IN "
                             "(SELECT key FROM sources WHERE described=1) ORDER BY source, t"):
            moments.setdefault(m["source"], []).append(
                {"t": m["t"], "hash": m["hash"], "text": m["text"], "kind": m["kind"]})
        checks = {}
        for c in con.execute("SELECT source, t, status, crop, checked FROM checks ORDER BY source, t"):
            checks.setdefault(c["source"], []).append(
                {"t": c["t"], "status": c["status"], "crop": json.loads(c["crop"]) if c["crop"] else None,
                 "checked": round(c["checked"], 1)})
    # gzip holds output back while writing, so parts are cut by raw size: this JSON shrinks 3-4 times.
    parts, lines, raw = [], [], 0
    def close():
        path = folder / f"part-{len(parts) + 1:02d}.jsonl.gz"
        temporary = path.with_name(path.name + ".tmp")
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as writer:
            writer.write(b"".join(lines))
        temporary.write_bytes(buffer.getvalue())
        os.replace(temporary, path)
        parts.append(path)
        lines.clear()
    for src in sources:
        item = {"row": json.loads(src["row"]), "topic": src["topic"], "moments": moments.get(src["key"], [])}
        if src["key"] in checks:
            item["checks"] = checks[src["key"]]
        line = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        lines.append((line + "\n").encode("utf-8"))
        raw += len(lines[-1])
        if raw >= part_bytes * 3:
            close()
            raw = 0
    if lines or not parts:
        close()
    for stale in folder.glob("part-*.jsonl.gz"):
        if stale not in parts:
            stale.unlink()
    return parts


def _rich(moments) -> bool:
    """Described frame by frame by the local model, with search words ("描述｜关键词", footage_local)."""
    return any("｜" in str(m.get("text") or "") for m in moments)


def install_seed(folder=SEED, progress=lambda *_: None) -> int:
    """Take in the index shipped with the app once per seed version: videos this machine lacks, and better
    descriptions (frame by frame, with search words) of videos it only has contact-sheet lines for; its
    own videos stay otherwise. Written in batches, so closing the app part-way loses little and the next
    start continues; the installed version is recorded inside the index, so a new or reset index file
    is seeded again. Clean-up checks are merged for every video in the seed (the newer answer wins)."""
    import gzip
    import hashlib
    parts = sorted(folder.glob("part-*.jsonl.gz")) if folder.is_dir() else []
    if not parts:
        return 0
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.read_bytes())
    fingerprint = digest.hexdigest()
    with _conn() as con:
        installed = con.execute("SELECT value FROM meta WHERE key='seed'").fetchone()
        merged = con.execute("SELECT 1 FROM meta WHERE key='optimized'").fetchone()
        if installed and installed["value"] == fingerprint and con.execute(
                "SELECT 1 FROM sources WHERE described=1 LIMIT 1").fetchone():
            if not merged:      # seeded by a version that did not merge the index yet
                optimize()
            return 0
        described = {r["key"] for r in con.execute("SELECT key FROM sources WHERE described=1")}
        rich = {r["source"] for r in con.execute("SELECT DISTINCT source FROM moments WHERE text LIKE '%｜%'")}
    added, batch, broken, checks = 0, [], [], []
    for index, part in enumerate(parts):
        try:
            with gzip.open(part, "rt", encoding="utf-8") as lines:
                for line in lines:
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    row = item.get("row") if isinstance(item, dict) else None
                    if not _valid(row):
                        continue
                    # Clean-up checks come in for every video, described here already or not.
                    checks += _seed_checks(_key(row), item.get("checks"))
                    if len(checks) >= 20 * BATCH:
                        _write_checks(checks)
                        checks = []
                    moments = [m for m in item.get("moments") or [] if isinstance(m, dict) and "t" in m]
                    if _key(row) in described and (_key(row) in rich or not _rich(moments)):
                        continue
                    batch.append((row, moments, str(item.get("topic", ""))[:30], True))
                    if len(batch) >= BATCH:
                        added += _write_batch(batch)
                        batch = []
                        progress(index + 1, len(parts), added)
        except (OSError, EOFError, ValueError, zlib.error):
            broken.append(part.name)        # a damaged part: keep the rest, retry this one next start
        added += _write_batch(batch)
        _write_checks(checks)
        batch, checks = [], []
        progress(index + 1, len(parts), added)
    if broken:
        raise ValueError(f"素材库种子文件损坏：{'、'.join(broken)}（其余部分已导入 {added} 个视频）")
    with _lock, _conn(write=True) as con:
        con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('seed', ?)", (fingerprint,))
    if added:
        optimize()
    return added


def _seed_checks(key, checks) -> list:
    """A seed line's checks as table rows. One without a time counts as older than any check made here."""
    rows = []
    for c in checks if isinstance(checks, list) else []:
        try:
            if isinstance(c, dict) and c.get("status"):
                rows.append((key, round(float(c["t"]), 2), str(c["status"])[:20], _crop_text(c.get("crop")), "",
                             float(c.get("checked") or 0)))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _write_checks(rows) -> None:
    if rows:
        with _lock, _conn(write=True) as con:
            _upsert_checks(con, rows)


def optimize() -> None:
    """Merge the search index's segments (written in batches of 500, they number in the thousands): a
    search then reads each word's rows in one place, two to three times faster on a cold disk cache."""
    with _lock, _conn(write=True, timeout=120) as con:
        con.execute("INSERT INTO terms(terms) VALUES('optimize')")
        con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('optimized', ?)", (str(int(time.time())),))


def mark_used(keys) -> None:
    keys = list(keys)
    if not keys:
        return
    with _lock, _conn(write=True, timeout=3) as con:
        con.executemany("UPDATE sources SET used=used+1 WHERE key=?", [(k,) for k in keys])


def _note_failed(items) -> None:
    """Videos whose sheets could not be described: keep them (bare) and count the try."""
    with _lock, _conn(write=True) as con:
        for topic, row, frames in items:
            _add(con, row, [{"t": f["t"], "hash": format(f["hash"], "x")} for f in frames], topic=topic)
            con.execute("UPDATE sources SET tries=tries+1 WHERE key=? AND described=0", (_key(row),))


def stats() -> dict:
    with _conn() as con:
        sources = con.execute("SELECT COUNT(*), COALESCE(SUM(described),0) FROM sources").fetchone()
        moments = con.execute("SELECT COUNT(*), COALESCE(SUM(text!=''),0) FROM moments").fetchone()
        topics = [dict(r) for r in con.execute(
            "SELECT topic, COUNT(*) AS sources FROM sources WHERE topic!='' GROUP BY topic ORDER BY sources DESC")]
    return {"sources": sources[0], "described": sources[1], "moments": moments[0], "described_moments": moments[1],
            "topics": topics, "packs": {name: len(queries) for name, queries in PACKS.items()},
            "file_mb": round(_file().stat().st_size / 1048576, 1) if _file().exists() else 0}


# ------------------------------------------------------------------ building
def _frames(bili, row, count=MOMENTS):
    """Evenly spread preview frames of one video (not dark, not repeats): [{t, image, hash}]. The sprite
    sheets are deleted once cropped (production fetches the ones it needs again), so long builds do not
    fill the disk."""
    from pathlib import Path
    from . import express_match as em, express_sources as es
    previews = bili.previews(row["id"], row["duration"])
    if not previews:
        return []
    try:
        duration = row["duration"] or previews[-1]["t"]
        inner = [p for p in previews if duration * 0.04 <= p["t"] <= duration * 0.96] or previews
        step = max(1, len(inner) / count)
        picked = [inner[min(len(inner) - 1, int(i * step + step / 2))] for i in range(min(count, len(inner)))]
        frames, hashes = [], []
        for p in picked:
            image = es.bilibili_tile(bili, p)
            if em._dark(image):
                continue
            h = em._dhash(image)
            if any(bin(h ^ other).count("1") <= 5 for other in hashes):
                continue
            hashes.append(h)
            frames.append({"t": p["t"], "image": image, "hash": h})
        return frames
    finally:
        for sprite in Path(previews[0]["root"]).glob("sprite-*.jpg"):
            try:
                sprite.unlink(missing_ok=True)
            except OSError:
                pass        # open elsewhere (Windows): production may still be cropping it


def _sleep(seconds, check):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        check()
        time.sleep(min(2.0, max(0.0, end - time.monotonic())))


def _describe(frames, check):
    """One vision request for a sheet of frames: {index: (text, kind)}. Rate limits are waited out
    (about three minutes at most); a refused key, permission or balance stops at once."""
    from . import express_match as em, planner
    sheet = _sheet(frames, SHEET[0])
    content = [{"type": "text", "text": f"共 {len(frames)} 格。"},
               {"type": "image_url", "image_url": {"url": em._jpeg(sheet)}}]
    result = None
    for attempt in range(6):
        check()
        try:
            result, _ = planner.chat(DESCRIBE_SYSTEM, content, vision=True, max_tokens=80 + 48 * len(frames),
                                     timeout=60, task="footage_index")
            break
        except planner.ModelUnavailable as exc:
            if "429" not in str(exc):
                raise          # key, permission or balance: stop the whole build
            if attempt == 5:
                raise RuntimeError("看图接口持续限流") from None
            _sleep(min(60, 5 * 2 ** attempt), check)
        except ValueError as exc:
            if "输出长度上限" in str(exc) and len(frames) > 1:
                # The answer was cut off: half the sheet at a time.
                half = len(frames) // 2
                first, second = _describe(frames[:half], check), _describe(frames[half:], check)
                return first | {half + i: value for i, value in second.items()}
            if attempt >= 3 or not any(k in str(exc) for k in ("超时", "HTTP 5", "JSON")):
                raise
            _sleep(2 * (attempt + 1), check)
    out = {}
    for key, value in (result or {}).items() if isinstance(result, dict) else []:
        if not str(key).isdigit() or not value:
            continue
        value = value if isinstance(value, list) else [str(value)]
        text, kind = str(value[0]), (str(value[1]) if len(value) > 1 else "")
        if not kind and re.search(r"[”\"]\s*[，,]\s*[“\"]", text):    # '描述”，“类型”]' inside one string
            text, kind = re.split(r"[”\"]\s*[，,]\s*[“\"]", text, maxsplit=1)
        text = re.sub(r"[“”\"\[\]]", "", text).strip(" ，,。")[:40]
        if text and 1 <= int(key) <= len(frames):
            out[int(key) - 1] = (text, _kind(kind))
    return out


def _kind(raw: str) -> str:
    """The answer's picture type, reduced to the fixed list (the model sometimes writes two or decorates it)."""
    for word, kind in (("主持", "主持人出镜"), ("黑屏", "黑屏或模糊"), ("模糊", "黑屏或模糊"), ("图表", "图表文字"),
                       ("文字", "图表文字"), ("动画", "动画"), ("影视", "影视剧"), ("实拍", "实拍")):
        if word in raw:
            return kind
    return ""


def _sheet(frames, cols):
    """Frames in a grid, numbered from 1 in yellow corner labels (the numbers the answer uses)."""
    from PIL import Image, ImageDraw
    from . import express_match as em
    w, h = em.TILE_W, em.TILE_H
    rows = -(-len(frames) // cols)
    sheet = Image.new("RGB", (cols * w, rows * h))
    draw = ImageDraw.Draw(sheet)
    font = em._font(26)
    for i, frame in enumerate(frames):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(frame["image"].resize((w, h)), (x, y))
        box = draw.textbbox((x + 4, y + 2), str(i + 1), font=font)
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(255, 214, 0))
        draw.text((x + 4, y + 2), str(i + 1), font=font, fill=(0, 0, 0))
    return sheet


GENERIC_TAGS = {"生活记录", "日常", "vlog", "搞笑", "知识", "科普", "历史", "纪录片", "视频", "记录", "分享", "必剪创作",
                "打卡挑战", "生活", "自制", "原创", "剪辑", "混剪", "合集", "教程", "高清", "4k", "中国", "bilibili", "哔哩哔哩",
                "全能打卡挑战", "热点", "新闻", "社会", "美食", "旅行", "旅游", "风景", "科技", "军事", "财经", "人文", "趣味",
                "涨知识", "干货", "热门", "推荐", "震撼", "经典", "解说", "影视", "电影", "电视剧", "音乐", "动画"}
CHUNK = 160             # videos framed and described together; each sheet's videos are saved as it returns
BLOCKED_PAUSE = 300     # seconds to wait when Bilibili's risk control blocks previews, before trying again


def frame_count(duration: float) -> int:
    """Longer videos hold more distinct scenes: 8 frames under 5 minutes, 12 under 20, else 16."""
    return 8 if duration < 300 else 12 if duration < 1200 else 16


def tag_searches(found: dict, used: set, per_topic: int = 8) -> list[tuple[str, str]]:
    """New searches from the tags a topic's videos share (Bilibili's own words for the subject).
    found: topic -> distinct videos."""
    from collections import Counter
    out = []
    for topic, rows in found.items():
        counts = Counter()
        for row in {_key(r): r for r in rows}.values():
            for tag in {t.strip() for t in str(row.get("tags", "")).split(",")}:
                if 2 <= len(tag) <= 10 and tag.lower() not in GENERIC_TAGS and not SKIP_TITLE.search(tag):
                    counts[tag] += 1
        picked = [tag for tag, n in counts.most_common() if n >= 3 and tag not in used][:per_topic]
        out += [(topic, tag) for tag in picked]
        used |= set(picked)
    return out


def _risk_paused(bili, check, progress, report, pauses, message):
    """When Bilibili's risk control refuses previews or searches: wait, reset the breakers, go on.
    Returns the new pause count; three pauses in one build stop it (finished videos stay saved)."""
    if bili.shots_blocked < 3 and bili.search_refused < 3:
        return pauses
    report["blocked"] = True
    if pauses >= 3:
        raise RuntimeError("B站接口持续风控，已停止扩库；已完成的视频已保存，稍后重新扩充会接着做")
    progress(message + f" · B站接口风控，暂停 {BLOCKED_PAUSE // 60} 分钟后继续")
    _sleep(BLOCKED_PAUSE, check)
    with bili.counts:
        bili.shots_blocked = bili.search_refused = 0
    return pauses + 1


def build(queries: list[tuple[str, str]], check, progress, *, pages: int = 1, describe: bool = True,
          limit: int = 100000, widen: int = 8, describer=None) -> dict:
    """Index the Bilibili results of [(topic, query)]: search, preview frames, descriptions. Each sheet's
    videos are saved as soon as it is described; videos already described (or failed twice on their own)
    are skipped, so a stopped or failed build continues where it was. describer(row, images) -> {frame
    index: (text, kind)} replaces the contact-sheet model with one that looks at a video at a time (a local
    model, scripts/footage_local.py)."""
    from . import express_sources as es
    bili = es.Bilibili()
    report = {"queries": len(queries), "found": 0, "new": 0, "indexed": 0, "described_frames": 0, "failed": 0,
              "blocked": False, "expanded": 0}
    pauses = 0
    try:
        rows, lock, by_topic = {}, threading.Lock(), {}
        def find(item):
            topic, query = item
            found = []
            for page in range(1, pages + 1):
                check()
                if bili.search_refused >= 3 and not bili.offline_search:
                    break                  # risk control: stop paging (the build pauses after this batch)
                try:
                    asked = not bili.cached(query, page)
                    got = bili.search(query, page)
                except Exception:
                    break
                found += got
                if not got:
                    break
                if asked and not bili.offline_search:
                    time.sleep(SEARCH_GAP)    # stay under the search API's burst limit
            with lock:
                for row in found:
                    if 20 <= row.get("duration", 0) <= 3 * 3600 and not SKIP_TITLE.search(row.get("title", "")):
                        rows.setdefault(_key(row), (topic, row))
                        by_topic.setdefault(topic, {})[_key(row)] = row
        search_pauses = 0
        def search_all(items, lo, hi, label):
            nonlocal search_pauses
            done = 0
            for start in range(0, len(items), 30):
                with execution.ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as pool:
                    for _ in pool.map(find, items[start:start + 30]):
                        done += 1
                        progress(lo + (hi - lo) * done / max(1, len(items)), f"{label} {done}/{len(items)} · 已找到 {len(rows)} 个视频")
                if bili.search_refused >= 3 and not bili.offline_search:
                    report["blocked"] = True
                    if search_pauses:
                        # Still refused after a pause: go on with the searches cached today and describe those.
                        bili.offline_search = True
                        report["search_offline"] = True
                        progress(lo + (hi - lo) * done / max(1, len(items)), f"{label} · B站搜索风控，本轮改用今天已缓存的搜索结果")
                    else:
                        progress(lo + (hi - lo) * done / max(1, len(items)),
                                 f"{label} · B站搜索风控，暂停 {BLOCKED_PAUSE // 60} 分钟后继续")
                        _sleep(BLOCKED_PAUSE, check)
                        search_pauses += 1
                    with bili.counts:
                        bili.search_refused = 0
        search_all(queries, 1, 12, "检索关键词")
        if widen:
            more = tag_searches({topic: list(found.values()) for topic, found in by_topic.items()},
                                {q for _, q in queries}, per_topic=widen)
            report["expanded"] = len(more)
            search_all(more, 12, 18, "按标签扩展检索")
        report["found"] = len(rows)
        done = known(rows)
        with _conn() as con:
            skip = {r["key"] for r in con.execute(f"SELECT key FROM sources WHERE described=1 OR tries>={MAX_TRIES}")}
        todo = [rows[k] for k in rows if k not in skip][:limit]
        report["new"] = len([k for k in rows if k not in done])
        chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
        index = 0
        while index < len(chunks):
            lo = 18 + 81 * index / max(1, len(chunks))
            hi = 18 + 81 * (index + 1) / max(1, len(chunks))
            label = f"第 {index + 1}/{len(chunks)} 批（共 {len(todo)} 个视频）"
            refused = _build_chunk(bili, chunks[index], check,
                                   lambda share, message: progress(lo + (hi - lo) * share, f"{label} · {message}"),
                                   report, describe, describer)
            if bili.shots_blocked >= 3:
                # Videos whose previews were refused go round again after the pause.
                pauses = _risk_paused(bili, check, lambda m: progress(hi, m), report, pauses, label)
                if refused:
                    chunks.insert(index + 1, refused)
            index += 1
        return report
    finally:
        bili.close()


class _Halted(Exception):
    """Another sheet stopped the build: do not start more model requests."""


def _build_chunk(bili, chunk, check, progress, report, describe, describer=None) -> list:
    """Frames and descriptions for up to CHUNK videos. Returns the videos whose previews were refused by
    risk control (to retry after a pause)."""
    from concurrent.futures import as_completed, wait
    from . import media
    from .planner import ModelUnavailable
    stops = (media.Cancelled, costs.BudgetExceeded, costs.PricingUnavailable, ModelUnavailable)
    def frames(item):
        topic, row = item
        check()
        try:
            return topic, row, _frames(bili, row, frame_count(float(row.get("duration") or 0))), False
        except stops:
            raise
        except Exception as exc:
            return topic, row, None, "风控" in str(exc)      # this video only
    refused = []
    if describer:
        # Each video goes to the model as soon as its frames are in, so the model is not left waiting
        # for the whole chunk's previews.
        def stream():
            with execution.ThreadPoolExecutor(max_workers=SHOT_WORKERS) as pool:
                for topic, row, got, blocked in pool.map(frames, chunk):
                    if got:
                        yield topic, row, got
                    elif blocked:
                        refused.append((topic, row))
                    else:
                        report["failed"] += 1
        _describe_videos(stream(), describer, check, progress, report, len(chunk))
        return refused
    framed = []
    with execution.ThreadPoolExecutor(max_workers=SHOT_WORKERS) as pool:
        for n, (topic, row, got, blocked) in enumerate(pool.map(frames, chunk), 1):
            if got:
                framed.append((topic, row, got))
            elif blocked:
                refused.append((topic, row))
            else:
                report["failed"] += 1
            if n % 20 == 0 or n == len(chunk):
                progress(0.4 * n / max(1, len(chunk)), f"预览帧 {n}/{len(chunk)}")
    if not describe:
        _write_batch([(row, [{"t": f["t"], "hash": format(f["hash"], "x")} for f in got], topic, False)
                      for topic, row, got in framed])
        report["indexed"] += len(framed)
        return refused
    # Whole videos to a sheet, so each answer completes its videos and they can be saved at once.
    per = SHEET[0] * SHEET[1]
    sheets, current, count = [], [], 0
    for item in framed:
        if current and count + len(item[2]) > per:
            sheets.append(current)
            current, count = [], 0
        current.append(item)
        count += len(item[2])
    if current:
        sheets.append(current)
    halt = threading.Event()
    streak = {"failed": 0}
    def guarded():
        check()
        if halt.is_set():
            raise _Halted()
    def describe_sheet(sheet):
        """{frame index: (text, kind)} for one sheet; a sheet that fails as a whole raises."""
        flat = [frame for _, _, got in sheet for frame in got]
        answer = _describe(flat, guarded)
        missing = [i for i in range(len(flat)) if i not in answer]
        if missing:          # cut-off or partial answer: ask once more for the skipped frames, keep what came
            try:
                again = _describe([flat[i] for i in missing], guarded)
            except stops + (_Halted,):
                raise
            except Exception:
                again = {}
            for j, i in enumerate(missing):
                if j in again:
                    answer[i] = again[j]
        return answer
    def run(sheet):
        """(sheet, kind, payload): "sheet" with one answer for the whole sheet, "videos" with an answer
        (or None when refused) per video, or "skip" when the service, not these videos, failed."""
        guarded()
        try:
            answer = describe_sheet(sheet)
        except stops + (_Halted,):
            raise
        except Exception as exc:
            check()
            with _lock:
                streak["failed"] += 1
                stop = streak["failed"] >= 8
                alone = streak["failed"] <= 1
            if stop:
                raise RuntimeError(f"连续 8 张联系表描述失败（{str(exc)[:80]}），已停止扩库；已完成的视频已保存，"
                                   "稍后重新扩充会接着做") from None
            # A refusal that may belong to these videos (HTTP 400 content check, unusable answer) is worked
            # out video by video; service-wide trouble (limits, timeouts, 5xx, several failures) counts nothing.
            if not (alone and isinstance(exc, ValueError) and "HTTP 400" in str(exc)):
                return sheet, "skip", None
            answers = []
            for item in sheet:
                try:
                    answers.append(describe_sheet([item]) if len(sheet) > 1 else None)
                except stops + (_Halted,):
                    raise
                except Exception:
                    answers.append(None)
            return sheet, "videos", answers
        with _lock:
            streak["failed"] = 0
        return sheet, "sheet", answer
    def save(sheet, kind, payload):
        if kind == "skip":
            return
        items, failed, position = [], [], 0
        for n, item in enumerate(sheet):
            topic, row, got = item
            if kind == "sheet":
                own, offset = payload, position
            else:
                own, offset = payload[n], 0
            position += len(got)
            if own is None:
                failed.append(item)
                continue
            moments = []
            for k, frame in enumerate(got):
                text, kind_ = own.get(offset + k, ("", ""))
                moments.append({"t": frame["t"], "hash": format(frame["hash"], "x"), "text": text, "kind": kind_})
            if any(m["text"] for m in moments):
                items.append((row, moments, topic, True))
                report["described_frames"] += sum(1 for m in moments if m["text"])
            else:
                failed.append(item)         # answered, but nothing usable for this video
        _write_batch(items)
        report["indexed"] += len(items)
        if failed:
            _note_failed(failed)
    with execution.ThreadPoolExecutor(max_workers=DESCRIBE_WORKERS) as pool:
        futures = [pool.submit(run, sheet) for sheet in sheets]
        saved = set()
        try:
            for n, future in enumerate(as_completed(futures), 1):
                save(*future.result())
                saved.add(future)
                progress(0.4 + 0.59 * n / max(1, len(sheets)), f"描述画面 {n}/{len(sheets)} 张联系表")
        except BaseException:
            halt.set()                   # running sheets finish their current request and stop
            for future in futures:
                future.cancel()
            wait(futures)
            for future in futures:       # answers that came back are paid for: keep them
                if future not in saved and not future.cancelled() and future.exception() is None:
                    try:
                        save(*future.result())
                    except Exception:
                        pass
            raise
    progress(1.0, "已保存")
    return refused


def _describe_videos(framed, describer, check, progress, report, total: int) -> None:
    """(topic, row, frames) items, as they come, described one video per request and each saved as it
    returns. A video the describer could not answer counts a try; eight failures in a row mean the model
    itself is down."""
    from concurrent.futures import as_completed, wait
    from . import media
    halt = threading.Event()
    def run(item):
        check()
        if halt.is_set():
            raise _Halted()
        try:
            return item, describer(item[1], [frame["image"] for frame in item[2]])
        except media.Cancelled:
            raise
        except Exception as exc:
            return item, exc
    state = {"streak": 0, "done": 0}
    def save(future):
        (topic, row, got), answer = future.result()
        moments = []
        if isinstance(answer, dict):
            for k, frame in enumerate(got):
                text, kind = answer.get(k, ("", ""))
                moments.append({"t": frame["t"], "hash": format(frame["hash"], "x"), "text": text, "kind": kind})
        if any(m["text"] for m in moments):
            state["streak"] = 0
            _write_batch([(row, moments, topic, True)])
            report["indexed"] += 1
            report["described_frames"] += sum(1 for m in moments if m["text"])
        else:
            state["streak"] += 1
            _note_failed([(topic, row, got)])
            if state["streak"] >= 8:
                raise RuntimeError(f"连续 8 个视频描述失败（{str(answer)[:80]}），已停止扩库")
        state["done"] += 1
        if state["done"] % 10 == 0:
            progress(0.99 * state["done"] / max(1, total), f"描述画面 {state['done']}/{total} 个视频")
    futures, pending = [], set()
    with execution.ThreadPoolExecutor(max_workers=DESCRIBE_WORKERS) as pool:
        try:
            for item in framed:
                future = pool.submit(run, item)
                futures.append(future)
                pending.add(future)
                for finished in [f for f in pending if f.done()]:
                    pending.discard(finished)
                    save(finished)
            for finished in as_completed(pending):
                save(finished)
        except BaseException:
            halt.set()
            for future in futures:
                future.cancel()
            wait(futures)
            raise
    progress(1.0, "已保存")
