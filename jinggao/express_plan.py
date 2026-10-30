"""Planning for the express pipeline.

The script is split locally into one subtitle line per shot (every character kept),
then a cheap text request (one per ~100 shots) groups shots that can share footage
searches and returns Chinese (Bilibili) and English (stock) queries, a short visual
target per shot, words to find the shot in the local footage library ("k"), and the
named people that must really appear.
"""
from __future__ import annotations

import functools
import math
import re

from . import captions, footage_index, identity, planner, subtitle_style

MAX_SHOTS = 400
BATCH = 40     # shots a planning request: with three picture-word sets a shot, 67 shots ran past the output limit

# The local library (footage_index.find) is searched by words that must all be on one described frame, each
# widened through the describer's own vocabulary (footage_index.expand: 父亲 → 父子/中年男…, 摔手机 → 砸东西/摔碎…).
# Its describer writes what it sees: people by look (中年男子 far more often than 父亲), plain actions, objects
# and places. These are words it wrote on filmed frames of the seed (assets/footage-index, 886k filmed frames,
# counted 2026-10-28): each is on 100+ frames, most on thousands. The plan prompt lists them so a cheap model
# asks in words the library has, not in ideas (教育, 透支) or search wording (空镜, 短剧片段).
LIBRARY_PEOPLE = ("男子 女子 男孩 女孩 少年 男生 女生 学生 小孩 年轻男子 年轻女子 中年男子 中年女子 老人 情侣 男女 夫妻 "
                  "一家三口 母子 父子 老师 医生 警察 工人 顾客 人群")
LIBRARY_ACTS = ("看手机 玩手机 打电话 交谈 争吵 争执 哭泣 拥抱 牵手 微笑 愤怒 低头 沉思 疲惫 无奈 吃饭 做饭 洗碗 写字 学习 "
                "看书 写作业 睡觉 行走 背影 排队 购物 上课 考试 开会 数钱")
LIBRARY_THINGS = ("手机 电脑 书包 行李箱 书本 试卷 台灯 钞票 餐桌 沙发 床上 卧室 客厅 厨房 宿舍 教室 校园 校门 操场 图书馆 "
                  "走廊 街道 医院 病房 办公室 超市 餐厅 车内 地铁 法庭 工厂 农村 小区 阳台")

SYSTEM = """你是营销号短视频的选材编辑。给你按口播切好的镜头，请为找“实拍视频素材”做检索规划。
只输出 JSON：
{"domain":"..","ancient":[],"event":[],"pool":[],"topic":{"zh":["..",".."],"en":["..",".."]},
 "groups":[{"id":"G1","zh":["..",".."],"en":["..",".."],"person":""}],
 "shots":[{"id":"S1","g":"G1","v":"..","k":["..","..",".."],"p":[]}]}
规则：
1. groups：把画面内容相近、可以共用同一批素材的镜头归为一组（每组2-4个镜头，可以不相邻），组数不少于镜头数的四分之一、不超过镜头数的一半，最多20组。组分得太粗，全片只有几批素材，画面会翻来覆去重复。
2. zh：2个B站检索词，每个2-4个短词、用空格分隔。必须带上这组口播里出现的具体名称（国家、地区、城市、组织、人名、品牌、型号之一），再写具体可见的画面主体，例如“也门 胡塞 导弹 袭击”“沙特 联军 空袭”“红海 驱逐舰”“集装箱码头 岸桥 吊装”。时事新闻（战争冲突、外交、政治、社会事件）加“新闻”或“现场”；历史题材可加“纪录片/历史影像”；只有泛泛的场景（城市、自然、生活）才用“航拍/空镜/素材”。讲人的镜头（官员、领导人、民众、士兵、当事人、开会、讲话、抗议）检索词写人物和场景（如“也门 总理 讲话”“萨那 街头 民众”“沙特 王储 会见”），不要写武器装备；同一篇里各组的检索词要互相错开，不要几组都搜同一类画面。情感、婚姻、家庭、职场、社会故事（没有真实新闻画面的个人经历）的镜头，检索词写情节加“电视剧片段”或“短剧”，如“夫妻 吵架 电视剧片段”“妻子 查看 丈夫手机 短剧”“婆媳 争执 电视剧”，这类片段有真人有情绪，比空镜好。不要写抽象概念、观点或整句，也不要只写“导弹”“军舰”“港口”这种不带名称的泛词（会搜到别国别处的画面）。
3. en：2个Pixabay英文检索词，2-4个英文单词，写通用可见画面，如“container port aerial”。外国图库只有通用的风景、城市、生活、行业画面：若该组讲的是具体国家地区正在发生的新闻事件（战争冲突、外交、政治、灾难、社会新闻）、中国特定历史事件、具名人物或特定中国文化物件，en 给空数组（图库里只会是不相干的游艇、海景、街道）；画面主体是中国人的生活场景（家庭、夫妻、父母孩子、职场、街坊邻里、农村）时也给空数组（图库里是外国人，观众一看就不对）。
4. person：只有当该组镜头的口播在讲某个具体真实人物、画面理应出现此人时，填写原文中的完整姓名，否则填空字符串。有人物的组只放讲这个人的镜头，zh 检索词必须包含这个姓名（如“某某 历史影像”）。
5. shots：每个镜头都要出现且顺序不变。g=所属组；v=6-14字的通用画面建议（场景+主体，容易在实拍素材中找到，不写细节动作、特写或文字内容；不要写截图、网页、评论区、弹幕、聊天记录这类屏幕画面，讲网传视频或网友热议时写与事件相关的实拍场景），普通画面只需大致相关；p=画面必须出现的真实具名人物姓名（必须与口播原文逐字一致），没有就空数组；社会故事、情感故事里的化名和代称（张女士、李先生、老周、小李、王某、刘阿姨）不是具名人物，不写进 p，检索词里也不要写这些名字。
6. topic：整篇文稿的通用画面检索词（zh 2个、en 2个），用于补位，要选不出现具体人物也能用的环境/场景画面。
7. 若有“用户画面要求”，必须体现在检索词和画面建议中。
9. ancient：口播里出现的、生活在照相术出现（约1840年）以前的具名古人姓名（如范仲淹、包拯、李白），没有就空数组；这些人不可能有真实影像。
8. domain：整篇所属的宽泛领域，2-6字（如“乒乓球”“集装箱港口”“兰州牛肉面”“宋朝历史”“人工智能”）。不要用文章标题、具体事件或人物（讲范仲淹救灾写“宋朝历史”，不能写“范仲淹荒政”）。时事新闻写事件所在的国家或地区（如“也门”“俄乌”“台海”“日本”），不要写“局势”“形势”“问题”“风云”这类抽象词（搜不到画面）。除带具体人名的检索词外，每个 zh 检索词都要包含这个领域词，避免搜到同名的其他领域（例如讲乒乓球世界排名，要写“乒乓球 世界排名”，不能只写“世界排名”）。
10. 古人（ancient 里的人）：他们没有真实影像，组的 zh 检索词写能搜到影视剧、纪录片片段的词，如“范仲淹 电视剧 片段”“范仲淹 纪录片”；普通历史画面的组写朝代场景，如“宋朝 市井 影视剧”“古代 粮仓 纪录片”“宋代 西湖 画舫”，尽量是有动作的视频画面，少用静态插画。
11. event：若整篇讲的是一件具体的新闻事件（有明确的当事方、地点或时间，例如某次袭击、某场会谈、某起案件），给 2-3 个能在 B 站搜到这件事新闻报道视频的检索词，每个 3-5 个短词，写当事方 + 地点 + 事件（如“胡塞 导弹 袭击 亚丁 总统府”“也门 内阁 会议 遇袭”“沙特 联军 空袭 萨那”）；讲的是泛泛的话题、个人故事、历史或科普，给空数组。
13. pool：整篇的“同题素材”检索词，用来在 B 站找讲的就是这件事、这个人、这段历史的长视频（比赛集锦、新闻报道、纪录片、历史影像、采访），全片尽量从这几部里取画面，画风统一、不违和。给 2-3 个，每个 2-5 个短词：体育比赛写两队或选手 + 赛事或日期 + 集锦（如“勇士 森林狼 集锦”“库里 44分 集锦”）；新闻事件写当事方 + 事件 + 新闻；讲具体人物、公司、品牌写名字 + 新闻或采访或纪录片（如“赵长鹏 币安 新闻”“王兴兴 宇树 采访”）；讲一段历史写时代和事件 + 历史影像或纪录片（如“抗日战争 历史影像 纪录片”“淞沪会战 纪录片”）；讲某个行业、地方、社会现象写现象 + 新闻报道或纪录片（如“国企改革 新闻报道”“县城 年轻人 就业 新闻”）。化名的情感家庭故事、纯观点议论，没有讲同一件事的现成视频，给空数组。
12. k：每个镜头 3 组“素材库画面词”，从具体到宽泛，用来在本地素材库里按词找画面。库里每一帧都配了一句看图写的短描述（如“中年男子坐在摊位前低头看手机”“一男一女在室内争执”），一组里的词要能同时出现在一句这样的描述里，所以只写镜头拍得到的人、动作、表情、物件、地点：
- 每组 1-3 个词，词和词之间空格隔开（写“男孩 玩手机”，不写“男孩玩手机”），每个词 2-4 个字。第1组最具体（人物+动作或物件），第2组放宽（换成更常见的说法或组合，如人物+表情、人物+地点），第3组最宽：这句的主要人物，或人物+表情、动作。不要整组只有物件或地点：单写“手机”搜到的是手机评测，单写“卧室”“城市”“校园”是空镜；物件最好连着动作写（玩手机、摔手机、拖行李箱），“中年男子 手机”搜到的也多是手机评测。
- 人物写镜头里看得出的人，不写名字：父亲、母亲、儿子、女儿、丈夫、妻子、大学生、老人这样的称呼可以直接写（库会换成看图描述里的说法）；恋人、男友、女友写“情侣”，丈夫和妻子在一起写“夫妻”，一家人写“一家三口”。
- 尽量用库里常见的词。人物：""" + LIBRARY_PEOPLE + "；动作表情：" + LIBRARY_ACTS + "；物件地点：" + LIBRARY_THINGS + """。其他题材也用这样朴素的画面词（如 士兵 坦克 爆炸 废墟 港口 集装箱 农田 古装）。
- 不写人名和化名（小夏、老周、张女士）、数字、金额、比例；不写抽象词（教育、底线、成长、压力、关系、透支）和屏幕画面（截图、文字、数据）；不写检索用语和风景（电视剧片段、短剧、新闻、现场、素材、空镜、航拍、风景、夜景）。只有这句讲的就是某个地方、里面没有人时，才只写地点。
- 讲观点、数据、道理的句子，写这个故事主角的一个具体场景（讲大学生同居写“情侣 争吵”“大学生 学习”，讲孩子沉迷手机写“男孩 玩手机”）。
- 讲中国的人和事不要写“外国”。
例：“父亲一时火起把手机摔了个稀碎”→"k":["男子 摔手机","摔手机","父亲 愤怒"]；“大学生情侣拖着行李箱搬进校外出租屋”→"k":["情侣 行李箱","大学生 行李箱","情侣 搬家"]；“每三对校园情侣里，就有一对选择了校外同居”→"k":["情侣 校园","情侣 牵手","情侣"]。"""


def _text(value, limit) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


# A story's stand-in for an ordinary person (张女士, 老周, 小李, 王某): no one to find footage of, and as a
# search word it finds strangers' vlogs with the same nickname.
# 老X/小X/阿X is one only when the plan names it as a person (老挝, 小米 and 阿里 are not people).
TITLED = re.compile(r"[\u4e00-\u9fff](?:女士|先生|小姐|某某|某|阿姨|大爷|大妈|大娘|大姐|大哥|师傅|奶奶|爷爷|叔|婶)")
NICKNAME = re.compile(r"(?:老|小|阿)[\u4e00-\u9fff]")


def placeholder(name: str, person: bool = True) -> bool:
    return bool(TITLED.fullmatch(name or "") or (person and NICKNAME.fullmatch(name or "")))


def _queries(values, limit, ascii_only=False) -> list[str]:
    result = []
    for value in values if isinstance(values, list) else []:
        text = _text(value, limit)
        if not text or (ascii_only and not re.fullmatch(r"[A-Za-z0-9 ,'&-]+", text)):
            continue
        if text not in result:
            result.append(text)
    return result[:2]


# More words the describer writes, for plans made before "k" existed and for a visual the user typed (their
# sets come from the text), kept apart as people, actions and things: a set needs a person or an action, since a
# lone thing or place finds the thing itself (手机 phone reviews, 城市 skylines, 卧室 empty rooms).
MORE_WHO = "孩子 年轻人 大学生 父母 少女 小男孩 小女孩 婴儿 护士 农民 厨师 司机 行人 同学 朋友 士兵 老板 员工"
MORE_ACTS = ("聊天 搬家 搬运 开车 骑车 跑步 奔跑 喝酒 抽烟 喝水 化妆 讲课 讲话 演讲 握手 鼓掌 跪地 倒地 摔倒 受伤 手术 "
             "输液 紧张 惊讶 担忧 失落 孤独 焦虑 大笑 皱眉 看电视 打游戏 玩游戏 摔手机 摔东西 打架 下棋 跳舞 钓鱼 唱歌 "
             "打扫 干活 劳作 上班 加班 工作 打球 游泳 散步 逛街 熬夜 发呆 洗澡 摆摊 吃药 爆炸 签字 会议 古装")
MORE_THINGS = ("外卖 快递 礼物 蛋糕 红包 现金 平板 耳机 冰箱 家具 装修 汽车 轿车 电动车 自行车 公交车 火车 高铁 飞机 机场 "
               "车站 学校 大学 小学 幼儿园 食堂 房间 出租屋 公寓 公园 河边 路边 天台 商场 夜市 菜市场 酒吧 会议室 车间 工地 "
               "田间 村庄 城市 街头 楼道 电梯 诊室 病床 黑板 讲台 校服 窗边 门口 镜子 电视 小吃 小狗 小猫 小船 小镇 小巷 "
               "老街 坦克 废墟 港口 集装箱 农田 导弹 军舰")
LIBRARY_WHO = frozenset((LIBRARY_PEOPLE + " " + MORE_WHO).split())
LIBRARY_DOING = frozenset((LIBRARY_ACTS + " " + MORE_ACTS).split())
LIBRARY_STUFF = frozenset((LIBRARY_THINGS + " " + MORE_THINGS).split())
LIBRARY_VOCAB = LIBRARY_WHO | LIBRARY_DOING | LIBRARY_STUFF
# People seen together: a frame of a couple is described as 情侣 or 一男一女, not as two 学生.
LIBRARY_PAIRS = ("情侣", "男女", "夫妻", "一家三口", "母子", "父子")
# footage_index.expand widens a word to the describer's (父亲 → 父子/中年男…, 男友 → 情侣/恋人…, 丈夫 → 夫妻/夫妇,
# 家长 → 父母/母子/中年女…), so a word its table knows is kept as the model wrote it: rewriting 男友 as 年轻男子
# or 丈夫 as 男子 here would lose the couple. These are role words its table lacks, as the closest word it knows.
ROLES = {"男孩子": "男孩", "女孩子": "女孩", "小伙": "小伙子", "孩子们": "孩子", "爸爸妈妈": "父母", "宝妈": "母亲",
         "宝爸": "父亲", "老两口": "老人", "老夫妻": "老人", "小夫妻": "夫妻", "年轻夫妻": "夫妻", "夫妻俩": "夫妻",
         "情侣们": "情侣", "年轻情侣": "情侣", "男大学生": "大学生", "女大学生": "大学生", "父子俩": "父子",
         "母子俩": "母子", "一家子": "一家人"}
# Words no frame description carries, or that match footage of anything: search wording, screen pictures,
# and the ideas a cheap model reaches for on opinion lines (unless footage_index knows them: 同居 → 情侣).
NOT_SEEN = frozenset("""空镜 素材 画面 镜头 视频 片段 实拍 影视剧 电视剧 电视剧片段 短剧 剧照 新闻 现场 特写 近景 远景 全景
高清 航拍 风景 夜景 天际线 截图 网页 评论区 弹幕 聊天记录 字幕 文字 图表 数据 数字 比例 调研 调查 教育 成长 未来 青春 压力
底线 困局 观点 现象 社会 问题 关系 心理 情绪 爱情 感情 价值 意义 概念 信任 沟通 妥协 理解 矛盾 同居 透支 代价 选择 自由
责任 独立 习惯 目标 计划 机会 现实 时间 金钱 道理 结果 影响 事件 事情 故事 生活 日常""".split())
# A word with search wording or scenery in it asks for that, whatever else it holds (校园空镜, 城市航拍, 短剧片段).
SEARCH_WORDING = re.compile(r"空镜|素材|片段|航拍|风景|全景|夜景|天际线|景色|美景|新闻|短剧|电视剧|影视剧|现场|特写|镜头|截图")
# A count or an amount in a word (十六岁, 三对情侣, 八千块钱, 16岁男孩); 一 and 两 alone are in 一家, 一起, 两口子.
COUNTED = re.compile(r"[0-9零〇一二两三四五六七八九十百千万亿几半]+[岁块元角年月日天号个次倍成对位名部台张件层]|[0-9]"
                     r"|[零〇二三四五六七八九十百千万亿几]{2,}")
# The describer marks foreigners ("外国男子"); a line about China asking for them finds the wrong people.
FOREIGN = re.compile(r"外国|国外|海外|欧美|西方|老外|洋人")
HAN_WORD = re.compile(r"[\u4e00-\u9fff]{2,6}")
# A person with a phone is filmed looking at it: "person + 手机" finds phone reviews and unboxings (中年男子 手机,
# 家长 手机: mostly reviewers), "person + 看手机" people on their phones (4 of 4 each, measured 2026-10-28).
HELD = {"手机": "看手机"}
# Lines no picture of a lone thing or place goes with: people are in them, or they give figures.
PEOPLE_LINE = re.compile(r"[我你他她]|[%％]|百分之|比例|调查|统计|数据|报告显示|研究显示")


@functools.lru_cache(maxsize=8192)
def _alternatives(word: str) -> tuple:
    """footage_index's alternatives for a word, the word first (just the word when its table lacks it)."""
    return tuple(footage_index.expand(word)) or (word,)


def _known(word: str) -> bool:
    return len(_alternatives(word)) > 1


def _role(word: str) -> str:
    return word if _known(word) else ROLES.get(word, word)


# The words the text fallback finds: the lists above, the role words, and the words footage_index's table
# widens (沉迷游戏, 摔手机, 家长, 出租屋 …); single characters (家, 哭) would match inside anything.
_VOCAB = LIBRARY_VOCAB | set(ROLES) | {w for w in getattr(footage_index, "_SYN", {})
                                       if HAN_WORD.fullmatch(w) and not SEARCH_WORDING.search(w)}
_LONGEST = max(map(len, _VOCAB))


@functools.lru_cache(maxsize=8192)
def _kind(word: str) -> str:
    """What one word (or "/" alternatives) shows, "who", "doing" or "thing": by the lists above, the word
    itself first, then its footage_index alternatives. A word in none that names no person counts as a thing,
    so it does not hold up a set alone on a line about people (江西 校园 is campus scenery)."""
    pieces = [p for p in word.split("/") if p]
    for lists in ((p,) for p in pieces), (_alternatives(p) for p in pieces):
        alternatives = [a for alts in lists for a in alts]
        for kind, vocab in (("who", LIBRARY_WHO), ("doing", LIBRARY_DOING), ("thing", LIBRARY_STUFF)):
            if any(a in vocab for a in alternatives):
                return kind
    if any(footage_index.PEOPLE.search(p) and not footage_index.NO_PEOPLE.search(p) for p in pieces):
        return "who"        # 中年夫妻, 外国男子
    return "thing"


def _pair(word: str) -> bool:
    return word in LIBRARY_PAIRS or any(a in ("情侣", "夫妻", "男女", "一家三口") for a in _alternatives(word))


def _vocab_words(text: str) -> list[str]:
    """Library words in a text, in order, the longest at each place (大学生, never 大学)."""
    found, i = [], 0
    while i < len(text):
        for size in range(min(_LONGEST, len(text) - i), 1, -1):
            piece = text[i:i + size]
            if piece in _VOCAB:
                found.append(_role(piece))
                i += size
                break
        else:
            i += 1
    return list(dict.fromkeys(found))


def _usable(word: str, avoid, narration: str) -> bool:
    known = _known(word)
    return bool(HAN_WORD.fullmatch(word) and not SEARCH_WORDING.search(word)
                and (known or (word not in NOT_SEEN and not COUNTED.search(word)))
                and (known or word in LIBRARY_VOCAB or not placeholder(word))   # 小夏/老周/张女士, but 小孩/阿姨
                and not any(len(name) >= 2 and name in word for name in avoid)
                and not (FOREIGN.search(word) and not FOREIGN.search(narration)))


def _library_word(raw: str, avoid, narration: str) -> list[str]:
    """One word as the model wrote it → the words to keep: itself; its usable "/" alternatives; for library
    words written together (男孩玩手机, 情侣争吵, 中年男子摔手机) each of them, since find() wants the whole word
    on one frame and the describer puts other words between (each part still matches the joined text, so this
    only loosens); for a word with a count or a name in it (十六岁男孩, 钱学森讲课), the library words inside
    it; for a phrase around one library word (中年夫妻, 校外同居), the phrase or that word."""
    raw = raw.strip(" \"'“”‘’[]【】()（）《》<>.:：;；!！?？。·")
    if "/" in raw:
        kept = [w for w in dict.fromkeys(_role(w) for w in raw.split("/")) if _usable(w, avoid, narration)]
        return ["/".join(kept[:3])] if kept else []
    word = _role(raw)
    if SEARCH_WORDING.search(word) or (word in NOT_SEEN and not _known(word)):   # 聊天记录 is a screenshot
        return []
    if word in _VOCAB:
        return [word] if _usable(word, avoid, narration) else []
    inside = [w for w in _vocab_words(word) if _usable(w, avoid, narration)]
    if len(inside) >= 2 or not _usable(word, avoid, narration):
        return inside
    return ["/".join([word] + inside)]


def _unique_sets(sets) -> list[list[str]]:
    out, seen = [], set()
    for words in sets:
        words = list(dict.fromkeys(words))[:3]
        if words and frozenset(words) not in seen:
            seen.add(frozenset(words))
            out.append(words)
    return out[:3]


def _as_seen(words: list[str]) -> list[str]:
    """A set with a person: a thing they hold as they are seen with it (男孩 手机 → 男孩 看手机), unless another
    word already says what is done with it (男子 摔手机 手机)."""
    if not any(_kind(w) == "who" for w in words):
        return words
    return [HELD[w] if w in HELD and not any(w in o for o in words if o != w) else w for w in words]


def only_things(words) -> bool:
    """A word set of only things and places, no person and nothing done: searched alone it finds the thing
    itself (手机 phone reviews, 城市 skylines, 卧室 empty rooms), not the line's people."""
    return all(_kind(w) == "thing" for w in words)


def _people_line(narration: str, avoid=()) -> bool:
    return bool(PEOPLE_LINE.search(narration) or any(len(n) >= 2 and n in narration for n in avoid)
                or any(_kind(w) == "who" for w in _vocab_words(narration)))


def library_sets(value, avoid=(), narration: str = "") -> list[list[str]]:
    """The plan's "k" for one shot, cleaned: at most 3 word sets (specific first) of at most 3 words, each
    2-6 Chinese characters, or up to 3 such alternatives joined by "/". Also takes what cheap models send
    instead of a list of strings: lists of lists, or one string with the sets split by commas. Names in
    avoid (real people, a story's stand-ins), numbers, abstract and search words are dropped, and so is a set
    of only things or places when the line has people or figures in it (it would be an empty-room 空镜)."""
    if isinstance(value, str):
        value = re.split(r"[，,;；|｜\n]+", value)
    sets = []
    for item in value if isinstance(value, list) else []:
        words = []
        for part in item if isinstance(item, list) else [item]:
            if isinstance(part, str):
                for raw in re.split(r"[\s、，,+＋]+", part):
                    words += _library_word(raw, avoid, narration)
        sets.append(_as_seen(words))
    if _people_line(narration, avoid):
        sets = [words for words in sets if words and not only_things(words)]
    return _unique_sets(sets)


def _text_sets(text: str, avoid=(), most: int = 3, things: bool = False) -> list[list[str]]:
    """Word sets from a visual or narration text: one person (a pair first) with up to most-1 other library
    words (things done before things seen), then fewer words. A narration line holds several scenes, so its
    words are not asked for together (most=2): 父母 手机 书包 from "parents bought a phone, the boy took his
    schoolbag" is on no frame. Without a person, something done leads; a text of only things and places gives
    nothing (things=True: a visual the user typed, which may ask for a place)."""
    words = [w for w in _vocab_words(text) if _usable(w, avoid, text)]
    who = ([w for w in words if _pair(w)] or [w for w in words if _kind(w) == "who"])[:1]
    other = [w for w in words if _kind(w) == "doing"] + [w for w in words if _kind(w) == "thing"]
    if not who:
        who = [w for w in other if _kind(w) != "thing"][:1] or (other[:1] if things else [])
        other = [w for w in other if w not in who]
    if not who:
        return []
    return _unique_sets(_as_seen(words) for words in (who + other[:most - 1], who + other[:1], who))


def library_words(shot: dict) -> list[list[str]]:
    """Word sets to look a shot up in the local footage library (footage_index.find takes one set), specific
    first: the plan's "k", topped up from the visual text. A plan made before "k" existed gets them from
    the visual alone (the narration when the visual holds no library word). A visual the user typed leads:
    it corrects what the plan read into the narration."""
    express = shot.get("express") if isinstance(shot.get("express"), dict) else {}
    names = {str(p.get("name") or "") for p in shot.get("persons") or [] if isinstance(p, dict)}
    narration = str(shot.get("narration") or "")
    planned = library_sets(express.get("k"), names, narration)
    visual = str(shot.get("visual") or "")
    typed = shot.get("visual_origin") == "manual"
    # A plan's visual is one scene of 6-14 characters; a longer one is the narration standing in for it.
    seen = _text_sets(visual, names, 3 if len(visual) <= 16 else 2, things=typed)
    if typed:
        return _unique_sets(seen + planned + _text_sets(narration, names, 2))
    return _unique_sets(planned + (seen or _text_sets(narration, names, 2)))


# A shot shorter than about this many full-width characters flashes past (about 2 s at 4.5 characters a
# second; Baidu 一镜 reads about 5); such pieces join a neighbour when the line allows.
MIN_WIDTH = 9
# After the real voice is aligned, a shot shorter than this joins a neighbour (before any footage is chosen).
MIN_SECONDS = 2.0
# Setting shot_length: how much narration one shot covers, as full-width characters (about 4.8 a second with
# the default voice): a shot takes subtitle lines until it reaches the first number and never passes the
# second. Similar channels cut every 6-7 s; one line per shot (about 2.5 s) needs 2-3 times the clips,
# so a long script takes that much longer to match and repeats more.
SHOT_WIDTHS = {"long": (28, 44), "medium": (19, 30), "line": None}


def shot_widths(values: dict | None = None):
    from . import config
    choice = str((values if values is not None else config.read()).get("shot_length") or "long")
    return SHOT_WIDTHS.get(choice, SHOT_WIDTHS["long"])


def line_shots(script: str, line_chars: int, widths: tuple[int, int] | None = None) -> list[dict]:
    """Subtitle lines (sentences, split at clauses, then words, when wider than a line), grouped into shots.
    A piece too short to speak for about a second joins its narrower neighbour in the same paragraph
    (lines up to a blank line), as long as the result still fits one subtitle line. With widths
    (target, most) neighbouring lines of a paragraph then share a shot until it is about target characters
    wide; the subtitles still show one line at a time. Without, one line per shot."""
    pieces, paragraph = [], 0
    for line in script.splitlines():
        if not line.strip():
            paragraph += 1
            continue
        for sentence in re.findall(r"[^。！？!?]+[。！？!?]*|[。！？!?]+", line):
            if sentence.strip():
                pieces += [[paragraph, piece] for piece in captions.parts(sentence.strip(), line_chars) if piece.strip()]
    width = subtitle_style.width
    merged = True
    while merged:
        merged = False
        for i, (n, text) in enumerate(pieces):
            if width(text) >= MIN_WIDTH:
                continue
            options = [(width(pieces[j][1]), j) for j in (i - 1, i + 1)
                       if 0 <= j < len(pieces) and pieces[j][0] == n and width(pieces[j][1]) + width(text) <= line_chars]
            if options:
                a, b = sorted((i, min(options)[1]))
                pieces[a][1] += pieces[b][1]
                del pieces[b]
                merged = True
                break
    if widths:
        target, most = widths
        shots = []
        for n, text in pieces:
            if shots and shots[-1][0] == n and width(shots[-1][1]) < target and width(shots[-1][1]) + width(text) <= most:
                shots[-1][1] += text
            else:
                shots.append([n, text])
        pieces = shots
    if len(pieces) > MAX_SHOTS:
        raise ValueError(f"稿件按字幕行拆分后超过 {MAX_SHOTS} 个镜头，请分成多个项目")
    if planner.normalized("".join(p for _, p in pieces)) != planner.normalized(script):
        raise ValueError("文稿包含无法自动切分的结构，请整理为口播正文")
    return [planner.new_shot(text) for _, text in pieces]


def merge_short(project: dict, indices: list[int]) -> None:
    """After real alignment: join each too-short shot to its shorter neighbour. Narration texts are
    concatenated in order (the full audio is unchanged); the kept shot keeps its footage and plan."""
    shots = project["shots"]
    for i in sorted(set(indices), reverse=True):
        if len(shots) < 2 or not 0 <= i < len(shots):
            continue
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(shots)]
        j = min(neighbours, key=lambda k: (subtitle_style.width(shots[k]["narration"]), k))
        a, b = sorted((i, j))
        keep, gone = shots[a], shots[b]
        keep["narration"] += gone["narration"]
        keep["duration"] = round(float(keep["duration"]) + float(gone["duration"]), 3)
        if not keep.get("selected_asset") and gone.get("selected_asset"):
            for key in ("selected_asset", "verification", "frame_crop", "crop_asset", "crop_bottom", "fit", "clip_start", "approved"):
                if key in gone:
                    keep[key] = gone[key]
        people = {p["name"]: {"name": p["name"], "aliases": p.get("aliases", [])}
                  for p in (keep.get("persons") or []) + (gone.get("persons") or [])}
        keep["persons"] = identity.persons(list(people.values())[:3])
        if keep["persons"]:
            keep["evidence_mode"] = "person"
        keep["approved"] = False
        del shots[b]
    info = project.get("plan_info") or {}
    for group in info.get("groups", []):
        group["shots"] = [sid for sid in group.get("shots", []) if any(s["id"] == sid for s in shots)]


OPENING = ("\nS1 是全片开头的钩子，观众看不看下去就看它：S1 单独成一组，检索词写开头这件事最具体、最有冲击力的现场画面"
           "（事件现场、冲突场面、当事人、标志性地点），不要写概念或泛泛的场景。")


def _ask(shots: list[dict], requirements: str, check=None, opening: bool = False) -> tuple[dict, dict]:
    lines = [f"S{i+1}：{shot['narration']}" for i, shot in enumerate(shots)]
    least, most = max(1, math.ceil(len(shots) / 4)), min(20, max(1, len(shots) // 2))
    content = ("用户画面要求：" + (requirements.strip() or "无") + f"\n\n共 {len(shots)} 个镜头，groups 分 {least}-{most} 组。"
               + (OPENING if opening else "") + "\n镜头：\n" + "\n".join(lines))
    # About 70 tokens a shot for its row and its group's queries, plus about 25 for its library words ("k").
    max_tokens = min(12000, 1200 + 130 * len(shots))
    if check:
        check()
    return planner.chat(SYSTEM, content, max_tokens=max_tokens, timeout=150, task="express_plan")


def annotate(shots: list[dict], script: str, requirements: str = "", check=None) -> dict:
    """Mutates shots in place with express fields; returns plan_info. Long scripts are planned in
    batches of about BATCH shots (cut at paragraph-like boundaries); group ids get a batch prefix."""
    batches, start = [], 0
    while start < len(shots):
        end = min(len(shots), start + BATCH)
        if end < len(shots):   # avoid leaving a tiny last batch
            end = min(end, len(shots) - BATCH // 3) if len(shots) - end < BATCH // 3 else end
        batches.append((start, end))
        start = end
    rows, groups, topic, usage, model, domain = {}, {}, {"zh": [], "en": []}, {}, None, ""
    stand_ins = set()
    if len(batches) > 1:   # independent requests: ask them together
        from .execution import ThreadPoolExecutor   # keeps the job's cost scope in the threads
        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as pool:
            answers = list(pool.map(lambda b: _ask(shots[b[0]:b[1]], requirements, check, opening=b[0] == 0), batches))
    else:
        answers = [_ask(shots, requirements, check, opening=True)]
    # One domain word for the whole script (the first batch that names one).
    domain = next((_text(r.get("domain"), 8) for r, _ in answers if isinstance(r, dict) and _text(r.get("domain"), 8)), "")
    # A concrete news event: queries for news reports about it (news_pool cuts their footage where they say the same).
    event = next((_queries(r.get("event"), 40) for r, _ in answers if isinstance(r, dict) and _queries(r.get("event"), 40)), [])
    # Long videos about this very story (highlights, news reports, documentaries): most of the film's footage.
    pool = next((_queries(r.get("pool"), 40) for r, _ in answers if isinstance(r, dict) and _queries(r.get("pool"), 40)), [])
    # People from before photography: no real footage can exist, so portraits and dramatisations stand in.
    ancient = sorted({name for r, _ in answers if isinstance(r, dict) and isinstance(r.get("ancient"), list)
                      for name in (_text(n, 40) for n in r["ancient"]) if len(name) >= 2 and name in script})
    for number, ((begin, end), (result, info)) in enumerate(zip(batches, answers, strict=True)):
        model = model or info.get("model")
        for key, value in (info.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
        prefix = f"B{number+1}" if len(batches) > 1 else ""
        for r in result.get("shots", []) if isinstance(result.get("shots"), list) else []:
            if isinstance(r, dict) and re.fullmatch(r"S\d+", str(r.get("id", "")).strip()):
                local = int(str(r["id"]).strip()[1:])
                if 1 <= local <= end - begin:
                    rows[f"S{begin + local}"] = r | {"g": prefix + str(r.get("g", "")).strip()[:12]}
        for row in result.get("groups", []) if isinstance(result.get("groups"), list) else []:
            if not isinstance(row, dict) or not str(row.get("id", "")).strip():
                continue
            gid = prefix + str(row["id"]).strip()[:12]
            person = _text(row.get("person"), 40)
            if placeholder(person):
                stand_ins.add(person)
            if person and (person not in script or placeholder(person)):
                person = ""
            zh = _queries(row.get("zh"), 40)
            if person:
                zh = [q if person in q else f"{person} {q}" for q in zh] or [f"{person} 历史影像"]
            elif domain:
                zh = [q if domain in q.replace(" ", "") else f"{domain} {q}" for q in zh]
            groups[gid] = {"id": gid, "zh": zh, "en": [] if person else _queries(row.get("en"), 60, ascii_only=True),
                           "person": person, "shots": []}
        found = result.get("topic") if isinstance(result.get("topic"), dict) else {}
        if not topic["zh"]:
            topic = {"zh": _queries(found.get("zh"), 40), "en": _queries(found.get("en"), 60, ascii_only=True)}
    for i, shot in enumerate(shots):
        row = rows.get(f"S{i+1}", {})
        gid = str(row.get("g", "")).strip()[:16]
        if gid not in groups:
            gid = f"X{i+1}"
            groups[gid] = {"id": gid, "zh": [_text(shot["narration"], 30)], "en": [], "person": "", "shots": []}
        group = groups[gid]
        names = []
        for name in row.get("p", []) if isinstance(row.get("p"), list) else []:
            name = _text(name, 40)
            if placeholder(name):
                stand_ins.add(name)
            elif 2 <= len(name) and name in shot["narration"] and name not in names:
                names.append(name)
        if group["person"] and group["person"] in shot["narration"] and group["person"] not in names:
            names.insert(0, group["person"])
        visual = _text(row.get("v"), 60) or _text(shot["narration"], 60)
        group["shots"].append(shot["id"])
        shot.update(
            visual=visual, visual_origin="express", query=(group["zh"] or [visual])[0],
            search_queries=list(group["zh"]), persons=identity.persons([{"name": n} for n in names[:3]]),
            evidence_mode="person" if names else "generic", event="", event_scope="exact",
            express={"group": gid, "en": list(group["en"]), "zh": list(group["zh"])})
    known = {g["person"] for g in groups.values() if g["person"]}
    known |= {p["name"] for shot in shots for p in shot.get("persons") or []}
    for shot in shots:
        names = [p["name"] for p in shot.get("persons") or []]
        missing = [n for n in sorted(known, key=len, reverse=True) if n in shot["narration"] and n not in names]
        if missing:
            shot["persons"] = identity.persons([{"name": n} for n in (names + missing)[:3]])
            shot["evidence_mode"] = "person"
    groups = {gid: g for gid, g in groups.items() if g["shots"]}
    for group in groups.values():
        # A story's nicknames found by the model, and any 张女士/王某 form, are not words to search for.
        kept = [" ".join(w for w in q.split() if w not in stand_ins and not placeholder(w, person=False))
                for q in group["zh"]]
        group["zh"] = [q for q in kept if q] or group["zh"]
    for shot in shots:
        group = groups.get(shot["express"]["group"])
        if group:
            shot.update(query=(group["zh"] or [shot["visual"]])[0], search_queries=list(group["zh"]))
            shot["express"]["zh"] = list(group["zh"])
    # Local library words: no names (the describer never writes them; a stand-in's name only finds strangers).
    avoid = known | stand_ins
    for i, shot in enumerate(shots):
        shot["express"]["k"] = library_sets(rows.get(f"S{i+1}", {}).get("k"), avoid, shot["narration"])
    if not topic["zh"]:
        topic["zh"] = [q for g in groups.values() if not g["person"] for q in g["zh"]][:2]
    if domain:
        topic["zh"] = [q if domain in q.replace(" ", "") else f"{domain} {q}" for q in topic["zh"]]
    return {"mode": "express", "groups": list(groups.values()), "topic": topic, "domain": domain, "ancient": ancient,
            "event": event, "pool": pool, "model": model, "usage": usage, "batches": len(batches)}


def plan(project: dict, check=None) -> tuple[list[dict], dict]:
    """Normalise the script (script_norm: punctuation, spoken numbering, obvious typos, one sentence per
    line), then subtitle lines grouped into shots (setting shot_length). info["script"] is the standard text
    narration will read."""
    from . import script_norm
    script = project.get("script", "").strip()
    if not script:
        raise ValueError("请先输入文稿")
    script, normalized = script_norm.normalize(script, check)
    shots = line_shots(script, subtitle_style.line_chars_for(project.get("aspect", "16:9")), shot_widths())
    info = annotate(shots, script, project.get("requirements", ""), check)
    return shots, info | {"script": script, "normalize": normalized}
