"""Editorial retrieval seeds, not a claim that these people/events have footage.

Entity associations mean a name occurs in source text, never facial recognition
or confirmation that a particular shot depicts the person/event.
"""
import hashlib
import re
from functools import lru_cache

EXTRA_TOPICS = {
    "互联网人物": "互联网人物访谈|内容创作者|视频博主|直播电商|知识博主|科技博主|美食博主|旅行博主|网络文化|短视频创作|自媒体创业|网络热点",
    "互联网事件": "互联网大会|科技产品发布会|电商购物节|直播带货现场|网络公益|互联网辟谣|数字经济大会|电子竞技赛事|人工智能大会|网络安全宣传|网红城市|互联网创业",
    "游戏电竞": "电子竞技|游戏比赛|游戏开发|游戏展会|王者荣耀比赛|英雄联盟比赛|黑神话悟空|国产游戏|电竞选手采访|游戏主机|游戏配音|游戏美术",
    "动漫二次元": "国漫创作|动画制作|动漫展|漫画创作|配音演员|模型手办|动画电影幕后|国风动画|动漫舞台|漫画展览|动漫产业|动画设计",
    "时尚美妆": "服装设计|时装秀|日常穿搭|化妆教学|护肤科普|美发造型|服装工厂|珠宝设计|国货化妆品|美甲制作|汉服展示|美妆测评",
    "母婴育儿": "亲子游戏|儿童阅读|婴儿护理|幼儿教育|儿童健康|孕期科普|儿童安全|亲子运动|婴幼儿辅食|家庭育儿|儿童游乐|亲子旅行",
    "搞笑喜剧": "小品表演|相声演出|喜剧演员访谈|脱口秀现场|喜剧电影幕后|幽默表演|曲艺演出|喜剧综艺|街头表演|杂技演出|民间趣事|喜剧创作",
    "影视幕后": "电影拍摄现场|电视剧幕后|导演访谈|编剧访谈|影视特效制作|摄影棚|电影配乐|电影节现场|纪录片拍摄|电影美术|演员排练|电影首映礼",
    "音乐演出": "演唱会现场|歌手访谈|乐队排练|交响乐演出|钢琴演奏|吉他演奏|民族乐器|音乐节|音乐制作|合唱演出|街头音乐|京剧演唱",
    "名人动态": "明星采访|艺人活动|演员发布会|运动员颁奖|作家新书发布|企业家演讲|科学家讲座|导演见面会|艺术家访谈|公众人物纪录片|歌手排练|名人公益",
    "法律社会": "法治宣传|法院开放日|法律科普|消费者权益|反诈宣传|交通执法|劳动权益|社区调解|普法课堂|知识产权|食品安全|个人信息保护",
    "国际见闻": "国际城市街景|世界文化遗产|国际展会|国际贸易|海外华人|国际体育交流|各国美食|世界博物馆|国际航运|世界建筑|国际科技展|世界地理",
    "人文阅读": "作家访谈|读书分享|文学讲座|图书出版|书店日常|诗歌朗诵|哲学讲座|文学纪录片|文艺评论|历史讲座|文化沙龙|人文纪实",
    "手工收藏": "手工制作|木工制作|模型制作|手工皮具|文物修复|手工编织|钟表修复|收藏展览|金属加工|手工雕刻|竹编工艺|玩具制作",
    "运动户外": "露营徒步|登山攀岩|公路骑行|马拉松比赛|户外装备|滑板运动|冲浪运动|钓鱼活动|滑雪训练|帆船运动|健身训练|越野跑",
    "房产消费": "房屋装修|看房现场|商场购物|二手交易|家具制造|家电评测|消费趋势|社区配套|楼市观察|家居设计|旧房改造|产品测评",
}

# Canonical names plus conservative public aliases. Ambiguous nickname-only
# entities (e.g. 老罗/大鹏/小杨哥) are deliberately not used as identity evidence.
PEOPLE = {
    "互联网人物": "罗永浩|董宇辉|李子柒|李佳琦|薇娅|辛有志=辛巴|张大仙|陈翔|张同学|何同学=老师好我叫何同学|毕导|李永乐|罗翔|戴建业|张雪峰|温铁军|储殷|王骁|陈睿|黄仁勋|张朝阳|周鸿祎|罗振宇|吴晓波|司马南|李诞|王尼玛|敬汉卿|朱一旦|张踩铃|房琪|詹姆士|王刚|任芸丽|李永钦",
    "科技商业": "马云|马化腾|雷军|任正非|余承东|孟晚舟|刘强东|王兴|张一鸣|梁文锋|李彦宏|丁磊|黄峥|段永平|董明珠|宗庆后|宗馥莉|王传福|李书福|曾毓群|何小鹏|李想|李斌|俞敏洪|张勇|刘永好|曹德旺|王健林|王石|潘石屹|张瑞敏|柳传志|杨元庆|李东生|梁建章|傅盛|程维|陈天桥|史玉柱|埃隆·马斯克=埃隆马斯克,马斯克|史蒂夫·乔布斯=乔布斯|比尔·盖茨=比尔盖茨|蒂姆·库克=蒂姆库克",
    "体育赛事": "马龙|张继科|樊振东|许昕|王楚钦|孙颖莎|陈梦|王曼昱|丁宁|刘诗雯|刘国梁|邓亚萍|郭跃|孙杨|潘展乐|汪顺|张雨霏|覃海洋|全红婵|陈芋汐|郭晶晶|吴敏霞|刘翔|苏炳添|巩立姣|王军霞|谷爱凌|苏翊鸣|武大靖|王濛|郎平|朱婷|张常宁|惠若琪|姚明|易建联|郭艾伦|林书豪|郑钦文|李娜|张帅|林丹|李宗伟|谌龙|石宇奇|黄雅琼|郑思维|武磊|孙继海|范志毅|梅西|克里斯蒂亚诺·罗纳尔多=C罗|勒布朗·詹姆斯=勒布朗詹姆斯|斯蒂芬·库里=斯蒂芬库里|科比·布莱恩特=科比",
    "影视文艺": "成龙|李连杰|周星驰|周润发|刘德华|梁朝伟|张学友|郭富城|黎明|张国荣|张曼玉|林青霞|巩俐|章子怡|周迅|舒淇|汤唯|刘亦菲|赵丽颖|杨幂|杨紫|迪丽热巴|孙俪|海清|闫妮|宋佳|马伊琍|秦海璐|咏梅|张颂文|张译|王宝强|黄渤|徐峥|沈腾|马丽|贾玲|吴京|胡歌|陈道明|陈宝国|李雪健|葛优|张国立|于和伟|靳东|雷佳音|易烊千玺|王俊凯|王源|肖战|王一博|黄晓明|邓超|孙红雷|黄磊|沙溢|张艺谋|陈凯歌|冯小刚|贾樟柯|宁浩|徐克|王家卫|李安|陈思诚|文牧野|郭帆|导演饺子=饺子导演",
    "音乐演出": "周杰伦|林俊杰|陈奕迅|王菲|那英|韩红|孙燕姿|梁静茹|蔡依林|张惠妹|邓紫棋|张韶涵|李宇春|张靓颖|周笔畅|毛不易|周深|薛之谦|许嵩|汪苏泷|胡彦斌|陶喆|王力宏|刘欢|韩磊|谭维维|单依纯|腾格尔|崔健|窦唯|许巍|朴树|汪峰|郑钧|赵雷|李健|李宗盛|罗大佑|齐秦|张信哲|任贤齐|张杰|华晨宇|郎朗|李云迪|谭盾|戴玉强|廖昌永",
    "人文科学": "袁隆平|屠呦呦|钟南山|张伯礼|张文宏|李兰娟|钱学森|邓稼先|钱三强|钱伟长|于敏|孙家栋|吴文俊|王选|南仁东|黄大年|黄旭华|潘建伟|施一公|颜宁|饶毅|谢晓亮|杨振宁|李政道|丘成桐|陶哲轩|杨利伟|聂海胜|景海鹏|刘洋|王亚平|桂海潮|朱光亚|叶嘉莹|余华|莫言|刘慈欣|王安忆|迟子建|铁凝|贾平凹|苏童|阿来|梁晓声|毕飞宇|余秋雨|钱钟书|杨绛|季羡林|金庸|古龙|琼瑶|汪曾祺|林清玄|三毛|史铁生|余光中|白先勇",
    "历史人物": "孔子|孟子|老子|庄子|屈原|秦始皇|汉武帝|司马迁|曹操|刘备|孙权|诸葛亮|李世民|武则天|李白|杜甫|白居易|王维|苏轼|王安石|司马光|岳飞|成吉思汗|忽必烈|朱元璋|郑和|于谦|王阳明|戚继光|李时珍|徐霞客|康熙|雍正|乾隆|林则徐|曾国藩|李鸿章|左宗棠|张之洞|康有为|梁启超|谭嗣同|孙中山|宋庆龄|鲁迅|胡适|蔡元培|陈独秀|李大钊|毛泽东|周恩来|刘少奇|朱德|邓小平|彭德怀|陈毅|叶剑英|聂荣臻|徐向前|罗荣桓|粟裕|张学良|蒋介石|冯玉祥|张自忠|李宗仁|白崇禧|钱穆|陈寅恪|傅斯年|梅贻琦|闻一多",
    "公共人物": "习近平|李强|赵乐际|王沪宁|蔡奇|丁薛祥|李希|王毅|华春莹|汪文斌|林剑|耿爽|傅聪|刘建超|张桂梅|黄文秀|支月英|张富清|雷锋|焦裕禄|孔繁森|杨善洲|白岩松|康辉|撒贝宁|董卿|敬一丹|水均益|王志|陈鲁豫|窦文涛|许知远|柴静|王冰冰|尼格买提|欧阳夏丹|海霞|李梓萌",
}

# Dated event seeds describe what to look up, not asserted video dates. Exact
# dates and participants still come from the selected original publication.
EVENTS = {
    "互联网事件": "2010年3Q大战|2014年阿里巴巴上市|2018年小米上市|2020年罗永浩直播首秀|2021年鸿星尔克捐款|2023年淄博烧烤|2024年哈尔滨冰雪旅游|2024年小米SU7发布会|2024年黑神话悟空上线|2025年DeepSeek发布|2025年哪吒2上映|2025年世界人工智能大会|东方甄选双语直播|李子柒非遗视频|董宇辉与辉同行|华为Mate60发布|小米汽车工厂|乌镇世界互联网大会",
    "体育赛事": "2008年北京奥运会开幕式|2022年北京冬奥会开幕式|2023年杭州亚运会|2024年巴黎奥运会马龙|2024年巴黎奥运会樊振东|2024年巴黎奥运会孙颖莎|2024年巴黎奥运会全红婵|2024年巴黎奥运会郑钦文|2024年巴黎奥运会潘展乐|2021年东京奥运会乒乓球|2022年世界杯阿根廷|2023年成都大运会",
    "航天国防": "2003年神舟五号发射|2008年神舟七号出舱|2020年嫦娥五号采样返回|2021年祝融号火星车|2022年中国空间站建成|2024年嫦娥六号月背采样|2019年国庆阅兵|2024年珠海航展",
    "历史档案": "1911年辛亥革命|1919年五四运动|1936年西安事变|1937年七七事变|1945年抗战胜利|1949年开国大典|1978年改革开放|1997年香港回归|1999年澳门回归|2001年中国加入世贸组织|2008年汶川地震救援|2020年武汉抗疫",
}


@lru_cache(maxsize=1)
def entities():
    result = {}
    for kind, groups in (("person", PEOPLE), ("event", EVENTS)):
        for domain, names in groups.items():
            for entry in names.split("|"):
                name, _, aliases = entry.partition("=")
                key = hashlib.sha256((kind+":"+name).encode()).hexdigest()[:16]
                result[key] = {"id": key, "kind": kind, "name": name, "domain": domain,
                               "aliases": [a for a in aliases.split(",") if a]}
    return result


@lru_cache(maxsize=1)
def matcher():
    names = {}
    for entity in entities().values():
        for name in [entity["name"], *entity["aliases"]]:
            names.setdefault(name, []).append(entity["id"])
    return re.compile("|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))), names


@lru_cache(maxsize=1)
def event_terms():
    result = []
    for entity in entities().values():
        if entity["kind"] != "event":
            continue
        dated = re.fullmatch(r"(\d{4})年(.+)",entity["name"])
        if not dated:
            continue
        subject = dated[2]
        # A date elsewhere in source text is a retrieval clue only. Split an
        # explicit person suffix so "马龙…2024…巴黎奥运会" can be found too.
        suffix = next((e["name"] for e in entities().values() if e["kind"] == "person"
                       and subject.endswith(e["name"]) and subject != e["name"]), "")
        terms = [subject[:-len(suffix)],suffix] if suffix else [subject]
        result.append((entity["id"],[dated[1],*terms]))
    return result


def mentions(text):
    pattern, names = matcher()
    found = {(eid, m[0]) for m in pattern.finditer(text) for eid in names[m[0]]}
    compact = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    for eid,terms in event_terms():
        if all(term in compact for term in terms):
            found.add((eid, " + ".join(terms)))
    return found


def queries():
    groups = {}
    for entity in entities().values():
        # Bare names work better than artificial event strings. The collector
        # links only mentions actually found in returned source text.
        topic = entity["domain"] if entity["domain"] in EXTRA_TOPICS else {
            "科技商业": "商业财经", "影视文艺": "娱乐文艺", "人文科学": "人物访谈",
            "历史人物": "历史档案", "公共人物": "时政公共事务",
        }.get(entity["domain"], entity["domain"])
        groups.setdefault((entity["kind"],entity["domain"]),[]).append((topic,entity["name"],entity["kind"]))
    # A partial pass should reach events and every profession early, instead
    # of exhausting hundreds of entertainers before its first dated event.
    for i in range(max(map(len,groups.values()))):
        for rows in groups.values():
            if i<len(rows):
                yield rows[i]
