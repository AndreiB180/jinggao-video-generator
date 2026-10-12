from PIL import Image, ImageDraw, ImageFont

from . import media, paths, planner, store

SCRIPT = "无缝钢轨并不意味着钢轨不会热胀冷缩。铁路工程通过锁定轨温、扣件和轨枕，控制钢轨的位移。日常巡检与应力管理，共同保障线路安全。"


def create() -> dict:
    p = store.new_project("无缝钢轨的秘密 · 操作示例", SCRIPT, media_policy="mixed")
    rows = [
        ("无缝钢轨并不意味着钢轨不会热胀冷缩。", "热胀冷缩", "温度变化 · 长度与应力", "#f0a067"),
        ("铁路工程通过锁定轨温、扣件和轨枕，控制钢轨的位移。", "约束与连接", "钢轨 / 扣件 / 轨枕", "#82c7b3"),
        ("日常巡检与应力管理，共同保障线路安全。", "持续监测", "巡检 · 维护 · 应力管理", "#8aabdd"),
    ]
    assets, shots = [], []
    folder = paths.project_dir(p["id"]) / "assets"
    folder.mkdir(exist_ok=True)
    font = str(paths.font_path())
    for i, (text, title, subtitle, color) in enumerate(rows):
        im = Image.new("RGB", (1600, 900), "#121b26")
        d = ImageDraw.Draw(im)
        for x in range(0, 1600, 60):
            d.line((x, 0, x, 900), fill="#1b2736")
        for y in range(0, 900, 60):
            d.line((0, y, 1600, y), fill="#1b2736")
        d.text((90, 60), f"JINGGAO  /  ENGINEERING NOTES  0{i+1}", fill=color,
               font=ImageFont.truetype(font, 26))
        d.text((90, 170), title, fill="#f0f2f4", font=ImageFont.truetype(font, 80))
        d.text((95, 288), subtitle, fill="#a9b7c8", font=ImageFont.truetype(font, 32))
        for x in range(150, 1490, 110):
            d.rounded_rectangle((x, 450, x+55, 670), radius=8, fill="#3b4858")
        for y in (490, 600):
            d.rounded_rectangle((90, y, 1490, y+25), radius=8, fill=color)
        d.text((95, 720), "工程概念示意 · 内置示例素材，非现场影像", fill="#718398", font=ImageFont.truetype(font, 24))
        target = folder / f"demo-{i}.jpg"
        im.save(target, quality=92)
        asset = media.inspect_asset(p["id"], target, title+"（工程示意图）", provider="demo",
                                    license="本项目原创示意图", description=text)
        shot = planner.new_shot(text, visual=title+"的工程示意", query=title+" 钢轨 中国铁路")
        shot.update(selected_asset=asset["id"], approved=True)
        assets.append(asset)
        shots.append(shot)
    return store.mutate(p["id"], lambda item: item.update(assets=assets, shots=shots,
                        plan_info={"method": "demo", "message": "内置操作示例，不代表自动检索效果"}))
