"""Download the subtitle fonts (all SIL Open Font License 1.1) into assets/fonts."""
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[1] / 'assets' / 'fonts'
root.mkdir(parents=True, exist_ok=True)
NOTO = 'https://raw.githubusercontent.com/notofonts/noto-cjk/main'
GOOGLE = 'https://raw.githubusercontent.com/google/fonts/main/ofl'
items = {
    'NotoSansSC-Regular.otf': f'{NOTO}/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf',
    'NotoSansSC-Black.otf': f'{NOTO}/Sans/SubsetOTF/SC/NotoSansSC-Black.otf',
    'LICENSE': f'{NOTO}/Sans/LICENSE',
    'NotoSerifSC-Black.otf': f'{NOTO}/Serif/SubsetOTF/SC/NotoSerifSC-Black.otf',
    'LICENSE-NotoSerifSC.txt': f'{NOTO}/Serif/LICENSE',
    'ZCOOLQingKeHuangYou-Regular.ttf': f'{GOOGLE}/zcoolqingkehuangyou/ZCOOLQingKeHuangYou-Regular.ttf',
    'OFL-ZCOOLQingKeHuangYou.txt': f'{GOOGLE}/zcoolqingkehuangyou/OFL.txt',
    'ZCOOLKuaiLe-Regular.ttf': f'{GOOGLE}/zcoolkuaile/ZCOOLKuaiLe-Regular.ttf',
    'OFL-ZCOOLKuaiLe.txt': f'{GOOGLE}/zcoolkuaile/OFL.txt',
}
for name, url in items.items():
    target = root / name
    if not target.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        target.write_bytes(data)
    print(name, target.stat().st_size)
