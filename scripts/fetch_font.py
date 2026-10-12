from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[1] / 'assets' / 'fonts'
root.mkdir(parents=True, exist_ok=True)
items = {
    'NotoSansSC-Regular.otf': 'https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/SC/NotoSansSC-Regular.otf',
    'LICENSE': 'https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/LICENSE',
}
for name, url in items.items():
    target = root / name
    if not target.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        target.write_bytes(data)
    print(name, target.stat().st_size)
