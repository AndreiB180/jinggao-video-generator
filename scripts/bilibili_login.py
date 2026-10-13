"""Bilibili app QR login for 720p/1080p footage access. Cookies stay in data/ (gitignored).

Usage: .venv/bin/python scripts/bilibili_login.py [--png PATH] [--minutes 30]
Regenerates the QR every ~150 s until scanned and confirmed in the Bilibili app.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jinggao import express_sources as es  # noqa: E402


def terminal_qr(url: str) -> str:
    import qrcode
    qr = qrcode.QRCode(border=2)
    qr.add_data(url)
    qr.make()
    rows = qr.get_matrix()
    lines = []
    for y in range(0, len(rows), 2):
        line = ""
        for x in range(len(rows[0])):
            top, bottom = rows[y][x], rows[y + 1][x] if y + 1 < len(rows) else False
            line += "█" if top and bottom else "▀" if top else "▄" if bottom else " "
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", default=str(Path.home() / "Desktop" / "bilibili-login-qr.png"))
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()
    deadline = time.time() + args.minutes * 60
    while time.time() < deadline:
        start = es.qr_login_start()
        Path(args.png).write_bytes(start["png"])
        print(f"QR saved: {args.png}\nlogin url: {start['url']}\n{terminal_qr(start['url'])}", flush=True)
        if not args.no_open and sys.platform == "darwin":
            subprocess.run(["open", args.png], check=False)
        began = time.time()
        while time.time() - began < 150 and time.time() < deadline:
            result = es.qr_login_poll(start["key"])
            if result["status"] == "success":
                print("LOGIN_OK", es.login_status(), flush=True)
                Path(args.png).unlink(missing_ok=True)
                return
            if result["status"] == "expired":
                break
            if result["status"] == "scanned":
                print("scanned, waiting for confirmation in app", flush=True)
            time.sleep(2)
        print("QR expired, regenerating", flush=True)
    print("LOGIN_TIMEOUT", flush=True)


if __name__ == "__main__":
    main()
