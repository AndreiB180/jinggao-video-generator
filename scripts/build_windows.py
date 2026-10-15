"""Build the Windows x64 portable distribution from macOS/Linux/Windows.

Requires Python 3.11+, uv, and a prebuilt frontend/dist. No host secrets/data are copied.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "build" / "downloads"
OUTPUT = ROOT / "build" / "Jinggao-Windows-x64"
PYTHON_URL = "https://www.python.org/ftp/python/3.13.12/python-3.13.12-embed-amd64.zip"
# FFmpeg 7.1 built against NVIDIA Video Codec SDK 11.1 headers: NVENC works with drivers 471.41+ (mid-2021).
# The gyan.dev 9.x builds need driver 610+, so most customers' cards silently fell back to the CPU.
FFMPEG_URL = ("https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-06-30-13-34/"
              "ffmpeg-n7.1.5-1-g7d0e842004-win64-gpl-7.1.zip")
FFMPEG_SHA256 = "405b190f746db40539eb453967f72c0e69d8bf260b10ceff36e0c2149a9ad22f"
YTDLP_BASE = "https://github.com/yt-dlp/yt-dlp/releases/download/2026.08.19/"
YTDLP_SHA256 = "66674953fe251b89f4d08c5f0e35e0728679bd67ab3d7d05c0562af101dd3e7a"
DENO_BASE = "https://github.com/denoland/deno/releases/download/v2.9.7/"
DENO_NAME = "deno-x86_64-pc-windows-msvc.zip"
DENO_SHA256 = "a0c3101b4158d1dfb7d6a78a7bf0f3de80c96bb423c152beec8beb22786f2238"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fetch(url, filename=None):
    path = CACHE / (filename or url.rsplit("/", 1)[1])
    if path.exists():
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    print("Downloading", path.name, flush=True)
    with urllib.request.urlopen(url, timeout=60) as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst, 1024*1024)
    temporary.replace(path)
    return path


def prepare_youtube(runtime):
    """Ship official public-video metadata tools, with pinned and vendor hashes."""
    executable = fetch(YTDLP_BASE + "yt-dlp.exe")
    sums = fetch(YTDLP_BASE + "SHA2-256SUMS").read_text()
    vendor_hash = next(line.split()[0] for line in sums.splitlines() if line.split()[-1].lstrip("*") == "yt-dlp.exe")
    if digest(executable) != YTDLP_SHA256 or vendor_hash != YTDLP_SHA256:
        raise RuntimeError("yt-dlp checksum mismatch")
    deno_archive = fetch(DENO_BASE + DENO_NAME)
    deno_hashes = re.findall(r"\b[0-9a-fA-F]{64}\b", fetch(DENO_BASE + DENO_NAME + ".sha256sum").read_text())
    deno_vendor = deno_hashes[0].lower() if len(deno_hashes) == 1 else ""
    if digest(deno_archive) != DENO_SHA256 or deno_vendor != DENO_SHA256:
        raise RuntimeError("Deno checksum mismatch")
    yt, deno = runtime / "yt-dlp", runtime / "deno"
    yt.mkdir(exist_ok=True)
    deno.mkdir(exist_ok=True)
    shutil.copy2(executable, yt / "yt-dlp.exe")
    with zipfile.ZipFile(deno_archive) as archive:
        (deno / "deno.exe").write_bytes(archive.read("deno.exe"))
    for url, name, folder in [
        ("https://raw.githubusercontent.com/yt-dlp/yt-dlp/2026.08.19/LICENSE", "yt-dlp-LICENSE.txt", yt),
        ("https://raw.githubusercontent.com/yt-dlp/yt-dlp/2026.08.19/THIRD_PARTY_LICENSES.txt", "yt-dlp-THIRD_PARTY_LICENSES.txt", yt),
        ("https://raw.githubusercontent.com/denoland/deno/v2.9.7/LICENSE.md", "deno-LICENSE.txt", deno),
    ]:
        shutil.copy2(fetch(url, name), folder / name)
    return {"yt-dlp": {"url": YTDLP_BASE + "yt-dlp.exe", "sha256": YTDLP_SHA256,
                       "vendor_checksum": YTDLP_BASE + "SHA2-256SUMS"},
            "deno": {"url": DENO_BASE + DENO_NAME, "sha256": DENO_SHA256,
                     "vendor_checksum": DENO_BASE + DENO_NAME + ".sha256sum"}}


def prepare(requirements=ROOT / "requirements.lock"):
    runtime = OUTPUT / "runtime"
    wheels = CACHE / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    if re.search(r"(?m)^jieba==0\.42\.1\s*$", requirements.read_text()):
        # Jieba ships a pure-Python sdist. Build its platform-independent wheel
        # on the host; compiled dependencies must still be Windows wheels.
        if not list(wheels.glob("jieba-0.42.1-*-none-any.whl")):
            subprocess.run(["uvx", "--from", "pip", "pip", "wheel", "--no-deps",
                            "--wheel-dir", str(wheels), "jieba==0.42.1"], check=True)
        if not list(wheels.glob("jieba-0.42.1-*-none-any.whl")):
            raise RuntimeError("Jieba did not produce a platform-independent wheel")
    python_zip = fetch(PYTHON_URL)
    ffmpeg_zip = fetch(FFMPEG_URL)
    checksum = FFMPEG_SHA256
    if digest(ffmpeg_zip) != checksum:
        raise RuntimeError("FFmpeg checksum mismatch. Clear build/downloads/ffmpeg* and rebuild.")
    py = runtime / "python"
    if not (py / "python.exe").exists():
        py.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(python_zip) as z:
            z.extractall(py)
    # Embedded CPython normalizes parent segments correctly with Windows separators.
    (py / "python313._pth").write_text("python313.zip\n.\nLib\\site-packages\n..\\..\nimport site\n", "utf-8")
    target = py / "Lib" / "site-packages"
    marker = target / ".jinggao-requirements.sha256"
    req_hash = digest(requirements)
    if not marker.exists() or marker.read_text() != req_hash:
        if target.exists():
            shutil.rmtree(target)
        subprocess.run(["uv", "pip", "install", "--target", str(target), "--python-version", "3.13",
                        "--python-platform", "x86_64-pc-windows-msvc", "--only-binary", ":all:",
                        "--find-links", str(CACHE / "wheels"),
                        "-r", str(requirements)], check=True)
        marker.write_text(req_hash)
    # The Windows fbank wheel keeps its dependency beside its Python package.
    # Embedded Python needs that explicit DLL directory for extension imports.
    (target / "sitecustomize.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "_jinggao_dll_handles = []\n"
        "if os.name == 'nt':\n"
        "    _directory = Path(__file__).parent / 'kaldi_native_fbank' / 'lib'\n"
        "    if _directory.is_dir():\n"
        "        _jinggao_dll_handles.append(os.add_dll_directory(str(_directory)))\n", encoding="utf-8")
    ff = runtime / "ffmpeg"
    source = ff / "source.txt"
    current = source.read_text("utf-8").strip() if source.exists() else ""
    if not (ff / "ffmpeg.exe").exists() or current != FFMPEG_URL:
        shutil.rmtree(ff, ignore_errors=True)       # a runtime prepared with another FFmpeg build is replaced
        ff.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ffmpeg_zip) as z:
            for name in z.namelist():
                if name.endswith(("/bin/ffmpeg.exe", "/bin/ffprobe.exe", "/LICENSE.txt", "/LICENSE", "/README.txt")):
                    (ff / Path(name).name).write_bytes(z.read(name))
        (ff / "source.txt").write_text(FFMPEG_URL + "\n", "utf-8")
    manifest = {"python": {"url": PYTHON_URL, "sha256": digest(python_zip)},
                "ffmpeg": {"url": FFMPEG_URL, "sha256": checksum,
                           "nvenc_minimum_driver": "471.41"}}
    manifest.update(prepare_youtube(runtime))
    (runtime / "downloads.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    print("Windows runtime ready:", runtime, flush=True)


def load_release(path):
    """Validate a frozen runtime before touching the output directory."""
    release = json.loads(path.read_text())
    source = Path(release["runtime"]).resolve()
    manifest = release["manifest"]
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Release manifest is empty")
    for rel, expected in manifest.items():
        relative = Path(rel)
        file = (source / relative).resolve()
        if relative.is_absolute() or not file.is_relative_to(source) or digest(file) != expected:
            raise ValueError("Release file changed or escaped its root: " + rel)
    required = {"launch.py", "jinggao/local_asr.py", "scripts/sapi.ps1", "frontend/dist/index.html"}
    if not required <= manifest.keys():
        raise ValueError("Release lacks required application files")
    return source, manifest


# Shipped asset folders: subtitle fonts, voice samples (试听) and the footage index seed (素材库).
ASSET_FOLDERS = ("assets/fonts/", "assets/voices/", "assets/footage-index/")


def app_path(rel):
    return (rel in {"launch.py", "scripts/sapi.ps1", "assets/source-index.jsonl.gz"}
            or rel.startswith(("jinggao/", "frontend/dist/") + ASSET_FOLDERS))


def copy_release(source, manifest):
    """Only declared application files; never development code or caches."""
    for folder in ("jinggao", "frontend/dist", *(f.rstrip("/") for f in ASSET_FOLDERS)):
        dst = OUTPUT / folder
        if dst.exists():
            shutil.rmtree(dst)
    copied = {}
    for rel, expected in manifest.items():
        if not app_path(rel):
            continue
        dst = OUTPUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, dst)
        if digest(dst) != expected:
            raise ValueError("Packaged release file differs: " + rel)
        copied[rel] = expected
    return copied


def bundle_captions(source, model_directory):
    """Copy only three public, hash-pinned weights, never the surrounding data."""
    tree = ast.parse((source / "jinggao/local_asr.py").read_text())
    files = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "FILES" for t in n.targets))
    if set(files) != {"model_quant.onnx", "am.mvn", "tokens.json"}:
        raise ValueError("Unexpected subtitle model manifest")
    for name, expected in files.items():
        if digest(model_directory / name) != expected:
            raise ValueError("Subtitle model checksum mismatch: " + name)
    copied = {}
    for name, expected in files.items():
        rel = "data/models/paraformer-zh-timestamp/" + name
        dst = OUTPUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_directory / name, dst)
        if digest(dst) != expected:
            raise ValueError("Packaged subtitle model differs: " + name)
        copied[rel] = expected
    return copied


def check_private_files(files, public_weights):
    forbidden = {"settings.json", "state.db", ".env", "projects.db", "jinggao.sqlite3",
                 "model-costs.sqlite3", "Cookies", "Login Data"}
    for path in files:
        rel = path.relative_to(OUTPUT).as_posix()
        if path.name in forbidden or path.relative_to(OUTPUT).parts[0] == "data":
            if rel not in public_weights or digest(path) != public_weights[rel]:
                raise RuntimeError("Private data found in distribution: " + str(path))


def package(release_path=None, model_directory=None, requirements=ROOT / "requirements.lock", bundle_pixabay=False):
    app_manifest, model_manifest = {}, {}
    if release_path:
        source, manifest = load_release(release_path)
        app_manifest = copy_release(source, manifest)
        if model_directory is None:
            raise ValueError("A release build needs --model-directory for offline subtitle alignment")
        model_manifest = bundle_captions(source, model_directory)
    else:
        for folder in ("jinggao", "frontend/dist", *(f.rstrip("/") for f in ASSET_FOLDERS)):
            dst = OUTPUT / folder
            if dst.exists():
                shutil.rmtree(dst)
            if (ROOT / folder).is_dir():
                shutil.copytree(ROOT / folder, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        seed = ROOT / "assets" / "source-index.jsonl.gz"
        if seed.is_file():
            shutil.copy2(seed, OUTPUT / "assets" / seed.name)
        (OUTPUT / "scripts").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "scripts" / "sapi.ps1", OUTPUT / "scripts" / "sapi.ps1")
        shutil.copy2(ROOT / "launch.py", OUTPUT / "launch.py")
        if model_directory:
            # Subtitle alignment works offline from the first start (no model download on the customer's PC).
            model_manifest = bundle_captions(ROOT, model_directory)
    defaults = OUTPUT / "defaults.json"
    defaults.unlink(missing_ok=True)
    if bundle_pixabay:
        # The vendor's free Pixabay key (config.BUNDLED_KEYS), read from this build machine's settings.
        import sys
        sys.path.insert(0, str(ROOT))
        from jinggao import config
        key = config.read().get("pixabay_api_key", "").strip()
        if not key:
            raise RuntimeError("--bundle-pixabay-key: this machine has no Pixabay key in its settings")
        defaults.write_text(json.dumps({"pixabay_api_key": key}), "utf-8")
        print("Bundled the Pixabay key (defaults.json)", flush=True)
    for name in ("README.md", "THIRD_PARTY_NOTICES.md", "先看这里.txt"):
        shutil.copy2(ROOT / name, OUTPUT / name)
    shutil.copy2(requirements, OUTPUT / "requirements.lock")
    if release_path:
        shutil.copy2(ROOT / "scripts/windows_bootstrap.py", OUTPUT / "windows_bootstrap.py")
        (OUTPUT / "BUILD-INFO.json").write_text(json.dumps({
            "release": release_path.name, "release_sha256": digest(release_path),
            "application": app_manifest, "public_subtitle_weights": model_manifest,
            "requirements_sha256": digest(requirements),
            "packaging_sha256": {name: digest(path) for name, path in {
                "windows_bootstrap.py": OUTPUT / "windows_bootstrap.py",
                "runtime/python/Lib/site-packages/sitecustomize.py": OUTPUT / "runtime/python/Lib/site-packages/sitecustomize.py",
            }.items()},
            "validation": "Windows launch and completed-video acceptance required before delivery"
        }, ensure_ascii=False, indent=2))
    licenses = OUTPUT / "frontend" / "licenses"
    licenses.mkdir(exist_ok=True)
    entries = []
    lock = json.loads((ROOT / "frontend" / "package-lock.json").read_text())
    for name, item in lock["packages"].items():
        if not name or item.get("dev"):
            continue
        entries.append({"package": name, "version": item.get("version"), "license": item.get("license")})
        folder = ROOT / "frontend" / name
        for f in folder.glob("LICENSE*"):
            if f.is_file():
                shutil.copy2(f, licenses / (name.removeprefix("node_modules/").replace("/", "_")+"-"+f.name))
    (OUTPUT / "frontend" / "licenses.json").write_text(json.dumps(entries, indent=2), "utf-8")
    (OUTPUT / "Start.cmd").write_bytes(b'@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\nset PYTHONUTF8=1\r\n"%~dp0runtime\\python\\python.exe" -X utf8 "%~dp0launch.py"\r\nif errorlevel 1 pause\r\n')
    if release_path:
        (OUTPUT / "Start.cmd").write_bytes(
            b'@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\nset PYTHONUTF8=1\r\n'
            b'"%~dp0runtime\\python\\python.exe" -X utf8 "%~dp0windows_bootstrap.py"\r\n'
            b'if errorlevel 1 (\r\n  pause\r\n  exit /b 1\r\n)\r\n'
            b'"%~dp0runtime\\python\\python.exe" -X utf8 "%~dp0launch.py"\r\nif errorlevel 1 pause\r\n')
    (OUTPUT / "双击启动镜稿.cmd").write_bytes((OUTPUT / "Start.cmd").read_bytes())
    (OUTPUT / "Check.cmd").write_bytes(b'@echo off\r\nsetlocal\r\ncd /d "%~dp0"\r\nset PYTHONUTF8=1\r\n"%~dp0runtime\\python\\python.exe" -X utf8 -c "from jinggao.media import hardware; import json; print(json.dumps(hardware(), ensure_ascii=True, indent=2))"\r\npause\r\n')
    files = [p for p in OUTPUT.rglob("*") if p.is_file()]
    check_private_files(files, model_manifest)
    (ROOT / "dist").mkdir(exist_ok=True)
    import sys
    sys.path.insert(0, str(ROOT))
    from jinggao import __version__
    archive = ROOT / "dist" / (OUTPUT.name + "-test.zip" if release_path else f"Jinggao-{__version__}-Windows-x64.zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            z.write(path, Path(OUTPUT.name) / path.relative_to(OUTPUT))
    (archive.with_suffix(".zip.sha256")).write_text(digest(archive) + "  " + archive.name + "\n")
    print("Built", archive, f"({archive.stat().st_size/1024/1024:.1f} MiB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--release", type=Path, help="Pinned runtime release JSON, not the experimental workspace")
    parser.add_argument("--model-directory", type=Path, help="Public Paraformer weights directory")
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--output-name", help="A new folder name under build/, also used for the test archive")
    parser.add_argument("--bundle-pixabay-key", action="store_true",
                        help="Ship this machine's Pixabay key as the package default (defaults.json)")
    args = parser.parse_args()
    if args.release:
        load_release(args.release)
        if not args.output_name:
            parser.error("--release requires an explicit --output-name")
        if not args.prepare_only and args.model_directory is None:
            parser.error("--release requires --model-directory")
    if args.output_name:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_name):
            parser.error("--output-name must be a simple folder name")
        OUTPUT = ROOT / "build" / args.output_name
    requirements = args.requirements or ROOT / ("requirements-windows.lock" if args.release else "requirements.lock")
    prepare(requirements)
    if not args.prepare_only:
        package(args.release, args.model_directory, requirements, bundle_pixabay=args.bundle_pixabay_key)
