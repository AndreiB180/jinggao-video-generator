"""Install the official Windows prerequisite only when the local runtime needs it."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

URL = 'https://aka.ms/vc14/vc_redist.x64.exe'


def runtime_ready():
    try:
        for name in ('msvcp140.dll', 'msvcp140_1.dll'):
            ctypes.WinDLL(name)
    except OSError:
        return False
    return True


def main():
    if os.name != 'nt' or runtime_ready():
        return
    root = Path(__file__).resolve().parent
    target = root / 'runtime/setup/vc_redist.x64.exe'
    target.parent.mkdir(parents=True, exist_ok=True)
    print('首次使用需要准备运行组件，正在获取微软官方安装程序。', flush=True)
    if not target.exists():
        temporary = target.with_suffix('.part')
        try:
            with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    # The official URL may update. Require a valid Microsoft signature before
    # executing either a new or previously downloaded installer.
    escaped = str(target).replace("'", "''")
    command = (
        "$ErrorActionPreference='Stop'; "
        f"$jgInstaller='{escaped}'; "
        "$jgSignature=Get-AuthenticodeSignature -LiteralPath $jgInstaller; "
        "if ($jgSignature.Status -ne 'Valid' -or "
        "$jgSignature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation(?:,|$)') "
        "{ throw 'Microsoft installer signature check failed' }; "
        "$jgProcess=Start-Process -FilePath $jgInstaller "
        "-ArgumentList '/install','/passive','/norestart' -Wait -PassThru "
        + ("" if ctypes.windll.shell32.IsUserAnAdmin() else "-Verb RunAs ")
        + "; exit $jgProcess.ExitCode"
    )
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command], check=False)
    if result.returncode not in (0, 3010) or not runtime_ready():
        raise RuntimeError('运行组件未安装完成。请重新运行 Start.cmd，按 Windows 提示完成安装。')
    print('运行组件已就绪。', flush=True)


if __name__ == '__main__':
    main()
