"""电脑体检: what on this computer slows video making down, for the diagnostics text.

A customer's PC once ran the local steps (speech alignment: no network, no model) 20-30 times slower than
our test PC, so concurrency and network were not the problem. These readings show why without screenshots
of the task manager: memory and paging, total CPU load and its clock, the processes using the most CPU and
memory, the power plan, antivirus products and the disk the program runs from. Windows only (WMI through
one PowerShell call, class names are not localised); elsewhere a short note.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import time

from . import paths

# Known power plan GUIDs (powercfg prints the localised name next to it).
PLANS = {"381b4222-f694-41f0-9685-ff5bb260df2e": "平衡", "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c": "高性能",
         "a1841308-3541-4fab-bc81-f71556f20b4a": "节能", "e9a42b02-d5df-448d-aa00-03f14749eb61": "卓越性能"}

SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$r = @{}
$os = Get-CimInstance Win32_OperatingSystem
$r.mem_total_mb = [int]($os.TotalVisibleMemorySize / 1024)
$r.mem_free_mb = [int]($os.FreePhysicalMemory / 1024)
$m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
$r.commit_mb = [int]($m.CommittedBytes / 1MB); $r.commit_limit_mb = [int]($m.CommitLimit / 1MB)
$r.pages_per_sec = [int]$m.PagesPerSec; $r.hard_faults = [int]$m.PageReadsPersec
$r.pagefile = @(Get-CimInstance Win32_PageFileUsage | ForEach-Object { @{name = $_.Name; mb = $_.AllocatedBaseSize; used = $_.CurrentUsage} })
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$r.cpu_name = $cpu.Name; $r.cpu_mhz = $cpu.CurrentClockSpeed; $r.cpu_max_mhz = $cpu.MaxClockSpeed
$r.cpu_load = [int](Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
$perf = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'"
if ($perf) { $r.cpu_performance = [int]$perf.PercentProcessorPerformance }
$procs = Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | Where-Object { $_.Name -ne '_Total' -and $_.Name -ne 'Idle' }
$cores = [Environment]::ProcessorCount
$r.top_cpu = @($procs | Sort-Object PercentProcessorTime -Descending | Select-Object -First 8 | ForEach-Object { @{name = $_.Name; pct = [math]::Round($_.PercentProcessorTime / $cores, 1); mb = [int]($_.WorkingSetPrivate / 1MB)} })
$r.top_mem = @($procs | Sort-Object WorkingSetPrivate -Descending | Select-Object -First 8 | ForEach-Object { @{name = $_.Name; mb = [int]($_.WorkingSetPrivate / 1MB)} })
$scheme = (powercfg /getactivescheme) | Out-String
if ($scheme -match '([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})') { $r.power_plan = $Matches[1] }
$r.power_plan_text = $scheme.Trim()
$r.antivirus = @(Get-CimInstance -Namespace root/SecurityCenter2 -ClassName AntiVirusProduct | ForEach-Object { $_.displayName })
$letter = $env:JINGGAO_ROOT.Substring(0, 1)
$number = (Get-Partition -DriveLetter $letter).DiskNumber
$disk = Get-PhysicalDisk | Where-Object { $_.DeviceId -eq [string]$number } | Select-Object -First 1
$r.app_disk = @{drive = $letter; media = [string]$disk.MediaType; bus = [string]$disk.BusType; model = $disk.FriendlyName}
$vol = Get-Volume -DriveLetter $letter
$r.app_disk.free_gb = [int]($vol.SizeRemaining / 1GB)
$d = Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk -Filter "Name='_Total'"
$r.disk_busy = [int]$d.PercentDiskTime; $r.disk_queue = [math]::Round($d.CurrentDiskQueueLength, 1)
[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($r | ConvertTo-Json -Depth 4 -Compress)))
"""

_last: dict = {}


def run(max_age: float = 60.0) -> dict:
    """The readings (cached for max_age seconds: the diagnostics text may be copied several times)."""
    if _last and time.time() - _last.get("checked", 0) < max_age:
        return _last
    result = {"checked": time.time()}
    if os.name != "nt":
        result["note"] = "电脑体检只在 Windows 上做"
    else:
        from .media import _windows_flags
        encoded = base64.b64encode(SCRIPT.encode("utf-16-le")).decode()
        try:
            done = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                                  capture_output=True, timeout=60, creationflags=_windows_flags(),
                                  env=os.environ | {"JINGGAO_ROOT": str(paths.ROOT)})
            lines = done.stdout.decode("utf-8", errors="replace").strip().splitlines()
            result |= json.loads(base64.b64decode(lines[-1]).decode("utf-8"))
        except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
            result["note"] = f"体检没有完成：{type(exc).__name__}"
    _last.clear()
    _last.update(result)
    return result


def lines(r: dict) -> list[str]:
    """Plain lines for the diagnostics text, worst signs first."""
    if r.get("note"):
        return [f"电脑体检：{r['note']}"]
    out, warn = [], []
    total, free = r.get("mem_total_mb") or 0, r.get("mem_free_mb") or 0
    if total:
        used = 100 - 100 * free / total
        out.append(f"内存：已用 {used:.0f}%（空闲 {free / 1024:.1f} GB / 共 {total / 1024:.1f} GB）· 提交 "
                   f"{(r.get('commit_mb') or 0) / 1024:.1f}/{(r.get('commit_limit_mb') or 0) / 1024:.1f} GB · "
                   f"换页 {r.get('pages_per_sec', 0)}/秒（读盘 {r.get('hard_faults', 0)}/秒）")
        if free < 1500 or (r.get("hard_faults") or 0) > 200:
            warn.append("内存不够：系统在把内存挪到硬盘上，所有步骤都会慢几十倍。少开别的软件，或把同时制作调低")
    if r.get("cpu_name"):
        out.append(f"CPU：{r['cpu_name'].strip()} · 占用 {r.get('cpu_load')}% · 频率 {r.get('cpu_mhz')}/{r.get('cpu_max_mhz')} MHz"
                   + (f" · 实际性能 {r['cpu_performance']}%" if r.get("cpu_performance") is not None else ""))
        if r.get("cpu_performance") is not None and r["cpu_performance"] < 60:
            warn.append("CPU 被降频（节能模式、过热或电源设置），只发挥了不到六成")
    plan = PLANS.get(str(r.get("power_plan", "")).lower())
    out.append(f"电源模式：{plan or (r.get('power_plan_text') or '未知')[:60]}")
    if plan == "节能":
        warn.append("电源是节能模式：控制面板 → 电源选项 → 选「高性能」或「平衡」")
    ours = {"python", "ffmpeg", "jinggao-assist"}
    others = [p for p in r.get("top_cpu") or [] if not any(p["name"].lower().startswith(o) for o in ours)]
    out.append("最占 CPU：" + "，".join(f"{p['name']} {p['pct']}%" for p in (r.get("top_cpu") or [])[:8]))
    out.append("最占内存：" + "，".join(f"{p['name']} {p['mb'] / 1024:.1f}G" for p in (r.get("top_mem") or [])[:8]))
    busy = [p for p in others if p["pct"] >= 15]
    if busy:
        warn.append("别的软件在占 CPU：" + "，".join(f"{p['name']} {p['pct']}%" for p in busy)
                    + "（镜稿让着前台软件，它们占得越多镜稿越慢）")
    if r.get("antivirus"):
        out.append("杀毒软件：" + "、".join(r["antivirus"]))
        if any(name for name in r["antivirus"] if "defender" not in name.lower()):
            warn.append("装了第三方杀毒软件：可能在扫描镜稿读写的每个文件，可把镜稿文件夹加进信任/白名单")
    disk = r.get("app_disk") or {}
    if disk:
        media = {"3": "机械硬盘", "4": "固态硬盘", "HDD": "机械硬盘", "SSD": "固态硬盘"}.get(str(disk.get("media")), str(disk.get("media") or "未知"))
        out.append(f"镜稿所在：{disk.get('drive')} 盘 · {media} · {disk.get('bus') or ''} · 剩 {disk.get('free_gb')} GB · "
                   f"硬盘忙 {r.get('disk_busy')}%（队列 {r.get('disk_queue')}）")
        if media == "机械硬盘":
            warn.append("镜稿装在机械硬盘上：同时做几条时读写会排队，建议放到固态硬盘（C 盘或其他 SSD）")
    return (["电脑体检（需要注意）：" + w for w in warn] or ["电脑体检：没发现明显问题"]) + ["  " + line for line in out]
