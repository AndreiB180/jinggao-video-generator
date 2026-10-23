from jinggao import health


def test_a_struggling_computer_is_named_plainly():
    reading = {"checked": 1, "mem_total_mb": 16200, "mem_free_mb": 600, "commit_mb": 21000, "commit_limit_mb": 23000,
               "pages_per_sec": 3200, "hard_faults": 900, "cpu_name": "13th Gen Intel(R) Core(TM) i5-13400",
               "cpu_mhz": 2500, "cpu_max_mhz": 2500, "cpu_load": 97, "cpu_performance": 41,
               "top_cpu": [{"name": "360Tray", "pct": 22.0, "mb": 300}, {"name": "python#2", "pct": 30.0, "mb": 900}],
               "top_mem": [{"name": "chrome", "mb": 4100}],
               "power_plan": "a1841308-3541-4fab-bc81-f71556f20b4a", "antivirus": ["360安全卫士"],
               "app_disk": {"drive": "D", "media": "HDD", "bus": "SATA", "free_gb": 80}, "disk_busy": 100, "disk_queue": 6.0}
    text = "\n".join(health.lines(reading))
    for sign in ("内存不够", "降频", "节能模式", "360Tray 22.0%", "第三方杀毒", "机械硬盘"):
        assert sign in text, sign
    assert "python#2" not in text.split("最占 CPU")[0]      # our own workers are not "other software"


def test_a_healthy_computer_gets_one_line_of_verdict():
    reading = {"checked": 1, "mem_total_mb": 48000, "mem_free_mb": 30000, "hard_faults": 0, "cpu_name": "x",
               "cpu_performance": 110, "power_plan": "381b4222-f694-41f0-9685-ff5bb260df2e",
               "antivirus": ["Windows Defender"], "app_disk": {"drive": "D", "media": "SSD"}}
    text = health.lines(reading)
    assert text[0] == "电脑体检：没发现明显问题" and any("平衡" in line for line in text)


def test_other_systems_get_a_note():
    assert health.lines({"note": "电脑体检只在 Windows 上做"}) == ["电脑体检：电脑体检只在 Windows 上做"]


def test_full_speed_is_windows_only_and_harmless_elsewhere(monkeypatch):
    import subprocess
    import sys
    from jinggao import power
    monkeypatch.setattr(sys, "platform", "darwin")
    before = subprocess.Popen._execute_child
    assert power.full_speed() is False
    power.unthrottle_children()
    assert subprocess.Popen._execute_child is before
