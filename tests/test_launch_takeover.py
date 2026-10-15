"""Starting a new version closes the copy that is already running (Windows netstat parsing)."""
import importlib.util
from pathlib import Path


def _launch():
    spec = importlib.util.spec_from_file_location("launch_module", Path(__file__).parents[1] / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_netstat_finds_only_the_process_listening_on_the_port():
    text = """
活动连接

  协议  本地地址          外部地址        状态           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1012
  TCP    127.0.0.1:8765         0.0.0.0:0              LISTENING       5678
  TCP    127.0.0.1:8765         127.0.0.1:50123        ESTABLISHED     5678
  TCP    127.0.0.1:50123        127.0.0.1:8765         ESTABLISHED     9999
  TCP    127.0.0.1:18765        0.0.0.0:0              LISTENING       4444
  TCP    [::1]:8770             [::]:0                 侦听            7777
"""
    launch = _launch()
    assert launch.netstat_listeners(text, 8765) == {5678}
    assert launch.netstat_listeners(text, 8770) == {7777}
    assert launch.netstat_listeners(text, 8888) == set()
    assert launch.DEFAULT_PORT != launch.LEGACY_PORT
