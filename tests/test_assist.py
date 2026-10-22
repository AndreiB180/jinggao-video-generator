import os
import sys
import time

import pytest

from jinggao import assist, paths


def fake_program(tmp_path, monkeypatch, lines):
    """A stand-in for tools/jinggao-assist: prints status lines, then waits until its input closes."""
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "assist.json").write_text("{}")
    script = tools / "jinggao-assist"
    script.write_text(f"#!{sys.executable}\nimport sys\nfor line in {lines!r}:\n    print(line, flush=True)\n"
                      "sys.stdin.read()\n")
    script.chmod(0o755)
    monkeypatch.setattr(assist, "TOOLS", tools)
    monkeypatch.setattr(paths, "DATA", tmp_path / "data")


def wait_for(state):
    for _ in range(50):
        if assist.status()["state"] == state:
            return assist.status()
        time.sleep(.1)
    raise AssertionError(assist.status())


@pytest.mark.skipif(os.name == "nt", reason="the stand-in program is a script")
def test_waits_for_approval_then_shows_the_code_and_stops(tmp_path, monkeypatch):
    fake_program(tmp_path, monkeypatch, ["KEY ssh-ed25519 AAAA jinggao-assist", "WAIT 等技术人员把这台电脑加进白名单",
                                         "LISTEN 23004 port443"])
    try:
        assert assist.status()["available"]
        assist.start(8770)
        on = wait_for("on")
        assert on["port"] == 23004 and on["key"] == "ssh-ed25519 AAAA jinggao-assist"
    finally:
        stopped = assist.stop()
    assert stopped["state"] == "off" and stopped["port"] is None


@pytest.mark.skipif(os.name == "nt", reason="the stand-in program is a script")
def test_the_customer_sees_the_key_while_waiting(tmp_path, monkeypatch):
    fake_program(tmp_path, monkeypatch, ["KEY ssh-ed25519 BBBB jinggao-assist", "WAIT 等技术人员把这台电脑加进白名单"])
    try:
        assist.start(8770)
        waiting = wait_for("waiting")
        assert waiting["key"].startswith("ssh-ed25519 BBBB") and "白名单" in waiting["message"]
    finally:
        assist.stop()


def test_no_button_without_the_program(tmp_path, monkeypatch):
    monkeypatch.setattr(assist, "TOOLS", tmp_path / "missing")
    assert assist.status()["available"] is False
