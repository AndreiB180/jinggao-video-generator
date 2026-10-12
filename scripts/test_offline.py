"""Run handoff regressions in temporary data with loopback-only networking."""
import os
from pathlib import Path
import socket
import sys
import tempfile


def guard(original):
    def connect(sock, address):
        if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError("Public networking is disabled in handoff tests")
        return original(sock, address)
    return connect


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    socket.socket.connect = guard(socket.socket.connect)
    socket.socket.connect_ex = guard(socket.socket.connect_ex)
    with tempfile.TemporaryDirectory(prefix="jinggao-offline-tests-") as data:
        os.environ["JINGGAO_DATA"] = data
        import pytest
        raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", *sys.argv[1:]]))
