"""Use a real local Range source and decoder with a delayed TCP handshake."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import httpcore
from httpcore._backends.sync import SyncBackend
import pytest

from jinggao import domestic, media, network, paths, segment_discovery as discovery, sources


@pytest.fixture
def delayed_tcp_source(tmp_path, monkeypatch):
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
        monkeypatch.delenv(key, raising=False)
    movie = tmp_path / 'source.mp4'
    media.run([paths.binary('ffmpeg'), '-v', 'error', '-y', '-f', 'lavfi', '-i',
               'testsrc2=s=1280x720:r=25', '-t', '3', '-c:v', 'libx264',
               '-preset', 'ultrafast', '-movflags', '+faststart', str(movie)], timeout=20)
    body = movie.read_bytes()
    requests, connections = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.headers.get('Range'))
            start = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
            self.send_response(206)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', str(len(body) - start))
            self.send_header('Content-Range', f'bytes {start}-{len(body)-1}/{len(body)}')
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                self.wfile.write(body[start:])
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.02), daemon=True)
    thread.start()
    upstream = f'http://127.0.0.1:{server.server_port}/source.mp4'

    def only_local(url):
        assert url == upstream, 'No external source may be contacted'

    monkeypatch.setattr(sources, 'public_url', only_local)
    monkeypatch.setattr(domestic, 'require', only_local)
    monkeypatch.setattr(sources, 'download_slot', lambda url: tmp_path / 'cooldown.json')
    client = network.client
    monkeypatch.setattr(network, 'client', lambda **kwargs: client(**(kwargs | {'trust_env': False})))
    def reason(exc):
        return type(exc).__name__ + (': ' + str(exc)[:200] if isinstance(exc, (TypeError, AssertionError)) else '')
    monkeypatch.setattr(discovery, '_reason', reason)
    original = SyncBackend.connect_tcp

    def connect(self, host, port, timeout=None, **kwargs):
        assert str(host) in ('127.0.0.1', "b'127.0.0.1'") and port == server.server_port
        connections.append(timeout)
        delay = 1.25
        if timeout is not None and timeout < delay:
            time.sleep(timeout)
            raise httpcore.ConnectTimeout('Controlled handshake exceeded supplied timeout')
        time.sleep(delay)
        return original(self, host, port, timeout=timeout - delay if timeout else None, **kwargs)

    monkeypatch.setattr(SyncBackend, 'connect_tcp', connect)
    try:
        yield upstream, requests, connections
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_real_frame_survives_handshake_over_one_second(tmp_path, delayed_tcp_source):
    url, requests, connections = delayed_tcp_source
    budget = discovery._Budget(9, 8 * 1024 * 1024, None)
    with discovery._proxy(url, budget, '') as (local, state):
        try:
            actual, width, height = discovery._extract(local, 1.0, tmp_path / 'frame.jpg',
                                                       budget, 720, file_seek=True)
        except ValueError as exc:
            raise AssertionError(state['error']) from exc
        assert state['error'] == ''
    assert requests and connections and actual == pytest.approx(1.0, abs=.05)
    assert (width, height) == (1280, 720)
    assert 0 < budget.bytes <= budget.limit and not budget.clients
    assert state['handlers_stopped']


def test_short_total_deadline_still_stops_connection_setup(tmp_path, delayed_tcp_source):
    url, requests, connections = delayed_tcp_source
    budget = discovery._Budget(.25, 8 * 1024 * 1024, None)
    started = time.monotonic()
    with pytest.raises((discovery.SampleLimit, ValueError)):
        with discovery._proxy(url, budget, '') as (local, state):
            discovery._extract(local, 1.0, tmp_path / 'frame.jpg', budget, 720, file_seek=True)
    assert time.monotonic() - started < 1
    assert connections and all(value <= .25 for value in connections)
    assert not requests and not budget.clients and budget.bytes == 0
    assert not (tmp_path / 'frame.jpg').exists() and state['handlers_stopped']


def test_cancel_during_connection_does_not_send_a_late_media_request(tmp_path, delayed_tcp_source):
    url, requests, connections = delayed_tcp_source
    cancelled = threading.Event()

    def check():
        if cancelled.is_set():
            raise media.Cancelled()

    budget = discovery._Budget(9, 8 * 1024 * 1024, check)
    timer = threading.Timer(.3, cancelled.set)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(media.Cancelled):
            with discovery._proxy(url, budget, '') as (local, state):
                discovery._extract(local, 1.0, tmp_path / 'frame.jpg', budget, 720, file_seek=True)
        assert time.monotonic() - started < 2
        assert connections and not requests and budget.bytes == 0
        assert not budget.clients and state['handlers_stopped']
    finally:
        timer.cancel()
        timer.join(timeout=1)
