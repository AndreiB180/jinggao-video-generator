"""Real FFmpeg sampling through the guarded range proxy, using loopback only."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

from PIL import Image
import pytest

from jinggao import domestic, media, paths, segment_discovery as discovery, sources


@pytest.fixture
def fixture_source(tmp_path, monkeypatch):
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
                'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy'):
        monkeypatch.delenv(key, raising=False)
    movie = tmp_path / 'fixture.mp4'
    media.run([paths.binary('ffmpeg'), '-v', 'error', '-y', '-f', 'lavfi', '-i',
               'color=c=gray:s=640x360:r=25', '-t', '7', '-c:v', 'libx264',
               '-preset', 'ultrafast', '-movflags', '+faststart', str(movie)], timeout=15)
    body = movie.read_bytes()
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            start = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
            requests.append(start)
            self.send_response(206)
            self.send_header('Content-Length', str(len(body)-start))
            self.send_header('Content-Range', f'bytes {start}-{len(body)-1}/{len(body)}')
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                self.wfile.write(body[start:])
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=lambda:server.serve_forever(poll_interval=.02), daemon=True)
    thread.start()
    upstream = f'http://127.0.0.1:{server.server_port}/fixture.mp4'
    def exact_source(url):
        assert url == upstream
    monkeypatch.setattr(sources, 'public_url', exact_source)
    monkeypatch.setattr(domestic, 'require', exact_source)
    monkeypatch.setattr(sources, 'download_slot', lambda url: tmp_path/'cooldown.json')
    original_response = discovery._response
    @contextmanager
    def delayed(url, budget, referer, headers=None, **kwargs):
        until = time.monotonic()+.15
        while time.monotonic() < until:
            budget.check()
            time.sleep(.01)
        with original_response(url, budget, referer, headers, **kwargs) as response:
            yield response
    monkeypatch.setattr(discovery, '_response', delayed)
    try:
        yield upstream, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_four_frames_use_one_decode_and_preserve_actual_timestamps(tmp_path, fixture_source):
    source, requests = fixture_source
    budget = discovery._Budget(6, 2*1024*1024, None)
    points = [.56, 2.52, 4.48, 6.44]
    with discovery._proxy(source,budget,'') as (url,state):
        rows = discovery._extract_sequence(url,points,tmp_path,budget,360)
        assert not state['error']
    assert len(rows) == 4 and 1 <= len(requests) < 4
    assert 0 < budget.bytes <= budget.limit and not budget.clients
    for row,point in zip(rows,points,strict=True):
        assert point-.01 <= row['at'] <= point+.04
        assert row['requested_at'] == point
        with Image.open(row['path']) as image:
            assert image.size == (640,360)
    assert not list(tmp_path.glob('sequence-*.jpg'))


@pytest.mark.parametrize('cancel', [False, True])
def test_single_pass_still_obeys_deadline_and_cancellation(tmp_path, fixture_source, cancel):
    source,_ = fixture_source
    stopped = threading.Event()
    def check():
        if stopped.is_set():
            raise media.Cancelled()
    budget = discovery._Budget(5 if cancel else .06,2*1024*1024,check)
    timer = threading.Timer(.04,stopped.set) if cancel else None
    if timer:
        timer.start()
    began = time.monotonic()
    try:
        with discovery._proxy(source,budget,'') as (url,_):
            with pytest.raises(media.Cancelled if cancel else discovery.SampleLimit):
                discovery._extract_sequence(url,[.56,2.52,4.48,6.44],tmp_path,budget,360)
        assert time.monotonic()-began < 1.5
        assert not list(tmp_path.glob('sequence-*.jpg'))
    finally:
        if timer:
            timer.cancel()
