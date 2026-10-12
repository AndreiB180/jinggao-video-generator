"""Exercise a frozen Windows package with existing audio; no paid services."""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

root = Path(sys.executable).resolve().parents[2]
shared = Path(sys.argv[1])
os.environ['JINGGAO_DATA'] = str(root / 'data')
report = {'started_at': time.time(), 'runtime': str(root), 'platform': platform.platform(),
          'python': sys.version, 'machine': platform.machine(), 'paid_requests': 0,
          'full_video_acceptance': False, 'checks': []}
process = None
try:
    subprocess.run([sys.executable, '-X', 'utf8', str(root / 'windows_bootstrap.py')], check=True, timeout=180)
    report['checks'].append({'name': 'first_launch_runtime_prerequisite', 'status': 'passed'})
    import httpx
    from PIL import Image
    from jinggao import config, media, network, paths, planner, store
    from jinggao.local_asr import _runtime, model_dir

    def no_paid(*args, **kwargs):
        raise AssertionError('Packaged smoke test must not call a paid model or speech service')
    network.post = no_paid
    started = time.monotonic()
    _runtime(str(model_dir()))
    import jieba
    assert jieba.lcut('蔬菜清洗')
    report['checks'].append({'name': 'local_subtitle_dependencies_and_model', 'status': 'passed',
                             'seconds': round(time.monotonic()-started, 3)})
    log = (root.parent / 'launch-smoke.log').open('wb')
    process = subprocess.Popen([sys.executable, str(root / 'launch.py'), '--port', '18771', '--no-browser'],
                               cwd=root, stdout=log, stderr=subprocess.STDOUT)
    with httpx.Client(base_url='http://127.0.0.1:18771', trust_env=False, timeout=2) as client:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('Packaged app exited before health check')
            try:
                response = client.get('/api/health')
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(.2)
        else:
            raise RuntimeError('Packaged app never became healthy')
        assert client.get('/').status_code == 200
        assert client.get('/api/projects').status_code == 200
    report['checks'].append({'name': 'app_health_frontend_projects', 'status': 'passed'})

    config.update({'encoder': 'cpu'})
    case = json.loads((shared / 'windows-smoke-input.json').read_text(encoding='utf-8'))
    project = store.new_project('Windows 包内字幕与导出检查', case['narration'], media_policy='mixed')
    folder = paths.project_dir(project['id'])
    (folder / 'audio').mkdir(exist_ok=True)
    shutil.copy2(shared / 'windows-smoke-input.wav', folder / 'audio/smoke.wav')
    Image.new('RGB', (1280, 720), '#223449').save(folder / 'smoke.jpg')
    shot = planner.new_shot(case['narration'])
    shot.update(duration=case['duration'], selected_asset='smoke-image', approved=True,
                visual_origin='manual', audio={'path': 'audio/smoke.wav', 'duration': case['duration']})
    store.mutate(project['id'], lambda p: p.update(shots=[shot], assets=[{
        'id': 'smoke-image', 'kind': 'image', 'path': 'smoke.jpg', 'width': 1280, 'height': 720,
        'provider': 'local', 'title': 'Local synthetic test background'}]), internal=True)
    project = store.get_project(project['id'])
    started = time.monotonic()
    result = media.render(project, store.uid(), {'resolution': '720p'}, lambda *args: None, lambda: None)
    target = paths.file_path(project['id'], result['path'])
    media.run([paths.binary('ffmpeg'), '-v', 'error', '-xerror', '-i', str(target), '-f', 'null', '-'], timeout=60)
    probe = media.probe(target)
    report['checks'].append({'name': 'existing_audio_local_alignment_and_720p_export', 'status': 'passed',
                             'seconds': round(time.monotonic()-started, 3), 'export': result, 'probe': probe})
    shutil.copy2(target, shared / 'windows-package-smoke.mp4')
    report['status'] = 'passed'
except Exception as exc:
    report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    raise
finally:
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    report['elapsed_seconds'] = round(time.time()-report['started_at'], 3)
    (shared / 'windows-package-smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)
