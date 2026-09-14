"""PhiView: image-only browser, server-owned camera, Gaussian rendering and physics.

Run inside a GPU allocation: python -m physicalview.phiview --demo --out OUTPUT.
One shared demo session, including camera and selection, is visible to all viewers.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import Future
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import queue
import shutil
import socket
import subprocess
import threading
import time
import traceback
from urllib.parse import urlparse, parse_qs

import numpy as np

from physicalview.phiview_scene import FlyCamera, GaussianScene
from physicalview.phiview_sim import DemoPhysics


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(path, value):
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False))
    tmp.replace(path)


class Demo:
    def __init__(self, args):
        from physicalview.config import load_config
        from physicalview.scene_state import discover_result_sets, load_scene
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('PhiView requires a server GPU; launch run/phiview.sbatch')
        self.args, self.out = args, Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.config = load_config(args.config)
        scenes = discover_result_sets(self.config)
        rs = next((s for s in scenes if s.name == args.scene), None)
        if rs is None:
            raise ValueError(f'Scene {args.scene!r} not found')
        self.state = load_scene(self.config, rs, device='cuda')
        self.scene = GaussianScene(self.state)
        active_inpaint = self.out/'active-inpaint.json'
        if active_inpaint.exists():
            from agents.core.common import load_gaussians
            saved = json.loads(active_inpaint.read_text())
            self.scene.prompt_backgrounds[frozenset(saved['objects'])] = load_gaussians(saved['path'], device='cuda')
        self.physics = DemoPhysics(self.state, self.out)
        self.wh = (args.width, args.height)
        self.native_wh = (self.state.W, self.state.H)
        self.camera_names = sorted(self.state.cameras)
        preferred = 'DSC01593.JPG'
        self.camera_name = preferred if preferred in self.state.cameras else self.camera_names[0]
        self.camera = FlyCamera.from_w2c(self.state.cameras[self.camera_name],
            float(np.rad2deg(2*np.arctan(self.state.H/(2*self.state.K[1, 1])))))
        self.mode, self.selected, self.highlight = 'original', None, True
        self.ready = False
        self.frame_id = 0
        self.frames = OrderedDict()
        self.jpeg = b''
        self.condition = threading.Condition()
        self.commands = queue.Queue(maxsize=128)
        self.stop = threading.Event()
        self.frame_error = None
        self.keys, self.look, self.boost = [], np.zeros(2), False
        self.last_input = 0.
        self.pipeline_thread = None
        self.pipeline_status = {'state': 'idle'}
        self.reload_pending = False
        self.dirty = True
        self.last_render_s = 0.
        self.status_cache = {}
        self.audit = open(self.out/'actions.jsonl', 'a', buffering=1)
        with open(rs.splat_ply, 'rb') as source_file:
            splat_sha256 = hashlib.file_digest(source_file, 'sha256').hexdigest()
        manifest = {'scene': rs.name, 'original_splat': str(rs.splat_ply),
                    'original_splat_sha256': splat_sha256, 'original_splat_bytes': rs.splat_ply.stat().st_size,
                    'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                        cwd=Path(__file__).resolve().parents[1], text=True).strip(),
                    'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                    'web_sha256': hashlib.sha256((Path(__file__).parent/'web/phiview.html').read_bytes()).hexdigest(),
                    'gaussians': self.scene.count, 'sh_degree': self.scene.raw['sh_degree'],
                    'rasterize_mode': self.scene.rasterize_mode,
                    'native_resolution': self.native_wh, 'stream_resolution': self.wh,
                    'hardware': torch.cuda.get_device_name(), 'node': socket.gethostname(),
                    'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                    'object_count': len(self.scene.names), 'collision_bodies': sorted(self.physics.available),
                    'mask_sources': self.scene.mask_sources,
                    'discovery_provenance': 'existing scene proposals; cached masks, including GT-referenced metadata',
                    'physics_scope': 'existing exported support slabs and collision objects; not verified full-room collision',
                    'client_geometry_bytes': 0, 'backend': 'gsplat CUDA + MuJoCo server physics',
                    'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in Path(__file__).parent.glob('phiview*.py')}}
        save_json(self.out/'manifest.json', manifest)
        self.manifest = manifest

    def status(self):
        objects = []
        for name, rec in self.state.objects.items():
            proposals = ['original'] + [s for s, p in rec.proposals.items() if p.gs_ply]
            objects.append({'id': name, 'label': rec.label, 'simulatable': name in self.physics.available,
                'enabled': name in self.physics.enabled, 'variant': self.scene.variants[name],
                'proposals': proposals, 'physics': self.physics.parameters(name),
                'mask_source': self.scene.mask_sources[name]})
        return {'ready': self.ready, 'scene': self.args.scene, 'objects': objects,
                'selected': self.selected, 'mode': self.mode, 'highlight': self.highlight,
                'background_completion': self.scene.clean is not None,
                'frame': self.frame_id, 'render_ms': round(self.last_render_s*1000, 1),
                'resolution': self.wh, 'native_resolution': self.native_wh,
                'gaussians': self.scene.count, 'gpu': self.manifest['hardware'],
                'running': self.physics.running, 'sim_time': float(self.physics.data.time),
                'robot': self.physics.robot_status, 'pipeline': self.pipeline_status,
                'error': self.frame_error, 'cameras': self.camera_names,
                'camera_name': self.camera_name,
                'projectile_contacts': self.physics.events[-10:]}

    def selected_required(self):
        if self.selected is None:
            raise ValueError('Select an object first')
        return self.selected

    def execute(self, msg):
        op = msg.get('op')
        if op == 'input':
            self.keys = [k for k in msg.get('keys', []) if k in 'wasdqe']
            look = np.asarray(msg.get('look', [0, 0]), float)
            if look.shape != (2,) or not np.isfinite(look).all():
                raise ValueError('Invalid mouse input')
            self.look += np.clip(look, -2000, 2000)
            self.boost = bool(msg.get('boost')); self.last_input = time.monotonic()
            return {'ok': True}
        if op == 'select':
            name = msg.get('object')
            if name not in self.state.objects:
                raise ValueError('Unknown object')
            self.selected = name
        elif op == 'pick' or op == 'shoot':
            fid = int(msg.get('frame', self.frame_id))
            frame = self.frames.get(fid)
            if frame is None:
                raise ValueError('Displayed frame expired; click the refreshed image')
            mask, depth, w2c, K = frame
            u, v = float(msg.get('x', .5)), float(msg.get('y', .5))
            if not (0 <= u < 1 and 0 <= v < 1):
                raise ValueError('Click outside image')
            x, y = int(u*mask.shape[1]), int(v*mask.shape[0])
            if op == 'pick':
                i = int(mask[y, x]); self.selected = self.scene.names[i-1] if i else None
            else:
                self.mode = 'simulation'
                c2w = np.linalg.inv(w2c)
                direction = c2w[:3, :3]@np.linalg.solve(K, [x, y, 1.])
                self.physics.shoot(c2w[:3, 3], direction, float(msg.get('speed', 12)))
        elif op == 'view':
            mode = msg.get('mode')
            if mode not in ('original', 'simulation', 'clean_selected', 'clean_all'):
                raise ValueError('Unknown view')
            if mode == 'clean_selected':
                self.selected_required()
            if mode in ('clean_selected', 'clean_all') and self.scene.clean is None:
                raise ValueError('Run inpainting before switching to this view')
            self.mode = mode
            if mode != 'simulation':
                self.physics.running = False
        elif op == 'highlight':
            self.highlight = bool(msg.get('value', True))
        elif op == 'enable':
            names = self.physics.available if msg.get('all') else [self.selected_required()]
            self.physics.enable(names); self.mode = 'simulation'
        elif op in ('fall', 'friction', 'throw'):
            self.physics.perturb(self.selected_required(), op, self.camera.forward(), msg.get('strength', 3))
            self.mode = 'simulation'
        elif op == 'friction_value':
            self.physics.set_friction(self.selected_required(), msg['value'])
        elif op == 'play':
            self.physics.enable(self.physics.available)
            self.physics.running = True; self.mode = 'simulation'
        elif op == 'pause':
            self.physics.running = False
        elif op == 'reset':
            self.physics.reset()
        elif op == 'variant':
            name = self.selected_required()
            source = msg['source']; previous = self.scene.variants[name]
            self.scene.choose(name, source)
            try:
                self.physics.replace_variant(name, None if source == 'original' else self.state.objects[name].proposals[source])
            except Exception:
                self.scene.variants[name] = previous
                raise
            self.mode = 'simulation'
            self.physics.running = False
        elif op in ('robot', 'robot_command'):
            name = self.selected_required()
            if op == 'robot':
                self.physics.add_robot(name, camera_position=self.camera.position)
            else:
                self.physics.command_robot(name, str(msg.get('command', ''))[:500], camera_position=self.camera.position)
            self.mode = 'simulation'
        elif op == 'camera':
            name = msg.get('name', self.camera_name)
            if name not in self.state.cameras:
                raise ValueError('Unknown source camera')
            self.camera = FlyCamera.from_w2c(self.state.cameras[name], self.camera.fov)
            self.camera_name = name
        elif op == 'focus':
            name = self.selected_required()
            p = np.asarray(self.state.objects[name].meta['centroid'])
            self.camera.position = p + [.6, .7, .45]
            direction = p-self.camera.position; direction /= np.linalg.norm(direction)
            self.camera.yaw = float(np.arctan2(direction[1], direction[0]))
            self.camera.pitch = float(np.arcsin(direction[2]))
        elif op == 'resolution':
            wh = msg.get('value')
            choices = {'720p': (1280, 720), '1080p': (1920, 1080), 'native': self.native_wh}
            if wh not in choices:
                raise ValueError('Unknown resolution')
            self.wh = choices[wh]
        elif op == 'speed':
            value = float(msg['value'])
            if not np.isfinite(value):
                raise ValueError('Invalid camera speed')
            self.camera.speed = float(np.clip(value, .05, 10))
        elif op in ('inpaint', 'generate', 'discover', 'build'):
            self.launch_pipeline(msg)
        elif op == 'snapshot':
            (self.out/f'frame-{self.frame_id:06d}.jpg').write_bytes(self.jpeg)
        else:
            raise ValueError('Unknown command')
        self.dirty = True
        self.audit.write(json.dumps({'time': time.time(), 'command': msg, 'selected': self.selected,
                                     'sim_time': float(self.physics.data.time)}, default=json_default)+'\n')
        return {'ok': True, 'selected': self.selected}

    def launch_pipeline(self, msg):
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            raise ValueError('A scene build is already running')
        if msg['op'] in ('generate', 'build') or (msg['op'] == 'inpaint' and not msg.get('all')):
            self.selected_required()
        selected = self.selected
        inpaint_names = sorted(self.physics.available) if msg.get('all') else ([selected] if selected else [])
        self.pipeline_status = {'state': 'preparing', 'operation': msg['op']}
        source = self.state.result_set.out_dir
        config = self.config
        def worker():
            from physicalview import pipeline
            from physicalview.app import Selection
            from physicalview.jobs import JobManager, JobSpec
            from physicalview.gpu import detect_gpu
            mgr = None
            try:
                # Every pipeline invocation gets its own writable copy. Existing scientific
                # scene artifacts and previous prompt versions can never be overwritten.
                work = self.out/'builds'/str(time.time_ns())
                work.mkdir(parents=True)
                folders = () if msg['op'] == 'discover' else ('objects', 'inpaint', 'sim_export')
                for folder in folders:
                    if (source/folder).exists():
                        shutil.copytree(source/folder, work/folder, symlinks=False)
                ctx = pipeline.StageContext(config, self.state.result_set.scene_id, work,
                    auto=self.state.result_set.kind == 'auto', scene_dir=self.state.result_set.scene_dir)
                ids = [self.state.objects[selected].index] if selected else []
                if msg['op'] == 'inpaint':
                    if msg.get('all'):
                        ids = [self.state.objects[n].index for n in self.physics.available]
                    if not ids:
                        raise ValueError('No objects available for inpainting')
                    prompt = str(msg.get('prompt', '')).strip()[:2000]
                    if not prompt:
                        raise ValueError('Enter an inpainting prompt')
                    for name in inpaint_names:
                        for cached in (work/'inpaint'/name).glob('inpainted_*.png'):
                            cached.unlink()
                    specs = pipeline.inpaint(ctx, Selection(kind='object', object_ids=ids), prompt,
                                              'qwen_image_edit', refine_iters=300)
                    prepared = True
                    for name in inpaint_names:
                        inp = work/'inpaint'/name
                        if not all((inp/p).exists() for p in ('views.json', 'plane.json', 'removal_idx.npy')):
                            prepared = False; break
                        views = json.loads((inp/'views.json').read_text())
                        if not views or not all((inp/f'mask_{i}.png').exists() for i in range(len(views))):
                            prepared = False; break
                    if prepared:
                        specs = specs[2:]  # reuse unchanged view masks; always rerun the prompt edit
                    for spec in specs:
                        if 'agents.edit.inpaint_qwen' in spec.argv:
                            spec.argv[spec.argv.index('agents.edit.inpaint_qwen')] = 'physicalview.phiview_inpaint'
                    for spec in specs:
                        spec.env['SIMANY_REQUIRE_QWEN'] = '1'
                        spec.env['HF_HOME'] = '/group/worldcept/hf_cache'
                    guard = JobSpec(name='inpaint:verify prompt execution',
                        argv=[str(config.interpreter('studio')), '-m', 'physicalview.phiview_inpaint_guard',
                              '--root', str(work/'inpaint'), '--objects', ','.join(inpaint_names), '--prompt', prompt],
                        env={'PYTHONPATH': str(config.package_root)+':'+str(config.repo_root)},
                        cwd=config.package_root, env_key='studio', needs_gpu=False,
                        tags={'chain': specs[0].tags['chain']})
                    specs.insert(-1, guard)
                elif msg['op'] == 'discover':
                    specs = pipeline.discover(ctx, 'sam3_auto')
                else:
                    specs = []
                    if msg['op'] == 'generate':
                        specs.append(pipeline.generate(ctx, msg.get('model', 'trellis'), ids))
                    specs += [pipeline.register(ctx, 'yaw_sweep_icp', ids), pipeline.physics(ctx, ids),
                              pipeline.export_mjcf(ctx, collision_mode='room')]
                mgr = JobManager(config, detect_gpu())
                jobs = mgr.submit_chain(specs)
                self.pipeline_status = {'state': 'running', 'operation': msg['op'], 'work': str(work),
                                         'jobs': [j.id for j in jobs]}
                for job in jobs:
                    job.wait()
                    if job.state.value != 'SUCCEEDED':
                        raise RuntimeError(f'{job.spec.name}: {job.state.value}; {job.tail(8)}')
                self.pipeline_status = {'state': 'succeeded', 'operation': msg['op'], 'work': str(work)}
                self.reload_pending = (work, msg['op'], inpaint_names)
            except Exception as exc:
                self.pipeline_status = {'state': 'failed', 'operation': msg['op'], 'error': str(exc)}
            finally:
                save_json(self.out/'pipeline-result.json', self.pipeline_status)
                if mgr:
                    mgr.shutdown()
        self.pipeline_thread = threading.Thread(target=worker, daemon=True)
        self.pipeline_thread.start()

    def reload_build(self):
        from dataclasses import replace
        from physicalview.scene_state import load_scene
        work, operation, inpaint_names = self.reload_pending; self.reload_pending = False
        if operation == 'inpaint':
            from agents.core.common import load_gaussians
            bg = load_gaussians(work/'inpaint'/'clean_background.ply', device='cuda')
            # A selected-object inpaint keeps all other original objects. Use the produced
            # scene directly, without restoring the all-object removal union.
            self.scene.prompt_backgrounds[frozenset(inpaint_names)] = bg
            if set(inpaint_names) == self.physics.available:
                self.scene.clean = bg; self.state.clean_bg_gs = bg; self.mode = 'clean_all'
            else:
                self.selected = inpaint_names[0]; self.mode = 'clean_selected'
            self.physics.running = False
            save_json(self.out/'active-inpaint.json', {'path': work/'inpaint'/'clean_background.ply',
                                                      'objects': inpaint_names})
        else:
            state = load_scene(self.config, replace(self.state.result_set, out_dir=work), device='cuda')
            self.state = state; self.scene = GaussianScene(state)
            if self.physics.renderer:
                self.physics.renderer.close()
            self.physics = DemoPhysics(state, self.out)
        self.dirty = True

    def render(self):
        from PIL import Image
        t0 = time.monotonic()
        image, mask, depth, w2c, K = self.scene.render(self.camera, self.wh, self.mode, self.selected,
            self.physics.available, self.physics.transforms() if self.mode == 'simulation' else {}, self.highlight)
        if self.mode == 'simulation':
            image = self.physics.overlay(image, depth, w2c, K)
            if self.physics.overlay_mask is not None:
                mask[self.physics.overlay_mask] = 0
        buf = io.BytesIO(); Image.fromarray(image).save(buf, format='JPEG', quality=92)
        self.last_render_s = time.monotonic()-t0
        with self.condition:
            self.frame_id += 1
            self.frames[self.frame_id] = (mask, depth, w2c, K)
            while len(self.frames) > 8:
                self.frames.popitem(last=False)
            self.jpeg = buf.getvalue()
            self.ready = True; self.frame_error = None
            self.condition.notify_all()
        self.status_cache = self.status()
        return image

    def loop(self):
        # Constructed and called on one thread: CUDA and EGL resources never migrate.
        self.render()
        (self.out/'original-highlighted.jpg').write_bytes(self.jpeg)
        save_json(self.out/'live.json', self.status())
        last = time.monotonic()
        while not self.stop.is_set():
            start = time.monotonic(); dt = min(start-last, .1); last = start
            for _ in range(64):
                try:
                    msg, future = self.commands.get_nowait()
                except queue.Empty:
                    break
                try:
                    result = self.execute(msg)
                    if not future.cancelled():
                        future.set_result(result)
                except Exception as exc:
                    if not future.cancelled():
                        future.set_exception(exc)
            try:
                if self.reload_pending:
                    self.reload_build()
                if start-self.last_input > .35:
                    self.keys = []
                moving = bool(self.keys) or bool(np.any(self.look))
                if moving:
                    self.camera.update(self.keys, dt, self.look, self.boost); self.look[:] = 0
                if self.physics.running:
                    self.physics.step(dt)
                if self.dirty or moving or self.physics.running:
                    self.render(); self.dirty = False
                self.status_cache = self.status()
            except Exception as exc:
                self.frame_error = str(exc); self.physics.running = False; self.dirty = False
                self.status_cache = self.status()
                traceback.print_exc()
            self.stop.wait(max(0, 1/self.args.fps-(time.monotonic()-start)))


def handler_for(demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def reply(self, status, data, mime='application/json', headers=None):
            if not isinstance(data, bytes):
                data = json.dumps(data, default=json_default, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', mime); self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/':
                return self.reply(200, (Path(__file__).parent/'web'/'phiview.html').read_bytes(), 'text/html; charset=utf-8')
            if url.path in ('/api/status', '/api/health'):
                return self.reply(200 if demo.ready else 503, demo.status_cache)
            if url.path == '/frame.jpg':
                try:
                    after = int(parse_qs(url.query).get('after', ['-1'])[0])
                except ValueError:
                    return self.reply(400, {'error': 'Invalid frame number'})
                with demo.condition:
                    demo.condition.wait_for(lambda: demo.frame_id > after or demo.stop.is_set(), timeout=10)
                    jpg, fid = demo.jpeg, demo.frame_id
                return self.reply(200, jpg, 'image/jpeg', {'X-Frame-Id': fid})
            self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if self.path != '/api/command':
                return self.reply(404, {'error': 'Not found'})
            # Browser commands are same-origin JSON. Reject cross-site mutation requests.
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                return self.reply(403, {'error': 'Cross-origin commands are disabled'})
            if self.headers.get_content_type() != 'application/json':
                return self.reply(415, {'error': 'JSON required'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('Invalid command length')
                msg = json.loads(self.rfile.read(length))
                if not isinstance(msg, dict):
                    raise ValueError('Command must be an object')
                future = Future()
                demo.commands.put_nowait((msg, future))
                return self.reply(200, future.result(timeout=30))
            except queue.Full:
                return self.reply(429, {'error': 'Command queue full'})
            except Exception as exc:
                return self.reply(400, {'error': str(exc)})
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--scene', default='c50d2d1d42_factory')
    ap.add_argument('--out', required=True)
    from physicalview.config import DEFAULT_CONFIG
    ap.add_argument('--config', default=str(DEFAULT_CONFIG))
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8095)
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--fps', type=float, default=20)
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args(argv)
    if not (64 <= args.width <= 4096 and 64 <= args.height <= 4096 and 1 <= args.fps <= 60):
        ap.error('Resolution must be 64..4096 pixels and fps 1..60')
    demo = Demo(args)
    if args.selftest:
        from physicalview.phiview_check import check_demo
        try:
            check_demo(demo)
        except Exception:
            if demo.physics.renderer:
                demo.physics.renderer.close()
            raise
    server = ThreadingHTTPServer((args.host, args.port), handler_for(demo))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'PhiView: http://{socket.gethostname()}:{args.port}', flush=True)
    try:
        demo.loop()
    except KeyboardInterrupt:
        pass
    finally:
        demo.stop.set(); server.shutdown()
        save_json(demo.out/'final-status.json', demo.status())
        demo.audit.close()
        if demo.physics.renderer:
            demo.physics.renderer.close()


if __name__ == '__main__':
    main()
