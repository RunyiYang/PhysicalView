# PhysicalView

**PhiView demo:** a separate, strictly image-only browser viewport with full Gaussian
rendering, visibility-aware picking/highlights, physics interactions, generated alternatives,
prompted inpainting and robot commands. Run `sbatch run/phiview.sbatch` for the H200 demo.
See [PhiView setup, controls and validation](docs/PHIVIEW.md). The existing viser studio
below remains available.

Interactive, GPU-backed web studio (viser) for the **SimAny real-to-sim pipeline**: load a
scanned room, discover objects, generate 3D assets with a chosen model, register them with a
chosen registration mode, annotate physics, inpaint a chosen region with a text prompt, export
the simulator, and drive a Franka arm with a chosen policy on a chosen task while watching
synchronized MuJoCo and photoreal views.

Runs on any GPU node of the cluster: **RTX A6000, A100-80GB, H200, RTX PRO 6000 Blackwell**
(one CUDA 12.8 environment, gsplat compiled for sm_80/86/90/120). Heavy model stages keep their
own SimAny environments and are dispatched as local or Slurm jobs automatically.

## Layout

```
physicalview/         the app (app.py entry, config, gpu, jobs, pipeline, scene_state, splats,
                      render, ik, robot, smoke) and panels/ (Scene, Generate, Inpaint, Robot, Jobs)
configs/default.yaml  environments, Slurm GPU targets, model registries, simany_root
run/                  setup_env.sh env.sh launch.sh studio.sbatch selftest_gpu.py selftest_all_gpus.sh
docs/                 ARCHITECTURE.md STATUS.md ENV.md
tests/                CPU unit tests (EGL/GPU tests skip without a GPU)
```

PhysicalView depends at runtime on a **SimAny checkout** (`agents/`, `robo/`, `models/`,
`third_party/mujoco_menagerie`, the pipeline stage scripts). Point `simany_root` in
`configs/default.yaml` (or `$SIMANY_ROOT`) at it. The stage scripts need the Studio CLI flags
(`--objects`, `--source-up`, inpainting `--prompt/--backend/--region-box`, ...) that live on the
SimAny branch `feature/studio-ui` until merged.

## Install

```bash
git clone git@github.com:RunyiYang/PhysicalView.git && cd PhysicalView
bash run/setup_env.sh          # uv venv (py3.11) + torch 2.9.1+cu128 + viser/mujoco/gsplat ...
# on a GPU node the script also JIT-builds gsplat for 8.0;8.6;9.0;12.0 into .envs/studio/torch_extensions
bash run/selftest_all_gpus.sh  # optional: PASS/FAIL matrix on all four GPU types
```

On this cluster the built env already exists; `.envs/studio` may simply be a symlink to it.

## Launch

```bash
# batch job (GPU type chosen at submit time; log prints the ssh tunnel line)
sbatch run/studio.sbatch                                                          # A6000 (debug)
sbatch --partition=batch --gres=gpu:h200:1 --exclude=msp3-[0-7] run/studio.sbatch # H200
sbatch --partition=batch --gres=gpu:a100-80g:1 run/studio.sbatch                  # A100
sbatch --partition=batch --gres=gpu:rtx6000:1 -w gcp-eu1-rtx6000-vz3w run/studio.sbatch
PORT=8081 sbatch run/studio.sbatch                                                # custom port
tail -f outputs/logs/studio_<jobid>.log

# interactive
bash run/launch.sh --gpu a6000|h200|a100|rtx6000 [--port 8080]

# from your laptop
ssh -L 8080:<node>:8080 <login-host>     ->  http://localhost:8080
```

Headless end-to-end check (no browser): `source run/env.sh && $STUDIO_PY -m physicalview.smoke`.

## Tabs

* **Scene** — pick any result set (ScanNet++ factory/auto/full, DROID, phone video, BEHAVIOR),
  layers (raw / clean background, mesh, object splats, collision), object table, selection and
  gizmos, camera snap, photoreal snapshot. **Display mode**: *Server render* (default) renders
  the view on the GPU with gsplat and streams JPEG frames as the viewer background, so the
  browser holds no splat data (JPEG quality and 720p/1080p/native controls; the robot is
  composited in); *Client splats* uploads capped splat arrays to WebGL (high browser memory).
* **Generate** — discovery model (GT segments / SAM3 auto), generation model (TRELLIS,
  ReconViaGen, SAM 3D Objects, evidence-gated hybrid), registration mode (yaw-sweep ICP, signed /
  alternative source-up), per-object generate / register / physics, drop-test report, MJCF export
  (full-room or shim collision), task generation, full pipeline; proposal cards with
  construction-time evidence and "Accept proposal".
* **Inpaint** — region = selected object or a draggable 3D box; text prompt + negative prompt;
  backend Qwen-Image-Edit or LaMa; refine iterations; versioned clean backgrounds (before/after).
* **Robot** — pick or author a task (target + receptacle/region + instruction); policy
  (pi0.5 DROID, pi0.5 sim-cotrained, scripted); start/check the policy server; raster or
  photoreal-composite observations; run/stop an episode with the grasp·lift·hover·place stage
  bar; 7 joint sliders + gripper; end-effector gizmo driven by damped-least-squares IK; robot
  geoms live in the 3D view (client mode) or are composited into the server render stream.
* **Jobs** — every stage is a job (local when the node's GPU supports that stage's env,
  otherwise `srun` to a compatible GPU type), live logs, cancel, GPU compatibility matrix.

## Verified

See docs/STATUS.md: headless smoke ALL PASS on hala (A6000) and gcp-eu1-rtx6000-vz3w
(RTX PRO 6000 Blackwell); env self-tests PASS on A100-80GB and H200.

## License

MIT (code). Model weights and third-party pipelines keep their own licenses.
