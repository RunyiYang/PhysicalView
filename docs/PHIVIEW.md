# PhiView

PhiView runs Gaussian rendering, object picking, highlights, camera motion, model stages,
and MuJoCo physics on the server. The browser displays JPEG frames in one HTML image
and sends input events. There is no canvas, WebGL, mesh, Gaussian array, or client physics.

## Run the H200 demo

```bash
cd /group/worldcept/code/PhysicalView
sbatch run/phiview.sbatch
```

The job runs real-GPU checks before starting port 8095. The log prints the SSH tunnel
command. Default scene: `c50d2d1d42_factory`; full original 1,499,998 Gaussians, SH degree 3.
The stream defaults to 1920×1080; Native resolution uses 1752×1168 for this capture.
No Gaussians are subsampled at either resolution. JPEG compression is used for transport.

```bash
source run/env.sh
"$STUDIO_PY" -m physicalview.phiview --demo --selftest \
  --scene c50d2d1d42_factory --out outputs/my-phiview-run --port 8095
```

Run the command inside a GPU allocation. A session shares one scene, selection and camera
among viewers. Job duration defaults to four hours. Files persist after the job ends.

## Controls

- WASD moves relative to the view; Q/E moves vertically. Shift triples speed. Right-drag
  rotates without roll; pitch stops short of the poles. The speed slider adjusts travel.
  F focuses the selected object. Inputs stop on window blur or a 350 ms heartbeat timeout.
- Click an outlined object, or choose it in the object list. Selected outlines are amber;
  other discovered objects are teal. Picking uses the membership raster for the displayed
  frame, including occlusion. Expired frame clicks return an error rather than selecting
  something at a newer camera pose.
- Original displays only the original splats. Simulatable displays the inpainted background
  plus original or chosen generated object appearance. Native restores the capture resolution.
- Make simulatable enables a constructed collision body. Build physics invokes registration,
  physics annotation and room collision export in a new writable build. Unsupported objects
  remain disabled until their body exists.
- Physical parameters show the active MuJoCo mass, inertia and friction alongside the source
  estimate. Friction edits affect the selected body's collision geoms. Fall raises the body
  0.3 m and releases it; Friction gives it 0.7 m/s horizontal velocity; Throw gives it the
  chosen camera-directed velocity plus upward velocity. Run/Pause/Reset control simulation.
- Shooting sends a real sphere along the clicked camera ray. The server simulates contacts,
  gravity and friction. Eight projectiles are reused. A projectile's speed is bounded at
  20 m/s. This is a discrete rigid-body simulation, not a ballistic penetration model.
- Generated alternatives need a valid registration and mesh. Choosing one rebuilds its
  physical collision hull and inertia as well as its visual Gaussians. Physics resets and
  the arm is removed when collision geometry changes. Collision uses a convex hull for
  these interactive switches; Build physics provides the pipeline's collision decomposition.
- Inpainted selected/all hides the respective objects using cached 3D clean backgrounds.
  Enter a prompt to run Qwen-Image-Edit-2511 and fit replacement background Gaussians.
  Each edit uses a new writable build. Existing masks can be reused; previously edited
  images are deleted only in the new copy, so the new prompt must execute. A guard rejects
  skipped/cached edits and LaMa fallback. Completed prompt versions are keyed by the exact
  removed-object set. The panel reports queued/running/failed/succeeded state and job paths.
- Place robot puts a Panda/Robotiq arm within reach of the selected object. Execute accepts
  reach, lift, pick/place left/right, move left/right, and push left/right commands. It runs
  bounded scripted IK through joint actuators and physical contacts. Unsupported commands
  are rejected. Reports distinguish completed motion, IK error and measured lift; no motion
  completion is labelled as grasp or task success.

## Environment

`configs/phiview.yaml` uses the shared Studio CUDA environment for rendering, the original
SimAny environments for generation, and `run/phiview_inference_python.sh` for SAM3 and Qwen.
The latter adds isolated packages under `.envs/phiview-inference` and official SAM3 source
under `.envs/phiview-sam3-source`; it does not modify the shared Studio environment.
`run/setup_phiview_inference.sh` restores these dependencies. Available shared weights:

- SAM3: `/group/worldcept/hf_cache/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt`
- Qwen 2511: `/group/worldcept/hf_cache/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`

The Qwen path can be overridden with `PHIVIEW_QWEN_MODEL`. Existing TRELLIS, ReconViaGen,
SAM3D and physics model availability is checked by their real pipeline stages. A failed
stage remains failed; it is not replaced with a dummy asset.

## Evidence and boundaries

Every run writes `manifest.json`, `actions.jsonl`, snapshots and `validation/checks.json`.
Checks cover pixel-exact original rendering, native resolution, real image picking,
selected/all removal, fall/friction, throw video, projectile contacts, generated alternatives
and robot command motion. Failed attempts remain in their own output directories.

The default scene contains 18 existing proposals and 16 constructed bodies. Two proposals
use approximate bounding-region selection; 16 use cached removal masks. Proposal metadata
includes GT references. Automatic highlighting is not evidence of exhaustive new discovery;
Discover with SAM3 invokes a fresh discovery into an empty build and invalidates prior
object/collision associations.

The inherited collision scene has support slabs, not independently verified full-room
collision. The inherited clean background leaves visible remnants around transparent
objects. Prompt editing is an additional construction step; image quality is not guaranteed
by a passing endpoint or mask test. Fresh selected-object prompt versions have no regenerated
3D object labels, so selection in that view uses the object list until switching back.

The scripted robot controller supports the listed command vocabulary. It is not a general
language policy, and a successful grasp or placement must be established from the actual
simulation, not inferred from a completed trajectory. These demo artifacts are separate
from paper evaluation results.

Rendering uses [gsplat's rasterization API](https://docs.gsplat.studio/main/apis/rasterization.html)
for full SH color, depth and object-membership passes; physics uses MuJoCo.
