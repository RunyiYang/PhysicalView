# PhiView paper image campaign

Requested scope: 10 ScanNet++ scenes, then 10 LIBERO layouts, then 10 BEHAVIOR
 task-scene configurations. Each dataset has 14 feature groups per scene (the
 second `f` in the request is named `f2_clean_all`): 420 requested scene/feature
 groups in total. This is qualitative demo evidence, not a benchmark evaluation.

Output root: `/group/worldcept/code/PhysicalView/outputs/paper-capture-20260913`.
`roster.json`, `libero-roster.json`, and `behavior-roster.json` identify sources.

ScanNet++ roster: `c50d2d1d42`, `45b0dac5e3`, `825d228aec`, `7b6477cb95`,
`578511c8a9`, `27dd4da69e`, `38d58a7a31`, `f9f95681fd`, `3e8bba0176`, `3864514494`.
Existing object/inpaint/simulation assets were copied to isolated campaign folders.
Six existing MJCF exports passed compilation after relative asset paths were repaired
in the copies. Original experiment artifacts remain unchanged.

## Capture and evidence

- All original Gaussian parameters remain loaded. Export is lossless PNG at
  2880×1920; this render size does not assert that the source observations have
  that resolution. Screenshots are saved directly from renders without post-render
  retouching. Prompted inpainting is an explicit, separately recorded scene operation.
- Survey the actual rendered object visibility, then move a real camera for
  target framing. Save the chosen pose before actions. Reuse that exact choice
  across prompted construction and final captures.
- Each PNG has a JSON sidecar with camera matrices/intrinsics, source resolution,
  object selection, active mass/inertia/friction, qpos/qvel, quality diagnostics,
  selected pixel bounds, and SHA-256. Full-scene context and target details are
  separate renders. Fall/friction/throw/robot have state traces and footage.
- Re-run Qwen-Image-Edit-2511 for clean backgrounds and a selected-object prompt,
  then fit replacement Gaussians for 1,000 iterations. Guards reject cached edits
  and non-Qwen fallback. Prompt text, per-view outcomes and model receipts persist.
- Fresh SAM3 has its own source image, binary masks and scores. It is single-view
  image inference using scene class prompts. Existing Gaussian object proposals
  include GT-assisted metadata; the highlight images do not establish exhaustive
  GT-free discovery. Keep these two evidence types distinct in paper captions.
- `features.json` separates captured, failed, and missing features. Captured only
  establishes execution and its stated checks. Publication quality is reviewed
  separately. Failed and interrupted attempts must not be counted as completed.
- SVG panels embed unchanged PNGs and keep titles/parameter text editable.
  `python -m physicalview.paper_pack --root OUTPUT_ROOT` creates `gallery.html`,
  `coverage.csv`, `coverage.json`, and per-feature `panel.svg` files.

## Dataset adapters

LIBERO: 10 distinct native layouts; original HDF5 `demo_0` state and XML retained.
All referenced meshes/textures resolved and all 10 geometry preflights passed.
The adapter freezes articulated fixtures at the recorded state and omits the native
robot before capture, allowing PhiView to add its own arm. Remaining geom positions
are checked against the recorded native model. It renders 80 posed RGB-D views at
1536×1024, then runs 15,000-step Gaussian training and the construction stages.
This is native simulator, GT-pose/GT-mesh assisted evidence; not real-world capture.
GPU rendering/training is a separate gate from the successful CPU preflight.

BEHAVIOR: 10 WDS task-scene sources indexed with explicit shard hints. Native RGB
observations are 320×180; previous 1280×720 exports are upscaled. The adapter uses
posed observations, derived geometry, SAM3 discovery, and generated assets. It does
not manufacture high-resolution detail. Distinct task IDs are not proof of ten
independent physical environments. These limitations must survive figure selection.

The requested order is ScanNet++ → LIBERO → BEHAVIOR. Start a bounded first scene
for each new adapter before releasing its remaining scenes; a queued or prepared
adapter is not completed evidence.

## Initial attempts

The initial H200 pilot (893906) completed 13 non-prompt features but was rejected
for publication: stacked floor boxes were a weak interaction target and inherited
clean backgrounds had visible remnants. Results remain under `pilot/`.

The first campaign exposed virtual-environment Python symlink resolution: resolving
`bin/python` to its base executable lost installed packages. The config loader now
resolves the parent directory while preserving the executable path; regression test
added. Original attempt logs are retained; retries write histories under `attempts/`.

The early `previews/38d58a7a31` run inherited four Slurm tasks and shared output paths;
its review record marks it invalid. The single-task preview was interrupted when its
parent allocation completed. Neither is included in the official scene/feature count.

Current scheduler and completion state must be read from Slurm and the per-scene
`campaign-status.json` files; job submission alone does not establish completion.

## Visual review and repair experiments (2026-09-14)

The completed first scenes are **draft captures**, not approved paper figures.
Scene `3864514494` has cup remnants in its clean backgrounds and the initial robot
placement was hidden by cabinets. `3e8bba0176` also has background remnants and one
registered RVG alternative with black texture defects. The bathroom scene has
visible completion patches. Per-feature `review.json` files record these failures.

The original Qwen adapter crops and composites with a mask but does not pass that
mask to the model. Actual view inspection found preserved/moved cups instead of
empty backgrounds. `physicalview.phiview_mask_edit` is an **experimental** runner
that saves its actual input, localization reference/mask, raw model output, prompt,
seed and output hash. Trials include Qwen reference/hole inputs, SDXL inpainting,
and a recorded LaMa prefill followed by low-strength prompted SDXL refinement.
Model inference crops may be resampled; this does not change the recorded source
resolution or justify claims of recovered image detail. These trials have not been
adopted as the campaign default. Their outputs are outside the official dataset
folders, and semantic removal still requires visual review.

`inpaint_surface_seeds` / `inpaint_surface_fill` are a separate experimental
background-geometry refinement. They seed on carved scan geometry and bounded local
plane extrapolations, train on every edited view, and report **zero held-out views**.
Their fit PSNR is not a generalization result or a publication-quality gate.

Robot placement now starts beside the current viewing direction rather than a
fixed world axis. The mount height is estimated and support is not verified.
`run/paper_robot_probe.py` records actual contacts, trajectories and endpoint error.
The IK solution residual and the live end-effector tracking error are distinct:
the kitchen arm was physically blocked by cabinets despite a small IK residual.
The bathroom push probe produced contacts and displacement; this is neither a
successful-grasp result nor a robot benchmark. Camera occlusion still needs review.
Paper feature restoration rebuilds the original physics model after robot/projectile
trials so these actors do not contaminate later feature comparisons.

`--features` permits a bounded recapture; the completion denominator remains 14.
`--robot-command` records the actual command used. Source hashes now cover the
renderer/physics adapters as well as capture modules.

Dataset transitions use `physicalview.paper_require_complete`: all ten roster
scenes must have all fourteen captured features with existing PNGs and sidecars.
This gate is distinct from publication review. `scheduler-chain.json` records the
submitted ScanNet++ -> LIBERO first scene -> remaining LIBERO -> BEHAVIOR first
scene -> remaining BEHAVIOR dependencies. A failed gate prevents later datasets
from silently proceeding on an incomplete predecessor.

A wider source-photo audit now covers 50 existing ScanNet++ factory outputs;
`wider-source-photo-review.jpg` is explicitly a **source-photo selection aid**, not
PhiView output. Less cluttered tabletop examples are candidates for subsequent
replacement if the initial scene group cannot meet the requested visual quality.
The refined cup trial under `lama-sdxl-repair/` reduced its silhouette but still
failed novel-view visual review; its trained-view fit score must not be used to
claim a clean reconstruction.

Bounded feature retries archive the replaced feature folders and previous evidence
under `feature-attempts/`, preserve unrelated groups, and clear stale approvals for
changed images. Each new PNG sidecar includes the renderer/capture source hashes.

LIBERO physics retains the original simulator contact geometry, masses, inertias
and coefficients at the recorded scene state. The adapter maps accepted generated
object IDs onto native free bodies; rejected proposals remain static fixtures.
All named geom positions are checked before/after the mapping. The native robot is
omitted, and choosing a registered generated alternative still rebuilds that body's
collision model. This is explicitly GT-assisted native-simulator physics, distinct
from ScanNet++'s reconstructed contact geometry.

BEHAVIOR extraction now explicitly sets `upscale=1` (the upstream extractor otherwise
automatically upscales to 1280 pixels). It also restricts input views to one episode's
first static object state, checking shared mesh translations within 2 cm and rotations
within 5 degrees after registration. It chooses the candidate episode with the most
consistent observed camera frames and requires at least eight before training.
`pose-selection.json` retains all acceptance/rejection measurements. Camera alignment
and this state filtering use dataset mesh trajectories, so these poses are GT-assisted.
This avoids silently mixing changed manipuland poses across episodes into one static
Gaussian scene; actual BEHAVIOR extraction remains a pending execution gate.

Room overview selection is now separate from the interaction target close-up. It
scores up to 49 observed cameras using visible proposals, pitch and black-pixel
coverage, without examining generated or simulated outcomes. The actual H200
`context-repair/c50d2d1d42` trial improved desk context; prior official frames are
preserved. `context-survey.json` records every candidate and the selected camera.

Native BEHAVIOR assets were subsequently located at
`SimAny/third_party/behavior1k_datasets`. All 51 scene configurations have their
referenced main encrypted USD object files, and OmniGibson 3.9.0 imports in the
preinstalled `/group/streetsplat/.conda/envs/behavior1k` environment. This does not
validate texture dependencies or renderer execution. A bounded A6000 RGB readiness
probe is queued after LIBERO; the BEHAVIOR first scene waits for its termination.
The WDS adapter remains the default until a native capture route is demonstrated.
The probe has no feature-count or paper-quality credit. Account QoS rejected the
`rendering` partition; the accepted submission uses `batch` with `normal` QoS.

`run/paper_carve_probe.py` compares observed Gaussians before/after removal without
generated fill. This is a diagnostic for separating residual source geometry from
fill artifacts, not evidence of prompt-conditioned completion. Its outputs remain
outside the official scene folders.

An exact observed-camera renderer diagnostic compared classic and antialiased
gsplat rendering while keeping all source Gaussian rows/parameters. For
`3864514494/DSC06115.JPG`, PSNR against that source frame was 27.8445 versus
30.6737 dB; for `c50d2d1d42/DSC01757.JPG`, 27.2389 versus 27.7714 dB. These two
checks are not a held-out evaluation. A quaternion-order diagnostic was worse and
was not adopted. PhiView now uses antialiased rasterization for both RGB/depth
and object-ID picking, and records the mode in manifests and PNG sidecars.
The actual H200 cup trial passed original/highlight/click/clean capture checks;
its clean background still has visible artifacts and is not approved for paper.
Job `894256` recaptures all ten constructed ScanNet++ scenes, archiving earlier
frames, before releasing LIBERO. It does not rerun or relabel construction models.

Paper panels keep fall and friction in separate rows and include all eight
navigation frames. The gallery uses lightweight review JPEGs; the editable SVGs
still embed the unchanged lossless PNG files. `panel-layout.json` records which
frames appear in each panel.
