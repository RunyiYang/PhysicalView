# Image selection and robot cameras

PhiView keeps rendering, segmentation, collision construction, policy inference,
and physics on the server. The browser displays JPEGs and sends input events.

* Click selects a known object or requests a SAM3 point mask for a new region.
* Left-drag draws a bounding box. A dominant existing object is selected directly;
  otherwise SAM3 receives the box on the exact displayed frame. Dragging pauses
  physics and pins that frame for up to 60 seconds. Right-drag still rotates the
  free camera. Selected objects stay bright red with a white outline when other
  highlights are disabled.
* **Make simulatable** creates a convex collision proxy from the selected visible
  Gaussian surface. Hidden geometry is unknown; mass and friction are unmeasured
  defaults. Object records and proxies are scoped to their source build so IDs
  from rediscovery cannot inherit an unrelated collision body.

Robot hardware and control policies have separate selectors. The installed rigs
are Franka with Robotiq 2F-85 (DROID) and Franka with its native Panda hand. π0.5
DROID policies require the former. Scripted IK works with either and remains
explicitly labeled as scripted.

The third-person view is an actual fixed exterior MuJoCo camera, positioned
at `[0.05, 0.57, 0.66]` m in the DROID base frame and aimed at
`[0.55, 0, 0.10]` m, with a 68-degree vertical field of view. The wrist camera is
attached to the gripper and uses the existing DROID rig's 58-degree field of view.
These are simulated cameras following the existing π0.5 rig preset; they are not
calibration measurements from a physical camera in the scanned room. Choose
**Free view** to navigate; WASDQE and right-drag do not move a fixed camera.

The two π0.5 checkpoint choices come from PhiRoom's policy registry. Executing a
command starts one private, session-owned checkpoint server using the configured
`openpi` interpreter. It does not connect to arbitrary existing policy servers.
First loading and JIT compilation can take minutes. Status shows loading,
inference, execution, or failure. Checkpoint and response receipts are stored in
the session's `policies/` directory.

Policy inputs are unhighlighted exterior and wrist images rendered at 1280×720,
resized with padding to 224×224, seven joint positions, gripper position, and the
user's command. The same physical camera transforms drive the viewer's camera
choices. Actions are chunks of 15 absolute 8D targets (seven joints and gripper),
executed at 15 Hz of simulation time with 40 physics substeps and a 0.2 rad joint
delta cap. A default command runs 150 ticks. This is not a promise of 15 FPS wall
time or manipulation success. Pause/reset cancel execution, and late inference
responses cannot restart it. The simulation has no physical robot connection.

Modules: `phiview_box.py` handles rectangle selection; `phiview_selection.py`
handles discovery and persistence; `phiview_rigs.py` handles hardware and camera
contracts; `phiview_policy.py` handles observations and asynchronous control;
`phiview_policy_server.py` loads checkpoints in their separate environment.
