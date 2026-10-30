# sim_xarm — xArm7 pouring simulation for FAVOR

MuJoCo environment for the real-robot pouring task (`claude/FAVOR_real_robot_experiment_design.md`):
xArm7 with a cup fixed in the closed gripper, 50.0 g of spherical beads, a 12 cm bowl on a scale,
two cameras (fixed side + wrist), 10 Hz absolute joint targets (7-D action, no gripper command).

Nothing outside `sim_xarm/` is changed except the joint-limit hook in the shared IK modules (see M3),
which leaves the Panda results bit-identical.

## Pinned versions

| component | version / commit | note |
|---|---|---|
| MuJoCo (python) | `mujoco==3.17.0` | scene built with `MjSpec`; Menagerie requires >= 2.3.3 |
| MuJoCo Menagerie | `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` | `ufactory_xarm7` last changed in `71f066a` (2026-07-04) |
| xarm_ros2 | `ac736c92b02de7ca7e0be6f737d77d46707464d8` | source of `config/kinematics/default/xarm7_default_kinematics.yaml` |
| pytorch_kinematics | `0.10.0` | |
| torch | `2.15.0` (CPU build used for validation) | |
| numpy / h5py / opencv-python-headless / PyYAML | 2.5.3 / 3.16.0 / 5.0.0.93 / 6.0.3 | |
| Python | 3.13 | the LIBERO docker image is older; run sim_xarm in its own env |

Setup:

```bash
cd sim_xarm
pip install -r requirements.txt
./setup_assets.sh                     # Menagerie at the pinned commit -> sim_xarm/third_party/
export MUJOCO_GL=egl                  # GPU server; use osmesa on CPU-only machines
```

## Menagerie `ufactory_xarm7`: requirements and known limits (M0)

From the model README and CHANGELOG at the pinned commit:

- Requires MuJoCo 2.3.3 or later. A *simplified* description derived from the ROS1 `xarm_ros` URDF.
- 2024-12-17: finger collision pads added, `armature = 0.1` on all joints.
- Actuators: position servos (`general`, affine bias = PD), Menagerie gains below. No joint velocity
  limit in the model; the official 180 deg/s is imposed by our command layer.
- `link_base` sits 0.12 m above the world origin in the Menagerie file (pedestal); we mount it at
  `robot.base_pos` instead.
- The gripper base is rotated 180 deg about the flange z axis relative to link7 (`quat="0 0 0 1"`),
  whereas the xarm_ros2 `xarm_gripper` attaches with identity rotation. Positions along the flange
  axis are unaffected; the in-plane orientation of the gripper differs. **TODO(real): check which
  matches the mounted gripper** (matters only for the gripper/TCP frame, not for link7).

Full numbers: [`reports/m0_model.md`](reports/m0_model.md). Summary vs the official specification:

| joint | Menagerie range (deg) | official (deg) | kp / kv | force limit (Nm) |
|---|---|---|---|---|
| J1 | ±360 | ±360 | 1500 / 150 | ±50 |
| J2 | −118.0 .. 120.0 | −117 .. 116 | 1500 / 150 | ±50 |
| J3 | ±360 | ±360 | 1000 / 100 | ±30 |
| J4 | −11.0 .. 225.0 | −6 .. 225 | 1000 / 100 | ±30 |
| J5 | ±360 | ±360 | 1000 / 100 | ±30 |
| J6 | −97.0 .. 180.0 | −97 .. 180 | 800 / 80 | ±20 |
| J7 | ±360 | ±360 | 800 / 80 | ±20 |

- J2 (+4°/−1°) and J4 (−5°) are wider than the official ranges → the scene and the IK use the
  official ranges (`robot.joint_limits_deg`).
- With the Menagerie gains, gravity alone leaves a steady-state tracking error of −0.52° (J2) and
  +0.89° (J4) at the home keyframe (≈ 5–10 mm at the cup). The real xArm servo holds position to
  ±0.1 mm repeatability, so the scene gravity-compensates the arm (`physics.gravity_comp`).
- The force limits have not been checked against the official rated joint torques (not in the
  public technical specification page). **TODO(real): check**.
