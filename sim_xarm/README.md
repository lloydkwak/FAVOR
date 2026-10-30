# sim_xarm — xArm7 pouring simulation for FAVOR

MuJoCo environment for the real-robot pouring task (`claude/FAVOR_real_robot_experiment_design.md`):
xArm7 with a cup fixed in the closed gripper, 50.0 g of spherical beads, a 12 cm bowl on a scale,
two cameras (fixed side + wrist), 10 Hz absolute joint targets (7-D action, no gripper command).
Uses: simulation data for the real policy (zero-shot / co-training), xArm7 fault experiments in
simulation, and offline safety checks of corrected trajectories before real runs.

Outside `sim_xarm/` only the joint-limit hook in the shared IK modules changed (M3); the Panda
results are bit-identical (`tests/test_panda_ik_unchanged.py`). IK parameters are unchanged.

## Pinned versions

| component | version / commit | note |
|---|---|---|
| MuJoCo (python) | `mujoco==3.17.0` | scene built with `MjSpec`; Menagerie requires >= 2.3.3 |
| MuJoCo Menagerie | `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` | `ufactory_xarm7` last changed in `71f066a` (2026-07-04) |
| xarm_ros2 | `ac736c92b02de7ca7e0be6f737d77d46707464d8` | joint origins of `assets/xarm7_favor.urdf` |
| pytorch_kinematics | `0.10.0` | |
| torch / torchvision | `2.15.0` / `0.29.2` (CPU, validation machine) | |
| numpy / h5py / opencv-python-headless / PyYAML | 2.5.3 / 3.16.0 / 5.0.0.93 / 6.0.3 | |
| diffusion_policy (data loader + training check) | upstream `5ba07ac` + zarr 2.18.4, numcodecs 0.15.2, diffusers 0.11.1, robomimic 0.2.0 | see "diffusion_policy" below |
| Python | 3.13 | the LIBERO docker image is older; run sim_xarm in its own env |

```bash
cd sim_xarm
pip install -r requirements.txt
./setup_assets.sh                    # Menagerie at the pinned commit -> sim_xarm/third_party/
export MUJOCO_GL=egl                 # GPU server (osmesa on CPU-only machines: ~0.2 s per frame)
```

All randomness is seeded: episode seed -> bowl cell / beads jitter, `seed + 7` -> plan parameters,
`[seed, 77]` -> domain randomisation, `(seed, call)` -> policy noise in rollouts.

## Layout

| path | content |
|---|---|
| `config/default.yaml` | **the one config tree**: scene dimensions, cameras, control, beads, task, randomisation. Every placeholder is marked `TODO(real): measure`. Override with `--config real.yaml` or `--set a.b=v` |
| `favor_xarm/scene.py` | MjSpec scene: table, xArm7 (+ xArm gripper or Robotiq 2F-85), cup (rigidly on the gripper), bowl on a scale, beads, cameras |
| `favor_xarm/env.py` | `PourEnv`: reset / 10 Hz `step(q_target)` / metrics (f, spill, cup residue, contact-force scale check) / safety flags / observations |
| `favor_xarm/control.py` | command layer: Mode 1 (servo stream) and Mode 6 (online re-plan) emulation, speed limit, latency, velocity feedforward |
| `favor_xarm/kinematics.py` | `XArm7Kinematics`, same interface as `PandaKinematics`, plus `q_lo/q_hi` |
| `favor_xarm/scripted.py` | scripted pouring planner |
| `favor_xarm/faults.py` | locked / range faults, mid-task onset |
| `favor_xarm/correction.py` | B1 / W-IK pos / W-IK pose / Priority IK / orientation-first on a policy chunk |
| `favor_xarm/randomization.py` | domain randomisation (off by default) |
| `dp/` | diffusion_policy task yaml, dataset class, null runner |
| `scripts/m*_*.py` | milestone scripts; results in `reports/` |
| `tests/` | `test_ik_xarm.py`, `test_panda_ik_unchanged.py`, `test_camera.py`, `test_faults.py`, `test_dp_dataset.py` |

## Scene decisions that differ from a first guess

- **Cup mount `hang` (default)**: the gripper points down and pinches a vertical handle plate on the
  cup side; the cup hangs beside the fingers. A forward-pointing gripper (cup axis perpendicular to
  the approach axis, option `cup.mount: side`) cannot reach the near grid rows (J6 lower limit).
- **Pouring direction**: yaw −120° (lip towards the robot's back-right), the only direction in a
  12-way sweep that is feasible over the whole grid at 125° tilt with ≥ 0.48 rad joint margin.
  Home: `[0.2517, −0.1347, −0.1080, 1.1221, −0.0152, 1.2560, −0.8979]` (cup mouth (0.40, 0, 0.30) m).
- **Physics**: CG solver, 20 iterations, 2 ms (Newton: ~700 ms per step with ~2,500 bead DoFs;
  10 CG iterations let beads through the walls). Cup and bowl have invisible collision shells 8 mm
  thicker than the visual walls (beads tunnelled through 3–4 mm walls).
- **Servo**: Menagerie PD gains with gravity compensation and velocity feedforward (tracking lag
  0.03–0.06° at 0.5 rad/s instead of 2.9°). All gains, damping, latency are config parameters.
- **Images**: row 0 = top, as a RealSense frame; unlike the LIBERO files, no flip (checked by
  projection in `tests/test_camera.py`). fx/fy/cx/cy intrinsics are realised exactly at render time.
- **Quaternion**: `robot0_eef_quat` is xyzw and made sign-continuous along each episode (the
  canonical w ≥ 0 form jumps when the pour rotation passes 180°). Apply the same rule on the robot.

## Menagerie `ufactory_xarm7`: requirements and known limits (M0)

From the model README and CHANGELOG at the pinned commit: requires MuJoCo >= 2.3.3; a simplified
description derived from the ROS1 `xarm_ros` URDF; 2024-12-17 finger pads + `armature = 0.1`.
Position servos (`general`, affine bias = PD), no velocity limit in the model. `link_base` sits 0.12 m
above the origin (pedestal) — we mount it at `robot.base_pos`. The gripper is rotated 180° about the
flange axis relative to link7, whereas xarm_ros2 attaches it with identity (**TODO(real): check**).

| joint | Menagerie range (deg) | official (deg) | kp / kv | force limit (Nm) |
|---|---|---|---|---|
| J1 | ±360 | ±360 | 1500 / 150 | ±50 |
| J2 | −118.0 .. 120.0 | −117 .. 116 | 1500 / 150 | ±50 |
| J3 | ±360 | ±360 | 1000 / 100 | ±30 |
| J4 | −11.0 .. 225.0 | −6 .. 225 | 1000 / 100 | ±30 |
| J5 | ±360 | ±360 | 1000 / 100 | ±30 |
| J6 | −97.0 .. 180.0 | −97 .. 180 | 800 / 80 | ±20 |
| J7 | ±360 | ±360 | 800 / 80 | ±20 |

J2 and J4 are wider than the official ranges → the scene and the IK use the official ranges.
Gravity alone gives −0.52° (J2) / +0.89° (J4) steady-state error with these gains → gravity
compensation. Force limits not checked against rated torques (**TODO(real)**). Full: `reports/m0_model.md`.

## Running the milestones (one line each, from `sim_xarm/`)

```bash
python scripts/m0_inspect_model.py
python scripts/solve_home.py                                   # prints home_q for config
python scripts/m1_scene_check.py                               # reports/m1_scene.md numbers
python scripts/m2_step_response.py                             # reports/m2_step_response.png
python scripts/m3_fk_check.py && python tests/test_ik_xarm.py && python tests/test_panda_ik_unchanged.py
python scripts/m4_generate.py --out out/m4 --reps 10 --workers 16 --save-states
python scripts/m4_report.py --run out/m4
python scripts/m5_export.py --run out/m4 --out out/xarm_pour_200.hdf5 --n 200 --filter success --workers 16
DIFFUSION_POLICY_ROOT=../diffusion_policy python tests/test_dp_dataset.py out/xarm_pour_200.hdf5
python scripts/m6_diagnostics.py --reps 10
python scripts/m7_dr_grid.py
python tests/test_camera.py && python tests/test_faults.py
```

Large-scale data with continuous bowl positions: `m4_generate.py --set task.placement=uniform --n N`;
domain randomisation: `--set randomization.enabled=true`. Generation resumes from `summary.csv`.

## diffusion_policy

`third_party/DIFFUSION_POLICY_COMMIT` (`20537a5`) is not in the public diffusion_policy repository,
and the FAVOR patch modifies `robomimic_replay_joint_action_dataset.py`, which upstream does not
have. sim_xarm therefore does not depend on that dataset class: `dp/xarm_pour_dataset.py` is
self-contained on upstream components (replay buffer, sampler, normalisers; no pytorch3d) and
range-normalises the 7-D joint action (upstream `RobomimicReplayImageDataset` would give it an
identity normaliser with `abs_action: False`, or a rot6d conversion with `True`).

```bash
cp sim_xarm/dp/xarm_pour_image_joint.yaml diffusion_policy/diffusion_policy/config/task/
export PYTHONPATH=$PWD/sim_xarm/dp:$PWD/sim_xarm:$PYTHONPATH
cd diffusion_policy && python train.py --config-name=train_diffusion_unet_hybrid_workspace.yaml \
  task=xarm_pour_image_joint task.dataset_path=/path/xarm_pour.hdf5 +checkpoint=joint_trainloss_topk
```

Rollouts and fault conditions: `scripts/m8_rollout.py --ckpt ... [--fault-joint 6 --method "Priority IK"]`
(import order matters: torchvision before mujoco, else triton segfaults with the GL backend).
