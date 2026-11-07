# sim_xarm — xArm7 pouring simulation for FAVOR

MuJoCo environment for the real-robot pouring task (`claude/FAVOR_real_robot_experiment_design.md`):
xArm7 holding a cup by its side in the closed gripper, 50.0 g of spherical beads, a 12 cm bowl,
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
| `favor_xarm/scene.py` | MjSpec scene: table, xArm7 (+ xArm gripper or Robotiq 2F-85), cup (rigidly in the gripper), bowl, beads, cameras |
| `favor_xarm/env.py` | `PourEnv`: reset / 10 Hz `step(q_target)` / metrics (f, spill, cup residue, contact-force cross-check) / safety flags / observations |
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

Numbers in `reports/m1*`, `m4*`, `m5*`, `m6*`, `m8*` before the commit "side grip" were measured with the
earlier hanging-cup layout (yaw −120°); re-run them for the current layout (README commands).


- **Cup mount `grip` (default)**: no handle; the fingers pinch the cup body from the side (pads at
  25 mm above the inner bottom). The finger command is computed from the cup diameter (68 mm → 74
  of 255 on the xArm gripper, 89 mm maximum opening). The bowl stands directly on the table (no
  scale plate; `bowl.scale_height > 0` adds one). Earlier layouts with a handle (`hang`, `side`)
  remain as options.
- **Roll pour**: the lip is on a finger side (`cup.lip_yaw_deg: 90`) and the cup turns about the
  gripper's approach axis. A sweep of lip side (0°, ±90°) × pour direction (12 yaws) over the grid
  corners found: lip on the far side from the palm (forward pour) — no direction feasible; lip on a
  finger side with the lip 30–50 mm above the rim — the gripper body crosses the rim; with the lip
  ≥ ~65 mm above the rim, pour direction 0° (away from the robot) is collision-free everywhere with
  ≥ 0.5 rad margin on J2/J4/J6 (`task.scripted.lip_height: [0.065, 0.085]`).
  Home: `[0.7423, 0.3094, −0.1474, 1.0617, −1.0825, 1.9405, 2.4654]` (cup mouth (0.40, 0, 0.30) m, lip +x).
- **Cameras = RealSense D435** (both), colour 640 × 480, vertical FOV 42°. The wrist D435 is modelled
  as in the lab photo (2026-11-07): 90 × 25 × 25 mm housing on a 12 mm bracket on the gripper
  housing top near the flange, long side along the finger axis, looking straight forward; it is a
  collision body and appears in the side camera's images. Looking forward, the wrist image shows
  mostly the cup rim and background and the bowl only during the pour; tilting it down does not
  show the cup interior (the camera is only ~36 mm above the cup mouth). Tilt is one config value.
- **Demo states are saved by default** (`m4_generate.py --no-states` only for timing runs): the
  exporter renders the policy images from them and refuses episodes without them.
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
python scripts/view.py [--camera eye_in_hand]                  # interactive viewer, scripted pours (needs a display)
python scripts/m0_inspect_model.py
python scripts/solve_home.py                                   # prints home_q for config
python scripts/m1_scene_check.py                               # reports/m1_scene.md numbers
python scripts/m2_step_response.py                             # reports/m2_step_response.png
python scripts/m3_fk_check.py && python tests/test_ik_xarm.py && python tests/test_panda_ik_unchanged.py
OMP_NUM_THREADS=1 python scripts/m4_generate.py --out out/m4 --reps 10 --workers 10   # 10 = physical cores
python scripts/m4_report.py --run out/m4
python scripts/m5_export.py --run out/m4 --out out/xarm_pour_200.hdf5 --n 200 --filter success --workers 16
DIFFUSION_POLICY_ROOT=../diffusion_policy python tests/test_dp_dataset.py out/xarm_pour_200.hdf5
python scripts/m6_diagnostics.py --reps 10
python scripts/m7_dr_grid.py
python tests/test_camera.py && python tests/test_faults.py
```

Large-scale data with continuous bowl positions: `m4_generate.py --set task.placement=uniform --n N`;
domain randomisation: `--set randomization.enabled=true`. Generation resumes from `summary.csv`.

## Training and evaluation (diffusion_policy)

`third_party/DIFFUSION_POLICY_COMMIT` (`20537a5`) is not in the public diffusion_policy repository,
so sim_xarm uses upstream `5ba07ac` with its own dataset class (`dp/xarm_pour_dataset.py`, 7-D joint
action range-normalised; no pytorch3d) and env runner (`dp/xarm_pour_runner.py`). Same conda env as
the simulator.

```bash
conda activate sim_xarm && cd ~/FAVOR/sim_xarm
dp/install_dp.sh                                   # once: clones ~/diffusion_policy @5ba07ac, pinned deps, task yaml
# data: 500 scripted demos, continuous bowl positions (training seeds 0..499)
OMP_NUM_THREADS=1 python scripts/m4_generate.py --out out/m4_train500 --n 500 --workers 10 --set task.placement=uniform
python scripts/m5_export.py --run out/m4_train500 --out out/xarm_pour_500.hdf5 --workers 8
DIFFUSION_POLICY_ROOT=~/diffusion_policy python tests/test_dp_dataset.py out/xarm_pour_500.hdf5
# GPU smoke run (minutes), then the real run
dp/train.sh out/xarm_pour_500.hdf5 smoke training.num_epochs=2 training.max_train_steps=50 task.env_runner.n_test=5
dp/train.sh out/xarm_pour_500.hdf5 pour500
# evaluation on the 25-cell grid (seeds 10000+), DDIM 10
DIFFUSION_POLICY_ROOT=~/diffusion_policy python scripts/m9_eval.py --ckpt out/train/pour500/checkpoints/<best>.ckpt --reps 2
DIFFUSION_POLICY_ROOT=~/diffusion_policy python scripts/m9_eval.py --ckpt ... --faults "locked:1,locked:6" --methods "B1,W-IK pos,W-IK pose,Priority IK,Priority IK (orient. first)"
```

- Policy and training = upstream `train_diffusion_unet_hybrid_workspace` defaults, as in the LIBERO
  runs (CNN U-Net 512/1024/2048, ε-prediction, DDPM 100 training steps, horizon 16, n_obs 2,
  n_action 8, crop 76, batch 64, EMA), 100 epochs. Checkpoints: top 5 by `test_mean_score` = success
  of 25 no-fault rollouts (one per grid cell, seeds 100000+) every 10 epochs, as LIBERO selected by
  success (`favor_fault_runner`, 50 rollouts).
- Inference in the runner and in `m9_eval.py`: DDIM 10 steps with the trained noise schedule (the
  real-robot plan: train DDPM 100, run DDIM 10). `--sampler ddpm` for the 100-step sampler.
- Rollouts run in parallel env processes; the policy noise is seeded per (episode seed, policy call),
  so methods and fault conditions are paired by seed. Repeat runs are bit-identical on CPU (GPU cuDNN
  kernels may add tiny differences).
- A trial ends at 30 s, or earlier once the cup has been tilted > 60°, is back < 10° and the arm has
  been still for 1 s (`--no-early-stop` to always run 30 s). Success: f ≥ 0.8 and no safety violation.
  `summary.md` reports the sim counterpart of the pre-registered gate (success ≥ 0.75 and mean f ≥ 0.85).
- Seeds: demos < 10000, evaluation 10000+, in-training rollouts 100000+.

