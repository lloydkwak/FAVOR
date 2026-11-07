# M1: scene checks

## Current layout: side grip, no handle, bowl on the table, roll pour (2026-11-07)

`scripts/m1_scene_check.py` on the current config (raw: `reports/m1/m1_results.json`, images in `reports/m1/`):

| check | result |
|---|---|
| (a) upright at home for 10 s | **0 / 420** beads outside the cup; settled fill height 29.9 mm of 50 mm |
| (b) scripted pour at the grid centre | f = **1.000**, spill 0.0 g, contact-force cross-check 50.0 g, max tilt 123.7°, no safety flag, no robot-bowl contact |
| (c) real-time factor | physics 0.35 (pour) / 0.19 (still) on the 2-core validation machine with a second job running |
| 25 cells × 2 scripted demos | **50 / 50** success, mean f 1.000 (min 0.993), max spill 0.36 g, 12.6–14.4 s (`reports/m4_demos_grip.md`) |
| 25 cells × 4 plans, kinematic check | IK error ≤ 0.01 mm, J2/J4/J6 margin ≥ 0.51 rad, 0 table / self / bowl contacts |

Everything below is the earlier hanging-cup layout (kept for the record).


Produced by `scripts/m1_scene_check.py` (raw numbers: `reports/m1/m1_results.json`) and
`scripts/m1_rtf_sweep.py` (`reports/m1_rtf_sweep.jsonl`). Machine: 2 CPU cores, no GPU, OSMesa
software rendering. Default scene: 420 beads of 6 mm (0.119 g each, 50.0 g), cup 62 mm × 50 mm,
bowl 120 mm × 50 mm, MuJoCo CG solver 20 iterations, timestep 2 ms.

## Cup mount and home pose (decided while building the scene)

The first layout (gripper pointing forward, cup axis perpendicular to the approach axis) could not
reach the near grid rows with a horizontal approach (J6 lower limit −97° binds; only straight-arm,
J4 ≈ 0 solutions existed). The default is now `cup.mount: hang`: the gripper points down and pinches
a vertical handle plate on the cup side; the cup hangs beside the fingers. The `side` layout is kept
as an option. A yaw sweep of the pouring direction over the grid corners (125° tilt) found −120°
(lip towards the robot's back-right) feasible everywhere; home is solved for that yaw:
`home_q = [0.2517, −0.1347, −0.1080, 1.1221, −0.0152, 1.2560, −0.8979]` (cup mouth at
(0.40, 0, 0.30) m, upright, 0.04 mm / 0.004° after rounding).

## Acceptance (final scene: collision shells, upright clearance, feedforward, no MSAA)

| check | result |
|---|---|
| (a) upright at home for 10 s | **0 / 420** particles ever outside the cup; settled fill height 30.0 mm of 50 mm; max particle speed at the end 1.0 mm/s |
| (b) scripted pour at the grid centre | f = **1.000** (50.0 g in the bowl), spill 0.0 g, cup residue 0 g, max tilt 123.7°, no safety flag, no robot-bowl contact; scale cross-check from bowl+scale contact forces **50.0 g** |
| (c) real-time factor | physics: 0.24 during the pour, 0.14 while the full cup sits still (dense packing → most contacts); rendering both cameras: 0.34 s/step with CPU OSMesa (GPU EGL: milliseconds) |
| (d) 84 × 84 images | `reports/m1/policy_images_84_grid.png` (rows: home, mid-pour; columns: side, wrist) and full-resolution frames in `reports/m1/` |

First version of the scene (before the M4 fixes): (b) gave f = 0.929 / 0.955 with 2–4 g spill; the
beads that were lost had tunnelled through the bowl bottom or been shaken out by cup-rim contact
(see README, "Scene decisions").

## (c) Is it fast enough for thousands of demonstrations?

One scripted demonstration is ~14 s of simulated time + 1 s settling at reset + 1 s weighing.

| beads | count (50 g ABS) | CG iterations | f (one pour) | spill (g) | physics RTF during pour | wall per pour, physics only (s) |
|---|---|---|---|---|---|---|
| 6 mm | 421 | 10 | 0.112 | 44.4 | 1.26 | 13 |
| 6 mm | 420 | 20 (default) | 0.955 / 0.929 | 2.3 / 3.6 | 0.27–0.29 | ~60 |
| 6 mm | 421 | 30 | 0.974 | 1.3 | 0.22 | 75 |
| 7 mm | 265 | 10 | 0.166 | 41.7 | 1.91 | 9 |
| 7 mm | 265 | 20 | 0.974 | 1.3 | 0.48 | 35 |
| 7 mm | 265 | 30 | 0.974 | 1.3 | 0.36 | 45 |
| 8 mm | 178 | 10 | 0.208 | 39.6 | 2.69 | 6 |
| 8 mm | 178 | 20 | 0.955 | 2.2 | 0.95 | 17 |

(The sweep rows were run before velocity feedforward (M2) and the collision shells / upright
clearance (M4); they measure speed and the iteration threshold, not the final f. The 6 mm / 20 row
gives both runs. With the final scene the same pour gives f = 1.000 and the 250 M4 demonstrations
mean f 1.000; their measured wall time on this machine was 55–75 s per episode per core.) With 10 CG iterations the beads pass through the cup walls during the transfer
(f ≈ 0.1–0.2), so **20 iterations is the minimum**; 7 mm gives identical f at 20 and 30.

Judgement: with the 6 mm default, one core makes ~50–60 demonstrations per hour (physics only,
GPU rendering), so 2,000 demonstrations need ~35–40 core-hours, i.e. ~2–3 h on a 16-core server
with one process per core (`m4_generate.py --workers N`). That is workable. 8 mm beads are 3.5×
faster; the bead size should be fixed by what the real beads will be (§3 question 8). Avoid the
CPU OSMesa renderer for data generation (0.5–0.9 s per step for the two cameras on this machine).
