# M1: scene checks

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

## Acceptance

| check | result |
|---|---|
| (a) upright at home for 10 s | **0 / 420** particles ever outside the cup; settled fill height 29.9 mm of 50 mm; max particle speed at the end 2.2 mm/s |
| (b) scripted pour at the grid centre | f = **0.929** (46.4 g in the bowl), spill 3.6 g, cup residue 0 g, max tilt 123.7°, no safety flag; scale cross-check from bowl+scale contact forces 48.9 g (includes beads that landed on the scale plate) |
| (c) real-time factor | physics: 0.27 during the pour, 0.09 while the full cup sits still (dense packing → most contacts); rendering both cameras: 0.86 s/step with CPU OSMesa (GPU EGL will be milliseconds) |
| (d) 84 × 84 images | `reports/m1/policy_images_84_grid.png` (rows: home, mid-pour; columns: side, wrist) and full-resolution frames in `reports/m1/` |

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

(The sweep rows were run before velocity feedforward was enabled in M2; the 6 mm / 20 row gives
both runs.) With 10 CG iterations the beads pass through the cup walls during the transfer
(f ≈ 0.1–0.2), so **20 iterations is the minimum**; 7 mm gives identical f at 20 and 30.

Judgement: with the 6 mm default, one core makes ~50–60 demonstrations per hour (physics only,
GPU rendering), so 2,000 demonstrations need ~35–40 core-hours, i.e. ~2–3 h on a 16-core server
with one process per core (`m4_generate.py --workers N`). That is workable. 8 mm beads are 3.5×
faster; the bead size should be fixed by what the real beads will be (§3 question 8). Avoid the
CPU OSMesa renderer for data generation (0.5–0.9 s per step for the two cameras on this machine).
