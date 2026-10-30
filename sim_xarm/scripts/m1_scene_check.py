"""M1 acceptance checks.
 (a) cup upright at home for 10 s: no particle leaves the cup.
 (b) scripted pour over the bowl (grid centre): particles flow out and settle in the bowl.
 (c) real-time factor (physics only, and physics + rendering both cameras).
 (d) 84x84 policy images of both cameras (home and mid-pour) + full-resolution frames.
Writes reports/m1_scene.md and reports/m1/*.png."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import mujoco
import numpy as np

from favor_xarm.config import add_cli, from_args, resolve
from favor_xarm.env import PourEnv
from favor_xarm.scripted import ScriptedPourPlanner


def save(path, img):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--hold", type=float, default=10.0)
    ap.add_argument("--grid-index", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = from_args(args)
    out = resolve("reports/m1")
    os.makedirs(out, exist_ok=True)
    env = PourEnv(cfg, render=True)
    R = {"particles": len(env.h.particle_qpos), "particle_mass_g": env.h.particle_mass * 1000,
         "timestep": env.m.opt.timestep, "solver": mujoco.mjtSolver(env.m.opt.solver).name,
         "iterations": int(env.m.opt.iterations), "nv": int(env.m.nv)}

    # (a) upright hold.
    t0 = time.time()
    env.reset(seed=args.seed, grid_index=args.grid_index)
    R["reset_wall_s"] = time.time() - t0
    R["fill_height_mm_spawn"] = env.fill_height * 1000
    zs = []
    for k in ("agentview", "eye_in_hand"):
        save(os.path.join(out, f"home_{k}_full.png"), env.render(k))
        save(os.path.join(out, f"home_{k}_84.png"), env.policy_image(k))
    steps = int(round(args.hold * cfg["control"]["policy_hz"]))
    out_of_cup = []
    t0 = time.time()
    for _ in range(steps):
        env.step(env.home_q)
        out_of_cup.append(int((~env.in_cup_mask()).sum()))
    wall = time.time() - t0
    P = env.particle_pos()
    bid = env.h.bodies["cup"]
    rel = (P - env.d.xpos[bid]) @ env.d.xmat[bid].reshape(3, 3)
    R["a_hold_s"] = args.hold
    R["a_max_particles_outside_cup"] = int(max(out_of_cup))
    R["a_fill_height_mm_settled"] = float((rel[:, 2].max() + cfg["particles"]["diameter_mm"] / 2000) * 1000)
    R["a_cup_depth_mm"] = cfg["cup"]["inner_depth"] * 1000
    R["a_max_particle_speed_end"] = float(env.particle_speed().max())
    R["c_rtf_physics_hold"] = args.hold / wall

    # (b) scripted pour at the grid centre (physics only), then weigh.
    pl = ScriptedPourPlanner(cfg, env.h)
    env.render_enabled = False
    env.reset(seed=args.seed, grid_index=args.grid_index)
    Q, info = pl.plan(env.bowl_xy, np.random.default_rng(args.seed))
    t0 = time.time()
    mid = None
    tilt_trace = []
    for i, q in enumerate(Q):
        env.step(q)
        tilt_trace.append(np.degrees(env.cup_tilt()))
        if mid is None and tilt_trace[-1] > 100:
            mid = i
            for k in ("agentview", "eye_in_hand"):
                save(os.path.join(out, f"pour_{k}_full.png"), env.render(k))
                save(os.path.join(out, f"pour_{k}_84.png"), env.policy_image(k))
    wall_pour = time.time() - t0
    t_sim = len(Q) / cfg["control"]["policy_hz"]
    env.hold(cfg["task"]["settle_after"])
    met = env.metrics()
    for k in ("agentview", "eye_in_hand"):
        save(os.path.join(out, f"end_{k}_full.png"), env.render(k))
    R["b_steps"] = len(Q)
    R["b_metrics"] = met
    R["c_rtf_physics_pour"] = t_sim / wall_pour

    # (c) rendering cost: both cameras at render_size then resize, per policy step.
    t0 = time.time()
    n = 20
    for _ in range(n):
        env.policy_image("agentview"); env.policy_image("eye_in_hand")
    R["c_render_ms_per_step"] = (time.time() - t0) / n * 1000
    phys_ms = 1000 / cfg["control"]["policy_hz"] / R["c_rtf_physics_pour"]
    R["c_rtf_with_render"] = (1000 / cfg["control"]["policy_hz"]) / (phys_ms + R["c_render_ms_per_step"])
    demo_wall = (t_sim + cfg["particles"]["settle_time"] + cfg["task"]["settle_after"]) / R["c_rtf_physics_pour"] + \
        R["c_render_ms_per_step"] * len(Q) / 1000
    R["c_wall_per_demo_s"] = demo_wall
    R["c_demos_per_core_hour"] = 3600 / demo_wall

    # (d) image grid.
    tiles = [cv2.imread(os.path.join(out, f"{ph}_{k}_84.png")) for ph in ("home", "pour") for k in ("agentview", "eye_in_hand")]
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(os.path.join(out, "policy_images_84_grid.png"), cv2.resize(grid, (grid.shape[1] * 3, grid.shape[0] * 3),
                                                                            interpolation=cv2.INTER_NEAREST))
    json.dump(R, open(os.path.join(out, "m1_results.json"), "w"), indent=1, default=float)
    print(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main()
