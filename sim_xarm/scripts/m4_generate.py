"""M4: scripted pouring demonstrations, in parallel worker processes.

Each episode: reset (bowl at a grid cell or uniform sample, seed) -> scripted plan -> 10 Hz
commanded joint targets executed through the command layer -> weigh after settle_after.
Per episode an .npz is written with, per policy step t (observation BEFORE action t):
  actions (T,7) commanded joint targets, robot0_eef_pos (T,3), robot0_eef_quat (T,4, xyzw),
  robot0_joint_pos (T,7), optionally states (T, nq) float32 for offline rendering (--save-states)
  and images (--images), plus the plan parameters and the metrics.
A summary.csv row per episode is appended; scripts/m4_report.py builds the per-cell table.

  python scripts/m4_generate.py --out out/m4 --reps 10 --workers 8 --save-states
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from favor_xarm.config import add_cli, from_args, resolve

FIELDS = ["episode", "grid_index", "rep", "seed", "bowl_x", "bowl_y", "T", "duration_s", "f", "spill_g", "bowl_g",
          "cup_g", "bowl_force_g", "success", "safety_violation", "safety_first", "max_tilt_deg", "ik_pos_err_max_mm",
          "dq_max", "clamped_ticks", "wall_s"]


def run_episode(job):
    cfg, ep, gi, rep, seed, bowl_xy, out_dir, save_states, images = job
    from favor_xarm.env import PourEnv
    from favor_xarm.scripted import ScriptedPourPlanner
    global _ENV, _PLAN
    if "_ENV" not in globals():
        _ENV = PourEnv(cfg, render=images)
        _PLAN = ScriptedPourPlanner(cfg, _ENV.h)
    env, pl = _ENV, _PLAN
    t0 = time.time()
    obs = env.reset(seed=seed, grid_index=gi, bowl_xy=bowl_xy)
    Q, info = pl.plan(env.bowl_xy, np.random.default_rng(seed + 7))
    rec = {k: [] for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos")}
    imgs = {"agentview_image": [], "robot0_eye_in_hand_image": []}
    states = []
    for q in Q:
        for k in rec:
            rec[k].append(obs[k])
        if images:
            for k in imgs:
                imgs[k].append(obs[k])
        if save_states:
            states.append(env.d.qpos.astype(np.float32).copy())
        obs = env.step(q)
    env.hold(cfg["task"]["settle_after"])
    m = env.metrics()
    data = {k: np.asarray(v) for k, v in rec.items()}
    data["actions"] = Q.astype(np.float32)
    if save_states:
        data["states"] = np.asarray(states)
    if images:
        for k, v in imgs.items():
            data[k] = np.asarray(v, dtype=np.uint8)
    meta = dict(episode=ep, grid_index=-1 if gi is None else gi, rep=rep, seed=seed,
                bowl_xy=[float(v) for v in env.bowl_xy], n_particles=int(len(env.h.particle_qpos)),
                params=info["params"], metrics=m, phase=info["phase"].tolist())
    np.savez_compressed(os.path.join(out_dir, f"ep_{ep:05d}.npz"), meta=json.dumps(meta, default=float), **data)
    row = dict(episode=ep, grid_index=meta["grid_index"], rep=rep, seed=seed, bowl_x=env.bowl_xy[0], bowl_y=env.bowl_xy[1],
               T=len(Q), duration_s=len(Q) / cfg["control"]["policy_hz"], f=m["f"], spill_g=m["spill_g"], bowl_g=m["bowl_g"],
               cup_g=m["cup_g"], bowl_force_g=m["bowl_force_g"], success=int(m["success"]),
               safety_violation=int(m["safety_violation"]), safety_first=str(m["safety_first"]),
               max_tilt_deg=m["max_tilt_deg"], ik_pos_err_max_mm=1000 * info["ik_pos_err_max"], dq_max=info["dq_max"],
               clamped_ticks=env.ctl.n_clamped_ticks, wall_s=time.time() - t0)
    return row


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=10, help="episodes per grid cell (grid mode)")
    ap.add_argument("--n", type=int, default=None, help="total episodes (uniform mode)")
    ap.add_argument("--cells", type=str, default=None, help="comma-separated grid indices (default all)")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--save-states", action="store_true")
    ap.add_argument("--images", action="store_true", help="render the policy images online")
    args = ap.parse_args()
    cfg = from_args(args)
    out = resolve(args.out)
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config_used.json"), "w"), indent=1)
    jobs = []
    if cfg["task"]["placement"] == "grid" and args.n is None:
        n_cells = cfg["task"]["grid_n"] ** 2
        cells = [int(c) for c in args.cells.split(",")] if args.cells else range(n_cells)
        ep = 0
        for rep in range(args.reps):
            for gi in cells:
                seed = args.base_seed + 1000 * gi + rep
                jobs.append((cfg, ep, gi, rep, seed, None, out, args.save_states, args.images))
                ep += 1
    else:
        rng = np.random.default_rng(args.base_seed)
        s, c = cfg["task"]["grid_size"], np.array(cfg["task"]["grid_center"])
        for ep in range(args.n):
            xy = c + rng.uniform(-s / 2, s / 2, size=2)
            jobs.append((cfg, ep, None, ep, args.base_seed + ep, xy, out, args.save_states, args.images))
    done = set()
    summ = os.path.join(out, "summary.csv")
    if os.path.exists(summ):
        done = {int(r["episode"]) for r in csv.DictReader(open(summ))}
    jobs = [j for j in jobs if j[1] not in done]
    print(f"{len(jobs)} episodes to run ({len(done)} already done) with {args.workers} workers", flush=True)
    new = not os.path.exists(summ)
    f = open(summ, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new:
        w.writeheader()
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for i, row in enumerate(pool.imap_unordered(run_episode, jobs)):
                w.writerow(row); f.flush()
                print(f"[{i + 1}/{len(jobs)}] ep {row['episode']} cell {row['grid_index']} f={row['f']:.3f} "
                      f"succ={row['success']} wall={row['wall_s']:.0f}s elapsed={time.time() - t0:.0f}s", flush=True)
    else:
        for i, j in enumerate(jobs):
            row = run_episode(j)
            w.writerow(row); f.flush()
            print(f"[{i + 1}/{len(jobs)}] ep {row['episode']} cell {row['grid_index']} f={row['f']:.3f} "
                  f"succ={row['success']} wall={row['wall_s']:.0f}s", flush=True)
    f.close()


if __name__ == "__main__":
    main()
