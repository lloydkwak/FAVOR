"""M1(c): physics real-time factor of one scripted pour (grid centre) for particle sizes / solver
iterations, with the resulting transfer fraction. Writes reports/m1_rtf_sweep.json."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from favor_xarm.config import add_cli, from_args, resolve
from favor_xarm.env import PourEnv
from favor_xarm.scripted import ScriptedPourPlanner

ap = add_cli(argparse.ArgumentParser())
ap.add_argument("--diam", type=float, required=True)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--density", type=float, default=1.05, help="g/cm^3 -> particle count for 50 g")
args = ap.parse_args()
cfg = from_args(args)
r = args.diam / 2 / 10.0
n = int(round(cfg["particles"]["total_mass_g"] / (args.density * 4 / 3 * np.pi * r ** 3)))
cfg["particles"]["diameter_mm"] = args.diam
cfg["particles"]["count"] = n
cfg["physics"]["iterations"] = args.iters
env = PourEnv(cfg, render=False)
t0 = time.time(); env.reset(seed=0, grid_index=12); t_reset = time.time() - t0
Q, _ = ScriptedPourPlanner(cfg, env.h).plan(env.bowl_xy, np.random.default_rng(0))
t0 = time.time()
for q in Q:
    env.step(q)
t_pour = time.time() - t0
t0 = time.time(); env.hold(cfg["task"]["settle_after"]); t_hold = time.time() - t0
m = env.metrics()
sim = len(Q) / cfg["control"]["policy_hz"]
res = dict(diam_mm=args.diam, count=n, iters=args.iters, f=m["f"], spill_g=m["spill_g"], force_g=m["bowl_force_g"],
           rtf_pour=sim / t_pour, reset_s=t_reset, hold_s=t_hold, wall_total_s=t_reset + t_pour + t_hold)
print(json.dumps(res))
path = resolve("reports/m1_rtf_sweep.jsonl")
open(path, "a").write(json.dumps(res) + "\n")
