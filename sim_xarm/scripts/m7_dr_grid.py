"""M7: sample image grid with domain randomisation on (one flag: randomization.enabled=true).
Columns: nominal (randomisation off) + --n randomised resets. Rows: side camera, wrist camera, at
the policy resolution (84 x 84, shown enlarged) after the reset settling. Writes
reports/m7_dr_grid.png and reports/m7_dr_samples.json."""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from favor_xarm.config import add_cli, from_args, resolve
from favor_xarm.env import PourEnv


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--settle", type=float, default=0.3)
    args = ap.parse_args()
    cfg = from_args(args)
    cfg["particles"]["settle_time"] = args.settle
    cols, samples = [], []
    off = copy.deepcopy(cfg); off["randomization"]["enabled"] = False
    on = copy.deepcopy(cfg); on["randomization"]["enabled"] = True
    env0 = PourEnv(off, render=True)
    env0.reset(seed=0, grid_index=12)
    cols.append([env0.policy_image("agentview"), env0.policy_image("eye_in_hand")])
    env0.close()
    env = PourEnv(on, render=True)
    for k in range(args.n):
        env.reset(seed=100 + k, grid_index=int(np.random.default_rng(k).integers(25)))
        cols.append([env.policy_image("agentview"), env.policy_image("eye_in_hand")])
        samples.append(env.dr_sample)
    grid = np.vstack([np.hstack([c[r] for c in cols]) for r in range(2)])
    grid = cv2.resize(grid, (grid.shape[1] * 2, grid.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(resolve("reports/m7_dr_grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    json.dump(samples, open(resolve("reports/m7_dr_samples.json"), "w"), indent=1, default=float)
    print(json.dumps(samples, indent=1, default=float))


if __name__ == "__main__":
    main()
