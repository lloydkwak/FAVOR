"""Fault layer checks on the scripted pour (30 beads for speed):
  locked J6 from reset : joint stays at its home angle (max deviation < 0.1 deg, the real-robot
                         check is < 0.1 deg of encoder drift), healthy joints still move
  range J7 +-0.1 rad   : joint stays inside the window
  locked J7, mid-task  : no fault before the FK tilt first exceeds 20 deg; locked 0.5 s later at
                         the angle it had then
Run: python sim_xarm/tests/test_faults.py"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from favor_xarm.config import load_config
from favor_xarm.env import PourEnv
from favor_xarm.faults import FaultLayer
from favor_xarm.scripted import ScriptedPourPlanner

cfg = load_config(overrides=["particles.count=30"])
env = PourEnv(cfg, render=False)
pl = ScriptedPourPlanner(cfg, env.h)
ok = True


def run(fault):
    env.fault = fault
    env.reset(seed=3, grid_index=12)
    Q, _ = pl.plan(env.bowl_xy, np.random.default_rng(10))
    qs, tilts = [], []
    for q in Q:
        env.step(q)
        qs.append(env.joint_pos()); tilts.append(env.cup_tilt())
    m = env.metrics()
    fault.restore(env)
    env.fault = None
    return np.array(qs), np.array(tilts), m, Q


home = np.array(cfg["robot"]["home_q"])
qs, _, m, Q = run(FaultLayer(5, "locked"))
dev = math.degrees(np.abs(qs[:, 5] - home[5]).max())
moved = np.abs(qs - home).max(0)
p = dev < 0.1 and moved[[0, 1, 3, 6]].max() > 0.1
ok &= p
print(f"locked J6: max |q6 - q_lock| {dev:.4f} deg, healthy joints max motion {np.round(moved, 2).tolist()} rad, "
      f"f={m['f']:.2f} -> {'PASS' if p else 'FAIL'}")

qs, _, m, Q = run(FaultLayer(6, "range", half_width=0.1))
lo, hi = home[6] - 0.1, home[6] + 0.1
p = qs[:, 6].min() >= lo - 1e-3 and qs[:, 6].max() <= hi + 1e-3 and np.abs(Q[:, 6] - home[6]).max() > 0.1
ok &= p
print(f"range J7 +-0.1: q7 in [{qs[:, 6].min() - home[6]:+.4f}, {qs[:, 6].max() - home[6]:+.4f}] rad around home "
      f"(commanded excursion {np.abs(Q[:, 6] - home[6]).max():.2f} rad), f={m['f']:.2f} -> {'PASS' if p else 'FAIL'}")

f = FaultLayer(6, "locked", onset="tilt20+0.5")
qs, tilts, m, Q = run(f)
k_cross = int(np.argmax(tilts > math.radians(20)))
t_on = f.onset_time
k_on = int(round(t_on * cfg["control"]["policy_hz"]))
after = qs[k_on + 1:, 6]
p = t_on is not None and t_on >= (k_cross - 1) / 10 + 0.5 - 0.11 and np.abs(after - f.q_lock).max() < math.radians(0.1) \
    and np.abs(qs[:k_on, 6] - Q[:k_on, 6]).max() < 0.05
ok &= p
print(f"mid-task J7: tilt > 20 deg at step {k_cross} (t~{k_cross / 10:.1f} s), locked at t={t_on:.2f} s, "
      f"q_lock-home={f.q_lock - home[6]:+.3f} rad, max drift after lock {math.degrees(np.abs(after - f.q_lock).max()):.4f} deg, "
      f"f={m['f']:.2f} -> {'PASS' if p else 'FAIL'}")
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
