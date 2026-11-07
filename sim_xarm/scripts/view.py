"""Open the pouring scene in the interactive MuJoCo viewer and play scripted pours.

  python scripts/view.py                         # grid centre, scripted pour, repeats over cells
  python scripts/view.py --cells 0,24 --speed 0.5
  python scripts/view.py --idle                  # robot holds home, beads settle; inspect the scene
  python scripts/view.py --fault-joint 6         # locked J6 (no correction) to see the effect
  python scripts/view.py --camera eye_in_hand    # look through the wrist camera (or agentview)

Viewer: drag to rotate / right-drag to pan / scroll to zoom; '[' and ']' cycle the scene cameras
(agentview = side camera, eye_in_hand = wrist camera), Esc returns to the free camera.
The cup and bowl collision shells (group 3) are hidden; press '3' in the viewer to show them.
Needs a display. MUJOCO_GL is forced to glfw here (egl is only for offscreen rendering).
"""
import argparse
import os
import sys
import time

os.environ["MUJOCO_GL"] = "glfw"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mujoco
import mujoco.viewer
import numpy as np

from favor_xarm.config import add_cli, from_args
from favor_xarm.env import PourEnv
from favor_xarm.faults import FaultLayer
from favor_xarm.scripted import ScriptedPourPlanner


class ViewerSync:
    """Uses the env's per-tick hook (the same one the fault layer uses) to refresh the viewer every
    10 ms of simulated time and to pace the simulation to --speed x real time (when it can)."""

    def __init__(self, viewer, speed, inner=None):
        self.v, self.speed, self.inner = viewer, speed, inner
        self.t0_wall = time.time()
        self.t_sim = 0.0

    def on_reset(self, env):
        if self.inner is not None:
            self.inner.on_reset(env)
        self.t0_wall, self.t_sim = time.time(), 0.0
        self.v.sync()

    def filter_command(self, env, q):
        return q if self.inner is None else self.inner.filter_command(env, q)

    def after_tick(self, env):
        if self.inner is not None:
            self.inner.after_tick(env)
        self.t_sim += 1.0 / env.cfg["control"]["stream_hz"]
        if not self.v.is_running():
            sys.stdout.flush()
            os._exit(0)
        self.v.sync()
        ahead = self.t0_wall + self.t_sim / self.speed - time.time()
        if ahead > 0:
            time.sleep(ahead)

    def get_fault_info(self):
        return None if self.inner is None else self.inner.get_fault_info()


def main():
    ap = add_cli(argparse.ArgumentParser())
    ap.add_argument("--cells", default="12", help="comma-separated grid cells, played in turn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="target speed vs real time")
    ap.add_argument("--idle", action="store_true", help="no motion, just hold home")
    ap.add_argument("--once", action="store_true", help="play each cell once, then keep the window open")
    ap.add_argument("--fault-joint", type=int, default=None, help="1..7, locked at home (no correction)")
    ap.add_argument("--camera", choices=["free", "agentview", "eye_in_hand"], default="free",
                    help="start looking through a scene camera ('[' / ']' switch, Esc = free camera)")
    ap.add_argument("--headless-steps", type=int, default=None, help=argparse.SUPPRESS)  # self-test without a display
    args = ap.parse_args()
    cfg = from_args(args)
    env = PourEnv(cfg, render=False)
    planner = ScriptedPourPlanner(cfg, env.h)
    fault = FaultLayer(args.fault_joint - 1, "locked") if args.fault_joint else None
    cells = [int(c) for c in args.cells.split(",")]

    if args.headless_steps is not None:          # used by the developer self-test only
        class _V:
            def sync(self): pass
            def is_running(self): return True
        env.fault = ViewerSync(_V(), speed=1e9, inner=fault)
        env.reset(seed=args.seed, grid_index=cells[0])
        Q, _ = planner.plan(env.bowl_xy, np.random.default_rng(args.seed + 7))
        for q in Q[:args.headless_steps]:
            env.step(q)
        print("headless ok", env.metrics()["f"], len(Q))
        return

    with mujoco.viewer.launch_passive(env.m, env.d) as v:
        v.cam.lookat[:] = [cfg["task"]["grid_center"][0] - 0.05, cfg["task"]["grid_center"][1], 0.15]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.3, -150.0, -25.0
        if args.camera != "free":
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = env.h.cameras[args.camera]
        env.fault = ViewerSync(v, args.speed, inner=fault)
        k = 0
        while v.is_running():
            gi = cells[k % len(cells)]
            seed = args.seed + k
            print(f"reset: cell {gi}, seed {seed} (beads settle for {cfg['particles']['settle_time']} s first)", flush=True)
            env.reset(seed=seed, grid_index=gi)
            if args.idle:
                while v.is_running():
                    env.step(env.home_q)
                return
            Q, info = planner.plan(env.bowl_xy, np.random.default_rng(seed + 7))
            for q in Q:
                env.step(q)
            env.hold(cfg["task"]["settle_after"])
            m = env.metrics()
            print(f"  f = {m['f']:.3f}, spill {m['spill_g']:.2f} g, cup {m['cup_g']:.2f} g, "
                  f"max tilt {m['max_tilt_deg']:.0f} deg, safety violation {m['safety_violation']} {m['safety_first']}",
                  flush=True)
            k += 1
            if args.once and k >= len(cells):
                print("done; close the window to exit", flush=True)
                while v.is_running():
                    v.sync()
                    time.sleep(0.05)
                return


if __name__ == "__main__":
    try:
        main()
    finally:
        # Skip interpreter teardown: closing the GLFW viewer thread during teardown can segfault.
        sys.stdout.flush()
        os._exit(0)
